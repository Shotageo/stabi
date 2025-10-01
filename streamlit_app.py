# streamlit_app.py — Lightweight fan view with up to 3 layers & crossing control (batch-run)
from __future__ import annotations
import streamlit as st
import numpy as np, heapq
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    arcs_from_center_by_entries_multi, clip_interfaces_to_ground
)

st.set_page_config(page_title="Stabi LEM（軽量・3層対応・一括実行）", layout="wide")
st.title("Stabi LEM（センター固定／首振り・最大3層・境界進入制御）")

# ---------------- UI: フォーム一括 ----------------
with st.form("params"):
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

        preset = st.selectbox("Ground preset", ["3-seg berm (default)"])
        if preset == "3-seg berm (default)":
            ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces: list[GroundPL] = []
        if n_layers >= 2:
            iface1 = make_interface1_example(H, L)
            interfaces.append(iface1)
        if n_layers >= 3:
            iface2 = make_interface2_example(H, L)
            interfaces.append(iface2)

        st.subheader("Soil parameters (top→bottom)")
        gamma1 = st.number_input("γ₁ (kN/m³)", 10.0, 25.0, 18.0, 0.5)
        c1     = st.number_input("c₁ (kPa)",   0.0, 200.0, 5.0, 0.5)
        phi1   = st.number_input("φ₁ (deg)",   0.0, 45.0, 30.0, 0.5)
        soil1  = Soil(gamma=gamma1, c=c1, phi=phi1)

        if n_layers >= 2:
            gamma2 = st.number_input("γ₂ (kN/m³)", 10.0, 25.0, 19.0, 0.5)
            c2     = st.number_input("c₂ (kPa)",   0.0, 200.0, 8.0, 0.5)
            phi2   = st.number_input("φ₂ (deg)",   0.0, 45.0, 28.0, 0.5)
            soil2  = Soil(gamma=gamma2, c=c2, phi=phi2)
        if n_layers >= 3:
            gamma3 = st.number_input("γ₃ (kN/m³)", 10.0, 25.0, 20.0, 0.5)
            c3     = st.number_input("c₃ (kPa)",   0.0, 200.0, 12.0, 0.5)
            phi3   = st.number_input("φ₃ (deg)",   0.0, 45.0, 25.0, 0.5)
            soil3  = Soil(gamma=gamma3, c=c3, phi=phi3)

        soils = [soil1] + ([soil2] if n_layers>=2 else []) + ([soil3] if n_layers>=3 else [])

        st.subheader("Crossing control (下層へ進入可否)")
        allow_cross = []
        if n_layers >= 2:
            # I1の下（＝Layer2）へ入れる？
            allow_cross_L2 = st.checkbox("Allow crossing into Layer 2 (below Interface 1)", True)
            allow_cross.append(allow_cross_L2)
        if n_layers >= 3:
            # I2の下（＝Layer3）へ入れる？（安定層ならOFFに）
            allow_cross_L3 = st.checkbox("Allow crossing into Layer 3 (below Interface 2)", False)
            allow_cross.append(allow_cross_L3)

    with colB:
        st.subheader("Center grid (初期枠)")
        x_min = st.number_input("Center x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("Center y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Fan parameters（軽量）")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        n_entries = st.slider("Entry samples on ground", 100, 2000, 600, 50)
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)
        show_k    = st.slider("Show top-K arcs (thin)", 10, 400, 120, 10)
        top_thick = st.slider("Emphasize top-N (thick)", 1, 30, 12, 1)
        show_radii = st.checkbox("Show radii to both ends", True)

        # モード保持（フォーム内）
        if "pick_mode" not in st.session_state:
            st.session_state["pick_mode"] = "Max arcs (robust)"
        mode = st.radio(
            "Center picking",
            ["Max arcs (robust)", "Min Fs (aggressive)"],
            horizontal=True,
            key="pick_mode"
        )

    run = st.form_submit_button("▶ 計算開始（バッチ実行）")

if not run:
    st.info("パラメータを調整して **[▶ 計算開始]** を押してね。実行まで再計算しません。")
    st.stop()

# --------------- ユーティリティ：センター走査＆自動拡張（軽量） ---------------
def count_arcs_and_minFs(center, *, limit_k: int | None = None):
    xc, yc = center
    cnt = 0
    Fs_min = None
    top_heap: list[tuple[float, tuple[float,float,float]]] = []
    for x1, x2, R, Fs in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=n_entries, method=method,
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross
    ):
        cnt += 1
        if (Fs_min is None) or (Fs < Fs_min):
            Fs_min = Fs
        if limit_k is not None:
            heapq.heappush(top_heap, (-Fs, (x1, x2, R)))
            if len(top_heap) > limit_k:
                heapq.heappop(top_heap)
    if limit_k is not None:
        top_list = sorted([(-fsneg, dat) for fsneg, dat in top_heap], key=lambda t: t[0])
    else:
        top_list = []
    return cnt, (Fs_min if Fs_min is not None else float("inf")), top_list

def near_boundary(xc, yc, x_min, x_max, y_min, y_max, nx, ny):
    dx = (x_max - x_min) / max(nx-1, 1)
    dy = (y_max - y_min) / max(ny-1, 1)
    epsx, epsy = 0.51*dx, 0.51*dy
    return (abs(xc - x_min) <= epsx or abs(xc - x_max) <= epsx or
            abs(yc - y_min) <= epsy or abs(yc - y_max) <= epsy)

def scan_best_center_with_auto_expand(x_min, x_max, y_min, y_max, nx, ny,
                                      mode="Max arcs (robust)", expand_steps=3):
    used_bounds = []
    for step in range(expand_steps+1):
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        centers = [(float(xc), float(yc)) for yc in ys for xc in xs]
        best_center = None

        if mode == "Max arcs (robust)":
            best_score = -1; best_top = []
            for c in centers:
                cnt, _, top = count_arcs_and_minFs(c, limit_k=show_k)
                if cnt > best_score:
                    best_score, best_center, best_top = cnt, c, top
        else:
            best_val = float("inf"); best_cnt = -1; best_top=[]
            for c in centers:
                cnt, Fs_min, top = count_arcs_and_minFs(c, limit_k=show_k)
                if Fs_min < best_val or (Fs_min == best_val and cnt > best_cnt):
                    best_val, best_cnt, best_center, best_top = Fs_min, cnt, c, top

        if best_center and near_boundary(best_center[0], best_center[1], x_min, x_max, y_min, y_max, nx, ny) and step < expand_steps:
            used_bounds.append((x_min, x_max, y_min, y_max))
            x_min = max(0.0, x_min - 0.20*L); x_max = x_max + 0.20*L
            y_min = max(0.0, y_min - 0.20*H); y_max = y_max + 0.20*H
            continue

        used_bounds.append((x_min, x_max, y_min, y_max))
        _, Fs_min, _ = count_arcs_and_minFs(best_center, limit_k=None)
        return best_center, best_top, used_bounds, Fs_min
    return None, [], used_bounds, float("inf")

# ----------------------- 中心決定（自動拡張付き） -----------------------
mode = st.session_state.get("pick_mode", "Max arcs (robust)")
with st.spinner("センター走査中（軽量・自動拡張あり） ..."):
    best_center, top_list, used_bounds, minFs_val = scan_best_center_with_auto_expand(
        x_min, x_max, y_min, y_max, nx, ny, mode=mode, expand_steps=3
    )
if best_center is None or len(top_list) == 0:
    st.error("有効な円弧が見つかりませんでした。パラメータや交差可否を見直して再実行してください。")
    st.stop()

xc, yc = best_center
x_min_u, x_max_u, y_min_u, y_max_u = used_bounds[-1]

# ----------------------- 描画（層クリップ & 扇） -----------------------
fig, ax = plt.subplots(figsize=(10, 7))

# 層の塗り（地表 ≥ I1 ≥ I2 ≥ 0 を保証しつつ）
Xdense = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = ground.y_at(Xdense)
if n_layers == 1:
    ax.fill_between(Xdense, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers == 2:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xdense)[0]
    ax.fill_between(Xdense, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xdense, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1, Y2 = clip_interfaces_to_ground(ground, [interfaces[0], interfaces[1]], Xdense)
    ax.fill_between(Xdense, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xdense, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xdense, 0.0, Y2, alpha=0.12, label="Layer3")

# 地表・層境界描画（クリップ後）
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers >= 2:
    Y1_line = clip_interfaces_to_ground(ground, [interfaces[0]], Xdense)[0]
    ax.plot(Xdense, Y1_line, linestyle="--", linewidth=1.4, label="Interface 1 (clipped)")
if n_layers >= 3:
    Y1_line, Y2_line = clip_interfaces_to_ground(ground, [interfaces[0], interfaces[1]], Xdense)
    ax.plot(Xdense, Y2_line, linestyle="--", linewidth=1.4, label="Interface 2 (clipped)")

# 外周枠
ax.plot([ground.X[-1], ground.X[-1]], [0.0, ground.y_at(ground.X[-1])], linewidth=1.2)
ax.plot([ground.X[0],  ground.X[-1]], [0.0, 0.0],                         linewidth=1.2)
ax.plot([ground.X[0],  ground.X[0]],  [0.0, ground.y_at(ground.X[0])],    linewidth=1.2)

# グリッド（自動拡張後）
ax.plot([x_min_u, x_max_u, x_max_u, x_min_u, x_min_u],
        [y_min_u, y_min_u, y_max_u, y_max_u, y_min_u],
        linestyle="--", alpha=0.65, label="Center-grid (used)")
XX, YY = np.meshgrid(np.linspace(x_min_u, x_max_u, nx),
                     np.linspace(y_min_u, y_max_u, ny))
ax.scatter(XX.ravel(), YY.ravel(), s=12, alpha=0.5)
ax.scatter([xc], [yc], s=65, marker="s", label="Chosen center")

# 扇：thin=top-K / thick=上位N
thin_all = top_list
thick_sel = top_list[:min(top_thick, len(top_list))]
for Fs, (x1, x2, R) in thin_all:
    xs_line = np.linspace(x1, x2, 200)
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=0.6, alpha=0.30)
    if show_radii:
        y1 = float(ground.y_at(x1)); y2 = float(ground.y_at(x2))
        ax.plot([xc, x1], [yc, y1], linewidth=0.35, alpha=0.25)
        ax.plot([xc, x2], [yc, y2], linewidth=0.35, alpha=0.25)
for Fs, (x1, x2, R) in thick_sel:
    xs_line = np.linspace(x1, x2, 400)
    ys_line = yc - np.sqrt(np.maximum(0.0, R**2 - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=2.6)

# 軸：100m以上確保＋見切れ防止
x_upper = max(1.18*L, x_max_u + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max_u + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title(
    f"[{mode}] Center=({xc:.2f},{yc:.2f}) • Method={method} • "
    f"Shown={len(thin_all)} arcs (thin), top {len(thick_sel)} thick"
)

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# メトリクス
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("Min Fs（このセンター）", f"{minFs_val:.3f}")
with mcol2:
    st.write(f"Center-grid（実使用）: x=[{x_min_u:.2f}, {x_max_u:.2f}], y=[{y_min_u:.2f}, {y_max_u:.2f}]")
with mcol3:
    if n_layers >= 2:
        txt = f"Crossing: L2={'OK' if allow_cross[0] else 'NG'}"
        if n_layers >= 3:
            txt += f", L3={'OK' if allow_cross[1] else 'NG'}"
        st.write(txt)
    else:
        st.write("Single layer")
