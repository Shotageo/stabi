# streamlit_app.py  — Lightweight fan view (batch-run, widened grid, fixed axes & fill)
from __future__ import annotations
import streamlit as st
import numpy as np, math, heapq
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL, make_ground_example,
    arcs_from_center_by_entries,
)

st.set_page_config(page_title="Stabi LEM（軽量描画・一括実行）", layout="wide")
st.title("Stabi LEM（軽量描画レイヤ・センター固定／首振り・一括実行）")

# ---------------- UI: フォームにまとめ、[計算開始]まで実行しない ----------------
with st.form("params"):
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Geometry & Soil")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

        preset = st.selectbox("Ground preset", ["3-seg berm (default)"])
        if preset == "3-seg berm (default)":
            pl = make_ground_example(H, L)

        gamma = st.number_input("γ (kN/m³)", 10.0, 25.0, 18.0, 0.5)
        cohesion = st.number_input("c (kPa)", 0.0, 200.0, 5.0, 0.5)
        phi = st.number_input("φ (deg)", 0.0, 45.0, 30.0, 0.5)
        soil = Soil(gamma=gamma, c=cohesion, phi=phi)

    with colB:
        st.subheader("Center grid (初期枠)")
        # 初期は右上に設定。後で自動拡張が入る
        x_min = st.number_input("Center x min", 0.20*L, 2.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 3.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 4.00*H, 1.60*H, 0.10*H,
                                help="地表から十分離したければ 1.5H〜2.0H など")
        y_max = st.number_input("Center y max", 1.00*H, 5.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Fan parameters（軽量）")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        n_entries = st.slider("Entry samples on ground", 100, 2000, 600, 50)
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)
        show_k    = st.slider("Show top-K arcs (thin)", 10, 400, 120, 10)
        top_thick = st.slider("Emphasize top-N (thick)", 1, 30, 12, 1)
        show_radii = st.checkbox("Show radii to both ends", True, help="円弧両端への半径を表示")

    run = st.form_submit_button("▶ 計算開始（バッチ実行）")

# ---------------- 計算：ボタンが押されるまで実行しない ----------------
if not run:
    st.info("スライダーを調整して **[▶ 計算開始]** を押すと計算します。")
    st.stop()

# --------------- ユーティリティ：センター走査＆自動拡張 ---------------
def count_arcs_and_minFs(center, *, limit_k: int | None = None):
    """描画せずに『個数』と『最小Fs』と『top-K（薄線描画用）』だけ拾う（軽量）"""
    xc, yc = center
    cnt = 0
    Fs_min = None
    top_heap: list[tuple[float, tuple[float,float,float]]] = []  # (-Fs, (x1,x2,R))
    for x1, x2, R, Fs in arcs_from_center_by_entries(
        pl, soil, xc, yc,
        n_entries=n_entries, method=method,
        depth_min=depth_min, depth_max=depth_max
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
    """選ばれたセンターがグリッドの外周近傍にいるか？（→拡張トリガ）"""
    dx = (x_max - x_min) / max(nx-1, 1)
    dy = (y_max - y_min) / max(ny-1, 1)
    epsx, epsy = 0.51*dx, 0.51*dy
    return (abs(xc - x_min) <= epsx or abs(xc - x_max) <= epsx or
            abs(yc - y_min) <= epsy or abs(yc - y_max) <= epsy)

def scan_best_center_with_auto_expand(x_min, x_max, y_min, y_max, nx, ny,
                                      mode="Max arcs (robust)", expand_steps=3):
    """外周に当たったら枠を広げて再走査（最大 expand_steps 回）"""
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
        else:  # "Min Fs (aggressive)"
            best_val = float("inf"); best_cnt = -1; best_top=[]
            for c in centers:
                cnt, Fs_min, top = count_arcs_and_minFs(c, limit_k=show_k)
                if Fs_min < best_val or (Fs_min == best_val and cnt > best_cnt):
                    best_val, best_cnt, best_center, best_top = Fs_min, cnt, c, top

        # 外周なら枠拡張
        if best_center and near_boundary(best_center[0], best_center[1], x_min, x_max, y_min, y_max, nx, ny) and step < expand_steps:
            used_bounds.append((x_min, x_max, y_min, y_max))
            # L/H 比例で拡張（左右上下ともに）
            x_min = max(0.0, x_min - 0.20*L)
            x_max = x_max + 0.20*L
            y_min = max(0.0, y_min - 0.20*H)
            y_max = y_max + 0.20*H
            # 次ループで再走査
            continue

        # 最終決定
        used_bounds.append((x_min, x_max, y_min, y_max))
        # 最小Fsは top から拾えない可能性があるので、ここで再評価して確実に出す
        _, Fs_min, _ = count_arcs_and_minFs(best_center, limit_k=None)
        return best_center, best_top, used_bounds, Fs_min

    # ここには原則来ない
    return None, [], used_bounds, float("inf")

# ----------------------- 中心決定（自動拡張付き） -----------------------
mode = st.radio("Center picking", ["Max arcs (robust)", "Min Fs (aggressive)"], horizontal=True)
with st.spinner("センター走査中（軽量・自動拡張あり） ..."):
    best_center, top_list, used_bounds, minFs_val = scan_best_center_with_auto_expand(
        x_min, x_max, y_min, y_max, nx, ny, mode=mode, expand_steps=3
    )

if best_center is None or len(top_list) == 0:
    st.error("有効な円弧が見つかりませんでした。パラメータを広げて再実行してください。")
    st.stop()

xc, yc = best_center
last_bounds = used_bounds[-1]
x_min_u, x_max_u, y_min_u, y_max_u = last_bounds

# ----------------------- 描画（軽量：top-Kのみ保持） -----------------------
fig, ax = plt.subplots(figsize=(10, 7))

# （修正1/4）地層塗り：地表ポリラインの“下”だけを塗る（はみ出し防止）
xpoly = np.concatenate(([pl.X[0]], pl.X, [pl.X[-1]], [pl.X[0]]))
ypoly = np.concatenate(([0.0],     pl.Y, [0.0],      [0.0]))
ax.fill(xpoly, ypoly, alpha=0.10, label="Section")   # 透かしは地表を超えない

# 地表線
ax.plot(pl.X, pl.Y, linewidth=2.2, label="Ground")

# （修正5）外周枠：地表—右側面—底面—左側面—地表 で囲う
ax.plot([pl.X[-1], pl.X[-1]], [0.0, pl.Y[-1]], linewidth=1.4)
ax.plot([pl.X[0],  pl.X[-1]], [0.0, 0.0],     linewidth=1.4)
ax.plot([pl.X[0],  pl.X[0]],  [0.0, pl.Y[0]], linewidth=1.4)

# グリッド矩形（自動拡張後の最終枠）＆点
ax.plot([x_min_u, x_max_u, x_max_u, x_min_u, x_min_u],
        [y_min_u, y_min_u, y_max_u, y_max_u, y_min_u],
        linestyle="--", alpha=0.65, label="Center-grid (used)")
XX, YY = np.meshgrid(np.linspace(x_min_u, x_max_u, nx),
                     np.linspace(y_min_u, y_max_u, ny))
ax.scatter(XX.ravel(), YY.ravel(), s=12, alpha=0.5)
ax.scatter([xc], [yc], s=65, marker="s", label="Chosen center")

# 薄線：top-K、太線：上位N
thin_all = top_list
thick_sel = top_list[:min(top_thick, len(top_list))]

for Fs, (x1, x2, R) in thin_all:
    xs_line = np.linspace(x1, x2, 200)  # 表示は間引き
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=0.6, alpha=0.30)
    if st.session_state.get("show_radii_cache", show_radii):
        y1 = float(pl.y_at(x1)); y2 = float(pl.y_at(x2))
        ax.plot([xc, x1], [yc, y1], linewidth=0.35, alpha=0.25)
        ax.plot([xc, x2], [yc, y2], linewidth=0.35, alpha=0.25)

for Fs, (x1, x2, R) in thick_sel:
    xs_line = np.linspace(x1, x2, 400)
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=2.6)

# （修正2）座標軸は100以上まで確保／（修正3）右端で見切れ防止
x_upper = max(1.18*L, x_max_u + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max_u + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# 凡例・タイトル
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • Shown={len(thin_all)} arcs (thin), top {len(thick_sel)} thick")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# （修正1）最小Fsの明示表示
mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("Min Fs（このセンター）", f"{minFs_val:.3f}")
with mcol2:
    st.write(f"Center-grid（実使用）: x=[{x_min_u:.2f}, {x_max_u:.2f}], y=[{y_min_u:.2f}, {y_max_u:.2f}]")
with mcol3:
    st.caption("※グリッド端で最小が出た場合は、自動で枠を拡張して再探索済み")
