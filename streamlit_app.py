# streamlit_app.py — Turbo v2 (FIX): Coarse→Quick（二段選定）＋タイムバジェット＋堅牢キャッシュ
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    arcs_from_center_by_entries_multi, clip_interfaces_to_ground, fs_given_R_multi,
    # 表示用: 端点取得
    arc_sample_poly_best_pair,
)

st.set_page_config(page_title="Stabi LEM（Turbo v2）", layout="wide")
st.title("Stabi LEM｜Turbo v2：Coarse→Quick（二段選定）＋タイムバジェット")

# =============== UI 一括フォーム ===============
with st.form("params"):
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

        ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces: list[GroundPL] = []
        if n_layers >= 2:
            interfaces.append(make_interface1_example(H, L))
        if n_layers >= 3:
            interfaces.append(make_interface2_example(H, L))

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
            allow_cross.append(st.checkbox("Allow crossing into Layer 2 (below Interface 1)", True))
        if n_layers >= 3:
            allow_cross.append(st.checkbox("Allow crossing into Layer 3 (below Interface 2)", False))

    with colB:
        st.subheader("Center grid (初期枠)")
        x_min = st.number_input("Center x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("Center y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Fan parameters")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)

        st.subheader("Turbo v2 settings")
        n_entries_final = st.slider("Final n_entries (ground samples)", 200, 3000, 1200, 100)
        n_slices_quick  = st.slider("Quick slices", 8, 40, 12, 1)
        n_slices_final  = st.slider("Final slices", 20, 80, 40, 5)
        limit_arcs_quick = st.slider("Quick: max arcs per center", 40, 400, 120, 10)
        probe_n_min_quick = st.slider("Quick: min probe points per arc", 61, 221, 101, 10,
                                      help="円弧サンプル点の最小値。小さいほど速いが粗い")
        show_k    = st.slider("Plot top-K arcs (refined)", 10, 400, 120, 10)
        top_thick = st.slider("Emphasize top-N thick", 1, 30, 12, 1)
        show_radii = st.checkbox("Show radii to both ends", True)

        # センター選定のCoarse設定とタイムバジェット
        coarse_subsample = st.selectbox("Coarse subsample", ["every 2nd", "every 3rd", "full"], index=0,
                                        help="センター探索の疎化（Coarse）")
        coarse_entries   = st.slider("Coarse n_entries", 50, 600, 200, 50)
        coarse_limit_arcs= st.slider("Coarse: max arcs per center", 20, 200, 60, 10)
        coarse_probe_min = st.slider("Coarse: min probe points per arc", 41, 161, 81, 10)

        st.subheader("Time budget")
        budget_coarse_s  = st.slider("Budget: Coarse (sec)", 0.1, 5.0, 0.6, 0.1)
        budget_quick_s   = st.slider("Budget: Quick (sec)", 0.1, 5.0, 0.8, 0.1)

        if "pick_mode" not in st.session_state:
            st.session_state["pick_mode"] = "Max arcs (robust)"
        mode = st.radio("Center picking", ["Max arcs (robust)", "Min Fs (aggressive)"],
                        horizontal=True, key="pick_mode")

    run = st.form_submit_button("▶ 計算開始（Turbo v2）")

if not run:
    st.info("パラメータを調整して **[▶ 計算開始]** を押してね。実行まで再計算しません。")
    st.stop()

# =============== キャッシュ：辞書（堅牢） ===============
@st.cache_resource(show_spinner=False)
def _quick_R_cache():
    return {}  # dict[key_tuple] = list_of_R

def _hash_key_for_Rcache(ground, interfaces, soils, allow_cross, xc, yc,
                         n_entries_final, n_slices_quick, limit_arcs_quick, probe_n_min_quick,
                         depth_min, depth_max):
    def arr(a): return tuple(np.round(np.asarray(a), 6).tolist())
    def soil_pack(s: Soil): return (round(s.gamma,6), round(s.c,6), round(s.phi,6))
    key = (
        arr(ground.X), arr(ground.Y),
        tuple((arr(i.X), arr(i.Y)) for i in interfaces),
        tuple(soil_pack(s) for s in soils),
        tuple(bool(x) for x in allow_cross),
        round(xc,6), round(yc,6),
        int(n_entries_final), int(n_slices_quick), int(limit_arcs_quick), int(probe_n_min_quick),
        round(depth_min,3), round(depth_max,3),
    )
    return key

# =============== Coarse：疎グリッド×超軽量×時間打切り ===============
def coarse_centers(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    if coarse_subsample == "every 2nd":
        xs = xs[::2] if len(xs)>1 else xs
        ys = ys[::2] if len(ys)>1 else ys
    elif coarse_subsample == "every 3rd":
        xs = xs[::3] if len(xs)>2 else xs
        ys = ys[::3] if len(ys)>2 else ys
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def coarse_score(center, deadline):
    xc, yc = center
    cnt = 0
    Fs_min = None
    for _x1, _x2, _R, Fs in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=coarse_entries, method="Fellenius",
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross,
        quick_mode=True, n_slices_quick=max(8, n_slices_quick//2),
        limit_arcs_per_center=coarse_limit_arcs,
        probe_n_min=coarse_probe_min,
    ):
        cnt += 1
        if (Fs_min is None) or (Fs < Fs_min):
            Fs_min = Fs
        if time.time() > deadline:
            break
    if Fs_min is None: Fs_min = float("inf")
    score = cnt if mode=="Max arcs (robust)" else -Fs_min
    return score, Fs_min, cnt

def pick_center_coarse(x_min, x_max, y_min, y_max, nx, ny, budget_s):
    deadline = time.time() + budget_s
    best = None
    for c in coarse_centers(x_min, x_max, y_min, y_max, nx, ny):
        score, Fs_min, cnt = coarse_score(c, deadline)
        if best is None:
            best = (score, Fs_min, cnt, c)
        else:
            if mode=="Max arcs (robust)":
                if (score > best[0]) or (score==best[0] and (Fs_min < best[1])):
                    best = (score, Fs_min, cnt, c)
            else:
                if (score > best[0]) or (score==best[0] and (cnt > best[2])):
                    best = (score, Fs_min, cnt, c)
        if time.time() > deadline:
            break
    return best[3] if best else None

# =============== Quick：候補Rをキャッシュつきで取得（時間打切り） ===============
def quick_R_candidates_for_center(center):
    xc, yc = center
    key = _hash_key_for_Rcache(
        ground, interfaces, soils, allow_cross, xc, yc,
        n_entries_final, n_slices_quick, limit_arcs_quick, probe_n_min_quick,
        depth_min, depth_max
    )
    cache = _quick_R_cache()
    if key in cache:
        return cache[key]

    heap_R = []  # (-Fs, R)
    q_deadline = time.time() + budget_quick_s
    for _x1, _x2, R, Fs in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=n_entries_final, method="Fellenius",
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross,
        quick_mode=True, n_slices_quick=n_slices_quick,
        limit_arcs_per_center=limit_arcs_quick,
        probe_n_min=probe_n_min_quick,
    ):
        heapq.heappush(heap_R, (-Fs, R))
        if len(heap_R) > max(show_k, top_thick + 20):
            heapq.heappop(heap_R)
        if time.time() > q_deadline:
            break

    R_list = [r for _fsneg, r in sorted([(-fsneg, R) for fsneg, R in heap_R], key=lambda t:t[0])]
    cache[key] = R_list  # set
    return R_list

# =============== 1) Coarse でセンター当て ===============
with st.spinner("Coarse pass（超軽量・時間打切り）..."):
    center_coarse = pick_center_coarse(x_min, x_max, y_min, y_max, nx, ny, budget_coarse_s)
if center_coarse is None:
    st.error("Coarse段でセンターが見つかりません。範囲や制約を見直してください。")
    st.stop()

xc, yc = center_coarse
x_min_u, x_max_u, y_min_u, y_max_u = x_min, x_max, y_min, y_max  # Coarseでは拡張なし（高速優先）

# =============== 2) Quick で候補Rを抽出（キャッシュ/時間打切り） ===============
with st.spinner("Quick pass（候補抽出・時間打切り）..."):
    R_candidates = quick_R_candidates_for_center(center_coarse)
if len(R_candidates) == 0:
    st.error("Quick段で有効な円弧候補がありません。条件を緩めてください。")
    st.stop()

# =============== 3) 精密再評価（Bishop/Fellenius） ===============
refined = []
for R in R_candidates[:show_k]:
    Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R, n_slices=n_slices_final)
    if Fs is None:
        continue
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
    if s is None:
        continue
    x1, x2, xs, ys, h = s
    refined.append((Fs, (x1, x2, R)))
if len(refined)==0:
    st.error("精密段で有効な円弧が得られませんでした。条件やタイムバジェットを調整してください。")
    st.stop()
refined.sort(key=lambda t:t[0])
thin_all = refined
thick_sel = refined[:min(top_thick, len(refined))]
minFs_val = refined[0][0]

# =============== 可視化（軽量） ===============
fig, ax = plt.subplots(figsize=(10, 7))
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

# センター印
ax.scatter([xc], [yc], s=65, marker="s", label="Chosen center")

# 扇
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

# 軸
x_upper = max(1.18*L, x_max + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title(
    f"[{mode}] Center=({xc:.2f},{yc:.2f}) • Method={method} • "
    f"Shown={len(thin_all)} arcs (refined), top {min(top_thick,len(refined))} thick • MinFs={minFs_val:.3f}"
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# メトリクス
c1, c2 = st.columns(2)
with c1: st.metric("Min Fs（精密）", f"{minFs_val:.3f}")
with c2: st.caption("Coarse→Quick の二段でセンターを決定（時間打切り＋キャッシュ）")
