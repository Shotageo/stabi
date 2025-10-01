# streamlit_app.py — Fsグラデーション＆必要抑止力（ターゲットFs）対応
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    arcs_from_center_by_entries_multi, clip_interfaces_to_ground, fs_given_R_multi,
    arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM（Fs色分け＋抑止力）", layout="wide")
st.title("Stabi LEM｜Fsカラーグラデーション ＋ 必要抑止力ピックアップ")

# ===== Quality presets（前回と同じ） =====
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=800,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  top_thick=10,
                   coarse_subsample="every 3rd", coarse_entries=150,
                   coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.5, budget_quick_s=0.7),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1200, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, top_thick=12,
                   coarse_subsample="every 2nd", coarse_entries=200,
                   coarse_limit_arcs=60, coarse_probe_min=81,
                   budget_coarse_s=0.6, budget_quick_s=0.8),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1600, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, top_thick=16,
                 coarse_subsample="full", coarse_entries=300,
                 coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.0, budget_quick_s=1.2),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, top_thick=20,
                      coarse_subsample="full", coarse_entries=400,
                      coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.6, budget_quick_s=2.0),
}

# ================== UI（フォーム） ==================
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

        st.subheader("Crossing control（下層進入可否）")
        allow_cross = []
        if n_layers >= 2:
            allow_cross.append(st.checkbox("Allow crossing into Layer 2 (below Interface 1)", True))
        if n_layers >= 3:
            allow_cross.append(st.checkbox("Allow crossing into Layer 3 (below Interface 2)", False))

        st.subheader("Reinforcement target")
        Fs_target = st.number_input("Target safety factor (for required stabilizing force)", 1.00, 2.00, 1.20, 0.05)

    with colB:
        st.subheader("Center grid（初期枠）")
        x_min = st.number_input("Center x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("Center y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        quality = st.select_slider("Quality (精度×速度)", options=list(QUALITY.keys()), value="Normal")

        with st.expander("Advanced（Qualityを上書き）", expanded=False):
            override = st.checkbox("上級設定で Quality を上書き", value=False)
            cols1 = st.columns(2)
            with cols1[0]:
                quick_slices_in  = st.slider("Quick slices", 6, 40, QUALITY[quality]["quick_slices"], 1, disabled=not override)
                final_slices_in  = st.slider("Final slices", 20, 80, QUALITY[quality]["final_slices"], 2, disabled=not override)
                n_entries_final_in = st.slider("Final n_entries", 200, 4000, QUALITY[quality]["n_entries_final"], 100, disabled=not override)
                show_k_in        = st.slider("Plot top-K arcs (refined)", 10, 600, QUALITY[quality]["show_k"], 10, disabled=not override)
                top_thick_in     = st.slider("Emphasize top-N thick", 1, 50, QUALITY[quality]["top_thick"], 1, disabled=not override)
            with cols1[1]:
                probe_min_q_in   = st.slider("Quick: min probe points / arc", 41, 221, QUALITY[quality]["probe_n_min_quick"], 10, disabled=not override)
                limit_arcs_q_in  = st.slider("Quick: max arcs / center", 20, 400, QUALITY[quality]["limit_arcs_quick"], 10, disabled=not override)
                coarse_subsample_in = st.selectbox("Coarse subsample", ["every 3rd","every 2nd","full"],
                                                   index=["every 3rd","every 2nd","full"].index(QUALITY[quality]["coarse_subsample"]),
                                                   disabled=not override)
                coarse_entries_in = st.slider("Coarse n_entries", 50, 800, QUALITY[quality]["coarse_entries"], 50, disabled=not override)
                coarse_limit_in   = st.slider("Coarse: max arcs / center", 20, 300, QUALITY[quality]["coarse_limit_arcs"], 10, disabled=not override)
                coarse_probe_in   = st.slider("Coarse: min probe points / arc", 41, 181, QUALITY[quality]["coarse_probe_min"], 10, disabled=not override)
            cols2 = st.columns(2)
            with cols2[0]:
                budget_coarse_in = st.slider("Budget: Coarse (sec)", 0.1, 4.0, QUALITY[quality]["budget_coarse_s"], 0.1, disabled=not override)
            with cols2[1]:
                budget_quick_in  = st.slider("Budget: Quick (sec)", 0.1, 4.0, QUALITY[quality]["budget_quick_s"], 0.1, disabled=not override)

        st.subheader("Depth range（鉛直深さ）")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)

        show_radii = st.checkbox("Show radii to both ends", True)
        picks_mode = st.multiselect("Highlight picks", ["Min Fs", "Max required T"], default=["Min Fs","Max required T"])

    run = st.form_submit_button("▶ 計算開始")

if not run:
    st.info("パラメータを調整して **[▶ 計算開始]** を押してね。")
    st.stop()

# ===== Quality → 実値へ =====
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in, show_k=show_k_in, top_thick=top_thick_in,
        coarse_subsample=coarse_subsample_in, coarse_entries=coarse_entries_in,
        coarse_limit_arcs=coarse_limit_in, coarse_probe_min=coarse_probe_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# ===== Quick候補キャッシュ =====
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
        int(n_slices_quick), int(n_entries_final), int(limit_arcs_quick), int(probe_n_min_quick),
        round(depth_min,3), round(depth_max,3),
    )
    return key

def _coarse_centers(x_min, x_max, y_min, y_max, nx, ny, subsample: str):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    if subsample == "every 3rd":
        xs = xs[::3] if len(xs)>2 else xs
        ys = ys[::3] if len(ys)>2 else ys
    elif subsample == "every 2nd":
        xs = xs[::2] if len(xs)>1 else xs
        ys = ys[::2] if len(ys)>1 else ys
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def _coarse_score(center, deadline):
    xc, yc = center
    cnt = 0
    Fs_min = None
    for _x1, _x2, _R, Fs in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=P["coarse_entries"], method="Fellenius",
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross,
        quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
        limit_arcs_per_center=P["coarse_limit_arcs"],
        probe_n_min=P["coarse_probe_min"],
    ):
        cnt += 1
        if (Fs_min is None) or (Fs < Fs_min):
            Fs_min = Fs
        if time.time() > deadline: break
    if Fs_min is None: Fs_min = float("inf")
    return cnt, Fs_min, cnt

def pick_center_coarse(x_min, x_max, y_min, y_max, nx, ny, budget_s):
    deadline = time.time() + budget_s
    best = None
    for c in _coarse_centers(x_min, x_max, y_min, y_max, nx, ny, P["coarse_subsample"]):
        score, Fs_min, cnt = _coarse_score(c, deadline)
        if (best is None) or (score > best[0]) or (score==best[0] and Fs_min < best[1]):
            best = (score, Fs_min, cnt, c)
        if time.time() > deadline: break
    return best[3] if best else None

def quick_R_candidates_for_center(center):
    xc, yc = center
    key = _hash_key_for_Rcache(
        ground, interfaces, soils, allow_cross, xc, yc,
        P["n_entries_final"], P["quick_slices"], P["limit_arcs_quick"], P["probe_n_min_quick"],
        depth_min, depth_max
    )
    cache = _quick_R_cache()
    if key in cache:
        return cache[key]
    heap_R = []
    q_deadline = time.time() + P["budget_quick_s"]
    for _x1, _x2, R, Fs in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=P["n_entries_final"], method="Fellenius",
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross,
        quick_mode=True, n_slices_quick=P["quick_slices"],
        limit_arcs_per_center=P["limit_arcs_quick"],
        probe_n_min=P["probe_n_min_quick"],
    ):
        heapq.heappush(heap_R, (-Fs, R))
        if len(heap_R) > max(P["show_k"], P["top_thick"] + 20):
            heapq.heappop(heap_R)
        if time.time() > q_deadline: break
    R_list = [r for _fsneg, r in sorted([(-fsneg, R) for fsneg, R in heap_R], key=lambda t:t[0])]
    cache[key] = R_list
    return R_list

# ===== 1) Coarse → center =====
with st.spinner("Coarse pass（疎・時間打切り）..."):
    center = pick_center_coarse(x_min, x_max, y_min, y_max, nx, ny, P["budget_coarse_s"])
if center is None:
    st.error("Coarse段でセンターが見つかりません。範囲や制約を見直してください。")
    st.stop()
xc, yc = center

# ===== 2) Quick → R candidates =====
with st.spinner("Quick pass（候補R抽出・時間打切り）..."):
    R_candidates = quick_R_candidates_for_center(center)
if len(R_candidates)==0:
    st.error("Quick段で有効な円弧候補がありません。条件を緩めてください。")
    st.stop()

# ===== 3) Refine（Fs精密）＋ 必要抑止力T_req算出 =====
refined = []
for R in R_candidates[:P["show_k"]]:
    Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
    if Fs is None: 
        continue
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
    if s is None:
        continue
    x1, x2, xs, ys, h = s
    # 駆動項 D = Σ(W sinα) を取得（Felleniusの分母）。T_req = max(0, (Fs_target - Fs)*D)
    D_pack = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
    if D_pack is None:
        continue
    D_sum, x1_d, x2_d = D_pack
    T_req = max(0.0, (Fs_target - Fs) * D_sum)   # 単位 kN/m（2D）
    refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
if not refined:
    st.error("精密段で有効な円弧が得られませんでした。設定やQualityを見直してください。")
    st.stop()
refined.sort(key=lambda d: d["Fs"])  # 昇順

# ===== ピックアップ =====
idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))
minFs_val = refined[idx_minFs]["Fs"]
maxT_val  = refined[idx_maxT]["T_req"]
minFs_R   = refined[idx_minFs]["R"]
maxT_R    = refined[idx_maxT]["R"]

# ===== Fs → 色（<1.0=赤 / 1.0-1.2=オレンジ~黄 / ≥1.2=緑） =====
def fs_to_color(fs: float):
    # RGB (0-1)
    if fs < 1.0:
        return (0.85, 0.0, 0.0)                 # 赤
    elif fs < 1.2:
        t = (fs - 1.0) / 0.2                    # 0→オレンジ, 1→黄
        r = 1.0
        g = 0.50 + 0.50*t
        b = 0.0
        return (r, g, b)
    else:
        return (0.0, 0.55, 0.0)                 # 緑

# ===== 可視化 =====
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

# 地表と外周
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers >= 2:
    Y1_line = clip_interfaces_to_ground(ground, [interfaces[0]], Xdense)[0]
    ax.plot(Xdense, Y1_line, linestyle="--", linewidth=1.4, label="Interface 1 (clipped)")
if n_layers >= 3:
    Y1_line, Y2_line = clip_interfaces_to_ground(ground, [interfaces[0], interfaces[1]], Xdense)
    ax.plot(Xdense, Y2_line, linestyle="--", linewidth=1.4, label="Interface 2 (clipped)")
ax.plot([ground.X[-1], ground.X[-1]], [0.0, ground.y_at(ground.X[-1])], linewidth=1.2)
ax.plot([ground.X[0],  ground.X[-1]], [0.0, 0.0],                         linewidth=1.2)
ax.plot([ground.X[0],  ground.X[0]],  [0.0, ground.y_at(ground.X[0])],    linewidth=1.2)

# センター
ax.scatter([xc], [yc], s=65, marker="s", label="Chosen center")

# 薄線：すべての円弧をFs色で描画
for d in refined:
    x1, x2, R, Fs = d["x1"], d["x2"], d["R"], d["Fs"]
    xs_line = np.linspace(x1, x2, 220)
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=0.8, alpha=0.65, color=fs_to_color(Fs))
    if show_radii:
        y1 = float(ground.y_at(x1)); y2 = float(ground.y_at(x2))
        ax.plot([xc, x1], [yc, y1], linewidth=0.35, alpha=0.25, color=fs_to_color(Fs))
        ax.plot([xc, x2], [yc, y2], linewidth=0.35, alpha=0.25, color=fs_to_color(Fs))

# 強調：上位N本（低Fs側）を少し太く
for d in refined[:min(P["top_thick"], len(refined))]:
    x1, x2, R, Fs = d["x1"], d["x2"], d["R"], d["Fs"]
    xs_line = np.linspace(x1, x2, 420)
    ys_line = yc - np.sqrt(np.maximum(0.0, R**2 - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=2.2, alpha=0.9, color=fs_to_color(Fs))

# ピックアップ：Min Fs / Max required T
if "Min Fs" in picks_mode and 0 <= idx_minFs < len(refined):
    d = refined[idx_minFs]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.4, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    if show_radii:
        y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
        ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.2, color=(0.9,0.0,0.0))
        ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.2, color=(0.9,0.0,0.0))

if "Max required T" in picks_mode and 0 <= idx_maxT < len(refined):
    d = refined[idx_maxT]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.4, linestyle="--", color=(0.3,0.0,0.8),
            label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
    if show_radii:
        y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
        ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.2, linestyle="--", color=(0.3,0.0,0.8))
        ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.2, linestyle="--", color=(0.3,0.0,0.8))

# 軸・凡例
x_upper = max(1.18*L, x_max + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

# カテゴリ凡例（色の意味）
from matplotlib.patches import Patch
legend_patches = [
    Patch(color=(0.85,0.0,0.0), label="Fs < 1.0"),
    Patch(color=(1.0,0.75,0.0), label="1.0 ≤ Fs < 1.2"),
    Patch(color=(0.0,0.55,0.0), label="Fs ≥ 1.2"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + legend_patches, labels + [p.get_label() for p in legend_patches],
          loc="upper right", fontsize=9)

ax.set_title(
    f"Quality={quality} • Center=({xc:.2f},{yc:.2f}) • Method={method} • "
    f"K={len(refined)} (refined) • MinFs={minFs_val:.3f} • MaxT={maxT_val:.1f} kN/m @Fs={refined[idx_maxT]['Fs']:.3f}"
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# メトリクス
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Min Fs（精密）", f"{minFs_val:.3f}", help="表示中の精密評価での最小安全率")
with c2:
    st.metric("Max required T", f"{maxT_val:.1f} kN/m", help=f"Fsを {Fs_target:.2f} にするための最大全体抑止力（2D）")
with c3:
    st.caption("必要抑止力 T_req = max(0, (Fs_target − Fs) × Σ(W sinα))")
