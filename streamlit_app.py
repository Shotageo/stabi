# streamlit_app.py — 後からの表示切替・center-grid可視化・全センター監査表示（Quick）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM｜表示切替＋センター監査", layout="wide")
st.title("Stabi LEM｜表示切替＋センター監査（Quick）")

# ========= Quality プリセット =========
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=800,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  top_thick=10,
                   coarse_subsample="every 3rd", coarse_entries=150,
                   coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.5, budget_quick_s=0.7,
                   audit_limit_per_center=15, audit_budget_s=1.0),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1200, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, top_thick=12,
                   coarse_subsample="every 2nd", coarse_entries=200,
                   coarse_limit_arcs=60, coarse_probe_min=81,
                   budget_coarse_s=0.6, budget_quick_s=0.8,
                   audit_limit_per_center=20, audit_budget_s=1.4),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1600, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, top_thick=16,
                 coarse_subsample="full", coarse_entries=300,
                 coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.0, budget_quick_s=1.2,
                 audit_limit_per_center=25, audit_budget_s=1.8),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, top_thick=20,
                      coarse_subsample="full", coarse_entries=400,
                      coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.6, budget_quick_s=2.4,
                      audit_limit_per_center=30, audit_budget_s=2.4),
}

# ========= ユーティリティ =========
def fs_to_color(fs: float):
    if fs < 1.0:
        return (0.85, 0.0, 0.0)                 # 赤
    elif fs < 1.2:
        t = (fs - 1.0) / 0.2                    # 0→オレンジ, 1→黄
        r = 1.0; g = 0.50 + 0.50*t; b = 0.0
        return (r, g, b)
    else:
        return (0.0, 0.55, 0.0)                 # 緑

def all_centers_grid(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# ========= 入力フォーム =========
with st.form("params"):
    cA, cB = st.columns(2)
    with cA:
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
        gamma1 = st.number_input("γ₁ (kN/m³)", 10.0, 25.0, 18.0, 0.5); c1 = st.number_input("c₁ (kPa)", 0.0, 200.0, 5.0, 0.5); phi1 = st.number_input("φ₁ (deg)", 0.0, 45.0, 30.0, 0.5)
        soil1 = Soil(gamma=gamma1, c=c1, phi=phi1)
        soils = [soil1]
        if n_layers >= 2:
            gamma2 = st.number_input("γ₂ (kN/m³)", 10.0, 25.0, 19.0, 0.5); c2 = st.number_input("c₂ (kPa)", 0.0, 200.0, 8.0, 0.5); phi2 = st.number_input("φ₂ (deg)", 0.0, 45.0, 28.0, 0.5)
            soils.append(Soil(gamma=gamma2, c=c2, phi=phi2))
        if n_layers >= 3:
            gamma3 = st.number_input("γ₃ (kN/m³)", 10.0, 25.0, 20.0, 0.5); c3 = st.number_input("c₃ (kPa)", 0.0, 200.0, 12.0, 0.5); phi3 = st.number_input("φ₃ (deg)", 0.0, 45.0, 25.0, 0.5)
            soils.append(Soil(gamma=gamma3, c=c3, phi=phi3))

        st.subheader("Crossing control（下層進入可否）")
        allow_cross = []
        if n_layers >= 2:
            allow_cross.append(st.checkbox("Allow crossing into Layer 2 (below Interface 1)", True))
        if n_layers >= 3:
            allow_cross.append(st.checkbox("Allow crossing into Layer 3 (below Interface 2)", True))

        st.subheader("Target safety")
        Fs_target = st.number_input("Target FS (for required T)", 1.00, 2.00, 1.20, 0.05)

    with cB:
        st.subheader("Center grid（探索枠・全点表示）")
        x_min = st.number_input("Center x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("Center y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        quality = st.select_slider("Quality（精度×速度）", options=list(QUALITY.keys()), value="Normal")
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

    run = st.form_submit_button("▶ 計算開始")

# ========= Qualityの実値展開 =========
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

# ========= 計算パラメタのキー（セッションに保持） =========
param_key = hash_params(dict(
    H=H, L=L, n_layers=n_layers,
    soils=[(s.gamma, s.c, s.phi) for s in soils],
    allow_cross=allow_cross, Fs_target=Fs_target,
    center=[x_min, x_max, y_min, y_max, nx, ny],
    method=method, quality=P, depth=[depth_min, depth_max],
))

# ========= 計算 or 前回結果の再利用 =========
def compute_once():
    # 1) Coarse：最有望センター選抜（サブサンプル＋時間打切り）
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

    def pick_center_coarse(budget_s):
        deadline = time.time() + budget_s
        best = None
        tested = []
        for c in _coarse_centers(x_min, x_max, y_min, y_max, nx, ny, P["coarse_subsample"]):
            xc, yc = c
            cnt = 0; Fs_min = None
            for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
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
            tested.append((xc,yc))
            if Fs_min is None: Fs_min = float("inf")
            score = (cnt, -Fs_min)
            if (best is None) or (score > best[0]):
                best = (score, (xc,yc))
            if time.time() > deadline: break
        return best[1] if best else None, tested

    with st.spinner("Coarse pass（最有望センター選抜）..."):
        center, tested_centers = pick_center_coarse(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarse段でセンターが見つかりません。枠や深さを広げてください。")
    xc, yc = center

    # 2) Quick：選抜センターでR候補（時間打切り）
    with st.spinner("Quick pass（R候補抽出）..."):
        heap_R = []
        q_deadline = time.time() + P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
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
        R_candidates = [r for _fsneg, r in sorted([(-fsneg, R) for fsneg, R in heap_R], key=lambda t:t[0])]
        if len(R_candidates)==0:
            return dict(error="Quick段で有効な円弧候補がありません。条件を緩めてください。")

    # 3) Refine：精密Fs＋必要抑止力
    refined = []
    for R in R_candidates[:P["show_k"]]:
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
        if Fs is None: 
            continue
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
        if s is None:
            continue
        x1, x2, xs, ys, h = s
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
        if packD is None:
            continue
        D_sum, _, _ = packD
        T_req = max(0.0, (Fs_target - Fs) * D_sum)
        refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="精密段で有効な円弧が得られませんでした。設定やQualityを見直してください。")
    refined.sort(key=lambda d: d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    # 監査用：探索全センター（表示用に保存するだけ）
    centers_all = all_centers_grid(x_min, x_max, y_min, y_max, nx, ny)
    return dict(
        center=(xc,yc), centers_all=centers_all, tested_centers=tested_centers,
        refined=refined, idx_minFs=idx_minFs, idx_maxT=idx_maxT,
        params_key=param_key
    )

# 初回 or [計算開始] 時に更新
if run or ("last_result" not in st.session_state) or (st.session_state.get("last_key") != param_key):
    res = compute_once()
    if "error" in res:
        st.error(res["error"])
        st.stop()
    st.session_state["last_result"] = res
    st.session_state["last_key"] = param_key

# 再表示（トグル変更時はここだけ走る）
res = st.session_state["last_result"]
xc, yc = res["center"]
centers_all = res["centers_all"]
refined = res["refined"]
idx_minFs = res["idx_minFs"]
idx_maxT  = res["idx_maxT"]
minFs_val = refined[idx_minFs]["Fs"]
maxT_val  = refined[idx_maxT]["T_req"]

# ========= 表示オプション（計算後に切替可能） =========
st.subheader("表示オプション（計算後に自由に切替）")
colv1, colv2, colv3, colv4 = st.columns([1,1,1,2])
with colv1:
    show_centers = st.checkbox("Show center-grid points", True)
with colv2:
    show_all_refined = st.checkbox("Show all refined arcs (color by Fs)", True)
with colv3:
    show_minFs = st.checkbox("Show Min Fs only", True)
    show_maxT  = st.checkbox("Show Max required T only", True)
with colv4:
    audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)
    audit_limit = st.slider("Audit: max arcs / center", 5, 60, QUALITY[quality]["audit_limit_per_center"], 5, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 0.5, 5.0, QUALITY[quality]["audit_budget_s"], 0.1, disabled=not audit_show)

# ========= （必要時）監査用の全センターQuick円弧 =========
def compute_audit_arcs(centers_all, audit_limit, audit_budget):
    arcs = []  # list of dict(x1,x2,R,Fs,xc,yc)
    deadline = time.time() + audit_budget
    for (cx,cy) in centers_all:
        count = 0
        for a_x1,a_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, soils, cx, cy,
            n_entries=min(P["n_entries_final"], 1200), method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=max(10, P["quick_slices"]),
            limit_arcs_per_center=audit_limit,
            probe_n_min=max(81, P["probe_n_min_quick"]),
        ):
            arcs.append(dict(x1=a_x1,x2=a_x2,R=R,Fs=Fs,xc=cx,yc=cy))
            count += 1
            if count >= audit_limit: break
        if time.time() > deadline: break
    return arcs

audit_arcs = []
if audit_show:
    # 監査はパラメタ／制限も含めてキー化してキャッシュ（再描画を高速に）
    audit_key = hash_params(dict(K=param_key, limit=audit_limit, budget=round(audit_budget,2)))
    cache = st.session_state.get("audit_cache", {})
    if (cache.get("key") == audit_key) and ("arcs" in cache):
        audit_arcs = cache["arcs"]
    else:
        with st.spinner("Audit（全センターQuickの可視化）..."):
            audit_arcs = compute_audit_arcs(centers_all, audit_limit, audit_budget)
        st.session_state["audit_cache"] = {"key": audit_key, "arcs": audit_arcs}

# ========= 描画 =========
fig, ax = plt.subplots(figsize=(10.5, 7.5))

# 地表＆層の塗り
Xd = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = ground.y_at(Xd)
if n_layers == 1:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers == 2:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1, Y2 = clip_interfaces_to_ground(ground, [interfaces[0], interfaces[1]], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
# 地表線・外周
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers >= 2:
    Y1_line = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.plot(Xd, Y1_line, linestyle="--", linewidth=1.2, label="Interface 1 (clipped)")
if n_layers >= 3:
    Y1_line, Y2_line = clip_interfaces_to_ground(ground, [interfaces[0], interfaces[1]], Xd)
    ax.plot(Xd, Y2_line, linestyle="--", linewidth=1.2, label="Interface 2 (clipped)")
ax.plot([ground.X[-1], ground.X[-1]], [0.0, ground.y_at(ground.X[-1])], linewidth=1.0)
ax.plot([ground.X[0],  ground.X[-1]], [0.0, 0.0],                         linewidth=1.0)
ax.plot([ground.X[0],  ground.X[0]],  [0.0, ground.y_at(ground.X[0])],    linewidth=1.0)

# center-grid（全点）
if show_centers:
    xs = [c[0] for c in centers_all]; ys = [c[1] for c in centers_all]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
# 選抜センター
ax.scatter([xc], [yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# 監査：全センターQuick円弧（薄線）
if audit_show and audit_arcs:
    for a in audit_arcs:
        cx, cy, R, x1, x2, Fs = a["xc"], a["yc"], a["R"], a["x1"], a["x2"], a["Fs"]
        xs = np.linspace(x1, x2, 160)
        ys = cy - np.sqrt(np.maximum(0.0, R*R - (xs - cx)**2))
        ax.plot(xs, ys, linewidth=0.6, alpha=0.25, color=fs_to_color(Fs))
    # 監査凡例
    from matplotlib.lines import Line2D
    ax.add_line(Line2D([], [], color='k', alpha=0.25, lw=0.6))
    ax.plot([], [], linewidth=0.6, alpha=0.25, color=(0.85,0,0), label="Audit (Fs-colored)")

# refined：全体（薄線）
if show_all_refined:
    for d in refined:
        xs = np.linspace(d["x1"], d["x2"], 220)
        ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, linewidth=0.9, alpha=0.7, color=fs_to_color(d["Fs"]))

# ピックアップ：MinFs／MaxT（太線・トグル）
if show_minFs and (0 <= idx_minFs < len(refined)):
    d = refined[idx_minFs]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    # radii
    y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
    ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
    ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if show_maxT and (0 <= idx_maxT < len(refined)):
    d = refined[idx_maxT]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
            label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
    # radii
    y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
    ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
    ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

# 軸・凡例・スコープ
x_upper = max(1.18*L, x_max + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches = [
    Patch(color=(0.85,0.0,0.0), label="Fs < 1.0"),
    Patch(color=(1.0,0.75,0.0), label="1.0 ≤ Fs < 1.2"),
    Patch(color=(0.0,0.55,0.0), label="Fs ≥ 1.2"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + legend_patches, labels + [p.get_label() for p in legend_patches],
          loc="upper right", fontsize=9)

title_tail = []
if show_all_refined: title_tail.append(f"K={len(refined)} refined")
if audit_show:       title_tail.append(f"audit arcs={len(audit_arcs)} (budget {audit_budget:.1f}s)")
title_tail.append(f"MinFs={minFs_val:.3f}")
title_tail.append(f"TargetFs={Fs_target:.2f}")
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))

st.pyplot(fig, use_container_width=True); plt.close(fig)

# メトリクス
c1,c2,c3 = st.columns(3)
with c1: st.metric("Min Fs（精密）", f"{minFs_val:.3f}")
with c2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
with c3: st.caption("表示はトグルで切替可能／監査はQuick薄線で全センター可視化")
