# streamlit_app.py — 確定実行モード（手入力しても計算しない／ボタン押下でだけ確定・計算）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json, random
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM｜確定実行モード", layout="wide")
st.title("Stabi LEM｜確定実行（パラメータは［計算開始］でだけ反映）")

# ---------------- Quality ----------------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=900,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  top_thick=10,
                   coarse_subsample="every 3rd", coarse_entries=160,
                   coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.6, budget_quick_s=0.9,
                   audit_limit_per_center=10, audit_budget_s=2.0),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, top_thick=12,
                   coarse_subsample="every 2nd", coarse_entries=220,
                   coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2,
                   audit_limit_per_center=12, audit_budget_s=2.8),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1700, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, top_thick=16,
                 coarse_subsample="full", coarse_entries=320,
                 coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.2, budget_quick_s=1.8,
                 audit_limit_per_center=16, audit_budget_s=3.2),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, top_thick=20,
                      coarse_subsample="full", coarse_entries=420,
                      coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.8, budget_quick_s=2.6,
                      audit_limit_per_center=20, audit_budget_s=4.0),
}

# ---------------- Utils ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def grid_points(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def near_edge(xc, yc, x_min, x_max, y_min, y_max, tol=1e-9):
    at_left  = abs(xc - x_min) < tol
    at_right = abs(xc - x_max) < tol
    at_bottom= abs(yc - y_min) < tol
    at_top   = abs(yc - y_max) < tol
    return at_left or at_right or at_bottom or at_top, dict(left=at_left,right=at_right,bottom=at_bottom,top=at_top)

# ===================== UI（キー付き） =====================
# ※ フォームを使わず、外部の［計算開始］ボタンだけで確定・計算する

# ---- 初期値のセット（一度だけ） ----
def _init_once():
    ss = st.session_state
    ss.setdefault("ui_H", 25.0)
    ss.setdefault("ui_L", 60.0)
    ss.setdefault("ui_layers", 3)
    ss.setdefault("ui_soils", [(18.0,5.0,30.0),(19.0,8.0,28.0),(20.0,12.0,25.0)])  # (γ,c,φ)
    ss.setdefault("ui_allow_cross", [True, True])  # into layer2, into layer3
    ss.setdefault("ui_Fs_target", 1.20)
    ss.setdefault("ui_xmin_ratio", 0.25)
    ss.setdefault("ui_xmax_ratio", 1.15)
    ss.setdefault("ui_ymin_ratio", 1.60)
    ss.setdefault("ui_ymax_ratio", 2.20)
    ss.setdefault("ui_nx", 14)
    ss.setdefault("ui_ny", 9)
    ss.setdefault("ui_method", "Fellenius")
    ss.setdefault("ui_quality_label", "Normal")
    ss.setdefault("ui_override", False)
    ss.setdefault("ui_depth_min", 0.5)
    ss.setdefault("ui_depth_max", 4.0)
    # overrideの値
    P0 = QUALITY[ss["ui_quality_label"]]
    ss.setdefault("ui_quick_slices", P0["quick_slices"])
    ss.setdefault("ui_final_slices", P0["final_slices"])
    ss.setdefault("ui_n_entries_final", P0["n_entries_final"])
    ss.setdefault("ui_probe_min_q", P0["probe_n_min_quick"])
    ss.setdefault("ui_limit_arcs_q", P0["limit_arcs_quick"])
    ss.setdefault("ui_budget_coarse", P0["budget_coarse_s"])
    ss.setdefault("ui_budget_quick", P0["budget_quick_s"])
    # 実行フラグ・確定値
    ss.setdefault("compute_request", False)
    if "committed_params" not in ss:
        # 初回はデフォルトで確定しておく（初期表示のため）
        params = build_params_from_ui()
        ss["committed_params"] = params
        ss["last_ui_hash"] = hash_params(params)

def build_params_from_ui():
    ss = st.session_state
    H = float(ss["ui_H"]); L = float(ss["ui_L"])
    n_layers = int(ss["ui_layers"])
    soils_tuple = ss["ui_soils"][:n_layers]
    soils = [Soil(*t) for t in soils_tuple]
    allow_cross = []
    if n_layers>=2: allow_cross.append(bool(ss["ui_allow_cross"][0]))
    if n_layers>=3: allow_cross.append(bool(ss["ui_allow_cross"][1]))
    quality_label = ss["ui_quality_label"]
    P = QUALITY[quality_label].copy()
    if ss["ui_override"]:
        P.update(dict(
            quick_slices=int(ss["ui_quick_slices"]),
            final_slices=int(ss["ui_final_slices"]),
            n_entries_final=int(ss["ui_n_entries_final"]),
            probe_n_min_quick=int(ss["ui_probe_min_q"]),
            limit_arcs_quick=int(ss["ui_limit_arcs_q"]),
            budget_coarse_s=float(ss["ui_budget_coarse"]),
            budget_quick_s=float(ss["ui_budget_quick"]),
        ))
    x_min = float(ss["ui_xmin_ratio"] * L)
    x_max = float(ss["ui_xmax_ratio"] * L)
    y_min = float(ss["ui_ymin_ratio"] * H)
    y_max = float(ss["ui_ymax_ratio"] * H)
    return dict(
        H=H, L=L, n_layers=n_layers,
        soils=[(s.gamma, s.c, s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=float(ss["ui_Fs_target"]),
        center=[x_min, x_max, y_min, y_max, int(ss["ui_nx"]), int(ss["ui_ny"])],
        method=str(ss["ui_method"]), quality=P, depth=[float(ss["ui_depth_min"]), float(ss["ui_depth_max"])],
        quality_label=quality_label,
    )

_init_once()

# ---- UI レイアウト（数値入力は全部 key=... でセッション管理） ----
A, B = st.columns(2)
with A:
    st.subheader("Geometry")
    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="ui_H")
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="ui_L")

    st.subheader("Layers")
    st.selectbox("Number of layers", [1,2,3], key="ui_layers")

    st.subheader("Soils (top→bottom)")
    # γ, c, φ を列挙
    for i in range(st.session_state["ui_layers"]):
        g,c,phi = st.session_state["ui_soils"][i]
        cols = st.columns(3)
        st.session_state["ui_soils"][i] = (
            cols[0].number_input(f"γ{i+1}", 10.0, 25.0, g, 0.5, key=f"ui_gamma_{i+1}"),
            cols[1].number_input(f"c{i+1}", 0.0, 200.0, c, 0.5, key=f"ui_c_{i+1}"),
            cols[2].number_input(f"φ{i+1}", 0.0, 45.0, phi, 0.5, key=f"ui_phi_{i+1}")
        )

    st.subheader("Crossing control")
    if st.session_state["ui_layers"] >= 2:
        st.session_state["ui_allow_cross"][0] = st.checkbox("Allow into Layer 2", st.session_state["ui_allow_cross"][0], key="ui_allow2")
    if st.session_state["ui_layers"] >= 3:
        st.session_state["ui_allow_cross"][1] = st.checkbox("Allow into Layer 3", st.session_state["ui_allow_cross"][1], key="ui_allow3")

    st.subheader("Target safety")
    st.number_input("Target FS", min_value=1.00, max_value=2.00, step=0.05, key="ui_Fs_target")

with B:
    st.subheader("Center grid（比率指定）")
    st.number_input("x min / L", 0.20, 3.00, key="ui_xmin_ratio", step=0.05, format="%.2f")
    st.number_input("x max / L", 0.30, 4.00, key="ui_xmax_ratio", step=0.05, format="%.2f")
    st.number_input("y min / H", 0.80, 7.00, key="ui_ymin_ratio", step=0.10, format="%.2f")
    st.number_input("y max / H", 1.00, 8.00, key="ui_ymax_ratio", step=0.10, format="%.2f")
    st.slider("nx", 6, 60, key="ui_nx")
    st.slider("ny", 4, 40, key="ui_ny")

    st.subheader("Method / Quality")
    st.selectbox("Method", ["Bishop (simplified)","Fellenius"], key="ui_method")
    st.select_slider("Quality", options=list(QUALITY.keys()), key="ui_quality_label")
    with st.expander("Advanced", expanded=False):
        st.checkbox("Override Quality", key="ui_override")
        Ptmp = QUALITY[st.session_state["ui_quality_label"]]
        if st.session_state["ui_override"]:
            st.slider("Quick slices", 6, 40, Ptmp["quick_slices"], 1, key="ui_quick_slices")
            st.slider("Final slices", 20, 80, Ptmp["final_slices"], 2, key="ui_final_slices")
            st.slider("Final n_entries", 200, 4000, Ptmp["n_entries_final"], 100, key="ui_n_entries_final")
            st.slider("Quick min probe", 41, 221, Ptmp["probe_n_min_quick"], 10, key="ui_probe_min_q")
            st.slider("Quick max arcs/center", 20, 400, Ptmp["limit_arcs_quick"], 10, key="ui_limit_arcs_q")
            st.slider("Budget Coarse (s)", 0.1, 5.0, Ptmp["budget_coarse_s"], 0.1, key="ui_budget_coarse")
            st.slider("Budget Quick (s)", 0.1, 5.0, Ptmp["budget_quick_s"], 0.1, key="ui_budget_quick")

    st.subheader("Depth range (vertical)")
    st.number_input("Depth min (m)", min_value=0.0, max_value=50.0, step=0.5, key="ui_depth_min")
    st.number_input("Depth max (m)", min_value=0.5, max_value=50.0, step=0.5, key="ui_depth_max")

# ---- 計算開始ボタン（外部・on_clickでフラグだけ立てる） ----
def _request_compute():
    st.session_state["compute_request"] = True
st.button("▶ 計算開始", type="primary", on_click=_request_compute)

# ---- UI変更と確定値の差分通知 ----
ui_params = build_params_from_ui()
ui_hash = hash_params(ui_params)
if ui_hash != st.session_state.get("last_ui_hash", ""):
    st.info("パラメータは変更されていますが、**まだ計算に反映されていません**。［▶ 計算開始］で確定します。")

# ===================== 以降は「確定パラメータ」だけを使用 =====================
def build_scene(H,L,n_layers):
    ground = make_ground_example(H, L)
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H, L))
    if n_layers>=3: interfaces.append(make_interface2_example(H, L))
    return ground, interfaces

def compute_once(params):
    H=params["H"]; L=params["L"]; n_layers=params["n_layers"]
    ground, interfaces = build_scene(H,L,n_layers)
    allow_cross=params["allow_cross"]; Fs_target=params["Fs_target"]
    method=params["method"]; P=params["quality"]
    x_min, x_max, y_min, y_max, nx, ny = params["center"]
    depth_min, depth_max = params["depth"]

    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        tag = P["coarse_subsample"]
        if tag == "every 3rd":
            xs = xs[::3] if len(xs)>2 else xs
            ys = ys[::3] if len(ys)>2 else ys
        elif tag == "every 2nd":
            xs = xs[::2] if len(xs)>1 else xs
            ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(budget_s):
        deadline = time.time() + budget_s
        best = None; tested=[]
        for (xc,yc) in subsampled_centers():
            cnt=0; Fs_min=None
            for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                ground, [Soil(*t) for t in params["soils"]], xc, yc,
                n_entries=P["coarse_entries"], method="Fellenius",
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                limit_arcs_per_center=P["coarse_limit_arcs"],
                probe_n_min=P["coarse_probe_min"],
            ):
                cnt+=1
                if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                if time.time() > deadline: break
            tested.append((xc,yc))
            score = (cnt, - (Fs_min if Fs_min is not None else 1e9))
            if (best is None) or (score > best[0]): best = (score, (xc,yc))
            if time.time() > deadline: break
        return (best[1] if best else None), tested

    with st.spinner("Coarse（最有望センター選抜）"):
        center, tested = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseで候補なし。枠/深さを広げてください。")
    xc, yc = center

    # 端ヒット時に監査枠だけ拡張
    hit, where = near_edge(xc,yc,x_min,x_max,y_min,y_max)
    x_min_a, x_max_a, y_min_a, y_max_a = x_min, x_max, y_min, y_max
    expand_note=None
    if hit:
        dx = (x_max - x_min); dy = (y_max - y_min)
        if where["left"]:  x_min_a = x_min - 0.20*dx
        if where["right"]: x_max_a = x_max + 0.20*dx
        if where["bottom"]:y_min_a = y_min - 0.20*dy
        if where["top"]:   y_max_a = y_max + 0.20*dy
        expand_note = f"Auto-extend audit grid: x[{x_min_a:.1f},{x_max_a:.1f}], y[{y_min_a:.1f},{y_max_a:.1f}]"

    # Quick（R候補抽出）
    with st.spinner("Quick（R候補抽出）"):
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, [Soil(*t) for t in params["soils"]], xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Qualityを緩めてください。")

    # Refine
    refined=[]
    for R in R_candidates[:P["show_k"]]:
        Fs_val = fs_given_R_multi(ground, [], [Soil(*t) for t in params["soils"]], allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
        if Fs_val is None: continue
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
        if s is None: continue
        x1,x2,*_ = s
        packD = driving_sum_for_R_multi(ground, [], [Soil(*t) for t in params["soils"]], allow_cross, xc, yc, R, n_slices=P["final_slices"])
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (params["Fs_target"] - Fs_val)*D_sum)
        refined.append(dict(Fs=float(Fs_val), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。")
    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    centers_audit= grid_points(x_min_a, x_max_a, y_min_a, y_max_a, nx, ny)

    return dict(
        params=params, ground=ground, interfaces=interfaces,
        center=(xc,yc), refined=refined,
        idx_minFs=idx_minFs, idx_maxT=idx_maxT,
        centers_disp=centers_disp, centers_audit=centers_audit,
        expand_note=expand_note
    )

# ---- 実行条件：初回 or compute_request=True のときだけ計算 ----
if ("res" not in st.session_state) or st.session_state.get("compute_request", False):
    # ボタン押下時：UIを確定→計算
    if st.session_state.get("compute_request", False):
        st.session_state["committed_params"] = ui_params
        st.session_state["last_ui_hash"] = ui_hash
        st.session_state["compute_request"] = False
    res = compute_once(st.session_state["committed_params"])
    if "error" in res: st.error(res["error"]); st.stop()
    st.session_state["res"] = res

res = st.session_state["res"]
params = res["params"]
ground = res["ground"]; interfaces = res["interfaces"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]; centers_audit = res["centers_audit"]
Fs_target = params["Fs_target"]; method=params["method"]; n_layers=params["n_layers"]
L=params["L"]; H=params["H"]
quality_label = params["quality_label"]

# ---------------- 表示オプション（表示だけ。計算は走らない） ----------------
st.subheader("表示オプション")
c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    show_centers = st.checkbox("Show center-grid (all points)", True)
with c2:
    show_all_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
with c3:
    show_minFs = st.checkbox("Show Min Fs", True)
    show_maxT  = st.checkbox("Show Max required T", True)
with c4:
    audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)
    audit_limit = st.slider("Audit: max arcs/center", 5, 40, QUALITY[quality_label]["audit_limit_per_center"], 1, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 1.0, 6.0, QUALITY[quality_label]["audit_budget_s"], 0.1, disabled=not audit_show)
    audit_seed   = st.number_input("Audit seed", 0, 9999, 0, disabled=not audit_show)

# ---------------- Audit computation（可視化だけ。確定Pで） ----------------
def compute_audit_arcs(centers, per_center_limit, total_budget_s, seed=0):
    rng = random.Random(int(seed))
    order = list(range(len(centers)))
    rng.shuffle(order)
    deadline = time.time() + total_budget_s
    arcs=[]; covered=set()
    for idx in order:
        if time.time() > deadline: break
        cx,cy = centers[idx]
        count=0
        for a_x1,a_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, [Soil(*t) for t in params["soils"]], cx, cy,
            n_entries=min(params["quality"]["n_entries_final"], 1200), method="Fellenius",
            depth_min=params["depth"][0], depth_max=params["depth"][1],
            interfaces=interfaces, allow_cross=params["allow_cross"],
            quick_mode=True, n_slices_quick=max(10, params["quality"]["quick_slices"]),
            limit_arcs_per_center=per_center_limit,
            probe_n_min=max(81, params["quality"]["probe_n_min_quick"]),
        ):
            arcs.append(dict(x1=a_x1,x2=a_x2,R=R,Fs=Fs,xc=cx,yc=cy))
            count+=1
            if count>=per_center_limit: break
        if count>0: covered.add(idx)
        if time.time() > deadline: break
    return arcs, len(covered), len(centers)

audit_arcs=[]; covered=0; total=0
if audit_show:
    akey = hash_params(dict(K=st.session_state["last_ui_hash"], limit=audit_limit, budget=round(audit_budget,2), seed=int(audit_seed)))
    cache = st.session_state.get("audit_cache", {})
    if cache.get("key")==akey and "arcs" in cache:
        audit_arcs=cache["arcs"]; covered=cache["covered"]; total=cache["total"]
    else:
        with st.spinner("Audit（全センターQuick可視化）..."):
            audit_arcs, covered, total = compute_audit_arcs(centers_audit, audit_limit, audit_budget, seed=audit_seed)
        st.session_state["audit_cache"] = {"key": akey, "arcs": audit_arcs, "covered": covered, "total": total}

# ---------------- Plot（確定パラメータの地形で描画） ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))

Xd = np.linspace(0.0, L, 600)
Yg = ground.y_at(Xd)

if n_layers==1:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers==2:
    Y1 = clip_interfaces_to_ground(ground, [make_interface1_example(H,L)], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [make_interface1_example(H,L), make_interface2_example(H,L)], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [make_interface1_example(H,L)], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [make_interface1_example(H,L), make_interface2_example(H,L)], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")

# 外周
ax.plot([ground.X[-1], ground.X[-1]],[0.0, ground.y_at(ground.X[-1])], linewidth=1.0)
ax.plot([ground.X[0],  ground.X[-1]],[0.0, 0.0],                       linewidth=1.0)
ax.plot([ground.X[0],  ground.X[0]], [0.0, ground.y_at(ground.X[0])],  linewidth=1.0)

# center-grid
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")

# chosen center
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# audit arcs
if audit_show and audit_arcs:
    for a in audit_arcs:
        cx,cy,R,x1,x2,Fs = a["xc"],a["yc"],a["R"],a["x1"],a["x2"],a["Fs"]
        xs = np.linspace(x1, x2, 140)
        ys = cy - np.sqrt(np.maximum(0.0, R*R - (xs - cx)**2))
        ax.plot(xs, ys, linewidth=0.6, alpha=0.25, color=fs_to_color(Fs))

# refined (chosen center)
if show_all_refined:
    for d in refined:
        xs=np.linspace(d["x1"], d["x2"], 200)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

# pick-ups
if show_minFs and 0<=idx_minFs<len(refined):
    d=refined[idx_minFs]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    y1=float(ground.y_at(d["x1"])); y2=float(ground.y_at(d["x2"]))
    ax.plot([xc,d["x1"]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
    ax.plot([xc,d["x2"]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if show_maxT and 0<=idx_maxT<len(refined):
    d=refined[idx_maxT]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
            label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
    y1=float(ground.y_at(d["x1"])); y2=float(ground.y_at(d["x2"]))
    ax.plot([xc,d["x1"]],[yc,y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
    ax.plot([xc,d["x2"]],[yc,y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

# axis & legend
x_min, x_max, y_min, y_max, nx, ny = params["center"]
x_upper = max(1.18*L, x_max + 0.05*L, 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
ax.set_ylim(0.0, y_upper)
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
h,l = ax.get_legend_handles_labels()
ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)

title_tail=[f"MinFs={refined[idx_minFs]['Fs']:.3f}", f"TargetFs={Fs_target:.2f}"]
if res.get("expand_note"): title_tail.append(res["expand_note"])
if 'audit_arcs' in locals() and audit_show:
    title_tail.append(f"audit cover {covered}/{total} centers, arcs={len(audit_arcs)}")
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))

st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（精密・選抜センター）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
