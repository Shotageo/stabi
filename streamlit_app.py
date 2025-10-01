# streamlit_app.py — audit改善・端ヒット時の自動拡張・被覆率表示（Audit既定OFF/描画安全化）
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

st.set_page_config(page_title="Stabi LEM｜監査＆自動拡張", layout="wide")
st.title("Stabi LEM｜全センター監査 & 端ヒット自動拡張")

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

# ---------------- Inputs ----------------
with st.form("params"):
    A, B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)
        ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces = []
        if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
        if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

        st.subheader("Soils (top→bottom)")
        s1 = Soil(st.number_input("γ₁", 10.0, 25.0, 18.0, 0.5),
                  st.number_input("c₁", 0.0, 200.0, 5.0, 0.5),
                  st.number_input("φ₁", 0.0, 45.0, 30.0, 0.5))
        soils = [s1]
        if n_layers >= 2:
            s2 = Soil(st.number_input("γ₂", 10.0, 25.0, 19.0, 0.5),
                      st.number_input("c₂", 0.0, 200.0, 8.0, 0.5),
                      st.number_input("φ₂", 0.0, 45.0, 28.0, 0.5))
            soils.append(s2)
        if n_layers >= 3:
            s3 = Soil(st.number_input("γ₃", 10.0, 25.0, 20.0, 0.5),
                      st.number_input("c₃", 0.0, 200.0, 12.0, 0.5),
                      st.number_input("φ₃", 0.0, 45.0, 25.0, 0.5))
            soils.append(s3)

        st.subheader("Crossing control")
        allow_cross=[]
        if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
        if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

        st.subheader("Target safety")
        Fs_target = st.number_input("Target FS", 1.00, 2.00, 1.20, 0.05)

    with B:
        st.subheader("Center grid")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14)
        ny = st.slider("ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = st.select_slider("Quality", options=list(QUALITY.keys()), value="Normal")
        with st.expander("Advanced", expanded=False):
            override = st.checkbox("Override Quality", value=False)
            quick_slices_in  = st.slider("Quick slices", 6, 40, QUALITY[quality]["quick_slices"], 1, disabled=not override)
            final_slices_in  = st.slider("Final slices", 20, 80, QUALITY[quality]["final_slices"], 2, disabled=not override)
            n_entries_final_in = st.slider("Final n_entries", 200, 4000, QUALITY[quality]["n_entries_final"], 100, disabled=not override)
            probe_min_q_in   = st.slider("Quick min probe", 41, 221, QUALITY[quality]["probe_n_min_quick"], 10, disabled=not override)
            limit_arcs_q_in  = st.slider("Quick max arcs/center", 20, 400, QUALITY[quality]["limit_arcs_quick"], 10, disabled=not override)
            budget_coarse_in = st.slider("Budget Coarse (s)", 0.1, 5.0, QUALITY[quality]["budget_coarse_s"], 0.1, disabled=not override)
            budget_quick_in  = st.slider("Budget Quick (s)", 0.1, 5.0, QUALITY[quality]["budget_quick_s"], 0.1, disabled=not override)

        st.subheader("Depth range (vertical)")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)

    run = st.form_submit_button("▶ 計算開始")

# ---------------- Quality expand ----------------
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# ---------------- Keys ----------------
def param_pack():
    return dict(
        H=H, L=L, n_layers=n_layers,
        soils=[(s.gamma, s.c, s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=Fs_target,
        center=[x_min, x_max, y_min, y_max, nx, ny],
        method=method, quality=P, depth=[depth_min, depth_max],
    )
param_key = hash_params(param_pack())

# ---------------- Compute ----------------
def compute_once():
    # 1) Coarse: pick best center with time budget
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
                ground, soils, xc, yc,
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

    # 端ヒットなら監査用に外側へ一段拡張（計算枠はそのまま）
    hit, where = near_edge(xc,yc,x_min,x_max,y_min,y_max)
    expand_note = None
    x_min_a, x_max_a, y_min_a, y_max_a = x_min, x_max, y_min, y_max
    if hit:
        dx = (x_max - x_min); dy = (y_max - y_min)
        if where["left"]:  x_min_a = x_min - 0.20*dx
        if where["right"]: x_max_a = x_max + 0.20*dx
        if where["bottom"]:y_min_a = y_min - 0.20*dy
        if where["top"]:   y_max_a = y_max + 0.20*dy
        expand_note = f"Auto-extend audit grid: x[{x_min_a:.1f},{x_max_a:.1f}], y[{y_min_a:.1f},{y_max_a:.1f}]"

    # 2) Quick at chosen center → R candidates
    with st.spinner("Quick（R候補抽出）"):
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
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
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Qualityを緩めてください。")

    # 3) Refine for chosen center
    refined=[]
    for R in R_candidates[:P["show_k"]]:
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
        if Fs is None: continue
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
        if s is None: continue
        x1,x2,*_ = s
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (Fs_target - Fs)*D_sum)
        refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。")
    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    # 全センター監査用グリッド（表示重視、計算枠はそのまま）
    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    centers_audit= grid_points(x_min_a, x_max_a, y_min_a, y_max_a, nx, ny)

    return dict(center=(xc,yc), tested_centers=tested, refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp, centers_audit=centers_audit,
                expand_note=expand_note)

# run
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res: st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]; centers_audit = res["centers_audit"]

# ---------------- After-run toggles ----------------
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
    # 既定OFF（重さの主犯を封じる）
    audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)
    audit_limit = st.slider("Audit: max arcs/center", 5, 40, QUALITY[quality]["audit_limit_per_center"], 1, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 1.0, 6.0, QUALITY[quality]["audit_budget_s"], 0.1, disabled=not audit_show)
    audit_seed   = st.number_input("Audit seed", 0, 9999, 0, disabled=not audit_show)

# ---------------- Audit computation (round-robin-ish) ----------------
def compute_audit_arcs(centers, per_center_limit, total_budget_s, seed=0):
    rng = random.Random(int(seed))
    order = list(range(len(centers)))  # ← 余計な ) を削除
    rng.shuffle(order)  # 偏り防止
    deadline = time.time() + total_budget_s
    arcs=[]; covered=set()
    for idx in order:
        if time.time() > deadline: break
        cx,cy = centers[idx]
        count=0
        for a_x1,a_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, soils, cx, cy,
            n_entries=min(P["n_entries_final"], 1200), method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=max(10, P["quick_slices"]),
            limit_arcs_per_center=per_center_limit,
            probe_n_min=max(81, P["probe_n_min_quick"]),
        ):
            arcs.append(dict(x1=a_x1,x2=a_x2,R=R,Fs=Fs,xc=cx,yc=cy))
            count+=1
            if count>=per_center_limit: break
        if count>0: covered.add(idx)
        if time.time() > deadline: break
    return arcs, len(covered), len(centers)

audit_arcs=[]; covered=0; total=0
if audit_show:
    akey = hash_params(dict(K=param_key, limit=audit_limit, budget=round(audit_budget,2), seed=int(audit_seed),
                            xa=centers_audit[0][0] if centers_audit else 0.0))
    cache = st.session_state.get("audit_cache", {})
    if cache.get("key")==akey and "arcs" in cache:
        audit_arcs=cache["arcs"]; covered=cache["covered"]; total=cache["total"]
    else:
        with st.spinner("Audit（全センターQuick可視化）..."):
            audit_arcs, covered, total = compute_audit_arcs(centers_audit, audit_limit, audit_budget, seed=audit_seed)
        st.session_state["audit_cache"] = {"key": akey, "arcs": audit_arcs, "covered": covered, "total": total}

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))

Xd = np.linspace(ground.X[0], ground.X[-1], 600)
# ベクトル非対応でも安全に：配列内包 → ndarray
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

if n_layers==1:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers==2:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")
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
if "expand_note" in res and res["expand_note"]: title_tail.append(res["expand_note"])
# covered/total は audit_show True のときだけ追加
if 'audit_arcs' in locals() and audit_show:
    title_tail.append(f"audit cover {covered}/{total} centers, arcs={len(audit_arcs)}")
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))

st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（精密・選抜センター）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
