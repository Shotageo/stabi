# streamlit_app.py — Full-span robust++（深さ方式選択・候補上位精査・自動フォールバック）
from __future__ import annotations
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq, time, json, hashlib
from dataclasses import dataclass

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
)

st.set_page_config(page_title="Stabi LEM｜Full-span robust++", layout="wide")
st.title("Stabi LEM｜Full-span robust++")

# ---------- utils ----------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    import hashlib as _h
    return _h.sha256(s.encode("utf-8")).hexdigest()[:16]

# 円×地表：全交点x座標
def circle_polyline_all_hits(ground: GroundPL, xc: float, yc: float, R: float):
    X = np.asarray(ground.X); Y = np.asarray(ground.Y)
    hits = []
    for i in range(len(X) - 1):
        x0, y0 = X[i],   Y[i]
        x1, y1 = X[i+1], Y[i+1]
        vx, vy = x1 - x0, y1 - y0
        A = vx*vx + vy*vy
        if A <= 0: 
            continue
        B = 2*((x0 - xc)*vx + (y0 - yc)*vy)
        C = (x0 - xc)**2 + (y0 - yc)**2 - R*R
        D = B*B - 4*A*C
        if D < 0: 
            continue
        sD = float(np.sqrt(D))
        for sgn in (-1.0, 1.0):
            t = (-B + sgn*sD) / (2*A)
            if 0.0 <= t <= 1.0:
                hits.append(float(x0 + t*vx))
    if not hits: 
        return []
    hits = sorted(hits)
    # 近接重複を除去
    ded = []
    for x in hits:
        if not ded or abs(x - ded[-1]) > 1e-6:
            ded.append(x)
    return ded

def arc_base_y(x, xc, yc, R):
    under = R*R - (x - xc)**2
    if under <= 0.0: 
        return None
    return yc - np.sqrt(under)

@dataclass
class SliceGeom:
    xL: float; xR: float; xC: float
    y_top: float; y_base: float
    h: float; dx: float
    sin_a: float; cos_a: float
    b: float

def make_slices_fullspan(ground: GroundPL, xc: float, yc: float, R: float,
                         x1: float, x2: float, n_slices: int):
    xs = np.linspace(x1, x2, n_slices+1)
    slices: list[SliceGeom] = []
    for i in range(n_slices):
        xL, xR = float(xs[i]), float(xs[i+1])
        xC = 0.5*(xL + xR)
        y_top = float(ground.y_at(xC))
        y_base = arc_base_y(xC, xc, yc, R)
        if y_base is None: 
            continue
        h = y_top - y_base
        if h <= 1e-6:
            continue
        dx = xR - xL
        sin_a = (xC - xc)/R
        cos_a = float(np.sqrt(max(1e-12, 1.0 - sin_a*sin_a)))
        b = dx / max(1e-12, cos_a)
        slices.append(SliceGeom(xL, xR, xC, y_top, y_base, h, dx, sin_a, cos_a, b))
    return slices

def soil_at_base(x: float, y_base: float, ground: GroundPL, interfaces: list[GroundPL], soils: list[Soil]):
    if not interfaces:
        return soils[0]
    yints = [float(ifc.y_at(x)) for ifc in interfaces]
    if y_base >= yints[0]: return soils[0]
    if len(interfaces) == 1: return soils[1]
    if y_base >= yints[1]: return soils[1]
    return soils[2]

def fs_fullspan(ground, interfaces, soils, method: str,
                xc: float, yc: float, R: float, x1: float, x2: float, n_slices: int):
    slices = make_slices_fullspan(ground, xc, yc, R, x1, x2, n_slices)
    if len(slices) < max(6, int(0.45*n_slices)):
        return None, None
    Ws=[]; sinA=[]; cosA=[]; bs=[]; cs=[]; tans=[]
    for s in slices:
        soil = soil_at_base(s.xC, s.y_base, ground, interfaces, soils)
        gamma, c, phi = soil.gamma, soil.c, soil.phi
        W = gamma * s.h * s.dx
        Ws.append(W); sinA.append(s.sin_a); cosA.append(s.cos_a); bs.append(s.b)
        cs.append(c); tans.append(np.tan(np.deg2rad(phi)))
    Ws=np.array(Ws); sinA=np.array(sinA); cosA=np.array(cosA)
    bs=np.array(bs); cs=np.array(cs); tans=np.array(tans)
    D = float(np.sum(Ws * sinA))
    if D <= 1e-9:
        return None, None

    if method.startswith("Fellenius"):
        Rnum = np.sum(cs*bs + (Ws * cosA) * tans)
        FS = float(Rnum / D)
        return FS, D

    # Bishop (simplified)
    FS = 1.20
    for _ in range(60):
        tanA = sinA/np.maximum(1e-12, cosA)
        m = 1.0 + (tans * tanA)/max(1e-12, FS)
        Rnum = np.sum((cs*bs + (Ws * cosA) * tans) / np.maximum(1e-12, m))
        FS_new = float(Rnum / D)
        if abs(FS_new - FS) < 1e-4:
            FS = FS_new; break
        FS = FS_new
    return FS, D

# ---------- presets ----------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=36, n_entries_final=900,
                   probe_n_min_quick=81, limit_arcs_quick=80,
                   budget_coarse_s=0.6, budget_quick_s=0.9, show_k=80,
                   pairs_eval_top=4),
    "Normal": dict(quick_slices=12, final_slices=44, n_entries_final=1300,
                   probe_n_min_quick=101, limit_arcs_quick=120,
                   budget_coarse_s=0.8, budget_quick_s=1.2, show_k=140,
                   pairs_eval_top=6),
    "Fine": dict(quick_slices=16, final_slices=56, n_entries_final=1700,
                 probe_n_min_quick=121, limit_arcs_quick=160,
                 budget_coarse_s=1.2, budget_quick_s=1.8, show_k=200,
                 pairs_eval_top=8),
}

# ---------- inputs ----------
with st.form("params"):
    A,B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)
        ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces=[]
        if n_layers>=2: interfaces.append(make_interface1_example(H, L))
        if n_layers>=3: interfaces.append(make_interface2_example(H, L))

        st.subheader("Soils (top→bottom)")
        soils=[Soil(st.number_input("γ₁",10.0,25.0,18.0,0.5),
                    st.number_input("c₁",0.0,200.0,5.0,0.5),
                    st.number_input("φ₁",0.0,45.0,30.0,0.5))]
        if n_layers>=2:
            soils.append(Soil(st.number_input("γ₂",10.0,25.0,19.0,0.5),
                              st.number_input("c₂",0.0,200.0,8.0,0.5),
                              st.number_input("φ₂",0.0,45.0,28.0,0.5)))
        if n_layers>=3:
            soils.append(Soil(st.number_input("γ₃",10.0,25.0,20.0,0.5),
                              st.number_input("c₃",0.0,200.0,12.0,0.5),
                              st.number_input("φ₃",0.0,45.0,25.0,0.5)))

        st.subheader("Crossing control")
        allow_cross=[]
        if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
        if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

        st.subheader("Target")
        Fs_target = st.number_input("Target FS (T_req)", 1.00, 2.00, 1.20, 0.05)

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

        st.subheader("Depth filter")
        depth_mode = st.selectbox("Filter mode", ["By apex (deepest)", "By endpoints", "Off"], index=0)
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 6.0, 0.5)

        st.subheader("Display")
        show_minFs = st.checkbox("Show Min Fs", True)
        show_maxT  = st.checkbox("Show Max required T", True)
        show_all_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
        show_centers = st.checkbox("Show center-grid points", True)

    run = st.form_submit_button("▶ 計算開始（Full-span robust++）")

# Quality適用
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# ---------- keys ----------
param_key = hash_params(dict(
    H=H, L=L, n_layers=n_layers,
    soils=[(s.gamma, s.c, s.phi) for s in soils],
    allow_cross=allow_cross, Fs_target=Fs_target,
    center=[x_min, x_max, y_min, y_max, nx, ny],
    method=method, quality=P, depth=[depth_mode, depth_min, depth_max],
))

# ---------- core ----------
def depth_pass(ground, xc, yc, R, x1, x2, mode, dmin, dmax):
    xs = np.linspace(x1, x2, 41)
    ds = []
    for x in xs:
        y_surf = float(ground.y_at(x))
        y_base = arc_base_y(x, xc, yc, R)
        if y_base is None: continue
        ds.append(y_surf - y_base)
    if not ds: return False
    d_left  = ds[1]
    d_right = ds[-2]
    d_apex  = float(np.max(ds))
    if mode == "Off":
        return True
    if mode == "By endpoints":
        return (dmin <= d_left <= dmax) and (dmin <= d_right <= dmax)
    # By apex (deepest)
    return (dmin <= d_apex <= dmax)

def compute_once():
    # 1) Coarse：選抜センター
    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xs = xs[::2] if len(xs)>1 else xs
        ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(budget_s):
        deadline = time.time() + budget_s
        best=None; tested=[]
        for (xc,yc) in subsampled_centers():
            cnt=0; Fs_best=None
            for qx1,qx2,_R,Fs_q in arcs_from_center_by_entries_multi(
                ground, soils, xc, yc,
                n_entries=min(800, P["n_entries_final"]), method="Fellenius",
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                limit_arcs_per_center=min(60, P["limit_arcs_quick"]),
                probe_n_min=max(61, P["probe_n_min_quick"]-40),
            ):
                cnt+=1
                if (Fs_best is None) or (Fs_q < Fs_best): Fs_best = Fs_q
                if time.time() > deadline: break
            tested.append((xc,yc))
            score=(cnt, -(Fs_best if Fs_best is not None else 1e9))
            if (best is None) or (score > best[0]): best=(score,(xc,yc))
            if time.time() > deadline: break
        return (best[1] if best else None), tested

    with st.spinner("Coarse（センター選抜）"):
        center, _tested = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseでセンターなし。枠/深さを調整してください。")
    xc, yc = center

    # 2) Quick：R候補＋ヒント
    with st.spinner("Quick（R候補抽出）"):
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        hints=[]
        for qx1,qx2,R,Fs_q in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs_q, R))
            hints.append((float(R), (qx1, qx2)))
            if len(heap_R) > P["show_k"]: heapq.heappop(heap_R)
            if time.time() > deadline: break
        if not heap_R:
            return dict(error="Quickで候補なし。条件を緩めてください。")
        tmp = sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])
        R_candidates = [float(r) for _fs, r in tmp]
        hint_map = {}
        for R, pair in hints:
            hint_map.setdefault(round(R,6), pair)

    # 3) Refine：全交点→ペアを評価、上位だけ精密Fs
    refined=[]
    rescue_msg=[]
    def try_refine(min_cov, pairs_top, mode_for_depth):
        nonlocal refined
        for R in R_candidates:
            xs_hits = circle_polyline_all_hits(ground, xc, yc, R)
            if len(xs_hits) < 2: 
                continue
            # 交点ペアを走査してスコア（平均厚・被覆率・ヒント近さ）
            cand=[]
            hint_pair = hint_map.get(round(R,6))
            hint_mid = None if hint_pair is None else 0.5*(hint_pair[0]+hint_pair[1])
            for i in range(len(xs_hits)-1):
                x1, x2 = xs_hits[i], xs_hits[i+1]
                xs = np.linspace(x1, x2, 25)
                valid=0; sum_h=0.0; miss=0
                for x in xs:
                    y_s = float(ground.y_at(x))
                    y_b = arc_base_y(x, xc, yc, R)
                    if y_b is None: 
                        miss += 1; continue
                    h = y_s - y_b
                    if h > 1e-6:
                        valid += 1; sum_h += h
                cov = valid / max(1, len(xs))
                if cov < min_cov: 
                    continue
                mean_h = sum_h / max(1, valid)
                hint_d = 0.0 if hint_mid is None else abs(0.5*(x1+x2) - hint_mid)
                # 厚いほど良い、ヒントに近いほど良い → (-mean_h, hint_d)
                cand.append((( -mean_h, hint_d), (x1, x2), cov))
            if not cand:
                continue
            cand.sort(key=lambda t: t[0])
            for _score, (x1,x2), cov in cand[:pairs_top]:
                if not depth_pass(ground, xc, yc, R, x1, x2, mode_for_depth, depth_min, depth_max):
                    continue
                FS, D = fs_fullspan(ground, interfaces, soils, method, xc, yc, R, x1, x2, n_slices=P["final_slices"])
                if FS is None or D is None: 
                    continue
                T_req = max(0.0, (Fs_target - FS) * D)
                refined.append(dict(R=float(R), x1=float(x1), x2=float(x2),
                                    Fs=float(FS), D=float(D), T_req=float(T_req), cov=float(cov)))
        return len(refined) > 0

    # 段階的フォールバック
    if not try_refine(min_cov=0.85, pairs_top=P["pairs_eval_top"], mode_for_depth=depth_mode):
        rescue_msg.append("fallback: min_cov→0.70")
        if not try_refine(min_cov=0.70, pairs_top=P["pairs_eval_top"], mode_for_depth=depth_mode):
            rescue_msg.append("fallback: min_cov→0.50")
            if not try_refine(min_cov=0.50, pairs_top=P["pairs_eval_top"]+2, mode_for_depth=depth_mode):
                # 最後の手：深さ方式を端点基準に切替
                if depth_mode != "By endpoints":
                    rescue_msg.append("fallback: depth filter→By endpoints")
                    if not try_refine(min_cov=0.50, pairs_top=P["pairs_eval_top"]+2, mode_for_depth="By endpoints"):
                        pass

    if not refined:
        msg = "Refineで有効弧が得られません。Depth/枠/Qualityを緩めるか、Depth filterをOff/Endpointsに。"
        if rescue_msg: msg += " [" + " , ".join(rescue_msg) + "]"
        return dict(error=msg)

    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = [(float(xc), float(yc)) for yc in np.linspace(y_min, y_max, ny)
                                      for xc in np.linspace(x_min, x_max, nx)]
    return dict(center=(xc,yc), refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp, rescue_msg="; ".join(rescue_msg))

# run
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res: st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs=res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(10.5, 7.5))
Xd = np.linspace(0.0, L, 600); Yg = [ground.y_at(x) for x in Xd]

# 塗り
if n_layers == 1:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif n_layers == 2:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

# 地表線＋外周
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.1)
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.1)
ax.plot([L, L],[0.0, ground.y_at(L)], linewidth=1.0)
ax.plot([0.0, L],[0.0, 0.0],         linewidth=1.0)
ax.plot([0.0, 0.0],[0.0, ground.y_at(0.0)], linewidth=1.0)

# センター
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# 全Refined弧（フルスパン）
if show_all_refined:
    for d in refined:
        xs = np.linspace(d["x1"], d["x2"], 240)
        ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

# ピックアップ
if show_minFs and 0 <= idx_minFs < len(refined):
    d = refined[idx_minFs]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
    ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
    ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if show_maxT and 0 <= idx_maxT < len(refined):
    d = refined[idx_maxT]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
            label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
    y1 = float(ground.y_at(d["x1"])); y2 = float(ground.y_at(d["x2"]))
    ax.plot([xc, d["x1"]], [yc, y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
    ax.plot([xc, d["x2"]], [yc, y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

# 軸・凡例
ax.set_xlim(min(-2.0, -0.05*L), max(1.18*L, 100.0))
ax.set_ylim(0.0, max(2.3*H, 100.0))
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
h,l = ax.get_legend_handles_labels()
ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)

subtitle = f"Center=({xc:.2f},{yc:.2f}) • Method={method} • Depth mode={depth_mode}"
if "rescue_msg" in res and res["rescue_msg"]:
    subtitle += f" • {res['rescue_msg']}"
ax.set_title(f"Full-span robust++ • MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_target:.2f}\n{subtitle}")

st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（Full-span, 精密）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
