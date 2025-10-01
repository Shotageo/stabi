# streamlit_app.py — Full-span GridMask 版（単純・高速・壊れにくい）
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

st.set_page_config(page_title="Stabi LEM｜Full-span GridMask", layout="wide")
st.title("Stabi LEM｜Full-span（GridMask）")

# ---------- utils ----------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# 円底
def arc_base_y_vec(X, xc, yc, R):
    under = R*R - (X - xc)**2
    Yb = np.full_like(X, np.nan, dtype=float)
    m = under > 0.0
    Yb[m] = yc - np.sqrt(under[m])
    return Yb

@dataclass
class SliceGeom:
    xL: float; xR: float; xC: float
    y_top: float; y_base: float
    h: float; dx: float
    sin_a: float; cos_a: float
    b: float

def make_slices_fullspan_grid(ground: GroundPL, xc: float, yc: float, R: float,
                              x1: float, x2: float, n_slices: int):
    xs = np.linspace(x1, x2, n_slices+1)
    slices: list[SliceGeom] = []
    for i in range(n_slices):
        xL, xR = float(xs[i]), float(xs[i+1])
        xC = 0.5*(xL + xR)
        y_top = float(ground.y_at(xC))
        under = R*R - (xC - xc)**2
        if under <= 0.0: 
            continue
        y_base = yc - np.sqrt(under)
        h = y_top - y_base
        if h <= 1e-6:
            continue
        dx = xR - xL
        sin_a = (xC - xc)/R
        cos_a = float(np.sqrt(max(1e-12, 1.0 - sin_a*sin_a)))
        b = dx / max(1e-12, cos_a)
        slices.append(SliceGeom(xL, xR, xC, y_top, y_base, h, dx, sin_a, cos_a, b))
    return slices

def soil_at_base(x: float, y_base: float, ground: GroundPL, interfaces, soils):
    if not interfaces: return soils[0]
    ys = [float(ifc.y_at(x)) for ifc in interfaces]
    if y_base >= ys[0]: return soils[0]
    if len(ys) == 1:    return soils[1]
    if y_base >= ys[1]: return soils[1]
    return soils[2]

def fs_fullspan(ground, interfaces, soils, method: str,
                xc: float, yc: float, R: float, x1: float, x2: float, n_slices: int):
    slices = make_slices_fullspan_grid(ground, xc, yc, R, x1, x2, n_slices)
    if len(slices) < max(6, int(0.45*n_slices)): return None, None
    Ws=[]; sinA=[]; cosA=[]; bs=[]; cs=[]; tans=[]
    for s in slices:
        soil = soil_at_base(s.xC, s.y_base, ground, interfaces, soils)
        W = soil.gamma * s.h * s.dx
        Ws.append(W); sinA.append(s.sin_a); cosA.append(s.cos_a); bs.append(s.b)
        cs.append(soil.c); tans.append(np.tan(np.deg2rad(soil.phi)))
    Ws=np.array(Ws); sinA=np.array(sinA); cosA=np.array(cosA)
    bs=np.array(bs); cs=np.array(cs); tans=np.array(tans)
    D = float(np.sum(Ws * sinA))
    if D <= 1e-9: return None, None
    if method.startswith("Fellenius"):
        Rnum = np.sum(cs*bs + (Ws * cosA) * tans)
        return float(Rnum / D), D
    # Bishop (simplified)
    FS = 1.20
    for _ in range(60):
        tanA = sinA/np.maximum(1e-12, cosA)
        m = 1.0 + (tans * tanA)/max(1e-12, FS)
        Rnum = np.sum((cs*bs + (Ws * cosA) * tans) / np.maximum(1e-12, m))
        FS_new = float(Rnum / D)
        if abs(FS_new - FS) < 1e-4: FS = FS_new; break
        FS = FS_new
    return FS, D

# ---------- presets ----------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=36, n_entries_final=900,
                   probe_n_min_quick=81, limit_arcs_quick=80,
                   budget_coarse_s=0.6, budget_quick_s=0.9, show_k=80,
                   NX=801),
    "Normal": dict(quick_slices=12, final_slices=44, n_entries_final=1300,
                   probe_n_min_quick=101, limit_arcs_quick=120,
                   budget_coarse_s=0.8, budget_quick_s=1.2, show_k=140,
                   NX=1201),
    "Fine": dict(quick_slices=16, final_slices=56, n_entries_final=1700,
                 probe_n_min_quick=121, limit_arcs_quick=160,
                 budget_coarse_s=1.2, budget_quick_s=1.8, show_k=200,
                 NX=1601),
}

# ---------- UI ----------
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
            NX_in = st.slider("Span grid NX", 401, 2001, QUALITY[quality]["NX"], 100, disabled=not override)

        st.subheader("Depth filter (GridMask上)")
        depth_mode = st.selectbox("Mode", ["By apex (deepest)", "By endpoints", "Off"], index=0)
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 6.0, 0.5)

        st.subheader("Display")
        show_minFs = st.checkbox("Show Min Fs", True)
        show_maxT  = st.checkbox("Show Max required T", True)
        show_all_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
        show_centers = st.checkbox("Show center-grid points", True)

    run = st.form_submit_button("▶ 計算開始（GridMask）")

# Quality適用
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
        NX=NX_in,
    ))

# ---------- key ----------
param_key = hash_params(dict(
    H=H, L=L, n_layers=n_layers,
    soils=[(s.gamma, s.c, s.phi) for s in soils],
    allow_cross=allow_cross, Fs_target=Fs_target,
    center=[x_min, x_max, y_min, y_max, nx, ny],
    method=method, quality=P, depth=[depth_mode, depth_min, depth_max],
))

# ---------- GridMask: 最長有効スパン抽出 ----------
def longest_valid_span_grid(Xg, Ys, xc, yc, R, mode, dmin, dmax, min_pts=6):
    Yb = arc_base_y_vec(Xg, xc, yc, R)
    good = np.isfinite(Yb)
    if not np.any(good): return None
    D = Ys - Yb
    mask = good & (D > 1e-6)

    if not np.any(mask): return None

    # 連続 True の区間を列挙
    idx = np.where(mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks+1]]
    ends   = np.r_[idx[breaks], idx[-1]]

    best = None
    for s,e in zip(starts, ends):
        if (e - s + 1) < min_pts: 
            continue
        x1, x2 = float(Xg[s]), float(Xg[e])
        ds = D[s:e+1]
        if mode == "Off":
            ok = True
        elif mode == "By endpoints":
            ok = (dmin <= ds[0] <= dmax) and (dmin <= ds[-1] <= dmax)
        else:  # apex
            ok = (dmin <= float(np.max(ds)) <= dmax)
        if not ok:
            continue
        span_len = x2 - x1
        if (best is None) or (span_len > best[0]):
            best = (span_len, x1, x2)
    if best is None: 
        return None
    _, x1, x2 = best
    return x1, x2

# ---------- 計算 ----------
def compute_once():
    # 1) Coarse：最有望センター選抜
    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        xs = xs[::2] if len(xs)>1 else xs
        ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(budget_s):
        deadline = time.time() + budget_s
        best=None
        for (xc,yc) in subsampled_centers():
            cnt=0; Fs_best=None
            for _x1,_x2,_R,Fs_q in arcs_from_center_by_entries_multi(
                ground, soils, xc, yc,
                n_entries=min(800, P["n_entries_final"]), method="Fellenius",
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                limit_arcs_per_center=min(60, P["limit_arcs_quick"]),
                probe_n_min=max(61, P["probe_n_min_quick"]-40),
            ):
                cnt += 1
                if (Fs_best is None) or (Fs_q < Fs_best): Fs_best = Fs_q
                if time.time() > deadline: break
            score=(cnt, -(Fs_best if Fs_best is not None else 1e9))
            if (best is None) or (score > best[0]): best=(score,(xc,yc))
            if time.time() > deadline: break
        return best[1] if best else None

    with st.spinner("Coarse（センター選抜）"):
        center = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseでセンターなし。枠/深さを調整してください。")
    xc, yc = center

    # 2) Quick：R候補抽出（低Fs順に保持）
    with st.spinner("Quick（R候補抽出）"):
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs_q in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs_q, float(R)))
            if len(heap_R) > P["show_k"]: heapq.heappop(heap_R)
            if time.time() > deadline: break
        if not heap_R: 
            return dict(error="Quickで候補なし。条件を緩めてください。")
        tmp = sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])
        R_candidates = [r for _fs, r in tmp]

    # 3) Refine（GridMaskで x1–x2 を確定 → 一貫評価）
    Xg = np.linspace(0.0, L, P["NX"])
    Ys = np.array([ground.y_at(x) for x in Xg], dtype=float)

    refined=[]
    for R in R_candidates:
        span = longest_valid_span_grid(Xg, Ys, xc, yc, R, depth_mode, depth_min, depth_max, min_pts=8)
        if span is None: 
            continue
        x1, x2 = span
        FS, D = fs_fullspan(ground, interfaces, soils, method, xc, yc, R, x1, x2, n_slices=P["final_slices"])
        if FS is None or D is None: 
            continue
        T_req = max(0.0, (Fs_target - FS) * D)
        refined.append(dict(R=float(R), x1=float(x1), x2=float(x2), Fs=float(FS), D=float(D), T_req=float(T_req)))

    if not refined:
        return dict(error="Refineで有効弧が得られません（GridMask）。Depthを緩める/NXを増やす/Qualityを上げる。")

    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = [(float(xc), float(yc)) for yc in np.linspace(y_min, y_max, ny)
                                      for xc in np.linspace(x_min, x_max, nx)]
    return dict(center=(xc,yc), refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp)

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

# 全Refined弧
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

ax.set_title(f"GridMask • Center=({xc:.2f},{yc:.2f}) • Method={method} • "
             f"MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_target:.2f}")
st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（Full-span, GridMask）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
