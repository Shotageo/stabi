# streamlit_app.py — Entry-sweep（復帰＋高速化）版
from __future__ import annotations
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq, time, json, hashlib
from dataclasses import dataclass

# 既存モジュール（stabi_lem.py）はそのまま利用
from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
)

st.set_page_config(page_title="Stabi LEM｜Entry-sweep (fast)", layout="wide")
st.title("Stabi LEM｜Entry-sweep（復帰＋高速化）")

# ---------------- 小物 ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)   # 赤
    if fs < 1.2:
        t = (fs - 1.0) / 0.2               # オレンジ→黄
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)                # 緑

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# 円弧底のy
def arc_base_y_scalar(x, xc, yc, R):
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

def make_slices_on_arc(ground: GroundPL, xc: float, yc: float, R: float,
                       x1: float, x2: float, n_slices: int):
    xs = np.linspace(x1, x2, n_slices+1)
    slices: list[SliceGeom] = []
    for i in range(n_slices):
        xL, xR = float(xs[i]), float(xs[i+1])
        xC = 0.5*(xL + xR)
        y_top = float(ground.y_at(xC))
        y_base = arc_base_y_scalar(xC, xc, yc, R)
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

def soil_at_base(x: float, y_base: float, ground, interfaces, soils):
    if not interfaces: return soils[0]
    ys = [float(ifc.y_at(x)) for ifc in interfaces]
    if y_base >= ys[0]: return soils[0]
    if len(ys) == 1:    return soils[1]
    if y_base >= ys[1]: return soils[1]
    return soils[2]

def fs_on_arc(ground, interfaces, soils, method: str,
              xc: float, yc: float, R: float, x1: float, x2: float, n_slices: int):
    slices = make_slices_on_arc(ground, xc, yc, R, x1, x2, n_slices)
    if len(slices) < max(6, int(0.45*n_slices)):
        return None, None
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

# ---------------- Quality（高速化パラメータ含む） ----------------
QUALITY = {
    "Coarse": dict(
        quick_slices=10, final_slices=36, n_entries_final=900,
        probe_n_min_quick=81, limit_arcs_quick=80, show_k=80,
        budget_coarse_s=0.6, budget_quick_s=0.9, subsample_step=2
    ),
    "Normal": dict(
        quick_slices=12, final_slices=44, n_entries_final=1300,
        probe_n_min_quick=101, limit_arcs_quick=120, show_k=140,
        budget_coarse_s=0.8, budget_quick_s=1.2, subsample_step=2
    ),
    "Fine": dict(
        quick_slices=16, final_slices=56, n_entries_final=1700,
        probe_n_min_quick=121, limit_arcs_quick=160, show_k=200,
        budget_coarse_s=1.2, budget_quick_s=1.8, subsample_step=2
    ),
}

# ---------------- UI（ボタン押下まで計算しない） ----------------
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
        st.subheader("Center grid window")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14)
        ny = st.slider("ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = st.select_slider("Quality", options=list(QUALITY.keys()), value="Normal")
        with st.expander("Advanced (speed)", expanded=False):
            override = st.checkbox("Override Quality", value=False)
            quick_slices_in  = st.slider("Quick slices", 6, 40, QUALITY[quality]["quick_slices"], 1, disabled=not override)
            final_slices_in  = st.slider("Final slices", 20, 80, QUALITY[quality]["final_slices"], 2, disabled=not override)
            n_entries_final_in = st.slider("Final n_entries", 200, 4000, QUALITY[quality]["n_entries_final"], 100, disabled=not override)
            probe_min_q_in   = st.slider("Quick min probe", 41, 221, QUALITY[quality]["probe_n_min_quick"], 10, disabled=not override)
            limit_arcs_q_in  = st.slider("Quick max arcs/center", 20, 400, QUALITY[quality]["limit_arcs_quick"], 10, disabled=not override)
            show_k_in        = st.slider("Show top-k per center", 20, 400, QUALITY[quality]["show_k"], 10, disabled=not override)
            budget_coarse_in = st.slider("Coarse time budget (s)", 0.1, 5.0, QUALITY[quality]["budget_coarse_s"], 0.1, disabled=not override)
            budget_quick_in  = st.slider("Quick time budget (s)", 0.1, 5.0, QUALITY[quality]["budget_quick_s"], 0.1, disabled=not override)
            subsample_step_in= st.slider("Coarse subsample step", 1, 4, QUALITY[quality]["subsample_step"], 1, disabled=not override)

        st.subheader("Depth (vertical)")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 6.0, 0.5)

        st.subheader("Center picking")
        pick_mode = st.radio("Choose center by", ["Max arcs (robust)","Min Fs (aggressive)"], index=0)

        st.subheader("Display")
        show_minFs = st.checkbox("Show Min Fs", True)
        show_maxT  = st.checkbox("Show Max required T", True)
        show_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
        show_centers = st.checkbox("Show center-grid points", True)

    run = st.form_submit_button("▶ 計算開始（Entry-sweep FAST）")

# Quality適用
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in, show_k=show_k_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
        subsample_step=subsample_step_in,
    ))

# ---------------- Key ----------------
param_key = hash_params(dict(
    H=H, L=L, n_layers=n_layers,
    soils=[(s.gamma, s.c, s.phi) for s in soils],
    allow_cross=allow_cross, Fs_target=Fs_target,
    center=[x_min, x_max, y_min, y_max, nx, ny],
    method=method, quality=P, depth=[depth_min, depth_max],
    pick_mode=pick_mode,
))

# ---------------- 本計算（高速化版） ----------------
def compute_once():
    # センターグリッド
    xs_full = np.linspace(x_min, x_max, nx)
    ys_full = np.linspace(y_min, y_max, ny)
    centers_all = [(float(xc), float(yc)) for yc in ys_full for xc in xs_full]

    # ---- Coarse：部分サブサンプリング＋時間バジェットで最有望センターを決定
    step = max(1, int(P["subsample_step"]))
    xs = xs_full[::step] if step > 1 else xs_full
    ys = ys_full[::step] if step > 1 else ys_full
    centers_coarse = [(float(xc), float(yc)) for yc in ys for xc in xs]

    deadline = time.time() + float(P["budget_coarse_s"])
    best = None  # (score tuple, (xc,yc))
    for (xc,yc) in centers_coarse:
        cnt=0; Fs_best=None
        for x1,x2,R,Fs_q in arcs_from_center_by_entries_multi(
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
        # スコア：Max arcsなら (cnt, -Fs_best) を最大化、Min Fsなら (-Fs_best, cnt) を最大化
        if Fs_best is None: 
            continue
        if pick_mode.startswith("Max arcs"):
            score = (cnt, -(Fs_best))
        else:
            score = (-(Fs_best), cnt)
        if (best is None) or (score > best[0]):
            best = (score, (xc,yc))
        if time.time() > deadline:
            break

    if best is None:
        return dict(error="Coarseで有望センターを得られません。枠/Depth/Qualityを緩めてください。")
    xc, yc = best[1]

    # ---- Quick：選抜センターについて、時間バジェット内で候補弧（Fs低い順にTop-K保持）
    heap = []  # keep best K（Fsが小さいほど良い）→ (-Fs, x1, x2, R)
    deadline = time.time() + float(P["budget_quick_s"])
    for x1,x2,R,Fs_q in arcs_from_center_by_entries_multi(
        ground, soils, xc, yc,
        n_entries=P["n_entries_final"], method="Fellenius",
        depth_min=depth_min, depth_max=depth_max,
        interfaces=interfaces, allow_cross=allow_cross,
        quick_mode=True, n_slices_quick=P["quick_slices"],
        limit_arcs_per_center=P["limit_arcs_quick"],
        probe_n_min=P["probe_n_min_quick"],
    ):
        heapq.heappush(heap, (-float(Fs_q), float(x1), float(x2), float(R)))
        if len(heap) > int(P["show_k"]):
            heapq.heappop(heap)   # 最悪（Fsが大きい）を捨てる
        if time.time() > deadline:
            break

    if not heap:
        return dict(error="Quick候補が得られません。Depth/枠/Qualityを緩めてください。")

    # Fs昇順に並べ直し
    cand_arcs = [(-negFs, x1, x2, R) for (negFs, x1, x2, R) in heap]
    cand_arcs.sort(key=lambda t: t[0])  # Fs小→大

    # ---- Refine：選抜センターの候補を指定法で精密Fs
    refined=[]
    for Fs_q, x1, x2, R in cand_arcs:
        FS, D = fs_on_arc(ground, interfaces, soils, method, xc, yc, R, x1, x2, n_slices=P["final_slices"])
        if FS is None or D is None: 
            continue
        T_req = max(0.0, (Fs_target - FS) * D)
        refined.append(dict(R=R, x1=x1, x2=x2, Fs=FS, D=D, T_req=T_req, Fs_q=Fs_q))

    if not refined:
        return dict(error="Refineで有効な円弧が得られません。Final slicesを下げる/Depthを緩める等を試してください。")

    refined.sort(key=lambda d: d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    return dict(center=(xc,yc), refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_all)

# 実行
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res: st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs=res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]

# ---------------- 描画 ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))
Xd = np.linspace(0.0, L, 600); Yg = [ground.y_at(x) for x in Xd]

# 層の塗り（地表からはみ出さないよう clip）
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

# 地表線＋外周（囲い）
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.1)
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.1)
ax.plot([L, L],[0.0, ground.y_at(L)], linewidth=1.0)
ax.plot([0.0, L],[0.0, 0.0],         linewidth=1.0)
ax.plot([0.0, 0.0],[0.0, ground.y_at(0.0)], linewidth=1.0)

# センター表示
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# Refined弧（Fsカラー）
if show_refined:
    for d in refined:
        xs = np.linspace(d["x1"], d["x2"], 240)
        ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, linewidth=0.9, alpha=0.85, color=fs_to_color(d["Fs"]))

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
ax.set_xlim(min(-2.0, -0.05*L), max(1.20*L, 100.0))
ax.set_ylim(0.0, max(2.30*H, 100.0))
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
h,l = ax.get_legend_handles_labels()
ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)

ax.set_title(f"Entry-sweep FAST • Center=({xc:.2f},{yc:.2f}) • Method={method} • "
             f"MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_target:.2f}")

st.pyplot(fig, use_container_width=True); plt.close(fig)

# メトリクス
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（Refined）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
