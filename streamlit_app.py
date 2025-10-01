# streamlit_app.py — フルスパン採用版（Fs/Dともに最外交点どうしで一貫評価）
from __future__ import annotations
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq, time, json, hashlib, random
from dataclasses import dataclass

# 既存コアはそのまま利用（候補Rの抽出など）
from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
)

st.set_page_config(page_title="Stabi LEM｜Full-span mode", layout="wide")
st.title("Stabi LEM｜Full-span（最外交点）での一貫評価")

# ---------------- ユーティリティ ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# ---- 円と折れ線（地表）の交点を全て取り出し、最小x/最大xをフルスパン端点にする ----
def circle_polyline_full_span(ground: GroundPL, xc: float, yc: float, R: float):
    X = np.asarray(ground.X); Y = np.asarray(ground.Y)
    hits = []
    for i in range(len(X) - 1):
        x0, y0 = X[i],   Y[i]
        x1, y1 = X[i+1], Y[i+1]
        vx, vy = x1 - x0, y1 - y0
        # 線分上の点 P(t) = (x0, y0) + t*(vx,vy), t∈[0,1] と円の交点 |P-C|=R
        # (vx^2+vy^2) t^2 + 2[(x0-xc)vx + (y0-yc)vy] t + [(x0-xc)^2+(y0-yc)^2 - R^2] = 0
        A = vx*vx + vy*vy
        B = 2*((x0 - xc)*vx + (y0 - yc)*vy)
        C = (x0 - xc)**2 + (y0 - yc)**2 - R*R
        if A <= 0: 
            continue
        D = B*B - 4*A*C
        if D < 0:
            continue
        sqrtD = np.sqrt(D)
        for sign in (-1.0, 1.0):
            t = (-B + sign*sqrtD) / (2*A)
            if 0.0 <= t <= 1.0:
                xi = x0 + t*vx
                yi = y0 + t*vy
                hits.append((float(xi), float(yi)))
    if len(hits) < 2:
        return None
    # x座標で最外の2点（min/max）を採用
    xs = [p[0] for p in hits]
    x1 = min(xs); x2 = max(xs)
    # 端点yは地表線上にあるが、安定のため再評価
    y1 = float(ground.y_at(x1)); y2 = float(ground.y_at(x2))
    return (x1, y1, x2, y2)

# ---- 指定した区間 [x1,x2] を“フルスパン”としてスライスし、Fellenius/Bishop でFsを計算 ----
@dataclass
class SliceGeom:
    xL: float; xR: float; xC: float
    y_top: float; y_base: float
    h: float; dx: float
    sin_a: float; cos_a: float
    b: float  # 基底長

def make_slices_fullspan(ground: GroundPL, xc: float, yc: float, R: float, x1: float, x2: float, n_slices: int):
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
        # 円弧の接線角 α：sinα = (xC - xc)/R, cosα = +sqrt(1 - sin^2)
        sin_a = (xC - xc)/R
        cos_a = float(np.sqrt(max(1e-12, 1.0 - sin_a*sin_a)))
        b = dx / max(1e-12, cos_a)  # 基底長
        slices.append(SliceGeom(xL, xR, xC, y_top, y_base, h, dx, sin_a, cos_a, b))
    return slices

# ---- 多層の土質パラメータをスライス底面の高さで引き当て（上から順番） ----
def soil_at_base(x: float, y_base: float, ground: GroundPL, interfaces: list[GroundPL], soils: list[Soil]):
    # interfaces は上から順に渡されている前提（Interface1の下がLayer2…）
    # xの位置における境界高をクリップして照会
    if not interfaces:
        return soils[0]
    # 1本 → 上: Layer1（地表〜I1）、下: Layer2（I1〜0）
    # 2本 → 上: L1（地表〜I1）, 中: L2（I1〜I2）, 下: L3（I2〜0）
    # 高い方から判定
    ys = [float(ifc.y_at(x)) for ifc in interfaces]  # I1, I2, ...
    if y_base >= ys[0]:
        return soils[0]
    if len(interfaces) == 1:
        return soils[1]
    # 2本以上
    if y_base >= ys[1]:
        return soils[1]
    return soils[2]

# ---- フルスパンでの Fs と 駆動項D（=ΣW sinα）を計算 ----
def fs_fullspan(ground, interfaces, soils, method: str,
                xc: float, yc: float, R: float, x1: float, x2: float, n_slices: int):
    slices = make_slices_fullspan(ground, xc, yc, R, x1, x2, n_slices)
    if len(slices) < max(6, int(0.4*n_slices)):
        return None, None  # スライスが足りない → 無効
    # 各スライスのW, c, φ を評価
    Ws=[]; sinA=[]; cosA=[]; bs=[]; cs=[]; tans=[]
    for s in slices:
        soil = soil_at_base(s.xC, s.y_base, ground, interfaces, soils)
        gamma = soil.gamma; c = soil.c; phi = soil.phi
        W = gamma * s.h * s.dx  # 単位奥行き 1m
        Ws.append(W); sinA.append(s.sin_a); cosA.append(s.cos_a); bs.append(s.b)
        cs.append(c); tans.append(np.tan(np.deg2rad(phi)))
    Ws = np.array(Ws); sinA = np.array(sinA); cosA = np.array(cosA)
    bs = np.array(bs); cs = np.array(cs); tans = np.array(tans)

    D = float(np.sum(Ws * sinA))  # 駆動項 = ΣW sinα

    if method.startswith("Fellenius"):
        # Fellenius (Ordinary) — 有効応力uは今は未導入（別パッチで水位/Ru法を入れる）
        Rnum = np.sum(cs*bs + (Ws * cosA) * tans)
        FS = float(Rnum / max(1e-12, D))
        return FS, D

    # Bishop (simplified) — 反復解
    FS = 1.20  # 初期値
    for _ in range(50):
        m = 1.0 + (tans * np.sin(np.arcsin(sinA))) * (sinA / np.maximum(1e-12, FS))  # tanφ * tanα / FS
        # tanα = sinα/cosα
        m = 1.0 + (tans * (sinA/np.maximum(1e-12, cosA))) / np.maximum(1e-12, FS)
        Rnum = np.sum((cs*bs + (Ws * cosA) * tans) / np.maximum(1e-12, m))
        FS_new = float(Rnum / max(1e-12, D))
        if abs(FS_new - FS) < 1e-4:
            FS = FS_new; break
        FS = FS_new
    return FS, D

# ---------------- Quality プリセット ----------------
QUALITY = {
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1200,
                   probe_n_min_quick=101, limit_arcs_quick=120,
                   budget_coarse_s=0.8, budget_quick_s=1.2,
                   show_k=120, audit_limit_per_center=12, audit_budget_s=2.8),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1600,
                 probe_n_min_quick=121, limit_arcs_quick=160,
                 budget_coarse_s=1.2, budget_quick_s=1.8,
                 show_k=180, audit_limit_per_center=16, audit_budget_s=3.2),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200,
                      probe_n_min_quick=141, limit_arcs_quick=220,
                      budget_coarse_s=2.6, budget_quick_s=2.6,
                      show_k=240, audit_limit_per_center=20, audit_budget_s=4.0),
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=900,
                   probe_n_min_quick=81, limit_arcs_quick=80,
                   budget_coarse_s=0.6, budget_quick_s=0.9,
                   show_k=60, audit_limit_per_center=10, audit_budget_s=2.0),
}

# ---------------- 入力フォーム ----------------
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
        soils = []
        gamma1 = st.number_input("γ₁ (kN/m³)", 10.0, 25.0, 18.0, 0.5)
        c1     = st.number_input("c₁ (kPa)",    0.0, 200.0, 5.0, 0.5)
        phi1   = st.number_input("φ₁ (deg)",    0.0, 45.0, 30.0, 0.5)
        soils.append(Soil(gamma=gamma1, c=c1, phi=phi1))
        if n_layers >= 2:
            gamma2 = st.number_input("γ₂ (kN/m³)", 10.0, 25.0, 19.0, 0.5)
            c2     = st.number_input("c₂ (kPa)",    0.0, 200.0, 8.0, 0.5)
            phi2   = st.number_input("φ₂ (deg)",    0.0, 45.0, 28.0, 0.5)
            soils.append(Soil(gamma=gamma2, c=c2, phi=phi2))
        if n_layers >= 3:
            gamma3 = st.number_input("γ₃ (kN/m³)", 10.0, 25.0, 20.0, 0.5)
            c3     = st.number_input("c₃ (kPa)",    0.0, 200.0, 12.0, 0.5)
            phi3   = st.number_input("φ₃ (deg)",    0.0, 45.0, 25.0, 0.5)
            soils.append(Soil(gamma=gamma3, c=c3, phi=phi3))

        st.subheader("Crossing control（下層進入可否）")
        allow_cross=[]
        if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
        if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

        st.subheader("Target safety")
        Fs_target = st.number_input("Target FS (for T_req)", 1.00, 2.00, 1.20, 0.05)

    with B:
        st.subheader("Center grid")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14)
        ny = st.slider("ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
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

        st.subheader("Display pick")
        show_minFs = st.checkbox("Show Min Fs", True)
        show_maxT  = st.checkbox("Show Max required T", True)
        show_all_refined = st.checkbox("Show refined arcs (Fs-colored)", True)
        show_centers = st.checkbox("Show center-grid points", True)
        audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)

    run = st.form_submit_button("▶ 計算開始（Full-span）")

# Quality適用
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# キー
param_key = hash_params(dict(
    H=H, L=L, n_layers=n_layers,
    soils=[(s.gamma, s.c, s.phi) for s in soils],
    allow_cross=allow_cross, Fs_target=Fs_target,
    center=[x_min, x_max, y_min, y_max, nx, ny],
    method=method, quality=P, depth=[depth_min, depth_max],
))

# ---------------- 計算本体（フルスパンでFs/D評価） ----------------
def compute_once():
    # 1) Coarse: 最有望センター（サブサンプル＋打切り）→ ここは簡略、選抜のみ
    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        # 2ステップ間引き（負荷軽減）
        xs = xs[::2] if len(xs)>1 else xs
        ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(budget_s):
        deadline = time.time() + budget_s
        best=None; tested=[]
        for (xc,yc) in subsampled_centers():
            cnt=0; Fs_best=None
            for _x1,_x2,_R,Fs_quick in arcs_from_center_by_entries_multi(
                ground, soils, xc, yc,
                n_entries=min(800, P["n_entries_final"]), method="Fellenius",
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                limit_arcs_per_center=min(60, P["limit_arcs_quick"]),
                probe_n_min=max(61, P["probe_n_min_quick"]-40),
            ):
                cnt += 1
                if (Fs_best is None) or (Fs_quick < Fs_best): Fs_best = Fs_quick
                if time.time() > deadline: break
            tested.append((xc,yc))
            score = (cnt, - (Fs_best if Fs_best is not None else 1e9))
            if (best is None) or (score > best[0]): best = (score, (xc,yc))
            if time.time() > deadline: break
        return (best[1] if best else None), tested

    with st.spinner("Coarse（最有望センター選抜）"):
        center, tested = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseで候補が見つかりません。枠/深さを広げてください。")
    xc, yc = center

    # 2) Quick：選抜センターでR候補抽出
    with st.spinner("Quick（R候補抽出）"):
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs_quick in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=depth_min, depth_max=depth_max,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            # Quick段の x1,x2 は使わず、Refineでフルスパンを決め直す
            heapq.heappush(heap_R, (-Fs_quick, R))
            if len(heap_R) > P["show_k"]: heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _neg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。条件を緩和してください。")

    # 3) Refine：各Rについてフルスパン端点を求め、同一区間で Fs と D を評価
    refined=[]
    for R in R_candidates:
        span = circle_polyline_full_span(ground, xc, yc, R)
        if span is None: 
            continue
        x1, y1, x2, y2 = span
        # 端点の鉛直深さチェック（レンジ外なら棄却）
        depth1 = y1 - (yc - np.sqrt(max(1e-12, R*R - (x1-xc)**2)))
        depth2 = y2 - (yc - np.sqrt(max(1e-12, R*R - (x2-xc)**2)))
        max_depth_on_span = max(depth1, depth2)
        min_depth_on_span = min(depth1, depth2)
        # 端点だけでなくスパン中央近辺も含めた深さをざっくり見る
        xm = 0.5*(x1+x2); ym_surf = float(ground.y_at(xm))
        ym_base = yc - np.sqrt(max(1e-12, R*R - (xm-xc)**2))
        max_depth_on_span = max(max_depth_on_span, ym_surf - ym_base)
        if (max_depth_on_span < depth_min) or (max_depth_on_span > depth_max):
            continue

        FS, D = fs_fullspan(ground, interfaces, soils, method, xc, yc, R, x1, x2, n_slices=P["final_slices"])
        if FS is None:
            continue
        T_req = max(0.0, (Fs_target - FS) * D)
        refined.append(dict(R=float(R), x1=float(x1), x2=float(x2), Fs=float(FS), D=float(D), T_req=float(T_req)))

    if not refined:
        return dict(error="Refineで有効弧が得られません（フルスパン）。深さ/枠/Qualityを調整してください。")

    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    # 監査用：可視化グリッド（センター点表示）
    centers_disp = [(float(xc), float(yc)) for yc in np.linspace(y_min, y_max, ny)
                                      for xc in np.linspace(x_min, x_max, nx)]
    return dict(center=(xc,yc), tested_centers=tested, refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp)

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

# ---------------- 描画（フルスパン端点で弧を描画） ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))
Xd = np.linspace(0.0, L, 600); Yg = [ground.y_at(x) for x in Xd]

# 層塗り
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

# センター表示
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# 全Refined弧（フルスパン端点で描画）
if show_all_refined:
    for d in refined:
        xs = np.linspace(d["x1"], d["x2"], 240)
        ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

# ピックアップ表示
if show_minFs and 0 <= idx_minFs < len(refined):
    d = refined[idx_minFs]
    xs = np.linspace(d["x1"], d["x2"], 500)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    # 半径線
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

# 軸など
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

ax.set_title(f"Full-span mode • Center=({xc:.2f},{yc:.2f}) • Method={method} • "
             f"MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_target:.2f}")

st.pyplot(fig, use_container_width=True); plt.close(fig)

# メトリクス
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（Full-span, 精密）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
