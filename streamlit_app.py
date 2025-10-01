# streamlit_app.py — R-sweep主導（ユーザー条件厳守）版
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,  # 互換のため残すが本版は使わない
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
    # 水位クラス（stabi_lem側にある前提）
    WaterLine, WaterPolyline, WaterOffset,
)

st.set_page_config(page_title="Stabi LEM｜R-sweep candidates", layout="wide")
st.title("Stabi LEM｜Rスイープ主導（条件厳守）")

# ---------------- Quality presets ----------------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, show_k=60, budget_coarse_s=0.6,
                   coarse_subsample="every 2nd", R_quick=120, R_coarse=60),
    "Normal": dict(quick_slices=12, final_slices=40, show_k=120, budget_coarse_s=0.9,
                   coarse_subsample="every 2nd", R_quick=180, R_coarse=80),
    "Fine":   dict(quick_slices=16, final_slices=50, show_k=180, budget_coarse_s=1.3,
                   coarse_subsample="full",    R_quick=240, R_coarse=100),
    "Very-fine": dict(quick_slices=20, final_slices=60, show_k=240, budget_coarse_s=1.8,
                   coarse_subsample="full",    R_quick=300, R_coarse=140),
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

def _mk_table_from_xy(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": np.asarray(X, dtype=float), "y": np.asarray(Y, dtype=float)})

def _xy_from_table(df: pd.DataFrame, L: float) -> tuple[np.ndarray, np.ndarray] | None:
    if df is None or not isinstance(df, pd.DataFrame): return None
    if "x" not in df.columns or "y" not in df.columns: return None
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2: return None
    x = np.clip(x, 0.0, L)
    order = np.argsort(x)
    return x[order], y[order]

# ---------------- Session defaults ----------------
if "geom_custom" not in st.session_state:
    st.session_state["geom_custom"] = False
if "ground_table" not in st.session_state:
    st.session_state["ground_table"] = None
if "iface_tables" not in st.session_state:
    st.session_state["iface_tables"] = {}

# ---------------- Inputs (FORM) ----------------
with st.form("params"):
    A, B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)
        ground_default = make_ground_example(H, L)

        st.checkbox("Edit geometry in tables (ground / interfaces)", value=st.session_state["geom_custom"], key="geom_custom")
        if st.session_state["geom_custom"]:
            if st.session_state["ground_table"] is None:
                st.session_state["ground_table"] = _mk_table_from_xy(ground_default.X, ground_default.Y)
            st.caption("Ground surface")
            st.data_editor(st.session_state["ground_table"], num_rows="dynamic", use_container_width=True, key="ground_tbl_edit")

            n_layers_edit = st.selectbox("Number of layers (tables)", [0,1,2], index=0,
                                         help="ここでの値は表編集用。下の『Layers』とは独立。")
            iface_tables_local = st.session_state["iface_tables"]
            for k in range(n_layers_edit):
                name = f"Interface {k+1}"
                if name not in iface_tables_local:
                    if k == 0: xi, yi = make_interface1_example(H, L)
                    else:      xi, yi = make_interface2_example(H, L)
                    iface_tables_local[name] = _mk_table_from_xy(xi, yi)
                st.caption(name)
                st.data_editor(iface_tables_local[name], num_rows="dynamic", use_container_width=True, key=f"iface_tbl_{k}")

        st.subheader("Layers (for calculation)")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)

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

        # ---- Water: デフォルトOFF（ON時のみ計算へ） ----
        st.subheader("Water table (Phreatic)")
        enable_w = st.checkbox("Enable water table", False)
        wsrc = st.radio("Source", ["Linear (2-point)", "CSV polyline", "Offset from ground"],
                        horizontal=True, disabled=not enable_w)
        gamma_w = st.number_input("γ_w (kN/m³)", 9.00, 10.50, 9.81, 0.01, disabled=not enable_w)
        pore_mode = st.radio("Pore handling", ["u-only (total W)", "buoyancy"],
                             horizontal=True, disabled=not enable_w)

        water_csv_file = None
        yWL = yWR = None
        w_offset = None
        if enable_w:
            if wsrc.startswith("Linear"):
                yWL = st.number_input("Water level at x=0 (m)", 0.0, 2.5*H, 0.6*H, 0.1)
                yWR = st.number_input("Water level at x=L (m)", 0.0, 2.5*H, 0.3*H, 0.1)
            elif "CSV" in wsrc:
                water_csv_file = st.file_uploader("Upload water polyline CSV (x,y columns)", type=["csv"])
            else:
                w_offset = st.number_input("Offset below ground (m)", 0.0, 50.0, 5.0, 0.5,
                                           help="地表面と同形状で、鉛直にこの距離だけ下げた水位線。")

    with B:
        st.subheader("Center grid")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14); ny = st.slider("ny", 4, 40, 9)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = st.select_slider("Quality", options=list(QUALITY.keys()), value="Normal")
        with st.expander("Advanced", expanded=False):
            override = st.checkbox("Override Quality", value=False)
            quick_slices_in  = st.slider("Quick slices", 6, 40, QUALITY[quality]["quick_slices"], 1, disabled=not override)
            final_slices_in  = st.slider("Final slices", 20, 80, QUALITY[quality]["final_slices"], 2, disabled=not override)
            show_k_in        = st.slider("Refine top-K", 20, 400, QUALITY[quality]["show_k"], 10, disabled=not override)
            R_quick_in       = st.slider("R-sweep (Quick) per center", 40, 400, QUALITY[quality]["R_quick"], 10, disabled=not override)
            R_coarse_in      = st.slider("R-sweep (Coarse) per center", 20, 200, QUALITY[quality]["R_coarse"], 10, disabled=not override)
            budget_coarse_in = st.slider("Budget Coarse (s)", 0.1, 5.0, QUALITY[quality]["budget_coarse_s"], 0.1, disabled=not override)

        st.subheader("Depth range (vertical)")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5,
                                    help="円弧最深点の地表からの鉛直深さレンジ")

    run = st.form_submit_button("▶ 計算開始")

# ===== Submit 後の確定ジオメトリ =====
ground_default = make_ground_example(H, L)
if st.session_state.get("geom_custom", False):
    if 'ground_tbl_edit' in st.session_state:
        st.session_state["ground_table"] = st.session_state["ground_tbl_edit"]
    gxy = _xy_from_table(st.session_state.get("ground_tbl_edit", st.session_state.get("ground_table")), L)
    ground = GroundPL(*gxy) if gxy is not None else ground_default

    iface_tables_local = st.session_state.get("iface_tables", {})
    interfaces = []
    if n_layers >= 2:
        df1 = st.session_state.get("iface_tbl_0", iface_tables_local.get("Interface 1"))
        xy1 = _xy_from_table(df1, L)
        interfaces.append(xy1 if xy1 is not None else make_interface1_example(H, L))
    if n_layers >= 3:
        df2 = st.session_state.get("iface_tbl_1", iface_tables_local.get("Interface 2"))
        xy2 = _xy_from_table(df2, L)
        interfaces.append(xy2 if xy2 is not None else make_interface2_example(H, L))
else:
    ground = ground_default
    interfaces = []
    if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

# ===== 水位オブジェクト（ON時のみ） =====
water = None
water_key = None
water_kwargs = {}
if enable_w:
    pore_tag = "buoyancy" if pore_mode.startswith("buoyancy") else "u-only"
    if wsrc.startswith("Linear"):
        water = WaterLine(0.0, L, float(yWL), float(yWR))
        water_key=("linear", round(float(yWL),3), round(float(yWR),3))
    elif "CSV" in wsrc and water_csv_file is not None:
        try:
            dfw = pd.read_csv(water_csv_file)
            xw = pd.to_numeric(dfw.iloc[:,0], errors="coerce").to_numpy()
            yw = pd.to_numeric(dfw.iloc[:,1], errors="coerce").to_numpy()
            m = np.isfinite(xw) & np.isfinite(yw)
            xw = xw[m]; yw = yw[m]
            order = np.argsort(xw)
            water = WaterPolyline(xw[order], yw[order]); water_key=("csv", len(xw))
        except Exception:
            water = None; water_key=None
    else:
        water = WaterOffset(float(w_offset)); water_key=("offset", round(float(w_offset),3))
    if water is not None:
        water_kwargs = dict(water=water, gamma_w=float(gamma_w), pore_mode=pore_tag)

# -------- Quality expand --------
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        show_k=show_k_in, R_quick=R_quick_in, R_coarse=R_coarse_in,
        budget_coarse_s=budget_coarse_in,
    ))

# -------- Param key --------
def param_pack():
    return dict(
        H=H, L=L, n_layers=n_layers,
        soils=[(s.gamma, s.c, s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=Fs_target,
        center=[x_min, x_max, y_min, y_max, nx, ny],
        method=method, quality=P, depth=[depth_min, depth_max],
        water_enabled=bool(enable_w), water_key=water_key,
        ghash=hash_params({"gx":ground.X.tolist(),"gy":ground.Y.tolist(),
                           "if": [ (np.asarray(ix).tolist(), np.asarray(iy).tolist()) for (ix,iy) in interfaces ]})
    )
param_key = hash_params(param_pack())

# ===== R-sweep：センター毎に半径レンジを深さから直決め =====
def r_range_from_depth(xc: float, yc: float, dmin: float, dmax: float) -> tuple[float,float] | None:
    ys = float(ground.y_at(xc))
    R_low  = yc - (ys + dmax)  # 大きい深さ → 小さい半径
    R_high = yc - (ys + dmin)  # 小さい深さ → 大きい半径
    R_low  = float(R_low); R_high = float(R_high)
    # 半径は正・かつレンジ幅あり
    if not np.isfinite(R_low) or not np.isfinite(R_high): return None
    if R_high <= 0.0: return None
    R_low = max(R_low, 0.05)  # 最小半径の下駄
    if R_high - R_low < 1e-6: return None
    return R_low, R_high

def quick_candidates_by_Rs(xc, yc, R_list, quick_slices, method, allow_cross, interfaces, water_kwargs):
    # Rごとに：交点を取り（無ければskip）→ Quick分割でFs評価
    heap=[]  # (-Fs, R, x1, x2)
    for R in R_list:
        s = arc_sample_poly_best_pair(ground, xc, yc, float(R), n=241)
        if s is None: continue
        x1, x2 = float(s[0]), float(s[1])
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method,
                              xc, yc, float(R), n_slices=int(quick_slices),
                              **water_kwargs)
        if Fs is None or not np.isfinite(Fs): continue
        heapq.heappush(heap, (-float(Fs), float(R), x1, x2))
    return heap

# ===== 計算 =====
def compute_once():
    # 1) Coarse：格子のサブサンプルで「候補が出る」センターを必ず拾う
    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx); ys = np.linspace(y_min, y_max, ny)
        tag = P["coarse_subsample"]
        if tag == "every 2nd":
            xs = xs[::2] if len(xs)>1 else xs; ys = ys[::2] if len(ys)>1 else ys
        elif tag == "full":
            pass
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    best_center=None; best_score=None
    tested_centers=[]
    deadline = time.time() + float(P["budget_coarse_s"])
    for (xc,yc) in subsampled_centers():
        if time.time() > deadline: break
        rng = r_range_from_depth(xc, yc, depth_min, depth_max)
        if rng is None:
            tested_centers.append((xc,yc,0,None))
            continue
        R_low, R_high = rng
        Rs = np.linspace(R_low, R_high, int(P["R_coarse"]))
        heap = quick_candidates_by_Rs(xc, yc, Rs, max(8, P["quick_slices"]//2),
                                      "Fellenius", allow_cross, interfaces, water_kwargs)
        cnt = len(heap)
        Fs_min = (-heap[0][0]) if cnt>0 else None
        tested_centers.append((xc,yc,cnt,Fs_min))
        if cnt>0:
            score = (cnt, -Fs_min)  # 多く・かつFsが小さいほど良い
            if (best_score is None) or (score > best_score):
                best_score = score; best_center=(xc,yc)

    if best_center is None:
        return dict(error="Coarseで候補なし。センター/深さ/水位を見直してください。")

    xc, yc = best_center

    # 2) Quick at chosen center：より細かいRスイープ
    rng = r_range_from_depth(xc, yc, depth_min, depth_max)
    if rng is None:
        return dict(error="選抜センターで半径レンジが成立しません。深さレンジまたはセンター枠を調整してください。")
    R_low, R_high = rng
    Rs = np.linspace(R_low, R_high, int(P["R_quick"]))
    heap = quick_candidates_by_Rs(xc, yc, Rs, P["quick_slices"], method, allow_cross, interfaces, water_kwargs)
    if not heap:
        return dict(error="Quickで円弧候補なし。条件が矛盾している可能性があります（深さ/水位/層跨ぎ）。")

    # 3) Refine：上位K（Fs小）を最終分割で精算
    topK = int(P["show_k"])
    refined=[]
    for _ in range(min(topK, len(heap))):
        fsneg, R, x1, x2 = heapq.heappop(heap)
        Fs_q = -fsneg
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method,
                              xc, yc, R, n_slices=P["final_slices"], **water_kwargs)
        if Fs is None or not np.isfinite(Fs): continue
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross,
                                        xc, yc, R, n_slices=P["final_slices"])
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (Fs_target - Fs)*D_sum)
        refined.append(dict(Fs=float(Fs), Fs_q=float(Fs_q), R=float(R),
                            x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Quality/水位を見直してください。")

    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    return dict(center=(xc,yc), tested=tested_centers, refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT, centers_disp=centers_disp)

# ---- Run or cache ----
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res:
        st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))
Xd = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

# layers fill（境界を地表にクリップ）
if len(interfaces)==0:
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif len(interfaces)==1:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
    ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

# ground & interfaces
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if len(interfaces)>=1:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if len(interfaces)>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")

# water（ON時のみ）
if enable_w and water is not None:
    try:
        Yw = np.array([water.y_at(float(x), ground) for x in Xd], dtype=float)
        ax.plot(Xd, Yw, color="tab:blue", linestyle="--", linewidth=1.3, label="Water table")
        ax.fill_between(Xd, np.minimum(Yw, Yg), 0.0, color="tab:blue", alpha=0.06)
    except Exception:
        pass

# 外周を閉じる
ax.plot([Xd[-1], Xd[-1]],[0.0, Yg[-1]], linewidth=1.0, color="k", alpha=0.7)
ax.plot([Xd[0],  Xd[-1]],[0.0, 0.0],     linewidth=1.0, color="k", alpha=0.7)
ax.plot([Xd[0],  Xd[0]], [0.0, Yg[0]],   linewidth=1.0, color="k", alpha=0.7)

# center-grid
xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# refined arcs（Fs色）
for d in refined:
    xs=np.linspace(d["x1"], d["x2"], 240)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=0.9, alpha=0.80, color=fs_to_color(d["Fs"]))

# pick-ups
if 0<=idx_minFs<len(refined):
    d=refined[idx_minFs]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
    y1=float(ground.y_at(d["x1"])); y2=float(ground.y_at(d["x2"]))
    ax.plot([xc,d["x1"]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
    ax.plot([xc,d["x2"]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if 0<=idx_maxT<len(refined):
    d=refined[idx_maxT]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
            label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
    y1=float(ground.y_at(d["x1"])); y2=float(ground.y_at(d["x2"]))
    ax.plot([xc,d["x1"]],[yc,y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
    ax.plot([xc,d["x2"]],[yc,y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

# axis & legend
x_span = ground.X[-1]-ground.X[0]
ax.set_xlim(min(ground.X[0]-0.05*x_span, -2.0), ground.X[-1]+max(0.05*x_span, 2.0))
ax.set_ylim(0.0, max(2.30*H, y_max + 0.05*H, 100.0))
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

from matplotlib.patches import Patch
legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
h,l = ax.get_legend_handles_labels()
ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)

title_tail=[f"MinFs={refined[idx_minFs]['Fs']:.3f}", f"TargetFs={Fs_target:.2f}"]
if enable_w and water is not None: title_tail.append("Water=ON")
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))
st.pyplot(fig, use_container_width=True); plt.close(fig)
