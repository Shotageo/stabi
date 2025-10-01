# streamlit_app.py — 常時プレビュー＋手動実行（R-sweep堅牢化）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

# ---- 水位クラスは存在しない環境でも落ちないようにガード ----
WATER_AVAILABLE = True
try:
    from stabi_lem import WaterLine, WaterPolyline, WaterOffset
except Exception:
    WATER_AVAILABLE = False
    WaterLine = WaterPolyline = WaterOffset = None

st.set_page_config(page_title="Stabi LEM｜Preview first, compute on click", layout="wide")
st.title("Stabi LEM｜プレビュー常時表示 ＆ 計算はボタン押下時のみ")

# ---------------- Quality presets ----------------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, show_k=80,  R_quick=180, R_coarse=80, sample_N=1201),
    "Normal": dict(quick_slices=12, final_slices=40, show_k=120, R_quick=220, R_coarse=100, sample_N=1501),
    "Fine":   dict(quick_slices=16, final_slices=50, show_k=180, R_quick=280, R_coarse=140, sample_N=1801),
    "Very-fine": dict(quick_slices=20, final_slices=60, show_k=240, R_quick=360, R_coarse=180, sample_N=2201),
}

# ---------------- Utils ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def grid_points(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx); ys = np.linspace(y_min, y_max, ny)
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

# --- 円と地表の交点（堅牢フォールバック） ---
def circle_ground_intersections(xc: float, yc: float, R: float, ground: GroundPL, L: float, N: int=1201):
    if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(R)): return None
    if R <= 0: return None
    a = max(0.0, float(xc - R)); b = min(float(xc + R), float(L))
    if b - a <= 1e-6: return None
    N = int(max(401, N))
    xs = np.linspace(a, b, N)
    yg = np.array([float(ground.y_at(float(x))) for x in xs], dtype=float)
    inside = np.maximum(0.0, R*R - (xs - xc)**2)
    ycirc = yc - np.sqrt(inside)
    f = ycirc - yg
    s = np.sign(f)
    idx = np.where(s[:-1]*s[1:] <= 0)[0]
    if idx.size < 2: return None
    def root_at(i):
        x0, x1 = xs[i], xs[i+1]; f0, f1 = f[i], f[i+1]
        if abs(f1 - f0) < 1e-12: return float(0.5*(x0+x1))
        t = -f0/(f1-f0); return float(x0 + t*(x1 - x0))
    x_roots = [root_at(int(i)) for i in idx]
    x1, x2 = x_roots[0], x_roots[-1]
    if x2 - x1 < 1e-3: return None
    return (x1, x2)

# ---------------- Session defaults ----------------
for k,v in dict(geom_custom=False, ground_table=None, iface_tables={}).items():
    st.session_state.setdefault(k, v)

# ---------------- Controls（フォーム無し：プレビュー反映は即時、計算は手動ボタン） ----------------
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

    st.subheader("Water table (Phreatic)")
    if WATER_AVAILABLE:
        enable_w = st.checkbox("Enable water table", False)
        wsrc = st.radio("Source", ["Linear (2-point)", "CSV polyline", "Offset from ground"],
                        horizontal=True, disabled=not enable_w)
        gamma_w = st.number_input("γ_w (kN/m³)", 9.00, 10.50, 9.81, 0.01, disabled=not enable_w)
        pore_mode = st.radio("Pore handling", ["u-only (total W)", "buoyancy"],
                             horizontal=True, disabled=not enable_w)
        water_csv_file = None; yWL = yWR = None; w_offset=None
        if enable_w:
            if wsrc.startswith("Linear"):
                yWL = st.number_input("Water level at x=0 (m)", 0.0, 2.5*H, 0.6*H, 0.1)
                yWR = st.number_input("Water level at x=L (m)", 0.0, 2.5*H, 0.3*H, 0.1)
            elif "CSV" in wsrc:
                water_csv_file = st.file_uploader("Upload water polyline CSV (x,y columns)", type=["csv"])
            else:
                w_offset = st.number_input("Offset below ground (m)", 0.0, 50.0, 5.0, 0.5,
                                           help="地表面と同形状で、鉛直にこの距離だけ下げた水位線。")
    else:
        enable_w=False
        st.info("Water options are hidden because Water* classes are not available in stabi_lem.")

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
        R_quick_in       = st.slider("R-sweep (Quick) per center", 40, 600, QUALITY[quality]["R_quick"], 10, disabled=not override)
        R_coarse_in      = st.slider("R-sweep (Coarse) per center", 20, 300, QUALITY[quality]["R_coarse"], 10, disabled=not override)
        sampleN_in       = st.slider("Intersection sample N", 401, 4001, QUALITY[quality]["sample_N"], 200, disabled=not override)

    st.subheader("Depth range (vertical)")
    depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
    depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5,
                                help="円弧最深点の地表からの鉛直深さレンジ")

# ===== 確定ジオメトリ（プレビュー用。フォームじゃないので即時反映） =====
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

# ===== 水位オブジェクト（プレビューにも反映／ON時のみ） =====
water = None; water_kwargs={}
if WATER_AVAILABLE and enable_w:
    pore_tag = "buoyancy" if pore_mode.startswith("buoyancy") else "u-only"
    if 'wsrc' in locals():
        if wsrc.startswith("Linear"):
            water = WaterLine(0.0, L, float(yWL), float(yWR))
        elif "CSV" in wsrc and 'water_csv_file' in locals() and water_csv_file is not None:
            try:
                dfw = pd.read_csv(water_csv_file)
                xw = pd.to_numeric(dfw.iloc[:,0], errors="coerce").to_numpy()
                yw = pd.to_numeric(dfw.iloc[:,1], errors="coerce").to_numpy()
                m = np.isfinite(xw) & np.isfinite(yw)
                xw = xw[m]; yw = yw[m]
                order = np.argsort(xw)
                water = WaterPolyline(xw[order], yw[order])
            except Exception:
                water = None
        else:
            water = WaterOffset(float(w_offset))
    if water is not None:
        water_kwargs = dict(water=water, gamma_w=float(gamma_w), pore_mode=pore_tag)

# ===== プレビュー描画（計算前でも常に表示） =====
st.subheader("Preview（計算前ビュー）")
figp, axp = plt.subplots(figsize=(10.5, 6.0))
Xd = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

# 層塗り（クリップ）
if len(interfaces)==0:
    axp.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
elif len(interfaces)==1:
    Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
    axp.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    axp.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
else:
    Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
    axp.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
    axp.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
    axp.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

# 地表・境界・水位
axp.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if len(interfaces)>=1:
    axp.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if len(interfaces)>=2:
    axp.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")
if WATER_AVAILABLE and enable_w and water is not None:
    try:
        Yw = np.array([water.y_at(float(x), ground) for x in Xd], dtype=float)
        axp.plot(Xd, Yw, color="tab:blue", linestyle="--", linewidth=1.3, label="Water table")
        axp.fill_between(Xd, np.minimum(Yw, Yg), 0.0, color="tab:blue", alpha=0.06)
    except Exception:
        pass

# 外周
axp.plot([Xd[-1], Xd[-1]],[0.0, Yg[-1]], linewidth=1.0, color="k", alpha=0.7)
axp.plot([Xd[0],  Xd[-1]],[0.0, 0.0],     linewidth=1.0, color="k", alpha=0.7)
axp.plot([Xd[0],  Xd[0]], [0.0, Yg[0]],   linewidth=1.0, color="k", alpha=0.7)

# センターグリッド（プレビュー）
centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
axp.scatter([c[0] for c in centers_disp], [c[1] for c in centers_disp], s=12, c="k", alpha=0.25, marker=".", label="Center grid")

x_span = ground.X[-1]-ground.X[0]
axp.set_xlim(min(ground.X[0]-0.05*x_span, -2.0), ground.X[-1]+max(0.05*x_span, 2.0))
axp.set_ylim(0.0, max(2.30*H, 100.0))
axp.set_aspect("equal", adjustable="box")
axp.grid(True); axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)")
axp.legend(loc="upper right", fontsize=9)
st.pyplot(figp, use_container_width=True); plt.close(figp)

# ===== ここから“計算は手動ボタン” =====
compute_clicked = st.button("▶ 計算開始（手動）", type="primary")

# Quality展開
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        show_k=show_k_in, R_quick=R_quick_in, R_coarse=R_coarse_in,
        sample_N=sampleN_in,
    ))

def r_range_from_depth(xc: float, yc: float, dmin: float, dmax: float) -> tuple[float,float] | None:
    ys = float(ground.y_at(xc))
    R_low  = yc - (ys + dmax)  # 深い→小半径
    R_high = yc - (ys + dmin)  # 浅い→大半径
    if not np.isfinite(R_low) or not np.isfinite(R_high): return None
    R_high = float(R_high); R_low = float(max(R_low, 0.05))
    if R_high <= 0.0: return None
    if R_high - R_low < 1e-6: return None
    return R_low, R_high

def quick_candidates_by_Rs(xc, yc, R_list, quick_slices, method, allow_cross, interfaces, water_kwargs, L, sampleN):
    heap=[]; diag=dict(tried=len(R_list), sampled_hit=0, fallback_hit=0, fs_ok=0)
    for R in R_list:
        xpair=None
        s = arc_sample_poly_best_pair(ground, xc, yc, float(R), n=241)
        if s is not None:
            xpair=(float(s[0]), float(s[1])); diag["sampled_hit"] += 1
        else:
            s2 = circle_ground_intersections(xc, yc, float(R), ground, L, N=sampleN)
            if s2 is not None:
                xpair=(float(s2[0]), float(s2[1])); diag["fallback_hit"] += 1
        if xpair is None: continue
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method,
                              xc, yc, float(R), n_slices=int(quick_slices),
                              **water_kwargs)
        if Fs is None or not np.isfinite(Fs): continue
        diag["fs_ok"] += 1
        heapq.heappush(heap, (-float(Fs), float(R), xpair[0], xpair[1]))
    return heap, diag

def compute_once():
    # 1) Coarse：全格子を軽く走査し、「候補>0」の中心を必ず拾う
    best_center=None; best_score=None; tested_centers=[]
    xs = np.linspace(x_min, x_max, nx); ys = np.linspace(y_min, y_max, ny)
    for cy in ys:
        for cx in xs:
            rng = r_range_from_depth(cx, cy, depth_min, depth_max)
            if rng is None:
                tested_centers.append((cx,cy,0,None)); continue
            R_low, R_high = rng
            Rs = np.linspace(R_low, R_high, int(P["R_coarse"]))
            heap, diag = quick_candidates_by_Rs(cx, cy, Rs, max(8, P["quick_slices"]//2),
                                                "Fellenius", allow_cross, interfaces, water_kwargs,
                                                L, P["sample_N"])
            cnt=len(heap); Fs_min=(-heap[0][0]) if cnt>0 else None
            tested_centers.append((cx,cy,cnt,Fs_min))
            if cnt>0:
                score = (cnt, -Fs_min)
                if (best_score is None) or (score > best_score):
                    best_score=score; best_center=(cx,cy)

    if best_center is None:
        return dict(error="Coarseで候補なし。センター/深さ/水位/層跨ぎの整合を確認してください。")

    xc, yc = best_center
    rng = r_range_from_depth(xc, yc, depth_min, depth_max)
    if rng is None:
        return dict(error="選抜センターで半径レンジが成立しません。深さレンジ/センター枠を調整してください。")
    R_low, R_high = rng
    Rs = np.linspace(R_low, R_high, int(P["R_quick"]))
    heap, diag_q = quick_candidates_by_Rs(xc, yc, Rs, P["quick_slices"], method, allow_cross, interfaces, water_kwargs,
                                          L, P["sample_N"])
    if not heap:
        return dict(error="Quickで円弧候補なし（交点検出の両系とも不成立）。")

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
        return dict(error="Refineで有効弧なし。設定/Quality/水位を再確認してください。")

    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    return dict(center=(xc,yc), tested=tested_centers, refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT, centers_disp=centers_disp,
                diag=dict(R_range=(float(R_low), float(R_high)), Quick_diag=diag_q))

# ===== 実行／結果表示 =====
if compute_clicked:
    res = compute_once()
    if "error" in res: st.error(res["error"])
    else:
        st.session_state["res"] = res

if "res" in st.session_state:
    st.subheader("Result（計算結果）")
    res = st.session_state["res"]
    if "error" in res:
        st.error(res["error"])
    else:
        xc,yc = res["center"]
        refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
        # 描画
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        # 地形はプレビューと同様に
        ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
        if len(interfaces)==0:
            ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
        elif len(interfaces)==1:
            Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
            ax.plot(Xd, Y1, linestyle="--", linewidth=1.2, label="Interface 1")
        else:
            Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
            ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
            ax.plot(Xd, Y1, linestyle="--", linewidth=1.2, label="Interface 1")
            ax.plot(Xd, Y2, linestyle="--", linewidth=1.2, label="Interface 2")
        if WATER_AVAILABLE and enable_w and water is not None:
            try:
                Yw = np.array([water.y_at(float(x), ground) for x in Xd], dtype=float)
                ax.plot(Xd, Yw, color="tab:blue", linestyle="--", linewidth=1.3, label="Water table")
                ax.fill_between(Xd, np.minimum(Yw, Yg), 0.0, color="tab:blue", alpha=0.06)
            except Exception:
                pass
        # 外周
        ax.plot([Xd[-1], Xd[-1]],[0.0, Yg[-1]], linewidth=1.0, color="k", alpha=0.7)
        ax.plot([Xd[0],  Xd[-1]],[0.0, 0.0],     linewidth=1.0, color="k", alpha=0.7)
        ax.plot([Xd[0],  Xd[0]], [0.0, Yg[0]],   linewidth=1.0, color="k", alpha=0.7)
        # グリッド/選抜センター
        ax.scatter([c[0] for c in res["centers_disp"]], [c[1] for c in res["centers_disp"]], s=12, c="k", alpha=0.25, marker=".", label="Center grid")
        ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")
        # 全精算弧
        for d in refined:
            xs=np.linspace(d["x1"], d["x2"], 240)
            ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=0.9, alpha=0.80, color=fs_to_color(d["Fs"]))
        # 特記事項
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
        # 軸
        x_span = ground.X[-1]-ground.X[0]
        ax.set_xlim(min(ground.X[0]-0.05*x_span, -2.0), ground.X[-1]+max(0.05*x_span, 2.0))
        ax.set_ylim(0.0, max(2.30*H, 100.0))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        from matplotlib.patches import Patch
        legend_patches=[Patch(color=(0.85,0,0),label="Fs<1.0"),
                        Patch(color=(1.0,0.75,0.0),label="1.0≤Fs<1.2"),
                        Patch(color=(0.0,0.55,0.0),label="Fs≥1.2")]
        h,l = ax.get_legend_handles_labels()
        ax.legend(h+legend_patches, l+[p.get_label() for p in legend_patches], loc="upper right", fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # 診断
        st.divider()
        st.subheader("Diagnostics")
        st.json(res.get("diag", {}), expanded=False)
else:
    st.caption("上のプレビューで条件を確認 →『▶ 計算開始（手動）』を押すと下に結果が出ます。")
