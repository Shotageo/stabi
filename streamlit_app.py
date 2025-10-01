# streamlit_app.py — 水位オフセット・表編集・救済パス付き
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json, random
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
    WaterLine, WaterPolyline, WaterOffset,
)

st.set_page_config(page_title="Stabi LEM｜水位/表編集/救済", layout="wide")
st.title("Stabi LEM｜水位（オフセット/CSV/直線）・ジオメトリエディタ・救済パス")

# ---------------- Quality presets ----------------
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
    xs = np.linspace(x_min, x_max, nx); ys = np.linspace(y_min, y_max, ny)
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

def _mk_table_from_xy(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x": np.asarray(X, dtype=float), "y": np.asarray(Y, dtype=float)})

def _xy_from_table(df: pd.DataFrame, L: float) -> tuple[np.ndarray, np.ndarray] | None:
    if "x" not in df.columns or "y" not in df.columns: return None
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2: return None
    # 0..L にクリップして昇順
    x = np.clip(x, 0.0, L)
    order = np.argsort(x)
    return x[order], y[order]

# ---------------- Session defaults for geometry editing ----------------
if "geom_custom" not in st.session_state:
    st.session_state["geom_custom"] = False
if "ground_table" not in st.session_state:
    st.session_state["ground_table"] = None
if "iface_tables" not in st.session_state:
    st.session_state["iface_tables"] = {}

# ---------------- Inputs ----------------
with st.form("params"):
    A, B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

        # base examples (used when not editing)
        ground_default = make_ground_example(H, L)

        st.checkbox("Edit geometry in tables (ground / interfaces)", value=st.session_state["geom_custom"], key="geom_custom")
        if st.session_state["geom_custom"]:
            # init tables if first time or size changed
            if st.session_state["ground_table"] is None:
                st.session_state["ground_table"] = _mk_table_from_xy(ground_default.X, ground_default.Y)
            st.caption("Ground surface")
            gt = st.data_editor(
                st.session_state["ground_table"],
                num_rows="dynamic", use_container_width=True, key="ground_tbl_edit",
            )

            n_layers_edit = st.selectbox("Number of layers (tables)", [0,1,2], index=0,
                                         help="ここでの値は“表編集”用。下の『Layers』の設定とは独立です。")
            iface_tables_local = st.session_state["iface_tables"]
            for k in range(n_layers_edit):
                name = f"Interface {k+1}"
                if name not in iface_tables_local:
                    # 初期形は地表 −(0.35H, 0.60H) 相当
                    if k == 0:
                        xi, yi = make_interface1_example(H, L)
                    else:
                        xi, yi = make_interface2_example(H, L)
                    iface_tables_local[name] = _mk_table_from_xy(xi, yi)
                st.caption(name)
                iface_tables_local[name] = st.data_editor(
                    iface_tables_local[name], num_rows="dynamic", use_container_width=True, key=f"iface_tbl_{k}"
                )
            st.session_state["iface_tables"] = iface_tables_local
            st.info("これらの表は、下の『Layers』の枚数に合わせて**計算時**に用いられます。")

        st.subheader("Layers (for calculation)")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces = []
        if n_layers >= 2:
            if st.session_state["geom_custom"] and "Interface 1" in st.session_state["iface_tables"]:
                xy = _xy_from_table(st.session_state["iface_tables"]["Interface 1"], L)
                interfaces.append(xy if xy is not None else make_interface1_example(H,L))
            else:
                interfaces.append(make_interface1_example(H, L))
        if n_layers >= 3:
            if st.session_state["geom_custom"] and "Interface 2" in st.session_state["iface_tables"]:
                xy = _xy_from_table(st.session_state["iface_tables"]["Interface 2"], L)
                interfaces.append(xy if xy is not None else make_interface2_example(H,L))
            else:
                interfaces.append(make_interface2_example(H, L))

        # finalize ground (edited or default)
        if st.session_state["geom_custom"]:
            gxy = _xy_from_table(st.session_state["ground_tbl_edit"], L)
            ground = GroundPL(*gxy) if gxy is not None else ground_default
        else:
            ground = ground_default

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
        wsrc = st.radio("Source", ["Linear (2-point)", "CSV polyline", "Offset from ground"], horizontal=True)
        enable_w = st.checkbox("Enable water table", True)
        gamma_w = st.number_input("γ_w (kN/m³)", 9.00, 10.50, 9.81, 0.01)
        pore_mode = st.radio("Pore handling", ["u-only (total W)", "buoyancy"], horizontal=True)
        water_csv_file = None
        yWL = yWR = None
        w_offset = None
        if wsrc.startswith("Linear"):
            yWL = st.number_input("Water level at x=0 (m)", 0.0, 2.5*H, 0.6*H, 0.1)
            yWR = st.number_input("Water level at x=L (m)", 0.0, 2.5*H, 0.3*H, 0.1)
        elif "CSV" in wsrc:
            water_csv_file = st.file_uploader("Upload water polyline CSV (x,y columns)", type=["csv"])
        else:
            w_offset = st.number_input("Offset below ground (m)", 0.0, 50.0, 5.0, 0.5,
                                       help="地表面と同じ形状で、鉛直にこの距離だけ下げた水位線。")

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
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5,
                                    help="円弧最深点の地表からの鉛直深さレンジ")

    run = st.form_submit_button("▶ 計算開始")

# -------- Quality expand --------
P = QUALITY[quality].copy()
if 'override' in locals() and override:
    P.update(dict(
        quick_slices=quick_slices_in, final_slices=final_slices_in,
        n_entries_final=n_entries_final_in, probe_n_min_quick=probe_min_q_in,
        limit_arcs_quick=limit_arcs_q_in,
        budget_coarse_s=budget_coarse_in, budget_quick_s=budget_quick_in,
    ))

# -------- Water line object --------
def load_water_csv(file) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        df = pd.read_csv(file)
    except Exception:
        return None
    cols = [c.lower() for c in df.columns]
    if "x" in cols and "y" in cols:
        x = df[df.columns[cols.index("x")]].to_numpy(dtype=float)
        y = df[df.columns[cols.index("y")]].to_numpy(dtype=float)
    else:
        if df.shape[1] < 2: return None
        x = df.iloc[:,0].to_numpy(dtype=float)
        y = df.iloc[:,1].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2: return None
    order = np.argsort(x)
    return x[order], y[order]

water = None
water_key = None
if enable_w:
    if wsrc.startswith("Linear"):
        water = WaterLine(0.0, L, float(yWL), float(yWR)); water_key=("linear", round(float(yWL),3), round(float(yWR),3))
    elif "CSV" in wsrc:
        if water_csv_file is not None:
            xy = load_water_csv(water_csv_file)
            if xy is not None:
                water = WaterPolyline(np.asarray(xy[0],dtype=float), np.asarray(xy[1],dtype=float))
                water_key=("csv", len(xy[0]), round(float(xy[0][0]),3), round(float(xy[0][-1]),3))
            else:
                st.warning("CSVを読めませんでした。ヘッダx,y、または先頭2列にx,yを入れてください。")
    else:
        water = WaterOffset(float(w_offset)); water_key=("offset", round(float(w_offset),3))

pore_tag = "buoyancy" if pore_mode.startswith("buoyancy") else "u-only"

# -------- Keys --------
def param_pack():
    return dict(
        H=H, L=L, n_layers=n_layers,
        soils=[(s.gamma, s.c, s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=Fs_target,
        center=[x_min, x_max, y_min, y_max, nx, ny],
        method=method, quality=P, depth=[depth_min, depth_max],
        water_key=water_key, gamma_w=gamma_w, pore=pore_tag,
        ghash=hash_params({"gx":ground.X.tolist(),"gy":ground.Y.tolist(),
                           "if": [ (np.asarray(ix).tolist(), np.asarray(iy).tolist()) for (ix,iy) in interfaces ]})
    )
param_key = hash_params(param_pack())

# -------- Compute --------
def compute_once():
    # 1) Coarse: pick best center with time budget
    def subsampled_centers():
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        tag = P["coarse_subsample"]
        if tag == "every 3rd":
            xs = xs[::3] if len(xs)>2 else xs; ys = ys[::3] if len(ys)>2 else ys
        elif tag == "every 2nd":
            xs = xs[::2] if len(xs)>1 else xs; ys = ys[::2] if len(ys)>1 else ys
        return [(float(xc), float(yc)) for yc in ys for xc in xs]

    def pick_center(budget_s):
        deadline = time.time() + budget_s
        best=None; tested=[]
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
                water=water, gamma_w=gamma_w, pore_mode=pore_tag,
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
            return dict(error="Coarseで候補なし。枠/深さ/水位を見直してください。")
    xc, yc = center

    # 端ヒット → 表示グリッド自動拡張（計算枠はそのまま）
    hit, where = near_edge(xc,yc,x_min,x_max,y_min,y_max)
    x_min_a, x_max_a, y_min_a, y_max_a = x_min, x_max, y_min, y_max
    expand_note=None
    if hit:
        dx=(x_max-x_min); dy=(y_max-y_min)
        if where["left"]:  x_min_a = x_min - 0.20*dx
        if where["right"]: x_max_a = x_max + 0.20*dx
        if where["bottom"]:y_min_a = y_min - 0.20*dy
        if where["top"]:   y_max_a = y_max + 0.20*dy
        expand_note=f"audit grid auto-extend x[{x_min_a:.1f},{x_max_a:.1f}], y[{y_min_a:.1f},{y_max_a:.1f}]"

    # 2) Quick: R候補抽出（時間バジェット）
    def quick_R_candidates(xc, yc, budget_s, rescue=False):
        heap_R=[]; deadline=time.time()+budget_s
        n_entries = int(P["n_entries_final"] * (1.2 if rescue else 1.0))
        probe_min = max(41, int(P["probe_n_min_quick"] * (0.6 if rescue else 1.0)))
        limit_arcs = int(P["limit_arcs_quick"] * (1.6 if rescue else 1.0))
        tol_R = 0.10 if rescue else 0.03
        dmin = max(0.0, depth_min * (0.5 if rescue else 1.0))
        dmax = max(dmin + 0.5, depth_max * (1.5 if rescue else 1.0))

        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=n_entries, method="Fellenius",
            depth_min=dmin, depth_max=dmax,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=limit_arcs,
            probe_n_min=probe_min,
            water=water, gamma_w=gamma_w, pore_mode=pore_tag,
            tol_R=tol_R,
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        return R_candidates

    with st.spinner("Quick（R候補抽出）"):
        R_candidates = quick_R_candidates(xc, yc, P["budget_quick_s"], rescue=False)
        rescue_used = False
        if not R_candidates:
            R_candidates = quick_R_candidates(xc, yc, max(0.6, 0.7*P["budget_quick_s"]), rescue=True)
            rescue_used = bool(R_candidates)

        if not R_candidates:
            msg = "Quickで円弧候補なし。深さ/水位/Qualityを緩めてください。"
            return dict(error=msg)

    # 3) Refine: 選抜Rで精算
    refined=[]
    for R in R_candidates[:P["show_k"]]:
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method,
                              xc, yc, R, n_slices=P["final_slices"],
                              water=water, gamma_w=gamma_w, pore_mode=pore_tag)
        if Fs is None: continue
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251)
        if s is None: continue
        x1,x2 = s
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross,
                                        xc, yc, R, n_slices=P["final_slices"])
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (Fs_target - Fs)*D_sum)
        refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Quality/水位を見直してください。")
    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)
    centers_audit= grid_points(x_min_a, x_max_a, y_min_a, y_max_a, nx, ny)

    return dict(center=(xc,yc), refined=refined, idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp, centers_audit=centers_audit, expand_note=expand_note)

# run or key change
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res:
        st.error(res["error"]); st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]; centers_audit = res["centers_audit"]

# -------- After-run toggles --------
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
    audit_limit = st.slider("Audit: max arcs/center", 5, 40, QUALITY[quality]["audit_limit_per_center"], 1, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 1.0, 6.0, QUALITY[quality]["audit_budget_s"], 0.1, disabled=not audit_show)
    audit_seed   = st.number_input("Audit seed", 0, 9999, 0, disabled=not audit_show)

# -------- Audit computation --------
def compute_audit_arcs(centers, per_center_limit, total_budget_s, seed=0):
    rng = random.Random(int(seed))
    order = list(range(len(centers))); rng.shuffle(order)
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
            water=water, gamma_w=gamma_w, pore_mode=pore_tag,
        ):
            arcs.append(dict(x1=a_x1,x2=a_x2,R=R,Fs=Fs,xc=cx,yc=cy))
            count+=1
            if count>=per_center_limit: break
        if count>0: covered.add(idx)
        if time.time() > deadline: break
    return arcs, len(covered), len(centers)

audit_arcs=[]; covered=0; total=0
if audit_show:
    akey = hash_params(dict(K=param_key, limit=audit_limit, budget=round(audit_budget,2), seed=int(audit_seed)))
    cache = st.session_state.get("audit_cache", {})
    if cache.get("key")==akey and "arcs" in cache:
        audit_arcs=cache["arcs"]; covered=cache["covered"]; total=cache["total"]
    else:
        with st.spinner("Audit（全センターQuick可視化）..."):
            audit_arcs, covered, total = compute_audit_arcs(centers_audit, audit_limit, audit_budget, seed=audit_seed)
        st.session_state["audit_cache"] = {"key": akey, "arcs": audit_arcs, "covered": covered, "total": total}

# -------- Plot --------
fig, ax = plt.subplots(figsize=(10.5, 7.5))
Xd = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

# layers fill
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

# water line
if water is not None:
    if isinstance(water, WaterOffset):
        # オフセットは地表に重なるので見やすい色で
        Yw = np.array([water.y_at(float(x), ground) for x in Xd], dtype=float)
        ax.plot(Xd, Yw, color="tab:blue", linestyle="-.", linewidth=1.4, label=f"Water offset {water.offset:.2f} m")
    else:
        Yw = np.array([water.y_at(float(x), ground) for x in Xd], dtype=float)
        ax.plot(Xd, Yw, color="tab:blue", linestyle="--", linewidth=1.3, label="Water table")
    ax.fill_between(Xd, np.minimum(Yw, Yg), 0.0, color="tab:blue", alpha=0.06)

# 外周を閉じる
ax.plot([Xd[-1], Xd[-1]],[0.0, Yg[-1]], linewidth=1.0, color="k", alpha=0.7)
ax.plot([Xd[0],  Xd[-1]],[0.0, 0.0],     linewidth=1.0, color="k", alpha=0.7)
ax.plot([Xd[0],  Xd[0]], [0.0, Yg[0]],   linewidth=1.0, color="k", alpha=0.7)

# center-grid
show_centers = st.session_state.get("show_centers", True)
xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# refined arcs
show_all_refined = st.session_state.get("show_all_refined", True)
for d in refined:
    xs=np.linspace(d["x1"], d["x2"], 200)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=0.9, alpha=0.80, color=fs_to_color(d["Fs"]))

# pick-ups
show_minFs = st.session_state.get("show_minFs", True)
show_maxT  = st.session_state.get("show_maxT", True)
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
x_upper = max(1.18*(ground.X[-1]-ground.X[0]), x_max + 0.05*(ground.X[-1]-ground.X[0]), 100.0)
y_upper = max(2.30*H, y_max + 0.05*H, 100.0)
ax.set_xlim(min(ground.X[0]-0.05*(ground.X[-1]-ground.X[0]), -2.0), ground.X[-1]+max(0.05*(ground.X[-1]-ground.X[0]), 2.0))
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
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))
st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（精密・選抜センター）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
