# streamlit_app.py — 水位/ru対応・（元の）監査UI・描画安全化（y>=0クリップ）
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

st.set_page_config(page_title="Stabi LEM｜監査", layout="wide")
st.title("Stabi LEM｜全センター監査")

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

def clip_yfloor(xs: np.ndarray, ys: np.ndarray, y_floor: float = 0.0):
    """描画安全化：y>=y_floor の区間だけ残す。2点未満なら None を返す。"""
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2:
        return None
    return xs[m], ys[m]

# ---------------- 水（WT/ru）ヘルパ ----------------
def build_water(ground: GroundPL, mode: str, ru_value: float, wt_points):
    """
    stabi_lem 側のインタフェース想定：
      - {"type":"dry"}       : 水の影響なし
      - {"type":"ru","ru":r} : 間隙水圧比 r を用いる（WT線は描かない）
      - {"type":"WT","z":callable} : x→水位z の関数
    """
    if mode == "None":
        return {"type":"dry"}, None
    if mode == "ru":
        return {"type":"ru", "ru": float(ru_value)}, None
    # "Water table"
    if wt_points is None or len(wt_points)==0:
        # デフォルト：地表から一定オフセット（UI側で作成した points が入ってくる想定）
        def zfun(x): return np.asarray(ground.y_at(x), dtype=float) - 2.0
        return {"type":"WT", "z": zfun}, zfun
    arr = np.asarray(wt_points, dtype=float)
    xp, zp = arr[:,0], arr[:,1]
    order = np.argsort(xp); xp = xp[order]; zp = zp[order]
    def zfun(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        out[x <= xp[0]] = zp[0]
        out[x >= xp[-1]] = zp[-1]
        mid = (x > xp[0]) & (x < xp[-1])
        out[mid] = np.interp(x[mid], xp, zp)
        return out
    return {"type":"WT", "z": zfun}, zfun

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

        # --- Water / ru ---
        st.subheader("Water condition")
        water_mode = st.radio("Water model", ["None", "Water table", "ru"], index=0, horizontal=True)
        ru_value = 0.0
        wt_points = None
        if water_mode == "Water table":
            wt_source = st.radio("Source", ["Offset from ground", "CSV / manual edit"], index=0, horizontal=True)
            if wt_source == "Offset from ground":
                wt_offset = st.number_input("Offset below ground (m, positive downward)", 0.0, 50.0, 2.0, 0.1)
                Xd = np.linspace(ground.X[0], ground.X[-1], 16)
                Yd = ground.y_at(Xd) - wt_offset
                if "wt_editor" not in st.session_state:
                    st.session_state["wt_editor"] = np.c_[Xd, Yd]
                else:
                    # groundが変わった時は再生成（簡易）
                    arr = np.asarray(st.session_state["wt_editor"])
                    if arr.shape[0] != len(Xd):
                        st.session_state["wt_editor"] = np.c_[Xd, Yd]
                wt_points = st.data_editor(
                    st.session_state["wt_editor"], num_rows="dynamic",
                    columns={"0":"x","1":"z(water)"},
                    key="wt_editor", use_container_width=True
                )
            else:
                # CSVアップロード
                up = st.file_uploader("Upload CSV (x,z)", type=["csv"])
                if up is not None:
                    content = up.read().decode("utf-8").strip().splitlines()
                    data = []
                    for line in content:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0].strip()); z = float(parts[1].strip())
                                data.append((x,z))
                            except:
                                pass
                    if data:
                        data = np.array(sorted(data, key=lambda t:t[0]), dtype=float)
                        st.session_state["wt_editor"] = data
                        wt_points = data
                # 手動編集（アップロードがなくても行は作れる）
                wt_points = st.data_editor(
                    st.session_state.get("wt_editor", np.c_[np.linspace(ground.X[0], ground.X[-1], 8),
                                                           ground.y_at(np.linspace(ground.X[0], ground.X[-1], 8))-2.0]),
                    num_rows="dynamic", columns={"0":"x","1":"z(water)"},
                    key="wt_editor_manual", use_container_width=True
                )
        elif water_mode == "ru":
            ru_value = st.slider("ru value", 0.0, 1.0, 0.30, 0.05)

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
        # 水条件をハッシュに含めて再計算トリガにする
        water_mode=water_mode, ru_value=ru_value,
        wt_points=(np.asarray(st.session_state.get("wt_editor")).tolist() if water_mode=="Water table" else None),
    )
param_key = hash_params(param_pack())

# ---------------- Compute ----------------
def compute_once():
    # 水条件の構築
    wt_pts = np.asarray(st.session_state.get("wt_editor")) if water_mode=="Water table" else None
    water, zfun = build_water(ground, water_mode, ru_value, wt_pts)

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
                water=water,  # ★ 水条件を渡す
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
            return dict(error="Coarseで候補なし。枠/深さを広げてください。", water=water, zfun=zfun)
    xc, yc = center

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
            water=water,  # ★
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Qualityを緩めてください。", water=water, zfun=zfun)

    # 3) Refine for chosen center
    refined=[]
    for R in R_candidates[:P["show_k"]]:
        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"], water=water)  # ★
        if Fs is None: continue
        s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251, y_floor=0.0)
        if s is None: continue
        x1,x2,*_ = s
        packD = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"], water=water)  # ★
        if packD is None: continue
        D_sum,_,_ = packD
        T_req = max(0.0, (Fs_target - Fs)*D_sum)
        refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
    if not refined:
        return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。", water=water, zfun=zfun)
    refined.sort(key=lambda d:d["Fs"])
    idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
    idx_maxT  = int(np.argmax([d["T_req"] for d in refined]))

    # 表示用 center-grid（ユーザー指定範囲のみ）
    centers_disp = grid_points(x_min, x_max, y_min, y_max, nx, ny)

    return dict(center=(xc,yc), refined=refined,
                idx_minFs=idx_minFs, idx_maxT=idx_maxT,
                centers_disp=centers_disp,
                water=water, zfun=zfun)

# run
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"] != param_key):
    res = compute_once()
    if "error" in res:
        st.error(res["error"])
        # エラー時でも地形と水位は描けるように保持
        st.session_state["last_key"] = param_key
        st.session_state["res"] = res
        st.stop()
    st.session_state["last_key"] = param_key
    st.session_state["res"] = res

res = st.session_state["res"]
xc,yc = res["center"]
refined = res["refined"]; idx_minFs = res["idx_minFs"]; idx_maxT=res["idx_maxT"]
centers_disp = res["centers_disp"]

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
    audit_show = st.checkbox("Show arcs from ALL centers (Quick audit)", False)
    audit_limit = st.slider("Audit: max arcs/center", 5, 40, QUALITY[quality]["audit_limit_per_center"], 1, disabled=not audit_show)
    audit_budget = st.slider("Audit: total budget (sec)", 1.0, 6.0, QUALITY[quality]["audit_budget_s"], 0.1, disabled=not audit_show)
    audit_seed   = st.number_input("Audit seed", 0, 9999, 0, disabled=not audit_show)

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.5))

Xd = np.linspace(ground.X[0], ground.X[-1], 600)
Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

# 層の塗り
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

# 地表と境界線
ax.plot(ground.X, ground.Y, linewidth=2.2, label="Ground")
if n_layers>=2:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.2, label="Interface 1")
if n_layers>=3:
    ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.2, label="Interface 2")

# 外周
ax.plot([ground.X[-1], ground.X[-1]],[0.0, ground.y_at(ground.X[-1])], linewidth=1.0)
ax.plot([ground.X[0],  ground.X[-1]],[0.0, 0.0],                       linewidth=1.0)
ax.plot([ground.X[0],  ground.X[0]], [0.0, ground.y_at(ground.X[0])],  linewidth=1.0)

# 水位線の描画（ru以外）
if res.get("water", {}).get("type") == "WT" and res.get("zfun") is not None:
    zf = res["zfun"]; Zw = np.asarray(zf(Xd), dtype=float)
    # 地表より上は薄く、地下は実線
    below = Zw <= Yg + 1e-9
    if np.any(below):
        ax.plot(Xd[below], Zw[below], color="tab:blue", linewidth=1.6, label="Water table")
    if np.any(~below):
        ax.plot(Xd[~below], Zw[~below], color="tab:blue", linewidth=1.0, alpha=0.35)

# center-grid
if show_centers:
    xs=[c[0] for c in centers_disp]; ys=[c[1] for c in centers_disp]
    ax.scatter(xs, ys, s=12, c="k", alpha=0.25, marker=".", label="Center grid")

# chosen center
ax.scatter([xc],[yc], s=70, marker="s", color="tab:blue", label="Chosen center")

# audit arcs（描画時クリップ）—（必要なら後でキャッシュ計算を追加してください）
if audit_show and 'audit_cache' in st.session_state:
    for a in st.session_state.get("audit_cache", {}).get("arcs", []):
        cx,cy,R,x1,x2,Fs = a["xc"],a["yc"],a["R"],a["x1"],a["x2"],a["Fs"]
        xs = np.linspace(x1, x2, 140)
        ys = cy - np.sqrt(np.maximum(0.0, R*R - (xs - cx)**2))
        clipped = clip_yfloor(xs, ys, y_floor=0.0)
        if clipped is None: 
            continue
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=0.6, alpha=0.25, color=fs_to_color(Fs))

# refined（描画時クリップ）
if show_all_refined:
    for d in refined:
        xs=np.linspace(d["x1"], d["x2"], 200)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped = clip_yfloor(xs, ys, y_floor=0.0)
        if clipped is None: 
            continue
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

# pick-ups（描画時クリップ）
if show_minFs and 0<=idx_minFs<len(refined):
    d=refined[idx_minFs]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    clipped = clip_yfloor(xs, ys, y_floor=0.0)
    if clipped is not None:
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
        y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
        ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
        ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

if show_maxT and 0<=idx_maxT<len(refined):
    d=refined[idx_maxT]
    xs=np.linspace(d["x1"], d["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
    clipped = clip_yfloor(xs, ys, y_floor=0.0)
    if clipped is not None:
        xs_c, ys_c = clipped
        ax.plot(xs_c, ys_c, linewidth=3.0, linestyle="--", color=(0.2,0.0,0.8),
                label=f"Max required T = {d['T_req']:.1f} kN/m (Fs={d['Fs']:.3f})")
        y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
        ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)
        ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, linestyle="--", color=(0.2,0.0,0.8), alpha=0.9)

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
if res.get("water",{}).get("type")=="ru":
    title_tail.append(f"ru={res['water'].get('ru',0):.2f}")
elif res.get("water",{}).get("type")=="WT":
    title_tail.append("water table: ON")
ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • Method={method} • " + " • ".join(title_tail))

st.pyplot(fig, use_container_width=True); plt.close(fig)

# metrics
m1,m2 = st.columns(2)
with m1: st.metric("Min Fs（精密・選抜センター）", f"{refined[idx_minFs]['Fs']:.3f}")
with m2: st.metric("Max required T", f"{refined[idx_maxT]['T_req']:.1f} kN/m")
