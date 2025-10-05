# streamlit_app.py — Stabi LEM + Soil Nail（安定・一括貼替版）
from __future__ import annotations

# ====== Imports ======
import os, sys, math, time, heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
ST = st  # ← 保険：どこかで st を数値で上書きしても ST を残す

# ---- stabi_lem の import（パッケージ/単体の両対応）----
try:
    import stabi.stabi_lem as lem
except ModuleNotFoundError:
    import stabi_lem as lem

# ★ “lem”経由で必要シンボルを束ねる（from lem import ... は使わない）
Soil = lem.Soil
GroundPL = lem.GroundPL
make_ground_example = lem.make_ground_example
make_interface1_example = lem.make_interface1_example
make_interface2_example = lem.make_interface2_example
clip_interfaces_to_ground = lem.clip_interfaces_to_ground
arcs_from_center_by_entries_multi = lem.arcs_from_center_by_entries_multi
fs_given_R_multi = lem.fs_given_R_multi
arc_sample_poly_best_pair = lem.arc_sample_poly_best_pair
driving_sum_for_R_multi = lem.driving_sum_for_R_multi

# ---- nail_engine / coupler が無い場合のフォールバック ----
try:
    from nail_engine import reinforce_nails
except Exception:
    def reinforce_nails(arc, ground, soils, nails_cfg, slices):
        N = len(slices.get("x_mid", []))
        return np.zeros(N, dtype=float), {"hits": [], "notes": "nail_engine not found"}

try:
    from coupler import bishop_with_reinforcement
except Exception:
    def bishop_with_reinforcement(slices, soil, Tt):
        D = float(np.sum(slices["W"] * np.sin(slices["alpha"])))
        if D <= 0: return slices.get("Fs0", np.nan)
        return slices.get("Fs0", 1.0) + float(np.sum(Tt)) / D

st.set_page_config(page_title="Stabi LEM｜cfg一元・安定版", layout="wide")
st.title("Stabi LEM｜多段UI（cfg一元・安定版）")

DEG = math.pi / 180.0

# ========= 共通ヘルパ =========
def inward_normal_angle(g: GroundPL, x: float, delta_beta_deg: float) -> float:
    """
    斜面接線角 tau から ±90° の法線2候補を作り，
    **常に地山側（下向き: sinθ < 0）** を選んで返す。
    """
    x2 = x + 1e-4
    y1 = float(g.y_at(x)); y2 = float(g.y_at(x2))
    tau = math.atan2((y2 - y1), (x2 - x))
    db = float(delta_beta_deg) * DEG
    cand1 = tau - math.pi/2 + db
    cand2 = tau + math.pi/2 + db
    th = cand1 if math.sin(cand1) < 0 else cand2
    if math.sin(th) >= 0:  # 念のため保険
        th -= math.pi
    return th

def nail_ray_hit_circle(xh, yh, theta, xc, yc, R):
    """ネイル頭(xh,yh)から方向thetaの光線と円の交点距離 t（t>0 の最小）"""
    ct, snt = math.cos(theta), math.sin(theta)
    B = 2 * ((xh - xc) * ct + (yh - yc) * snt)
    C = (xh - xc)**2 + (yh - yc)**2 - R**2
    disc = B*B - 4*C
    if disc <= 0:
        return None
    rt = math.sqrt(max(0.0, disc))
    cand = [(-B - rt) / 2.0, (-B + rt) / 2.0]
    tpos = [t for t in cand if t > 1e-9]
    return (min(tpos) if tpos else None)

def grout_pullout_capacity_kNm_per_m(d_g, Lb, tau_kPa):
    """τ[kPa]=kN/m^2, 周長 π d_g, 長さ Lb → kN/m（2D 1m幅）"""
    return math.pi * float(d_g) * float(max(0.0, Lb)) * float(max(0.0, tau_kPa))

def steel_tension_capacity_kN_per_m(d_s, fy_MPa, gamma_m):
    """As=π d^2/4, fy[MPa]×1000=kN/m^2 → kN/m"""
    As = math.pi * (float(d_s)**2) / 4.0
    fy_kNm2 = float(fy_MPa) * 1000.0
    return As * fy_kNm2 / max(1.0, float(gamma_m))

# ===================== cfg（正本） =====================
def default_cfg():
    return {
        "geom": {"H": 25.0, "L": 60.0},
        "water": {"mode": "WT", "ru": 0.0, "offset": -2.0, "wl_points": None},
        "layers": {
            "n": 2,
            "mat": {
                1: {"gamma": 18.0, "c": 5.0,  "phi": 30.0, "tau": 150.0},
                2: {"gamma": 19.0, "c": 8.0,  "phi": 28.0, "tau": 180.0},
                3: {"gamma": 20.0, "c": 12.0, "phi": 25.0, "tau": 200.0},
            },
            "tau_grout_cap_kPa": 150.0,
            "d_g": 0.125,  # m
            "d_s": 0.022,  # m
            "fy": 1000.0, "gamma_m": 1.20, "mu": 0.0,
        },
        "grid": {
            "x_min": None, "x_max": None, "y_min": None, "y_max": None,
            "pitch": 5.0,
            "method": "Bishop (simplified)",
            "quality": "Normal",
            "Fs_target": 1.20,
            "allow_cross2": True, "allow_cross3": True,
        },
        "nails": {
            "s_start": 5.0, "s_end": 35.0, "S_surf": 2.0, "S_row": 2.0,
            "tiers": 1,
            "angle_mode": "Slope-Normal (⊥斜面)",
            "beta_deg": 15.0, "delta_beta": 0.0,
            "L_mode": "パターン1：固定長", "L_nail": 5.0, "d_embed": 1.0,
        },
        "results": {
            "unreinforced": None,
            "chosen_arc": None,
            "nail_heads": [],
            "reinforced": None,
        }
    }

# --- 数値キー対応 cfg_get/cfg_set/ui_seed ---
def _maybe_int_key(p):
    if isinstance(p, str) and p.isdigit():
        try: return int(p)
        except: return p
    return p

def cfg_get(path, default=None):
    node = st.session_state["cfg"]
    for p in path.split("."):
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node: node = node[p]
            elif p_try in node: node = node[p_try]
            else: return default
        else: return default
    return node

def cfg_set(path, value):
    node = st.session_state["cfg"]; parts = path.split(".")
    for p in parts[:-1]:
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node: node = node[p]
            elif p_try in node: node = node[p_try]
            else:
                node[p_try] = {}; node = node[p_try]
        else:
            raise KeyError(f"cfg_set: '{p}' below is not a dict")
    last = _maybe_int_key(parts[-1])
    if isinstance(node, dict): node[last] = value
    else: raise KeyError(f"cfg_set: cannot set at '{parts[-1]}'")

def ui_seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# 起動時に cfg を1度だけ生成
if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

# ===== サイドバー（ページ選択／初期化）=====
with st.sidebar:
    st.header("Pages")
    page = st.radio(
        "",
        ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"],
        key="__page__"
    )
    st.caption("cfgが正本。保存しない限り自動上書きしません。")
    if st.button("⚠ すべて初期化（cfgを再作成）"):
        st.session_state["cfg"] = default_cfg()
        st.success("初期化しました。")

# ========= 描画小物 =========
def make_ground_from_cfg():
    H = float(cfg_get("geom.H")); L = float(cfg_get("geom.L"))
    return H, L, make_ground_example(H,L)

def set_axes(ax, H, L, ground):
    x_upper = max(1.18*L, float(ground.X[-1])+0.05*L, 100.0)
    y_upper = max(2.30*H, 0.05*H+2.0*H, 100.0)
    ax.set_xlim(min(0.0-0.05*L, -2.0), x_upper)
    ax.set_ylim(0.0, y_upper)
    ax.set_aspect("equal", adjustable="box")

def draw_layers_and_ground(ax, ground, n_layers, interfaces):
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
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
    ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")
    return Xd, Yg

def draw_water(ax, ground, Xd, Yg):
    wm = cfg_get("water.mode")
    if not str(wm).startswith("WT"): return
    W = cfg_get("water.wl_points")
    if W is not None:
        W = np.asarray(W, dtype=float)
    if W is not None and isinstance(W, np.ndarray) and W.ndim==2 and W.shape[1]==2:
        Yw = np.interp(Xd, W[:,0], W[:,1], left=W[0,1], right=W[-1,1])
        Yw = np.clip(Yw, 0.0, Yg)
        ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (saved)")
    else:
        off = float(cfg_get("water.offset",-2.0))
        Yw = np.clip(Yg + off, 0.0, Yg)
        ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (offset preview)")

def fs_to_color(fs):
    if fs < 1.0: return (0.85,0,0)
    if fs < 1.2:
        t=(fs-1.0)/0.2; return (1.0,0.50+0.50*t,0.0)
    return (0.0,0.55,0.0)

def clip_yfloor(xs, ys, y_floor=0.0):
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2: return None
    return xs[m], ys[m]

# ========= QUALITY プリセット =========
QUALITY = {
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, coarse_subsample="every 2nd",
                   coarse_entries=220, coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2),
}

# ===================== Page1: 地形・水位 =====================
if page.startswith("1"):
    # UI seed
    ui_seed("H", cfg_get("geom.H"))
    ui_seed("L", cfg_get("geom.L"))
    ui_seed("water_mode", cfg_get("water.mode"))
    ui_seed("ru", cfg_get("water.ru"))
    ui_seed("wt_offset", cfg_get("water.offset"))

    st.subheader("Geometry")
    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H", value=st.session_state["H"])
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L", value=st.session_state["L"])

    st.subheader("Water model")
    st.selectbox("Water model", ["WT","ru","WT+ru"], key="water_mode", index=["WT","ru","WT+ru"].index(st.session_state["water_mode"]))
    st.slider("r_u (if ru mode)", 0.0, 0.9, step=0.05, key="ru", value=float(st.session_state["ru"]))
    st.slider("Water level offset from ground (m, negative=below)", -30.0, 5.0, step=0.5, key="wt_offset", value=float(st.session_state["wt_offset"]))

    c1,c2 = st.columns(2)
    with c1:
        if st.button("💾 形状・水位パラメータを保存（cfgへ）"):
            cfg_set("geom.H", float(st.session_state["H"]))
            cfg_set("geom.L", float(st.session_state["L"]))
            cfg_set("water.mode", st.session_state["water_mode"])
            cfg_set("water.ru", float(st.session_state["ru"]))
            cfg_set("water.offset", float(st.session_state["wt_offset"]))
            if cfg_get("grid.x_min") is None:
                L = cfg_get("geom.L"); H = cfg_get("geom.H")
                cfg_set("grid.x_min", 0.25*L); cfg_set("grid.x_max", 1.15*L)
                cfg_set("grid.y_min", 1.60*H); cfg_set("grid.y_max", 2.20*H)
            st.success("cfgに保存しました。")
    with c2:
        if st.button("💾 WT水位線を offset から生成/更新（cfg.water.wl_points）"):
            H_ui = float(st.session_state["H"]); L_ui = float(st.session_state["L"])
            ground_ui = make_ground_example(H_ui, L_ui)
            Xd = np.linspace(ground_ui.X[0], ground_ui.X[-1], 400)
            Yg = np.array([float(ground_ui.y_at(x)) for x in Xd])
            off = float(st.session_state["wt_offset"])
            Yw = np.clip(Yg + off, 0.0, Yg)
            W = np.vstack([Xd, Yw]).T
            cfg_set("water.wl_points", np.asarray(W, dtype=float))
            st.success("水位線をcfgに保存しました（以後この線が最優先）。")

    # プレビュー
    H_ui = float(st.session_state["H"])
    L_ui = float(st.session_state["L"])
    ground_ui = make_ground_example(H_ui, L_ui)

    n_layers_cfg = int(cfg_get("layers.n"))
    interfaces_ui = []
    if n_layers_cfg >= 2: interfaces_ui.append(make_interface1_example(H_ui, L_ui))
    if n_layers_cfg >= 3: interfaces_ui.append(make_interface2_example(H_ui, L_ui))

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    Xd = np.linspace(ground_ui.X[0], ground_ui.X[-1], 600)
    Yg = np.array([float(ground_ui.y_at(x)) for x in Xd])

    if n_layers_cfg == 1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif n_layers_cfg == 2:
        Y1 = clip_interfaces_to_ground(ground_ui, [interfaces_ui[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1, Y2 = clip_interfaces_to_ground(ground_ui, [interfaces_ui[0], interfaces_ui[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

    ax.plot(ground_ui.X, ground_ui.Y, linewidth=2.0, label="Ground")
    draw_water(ax, ground_ui, Xd, Yg)
    set_axes(ax, H_ui, L_ui, ground_ui); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page2: 地層・材料 =====================
elif page.startswith("2"):
    if cfg_get("layers.mat") is None:
        cfg_set("layers.mat", default_cfg()["layers"]["mat"])

    # UI seed
    ui_seed("n_layers", cfg_get("layers.n"))
    m1 = cfg_get("layers.mat.1"); m2 = cfg_get("layers.mat.2"); m3 = cfg_get("layers.mat.3")
    ui_seed("gamma1", m1["gamma"]); ui_seed("c1", m1["c"]); ui_seed("phi1", m1["phi"]); ui_seed("tau1", m1["tau"])
    ui_seed("gamma2", m2["gamma"]); ui_seed("c2", m2["c"]); ui_seed("phi2", m2["phi"]); ui_seed("tau2", m2["tau"])
    ui_seed("gamma3", m3["gamma"]); ui_seed("c3", m3["c"]); ui_seed("phi3", m3["phi"]); ui_seed("tau3", m3["tau"])
    ui_seed("tau_grout_cap_kPa", cfg_get("layers.tau_grout_cap_kPa"))
    ui_seed("d_g_mm", int(round(cfg_get("layers.d_g")*1000)))
    ui_seed("d_s_mm", int(round(cfg_get("layers.d_s")*1000)))
    ui_seed("fy", cfg_get("layers.fy")); ui_seed("gamma_m", cfg_get("layers.gamma_m")); ui_seed("mu", cfg_get("layers.mu"))

    H,L,ground = make_ground_from_cfg()
    st.subheader("Layers & Materials")
    st.selectbox("Number of layers", [1,2,3], key="n_layers", index=[1,2,3].index(st.session_state["n_layers"]))

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Layer1 (top)**")
        st.number_input("γ₁", 10.0, 25.0, step=0.5, key="gamma1", value=float(st.session_state["gamma1"]))
        st.number_input("c₁", 0.0, 200.0, step=0.5, key="c1", value=float(st.session_state["c1"]))
        st.number_input("φ₁", 0.0, 45.0, step=0.5, key="phi1", value=float(st.session_state["phi1"]))
        st.number_input("τ₁ (kPa)", 0.0, 2000.0, step=10.0, key="tau1", value=float(st.session_state["tau1"]))
    if st.session_state["n_layers"]>=2:
        with cols[1]:
            st.markdown("**Layer2**")
            st.number_input("γ₂", 10.0, 25.0, step=0.5, key="gamma2", value=float(st.session_state["gamma2"]))
            st.number_input("c₂", 0.0, 200.0, step=0.5, key="c2", value=float(st.session_state["c2"]))
            st.number_input("φ₂", 0.0, 45.0, step=0.5, key="phi2", value=float(st.session_state["phi2"]))
            st.number_input("τ₂ (kPa)", 0.0, 2000.0, step=10.0, key="tau2", value=float(st.session_state["tau2"]))
    if st.session_state["n_layers"]>=3:
        with cols[2]:
            st.markdown("**Layer3 (bottom)**")
            st.number_input("γ₃", 10.0, 25.0, step=0.5, key="gamma3", value=float(st.session_state["gamma3"]))
            st.number_input("c₃", 0.0, 200.0, step=0.5, key="c3", value=float(st.session_state["c3"]))
            st.number_input("φ₃", 0.0, 45.0, step=0.5, key="phi3", value=float(st.session_state["phi3"]))
            st.number_input("τ₃ (kPa)", 0.0, 2000.0, step=10.0, key="tau3", value=float(st.session_state["tau3"]))
    with cols[-1]:
        st.markdown("**Grout / Nail**")
        st.number_input("τ_grout_cap (kPa)", 0.0, 5000.0, step=10.0, key="tau_grout_cap_kPa", value=float(st.session_state["tau_grout_cap_kPa"]))
        st.number_input("削孔(=グラウト)径 d_g (mm)", 50, 300, step=1, key="d_g_mm", value=int(st.session_state["d_g_mm"]))
        st.number_input("鉄筋径 d_s (mm)", 10, 50, step=1, key="d_s_mm", value=int(st.session_state["d_s_mm"]))
        st.number_input("引張強さ fy (MPa)", 200.0, 2000.0, step=50.0, key="fy", value=float(st.session_state["fy"]))
        st.number_input("材料安全率 γ_m", 1.00, 2.00, step=0.05, key="gamma_m", value=float(st.session_state["gamma_m"]))
        st.select_slider("逓減係数 μ（0〜0.9, 0.1刻み）", options=[round(0.1*i,1) for i in range(10)], key="mu", value=float(st.session_state["mu"]))

    if st.button("💾 材料を保存（cfgへ）"):
        cfg_set("layers.n", int(st.session_state["n_layers"]))
        cfg_set("layers.mat.1", {"gamma":float(st.session_state["gamma1"]), "c":float(st.session_state["c1"]), "phi":float(st.session_state["phi1"]), "tau":float(st.session_state["tau1"])})
        cfg_set("layers.mat.2", {"gamma":float(st.session_state["gamma2"]), "c":float(st.session_state["c2"]), "phi":float(st.session_state["phi2"]), "tau":float(st.session_state["tau2"])})
        cfg_set("layers.mat.3", {"gamma":float(st.session_state["gamma3"]), "c":float(st.session_state["c3"]), "phi":float(st.session_state["phi3"]), "tau":float(st.session_state["tau3"])})
        cfg_set("layers.tau_grout_cap_kPa", float(st.session_state["tau_grout_cap_kPa"]))
        cfg_set("layers.d_g", float(st.session_state["d_g_mm"])/1000.0)
        cfg_set("layers.d_s", float(st.session_state["d_s_mm"])/1000.0)
        cfg_set("layers.fy", float(st.session_state["fy"]))
        cfg_set("layers.gamma_m", float(st.session_state["gamma_m"]))
        cfg_set("layers.mu", float(st.session_state["mu"]))
        st.success("cfgに保存しました。")

    fig,ax = plt.subplots(figsize=(9.5,5.8))
    Xd,Yg = draw_layers_and_ground(ax, ground, int(cfg_get("layers.n")),
                                   ([make_interface1_example(H,L)] if int(cfg_get("layers.n"))>=2 else []) +
                                   ([make_interface2_example(H,L)] if int(cfg_get("layers.n"))>=3 else []))
    draw_water(ax, ground, Xd, Yg)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page3: 円弧探索（未補強） =====================
elif page.startswith("3"):
    H,L,ground = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(make_interface2_example(H,L))

    if cfg_get("grid.x_min") is None:
        cfg_set("grid.x_min", 0.25*L); cfg_set("grid.x_max", 1.15*L)
        cfg_set("grid.y_min", 1.60*H); cfg_set("grid.y_max", 2.20*H)

    # UI seed
    ui_seed("p3_x_min", cfg_get("grid.x_min"))
    ui_seed("p3_x_max", cfg_get("grid.x_max"))
    ui_seed("p3_y_min", cfg_get("grid.y_min"))
    ui_seed("p3_y_max", cfg_get("grid.y_max"))
    ui_seed("p3_pitch", cfg_get("grid.pitch"))
    ui_seed("p3_method", cfg_get("grid.method"))
    ui_seed("p3_quality", cfg_get("grid.quality"))
    ui_seed("p3_Fs_t", cfg_get("grid.Fs_target"))
    ui_seed("p3_allow2", cfg_get("grid.allow_cross2"))
    ui_seed("p3_allow3", cfg_get("grid.allow_cross3"))

    st.subheader("円弧探索（未補強）")
    with st.form("arc_params"):
        colA,colB = st.columns([1.3,1])
        with colA:
            st.number_input("x min (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_min", value=float(st.session_state["p3_x_min"]))
            st.number_input("x max (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_max", value=float(st.session_state["p3_x_max"]))
            st.number_input("y min (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_min", value=float(st.session_state["p3_y_min"]))
            st.number_input("y max (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_max", value=float(st.session_state["p3_y_max"]))
            st.number_input("Center-grid ピッチ (m)", min_value=0.1, step=0.1, format="%.2f", key="p3_pitch", value=float(st.session_state["p3_pitch"]))
        with colB:
            st.selectbox("Method", ["Bishop (simplified)","Fellenius"], key="p3_method", index=["Bishop (simplified)","Fellenius"].index(st.session_state["p3_method"]))
            st.select_slider("Quality", options=list(QUALITY.keys()), key="p3_quality", value=st.session_state["p3_quality"])
            st.number_input("Target FS", min_value=1.00, max_value=2.00, step=0.05, format="%.2f", key="p3_Fs_t", value=float(st.session_state["p3_Fs_t"]))
        if n_layers>=2: st.checkbox("Allow into Layer 2", key="p3_allow2", value=bool(st.session_state["p3_allow2"]))
        if n_layers>=3: st.checkbox("Allow into Layer 3", key="p3_allow3", value=bool(st.session_state["p3_allow3"]))
        saved = st.form_submit_button("💾 設定を保存（cfgへ）")

    def sync_grid_ui_to_cfg():
        x_min=float(st.session_state["p3_x_min"]); x_max=float(st.session_state["p3_x_max"])
        y_min=float(st.session_state["p3_y_min"]); y_max=float(st.session_state["p3_y_max"])
        if x_max < x_min: x_min,x_max = x_max,x_min
        if y_max < y_min: y_min,y_max = y_max,y_min
        cfg_set("grid.x_min", x_min); cfg_set("grid.x_max", x_max)
        cfg_set("grid.y_min", y_min); cfg_set("grid.y_max", y_max)
        cfg_set("grid.pitch", float(max(0.1, st.session_state["p3_pitch"])))
        cfg_set("grid.method", st.session_state["p3_method"])
        cfg_set("grid.quality", st.session_state["p3_quality"])
        cfg_set("grid.Fs_target", float(st.session_state["p3_Fs_t"]))
        cfg_set("grid.allow_cross2", bool(st.session_state["p3_allow2"]))
        cfg_set("grid.allow_cross3", bool(st.session_state["p3_allow3"]))

    if saved:
        sync_grid_ui_to_cfg()
        st.success("cfgに保存しました。")

    # 可視化（grid）
    x_min=cfg_get("grid.x_min"); x_max=cfg_get("grid.x_max")
    y_min=cfg_get("grid.y_min"); y_max=cfg_get("grid.y_max")
    pitch=cfg_get("grid.pitch")
    method=cfg_get("grid.method"); quality=cfg_get("grid.quality"); Fs_t=float(cfg_get("grid.Fs_target"))
    fig,ax = plt.subplots(figsize=(10.0,6.8))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    gx = np.arange(x_min, x_max+1e-9, pitch); gy = np.arange(y_min, y_max+1e-9, pitch)
    if gx.size<1: gx=np.array([x_min])
    if gy.size<1: gy=np.array([y_min])
    xs=[float(x) for x in gx for _ in gy]; ys=[float(y) for y in gy for _ in gx]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    ax.plot([x_min,x_max,x_max,x_min,x_min],[y_min,y_min,y_max,y_max,y_min], c="k", lw=1.0, alpha=0.4)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend(loc="upper right")
    st.pyplot(fig); plt.close(fig)

    # soils & allow_cross
    mats = cfg_get("layers.mat")
    soils=[Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
    allow_cross=[]
    if n_layers>=2:
        soils.append(Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross2")))
    if n_layers>=3:
        soils.append(Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross3")))
    P = QUALITY["Normal"].copy()

    def compute_once():
        Hc,Lc,groundL = make_ground_from_cfg()
        ifaces=[]
        if n_layers>=2: ifaces.append(make_interface1_example(Hc,Lc))
        if n_layers>=3: ifaces.append(make_interface2_example(Hc,Lc))

        def subsampled():
            xs = np.arange(x_min, x_max+1e-9, pitch)
            ys = np.arange(y_min, y_max+1e-9, pitch)
            tag = "every 2nd"
            if tag=="every 2nd":
                xs = xs[::2] if xs.size>1 else xs; ys = ys[::2] if ys.size>1 else ys
            return [(float(xc),float(yc)) for yc in ys for xc in xs]

        def pick_center(budget_s):
            deadline = time.time()+budget_s; best=None
            for (xc,yc) in subsampled():
                cnt=0; Fs_min=None
                for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                    groundL, soils, xc, yc,
                    n_entries=220, method="Fellenius",
                    depth_min=0.5, depth_max=4.0,
                    interfaces=ifaces, allow_cross=allow_cross,
                    quick_mode=True, n_slices_quick=max(8,P["quick_slices"]//2),
                    limit_arcs_per_center=70,
                    probe_n_min=81,
                ):
                    cnt+=1
                    if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                    if time.time()>deadline: break
                score=(cnt, -(Fs_min if Fs_min is not None else 1e9))
                if (best is None) or (score>best[0]): best=(score,(xc,yc))
                if time.time()>deadline: break
            return (best[1] if best else None)

        center = pick_center(0.8)
        if center is None: return dict(error="Coarseで候補なし。範囲/ピッチ/深さを見直してください。")
        xc,yc = center

        heap_R=[]; deadline=time.time()+1.2
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            groundL, soils, xc, yc,
            n_entries=1300, method="Fellenius",
            depth_min=0.5, depth_max=4.0,
            interfaces=ifaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=120,
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs,R))
            if len(heap_R) > 120: heapq.heappop(heap_R)
            if time.time()>deadline: break
        R_candidates = [r for _fsneg,r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。進入可/Quality/ピッチを調整してください。")

        refined=[]
        for R in R_candidates[:120]:
            Fs = fs_given_R_multi(groundL, ifaces, soils, allow_cross, method, xc, yc, R, n_slices=40)
            if Fs is None: continue
            s = arc_sample_poly_best_pair(groundL, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(groundL, ifaces, soils, allow_cross, xc, yc, R, n_slices=40)
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_t - Fs)*D_sum)
            refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined: return dict(error="Refineで有効弧なし。設定/Quality/ピッチを見直してください。")
        refined.sort(key=lambda d:d["Fs"])
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        return dict(center=(xc,yc), refined=refined, idx_minFs=idx_minFs)

    if st.button("▶ 計算開始（未補強）"):
        sync_grid_ui_to_cfg()
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        cfg_set("results.unreinforced", res)
        xc,yc = res["center"]; d = res["refined"][res["idx_minFs"]]
        cfg_set("results.chosen_arc", dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"]))
        st.success("未補強結果を保存しました（cfg.results）。")

    res = cfg_get("results.unreinforced")
    if res:
        xc,yc = res["center"]; refined=res["refined"]; idx_minFs=res["idx_minFs"]
        fig,ax = plt.subplots(figsize=(10.0,7.0))
        Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
        draw_water(ax, ground, Xd, Yg)
        for d in refined[:30]:
            xs=np.linspace(d["x1"], d["x2"], 200); ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
            clipped=clip_yfloor(xs, ys, 0.0)
            if clipped is None: continue
            xs_c,ys_c = clipped
            ax.plot(xs_c, ys_c, lw=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))
        d=refined[idx_minFs]
        xs=np.linspace(d["x1"], d["x2"], 400); ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped=clip_yfloor(xs, ys, 0.0)
        if clipped is not None:
            xs_c,ys_c = clipped
            ax.plot(xs_c, ys_c, lw=3.0, color=(0.9,0,0), label=f"Min Fs = {d['Fs']:.3f}")
            y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
            ax.plot([xc,xs_c[0]],[yc,y1], lw=1.1, color=(0.9,0,0), alpha=0.9)
            ax.plot([xc,xs_c[-1]],[yc,y2], lw=1.1, color=(0.9,0,0), alpha=0.9)
        set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
        ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_t:.2f} • pitch={pitch:.2f}m")
        st.pyplot(fig); plt.close(fig)

# ===================== Page4: ネイル配置 =====================
elif page.startswith("4"):
    st = ST  # 念のため毎回復旧

    H, L, ground = make_ground_from_cfg()

    # レイヤー定義
    n_layers = int(cfg_get("layers.n"))
    interfaces = []
    if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

    st.subheader("ソイルネイル配置（頭位置＋軸＋ボンド可視化）")

    # chosen_arc を確定
    arc = cfg_get("results.chosen_arc")
    if not arc:
        res_un = cfg_get("results.unreinforced")
        if res_un and "center" in res_un and "refined" in res_un and res_un["refined"]:
            xc0, yc0 = res_un["center"]
            idx0 = res_un.get("idx_minFs", int(np.argmin([d["Fs"] for d in res_un["refined"]])))
            d0 = res_un["refined"][idx0]
            arc = dict(xc=xc0, yc=yc0, R=d0["R"], x1=d0["x1"], x2=d0["x2"], Fs=d0["Fs"])
            cfg_set("results.chosen_arc", arc)
        else:
            st.info("未補強MinFs円弧が未確定です。Page3で実行してください。")
            st.stop()

    # --- UI seed
    nails = cfg_get("nails")
    ui_seed("s_start", nails["s_start"]); ui_seed("s_end", nails["s_end"])
    ui_seed("S_surf", nails["S_surf"]);   ui_seed("S_row", nails["S_row"])
    ui_seed("tiers", nails["tiers"])
    ui_seed("angle_mode", nails["angle_mode"])
    ui_seed("beta_deg", nails["beta_deg"]); ui_seed("delta_beta", nails["delta_beta"])
    ui_seed("L_mode", nails["L_mode"]); ui_seed("L_nail", nails["L_nail"]); ui_seed("d_embed", nails["d_embed"])

    # 測地長 s
    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seg = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = float(s_cum[-1])

    # 入力UI
    st.slider("s_start (m)", 0.0, s_total, step=0.5, key="s_start", value=float(st.session_state["s_start"]))
    st.slider("s_end (m)", st.session_state["s_start"], s_total, step=0.5, key="s_end", value=float(st.session_state["s_end"]))
    st.slider("斜面ピッチ S_surf (m)", 0.5, 5.0, step=0.1, key="S_surf", value=float(st.session_state["S_surf"]))
    st.slider("段間隔 S_row (法線方向 m) [未実装]", 0.5, 5.0, step=0.5, key="S_row", value=float(st.session_state["S_row"]))
    st.number_input("段数（表示のみ）", 1, 5, step=1, key="tiers", value=int(st.session_state["tiers"]))

    st.radio("角度モード", ["Slope-Normal (⊥斜面)", "Horizontal-Down (β°)"],
             key="angle_mode",
             index=["Slope-Normal (⊥斜面)","Horizontal-Down (β°)"].index(st.session_state["angle_mode"]))
    if st.session_state["angle_mode"].endswith("β°"):
        st.slider("β（水平から下向き °）", 0.0, 45.0, step=1.0, key="beta_deg", value=float(st.session_state["beta_deg"]))
    else:
        st.slider("法線からの微調整 ±Δβ（°）", -10.0, 10.0, step=1.0, key="delta_beta", value=float(st.session_state["delta_beta"]))

    st.radio("長さモード", ["パターン1：固定長", "パターン2：すべり面より +Δm", "パターン3：FS目標で自動"],
             key="L_mode",
             index=["パターン1：固定長","パターン2：すべり面より +Δm","パターン3：FS目標で自動"].index(st.session_state["L_mode"]))
    if st.session_state["L_mode"]=="パターン1：固定長":
        st.slider("ネイル長 L (m)", 1.0, 15.0, step=0.5, key="L_nail", value=float(st.session_state["L_nail"]))
    elif st.session_state["L_mode"]=="パターン2：すべり面より +Δm":
        st.slider("すべり面より +Δm (m)", 0.0, 5.0, step=0.5, key="d_embed", value=float(st.session_state["d_embed"]))

    # s→x 補間
    def x_at_s(sv):
        idx = np.searchsorted(s_cum, sv, side="right") - 1
        idx = max(0, min(idx, len(Xd) - 2))
        t = (sv - s_cum[idx]) / (seg[idx] if seg[idx] > 1e-12 else 1e-12)
        return float((1 - t) * Xd[idx] + t * Xd[idx + 1])

    # 自動配置（頭）
    s_vals = list(np.arange(st.session_state["s_start"], st.session_state["s_end"] + 1e-9, st.session_state["S_surf"]))
    nail_heads = [(x_at_s(sv), float(ground.y_at(x_at_s(sv)))) for sv in s_vals]
    cfg_set("results.nail_heads", nail_heads)

    if st.button("💾 ネイル設定を保存（cfgへ）"):
        cfg_set("nails.s_start", float(st.session_state["s_start"]))
        cfg_set("nails.s_end", float(st.session_state["s_end"]))
        cfg_set("nails.S_surf", float(st.session_state["S_surf"]))
        cfg_set("nails.S_row", float(st.session_state["S_row"]))
        cfg_set("nails.tiers", int(st.session_state["tiers"]))
        cfg_set("nails.angle_mode", st.session_state["angle_mode"])
        cfg_set("nails.beta_deg", float(st.session_state.get("beta_deg", 15.0)))
        cfg_set("nails.delta_beta", float(st.session_state.get("delta_beta", 0.0)))
        cfg_set("nails.L_mode", st.session_state["L_mode"])
        cfg_set("nails.L_nail", float(st.session_state.get("L_nail", 5.0)))
        cfg_set("nails.d_embed", float(st.session_state.get("d_embed", 1.0)))
        st.success("cfgに保存しました。")

    # 図化
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd2,Yg2 = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd2, Yg2)

    xc, yc, R = arc["xc"], arc["yc"], arc["R"]
    xs = np.linspace(arc["x1"], arc["x2"], 400)
    ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")

    NH = cfg_get("results.nail_heads", [])
    if NH:
        ax.scatter([p[0] for p in NH], [p[1] for p in NH],
                   s=30, color="tab:blue", label=f"Nail heads ({len(NH)})")

    # 軸＆ボンド（常に地山側へ）
    try:
        angle_mode = cfg_get("nails.angle_mode")
        beta_deg   = float(cfg_get("nails.beta_deg", 15.0))
        delta_beta = float(cfg_get("nails.delta_beta", 0.0))
        L_mode     = cfg_get("nails.L_mode")
        L_nail     = float(cfg_get("nails.L_nail", 5.0))
        d_embed    = float(cfg_get("nails.d_embed", 1.0))

        for (xh, yh) in NH:
            if str(angle_mode).startswith("Slope-Normal"):
                theta = inward_normal_angle(ground, xh, delta_beta)
            else:
                theta = -beta_deg * DEG

            t = nail_ray_hit_circle(xh, yh, theta, xc, yc, R)
            ct, snt = math.cos(theta), math.sin(theta)

            if t is None:
                ax.plot([xh, xh + ct * L_nail], [yh, yh + snt * L_nail],
                        color="tab:blue", lw=1.5, alpha=0.4)
                continue

            xq, yq = xh + ct * t, yh + snt * t
            ax.plot([xh, xq], [yh, yq], color="tab:blue", lw=1.8, alpha=0.9)

            Lb = (max(0.0, d_embed) if str(L_mode).startswith("パターン2")
                  else max(0.0, L_nail - t))
            if Lb > 1e-3:
                xb2, yb2 = xq + ct * Lb, yq + snt * Lb
                ax.plot([xq, xb2], [yq, yb2], color="tab:green", lw=2.2, alpha=0.9)
    except Exception as e:
        ST.warning(f"nail drawing skipped: {e}")

    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    st = ST
    H, L, ground = make_ground_from_cfg()
    st.subheader("補強後解析（簡易合成）")
    arc = cfg_get("results.chosen_arc")
    NH = cfg_get("results.nail_heads", [])

    if not (arc and NH):
        missing=[]
        if not arc: missing.append("Page3のMin Fs円弧")
        if not NH:  missing.append("Page4のネイル頭配置")
        st.info("必要情報: " + "、".join(missing))
        st.stop()

    Fs0 = float(arc["Fs"])
    # D_sum 取得（未補強分母）
    packD = driving_sum_for_R_multi(
        *make_ground_from_cfg()[2:],  # ground は不要なので捨て値防止
        # 上の * は使いにくいので、下で改めて呼び出し
    )
    # ↑ うまくいかないので素直に再作成
    H_, L_, ground_ = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n"))
    ifaces=[]
    if n_layers>=2: ifaces.append(make_interface1_example(H_,L_))
    if n_layers>=3: ifaces.append(make_interface2_example(H_,L_))
    mats = cfg_get("layers.mat")
    soils=[Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
    allow_cross=[]
    if n_layers>=2:
        soils.append(Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross2")))
    if n_layers>=3:
        soils.append(Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross3")))
    packD = driving_sum_for_R_multi(ground_, ifaces, soils, allow_cross, arc["xc"], arc["yc"], arc["R"], n_slices=40)
    if packD is None:
        st.error("D_sum が取得できませんでした。")
        st.stop()
    D_sum, _, _ = packD
    N0 = Fs0 * D_sum

    # 材料パラメータ
    tau_cap = float(cfg_get("layers.tau_grout_cap_kPa", 150.0))
    d_g     = float(cfg_get("layers.d_g", 0.125))
    d_s     = float(cfg_get("layers.d_s", 0.022))
    fy      = float(cfg_get("layers.fy", 1000.0))
    gamma_m = float(cfg_get("layers.gamma_m", 1.20))

    # ネイル効果
    angle_mode = cfg_get("nails.angle_mode")
    beta_deg   = float(cfg_get("nails.beta_deg", 15.0))
    delta_beta = float(cfg_get("nails.delta_beta", 0.0))
    L_mode     = cfg_get("nails.L_mode")
    L_nail     = float(cfg_get("nails.L_nail", 5.0))
    d_embed    = float(cfg_get("nails.d_embed", 1.0))

    T_total = 0.0
    for (xh, yh) in NH:
        theta = (inward_normal_angle(ground_, xh, delta_beta)
                 if str(angle_mode).startswith("Slope-Normal")
                 else -beta_deg * DEG)
        t = nail_ray_hit_circle(xh, yh, theta, arc["xc"], arc["yc"], arc["R"])
        if t is None:
            continue
        Lb = (max(0.0, d_embed) if str(L_mode).startswith("パターン2")
              else max(0.0, L_nail - t))
        if Lb <= 1e-6:
            continue
        T_pull = grout_pullout_capacity_kNm_per_m(d_g, Lb, tau_cap)
        T_steel= steel_tension_capacity_kN_per_m(d_s, fy, gamma_m)
        T_total += min(T_pull, T_steel)

    Fs_after = (N0 + T_total) / D_sum if D_sum > 0 else np.nan

    cfg_set("results.reinforced", {
        "n_nails": len(NH),
        "arc_Fs_unreinforced": Fs0,
        "Fs_after": float(Fs_after),
        "T_total": float(T_total),
    })

    # 可視化
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd,Yg = draw_layers_and_ground(ax, ground_, n_layers, ifaces)
    draw_water(ax, ground_, Xd, Yg)
    xs = np.linspace(arc["x1"], arc["x2"], 400)
    ys = arc["yc"] - np.sqrt(np.maximum(0.0, arc["R"]**2 - (xs - arc["xc"])**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Slip arc (Fs0={Fs0:.3f} → {Fs_after:.3f})")

    # ネイル描画（命中は緑のボンド表示）
    if NH:
        ax.scatter([p[0] for p in NH], [p[1] for p in NH],
                   s=30, color="tab:blue", label=f"Nail heads ({len(NH)})")

    for (xh, yh) in NH:
        theta = (inward_normal_angle(ground_, xh, delta_beta)
                 if str(angle_mode).startswith("Slope-Normal")
                 else -beta_deg * DEG)
        ct, snt = math.cos(theta), math.sin(theta)
        t = nail_ray_hit_circle(xh, yh, theta, arc["xc"], arc["yc"], arc["R"])
        if t is None:
            ax.plot([xh, xh + ct * L_nail], [yh, yh + snt * L_nail],
                    color="tab:blue", lw=1.5, alpha=0.4)
            continue
        xq, yq = xh + ct * t, yh + snt * t
        ax.plot([xh, xq], [yh, yq], color="tab:blue", lw=1.8, alpha=0.9)
        Lb = (max(0.0, d_embed) if str(L_mode).startswith("パターン2")
              else max(0.0, L_nail - t))
        if Lb > 1e-3:
            xb2, yb2 = xq + ct * Lb, yq + snt * Lb
            ax.plot([xq, xb2], [yq, yb2], color="tab:green", lw=2.2, alpha=0.9)

    set_axes(ax, H_, L_, ground_); ax.grid(True); ax.legend()

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("ネイル本数", f"{len(NH)}")
    with col2: st.metric("未補強Fs", f"{Fs0:.3f}")
    with col3: st.metric("補強後Fs", f"{Fs_after:.3f}")

    st.pyplot(fig); plt.close(fig)