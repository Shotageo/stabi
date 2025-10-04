# streamlit_app.py — Stabi 安定板２＋ソイルネイル描画＋補強後Fs近似（展示会用MVP）

from __future__ import annotations
import os, sys, math, time, heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---- stabi パッケージを親から解決（ローカル/Cloud両対応） ----
_CUR = os.path.dirname(__file__)
_PARENT = os.path.dirname(_CUR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ---- コア演算モジュール ----
import stabi.stabi_lem as lem
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

# ---- 画面設定 ----
st.set_page_config(page_title="Stabi LEM｜安定板２＋SoilNail", layout="wide")
plt.rcParams["figure.autolayout"] = True

# ===================== cfg（正本） =====================
def default_cfg():
    return {
        "geom": {"H": 25.0, "L": 60.0},
        "water": {"mode": "WT", "ru": 0.0, "offset": -2.0, "wl_points": None},
        "layers": {
            "n": 3,
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
            "unreinforced": None,   # {"center":(xc,yc),"refined":[...],"idx_minFs":int}
            "chosen_arc": None,
            "nail_heads": [],
            "reinforced": None,
        }
    }

# ---- セッションにcfg確保 ----
if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

# ---- cfgユーティリティ ----
def _maybe_int_key(p):
    if isinstance(p, str) and p.isdigit():
        try: return int(p)
        except Exception: return p
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
    node = st.session_state["cfg"]
    parts = path.split(".")
    for p in parts[:-1]:
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node: node = node[p]
            elif p_try in node: node = node[p_try]
            else:
                node[p_try] = {}
                node = node[p_try]
        else:
            raise KeyError(f"cfg_set: '{p}' below is not a dict")
    last = _maybe_int_key(parts[-1])
    if isinstance(node, dict): node[last] = value
    else: raise KeyError(f"cfg_set: cannot set at '{parts[-1]}'")

def ui_seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# ---- 補助（地形・表示） ----
def make_ground_from_cfg():
    H = float(cfg_get("geom.H")); L = float(cfg_get("geom.L"))
    return H, L, make_ground_example(H,L)

def set_axes(ax, H, L, ground):
    x_upper = max(1.18*L, float(ground.X[-1])+0.05*L, 100.0)
    y_upper = max(2.30*H, 0.05*H+2.0*H, 100.0)
    ax.set_xlim(min(0.0-0.05*L, -2.0), x_upper)
    ax.set_ylim(0.0, y_upper)
    ax.set_aspect("equal", adjustable="box")

def fs_to_color(fs):
    if fs < 1.0: return (0.85,0,0)
    if fs < 1.2:
        t=(fs-1.0)/0.2; return (1.0,0.50+0.50*t,0.0)
    return (0.0,0.55,0.0)

def clip_yfloor(xs, ys, y_floor=0.0):
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2: return None
    return xs[m], ys[m]

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

# ===================== Nail helpers（描画・容量・投影） =====================
def _slope_dydx_at_x(ground: GroundPL, x: float) -> float:
    X, Y = ground.X, ground.Y
    if x <= X[0]:  i = 0
    elif x >= X[-1]: i = len(X)-2
    else:
        i = int(np.searchsorted(X, x) - 1)
        i = max(0, min(i, len(X)-2))
    dx = X[i+1]-X[i]
    dy = Y[i+1]-Y[i]
    return float(dy/max(dx, 1e-12))

def _unit_normal_from_slope(dydx: float, delta_deg: float = 0.0) -> np.ndarray:
    theta_t = math.atan2(dydx, 1.0)
    cand = [theta_t - math.pi/2.0, theta_t + math.pi/2.0]
    theta_n = min(cand, key=lambda th: math.sin(th))
    theta_n = theta_n + (delta_deg * math.pi/180.0)
    return np.array([math.cos(theta_n), math.sin(theta_n)], dtype=float)

def _unit_vec_horizontal_down(beta_deg: float) -> np.ndarray:
    th = -beta_deg * math.pi/180.0
    return np.array([math.cos(th), math.sin(th)], dtype=float)

def _ray_circle_forward_intersection(P: np.ndarray, v: np.ndarray,
                                     xc: float, yc: float, R: float,
                                     x1_arc: float, x2_arc: float) -> tuple[bool, np.ndarray | None]:
    px, py = P; vx, vy = v
    a = vx*vx + vy*vy
    b = 2*((px - xc)*vx + (py - yc)*vy)
    c = (px - xc)**2 + (py - yc)**2 - R*R
    disc = b*b - 4*a*c
    if disc < 0: return False, None
    s = math.sqrt(max(0.0, disc))
    ts = [t for t in ((-b - s)/(2*a), (-b + s)/(2*a)) if t >= 0.0]
    if not ts: return False, None
    ts.sort()
    for t in ts:
        qx = px + t*vx; qy = py + t*vy
        if x1_arc - 1e-6 <= qx <= x2_arc + 1e-6:
            inside = R*R - (qx - xc)**2
            if inside >= -1e-6:
                y_expected = yc - math.sqrt(max(0.0, inside))
                if abs(qy - y_expected) <= 1e-3:
                    return True, np.array([qx, qy], dtype=float)
    return False, None

def build_nails_geometry(ground: GroundPL,
                         nail_heads: list[tuple[float,float]],
                         arc: dict | None,
                         angle_mode: str, beta_deg: float, delta_beta: float,
                         L_mode: str, L_nail: float, d_embed: float):
    out = []
    if not nail_heads: return out
    xc = yc = R = x1 = x2 = None
    if arc:
        xc, yc, R = arc["xc"], arc["yc"], arc["R"]
        x1, x2    = arc["x1"], arc["x2"]
    for (xh, yh) in nail_heads:
        if angle_mode.startswith("Slope-Normal"):
            dydx = _slope_dydx_at_x(ground, float(xh))
            v = _unit_normal_from_slope(dydx, delta_beta)
        else:
            v = _unit_vec_horizontal_down(beta_deg)
        hit_arc = False; Q = None
        if arc is not None:
            hit_arc, Q = _ray_circle_forward_intersection(
                np.array([xh, yh], dtype=float), v, xc, yc, R, x1, x2
            )
        if L_mode.startswith("パターン1"):
            tip = np.array([xh, yh]) + v * float(L_nail)
        elif L_mode.startswith("パターン2") and hit_arc:
            tip = (Q if Q is not None else np.array([xh, yh])) + v * float(max(0.0, d_embed))
        else:
            tip = np.array([xh, yh]) + v * float(L_nail)
        out.append(dict(head=(float(xh), float(yh)),
                        tip=(float(tip[0]), float(tip[1])),
                        xing=(None if Q is None else (float(Q[0]), float(Q[1]))),
                        hit_arc=bool(hit_arc)))
    return out

# ---- capacity / Fs近似 ----
def _area_steel(d_s: float) -> float:
    r = 0.5 * d_s
    return math.pi * r * r

def _layer_index_at_depth(ground: GroundPL, interfaces: list[GroundPL], x: float, y: float) -> int:
    if not interfaces:
        return 1
    for j, pl in enumerate(interfaces, start=1):
        yb = float(pl.y_at(x))
        if y >= yb - 1e-9:
            return j
    return len(interfaces) + 1

def _tau_eff_kPa_at_point(layers_cfg: dict, layer_idx: int) -> float:
    tau_cap = float(layers_cfg.get("tau_grout_cap_kPa", 150.0))
    mats = layers_cfg["mat"]
    tau_layer = float(mats[layer_idx]["tau"])
    return float(min(tau_layer, tau_cap))

def _bond_length_from_geom(head, tip, xing, mode: str, L_nail: float, d_embed: float) -> float:
    hx, hy = head; tx, ty = tip
    total = math.hypot(tx - hx, ty - hy)
    if xing is None:
        return float(max(0.0, total))
    qx, qy = xing
    hq = math.hypot(qx - hx, qy - hy)
    if mode.startswith("パターン2"):
        return float(max(0.0, total - hq + d_embed))
    else:
        return float(max(0.0, total - hq))

def _alpha_at_x_on_arc(xc: float, yc: float, R: float, x: float) -> float:
    inside = R*R - (x - xc)**2
    inside = inside if inside > 0 else 1e-12
    y_arc = yc - math.sqrt(inside)
    denom = (y_arc - yc)
    denom = denom if abs(denom) > 1e-12 else (math.copysign(1e-12, denom))
    dydx = - (x - xc) / denom
    return -math.atan(dydx)

def _angle_of_vec(vx: float, vy: float) -> float:
    return math.atan2(vy, vx)

def _proj_stabilizing(T_cap: float, theta_nail: float, alpha_tangent: float) -> float:
    dtheta = theta_nail - alpha_tangent
    c = math.cos(dtheta)
    return float(max(0.0, T_cap * c))

def compute_T_add_for_arc(ground: GroundPL, interfaces: list[GroundPL], layers_cfg: dict,
                          arc: dict, nails_geom: list[dict],
                          d_g: float, d_s: float, fy_MPa: float, gamma_m: float):
    xc, yc, R = arc["xc"], arc["yc"], arc["R"]
    details = []
    T_add_sum = 0.0
    As = _area_steel(d_s)
    Fy = fy_MPa * 1e6
    T_steel = Fy * As / max(gamma_m, 1e-6)      # N
    T_steel_kN = T_steel / 1000.0               # kN
    for g in nails_geom:
        hx, hy = g["head"]; tx, ty = g["tip"]; q = g["xing"]
        vx, vy = (tx - hx), (ty - hy)
        L = math.hypot(vx, vy)
        if L < 1e-9: continue
        theta_nail = _angle_of_vec(vx, vy)
        px, py = (q if q is not None else (tx, ty))
        layer_idx = _layer_index_at_depth(ground, interfaces, px, py)
        tau_eff_kPa = _tau_eff_kPa_at_point(layers_cfg, layer_idx)
        tau_eff_Pa = tau_eff_kPa * 1000.0
        L_bond = _bond_length_from_geom((hx,hy),(tx,ty), q,
                                        mode=cfg_get("nails.L_mode"),
                                        L_nail=cfg_get("nails.L_nail"),
                                        d_embed=cfg_get("nails.d_embed"))
        T_bond = tau_eff_Pa * math.pi * d_g * L_bond
        T_bond_kN = T_bond / 1000.0
        T_cap_kN = min(T_bond_kN, T_steel_kN)
        alpha = _alpha_at_x_on_arc(xc, yc, R, (q[0] if q is not None else hx))
        T_add_i = _proj_stabilizing(T_cap_kN, theta_nail, alpha)
        T_add_sum += T_add_i
        details.append(dict(
            head=(hx,hy), tip=(tx,ty), xing=q, layer=layer_idx,
            tau_eff_kPa=float(tau_eff_kPa),
            L_bond=float(L_bond),
            T_bond_kN=float(T_bond_kN),
            T_steel_kN=float(T_steel_kN),
            T_cap_kN=float(T_cap_kN),
            alpha_deg=float(alpha*180.0/math.pi),
            theta_nail_deg=float(theta_nail*180.0/math.pi),
            T_add_kN=float(T_add_i),
        ))
    return float(T_add_sum), details

# ===================== サイドバー =====================
with st.sidebar:
    st.header("Pages")
    page = st.radio("", ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"], key="__page__")
    st.caption("cfgが正本。保存しない限り自動上書きしません。")
    if st.button("⚠ すべて初期化（cfgを再作成）"):
        st.session_state["cfg"] = default_cfg()
        st.success("初期化しました。")

# ===================== Page1: 地形・水位 =====================
if page.startswith("1"):
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

    H_ui = float(st.session_state["H"]); L_ui = float(st.session_state["L"])
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

    if str(cfg_get("water.mode")).startswith("WT"):
        W = cfg_get("water.wl_points")
        if W is not None:
            W = np.asarray(W, dtype=float)
            Yw = np.interp(Xd, W[:,0], W[:,1], left=W[0,1], right=W[-1,1])
            Yw = np.clip(Yw, 0.0, Yg)
            ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (saved)")
        else:
            off = float(st.session_state["wt_offset"])
            Yw_off = np.clip(Yg + off, 0.0, Yg)
            ax.plot(Xd, Yw_off, "-.", color="tab:blue", label="WT (offset preview)")

    set_axes(ax, H_ui, L_ui, ground_ui)
    ax.grid(True); ax.legend()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

# ===================== Page2: 地層・材料 =====================
elif page.startswith("2"):
    if cfg_get("layers.mat") is None:
        cfg_set("layers.mat", default_cfg()["layers"]["mat"])
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
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(make_interface2_example(H,L))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page3: 円弧探索（未補強） =====================
# ===================== Page3: 円弧探索（未補強） =====================
elif page.startswith("3"):
    H,L,ground = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(make_interface2_example(H,L))

    # 初期枠（未設定なら H/L から種）
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
            st.select_slider("Quality（表示のみ）", options=["Coarse","Normal","Fine","Very-fine"], key="p3_quality", value=st.session_state["p3_quality"])
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

    # グリッドの可視化
    x_min=cfg_get("grid.x_min"); x_max=cfg_get("grid.x_max")
    y_min=cfg_get("grid.y_min"); y_max=cfg_get("grid.y_max")
    pitch=cfg_get("grid.pitch")
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

       # ▶ 計算開始（未補強）ボタン — ここで保存＆描画
    if st.button("▶ 計算開始（未補強）"):
        sync_grid_ui_to_cfg()
        method = cfg_get("grid.method")

        # 層構成（代表値）：いまは上から順に最大3層を許容
        mats = cfg_get("layers.mat")
        n_layers = int(cfg_get("layers.n"))
        soils = [Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
        allow_cross = []
        if n_layers >= 2:
            soils.append(Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
            allow_cross.append(bool(cfg_get("grid.allow_cross2")))
        if n_layers >= 3:
            soils.append(Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
            allow_cross.append(bool(cfg_get("grid.allow_cross3")))

        # 中心候補を複数サンプリング（範囲の 7×5 格子）
        xs_c = np.linspace(x_min, x_max, 7)
        ys_c = np.linspace(y_min, y_max, 5)

        # 走査パラメータ（“安定板２”のクイック探索近似）
        quick_slices = 12
        n_entries    = 240       # 円弧エントリ数（x側サンプリング密度）
        depth_min    = 0.5
        depth_max    = 4.0       # 必要なら 6.0〜8.0 に上げると当たりやすい
        limit_arcs   = 160       # 中心1点あたりの最大本数
        probe_n_min  = 101

        best = None  # (Fs, pack) を保持
        # interfaces は描画用に既に用意済み
        interfaces=[]
        if n_layers>=2: interfaces.append(make_interface1_example(H,L))
        if n_layers>=3: interfaces.append(make_interface2_example(H,L))

        # 複数中心を走査して、成立する円弧の中から最小Fsを選ぶ
        for yc_try in ys_c:
            for xc_try in xs_c:
                # クイック探索：この中心で候補Rを大量に生成
                R_list = []
                for _x1,_x2,R,Fs_q in lem.arcs_from_center_by_entries_multi(
                        ground, soils, float(xc_try), float(yc_try),
                        n_entries=n_entries, method="Fellenius",
                        depth_min=depth_min, depth_max=depth_max,
                        interfaces=interfaces, allow_cross=allow_cross,
                        quick_mode=True, n_slices_quick=quick_slices,
                        limit_arcs_per_center=limit_arcs, probe_n_min=probe_n_min):
                    R_list.append(R)

                if not R_list:
                    continue  # この中心では円弧が作れない→次の中心へ

                # 候補Rで本計算（指定法）→最小Fsを採用
                refined=[]
                for R in R_list:
                    Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross, method, float(xc_try), float(yc_try), float(R), n_slices=40)
                    if Fs is None: 
                        continue
                    s = arc_sample_poly_best_pair(ground, float(xc_try), float(yc_try), float(R), n=251, y_floor=0.0)
                    if s is None: 
                        continue
                    x1,x2,*_ = s
                    refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2)))

                if not refined:
                    continue

                refined.sort(key=lambda d: d["Fs"])
                cand = refined[0]
                score = cand["Fs"]
                if (best is None) or (score < best[0]):
                    best = (score, dict(center=(float(xc_try), float(yc_try)), refined=refined, idx_minFs=0))

        if best is None:
            st.error("この設定では有効な円弧が見つかりませんでした。中心範囲やdepth_max（例: 6〜8m）、Allow into Layer を見直してください。")
        else:
            Fs_min, pack = best
            cfg_set("results.unreinforced", pack)
            xc,yc = pack["center"]; d = pack["refined"][pack["idx_minFs"]]
            cfg_set("results.chosen_arc", dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"]))
            st.success(f"未補強の結果を保存しました（Fs={d['Fs']:.3f} @ Center=({xc:.2f},{yc:.2f})）。")



# ===================== Page4: ネイル配置 =====================
elif page.startswith("4"):
    H,L,ground = make_ground_from_cfg()

    n_layers = int(cfg_get("layers.n"))
    interfaces = []
    if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

    st.subheader("ソイルネイル配置（頭位置→軸線プレビュー）")

    arc = cfg_get("results.chosen_arc")
    if not arc:
        res_un = cfg_get("results.unreinforced")
        if res_un and "center" in res_un and "refined" in res_un and res_un["refined"]:
            xc,yc = res_un["center"]
            idx = res_un.get("idx_minFs", int(np.argmin([d["Fs"] for d in res_un["refined"]])))
            d = res_un["refined"][idx]
            arc = dict(xc=xc, yc=yc, R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"])
            cfg_set("results.chosen_arc", arc)
        else:
            st.info("未補強の Min Fs 円弧が未確定です。Page3 で実行してから来てください。")
            st.stop()

    nails = cfg_get("nails")
    ui_seed("s_start", nails["s_start"]); ui_seed("s_end", nails["s_end"])
    ui_seed("S_surf", nails["S_surf"]);   ui_seed("S_row", nails["S_row"])
    ui_seed("tiers", nails["tiers"])
    ui_seed("angle_mode", nails["angle_mode"])
    ui_seed("beta_deg", nails["beta_deg"]); ui_seed("delta_beta", nails["delta_beta"])
    ui_seed("L_mode", nails["L_mode"]); ui_seed("L_nail", nails["L_nail"]); ui_seed("d_embed", nails["d_embed"])

    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seg = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = float(s_cum[-1])

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

    def x_at_s(sv):
        idx = np.searchsorted(s_cum, sv, side="right")-1
        idx = max(0, min(idx, len(Xd)-2))
        t = (sv - s_cum[idx]) / (seg[idx] if seg[idx]>1e-12 else 1e-12)
        return float((1-t)*Xd[idx] + t*Xd[idx+1])

    s_vals = list(np.arange(st.session_state["s_start"], st.session_state["s_end"]+1e-9, st.session_state["S_surf"]))
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

    fig,ax = plt.subplots(figsize=(10.0,7.0))
    Xd2,Yg2 = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd2, Yg2)

    xc,yc,R = arc["xc"],arc["yc"],arc["R"]
    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")

    NH = cfg_get("results.nail_heads", [])
    if NH:
        angle_mode = st.session_state["angle_mode"]; beta_deg = float(st.session_state.get("beta_deg", 15.0))
        delta_beta = float(st.session_state.get("delta_beta", 0.0))
        L_mode = st.session_state["L_mode"]; L_nail = float(st.session_state.get("L_nail", 5.0))
        d_embed = float(st.session_state.get("d_embed", 1.0))
        geom = build_nails_geometry(ground, NH, arc, angle_mode, beta_deg, delta_beta, L_mode, L_nail, d_embed)
        for g in geom:
            hx, hy = g["head"]; tx, ty = g["tip"]
            ax.plot([hx, tx], [hy, ty], lw=1.6, color=("tab:blue" if g["hit_arc"] else "0.6"),
                    alpha=(1.0 if g["hit_arc"] else 0.7))
            if g["xing"] is not None:
                qx, qy = g["xing"]
                ax.plot(qx, qy, "x", ms=6, mew=1.5, color="tab:orange")

    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    H,L,ground = make_ground_from_cfg()
    st.subheader("補強後解析（近似：ΣT_add / D_sum）")

    arc = cfg_get("results.chosen_arc")
    NH = cfg_get("results.nail_heads", [])
    if not (arc and NH):
        missing=[]
        if not arc: missing.append("Page3のMin Fs円弧")
        if not NH:  missing.append("Page4のネイル頭配置")
        st.info("必要情報: " + "、".join(missing))
        st.stop()

    n_layers = int(cfg_get("layers.n"))
    interfaces = []
    if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

    # 材料セット
    layers_cfg = cfg_get("layers")
    d_g = float(layers_cfg["d_g"]); d_s = float(layers_cfg["d_s"])
    fy  = float(layers_cfg["fy"]); gamma_m = float(layers_cfg["gamma_m"])

    nails_cfg = cfg_get("nails")
    angle_mode = nails_cfg["angle_mode"]
    beta_deg   = float(nails_cfg["beta_deg"])
    delta_beta = float(nails_cfg["delta_beta"])
    L_mode     = nails_cfg["L_mode"]
    L_nail     = float(nails_cfg["L_nail"])
    d_embed    = float(nails_cfg["d_embed"])

    geom = build_nails_geometry(ground, NH, arc,
                                angle_mode=angle_mode, beta_deg=beta_deg, delta_beta=delta_beta,
                                L_mode=L_mode, L_nail=L_nail, d_embed=d_embed)

    # D_sum（未補強）を取得（簡易：1層物性で代表）
    soils = [Soil(cfg_get("layers.mat.1")["gamma"], cfg_get("layers.mat.1")["c"], cfg_get("layers.mat.1")["phi"])]
    allow_cross=[]
    D_pack = driving_sum_for_R_multi(ground, interfaces[:max(0, len(soils)-1)], soils, allow_cross, arc["xc"], arc["yc"], arc["R"], n_slices=40)
    D_sum = float(D_pack[0]) if D_pack is not None else None

    # ΣT_add
    T_add_kN, details = compute_T_add_for_arc(ground, interfaces, layers_cfg, arc, geom, d_g=d_g, d_s=d_s, fy_MPa=fy, gamma_m=gamma_m)

    Fs_un = float(arc["Fs"])
    Fs_after = Fs_un + (T_add_kN / D_sum if (D_sum and D_sum>1e-9) else 0.0)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("未補強 Fs", f"{Fs_un:.3f}")
    with col2: st.metric("ΣT_add (kN)", f"{T_add_kN:.1f}")
    with col3: st.metric("補強後 Fs（近似）", f"{Fs_after:.3f}")

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd, Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)

    xc, yc, R = arc["xc"], arc["yc"], arc["R"]
    xs = np.linspace(arc["x1"], arc["x2"], 400)
    ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Slip arc（Fs_un={Fs_un:.3f} → Fs≈{Fs_after:.3f}）")

    # ネイル（律速で色分け）
    for g, d in zip(geom, details if len(details)==len(geom) else geom):
        hx, hy = g["head"]; tx, ty = g["tip"]
        color = ("tab:green" if isinstance(d, dict) and d.get("T_bond_kN",0.0) > d.get("T_steel_kN",1e9) else "tab:blue")
        ax.plot([hx, tx], [hy, ty], lw=1.8, color=color, alpha=0.95)
        if g["xing"] is not None:
            qx, qy = g["xing"]
            ax.plot(qx, qy, "x", ms=6, mew=1.5, color="tab:orange")

    set_axes(ax, H, L, ground)
    ax.grid(True); ax.legend()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"横断図｜Fs_un={Fs_un:.3f}  →  Fs_after≈{Fs_after:.3f}  （ΣT_add={T_add_kN:.1f} kN）")
    st.pyplot(fig); plt.close(fig)
