# streamlit_app.py — cfg一本化で“保存→リセット”根絶版（WTは保存最優先／ページ跨ぎで値維持）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM｜安定UI（cfg一本化）", layout="wide")
st.title("Stabi LEM｜多段UI（設定は cfg に一元保存）")

# ===================== cfg（正本）とUIキーの関係 =====================
def default_cfg():
    return {
        "geom": {"H": 25.0, "L": 60.0},
        "water": {"mode": "WT", "ru": 0.0, "offset": -2.0, "wl_points": None},  # wl_points最優先
        "layers": {
            "n": 3,
            "mat": {  # top→bottom
                1: {"gamma": 18.0, "c": 5.0, "phi": 30.0, "tau": 150.0},
                2: {"gamma": 19.0, "c": 8.0, "phi": 28.0, "tau": 180.0},
                3: {"gamma": 20.0, "c": 12.0, "phi": 25.0, "tau": 200.0},
            },
            "tau_grout_cap_kPa": 150.0,
            "d_g": 0.125,  # m
            "d_s": 0.022,  # m
            "fy": 1000.0, "gamma_m": 1.20, "mu": 0.0,
        },
        "grid": {
            "x_min": None, "x_max": None, "y_min": None, "y_max": None,  # 初回はH,Lから自動種
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
            "unreinforced": None,  # {"center":(xc,yc),"refined":[...],"idx_minFs":int}
            "chosen_arc": None,
            "nail_heads": [],
            "reinforced": None,
        }
    }

def cfg_get(path, default=None):
    """path: 'section.key' or 'section.sub.key'"""
    node = st.session_state["cfg"]
    for p in path.split("."):
        if p not in node: return default
        node = node[p]
    return node

def cfg_set(path, value):
    node = st.session_state["cfg"]; parts = path.split(".")
    for p in parts[:-1]:
        if p not in node: node[p]={}
        node = node[p]
    node[parts[-1]] = value

def ui_seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# ===================== 起動時一度だけ：cfgを用意 =====================
if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

# ===================== 共有小物 =====================
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=900,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  coarse_subsample="every 3rd",
                   coarse_entries=160, coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.6, budget_quick_s=0.9),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, coarse_subsample="every 2nd",
                   coarse_entries=220, coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1700, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, coarse_subsample="full",
                 coarse_entries=320, coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.2, budget_quick_s=1.8),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, coarse_subsample="full",
                      coarse_entries=420, coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.8, budget_quick_s=2.6),
}

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
    if W is not None and isinstance(W, np.ndarray) and W.ndim==2 and W.shape[1]==2:
        Yw = np.interp(Xd, W[:,0], W[:,1], left=W[0,1], right=W[-1,1])
        Yw = np.clip(Yw, 0.0, Yg)
        ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (saved)")
    else:
        off = float(cfg_get("water.offset",-2.0))
        Yw = np.clip(Yg + off, 0.0, Yg)
        ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (offset preview)")

# ===================== サイドバー =====================
with st.sidebar:
    st.header("Pages")
    page = st.radio("", ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"], key="__page__")
    st.caption("cfgに保存された値が正本。保存しない限り自動上書きはしません。")
    if st.button("⚠ すべて初期化（cfgを再作成）"):
        st.session_state["cfg"] = default_cfg()
        # UIキーはそのままでもOK。必要なら次の各ページでseedします。
        st.success("初期化しました。")

# ===================== Page1: 地形・水位 =====================
if page.startswith("1"):
    # UI seed（注意：seedは“なければ入れる”だけ／cfgは書き換えない）
    ui_seed("H", cfg_get("geom.H"))
    ui_seed("L", cfg_get("geom.L"))
    ui_seed("water_mode", cfg_get("water.mode"))
    ui_seed("ru", cfg_get("water.ru"))
    ui_seed("wt_offset", cfg_get("water.offset"))

    st.subheader("Geometry")
    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H")
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L")
    # 表示は常に cfg ベース（正本）
    H,L,ground = make_ground_from_cfg()

    st.subheader("Water model")
    st.selectbox("Water model", ["WT","ru","WT+ru"], key="water_mode")
    st.slider("r_u (if ru mode)", 0.0, 0.9, step=0.05, key="ru")
    st.slider("Water level offset from ground (m, negative=below)", -30.0, 5.0, step=0.5, key="wt_offset")

    c1,c2 = st.columns(2)
    with c1:
        if st.button("💾 形状・水位パラメータを保存（cfgへ）"):
            # 形状・水位の“数値”だけ保存。wl_pointsは更新しない（ユーザの明示操作でのみ更新）
            cfg_set("geom.H", float(st.session_state["H"]))
            cfg_set("geom.L", float(st.session_state["L"]))
            cfg_set("water.mode", st.session_state["water_mode"])
            cfg_set("water.ru", float(st.session_state["ru"]))
            cfg_set("water.offset", float(st.session_state["wt_offset"]))
            # grid 初期枠（未設定時のみH,Lから種）
            if cfg_get("grid.x_min") is None:
                cfg_set("grid.x_min", 0.25*cfg_get("geom.L"))
                cfg_set("grid.x_max", 1.15*cfg_get("geom.L"))
                cfg_set("grid.y_min", 1.60*cfg_get("geom.H"))
                cfg_set("grid.y_max", 2.20*cfg_get("geom.H"))
            st.success("cfgに保存しました（wl_pointsは変更していません）。")
    with c2:
        if st.button("💾 WT水位線を offset から生成/更新（cfg.water.wl_points）"):
            H,L,ground = make_ground_from_cfg()
            Xd = np.linspace(ground.X[0], ground.X[-1], 400)
            Yg = np.array([float(ground.y_at(x)) for x in Xd])
            off = float(st.session_state["wt_offset"])
            Yw = np.clip(Yg + off, 0.0, Yg)
            cfg_set("water.wl_points", np.vstack([Xd, Yw]).T)
            st.success("水位線をcfgに保存しました（以後この線が最優先で使われます）。")

    # プレビュー（cfg正本を描画）
    H,L,ground = make_ground_from_cfg()
    fig,ax = plt.subplots(figsize=(9.6,5.8))
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(make_interface2_example(H,L))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

# ===================== Page2: 地層・材料 =====================
elif page.startswith("2"):
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
    st.selectbox("Number of layers", [1,2,3], key="n_layers")

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Layer1 (top)**")
        st.number_input("γ₁", 10.0, 25.0, step=0.5, key="gamma1")
        st.number_input("c₁", 0.0, 200.0, step=0.5, key="c1")
        st.number_input("φ₁", 0.0, 45.0, step=0.5, key="phi1")
        st.number_input("τ₁ (kPa)", 0.0, 2000.0, step=10.0, key="tau1")
    if st.session_state["n_layers"]>=2:
        with cols[1]:
            st.markdown("**Layer2**")
            st.number_input("γ₂", 10.0, 25.0, step=0.5, key="gamma2")
            st.number_input("c₂", 0.0, 200.0, step=0.5, key="c2")
            st.number_input("φ₂", 0.0, 45.0, step=0.5, key="phi2")
            st.number_input("τ₂ (kPa)", 0.0, 2000.0, step=10.0, key="tau2")
    if st.session_state["n_layers"]>=3:
        with cols[2]:
            st.markdown("**Layer3 (bottom)**")
            st.number_input("γ₃", 10.0, 25.0, step=0.5, key="gamma3")
            st.number_input("c₃", 0.0, 200.0, step=0.5, key="c3")
            st.number_input("φ₃", 0.0, 45.0, step=0.5, key="phi3")
            st.number_input("τ₃ (kPa)", 0.0, 2000.0, step=10.0, key="tau3")
    with cols[-1]:
        st.markdown("**Grout / Nail**")
        st.number_input("τ_grout_cap (kPa)", 0.0, 5000.0, step=10.0, key="tau_grout_cap_kPa")
        st.number_input("削孔(=グラウト)径 d_g (mm)", 50, 300, step=1, key="d_g_mm")
        st.number_input("鉄筋径 d_s (mm)", 10, 50, step=1, key="d_s_mm")
        st.number_input("引張強さ fy (MPa)", 200.0, 2000.0, step=50.0, key="fy")
        st.number_input("材料安全率 γ_m", 1.00, 2.00, step=0.05, key="gamma_m")
        st.select_slider("逓減係数 μ（0〜0.9, 0.1刻み）", options=[round(0.1*i,1) for i in range(10)], key="mu")

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

    # プレビュー
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
elif page.startswith("3"):
    H,L,ground = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(make_interface2_example(H,L))

    # UI seed（grid系）
    def seed_grid_if_none():
        if cfg_get("grid.x_min") is None:
            cfg_set("grid.x_min", 0.25*L); cfg_set("grid.x_max", 1.15*L)
            cfg_set("grid.y_min", 1.60*H); cfg_set("grid.y_max", 2.20*H)
    seed_grid_if_none()
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
            st.number_input("x min (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_min")
            st.number_input("x max (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_max")
            st.number_input("y min (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_min")
            st.number_input("y max (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_max")
            st.number_input("Center-grid ピッチ (m)", min_value=0.1, step=0.1, format="%.2f", key="p3_pitch")
        with colB:
            st.selectbox("Method", ["Bishop (simplified)","Fellenius"], key="p3_method")
            st.select_slider("Quality", options=list(QUALITY.keys()), key="p3_quality")
            st.number_input("Target FS", min_value=1.00, max_value=2.00, step=0.05, format="%.2f", key="p3_Fs_t")
        if n_layers>=2: st.checkbox("Allow into Layer 2", key="p3_allow2")
        if n_layers>=3: st.checkbox("Allow into Layer 3", key="p3_allow3")
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
        st.success("cfgに保存しました（cfgは正本なのでリセットしません）。")

    # 可視化（cfg正本）
    x_min=cfg_get("grid.x_min"); x_max=cfg_get("grid.x_max")
    y_min=cfg_get("grid.y_min"); y_max=cfg_get("grid.y_max")
    pitch=cfg_get("grid.pitch")
    method=cfg_get("grid.method"); quality=cfg_get("grid.quality"); Fs_t=float(cfg_get("grid.Fs_target"))

    fig,ax = plt.subplots(figsize=(10.0,6.8))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    gx = np.arange(x_min, x_max+1e-9, pitch); gy = np.arange(y_min, y_max+1e-9, pitch)
    if gx.size<1: gx=np.array([x_min]); 
    if gy.size<1: gy=np.array([y_min])
    xs=[float(x) for x in gx for _ in gy]; ys=[float(y) for y in gy for _ in gx]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    ax.plot([x_min,x_max,x_max,x_min,x_min],[y_min,y_min,y_max,y_max,y_min], c="k", lw=1.0, alpha=0.4)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend(loc="upper right")
    st.pyplot(fig); plt.close(fig)

    # soils & allow_cross（cfg→計算）
    mats = cfg_get("layers.mat")
    soils=[Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
    allow_cross=[]
    if n_layers>=2:
        soils.append(Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross2")))
    if n_layers>=3:
        soils.append(Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross3")))
    P = QUALITY[quality].copy()

    def compute_once():
        Hc,Lc,groundL = make_ground_from_cfg()
        ifaces=[]
        if n_layers>=2: ifaces.append(make_interface1_example(Hc,Lc))
        if n_layers>=3: ifaces.append(make_interface2_example(Hc,Lc))

        def subsampled():
            xs = np.arange(x_min, x_max+1e-9, pitch)
            ys = np.arange(y_min, y_max+1e-9, pitch)
            tag = P["coarse_subsample"]
            if tag=="every 3rd":
                xs = xs[::3] if xs.size>2 else xs; ys = ys[::3] if ys.size>2 else ys
            elif tag=="every 2nd":
                xs = xs[::2] if xs.size>1 else xs; ys = ys[::2] if ys.size>1 else ys
            return [(float(xc),float(yc)) for yc in ys for xc in xs]

        def pick_center(budget_s):
            deadline = time.time()+budget_s; best=None
            for (xc,yc) in subsampled():
                cnt=0; Fs_min=None
                for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                    groundL, soils, xc, yc,
                    n_entries=P["coarse_entries"], method="Fellenius",
                    depth_min=0.5, depth_max=4.0,
                    interfaces=ifaces, allow_cross=allow_cross,
                    quick_mode=True, n_slices_quick=max(8,P["quick_slices"]//2),
                    limit_arcs_per_center=P["coarse_limit_arcs"],
                    probe_n_min=P["coarse_probe_min"],
                ):
                    cnt+=1
                    if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                    if time.time()>deadline: break
                score=(cnt, -(Fs_min if Fs_min is not None else 1e9))
                if (best is None) or (score>best[0]): best=(score,(xc,yc))
                if time.time()>deadline: break
            return (best[1] if best else None)

        center = pick_center(P["budget_coarse_s"])
        if center is None: return dict(error="Coarseで候補なし。範囲/ピッチ/深さを見直してください。")
        xc,yc = center

        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            groundL, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=0.5, depth_max=4.0,
            interfaces=ifaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs,R))
            if len(heap_R) > max(P["show_k"],20): heapq.heappop(heap_R)
            if time.time()>deadline: break
        R_candidates = [r for _fsneg,r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。進入可/Quality/ピッチを調整してください。")

        refined=[]
        for R in R_candidates[:P["show_k"]]:
            Fs = fs_given_R_multi(groundL, ifaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
            if Fs is None: continue
            s = arc_sample_poly_best_pair(groundL, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(groundL, ifaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_t - Fs)*D_sum)
            refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined: return dict(error="Refineで有効弧なし。設定/Quality/ピッチを見直してください。")
        refined.sort(key=lambda d:d["Fs"])
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        return dict(center=(xc,yc), refined=refined, idx_minFs=idx_minFs)

    # 計算開始（直前にUI→cfg同期してから計算）
    if st.button("▶ 計算開始（未補強）"):
        sync_grid_ui_to_cfg()
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        cfg_set("results.unreinforced", res)
        xc,yc = res["center"]; d = res["refined"][res["idx_minFs"]]
        cfg_set("results.chosen_arc", dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"]))
        st.success("未補強結果を保存しました（cfg.results）。")

    # 結果図（cfg正本から）
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
    H,L,ground = make_ground_from_cfg()
    st.subheader("ソイルネイル配置（試作：頭位置のみ）")
    arc = cfg_get("results.chosen_arc")
    if not arc:
        st.info("Page3でMin Fs円弧を確定してから来てね。"); st.stop()

    # UI seed
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

    st.slider("s_start (m)", 0.0, s_total, step=0.5, key="s_start")
    st.slider("s_end (m)", st.session_state["s_start"], s_total, step=0.5, key="s_end")
    st.slider("斜面ピッチ S_surf (m)", 0.5, 5.0, step=0.1, key="S_surf")
    st.slider("段間隔 S_row (法線方向 m) [未実装]", 0.5, 5.0, step=0.5, key="S_row")
    st.number_input("段数（表示のみ）", 1, 5, step=1, key="tiers")
    st.radio("角度モード", ["Slope-Normal (⊥斜面)", "Horizontal-Down (β°)"], key="angle_mode")
    if st.session_state["angle_mode"].endswith("β°"):
        st.slider("β（水平から下向き °）", 0.0, 45.0, step=1.0, key="beta_deg")
    else:
        st.slider("法線からの微調整 ±Δβ（°）", -10.0, 10.0, step=1.0, key="delta_beta")
    st.radio("長さモード", ["パターン1：固定長", "パターン2：すべり面より +Δm", "パターン3：FS目標で自動"], key="L_mode")
    if st.session_state["L_mode"]=="パターン1：固定長":
        st.slider("ネイル長 L (m)", 1.0, 15.0, step=0.5, key="L_nail")
    elif st.session_state["L_mode"]=="パターン2：すべり面より +Δm":
        st.slider("すべり面より +Δm (m)", 0.0, 5.0, step=0.5, key="d_embed")

    def x_at_s(sv):
        idx = np.searchsorted(s_cum, sv, side="right")-1
        idx = max(0, min(idx, len(Xd)-2))
        t = (sv - s_cum[idx]) / (seg[idx] if seg[idx]>1e-12 else 1e-12)
        return float((1-t)*Xd[idx] + t*Xd[idx+1])

    s_vals = list(np.arange(st.session_state["s_start"], st.session_state["s_end"]+1e-9, st.session_state["S_surf"]))
    nail_heads = [(x_at_s(sv), float(ground.y_at(x_at_s(sv)))) for sv in s_vals]
    cfg_set("results.nail_heads", nail_heads)  # 表示と同時に保存

    # 保存ボタン（ネイル設定一括）
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

    # 表示
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(make_interface1_example(H, L))
    if n_layers>=3: interfaces.append(make_interface2_example(H, L))
    fig,ax = plt.subplots(figsize=(10.0,7.0))
    Xd2,Yg2 = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd2, Yg2)
    arc = cfg_get("results.chosen_arc")
    xc,yc,R = arc["xc"],arc["yc"],arc["R"]
    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")
    NH = cfg_get("results.nail_heads", [])
    if NH:
        ax.scatter([p[0] for p in NH], [p[1] for p in NH], s=30, color="tab:blue", label=f"Nail heads ({len(NH)})")
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    H,L,ground = make_ground_from_cfg()
    st.subheader("補強後解析（試作）")
    arc = cfg_get("results.chosen_arc")
    NH = cfg_get("results.nail_heads", [])
    btn = st.button("▶ 補強後の計算を実行", disabled=not (arc and NH))
    if not (arc and NH):
        missing=[]
        if not arc: missing.append("Page3のMin Fs円弧")
        if not NH: missing.append("Page4のネイル頭配置")
        st.info("必要情報: " + "、".join(missing))
    elif btn:
        with st.spinner("（将来）ネイル効果を連成計算中…"):
            cfg_set("results.reinforced", {
                "n_nails": len(NH),
                "arc_Fs_unreinforced": arc["Fs"],
                "note": "Phase-2で Tpullout/Tstrip(μ)/Ttens → Tt/Tn投影 → FS更新を実装予定。",
            })
    r = cfg_get("results.reinforced")
    if r:
        col1,col2 = st.columns(2)
        with col1: st.metric("ネイル本数", f"{r['n_nails']}")
        with col2: st.metric("未補強Fs（参照）", f"{r['arc_Fs_unreinforced']:.3f}")
        st.caption(r["note"])
