# streamlit_app.py — Stabi LEM 多段UI（安定＋水位クリップ＋ピッチ指定＋径mm＋フォーム保存）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, json
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

# （任意）教授コメントモジュール：なくてもOK
try:
    from stabi_suggest import (
        lint_geometry_and_water, lint_soils_and_materials, lint_arc_and_slices, lint_nails_layout,
        render_suggestions, default_dispatcher
    )
    HAS_SUGG = True
except Exception:
    HAS_SUGG = False
    def render_suggestions(*args, **kwargs): pass
    def lint_geometry_and_water(*args, **kwargs): return []
    def lint_soils_and_materials(*args, **kwargs): return []
    def lint_arc_and_slices(*args, **kwargs): return []
    def lint_nails_layout(*args, **kwargs): return []

st.set_page_config(page_title="Stabi LEM｜安定UI", layout="wide")
st.title("Stabi LEM｜多段UI（安定版）")

# =========================
# Session State 初期化＋欠損補填
# =========================
def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def init_defaults():
    # Page1: geometry & water
    _ss_default("H", 25.0)          # m
    _ss_default("L", 60.0)          # m
    _ss_default("water_mode", "WT") # "WT" | "ru" | "WT+ru"
    _ss_default("ru", 0.0)          # 0..0.9
    _ss_default("wt_offset", -2.0)  # m（地表からの相対）
    _ss_default("wl_points", None)  # np.ndarray[[x,y],...]

    # Page2: layers & materials
    _ss_default("n_layers", 3)
    _ss_default("gamma1", 18.0); _ss_default("c1", 5.0);  _ss_default("phi1", 30.0); _ss_default("tau1", 150.0)
    _ss_default("gamma2", 19.0); _ss_default("c2", 8.0);  _ss_default("phi2", 28.0); _ss_default("tau2", 180.0)
    _ss_default("gamma3", 20.0); _ss_default("c3", 12.0); _ss_default("phi3", 25.0); _ss_default("tau3", 200.0)
    # Nail/Grout（内部単位は m / kPa / MPa）
    _ss_default("tau_grout_cap_kPa", 150.0)
    _ss_default("d_g", 0.125)   # m（UIはmmで入力→mに反映）
    _ss_default("d_s", 0.022)   # m
    _ss_default("fy", 1000.0)   # MPa
    _ss_default("gamma_m", 1.20)
    _ss_default("mu", 0.0)

    # Page3: grid & method（絶対値で保持）
    _ss_default("x_min_abs", 0.25 * st.session_state["L"])
    _ss_default("x_max_abs", 1.15 * st.session_state["L"])
    _ss_default("y_min_abs", 1.60 * st.session_state["H"])
    _ss_default("y_max_abs", 2.20 * st.session_state["H"])
    _ss_default("grid_pitch_m", 5.0)  # 中心グリッドのピッチ(m)
    _ss_default("method", "Bishop (simplified)")
    _ss_default("quality", "Normal")
    _ss_default("Fs_target", 1.20)
    _ss_default("allow_cross2", True)
    _ss_default("allow_cross3", True)

    # Page4: nails layout（試作）
    _ss_default("s_start", 5.0)
    _ss_default("s_end",   35.0)
    _ss_default("S_surf",  2.0)
    _ss_default("S_row",   2.0)
    _ss_default("tiers",   1)
    _ss_default("angle_mode", "Slope-Normal (⊥斜面)")
    _ss_default("beta_deg", 15.0)
    _ss_default("delta_beta", 0.0)
    _ss_default("L_mode", "パターン1：固定長")
    _ss_default("L_nail", 5.0)
    _ss_default("d_embed", 1.0)

    # 結果置き場
    _ss_default("chosen_arc", None)
    _ss_default("res3", None)
    _ss_default("result_reinforced", None)

if "_initialized" not in st.session_state:
    init_defaults()
    st.session_state["_initialized"] = True

# 欠損補填（method切替等のrerunでも安全）
def ensure_core_keys():
    base = {
        "H": 25.0, "L": 60.0, "water_mode": "WT", "ru": 0.0, "wt_offset": -2.0, "wl_points": None,
        "n_layers": 3,
        "gamma1": 18.0, "c1": 5.0, "phi1": 30.0, "tau1": 150.0,
        "gamma2": 19.0, "c2": 8.0, "phi2": 28.0, "tau2": 180.0,
        "gamma3": 20.0, "c3": 12.0, "phi3": 25.0, "tau3": 200.0,
        "tau_grout_cap_kPa": 150.0, "d_g": 0.125, "d_s": 0.022, "fy": 1000.0, "gamma_m": 1.20, "mu": 0.0,
        "x_min_abs": 0.25 * st.session_state.get("L", 60.0),
        "x_max_abs": 1.15 * st.session_state.get("L", 60.0),
        "y_min_abs": 1.60 * st.session_state.get("H", 25.0),
        "y_max_abs": 2.20 * st.session_state.get("H", 25.0),
        "grid_pitch_m": 5.0, "method": "Bishop (simplified)", "quality": "Normal", "Fs_target": 1.20,
        "allow_cross2": True, "allow_cross3": True,
        "s_start": 5.0, "s_end": 35.0, "S_surf": 2.0, "S_row": 2.0, "tiers": 1,
        "angle_mode": "Slope-Normal (⊥斜面)", "beta_deg": 15.0, "delta_beta": 0.0,
        "L_mode": "パターン1：固定長", "L_nail": 5.0, "d_embed": 1.0,
        "chosen_arc": None, "res3": None, "result_reinforced": None,
    }
    for k,v in base.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_core_keys()

# mm UI 同期（内部m→mm）
if "d_g_mm" not in st.session_state:
    st.session_state["d_g_mm"] = int(round(st.session_state["d_g"] * 1000))
if "d_s_mm" not in st.session_state:
    st.session_state["d_s_mm"] = int(round(st.session_state["d_s"] * 1000))

# ヘルパ
def HL_ground():
    H = float(st.session_state.get("H", 25.0))
    L = float(st.session_state.get("L", 60.0))
    if "H" not in st.session_state: st.session_state["H"] = H
    if "L" not in st.session_state: st.session_state["L"] = L
    return H, L, make_ground_example(H, L)

def ensure_waterline(ground: GroundPL):
    Xd = np.linspace(ground.X[0], ground.X[-1], 200, dtype=float)
    Yg = np.array([float(ground.y_at(x)) for x in Xd], dtype=float)
    offset = st.session_state.get("wt_offset", -2.0)
    Yw = np.clip(Yg + offset, 0.0, Yg)  # 0..地表にクリップ
    wl = np.vstack([Xd, Yw]).T
    st.session_state["wl_points"] = wl
    return wl

def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def clip_yfloor(xs: np.ndarray, ys: np.ndarray, y_floor: float = 0.0):
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2: return None
    return xs[m], ys[m]

def set_axes_fixed(ax, H: float, L: float, ground: GroundPL):
    x_upper = max(1.18*L, float(ground.X[-1]) + 0.05*L, 100.0)
    y_upper = max(2.30*H, 0.05*H + 2.0*H, 100.0)
    ax.set_xlim(min(0.0 - 0.05*L, -2.0), x_upper)
    ax.set_ylim(0.0, y_upper)
    ax.set_aspect("equal", adjustable="box")

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

# =========================
# 画面ナビ
# =========================
page = st.sidebar.radio(
    "Pages",
    ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"],
)

# =========================
# Page 1: 地形・水位
# =========================
if page.startswith("1"):
    H, L, ground = HL_ground()

    colL, colR = st.columns([3,1])
    with colL:
        st.subheader("Geometry")
        st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H")
        st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L")
        H, L, ground = HL_ground()

        st.subheader("Water")
        st.selectbox("Water model", ["WT", "ru", "WT+ru"], key="water_mode")
        st.slider("r_u (if ru mode)", 0.0, 0.9, step=0.05, key="ru")

        # オフセットWT（0..地表にクリップ）
        st.slider("Water level offset from ground (m, negative=below)", -30.0, 5.0, step=0.5, key="wt_offset")
        Xd = np.linspace(ground.X[0], ground.X[-1], 200)
        Yg = np.array([float(ground.y_at(x)) for x in Xd])
        Yw_raw = Yg + st.session_state["wt_offset"]
        Yw = np.clip(Yw_raw, 0.0, Yg)  # 0〜地表
        st.session_state["wl_points"] = np.vstack([Xd, Yw]).T

        # 図
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Soil")
        if st.session_state["water_mode"].startswith("WT"):
            ax.plot(Xd, Yw, linestyle="-.", color="tab:blue", label="WT (offset, clipped 0..Ground)")
        set_axes_fixed(ax, H, L, ground)
        ax.grid(True); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        st.pyplot(fig); plt.close(fig)

    with colR:
        st.subheader("教授コメント")
        render_suggestions(
            lint_geometry_and_water(ground, st.session_state["water_mode"], st.session_state["ru"], st.session_state["wl_points"]),
            key_prefix="p1", dispatcher=default_dispatcher
        ) if HAS_SUGG else st.caption("（教授コメント: モジュール未読込）")

# =========================
# Page 2: 地層・材料（ネイル径UIはmm）
# =========================
elif page.startswith("2"):
    H, L, ground = HL_ground()

    st.subheader("Layers & Materials")

    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H")
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L")
    H, L, ground = HL_ground()

    st.selectbox("Number of layers", [1,2,3], key="n_layers")

    interfaces = []
    if st.session_state["n_layers"] >= 2: interfaces.append(make_interface1_example(H, L))
    if st.session_state["n_layers"] >= 3: interfaces.append(make_interface2_example(H, L))

    cols = st.columns(4)
    soils_tbl = []
    with cols[0]:
        st.markdown("**Layer1 (top)**")
        st.number_input("γ₁", 10.0, 25.0, step=0.5, key="gamma1")
        st.number_input("c₁", 0.0, 200.0, step=0.5, key="c1")
        st.number_input("φ₁", 0.0, 45.0, step=0.5, key="phi1")
        st.number_input("τ₁ (kPa)", 0.0, 1000.0, step=10.0, key="tau1")
        soils_tbl.append(dict(gamma=st.session_state["gamma1"], c=st.session_state["c1"], phi=st.session_state["phi1"], tau_kPa=st.session_state["tau1"]))
    if st.session_state["n_layers"] >= 2:
        with cols[1]:
            st.markdown("**Layer2**")
            st.number_input("γ₂", 10.0, 25.0, step=0.5, key="gamma2")
            st.number_input("c₂", 0.0, 200.0, step=0.5, key="c2")
            st.number_input("φ₂", 0.0, 45.0, step=0.5, key="phi2")
            st.number_input("τ₂ (kPa)", 0.0, 1000.0, step=10.0, key="tau2")
            soils_tbl.append(dict(gamma=st.session_state["gamma2"], c=st.session_state["c2"], phi=st.session_state["phi2"], tau_kPa=st.session_state["tau2"]))
    if st.session_state["n_layers"] >= 3:
        with cols[2]:
            st.markdown("**Layer3 (bottom)**")
            st.number_input("γ３", 10.0, 25.0, step=0.5, key="gamma3")
            st.number_input("c３", 0.0, 200.0, step=0.5, key="c3")
            st.number_input("φ３", 0.0, 45.0, step=0.5, key="phi3")
            st.number_input("τ３ (kPa)", 0.0, 1000.0, step=10.0, key="tau3")
            soils_tbl.append(dict(gamma=st.session_state["gamma3"], c=st.session_state["c3"], phi=st.session_state["phi3"], tau_kPa=st.session_state["tau3"]))
    with cols[-1]:
        st.markdown("**Grout / Nail**")
        st.number_input("τ_grout_cap (kPa)", 0.0, 2000.0, step=10.0, key="tau_grout_cap_kPa")

        # mm入力 → mに反映（UIはmm、内部はm）
        st.number_input("削孔(=グラウト)径 d_g (mm)", 50, 300, step=1, key="d_g_mm")
        st.session_state["d_g"] = float(st.session_state["d_g_mm"]) / 1000.0

        st.number_input("鉄筋径 d_s (mm)", 10, 50, step=1, key="d_s_mm")
        st.session_state["d_s"] = float(st.session_state["d_s_mm"]) / 1000.0

        st.number_input("引張強さ fy (MPa=kN/m²)", 200.0, 2000.0, step=50.0, key="fy")
        st.number_input("材料安全率 γ_m", 1.00, 2.00, step=0.05, key="gamma_m")
        st.select_slider("逓減係数 μ（0〜0.9, 0.1刻み。μ=1.0はStrip無視）",
                         options=[round(0.1*i,1) for i in range(10)], key="mu")

    # 可視化
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if st.session_state["n_layers"]==1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif st.session_state["n_layers"]==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    if st.session_state["n_layers"]>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.0)
    if st.session_state["n_layers"]>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.0)
    set_axes_fixed(ax, H, L, ground)
    ax.grid(True); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    if HAS_SUGG:
        render_suggestions(lint_soils_and_materials(soils_tbl, st.session_state["tau_grout_cap_kPa"]),
                           key_prefix="p2", dispatcher=default_dispatcher)

# =========================
# Page 3: 円弧探索（未補強） — フォーム保存で揺れ防止
# =========================
elif page.startswith("3"):
    H, L, ground = HL_ground()
    st.subheader("円弧探索（未補強）")

    # 現在値
    x_min0 = float(st.session_state["x_min_abs"])
    x_max0 = float(st.session_state["x_max_abs"])
    y_min0 = float(st.session_state["y_min_abs"])
    y_max0 = float(st.session_state["y_max_abs"])
    pitch0 = float(st.session_state["grid_pitch_m"])
    method0 = st.session_state["method"]
    quality0 = st.session_state["quality"]
    Fs_t0 = float(st.session_state["Fs_target"])
    allow2_0 = bool(st.session_state["allow_cross2"])
    allow3_0 = bool(st.session_state["allow_cross3"])

    # ---- フォーム：確定時だけ state を更新（切替rerunで値が揺れない）
    with st.form("arc_params"):
        colA, colB = st.columns([1.3, 1])
        with colA:
            x_min = st.number_input("x min (m)", value=x_min0, step=max(0.1, 0.05*L), format="%.3f")
            x_max = st.number_input("x max (m)", value=x_max0, step=max(0.1, 0.05*L), format="%.3f")
            y_min = st.number_input("y min (m)", value=y_min0, step=max(0.1, 0.10*H), format="%.3f")
            y_max = st.number_input("y max (m)", value=y_max0, step=max(0.1, 0.10*H), format="%.3f")
            pitch = st.number_input("Center-grid ピッチ (m)", value=pitch0, min_value=0.1, step=0.1, format="%.2f")
            st.caption(f"ヒント: 参考レンジ x∈[{0.2*L:.1f},{4.0*L:.1f}], y∈[{0.8*H:.1f},{8.0*H:.1f}]")
        with colB:
            method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"], index=["Bishop (simplified)","Fellenius"].index(method0))
            quality = st.select_slider("Quality", options=list(QUALITY.keys()), value=quality0)
            Fs_t = st.number_input("Target FS", value=Fs_t0, min_value=1.00, max_value=2.00, step=0.05, format="%.2f")

        allow2 = allow2_0
        allow3 = allow3_0
        if st.session_state["n_layers"] >= 2: allow2 = st.checkbox("Allow into Layer 2", value=allow2_0)
        if st.session_state["n_layers"] >= 3: allow3 = st.checkbox("Allow into Layer 3", value=allow3_0)

        submitted = st.form_submit_button("🔁 設定を確定（保存）")

    if submitted:
        if x_max < x_min: x_min, x_max = x_max, x_min
        if y_max < y_min: y_min, y_max = y_max, y_min
        st.session_state["x_min_abs"] = float(x_min)
        st.session_state["x_max_abs"] = float(x_max)
        st.session_state["y_min_abs"] = float(y_min)
        st.session_state["y_max_abs"] = float(y_max)
        st.session_state["grid_pitch_m"] = float(max(0.1, pitch))
        st.session_state["method"] = method
        st.session_state["quality"] = quality
        st.session_state["Fs_target"] = float(Fs_t)
        st.session_state["allow_cross2"] = bool(allow2)
        st.session_state["allow_cross3"] = bool(allow3)
        st.success("設定を保存しました。")

    # 保存済みの値のみで以降を描画
    x_min = float(st.session_state["x_min_abs"]); x_max = float(st.session_state["x_max_abs"])
    y_min = float(st.session_state["y_min_abs"]); y_max = float(st.session_state["y_max_abs"])
    pitch = float(st.session_state["grid_pitch_m"])
    method = st.session_state["method"]; quality = st.session_state["quality"]; Fs_t = float(st.session_state["Fs_target"])

    allow_cross=[]
    if st.session_state["n_layers"]>=2: allow_cross.append(bool(st.session_state["allow_cross2"]))
    if st.session_state["n_layers"]>=3: allow_cross.append(bool(st.session_state["allow_cross3"]))

    # プレビュー（地形・層・水位・センターグリッド）
    interfaces = []
    if st.session_state["n_layers"] >= 2: interfaces.append(make_interface1_example(H, L))
    if st.session_state["n_layers"] >= 3: interfaces.append(make_interface2_example(H, L))

    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    if st.session_state["n_layers"]==1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif st.session_state["n_layers"]==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    if st.session_state["n_layers"]>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.0)
    if st.session_state["n_layers"]>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.0)

    wm = st.session_state.get("water_mode", "WT")
    wl = st.session_state.get("wl_points", None)
    if wm.startswith("WT"):
        if wl is None:
            wl = ensure_waterline(ground)
        ax.plot(wl[:,0], wl[:,1], linestyle="-.", color="tab:blue", alpha=0.8, label="WT (clipped)")

    # center-grid（ピッチから生成）
    gx_mid = np.arange(x_min, x_max + 1e-9, pitch)
    gy_mid = np.arange(y_min, y_max + 1e-9, pitch)
    if gx_mid.size < 2: gx_mid = np.array([x_min, x_max])
    if gy_mid.size < 2: gy_mid = np.array([y_min, y_max])
    xs = [float(x) for x in gx_mid for _ in gy_mid]
    ys = [float(y) for y in gy_mid for _ in gx_mid]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    # 外枠
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color="k", linewidth=1.0, alpha=0.4)

    set_axes_fixed(ax, H, L, ground)
    ax.grid(True); ax.legend(loc="upper right"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    # Soils（stabi_lem.Soil は 3引数版）
    soils = [Soil(st.session_state["gamma1"], st.session_state["c1"], st.session_state["phi1"])]
    if st.session_state["n_layers"] >= 2:
        soils.append(Soil(st.session_state["gamma2"], st.session_state["c2"], st.session_state["phi2"]))
    if st.session_state["n_layers"] >= 3:
        soils.append(Soil(st.session_state["gamma3"], st.session_state["c3"], st.session_state["phi3"]))

    # 品質パラメータ
    P = QUALITY[quality].copy()

    def compute_once():
        Hc, Lc = st.session_state["H"], st.session_state["L"]
        ground_local = make_ground_example(Hc, Lc)
        interfaces_local = []
        if st.session_state["n_layers"] >= 2: interfaces_local.append(make_interface1_example(Hc, Lc))
        if st.session_state["n_layers"] >= 3: interfaces_local.append(make_interface2_example(Hc, Lc))

        def subsampled_centers():
            xs = np.arange(x_min, x_max + 1e-9, pitch)
            ys = np.arange(y_min, y_max + 1e-9, pitch)
            if xs.size < 2: xs = np.array([x_min, x_max])
            if ys.size < 2: ys = np.array([y_min, y_max])
            tag = P["coarse_subsample"]
            if tag == "every 3rd":
                xs = xs[::3] if xs.size > 2 else xs
                ys = ys[::3] if ys.size > 2 else ys
            elif tag == "every 2nd":
                xs = xs[::2] if xs.size > 1 else xs
                ys = ys[::2] if ys.size > 1 else ys
            return [(float(xc), float(yc)) for yc in ys for xc in xs]

        def pick_center(budget_s):
            deadline = time.time() + budget_s
            best = None
            for (xc,yc) in subsampled_centers():
                cnt=0; Fs_min=None
                for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                    ground_local, soils, xc, yc,
                    n_entries=P["coarse_entries"], method="Fellenius",
                    depth_min=0.5, depth_max=4.0,
                    interfaces=interfaces_local, allow_cross=allow_cross,
                    quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                    limit_arcs_per_center=P["coarse_limit_arcs"],
                    probe_n_min=P["coarse_probe_min"],
                ):
                    cnt+=1
                    if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                    if time.time() > deadline: break
                score = (cnt, - (Fs_min if Fs_min is not None else 1e9))
                if (best is None) or (score > best[0]): best = (score, (xc,yc))
                if time.time() > deadline: break
            return (best[1] if best else None)

        center = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseで候補なし。枠/深さ/ピッチを見直してください。")
        xc, yc = center

        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground_local, soils, xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=0.5, depth_max=4.0,
            interfaces=interfaces_local, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Quality/ピッチを調整してください。")

        refined=[]
        for R in R_candidates[:P["show_k"]]:
            Fs = fs_given_R_multi(ground_local, interfaces_local, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
            if Fs is None: continue
            s = arc_sample_poly_best_pair(ground_local, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(ground_local, interfaces_local, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_t - Fs)*D_sum)
            refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined:
            return dict(error="Refineで有効弧なし。設定/Quality/ピッチを見直してください。")
        refined.sort(key=lambda d:d["Fs"])
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        return dict(center=(xc,yc), refined=refined, idx_minFs=idx_minFs)

    if st.button("▶ 計算開始（未補強）"):
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        st.session_state["res3"] = res
        # 採用円弧を保存（Page4/5用）
        xc,yc = res["center"]; d = res["refined"][res["idx_minFs"]]
        st.session_state["chosen_arc"] = dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"])

    # 結果描画
    if st.session_state["res3"]:
        res = st.session_state["res3"]
        xc,yc = res["center"]; refined=res["refined"]; idx_minFs=res["idx_minFs"]

        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        # 背景
        if st.session_state["n_layers"]==1:
            ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
        elif st.session_state["n_layers"]==2:
            Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1"); ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
        else:
            Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
            ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
        ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")

        # Refined arcs
        for d in refined[:30]:
            xs=np.linspace(d["x1"], d["x2"], 200)
            ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
            clipped = clip_yfloor(xs, ys, 0.0)
            if clipped is None: continue
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))

        # Min Fs
        d=refined[idx_minFs]
        xs=np.linspace(d["x1"], d["x2"], 400)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped = clip_yfloor(xs, ys, 0.0)
        if clipped is not None:
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
            y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
            ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
            ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)

        set_axes_fixed(ax, H, L, ground)
        ax.grid(True); ax.legend()
        ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_t:.2f} • pitch={pitch:.2f}m")
        st.pyplot(fig); plt.close(fig)

        if HAS_SUGG:
            render_suggestions(lint_arc_and_slices(hit_edge=False, n_slices=QUALITY[st.session_state["quality"]]["final_slices"]),
                               key_prefix="p3", dispatcher=default_dispatcher)

# =========================
# Page 4: ネイル配置（試作）
# =========================
elif page.startswith("4"):
    H, L, ground = HL_ground()

    st.subheader("ソイルネイル配置（試作段階：頭位置のみ可視化）")
    if not st.session_state["chosen_arc"]:
        st.info("Page3で補強対象の円弧（Min Fs）を確定させてから来てください。"); st.stop()

    arc = st.session_state["chosen_arc"]

    # 斜面測線
    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seglen = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
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
    if st.session_state["L_mode"] == "パターン1：固定長":
        st.slider("ネイル長 L (m)", 1.0, 15.0, step=0.5, key="L_nail")
    elif st.session_state["L_mode"] == "パターン2：すべり面より +Δm":
        st.slider("すべり面より +Δm (m)", 0.0, 5.0, step=0.5, key="d_embed")

    # s→x の補間
    def x_at_s(s_val: float) -> float:
        idx = np.searchsorted(s_cum, s_val, side="right")-1
        idx = max(0, min(idx, len(Xd)-2))
        ds = s_val - s_cum[idx]
        segS = seglen[idx] if seglen[idx]>1e-12 else 1e-12
        t = ds/segS
        return float((1-t)*Xd[idx] + t*Xd[idx+1])

    s_vals = list(np.arange(st.session_state["s_start"], st.session_state["s_end"]+1e-9, st.session_state["S_surf"]))
    nail_heads = [(x_at_s(sv), float(ground.y_at(x_at_s(sv)))) for sv in s_vals]

    # 図
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    Xp = np.linspace(ground.X[0], ground.X[-1], 600); Yp = np.array([float(ground.y_at(x)) for x in Xp])
    if st.session_state["n_layers"]==1:
        ax.fill_between(Xp, 0.0, Yp, alpha=0.12, label="Layer1")
    elif st.session_state["n_layers"]==2:
        Y1 = clip_interfaces_to_ground(ground, [make_interface1_example(H, L)], Xp)[0]
        ax.fill_between(Xp, Y1, Yp, alpha=0.12, label="Layer1"); ax.fill_between(Xp, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [make_interface1_example(H, L), make_interface2_example(H, L)], Xp)
        ax.fill_between(Xp, Y1, Yp, alpha=0.12, label="Layer1")
        ax.fill_between(Xp, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xp, 0.0, Y2, alpha=0.12, label="Layer3")
    # 円弧
    xc,yc,R = arc["xc"],arc["yc"],arc["R"]
    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")
    # ネイル頭
    ax.scatter([p[0] for p in nail_heads], [p[1] for p in nail_heads], s=30, color="tab:blue", label=f"Nail heads ({len(nail_heads)})")

    set_axes_fixed(ax, H, L, ground)
    ax.grid(True); ax.legend()
    ax.set_title("配置プレビュー（Phase-1：頭位置のみ）")
    st.pyplot(fig); plt.close(fig)

    if HAS_SUGG:
        stats = dict(has_outward=False, too_dense=False, n_Lo_short=0, dominant_mode="pullout")
        render_suggestions(lint_nails_layout(dict(S_surf=st.session_state["S_surf"],S_row=st.session_state["S_row"],
                                                  mode=st.session_state["angle_mode"],Lo_min=1.0), stats),
                           key_prefix="p4", dispatcher=default_dispatcher)

    st.session_state["nail_heads"] = nail_heads  # Page5用

# =========================
# Page 5: 補強後解析（ボタンあり）
# =========================
elif page.startswith("5"):
    H, L, ground = HL_ground()

    st.subheader("補強後解析（Phase-2でFS連成予定）")

    ok_arc = st.session_state["chosen_arc"] is not None
    ok_heads = "nail_heads" in st.session_state and st.session_state["nail_heads"]

    btn_disabled = not (ok_arc and ok_heads)
    btn = st.button("▶ 補強後の計算を実行", disabled=btn_disabled)

    if btn_disabled:
        missing = []
        if not ok_arc: missing.append("補強対象の円弧（Page3のMin Fs）")
        if not ok_heads: missing.append("ネイル頭の配置（Page4）")
        st.warning("実行に必要な情報が足りません：" + "、".join(missing))
    elif btn:
        with st.spinner("補強後FS（試作）を計算中…"):
            res = {
                "n_nails": len(st.session_state.get("nail_heads", [])),
                "arc_Fs_unreinforced": st.session_state["chosen_arc"]["Fs"],
                "note": "Phase-2で Tpullout/Tstrip(μ)/Ttens → Tt/Tn投影 → FS連成を実装します。",
            }
            st.session_state["result_reinforced"] = res

    if st.session_state["result_reinforced"]:
        r = st.session_state["result_reinforced"]
        st.success("補強後計算（試作）を実行しました。")
        col1,col2 = st.columns(2)
        with col1: st.metric("ネイル本数", f"{r['n_nails']}")
        with col2: st.metric("未補強Fs（参照）", f"{r['arc_Fs_unreinforced']:.3f}")
        st.caption(r["note"])
        st.info("※本ページはPhase-2で：ネイル毎のTpullout/Tstrip/Ttens 動員比、Tt/Tn投影、FS更新・比較表、最適化サジェストなどを実装します。")