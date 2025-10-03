# streamlit_app.py — 全体安定版（セッション一貫・水位は保存最優先・KeyError根絶・ページ跨ぎで値は保持）
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

st.set_page_config(page_title="Stabi LEM｜安定UI", layout="wide")
st.title("Stabi LEM｜多段UI（安定版）")

# ===================== セッション管理（存在保証と一度だけ初期化） =====================
def ss_get(k, default):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]

def init_session_defaults():
    # Page1: 形状・水位（初期）
    st.session_state.update(dict(
        H=25.0, L=60.0,
        water_mode="WT",   # "WT" | "ru" | "WT+ru"
        ru=0.0,
        wt_offset=-2.0,
        wl_points=None,    # 保存された水位線（優先）
    ))
    # Page2: 層・材料
    st.session_state.update(dict(
        n_layers=3,
        gamma1=18.0, c1=5.0,  phi1=30.0, tau1=150.0,
        gamma2=19.0, c2=8.0,  phi2=28.0, tau2=180.0,
        gamma3=20.0, c3=12.0, phi3=25.0, tau3=200.0,
        tau_grout_cap_kPa=150.0,
        d_g=0.125, d_s=0.022,    # m
        d_g_mm=125, d_s_mm=22,   # 表示用mm
        fy=1000.0, gamma_m=1.20, mu=0.0,
    ))
    # Page3: 円弧探索・絶対範囲
    H, L = st.session_state["H"], st.session_state["L"]
    st.session_state.update(dict(
        x_min_abs=0.25*L, x_max_abs=1.15*L,
        y_min_abs=1.60*H, y_max_abs=2.20*H,
        grid_pitch_m=5.0,
        method="Bishop (simplified)",
        quality="Normal",
        Fs_target=1.20,
        allow_cross2=True, allow_cross3=True,
    ))
    # Page4/5: ネイル設定・結果
    st.session_state.update(dict(
        s_start=5.0, s_end=35.0,
        S_surf=2.0, S_row=2.0,
        tiers=1,
        angle_mode="Slope-Normal (⊥斜面)",
        beta_deg=15.0, delta_beta=0.0,
        L_mode="パターン1：固定長", L_nail=5.0, d_embed=1.0,
        res3=None, chosen_arc=None, nail_heads=[],
        result_reinforced=None,
    ))

def ensure_missing_defaults():
    # Page1
    ss_get("H", 25.0); ss_get("L", 60.0)
    ss_get("water_mode","WT"); ss_get("ru",0.0); ss_get("wt_offset",-2.0); ss_get("wl_points", None)
    # Page2
    ss_get("n_layers",3)
    ss_get("gamma1",18.0); ss_get("c1",5.0); ss_get("phi1",30.0); ss_get("tau1",150.0)
    ss_get("gamma2",19.0); ss_get("c2",8.0); ss_get("phi2",28.0); ss_get("tau2",180.0)
    ss_get("gamma3",20.0); ss_get("c3",12.0); ss_get("phi3",25.0); ss_get("tau3",200.0)
    ss_get("tau_grout_cap_kPa",150.0)
    ss_get("d_g",0.125); ss_get("d_s",0.022)
    ss_get("d_g_mm", int(round(st.session_state["d_g"]*1000)))
    ss_get("d_s_mm", int(round(st.session_state["d_s"]*1000)))
    ss_get("fy",1000.0); ss_get("gamma_m",1.20); ss_get("mu",0.0)
    # Page3
    H=float(st.session_state["H"]); L=float(st.session_state["L"])
    ss_get("x_min_abs", 0.25*L); ss_get("x_max_abs", 1.15*L)
    ss_get("y_min_abs", 1.60*H); ss_get("y_max_abs", 2.20*H)
    ss_get("grid_pitch_m", 5.0)
    ss_get("method", "Bishop (simplified)"); ss_get("quality", "Normal"); ss_get("Fs_target",1.20)
    ss_get("allow_cross2", True); ss_get("allow_cross3", True)
    # Page4/5
    ss_get("s_start",5.0); ss_get("s_end",35.0)
    ss_get("S_surf",2.0); ss_get("S_row",2.0)
    ss_get("tiers",1)
    ss_get("angle_mode","Slope-Normal (⊥斜面)")
    ss_get("beta_deg",15.0); ss_get("delta_beta",0.0)
    ss_get("L_mode","パターン1：固定長"); ss_get("L_nail",5.0); ss_get("d_embed",1.0)
    ss_get("res3", None); ss_get("chosen_arc", None); ss_get("nail_heads", []); ss_get("result_reinforced", None)

# 起動時：一度だけフル初期化
if "BOOTED" not in st.session_state:
    init_session_defaults()
    st.session_state["BOOTED"]=True
# 毎リラン：欠損だけ埋める（上書きはしない）
ensure_missing_defaults()

# ===================== 小物ヘルパ =====================
def HL_ground():
    H = float(st.session_state["H"]); L = float(st.session_state["L"])
    return H, L, make_ground_example(H, L)

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

# ===================== サイドバー（ページ選択と明示リセット） =====================
with st.sidebar:
    st.header("Pages")
    page = st.radio("", ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"], key="__page__")
    st.caption("※値はページを跨いでも保持されます。")
    if st.button("⚠ すべてリセット（初期値に戻す）"):
        keep = {"__page__"}
        for k in list(st.session_state.keys()):
            if k not in keep:
                del st.session_state[k]
        init_session_defaults()
        st.success("初期化しました。")

# ===================== Page1: 地形・水位 =====================
if page.startswith("1"):
    H,L,ground = HL_ground()

    st.subheader("Geometry")
    # ウィジェットの default 警告回避のため「先にキーを持っている」状態
    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H")
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L")
    H,L,ground = HL_ground()

    st.subheader("Water model")
    st.selectbox("Water model", ["WT","ru","WT+ru"], key="water_mode")
    st.slider("r_u (if ru mode)", 0.0, 0.9, step=0.05, key="ru")
    st.slider("Water level offset from ground (m, negative=below)", -30.0, 5.0, step=0.5, key="wt_offset")

    # 保存ボタン：この時だけ wl_points を作り直す（普段は上書きしない）
    if st.button("💾 水位を保存（WTのとき）"):
        Xd = np.linspace(ground.X[0], ground.X[-1], 300)
        Yg = np.array([float(ground.y_at(x)) for x in Xd])
        Yw = np.clip(Yg + float(st.session_state["wt_offset"]), 0.0, Yg)
        st.session_state["wl_points"] = np.vstack([Xd, Yw]).T
        st.success("水位線（wl_points）を保存しました。Page3/4/5でもこの水位が使われます。")

    # 表示：保存済みがあればそれを最優先で描画
    Xd = np.linspace(ground.X[0], ground.X[-1], 400)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    fig,ax = plt.subplots(figsize=(9.6,5.8))
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Soil")
    # 表示用WT：保存済があれば補間して表示、なければオフセット表示
    if st.session_state["water_mode"].startswith("WT"):
        if st.session_state["wl_points"] is not None:
            W = st.session_state["wl_points"]
            # Xd に線形補間して表示（範囲外は 0..地表でクリップ）
            Yw = np.interp(Xd, W[:,0], W[:,1], left=W[0,1], right=W[-1,1])
            Yw = np.clip(Yw, 0.0, Yg)
            ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (saved)")
        else:
            Yw_off = np.clip(Yg + float(st.session_state["wt_offset"]), 0.0, Yg)
            ax.plot(Xd, Yw_off, "-.", color="tab:blue", label="WT (offset preview)")
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

# ===================== Page2: 地層・材料 =====================
elif page.startswith("2"):
    H,L,ground = HL_ground()
    st.subheader("Layers & Materials")

    st.number_input("H (m)", min_value=5.0, max_value=200.0, step=0.5, key="H")
    st.number_input("L (m)", min_value=5.0, max_value=400.0, step=0.5, key="L")
    H,L,ground = HL_ground()

    st.selectbox("Number of layers", [1,2,3], key="n_layers")
    interfaces=[]
    if st.session_state["n_layers"]>=2: interfaces.append(make_interface1_example(H,L))
    if st.session_state["n_layers"]>=3: interfaces.append(make_interface2_example(H,L))

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Layer1 (top)**")
        st.number_input("γ₁", 10.0, 25.0, step=0.5, key="gamma1")
        st.number_input("c₁", 0.0, 200.0, step=0.5, key="c1")
        st.number_input("φ₁", 0.0, 45.0, step=0.5, key="phi1")
        st.number_input("τ₁ (kPa)", 0.0, 1000.0, step=10.0, key="tau1")
    if st.session_state["n_layers"]>=2:
        with cols[1]:
            st.markdown("**Layer2**")
            st.number_input("γ₂", 10.0, 25.0, step=0.5, key="gamma2")
            st.number_input("c₂", 0.0, 200.0, step=0.5, key="c2")
            st.number_input("φ₂", 0.0, 45.0, step=0.5, key="phi2")
            st.number_input("τ₂ (kPa)", 0.0, 1000.0, step=10.0, key="tau2")
    if st.session_state["n_layers"]>=3:
        with cols[2]:
            st.markdown("**Layer3 (bottom)**")
            st.number_input("γ₃", 10.0, 25.0, step=0.5, key="gamma3")
            st.number_input("c₃", 0.0, 200.0, step=0.5, key="c3")
            st.number_input("φ₃", 0.0, 45.0, step=0.5, key="phi3")
            st.number_input("τ₃ (kPa)", 0.0, 1000.0, step=10.0, key="tau3")
    with cols[-1]:
        st.markdown("**Grout / Nail**")
        st.number_input("τ_grout_cap (kPa)", 0.0, 2000.0, step=10.0, key="tau_grout_cap_kPa")
        st.number_input("削孔(=グラウト)径 d_g (mm)", 50, 300, step=1, key="d_g_mm")
        st.number_input("鉄筋径 d_s (mm)", 10, 50, step=1, key="d_s_mm")
        st.session_state["d_g"] = float(st.session_state["d_g_mm"])/1000.0
        st.session_state["d_s"] = float(st.session_state["d_s_mm"])/1000.0
        st.number_input("引張強さ fy (MPa)", 200.0, 2000.0, step=50.0, key="fy")
        st.number_input("材料安全率 γ_m", 1.00, 2.00, step=0.05, key="gamma_m")
        st.select_slider("逓減係数 μ（0〜0.9, 0.1刻み）", options=[round(0.1*i,1) for i in range(10)], key="mu")

    # プレビュー
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    fig,ax = plt.subplots(figsize=(9.5,5.8))
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
    ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")
    if st.session_state["n_layers"]>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], "--", lw=1.0)
    if st.session_state["n_layers"]>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], "--", lw=1.0)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page3: 円弧探索（未補強） =====================
elif page.startswith("3"):
    H,L,ground = HL_ground()
    st.subheader("円弧探索（未補強）")

    # p3_* バッファの初期化（初回のみ・本値は上書きしない）
    def seed_once(k_ui, v): 
        if k_ui not in st.session_state: st.session_state[k_ui]=v
    seed_once("p3_x_min", float(st.session_state["x_min_abs"]))
    seed_once("p3_x_max", float(st.session_state["x_max_abs"]))
    seed_once("p3_y_min", float(st.session_state["y_min_abs"]))
    seed_once("p3_y_max", float(st.session_state["y_max_abs"]))
    seed_once("p3_pitch", float(st.session_state["grid_pitch_m"]))
    seed_once("p3_method", st.session_state["method"])
    seed_once("p3_quality", st.session_state["quality"])
    seed_once("p3_Fs_t", float(st.session_state["Fs_target"]))
    seed_once("p3_allow2", bool(st.session_state["allow_cross2"]))
    seed_once("p3_allow3", bool(st.session_state["allow_cross3"]))

    with st.form("arc_params"):
        colA,colB = st.columns([1.3,1])
        with colA:
            st.number_input("x min (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_min")
            st.number_input("x max (m)", step=max(0.1,0.05*L), format="%.3f", key="p3_x_max")
            st.number_input("y min (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_min")
            st.number_input("y max (m)", step=max(0.1,0.10*H), format="%.3f", key="p3_y_max")
            st.number_input("Center-grid ピッチ (m)", min_value=0.1, step=0.1, format="%.2f", key="p3_pitch")
            st.caption(f"ヒント: x≈[{0.2*L:.1f},{4.0*L:.1f}], y≈[{0.8*H:.1f},{8.0*H:.1f}]")
        with colB:
            st.selectbox("Method", ["Bishop (simplified)","Fellenius"], key="p3_method")
            st.select_slider("Quality", options=list(QUALITY.keys()), key="p3_quality")
            st.number_input("Target FS", min_value=1.00, max_value=2.00, step=0.05, format="%.2f", key="p3_Fs_t")
        if st.session_state["n_layers"]>=2:
            st.checkbox("Allow into Layer 2", key="p3_allow2")
        if st.session_state["n_layers"]>=3:
            st.checkbox("Allow into Layer 3", key="p3_allow3")
        submitted = st.form_submit_button("💾 設定を確定（保存）")

    def sync_p3_to_main():
        x_min = float(st.session_state["p3_x_min"]); x_max = float(st.session_state["p3_x_max"])
        y_min = float(st.session_state["p3_y_min"]); y_max = float(st.session_state["p3_y_max"])
        if x_max < x_min: x_min, x_max = x_max, x_min
        if y_max < y_min: y_min, y_max = y_max, y_min
        st.session_state["x_min_abs"]=x_min; st.session_state["x_max_abs"]=x_max
        st.session_state["y_min_abs"]=y_min; st.session_state["y_max_abs"]=y_max
        st.session_state["grid_pitch_m"]=float(max(0.1, st.session_state["p3_pitch"]))
        st.session_state["method"]=st.session_state["p3_method"]
        st.session_state["quality"]=st.session_state["p3_quality"]
        st.session_state["Fs_target"]=float(st.session_state["p3_Fs_t"])
        st.session_state["allow_cross2"]=bool(st.session_state["p3_allow2"])
        st.session_state["allow_cross3"]=bool(st.session_state["p3_allow3"])

    if submitted:
        sync_p3_to_main()
        st.success("円弧探索の設定を保存しました。")

    # 可視化（保存済み水位を最優先）
    x_min = float(st.session_state["x_min_abs"]); x_max = float(st.session_state["x_max_abs"])
    y_min = float(st.session_state["y_min_abs"]); y_max = float(st.session_state["y_max_abs"])
    pitch = float(st.session_state["grid_pitch_m"])
    method = st.session_state["method"]; quality = st.session_state["quality"]; Fs_t=float(st.session_state["Fs_target"])

    interfaces=[]
    if st.session_state["n_layers"]>=2: interfaces.append(make_interface1_example(H,L))
    if st.session_state["n_layers"]>=3: interfaces.append(make_interface2_example(H,L))
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])

    fig,ax = plt.subplots(figsize=(10.0,6.8))
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
    ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")
    if st.session_state["water_mode"].startswith("WT"):
        if st.session_state["wl_points"] is not None:
            W = st.session_state["wl_points"]
            Yw = np.interp(Xd, W[:,0], W[:,1], left=W[0,1], right=W[-1,1])
            Yw = np.clip(Yw, 0.0, Yg)
            ax.plot(Xd, Yw, "-.", color="tab:blue", label="WT (saved)")
        else:
            Yw_off = np.clip(Yg + float(st.session_state["wt_offset"]), 0.0, Yg)
            ax.plot(Xd, Yw_off, "-.", color="tab:blue", label="WT (offset preview)")
    gx = np.arange(x_min, x_max+1e-9, pitch)
    gy = np.arange(y_min, y_max+1e-9, pitch)
    if gx.size<1: gx=np.array([x_min])
    if gy.size<1: gy=np.array([y_min])
    xs=[float(x) for x in gx for _ in gy]; ys=[float(y) for y in gy for _ in gx]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    ax.plot([x_min,x_max,x_max,x_min,x_min],[y_min,y_min,y_max,y_max,y_min], c="k", lw=1.0, alpha=0.4)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend(loc="upper right")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    # soils & allow_cross
    soils=[Soil(st.session_state["gamma1"],st.session_state["c1"],st.session_state["phi1"])]
    allow_cross=[]
    if st.session_state["n_layers"]>=2:
        soils.append(Soil(st.session_state["gamma2"],st.session_state["c2"],st.session_state["phi2"]))
        allow_cross.append(bool(st.session_state["allow_cross2"]))
    if st.session_state["n_layers"]>=3:
        soils.append(Soil(st.session_state["gamma3"],st.session_state["c3"],st.session_state["phi3"]))
        allow_cross.append(bool(st.session_state["allow_cross3"]))
    P = QUALITY[quality].copy()

    # 計算コア
    def compute_once():
        Hc,Lc = float(st.session_state["H"]), float(st.session_state["L"])
        groundL = make_ground_example(Hc, Lc)
        ifaces=[]
        if st.session_state["n_layers"]>=2: ifaces.append(make_interface1_example(Hc,Lc))
        if st.session_state["n_layers"]>=3: ifaces.append(make_interface2_example(Hc,Lc))

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

    if st.button("▶ 計算開始（未補強）"):
        # “保存”とは独立に、計算の直前に同期（リセットはしない）
        sync_p3_to_main()
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        st.session_state["res3"]=res
        xc,yc = res["center"]; d = res["refined"][res["idx_minFs"]]
        st.session_state["chosen_arc"] = dict(xc=xc,yc=yc,R=d["R"], x1=d["x1"], x2=d["x2"], Fs=d["Fs"])

    if st.session_state["res3"]:
        res = st.session_state["res3"]
        xc,yc = res["center"]; refined=res["refined"]; idx_minFs=res["idx_minFs"]

        fig,ax = plt.subplots(figsize=(10.0,7.0))
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
        ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")

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
    H,L,ground = HL_ground()
    st.subheader("ソイルネイル配置（試作：頭位置のみ）")
    if not st.session_state["chosen_arc"]:
        st.info("Page3でMin Fs円弧を確定してから来てね。"); st.stop()
    arc = st.session_state["chosen_arc"]

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
    st.session_state["nail_heads"] = nail_heads

    fig,ax = plt.subplots(figsize=(10.0,7.0))
    ax.plot(ground.X, ground.Y, lw=2.0, label="Ground")
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
    xc,yc,R = arc["xc"],arc["yc"],arc["R"]
    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")
    ax.scatter([p[0] for p in nail_heads], [p[1] for p in nail_heads], s=30, color="tab:blue", label=f"Nail heads ({len(nail_heads)})")
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    H,L,ground = HL_ground()
    st.subheader("補強後解析（試作）")
    ok_arc = st.session_state["chosen_arc"] is not None
    ok_heads = bool(st.session_state.get("nail_heads", []))
    btn = st.button("▶ 補強後の計算を実行", disabled=not (ok_arc and ok_heads))
    if not (ok_arc and ok_heads):
        missing=[]
        if not ok_arc: missing.append("Page3のMin Fs円弧")
        if not ok_heads: missing.append("Page4のネイル頭配置")
        st.info("必要情報: " + "、".join(missing))
    elif btn:
        with st.spinner("（将来）ネイル効果を連成計算中…"):
            st.session_state["result_reinforced"] = {
                "n_nails": len(st.session_state.get("nail_heads", [])),
                "arc_Fs_unreinforced": st.session_state["chosen_arc"]["Fs"],
                "note": "Phase-2で Tpullout/Tstrip(μ)/Ttens → Tt/Tn投影 → FS更新を実装予定。",
            }
    if st.session_state["result_reinforced"]:
        r = st.session_state["result_reinforced"]
        col1,col2 = st.columns(2)
        with col1: st.metric("ネイル本数", f"{r['n_nails']}")
        with col2: st.metric("未補強Fs（参照）", f"{r['arc_Fs_unreinforced']:.3f}")
        st.caption(r["note"])
