# streamlit_app.py — Stabi LEM 多段UI + SoilNail簡易合成（安定版・フル）
from __future__ import annotations

# ===== 基本import =====
import math, time, heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ===== LEMコア（同階層 stabi_lem.py） =====
import stabi_lem as lem

st.set_page_config(page_title="Stabi LEM｜cfg一元・安定版", layout="wide")
st.title("Stabi LEM｜多段UI（cfg一元・安定版）")

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
            "tau_grout_cap_kPa": 150.0,   # グラウト-地山付着上限
            "d_g": 0.125,                 # グラウト径 [m]
            "d_s": 0.022,                 # 鋼材径 [m]
            "fy": 1000.0,                 # MPa
            "gamma_m": 1.20,
            "mu": 0.3,                    # 逓減係数
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
            "L_mode": "パターン2：すべり面より +Δm",
            "L_nail": 5.0, "d_embed": 1.0,
        },
        "results": {
            "unreinforced": None,   # {"center":(xc,yc),"refined":[...],"idx_minFs":int}
            "chosen_arc": None,
            "nail_heads": [],
            "reinforced": None,
        }
    }

# --- 数値キーを安全に辿る cfg_get/cfg_set ---
def _maybe_int_key(p):
    if isinstance(p, str) and p.isdigit():
        try:
            return int(p)
        except Exception:
            return p
    return p

def cfg_get(path, default=None):
    node = st.session_state["cfg"]
    for p in path.split("."):
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node: node = node[p]
            elif p_try in node: node = node[p_try]
            else: return default
        else:
            return default
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
                node[p_try] = {}; node = node[p_try]
        else:
            raise KeyError(f"cfg_set: '{p}' below is not a dict")
    last = _maybe_int_key(parts[-1])
    if isinstance(node, dict):
        node[last] = value
    else:
        raise KeyError(f"cfg_set: cannot set at '{parts[-1]}'")

def ui_seed(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# 起動時に cfg を1度だけ生成
if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

# ===================== 共通小物 =====================
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
    return H, L, lem.make_ground_example(H, L)

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
        Y1 = lem.clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = lem.clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
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
            # grid 初期枠（未設定時のみH,Lから種）
            if cfg_get("grid.x_min") is None:
                L = cfg_get("geom.L"); H = cfg_get("geom.H")
                cfg_set("grid.x_min", 0.25*L); cfg_set("grid.x_max", 1.15*L)
                cfg_set("grid.y_min", 1.60*H); cfg_set("grid.y_max", 2.20*H)
            st.success("cfgに保存しました。")
    with c2:
        if st.button("💾 WT水位線を offset から生成/更新（cfg.water.wl_points）"):
            H_ui = float(st.session_state["H"]); L_ui = float(st.session_state["L"])
            ground_ui = lem.make_ground_example(H_ui, L_ui)
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
    ground_ui = lem.make_ground_example(H_ui, L_ui)

    n_layers_cfg = int(cfg_get("layers.n"))
    interfaces_ui = []
    if n_layers_cfg >= 2: interfaces_ui.append(lem.make_interface1_example(H_ui, L_ui))
    if n_layers_cfg >= 3: interfaces_ui.append(lem.make_interface2_example(H_ui, L_ui))

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    Xd = np.linspace(ground_ui.X[0], ground_ui.X[-1], 600)
    Yg = np.array([float(ground_ui.y_at(x)) for x in Xd])

    if n_layers_cfg == 1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif n_layers_cfg == 2:
        Y1 = lem.clip_interfaces_to_ground(ground_ui, [interfaces_ui[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1, Y2 = lem.clip_interfaces_to_ground(ground_ui, [interfaces_ui[0], interfaces_ui[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")

    ax.plot(ground_ui.X, ground_ui.Y, linewidth=2.0, label="Ground")
    draw_water(ax, ground_ui, Xd, Yg)
    set_axes(ax, H_ui, L_ui, ground_ui); ax.grid(True); ax.legend()
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)
# >>> DXF_PLAN_PREVIEW START（ここから追記：DXFの中心線＋横断群プレビュー。既存UI/計算は不変更） >>>
with st.expander("🗺️ DXF：中心線＋横断群のプレビュー（実験）", expanded=False):
    st.caption("DXFから Alignment（中心線形）と XS*（横断法線）を読み込み、平面図に重ねて表示します。解析・cfgには影響しません。")
    dxf_file = st.file_uploader("DXFファイルを選択", type=["dxf"], key="__dxf_plan__")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        layer_align = st.text_input("中心線レイヤ名ヒント", value="ALIGN")
    with colB:
        layer_xs = st.text_input("横断レイヤ名（接頭辞OK）", value="XS")
    with colC:
        highlight = st.text_input("強調表示する横断ID（任意）", value="")

    try:
        if dxf_file is not None:
            # 依存ライブラリはローカルにのみ要求
            try:
                import tempfile, os
                from io.dxf_sections import load_alignment, load_sections, attach_stationing
                from viz.plan_preview import plot_plan_preview
            except ImportError as e:
                st.error("必要なモジュールが見つかりません。`pip install ezdxf` を実行してください。")
                st.stop()

            # 一時保存して ezdxf に渡す
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tf:
                tf.write(dxf_file.read())
                dxf_path = tf.name

            try:
                ali = load_alignment(dxf_path, layer_hint=layer_align.strip() or None)
                xs_raw = load_sections(dxf_path, layer_filter=layer_xs.strip() or "XS")
                xs = attach_stationing(xs_raw, ali)
                if not xs:
                    st.warning("横断レイヤ（XS*）が見つかりませんでした。レイヤ名を確認してください。")
                else:
                    st.success(f"読み込み成功：Alignment={ali.length:.1f} m、横断本数={len(xs)}")
                    fig2, ax2 = plt.subplots(figsize=(8.6, 6.0))
                    plot_plan_preview(ax2, ali, xs, highlight_id=(highlight or None))
                    st.pyplot(fig2); plt.close(fig2)
                    st.caption("※ ここは“プレビューのみ”。解析・cfgは変更しません。")
            finally:
                try:
                    os.unlink(dxf_path)
                except Exception:
                    pass
        else:
            st.info("DXFを選択すると平面図プレビューが表示されます。レイヤ名は任意（既定：ALIGN/XS）。")
    except Exception as e:
        st.error(f"DXFプレビューでエラーが発生しました：{e}")
# <<< DXF_PLAN_PREVIEW END（ここまで追記） <<<

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

    # プレビュー（cfg正本）
    fig,ax = plt.subplots(figsize=(9.5,5.8))
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(lem.make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(lem.make_interface2_example(H,L))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page3: 円弧探索（未補強） =====================
elif page.startswith("3"):
    H,L,ground = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n"))
    interfaces=[]
    if n_layers>=2: interfaces.append(lem.make_interface1_example(H,L))
    if n_layers>=3: interfaces.append(lem.make_interface2_example(H,L))

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

    # 可視化（cfg正本）— RangeError対策の正規化
    x_min=cfg_get("grid.x_min"); x_max=cfg_get("grid.x_max")
    y_min=cfg_get("grid.y_min"); y_max=cfg_get("grid.y_max")
    pitch=cfg_get("grid.pitch")
    try:
        x_min,x_max,y_min,y_max,pitch = float(x_min),float(x_max),float(y_min),float(y_max),float(pitch)
    except Exception:
        x_min,x_max,y_min,y_max,pitch = 10.0,70.0,30.0,80.0,5.0
    if not np.isfinite(pitch) or pitch<=0: pitch=5.0
    if x_max<=x_min: x_max=x_min+max(1.0,pitch)
    if y_max<=y_min: y_max=y_min+max(1.0,pitch)

    method=cfg_get("grid.method"); quality=cfg_get("grid.quality"); Fs_t=float(cfg_get("grid.Fs_target"))

    fig,ax = plt.subplots(figsize=(10.0,6.8))
    Xd,Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    gx = np.arange(x_min, x_max+1e-9, max(pitch,1e-6))
    gy = np.arange(y_min, y_max+1e-9, max(pitch,1e-6))
    if gx.size<1: gx=np.array([x_min])
    if gy.size<1: gy=np.array([y_min])
    xs=[float(x) for x in gx for _ in gy]; ys=[float(y) for y in gy for _ in gx]
    ax.scatter(xs, ys, s=10, c="k", alpha=0.25, marker=".", label=f"Center grid (pitch={pitch:.2f} m)")
    ax.plot([x_min,x_max,x_max,x_min,x_min],[y_min,y_min,y_max,y_max,y_min], c="k", lw=1.0, alpha=0.4)
    set_axes(ax, H, L, ground); ax.grid(True); ax.legend(loc="upper right")
    st.pyplot(fig); plt.close(fig)

    # soils & allow_cross
    mats = cfg_get("layers.mat")
    soils=[lem.Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
    allow_cross=[]
    if n_layers>=2:
        soils.append(lem.Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross2")))
    if n_layers>=3:
        soils.append(lem.Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross3")))
    P = QUALITY[quality].copy()

    def compute_once():
        Hc,Lc,groundL = make_ground_from_cfg()
        ifaces=[]
        if n_layers>=2: ifaces.append(lem.make_interface1_example(Hc,Lc))
        if n_layers>=3: ifaces.append(lem.make_interface2_example(Hc,Lc))

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
                for _x1,_x2,_R,Fs in lem.arcs_from_center_by_entries_multi(
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
        for _x1,_x2,R,Fs in lem.arcs_from_center_by_entries_multi(
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
            Fs = lem.fs_given_R_multi(groundL, ifaces, soils, allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
            if Fs is None: continue
            s = lem.arc_sample_poly_best_pair(groundL, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = lem.driving_sum_for_R_multi(groundL, ifaces, soils, allow_cross, xc, yc, R, n_slices=P["final_slices"])
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
    H,L,ground = make_ground_from_cfg()

    n_layers = int(cfg_get("layers.n"))
    interfaces = []
    if n_layers >= 2: interfaces.append(lem.make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(lem.make_interface2_example(H, L))

    st.subheader("ソイルネイル配置")

    # chosen_arc が無ければ復元
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

    # UI seed
    nails = cfg_get("nails")
    ui_seed("s_start", nails["s_start"]); ui_seed("s_end", nails["s_end"])
    ui_seed("S_surf", nails["S_surf"]);   ui_seed("S_row", nails["S_row"])
    ui_seed("tiers", nails["tiers"])
    ui_seed("angle_mode", nails["angle_mode"])
    ui_seed("beta_deg", nails["beta_deg"]); ui_seed("delta_beta", nails["delta_beta"])
    ui_seed("L_mode", nails["L_mode"]); ui_seed("L_nail", nails["L_nail"]); ui_seed("d_embed", nails["d_embed"])

    # 斜面の測地長（s）
    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seg = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = float(s_cum[-1])

    # 入力 UI
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

    # s→(x,y)
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

    # --- 可視化（ネイル軸 & ボンド） ---
    def slope_tangent_angle(ground, x):
        x2 = x + 1e-4
        y1 = float(ground.y_at(x)); y2 = float(ground.y_at(x2))
        return math.atan2((y2 - y1), (x2 - x))

    def choose_inward_dir(xh, yh, tau):
        # 斜面法線の2方向
        cand = [tau + math.pi/2, tau - math.pi/2]
        for th in cand:
            ct, stn = math.cos(th), math.sin(th)
            # 円との交点（t>0の最小）
            B = 2*((xh - arc["xc"])*ct + (yh - arc["yc"])*stn)
            C = (xh - arc["xc"])**2 + (yh - arc["yc"])**2 - arc["R"]**2
            disc = B*B - 4*C
            if disc <= 0: continue
            t = min([t for t in [(-B - math.sqrt(disc))/2.0, (-B + math.sqrt(disc))/2.0] if t>1e-9], default=None)
            if t is None: continue
            xq, yq = xh + ct*t, yh + stn*t
            # 地山側判定：交点が地表より下
            if yq <= float(ground.y_at(xq)) - 1e-6:
                return th, t, (xq,yq)
        # どちらもダメなら最初で返す
        th = cand[0]
        ct, stn = math.cos(th), math.sin(th)
        return th, 0.0, (xh, yh)

    fig,ax = plt.subplots(figsize=(10.0,7.0))
    Xd2,Yg2 = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd2, Yg2)

    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=arc["yc"] - np.sqrt(np.maximum(0.0, arc["R"]**2 - (xs - arc["xc"])**2))
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Chosen slip arc (Fs={arc['Fs']:.3f})")

    NH = cfg_get("results.nail_heads", [])
    if NH:
        ax.scatter([p[0] for p in NH], [p[1] for p in NH], s=30, color="tab:blue", label=f"Nail heads ({len(NH)})")

    angle_mode = cfg_get("nails.angle_mode")
    beta_deg   = float(cfg_get("nails.beta_deg", 15.0))
    delta_beta = float(cfg_get("nails.delta_beta", 0.0))
    L_mode     = cfg_get("nails.L_mode")
    L_nail     = float(cfg_get("nails.L_nail", 5.0))
    d_embed    = float(cfg_get("nails.d_embed", 1.0))

    for (xh, yh) in NH:
        if str(angle_mode).startswith("Slope-Normal"):
            tau = slope_tangent_angle(ground, float(xh))
            th0 = tau + (delta_beta*math.pi/180.0)
            th, t_hit, (xq,yq) = choose_inward_dir(xh, yh, th0)
        else:
            th = -beta_deg*math.pi/180.0
            ct, stn = math.cos(th), math.sin(th)
            B = 2*((xh - arc["xc"])*ct + (yh - arc["yc"])*stn)
            C = (xh - arc["xc"])**2 + (yh - arc["yc"])**2 - arc["R"]**2
            disc = B*B - 4*C
            t_hit = min([t for t in [(-B - math.sqrt(disc))/2.0, (-B + math.sqrt(disc))/2.0] if t>1e-9], default=0.0)
            xq, yq = (xh + ct*t_hit, yh + stn*t_hit) if t_hit>0 else (xh, yh)

        ct, stn = math.cos(th), math.sin(th)
        # 頭→すべり面
        ax.plot([xh, xq], [yh, yq], color="tab:blue", lw=1.8, alpha=0.9)

        # ボンド区間
        if str(L_mode).startswith("パターン2"):
            Lb = max(0.0, d_embed)
        else:
            Lb = max(0.0, L_nail - t_hit)
        if Lb > 1e-3 and t_hit>0:
            xb2, yb2 = xq + ct*Lb, yq + stn*Lb
            ax.plot([xq, xb2], [yq, yb2], color="tab:green", lw=2.2, alpha=0.9)

    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- cfg を必ず用意 ----
    if "cfg" not in st.session_state or not isinstance(st.session_state["cfg"], dict):
        st.session_state["cfg"] = default_cfg()

    st.subheader("補強後解析（簡易合成）")

    # ========== 必須データ ==========
    arc = cfg_get("results.chosen_arc")
    NH  = cfg_get("results.nail_heads", [])

    # ---- プリフライト（壊れててもここで復旧）----
    need = {"xc","yc","R","x1","x2","Fs"}
    if not isinstance(arc, dict) or not need.issubset(arc.keys()):
        st.info("未補強すべり面が未確定です。Page3で未補強の計算を実行してください。")
        st.stop()
    try:
        for k in list(need):
            arc[k] = float(arc[k])
            if not np.isfinite(arc[k]): raise ValueError(k)
    except Exception:
        st.error("results.chosen_arc に不正な数値があります。Page3で再計算してください。")
        st.stop()

    try:
        NH = [(float(x), float(y)) for (x, y) in NH if np.isfinite(x) and np.isfinite(y)]
    except Exception:
        NH = []
    if not NH:
        st.info("ネイル頭が未配置です。Page4でネイルを配置してください。")
        st.stop()

    mats_raw = cfg_get("layers.mat")
    if not isinstance(mats_raw, dict) or not mats_raw:
        mats_raw = default_cfg()["layers"]["mat"]
    def _to_int(k):
        try: return int(k)
        except: return k
    mats = {}
    for k, m in mats_raw.items():
        if not isinstance(m, dict): continue
        mats[int(_to_int(k))] = {
            "gamma": float(m["gamma"]), "c": float(m["c"]),
            "phi": float(m["phi"]), "tau": float(m.get("tau", 0.0))
        }
    mats = dict(sorted(mats.items(), key=lambda t: t[0]))
    cfg_set("layers.mat", mats)

    if "QUALITY" not in globals() or not isinstance(QUALITY, dict):
        QUALITY = {"Normal": {"final_slices": 40}}

    # ========== 地形・層 ==========
    H, L, ground = make_ground_from_cfg()
    n_layers = int(cfg_get("layers.n", 1))
    interfaces = []
    if n_layers >= 2: interfaces.append(lem.make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(lem.make_interface2_example(H, L))

    # ========== 図の土台（常に表示：プレビュー可） ==========
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd, Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)

    xc, yc, R = arc["xc"], arc["yc"], arc["R"]
    xs = np.linspace(arc["x1"], arc["x2"], 400)
    ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, lw=2.4, color="tab:red", label=f"Slip arc (Fs0={arc['Fs']:.3f})")

    # ネイル頭は常に表示（プレビュー）
    ax.scatter([p[0] for p in NH], [p[1] for p in NH],
               s=26, color="tab:blue", label=f"Nail heads ({len(NH)})")

    # ---- ネイル描画用ヘルパ ----
    angle_mode = str(cfg_get("nails.angle_mode", "Slope-Normal (⊥斜面)"))
    beta_deg   = float(cfg_get("nails.beta_deg", 15.0))
    delta_beta = float(cfg_get("nails.delta_beta", 0.0))
    L_mode     = str(cfg_get("nails.L_mode", "パターン1：固定長"))
    L_nail     = float(cfg_get("nails.L_nail", 5.0))
    d_embed    = float(cfg_get("nails.d_embed", 1.0))

    def slope_tangent_angle(x):
        x2 = x + 1e-4
        y1 = float(ground.y_at(x)); y2 = float(ground.y_at(x2))
        return math.atan2((y2 - y1), (x2 - x))

    # すべり面との交点を探して、プレビュー線（青）だけは常に描く
    for (xh, yh) in NH:
        if angle_mode.startswith("Slope-Normal"):
            tau = slope_tangent_angle(xh)
            theta = tau - math.pi/2 + math.radians(delta_beta)  # 地山側
        else:
            theta = -abs(math.radians(beta_deg))                # 水平から下向き
        ct, st_sin = math.cos(theta), math.sin(theta)
        B = 2.0 * ((xh - xc)*ct + (yh - yc)*st_sin)
        C = (xh - xc)**2 + (yh - yc)**2 - R**2
        disc = B*B - 4.0*C
        if disc <= 0: 
            # 交点なし：固定長のプレビュー
            ax.plot([xh, xh + ct*L_nail], [yh, yh + st_sin*L_nail],
                    color="tab:blue", lw=1.2, alpha=0.5)
            continue
        sdisc = math.sqrt(max(0.0, disc))
        t_pos = [t for t in [(-B - sdisc)/2.0, (-B + sdisc)/2.0] if t > 1e-9]
        if not t_pos:
            ax.plot([xh, xh + ct*L_nail], [yh, yh + st_sin*L_nail],
                    color="tab:blue", lw=1.2, alpha=0.5)
            continue
        t = min(t_pos)
        xq, yq = xh + ct*t, yh + st_sin*t
        ax.plot([xh, xq], [yh, yq], color="tab:blue", lw=1.6, alpha=0.9)
        # ボンド区間プレビュー（緑）
        Lb_prev = (max(0.0, d_embed) if "パターン2" in L_mode else max(0.0, L_nail - t))
        if Lb_prev > 1e-3:
            xb2, yb2 = xq + ct*Lb_prev, yq + st_sin*Lb_prev
            ax.plot([xq, xb2], [yq, yb2], color="tab:green", lw=2.0, alpha=0.9)

    set_axes(ax, H, L, ground); ax.grid(True); ax.legend()
    st.pyplot(fig); plt.close(fig)

    # ========== ここから「ボタンで実行」 ==========
    btn = st.button("▶ 補強後の計算を実行")
    if not btn:
        st.caption("ボタンを押すと、ネイルの引抜/鋼材と D=Σ(W sinα) から Fs を合成します。")
        st.stop()

    # ---- 材料パラメータ（kN系）----
    tau_cap_kPa = float(cfg_get("layers.tau_grout_cap_kPa", 150.0))  # kPa
    d_g   = float(cfg_get("layers.d_g", 0.125))                      # m
    d_s   = float(cfg_get("layers.d_s", 0.022))                      # m
    fy    = float(cfg_get("layers.fy", 1000.0))                      # MPa
    gamma_m = float(cfg_get("layers.gamma_m", 1.2))
    mu_decay= float(cfg_get("layers.mu", 0.0))

    tau_cap = tau_cap_kPa * 1e-3             # kN/m^2
    As = math.pi * (d_s**2) / 4.0            # m^2（1m幅）
    T_steel = fy * 1e3 * As / max(gamma_m, 1e-6)  # kN

    # ---- ネイル合力（ボタン押下時のみ計算）----
    T_sum = 0.0
    for i, (xh, yh) in enumerate(NH):
        if angle_mode.startswith("Slope-Normal"):
            tau = slope_tangent_angle(xh)
            theta = tau - math.pi/2 + math.radians(delta_beta)
        else:
            theta = -abs(math.radians(beta_deg))
        ct, st_sin = math.cos(theta), math.sin(theta)
        B = 2.0 * ((xh - xc)*ct + (yh - yc)*st_sin)
        C = (xh - xc)**2 + (yh - yc)**2 - R**2
        disc = B*B - 4.0*C
        if disc <= 0: 
            continue
        sdisc = math.sqrt(max(0.0, disc))
        t_pos = [t for t in [(-B - sdisc)/2.0, (-B + sdisc)/2.0] if t > 1e-9]
        if not t_pos: 
            continue
        t = min(t_pos)
        Lb = (max(0.0, d_embed) if "パターン2" in L_mode else max(0.0, L_nail - t))
        if Lb <= 1e-3:
            continue
        T_grout = tau_cap * math.pi * d_g * Lb
        T_cap   = min(T_grout, T_steel)
        if mu_decay > 0 and len(NH) > 1:
            T_cap *= max(0.0, 1.0 - mu_decay * (i / (len(NH) - 1)))
        T_sum += T_cap

    # ---- D=Σ(W sinα) ----
    soils = [lem.Soil(mats[1]["gamma"], mats[1]["c"], mats[1]["phi"])]
    allow_cross = []
    if n_layers >= 2:
        soils.append(lem.Soil(mats[2]["gamma"], mats[2]["c"], mats[2]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross2")))
    if n_layers >= 3:
        soils.append(lem.Soil(mats[3]["gamma"], mats[3]["c"], mats[3]["phi"]))
        allow_cross.append(bool(cfg_get("grid.allow_cross3")))

    qname = str(cfg_get("grid.quality", "Normal"))
    n_slices = QUALITY.get(qname, QUALITY["Normal"])["final_slices"]
    packD = lem.driving_sum_for_R_multi(ground, interfaces, soils, allow_cross,
                                        xc, yc, R, n_slices=n_slices)
    if not packD:
        st.error("D=Σ(W sinα) の評価に失敗しました（地形/層設定を確認）。")
        st.stop()
    D_sum, _, _ = packD
    if not (np.isfinite(D_sum) and D_sum > 0):
        st.error("D が不正（≤0 or NaN）です。設定を見直してください。")
        st.stop()

    # ---- Fs（簡易合成）----
    Fs0 = float(arc["Fs"])
    Fs_after = Fs0 + T_sum / max(D_sum, 1e-9)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("ネイル本数", f"{len(NH)}")
    with c2: st.metric("未補強Fs（参照）", f"{Fs0:.3f}")
    with c3: st.metric("補強後Fs（簡易）", f"{Fs_after:.3f}")
