# streamlit_app.py — cfg一本化 + 数値キー対応 + UI値プレビュー（安定版・パッチ適用）

from __future__ import annotations

# ---- 追加：親ディレクトリを sys.path に入れて stabi パッケージを解決 ----
import os, sys
_PKG_DIR = os.path.dirname(__file__)          # .../stabi
_PARENT  = os.path.dirname(_PKG_DIR)          # .../
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import streamlit as st
import numpy as np, heapq, time
import matplotlib.pyplot as plt

# ---- 置換：stabi コアはモジュール一括インポート → 既存名にエイリアス ----
import stabi.stabi_lem as lem
Soil                      = lem.Soil
GroundPL                  = lem.GroundPL
make_ground_example       = lem.make_ground_example
make_interface1_example   = lem.make_interface1_example
make_interface2_example   = lem.make_interface2_example
clip_interfaces_to_ground = lem.clip_interfaces_to_ground
arcs_from_center_by_entries_multi = lem.arcs_from_center_by_entries_multi
fs_given_R_multi          = lem.fs_given_R_multi
arc_sample_poly_best_pair = lem.arc_sample_poly_best_pair
driving_sum_for_R_multi   = lem.driving_sum_for_R_multi

st.set_page_config(page_title="Stabi LEM｜cfg一元・安定版", layout="wide")
st.title("Stabi LEM｜多段UI（cfg一元・安定版）")

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

# --- 数値キーを安全に辿る cfg_get/cfg_set ---
def _maybe_int_key(p):
    if isinstance(p, str) and p.isdigit():
        try:
            return int(p)
        except Exception:
            return p
    return p

def cfg_get(path, default=None):
    """path: 'section.key' or 'section.sub.key' （数値キーは自動で int 化）"""
    node = st.session_state["cfg"]
    for p in path.split("."):
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node:
                node = node[p]
            elif p_try in node:
                node = node[p_try]
            else:
                return default
        else:
            return default
    return node

def cfg_set(path, value):
    """path に value をセット（途中が無ければ dict を作成。数値キーも対応）"""
    node = st.session_state["cfg"]
    parts = path.split(".")
    for p in parts[:-1]:
        p_try = _maybe_int_key(p)
        if isinstance(node, dict):
            if p in node:
                node = node[p]
            elif p_try in node:
                node = node[p_try]
            else:
                node[p_try] = {}
                node = node[p_try]
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
    # （…中略：あなたの安定板２のPage1ブロックを**そのまま**保持…）
    # ここから下は、あなたが貼ってくれた“安定板２”のコードを一切いじっていません。
    # 省略表示の都合で、以降はあなたの元コードをそのまま使ってください。
    # ─────────────────────────────────────────
    # 直前に共有いただいた「長い安定板２コード」を、この場所に全コピーでOK。
    # ─────────────────────────────────────────
    pass

# ===================== Page2: 地層・材料 =====================
elif page.startswith("2"):
    # （安定板２のPage2をそのまま）
    pass

# ===================== Page3: 円弧探索（未補強） =====================
elif page.startswith("3"):
    # （安定板２のPage3をそのまま）
    pass

# ===================== Page4: ネイル配置 =====================
elif page.startswith("4"):
    # （安定板２のPage4をそのまま）
    pass

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    # （安定板２のPage5をそのまま）
    pass
