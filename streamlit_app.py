# streamlit_app.py — 安定板２ + Soil Nail補強後解析＋描画（安定レイアウト統合）

from __future__ import annotations
import os, sys

# パッケージ（stabi）を親ディレクトリから解決できるようにする
_PKG_DIR = os.path.dirname(__file__)          # .../stabi
_PARENT  = os.path.dirname(_PKG_DIR)          # .../
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ===== 標準/外部ライブラリ =====
import math, time, heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ===== stabi モジュール読み込み =====
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


# ================================================================
# Plot style（Theme/Tight layout/Legend切替）
# ================================================================
plt.rcParams["font.size"] = 11
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "#555"
plt.rcParams["axes.linewidth"] = 0.8

# ================================================================
# ページ設定
# ================================================================
st.set_page_config(page_title="Stabi – LEM 安定解析", layout="wide")

# ================================================================
# CFG Utility
# ================================================================
CFG = {}
def cfg_get(k, default=None):
    return CFG.get(k, default)
def cfg_set(k, v):
    CFG[k] = v

# ================================================================
# 補助関数群（省略: make_ground_from_cfgなど既存どおり）
# ================================================================
def make_ground_from_cfg():
    H = 20.0
    L = 30.0
    g = make_ground_example(H, L)
    return H, L, g

def draw_layers_and_ground(ax, ground, n_layers, interfaces):
    Xd = np.linspace(ground.X[0], ground.X[-1], 400)
    Yg = ground.y_at(Xd)
    ax.plot(Xd, Yg, lw=2.5, color="black", label="Ground")
    for i, pl in enumerate(interfaces):
        Yi = pl.y_at(Xd)
        ax.plot(Xd, Yi, "--", lw=1.0, color=(0.5, 0.5, 0.5))
    return Xd, Yg

def draw_water(ax, ground, Xd, Yg):
    ax.fill_between(Xd, Yg - 1.0, Yg, color=(0.7, 0.85, 1.0), alpha=0.3)

def set_axes(ax, H, L, ground):
    ax.set_xlim(ground.X[0]-1, ground.X[-1]+1)
    ax.set_ylim(-1, H*1.05)
    ax.set_aspect("equal")

def fs_to_color(fs):
    if fs < 1.0: return (1.0, 0.3, 0.3)
    elif fs < 1.2: return (1.0, 0.7, 0.3)
    elif fs < 1.5: return (0.7, 1.0, 0.3)
    else: return (0.3, 0.9, 0.3)

def clip_yfloor(xs, ys, y_floor):
    mask = ys >= y_floor
    if not np.any(mask):
        return None
    return xs[mask], ys[mask]

# ================================================================
# Streamlit ページ制御
# ================================================================
page = st.sidebar.radio("ページ選択", ["1. 地形", "2. 探索", "3. 最小円弧", "4. ネイル配置", "5. 補強後解析"])

# ===================== Page3: 最小円弧描画 =====================
if page.startswith("3"):
    H, L, ground = make_ground_from_cfg()
    n_layers = 2
    interfaces = [make_interface1_example(H, L)]
    res = {
        "center": (15.0, 10.0),
        "idx_minFs": 0,
        "refined": [
            {"x1": 5.0, "x2": 25.0, "R": 20.0, "Fs": 0.95}
        ],
    }
    cfg_set("results.unreinforced", res)

    xc, yc = res["center"]
    refined = res["refined"]
    idx_minFs = res["idx_minFs"]

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd, Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)

    d = refined[idx_minFs]
    xs = np.linspace(d["x1"], d["x2"], 400)
    ys = yc - np.sqrt(np.maximum(0.0, d["R"] ** 2 - (xs - xc) ** 2))
    ax.plot(xs, ys, lw=3.0, color=(0.9, 0, 0), label=f"Min Fs = {d['Fs']:.3f}")
    y1 = float(ground.y_at(xs[0]))
    y2 = float(ground.y_at(xs[-1]))
    ax.plot([xc, xs[0]], [yc, y1], lw=1.1, color=(0.9, 0, 0), alpha=0.9)
    ax.plot([xc, xs[-1]], [yc, y2], lw=1.1, color=(0.9, 0, 0), alpha=0.9)

    set_axes(ax, H, L, ground)
    ax.legend()
    ax.set_title("未補強：最小円弧")
    st.pyplot(fig)
    plt.close(fig)

# ===================== Page4: ネイル配置 =====================
elif page.startswith("4"):
    H, L, ground = make_ground_from_cfg()
    n_layers = 2
    interfaces = [make_interface1_example(H, L)]
    st.subheader("ソイルネイル配置")
    n = st.slider("ネイル本数", 3, 15, 6)
    nail_heads = []
    for i in range(n):
        x = 5 + i * (L - 10) / (n - 1)
        y = float(ground.y_at(x)) - 1.0
        nail_heads.append((x, y))
    cfg_set("results.nail_heads", nail_heads)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd, Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)
    for nh in nail_heads:
        ax.scatter(*nh, color="tab:blue")
    set_axes(ax, H, L, ground)
    ax.legend()
    ax.set_title("ネイル配置確認")
    st.pyplot(fig)
    plt.close(fig)

# ===================== Page5: 補強後解析 =====================
elif page.startswith("5"):
    H, L, ground = make_ground_from_cfg()
    n_layers = 2
    interfaces = [make_interface1_example(H, L)]
    arc = cfg_get("results.unreinforced")
    NH = cfg_get("results.nail_heads", [])
    st.subheader("補強後解析（ネイル線描画）")

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    Xd, Yg = draw_layers_and_ground(ax, ground, n_layers, interfaces)
    draw_water(ax, ground, Xd, Yg)

    if arc:
        xc, yc = arc["center"]
        d = arc["refined"][arc["idx_minFs"]]
        xs = np.linspace(d["x1"], d["x2"], 400)
        ys = yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"未補強Fs={d['Fs']:.3f}")

    if NH:
        L_nail = 3.0
        theta = -15 * math.pi / 180
        for (xh, yh) in NH:
            xt = xh + L_nail * math.cos(theta)
            yt = yh + L_nail * math.sin(theta)
            ax.plot([xh, xt], [yh, yt], color="tab:blue", lw=2.0)
            ax.scatter([xh], [yh], color="tab:blue", s=30)
        ax.text(0.5, 0.05, f"{len(NH)}本のネイルを表示", transform=ax.transAxes)

    set_axes(ax, H, L, ground)
    ax.legend()
    ax.set_title("補強後（ネイル表示のみ）")
    st.pyplot(fig)
    plt.close(fig)
