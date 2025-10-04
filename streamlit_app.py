# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Stabi｜安定板２（完全復帰仕様）
多段UI + cfg一元化 + 横断図 + 水位オフセット + 無補強Fs + ネイルモック
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from stabi_lem import Soil, CircleSlip, generate_slices_on_arc, bishop_fs_unreinforced

# --------------------------
# 基本設定・テーマ
# --------------------------
st.set_page_config(page_title="Stabi｜安定板２（復帰仕様）", layout="wide")

# Plot style
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"], index=0)
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

# --------------------------
# cfg 一元管理
# --------------------------
if "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "H": 12.0,
        "beta": 35.0,
        "gamma": 18.0,
        "c": 10.0,
        "phi": 30.0,
        "x_left": -5.0,
        "x_right": 30.0,
        "y_min": -10.0,
        "y_max": 20.0,
        "water_offset": 0.0,
        "layers": [],
        "audit": False,
        "quality": "Normal",
        "budget_coarse": 0.8,
        "budget_quick": 1.2,
    }
cfg = st.session_state["cfg"]

st.title("🧩 Stabi｜安定板２（完全復帰仕様）")

# --------------------------
# 1️⃣ 地形・材料設定
# --------------------------
st.header("1️⃣ 地形・材料設定")

col1, col2, col3 = st.columns(3)
with col1:
    cfg["H"] = st.number_input("法高さ H [m]", 1.0, 200.0, cfg["H"])
    cfg["beta"] = st.number_input("法勾配 β [°]", 5.0, 85.0, cfg["beta"])
with col2:
    cfg["gamma"] = st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, cfg["gamma"])
    cfg["c"] = st.number_input("粘着力 c [kPa]", 0.0, 300.0, cfg["c"])
with col3:
    cfg["phi"] = st.number_input("内部摩擦角 φ [°]", 0.0, 45.0, cfg["phi"])
    cfg["water_offset"] = st.number_input("水位オフセット h_off [m]", -10.0, 20.0, cfg["water_offset"])

tanb = math.tan(math.radians(cfg["beta"]))
soil = Soil(cfg["gamma"], cfg["c"], cfg["phi"])

# 安全評価関数
def ground_y_at(X):
    X = np.asarray(X, float)
    return cfg["H"] - tanb * X

# --------------------------
# 2️⃣ 地層設定
# --------------------------
st.header("2️⃣ 地層設定（任意）")

if "layers" not in cfg:
    cfg["layers"] = []
layers = cfg["layers"]

with st.expander("層を追加", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        z_top = st.number_input("層上端 z_top [m]", -100, 100, 8.0)
        z_bot = st.number_input("層下端 z_bot [m]", -100, 100, 4.0)
    with c2:
        color = st.color_picker("色", "#e7d7a8")
        name = st.text_input("層名", "Layer")
    with c3:
        if st.button("＋追加"):
            layers.append({"name": name, "z_top": z_top, "z_bot": z_bot, "color": color})
            st.success("層を追加しました。")

if layers:
    st.write(f"登録済み層数: {len(layers)}")
    if st.button("全削除"):
        layers.clear()
        st.success("地層を削除しました。")

# --------------------------
# 3️⃣ 水位設定（法尻オフセット基準）
# --------------------------
st.header("3️⃣ 水位設定（法尻基準オフセット）")
h_w = cfg["H"] - cfg["water_offset"]
st.write(f"水平水位線: y = {h_w:.2f} m")

# --------------------------
# 4️⃣ 横断図プレビュー
# --------------------------
st.header("4️⃣ 横断図プレビュー")

X = np.linspace(cfg["x_left"], cfg["x_right"], 500)
Yg = ground_y_at(X)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(X, Yg, lw=2, label="Ground")

# 層
for L in layers:
    ax.fill_between(X, np.minimum(Yg, L["z_top"]), np.minimum(Yg, L["z_bot"]),
                    color=L.get("color", "#dddddd"), alpha=0.3, step="mid", label=L.get("name","layer"))

# 水位線
ax.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water level")

ax.set_aspect("equal", "box")
ax.set_xlim(cfg["x_left"], cfg["x_right"])
ax.set_ylim(cfg["y_min"], cfg["y_max"])
ax.grid(True, alpha=0.4)
if show_legend: ax.legend()
if tight: plt.tight_layout()
st.pyplot(fig)

# --------------------------
# 5️⃣ 円弧探索（無補強）
# --------------------------
st.header("5️⃣ 無補強円弧探索（Bishop簡略法）")

cfg["quality"] = st.selectbox("探索クオリティ", ["Fast", "Normal", "High"], index=["Fast","Normal","High"].index(cfg["quality"]))
cfg["audit"] = st.checkbox("Audit（センター可視化）", value=False)

# 時間バジェット（参考表示）
st.caption(f"⏱️ Budget Coarse ≈ {cfg['budget_coarse']} s, Quick ≈ {cfg['budget_quick']} s")

def search_best():
    H = cfg["H"]
    xc_range = np.linspace(-0.2*H, 1.2*H, 22)
    yc_range = np.linspace(-2.0*H, 0.5*H, 14)
    R_range = np.linspace(0.6*H, 2.0*H, 16)

    best = None
    centers = []

    for xc in xc_range:
        for yc in yc_range:
            centers.append((xc, yc))
            for R in R_range:
                slip = CircleSlip(xc, yc, R)
                sls = generate_slices_on_arc(ground_y_at, slip, 36,
                                             x_min=cfg["x_left"], x_max=cfg["x_right"], soil_gamma=cfg["gamma"])
                if not sls:
                    continue
                Fs = bishop_fs_unreinforced(sls, soil)
                if (best is None) or (Fs < best[0]):
                    best = (Fs, slip, sls)
    return best, centers
    
    # ======== 後半：探索実行〜ネイル・結果 ========

if st.button("探索開始", type="primary"):
    start = time.time()
    best, centers = search_best()
    elapsed = time.time() - start

    if best:
        Fs, slip, slices = best
        st.session_state["selected_slip"] = {"xc": slip.xc, "yc": slip.yc, "R": slip.R}
        st.session_state["slices_best"] = slices
        st.session_state["Fs_last"] = Fs
        st.success(f"最小Fs = {Fs:.3f}  （{elapsed:.2f} s）")
    else:
        st.warning("安定した円弧が見つかりませんでした。")

    # 可視化
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(X, Yg, lw=2, label="Ground")

    if best:
        slip = best[1]
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(slip.xc + slip.R*np.cos(th),
                slip.yc + slip.R*np.sin(th), "--", lw=2, label="Slip circle")

    if cfg["audit"] and centers:
        xs, ys = zip(*centers)
        ax.scatter(xs, ys, s=8, alpha=0.3, label="Centers")

    ax.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water")
    ax.set_aspect("equal", "box")
    ax.set_xlim(cfg["x_left"], cfg["x_right"])
    ax.set_ylim(cfg["y_min"], cfg["y_max"])
    ax.grid(True, alpha=0.4)
    if show_legend: ax.legend()
    if tight: plt.tight_layout()
    st.pyplot(fig)


# --------------------------
# 6️⃣ ソイルネイル配置（モック）
# --------------------------
st.header("6️⃣ ソイルネイル配置（モック表示）")

if "nails" not in st.session_state:
    st.session_state["nails"] = []
nails = st.session_state["nails"]

with st.expander("ネイルを追加"):
    n1, n2 = st.columns(2)
    with n1:
        x1 = st.number_input("x1 [m]", -50.0, 100.0, 2.0)
        y1 = st.number_input("y1 [m]", -50.0, 100.0, 6.0)
        spacing = st.number_input("Spacing [m]", 0.1, 10.0, 1.5)
    with n2:
        x2 = st.number_input("x2 [m]", -50.0, 100.0, 8.0)
        y2 = st.number_input("y2 [m]", -50.0, 100.0, 8.0)
        T = st.number_input("T_yield [kN]", 1.0, 5000.0, 200.0)
    if st.button("追加", key="add_nail"):
        nails.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"spacing":spacing,"T":T})
        st.success("追加しました。")

if nails:
    st.write(f"登録済みネイル数: {len(nails)}")
    if st.button("全削除", key="clr_nail"):
        nails.clear()
        st.success("ネイルを全削除しました。")

# ネイル表示
if "selected_slip" in st.session_state:
    slip = st.session_state["selected_slip"]
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(X, Yg, lw=2, label="Ground")

    th = np.linspace(0, 2*np.pi, 400)
    ax2.plot(slip["xc"] + slip["R"]*np.cos(th),
             slip["yc"] + slip["R"]*np.sin(th),
             "--", lw=2, label="Slip circle")

    for i, nl in enumerate(nails):
        ax2.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]], lw=2,
                 label=(f"Nail {i+1}" if show_legend else None))

    ax2.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water")
    ax2.set_aspect("equal", "box")
    ax2.set_xlim(cfg["x_left"], cfg["x_right"])
    ax2.set_ylim(cfg["y_min"], cfg["y_max"])
    ax2.grid(True, alpha=0.4)
    if show_legend: ax2.legend()
    if tight: plt.tight_layout()
    st.pyplot(fig2)

# --------------------------
# 7️⃣ 結果表示（補強はモック）
# --------------------------
st.header("7️⃣ 結果（補強はモック）")

if "Fs_last" in st.session_state:
    Fs_un = st.session_state["Fs_last"]
    st.metric("未補強 Fs", f"{Fs_un:.3f}")
    st.metric("補強後（モック） Fs", f"{Fs_un:.3f}")
    st.caption("※ 現バージョンでは補強後Fsは未計算（モック表示）")
else:
    st.write("まだ安定計算が実行されていません。")

st.caption("🪶 この版は '安定板２（完全復帰仕様）' — cfg一元化・水位オフセット・多段UI・ネイルモック含む。")