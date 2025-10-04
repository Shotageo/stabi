# streamlit_app.py
# -*- coding: utf-8 -*-
import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt

from stabi_lem import Soil, CircleSlip, Nail, bishop_fs_unreinforced, bishop_fs_with_nails

# ---------------- Plot style（Theme/Tight layout/Legend切替：あなたの要望仕様） ----------------
st.set_page_config(page_title="Stabi｜安定板２", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", value=True)
show_legend = st.sidebar.checkbox("Show legend", value=True)
if theme == "dark_background":
    plt.style.use("dark_background")
else:
    plt.style.use("default")
# ------------------------------------------------------------------------------------------------

st.title("Stabi｜安定板２：ソイルネイル補強後Fs（本実装 初版）")

# ---- 地盤・円弧・簡易スライサ ----
colA, colB, colC = st.columns(3)
with colA:
    gamma = st.number_input("単位体積重量 γ [kN/m³]", 15.0, 30.0, 18.0, 0.5)
    c = st.number_input("粘着力 c [kPa]", 0.0, 200.0, 10.0, 1.0)
    phi = st.number_input("内部摩擦角 φ [deg]", 0.0, 45.0, 30.0, 1.0)
with colB:
    xc = st.number_input("円弧中心 xc [m]", -50.0, 50.0, 0.0, 0.5)
    yc = st.number_input("円弧中心 yc [m]", -50.0, 50.0, -10.0, 0.5)
    R  = st.number_input("半径 R [m]", 0.5, 200.0, 15.0, 0.5)
with colC:
    n_slices = st.slider("スライス数", 10, 80, 30, 5)
    slope_height = st.number_input("法高さH [m]（ダミー）", 1.0, 60.0, 10.0, 0.5)
    slope_angle_deg = st.number_input("法勾配角 [deg]（ダミー）", 10.0, 80.0, 35.0, 0.5)

soil = Soil(gamma=gamma, c=c, phi=phi)
slip = CircleSlip(xc=xc, yc=yc, R=R)

# ---- ダミー地形のスライス生成（安定板２の実地形/meshがあるなら置換OK） ----
alpha_slope = math.radians(slope_angle_deg)
x_top = slope_height / math.tan(alpha_slope)
x_left, x_right = 0.0, x_top
xs = np.linspace(x_left, x_right, n_slices+1)
slices = []
for i in range(n_slices):
    xL, xR = xs[i], xs[i+1]
    xm = 0.5*(xL+xR)
    ym = math.tan(alpha_slope)*(-xm) + slope_height
    alpha = math.atan2(ym - yc, xm - xc)
    alpha_slice = alpha + math.pi/2.0
    width = (xR - xL)
    yL = math.tan(alpha_slope)*(-xL) + slope_height
    yR = math.tan(alpha_slope)*(-xR) + slope_height
    area = 0.5*(yL + yR) * width
    W = gamma * area
    slices.append({'alpha': alpha_slice, 'width': width, 'W': W, 'area': area})

# ---- ソイルネイル（線分定義） ----
st.subheader("ソイルネイル（線分で表現）")
if "nails" not in st.session_state:
    st.session_state.nails = []

with st.expander("ネイルを追加"):
    c1, c2, c3 = st.columns(3)
    with c1:
        x1 = st.number_input("x1 [m]", -50.0, 50.0, 2.0, 0.1, key="nx1")
        y1 = st.number_input("y1 [m]", -50.0, 50.0, 6.0, 0.1, key="ny1")
        spacing = st.number_input("配置間隔 spacing [m]", 0.05, 5.0, 1.5, 0.05, key="nsp")
    with c2:
        x2 = st.number_input("x2 [m]", -50.0, 50.0, 8.0, 0.1, key="nx2")
        y2 = st.number_input("y2 [m]", -50.0, 50.0, 8.0, 0.1, key="ny2")
        T_yield = st.number_input("降伏耐力 T_yield [kN/本]", 10.0, 2000.0, 200.0, 10.0, key="nyld")
    with c3:
        bond = st.number_input("付着強度 bond_strength [kN/m]", 1.0, 500.0, 80.0, 1.0, key="nbnd")
        embed_each = st.number_input("有効定着長(片側) [m]", 0.1, 3.0, 0.5, 0.1, key="nemb")
        if st.button("＋このネイルを追加", type="primary"):
            st.session_state.nails.append(Nail(x1=x1, y1=y1, x2=x2, y2=y2,
                                               spacing=spacing, T_yield=T_yield,
                                               bond_strength=bond, embed_length_each_side=embed_each))

if st.session_state.nails:
    idx_del = st.selectbox("削除対象（インデックス）", list(range(len(st.session_state.nails))), index=0)
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("選択ネイルを削除"):
            st.session_state.nails.pop(idx_del)
    with col_del2:
        if st.button("全削除", help="すべてのネイルを削除します"):
            st.session_state.nails.clear()

# ---- 計算 ----
Fs_un = bishop_fs_unreinforced(slices, soil, slip)
Fs_re = bishop_fs_with_nails(slices, soil, slip, st.session_state.nails)

# ---- 結果表示 ----
st.subheader("結果")
cR1, cR2 = st.columns(2)
with cR1:
    st.metric("未補強 Fs", f"{Fs_un:.3f}")
with cR2:
    st.metric("補強後 Fs", f"{Fs_re:.3f}")

# ---- 図化（法面直線・円弧・ネイル線） ----
fig, ax = plt.subplots(figsize=(7, 6))
x_line = np.array([x_left, x_right])
y_line = slope_height - np.tan(alpha_slope)*x_line
ax.plot(x_line, y_line, label="Slope", linewidth=2)

th = np.linspace(0, 2*math.pi, 400)
ax.plot(slip.xc + slip.R*np.cos(th), slip.yc + slip.R*np.sin(th), linestyle="--", label="Slip circle")

for i, nl in enumerate(st.session_state.nails):
    ax.plot([nl.x1, nl.x2], [nl.y1, nl.y2], linewidth=2, label=f"Nail {i}" if show_legend else None)

ax.set_aspect('equal', 'box')
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True)
ax.set_title(f"Fs_un={Fs_un:.3f} / Fs_re={Fs_re:.3f}")
if show_legend:
    ax.legend(loc="best")
if tight:
    plt.tight_layout()
st.pyplot(fig)
