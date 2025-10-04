# pages/20_layers.py
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import math

st.set_page_config(page_title="安定板２｜2) 地層", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("2) 地層（保存とプレビューのみ・既存仕様は一切変更しない）")

# 既存 layers があればそれを見せる（編集は任意）
layers = st.session_state.get("layers", [])
st.write("現在保存されている層：", layers if layers else "（未設定）")

with st.expander("層を追加（水平層想定・任意）", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        z_top = st.number_input("層上端 z_top [m]", -1000.0, 1000.0, 8.0, 0.1)
        z_bot = st.number_input("層下端 z_bot [m]", -1000.0, 1000.0, 4.0, 0.1)
    with c2:
        name  = st.text_input("層名", "Layer")
        color = st.color_picker("表示色", "#f7e4b1")
    with c3:
        if st.button("＋この層を追加", type="primary"):
            L = {"name":name, "z_top":z_top, "z_bot":z_bot, "color":color}
            layers = list(layers) + [L]
            st.session_state["layers"] = layers
            st.success("追加しました。")

if layers:
    if st.button("全層を削除"):
        st.session_state["layers"] = []
        layers = []
        st.success("削除しました。")

# プレビュー（地形・水位と重ね書き）
H = float(st.session_state.get("H", 12.0))
beta = float(st.session_state.get("beta_deg", 35.0))
tanb = math.tan(math.radians(beta))
def ground_y_at(X): return H - tanb * X
x_left  = float(st.session_state.get("x_left", -1.0))
x_right = float(st.session_state.get("x_right",  1.0))
y_min   = float(st.session_state.get("y_min", -10.0))
y_max   = float(st.session_state.get("y_max",  10.0))

X = np.linspace(x_left, x_right, 400)
Yg = ground_y_at(X)
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(X, Yg, lw=2, label="Ground")

if isinstance(layers, list):
    for L in layers:
        try:
            zt = float(L["z_top"]); zb = float(L["z_bot"])
            color = L.get("color", "#e8e8e8")
            ax.fill_between(X, np.minimum(Yg, zt), np.minimum(Yg, zb), color=color, alpha=0.35, step="mid", label=L.get("name"))
        except Exception:
            pass

if "water_y" in st.session_state:
    ax.plot([x_left, x_right], [st.session_state["water_y"]]*2, linestyle="--", label="Water table")
if {"xw1","yw1","xw2","yw2"} <= set(st.session_state.keys()):
    xw1, yw1 = float(st.session_state["xw1"]), float(st.session_state["yw1"])
    xw2, yw2 = float(st.session_state["xw2"]), float(st.session_state["yw2"])
    ax.plot([xw1, xw2], [yw1, yw2], linestyle="--", label="Water table")

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.35); ax.legend(loc="best")
st.pyplot(fig)

st.page_link("pages/30_slip_search.py", label="→ 3) 円弧探索へ", icon="🌀")
