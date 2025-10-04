# pages/10_terrain_material.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="安定板２｜1) 地形・材料", layout="wide")
st.title("1) 地形・材料（横断図プレビューつき）")

# ---------- 入力 ----------
c1, c2, c3 = st.columns(3)
with c1:
    H = st.number_input("法高さ H [m]", 1.0, 200.0, float(st.session_state.get("H", 12.0)), 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 85.0, float(st.session_state.get("beta_deg", 35.0)), 0.5)
with c2:
    gamma = st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, float(st.session_state.get("gamma", 18.0)), 0.5)
    c = st.number_input("粘着力 c [kPa]", 0.0, 300.0, float(st.session_state.get("c", 10.0)), 1.0)
with c3:
    phi = st.number_input("内部摩擦角 φ [deg]", 0.0, 45.0, float(st.session_state.get("phi", 30.0)), 0.5)

# ---------- 表示範囲（x/yレンジ）もここで編集できるように ----------
tanb = math.tan(math.radians(beta))
x_top_default = H / max(tanb, 1e-6)
defaults = {
    "x_left":  st.session_state.get("x_left",  -0.1 * x_top_default),
    "x_right": st.session_state.get("x_right",  1.1 * x_top_default),
    "y_min":   st.session_state.get("y_min",   -1.2 * H),
    "y_max":   st.session_state.get("y_max",    1.2 * H),
}

st.subheader("横断図の表示範囲")
r1, r2, r3, r4 = st.columns(4)
with r1:
    x_left  = st.number_input("x_left [m]",  -1000.0, 1000.0, float(defaults["x_left"]),  0.5, format="%.2f")
with r2:
    x_right = st.number_input("x_right [m]", -1000.0, 1000.0, float(defaults["x_right"]), 0.5, format="%.2f")
with r3:
    y_min   = st.number_input("y_min [m]",   -1000.0, 1000.0, float(defaults["y_min"]),   0.5, format="%.2f")
with r4:
    y_max   = st.number_input("y_max [m]",   -1000.0, 1000.0, float(defaults["y_max"]),   0.5, format="%.2f")

def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

# ---------- 横断図プレビュー ----------
st.subheader("横断図プレビュー")
X = np.linspace(x_left, x_right, 400)
Y = ground_y_at(X)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(X, Y, linewidth=2, label="Ground")
ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right)
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.35)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("横断図（地表ライン）")
ax.legend(loc="best")
st.pyplot(fig)

# ---------- 保存 ----------
if st.button("この条件を保存", type="primary"):
    st.session_state.update({
        "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
        "x_left": x_left, "x_right": x_right, "y_min": y_min, "y_max": y_max
    })
    st.success("保存しました。次へ：左の『2) 地層』または下のリンクから進んでください。")

st.page_link("pages/20_layers.py", label="→ 2) 地層", icon="🧱")
