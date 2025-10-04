# pages/10_terrain_material.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="安定板２｜1) 地形・材料", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("1) 地形・材料（横断図プレビューのみ・計算は既存のまま）")

# 既存のキーを尊重して上書き保存（未設定なら既定値）
H    = float(st.session_state.get("H", 12.0))
beta = float(st.session_state.get("beta_deg", 35.0))
gamma= float(st.session_state.get("gamma", 18.0))
c    = float(st.session_state.get("c", 10.0))
phi  = float(st.session_state.get("phi", 30.0))

c1, c2, c3 = st.columns(3)
with c1:
    H    = st.number_input("法高さ H [m]", 1.0, 200.0, H, 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 85.0, beta, 0.5)
with c2:
    gamma= st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, gamma, 0.5)
    c    = st.number_input("粘着力 c [kPa]", 0.0, 300.0, c, 1.0)
with c3:
    phi  = st.number_input("内部摩擦角 φ [deg]", 0.0, 45.0, phi, 0.5)

# 表示レンジは“既存値があればそのまま”。無ければ自動推定。
tanb = math.tan(math.radians(beta))
x_top = H / max(tanb, 1e-6)
x_left  = float(st.session_state.get("x_left",  -0.1 * x_top))
x_right = float(st.session_state.get("x_right",  1.1 * x_top))
y_min   = float(st.session_state.get("y_min",   -1.2 * H))
y_max   = float(st.session_state.get("y_max",    1.2 * H))

# 保存ボタン（ここでは“保存するだけ”。下流の式は一切変更しない）
if st.button("この条件を保存", type="primary"):
    st.session_state.update({
        "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
        "x_left": x_left, "x_right": x_right, "y_min": y_min, "y_max": y_max
    })
    st.success("保存しました。")

# ===== 横断図プレビュー（描くだけ） =====
st.subheader("横断図プレビュー（既存の仕様そのまま表示）")

def ground_y_at(X):
    X = np.asarray(X, float)
    return H - tanb * X

X = np.linspace(x_left, x_right, 400)
Yg = ground_y_at(X)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(X, Yg, lw=2, label="Ground")

# （任意）水位：既存の保存キーがあれば描く。無ければ何もしない。
# 例：水位を水平 y=const で保存している場合
if "water_y" in st.session_state:
    ax.plot([x_left, x_right], [st.session_state["water_y"]]*2, linestyle="--", label="Water table")
# 例：2点指定 (xw1,yw1)-(xw2,yw2)
if {"xw1","yw1","xw2","yw2"} <= set(st.session_state.keys()):
    xw1, yw1 = float(st.session_state["xw1"]), float(st.session_state["yw1"])
    xw2, yw2 = float(st.session_state["xw2"]), float(st.session_state["yw2"])
    ax.plot([xw1, xw2], [yw1, yw2], linestyle="--", label="Water table")

# 地層：既存の layers があれば色帯で重ねる（水平層想定。無ければ描かない）
# 期待する形式の例: [{"name":"A","z_top":8,"z_bot":4,"color":"#f7e4b1"}, ...]
layers = st.session_state.get("layers", None)
if isinstance(layers, list) and layers:
    for L in layers:
        try:
            zt = float(L["z_top"]); zb = float(L["z_bot"])
            color = L.get("color", "#e8e8e8")
            ax.fill_between(X, np.minimum(Yg, zt), np.minimum(Yg, zb), color=color, alpha=0.35, step="mid")
        except Exception:
            pass

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.35)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.legend(loc="best")
st.pyplot(fig)

st.page_link("pages/20_layers.py", label="→ 2) 地層へ", icon="🧱")
