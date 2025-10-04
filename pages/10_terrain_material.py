# pages/10_terrain_material.py
# -*- coding: utf-8 -*-
import math
import streamlit as st

st.set_page_config(page_title="安定板２｜1) 地形・材料", layout="wide")
st.title("1) 地形・材料")

c1, c2, c3 = st.columns(3)
with c1:
    H = st.number_input("法高さ H [m]", 1.0, 200.0, float(st.session_state.get("H", 12.0)), 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 85.0, float(st.session_state.get("beta_deg", 35.0)), 0.5)
with c2:
    gamma = st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, float(st.session_state.get("gamma", 18.0)), 0.5)
    c = st.number_input("粘着力 c [kPa]", 0.0, 300.0, float(st.session_state.get("c", 10.0)), 1.0)
with c3:
    phi = st.number_input("内部摩擦角 φ [deg]", 0.0, 45.0, float(st.session_state.get("phi", 30.0)), 0.5)

# 表示レンジもここで決めて保存
tanb = math.tan(math.radians(beta))
x_top = H / max(tanb, 1e-6)
st.session_state.update({
    "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
    "x_left": -0.1*x_top, "x_right": 1.1*x_top,
    "y_min": -1.2*H, "y_max": 1.2*H
})

st.success("保存しました。次へ：左の『2) 地層』 または 下のリンクから進んでください。")
st.page_link("pages/20_layers.py", label="→ 2) 地層", icon="🧱")

