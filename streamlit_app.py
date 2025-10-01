# streamlit_app.py — Lightweight fan view with 2-layer option (clipped interface), batch-run
from __future__ import annotations
import streamlit as st
import numpy as np, math, heapq
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface_example,
    arcs_from_center_by_entries,
)

st.set_page_config(page_title="Stabi LEM（軽量・2層対応・一括実行）", layout="wide")
st.title("Stabi LEM（軽量描画レイヤ・センター固定／首振り・2層対応）")

# ---------------- UI: フォーム一括（押すまで計算しない） ----------------
with st.form("params"):
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

        preset = st.selectbox("Ground preset", ["3-seg berm (default)"])
        if preset == "3-seg berm (default)":
            pl = make_ground_example(H, L)

        st.subheader("Layers")
        use_two_layers = st.checkbox("Use two layers", True)
        if use_two_layers:
            iface_preset = st.selectbox("Interface preset", ["Dipping interface (default)"])
            if iface_preset == "Dipping interface (default)":
                interface = make_interface_example(H, L)
        else:
            interface = None

        st.subheader("Soil (upper / single-layer)")
        gamma_u = st.number_input("γ_upper (kN/m³)", 10.0, 25.0, 18.0, 0.5)
        c_u     = st.number_input("c_upper (kPa)",   0.0, 200.0, 5.0, 0.5)
        phi_u   = st.number_input("φ_upper (deg)",   0.0, 45.0, 30.0, 0.5)
        soil_upper = Soil(gamma=gamma_u, c=c_u, phi=phi_u)

        if use_two_layers:
            st.subheader("Soil (lower)")
            gamma_l = st.number_input("γ_lower (kN/m³)", 10.0, 25.0, 19.0, 0.5)
            c_l     = st.number_input("c_lower (kPa)",   0.0, 200.0, 8.0, 0.5)
            phi_l   = st.number_input("φ_lower (deg)",   0.0, 45.0, 28.0, 0.5)
            soil_lower = Soil(gamma=gamma_l, c=c_l, phi=phi_l)
        else:
            soil_lower = None

    with colB:
        st.subheader("Center grid（初期枠）")
        x_min = st.number_input("Center x min", 0.20*L, 2.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("Center x max", 0.30*L, 3.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("Center y min", 0.80*H, 6.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("Center y max", 1.00*H, 7.00*H, 2.20*H, 0.10*H)
        nx = st.slider("Grid nx", 6, 60, 14)
        ny = st.slider("Grid ny", 4, 40, 9)

        st.subheader("Fan parameters（軽量）")
        method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
        n_entries = st.slider("Entry samples on ground", 100, 2000, 600, 50)
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)
        show_k    = st.slider("Show top-K arcs (thin)", 10, 400, 120, 10)
        top_thick = st.slider("Emphasize top-N (thick)", 1, 30, 12, 1)
        show_radii = st.checkbox("Show radii to both ends", True)

        # ---- Center picking をフォーム内＆状態保持 ----
        if "pick_mode" not in st.session_state:
           
