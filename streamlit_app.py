# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import Slope, Soil, grid_search

st.set_page_config(page_title="Stabi LEM (MVP)", layout="wide")
st.title("Stabi LEM (MVP) : Bishop / Fellenius")

# ---- Helpers ----
def _rerun():
    # Streamlit 1.27+ uses st.rerun(); older versions had st.experimental_rerun()
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def apply_preset(name: str):
    """Quick presets -> set session_state then rerun."""
    ss = st.session_state
    if name == "Cut Slope (標準土)":
        ss["H"] = 20.0; ss["L"] = 40.0
        ss["gamma"] = 18.0; ss["c"] = 5.0; ss["phi"] = 30.0
        ss["yc"] = ss["H"] + 5.0
        ss["x_left"] = -ss["L"]; ss["x_right"] = 0.0
        ss["Rmin"] = 5.0; ss["Rmax"] = 60.0
        ss["nx"] = 15; ss["nR"] = 20; ss["n_slices"] = 40
    elif name == "Steep Slope (急勾配)":
        ss["H"] = 30.0; ss["L"] = 30.0
        ss["gamma"] = 19.0; ss["c"] = 3.0; ss["phi"] = 28.0
        ss["yc"] = ss["H"] + 8.0
        ss["x_left"] = -ss["L"] * 1.2; ss["x_right"] = 0.0
        ss["Rmin"] = 5.0; ss["Rmax"] = 80.0
        ss["nx"] = 18; ss["nR"] = 24; ss["n_slices"] = 50
    elif name == "Gentle Slope (緩勾配)":
        ss["H"] = 15.0; ss["L"] = 60.0
        ss["gamma"] = 18.0; ss["c"] = 8.0; ss["phi"] = 32.0
        ss["yc"] = ss["H"] + 5.0
        ss["x_left"] = -ss["L"]; ss["x_right"] = 0.0
        ss["Rmin"] = 10.0; ss["Rmax"] = 120.0
        ss["nx"] = 15; ss["nR"] = 25; ss["n_slices"] = 40
    _rerun()

with st.sidebar:
    st.subheader("Quick Presets")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Cut Slope (標準土)"):
            apply_preset("Cut Slope (標準土)")
    with c2:
        if st.button("Steep Slope (急勾配)"):
            apply_preset("Steep Slope (急勾配)")
    with c3:
        if st.button("Gentle Slope (緩勾配)"):
            apply_preset("Gentle Slope (緩勾配)")

    st.subheader("Geometry")
    H = st.number_input("H (m) crest height", 1.0, 200.0, st.session_state.get("H", 20.0), 1.0, key="H")
    L = st.number_input("L (m) horizontal length", 1.0, 500.0, st.session_state.get("L", 40.0), 1.0, key="L")

    st.subheader("Soil")
    gamma = st.number_input("Unit weight γ (kN/m³)", 10.0, 30.0, st.session_state.get("gamma", 18.0), 0.5, key="gamma")
    c = st.number_input("Cohesion c (kPa)", 0.0, 500.0, st.session_state.get("c", 5.0), 0.5, key="c")
    phi = st.number_input("Friction φ (deg)", 0.0, 45.0, st.session_state.get("phi", 30.0), 0.5, key="phi")

    st.subheader("Search")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    nx = st.slider("Center grid count (x)", 5, 50, st.session_state.get("nx", 15), key="nx")
    nR = st.slider("Radius samples", 5, 60, st.session_state.get("nR", 20), key="nR")
    n_slices = st.slider("Number of slices", 10, 200, st.session_state.get("n_slices", 40), key="n_slices")

    yc = st.number_input("Center y (m)", -500.0, 500.0, st.session_state.get("yc", H + 5.0), 1.0, key="yc")
    x_left = st.number_input("Center x min", -500.0, 500.0, st.session_state.get("x_left", -L), 1.0, key="x_left")
    x_right = st.number_input("Center x max", -500.0, 500.0, st.session_state.get("x_right", 0.0), 1.0, key="x_right")

    Rmin = st.number_input("Radius min (m)", 1.0, 1000.0, st.session_state.get("Rmin", 5.0), 1.0, key="Rmin")
    Rmax = st.number_input("Radius max (m)", 1.0, 2000.0, st.session_state.get("Rmax", 60.0), 1.0, key="Rmax")

slope = Slope(H=H, L=L)
soil = Soil(gamma=gamma, c=c, phi=phi)
method_key = "bishop" if method.lower().startswith("bishop") else "fellenius"

with st.spinner("Searching slip circles..."):
    result = grid_search(
        slope, soil,
        x_center_range=(x_left, x_right),
        y_center=yc,
        R_range=(Rmin, Rmax),
        nx=nx, nR=nR,
        method=method_key,
        n_slices=n_slices
    )

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    candidates = result.get("candidates", [])
    best = result.get("best", None)

    if not candidates:
        fig, ax = plt.subplots(figsize=(8, 5))
        xg = np.array([0.0, L]); yg = np.array([H, 0.0])
        ax.plot(xg, yg, linewidth=
