# streamlit_app.py
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from stabi_lem import Slope, Soil, grid_search

st.set_page_config(page_title="Stabi LEM (MVP)", layout="wide")
st.title("Stabi LEM (MVP) : Bishop / Fellenius")

with st.sidebar:
    st.subheader("Geometry")
    H = st.number_input("H (m) crest height", 1.0, 200.0, 20.0, 1.0)
    L = st.number_input("L (m) horizontal length", 1.0, 500.0, 40.0, 1.0)

    st.subheader("Soil")
    gamma = st.number_input("Unit weight γ (kN/m³)", 10.0, 30.0, 18.0, 0.5)
    c = st.number_input("Cohesion c (kPa)", 0.0, 500.0, 5.0, 0.5)
    phi = st.number_input("Friction φ (deg)", 0.0, 45.0, 30.0, 0.5)

    st.subheader("Search")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    nx = st.slider("Center grid count (x)", 5, 50, 15)
    nR = st.slider("Radius samples", 5, 60, 20)
    n_slices = st.slider("Number of slices", 10, 200, 40)

    # center y is set above crest by offset (behind slope)
    yc = st.number_input("Center y (m)", min_value=-500.0, max_value=500.0, value=H + 5.0, step=1.0)
    x_left = st.number_input("Center x min", -500.0, 500.0, -L, 1.0)
    x_right = st.number_input("Center x max", -500.0, 500.0, 0.0, 1.0)

    Rmin = st.number_input("Radius min (m)", 1.0, 1000.0, 5.0, 1.0)
    Rmax = st.number_input("Radius max (m)", 1.0, 2000.0, 60.0, 1.0)

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
        # draw slope line only
        xg = np.array([0.0, L])
        yg = np.array([H, 0.0])
        ax.plot(xg, yg, linewidth=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("No valid slip circles. Adjust search ranges.")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # draw slope line
        xg = np.array([0.0, L])
        yg = np.array([H, 0.0])
        ax.plot(xg, yg, linewidth=2)

        # slider guarded by candidate count
        cand_len = len(candidates)
        max_show = min(300, cand_len)
        default_show = min(150, cand_len)
        show_top = st.slider(
            "Show first N candidates (for speed)",
            1,
            max_show,
            default_show
        )

        for rec in candidates[:show_top]:
            xc, yc0, R = rec["xc"], rec["yc"], rec["R"]
            x1, x2 = rec["x1"], rec["x2"]
            xs = np.linspace(x1, x2, 200)
            inside = R**2 - (xs - xc)**2
            ys = yc0 - np.sqrt(np.maximum(0.0, inside))
            ax.plot(xs, ys, linewidth=0.8, alpha=0.3)

        # best in red
        if best:
            xc, yc0, R = best["xc"], best["yc"], best["R"]
            x1, x2 = best["x1"], best["x2"]
            xs = np.linspace(x1, x2, 400)
            ys = yc0 - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=2.5, color="red")

        # grid centers
        xs_centers = np.linspace(x_left, x_right, nx)
        ax.scatter(xs_centers, np.full_like(xs_centers, yc), s=10)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Slip circles (gray), Best (red)")
        st.pyplot(fig)

with col2:
    st.subheader("Results")
    if not candidates:
        st.error("No valid slip circles found. Adjust center/radius ranges or geometry.")
    else:
        st.metric("Candidates", f"{len(candidates)}")
        if result["best"]:
            st.metric("Min Fs", f"{result['best']['Fs']:.3f}")
            st.write(f"Method: **{method}**")
            st.write(f"xc={result['best']['xc']:.2f}, yc={result['best']['yc']:.2f}, R={result['best']['R']:.2f}")
        with st.expander("Top 10 (lowest Fs)"):
            arr = sorted(candidates, key=lambda r: r["Fs"])[:10]
            for i, r in enumerate(arr, 1):
                st.write(f"{i}. Fs={r['Fs']:.3f} | xc={r['xc']:.2f}, R={r['R']:.2f}")
