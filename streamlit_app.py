# streamlit_app.py
import math
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import Slope, Soil, grid_search

st.set_page_config(page_title="Stabi LEM (MVP)", layout="wide")
st.title("Stabi LEM (MVP) : Bishop / Fellenius")

# ---- Helpers: Presets ----
def apply_preset(name: str):
    """Set session_state values for quick presets and rerun."""
    if name == "Cut Slope (標準土)":
        st.session_state["H"] = 20.0
        st.session_state["L"] = 40.0
        st.session_state["gamma"] = 18.0
        st.session_state["c"] = 5.0
        st.session_state["phi"] = 30.0
        st.session_state["yc"] = st.session_state["H"] + 5.0
        st.session_state["x_left"] = -st.session_state["L"]
        st.session_state["x_right"] = 0.0
        st.session_state["Rmin"] = 5.0
        st.session_state["Rmax"] = 60.0
        st.session_state["nx"] = 15
        st.session_state["nR"] = 20
        st.session_state["n_slices"] = 40
    elif name == "Steep Slope (急勾配)":
        st.session_state["H"] = 30.0
        st.session_state["L"] = 30.0
        st.session_state["gamma"] = 19.0
        st.session_state["c"] = 3.0
        st.session_state["phi"] = 28.0
        st.session_state["yc"] = st.session_state["H"] + 8.0
        st.session_state["x_left"] = -st.session_state["L"] * 1.2
        st.session_state["x_right"] = 0.0
        st.session_state["Rmin"] = 5.0
        st.session_state["Rmax"] = 80.0
        st.session_state["nx"] = 18
        st.session_state["nR"] = 24
        st.session_state["n_slices"] = 50
    elif name == "Gentle Slope (緩勾配)":
        st.session_state["H"] = 15.0
        st.session_state["L"] = 60.0
        st.session_state["gamma"] = 18.0
        st.session_state["c"] = 8.0
        st.session_state["phi"] = 32.0
        st.session_state["yc"] = st.session_state["H"] + 5.0
        st.session_state["x_left"] = -st.session_state["L"]
        st.session_state["x_right"] = 0.0
        st.session_state["Rmin"] = 10.0
        st.session_state["Rmax"] = 120.0
        st.session_state["nx"] = 15
        st.session_state["nR"] = 25
        st.session_state["n_slices"] = 40
    st.experimental_rerun()

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
    H = st.number_input("H (m) crest height", 1.0, 200.0, 20.0, 1.0, key="H")
    L = st.number_input("L (m) horizontal length", 1.0, 500.0, 40.0, 1.0, key="L")

    st.subheader("Soil")
    gamma = st.number_input("Unit weight γ (kN/m³)", 10.0, 30.0, 18.0, 0.5, key="gamma")
    c = st.number_input("Cohesion c (kPa)", 0.0, 500.0, 5.0, 0.5, key="c")
    phi = st.number_input("Friction φ (deg)", 0.0, 45.0, 30.0, 0.5, key="phi")

    st.subheader("Search")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    nx = st.slider("Center grid count (x)", 5, 50, st.session_state.get("nx", 15), key="nx")
    nR = st.slider("Radius samples", 5, 60, st.session_state.get("nR", 20), key="nR")
    n_slices = st.slider("Number of slices", 10, 200, st.session_state.get("n_slices", 40), key="n_slices")

    # center y is set above crest by offset (behind slope)
    yc = st.number_input("Center y (m)", min_value=-500.0, max_value=500.0,
                         value=st.session_state.get("yc", H + 5.0), step=1.0, key="yc")
    x_left = st.number_input("Center x min", -500.0, 500.0,
                             st.session_state.get("x_left", -L), 1.0, key="x_left")
    x_right = st.number_input("Center x max", -500.0, 500.0,
                              st.session_state.get("x_right", 0.0), 1.0, key="x_right")

    Rmin = st.number_input("Radius min (m)", 1.0, 1000.0,
                           st.session_state.get("Rmin", 5.0), 1.0, key="Rmin")
    Rmax = st.number_input("Radius max (m)", 1.0, 2000.0,
                           st.session_state.get("Rmax", 60.0), 1.0, key="Rmax")

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

        # Download CSV
        with st.expander("Candidates table / CSV"):
            df = pd.DataFrame(candidates)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="candidates.csv", mime="text/csv")

    # Educational formulas
    with st.expander("Formulas (教育用)"):
        st.markdown("**Fellenius (Ordinary Method of Slices)**")
        st.latex(r"""
        FS = \frac{\sum_i \left(c\,b_i + W_i \cos\alpha_i \tan\phi\right)}
                   {\sum_i W_i \sin\alpha_i}
        """)
        st.markdown("**Bishop (Simplified)**")
        st.latex(r"""
        FS = \frac{\sum_i \dfrac{c\,b_i + W_i \tan\phi \cos\alpha_i}
                               {1 + \dfrac{\tan\phi\,\tan\alpha_i}{FS}}}
                   {\sum_i W_i \sin\alpha_i}
        """)
        st.caption("※本MVPは地下水圧 u=0、単一層、等厚1m（平面ひずみ）を前提。")
