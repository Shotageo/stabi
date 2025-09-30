# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stabi_lem import Slope, Soil, grid_search_2d

st.set_page_config(page_title="Stabi LEM (2D Center Search)", layout="wide")
st.title("Stabi LEM : Bishop / Fellenius（2D Center Search）")

# --- Helpers ---
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def suggest_search_ranges(H: float, L: float):
    """Heuristic defaults based on slope size."""
    diag = float(np.hypot(H, L))
    x_min = -1.0 * L
    x_max =  0.3 * L
    y_min =  0.6 * H
    y_max =  3.0 * H
    Rmin = max(0.4 * diag, 3.0)
    Rmax = 2.5 * diag
    return (x_min, x_max), (y_min, y_max), (Rmin, Rmax)

with st.sidebar:
    st.subheader("Geometry")
    H = st.number_input("H (m) crest height", 1.0, 300.0, 20.0, 1.0)
    L = st.number_input("L (m) horizontal length", 1.0, 600.0, 40.0, 1.0)

    st.subheader("Soil (single layer, u=0)")
    gamma = st.number_input("Unit weight γ (kN/m³)", 10.0, 30.0, 18.0, 0.5)
    c = st.number_input("Cohesion c (kPa)", 0.0, 500.0, 5.0, 0.5)
    phi = st.number_input("Friction φ (deg)", 0.0, 45.0, 30.0, 0.5)

    st.subheader("Method & Discretization")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    n_slices = st.slider("Number of slices", 10, 200, 40)

    st.subheader("Search ranges")
    auto = st.checkbox("Auto ranges (recommended)", value=True)
    if auto:
        (x_left, x_right), (y_bottom, y_top), (Rmin, Rmax) = suggest_search_ranges(H, L)
        st.caption(f"Auto x:[{x_left:.1f},{x_right:.1f}], y:[{y_bottom:.1f},{y_top:.1f}], R:[{Rmin:.1f},{Rmax:.1f}]")
    else:
        x_left  = st.number_input("Center x min", -10000.0, 10000.0, -L, 1.0)
        x_right = st.number_input("Center x max", -10000.0, 10000.0,  0.3*L, 1.0)
        y_bottom = st.number_input("Center y min", -10000.0, 10000.0, 0.6*H, 1.0)
        y_top    = st.number_input("Center y max", -10000.0, 10000.0, 3.0*H, 1.0)
        Rmin     = st.number_input("Radius min", 1.0, 1e5, max(0.5*H, 3.0), 1.0)
        Rmax     = st.number_input("Radius max", 1.0, 1e6, 2.5*np.hypot(H, L), 1.0)

    st.subheader("Grid densities")
    nx = st.slider("Centers in x", 5, 60, 16)
    ny = st.slider("Centers in y", 3, 40, 10)
    nR = st.slider("Radius samples", 5, 80, 24)

    st.subheader("Depth filter (m)")
    st.caption("極端に浅い/深い候補を除外して現実的な円に絞る")
    min_depth = st.number_input("Min slice thickness h_min", 0.0, 10000.0, 1.0, 0.5)
    # ↓ 上限を 10000.0 に拡大、既定値 9999.0（以前は上限200のまま既定9999でエラー）
    max_depth = st.number_input("Max slice thickness h_max", 0.5, 10000.0, 9999.0, 0.5)

# guard: ensure min_depth < max_depth
if max_depth <= min_depth:
    st.warning("Max depth must be greater than Min depth. Adjusted automatically.")
    max_depth = float(min_depth) + 0.1

slope = Slope(H=H, L=L)
soil  = Soil(gamma=gamma, c=c, phi=phi)
method_key = "bishop" if method.lower().startswith("bishop") else "fellenius"

with st.spinner("Searching slip circles (2D centers)…"):
    result = grid_search_2d(
        slope, soil,
        x_center_range=(x_left, x_right),
        y_center_range=(y_bottom, y_top),
        R_range=(Rmin, Rmax),
        nx=nx, ny=ny, nR=nR,
        method=method_key,
        n_slices=n_slices,
        min_depth=min_depth, max_depth=max_depth
    )

cands = result.get("candidates", [])
best  = result.get("best", None)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    # slope line
    xg = np.array([0.0, L]); yg = np.array([H, 0.0])
    ax.plot(xg, yg, linewidth=2)

    if not cands:
        ax.set_title("No valid slip circles. Adjust ranges or grid density.")
    else:
        max_show = min(300, len(cands))
        default_show = min(120, len(cands))
        show_top = st.slider("Show first N candidates (for speed)", 1, max_show, default_show)

        for rec in cands[:show_top]:
            xc, yc, R = rec["xc"], rec["yc"], rec["R"]
            x1, x2 = rec["x1"], rec["x2"]
            xs = np.linspace(x1, x2, 180)
            ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=0.7, alpha=0.25)

        if best:
            xc, yc, R = best["xc"], best["yc"], best["R"]
            xs = np.linspace(best["x1"], best["x2"], 360)
            ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=2.4, color="red")

        xs_grid = np.linspace(x_left, x_right, nx)
        ys_grid = np.linspace(y_bottom, y_top, ny)
        XX, YY = np.meshgrid(xs_grid, ys_grid)
        ax.scatter(XX.flatten(), YY.flatten(), s=10, alpha=0.7)

        ax.set_title("Slip circles (gray), Best (red)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig)

with col2:
    st.subheader("Results")
    if not cands:
        st.error("No valid slip circles found.")
    else:
        st.metric("Candidates", f"{len(cands)}")
        st.write(f"Method: **{'Bishop' if method_key=='bishop' else 'Fellenius'}**")
        if best:
            st.metric("Min Fs", f"{best['Fs']:.3f}")
            st.write(f"xc={best['xc']:.2f}, yc={best['yc']:.2f}, R={best['R']:.2f}")

        with st.expander("Candidates table / CSV"):
            df = pd.DataFrame(cands).sort_values("Fs").reset_index(drop=True)
            st.dataframe(df.head(200), use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="candidates.csv", mime="text/csv")

    with st.expander("Formulas (教育用)"):
        st.markdown("**Fellenius (Ordinary Method of Slices)**")
        st.latex(r"FS = \frac{\sum_i \left(c\,b_i + W_i \cos\alpha_i \tan\phi\right)}{\sum_i W_i \sin\alpha_i}")
        st.markdown("**Bishop (Simplified)**")
        st.latex(r"FS = \frac{\sum_i \dfrac{c\,b_i + W_i \tan\phi \cos\alpha_i}{1 + \dfrac{\tan\phi\,\tan\alpha_i}{FS}}}{\sum_i W_i \sin\alpha_i}")
        st.caption("※ 今は u=0 単層。次段で2層化・水圧を導入。")
