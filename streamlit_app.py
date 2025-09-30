# streamlit_app.py
import os, importlib.util
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BUILD_TAG = "build-2025-09-30-13:30JST"  # 反映確認用

# ---- safe import (ホットリロード対策) ----
try:
    from stabi_lem import Slope, Soil, grid_search_2d
except Exception:
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("stabi_lem", os.path.join(here, "stabi_lem.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    Slope, Soil, grid_search_2d = mod.Slope, mod.Soil, mod.grid_search_2d

st.set_page_config(page_title="Stabi LEM (2D Center Search)", layout="wide")
st.title("Stabi LEM : Bishop / Fellenius（2D Center Search）")
st.caption(f"🔧 {BUILD_TAG}")

# -------- Session State の初期化（ここで一度だけ設定） --------
def ensure_defaults():
    ss = st.session_state
    ss.setdefault("H", 20.0)
    ss.setdefault("L", 40.0)
    ss.setdefault("gamma", 18.0)
    ss.setdefault("c", 5.0)
    ss.setdefault("phi", 30.0)
    ss.setdefault("nx", 18)
    ss.setdefault("ny", 12)
    ss.setdefault("nR", 28)
    ss.setdefault("n_slices", 40)
    ss.setdefault("h_min", 0.2)
    ss.setdefault("h_max", 1000.0)
    ss.setdefault("pct", 15)
ensure_defaults()

# ---- Quick テストプリセット（コールバック内で rerun はしない） ----
def apply_quick_preset():
    ss = st.session_state
    ss["H"] = 20.0; ss["L"] = 40.0
    ss["gamma"] = 18.0; ss["c"] = 5.0; ss["phi"] = 30.0
    ss["nx"] = 18; ss["ny"] = 12; ss["nR"] = 28
    ss["n_slices"] = 40
    ss["h_min"] = 0.2; ss["h_max"] = 1000.0; ss["pct"] = 15
    # Streamlit 1.50+: コールバック終了後に自動で再実行されるため st.rerun() は不要

def suggest_ranges(H: float, L: float):
    diag = float(np.hypot(H, L))
    return (-1.0*L, 0.3*L), (0.6*H, 3.0*H), (max(0.35*diag, 3.0), 2.5*diag)

with st.sidebar:
    st.button("Quick test preset", on_click=apply_quick_preset)

    st.subheader("Geometry")
    H = st.number_input("H (m) crest height", min_value=1.0, max_value=300.0, step=1.0, key="H")
    L = st.number_input("L (m) horizontal length", min_value=1.0, max_value=600.0, step=1.0, key="L")

    st.subheader("Soil (single layer, u=0)")
    gamma = st.number_input("Unit weight γ (kN/m³)", min_value=10.0, max_value=30.0, step=0.5, key="gamma")
    c = st.number_input("Cohesion c (kPa)", min_value=0.0, max_value=500.0, step=0.5, key="c")
    phi = st.number_input("Friction φ (deg)", min_value=0.0, max_value=45.0, step=0.5, key="phi")

    st.subheader("Method & Discretization")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    n_slices = st.slider("Number of slices", min_value=10, max_value=200, step=1, key="n_slices")

    st.subheader("Search ranges")
    auto = st.checkbox("Auto ranges (recommended)", value=True)
    if auto:
        (x_left, x_right), (y_bottom, y_top), (Rmin, Rmax) = suggest_ranges(H, L)
        st.caption(f"Auto x:[{x_left:.1f},{x_right:.1f}], y:[{y_bottom:.1f},{y_top:.1f}], R:[{Rmin:.1f},{Rmax:.1f}]")
    else:
        x_left  = st.number_input("Center x min", -10000.0, 10000.0, -L, 1.0)
        x_right = st.number_input("Center x max", -10000.0, 10000.0,  0.3*L, 1.0)
        y_bottom = st.number_input("Center y min", -10000.0, 10000.0, 0.6*H, 1.0)
        y_top    = st.number_input("Center y max", -10000.0, 10000.0, 3.0*H, 1.0)
        Rmin     = st.number_input("Radius min", 1.0, 1e5, max(0.35*np.hypot(H, L), 3.0), 1.0)
        Rmax     = st.number_input("Radius max", 1.0, 1e6, 2.5*np.hypot(H, L), 1.0)

    st.subheader("Grid densities")
    nx = st.slider("Centers in x", min_value=5, max_value=60, step=1, key="nx")
    ny = st.slider("Centers in y", min_value=3, max_value=40, step=1, key="ny")
    nR = st.slider("Radius samples", min_value=5, max_value=80, step=1, key="nR")

    st.subheader("Depth filter (m)")
    st.caption("分位点ベース（デフォルト15%）で薄すぎる円を除外")
    h_min = st.number_input("h_min (m)", min_value=0.0, max_value=10000.0, step=0.1, key="h_min")
    h_max = st.number_input("h_max (m)", min_value=0.5, max_value=10000.0, step=0.5, key="h_max")
    pct   = st.slider("Depth percentile (%)", min_value=0, max_value=50, step=1, key="pct")

# guard
if st.session_state.h_max <= st.session_state.h_min:
    st.session_state.h_max = float(st.session_state.h_min) + 0.1
    st.info("Adjusted h_max to be greater than h_min.")

slope = Slope(H=H, L=L)
soil  = Soil(gamma=gamma, c=c, phi=phi)
method_key = "bishop" if method.lower().startswith("bishop") else "fellenius"

with st.spinner("Searching slip circles (2D centers)…"):
    res = grid_search_2d(
        slope, soil,
        x_center_range=(x_left, x_right),
        y_center_range=(y_bottom, y_top),
        R_range=(Rmin, Rmax),
        nx=nx, ny=ny, nR=nR,
        method=method_key,
        n_slices=n_slices,
        h_min=h_min, h_max=h_max, pct=float(pct)
    )

cands = res.get("candidates", [])
best  = res.get("best", None)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    xg = np.array([0.0, L]); yg = np.array([H, 0.0])
    ax.plot(xg, yg, linewidth=2)

    if not cands:
        ax.set_title("No valid slip circles. Widen ranges or increase grid density.")
    else:
        max_show = min(400, len(cands))
        default_show = min(150, len(cands))
        show_top = st.slider("Show first N candidates (for speed)", min_value=1, max_value=max_show, value=default_show)

        for r in cands[:show_top]:
            xc, yc, R = r["xc"], r["yc"], r["R"]
            xs = np.linspace(r["x1"], r["x2"], 180)
            ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=0.7, alpha=0.28)

        if best:
            xs = np.linspace(best["x1"], best["x2"], 360)
            ys = best["yc"] - np.sqrt(np.maximum(0.0, best["R"]**2 - (xs - best["xc"])**2))
            ax.plot(xs, ys, linewidth=2.6, color="red")

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
            st.dataframe(df.head(300), use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="candidates.csv", mime="text/csv")
