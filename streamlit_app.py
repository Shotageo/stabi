# streamlit_app.py
import os, importlib.util
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BUILD_TAG = "build-2025-09-30-14:05JST"  # 反映確認用

# ---- safe import (hot-reload対策) ----
try:
    from stabi_lem import Slope, Soil, grid_search_adaptive
except Exception:
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("stabi_lem", os.path.join(here, "stabi_lem.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    Slope, Soil, grid_search_adaptive = mod.Slope, mod.Soil, mod.grid_search_adaptive

st.set_page_config(page_title="Stabi LEM (Adaptive)", layout="wide")
st.title("Stabi LEM : Bishop / Fellenius（逐次絞り込みサーチ）")
st.caption(f"🔧 {BUILD_TAG}")

# -------- Session State defaults --------
def ensure_defaults():
    ss = st.session_state
    ss.setdefault("H", 20.0); ss.setdefault("L", 40.0)
    ss.setdefault("gamma", 18.0); ss.setdefault("c", 5.0); ss.setdefault("phi", 30.0)
    ss.setdefault("nx", 18); ss.setdefault("ny", 12); ss.setdefault("nR", 28)
    ss.setdefault("n_slices", 40)
    ss.setdefault("h_min", 0.2); ss.setdefault("h_max", 1000.0); ss.setdefault("pct", 15)
    ss.setdefault("refine_levels", 2); ss.setdefault("top_k", 8)
    ss.setdefault("tick_step", 0.0); ss.setdefault("fill_section", True); ss.setdefault("show_centers", True)
ensure_defaults()

def suggest_ranges(H: float, L: float):
    diag = float(np.hypot(H, L))
    return (-1.0*L, 0.3*L), (0.6*H, 3.0*H), (max(0.35*diag, 3.0), 2.5*diag)

def nice_step(max_val: float, target_ticks: int = 8) -> float:
    """同一間隔の“気持ちの良い”目盛り幅（1-2-5系列）"""
    if max_val <= 0: return 1.0
    raw = max_val / max(target_ticks, 1)
    base = 10 ** int(np.floor(np.log10(raw)))
    for m in [1, 2, 5, 10]:
        if raw <= m * base:
            return m * base
    return 10 * base

# ---- Quick preset without rerun ----
def apply_quick_preset():
    ss = st.session_state
    ss.update(dict(H=20.0, L=40.0, gamma=18.0, c=5.0, phi=30.0,
                   nx=18, ny=12, nR=28, n_slices=40,
                   h_min=0.2, h_max=1000.0, pct=15,
                   refine_levels=2, top_k=8))
with st.sidebar:
    st.button("Quick test preset", on_click=apply_quick_preset)

    st.subheader("Geometry")
    H = st.number_input("H (m) crest height", 1.0, 300.0, step=1.0, key="H")
    L = st.number_input("L (m) horizontal length", 1.0, 600.0, step=1.0, key="L")

    st.subheader("Soil (single layer, u=0)")
    gamma = st.number_input("Unit weight γ (kN/m³)", 10.0, 30.0, step=0.5, key="gamma")
    c = st.number_input("Cohesion c (kPa)", 0.0, 500.0, step=0.5, key="c")
    phi = st.number_input("Friction φ (deg)", 0.0, 45.0, step=0.5, key="phi")

    st.subheader("Method & Discretization")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius (ordinary)"])
    n_slices = st.slider("Number of slices", 10, 200, key="n_slices")
    refine_levels = st.slider("Refine levels", 0, 3, key="refine_levels")
    top_k = st.slider("Top-K seeds per level", 1, 20, key="top_k")

    st.subheader("Search ranges")
    auto = st.checkbox("Auto ranges (recommended)", value=True)
    if auto:
        (x_left, x_right), (y_bottom, y_top), (Rmin, Rmax) = suggest_ranges(H, L)
        st.caption(f"Auto x:[{x_left:.1f},{x_right:.1f}], y:[{y_bottom:.1f},{y_top:.1f}], R:[{Rmin:.1f},{Rmax:.1f}]")
    else:
        # ensure keys only when needed
        (x0l, x0r), (y0b, y0t), (r0min, r0max) = suggest_ranges(H, L)
        ss = st.session_state
        ss.setdefault("x_left", x0l); ss.setdefault("x_right", x0r)
        ss.setdefault("y_bottom", y0b); ss.setdefault("y_top", y0t)
        ss.setdefault("Rmin", r0min); ss.setdefault("Rmax", r0max)
        x_left  = st.number_input("Center x min", -10000.0, 10000.0, step=1.0, key="x_left")
        x_right = st.number_input("Center x max", -10000.0, 10000.0, step=1.0, key="x_right")
        y_bottom = st.number_input("Center y min", -10000.0, 10000.0, step=1.0, key="y_bottom")
        y_top    = st.number_input("Center y max", -10000.0, 10000.0, step=1.0, key="y_top")
        Rmin     = st.number_input("Radius min", 1.0, 1e6, step=1.0, key="Rmin")
        Rmax     = st.number_input("Radius max", 1.0, 1e6, step=1.0, key="Rmax")

    st.subheader("Grid densities")
    nx = st.slider("Centers in x", 5, 60, key="nx")
    ny = st.slider("Centers in y", 3, 40, key="ny")
    nR = st.slider("Radius samples", 5, 80, key="nR")

    st.subheader("Depth filter (m)")
    st.caption("分位点ベースで極端に薄い円を除外（端部スライスの極薄に頑健）")
    h_min = st.number_input("h_min (m)", 0.0, 10000.0, step=0.1, key="h_min")
    h_max = st.number_input("h_max (m)", 0.5, 10000.0, step=0.5, key="h_max")
    pct   = st.slider("Depth percentile (%)", 0, 50, key="pct")

    st.subheader("Plot options")
    tick_step = st.number_input("Tick step for both axes (0=auto)", 0.0, 1000.0, step=0.5, key="tick_step")
    fill_section = st.checkbox("Fill closed cross-section", value=st.session_state.fill_section, key="fill_section")
    show_centers = st.checkbox("Show center grid points", value=st.session_state.show_centers, key="show_centers")

# guard
if h_max <= h_min:
    h_max = float(h_min) + 0.1
    st.info("Adjusted h_max to be greater than h_min.")

# compute (adaptive)
slope = Slope(H=H, L=L)
soil  = Soil(gamma=gamma, c=c, phi=phi)
method_key = "bishop" if method.lower().startswith("bishop") else "fellenius"

with st.spinner("Searching slip circles (coarse + refine)…"):
    result = grid_search_adaptive(
        slope, soil,
        x_center_range=(x_left, x_right),
        y_center_range=(y_bottom, y_top),
        R_range=(Rmin, Rmax),
        nx=nx, ny=ny, nR=nR,
        refine_levels=refine_levels, top_k=top_k, shrink=0.45,
        method=method_key,
        n_slices=n_slices,
        h_min=h_min, h_max=h_max, pct=float(pct)
    )

cands = result.get("candidates", [])
best  = result.get("best", None)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- 地形（等尺・閉じた横断図）---
    # ground line
    xg = np.array([0.0, L]); yg = np.array([H, 0.0])
    ax.plot(xg, yg, linewidth=2, label="Ground")

    # fill closed section to (0,0) if requested
    if fill_section:
        ax.fill([0.0, 0.0, L], [0.0, H, 0.0], alpha=0.15, label="Section")

    # candidates & best
    if not cands:
        ax.set_title("No valid slip circles. Adjust ranges/densities.")
    else:
        max_show = min(500, len(cands))
        default_show = min(180, len(cands))
        show_top = st.slider("Show first N candidates (for speed)", 1, max_show, default_show)

        for r in cands[:show_top]:
            xc, yc, R = r["xc"], r["yc"], r["R"]
            xs = np.linspace(r["x1"], r["x2"], 180)
            ys = yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
            ax.plot(xs, ys, linewidth=0.6, alpha=0.25)

        if best:
            xs = np.linspace(best["x1"], best["x2"], 400)
            ys = best["yc"] - np.sqrt(np.maximum(0.0, best["R"]**2 - (xs - best["xc"])**2))
            ax.plot(xs, ys, linewidth=2.8, color="red", label="Best")

        if show_centers:
            # show center grid in used box
            xs_grid = np.linspace(x_left, x_right, nx)
            ys_grid = np.linspace(y_bottom, y_top, ny)
            XX, YY = np.meshgrid(xs_grid, ys_grid)
            ax.scatter(XX.flatten(), YY.flatten(), s=9, alpha=0.65, label="Centers")

        ax.legend(loc="best", fontsize=9)
        ax.set_title("Slip circles (gray), Best (red)")

    # --- スケール：両軸の目盛り間隔を同一に ---
    ax.set_aspect("equal", adjustable="box")  # 1m=1m の等尺
    # limits with a small margin (centers may lie x<0)
    xmin = min(0.0, x_left) - 0.05*L
    xmax = L + 0.05*L
    ymin = 0.0
    ymax = max(H, y_top) + 0.05*max(H, 1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ticks: same spacing on both axes
    step = tick_step if tick_step and tick_step > 0 else nice_step(max(H, L))
    xticks = np.arange(0.0, xmax + step, step)
    yticks = np.arange(0.0, ymax + step, step)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, linestyle=":", alpha=0.4)

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
            st.dataframe(df.head(400), use_container_width=True)
            st.download_button("Download CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name="candidates.csv", mime="text/csv")

    with st.expander("Diagnostics"):
        st.write({
            "BUILD_TAG": BUILD_TAG,
            "H": H, "L": L, "gamma": gamma, "c": c, "phi": phi,
            "x_range": (float(x_left), float(x_right)),
            "y_range": (float(y_bottom), float(y_top)),
            "R_range": (float(Rmin), float(Rmax)),
            "nx_ny_nR": (int(nx), int(ny), int(nR)),
            "n_slices": int(n_slices),
            "h_min/h_max/pct": (float(h_min), float(h_max), float(pct)),
            "refine_levels": int(refine_levels), "top_k": int(top_k),
            "candidates": len(cands),
        })
