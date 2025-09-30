# streamlit_app.py
from __future__ import annotations
import os, importlib.util
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

BUILD_TAG = "build-2025-10-01-GridBasic"

# ---- safe import（stabi_lem.pyを直読みフォールバック） ----
try:
    from stabi_lem import Soil, Slope, search_center_grid
except Exception:
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("stabi_lem", os.path.join(here, "stabi_lem.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    Soil, Slope, search_center_grid = mod.Soil, mod.Slope, mod.search_center_grid

st.set_page_config(page_title="Stabi LEM (Center-Grid Basic)", layout="wide")
st.title("Stabi LEM：Center-Grid Basic（Bishop / Fellenius）")
st.caption(f"🔧 {BUILD_TAG}")

# -------------------- Defaults --------------------
def ensure_defaults():
    ss = st.session_state
    ss.setdefault("H", 20.0); ss.setdefault("L", 40.0)
    ss.setdefault("gamma", 18.0); ss.setdefault("c", 5.0); ss.setdefault("phi", 30.0)
    ss.setdefault("method", "Bishop (simplified)"); ss.setdefault("n_slices", 40)

    # Center-grid（範囲は H,L から自動提案）
    ss.setdefault("x_min_scale", -0.50); ss.setdefault("x_max_scale", 0.30)
    ss.setdefault("y_min_scale",  0.30); ss.setdefault("y_max_scale", 2.00)
    ss.setdefault("nx", 14); ss.setdefault("ny", 10)

    # 半径は対角長スケール
    ss.setdefault("R_min_scale", 0.30); ss.setdefault("R_max_scale", 3.00); ss.setdefault("nR", 40)

    # Plot toggles
    ss.setdefault("show_grid_rect", True); ss.setdefault("show_centers", True)
    ss.setdefault("tick_step", 0.0); ss.setdefault("fill_section", True)
ensure_defaults()

def nice_step(max_val: float, target_ticks: int = 8) -> float:
    if max_val <= 0: return 1.0
    raw = max_val / max(target_ticks, 1)
    base = 10 ** int(np.floor(np.log10(raw)))
    for m in [1, 2, 5, 10]:
        if raw <= m * base:
            return m * base
    return 10 * base

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Geometry")
    H = st.number_input("H (m)", 1.0, 1000.0, value=float(st.session_state.H), step=1.0, key="H")
    L = st.number_input("L (m)", 1.0, 2000.0, value=float(st.session_state.L), step=1.0, key="L")

    st.subheader("Soil (u=0)")
    gamma = st.number_input("γ (kN/m³)", 10.0, 30.0, value=float(st.session_state.gamma), step=0.5, key="gamma")
    c = st.number_input("c' (kPa)", 0.0, 500.0, value=float(st.session_state.c), step=0.5, key="c")
    phi = st.number_input("φ' (deg)", 0.0, 45.0, value=float(st.session_state.phi), step=0.5, key="phi")

    st.subheader("Method / Discretization")
    method = st.selectbox("LEM", ["Bishop (simplified)", "Fellenius (ordinary)"], index=0 if st.session_state.method.startswith("B") else 1, key="method")
    n_slices = st.slider("Number of slices", 10, 200, value=int(st.session_state.n_slices), key="n_slices")

    st.subheader("Center-Grid (relative to H, L)")
    st.caption("x: Lを基準, y: Hを基準としたスケール。矩形内に中心格子を敷設。")
    x_min_scale = st.number_input("x_min / L", -2.0, 2.0, value=float(st.session_state.x_min_scale), step=0.05, key="x_min_scale")
    x_max_scale = st.number_input("x_max / L", -2.0, 2.0, value=float(st.session_state.x_max_scale), step=0.05, key="x_max_scale")
    y_min_scale = st.number_input("y_min / H",  0.0, 10.0, value=float(st.session_state.y_min_scale), step=0.05, key="y_min_scale")
    y_max_scale = st.number_input("y_max / H",  0.1, 20.0, value=float(st.session_state.y_max_scale), step=0.1, key="y_max_scale")
    nx = st.slider("nx (centers in x)", 5, 60, value=int(st.session_state.nx), key="nx")
    ny = st.slider("ny (centers in y)", 5, 60, value=int(st.session_state.ny), key="ny")

    st.subheader("Radius range (diag scales)")
    R_min_scale = st.number_input("R_min / diag", 0.10, 5.00, value=float(st.session_state.R_min_scale), step=0.05, key="R_min_scale")
    R_max_scale = st.number_input("R_max / diag", 0.20, 6.00, value=float(st.session_state.R_max_scale), step=0.10, key="R_max_scale")
    nR = st.slider("Radius samples", 5, 120, value=int(st.session_state.nR), key="nR")

    st.subheader("Plot")
    show_grid_rect = st.checkbox("Show grid rectangle", value=st.session_state.show_grid_rect, key="show_grid_rect")
    show_centers   = st.checkbox("Show grid centers", value=st.session_state.show_centers, key="show_centers")
    tick_step = st.number_input("Tick step (0=auto)", 0.0, 10000.0, value=float(st.session_state.tick_step), step=0.5, key="tick_step")
    fill_section = st.checkbox("Fill closed cross-section", value=st.session_state.fill_section, key="fill_section")

# guards
if st.session_state.x_max_scale <= st.session_state.x_min_scale:
    st.session_state.x_max_scale = st.session_state.x_min_scale + 0.05
if st.session_state.y_max_scale <= st.session_state.y_min_scale:
    st.session_state.y_max_scale = st.session_state.y_min_scale + 0.05
if st.session_state.R_max_scale <= st.session_state.R_min_scale:
    st.session_state.R_max_scale = st.session_state.R_min_scale + 0.10

# -------------------- Compute --------------------
slope = Slope(H=float(st.session_state.H), L=float(st.session_state.L))
soil  = Soil(gamma=float(st.session_state.gamma), c=float(st.session_state.c), phi=float(st.session_state.phi))
diag = float(np.hypot(slope.H, slope.L))

x_rng = (float(st.session_state.x_min_scale)*slope.L, float(st.session_state.x_max_scale)*slope.L)
y_rng = (float(st.session_state.y_min_scale)*slope.H, float(st.session_state.y_max_scale)*slope.H)
R_rng = (max(1.0, float(st.session_state.R_min_scale)*diag), max(1.1, float(st.session_state.R_max_scale)*diag))

with st.spinner("Searching circular surfaces by Center-Grid…"):
    result = search_center_grid(
        slope, soil,
        x_center_range=x_rng, y_center_range=y_rng,
        R_range=R_rng, nx=int(st.session_state.nx), ny=int(st.session_state.ny), nR=int(st.session_state.nR),
        method=("bishop" if st.session_state.method.lower().startswith("bishop") else "fellenius"),
        n_slices=int(st.session_state.n_slices),
        refine_levels=1, top_k=20, shrink=0.5
    )

cands = result.get("candidates", [])
best  = result.get("best", None)
rescue = result.get("rescue", "")

# -------------------- Plot --------------------
col1, col2 = st.columns([2.0, 1.0], gap="large")
with col1:
    fig, ax = plt.subplots(figsize=(9, 6))

    # 閉じた断面の塗り
    if st.session_state.fill_section:
        ax.fill([0.0, 0.0, slope.L], [0.0, slope.H, 0.0], alpha=0.15, label="Section")
    # 地表線
    ax.plot([0.0, slope.L], [slope.H, 0.0], color="k", linewidth=2, label="Ground")

    # グリッド矩形＆中心点
    if st.session_state.show_grid_rect:
        ax.plot([x_rng[0], x_rng[1], x_rng[1], x_rng[0], x_rng[0]],
                [y_rng[0], y_rng[0], y_rng[1], y_rng[1], y_rng[0]],
                linestyle="--", alpha=0.6)
    if st.session_state.show_centers:
        xs = np.linspace(x_rng[0], x_rng[1], int(st.session_state.nx))
        ys = np.linspace(y_rng[0], y_rng[1], int(st.session_state.ny))
        XX, YY = np.meshgrid(xs, ys)
        ax.scatter(XX.ravel(), YY.ravel(), s=10, alpha=0.6)

    if not cands:
        ttl = "No valid slip circles."
        if rescue:
            ttl += f" (rescue tried: {rescue})"
        ax.set_title(ttl)
    else:
        Fs_vals = np.array([r["Fs"] for r in cands], dtype=float)
        norm = colors.Normalize(vmin=float(Fs_vals.min()), vmax=float(Fs_vals.max()))
        cmap = cm.get_cmap("viridis")
        max_show = min(800, len(cands))
        default_show = min(300, len(cands))
        show_top = st.slider("Show first N candidates", 10, max_show, default_show)

        for r in cands[:show_top]:
            xs = np.linspace(r["x1"], r["x2"], 220)
            ys = r["yc"] - np.sqrt(np.maximum(0.0, r["R"]**2 - (xs - r["xc"])**2))
            ax.plot(xs, ys, linewidth=0.7, alpha=0.45, color=cmap(norm(r["Fs"])))

        if best:
            xs = np.linspace(best["x1"], best["x2"], 480)
            ys = best["yc"] - np.sqrt(np.maximum(0.0, best["R"]**2 - (xs - best["xc"])**2))
            ax.plot(xs, ys, linewidth=3.0, color="red", label=f"Best (Fs={best['Fs']:.3f})")

        label = "Slip circles (Fs-colored), Best in red"
        if rescue:
            label += f" — rescue:{rescue}"
        ax.set_title(label)
        ax.legend(loc="best", fontsize=9)

    # 等尺＆同一目盛
    ax.set_aspect("equal", adjustable="box")
    xmin = min(-0.1*slope.L, x_rng[0]); xmax = max(1.05*slope.L, x_rng[1])
    ymin = 0.0; ymax = max(slope.H*1.08, y_rng[1]*1.02)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    step = float(st.session_state.tick_step) if st.session_state.tick_step > 0 else nice_step(max(slope.H, slope.L))
    ax.set_xticks(np.arange(0.0, xmax + step, step))
    ax.set_yticks(np.arange(0.0, ymax + step, step))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    st.pyplot(fig)

with col2:
    st.subheader("Results")
    if not cands:
        if rescue:
            st.warning(f"No valid circles from basic search. Rescue used: {rescue}")
        else:
            st.error("No valid circles. Enlarge grid or radius range.")
    else:
        st.metric("Candidates", f"{len(cands)}")
        if best:
            st.metric("Min Fs", f"{best['Fs']:.3f}")
            st.write(f"Center (xc, yc) = ({best['xc']:.2f}, {best['yc']:.2f}),  R = {best['R']:.2f}")

        with st.expander("Top candidates / CSV"):
            df = pd.DataFrame(cands).sort_values("Fs").reset_index(drop=True)
            st.dataframe(df.head(500), use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="candidates.csv", mime="text/csv")

    with st.expander("Diagnostics"):
        st.write({
            "BUILD_TAG": BUILD_TAG,
            "Geom": {"H": float(slope.H), "L": float(slope.L)},
            "Soil": {"gamma": float(soil.gamma), "c'": float(soil.c), "phi'": float(soil.phi)},
            "Method": st.session_state.method, "n_slices": int(st.session_state.n_slices),
            "CenterGrid": {"x_rng": x_rng, "y_rng": y_rng, "nx": int(st.session_state.nx), "ny": int(st.session_state.ny)},
            "Radius": {"R_rng": R_rng, "nR": int(st.session_state.nR)},
            "rescue": rescue,
            "candidates": len(cands)
        })