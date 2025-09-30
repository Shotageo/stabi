# streamlit_app.py
from __future__ import annotations
import os, importlib.util
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

BUILD_TAG = "build-2025-09-30-17:20JST (rescue enabled)"

# safe import
try:
    from stabi_lem import Soil, Slope, search_entry_exit_adaptive
except Exception:
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("stabi_lem", os.path.join(here, "stabi_lem.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    Soil, Slope = mod.Soil, mod.Slope
    search_entry_exit_adaptive = mod.search_entry_exit_adaptive

st.set_page_config(page_title="Stabi LEM (Entry–Exit + Rescue)", layout="wide")
st.title("Stabi LEM：Bishop / Fellenius（Entry–Exit + Auto-Refine + Rescue）")
st.caption(f"🔧 {BUILD_TAG}")

def ensure_defaults():
    ss = st.session_state
    ss.setdefault("H", 20.0); ss.setdefault("L", 40.0)
    ss.setdefault("gamma", 18.0); ss.setdefault("c", 5.0); ss.setdefault("phi", 30.0)
    ss.setdefault("e0", 0.30); ss.setdefault("e1", 0.70)
    ss.setdefault("x0", 0.80); ss.setdefault("x1", 1.00)
    ss.setdefault("n_entry", 40); ss.setdefault("n_exit", 40)
    ss.setdefault("R_scale_min", 0.35); ss.setdefault("R_scale_max", 2.5); ss.setdefault("nR", 30)
    ss.setdefault("method", "Bishop (simplified)"); ss.setdefault("n_slices", 40)
    ss.setdefault("h_min", 0.2); ss.setdefault("h_max", 1000.0); ss.setdefault("pct", 15); ss.setdefault("min_eff_ratio", 0.6)
    ss.setdefault("refine_levels", 2); ss.setdefault("top_k", 10); ss.setdefault("shrink", 0.45)
    ss.setdefault("tick_step", 0.0); ss.setdefault("fill_section", True); ss.setdefault("show_entry_exit_band", True)
ensure_defaults()

def nice_step(max_val: float, target_ticks: int = 8) -> float:
    if max_val <= 0: return 1.0
    raw = max_val / max(target_ticks, 1)
    base = 10 ** int(np.floor(np.log10(raw)))
    for m in [1, 2, 5, 10]:
        if raw <= m * base:
            return m * base
    return 10 * base

with st.sidebar:
    st.subheader("Geometry")
    H = st.number_input("H (m)", 1.0, 1000.0, step=1.0, key="H")
    L = st.number_input("L (m)", 1.0, 2000.0, step=1.0, key="L")

    st.subheader("Soil (u=0)")
    gamma = st.number_input("γ (kN/m³)", 10.0, 30.0, step=0.5, key="gamma")
    c = st.number_input("c' (kPa)", 0.0, 500.0, step=0.5, key="c")
    phi = st.number_input("φ' (deg)", 0.0, 45.0, step=0.5, key="phi")

    st.subheader("Method / Discretization")
    method = st.selectbox("LEM", ["Bishop (simplified)", "Fellenius (ordinary)"], key="method")
    n_slices = st.slider("Number of slices", 10, 200, key="n_slices")

    st.subheader("Entry–Exit bands (s on surface 0–1)")
    e0 = st.number_input("Entry s0", 0.0, 1.0, step=0.01, key="e0")
    e1 = st.number_input("Entry s1", 0.0, 1.0, step=0.01, key="e1")
    x0 = st.number_input("Exit s0",  0.0, 1.0, step=0.01, key="x0")
    x1 = st.number_input("Exit s1",  0.0, 1.0, step=0.01, key="x1")
    n_entry = st.slider("Entry divisions", 5, 120, key="n_entry")
    n_exit  = st.slider("Exit divisions",  5, 120, key="n_exit")

    st.subheader("Radius range (diag scales)")
    R_scale_min = st.number_input("R_min scale", 0.10, 5.00, step=0.05, key="R_scale_min")
    R_scale_max = st.number_input("R_max scale", 0.30, 5.00, step=0.10, key="R_scale_max")
    nR = st.slider("Radius samples", 5, 120, key="nR")

    st.subheader("Filters")
    h_min = st.number_input("h_min (m, percentile)", 0.0, 10000.0, step=0.1, key="h_min")
    h_max = st.number_input("h_max (m)", 0.5, 10000.0, step=0.5, key="h_max")
    pct   = st.slider("Depth percentile (%)", 0, 50, key="pct")
    min_eff_ratio = st.slider("Min effective slice ratio", 10, 100, value=int(st.session_state.min_eff_ratio*100)) / 100.0
    st.session_state.min_eff_ratio = float(min_eff_ratio)

    st.subheader("Auto-Refine")
    refine_levels = st.slider("Levels", 0, 3, key="refine_levels")
    top_k        = st.slider("Top-K seeds", 1, 30, key="top_k")
    shrink       = st.slider("Shrink factor", 0.20, 0.90, value=float(st.session_state.shrink), step=0.05, key="shrink")

    st.subheader("Plot")
    tick_step = st.number_input("Tick step (0=auto)", 0.0, 10000.0, step=0.5, key="tick_step")
    fill_section = st.checkbox("Fill closed cross-section", value=st.session_state.fill_section, key="fill_section")
    show_entry_exit_band = st.checkbox("Show Entry/Exit bands", value=st.session_state.show_entry_exit_band, key="show_entry_exit_band")

# guards
if st.session_state.e1 < st.session_state.e0:
    st.session_state.e1 = float(st.session_state.e0)
if st.session_state.x1 < st.session_state.x0:
    st.session_state.x1 = float(st.session_state.x0)
if st.session_state.h_max <= st.session_state.h_min:
    st.session_state.h_max = float(st.session_state.h_min) + 0.1

# compute
slope = Slope(H=H, L=L); soil = Soil(gamma=gamma, c=c, phi=phi)
diag = float(np.hypot(H, L))
Rmin = max(1.0, float(st.session_state.R_scale_min) * diag)
Rmax = max(Rmin + 1.0, float(st.session_state.R_scale_max) * diag)

with st.spinner("Searching circular surfaces…"):
    result = search_entry_exit_adaptive(
        slope, soil,
        entry_s_range=(st.session_state.e0, st.session_state.e1),
        exit_s_range=(st.session_state.x0, st.session_state.x1),
        n_entry=st.session_state.n_entry, n_exit=st.session_state.n_exit,
        Rmin=Rmin, Rmax=Rmax, nR=st.session_state.nR,
        method=("bishop" if st.session_state.method.lower().startswith("bishop") else "fellenius"),
        n_slices=st.session_state.n_slices,
        h_min=st.session_state.h_min, h_max=st.session_state.h_max, pct=float(st.session_state.pct),
        min_eff_ratio=float(st.session_state.min_eff_ratio),
        refine_levels=st.session_state.refine_levels, top_k=st.session_state.top_k, shrink=float(st.session_state.shrink)
    )

cands = result.get("candidates", [])
best  = result.get("best", None)
rescue_used = result.get("rescue_used", False)
rescue_stage = result.get("rescue_stage", "")

col1, col2 = st.columns([2.0, 1.0], gap="large")

with col1:
    fig, ax = plt.subplots(figsize=(9, 6))

    # closed section
    if st.session_state.fill_section:
        ax.fill([0.0, 0.0, L], [0.0, H, 0.0], alpha=0.15, label="Section")

    # ground & bands
    if show_entry_exit_band:
        Ex0, Ey0 = slope.points_on_surface(st.session_state.e0)
        Ex1, Ey1 = slope.points_on_surface(st.session_state.e1)
        Xx0, Xy0 = slope.points_on_surface(st.session_state.x0)
        Xx1, Xy1 = slope.points_on_surface(st.session_state.x1)
        ax.plot([0.0, L], [H, 0.0], linewidth=2, color="k")
        ax.plot([Ex0, Ex1], [Ey0, Ey1], linewidth=6, alpha=0.25, color="tab:blue", label="Entry band")
        ax.plot([Xx0, Xx1], [Xy0, Xy1], linewidth=6, alpha=0.25, color="tab:orange", label="Exit band")
    else:
        ax.plot([0.0, L], [H, 0.0], linewidth=2, color="k", label="Ground")

    if not cands:
        ax.set_title("No valid slip circles after rescue. Broaden bands / radius range.")
    else:
        Fs_vals = np.array([r["Fs"] for r in cands], dtype=float)
        norm = colors.Normalize(vmin=float(Fs_vals.min()), vmax=float(Fs_vals.max()))
        cmap = cm.get_cmap("viridis")

        max_show = min(800, len(cands))
        default_show = min(300, len(cands))
        show_top = st.slider("Show first N candidates (speed)", 10, max_show, default_show)

        for r in cands[:show_top]:
            xs = np.linspace(r["x1"], r["x2"], 220)
            inside = r["R"] ** 2 - (xs - r["xc"]) ** 2
            ys = r["yc"] - np.sqrt(np.maximum(0.0, inside))
            ax.plot(xs, ys, linewidth=0.7, alpha=0.45, color=cmap(norm(r["Fs"])))

        if best:
            xs = np.linspace(best["x1"], best["x2"], 480)
            ys = best["yc"] - np.sqrt(np.maximum(0.0, best["R"] ** 2 - (xs - best["xc"]) ** 2))
            ax.plot(xs, ys, linewidth=3.0, color="red", label=f"Best (Fs={best['Fs']:.3f})")

        ax.legend(loc="best", fontsize=9)
        title = "Slip circles (colored by Fs), Best in red"
        if rescue_used:
            title += f"  —  rescue:{rescue_stage}"
        ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    xmin = -0.05 * L; xmax = L * 1.05
    ymin = 0.0; ymax = max(H, 1.0) * 1.08
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    step = st.session_state.tick_step if st.session_state.tick_step > 0 else (lambda v: (1.0 if v <= 8 else (2.0 if v <= 20 else (5.0 if v <= 60 else 10.0))))(max(H, L))
    ax.set_xticks(np.arange(0.0, xmax + step, step))
    ax.set_yticks(np.arange(0.0, ymax + step, step))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig)

with col2:
    if rescue_used:
        st.warning(f"Rescue search was triggered ({rescue_stage}). 入力が厳しすぎる/帯が狭すぎる可能性。")

    st.subheader("Results")
    if not cands:
        st.error("No valid slip circles found (even after rescue). Try widening bands or radius range.")
    else:
        st.metric("Candidates", f"{len(cands)}")
        if best:
            st.metric("Min Fs", f"{best['Fs']:.3f}")
            st.write(f"Center: (xc={best['xc']:.2f}, yc={best['yc']:.2f}),  R={best['R']:.2f}")

        with st.expander("Top candidates / CSV"):
            df = pd.DataFrame(cands).sort_values("Fs").reset_index(drop=True)
            st.dataframe(df.head(500), use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="candidates.csv", mime="text/csv")

    with st.expander("Diagnostics"):
        st.write({
            "BUILD_TAG": BUILD_TAG,
            "Geom": {"H": float(H), "L": float(L)},
            "Soil": {"gamma": float(gamma), "c'": float(c), "phi'": float(phi)},
            "Method": method, "n_slices": int(n_slices),
            "Entry band s": (float(st.session_state.e0), float(st.session_state.e1)),
            "Exit band s": (float(st.session_state.x0), float(st.session_state.x1)),
            "n_entry/n_exit/nR": (int(st.session_state.n_entry), int(st.session_state.n_exit), int(st.session_state.nR)),
            "R_range": (float(Rmin), float(Rmax)),
            "Filters": {"h_min": float(st.session_state.h_min), "h_max": float(st.session_state.h_max), "pct": int(st.session_state.pct),
                        "min_eff_ratio": float(st.session_state.min_eff_ratio)},
            "Refine": {"levels": int(st.session_state.refine_levels), "top_k": int(st.session_state.top_k), "shrink": float(st.session_state.shrink)},
            "rescue_used": rescue_used, "rescue_stage": rescue_stage,
            "candidates": len(cands)
        })