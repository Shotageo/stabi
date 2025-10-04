# streamlit_app.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stabi_lem import (
    Soil, CircleSlip,
    generate_slices_on_arc,
    bishop_fs_unreinforced,
    circle_xy_from_theta
)

# ===== Plot style =====
st.set_page_config(page_title="Stabi｜安定板２", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", value=True)
show_legend = st.sidebar.checkbox("Show legend", value=True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

st.title("Stabi｜安定板２：無補強の円弧探索")

# ===== 探索設定 =====
st.sidebar.header("探索設定")
quality = st.sidebar.selectbox("Quality", ["Fast", "Normal", "High"], index=1)
audit = st.sidebar.checkbox("Audit（センター可視化）", value=False)

# ===== 地形・材料 =====
st.subheader("地形・材料")
cA, cB, cC = st.columns(3)
with cA:
    H  = st.number_input("法高さ H [m]", 1.0, 100.0, float(st.session_state.get("H", 12.0)), 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 80.0, float(st.session_state.get("beta_deg", 35.0)), 0.5)
with cB:
    gamma = st.number_input("γ [kN/m³]", 10.0, 30.0, float(st.session_state.get("gamma", 18.0)), 0.5)
    c = st.number_input("c [kPa]", 0.0, 300.0, float(st.session_state.get("c", st.session_state.get("c_kpa", 10.0))), 1.0)
with cC:
    phi = st.number_input("φ [deg]", 0.0, 45.0, float(st.session_state.get("phi", st.session_state.get("phi_deg", 30.0))), 0.5)

soil = Soil(gamma=gamma, c=c, phi=phi)

# 地表（配列安全）
beta_rad = math.radians(beta)
tanb = math.tan(beta_rad)
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

# 表示レンジ（他ページへ保存）
x_top  = H / max(tanb, 1e-6)
x_left, x_right = -0.1 * x_top, 1.1 * x_top
y_min, y_max = -1.2 * H, 1.2 * H
st.session_state["H"] = H
st.session_state["beta_deg"] = beta
st.session_state["gamma"] = gamma
st.session_state["c"] = c
st.session_state["phi"] = phi
st.session_state["x_left"] = x_left
st.session_state["x_right"] = x_right
st.session_state["y_min"] = y_min
st.session_state["y_max"] = y_max

# ===== 円弧探索（Coarse→Quick→Refine） =====
def search_best_circle():
    if quality == "Fast":
        coarse_n = (18, 10); quick_n = (12, 8); refine_nR = 12
    elif quality == "High":
        coarse_n = (42, 20); quick_n = (28, 14); refine_nR = 24
    else:
        coarse_n = (30, 14); quick_n = (20, 10); refine_nR = 18

    xc_min, xc_max = -0.2 * x_top, 1.2 * x_top
    yc_min, yc_max = -2.0 * H, 0.2 * H
    R_min, R_max   = 0.6 * H, 2.2 * H

    best = None  # (Fs, slip, slices)
    centers = []

    def eval_grid(nx, ny, nR, xc_lo, xc_hi, yc_lo, yc_hi, R_lo, R_hi, nslices):
        nonlocal best, centers
        xs = np.linspace(xc_lo, xc_hi, nx)
        ys = np.linspace(yc_lo, yc_hi, ny)
        Rs = np.linspace(R_lo, R_hi, nR)
        for xc in xs:
            for yc in ys:
                centers.append((xc, yc))
                for R in Rs:
                    slip = CircleSlip(xc=xc, yc=yc, R=R)
                    slices = generate_slices_on_arc(
                        ground_y_at, slip, n_slices=nslices,
                        x_min=x_left, x_max=x_right, soil_gamma=soil.gamma
                    )
                    if not slices:
                        continue
                    Fs_un = bishop_fs_unreinforced(slices, soil)
                    if not (Fs_un > 0 and np.isfinite(Fs_un)):
                        continue
                    if (best is None) or (Fs_un < best[0]):
                        best = (Fs_un, slip, slices)

    eval_grid(coarse_n[0], coarse_n[1], max(10, int((R_max-R_min)/H*8)),
              xc_min, xc_max, yc_min, yc_max, R_min, R_max, nslices=36)

    if best is not None:
        _, s0, _ = best
        dx = 0.25 * x_top; dy = 0.35 * H; dR = 0.35 * H
        eval_grid(quick_n[0], quick_n[1], max(12, int((R_max-R_min)/H*10)),
                  s0.xc - dx, s0.xc + dx,
                  s0.yc - dy, s0.yc + dy,
                  max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                  nslices=40)

    if best is not None:
        _, s0, _ = best
        dx = 0.12 * x_top; dy = 0.20 * H; dR = 0.22 * H
        eval_grid(quick_n[0]+4, quick_n[1]+4, refine_nR,
                  s0.xc - dx, s0.xc + dx,
                  s0.yc - dy, s0.yc + dy,
                  max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                  nslices=44)
    return best, centers

best, centers_seen = search_best_circle()

# ===== 結果・選択円弧の保存 =====
if best is None:
    Fs_un = float("nan"); slip_best = None; slices_best = []
else:
    Fs_un, slip_best, slices_best = best
    st.session_state["selected_slip"] = {"xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)}

st.subheader("結果（無補強）")
st.metric("最小 Fs", f"{Fs_un:.3f}" if best else "—")

# ===== 可視化 =====
fig, ax = plt.subplots(figsize=(9, 6))
Xg = np.linspace(x_left, x_right, 400)
ax.plot(Xg, ground_y_at(Xg), label="Ground", linewidth=2)

if slip_best is not None:
    th = np.linspace(0, 2*math.pi, 400)
    Xc, Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, "--", label="Selected slip circle")

if audit and centers_seen:
    xs = [p[0] for p in centers_seen]; ys = [p[1] for p in centers_seen]
    ax.scatter(xs, ys, s=8, alpha=0.25, label="Coarse centers")

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.4)
ax.set_title(f"Fs_un(min)={Fs_un:.3f}" if best else "円弧が成立しません（地表との土塊なし）")
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

st.caption("※ 次のページ［Soil Nail Reinforcement］で、この“選択円弧”に補強を適用します。")
