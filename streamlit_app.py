# streamlit_app.py
# -*- coding: utf-8 -*-
"""
安定板２（未補強）の UI を1ページで完結。
- 地形・材料の入力と保存
- （任意）地層／水位の表示は“ある場合のみ”重ね描き（計算には使わない）
- 円弧探索（グリッド＋局所絞り込み）
- 選択円弧と Fs の表示
- selected_slip / slices_best を session_state に保存（補強ページがそのまま参照できるように）
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stabi_lem import (
    Soil, CircleSlip,
    generate_slices_on_arc, bishop_fs_unreinforced,
    circle_xy_from_theta,
)

# ---------------- 基本設定 ----------------
st.set_page_config(page_title="Stabi｜安定板２（未補強）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"], index=0)
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

st.title("Stabi｜安定板２（未補強）")

# ---------------- 地形・材料の入力 ----------------
c1, c2, c3 = st.columns(3)
with c1:
    H = st.number_input("法高さ H [m]", 1.0, 200.0, float(st.session_state.get("H", 12.0)), 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 85.0, float(st.session_state.get("beta_deg", 35.0)), 0.5)
with c2:
    gamma = st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, float(st.session_state.get("gamma", 18.0)), 0.5)
    c = st.number_input("粘着力 c [kPa]", 0.0, 300.0, float(st.session_state.get("c", 10.0)), 1.0)
with c3:
    phi = st.number_input("内部摩擦角 φ [deg]", 0.0, 45.0, float(st.session_state.get("phi", 30.0)), 0.5)

soil = Soil(gamma=gamma, c=c, phi=phi)

beta_rad = math.radians(beta)
tanb = math.tan(beta_rad)

# 表示レンジ（未設定なら自動）
x_top  = H / max(tanb, 1e-6)
x_left  = float(st.session_state.get("x_left",  -0.1 * x_top))
x_right = float(st.session_state.get("x_right",  1.1 * x_top))
y_min   = float(st.session_state.get("y_min",   -1.2 * H))
y_max   = float(st.session_state.get("y_max",    1.2 * H))

# 保存ボタン（値だけ保存。計算仕様は変更しない）
if st.button("条件を保存", type="primary"):
    st.session_state.update({
        "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
        "x_left": x_left, "x_right": x_right, "y_min": y_min, "y_max": y_max
    })
    st.success("保存しました。")

# 地表関数
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * np.asarray(X, float)

# ---------------- 横断図プレビュー（表示のみ） ----------------
with st.expander("横断図プレビュー（表示のみ。計算には未反映）", expanded=True):
    Xv = np.linspace(x_left, x_right, 400)
    Yv = ground_y_at(Xv)

    figv, axv = plt.subplots(figsize=(9, 4))
    axv.plot(Xv, Yv, lw=2, label="Ground")

    # 地層表示：session_state["layers"] があれば水平帯で重畳（例：[{name, z_top, z_bot, color}]）
    layers = st.session_state.get("layers", None)
    if isinstance(layers, list):
        for L in layers:
            try:
                zt = float(L["z_top"]); zb = float(L["z_bot"])
                color = L.get("color", "#e8e8e8")
                axv.fill_between(Xv, np.minimum(Yv, zt), np.minimum(Yv, zb),
                                 color=color, alpha=0.35, step="mid", label=L.get("name","layer"))
            except Exception:
                pass

    # 水位表示：いずれかのキーがあれば線を入れる（水平 or 2点）
    if "water_y" in st.session_state:
        wy = float(st.session_state["water_y"])
        axv.plot([x_left, x_right], [wy, wy], linestyle="--", label="Water table")
    if {"xw1","yw1","xw2","yw2"} <= set(st.session_state.keys()):
        xw1,yw1 = float(st.session_state["xw1"]), float(st.session_state["yw1"])
        xw2,yw2 = float(st.session_state["xw2"]), float(st.session_state["yw2"])
        axv.plot([xw1,xw2],[yw1,yw2], linestyle="--", label="Water table")

    axv.set_aspect('equal','box')
    axv.set_xlim(x_left, x_right); axv.set_ylim(y_min, y_max)
    axv.grid(True, alpha=0.35); axv.set_xlabel("x [m]"); axv.set_ylabel("y [m]")
    if show_legend: axv.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(figv)

# ---------------- 円弧探索設定 ----------------
st.subheader("円弧探索（未補強）")
quality = st.selectbox("探索クオリティ", ["Fast", "Normal", "High"], index=1)
audit   = st.checkbox("探索センター可視化（Debug）", value=False)

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

    def eval_grid(nx, ny, nR, xl, xh, yl, yh, Rl, Rh, nslices):
        nonlocal best, centers
        xs = np.linspace(xl, xh, nx)
        ys = np.linspace(yl, yh, ny)
        Rs = np.linspace(Rl, Rh, nR)
        for xc in xs:
            for yc in ys:
                centers.append((xc, yc))
                for R in Rs:
                    slip = CircleSlip(xc=xc, yc=yc, R=R)
                    sls = generate_slices_on_arc(
                        ground_y_at, slip, n_slices=nslices,
                        x_min=x_left, x_max=x_right, soil_gamma=soil.gamma
                    )
                    if not sls:
                        continue
                    Fs_un = bishop_fs_unreinforced(sls, soil)
                    if not (Fs_un > 0 and np.isfinite(Fs_un)):
                        continue
                    if (best is None) or (Fs_un < best[0]):
                        best = (Fs_un, slip, sls)

    # 粗探索 → クイック → リファイン
    eval_grid(coarse_n[0], coarse_n[1], max(10, int((R_max - R_min) / H * 8)),
              xc_min, xc_max, yc_min, yc_max, R_min, R_max, nslices=36)

    if best is not None:
        _, s0, _ = best
        dx, dy, dR = 0.25 * x_top, 0.35 * H, 0.35 * H
        eval_grid(quick_n[0], quick_n[1], max(12, int((R_max - R_min) / H * 10)),
                  s0.xc - dx, s0.xc + dx,
                  s0.yc - dy, s0.yc + dy,
                  max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                  nslices=40)

    if best is not None:
        _, s0, _ = best
        dx, dy, dR = 0.12 * x_top, 0.20 * H, 0.22 * H
        eval_grid(quick_n[0] + 4, quick_n[1] + 4, refine_nR,
                  s0.xc - dx, s0.xc + dx,
                  s0.yc - dy, s0.yc + dy,
                  max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                  nslices=44)

    return best, centers

best, centers_seen = search_best_circle()

# ---------------- 結果表示＆保存 ----------------
if best is None:
    Fs_un = float("nan"); slip_best = None; slices_best = []
else:
    Fs_un, slip_best, slices_best = best

st.subheader("結果（未補強）")
st.metric("最小 Fs", f"{Fs_un:.3f}" if best else "—")

# 図
fig, ax = plt.subplots(figsize=(9, 6))
Xg = np.linspace(x_left, x_right, 400)
ax.plot(Xg, ground_y_at(Xg), label="Ground", linewidth=2)

if slip_best is not None:
    th = np.linspace(0, 2 * math.pi, 400)
    Xc, Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, "--", label="Selected slip circle")
if audit and centers_seen:
    xs = [p[0] for p in centers_seen]; ys = [p[1] for p in centers_seen]
    ax.scatter(xs, ys, s=8, alpha=0.25, label="Centers")

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.35)
ax.set_title(f"Fs_un(min)={Fs_un:.3f}" if best else "円弧が成立しません")
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

# 補強ページが読めるように保存（既存名）
if slip_best is not None:
    st.session_state["selected_slip"] = {
        "xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)
    }
if slices_best:
    st.session_state["slices_best"] = slices_best

with st.expander("セッション状態（確認用・読み取り専用）", expanded=False):
    st.write({
        "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
        "x_left": x_left, "x_right": x_right, "y_min": y_min, "y_max": y_max,
        "selected_slip": st.session_state.get("selected_slip", None),
        "slices_best": f"{len(st.session_state.get('slices_best', []))} slices"
    })
