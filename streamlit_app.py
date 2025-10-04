# streamlit_app.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from time import time

from stabi_lem import (
    Soil, CircleSlip, Nail,
    generate_slices_on_arc,
    bishop_fs_unreinforced, bishop_fs_with_nails,
    circle_xy_from_theta
)

# ---------------- Plot style（Theme/Tight layout/Legend切替：既定） ----------------
st.set_page_config(page_title="Stabi｜安定板２（統合・一体化）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", value=True)
show_legend = st.sidebar.checkbox("Show legend", value=True)
plt.style.use("dark_background" if theme == "dark_background" else "default")
# ---------------------------------------------------------------------------------

st.title("Stabi｜安定板２：最小Fs円弧 → ソイルネイルで補強（統合版）")

# ====== 探索設定（安定板２風） ======
st.sidebar.header("探索設定")
quality = st.sidebar.selectbox("Quality", ["Fast", "Normal", "High"], index=1)
budget_coarse = st.sidebar.number_input("Budget Coarse [s]", value=0.8, step=0.1, min_value=0.1)
budget_quick  = st.sidebar.number_input("Budget Quick  [s]", value=1.2, step=0.1, min_value=0.1)
audit = st.sidebar.checkbox("Audit（センター可視化）", value=False)

# ====== 地形・材料 ======
st.subheader("地形・材料")
cA, cB, cC = st.columns(3)
with cA:
    H  = st.number_input("法高さ H [m]", 1.0, 100.0, 12.0, 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 80.0, 35.0, 0.5)
with cB:
    gamma = st.number_input("γ [kN/m³]", 10.0, 30.0, 18.0, 0.5)
    c = st.number_input("c [kPa]", 0.0, 300.0, 10.0, 1.0)
with cC:
    phi = st.number_input("φ [deg]", 0.0, 45.0, 30.0, 0.5)

soil = Soil(gamma=gamma, c=c, phi=phi)

# 地表（配列安全版）
beta_rad = math.radians(beta)
tanb = math.tan(beta_rad)
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

# 画面範囲
x_top = H / max(tanb, 1e-6)
x_left, x_right = -0.1*x_top, x_top*1.1
y_min, y_max = -H*1.2, H*1.2

# ====== ソイルネイル（線分管理） ======
st.subheader("ソイルネイル（線分で表現）")
if "nails" not in st.session_state:
    st.session_state.nails = []

with st.expander("ネイルを追加"):
    d1, d2, d3 = st.columns(3)
    with d1:
        x1 = st.number_input("x1 [m]", -200.0, 200.0, max(1.0, 0.15*x_top), 0.1, key="nx1")
        y1 = st.number_input("y1 [m]", -200.0, 200.0, H*0.55, 0.1, key="ny1")
        spacing = st.number_input("spacing [m]", 0.05, 5.0, 1.5, 0.05, key="nsp")
    with d2:
        x2 = st.number_input("x2 [m]", -200.0, 200.0, max(1.0, 0.55*x_top), 0.1, key="nx2")
        y2 = st.number_input("y2 [m]", -200.0, 200.0, H*0.7, 0.1, key="ny2")
        T_yield = st.number_input("T_yield [kN/本]", 10.0, 3000.0, 200.0, 10.0, key="nyld")
    with d3:
        bond = st.number_input("bond_strength [kN/m]", 1.0, 500.0, 80.0, 1.0, key="nbnd")
        emb  = st.number_input("有効定着長(片側) [m]", 0.1, 5.0, 0.5, 0.1, key="nemb")
        if st.button("＋このネイルを追加", type="primary"):
            st.session_state.nails.append(Nail(x1=x1, y1=y1, x2=x2, y2=y2,
                                               spacing=spacing, T_yield=T_yield,
                                               bond_strength=bond, embed_length_each_side=emb))

if st.session_state.nails:
    idx = st.selectbox("削除対象（インデックス）", list(range(len(st.session_state.nails))), index=0)
    cdel1, cdel2 = st.columns(2)
    with cdel1:
        if st.button("選択ネイルを削除"):
            st.session_state.nails.pop(idx)
    with cdel2:
        if st.button("全削除"):
            st.session_state.nails.clear()

# ====== 円弧探索（Coarse→Quick→Refine / 既存ロジック風の簡易実装） ======
def search_best_circle():
    # “質”によってグリッド解像度を可変（時間バジェットは目安）
    if quality == "Fast":
        coarse_n = (18, 10); quick_n = (12, 8); refine_nR = 12
    elif quality == "High":
        coarse_n = (42, 20); quick_n = (28, 14); refine_nR = 24
    else:  # Normal
        coarse_n = (30, 14); quick_n = (20, 10); refine_nR = 18

    # 探索窓：中心は法尻より左〜右、深さは -2H〜0付近
    xc_min, xc_max = -0.2*x_top, 1.2*x_top
    yc_min, yc_max = -2.0*H, 0.2*H
    R_min, R_max = 0.6*H, 2.2*H

    best = None  # (Fs, CircleSlip)

    def eval_grid(nx, ny, nR, xc_lo, xc_hi, yc_lo, yc_hi, R_lo, R_hi):
        nonlocal best
        xs = np.linspace(xc_lo, xc_hi, nx)
        ys = np.linspace(yc_lo, yc_hi, ny)
        Rs = np.linspace(R_lo, R_hi, nR)
        centers_record = []
        for xc in xs:
            for yc in ys:
                centers_record.append((xc, yc))
                for R in Rs:
                    slip = CircleSlip(xc=xc, yc=yc, R=R)
                    # 交差がなければ無視
                    slices = generate_slices_on_arc(ground_y_at, slip, n_slices=36, 
                                                    x_min=x_left, x_max=x_right, soil_gamma=soil.gamma)
                    if not slices:
                        continue
                    Fs_un = bishop_fs_unreinforced(slices, soil)
                    if not (Fs_un > 0 and np.isfinite(Fs_un)):
                        continue
                    if (best is None) or (Fs_un < best[0]):
                        best = (Fs_un, slip, slices)
        return centers_record

    # Coarse
    t0 = time()
    coarse_centers = eval_grid(coarse_n[0], coarse_n[1], max(10, int((R_max-R_min)/H*8)), 
                               xc_min, xc_max, yc_min, yc_max, R_min, R_max)
    # Quick（best周辺をズーム）
    if best is not None:
        _, slip_best, _ = best
        dx = 0.25*x_top; dy = 0.35*H; dR = 0.35*H
        quick_centers = eval_grid(quick_n[0], quick_n[1], max(12, int((R_max-R_min)/H*10)),
                                  slip_best.xc-dx, slip_best.xc+dx,
                                  slip_best.yc-dy, slip_best.yc+dy,
                                  max(R_min, slip_best.R-dR), min(R_max, slip_best.R+dR))
    else:
        quick_centers = []

    # Refine（さらに絞る）
    if best is not None:
        _, slip_best, _ = best
        dx = 0.12*x_top; dy = 0.20*H; dR = 0.22*H
        _ = eval_grid(quick_n[0]+4, quick_n[1]+4, refine_nR,
                      slip_best.xc-dx, slip_best.xc+dx,
                      slip_best.yc-dy, slip_best.yc+dy,
                      max(R_min, slip_best.R-dR), min(R_max, slip_best.R+dR))
    return best, coarse_centers

best, centers_seen = search_best_circle()

# ====== 結果（無補強最小円弧 → 補強後Fs） ======
if best is None:
    Fs_un = float("nan")
    Fs_re = float("nan")
    slip_best = None
    slices_best = []
else:
    Fs_un, slip_best, slices_best = best
    Fs_re = bishop_fs_with_nails(slices_best, soil, slip_best, st.session_state.nails)

st.subheader("結果")
cR1, cR2 = st.columns(2)
with cR1:
    st.metric("未補強 最小Fs", f"{Fs_un:.3f}" if best else "—")
with cR2:
    st.metric("補強後 Fs（選択円弧に適用）", f"{Fs_re:.3f}" if best else "—")

# ====== 可視化（既存と同じ：選択された円弧をそのまま描画） ======
fig, ax = plt.subplots(figsize=(9, 6))

# 地表
Xg = np.linspace(x_left, x_right, 400)
Yg = ground_y_at(Xg)
ax.plot(Xg, Yg, label="Ground", linewidth=2)

# 最小円弧（そのまま全周表示。既存の見せ方に合わせる）
if slip_best is not None:
    th = np.linspace(0, 2*math.pi, 400)
    Xc, Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, linestyle="--", label="Selected slip circle")

# ネイル
for i, nl in enumerate(st.session_state.nails):
    ax.plot([nl.x1, nl.x2], [nl.y1, nl.y2], linewidth=2, label=f"Nail {i}" if show_legend else None)

# Audit：センター可視化（薄い点）
if audit and centers_seen:
    xs = [p[0] for p in centers_seen]; ys = [p[1] for p in centers_seen]
    ax.scatter(xs, ys, s=8, alpha=0.25, label="Coarse centers")

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.4)
title_txt = f"Fs_un(min)={Fs_un:.3f} / Fs_re(on selected)={Fs_re:.3f}" if best else "円弧が成立しません（地表との土塊なし）"
ax.set_title(title_txt)
if show_legend:
    ax.legend(loc="best")
if tight:
    plt.tight_layout()
st.pyplot(fig)
