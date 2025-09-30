# streamlit_app.py
from __future__ import annotations
import streamlit as st
import numpy as np, math
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL, make_ground_example,
    arcs_from_center_by_entries
)

st.set_page_config(page_title="Stabi LEM - Lightweight Fan", layout="wide")
st.title("Stabi LEM（軽量描画レイヤ・センター固定／首振り）")

# ---------- Sidebar: 断面 & パラメータ ----------
with st.sidebar:
    st.header("Geometry & Soil")
    H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
    L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)

    preset = st.selectbox("Ground preset", ["3-seg berm (default)"])
    if preset == "3-seg berm (default)":
        pl = make_ground_example(H, L)

    gamma = st.number_input("γ (kN/m³)", 10.0, 25.0, 18.0, 0.5)
    cohesion = st.number_input("c (kPa)", 0.0, 200.0, 5.0, 0.5)
    phi = st.number_input("φ (deg)", 0.0, 45.0, 30.0, 0.5)
    soil = Soil(gamma=gamma, c=cohesion, phi=phi)

    st.header("Center selection")
    x_min = st.number_input("Center x min", 0.20*L, 1.50*L, 0.25*L, 0.05*L)
    x_max = st.number_input("Center x max", 0.30*L, 2.00*L, 1.15*L, 0.05*L)
    y_min = st.number_input("Center y min", 0.80*H, 3.00*H, 1.60*H, 0.10*H,
                            help="地表から十分離したければ 1.5H〜2.0H など")
    y_max = st.number_input("Center y max", 1.00*H, 3.50*H, 2.20*H, 0.10*H)
    nx = st.slider("Grid nx", 6, 30, 14)
    ny = st.slider("Grid ny", 4, 20, 9)

    st.header("Fan parameters (lightweight)")
    method = st.selectbox("Method", ["Bishop (simplified)", "Fellenius"])
    n_entries = st.slider("Entry samples on ground", 100, 1200, 600, 50)
    depth_min = st.number_input("Depth min (m)", 0.0, 20.0, 0.5, 0.5)
    depth_max = st.number_input("Depth max (m)", 0.5, 30.0, 4.0, 0.5)
    show_k    = st.slider("Show top-K arcs (thin)", 10, 300, 120, 10)
    top_thick = st.slider("Emphasize top-N (thick)", 1, 20, 12, 1)
    show_radii = st.checkbox("Show radii to both ends", True)

    st.caption("※軽量化：計算は逐次→描画して即破棄。配列・図は保持しません。")

# ---------- Center grid ----------
xs = np.linspace(x_min, x_max, nx)
ys = np.linspace(y_min, y_max, ny)
centers = [(float(xc), float(yc)) for yc in ys for xc in xs]

# ---------- センターの選び方 ----------
mode = st.radio("Center picking", ["Max arcs (robust)", "Min Fs (aggressive)"], horizontal=True)

def count_arcs_for_center(center):
    """描画せずに個数と最小Fsだけ拾う（軽量）"""
    xc, yc = center
    cnt = 0
    Fs_min = None
    for x1, x2, R, Fs in arcs_from_center_by_entries(
        pl, soil, xc, yc, n_entries=n_entries, method=method,
        depth_min=depth_min, depth_max=depth_max
    ):
        cnt += 1
        if (Fs_min is None) or (Fs < Fs_min):
            Fs_min = Fs
        # 早めに抜けたいときはここに上限を入れる
    return cnt, (Fs_min if Fs_min is not None else float("inf"))

with st.spinner("Scanning centers (light) ..."):
    best_center = None
    if mode == "Max arcs (robust)":
        best_score = -1
        for c in centers:
            cnt, _ = count_arcs_for_center(c)
            if cnt > best_score:
                best_score, best_center = cnt, c
    else:  # Min Fs (aggressive)
        best_val = float("inf"); best_cnt = 0
        for c in centers:
            cnt, Fs_min = count_arcs_for_center(c)
            if Fs_min < best_val or (Fs_min == best_val and cnt > best_cnt):
                best_val, best_cnt, best_center = Fs_min, cnt, c

# ---------- 扇状を軽量描画 ----------
xc, yc = best_center
fig, ax = plt.subplots(figsize=(10, 7))

# 閉じた断面（塗り）
secX = np.array([0.0, 0.0, L, L, 0.0]); secY = np.array([0.0, H, H, 0.0, 0.0])
ax.fill(secX, secY, alpha=0.10, label="Section")
# 地表
ax.plot(pl.X, pl.Y, linewidth=2.2, label="Ground")

# グリッド矩形＆点
ax.plot([x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min], linestyle="--", alpha=0.6, label="Center-grid")
XX, YY = np.meshgrid(xs, ys)
ax.scatter(XX.ravel(), YY.ravel(), s=12, alpha=0.5)
ax.scatter([xc], [yc], s=65, marker="s", label="Chosen center")

# 逐次で top-K を抽出（配列保持せず）
import heapq
top_heap: list[tuple[float, tuple[float,float,float]]] = []  # (-Fs, (x1,x2,R))
for x1, x2, R, Fs in arcs_from_center_by_entries(
    pl, soil, xc, yc, n_entries=n_entries, method=method,
    depth_min=depth_min, depth_max=depth_max
):
    # heapはFs小さい方を残す
    heapq.heappush(top_heap, (-Fs, (x1, x2, R)))
    if len(top_heap) > show_k:
        heapq.heappop(top_heap)

# Fs 昇順にして描画
top_list = sorted([(-fsneg, dat) for fsneg, dat in top_heap], key=lambda t: t[0])
# 薄線でtop-K全て、太線で上位N
thin_all = top_list
thick_sel = top_list[:min(top_thick, len(top_list))]

for Fs, (x1, x2, R) in thin_all:
    xs_line = np.linspace(x1, x2, 200)  # 表示は間引き
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=0.6, alpha=0.30)
    if show_radii:
        # 端点の地表yを都度算出（配列保持しない）
        y1 = float(pl.y_at(x1)); y2 = float(pl.y_at(x2))
        ax.plot([xc, x1], [yc, y1], linewidth=0.35, alpha=0.25)
        ax.plot([xc, x2], [yc, y2], linewidth=0.35, alpha=0.25)

for Fs, (x1, x2, R) in thick_sel:
    xs_line = np.linspace(x1, x2, 400)
    ys_line = yc - np.sqrt(np.maximum(0.0, R*R - (xs_line - xc)**2))
    ax.plot(xs_line, ys_line, linewidth=2.6, label=None)

title = f"Center=({xc:.2f},{yc:.2f}) • Method={method} • Shown={len(top_list)} arcs (thin), top {len(thick_sel)} thick"
ax.set_title(title)
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.05*L, 1.18*L); ax.set_ylim(0.0, 2.30*H)
ax.grid(True); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="upper right", fontsize=9)

st.pyplot(fig, use_container_width=True)
plt.close(fig)  # ← 重要：図オブジェクトを即時破棄
