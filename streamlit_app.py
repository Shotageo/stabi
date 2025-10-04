# streamlit_app.py
# -*- coding: utf-8 -*-
"""
I：安定板２ 完全版（multi-UI／補強はモック）
- 1) 地形・材料  … 入力＆保存
- 2) 地層         … 任意で追加・横断図に重畳（計算には未使用）
- 3) 水位         … 任意で追加・横断図に重畳（計算には未使用）
- 4) 円弧探索     … 無補強（最小Fs円弧を決定）
- 5) ネイル配置   … UIのみ保持（補強はモック表示）
- 6) 結果         … 無補強Fsと“補強（モック）Fs”を表示
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

# ----------------- 共通設定 -----------------
st.set_page_config(page_title="Stabi｜安定板２（I：multi-UI／補強モック）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"], index=0)
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

st.title("Stabi｜安定板２（I：multi-UI／補強はモック）")

# ----------------- 1) 地形・材料 -----------------
st.subheader("1) 地形・材料")
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
tanb = math.tan(math.radians(beta))

# 表示レンジ（未設定なら自動）
x_top  = H / max(tanb, 1e-6)
x_left  = float(st.session_state.get("x_left",  -0.1 * x_top))
x_right = float(st.session_state.get("x_right",  1.1 * x_top))
y_min   = float(st.session_state.get("y_min",   -1.2 * H))
y_max   = float(st.session_state.get("y_max",    1.2 * H))

if st.button("条件を保存", type="primary"):
    st.session_state.update({
        "H": H, "beta_deg": beta, "gamma": gamma, "c": c, "phi": phi,
        "x_left": x_left, "x_right": x_right, "y_min": y_min, "y_max": y_max
    })
    st.success("保存しました。")

def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * np.asarray(X, float)

# ----------------- 2) 地層（表示のみ） -----------------
st.subheader("2) 地層（表示のみ）")
layers = st.session_state.get("layers", [])
with st.expander("層を追加（水平層想定・計算には未使用）", expanded=False):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        z_top = st.number_input("層上端 z_top [m]", -1000.0, 1000.0, 8.0, 0.1, key="ly_zt")
        z_bot = st.number_input("層下端 z_bot [m]", -1000.0, 1000.0, 4.0, 0.1, key="ly_zb")
    with cc2:
        name  = st.text_input("層名", "Layer", key="ly_name")
        color = st.color_picker("表示色", "#f7e4b1", key="ly_col")
    with cc3:
        if st.button("＋この層を追加", key="ly_add"):
            L = {"name": name, "z_top": z_top, "z_bot": z_bot, "color": color}
            st.session_state["layers"] = layers = list(layers) + [L]
            st.success("追加しました。")
if layers:
    if st.button("全層を削除", key="ly_del"):
        st.session_state["layers"] = layers = []
        st.success("削除しました。")

# ----------------- 3) 水位（表示のみ） -----------------
st.subheader("3) 水位（表示のみ）")
with st.expander("水位を設定（水平 or 2点）", expanded=False):
    wtab = st.radio("形式", ["なし", "水平 y=const", "2点 (x1,y1)-(x2,y2)"], index=0, horizontal=True)
    if wtab == "水平 y=const":
        wy = st.number_input("水位高さ y [m]", -1000.0, 1000.0, float(st.session_state.get("water_y", 5.0)), 0.1)
        if st.button("保存（水位：水平）"):
            st.session_state["water_y"] = wy
            for k in ("xw1","yw1","xw2","yw2"):
                st.session_state.pop(k, None)
            st.success("保存しました。")
    elif wtab == "2点 (x1,y1)-(x2,y2)":
        xw1 = st.number_input("x1 [m]", -1000.0, 1000.0, float(st.session_state.get("xw1", 0.0)), 0.1)
        yw1 = st.number_input("y1 [m]", -1000.0, 1000.0, float(st.session_state.get("yw1", H*0.5)), 0.1)
        xw2 = st.number_input("x2 [m]", -1000.0, 1000.0, float(st.session_state.get("xw2", x_right)), 0.1)
        yw2 = st.number_input("y2 [m]", -1000.0, 1000.0, float(st.session_state.get("yw2", H*0.2)), 0.1)
        if st.button("保存（水位：2点）"):
            st.session_state.update({"xw1":xw1,"yw1":yw1,"xw2":xw2,"yw2":yw2})
            st.session_state.pop("water_y", None)
            st.success("保存しました。")
    else:
        if st.button("水位を消す"):
            for k in ("water_y","xw1","yw1","xw2","yw2"):
                st.session_state.pop(k, None)
            st.success("削除しました。")

# ----------------- 横断図プレビュー（地表＋層＋水位） -----------------
st.subheader("横断図プレビュー（計算には未反映）")
Xv = np.linspace(x_left, x_right, 500)
Yv = ground_y_at(Xv)
figv, axv = plt.subplots(figsize=(9, 5))
axv.plot(Xv, Yv, lw=2, label="Ground")

if isinstance(layers, list):
    for L in layers:
        try:
            zt = float(L["z_top"]); zb = float(L["z_bot"])
            color = L.get("color", "#e8e8e8")
            axv.fill_between(Xv, np.minimum(Yv, zt), np.minimum(Yv, zb),
                             color=color, alpha=0.35, step="mid", label=L.get("name","layer"))
        except Exception:
            pass

if "water_y" in st.session_state:
    wy = float(st.session_state["water_y"])
    axv.plot([x_left, x_right], [wy, wy], linestyle="--", label="Water")
if {"xw1","yw1","xw2","yw2"} <= set(st.session_state.keys()):
    xw1,yw1 = float(st.session_state["xw1"]), float(st.session_state["yw1"])
    xw2,yw2 = float(st.session_state["xw2"]), float(st.session_state["yw2"])
    axv.plot([xw1,xw2],[yw1,yw2], linestyle="--", label="Water")

axv.set_aspect('equal','box'); axv.set_xlim(x_left,x_right); axv.set_ylim(y_min,y_max)
axv.grid(True, alpha=0.35)
if show_legend: axv.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(figv)

# ----------------- 4) 円弧探索（未補強） -----------------
st.subheader("4) 円弧探索（未補強）")
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

    best = None
    centers = []

    def eval_grid(nx, ny, nR, xl, xh, yl, yh, Rl, Rh, ns):
        nonlocal best, centers
        xs = np.linspace(xl, xh, nx)
        ys = np.linspace(yl, yh, ny)
        Rs = np.linspace(Rl, Rh, nR)
        for xc in xs:
            for yc in ys:
                centers.append((xc, yc))
                for R in Rs:
                    slip = CircleSlip(xc=xc, yc=yc, R=R)
                    sls = generate_slices_on_arc(ground_y_at, slip, ns,
                                                 x_min=x_left, x_max=x_right, soil_gamma=soil.gamma)
                    if not sls:
                        continue
                    Fs = bishop_fs_unreinforced(sls, soil)
                    if not (Fs > 0 and np.isfinite(Fs)):
                        continue
                    if (best is None) or (Fs < best[0]):
                        best = (Fs, slip, sls)

    eval_grid(coarse_n[0], coarse_n[1], max(10, int((R_max-R_min)/H*8)),
              xc_min, xc_max, yc_min, yc_max, R_min, R_max, ns=36)

    if best is not None:
        _, s0, _ = best
        dx, dy, dR = 0.25*x_top, 0.35*H, 0.35*H
        eval_grid(quick_n[0], quick_n[1], max(12, int((R_max-R_min)/H*10)),
                  s0.xc-dx, s0.xc+dx, s0.yc-dy, s0.yc+dy,
                  max(R_min, s0.R-dR), min(R_max, s0.R+dR), ns=40)

    if best is not None:
        _, s0, _ = best
        dx, dy, dR = 0.12*x_top, 0.20*H, 0.22*H
        eval_grid(quick_n[0]+4, quick_n[1]+4, refine_nR,
                  s0.xc-dx, s0.xc+dx, s0.yc-dy, s0.yc+dy,
                  max(R_min, s0.R-dR), min(R_max, s0.R+dR), ns=44)
    return best, centers

best, centers_seen = search_best_circle()
if best is None:
    Fs_un = float("nan"); slip_best = None; slices_best = []
else:
    Fs_un, slip_best, slices_best = best

# 保存（補強ステップが読めるように）
if slip_best is not None:
    st.session_state["selected_slip"] = {"xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)}
if slices_best:
    st.session_state["slices_best"] = slices_best

# 可視化
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.linspace(x_left, x_right, 400), ground_y_at(np.linspace(x_left, x_right, 400)),
        label="Ground", lw=2)
if slip_best is not None:
    th = np.linspace(0, 2*math.pi, 400)
    Xc, Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, "--", label="Slip circle")
if audit and centers_seen:
    xs = [p[0] for p in centers_seen]; ys = [p[1] for p in centers_seen]
    ax.scatter(xs, ys, s=8, alpha=0.25, label="Centers")
ax.set_aspect('equal','box'); ax.set_xlim(x_left,x_right); ax.set_ylim(y_min,y_max)
ax.grid(True, alpha=0.35)
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

# ----------------- 5) ネイル配置（UIのみ／保存） -----------------
st.subheader("5) ソイルネイル配置（UIのみ：補強はモック）")
if "nails" not in st.session_state:
    st.session_state["nails"] = []
nails = st.session_state["nails"]

with st.expander("ネイルを追加（線分）"):
    n1, n2, n3 = st.columns(3)
    with n1:
        x1 = st.number_input("x1 [m]", -1000.0, 1000.0, 2.0, 0.1, key="nx1")
        y1 = st.number_input("y1 [m]", -1000.0, 1000.0, 6.0, 0.1, key="ny1")
        spacing = st.number_input("spacing [m]", 0.05, 10.0, 1.5, 0.05, key="nsp")
    with n2:
        x2 = st.number_input("x2 [m]", -1000.0, 1000.0, 8.0, 0.1, key="nx2")
        y2 = st.number_input("y2 [m]", -1000.0, 1000.0, 8.0, 0.1, key="ny2")
        T_yield = st.number_input("T_yield [kN/本]", 10.0, 5000.0, 200.0, 10.0, key="nyld")
    with n3:
        bond = st.number_input("bond_strength [kN/m]", 1.0, 1000.0, 80.0, 1.0, key="nbond")
        emb  = st.number_input("有効定着長(片側) [m]", 0.05, 10.0, 0.5, 0.05, key="nemb")
        if st.button("＋このネイルを追加", type="primary"):
            nails.append({
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "spacing":spacing,"T_yield":T_yield,
                "bond_strength":bond,"embed_each":emb
            })
            st.success("追加しました。")

if nails:
    idx = st.selectbox("削除対象（インデックス）", list(range(len(nails))), 0)
    d1, d2 = st.columns(2)
    with d1:
        if st.button("選択ネイルを削除"):
            nails.pop(idx); st.success("削除しました。")
    with d2:
        if st.button("全削除"):
            nails.clear(); st.success("全削除しました。")

# ネイルを図に重ねて確認
if slip_best is not None:
    fign, axn = plt.subplots(figsize=(9, 5))
    Xg = np.linspace(x_left, x_right, 400)
    axn.plot(Xg, ground_y_at(Xg), lw=2, label="Ground")
    th = np.linspace(0, 2*math.pi, 400)
    axn.plot(slip_best.xc + slip_best.R*np.cos(th),
             slip_best.yc + slip_best.R*np.sin(th), "--", label="Slip circle")
    for i, nl in enumerate(nails):
        axn.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]], lw=2,
                 label=(f"Nail {i}" if show_legend else None))
    axn.set_aspect('equal','box'); axn.set_xlim(x_left,x_right); axn.set_ylim(y_min,y_max)
    axn.grid(True, alpha=0.35)
    if show_legend: axn.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(fign)

# ----------------- 6) 結果（補強：モック） -----------------
st.subheader("6) 結果")
cA, cB = st.columns(2)
with cA: st.metric("未補強 Fs", f"{Fs_un:.3f}" if best else "—")
with cB: st.metric("補強（モック）Fs", f"{Fs_un:.3f}" if best else "—")
st.caption("※ この I 版では補強後Fsはモック（未補強と同値）。本実装は後段ページで差し替え。")