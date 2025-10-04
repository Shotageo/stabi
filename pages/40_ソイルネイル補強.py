# pages/40_ソイルネイル補強.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stabi_lem import (
    Soil, CircleSlip, Nail,
    generate_slices_on_arc,
    bishop_fs_unreinforced, bishop_fs_with_nails,
)

# ---------- Plot style（Theme/Tight layout/Legend切替） ----------
st.set_page_config(page_title="Stabi｜安定板２｜ソイルネイル補強", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", value=True)
show_legend = st.sidebar.checkbox("Show legend", value=True)
plt.style.use("dark_background" if theme == "dark_background" else "default")
# ---------------------------------------------------------------

st.title("Stabi｜安定板２：ソイルネイル補強（“選択円弧”に適用）")

# ==== 既存ステップの状態を回収 ====
H = float(st.session_state.get("H", 12.0))
beta = float(st.session_state.get("beta_deg", 35.0))
gamma = float(st.session_state.get("gamma", 18.0))
c = float(st.session_state.get("c", 10.0))
phi = float(st.session_state.get("phi", 30.0))

soil = Soil(gamma=gamma, c=c, phi=phi)

beta_rad = math.radians(beta)
tanb = math.tan(beta_rad)
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

x_top = H / max(tanb, 1e-6)
x_left  = float(st.session_state.get("x_left", -0.1 * x_top))
x_right = float(st.session_state.get("x_right",  1.1 * x_top))
y_min   = float(st.session_state.get("y_min", -1.2 * H))
y_max   = float(st.session_state.get("y_max",  1.2 * H))

# ==== 選択円弧を取得（なければ控えめなUI） ====
sel = st.session_state.get("selected_slip", None)
if isinstance(sel, CircleSlip):
    slip = sel
elif isinstance(sel, dict) and {"xc","yc","R"} <= sel.keys():
    slip = CircleSlip(xc=float(sel["xc"]), yc=float(sel["yc"]), R=float(sel["R"]))
else:
    st.info("無補強ページで“選択円弧（selected_slip）”が見つかりません。ここで一時指定できます。")
    c1, c2, c3 = st.columns(3)
    with c1: xc = st.number_input("円弧中心 xc [m]", -200.0, 200.0, x_top*0.25, 0.5)
    with c2: yc = st.number_input("円弧中心 yc [m]", -200.0, 200.0, -H*0.8, 0.5)
    with c3: R  = st.number_input("半径 R [m]", 0.5, 300.0, H*1.4, 0.5)
    slip = CircleSlip(xc=xc, yc=yc, R=R)

# ==== ソイルネイル（線分） ====
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

# ==== Fs 計算（選択円弧に対して） ====
slices = generate_slices_on_arc(ground_y_at, slip, n_slices=40,
                                x_min=x_left, x_max=x_right, soil_gamma=soil.gamma)
Fs_un = bishop_fs_unreinforced(slices, soil) if slices else float("nan")
Fs_re = bishop_fs_with_nails(slices, soil, slip, st.session_state.nails) if slices else float("nan")

# ==== 表示 ====
st.subheader("結果（選択円弧）")
cR1, cR2 = st.columns(2)
with cR1: st.metric("未補強 Fs", f"{Fs_un:.3f}" if slices else "—")
with cR2: st.metric("補強後 Fs", f"{Fs_re:.3f}" if slices else "—")

fig, ax = plt.subplots(figsize=(9, 6))
Xg = np.linspace(x_left, x_right, 400)
Yg = ground_y_at(Xg)
ax.plot(Xg, Yg, label="Ground", linewidth=2)

th = np.linspace(0, 2*math.pi, 400)
ax.plot(slip.xc + slip.R*np.cos(th), slip.yc + slip.R*np.sin(th), "--", label="Selected slip circle")

for i, nl in enumerate(st.session_state.nails):
    ax.plot([nl.x1, nl.x2], [nl.y1, nl.y2], linewidth=2, label=f"Nail {i}" if show_legend else None)

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.4)
ax.set_title(f"Fs_un={Fs_un:.3f} / Fs_re={Fs_re:.3f}" if slices else "（この円弧では土塊が形成されません）")
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

st.caption("※ 無補強ページで保存された “selected_slip” にネイル寄与を上乗せするページ。")
