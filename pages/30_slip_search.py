# pages/30_slip_search.py
# -*- coding: utf-8 -*-
import math, numpy as np, matplotlib.pyplot as plt, streamlit as st
from stabi_lem import Soil, CircleSlip, generate_slices_on_arc, bishop_fs_unreinforced, circle_xy_from_theta

st.set_page_config(page_title="安定板２｜3) 円弧探索（無補強）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("3) 円弧探索（無補強）")

# —— ここから下は**あなたの安定板２の探索ロジック**をそのまま残してください ——
# 下の地形・表示レンジ取得と「横断図プレビュー」は“描くだけ”。式や探索には一切手を触れません。

H    = float(st.session_state.get("H", 12.0))
beta = float(st.session_state.get("beta_deg", 35.0))
gamma= float(st.session_state.get("gamma", 18.0))
c    = float(st.session_state.get("c", 10.0))
phi  = float(st.session_state.get("phi", 30.0))
x_left  = float(st.session_state.get("x_left", -1.0))
x_right = float(st.session_state.get("x_right",  1.0))
y_min   = float(st.session_state.get("y_min",  -10.0))
y_max   = float(st.session_state.get("y_max",   10.0))

tanb = math.tan(math.radians(beta))
def ground_y_at(X): return H - tanb * np.asarray(X, float)

# --- 横断図プレビュー（読み取り専用） ---
with st.expander("横断図プレビュー（確認用）", expanded=False):
    Xv = np.linspace(x_left, x_right, 400); Yv = ground_y_at(Xv)
    figv, axv = plt.subplots(figsize=(9,3))
    axv.plot(Xv, Yv, lw=2, label="Ground")
    # 層
    layers = st.session_state.get("layers", None)
    if isinstance(layers, list):
        for L in layers:
            try:
                zt = float(L["z_top"]); zb = float(L["z_bot"])
                color = L.get("color", "#e8e8e8")
                axv.fill_between(Xv, np.minimum(Yv, zt), np.minimum(Yv, zb), color=color, alpha=0.35, step="mid")
            except Exception:
                pass
    # 水位
    if "water_y" in st.session_state:
        axv.plot([x_left, x_right], [st.session_state["water_y"]]*2, linestyle="--", label="Water")
    if {"xw1","yw1","xw2","yw2"} <= set(st.session_state.keys()):
        xw1,yw1 = float(st.session_state["xw1"]), float(st.session_state["yw1"])
        xw2,yw2 = float(st.session_state["xw2"]), float(st.session_state["yw2"])
        axv.plot([xw1,xw2],[yw1,yw2], linestyle="--", label="Water")
    axv.set_aspect('equal','box'); axv.set_xlim(x_left,x_right); axv.set_ylim(y_min,y_max); axv.grid(True,alpha=0.3)
    st.pyplot(figv)

# —— 以降：あなたの既存の探索コード（最小Fs、slip_best、slices_best を決定）をそのまま —— 
# （ここでは簡易版の骨格だけ残しておきます。あなたの実装で上書きしてください）

soil = Soil(gamma=gamma, c=c, phi=phi)

# ・・・（既存の探索器）・・・

# 既存の変数名：Fs_un, slip_best, slices_best を想定
# ※ここはあなたの実装のまま。保存だけは念のため入れておきます。
try:
    Fs_un
    slip_best
    slices_best
except NameError:
    # もしあなたの実装ブロックに上書きする前の暫定ダミー（削除可）
    Fs_un = float("nan"); slip_best = None; slices_best = []

if slip_best is not None:
    st.session_state["selected_slip"] = {"xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)}
if slices_best:
    st.session_state["slices_best"] = slices_best

# 可視化（あなたの既存表示でOK）
if slip_best is not None:
    fig, ax = plt.subplots(figsize=(9, 6))
    Xg = np.linspace(x_left, x_right, 400); ax.plot(Xg, ground_y_at(Xg), lw=2, label="Ground")
    th = np.linspace(0, 2*math.pi, 400); Xc,Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, "--", label="Selected slip circle")
    ax.set_aspect('equal','box'); ax.set_xlim(x_left,x_right); ax.set_ylim(y_min,y_max)
    ax.grid(True, alpha=0.35); ax.legend(loc="best")
    st.pyplot(fig)

st.page_link("pages/40_soil_nail.py", label="→ 4) ソイルネイル補強へ", icon="🪛")
