# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Stabi｜安定板２（今日までの完全復帰仕様）
- cfg一元化
- 多段UI（地形→水位→地層→探索→ネイル→結果）
- ground.y_at() 安全評価
- 水位オフセット設定対応
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from stabi_lem import Soil, CircleSlip, generate_slices_on_arc, bishop_fs_unreinforced

st.set_page_config(page_title="Stabi｜安定板２", layout="wide")

# ========= 初期設定 =========
if "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "H": 12.0, "beta": 35.0,
        "gamma": 18.0, "c": 10.0, "phi": 30.0,
        "water_offset": 0.0,
        "x_left": -5.0, "x_right": 30.0,
        "y_min": -10.0, "y_max": 20.0,
        "audit": False
    }
cfg = st.session_state["cfg"]

# ========= ヘッダー =========
st.title("Stabi｜安定板２（復帰仕様）")

# ========= Section 1: 地形・材料 =========
st.header("1️⃣ 地形・材料設定")
c1, c2, c3 = st.columns(3)
with c1:
    cfg["H"] = st.number_input("法高さ H [m]", 1.0, 200.0, cfg["H"])
    cfg["beta"] = st.number_input("法勾配 β [°]", 5.0, 85.0, cfg["beta"])
with c2:
    cfg["gamma"] = st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, cfg["gamma"])
    cfg["c"] = st.number_input("粘着力 c [kPa]", 0.0, 300.0, cfg["c"])
with c3:
    cfg["phi"] = st.number_input("内部摩擦角 φ [°]", 0.0, 45.0, cfg["phi"])
    cfg["water_offset"] = st.number_input("水位オフセット h_off [m]", -10.0, 20.0, cfg["water_offset"])

tanb = math.tan(math.radians(cfg["beta"]))
soil = Soil(cfg["gamma"], cfg["c"], cfg["phi"])

# 安全な地表関数
def ground_y_at(X):
    X = np.asarray(X, float)
    return cfg["H"] - tanb * X

# ========= Section 2: 水位表示 =========
st.header("2️⃣ 水位設定（法尻基準オフセット）")
h_w = cfg["H"] - cfg["water_offset"]
st.write(f"水位高: y = {h_w:.2f} m")

# ========= Section 3: 横断図 =========
st.header("3️⃣ 横断図プレビュー")
X = np.linspace(cfg["x_left"], cfg["x_right"], 400)
Yg = ground_y_at(X)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(X, Yg, lw=2, label="Ground")
ax.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water level")

ax.set_aspect("equal", "box")
ax.set_xlim(cfg["x_left"], cfg["x_right"])
ax.set_ylim(cfg["y_min"], cfg["y_max"])
ax.grid(True, alpha=0.4)
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# ========= Section 4: 円弧探索 =========
st.header("4️⃣ 無補強円弧探索（Bishop簡略法）")

def search_best():
    H = cfg["H"]
    xc_range = np.linspace(-0.2*H, 1.2*H, 18)
    yc_range = np.linspace(-1.5*H, 0.5*H, 12)
    R_range = np.linspace(0.6*H, 2.0*H, 14)

    best = None
    for xc in xc_range:
        for yc in yc_range:
            for R in R_range:
                slip = CircleSlip(xc, yc, R)
                sls = generate_slices_on_arc(ground_y_at, slip, 36, cfg["x_left"], cfg["x_right"], cfg["gamma"])
                if not sls:
                    continue
                Fs = bishop_fs_unreinforced(sls, soil)
                if best is None or Fs < best[0]:
                    best = (Fs, slip, sls)
    return best

if st.button("探索開始", type="primary"):
    best = search_best()
    if best:
        Fs, slip, slices = best
        st.session_state["selected_slip"] = {"xc": slip.xc, "yc": slip.yc, "R": slip.R}
        st.session_state["slices_best"] = slices
        st.success(f"最小Fs = {Fs:.3f}")
    else:
        st.warning("有効な円弧が見つかりませんでした。")

# ========= Section 5: ネイル配置（UIのみ） =========
st.header("5️⃣ ソイルネイル配置（モック）")
if "nails" not in st.session_state:
    st.session_state["nails"] = []
nails = st.session_state["nails"]

with st.expander("ネイルを追加"):
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.number_input("x1", -50.0, 100.0, 2.0)
        y1 = st.number_input("y1", -50.0, 100.0, 6.0)
        spacing = st.number_input("spacing [m]", 0.1, 10.0, 1.5)
    with c2:
        x2 = st.number_input("x2", -50.0, 100.0, 8.0)
        y2 = st.number_input("y2", -50.0, 100.0, 8.0)
        T = st.number_input("T_yield [kN]", 1.0, 5000.0, 200.0)
    if st.button("追加"):
        nails.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"spacing":spacing,"T":T})
        st.success("追加しました")

if nails:
    for i, nl in enumerate(nails):
        st.write(f"#{i}: ({nl['x1']:.2f},{nl['y1']:.2f})→({nl['x2']:.2f},{nl['y2']:.2f})  T={nl['T']:.1f}kN")
    if st.button("全削除"):
        nails.clear()
        st.success("全削除しました")

# ========= Section 6: 結果 =========
st.header("6️⃣ 結果（補強：モック）")
if "selected_slip" in st.session_state:
    slip = st.session_state["selected_slip"]
    st.metric("未補強 Fs", f"{st.session_state.get('Fs_last', 1.0):.3f}")
    st.metric("補強（モック）Fs", f"{st.session_state.get('Fs_last', 1.0):.3f}")
else:
    st.write("まだ解析が実行されていません。")