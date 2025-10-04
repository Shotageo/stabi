# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Stabi｜安定板２（完全復帰仕様）
- cfg一元化（st.session_state["cfg"]）
- 多段UI：地形→地層→水位→横断図→円弧探索→ネイル→結果
- 水位は【法尻(= y=0)基準のオフセット h_off】で水平線を表示
- Bishop簡略法で未補強Fs（補強はモック表示）
- ground_y_at は array-safe
- Audit は既定OFF、Quality は Normal
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from stabi_lem import Soil, CircleSlip, generate_slices_on_arc, bishop_fs_unreinforced

# ---------------- 基本設定・テーマ ----------------
st.set_page_config(page_title="Stabi｜安定板２（復帰）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"], index=0)
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

# ---------------- cfg 一元化 ----------------
if "cfg" not in st.session_state:
    st.session_state["cfg"] = {
        "H": 12.0, "beta": 35.0,
        "gamma": 18.0, "c": 10.0, "phi": 30.0,
        "x_left": -5.0, "x_right": 30.0,
        "y_min": -10.0, "y_max": 20.0,
        # ※ 水位は“法尻(=y=0)”基準オフセット[m]
        "water_offset": 5.0,
        "layers": [],
        "quality": "Normal",
        "audit": False,
        "budget_coarse": 0.8,
        "budget_quick": 1.2,
    }
cfg = st.session_state["cfg"]

st.title("🧩 Stabi｜安定板２（完全復帰仕様）")

# ---------------- 1) 地形・材料 ----------------
st.header("1️⃣ 地形・材料")
c1, c2, c3 = st.columns(3)
with c1:
    cfg["H"]    = st.number_input("法高さ H [m]", 1.0, 200.0, float(cfg["H"]), 0.5)
    cfg["beta"] = st.number_input("法勾配 β [°]", 5.0, 85.0,  float(cfg["beta"]), 0.5)
with c2:
    cfg["gamma"]= st.number_input("単位体積重量 γ [kN/m³]", 10.0, 30.0, float(cfg["gamma"]), 0.5)
    cfg["c"]    = st.number_input("粘着力 c [kPa]", 0.0, 300.0, float(cfg["c"]), 1.0)
with c3:
    cfg["phi"]  = st.number_input("内部摩擦角 φ [°]", 0.0, 45.0, float(cfg["phi"]), 0.5)
    # ← ここは“法尻基準のオフセット”。y=0 を基準に、上向き正の水平水位。
    cfg["water_offset"] = st.number_input("水位オフセット h_off（法尻基準）[m]",
                                          -10.0, 50.0, float(cfg["water_offset"]), 0.1)

tanb = math.tan(math.radians(cfg["beta"]))
soil = Soil(cfg["gamma"], cfg["c"], cfg["phi"])

def ground_y_at(X):
    X = np.asarray(X, float)
    return cfg["H"] - tanb * X

# ---------------- 2) 地層（任意） ----------------
st.header("2️⃣ 地層（任意・表示のみ）")
layers = cfg.get("layers", [])
with st.expander("層を追加（水平帯）", expanded=False):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        z_top = st.number_input("層上端 z_top [m]", -100.0, 100.0, 8.0)
        z_bot = st.number_input("層下端 z_bot [m]", -100.0, 100.0, 4.0)
    with cc2:
        name  = st.text_input("層名", "Layer")
        color = st.color_picker("色", "#e7d7a8")
    with cc3:
        if st.button("＋追加"):
            layers.append({"name": name, "z_top": float(z_top), "z_bot": float(z_bot), "color": color})
            cfg["layers"] = layers
            st.success("層を追加しました。")
if layers:
    st.write(f"登録済み層数: {len(layers)}")
    if st.button("全削除（地層）"):
        layers.clear(); cfg["layers"] = layers
        st.success("地層を削除しました。")

# ---------------- 3) 水位（法尻オフセット） ----------------
st.header("3️⃣ 水平水位（法尻= y=0 基準オフセット）")
h_w = float(cfg["water_offset"])  # ← toe(y=0)からのオフセットそのもの
st.write(f"水平水位: y = {h_w:.2f} m  （法尻基準）")

# ---------------- 4) 横断図プレビュー ----------------
st.header("4️⃣ 横断図プレビュー")
X = np.linspace(float(cfg["x_left"]), float(cfg["x_right"]), 500)
Yg = ground_y_at(X)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(X, Yg, lw=2, label="Ground")
# 層
for L in layers:
    try:
        zt = float(L["z_top"]); zb = float(L["z_bot"])
        ax.fill_between(X, np.minimum(Yg, zt), np.minimum(Yg, zb),
                        color=L.get("color", "#dddddd"), alpha=0.3, step="mid", label=L.get("name","layer"))
    except Exception:
        pass
# 水位（法尻基準の水平線）
ax.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water level")

ax.set_aspect("equal","box")
ax.set_xlim(float(cfg["x_left"]), float(cfg["x_right"]))
ax.set_ylim(float(cfg["y_min"]),  float(cfg["y_max"]))
ax.grid(True, alpha=0.35)
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

# ---------------- 5) 無補強円弧探索 ----------------
st.header("5️⃣ 無補強円弧探索（Bishop簡略法）")
cfg["quality"] = st.selectbox("探索クオリティ", ["Fast","Normal","High"],
                              index=["Fast","Normal","High"].index(cfg["quality"]))
cfg["audit"]   = st.checkbox("Audit（センター表示）", value=False)

st.caption(f"⏱️ Budget: Coarse≈{cfg['budget_coarse']}s / Quick≈{cfg['budget_quick']}s（目安）")

def search_best():
    H = float(cfg["H"])
    # グリッドは安定板２の既定レンジ
    xc_range = np.linspace(-0.2*H, 1.2*H, 22 if cfg["quality"]!="Fast" else 16)
    yc_range = np.linspace(-2.0*H, 0.5*H, 14 if cfg["quality"]!="Fast" else 10)
    R_range  = np.linspace(0.6*H,  2.2*H, 16 if cfg["quality"]!="Fast" else 12)

    best = None
    centers = []
    for xc in xc_range:
        for yc in yc_range:
            centers.append((xc, yc))
            for R in R_range:
                slip = CircleSlip(float(xc), float(yc), float(R))
                sls = generate_slices_on_arc(ground_y_at, slip, 36,
                                             x_min=float(cfg["x_left"]), x_max=float(cfg["x_right"]),
                                             soil_gamma=float(cfg["gamma"]))
                if not sls: 
                    continue
                Fs = bishop_fs_unreinforced(sls, soil)
                if (best is None) or (Fs < best[0]):
                    best = (Fs, slip, sls)
    return best, centers

if st.button("探索開始", type="primary"):
    t0 = time.time()
    best, centers = search_best()
    t1 = time.time()

    if best:
        Fs, slip, slices = best
        st.session_state["selected_slip"] = {"xc": float(slip.xc), "yc": float(slip.yc), "R": float(slip.R)}
        st.session_state["slices_best"]   = slices
        st.session_state["Fs_last"]       = float(Fs)
        st.success(f"最小Fs = {Fs:.3f}  （{t1 - t0:.2f} s）")
    else:
        st.warning("有効な円弧が見つかりませんでした。")

    # 可視化（センターが枠端に当たる場合は“表示のみ”自動拡張）
    hx = float(cfg["x_left"]); hx2 = float(cfg["x_right"])
    hy = float(cfg["y_min"]);  hy2 = float(cfg["y_max"])
    if centers:
        xs, ys = zip(*centers)
        if min(xs) < hx:  hx  = min(xs) - 0.1*abs(hx2-hx)
        if max(xs) > hx2: hx2 = max(xs) + 0.1*abs(hx2-hx)
        if min(ys) < hy:  hy  = min(ys) - 0.1*abs(hy2-hy)
        if max(ys) > hy2: hy2 = max(ys) + 0.1*abs(hy2-hy)

    figR, axR = plt.subplots(figsize=(9, 6))
    Xp = np.linspace(hx, hx2, 600); Yp = ground_y_at(Xp)
    axR.plot(Xp, Yp, lw=2, label="Ground")
    if best:
        th = np.linspace(0, 2*np.pi, 400)
        axR.plot(slip.xc + slip.R*np.cos(th), slip.yc + slip.R*np.sin(th),
                 "--", lw=2, label="Slip circle")
    if cfg["audit"] and centers:
        axR.scatter(xs, ys, s=8, alpha=0.3, label="Centers")
    axR.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water")
    axR.set_aspect("equal","box")
    axR.set_xlim(hx, hx2); axR.set_ylim(hy, hy2)
    axR.grid(True, alpha=0.35)
    if show_legend: axR.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(figR)

# ---------------- 6) ソイルネイル配置（モック） ----------------
st.header("6️⃣ ソイルネイル配置（モック）")
if "nails" not in st.session_state:
    st.session_state["nails"] = []
nails = st.session_state["nails"]

with st.expander("ネイルを追加（線分）", expanded=False):
    n1, n2, n3 = st.columns(3)
    with n1:
        x1 = st.number_input("x1 [m]", -1000.0, 1000.0, 2.0, 0.1)
        y1 = st.number_input("y1 [m]", -1000.0, 1000.0, 6.0, 0.1)
        spacing = st.number_input("spacing [m]", 0.05, 10.0, 1.5, 0.05)
    with n2:
        x2 = st.number_input("x2 [m]", -1000.0, 1000.0, 8.0, 0.1)
        y2 = st.number_input("y2 [m]", -1000.0, 1000.0, 8.0, 0.1)
        Ty = st.number_input("T_yield [kN/本]", 1.0, 5000.0, 200.0, 1.0)
    with n3:
        bond = st.number_input("bond_strength [kN/m]", 1.0, 1000.0, 80.0, 1.0)
        emb  = st.number_input("有効定着長(片側) [m]", 0.05, 10.0, 0.5, 0.05)
        if st.button("＋追加", type="primary"):
            nails.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                          "spacing":spacing,"T_yield":Ty,
                          "bond_strength":bond,"embed_each":emb})
            st.success("追加しました。")

if nails:
    cdel1, cdel2 = st.columns(2)
    with cdel1:
        if st.button("全削除（ネイル）"):
            nails.clear(); st.success("ネイルを全削除しました。")

# ネイル重ね表示（現在の最小円弧がある場合）
sel = st.session_state.get("selected_slip", None)
if sel:
    figN, axN = plt.subplots(figsize=(9, 6))
    Xn = np.linspace(float(cfg["x_left"]), float(cfg["x_right"]), 500)
    axN.plot(Xn, ground_y_at(Xn), lw=2, label="Ground")
    th = np.linspace(0, 2*np.pi, 400)
    axN.plot(float(sel["xc"]) + float(sel["R"])*np.cos(th),
             float(sel["yc"]) + float(sel["R"])*np.sin(th),
             "--", lw=2, label="Slip circle")
    for i, nl in enumerate(nails):
        axN.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]],
                 lw=2, label=(f"Nail {i+1}" if show_legend else None))
    axN.axhline(y=h_w, color="deepskyblue", linestyle="--", label="Water")
    axN.set_aspect("equal","box")
    axN.set_xlim(float(cfg["x_left"]), float(cfg["x_right"]))
    axN.set_ylim(float(cfg["y_min"]),  float(cfg["y_max"]))
    axN.grid(True, alpha=0.35)
    if show_legend: axN.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(figN)

# ---------------- 7) 結果（補強はモック） ----------------
st.header("7️⃣ 結果（補強はモック）")
if "Fs_last" in st.session_state:
    Fs_un = float(st.session_state["Fs_last"])
    st.metric("未補強 Fs", f"{Fs_un:.3f}")
    st.metric("補強後 Fs（モック）", f"{Fs_un:.3f}")
    st.caption("※ この版では補強後Fsは未計算（モック表示）。II段階で本実装に差し替え。")
else:
    st.write("まだ解析が実行されていません。")