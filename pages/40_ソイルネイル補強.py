# pages/40_ソイルネイル補強.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# —— 既存モジュール（安定板２の核）から利用 —— 
# 下の4つが stabi_lem.py に無い場合は、先に実装を入れてね（後述の注記を参照）
from stabi_lem import (
    CircleSlip, Nail,
    generate_slices_on_arc,
    bishop_fs_unreinforced, bishop_fs_with_nails,
)

# ================= Plot style（Theme/Tight layout/Legend切替：あなたの既定） =================
st.set_page_config(page_title="Stabi｜安定板２｜ソイルネイル補強", layout="wide")
st.sidebar.header("Plot style")
_theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
_tight = st.sidebar.checkbox("Tight layout", value=True)
_show_legend = st.sidebar.checkbox("Show legend", value=True)
plt.style.use("dark_background" if _theme == "dark_background" else "default")
# ============================================================================================

st.title("Stabi｜安定板２：ソイルネイル補強（既存ステップの“選択円弧”に適用）")

# --------------------------------------------------------------------------------------------
# 1) 既存ステップからの「選択円弧」「地形/材料」「表示範囲」をできるだけ拾う
#    - キー名はプロジェクト差が出やすいので、複数候補を順に探す
#    - 見つからない場合は控えめな UI を出して手動指定できるようにする
# --------------------------------------------------------------------------------------------
def _to_slip(obj):
    """dictやオブジェクトから CircleSlip へ寄せる小さな変換器"""
    if obj is None:
        return None
    if isinstance(obj, CircleSlip):
        return obj
    for k in ("xc", "x_c", "x0"):
        if isinstance(obj, dict) and k in obj:
            try:
                return CircleSlip(xc=float(obj.get("xc", obj.get("x_c", obj.get("x0")))),
                                  yc=float(obj.get("yc", obj.get("y_c", obj.get("y0")))),
                                  R=float(obj.get("R", obj.get("r", obj.get("radius")))))
            except Exception:
                return None
    # dataclass風の属性
    try:
        return CircleSlip(xc=float(getattr(obj, "xc")), yc=float(getattr(obj, "yc")), R=float(getattr(obj, "R")))
    except Exception:
        return None

# 候補キー：最小Fs円弧・ユーザ選択円弧など
_SLIP_KEYS = [
    "selected_slip", "best_slip", "min_fs_slip", "slip_selected",
    "lem.selected_slip", "lem.best_slip", "result.min_circle",
]

_selected_slip = None
for k in _SLIP_KEYS:
    if k in st.session_state:
        _selected_slip = _to_slip(st.session_state[k])
        if _selected_slip:
            break

# 地形パラメータの回収（なければ最小UI）
H = st.session_state.get("H", None)
beta = st.session_state.get("beta_deg", None) or st.session_state.get("beta", None)
if (H is None) or (beta is None):
    st.info("既存ページから法面条件を拾えなかったので、ここだけ最小入力を出します。")
    cA, cB = st.columns(2)
    with cA:
        H  = st.number_input("法高さ H [m]", 1.0, 100.0, 12.0, 0.5)
    with cB:
        beta = st.number_input("法勾配角 β [deg]", 5.0, 80.0, 35.0, 0.5)
    # 保存しておく（他ページでも使えるように）
    st.session_state["H"] = H
    st.session_state["beta_deg"] = beta

# 表示範囲の回収/設定
x_left  = st.session_state.get("x_left", None)
x_right = st.session_state.get("x_right", None)
beta_rad = math.radians(float(beta))
tanb = math.tan(beta_rad)
x_top = H / max(tanb, 1e-6)
if x_left is None:  x_left  = -0.1 * x_top
if x_right is None: x_right =  1.1 * x_top
y_min = st.session_state.get("y_min", -1.2 * H)
y_max = st.session_state.get("y_max",  1.2 * H)

# 地表関数（“安定板２”の安全仕様：配列対応）
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

# 既存が円弧未選択なら、控えめに円弧入力UI
if _selected_slip is None:
    st.warning("既存ステップから“選択円弧”が見つかりませんでした。ここで一時的に円弧を指定できます。")
    c1, c2, c3 = st.columns(3)
    with c1:
        xc = st.number_input("円弧中心 xc [m]", -200.0, 200.0, x_top * 0.25, 0.5)
    with c2:
        yc = st.number_input("円弧中心 yc [m]", -200.0, 200.0, -H * 0.8, 0.5)
    with c3:
        R  = st.number_input("半径 R [m]", 0.5, 300.0, H * 1.4, 0.5)
    _selected_slip = CircleSlip(xc=xc, yc=yc, R=R)

# --------------------------------------------------------------------------------------------
# 2) ネイル入力（線分）と保存
# --------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------
# 3) 無補強Fs（選択円弧に対して）→ 補強後Fs を算出
# --------------------------------------------------------------------------------------------
# 既存の材料は他ページにある想定：ここでは最小限、γ/c/φ を拾える場合だけ拾う
gamma = st.session_state.get("gamma", 18.0)
c_val = st.session_state.get("c_kpa", None) or st.session_state.get("c", 10.0)
phi_deg = st.session_state.get("phi_deg", None) or st.session_state.get("phi", 30.0)

class _Soil:  # 依存を増やさないための局所構造（stabi_lem.Soil不要）
    def __init__(self, gamma, c, phi):
        self.gamma = gamma; self.c = c; self.phi = phi
soil_like = _Soil(gamma, c_val, phi_deg)

# 選択円弧でスライス化（既存の表示範囲を使用）
slices = generate_slices_on_arc(ground_y_at, _selected_slip, n_slices=40,
                                x_min=x_left, x_max=x_right, soil_gamma=soil_like.gamma)

Fs_un = bishop_fs_unreinforced(slices, soil_like) if slices else float("nan")
Fs_re = bishop_fs_with_nails(slices, soil_like, _selected_slip, st.session_state.nails) if slices else float("nan")

# --------------------------------------------------------------------------------------------
# 4) 結果と図
# --------------------------------------------------------------------------------------------
st.subheader("結果（既存ページの“選択円弧”に対する評価）")
cR1, cR2 = st.columns(2)
with cR1:
    st.metric("未補強 Fs（この円弧）", f"{Fs_un:.3f}" if slices else "—")
with cR2:
    st.metric("補強後 Fs", f"{Fs_re:.3f}" if slices else "—")

fig, ax = plt.subplots(figsize=(9, 6))
# 地表
Xg = np.linspace(x_left, x_right, 400)
Yg = ground_y_at(Xg)
ax.plot(Xg, Yg, label="Ground", linewidth=2)

# 選択円弧（既存ページと同じく全周のまま）
th = np.linspace(0, 2*math.pi, 400)
Xc = _selected_slip.xc + _selected_slip.R*np.cos(th)
Yc = _selected_slip.yc + _selected_slip.R*np.sin(th)
ax.plot(Xc, Yc, linestyle="--", label="Selected slip circle")

# ネイル
for i, nl in enumerate(st.session_state.nails):
    ax.plot([nl.x1, nl.x2], [nl.y1, nl.y2], linewidth=2, label=f"Nail {i}" if _show_legend else None)

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.4)
ax.set_title(f"Fs_un={Fs_un:.3f} / Fs_re={Fs_re:.3f}" if slices else "（この円弧では土塊が形成されません）")
if _show_legend:
    ax.legend(loc="best")
if _tight:
    plt.tight_layout()
st.pyplot(fig)

st.caption("※ このページは“既存ステップの円弧選択”を前提に、ネイル寄与だけを上乗せする専用ページです。")
