# pages/40_soil_nail.py
# -*- coding: utf-8 -*-
import math, numpy as np, matplotlib.pyplot as plt, streamlit as st

# 既存の未補強Fs（あなたの stabi_lem.py のまま）を使う
from stabi_lem import bishop_fs_unreinforced

st.set_page_config(page_title="安定板２｜4) ソイルネイル補強", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("4) ソイルネイル補強（安定板２の結果を“読むだけ”で上乗せ）")

ss = st.session_state
H    = float(ss.get("H", 12.0))
beta = float(ss.get("beta_deg", 35.0))
gamma= float(ss.get("gamma", 18.0))
c    = float(ss.get("c", 10.0))
phi  = float(ss.get("phi", 30.0))
x_left  = float(ss.get("x_left", -1.0))
x_right = float(ss.get("x_right",  1.0))
y_min   = float(ss.get("y_min",  -10.0))
y_max   = float(ss.get("y_max",   10.0))
tanb = math.tan(math.radians(beta))
def ground_y_at(X): return H - tanb * np.asarray(X, float)

sel = ss.get("selected_slip", None)
if not (isinstance(sel, dict) and {"xc","yc","R"} <= sel.keys()):
    st.error("『3) 円弧探索』で最小Fs円弧を確定してください（selected_slip 未設定）。")
    st.stop()
xc,yc,R = float(sel["xc"]), float(sel["yc"]), float(sel["R"])

slices_best = ss.get("slices_best", None)
if not slices_best:
    st.error("『3) 円弧探索』のスライス情報（slices_best）が見つかりません。同一分割での比較ができません。")
    st.stop()

# ネイル UI（ページ間共有）
if "nails" not in ss: ss.nails = []
with st.expander("ネイルを追加", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        x1 = st.number_input("x1 [m]", -200.0, 200.0, 2.0, 0.1, key="nx1")
        y1 = st.number_input("y1 [m]", -200.0, 200.0, 6.0, 0.1, key="ny1")
        spacing = st.number_input("spacing [m]", 0.05, 10.0, 1.5, 0.05, key="nsp")
    with c2:
        x2 = st.number_input("x2 [m]", -200.0, 200.0, 8.0, 0.1, key="nx2")
        y2 = st.number_input("y2 [m]", -200.0, 200.0, 8.0, 0.1, key="ny2")
        T_yield = st.number_input("T_yield [kN/本]", 10.0, 5000.0, 200.0, 10.0, key="nyld")
    with c3:
        bond = st.number_input("bond_strength [kN/m]", 1.0, 1000.0, 80.0, 1.0, key="nbnd")
        emb  = st.number_input("有効定着長(片側) [m]", 0.05, 10.0, 0.5, 0.05, key="nemb")
        if st.button("＋このネイルを追加", type="primary"):
            ss.nails.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"spacing":spacing,
                             "T_yield":T_yield,"bond_strength":bond,"embed_each":emb})

if ss.nails:
    idx = st.selectbox("削除対象（インデックス）", list(range(len(ss.nails))), 0)
    cdel1, cdel2 = st.columns(2)
    with cdel1:
        if st.button("選択ネイルを削除"): ss.nails.pop(idx)
    with cdel2:
        if st.button("全削除"): ss.nails.clear()

# ---- “上乗せ”の最小実装（既存式は一切変更しない） ----
def _line_circle_hits(x1,y1,x2,y2, xc,yc,R):
    dx,dy = (x2-x1),(y2-y1); fx,fy = (x1-xc),(y1-yc)
    A = dx*dx + dy*dy; B = 2*(fx*dx + fy*dy); C = fx*fx + fy*fy - R*R
    disc = B*B - 4*A*C
    if A==0 or disc<0: return []
    rt = math.sqrt(disc); out=[]
    for s in (-1,1):
        t = (-B + s*rt)/(2*A)
        if 0<=t<=1:
            x = x1 + t*dx; y = y1 + t*dy
            th = math.atan2(y-yc, x-xc)
            out.append((x,y,th))
    return out

def _nail_T_tan_hits(nails):
    hits=[]
    for nl in nails:
        pts = _line_circle_hits(nl["x1"],nl["y1"],nl["x2"],nl["y2"], xc,yc,R)
        if not pts: continue
        nx,ny = (nl["x2"]-nl["x1"]), (nl["y2"]-nl["y1"])
        nlen = math.hypot(nx,ny)
        if nlen==0: continue
        nx/=nlen; ny/=nlen
        L_embed = 2.0 * nl["embed_each"]
        T_ax1 = min(nl["T_yield"], nl["bond_strength"]*L_embed)  # kN/本
        T_ax  = T_ax1 / max(nl["spacing"],1e-6)                   # kN/m
        for (x,y,th) in pts:
            tx,ty = -math.sin(th), math.cos(th)
            cosd = abs(nx*tx + ny*ty)  # |cos|
            T_tan = T_ax * cosd
            hits.append((x,T_tan))
    return hits

def _fs_with_addition(slices, nails):
    # 未補強Fs（あなたの式のまま）
    soil = type("Soil", (), {"gamma":gamma,"c":c,"phi":phi})()
    Fs_un = bishop_fs_unreinforced(slices, soil)

    # 分母（駆動）を未補強の式から算出
    den = 0.0
    for s in slices:
        den += float(s["W"]) * math.sin(float(s["alpha"]))
    # ネイル接線成分の総和（スライスに割当て→総和、で十分）
    add = sum(Tt for (_,Tt) in _nail_T_tan_hits(nails))
    Fs_re = Fs_un + add / max(den, 1e-9)
    return max(Fs_un,1e-6), max(Fs_re,1e-6)

Fs_un, Fs_re = _fs_with_addition(slices_best, ss.nails)

# 表示（安定板２の見た目を崩さない）
st.subheader("結果（同一円弧・同一スライス分割）")
c1,c2 = st.columns(2)
with c1: st.metric("未補強 Fs", f"{Fs_un:.3f}")
with c2: st.metric("補強後 Fs", f"{Fs_re:.3f}")

fig, ax = plt.subplots(figsize=(9,6))
Xg = np.linspace(x_left, x_right, 400); Yg = ground_y_at(Xg)
ax.plot(Xg, Yg, lw=2, label="Slope")
th = np.linspace(0,2*math.pi,400)
ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), "--", label="Slip circle")
for i,nl in enumerate(ss.nails):
    ax.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]], lw=2, label=(f"Nail {i}" if show_legend else None))
ax.set_aspect('equal','box'); ax.set_xlim(x_left,x_right); ax.set_ylim(y_min,y_max)
ax.grid(True, alpha=0.35); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)
