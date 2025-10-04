# pages/99_run_with_nails.py
# -*- coding: utf-8 -*-
"""
安定板２で決定・保存済みの最小円弧（selected_slip）とスライス（slices_best）を
“そのまま”読み込んで、セット済みのソイルネイルを含めて補強後 Fs を上乗せ算定するだけのページ。
既存ロジック・既存ページには一切手を入れません。
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 既存の未補強Fs関数をそのまま使用（安定板２の stabi_lem.py）
from stabi_lem import bishop_fs_unreinforced

st.set_page_config(page_title="安定板２｜補強後Fs（読取→上乗せ）", layout="wide")
st.sidebar.header("Plot")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"], index=0)
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Legend", False)
plt.style.use("dark_background" if theme == "dark_background" else "default")

st.title("安定板２：補強後Fs（既存結果を読むだけで上乗せ）")

ss = st.session_state

# --- 1) 入力（“読むだけ”） ---
# 地形・表示レンジ
H      = float(ss.get("H", 12.0))
beta   = float(ss.get("beta_deg", 35.0))
gamma  = float(ss.get("gamma", 18.0))
c      = float(ss.get("c", 10.0))
phi    = float(ss.get("phi", 30.0))
x_left = float(ss.get("x_left", -1.0))
x_right= float(ss.get("x_right",  1.0))
y_min  = float(ss.get("y_min",  -10.0))
y_max  = float(ss.get("y_max",   10.0))

# 円弧（既存名を尊重：dict でも dataclass でも拾えるように）
sel = ss.get("selected_slip", None)
if isinstance(sel, dict) and {"xc","yc","R"} <= set(sel.keys()):
    xc, yc, R = float(sel["xc"]), float(sel["yc"]), float(sel["R"])
else:
    # dataclass 風のバックアップ
    for k in ("slip_best", "best_circle", "selected_circle", "circle_best"):
        obj = ss.get(k, None)
        if obj is not None:
            try:
                xc, yc, R = float(obj.xc), float(obj.yc), float(obj.R)
                break
            except Exception:
                pass
    else:
        st.error("最小円弧（selected_slip）が未保存です。安定板２（無補強）で最小円弧を確定してください。")
        st.stop()

# スライス（“同一分割”を再利用）
slices_best = None
for k in ("slices_best", "best_slices", "slices", "selected_slices"):
    if ss.get(k):
        slices_best = ss.get(k)
        break
if not slices_best:
    st.error("安定板２のスライス情報（slices_best）が見つかりません。同一分割での比較ができません。")
    st.stop()

# ネイル（セット済みを読むだけ。無ければ0本で計算）
nails = ss.get("nails", [])

# --- 2) 未補強Fs：既存式をそのまま使用 ---
SoilLite = type("SoilLite", (), {"gamma": gamma, "c": c, "phi": phi})
Fs_un = bishop_fs_unreinforced(slices_best, SoilLite())

# --- 3) ネイル接線成分 T_tan の総和（x位置でスライスに束ねる必要はなく総和でOK） ---
def _line_circle_hits(x1, y1, x2, y2, xc, yc, R):
    dx, dy = (x2 - x1), (y2 - y1)
    fx, fy = (x1 - xc), (y1 - yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - R*R
    disc = B*B - 4*A*C
    if A == 0 or disc < 0:
        return []
    rt = math.sqrt(disc); out = []
    for sgn in (-1.0, 1.0):
        t = (-B + sgn*rt) / (2*A)
        if 0.0 <= t <= 1.0:
            x = x1 + t*dx; y = y1 + t*dy
            th = math.atan2(y - yc, x - xc)
            out.append((x, y, th))
    return out

def _sum_T_tan(nails):
    total = 0.0
    for nl in nails:
        # 想定キー：x1,y1,x2,y2, spacing, T_yield, bond_strength, embed_each
        try:
            x1,y1,x2,y2 = nl["x1"], nl["y1"], nl["x2"], nl["y2"]
            spacing = float(nl["spacing"]); Ty = float(nl["T_yield"])
            bond = float(nl["bond_strength"]); emb = float(nl["embed_each"])
        except Exception:
            continue

        pts = _line_circle_hits(x1,y1,x2,y2, xc,yc,R)
        if not pts:
            continue

        # ネイル軸
        nx, ny = (x2 - x1), (y2 - y1)
        nlen = math.hypot(nx, ny)
        if nlen == 0:
            continue
        nx /= nlen; ny /= nlen

        # 有効軸力（本→m換算）
        L_embed = 2.0 * emb
        T_axial_1 = min(Ty, bond * L_embed)   # kN/本
        T_axial   = T_axial_1 / max(spacing, 1e-6)  # kN/m

        for (_, _, th) in pts:
            tx, ty = -math.sin(th), math.cos(th)   # 円弧接線
            cosd = abs(nx*tx + ny*ty)              # 接線方向の寄与 |cosΔ|
            total += T_axial * cosd
    return total

sum_Ttan = _sum_T_tan(nails)

# --- 4) 補強後Fs（未補強式の“分母”を流用して上乗せ） ---
den = 0.0
for s in slices_best:
    den += float(s["W"]) * math.sin(float(s["alpha"]))
Fs_re = Fs_un + (sum_Ttan / max(den, 1e-9))

# --- 5) 結果表示（描画も“見るだけ”） ---
st.subheader("結果（同一円弧・同一分割・既存式そのまま）")
c1, c2 = st.columns(2)
with c1: st.metric("未補強 Fs", f"{Fs_un:.3f}")
with c2: st.metric("補強後 Fs", f"{Fs_re:.3f}")

# 図（地形ラインと円弧・ネイル。既存レンジをそのまま使用。）
tanb = math.tan(math.radians(beta))
def ground_y_at(X): return H - tanb * np.asarray(X, float)

fig, ax = plt.subplots(figsize=(9, 6))
Xg = np.linspace(x_left, x_right, 400)
ax.plot(Xg, ground_y_at(Xg), lw=2, label="Ground")
th = np.linspace(0, 2*math.pi, 400)
ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), "--", label="Slip circle")

for i, nl in enumerate(nails):
    try:
        ax.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]],
                lw=2, label=(f"Nail {i}" if show_legend else None))
    except Exception:
        pass

ax.set_aspect('equal', 'box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.35)
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

st.caption("注）安定板２の計算ロジックは一切変更していません。未補強Fsの分母を流用し、ネイルの接線成分ΣT_tanを抵抗側に上乗せする最小実装です。")

