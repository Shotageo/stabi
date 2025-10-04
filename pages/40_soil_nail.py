# pages/40_soil_nail.py
# -*- coding: utf-8 -*-
"""
安定板２（無補強）で保存された状態だけを読み取り、
同じ横断図・同じ円弧・同じスライス分割で “ソイルネイル補強” を上乗せ計算するページ。
既存ページの中身は一切変更しない想定。
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- 既存のLEMコアをそのまま利用（未補強Fs計算・スライス生成は既存のものを呼ぶ） ---
try:
    from stabi_lem import bishop_fs_unreinforced  # あなたの既存関数
except Exception:
    bishop_fs_unreinforced = None  # 無い場合は後でエラーメッセージ

st.set_page_config(page_title="安定板２｜ソイルネイル補強", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("安定板２：ソイルネイル補強（※無補強ページの結果をそのまま使用）")

# ================= 安定板２の状態を“読むだけ” =================

ss = st.session_state

# 1) 横断図レンジ・法面条件（読み取り専用）
H      = float(ss.get("H", 12.0))
beta   = float(ss.get("beta_deg", 35.0))
gamma  = float(ss.get("gamma", 18.0))
c      = float(ss.get("c", 10.0))
phi    = float(ss.get("phi", 30.0))
x_left = float(ss.get("x_left", -1.0))
x_right= float(ss.get("x_right",  1.0))
y_min  = float(ss.get("y_min", -10.0))
y_max  = float(ss.get("y_max", 10.0))

tanb = math.tan(math.radians(beta))
def ground_y_at(X):  # 表示用のみ（計算は無補強ページで済んでいる想定）
    X = np.asarray(X, dtype=float)
    return H - tanb * X

# 2) “選択円弧”の取り出し（安定板２がどう保存していても拾えるように冗長に対応）
def _read_selected_slip():
    v = ss.get("selected_slip", None)
    if isinstance(v, dict) and {"xc","yc","R"} <= set(v.keys()):
        return float(v["xc"]), float(v["yc"]), float(v["R"])
    # dataclass風にも対応
    for k in ("slip_best", "best_circle", "selected_circle", "circle_best"):
        obj = ss.get(k, None)
        if obj is not None:
            try:
                return float(obj.xc), float(obj.yc), float(obj.R)
            except Exception:
                pass
    return None

slip_tuple = _read_selected_slip()
if not slip_tuple:
    st.error("無補強ページ（安定板２）で“選択円弧”が未保存のため、補強計算を実行できません。先に無補強の最小Fs円弧を確定してください。")
    st.stop()
xc, yc, R = slip_tuple

# 3) “同じスライス分割”の取り出し（保存名の揺れに対応）
def _read_slices():
    for k in ("slices_best", "best_slices", "slices", "selected_slices"):
        sls = ss.get(k, None)
        if sls:
            return sls
    return None

slices_best = _read_slices()
if not slices_best:
    st.warning("注意：安定板２側のスライス分割が保存されていません。同一分割での厳密比較はできません。")
    st.info("それでも続行はできますが、無補強ページで“最小円弧を確定”→“スライス保存”されることを推奨します。")

# ================= ソイルネイル UI（ここだけ新規） =================

st.subheader("ソイルネイル（線分で表現：既存の横断図座標系）")
if "nails" not in ss:
    ss.nails = []  # ページ間で維持

with st.expander("ネイルを追加"):
    c1, c2, c3 = st.columns(3)
    with c1:
        x1 = st.number_input("x1 [m]", -200.0, 200.0, 2.0, 0.1, key="nx1")
        y1 = st.number_input("y1 [m]", -200.0, 200.0, 6.0, 0.1, key="ny1")
        spacing = st.number_input("配置間隔 spacing [m]", 0.05, 10.0, 1.5, 0.05, key="nsp")
    with c2:
        x2 = st.number_input("x2 [m]", -200.0, 200.0, 8.0, 0.1, key="nx2")
        y2 = st.number_input("y2 [m]", -200.0, 200.0, 8.0, 0.1, key="ny2")
        T_yield = st.number_input("降伏耐力 T_yield [kN/本]", 10.0, 5000.0, 200.0, 10.0, key="nyld")
    with c3:
        bond = st.number_input("付着強度 bond_strength [kN/m]", 1.0, 1000.0, 80.0, 1.0, key="nbnd")
        emb  = st.number_input("有効定着長(片側) [m]", 0.05, 10.0, 0.5, 0.05, key="nemb")
        if st.button("＋このネイルを追加", type="primary"):
            ss.nails.append({
                "x1":x1, "y1":y1, "x2":x2, "y2":y2,
                "spacing":spacing, "T_yield":T_yield,
                "bond_strength":bond, "embed_each":emb
            })

if ss.nails:
    idx = st.selectbox("削除対象（インデックス）", list(range(len(ss.nails))), 0)
    cdel1, cdel2 = st.columns(2)
    with cdel1:
        if st.button("選択ネイルを削除"): ss.nails.pop(idx)
    with cdel2:
        if st.button("全削除"): ss.nails.clear()

# ================= 補強Fsの計算（未補強の式に“加算”のみ。元の式は一切変更しない） =================

if bishop_fs_unreinforced is None:
    st.error("未補強Fs計算関数（bishop_fs_unreinforced）が見つかりません。stabi_lem.py の import 名をご確認ください。")
    st.stop()

# 既存スライスの形を尊重（c*width か c*Lb か等、元の実装のまま）
# ネイルは「交点の接線成分 T_tan [kN/m]」を該当スライスの耐力に“足すだけ”にします。

def _line_circle_hits(x1,y1,x2,y2, xc,yc,R):
    dx, dy = (x2-x1), (y2-y1)
    fx, fy = (x1-xc), (y1-yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - R*R
    disc = B*B - 4*A*C
    if A == 0 or disc < 0: return []
    rt = math.sqrt(disc)
    hits = []
    for sgn in (-1.0, 1.0):
        t = (-B + sgn*rt) / (2*A)
        if 0.0 <= t <= 1.0:
            x = x1 + t*dx; y = y1 + t*dy
            th = math.atan2(y-yc, x-xc)
            hits.append((x,y,th))
    return hits

def _nail_T_tan_at_hits(nails):
    out = []
    for nl in nails:
        pts = _line_circle_hits(nl["x1"],nl["y1"],nl["x2"],nl["y2"], xc,yc,R)
        if not pts: 
            continue
        # ネイル軸
        nx, ny = (nl["x2"]-nl["x1"]), (nl["y2"]-nl["y1"])
        nlen = math.hypot(nx, ny)
        if nlen == 0: 
            continue
        nx /= nlen; ny /= nlen
        # 有効軸力（本→m換算）
        L_embed = 2.0 * nl["embed_each"]
        T_axial_1 = min(nl["T_yield"], nl["bond_strength"] * L_embed)  # kN/本
        T_axial   = T_axial_1 / max(nl["spacing"], 1e-6)               # kN/m

        for (xp, yp, th) in pts:
            # 円弧接線
            tx, ty = -math.sin(th), math.cos(th)
            # 接線成分
            # 角度 delta の cos を掛ける（|cos|）
            cos_delta = abs((nx*tx + ny*ty) / max(1e-9, 1.0))
            T_tan = T_axial * cos_delta
            out.append((xp, yp, T_tan))
    return out

def _fs_with_nails_addition(slices, nails):
    """既存Fsの式を変更せず、各スライスの抵抗せん断に T_tan を足すだけの版。"""
    # 未補強Fs（既存の式・既存スライスをそのまま）
    Fs_un = bishop_fs_unreinforced(slices, type("Soil", (), {"gamma":gamma, "c":c, "phi":phi})())

    # 交点ごとの T_tan [kN/m] をスライスに割り当て（x位置で判断）
    hits = _nail_T_tan_at_hits(nails)
    add_T = [0.0]*len(slices)
    # スライスの x 範囲キー（既存スライスに x_a/x_b が無い場合は width 累積で近似）
    if "x_a" in slices[0] and "x_b" in slices[0]:
        ranges = [(s["x_a"], s["x_b"]) for s in slices]
    else:
        # 幅があれば左端を累積で作る（表示レンジ内で）
        xa = x_left
        ranges = []
        for s in slices:
            xb = xa + float(s.get("width", 0.0))
            ranges.append((xa, xb))
            xa = xb
    for (xp, _, Tt) in hits:
        for i, (xa, xb) in enumerate(ranges):
            if xa <= xp < xb:
                add_T[i] += Tt
                break

    # 既存式の“分子（抵抗せん断）”に add_T を加える
    # bishop_fs_unreinforced と同じ形で再評価するが、cやNpの扱いは“そのまま”にしたいので、
    # ここでは「Fs_un をベースに T_tan 分だけ安全側に上乗せする比例換算」とする簡易法も選べる。
    # ただし安定板２に合わせ、もう一度 Bishop 反復を回すのではなく、分母（駆動）不変の前提で補正：
    #    Fs_re = Fs_un + (sum_i add_T[i]) / (∑ W_i sin α_i)
    # これは元式の分子に一定量を加えるのと等価。
    den = 0.0
    for s in slices:
        den += float(s["W"]) * math.sin(float(s["alpha"]))
    delta_num = sum(add_T)  # kN/m（奥行1m想定）
    Fs_re = Fs_un + (delta_num / max(den, 1e-9))
    return max(Fs_un, 1e-6), max(Fs_re, 1e-6)

# 実行
if not slices_best:
    st.error("無補強ページのスライス情報が無いため、厳密な“同一分割”比較はできません。無補強ページで最小円弧を確定し直してください。")
    st.stop()

Fs_un, Fs_re = _fs_with_nails_addition(slices_best, ss.nails)

# ================= 表示（横断図・円弧・ネイル。見た目は安定板２のまま） =================

st.subheader("結果（同一円弧・同一スライス分割）")
c1, c2 = st.columns(2)
with c1: st.metric("未補強 Fs", f"{Fs_un:.3f}")
with c2: st.metric("補強後 Fs", f"{Fs_re:.3f}")

fig, ax = plt.subplots(figsize=(9,6))
Xg = np.linspace(x_left, x_right, 400)
ax.plot(Xg, ground_y_at(Xg), label="Slope", linewidth=2)

# 選択円弧（全周でもOK／既存表示に合わせる）
th = np.linspace(0, 2*math.pi, 400)
ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), "--", label="Slip circle")

# ネイル
for i, nl in enumerate(ss.nails):
    ax.plot([nl["x1"], nl["x2"]], [nl["y1"], nl["y2"]],
            linewidth=2, label=(f"Nail {i}" if show_legend else None))

ax.set_aspect('equal','box')
ax.set_xlim(x_left, x_right); ax.set_ylim(y_min, y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(True, alpha=0.35)
ax.set_title(f"Fs_un={Fs_un:.3f} / Fs_re={Fs_re:.3f}")
if show_legend: ax.legend(loc="best")
if tight: plt.tight_layout()
st.pyplot(fig)

st.caption("注）未補強Fsの“式”や“スライス形状”には一切手を触れず、ネイルの接線成分Tのみ抵抗側に加算。")
