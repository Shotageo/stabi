# streamlit_app.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stabi_lem import (
    Soil, CircleSlip, Nail,
    generate_slices_on_arc,
    bishop_fs_unreinforced, bishop_fs_with_nails,
    circle_xy_from_theta
)

# ================== 基本設定 ==================
st.set_page_config(page_title="Stabi｜安定板２（統合）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", value=True)
show_legend = st.sidebar.checkbox("Show legend", value=True)
plt.style.use("dark_background" if theme == "dark_background" else "default")

st.title("Stabi｜安定板２（統合）：無補強の最小円弧 → 同一ページでソイルネイル補強")

# Debug（任意）
with st.sidebar.expander("Debug / 状態", expanded=False):
    ss = st.session_state
    st.write({
        "selected_slip": ss.get("selected_slip"),
        "H": ss.get("H"), "beta_deg": ss.get("beta_deg"),
        "gamma": ss.get("gamma"), "c": ss.get("c"), "phi": ss.get("phi"),
        "x_left": ss.get("x_left"), "x_right": ss.get("x_right"),
        "nails_count": len(ss.get("nails", [])),
    })
    if st.button("selected_slip をクリア"): ss.pop("selected_slip", None)
    if st.button("nails を全消去"): ss["nails"] = []

# ================== 入力：地形・材料 ==================
st.subheader("地形・材料")
cA, cB, cC = st.columns(3)
with cA:
    H  = st.number_input("法高さ H [m]", 1.0, 100.0, float(st.session_state.get("H", 12.0)), 0.5)
    beta = st.number_input("法勾配角 β [deg]", 5.0, 80.0, float(st.session_state.get("beta_deg", 35.0)), 0.5)
with cB:
    gamma = st.number_input("γ [kN/m³]", 10.0, 30.0, float(st.session_state.get("gamma", 18.0)), 0.5)
    c = st.number_input("c [kPa]", 0.0, 300.0, float(st.session_state.get("c", st.session_state.get("c_kpa", 10.0))), 1.0)
with cC:
    phi = st.number_input("φ [deg]", 0.0, 45.0, float(st.session_state.get("phi", st.session_state.get("phi_deg", 30.0))), 0.5)

soil = Soil(gamma=gamma, c=c, phi=phi)

# 地表（配列安全）
beta_rad = math.radians(beta)
tanb = math.tan(beta_rad)
def ground_y_at(X: np.ndarray) -> np.ndarray:
    return H - tanb * X

# 表示レンジ（保存しておく）
x_top  = H / max(tanb, 1e-6)
x_left, x_right = -0.1 * x_top, 1.1 * x_top
y_min, y_max = -1.2 * H, 1.2 * H
st.session_state["H"] = H
st.session_state["beta_deg"] = beta
st.session_state["gamma"] = gamma
st.session_state["c"] = c
st.session_state["phi"] = phi
st.session_state["x_left"] = x_left
st.session_state["x_right"] = x_right
st.session_state["y_min"] = y_min
st.session_state["y_max"] = y_max

# ================== タブ：①無補強探索／②ソイルネイル補強 ==================
tab1, tab2 = st.tabs(["① 無補強の最小Fs円弧を探索", "② 同じ円弧にソイルネイル補強"])

# セッション初期化（ネイル管理）
if "nails" not in st.session_state:
    st.session_state.nails = []

# ------------------------------------------------------------
# ① 無補強の最小Fs円弧（Coarse→Quick→Refine）
# ------------------------------------------------------------
with tab1:
    st.markdown("無補強で最小Fsの円弧を探索し、**このページ内のタブ②にそのまま引き渡します**。")
    quality = st.selectbox("Quality", ["Fast", "Normal", "High"], index=1, key="qual_unreinforced")
    audit = st.checkbox("Audit（センター可視化）", value=False, key="audit_unreinforced")

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

        best = None  # (Fs, slip, slices)
        centers = []

        def eval_grid(nx, ny, nR, xc_lo, xc_hi, yc_lo, yc_hi, R_lo, R_hi, nslices):
            nonlocal best, centers
            xs = np.linspace(xc_lo, xc_hi, nx)
            ys = np.linspace(yc_lo, yc_hi, ny)
            Rs = np.linspace(R_lo, R_hi, nR)
            for xc in xs:
                for yc in ys:
                    centers.append((xc, yc))
                    for R in Rs:
                        slip = CircleSlip(xc=xc, yc=yc, R=R)
                        slices = generate_slices_on_arc(
                            ground_y_at, slip, n_slices=nslices,
                            x_min=x_left, x_max=x_right, soil_gamma=soil.gamma
                        )
                        if not slices:
                            continue
                        Fs_un = bishop_fs_unreinforced(slices, soil)
                        if not (Fs_un > 0 and np.isfinite(Fs_un)):
                            continue
                        if (best is None) or (Fs_un < best[0]):
                            best = (Fs_un, slip, slices)

        eval_grid(coarse_n[0], coarse_n[1], max(10, int((R_max-R_min)/H*8)),
                  xc_min, xc_max, yc_min, yc_max, R_min, R_max, nslices=36)

        if best is not None:
            _, s0, _ = best
            dx = 0.25 * x_top; dy = 0.35 * H; dR = 0.35 * H
            eval_grid(quick_n[0], quick_n[1], max(12, int((R_max-R_min)/H*10)),
                      s0.xc - dx, s0.xc + dx,
                      s0.yc - dy, s0.yc + dy,
                      max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                      nslices=40)

        if best is not None:
            _, s0, _ = best
            dx = 0.12 * x_top; dy = 0.20 * H; dR = 0.22 * H
            eval_grid(quick_n[0]+4, quick_n[1]+4, refine_nR,
                      s0.xc - dx, s0.xc + dx,
                      s0.yc - dy, s0.yc + dy,
                      max(R_min, s0.R - dR), min(R_max, s0.R + dR),
                      nslices=44)
        return best, centers

    best, centers_seen = search_best_circle()

    # 結果・保存
    if best is None:
        Fs_un = float("nan"); slip_best = None; slices_best = []
    else:
        Fs_un, slip_best, slices_best = best
        st.session_state["selected_slip"] = {"xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)}
        st.session_state["slices_best"]  = slices_best  # そのまま再利用

    st.subheader("結果（無補強）")
    st.metric("最小 Fs", f"{Fs_un:.3f}" if best else "—")

    # 図
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    Xg = np.linspace(x_left, x_right, 400)
    ax1.plot(Xg, ground_y_at(Xg), label="Ground", linewidth=2)
    if slip_best is not None:
        th = np.linspace(0, 2*math.pi, 400)
        Xc, Yc = circle_xy_from_theta(slip_best, th)
        ax1.plot(Xc, Yc, "--", label="Selected slip circle")
    if audit and centers_seen:
        xs = [p[0] for p in centers_seen]; ys = [p[1] for p in centers_seen]
        ax1.scatter(xs, ys, s=8, alpha=0.25, label="Coarse centers")
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(x_left, x_right); ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.grid(True, alpha=0.4)
    ax1.set_title(f"Fs_un(min)={Fs_un:.3f}" if best else "円弧が成立しません（地表との土塊なし）")
    if show_legend: ax1.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(fig1)

# ------------------------------------------------------------
# ② 同じ円弧にソイルネイル補強（同一ページ）
# ------------------------------------------------------------
with tab2:
    st.markdown("タブ①で求めた **“選択円弧”** にネイル寄与を上乗せして、**補強後 Fs** を計算します。")

    # 円弧の受け取り
    sel = st.session_state.get("selected_slip", None)
    if isinstance(sel, dict) and {"xc","yc","R"} <= sel.keys():
        slip = CircleSlip(xc=float(sel["xc"]), yc=float(sel["yc"]), R=float(sel["R"]))
    else:
        st.error("先にタブ①で無補強の最小Fs円弧を確定してください。")
        st.stop()

    # ネイルUI
    st.subheader("ソイルネイル（線分で表現）")
    with st.expander("ネイルを追加"):
        col1, col2, col3 = st.columns(3)
        with col1:
            x1 = st.number_input("x1 [m]", -200.0, 200.0, max(1.0, 0.15*x_top), 0.1, key="nx1")
            y1 = st.number_input("y1 [m]", -200.0, 200.0, H*0.55, 0.1, key="ny1")
            spacing = st.number_input("spacing [m]", 0.05, 5.0, 1.5, 0.05, key="nsp")
        with col2:
            x2 = st.number_input("x2 [m]", -200.0, 200.0, max(1.0, 0.55*x_top), 0.1, key="nx2")
            y2 = st.number_input("y2 [m]", -200.0, 200.0, H*0.7, 0.1, key="ny2")
            T_yield = st.number_input("T_yield [kN/本]", 10.0, 3000.0, 200.0, 10.0, key="nyld")
        with col3:
            bond = st.number_input("bond_strength [kN/m]", 1.0, 500.0, 80.0, 1.0, key="nbnd")
            emb  = st.number_input("有効定着長(片側) [m]", 0.1, 5.0, 0.5, 0.1, key="nemb")
            if st.button("＋このネイルを追加", type="primary"):
                st.session_state.nails.append(
                    Nail(x1=x1, y1=y1, x2=x2, y2=y2,
                         spacing=spacing, T_yield=T_yield,
                         bond_strength=bond, embed_length_each_side=emb)
                )

    if st.session_state.nails:
        idx = st.selectbox("削除対象（インデックス）", list(range(len(st.session_state.nails))), 0)
        cdel1, cdel2 = st.columns(2)
        with cdel1:
            if st.button("選択ネイルを削除"): st.session_state.nails.pop(idx)
        with cdel2:
            if st.button("全削除"): st.session_state.nails.clear()

    # スライスはタブ①の結果を再利用（なければ生成）
    slices_best = st.session_state.get("slices_best", None)
    if not slices_best:
        slices_best = generate_slices_on_arc(ground_y_at, slip, n_slices=40,
                                             x_min=x_left, x_max=x_right, soil_gamma=soil.gamma)

    Fs_un = bishop_fs_unreinforced(slices_best, soil) if slices_best else float("nan")
    Fs_re = bishop_fs_with_nails(slices_best, soil, slip, st.session_state.nails) if slices_best else float("nan")

    st.subheader("結果（同一の選択円弧）")
    c1, c2 = st.columns(2)
    with c1: st.metric("未補強 Fs", f"{Fs_un:.3f}" if slices_best else "—")
    with c2: st.metric("補強後 Fs", f"{Fs_re:.3f}" if slices_best else "—")

    # 図
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    Xg = np.linspace(x_left, x_right, 400)
    ax2.plot(Xg, ground_y_at(Xg), label="Ground", linewidth=2)
    th = np.linspace(0, 2*math.pi, 400)
    ax2.plot(slip.xc + slip.R*np.cos(th), slip.yc + slip.R*np.sin(th), "--", label="Selected slip circle")
    for i, nl in enumerate(st.session_state.nails):
        ax2.plot([nl.x1, nl.x2], [nl.y1, nl.y2], linewidth=2, label=f"Nail {i}" if show_legend else None)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(x_left, x_right); ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.grid(True, alpha=0.4)
    ax2.set_title(f"Fs_un={Fs_un:.3f} / Fs_re={Fs_re:.3f}" if slices_best else "（この円弧では土塊が形成されません）")
    if show_legend: ax2.legend(loc="best")
    if tight: plt.tight_layout()
    st.pyplot(fig2)
