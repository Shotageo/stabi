# pages/30_slip_search.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from stabi_lem import Soil, CircleSlip, generate_slices_on_arc, bishop_fs_unreinforced, circle_xy_from_theta

st.set_page_config(page_title="安定板２｜3) 円弧探索（無補強）", layout="wide")
st.sidebar.header("Plot style")
theme = st.sidebar.selectbox("Theme", ["default", "dark_background"])
tight = st.sidebar.checkbox("Tight layout", True)
show_legend = st.sidebar.checkbox("Show legend", True)
plt.style.use("dark_background" if theme=="dark_background" else "default")

st.title("3) 円弧探索（無補強）")

# 入力をセッションから
H    = float(st.session_state.get("H", 12.0))
beta = float(st.session_state.get("beta_deg", 35.0))
gamma= float(st.session_state.get("gamma", 18.0))
c    = float(st.session_state.get("c", 10.0))
phi  = float(st.session_state.get("phi", 30.0))
x_left  = float(st.session_state.get("x_left", -1.0))
x_right = float(st.session_state.get("x_right",  1.0))
y_min   = float(st.session_state.get("y_min",  -10.0))
y_max   = float(st.session_state.get("y_max",   10.0))

soil = Soil(gamma=gamma, c=c, phi=phi)
tanb = math.tan(math.radians(beta))
def ground_y_at(X): return H - tanb * X

quality = st.sidebar.selectbox("Quality", ["Fast", "Normal", "High"], index=1)
audit = st.sidebar.checkbox("Audit（センター可視化）", False)

def search_best_circle():
    if quality == "Fast":
        coarse_n = (18,10); quick_n = (12,8); refine_nR = 12
    elif quality == "High":
        coarse_n = (42,20); quick_n = (28,14); refine_nR = 24
    else:
        coarse_n = (30,14); quick_n = (20,10); refine_nR = 18
    x_top  = H / max(tanb, 1e-6)
    xc_min, xc_max = -0.2*x_top, 1.2*x_top
    yc_min, yc_max = -2.0*H, 0.2*H
    R_min, R_max   = 0.6*H,  2.2*H

    best=None; centers=[]
    def eval_grid(nx,ny,nR, xl,xh, yl,yh, Rl,Rh, ns):
        nonlocal best, centers
        xs = np.linspace(xl,xh,nx); ys = np.linspace(yl,yh,ny); Rs = np.linspace(Rl,Rh,nR)
        for xc in xs:
            for yc in ys:
                centers.append((xc,yc))
                for R in Rs:
                    slip = CircleSlip(xc=xc,yc=yc,R=R)
                    sls = generate_slices_on_arc(ground_y_at, slip, ns, x_left, x_right, soil.gamma)
                    if not sls: continue
                    Fs = bishop_fs_unreinforced(sls, soil)
                    if not (Fs>0 and np.isfinite(Fs)): continue
                    if (best is None) or (Fs < best[0]): best = (Fs, slip, sls)

    x_top  = H / max(tanb, 1e-6)
    eval_grid(coarse_n[0], coarse_n[1], max(10,int((R_max-R_min)/H*8)),
              -0.2*x_top, 1.2*x_top, -2.0*H, 0.2*H, R_min, R_max, 36)
    if best is not None:
        _, s0, _ = best
        eval_grid(quick_n[0], quick_n[1], max(12,int((R_max-R_min)/H*10)),
                  s0.xc-0.25*x_top, s0.xc+0.25*x_top,
                  s0.yc-0.35*H,     s0.yc+0.35*H,
                  max(R_min,s0.R-0.35*H), min(R_max,s0.R+0.35*H), 40)
    if best is not None:
        _, s0, _ = best
        eval_grid(quick_n[0]+4, quick_n[1]+4, refine_nR,
                  s0.xc-0.12*x_top, s0.xc+0.12*x_top,
                  s0.yc-0.20*H,     s0.yc+0.20*H,
                  max(R_min,s0.R-0.22*H), min(R_max,s0.R+0.22*H), 44)
    return best, centers

best, centers = search_best_circle()

if best is None:
    Fs_un=float("nan"); slip_best=None; sls_best=[]
else:
    Fs_un, slip_best, sls_best = best
    st.session_state["selected_slip"] = {"xc": float(slip_best.xc), "yc": float(slip_best.yc), "R": float(slip_best.R)}
    st.session_state["slices_best"] = sls_best

st.subheader("結果（無補強）")
st.metric("最小 Fs", f"{Fs_un:.3f}" if best else "—")

fig, ax = plt.subplots(figsize=(9,6))
Xg = np.linspace(x_left, x_right, 400); ax.plot(Xg, ground_y_at(Xg), label="Ground", lw=2)
if slip_best is not None:
    th = np.linspace(0,2*math.pi,400)
    Xc, Yc = circle_xy_from_theta(slip_best, th)
    ax.plot(Xc, Yc, "--", label="Selected slip circle")
if audit and centers:
    xs=[p[0] for p in centers]; ys=[p[1] for p in centers]
    ax.scatter(xs, ys, s=8, alpha=0.25, label="Coarse centers")
ax.set_aspect('equal','box'); ax.set_xlim(x_left,x_right); ax.set_ylim(y_min,y_max)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True,alpha=0.4)
ax.set_title(f"Fs_un(min)={Fs_un:.3f}" if best else "円弧が成立しません")
if show_legend: ax.legend(loc="best")
if tight: st.pyplot(fig, clear_figure=False)
else: st.pyplot(fig)

st.page_link("pages/40_soil_nail.py", label="→ 4) ソイルネイル補強へ", icon="🪛")

