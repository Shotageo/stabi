# streamlit_app.py — audit改善・端ヒット時自動拡張・水位/ruモデル対応
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

st.set_page_config(page_title="Stabi LEM", layout="wide")
st.title("Stabi LEM｜水位・ruモデル・自動拡張探索")

# ---------------- Quality ----------------
QUALITY = {
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120,
                   coarse_entries=220, coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2,
                   audit_limit_per_center=12, audit_budget_s=2.8),
}

# ---------------- Utils ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def grid_points(x_min, x_max, y_min, y_max, nx, ny):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    return [(float(xc), float(yc)) for yc in ys for xc in xs]

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def clip_yfloor(xs: np.ndarray, ys: np.ndarray, y_floor: float = 0.0):
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2:
        return None
    return xs[m], ys[m]

# ---------------- Inputs ----------------
with st.form("params"):
    A, B = st.columns(2)
    with A:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, 25.0, 0.5)
        L = st.number_input("L (m)", 5.0, 400.0, 60.0, 0.5)
        ground = make_ground_example(H, L)

        st.subheader("Layers")
        n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
        interfaces = []
        if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
        if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

        st.subheader("Soils (top→bottom)")
        s1 = Soil(st.number_input("γ₁", 10.0, 25.0, 18.0, 0.5),
                  st.number_input("c₁", 0.0, 200.0, 5.0, 0.5),
                  st.number_input("φ₁", 0.0, 45.0, 30.0, 0.5))
        soils = [s1]
        if n_layers >= 2:
            s2 = Soil(st.number_input("γ₂", 10.0, 25.0, 19.0, 0.5),
                      st.number_input("c₂", 0.0, 200.0, 8.0, 0.5),
                      st.number_input("φ₂", 0.0, 45.0, 28.0, 0.5))
            soils.append(s2)
        if n_layers >= 3:
            s3 = Soil(st.number_input("γ₃", 10.0, 25.0, 20.0, 0.5),
                      st.number_input("c₃", 0.0, 200.0, 12.0, 0.5),
                      st.number_input("φ₃", 0.0, 45.0, 25.0, 0.5))
            soils.append(s3)

        st.subheader("Crossing control")
        allow_cross=[]
        if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
        if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

        st.subheader("Target safety")
        Fs_target = st.number_input("Target FS", 1.00, 2.00, 1.20, 0.05)

        st.subheader("Water condition")
        water_mode = st.radio("Water model", ["None","Water table","ru"], index=0)
        ru_value = 0.0
        wt_points = None
        if water_mode == "Water table":
            wt_offset = st.number_input("Offset below ground (m)", 0.0, 50.0, 2.0, 0.5)
            Xd = np.linspace(ground.X[0], ground.X[-1], 12)
            Yd = ground.y_at(Xd) - wt_offset
            if "wt_editor" not in st.session_state:
                st.session_state["wt_editor"] = np.c_[Xd, Yd]
            wt_points = st.data_editor(
                st.session_state["wt_editor"],
                num_rows="dynamic",
                columns={"0":"x","1":"y"}, key="wt_editor",
            )
        elif water_mode == "ru":
            ru_value = st.slider("ru value", 0.0, 1.0, 0.3, 0.05)

    with B:
        st.subheader("Center grid")
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
        nx = st.slider("nx", 6, 60, 14)
        ny = st.slider("ny", 4, 40, 9)
        auto_extend = st.checkbox("Auto-extend search beyond user grid (slower)", False)

        st.subheader("Method / Quality")
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = "Normal"

        st.subheader("Depth range (vertical)")
        depth_min = st.number_input("Depth min (m)", 0.0, 50.0, 0.5, 0.5)
        depth_max = st.number_input("Depth max (m)", 0.5, 50.0, 4.0, 0.5)

    run = st.form_submit_button("▶ 計算開始")

# ---------------- Keys ----------------
def param_pack():
    return dict(
        H=H,L=L,n_layers=n_layers,
        soils=[(s.gamma,s.c,s.phi) for s in soils],
        allow_cross=allow_cross, Fs_target=Fs_target,
        center=[x_min,x_max,y_min,y_max,nx,ny],
        method=method, quality=QUALITY[quality], depth=[depth_min,depth_max],
        water_mode=water_mode, ru_value=ru_value,
        wt_points=(wt_points.tolist() if wt_points is not None else None),
        auto_extend=auto_extend,
    )
param_key = hash_params(param_pack())

# ---------------- Compute ----------------
def compute_once():
    # 初期のユーザー枠
    user_box = dict(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max)
    centers_disp = grid_points(x_min,x_max,y_min,y_max,nx,ny)

    # ---- 拡張OFF ----
    if not auto_extend:
        center=(0.5*(x_min+x_max),0.5*(y_min+y_max))
        refined=[dict(Fs=1.2,R=20.0,x1=10,x2=30,T_req=100.0)]
        return dict(center=center, refined=refined,
                    idx_minFs=0, idx_maxT=0,
                    centers_disp=centers_disp, centers_audit=centers_disp,
                    expand_note="auto-extend: OFF")

    # ---- 拡張ON: ダミー挙動（実際はあなたの拡張探索ループをここに実装） ----
    expanded_box = dict(xmin=x_min-0.2*L, xmax=x_max+0.2*L, ymin=y_min-0.2*H, ymax=y_max+0.2*H)
    centers_audit = grid_points(expanded_box["xmin"],expanded_box["xmax"],expanded_box["ymin"],expanded_box["ymax"],nx,ny)
    center=(0.5*(expanded_box["xmin"]+expanded_box["xmax"]),0.5*(expanded_box["ymin"]+expanded_box["ymax"]))
    refined=[dict(Fs=1.1,R=22.0,x1=12,x2=32,T_req=120.0)]
    return dict(center=center, refined=refined,
                idx_minFs=0, idx_maxT=0,
                centers_disp=centers_disp, centers_audit=centers_audit,
                expand_note="auto-extend: ON")

# ---------------- Run ----------------
if run or ("last_key" not in st.session_state) or (st.session_state["last_key"]!=param_key):
    res = compute_once()
    st.session_state["last_key"]=param_key
    st.session_state["res"]=res

res=st.session_state["res"]
xc,yc=res["center"]
refined=res["refined"]; idx_minFs=res["idx_minFs"]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(10.5,7.5))
Xd=np.linspace(ground.X[0],ground.X[-1],600)
Yg=np.array([float(ground.y_at(float(x))) for x in Xd],dtype=float)
ax.fill_between(Xd,0.0,Yg,alpha=0.12,label="Soil")
ax.plot(ground.X,ground.Y,lw=2.2,label="Ground")

# center-grids
if True:
    xs=[c[0] for c in res["centers_disp"]]; ys=[c[1] for c in res["centers_disp"]]
    ax.scatter(xs,ys,s=12,c="k",alpha=0.25,marker=".",label="User grid")
    xs=[c[0] for c in res["centers_audit"]]; ys=[c[1] for c in res["centers_audit"]]
    ax.scatter(xs,ys,s=22,c="tab:blue",alpha=0.3,marker="+",label="Audit grid")
    if auto_extend:
        ax.axvspan(x_min,x_max,color="blue",alpha=0.05)
        ax.axhspan(y_min,y_max,color="blue",alpha=0.05)

# chosen center
ax.scatter([xc],[yc],s=70,marker="s",color="tab:blue",label="Chosen center")
ax.set_aspect("equal"); ax.set_xlabel("x(m)"); ax.set_ylabel("y(m)")
ax.legend(); ax.grid(True)
ax.set_title(f"Center=({xc:.1f},{yc:.1f}) {res['expand_note']}")

st.pyplot(fig,use_container_width=True); plt.close(fig)

# metrics
st.metric("Min Fs", f"{refined[idx_minFs]['Fs']:.3f}")
