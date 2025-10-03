# streamlit_app.py — 多段UI + 教授コメント（Page3に設定中の横断図を追加／WTクリップ／H/L共有）
from __future__ import annotations
import streamlit as st
import numpy as np, heapq, time, hashlib, json, random, math
import matplotlib.pyplot as plt

from stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

from stabi_suggest import (
    lint_geometry_and_water, lint_soils_and_materials, lint_arc_and_slices, lint_nails_layout,
    render_suggestions, default_dispatcher
)

st.set_page_config(page_title="Stabi LEM｜多段UI+教授コメント", layout="wide")
st.title("Stabi LEM｜多段UI + 教授コメント（Phase-1）")

# ---------------- 共有セッションの初期化（H/L 共有など） ----------------
if "H" not in st.session_state: st.session_state["H"] = 25.0
if "L" not in st.session_state: st.session_state["L"] = 60.0
if "water_mode" not in st.session_state: st.session_state["water_mode"] = "WT"  # or "ru" or "WT+ru"
if "tau_grout_cap_kPa" not in st.session_state: st.session_state["tau_grout_cap_kPa"] = None
if "mu" not in st.session_state: st.session_state["mu"] = 0.0
if "ru" not in st.session_state: st.session_state["ru"] = 0.0
if "wl_points" not in st.session_state: st.session_state["wl_points"] = None

# ---------------- Utils ----------------
def fs_to_color(fs: float):
    if fs < 1.0: return (0.85, 0.0, 0.0)
    if fs < 1.2:
        t = (fs - 1.0) / 0.2
        return (1.0, 0.50 + 0.50*t, 0.0)
    return (0.0, 0.55, 0.0)

def hash_params(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def clip_yfloor(xs: np.ndarray, ys: np.ndarray, y_floor: float = 0.0):
    m = ys >= (y_floor - 1e-12)
    if np.count_nonzero(m) < 2: return None
    return xs[m], ys[m]

# ---------------- Global quality presets ----------------
QUALITY = {
    "Coarse": dict(quick_slices=10, final_slices=30, n_entries_final=900,  probe_n_min_quick=81,
                   limit_arcs_quick=80,  show_k=60,  top_thick=10,
                   coarse_subsample="every 3rd", coarse_entries=160,
                   coarse_limit_arcs=50, coarse_probe_min=61,
                   budget_coarse_s=0.6, budget_quick_s=0.9),
    "Normal": dict(quick_slices=12, final_slices=40, n_entries_final=1300, probe_n_min_quick=101,
                   limit_arcs_quick=120, show_k=120, top_thick=12,
                   coarse_subsample="every 2nd", coarse_entries=220,
                   coarse_limit_arcs=70, coarse_probe_min=81,
                   budget_coarse_s=0.8, budget_quick_s=1.2),
    "Fine": dict(quick_slices=16, final_slices=50, n_entries_final=1700, probe_n_min_quick=121,
                 limit_arcs_quick=160, show_k=180, coarse_subsample="full", coarse_entries=320,
                 coarse_limit_arcs=100, coarse_probe_min=101,
                 budget_coarse_s=1.2, budget_quick_s=1.8),
    "Very-fine": dict(quick_slices=20, final_slices=60, n_entries_final=2200, probe_n_min_quick=141,
                      limit_arcs_quick=220, show_k=240, coarse_subsample="full", coarse_entries=420,
                      coarse_limit_arcs=140, coarse_probe_min=121,
                      budget_coarse_s=1.8, budget_quick_s=2.6),
}

# ---------------- Sidebar nav ----------------
page = st.sidebar.radio("Pages", ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強）", "4) ネイル配置", "5) 補強後解析"])

# ---------------- Page 1: 地形・水位 ----------------
if page.startswith("1"):
    colL, colR = st.columns([3,1])
    with colL:
        st.subheader("Geometry")
        H = st.number_input("H (m)", 5.0, 200.0, st.session_state["H"], 0.5, key="H")
        L = st.number_input("L (m)", 5.0, 400.0, st.session_state["L"], 0.5, key="L")
        ground = make_ground_example(st.session_state["H"], st.session_state["L"])

        st.subheader("Water")
        water_mode = st.selectbox("Water model", ["WT", "ru", "WT+ru"],
                                  index=["WT","ru","WT+ru"].index(st.session_state["water_mode"]))
        st.session_state["water_mode"] = water_mode
        ru_val = st.slider("r_u (if ru mode)", 0.0, 0.9, st.session_state.get("ru", 0.0), 0.05)
        st.session_state["ru"] = ru_val

        # オフセットWT（地表にクリップ）
        offset = st.slider("Water level offset from ground (m, negative=below)", -30.0, 5.0, -2.0, 0.5)
        Xd = np.linspace(ground.X[0], ground.X[-1], 200)
        Yg = np.array([float(ground.y_at(x)) for x in Xd])
        Yw_raw = Yg + offset
        Yw = np.minimum(Yw_raw, Yg)  # ★地表越え防止
        st.session_state["wl_points"] = np.vstack([Xd, Yw]).T

        # プロット
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Soil")
        if water_mode.startswith("WT"):
            ax.plot(Xd, Yw, linestyle="-.", color="tab:blue", label="WT (offset, clipped)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        st.pyplot(fig); plt.close(fig)

    with colR:
        st.subheader("教授コメント")
        suggs = lint_geometry_and_water(ground, st.session_state["water_mode"], st.session_state.get("ru",0.0), st.session_state.get("wl_points"))
        render_suggestions(suggs, key_prefix="p1", dispatcher=default_dispatcher)

# ---------------- Page 2: 地層・材料 ----------------
elif page.startswith("2"):
    st.subheader("Layers & Materials")

    # Page1 と同じキーで H/L を共有（ここでも変更可）
    H = st.number_input("H (m)", 5.0, 200.0, st.session_state["H"], 0.5, key="H")
    L = st.number_input("L (m)", 5.0, 400.0, st.session_state["L"], 0.5, key="L")
    ground = make_ground_example(st.session_state["H"], st.session_state["L"])

    n_layers = st.selectbox("Number of layers", [1,2,3], index=2)
    interfaces = []
    if n_layers >= 2: interfaces.append(make_interface1_example(H, L))
    if n_layers >= 3: interfaces.append(make_interface2_example(H, L))

    cols = st.columns(4)
    soils: list[Soil] = []
    with cols[0]:
        st.markdown("**Layer1 (top)**")
        g1 = st.number_input("γ₁", 10.0, 25.0, 18.0, 0.5)
        c1 = st.number_input("c₁", 0.0, 200.0, 5.0, 0.5)
        p1 = st.number_input("φ₁", 0.0, 45.0, 30.0, 0.5)
        t1 = st.number_input("τ₁ (kPa)", 0.0, 1000.0, 150.0, 10.0)
        soils.append(Soil(g1, c1, p1, t1))
    if n_layers >= 2:
        with cols[1]:
            st.markdown("**Layer2**")
            g2 = st.number_input("γ₂", 10.0, 25.0, 19.0, 0.5)
            c2 = st.number_input("c₂", 0.0, 200.0, 8.0, 0.5)
            p2 = st.number_input("φ₂", 0.0, 45.0, 28.0, 0.5)
            t2 = st.number_input("τ₂ (kPa)", 0.0, 1000.0, 180.0, 10.0)
            soils.append(Soil(g2, c2, p2, t2))
    if n_layers >= 3:
        with cols[2]:
            st.markdown("**Layer3 (bottom)**")
            g3 = st.number_input("γ₃", 10.0, 25.0, 20.0, 0.5)
            c3 = st.number_input("c₃", 0.0, 200.0, 12.0, 0.5)
            p3 = st.number_input("φ₃", 0.0, 45.0, 25.0, 0.5)
            t3 = st.number_input("τ₃ (kPa)", 0.0, 1000.0, 200.0, 10.0)
            soils.append(Soil(g3, c3, p3, t3))
    with cols[-1]:
        st.markdown("**Grout / Nail**")
        tau_grout_cap = st.number_input("τ_grout_cap (kPa)", 0.0, 2000.0, float(st.session_state.get("tau_grout_cap_kPa") or 150.0), 10.0)
        st.session_state["tau_grout_cap_kPa"] = tau_grout_cap
        d_g = st.number_input("削孔(=グラウト)径 d_g (m)", 0.05, 0.30, 0.125, 0.005)
        d_s = st.number_input("鉄筋径 d_s (m)", 0.010, 0.050, 0.022, 0.001)
        fy  = st.number_input("引張強さ fy (MPa=kN/m²)", 200.0, 2000.0, 1000.0, 50.0)
        gamma_m = st.number_input("材料安全率 γ_m", 1.00, 2.00, 1.20, 0.05)
        mu = st.select_slider("逓減係数 μ（0〜0.9, 0.1刻み。μ=1.0はStrip無視）",
                              options=[round(0.1*i,1) for i in range(10)], value=float(st.session_state.get("mu",0.0)))
        st.session_state["mu"] = mu

    # 可視化（層線）
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    if n_layers==1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif n_layers==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    if n_layers>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.0)
    if n_layers>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.0)
    ax.set_aspect("equal"); ax.grid(True); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    # 教授コメント
    st.sidebar.markdown("### 教授コメント")
    soils_table = [dict(gamma=s.gamma, c=s.c, phi=s.phi, tau_kPa=getattr(s, "tau_kPa", 0.0)) for s in soils]
    render_suggestions(lint_soils_and_materials(soils_table, st.session_state.get("tau_grout_cap_kPa")), key_prefix="p2", dispatcher=default_dispatcher)

    # 保存（Page3以降で使用）
    st.session_state["ground_pack"] = dict(H=H,L=L,n_layers=n_layers,interfaces=interfaces,soils=soils)

# ---------------- Page 3: 円弧探索（未補強） ----------------
elif page.startswith("3"):
    st.subheader("円弧探索（未補強）")
    if "ground_pack" not in st.session_state:
        st.info("先に Page2 で地層・材料を設定してください。"); st.stop()
    gp = st.session_state["ground_pack"]
    H,L,n_layers,interfaces,soils = gp["H"],gp["L"],gp["n_layers"],gp["interfaces"],gp["soils"]
    ground = make_ground_example(H, L)

    # Center grid & Quality
    colA, colB = st.columns([1.3, 1])
    with colA:
        x_min = st.number_input("x min", 0.20*L, 3.00*L, 0.25*L, 0.05*L)
        x_max = st.number_input("x max", 0.30*L, 4.00*L, 1.15*L, 0.05*L)
        y_min = st.number_input("y min", 0.80*H, 7.00*H, 1.60*H, 0.10*H)
        y_max = st.number_input("y max", 1.00*H, 8.00*H, 2.20*H, 0.10*H)
    with colB:
        nx = st.slider("nx", 6, 60, 14); ny = st.slider("ny", 4, 40, 9)
        method = st.selectbox("Method", ["Bishop (simplified)","Fellenius"])
        quality = st.select_slider("Quality", options=list(QUALITY.keys()), value="Normal")
        P = QUALITY[quality].copy()
        Fs_target = st.number_input("Target FS", 1.00, 2.00, 1.20, 0.05)

    allow_cross=[]
    if n_layers>=2: allow_cross.append(st.checkbox("Allow into Layer 2", True))
    if n_layers>=3: allow_cross.append(st.checkbox("Allow into Layer 3", True))

    # ---- 設定プレビュー：横断図＋水位＋地層＋センターグリッド（設定中でも見える） ----
    st.markdown("**設定プレビュー（横断図）**")
    Xd = np.linspace(ground.X[0], ground.X[-1], 600)
    Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 6.8))
    # 層塗り
    if n_layers==1:
        ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
    elif n_layers==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
        ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
        ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
    # 地表線
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    if n_layers>=2: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0], linestyle="--", linewidth=1.0)
    if n_layers>=3: ax.plot(Xd, clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)[1], linestyle="--", linewidth=1.0)
    # 水位（Page1で設定されていれば表示）
    if st.session_state.get("water_mode","WT").startswith("WT") and st.session_state.get("wl_points") is not None:
        wl = st.session_state["wl_points"]
        ax.plot(wl[:,0], wl[:,1], linestyle="-.", color="tab:blue", alpha=0.8, label="WT (clipped)")
    # センターグリッドの点群（設定値でプレビュー）
    grid_xs = np.linspace(x_min, x_max, nx)
    grid_ys = np.linspace(y_min, y_max, ny)
    gx = [float(x) for x in grid_xs for _ in grid_ys]
    gy = [float(y) for y in grid_ys for _ in grid_xs]
    ax.scatter(gx, gy, s=10, c="k", alpha=0.25, marker=".", label="Center grid (preview)")
    # 外周枠
    ax.plot([x_min,x_max,x_max,x_min,x_min], [y_min,y_min,y_max,y_max,y_min], color="k", linewidth=1.0, alpha=0.4)
    # 軸・凡例
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True); ax.legend(loc="upper right"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    st.pyplot(fig); plt.close(fig)

    # ---- ここから計算本体 ----
    def compute_once():
        def subsampled_centers():
            xs = np.linspace(x_min, x_max, nx)
            ys = np.linspace(y_min, y_max, ny)
            tag = P["coarse_subsample"]
            if tag == "every 3rd":
                xs = xs[::3] if len(xs)>2 else xs; ys = ys[::3] if len(ys)>2 else ys
            elif tag == "every 2nd":
                xs = xs[::2] if len(xs)>1 else xs; ys = ys[::2] if len(ys)>1 else ys
            return [(float(xc), float(yc)) for yc in ys for xc in xs]

        def pick_center(budget_s):
            deadline = time.time() + budget_s
            best = None; tested=[]
            for (xc,yc) in subsampled_centers():
                cnt=0; Fs_min=None
                for _x1,_x2,_R,Fs in arcs_from_center_by_entries_multi(
                    ground, gp["soils"], xc, yc,
                    n_entries=P["coarse_entries"], method="Fellenius",
                    depth_min=0.5, depth_max=4.0,
                    interfaces=interfaces, allow_cross=allow_cross,
                    quick_mode=True, n_slices_quick=max(8, P["quick_slices"]//2),
                    limit_arcs_per_center=P["coarse_limit_arcs"],
                    probe_n_min=P["coarse_probe_min"],
                ):
                    cnt+=1
                    if (Fs_min is None) or (Fs < Fs_min): Fs_min = Fs
                    if time.time() > deadline: break
                tested.append((xc,yc))
                score = (cnt, - (Fs_min if Fs_min is not None else 1e9))
                if (best is None) or (score > best[0]): best = (score, (xc,yc))
                if time.time() > deadline: break
            return (best[1] if best else None), tested

        center, tested = pick_center(P["budget_coarse_s"])
        if center is None:
            return dict(error="Coarseで候補なし。枠/深さを広げてください。")
        xc, yc = center
        heap_R=[]; deadline=time.time()+P["budget_quick_s"]
        for _x1,_x2,R,Fs in arcs_from_center_by_entries_multi(
            ground, gp["soils"], xc, yc,
            n_entries=P["n_entries_final"], method="Fellenius",
            depth_min=0.5, depth_max=4.0,
            interfaces=interfaces, allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=P["quick_slices"],
            limit_arcs_per_center=P["limit_arcs_quick"],
            probe_n_min=P["probe_n_min_quick"],
        ):
            heapq.heappush(heap_R, (-Fs, R))
            if len(heap_R) > max(P["show_k"], P["top_thick"] + 20): heapq.heappop(heap_R)
            if time.time() > deadline: break
        R_candidates = [r for _fsneg, r in sorted([(-fsneg,R) for fsneg,R in heap_R], key=lambda t:t[0])]
        if not R_candidates:
            return dict(error="Quickで円弧候補なし。深さ/進入可/Qualityを緩めてください。")

        refined=[]
        for R in R_candidates[:P["show_k"]]:
            Fs = fs_given_R_multi(ground, interfaces, gp["soils"], allow_cross, method, xc, yc, R, n_slices=P["final_slices"])
            if Fs is None: continue
            s = arc_sample_poly_best_pair(ground, xc, yc, R, n=251, y_floor=0.0)
            if s is None: continue
            x1,x2,*_ = s
            packD = driving_sum_for_R_multi(ground, interfaces, gp["soils"], allow_cross, xc, yc, R, n_slices=P["final_slices"])
            if packD is None: continue
            D_sum,_,_ = packD
            T_req = max(0.0, (Fs_target - Fs)*D_sum)
            refined.append(dict(Fs=float(Fs), R=float(R), x1=float(x1), x2=float(x2), T_req=float(T_req)))
        if not refined:
            return dict(error="Refineで有効弧なし。設定/Qualityを見直してください。")
        refined.sort(key=lambda d:d["Fs"])
        idx_minFs = int(np.argmin([d["Fs"] for d in refined]))
        return dict(center=center, refined=refined, idx_minFs=idx_minFs)

    if st.button("▶ 計算開始"):
        res = compute_once()
        if "error" in res: st.error(res["error"]); st.stop()
        st.session_state["res3"] = res

    if "res3" in st.session_state:
        res = st.session_state["res3"]
        xc,yc = res["center"]; refined=res["refined"]; idx_minFs=res["idx_minFs"]
        # Plot（結果）
        Xd = np.linspace(ground.X[0], ground.X[-1], 600); Yg = np.array([float(ground.y_at(float(x))) for x in Xd], dtype=float)
        fig, ax = plt.subplots(figsize=(10.0, 7.0))
        if n_layers==1:
            ax.fill_between(Xd, 0.0, Yg, alpha=0.12, label="Layer1")
        elif n_layers==2:
            Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xd)[0]
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, 0.0, Y1, alpha=0.12, label="Layer2")
        else:
            Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xd)
            ax.fill_between(Xd, Y1, Yg, alpha=0.12, label="Layer1")
            ax.fill_between(Xd, Y2, Y1, alpha=0.12, label="Layer2")
            ax.fill_between(Xd, 0.0, Y2, alpha=0.12, label="Layer3")
        ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
        for d in refined[:30]:
            xs=np.linspace(d["x1"], d["x2"], 200)
            ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
            clipped = clip_yfloor(xs, ys, 0.0)
            if clipped is None: continue
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=0.9, alpha=0.75, color=fs_to_color(d["Fs"]))
        # pick min Fs
        d=refined[idx_minFs]
        xs=np.linspace(d["x1"], d["x2"], 400)
        ys=yc - np.sqrt(np.maximum(0.0, d["R"]**2 - (xs - xc)**2))
        clipped = clip_yfloor(xs, ys, 0.0)
        if clipped is not None:
            xs_c, ys_c = clipped
            ax.plot(xs_c, ys_c, linewidth=3.0, color=(0.9,0.0,0.0), label=f"Min Fs = {d['Fs']:.3f}")
            y1=float(ground.y_at(xs_c[0])); y2=float(ground.y_at(xs_c[-1]))
            ax.plot([xc,xs_c[0]],[yc,y1], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
            ax.plot([xc,xs_c[-1]],[yc,y2], linewidth=1.1, color=(0.9,0.0,0.0), alpha=0.9)
        ax.set_aspect("equal"); ax.grid(True); ax.legend()
        ax.set_title(f"Center=({xc:.2f},{yc:.2f}) • MinFs={refined[idx_minFs]['Fs']:.3f} • TargetFs={Fs_target:.2f}")
        st.pyplot(fig); plt.close(fig)

        # 教授コメント（端ヒット検知は簡略：ここでは常時 False）
        render_suggestions(lint_arc_and_slices(hit_edge=False, n_slices=P["final_slices"]), key_prefix="p3", dispatcher=default_dispatcher)

        # 採用円弧の保存（Page4で使う）
        st.session_state["chosen_arc"] = dict(xc=xc,yc=yc,R=refined[idx_minFs]["R"], x1=refined[idx_minFs]["x1"], x2=refined[idx_minFs]["x2"])

# ---------------- Page 4: ネイル配置（可視化＋教授コメント） ----------------
elif page.startswith("4"):
    st.subheader("ソイルネイル配置（試作段階：可視化＋教授コメント）")
    if "ground_pack" not in st.session_state or "chosen_arc" not in st.session_state:
        st.info("Page3で補強対象の円弧を確定させてから来てください。"); st.stop()
    gp = st.session_state["ground_pack"]
    H,L,n_layers,interfaces,soils = gp["H"],gp["L"],gp["n_layers"],gp["interfaces"],gp["soils"]
    ground = make_ground_example(H, L)
    arc = st.session_state["chosen_arc"]

    # 斜面実長ベースの等間隔配置（1段）
    st.markdown("**配置範囲（斜面累積長 s）**")
    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seglen = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s_total = float(s_cum[-1])
    s_start = st.slider("s_start (m)", 0.0, s_total, 5.0, 0.5)
    s_end   = st.slider("s_end (m)", s_start, s_total, min(s_start+30.0, s_total), 0.5)
    S_surf  = st.slider("斜面ピッチ S_surf (m)", 0.5, 5.0, 2.0, 0.1)
    S_row   = st.slider("段間隔 S_row (法線方向 m) [未実装]", 0.5, 5.0, 2.0, 0.5)
    tiers   = st.number_input("段数 [表示のみ]", 1, 5, 1, 1)

    angle_mode = st.radio("角度モード", ["Slope-Normal (⊥斜面)", "Horizontal-Down (β°)"], index=0)
    if angle_mode.endswith("β°"):
        beta_deg = st.slider("β（水平から下向き °）", 0.0, 45.0, 15.0, 1.0)
    else:
        delta_beta = st.slider("法線からの微調整 ±Δβ（°）", -10.0, 10.0, 0.0, 1.0)

    L_mode = st.radio("長さモード", ["パターン1：固定長", "パターン2：すべり面より +Δm", "パターン3：FS目標で自動"], index=0)
    if L_mode == "パターン1：固定長":
        L_nail = st.slider("ネイル長 L (m)", 1.0, 15.0, 5.0, 0.5)
    elif L_mode == "パターン2：すべり面より +Δm":
        d_embed = st.slider("すべり面より +Δm (m)", 0.0, 5.0, 1.0, 0.5)
        L_nail = None
    else:
        st.info("Phase-2 で実装（FS目標に合わせて最小長を探索）")
        L_nail = None

    # s 等間隔の x を取る関数
    def x_at_s(s_val: float) -> float:
        idx = np.searchsorted(s_cum, s_val, side="right")-1
        idx = max(0, min(idx, len(Xd)-2))
        ds = s_val - s_cum[idx]
        segS = seglen[idx] if seglen[idx]>1e-12 else 1e-12
        t = ds/segS
        return float((1-t)*Xd[idx] + t*Xd[idx+1])

    # 1段のみ生成（Phase-1）
    s_vals = list(np.arange(s_start, s_end+1e-9, S_surf))
    pts = [(x_at_s(sv), float(ground.y_at(x_at_s(sv)))) for sv in s_vals]

    # 可視化（円弧＋ネイルの頭のみ）
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.plot(ground.X, ground.Y, linewidth=2.0, label="Ground")
    Xp = np.linspace(ground.X[0], ground.X[-1], 600); Yp = np.array([float(ground.y_at(x)) for x in Xp])
    if n_layers==1:
        ax.fill_between(Xp, 0.0, Yp, alpha=0.12, label="Layer1")
    elif n_layers==2:
        Y1 = clip_interfaces_to_ground(ground, [interfaces[0]], Xp)[0]
        ax.fill_between(Xp, Y1, Yp, alpha=0.12, label="Layer1"); ax.fill_between(Xp, 0.0, Y1, alpha=0.12, label="Layer2")
    else:
        Y1,Y2 = clip_interfaces_to_ground(ground, [interfaces[0],interfaces[1]], Xp)
        ax.fill_between(Xp, Y1, Yp, alpha=0.12, label="Layer1")
        ax.fill_between(Xp, Y2, Y1, alpha=0.12, label="Layer2")
        ax.fill_between(Xp, 0.0, Y2, alpha=0.12, label="Layer3")
    # 円弧（採用）
    xc,yc,R = arc["xc"],arc["yc"],arc["R"]
    xs=np.linspace(arc["x1"], arc["x2"], 400)
    ys=yc - np.sqrt(np.maximum(0.0, R**2 - (xs - xc)**2))
    ax.plot(xs, ys, linewidth=2.5, color="tab:red", label="Chosen slip arc")

    # ネイル頭の表示
    xs_h = [p[0] for p in pts]; ys_h=[p[1] for p in pts]
    ax.scatter(xs_h, ys_h, s=30, color="tab:blue", label=f"Nail heads ({len(pts)})")
    ax.set_aspect("equal"); ax.grid(True); ax.legend()
    ax.set_title("Phase-1：配置プレビュー（頭位置のみ）")
    st.pyplot(fig); plt.close(fig)

    # 教授コメント（ダミー統計でレンダ）
    stats = dict(has_outward=False, too_dense=False, n_Lo_short=0, dominant_mode="pullout")
    render_suggestions(lint_nails_layout(dict(S_surf=S_surf,S_row=S_row,mode=angle_mode,Lo_min=1.0), stats), key_prefix="p4", dispatcher=default_dispatcher)

    st.info("Phase-2 で：交点s*の算定・Lo/ Li 積分・Tpullout/Tstrip/Ttens の最小化・Tt/Tn投影・分配まで連成して、FSを更新します。")

# ---------------- Page 5: 補強後解析（プレースホルダ） ----------------
elif page.startswith("5"):
    st.subheader("補強後解析（Phase-2 でFS連成を実装）")
    st.write("ここでは、補強前後の比較表・図、サジェストカード（最小コスト案/最短L案/余裕10%案）を表示する予定です。")
    st.success("Phase-1 は「UIと教授コメント」を完成させる到達点です。補強FSの厳密連成は次段で差し込みます。")

# ---------------- 共通：教授コメントのアクション反映ハブ ----------------
if st.session_state.get("_recompute", False):
    st.session_state["_recompute"] = False
    if st.session_state.get("_cmd_pitch_delta") is not None:
        st.session_state["_cmd_pitch_delta"] = None
    if st.session_state.get("_cmd_length_delta") is not None:
        st.session_state["_cmd_length_delta"] = None
    if st.session_state.get("_cmd_scale_xy") is not None:
        # （本番は ground.X/Y を更新する処理をここへ）
        st.session_state["_cmd_scale_xy"] = None
    if st.session_state.get("_cmd_wl_clip"):
        # 既にPage1で地表クリップしているためここではフラグを落とすだけ
        st.session_state["_cmd_wl_clip"] = None
    if st.session_state.get("_cmd_expand_grid") is not None:
        st.session_state["_cmd_expand_grid"] = None
    if st.session_state.get("_cmd_bar_next"):
        st.session_state["_cmd_bar_next"] = None
    if st.session_state.get("_cmd_mu_down"):
        st.session_state["mu"] = max(0.0, float(st.session_state["mu"]) - 0.1)
        st.session_state["_cmd_mu_down"] = None
