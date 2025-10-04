# streamlit_app.py — ソイルネイル統合・最小円弧オート保存版（フル）

from __future__ import annotations

# --- パッケージ解決（stabi を親ディレクトリから見つける） ---
import os, sys
_PKG_DIR = os.path.dirname(__file__)          # .../stabi
_PARENT  = os.path.dirname(_PKG_DIR)          # .../
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ===== 標準/外部ライブラリ =====
import math, time, heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ===== stabi コアは“丸ごと”読み込んで属性参照に統一 =====
import stabi.stabi_lem as lem

# 以降の既存コードで使っている名前に合わせてエイリアスを定義
Soil                      = lem.Soil
GroundPL                  = lem.GroundPL
make_ground_example       = lem.make_ground_example
make_interface1_example   = lem.make_interface1_example
make_interface2_example   = lem.make_interface2_example
clip_interfaces_to_ground = lem.clip_interfaces_to_ground
arcs_from_center_by_entries_multi = lem.arcs_from_center_by_entries_multi
fs_given_R_multi          = lem.fs_given_R_multi
arc_sample_poly_best_pair = lem.arc_sample_poly_best_pair
driving_sum_for_R_multi   = lem.driving_sum_for_R_multi


# ===== Streamlit 基本設定 =====
st.set_page_config(page_title="Stabi LEM (Nails integrated)", layout="wide")
st.title("Stabi LEM｜ソイルネイル統合デモ（最小円弧オート保存）")

# ---------------------------------------------------------------
# 共通 cfg（正本）
# ---------------------------------------------------------------
def default_cfg():
    return {
        "geom": {"H": 25.0, "L": 60.0},
        "water": {"mode": "WT", "offset": -2.0, "wl_points": None},
        "layers": {
            "n": 3,
            "mat": {
                1: {"gamma": 18.0, "c": 5.0,  "phi": 30.0},
                2: {"gamma": 19.0, "c": 8.0,  "phi": 28.0},
                3: {"gamma": 20.0, "c": 12.0, "phi": 25.0},
            },
            "tau_grout_cap_kPa": 150.0,
            "d_g": 0.125,  # m（削孔=グラウト径）
            "d_s": 0.022,  # m（鉄筋径）
            "fy": 1000.0,  # MPa
            "gamma_m": 1.20,
        },
        "grid": {
            "method": "Bishop (simplified)",
            "Fs_target": 1.20,
        },
        "nails": {
            "s_start": 5.0, "s_end": 35.0,
            "S_surf": 2.0, "S_row": 2.0,
            "tiers": 1,
            "angle_mode": "Slope-Normal (⊥斜面)",
            "beta_deg": 15.0, "delta_beta": 0.0,
            "L_mode": "パターン1：固定長", "L_nail": 5.0, "d_embed": 1.0,
        },
        "results": {
            "unreinforced": None,   # Page3: 参考情報（任意）
            "chosen_arc": None,     # Page3で確定→Page5で使用
            "nail_heads": [],       # Page4で保存
            "reinforced": None,     # Page5の結果
        }
    }

def cfg_get(path, default=None):
    node = st.session_state["cfg"]
    for p in path.split("."):
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return default
    return node

def cfg_set(path, value):
    node = st.session_state["cfg"]
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value

if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

def make_ground_from_cfg():
    H = float(cfg_get("geom.H")); L = float(cfg_get("geom.L"))
    return H, L, make_ground_example(H, L)

# ---------------------------------------------------------------
# Soil Nail ヘルパ（UI側）
# ---------------------------------------------------------------
EPS = 1e-12

def _axis_unit(azimuth_rad: float):
    return math.cos(azimuth_rad), math.sin(azimuth_rad)

def _ray_circle_intersection(head, azimuth, C, R):
    (x0,y0) = head; (cx,cy) = C
    ux, uy = _axis_unit(azimuth)
    dx, dy = (x0 - cx), (y0 - cy)
    A = 1.0
    B = 2.0*(dx*ux + dy*uy)
    Cq = dx*dx + dy*dy - R*R
    D = B*B - 4*A*Cq
    if D < 0: return None
    sD = math.sqrt(max(0.0, D))
    t1 = (-B - sD)/(2*A)
    t2 = (-B + sD)/(2*A)
    cand = [t for t in (t1, t2) if t >= -1e-10]
    if not cand: return None
    t = min(cand)
    xi, yi = x0 + t*ux, y0 + t*uy
    return (xi, yi, float(t))

def _circle_tangent_unit_at(P, C):
    (px,py),(cx,cy) = P,C
    rx, ry = (px-cx), (py-cy)
    rn = math.hypot(rx, ry)
    if rn < EPS: return (1.0,0.0)
    return (-ry/rn, rx/rn)

def _segment_circle_intersection(P0, P1, C, R):
    (x0,y0),(x1,y1) = P0,P1; (cx,cy) = C
    dx, dy = (x1-x0), (y1-y0)
    A = dx*dx + dy*dy
    if A < EPS: return None
    fx, fy = (x0-cx), (y0-cy)
    B = 2*(fx*dx + fy*dy)
    Cq = fx*fx + fy*fy - R*R
    D = B*B - 4*A*Cq
    if D < 0: return None
    sD = math.sqrt(max(0.0, D))
    t1 = (-B - sD)/(2*A)
    t2 = (-B + sD)/(2*A)
    cand = [t for t in (t1,t2) if -1e-10 <= t <= 1+1e-10]
    if not cand: return None
    t = min(cand)
    xi, yi = x0 + t*dx, y0 + t*dy
    return (xi, yi, float(t))

def build_nails_from_cfg(ground: GroundPL, nail_heads, cfg_nails, arc):
    xc, yc, R = arc["xc"], arc["yc"], arc["R"]
    nails = []
    for (xh, yh) in nail_heads:
        # 斜面法線（近傍差分）
        x0 = max(min(xh-0.05, ground.X[-1]), ground.X[0])
        x1 = min(max(xh+0.05, ground.X[0]), ground.X[-1])
        y0 = float(ground.y_at(x0)); y1 = float(ground.y_at(x1))
        slope = math.atan2((y1-y0),(x1-x0)+EPS)
        normal = slope + math.pi/2

        if cfg_nails["angle_mode"].startswith("Slope-Normal"):
            azimuth = normal + math.radians(cfg_nails.get("delta_beta", 0.0))
        else:
            azimuth = -math.radians(abs(cfg_nails.get("beta_deg", 15.0)))  # 水平から下向きを正

        # すべり面との交点距離
        hit = _ray_circle_intersection((xh,yh), azimuth, (xc,yc), R)
        dist_to_arc = hit[2] if hit else 0.0

        # 長さ
        if cfg_nails["L_mode"].startswith("パターン2"):
            L = max(0.5, dist_to_arc + float(cfg_nails.get("d_embed", 1.0)))
        else:
            L = float(cfg_nails.get("L_nail", 5.0))

        nails.append({"x": float(xh), "y": float(yh), "azimuth": float(azimuth), "length": float(L)})
    return nails

def nails_R_sum_kN_per_m(nails, xc, yc, R, mat, x_min=None, x_max=None):
    tau_kN_m2 = float(mat["tau_grout_cap_kPa"])
    fy_kN_m2  = float(mat["fy"]) * 1e3
    d_g = float(mat["d_g"]); d_s = float(mat["d_s"])
    A_s = math.pi*(d_s**2)/4.0
    eta = float(mat.get("eta_mob", 0.9))
    gamma_m = float(mat.get("gamma_m", 1.2))

    R_sum = 0.0; logs = []; C=(xc,yc)
    for i, n in enumerate(nails):
        xh,yh = n["x"], n["y"]
        ux,uy = _axis_unit(n["azimuth"])
        tip = (xh + ux*n["length"], yh + uy*n["length"])
        hit_seg = _segment_circle_intersection((xh,yh), tip, C, R)
        if not hit_seg:
            logs.append({"id": i, "ok": False, "reason": "no_intersection"}); continue
        xi, yi, t = hit_seg
        if (x_min is not None and xi < x_min-1e-9) or (x_max is not None and xi > x_max+1e-9):
            logs.append({"id": i, "ok": False, "reason": "outside_arc"}); continue
        embed = max(0.0, n["length"]*(1.0 - t))
        T_y_kN = (A_s * fy_kN_m2) / max(gamma_m, 1e-6)
        T_b_kN = (math.pi * d_g * embed * tau_kN_m2) / max(gamma_m, 1e-6)
        T_cap  = min(T_y_kN, T_b_kN)
        T_kN   = eta * T_cap
        tx,ty = _circle_tangent_unit_at((xi,yi), C)
        dot_t = ux*tx + uy*ty
        Rk = max(0.0, T_kN * dot_t)
        R_sum += Rk
        logs.append({"id": i, "ok": True, "x":xi, "y":yi, "t":t, "embed":embed,
                     "T_kN":T_kN, "dot_t":dot_t, "Rk":Rk,
                     "head_x":xh, "head_y":yh, "tip_x":tip[0], "tip_y":tip[1]})
    return R_sum, logs

def integrate_nails_for_best_circle(
    ground, interfaces, soils, allow_cross,
    xc, yc, R, n_slices,
    Fs_before: float,
    nails: list,
    mat: dict,
):
    out = {"Fs_before": Fs_before, "Fs_after": Fs_before, "R_sum": 0.0,
           "D_sum": None, "logs": [], "x_min": None, "x_max": None}
    if Fs_before is None:
        return out
    pack = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices)
    if pack is None:
        return out
    D_sum, x_min, x_max = pack
    out["D_sum"] = D_sum; out["x_min"] = x_min; out["x_max"] = x_max
    if not nails:
        return out
    R_sum, logs = nails_R_sum_kN_per_m(nails, xc, yc, R, mat, x_min=x_min, x_max=x_max)
    out["R_sum"] = R_sum; out["logs"] = logs
    if D_sum and D_sum > 0:
        out["Fs_after"] = float(Fs_before) + float(R_sum)/float(D_sum)
    return out

def draw_cross_section_with_nails(
    ground, xc, yc, R,
    x_min, x_max,
    nails_logs: list,
    Fs_before: float, Fs_after: float,
    R_sum: float, D_sum: float,
    nails_raw: list = None,
    title: str = "Cross-section (Nails × Slip Circle)",
):
    xs = np.linspace(x_min if x_min is not None else (xc-R),
                     x_max if x_max is not None else (xc+R), 401)
    inside = R*R - (xs - xc)**2
    m = inside > 0
    xs = xs[m]
    y_arc = yc - np.sqrt(inside[m])
    y_g   = np.array([ground.y_at(x) for x in xs])

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(ground.X, ground.Y, '-', lw=2.0, label='Ground')
    ax.plot(xs, y_arc, '--', lw=2.0, label='Slip circle')
    ax.fill_between(xs, y_arc, y_g, where=(y_g>=y_arc), alpha=0.15, label='Sliding mass')

    if nails_logs:
        if nails_raw and len(nails_raw) >= len(nails_logs):
            for log, n in zip(nails_logs, nails_raw):
                if not log.get('ok'): continue
                xh, yh = n['x'], n['y']
                xt, yt = log["tip_x"], log["tip_y"]
                xi, yi = log['x'], log['y']
                ax.plot([xh, xi], [yh, yi], '-', lw=2.0, color='tab:red')   # Active
                ax.plot([xi, xt], [yi, yt], '-', lw=2.0, color='tab:blue')  # Passive
                size = max(24, min(140, 10 + 1.0*log.get('T_kN', 0.0)))
                ax.scatter([xi], [yi], s=size, zorder=3)
        else:
            for log in nails_logs:
                if not log.get('ok'): continue
                xi, yi = log['x'], log['y']
                size = max(24, min(140, 10 + 1.0*log.get('T_kN', 0.0)))
                ax.scatter([xi], [yi], s=size, zorder=3)

    if (Fs_before is not None) and (Fs_after is not None):
        delta = (Fs_after - Fs_before)
        txt = f"Fs_before = {Fs_before:.3f}\nFs_after  = {Fs_after:.3f}\nΔFs = {delta:.3f}"
        if (R_sum is not None) and (D_sum is not None) and D_sum>0:
            txt += f"\nΣRk = {R_sum:.3f} kN/m\nD = {D_sum:.3f} kN/m"
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75), fontsize=10)

    ax.set_title(title); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    return fig, ax

# ---------------------------------------------------------------
# Sidebar ナビ
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Pages")
    page = st.radio("", ["1) 地形・水位", "2) 地層・材料", "3) 円弧探索（未補強・自動保存）", "4) ネイル配置", "5) 補強後解析"])
    if st.button("⚠ cfg初期化"):
        st.session_state["cfg"] = default_cfg()
        st.success("初期化しました。")

# ---------------------------------------------------------------
# Page1: 地形・水位（簡易プレビュー）
# ---------------------------------------------------------------
if page.startswith("1"):
    H,L,ground = make_ground_from_cfg()
    fig,ax = plt.subplots(figsize=(9,5.4))
    Xd = np.linspace(0, L, 500)
    Yg = np.array([ground.y_at(x) for x in Xd])
    ax.plot(Xd, Yg, lw=2.0, label="Ground")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Ground preview")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------
# Page2: 地層・材料
# ---------------------------------------------------------------
elif page.startswith("2"):
    st.subheader("Materials")
    m = cfg_get("layers.mat")
    cols = st.columns(3)
    for i,c in enumerate(cols, start=1):
        with c:
            st.markdown(f"**Layer{i}**")
            gamma = st.number_input(f"γ{i} (kN/m³)", 10.0, 25.0, step=0.5, value=float(m[i]["gamma"]), key=f"gamma{i}")
            coh   = st.number_input(f"c{i} (kPa)", 0.0, 200.0, step=0.5, value=float(m[i]["c"]), key=f"c{i}")
            phi   = st.number_input(f"φ{i} (deg)", 0.0, 45.0, step=0.5, value=float(m[i]["phi"]), key=f"phi{i}")
            m[i] = {"gamma": gamma, "c": coh, "phi": phi}
    tau_cap = st.number_input("τ_grout_cap (kPa)", 10.0, 5000.0, step=10.0, value=float(cfg_get("layers.tau_grout_cap_kPa")))
    d_g     = st.number_input("削孔径 d_g (m)", 0.05, 0.30, step=0.005, value=float(cfg_get("layers.d_g")))
    d_s     = st.number_input("鉄筋径 d_s (m)", 0.010, 0.050, step=0.001, value=float(cfg_get("layers.d_s")))
    fy      = st.number_input("fy (MPa)", 200.0, 2000.0, step=50.0, value=float(cfg_get("layers.fy")))
    gamma_m = st.number_input("γ_m", 1.00, 2.00, step=0.05, value=float(cfg_get("layers.gamma_m")))
    if st.button("💾 材料保存"):
        cfg_set("layers.mat", m)
        cfg_set("layers.tau_grout_cap_kPa", tau_cap)
        cfg_set("layers.d_g", d_g); cfg_set("layers.d_s", d_s)
        cfg_set("layers.fy", fy); cfg_set("layers.gamma_m", gamma_m)
        st.success("保存しました。")

# ---------------------------------------------------------------
# Page3: 円弧探索（未補強・自動保存）
# ---------------------------------------------------------------
elif page.startswith("3"):
    st.subheader("円弧探索（未補強）→ 最小Fsを自動保存")
    H,L,ground = make_ground_from_cfg()

    # シンプル：代表中心（例）を数点スキャンして最小Fsを採用（デモ用）
    centers = [
        (0.45*L, 1.40*H),
        (0.55*L, 1.35*H),
        (0.60*L, 1.50*H),
    ]

    # soils（3層定義を簡易に扱う）
    m = cfg_get("layers.mat")
    soils = [Soil(m[1]["gamma"], m[1]["c"], m[1]["phi"])]
    soils.append(Soil(m[2]["gamma"], m[2]["c"], m[2]["phi"]))
    soils.append(Soil(m[3]["gamma"], m[3]["c"], m[3]["phi"]))
    allow_cross = [True, True]

    best = None
    for (xc,yc) in centers:
        cand = []
        for _x1,_x2,R,Fs_quick in arcs_from_center_by_entries_multi(
            ground, soils, xc, yc,
            n_entries=60, method="Fellenius",
            depth_min=0.8, depth_max=5.0,
            interfaces=[], allow_cross=allow_cross,
            quick_mode=True, n_slices_quick=10,
            limit_arcs_per_center=50, probe_n_min=61,
        ):
            Fs = fs_given_R_multi(ground, [], soils, allow_cross, "Bishop", xc, yc, R, n_slices=40)
            if Fs is None: continue
            s = arc_sample_poly_best_pair(ground, xc, yc, R, n=241, y_floor=0.0)
            if s is None: continue
            x1, x2, _R = s
            cand.append((Fs, xc, yc, _R, x1, x2))
        if cand:
            cmin = min(cand, key=lambda t: t[0])
            if (best is None) or (cmin[0] < best[0]):
                best = cmin

    if best is None:
        st.error("候補が見つかりませんでした。H/Lや centers を見直してください。")
        st.stop()

    Fs_min, xc, yc, R, x1, x2 = best
    # 🔵 ここで自動保存（Page5が直ちに使える）
    cfg_set("results.chosen_arc", {"xc": xc, "yc": yc, "R": R, "x1": x1, "x2": x2, "Fs": float(Fs_min)})
    cfg_set("results.unreinforced", {"center": (xc,yc), "minFs": float(Fs_min)})

    # 可視化
    Xd = np.linspace(0, L, 600); Yg = np.array([ground.y_at(x) for x in Xd])
    xs = np.linspace(x1, x2, 400)
    ys = yc - np.sqrt(np.maximum(0.0, R*R - (xs-xc)**2))

    fig,ax = plt.subplots(figsize=(9.5,6))
    ax.plot(Xd, Yg, lw=2.0, label="Ground")
    ax.plot(xs, ys, lw=2.5, color="tab:red", label=f"Min Fs = {Fs_min:.3f}")
    ax.plot([xc, xs[0]],[yc, ground.y_at(xs[0])], lw=1.0, color="tab:red")
    ax.plot([xc, xs[-1]],[yc, ground.y_at(xs[-1])], lw=1.0, color="tab:red")
    ax.set_title("最小Fs円弧（保存済）")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    st.success("最小円弧（Fs・R・x1/x2）を cfg.results.chosen_arc に保存しました。")

# ---------------------------------------------------------------
# Page4: ネイル配置（頭位置だけ定義）
# ---------------------------------------------------------------
elif page.startswith("4"):
    st.subheader("ソイルネイル配置（頭位置のみ）")
    H,L,ground = make_ground_from_cfg()

    # 斜面の測地長を作る（sパラメータ）
    Xd = np.linspace(ground.X[0], ground.X[-1], 1200)
    Yg = np.array([float(ground.y_at(x)) for x in Xd])
    seg = np.sqrt(np.diff(Xd)**2 + np.diff(Yg)**2)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = float(s_cum[-1])

    nails_cfg = cfg_get("nails")
    s_start = st.slider("s_start (m)", 0.0, s_total, nails_cfg["s_start"], 0.5)
    s_end   = st.slider("s_end (m)", s_start, s_total, nails_cfg["s_end"], 0.5)
    S_surf  = st.slider("斜面ピッチ S_surf (m)", 0.5, 5.0, nails_cfg["S_surf"], 0.1)

    st.radio("角度モード", ["Slope-Normal (⊥斜面)", "Horizontal-Down (β°)"], index=0, key="angle_mode")
    if st.session_state["angle_mode"].endswith("β°"):
        beta_deg = st.slider("β（水平から下向き °）", 0.0, 45.0, nails_cfg["beta_deg"], 1.0)
        nails_cfg["beta_deg"] = float(beta_deg)
    else:
        delta_beta = st.slider("法線からの微調整 ±Δβ（°）", -10.0, 10.0, nails_cfg["delta_beta"], 1.0)
        nails_cfg["delta_beta"] = float(delta_beta)

    L_mode = st.radio("長さモード", ["パターン1：固定長", "パターン2：すべり面より +Δm"], index=0)
    nails_cfg["L_mode"] = L_mode
    if L_mode.startswith("パターン1"):
        L_nail = st.slider("ネイル長 L (m)", 1.0, 15.0, nails_cfg["L_nail"], 0.5)
        nails_cfg["L_nail"] = float(L_nail)
    else:
        d_embed = st.slider("すべり面より +Δm (m)", 0.0, 5.0, nails_cfg["d_embed"], 0.5)
        nails_cfg["d_embed"] = float(d_embed)

    nails_cfg["s_start"] = float(s_start); nails_cfg["s_end"] = float(s_end)
    nails_cfg["S_surf"]  = float(S_surf)
    nails_cfg["angle_mode"] = st.session_state["angle_mode"]
    cfg_set("nails", nails_cfg)

    # s→(x,y) 補間
    def x_at_s(sv):
        idx = np.searchsorted(s_cum, sv, side="right")-1
        idx = max(0, min(idx, len(Xd)-2))
        t = (sv - s_cum[idx]) / (seg[idx] if seg[idx]>1e-12 else 1e-12)
        return float((1-t)*Xd[idx] + t*Xd[idx+1])

    s_vals = list(np.arange(s_start, s_end+1e-9, S_surf))
    nail_heads = [(x_at_s(sv), float(ground.y_at(x_at_s(sv)))) for sv in s_vals]
    cfg_set("results.nail_heads", nail_heads)

    # プレビュー
    fig,ax = plt.subplots(figsize=(9.5,6))
    ax.plot(Xd, Yg, lw=2.0, label="Ground")
    if nail_heads:
        ax.scatter([p[0] for p in nail_heads], [p[1] for p in nail_heads], s=36, color="tab:blue", label=f"Heads: {len(nail_heads)}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)
    st.success("ネイル頭（results.nail_heads）を保存しました。")

# ---------------------------------------------------------------
# Page5: 補強後解析（最小円弧 × ネイル）
# ---------------------------------------------------------------
elif page.startswith("5"):
    st.subheader("補強後解析（最小円弧 × ソイルネイル）")

    arc = cfg_get("results.chosen_arc")
    nail_heads = cfg_get("results.nail_heads", [])
    if not arc:
        st.info("最小円弧が未確定です。Page3で『円弧探索（未補強）』を実行してください。"); st.stop()
    if not nail_heads:
        st.info("ネイル頭が未配置です。Page4で配置してください。"); st.stop()

    H,L,ground = make_ground_from_cfg()
    # 層（簡易）：3層固定
    m = cfg_get("layers.mat")
    soils=[Soil(m[1]["gamma"], m[1]["c"], m[1]["phi"]),
           Soil(m[2]["gamma"], m[2]["c"], m[2]["phi"]),
           Soil(m[3]["gamma"], m[3]["c"], m[3]["phi"])]
    allow_cross=[True, True]
    interfaces=[]  # デモ簡略（境界線を使う場合は GroundPL を用意してもOK）

    # ネイル材料/動員（Sidebar）
    with st.sidebar:
        st.subheader("Soil Nail（材料/動員）")
        eta_mob = st.slider("η_mob（動員率）", 0.1, 1.0, 0.9, 0.05)
        gamma_m = st.slider("γ_m（部分係数）", 1.0, 2.0, float(cfg_get("layers.gamma_m", 1.2)), 0.05)
    mat = {
        "tau_grout_cap_kPa": float(cfg_get("layers.tau_grout_cap_kPa")),
        "d_g": float(cfg_get("layers.d_g")),
        "d_s": float(cfg_get("layers.d_s")),
        "fy":  float(cfg_get("layers.fy")),
        "gamma_m": float(gamma_m),
        "eta_mob": float(eta_mob),
    }

    # nails生成（Page4設定 + 最小円弧）
    nails_cfg = cfg_get("nails")
    nails = build_nails_from_cfg(ground, nail_heads, nails_cfg, arc)

    # Fs_before は Page3で保存した未補強Fs
    Fs_before = float(arc.get("Fs", 1.0))
    xc,yc,R = arc["xc"], arc["yc"], arc["R"]

    result = integrate_nails_for_best_circle(
        ground=ground, interfaces=interfaces, soils=soils, allow_cross=allow_cross,
        xc=xc, yc=yc, R=R, n_slices=40,
        Fs_before=Fs_before, nails=nails, mat=mat,
    )
    cfg_set("results.reinforced", result)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Fs (before)", f"{result['Fs_before']:.3f}")
    with c2: st.metric("Fs (after)",  f"{result['Fs_after']:.3f}")
    with c3: st.metric("ΔFs",         f"{(result['Fs_after']-result['Fs_before']):.3f}")
    st.caption(f"ΣRk = {result['R_sum']:.3f} kN/m D = { (result['D_sum'] or 0.0):.3f} kN/m")

    fig, ax = draw_cross_section_with_nails(
        ground=ground,
        xc=xc, yc=yc, R=R,
        x_min=(result['x_min'] if result['x_min'] is not None else (xc-R)),
        x_max=(result['x_max'] if result['x_max'] is not None else (xc+R)),
        nails_logs=result['logs'],
        Fs_before=result['Fs_before'], Fs_after=result['Fs_after'],
        R_sum=result['R_sum'], D_sum=result['D_sum'],
        nails_raw=nails,
        title="補強後滑り円弧とネイルの横断図",
    )
    st.pyplot(fig); plt.close(fig)

    with st.expander("Nail logs (debug)"):
        st.json(result['logs'])

# ---------------------------------------------------------------
# fallback
# ---------------------------------------------------------------
else:
    st.info("Page1〜5を順にご利用ください。")
