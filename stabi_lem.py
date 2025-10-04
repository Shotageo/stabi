# ================================================================
# Soil Nail Integration (UI drop-in) v1.2 + Cross-section v1.3
#   - stabi_lem.py は改変不要
#   - 最小円弧の未補強 Fs と D_sum を使って Fs_after を合成
#   - 横断図で Ground / Slip circle / Sliding mass / Nail交点 を表示
#   - 2D 1m 幅想定（kN/m 系）
# ================================================================
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt



# ================================================================
# Plot style（Theme/Tight layout/Legend切替）— ユーザー希望ブロック
# ================================================================
def apply_plot_style(tight=True):
    if tight:
        try:
            plt.tight_layout()
        except Exception:
            pass

def show_legend(ax, show=True):
    if show:
        ax.legend(loc='upper left', frameon=True)

# ================================================================
# サイドバー設定（Nail）
# ================================================================
st.sidebar.subheader('Soil Nail')

if 'nails_cfg' not in st.session_state:
    st.session_state['nails_cfg'] = {
        'enabled': False,
        'eta_mob': 0.9,    # 動員率（初期は定数）
        'gamma_s': 1.15,   # 材料部分係数
        'gamma_b': 1.25,   # 付着部分係数
        'tau_bond': 100.0, # [kPa] デフォ100
        'Fy': 500.0,       # [MPa] デフォ500
        'bar_d': 25.0,     # [mm]  デフォ25
    }
CFG_N = st.session_state['nails_cfg']

CFG_N['enabled'] = st.sidebar.toggle('Enable nails (Refine circle only)', value=CFG_N['enabled'])
colN1, colN2, colN3 = st.sidebar.columns(3)
with colN1:
    CFG_N['eta_mob'] = st.number_input('η_mob', 0.0, 1.0, CFG_N['eta_mob'], 0.05)
with colN2:
    CFG_N['gamma_s'] = st.number_input('γ_s', 1.0, 2.0, CFG_N['gamma_s'], 0.05)
with colN3:
    CFG_N['gamma_b'] = st.number_input('γ_b', 1.0, 2.0, CFG_N['gamma_b'], 0.05)
colN4, colN5, colN6 = st.sidebar.columns(3)
with colN4:
    CFG_N['tau_bond'] = st.number_input('τ_bond [kPa]', 10.0, 5000.0, CFG_N['tau_bond'], 10.0)
with colN5:
    CFG_N['Fy'] = st.number_input('Fy [MPa]', 200.0, 1000.0, CFG_N['Fy'], 10.0)
with colN6:
    CFG_N['bar_d'] = st.number_input('bar d [mm]', 10.0, 50.0, CFG_N['bar_d'], 1.0)

# ================================================================
# Nail × Circle helpers（UI側／自給）
# ================================================================
EPS = 1e-12

def _axis_unit(azimuth_rad: float):
    return math.cos(azimuth_rad), math.sin(azimuth_rad)

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
    return (xi, yi, t)

def _circle_tangent_unit_at(P, C):
    (px,py),(cx,cy) = P,C
    rx, ry = (px-cx), (py-cy)
    rn = math.hypot(rx, ry)
    if rn < EPS: return (1.0,0.0)
    # 接線（反時計回り）
    return (-ry/rn, rx/rn)

def nails_R_sum_kN_per_m(nails, xc, yc, R, CFG_N, x_min=None, x_max=None):
    """
    最小円弧と交差するネイルの接線方向合力 ΣRk [kN/m] を返す。
    nails: [{x,y,azimuth,length}, ...]  # 単位 m, rad, m
    CFG_N: η, γs, γb, τ_bond[kPa], Fy[MPa], bar_d[mm]
    """
    # 単位変換（すべて kN/m 系へ）
    tau_bond_kPa = float(CFG_N['tau_bond'])
    Fy_MPa       = float(CFG_N['Fy'])
    d_mm         = float(CFG_N['bar_d'])
    eta          = float(CFG_N['eta_mob'])
    gamma_s      = float(CFG_N['gamma_s'])
    gamma_b      = float(CFG_N['gamma_b'])

    d_m      = d_mm / 1000.0
    A_m2     = math.pi*(d_m**2)/4.0
    # 応力度: MPa(=N/mm^2) -> kN/m^2 : 1 MPa = 1e6 Pa = 1e6 N/m^2 = 1e3 kN/m^2
    Fy_kN_m2 = Fy_MPa * 1e3
    # 付着: kPa -> kN/m^2 : 1 kPa = 1e3 N/m^2 = 1 kN/m^2
    tau_kN_m2 = tau_bond_kPa * 1.0

    R_sum = 0.0
    logs  = []
    C = (xc, yc)
    for i, n in enumerate(nails):
        xh, yh = n['x'], n['y']
        ux, uy = _axis_unit(n['azimuth'])
        tip = (xh + ux*n['length'], yh + uy*n['length'])
        hit = _segment_circle_intersection((xh,yh), tip, C, R)
        if not hit:
            logs.append({"id": i, "ok": False, "reason": "no_intersection"}); continue
        xi, yi, t = hit
        if (x_min is not None and xi < x_min-1e-9) or (x_max is not None and xi > x_max+1e-9):
            logs.append({"id": i, "ok": False, "reason": "outside_arc"}); continue

        # 埋込み長（交点→先端）
        embed = max(0.0, n['length']*(1.0 - t))

        # 上限（材料 vs 付着）
        T_y_kN = (A_m2 * Fy_kN_m2) / max(gamma_s, 1e-6)                  # kN
        T_b_kN = (math.pi * d_m * embed * tau_kN_m2) / max(gamma_b, 1e-6) # kN
        T_cap  = min(T_y_kN, T_b_kN)
        T_kN   = eta * T_cap

        # 接線投影
        tx, ty = _circle_tangent_unit_at((xi,yi), C)
        dot_t  = ux*tx + uy*ty
        Rk = max(0.0, T_kN * dot_t)  # kN（=kN/m の 2D 等価）
        R_sum += Rk
        logs.append({"id": i, "ok": True, "x": xi, "y": yi, "t": t, "embed": embed,
                     "T_kN": T_kN, "dot_t": dot_t, "Rk": Rk,
                     # 追加：ヘッド/先端も出しておくと二色線分に使える
                     "head_x": xh, "head_y": yh, "tip_x": tip[0], "tip_y": tip[1],
                     })
    return R_sum, logs

# ================================================================
# 統合関数：最小円弧の Fs_after を合成
# ================================================================
def integrate_nails_for_best_circle(
    ground, interfaces, soils, allow_cross,
    xc, yc, R, n_slices,
    Fs_before: float,
    nails: list,
    cfgN: dict,
):
    """
    - D_sum = Σ(W sinα) を stabi_lem から取得
    - R_sum を UI 側で計算
    - Fs_after = Fs_before + R_sum / D_sum
    戻り値: dict(Fs_before, Fs_after, R_sum, D_sum, logs, x_min, x_max)
    """
    out = {
        "Fs_before": Fs_before,
        "Fs_after": Fs_before,
        "R_sum": 0.0,
        "D_sum": None,
        "logs": [],
        "x_min": None, "x_max": None,
    }
    if Fs_before is None:
        return out

    # D_sum と弧の x 範囲
    pack = driving_sum_for_R_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices)
    if pack is None:
        return out
    D_sum, x_min, x_max = pack
    out["D_sum"] = D_sum
    out["x_min"], out["x_max"] = x_min, x_max

    if not cfgN.get('enabled', False) or not nails:
        return out

    R_sum, logs = nails_R_sum_kN_per_m(nails, xc, yc, R, cfgN, x_min=x_min, x_max=x_max)
    out["R_sum"] = R_sum
    out["logs"]  = logs

    if (D_sum is not None) and (D_sum > 0.0):
        out["Fs_after"] = float(Fs_before) + float(R_sum)/float(D_sum)
    return out

# ================================================================
# 横断図描画（Cross-section）
# ================================================================
def draw_cross_section_with_nails(
    ground, xc, yc, R,
    x_min, x_max,
    nails_logs: list,
    Fs_before: float, Fs_after: float,
    R_sum: float, D_sum: float,
    nails_raw: list = None,           # ←二色描画したいときに渡す
    title: str = "Cross-section (Nails × Slip Circle)",
):
    """
    - ground: GroundPL
    - (xc,yc,R): 最小円弧
    - x_min, x_max: 有効弧の左右端（driving_sum_for_R_multiの戻り）
    - nails_logs: integrate 関数の logs（交点/力など）
    - nails_raw: [{x,y,azimuth,length}, ...] を渡すとヘッド→交点／交点→先端を二色で描画
    """
    # 円弧の可視化用サンプル（x_min..x_max を細分）
    xs = np.linspace(x_min, x_max, 401)
    inside = R*R - (xs - xc)**2
    valid = inside > 0
    xs = xs[valid]
    y_arc = yc - np.sqrt(inside[valid])
    y_g   = ground.y_at(xs)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # 地表と円弧
    Xg, Yg = ground.X, ground.Y
    ax.plot(Xg, Yg, '-', lw=2.0, label='Ground')
    ax.plot(xs, y_arc, '--', lw=2.0, label='Slip circle')

    # すべり土塊（地表と円弧の間を塗る）
    ax.fill_between(xs, y_arc, y_g, where=(y_g>=y_arc), alpha=0.15, label='Sliding mass')

    # ネイル描画
    if nails_logs:
        if nails_raw and len(nails_raw) >= len(nails_logs):
            # 二色線分（ヘッド→交点：赤、交点→先端：青）
            for log, n in zip(nails_logs, nails_raw):
                if not log.get('ok'):
                    continue
                xh, yh = n['x'], n['y']
                ux, uy = math.cos(n['azimuth']), math.sin(n['azimuth'])
                xt, yt = xh + ux*n['length'], yh + uy*n['length']
                xi, yi = log['x'], log['y']
                ax.plot([xh, xi], [yh, yi], '-', lw=2.0, label=None)   # Active側
                ax.plot([xi, xt], [yi, yt], '-', lw=2.0, label=None)   # Passive側
                size = max(24, min(140, 10 + 1.0*log.get('T_kN', 0.0)))
                ax.scatter([xi], [yi], s=size, label=None)
        else:
            # 交点マーカーのみ（簡易）
            for log in nails_logs:
                if not log.get('ok'):
                    continue
                xi, yi = log['x'], log['y']
                size = max(24, min(140, 10 + 1.0*log.get('T_kN', 0.0)))
                ax.scatter([xi], [yi], s=size, label=None)

    # 注記（Fsなど）
    if (Fs_before is not None) and (Fs_after is not None):
        delta = (Fs_after - Fs_before)
        txt = f"Fs_before = {Fs_before:.3f}\nFs_after  = {Fs_after:.3f}\nΔFs = {delta:.3f}"
        if (R_sum is not None) and (D_sum is not None) and D_sum>0:
            txt += f"\nΣRk = {R_sum:.3f} kN/m\nD = {D_sum:.3f} kN/m"
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75), fontsize=10)

    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim(min(Xg.min(), x_min), max(Xg.max(), x_max))
    y_min = min(Yg.min(), (y_arc.min() if len(y_arc)>0 else Yg.min()) - 0.1*(Yg.ptp()+1))
    ax.set_ylim(y_min, Yg.max()*1.05)
    show_legend(ax, show=True)
    apply_plot_style(tight=True)
    return fig, ax

# ================================================================
# 呼び出し例（あなたの変数に置換）
# ================================================================
# >>> REPLACE HERE: あなたの既存コンテキストに合わせて以下を与えてください
# - 最小円弧（Refine確定済み）
#     xc, yc, R, n_slices = best_circle.cx, best_circle.cy, best_circle.radius, best_n_slices
# - 未補強Fs
#     Fs_before = fs_given_R_multi(...) or fs_bishop_poly_multi(...) 等で得た値
# - ネイル配列（UIで保持）
#     nails = st.session_state.get('nails', [])
#
# 例（ダミー取得の形だけ。実装では既存変数に置換）:
# xc, yc, R        = best_circle_cx, best_circle_cy, best_circle_R   # ←置換
# n_slices         = 40                                              # ←置換
# Fs_before        = best_fs_before                                  # ←置換
# nails            = st.session_state.get('nails', [])               # ←置換
# ground, interfaces, soils, allow_cross をあなたの既存オブジェクトに合わせて渡す

# --- 実行（上の変数が与えられている前提） ---
# result = integrate_nails_for_best_circle(
#     ground=ground, interfaces=interfaces, soils=soils, allow_cross=allow_cross,
#     xc=xc, yc=yc, R=R, n_slices=n_slices,
#     Fs_before=Fs_before, nails=nails, cfgN=CFG_N,
# )
#
# # 数値表示
# st.markdown(f"**Fs (before)**: {Fs_before:.3f}")
# if CFG_N['enabled']:
#     st.markdown(f"**Fs (after)**: {result['Fs_after']:.3f}")
#     if result['D_sum'] is not None and result['D_sum'] > 0:
#         delta = result['R_sum']/result['D_sum']
#         st.caption(f"ΣRk = {result['R_sum']:.3f} kN/m,  D = {result['D_sum']:.3f} kN/m  →  ΔFs = {delta:.3f}")
#
# # 横断図
# if CFG_N['enabled'] and result['x_min'] is not None and result['x_max'] is not None:
#     fig, ax = draw_cross_section_with_nails(
#         ground=ground,
#         xc=xc, yc=yc, R=R,
#         x_min=result['x_min'], x_max=result['x_max'],
#         nails_logs=result['logs'],
#         Fs_before=result['Fs_before'], Fs_after=result['Fs_after'],
#         R_sum=result['R_sum'], D_sum=result['D_sum'],
#         nails_raw=nails,  # ←二色描画したくなければ外してOK
#     )
#     st.pyplot(fig)
#
# # デバッグログ（任意）
# if CFG_N['enabled'] and result['logs']:
#     with st.expander('Nail logs (debug)'):
#         st.json(result['logs'])
