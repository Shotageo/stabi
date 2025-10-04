# ================================================================
# Soil Nail Integration (UI drop-in) v1.2 — for 安定板２
#   - stabi_lem.py は改変不要
#   - 最小円弧の未補強 Fs と D_sum を使って Fs_after を合成
#   - 2D 1m 幅想定（kN/m 系）
# ================================================================
import math
import streamlit as st
import matplotlib.pyplot as plt

# ---- 1) 必要インポート：stabi_lem の D_sum 取得を使う ----
from stabi_lem import driving_sum_for_R_multi

# ================================================================
# 2) サイドバー設定（Nail）
# ================================================================
st.sidebar.subheader('Soil Nail')

if 'nails_cfg' not in st.session_state:
    st.session_state['nails_cfg'] = {
        'enabled': False,
        'eta_mob': 0.9,   # 動員率（初期は定数）
        'gamma_s': 1.15,  # 材料部分係数
        'gamma_b': 1.25,  # 付着部分係数
        'tau_bond': 100.0,# [kPa] デフォ100
        'Fy': 500.0,      # [MPa] デフォ500
        'bar_d': 25.0,    # [mm]  デフォ25
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
# 3) Nail × Circle helpers（UI側／自給）
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

    d_m          = d_mm / 1000.0
    A_m2         = math.pi*(d_m**2)/4.0
    # 応力度: MPa(=N/mm^2) -> kN/m^2 : 1 MPa = 1e6 Pa = 1e6 N/m^2 = 1e3 kN/m^2
    Fy_kN_m2     = Fy_MPa * 1e3
    # 付着: kPa -> kN/m^2 : 1 kPa = 1e3 N/m^2 = 1 kN/m^2
    tau_kN_m2    = tau_bond_kPa * 1.0

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
        T_y_kN = (A_m2 * Fy_kN_m2) / max(gamma_s, 1e-6)             # kN
        T_b_kN = (math.pi * d_m * embed * tau_kN_m2) / max(gamma_b, 1e-6)  # kN
        T_cap  = min(T_y_kN, T_b_kN)
        T_kN   = eta * T_cap

        # 接線投影
        tx, ty = _circle_tangent_unit_at((xi,yi), C)
        dot_t  = ux*tx + uy*ty
        Rk = max(0.0, T_kN * dot_t)  # kN（=kN/m の 2D 等価）
        R_sum += Rk
        logs.append({"id": i, "ok": True, "x": xi, "y": yi, "t": t, "embed": embed,
                     "T_kN": T_kN, "dot_t": dot_t, "Rk": Rk})
    return R_sum, logs

# ================================================================
# 4) 統合関数：最小円弧の Fs_after を合成
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
# 5) ここから “呼び出し側” の最低限サンプル
# ================================================================
# >>> REPLACE HERE: あなたの既存コンテキストに合わせて以下の変数を提供してください
# 最小円弧（Refine確定済み）
#   xc, yc, R, method, n_slices = best_circle.cx, best_circle.cy, best_circle.radius, "Bishop", 40 など
# 未補強Fs
#   Fs_before = fs_given_R_multi(...) or fs_bishop_poly_multi(...) で得た値
# ネイル配列
#   nails = st.session_state.get('nails', [])
#
# 例（ダミー取得の形だけ示します。実プロジェクトでは既存変数に差し替え）:
# xc, yc, R        = best_circle_cx, best_circle_cy, best_circle_R     # ←置換
# n_slices, method = 40, "Bishop"                                       # ←置換
# Fs_before        = best_fs_before                                     # ←置換
# nails            = st.session_state.get('nails', [])                  # ←置換

# --- 実行（上の変数が与えられている前提） ---
# result = integrate_nails_for_best_circle(
#     ground=ground, interfaces=interfaces, soils=soils, allow_cross=allow_cross,
#     xc=xc, yc=yc, R=R, n_slices=n_slices,
#     Fs_before=Fs_before, nails=nails, cfgN=CFG_N,
# )

# --- 表示例 ---
# st.markdown(f"**Fs (before)**: {Fs_before:.3f}")
# if CFG_N['enabled']:
#     st.markdown(f"**Fs (after)**: {result['Fs_after']:.3f}")
#     if result['D_sum'] is not None:
#         delta = (result['R_sum']/result['D_sum']) if result['D_sum']>0 else 0.0
#         st.caption(f"ΣRk = {result['R_sum']:.3f} kN/m,  D = {result['D_sum']:.3f} kN/m  →  ΔFs = {delta:.3f}")

# --- 可視化：交点マーカー（既存の図に重ねる想定） ---
# fig, ax = plt.subplots(figsize=(6,4))
# # ここで地形、最小円弧、ネイル本体など既存の描画を実施…
# if CFG_N['enabled'] and result['logs']:
#     for log in result['logs']:
#         if not log.get('ok'): 
#             continue
#         ax.scatter([log['x']], [log['y']], s=max(20, min(140, 10+0.5*log['T_kN'])), label=None)
# st.pyplot(fig)

# --- デバッグ出力（任意） ---
# if CFG_N['enabled'] and result['logs']:
#     with st.expander('Nail logs (debug)'):
#         st.json(result['logs'])
