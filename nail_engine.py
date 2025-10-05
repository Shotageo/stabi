# nail_engine.py — minimal soil nail engine (single-layer, no-u)
from __future__ import annotations
import numpy as np, math

DEG = math.pi/180.0
EPS = 1e-9

def _slope_tangent_angle(ground, x):
    x2 = x + 1e-4
    y1 = float(ground.y_at(x)); y2 = float(ground.y_at(x2))
    return math.atan2((y2 - y1), (x2 - x))

def reinforce_nails(arc: dict, ground, soils, nails_cfg: dict, slices: dict):
    """
    returns:
      Tt_per_slice (np.ndarray N), diag(dict with hits)
    前提：単層。τ はグラウト上限で制限（層は未分割）
    nails_cfg 必要キー:
      heads[[(x,y),..]], angle_mode('Slope-Normal (⊥斜面)'|'Horizontal-Down (β°)'),
      beta_deg, delta_beta, L_mode('パターン1：固定長'|'パターン2：すべり面より +Δm'),
      L_nail, d_embed, d_g, d_s, fy, gamma_m, tau_grout_cap_kPa
    """
    xmid = slices["x_mid"]; alpha = slices["alpha"]
    N = len(xmid)
    Tt = np.zeros(N, dtype=float)
    hits = []

    heads = nails_cfg.get("heads", []) or nails_cfg.get("nail_heads", [])
    if not heads:
        return Tt, {"hits": hits, "notes": "no nails"}

    xc, yc, R = float(arc["xc"]), float(arc["yc"]), float(arc["R"])

    # material
    d_g = float(nails_cfg.get("d_g", 0.125))
    d_s = float(nails_cfg.get("d_s", 0.022))
    fy = float(nails_cfg.get("fy", 1000.0))
    gamma_m = float(nails_cfg.get("gamma_m", 1.2))
    tau_cap = float(nails_cfg.get("tau_grout_cap_kPa", 150.0))  # kPa

    angle_mode = str(nails_cfg.get("angle_mode", "Slope-Normal (⊥斜面)"))
    beta = float(nails_cfg.get("beta_deg", 15.0)) * DEG
    delta = float(nails_cfg.get("delta_beta", 0.0)) * DEG
    L_mode = str(nails_cfg.get("L_mode", "パターン1：固定長"))
    L_nail = float(nails_cfg.get("L_nail", 5.0))
    d_embed = float(nails_cfg.get("d_embed", 1.0))

    for (xh, yh) in heads:
        # 方向角 theta
        if angle_mode.startswith("Slope-Normal"):
            tau = _slope_tangent_angle(ground, float(xh))
            theta = tau - math.pi/2 + delta
        else:
            theta = -beta  # 水平から下向きβ°

        ct, st = math.cos(theta), math.sin(theta)

        # 直線(頭から)と円の交点（t>0側）
        # p(t) = (xh,yh) + t*(ct,st)
        A = 1.0
        B = 2*((xh - xc)*ct + (yh - yc)*st)
        C = (xh - xc)**2 + (yh - yc)**2 - R**2
        disc = B*B - 4*A*C
        if disc < 0:
            hits.append(dict(reason="no_intersection", xq=None, yq=None, idx=None, Tt=0.0))
            continue
        rt = math.sqrt(max(0.0, disc))
        t1 = (-B - rt)/2.0; t2 = (-B + rt)/2.0
        t = t2 if t2 > t1 else t1
        if t <= EPS:
            hits.append(dict(reason="intersection_behind", xq=None, yq=None, idx=None, Tt=0.0))
            continue
        xq, yq = float(xh + t*ct), float(yh + t*st)

        # ボンド長（簡易）
        if L_mode.startswith("パターン2"):
            L_bond = max(0.0, L_nail - t + d_embed)
        else:
            L_bond = max(0.0, L_nail - t)
        if L_bond < 0.20:
            hits.append(dict(reason="short_bond", xq=xq, yq=yq, idx=None, Tt=0.0))
            continue

        # 能力：bond（kPa→kN/m）、steel（MPa→kN、2DなのでkN/m換算）
        T_bond = tau_cap * 1e-3 * (math.pi * d_g * L_bond)         # kN/m
        A_s = math.pi*(d_s**2)/4.0                                 # m^2
        T_steel = (A_s * fy) / gamma_m / 1.0                       # ≈ kN/m（幅1mと解釈）
        T_cap = min(T_bond, T_steel)

        # 所属スライスと投影
        idx = int(np.searchsorted(xmid, xq) - 1)
        idx = max(0, min(idx, N-1))
        proj = math.cos(theta - float(alpha[idx]))
        if proj <= 0.0:
            hits.append(dict(reason="unfavorable_angle", xq=xq, yq=yq, idx=idx, Tt=0.0))
            continue

        Tti = T_cap * proj
        Tt[idx] += Tti
        hits.append(dict(reason="ok", xq=xq, yq=yq, idx=idx, Tt=Tti, L_bond=L_bond))

    return Tt, {"hits": hits}

