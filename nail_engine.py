# nail_engine.py — 修正版（能力の単位を是正、交点は「最小の正の根」を採用）
from __future__ import annotations
import numpy as np, math

DEG = math.pi/180.0
EPS = 1e-9

def _slope_tangent_angle(ground, x):
    x2 = x + 1e-4
    y1 = float(ground.y_at(x)); y2 = float(ground.y_at(x2))
    return math.atan2((y2 - y1), (x2 - x))

def reinforce_nails(arc: dict, ground, soils, nails_cfg: dict, slices: dict):
    xmid = slices["x_mid"]; alpha = slices["alpha"]
    N = len(xmid)
    Tt = np.zeros(N, dtype=float)
    hits = []

    heads = nails_cfg.get("heads", []) or nails_cfg.get("nail_heads", [])
    if not heads:
        return Tt, {"hits": hits, "notes": "no nails"}

    xc, yc, R = float(arc["xc"]), float(arc["yc"]), float(arc["R"])

    # materials
    d_g = float(nails_cfg.get("d_g", 0.125))      # m
    d_s = float(nails_cfg.get("d_s", 0.022))      # m
    fy  = float(nails_cfg.get("fy", 1000.0))      # MPa
    gamma_m = float(nails_cfg.get("gamma_m", 1.2))
    tau_cap = float(nails_cfg.get("tau_grout_cap_kPa", 150.0))  # kPa = kN/m^2

    angle_mode = str(nails_cfg.get("angle_mode", "Slope-Normal (⊥斜面)"))
    beta  = float(nails_cfg.get("beta_deg", 15.0)) * DEG
    delta = float(nails_cfg.get("delta_beta", 0.0)) * DEG
    L_mode  = str(nails_cfg.get("L_mode", "パターン1：固定長"))
    L_nail  = float(nails_cfg.get("L_nail", 5.0))
    d_embed = float(nails_cfg.get("d_embed", 1.0))  # パターン2：すべり面より +Δm

    for (xh, yh) in heads:
        # --- 方向角：斜面法線は “地山側” へ（+π/2）
        if angle_mode.startswith("Slope-Normal"):
            tau = _slope_tangent_angle(ground, float(xh))
            theta = tau + math.pi/2 + delta
        else:
            theta = -beta  # 水平から下向きβ°
        ct, st = math.cos(theta), math.sin(theta)

        # --- 円との交点（光線：t>0 最小根）
        B = 2*((xh - xc)*ct + (yh - yc)*st)
        C = (xh - xc)**2 + (yh - yc)**2 - R**2
        disc = B*B - 4*C
        if disc < 0:
            hits.append(dict(reason="no_intersection", xq=None, yq=None, idx=None, Tt=0.0)); continue
        sdisc = math.sqrt(max(0.0, disc))
        t_candidates = [(-B - sdisc)/2.0, (-B + sdisc)/2.0]
        t_pos = [t for t in t_candidates if t > EPS]
        if not t_pos:
            hits.append(dict(reason="intersection_behind", xq=None, yq=None, idx=None, Tt=0.0)); continue
        t = min(t_pos)
        xq, yq = float(xh + t*ct), float(yh + t*st)

        # --- ボンド長
        if L_mode.startswith("パターン2"):            # すべり面より +Δm
            L_bond = max(0.0, d_embed)               # ← Δm をそのままボンド長に採用
        else:                                        # 固定長
            L_bond = max(0.0, L_nail - t)
        if L_bond < 0.20:
            hits.append(dict(reason="short_bond", xq=xq, yq=yq, idx=None, Tt=0.0)); continue

        # --- 能力（単位整合）
        # τ[kPa]=kN/m^2 × (π d_g L)[m^2] → kN/m
        T_bond  = tau_cap * (math.pi * d_g * L_bond)
        # fy[MPa]→kN: A[m^2] * fy[MPa]*1e3[kN/m^2]
        A_s     = math.pi * (d_s**2) / 4.0
        T_steel = (A_s * fy * 1e3) / gamma_m
        T_cap   = min(T_bond, T_steel)

        # --- スライス所属 & 投影（負は寄与なし）
        idx = int(np.searchsorted(xmid, xq) - 1)
        idx = max(0, min(idx, N-1))
        proj = math.cos(theta - float(alpha[idx]))
        if proj <= 0.0:
            hits.append(dict(reason="unfavorable_angle", xq=xq, yq=yq, idx=idx, Tt=0.0)); continue

        Tti = T_cap * proj
        Tt[idx] += Tti
        hits.append(dict(reason="ok", xq=xq, yq=yq, idx=idx, Tt=Tti, L_bond=L_bond, T_bond=T_bond, T_steel=T_steel))

    return Tt, {"hits": hits}
