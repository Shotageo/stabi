# stabi_lem.py
# -*- coding: utf-8 -*-
"""
安定板２（I：multi-UI版／補強はモック）のコア:
- Soil / CircleSlip
- 円弧と地表の幾何
- 円弧上スライス生成（地表下側を採用、最長区間のみ）
- Bishop簡略法（未補強）で Fs 計算
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
import math
import numpy as np


# ========= 基本データ構造 =========
@dataclass
class Soil:
    gamma: float  # kN/m3
    c: float      # kPa (kN/m2)
    phi: float    # deg


@dataclass
class CircleSlip:
    xc: float
    yc: float
    R: float


# ========= 幾何ユーティリティ =========
def circle_xy_from_theta(slip: CircleSlip, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return slip.xc + slip.R*np.cos(theta), slip.yc + slip.R*np.sin(theta)

def circle_theta_from_x(slip: CircleSlip, x: np.ndarray) -> np.ndarray:
    v = np.clip((np.asarray(x) - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)


# ========= スライス生成 =========
def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    """
    地表 y_g(x) と円弧の上下解 y_u=yc+R sinθ, y_l=yc-R sinθ を比較し、
    地表以下(<=y_g)に存在する円弧側を採用。最長の連続区間だけ x 等分。
    返す各スライス: {alpha, width, W, area, x_a, x_b}
    """
    X = np.linspace(x_min, x_max, 1201)
    th = circle_theta_from_x(slip, X)
    yu = slip.yc + slip.R*np.sin(th)
    yl = slip.yc - slip.R*np.sin(th)
    yg = ground_y_at(X)

    under_u = yu <= yg
    under_l = yl <= yg
    yc = np.where(under_u & under_l, np.maximum(yu, yl),
         np.where(under_u, yu, np.where(under_l, yl, np.nan)))
    valid = np.isfinite(yc)
    if not np.any(valid):
        return []

    idx = np.where(valid)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, brk+1]
    ends   = np.r_[brk, len(idx)-1]
    lengths = idx[ends] - idx[starts]
    k = int(np.argmax(lengths))
    i0, i1 = idx[starts[k]], idx[ends[k]]
    xL, xR = float(X[i0]), float(X[i1])
    if xR - xL <= 1e-9:
        return []

    xs = np.linspace(xL, xR, n_slices+1)
    out: List[dict] = []
    for i in range(n_slices):
        x_a, x_b = float(xs[i]), float(xs[i+1])
        xm = 0.5*(x_a+x_b)
        th_m = float(circle_theta_from_x(slip, np.array([xm]))[0])
        tx, ty = -math.sin(th_m), math.cos(th_m)  # 円の接線
        alpha = math.atan2(ty, tx)

        grid = np.linspace(x_a, x_b, 9)
        thg = circle_theta_from_x(slip, grid)
        yu_g = slip.yc + slip.R*np.sin(thg)
        yl_g = slip.yc - slip.R*np.sin(thg)
        yg_g = ground_y_at(grid)
        under_u_g = yu_g <= yg_g
        under_l_g = yl_g <= yg_g
        yc_seg = np.where(under_u_g & under_l_g, np.maximum(yu_g, yl_g),
                np.where(under_u_g, yu_g, np.where(under_l_g, yl_g, yg_g)))
        h = np.maximum(yg_g - yc_seg, 0.0)
        area = float(np.trapz(h, grid))
        W = soil_gamma * area  # kN（奥行1m）

        out.append({
            "alpha": alpha,
            "width": (x_b - x_a),
            "W": W,
            "area": area,
            "x_a": x_a, "x_b": x_b
        })
    return out


# ========= Bishop簡略法（未補強） =========
def bishop_fs_unreinforced(
    slices: List[dict],
    soil: Soil,
    max_iter: int = 80,
    tol: float = 1e-5
) -> float:
    """
    Bishop簡略法（未補強）。
    cはスライス幅bで c*b、摩擦は N' tanφ。反復してFsを収束。
    """
    phi = math.radians(float(soil.phi))
    c   = float(soil.c)

    Fs = 1.2
    for _ in range(max_iter):
        num = den = 0.0
        for s in slices:
            alpha = float(s["alpha"])
            W     = float(s["W"])
            b     = float(s["width"])
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-9)
            Np = (W * math.cos(alpha)) / m
            shear_res = c*b + (Np * math.tan(phi))
            num += shear_res
            den += W * math.sin(alpha)
        Fs_new = num / max(den, 1e-12)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)