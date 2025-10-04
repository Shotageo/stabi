# stabi_lem.py
# -*- coding: utf-8 -*-
"""
安定板２（復帰仕様）
- Soil, CircleSlip
- 安全な ground.y_at() 評価
- Bishop簡略法（未補強）
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
import math
import numpy as np


@dataclass
class Soil:
    gamma: float
    c: float
    phi: float


@dataclass
class CircleSlip:
    xc: float
    yc: float
    R: float


# ============ 幾何 ============
def circle_theta_from_x(slip: CircleSlip, x: np.ndarray) -> np.ndarray:
    v = np.clip((x - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)


def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    X = np.linspace(x_min, x_max, 1001)
    th = circle_theta_from_x(slip, X)
    yu = slip.yc + slip.R * np.sin(th)
    yl = slip.yc - slip.R * np.sin(th)
    yg = ground_y_at(X)

    under_u = yu <= yg
    under_l = yl <= yg
    y_arc = np.where(under_u & under_l, np.maximum(yu, yl),
             np.where(under_u, yu, np.where(under_l, yl, np.nan)))
    valid = np.isfinite(y_arc)
    if not np.any(valid):
        return []

    idx = np.where(valid)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, len(idx) - 1]
    lengths = idx[ends] - idx[starts]
    k = int(np.argmax(lengths))
    i0, i1 = idx[starts[k]], idx[ends[k]]
    xL, xR = X[i0], X[i1]
    if xR - xL <= 1e-6:
        return []

    xs = np.linspace(xL, xR, n_slices + 1)
    out = []
    for i in range(n_slices):
        x_a, x_b = xs[i], xs[i+1]
        xm = 0.5 * (x_a + x_b)
        th_m = float(circle_theta_from_x(slip, np.array([xm]))[0])
        tx, ty = -math.sin(th_m), math.cos(th_m)
        alpha = math.atan2(ty, tx)

        grid = np.linspace(x_a, x_b, 9)
        thg = circle_theta_from_x(slip, grid)
        yu_g = slip.yc + slip.R*np.sin(thg)
        yl_g = slip.yc - slip.R*np.sin(thg)
        yg_g = ground_y_at(grid)
        under_u_g = yu_g <= yg_g
        under_l_g = yl_g <= yg_g
        y_seg = np.where(under_u_g & under_l_g, np.maximum(yu_g, yl_g),
                 np.where(under_u_g, yu_g, np.where(under_l_g, yl_g, yg_g)))
        h = np.maximum(yg_g - y_seg, 0.0)
        area = float(np.trapz(h, grid))
        W = soil_gamma * area
        out.append({"alpha": alpha, "width": x_b - x_a, "W": W})
    return out


def bishop_fs_unreinforced(slices: List[dict], soil: Soil) -> float:
    phi = math.radians(soil.phi)
    c = soil.c
    Fs = 1.2
    for _ in range(100):
        num = den = 0.0
        for s in slices:
            a = s["alpha"]; W = s["W"]; b = s["width"]
            m = 1.0 + (math.tan(phi) * math.sin(a)) / max(Fs, 1e-9)
            Np = (W * math.cos(a)) / m
            shear = c*b + Np*math.tan(phi)
            num += shear
            den += W * math.sin(a)
        new_Fs = num / max(den, 1e-12)
        if abs(new_Fs - Fs) < 1e-5:
            return new_Fs
        Fs = new_Fs
    return Fs