# stabi_lem.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math
from typing import List, Tuple, Callable, Optional
import numpy as np

# =========================
# 基本データ構造
# =========================
@dataclass
class Soil:
    gamma: float   # kN/m3
    c: float       # kPa
    phi: float     # deg

@dataclass
class CircleSlip:
    xc: float
    yc: float
    R: float

@dataclass
class Nail:
    x1: float
    y1: float
    x2: float
    y2: float
    spacing: float            # m（1m幅換算用）
    T_yield: float            # kN/本
    bond_strength: float      # kN/m
    embed_length_each_side: float = 0.5  # m

# =========================
# 幾何ユーティリティ
# =========================
def _dot(ax, ay, bx, by): return ax * bx + ay * by
def _norm(ax, ay): return math.hypot(ax, ay)

def _angle_between(ax, ay, bx, by) -> float:
    na = _norm(ax, ay); nb = _norm(bx, by)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, _dot(ax, ay, bx, by) / (na * nb)))
    return math.acos(cosv)

def circle_xy_from_theta(slip: CircleSlip, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return slip.xc + slip.R*np.cos(theta), slip.yc + slip.R*np.sin(theta)

def circle_theta_from_x(slip: CircleSlip, x: np.ndarray) -> np.ndarray:
    v = np.clip((x - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)

def line_circle_intersections_segment(x1, y1, x2, y2, slip: CircleSlip) -> List[Tuple[float,float]]:
    dx, dy = (x2-x1), (y2-y1)
    fx, fy = (x1-slip.xc), (y1-slip.yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - slip.R*slip.R
    disc = B*B - 4*A*C
    if A == 0 or disc < 0:
        return []
    rt = math.sqrt(disc)
    pts = []
    for s in (-1.0, 1.0):
        t = (-B + s*rt) / (2*A)
        if 0.0 <= t <= 1.0:
            pts.append((x1 + t*dx, y1 + t*dy))
    return pts

# =========================
# スライス生成（地表と円弧から）
# =========================
def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    """
    地表 y_g(x) と円弧 y_c(x) を比較し、y_g >= y_c の領域（地表で覆土のある側）で
    弧の有効区間 [xL, xR] を抽出し、そこでスライス化して重量Wと底面角alphaを与える。
    """
    X = np.linspace(x_min, x_max, 1201)
    theta = circle_theta_from_x(slip, X)
    y_upper = slip.yc + slip.R*np.sin(theta)
    y_lower = slip.yc - slip.R*np.sin(theta)
    yg = ground_y_at(X)

    # 地表に近い側の弧（見かけの交差側）を採用
    yc = np.where(np.abs(yg - y_upper) < np.abs(yg - y_lower), y_upper, y_lower)

    mask = (yg >= yc)
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    xL, xR = X[idx[0]], X[idx[-1]]
    if xR - xL <= 1e-6:
        return []

    xs = np.linspace(xL, xR, n_slices+1)
    slices = []
    for i in range(n_slices):
        x_a, x_b = xs[i], xs[i+1]
        xm = 0.5*(x_a + x_b)

        # 接線角：弧のパラメトリックから（θでの接線は (-sinθ, +cosθ)）
        th_m = float(circle_theta_from_x(slip, np.array([xm]))[0])
        tx, ty = -math.sin(th_m), math.cos(th_m)
        alpha = math.atan2(ty, tx)  # 底面勾配角

        # 面積（台形近似）
        Nx = 8
        grid = np.linspace(x_a, x_b, Nx+1)
        thg = circle_theta_from_x(slip, grid)
        yu = slip.yc + slip.R*np.sin(thg)
        yl = slip.yc - slip.R*np.sin(thg)
        yg_seg = ground_y_at(grid)
        yc_seg = np.where(np.abs(yg_seg - yu) < np.abs(yg_seg - yl), yu, yl)
        heights = np.maximum(yg_seg - yc_seg, 0.0)
        area = np.trapz(heights, grid)  # m^2（奥行1m）
        W = soil_gamma * area          # kN

        slices.append({
            'alpha': alpha,
            'width': (x_b - x_a),
            'W': W,
            'area': area,
        })

    return slices

# =========================
# Bishop（未補強 / 補強）核
# =========================
def bishop_fs_unreinforced(slices: List[dict], soil: Soil, max_iter: int = 80, tol: float = 1e-5) -> float:
    phi = math.radians(soil.phi); c = soil.c
    Fs = 1.2
    for _ in range(max_iter):
        num = den = 0.0
        for s in slices:
            alpha = s['alpha']; W = s['W']; b = s['width']
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * b + (Np * math.tan(phi))
            num += shear_res
            den += W * math.sin(alpha)
        Fs_new = num / max(den, 1e-9)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)

def bishop_fs_with_nails(
    slices: List[dict],
    soil: Soil,
    slip: CircleSlip,
    nails: List[Nail],
    max_iter: int = 80,
    tol: float = 1e-5
) -> float:
    Fs = bishop_fs_unreinforced(slices, soil, max_iter=30, tol=1e-4)
    phi = math.radians(soil.phi); c = soil.c; R = slip.R

    def circle_tangent_at_xy(xp, yp) -> Tuple[float,float]:
        rx, ry = xp - slip.xc, yp - slip.yc
        tx, ty = -ry, rx
        n = math.hypot(tx, ty)
        return (1.0, 0.0) if n == 0 else (tx/n, ty/n)

    for _ in range(max_iter):
        num = den = 0.0
        M_reinf = 0.0

        for s in slices:
            alpha = s['alpha']; W = s['W']; b = s['width']
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * b + (Np * math.tan(phi))
            num += shear_res
            den += W * math.sin(alpha)

        for nl in nails:
            pts = line_circle_intersections_segment(nl.x1, nl.y1, nl.x2, nl.y2, slip)
            if not pts: 
                continue
            xp, yp = sorted(pts, key=lambda p:(p[0]-slip.xc)**2 + (p[1]-slip.yc)**2)[0]
            tx, ty = circle_tangent_at_xy(xp, yp)
            nx, ny = (nl.x2-nl.x1), (nl.y2-nl.y1)
            nlen = math.hypot(nx, ny)
            if nlen == 0: 
                continue
            nx /= nlen; ny /= nlen
            delta = _angle_between(nx, ny, tx, ty)
            L_embed = 2.0 * nl.embed_length_each_side
            T_axial_1 = min(nl.T_yield, nl.bond_strength * L_embed)   # kN/本
            T_axial   = T_axial_1 / max(nl.spacing, 1e-6)              # kN/m
            T_tan = T_axial * abs(math.cos(delta))
            M_reinf += T_tan * R

        Fs_new = (num * R + M_reinf) / max(den * R, 1e-9)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)
