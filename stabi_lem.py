# stabi_lem.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math
from typing import List, Tuple, Callable
import numpy as np

# ===== 基本データ構造 =====
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
    x1: float; y1: float
    x2: float; y2: float
    spacing: float            # m（1m幅換算）
    T_yield: float            # kN/本
    bond_strength: float      # kN/m
    embed_length_each_side: float = 0.5  # m

# ===== 幾何ユーティリティ =====
def _dot(ax, ay, bx, by): return ax * bx + ay * by
def _norm(ax, ay): return math.hypot(ax, ay)
def _angle_between(ax, ay, bx, by):
    na = _norm(ax, ay); nb = _norm(bx, by)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, _dot(ax, ay, bx, by) / (na * nb)))
    return math.acos(cosv)

def circle_xy_from_theta(slip: CircleSlip, theta: np.ndarray):
    return slip.xc + slip.R*np.cos(theta), slip.yc + slip.R*np.sin(theta)

def circle_theta_from_x(slip: CircleSlip, x: np.ndarray):
    v = np.clip((x - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)

def line_circle_intersections_segment(x1, y1, x2, y2, slip: CircleSlip) -> List[Tuple[float,float]]:
    dx, dy = (x2 - x1), (y2 - y1)
    fx, fy = (x1 - slip.xc), (y1 - slip.yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - slip.R*slip.R
    disc = B*B - 4*A*C
    if A == 0 or disc < 0: return []
    rt = math.sqrt(disc); pts = []
    for s in (-1.0, 1.0):
        t = (-B + s*rt) / (2*A)
        if 0.0 <= t <= 1.0:
            pts.append((x1 + t*dx, y1 + t*dy))
    return pts

# ===== スライス生成（ロバスト化） =====
def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    """
    地表 y_g(x) と円弧の上下解 y_u(x), y_l(x) を比較。
    - まず「地表より下側（<= y_g）」の解だけを候補にする
    - 両方下なら、地表により近い側（=値が大きい方）を選ぶ
    - 分断された場合は“最長の連続区間”だけ採用
    """
    X = np.linspace(x_min, x_max, 1201)
    theta = circle_theta_from_x(slip, X)
    yu = slip.yc + slip.R*np.sin(theta)
    yl = slip.yc - slip.R*np.sin(theta)
    yg = ground_y_at(X)

    under_u = yu <= yg
    under_l = yl <= yg
    yc = np.where(
        under_u & under_l, np.maximum(yu, yl),
        np.where(under_u, yu, np.where(under_l, yl, np.nan))
    )

    valid = np.isfinite(yc)
    if not np.any(valid): return []

    idx = np.where(valid)[0]
    # 連続ブロック抽出
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks+1]
    ends   = np.r_[breaks, len(idx)-1]
    lengths = idx[ends] - idx[starts]
    k = int(np.argmax(lengths))
    i0, i1 = idx[starts[k]], idx[ends[k]]
    xL, xR = X[i0], X[i1]
    if xR - xL <= 1e-6: return []

    xs = np.linspace(xL, xR, n_slices+1)
    slices = []
    for i in range(n_slices):
        x_a, x_b = xs[i], xs[i+1]
        xm = 0.5*(x_a + x_b)

        # 接線角
        th_m = float(circle_theta_from_x(slip, np.array([xm]))[0])
        tx, ty = -math.sin(th_m), math.cos(th_m)
        alpha = math.atan2(ty, tx)

        # 面積（台形積分）
        Nx = 8
        grid = np.linspace(x_a, x_b, Nx+1)
        thg = circle_theta_from_x(slip, grid)
        yu_g = slip.yc + slip.R*np.sin(thg)
        yl_g = slip.yc - slip.R*np.sin(thg)
        yg_seg = ground_y_at(grid)
        under_u_g = yu_g <= yg_seg
        under_l_g = yl_g <= yg_seg
        yc_seg = np.where(
            under_u_g & under_l_g, np.maximum(yu_g, yl_g),
            np.where(under_u_g, yu_g, np.where(under_l_g, yl_g, yg_seg))
        )
        heights = np.maximum(yg_seg - yc_seg, 0.0)
        area = np.trapz(heights, grid)  # m^2
        W = soil_gamma * area           # kN（奥行1m）

        slices.append({'alpha': alpha, 'width': (x_b-x_a), 'W': W, 'area': area})
    return slices

# ===== Bishop（未補強 / 補強） =====
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
            num += shear_res; den += W * math.sin(alpha)
        Fs_new = num / max(den, 1e-9)
        if abs(Fs_new - Fs) < tol: return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)

def bishop_fs_with_nails(
    slices: List[dict], soil: Soil, slip: CircleSlip, nails: List[Nail],
    max_iter: int = 80, tol: float = 1e-5
) -> float:
    Fs = bishop_fs_unreinforced(slices, soil, max_iter=30, tol=1e-4)
    phi = math.radians(soil.phi); c = soil.c; R = slip.R

    def _tangent_at(xp, yp):
        rx, ry = xp - slip.xc, yp - slip.yc
        tx, ty = -ry, rx
        n = math.hypot(tx, ty)
        return (1.0, 0.0) if n == 0 else (tx/n, ty/n)

    for _ in range(max_iter):
        num = den = 0.0; M_reinf = 0.0
        for s in slices:
            alpha = s['alpha']; W = s['W']; b = s['width']
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * b + (Np * math.tan(phi))
            num += shear_res; den += W * math.sin(alpha)

        for nl in nails:
            pts = line_circle_intersections_segment(nl.x1, nl.y1, nl.x2, nl.y2, slip)
            if not pts: 
                continue
            xp, yp = sorted(pts, key=lambda p:(p[0]-slip.xc)**2 + (p[1]-slip.yc)**2)[0]
            tx, ty = _tangent_at(xp, yp)
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
        if abs(Fs_new - Fs) < tol: return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)
