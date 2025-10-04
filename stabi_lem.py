# stabi_lem.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math
from typing import List, Tuple

# ---- 型定義 ----
@dataclass
class Soil:
    gamma: float   # kN/m3
    c: float       # kPa (kN/m2)
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
    spacing: float            # m（奥行方向の配置間隔：kN/本→kN/m換算用）
    T_yield: float            # kN（1本）
    bond_strength: float      # kN/m（付着耐力：定着長1mあたり）
    embed_length_each_side: float = 0.5  # m（交点から両側の有効定着長：簡易）

# ---- ユーティリティ ----
def _dot(ax, ay, bx, by):
    return ax * bx + ay * by

def _norm(ax, ay):
    return math.hypot(ax, ay)

def _angle_between(ax, ay, bx, by) -> float:
    na = _norm(ax, ay); nb = _norm(bx, by)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, _dot(ax, ay, bx, by) / (na * nb)))
    return math.acos(cosv)

def _line_circle_intersections(x1, y1, x2, y2, xc, yc, R) -> List[Tuple[float, float]]:
    dx, dy = (x2 - x1), (y2 - y1)
    fx, fy = (x1 - xc), (y1 - yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - R*R
    disc = B*B - 4*A*C
    pts = []
    if disc < 0 or A == 0:
        return pts
    sqrt_disc = math.sqrt(disc)
    for s in (-1.0, 1.0):
        t = (-B + s*sqrt_disc) / (2*A)
        if 0.0 <= t <= 1.0:
            px = x1 + t*dx
            py = y1 + t*dy
            pts.append((px, py))
    return pts

def _circle_tangent_at_point(xp, yp, xc, yc):
    rx, ry = xp - xc, yp - yc
    tx, ty = -ry, rx
    n = _norm(tx, ty)
    if n == 0: return (1.0, 0.0)
    return (tx/n, ty/n)

# ---- Bishop簡略法（未補強） ----
def bishop_fs_unreinforced(slices: List[dict], soil: Soil, slip: CircleSlip, water_gamma_red: float = 0.0,
                           max_iter: int = 80, tol: float = 1e-5) -> float:
    """
    slices: {'width':m, 'alpha':rad, 'W':kN, 'area':m2, 'u':kPa(任意)} のリスト想定（奥行1m）
    """
    phi = math.radians(soil.phi)
    c = soil.c
    Fs = 1.2
    for _ in range(max_iter):
        num = 0.0
        den = 0.0
        for s in slices:
            alpha = s['alpha']
            W = max(0.0, s['W'] - water_gamma_red * s.get('area', 0.0))
            b = s['width']
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

# ---- ネイル寄与付き：補強後Fs ----
def bishop_fs_with_nails(slices: List[dict], soil: Soil, slip: CircleSlip, nails: List[Nail],
                         water_gamma_red: float = 0.0, max_iter: int = 80, tol: float = 1e-5) -> float:
    """
    ネイル寄与＝円弧接線方向の抵抗力T_tanをモーメント化し加算：M_reinf = Σ(T_tan * R)
    T_tan = T_axial * |cos(δ)|、T_axial = min(T_yield, bond_strength*L_embed)/spacing
    """
    Fs = bishop_fs_unreinforced(slices, soil, slip, water_gamma_red, max_iter=30, tol=1e-4)

    phi = math.radians(soil.phi)
    c = soil.c
    R = slip.R

    for _ in range(max_iter):
        num = 0.0
        den = 0.0
        M_reinf = 0.0

        for s in slices:
            alpha = s['alpha']
            W = max(0.0, s['W'] - water_gamma_red * s.get('area', 0.0))
            b = s['width']
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * b + (Np * math.tan(phi))
            num += shear_res
            den += W * math.sin(alpha)

        for nail in nails:
            pts = _line_circle_intersections(nail.x1, nail.y1, nail.x2, nail.y2, slip.xc, slip.yc, R)
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda p: (p[0]-slip.xc)**2 + (p[1]-slip.yc)**2)
            xp, yp = pts_sorted[0]

            tx, ty = _circle_tangent_at_point(xp, yp, slip.xc, slip.yc)
            nx, ny = (nail.x2 - nail.x1), (nail.y2 - nail.y1)
            nlen = _norm(nx, ny)
            if nlen == 0: 
                continue
            nx_u, ny_u = nx / nlen, ny / nlen

            delta = _angle_between(nx_u, ny_u, tx, ty)

            L_embed = nail.embed_length_each_side * 2.0
            T_pullout = nail.bond_strength * L_embed           # kN/本
            T_axial_1 = min(nail.T_yield, T_pullout)           # kN/本
            T_axial   = T_axial_1 / max(nail.spacing, 1e-6)    # kN/m（1m幅）

            T_tan = T_axial * abs(math.cos(delta))
            M_reinf += T_tan * R

        Fs_new = (num * R + M_reinf) / max(den * R, 1e-9)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new

    return max(Fs, 1e-6)
