# stabi_lem.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

DEG = np.pi / 180.0

@dataclass
class Soil:
    gamma: float  # unit weight (kN/m^3)
    c: float      # cohesion (kPa = kN/m^2)
    phi: float    # friction angle (deg)

@dataclass
class Slope:
    H: float  # crest height at x=0
    L: float  # horizontal length to toe at y=0

    @property
    def m(self) -> float:
        # ground line: y = m x + H through (0,H) and (L,0)
        return -self.H / self.L

    def y_ground(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.H

    def on_segment(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (x >= -tol) & (x <= self.L + tol) & (np.abs(y - self.y_ground(x)) < 1e-5)

def circle_line_intersections(slope: Slope, xc: float, yc: float, R: float) -> Optional[Tuple[float, float]]:
    """Intersection of circle (xc,yc,R) with slope line y = m x + H, restricted to the segment [0,L]."""
    m, H = slope.m, slope.H
    # (x-xc)^2 + (mx+H - yc)^2 = R^2 -> A x^2 + B x + C = 0
    A = 1 + m**2
    B = 2*(m*(H - yc) - xc)
    C = xc**2 + (H - yc)**2 - R**2
    disc = B*B - 4*A*C
    if disc <= 0:
        return None
    sqrt_disc = np.sqrt(disc)
    x1 = (-B - sqrt_disc) / (2*A)
    x2 = (-B + sqrt_disc) / (2*A)
    x_low, x_high = (x1, x2) if x1 < x2 else (x2, x1)
    # must cut the ground segment in two points
    y1 = m * x_low + H
    y2 = m * x_high + H
    if slope.on_segment(np.array([x_low, x_high]), np.array([y1, y2])).all():
        return x_low, x_high
    return None

def _slice_geometry(xmid: np.ndarray, xc: float, yc: float, R: float):
    """Return y_arc (lower branch), base angle alpha, and guard arrays."""
    inside = R**2 - (xmid - xc)**2
    if np.any(inside <= 0):
        return None, None, None
    y_arc = yc - np.sqrt(inside)  # lower branch
    # slope of circle at base: dy/dx = -(x - xc)/(y - yc)
    dydx = -(xmid - xc) / (y_arc - yc)
    alpha = np.arctan(dydx)  # base inclination to horizontal
    return y_arc, alpha, inside

def fs_fellenius(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
                 n_slices: int = 30, min_depth: float = 0.0, max_depth: float = 1e9) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha, inside = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None
    h = y_g - y_arc
    if np.any(h <= 0):
        return None

    # depth filter to avoid silly very-shallow/very-deep circles
    h_min = float(np.min(h))
    h_max = float(np.max(h))
    if h_min < min_depth or h_max > max_depth:
        return None

    # base length along arc
    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0)):
        return None
    b_len = dx / cos_a

    W = soil.gamma * h * dx  # kN per unit thickness
    c = soil.c
    tanphi = np.tan(soil.phi * DEG)
    num = np.sum(c * b_len + W * np.cos(alpha) * tanphi)
    den = np.sum(W * np.sin(alpha))
    if den <= 0:
        return None
    return num / den

def fs_bishop(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
              n_slices: int = 30, max_iter: int = 100, tol: float = 1e-6,
              min_depth: float = 0.0, max_depth: float = 1e9) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha, inside = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None

    h = y_g - y_arc
    if np.any(h <= 0):
        return None

    # depth filter
    h_min = float(np.min(h))
    h_max = float(np.max(h))
    if h_min < min_depth or h_max > max_depth:
        return None

    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0)):
        return None
    b_len = dx / cos_a

    W = soil.gamma * h * dx
    c = soil.c
    tanphi = np.tan(soil.phi * DEG)

    FS = 1.2
    for _ in range(max_iter):
        m_alpha = 1.0 + (tanphi * np.tan(alpha)) / FS
        num = np.sum((c * b_len + W * tanphi * np.cos(alpha)) / m_alpha)
        den = np.sum(W * np.sin(alpha))
        if den <= 0:
            return None
        FS_new = num / den
        if not np.isfinite(FS_new) or FS_new <= 0:
            return None
        if abs(FS_new - FS) < tol:
            return FS_new
        FS = FS_new
    return FS

def grid_search_2d(
    slope: Slope,
    soil: Soil,
    x_center_range: Tuple[float, float],
    y_center_range: Tuple[float, float],
    R_range: Tuple[float, float],
    nx: int = 16,
    ny: int = 10,
    nR: int = 24,
    method: str = "bishop",
    n_slices: int = 40,
    min_depth: float = 1.0,
    max_depth: float = 1e9
) -> Dict:
    """2D grid over (xc, yc) and radius. Return candidates and the min-Fs record."""
    xcs = np.linspace(x_center_range[0], x_center_range[1], nx)
    ycs = np.linspace(y_center_range[0], y_center_range[1], ny)
    Rs  = np.linspace(R_range[0], R_range[1], nR)

    fs_func = fs_bishop if method.lower().startswith("b") else fs_fellenius
    records: List[Dict] = []

    for xc in xcs:
        for yc in ycs:
            for R in Rs:
                fs = fs_func(slope, soil, xc, yc, R, n_slices=n_slices,
                             min_depth=min_depth, max_depth=max_depth)
                if fs is None:
                    continue
                inter = circle_line_intersections(slope, xc, yc, R)
                if inter is None:
                    continue
                x1, x2 = inter
                records.append({
                    "xc": float(xc),
                    "yc": float(yc),
                    "R": float(R),
                    "Fs": float(fs),
                    "x1": float(x1),
                    "x2": float(x2),
                })
    if not records:
        return {"candidates": [], "best": None}
    best = min(records, key=lambda r: r["Fs"])
    return {"candidates": records, "best": best}
