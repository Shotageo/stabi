# stabi_lem.py
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List, Dict, Optional

DEG = np.pi / 180.0

@dataclass
class Soil:
    gamma: float  # unit weight (kN/m^3)
    c: float      # cohesion (kPa = kN/m^2)
    phi: float    # friction angle (deg)

@dataclass
class Slope:
    H: float  # crest height (m) at x=0
    L: float  # horizontal length to toe (m) at y=0

    @property
    def m(self) -> float:
        # ground line: y = m x + b, passing (0,H) and (L,0)
        return (0 - self.H) / (self.L - 0)

    @property
    def b(self) -> float:
        return self.H

    def y_ground(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.b

    def on_segment(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        # check points are on the segment between (0,H) and (L,0)
        return (x >= -tol) & (x <= self.L + tol) & (np.abs(y - self.y_ground(x)) < 1e-5)

def circle_y(x: np.ndarray, xc: float, yc: float, R: float) -> np.ndarray:
    # lower intersection (slip surface)
    inside = R**2 - (x - xc)**2
    y = np.full_like(x, np.nan, dtype=float)
    ok = inside >= 0.0
    y[ok] = yc - np.sqrt(inside[ok])
    return y

def circle_line_intersections(slope: Slope, xc: float, yc: float, R: float) -> Optional[Tuple[float, float]]:
    # Solve circle (x-xc)^2 + (y-yc)^2 = R^2 and line y = m x + b
    m, b = slope.m, slope.b
    # (x-xc)^2 + (mx+b-yc)^2 = R^2  => ax^2 + bx + c = 0
    A = 1 + m**2
    B = 2*(m*(b - yc) - xc)
    C = xc**2 + (b - yc)**2 - R**2
    disc = B*B - 4*A*C
    if disc <= 0:
        return None
    x1 = (-B - np.sqrt(disc)) / (2*A)
    x2 = (-B + np.sqrt(disc)) / (2*A)
    x_low, x_high = min(x1, x2), max(x1, x2)
    # require crossing within the segment
    y1 = m * x_low + b
    y2 = m * x_high + b
    if slope.on_segment(np.array([x_low, x_high]), np.array([y1, y2])).all():
        return x_low, x_high
    return None

def base_angle_alpha(x: np.ndarray, xc: float, yc: float) -> np.ndarray:
    # tangent angle of circle at (x, y_arc)
    # dy/dx = -(x - xc)/(y - yc); alpha = arctan(dy/dx)
    # compute y on arc:
    y = yc - np.sqrt(np.maximum(0.0, (0 + (x - xc)**2) * 0 + ( (x - xc)**2 )))  # placeholder to keep shape
    # We'll compute y properly where used to avoid duplicated sqrt:
    return np.zeros_like(x)

def fs_fellenius(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
                 n_slices: int = 30) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    # discretize vertical slices between x1..x2
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_ground = slope.y_ground(xmid)
    # slip surface y on circle (lower branch)
    inside = R**2 - (xmid - xc)**2
    if np.any(inside <= 0):
        return None
    y_arc = yc - np.sqrt(inside)

    h = y_ground - y_arc
    if np.any(h <= 0):
        return None  # arc above ground → invalid

    # geometry at base
    # slope of circle: dy/dx = -(x - xc)/(y - yc), evaluated at base point (xmid, y_arc)
    dydx = -(xmid - xc) / (y_arc - yc)
    alpha = np.arctan(dydx)  # base angle to horizontal

    b_len = dx / np.cos(alpha)  # base length along slip
    W = soil.gamma * h * dx     # slice weight (unit thickness)

    c = soil.c
    phi = soil.phi * DEG
    # Fellenius: FS = sum(c*b + W*cosα*tanφ) / sum(W*sinα)
    num = np.sum(c * b_len + W * np.cos(alpha) * np.tan(phi))
    den = np.sum(W * np.sin(alpha))
    if den <= 0:
        return None
    return num / den

def fs_bishop(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
              n_slices: int = 30, max_iter: int = 100, tol: float = 1e-6) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_ground = slope.y_ground(xmid)
    inside = R**2 - (xmid - xc)**2
    if np.any(inside <= 0):
        return None
    y_arc = yc - np.sqrt(inside)
    h = y_ground - y_arc
    if np.any(h <= 0):
        return None

    dydx = -(xmid - xc) / (y_arc - yc)
    alpha = np.arctan(dydx)
    b_len = dx / np.cos(alpha)
    W = soil.gamma * h * dx

    c = soil.c
    phi = soil.phi * DEG
    tanphi = np.tan(phi)

    # Bishop simplified: iterate FS
    FS = 1.2
    for _ in range(max_iter):
        m_alpha = 1.0 + (tanphi * np.tan(alpha)) / FS
        num = np.sum((c * b_len + W * tanphi * np.cos(alpha)) / m_alpha)
        den = np.sum(W * np.sin(alpha))
        if den <= 0:
            return None
        FS_new = num / den
        if np.isnan(FS_new) or FS_new <= 0:
            return None
        if abs(FS_new - FS) < tol:
            return FS_new
        FS = FS_new
    return FS  # may be slightly off if no convergence, but usually converges

def grid_search(
    slope: Slope,
    soil: Soil,
    x_center_range: Tuple[float, float],
    y_center: float,
    R_range: Tuple[float, float],
    nx: int = 15,
    nR: int = 20,
    method: str = "bishop",
    n_slices: int = 30
) -> Dict:
    xcs = np.linspace(x_center_range[0], x_center_range[1], nx)
    Rs = np.linspace(R_range[0], R_range[1], nR)

    records = []
    fs_func = fs_bishop if method.lower().startswith("b") else fs_fellenius

    for xc in xcs:
        for R in Rs:
            fs = fs_func(slope, soil, xc, y_center, R, n_slices=n_slices)
            if fs is None:
                continue
            inter = circle_line_intersections(slope, xc, y_center, R)
            if inter is None:
                continue
            records.append({
                "xc": xc,
                "yc": y_center,
                "R": R,
                "Fs": fs,
                "x1": inter[0],
                "x2": inter[1],
            })
    if not records:
        return {"candidates": [], "best": None}

    # 最小Fsを選択
    best = min(records, key=lambda r: r["Fs"])
    return {"candidates": records, "best": best}
