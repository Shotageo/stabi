# stabi_lem.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

DEG = np.pi / 180.0

@dataclass
class Soil:
    gamma: float  # kN/m^3
    c: float      # kPa (=kN/m^2)
    phi: float    # deg

@dataclass
class Slope:
    H: float  # crest at (0,H)
    L: float  # toe at (L,0)

    @property
    def m(self) -> float:
        return -self.H / self.L  # ground y = m x + H

    def y_ground(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.H

    def on_segment(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (x >= -tol) & (x <= self.L + tol) & (np.abs(y - self.y_ground(x)) < 1e-5)

def circle_line_intersections(slope: Slope, xc: float, yc: float, R: float) -> Optional[Tuple[float, float]]:
    m, H = slope.m, slope.H
    A = 1 + m**2
    B = 2*(m*(H - yc) - xc)
    C = xc**2 + (H - yc)**2 - R**2
    disc = B*B - 4*A*C
    if disc <= 0:
        return None
    s = np.sqrt(disc)
    x1 = (-B - s) / (2*A)
    x2 = (-B + s) / (2*A)
    xl, xh = (x1, x2) if x1 < x2 else (x2, x1)
    y1, y2 = m*xl + H, m*xh + H
    if slope.on_segment(np.array([xl, xh]), np.array([y1, y2])).all():
        return float(xl), float(xh)
    return None

def _slice_geometry(xmid: np.ndarray, xc: float, yc: float, R: float):
    inside = R**2 - (xmid - xc)**2
    if np.any(inside <= 0):
        return None, None
    y_arc = yc - np.sqrt(inside)                 # lower branch
    dydx = -(xmid - xc) / (y_arc - yc)           # dy/dx on circle
    alpha = np.arctan(dydx)                      # base angle to horizontal
    return y_arc, alpha

def _depth_ok(h: np.ndarray, h_min: float, h_max: float, pct: float) -> bool:
    if np.any(~np.isfinite(h)):
        return False
    if float(np.max(h)) > h_max:
        return False
    if pct <= 0.0:
        return True
    return float(np.percentile(h, pct)) >= h_min

def fs_fellenius(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
                 n_slices: int = 30, h_min: float = 0.0, h_max: float = 1e9, pct: float = 15.0) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None
    h = y_g - y_arc
    if np.any(h <= 0) or not _depth_ok(h, h_min, h_max, pct):
        return None

    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0)):
        return None
    b = dx / cos_a

    W = soil.gamma * h * dx
    tanp = np.tan(soil.phi * DEG)
    num = np.sum(soil.c * b + W * np.cos(alpha) * tanp)
    den = np.sum(W * np.sin(alpha))
    if den <= 0:
        return None
    return float(num / den)

def fs_bishop(slope: Slope, soil: Soil, xc: float, yc: float, R: float,
              n_slices: int = 30, max_iter: int = 100, tol: float = 1e-6,
              h_min: float = 0.0, h_max: float = 1e9, pct: float = 15.0) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None
    h = y_g - y_arc
    if np.any(h <= 0) or not _depth_ok(h, h_min, h_max, pct):
        return None

    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0)):
        return None
    b = dx / cos_a

    W = soil.gamma * h * dx
    tanp = np.tan(soil.phi * DEG)

    FS = 1.2
    for _ in range(max_iter):
        m = 1.0 + (tanp * np.tan(alpha)) / FS
        num = np.sum((soil.c * b + W * tanp * np.cos(alpha)) / m)
        den = np.sum(W * np.sin(alpha))
        if den <= 0:
            return None
        FS_new = num / den
        if not np.isfinite(FS_new) or FS_new <= 0:
            return None
        if abs(FS_new - FS) < tol:
            return float(FS_new)
        FS = FS_new
    return float(FS)

def grid_search_2d(slope: Slope, soil: Soil,
                   x_center_range: Tuple[float, float],
                   y_center_range: Tuple[float, float],
                   R_range: Tuple[float, float],
                   nx: int = 16, ny: int = 10, nR: int = 24,
                   method: str = "bishop", n_slices: int = 40,
                   h_min: float = 0.2, h_max: float = 1e9, pct: float = 15.0) -> Dict:
    xcs = np.linspace(x_center_range[0], x_center_range[1], nx)
    ycs = np.linspace(y_center_range[0], y_center_range[1], ny)
    Rs  = np.linspace(R_range[0], R_range[1], nR)
    fs_fun = fs_bishop if method.lower().startswith("b") else fs_fellenius

    recs: List[Dict] = []
    for xc in xcs:
        for yc in ycs:
            for R in Rs:
                fs = fs_fun(slope, soil, float(xc), float(yc), float(R),
                            n_slices=n_slices, h_min=h_min, h_max=h_max, pct=pct)
                if fs is None:
                    continue
                inter = circle_line_intersections(slope, float(xc), float(yc), float(R))
                if inter is None:
                    continue
                x1, x2 = inter
                recs.append({"xc": float(xc), "yc": float(yc), "R": float(R),
                             "Fs": float(fs), "x1": float(x1), "x2": float(x2)})
    if not recs:
        return {"candidates": [], "best": None}
    best = min(recs, key=lambda r: r["Fs"])
    return {"candidates": recs, "best": best}
