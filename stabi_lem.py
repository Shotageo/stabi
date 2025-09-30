# stabi_lem.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

DEG = np.pi / 180.0
EPS = 1e-12

@dataclass
class Soil:
    gamma: float  # unit weight (kN/m^3)
    c: float      # cohesion (kPa = kN/m^2)
    phi: float    # friction angle (deg)

@dataclass
class Slope:
    """Straight ground line from (0,H) to (L,0)."""
    H: float
    L: float

    @property
    def m(self) -> float:
        # y = m x + H  through (0,H) and (L,0)
        return -self.H / max(self.L, EPS)

    def y_ground(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.H

    def on_segment(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return (x >= -tol) & (x <= self.L + tol) & (np.abs(y - self.y_ground(x)) < 1e-5)

# ---------- 幾何ユーティリティ ----------
def circle_line_intersections(slope: Slope, xc: float, yc: float, R: float) -> Optional[Tuple[float, float]]:
    """Intersection of circle (xc,yc,R) with ground segment y = m x + H (0<=x<=L)."""
    if R <= 0.0:
        return None
    m, H = slope.m, slope.H
    A = 1 + m**2
    B = 2 * (m * (H - yc) - xc)
    C = xc**2 + (H - yc)**2 - R**2
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    s = float(np.sqrt(disc))
    x1 = (-B - s) / (2 * A)
    x2 = (-B + s) / (2 * A)
    xl, xh = (x1, x2) if x1 < x2 else (x2, x1)
    y1, y2 = m * xl + H, m * xh + H
    if slope.on_segment(np.array([xl, xh]), np.array([y1, y2])).all():
        return float(xl), float(xh)
    return None

def _slice_geometry(xmid: np.ndarray, xc: float, yc: float, R: float):
    """Lower arc ordinates and base angle alpha (to horizontal) at each slice mid."""
    inside = R**2 - (xmid - xc)**2
    if np.any(inside <= 0):
        return None, None
    y_arc = yc - np.sqrt(inside)  # lower branch only
    # slope dy/dx of the circle at (xmid, y_arc)
    denom = (y_arc - yc)
    denom = np.where(np.abs(denom) < EPS, np.sign(denom) * EPS, denom)
    dydx = -(xmid - xc) / denom
    alpha = np.arctan(dydx)
    return y_arc, alpha

# ---------- 品質・妥当性フィルタ ----------
def _depth_filter(h: np.ndarray, h_min: float, h_max: float, pct: float, min_effective_ratio: float = 0.6) -> bool:
    """Practical filter:
       - percentile(pct) >= h_min（端部スライスの極薄に過度反応しない）
       - max(h) <= h_max
       - 有効スライス（h>1e-4）の割合が十分（>= min_effective_ratio）
    """
    if np.any(~np.isfinite(h)):
        return False
    if float(np.max(h)) > h_max:
        return False
    eff = h > 1e-4
    if eff.mean() < min_effective_ratio:
        return False
    if pct <= 0.0:
        return True
    return float(np.percentile(h, pct)) >= h_min

# ---------- FS計算（Fellenius / Bishop簡略） ----------
def fs_fellenius(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 30, h_min: float = 0.0, h_max: float = 1e9, pct: float = 15.0
) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-6:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None
    h = y_g - y_arc
    if np.any(h <= 0) or not _depth_filter(h, h_min, h_max, pct):
        return None

    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0, atol=1e-8)):
        return None
    b = dx / cos_a

    W = soil.gamma * h * dx                 # kN (per unit thickness)
    tanp = np.tan(soil.phi * DEG)
    num = np.sum(soil.c * b + W * np.cos(alpha) * tanp)
    den = np.sum(W * np.sin(alpha))
    if den <= 0:
        return None
    FS = float(num / den)
    return FS if np.isfinite(FS) and FS > 0 else None

def fs_bishop(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 30, max_iter: int = 100, tol: float = 1e-6,
    h_min: float = 0.0, h_max: float = 1e9, pct: float = 15.0
) -> Optional[float]:
    inter = circle_line_intersections(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-6:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None
    h = y_g - y_arc
    if np.any(h <= 0) or not _depth_filter(h, h_min, h_max, pct):
        return None

    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0, atol=1e-8)):
        return None
    b = dx / cos_a

    W = soil.gamma * h * dx
    tanp = np.tan(soil.phi * DEG)

    FS = 1.2
    for _ in range(max_iter):
        m_alpha = 1.0 + (tanp * np.tan(alpha)) / max(FS, EPS)
        num = np.sum((soil.c * b + W * tanp * np.cos(alpha)) / m_alpha)
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

# ---------- グリッド探索（基礎） ----------
def grid_search_2d(
    slope: Slope, soil: Soil,
    x_center_range: Tuple[float, float], y_center_range: Tuple[float, float], R_range: Tuple[float, float],
    nx: int = 16, ny: int = 10, nR: int = 24,
    method: str = "bishop", n_slices: int = 40,
    h_min: float = 0.2, h_max: float = 1e9, pct: float = 15.0
) -> Dict:
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
                x1, x2 = circle_line_intersections(slope, float(xc), float(yc), float(R)) or (None, None)
                if x1 is None:
                    continue
                recs.append({"xc": float(xc), "yc": float(yc), "R": float(R),
                             "Fs": float(fs), "x1": float(x1), "x2": float(x2)})
    if not recs:
        return {"candidates": [], "best": None}
    best = min(recs, key=lambda r: r["Fs"])
    return {"candidates": recs, "best": best}

# ---------- 逐次絞り込み探索（coarse→refine×L） ----------
def grid_search_adaptive(
    slope: Slope, soil: Soil,
    x_center_range: Tuple[float, float], y_center_range: Tuple[float, float], R_range: Tuple[float, float],
    nx: int = 16, ny: int = 10, nR: int = 24,
    refine_levels: int = 2, top_k: int = 8, shrink: float = 0.45,
    method: str = "bishop", n_slices: int = 40,
    h_min: float = 0.2, h_max: float = 1e9, pct: float = 15.0
) -> Dict:
    """Two-stage+ adaptive search around top-K seeds."""
    # 1) coarse
    out = grid_search_2d(slope, soil, x_center_range, y_center_range, R_range,
                         nx, ny, nR, method, n_slices, h_min, h_max, pct)
    cands = out["candidates"]
    if not cands:
        return out

    seeds = sorted(cands, key=lambda r: r["Fs"])[:top_k]
    xr, yr, rr = (x_center_range[1] - x_center_range[0]), (y_center_range[1] - y_center_range[0]), (R_range[1] - R_range[0])
    all_recs = cands[:]

    for lvl in range(refine_levels):
        dx = max(xr / max(nx-1, 1), EPS) * (shrink ** lvl)
        dy = max(yr / max(ny-1, 1), EPS) * (shrink ** lvl)
        dR = max(rr / max(nR-1, 1), EPS) * (shrink ** lvl)

        for s in seeds:
            xrng = (s["xc"] - 3*dx, s["xc"] + 3*dx)
            yrng = (s["yc"] - 3*dy, s["yc"] + 3*dy)
            rrng = (max(1.0, s["R"] - 3*dR), s["R"] + 3*dR)
            res = grid_search_2d(slope, soil, xrng, yrng, rrng,
                                 nx=min(11, nx+2), ny=min(9, ny+2), nR=min(11, nR+2),
                                 method=method, n_slices=n_slices, h_min=h_min, h_max=h_max, pct=pct)
            all_recs.extend(res["candidates"])
        # 次段の種を更新
        if all_recs:
            seeds = sorted(all_recs, key=lambda r: r["Fs"])[:top_k]

    if not all_recs:
        return {"candidates": [], "best": None}
    best = min(all_recs, key=lambda r: r["Fs"])
    # limit returned candidates for UI speed
    all_recs = sorted(all_recs, key=lambda r: r["Fs"])[:1200]
    return {"candidates": all_recs, "best": best}
