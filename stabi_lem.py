# stabi_lem.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

DEG = np.pi / 180.0
EPS = 1e-12

# ----------------------------- Data -----------------------------
@dataclass
class Soil:
    gamma: float  # kN/m^3
    c: float      # kPa (=kN/m^2)
    phi: float    # deg

@dataclass
class Slope:
    """Straight ground from (0,H) to (L,0).  y = m x + H"""
    H: float
    L: float

    @property
    def m(self) -> float:
        return -self.H / max(self.L, EPS)

    def y_ground(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.m * np.asarray(x) + self.H

# --------------------------- Geometry ---------------------------
def circle_line_intersections_straight_ground(
    slope: Slope, xc: float, yc: float, R: float
) -> Optional[Tuple[float, float]]:
    """円と地表直線の交点 x（区間[0,L]の中に2点そろって存在するときのみ返す）"""
    m, H = slope.m, slope.H
    A = 1 + m*m
    B = 2*(m*(H - yc) - xc)
    C = xc*xc + (H - yc)*(H - yc) - R*R
    disc = B*B - 4*A*C
    if disc <= 0:
        return None
    s = np.sqrt(disc)
    x1 = (-B - s)/(2*A)
    x2 = (-B + s)/(2*A)
    xl, xh = (x1, x2) if x1 < x2 else (x2, x1)
    if xl < -1e-8 or xh > slope.L + 1e-8:
        return None
    return float(xl), float(xh)

def _slice_geometry(xmid: np.ndarray, xc: float, yc: float, R: float):
    inside = R*R - (xmid - xc)**2
    if np.any(inside <= 0):
        return None, None, None
    y_arc = yc - np.sqrt(inside)             # 下側分枝（地表の下）
    denom = y_arc - yc
    denom = np.where(np.abs(denom) < EPS, np.sign(denom)*EPS, denom)
    dydx = -(xmid - xc) / denom              # 円の接線勾配
    alpha = np.arctan(dydx)                  # 底面角
    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0, atol=1e-10)):
        return None, None, None
    return y_arc, alpha, cos_a

def _depth_filters(h: np.ndarray, h_min: float, h_max: float,
                   pct: float, min_eff_ratio: float) -> bool:
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        return False
    if float(np.max(h)) > h_max:
        return False
    eff = h > 1e-4
    if eff.mean() < min_eff_ratio:
        return False
    if pct > 0 and float(np.percentile(h, pct)) < h_min:
        return False
    return True

# ----------------------- Factor of Safety -----------------------
def fs_fellenius(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 40, h_min: float = 0.0, h_max: float = 1e9, pct: float = 0.0,
    min_eff_ratio: float = 0.2
) -> Optional[float]:
    inter = circle_line_intersections_straight_ground(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-8:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5*(xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha, cos_a = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None

    h = y_g - y_arc
    if not _depth_filters(h, h_min, h_max, pct, min_eff_ratio):
        return None

    b = dx / cos_a
    W = soil.gamma * h * dx
    tanp = np.tan(soil.phi * DEG)

    num = np.sum(soil.c * b + W * np.cos(alpha) * tanp)
    den = np.sum(W * np.sin(alpha))
    if den <= 0:
        return None
    Fs = float(num/den)
    if not np.isfinite(Fs) or Fs <= 0:
        return None
    return Fs

def fs_bishop(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 40, max_iter: int = 120, tol: float = 1e-6,
    h_min: float = 0.0, h_max: float = 1e9, pct: float = 0.0,
    min_eff_ratio: float = 0.2
) -> Optional[float]:
    inter = circle_line_intersections_straight_ground(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-8:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5*(xs[:-1] + xs[1:])
    dx = (x2 - x1) / n_slices

    y_g = slope.y_ground(xmid)
    y_arc, alpha, cos_a = _slice_geometry(xmid, xc, yc, R)
    if y_arc is None:
        return None

    h = y_g - y_arc
    if not _depth_filters(h, h_min, h_max, pct, min_eff_ratio):
        return None

    b = dx / cos_a
    W = soil.gamma * h * dx
    tanp = np.tan(soil.phi * DEG)

    Fs = 1.3
    for _ in range(max_iter):
        denom_alpha = 1.0 + (tanp * np.tan(alpha)) / max(Fs, EPS)
        num = np.sum((soil.c * b + W * tanp * np.cos(alpha)) / denom_alpha)
        den = np.sum(W * np.sin(alpha))
        if den <= 0:
            return None
        Fs_new = num/den
        if not np.isfinite(Fs_new) or Fs_new <= 0:
            return None
        if abs(Fs_new - Fs) < tol:
            return float(Fs_new)
        Fs = Fs_new
    return float(Fs)

# ---------------------- Candidate Generation -------------------
def gen_candidates_center_grid(
    slope: Slope,
    x_center_range: Tuple[float, float], y_center_range: Tuple[float, float],
    R_range: Tuple[float, float], nx: int, ny: int, nR: int
) -> List[Dict]:
    xcs = np.linspace(x_center_range[0], x_center_range[1], nx)
    ycs = np.linspace(y_center_range[0], y_center_range[1], ny)
    Rs  = np.linspace(R_range[0], R_range[1], nR)
    recs: List[Dict] = []
    for xc in xcs:
        for yc in ycs:
            for R in Rs:
                inter = circle_line_intersections_straight_ground(slope, float(xc), float(yc), float(R))
                if inter is None:
                    continue
                x1, x2 = inter
                recs.append({"xc": float(xc), "yc": float(yc), "R": float(R),
                             "x1": float(x1), "x2": float(x2)})
    return recs

def evaluate_candidates(
    slope: Slope, soil: Soil, recs: List[Dict], method: str,
    n_slices: int, h_min: float, h_max: float, pct: float, min_eff_ratio: float
) -> List[Dict]:
    fs_fun = fs_bishop if method.lower().startswith("b") else fs_fellenius
    out: List[Dict] = []
    for r in recs:
        fs = fs_fun(slope, soil, r["xc"], r["yc"], r["R"],
                    n_slices=n_slices, h_min=h_min, h_max=h_max,
                    pct=pct, min_eff_ratio=min_eff_ratio)
        if fs is None:
            continue
        rr = dict(r); rr["Fs"] = float(fs)
        out.append(rr)
    return out

def _pack(evals: List[Dict], rescue: str = "") -> Dict:
    if not evals:
        return {"candidates": [], "best": None, "rescue": rescue}
    best = min(evals, key=lambda r: r["Fs"])
    # UI速度のため上位のみ返す
    return {"candidates": sorted(evals, key=lambda r: r["Fs"])[:1200], "best": best, "rescue": rescue}

def search_center_grid(
    slope: Slope, soil: Soil,
    x_center_range: Tuple[float, float], y_center_range: Tuple[float, float],
    R_range: Tuple[float, float], nx: int, ny: int, nR: int,
    method: str, n_slices: int,
    refine_levels: int = 1, top_k: int = 20, shrink: float = 0.5
) -> Dict:
    # 既定：フィルタはゆるめ（ベーシックUIで確実に出す）
    h_min, h_max, pct, min_eff_ratio = 0.0, 1e9, 0.0, 0.2

    # 粗探索
    recs = gen_candidates_center_grid(slope, x_center_range, y_center_range, R_range, nx, ny, nR)
    evals = evaluate_candidates(slope, soil, recs, method, n_slices, h_min, h_max, pct, min_eff_ratio)
    if evals and refine_levels > 0:
        seeds = sorted(evals, key=lambda r: r["Fs"])[:max(1, top_k)]
        xr = x_center_range; yr = y_center_range; rr = R_range
        all_e = list(evals)
        for lvl in range(refine_levels):
            f = shrink ** (lvl + 1)
            dx = 0.5*(xr[1]-xr[0])*f; dy = 0.5*(yr[1]-yr[0])*f; dR = 0.5*(rr[1]-rr[0])*f
            for s in seeds:
                x_rng = (s["xc"]-dx, s["xc"]+dx)
                y_rng = (max(0.1, s["yc"]-dy), s["yc"]+dy)
                R_rng = (max(0.5*s["R"], rr[0]), min(1.5*s["R"], rr[1]))
                recs2 = gen_candidates_center_grid(slope, x_rng, y_rng, R_rng,
                                                   max(6, nx//2), max(6, ny//2), max(8, nR//2))
                evals2 = evaluate_candidates(slope, soil, recs2, method, n_slices, h_min, h_max, pct, min_eff_ratio)
                all_e.extend(evals2)
            seeds = sorted(all_e, key=lambda r: r["Fs"])[:max(1, top_k)]
        return _pack(all_e)

    if evals:
        return _pack(evals)

    # -------- Rescue（広域＆Felleniusで必ず拾う） --------
    diag = float(np.hypot(slope.H, slope.L))
    recsR = gen_candidates_center_grid(
        slope,
        x_center_range=( -0.8*slope.L, 0.4*slope.L ),
        y_center_range=( 0.3*slope.H,  3.0*slope.H ),
        R_range=( max(0.2*diag, 1.0), 4.0*diag ),
        nx=22, ny=14, nR=40
    )
    evalsR = evaluate_candidates(slope, soil, recsR, "fellenius", max(30, n_slices//2), 0.0, 1e9, 0.0, 0.15)
    return _pack(evalsR, rescue="center-grid-wide-fellenius")