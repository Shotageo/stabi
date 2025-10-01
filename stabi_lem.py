# stabi_lem.py — (unchanged parts omitted for brevity) —
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import numpy as np, math

DEG = math.pi / 180.0
EPS = 1e-12

@dataclass
class Soil:
    gamma: float   # kN/m3
    c: float       # kPa
    phi: float     # deg

@dataclass
class GroundPL:
    X: np.ndarray
    Y: np.ndarray
    def y_at(self, x):
        x = np.asarray(x, dtype=float)
        y = np.empty_like(x)
        for i, xv in np.ndenumerate(x):
            if xv <= self.X[0]:
                y[i] = self.Y[0]; continue
            if xv >= self.X[-1]:
                y[i] = self.Y[-1]; continue
            k = np.searchsorted(self.X, xv) - 1
            x0, x1 = self.X[k], self.X[k+1]
            y0, y1 = self.Y[k], self.Y[k+1]
            t = (xv - x0) / max(x1 - x0, 1e-12)
            y[i] = (1 - t) * y0 + t * y1
        return y if y.shape != () else float(y)

# ……（交点計算・補助関数は既存どおり）……

def circle_segment_intersections(xc, yc, R, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    a = dx*dx + dy*dy
    b = 2*((x0 - xc)*dx + (y0 - yc)*dy)
    c = (x0 - xc)**2 + (y0 - yc)**2 - R*R
    disc = b*b - 4*a*c
    if disc < 0: return []
    s = math.sqrt(max(0.0, disc))
    out = []
    for t in ((-b - s)/(2*a), (-b + s)/(2*a)):
        if -1e-10 <= t <= 1 + 1e-10:
            out.append((float(x0 + t*dx), float(y0 + t*dy)))
    uniq = []
    for p in out:
        if not any(abs(p[0]-q[0])<1e-9 and abs(p[1]-q[1])<1e-9 for q in uniq):
            uniq.append(p)
    return uniq

def circle_polyline_intersections(xc, yc, R, pl: GroundPL) -> List[Tuple[float,float]]:
    pts = []
    for i in range(len(pl.X)-1):
        pts.extend(circle_segment_intersections(xc, yc, R, pl.X[i], pl.Y[i], pl.X[i+1], pl.Y[i+1]))
    pts = sorted(pts, key=lambda p: p[0])
    out = []
    for p in pts:
        if not out or abs(p[0]-out[-1][0])>1e-8 or abs(p[1]-out[-1][1])>1e-8:
            out.append(p)
    return out

def arc_sample_poly_best_pair(pl: GroundPL, xc, yc, R, n=241):
    pts = circle_polyline_intersections(xc, yc, R, pl)
    if len(pts) < 2: return None
    best = None
    for i in range(len(pts)-1):
        (x1,y1),(x2,y2) = pts[i], pts[i+1]
        if x2 - x1 <= 1e-10:
            continue
        xs = np.linspace(x1, x2, n)
        inside = R*R - (xs - xc)**2
        if np.any(inside <= 0): 
            continue
        ys = yc - np.sqrt(inside)
        h  = pl.y_at(xs) - ys
        if np.any(h <= 0) or np.any(~np.isfinite(h)):
            continue
        dmax = float(np.max(h))
        if (best is None) or (dmax > best[-1]):
            best = (x1, x2, xs, ys, h, dmax)
    if best is None: return None
    x1, x2, xs, ys, h, _ = best
    return x1, x2, xs, ys, h

def _alpha_cos(xc, yc, R, xs):
    inside = R*R - (xs - xc)**2
    y_arc = yc - np.sqrt(inside)
    denom = y_arc - yc
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom)*1e-12, denom)
    dydx  = -(xs - xc) / denom
    alpha = -np.arctan(dydx)
    return alpha, np.cos(alpha), y_arc

def clip_interfaces_to_ground(ground: GroundPL, interfaces: List[GroundPL], x) -> List[np.ndarray]:
    Yg = ground.y_at(x)
    ys_list = []
    y_top = Yg
    for pl_if in interfaces:
        yi = np.minimum(pl_if.y_at(x), y_top)
        ys_list.append(yi)
        y_top = yi
    return ys_list

def barrier_y_from_flags(Yifs: List[np.ndarray], allow_cross: List[bool]) -> np.ndarray:
    if not Yifs or not allow_cross:
        return np.full_like(Yifs[0] if Yifs else np.array([0.0]), -1e9, dtype=float)
    assert len(allow_cross) >= len(Yifs)
    blocked = [Yifs[j] for j in range(len(Yifs)) if not allow_cross[j]]
    if not blocked:
        return np.full_like(Yifs[0], -1e9, dtype=float)
    B = blocked[0].copy()
    for arr in blocked[1:]:
        B = np.maximum(B, arr)
    return B

def base_soil_vectors_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil],
                            xmid: np.ndarray, y_arc: np.ndarray):
    nL = len(soils)
    Yifs = clip_interfaces_to_ground(ground, interfaces[:max(0, nL-1)], xmid)
    if nL == 1:
        gamma = np.full_like(xmid, soils[0].gamma, dtype=float)
        c     = np.full_like(xmid, soils[0].c,     dtype=float)
        phi   = np.full_like(xmid, soils[0].phi,   dtype=float)
        return gamma, c, phi
    if nL == 2:
        Y1 = Yifs[0]
        mask1 = (y_arc >= Y1)
        gamma = np.where(mask1, soils[0].gamma, soils[1].gamma)
        c     = np.where(mask1, soils[0].c,     soils[1].c)
        phi   = np.where(mask1, soils[0].phi,   soils[1].phi)
        return gamma.astype(float), c.astype(float), phi.astype(float)
    Y1, Y2 = Yifs[0], Yifs[1]
    mask1 = (y_arc >= Y1)
    mask3 = (y_arc <  Y2)
    mask2 = ~(mask1 | mask3)
    gamma = np.where(mask1, soils[0].gamma, np.where(mask2, soils[1].gamma, soils[2].gamma))
    c     = np.where(mask1, soils[0].c,     np.where(mask2, soils[1].c,     soils[2].c))
    phi   = np.where(mask1, soils[0].phi,   np.where(mask2, soils[1].phi,   soils[2].phi))
    return gamma.astype(float), c.astype(float), phi.astype(float)

# ---------- Fellenius / Bishop（既存と同じ） ----------
def fs_fellenius_poly_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                            xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = ground.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    Yifs = clip_interfaces_to_ground(ground, interfaces[:max(0, len(soils)-1)], xmid)
    B    = barrier_y_from_flags(Yifs, allow_cross[:max(0, len(soils)-1)])
    if np.any(y_arc < B - 1e-9): return None
    gamma, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    b     = dx / cos_a
    W     = gamma * hmid * dx
    tanp  = np.tan(phi*DEG)
    num   = float(np.sum(c*b + W*np.cos(alpha)*tanp))
    den   = float(np.sum(W*np.sin(alpha)))
    if den <= 0: return None
    Fs    = num/den
    return Fs if (np.isfinite(Fs) and Fs>0) else None

def fs_bishop_poly_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                         xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = ground.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    Yifs = clip_interfaces_to_ground(ground, interfaces[:max(0, len(soils)-1)], xmid)
    B    = barrier_y_from_flags(Yifs, allow_cross[:max(0, len(soils)-1)])
    if np.any(y_arc < B - 1e-9): return None
    gamma, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    b     = dx / cos_a
    W     = gamma * hmid * dx
    tanp  = np.tan(phi*DEG)
    Fs    = 1.3
    for _ in range(120):
        denom_a = 1.0 + (tanp*np.tan(alpha))/max(Fs,1e-12)
        num = float(np.sum((c*b + W*tanp*np.cos(alpha))/denom_a))
        den = float(np.sum(W*np.sin(alpha)))
        if den <= 0: return None
        Fs_new = num/den
        if not (np.isfinite(Fs_new) and Fs_new>0): return None
        if abs(Fs_new-Fs) < 1e-6: return float(Fs_new)
        Fs = Fs_new
    return float(Fs)

def fs_given_R_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                     method: str, xc: float, yc: float, R: float, n_slices: int) -> Optional[float]:
    if method.lower().startswith("b"):
        return fs_bishop_poly_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices)
    else:
        return fs_fellenius_poly_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices)

# ---------- 追加：必要抑止力計算に使う「D=Σ(W sinα)」を返す ----------
def driving_sum_for_R_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                            xc, yc, R, n_slices=40) -> Optional[Tuple[float,float,float]]:
    """
    戻り値: (D_sum, x1, x2)
      D_sum = Σ (W_i * sinα_i) [kN/m] …… Felleniusの分母（駆動項）
      x1, x2 = その円弧の地表との交点のx（描画用）
    """
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = ground.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    Yifs = clip_interfaces_to_ground(ground, interfaces[:max(0, len(soils)-1)], xmid)
    B    = barrier_y_from_flags(Yifs, allow_cross[:max(0, len(soils)-1)])
    if np.any(y_arc < B - 1e-9): return None
    gamma, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    W     = gamma * hmid * dx
    D_sum = float(np.sum(W*np.sin(alpha)))
    if D_sum <= 0 or not np.isfinite(D_sum): return None
    return D_sum, float(x1), float(x2)

# ……（サンプル地形などは既存どおり）……
def make_ground_example(H: float, L: float) -> GroundPL:
    X = np.array([0.0, 0.30*L, 0.63*L, L], dtype=float)
    Y = np.array([H,   0.88*H, 0.46*H, 0.0], dtype=float)
    return GroundPL(X=X, Y=Y)

def make_interface1_example(H: float, L: float) -> GroundPL:
    X = np.array([0.0, 0.35*L, 0.70*L, L], dtype=float)
    Y = np.array([0.70*H, 0.60*H, 0.38*H, 0.20*H], dtype=float)
    return GroundPL(X=X, Y=Y)

def make_interface2_example(H: float, L: float) -> GroundPL:
    X = np.array([0.0, 0.40*L, 0.75*L, L], dtype=float)
    Y = np.array([0.45*H, 0.38*H, 0.22*H, 0.10*H], dtype=float)
    return GroundPL(X=X, Y=Y)

__all__ = [
    "Soil", "GroundPL",
    "make_ground_example", "make_interface1_example", "make_interface2_example",
    "arcs_from_center_by_entries_multi", "fs_given_R_multi",
    "fs_bishop_poly_multi", "fs_fellenius_poly_multi",
    "clip_interfaces_to_ground", "barrier_y_from_flags", "base_soil_vectors_multi",
    "arc_sample_poly_best_pair", "driving_sum_for_R_multi",
]
