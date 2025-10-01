# stabi_lem.py  — 2-layer support (clipped interface), lightweight calcs
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import numpy as np, math

DEG = math.pi / 180.0
EPS = 1e-12

# ---------- 基本データ ----------
@dataclass
class Soil:
    gamma: float   # kN/m3
    c: float       # kPa
    phi: float     # deg

@dataclass
class GroundPL:
    """x単調増加の折線"""
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

# ---------- 交点 ----------
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
    # 重複削除
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

# ---------- 円弧サンプリング（最適隣接ペア） ----------
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
        ys = yc - np.sqrt(inside)         # 下面枝
        h  = pl.y_at(xs) - ys
        if np.any(h <= 0) or np.any(~np.isfinite(h)):
            continue
        dmax = float(np.max(h))
        if (best is None) or (dmax > best[-1]):
            best = (x1, x2, xs, ys, h, dmax)
    if best is None: return None
    x1, x2, xs, ys, h, _ = best
    return x1, x2, xs, ys, h

# ---------- 幾何補助 ----------
def _alpha_cos(xc, yc, R, xs):
    inside = R*R - (xs - xc)**2
    y_arc = yc - np.sqrt(inside)
    denom = y_arc - yc
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom)*1e-12, denom)
    dydx  = -(xs - xc) / denom
    alpha = -np.arctan(dydx)     # 右下向きが正
    return alpha, np.cos(alpha), y_arc

# ---------- 単層Fs ----------
def fs_fellenius_poly(pl: GroundPL, soil: Soil, xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(pl, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = pl.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    b     = dx / cos_a
    W     = soil.gamma * hmid * dx
    tanp  = math.tan(soil.phi*DEG)
    num   = float(np.sum(soil.c*b + W*np.cos(alpha)*tanp))
    den   = float(np.sum(W*np.sin(alpha)))
    if den <= 0: return None
    Fs    = num/den
    return Fs if (np.isfinite(Fs) and Fs>0) else None

def fs_bishop_poly(pl: GroundPL, soil: Soil, xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(pl, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = pl.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    b     = dx / cos_a
    W     = soil.gamma * hmid * dx
    tanp  = math.tan(soil.phi*DEG)
    Fs    = 1.3
    for _ in range(120):
        denom_a = 1.0 + (tanp*np.tan(alpha))/max(Fs,1e-12)
        num = float(np.sum((soil.c*b + W*tanp*np.cos(alpha))/denom_a))
        den = float(np.sum(W*np.sin(alpha)))
        if den <= 0: return None
        Fs_new = num/den
        if not (np.isfinite(Fs_new) and Fs_new>0): return None
        if abs(Fs_new-Fs) < 1e-6: return float(Fs_new)
        Fs = Fs_new
    return float(Fs)

# ---------- 2層Fs（層境界は“地表でクリップ”して判定） ----------
def _base_soil_vectors(pl_ground: GroundPL, interface: GroundPL,
                       soil_upper: Soil, soil_lower: Soil, xmid, y_arc):
    # 判定用の層境界を地表でクリップ
    yint_raw = interface.y_at(xmid)
    ygrd = pl_ground.y_at(xmid)
    yint = np.minimum(yint_raw, ygrd)
    mask_upper = (y_arc >= yint)
    gamma = np.where(mask_upper, soil_upper.gamma, soil_lower.gamma).astype(float)
    c     = np.where(mask_upper, soil_upper.c,     soil_lower.c    ).astype(float)
    phi   = np.where(mask_upper, soil_upper.phi,   soil_lower.phi  ).astype(float)
    return gamma, c, phi

def fs_fellenius_poly_layered(pl: GroundPL, interface: GroundPL,
                              soil_upper: Soil, soil_lower: Soil,
                              xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(pl, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = pl.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    b     = dx / cos_a
    gamma, c, phi = _base_soil_vectors(pl, interface, soil_upper, soil_lower, xmid, y_arc)
    W     = gamma * hmid * dx
    tanp  = np.tan(phi*DEG)
    num   = float(np.sum(c*b + W*np.cos(alpha)*tanp))
    den   = float(np.sum(W*np.sin(alpha)))
    if den <= 0: return None
    Fs    = num/den
    return Fs if (np.isfinite(Fs) and Fs>0) else None

def fs_bishop_poly_layered(pl: GroundPL, interface: GroundPL,
                           soil_upper: Soil, soil_lower: Soil,
                           xc, yc, R, n_slices=40) -> Optional[float]:
    s = arc_sample_poly_best_pair(pl, xc, yc, R, n=max(2*n_slices+1,201))
    if s is None: return None
    x1, x2, xs, ys, h = s
    xs_e  = np.linspace(x1, x2, n_slices+1)
    xmid  = 0.5*(xs_e[:-1] + xs_e[1:])
    dx    = (x2 - x1)/n_slices
    alpha, cos_a, y_arc = _alpha_cos(xc, yc, R, xmid)
    if np.any(np.isclose(cos_a,0,atol=1e-10)): return None
    hmid  = pl.y_at(xmid) - y_arc
    if np.any(hmid<=0): return None
    b     = dx / cos_a
    gamma, c, phi = _base_soil_vectors(pl, interface, soil_upper, soil_lower, xmid, y_arc)
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

# ---------- 首振り：エントリ点掃引（単層/2層 両対応） ----------
def arcs_from_center_by_entries(
    pl: GroundPL,
    soil_or_upper: Soil,
    xc: float, yc: float,
    n_entries: int = 500,
    method: str = "bishop",
    depth_min: float = 0.0, depth_max: float = 1e9,
    interface: GroundPL | None = None,
    soil_lower: Soil | None = None,
) -> Iterable[Tuple[float,float,float,float]]:
    """
    戻り値: 反復で (x1, x2, R, Fs) を順次返す（配列は保持しない）
    - 単層: interface=None
    - 2層:  interface!=None かつ soil_lower!=None（内部判定は地表でクリップ）
    """
    xs = np.linspace(pl.X[0], pl.X[-1], max(n_entries, 50))
    ys = pl.y_at(xs)
    use_bishop = method.lower().startswith("b")
    layered = (interface is not None and soil_lower is not None)

    for xe, ye in zip(xs, ys):
        R = float(math.hypot(xe - xc, ye - yc))
        s = arc_sample_poly_best_pair(pl, xc, yc, R, n=221)
        if s is None:
            continue
        x1, x2, xs_s, ys_s, h = s
        dmax = float(np.max(h))
        if dmax < depth_min - 1e-9 or dmax > depth_max + 1e-9:
            continue

        if not layered:
            if use_bishop:
                Fs = fs_bishop_poly(pl, soil_or_upper, xc, yc, R, n_slices=40)
                if Fs is None:
                    Fs = fs_fellenius_poly(pl, soil_or_upper, xc, yc, R, n_slices=40)
            else:
                Fs = fs_fellenius_poly(pl, soil_or_upper, xc, yc, R, n_slices=40)
        else:
            upper = soil_or_upper
            lower = soil_lower
            if use_bishop:
                Fs = fs_bishop_poly_layered(pl, interface, upper, lower, xc, yc, R, n_slices=40)
                if Fs is None:
                    Fs = fs_fellenius_poly_layered(pl, interface, upper, lower, xc, yc, R, n_slices=40)
            else:
                Fs = fs_fellenius_poly_layered(pl, interface, upper, lower, xc, yc, R, n_slices=40)

        if Fs is None:
            continue
        yield (float(x1), float(x2), float(R), float(Fs))

# ---------- 例: 地表／層境界プリセット ----------
def make_ground_example(H: float, L: float) -> GroundPL:
    # 3セグメント（緩い段付き）—上面
    X = np.array([0.0, 0.30*L, 0.63*L, L], dtype=float)
    Y = np.array([H,   0.88*H, 0.46*H, 0.0], dtype=float)
    return GroundPL(X=X, Y=Y)

def make_interface_example(H: float, L: float) -> GroundPL:
    # 層境界（右下がり）。見た目上は地表を超えることがあるため、
    # 描画時・判定時は地表でクリップする（本関数は“素の形”を返す）
    X = np.array([0.0, 0.35*L, 0.70*L, L], dtype=float)
    Y = np.array([0.70*H, 0.60*H, 0.38*H, 0.20*H], dtype=float)
    return GroundPL(X=X, Y=Y)

__all__ = [
    "Soil", "GroundPL",
    "make_ground_example", "make_interface_example",
    "arcs_from_center_by_entries",
    "fs_bishop_poly", "fs_fellenius_poly",
    "fs_bishop_poly_layered", "fs_fellenius_poly_layered",
]
