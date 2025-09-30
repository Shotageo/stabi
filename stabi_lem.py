# stabi_lem.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

DEG = np.pi / 180.0
EPS = 1e-12

# -------------------------------------------------
# Soil / Slope（地形は直線：将来ポリライン拡張を想定）
# -------------------------------------------------
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

    def points_on_surface(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """s in [0,1] along ground from (0,H) to (L,0)."""
        s = np.asarray(s)
        x = s * self.L
        y = self.y_ground(x)
        return x, y

# -------------------------------------------------
# 幾何ユーティリティ
# -------------------------------------------------
def circle_center_from_chord_radius(
    x1: float, y1: float, x2: float, y2: float, R: float, prefer_lower_arc_below_ground: bool,
    y_ground_mid: float
) -> Optional[Tuple[float, float]]:
    """エントリ/エグジットと半径から中心を決定（弦の垂直二等分線上）。
       prefer_lower_arc_below_ground=True のとき，中点で円弧が地表より下になる方を選ぶ。
    """
    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)
    if not np.isfinite(d) or d <= 0.0:
        return None
    # 幾何制約：半径は d/2 より大きい必要
    if R <= 0.5 * d:
        return None

    # 中点と単位法線（弦を左回り90°回転→( -dy, dx )）
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx, ny = -dy, dx
    nlen = np.hypot(nx, ny)
    if nlen <= 0:
        return None
    nx, ny = nx / nlen, ny / nlen

    # 中心は垂直二等分線上の距離 s
    s = np.sqrt(max(R * R - (0.5 * d) ** 2, 0.0))
    # 候補2つ
    cx1, cy1 = mx + s * nx, my + s * ny
    cx2, cy2 = mx - s * nx, my - s * ny

    # 中点で下側弧が地表より下になる方を選ぶ
    def y_arc_at(cx, cy):
        inside = R * R - (mx - cx) ** 2
        if inside <= 0:
            return None
        return cy - np.sqrt(inside)  # 下側分枝
    ya1 = y_arc_at(cx1, cy1)
    ya2 = y_arc_at(cx2, cy2)
    if ya1 is None and ya2 is None:
        return None

    if prefer_lower_arc_below_ground:
        cand = []
        if ya1 is not None:
            cand.append((abs((y_ground_mid - ya1)), cx1, cy1, ya1 < y_ground_mid))
        if ya2 is not None:
            cand.append((abs((y_ground_mid - ya2)), cx2, cy2, ya2 < y_ground_mid))
        # 弧が地表の下にある候補を優先し、距離が近い方
        cand.sort(key=lambda t: (not t[3], t[0]))
        _, cx, cy, _ = cand[0]
        return (cx, cy)
    else:
        # 単純に距離が近い方
        if ya1 is None:
            return (cx2, cy2)
        if ya2 is None:
            return (cx1, cy1)
        return (cx1, cy1) if abs(y_ground_mid - ya1) <= abs(y_ground_mid - ya2) else (cx2, cy2)

def arc_y_at(x: np.ndarray, xc: float, yc: float, R: float) -> Optional[np.ndarray]:
    inside = R ** 2 - (x - xc) ** 2
    if np.any(inside <= 0):
        return None
    return yc - np.sqrt(inside)  # 下側分枝

def circle_line_intersections_straight_ground(
    slope: Slope, xc: float, yc: float, R: float
) -> Optional[Tuple[float, float]]:
    """円と直線地表の交点（xのみ返す、区間内チェックを含む）"""
    m, H = slope.m, slope.H
    A = 1 + m * m
    B = 2 * (m * (H - yc) - xc)
    C = xc * xc + (H - yc) * (H - yc) - R * R
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    s = np.sqrt(disc)
    x1 = (-B - s) / (2 * A)
    x2 = (-B + s) / (2 * A)
    xl, xh = (x1, x2) if x1 < x2 else (x2, x1)
    # 地表区間 [0,L] 内に両交点があること
    if xl < -1e-8 or xh > slope.L + 1e-8:
        return None
    return float(xl), float(xh)

# -------------------------------------------------
# スライス幾何とFS（Fellenius / Bishop 簡略）
# -------------------------------------------------
def _slice_geometry(xmid: np.ndarray, xc: float, yc: float, R: float):
    inside = R ** 2 - (xmid - xc) ** 2
    if np.any(inside <= 0):
        return None, None, None
    y_arc = yc - np.sqrt(inside)
    denom = (y_arc - yc)
    denom = np.where(np.abs(denom) < EPS, np.sign(denom) * EPS, denom)
    dydx = -(xmid - xc) / denom
    alpha = np.arctan(dydx)
    cos_a = np.cos(alpha)
    if np.any(np.isclose(cos_a, 0.0, atol=1e-10)):
        return None, None, None
    return y_arc, alpha, cos_a

def _depth_filters(h: np.ndarray, h_min: float, h_max: float, pct: float, min_eff_ratio: float) -> bool:
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        return False
    if float(np.max(h)) > h_max:
        return False
    eff = h > 1e-4
    if eff.mean() < min_eff_ratio:
        return False
    if pct > 0:
        if float(np.percentile(h, pct)) < h_min:
            return False
    return True

def fs_fellenius(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 40, h_min: float = 0.1, h_max: float = 1e9, pct: float = 15.0,
    min_eff_ratio: float = 0.6
) -> Optional[float]:
    inter = circle_line_intersections_straight_ground(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-8:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
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
    Fs = float(num / den)
    if not np.isfinite(Fs) or Fs <= 0:
        return None
    return Fs

def fs_bishop(
    slope: Slope, soil: Soil, xc: float, yc: float, R: float,
    n_slices: int = 40, max_iter: int = 100, tol: float = 1e-6,
    h_min: float = 0.1, h_max: float = 1e9, pct: float = 15.0,
    min_eff_ratio: float = 0.6
) -> Optional[float]:
    inter = circle_line_intersections_straight_ground(slope, xc, yc, R)
    if inter is None:
        return None
    x1, x2 = inter
    if x2 - x1 < 1e-8:
        return None

    xs = np.linspace(x1, x2, n_slices + 1)
    xmid = 0.5 * (xs[:-1] + xs[1:])
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
        m_alpha = 1.0 + (tanp * np.tan(alpha)) / max(Fs, EPS)
        num = np.sum((soil.c * b + W * tanp * np.cos(alpha)) / m_alpha)
        den = np.sum(W * np.sin(alpha))
        if den <= 0:
            return None
        Fs_new = num / den
        if not np.isfinite(Fs_new) or Fs_new <= 0:
            return None
        if abs(Fs_new - Fs) < tol:
            return float(Fs_new)
        Fs = Fs_new
    return float(Fs)

# -------------------------------------------------
# 候補生成：Entry–Exit 主軸 ＋ Grid&Radius（補助）＋Refine
# -------------------------------------------------
def gen_candidates_entry_exit(
    slope: Slope,
    entry_s_range: Tuple[float, float], exit_s_range: Tuple[float, float],
    n_entry: int, n_exit: int,
    Rmin: float, Rmax: float, nR: int,
) -> List[Dict]:
    """Entry/Exit を等分し、各弦×半径で円を作る（幾何のみ）。"""
    e0, e1 = entry_s_range
    xE, yE = slope.points_on_surface(np.linspace(e0, e1, n_entry))
    xX, yX = slope.points_on_surface(np.linspace(exit_s_range[0], exit_s_range[1], n_exit))

    Rs = np.linspace(Rmin, Rmax, nR)
    recs: List[Dict] = []
    for xe, ye in zip(xE, yE):
        for xx, yx in zip(xX, yX):
            if xx <= xe:  # 下り斜面の進行方向で Exit は Entry より右（一般化は後続）
                continue
            d = np.hypot(xx - xe, yx - ye)
            if d <= 1e-6:
                continue
            # その弦に対して許される最小半径
            Rmin_local = max(Rs[0], 0.51 * d)
            for R in Rs:
                if R < Rmin_local:
                    continue
                ymid_ground = slope.y_ground((xe + xx) / 2.0)
                ctr = circle_center_from_chord_radius(
                    xe, ye, xx, yx, R, prefer_lower_arc_below_ground=True, y_ground_mid=float(ymid_ground)
                )
                if ctr is None:
                    continue
                xc, yc = ctr
                recs.append({"xc": float(xc), "yc": float(yc), "R": float(R),
                             "x1": float(xe), "y1": float(ye), "x2": float(xx), "y2": float(yx)})
    return recs

def gen_candidates_center_grid(
    slope: Slope,
    x_center_range: Tuple[float, float], y_center_range: Tuple[float, float],
    R_range: Tuple[float, float],
    nx: int, ny: int, nR: int
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
                y1 = slope.y_ground(x1)
                y2 = slope.y_ground(x2)
                recs.append({"xc": float(xc), "yc": float(yc), "R": float(R),
                             "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
    return recs

def evaluate_candidates(
    slope: Slope, soil: Soil, recs: List[Dict], method: str,
    n_slices: int, h_min: float, h_max: float, pct: float, min_eff_ratio: float
) -> List[Dict]:
    fs_fun = fs_bishop if method.lower().startswith("b") else fs_fellenius
    out: List[Dict] = []
    for r in recs:
        fs = fs_fun(slope, soil, r["xc"], r["yc"], r["R"],
                    n_slices=n_slices, h_min=h_min, h_max=h_max, pct=pct, min_eff_ratio=min_eff_ratio)
        if fs is None:
            continue
        rr = dict(r)
        rr["Fs"] = float(fs)
        out.append(rr)
    return out

def search_entry_exit_adaptive(
    slope: Slope, soil: Soil,
    entry_s_range: Tuple[float, float], exit_s_range: Tuple[float, float],
    n_entry: int, n_exit: int,
    Rmin: float, Rmax: float, nR: int,
    method: str, n_slices: int,
    h_min: float, h_max: float, pct: float, min_eff_ratio: float,
    refine_levels: int = 2, top_k: int = 10, shrink: float = 0.45
) -> Dict:
    # 粗探索
    recs_geo = gen_candidates_entry_exit(slope, entry_s_range, exit_s_range, n_entry, n_exit, Rmin, Rmax, nR)
    evals = evaluate_candidates(slope, soil, recs_geo, method, n_slices, h_min, h_max, pct, min_eff_ratio)

    if not evals or refine_levels <= 0:
        return _pack_results(evals)

    seeds = sorted(evals, key=lambda r: r["Fs"])[:max(1, top_k)]
    all_evals = list(evals)

    e_span = (entry_s_range[1] - entry_s_range[0])
    x_span = (exit_s_range[1] - exit_s_range[0])
    for lvl in range(refine_levels):
        factor = (shrink ** (lvl + 1))
        de = 0.5 * e_span * factor
        dx = 0.5 * x_span * factor
        for s in seeds:
            # 種の entry/exit を逆算（近似）：x1,x2 から s を推定
            se = s["x1"] / slope.L
            sx = s["x2"] / slope.L
            e_rng = (max(0.0, se - de), min(1.0, se + de))
            x_rng = (max(0.0, sx - dx), min(1.0, sx + dx))
            # R も局所絞り込み
            r_rng = (max(0.5 * s["R"], Rmin), min(1.5 * s["R"], Rmax))
            recs_geo2 = gen_candidates_entry_exit(slope, e_rng, x_rng, max(5, n_entry//2), max(5, n_exit//2),
                                                  r_rng[0], r_rng[1], max(5, nR//2))
            evals2 = evaluate_candidates(slope, soil, recs_geo2, method, n_slices, h_min, h_max, pct, min_eff_ratio)
            all_evals.extend(evals2)
        if all_evals:
            seeds = sorted(all_evals, key=lambda r: r["Fs"])[:max(1, top_k)]

    return _pack_results(all_evals)

def _pack_results(evals: List[Dict]) -> Dict:
    if not evals:
        return {"candidates": [], "best": None}
    best = min(evals, key=lambda r: r["Fs"])
    # UI速度のため上位のみ
    evals_sorted = sorted(evals, key=lambda r: r["Fs"])[:1200]
    return {"candidates": evals_sorted, "best": best}
