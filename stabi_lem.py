# stabi_lem.py
# ------------------------------------------------------------
# 安定板３の骨格を維持しつつ、Quick段階での円弧診断（diagnostics）を追加。
# 可視化は診断を読む側（streamlit_app.py）が担当。既存呼び出しは後方互換。
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

# ====== データモデル ======

@dataclass
class Ground:
    xs: np.ndarray
    ys: np.ndarray

    @staticmethod
    def from_points(xs: List[float], ys: List[float]) -> "Ground":
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        assert x.ndim == 1 and y.ndim == 1 and x.size == y.size
        return Ground(x, y)

    def y_at(self, qx: np.ndarray | List[float]) -> np.ndarray:
        qx = np.asarray(qx, dtype=float)
        return np.interp(qx, self.xs, self.ys, left=self.ys[0], right=self.ys[-1])

@dataclass
class Layer:
    gamma: float   # 単位体積重量
    phi_deg: float # 内部摩擦角
    c: float       # 粘着力
    extras: Optional[Dict[str, Any]] = None

@dataclass
class Nail:
    x: float
    y: float
    length: float
    angle_deg: float
    bond: float  # 1本あたりの簡略抵抗

@dataclass
class Config:
    grid_xmin: float
    grid_xmax: float
    grid_ymin: float
    grid_ymax: float
    grid_step: float
    r_min: float
    r_max: float
    coarse_step: int = 6
    quick_step: int = 3
    refine_step: int = 1
    budget_coarse_s: float = 0.8
    budget_quick_s: float = 1.2

# ====== 評価関数（可視化向けの安定な簡略式） ======

def _fs_for_arc_simple(ground: Ground, layers: List[Layer], cx: float, cy: float, r: float) -> float:
    th = np.linspace(0.0, 2*np.pi, 180)
    xs = cx + r * np.cos(th)
    ys = cy + r * np.sin(th)
    yg = ground.y_at(xs)

    # 「地盤の下側」を原則として採用（点が少なければ上側を代替）
    mask = ys <= yg
    if np.count_nonzero(mask) < 3:
        mask_alt = ys >= yg
        if np.count_nonzero(mask_alt) < 3:
            return float("inf")
        mask = mask_alt

    xs = xs[mask]; ys = ys[mask]; yg = yg[mask]

    # 代表材料（最上層）
    if layers:
        phi = float(layers[0].phi_deg)
        c   = float(layers[0].c)
        gamma = float(layers[0].gamma)
    else:
        phi, c, gamma = 30.0, 0.0, 18.0

    # 接線近似（円：接線は(-sin, +cos)）
    tx = -np.sin(th[mask]); ty = np.cos(th[mask])
    alpha = np.arctan2(ty, tx)

    dx = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0]))
    thickness = np.maximum(yg - ys, 0.0)
    W = thickness * gamma * dx

    sin_a = np.abs(np.sin(alpha)); cos_a = np.abs(np.cos(alpha))
    driving  = np.sum(W * sin_a + 1e-12)
    resisting = np.sum(c * dx + W * cos_a * math.tan(math.radians(phi)))
    if driving <= 0:
        return float("inf")
    return max(1e-6, resisting / driving)

def _nail_contribution_simple(nails: List[Nail], cx: float, cy: float, r: float) -> float:
    if not nails:
        return 0.0
    contrib = 0.0
    for n in nails:
        ang = math.radians(n.angle_deg)
        x2 = n.x + n.length * math.cos(ang)
        y2 = n.y + n.length * math.sin(ang)
        d1 = math.hypot(n.x - cx, n.y - cy) - r
        d2 = math.hypot(x2 - cx, y2 - cy) - r
        if d1 == 0: d1 = 1e-9
        if d2 == 0: d2 = -1e-9
        if d1 * d2 < 0:
            contrib += max(0.0, n.bond)
    return contrib

# ====== 探索グリッド ======

def _gen_centers(cfg: Config, step_mul: int) -> List[Tuple[float,float]]:
    xs = np.arange(cfg.grid_xmin, cfg.grid_xmax + 1e-9, cfg.grid_step * step_mul)
    ys = np.arange(cfg.grid_ymin, cfg.grid_ymax + 1e-9, cfg.grid_step * step_mul)
    return [(float(x), float(y)) for x in xs for y in ys]

def _radius_candidates(cfg: Config, step_mul: int, num: int) -> np.ndarray:
    return np.geomspace(max(cfg.r_min, 1e-2), max(cfg.r_max, cfg.r_min + 1e-2), num=max(3, num // max(1, step_mul)))

def _evaluate_stage(ground: Ground, layers: List[Layer], nails: List[Nail],
                    centers: List[Tuple[float,float]], radii: np.ndarray,
                    collect_quick: bool, fs_cutoff_collect: float) -> Tuple[float, Tuple[float,float,float], List[Dict[str,Any]]]:
    best_fs = float("inf")
    best_arc = (0.0, 0.0, 1.0)
    records: List[Dict[str,Any]] = []

    for cx, cy in centers:
        for r in radii:
            fs0 = _fs_for_arc_simple(ground, layers, cx, cy, r)
            fs = max(1e-6, fs0 + _nail_contribution_simple(nails, cx, cy, r))
            if fs < best_fs:
                best_fs = fs
                best_arc = (cx, cy, r)
            if collect_quick and (r > 0) and np.isfinite(fs) and (fs < fs_cutoff_collect):
                records.append({"cx": float(cx), "cy": float(cy), "r": float(r), "fs": float(fs), "stage": "quick"})
    return best_fs, best_arc, records

# ====== 公開API ======

def run_analysis(
    ground: Ground,
    layers: List[Layer],
    nails: List[Nail],
    cfg: Config,
    fs_cutoff_collect: float = 1.3
) -> Dict[str, Any]:
    # Coarse
    centers_c = _gen_centers(cfg, cfg.coarse_step)
    radii_c   = _radius_candidates(cfg, cfg.coarse_step, num=12)
    fs_c, arc_c, _ = _evaluate_stage(ground, layers, nails, centers_c, radii_c, collect_quick=False, fs_cutoff_collect=fs_cutoff_collect)

    # Quick（収集）
    centers_q = _gen_centers(cfg, cfg.quick_step)
    radii_q   = _radius_candidates(cfg, cfg.quick_step, num=16)
    fs_q, arc_q, quick_records = _evaluate_stage(ground, layers, nails, centers_q, radii_q, collect_quick=True, fs_cutoff_collect=fs_cutoff_collect)

    # Refine（近傍）
    cx_b, cy_b, r_b = arc_q
    step = max(1, cfg.refine_step)
    centers_r = [(cx_b + dx*cfg.grid_step/step, cy_b + dy*cfg.grid_step/step) for dx in (-1,0,1) for dy in (-1,0,1)]
    radii_r   = np.geomspace(max(cfg.r_min, r_b*0.7), min(cfg.r_max, r_b*1.3), num=10)
    fs_r, arc_r, _ = _evaluate_stage(ground, layers, nails, centers_r, radii_r, collect_quick=False, fs_cutoff_collect=fs_cutoff_collect)

    # Fs（簡略）：before=Quick最良、after=Refine最良（ネイル寄与込み）
    Fs_before = float(fs_q)
    Fs_after  = float(fs_r)

    diagnostics = {
        "quick_arcs": quick_records,
        "grid_bbox": (cfg.grid_xmin, cfg.grid_xmax, cfg.grid_ymin, cfg.grid_ymax),
        "grid_step": float(cfg.grid_step),
        "grid_centers_sampled": [(float(x), float(y)) for (x, y) in centers_q],
        "quick_best": {"cx": cx_b, "cy": cy_b, "r": r_b, "fs": fs_q}
    }

    return {
        "Fs_before": Fs_before,
        "Fs_after":  Fs_after,
        "diagnostics": diagnostics
    }
