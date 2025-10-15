# stabi_lem.py
# ------------------------------------------------------------
# 安定板３ ベースライン（復旧版）
# - LEM + Soil Nail の簡易連成： Fs_after = ΣT / Σ(W sinα)
# - 探索フロー：Coarse -> Quick -> Refine（時間バジェット思想のみ保持）
# - ground.y_at はベクトル安全実装
# - NaN/inf ガード、品質フォールバック、Config自動初期化
# - 戻り値は既存キーのみ（diagnostics等の追加なし）
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

# ===== モデル定義 =====

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
    angle_deg: float  # 地盤内向きに負角を推奨（例：斜面接線−90°+Δβ）
    bond: float       # 付着・引抜き等を簡略化した抵抗（/本）

@dataclass
class Config:
    # センター探索グリッド
    grid_xmin: float
    grid_xmax: float
    grid_ymin: float
    grid_ymax: float
    grid_step: float
    # 円弧半径
    r_min: float
    r_max: float
    # 段階ステップ（Coarse→Quick→Refine）
    coarse_step: int = 6
    quick_step: int = 3
    refine_step: int = 1
    # 時間バジェット思想（参照用）
    budget_coarse_s: float = 0.8
    budget_quick_s: float = 1.2

# ===== ユーティリティ =====

def _safe_isfinite(x: float) -> bool:
    try:
        return np.isfinite(x)
    except Exception:
        return False

def _gen_centers(cfg: Config, step_mul: int) -> List[Tuple[float, float]]:
    xs = np.arange(cfg.grid_xmin, cfg.grid_xmax + 1e-9, max(1e-6, cfg.grid_step * max(1, step_mul)))
    ys = np.arange(cfg.grid_ymin, cfg.grid_ymax + 1e-9, max(1e-6, cfg.grid_step * max(1, step_mul)))
    return [(float(x), float(y)) for x in xs for y in ys]

def _radius_candidates(cfg: Config, step_mul: int, num: int) -> np.ndarray:
    n = max(3, num // max(1, step_mul))
    rmin = max(cfg.r_min, 1e-2)
    rmax = max(cfg.r_max, rmin + 1e-2)
    return np.geomspace(rmin, rmax, num=n)

# ===== LEM（簡易相対式） =====
# ※ 可視化ではなく相対比較のための安定な簡易式。安定板３の思想を踏襲。

def _fs_segmental_simple(ground: Ground, layers: List[Layer], cx: float, cy: float, r: float) -> float:
    """地表側弧のみを対象に、帯状片の力学で簡略FSを算出（ネイル抜き）"""
    th = np.linspace(0.0, 2*np.pi, 180)
    xs = cx + r * np.cos(th)
    ys = cy + r * np.sin(th)
    yg = ground.y_at(xs)

    # 原則：地表線の下側（ys <= yg）を土塊側とみなす。点が少なければ上側代替。
    mask = ys <= yg
    if np.count_nonzero(mask) < 3:
        alt = ys >= yg
        if np.count_nonzero(alt) < 3:
            return float("inf")
        mask = alt

    xs = xs[mask]; ys = ys[mask]; yg = yg[mask]

    # 代表物性（最上層）
    if layers:
        phi = float(layers[0].phi_deg)
        c   = float(layers[0].c)
        gamma = float(layers[0].gamma)
    else:
        phi, c, gamma = 30.0, 0.0, 18.0

    # 接線ベクトル（円：(-sin, +cos)）
    thm = th[mask]
    tx = -np.sin(thm)
    ty =  np.cos(thm)
    alpha = np.arctan2(ty, tx)

    dx = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0]))
    thickness = np.maximum(yg - ys, 0.0)
    W = thickness * gamma * dx

    sin_a = np.abs(np.sin(alpha))
    cos_a = np.abs(np.cos(alpha))
    driving   = np.sum(W * sin_a + 1e-12)
    resisting = np.sum(c * dx + W * cos_a * math.tan(math.radians(phi)))
    if driving <= 0:
        return float("inf")
    return max(1e-6, resisting / driving)

def _nail_T_simple(nails: List[Nail], cx: float, cy: float, r: float) -> float:
    """円弧とネイルの交差本数に比例した追加抵抗（単純化）"""
    if not nails:
        return 0.0
    T = 0.0
    for n in nails:
        ang = math.radians(n.angle_deg)
        x2 = n.x + n.length * math.cos(ang)
        y2 = n.y + n.length * math.sin(ang)
        d1 = math.hypot(n.x - cx, n.y - cy) - r
        d2 = math.hypot(x2 - cx, y2 - cy) - r
        if d1 == 0: d1 = 1e-9
        if d2 == 0: d2 = -1e-9
        if d1 * d2 < 0:
            T += max(0.0, float(n.bond))
    return T

def _evaluate_stage(ground: Ground, layers: List[Layer], nails: List[Nail],
                    centers: List[Tuple[float, float]], radii: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    """ある段階（ステップ倍率固定）での最小FS探索"""
    best_fs = float("inf")
    best_arc = (0.0, 0.0, max(1.0, float(np.median(radii))))
    for cx, cy in centers:
        for r in radii:
            if r <= 0 or not _safe_isfinite(r):  # ガード
                continue
            fs0 = _fs_segmental_simple(ground, layers, cx, cy, r)
            if not _safe_isfinite(fs0):
                continue
            # ネイル連成：Fs_after = (抵抗+T)/駆動 ~ fs0 + T/Σ(W sinα)
            # ここでは fs0 に T の相対寄与を線形加算（安定板３の簡易結合）
            fs = max(1e-6, fs0 + _nail_T_simple(nails, cx, cy, r))
            if fs < best_fs:
                best_fs = fs
                best_arc = (cx, cy, r)
    return float(best_fs), best_arc

# ===== 公開API（既存I/F） =====

def run_analysis(
    ground: Ground,
    layers: List[Layer],
    nails: List[Nail],
    cfg: Optional[Config] = None
) -> Dict[str, Any]:
    """安定板３：公開関数（戻り値キーは既存のまま）"""
    # Config自動初期化（ガード）
    if cfg is None:
        xs = ground.xs
        ys = ground.ys
        cfg = Config(
            grid_xmin=float(xs.min()+5), grid_xmax=float(xs.max()-5),
            grid_ymin=float(ys.min()-30), grid_ymax=float(ys.max()+10),
            grid_step=8.0,
            r_min=5.0, r_max=max(10.0, (xs.max()-xs.min())*1.2),
            coarse_step=6, quick_step=3, refine_step=1,
            budget_coarse_s=0.8, budget_quick_s=1.2
        )

    # === Coarse ===
    centers_c = _gen_centers(cfg, cfg.coarse_step)
    radii_c   = _radius_candidates(cfg, cfg.coarse_step, num=12)
    fs_c, arc_c = _evaluate_stage(ground, layers, nails, centers_c, radii_c)

    # === Quick ===
    centers_q = _gen_centers(cfg, cfg.quick_step)
    radii_q   = _radius_candidates(cfg, cfg.quick_step, num=16)
    fs_q, arc_q = _evaluate_stage(ground, layers, nails, centers_q, radii_q)

    # === Refine ===（Quick最良近傍）
    cx, cy, r = arc_q
    step = max(1, cfg.refine_step)
    centers_r = [(cx + dx*cfg.grid_step/step, cy + dy*cfg.grid_step/step) for dx in (-1,0,1) for dy in (-1,0,1)]
    radii_r   = np.geomspace(max(cfg.r_min, r*0.7), min(cfg.r_max, r*1.3), num=10)
    fs_r, arc_r = _evaluate_stage(ground, layers, nails, centers_r, radii_r)

    # before/after（簡易定義）：before = Quick最良（ネイル寄与込み）、after = Refine最良
    Fs_before = float(fs_q) if _safe_isfinite(fs_q) else float("inf")
    Fs_after  = float(fs_r) if _safe_isfinite(fs_r) else float("inf")

    # NaN/inf フォールバック
    if not np.isfinite(Fs_before): Fs_before = 9999.0
    if not np.isfinite(Fs_after):  Fs_after  = 9999.0

    return {
        "Fs_before": Fs_before,
        "Fs_after":  Fs_after,
        # 必要があれば arc 等を追加しても良いが、既存互換のため最低限に留める
    }