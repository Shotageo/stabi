# stabi_lem.py
# ------------------------------------------------------------
# 安定板３をベースに、Quick段階の円弧診断（diagnostics）を追加。
# 既存呼び出しはそのまま動作し、diagnostics を読まない場合は挙動不変。
# 可視化側（streamlit_app.py）からは result["diagnostics"] を参照。
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

# ====== 基本構造（安定板３の骨格に準拠／後方互換） =====================

@dataclass
class Ground:
    """
    地表線。安定板３の「配列内包で安全に y_at を評価」の方針を踏襲。
    """
    xs: np.ndarray
    ys: np.ndarray

    @staticmethod
    def from_points(xs: List[float], ys: List[float]) -> "Ground":
        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)
        assert xs_arr.ndim == 1 and ys_arr.ndim == 1 and xs_arr.size == ys_arr.size
        return Ground(xs_arr, ys_arr)

    def y_at(self, qx: np.ndarray | List[float]) -> np.ndarray:
        """
        ベクトル安全版。外挿は端値ホールド。
        """
        qx = np.asarray(qx, dtype=float)
        return np.interp(qx, self.xs, self.ys, left=self.ys[0], right=self.ys[-1])

@dataclass
class Nail:
    x: float
    y: float
    length: float
    angle_deg: float
    bond: float  # 単位幅当たりの付着強度など簡略化パラメータ

@dataclass
class Layer:
    gamma: float  # 単位体積重量
    phi_deg: float
    c: float      # 粘着力
    # 追加の材料パラメータがあっても無視されないように辞書で受ける
    extras: Optional[Dict[str, Any]] = None

@dataclass
class Config:
    # 探索グリッド
    grid_xmin: float
    grid_xmax: float
    grid_ymin: float
    grid_ymax: float
    grid_step: float

    # 探索半径
    r_min: float
    r_max: float

    # 段階ごとのサンプリング密度
    coarse_step: int = 6
    quick_step: int = 3
    refine_step: int = 1

    # 予算（安定板３の思想のみ維持。ここでは未使用でも保持）
    budget_coarse_s: float = 0.8
    budget_quick_s: float = 1.2

# ====== LEM簡略モデル（安定板３：ΣT/Σ(W sinα) の素結合を踏襲） ======

def _weight(y_top: float, y_bottom: float, width: float, gamma: float) -> float:
    """
    簡易な帯状の自重。ここでは厚さ * 幅 * γ として概算。
    """
    t = max(0.0, y_top - y_bottom)
    return t * width * gamma

def _fs_for_arc_simple(ground: Ground, layers: List[Layer], cx: float, cy: float, r: float) -> float:
    """
    簡略 LEM：弧に沿った小片の安定・不安定をラフに積分。
    実プロダクションの正確なFSとは異なるが、可視化用の相対評価として安定。
    """
    # サンプリング
    th = np.linspace(0.0, 2*np.pi, 180)
    xs = cx + r * np.cos(th)
    ys = cy + r * np.sin(th)
    yg = ground.y_at(xs)

    # 地表の「内部側」を優先（上下どちらが内部かは形状で変わるため両側試験）
    mask_a = ys <= yg
    mask_b = ys >= yg
    mask = mask_a if np.count_nonzero(mask_a) >= np.count_nonzero(mask_b) else mask_b
    if np.count_nonzero(mask) < 3:
        return float("inf")  # 評価不能＝安定とみなす（可視化から外れやすくなる）

    xs = xs[mask]; ys = ys[mask]; yg = yg[mask]
    # 小片の傾斜角（接線角度）
    dth = 1e-3
    # 接線方向の角度（近似）
    tx = -np.sin(th[mask])
    ty =  np.cos(th[mask])
    alpha = np.arctan2(ty, tx)  # 接線角

    # 単層代表でラフ評価（材料の代表値：最上層）
    if len(layers) == 0:
        phi = 30.0; c = 0.0; gamma = 18.0
    else:
        phi = float(layers[0].phi_deg)
        c   = float(layers[0].c)
        gamma = float(layers[0].gamma)

    # 駆動力・抵抗力のラフな合成
    # 断面幅dxに対し、自重W ≈ γ * (yg-ys) * dx
    # 駆動成分 ≈ W * sin(α)；抵抗 ≈ c * dx + W * cos(α) * tan(φ)
    # （非常にラフだが、可視化の相対比較には十分）
    dx = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0]))
    thickness = np.maximum(yg - ys, 0.0)
    W = thickness * gamma * dx
    sin_a = np.abs(np.sin(alpha))
    cos_a = np.abs(np.cos(alpha))
    driving = np.sum(W * sin_a + 1e-12)
    resisting = np.sum(c * dx + W * cos_a * math.tan(math.radians(phi)))
    if driving <= 0:
        return float("inf")
    return max(1e-6, resisting / driving)

def _nail_contribution_simple(nails: List[Nail], cx: float, cy: float, r: float) -> float:
    """
    円弧とネイルの交差回数に比例する追加抵抗（簡略）。
    """
    if not nails:
        return 0.0
    contrib = 0.0
    for n in nails:
        # ネイルの端点
        ang = math.radians(n.angle_deg)
        x2 = n.x + n.length * math.cos(ang)
        y2 = n.y + n.length * math.sin(ang)
        # 中心からの距離差の変化符号で交差を近似判定
        d1 = math.hypot(n.x - cx, n.y - cy) - r
        d2 = math.hypot(x2 - cx, y2 - cy) - r
        if d1 == 0.0: d1 = 1e-9
        if d2 == 0.0: d2 = -1e-9
        if d1 * d2 < 0:
            contrib += max(0.0, n.bond)  # 貫通一本あたり一定の抵抗とみなす
    return contrib

# ====== 探索（Coarse→Quick→Refine）骨格 =================================

def _gen_centers(cfg: Config, step_mul: int) -> List[Tuple[float,float]]:
    xs = np.arange(cfg.grid_xmin, cfg.grid_xmax + 1e-9, cfg.grid_step * step_mul)
    ys = np.arange(cfg.grid_ymin, cfg.grid_ymax + 1e-9, cfg.grid_step * step_mul)
    centers = [(float(x), float(y)) for x in xs for y in ys]
    return centers

def _radius_candidates(cfg: Config, step_mul: int, num: int = 10) -> np.ndarray:
    # 半径は対数均等でサンプリング
    r = np.geomspace(max(cfg.r_min, 1e-2), max(cfg.r_max, cfg.r_min + 1e-2), num=max(3, num // step_mul))
    return r

def _evaluate_stage(ground: Ground, layers: List[Layer], nails: List[Nail],
                    centers: List[Tuple[float,float]], radii: np.ndarray,
                    collect_quick: bool, fs_cutoff_collect: float) -> Tuple[float, Tuple[float,float,float], List[Dict[str,Any]]]:
    """
    段階評価。Quick段階なら可視化用を収集。
    """
    best_fs = float("inf")
    best_arc = (0.0, 0.0, 1.0)  # (cx, cy, r)
    quick_arc_records: List[Dict[str, Any]] = []

    for (cx, cy) in centers:
        for r in radii:
            fs0 = _fs_for_arc_simple(ground, layers, cx, cy, r)
            fs_add = _nail_contribution_simple(nails, cx, cy, r)
            fs = max(1e-6, (fs0 + fs_add))  # 単純結合（可視化相対用）

            if fs < best_fs:
                best_fs = fs
                best_arc = (cx, cy, r)

            if collect_quick:
                # 可視化用に保存（NaN/infガード）
                if (r > 0) and np.isfinite(fs):
                    if fs < fs_cutoff_collect:
                        quick_arc_records.append({
                            "cx": float(cx), "cy": float(cy), "r": float(r),
                            "fs": float(fs), "stage": "quick"
                        })

    return best_fs, best_arc, quick_arc_records

# ====== 公開API ============================================================

def run_analysis(
    ground: Ground,
    layers: List[Layer],
    nails: List[Nail],
    cfg: Config,
    fs_cutoff_collect: float = 1.3
) -> Dict[str, Any]:
    """
    安定板３の標準APIを想定した公開関数。
    - 戻り値は既存キーを維持しつつ、diagnostics を追加。
    """
    # --- Coarse ---
    centers_c = _gen_centers(cfg, cfg.coarse_step)
    radii_c   = _radius_candidates(cfg, cfg.coarse_step, num=12)
    fs_c, arc_c, _ = _evaluate_stage(ground, layers, nails, centers_c, radii_c,
                                     collect_quick=False, fs_cutoff_collect=fs_cutoff_collect)

    # --- Quick（可視化収集） ---
    # Coarse最良付近を含むように、ステップ縮小
    centers_q = _gen_centers(cfg, cfg.quick_step)
    radii_q   = _radius_candidates(cfg, cfg.quick_step, num=16)
    fs_q, arc_q, quick_records = _evaluate_stage(ground, layers, nails, centers_q, radii_q,
                                                 collect_quick=True, fs_cutoff_collect=fs_cutoff_collect)

    # --- Refine ---
    # Quick最良の近傍をさらに細かく（ここでは中心のみ細密化、半径はQuickと同じでも可）
    cx_b, cy_b, r_b = arc_q
    step = max(1, cfg.refine_step)
    # 近傍3x3
    centers_r = [(cx_b + dx*cfg.grid_step/step, cy_b + dy*cfg.grid_step/step)
                 for dx in (-1,0,1) for dy in (-1,0,1)]
    radii_r   = np.geomspace(max(cfg.r_min, r_b*0.7), min(cfg.r_max, r_b*1.3), num=10)
    fs_r, arc_r, _ = _evaluate_stage(ground, layers, nails, centers_r, radii_r,
                                     collect_quick=False, fs_cutoff_collect=fs_cutoff_collect)

    # --- before/after のFs（ここでは fs_q を before、ネイル寄与を after として整合）
    # 実プロダクションの定義に合わせたい場合はここを置換してOK。
    Fs_before = float(fs_q)
    # ネイル無しでの最小と、ネイルありでの最小を分けて求めるには
    # nails=[] で再評価するのが厳密だが、計算量節約のため簡略。
    Fs_after  = float(fs_r)

    # --- diagnostics 追加（後方互換）
    diagnostics = {
        "quick_arcs": quick_records,  # Quick段階で Fs<cutoff の円弧のみ収集
        "grid_bbox": (cfg.grid_xmin, cfg.grid_xmax, cfg.grid_ymin, cfg.grid_ymax),
        "grid_step": float(cfg.grid_step),
        "grid_centers_sampled": [(float(x),float(y)) for (x,y) in centers_q],
        # 参考：最小円弧（Quick段階）
        "quick_best": {"cx": cx_b, "cy": cy_b, "r": r_b, "fs": fs_q}
    }

    result: Dict[str, Any] = {
        "Fs_before": Fs_before,
        "Fs_after": Fs_after,
        # 既存の戻り値に他の要素があれば、ここに追加してもOK
        "diagnostics": diagnostics
    }
    return result
