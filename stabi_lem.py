# stabi_lem.py
# -*- coding: utf-8 -*-
"""
安定板２（未補強）用の最小セット：
- Soil / CircleSlip のデータ構造
- 円弧と地表の幾何ユーティリティ
- 円弧上のスライス生成（地表下側の円弧を採用）
- Bishop簡略法（未補強）による Fs 計算
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
import math
import numpy as np


# ===================== データ構造 =====================
@dataclass
class Soil:
    gamma: float  # kN/m3
    c: float      # kPa(=kN/m2)
    phi: float    # deg


@dataclass
class CircleSlip:
    xc: float
    yc: float
    R: float


# ===================== 幾何ユーティリティ =====================
def circle_xy_from_theta(slip: CircleSlip, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return slip.xc + slip.R * np.cos(theta), slip.yc + slip.R * np.sin(theta)

def circle_theta_from_x(slip: CircleSlip, x: np.ndarray) -> np.ndarray:
    """x座標から円パラメータ角θ（0〜π）を返す。|x-xc|>R のときは acos の都合で端にクリップ。"""
    v = np.clip((np.asarray(x) - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)


# ===================== スライス生成 =====================
def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    """
    地表 y_g(x) と円弧の上下解 y_u=yc+R sinθ, y_l=yc-R sinθ を比較。
    地表以下(<=y_g)に存在する円弧側を採用し、最長連続区間を x 等分してスライス化。
    スライス辞書：{alpha, width, W, area, x_a, x_b}
      alpha: すべり面傾斜角（接線の角度）[-π, π]
      width: スライス幅 (x_b - x_a) [m]
      W    : 自重（奥行1m）[kN]
      area : スライス断面積 [m2]（図心計算には使っていない）
      x_a, x_b: スライスの左右端 x
    """
    # 円弧の上下解
    X = np.linspace(x_min, x_max, 1201)
    theta = circle_theta_from_x(slip, X)
    yu = slip.yc + slip.R * np.sin(theta)   # 上側解
    yl = slip.yc - slip.R * np.sin(theta)   # 下側解
    yg = ground_y_at(X)

    under_u = (yu <= yg)
    under_l = (yl <= yg)
    # 地表以下に存在する側を採用。両方地表以下なら“より上側”（=値が大きい方）
    yc = np.where(under_u & under_l, np.maximum(yu, yl),
         np.where(under_u, yu, np.where(under_l, yl, np.nan)))
    valid = np.isfinite(yc)
    if not np.any(valid):
        return []

    # 最長の連続区間のみ採用
    idx = np.where(valid)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, brk + 1]
    ends   = np.r_[brk, len(idx) - 1]
    lengths = idx[ends] - idx[starts]
    k = int(np.argmax(lengths))
    i0, i1 = idx[starts[k]], idx[ends[k]]
    xL, xR = float(X[i0]), float(X[i1])
    if xR - xL <= 1e-9:
        return []

    # スライス分割
    xs = np.linspace(xL, xR, n_slices + 1)
    slices: List[dict] = []
    for i in range(n_slices):
        x_a, x_b = float(xs[i]), float(xs[i + 1])
        xm = 0.5 * (x_a + x_b)
        th_m = float(circle_theta_from_x(slip, np.array([xm]))[0])
        # 円の接線ベクトル (-sinθ, cosθ) → 角度 alpha
        tx, ty = -math.sin(th_m), math.cos(th_m)
        alpha = math.atan2(ty, tx)  # [-π, π]

        # 自重（地表−すべり面間を台形で積分）
        grid = np.linspace(x_a, x_b, 9)
        thg = circle_theta_from_x(slip, grid)
        yu_g = slip.yc + slip.R * np.sin(thg)
        yl_g = slip.yc - slip.R * np.sin(thg)
        yg_g = ground_y_at(grid)
        under_u_g = (yu_g <= yg_g)
        under_l_g = (yl_g <= yg_g)
        yc_seg = np.where(under_u_g & under_l_g, np.maximum(yu_g, yl_g),
                np.where(under_u_g, yu_g, np.where(under_l_g, yl_g, yg_g)))
        h = np.maximum(yg_g - yc_seg, 0.0)  # [m]
        area = float(np.trapz(h, grid))     # [m2]（奥行1m想定）
        W = soil_gamma * area               # [kN]

        slices.append({
            "alpha": alpha,
            "width": (x_b - x_a),
            "W": W,
            "area": area,
            "x_a": x_a,
            "x_b": x_b,
        })

    return slices


# ===================== Bishop簡略法（未補強） =====================
def bishop_fs_unreinforced(
    slices: List[dict],
    soil: Soil,
    max_iter: int = 80,
    tol: float = 1e-5
) -> float:
    """
    Bishop簡略法（未補強）。
    - c はスライス幅 b として c*b を採用（円弧弧長は使わない）
    - 反復で m を更新しつつ Fs 収束解を得る
    """
    phi = math.radians(float(soil.phi))
    c   = float(soil.c)

    Fs = 1.2  # 初期値
    for _ in range(max_iter):
        num = 0.0
        den = 0.0
        for s in slices:
            alpha = float(s["alpha"])
            W     = float(s["W"])
            b     = float(s["width"])
            # 安定係数 m
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-9)
            # 有効法線力 N'
            Np = (W * math.cos(alpha)) / m
            # せん断抵抗
            shear_res = c * b + (Np * math.tan(phi))
            num += shear_res
            den += W * math.sin(alpha)

        Fs_new = num / max(den, 1e-12)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new

    return max(Fs, 1e-6)
