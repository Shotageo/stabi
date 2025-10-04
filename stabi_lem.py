# stabi_lem.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
import math
from typing import List, Tuple, Callable, Optional
import numpy as np

# ========= データ構造 =========
@dataclass
class Soil:
    gamma: float   # kN/m3
    c: float       # kPa (=kN/m2)
    phi: float     # deg

@dataclass
class CircleSlip:
    xc: float
    yc: float
    R: float

@dataclass
class Nail:
    x1: float; y1: float
    x2: float; y2: float
    spacing: float            # m（本→m換算）
    T_yield: float            # kN/本
    bond_strength: float      # kN/m
    embed_length_each_side: float = 0.5  # m

# ========= 幾何補助 =========
def _dot(ax, ay, bx, by): return ax*bx + ay*by
def _norm(ax, ay): return math.hypot(ax, ay)
def _angle_between(ax, ay, bx, by):
    na = _norm(ax, ay); nb = _norm(bx, by)
    if na == 0 or nb == 0: return 0.0
    cosv = max(-1.0, min(1.0, _dot(ax, ay, bx, by)/(na*nb)))
    return math.acos(cosv)

def circle_xy_from_theta(slip: CircleSlip, theta: np.ndarray):
    return slip.xc + slip.R*np.cos(theta), slip.yc + slip.R*np.sin(theta)

def circle_theta_from_x(slip: CircleSlip, x: np.ndarray):
    v = np.clip((x - slip.xc) / max(slip.R, 1e-9), -1.0, 1.0)
    return np.arccos(v)

def line_circle_intersections_segment(x1, y1, x2, y2, slip: CircleSlip) -> List[Tuple[float,float,float]]:
    """返り値：(x, y, theta) ; theta は接点の円パラメータ角"""
    dx, dy = (x2-x1), (y2-y1)
    fx, fy = (x1-slip.xc), (y1-slip.yc)
    A = dx*dx + dy*dy
    B = 2*(fx*dx + fy*dy)
    C = fx*fx + fy*fy - slip.R*slip.R
    disc = B*B - 4*A*C
    if A == 0 or disc < 0: return []
    rt = math.sqrt(disc); pts = []
    for s in (-1.0, 1.0):
        t = (-B + s*rt)/(2*A)
        if 0.0 <= t <= 1.0:
            x = x1 + t*dx; y = y1 + t*dy
            th = math.atan2(y-slip.yc, x-slip.xc)
            if th < 0: th += 2*math.pi
            pts.append((x, y, th))
    return pts

# ========= スライス生成（弧長 Lb と θ境界を付与） =========
def generate_slices_on_arc(
    ground_y_at: Callable[[np.ndarray], np.ndarray],
    slip: CircleSlip,
    n_slices: int,
    x_min: float,
    x_max: float,
    soil_gamma: float
) -> List[dict]:
    """
    地表 y_g(x) と円弧の上下解を比較し、地表より下の円弧を採用。
    “最長の連続区間”だけを対象に、x等分 → θ境界（θ_a, θ_b） を算出。
    スライス辞書に以下を格納:
      alpha, width(b), area, W, theta_a, theta_b, Lb (=R·Δθ), x_a, x_b
    """
    X = np.linspace(x_min, x_max, 1201)
    theta = circle_theta_from_x(slip, X)
    yu = slip.yc + slip.R*np.sin(theta)
    yl = slip.yc - slip.R*np.sin(theta)
    yg = ground_y_at(X)

    under_u = yu <= yg
    under_l = yl <= yg
    yc = np.where(under_u & under_l, np.maximum(yu, yl),
         np.where(under_u, yu, np.where(under_l, yl, np.nan)))
    valid = np.isfinite(yc)
    if not np.any(valid): return []

    idx = np.where(valid)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, brk+1]; ends = np.r_[brk, len(idx)-1]
    lengths = idx[ends] - idx[starts]
    k = int(np.argmax(lengths))
    i0, i1 = idx[starts[k]], idx[ends[k]]
    xL, xR = X[i0], X[i1]
    if xR - xL <= 1e-6: return []

    xs = np.linspace(xL, xR, n_slices+1)
    sls = []
    for i in range(n_slices):
        x_a, x_b = xs[i], xs[i+1]
        xm = 0.5*(x_a+x_b)

        th_a = float(circle_theta_from_x(slip, np.array([x_a]))[0])
        th_b = float(circle_theta_from_x(slip, np.array([x_b]))[0])
        # 円の接線ベクトル：(-sinθ, cosθ)
        tx, ty = -math.sin((th_a+th_b)/2), math.cos((th_a+th_b)/2)
        alpha = math.atan2(ty, tx)

        # 弧長
        dth = abs(th_b - th_a)
        Lb = slip.R * dth

        # 台形積分で面積と重量
        grid = np.linspace(x_a, x_b, 9)
        thg = circle_theta_from_x(slip, grid)
        yu_g = slip.yc + slip.R*np.sin(thg)
        yl_g = slip.yc - slip.R*np.sin(thg)
        yg_g = ground_y_at(grid)
        under_u_g = yu_g <= yg_g
        under_l_g = yl_g <= yg_g
        yc_g = np.where(under_u_g & under_l_g, np.maximum(yu_g, yl_g),
                np.where(under_u_g, yu_g, np.where(under_l_g, yl_g, yg_g)))
        h = np.maximum(yg_g - yc_g, 0.0)
        area = float(np.trapz(h, grid))
        W = soil_gamma * area  # kN（奥行1m）

        sls.append({
            "alpha": alpha, "width": (x_b-x_a), "area": area, "W": W,
            "theta_a": th_a, "theta_b": th_b, "Lb": Lb,
            "x_a": x_a, "x_b": x_b
        })
    return sls

# ========= Bishop 未補強（c·Lb を使用） =========
def bishop_fs_unreinforced(slices: List[dict], soil: Soil, max_iter=80, tol=1e-5) -> float:
    phi = math.radians(soil.phi); c = soil.c
    Fs = 1.2
    for _ in range(max_iter):
        num = den = 0.0
        for s in slices:
            alpha = s["alpha"]; W = s["W"]; Lb = s.get("Lb", s["width"])  # 既存互換で width も許容
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * Lb + (Np * math.tan(phi))  # ← c·Lb（弧長）
            num += shear_res
            den += W * math.sin(alpha)
        Fs_new = num / max(den, 1e-9)
        if abs(Fs_new - Fs) < tol:
            return max(Fs_new, 1e-6)
        Fs = Fs_new
    return max(Fs, 1e-6)

# ========= Bishop + Nails（デフォルト：スライス法。オプションで全体モーメントも可） =========
def bishop_fs_with_nails(
    slices: List[dict], soil: Soil, slip: CircleSlip, nails: List[Nail],
    mode: str = "slice",  # "slice" or "moment"
    max_iter: int = 80, tol: float = 1e-5
) -> float:
    """
    mode="slice": ネイルの接線成分 T_tan を該当スライスの抵抗せん断に加算
    mode="moment": 旧来の全体モーメント加算（参考）
    """
    # 事前にスライスの x 区間を取り出し
    x_edges = [(s["x_a"], s["x_b"]) for s in slices]

    # ネイル→交点θ→交点x→スライス番号
    def nail_hits():
        hits = []
        for nl in nails:
            pts = line_circle_intersections_segment(nl.x1, nl.y1, nl.x2, nl.y2, slip)
            if not pts: 
                continue
            # 交点ごとに有効軸力（spacing換算・付着/降伏制限）と接線成分
            for (xp, yp, th) in pts:
                # 円弧接線
                tx, ty = -math.sin(th), math.cos(th)
                # ネイル軸
                nx, ny = (nl.x2 - nl.x1), (nl.y2 - nl.y1)
                nlen = math.hypot(nx, ny)
                if nlen == 0: 
                    continue
                nx /= nlen; ny /= nlen
                delta = _angle_between(nx, ny, tx, ty)
                L_embed = 2.0 * nl.embed_length_each_side
                T_axial_1 = min(nl.T_yield, nl.bond_strength * L_embed)   # kN/本
                T_axial   = T_axial_1 / max(nl.spacing, 1e-6)              # kN/m
                T_tan = T_axial * abs(math.cos(delta))

                hits.append((xp, yp, th, T_tan))
        return hits

    phi = math.radians(soil.phi); c = soil.c
    Fs = bishop_fs_unreinforced(slices, soil, max_iter=30, tol=1e-4)

    if mode == "moment":
        # 参考：全体モーメント寄与（以前の実装）
        R = slip.R
        for _ in range(max_iter):
            num = den = 0.0; M_reinf = 0.0
            for s in slices:
                alpha = s["alpha"]; W = s["W"]; Lb = s.get("Lb", s["width"])
                m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
                Np = (W * math.cos(alpha)) / m
                shear_res = c * Lb + (Np * math.tan(phi))
                num += shear_res
                den += W * math.sin(alpha)
            for (_, _, __, T_tan) in nail_hits():
                M_reinf += T_tan * R
            Fs_new = (num * R + M_reinf) / max(den * R, 1e-9)
            if abs(Fs_new - Fs) < tol: return max(Fs_new, 1e-6)
            Fs = Fs_new
        return max(Fs, 1e-6)

    # === デフォルト：スライス法 ===
    # 各反復でネイル寄与を再評価（Fs依存は弱いが手順統一）
    for _ in range(max_iter):
        num = den = 0.0

        # 交点と T_tan を一括算定
        hits = nail_hits()

        # 交点x→スライス i を探す（端は右閉じにしない）
        add_T = [0.0]*len(slices)
        for (xp, _, __, T_tan) in hits:
            for i, (xa, xb) in enumerate(x_edges):
                if xa <= xp < xb:
                    add_T[i] += T_tan
                    break

        for i, s in enumerate(slices):
            alpha = s["alpha"]; W = s["W"]; Lb = s.get("Lb", s["width"])
            m = 1.0 + (math.tan(phi) * math.sin(alpha)) / max(Fs, 1e-6)
            Np = (W * math.cos(alpha)) / m
            shear_res = c * Lb + (Np * math.tan(phi)) + add_T[i]  # ★ ネイル接線成分を抵抗せん断に加算
            num += shear_res
            den += W * math.sin(alpha)

        Fs_new = num / max(den, 1e-9)
        if abs(Fs_new - Fs) < tol: return max(Fs_new, 1e-6)
        Fs = Fs_new

    return max(Fs, 1e-6)
