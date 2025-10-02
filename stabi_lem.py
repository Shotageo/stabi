# stabi_lem.py — full replacement (centers→R生成/探索/FS評価/水位CSV/オフセット/可視化)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Callable, Dict, Any
import numpy as np, math, csv, os

# ----------------- 基本定数 -----------------
DEG = math.pi/180.0
EPS = 1e-12
GAMMA_W = 9.81  # kN/m3

# ----------------- 基本データ構造 -----------------
@dataclass
class Soil:
    gamma: float          # kN/m3  （後方互換：従来の総重量 or 飽和重量のどちらでもOK）
    c: float              # kPa
    phi: float            # deg
    # 拡張（任意）：浸水評価の精度向上用
    gamma_sat: Optional[float] = None  # 飽和単位体積重量 (kN/m3)
    gamma_dry: Optional[float] = None  # 乾燥単位体積重量 (kN/m3)

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
            y[i] = (1-t)*y0 + t*y1
        return y if y.shape != () else float(y)

# ----------------- 円×折れ線 交点 -----------------
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
    # ほぼ同一点の重複除去
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

# ----------------- 円弧サンプリング（最良の地表区間） -----------------
def arc_sample_poly_best_pair(pl: GroundPL, xc, yc, R, n=241, y_floor: float = -float("inf")):
    """
    地表折れ線と円の交点列から、最良の区間 [x1,x2] を選びサンプルする。
    y_floor 下限（既定: -inf）→ 深い円弧の取り逃しを防止。
    """
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
        # 下限チェック（以前は0.0でバイアス）
        if np.any(ys < y_floor - 1e-9):
            continue
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

# ----------------- 複層処理（地層境界：上から順にクリップ） -----------------
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

# ----------------- 水位モデル（CSV / オフセット / ru） -----------------
def _interp1d_linear(xq: np.ndarray, xp: np.ndarray, yp: np.ndarray):
    """簡易1D線形補間（外挿は端値保持）"""
    xq = np.asarray(xq, dtype=float)
    out = np.empty_like(xq)
    out[xq <= xp[0]] = yp[0]
    out[xq >= xp[-1]] = yp[-1]
    mid = (xq > xp[0]) & (xq < xp[-1])
    out[mid] = np.interp(xq[mid], xp, yp)
    return out

def make_water_from_csv(csv_path: str) -> Dict[str, Any]:
    """
    CSV ロード: 列は [x, z] を想定（ヘッダ有無は自動判定）
    戻り値: {"type":"WT","z": callable(x)->z}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    xs, zs = [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        snif = csv.Sniffer()
        sample = f.read(1024)
        f.seek(0)
        has_header = snif.has_header(sample)
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 2: continue
            try:
                x = float(row[0]); z = float(row[1])
            except:
                continue
            xs.append(x); zs.append(z)
    if len(xs) < 2:
        raise ValueError("CSV must have at least 2 points (x,z).")
    xp = np.array(xs, dtype=float)
    yp = np.array(zs, dtype=float)
    def zfun(x):
        x = np.asarray(x, dtype=float)
        return _interp1d_linear(x, xp, yp)
    return {"type":"WT", "z": zfun, "meta":{"source":"csv","path":csv_path}}

def make_water_from_offset(ground: GroundPL, offset: float) -> Dict[str, Any]:
    """
    地表からのオフセット（+上 / -下）。例: offset=-1.5 → 地表より1.5m下
    """
    def zfun(x):
        return ground.y_at(x) + offset
    return {"type":"WT", "z": zfun, "meta":{"source":"offset","offset": float(offset)}}

# ----------------- Fs 計算（有効応力 with 水位） -----------------
def _effective_unit_weight_slice(gamma_soil: np.ndarray, submerged_mask: np.ndarray,
                                 gamma_sat_opt: Optional[np.ndarray]=None,
                                 gamma_dry_opt: Optional[np.ndarray]=None):
    """
    シンプル運用：
      - submerged_mask True の区間は γ' ≈ γ_sat - γ_w （情報なければ gamma_soil - γ_w と近似）
      - 乾燥側は gamma_dry_opt があればそれ、なければ gamma_soil
    """
    if gamma_sat_opt is None:
        gamma_sat_opt = gamma_soil
    if gamma_dry_opt is None:
        gamma_dry_opt = gamma_soil
    gamma_sub = np.maximum(0.1, gamma_sat_opt - GAMMA_W)  # 安全側に僅かな下限
    gamma_eff = np.where(submerged_mask, gamma_sub, gamma_dry_opt)
    return gamma_eff

def _water_u_at_slice(xmid: np.ndarray, y_arc: np.ndarray, water: Optional[Dict[str,Any]]):
    if water is None or water.get("type") == "dry":
        return np.zeros_like(xmid), np.zeros_like(xmid, dtype=bool), np.full_like(xmid, -1e9)
    if water["type"] == "ru":
        # 簡易ruモデル：底面有効法に u を一定比として入れる近似（実務ではWT推奨）
        ru = float(water.get("ru", 0.0))
        # ここでは単位を合わせる簡易近似として、N' = N - (ru * N) を後段で扱う方が自然だが、
        # 実装簡素化のため、u*b = ru*N と見なす（bは後段にあるため N_eff計算時に ru を使用）
        # → 戻りは u=0 とし、ruは呼び出し側で使う。
        return np.zeros_like(xmid), np.zeros_like(xmid, dtype=bool), np.full_like(xmid, -1e9)
    if water["type"] == "WT":
        z = water["z"]
        zw = z(xmid) if callable(z) else float(z)
        hw = np.maximum(0.0, zw - y_arc)
        u = GAMMA_W * hw
        submerged = (y_arc < zw)
        return u, submerged, zw
    return np.zeros_like(xmid), np.zeros_like(xmid, dtype=bool), np.full_like(xmid, -1e9)

def fs_fellenius_poly_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                            xc, yc, R, n_slices=40, water: Optional[Dict[str,Any]] = None) -> Optional[float]:
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201), y_floor=-float("inf"))
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

    gamma0, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    tanp  = np.tan(phi*DEG)

    # 水位・有効応力
    u, submerged_mask, zw = _water_u_at_slice(xmid, y_arc, water)
    # optional soil gammas
    gamma_sat = None
    gamma_dry = None
    # soils がスライスで混在するので、簡易に「表示用の代表値」は使わず、
    # ここでは gamma0 をベースに浸水部のみ γ' 近似へ置換
    gamma_eff = _effective_unit_weight_slice(gamma0, submerged_mask, gamma_sat, gamma_dry)

    b     = dx / cos_a
    W     = gamma_eff * hmid * dx  # 自重：浸水部は水中重量で抑制
    N     = W * np.cos(alpha)

    # ruモデル対応（uを個別に与えずNを縮退）
    if water is not None and water.get("type") == "ru":
        ru = float(water.get("ru", 0.0))
        N_eff = (1.0 - ru) * N
    else:
        N_eff = N - u * b

    num   = float(np.sum(c*b + N_eff * tanp))
    den   = float(np.sum(W*np.sin(alpha)))
    if den <= 0: return None
    Fs    = num/den
    return Fs if (np.isfinite(Fs) and Fs>0) else None

def fs_bishop_poly_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                         xc, yc, R, n_slices=40, water: Optional[Dict[str,Any]] = None) -> Optional[float]:
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201), y_floor=-float("inf"))
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

    gamma0, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    tanp  = np.tan(phi*DEG)

    # 水位・有効応力
    u, submerged_mask, zw = _water_u_at_slice(xmid, y_arc, water)
    gamma_eff = _effective_unit_weight_slice(gamma0, submerged_mask, None, None)

    b     = dx / cos_a
    W     = gamma_eff * hmid * dx
    N     = W * np.cos(alpha)

    Fs    = 1.3
    for _ in range(160):
        if water is not None and water.get("type") == "ru":
            ru = float(water.get("ru", 0.0))
            N_eff = (1.0 - ru) * N
        else:
            N_eff = N - u * b
        denom_a = 1.0 + (tanp*np.tan(alpha))/max(Fs,1e-12)
        num = float(np.sum((c*b + N_eff*tanp)/denom_a))
        den = float(np.sum(W*np.sin(alpha)))
        if den <= 0: return None
        Fs_new = num/den
        if not (np.isfinite(Fs_new) and Fs_new>0): return None
        if abs(Fs_new-Fs) < 1e-5: return float(Fs_new)
        Fs = Fs_new
    return float(Fs)

def fs_given_R_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                     method: str, xc: float, yc: float, R: float, n_slices: int,
                     water: Optional[Dict[str,Any]] = None) -> Optional[float]:
    if method.lower().startswith("b"):
        return fs_bishop_poly_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices, water=water)
    else:
        return fs_fellenius_poly_multi(ground, interfaces, soils, allow_cross, xc, yc, R, n_slices=n_slices, water=water)

# 追加：必要抑止力計算で使う駆動項 D = Σ(W sinα)
def driving_sum_for_R_multi(ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
                            xc, yc, R, n_slices=40, water: Optional[Dict[str,Any]] = None) -> Optional[Tuple[float,float,float]]:
    s = arc_sample_poly_best_pair(ground, xc, yc, R, n=max(2*n_slices+1,201), y_floor=-float("inf"))
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

    gamma0, c, phi = base_soil_vectors_multi(ground, interfaces, soils, xmid, y_arc)
    u, submerged_mask, zw = _water_u_at_slice(xmid, y_arc, water)
    gamma_eff = _effective_unit_weight_slice(gamma0, submerged_mask, None, None)

    W     = gamma_eff * hmid * dx
    D_sum = float(np.sum(W*np.sin(alpha)))
    if D_sum <= 0 or not np.isfinite(D_sum): return None
    return D_sum, float(x1), float(x2)

# ----------------- 円弧候補の生成（中心→{x,depth}格子からRを作る） -----------------
def _R_from_x_and_depth(ground: GroundPL, xc: float, yc: float, x: float, h: float) -> float:
    yg = float(ground.y_at(x))
    return math.sqrt(max(0.0, (yg - h - yc)**2 + (x - xc)**2))

def arcs_from_center_by_entries_multi(
    ground: GroundPL, soils: List[Soil], xc: float, yc: float,
    n_entries: int, method: str,
    depth_min: float, depth_max: float,
    interfaces: List[GroundPL], allow_cross: List[bool],
    quick_mode: bool, n_slices_quick: int,
    limit_arcs_per_center: int, probe_n_min: int,
    water: Optional[Dict[str,Any]] = None,
) -> Iterable[Tuple[float,float,float,float]]:
    """
    生成器: (x1, x2, R, Fs_quick) を yield
      - ground.X[0]..ground.X[-1] を n_entries でサンプルした x と
        depth ∈ [depth_min, depth_max]（等間隔）から R を作成。
      - その R で arc_sample_poly_best_pair により (x1,x2) を決め、
        quick（Fellenius / n_slices_quick）で Fs を評価。
    """
    if n_entries < 2 or depth_max <= 0 or depth_max < depth_min:
        return

    x0, x1 = float(ground.X[0]), float(ground.X[-1])
    xs = np.linspace(x0+1e-6, x1-1e-6, n_entries)
    n_depth = max(5, int(round((depth_max - depth_min)/max(0.5, (depth_max - depth_min)/10))) + 1)
    hs = np.linspace(max(0.01, depth_min), depth_max, n_depth)

    count = 0
    for xi in xs:
        for h in hs:
            R = _R_from_x_and_depth(ground, xc, yc, float(xi), float(h))
            if not (np.isfinite(R) and R > 0.5):
                continue
            n_probe = max(probe_n_min, 201) if not quick_mode else max(101, probe_n_min)
            s = arc_sample_poly_best_pair(ground, xc, yc, R, n=n_probe, y_floor=-float("inf"))
            if s is None:
                continue
            a_x1, a_x2, _xs, _ys, _h = s
            Fs_q = fs_fellenius_poly_multi(ground, interfaces, soils, allow_cross, xc, yc, R,
                                           n_slices=n_slices_quick, water=water)
            if Fs_q is None:
                continue
            yield float(a_x1), float(a_x2), float(R), float(Fs_q)
            count += 1
            if count >= limit_arcs_per_center:
                return

# ----------------- 広域→局所：最小FS探索（Coarse→Quick→Refine） -----------------
def search_min_fs_multi(
    ground: GroundPL, interfaces: List[GroundPL], soils: List[Soil], allow_cross: List[bool],
    method: str = "bishop",
    grid_box: Optional[Tuple[float,float,float,float]] = None,
    coarse: Tuple[int,int] = (15, 11),          # 中心グリッド（x,y）
    quick_k: int = 16,                          # Coarse上位をQuickへ
    n_slices: Tuple[int,int] = (40, 90),        # (Quick, Refine)
    depth: Tuple[float, Optional[float]] = (0.5, None),
    limit_per_center: int = 24,
    seed: int = 42,
    water: Optional[Dict[str,Any]] = None,
) -> Optional[Dict[str,Any]]:
    """
    戻り: {"FS":float,"center":(xc,yc),"radius":R,"span":(x1,x2),"meta":{...}}
    """
    rng = np.random.default_rng(seed)
    H = float(max(ground.Y))
    L = float(max(ground.X) - min(ground.X))
    depth_min, depth_max = depth[0], depth[1] or (0.9*H)

    x0, x1 = float(ground.X[0]), float(ground.X[-1])
    if not grid_box:
        grid_box = (x0 - 0.30*L, x1 + 0.30*L,
                    max(ground.Y)+0.20*H, max(ground.Y)+1.50*H)
    gx0, gx1, gy0, gy1 = grid_box
    xs = np.linspace(gx0, gx1, coarse[0])
    ys = np.linspace(gy0, gy1, coarse[1])

    # --- Coarse: 広域疎探索（Fellenius/Quick切り上げ） ---
    cand = []
    for xc in xs:
        for yc in ys:
            for (a_x1,a_x2,R,Fs_q) in arcs_from_center_by_entries_multi(
                ground, soils, xc, yc, n_entries=11, method=method,
                depth_min=depth_min, depth_max=depth_max,
                interfaces=interfaces, allow_cross=allow_cross,
                quick_mode=True, n_slices_quick=n_slices[0],
                limit_arcs_per_center=limit_per_center, probe_n_min=81,
                water=water):
                cand.append((Fs_q, xc, yc, R, a_x1, a_x2))
    if not cand:
        return None
    cand.sort(key=lambda t: t[0])
    top = cand[:max(1, quick_k)]

    # --- Quick→Refine: 上位近傍の局所精密化 ---
    best = None
    for Fs_q, xc, yc, R, a_x1, a_x2 in top:
        # 近傍微調整（中心±Δ, R±Δ）：ランダム＆格子のハイブリッド
        for _ in range(4):
            for dxc in np.linspace(-0.12*L, 0.12*L, 5):
                for dyc in np.linspace(-0.12*H, 0.12*H, 5):
                    for dR in np.linspace(-0.18*R, 0.18*R, 5):
                        xc2 = xc + dxc + 0.02*L*rng.normal()
                        yc2 = yc + dyc + 0.02*H*rng.normal()
                        R2  = max(0.5, R + dR + 0.03*R*rng.normal())
                        Fs = fs_given_R_multi(ground, interfaces, soils, allow_cross,
                                              method, xc2, yc2, R2, n_slices=n_slices[1], water=water)
                        if Fs is None: continue
                        if (best is None) or (Fs < best["FS"]) or (
                            abs(Fs - best["FS"]) < 1e-6 and R2 > best["radius"]):
                            best = {"FS":Fs, "center":(float(xc2), float(yc2)),
                                    "radius":float(R2), "span":(float(a_x1), float(a_x2)),
                                    "meta":{"stage":"refine","tie_break":"longer_arc","seed":int(seed)}}
    # 念のため候補ゼロを回避
    if best is None:
        Fs_q, xc, yc, R, a_x1, a_x2 = cand[0]
        best = {"FS":Fs_q, "center":(float(xc), float(yc)),
                "radius":float(R), "span":(float(a_x1), float(a_x2)),
                "meta":{"stage":"coarse_only","seed":int(seed)}}
    return best

# ----------------- サンプル地形（UI用） -----------------
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

# ----------------- 可視化ユーティリティ -----------------
# Plot style（Theme/Tight layout/Legend切替）—★あなたの要望に合わせて同梱★
def plot_result(ground: GroundPL, result: Dict[str,Any], ax=None, show_legend=True, theme="default"):
    """
    依存：matplotlib（UI側で用意）/ seabornは不使用。
    theme: "default" or "dark"
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if theme == "dark":
        ax.set_facecolor("#111111")
        ax.figure.set_facecolor("#111111")
        ax.tick_params(colors="#DDDDDD")
        for spine in ax.spines.values(): spine.set_color("#888888")
        fg = "#DDDDDD"
    else:
        fg = "#222222"
    # 地表
    ax.plot(ground.X, ground.Y, label="Ground")
    # 最小FS円弧（投影）
    xc, yc = result["center"]
    R = result["radius"]
    x1, x2 = result["span"]
    xs = np.linspace(x1, x2, 200)
    ys = yc - np.sqrt(np.maximum(EPS, R*R - (xs - xc)**2))
    ax.plot(xs, ys, linestyle="--", label=f"Critical arc FS={result['FS']:.3f}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    if show_legend: ax.legend()
    # Tight layout
    try:
        ax.figure.tight_layout()
    except Exception:
        pass
    return ax

__all__ = [
    "Soil","GroundPL",
    "make_ground_example","make_interface1_example","make_interface2_example",
    "arc_sample_poly_best_pair","circle_polyline_intersections",
    "clip_interfaces_to_ground","barrier_y_from_flags","base_soil_vectors_multi",
    "fs_fellenius_poly_multi","fs_bishop_poly_multi","fs_given_R_multi",
    "arcs_from_center_by_entries_multi","driving_sum_for_R_multi",
    "make_water_from_csv","make_water_from_offset","search_min_fs_multi","plot_result",
]
