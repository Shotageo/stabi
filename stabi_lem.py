# stabi_lem.py — LEM core with phreatic line (linear or polyline CSV)
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

# ---------------- Basic data ----------------
@dataclass
class Soil:
    gamma: float  # total unit weight [kN/m3]
    c: float      # cohesion (effective) [kPa = kN/m2]
    phi: float    # friction angle (effective) [deg]

class GroundPL:
    """Piecewise-linear ground surface. X, Y are 1D numpy arrays (monotonic in X)."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        assert len(self.X) >= 2 and self.X.shape == self.Y.shape

    def y_at(self, x: float) -> float:
        X, Y = self.X, self.Y
        if x <= X[0]: return float(Y[0])
        if x >= X[-1]: return float(Y[-1])
        i = np.searchsorted(X, x) - 1
        i = max(0, min(i, len(X)-2))
        t = (x - X[i]) / (X[i+1]-X[i])
        return float((1-t)*Y[i] + t*Y[i+1])

# ---------------- Example geometry ----------------
def make_ground_example(H: float, L: float) -> GroundPL:
    # やや折れた2段勾配（教育・検証用途の標準形）
    X = np.array([0.0, 0.4*L, L])
    Y = np.array([H,   0.65*H, 0.0])
    return GroundPL(X, Y)

def make_interface1_example(H: float, L: float):
    x = np.linspace(0.0, L, 64)
    g = make_ground_example(H, L)
    yg = np.array([g.y_at(float(xx)) for xx in x])
    y = np.maximum(0.0, yg - (0.35*H + 0.05*yg))
    return (x, y)

def make_interface2_example(H: float, L: float):
    x = np.linspace(0.0, L, 64)
    g = make_ground_example(H, L)
    yg = np.array([g.y_at(float(xx)) for xx in x])
    y = np.maximum(0.0, yg - (0.60*H + 0.05*yg))
    return (x, y)

def _interp_xy(x_src: np.ndarray, y_src: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x_src, y_src, left=y_src[0], right=y_src[-1])

def clip_interfaces_to_ground(ground: GroundPL, interfaces: list[tuple[np.ndarray,np.ndarray]], xq: np.ndarray):
    out = []
    yg = np.array([ground.y_at(float(x)) for x in xq])
    for (xi, yi) in interfaces:
        yiq = _interp_xy(np.asarray(xi), np.asarray(yi), xq)
        out.append(np.minimum(yiq, yg))
    return out

# ---------------- Water line (phreatic) ----------------
@dataclass
class WaterLine:
    """Simple 2-point phreatic line (x=0->y0, x=L->y1), clipped to ground."""
    x0: float
    x1: float
    y0: float
    y1: float
    def y_at(self, x: float, ground: GroundPL) -> float:
        if x <= self.x0: y = self.y0
        elif x >= self.x1: y = self.y1
        else:
            t = (x - self.x0) / (self.x1 - self.x0)
            y = (1-t)*self.y0 + t*self.y1
        return min(float(y), float(ground.y_at(x)))  # never above ground

@dataclass
class WaterPolyline:
    """Phreatic line defined by arbitrary polyline (from CSV), clipped to ground."""
    X: np.ndarray  # sorted
    Y: np.ndarray
    def y_at(self, x: float, ground: GroundPL) -> float:
        y = float(np.interp(x, self.X, self.Y, left=self.Y[0], right=self.Y[-1]))
        return min(y, float(ground.y_at(x)))

# ---------------- Helpers: circle/arc ----------------
def _circle_y(x: np.ndarray, xc: float, yc: float, R: float) -> np.ndarray:
    dx = x - xc
    rad = np.maximum(0.0, R*R - dx*dx)
    return yc - np.sqrt(rad)

def _arc_intersections_with_ground(ground: GroundPL, xc: float, yc: float, R: float, n_scan: int=600):
    xs = np.linspace(ground.X[0], ground.X[-1], n_scan)
    yg = np.array([ground.y_at(float(xx)) for xx in xs])
    ya = _circle_y(xs, xc, yc, R)
    f = ya - yg
    s = np.sign(f)
    hits = []
    for i in range(len(xs)-1):
        if s[i] == 0.0:
            hits.append(xs[i])
        elif s[i]*s[i+1] < 0.0:
            a, b = xs[i], xs[i+1]
            fa, fb = f[i], f[i+1]
            for _ in range(30):
                m = 0.5*(a+b)
                fm = _circle_y(np.array([m]), xc, yc, R)[0] - ground.y_at(m)
                if fa*fm <= 0: b, fb = m, fm
                else: a, fa = m, fm
            hits.append(0.5*(a+b))
    hits = sorted(hits)
    if len(hits) >= 2:
        return (hits[0], hits[-1])
    return None

# ---------------- Layer picking ----------------
def _layer_index_at(interfaces_clip: list[np.ndarray], xgrid: np.ndarray, x: float, y: float) -> int:
    if not interfaces_clip: return 0
    i = int(np.clip(np.searchsorted(xgrid, x) - 1, 0, len(xgrid)-2))
    def yi_at(arr):
        t = (x - xgrid[i]) / (xgrid[i+1]-xgrid[i] + 1e-12)
        return float((1-t)*arr[i] + t*arr[i+1])
    for k, Yk in enumerate(interfaces_clip, start=1):
        if y >= yi_at(Yk):
            return k-1
    return len(interfaces_clip)

# ---------------- Driving sum (for required T calc) ----------------
def driving_sum_for_R_multi(ground: GroundPL, interfaces: list[tuple[np.ndarray,np.ndarray]],
                            soils: list[Soil], allow_cross: list[bool],
                            xc: float, yc: float, R: float, n_slices: int=40) -> tuple[float,float,float] | None:
    pair = _arc_intersections_with_ground(ground, xc, yc, R)
    if pair is None: return None
    x1, x2 = pair
    if x2 - x1 < 1e-3: return None
    b = (x2 - x1) / n_slices
    Xd = np.linspace(ground.X[0], ground.X[-1], 256)
    ints = clip_interfaces_to_ground(ground, interfaces, Xd) if interfaces else []
    D_sum = 0.0; W_sum = 0.0; S_sum = 0.0
    for i in range(n_slices):
        xm = x1 + (i+0.5)*b
        yg = ground.y_at(xm)
        ys = _circle_y(np.array([xm]), xc, yc, R)[0]
        h = max(0.0, yg - ys)
        if h <= 0.0: continue
        li = _layer_index_at(ints, Xd, xm, 0.5*(ys+yg))
        soil = soils[min(li, len(soils)-1)]
        W = soil.gamma * b * h
        sin_a = (xm - xc) / R
        D_sum += W * max(0.0, sin_a)
        W_sum += W
        S_sum += b
    return (D_sum, W_sum, S_sum)

# ---------------- Fs for a given R (Bishop / Fellenius) with water ----------------
def fs_given_R_multi(ground: GroundPL, interfaces: list[tuple[np.ndarray,np.ndarray]], soils: list[Soil],
                     allow_cross: list[bool], method: str,
                     xc: float, yc: float, R: float, n_slices: int=40,
                     water=None, gamma_w: float=9.81, pore_mode: str="u-only") -> float | None:
    pair = _arc_intersections_with_ground(ground, xc, yc, R)
    if pair is None: return None
    x1, x2 = pair
    if x2 - x1 < 1e-3: return None

    Xd = np.linspace(ground.X[0], ground.X[-1], 256)
    ints = clip_interfaces_to_ground(ground, interfaces, Xd) if interfaces else []

    b = (x2 - x1) / n_slices
    W = np.zeros(n_slices); sin_a = np.zeros(n_slices); cos_a = np.zeros(n_slices)
    ub = np.zeros(n_slices)
    phi = np.zeros(n_slices); cc = np.zeros(n_slices)

    for i in range(n_slices):
        xm = x1 + (i+0.5)*b
        yg = ground.y_at(xm)
        ys = _circle_y(np.array([xm]), xc, yc, R)[0]
        h = max(0.0, yg - ys)
        if h <= 0.0: return None
        li = _layer_index_at(ints, Xd, xm, 0.5*(ys+yg))
        soil = soils[min(li, len(soils)-1)]
        phi[i] = math.radians(float(soil.phi))
        cc[i] = float(soil.c)

        if water and pore_mode == "buoyancy":
            yw = water.y_at(xm, ground)
            h_w = max(0.0, min(yg, yw) - ys)
            h_a = max(0.0, yg - max(ys, yw))
            gamma_eff = soil.gamma * h_a + max(0.0, soil.gamma - gamma_w) * h_w
            W[i] = gamma_eff * b
        else:
            W[i] = soil.gamma * b * h

        s = (xm - xc) / R
        s = max(-1.0, min(1.0, s))
        sin_a[i] = s
        cos_a[i] = math.sqrt(max(0.0, 1.0 - s*s))

        if water:
            yw = water.y_at(xm, ground)
            head = max(0.0, yw - ys)
            ub[i] = gamma_w * head * b
        else:
            ub[i] = 0.0

    if method.startswith("Fellenius"):
        num = 0.0; den = 0.0
        for i in range(n_slices):
            N_eff = W[i]*cos_a[i] - ub[i]
            num += cc[i]*b + max(0.0, N_eff) * math.tan(phi[i])
            den += W[i]*sin_a[i]
        if den <= 0.0: return None
        return float(num / den)

    Fs = 1.20
    for _ in range(60):
        num = 0.0; den = 0.0
        for i in range(n_slices):
            S = cc[i]*b + (W[i] - ub[i]) * math.tan(phi[i])
            m = cos_a[i] + (math.tan(phi[i])*sin_a[i])/max(Fs, 1e-9)
            num += S / max(m, 1e-9)
            den += W[i]*sin_a[i]
        if den <= 0.0: return None
        Fs_new = num / den
        if not math.isfinite(Fs_new): return None
        if abs(Fs_new - Fs) < 1e-4: return float(Fs_new)
        Fs = 0.5*Fs + 0.5*Fs_new
    return float(Fs)

# ---------------- Entry/exit → R generation (Quick) ----------------
def arcs_from_center_by_entries_multi(
    ground: GroundPL, soils: list[Soil], xc: float, yc: float,
    n_entries: int=220, method: str="Fellenius",
    depth_min: float=0.5, depth_max: float=4.0,
    interfaces: list[tuple[np.ndarray,np.ndarray]]|None=None, allow_cross: list[bool]|None=None,
    quick_mode: bool=True, n_slices_quick: int=12,
    limit_arcs_per_center: int=120, probe_n_min: int=81,
    water=None, gamma_w: float=9.81, pore_mode: str="u-only",
):
    """Generate slip circles for a fixed center by pairing entry/exit points along ground."""
    L0, L1 = float(ground.X[0]), float(ground.X[-1])
    Xs = np.linspace(L0, L1, int(n_entries))
    Yg = np.array([ground.y_at(float(x)) for x in Xs])

    out = 0
    min_span = max(0.04*(L1-L0), 0.8)
    step = max(1, len(Xs)//probe_n_min)
    for i in range(0, len(Xs)-2, step):
        x1 = float(Xs[i]); y1 = float(Yg[i])
        R1 = math.hypot(x1 - xc, y1 - yc)
        for j in range(i+2, len(Xs), step):
            x2 = float(Xs[j]); y2 = float(Yg[j])
            if x2 - x1 < min_span: continue
            R2 = math.hypot(x2 - xc, y2 - yc)
            if abs(R2 - R1) > 0.03*R1:  # 同一円近似
                continue
            R = 0.5*(R1 + R2)

            xm = 0.5*(x1+x2)
            yg = ground.y_at(xm)
            ys = yc - math.sqrt(max(0.0, R*R - (xm - xc)**2))
            dvert = max(0.0, yg - ys)
            if not (depth_min <= dvert <= depth_max):
                continue

            Fs_q = fs_given_R_multi(
                ground, interfaces or [], soils, allow_cross or [], method,
                xc, yc, R, n_slices=n_slices_quick,
                water=water, gamma_w=gamma_w, pore_mode=pore_mode
            )
            if Fs_q is None or not math.isfinite(Fs_q):
                continue

            yield (x1, x2, R, float(Fs_q))
            out += 1
            if out >= limit_arcs_per_center:
                return

# ---------------- Arc sampling for plot ----------------
def arc_sample_poly_best_pair(ground: GroundPL, xc: float, yc: float, R: float, n: int=201):
    pr = _arc_intersections_with_ground(ground, xc, yc, R)
    if pr is None: return None
    return (float(pr[0]), float(pr[1]))
