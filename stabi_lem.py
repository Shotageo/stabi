# stabi_lem.py — LEM core (no water). Fellenius & Bishop(simplified), multilayer, robust intersections.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# =========================
# Basic data types
# =========================
@dataclass
class Soil:
    gamma: float  # kN/m^3
    c: float      # kPa (kN/m^2)
    phi: float    # deg
    @property
    def phi_rad(self) -> float:
        return np.deg2rad(self.phi)

class GroundPL:
    """Piecewise-linear ground surface y(x). X must be sorted (monotonic)."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
        assert X.ndim==1 and Y.ndim==1 and len(X)==len(Y)>=2
        order = np.argsort(X)
        self.X = X[order]
        self.Y = Y[order]
        self.X0 = float(self.X[0]); self.X1 = float(self.X[-1])

    def y_at(self, x):
        # accept scalar or array
        return np.interp(x, self.X, self.Y, left=self.Y[0], right=self.Y[-1])

# =========================
# Example geometries
# =========================
def make_ground_example(H: float, L: float) -> GroundPL:
    X = np.array([0.0, L], dtype=float)
    Y = np.array([H, 0.0], dtype=float)
    return GroundPL(X, Y)

def make_interface1_example(H: float, L: float):
    X = np.array([0.0, L], dtype=float)
    Y = np.array([0.50*H, 0.15*H], dtype=float)
    return (X, Y)

def make_interface2_example(H: float, L: float):
    X = np.array([0.0, L], dtype=float)
    Y = np.array([0.28*H, 0.05*H], dtype=float)
    return (X, Y)

def _interp_polyline_y(Xi: np.ndarray, Yi: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, np.asarray(Xi, dtype=float), np.asarray(Yi, dtype=float),
                     left=Yi[0], right=Yi[-1])

def clip_interfaces_to_ground(ground: GroundPL, interfaces, Xd: np.ndarray):
    Yg = ground.y_at(Xd)
    out = []
    for (Xi, Yi) in interfaces:
        Yi_q = _interp_polyline_y(Xi, Yi, Xd)
        Yi_q = np.clip(Yi_q, 0.0, Yg)
        out.append(Yi_q)
    return out

# =========================
# Circle/ground intersections
# =========================
def _arc_y(x, xc, yc, R):
    inside = np.maximum(0.0, R*R - (x - xc)**2)
    return yc - np.sqrt(inside)

def _intersections_by_sampling(ground: GroundPL, xc, yc, R, N=241):
    a = max(ground.X0, float(xc - R))
    b = min(ground.X1, float(xc + R))
    if b - a <= 1e-8: return None
    xs = np.linspace(a, b, int(max(201, N)))
    yg = ground.y_at(xs)
    ya = _arc_y(xs, xc, yc, R)
    f = ya - yg
    s = np.sign(f)
    idx = np.where(s[:-1]*s[1:] <= 0)[0]
    if idx.size < 2:
        return None
    def root(i):
        x0, x1 = xs[i], xs[i+1]; f0, f1 = f[i], f[i+1]
        if abs(f1 - f0) < 1e-12: return float(0.5*(x0+x1))
        t = -f0/(f1-f0); return float(x0 + t*(x1 - x0))
    x1 = root(int(idx[0])); x2 = root(int(idx[-1]))
    if x2 - x1 < 1e-3: return None
    return (x1, x2)

def circle_ground_intersections(ground: GroundPL, xc: float, yc: float, R: float,
                                N: int = 1201):
    if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(R)): 
        return None
    if R <= 0:
        return None
    a = max(ground.X0, float(xc - R))
    b = min(ground.X1, float(xc + R))
    if b - a <= 1e-6:
        return None
    xs = np.linspace(a, b, int(max(401, N)))
    yg = ground.y_at(xs)
    ya = _arc_y(xs, xc, yc, R)
    f = ya - yg
    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if idx.size < 2:
        return None
    def root(i):
        x0, x1 = xs[i], xs[i+1]; f0, f1 = f[i], f[i+1]
        if abs(f1 - f0) < 1e-12:
            return float(0.5*(x0+x1))
        t = -f0/(f1 - f0)
        return float(x0 + t*(x1 - x0))
    x1 = root(int(idx[0]))
    x2 = root(int(idx[-1]))
    if x2 - x1 < 1e-3:
        return None
    return (x1, x2)

def arc_sample_poly_best_pair(ground: GroundPL, xc: float, yc: float, R: float, n=241):
    res = _intersections_by_sampling(ground, xc, yc, R, N=max(201,int(n)))
    if res is not None:
        return res
    return circle_ground_intersections(ground, xc, yc, R, N=1201)

# =========================
# Layer helpers
# =========================
def _interfaces_y_at(interfaces, x):
    ys = []
    for (Xi, Yi) in interfaces:
        ys.append(float(np.interp(float(x), Xi, Yi, left=Yi[0], right=Yi[-1])))
    return ys  # top→bottom

def _layer_index_at_depth(ground: GroundPL, interfaces, x, y):
    """Return 1-based layer index at (x,y). Layer1: between ground and iface1."""
    if not interfaces:
        return 1
    ys = _interfaces_y_at(interfaces, x)  # [y_if1, y_if2, ...], top→bottom
    if y >= ys[0]:
        return 1
    for i in range(len(ys)-1):
        if ys[i+1] <= y < ys[i]:
            return i+2
    return len(ys)+1

def _soil_for_layer(soils, idx):
    idx = int(max(1, min(idx, len(soils))))
    return soils[idx-1]

def _allowed_max_layer(allow_cross, n_layers):
    """
    Layer1は常に許可。allow_cross[k] が True なら layer (k+2) を許可。
    例: n_layers=3, allow_cross=[True] -> layer1,2まで。 [True, False] -> layer1,2で停止。
    """
    allowed = 1
    for ok in list(allow_cross)[:max(0, n_layers-1)]:
        if ok: allowed += 1
        else: break
    return min(allowed, n_layers)

# =========================
# Slice integrals
# =========================
def _slice_geometry(ground: GroundPL, xc, yc, R, x_edges):
    xm = 0.5*(x_edges[:-1] + x_edges[1:])
    ya = _arc_y(xm, xc, yc, R)
    S = np.maximum(1e-12, R*R - (xm - xc)**2)
    slope = (xm - xc)/np.sqrt(S)
    alpha = np.arctan(slope)
    dx = (x_edges[1:] - x_edges[:-1])
    b = dx/np.cos(np.clip(alpha, -1.553, 1.553))  # avoid blow-up
    h = np.maximum(0.0, ground.y_at(xm) - ya)
    return xm, ya, dx, alpha, b, h

def _weights_by_layer(ground: GroundPL, interfaces, soils, xm, ya, dx, h):
    W = np.zeros_like(xm)
    for i, x in enumerate(xm):
        yb = ya[i]
        layer = _layer_index_at_depth(ground, interfaces, x, yb)
        s = _soil_for_layer(soils, layer)
        area = max(0.0, h[i]) * dx[i]
        W[i] = s.gamma * area
    return W

def _c_phi_by_layer(ground: GroundPL, interfaces, soils, xm, ya):
    c = np.zeros_like(xm); tanphi = np.zeros_like(xm)
    for i, x in enumerate(xm):
        layer = _layer_index_at_depth(ground, interfaces, x, ya[i])
        s = _soil_for_layer(soils, layer)
        c[i] = s.c
        tanphi[i] = np.tan(s.phi_rad)
    return c, tanphi

# =========================
# Fs computation
# =========================
def _fellenius_fs_with_pair(ground, interfaces, soils, xc, yc, R, x1, x2, n_slices):
    x_edges = np.linspace(x1, x2, int(max(10, n_slices))+1)
    xm, ya, dx, alpha, b, h = _slice_geometry(ground, xc, yc, R, x_edges)
    W = _weights_by_layer(ground, interfaces, soils, xm, ya, dx, h)
    c, tanphi = _c_phi_by_layer(ground, interfaces, soils, xm, ya)
    T = np.sum(W * np.sin(alpha))
    S = np.sum(c * b + (W * np.cos(alpha)) * tanphi)
    if T <= 1e-9: return None
    FS = S / T
    if not np.isfinite(FS) or FS <= 0: return None
    return float(FS)

def _bishop_fs_with_pair(ground, interfaces, soils, xc, yc, R, x1, x2, n_slices):
    x_edges = np.linspace(x1, x2, int(max(10, n_slices))+1)
    xm, ya, dx, alpha, b, h = _slice_geometry(ground, xc, yc, R, x_edges)
    W = _weights_by_layer(ground, interfaces, soils, xm, ya, dx, h)
    c, tanphi = _c_phi_by_layer(ground, interfaces, soils, xm, ya)

    T = np.sum(W * np.sin(alpha))
    if T <= 1e-12: return None
    FS = 1.0
    for _ in range(50):
        m = 1.0 / (1.0 + (tanphi * np.tan(alpha)) / max(FS, 1e-6))
        num = np.sum((c * b + W * tanphi) * m)
        den = T
        FS_new = num / den
        if not np.isfinite(FS_new) or FS_new <= 0: return None
        if abs(FS_new - FS) < 1e-4:
            FS = FS_new; break
        FS = FS_new
    return float(FS)

def fs_given_R_multi(ground: GroundPL, interfaces, soils, allow_cross, method: str,
                     xc: float, yc: float, R: float, n_slices: int=30):
    pair = arc_sample_poly_best_pair(ground, xc, yc, R, n=241)
    if pair is None: return None
    x1, x2 = pair
    y_bottom = yc - R
    deepest_layer = _layer_index_at_depth(ground, interfaces, xc, y_bottom + 1e-6)
    allowed_max = _allowed_max_layer(allow_cross, len(soils))
    if deepest_layer > allowed_max:
        return None
    if method.lower().startswith("bishop"):
        return _bishop_fs_with_pair(ground, interfaces, soils, xc, yc, R, x1, x2, n_slices)
    else:
        return _fellenius_fs_with_pair(ground, interfaces, soils, xc, yc, R, x1, x2, n_slices)

def driving_sum_for_R_multi(ground: GroundPL, interfaces, soils, allow_cross,
                            xc: float, yc: float, R: float, n_slices: int=30):
    pair = arc_sample_poly_best_pair(ground, xc, yc, R, n=241)
    if pair is None: return None
    x1, x2 = pair
    y_bottom = yc - R
    deepest_layer = _layer_index_at_depth(ground, interfaces, xc, y_bottom + 1e-6)
    allowed_max = _allowed_max_layer(allow_cross, len(soils))
    if deepest_layer > allowed_max:
        return None

    x_edges = np.linspace(x1, x2, int(max(10, n_slices))+1)
    xm, ya, dx, alpha, b, h = _slice_geometry(ground, xc, yc, R, x_edges)
    W = _weights_by_layer(ground, interfaces, soils, xm, ya, dx, h)
    D = np.sum(W * np.sin(alpha))
    N = np.sum(W * np.cos(alpha))
    B = np.sum(b)
    if D <= 1e-12: return None
    return float(D), float(N), float(B)

# =========================
# Candidate generator (Quick)
# =========================
def arcs_from_center_by_entries_multi(ground: GroundPL, soils, xc: float, yc: float,
                                      n_entries: int, method: str,
                                      depth_min: float, depth_max: float,
                                      interfaces, allow_cross,
                                      quick_mode: bool=True, n_slices_quick: int=12,
                                      limit_arcs_per_center: int=120, probe_n_min: int=81):
    """
    Yield (x1, x2, R, Fs_quick) for a fixed center by sweeping R derived from depth band.
    depth d = ground(xc) - (yc - R) = R - (yc - yg)  ⇒  R = d + (yc - yg)
    """
    yg = float(ground.y_at(xc))
    R_min = float(depth_min + (yc - yg))
    R_max = float(depth_max + (yc - yg))

    if not np.isfinite(R_min) or not np.isfinite(R_max):
        return
    R_min = max(0.05, R_min)
    R_max = max(R_min + 1e-6, R_max)
    if R_max <= R_min + 1e-6:
        return

    nR = int(max(20, min(n_entries, 3000)))
    Rs = np.linspace(R_min, R_max, nR)

    count = 0
    for R in Rs:
        pair = arc_sample_poly_best_pair(ground, xc, yc, float(R), n=201)
        if pair is None:
            continue
        x1, x2 = pair

        # crossing constraint at deepest point
        yb = yc - R
        deepest_layer = _layer_index_at_depth(ground, interfaces, xc, yb + 1e-6)
        allowed_max = _allowed_max_layer(allow_cross, len(soils))
        if deepest_layer > allowed_max:
            continue

        nsl = int(max(8, n_slices_quick if quick_mode else max(12, n_slices_quick)))
        if method.lower().startswith("bishop"):
            Fs = _bishop_fs_with_pair(ground, interfaces, soils, xc, yc, float(R), x1, x2, nsl)
        else:
            Fs = _fellenius_fs_with_pair(ground, interfaces, soils, xc, yc, float(R), x1, x2, nsl)

        if Fs is None:
            continue
        yield (float(x1), float(x2), float(R), float(Fs))
        count += 1
        if count >= int(limit_arcs_per_center):
            break
    return