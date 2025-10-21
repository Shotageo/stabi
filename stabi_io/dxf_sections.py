from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import ezdxf
except Exception:
    ezdxf = None

# ----------------------- station text -----------------------
_ST_RE = re.compile(r"(?:STA|KP|No\.?|NO\.?|No|NO)?\s*([0-9]+)\s*[+−-]\s*([0-9]+)", re.IGNORECASE)

def normalize_no_key(text: str) -> Optional[str]:
    t = str(text).replace("−", "-")
    m = _ST_RE.search(t)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return f"No.{a}+{b:02d}"

# ----------------------- layer listing ----------------------
@dataclass
class LayerInfo:
    name: str
    entity_counts: Dict[str, int]
    length_sum: float

def _safe_len(entity) -> float:
    try:
        return float(entity.length())
    except Exception:
        return 0.0

def list_layers(dxf_path: str) -> List[LayerInfo]:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed. Add 'ezdxf' to requirements.txt.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    counter: Dict[str, LayerInfo] = {}
    for e in msp:
        name = e.dxf.layer
        info = counter.get(name) or LayerInfo(name, {}, 0.0)
        et = e.dxftype()
        info.entity_counts[et] = info.entity_counts.get(et, 0) + 1
        info.length_sum += _safe_len(e)
        counter[name] = info
    return sorted(counter.values(), key=lambda x: (-x.length_sum, x.name.lower()))

# ----------------------- geometry utils ---------------------
def _sample_entity_2d(ent) -> np.ndarray:
    import numpy as _np
    if ent.dxftype() == "LINE":
        pts = _np.array([[ent.dxf.start.x, ent.dxf.start.y],
                         [ent.dxf.end.x,   ent.dxf.end.y]], dtype=float)
    else:
        try:
            pts = _np.array([[p[0], p[1]] for p in ent.flattening(1.0)], dtype=float)
        except Exception:
            try:
                pts = _np.array([[v[0], v[1]] for v in ent.get_points()], dtype=float)
            except Exception:
                pts = _np.empty((0,2), dtype=float)
    return pts

def read_centerline(dxf_path: str, layer_whitelist: List[str], unit_scale: float = 1.0) -> np.ndarray:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    allow = set(layer_whitelist or [])
    candidates = []
    for e in msp.query("*"):
        if allow and e.dxf.layer not in allow:
            continue
        if e.dxftype() in ("LWPOLYLINE", "POLYLINE", "LINE", "SPLINE"):
            candidates.append((_safe_len(e), e))
    if not candidates:
        raise ValueError("No centerline candidate found in selected layers.")
    _, ent = max(candidates, key=lambda t: t[0])
    xy = _sample_entity_2d(ent) * unit_scale
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] <= 0:
        raise ValueError("Centerline length is zero.")
    return xy

def project_point_to_polyline(poly: np.ndarray, pt: np.ndarray) -> Tuple[float, float]:
    if poly.shape[0] < 2:
        return 0.0, float("inf")
    segs = np.diff(poly, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    best = (0.0, float("inf"))
    P0 = poly[:-1]
    for i, (p, v, L) in enumerate(zip(P0, segs, lens)):
        if L == 0:
            continue
        t = np.dot(pt - p, v) / (L*L)
        t = max(0.0, min(1.0, t))
        proj = p + t * v
        dist = float(np.linalg.norm(pt - proj))
        s = float(cum[i] + t*L)
        if dist < best[1]:
            best = (s, dist)
    return best

# --------------------- labels / circles ---------------------
def extract_no_labels(dxf_path: str, label_layers: List[str], unit_scale: float = 1.0) -> List[Dict]:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    allow = set(label_layers or [])
    out: List[Dict] = []
    for e in msp.query("TEXT MTEXT"):
        if allow and e.dxf.layer not in allow:
            continue
        try:
            raw = e.dxf.text if e.dxftype() == "TEXT" else e.plain_text()
            key = normalize_no_key(raw)
            if not key:
                continue
            ins = e.dxf.insert
            pos = (float(ins.x)*unit_scale, float(ins.y)*unit_scale)
            out.append({"key": key, "pos": pos, "raw": raw, "layer": e.dxf.layer})
        except Exception:
            continue
    return out

def extract_circles(dxf_path: str, circle_layers: List[str], unit_scale: float = 1.0) -> List[Dict]:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    allow = set(circle_layers or [])
    circles: List[Dict] = []
    for c in msp.query("CIRCLE"):
        if allow and c.dxf.layer not in allow:
            continue
        try:
            cx, cy = float(c.dxf.center.x)*unit_scale, float(c.dxf.center.y)*unit_scale
            r = float(c.dxf.radius)*unit_scale
            circles.append({"center": (cx, cy), "r": r, "layer": c.dxf.layer})
        except Exception:
            pass
    try:
        for ref in msp.query("INSERT"):
            try:
                for v in ref.virtual_entities():
                    if v.dxftype() == "CIRCLE":
                        if allow and v.dxf.layer not in allow:
                            continue
                        cx, cy = float(v.dxf.center.x)*unit_scale, float(v.dxf.center.y)*unit_scale
                        r = float(v.dxf.radius)*unit_scale
                        circles.append({"center": (cx, cy), "r": r, "layer": v.dxf.layer})
            except Exception:
                continue
    except Exception:
        pass
    return circles

# --- ground-likeness scoring helpers ---
def _is_closed_poly(pts: np.ndarray, tol=1e-6) -> bool:
    if pts.shape[0] < 3: 
        return False
    return float(np.linalg.norm(pts[0] - pts[-1])) < tol

def _bbox(pts: np.ndarray):
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return xmin, xmax, ymin, ymax

def _poly_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def _rect_like_penalty(pts: np.ndarray) -> float:
    if pts.shape[0] < 4:
        return 0.0
    v = np.diff(pts, axis=0)
    v = v[np.linalg.norm(v, axis=1) > 1e-9]
    if len(v) < 2:
        return 0.0
    u = v[:-1] / np.linalg.norm(v[:-1], axis=1)[:, None]
    w = v[1:]  / np.linalg.norm(v[1:],  axis=1)[:, None]
    cosang = np.clip(np.sum(u*w, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(np.mean((np.abs(ang-90.0) < 10.0).astype(float)))

def _groundlike_score(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    L = _poly_length(pts)
    xmin, xmax, ymin, ymax = _bbox(pts)
    w = max(1e-9, xmax - xmin)
    h = max(1e-9, ymax - ymin)
    aspect = h / w
    nverts = pts.shape[0]
    closed = _is_closed_poly(pts, tol=1e-6)
    rectp = _rect_like_penalty(pts)

    s_len   = 1.0 - np.exp(-L/20.0)
    s_aspr  = np.clip(1.0 - aspect*2.0, 0.0, 1.0)
    s_nv    = np.clip((nverts-4)/20.0, 0.0, 1.0)
    s_open  = 0.0 if closed else 1.0
    s_rect  = 1.0 - np.clip(rectp, 0.0, 1.0)
    s = 0.35*s_aspr + 0.25*s_len + 0.15*s_nv + 0.15*s_open + 0.10*s_rect
    return float(np.clip(s, 0.0, 1.0))

# ----------------- cross-section (robust) -------------------
def read_single_section_file(
    path: str,
    layer_name: Optional[str] = None,
    unit_scale: float = 1.0,
    aggregate: str = "median",   # "median" / "lower" / "upper"
    smooth_k: int = 7,
    max_slope: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Strategy:
      1) Score LWPOLYLINE one-by-one (ground-likeness). If top score>=0.45, return that chain.
      2) Else fallback: collect all (LWPOLYLINE + LINE) points → PCA → segment by gaps → bin-aggregate → smooth.
    Returns Nx2 array in a local (u,v) frame.
    """
    if path.lower().endswith(".csv"):
        try:
            data = np.loadtxt(path, delimiter=",", dtype=float)
            pts = np.asarray(data[:, :2], dtype=float) * unit_scale
            if pts.ndim != 2 or pts.shape[1] < 2:
                return None
        except Exception:
            return None
        P = pts
    else:
        if ezdxf is None:
            raise RuntimeError("ezdxf not installed.")
        try:
            doc = ezdxf.readfile(path)
        except Exception:
            return None
        msp = doc.modelspace()

        scored: List[Tuple[float, np.ndarray]] = []
        for e in msp.query("LWPOLYLINE"):
            if (layer_name is None) or (e.dxf.layer == layer_name):
                pts = _sample_entity_2d(e)
                if pts.size:
                    pts = pts.astype(float) * unit_scale
                    scored.append((_groundlike_score(pts), pts))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored and scored[0][0] >= 0.45:
            pts = scored[0][1]
            order = np.argsort(pts[:,0]); Q = pts[order]
            x = Q[:,0]; y = Q[:,1]
            xx = np.round(x, 4)
            _, first = np.unique(xx, return_index=True)
            out_x, out_y = [], []
            for i0, i1 in zip(first, list(first[1:]) + [len(xx)]):
                out_x.append(float(np.median(x[i0:i1])))
                out_y.append(float(np.median(y[i0:i1])))
            return np.column_stack([out_x, out_y])

        pts_list = []
        for e in msp.query("LWPOLYLINE"):
            if (layer_name is None) or (e.dxf.layer == layer_name):
                Q = _sample_entity_2d(e)
                if Q.size: pts_list.append(Q)
        for ln in msp.query("LINE"):
            if (layer_name is None) or (ln.dxf.layer == layer_name):
                pts_list.append(np.array([[ln.dxf.start.x, ln.dxf.start.y],
                                          [ln.dxf.end.x,   ln.dxf.end.y]], dtype=float))
        if not pts_list:
            return None
        P = (np.vstack(pts_list)).astype(float) * unit_scale

    def _pca_local(points: np.ndarray):
        mu = np.mean(points, axis=0)
        X = points - mu
        cov = np.cov(X.T)
        w, V = np.linalg.eigh(cov)
        V = V[:, ::-1]
        return mu, V, X

    def _segments_by_gap(u: np.ndarray, factor: float = 6.0):
        order = np.argsort(u)
        uo = u[order]
        du = np.diff(uo)
        med = np.median(np.abs(du)) or 1.0
        cuts = np.where(du > factor * med)[0]
        idxs = np.r_[0, cuts + 1, len(uo)]
        segs = [slice(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
        return order, segs

    def _bin_aggregate(u: np.ndarray, v: np.ndarray, nbin: int, mode: str) -> np.ndarray:
        umin, umax = float(np.min(u)), float(np.max(u))
        if umax - umin <= 0:
            return np.empty((0,2))
        nbin = int(np.clip(nbin, 120, 480))
        bins = np.linspace(umin, umax, nbin+1)
        idx = np.digitize(u, bins) - 1
        xs, ys = [], []
        for b in range(nbin):
            mask = (idx == b)
            if np.count_nonzero(mask) >= 5:
                xs.append(0.5*(bins[b]+bins[b+1]))
                vb = v[mask]
                if mode == "lower":
                    ys.append(float(np.percentile(vb, 10)))
                elif mode == "upper":
                    ys.append(float(np.percentile(vb, 90)))
                else:
                    ys.append(float(np.median(vb)))
        if not xs:
            return np.empty((0,2))
        P = np.column_stack([np.asarray(xs), np.asarray(ys)])
        order = np.argsort(P[:,0]); P = P[order]
        _, uniq = np.unique(np.round(P[:,0], 5), return_index=True)
        return P[np.sort(uniq)]

    def _median_filter(y: np.ndarray, k: int = 5) -> np.ndarray:
        k = max(1, int(k) | 1);  r = k//2
        yy = y.copy()
        for i in range(len(y)):
            a, b = max(0, i-r), min(len(y), i+r+1)
            yy[i] = np.median(y[a:b])
        return yy

    def _clip_slope(x: np.ndarray, y: np.ndarray, max_slope: float) -> np.ndarray:
        if len(x) < 3 or max_slope <= 0:
            return y
        dy = np.diff(y); dx = np.diff(x)
        s  = np.divide(dy, dx, out=np.zeros_like(dy), where=dx!=0)
        y2 = y.copy()
        for i in range(1, len(y)-1):
            if abs(s[i-1]) > max_slope or abs(s[min(i, len(s)-1)]) > max_slope:
                y2[i] = 0.5*(y2[i-1] + y2[i+1])
        return y2

    mu, V, Xc = _pca_local(P)
    U = Xc @ V
    order, segs = _segments_by_gap(U[:,0], factor=6.0)
    uo, vo = U[order, 0], U[order, 1]
    if not segs:
        return None
    best = max(segs, key=lambda s: s.stop - s.start)
    u_main = uo[best]; v_main = vo[best]

    span = max(1, int((np.max(u_main) - np.min(u_main)) / max((np.percentile(np.diff(u_main), 90) or 1e-3), 1e-3)))
    nbin = int(np.clip(span, 120, 480))
    chain = _bin_aggregate(u_main, v_main, nbin=nbin, mode=aggregate)
    if chain.size == 0:
        return None

    x, y = chain[:,0], chain[:,1]
    y = _median_filter(y, k=smooth_k)
    y = _clip_slope(x, y, max_slope=max_slope)
    return np.column_stack([x, y])
