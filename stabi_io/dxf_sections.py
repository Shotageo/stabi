# stabi_io/dxf_sections.py
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

def parse_station(text: str) -> Optional[float]:
    if not text:
        return None
    m = _ST_RE.search(str(text).replace("−", "-"))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return float(a*100 + b)

def normalize_no_key(text: str) -> Optional[str]:
    m = _ST_RE.search(str(text).replace("−", "-"))
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
    candidates = []
    allow = set(layer_whitelist or [])
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

    # best-effort for circles inside blocks
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

# ----------------- read a cross-section (robust) -------------
def _pca_2d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, eigvecs(2x2), centered points."""
    C = np.asarray(points, dtype=float)
    mu = np.mean(C, axis=0)
    X = C - mu
    cov = np.cov(X.T)
    w, V = np.linalg.eigh(cov)          # ascending
    V = V[:, ::-1]                      # first column = major axis
    return mu, V, X

def _median_chain(u: np.ndarray, v: np.ndarray, nbin: int = 200) -> np.ndarray:
    """Make a single polyline by binning along u and taking median v."""
    umin, umax = float(np.min(u)), float(np.max(u))
    if umax - umin <= 0:
        return np.empty((0,2))
    bins = np.linspace(umin, umax, nbin+1)
    idx = np.digitize(u, bins) - 1
    xs, ys = [], []
    for b in range(nbin):
        mask = (idx == b)
        if np.count_nonzero(mask) >= 3:
            xs.append(0.5*(bins[b]+bins[b+1]))
            ys.append(float(np.median(v[mask])))
    if not xs:
        return np.empty((0,2))
    P = np.column_stack([np.asarray(xs), np.asarray(ys)])
    # remove duplicates / sort
    order = np.argsort(P[:,0])
    P = P[order]
    _, uniq = np.unique(P[:,0], return_index=True)
    return P[np.sort(uniq)]

def read_single_section_file(path: str, layer_name: Optional[str] = None, unit_scale: float = 1.0) -> Optional[np.ndarray]:
    """
    Read one DXF/CSV and return Nx2 array (X, Y) in the *file's coordinates*.
    - If LWPOLYLINE exists → use the longest one.
    - Else if LINE only → collect endpoints, PCA → bin-median to a single chain.
    NOTE: 軸の意味（offset/elev）やローカル化は呼び出し側で行います。
    """
    # CSV
    if path.lower().endswith(".csv"):
        try:
            data = np.loadtxt(path, delimiter=",", dtype=float)
            pts = np.asarray(data, dtype=float)
            if pts.ndim != 2 or pts.shape[1] < 2:
                return None
            return pts[:, :2] * unit_scale
        except Exception:
            return None

    # DXF
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    try:
        doc = ezdxf.readfile(path)
    except Exception:
        return None
    msp = doc.modelspace()

    # 1) LWPOLYLINE 優先
    polys = []
    for e in msp.query("LWPOLYLINE"):
        if (layer_name is None) or (e.dxf.layer == layer_name):
            pts = _sample_entity_2d(e)
            if pts.size:
                polys.append((e, pts))
    if polys:
        ent, pts = max(polys, key=lambda t: _safe_len(t[0]))
        P = pts * unit_scale
        # sort by X to stabilize
        order = np.argsort(P[:,0])
        P = P[order]
        _, uniq = np.unique(P[:,0], return_index=True)
        return P[np.sort(uniq)][:, :2]

    # 2) LINE 群から推定
    pts_list = []
    for ln in msp.query("LINE"):
        if (layer_name is None) or (ln.dxf.layer == layer_name):
            pts_list.append([ln.dxf.start.x, ln.dxf.start.y])
            pts_list.append([ln.dxf.end.x,   ln.dxf.end.y])
    if len(pts_list) < 6:
        return None

    P = np.asarray(pts_list, dtype=float) * unit_scale
    mu, V, Xc = _pca_2d(P)             # mean, eigenvectors, centered
    U = Xc @ V                         # rotate to PCA frame (u: major, v: minor)

    chain = _median_chain(U[:,0], U[:,1], nbin=240)
    if chain.size == 0:
        return None

    # return in PCA frame (u,v)  ← 呼び出し側で offset/elev として扱う想定
    return chain
