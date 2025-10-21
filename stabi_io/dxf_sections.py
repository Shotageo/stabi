# stabi_io/dxf_sections.py
from __future__ import annotations
import re, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import ezdxf
except Exception:
    ezdxf = None

# -------------------------------------------------------------
# Patterns
_ST_RE = re.compile(r"(?:STA|KP|No\.?|NO\.?|No|NO)?\s*([0-9]+)\s*[+−-]\s*([0-9]+)", re.IGNORECASE)

def parse_station(text: str) -> Optional[float]:
    """'100+00' / 'No.0+20' / 'KP12+350' -> meters (A*100 + B)"""
    if not text:
        return None
    m = _ST_RE.search(str(text).replace("−", "-"))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return float(a*100 + b)

def normalize_no_key(text: str) -> Optional[str]:
    """Return normalized No key like 'No.0+20' if found; else None."""
    m = _ST_RE.search(str(text).replace("−", "-"))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return f"No.{a}+{b:02d}"

# -------------------------------------------------------------
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
    """Rough layer stats (counts + total length-like)."""
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

# -------------------------------------------------------------
def _sample_entity_2d(ent):
    """Return Nx2 array of (x,y) for common curve entities."""
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
    """Pick the longest curve in selected layers as centerline (2D XY)."""
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
            length = _safe_len(e)
            candidates.append((length, e))
    if not candidates:
        raise ValueError("No centerline candidate found in selected layers.")
    _, ent = max(candidates, key=lambda t: t[0])
    xy = _sample_entity_2d(ent) * unit_scale
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] <= 0:
        raise ValueError("Centerline length is zero.")
    return xy

# -------------------------------------------------------------
def project_point_to_polyline(poly: np.ndarray, pt: np.ndarray) -> Tuple[float, float]:
    """
    Project point pt onto polyline; return (s_along_polyline_m, distance_m).
    """
    if poly.shape[0] < 2:
        return 0.0, float("inf")
    segs = np.diff(poly, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    best = (0.0, float("inf"))  # (s, dist)
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

# -------------------------------------------------------------
def extract_no_labels(dxf_path: str, label_layers: List[str], unit_scale: float = 1.0) -> List[Dict]:
    """
    Returns: [{'key': 'No.0+80', 'pos': (x,y), 'raw': 'NO. 0+80', 'layer': '...'}, ...]
    """
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
    """
    Returns: [{'center': (x,y), 'r': radius, 'layer': '...'}, ...]
    """
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    allow = set(circle_layers or [])
    circles: List[Dict] = []

    # direct circles
    for c in msp.query("CIRCLE"):
        if allow and c.dxf.layer not in allow:
            continue
        try:
            cx, cy = float(c.dxf.center.x)*unit_scale, float(c.dxf.center.y)*unit_scale
            r = float(c.dxf.radius)*unit_scale
            circles.append({"center": (cx, cy), "r": r, "layer": c.dxf.layer})
        except Exception:
            pass

    # circles inside blocks (best-effort)
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

# -------------------------------------------------------------
def read_single_section_file(path: str, layer_name: Optional[str] = None, unit_scale: float = 1.0) -> Optional[np.ndarray]:
    """Read a single DXF/CSV as (offset, elev) array. Assumes X=offset, Y=elev for DXF."""
    if path.lower().endswith(".csv"):
        try:
            data = np.loadtxt(path, delimiter=",", dtype=float)
            oz = np.asarray(data, dtype=float) * unit_scale
            if oz.ndim != 2 or oz.shape[1] < 2:
                return None
            return oz[:, :2]
        except Exception:
            return None
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    try:
        doc = ezdxf.readfile(path)
    except Exception:
        return None
    msp = doc.modelspace()
    polys = []
    for e in msp.query("LWPOLYLINE"):
        if (layer_name is None) or (e.dxf.layer == layer_name):
            pts = _sample_entity_2d(e)
            polys.append((e, pts))
    if not polys:
        return None
    ent, pts = max(polys, key=lambda t: _safe_len(t[0]))
    oz = pts * unit_scale
    idx = np.argsort(oz[:,0]); oz = oz[idx]
    _, uniq = np.unique(oz[:,0], return_index=True)
    oz = oz[np.sort(uniq)]
    return oz[:, :2]
