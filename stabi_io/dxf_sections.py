from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

try:
    import ezdxf
except Exception:
    ezdxf = None


@dataclass
class Alignment:
    points: List[Tuple[float, float]]
    length: float
    seglen: List[float]
    cumlen: List[float]


@dataclass
class SectionLine:
    id: str
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    station: Optional[float] = None
    foot: Optional[Tuple[float, float]] = None
    nvec: Optional[Tuple[float, float]] = None  # 法線（XY 2D）


def _polyline_points(e) -> List[Tuple[float, float]]:
    pts = []
    try:
        if e.dxftype() == "LWPOLYLINE":
            for p in e.get_points("xy"):
                pts.append((float(p[0]), float(p[1])))
        elif e.dxftype() == "POLYLINE":
            for v in e.vertices:
                pts.append((float(v.dxf.location.x), float(v.dxf.location.y)))
    except Exception:
        pass
    out = []
    for x, y in pts:
        if out and abs(out[-1][0] - x) < 1e-12 and abs(out[-1][1] - y) < 1e-12:
            continue
        out.append((x, y))
    return out


def _build_alignment(points: List[Tuple[float, float]]) -> Alignment:
    seglen = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seglen.append(math.hypot(x1 - x0, y1 - y0))
    cum = [0.0]
    for s in seglen:
        cum.append(cum[-1] + s)
    return Alignment(points=points, length=cum[-1], seglen=seglen, cumlen=cum)


def _project_point_to_polyline(x: float, y: float, ali: Alignment) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    # 戻り値： (station, 最近点, その区間の接線ベクトルt)
    best_d2 = float("inf")
    best_s = 0.0
    best_xy = ali.points[0]
    best_t = (1.0, 0.0)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(ali.points[:-1], ali.points[1:])):
        vx, vy = x1 - x0, y1 - y0
        L2 = vx * vx + vy * vy
        if L2 <= 1e-18:
            d2 = (x - x0) ** 2 + (y - y0) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = ali.cumlen[i]
                best_xy = (x0, y0)
                best_t = (1.0, 0.0)
            continue
        t = ((x - x0) * vx + (y - y0) * vy) / L2
        t = max(0.0, min(1.0, t))
        qx, qy = x0 + t * vx, y0 + t * vy
        d2 = (x - qx) ** 2 + (y - qy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = ali.cumlen[i] + math.hypot(qx - x0, qy - y0)
            best_xy = (qx, qy)
            L = math.hypot(vx, vy)
            best_t = (vx / L, vy / L)
    return best_s, best_xy, best_t


def load_alignment(dxf_path: str, layer_hint: Optional[str] = "ALIGN") -> Alignment:
    if ezdxf is None:
        raise RuntimeError("ezdxf が見つかりません。`pip install ezdxf` を実行してください。")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    cand = []
    if layer_hint:
        for e in msp.query("LWPOLYLINE"):
            if str(e.dxf.layer).upper().startswith(str(layer_hint).upper()):
                pts = _polyline_points(e)
                if len(pts) >= 2:
                    cand.append(pts)
    if not cand:
        for e in msp.query("LWPOLYLINE"):
            pts = _polyline_points(e)
            if len(pts) >= 2:
                cand.append(pts)
    if not cand:
        raise ValueError("中心線形となる LWPOLYLINE が見つかりません。")
    def _length(ps):
        return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(ps[:-1], ps[1:]))
    pts_best = max(cand, key=_length)
    return _build_alignment(pts_best)


def load_sections(dxf_path: str, layer_filter: str = "XS") -> List[SectionLine]:
    if ezdxf is None:
        raise RuntimeError("ezdxf が見つかりません。`pip install ezdxf` を実行してください。")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    xs: List[SectionLine] = []
    for e in msp.query("LINE"):
        if not str(e.dxf.layer).upper().startswith(layer_filter.upper()):
            continue
        x0, y0 = float(e.dxf.start.x), float(e.dxf.start.y)
        x1, y1 = float(e.dxf.end.x), float(e.dxf.end.y)
        xs.append(SectionLine(id=f"LINE:{e.dxf.handle}", p0=(x0, y0), p1=(x1, y1)))
    for e in msp.query("LWPOLYLINE"):
        if not str(e.dxf.layer).upper().startswith(layer_filter.upper()):
            continue
        pts = _polyline_points(e)
        if len(pts) >= 2:
            p0, p1 = pts[0], pts[-1]
            xs.append(SectionLine(id=f"LWP:{e.dxf.handle}", p0=p0, p1=p1))
    return xs


def attach_stationing(xs_lines: List[SectionLine], ali: Alignment) -> List[SectionLine]:
    out: List[SectionLine] = []
    for s in xs_lines:
        s0, q0, t0 = _project_point_to_polyline(s.p0[0], s.p0[1], ali)
        s1, q1, t1 = _project_point_to_polyline(s.p1[0], s.p1[1], ali)
        d0 = math.hypot(s.p0[0] - q0[0], s.p0[1] - q0[1])
        d1 = math.hypot(s.p1[0] - q1[0], s.p1[1] - q1[1])
        if d0 <= d1:
            s.station, s.foot, t = s0, q0, t0
        else:
            s.station, s.foot, t = s1, q1, t1
        # 左手系で法線を作る（道路に直交）
        nx, ny = -t[1], t[0]
        s.nvec = (nx, ny)
        out.append(s)
    out.sort(key=lambda t: (float("inf") if t.station is None else t.station))
    return out

