# io/dxf_sections.py
# DXFから「中心線形（ALIGN）」と「横断法線（XS*）」を読み取る最小ローダ
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

try:
    import ezdxf
except Exception as e:
    ezdxf = None

@dataclass
class Alignment:
    points: List[Tuple[float, float]]           # [(x,y), ...]
    length: float                               # 線形の延長
    seglen: List[float]                         # 各区間長
    cumlen: List[float]                         # 累積距離（先頭=0）

@dataclass
class SectionLine:
    id: str
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    # 中心線への射影情報（読込後に付与）
    station: Optional[float] = None
    foot: Optional[Tuple[float,float]] = None   # 最近点（中心線上）

def _polyline_points(e) -> List[Tuple[float, float]]:
    # LWPOLYLINE / POLYLINE に対応（bulgeは無視して折線化）
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
    # 重複やNaNは最低限除去
    out = []
    for x, y in pts:
        if out and abs(out[-1][0]-x) < 1e-12 and abs(out[-1][1]-y) < 1e-12:
            continue
        out.append((x, y))
    return out

def _build_alignment(points: List[Tuple[float, float]]) -> Alignment:
    seglen = []
    for (x0,y0),(x1,y1) in zip(points[:-1], points[1:]):
        seglen.append(math.hypot(x1-x0, y1-y0))
    cum = [0.0]
    for s in seglen:
        cum.append(cum[-1] + s)
    return Alignment(points=points, length=cum[-1], seglen=seglen, cumlen=cum)

def _project_point_to_polyline(x: float, y: float, ali: Alignment) -> Tuple[float, Tuple[float,float]]:
    # 折線上の最近点と、そのStation（累積距離）を返す
    best_d2 = float("inf")
    best_s = 0.0
    best_xy = (ali.points[0][0], ali.points[0][1])
    for i, ((x0,y0),(x1,y1)) in enumerate(zip(ali.points[:-1], ali.points[1:])):
        vx, vy = x1-x0, y1-y0
        L2 = vx*vx + vy*vy
        if L2 <= 1e-18:
            # 退化区間は端点判定
            d2 = (x-x0)**2 + (y-y0)**2
            if d2 < best_d2:
                best_d2 = d2
                best_s = ali.cumlen[i]
                best_xy = (x0,y0)
            continue
        t = ((x-x0)*vx + (y-y0)*vy) / L2
        t = max(0.0, min(1.0, t))
        qx, qy = x0 + t*vx, y0 + t*vy
        d2 = (x-qx)**2 + (y-qy)**2
        if d2 < best_d2:
            best_d2 = d2
            best_s = ali.cumlen[i] + math.sqrt((qx-x0)**2 + (qy-y0)**2)
            best_xy = (qx, qy)
    return best_s, best_xy

def load_alignment(dxf_path: str, layer_hint: Optional[str] = "ALIGN") -> Alignment:
    if ezdxf is None:
        raise RuntimeError("ezdxf が見つかりません。`pip install ezdxf` を実行してください。")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # レイヤヒントがある場合は優先
    cand: List[List[Tuple[float,float]]] = []
    if layer_hint:
        for e in msp.query("LWPOLYLINE"):
            if str(e.dxf.layer).upper().startswith(str(layer_hint).upper()):
                pts = _polyline_points(e)
                if len(pts) >= 2:
                    cand.append(pts)
    # fallback: 最長のLWPOLYLINEを中心線形とみなす
    if not cand:
        for e in msp.query("LWPOLYLINE"):
            pts = _polyline_points(e)
            if len(pts) >= 2:
                cand.append(pts)
    if not cand:
        raise ValueError("中心線形となる LWPOLYLINE が見つかりません。レイヤ名を確認してください。")
    # 最も延長の長いものを採用
    def _length(ps):
        return sum(math.hypot(x1-x0, y1-y0) for (x0,y0),(x1,y1) in zip(ps[:-1], ps[1:]))
    pts_best = max(cand, key=_length)
    return _build_alignment(pts_best)

def load_sections(dxf_path: str, layer_filter: str = "XS") -> List[SectionLine]:
    if ezdxf is None:
        raise RuntimeError("ezdxf が見つかりません。`pip install ezdxf` を実行してください。")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    xs: List[SectionLine] = []

    # 直線（LINE）
    for e in msp.query("LINE"):
        if not str(e.dxf.layer).upper().startswith(layer_filter.upper()):
            continue
        x0, y0 = float(e.dxf.start.x), float(e.dxf.start.y)
        x1, y1 = float(e.dxf.end.x), float(e.dxf.end.y)
        xs.append(SectionLine(id=f"LINE:{e.dxf.handle}", p0=(x0,y0), p1=(x1,y1)))

    # 折線（LWPOLYLINE）も許容：最短端点対で代表直線を作る
    for e in msp.query("LWPOLYLINE"):
        if not str(e.dxf.layer).upper().startswith(layer_filter.upper()):
            continue
        pts = _polyline_points(e)
        if len(pts) >= 2:
            p0, p1 = pts[0], pts[-1]
            xs.append(SectionLine(id=f"LWP:{e.dxf.handle}", p0=p0, p1=p1))
    return xs

def attach_stationing(xs_lines: List[SectionLine], ali: Alignment) -> List[SectionLine]:
    # 各横断線の「中心線に最も近い点」をp0/p1から選び、station/footを付与
    out: List[SectionLine] = []
    for s in xs_lines:
        s0, q0 = _project_point_to_polyline(s.p0[0], s.p0[1], ali)
        s1, q1 = _project_point_to_polyline(s.p1[0], s.p1[1], ali)
        # 近い方を採用
        d0 = math.hypot(s.p0[0]-q0[0], s.p0[1]-q0[1])
        d1 = math.hypot(s.p1[0]-q1[0], s.p1[1]-q1[1])
        if d0 <= d1:
            s.station, s.foot = s0, q0
        else:
            s.station, s.foot = s1, q1
        out.append(s)
    # stationで昇順にして返す
    out.sort(key=lambda t: (float("inf") if t.station is None else t.station))
    return out

