# stabi_io/dxf_sections.py
from __future__ import annotations
import re, math, os, io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import ezdxf
except Exception as e:
    ezdxf = None

# ---------- Helpers ----------
_ST_RE = re.compile(r"(?:STA|KP)?\s*(\d+)\s*[+−-]\s*(\d+)", re.IGNORECASE)

def parse_station(text: str) -> Optional[float]:
    """
    '100+00' -> 100.00m, 'KP12+350' -> 12350m ではなく 12+350= 12350? 日本式KPは km+ m なので 12km+350m=12350m。
    ここでは 'A+B' を A*100 + B [m] として扱う（道路平面図慣習）。
    """
    m = _ST_RE.search(text.replace("−", "-"))
    if not m: 
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return float(a*100 + b)

@dataclass
class LayerInfo:
    name: str
    entity_counts: Dict[str, int]
    length_sum: float

def _safe_len(entity) -> float:
    try:
        return float(entity.length())  # ezdxfのpolyline等はlength()を持つ
    except Exception:
        return 0.0

# ---------- Public API ----------
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

def read_centerline(dxf_path: str, layer_whitelist: List[str], unit_scale: float = 1.0) -> np.ndarray:
    """
    Returns Nx2 XY polyline (m). Picks the longest LWPOLYLINE/LINE/SPLINE among given layers.
    """
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed.")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    candidates = []
    for e in msp.query("*"):
        if e.dxf.layer not in set(layer_whitelist):
            continue
        if e.dxftype() in ("LWPOLYLINE", "POLYLINE", "LINE", "SPLINE"):
            length = _safe_len(e)
            candidates.append((length, e))
    if not candidates:
        raise ValueError("No centerline candidate found in selected layers.")
    _, ent = max(candidates, key=lambda t: t[0])

    def sample_entity(ent) -> np.ndarray:
        if ent.dxftype() == "LINE":
            pts = np.array([[ent.dxf.start.x, ent.dxf.start.y],
                            [ent.dxf.end.x,   ent.dxf.end.y]], dtype=float)
        else:
            # たいていのエンティティはflatteningで点列化できる
            try:
                pts = np.array([[p[0], p[1]] for p in ent.flattening(1.0)], dtype=float)
            except Exception:
                pts = np.array([[v[0], v[1]] for v in ent.get_points()], dtype=float)
        return pts

    xy = sample_entity(ent) * unit_scale
    # 方向を「累積距離が増加する向き」に正規化（左→右等は関与しない）
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] <= 0:
        raise ValueError("Centerline length is zero.")
    # 開始点をユーザに委ねず、このまま採用。必要ならreverse。
    return xy

def read_cross_sections_from_folder(folder: str,
                                    layer_name: Optional[str] = None,
                                    unit_scale: float = 1.0) -> Dict[float, np.ndarray]:
    """
    読み取り結果: { station_m : array([[offset, elev], ...]) }
    支援フォーマット:
      - DXF: 横断ごとに1ファイル。対象LWPOLYLINEを layer_name で指定（なければ最長の1本）
      - CSV: 'offset_m,elev_m' の2列。ファイル名から station を抽出。
    """
    result: Dict[float, np.ndarray] = {}
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        st = parse_station(fname)  # ファイル名から距離程推定
        if path.lower().endswith(".csv"):
            data = np.loadtxt(path, delimiter=",", dtype=float)
            oz = np.asarray(data, dtype=float)
            if oz.ndim != 2 or oz.shape[1] < 2:
                continue
            result[float(st if st is not None else len(result))] = np.column_stack([oz[:,0], oz[:,1]])
            continue

        if path.lower().endswith(".dxf"):
            if ezdxf is None:
                raise RuntimeError("ezdxf not installed.")
            try:
                doc = ezdxf.readfile(path)
            except Exception:
                continue
            if st is None:
                # DXF内のTEXT/MTEXTから抽出を試みる
                for e in doc.modelspace().query("TEXT MTEXT"):
                    st = parse_station(e.dxf.text if e.dxftype()=="TEXT" else e.plain_text())
                    if st is not None:
                        break
            msp = doc.modelspace()
            polys = []
            for e in msp.query("LWPOLYLINE"):
                if (layer_name is None) or (e.dxf.layer == layer_name):
                    pts = np.array([[p[0], p[1]] for p in e.get_points()], dtype=float)
                    polys.append((e, pts))
            if not polys:
                continue
            # 候補のうち「最長」を採用
            ent, pts = max(polys, key=lambda t: _safe_len(t[0]))
            oz = pts * unit_scale
            # X=offset, Y=elev である前提／逆なら軸入れ替えスイッチをあとで追加
            # 昇順化＆重複除去
            idx = np.argsort(oz[:,0])
            oz = oz[idx]
            _, uniq = np.unique(oz[:,0], return_index=True)
            oz = oz[np.sort(uniq)]
            result[float(st if st is not None else len(result))] = oz[:, :2]
    if not result:
        raise ValueError("No cross-sections found in folder.")
    return dict(sorted(result.items(), key=lambda kv: kv[0]))
