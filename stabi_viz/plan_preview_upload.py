# stabi_viz/plan_preview_upload.py  — 2025-10-24 full replace

from __future__ import annotations

import hashlib
import tempfile
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
# LEM（フォールバック付き：ライブラリが無くてもプレビューは動く）
try:
    from stabi_core.stabi_lem import compute_min_circle  # 既存実装があれば使用
    _LEM_OK = True
except Exception:
    _LEM_OK = False

    def compute_min_circle(cfg):
        oz = np.asarray(cfg.get("section"))
        if oz is None or oz.size == 0:
            return {"fs": 1.10, "circle": {"oc": 0.0, "zc": 0.0, "R": 10.0}}
        oc = float(np.median(oz[:, 0]))
        zc = float(np.percentile(oz[:, 1], 25))
        R = float(max(6.0, (np.max(oz[:, 0]) - np.min(oz[:, 0])) * 0.35))
        return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R}, "meta": {"fallback": True}}

# ──────────────────────────────────────────────────────────────
# DXF/CSV ユーティリティ（既存モジュール）
from stabi_io.dxf_sections import (
    list_layers,
    read_centerline,
    extract_no_labels,
    extract_circles,
    project_point_to_polyline,
    read_single_section_file,
    normalize_no_key,
    detect_section_centerline_u,
)

# =============================================================
# 追加ヘルパー：DXF 単位診断 / 中心線連結 / 2点法スケール
# =============================================================
INSUNITS_TO_M = {
    0: 1.0,     # Unitless
    1: 0.0254,  # Inch
    2: 0.3048,  # Foot
    3: 1609.344,# Mile
    4: 0.001,   # Millimeter
    5: 0.01,    # Centimeter
    6: 1.0,     # Meter
    7: 1000.0,  # Kilometer
    10: 0.9144, # Yard
    14: 0.1,    # Decimeter
    15: 10.0,   # Decameter
    16: 100.0   # Hectometer
}


def _load_doc_from_path(path: str):
    """ezdxf で DXF を読む。失敗時は None。"""
    try:
        import ezdxf
        return ezdxf.readfile(path)
    except Exception:
        return None


def _read_insunits(doc) -> Tuple[int, float]:
    try:
        code = int(doc.header.get("$INSUNITS", 0))
    except Exception:
        code = 0
    return code, INSUNITS_TO_M.get(code, 1.0)


def _poly_vertices(e) -> Optional[np.ndarray]:
    try:
        if e.dxftype() == "LINE":
            p0 = e.dxf.start
            p1 = e.dxf.end
            return np.array([[p0.x, p0.y], [p1.x, p1.y]], float)
        if e.dxftype() == "LWPOLYLINE":
            pts = np.array([(p[0], p[1]) for p in e.get_points("xy")], float)
            return pts
        if e.dxftype() == "SPLINE":
            pts = np.array([(p.x, p.y) for p in e.approximate(300)], float)
            return pts
    except Exception:
        return None
    return None


def _diagnose_dxf_bbox(path: str, layer_filter: Optional[List[str]] = None) -> Dict:
    """図面の XY 外接箱・セグメント統計と INSUNITS→m の倍率を返す"""
    doc = _load_doc_from_path(path)
    if doc is None:
        return {}
    msp = doc.modelspace()
    code, to_m = _read_insunits(doc)
    try:
        ents = msp.query("LINE LWPOLYLINE SPLINE")
    except Exception:
        ents = [e for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "SPLINE")]

    bb_min = np.array([np.inf, np.inf])
    bb_max = -bb_min
    seg_lens = []
    n = 0
    for e in ents:
        if layer_filter and e.dxf.layer not in layer_filter:
            continue
        arr = _poly_vertices(e)
        if arr is None or len(arr) < 2:
            continue
        bb_min = np.minimum(bb_min, np.min(arr, axis=0))
        bb_max = np.maximum(bb_max, np.max(arr, axis=0))
        d = np.diff(arr, axis=0)
        if len(d):
            seg_lens.extend(np.linalg.norm(d, axis=1))
        n += 1
    if n == 0:
        return {}
    span = (bb_max - bb_min).tolist()
    return {
        "units_code": code,
        "units_to_m": to_m,
        "bbox_min": bb_min.tolist(),
        "bbox_max": bb_max.tolist(),
        "span": span,
        "entities": n,
        "median_seg": float(np.median(seg_lens)) if seg_lens else 0.0,
    }


def _suggest_unit_scale_from_diag(diag: Dict) -> float:
    """INSUNITS と図面スパンから現実的な m スケールを推定（mm→m など）"""
    if not diag:
        return 1.0
    factor = float(diag.get("units_to_m", 1.0))
    span = diag.get("span", [0, 0])
    max_span = max(span) if span else 0.0
    # INSUNITS が未設定（=1.0）かつスパンが数万以上なら mm 運用の可能性が高い
    if factor == 1.0 and max_span > 50000:
        factor = 0.001
    return factor


def _connect_segments(segs: List[np.ndarray], tol: float) -> np.ndarray:
    """端点が tol 以内なら結合して一本化（最長経路を返す）"""
    used = [False] * len(segs)
    paths: List[np.ndarray] = []
    for i, arr in enumerate(segs):
        if used[i]:
            continue
        path = arr.copy()
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, arr2 in enumerate(segs):
                if used[j]:
                    continue
                if np.linalg.norm(path[-1] - arr2[0]) <= tol:
                    path = np.vstack([path, arr2[1:]])
                    used[j] = True
                    changed = True
                    continue
                if np.linalg.norm(path[-1] - arr2[-1]) <= tol:
                    path = np.vstack([path, arr2[-2::-1]])
                    used[j] = True
                    changed = True
                    continue
                if np.linalg.norm(path[0] - arr2[-1]) <= tol:
                    path = np.vstack([arr2[:-1], path])
                    used[j] = True
                    changed = True
                    continue
                if np.linalg.norm(path[0] - arr2[0]) <= tol:
                    path = np.vstack([arr2[::-1][:-1], path])
                    used[j] = True
                    changed = True
                    continue
        paths.append(path)
    if not paths:
        return np.empty((0, 2))
    lens = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for p in paths]
    return paths[int(np.argmax(lens))]


def _join_centerline_segments(path: str, layers: List[str], tol_m: float, scale: float) -> Optional[np.ndarray]:
    """CL レイヤ(複数可)の線分を結合して一本の中心線に。戻り値は [N,2] (m 単位)"""
    doc = _load_doc_from_path(path)
    if doc is None:
        return None
    msp = doc.modelspace()
    try:
        ents = msp.query("LINE LWPOLYLINE SPLINE")
    except Exception:
        ents = [e for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "SPLINE")]
    segs = []
    for e in ents:
        if layers and e.dxf.layer not in layers:
            continue
        arr = _poly_vertices(e)
        if arr is None or len(arr) < 2:
            continue
        segs.append(arr * float(scale))
    if not segs:
        return None
    joined = _connect_segments(segs, tol=float(tol_m))
    return joined if len(joined) >= 2 else None


def _scale_from_two_points(p1: Tuple[float, float], p2: Tuple[float, float], real_dist_m: float) -> float:
    """2 点の CAD 座標から実距離[m]でスケールを逆算（m スケール）"""
    d = float(np.linalg.norm(np.array(p2) - np.array(p1)))
    if d <= 0:
        return 1.0
    return float(real_dist_m) / d


# ──────────────────────────────────────────────────────────────
# 幾何ヘルパ
def _tangent_normal(centerline: np.ndarray, s: float):
    if centerline.shape[0] < 2:
        return centerline[0], np.array([1.0, 0.0]), np.array([0.0, 1.0])
    segs = np.diff(centerline, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(segs) - 1))
    Li = lens[i] if lens[i] > 0 else 1.0
    tau = (s - cum[i]) / Li
    P = centerline[i] + tau * segs[i]
    t = segs[i] / Li
    n = np.array([-t[1], t[0]])
    return P, t, n


def _xs_to_world3D(P: np.ndarray, n: np.ndarray, oz: np.ndarray, z_scale: float = 1.0):
    X = P[0] + oz[:, 0] * n[0]
    Y = P[1] + oz[:, 0] * n[1]
    Z = oz[:, 1] * float(z_scale)
    return X, Y, Z


def _decimate(arr: np.ndarray, max_pts: int) -> np.ndarray:
    if arr is None or arr.ndim != 2 or arr.shape[0] <= max_pts:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_pts).astype(int)
    return arr[idx]


def _decimate1d(arr: np.ndarray, max_pts: int) -> np.ndarray:
    if arr is None or arr.ndim != 2 or arr.shape[0] <= max_pts:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_pts).astype(int)
    return arr[idx]


# ──────────────────────────────────────────────────────────────
# 縦断 s→z
def _profile_eval(s: float) -> Optional[float]:
    prof = st.session_state.get("profile_s")
    if prof is None or len(prof) < 2:
        return None
    ss = prof[:, 0]
    zz = prof[:, 1]
    s = float(s)
    if s <= ss[0]:
        return float(zz[0])
    if s >= ss[-1]:
        return float(zz[-1])
    return float(np.interp(s, ss, zz))


# ──────────────────────────────────────────────────────────────
# 設定保持（平面DXF）
def _set_plan_bytes(file):
    if file is None:
        return
    data = bytes(file.getbuffer())
    h = hashlib.sha1(data).hexdigest()[:12]
    st.session_state.plan_bytes = data
    st.session_state.plan_hash = h
    st.session_state.plan_layers = None
    st.session_state.plan_layer_choice = None
    st.session_state.plan_label_layers = []  # 未選択スタート
    st.session_state.plan_circle_layers = []  # 未選択スタート
    st.session_state.centerline_raw = None
    st.session_state.centerline_joined_applied = False
    st.session_state.labels_raw = None
    st.session_state.circles_raw = None
    st.session_state.no_table = None
    st.session_state._plan_diag = None


def _ensure_plan_layers():
    if not st.session_state.get("plan_bytes"):
        return
    if st.session_state.get("plan_layers") is not None:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(st.session_state.plan_bytes)
        tmp.flush()
        st.session_state._plan_path = tmp.name
    try:
        layers = list_layers(st.session_state._plan_path)
        st.session_state.plan_layers = layers
        if layers and st.session_state.get("plan_layer_choice") is None:
            st.session_state.plan_layer_choice = layers[0].name
    except Exception as e:
        st.session_state.plan_layers = []
        st.session_state.plan_layer_choice = None
        st.warning(f"レイヤ一覧の取得に失敗: {e}")


def _rebuild_s_map_from_raw() -> Dict[str, float]:
    """No → s の再投影（円があれば円、無ければラベルで CL に直投影）"""
    s_map: Dict[str, float] = {}
    if "centerline_raw" not in st.session_state or "no_table" not in st.session_state:
        return s_map
    cl_raw = st.session_state.centerline_raw
    k = float(st.session_state.get("unit_scale_plan_ui", 1.0))
    cl = cl_raw * k

    # ラベル位置（スケール済み）
    labels = st.session_state.get("labels_raw") or []
    label_pos = {lab["key"]: np.array(lab["pos"], float) * k for lab in labels}

    for row in st.session_state.no_table or []:
        key = row["key"]
        circ_raw = np.array(row["circle_xy"], float) if row.get("circle_xy") is not None else None
        src = np.array(circ_raw, float) * k if circ_raw is not None else label_pos.get(key)
        if src is not None:
            s_raw, _ = project_point_to_polyline(cl, src)
            s_map[key] = float(s_raw)
        else:
            # 最後の手段：保存されている s（Step1 時点）を使う
            s_map[key] = float(row.get("s", 0.0))
    return s_map


def _build_assigned_from_raw():
    """割当を再構築（倍率・原点合わせ・Zアンカー等を反映）"""
    if "raw_sections" not in st.session_state or "no_table" not in st.session_state or "centerline_raw" not in st.session_state:
        return

    unit_scale_plan = float(st.session_state.get("unit_scale_plan_ui", 1.0))
    cl_raw = st.session_state.centerline_raw
    cl = cl_raw * unit_scale_plan

    s_map = _rebuild_s_map_from_raw()
    delta_s = float(st.session_state.get("delta_s_all_ui", 0.0))
    for k in list(s_map.keys()):
        s_map[k] += delta_s

    no_to_circle_raw = {
        d["key"]: (np.array(d["circle_xy"]) if d.get("circle_xy") is not None else None)
        for d in st.session_state.no_table or []
    }

    offset_scale = float(st.session_state.get("offset_scale_ui", 1.0))
    elev_scale = float(st.session_state.get("elev_scale_ui", 1.0))
    center_by_section_cl = bool(st.session_state.get("center_by_section_cl_ui", True))
    center_by_circle = bool(st.session_state.get("center_by_circle_ui", False))
    center_o = bool(st.session_state.get("center_o_ui", False))
    user_center_offset = float(st.session_state.get("user_center_offset_ui", 0.0))
    flip_o = bool(st.session_state.get("flip_o_ui", False))
    flip_z = bool(st.session_state.get("flip_z_ui", False))
    z_anchor_mode = st.session_state.get("z_anchor_mode_ui", "横断CLを0に（相対）")

    assigned: Dict[str, Dict] = {}
    for key, rec in st.session_state.get("raw_sections", {}).items():
        sel = rec.get("no_key")
        manual_s = rec.get("manual_s")  # 直接入力 s[m]（任意）
        if sel in s_map:
            s = float(s_map[sel])
        elif manual_s is not None:
            s = float(manual_s)
        else:
            st.warning(f"[{key}] は No も s[m] も未指定のため配置できません。")
            continue

        oz_raw = np.asarray(rec["oz_raw"], float)
        if oz_raw.ndim != 2 or oz_raw.shape[1] < 2:
            continue
        o = oz_raw[:, 0] * offset_scale
        z = oz_raw[:, 1] * elev_scale
        if flip_o:
            o *= -1.0
        if flip_z:
            z *= -1.0

        P, _, n = _tangent_normal(cl, s)

        if center_by_section_cl and (rec.get("o0_from_section") is not None):
            o0 = float(rec["o0_from_section"]) * offset_scale
            o = o - o0
        else:
            circ_raw = no_to_circle_raw.get(sel) if sel else None
            if center_by_circle and circ_raw is not None:
                circ_scaled = circ_raw * unit_scale_plan
                oc0 = float(np.dot(circ_scaled - P, n))
                o = o - oc0
            elif center_o:
                o = o - float(np.median(o))
            else:
                o = o - float(user_center_offset)

        # Zアンカー
        idx = np.argsort(o)
        oo = o[idx]
        zz = z[idx]
        z0 = float(np.interp(0.0, oo, zz)) if len(oo) >= 2 else (float(zz[0]) if len(zz) else 0.0)
        if z_anchor_mode.startswith("横断CLを0に"):
            z = z - z0
        elif z_anchor_mode.startswith("縦断CSVに合わせる"):
            base = _profile_eval(s)
            z = (z - z0) + (float(base) if base is not None else 0.0)
        elif z_anchor_mode == "最小を0（簡易）":
            z = z - float(np.min(z))
        elif z_anchor_mode == "中央値を0（簡易）":
            z = z - float(np.median(z))

        assigned[key] = {"oz": np.column_stack([o, z]).astype(np.float32), "no_key": sel, "s": s}

    st.session_state.centerline = cl.astype(np.float32)
    st.session_state._assigned = assigned


# ──────────────────────────────────────────────────────────────
# 2D QA：Δz
def _qa_delta_metrics(g: np.ndarray, b: np.ndarray, step: float = 0.5) -> Dict[str, float]:
    if g is None or b is None or len(g) < 2 or len(b) < 2:
        return {}
    o_min = max(float(np.min(g[:, 0])), float(np.min(b[:, 0])))
    o_max = min(float(np.max(g[:, 0])), float(np.max(b[:, 0])))
    if o_max <= o_min:
        return {}
    o = np.arange(o_min, o_max + step / 2, step)
    zg = np.interp(o, g[:, 0], g[:, 1])
    zb = np.interp(o, b[:, 0], b[:, 1])
    dz = zg - zb
    return {
        "count": len(dz),
        "dz_max": float(np.max(dz)),
        "dz_min": float(np.min(dz)),
        "dz_mean": float(np.mean(dz)),
        "dz_std": float(np.std(dz)),
    }


# ──────────────────────────────────────────────────────────────
# 画面本体
def page():
    st.title("DXF取り込み｜No×測点円スナップ → 横断の立体配置（CAD忠実モード＋スケール診断）")

    # ===== Step 1: 平面 =====
    with st.expander("Step 1｜平面（中心線＋No.ラベル＋測点円）", expanded=True):
        plan_up = st.file_uploader("平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False, key="plan")

        if plan_up is not None and st.button("このファイルを読み込む/更新", type="primary"):
            _set_plan_bytes(plan_up)
            st.success("平面DXFを読み込みました。")

        unit_scale_plan = st.number_input(
            "平面倍率（mm→m は 0.001）",
            value=float(st.session_state.get("unit_scale_plan_ui", 1.0)),
            step=0.001,
            format="%.3f",
            key="unit_scale_plan_ui",
        )
        find_radius = st.number_input("ラベル→円 紐付け距離しきい[m]", value=12.0, step=1.0, format="%.1f")

        _ensure_plan_layers()
        layers = st.session_state.get("plan_layers") or []
        if layers:
            layer_names = [L.name for L in layers]
            show_names = [f"{L.name}  (len≈{getattr(L,'length_sum',0.0):.1f})" for L in layers]

            # 中心線レイヤ
            default_idx = (
                layer_names.index(st.session_state.get("plan_layer_choice", layer_names[0])) if layer_names else 0
            )
            idx = st.radio(
                "中心線レイヤを選択",
                list(range(len(layer_names))),
                format_func=lambda i: show_names[i],
                index=default_idx if layer_names else 0,
            )
            st.session_state.plan_layer_choice = layer_names[int(idx)]

            # 追加：スケール診断 / 連結 / 2点法
            st.markdown("### 🔧 スケール診断と中心線の整備")
            col_diag = st.columns([1, 1, 1, 1])
            with col_diag[0]:
                if st.button("診断（INSUNITS/外接箱）", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmpx:
                        tmpx.write(st.session_state.plan_bytes)
                        tmpx.flush()
                        st.session_state._plan_path = tmpx.name
                    diag = _diagnose_dxf_bbox(tmpx.name, [st.session_state.plan_layer_choice])
                    if not diag:
                        st.warning("診断データが取得できませんでした。")
                    else:
                        st.session_state._plan_diag = diag
                        st.success("診断結果を保存しました。下の表を確認してください。")

            with col_diag[1]:
                diag_ready = bool(st.session_state.get("_plan_diag"))
                if st.button("推奨倍率を適用", use_container_width=True, disabled=(not diag_ready)):
                    diag = st.session_state.get("_plan_diag") or {}
                    if diag:
                        rec = _suggest_unit_scale_from_diag(diag)
                        st.session_state.unit_scale_plan_ui = float(rec)
                        st.success(f"平面倍率を {rec:g} に設定しました。")

            with col_diag[2]:
                tol = st.number_input("CL連結 端点許容[m]", value=0.50, step=0.1, format="%.2f")
            with col_diag[3]:
                if st.button("中心線を連結して採用", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmpx:
                        tmpx.write(st.session_state.plan_bytes)
                        tmpx.flush()
                        st.session_state._plan_path = tmpx.name
                    joined = _join_centerline_segments(
                        tmpx.name,
                        layers=[st.session_state.plan_layer_choice],
                        tol_m=float(tol),
                        scale=float(st.session_state.unit_scale_plan_ui),
                    )
                    if joined is None or len(joined) < 2:
                        st.warning("連結できませんでした。レイヤ選択や許容距離を見直してください。")
                    else:
                        # raw は「未スケール」で持つ設計 → 逆換算で格納
                        st.session_state.centerline_raw = joined / float(st.session_state.unit_scale_plan_ui)
                        st.session_state.centerline_joined_applied = True
                        st.success(f"中心線を連結して登録しました（{len(joined)} 点）。")

            # 診断サマリー
            diag = st.session_state.get("_plan_diag")
            if diag:
                st.write("**診断結果 (INSUNITS & 図面スパン)**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "units_code": [diag["units_code"]],
                            "units_to_m(INSUNITS)": [diag["units_to_m"]],
                            "span_x": [diag["span"][0]],
                            "span_y": [diag["span"][1]],
                            "entities": [diag["entities"]],
                            "median_seg_len": [diag["median_seg"]],
                            "suggest_scale": [_suggest_unit_scale_from_diag(diag)],
                        }
                    ),
                    use_container_width=True,
                )

            # 2点法キャリブレーション
            st.markdown("### 📏 2点法キャリブレーション（平面倍率）")
            cxy = st.columns(3)
            with cxy[0]:
                p1 = st.text_input("点A (x,y)", value="")
            with cxy[1]:
                p2 = st.text_input("点B (x,y)", value="")
            with cxy[2]:
                real_d = st.number_input("実距離[m]", value=0.0, step=1.0, format="%.3f")
            if st.button("2点法を適用", disabled=(not p1 or not p2 or real_d <= 0)):
                try:
                    x1, y1 = [float(v) for v in p1.split(",")]
                    x2, y2 = [float(v) for v in p2.split(",")]
                    k = _scale_from_two_points((x1, y1), (x2, y2), real_d)
                    st.session_state.unit_scale_plan_ui = float(k)
                    st.success(f"平面倍率を {k:g} に設定しました。")
                except Exception as e:
                    st.error(f"入力を確認してください: {e}")

            c1, c2 = st.columns(2)
            with c1:
                current = st.session_state.get("plan_label_layers") or []
                ms = st.multiselect(
                    "測点ラベルレイヤ（TEXT/MTEXT）",
                    layer_names,
                    default=current,
                    key="plan_label_layers_ms",
                )
                st.session_state.plan_label_layers = ms
                b = st.columns([1, 1, 4])
                with b[0]:
                    if st.button("全選択", key="lab_all"):
                        st.session_state.plan_label_layers = layer_names
                        st.session_state.plan_label_layers_ms = layer_names
                with b[1]:
                    if st.button("全解除", key="lab_none"):
                        st.session_state.plan_label_layers = []
                        st.session_state.plan_label_layers_ms = []
            with c2:
                current2 = st.session_state.get("plan_circle_layers") or []
                ms2 = st.multiselect("測点円レイヤ（CIRCLE）", layer_names, default=current2, key="plan_circle_layers_ms")
                st.session_state.plan_circle_layers = ms2
                b2 = st.columns([1, 1, 4])
                with b2[0]:
                    if st.button("全選択", key="circ_all"):
                        st.session_state.plan_circle_layers = layer_names
                        st.session_state.plan_circle_layers_ms = layer_names
                with b2[1]:
                    if st.button("全解除", key="circ_none"):
                        st.session_state.plan_circle_layers = []
                        st.session_state.plan_circle_layers_ms = []

            if st.button("中心線＋No.＋円 抽出を実行", type="primary"):
                if not st.session_state.plan_label_layers and not st.session_state.plan_circle_layers:
                    st.warning("少なくとも『測点ラベル』または『測点円』のレイヤを1つ以上選んでください。")
                else:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                            tmp.write(st.session_state.plan_bytes)
                            tmp.flush()
                            st.session_state._plan_path = tmp.name

                        # 連結した中心線があればそれを優先
                        cl_raw = st.session_state.get("centerline_raw")
                        if (cl_raw is None) or (not st.session_state.get("centerline_joined_applied", False)):
                            cl_raw = read_centerline(tmp.name, allow_layers=[st.session_state.plan_layer_choice], unit_scale=1.0)
                            if cl_raw is None or len(cl_raw) < 2:
                                cl_raw = read_centerline(tmp.name, allow_layers=None, unit_scale=1.0)
                                st.warning("選択レイヤで中心線が見つからなかったため、全レイヤから探索しました。")
                            st.session_state.centerline_raw = cl_raw
                            st.session_state.centerline_joined_applied = False

                        labels = extract_no_labels(tmp.name, st.session_state.plan_label_layers, unit_scale=1.0)
                        circles = extract_circles(tmp.name, st.session_state.plan_circle_layers, unit_scale=1.0)

                        cl_scaled = st.session_state.centerline_raw * float(unit_scale_plan)
                        rows = []
                        for lab in labels:
                            key = lab["key"]
                            Lxy_s = np.array(lab["pos"], float) * float(unit_scale_plan)
                            cand = []
                            for c in circles:
                                Cxy_s = np.array(c["center"], float) * float(unit_scale_plan)
                                d = float(np.linalg.norm(Lxy_s - Cxy_s))
                                if d <= float(find_radius):
                                    cand.append((d, Cxy_s, c.get("r", None), c.get("layer", "")))
                            if cand:
                                d, Cxy_s, r, lay = min(cand, key=lambda t: t[0])
                                s_now, dist = project_point_to_polyline(cl_scaled, Cxy_s)
                                rows.append(
                                    {
                                        "key": key,
                                        "s": float(s_now),
                                        "label_to_circle": d,
                                        "circle_to_cl": dist,
                                        "circle_r": r,
                                        "circle_layer": lay,
                                        "circle_xy": tuple(Cxy_s / float(unit_scale_plan)),
                                        "status": "OK(circle)",
                                    }
                                )
                            else:
                                s_now, dist = project_point_to_polyline(cl_scaled, Lxy_s)
                                rows.append(
                                    {
                                        "key": key,
                                        "s": float(s_now),
                                        "label_to_circle": None,
                                        "circle_to_cl": dist,
                                        "circle_r": None,
                                        "circle_layer": None,
                                        "circle_xy": None,
                                        "status": "FALLBACK(label→CL)",
                                    }
                                )
                        rows.sort(key=lambda d: d["s"])

                        st.session_state.labels_raw = labels
                        st.session_state.circles_raw = circles
                        st.session_state.no_table = rows
                        st.success(f"中心線: {len(st.session_state.centerline_raw)}点 / No.: {len(rows)}件 を抽出しました。")
                    except Exception as e:
                        st.error(f"抽出に失敗しました: {e}")
        else:
            st.info("DXFを読み込み、『このファイルを読み込む/更新』を押すとレイヤ一覧が表示されます。")

    # ===== Step 1.5: 縦断 =====
    with st.expander("Step 1.5｜縦断（中心線の標高）を設定（任意）", expanded=False):
        up = st.file_uploader("縦断CSV（s,z）", type=["csv"])
        s_scale = st.number_input("s の倍率", value=1.0, step=0.001, format="%.3f")
        z_scale = st.number_input("z の倍率", value=1.0, step=0.001, format="%.3f")
        if up is not None and st.button("縦断を読み込む"):
            try:
                df = pd.read_csv(up)
                ss = df["s"].astype(float).to_numpy() * float(s_scale)
                zz = df["z"].astype(float).to_numpy() * float(z_scale)
                order = np.argsort(ss)
                st.session_state.profile_s = np.column_stack([ss[order], zz[order]]).astype(np.float32)
                st.success(f"縦断を読み込みました（{len(ss)} 点）")
            except Exception as e:
                st.error(f"読み込み失敗: {e}")
        if st.session_state.get("profile_s") is not None:
            prof = st.session_state.profile_s
            st.line_chart({"z": prof[:, 1]}, height=120)

    # ===== Step 2: 横断 =====
    with st.expander("Step 2｜横断を読み込み → 集約/忠実モード → No割当（地層も選択可）", expanded=True):
        xs_files = st.file_uploader("横断DXF/CSV（複数可）", type=["dxf", "csv"], accept_multiple_files=True, key="xs")

        axis_mode = st.selectbox("軸割り", ["X=offset / Y=elev（標準）", "X=elev / Y=offset（入替）"])
        offset_scale = st.number_input(
            "オフセット倍率（mm→m は 0.001）",
            value=float(st.session_state.get("offset_scale_ui", 1.0)),
            step=0.001,
            format="%.3f",
            key="offset_scale_ui",
        )
        elev_scale = st.number_input(
            "標高倍率（mm→m は 0.001）",
            value=float(st.session_state.get("elev_scale_ui", 1.0)),
            step=0.001,
            format="%.3f",
            key="elev_scale_ui",
        )
        flip_o = st.checkbox("オフセット左右反転", value=bool(st.session_state.get("flip_o_ui", False)), key="flip_o_ui")
        flip_z = st.checkbox("標高上下反転", value=bool(st.session_state.get("flip_z_ui", False)), key="flip_z_ui")

        st.markdown("**取り込み方式**")
        exact_mode = st.toggle("CAD忠実モード（Exact / 再標本化・平滑なし）", value=True)
        show_2d = st.checkbox("2Dプレビューを表示", value=True)

        # 従来集約モード（忠実モードOFFのときのみ）
        agg_mode = st.selectbox("複数線の集約（断面本体・忠実モードOFF時）", ["中央値（推奨）", "下包絡（最小）", "上包絡（最大）"])
        smooth_k = st.slider("平滑ウィンドウ（奇数、0で無効）", 0, 21, 3, step=2)
        max_slope = st.slider("最大許容勾配 |dz/dx|（0で無効）", 0.0, 30.0, 0.0, step=0.5)
        target_step = st.number_input("出力間隔 step [m]（忠実モードでは無効）", value=0.50, step=0.05, format="%.2f")

        center_by_section_cl = st.checkbox(
            "横断内のCL縦線を 0 に（自動・推奨）", value=bool(st.session_state.get("center_by_section_cl_ui", True)), key="center_by_section_cl_ui"
        )
        center_by_circle = st.checkbox(
            "道路中心オフセット値を0に（円中心=0, 平面併用）",
            value=bool(st.session_state.get("center_by_circle_ui", False)),
            key="center_by_circle_ui",
        )
        center_o = st.checkbox("オフセット中央値を0に（フォールバック）", value=bool(st.session_state.get("center_o_ui", False)), key="center_o_ui")
        user_center_offset = st.number_input("（手動）道路中心オフセット値", value=float(st.session_state.get("user_center_offset_ui", 0.0)), step=0.1, format="%.3f", key="user_center_offset_ui")

        z_anchor_mode = st.selectbox(
            "高さの合わせ方（Zアンカー）",
            ["しない（CAD絶対標高）", "横断CLを0に（相対）", "縦断CSVに合わせる（CL基準）", "最小を0（簡易）", "中央値を0（簡易）"],
            index=0 if exact_mode else 1,
            key="z_anchor_mode_ui",
        )

        # 忠実モードの微調整
        eps_o = st.number_input("忠実モード：縦棒除去の閾値 |Δo|< [m]", value=0.02, step=0.01, format="%.2f")

        # 地層ライン（従来モード用の既定）
        geol_min_span = st.number_input("地層: 短片除外の最小スパン [m]（忠実モードでは無効）", value=5.0, step=0.5)
        geol_roll_win = st.slider("地層: 平滑化（ローリング中央値の窓）（忠実モードでは無効）", min_value=1, max_value=21, value=7, step=2)
        geol_agg_mode = st.selectbox("地層: 既定の集約モード（忠実モードでは無効）", ["中央値（ロバスト）", "上包絡（90%）", "下包絡（10%）", "最長曲線（最大スパン）"])

        st.session_state.setdefault("raw_sections", {})
        st.session_state.setdefault("raw_sections_bytes", {})
        st.session_state.setdefault("lem_horizons", {})

        no_choices = [d["key"] for d in (st.session_state.get("no_table") or [])]
        agg_map = {"中央値（推奨）": "median", "下包絡（最小）": "lower", "上包絡（最大）": "upper"}

        if xs_files:
            for f in xs_files:
                with st.expander(f"割当：{f.name}", expanded=False):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + f.name.split(".")[-1])
                    tmp.write(f.getbuffer())
                    tmp.flush()
                    tmp.close()

                    st.session_state.raw_sections_bytes[f.name] = bytes(f.getbuffer())

                    # レイヤ選択
                    layer_name = None
                    geology_layers: List[str] = []
                    cl_hint_layers: List[str] = []
                    all_layer_names: List[str] = []

                    if f.name.lower().endswith(".dxf"):
                        try:
                            scan_layers = list_layers(tmp.name)
                            all_layer_names = [L.name for L in scan_layers]
                            layer_name = st.selectbox("横断レイヤ（任意／未選択=自動）", ["（未選択）"] + all_layer_names)
                            if layer_name == "（未選択）":
                                layer_name = None
                            cl_hint_layers = st.multiselect("CL縦線レイヤ（任意）", all_layer_names, default=[])
                            geology_layers = st.multiselect("地層レイヤ（任意・複数）", all_layer_names, default=[])
                        except Exception:
                            pass

                        # 追加：スケール診断 & 2点法（横断）
                        st.markdown("#### 🔧 スケール診断（この横断DXF）")
                        if st.button(f"[{f.name}] 診断を実行", key=f"diag_{f.name}"):
                            diag_xs = _diagnose_dxf_bbox(tmp.name, None)
                            st.session_state[f"_xs_diag_{f.name}"] = diag_xs
                            if diag_xs:
                                st.success("診断結果を保存しました。下に表示します。")
                            else:
                                st.warning("診断できませんでした。")

                        diag_xs = st.session_state.get(f"_xs_diag_{f.name}")
                        if diag_xs:
                            k_suggest = _suggest_unit_scale_from_diag(diag_xs)
                            st.write(
                                pd.DataFrame(
                                    {
                                        "units_code": [diag_xs["units_code"]],
                                        "units_to_m(INSUNITS)": [diag_xs["units_to_m"]],
                                        "span_x": [diag_xs["span"][0]],
                                        "span_y": [diag_xs["span"][1]],
                                        "suggest_scale": [k_suggest],
                                    }
                                )
                            )
                            c_apply = st.columns(3)
                            with c_apply[0]:
                                if st.button("offset に適用", key=f"xs_apply_o_{f.name}"):
                                    st.session_state.offset_scale_ui = float(k_suggest)
                            with c_apply[1]:
                                if st.button("elev に適用", key=f"xs_apply_z_{f.name}"):
                                    st.session_state.elev_scale_ui = float(k_suggest)
                            with c_apply[2]:
                                if st.button("offset & elev 両方に適用", key=f"xs_apply_both_{f.name}"):
                                    st.session_state.offset_scale_ui = float(k_suggest)
                                    st.session_state.elev_scale_ui = float(k_suggest)

                        st.markdown("#### 📏 2点法キャリブレーション（横断スケール）")
                        cxy2 = st.columns(4)
                        with cxy2[0]:
                            p1x = st.text_input("点A x", key=f"xs_p1x_{f.name}", value="")
                        with cxy2[1]:
                            p1y = st.text_input("点A y", key=f"xs_p1y_{f.name}", value="")
                        with cxy2[2]:
                            p2x = st.text_input("点B x", key=f"xs_p2x_{f.name}", value="")
                        with cxy2[3]:
                            p2y = st.text_input("点B y", key=f"xs_p2y_{f.name}", value="")
                        cxy3 = st.columns(3)
                        with cxy3[0]:
                            real_d2 = st.number_input(
                                "実距離[m]（offsetに適用）", value=0.0, step=0.1, format="%.3f", key=f"xs_real_o_{f.name}"
                            )
                        with cxy3[1]:
                            real_h2 = st.number_input(
                                "実高低差[m]（elevに適用）", value=0.0, step=0.1, format="%.3f", key=f"xs_real_z_{f.name}"
                            )
                        with cxy3[2]:
                            if st.button("2点法を適用", key=f"xs_apply_2pt_{f.name}"):
                                try:
                                    p1 = (float(p1x), float(p1y))
                                    p2 = (float(p2x), float(p2y))
                                    if real_d2 > 0:
                                        ko = _scale_from_two_points(p1, p2, real_d2)
                                        st.session_state.offset_scale_ui = float(ko)
                                    if real_h2 > 0:
                                        # 軸割に応じて elev の差分を抽出
                                        if axis_mode.startswith("X=offset"):
                                            dz_cad = abs(float(p2[1] - p1[1]))
                                        else:
                                            dz_cad = abs(float(p2[0] - p1[0]))
                                        kz = float(real_h2) / dz_cad if dz_cad > 0 else st.session_state.elev_scale_ui
                                        st.session_state.elev_scale_ui = float(kz)
                                    st.success("2点法スケールを適用しました。")
                                except Exception as e:
                                    st.error(f"入力を確認してください: {e}")

                    # u0（CL縦線 or 自動）検出（忠実モード時）
                    u0 = None
                    try:
                        if f.name.lower().endswith(".dxf") and exact_mode and center_by_section_cl:
                            u0 = detect_section_centerline_u(tmp.name, layer_hint=(cl_hint_layers or None), unit_scale=1.0)
                    except Exception:
                        u0 = None

                    # ===== 忠実モード（DXFのみ） =====
                    if exact_mode and f.name.lower().endswith(".dxf"):
                        doc = _load_doc_from_path(tmp.name)
                        if doc is None:
                            st.error("ezdxf が必要です。requirements.txt に 'ezdxf' を追加してください。")
                            continue
                        msp = doc.modelspace()
                        try:
                            ents = msp.query("LINE LWPOLYLINE SPLINE")
                        except Exception:
                            ents = [e for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "SPLINE")]

                        # レイヤ→ポリライン
                        layer_to_polys: Dict[str, List[np.ndarray]] = {}
                        for e in ents:
                            arr = _poly_vertices(e)
                            if arr is None or len(arr) < 2:
                                continue
                            layer_to_polys.setdefault(e.dxf.layer, []).append(arr)

                        # 断面候補（横断レイヤ未指定なら全レイヤ）
                        sec_cands: List[Tuple[str, np.ndarray]] = []
                        targets_for_sec = [layer_name] if layer_name else list(layer_to_polys.keys())
                        for lay in targets_for_sec:
                            for arr in layer_to_polys.get(lay, []):
                                sec_cands.append((lay, arr))
                        if not sec_cands:
                            st.warning("断面候補が見つかりませんでした。横断レイヤを指定してみてください。")
                            continue

                        def _span_xy(a: np.ndarray) -> float:
                            return float(np.max(a[:, 0]) - np.min(a[:, 0]))

                        labels_sec = [f"{i}: {lay}  span={_span_xy(a):.2f}" for i, (lay, a) in enumerate(sec_cands)]
                        def_idx = int(np.argmax([_span_xy(a) for _, a in sec_cands]))

                        # ★ 地表（断面）を複数選択可能に
                        sel_sec_multi = st.multiselect(
                            "忠実モード：『地表（断面）』に採用するポリライン（複数可）",
                            list(range(len(sec_cands))),
                            default=[def_idx],
                            format_func=lambda i: labels_sec[i],
                            key=f"exact_sec_multi_{f.name}",
                        )

                        # 地層候補（複数）
                        geo_cands: List[Tuple[str, int, np.ndarray]] = []
                        pick_layers = geology_layers[:]  # copy
                        if layer_name and (layer_name not in pick_layers):
                            pick_layers.append(layer_name)  # 同一レイヤの境界も拾いたいケース
                        for lay in pick_layers:
                            for j, arr in enumerate(layer_to_polys.get(lay, [])):
                                geo_cands.append((lay, j, arr))
                        labels_geo = [f"{lay}#[{j}]  span={_span_xy(arr):.2f}" for (lay, j, arr) in geo_cands]
                        sel_geo = st.multiselect(
                            "忠実モード：『地層』に採用するポリライン（複数可）",
                            list(range(len(geo_cands))),
                            default=[],
                            format_func=lambda i: labels_geo[i],
                            key=f"exact_geo_{f.name}",
                        )

                        # 2D プレビュー用 figure
                        if show_2d:
                            fig2 = go.Figure()

                        # 各“地表（断面）”について個別に登録
                        for k_idx, k in enumerate(sel_sec_multi):
                            sec_xy = sec_cands[k][1]
                            sec_oz = _to_section_oz(
                                sec_xy,
                                axis=axis_mode,
                                off_scale=float(offset_scale),
                                z_scale=float(elev_scale),
                                flip_o=flip_o,
                                flip_z=flip_z,
                                u0=u0,
                            )
                            sec_oz = _filter_near_vertical(sec_oz, eps_o=float(eps_o))

                            # 地層も計算（同じ選択を各断面に適用）
                            geology_over: Dict[str, np.ndarray] = {}
                            for k_geo in sel_geo:
                                lay, j, arr = geo_cands[k_geo]
                                oz = _to_section_oz(
                                    arr,
                                    axis=axis_mode,
                                    off_scale=float(offset_scale),
                                    z_scale=float(elev_scale),
                                    flip_o=flip_o,
                                    flip_z=flip_z,
                                    u0=u0,
                                )
                                oz = _filter_near_vertical(oz, eps_o=float(eps_o))
                                geology_over[f"{lay}#{j}"] = oz

                            # No 候補＋フォールバック
                            guess = normalize_no_key(f.name) or ""
                            choices = ["（未選択）"] + ([guess] if (guess and guess not in no_choices) else []) + no_choices
                            sel_no = st.selectbox(
                                f"[{f.name} 断面#{k_idx}] 割当No.",
                                choices,
                                index=choices.index(guess) if guess in choices else 0,
                                key=f"no_sel_{f.name}_{k}",
                            )
                            custom_no = st.text_input(
                                f"[{f.name} 断面#{k_idx}] No. 手入力（任意）",
                                value="",
                                key=f"no_custom_{f.name}_{k}",
                                placeholder="例: No.0+40 など（ラベルと一致すれば優先）",
                            )
                            manual_s = st.number_input(
                                f"[{f.name} 断面#{k_idx}] No無し：中心線 s[m] を直接入力（任意）",
                                value=0.0,
                                min_value=0.0,
                                step=1.0,
                                format="%.3f",
                                key=f"manual_s_{f.name}_{k}",
                            )

                            # 2D プレビュー
                            if show_2d:
                                fig2.add_trace(
                                    go.Scatter(
                                        x=sec_oz[:, 0],
                                        y=sec_oz[:, 1],
                                        mode="lines",
                                        name=f"断面#{k_idx}",
                                        line=dict(width=3, color="#FFFFFF"),
                                    )
                                )
                                if u0 is not None:
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=[0, 0],
                                            y=[float(np.nanmin(sec_oz[:, 1])), float(np.nanmax(sec_oz[:, 1]))],
                                            mode="lines",
                                            name=f"CL#{k_idx}",
                                            line=dict(width=1, dash="dot", color="#8AA0FF"),
                                        )
                                    )
                                for nm, arr in geology_over.items():
                                    fig2.add_trace(
                                        go.Scatter(
                                            x=arr[:, 0],
                                            y=arr[:, 1],
                                            mode="lines",
                                            name=f"層:{nm} (#{k_idx})",
                                            line=dict(width=2, color="#FFB0A0"),
                                        )
                                    )

                            # QA（Δz）— 先頭の層のみ
                            if geology_over:
                                first_nm = list(geology_over.keys())[0]
                                met = _qa_delta_metrics(sec_oz, geology_over[first_nm], step=0.5)
                                if met:
                                    st.info(
                                        f"Δz QA（断面#{k_idx} − 層 '{first_nm}'）: "
                                        f"count={met['count']}  max={met['dz_max']:.3f}  "
                                        f"min={met['dz_min']:.3f}  mean={met['dz_mean']:.3f}  std={met['dz_std']:.3f}"
                                    )

                            # 永続化（複数断面をそれぞれ個別キーで保存）
                            sec_key = f"{f.name}#G{k}"
                            # No の決定
                            final_no = None
                            if sel_no and sel_no != "（未選択）":
                                final_no = sel_no
                            elif custom_no.strip():
                                final_no = custom_no.strip()

                            st.session_state.raw_sections[sec_key] = {
                                "oz_raw": sec_oz,
                                "guess_no": final_no,
                                "no_key": final_no,
                                "manual_s": (None if manual_s <= 0 else float(manual_s)),
                                "o0_from_section": 0.0,  # 忠実モードは u0 を既に引いた o
                            }
                            st.session_state.lem_horizons[sec_key] = geology_over

                        if show_2d:
                            fig2.update_layout(
                                height=320,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="offset [m]",
                                yaxis_title="elev [m]",
                                paper_bgcolor="#0f1115",
                                plot_bgcolor="#0f1115",
                            )
                            fig2.update_xaxes(gridcolor="#2a2f3a")
                            fig2.update_yaxes(gridcolor="#2a2f3a")
                            st.plotly_chart(fig2, use_container_width=True)

                        st.success("忠実モードで取り込み（複数断面）を登録しました。")

                    # ===== 従来モード（参考） =====
                    else:
                        sec = read_single_section_file(
                            tmp.name,
                            layer_name=layer_name,
                            unit_scale=1.0,
                            aggregate=agg_map[agg_mode],
                            smooth_k=int(smooth_k),
                            max_slope=float(max_slope),
                            target_step=float(target_step),
                        )
                        if sec is None:
                            st.warning("断面を認識できませんでした。レイヤや倍率をご確認ください。")
                            continue

                        # 軸入替
                        if axis_mode.startswith("X=elev"):
                            o_raw = sec[:, 1].astype(float)
                            z_raw = sec[:, 0].astype(float)
                        else:
                            o_raw = sec[:, 0].astype(float)
                            z_raw = sec[:, 1].astype(float)
                        oz_raw = np.column_stack([o_raw, z_raw])

                        # No 候補＋フォールバック
                        guess = normalize_no_key(f.name) or ""
                        choices = ["（未選択）"] + ([guess] if (guess and guess not in no_choices) else []) + no_choices
                        sel = st.selectbox("割当No.", choices, index=choices.index(guess) if guess in choices else 0)
                        custom_no = st.text_input("No. 手入力（任意）", value="", placeholder="例: No.1+40 など")
                        manual_s = st.number_input("No無し：中心線 s[m] を直接入力（任意）", value=0.0, min_value=0.0, step=1.0, format="%.3f")

                        # 地層（従来集約）
                        geology_over: Dict[str, np.ndarray] = {}
                        if geology_layers and f.name.lower().endswith(".dxf"):
                            doc = _load_doc_from_path(tmp.name)
                            if doc is None:
                                st.info("地層レイヤの抽出には ezdxf が必要です。requirements.txt に 'ezdxf' を追記してください。")
                            else:
                                msp = doc.modelspace()
                                try:
                                    ents = msp.query("LINE LWPOLYLINE SPLINE")
                                except Exception:
                                    ents = [e for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "SPLINE")]
                                for lay in geology_layers:
                                    segs = []
                                    for e in ents:
                                        if e.dxf.layer != lay:
                                            continue
                                        arr = _poly_vertices(e)
                                        if arr is None or len(arr) < 2:
                                            continue
                                        segs.append(arr)
                                    if not segs:
                                        continue

                                    oz_list = []
                                    for sgm in segs:
                                        oz_l = _to_section_oz(
                                            sgm,
                                            axis=axis_mode,
                                            off_scale=float(offset_scale),
                                            z_scale=float(elev_scale),
                                            flip_o=flip_o,
                                            flip_z=flip_z,
                                            u0=None,
                                        )
                                        if len(oz_l) >= 2:
                                            oz_list.append(oz_l)

                                    if oz_list:
                                        merged = _merge_segments_geo(
                                            oz_list,
                                            step=float(target_step),
                                            min_span=float(geol_min_span),
                                            roll=int(geol_roll_win),
                                            mode=geol_agg_mode,
                                        )
                                        if len(merged) > 0:
                                            geology_over[lay] = merged

                        if show_2d:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=oz_raw[:, 0], y=oz_raw[:, 1], mode="lines", name="断面", line=dict(width=3, color="#FFFFFF")))
                            for nm, arr in geology_over.items():
                                fig2.add_trace(go.Scatter(x=arr[:, 0], y=arr[:, 1], mode="lines", name=f"層:{nm}", line=dict(width=2)))
                            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="offset [m]", yaxis_title="elev [m]",
                                               paper_bgcolor="#0f1115", plot_bgcolor="#0f1115")
                            fig2.update_xaxes(gridcolor="#2a2f3a")
                            fig2.update_yaxes(gridcolor="#2a2f3a")
                            st.plotly_chart(fig2, use_container_width=True)

                        final_no = None
                        if sel and sel != "（未選択）":
                            final_no = sel
                        elif custom_no.strip():
                            final_no = custom_no.strip()

                        st.session_state.raw_sections[f.name] = {
                            "oz_raw": oz_raw,
                            "guess_no": final_no,
                            "no_key": final_no,
                            "manual_s": (None if manual_s <= 0 else float(manual_s)),
                            "o0_from_section": None,
                        }
                        st.session_state.lem_horizons[f.name] = geology_over

        if st.button("変更を適用（再計算）", type="primary"):
            try:
                _build_assigned_from_raw()
                st.success("割当を再構築しました。")
            except Exception as e:
                st.error(f"再適用エラー: {e}")

        st.info(f"割当済み：{len(st.session_state.get('_assigned', {}))} / 取り込み済み：{len(st.session_state.get('raw_sections', {}))}")

    # ===== Step 3: 3D プレビュー =====
    with st.expander("Step 3｜3Dプレビュー（立体配置＋円弧の簡易表示）", expanded=True):
        can_run = ("centerline" in st.session_state) and ("_assigned" in st.session_state) and st.session_state._assigned
        if not can_run:
            st.warning("まず Step1/2 を完了してください。")
            return

        cl = st.session_state.centerline
        assigned = st.session_state._assigned

        # ---- 表示フィルタ ----
        items_sorted = sorted(assigned.items(), key=lambda kv: kv[1]["s"])
        option_labels = [f"{(rec['no_key'] or key)}  |  s={rec['s']:.1f} m" for key, rec in items_sorted]
        option_keys = [key for key, _ in items_sorted]

        default_labels = st.session_state.get("step3_show_labels", option_labels)
        col_btn = st.columns([1, 1, 8])
        with col_btn[0]:
            if st.button("全選択", use_container_width=True):
                default_labels = option_labels
        with col_btn[1]:
            if st.button("全解除", use_container_width=True):
                default_labels = []

        selected_labels = st.multiselect(
            "表示する断面（複数可）",
            option_labels,
            default=default_labels,
            key="step3_show_labels",
        )
        selected_keys = {option_keys[option_labels.index(lbl)] for lbl in selected_labels}

        # ---- 表示オプション ----
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            z_scale = st.number_input("縦倍率（標高）", value=1.0, step=0.1, format="%.1f")
        with c2:
            max_pts_cl = st.number_input("中心線の最大点数", value=4000, step=500, min_value=500, max_value=20000)
        with c3:
            max_pts_xs = st.number_input("断面1本あたりの最大点数", value=1200, step=100, min_value=200, max_value=10000)
        with c4:
            show_arcs = st.checkbox("円弧を表示", value=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            show_centerline = st.checkbox("中心線を表示", value=True)
        with c6:
            show_zero_axis = st.checkbox("o=0 軸を表示", value=False)
        with c7:
            show_cl_axis = st.checkbox("CL 縦線を表示", value=True)

        # ---- 描画 ----
        fig = go.Figure()
        if show_centerline:
            cl_plot = _decimate1d(cl, int(max_pts_cl))
            fig.add_trace(
                go.Scatter3d(
                    x=cl_plot[:, 0],
                    y=cl_plot[:, 1],
                    z=np.zeros(len(cl_plot)),
                    mode="lines",
                    name="Centerline",
                    line=dict(width=4, color="#A0A6B3"),
                )
            )

        if not selected_keys:
            st.plotly_chart(fig, use_container_width=True, height=760)
            st.info("表示する断面が選択されていません。")
            return

        for sec_key, rec in items_sorted:
            if sec_key not in selected_keys:
                continue
            s = float(rec["s"])
            oz = np.asarray(rec["oz"], float)

            P, t, n = _tangent_normal(cl, s)
            oz_plot = _decimate(oz, int(max_pts_xs))
            X, Y, Z = _xs_to_world3D(P, n, oz_plot, z_scale=z_scale)
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="lines",
                    name=f"{rec['no_key'] or sec_key}",
                    line=dict(width=5, color="#FFFFFF"),
                    opacity=0.98,
                )
            )

            # o=0 軸（水平）
            if show_zero_axis:
                omin, omax = float(np.min(oz_plot[:, 0])), float(np.max(oz_plot[:, 0]))
                Xb, Yb, Zb = _xs_to_world3D(P, n, np.array([[omin, 0.0], [omax, 0.0]]), z_scale=z_scale)
                fig.add_trace(
                    go.Scatter3d(
                        x=Xb,
                        y=Yb,
                        z=Zb,
                        mode="lines",
                        showlegend=False,
                        line=dict(width=2, color="#777777"),
                    )
                )

            # CL 縦線（o=0 の鉛直）
            if show_cl_axis:
                zmin, zmax = float(np.min(oz_plot[:, 1])) * z_scale, float(np.max(oz_plot[:, 1])) * z_scale
                Xp, Yp, Zp = _xs_to_world3D(P, n, np.array([[0.0, zmin], [0.0, zmax]]), z_scale=1.0)
                fig.add_trace(
                    go.Scatter3d(
                        x=Xp,
                        y=Yp,
                        z=Zp,
                        mode="lines",
                        showlegend=False,
                        line=dict(width=3, color="#8888FF"),
                    )
                )

            # 簡易円弧
            if show_arcs:
                try:
                    res = compute_min_circle({"section": oz})
                    circ = res.get("circle", {}) if res else {}
                    oc, zc, R = circ.get("oc"), circ.get("zc"), circ.get("R")
                    if oc is not None and zc is not None and R is not None and np.isfinite(R):
                        th = np.linspace(-np.pi, np.pi, 240)
                        xo = float(oc) + float(R) * np.cos(th)
                        zo = float(zc) + float(R) * np.sin(th)
                        X2 = P[0] + xo * n[0]
                        Y2 = P[1] + xo * n[1]
                        Z2 = zo * z_scale
                        fig.add_trace(
                            go.Scatter3d(
                                x=X2,
                                y=Y2,
                                z=Z2,
                                mode="lines",
                                showlegend=False,
                                line=dict(width=3, color="#FF9500"),
                            )
                        )
                except Exception:
                    pass

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            paper_bgcolor="#0f1115",
            plot_bgcolor="#0f1115",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, height=760)


# ──────────────────────────────────────────────────────────────
# 既存の地層集約（忠実モード OFF 用）— そのまま流用（簡略版）
def _merge_segments_geo(
    oz_list: List[np.ndarray],
    step: float = 0.2,
    min_span: float = 5.0,
    roll: int = 7,
    mode: str = "中央値（ロバスト）",
) -> np.ndarray:
    """複数断面をロバストにマージ（既存ロジック簡略版）"""
    if not oz_list:
        return np.empty((0, 2), float)

    omin = min(float(np.min(oz[:, 0])) for oz in oz_list)
    omax = max(float(np.max(oz[:, 0])) for oz in oz_list)
    if omax <= omin:
        return np.empty((0, 2), float)
    grid = np.arange(omin, omax + step / 2.0, step)

    stack = []
    for oz in oz_list:
        if len(oz) < 2:
            continue
        z = np.interp(grid, oz[:, 0], oz[:, 1], left=np.nan, right=np.nan)
        stack.append(z)
    if not stack:
        return np.empty((0, 2), float)
    A = np.vstack(stack)  # n × m
    with np.errstate(invalid="ignore"):
        if mode.startswith("最長"):
            # 有効データ点数が最大の系列（=実質一番長い曲線）
            # ここでは安定のため中央値を採用
            z = np.nanmedian(A, axis=0)
        elif mode.startswith("上包絡"):
            z = np.nanpercentile(A, 90, axis=0)
        elif mode.startswith("下包絡"):
            z = np.nanpercentile(A, 10, axis=0)
        else:
            z = np.nanmedian(A, axis=0)

    m = ~np.isnan(z)
    if not np.any(m):
        return np.empty((0, 2), float)
    oz = np.column_stack([grid[m], z[m]])
    # 簡易ローリング中央値
    if roll and roll >= 3 and roll % 2 == 1:
        k = roll
        z2 = oz[:, 1].copy()
        for i in range(len(oz)):
            a = max(0, i - k // 2)
            b = min(len(oz), i + k // 2 + 1)
            z2[i] = np.median(oz[a:b, 1])
        oz[:, 1] = z2
    return oz
