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
# DXF/CSV ユーティリティ
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
# DXF リーダ（忠実モードでも使用）
def _load_doc_from_path(path: str):
    try:
        import ezdxf
        return ezdxf.readfile(path)
    except Exception:
        return None


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


def _to_section_oz(
    seg: np.ndarray,
    *,
    axis="X=offset / Y=elev（標準）",
    off_scale=1.0,
    z_scale=1.0,
    flip_o=False,
    flip_z=False,
    u0: Optional[float] = None,
) -> np.ndarray:
    """平面XY → 断面(o,z) に変換（忠実モード：並べ替え以外は加工しない）"""
    if axis.startswith("X=elev"):
        o = seg[:, 1].astype(float) * off_scale
        z = seg[:, 0].astype(float) * z_scale
    else:
        o = seg[:, 0].astype(float) * off_scale
        z = seg[:, 1].astype(float) * z_scale
    if flip_o:
        o *= -1.0
    if flip_z:
        z *= -1.0
    if u0 is not None:
        o = o - float(u0) * float(off_scale)
    # o で昇順、同一 o は平均化（可視化安定用）
    idx = np.argsort(o)
    o = o[idx]
    z = z[idx]
    if len(o) >= 2:
        uniq, inv = np.unique(np.round(o, 6), return_inverse=True)
        acc = np.zeros_like(uniq)
        cnt = np.zeros_like(uniq)
        for i, j in enumerate(inv):
            acc[j] += z[i]
            cnt[j] += 1
        z = acc / np.maximum(cnt, 1)
        o = uniq
    return np.column_stack([o, z])


def _filter_near_vertical(oz: np.ndarray, eps_o: float = 0.02) -> np.ndarray:
    """ほぼ縦棒（|Δo|<eps）を除去（値の破壊最小）"""
    if oz is None or len(oz) < 2:
        return oz
    o = oz[:, 0]
    z = oz[:, 1]
    keep = [0]
    for i in range(1, len(o)):
        if abs(o[i] - o[i - 1]) >= eps_o:
            keep.append(i)
    if keep[-1] != len(o) - 1:
        keep.append(len(o) - 1)
    keep = sorted(set(keep))
    return np.column_stack([o[keep], z[keep]])


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
# Step1: 平面DXFの永続化・レイヤ一覧
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
    st.session_state.labels_raw = None
    st.session_state.circles_raw = None
    st.session_state.no_table = None


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
            # s が決まらないと立体配置できない
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

        # CL/円/中央値/手動の順で 0 オフセット
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
# 画面本体
def page():
    st.title("DXF取り込み｜No×測点円スナップ → 横断の立体配置（CAD忠実モード対応）")

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

            c1, c2 = st.columns(2)
            with c1:
                current = st.session_state.get("plan_label_layers") or []
                ms = st.multiselect(
                    "測点ラベルレイヤ（TEXT/MTEXT）", layer_names, default=current, key="plan_label_layers_ms"
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

                        cl_raw = read_centerline(tmp.name, allow_layers=[st.session_state.plan_layer_choice], unit_scale=1.0)
                        if cl_raw is None or len(cl_raw) < 2:
                            cl_raw = read_centerline(tmp.name, allow_layers=None, unit_scale=1.0)
                            st.warning("選択レイヤで中心線が見つからなかったため、全レイヤから探索しました。")

                        labels = extract_no_labels(tmp.name, st.session_state.plan_label_layers, unit_scale=1.0)
                        circles = extract_circles(tmp.name, st.session_state.plan_circle_layers, unit_scale=1.0)

                        cl_scaled = cl_raw * float(unit_scale_plan)
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

                        st.session_state.centerline_raw = cl_raw
                        st.session_state.labels_raw = labels
                        st.session_state.circles_raw = circles
                        st.session_state.no_table = rows
                        st.success(f"中心線: {len(cl_raw)}点 / No.: {len(rows)}件 を抽出しました。")
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

                    # u0（CL縦線 or 自動）検出
                    u0 = None
                    try:
                        if f.name.lower().endswith(".dxf") and center_by_section_cl:
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

        z_scale = st.number_input("縦倍率（標高）", value=1.0, step=0.1, format="%.1f")
        max_pts_cl = st.number_input("中心線の最大点数（3D間引き）", value=4000, step=500, min_value=500, max_value=20000)
        max_pts_xs = st.number_input("断面1本あたりの最大点数（3D間引き）", value=1200, step=100, min_value=200, max_value=10000)
        show_arcs = st.checkbox("円弧を表示する（簡易）", value=True)

        fig = go.Figure()
        cl_plot = _decimate1d(cl, int(max_pts_cl))
        fig.add_trace(go.Scatter3d(x=cl_plot[:, 0], y=cl_plot[:, 1], z=np.zeros(len(cl_plot)), mode="lines",
                                   name="Centerline", line=dict(width=4, color="#A0A6B3")))

        for sec_key, rec in sorted(assigned.items(), key=lambda kv: kv[1]["s"]):
            s = float(rec["s"])
            oz = np.asarray(rec["oz"], float)
            P, t, n = _tangent_normal(cl, s)
            oz_plot = _decimate(oz, int(max_pts_xs))
            X, Y, Z = _xs_to_world3D(P, n, oz_plot, z_scale=z_scale)
            fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", name=f"{rec['no_key'] or sec_key}", line=dict(width=5, color="#FFFFFF"), opacity=0.98))

            omin, omax = float(np.min(oz_plot[:, 0])), float(np.max(oz_plot[:, 0]))
            Xb, Yb, Zb = _xs_to_world3D(P, n, np.array([[omin, 0.0], [omax, 0.0]]), z_scale=z_scale)
            fig.add_trace(go.Scatter3d(x=Xb, y=Yb, z=Zb, mode="lines", showlegend=False, line=dict(width=2, color="#777777")))

            zmin, zmax = float(np.min(oz_plot[:, 1])) * z_scale, float(np.max(oz_plot[:, 1])) * z_scale
            Xp, Yp, Zp = _xs_to_world3D(P, n, np.array([[0.0, zmin], [0.0, zmax]]), z_scale=1.0)
            fig.add_trace(go.Scatter3d(x=Xp, y=Yp, z=Zp, mode="lines", showlegend=False, line=dict(width=3, color="#8888FF")))

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
                        fig.add_trace(go.Scatter3d(x=X2, y=Y2, z=Z2, mode="lines", showlegend=False, line=dict(width=3, color="#FF9500")))
                except Exception:
                    pass

        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
                          paper_bgcolor="#0f1115", plot_bgcolor="#0f1115", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, height=760)


# ──────────────────────────────────────────────────────────────
# 既存の地層集約（忠実モード OFF 用）— そのまま流用
def _merge_segments_geo(oz_list: List[np.ndarray], step: float = 0.2, min_span: float = 5.0, roll: int = 7, mode: str = "中央値（ロバスト）") -> np.ndarray:
    """複数断面をロバストにマージ（既存ロジック簡略版）"""
    if not oz_list:
        return np.empty((0, 2), float)

    # すべての o 範囲を結合
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
            valid_counts = np.sum(~np.isnan(A), axis=0)
            best = np.argmax(valid_counts)  # 参考：単純選択（厳密ではない）
            z = np.nanmedian(A, axis=0)  # 安定のため中央値
        elif mode.startswith("上包絡"):
            z = np.nanpercentile(A, 90, axis=0)
        elif mode.startswith("下包絡"):
            z = np.nanpercentile(A, 10, axis=0)
        else:
            z = np.nanmedian(A, axis=0)

    # 短片除外（NaN 部分を除く）
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
