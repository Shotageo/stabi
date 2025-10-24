# stabi_viz/plan_preview_upload.py  — minimal 3-step workflow
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 既存 I/O ユーティリティ
from stabi_io.dxf_sections import (
    list_layers,
    read_centerline,
    extract_no_labels,
    extract_circles,
    project_point_to_polyline,
    read_single_section_file,
)

# ezdxf は INSUNITS の取得にのみ使用（無ければスキップ）
try:
    import ezdxf  # type: ignore

    _HAS_EZDXF = True
except Exception:
    _HAS_EZDXF = False


# ──────────────────────────────────────────────────────────────
# 1) 共通小物
INSUNITS_TO_M = {
    0: 1.0,  # Unitless
    1: 0.0254,  # Inch
    2: 0.3048,  # Foot
    3: 1609.344,  # Mile
    4: 0.001,  # Millimeter
    5: 0.01,  # Centimeter
    6: 1.0,  # Meter
    7: 1000.0,  # Kilometer
    10: 0.9144,  # Yard
    14: 0.1,  # Decimeter
}

STATION_PAT = re.compile(r"(?:no[.\s]*|n[.\s]*o[.\s]*)?(\d+)[+･\-\s]?(\d+)?", re.IGNORECASE)


def parse_station_to_m(text: str) -> Optional[float]:
    """
    'No.1+40' -> 140.0, '1+20' -> 120.0, '120' -> 120.0
    """
    if not text:
        return None
    m = STATION_PAT.search(text)
    if not m:
        # 純粋な数値だけの場合も拾う
        try:
            return float(text)
        except Exception:
            return None
    a = float(m.group(1))
    b = float(m.group(2) or 0)
    return a * 100.0 + b


def read_insunits_scale(dxf_path: str) -> Optional[float]:
    """DXF の $INSUNITS から m への倍率を返す（無ければ None）"""
    if not _HAS_EZDXF:
        return None
    try:
        doc = ezdxf.readfile(dxf_path)
        code = int(doc.header.get("$INSUNITS", 0))
        return float(INSUNITS_TO_M.get(code, 1.0))
    except Exception:
        return None


def robust_median_ratio(xs: List[float], ys: List[float]) -> Optional[float]:
    """
    比のロバスト推定： ratio_i = xs[i]/ys[i] の中央値を返す。
    0 や NaN は弾く。十分なデータが無ければ None。
    """
    arr = []
    for x, y in zip(xs, ys):
        if y is None or x is None:
            continue
        if y == 0:
            continue
        r = float(x) / float(y)
        if np.isfinite(r):
            arr.append(r)
    if len(arr) >= 1:
        return float(np.median(arr))
    return None


def polyline_length(p: np.ndarray) -> float:
    if p is None or len(p) < 2:
        return 0.0
    d = np.diff(p, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


@dataclass
class PlanAnalysis:
    scale_suggest: float
    scale_source: str  # "station", "insunits", "heuristics"
    cl_raw: np.ndarray  # 未スケール座標
    cl_scaled: np.ndarray  # スケール適用後
    no_rows: List[Dict]  # {"label":..,"station_m":..,"s":..,"dist":..}
    summary: Dict[str, float]  # 長さ/件数など


# ──────────────────────────────────────────────────────────────
# 2) 平面解析（シンプル版）
def analyze_plan(
    dxf_bytes: bytes,
    cl_layer: str,
    label_layers: List[str],
    circle_layers: List[str],
    link_radius_m: float = 12.0,
) -> PlanAnalysis:
    # temp 保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(dxf_bytes)
        tmp.flush()
        path = tmp.name

    # 1) 中心線（未スケール）— read_centerline は分断もまとめてくれる前提
    cl_raw = read_centerline(path, allow_layers=[cl_layer], unit_scale=1.0)
    if cl_raw is None or len(cl_raw) < 2:
        # レイヤ誤りの可能性 → 全レイヤ探索
        cl_raw = read_centerline(path, allow_layers=None, unit_scale=1.0)
        if cl_raw is None or len(cl_raw) < 2:
            raise RuntimeError("中心線を検出できませんでした。レイヤ選択をご確認ください。")

    # 2) ラベル／円の抽出（未スケール座標）
    labels = extract_no_labels(path, label_layers, unit_scale=1.0) if label_layers else []
    circles = extract_circles(path, circle_layers, unit_scale=1.0) if circle_layers else []

    # 3) 駅測 → スケール推定
    #   ラベル文字列→ station_m、対応する円（近いもの）→ 円中心を CL に投影 → s
    #   2点以上あれば scale = median( Δstation / Δs )
    pairs: List[Tuple[float, np.ndarray]] = []
    for lab in labels:
        st_m = parse_station_to_m(lab["text"])
        if st_m is None:
            continue
        P_lab = np.array(lab["pos"], float)
        # 近い円を見つける（なければラベル自身を使う）
        cand = None
        bestd = 1e30
        for c in circles:
            Pc = np.array(c["center"], float)
            d = float(np.linalg.norm(P_lab - Pc))
            if d < bestd:
                bestd = d
                cand = Pc
        if cand is None or bestd > link_radius_m / 0.001:  # まだ縮尺不明→ゆるく（mm図面想定で係数）
            cand = P_lab
        s_raw, _ = project_point_to_polyline(cl_raw, cand)
        pairs.append((st_m, np.array([s_raw], float)))

    scale_source = "unknown"
    k_station = None
    if len(pairs) >= 2:
        pairs = sorted(pairs, key=lambda t: t[0])
        d_station = []
        d_s = []
        for i in range(1, len(pairs)):
            d_station.append(pairs[i][0] - pairs[i - 1][0])
            d_s.append(float(pairs[i][1][0] - pairs[i - 1][1][0]))
        k_station = robust_median_ratio(d_station, d_s)
        if k_station and k_station > 0:
            scale_source = "station"

    # 4) INSUNITS → 補助
    k_units = read_insunits_scale(path) or None
    # 5) ヒューリスティクス（幅が大きい→mm）
    #   ざっくり外接箱スパンを read_centerline の範囲から推測
    span = np.max(cl_raw, axis=0) - np.min(cl_raw, axis=0)
    k_heur = 0.001 if max(span) > 50000 else 1.0

    # 最終候補
    k = None
    if k_station:
        k = float(k_station)
        scale_source = "station"
    elif k_units:
        k = float(k_units)
        scale_source = "insunits"
    else:
        k = float(k_heur)
        scale_source = "heuristics"

    cl_scaled = cl_raw * k

    # No テーブル（s[m] を確定）
    rows = []
    for lab in labels:
        st_m = parse_station_to_m(lab["text"])
        if st_m is None:
            continue
        # 近い円（縮尺適用して距離判定）
        P_lab_s = np.array(lab["pos"], float) * k
        best = None
        bestd = 1e30
        for c in circles:
            Pc_s = np.array(c["center"], float) * k
            d = float(np.linalg.norm(P_lab_s - Pc_s))
            if d < bestd:
                bestd = d
                best = Pc_s
        src = best if (best is not None and bestd <= link_radius_m) else P_lab_s
        s_now, dist = project_point_to_polyline(cl_scaled, src)
        rows.append(
            {"label": lab["text"], "station_m": float(st_m), "s": float(s_now), "dist": float(dist)}
        )
    rows = sorted(rows, key=lambda r: r["station_m"])

    return PlanAnalysis(
        scale_suggest=float(k),
        scale_source=scale_source,
        cl_raw=cl_raw,
        cl_scaled=cl_scaled,
        no_rows=rows,
        summary={
            "cl_len_m": polyline_length(cl_scaled),
            "n_labels": float(len(labels)),
            "n_circles": float(len(circles)),
            "pairable": float(len(pairs)),
        },
    )


# ──────────────────────────────────────────────────────────────
# 3) 3D 変換のヘルパ
def _tangent_normal(cl: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    segs = np.diff(cl, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, "right") - 1)
    i = max(0, min(i, len(segs) - 1))
    Li = lens[i] if lens[i] > 0 else 1.0
    tau = (s - cum[i]) / Li
    P = cl[i] + tau * segs[i]
    t = segs[i] / Li
    n = np.array([-t[1], t[0]])
    return P, t, n


def _to_world(P: np.ndarray, n: np.ndarray, oz: np.ndarray, z_scale: float = 1.0):
    X = P[0] + oz[:, 0] * n[0]
    Y = P[1] + oz[:, 0] * n[1]
    Z = oz[:, 1] * z_scale
    return X, Y, Z


# ──────────────────────────────────────────────────────────────
# 4) 画面
def page():
    st.title("DXF取り込み（最小ワークフロー）")

    # =========================
    # Step 1: 平面
    # =========================
    with st.expander("Step 1｜平面（中心線＋Noラベル＋測点円）", expanded=True):
        plan = st.file_uploader("平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False)
        if plan is not None:
            st.session_state["_plan_bytes"] = bytes(plan.getbuffer())
            st.success("DXF を受け取りました。下でレイヤを選び、［解析］を押してください。")

        if "_plan_bytes" in st.session_state:
            # レイヤ一覧
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(st.session_state["_plan_bytes"])
                tmp.flush()
                path = tmp.name
            try:
                layers = list_layers(path)
                names = [L.name for L in layers]
            except Exception:
                layers = []
                names = []

            cl_layer = st.selectbox("中心線レイヤ", names)
            lbl_layers = st.multiselect("Noラベル（TEXT/MTEXT）", names)
            cir_layers = st.multiselect("測点円（CIRCLE）", names)

            col = st.columns([1, 1, 2])
            with col[0]:
                link_r = st.number_input("ラベル↔円 距離しきい[m]", value=12.0, step=1.0, format="%.1f")
            with col[1]:
                btn = st.button("解析（縮尺を推定）", type="primary")

            if btn:
                try:
                    pa = analyze_plan(
                        dxf_bytes=st.session_state["_plan_bytes"],
                        cl_layer=cl_layer,
                        label_layers=lbl_layers,
                        circle_layers=cir_layers,
                        link_radius_m=float(link_r),
                    )
                    st.session_state["_plan_analysis"] = pa
                    st.success("解析完了：縮尺を推定しました。下の確認カードで確定してください。")
                except Exception as e:
                    st.error(f"解析に失敗しました: {e}")

            pa: Optional[PlanAnalysis] = st.session_state.get("_plan_analysis")
            if pa:
                # 確認カード
                src_jp = {"station": "駅測校正", "insunits": "INSUNITS", "heuristics": "ヒューリスティクス"}.get(
                    pa.scale_source, pa.scale_source
                )
                st.markdown(
                    f"""
**推定縮尺**: `{pa.scale_suggest:g}`（出所: **{src_jp}**）  
**中心線長**: {pa.summary['cl_len_m']:.1f} m / **ラベル**: {int(pa.summary['n_labels'])} / **円**: {int(pa.summary['n_circles'])} / **駅測に使えたペア**: {int(pa.summary['pairable'])}
                    """
                )
                col2 = st.columns([1, 1, 2])
                with col2[0]:
                    if st.button("この縮尺を使う", type="primary"):
                        st.session_state["plan_scale"] = float(pa.scale_suggest)
                        st.session_state["centerline"] = pa.cl_scaled.copy().astype(np.float32)
                        # No→s テーブル（station をキーに）
                        st.session_state["no_table"] = [
                            {"no": r["label"], "station_m": r["station_m"], "s": r["s"]} for r in pa.no_rows
                        ]
                        st.success("縮尺を確定。Step 2 へ進めます。")
                with col2[1]:
                    manual_scale = st.number_input("（任意）縮尺を手で上書き", value=float(pa.scale_suggest), step=0.001)
                    if st.button("この値で確定"):
                        k = float(manual_scale)
                        st.session_state["plan_scale"] = k
                        st.session_state["centerline"] = (pa.cl_raw * k).astype(np.float32)
                        # s はスケールに依存しない（距離定義が変わるので再投影したいが簡素化のためそのまま利用）
                        st.session_state["no_table"] = [
                            {"no": r["label"], "station_m": r["station_m"], "s": r["s"] * (k / pa.scale_suggest)}
                            for r in pa.no_rows
                        ]
                        st.success("縮尺を上書き確定。Step 2 へ進めます。")

                # 簡単な表
                if pa.no_rows:
                    df = pd.DataFrame(pa.no_rows)
                    st.dataframe(df, use_container_width=True, height=220)

    # =========================
    # Step 2: 横断
    # =========================
    with st.expander("Step 2｜横断（DXF/CSV 複数）→ No 割当", expanded=True):
        if "plan_scale" not in st.session_state or "centerline" not in st.session_state:
            st.info("まず Step 1 で縮尺を確定してください。")
        else:
            xs_files = st.file_uploader("横断ファイル（複数可）", type=["dxf", "csv"], accept_multiple_files=True, key="xs")
            if xs_files:
                st.session_state.setdefault("_sections", {})
                for f in xs_files:
                    with st.expander(f.name, expanded=False):
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + f.name.split(".")[-1])
                        tmp.write(f.getbuffer())
                        tmp.flush()
                        tmp.close()
                        path = tmp.name

                        # 既定：Plan と同じ縮尺
                        k = float(st.session_state["plan_scale"])
                        # INSUNITS が Plan と違うか確認（参考）
                        k_units = read_insunits_scale(path)
                        if k_units and abs(k_units - k) > 1e-9:
                            st.warning(f"INSUNITS 推定 {k_units:g}（Plan={k:g}）。必要なら下で上書きしてください。")
                        k_over = st.number_input("この横断の縮尺（未入力なら Plan と同じ）", value=float(k))

                        layer = None
                        layer_names = []
                        if f.name.lower().endswith(".dxf"):
                            try:
                                scan = list_layers(path)
                                layer_names = [L.name for L in scan]
                                layer = st.selectbox("横断レイヤ（任意。未選択=自動）", ["（未選択）"] + layer_names)
                                if layer == "（未選択）":
                                    layer = None
                            except Exception:
                                pass

                        sec = read_single_section_file(
                            path,
                            layer_name=layer,
                            unit_scale=float(k_over),  # ここで縮尺を掛けて読み取る
                            aggregate="median",
                            smooth_k=0,
                            max_slope=0.0,
                            target_step=0.20,
                        )
                        if sec is None or len(sec) < 2:
                            st.error("断面を抽出できませんでした。レイヤや縮尺をご確認ください。")
                            continue

                        # 軸は固定（X=offset / Y=elev）
                        oz = np.column_stack([sec[:, 0], sec[:, 1]]).astype(np.float32)

                        # 中央値基準でオフセット0合わせ（シンプル）
                        o_med = float(np.median(oz[:, 0]))
                        oz[:, 0] -= o_med

                        # No 候補
                        no_choices = [r["no"] for r in (st.session_state.get("no_table") or [])]
                        sel = st.selectbox("割当 No.", ["（未選択）"] + no_choices)
                        if sel != "（未選択）":
                            st.session_state["_sections"][f.name] = {"no": sel, "oz": oz}

                        # 2D 確認
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=oz[:, 0], y=oz[:, 1], mode="lines", name="断面", line=dict(width=3)))
                        fig2.update_layout(
                            height=240,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="offset [m]",
                            yaxis_title="elev [m]",
                            paper_bgcolor="#0f1115",
                            plot_bgcolor="#0f1115",
                        )
                        fig2.update_xaxes(gridcolor="#2a2f3a")
                        fig2.update_yaxes(gridcolor="#2a2f3a")
                        st.plotly_chart(fig2, use_container_width=True)

            st.info(f"割当済み断面：{len(st.session_state.get('_sections', {}))}")

    # =========================
    # Step 3: 3D プレビュー
    # =========================
    with st.expander("Step 3｜3D プレビュー（確認）", expanded=True):
        cl = st.session_state.get("centerline")
        secs = st.session_state.get("_sections", {})
        if cl is None or not secs:
            st.info("Step 1 で縮尺確定、Step 2 で No 割当を行ってください。")
            return

        # No -> s を作る
        s_map: Dict[str, float] = {}
        for r in st.session_state.get("no_table", []):
            s_map[r["no"]] = float(r["s"])

        # 表示選択
        keys = list(secs.keys())
        show = st.multiselect("表示する断面", keys, default=keys)

        z_scale = st.number_input("縦倍率", value=1.0, step=0.1)

        fig = go.Figure()
        # 中心線
        fig.add_trace(
            go.Scatter3d(
                x=cl[:, 0],
                y=cl[:, 1],
                z=np.zeros(len(cl)),
                mode="lines",
                name="Centerline",
                line=dict(width=4, color="#A0A6B3"),
            )
        )

        for k in show:
            rec = secs[k]
            no = rec["no"]
            s = s_map.get(no)
            if s is None:
                continue
            P, t, n = _tangent_normal(cl, s)
            X, Y, Z = _to_world(P, n, rec["oz"], z_scale=float(z_scale))
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="lines",
                    name=no,
                    line=dict(width=5, color="#FFFFFF"),
                )
            )

        fig.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
            paper_bgcolor="#0f1115",
            plot_bgcolor="#0f1115",
            margin=dict(l=0, r=0, t=0, b=0),
            height=760,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 簡易 Sanity Check
        cl_len = polyline_length(cl)
        widths = []
        for rec in secs.values():
            w = float(np.max(rec["oz"][:, 0]) - np.min(rec["oz"][:, 0]))
            widths.append(w)
        if widths:
            w95 = np.percentile(widths, 95)
            if cl_len < 3 * w95:
                st.warning(f"中心線が短すぎる可能性（CL長={cl_len:.1f} m / 断面幅95%={w95:.1f} m）→ レイヤ選択や縮尺をご確認ください。")
