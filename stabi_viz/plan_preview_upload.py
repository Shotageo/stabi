# stabi_viz/plan_preview_upload.py
from __future__ import annotations

import tempfile
from typing import Dict, Optional, List

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────
# LEM が未実装でも動くフォールバック
try:
    from stabi_core.stabi_lem import compute_min_circle  # type: ignore
    _LEM_OK = True
except Exception:
    _LEM_OK = False

    def compute_min_circle(cfg):
        """お試し用の簡易円（視覚確認目的）"""
        oz = np.asarray(cfg.get("section"))
        if oz is None or oz.size == 0:
            return {"fs": 1.10, "circle": {"oc": 0.0, "zc": 0.0, "R": 10.0}}
        oc = float(np.median(oz[:, 0]))
        zc = float(np.percentile(oz[:, 1], 25))
        R = float(max(6.0, (np.max(oz[:, 0]) - np.min(oz[:, 0])) * 0.35))
        return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R}}


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
    detect_section_centerline_u,   # 横断内CL縦線の検出
)

# ──────────────────────────────────────────────────────────────
# 内部ヘルパ（進行方向に対して左法線が + ）
def _tangent_normal(centerline: np.ndarray, s: float):
    """
    中心線 polyline と弧長 s から、投影点 P、接線 t（単位ベクトル）、左法線 n を返す。
    頂点に丸めず、区間内で線形内挿。
    """
    if centerline.shape[0] < 2:
        return centerline[0], np.array([1.0, 0.0]), np.array([0.0, 1.0])

    segs = np.diff(centerline, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.r_[0.0, np.cumsum(lens)]
    Ltot = float(cum[-1])
    if Ltot <= 0:
        return centerline[0], np.array([1.0, 0.0]), np.array([0.0, 1.0])

    s = float(np.clip(s, 0.0, Ltot))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(segs) - 1))
    Li = lens[i] if lens[i] > 0 else 1.0
    tau = (s - cum[i]) / Li
    p0 = centerline[i]
    v = segs[i]

    P = p0 + tau * v
    t = v / Li
    n = np.array([-t[1], t[0]])  # 左法線を正
    return P, t, n


def _xs_to_world3D(P: np.ndarray, n: np.ndarray, oz: np.ndarray, z_scale: float = 1.0):
    """(offset, elev) → 世界座標（XYは法線方向、Zはそのまま）"""
    X = P[0] + oz[:, 0] * n[0]
    Y = P[1] + oz[:, 0] * n[1]
    Z = oz[:, 1] * float(z_scale)
    return X, Y, Z


# ──────────────────────────────────────────────────────────────
# 軽量化用：3D間引き
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
# UI の現在値を session_state に同期
def _sync_ui_value(key: str, value):
    st.session_state[key] = value
    return value


# ──────────────────────────────────────────────────────────────
# 縦断（CL標高）: s→z を取得（線形補間、なければ None）
def _profile_eval(s: float) -> Optional[float]:
    prof = st.session_state.get("profile_s")  # ndarray (N,2) : [s, z]
    if prof is None or len(prof) < 2:
        return None
    ss = prof[:, 0]; zz = prof[:, 1]
    s = float(s)
    # 端はクランプ
    if s <= ss[0]:
        return float(zz[0])
    if s >= ss[-1]:
        return float(zz[-1])
    return float(np.interp(s, ss, zz))


# s（中心線距離）を “生データから” 再計算（倍率変更後でも破綻しない）
def _rebuild_s_map_from_raw() -> Dict[str, float]:
    """
    返り値: key(No.) -> s（現在の平面倍率を掛けたスケールでの距離）
    円が無い場合はラベル位置→中心線の射影でフォールバック。
    """
    s_map: Dict[str, float] = {}
    if "centerline_raw" not in st.session_state or "no_table" not in st.session_state:
        return s_map

    cl_raw = st.session_state.centerline_raw
    k = float(st.session_state.get("unit_scale_plan_ui", 1.0))

    # ラベルの生座標（フォールバック用）
    label_pos: Dict[str, np.ndarray] = {}
    for lab in st.session_state.get("labels_raw", []):
        key = lab.get("key")
        pos = np.array(lab.get("pos"), float)
        if key is not None:
            label_pos[key] = pos

    # circle_xy は “生単位” を保存済み（Step1）
    for row in st.session_state.no_table:
        key = row["key"]
        circ_raw = np.array(row["circle_xy"], float) if row.get("circle_xy") is not None else None
        src = circ_raw if circ_raw is not None else label_pos.get(key)
        if src is None:
            s_map[key] = float(row.get("s", 0.0)) * k
            continue
        s_raw, _ = project_point_to_polyline(cl_raw, src)
        s_map[key] = float(s_raw) * k
    return s_map


# ──────────────────────────────────────────────────────────────
# 生データ raw_sections と 現在の UI 設定から assigned を再構築
def build_assigned_from_raw():
    if "raw_sections" not in st.session_state:
        return
    if "no_table" not in st.session_state:
        return
    if "centerline_raw" not in st.session_state:
        return

    # 平面（中心線）：UI倍率適用版
    unit_scale_plan = float(st.session_state.get("unit_scale_plan_ui", 1.0))
    cl_raw = st.session_state.centerline_raw
    cl = cl_raw * unit_scale_plan

    # s を現在倍率で再計算
    s_map = _rebuild_s_map_from_raw()
    delta_s = float(st.session_state.get("delta_s_all_ui", 0.0))
    for k in list(s_map.keys()):
        s_map[k] += delta_s

    # No→測点円中心（生座標）
    no_to_circle_raw = {
        d["key"]: (np.array(d["circle_xy"]) if d.get("circle_xy") is not None else None)
        for d in st.session_state.no_table
    }

    # UI パラメータ（横・縦）
    offset_scale = float(st.session_state.get("offset_scale_ui", 1.0))
    elev_scale = float(st.session_state.get("elev_scale_ui", 1.0))
    center_by_section_cl = bool(st.session_state.get("center_by_section_cl_ui", True))  # 横方向の原点：CL縦線優先
    center_by_circle = bool(st.session_state.get("center_by_circle_ui", False))
    center_o = bool(st.session_state.get("center_o_ui", False))
    user_center_offset = float(st.session_state.get("user_center_offset_ui", 0.0))
    flip_o = bool(st.session_state.get("flip_o_ui", False))
    flip_z = bool(st.session_state.get("flip_z_ui", False))

    # 高さ合わせ（新）
    z_anchor_mode = st.session_state.get("z_anchor_mode_ui", "横断CLを0に（相対）")

    assigned: Dict[str, Dict] = {}

    for fname, rec in st.session_state.raw_sections.items():
        sel = rec.get("no_key") or rec.get("guess_no")
        if not sel or sel not in s_map:
            continue

        oz_raw = np.asarray(rec["oz_raw"], float)
        if oz_raw.ndim != 2 or oz_raw.shape[1] < 2:
            continue

        # 倍率・反転
        o = oz_raw[:, 0] * offset_scale
        z = oz_raw[:, 1] * elev_scale
        if flip_o:
            o *= -1.0
        if flip_z:
            z *= -1.0

        # ── 横方向の原点合わせ（優先順位：CL縦線 → 円中心 → 中央値 → 手動）
        s = float(s_map[sel])
        P, _, n = _tangent_normal(cl, s)

        if center_by_section_cl and (rec.get("o0_from_section") is not None):
            o0 = float(rec["o0_from_section"]) * offset_scale
            o = o - o0
        else:
            circ_raw = no_to_circle_raw.get(sel)
            if center_by_circle and circ_raw is not None:
                circ_scaled = circ_raw * unit_scale_plan
                oc0 = float(np.dot(circ_scaled - P, n))
                o = o - oc0
            elif center_o:
                o = o - float(np.median(o))
            else:
                o = o - float(user_center_offset)

        # ── 縦方向（高さ）の合わせ方
        # 0点（offset=0）での高さ z0 を補間で求める
        idx = np.argsort(o)
        oo = o[idx]; zz = z[idx]
        # np.interp は範囲外は端値でクランプ
        z0 = float(np.interp(0.0, oo, zz)) if len(oo) >= 2 else float(zz[0]) if len(zz) else 0.0

        if z_anchor_mode.startswith("横断CLを0に"):
            # 相対高さ：CL高さを 0 に
            z = z - z0
        elif z_anchor_mode.startswith("縦断CSVに合わせる"):
            # 絶対高さ：縦断の CL 標高(s) に合わせて上乗せ
            base = _profile_eval(s)
            if base is None:
                # 縦断が未設定なら相対にフォールバック
                z = z - z0
            else:
                z = (z - z0) + float(base)
        elif z_anchor_mode == "最小を0（簡易）":
            z = z - float(np.min(z))
        elif z_anchor_mode == "中央値を0（簡易）":
            z = z - float(np.median(z))
        else:
            # しない：そのまま
            pass

        assigned[fname] = {"oz": np.column_stack([o, z]), "no_key": sel, "s": s}

    # 出力反映（float32 化）
    st.session_state.centerline = cl.astype(np.float32)
    st.session_state._assigned = {
        k: {"oz": v["oz"].astype(np.float32), "no_key": v["no_key"], "s": float(v["s"])}
        for k, v in assigned.items()
    }


# ──────────────────────────────────────────────────────────────
# 本体ページ
def page():
    st.title("DXF取り込み｜No×測点円スナップ → 横断の立体配置（CLで“横”と“高さ”を合わせる）")

    # ── Step 1: 平面（中心線＋No.ラベル＋測点円）
    with st.expander("Step 1｜平面（中心線＋No.ラベル＋測点円）", expanded=True):
        plan_up = st.file_uploader("平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False, key="plan")
        unit_scale_plan = _sync_ui_value(
            "unit_scale_plan_ui",
            st.number_input("平面倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"),
        )
        find_radius = st.number_input("ラベル→円 紐付け距離しきい[m]", value=12.0, step=1.0, format="%.1f")

        if plan_up is not None and st.button("中心線＋No.＋円 抽出を実行", type="primary"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
            tmp.write(plan_up.read()); tmp.flush(); tmp.close()
            st.session_state._plan_path = tmp.name

            try:
                layers = list_layers(tmp.name)
                layer_names = [L.name for L in layers]
                idx = st.radio(
                    "中心線レイヤを選択",
                    list(range(len(layer_names))),
                    format_func=lambda i: f"{layer_names[i]}  (len≈{layers[i].length_sum:.1f})",
                )
                cl_layer = layer_names[int(idx)]
                label_layers = st.multiselect("測点ラベルレイヤ（TEXT/MTEXT）", layer_names, default=layer_names)
                circle_layers = st.multiselect("測点円レイヤ（CIRCLE）", layer_names, default=layer_names)

                cl_raw = read_centerline(tmp.name, allow_layers=[cl_layer], unit_scale=1.0)
                labels = extract_no_labels(tmp.name, label_layers, unit_scale=1.0)
                circles = extract_circles(tmp.name, circle_layers, unit_scale=1.0)

                cl_scaled = cl_raw * float(unit_scale_plan)
                no_rows = []
                for lab in labels:
                    key = lab["key"]
                    Lxy_s = np.array(lab["pos"], float) * float(unit_scale_plan)
                    # しきい内の円だけ候補にする
                    candidates = []
                    for c in circles:
                        Cxy_s = np.array(c["center"], float) * float(unit_scale_plan)
                        d = float(np.linalg.norm(Lxy_s - Cxy_s))
                        if d <= float(find_radius):
                            candidates.append((d, Cxy_s, c.get("r", None), c.get("layer", "")))
                    if candidates:
                        d, Cxy_s, r, lay = min(candidates, key=lambda t: t[0])
                        s_now, dist = project_point_to_polyline(cl_scaled, Cxy_s)
                        no_rows.append(
                            {"key": key, "s": float(s_now),
                             "label_to_circle": d, "circle_to_cl": dist,
                             "circle_r": r, "circle_layer": lay,
                             "circle_xy": tuple(Cxy_s / float(unit_scale_plan)),  # 生単位で保存
                             "status": "OK(circle)"},
                        )
                    else:
                        s_now, dist = project_point_to_polyline(cl_scaled, Lxy_s)
                        no_rows.append(
                            {"key": key, "s": float(s_now),
                             "label_to_circle": None, "circle_to_cl": dist,
                             "circle_r": None, "circle_layer": None, "circle_xy": None,
                             "status": "FALLBACK(label→CL)"},
                        )

                no_rows.sort(key=lambda d: d["s"])
                st.session_state.centerline_raw = cl_raw
                st.session_state.labels_raw = labels
                st.session_state.circles_raw = circles
                st.session_state.no_table = no_rows
                st.session_state.cl_layer = cl_layer

                st.success(f"中心線: {len(cl_raw)} 点（表示倍率 {unit_scale_plan:g}）、No.: {len(no_rows)} 件")
            except Exception as e:
                st.error(f"抽出失敗: {e}")

    # ── Step 1.5: 縦断（CL標高）オプション
    with st.expander("Step 1.5｜縦断（中心線の標高）を設定（任意）", expanded=False):
        st.caption("CSV 形式の縦断（列名: s, z）。単位は平面の s と同じ [m] を想定。")
        prof_up = st.file_uploader("縦断CSV（s[m], z[m]）", type=["csv"], accept_multiple_files=False, key="prof")
        s_scale = st.number_input("s の倍率（例：mm→m は 0.001）", value=1.0, step=0.001, format="%.3f")
        z_scale = st.number_input("z の倍率（例：mm→m は 0.001）", value=1.0, step=0.001, format="%.3f")

        if prof_up is not None and st.button("縦断CSVを読み込み", type="primary"):
            try:
                import pandas as pd
                df = pd.read_csv(prof_up)
                # 列名解釈
                if "s" in df.columns and "z" in df.columns:
                    ss = df["s"].astype(float).to_numpy() * float(s_scale)
                    zz = df["z"].astype(float).to_numpy() * float(z_scale)
                else:
                    raise ValueError("CSV に列名 's' と 'z' が必要です。")
                order = np.argsort(ss)
                prof = np.column_stack([ss[order], zz[order]])
                st.session_state.profile_s = prof.astype(np.float32)
                st.success(f"縦断を読み込みました（点数: {len(prof)}）")
            except Exception as e:
                st.error(f"縦断の読み込み失敗: {e}")

        # プレビュー
        if st.session_state.get("profile_s") is not None:
            prof = st.session_state.profile_s
            st.line_chart({"z": prof[:, 1]}, height=140)

    # ── Step 2: 横断読み込み・集約・No割当（CL縦線=0）
    with st.expander("Step 2｜横断を読み込み→集約→No割当（横と高さの基準合わせ・再アップ不要）", expanded=True):
        xs_files = st.file_uploader("横断DXF/CSV（複数可）", type=["dxf", "csv"], accept_multiple_files=True, key="xs")

        # 追加の軽量化スイッチ
        reparse_all = st.checkbox("取り込み済みでも再解析する（重い）", value=False)
        show_2d_preview = st.checkbox("2Dプレビューも描画する（重い時はOFF）", value=False)

        # 軸と倍率・反転（横）
        axis_mode = st.selectbox("軸割り", ["X=offset / Y=elev（標準）", "X=elev / Y=offset（入替）"])
        offset_scale = _sync_ui_value("offset_scale_ui",
                                      st.number_input("オフセット倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"))
        elev_scale = _sync_ui_value("elev_scale_ui",
                                    st.number_input("標高倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"))
        flip_o = _sync_ui_value("flip_o_ui", st.checkbox("オフセット左右反転", value=False))
        flip_z = _sync_ui_value("flip_z_ui", st.checkbox("標高上下反転", value=False))

        # 集約とリサンプリング
        agg_mode = st.selectbox("複数線の集約", ["中央値（推奨）", "下包絡（最小）", "上包絡（最大）"])
        smooth_k = st.slider("平滑ウィンドウ（奇数、0で無効）", 0, 21, 0, step=1)
        max_slope = st.slider("最大許容勾配 |dz/dx|（0で無効）", 0.0, 30.0, 0.0, step=0.5)
        target_step = st.number_input("出力間隔 step [m]（小さいほど精細）", value=0.50, step=0.05, format="%.2f")

        # 横方向の中心合わせ（優先順位に関わるスイッチ）
        center_by_section_cl = _sync_ui_value(
            "center_by_section_cl_ui", st.checkbox("横断内のCL縦線を 0 に（自動・推奨）", value=True)
        )
        center_by_circle = _sync_ui_value(
            "center_by_circle_ui", st.checkbox("道路中心オフセット値を0に（円中心=0, 平面併用）", value=False)
        )
        center_o = _sync_ui_value("center_o_ui", st.checkbox("オフセット中央値を0に（フォールバック）", value=False))
        user_center_offset = _sync_ui_value(
            "user_center_offset_ui", st.number_input("（手動）道路中心オフセット値", value=0.0, step=0.1, format="%.3f")
        )

        # 縦方向の高さ合わせ（新）
        z_anchor_mode = _sync_ui_value(
            "z_anchor_mode_ui",
            st.selectbox(
                "高さの合わせ方（Zアンカー）",
                ["横断CLを0に（相対）", "縦断CSVに合わせる（CL基準）", "最小を0（簡易）", "中央値を0（簡易）", "しない"],
                index=0,
            ),
        )

        # 生データバッファ
        if "raw_sections" not in st.session_state:
            st.session_state.raw_sections = {}

        # No 選択候補
        no_choices = [d["key"] for d in st.session_state.get("no_table", [])]
        agg_map = {"中央値（推奨）": "median", "下包絡（最小）": "lower", "上包絡（最大）": "upper"}

        if xs_files:
            for f in xs_files:
                with st.expander(f"割当：{f.name}", expanded=False):
                    # 既に取り込み済みなら再解析をスキップ
                    existing = st.session_state.raw_sections.get(f.name)
                    if existing and not reparse_all:
                        guess = existing.get("no_key") or existing.get("guess_no") or ""
                        idx = (no_choices.index(guess) + 1) if (guess and guess in no_choices) else 0
                        sel = st.selectbox("割当No.", ["（未選択）"] + no_choices, index=idx)
                        existing["no_key"] = sel if sel != "（未選択）" else None

                        if show_2d_preview:
                            import matplotlib.pyplot as plt
                            o_raw = existing["oz_raw"][:, 0]; z_raw = existing["oz_raw"][:, 1]
                            o = o_raw * float(offset_scale); z = z_raw * float(elev_scale)
                            if st.session_state.get("flip_o_ui", False): o *= -1.0
                            if st.session_state.get("flip_z_ui", False): z *= -1.0
                            fig2, ax = plt.subplots(figsize=(5.0, 2.2))
                            ax.plot(o, z, lw=2.0); ax.grid(True, alpha=0.3)
                            ax.set_xlabel("offset [m]"); ax.set_ylabel("elev [m]")
                            st.pyplot(fig2, use_container_width=True)
                        continue

                    # ここから新規解析
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + f.name.split(".")[-1])
                    tmp.write(f.getbuffer()); tmp.flush(); tmp.close()

                    layer_name: Optional[str] = None
                    cl_hint_layers: Optional[List[str]] = None
                    if f.name.lower().endswith(".dxf"):
                        try:
                            scan_layers = list_layers(tmp.name)
                            layer_name = st.selectbox(
                                "横断レイヤ（任意／未選択=自動）",
                                ["（未選択）"] + [L.name for L in scan_layers]
                            )
                            if layer_name == "（未選択）":
                                layer_name = None
                            # CL縦線のレイヤヒント（任意）
                            cl_hint_layers = st.multiselect(
                                "CL縦線レイヤ（任意：分かっていれば絞り込み）",
                                [L.name for L in scan_layers],
                                default=[]
                            ) or None
                        except Exception:
                            pass

                    sec = read_single_section_file(
                        tmp.name,
                        layer_name=layer_name,
                        unit_scale=1.0,  # 生で保持
                        aggregate=agg_map[agg_mode],
                        smooth_k=int(smooth_k),
                        max_slope=float(max_slope),
                        target_step=float(target_step),
                    )
                    if sec is None:
                        st.warning("断面を認識できませんでした。レイヤや倍率をご確認ください。")
                        continue

                    # 軸入替だけ先に済ませて「生」を保存
                    if axis_mode.startswith("X=elev"):
                        o_raw = sec[:, 1].astype(float); z_raw = sec[:, 0].astype(float)
                    else:
                        o_raw = sec[:, 0].astype(float); z_raw = sec[:, 1].astype(float)
                    oz_raw = np.column_stack([o_raw, z_raw])

                    # CL縦線の自動検出（u0）→ 生のまま記録
                    u0 = None
                    try:
                        u0 = detect_section_centerline_u(tmp.name, layer_hint=cl_hint_layers, unit_scale=1.0)
                    except Exception:
                        u0 = None

                    # No 推定（ファイル名から）
                    guess = normalize_no_key(f.name) or ""
                    guess_idx = no_choices.index(guess) + 1 if no_choices and guess in no_choices else 0
                    sel = st.selectbox("割当No.", ["（未選択）"] + no_choices, index=guess_idx)

                    # 生データ保存
                    st.session_state.raw_sections[f.name] = {
                        "oz_raw": oz_raw,
                        "guess_no": sel if sel != "（未選択）" else None,
                        "no_key": sel if sel != "（未選択）" else None,
                        "o0_from_section": u0,  # CL縦線の横位置
                    }

                    if u0 is not None:
                        st.caption(f"CL縦線を検出：u0 ≈ {u0:.3f}（横断の原点として採用します）")
                    else:
                        st.caption("CL縦線は見つかりませんでした → 円中心/中央値/手動で合わせます。")

                    if show_2d_preview:
                        import matplotlib.pyplot as plt
                        o = o_raw * float(offset_scale); z = z_raw * float(elev_scale)
                        if st.session_state.get("flip_o_ui", False): o *= -1.0
                        if st.session_state.get("flip_z_ui", False): z *= -1.0
                        fig2, ax = plt.subplots(figsize=(5.0, 2.4))
                        ax.plot(o, z, lw=2.0); ax.grid(True, alpha=0.3)
                        ax.set_xlabel("offset [m]"); ax.set_ylabel("elev [m]")
                        if u0 is not None:
                            ax.axvline(u0 * float(offset_scale), color="#ff8c00", lw=1.2, alpha=0.7)
                        st.pyplot(fig2, use_container_width=True)

        # 「変更を適用（再計算）」で生→現在UIを一括適用して割当辞書を再構築
        if st.button("変更を適用（再計算）", type="primary"):
            try:
                build_assigned_from_raw()
                st.success("横方向（CL）と縦方向（高さ）の基準を再適用しました。再アップロードは不要です。")
            except Exception as e:
                st.error(f"再適用でエラー: {e}")

        assigned_cnt = len(st.session_state.get("_assigned", {}))
        raw_cnt = len(st.session_state.get("raw_sections", {}))
        st.info(f"割当済み：{assigned_cnt} / 取り込み済みファイル：{raw_cnt}")

    # ── Step 3: 3D プレビュー（立体配置）
    with st.expander("Step 3｜3Dプレビュー（立体配置）", expanded=True):
        can_run = ("centerline" in st.session_state) and ("_assigned" in st.session_state) and st.session_state._assigned
        if not can_run:
            st.warning("中心線＋No. と 横断の割当（再適用）を完了してください。")
            return

        cl = st.session_state.centerline
        assigned = st.session_state._assigned

        # 3D 負荷制御
        z_scale = st.number_input("縦倍率（標高）", value=1.0, step=0.1, format="%.1f")
        _ = st.number_input("表示ピッチ（情報用・今は配置に影響しません）", value=20.0, step=1.0, format="%.1f")
        max_pts_cl = st.number_input("中心線の最大点数（3D間引き）", min_value=500, max_value=20000, value=4000, step=500)
        max_pts_xs = st.number_input("断面1本あたりの最大点数（3D間引き）", min_value=200, max_value=10000, value=1200, step=100)
        show_arcs = st.checkbox("最小円（LEM円弧）を表示する", value=False)
        delta_s = _sync_ui_value("delta_s_all_ui", st.number_input("Δs 全体 [m]（微調整、±2m 程度）", value=0.0, step=0.1, format="%.1f"))

        fig = go.Figure()

        # 中心線（間引き）
        cl_plot = _decimate1d(cl, int(max_pts_cl))
        fig.add_trace(go.Scatter3d(
            x=cl_plot[:, 0], y=cl_plot[:, 1], z=np.zeros(len(cl_plot)),
            mode="lines", name="Centerline", line=dict(width=4, color="#A0A6B3")
        ))

        # 断面群
        for _, rec in sorted(assigned.items(), key=lambda kv: kv[1]["s"]):
            s = float(rec["s"]); oz = np.asarray(rec["oz"], float)
            P, t, n = _tangent_normal(cl, s)

            # 断面形状（間引き）
            oz_plot = _decimate(oz, int(max_pts_xs))
            X, Y, Z = _xs_to_world3D(P, n, oz_plot, z_scale=z_scale)
            fig.add_trace(go.Scatter3d(
                x=X, y=Y, z=Z, mode="lines",
                name=f"{rec['no_key']}", line=dict(width=5, color="#FFFFFF"), opacity=0.98,
            ))

            # 基線
            omin, omax = float(np.min(oz_plot[:, 0])), float(np.max(oz_plot[:, 0]))
            Xb, Yb, Zb = _xs_to_world3D(P, n, np.array([[omin, 0.0], [omax, 0.0]]), z_scale=z_scale)
            fig.add_trace(go.Scatter3d(x=Xb, y=Yb, z=Zb, mode="lines", showlegend=False, line=dict(width=2, color="#777777")))

            # 縦ポール（offset=0）
            zmin, zmax = float(np.min(oz_plot[:, 1])) * z_scale, float(np.max(oz_plot[:, 1])) * z_scale
            Xp, Yp, Zp = _xs_to_world3D(P, n, np.array([[0.0, zmin], [0.0, zmax]]), z_scale=1.0)
            fig.add_trace(go.Scatter3d(x=Xp, y=Yp, z=Zp, mode="lines", showlegend=False, line=dict(width=3, color="#8888FF")))

            # 円弧（LEM）※重いのでトグル
            if show_arcs:
                try:
                    res = compute_min_circle({"section": oz})
                    oc, zc, R = res["circle"]["oc"], res["circle"]["zc"], res["circle"]["R"]
                    ph = np.linspace(-np.pi, np.pi, 241)
                    xo = oc + R * np.cos(ph); zo = zc + R * np.sin(ph)
                    X2 = P[0] + xo * n[0]; Y2 = P[1] + xo * n[1]; Z2 = zo * z_scale
                    fig.add_trace(go.Scatter3d(x=X2, y=Y2, z=Z2, mode="lines", showlegend=False, line=dict(width=3, color="#E65454")))
                except Exception:
                    pass

        fig.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
            paper_bgcolor="#0f1115", plot_bgcolor="#0f1115", margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, height=780)


# 単体実行用
if __name__ == "__main__":
    page()