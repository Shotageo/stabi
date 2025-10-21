# stabi_viz/plan_preview_upload.py
from __future__ import annotations

import tempfile
from typing import Dict, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────
# LEM が未実装でも動くようにフォールバック
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
)

# ──────────────────────────────────────────────────────────────
# 内部ヘルパ（進行方向に対して左法線が + ）
def _tangent_normal(centerline: np.ndarray, s: float):
    lens = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))]
    s = float(np.clip(s, lens[0], lens[-1]))
    i = int(np.searchsorted(lens, s))
    i0 = max(1, min(len(centerline) - 1, i))
    t = centerline[i0] - centerline[i0 - 1]
    if np.linalg.norm(t) == 0:
        t = np.array([1.0, 0.0])
    else:
        t = t / np.linalg.norm(t)
    # 左法線が正
    n = np.array([-t[1], t[0]])
    P = centerline[i0]
    return P, t, n


def _xs_to_world3D(P: np.ndarray, n: np.ndarray, oz: np.ndarray, z_scale: float = 1.0):
    """(offset, elev) → 世界座標（XYは法線方向、Zはそのまま）"""
    X = P[0] + oz[:, 0] * n[0]
    Y = P[1] + oz[:, 0] * n[1]
    Z = oz[:, 1] * float(z_scale)
    return X, Y, Z


# ──────────────────────────────────────────────────────────────
# UI の現在値を session_state に同期
def _sync_ui_value(key: str, value):
    st.session_state[key] = value
    return value


# ──────────────────────────────────────────────────────────────
# 生データ raw_sections と 現在の UI 設定から assigned を再構築
def build_assigned_from_raw():
    if "raw_sections" not in st.session_state:
        return
    if "no_table" not in st.session_state:
        return
    if "centerline_raw" not in st.session_state:
        return

    # 平面（中心線）：UI倍率適用版を作る
    unit_scale_plan = float(st.session_state.get("unit_scale_plan_ui", 1.0))
    cl_raw = st.session_state.centerline_raw
    cl = cl_raw * unit_scale_plan

    # No→測点 s / 測点円中心
    no_to_s = {d["key"]: d["s"] for d in st.session_state.no_table}
    no_to_circle = {
        d["key"]: (np.array(d["circle_xy"]) if d["circle_xy"] is not None else None)
        for d in st.session_state.no_table
    }

    # UI パラメータ
    offset_scale = float(st.session_state.get("offset_scale_ui", 1.0))
    elev_scale = float(st.session_state.get("elev_scale_ui", 1.0))
    center_o = bool(st.session_state.get("center_o_ui", True))
    center_by_circle = bool(st.session_state.get("center_by_circle_ui", False))
    user_center_offset = float(st.session_state.get("user_center_offset_ui", 0.0))
    elev_zero_mode = st.session_state.get("elev_zero_mode_ui", "最小を0")
    flip_o = bool(st.session_state.get("flip_o_ui", False))
    flip_z = bool(st.session_state.get("flip_z_ui", False))

    assigned: Dict[str, Dict] = {}

    for fname, rec in st.session_state.raw_sections.items():
        sel = rec.get("no_key") or rec.get("guess_no")
        if not sel or sel not in no_to_s:
            # No 未確定はスキップ（UI側で選んでから適用で反映）
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

        # センタリング
        s = float(no_to_s[sel])
        P, _, n = _tangent_normal(cl, s)
        circ = no_to_circle.get(sel)
        if center_by_circle and circ is not None:
            # UI倍率を掛けた円中心座標を使用
            circ_scaled = circ * unit_scale_plan
            oc0 = float(np.dot(circ_scaled - P, n))
            o = o - oc0
        elif center_o:
            o = o - float(np.median(o))
        else:
            o = o - float(user_center_offset)

        if elev_zero_mode == "最小を0":
            z = z - float(np.min(z))
        elif elev_zero_mode == "中央値を0":
            z = z - float(np.median(z))
        # しない → 変更なし

        assigned[fname] = {"oz": np.column_stack([o, z]), "no_key": sel, "s": s}

    # 出力を反映
    st.session_state.centerline = cl
    st.session_state._assigned = assigned


# ──────────────────────────────────────────────────────────────
# 本体ページ
def page():
    st.title("DXF取り込み（No×測点円スナップ → 横断の立体配置）")

    # ── Step 1: 平面（中心線＋No.ラベル＋測点円）
    with st.expander("Step 1｜平面（中心線＋No.ラベル＋測点円）", expanded=True):
        plan_up = st.file_uploader(
            "平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False, key="plan"
        )
        unit_scale_plan = _sync_ui_value(
            "unit_scale_plan_ui",
            st.number_input("平面倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"),
        )
        find_radius = st.number_input("ラベル→円 検索半径[m]", value=15.0, step=1.0, format="%.1f")

        if plan_up is not None and st.button("中心線＋No.＋円 抽出を実行", type="primary"):
            # 一時ファイルとして保存
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
            tmp.write(plan_up.read())
            tmp.flush()
            tmp.close()
            st.session_state._plan_path = tmp.name

            try:
                # 生のまま読み込んで保存（倍率は後段で適用）
                cl_raw = read_centerline(tmp.name, allow_layers=[], unit_scale=1.0)
                layers = list_layers(tmp.name)
                # UI でレイヤ選択させるため、一度一覧を出す
                st.info("中心線は最長形状を自動選択しました。必要ならレイヤ選択版に差し替え可能です。")

                labels = extract_no_labels(tmp.name, [L.name for L in layers], unit_scale=1.0)
                circles = extract_circles(tmp.name, [L.name for L in layers], unit_scale=1.0)

                # 円の中心を中心線上に投影して s を決定
                no_rows = []
                cl_scaled = cl_raw * float(unit_scale_plan)
                for lab in labels:
                    key = lab["key"]
                    # ラベル近傍の円（最短）を探す
                    Lxy = np.array(lab["pos"], float) * float(unit_scale_plan)
                    # 最も近い円
                    best = None
                    for c in circles:
                        Cxy = np.array(c["center"], float) * float(unit_scale_plan)
                        d = float(np.linalg.norm(Lxy - Cxy))
                        if best is None or d < best[0]:
                            best = (d, Cxy, c.get("r", None), c.get("layer", ""))
                    if best is not None:
                        d, Cxy, r, lay = best
                        s, dist = project_point_to_polyline(cl_scaled, Cxy)
                        no_rows.append(
                            {
                                "key": key,
                                "s": s,
                                "label_to_circle": d,
                                "circle_to_cl": dist,
                                "circle_r": r,
                                "circle_layer": lay,
                                "circle_xy": tuple(Cxy / float(unit_scale_plan)),  # ←生単位で保存
                                "status": "OK",
                            }
                        )

                no_rows.sort(key=lambda d: d["s"])

                # 生データとして保持（倍率は都度適用）
                st.session_state.centerline_raw = cl_raw
                st.session_state.labels_raw = labels
                st.session_state.circles_raw = circles
                st.session_state.no_table = no_rows

                st.success(
                    f"中心線: {len(cl_raw)} 点（表示倍率 {unit_scale_plan:g}）、No.: {len(no_rows)} 件"
                )
            except Exception as e:
                st.error(f"抽出失敗: {e}")

    # ── Step 2: 横断読み込み・集約・No割当
    with st.expander("Step 2｜横断を読み込み→集約→No割当（倍率変更は再アップ不要）", expanded=True):
        xs_files = st.file_uploader(
            "横断DXF/CSV（複数可）", type=["dxf", "csv"], accept_multiple_files=True, key="xs"
        )

        # 軸と倍率・反転
        axis_mode = st.selectbox("軸割り", ["X=offset / Y=elev（標準）", "X=elev / Y=offset（入替）"])
        offset_scale = _sync_ui_value(
            "offset_scale_ui",
            st.number_input("オフセット倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"),
        )
        elev_scale = _sync_ui_value(
            "elev_scale_ui",
            st.number_input("標高倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f"),
        )
        flip_o = _sync_ui_value("flip_o_ui", st.checkbox("オフセット左右反転", value=False))
        flip_z = _sync_ui_value("flip_z_ui", st.checkbox("標高上下反転", value=False))

        # 集約とリサンプリング
        agg_mode = st.selectbox("複数線の集約", ["中央値（推奨）", "下包絡（最小）", "上包絡（最大）"])
        smooth_k = st.slider("平滑ウィンドウ（奇数、0で無効）", 0, 21, 0, step=1)
        max_slope = st.slider("最大許容勾配 |dz/dx|（0で無効）", 0.0, 30.0, 0.0, step=0.5)
        target_step = st.number_input("出力間隔 step [m]（小さいほど精細）", value=0.20, step=0.05, format="%.2f")

        # センタリング・基準
        center_o = _sync_ui_value("center_o_ui", st.checkbox("オフセット中央値を0に", value=True))
        center_by_circle = _sync_ui_value(
            "center_by_circle_ui",
            st.checkbox("道路中心オフセット値を0に（円中心=0, 自動）", value=False),
        )
        user_center_offset = _sync_ui_value(
            "user_center_offset_ui",
            st.number_input("（手動）道路中心オフセット値", value=0.0, step=0.1, format="%.3f"),
        )
        elev_zero_mode = _sync_ui_value(
            "elev_zero_mode_ui",
            st.selectbox("標高の基準シフト", ["しない", "最小を0", "中央値を0"], index=1),
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
                    # ファイルを一旦解析して「素の offset/elev」を保存（軸入替のみ／倍率やセンタリングは未適用）
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + f.name.split(".")[-1])
                    tmp.write(f.getbuffer())
                    tmp.flush()
                    tmp.close()

                    layer_name: Optional[str] = None
                    if f.name.lower().endswith(".dxf"):
                        # 参考用にレイヤ一覧（選択しない = 自動）
                        try:
                            scan_layers = list_layers(tmp.name)
                            layer_name = st.selectbox(
                                "横断レイヤ（任意／未選択=自動）",
                                ["（未選択）"] + [L.name for L in scan_layers],
                            )
                            if layer_name == "（未選択）":
                                layer_name = None
                        except Exception:
                            pass

                    sec = read_single_section_file(
                        tmp.name,
                        layer_name=layer_name,
                        unit_scale=1.0,  # 「生」で保持
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
                        o_raw = sec[:, 1].astype(float)
                        z_raw = sec[:, 0].astype(float)
                    else:
                        o_raw = sec[:, 0].astype(float)
                        z_raw = sec[:, 1].astype(float)
                    oz_raw = np.column_stack([o_raw, z_raw])

                    # No 推定（ファイル名から）
                    guess = normalize_no_key(f.name) or ""
                    if no_choices and guess in no_choices:
                        guess_idx = no_choices.index(guess) + 1
                    else:
                        guess_idx = 0
                    sel = st.selectbox("割当No.", ["（未選択）"] + no_choices, index=guess_idx)

                    # 生データ保存／No だけ持つ
                    st.session_state.raw_sections[f.name] = {
                        "oz_raw": oz_raw,
                        "guess_no": sel if sel != "（未選択）" else None,
                        "no_key": sel if sel != "（未選択）" else None,
                    }

                    # 断面プレビュー（生→現在UIで適用する簡易可視化）
                    # ※ 見やすさ重視で 2D プロットにしておく
                    import matplotlib.pyplot as plt

                    # 現在のUI適用（センタリングは無し、倍率と反転のみ）
                    o = o_raw * float(offset_scale)
                    z = z_raw * float(elev_scale)
                    if flip_o:
                        o *= -1.0
                    if flip_z:
                        z *= -1.0

                    fig2, ax = plt.subplots(figsize=(5.0, 2.4))
                    ax.plot(o, z, lw=2.0)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("offset [m]")
                    ax.set_ylabel("elev [m]")
                    st.pyplot(fig2, use_container_width=True)

        # 「変更を適用（再計算）」で生→現在UIを一括適用して割当辞書を再構築
        if st.button("変更を適用（再計算）", type="primary"):
            try:
                build_assigned_from_raw()
                st.success("倍率・反転・センタリングを再適用しました。再アップロードは不要です。")
            except Exception as e:
                st.error(f"再適用でエラー: {e}")

        # 状況表示
        assigned_cnt = len(st.session_state.get("_assigned", {}))
        raw_cnt = len(st.session_state.get("raw_sections", {}))
        st.info(f"割当済み：{assigned_cnt} / 取り込み済みファイル：{raw_cnt}")

    # ── Step 3: 3D プレビュー（立体配置）
    with st.expander("Step 3｜3Dプレビュー（立体配置）", expanded=True):
        can_run = (
            ("centerline" in st.session_state)
            and ("_assigned" in st.session_state)
            and st.session_state._assigned
        )
        if not can_run:
            st.warning("中心線＋No. と 横断の割当（再適用）を完了してください。")
            return

        cl = st.session_state.centerline
        assigned = st.session_state._assigned

        z_scale = st.number_input("縦倍率（標高）", value=1.0, step=0.1, format="%.1f")
        _ = st.number_input("表示ピッチ（情報用・今は配置に影響しません）", value=20.0, step=1.0, format="%.1f")

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

        # 断面群
        for _, rec in sorted(assigned.items(), key=lambda kv: kv[1]["s"]):
            s = float(rec["s"])
            oz = np.asarray(rec["oz"], float)
            P, t, n = _tangent_normal(cl, s)

            # 断面形状本体
            X, Y, Z = _xs_to_world3D(P, n, oz, z_scale=z_scale)
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="lines",
                    name=f"{rec['no_key']}",
                    line=dict(width=5, color="#FFFFFF"),
                    opacity=0.98,
                )
            )

            # 基線（offset 範囲）
            omin, omax = float(np.min(oz[:, 0])), float(np.max(oz[:, 0]))
            xb = np.array([omin, omax])
            yb = np.zeros_like(xb)
            Xb, Yb, Zb = _xs_to_world3D(P, n, np.column_stack([xb, yb]), z_scale=z_scale)
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

            # 縦ポール（offset=0）
            zmin, zmax = float(np.min(oz[:, 1])) * z_scale, float(np.max(oz[:, 1])) * z_scale
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

            # 円弧（LEMの最小円）
            try:
                res = compute_min_circle({"section": oz})
                oc, zc, R = res["circle"]["oc"], res["circle"]["zc"], res["circle"]["R"]
                ph = np.linspace(-np.pi, np.pi, 241)
                xo = oc + R * np.cos(ph)
                zo = zc + R * np.sin(ph)
                X2 = P[0] + xo * n[0]
                Y2 = P[1] + xo * n[1]
                Z2 = zo * z_scale
                fig.add_trace(
                    go.Scatter3d(
                        x=X2, y=Y2, z=Z2, mode="lines", showlegend=False, line=dict(width=3, color="#E65454")
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
        st.plotly_chart(fig, use_container_width=True, height=780)


# 単体実行用
if __name__ == "__main__":
    page()
