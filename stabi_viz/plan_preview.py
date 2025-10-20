# stabi_viz/plan_preview.py
from __future__ import annotations
import os, math
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from stabi_io.dxf_sections import list_layers, read_centerline, read_cross_sections_from_folder
from stabi_core.stabi_lem import compute_min_circle  # 既存の最小円弧APIを想定（契約スキーマ）
from plot_utils import plot_style

# --------- Utility: tangent/normal ----------
def _tangent_normal(centerline: np.ndarray, s: float):
    lens = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))]
    s = float(np.clip(s, lens[0], lens[-1]))
    i = int(np.searchsorted(lens, s))
    i0 = max(1, min(len(centerline)-1, i))
    t = centerline[i0] - centerline[i0-1]
    t = t / np.linalg.norm(t)
    n = np.array([-t[1], t[0]])
    P = centerline[i0]
    return P, t, n

def _xs_to_world3D(P, n, oz):
    X = P[0] + oz[:,0] * n[0]
    Y = P[1] + oz[:,0] * n[1]
    Z = oz[:,1]
    return X, Y, Z

# --------- Streamlit page entry ----------
def page():
    st.header("DXF取り込み・プレビュー（中心線＋横断）")

    colL, colR = st.columns([2,1])

    with colL:
        plan_path = st.text_input("平面図DXFへのパス（repo相対）", "io/plan.dxf")
        xs_folder = st.text_input("横断DXF/CSVフォルダ（repo相対）", "io/sections")

        unit_scale_plan = st.number_input("平面DXFの倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f")
        unit_scale_xs   = st.number_input("横断の倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f")

        if st.button("レイヤ一覧を読む", type="primary"):
            try:
                layers = list_layers(plan_path)
                st.session_state._stabi_layers = layers
                st.success(f"{len(layers)} レイヤを検出")
            except Exception as e:
                st.error(f"DXF読み込みに失敗: {e}")

        layers = st.session_state.get("_stabi_layers", [])
        if layers:
            st.subheader("レイヤ選択（中心線候補）")
            options = [f"{L.name}  ({sum(L.entity_counts.values())} ents, len≈{L.length_sum:.1f})" for L in layers]
            chosen = st.radio("中心線レイヤを1つ選択", options, index=0)
            center_layer = layers[options.index(chosen)].name
        else:
            center_layer = None

        if st.button("取り込み → プレビュー生成", disabled=(center_layer is None)):
            try:
                cl = read_centerline(plan_path, [center_layer], unit_scale=unit_scale_plan)
                xs = read_cross_sections_from_folder(xs_folder, unit_scale=unit_scale_xs, layer_name=None)
                st.session_state.centerline = cl
                st.session_state.sections = xs
                st.success(f"中心線: {len(cl)}点, 横断: {len(xs)}本")
            except Exception as e:
                st.error(f"取り込みに失敗: {e}")

    with colR:
        st.caption("Tips")
        st.write("- 平面DXFは **LWPOLYLINE/LINE/SPLINE** いずれもOK（最長を中心線として採用）")
        st.write("- 横断はフォルダ内の **DXF(1断面=1ファイル)** または **CSV(offset,elev)** に対応")
        st.write("- CSV/ファイル名に `100+00` / `KP12+350` を含めると距離程を自動認識")

    # --------- 3D Overview ---------
    if "centerline" in st.session_state and "sections" in st.session_state:
        cl: np.ndarray = st.session_state.centerline
        sections: Dict[float, np.ndarray] = st.session_state.sections

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=cl[:,0], y=cl[:,1], z=np.zeros(len(cl)),
                                   mode="lines", name="Centerline",
                                   line=dict(width=4, color="#A0A6B3")))
        # downsample表示ピッチ
        pitch = st.slider("表示横断ピッチ[m]", 5, 50, 20, step=5, help="描画負荷を抑えるための間引き")
        show_keys = []
        for s in sorted(sections.keys()):
            if int(s) % pitch == 0:
                show_keys.append(s)
        if not show_keys:
            show_keys = list(sections.keys())[::max(1, len(sections)//10)]

        for s in show_keys:
            P, t, n = _tangent_normal(cl, s)
            oz = sections[s]
            X,Y,Z = _xs_to_world3D(P, n, oz)
            fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines",
                                       name=f"XS {s:.0f}", line=dict(width=5, color="#FFFFFF"), opacity=0.95))
            # 既存エンジンで最小円弧を取得（契約スキーマ）
            try:
                res = compute_min_circle({"section": oz})
                c = res["circle"]; fs = res["fs"]
                # 円弧離散
                ph = np.linspace(-np.pi, np.pi, 181)
                xo = c["oc"] + c["R"]*np.cos(ph)
                zo = c["zc"] + c["R"]*np.sin(ph)
                X2 = P[0] + xo * n[0]; Y2 = P[1] + xo * n[1]; Z2 = zo
                fig.add_trace(go.Scatter3d(x=X2, y=Y2, z=Z2, mode="lines",
                                           showlegend=False, line=dict(width=3, color="#E65454")))
                fig.add_trace(go.Scatter3d(x=[P[0]], y=[P[1]], z=[float(np.max(Z)+1.5)], mode="text",
                                           text=[f"Fs={fs:.2f}"], textfont=dict(color="#FFD34D", size=12),
                                           showlegend=False))
            except Exception:
                pass

        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                     aspectmode="data"),
                          paper_bgcolor="#0f1115", plot_bgcolor="#0f1115",
                          margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True, height=720)

        st.success("俯瞰プレビューOK。クリックズーム・DXF出力は次ページ（Viewer）で実装。")
