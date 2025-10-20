# stabi_viz/plan_preview.py
from __future__ import annotations
import os, math
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Try to import the real LEM. Fallback to a mock if unavailable.
try:
    from stabi_core.stabi_lem import compute_min_circle  # expected real API
    _LEM_OK = True
except Exception:
    _LEM_OK = False
    def compute_min_circle(cfg):
        """Fallback mock: returns a simple circle & Fs so the page works even without LEM."""
        oz = np.asarray(cfg.get("section"))
        if oz is None or oz.size == 0:
            return {"fs": 1.10, "circle": {"oc": 0.0, "zc": 0.0, "R": 10.0, "x1": -5.0, "x2": 5.0}}
        oc = float(np.median(oz[:,0]))
        zc = float(np.min(oz[:,1]) + (np.max(oz[:,1]) - np.min(oz[:,1]))*0.25)
        R  = float(max(6.0, (np.max(oz[:,0]) - np.min(oz[:,0]))*0.35))
        return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R, "x1": oc-R*0.6, "x2": oc+R*0.6}}

from stabi_io.dxf_sections import list_layers, read_centerline, read_cross_sections_from_folder

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
    st.title("DXF取り込み・プレビュー")
    st.caption("平面の中心線レイヤを選んで、横断（DXF/CSV）を取り込み、3Dで直交配置してプレビューします。")

    colL, colR = st.columns([2,1])

    with colL:
        plan_path = st.text_input("平面図DXFへのパス（repo相対も可）", "io/plan.dxf")
        xs_folder = st.text_input("横断DXF/CSVフォルダ（repo相対も可）", "io/sections")

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
        center_layer = None
        if layers:
            st.subheader("レイヤ選択（中心線候補）")
            options = [f"{L.name}  (ents: {sum(L.entity_counts.values())}, len≈{L.length_sum:.1f})" for L in layers]
            idx = st.radio("中心線レイヤを1つ選択", list(range(len(options))), format_func=lambda i: options[i])
            center_layer = layers[int(idx)].name

        if st.button("取り込み → プレビュー生成", disabled=(center_layer is None)):
            try:
                cl = read_centerline(plan_path, [center_layer], unit_scale=unit_scale_plan)
                xs = read_cross_sections_from_folder(xs_folder, unit_scale=unit_scale_xs, layer_name=None)
                st.session_state.centerline = cl
                st.session_state.sections = xs
                st.session_state.center_layer = center_layer
                st.success(f"中心線: {len(cl)}点, 横断: {len(xs)}本")
            except Exception as e:
                st.error(f"取り込みに失敗: {e}")

    with colR:
        st.caption("Tips / 注意")
        st.write("- 平面は **LWPOLYLINE/LINE/SPLINE** から最長を採用します。")
        st.write("- 横断はフォルダ内の **DXF(1断面=1ファイル)** または **CSV(offset,elev)** を読み取ります。")
        st.write("- ファイル名やDXF内のTEXTに `100+00` / `KP12+350` があれば距離程を自動認識します。")
        if not _LEM_OK:
            st.warning("stabi_core.stabi_lem が見つからないため、円弧はダミーで表示しています。LEMが整ったら自動で切り替わります。")

    # --------- 3D Overview ---------
    if "centerline" in st.session_state and "sections" in st.session_state:
        cl: np.ndarray = st.session_state.centerline
        sections: Dict[float, np.ndarray] = st.session_state.sections

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=cl[:,0], y=cl[:,1], z=np.zeros(len(cl)),
                                   mode="lines", name="Centerline",
                                   line=dict(width=4, color="#A0A6B3")))
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
            # LEM result (real or mock)
            try:
                res = compute_min_circle({"section": oz})
                c = res["circle"]; fs = res["fs"]
                # arc polyline
                ph = np.linspace(-np.pi, np.pi, 241)
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

if __name__ == "__main__":
    page()
