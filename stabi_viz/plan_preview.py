# stabi_viz/plan_preview_upload.py
from __future__ import annotations
import os, io, tempfile
from typing import Dict, List, Optional
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
        oz = np.asarray(cfg.get("section"))
        if oz is None or oz.size == 0:
            return {"fs": 1.10, "circle": {"oc": 0.0, "zc": 0.0, "R": 10.0, "x1": -5.0, "x2": 5.0}}
        oc = float(np.median(oz[:,0]))
        zc = float(np.min(oz[:,1]) + (np.max(oz[:,1]) - np.min(oz[:,1]))*0.25)
        R  = float(max(6.0, (np.max(oz[:,0]) - np.min(oz[:,0]))*0.35))
        return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R, "x1": oc-R*0.6, "x2": oc+R*0.6}}

from stabi_io.dxf_sections import list_layers, read_centerline, read_cross_sections_from_folder, parse_station

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

def _save_upload_to_temp(upload, suffix: str) -> str:
    data = upload.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name

def page():
    st.title("DXF取り込み（ドラッグ＆ドロップ版）")
    st.caption("平面DXFと横断DXF/CSVをドラッグ＆ドロップで読み込み、3Dプレビューします。" )

    colA, colB = st.columns([1,1])
    with colA:
        st.subheader("① 平面（Centerline）")
        plan_up = st.file_uploader("平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False)
        unit_scale_plan = st.number_input("倍率（mm→mは0.001）", value=1.0, step=0.001, format="%.3f", key="u_plan")

        if plan_up is not None:
            temp_plan_path = _save_upload_to_temp(plan_up, ".dxf")
            try:
                layers = list_layers(temp_plan_path)
                st.session_state._layers = layers
                st.success(f"{len(layers)} レイヤを検出")
            except Exception as e:
                st.error(f"DXF読み込みに失敗: {e}")
        else:
            layers = []

        if layers:
            options = [f"{L.name}  (ents: {sum(L.entity_counts.values())}, len≈{L.length_sum:.1f})" for L in layers]
            idx = st.radio("中心線レイヤを選択", list(range(len(options))), format_func=lambda i: options[i], key="laypick")
            center_layer = layers[int(idx)].name
        else:
            center_layer = None

    with colB:
        st.subheader("② 横断（Cross Sections）")
        xs_files = st.file_uploader("横断DXF/CSV（複数可）", type=["dxf","csv"], accept_multiple_files=True)
        unit_scale_xs = st.number_input("倍率（mm→mは0.001）", value=1.0, step=0.001, format="%.3f", key="u_xs" )

    run = st.button("読み込み → プレビュー", type="primary", disabled=plan_up is None or center_layer is None or not xs_files)

    if run:
        import tempfile, os
        tmpdir = tempfile.mkdtemp(prefix="stabi_xs_")
        for f in xs_files:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as w:
                w.write(f.getbuffer())

        try:
            cl = read_centerline(temp_plan_path, [center_layer], unit_scale=unit_scale_plan)
            xs = read_cross_sections_from_folder(tmpdir, unit_scale=unit_scale_xs, layer_name=None)
            st.session_state.centerline = cl
            st.session_state.sections = xs
            st.success(f"中心線: {len(cl)}点, 横断: {len(xs)}本 でプレビューを生成します。")
        except Exception as e:
            st.error(f"取り込みに失敗: {e}")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=cl[:,0], y=cl[:,1], z=np.zeros(len(cl)),
                                   mode="lines", name="Centerline",
                                   line=dict(width=4, color="#A0A6B3")))

        pitch = st.slider("表示横断ピッチ[m]", 5, 50, 20, step=5)
        show_keys = []
        for s in sorted(st.session_state.sections.keys()):
            if int(s) % pitch == 0:
                show_keys.append(s)
        if not show_keys:
            show_keys = list(st.session_state.sections.keys())[::max(1, len(st.session_state.sections)//10)]

        for s in show_keys:
            P, t, n = _tangent_normal(cl, s)
            oz = st.session_state.sections[s]
            X,Y,Z = _xs_to_world3D(P, n, oz)
            fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines",
                                       name=f"XS {s:.0f}", line=dict(width=5, color="#FFFFFF"), opacity=0.95))
            try:
                res = compute_min_circle({"section": oz})
                c = res["circle"]; fs = res["fs"]
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
