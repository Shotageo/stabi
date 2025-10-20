# stabi_viz/plan_preview_upload.py
from __future__ import annotations
import os, io, tempfile
from typing import Dict, List, Optional
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# LEM optional
try:
    from stabi_core.stabi_lem import compute_min_circle  # expected real API
    _LEM_OK = True
except Exception:
    _LEM_OK = False
    def compute_min_circle(cfg):
        oz = np.asarray(cfg.get("section"))
        if oz is None or oz.size == 0:
            return {"fs": 1.10, "circle": {"oc": 0.0, "zc": 0.0, "R": 10.0, "x1": -5.0, "x2": 5.0}}
        oc = float(np.median(oz[:,0])); zc = float(np.min(oz[:,1]) + (np.max(oz[:,1])-np.min(oz[:,1]))*0.25)
        R  = float(max(6.0, (np.max(oz[:,0]) - np.min(oz[:,0]))*0.35))
        return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R, "x1": oc-R*0.6, "x2": oc+R*0.6}}

from stabi_io.dxf_sections import (
    list_layers, read_centerline, extract_no_labels_with_station,
    read_single_section_file, normalize_no_key, parse_station
)

# ---------- helpers ----------
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

def page():
    st.title("DXF取り込み（No.割当ウィザード）")
    st.caption("平面のNo.ラベルを抽出 → 横断ファイルにNo.を割当 → 中心線に直交配置して3D表示")

    # ------------------- Step 1: 平面 -------------------
    with st.expander("Step 1｜平面を読み込み（中心線＋No.ラベル）", expanded=True):
        plan_up = st.file_uploader("平面DXF（1ファイル）", type=["dxf"], accept_multiple_files=False, key="plan")
        unit_scale_plan = st.number_input("平面倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f", key="u_plan")
        if plan_up is not None:
            # save temp
            temp_plan_path = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
            temp_plan_path.write(plan_up.read()); temp_plan_path.flush(); temp_plan_path.close()
            st.session_state._plan_path = temp_plan_path.name

            # choose layers
            layers = list_layers(st.session_state._plan_path)
            st.success(f"レイヤ検出：{len(layers)}")
            options = [f"{L.name}  (len≈{L.length_sum:.1f})" for L in layers]
            idx = st.radio("中心線レイヤ", list(range(len(options))), format_func=lambda i: options[i], key="cl_pick")
            cl_layer = layers[int(idx)].name

            label_layers = st.multiselect("測点ラベルレイヤ（TEXT/MTEXT）", [L.name for L in layers], default=[])

            if st.button("中心線＋No.抽出を実行", type="primary"):
                try:
                    cl = read_centerline(st.session_state._plan_path, [cl_layer], unit_scale=unit_scale_plan)
                    labels = extract_no_labels_with_station(st.session_state._plan_path, label_layers, cl, unit_scale=unit_scale_plan)
                    st.session_state.centerline = cl
                    st.session_state.no_table = labels
                    st.success(f"中心線: {len(cl)}点, No.ラベル: {len(labels)}件")
                except Exception as e:
                    st.error(f"抽出失敗: {e}")

        # show table
        if "no_table" in st.session_state:
            st.write("検出No.（中心線へ投影済み）")
            rows = [f"{d['key']}  →  s={d['station_m']:.2f} m  (offset={d['dist']:.2f} m)" for d in st.session_state.no_table]
            st.code("\n".join(rows))

    # ------------------- Step 2: 横断 -------------------
    with st.expander("Step 2｜横断ファイルをアップロードしてNo.を割当", expanded=True):
        xs_files = st.file_uploader("横断DXF/CSV（複数可）", type=["dxf","csv"], accept_multiple_files=True, key="xs")
        unit_scale_xs = st.number_input("横断倍率（mm→m は 0.001）", value=1.0, step=0.001, format="%.3f", key="u_xs")

        # axis corrections (global, simple)
        st.caption("補正（必要に応じてON）")
        axis_mode = st.selectbox("軸割り", ["X=offset / Y=elev（標準）", "X=elev / Y=offset（入替）"])
        flip_o = st.checkbox("オフセット左右反転", value=False)
        flip_z = st.checkbox("標高上下反転", value=False)
        center_o = st.checkbox("オフセット中央値を0に", value=True)

        if xs_files and "no_table" in st.session_state:
            # Build No choices
            no_choices = [d["key"] for d in st.session_state.no_table]
            no_to_s = {d["key"]: d["station_m"] for d in st.session_state.no_table}

            # Per-file assignment UI
            assigned: Dict[str, Dict] = {}
            for f in xs_files:
                with st.expander(f"割当：{f.name}", expanded=False):
                    # prefill from filename
                    guess = normalize_no_key(f.name) or ""
                    sel = st.selectbox("割当No.", ["（未選択）"] + no_choices, index=(no_choices.index(guess)+1) if guess in no_choices else 0, key=f"sel_{f.name}")
                    # quick preview (2D)
                    import matplotlib.pyplot as plt
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="."+f.name.split(".")[-1])
                    tmp.write(f.getbuffer()); tmp.flush(); tmp.close()
                    sec = read_single_section_file(tmp.name, layer_name=None, unit_scale=unit_scale_xs)
                    if sec is not None:
                        oz = sec.copy()
                        if axis_mode.startswith("X=elev"):
                            oz = oz[:, [1,0]]
                        if flip_o: oz[:,0] *= -1
                        if flip_z: oz[:,1] *= -1
                        if center_o: oz[:,0] -= np.median(oz[:,0])
                        fig2, ax = plt.subplots(figsize=(5.2,2.6))
                        ax.plot(oz[:,0], oz[:,1], lw=2.0)
                        ax.grid(True, alpha=0.3); ax.set_xlabel("offset [m]"); ax.set_ylabel("elev [m]")
                        st.pyplot(fig2, use_container_width=True)
                        if sel != "（未選択）":
                            assigned[f.name] = {"path": tmp.name, "oz": oz, "no_key": sel, "s": no_to_s[sel]}

            st.session_state._assigned = assigned
            st.info(f"割当済み：{len(assigned)} / {len(xs_files)}")

    # ------------------- Step 3: 3D配置 -------------------
    with st.expander("Step 3｜3Dプレビュー", expanded=True):
        can_run = ("centerline" in st.session_state) and ("_assigned" in st.session_state) and st.session_state._assigned
        if not can_run:
            st.warning("中心線＋No. と 横断の割当を完了してください。")
            return
        cl = st.session_state.centerline
        assigned = st.session_state._assigned

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=cl[:,0], y=cl[:,1], z=np.zeros(len(cl)),
                                   mode="lines", name="Centerline",
                                   line=dict(width=4, color="#A0A6B3")))

        # place assigned sections
        for name, rec in sorted(assigned.items(), key=lambda kv: kv[1]["s"]):
            s = float(rec["s"]); oz = rec["oz"]; P, t, n = _tangent_normal(cl, s)
            X,Y,Z = _xs_to_world3D(P, n, oz)
            fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines",
                                       name=f"{rec['no_key']}", line=dict(width=5, color="#FFFFFF"), opacity=0.95))
            # LEM
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
                                           text=[f"{rec['no_key']}  Fs={fs:.2f}"], textfont=dict(color="#FFD34D", size=12),
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
