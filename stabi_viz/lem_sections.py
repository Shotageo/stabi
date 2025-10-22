# stabi_viz/lem_sections.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def _plot_section(oz: np.ndarray,
                  horizons: dict[str, np.ndarray] | None = None,
                  water: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    # 断面（地表）
    fig.add_trace(go.Scatter(x=oz[:,0], y=oz[:,1],
                             mode="lines", name="地表（断面）",
                             line=dict(width=3, color="#FFFFFF")))
    # offset=0
    y0 = float(np.nanmin(oz[:,1])); y1 = float(np.nanmax(oz[:,1]))
    fig.add_trace(go.Scatter(x=[0,0], y=[y0, y1],
                             mode="lines", name="CL",
                             line=dict(width=1, dash="dot", color="#8AA0FF")))
    # 層境界
    if horizons:
        for name, arr in horizons.items():
            if arr is None or len(arr)==0: continue
            fig.add_trace(go.Scatter(x=arr[:,0], y=arr[:,1],
                                     mode="lines", name=f"層境界:{name}",
                                     line=dict(width=2)))
    # 水位
    if water is not None and len(water)>0:
        fig.add_trace(go.Scatter(x=water[:,0], y=water[:,1],
                                 mode="lines", name="水位",
                                 line=dict(width=2, color="#33C3FF")))
    fig.update_layout(
        height=420, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="offset [m]", yaxis_title="elev [m]",
        paper_bgcolor="#0f1115", plot_bgcolor="#0f1115",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(gridcolor="#2a2f3a"); fig.update_yaxes(gridcolor="#2a2f3a")
    return fig

def _get_assigned():
    assigned = st.session_state.get("_assigned", {})
    if not assigned:
        st.info("先に『DXF取り込み・プレビュー』で断面を割当ててください。")
        return None
    return assigned

def page():
    st.title("LEM｜1 断面プレビュー＆地層・水位入力")

    assigned = _get_assigned()
    if not assigned: return

    keys = list(assigned.keys())
    sec_key = st.selectbox("断面を選択（アップロード名）", keys, index=0)
    rec = assigned[sec_key]
    oz = np.asarray(rec["oz"], float)

    # 既存データ
    st.session_state.setdefault("lem_horizons", {})  # {sec_key: {name: Nx2}}
    st.session_state.setdefault("lem_water_lines", {})  # {sec_key: Nx2}
    horizons: dict[str, np.ndarray] = st.session_state.lem_horizons.get(sec_key, {})
    water: np.ndarray | None = st.session_state.lem_water_lines.get(sec_key)

    with st.expander("プレビュー", expanded=True):
        st.plotly_chart(_plot_section(oz, horizons, water), use_container_width=True)

    with st.expander("層境界（層ごとの上面ライン）を追加／更新", expanded=True):
        st.caption("CSV形式で offset,elev 列を持つデータを層ごとに読み込みます。")
        col = st.columns([2,1,1,1])
        with col[0]:
            layer_name = st.text_input("層名（例: 表土, 砂層, 風化岩, 基盤岩）", value="砂層")
        with col[1]:
            up = st.file_uploader("CSV", type=["csv"], key="hcsv")
        with col[2]:
            add_btn = st.button("読み込んで追加/更新")
        with col[3]:
            clear_btn = st.button("この断面の全層境界をクリア")

        if add_btn and up is not None and layer_name.strip():
            try:
                df = pd.read_csv(up)
                arr = df[["offset","elev"]].to_numpy(float)
                arr = arr[np.argsort(arr[:,0])]
                horizons = dict(horizons)  # copy
                horizons[layer_name.strip()] = arr
                st.session_state.lem_horizons[sec_key] = horizons
                st.success(f"層境界 {layer_name} を登録しました。")
            except Exception as e:
                st.error(f"読み込み失敗: {e}")

        if clear_btn:
            st.session_state.lem_horizons[sec_key] = {}
            st.success("この断面の層境界を全クリアしました。")

        if horizons:
            names = list(horizons.keys())
            pick = st.selectbox("既存の層境界", names, index=0)
            if st.button("選択した層境界を削除"):
                horizons = dict(horizons)
                horizons.pop(pick, None)
                st.session_state.lem_horizons[sec_key] = horizons
                st.success(f"{pick} を削除しました。")

    with st.expander("水位線を追加／更新（任意）", expanded=False):
        upw = st.file_uploader("水位線CSV（offset,elev）", type=["csv"], key="wcsv")
        c1,c2 = st.columns([1,1])
        with c1:
            set_btn = st.button("水位線を設定")
        with c2:
            clr_btn = st.button("水位線をクリア")
        if set_btn and upw is not None:
            try:
                df = pd.read_csv(upw)
                arr = df[["offset","elev"]].to_numpy(float)
                arr = arr[np.argsort(arr[:,0])]
                st.session_state.lem_water_lines[sec_key] = arr
                st.success("水位線を設定しました。")
            except Exception as e:
                st.error(f"読み込み失敗: {e}")
        if clr_btn:
            st.session_state.lem_water_lines.pop(sec_key, None)
            st.success("水位線をクリアしました。")

    with st.expander("ダウンロード（確認用）", expanded=False):
        if st.button("この断面の層境界をCSVに書き出し"):
            buf = []
            for name, arr in (st.session_state.lem_horizons.get(sec_key, {})).items():
                df = pd.DataFrame(arr, columns=["offset","elev"])
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"{name}.csv をダウンロード", csv, file_name=f"{sec_key}_{name}.csv", mime="text/csv")
