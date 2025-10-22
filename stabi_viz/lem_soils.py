# stabi_viz/lem_soils.py
from __future__ import annotations
import pandas as pd
import streamlit as st

_DEF = [
    {"name":"表土",    "phi_deg":28.0, "c_kPa":  8.0, "gamma_kN_m3":18.0, "ru":0.0, "partial_gammaM":1.0},
    {"name":"砂層",    "phi_deg":32.0, "c_kPa": 10.0, "gamma_kN_m3":19.0, "ru":0.0, "partial_gammaM":1.0},
    {"name":"風化岩",  "phi_deg":36.0, "c_kPa": 20.0, "gamma_kN_m3":20.5,"ru":0.0, "partial_gammaM":1.0},
    {"name":"基盤岩",  "phi_deg":40.0, "c_kPa": 50.0, "gamma_kN_m3":22.0, "ru":0.0, "partial_gammaM":1.0},
]

def page():
    st.title("LEM｜2 土質パラメータ")

    st.session_state.setdefault("lem_soils_table", pd.DataFrame(_DEF))
    st.session_state.setdefault("lem_layer_order", [r["name"] for r in _DEF])
    st.session_state.setdefault("lem_ru_default", 0.0)
    st.session_state.setdefault("lem_partial_gammaF", 1.0)

    st.caption("層テーブル（列: name, φ[deg], c[kPa], γ[kN/m³], ru, 部分係数γM）")
    df = st.data_editor(
        st.session_state.lem_soils_table,
        use_container_width=True, hide_index=True, num_rows="dynamic",
        column_config={
            "name":"層名", "phi_deg":"φ[deg]", "c_kPa":"c[kPa]",
            "gamma_kN_m3":"γ[kN/m³]", "ru":"ru", "partial_gammaM":"γM"
        }
    )
    st.session_state.lem_soils_table = df

    st.divider()
    st.subheader("層の上下順（上から下）")
    opts = list(df["name"].astype(str))
    if not opts:
        st.warning("層名がありません。上の表に1行以上追加してください。")
    else:
        order = st.multiselect("順序（上から下に並べる）", opts, default=st.session_state.lem_layer_order)
        if order: st.session_state.lem_layer_order = order

    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        st.session_state.lem_ru_default = st.number_input("ru 既定値（水位が無い断面に適用）", value=float(st.session_state.lem_ru_default), step=0.05)
    with c2:
        st.session_state.lem_partial_gammaF = st.number_input("部分係数 γF（荷重）", value=float(st.session_state.lem_partial_gammaF), step=0.05)

    st.success("設定は session_state に保存され、次ページの解析で使われます。")
