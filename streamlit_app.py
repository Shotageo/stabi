# streamlit_app.py
# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Stabi｜安定板２（ハブ）", layout="wide")

st.title("Stabi｜安定板２（ハブ）")
st.markdown("""
段階ごとに進めます。左のサイドバーの **Pages** にも同じ順番で並びます。

1. **地形・材料**（基本条件の保存）  
2. **地層**（必要なら；今は一括土質として保存）  
3. **円弧探索（無補強）**：最小Fs円弧を決定して保存  
4. **ソイルネイル補強**：③の“同じ円弧・同じスライス分割”で補強後Fs
""")

st.subheader("クイックリンク")
st.page_link("pages/10_terrain_material.py", label="→ 1) 地形・材料", icon="🪧")
st.page_link("pages/20_layers.py",         label="→ 2) 地層",       icon="🧱")
st.page_link("pages/30_slip_search.py",    label="→ 3) 円弧探索",   icon="🌀")
st.page_link("pages/40_soil_nail.py",      label="→ 4) ソイルネイル補強", icon="🪛")

with st.expander("セッションの状態（確認用）"):
    st.write(dict(st.session_state))
