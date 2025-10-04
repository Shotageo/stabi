# pages/20_layers.py
# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="安定板２｜2) 地層", layout="wide")
st.title("2) 地層（簡易版）")

st.info("現状は 1 層相当（前ページの γ・c・φ を採用）。複層は今後このページで拡張。")

# 拡張用の枠（今は保存だけ）
if st.button("OK（1層として続行）", type="primary"):
    st.success("保存済み。『3) 円弧探索』へ。")
st.page_link("pages/30_slip_search.py", label="→ 3) 円弧探索", icon="🌀")

