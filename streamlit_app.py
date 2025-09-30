# streamlit_app.py
import streamlit as st
from stabi_lem import demo_fs

st.set_page_config(page_title="Stabi LEM Mock", layout="wide")
st.title("Stabi LEM Mock")

st.write("ローカル動作確認用。後でBishop/Felleniusに差し替える。")

fs = demo_fs()
st.metric("Fs (demo)", f"{fs:.3f}")
