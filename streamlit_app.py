# streamlit_app.py  --- BOOT STUB ---
import streamlit as st, sys, platform, os, importlib.util, traceback
st.set_page_config(page_title="Stabi Boot Stub", layout="wide")
st.title("Stabi Boot Stub ✅")
st.caption("このページが表示されれば、アプリは起動できています。")

st.write({
    "python": sys.version,
    "platform": platform.platform(),
    "cwd": os.getcwd(),
    "files": sorted(os.listdir(".")),
})

st.divider()
st.subheader("stabi_lem.py インポート試験")
here = os.path.dirname(__file__)
path = os.path.join(here, "stabi_lem.py")
st.write({"stabi_lem.py_exists": os.path.exists(path)})

if os.path.exists(path):
    try:
        spec = importlib.util.spec_from_file_location("stabi_lem", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
        st.success("stabi_lem.py import OK")
        st.write({"members": [k for k in dir(mod) if k in ("Soil","Slope","search_center_grid","search_entry_exit_adaptive")]})
    except Exception as e:
        st.error("stabi_lem.py import NG")
        st.code("".join(traceback.format_exception(e)))
else:
    st.warning("リポジトリ直下に stabi_lem.py がありません。")