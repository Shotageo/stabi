# streamlit_app.py
import streamlit as st
import importlib
import sys
import traceback

# ===============================================================
# アプリタイトルと基本設定
# ===============================================================
st.set_page_config(
    page_title="Stabi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body { background-color: #0f1115; color: #EEE; }
    .stButton>button { border-radius: 10px; padding: 0.4em 1em; font-weight: 600; }
    .stSlider label, .stTextInput label, .stNumberInput label { color: #CCC !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🪨 Stabi – 斜面安定解析ビューワー")

# ===============================================================
# ページ登録
# ===============================================================
PAGES = {}

try:
    from stabi_viz import plan_preview
    PAGES["DXF取り込み・プレビュー"] = plan_preview.page
except Exception as e:
    st.sidebar.warning(f"plan_preview 読み込み失敗: {e}")

try:
    from stabi_viz import plan_preview_upload
    PAGES["DXF取り込み（ドラッグ＆ドロップ）"] = plan_preview_upload.page
except Exception as e:
    st.sidebar.warning(f"plan_preview_upload 読み込み失敗: {e}")

# 他のページを自動登録する場合（例: 基本設定, LEM解析など）
for mod_name in ["basic", "lem", "result"]:
    try:
        mod = importlib.import_module(f"stabi_viz.{mod_name}")
        if hasattr(mod, "page"):
            PAGES[f"{mod_name.title()}"] = mod.page
    except Exception:
        pass

# ===============================================================
# サイドバー構成
# ===============================================================
st.sidebar.header("ページ選択")
if not PAGES:
    st.sidebar.error("まだページが登録されていません。`stabi_viz/plan_preview_upload.py` を追加してください。")
    st.stop()

page_name = st.sidebar.radio("ページ", list(PAGES.keys()))

# ===============================================================
# ページ実行
# ===============================================================
try:
    PAGES[page_name]()  # ページ関数を呼び出し
except Exception as e:
    st.error(f"ページ実行中にエラーが発生しました: {e}")
    st.exception(e)
    st.code("".join(traceback.format_exception(*sys.exc_info())), language="python")

# ===============================================================
# フッター
# ===============================================================
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Stabi project")
