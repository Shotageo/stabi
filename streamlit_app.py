# streamlit_app.py
from __future__ import annotations

import importlib
import traceback
import streamlit as st

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stabi – 斜面安定解析ビューアー",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Optional: おしゃれテーマ（ある場合のみ適用）
# ------------------------------------------------------------
try:
    from stabi_viz import ui_theme as ui  # ない場合は except 側へ
    ui.apply_global_style()
    ui.topbar(
        title="Stabi – 斜面安定解析ビューアー",
        right_html="DXF連携 • CLスナップ • LEM連携"
    )
except Exception:
    # ui_theme が無くてもそのまま続行
    pass

# ------------------------------------------------------------
# ページ登録
# ------------------------------------------------------------
PAGES = {}

# ① 新ビューワー（DXF取り込み→Noスナップ→3D配置＋LEM連携）
#    これを既定の「DXF取り込み・プレビュー」に割当
try:
    from stabi_viz import plan_preview_upload
    PAGES["DXF取り込み・プレビュー"] = plan_preview_upload.page
except Exception as e:
    st.sidebar.warning(f"plan_preview_upload 読み込み失敗: {e}")

# ② 旧ビューワー（必要なら残す／不要ならこのブロックごと削除）
try:
    from stabi_viz import plan_preview
    PAGES["DXF取り込み（旧・簡易）"] = plan_preview.page
except Exception as e:
    st.sidebar.warning(f"plan_preview 読み込み失敗: {e}")

# ③ ★ 追加：LEM 解析ラボ（土質・探索レンジを設定してバッチ解析）
try:
    from stabi_viz import lem_lab
    PAGES["LEM 解析ラボ（土質・探索）"] = lem_lab.page
except Exception as e:
    st.sidebar.warning(f"lem_lab 読み込み失敗: {e}")

# ④ その他（互換）: stabi_viz/basic, lem, result に page() があれば登録
for mod_name, title in [
    ("basic",  "Basic"),
    ("lem",    "LEM（旧）"),
    ("result", "Result"),
]:
    try:
        mod = importlib.import_module(f"stabi_viz.{mod_name}")
        if hasattr(mod, "page"):
            PAGES[title] = mod.page
    except Exception:
        # 見つからない/読み込めないときは黙ってスキップ
        pass

# ------------------------------------------------------------
# Sidebar – ページ選択
# ------------------------------------------------------------
st.sidebar.title("ページ選択")
if not PAGES:
    st.sidebar.error("まだページが登録されていません。\n\n"
                     "stabi_viz/plan_preview_upload.py を追加してください。")
    st.stop()

page_names = list(PAGES.keys())

# 先頭（DXF取り込み・プレビュー）を既定に
default_ix = 0 if page_names else 0
selected_name = st.sidebar.radio("ページ", page_names, index=default_ix)

# ------------------------------------------------------------
# 実行
# ------------------------------------------------------------
page_fn = PAGES.get(selected_name)
if page_fn is None:
    st.error(f"ページ '{selected_name}' が見つかりません。")
    st.stop()

try:
    page_fn()
except Exception as exc:
    st.error("ページ実行中にエラーが発生しました。")
    st.exception(exc)
    # 追加でトレース全文を表示（開発時の診断用）
    st.code("".join(traceback.format_exc()), language="text")
