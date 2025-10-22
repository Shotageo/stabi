# streamlit_app.py
from __future__ import annotations

import importlib
import traceback
import streamlit as st

st.set_page_config(
    page_title="Stabi – 斜面安定解析ビューアー",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# optional: スタイル（無ければスキップ）
try:
    from stabi_viz import ui_theme as ui
    ui.apply_global_style()
    ui.topbar(title="Stabi – 斜面安定解析ビューアー",
              right_html="DXF連携 • CLスナップ • LEM連携")
except Exception:
    pass

PAGES = {}

# ① 新ビューワー（DXF→Noスナップ→3D配置＋LEM結果重畳）
try:
    from stabi_viz import plan_preview_upload
    PAGES["DXF取り込み・プレビュー"] = plan_preview_upload.page
except Exception as e:
    st.sidebar.warning(f"plan_preview_upload 読み込み失敗: {e}")

# ② LEM: 3段ページ
try:
    from stabi_viz import lem_sections      # 断面プレビュー＋地層・水位の入力
    PAGES["LEM｜1 断面プレビュー＆地層入力"] = lem_sections.page
except Exception as e:
    st.sidebar.warning(f"lem_sections 読み込み失敗: {e}")

try:
    from stabi_viz import lem_soils         # 土質パラメータ（層テーブル）
    PAGES["LEM｜2 土質パラメータ"] = lem_soils.page
except Exception as e:
    st.sidebar.warning(f"lem_soils 読み込み失敗: {e}")

try:
    from stabi_viz import lem_run           # 探索レンジ設定＋解析実行
    PAGES["LEM｜3 解析実行"] = lem_run.page
except Exception as e:
    st.sidebar.warning(f"lem_run 読み込み失敗: {e}")

# ③ 旧ビューワー（必要なら残す）
try:
    from stabi_viz import plan_preview
    PAGES["DXF取り込み（旧・簡易）"] = plan_preview.page
except Exception:
    pass

# ④ 既存の互換ページ（あれば）
for mod_name, title in [("basic",  "Basic"), ("lem", "LEM（旧）"), ("result", "Result")]:
    try:
        mod = importlib.import_module(f"stabi_viz.{mod_name}")
        if hasattr(mod, "page"):
            PAGES[title] = mod.page
    except Exception:
        pass

# ---- サイドバー ----
st.sidebar.title("ページ選択")
if not PAGES:
    st.sidebar.error("まだページが登録されていません。")
    st.stop()

page_names = list(PAGES.keys())
selected_name = st.sidebar.radio("ページ", page_names, index=0)

# ---- 実行 ----
try:
    PAGES[selected_name]()
except Exception as exc:
    st.error("ページ実行中にエラーが発生しました。")
    st.exception(exc)
    st.code("".join(traceback.format_exc()), language="text")
