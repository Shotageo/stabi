# ============================================================================
# Stabi — Streamlit App (full file)
# ----------------------------------------------------------------------------
# Drop this file at repo root as `streamlit_app.py`.
# Runs both locally and on Streamlit Cloud.
#
# Pages included:
#   - DXF取り込み・プレビュー : stabi_viz.plan_preview.page
#   - 既存互換ページ          : （見つかれば自動登録）
# ============================================================================

from __future__ import annotations
import importlib
import os
import sys
from typing import Callable, Dict

import streamlit as st

# ----------------------------- App Appearance --------------------------------
st.set_page_config(page_title="Stabi", layout="wide", page_icon="🛰️")

# Minimal dark UI tweaks (safe inline CSS)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
      .stApp { background-color: #0f1115; color: #eaeef2; }
      .stSidebar { background-color: #0d0f14; }
      header, footer { visibility: hidden; }
      .stMetric { background: rgba(20,22,28,.65); border-radius: 10px; padding: 8px 10px; }
      .st-emotion-cache-1r6slb0 { border-color: rgba(255,255,255,.08) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Page Loader ---------------------------------
PageFn = Callable[[], None]
PAGES: Dict[str, PageFn] = {}

def _try_register(title: str, module_path: str, attr: str = "page"):
    """Lazy-import a page module if present; ignore if missing."""
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr, None)
        if callable(fn):
            PAGES[title] = fn
            return True
    except Exception as e:
        # Keep silent but show in diagnostics section if user opens it
        _diagnostics.append((title, module_path, str(e)))
    return False

_diagnostics = []  # capture optional import errors (shown on demand)

# Mandatory: the new DXF flow
_try_register("DXF取り込み・プレビュー", "stabi_viz.plan_preview")

# Optional legacy/other pages in the repo (best-effort)
_try_register("補強後の計算", "pages._40_soil_nail") or _try_register("補強後の計算", "pages.40_soil_nail")
_try_register("プレビュー（平面）", "stabi_viz.plan_preview_legacy")
_try_register("テスト/可視化ユーティリティ", "viz.plan_preview")

# If no page could be registered (fresh repo), add a placeholder
if not PAGES:
    def _placeholder():
        st.title("Stabi")
        st.info(
            "まだページが登録されていません。`stabi_viz/plan_preview.py` を追加するか、"
            "`stabi_viz.plan_preview.page()` を提供してください。"
        )
    PAGES["ようこそ"] = _placeholder

# --------------------------------- Sidebar -----------------------------------
st.sidebar.title("Stabi")
choice = st.sidebar.radio("ページ", list(PAGES.keys()), index=0)

# Diagnostics expander
with st.sidebar.expander("Diagnostics", expanded=False):
    if _diagnostics:
        st.write("Optional pages that failed to import:")
        for title, mod, err in _diagnostics:
            st.code(f"{title} <- {mod}\n{err}")
    else:
        st.write("No import issues detected.")

# --------------------------------- Router ------------------------------------
PAGES[choice]()

# --------------------------------- Footer ------------------------------------
st.markdown(
    """
    <div style="position:fixed; right:18px; bottom:12px; opacity:.55; font-size:12px;">
      Stabi · Streamlit Viewer
    </div>
    """,
    unsafe_allow_html=True,
)
