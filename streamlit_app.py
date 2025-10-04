# streamlit_app.py — cfg一本化 + 数値キー対応 + UI値プレビュー（安定版）

from __future__ import annotations

# ---- 標準/外部ライブラリ ----
import math, heapq, time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---- stabiパッケージから正規インポート（__init__.py 必須） ----
from stabi.stabi_lem import (
    Soil, GroundPL,
    make_ground_example, make_interface1_example, make_interface2_example,
    clip_interfaces_to_ground, arcs_from_center_by_entries_multi,
    fs_given_R_multi, arc_sample_poly_best_pair, driving_sum_for_R_multi,
)

# ================================================================
# Plot style（Theme/Tight layout/Legend切替）— ユーザー希望ブロック
# ================================================================
plt.style.use("default")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.frameon"] = True

# ================================================================
# Streamlit page basic setup
# ================================================================
st.set_page_config(page_title="Stabi LEM", layout="wide")
st.title("Stabi LEM｜cfg一元・安定版")

# --- cfg例（短縮版） ---
def default_cfg():
    return {
        "geom": {"H": 25.0, "L": 60.0},
        "water": {"mode": "WT", "offset": -2.0, "wl_points": None},
        "layers": {
            "n": 3,
            "mat": {
                1: {"gamma": 18.0, "c": 5.0,  "phi": 30.0},
                2: {"gamma": 19.0, "c": 8.0,  "phi": 28.0},
                3: {"gamma": 20.0, "c": 12.0, "phi": 25.0},
            },
        },
        "results": {"unreinforced": None, "chosen_arc": None}
    }

if "cfg" not in st.session_state:
    st.session_state["cfg"] = default_cfg()

# --- サンプル Ground 表示 ---
H = st.session_state["cfg"]["geom"]["H"]
L = st.session_state["cfg"]["geom"]["L"]
ground = make_ground_example(H, L)

fig, ax = plt.subplots(figsize=(8, 5))
Xd = np.linspace(0, L, 400)
Yg = np.array([ground.y_at(x) for x in Xd])
ax.plot(Xd, Yg, lw=2.0, label="Ground")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Ground preview")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.success("stabi_lem.py との接続成功。循環Importなし。")
