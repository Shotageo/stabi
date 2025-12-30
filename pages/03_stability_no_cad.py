import streamlit as st
import numpy as np

from stabi_lem import compute_fs  # ← 名前が違ってたら後で直す

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stabi | 安定板３（No CAD）", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def make_simple_slope_polyline(H: float, m: float, crest_len: float = 10.0, toe_len: float = 10.0):
    """
    Simple piecewise-linear ground surface:
      left horizontal crest -> straight slope -> right horizontal toe
    Coordinates: x-z (z up)
    """
    x0 = 0.0
    z0 = H
    x1 = crest_len
    z1 = H
    x2 = crest_len + m * H
    z2 = 0.0
    x3 = x2 + toe_len
    z3 = 0.0
    xs = np.array([x0, x1, x2, x3], dtype=float)
    zs = np.array([z0, z1, z2, z3], dtype=float)
    return xs, zs

def plot_section_and_circle(ax, xs, zs, circle=None, title=""):
    ax.plot(xs, zs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.grid(True)

    if circle is not None:
        xc, zc, R = circle
        th = np.linspace(0, 2*np.pi, 720)
        ax.plot(xc + R*np.cos(th), zc + R*np.sin(th))
        ax.scatter([xc], [zc])

def resolve_lem_callable():
    """
    Try to find a callable in stabi_lem.py without knowing exact function name.
    We try a set of common names. If nothing found, return (None, diagnostics).
    """
    import importlib
    mod = importlib.import_module("stabi_lem")

    candidates = [
        "compute_fs",
        "compute_Fs",
        "run_lem",
        "run_LEM",
        "bishop_fs",
        "bishop_simplified",
        "search_critical_circle",
        "solve",
        "analyze",
    ]

    found = []
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            found.append(name)

    return mod, found

def call_lem(mod, fn_name, xs, zs, soil, water, search):
    """
    Call the discovered LEM function with multiple possible signatures.
    We attempt several argument styles to maximize chance of working.
    """
    fn = getattr(mod, fn_name)

    # Common payloads
    section = {"xs": xs, "zs": zs}
    params = {
        "gamma": soil["gamma"],
        "c": soil["c"],
        "phi_deg": soil["phi_deg"],
        "layer_thickness": soil["t"],
    }
    water_params = {
        "enabled": water["enabled"],
        "zw": water["zw"],
    }
    search_params = {
        "n_slices": search["n_slices"],
        "grid_nx": search["grid_nx"],
        "grid_nz": search["grid_nz"],
        "grid_nr": search["grid_nr"],
        "xc_min": search["xc_min"],
        "xc_max": search["xc_max"],
        "zc_min": search["zc_min"],
        "zc_max": search["zc_max"],
        "R_min": search["R_min"],
        "R_max": search["R_max"],
    }

    attempts = []

    # Attempt patterns (from most structured to simplest)
    attempts.append(("dict_section_params", lambda: fn(section, params, water_params, search_params)))
    attempts.append(("named_args_section", lambda: fn(xs=xs, zs=zs, **params, **water_params, **search_params)))
    attempts.append(("positional_basic", lambda: fn(xs, zs, params)))
    attempts.append(("positional_soil", lambda: fn(xs, zs, soil["gamma"], soil["c"], soil["phi_deg"])))
    attempts.append(("positional_soil_slices", lambda: fn(xs, zs, soil["gamma"], soil["c"], soil["phi_deg"], search["n_slices"])))

    last_err = None
    for label, thunk in attempts:
        try:
            out = thunk()
            return out, f"OK: call style = {label}"
        except Exception as e:
            last_err = e

    raise RuntimeError(f"LEM function '{fn_name}' was found but none of the call signatures worked. Last error: {last_err}")

# -----------------------------
# UI
# -----------------------------
st.title("安定板３（No CAD）")
st.caption("DXFは一切使わない。手入力断面 → LEM → 図とFSだけ。")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.subheader("断面（手入力）")
    H = st.number_input("斜面高さ H [m]", min_value=1.0, value=10.0, step=0.5)
    m = st.number_input("勾配 1:m（水平/鉛直）", min_value=0.2, value=1.2, step=0.1)
    crest_len = st.number_input("天端水平長 [m]", min_value=0.0, value=10.0, step=1.0)
    toe_len = st.number_input("法尻水平長 [m]", min_value=0.0, value=10.0, step=1.0)

with c2:
    st.subheader("土質（最小セット）")
    gamma = st.number_input("単位体積重量 γ [kN/m³]", min_value=10.0, value=18.0, step=0.5)
    c = st.number_input("粘着力 c [kPa]", min_value=0.0, value=5.0, step=1.0)
    phi = st.number_input("内部摩擦角 φ [deg]", min_value=0.0, value=30.0, step=1.0)
    t = st.number_input("表層厚 t [m]", min_value=0.1, value=2.0, step=0.1)

with c3:
    st.subheader("探索（円弧すべり）")
    n_slices = st.slider("Slice数", min_value=10, max_value=60, value=25, step=1)

    st.markdown("**中心探索範囲**（ざっくりで良い）")
    xc_min = st.number_input("xc min", value=-20.0, step=1.0)
    xc_max = st.number_input("xc max", value= 40.0, step=1.0)
    zc_min = st.number_input("zc min", value=-40.0, step=1.0)
    zc_max = st.number_input("zc max", value= 20.0, step=1.0)

    st.markdown("**半径範囲**")
    R_min = st.number_input("R min", value= 5.0, step=1.0)
    R_max = st.number_input("R max", value=80.0, step=1.0)

    st.markdown("**グリッド粗さ**（まず粗くてOK）")
    grid_nx = st.slider("xc 分割数", 3, 40, 10)
    grid_nz = st.slider("zc 分割数", 3, 40, 10)
    grid_nr = st.slider("R 分割数", 3, 40, 10)

st.divider()

w1, w2 = st.columns([1, 2])

with w1:
    st.subheader("地下水（今はOFF推奨）")
    water_enabled = st.checkbox("地下水を考慮する", value=False)
    zw = st.number_input("水位 z_w [m]（z=0が法尻）", value=0.0, step=0.5, disabled=(not water_enabled))

    run = st.button("計算する", type="primary", use_container_width=True)

with w2:
    xs, zs = make_simple_slope_polyline(H, m, crest_len=crest_len, toe_len=toe_len)

    fig, ax = plt.subplots()
    plot_section_and_circle(ax, xs, zs, circle=None, title="断面（手入力）")
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Run analysis
# -----------------------------
if run:
    soil = {"gamma": float(gamma), "c": float(c), "phi_deg": float(phi), "t": float(t)}
    water = {"enabled": bool(water_enabled), "zw": float(zw)}
    search = {
        "n_slices": int(n_slices),
        "grid_nx": int(grid_nx), "grid_nz": int(grid_nz), "grid_nr": int(grid_nr),
        "xc_min": float(xc_min), "xc_max": float(xc_max),
        "zc_min": float(zc_min), "zc_max": float(zc_max),
        "R_min": float(R_min), "R_max": float(R_max),
    }

    st.subheader("結果")
    try:
        mod, found = resolve_lem_callable()

        if not found:
            st.error(
                "stabi_lem.py の中に呼べそうな関数名が見つからない。\n"
                "次のどちらかをして：\n"
                "1) stabi_lem.py の先頭〜関数定義あたりをここに貼る（50〜120行で十分）\n"
                "2) 関数名だけ教える（例：search_critical_circle とか compute_fs とか）"
            )
            st.stop()

        # まずは先頭候補を試す
        fn_name = found[0]
        out, diag = call_lem(mod, fn_name, xs, zs, soil, water, search)

        # 出力の取り出し：dictでもtupleでも対応
        FSmin = None
        circle = None

        if isinstance(out, dict):
            for k in ["FSmin", "fs_min", "FS", "fs"]:
                if k in out:
                    FSmin = float(out[k])
                    break
            for k in ["circle", "critical_circle", "crit_circle"]:
                if k in out and out[k] is not None:
                    circle = out[k]
                    break
        elif isinstance(out, (tuple, list)) and len(out) >= 1:
            # よくある: (FSmin, circle, ...)
            FSmin = float(out[0]) if out[0] is not None else None
            if len(out) >= 2:
                circle = out[1]

        if FSmin is None:
            st.warning("計算は返ってきたけど、FSmin を取り出せなかった。out を表示するわ。")
            st.code(repr(out))
        else:
            st.metric("最小安全率 FSmin", f"{FSmin:.3f}")
            st.caption(diag)

            # 描画
            fig2, ax2 = plt.subplots()
            plot_section_and_circle(ax2, xs, zs, circle=circle, title="断面 + すべり円（候補）")
            st.pyplot(fig2, clear_figure=True)

            with st.expander("デバッグ出力（必要なら）", expanded=False):
                st.code(repr(out))

    except Exception as e:
        st.error(f"計算で落ちた：{e}")
        st.info(
            "対処：stabi_lem.py の『実際に呼ぶべき関数名』と、その関数の引数を特定する。\n"
            "stabi_lem.py の先頭〜主要関数を 80行くらい貼れば、私が“正しい呼び出し”に直す。"
        )
