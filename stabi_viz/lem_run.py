# stabi_viz/lem_run.py
from __future__ import annotations
import json, hashlib
from typing import Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# エンジン：安定版3→既存→フォールバック
_ENGINE = "fallback"
try:
    from stabi_core.stabi_lem_v3 import solve_min_circle as _solve
    _ENGINE = "stabi_core.stabi_lem_v3.solve_min_circle"
except Exception:
    try:
        from stabi_core.stabi_lem import compute_min_circle as _solve
        _ENGINE = "stabi_core.stabi_lem.compute_min_circle"
    except Exception:
        def _solve(cfg: Dict) -> Dict:
            sec = np.asarray(cfg["section"], float)
            oc = float(np.median(sec[:,0])); zc = float(np.percentile(sec[:,1],25))
            R  = float(max(6.0, (sec[:,0].max()-sec[:,0].min())*0.35))
            return {"fs":1.12, "circle":{"oc":oc,"zc":zc,"R":R}, "meta":{"fallback":True}}

def _sig(oz: np.ndarray, params: Dict) -> str:
    blob = np.asarray(oz, np.float32).tobytes() + json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]

def _plot_result_2d(oz: np.ndarray, circ: Dict|None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oz[:,0], y=oz[:,1], mode="lines",
                             name="断面", line=dict(width=3, color="#FFFFFF")))
    if circ and all(k in circ for k in ("oc","zc","R")):
        oc,zc,R = float(circ["oc"]), float(circ["zc"]), float(circ["R"])
        th = np.linspace(-np.pi, np.pi, 241)
        xo = oc + R*np.cos(th); zo = zc + R*np.sin(th)
        fig.add_trace(go.Scatter(x=xo, y=zo, mode="lines",
                                 name="円弧", line=dict(width=2, color="#FF9500")))
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis_title="offset [m]", yaxis_title="elev [m]",
                      paper_bgcolor="#0f1115", plot_bgcolor="#0f1115")
    fig.update_xaxes(gridcolor="#2a2f3a"); fig.update_yaxes(gridcolor="#2a2f3a")
    return fig

def _gather_geo_for_section(sec_key: str) -> Dict:
    """断面ごとの地層・水位情報を集約"""
    horizons = st.session_state.get("lem_horizons", {}).get(sec_key, {})
    water    = st.session_state.get("lem_water_lines", {}).get(sec_key)
    soils_df: pd.DataFrame = st.session_state.get("lem_soils_table")
    order    = st.session_state.get("lem_layer_order", [])
    soils = soils_df.to_dict(orient="records") if isinstance(soils_df, pd.DataFrame) else []
    return {"horizons": horizons, "water_line": water, "soils": soils, "layer_order": order}

def page():
    st.title("LEM｜3 解析実行")
    st.caption(f"エンジン: **{_ENGINE}**")

    assigned = st.session_state.get("_assigned", {})
    if not assigned:
        st.info("先に『DXF取り込み・プレビュー』で断面を割当ててください。")
        return

    # 探索レンジ
    st.subheader("探索レンジ・刻み")
    col = st.columns(3)
    with col[0]:
        R_min = st.number_input("R_min [m]", value=5.0, step=0.5, format="%.1f")
        oc_sp  = st.number_input("oc スパン [m]", value=30.0, step=1.0, format="%.1f")
    with col[1]:
        R_max = st.number_input("R_max [m]", value=120.0, step=1.0, format="%.1f")
        zc_sp  = st.number_input("zc スパン [m]", value=30.0, step=1.0, format="%.1f")
    with col[2]:
        R_step = st.number_input("R_step [m]", value=2.0, step=0.5, format="%.1f")
        grid   = st.number_input("グリッド刻み [m]", value=2.0, step=0.5, format="%.1f")

    st.subheader("設計パラメータ")
    method = st.selectbox("手法", ["bishop","fellenius"], index=0)
    allow_tension = st.checkbox("引張土圧を許容", value=False)
    gammaF = st.session_state.get("lem_partial_gammaF", 1.0)
    ru_def = st.session_state.get("lem_ru_default", 0.0)

    # 対象断面
    keys = list(assigned.keys())
    targets = st.multiselect("解析する断面（複数選択可）", keys, default=keys)

    # 実行
    if st.button("解析を実行して保存", type="primary"):
        results = st.session_state.get("lem_results", {})
        prog = st.progress(0.0, text="実行中…")

        params = {
            "method": method,
            "search": dict(R_min=R_min, R_max=R_max, R_step=R_step,
                           oc_span=oc_sp, zc_span=zc_sp, grid_step=grid),
            "design": dict(allow_tension=allow_tension,
                           partial_gamma_F=float(gammaF)),
            "water": dict(default_ru=float(ru_def)),
        }

        for i, k in enumerate(targets):
            rec = assigned[k]
            oz = np.asarray(rec["oz"], float)
            geo = _gather_geo_for_section(k)
            cfg = dict(params)
            cfg["section"] = oz
            cfg["horizons"] = {name: arr.tolist() for name, arr in geo["horizons"].items()}
            cfg["water_line"] = geo["water_line"].tolist() if geo["water_line"] is not None else None
            cfg["soils"] = geo["soils"]
            cfg["layer_order"] = geo["layer_order"]

            try:
                res = _solve(cfg) or {}
                fs = float(res.get("fs", np.nan))
                circ = dict(res.get("circle", {}))
                out = {
                    "fs": fs,
                    "circle": {"oc": circ.get("oc"), "zc": circ.get("zc"), "R": circ.get("R")},
                    "meta": {"sig": _sig(oz, params), "engine": _ENGINE},
                    "params": params,
                }
                results[k] = out
                prog.progress((i+1)/max(1,len(targets)), text=f"{k}: FS={fs:.3f}")
            except Exception as e:
                results[k] = {"fs": np.nan, "circle": {}, "meta": {"error": str(e)}, "params": params}
                prog.progress((i+1)/max(1,len(targets)), text=f"{k}: ERROR")

        st.session_state.lem_results = results
        st.success("結果を保存しました。ビューワーの3Dに自動反映されます。")

    # 一覧＋個別プレビュー
    if st.session_state.get("lem_results"):
        st.subheader("結果一覧")
        rows = []
        for k, res in st.session_state.lem_results.items():
            c = res.get("circle", {})
            rows.append({"断面":k, "FS":res.get("fs"), "oc":c.get("oc"), "zc":c.get("zc"), "R":c.get("R")})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        pick = st.selectbox("個別プレビュー", list(st.session_state.lem_results.keys()))
        if pick:
            oz = np.asarray(assigned[pick]["oz"], float)
            circ = st.session_state.lem_results[pick].get("circle", {})
            st.plotly_chart(_plot_result_2d(oz, circ), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("CSVダウンロード", data=csv, file_name="lem_results.csv", mime="text/csv")

        if st.button("全ての結果をクリア"):
            st.session_state.lem_results = {}
            st.info("結果を削除しました。")
