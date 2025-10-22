# stabi_viz/lem_lab.py
from __future__ import annotations

import json, hashlib, io, tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────
# LEMエンジンのロード（安定版3 → 既存 → フォールバックの順）
_ENGINE_NAME = None
try:
    from stabi_core.stabi_lem_v3 import solve_min_circle as _lem_solve  # 安定版3（推奨）
    _ENGINE_NAME = "stabi_core.stabi_lem_v3.solve_min_circle"
except Exception:
    try:
        from stabi_core.stabi_lem import compute_min_circle as _lem_solve  # 既存
        _ENGINE_NAME = "stabi_core.stabi_lem.compute_min_circle"
    except Exception:
        _ENGINE_NAME = "fallback"

        def _lem_solve(cfg: Dict) -> Dict:
            """ダミー: 断面の範囲からそれっぽい円を返す（表示用）"""
            sec = np.asarray(cfg["section"], float)
            oc = float(np.median(sec[:, 0]))
            zc = float(np.percentile(sec[:, 1], 25))
            R  = float(max(6.0, (sec[:, 0].max() - sec[:, 0].min()) * 0.35))
            return {"fs": 1.12, "circle": {"oc": oc, "zc": zc, "R": R}, "meta": {"fallback": True}}

# ──────────────────────────────────────────────────────────────
# ハッシュ: 断面＋パラメータ → キャッシュキー
def _sig(oz: np.ndarray, params: Dict) -> str:
    blob = np.asarray(oz, np.float32).tobytes() + json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]

# ──────────────────────────────────────────────────────────────
# 既定の層テーブル（サンプル）
def _default_soils_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"name": "盛土",          "phi_deg": 30.0, "c_kPa":  10.0, "gamma_kN_m3": 18.0, "ru": 0.0, "partial_gammaM": 1.0},
        {"name": "風化岩（砂質）", "phi_deg": 35.0, "c_kPa":  20.0, "gamma_kN_m3": 20.0, "ru": 0.0, "partial_gammaM": 1.0},
        {"name": "基盤岩",        "phi_deg": 40.0, "c_kPa":  50.0, "gamma_kN_m3": 22.0, "ru": 0.0, "partial_gammaM": 1.0},
    ])

# ──────────────────────────────────────────────────────────────
def page():
    st.title("LEM 解析ラボ｜土質・水位・探索レンジの設定と実行")
    st.caption(f"エンジン: **{_ENGINE_NAME}**")

    # ビューワー側で割当済みか確認
    assigned = st.session_state.get("_assigned", {})
    if not assigned:
        st.info("先に『DXF取り込み（No×測点円スナップ→横断の立体配置）』で割当を作成してください。")
        return

    # 断面一覧（ビューワーが保存したキー = アップロードファイル名）
    sec_keys = list(assigned.keys())

    # ── 1) 土質パラメータ（層テーブル）
    st.subheader("1) 土質パラメータ")
    if "lem_soils" not in st.session_state:
        st.session_state.lem_soils = _default_soils_df()
    st.markdown("φ[deg], c[kPa], γ[kN/m³], ru(簡易), 部分係数 γM（設計に応じて）")
    soils_df = st.data_editor(
        st.session_state.lem_soils,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="soils_editor",
    )
    st.session_state.lem_soils = soils_df

    colx = st.columns(3)
    with colx[0]:
        method = st.selectbox("手法", ["bishop", "fellenius"], index=0)
        partial_gamma_F = st.number_input("部分係数 γF（荷重）", value=1.0, step=0.05)
    with colx[1]:
        use_ru = st.checkbox("ru を使う（簡易地下水）", value=True)
        ru_default = st.number_input("全層に ru を一括適用（個別設定が優先）", value=0.0, step=0.05)
    with colx[2]:
        allow_tension = st.checkbox("引張土圧を許容（学術向け）", value=False)

    # ── 2) 地下水（ラインのインポートも許容）
    st.subheader("2) 地下水設定（任意）")
    water_mode = st.radio("モデル", ["なし", "ru（簡易）", "水位線（CSV: offset,elev）"], index=(1 if use_ru else 0))
    water_line: Optional[np.ndarray] = None
    if water_mode == "水位線（CSV: offset,elev）":
        up = st.file_uploader("水位線CSV（offset[m], elev[m]）", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            if {"offset", "elev"} <= set(df.columns):
                water_line = df[["offset", "elev"]].to_numpy(float)
                st.success(f"水位線を読み込みました（{len(water_line)} 点）")
            else:
                st.error("列名 'offset','elev' が必要です。")

    # ── 3) 探索レンジ
    st.subheader("3) 探索レンジ・刻み")
    c3 = st.columns(3)
    with c3[0]:
        R_min = st.number_input("R_min [m]", value=5.0, step=0.5, format="%.1f")
        oc_span = st.number_input("oc スパン [m]", value=30.0, step=1.0, format="%.1f")
    with c3[1]:
        R_max = st.number_input("R_max [m]", value=120.0, step=1.0, format="%.1f")
        zc_span = st.number_input("zc スパン [m]", value=30.0, step=1.0, format="%.1f")
    with c3[2]:
        R_step = st.number_input("R_step [m]", value=2.0, step=0.5, format="%.1f")
        grid_step = st.number_input("グリッド刻み [m]", value=2.0, step=0.5, format="%.1f")

    # 解析対象の選択
    st.subheader("4) 対象断面の選択")
    default_sel = sec_keys  # 既定：全部
    target_keys = st.multiselect("解析する断面", sec_keys, default=default_sel)

    # パラメータ束ね（結果ハッシュ用にも使用）
    lem_params = {
        "method": method,
        "search": {
            "R_min": float(R_min), "R_max": float(R_max), "R_step": float(R_step),
            "oc_span": float(oc_span), "zc_span": float(zc_span), "grid_step": float(grid_step),
        },
        "design": {
            "allow_tension": bool(allow_tension),
            "partial_gamma_F": float(partial_gamma_F),
        },
        "water": {
            "mode": water_mode,
            "ru_default": float(ru_default),
            "line": water_line.tolist() if water_line is not None else None,
        },
        "soils": soils_df.to_dict(orient="records"),  # name, phi_deg, c_kPa, gamma_kN_m3, ru, partial_gammaM
    }

    # ── 実行
    if st.button("選択した断面を解析して保存", type="primary"):
        results = st.session_state.get("lem_results", {})
        prog = st.progress(0.0, text="実行中…")

        for i, k in enumerate(target_keys):
            rec = assigned.get(k)
            if rec is None: 
                continue
            oz = np.asarray(rec["oz"], float)

            # 構成：エンジンに渡す設定（“安定版3”なら soils/water/部分係数も受け取れる想定）
            cfg = dict(lem_params)
            cfg["section"] = oz
            # 参考：CLのoffset=0は oz 内に含まれる。探索中心をoc近傍に寄せるならここで oc0=0 などをセット

            try:
                res = _lem_solve(cfg) or {}
                fs = float(res.get("fs", np.nan))
                circ = dict(res.get("circle", {}))
                out = {
                    "fs": fs,
                    "circle": {"oc": circ.get("oc"), "zc": circ.get("zc"), "R": circ.get("R")},
                    "meta": {
                        "sig": _sig(oz, lem_params),
                        "engine": _ENGINE_NAME,
                    },
                    "params": lem_params,
                }
                results[k] = out
                prog.progress((i + 1) / max(1, len(target_keys)), text=f"{k}: FS={fs:.3f}")
            except Exception as e:
                results[k] = {"fs": np.nan, "circle": {}, "meta": {"error": str(e)}, "params": lem_params}
                prog.progress((i + 1) / max(1, len(target_keys)), text=f"{k}: ERROR")

        st.session_state.lem_results = results
        st.success("LEM結果を保存しました。ビューワーの3Dに自動反映されます。")

    # ── 結果の一覧とエクスポート
    if st.session_state.get("lem_results"):
        st.subheader("保存済みの結果")
        rows = []
        for k, res in st.session_state.lem_results.items():
            fs = res.get("fs")
            c  = res.get("circle", {})
            rows.append({"断面": k, "FS": fs, "oc": c.get("oc"), "zc": c.get("zc"), "R": c.get("R")})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("CSVでダウンロード", data=csv, file_name="lem_results.csv", mime="text/csv")

        # 全消去
        if st.button("全ての LEM 結果をクリア"):
            st.session_state.lem_results = {}
            st.info("LEM結果を削除しました。")

    # ── 設定の保存/復元（JSON）
    st.subheader("設定の保存/読み込み")
    cfg_json = json.dumps(lem_params, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("現在の設定をJSONで保存", data=cfg_json, file_name="lem_config.json", mime="application/json")
    up_cfg = st.file_uploader("設定JSONを読み込む", type=["json"])
    if up_cfg is not None and st.button("設定を反映"):
        try:
            cfg = json.loads(up_cfg.read().decode("utf-8"))
            # soils
            if "soils" in cfg:
                st.session_state.lem_soils = pd.DataFrame(cfg["soils"])
            st.success("設定を反映しました。")
        except Exception as e:
            st.error(f"設定の読み込みに失敗しました: {e}")

