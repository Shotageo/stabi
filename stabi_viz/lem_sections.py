# stabi_viz/lem_sections.py
from __future__ import annotations
import re, io, tempfile
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 既存ユーティリティ（CL縦線の自動検出）
try:
    from stabi_io.dxf_sections import list_layers, detect_section_centerline_u
except Exception:
    list_layers = None
    detect_section_centerline_u = lambda *_args, **_kw: None  # fallback

# ──────────────────────────────────────────────────────────────
# ezdxf (任意)
def _load_doc_from_bytes(data: bytes):
    try:
        import ezdxf  # 要: requirements.txt に ezdxf を追記
        return ezdxf.read(io.BytesIO(data))
    except Exception as e:
        st.warning(f"DXF 読み込みに ezdxf が必要です: {e}")
        return None

def _poly_vertices(e) -> Optional[np.ndarray]:
    try:
        if e.dxftype() == "LINE":
            p0 = e.dxf.start; p1 = e.dxf.end
            return np.array([[p0.x, p0.y], [p1.x, p1.y]], float)
        if e.dxftype() == "LWPOLYLINE":
            pts = np.array([(p[0], p[1]) for p in e.get_points("xy")], float)
            return pts
        if e.dxftype() == "SPLINE":
            pts = np.array([(p.x, p.y) for p in e.approximate(300)], float)
            return pts
    except Exception:
        return None
    return None

def _scan_layers_for_horizons(doc, *, allow_regex: Optional[str]=None) -> Dict[str, List[np.ndarray]]:
    """レイヤ→ポリライン座標（ワールドXY）の辞書"""
    out: Dict[str, List[np.ndarray]] = {}
    if doc is None:
        return out
    msp = doc.modelspace()
    # 空白区切りのクエリ＋フォールバック
    try:
        ents = list(msp.query("LINE LWPOLYLINE SPLINE"))
    except Exception:
        ents = [e for e in msp if e.dxftype() in ("LINE", "LWPOLYLINE", "SPLINE")]
    pat = re.compile(allow_regex) if allow_regex else None
    for e in ents:
        layer = e.dxf.layer
        if pat and not pat.search(layer):
            continue
        arr = _poly_vertices(e)
        if arr is None or len(arr) < 2:
            continue
        out.setdefault(layer, []).append(arr)
    return out

def _to_section_oz(seg: np.ndarray, *,
                   axis="X=offset/Y=elev",
                   off_scale=1.0, z_scale=1.0,
                   flip_o=False, flip_z=False,
                   u0: Optional[float]=None) -> np.ndarray:
    if axis.startswith("X=elev"):
        o = seg[:,1].astype(float) * off_scale
        z = seg[:,0].astype(float) * z_scale
    else:
        o = seg[:,0].astype(float) * off_scale
        z = seg[:,1].astype(float) * z_scale
    if flip_o: o *= -1.0
    if flip_z: z *= -1.0
    if u0 is not None:
        o = o - float(u0) * float(off_scale)
    idx = np.argsort(o); o=o[idx]; z=z[idx]
    if len(o) >= 2:
        uniq_o, inv = np.unique(np.round(o, 4), return_inverse=True)
        acc = np.zeros_like(uniq_o); cnt = np.zeros_like(uniq_o)
        for i, j in enumerate(inv):
            acc[j] += z[i]; cnt[j] += 1
        z = acc / np.maximum(cnt, 1)
        o = uniq_o
    return np.column_stack([o, z])

def _merge_segments(segments: List[np.ndarray], *, step: float = 0.50) -> np.ndarray:
    if not segments:
        return np.zeros((0,2), float)
    arr = np.vstack(segments)
    arr = arr[np.argsort(arr[:,0])]
    o = arr[:,0]; z = arr[:,1]
    if len(o) < 2:
        return arr
    o_new = np.arange(o.min(), o.max()+step/2, step)
    z_new = np.interp(o_new, o, z)
    return np.column_stack([o_new, z_new])

def _plot_section(oz: np.ndarray,
                  horizons: dict[str, np.ndarray] | None = None,
                  water: np.ndarray | None = None,
                  title: str = "断面プレビュー") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oz[:,0], y=oz[:,1], mode="lines",
                             name="地表/断面", line=dict(width=3, color="#FFFFFF")))
    ymin, ymax = float(np.nanmin(oz[:,1])), float(np.nanmax(oz[:,1]))
    fig.add_trace(go.Scatter(x=[0,0], y=[ymin,ymax], mode="lines",
                             name="CL", line=dict(width=1, dash="dot", color="#8AA0FF")))
    if horizons:
        for name, arr in horizons.items():
            if arr is None or len(arr)==0: continue
            fig.add_trace(go.Scatter(x=arr[:,0], y=arr[:,1], mode="lines",
                                     name=f"層:{name}", line=dict(width=2)))
    if water is not None and len(water)>0:
        fig.add_trace(go.Scatter(x=water[:,0], y=water[:,1], mode="lines",
                                 name="水位", line=dict(width=2, color="#33C3FF")))
    fig.update_layout(
        title=title, height=420, margin=dict(l=10,r=10,t=30,b=10),
        xaxis_title="offset [m]", yaxis_title="elev [m]",
        paper_bgcolor="#0f1115", plot_bgcolor="#0f1115",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(gridcolor="#2a2f3a"); fig.update_yaxes(gridcolor="#2a2f3a")
    return fig

def _auto_suggest(layer_stats: pd.DataFrame) -> pd.DataFrame:
    """レイヤ表に use/role/name を自動付与"""
    df = layer_stats.copy()
    df["use"] = False; df["role"] = "層境界"; df["name"] = df["layer"]

    # 名前でGL候補
    mask_gl = df["layer"].str.contains(r"GL|地表|地山|surface", case=False, regex=True)
    if mask_gl.any():
        df.loc[mask_gl, "use"] = True
        df.loc[mask_gl, "role"] = "地表"
    else:
        if len(df):
            i = int(df["z_med"].idxmax())
            df.loc[i, "use"] = True; df.loc[i, "role"] = "地表"

    # 砂/粘/礫/岩の自動ラベル
    for pat, name in [
        (r"砂|sand", "砂層"),
        (r"粘|clay", "粘土層"),
        (r"礫|gravel|砕石", "礫層"),
        (r"岩|tuff|rock|bedrock|基盤", "岩盤"),
    ]:
        m = df["layer"].str.contains(pat, case=False, regex=True)
        df.loc[m, "use"] = True
        df.loc[m, "role"] = "層境界"
        df.loc[m, "name"] = name
    return df

# ──────────────────────────────────────────────────────────────
def _get_assigned():
    assigned = st.session_state.get("_assigned", {})
    if not assigned:
        st.info("先に『DXF取り込み・プレビュー』で断面を割当ててください。")
        return None
    return assigned

def page():
    st.title("LEM｜1 断面プレビュー＆地層・水位入力")

    assigned = _get_assigned()
    if not assigned: return

    keys = list(assigned.keys())
    sec_key = st.selectbox("断面を選択（アップロード名）", keys, index=0)
    rec = assigned[sec_key]
    oz = np.asarray(rec["oz"], float)

    # state-bucket
    st.session_state.setdefault("lem_horizons", {})      # {sec_key: {name: Nx2}}
    st.session_state.setdefault("lem_water_lines", {})   # {sec_key: Nx2}
    st.session_state.setdefault("lem_recipe", {})        # {regex -> {"role": "...", "name": "..."}}

    horizons: dict[str, np.ndarray] = st.session_state.lem_horizons.get(sec_key, {})
    water: np.ndarray | None = st.session_state.lem_water_lines.get(sec_key)

    with st.expander("プレビュー", expanded=True):
        st.plotly_chart(_plot_section(oz, horizons, water), use_container_width=True)

    # ── A) CSV インポートは従来通り
    with st.expander("層境界（CSV）を追加／更新", expanded=False):
        st.caption("CSV: offset,elev 列。層ごとに読み込みます。")
        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1: layer_name = st.text_input("層名", value="砂層")
        with c2: up = st.file_uploader("CSV", type=["csv"], key="hcsv")
        with c3: add_btn = st.button("読み込み→追加/更新")
        with c4: clr_btn = st.button("この断面の層境界を全クリア")
        if add_btn and up is not None and layer_name.strip():
            try:
                df = pd.read_csv(up)
                arr = df[["offset","elev"]].to_numpy(float)
                arr = arr[np.argsort(arr[:,0])]
                horizons = dict(horizons)
                horizons[layer_name.strip()] = arr
                st.session_state.lem_horizons[sec_key] = horizons
                st.success(f"層境界 {layer_name} を登録しました。")
            except Exception as e:
                st.error(f"読み込み失敗: {e}")
        if clr_btn:
            st.session_state.lem_horizons[sec_key] = {}
            st.success("層境界を全クリアしました。")

    # ── B) DXF から地表・層境界をスキャン
    with st.expander("DXF から地層・地表をスキャン（ベータ）", expanded=True):
        # 断面DXFのバイトを再利用 or 改めてアップロード
        src_opt = "この断面のDXF（Step2で保存済み）" if (st.session_state.get("raw_sections_bytes", {}) and sec_key in st.session_state.raw_sections_bytes) else "DXFをアップロード"
        src = st.radio("ソース", [src_opt, "DXFをアップロード"], horizontal=True)
        dxf_bytes = None
        if src == "DXFをアップロード":
            upx = st.file_uploader("断面DXF", type=["dxf"], key="xs_dxf")
            if upx is not None:
                dxf_bytes = upx.getbuffer().tobytes()
        else:
            dxf_bytes = st.session_state.raw_sections_bytes.get(sec_key)

        axis_mode = st.selectbox("軸割り", ["X=offset / Y=elev（標準）", "X=elev / Y=offset（入替）"])
        off_scale = st.number_input("offset 倍率（mm→m は 0.001）", value=float(st.session_state.get("offset_scale_ui",1.0)),
                                    step=0.001, format="%.3f")
        z_scale   = st.number_input("elev 倍率（mm→m は 0.001）", value=float(st.session_state.get("elev_scale_ui",1.0)),
                                    step=0.001, format="%.3f")
        flip_o = st.checkbox("左右反転", value=bool(st.session_state.get("flip_o_ui", False)))
        flip_z = st.checkbox("上下反転", value=bool(st.session_state.get("flip_z_ui", False)))
        use_u0 = st.checkbox("CL縦線で offset=0 を合わせる（推奨）", value=True)
        step   = st.number_input("再サンプル間隔 step [m]", value=0.50, step=0.10, format="%.2f")
        layer_filter = st.text_input("レイヤ正規表現フィルタ（空で全て）", value="")

        scan_btn = st.button("スキャンして候補を作る", type="primary", disabled=(dxf_bytes is None))
        if scan_btn:
            doc = _load_doc_from_bytes(dxf_bytes)
            if doc is None:
                st.stop()

            # CL縦線検出（オフセット原点）
            u0 = None
            if use_u0 and detect_section_centerline_u is not None:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                        tmp.write(dxf_bytes); tmp.flush()
                        u0 = detect_section_centerline_u(tmp.name, layer_hint=None, unit_scale=1.0)
                except Exception:
                    u0 = None

            raw = _scan_layers_for_horizons(doc, allow_regex=(layer_filter or None))
            rows, previews, layer_oz = [], {}, {}
            for layer, segs in raw.items():
                # 各セグメントを (o,z) にして結合
                oz_list = []
                for sgm in segs:
                    oz_l = _to_section_oz(sgm, axis=axis_mode, off_scale=off_scale, z_scale=z_scale,
                                          flip_o=flip_o, flip_z=flip_z, u0=u0)
                    if len(oz_l) >= 2: oz_list.append(oz_l)
                if not oz_list: 
                    continue
                merged = _merge_segments(oz_list, step=float(step))
                layer_oz[layer] = merged
                span = float(merged[:,0].max() - merged[:,0].min()) if len(merged)>=2 else 0.0
                z_med = float(np.median(merged[:,1])) if len(merged) else np.nan
                rows.append({"layer": layer, "pts": len(merged), "span": span, "z_med": z_med})

            if not rows:
                st.warning("候補が見つかりませんでした。フィルタや倍率を見直してください。")
            else:
                df = pd.DataFrame(rows).sort_values(["z_med","span","pts"], ascending=[False,False,False], ignore_index=True)
                st.session_state["_scan_layers_df"] = df
                st.session_state["_scan_layers_oz"] = layer_oz
                st.success(f"{len(df)} レイヤ候補を見つけました。下でマッピングしてください。")

        # 候補→マッピング
        if st.session_state.get("_scan_layers_df") is not None:
            df = st.session_state["_scan_layers_df"].copy()
            st.caption("右端で『自動提案』→表で微修正→『選択を断面に反映』")
            cL, cR = st.columns([3,1])
            with cR:
                if st.button("自動提案"):
                    st.session_state["_map_table"] = _auto_suggest(df)

            if "_map_table" not in st.session_state:
                df["use"] = False; df["role"] = "層境界"; df["name"] = df["layer"]
                st.session_state["_map_table"] = df[["use","role","name","layer","pts","span","z_med"]]

            ed = st.data_editor(
                st.session_state["_map_table"],
                use_container_width=True, hide_index=True,
                column_config={
                    "use": st.column_config.CheckboxColumn("使う"),
                    "role": st.column_config.SelectboxColumn("役割", options=["地表","層境界"]),
                    "name": st.column_config.TextColumn("名称（表示名）"),
                    "layer": st.column_config.TextColumn("レイヤ名", disabled=True),
                    "pts": st.column_config.NumberColumn("点数", disabled=True),
                    "span": st.column_config.NumberColumn("スパン[m]", disabled=True, format="%.2f"),
                    "z_med": st.column_config.NumberColumn("中位標高[m]", disabled=True, format="%.2f"),
                },
                num_rows="dynamic"
            )
            st.session_state["_map_table"] = ed

            # プレビュー（選択だけ重ねる）
            sel = ed[ed["use"]==True]
            if len(sel):
                over = {}
                for _, r in sel.iterrows():
                    arr = st.session_state["_scan_layers_oz"].get(r["layer"])
                    if arr is not None:
                        over[r["name"]] = arr
                st.plotly_chart(_plot_section(oz, over, water, "候補の重ね描きプレビュー"), use_container_width=True)

            c1,c2,c3 = st.columns([1.5,1,1])
            with c1:
                if st.button("選択を断面に反映", type="primary"):
                    new_h: Dict[str, np.ndarray] = {}
                    for _, r in ed.iterrows():
                        if not r.get("use"): continue
                        name = str(r.get("name") or r.get("layer"))
                        arr  = st.session_state["_scan_layers_oz"].get(r["layer"])
                        if arr is not None:
                            new_h[name] = arr
                    st.session_state.lem_horizons[sec_key] = new_h
                    st.success(f"{len(new_h)} 本のラインを断面に保存しました。")
            with c2:
                if st.button("このマッピングをレシピ保存"):
                    recipe = { r["layer"]: {"role":r["role"], "name":r["name"]} for _,r in ed.iterrows() if r["use"] }
                    st.session_state.lem_recipe = recipe
                    st.success("レシピを保存しました。次の断面で『レシピ適用』できます。")
            with c3:
                if st.button("レシピ適用（名前一致ベース）"):
                    recp: Dict = st.session_state.get("lem_recipe", {})
                    new_h: Dict[str,np.ndarray] = {}
                    for lay, m in recp.items():
                        arr = st.session_state["_scan_layers_oz"].get(lay)
                        if arr is not None:
                            new_h[m.get("name",lay)] = arr
                    if new_h:
                        st.session_state.lem_horizons[sec_key] = new_h
                        st.success(f"レシピから {len(new_h)} 本を適用しました。")

    # ── 水位線（従来）
    with st.expander("水位線を追加／更新（任意）", expanded=False):
        upw = st.file_uploader("水位線CSV（offset,elev）", type=["csv"], key="wcsv")
        c1,c2 = st.columns([1,1])
        if c1.button("水位線を設定") and upw is not None:
            try:
                dfw = pd.read_csv(upw)
                arr = dfw[["offset","elev"]].to_numpy(float)
                arr = arr[np.argsort(arr[:,0])]
                st.session_state.lem_water_lines[sec_key] = arr
                st.success("水位線を設定しました。")
            except Exception as e:
                st.error(f"読み込み失敗: {e}")
        if c2.button("水位線をクリア"):
            st.session_state.lem_water_lines.pop(sec_key, None)
            st.success("水位線をクリアしました。")

    with st.expander("確認プレビュー（最終）", expanded=False):
        horizons = st.session_state.lem_horizons.get(sec_key, {})
        water = st.session_state.lem_water_lines.get(sec_key)
        st.plotly_chart(_plot_section(oz, horizons, water), use_container_width=True)
