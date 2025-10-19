# streamlit_app.py
# ------------------------------------------------------------
# 安定板３ ベースライン（復旧版）
# - ページ分割UI（1:地表 2:地層 3:補強 4:設定 5:解析）
# - Page5 に「▶ 補強後の計算を実行」ボタン
# - ネイルは地盤方向に描画（緑）、Fs_after = ΣT / Σ(W sinα) 簡易連成
# - Audit（全センター表示）は既定OFF、描画は安全ガード
# - Plot style ブロック同梱
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from stabi_lem import Ground, Layer, Nail, Config, run_analysis

st.set_page_config(page_title="Stabi - 安定板３", layout="wide")

# ==== Plot style（Theme/Tight layout/Legend切替） ====
def apply_plot_style(ax, title=None, show_legend=False):
    if title:
        ax.set_title(title)
    if show_legend:
        leg = ax.legend(loc="best")
        if leg is not None:
            try:
                leg.set_draggable(True)
            except Exception:
                pass
    try:
        ax.figure.tight_layout()
    except Exception:
        pass

# ==== セッション初期化（ガード付き） ====
if "ground_xs" not in st.session_state:
    st.session_state.ground_xs = np.linspace(-5, 100, 300)
if "ground_slope" not in st.session_state:
    st.session_state.ground_slope = 0.3
if "ground_offset" not in st.session_state:
    st.session_state.ground_offset = 20.0

if "layers" not in st.session_state:
    st.session_state.layers = [Layer(gamma=18.0, phi_deg=30.0, c=0.0)]
if "nails" not in st.session_state:
    st.session_state.nails = []
if "cfg" not in st.session_state:
    xs = st.session_state.ground_xs
    ys = st.session_state.ground_offset - st.session_state.ground_slope * (xs - xs.min())
    st.session_state.cfg = Config(
        grid_xmin=float(xs.min()+5), grid_xmax=float(xs.max()-5),
        grid_ymin=float(ys.min()-30), grid_ymax=float(ys.max()+10),
        grid_step=8.0,
        r_min=5.0, r_max=max(10.0, (xs.max()-xs.min())*1.2),
        coarse_step=6, quick_step=3, refine_step=1,
        budget_coarse_s=0.8, budget_quick_s=1.2
    )

# ==== ページ分割 ====
st.sidebar.header("メニュー")
page = st.sidebar.radio("ページを選択", [
    "1) 地表", "2) 地層", "3) 補強", "4) 設定", "5) 解析"
], index=4)

# ==== ページ1：地表 ====
if page.startswith("1"):
    st.header("地表（プロファイル）")
    x0, x1 = st.slider("地表線X範囲", -50.0, 200.0, (-5.0, 100.0), 1.0)
    slope  = st.slider("斜面勾配（下がり）", 0.0, 1.5, st.session_state.ground_slope, 0.05)
    offset = st.slider("上端高さ", -20.0, 60.0, st.session_state.ground_offset, 1.0)

    xs = np.linspace(x0, x1, 300)
    ys = offset - slope * (xs - xs.min())
    st.session_state.ground_xs = xs
    st.session_state.ground_slope = slope
    st.session_state.ground_offset = offset

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot(xs, ys, color="black")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    apply_plot_style(ax, title=None, show_legend=False)
    st.pyplot(fig, use_container_width=True)
# >>> DXF_PLAN_PREVIEW START（ここから追記：DXFの中心線＋横断群プレビュー。既存UI/計算は不変更） >>>
with st.expander("🗺️ DXF：中心線＋横断群のプレビュー（実験）", expanded=False):
    st.caption("DXFから Alignment（中心線形）と XS*（横断法線）を読み込み、平面図に重ねて表示します。解析・cfgには影響しません。")
    dxf_file = st.file_uploader("DXFファイルを選択", type=["dxf"], key="__dxf_plan__")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        layer_align = st.text_input("中心線レイヤ名ヒント", value="ALIGN")
    with colB:
        layer_xs = st.text_input("横断レイヤ名（接頭辞OK）", value="XS")
    with colC:
        highlight = st.text_input("強調表示する横断ID（任意）", value="")

    try:
        if dxf_file is not None:
            # 依存ライブラリはローカルにのみ要求
            try:
                import tempfile, os
                from io.dxf_sections import load_alignment, load_sections, attach_stationing
                from viz.plan_preview import plot_plan_preview
            except ImportError as e:
                st.error("必要なモジュールが見つかりません。`pip install ezdxf` を実行してください。")
                st.stop()

            # 一時保存して ezdxf に渡す
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tf:
                tf.write(dxf_file.read())
                dxf_path = tf.name

            try:
                ali = load_alignment(dxf_path, layer_hint=layer_align.strip() or None)
                xs_raw = load_sections(dxf_path, layer_filter=layer_xs.strip() or "XS")
                xs = attach_stationing(xs_raw, ali)
                if not xs:
                    st.warning("横断レイヤ（XS*）が見つかりませんでした。レイヤ名を確認してください。")
                else:
                    st.success(f"読み込み成功：Alignment={ali.length:.1f} m、横断本数={len(xs)}")
                    fig2, ax2 = plt.subplots(figsize=(8.6, 6.0))
                    plot_plan_preview(ax2, ali, xs, highlight_id=(highlight or None))
                    st.pyplot(fig2); plt.close(fig2)
                    st.caption("※ ここは“プレビューのみ”。解析・cfgは変更しません。")
            finally:
                try:
                    os.unlink(dxf_path)
                except Exception:
                    pass
        else:
            st.info("DXFを選択すると平面図プレビューが表示されます。レイヤ名は任意（既定：ALIGN/XS）。")
    except Exception as e:
        st.error(f"DXFプレビューでエラーが発生しました：{e}")
# <<< DXF_PLAN_PREVIEW END（ここまで追記） <<<

# ==== ページ2：地層 ====
elif page.startswith("2"):
    st.header("地層（代表値）")
    cur = st.session_state.layers[0]
    gamma = st.slider("γ (kN/m³)", 10.0, 25.0, float(cur.gamma), 0.5)
    phi   = st.slider("φ (deg)",   10.0, 45.0, float(cur.phi_deg), 1.0)
    c     = st.slider("c (kPa)",    0.0, 40.0, float(cur.c), 1.0)
    st.session_state.layers = [Layer(gamma=gamma, phi_deg=phi, c=c)]
    st.success("更新しました。")

# ==== ページ3：補強 ====
elif page.startswith("3"):
    st.header("補強（ソイルネイル）")
    cols = st.columns([1,1,1,1,1])
    with cols[0]:
        n_count = st.number_input("本数", min_value=0, max_value=50, value=max(0, len(st.session_state.nails)), step=1)
    with cols[1]:
        length = st.number_input("長さ", min_value=0.5, max_value=30.0, value=6.0, step=0.5)
    with cols[2]:
        angle  = st.number_input("角度(deg)", min_value=-90.0, max_value=90.0, value=-20.0, step=1.0)
    with cols[3]:
        bond   = st.number_input("bond(簡略抵抗)", min_value=0.0, max_value=5.0, value=0.15, step=0.05)
    with cols[4]:
        head_depth  = st.number_input("頭部埋込み（地表下）", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

    xs = st.session_state.ground_xs
    ys = st.session_state.ground_offset - st.session_state.ground_slope * (xs - xs.min())
    ground_tmp = Ground.from_points(xs.tolist(), ys.tolist())

    nails = []
    if n_count > 0:
        for i in range(n_count):
            nx = xs.min() + (i+1) * (xs.max() - xs.min()) / (n_count + 1)
            ny = ground_tmp.y_at([nx])[0] - head_depth
            nails.append(Nail(x=float(nx), y=float(ny), length=float(length), angle_deg=float(angle), bond=float(bond)))
    st.session_state.nails = nails

    # 簡易プレビュー
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot(xs, ys, color="black")
    for n in nails:
        ang = np.radians(n.angle_deg)
        x2 = n.x + n.length*np.cos(ang)
        y2 = n.y + n.length*np.sin(ang)
        ax.plot([n.x, x2], [n.y, y2], color="#2ecc71", linewidth=2.0, alpha=0.85)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    apply_plot_style(ax, title=None, show_legend=False)
    st.pyplot(fig, use_container_width=True)

# ==== ページ4：設定 ====
elif page.startswith("4"):
    st.header("探索グリッド／半径 設定")
    cfg = st.session_state.cfg
    x0 = float(st.number_input("grid_xmin", value=cfg.grid_xmin))
    x1 = float(st.number_input("grid_xmax", value=cfg.grid_xmax))
    y0 = float(st.number_input("grid_ymin", value=cfg.grid_ymin))
    y1 = float(st.number_input("grid_ymax", value=cfg.grid_ymax))
    step = float(st.number_input("grid_step", value=cfg.grid_step))
    rmin = float(st.number_input("r_min", value=cfg.r_min))
    rmax = float(st.number_input("r_max", value=cfg.r_max))
    st.session_state.cfg = Config(
        grid_xmin=x0, grid_xmax=x1, grid_ymin=y0, grid_ymax=y1, grid_step=step,
        r_min=rmin, r_max=rmax,
        coarse_step=cfg.coarse_step, quick_step=cfg.quick_step, refine_step=cfg.refine_step,
        budget_coarse_s=cfg.budget_coarse_s, budget_quick_s=cfg.budget_quick_s
    )
    st.success("更新しました。")

# ==== ページ5：解析 ====
else:
    st.header("解析")
    xs = st.session_state.ground_xs
    ys = st.session_state.ground_offset - st.session_state.ground_slope * (xs - xs.min())
    ground = Ground.from_points(xs.tolist(), ys.tolist())
    layers = st.session_state.layers
    nails  = st.session_state.nails
    cfg    = st.session_state.cfg

    col_run, col_info = st.columns([1,3])
    with col_run:
        run = st.button("▶ 補強後の計算を実行", use_container_width=True)
    with col_info:
        st.markdown("**計算フロー:** Coarse → Quick → Refine（Audit既定OFF）")

    # 図
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(False)

    # 地表線
    ax.plot(xs, ys, color="black", linewidth=1.8, label="Ground")

    # ネイル（地盤向き）
    for n in nails:
        ang = np.radians(n.angle_deg)
        x2 = n.x + n.length*np.cos(ang)
        y2 = n.y + n.length*np.sin(ang)
        ax.plot([n.x, x2], [n.y, y2], color="#2ecc71", linewidth=2.0, alpha=0.85)

    result = None
    if run:
        # 本体解析（診断等の追加無し）
        result = run_analysis(ground, layers, nails, cfg)

    # Fs表示
    if result is not None:
        fsb = result.get("Fs_before", None)
        fsa = result.get("Fs_after", None)
        txt = []
        if fsb is not None: txt.append(f"Fs_before={fsb:.3f}")
        if fsa is not None: txt.append(f"Fs_after={fsa:.3f}")
        if txt:
            ax.text(0.98, 0.02, " / ".join(txt), transform=ax.transAxes,
                    va="bottom", ha="right", fontsize=11, alpha=0.9)

    ax.set_xlabel("X"); ax.set_ylabel("Y")
    apply_plot_style(ax, title=None, show_legend=False)
    st.pyplot(fig, use_container_width=True)
