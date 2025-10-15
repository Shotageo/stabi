# streamlit_app.py
# ------------------------------------------------------------
# 既存の「ページ分割UI」を維持したまま、Page5で
# グリッド常時ON＋Quick円弧（Fs<1.3）の可視化を追加。
# Plot style ブロック同梱。アスペクト等倍。
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from stabi_lem import Ground, Layer, Nail, Config, run_analysis

st.set_page_config(page_title="Stabi - 安定板３", layout="wide")

# ====== Plot style（Theme/Tight layout/Legend切替） ======
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

# ====== 内部可視化パラメータ（UI非公開） ======
_FS_CUTOFF_FOR_ARCS = 1.3
_MAX_ARCS_TO_DRAW   = 500
_ARC_ALPHA          = 0.22
_ARC_ALPHA_MIN      = 0.60
_COLOR_GRID_LINE    = "#CCCCCC"
_COLOR_GRID_POINT   = "#999999"
_COLOR_ARC          = "#3399FF"
_COLOR_ARC_MIN      = "#0066CC"

def _draw_grid(ax, bbox, step):
    try:
        xmin, xmax, ymin, ymax = bbox
        if None in (xmin, xmax, ymin, ymax) or not step or not np.isfinite(step):
            return
        x = xmin
        while x <= xmax + 1e-9:
            ax.plot([x, x], [ymin, ymax], color=_COLOR_GRID_LINE, alpha=0.30, linewidth=1.0)
            x += step
        y = ymin
        while y <= ymax + 1e-9:
            ax.plot([xmin, xmax], [y, y], color=_COLOR_GRID_LINE, alpha=0.30, linewidth=1.0)
            y += step
    except Exception:
        pass

def _draw_grid_points(ax, centers):
    try:
        if not centers: return
        xs_, ys_ = zip(*centers)
        ax.scatter(xs_, ys_, s=8, c=_COLOR_GRID_POINT, alpha=0.70, linewidths=0)
    except Exception:
        pass

def _plot_arc_segment(ax, ground, cx, cy, r, color, alpha):
    try:
        th = np.linspace(0.0, 2.0*np.pi, 128)
        xs_ = cx + r * np.cos(th); ys_ = cy + r * np.sin(th)
        yg  = ground.y_at(xs_)
        mask = ys_ <= yg
        if np.count_nonzero(mask) < 2:
            mask_alt = ys_ >= yg
            if np.count_nonzero(mask_alt) < 2: return
            mask = mask_alt
        idx = np.where(mask)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        seg_starts = [0] + (splits + 1).tolist()
        seg_ends   = splits.tolist() + [len(idx) - 1]
        for s, e in zip(seg_starts, seg_ends):
            seg = idx[s:e+1]
            if len(seg) >= 2:
                ax.plot(xs_[seg], ys_[seg], color=color, alpha=alpha, linewidth=1.2)
    except Exception:
        pass

def _draw_quick_arcs(ax, ground, quick_arcs):
    arcs_f = [a for a in quick_arcs if isinstance(a, dict) and float(a.get("fs", 1e9)) < _FS_CUTOFF_FOR_ARCS]
    if not arcs_f:
        ax.text(0.02, 0.98, f"Quick段階で可視化対象の円弧なし (Fs ≥ {_FS_CUTOFF_FOR_ARCS:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, alpha=0.7)
        return
    arcs_f.sort(key=lambda a: a.get("fs", 1e9))
    arcs_draw = arcs_f[:_MAX_ARCS_TO_DRAW]
    arc_min = arcs_f[0]
    for a in arcs_draw:
        if a is arc_min: continue
        _plot_arc_segment(ax, ground, a["cx"], a["cy"], a["r"], _COLOR_ARC, _ARC_ALPHA)
    _plot_arc_segment(ax, ground, arc_min["cx"], arc_min["cy"], arc_min["r"], _COLOR_ARC_MIN, _ARC_ALPHA_MIN)

# ====== サイドバー：ページ分割 ======
st.sidebar.header("メニュー")
page = st.sidebar.radio("ページを選択", ["1) 地表", "2) 地層", "3) 補強", "4) 設定", "5) 解析・可視化"], index=4)

# ====== 共通データの保持（セッション状態） ======
if "ground_xs" not in st.session_state:
    st.session_state.ground_xs = np.linspace(-5, 100, 300)
    st.session_state.ground_slope = 0.3
    st.session_state.ground_offset = 20.0

if "layers" not in st.session_state:
    st.session_state.layers = [Layer(gamma=18.0, phi_deg=30.0, c=0.0)]

if "nails" not in st.session_state:
    st.session_state.nails = []

if "cfg" not in st.session_state:
    st.session_state.cfg = Config(
        grid_xmin=0.0, grid_xmax=95.0,
        grid_ymin=-30.0, grid_ymax=30.0,
        grid_step=8.0,
        r_min=5.0, r_max=120.0,
        coarse_step=6, quick_step=3, refine_step=1,
        budget_coarse_s=0.8, budget_quick_s=1.2
    )

# ====== ページ1：地表 ======
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
    st.pyplot(fig, use_container_width=True)

# ====== ページ2：地層 ======
elif page.startswith("2"):
    st.header("地層（代表値）")
    gamma = st.slider("γ (kN/m³)", 10.0, 25.0, st.session_state.layers[0].gamma, 0.5)
    phi   = st.slider("φ (deg)",   10.0, 45.0, st.session_state.layers[0].phi_deg, 1.0)
    c     = st.slider("c (kPa)",    0.0, 40.0, st.session_state.layers[0].c, 1.0)
    st.session_state.layers = [Layer(gamma=gamma, phi_deg=phi, c=c)]
    st.success("更新しました。")

# ====== ページ3：補強 ======
elif page.startswith("3"):
    st.header("補強（ネイル）")
    cols = st.columns([1,1,1,1,1])
    with cols[0]:
        n_count = st.number_input("本数", min_value=0, max_value=30, value=max(0, len(st.session_state.nails)), step=1)
    with cols[1]:
        length = st.number_input("長さ", min_value=0.5, max_value=30.0, value=6.0, step=0.5)
    with cols[2]:
        angle  = st.number_input("角度(deg)", min_value=-90.0, max_value=90.0, value=-20.0, step=1.0)
    with cols[3]:
        bond   = st.number_input("bond", min_value=0.0, max_value=5.0, value=0.15, step=0.05)
    with cols[4]:
        depth  = st.number_input("頭部埋込み深さ（地表下）", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

    xs = st.session_state.ground_xs
    # ground は仮にページ内で一時生成
    ys = st.session_state.ground_offset - st.session_state.ground_slope * (xs - xs.min())
    gtmp = Ground.from_points(xs.tolist(), ys.tolist())

    nails = []
    if n_count > 0:
        for i in range(n_count):
            nx = xs.min() + (i+1) * (xs.max() - xs.min()) / (n_count + 1)
            ny = gtmp.y_at([nx])[0] - depth
            nails.append(Nail(x=float(nx), y=float(ny), length=float(length), angle_deg=float(angle), bond=float(bond)))
    st.session_state.nails = nails

    st.info(f"設定中のネイル：{len(nails)} 本")

# ====== ページ4：設定（探索グリッドなど） ======
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

# ====== ページ5：解析・可視化 ======
else:
    st.header("解析・可視化")
    # 入力を合成
    xs = st.session_state.ground_xs
    ys = st.session_state.ground_offset - st.session_state.ground_slope * (xs - xs.min())
    ground = Ground.from_points(xs.tolist(), ys.tolist())
    layers = st.session_state.layers
    nails  = st.session_state.nails
    cfg    = st.session_state.cfg

    # 実行ボタン
    col_run, col_info = st.columns([1,3])
    with col_run:
        run = st.button("▶ 補強後の計算を実行", use_container_width=True)
    with col_info:
        st.markdown("**計算フロー:** Coarse → Quick → Refine（グリッド常時ON、Quick円弧のみ・Fs<1.3）")

    # キャンバス
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(False)

    # 地表線（先に描いておくと基準が分かりやすい）
    ax.plot(xs, ys, color="black", linewidth=1.8, label="Ground")

    # ネイル
    for n in nails:
        ang = np.radians(n.angle_deg)
        x2 = n.x + n.length * np.cos(ang)
        y2 = n.y + n.length * np.sin(ang)
        ax.plot([n.x, x2], [n.y, y2], color="#2ecc71", linewidth=2.0, alpha=0.85)

    result = None
    if run:
        result = run_analysis(ground, layers, nails, cfg, fs_cutoff_collect=_FS_CUTOFF_FOR_ARCS)

    # 診断に基づく描画（グリッドは常時）
    diag = (result or {}).get("diagnostics", {}) if result is not None else {}
    quick_arcs = diag.get("quick_arcs", [])
    grid_bbox  = diag.get("grid_bbox", (cfg.grid_xmin, cfg.grid_xmax, cfg.grid_ymin, cfg.grid_ymax))
    grid_step  = diag.get("grid_step", cfg.grid_step)
    grid_centers = diag.get("grid_centers_sampled", [])

    _draw_grid(ax, grid_bbox, grid_step)
    _draw_grid_points(ax, grid_centers)
    if quick_arcs:
        _draw_quick_arcs(ax, ground, quick_arcs)

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
