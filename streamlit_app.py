# streamlit_app.py
# ------------------------------------------------------------
# 安定板３ UIに、グリッド常時ON＋Quick段階の円弧（Fs<1.3）可視化を追加。
# Plot style（Theme/Tight layout/Legend切替）を同梱。
# diagnostics が無い場合は何も描かず、既存描画のまま（後方互換）。
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from stabi_lem import Ground, Layer, Nail, Config, run_analysis

st.set_page_config(page_title="Stabi - 安定板３（Quick円弧可視化）", layout="wide")

# ====== 可視化パラメータ（A方針：グリッド常時ON／Quick限定） ======
FS_CUTOFF_FOR_ARCS = 1.3      # これ以上のFsは描かない
MAX_ARCS_TO_DRAW   = 500      # 描画上限（負荷対策）
ARC_ALPHA          = 0.22     # 通常円弧の透明度
ARC_ALPHA_MIN      = 0.60     # 最小Fs円弧のみ濃度を上げる（線幅は同じ）
GRID_LINE_ALPHA    = 0.30     # グリッド線の透明度
GRID_POINT_ALPHA   = 0.70     # グリッド点の透明度

COLOR_GRID_LINE    = "#CCCCCC"
COLOR_GRID_POINT   = "#999999"
COLOR_ARC          = "#3399FF"
COLOR_ARC_MIN      = "#0066CC"

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

# ====== Demo用の入力UI（あなたの既存UIに合わせて置換可） ======
st.sidebar.header("入力")
# 地表線（単純なプロファイル例）
x0, x1 = st.sidebar.slider("地表線X範囲", -20.0, 120.0, (-5.0, 100.0), 1.0)
slope   = st.sidebar.slider("斜面勾配（下がり）", 0.0, 1.0, 0.3, 0.05)
offset  = st.sidebar.slider("上端高さ", 0.0, 40.0, 20.0, 1.0)

xs = np.linspace(x0, x1, 300)
ys = offset - slope * (xs - xs.min())
ground = Ground.from_points(xs.tolist(), ys.tolist())

# 層・ネイル（最小限のパラメータ）
gamma = st.sidebar.slider("γ (kN/m³)", 10.0, 25.0, 18.0, 0.5)
phi   = st.sidebar.slider("φ (deg)", 10.0, 45.0, 30.0, 1.0)
c     = st.sidebar.slider("c (kPa)", 0.0, 40.0, 0.0, 1.0)
layers = [Layer(gamma=gamma, phi_deg=phi, c=c)]

nail_count = st.sidebar.slider("ネイル本数", 0, 10, 3, 1)
nails = []
for i in range(nail_count):
    nx = x0 + (i+1) * (x1 - x0) / (nail_count + 1)
    ny = ground.y_at([nx])[0] - 3.0
    nails.append(Nail(x=nx, y=ny, length=6.0, angle_deg=-20.0, bond=0.15))

# 探索グリッドと半径設定
st.sidebar.subheader("探索設定")
grid_step = st.sidebar.slider("grid_step", 1.0, 20.0, 8.0, 1.0)
cfg = Config(
    grid_xmin = x0 + 5.0,
    grid_xmax = x1 - 5.0,
    grid_ymin = min(ys) - 30.0,
    grid_ymax = max(ys) + 10.0,
    grid_step = grid_step,
    r_min = 5.0,
    r_max = max(10.0, (x1 - x0) * 1.2),
    coarse_step = 6, quick_step = 3, refine_step = 1,
    budget_coarse_s = 0.8, budget_quick_s = 1.2
)

# ====== 実行ボタン（安定板３のUIに準拠） ======
col_run, col_info = st.columns([1,3])
with col_run:
    run = st.button("▶ 補強後の計算を実行", use_container_width=True)
with col_info:
    st.markdown("**計算フロー:** Coarse → Quick → Refine（グリッド常時ON、Quick円弧のみ描画・Fs<1.3）")

# ====== 描画キャンバス ======
fig, ax = plt.subplots(figsize=(9, 6))
ax.grid(False)

# 地表線の描画（既存踏襲）
ax.plot(xs, ys, color="black", linewidth=1.8, label="Ground")

# ネイルの描画（既存踏襲：緑のbond区間イメージ）
for n in nails:
    ang = np.radians(n.angle_deg)
    x2 = n.x + n.length*np.cos(ang)
    y2 = n.y + n.length*np.sin(ang)
    ax.plot([n.x, x2], [n.y, y2], color="#2ecc71", linewidth=2.0, alpha=0.8)

# 実行
result = None
if run:
    result = run_analysis(ground, layers, nails, cfg, fs_cutoff_collect=FS_CUTOFF_FOR_ARCS)

# ====== Quick円弧の可視化（diagnostics がある時だけ実行） ======
def _draw_grid(ax, bbox, step):
    try:
        xmin, xmax, ymin, ymax = bbox
        if None in (xmin, xmax, ymin, ymax) or step in (None, 0, float('inf')):
            return
        x = xmin
        while x <= xmax + 1e-9:
            ax.plot([x, x], [ymin, ymax], color=COLOR_GRID_LINE, alpha=GRID_LINE_ALPHA, linewidth=1.0)
            x += step
        y = ymin
        while y <= ymax + 1e-9:
            ax.plot([xmin, xmax], [y, y], color=COLOR_GRID_LINE, alpha=GRID_LINE_ALPHA, linewidth=1.0)
            y += step
    except Exception:
        pass

def _draw_grid_points(ax, centers):
    try:
        if not centers:
            return
        xs_, ys_ = zip(*centers)
        ax.scatter(xs_, ys_, s=8, c=COLOR_GRID_POINT, alpha=GRID_POINT_ALPHA, linewidths=0)
    except Exception:
        pass

def _plot_arc_segment(ax, cx, cy, r, color, alpha):
    import numpy as np
    try:
        th = np.linspace(0.0, 2.0*np.pi, 128)
        xs_ = cx + r * np.cos(th)
        ys_ = cy + r * np.sin(th)
        yg  = ground.y_at(xs_)

        mask_a = ys_ <= yg
        mask_b = ys_ >= yg
        cnt_a = np.count_nonzero(mask_a)
        cnt_b = np.count_nonzero(mask_b)
        mask = mask_a if cnt_a >= cnt_b else mask_b
        if np.count_nonzero(mask) < 2:
            return

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

def _draw_quick_arcs(ax, arcs):
    arcs_f = [a for a in arcs if isinstance(a, dict) and float(a.get("fs", 1e9)) < FS_CUTOFF_FOR_ARCS]
    if not arcs_f:
        ax.text(0.02, 0.98, f"Quick段階で可視化対象の円弧なし (Fs ≥ {FS_CUTOFF_FOR_ARCS:.2f})",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, alpha=0.7)
        return

    arcs_f.sort(key=lambda a: a.get("fs", 1e9))
    arcs_draw = arcs_f[:MAX_ARCS_TO_DRAW]
    # Quick全体の最小を別に識別（フィルタ前でも良いが、ここでは後でOK）
    arc_min = arcs_f[0] if arcs_f else None

    for a in arcs_draw:
        if arc_min is not None and a is arc_min:
            continue
        _plot_arc_segment(ax, a["cx"], a["cy"], a["r"], color=COLOR_ARC, alpha=ARC_ALPHA)

    if arc_min is not None:
        _plot_arc_segment(ax, arc_min["cx"], arc_min["cy"], arc_min["r"], color=COLOR_ARC_MIN, alpha=ARC_ALPHA_MIN)

# diagnostics を取り出し、描画
diag = (result or {}).get("diagnostics", {}) if result is not None else {}
quick_arcs = diag.get("quick_arcs", [])
grid_bbox  = diag.get("grid_bbox", (None, None, None, None))
grid_step  = diag.get("grid_step", None)
grid_centers = diag.get("grid_centers_sampled", [])

# グリッドは常時ON（結果がなくても描ける範囲で描く）
_draw_grid(ax, grid_bbox, grid_step)
_draw_grid_points(ax, grid_centers)

# Quick円弧（結果がある時のみ）
if quick_arcs:
    _draw_quick_arcs(ax, quick_arcs)

# Fs表示（参考）
if result is not None:
    fsb = result.get("Fs_before", None)
    fsa = result.get("Fs_after", None)
    if fsb is not None or fsa is not None:
        txt = []
        if fsb is not None: txt.append(f"Fs_before={fsb:.3f}")
        if fsa is not None: txt.append(f"Fs_after={fsa:.3f}")
        ax.text(0.98, 0.02, " / ".join(txt), transform=ax.transAxes,
                va="bottom", ha="right", fontsize=11, alpha=0.9)

# 体裁
ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys)-40, max(ys)+15)
ax.set_xlabel("X")
ax.set_ylabel("Y")
apply_plot_style(ax, title=None, show_legend=False)

st.pyplot(fig, use_container_width=True)
