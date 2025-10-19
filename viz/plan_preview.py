# viz/plan_preview.py
from __future__ import annotations
from typing import List, Optional
import matplotlib.pyplot as plt
from io.dxf_sections import Alignment, SectionLine


def plot_plan_preview(ax, ali: Alignment, xs_lines: List[SectionLine], highlight_id: Optional[str] = None):
    xs = [p[0] for p in ali.points]
    ys = [p[1] for p in ali.points]
    ax.plot(xs, ys, color="k", lw=2.0, label="Alignment")

    for s in xs_lines:
        x0, y0 = s.p0
        x1, y1 = s.p1
        col = "0.65"
        lw = 1.2
        z = 2
        if highlight_id and s.id == highlight_id:
            col = "0.25"
            lw = 2.0
            z = 3
        ax.plot([x0, x1], [y0, y1], color=col, lw=lw, zorder=z)
        if s.station is not None and s.foot is not None:
            ax.text(s.foot[0], s.foot[1], f"S={s.station:.1f}", fontsize=8, color="0.3")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
