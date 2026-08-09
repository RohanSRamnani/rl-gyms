"""Channel drawing. Knows about posts and strikes; nothing about training."""
import numpy as np
from matplotlib.patches import Rectangle

from domains.content.state import (
    CYCLE_TICKS, FORMAT_NAMES, MAX_STRIKES, VIEWS_REF, ContentState,
)

FORMAT_COLOURS = ("tab:blue", "tab:olive", "tab:purple")


def draw_board(ax, state: ContentState) -> None:
    """Draw one frame of the account's cycle onto `ax`. Clears the axes first."""
    ax.clear()
    ax.set_xlim(-0.5, CYCLE_TICKS - 0.5)
    ax.set_ylim(0.0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])

    if state.restricted:
        ax.set_facecolor("#ffe5e5")
        ax.text(CYCLE_TICKS / 2, 1.25, "RESTRICTED",
                ha="center", color="tab:red", fontsize=12, fontweight="bold")
    else:
        ax.set_facecolor("white")

    median = float(np.median([p.views for p in state.history])) if state.history else 0.0
    for post in state.history:
        height = min(np.log1p(post.views) / np.log1p(VIEWS_REF), 1.0)
        ax.add_patch(Rectangle((post.day - 0.4, 0.0), 0.8, height,
                               facecolor=FORMAT_COLOURS[post.format],
                               edgecolor="0.3", linewidth=0.5))
        if median > 0.0 and post.views >= 2.0 * median:
            ax.plot([post.day], [height + 0.06], marker="*", markersize=10,
                    color="tab:orange")

    for strike in range(MAX_STRIKES):
        filled = strike < state.flag_strikes
        ax.plot([1 + strike], [1.2], marker="o", markersize=8,
                color=("tab:red" if filled else "0.85"))
    ax.text(CYCLE_TICKS - 1, 1.2,
            f"{state.followers:,.0f} followers | day {state.ticks}",
            ha="right", fontsize=9, va="center")
    for i, name in enumerate(FORMAT_NAMES):
        ax.text(0.5 + i * 9, 1.08, name, fontsize=7,
                color=FORMAT_COLOURS[i], va="center")
