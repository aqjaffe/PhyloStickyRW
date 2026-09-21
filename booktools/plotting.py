"""Plotting helpers for the open-book / spider space visualization.

This module draws the *space* side of the spider figures: a k-spider as ``K``
arrows emanating from the origin (the spine), with a marker at the running
Frechet mean -- a coloured dot out on the active leg, or a black dot at the
origin when the mean is sticky on the spine.

The *random-walk* side of every figure (spider and BHV alike) is drawn by the
shared ``bhvtools.plotting.draw_folded_walks``, so all figures use one common
folded-walk style. Pass the same ``leg_colors`` to both drawers so a leg's walk
colour matches its dot colour.

Only ``numpy`` is imported at module level; functions draw onto a Matplotlib
``Axes`` supplied by the caller, so importing ``booktools`` stays dependency-free
(this submodule is not imported by ``booktools/__init__.py``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Matplotlib default cycle C0/C1/C2; kept local so booktools does not import bhvtools.
DEFAULT_LEG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def spider_directions(K: int, length: float = 1.0) -> np.ndarray:
    """Unit endpoints of ``K`` evenly spaced legs (leg 0 points up). Shape ``(K, 2)``."""
    ang = np.deg2rad(90.0 - np.arange(K) * 360.0 / K)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1) * length


def draw_spider(ax, dirs, *, page: Optional[int] = None, frac: float = 0.0,
                sticky: bool = True, leg_colors: Optional[Sequence[str]] = None,
                title: Optional[str] = None, title_color: Optional[str] = None,
                title_fontsize: float = 14.0,
                arrow_color: str = "black", dot_size: float = 55.0,
                lim: float = 1.1):
    """Draw a k-spider with a marker at the mean.

    Parameters
    ----------
    dirs : (K, 2) leg endpoints (see :func:`spider_directions`).
    page : active leg index, or ``None`` when sticky.
    frac : radial position of the dot along the active leg, in ``[0, 1]``
        (typically the leg's folded-walk height divided by the global maximum).
    sticky : if ``True`` the dot is a black dot at the origin (the spine).
    leg_colors : per-leg colours for the dot (defaults to C0/C1/C2).
    """
    ax.clear()
    if leg_colors is None:
        leg_colors = DEFAULT_LEG_COLORS
    K = len(dirs)
    for i in range(K):
        ax.annotate("", xy=(dirs[i, 0], dirs[i, 1]), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.6,
                                    shrinkA=0),
                    zorder=2)
    if sticky or page is None:
        ax.scatter([0.0], [0.0], s=dot_size, color="black", zorder=5)
    else:
        f = float(np.clip(frac, 0.0, 1.0))
        ax.scatter([dirs[page, 0] * f], [dirs[page, 1] * f],
                   s=dot_size, color=leg_colors[page], zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    if title is not None:
        ax.set_title(title, color=(title_color if title_color else "black"),
                     fontsize=title_fontsize, pad=4)
