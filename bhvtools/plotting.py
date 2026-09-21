"""Shared plotting helpers for the BHV / spider figures.

This module consolidates the drawing code that was previously duplicated across
the figure scripts: the fixed-layout phylogram drawer (:func:`draw_tree`) and
the folded-random-walk panel (:func:`draw_folded_walks`). Centralising it here
means a style change is made in exactly one place.

Line styles for the folded-walk panel are module constants:

    POS_LINESTYLE   walks strictly above y = 0   (default solid  "-")
    NEG_LINESTYLE   walks at or below  y = 0      (default dotted ":")
    ZERO_LINESTYLE  the reference line at y = 0    (default dashed "--")

To restyle every folded-walk panel in the repository, edit these constants (or
pass the corresponding keyword arguments per call).

Only ``numpy`` and the standard library are imported at module level; the
functions draw onto a Matplotlib ``Axes`` passed in by the caller, so importing
``bhvtools`` itself remains dependency-free.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .newick import _parse_newick_string

# --- centralized style constants -------------------------------------------
# Folded walks are solid both above and below y = 0; above is drawn thicker,
# below thinner. Edit these to restyle every folded-walk panel at once.
POS_LINESTYLE = "-"      # folded walk above y = 0  (solid)
NEG_LINESTYLE = "-"      # folded walk at/below y = 0 (solid)
POS_LINEWIDTH = 2.2      # above y = 0 (thicker)
NEG_LINEWIDTH = 1.1      # at/below y = 0 (thinner)
ZERO_LINESTYLE = "--"    # the y = 0 reference line

DEFAULT_LEAF_Y: Dict[str, int] = {"a": 4, "b": 3, "c": 2, "d": 1, "e": 0}
DEFAULT_LEG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green

# Frame rate shared by all animation scripts (so every GIF plays at one speed).
GIF_FPS = 10
# Rendering resolution (dots per inch) for saved GIF frames. Higher = finer
# (and larger files). PNG/PDF stills are saved at their own dpi by the scripts.
GIF_DPI = 200
# Absolute figure size (inches) shared by both animation scripts, so the GIFs
# come out at identical pixel dimensions (figsize x GIF_DPI) and their text --
# set in points -- renders at the same physical size side by side.
ANIM_FIGSIZE = (11.5, 4.3)
# Shared two-panel layout (left = folded-walk trace, right = space). Using the
# same width ratios and margins in both animation scripts makes the random-walk
# panel occupy the identical region/proportion in both GIFs.
ANIM_WIDTH_RATIOS = (1.15, 1.0)
ANIM_MARGINS = dict(left=0.08, right=0.96, top=0.88, bottom=0.15, wspace=0.18)

# Matplotlib style for paper-ready (LaTeX-like) figures: serif text with a
# Palatino preference, falling back to other serifs if Palatino is unavailable
# (e.g. in CI), so output is always serif rather than the sans-serif default.
PAPER_RC = {
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua",
                   "TeX Gyre Pagella", "DejaVu Serif", "Times New Roman", "serif"],
}


def use_paper_style():
    """Apply :data:`PAPER_RC` to the global Matplotlib rcParams."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_RC)


# ===========================================================================
# Phylogram (tree) drawing
# ===========================================================================
def layout(node, x0, pos, leaf_y=DEFAULT_LEAF_Y):
    """Assign (x, y) to every node; x = cumulative branch length, y from leaves."""
    bl = node.length if node.length else 0.0
    x = x0 + bl
    if not node.children:
        y = leaf_y[node.name]
    else:
        ys = [layout(c, x, pos, leaf_y) for c in node.children]
        y = sum(ys) / len(ys)
    pos[id(node)] = (x, y)
    return y


def collapsed_root(mu, collapse_tol):
    """Parse ``mu`` to a node tree with internal edges below ``collapse_tol`` removed."""
    drawn = mu.copy()
    drawn.clades = {m: l for m, l in drawn.clades.items() if l >= collapse_tol}
    return _parse_newick_string(drawn.to_newick())


def tree_depth(mu, collapse_tol, leaf_y=DEFAULT_LEAF_Y):
    """Maximum root-to-leaf branch-length depth of the collapsed tree."""
    pos = {}
    layout(collapsed_root(mu, collapse_tol), 0.0, pos, leaf_y)
    return max(x for x, _ in pos.values())


def draw_tree(ax, mu, color, *, label_x, xlim, ylim, collapse_tol,
              leaf_y=DEFAULT_LEAF_Y, leaf_color=None,
              title=None, default_title="Fréchet mean tree $F_n$",
              title_color=None, show_labels=True, show_leaders=True,
              show_xaxis=True, ylabel=None, label_offset=0.08,
              label_mathtext=True):
    """Draw ``mu`` as a fixed-layout phylogram on ``ax``.

    Parameters
    ----------
    color : branch colour. ``leaf_color`` defaults to ``color``.
    label_x, xlim, ylim, collapse_tol : layout/extent controls (caller-fixed so
        every frame shares identical axes).
    title : explicit panel title; if ``None`` uses ``default_title``.
    show_labels / show_leaders / show_xaxis : toggles for the filmstrip layout.
    label_offset : horizontal gap between ``label_x`` and the leaf label text
        (default 0.08, matching the original hardcoded spacing). Callers with
        a small ``xlim`` span (e.g. an axis capped well below the default
        scale) will typically want a smaller value here.
    label_mathtext : if True (default), leaf names are rendered as mathtext
        ("$name$"), matching earlier figures. Taxon names containing an
        underscore (e.g. "Tree_Shrew") render with a mathtext subscript in
        this mode; pass False for plain upright text with underscores shown
        as spaces instead (no subscript, and no italics).
    """
    ax.clear()
    pos = {}
    root = collapsed_root(mu, collapse_tol)
    layout(root, 0.0, pos, leaf_y)
    leaf_color = leaf_color if leaf_color is not None else color

    def render(node, parent_x):
        x, y = pos[id(node)]
        ax.plot([parent_x, x], [y, y], "-", color=color, lw=2.0, solid_capstyle="round")
        if node.children:
            cy = [pos[id(c)][1] for c in node.children]
            ax.plot([x, x], [min(cy), max(cy)], "-", color=color, lw=2.0)
            for c in node.children:
                render(c, x)
        else:
            if show_leaders:
                ax.plot([x, label_x], [y, y], ":", color="0.75", lw=1.0)
            ax.scatter([x], [y], s=22, color=leaf_color, zorder=3)
            if show_labels:
                if label_mathtext:
                    ax.text(label_x + label_offset, y, "$%s$" % node.name,
                            va="center", ha="left", fontsize=14, color="black")
                else:
                    ax.text(label_x + label_offset, y, node.name.replace("_", " "),
                            va="center", ha="left", fontsize=14, color="black")
    render(root, 0.0)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, color="black", labelpad=8)
    if show_xaxis:
        ax.set_xlabel("branch length")
    else:
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if title is None:
        ax.set_title(default_title, fontsize=15, pad=12)
    else:
        ax.set_title(title, color=(title_color if title_color else color),
                     fontsize=14, pad=4)


# ===========================================================================
# Folded random walks
# ===========================================================================
def sign_segments(xs, ys):
    """Yield (xseg, yseg, is_positive) pieces split exactly at zero crossings."""
    cx, cy = [xs[0]], [ys[0]]
    cur = ys[0] > 0.0
    for i in range(1, len(xs)):
        prev = ys[i - 1] > 0.0
        now = ys[i] > 0.0
        if now != prev:                       # crossing: interpolate x at y = 0
            t = ys[i - 1] / (ys[i - 1] - ys[i])
            xc = xs[i - 1] + t * (xs[i] - xs[i - 1])
            cx.append(xc); cy.append(0.0)
            yield np.array(cx), np.array(cy), cur
            cx, cy = [xc, xs[i]], [0.0, ys[i]]
            cur = now
        else:
            cx.append(xs[i]); cy.append(ys[i])
    yield np.array(cx), np.array(cy), cur


def sticky_segments(That, k):
    """Yield (x_start, x_end) spans of n in 1..k+1 where all walks are <= 0."""
    sticky = (That[:, :k + 1] <= 0.0).all(axis=0)
    n = sticky.size
    i = 0
    while i < n:
        if sticky[i]:
            j = i
            while j + 1 < n and sticky[j + 1]:
                j += 1
            yield (i + 1, j + 1)              # n is 1-indexed
            i = j + 1
        else:
            i += 1


def draw_folded_walks(ax, That, n_total, k=None, *, leg_colors=None, ylim=None,
                      xlabel="number of samples $n$", ylabel=None, title=None,
                      pos_linestyle=POS_LINESTYLE, neg_linestyle=NEG_LINESTYLE,
                      pos_linewidth=POS_LINEWIDTH, neg_linewidth=NEG_LINEWIDTH,
                      zero_linestyle=ZERO_LINESTYLE, sticky_lw=2.4,
                      show_endpoint_markers=True):
    """Plot folded random walks ``That[:, :k+1]`` on ``ax``.

    Each walk is solid both above and below 0, drawn thicker (``pos_linewidth``)
    above y = 0 and thinner (``neg_linewidth``) at/below it; the y = 0 line uses
    ``zero_linestyle``; sticky spans (all walks <= 0) get a solid black bar at
    y = 0. ``n_total`` fixes the x-axis; ``k`` is the last index drawn (defaults
    to the final column).

    show_endpoint_markers : if True (default), draw a scatter dot at each
        walk's current endpoint (n = k+1) -- appropriate for an animation
        frame, where the endpoint moves. Static figures showing the full
        sample typically want this off (the "endpoint" is just wherever the
        line happens to stop, not a meaningful marker).
    """
    That = np.asarray(That, dtype=float)
    K = That.shape[0]
    if k is None:
        k = That.shape[1] - 1
    if leg_colors is None:
        leg_colors = DEFAULT_LEG_COLORS[:K]

    ax.clear()
    xs = np.arange(1, k + 2, dtype=float)
    for j in range(K):
        ys = That[j, :k + 1]
        if len(xs) == 1:
            up = ys[0] > 0
            ax.plot(xs, ys, color=leg_colors[j],
                    lw=(pos_linewidth if up else neg_linewidth),
                    ls=(pos_linestyle if up else neg_linestyle))
        else:
            for xseg, yseg, pos in sign_segments(xs, ys):
                ax.plot(xseg, yseg, color=leg_colors[j],
                        lw=(pos_linewidth if pos else neg_linewidth),
                        ls=(pos_linestyle if pos else neg_linestyle))
        if show_endpoint_markers:
            ax.scatter([xs[-1]], [ys[-1]], color=leg_colors[j], s=20, zorder=3)

    ax.axhline(0.0, ls=zero_linestyle, color="black", lw=1.0)
    for x0, x1 in sticky_segments(That, k):
        ax.plot([x0, x1], [0.0, 0.0], "-", color="black", lw=sticky_lw,
                solid_capstyle="butt", zorder=4)

    ax.set_xlim(0.5, n_total + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=15, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
