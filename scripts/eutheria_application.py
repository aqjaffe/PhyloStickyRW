"""Real-data application to the 14-taxon eutherian gene-tree sample.

This single script combines the two former stand-alone figures, plus an
animated companion to the first:

  1. The BHV Frechet (Sturm) mean of the 424 gene trees, drawn as a black-and-
     white phylogram with the fixed-x = cumulative-branch-length layout
                                                      -> eutheria_mean.{pdf,png}

  1b. An animated version of (1): the running Frechet mean tree F_n for
      n = 10, ..., 424, taxa held at (1)'s fixed row order throughout
                                                  -> eutheria_mean_animation.gif

  2. Folded random walks at the internal {Primates, Glires, Tree Shrew}
     trifurcation -- a codimension-1 stratum of *rooted* BHV tree space
                                                   -> eutheria_RW.{pdf,png}

The random-walk panel is drawn with the shared paper style used by the
simulation figures (``bhv_simulation_*`` / ``spider_simulation_*``): each folded
walk is a *solid* line, drawn thinner below y = 0 and thicker above it, via
``bhvtools.plotting.draw_folded_walks``. Sticky spans (all legs <= 0) are marked
by a solid black bar at y = 0.

Finally the script prints the bootstrap bias-corrected estimate of the
unsticking probability P(T > n | Y_1,...,Y_n) at that trifurcation
(``bhvtools.stickiness.unsticking_probability_bootstrap`` on all 424 trees).

Input : data/primates_14taxa_rooted.txt  (one rooted Newick per line, all on the
        same 14 taxa, rooted on Sloth).
Output: figs/eutheria_mean.{pdf,png}, figs/eutheria_mean_animation.gif,
        figs/eutheria_RW.{pdf,png}
"""

import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

# make the in-repo packages importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bhvtools as bm
from bhvtools import plotting as bplt
from bhvtools import stickiness
from bhvtools.newick import _parse_newick_string

# paper-ready serif style shared with the simulation figures
bplt.use_paper_style()

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data"
_FIGS = _ROOT / "figs"

INFILE = _DATA / "primates_14taxa_rooted.txt"
MAX_EPOCHS = 200
SEED = 0
COLLAPSE_TOL = 1e-4          # substitutions/site; collapses only ~zero edges

# ---------------------------------------------------------------------------
# 0. Load trees once (shared by both figures and the stickiness estimate)
# ---------------------------------------------------------------------------
lines = [l.strip() for l in open(INFILE) if l.strip()]
trees, taxa = bm.load_newick_strings(lines)

# --- the internal trifurcation: A = Tree Shrew, B = Glires, C = Primates ---
# (leaf-name sets; shared by the mean-tree marker, the folded walks and the
#  unsticking estimate. These three are genuinely primate/glire/scandentian
#  clades, so their names are NOT renamed to "eutheria".)
A = {"Tree_Shrew"}                                            # Scandentia
B = {"Rabbit", "Rat"}                                         # Glires
C = {"Galago", "Mouse_Lemur", "Chimpanzee", "Human", "Gorilla",
     "Orangutan", "Macaque", "Marmoset", "Tarsier"}           # Primates
TRIF_LEAVES = A | B | C                                       # MRCA = trifurcation node


# ===========================================================================
# FIGURE 1: Frechet mean phylogram (black & white)
# ===========================================================================
mu = bm.frechet_mean(trees, max_epochs=MAX_EPOCHS, seed=SEED)

# drop internal edges below tolerance -> multifurcations
drawn = mu.copy()
drawn.clades = {m: l for m, l in drawn.clades.items() if l >= COLLAPSE_TOL}
root = _parse_newick_string(drawn.to_newick())


def _nleaves(node):
    if not node.children:
        return 1
    return sum(_nleaves(c) for c in node.children)


def ladderize(node):
    """Sort children by descending subtree size, for a clean comb layout."""
    if node.children:
        for c in node.children:
            ladderize(c)
        node.children.sort(key=lambda c: _nleaves(c))


ladderize(root)

_counter = [0]


def layout(node, x0, pos):
    x = x0 + (node.length if node.length else 0.0)
    if not node.children:
        y = _counter[0]
        _counter[0] += 1
    else:
        ys = [layout(c, x, pos) for c in node.children]
        y = sum(ys) / len(ys)
    pos[id(node)] = (x, y)
    return y


pos = {}
layout(root, 0.0, pos)
maxx = max(x for x, _ in pos.values())
nleaf = _counter[0]
LABEL_X = maxx + 0.02 * maxx


def leaf_names_below(node):
    if not node.children:
        return {node.name}
    s = set()
    for c in node.children:
        s |= leaf_names_below(c)
    return s


def find_node(node, target):
    """Return the node whose descendant-leaf set equals ``target`` (or None)."""
    if leaf_names_below(node) == target:
        return node
    for c in node.children:
        r = find_node(c, target)
        if r is not None:
            return r
    return None


BRANCH = "black"     # black-and-white tree
TIP = "black"        # leaf dots in black

fig, ax = plt.subplots(figsize=(8.2, 6.6))


def render(node, parent_x):
    x, y = pos[id(node)]
    ax.plot([parent_x, x], [y, y], "-", color=BRANCH, lw=2.0, solid_capstyle="round")
    if node.children:
        cy = [pos[id(c)][1] for c in node.children]
        ax.plot([x, x], [min(cy), max(cy)], "-", color=BRANCH, lw=2.0)
        for c in node.children:
            render(c, x)
    else:
        ax.plot([x, LABEL_X], [y, y], ":", color="0.75", lw=1.0)
        ax.scatter([x], [y], s=24, color=TIP, zorder=3)
        ax.text(LABEL_X + 0.01 * maxx, y, node.name.replace("_", " "),
                va="center", ha="left", fontsize=12)


render(root, 0.0)

# emphasise the {Primates, Glires, Tree Shrew} trifurcation with a black asterisk
# in the whitespace just left of the trifurcation node, placed vertically at the
# midpoint of the Rabbit/Rat (Glires) rows.
trif_node = find_node(root, TRIF_LEAVES)
rat_node = find_node(root, {"Rat"})
rabbit_node = find_node(root, {"Rabbit"})
if None not in (trif_node, rat_node, rabbit_node):
    x_trif = pos[id(trif_node)][0]
    y_mid = 0.5 * (pos[id(rat_node)][1] + pos[id(rabbit_node)][1])
    x_marker = x_trif - min(0.02 * maxx, 0.5 * x_trif)
    ax.scatter([x_marker], [y_mid], marker=(6, 2, 0), s=80,
               color="black", linewidths=1.3, zorder=5)
else:
    print("warning: {Primates, Glires, Tree Shrew} node not found in mean tree")

# ---------------------------------------------------------------------------
# Curly braces to the right of the taxon labels, marking named clades
# ---------------------------------------------------------------------------


def square_bracket(ax, x, y0, y1, label, *, width=0.012 * maxx, lw=1.3,
                    pad=0.024 * maxx, fontsize=12):
    """Square bracket spanning [y0, y1], opening toward the taxa (leftward):
    the spine sits at x + width (near the label) and the two ticks reach
    back to x, cradling the clade. ``label`` is written further right,
    rotated 90 degrees to read parallel to the spine."""
    ymid = 0.5 * (y0 + y1)
    ax.plot([x, x + width, x + width, x], [y0, y0, y1, y1],
            color="black", lw=lw, solid_capstyle="round", clip_on=False)
    ax.text(x + width + pad, ymid, label, va="center", ha="center",
            fontsize=fontsize, rotation=90)


BRACE_X = LABEL_X + 0.30 * maxx  # to the right of the taxon name text

primates_node = find_node(root, C)
glires_node = find_node(root, B)
if primates_node is not None:
    ys = [pos[id(find_node(root, {n}))][1] for n in C]
    square_bracket(ax, BRACE_X, min(ys) - 0.32, max(ys) + 0.32, "Primates")
else:
    print("warning: Primates clade not found in mean tree")
if glires_node is not None:
    ys = [pos[id(find_node(root, {n}))][1] for n in B]
    square_bracket(ax, BRACE_X, min(ys) - 0.32, max(ys) + 0.32, "Glires")
else:
    print("warning: Glires clade not found in mean tree")

ax.set_xlim(-0.02 * maxx, BRACE_X + 0.06 * maxx)
ax.set_ylim(-0.7, nleaf - 0.3)
ax.set_yticks([])
AXIS_MAX = 0.20  # cut the tick axis off near the taxa labels, not the brackets
ax.set_xticks(np.arange(0.0, AXIS_MAX + 1e-9, 0.05))
ax.spines["bottom"].set_bounds(0.0, AXIS_MAX)
ax.set_xlabel("branch length (substitutions / site)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_title("Fréchet mean of 424 eutherian gene trees", fontsize=14, pad=12)

fig.tight_layout()
fig.savefig(_FIGS / "eutheria_mean.pdf", bbox_inches="tight")
fig.savefig(_FIGS / "eutheria_mean.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote eutheria_mean.pdf and eutheria_mean.png")
print("mean newick:", mu.to_newick())


# ===========================================================================
# FIGURE 1b: animated companion to Figure 1 -- the running Frechet mean tree
#            F_n for n = 10, ..., 424 (formerly eutheria_mean_animation.py)
# ===========================================================================
# Taxa are held at Figure 1's fixed row positions throughout (read off `pos`
# above), via draw_tree's leaf_y argument: topology changes across n are
# absorbed by re-drawing branches between fixed leaf rows, rather than by
# re-ordering the rows. Matches Figure 1's black-and-white style (no colour
# coding here; that is left to a possible future combined RW+tree animation).
#
# n = 1..9 are skipped: with so few trees the running mean fluctuates too
# much for a legible animation. Frames are subsampled (every FRAME_STRIDE-th
# n) to keep the .gif a reasonable size; the final n = 424 frame is always
# included.
#
# Running F_n for every n via bhvtools.frechet.frechet_mean(trees[:n], ...)
# (as used for the Figure 1 still, above) is prohibitively slow at N = 424:
# each call re-runs MAX_EPOCHS shuffled passes over a growing prefix, giving
# O(N^2 * MAX_EPOCHS) total cost. Instead we use Sturm's single-pass
# inductive-mean iteration, already documented in bhvtools/frechet.py's
# module docstring and justified there (Sturm, Contemp. Math. 338:357-390,
# 2003, Thm 4.7):
#
#     mu_1 = T_1,   mu_n = geodesic(mu_{n-1}, T_n)(1/n),   n = 2, 3, ...
#
# which is the same update frechet_mean performs internally per sample within
# an epoch, just run once (no shuffled re-passes) -- an O(N) proxy for the
# batch-optimised mean, appropriate for an animation. Its n=424 endpoint
# agrees with the batch Frechet mean `mu` above to the precision shown in the
# printed Newick string.
N_START = 10                 # skip n=1..9: the running mean is too unstable there
ANIM_FRAME_STRIDE = 3        # keep every 3rd n (plus the final n=N frame)
ANIM_AXIS_MAX = 0.20         # cap the branch-length axis to emphasise fluctuations

# fixed row order and y-positions, taken directly from Figure 1's own layout
LEAF_Y = {leaf: pos[id(find_node(root, {leaf}))][1] for leaf in taxa.names}

running_means = [trees[0].copy()]
_mu = trees[0].copy()
for _n in range(2, len(trees) + 1):
    _mu = bm.Geodesic(_mu, trees[_n - 1]).point(1.0 / _n)
    running_means.append(_mu.copy())

anim_frame_ns = list(range(N_START, len(trees) + 1, ANIM_FRAME_STRIDE))
if anim_frame_ns[-1] != len(trees):
    anim_frame_ns.append(len(trees))

anim_maxdepth = max(bplt.tree_depth(running_means[n - 1], COLLAPSE_TOL, LEAF_Y)
                    for n in anim_frame_ns)
anim_label_x = anim_maxdepth + 0.015     # tight gap between leaders and labels
anim_brace_x = ANIM_AXIS_MAX + 0.19      # square brackets, clear of the labels
anim_xlim = (-0.02 * ANIM_AXIS_MAX, anim_brace_x + 0.05)
anim_ylim = (-0.7, nleaf - 0.3)

anim_primates_ys = [LEAF_Y[t] for t in C]
anim_glires_ys = [LEAF_Y[t] for t in B]

anim_fig, anim_ax = plt.subplots(figsize=(8.2, 6.6))


def _render_anim_frame(n):
    bplt.draw_tree(
        anim_ax, running_means[n - 1], BRANCH, label_x=anim_label_x,
        xlim=anim_xlim, ylim=anim_ylim, collapse_tol=COLLAPSE_TOL, leaf_y=LEAF_Y,
        title="Fréchet mean of $n = %d$ eutherian gene trees" % n,
        label_offset=0.008, label_mathtext=False,
    )
    anim_ax.set_xlabel("branch length (substitutions / site)")
    anim_ax.set_xticks(np.arange(0.0, ANIM_AXIS_MAX + 1e-9, 0.05))
    anim_ax.spines["bottom"].set_bounds(0.0, ANIM_AXIS_MAX)
    square_bracket(anim_ax, anim_brace_x, min(anim_primates_ys) - 0.32,
                   max(anim_primates_ys) + 0.32, "Primates",
                   width=0.003, pad=0.014)
    square_bracket(anim_ax, anim_brace_x, min(anim_glires_ys) - 0.32,
                   max(anim_glires_ys) + 0.32, "Glires",
                   width=0.003, pad=0.014)


_anim_frame_images = []
for _n in anim_frame_ns:
    _render_anim_frame(_n)
    _buf = io.BytesIO()
    anim_fig.savefig(_buf, format="png", dpi=bplt.GIF_DPI)
    _buf.seek(0)
    _anim_frame_images.append(Image.open(_buf).copy().convert("RGB"))
plt.close(anim_fig)

_anim_frame_images[0].save(
    str(_FIGS / "eutheria_mean_animation.gif"), save_all=True,
    append_images=_anim_frame_images[1:], duration=int(1000 / bplt.GIF_FPS), loop=0,
)
print("wrote eutheria_mean_animation.gif  (n = %d..%d, stride %d, %d frames)" %
      (N_START, len(trees), ANIM_FRAME_STRIDE, len(anim_frame_ns)))


# ===========================================================================
# FIGURE 2: Folded random walks at the internal trifurcation
#           (was folded_walks_internal.py, restyled like bhv_simulation_*)
# ===========================================================================
name2bit = {n: 1 << i for i, n in enumerate(taxa.names)}


def mask(S):
    m = 0
    for n in S:
        m |= name2bit[n]
    return m


def leglen(T, S):
    return T.clades.get(mask(S), 0.0)


# each leg is one resolution (page) of the trifurcation; label = resolved clade
LEGS = [(A | B, r"{{Tree Shrew, Glires}, Primates}"),
        (A | C, r"{{Tree Shrew, Primates}, Glires}"),
        (B | C, r"{{Glires, Primates}, Tree Shrew}")]
COLOR = ["#1f77b4", "#ff7f0e", "#2ca02c"]

L = np.array([[leglen(T, S) for S, _ in LEGS] for T in trees])   # (N,3)
S = L.sum(1, keepdims=True)
That = np.cumsum(2 * L - S, axis=0)                              # (N,3) folded walks
N = len(trees)

# bhvtools.plotting.draw_folded_walks expects walks as rows: shape (K_legs, N)
That_kn = That.T                                                 # (3, N)

pad = 0.05 * (That.max() - That.min())
ylim = (That.min() - pad, max(That.max(), 0.0) + pad)

fig, ax = plt.subplots(figsize=(7.6, 4.6))

# solid line, thinner below y=0 / thicker above it, + black sticky bars at y=0
bplt.draw_folded_walks(
    ax, That_kn, n_total=N, k=N - 1,
    leg_colors=COLOR, ylim=ylim,
    xlabel="number of gene trees $n$",
    ylabel=None,
    title="folded random walks at {Primates, Glires, Tree Shrew} trifurcation",
    show_endpoint_markers=False,
)

# legend (draw_folded_walks does not label the lines): use solid proxies
proxies = [Line2D([0], [0], color=COLOR[j], lw=bplt.POS_LINEWIDTH, ls="-")
           for j, _ in enumerate(LEGS)]
ax.legend(proxies, [lab for _, lab in LEGS],
          loc="lower left", frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(_FIGS / "eutheria_RW.pdf", bbox_inches="tight")
fig.savefig(_FIGS / "eutheria_RW.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote eutheria_RW.pdf and eutheria_RW.png")
print("final folded coords:", np.round(That[-1], 4).tolist(),
      "  sticky at n=N?", bool((That[-1] <= 0).all()))


# ===========================================================================
# Unsticking (stickiness) estimate on the full sample
# ===========================================================================
# The three resolutions (pages) of the trifurcation are the pairwise-union
# clades, i.e. exactly the LEG clades above.
resolutions = [frozenset(A | B), frozenset(A | C), frozenset(B | C)]
_p = stickiness.unsticking_probability_bootstrap(trees, resolutions=resolutions)
print("unsticking probability estimate (bias-corrected):", _p)
