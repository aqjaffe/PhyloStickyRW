"""Sticky-Frechet-mean simulation in BHV phylogenetic tree space.

A single run produces three views of one sampled process, written to figs/:

  bhv_simulation_<seed>.pdf / .png   a journal-ready *still*: a filmstrip of
        Frechet-mean trees F_n at selected sample sizes (top), over the folded
        random walks across all n (bottom), with a marker tying each tree to
        its time.
  bhv_simulation_<seed>.gif          the *animation*: folded random walks (left)
        beside the live Frechet-mean tree F_n (right), frame by frame.

Both views share the same data. We sample 5-taxon trees whose {a,b,c} subtree
resolves one of the cherries alpha=(a,b), beta=(a,c), gamma=(b,c); the folded
walks hat T_n^i are the folding-map image at that trifurcation (Hotz et al.
2013; Barden-Le 2018). At most one walk exceeds 0 at any n; the sticky spans
(all walks <= 0, i.e. an unresolved trifurcation) are the black bar at y = 0.

Tree/folded-walk drawing and the shared style are in bhvtools.plotting.

Run:  python bhv_simulation.py
Out:  figs/bhv_simulation_<seed>.pdf, .png, .gif
"""

import random
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bhvtools as bm
from bhvtools import plotting as bplt

_FIGS = Path(__file__).resolve().parents[1] / "figs"
_FIGS.mkdir(exist_ok=True)
bplt.use_paper_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = [
    "(((a:1,b:1):0.7,c:1):1,(d:1,e:1):1);",   # cherry (a,b) = alpha
    "(((a:1,b:1):0.6,c:1):1,(d:1,e:1):1);",   # cherry (a,b) = alpha
    "(((a:1,c:1):1.0,b:1):1,(d:1,e:1):1);",   # cherry (a,c) = beta
    "(((b:1,c:1):1.0,a:1):1,(d:1,e:1):1);",   # cherry (b,c) = gamma
]

N = 100
SAMPLE_SEED = 39
SOLVER_SEED = 0
MAX_EPOCHS = 200
TRIFURC_TOL = 0.001
COLLAPSE_TOL = TRIFURC_TOL
FILMSTRIP_N = [10, 30, 50, 70, 90]   # sample sizes shown as F_n in the still

WIDTH_SCALE = 0.75
FIG_HEIGHT = 7.248694779116462  # matches the GIF's folded-walk panel height

OUTBASE = str(_FIGS / ("bhv_simulation_%d" % SAMPLE_SEED))

LEG_COLOR = bplt.DEFAULT_LEG_COLORS                        # blue, orange, green
LEAF_Y = bplt.DEFAULT_LEAF_Y
BLACK = "#000000"
CHERRIES = {("a", "b"): 0, ("a", "c"): 1, ("b", "c"): 2}   # alpha, beta, gamma
LEG_NAME = {0: "blue (a,b)", 1: "orange (a,c)", 2: "green (b,c)", None: "STICKY (trifurcation)"}
# trifurcation resolved by each leg, in the {{.,.},.} notation used in the legend
LEG_LABEL = {0: r"$\{\{a,b\},c\}$", 1: r"$\{b,\{a,c\}\}$", 2: r"$\{a,\{b,c\}\}$"}
RW_TEXT = r"folded random walks at $\{a,b,c\}$ trifurcation"

# ---------------------------------------------------------------------------
# Data: continuum sampling, Frechet means, folded walks (computed once)
# ---------------------------------------------------------------------------
trees4, taxa = bm.load_newick_strings(BASE)


def randomize_branch_lengths(newick, rng):
    """Replace every branch length w by an independent Exp(mean = w) variate."""
    return re.sub(
        r":([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        lambda mo: ":%.12g" % rng.expovariate(1.0 / float(mo.group(1))),
        newick,
    )


def cherry_of(mu):
    """(leg index in {0,1,2}, cherry length) of the {a,b,c}-resolving edge."""
    for m, l in mu.clades.items():
        nm = tuple(sorted(taxa.names[i] for i in range(taxa.n) if m >> i & 1))
        if nm in CHERRIES:
            return CHERRIES[nm], l
    return None, 0.0


def winning_leg(That, k):
    """Index of the unique walk above 0 at frame k, or None (sticky)."""
    col = That[:, k]
    j = int(np.argmax(col))
    return j if col[j] > 0.0 else None


def regime_spans(That):
    """Runs (n_start, n_end, leg) of constant winning leg, n 1-indexed."""
    labels = [winning_leg(That, k) for k in range(N)]
    spans, s = [], 0
    for k in range(1, N + 1):
        if k == N or labels[k] != labels[s]:
            spans.append((s + 1, k, labels[s]))
            s = k
    return spans


def compute():
    rng = random.Random(SAMPLE_SEED)
    samples = []
    for _ in range(N):
        j = rng.randrange(len(BASE))
        tree, _ = bm.load_newick_strings([randomize_branch_lengths(BASE[j], rng)])
        samples.append(tree[0])

    means, prev = [], None
    for n in range(1, N + 1):
        prev = bm.frechet_mean(samples[:n], init=prev, max_epochs=MAX_EPOCHS, seed=SOLVER_SEED)
        means.append(prev)

    phi = np.zeros((3, N))
    for i, T in enumerate(samples):
        leg, r = cherry_of(T)
        for j in range(3):
            phi[j, i] = r if j == leg else -r
    That = np.cumsum(phi, axis=1)                          # (3, N) unnormalised RW
    return means, That


# ---------------------------------------------------------------------------
# Still (PDF + PNG): filmstrip of F_n over the folded walks
# ---------------------------------------------------------------------------
def build_still(means, That, outbase, filmstrip_n):
    fn = sorted(int(n) for n in filmstrip_n)
    assert fn and all(1 <= n <= N for n in fn), "FILMSTRIP_N must be integers in 1..N"
    cols = len(fn)

    maxdepth = max(bplt.tree_depth(means[n - 1], COLLAPSE_TOL, LEAF_Y) for n in fn)
    label_x = maxdepth + 0.15
    xlim = (-0.15, label_x + 0.35)

    fig = plt.figure(figsize=(WIDTH_SCALE * (2.0 * cols + 1.0), FIG_HEIGHT))
    gs = fig.add_gridspec(2, cols, height_ratios=[1.0, 1.5],
                          hspace=0.30, wspace=0.18,
                          top=0.93, bottom=0.10, left=0.05, right=0.95)

    ax_first = None
    for c, n in enumerate(fn):
        k = n - 1
        j = winning_leg(That, k)
        color = LEG_COLOR[j] if j is not None else BLACK
        last = (c == cols - 1)
        ax = fig.add_subplot(gs[0, c])
        if c == 0:
            ax_first = ax
        bplt.draw_tree(ax, means[k], color, label_x=label_x, xlim=xlim, ylim=(-0.7, 4.7),
                       collapse_tol=COLLAPSE_TOL, leaf_y=LEAF_Y,
                       title=r"$F_{%d}$" % n, show_labels=last,
                       show_leaders=last, show_xaxis=False,
                       ylabel="Fréchet mean tree" if c == 0 else None)

    axB = fig.add_subplot(gs[1, :])
    bplt.draw_folded_walks(axB, That, n_total=N, k=N - 1, leg_colors=LEG_COLOR,
                           ylim=(-40, 10), ylabel=RW_TEXT,
                           show_endpoint_markers=False)

    proxies = [Line2D([0], [0], color=LEG_COLOR[j], lw=bplt.POS_LINEWIDTH, ls="-")
               for j in range(3)]
    axB.legend(proxies, [LEG_LABEL[j] for j in range(3)],
               loc="lower left", frameon=False, fontsize=10)

    for n in fn:
        k = n - 1
        j = winning_leg(That, k)
        if j is None:
            y, color = 0.0, BLACK
        else:
            y, color = float(That[j, k]), LEG_COLOR[j]
        axB.scatter([n], [y], s=50, color=color, zorder=6, linewidths=0.8)
        axB.annotate(r"$F_{%d}$" % n, (n, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", color=color, fontsize=10)

    fig.align_ylabels([ax_first, axB])
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation (GIF): folded walks (left) beside the live mean tree (right)
# ---------------------------------------------------------------------------
def build_animation(means, That, outpath):
    maxdepth = max(bplt.tree_depth(mu, COLLAPSE_TOL, LEAF_Y) for mu in means)
    label_x = maxdepth + 0.15
    xlim = (-0.15, label_x + 0.45)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=bplt.ANIM_FIGSIZE,
                                   gridspec_kw={"width_ratios": bplt.ANIM_WIDTH_RATIOS})
    fig.subplots_adjust(**bplt.ANIM_MARGINS)

    def update(k):
        bplt.draw_folded_walks(axL, That, n_total=N, k=k, leg_colors=LEG_COLOR,
                               ylim=(-40, 10), title=RW_TEXT)
        j = winning_leg(That, k)
        bplt.draw_tree(axR, means[k], LEG_COLOR[j] if j is not None else BLACK,
                       label_x=label_x, xlim=xlim, ylim=(-0.7, 4.7),
                       collapse_tol=COLLAPSE_TOL, leaf_y=LEAF_Y)
        return []

    anim = FuncAnimation(fig, update, frames=N, interval=300, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=bplt.GIF_FPS), dpi=bplt.GIF_DPI)
    plt.close(fig)


def main():
    means, That = compute()
    print("regime spans (n_start..n_end : leg):")
    for a, b, leg in regime_spans(That):
        print("  %3d..%-3d  %s" % (a, b, LEG_NAME[leg]))

    build_still(means, That, OUTBASE, FILMSTRIP_N)
    build_animation(means, That, OUTBASE + ".gif")
    print("wrote %s.pdf, .png, .gif  (filmstrip n = %s)" % (OUTBASE, FILMSTRIP_N))


if __name__ == "__main__":
    main()
