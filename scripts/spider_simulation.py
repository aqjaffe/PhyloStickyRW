"""Sticky-Frechet-mean simulation on the k-spider (open book B_{K,1}).

A single run produces three views of one sampled process, written to figs/:

  spider_simulation_<seed>.pdf / .png   a journal-ready *still*: a filmstrip of
        the spider mean F_n at selected sample sizes (top) over the folded
        random walks across all n (bottom), with a marker tying each spider to
        its time.
  spider_simulation_<seed>.gif          the *animation*: folded random walks
        (left) beside the live spider with a marker at the mean (right).

Both views share the same data: i.i.d. samples on the K-spider (a uniformly
chosen leg, unit radius), the folded walks hat T_n^j, and the running Frechet
mean -- a coloured dot out on the active leg, or a black dot at the origin
(spine) when the mean is sticky. See Hotz et al. 2013.

Random-walk panel: bhvtools.plotting.draw_folded_walks (shared with the BHV
figure). Spider drawing: booktools.plotting.draw_spider.

Run:  python spider_simulation.py
Out:  figs/spider_simulation_<seed>.pdf, .png, .gif
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import booktools as bk
from booktools import plotting as bkplt          # spider (space) drawer
from bhvtools import plotting as bplt            # shared folded-walk drawer + style

_FIGS = Path(__file__).resolve().parents[1] / "figs"
_FIGS.mkdir(exist_ok=True)
bplt.use_paper_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 3
N = 100
SEED = 5555
RADIUS = bk.constant_radius(1.0)               # unit radii (as in sticky_RW_tikz)
# RADIUS = bk.exponential_radius(1.0)          # uncomment for Exp(1) radii
FILMSTRIP_N = [7, 13, 25, 55, 100]             # leg resolved early, then sticky

FIG_WIDTH = 5.517862646089656   # matches the GIF's folded-walk panel width
FIG_HEIGHT = 5.678855421686746  # top row sized to the spider panels; bottom row
                                 # matches the GIF's folded-walk panel height

OUTBASE = str(_FIGS / ("spider_simulation_%d" % SEED))
LEG_COLORS = bplt.DEFAULT_LEG_COLORS
BLACK = "#000000"
LEG_LABEL = {j: "leg %d" % (j + 1) for j in range(K)}

# ---------------------------------------------------------------------------
# Data (computed once)
# ---------------------------------------------------------------------------
book = bk.OpenBook(K=K, m=1)
pages, radii, spines = bk.sample(book, N, radial_sampler=RADIUS, rng=SEED)
That = bk.folded_walks(book, pages, radii)               # (K, N) folded walks
run = bk.running_frechet_means(book, pages, radii, spines)
rmax = max(float(run.r.max()), 1e-9)                     # scale for the spider dot (= max Frechet-mean radius F_n)
dirs = bkplt.spider_directions(K)
YL = (min(0.0, float(That.min())) * 1.05 - 1.0, max(0.0, float(That.max())) + 3.0)


def dot_args(k):
    """(page or None, frac, sticky, colour) for the Frechet-mean marker at frame k.

    The dot's radial position encodes the Frechet mean F_n = run.r[k] (the mean
    radius on the active leg) -- NOT the folded random walk hat T_n = n * F_n
    shown in the left/bottom panel. The two are deliberately distinct: the walk
    accumulates, while the mean is the walk divided by n.
    """
    sticky = bool(run.sticky[k])
    page = int(run.page[k])
    color = BLACK if sticky else LEG_COLORS[page]
    frac = 0.0 if sticky else float(run.r[k]) / rmax
    return (None if sticky else page), frac, sticky, color


# ---------------------------------------------------------------------------
# Still (PDF + PNG): filmstrip of spider mean over the folded walks
# ---------------------------------------------------------------------------
def build_still(outbase, filmstrip_n):
    fn = sorted(int(n) for n in filmstrip_n)
    assert fn and all(1 <= n <= N for n in fn), "FILMSTRIP_N must be integers in 1..N"
    cols = len(fn)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(2, cols, height_ratios=[0.43007327174259324, 1.0],
                          hspace=0.10, wspace=0.18,
                          top=0.93, bottom=0.10, left=0.06, right=0.96)

    for c, n in enumerate(fn):
        k = n - 1
        page, frac, sticky, color = dot_args(k)
        ax = fig.add_subplot(gs[0, c])
        bkplt.draw_spider(ax, dirs, page=page, frac=frac, sticky=sticky,
                          leg_colors=LEG_COLORS, title=r"$F_{%d}$" % n, title_color=color)
        if c == 0:
            ax.set_ylabel(r"Fréchet mean $F_n$", fontsize=13, labelpad=8)

    axB = fig.add_subplot(gs[1, :])
    bplt.draw_folded_walks(axB, That, n_total=N, k=N - 1, leg_colors=LEG_COLORS,
                           ylim=YL, ylabel="folded random walks",
                           show_endpoint_markers=False)

    proxies = [Line2D([0], [0], color=LEG_COLORS[j], lw=bplt.POS_LINEWIDTH, ls="-")
               for j in range(K)]
    axB.legend(proxies, [LEG_LABEL[j] for j in range(K)],
               loc="lower left", frameon=False, fontsize=10)

    for n in fn:
        k = n - 1
        if run.sticky[k]:
            y, color = 0.0, BLACK
        else:
            page = int(run.page[k])
            y, color = float(That[page, k]), LEG_COLORS[page]
        axB.scatter([n], [y], s=50, color=color, zorder=6, linewidths=0.8)
        axB.annotate(r"$F_{%d}$" % n, (n, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", color=color, fontsize=10)

    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation (GIF): folded walks (left) beside the live spider (right)
# ---------------------------------------------------------------------------
def build_animation(outpath):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=bplt.ANIM_FIGSIZE,
                                   gridspec_kw={"width_ratios": bplt.ANIM_WIDTH_RATIOS})
    fig.subplots_adjust(**bplt.ANIM_MARGINS)

    def update(k):
        bplt.draw_folded_walks(axL, That, n_total=N, k=k, leg_colors=LEG_COLORS,
                               ylim=YL, title="folded random walks")
        page, frac, sticky, color = dot_args(k)
        bkplt.draw_spider(axR, dirs, page=page, frac=frac, sticky=sticky,
                          leg_colors=LEG_COLORS, title=r"Fréchet mean $F_n$",
                          title_color=color, title_fontsize=15)
        return []

    anim = FuncAnimation(fig, update, frames=N, interval=120, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=bplt.GIF_FPS), dpi=bplt.GIF_DPI)
    plt.close(fig)


def main():
    build_still(OUTBASE, FILMSTRIP_N)
    build_animation(OUTBASE + ".gif")
    print("wrote %s.pdf, .png, .gif  (filmstrip n = %s)" % (OUTBASE, FILMSTRIP_N))


if __name__ == "__main__":
    main()
