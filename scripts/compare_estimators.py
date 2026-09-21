"""Compare the standard and bootstrap-corrected unsticking estimators across n.

For each n in {100, 1000, 10000} we run B Monte Carlo trials on the symmetric
open book B_{3,1} (uniform pages, Exp(1) radii), where the population Cramer
root theta* = (K-2)/K is known, so the oracle log-probability

    log p_n = theta* * max_j S_{j,n}

is available per trial. For every (sticky) trial we record the relative error

    (log p_hat_n - log p_n) / log p_n

for BOTH estimators -- the standard one (booktools.unsticking_probability, whose
log-estimate is the trial's ``logprob_hat``) and the bootstrap bias-corrected one
(booktools.unsticking_probability_bootstrap, ``logprob_bc``).

The figure is a 2x3 grid: columns are n increasing left -> right; the top row is
the standard procedure (UnstickingProbability) and the bottom row the
bias-corrected procedure (UnstickingProbabilityBootstrap). Each panel shows one
histogram (black, alpha=0.4) with a dashed vertical line at 0. Moving right (n
grows) both rows concentrate at 0 (consistency); the bottom row sits closer to 0
than the top (reduced winner's-curse bias), most visibly at the smallest n. Row
names are drawn in an emulated small-caps face (\\textsc-like) to the left of the
y-axis.

Run:  python scripts/compare_estimators.py
Out:  figs/compare_estimators.{pdf,png}
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import booktools as bk
from bhvtools import plotting as bplt

_FIGS = Path(__file__).resolve().parents[1] / "figs"
_FIGS.mkdir(exist_ok=True)
bplt.use_paper_style()


# ---------------------------------------------------------------------------
# Small-caps emulation (portable; no dependence on an installed SC font file).
# Renders like LaTeX \textsc{}: originally-uppercase letters at full cap height,
# originally-lowercase letters as reduced-size caps on a shared baseline.
# ---------------------------------------------------------------------------
def _sc_runs(text, cap_size, sc_ratio):
    """Split into case-runs, uppercased, with a font size per run."""
    runs, cur, curcap = [], "", None
    for ch in text:
        cap = (not ch.isalpha()) or ch.isupper()
        if curcap is None:
            curcap, cur = cap, ch.upper()
        elif cap == curcap:
            cur += ch.upper()
        else:
            runs.append((cur, cap_size if curcap else cap_size * sc_ratio))
            cur, curcap = ch.upper(), cap
    if cur:
        runs.append((cur, cap_size if curcap else cap_size * sc_ratio))
    return runs


def draw_small_caps(ax, x_ax, y_ax, text, cap_size, *, sc_ratio=0.76,
                    family="serif", color="black", fit_frac=0.95):
    """Draw vertical (rotation=90) small-caps ``text`` centred at axes-fraction
    ``(x_ax, y_ax)``. Font auto-shrinks so the string fits ``fit_frac`` of the
    panel height. Placement is in axes fractions, so it is dpi-robust. Returns
    the (possibly reduced) ``cap_size`` actually used."""
    fig = ax.figure
    if fig.canvas.get_renderer() is None:
        fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h_px = ax.get_window_extent(r).height

    def advances(cs):
        runs = _sc_runs(text, cs, sc_ratio)
        adv = []
        for seg, sz in runs:
            t = mtext.Text(0, 0, seg, fontsize=sz, family=family)
            t.set_figure(fig)
            adv.append(t.get_window_extent(r).width)   # baseline advance (px)
        return runs, adv

    runs, adv = advances(cap_size)
    total = sum(adv)
    if total > fit_frac * h_px:                        # shrink to fit the panel
        cap_size *= fit_frac * h_px / total
        runs, adv = advances(cap_size)

    frac = [a / h_px for a in adv]                      # advance in axes fractions
    y = y_ax - sum(frac) / 2.0
    for (seg, sz), f in zip(runs, frac):
        ax.text(x_ax, y, seg, transform=ax.transAxes, rotation=90,
                rotation_mode="anchor", ha="left", va="baseline",
                fontsize=sz, family=family, color=color)
        y += f
    return cap_size


def cramer_root_symmetric(K, scale=1.0):
    """theta* for the symmetric book, Exp(mean=scale) radii: (K-2)/(K*scale)."""
    return (K - 2) / (K * scale)


def relative_errors(book, theta_star, n, B, n_bootstrap, seed):
    """Relative errors of both estimators over B sticky trials at size n."""
    rng = np.random.default_rng(seed)
    rel_std, rel_bc = [], []
    K = book.K
    for _ in range(B):
        pages, radii, _ = bk.sample(book, n, rng=rng)
        phi = np.stack([book.fold(j, pages, radii) for j in range(K)])
        max_S = float(phi.sum(axis=1).max())
        if max_S >= -1e-12:                       # keep the sticky regime only
            continue
        logp_oracle = theta_star * max_S
        _, d = bk.unsticking_probability_bootstrap(
            book, pages, radii, n_bootstrap=n_bootstrap, rng=rng, return_details=True)
        rel_std.append((d["logprob_hat"] - logp_oracle) / logp_oracle)
        rel_bc.append((d["logprob_bc"] - logp_oracle) / logp_oracle)
    return np.array(rel_std), np.array(rel_bc)


def main():
    K = 3
    n_values = [100, 1000, 10000]
    B = 300
    n_bootstrap = 120
    seed = 0
    book = bk.OpenBook(K=K, m=1)
    theta_star = cramer_root_symmetric(K)

    results = {}
    for n in n_values:
        t0 = time.time()
        results[n] = relative_errors(book, theta_star, n, B, n_bootstrap, seed)
        print(f"n = {n:6d}: {len(results[n][0])}/{B} sticky trials "
              f"({time.time() - t0:.1f}s)")

    # Shared x-range from the widest (smallest-n) data, robust to outliers.
    widest = np.concatenate(results[min(n_values)])
    xlo = min(np.percentile(widest, 1.0), 0.0) - 0.05
    xhi = max(np.percentile(widest, 99.0), 0.0) + 0.05
    bins = np.linspace(xlo, xhi, 41)

    order = sorted(n_values)                        # n increases left -> right
    row_labels = ["UnstickingProbability", "UnstickingProbabilityBootstrap"]
    xlabel = r"$(\log\hat p_n - \log p_n)/\log p_n$"

    # Denser, bhv_simulation-like styling: compact panels, larger axis labels,
    # slightly heavier frame / reference lines.
    LABEL_FS, TITLE_FS = 14, 15
    plt.rcParams.update({"axes.linewidth": 1.1,
                         "xtick.major.width": 1.0, "ytick.major.width": 1.0})

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharex=True, sharey=True)
    for col, n in enumerate(order):
        series = results[n]                         # (rel_std, rel_bc) -> rows
        for row in range(2):
            ax = axes[row, col]
            data = np.clip(series[row], xlo, xhi)
            ax.hist(data, bins=bins,
                    weights=np.full(data.size, 1.0 / data.size),
                    color="black", alpha=0.4)              # per-bin probability
            ax.axvline(0.0, ls="--", color="black", lw=1.4)
            # keep all four spines (full box) so the dashed zero line cannot be
            # mistaken for the right-hand plot edge
            if row == 0:                            # titles only on the top row
                ax.set_title(f"$n = {n:,}$", fontsize=TITLE_FS)
            if row == 1:                            # x-label on the bottom row
                ax.set_xlabel(xlabel, fontsize=LABEL_FS)
            if col == 0:                            # y-label on the left column
                ax.set_ylabel("frequency", fontsize=LABEL_FS)

    fig.tight_layout()

    # Row names in small caps, close to the y-axis (drawn after tight_layout so
    # the axes geometry is final). Fit the longer (bottom) label first, then use
    # the same size for the top one.
    X_SC = -0.23
    used = draw_small_caps(axes[1, 0], X_SC, 0.5, row_labels[1],
                           cap_size=15, sc_ratio=0.74, fit_frac=0.96)
    draw_small_caps(axes[0, 0], X_SC, 0.5, row_labels[0],
                    cap_size=used, sc_ratio=0.74, fit_frac=2.0)

    out = _FIGS / "compare_estimators"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
