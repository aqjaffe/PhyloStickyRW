# sticky-frechet-means

Code accompanying a paper on the **stickiness of Fréchet means**. It provides
two small, self-contained Python packages and the scripts that generate the
paper's figures:

- **`bhvtools`** — pure-Python computation of Fréchet (Karcher) means of rooted
  phylogenetic trees in the **BHV metric** (Billera–Holmes–Vogtmann tree space),
  via the Owen–Provan geodesic algorithm and Sturm's inductive-mean iteration.
  A from-scratch port of the geodesic machinery in Megan Owen's Java
  `treestats`. See [`bhvtools/README.md`](bhvtools/README.md) for the full
  mathematical description, correctness argument, and references.
- **`booktools`** — Fréchet means and stickiness on **open books** `B_{K,m}`:
  `K` half-space pages `R_{>=0} × R^{m-1}` glued along a common spine `R^{m-1}`
  (Hotz et al. 2013). The `k`-spider is the case `m = 1`. The open-book mean is
  available in **closed form** (no iteration): the spine coordinate is the
  Euclidean mean, and the page/radius part is solved by the folding-map
  argument, which is exactly where stickiness appears.

The two settings are the two faces of the same phenomenon: in BHV tree space a
trifurcation (an unresolved internal edge) plays the role of the open-book
spine, and the mean "sticks" there for an open set of samples.

## Repository layout

```
bhvtools/        BHV tree-space package (pure standard library)
  stickiness.py   unsticking-probability estimator at a codim-1 stratum
  plotting.py     shared figure helpers: phylogram drawer + folded-walk drawer (numpy/matplotlib)
booktools/       open-book package B_{K,m} (numpy)
  stickiness.py   unsticking-probability estimator on the open book
  plotting.py     spider (open-book) space drawer (numpy/matplotlib)
scripts/        figure-generating and analysis scripts
  spider_simulation.py     k-spider: writes spider_simulation_<seed>.{pdf,png,gif}
  bhv_simulation.py        BHV tree space: writes bhv_simulation_<seed>.{pdf,png,gif}
  unsticking_demo.py       minimal demo of the unsticking-probability estimator
  unsticking_simulation.py runs the spider/BHV processes, then the estimator on the endpoints
  unsticking_vs_n.py       two-row figure: folded walks (top) + running P_hat(T>n) vs n (bottom)
  bootstrap_bias_experiment.py  evaluates the bootstrap bias-corrected estimator vs the vanilla one
  compare_estimators.py    3-panel figure: relative-error histograms of both estimators across n
tests/          test suites for both packages (incl. test_stickiness.py)
figs/           output directory for generated figures
```

Each script runs one sampled process and emits three views of it: a
journal-ready **still** (`.pdf`/`.png`) — a filmstrip of the mean F_n over the
folded random walks — and an **animation** (`.gif`) — the folded walks beside
the live space (a BHV mean tree, or the spider). Every figure thus pairs a
random-walk trace with a visualization of the underlying space.

All figures draw the folded-walk panel through the single shared
`bhvtools.plotting.draw_folded_walks`. Its appearance is centralized as module
constants: walks are solid both sides, drawn thicker above `y = 0`
(`POS_LINEWIDTH`) and thinner below (`NEG_LINEWIDTH`), with `POS_LINESTYLE`,
`NEG_LINESTYLE`, and `ZERO_LINESTYLE` for line styles; editing one constant
restyles every panel. The scripts call `bhvtools.plotting.use_paper_style()` for
LaTeX-like serif text (Palatino preferred, with serif fallbacks) and share the
animation frame rate and resolution, `bhvtools.plotting.GIF_FPS` and `GIF_DPI`.

## Installation

Python ≥ 3.9. From the repository root:

```bash
pip install -e ".[figures]"     # installs both packages + matplotlib for figures
```

`bhvtools` is pure standard library; `booktools` needs only `numpy`; the figure
scripts additionally need `matplotlib`. The scripts also insert the repository
root on `sys.path`, so they can be run from a clone without installation.

## Quick start

BHV tree space:

```python
import bhvtools as bm
trees, taxa = bm.load_newick_strings([
    "((a:1,b:1):1,(c:1,d:1):1);",
    "((a:1,c:1):1,(b:1,d:1):1);",
    "((a:1,b:1):1,(c:1,d:1):1);",
])
mu = bm.frechet_mean(trees)
print(mu.to_newick(), bm.frechet_variance(trees, mu))
```

Open book (the 3-spider):

```python
import booktools as bk
book = bk.OpenBook(K=3, m=1)
pages, radii, spines = bk.sample(book, 200, rng=0)
F = bk.frechet_mean(book, pages, radii, spines)
print("sticky" if F.sticky else f"page {F.page}, r = {F.r:.3f}")
run = bk.running_frechet_means(book, pages, radii, spines)   # F_1 .. F_n
```

## Estimating stickiness (the unsticking-probability estimator)

Both packages implement `UnstickingProbability`, which returns a single real
number in `(0, 1]` — an estimate of `P(T > n | Y_1,…,Y_n)`, where `T` is the
unsticking time of the empirical Fréchet mean. The estimator depends on the data
only through the *folding-map coordinates* `Φ_j(Y_i)` (the signed length of
sample `i` on page `j`). For each page it solves for the largest root of the
empirical MGF equation `(1/n) Σ_i exp(λ Φ_j(Y_i)) = 1` — the adjustment
coefficient / Lundberg exponent — and returns `exp(max_j θ_j Σ_i Φ_j(Y_i))`.
Because at most one folded mean is positive (Hotz et al. 2013), the value is `≤
1`, and it equals `1` exactly when the mean has already unstuck (`T ≤ n`).

```python
import numpy as np
import booktools as bk
book = bk.OpenBook(K=3, m=1)
pages, radii, _ = bk.sample(book, 200, rng=0)
print(bk.unsticking_probability(book, pages, radii))      # P_hat(T > n) in (0, 1]

import bhvtools as bm
samples, taxa = bm.load_newick_strings([...])             # trees Y_1..Y_n
Tstar, _ = bm.load_newick_strings(["((a:1,b:1,c:1):1,(d:1,e:1):1);"], taxa)
print(bm.unsticking_probability(samples, base_tree=Tstar[0]))   # codim-1 stratum
```

In BHV tree space a codimension-one stratum is locally the open book `B_{3,1}`,
so the log map at the base tree `T*` is available in closed form: each sample's
transverse coordinate is which of the three trifurcation resolutions it uses and
the length of that resolving edge. `bhvtools` projects to those page coordinates
(`page_coordinates` / `codim1_resolutions`) and then runs the same per-page tilt
as `booktools`. The mathematics and references (Asmussen 2000; Dembo–Zeitouni
1998; Hotz et al. 2013) are documented in `booktools/stickiness.py`.

```bash
python scripts/unsticking_demo.py          # minimal demo on sampled data
python scripts/unsticking_simulation.py    # runs the spider/BHV processes, estimator on endpoints
python scripts/unsticking_vs_n.py          # figs/unsticking_vs_n_{spider,bhv}_<seed>.{pdf,png}
```

### Bootstrap bias-corrected variant

The estimator maximises over noisy per-page tilts, so its log-probability has an
$O(\sqrt n)$ winner's-curse bias (relative bias $O(1/\sqrt n)$). Both packages
also provide `unsticking_probability_bootstrap`, an $m$-out-of-$n$ subsampling
bias correction (Richardson-extrapolated; default subsample size $m=\lfloor
n/2\rfloor$, overridable via `m=` as an int or a fraction in $(0,1)$). The
vanilla estimator is left unchanged, so the two can be compared directly. The
correction's behaviour (it removes most of the bias for $n\gtrsim100$, trading
some variance, controlled by $m/n$) is characterised in
`scripts/bootstrap_bias_experiment.py`.

## Reproducing the figures

```bash
python scripts/spider_simulation.py     # figs/spider_simulation_<seed>.{pdf,png,gif}
python scripts/bhv_simulation.py         # figs/bhv_simulation_<seed>.{pdf,png,gif}
```

Each script writes all three files for one process. Sampling, seeds, sample
size, and `FILMSTRIP_N` (the sample sizes shown in the still) are set near the
top of each script. All outputs go to `figs/`.

## Tests

```bash
python -m pytest -q          # or: python tests/test_bhvtools.py ; python tests/test_booktools.py
```

The `booktools` suite includes a direct numerical check that the closed-form
mean minimises the Fréchet functional (random candidates plus a fine radius
search on every page and the spine), and a check that the `m = 1` case
reproduces the spider folding logic. `tests/test_stickiness.py` checks the
unsticking-probability estimator: that each per-page tilt `θ_j` really solves
the MGF equation, the sign rule, the value lies in `(0, 1]`, a symmetric
closed-form case, and that the BHV codim-1 estimator reduces exactly to the
open-book one on matched data.

## References

1. T. Hotz, S. Huckemann, H. Le, J. S. Marron, J. C. Mattingly, E. Miller,
   J. Nolen, M. Owen, V. Patrangenaru, S. Skwerer. *Sticky central limit
   theorems on open books.* Ann. Appl. Probab. 23(6) (2013) 2238–2258.
   DOI 10.1214/12-AAP899.
2. L. J. Billera, S. P. Holmes, K. Vogtmann. *Geometry of the space of
   phylogenetic trees.* Adv. Appl. Math. 27 (2001) 733–767.
3. M. Owen, J. S. Provan. *A fast algorithm for computing geodesic distances in
   tree space.* IEEE/ACM Trans. Comput. Biol. Bioinform. 8(1) (2011) 2–13.
4. K.-T. Sturm. *Probability measures on metric spaces of nonpositive
   curvature.* Contemp. Math. 338 (2003) 357–390.
5. E. Miller, M. Owen, J. S. Provan. *Polyhedral computational geometry for
   averaging metric phylogenetic trees.* Adv. Appl. Math. 68 (2015) 51–91.

(Additional references specific to the BHV geodesic algorithm are in
[`bhvtools/README.md`](bhvtools/README.md).)

## License

MIT — see [`LICENSE`](LICENSE). Author, paper title, and arXiv ID are marked
`TODO` in `LICENSE`, `CITATION.cff`, and `pyproject.toml`; fill these in before
release.
