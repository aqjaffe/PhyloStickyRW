# Phylogenetic Inference and the Stickiness of Fréchet Means, via Precise Asymptotics of an Embedded Random Walk

Code accompanying a paper on the stickiness of Fréchet means in BHV treespaces and open books. It provides
two small, self-contained Python packages and the scripts that generate the
paper's figures:

- **`bhvtools`** — Computation of Fréchet means of rooted
  phylogenetic trees in the Billera–Holmes–Vogtmann (BHV) metric.
  Much of this package consists of translating Megan Owen's Java
  `treestats` package into Python; see [`bhvtools/README.md`](bhvtools/README.md) for the full
  mathematical description and references.
- **`booktools`** — Fréchet means and stickiness on*open books, i.e. half-spaces glued along a common spine.
  (The `k`-spider is the univariate case). Fréchet means in the open-book are computable in closed form (i.e., no iteration required).

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
  eutheria_application     real-data application, writes several figures
  compare_estimators.py    3-panel figure: relative-error histograms of both estimators across n
figs/           output directory for generated figures
```

Each script runs one sampled process and outputs three views of it: a
journal-ready still (`.pdf`/`.png`) which is a filmstrip of the mean F_n over the
folded random walks, and and an animation (`.gif`) version. Every figure thus pairs a
random-walk trace with a visualization of the underlying space.

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
sample `i` on page `j`). For each page it estimates the Cramér root by solving for the largest root of the
empirical MGF equation `(1/n) Σ_i exp(λ Φ_j(Y_i)) = 1`, then and returns `exp(max_j θ_j Σ_i Φ_j(Y_i))`.
Because at most one folded mean is positive, the value is `≤
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

The estimator exhibits negative bias (because of the maximum), so both packages
also provide `unsticking_probability_bootstrap`, a subsampling
bias correction. The correction removes most of the bias for $n\gtrsim100$, at the cost of slightly
increasing the variance.

## Reproducing the figures

```bash
python scripts/spider_simulation.py     # figs/spider_simulation_<seed>.{pdf,png,gif}
python scripts/bhv_simulation.py         # figs/bhv_simulation_<seed>.{pdf,png,gif}
```

Each script writes all three files for one process. Sampling, seeds, sample
size, and `FILMSTRIP_N` (the sample sizes shown in the still) are set near the
top of each script. All outputs go to `figs/`.

## License

MIT — see [`LICENSE`](LICENSE). Author, paper title, and arXiv ID are marked
`TODO` in `LICENSE`, `CITATION.cff`, and `pyproject.toml`; fill these in before
release.
