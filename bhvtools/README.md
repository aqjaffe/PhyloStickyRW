# bhvtools

Pure-Python computation of **Fréchet (Karcher) means of rooted phylogenetic
trees in the BHV metric**. No external dependencies (standard library only;
Python ≥ 3.9). A from-scratch port of the geodesic machinery in Megan Owen's
`treestats`.

All input trees must be **rooted** and share the **same leaf-label set**.
**Multifurcations** are supported (an internal branch of length 0 or with no
length is read as an unresolved/contracted edge).

## Install / use

No build step. Drop the `bhvtools/` package on your path (or `pip install -e .`)
and import it.

```python
import bhvtools as bm

trees, taxa = bm.load_newick_file("mytrees.nwk")   # one Newick tree per line
mu = bm.frechet_mean(trees)

print(mu.to_newick())
print("Fréchet variance:", bm.frechet_variance(trees, mu))

# pairwise geodesic distance and interpolation
d   = bm.distance(trees[0], trees[1])
mid = bm.geodesic_point(trees[0], trees[1], 0.5)   # midpoint tree
```

`load_newick_strings([...])` does the same from a list of Newick strings. Pass
an existing `taxa` to force a fixed leaf ordering across calls.

## What it computes, and why it is correct

**Space.** BHV tree space (Billera–Holmes–Vogtmann 2001) is a complete CAT(0)
(Hadamard) space, so geodesics and Fréchet means are **unique**.

**Geodesics.** `distance`/`Geodesic` implement the GTP algorithm of
Owen & Provan (2011): edges are partitioned into common edges, pendant (leaf)
edges, and disjoint sets `A`, `B`; the geodesic support — an ordered partition
`(A_i, B_i)` with strictly increasing ratios `‖A_i‖/‖B_i‖` and `A_i` compatible
with `B_j` for `i>j` — is found by repeatedly refining each pair via a
**minimum-weight vertex cover** (computed as a min `s`–`t` cut by Edmonds–Karp
max-flow) of the incompatibility graph. A pair splits iff its minimum cover has
weight `< 1` (the Geodesic Path Theorem). Then

```
d(T,T')^2 = Σ_common (ℓ-ℓ')^2 + Σ_leaves (ℓ-ℓ')^2 + Σ_i (‖A_i‖ + ‖B_i‖)^2.
```

Rooted compatibility of two clades is the laminar condition
`X ⊆ Y or Y ⊆ X or X ∩ Y = ∅` (Semple & Steel 2003).

**Fréchet mean.** `frechet_mean` uses Sturm's inductive-mean / proximal-point
iteration: `μ_1 = T_{s_1}`, `μ_k = geodesic(μ_{k-1}, T_{s_k})(1/k)` with `s_k`
sampled uniformly. The `1/k` schedule is Robbins–Monro (`Σ a_k = ∞`,
`Σ a_k² < ∞`), so `μ_k → μ` almost surely in any Hadamard space (Sturm 2003;
Bačák 2014). We run shuffled epochs over the sample. When all trees share one
topology the sample lies in a single (Euclidean) orthant and the exact
coordinatewise arithmetic mean is returned directly.

## Accuracy / performance notes

* The stochastic Sturm iteration converges at rate **O(1/iterations)**. The
  default (`max_epochs=200`) is adequate for ~1e-4 accuracy on moderate data;
  raise `max_epochs` for more precision. Cost per epoch is
  `O(N · geodesic)`; for hundreds of trees with < 50 leaves this is seconds.
* `tol` early-stops when successive epoch means are within `tol` (BHV
  distance). Because the harmonic schedule slows late motion, treat early
  stopping as heuristic; increasing `max_epochs` is the reliable accuracy knob.
* Results are reproducible via `seed`; pass `init=` to choose the start tree.

## Estimating stickiness

`bhvtools.unsticking_probability(samples, base_tree=T*)` estimates `P(T > n |
Y_1,…,Y_n)` at a **codimension-one stratum** — a base tree `T*` with a single
unresolved trifurcation. The stratum is locally the open book `B_{3,1}`, so the
log map at `T*` is closed-form: each sample's transverse coordinate is which of
the three resolutions it uses and that resolving edge's length
(`page_coordinates`, `codim1_resolutions`). The per-page Cramér/Lundberg tilt
and the returned value are identical to the open-book case; see
`stickiness.py` and `booktools/stickiness.py` for the full derivation and
references.

## Validation

`tests/test_bhvtools.py` checks (largely implementation-independent):
`d(T,T)=0`, symmetry, the triangle inequality; `d ≤` cone-path length; the
support invariants (strictly increasing ratios, `A_i` vs `B_j` compatibility);
**arc-length parameterisation** `d(T_1,γ(t)) = t·d` and `d(γ(t),T_2)=(1-t)·d`
(a joint check of distance *and* interpolation); the exact 4-taxon cone path
`d=2√2`; single-orthant mean = arithmetic mean; and that the mean lowers the
sum of squared distances below every sample member. Run:

```
python tests/test_bhvtools.py
```

## References

1. L. J. Billera, S. P. Holmes, K. Vogtmann. *Geometry of the space of
   phylogenetic trees.* Adv. Appl. Math. 27 (2001) 733–767.
2. M. Owen, J. S. Provan. *A fast algorithm for computing geodesic distances in
   tree space.* IEEE/ACM Trans. Comput. Biol. Bioinform. 8(1) (2011) 2–13.
3. K.-T. Sturm. *Probability measures on metric spaces of nonpositive
   curvature.* Contemp. Math. 338 (2003) 357–390.
4. M. Bačák. *Computing medians and means in Hadamard spaces.* SIAM J. Optim.
   24(3) (2014) 1542–1566.
5. E. Miller, M. Owen, J. S. Provan. *Polyhedral computational geometry for
   averaging metric phylogenetic trees.* Adv. Appl. Math. 68 (2015) 51–91.
6. C. Semple, M. Steel. *Phylogenetics.* Oxford University Press, 2003
   (rooted/laminar compatibility).
