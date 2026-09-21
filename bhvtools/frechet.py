"""Frechet (Karcher) mean of trees in BHV space.

BHV space is a Hadamard (complete CAT(0)) space, so for any finite sample the
Frechet mean
        mu = argmin_x  sum_i d(x, T_i)^2
exists and is unique (Sturm, *Contemp. Math.* 338:357-390, 2003, Prop. 4.3).

We approximate mu with Sturm's inductive-mean / proximal-point iteration
(a.k.a. the "law of large numbers" algorithm):

    mu_1 = T_{s_1}
    mu_k = geodesic( mu_{k-1}, T_{s_k} )( 1/k ),   k = 2, 3, ...

with s_k drawn uniformly from the sample. mu_k converges a.s. to mu in any
Hadamard space (Sturm 2003, Thm. 4.7; Bacak, *SIAM J. Optim.* 24:1542-1566,
2014). For phylogenetic tree space specifically see Miller, Owen & Provan,
"Polyhedral computational geometry for averaging metric phylogenetic trees",
Adv. Appl. Math. 68:51-91, 2015.

Here we run the iteration in shuffled epochs (each epoch is a random
permutation of the sample) and stop when the BHV distance between successive
epoch estimates drops below ``tol``.
"""

from __future__ import annotations
import random
from typing import List, Optional
from .tree import Tree, Taxa
from .geodesic import Geodesic, distance


def frechet_mean(trees: List[Tree],
                 max_epochs: int = 200,
                 tol: float = 1e-8,
                 seed: Optional[int] = 0,
                 init: Optional[Tree] = None,
                 return_history: bool = False):
    """Approximate the Frechet mean of ``trees``.

    Parameters
    ----------
    trees : list[Tree]      sample (>= 1 tree), all sharing one ``Taxa``.
    max_epochs : int        max number of shuffled passes over the data.
    tol : float             stop when d(mu_epoch, mu_prev_epoch) < tol.
    seed : int | None       RNG seed for reproducibility.
    init : Tree | None      starting estimate (defaults to trees[0]).
    return_history : bool   if True also return the list of per-epoch means.

    Returns
    -------
    Tree  (or (Tree, list[Tree]) when return_history).
    """
    if not trees:
        raise ValueError("Need at least one tree.")
    taxa = trees[0].taxa
    for t in trees:
        if t.taxa.names != taxa.names:
            raise ValueError("All trees must share the same taxon set.")
    if len(trees) == 1:
        return (trees[0].copy(), [trees[0].copy()]) if return_history else trees[0].copy()

    # Exact fast path: if every tree has the same topology (same clade set),
    # the whole sample lies in one orthant where BHV is Euclidean, so the
    # Frechet mean is exactly the coordinatewise arithmetic mean.
    common_topology = set(trees[0].clades)
    if all(set(t.clades) == common_topology for t in trees):
        N = len(trees)
        clades = {m: sum(t.clades[m] for t in trees) / N for m in common_topology}
        leaf = {i: sum(t.leaf_lengths[i] for t in trees) / N for i in range(taxa.n)}
        mu = Tree(clades, leaf, taxa)
        return (mu, [mu.copy()]) if return_history else mu

    rng = random.Random(seed)
    mu = (init if init is not None else trees[0]).copy()
    step = 0
    history = [mu.copy()]

    for epoch in range(max_epochs):
        order = list(range(len(trees)))
        rng.shuffle(order)
        prev = mu
        for idx in order:
            step += 1
            # weight 1/(step+1): Robbins-Monro schedule (sum a_k = inf,
            # sum a_k^2 < inf) => a.s. convergence to the Frechet mean.
            t = 1.0 / (step + 1)
            mu = Geodesic(mu, trees[idx]).point(t)
        history.append(mu.copy())
        if epoch > 0 and distance(mu, prev) < tol:
            break

    return (mu, history) if return_history else mu


def frechet_variance(trees: List[Tree], mu: Tree) -> float:
    """Frechet variance (1/N) sum_i d(mu, T_i)^2 about the point ``mu``."""
    if not trees:
        return 0.0
    return sum(distance(mu, t) ** 2 for t in trees) / len(trees)


def sum_of_squared_distances(trees: List[Tree], x: Tree) -> float:
    return sum(distance(x, t) ** 2 for t in trees)
