"""Estimating stickiness in BHV tree space at a codimension-one stratum.

This module implements ``UnstickingProbability`` (Algorithm 2) for BHV tree
space ``T_N``. Given samples ``Y_1,...,Y_n`` and a base tree ``T*`` on a
codimension-one stratum -- a tree with a single degree-three (unresolved)
internal node whose three incident subtrees give the ``K = 3`` resolutions
(pages) ``A_1, A_2, A_3`` -- it returns a single real number in ``(0, 1]``: an
estimate of ``P(T > n | Y_1,...,Y_n)``, the unsticking probability.

Reduction to the 3-spider
--------------------------
A codimension-one stratum of BHV space is locally an open book ``B_{3,m}``: the
three orthants meeting along it are the three resolutions of the trifurcation,
and the spine is the stratum itself (Billera-Holmes-Vogtmann 2001; the
trifurcation plays the role of the open-book spine, see the package README).
The transverse coordinate is the length of the single resolving edge, so the
**log map at ``T*`` has a closed form on the codimension-one stratum**: for a
sample ``Y_i`` the only varying quantity is which of the three resolutions it
uses for that node and the length ``r_i`` of the corresponding edge. Writing
``page(Y_i) in {0,1,2}`` for that resolution (``None`` if ``Y_i`` is itself
unresolved at the node, i.e. on the spine) and ``r_i >= 0`` for the resolving
edge length, the page coordinate is exactly the open-book folding map

    Phi_j(Y_i) = +r_i  if page(Y_i) = j,   -r_i otherwise,   0 if unresolved.

Step 2 (the per-page Cramer/Lundberg tilt) and the returned value are then
**identical to the open-book case**; the mathematics and references are
documented in :mod:`booktools.stickiness`. To keep ``bhvtools`` free of
third-party dependencies (it is pure standard library), the tiny root solver is
reimplemented here rather than imported from ``booktools`` (which uses numpy).

Reference
---------
L. Billera, S. Holmes, K. Vogtmann, "Geometry of the space of phylogenetic
trees", Adv. Appl. Math. 27:733-767, 2001 (BHV space, CAT(0); local structure of
strata). T. Hotz et al., "Sticky central limit theorems on open books", Ann.
Appl. Probab. 23(6):2238-2258, 2013.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .tree import Taxa, Tree, compatible
from . import newick


# ---------------------------------------------------------------------------
# Step 2 core: the per-page Lundberg root and the estimator (pure stdlib).
# Mathematically identical to booktools.stickiness; see that module for proofs.
# ---------------------------------------------------------------------------

def _lundberg_root(phi: Sequence[float], tol: float = 1e-12, maxiter: int = 200) -> float:
    """Largest root ``theta = max{lambda : (1/n) sum_i exp(lambda*phi_i) = 1}``.

    Convex log-sum-exp formulation with Newton-from-the-right and a bisection
    safeguard. ``theta > 0`` iff ``mean(phi) < 0`` and ``max(phi) > 0``; else
    ``theta = 0``. See :func:`booktools.stickiness._lundberg_root` for the proof.
    """
    n = len(phi)
    if n == 0:
        raise ValueError("Need at least one sample.")
    logn = math.log(n)
    mean = sum(phi) / n
    pmax = max(phi)
    if mean >= 0.0 or pmax <= 0.0:
        return 0.0

    def h_dh(lam: float) -> Tuple[float, float]:
        z = [lam * p for p in phi]
        zmax = max(z)
        e = [math.exp(zi - zmax) for zi in z]
        s = sum(e)
        h = zmax + math.log(s) - logn
        dh = sum((ei / s) * p for ei, p in zip(e, phi))
        return h, dh

    lam_hi = math.log(2.0 * n) / pmax
    h_hi, _ = h_dh(lam_hi)
    while h_hi <= 0.0:
        lam_hi *= 2.0
        h_hi, _ = h_dh(lam_hi)

    lo, hi = 0.0, lam_hi
    lam = lam_hi
    for _ in range(maxiter):
        h, dh = h_dh(lam)
        if h > 0.0:
            hi = lam
        else:
            lo = lam
        if abs(h) < tol:
            break
        if dh <= 0.0:
            lam = 0.5 * (lo + hi)
            continue
        nxt = lam - h / dh
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - lam) < tol * (1.0 + abs(lam)):
            lam = nxt
            break
        lam = nxt
    return max(lam, 0.0)


def unsticking_probability_from_phi(phi: Sequence[Sequence[float]], return_details: bool = False):
    """Run the Step-2 estimator on a precomputed folding table ``phi``.

    ``phi[j][i] = Phi_j(Y_i)``. Returns ``exp(max_j theta_j * sum_i phi[j][i])``
    in ``(0, 1]`` (optionally with the per-page diagnostics).
    """
    K = len(phi)
    if K == 0:
        raise ValueError("phi must have at least one page.")
    theta = [_lundberg_root(row) for row in phi]
    term = [t * sum(row) for t, row in zip(theta, phi)]
    j_star = max(range(K), key=lambda j: term[j])
    value = math.exp(term[j_star])
    if return_details:
        return value, {"theta": theta, "term": term, "argmax_page": j_star}
    return value


# ---------------------------------------------------------------------------
# Step 1 (codim-1 closed form): page coordinates Phi_j(Y_i) at a base tree T*.
# ---------------------------------------------------------------------------

def _node_tree(t: Tree):
    """Parsed node tree of ``t`` (re-parses ``t.to_newick()``)."""
    return newick._parse_newick_string(t.to_newick())


def _leafsets(node) -> Tuple[FrozenSet[str], List[FrozenSet[str]]]:
    """Return ``(leaves_below, child_leafsets)`` for a parsed node."""
    if not node.children:
        return frozenset([node.name]), []
    child_sets = []
    below = frozenset()
    for c in node.children:
        cs, _ = _leafsets(c)
        child_sets.append(cs)
        below |= cs
    return below, child_sets


def codim1_resolutions(base_tree: Tree) -> List[FrozenSet[str]]:
    """The ``K = 3`` resolution clades ``A_1, A_2, A_3`` of a codim-1 ``T*``.

    ``T*`` must have exactly one degree-three (unresolved) node, i.e. a single
    node with three incident subtrees of leaf sets ``C_1, C_2, C_3``. The three
    resolutions are the pairwise unions ``C_1 u C_2``, ``C_1 u C_3``,
    ``C_2 u C_3`` -- the clade created by each way of resolving the trifurcation.
    Returned as leaf-name :class:`frozenset`\\ s.
    """
    root = _node_tree(base_tree)

    # A rooted tree's trifurcation is a node with three children. (The root may
    # itself be that node.) Collect all nodes with >= 3 children.
    polytomies = []

    def walk(node):
        _, child_sets = _leafsets(node)
        if len(node.children) >= 3:
            polytomies.append(child_sets)
        for c in node.children:
            walk(c)

    walk(root)
    if len(polytomies) != 1:
        raise ValueError(
            f"Expected exactly one trifurcation (codim-1 stratum); found "
            f"{len(polytomies)} polytomy node(s)."
        )
    child_sets = polytomies[0]
    if len(child_sets) != 3:
        raise ValueError(
            f"Trifurcation must have exactly three incident subtrees (K=3); "
            f"found {len(child_sets)}."
        )
    c1, c2, c3 = child_sets
    return [c1 | c2, c1 | c3, c2 | c3]


def page_coordinates(
    samples: List[Tree],
    resolutions: Sequence[FrozenSet[str]],
    taxa: Optional[Taxa] = None,
) -> List[List[float]]:
    """Folding coordinates ``Phi_j(Y_i)`` of the samples at the stratum.

    Parameters
    ----------
    samples : list[Tree]
        The samples ``Y_1,...,Y_n`` (sharing one ``Taxa``).
    resolutions : sequence of (frozenset[str] | int)
        The ``K`` resolution clades ``A_1,...,A_K``, each given either as a set
        of leaf names or as a clade bitmask. The three resolutions of a single
        trifurcation are pairwise incompatible, so each sample matches at most
        one (see ``compatible`` in ``tree.py``).
    taxa : Taxa, optional
        Shared taxa; defaults to ``samples[0].taxa``.

    Returns
    -------
    list[list[float]]   ``phi[j][i] = Phi_j(Y_i)``, shape ``(K, n)``.

    For each sample, its *radius* ``r_i`` is the length of whichever resolution
    edge it contains (``0`` if none -- the sample is unresolved at the node, on
    the spine). Then ``Phi_j(Y_i) = +r_i`` on the matching page and ``-r_i``
    elsewhere; all zero on the spine.
    """
    if not samples:
        raise ValueError("Need at least one sample.")
    if taxa is None:
        taxa = samples[0].taxa
    # Normalise resolutions to clade bitmasks.
    masks: List[int] = []
    for a in resolutions:
        masks.append(int(a) if isinstance(a, int) else taxa.mask_of(a))
    K = len(masks)

    n = len(samples)
    phi = [[0.0] * n for _ in range(K)]
    for i, Y in enumerate(samples):
        page = None
        r = 0.0
        for j, m in enumerate(masks):
            length = Y.clades.get(m)
            if length is not None and length > 0.0:
                page, r = j, float(length)
                break  # at most one resolution can be present
        for j in range(K):
            phi[j][i] = r if j == page else -r
    return phi


def unsticking_probability(
    samples: List[Tree],
    base_tree: Optional[Tree] = None,
    resolutions: Optional[Sequence[FrozenSet[str]]] = None,
    return_details: bool = False,
):
    """Estimate ``P(T > n | Y_1,...,Y_n)`` at a codimension-one stratum.

    Provide the ``K = 3`` ``resolutions`` directly, or a ``base_tree`` ``T*``
    from which they are derived (:func:`codim1_resolutions`). Returns a single
    real number in ``(0, 1]`` (optionally with per-page diagnostics).

    Parameters
    ----------
    samples : list[Tree]                         the samples ``Y_1,...,Y_n``
    base_tree : Tree, optional                   the codim-1 base tree ``T*``
    resolutions : sequence of clades, optional   the pages ``A_1,...,A_K``
    return_details : bool
    """
    if resolutions is None:
        if base_tree is None:
            raise ValueError("Provide either `resolutions` or a `base_tree` (T*).")
        resolutions = codim1_resolutions(base_tree)
    phi = page_coordinates(samples, resolutions)
    return unsticking_probability_from_phi(phi, return_details=return_details)


# ===========================================================================
# Bootstrap bias correction (winner's-curse correction for the max-over-pages)
# ===========================================================================
# Identical method to booktools.stickiness (see that module for the derivation
# and references). The estimator max_j theta_hat_j S_{j,n} maximises over noisy
# per-page tilts and so carries a winner's-curse bias.
#
# Key structural observation. Because S_{j,n} = n * Mbar_j, the estimator is an
# exactly-scaled plug-in of an *n-free* statistical functional:
#
#     log p_hat_n = n * g(F_hat_n),
#     g(F) := max_{1<=j<=K} theta_j(F) * mu_j(F),
#
# with theta_j(F) the Lundberg root of Phi_j under F and mu_j(F) = E_F[Phi_j].
# So we bias-correct the O(1) functional g and then multiply by n. This is
# Efron's ordinary (n-out-of-n, with-replacement) bootstrap bias estimate:
#
#     bias_hat(g) = E*[g(F_hat_n*)] - g(F_hat_n),
#     g_bc        = 2 g(F_hat_n) - (1/B) sum_b g(F_hat_n^{*b}),
#     log p_bc    = n * g_bc  =  log p_hat_n - n * bias_hat(g).
#
# No subsample size, no rate assumption, no Richardson extrapolation, and no
# finite-population correction: the only tuning knob is B (Monte Carlo error).
#
# Justification. Consistency of the bootstrap bias estimate holds whenever g is
# Hadamard differentiable at F with suitable moment conditions:
#   - B. Efron, Ann. Statist. 7:1-26, 1979, Sec. 5.
#   - Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993, Ch. 10.
#   - Shao & Tu, *The Jackknife and Bootstrap*, Springer 1995, Sec. 3.2
#     (Thm 3.9 and the differentiability conditions of Sec. 3.2.2).
#   - Hall, *The Bootstrap and Edgeworth Expansion*, 1992, Sec. 1.3, 3.11, for
#     the O(n^{-1}) residual after correction.
# Here theta_j(F) is Hadamard differentiable at any F with mu_j(F) < 0 and
# E_F exp(lambda Phi_j) < inf near the root (implicit function theorem applied
# to lambda -> E_F exp(lambda Phi_j) - 1, whose lambda-derivative at theta_j is
# strictly positive by strict convexity). The only failure point of g is a tie
# in argmax_j theta_j mu_j, where max(.) is merely *directionally* differentiable
# and the n-out-of-n bootstrap is inconsistent (Duembgen, PTRF 95:125-140, 1993;
# Fang & Santos, Rev. Econ. Stud. 86:377-412, 2019). In the sticky regime one
# page wins strictly, so ties have probability zero; the diagnostic
# ``argmax_stability`` in the details dict reports the empirical fraction of
# resamples attaining the full-sample argmax, so a tie problem is visible.
#
# Pure standard library, so the tiny helpers are reimplemented here rather than
# imported from booktools.


def _g_functional(phi) -> Tuple[float, List[float], int]:
    """The per-observation log-probability functional ``g`` on a folding table.

    ``g(F_hat) = max_j theta_j(F_hat) * mean_i phi[j][i]``. Returns
    ``(g, theta, argmax_page)``. ``phi`` is a ``K x n`` list of lists.
    """
    K = len(phi)
    n = len(phi[0])
    theta = [_lundberg_root(phi[j]) for j in range(K)]
    term = [theta[j] * (sum(phi[j]) / n) for j in range(K)]
    j_star = max(range(K), key=lambda j: term[j])
    return term[j_star], theta, j_star


def _bootstrap_logprob_bias(phi, n_bootstrap: int, rng: random.Random):
    """Efron bootstrap estimate of the bias of ``log p_hat_n = n * g(F_hat_n)``.

    Resamples the ``n`` columns of ``phi`` *with* replacement, ``n_bootstrap``
    times. Returns ``(bias, mc_se, argmax_stability)`` where

        bias = n * ( mean_b g(F_hat_n^{*b}) - g(F_hat_n) )

    is the estimated bias of ``log p_hat_n``, ``mc_se`` its Monte Carlo standard
    error, and ``argmax_stability`` the fraction of resamples whose winning page
    agrees with the full-sample one (a tie diagnostic; values well below 1 mean
    the max is not locally single-valued and the plain bootstrap is suspect).
    """
    K = len(phi)
    n = len(phi[0])
    g_hat, _, k_hat = _g_functional(phi)

    gs: List[float] = []
    agree = 0
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]      # with replacement
        sub = [[phi[j][i] for i in idx] for j in range(K)]
        g_b, _, k_b = _g_functional(sub)
        gs.append(g_b)
        if k_b == k_hat:
            agree += 1

    bias = n * (sum(gs) / n_bootstrap - g_hat)
    sd = statistics.stdev(gs) if n_bootstrap > 1 else 0.0
    mc_se = n * sd / math.sqrt(n_bootstrap)
    return bias, mc_se, agree / float(n_bootstrap)


def unsticking_probability_bootstrap(
    samples: List[Tree],
    base_tree: Optional[Tree] = None,
    resolutions: Optional[Sequence[FrozenSet[str]]] = None,
    *,
    m=None,
    n_bootstrap: int = 200,
    seed: int = 0,
    return_details: bool = False,
):
    """Bootstrap bias-corrected unsticking probability at a codim-1 stratum.

    Uses Efron's ordinary (n-out-of-n, with-replacement) bootstrap applied to the
    per-observation functional ``g(F) = max_j theta_j(F) mu_j(F)``, for which
    ``log p_hat_n = n * g(F_hat_n)``. The corrected estimate is
    ``exp(n * (2 g(F_hat_n) - E*[g(F_hat_n*)]))``, clipped to ``(0, 1]``.

    Same inputs as :func:`unsticking_probability`, plus:

    Parameters
    ----------
    m : ignored
        Retained for signature compatibility with the previous subsampling-based
        implementation. The ordinary bootstrap has no subsample size.
    n_bootstrap : int
        Number of bootstrap resamples used to estimate the bias.
    seed : int
        Seed for the resampling (reproducible).

    Returns
    -------
    float (or (float, dict))
        The corrected estimate ``min(exp(log p_hat_n - bias_hat), 1)`` in
        ``(0, 1]``. When the mean has already unstuck the estimate is ``1`` and
        no correction is applied. With ``return_details`` the dict carries
        ``logprob_hat``, ``bias``, ``bias_mc_se``, ``logprob_bc``, ``theta``,
        ``argmax_page``, ``argmax_stability``, ``m``, ``n_bootstrap`` and
        ``sticky``.
    """
    if resolutions is None:
        if base_tree is None:
            raise ValueError("Provide either `resolutions` or a `base_tree` (T*).")
        resolutions = codim1_resolutions(base_tree)
    phi = page_coordinates(samples, resolutions)         # list of K lists
    n = len(phi[0])
    g_hat, theta, k_hat = _g_functional(phi)
    logp_hat = n * g_hat

    if logp_hat >= -1e-12 or n < 4:
        value = min(math.exp(logp_hat), 1.0)
        if return_details:
            return value, {"value": value, "logprob_hat": logp_hat, "bias": 0.0,
                           "bias_mc_se": 0.0, "logprob_bc": logp_hat,
                           "theta": theta, "argmax_page": k_hat,
                           "argmax_stability": 1.0, "m": None,
                           "n_bootstrap": 0, "sticky": logp_hat < -1e-12}
        return value

    bias, mc_se, stability = _bootstrap_logprob_bias(
        phi, n_bootstrap, random.Random(seed)
    )
    logp_bc = logp_hat - bias
    value = min(math.exp(logp_bc), 1.0)
    if return_details:
        return value, {"value": value, "logprob_hat": logp_hat, "bias": bias,
                       "bias_mc_se": mc_se, "logprob_bc": logp_bc,
                       "theta": theta, "argmax_page": k_hat,
                       "argmax_stability": stability, "m": None,
                       "n_bootstrap": n_bootstrap, "sticky": True}
    return value
