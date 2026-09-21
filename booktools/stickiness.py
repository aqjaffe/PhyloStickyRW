"""Estimating stickiness on an open book: the *unsticking-probability* estimator.

This module implements ``UnstickingProbability`` (Algorithm 1) on the K-spider /
open book ``B_{K,m}``. Given samples ``Y_1,...,Y_n`` it returns a single real
number in ``(0, 1]`` -- an estimate of ``P(T > n | Y_1,...,Y_n)``, where ``T`` is
the unsticking time of the empirical Frechet mean (to be defined precisely in
the accompanying paper).

Folding coordinates
-------------------
The estimator depends on the data only through the *folding-map coordinates*

    Phi_j(Y_i) = eps_j(page(Y_i)) * r_i = +r_i if Y_i on page j, else -r_i,

i.e. the signed radial coordinate of ``Y_i`` on page ``j`` (``openbook.fold``).
The spine coordinate plays no role: the Frechet function separates (see
``frechet.py``), so stickiness is governed entirely by the page/radius part.

The per-page tilt (Cramer / Lundberg root)
------------------------------------------
For each page ``j`` define the empirical moment generating function

    psi_j(lambda) = (1/n) sum_i exp(lambda * Phi_j(Y_i)),     psi_j(0) = 1.

``psi_j`` is convex with ``psi_j'(0) = (1/n) sum_i Phi_j(Y_i) = M_j`` (the folded
mean of page ``j``). The algorithm sets

    theta_j = max { lambda in R : psi_j(lambda) = 1 }.

Because ``psi_j`` is convex and ``psi_j(0) = 1``, the level set ``{psi_j = 1}``
is ``{0}`` or ``{0, theta*}`` for a single ``theta* != 0``. Hence (proof in
``_lundberg_root``):

    theta_j > 0  <=>  M_j < 0  and  max_i Phi_j(Y_i) > 0,

and ``theta_j = 0`` otherwise. The nonzero root is the *adjustment coefficient*
/ Lundberg exponent of the increments ``Phi_j(Y_i)`` (Asmussen, *Ruin
Probabilities*, World Scientific 2000, Ch. XIII; the Cramer transform, Dembo &
Zeitouni, *Large Deviations Techniques and Applications*, 2nd ed., Springer
1998, Sec. 2.2). The returned

    exp( max_j theta_j * sum_i Phi_j(Y_i) )

is the corresponding Lundberg/Wald martingale estimate of the crossing
(unsticking) probability.

Since at most one ``M_j`` is positive (Hotz et al. 2013; see ``frechet.py``) and
``theta_j > 0`` forces ``M_j < 0`` (so ``theta_j * sum_i Phi_j = n*theta_j*M_j <=
0``), every summand is ``<= 0`` and the returned value lies in ``(0, 1]``.

Reference
---------
T. Hotz et al., "Sticky central limit theorems on open books", Ann. Appl.
Probab. 23(6):2238-2258, 2013. DOI 10.1214/12-AAP899.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .openbook import OpenBook


def _lundberg_root(phi: np.ndarray, tol: float = 1e-12, maxiter: int = 200) -> float:
    """Largest root ``theta = max{lambda : (1/n) sum_i exp(lambda*phi_i) = 1}``.

    ``phi`` is a 1-D array of folding coordinates ``Phi_j(Y_i)`` for one page.

    Correctness. Write ``h(lambda) = logsumexp_i(lambda*phi_i) - log n``; then
    ``psi(lambda) = exp(h(lambda))`` so ``{psi = 1} = {h = 0}``. ``h`` is convex
    (log-sum-exp is convex), ``h(0) = 0`` and ``h'(0) = mean(phi) = M``.

    * If ``M >= 0`` the convex ``h`` has its only nonnegative root at 0, and any
      other root is ``<= 0``; the max root is ``0``.
    * If ``M < 0`` and ``max(phi) <= 0`` then ``h`` is non-increasing on
      ``lambda >= 0`` (each ``phi_i <= 0``), so again the only root is ``0``.
    * If ``M < 0`` and ``max(phi) > 0`` then ``h`` is strictly decreasing at 0
      and ``h(lambda) -> +inf`` (dominated by the positive ``phi_i``), so there
      is a unique root ``theta* > 0``; it is the value returned.

    The root is found by Newton's method started from the right of the root,
    which converges monotonically for a convex function, with a bisection
    safeguard so the iterate never leaves a sign-change bracket. Computations
    use the log-sum-exp form, so no ``exp`` overflows.
    """
    phi = np.asarray(phi, dtype=float)
    n = phi.shape[0]
    if n == 0:
        raise ValueError("Need at least one sample.")
    logn = math.log(n)
    mean = float(phi.mean())
    pmax = float(phi.max())
    if mean >= 0.0 or pmax <= 0.0:
        return 0.0  # no positive root (winning page, or no positive coordinate)

    def h_dh(lam: float) -> Tuple[float, float]:
        z = lam * phi
        zmax = float(z.max())
        e = np.exp(z - zmax)
        s = float(e.sum())
        h = zmax + math.log(s) - logn          # logsumexp - log n
        w = e / s                              # softmax weights (sum to 1)
        dh = float(np.dot(w, phi))             # h'(lam) = sum_i w_i phi_i
        return h, dh

    # Upper bracket: psi(lam) >= exp(lam*pmax)/n, so h(lam) > 0 once
    # lam > log(2n)/pmax. Grow geometrically to be safe.
    lam_hi = math.log(2.0 * n) / pmax
    h_hi, _ = h_dh(lam_hi)
    while h_hi <= 0.0:
        lam_hi *= 2.0
        h_hi, _ = h_dh(lam_hi)

    lo, hi = 0.0, lam_hi          # h(lo) = 0 (>=, treat as lower), h(hi) > 0
    lam = lam_hi
    for _ in range(maxiter):
        h, dh = h_dh(lam)
        if h > 0.0:
            hi = lam
        else:
            lo = lam
        if abs(h) < tol:
            break
        if dh <= 0.0:             # past-root region has dh > 0; safeguard anyway
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


def unsticking_probability_from_phi(phi: np.ndarray, return_details: bool = False):
    """Run ``UnstickingProbability`` on a precomputed folding matrix ``phi``.

    Parameters
    ----------
    phi : array_like, shape (K, n)
        ``phi[j, i] = Phi_j(Y_i)`` -- the signed length of sample ``i`` on page
        ``j`` (``+r`` on its own page, ``-r`` elsewhere).
    return_details : bool
        If ``True`` also return ``(theta, term)`` with ``theta[j] = theta_j`` and
        ``term[j] = theta_j * sum_i phi[j, i]``.

    Returns
    -------
    float  (or (float, dict) if ``return_details``)
        ``exp( max_j theta_j * sum_i phi[j, i] )`` in ``(0, 1]``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must have shape (K, n).")
    theta = np.array([_lundberg_root(phi[j]) for j in range(phi.shape[0])])
    term = theta * phi.sum(axis=1)
    value = float(math.exp(float(term.max())))
    if return_details:
        return value, {"theta": theta, "term": term, "argmax_page": int(term.argmax())}
    return value


def unsticking_probability(
    book: OpenBook,
    pages,
    radii,
    spines=None,
    return_details: bool = False,
):
    """Estimate ``P(T > n | Y_1,...,Y_n)`` for a sample on an open book.

    The spine coordinates are accepted for a uniform signature with
    :func:`booktools.frechet_mean` but are not used: stickiness depends only on
    the page/radius part (the Frechet function separates, see ``frechet.py``).

    Parameters
    ----------
    book : OpenBook
    pages : array_like[int], shape (n,)    page index of each sample
    radii : array_like[float], shape (n,)  radial coordinate (>= 0) of each sample
    spines : ignored (accepted for signature compatibility)
    return_details : bool                  also return the per-page diagnostics

    Returns
    -------
    float  (or (float, dict))   a single real number in ``(0, 1]``.
    """
    pages = np.asarray(pages, dtype=int)
    radii = np.asarray(radii, dtype=float)
    if pages.shape[0] != radii.shape[0]:
        raise ValueError("pages and radii must have the same length.")
    if pages.shape[0] == 0:
        raise ValueError("Need at least one sample.")
    # Phi[j, i] = fold of sample i onto page j.
    phi = np.stack([book.fold(j, pages, radii) for j in range(book.K)])  # (K, n)
    return unsticking_probability_from_phi(phi, return_details=return_details)


# ===========================================================================
# Bootstrap bias correction (winner's-curse correction for the max-over-pages)
# ===========================================================================
# The estimator log p_hat_n = max_j theta_hat_j * S_{j,n} maximises over noisy
# per-page tilts, so it suffers a winner's-curse bias.
#
# Key structural observation. Because S_{j,n} = n * Mbar_j, the estimator is an
# exactly-scaled plug-in of an *n-free* statistical functional:
#
#     log p_hat_n = n * g(F_hat_n),
#     g(F) := max_{1<=j<=K} theta_j(F) * mu_j(F),
#
# with theta_j(F) the Lundberg root of Phi_j under F and mu_j(F) = E_F[Phi_j].
# So we bias-correct the O(1) functional g, then multiply by n. This is Efron's
# ordinary (n-out-of-n, with-replacement) bootstrap bias estimate:
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


def _g_functional(phi: np.ndarray) -> Tuple[float, np.ndarray, int]:
    """The per-observation log-probability functional ``g`` on a folding matrix.

    ``g(F_hat) = max_j theta_j(F_hat) * mean_i phi[j, i]``. Returns
    ``(g, theta, argmax_page)``.
    """
    K = phi.shape[0]
    theta = np.array([_lundberg_root(phi[j]) for j in range(K)])
    term = theta * phi.mean(axis=1)
    k = int(term.argmax())
    return float(term[k]), theta, k


def _bootstrap_logprob_bias(phi: np.ndarray, n_bootstrap: int, rng):
    """Efron bootstrap estimate of the bias of ``log p_hat_n = n * g(F_hat_n)``.

    Resamples the ``n`` columns of ``phi`` *with* replacement, ``n_bootstrap``
    times. Returns ``(bias, mc_se, argmax_stability)`` where

        bias  = n * ( mean_b g(F_hat_n^{*b}) - g(F_hat_n) )

    is the estimated bias of ``log p_hat_n``, ``mc_se`` its Monte Carlo standard
    error, and ``argmax_stability`` the fraction of resamples whose winning page
    agrees with the full-sample one (a tie diagnostic; values well below 1 mean
    the max is not locally single-valued and the plain bootstrap is suspect).
    """
    n = phi.shape[1]
    g_hat, _, k_hat = _g_functional(phi)
    gs = np.empty(n_bootstrap)
    agree = 0
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)          # with replacement
        gs[b], _, k = _g_functional(phi[:, idx])
        agree += int(k == k_hat)
    bias = float(n * (gs.mean() - g_hat))
    mc_se = float(n * gs.std(ddof=1) / math.sqrt(n_bootstrap))
    return bias, mc_se, agree / float(n_bootstrap)


def unsticking_probability_bootstrap(
    book: OpenBook,
    pages,
    radii,
    spines=None,
    *,
    m=None,
    n_bootstrap: int = 200,
    rng=None,
    return_details: bool = False,
):
    """Bootstrap bias-corrected unsticking probability on an open book.

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
    rng : None | int | numpy.random.Generator
        Seed/generator for the resampling (reproducible).

    Returns
    -------
    float (or (float, dict))
        The corrected estimate ``min(exp(log p_hat_n - bias_hat), 1)`` in
        ``(0, 1]``. When the mean has already unstuck (``log p_hat_n = 0``) the
        estimate is ``1`` and no correction is applied. With ``return_details``
        the dict carries ``logprob_hat``, ``bias``, ``bias_mc_se``,
        ``logprob_bc``, ``theta``, ``argmax_page``, ``argmax_stability``, ``m``,
        ``n_bootstrap`` and ``sticky``.
    """
    pages = np.asarray(pages, dtype=int)
    radii = np.asarray(radii, dtype=float)
    n = pages.shape[0]
    if n != radii.shape[0]:
        raise ValueError("pages and radii must have the same length.")
    if n == 0:
        raise ValueError("Need at least one sample.")

    phi = np.stack([book.fold(j, pages, radii) for j in range(book.K)])   # (K, n)
    g_hat, theta, k_hat = _g_functional(phi)
    logp_hat = float(n * g_hat)

    # Unstuck (or too few samples to resample): return the vanilla estimate.
    if logp_hat >= -1e-12 or n < 4:
        value = float(min(math.exp(logp_hat), 1.0))
        if return_details:
            return value, {"value": value, "logprob_hat": logp_hat, "bias": 0.0,
                           "bias_mc_se": 0.0, "logprob_bc": logp_hat,
                           "theta": theta, "argmax_page": k_hat,
                           "argmax_stability": 1.0, "m": None,
                           "n_bootstrap": 0, "sticky": logp_hat < -1e-12}
        return value

    bias, mc_se, stability = _bootstrap_logprob_bias(
        phi, n_bootstrap, np.random.default_rng(rng)
    )
    logp_bc = logp_hat - bias
    value = float(min(math.exp(logp_bc), 1.0))
    if return_details:
        return value, {"value": value, "logprob_hat": logp_hat, "bias": bias,
                       "bias_mc_se": mc_se, "logprob_bc": logp_bc,
                       "theta": theta, "argmax_page": k_hat,
                       "argmax_stability": stability, "m": None,
                       "n_bootstrap": n_bootstrap, "sticky": True}
    return value
