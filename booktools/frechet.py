"""Closed-form Frechet means on an open book, and the running (online) mean.

Setup. Let ``X_1, ..., X_n`` be points of the open book ``B_{K,m}``, with
``X_k = (page_k, r_k, s_k)``, ``r_k >= 0``, ``s_k in R^{m-1}``. The Frechet
function of a candidate ``x = (i, r, s)`` is ``F(x) = sum_k d(x, X_k)^2``.

Separation. Because the spine term ``|s - s_k|^2`` appears in ``d(x, X_k)^2``
for every page assignment (same- or different-page), the Frechet function
splits as

    F(i, r, s) = [ sum_k (r - eps_i(page_k) r_k)^2 ]  +  [ sum_k |s - s_k|^2 ],
    eps_i(j) = +1 if j = i else -1,

a sum of a page/radius part (depending only on ``i, r``) and a spine part
(depending only on ``s``). Hence:

* the minimiser's spine coordinate is the ordinary Euclidean mean
  ``s_bar = (1/n) sum_k s_k``, independent of ``(i, r)``;
* the page/radius part is exactly the spider (``m = 1``) problem
  ``minimise_{i, r >= 0}  sum_k (r - eps_i(page_k) r_k)^2``.

Solving the page/radius part. For a fixed page ``i`` the unconstrained optimum
in ``r`` is the folded mean

    M_i = (1/n) sum_k eps_i(page_k) r_k = (2 S_i - S_tot) / n,

where ``S_i`` is the total radius on page ``i`` and ``S_tot`` the total radius.
For ``i != j`` one has ``M_i + M_j = -(2/n) sum_{page_k not in {i,j}} r_k <= 0``,
so **at most one** ``M_i`` is positive. Therefore, with ``i* = argmax_i M_i``:

* if ``M_{i*} > 0``  the Frechet mean is on page ``i*`` at radius ``M_{i*}``;
* if ``M_{i*} <= 0`` (all folded means non-positive) the constrained optimum is
  ``r = 0``: the mean is **sticky** on the spine.

The result is exact -- no iteration is needed -- which is the key structural
difference from the BHV tree-space mean (see the ``bhvtools`` package). The
stickiness dichotomy and the spine/page splitting are exactly the mechanism
behind the law of large numbers and central limit theorem of Hotz et al. 2013.

Reference
---------
T. Hotz et al., "Sticky central limit theorems on open books",
Ann. Appl. Probab. 23(6):2238-2258, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .openbook import BookPoint, OpenBook


def _as_arrays(book: OpenBook, pages, radii, spines):
    pages = np.asarray(pages, dtype=int)
    radii = np.asarray(radii, dtype=float)
    n = pages.shape[0]
    if radii.shape[0] != n:
        raise ValueError("pages and radii must have the same length.")
    if book.spine_dim == 0:
        spines = np.zeros((n, 0))
    else:
        spines = np.asarray(spines, dtype=float)
        if spines.shape != (n, book.spine_dim):
            raise ValueError(f"spines must have shape ({n}, {book.spine_dim}).")
    return pages, radii, spines, n


def frechet_mean(book: OpenBook, pages, radii, spines=None) -> BookPoint:
    """Exact Frechet mean of the sample on ``book``. Returns a :class:`BookPoint`.

    Parameters
    ----------
    book : OpenBook
    pages : array_like[int], shape (n,)     page index of each sample
    radii : array_like[float], shape (n,)   radial coordinate (>= 0) of each sample
    spines : array_like[float], shape (n, m-1), optional
        spine coordinate of each sample (ignored / zeros when ``m == 1``).
    """
    pages, radii, spines, n = _as_arrays(book, pages, radii, spines)
    if n == 0:
        raise ValueError("Need at least one sample.")

    # Spine part: Euclidean mean.
    s_bar = spines.mean(axis=0) if book.spine_dim else np.zeros(0)

    # Page/radius part: folded means M_i = (2 S_i - S_tot) / n.
    S_tot = radii.sum()
    S = np.array([radii[pages == i].sum() for i in range(book.K)])
    M = (2.0 * S - S_tot) / n

    i_star = int(np.argmax(M))
    if M[i_star] <= 0.0:
        return BookPoint(page=None, r=0.0, spine=s_bar, sticky=True)
    return BookPoint(page=i_star, r=float(M[i_star]), spine=s_bar, sticky=False)


@dataclass
class RunningMeans:
    """Running Frechet means ``F_1, ..., F_n`` of the prefixes of a sample.

    Attributes (all length-``n`` along the sample axis)
    ---------------------------------------------------
    page : np.ndarray[int]     page of ``F_k`` (``-1`` when sticky)
    r : np.ndarray[float]      radius of ``F_k`` (``0`` when sticky)
    spine : np.ndarray[float]  shape ``(n, m-1)``: spine coordinate of ``F_k``
    sticky : np.ndarray[bool]  whether ``F_k`` lies on the spine
    """

    page: np.ndarray
    r: np.ndarray
    spine: np.ndarray
    sticky: np.ndarray

    def __len__(self):
        return self.page.shape[0]

    def point(self, k: int) -> BookPoint:
        """The k-th running mean ``F_{k+1}`` as a :class:`BookPoint` (0-indexed)."""
        pg = None if self.sticky[k] else int(self.page[k])
        return BookPoint(page=pg, r=float(self.r[k]), spine=self.spine[k])


def running_frechet_means(book: OpenBook, pages, radii, spines=None) -> RunningMeans:
    """Frechet means of every prefix ``X_1..X_k`` (``k = 1..n``), vectorised.

    Equivalent to calling :func:`frechet_mean` on each prefix, but computed in
    one pass via cumulative sums.
    """
    pages, radii, spines, n = _as_arrays(book, pages, radii, spines)
    counts = np.arange(1, n + 1, dtype=float)

    S_tot_cum = np.cumsum(radii)                                  # (n,)
    onehot = (pages[None, :] == np.arange(book.K)[:, None])       # (K, n)
    S_cum = np.cumsum(onehot * radii[None, :], axis=1)            # (K, n)
    M = (2.0 * S_cum - S_tot_cum[None, :]) / counts[None, :]      # (K, n)

    page = np.argmax(M, axis=0).astype(int)                       # (n,)
    M_star = M[page, np.arange(n)]                                # (n,)
    sticky = M_star <= 0.0
    r = np.where(sticky, 0.0, M_star)
    page = np.where(sticky, -1, page)

    if book.spine_dim:
        spine = np.cumsum(spines, axis=0) / counts[:, None]       # (n, m-1)
    else:
        spine = np.zeros((n, 0))

    return RunningMeans(page=page, r=r, spine=spine, sticky=sticky)


def frechet_variance(book: OpenBook, pages, radii, spines, mean: Optional[BookPoint] = None) -> float:
    """Frechet variance ``(1/n) sum_k d(F, X_k)^2`` about the mean ``F``."""
    pages, radii, spines, n = _as_arrays(book, pages, radii, spines)
    if mean is None:
        mean = frechet_mean(book, pages, radii, spines)
    samples = [BookPoint(page=int(pages[k]), r=float(radii[k]),
                         spine=(spines[k] if book.spine_dim else np.zeros(0)))
               for k in range(n)]
    return sum(book.distance(mean, x) ** 2 for x in samples) / n


def folded_walks(book: OpenBook, pages, radii) -> np.ndarray:
    """Unnormalised folded random walks ``T_n^i = sum_{k<=n} eps_i(page_k) r_k``.

    Shape ``(K, n)``. These are the ``hat T_n^i`` of Hotz et al. / Barden-Le;
    note ``M_i^{(n)} = T_n^i / n`` recovers the folded means used by the mean.
    At most one walk is positive at any ``n`` (see module docstring).
    """
    pages, radii, _, n = _as_arrays(book, pages, radii, None)
    signed = np.where(pages[None, :] == np.arange(book.K)[:, None], 1.0, -1.0) * radii[None, :]
    return np.cumsum(signed, axis=1)


def folded_mean_image(book: OpenBook, running: RunningMeans) -> np.ndarray:
    """Folding-map image ``Phi_i(F_k)`` of the running mean, shape ``(K, n)``.

    For frame ``k``: if ``F_k`` is sticky the column is all zeros; otherwise the
    winning page ``i*`` carries ``+r`` and every other page carries ``-r``,
    where ``r`` is the radius of ``F_k``. This is what the spider figures plot
    (one trace per page).
    """
    n = len(running)
    img = np.zeros((book.K, n))
    for k in range(n):
        if running.sticky[k]:
            continue
        i_star = int(running.page[k])
        r = float(running.r[k])
        img[:, k] = -r
        img[i_star, k] = r
    return img
