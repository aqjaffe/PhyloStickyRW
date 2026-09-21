"""Open-book geometry ``B_{K,m}``: K half-space pages glued along a spine.

An *open book* (Hotz et al. 2013) is the metric space obtained by gluing a
disjoint union of ``K`` copies of the closed half-space

        page_i  =  R_{>=0} x R^{m-1},     i = 0, ..., K-1,

along their common boundary hyperplane, the *spine* ``R^{m-1}`` (the locus
``r = 0``). Here ``m >= 1`` is the dimension of each page (equivalently, of the
whole open book) and the spine has dimension ``m-1``. The **k-spider** -- k
half-lines glued at a single point -- is the case ``m = 1``, where the spine
``R^0`` is one point. (In Hotz et al. the spider on k leaves is "the open book
of dimension 1 with k leaves".)

A point is written ``(page i, radius r >= 0, spine coordinate s in R^{m-1})``;
every point with ``r = 0`` is identified with the spine, regardless of ``i``.

Metric
------
Within one page the space is a convex Euclidean half-space, so the geodesic
between two same-page points is the straight segment and

        d((i,r,s), (i,r',s'))^2 = (r - r')^2 + |s - s'|^2.

For points on different pages every path must cross the spine; reflecting one
page onto the other across the spine hyperplane (an isometry of ``R^m``) gives

        d((i,r,s), (j,r',s'))^2 = (r + r')^2 + |s - s'|^2,    i != j.

When ``r = 0`` or ``r' = 0`` the two formulas coincide, consistent with the
identification on the spine. (See Hotz et al. 2013, Sec. 2.)

Folding maps
------------
For each page ``i`` the *folding map*

        F_i(j, r, s) = ( eps_i(j) * r ,  s )  in  R^m = R x R^{m-1},
        eps_i(j)     = +1 if j = i else -1,

is 1-Lipschitz and restricts to an isometry onto its image on page ``i``. The
folding maps linearise the Frechet-mean computation; see ``frechet.py``.

Reference
---------
T. Hotz, S. Huckemann, H. Le, J. S. Marron, J. C. Mattingly, E. Miller,
J. Nolen, M. Owen, V. Patrangenaru, S. Skwerer,
"Sticky central limit theorems on open books",
Ann. Appl. Probab. 23(6):2238-2258, 2013. DOI 10.1214/12-AAP899.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BookPoint:
    """A single point of an open book.

    Attributes
    ----------
    page : int | None
        Page index in ``0..K-1``; ``None`` iff the point lies on the spine.
    r : float
        Radial coordinate (distance to the spine within the page), ``r >= 0``.
    spine : np.ndarray
        Spine coordinate, shape ``(m-1,)``. Empty array when ``m == 1``.
    sticky : bool
        ``True`` iff the point lies on the spine (equivalently ``r == 0``).
    """

    page: Optional[int]
    r: float
    spine: np.ndarray
    sticky: bool = False

    def __post_init__(self):
        self.spine = np.asarray(self.spine, dtype=float).reshape(-1)
        if self.r <= 0.0:
            self.r = 0.0
            self.page = None
            self.sticky = True


class OpenBook:
    """The open book ``B_{K,m}``: ``K`` pages of dimension ``m``.

    Parameters
    ----------
    K : int
        Number of pages (leaves), ``K >= 1``.
    m : int
        Dimension of each page, ``m >= 1``. The spine has dimension ``m-1``.
        ``m == 1`` is the k-spider.
    """

    def __init__(self, K: int, m: int = 1):
        if K < 1:
            raise ValueError("K (number of pages) must be >= 1.")
        if m < 1:
            raise ValueError("m (page dimension) must be >= 1.")
        self.K = int(K)
        self.m = int(m)
        self.spine_dim = self.m - 1

    def __repr__(self):
        return f"OpenBook(K={self.K}, m={self.m}, spine_dim={self.spine_dim})"

    # -- points ------------------------------------------------------------
    def point(self, page: Optional[int], r: float, spine=None) -> BookPoint:
        """Construct a :class:`BookPoint` on this open book (with validation)."""
        if spine is None:
            spine = np.zeros(self.spine_dim)
        spine = np.asarray(spine, dtype=float).reshape(-1)
        if spine.size != self.spine_dim:
            raise ValueError(f"spine must have length {self.spine_dim}, got {spine.size}.")
        if r > 0.0 and not (0 <= (page if page is not None else -1) < self.K):
            raise ValueError(f"page must be in 0..{self.K - 1} for an off-spine point.")
        return BookPoint(page=page, r=float(r), spine=spine)

    def spine_point(self, spine=None) -> BookPoint:
        """The spine point with the given spine coordinate (``r = 0``)."""
        return self.point(None, 0.0, spine)

    # -- metric ------------------------------------------------------------
    def distance(self, p: BookPoint, q: BookPoint) -> float:
        """Geodesic distance ``d(p, q)`` in the open-book metric."""
        ds2 = float(np.sum((p.spine - q.spine) ** 2)) if self.spine_dim else 0.0
        # same page, or either point on the spine -> Euclidean within a page;
        # different pages with both off-spine -> folded (r + r') term.
        if p.page == q.page or p.r == 0.0 or q.r == 0.0:
            dr2 = (p.r - q.r) ** 2
        else:
            dr2 = (p.r + q.r) ** 2
        return math.sqrt(dr2 + ds2)

    # -- folding -----------------------------------------------------------
    def fold(self, i: int, pages, radii) -> np.ndarray:
        """Folded radial coordinate ``eps_i(page) * r`` for page ``i``.

        Vectorised over arrays ``pages`` (int) and ``radii`` (float): returns
        ``+r`` where ``page == i`` and ``-r`` elsewhere -- the first component
        of the folding map ``F_i`` (the spine component is the identity).
        """
        pages = np.asarray(pages)
        radii = np.asarray(radii, dtype=float)
        sign = np.where(pages == i, 1.0, -1.0)
        return sign * radii
