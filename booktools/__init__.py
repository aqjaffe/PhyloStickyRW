"""booktools -- Frechet means and stickiness on open books ``B_{K,m}``.

An open book is ``K`` half-space pages ``R_{>=0} x R^{m-1}`` glued along a
common spine ``R^{m-1}`` (Hotz et al. 2013). The k-spider is the case
``m = 1``. Unlike the BHV tree-space mean (see the ``bhvtools`` package), the
open-book Frechet mean is available in **closed form**: the spine coordinate is
the Euclidean mean of the spine coordinates, and the page/radius part is solved
by the folding-map argument, exhibiting *stickiness* (the mean lies on the
measure-zero spine for an open set of samples).

Quick start
-----------
    import numpy as np
    import booktools as bk

    book = bk.OpenBook(K=3, m=1)                 # the 3-spider
    pages, radii, spines = bk.sample(book, 200, rng=0)
    F = bk.frechet_mean(book, pages, radii, spines)
    print("sticky" if F.sticky else f"page {F.page}, r = {F.r:.3f}")

    run = bk.running_frechet_means(book, pages, radii, spines)   # F_1..F_n

Reference: T. Hotz et al., "Sticky central limit theorems on open books",
Ann. Appl. Probab. 23(6):2238-2258, 2013. DOI 10.1214/12-AAP899.
"""

from .openbook import OpenBook, BookPoint
from .frechet import (
    frechet_mean,
    running_frechet_means,
    frechet_variance,
    folded_walks,
    folded_mean_image,
    RunningMeans,
)
from .sampling import (
    sample,
    exponential_radius,
    constant_radius,
    normal_spine,
)
from .stickiness import (
    unsticking_probability,
    unsticking_probability_from_phi,
    unsticking_probability_bootstrap,
)

__all__ = [
    "OpenBook", "BookPoint",
    "frechet_mean", "running_frechet_means", "frechet_variance",
    "folded_walks", "folded_mean_image", "RunningMeans",
    "sample", "exponential_radius", "constant_radius", "normal_spine",
    "unsticking_probability", "unsticking_probability_from_phi",
    "unsticking_probability_bootstrap",
]

__version__ = "0.1.0"
