"""Sampling i.i.d. points on an open book ``B_{K,m}``.

A sample point is drawn by choosing a page (categorical with probabilities
``page_probs``), a radius (``radial_sampler``), and -- when ``m > 1`` -- a spine
coordinate (``spine_sampler``). Defaults give the standard exponential radius
and standard normal spine used in the examples.

The samplers are callables ``f(n, rng) -> np.ndarray`` so the caller controls
the distributions; ``rng`` is a :class:`numpy.random.Generator`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .openbook import OpenBook

Sampler = Callable[[int, np.random.Generator], np.ndarray]


def exponential_radius(scale: float = 1.0) -> Sampler:
    """Radial sampler: i.i.d. Exponential(mean = ``scale``)."""
    return lambda n, rng: rng.exponential(scale=scale, size=n)


def constant_radius(value: float = 1.0) -> Sampler:
    """Radial sampler: every radius equal to ``value`` (deterministic)."""
    return lambda n, rng: np.full(n, float(value))


def normal_spine(scale: float = 1.0) -> Callable[[int, int, np.random.Generator], np.ndarray]:
    """Spine sampler factory: i.i.d. ``N(0, scale^2 I)`` in ``R^{m-1}``."""
    return lambda n, dim, rng: rng.normal(scale=scale, size=(n, dim))


def sample(
    book: OpenBook,
    n: int,
    page_probs: Optional[Sequence[float]] = None,
    radial_sampler: Optional[Sampler] = None,
    spine_sampler: Optional[Callable[[int, int, np.random.Generator], np.ndarray]] = None,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` i.i.d. points on ``book``.

    Returns ``(pages, radii, spines)`` with shapes ``(n,)``, ``(n,)`` and
    ``(n, m-1)`` (the last is ``(n, 0)`` when ``m == 1``).

    Defaults: uniform page probabilities, Exponential(1) radii, and -- if
    ``m > 1`` -- standard-normal spine coordinates.
    """
    rng = np.random.default_rng(rng)
    if page_probs is None:
        page_probs = np.full(book.K, 1.0 / book.K)
    page_probs = np.asarray(page_probs, dtype=float)

    pages = rng.choice(book.K, size=n, p=page_probs)

    if radial_sampler is None:
        radial_sampler = exponential_radius(1.0)
    radii = np.asarray(radial_sampler(n, rng), dtype=float)

    if book.spine_dim == 0:
        spines = np.zeros((n, 0))
    else:
        if spine_sampler is None:
            spine_sampler = normal_spine(1.0)
        spines = np.asarray(spine_sampler(n, book.spine_dim, rng), dtype=float)

    return pages, radii, spines
