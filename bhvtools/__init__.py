"""bhvtools - Frechet means of phylogenetic trees in the BHV metric (pure Python).

Quick start
-----------
    import bhvtools as bm
    trees, taxa = bm.load_newick_file("mytrees.nwk")   # one Newick per line
    mu = bm.frechet_mean(trees)
    print(mu.to_newick())
    print("variance:", bm.frechet_variance(trees, mu))

All trees must be rooted and share the same leaf labels. Multifurcations and
zero/missing internal branch lengths are handled (treated as unresolved edges).
"""

from .tree import Tree, Taxa, compatible
from .geodesic import Geodesic, distance, geodesic_point
from .frechet import frechet_mean, frechet_variance, sum_of_squared_distances
from .stickiness import (
    unsticking_probability,
    unsticking_probability_from_phi,
    unsticking_probability_bootstrap,
    page_coordinates,
    codim1_resolutions,
)
from . import newick


def load_newick_strings(strings, taxa: Taxa = None):
    """Build Trees from a list of Newick strings. Returns (trees, taxa)."""
    if taxa is None:
        all_names = set()
        for s in strings:
            all_names |= set(newick.parse(s).leaf_names)
        taxa = Taxa(sorted(all_names))
    trees = [Tree.from_newick(s, taxa) for s in strings]
    return trees, taxa


def load_newick_file(path, taxa: Taxa = None):
    """Load a file with one Newick tree per line. Returns (trees, taxa)."""
    return load_newick_strings(newick.read_file(path), taxa)


__all__ = [
    "Tree", "Taxa", "compatible",
    "Geodesic", "distance", "geodesic_point",
    "frechet_mean", "frechet_variance", "sum_of_squared_distances",
    "unsticking_probability", "unsticking_probability_from_phi",
    "unsticking_probability_bootstrap",
    "page_coordinates", "codim1_resolutions",
    "newick", "load_newick_strings", "load_newick_file",
]
