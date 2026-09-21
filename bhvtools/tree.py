"""The ``Tree`` object: a point in BHV tree space for a fixed leaf set.

Internal edges are stored as a dict ``{clade_bitmask: length}`` (length > 0),
pendant edges as a list/array indexed by leaf. All trees in a computation
share a single ``Taxa`` object fixing the leaf-label <-> index map, so that
clade bitmasks are comparable across trees.

Rooted compatibility of two clades X, Y is the *laminar* condition
    X ⊆ Y   or   Y ⊆ X   or   X ∩ Y = ∅,
which is the standard pairwise-compatibility criterion for rooted trees
(Semple & Steel, *Phylogenetics*, OUP 2003, Thm 3.1.4 / "hierarchy").
"""

from __future__ import annotations
from typing import Dict, List
from . import newick


class Taxa:
    """Fixed bijection leaf label <-> bit index, shared across a tree set."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        self.index = {nm: i for i, nm in enumerate(self.names)}
        self.n = len(self.names)
        self.full = (1 << self.n) - 1

    def mask_of(self, name_set) -> int:
        m = 0
        for nm in name_set:
            m |= 1 << self.index[nm]
        return m


def compatible(x: int, y: int) -> bool:
    """Rooted (laminar) compatibility of two clade bitmasks."""
    inter = x & y
    return inter == 0 or inter == x or inter == y


class Tree:
    """A phylogenetic tree as a point in BHV space.

    Parameters
    ----------
    clades : dict[int, float]   internal-edge bitmask -> positive length
    leaf_lengths : dict[int, float]  leaf index -> pendant length
    taxa : Taxa
    """

    def __init__(self, clades: Dict[int, float], leaf_lengths: Dict[int, float], taxa: Taxa):
        self.taxa = taxa
        self.clades = {m: float(l) for m, l in clades.items() if l > 0.0}
        self.leaf_lengths = {i: float(leaf_lengths.get(i, 0.0)) for i in range(taxa.n)}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_newick(cls, nwk: str, taxa: Taxa) -> "Tree":
        pt = newick.parse(nwk)
        if set(pt.leaf_names) != set(taxa.names):
            missing = set(taxa.names) ^ set(pt.leaf_names)
            raise ValueError(f"Leaf set differs from the shared taxon set; symmetric diff: {sorted(missing)}")
        clades = {taxa.mask_of(c): l for c, l in pt.clades.items()}
        leaf_lengths = {taxa.index[nm]: l for nm, l in pt.leaf_lengths.items()}
        return cls(clades, leaf_lengths, taxa)

    def to_newick(self) -> str:
        return newick.write(self.clades, self.leaf_lengths, self.taxa.names)

    # -- helpers -----------------------------------------------------------
    def copy(self) -> "Tree":
        return Tree(dict(self.clades), dict(self.leaf_lengths), self.taxa)

    def __repr__(self):
        return f"Tree(n={self.taxa.n}, internal_edges={len(self.clades)})"
