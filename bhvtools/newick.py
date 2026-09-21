"""Minimal Newick I/O for *rooted* phylogenetic trees, no external dependencies.

A tree is parsed into a clade/edge-length representation suitable for BHV
geometry (see ``tree.py``):

  * a global, fixed ordering of leaf labels -> integer indices,
  * each internal edge is the *clade* (set of descendant leaves) hanging
    below it, encoded as a Python ``int`` bitmask,
  * pendant (leaf) edges are stored separately.

Multifurcations are handled natively: a node with k>2 children simply
contributes one clade. Internal edges with non-positive or missing length
are treated as *absent* (i.e. an unresolved/contracted edge), which is the
standard reading of a zero-length internal branch.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Tokenizer + recursive-descent parser
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("name", "length", "children")

    def __init__(self):
        self.name: Optional[str] = None
        self.length: Optional[float] = None
        self.children: List["_Node"] = []


def _parse_newick_string(s: str) -> _Node:
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1]
    pos = 0

    def parse_clade() -> _Node:
        nonlocal pos
        node = _Node()
        if s[pos] == "(":
            pos += 1  # consume '('
            while True:
                node.children.append(parse_clade())
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"Malformed Newick near position {pos}: {s[pos-5:pos+5]!r}")
        # optional name/label
        start = pos
        while pos < len(s) and s[pos] not in ",()":
            pos += 1
        label = s[start:pos]
        # split off branch length if present
        if ":" in label:
            nm, _, ln = label.partition(":")
            node.name = nm if nm else None
            try:
                node.length = float(ln)
            except ValueError:
                node.length = None
        else:
            node.name = label if label else None
            node.length = None
        return node

    root = parse_clade()
    if pos != len(s):
        raise ValueError(f"Trailing characters in Newick string at position {pos}: {s[pos:]!r}")
    return root


# ---------------------------------------------------------------------------
# Public structures returned by the parser
# ---------------------------------------------------------------------------

class ParsedTree:
    """Topology-independent record extracted from one Newick string.

    Attributes
    ----------
    leaf_names : list of str (in encounter order; sorted later globally)
    clades : dict[frozenset[str] -> float]
        internal edges keyed by the *set of leaf names* below them,
        with positive branch length.
    leaf_lengths : dict[str -> float]
        pendant edge length for each leaf (0.0 if missing).
    """

    def __init__(self, leaf_names, clades, leaf_lengths):
        self.leaf_names = leaf_names
        self.clades = clades
        self.leaf_lengths = leaf_lengths


def parse(newick: str) -> ParsedTree:
    root = _parse_newick_string(newick)

    leaf_names: List[str] = []
    clades: Dict[frozenset, float] = {}
    leaf_lengths: Dict[str, float] = {}

    def visit(node: _Node) -> frozenset:
        if not node.children:  # leaf
            if node.name is None:
                raise ValueError("Unnamed leaf encountered.")
            leaf_names.append(node.name)
            leaf_lengths[node.name] = float(node.length) if node.length else 0.0
            return frozenset([node.name])
        below = frozenset()
        for c in node.children:
            below |= visit(c)
        # record this internal edge unless it is trivial (the whole tree)
        # or unresolved (non-positive length). The root has length None.
        if node.length is not None and node.length > 0.0:
            clades[below] = clades.get(below, 0.0) + float(node.length)
        return below

    full = visit(root)
    # drop a clade that equals the full leaf set (trivial / root edge)
    clades.pop(full, None)
    return ParsedTree(sorted(set(leaf_names)), clades, leaf_lengths)


# ---------------------------------------------------------------------------
# Writer: reconstruct Newick from a laminar family of clades
# ---------------------------------------------------------------------------

def write(clade_masks: Dict[int, float],
          leaf_lengths: Dict[int, float],
          leaf_names: List[str]) -> str:
    """Serialise a tree given as bitmask clades back to a Newick string.

    ``clade_masks`` must be a laminar (pairwise nested-or-disjoint) family of
    internal-edge bitmasks with positive lengths; ``leaf_lengths`` maps leaf
    *index* -> pendant length; ``leaf_names[i]`` is the label of leaf i.
    """
    n = len(leaf_names)
    full = (1 << n) - 1

    # Build the nesting forest: parent(c) = smallest clade strictly containing c.
    nodes = sorted(clade_masks.keys(), key=lambda m: bin(m).count("1"))

    def is_subset(x, y):  # x strictly inside y
        return x != y and (x & y) == x

    children: Dict[int, List[int]] = {m: [] for m in clade_masks}
    parent_of: Dict[int, Optional[int]] = {}
    for i, c in enumerate(nodes):
        par = None
        best = None
        for d in nodes[i + 1:]:
            if is_subset(c, d):
                if best is None or bin(d).count("1") < best:
                    best = bin(d).count("1")
                    par = d
        parent_of[c] = par
        if par is not None:
            children[par].append(c)

    top_clades = [c for c in nodes if parent_of[c] is None]

    def leaves_of(mask):
        return [i for i in range(n) if mask & (1 << i)]

    def render(mask, length: Optional[float]):
        # children of this clade = sub-clades whose parent is `mask`,
        # plus singleton leaves directly under it.
        sub = children[mask]
        covered = 0
        for sc in sub:
            covered |= sc
        loose_leaves = [i for i in leaves_of(mask) if not (covered & (1 << i))]
        parts = []
        for sc in sub:
            parts.append(render(sc, clade_masks[sc]))
        for li in loose_leaves:
            parts.append(f"{leaf_names[li]}:{leaf_lengths.get(li, 0.0):.10g}")
        inner = ",".join(parts)
        s = f"({inner})"
        if length is not None:
            s += f":{length:.10g}"
        return s

    # root spans the full leaf set
    covered = 0
    for c in top_clades:
        covered |= c
    loose = [i for i in range(n) if not (covered & (1 << i))]
    parts = [render(c, clade_masks[c]) for c in top_clades]
    for li in loose:
        parts.append(f"{leaf_names[li]}:{leaf_lengths.get(li, 0.0):.10g}")
    return "(" + ",".join(parts) + ");"


def read_file(path: str) -> List[str]:
    """Read a file of Newick trees, one per line (Owen-style). Blank lines skipped."""
    trees = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                trees.append(line)
    return trees
