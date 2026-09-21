"""Geodesics in BHV tree space via the GTP algorithm (Owen & Provan 2011).

Reference
---------
M. Owen and J. S. Provan, "A fast algorithm for computing geodesic distances
in tree space", IEEE/ACM Trans. Comput. Biol. Bioinform. 8(1):2-13, 2011.
Underlying geometry: L. Billera, S. Holmes, K. Vogtmann, "Geometry of the
space of phylogenetic trees", Adv. Appl. Math. 27:733-767, 2001 (BHV space is
CAT(0), so geodesics are unique).

Algorithm summary
------------------
Given trees T1, T2 with the same leaves, edges split into:
  * common edges (same clade in both): contribute (l - l')^2 each;
  * disjoint sets A = edges(T1) \\ edges(T2), B = edges(T2) \\ edges(T1).
The squared geodesic distance is
    d^2 = sum_common (l-l')^2  +  sum_leaves (l-l')^2  +  sum_i (||A_i|| + ||B_i||)^2,
where {(A_i, B_i)} is the *support*: an ordered partition of (A, B) with
ratios ||A_1||/||B_1|| < ... < ||A_k||/||B_k|| and A_i compatible with B_j for
i > j. The support is found by repeatedly refining a pair via a minimum-weight
vertex cover (computed as a min s-t cut) of the incompatibility graph; a pair
splits iff that minimum cover has weight < 1 (Owen-Provan, "Geodesic Path
Theorem"). See ``frechet.py`` for the Frechet mean built on top of this.
"""

from __future__ import annotations
from math import sqrt
from typing import Dict, List, Tuple
from .tree import Tree, Taxa, compatible

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Min-weight vertex cover of a bipartite graph via Edmonds-Karp max-flow.
# Real-valued capacities: BFS (shortest) augmenting paths guarantee termination.
# ---------------------------------------------------------------------------

def _min_vertex_cover(A: List[int], B: List[int],
                      wA: Dict[int, float], wB: Dict[int, float],
                      incident: Dict[int, List[int]]):
    """Return (flow_value, sourceside_A, sourceside_B).

    Nodes: source s=0, sink t=1, A-vertices 2..2+|A|-1, B-vertices after.
    Capacities: s->a = wA[a]; b->t = wB[b]; a->b = INF for incompatibilities.
    Min cut weight = min-weight vertex cover (Koenig / LP duality).
    Returns the flow value and the residual-reachable A/B vertices (source side).
    """
    nA, nB = len(A), len(B)
    a_id = {a: 2 + i for i, a in enumerate(A)}
    b_id = {b: 2 + nA + j for j, b in enumerate(B)}
    N = 2 + nA + nB
    INF = sum(wA.values()) + sum(wB.values()) + 1.0

    # adjacency with parallel residual edges; cap as dict keyed (u,v)
    adj: Dict[int, List[int]] = {u: [] for u in range(N)}
    cap: Dict[Tuple[int, int], float] = {}

    def add(u, v, c):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
        cap[(u, v)] = cap.get((u, v), 0.0) + c
        cap.setdefault((v, u), 0.0)

    for a in A:
        add(0, a_id[a], wA[a])
    for b in B:
        add(b_id[b], 1, wB[b])
    for a in A:
        for b in incident.get(a, ()):
            add(a_id[a], b_id[b], INF)

    from collections import deque

    def bfs_augment():
        parent = {0: None}
        q = deque([0])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent and cap.get((u, v), 0.0) > _EPS:
                    parent[v] = u
                    if v == 1:
                        # found path; compute bottleneck
                        path = []
                        cur = 1
                        bott = INF
                        while parent[cur] is not None:
                            p = parent[cur]
                            bott = min(bott, cap[(p, cur)])
                            path.append((p, cur))
                            cur = p
                        for (p, c) in path:
                            cap[(p, c)] -= bott
                            cap[(c, p)] += bott
                        return bott
                    q.append(v)
        return 0.0

    flow = 0.0
    while True:
        aug = bfs_augment()
        if aug <= _EPS:
            break
        flow += aug

    # residual reachable set from source
    reach = set()
    from collections import deque as _dq
    q = _dq([0]); reach.add(0)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in reach and cap.get((u, v), 0.0) > _EPS:
                reach.add(v); q.append(v)

    srcA = [a for a in A if a_id[a] in reach]
    srcB = [b for b in B if b_id[b] in reach]
    return flow, srcA, srcB


# ---------------------------------------------------------------------------
# Support construction
# ---------------------------------------------------------------------------

def _norm(masks: List[int], lengths: Dict[int, float]) -> float:
    return sqrt(sum(lengths[m] ** 2 for m in masks))


def _build_support(A: List[int], B: List[int],
                   lenA: Dict[int, float], lenB: Dict[int, float]):
    """Return list of (A_i, B_i) pairs (lists of masks), sorted by ratio."""
    work = [(list(A), list(B))]
    done = []
    while work:
        Ai, Bi = work.pop()
        if not Ai or not Bi:
            done.append((Ai, Bi))
            continue
        # incompatibility edges within this pair
        incident = {a: [b for b in Bi if not compatible(a, b)] for a in Ai}
        has_edge = any(incident[a] for a in Ai)
        if not has_edge:
            # fully compatible: shrink/grow independently (Euclidean halves)
            done.append((Ai, []))
            done.append(([], Bi))
            continue
        normA = _norm(Ai, lenA)
        normB = _norm(Bi, lenB)
        wA = {a: lenA[a] ** 2 / normA ** 2 for a in Ai}
        wB = {b: lenB[b] ** 2 / normB ** 2 for b in Bi}
        flow, srcA, srcB = _min_vertex_cover(Ai, Bi, wA, wB, incident)
        if flow >= 1.0 - 1e-9:
            done.append((Ai, Bi))  # non-refinable terminal pair
        else:
            srcA_set, srcB_set = set(srcA), set(srcB)
            # pair 1 (smaller ratio) = sink-side vertices; pair 2 = source-side
            A1 = [a for a in Ai if a not in srcA_set]
            B1 = [b for b in Bi if b not in srcB_set]
            A2 = [a for a in Ai if a in srcA_set]
            B2 = [b for b in Bi if b in srcB_set]
            work.append((A1, B1))
            work.append((A2, B2))

    def ratio(pair):
        Ai, Bi = pair
        nA = _norm(Ai, lenA)
        nB = _norm(Bi, lenB)
        if nB == 0.0:
            return float("inf")
        return nA / nB

    done = [p for p in done if p[0] or p[1]]
    done.sort(key=ratio)
    return done


def _decompose(t1: Tree, t2: Tree):
    """Split into common edges, disjoint A/B, and leaf-edge contribution."""
    c1, c2 = t1.clades, t2.clades
    common = c1.keys() & c2.keys()
    A = [m for m in c1 if m not in common]
    B = [m for m in c2 if m not in common]
    common_sq = sum((c1[m] - c2[m]) ** 2 for m in common)
    leaf_sq = sum((t1.leaf_lengths[i] - t2.leaf_lengths[i]) ** 2 for i in range(t1.taxa.n))
    return common, A, B, common_sq, leaf_sq


class Geodesic:
    """Cached geodesic between two trees; supports distance and interpolation."""

    def __init__(self, t1: Tree, t2: Tree):
        if t1.taxa is not t2.taxa and t1.taxa.names != t2.taxa.names:
            raise ValueError("Trees must share the same taxon set.")
        self.t1, self.t2 = t1, t2
        self.taxa = t1.taxa
        self.common, A, B, self.common_sq, self.leaf_sq = _decompose(t1, t2)
        self.support = _build_support(A, B, t1.clades, t2.clades)

    def distance(self) -> float:
        s = self.common_sq + self.leaf_sq
        for Ai, Bi in self.support:
            nA = _norm(Ai, self.t1.clades)
            nB = _norm(Bi, self.t2.clades)
            s += (nA + nB) ** 2
        return sqrt(max(s, 0.0))

    def point(self, t: float) -> Tree:
        """Tree at parameter t in [0,1]; point(0)=t1, point(1)=t2."""
        c1, c2 = self.t1.clades, self.t2.clades
        clades: Dict[int, float] = {}
        # common edges: linear interpolation
        for m in self.common:
            l = (1 - t) * c1[m] + t * c2[m]
            if l > _EPS:
                clades[m] = l
        # support pairs
        for Ai, Bi in self.support:
            nA = _norm(Ai, c1)
            nB = _norm(Bi, c2)
            tot = nA + nB
            if tot == 0.0:
                continue
            tau = nA / tot  # transition time
            if t <= tau:
                # A edges shrinking, B absent
                coefA = (nA - t * tot) / nA if nA > 0 else 0.0
                for a in Ai:
                    l = coefA * c1[a]
                    if l > _EPS:
                        clades[a] = l
            if t >= tau:
                coefB = (t * tot - nA) / nB if nB > 0 else 0.0
                for b in Bi:
                    l = coefB * c2[b]
                    if l > _EPS:
                        clades[b] = l
        # leaf edges
        leaf = {i: (1 - t) * self.t1.leaf_lengths[i] + t * self.t2.leaf_lengths[i]
                for i in range(self.taxa.n)}
        return Tree(clades, leaf, self.taxa)


def distance(t1: Tree, t2: Tree) -> float:
    """BHV geodesic distance between two trees."""
    return Geodesic(t1, t2).distance()


def geodesic_point(t1: Tree, t2: Tree, t: float) -> Tree:
    """Point at fraction ``t`` along the geodesic from ``t1`` to ``t2``."""
    return Geodesic(t1, t2).point(t)
