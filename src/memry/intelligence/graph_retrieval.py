"""Relational retrieval: reach the memories that similarity cannot.

Experiments (see evals) showed pure hybrid retrieval scores a flat zero on
multi-hop questions - "what tool does Ada use?" is answered by a memory that
names neither "Ada" nor "tool", so no embedder can find it. Following typed
relations from the query's entities does find it, reliably, at any store size.

The move is deliberately ranked by graph distance, not by text similarity: the
multi-hop answer is relevant *because* it is two typed hops from the query, even
though it shares no words with it. Those candidates are then fused (RRF) with the
hybrid results, so direct lookups keep their strong lexical/semantic ranking and
relational questions gain the hop-reachable memories on top.
"""

from __future__ import annotations

from collections import defaultdict

from ..backends.base import MemoryBackend
from ..models import Scope

_MIN_SURFACE = 3  # ignore 1-2 char "entities" that would match everything


def detect_query_entities(
    backend: MemoryBackend, scope: Scope, query: str, *, cap: int = 200_000
) -> list[str]:
    """Entity ids whose name appears in the query. Cheap surface match against
    the (bounded) entity vocabulary; good enough to seed traversal."""
    ql = f" {query.lower()} "
    hits: list[str] = []
    for e in backend.list_entities(scope, limit=cap):
        name = (e.normalized or e.name.lower()).strip()
        if len(name) >= _MIN_SURFACE and (name in ql or f" {name} " in ql):
            hits.append(e.id)
    return hits


def expand_entities(
    backend: MemoryBackend, seeds: list[str], *, hops: int = 2
) -> list[str]:
    """Entities reachable from the seeds over typed relations, nearest first.

    Falls back to co-occurrence (entities sharing a memory) only if no typed
    relation touches the seeds, mirroring the PPR-as-fallback result: typed
    edges when present, structural proximity otherwise."""
    reached: set[str] = set(seeds)
    order: list[str] = []
    frontier = set(seeds)
    for _ in range(hops):
        rels = backend.relations_of(list(frontier))
        nxt: set[str] = set()
        for r in rels:
            for endpoint in (r.subject, r.object):
                if endpoint not in reached:
                    nxt.add(endpoint)
        for e in nxt:
            reached.add(e)
            order.append(e)
        frontier = nxt
        if not frontier:
            break
    if not order:
        # No typed edges (e.g. memories predating relation extraction): fall back
        # to co-occurrence proximity. The benchmark showed localized PageRank over
        # co-occurrence is the reliable relation-free option (~0.90, scale-stable),
        # so weight neighbours by how strongly they co-occur with the seeds rather
        # than dumping a flat pool.
        order = _cooccurrence_expand(backend, seeds, hops=hops)
    return order


def _cooccurrence_expand(
    backend: MemoryBackend, seeds: list[str], *, hops: int, per_entity: int = 25
) -> list[str]:
    """Entities near the seeds by shared-memory co-occurrence, ranked by a
    localized PageRank so the strongest structural neighbours come first."""
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reached: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: set[str] = set()
        for e in frontier:
            for mem in backend.entity_memories(e, limit=per_entity):
                others = [x.id for x in backend.entities_of_memory(mem.id)]
                for a in others:
                    for b in others:
                        if a != b:
                            adj[a][b] += 1.0
                    if a not in reached:
                        nxt.add(a)
        reached |= nxt
        frontier = nxt
        if not frontier:
            break
    nodes = list(reached | set(adj))
    if not nodes:
        return []
    # weighted personalized PageRank, localized to this neighbourhood
    idx = {n: i for i, n in enumerate(nodes)}
    seed_mass = 1.0 / max(len(seeds), 1)
    rank = {n: (seed_mass if n in seeds else 0.0) for n in nodes}
    alpha = 0.85
    for _ in range(20):
        nxt_rank = {n: (1 - alpha) * (seed_mass if n in seeds else 0.0) for n in nodes}
        for n in nodes:
            out = adj.get(n)
            if not out:
                continue
            total = sum(out.values())
            share = alpha * rank[n] / total
            for m, w in out.items():
                if m in idx:
                    nxt_rank[m] += share * w
        rank = nxt_rank
    ranked = sorted((n for n in nodes if n not in seeds), key=lambda n: -rank[n])
    return ranked


def relational_memory_ids(
    backend: MemoryBackend,
    scope: Scope,
    query: str,
    *,
    hops: int = 2,
    per_entity: int = 25,
) -> list[str]:
    """Ordered memory ids reached by traversing relations from the query's
    entities. Empty when the query names no known entity."""
    seeds = detect_query_entities(backend, scope, query)
    if not seeds:
        return []
    expanded = expand_entities(backend, seeds, hops=hops)
    ids: list[str] = []
    seen: set[str] = set()
    # expanded (hop-reachable) entities first - those carry the non-obvious,
    # multi-hop answers; the seed's own memories come after (hybrid already has
    # them covered).
    for entity_id in expanded + seeds:
        for mem in backend.entity_memories(entity_id, limit=per_entity):
            if mem.id not in seen:
                seen.add(mem.id)
                ids.append(mem.id)
    return ids
