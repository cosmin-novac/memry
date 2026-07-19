"""Hybrid retrieval: vector + BM25 fused with Reciprocal Rank Fusion, then
boosted by recency and importance. Every score component is returned in
``SearchResult.signals`` so ranking stays explainable (and tunable in evals).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .backends.base import MemoryBackend
from .config import RetrievalConfig
from .models import Memory, Scope, SearchResult, parse_ts
from .providers.embeddings import Embedder


def recency_score(memory: Memory, now: datetime, half_life_days: float) -> float:
    try:
        age_days = max(0.0, (now - parse_ts(memory.updated_at)).total_seconds() / 86400.0)
    except ValueError:
        return 0.5
    return math.pow(0.5, age_days / max(half_life_days, 0.01))


def hybrid_search(
    *,
    backend: MemoryBackend,
    embedder: Embedder,
    query: str,
    scope: Scope,
    limit: int = 10,
    cfg: RetrievalConfig | None = None,
    include_invalid: bool = False,
    categories: list[str] | None = None,
    now: datetime | None = None,
) -> list[SearchResult]:
    cfg = cfg or RetrievalConfig()
    now = now or datetime.now(timezone.utc)
    n = max(limit * cfg.candidate_multiplier, limit)

    # Backends with their own fused retrieval (e.g. the Mem0 adapter) short-
    # circuit the vector/keyword fusion but still get recency/importance boosts.
    native = backend.native_search(query, scope, n)
    if native is not None:
        memories = {m.id: m for m, _ in native}
        fused = {m.id: s for m, s in native}
        signals_by_id = {m.id: {"native": s} for m, s in native}
    else:
        keyword = backend.keyword_search(
            query, scope, n, include_invalid=include_invalid, categories=categories
        )
        vector: list[tuple[Memory, float]] = []
        if embedder.dimensions:
            try:
                qvec = embedder.embed([query])[0]
                vector = backend.vector_search(
                    qvec, embedder.model_id, scope, n,
                    include_invalid=include_invalid, categories=categories,
                )
            except Exception:
                vector = []  # embedding service down -> degrade to keyword-only

        memories = {}
        fused = {}
        signals_by_id: dict[str, dict[str, float]] = {}
        for rank, (memory, sim) in enumerate(vector):
            memories[memory.id] = memory
            fused[memory.id] = fused.get(memory.id, 0.0) + cfg.vector_weight / (
                cfg.rrf_k + rank + 1
            )
            signals_by_id.setdefault(memory.id, {})["vector"] = sim
        for rank, (memory, score) in enumerate(keyword):
            memories.setdefault(memory.id, memory)
            fused[memory.id] = fused.get(memory.id, 0.0) + cfg.keyword_weight / (
                cfg.rrf_k + rank + 1
            )
            signals_by_id.setdefault(memory.id, {})["keyword"] = score

    if not memories:
        return []

    max_fused = max(fused.values()) or 1.0
    results: list[SearchResult] = []
    for memory_id, memory in memories.items():
        rec = recency_score(memory, now, cfg.recency_half_life_days)
        imp = min(max(memory.importance, 0.0), 1.0)
        norm_fused = fused[memory_id] / max_fused
        final = (
            cfg.fused_weight * norm_fused
            + cfg.recency_weight * rec
            + cfg.importance_weight * imp
        )
        signals = signals_by_id.get(memory_id, {})
        signals.update({"fused": norm_fused, "recency": rec, "importance": imp})
        results.append(SearchResult(memory=memory, score=final, signals=signals))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
