"""MemoryStore - the public API of Memry.

Applications (and the MCP/REST servers) call this facade only; storage
backends, LLMs, and embedders are all replaceable underneath it.

    from memry import MemoryStore

    store = MemoryStore()
    store.add("I'm Ada. I prefer TypeScript and live in Berlin.", user_id="ada")
    results = store.search("where does the user live?", user_id="ada")
    context = store.reconstruct_context("help me set up my editor", user_id="ada")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .backends import build_backend
from .backends.base import MemoryBackend
from .config import Config
from .intelligence.clustering import propose_synthetic_tags, suggest_canonical_merges
from .intelligence.summaries import cluster_vectors, summarize_cluster
from .intelligence.context import build_context
from .intelligence.decay import decay_sweep, effective_importance
from .intelligence.entities import (
    classify_entity_types,
    resolve_mentions,
    resolve_open_proposals,
)
from .intelligence.graph_retrieval import relational_memory_ids
from .intelligence.extraction import (
    extract_facts,
    extract_relations,
    verbatim_candidates,
    verify_coverage,
)
from .intelligence.reconcile import reconcile_candidate
from .models import (
    MEMORY_TYPES,
    AddAction,
    AddResult,
    CandidateFact,
    Collection,
    ContextResult,
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MemoryType,
    MergeProposal,
    Relation,
    Scope,
    SearchResult,
    SyntheticTag,
    utcnow,
)
from .providers.embeddings import Embedder, build_embedder
from .providers.llm import LLM, build_llm
from .retrieval import hybrid_search


def _owned(record: Any, owner_prefix: str | None) -> bool:
    """Ownership gate for every id-addressed operation.

    ``owner_prefix`` None means admin: no confinement. Otherwise the record's
    ``user_id`` must sit under that namespace prefix.

    This lives in the store rather than at each call site on purpose. Ids are
    guessable-ish and callers are many (REST handlers, MCP tools, the CLI, the
    dashboard); one forgotten check is a cross-account read. Putting the gate
    behind the same door as the data means a new caller cannot skip it.
    """
    if record is None:
        return False
    if owner_prefix is None:
        return True
    user_id = getattr(record, "user_id", None)
    return bool(user_id) and str(user_id).startswith(owner_prefix)


def _tag_run_key(user_id: str | None) -> str:
    """Meta key under which the last tag-abstraction run time is stamped."""
    return f"tag_abstraction:last_run:{user_id or ''}"


def _within(created_at: str, since: str | None, until: str | None) -> bool:
    """Is an ISO ``created_at`` inside the [since, until] window?

    Bounds accept a plain date (YYYY-MM-DD) or a full ISO timestamp; a date-only
    ``until`` is inclusive of that whole day, which is what a human means by
    "up to the 22nd".
    """
    def _dt(value: str, *, end_of_day: bool) -> datetime | None:
        value = value.strip()
        if not value:
            return None
        try:
            if len(value) == 10:  # date only
                base = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
                return base + timedelta(days=1) if end_of_day else base
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            return None

    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return True  # unparseable timestamps are never filtered out
    lo = _dt(since, end_of_day=False) if since else None
    hi = _dt(until, end_of_day=True) if until else None
    if lo is not None and created < lo:
        return False
    if hi is not None and created >= hi:
        return False
    return True


class MemoryStore:
    def __init__(
        self,
        config: Config | None = None,
        *,
        backend: MemoryBackend | None = None,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or Config.load()
        self.backend = backend or build_backend(self.config)
        self.llm = llm or build_llm(self.config.llm)
        self.embedder = embedder or build_embedder(self.config.embedding)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------
    def add(
        self,
        content: str | list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        memory_type: MemoryType = "semantic",
        importance: float = 0.5,
        categories: list[str] | None = None,
    ) -> AddResult:
        """Record raw content and derive memories from it.

        ``infer=True`` runs extraction + reconciliation (needs an LLM;
        degrades to verbatim mode without one). ``infer=False`` stores the
        content directly as a single memory - the "just save this fact" path.
        """
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        messages = (
            [{"role": "user", "content": content}] if isinstance(content, str) else content
        )
        episodes = [
            Episode(
                content=m.get("content", ""),
                role=m.get("role", "user"),
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata or {},
            )
            for m in messages
            if (m.get("content") or "").strip()
        ]
        if not episodes:
            return AddResult()
        self.backend.add_episodes(episodes)
        episode_ids = [e.id for e in episodes]

        candidates: list[CandidateFact]
        warnings: list[str] = []
        if not infer:
            text = content if isinstance(content, str) else "\n".join(
                m.get("content", "") for m in messages
            )
            candidates = [
                CandidateFact(
                    content=text.strip(),
                    memory_type=memory_type,
                    importance=importance,
                    categories=categories or [],
                )
            ]
        elif self.llm.available:
            try:
                candidates = extract_facts(self.llm, messages)
            except Exception as exc:
                # Provider outage / exhausted credits must not lose the save:
                # degrade to verbatim, tell the caller, and flag the memories
                # so distillation can be re-run later (store.distill).
                candidates = self._pending_verbatim(messages)
                warnings.append(
                    f"extraction failed; stored verbatim instead (distill later): {exc}"
                )
        else:
            candidates = self._pending_verbatim(messages)

        actions = self._apply_candidates(candidates, scope, episode_ids)

        # Post-write audit: extraction is lossy and non-deterministic, and a
        # dropped constraint is invisible in a "success" response. One cheap
        # LLM pass compares input against what landed and reports the gap.
        if infer and self.llm.available and actions:
            stored = [a.content for a in actions if a.content]
            try:
                missing = verify_coverage(self.llm, messages, stored)
            except Exception:
                missing = []  # audit is best-effort; never fail the save
            if missing:
                warnings.append(
                    "some details were not captured as facts; consider saving "
                    "them explicitly: " + "; ".join(missing)
                )
        return AddResult(episode_ids=episode_ids, actions=actions, warnings=warnings)

    @staticmethod
    def _pending_verbatim(messages: list[dict[str, str]]) -> list[CandidateFact]:
        """Verbatim candidates flagged for later distillation."""
        candidates = verbatim_candidates(messages)
        for candidate in candidates:
            candidate.metadata = {"pending_distillation": True}
        return candidates

    def _apply_candidates(
        self,
        candidates: list[CandidateFact],
        scope: Scope,
        episode_ids: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[AddAction]:
        """Reconcile candidates into the store (shared by add and distill).

        ``exclude_ids`` keeps memories out of the similarity set: distillation
        must not reconcile facts against the verbatim memory they came from,
        and candidates from ONE call must not reconcile against each other.
        The extractor already split the payload into discrete facts; without
        this, fact N finds facts 1..N-1 as "similar" and the reconciler chains
        them into a single memory via lossy UPDATE rewrites (the observed
        ADD followed by N UPDATEs on one id). Memories touched by this call
        are therefore accumulated into the exclusion set as we go.
        """
        excluded: set[str] = set(exclude_ids or ())
        actions: list[AddAction] = []
        for candidate in candidates:
            similar = hybrid_search(
                backend=self.backend,
                embedder=self.embedder,
                query=candidate.content,
                scope=scope,
                limit=self.config.retrieval.reconcile_similarity_limit,
                cfg=self.config.retrieval,
            )
            if excluded:
                similar = [r for r in similar if r.memory.id not in excluded]
            action = reconcile_candidate(
                candidate=candidate,
                scope=scope,
                similar=similar,
                backend=self.backend,
                embedder=self.embedder,
                llm=self.llm,
                episode_ids=episode_ids,
                retrieval_cfg=self.config.retrieval,
            )
            actions.append(action)
            if action.event != "NONE" and action.memory_id:
                excluded.add(action.memory_id)
            # Entity mentions attach to the memory the action landed on
            # (conservative disambiguation; see intelligence/entities.py).
            if action.event != "NONE" and action.memory_id and candidate.entities:
                resolved = resolve_mentions(
                    backend=self.backend,
                    llm=self.llm,
                    scope=scope,
                    memory_id=action.memory_id,
                    memory_content=action.content or candidate.content,
                    surfaces=candidate.entities,
                    types=candidate.entity_types,
                )
                self._resolve_relations(
                    candidate.relations, resolved, scope, action.memory_id
                )
        return actions

    def _resolve_relations(
        self,
        relations: list[dict[str, str]],
        resolved: dict[str, Any],
        scope: Scope,
        memory_id: str,
    ) -> None:
        """Turn (subject, predicate, object) surface triples into typed edges
        between the entities they linked to. Both endpoints must have resolved
        to real entities in this same memory, so an edge is always grounded."""
        for rel in relations:
            subj = resolved.get(str(rel.get("subject", "")).strip().lower())
            obj = resolved.get(str(rel.get("object", "")).strip().lower())
            predicate = str(rel.get("predicate", "")).strip().lower()
            if subj is None or obj is None or not predicate or subj.id == obj.id:
                continue
            self.backend.add_relation(
                Relation(
                    subject=subj.id,
                    predicate=predicate,
                    object=obj.id,
                    user_id=scope.user_id,
                    memory_id=memory_id,
                )
            )

    def import_verbatim(
        self, rows: list[dict[str, Any]], *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Bulk verbatim import (dashboard/CLI import, restoring exports).

        Rows are trusted as already-curated facts: no extraction, no
        reconciliation, and embeddings are fetched in batched provider calls,
        so importing N memories costs O(N/batch) HTTP round-trips instead of
        two or three per row. Each row needs "content"; user_id, categories,
        memory_type, and importance are optional (a row's own user_id wins
        over the call-level default)."""
        default_uid = user_id or self.config.default_user_id
        prepared: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            content = str(row.get("content") or "").strip()
            if not content:
                skipped += 1
                continue
            categories = row.get("categories") or []
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]
            memory_type = row.get("memory_type", "semantic")
            if memory_type not in MEMORY_TYPES:
                memory_type = "semantic"
            try:
                importance = float(row.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            prepared.append(
                {
                    "content": content,
                    "user_id": str(row.get("user_id") or default_uid),
                    "agent_id": row.get("agent_id") or None,
                    "run_id": row.get("run_id") or None,
                    "categories": [str(c) for c in categories],
                    "memory_type": memory_type,
                    "importance": importance,
                }
            )
        if not prepared:
            return {"imported": 0, "skipped": skipped, "memory_ids": []}

        episodes = [
            Episode(
                content=p["content"],
                user_id=p["user_id"],
                agent_id=p["agent_id"],
                run_id=p["run_id"],
                metadata={"imported": True},
            )
            for p in prepared
        ]
        self.backend.add_episodes(episodes)

        vectors: list[list[float] | None] = [None] * len(prepared)
        if self.embedder.dimensions:
            chunk = 256  # stay well under provider batch limits
            for start in range(0, len(prepared), chunk):
                batch = prepared[start : start + chunk]
                try:
                    embedded = self.embedder.embed([p["content"] for p in batch])
                except Exception:
                    embedded = []  # import anyway; `memry reindex` can backfill
                for offset, vec in enumerate(embedded):
                    vectors[start + offset] = vec or None

        memory_ids: list[str] = []
        for p, episode, vector in zip(prepared, episodes, vectors):
            memory = Memory(
                content=p["content"],
                memory_type=p["memory_type"],
                user_id=p["user_id"],
                agent_id=p["agent_id"],
                run_id=p["run_id"],
                importance=p["importance"],
                categories=p["categories"],
                source_episode_ids=[episode.id],
            )
            if vector:
                memory.embedding_model = self.embedder.model_id
            stored = self.backend.insert_memory(memory, vector)
            self.backend.add_event(
                MemoryEvent(
                    memory_id=stored.id,
                    event="ADD",
                    new_content=stored.content,
                    reason="imported",
                )
            )
            memory_ids.append(stored.id)
        return {"imported": len(memory_ids), "skipped": skipped, "memory_ids": memory_ids}

    def distill(
        self, memory_id: str, *, owner_prefix: str | None = None
    ) -> AddResult | None:
        """Run extraction on a memory that was stored verbatim (LLM missing or
        failing at save time) and replace it with the distilled facts.

        The original memory is invalidated (kept in the audit history) and
        superseded by the first fact that landed. If extraction finds nothing
        worth keeping, the memory stays as-is with its pending flag cleared.
        Returns None for unknown or already-invalidated memories; raises if
        no LLM is configured or the LLM call fails.
        """
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix) or memory.invalid_at is not None:
            return None
        if not self.llm.available:
            raise ValueError("no LLM configured; distillation needs one")
        scope = Scope(
            user_id=memory.user_id, agent_id=memory.agent_id, run_id=memory.run_id
        )
        candidates = extract_facts(
            self.llm, [{"role": "user", "content": memory.content}]
        )
        episode_ids = memory.source_episode_ids
        if not candidates:
            meta = {
                k: v for k, v in memory.metadata.items() if k != "pending_distillation"
            }
            self.backend.update_memory(memory_id, metadata=meta, touch=False)
            return AddResult(
                episode_ids=episode_ids,
                warnings=["no facts extracted; memory kept verbatim"],
            )
        actions = self._apply_candidates(
            candidates, scope, episode_ids, exclude_ids={memory_id}
        )
        landed = sum(1 for a in actions if a.event != "NONE")
        new_id = next(
            (a.memory_id for a in actions if a.event != "NONE" and a.memory_id), None
        )
        self.backend.invalidate_memory(memory_id, superseded_by=new_id)
        self.backend.add_event(
            MemoryEvent(
                memory_id=memory_id,
                event="SUPERSEDE",
                old_content=memory.content,
                reason=f"distilled into {landed} fact(s)",
            )
        )
        return AddResult(episode_ids=episode_ids, actions=actions)

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        relational: bool = True,
    ) -> list[SearchResult]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        # No query text = browse by tag/date rather than rank by relevance.
        if not (query or "").strip():
            memories = self.get_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id,
                include_invalid=include_invalid, limit=limit,
                categories=categories, since=since, until=until,
            )
            return [SearchResult(memory=m, score=0.0) for m in memories]
        # Over-fetch when we will post-filter or fuse, so a full page survives.
        wide = (since or until) or (relational and not categories)
        fetch = limit if not wide else min(max(limit * 8, 40), 500)
        results = hybrid_search(
            backend=self.backend,
            embedder=self.embedder,
            query=query,
            scope=scope,
            limit=fetch,
            cfg=self.config.retrieval,
            include_invalid=include_invalid,
            categories=categories,
        )
        # Relational fusion: add memories reachable by typed relations from the
        # query's entities (multi-hop answers hybrid alone scores at zero).
        if relational and not categories:
            rel_ids = relational_memory_ids(self.backend, scope, query, hops=2)
            if rel_ids:
                results = self._fuse_relational(results, rel_ids, include_invalid)
        if since or until:
            results = [r for r in results if _within(r.memory.created_at, since, until)]
        return results[:limit]

    def _fuse_relational(
        self,
        hybrid_results: list[SearchResult],
        rel_ids: list[str],
        include_invalid: bool,
    ) -> list[SearchResult]:
        """RRF-fuse hybrid results (ranked by relevance) with relational
        candidates (ranked by graph distance). Hybrid keeps direct answers on
        top; the relational list injects the hop-reachable ones."""
        k = 60
        rescue = 10  # only rescue memories hybrid buried (rank >= this) or missed
        hybrid_rank = {r.memory.id: rank for rank, r in enumerate(hybrid_results)}
        score: dict[str, float] = {}
        for rank, r in enumerate(hybrid_results):
            score[r.memory.id] = 1.0 / (k + rank)
        for rank, mid in enumerate(rel_ids):
            hr = hybrid_rank.get(mid)
            # A memory hybrid already surfaced needs no graph boost; adding it
            # would let a well-ranked neighbor outrank the true direct answer.
            # Only the buried/absent (the multi-hop answers) get rescued.
            if hr is None or hr >= rescue:
                score[mid] = score.get(mid, 0.0) + 1.0 / (k + rank)
        have: dict[str, SearchResult] = {r.memory.id: r for r in hybrid_results}
        for mid in rel_ids:
            if mid not in have:
                memory = self.backend.get_memory(mid)
                if memory is not None and (include_invalid or memory.invalid_at is None):
                    have[mid] = SearchResult(memory=memory, score=0.0)
        ranked = sorted(have.values(), key=lambda r: -score.get(r.memory.id, 0.0))
        for r in ranked:
            r.signals = {**r.signals, "fused": round(score.get(r.memory.id, 0.0), 5)}
        return ranked

    def reconstruct_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        token_budget: int = 1200,
        limit: int = 20,
    ) -> ContextResult:
        results = self.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
        )
        return build_context(results, token_budget=token_budget)

    def get(self, memory_id: str, *, owner_prefix: str | None = None) -> Memory | None:
        memory = self.backend.get_memory(memory_id)
        return memory if _owned(memory, owner_prefix) else None

    def categories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Category histogram over active memories, largest count first."""
        counter: dict[str, int] = {}
        for memory in self.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, limit=1_000_000
        ):
            for raw in memory.categories or []:
                category = str(raw).strip().lower()
                if category:
                    counter[category] = counter.get(category, 0) + 1
        return [
            {"category": c, "count": n}
            for c, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_invalid: bool = False,
        limit: int = 100,
        offset: int = 0,
        categories: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Memory]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if not (since or until):
            return self.backend.list_memories(
                scope, include_invalid=include_invalid, limit=limit, offset=offset,
                categories=categories,
            )
        # Date-windowed browse: the filter is backend-agnostic (applied here), so
        # pull a broad page ordered by the backend, filter, then paginate.
        rows = self.backend.list_memories(
            scope, include_invalid=include_invalid, limit=1_000_000, offset=0,
            categories=categories,
        )
        rows = [m for m in rows if _within(m.created_at, since, until)]
        return rows[offset : offset + limit]

    def history(
        self, memory_id: str, *, owner_prefix: str | None = None
    ) -> list[MemoryEvent]:
        # Admin path is untouched: events outlive their memory row (hard delete
        # keeps the audit trail), so only look the row up when confining.
        if owner_prefix is not None and not _owned(
            self.backend.get_memory(memory_id), owner_prefix
        ):
            return []
        return self.backend.history(memory_id)

    def episodes(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return self.backend.list_episodes(scope, limit=limit)

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        categories: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_prefix: str | None = None,
    ) -> Memory | None:
        old = self.backend.get_memory(memory_id)
        if not _owned(old, owner_prefix):
            return None
        embedding = None
        embedding_model = None
        if content is not None and self.embedder.dimensions:
            try:
                embedding = self.embedder.embed([content])[0]
                embedding_model = self.embedder.model_id
            except Exception:
                embedding = None
        updated = self.backend.update_memory(
            memory_id,
            content=content,
            embedding=embedding,
            embedding_model=embedding_model,
            importance=importance,
            categories=categories,
            metadata=metadata,
        )
        if updated and content is not None and content != old.content:
            self.backend.add_event(
                MemoryEvent(
                    memory_id=memory_id,
                    event="UPDATE",
                    old_content=old.content,
                    new_content=content,
                    reason="manual update",
                    actor="user",
                )
            )
        return updated

    def delete(
        self, memory_id: str, *, hard: bool = False, owner_prefix: str | None = None
    ) -> bool:
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if hard:
            ok = self.backend.delete_memory(memory_id)
        else:
            ok = self.backend.invalidate_memory(memory_id) is not None
        if ok:
            self.backend.add_event(
                MemoryEvent(
                    memory_id=memory_id,
                    event="DELETE",
                    old_content=memory.content,
                    reason="hard delete" if hard else "manual delete (invalidated)",
                    actor="user",
                )
            )
        return ok

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        hard: bool = False,
    ) -> int:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        memories = self.backend.list_memories(scope, limit=1_000_000)
        for memory in memories:
            self.delete(memory.id, hard=hard)
        return len(memories)

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------
    def entities(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_merged: bool = False,
        limit: int = 100,
    ) -> list[Entity]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return self.backend.list_entities(scope, include_merged=include_merged, limit=limit)

    def relations(self, *, user_id: str | None = None, limit: int = 1000) -> list[Relation]:
        return self.backend.list_relations(Scope(user_id=user_id), limit=limit)

    def repair_updated_at(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Reconstruct each memory's updated_at from its audit trail.

        Housekeeping (tagging, relation backfill, re-embedding) used to bump
        updated_at; this recomputes the true value as the time of the last
        content-changing event (ADD/UPDATE/SUPERSEDE), or created_at if there was
        none. Token-free; idempotent. Fixes recency and decay after such a run.
        """
        fixed = 0
        for memory in self.get_all(user_id=user_id, include_invalid=True, limit=1_000_000):
            times = [
                e.created_at for e in self.backend.history(memory.id)
                if e.event in ("ADD", "UPDATE", "SUPERSEDE")
            ]
            true_ts = max([memory.created_at, *times])
            if true_ts != memory.updated_at:
                self.backend.set_memory_timestamp(memory.id, true_ts)
                fixed += 1
        return {"fixed": fixed}

    def backfill_relations(
        self, *, user_id: str | None = None, limit: int = 100_000
    ) -> dict[str, Any]:
        """One-time: extract typed relations from existing memories.

        Only memories with 2+ linked entities are considered (a relation needs
        two), each does one small focused LLM call, and each is marked done so a
        re-run spends no tokens. Cheap and resumable by design.
        """
        summary = {"scanned": 0, "processed": 0, "relations_added": 0, "skipped": 0}
        if not self.llm.available:
            summary["error"] = "no LLM configured"
            return summary
        for memory in self.get_all(user_id=user_id, limit=limit):
            summary["scanned"] += 1
            if memory.metadata.get("relations_backfilled"):
                continue
            entities = self.backend.entities_of_memory(memory.id)
            if len(entities) < 2:
                summary["skipped"] += 1
                self.backend.update_memory(
                    memory.id, metadata={**memory.metadata, "relations_backfilled": True},
                    touch=False,
                )
                continue
            by_norm = {e.normalized or e.name.lower(): e for e in entities}
            try:
                triples = extract_relations(
                    self.llm, memory.content, [e.name for e in entities]
                )
            except Exception:
                continue  # provider hiccup: leave unmarked, retry on next run
            for t in triples:
                subj = by_norm.get(t["subject"].strip().lower())
                obj = by_norm.get(t["object"].strip().lower())
                if subj is None or obj is None or subj.id == obj.id:
                    continue
                self.backend.add_relation(
                    Relation(subject=subj.id, predicate=t["predicate"], object=obj.id,
                             user_id=memory.user_id, memory_id=memory.id)
                )
                summary["relations_added"] += 1
            summary["processed"] += 1
            self.backend.update_memory(
                memory.id, metadata={**memory.metadata, "relations_backfilled": True},
                touch=False,
            )
        return summary

    def backfill_entity_types(
        self, *, user_id: str | None = None, batch: int = 40
    ) -> dict[str, Any]:
        """Classify entities that were linked before typing existed. Batched:
        one LLM call per ``batch`` entities, so a whole namespace is a handful of
        calls. Only untyped entities are touched, so re-runs cost nothing."""
        summary = {"typed": 0}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        untyped = [
            e for e in self.backend.list_entities(Scope(user_id=user_id), limit=1_000_000)
            if not e.entity_type
        ]
        for i in range(0, len(untyped), batch):
            group = untyped[i : i + batch]
            try:
                types = classify_entity_types(self.llm, [e.name for e in group])
            except Exception:
                continue
            for e in group:
                etype = types.get(e.name.lower())
                if etype:
                    self.backend.set_entity_type(e.id, etype)
                    summary["typed"] += 1
        return summary

    def entity(
        self, entity_id: str, *, owner_prefix: str | None = None
    ) -> dict[str, Any] | None:
        """One entity with its mentions and the memories that mention it."""
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return None
        return {
            "entity": entity,
            "mentions": self.backend.entity_mentions(entity_id),
            "memories": self.backend.entity_memories(entity_id, limit=20),
        }

    def merge_proposals(
        self,
        *,
        user_id: str | None = None,
        status: str | None = "proposed",
        limit: int = 100,
    ) -> list[MergeProposal]:
        return self.backend.list_proposals(Scope(user_id=user_id), status=status, limit=limit)

    def confirm_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """User (or system) confirms: entity_b is folded into entity_a."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        if not self.backend.merge_entities(proposal.entity_a, proposal.entity_b):
            return False
        self.backend.set_proposal_status(proposal_id, "confirmed")
        return True

    def reject_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """User says: these are different entities. They stay separate for good."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        self.backend.set_proposal_status(proposal_id, "rejected")
        return True

    def merge_entities(
        self, keep_id: str, merge_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """Direct user-driven merge outside of any proposal."""
        if owner_prefix is not None and not all(
            _owned(self.backend.get_entity(eid), owner_prefix)
            for eid in (keep_id, merge_id)
        ):
            return False
        return self.backend.merge_entities(keep_id, merge_id)

    def resolve_entities(self, *, user_id: str | None = None) -> dict[str, int]:
        """Re-judge open proposals with accumulated evidence; auto-confirm only
        clear, high-confidence matches. Everything ambiguous stays proposed."""
        return resolve_open_proposals(
            backend=self.backend, llm=self.llm, scope=Scope(user_id=user_id)
        )

    # ------------------------------------------------------------------
    # tag abstraction
    # ------------------------------------------------------------------
    def synthetic_tags(self, *, user_id: str | None = None) -> list[SyntheticTag]:
        """The higher-level tags the system invented for this namespace."""
        return self.backend.list_synthetic_tags(Scope(user_id=user_id))

    def abstract_tags(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Ask the LLM for higher-level tags over this namespace's tags, then
        write each onto every memory carrying one of its member tags.

        Returns a summary dict. A no-op (``{"applied": []}``) when no LLM is
        configured or there are too few tags to cluster - callers can run this
        unconditionally. Records each synthetic tag and stamps the run time.
        """
        cfg = self.config.tags
        summary: dict[str, Any] = {"user_id": user_id, "applied": []}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        histogram = self.categories(user_id=user_id)
        if len(histogram) < cfg.min_tags:
            summary["skipped"] = f"only {len(histogram)} tags (< {cfg.min_tags})"
            self._stamp_tag_run(user_id)
            return summary

        existing = [t.tag for t in self.synthetic_tags(user_id=user_id)]
        proposals = propose_synthetic_tags(
            self.llm, histogram,
            existing_synthetic=existing,
            max_new=cfg.max_new_tags,
            min_cluster=cfg.min_cluster_size,
        )
        for proposal in proposals:
            tag = proposal["tag"]
            members = proposal["members"]
            tagged = self._apply_tag_to_members(user_id, tag, members)
            self.backend.record_synthetic_tag(
                SyntheticTag(tag=tag, source_tags=members, user_id=user_id)
            )
            summary["applied"].append(
                {"tag": tag, "source_tags": members, "memories_tagged": tagged}
            )
        self._stamp_tag_run(user_id)
        return summary

    def collections(self, *, user_id: str | None = None) -> list[Collection]:
        return self.backend.list_collections(Scope(user_id=user_id))

    def build_collections(
        self, *, user_id: str | None = None, sample: int = 2000
    ) -> dict[str, Any]:
        """Cluster memory embeddings and summarize the largest clusters into
        titled collections. Clustering is free; only a few clusters are sent to
        the LLM (capped), so a run costs a small, bounded number of tokens.
        Rebuilds from scratch each run (idempotent)."""
        summary: dict[str, Any] = {"collections": 0}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        pairs = self.backend.memory_vectors(Scope(user_id=user_id), limit=sample)
        if len(pairs) < 4:
            summary["skipped"] = "too few embedded memories"
            return summary
        ids = [p[0] for p in pairs]
        vectors = np.stack([p[1] for p in pairs])
        clusters = cluster_vectors(ids, vectors)
        if not clusters:
            summary["skipped"] = "no coherent clusters"
            return summary
        self.backend.clear_collections(Scope(user_id=user_id))
        by_id = {m.id: m for m in self.get_all(user_id=user_id, limit=1_000_000)}
        for member_ids in clusters:
            contents = [by_id[i].content for i in member_ids if i in by_id]
            if not contents:
                continue
            named = summarize_cluster(self.llm, contents)
            if not named:
                continue
            self.backend.record_collection(
                Collection(title=named["title"], summary=named.get("summary", ""),
                           memory_ids=member_ids, user_id=user_id)
            )
            summary["collections"] += 1
        return summary

    def suggest_tag_merges(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """One-shot canonicalization suggestions (variant/synonym merges) for the
        Tag manager. No LLM -> empty list, so callers can call unconditionally."""
        if not self.llm.available:
            return []
        return suggest_canonical_merges(self.llm, self.categories(user_id=user_id))

    def _apply_tag_to_members(
        self, user_id: str | None, tag: str, members: list[str]
    ) -> int:
        """Add ``tag`` to every active memory carrying one of ``members``."""
        tagged = 0
        for memory in self.get_all(
            user_id=user_id, categories=members, limit=1_000_000
        ):
            current = [str(c).strip().lower() for c in memory.categories or []]
            if tag in current:
                continue
            self.backend.update_memory(
                memory.id, categories=[*(memory.categories or []), tag], touch=False
            )
            tagged += 1
        return tagged

    # -- manual tag curation -----------------------------------------------
    def rename_tag(self, tag: str, to: str, *, user_id: str | None = None) -> int:
        """Rename one tag to another across every memory. Returns the count."""
        return self._retag(user_id, {tag.strip().lower()}, to.strip().lower())

    def merge_tags(self, tags: list[str], to: str, *, user_id: str | None = None) -> int:
        """Combine several tags into one across every memory."""
        remove = {t.strip().lower() for t in tags if t.strip()}
        return self._retag(user_id, remove, to.strip().lower())

    def delete_tag(self, tag: str, *, user_id: str | None = None) -> int:
        """Remove a tag from every memory (the memories stay)."""
        return self._retag(user_id, {tag.strip().lower()}, None)

    def _retag(
        self, user_id: str | None, remove: set[str], add: str | None
    ) -> int:
        """Strip ``remove`` tags from matching memories and optionally add
        ``add``, preserving the other tags and their original casing.

        Once a tag is curated by hand its synthetic marker is dropped: the tag
        is now the user's, not the system's guess.
        """
        remove = {r for r in remove if r}
        if not remove:
            return 0
        changed = 0
        for memory in self.get_all(
            user_id=user_id, categories=list(remove), limit=1_000_000
        ):
            cats = list(memory.categories or [])
            kept = [c for c in cats if str(c).strip().lower() not in remove]
            if add and add not in {str(c).strip().lower() for c in kept}:
                kept.append(add)
            if kept != cats:
                self.backend.update_memory(memory.id, categories=kept, touch=False)
                changed += 1
        scope = Scope(user_id=user_id)
        for tag in remove:
            self.backend.delete_synthetic_tag(scope, tag)
        return changed

    def _stamp_tag_run(self, user_id: str | None) -> None:
        self.backend.set_meta(_tag_run_key(user_id), utcnow())

    def last_tag_run(self, user_id: str | None) -> str | None:
        return self.backend.get_meta(_tag_run_key(user_id))

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def decay_sweep(self, threshold: float = 0.1) -> list[str]:
        return decay_sweep(self.backend, self.config.decay, threshold=threshold)

    def effective_importance(self, memory: Memory) -> float:
        return effective_importance(memory, self.config.decay)

    def reindex(self) -> int:
        """Re-embed every memory with the currently configured embedder, then
        rebuild the ANN sidecar (when available)."""
        if not self.embedder.dimensions:
            return 0
        memories = self.backend.all_memories_iter(include_invalid=True)
        count = 0
        batch_size = 64
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            vectors = self.embedder.embed([m.content for m in batch])
            for memory, vector in zip(batch, vectors):
                if vector:
                    self.backend.update_memory(
                        memory.id, embedding=vector, embedding_model=self.embedder.model_id,
                        touch=False
                    )
                    count += 1
        rebuild = getattr(self.backend, "rebuild_ann", None)
        if rebuild is not None:
            rebuild(self.embedder.model_id, self.embedder.dimensions)
        return count

    def stats(self) -> dict[str, Any]:
        data = self.backend.stats()
        data.update(
            {
                "llm": f"{self.llm.name}"
                + (f":{getattr(self.llm, 'model', '')}" if getattr(self.llm, "model", "") else ""),
                "embedder": self.embedder.model_id,
                "generated_at": utcnow(),
            }
        )
        return data

    def reset(self) -> None:
        self.backend.reset()

    def close(self) -> None:
        self.backend.close()
