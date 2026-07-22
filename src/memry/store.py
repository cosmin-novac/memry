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

from typing import Any

from .backends import build_backend
from .backends.base import MemoryBackend
from .config import Config
from .intelligence.context import build_context
from .intelligence.decay import decay_sweep, effective_importance
from .intelligence.entities import resolve_mentions, resolve_open_proposals
from .intelligence.extraction import extract_facts, verbatim_candidates, verify_coverage
from .intelligence.reconcile import reconcile_candidate
from .models import (
    MEMORY_TYPES,
    AddAction,
    AddResult,
    CandidateFact,
    ContextResult,
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MemoryType,
    MergeProposal,
    Scope,
    SearchResult,
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
                resolve_mentions(
                    backend=self.backend,
                    llm=self.llm,
                    scope=scope,
                    memory_id=action.memory_id,
                    memory_content=action.content or candidate.content,
                    surfaces=candidate.entities,
                )
        return actions

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
            self.backend.update_memory(memory_id, metadata=meta)
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
    ) -> list[SearchResult]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return hybrid_search(
            backend=self.backend,
            embedder=self.embedder,
            query=query,
            scope=scope,
            limit=limit,
            cfg=self.config.retrieval,
            include_invalid=include_invalid,
            categories=categories,
        )

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
    ) -> list[Memory]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return self.backend.list_memories(
            scope, include_invalid=include_invalid, limit=limit, offset=offset,
            categories=categories,
        )

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
                        memory.id, embedding=vector, embedding_model=self.embedder.model_id
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
