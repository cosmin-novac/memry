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
import re
from typing import Any

import numpy as np

from .backends.base import MemoryBackend
from .backends.local import LocalBackend
from .config import Config
from .intelligence.clustering import (
    obvious_canonical_merges,
    propose_synthetic_tags,
    semantic_duplicate_tags,
    suggest_canonical_merges,
)
from .intelligence.consolidate import judge_group, representative, similarity_groups
from .intelligence.context import build_context, estimate_tokens
from .intelligence.decay import decay_sweep, effective_importance
from .intelligence.entities import (
    classify_entity_types,
    propose_same_name_duplicates,
    resolve_mentions,
    resolve_open_proposals,
    synthesize_entity_description,
)
from .intelligence.graph_retrieval import detect_query_entities, relational_memory_ids
from .intelligence.extraction import (
    VOCABULARY_LIMIT,
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
    Topic,
    TopicRelation,
    utcnow,
)
from .providers.embeddings import Embedder, build_embedder
from .providers.llm import LLM, build_llm
from .retrieval import hybrid_search


_ENRICHMENT_KEY = "_enrichment"
_ENRICHMENT_BATCH_SIZE = 8
_ENRICHMENT_MAX_BACKOFF_SECONDS = 300


def _normalized_content(text: str) -> str:
    """Casefolded, punctuation-free text, for spotting identical restatements."""
    return " ".join(re.findall(r"[^\W_]+", (text or "").casefold()))


def _owned(record: Any, owner_prefix: str | None) -> bool:
    """Ownership gate for every id-addressed operation.

    ``owner_prefix`` None means operator access with no confinement. A selector
    ending in ``::`` is a tenant prefix; any other value is one exact account
    namespace.

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
    if not user_id:
        return False
    value = str(user_id)
    return (
        value.startswith(owner_prefix)
        if owner_prefix.endswith("::")
        else value == owner_prefix
    )


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
        self.backend = backend or LocalBackend(self.config.db_path, ann=self.config.ann)
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
        if episodes:
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
                candidates = extract_facts(
                    self.llm,
                    messages,
                    vocabulary=self._tag_vocabulary(
                        scope,
                        text="\n".join(
                            str(m.get("content") or "") for m in messages
                        ),
                    ),
                )
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

    def add_deferred(
        self,
        content: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: MemoryType = "episodic",
        importance: float = 0.5,
        categories: list[str] | None = None,
    ) -> AddResult:
        """Durably save raw text for managed background enrichment.

        This path performs no provider calls. The episode and searchable pending
        memory are committed before the caller receives the result; the pending
        metadata is the restart-safe work marker consumed by the server worker.
        """
        text = content.strip()
        if not text:
            return AddResult()
        queued_at = utcnow()
        episode = Episode(
            content=text,
            role="user",
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {},
            created_at=queued_at,
        )
        pending_metadata = dict(metadata or {})
        pending_metadata["pending_distillation"] = True
        pending_metadata[_ENRICHMENT_KEY] = {
            "status": "pending",
            "attempts": 0,
            "queued_at": queued_at,
        }
        memory = Memory(
            content=text,
            memory_type=memory_type,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            importance=importance,
            categories=categories or [],
            metadata=pending_metadata,
            source_episode_ids=[episode.id],
            created_at=queued_at,
            updated_at=queued_at,
        )
        self.backend.add_episodes([episode])
        self.backend.insert_memory(memory)
        self.backend.add_event(
            MemoryEvent(
                memory_id=memory.id,
                event="ADD",
                new_content=text,
                reason="durably queued for background enrichment",
            )
        )
        return AddResult(
            episode_ids=[episode.id],
            actions=[
                AddAction(
                    event="ADD",
                    memory_id=memory.id,
                    content=memory.content,
                    reason="pending background enrichment",
                )
            ],
        )

    @staticmethod
    def _pending_verbatim(messages: list[dict[str, str]]) -> list[CandidateFact]:
        """Verbatim candidates flagged for later distillation."""
        candidates = verbatim_candidates(messages)
        for candidate in candidates:
            candidate.metadata = {"pending_distillation": True}
        return candidates

    def _canonicalize_obvious_topics(
        self, candidates: list[CandidateFact], scope: Scope
    ) -> None:
        incoming = {
            str(category).strip().casefold()
            for candidate in candidates
            for category in candidate.categories
            if str(category).strip()
        }
        if not incoming:
            return
        existing = {
            topic.normalized
            for topic in self.backend.list_topics(scope, limit=100_000)
        }
        groups = obvious_canonical_merges(
            [{"category": topic} for topic in existing | incoming]
        )
        replacements: dict[str, str] = {}
        for group in groups:
            canonical = group["canonical"]
            variants = set(group["variants"])
            replacements.update({variant: canonical for variant in variants})
            stored_variants = (variants - {canonical}) & existing
            if stored_variants:
                self.backend.retag_topics(scope, stored_variants, canonical)
        for candidate in candidates:
            rewritten: list[str] = []
            seen: set[str] = set()
            for raw in candidate.categories:
                normalized = str(raw).strip().casefold()
                canonical = replacements.get(normalized, normalized)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    rewritten.append(canonical)
            candidate.categories = rewritten

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
        self._canonicalize_obvious_topics(candidates, scope)
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
                prepare_update=lambda memory_id, final_content: (
                    self._reanalyze_edited_entities(memory_id, final_content, scope)
                ),
            )
            actions.append(action)
            if action.event != "NONE" and action.memory_id:
                excluded.add(action.memory_id)
            # Entity mentions attach to the memory the action landed on
            # (conservative disambiguation; see intelligence/entities.py).
            if action.event not in ("NONE", "UPDATE") and action.memory_id and candidate.entities:
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

    def _reanalyze_edited_entities(
        self, memory_id: str, content: str, scope: Scope
    ) -> dict[str, Any]:
        """Return the complete entity fields for edited memory text.

        With an LLM, extraction and identity resolution finish before the
        caller replaces the stored text and mentions. Without one, Memry can
        still retain or remove existing links by matching their known aliases;
        zero-key mode cannot discover a brand-new entity name.
        """
        if not self.llm.available:
            surfaces: list[str] = []
            mentions: list[EntityMention] = []
            for entity in self.backend.entities_of_memory(memory_id):
                aliases = sorted(
                    self.backend.entity_aliases(entity.id), key=len, reverse=True
                )
                surface = next(
                    (
                        alias for alias in aliases
                        if alias and re.search(
                            rf"(?<!\w){re.escape(alias)}(?!\w)",
                            content,
                            flags=re.IGNORECASE,
                        )
                    ),
                    None,
                )
                if surface:
                    surfaces.append(surface)
                    mentions.append(
                        EntityMention(
                            entity_id=entity.id,
                            memory_id=memory_id,
                            surface=surface,
                        )
                    )
            return {"entities": surfaces, "mentions": mentions}
        try:
            candidates = extract_facts(
                self.llm, [{"role": "user", "content": content}]
            )
            surfaces = []
            types: dict[str, str] = {}
            seen: set[str] = set()
            for candidate in candidates:
                types.update(candidate.entity_types)
                for surface in candidate.entities:
                    normalized = surface.strip().lower()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        surfaces.append(surface.strip())
            resolved = resolve_mentions(
                backend=self.backend,
                llm=self.llm,
                scope=scope,
                memory_id=memory_id,
                memory_content=content,
                surfaces=surfaces,
                types=types,
                attach=False,
            )
        except Exception as exc:
            raise ValueError(
                f"memory text was not changed because entity re-analysis failed: {exc}"
            ) from exc
        mentions = [
            EntityMention(
                entity_id=resolved[surface.lower()].id,
                memory_id=memory_id,
                surface=surface,
            )
            for surface in surfaces
            if surface.lower() in resolved
        ]
        return {"entities": surfaces, "mentions": mentions}

    def _has_near_duplicate(
        self,
        embedding: list[float],
        scope: Scope,
        threshold: float,
    ) -> bool:
        matches = self.backend.vector_search(
            embedding,
            self.embedder.model_id,
            scope,
            limit=1,
        )
        return bool(matches and matches[0][1] >= threshold)

    def import_verbatim(
        self,
        rows: list[dict[str, Any]],
        *,
        user_id: str | None = None,
        dedup: bool = True,
        dedup_threshold: float = 0.97,
    ) -> dict[str, Any]:
        """Bulk verbatim import without extraction or reconciliation.

        Embeddings are fetched in batches. Duplicate rows in the same import and
        near-identical memories already in the target user scope are skipped by
        default without creating orphan episodes.
        """
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
            return {
                "imported": 0,
                "skipped": skipped,
                "deduplicated": 0,
                "memory_ids": [],
            }

        vectors: list[list[float] | None] = [None] * len(prepared)
        if self.embedder.dimensions:
            chunk = 256  # stay well under provider batch limits
            for start in range(0, len(prepared), chunk):
                batch = prepared[start : start + chunk]
                try:
                    embedded = self.embedder.embed([p["content"] for p in batch])
                except Exception:
                    embedded = []  # import anyway; `memry reindex` can backfill
                for offset, vector in enumerate(embedded):
                    vectors[start + offset] = vector or None

        accepted: list[tuple[dict[str, Any], list[float] | None]] = []
        deduplicated = 0
        seen_content: set[tuple[str, str]] = set()
        for row, vector in zip(prepared, vectors):
            if dedup:
                key = (row["user_id"], row["content"].strip().lower())
                if key in seen_content:
                    deduplicated += 1
                    continue
                seen_content.add(key)
                if vector and self._has_near_duplicate(
                    vector, Scope(user_id=row["user_id"]), dedup_threshold
                ):
                    deduplicated += 1
                    continue
            accepted.append((row, vector))

        episodes = [
            Episode(
                content=row["content"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                run_id=row["run_id"],
                metadata={"imported": True},
            )
            for row, _ in accepted
        ]
        if episodes:
            self.backend.add_episodes(episodes)

        memory_ids: list[str] = []
        for (row, vector), episode in zip(accepted, episodes):
            memory = Memory(
                content=row["content"],
                memory_type=row["memory_type"],
                user_id=row["user_id"],
                agent_id=row["agent_id"],
                run_id=row["run_id"],
                importance=row["importance"],
                categories=row["categories"],
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
        return {
            "imported": len(memory_ids),
            "skipped": skipped,
            "deduplicated": deduplicated,
            "memory_ids": memory_ids,
        }
    def export_backup(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Export exact knowledge records for this scope as one versioned bundle."""
        return self.backend.export_backup(
            Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        )

    def import_backup(
        self, backup: dict[str, Any], *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        """Restore a Memry backup exactly and transactionally."""
        return self.backend.import_backup(backup, owner_prefix=owner_prefix)

    @staticmethod
    def _clear_enrichment_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in metadata.items()
            if key not in ("pending_distillation", _ENRICHMENT_KEY)
        }

    def process_pending_enrichments(
        self, limit: int = _ENRICHMENT_BATCH_SIZE
    ) -> dict[str, Any]:
        """Process one bounded batch of durable pending memories.

        Each item keeps independent provenance and retry state. Work is batched
        at the scheduler/database level, but different user payloads are never
        merged into one prompt because that would mix provenance and failures.
        """
        summary: dict[str, Any] = {
            "claimed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
        }
        if not self.llm.available:
            summary["blocked"] = "no LLM configured"
            return summary
        now = datetime.now(timezone.utc)
        pending = self.backend.list_pending_memories(
            max(1, limit), due_before=now.isoformat(timespec="seconds")
        )
        for memory in pending:
            current = self.backend.get_memory(memory.id)
            if (
                current is None
                or current.invalid_at is not None
                or not current.metadata.get("pending_distillation")
            ):
                continue
            job = dict(current.metadata.get(_ENRICHMENT_KEY) or {})
            attempts = int(job.get("attempts") or 0) + 1
            job.update(
                {
                    "status": "processing",
                    "attempts": attempts,
                    "last_started_at": utcnow(),
                }
            )
            job.pop("next_attempt_at", None)
            processing_metadata = dict(current.metadata)
            processing_metadata[_ENRICHMENT_KEY] = job
            self.backend.update_memory(
                current.id, metadata=processing_metadata, touch=False
            )
            summary["claimed"] += 1
            try:
                result = self.distill(current.id)
                if result is None:
                    continue
                summary["succeeded"] += 1
            except Exception as exc:
                latest = self.backend.get_memory(current.id)
                if (
                    latest is not None
                    and latest.invalid_at is None
                    and latest.metadata.get("pending_distillation")
                ):
                    retry_job = dict(latest.metadata.get(_ENRICHMENT_KEY) or job)
                    delay = min(
                        2 ** min(attempts, 8), _ENRICHMENT_MAX_BACKOFF_SECONDS
                    )
                    retry_job.update(
                        {
                            "status": "retry",
                            "attempts": attempts,
                            "last_error": str(exc)[:500],
                            "next_attempt_at": (
                                datetime.now(timezone.utc) + timedelta(seconds=delay)
                            ).isoformat(timespec="seconds"),
                        }
                    )
                    retry_metadata = dict(latest.metadata)
                    retry_metadata[_ENRICHMENT_KEY] = retry_job
                    self.backend.update_memory(
                        latest.id, metadata=retry_metadata, touch=False
                    )
                summary["failed"] += 1
                summary["errors"].append(
                    {"memory_id": current.id, "error": str(exc)[:500]}
                )
        return summary

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
            meta = self._clear_enrichment_metadata(memory.metadata)
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
        invalidated = self.backend.invalidate_memory(memory_id, superseded_by=new_id)
        if invalidated is not None:
            self.backend.update_memory(
                memory_id,
                metadata=self._clear_enrichment_metadata(invalidated.metadata),
                touch=False,
            )
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
        entity_id: str | list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        relational: bool = True,
    ) -> list[SearchResult]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if entity_id:
            entity_id = self._resolve_entity_filter(entity_id)
            if not entity_id:
                return []
        # No query text = browse by tag/date rather than rank by relevance.
        if not (query or "").strip():
            memories = self.get_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id,
                include_invalid=include_invalid, limit=limit,
                categories=categories, entity_id=entity_id, since=since, until=until,
            )
            return [SearchResult(memory=m, score=0.0) for m in memories]
        # Over-fetch when we will post-filter or fuse, so a full page survives.
        wide = (since or until) or (relational and not categories and not entity_id)
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
            entity_id=entity_id,
        )
        # Relational fusion: add memories reachable by typed relations from the
        # query's entities (multi-hop answers hybrid alone scores at zero).
        if relational and not categories and not entity_id:
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
        candidates (ranked by graph distance).

        The graph boost is what makes multi-hop work: the answer to "what tool
        does Ada use?" is usually present but buried, and lifting it is the
        whole point. The same lift is also the cost, because on an ordinary
        query a buried graph neighbour can leapfrog the correct answer:
        ``1/(k+relrank)`` added to ``1/(k+hybridrank)`` can exceed the score of
        hybrid's own top hit.

        Measured on a 456-memory store with a dense entity graph, the two
        effects share one mechanism and no weighting or injection cap separates
        them. Reserving the strongest hybrid results does: graph distance
        competes for the rest of the page but can never evict a top direct
        answer. That keeps multi-hop hit@10 at 0.917 (against 0.417 for hybrid
        alone) while ordinary recall@10 returns to 0.649 from 0.474.
        """
        k = 60
        rescue = 10  # only rescue memories hybrid buried (rank >= this) or missed
        protect = max(0, self.config.retrieval.relational_protect_top)
        pinned = [r for r in hybrid_results[:protect]]
        pinned_ids = {r.memory.id for r in pinned}
        hybrid_rank = {r.memory.id: rank for rank, r in enumerate(hybrid_results)}
        score: dict[str, float] = {}
        for rank, r in enumerate(hybrid_results):
            score[r.memory.id] = 1.0 / (k + rank)
        for rank, mid in enumerate(rel_ids):
            hr = hybrid_rank.get(mid)
            # A memory hybrid already ranked highly needs no graph boost; adding
            # it would let a well-ranked neighbor outrank the true direct answer.
            # Only the buried/absent (the multi-hop answers) get rescued.
            if hr is None or hr >= rescue:
                score[mid] = score.get(mid, 0.0) + 1.0 / (k + rank)
        have: dict[str, SearchResult] = {r.memory.id: r for r in hybrid_results}
        for mid in rel_ids:
            if mid not in have:
                memory = self.backend.get_memory(mid)
                if memory is not None and (include_invalid or memory.invalid_at is None):
                    have[mid] = SearchResult(memory=memory, score=0.0)
        fused = sorted(have.values(), key=lambda r: -score.get(r.memory.id, 0.0))
        ranked = pinned + [r for r in fused if r.memory.id not in pinned_ids]
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
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        results = self.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
        )
        entity_text, entity_memory_ids = self._entity_context(
            scope, query, token_budget=min(300, max(80, token_budget // 4))
        )
        remaining = max(0, token_budget - estimate_tokens(entity_text))
        memory_context = build_context(results, token_budget=remaining)
        parts = [part for part in (entity_text, memory_context.text) if part]
        combined = "\n\n".join(parts)
        memory_ids = list(dict.fromkeys([*entity_memory_ids, *memory_context.memory_ids]))
        return ContextResult(
            text=combined,
            memory_ids=memory_ids,
            token_estimate=estimate_tokens(combined) if combined else 0,
        )

    def _entity_context(
        self, scope: Scope, query: str, *, token_budget: int
    ) -> tuple[str, list[str]]:
        entity_ids = detect_query_entities(self.backend, scope, query)[:3]
        if not entity_ids:
            return "", []
        header = "## Known entities (memry)\n"
        used = estimate_tokens(header)
        lines: list[str] = []
        memory_ids: list[str] = []
        for entity_id in entity_ids:
            entity = self._refresh_entity_description(entity_id)
            if entity is None or not entity.description:
                continue
            label = entity.name
            if entity.entity_type:
                label += f" ({entity.entity_type})"
            line = f"- {label}: {entity.description}"
            cost = estimate_tokens(line) + 1
            if used + cost > token_budget:
                continue
            lines.append(line)
            used += cost
            memory_ids.extend(
                memory.id for memory in self.backend.entity_memories(entity.id, limit=20)
            )
        if not lines:
            return "", []
        return header + "\n".join(lines), list(dict.fromkeys(memory_ids))

    def _resolve_entity_filter(
        self, entity_id: str | list[str]
    ) -> str | list[str] | None:
        """Follow merge history for one entity filter, or several.

        A merged entity keeps its old id as a redirect, so a filter saved before
        a merge must still land on the surviving record.
        """
        if isinstance(entity_id, str):
            return self.backend.resolve_entity_id(entity_id)
        resolved = [self.backend.resolve_entity_id(e) for e in entity_id if e]
        return [e for e in resolved if e] or None

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
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        indexed = self.backend.topic_counts(scope)
        if indexed is not None:
            return indexed
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

    def _tag_vocabulary(
        self, scope: Scope, text: str = "", limit: int = VOCABULARY_LIMIT
    ) -> list[str]:
        """Direct tags offered back to extraction, so tagging stays convergent.

        Parents are deliberately excluded: offering ``health`` back would invite
        extraction to tag straight at the level that retrieval does worst with.

        Selection is by relevance first, then by frequency. Sending only the
        most-used tags works until a store passes the budget, at which point the
        long tail stops being offered - and an unoffered tag is precisely the one
        that gets a near-synonym coined for it next time its subject comes up.
        A conversation about liver results must see ``liver lab results`` even
        when it is the 300th most common tag.
        """
        try:
            counts = self.backend.direct_topic_counts(scope)
        except Exception:
            return []
        if not counts:
            return []
        names = [str(row["category"]) for row in counts if row.get("category")]
        if len(names) <= limit or not text.strip() or not self.embedder.dimensions:
            return names[:limit]

        # Half the budget goes to what this conversation is actually about, the
        # rest stays frequency-ranked so common tags are always on offer.
        relevant_budget = limit // 2
        try:
            vectors = self.embedder.embed([text[:4000], *names])
        except Exception:
            return names[:limit]
        query = np.asarray(vectors[0], dtype=float)
        matrix = np.asarray(vectors[1:], dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query) or 1.0
        similarity = (matrix @ query) / (np.where(norms == 0, 1.0, norms) * query_norm)
        nearest = [names[i] for i in np.argsort(-similarity)[:relevant_budget]]
        chosen = list(dict.fromkeys(nearest))
        for name in names:  # top up with the most-used, skipping duplicates
            if len(chosen) >= limit:
                break
            if name not in chosen:
                chosen.append(name)
        return chosen

    def direct_categories(
        self, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Histogram over tags attached straight to memories, no parent rollup.

        ``categories()`` rolls descendants up into their parents, which is what
        the Knowledge UI and parent filtering want. Abstraction wants the
        opposite: if a system-generated parent appears in its own input, the
        next run happily clusters ``liver health`` and ``weekly gym`` into
        ``health`` and the useful level decays one run at a time.
        """
        direct = self.backend.direct_topic_counts(Scope(user_id=user_id))
        return direct if direct is not None else self.categories(user_id=user_id)

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
        entity_id: str | list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Memory]:
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        if entity_id:
            entity_id = self._resolve_entity_filter(entity_id)
            if not entity_id:
                return []
        if not (since or until):
            return self.backend.list_memories(
                scope, include_invalid=include_invalid, limit=limit, offset=offset,
                categories=categories, entity_id=entity_id,
            )
        # Date-windowed browse: the filter is backend-agnostic (applied here), so
        # pull a broad page ordered by the backend, filter, then paginate.
        rows = self.backend.list_memories(
            scope, include_invalid=include_invalid, limit=1_000_000, offset=0,
            categories=categories, entity_id=entity_id,
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
        entity_update: dict[str, Any] = {}
        if content is not None and content != old.content:
            entity_update = self._reanalyze_edited_entities(
                memory_id, content, old.scope()
            )
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
            **entity_update,
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

    def forgotten(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Memories that were removed, with why and by whom.

        Removed, not replaced: a memory that was superseded (reconciled away,
        consolidated, distilled) has ``superseded_by`` pointing at whatever took
        its place and is part of that memory's history, not something the user
        threw out. Only records with nothing standing in for them belong here.
        """
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        out: list[dict[str, Any]] = []
        for memory in self.backend.list_memories(
            scope, include_invalid=True, limit=1_000_000
        ):
            if memory.invalid_at is None or memory.superseded_by:
                continue
            removal = next(
                (
                    event
                    for event in reversed(self.backend.history(memory.id))
                    if event.event == "DELETE"
                ),
                None,
            )
            out.append({
                "memory": memory,
                "forgotten_at": memory.invalid_at,
                "actor": removal.actor if removal else "system",
                "reason": removal.reason if removal else None,
            })
            if len(out) >= limit:
                break
        out.sort(key=lambda row: row["forgotten_at"] or "", reverse=True)
        return out

    def purge(self, memory_id: str, *, owner_prefix: str | None = None) -> bool:
        """Delete a forgotten memory for good. Refuses anything still active.

        Permanent deletion is the one operation with no audit trail left to
        inspect afterwards, so it is deliberately a second step: a memory has to
        have been forgotten first. That makes an accidental irreversible delete
        take two decisions instead of one bad click.
        """
        memory = self.backend.get_memory(memory_id)
        if not _owned(memory, owner_prefix):
            return False
        if memory.invalid_at is None:
            raise ValueError("only a forgotten memory can be permanently deleted")
        return self.backend.delete_memory(memory_id)

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

    def _refresh_entity_description(
        self, entity_id: str, *, force: bool = False
    ) -> Entity | None:
        entity = self.backend.get_entity(entity_id)
        if entity is None or not entity.is_active:
            return None
        evidence_updated_at = self.backend.entity_evidence_updated_at(entity_id)
        if (
            not force
            and entity.description is not None
            and entity.description_updated_at is not None
            and (
                evidence_updated_at is None
                or entity.description_updated_at >= evidence_updated_at
            )
        ):
            return entity
        memories = self.backend.entity_memories(entity_id, limit=50)
        description = synthesize_entity_description(
            self.llm,
            entity,
            [memory.content for memory in memories],
            self.backend.entity_aliases(entity_id),
        )
        generated_at = utcnow()
        stored = self.backend.set_entity_description(
            entity_id, description, generated_at
        )
        if stored is not None:
            return stored
        entity.description = description
        entity.description_updated_at = generated_at
        return entity

    def entity(
        self,
        entity_id: str,
        *,
        owner_prefix: str | None = None,
        refresh_description: bool = True,
    ) -> dict[str, Any] | None:
        """One entity hub with aliases and active supporting memories."""
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return None
        if refresh_description:
            entity = self._refresh_entity_description(entity_id)
            if entity is None:
                return None
        # Relations belong to the entity being looked at, not to a list of every
        # edge in the store: an edge only means something next to the thing it
        # connects. These are also what relational retrieval traverses, so
        # seeing them here is seeing why a search reached what it reached.
        relations = self.backend.relations_of([entity_id])
        endpoints = {r.subject for r in relations} | {r.object for r in relations}
        names = {
            other.id: other.name
            for other in (self.backend.get_entity(e) for e in endpoints)
            if other is not None
        }
        return {
            "entity": entity,
            "aliases": self.backend.entity_aliases(entity_id),
            "mentions": self.backend.entity_mentions(entity_id),
            "memories": self.backend.entity_memories(entity_id, limit=20),
            "relations": relations,
            "relation_names": names,
        }

    def add_entity_alias(
        self, entity_id: str, alias: str, *, owner_prefix: str | None = None
    ) -> Entity | None:
        entity = self.backend.get_entity(entity_id)
        if not _owned(entity, owner_prefix):
            return None
        return self.backend.add_entity_alias(entity_id, alias)

    def merge_proposals(
        self,
        *,
        user_id: str | None = None,
        status: str | None = "proposed",
        limit: int = 100,
    ) -> list[MergeProposal]:
        proposals = self.backend.list_proposals(
            Scope(user_id=user_id), status=status, limit=limit
        )
        if status != "proposed":
            return proposals
        active: list[MergeProposal] = []
        for proposal in proposals:
            entity_a = self.backend.resolve_entity_id(proposal.entity_a)
            entity_b = self.backend.resolve_entity_id(proposal.entity_b)
            if entity_a is None or entity_b is None:
                self.backend.set_proposal_status(proposal.id, "rejected")
                continue
            if entity_a == entity_b:
                self.backend.set_proposal_status(proposal.id, "confirmed")
                continue
            active.append(
                proposal.model_copy(update={"entity_a": entity_a, "entity_b": entity_b})
            )
        return active

    def confirm_merge(
        self, proposal_id: str, *, owner_prefix: str | None = None
    ) -> bool:
        """Confirm a proposal after resolving both endpoints through merge history."""
        proposal = self.backend.get_proposal(proposal_id)
        if not _owned(proposal, owner_prefix) or proposal.status != "proposed":
            return False
        entity_a = self.backend.resolve_entity_id(proposal.entity_a)
        entity_b = self.backend.resolve_entity_id(proposal.entity_b)
        if entity_a is None or entity_b is None:
            return False
        if owner_prefix is not None and not all(
            _owned(self.backend.get_entity(entity_id), owner_prefix)
            for entity_id in (entity_a, entity_b)
        ):
            return False
        if entity_a != entity_b and not self.backend.merge_entities(entity_a, entity_b):
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
        """Idempotent direct merge outside of a proposal."""
        keep_root = self.backend.resolve_entity_id(keep_id)
        merge_root = self.backend.resolve_entity_id(merge_id)
        if keep_root is None or merge_root is None:
            return False
        if owner_prefix is not None and not all(
            _owned(self.backend.get_entity(entity_id), owner_prefix)
            for entity_id in (keep_root, merge_root)
        ):
            return False
        return self.backend.merge_entities(keep_root, merge_root)

    def resolve_entities(self, *, user_id: str | None = None) -> dict[str, int]:
        """Re-judge open proposals with accumulated evidence; auto-confirm only
        clear, high-confidence matches. Everything ambiguous stays proposed.

        Then drop entities nothing references. Extraction inevitably produces
        some records that never attach to anything, and without this they
        accumulate forever: a real store reached 206 such rows out of 519.
        """
        scope = Scope(user_id=user_id)
        # Surface same-name duplicates first: proposals are otherwise only made
        # at write time, so anything already duplicated has nothing scheduled to
        # look at it again and would sit there for good.
        proposed = propose_same_name_duplicates(backend=self.backend, scope=scope)
        outcome = resolve_open_proposals(
            backend=self.backend, llm=self.llm, scope=scope
        )
        outcome["proposed"] = proposed
        outcome["purged"] = self.backend.purge_orphan_entities(scope)
        return outcome

    # ------------------------------------------------------------------
    # tag abstraction
    # ------------------------------------------------------------------
    def synthetic_tags(self, *, user_id: str | None = None) -> list[SyntheticTag]:
        """The higher-level tags the system invented for this namespace."""
        return self.backend.list_synthetic_tags(Scope(user_id=user_id))

    def abstract_tags(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Create higher-level topic nodes and hierarchy edges.

        Parent labels are not copied onto memories. Query-time hierarchy
        expansion makes a parent filter include memories linked to its children.
        """
        cfg = self.config.tags
        summary: dict[str, Any] = {"user_id": user_id, "applied": []}
        if not self.llm.available:
            summary["skipped"] = "no LLM configured"
            return summary
        histogram = self.direct_categories(user_id=user_id)
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
        all_topics = self.backend.list_topics(Scope(user_id=user_id), limit=100_000)
        for proposal in proposals:
            tag = proposal["tag"].strip().lower()
            members = [
                member.strip().lower()
                for member in proposal["members"]
                if member.strip()
            ]
            parent = self.backend.upsert_topic(
                Topic(name=tag, normalized=tag, user_id=user_id, provenance="synthetic")
            )
            relations_added = 0
            for member in members:
                children = [topic for topic in all_topics if topic.normalized == member]
                if not children:
                    children = [
                        self.backend.upsert_topic(
                            Topic(
                                name=member,
                                normalized=member,
                                user_id=user_id,
                                provenance="memory",
                            )
                        )
                    ]
                    all_topics.extend(children)
                for child in children:
                    if child.id == parent.id:
                        continue
                    self.backend.add_topic_relation(
                        TopicRelation(
                            broader_topic_id=parent.id,
                            narrower_topic_id=child.id,
                            user_id=user_id,
                            provenance="synthetic",
                        )
                    )
                    relations_added += 1
            self.backend.record_synthetic_tag(
                SyntheticTag(tag=tag, source_tags=members, user_id=user_id)
            )
            all_topics.append(parent)
            summary["applied"].append(
                {"tag": tag, "source_tags": members, "relations_added": relations_added}
            )
        self._stamp_tag_run(user_id)
        return summary

    def merge_obvious_topics(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Automatically collapse deterministic formatting/plural duplicates."""
        scope = Scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        categories = self.categories(
            user_id=user_id, agent_id=agent_id, run_id=run_id
        )
        groups = obvious_canonical_merges(categories)
        changed = 0
        for group in groups:
            remove = set(group["variants"]) - {group["canonical"]}
            result = self.backend.retag_topics(scope, remove, group["canonical"])
            changed += result or 0
        return {"groups_merged": len(groups), "memories_changed": changed}

    def consolidate_memories(
        self,
        *,
        user_id: str | None = None,
        threshold: float = 0.90,
        max_groups: int = 25,
        apply: bool = True,
        only: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        """Merge memories that record the same fact more than once.

        ``apply=False`` returns exactly what would happen without touching
        anything, which is what the dashboard shows before the user confirms.
        ``only`` restricts the run to the listed groups, so a user can accept
        some proposals from a preview and leave the rest alone.

        The merged text becomes a NEW memory carrying the union of the group's
        tags, entities and provenance, and every original is invalidated with
        ``superseded_by`` pointing at it. Rewriting one of the originals in
        place would have made an arbitrary member masquerade as the merge, with
        its own creation date and history; a distinct record says plainly that
        this text came from consolidating several. Nothing is destroyed, so the
        audit trail and time-travel still resolve.
        """
        scope = Scope(user_id=user_id)
        wanted = {frozenset(group) for group in only} if only else None
        vectors = self.backend.memory_vectors(scope, limit=1_000_000)
        summary: dict[str, Any] = {
            "scanned": len(vectors), "groups": [], "merged": 0, "superseded": 0,
        }
        groups = similarity_groups(vectors, threshold=threshold)
        if not groups:
            return summary

        # densest first: the most obviously redundant families are worth the
        # LLM budget before a long tail of borderline pairs
        groups.sort(key=len, reverse=True)
        for member_ids in groups[:max_groups]:
            if wanted is not None and frozenset(member_ids) not in wanted:
                continue
            memories = [m for m in (self.backend.get_memory(i) for i in member_ids)
                        if m is not None and m.invalid_at is None]
            if len(memories) < 2:
                continue
            normalized = {_normalized_content(m.content) for m in memories}
            if len(normalized) == 1:
                verdict = {"same_fact": True,
                           "content": representative(memories).content,
                           "reason": "identical text"}
            elif self.llm.available:
                verdict = judge_group(self.llm, memories)
            else:
                continue  # never merge on similarity alone

            entry = {
                "memory_ids": [m.id for m in memories],
                "contents": [m.content for m in memories],
                "same_fact": bool(verdict["same_fact"]),
                "reason": verdict["reason"],
                "merged_content": verdict["content"],
            }
            summary["groups"].append(entry)
            if not verdict["same_fact"] or not apply:
                continue

            content = verdict["content"]
            oldest = min(memories, key=lambda m: m.created_at)
            merged = Memory(
                content=content,
                memory_type=oldest.memory_type,
                user_id=user_id,
                importance=max(m.importance for m in memories),
                categories=list(dict.fromkeys(
                    [c for m in memories for c in (m.categories or [])])),
                entities=list(dict.fromkeys(
                    [e for m in memories for e in (m.entities or [])])),
                # keep the earliest creation date: the fact is as old as the
                # first time it was recorded, not as old as the merge
                created_at=oldest.created_at,
                source_episode_ids=list(dict.fromkeys(
                    [e for m in memories for e in (m.source_episode_ids or [])])),
                metadata={"consolidated_from": [m.id for m in memories]},
            )
            embedding = (
                self.embedder.embed([content])[0] if self.embedder.dimensions else None
            )
            stored = self.backend.insert_memory(merged, embedding=embedding)
            self.backend.add_event(MemoryEvent(
                memory_id=stored.id, event="ADD", new_content=content,
                reason=f"consolidated {len(memories)} duplicate memories",
            ))
            # every original is forgotten, including the one it reads most like
            for memory in memories:
                self.backend.invalidate_memory(memory.id, superseded_by=stored.id)
                self.backend.add_event(MemoryEvent(
                    memory_id=memory.id, event="SUPERSEDE",
                    old_content=memory.content, new_content=content,
                    reason=f"consolidated into {stored.id}",
                ))
                summary["superseded"] += 1
            entry["survivor"] = stored.id
            summary["merged"] += 1
        return summary

    def semantic_tag_duplicates(
        self, *, user_id: str | None = None, threshold: float = 0.93
    ) -> list[dict[str, Any]]:
        """Tags that have split one subject, judged by the stored vectors.

        Needs no LLM and no new storage: a tag's centroid is the mean of its
        members' existing embeddings. This is the drift that matters over a long
        life - a fragmented tag silently caps recall, because filtering to it
        excludes memories the question needed.
        """
        scope = Scope(user_id=user_id)
        links = self.backend.topic_memory_ids(scope)
        if not links:
            return []
        vectors = dict(self.backend.memory_vectors(scope, limit=1_000_000))
        if not vectors:
            return []

        members: dict[str, list[str]] = {}
        for tag, memory_id in links:
            if memory_id in vectors:
                members.setdefault(tag, []).append(memory_id)
        counts = {tag: len(ids) for tag, ids in members.items()}

        by_memory: dict[str, list[str]] = {}
        for tag, memory_id in links:
            by_memory.setdefault(memory_id, []).append(tag)
        cooccurrence: dict[tuple[str, str], int] = {}
        for tags in by_memory.values():
            unique = sorted(set(tags))
            for i, a in enumerate(unique):
                for b in unique[i + 1:]:
                    cooccurrence[(a, b)] = cooccurrence.get((a, b), 0) + 1

        centroids = {
            tag: np.mean([vectors[m] for m in ids], axis=0)
            for tag, ids in members.items()
            if len(ids) >= 2
        }
        # Embedding the tag names lets the detector require that the LABELS mean
        # the same thing, not just that the member memories sit close together.
        # On a real store, centroid similarity alone either found nothing or
        # proposed wrong merges; the conjunction found exactly the true split.
        # Only a semantic embedder can make that judgement - the hash embedder
        # scores "tech"/"technical" at 0.14, so with it the conjunction would
        # simply disable the detector. Zero-key mode keeps centroids alone.
        labels: dict[str, Any] | None = None
        if self.embedder.dimensions and centroids and self.embedder.name != "hash":
            names = sorted(centroids)
            try:
                labels = dict(zip(names, self.embedder.embed(names)))
            except Exception:
                labels = None
        return semantic_duplicate_tags(
            centroids, counts, cooccurrence, labels=labels, threshold=threshold
        )

    def tag_health(self, *, user_id: str | None = None) -> dict[str, Any]:
        """Cheap, deterministic signals on how well the tag vocabulary is holding.

        Fragmentation is the failure mode that costs recall silently: filtering
        to a tag that has split its subject drops the memories the question
        needed, before ranking ever runs. Nothing surfaces that today unless
        somebody presses "suggest merges", so a store can drift for months.

        No LLM, no writes. Everything here comes from counts already indexed and
        vectors already stored.
        """
        scope = Scope(user_id=user_id)
        counts = self.backend.direct_topic_counts(scope) or []
        total = len(self.get_all(user_id=user_id, limit=1_000_000))
        tagged = len({mid for _, mid in (self.backend.topic_memory_ids(scope) or [])})
        singles = sum(1 for row in counts if row.get("count") == 1)
        splits = self.semantic_tag_duplicates(user_id=user_id)
        return {
            "memories": total,
            "untagged": max(0, total - tagged),
            "tags": len(counts),
            "single_use_tags": singles,
            "single_use_share": round(singles / len(counts), 3) if counts else 0.0,
            "suspected_splits": len(splits),
            "splits": splits[:10],
            "largest_tags": [
                {"tag": row["category"], "count": row["count"]} for row in counts[:5]
            ],
        }

    def suggest_tag_merges(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """Suggest duplicate tags: spelling variants, then synonyms, then splits.

        Three detectors, cheapest first, each catching what the previous cannot:
        deterministic inflection, an LLM synonym pass, and vector-centroid
        overlap for the near-synonyms that share no words ("liver bloods" beside
        "liver lab results").
        """
        proposals = suggest_canonical_merges(self.llm, self.categories(user_id=user_id))
        seen = {v for group in proposals for v in group["variants"]}
        for pair in self.semantic_tag_duplicates(user_id=user_id):
            if not seen.intersection(pair["variants"]):
                proposals.append(pair)
                seen.update(pair["variants"])
        return proposals

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
        scope = Scope(user_id=user_id)
        indexed = self.backend.retag_topics(scope, remove, add)
        if indexed is not None:
            for tag in remove:
                self.backend.delete_synthetic_tag(scope, tag)
            return indexed
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
        try:
            self.llm.close()
        finally:
            try:
                self.embedder.close()
            finally:
                self.backend.close()
