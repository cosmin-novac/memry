"""Storage backend interface.

MemoryStore and the intelligence layer use this interface to isolate
persistence behavior. Production always constructs the local SQLite engine.
Explicit backend injection exists only for tests and comparison/import
utilities; it is not a runtime configuration choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing import TYPE_CHECKING

from ..models import (
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Relation,
    Scope,
    SyntheticTag,
    Topic,
    TopicRelation,
)

if TYPE_CHECKING:
    import numpy as np


class MemoryBackend(ABC):
    """Persistence contract: episodes (raw), memories (derived), events (audit)."""

    # -- episodes -------------------------------------------------------
    @abstractmethod
    def add_episodes(self, episodes: list[Episode]) -> None: ...

    @abstractmethod
    def list_episodes(self, scope: Scope, limit: int = 100) -> list[Episode]: ...

    # -- memories -------------------------------------------------------
    @abstractmethod
    def insert_memory(self, memory: Memory, embedding: list[float] | None = None) -> Memory:
        """Persist a new memory. Returns the stored memory (backends may
        assign their own id)."""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        entities: list[str] | None = None,
        mentions: list[EntityMention] | None = None,
        metadata: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
        touch: bool = True,
    ) -> Memory | None:
        """``touch=False`` updates stored fields WITHOUT moving ``updated_at``
        (for housekeeping like tagging/backfill/re-embedding, which must not
        reset a memory's recency or decay age)."""

    def list_pending_memories(
        self, limit: int = 100, *, due_before: str | None = None
    ) -> list[Memory]:
        """Active verbatim memories awaiting background enrichment."""
        return []

    @abstractmethod
    def invalidate_memory(
        self, memory_id: str, *, superseded_by: str | None = None
    ) -> Memory | None:
        """Temporal soft-delete: mark the memory as no longer valid."""

    def revalidate_memory(self, memory_id: str) -> "Memory | None":
        """Undo an invalidation: the memory is believed true again."""
        return None

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Hard delete (rarely what you want; prefer invalidate)."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None: ...

    def set_memory_timestamp(self, memory_id: str, updated_at: str) -> None:
        """Set updated_at directly (for repairing dates from the audit trail)."""
        return None

    @abstractmethod
    def list_memories(
        self,
        scope: Scope,
        *,
        include_invalid: bool = False,
        limit: int = 100,
        offset: int = 0,
        categories: list[str] | None = None,
        entity_id: str | None = None,
    ) -> list[Memory]: ...

    # -- search primitives ---------------------------------------------
    @abstractmethod
    def vector_search(
        self,
        embedding: list[float],
        embedding_model: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Cosine similarity over stored vectors (same embedding model only)."""

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        scope: Scope,
        limit: int = 20,
        include_invalid: bool = False,
        categories: list[str] | None = None,
        entity_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Full-text (BM25) search. Higher score = better."""

    def native_search(
        self, query: str, scope: Scope, limit: int = 20
    ) -> list[tuple[Memory, float]] | None:
        """Backends with their own fused retrieval (e.g. Mem0) return results
        here; ``None`` means "use memry's hybrid fusion" (the default)."""
        return None

    # -- topics -----------------------------------------------------------
    def upsert_topic(self, topic: Topic) -> Topic:
        return topic

    def list_topics(self, scope: Scope, *, limit: int = 1000) -> list[Topic]:
        return []

    def topic_counts(self, scope: Scope) -> list[dict[str, Any]] | None:
        """Active-memory counts, or ``None`` when topics are unsupported.

        Counts roll up through the hierarchy: a parent includes the memories
        of its descendants. Correct for browsing and for parent filtering.
        """
        return None

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an entity and its mentions/relations/proposals. Memories stay."""
        return False

    def purge_orphan_entities(self, scope: Scope) -> int:
        """Delete active entities that nothing references. Returns the count.

        An entity with no mentions, no relations and no merge history is not
        evidence of anything; it is a record of an extraction that went nowhere.
        """
        return 0

    def topic_memory_ids(self, scope: Scope) -> list[tuple[str, str]] | None:
        """``(topic_normalized, memory_id)`` for direct links on active memories.

        Cheap enough to group in Python, which is what tag-health checks need:
        a tag's centroid is the mean of its members' existing vectors.
        """
        return None

    def direct_topic_counts(self, scope: Scope) -> list[dict[str, Any]] | None:
        """Active-memory counts for directly-attached topics only.

        Abstraction must run on this, never on ``topic_counts``: a rolled-up
        histogram lists system-generated parents as if they were ordinary tags,
        so the next run clusters ``liver health`` and ``weekly gym`` into
        ``health`` and the useful level is lost a run at a time.
        """
        return None

    def retag_topics(
        self, scope: Scope, remove: set[str], add: str | None
    ) -> int | None:
        """Set-based topic edit, or ``None`` when an adapter has no topic store."""
        return None

    def add_topic_relation(self, relation: TopicRelation) -> TopicRelation:
        return relation

    def list_topic_relations(self, scope: Scope) -> list[TopicRelation]:
        return []
    # -- entities ---------------------------------------------------------
    # Default implementations are no-ops so adapters without entity support
    # (e.g. Mem0) stay valid; LocalBackend implements the production behavior.
    def insert_entity(self, entity: Entity) -> Entity:
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return None

    def resolve_entity_id(self, entity_id: str) -> str | None:
        """Follow ``merged_into`` links to the active entity ID."""
        current = entity_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            entity = self.get_entity(current)
            if entity is None:
                return None
            if entity.merged_into is None:
                return entity.id
            current = entity.merged_into
        return None

    def find_entities(self, normalized: str, scope: Scope) -> list[Entity]:
        """Active (unmerged) entities with this normalized name, in scope."""
        return []

    def find_entity_candidates(
        self, normalized: str, scope: Scope, *, limit: int = 20
    ) -> list[Entity]:
        """Active entities matching a canonical name or derived alias."""
        return self.find_entities(normalized, scope)[:limit]

    def find_entities_by_aliases(
        self, normalized: list[str], scope: Scope, *, limit: int = 50
    ) -> list[Entity]:
        seen: set[str] = set()
        matches: list[Entity] = []
        for value in normalized:
            for entity in self.find_entity_candidates(value, scope, limit=limit):
                if entity.id not in seen:
                    seen.add(entity.id)
                    matches.append(entity)
                    if len(matches) >= limit:
                        return matches
        return matches

    def entity_aliases(self, entity_id: str) -> list[str]:
        entity = self.get_entity(entity_id)
        return [entity.name] if entity else []

    def add_entity_alias(self, entity_id: str, alias: str) -> Entity | None:
        return None

    def set_entity_description(
        self, entity_id: str, description: str, generated_at: str
    ) -> Entity | None:
        return None

    def entity_evidence_updated_at(self, entity_id: str) -> str | None:
        return None

    def set_entity_type(self, entity_id: str, entity_type: str) -> None:
        return None

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100
    ) -> list[Entity]:
        return []

    def add_mention(self, mention: EntityMention) -> None:
        return None

    def entity_mentions(self, entity_id: str) -> list[EntityMention]:
        return []

    def entity_memories(
        self, entity_id: str, limit: int = 10, *, include_invalid: bool = False
    ) -> list[Memory]:
        """Memories that mention this entity. Active evidence is the default."""
        return []

    def entities_of_memory(self, memory_id: str) -> list[Entity]:
        """The entities a single memory mentions (for relation backfill)."""
        return []

    def touch_entity(self, entity_id: str) -> None:
        """Mark an entity hub stale after its evidence changes."""
        return None

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Fold ``merge_id`` into ``keep_id`` (repoint mentions, mark merged)."""
        return False

    # -- typed relations (anchor -> anchor edges) -------------------------
    # Default no-ops; LocalBackend implements. A backend without relations
    # simply has no multi-hop graph; retrieval falls back to hybrid.
    def add_relation(self, relation: Relation) -> Relation:
        return relation

    def list_relations(self, scope: Scope, *, limit: int = 1000) -> list[Relation]:
        return []

    def relations_of(self, entity_ids: list[str]) -> list[Relation]:
        """Active relations touching any of these entities (either endpoint)."""
        return []

    # -- vectors ----------------------------------------------------------
    def memory_vectors(
        self, scope: Scope, *, limit: int = 5000
    ) -> list[tuple[str, "np.ndarray"]]:
        """(memory_id, embedding) for active memories.

        Used by consolidation and tag health to compare what is stored without
        re-embedding anything.
        """
        return []

    def add_proposal(self, proposal: MergeProposal) -> MergeProposal:
        return proposal

    def get_proposal(self, proposal_id: str) -> MergeProposal | None:
        return None

    def find_proposal(self, entity_a: str, entity_b: str) -> MergeProposal | None:
        """Existing proposal for this unordered pair, any status."""
        return None

    def list_proposals(
        self, scope: Scope, *, status: str | None = "proposed", limit: int = 100
    ) -> list[MergeProposal]:
        return []

    def set_proposal_status(self, proposal_id: str, status: str) -> MergeProposal | None:
        return None

    # -- synthetic tags + key/value meta ----------------------------------
    # Default no-ops so adapters without their own storage (e.g. Mem0) stay
    # valid; LocalBackend implements persistence. An adapter that does not
    # persist these simply won't remember synthetic tags or scheduler state -
    # tag abstraction degrades to "runs but doesn't record", never crashes.
    def record_synthetic_tag(self, tag: SyntheticTag) -> None:
        return None

    def list_synthetic_tags(self, scope: Scope) -> list[SyntheticTag]:
        return []

    def delete_synthetic_tag(self, scope: Scope, tag: str) -> None:
        return None

    def distinct_user_ids(self) -> list[str | None]:
        """Namespaces present in the store, for the maintenance scheduler."""
        return []

    def get_meta(self, key: str) -> str | None:
        return None

    def set_meta(self, key: str, value: str) -> None:
        return None

    # -- events / audit ---------------------------------------------------
    @abstractmethod
    def add_event(self, event: MemoryEvent) -> None: ...

    @abstractmethod
    def history(self, memory_id: str) -> list[MemoryEvent]: ...

    # -- lossless backup / restore ---------------------------------------
    def export_backup(self, scope: Scope) -> dict[str, Any]:
        raise NotImplementedError("this backend cannot create lossless Memry backups")

    def import_backup(
        self, backup: dict[str, Any], *, owner_prefix: str | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError("this backend cannot restore lossless Memry backups")

    # -- maintenance ------------------------------------------------------
    @abstractmethod
    def all_memories_iter(self, include_invalid: bool = True) -> list[Memory]:
        """All memories, for reindexing/decay sweeps."""

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...

    @abstractmethod
    def reset(self) -> None: ...

    def close(self) -> None:  # pragma: no cover - trivial default
        pass
