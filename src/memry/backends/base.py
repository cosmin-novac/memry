"""Storage backend interface.

The application layer (``MemoryStore``) and the intelligence layer only ever
talk to this interface, so backends are replaceable: the default is the local
SQLite engine; a Mem0 adapter ships as an optional interop/benchmark backend,
and Postgres/Qdrant/etc. can be added without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing import TYPE_CHECKING

from ..models import (
    Collection,
    Entity,
    EntityMention,
    Episode,
    Memory,
    MemoryEvent,
    MergeProposal,
    Relation,
    Scope,
    SyntheticTag,
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
        metadata: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
    ) -> Memory | None: ...

    @abstractmethod
    def invalidate_memory(
        self, memory_id: str, *, superseded_by: str | None = None
    ) -> Memory | None:
        """Temporal soft-delete: mark the memory as no longer valid."""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Hard delete (rarely what you want; prefer invalidate)."""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Memory | None: ...

    @abstractmethod
    def list_memories(
        self,
        scope: Scope,
        *,
        include_invalid: bool = False,
        limit: int = 100,
        offset: int = 0,
        categories: list[str] | None = None,
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
    ) -> list[tuple[Memory, float]]:
        """Full-text (BM25) search. Higher score = better."""

    def native_search(
        self, query: str, scope: Scope, limit: int = 20
    ) -> list[tuple[Memory, float]] | None:
        """Backends with their own fused retrieval (e.g. Mem0) return results
        here; ``None`` means "use memry's hybrid fusion" (the default)."""
        return None

    # -- entities ---------------------------------------------------------
    # Default implementations are no-ops so adapters without entity support
    # (e.g. Mem0) stay valid; LocalBackend and PostgresBackend override all.
    def insert_entity(self, entity: Entity) -> Entity:
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return None

    def find_entities(self, normalized: str, scope: Scope) -> list[Entity]:
        """Active (unmerged) entities with this normalized name, in scope."""
        return []

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100
    ) -> list[Entity]:
        return []

    def add_mention(self, mention: EntityMention) -> None:
        return None

    def entity_mentions(self, entity_id: str) -> list[EntityMention]:
        return []

    def entity_memories(self, entity_id: str, limit: int = 10) -> list[Memory]:
        """Memories that mention this entity (following merges)."""
        return []

    def entities_of_memory(self, memory_id: str) -> list[Entity]:
        """The entities a single memory mentions (for relation backfill)."""
        return []

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

    # -- collections + vectors (for RAPTOR-lite summaries) ----------------
    def memory_vectors(
        self, scope: Scope, *, limit: int = 5000
    ) -> list[tuple[str, "np.ndarray"]]:
        """(memory_id, embedding) for active memories, for clustering."""
        return []

    def record_collection(self, collection: Collection) -> None:
        return None

    def list_collections(self, scope: Scope) -> list[Collection]:
        return []

    def clear_collections(self, scope: Scope) -> int:
        return 0

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
    # valid; LocalBackend and PostgresBackend override. A backend that does not
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
