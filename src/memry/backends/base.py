"""Storage backend interface.

MemoryStore and the intelligence layer use this interface to isolate
persistence behavior. Production always constructs the local SQLite engine.
Explicit backend injection exists only for tests and comparison/import
utilities; it is not a runtime configuration choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
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
    TOPIC_TYPE,
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

    def episodes_by_id(self, episode_ids: list[str]) -> dict[str, Episode]:
        """The stored episodes among ``episode_ids``. A backend that cannot look
        episodes up by id returns none."""
        return {}

    # An episode is searched only as evidence of the memories resting on it
    # (``MemoryStore.evidence``): by its vector and its full-text entry.
    def set_episode_vectors(self, vectors: dict[str, list[float]], embedding_model: str) -> None:
        """Store each episode's embedding, as a memory's is stored."""
        return None

    def episode_vectors_of(
        self, episode_ids: list[str], embedding_model: str
    ) -> dict[str, "np.ndarray"]:
        """The stored vectors of these episodes made by ``embedding_model``."""
        return {}

    def episodes_to_embed(self, embedding_model: str, *, limit: int = 1000) -> list[Episode]:
        """Episodes with no vector of ``embedding_model`` yet, oldest first."""
        return []

    def episode_keyword_scores(self, query: str, episode_ids: list[str]) -> dict[str, float]:
        """The full-text (BM25) score of each of these episodes that matches
        ``query``, higher is better; an episode that does not match is left out."""
        return {}

    def evidence_episodes(self, episode_ids: list[str]) -> list[Episode]:
        """The episodes among these that may be shown as evidence of a memory,
        in the order they were said (time, then the order they were saved):
        not withheld (``Episode.withheld_at``), and with at least one memory
        resting on them in use and none removed (out of use with nothing in its
        place: forgotten). A backend that cannot tell shows none."""
        return []

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
        self, memory_id: str, *, superseded_by: str | None = None, at: str | None = None
    ) -> Memory | None:
        """Temporal soft-delete: mark the memory as no longer valid. ``at``
        (ISO 8601) is when, the memory's ``invalid_at``: a replayed save's time
        (``MemoryStore.add(created_at=...)``); the clock when None. Its
        ``updated_at`` becomes the later of its own and ``at``, never earlier,
        so a replayed save older than the memory's last change does not move
        it back. Whoever records the SUPERSEDE event gives it ``at`` as its
        time, which ``MemoryStore.repair_updated_at`` reads."""

    def revalidate_memory(self, memory_id: str) -> "Memory | None":
        """Undo an invalidation: the memory is believed true again."""
        return None

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Hard delete (rarely what you want; prefer invalidate). The
        memories it had replaced point at nothing afterwards: their
        ``superseded_by`` is cleared (``replaced_by`` lists them first)."""

    def replaced_by(self, memory_id: str) -> list[Memory]:
        """The memories ``memory_id`` replaced: those whose ``superseded_by``
        it is (the originals a consolidation or a distillation made it of, or
        the older memories it contradicted or updated)."""
        return []

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
    ) -> list[Memory]:
        """Memories in scope, newest first (``updated_at``, a tie by memory id)."""

    def update_proposal_judgement(
        self, proposal_id: str, *, confidence: float, reason: str | None,
        compared_step: int | None = None, different: float | None = None,
        belongs: dict[str, float] | None = None,
    ) -> None:
        """Record the latest comparison on an open proposal, so the list shows
        the provider's latest answer instead of its first, and the step of the
        comparison funnel it was made at (``identity.PAIR_STEPS``). ``belongs``
        is the latest answer to whether one entity is a version or a part of the
        other (``identity.BELONGS_QUESTION``). A backend that cannot store the
        latest answer keeps the first."""
        return None

    def count_memories(self, owner_prefix: str | None = None) -> dict[str, int]:
        """Active, invalidated and forgotten memory counts, optionally for one owner.

        ``owner_prefix`` follows ``Principal.prefix``: None is every memory, a
        value ending in ``::`` is a tenant prefix, anything else one exact
        namespace. Forgotten means removed with nothing standing in for it
        (``superseded_by`` empty), which is what the Forgotten tab lists.

        This default walks every memory so any backend gets a correct answer;
        a backend that can count in its store should override it.
        """
        def owned(user_id: str | None) -> bool:
            if owner_prefix is None:
                return True
            if not user_id:
                return False
            if owner_prefix.endswith("::"):
                return user_id.startswith(owner_prefix)
            return user_id == owner_prefix

        counts = {"active": 0, "invalidated": 0, "forgotten": 0}
        for memory in self.list_memories(Scope(), include_invalid=True, limit=10**12):
            if not owned(memory.user_id):
                continue
            if memory.invalid_at is None:
                counts["active"] += 1
            else:
                counts["invalidated"] += 1
                if not memory.superseded_by:
                    counts["forgotten"] += 1
        return counts

    def knowledge_map(self, scope: Scope) -> dict[str, Any]:
        """Aggregate active knowledge for visualization without exposing content.

        Runtime storage uses the SQLite override below. This bounded generic
        implementation keeps injected comparison/test backends compatible.
        """
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[tuple[str, str], int]] = {
            "tags": {},
            "entities": {},
        }
        memories = self.list_memories(scope, limit=1_000_000)
        entity_memory_ids: set[str] = set()

        def add_node(
            key: str,
            label: str,
            kind: str,
            memory_type: str,
            **extra: Any,
        ) -> None:
            node = nodes.setdefault(
                key,
                {
                    "key": key,
                    "label": label,
                    "kind": kind,
                    "count": 0,
                    "type_counts": {},
                    **extra,
                },
            )
            node["count"] += 1
            counts = node["type_counts"]
            counts[memory_type] = counts.get(memory_type, 0) + 1

        def add_edges(kind: str, keys: list[str]) -> None:
            unique = sorted(set(keys))
            for index, first in enumerate(unique):
                for second in unique[index + 1 :]:
                    pair = (first, second)
                    edges[kind][pair] = edges[kind].get(pair, 0) + 1

        for memory in memories:
            memory_type = str(memory.memory_type or "semantic")
            tags = sorted(
                {
                    str(category).strip().lower()
                    for category in (memory.categories or [])
                    if str(category).strip()
                }
            ) or ["(untagged)"]
            tag_keys = [f"tag:{tag}" for tag in tags]
            for tag, key in zip(tags, tag_keys):
                add_node(key, tag, "tag", memory_type)
            add_edges("tags", tag_keys)

            entities = self.entities_of_memory(memory.id)
            if entities:
                entity_memory_ids.add(memory.id)
            entity_keys = [f"entity:{entity.id}" for entity in entities]
            for entity, key in zip(entities, entity_keys):
                add_node(
                    key,
                    entity.name,
                    "entity",
                    memory_type,
                    entity_id=entity.id,
                    entity_type=entity.entity_type or "untyped",
                )
            add_edges("entities", entity_keys)

        def edge_rows(kind: str) -> list[dict[str, Any]]:
            return [
                {"a": a, "b": b, "weight": weight}
                for (a, b), weight in sorted(
                    edges[kind].items(), key=lambda item: (-item[1], item[0])
                )
            ]

        return {
            "memories": len(memories),
            "entity_memories": len(entity_memory_ids),
            "tags": [node for node in nodes.values() if node["kind"] == "tag"],
            "tag_edges": edge_rows("tags"),
            "entities": [
                node for node in nodes.values() if node["kind"] == "entity"
            ],
            "entity_edges": edge_rows("entities"),
        }

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
        """Cosine similarity over stored vectors (same embedding model only),
        best first, a tie by memory id."""

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
        """Full-text (BM25) search. Higher score = better, a tie by memory id."""

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

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an entity and its mentions/relations/proposals. Memories stay."""
        return False

    def retire_entity(self, entity_id: str, reason: str = "removed") -> bool:
        """Remove an entity after keeping a snapshot of everything removed.

        The recoverable counterpart of ``delete_entity``, and what the store
        uses: nothing the user can see disappears without a way back.
        """
        return False

    def restore_entity(self, entity_id: str) -> bool:
        """Undo a retirement. False when the id is not retired or is in use."""
        return False

    def list_retired_entities(
        self, scope: Scope, *, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Retired entities newest first: id, name, type, reason, timestamp."""
        return []

    def purge_orphan_entities(
        self, scope: Scope, *, reason: str = "nothing referenced it"
    ) -> int:
        """Retire active entities that nothing references. Returns the count.

        An entity with no mentions, no relations and no merge history is not
        evidence of anything; it is a record of an extraction that went nowhere.
        A tag (a topic entity) nothing mentions and no tombstone points at
        files nothing and is retired the same way.
        """
        return 0

    def retag_topics(
        self, scope: Scope, remove: set[str], add: str | None, *, exact_user: bool = False
    ) -> int | None:
        """Set-based topic edit, or ``None`` when an adapter has no topic store."""
        return None

    def tag_namespaces(self, names: Iterable[str]) -> list[str | None] | None:
        """Every namespace (``user_id``, None for the memories without one)
        that carries one of these tags: an active topic entity of that name,
        a memory filed under it, or a synthetic tag of it. ``None`` when an
        adapter has no topic store."""
        return None

    def add_topic_relation(self, relation: TopicRelation) -> TopicRelation:
        return relation

    # -- tags as topic entities --------------------------------------------
    # A tag is an entity of type ``models.TOPIC_TYPE``, one per user and
    # normalized tag. A backend that stores entities creates it the first time
    # a memory carries the tag and keeps each memory's mentions of tags in line
    # with its ``categories`` column; one that does not returns None from the
    # counts, and the store counts the column instead.
    def topic_entity(
        self, name: str, scope: Scope, *, create: bool = True, follow_merged: bool = False
    ) -> Entity | None:
        """The active topic entity of tag ``name`` for ``scope.user_id``,
        created when missing and ``create`` (one per user and name, however
        many processes create it at once). A tag merged away has none of its
        own and is never given a fresh one: with ``follow_merged`` or
        ``create`` it resolves to the entity it went into (another topic, or a
        named thing), as a memory's column naming it is filed there; else
        None. Merges resolve the names they merge away without it, so a name
        merged away is never merged again through its tombstone.

        ``user_id`` is a namespace exactly: "" is a user of its own, not the
        memories without one (None)."""
        return None

    def tag_filing(self, names: Iterable[str], scope: Scope) -> dict[str, str]:
        """For each tag (normalized), the name a memory's column files it
        under for ``scope.user_id``, creating nothing: itself, when it is an
        active topic or no topic at all; for a tag merged away, its survivor,
        following tombstones topic to topic; for one merged into a named
        thing, the last tag name of its chain (its mention goes to the
        thing). The one resolution every write of a column applies."""
        return {str(name).strip().lower(): str(name).strip().lower()
                for name in names if str(name).strip()}

    def topic_names(
        self, scope: Scope, *, prefixes: Iterable[str] | None = None
    ) -> dict[str, bool]:
        """The tag names of ``scope.user_id``'s topic entities, each with
        whether an active topic has it (False: only merged away). With
        ``prefixes``, only the names starting with one of them (leading
        separators aside), which is how the store narrows the vocabulary to a
        few tags' obvious variants."""
        return {}

    def topic_mention_counts(
        self, scope: Scope, *, exact_user: bool = False
    ) -> list[dict[str, Any]] | None:
        """Active memories per tag (``{"category", "count"}``), counted from
        the topic entities' mentions, largest first. ``exact_user`` reads
        ``scope.user_id`` None as the memories without a user, not as all."""
        return None

    def topic_mention_links(
        self, scope: Scope, *, exact_user: bool = False
    ) -> list[tuple[str, str]] | None:
        """``(tag, memory_id)`` for every active memory mentioning a tag."""
        return None

    def tags_to_topics(
        self, *, user_id: str | None = None, all_users: bool = True, dry_run: bool = False
    ) -> list[dict[str, Any]]:
        """Migrate the legacy ``topics``/``memory_topics`` rows to topic
        entities and mentions, per user. A backend without either has nothing
        to migrate."""
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

    #: Called after a write changed which names read "it" in an entity's
    #: memories, with the ids of the entities concerned: by the backend for
    #: the one a merge kept, one renamed, one given an alias, one a mention
    #: calls by a new wording and one restored; by the identity code for both
    #: of a pair given a new answer to whether one belongs to the other (a
    #: home's names read "it" in its parts' memories). ``MemoryStore`` sets it
    #: to refresh the property vectors of those memories, which would
    #: otherwise keep the old names until the weekly refresh. A retired
    #: entity has no memories left to name, so the store refreshes those it
    #: read before retiring it (``MemoryStore._retire``).
    names_changed: Callable[[list[str]], None] | None = None

    def entity_aliases(self, entity_id: str) -> list[str]:
        entity = self.get_entity(entity_id)
        return [entity.name] if entity else []

    def add_entity_alias(self, entity_id: str, alias: str) -> Entity | None:
        return None

    def rename_entity(self, entity_id: str, name: str) -> Entity | None:
        return None

    def set_entity_description(
        self, entity_id: str, description: str, generated_at: str
    ) -> Entity | None:
        return None

    def entity_evidence_updated_at(self, entity_id: str) -> str | None:
        return None

    def set_entity_type(self, entity_id: str, entity_type: str) -> None:
        return None

    def set_entity_metadata(self, entity_id: str, metadata: dict[str, Any]) -> None:
        """Replace an entity's metadata. Derived notes only (home, screening);
        it never touches ``updated_at``, so recomputing them changes no order."""
        return None

    def entity_memory_links(
        self, scope: Scope, *, kind: str = "named"
    ) -> list[tuple[str, str]]:
        """(entity_id, memory_id) for every active entity and active memory.
        ``kind`` as for ``list_entities``."""
        return []

    def list_entities(
        self, scope: Scope, *, include_merged: bool = False, limit: int = 100,
        kind: str = "named",
    ) -> list[Entity]:
        """Entities in scope. ``kind`` "named" (the default) is every type but
        ``models.TOPIC_TYPE``, "topic" only tags, "any" both. Name lookups
        (``find_entities``, ``find_entity_candidates``) only ever find named
        entities."""
        return []

    def add_mention(self, mention: EntityMention) -> None:
        return None

    def entity_mentions(self, entity_id: str) -> list[EntityMention]:
        return []

    def entity_memories(
        self, entity_id: str, limit: int = 10, *, include_invalid: bool = False,
        scope: Scope | None = None,
    ) -> list[Memory]:
        """Memories that mention this entity, newest first (``updated_at``, a
        tie by memory id, so memories of one time read alike in every build of
        a store). Active evidence is the default. ``scope`` keeps to the
        memories of that user, agent and run (a field None matches any) before
        ``limit`` counts, so a run's memories of an entity other runs mention
        far more are still read."""
        return []

    def count_entity_memories(self, entity_id: str) -> int:
        """How many active memories mention this entity."""
        return len(self.entity_memories(entity_id, limit=100_000))

    def entities_of_memory(self, memory_id: str, *, kind: str = "named") -> list[Entity]:
        """The entities a single memory mentions (for relation backfill).
        ``kind`` as for ``list_entities``: its tags only when asked for."""
        return []

    def entities_of_memories(
        self, memory_ids: list[str], *, kind: str = "named"
    ) -> dict[str, list[Entity]]:
        """``entities_of_memory`` of each of these memories, read at once
        where the backend can: every id is a key, with an empty list when the
        memory mentions nothing of that ``kind``."""
        return {mid: self.entities_of_memory(mid, kind=kind) for mid in memory_ids}

    def entity_memory_counts(
        self, entity_ids: list[str], *, scope: Scope | None = None
    ) -> dict[str, int]:
        """How many active memories mention each of these entities, counting
        only those of ``scope`` (its user, agent and run; a field None matches
        any) when given; read at once where the backend can."""
        return {entity_id: len(self.entity_memories(entity_id, limit=100_000, scope=scope))
                for entity_id in entity_ids}

    def topic_ids(self, entity_ids: Iterable[str]) -> set[str]:
        """Which of these entities are tags (topic entities); read at once
        where the backend can."""
        out: set[str] = set()
        for entity_id in set(entity_ids):
            entity = self.get_entity(entity_id)
            if entity is not None and entity.entity_type == TOPIC_TYPE:
                out.add(entity_id)
        return out

    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Fold ``merge_id`` into ``keep_id`` (repoint mentions, mark merged).
        A tag and a named thing are folded into the thing either way round."""
        return False

    def list_merges(self, scope: Scope, *, limit: int = 200) -> list[dict[str, Any]]:
        """Merges that can be undone, newest first. A backend that records
        none lists none."""
        return []

    def merge_record(self, entity_id: str) -> dict[str, Any] | None:
        """The last merge of ``entity_id`` into another on record, if any."""
        return None

    def undo_merge(self, entity_id: str) -> bool:
        """Undo the last merge of ``entity_id`` into another entity; False
        when there is none to undo."""
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

    def proposals_of(self, entity_ids: list[str]) -> list[MergeProposal]:
        """Compared pairs touching any of these entities that are not merged:
        open ones and ruled-out ones, with the judge's latest answers."""
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

    def vectors_of(
        self, memory_ids: list[str], embedding_model: str | None = None
    ) -> dict[str, "np.ndarray"]:
        """The stored embedding of each of these memories that has one, only
        from ``embedding_model`` when given."""
        return {}

    def unlabelled_vector_ids(self, scope: Scope) -> list[str]:
        """Valid memories that have a vector but no embedding model on it."""
        return []

    def consolidated_memories(self, scope: Scope) -> list["Memory"]:
        """Valid memories made by consolidating others."""
        return []

    def set_property_vectors(
        self, vectors: dict[str, list[float]], embedding_model: str,
        hashes: dict[str, str] | None = None,
    ) -> None:
        """Store each memory's property vector: its text with its own entity
        names replaced by "it", embedded (``graph_retrieval.mask_names``), and
        a hash of that masked text."""
        return None

    def property_vectors_of(
        self, memory_ids: list[str], embedding_model: str | None = None
    ) -> dict[str, "np.ndarray"]:
        """The property vector of each of these memories that has one, only
        from ``embedding_model`` when given."""
        return {}

    def property_vector_hashes(
        self, memory_ids: list[str]
    ) -> dict[str, tuple[str | None, str | None]]:
        """(masked text hash, embedding model) of each stored property vector."""
        return {}

    def delete_property_vectors(self, memory_ids: list[str]) -> None:
        """Drop these memories' property vectors."""
        return None

    def session_memories(
        self, memory: Memory, *, hours: float = 3.0, limit: int = 50
    ) -> list[Memory]:
        """Other active memories saved in the same conversation as ``memory``,
        within ``hours`` of it: the same session, or without one the same client
        and context label. A backend that cannot tell returns none."""
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

    def set_proposal_status(
        self, proposal_id: str, status: str, reason: str | None = None
    ) -> MergeProposal | None:
        """Decide a proposal; ``reason``, when given, says what decided it (a
        rule, or a person) in place of the answer it held."""
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
    def reset(self, *, keep_meta: Iterable[str] = ()) -> None:
        """Delete everything. Of the key/value meta, only the backend's own
        schema markers stay, and the keys starting with one of ``keep_meta``
        (the caller's settings)."""

    def close(self) -> None:  # pragma: no cover - trivial default
        pass
