"""Core data models for Memry.

Design principles (see docs/research/competitive-analysis.md):
- Raw *episodes* are immutable and stored separately from derived *memories*
  (memories are a derived index; episodes are the source of truth).
- Memories are bi-temporal: they carry both transaction time (created/updated)
  and validity (valid_from / invalid_at). Contradicted memories are invalidated
  and superseded, never silently destroyed.
- Every mutation is recorded as a MemoryEvent (audit trail).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

MemoryType = Literal["semantic", "episodic", "procedural", "working"]
EventType = Literal["ADD", "UPDATE", "DELETE", "SUPERSEDE", "NONE"]

MEMORY_TYPES: tuple[str, ...] = ("semantic", "episodic", "procedural", "working")


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> str:
    """ISO-8601 UTC timestamp (second precision, sorts lexicographically)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class Scope(BaseModel):
    """Memory scoping, mem0-compatible: any combination of user/agent/run.

    A ``None`` field means "don't filter on this dimension".
    """

    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None

    def is_empty(self) -> bool:
        return self.user_id is None and self.agent_id is None and self.run_id is None


class Episode(BaseModel):
    """An immutable raw event (one conversation message or ingested record)."""

    id: str = Field(default_factory=new_id)
    content: str
    role: str = "user"
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)


class Memory(BaseModel):
    """A derived, reconciled unit of long-term memory."""

    id: str = Field(default_factory=new_id)
    content: str
    memory_type: MemoryType = "semantic"
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    importance: float = 0.5
    categories: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    valid_from: str | None = None
    invalid_at: str | None = None  # set => memory no longer believed true
    superseded_by: str | None = None  # id of the memory that replaced this one
    source_episode_ids: list[str] = Field(default_factory=list)  # provenance
    embedding_model: str | None = None

    @property
    def is_active(self) -> bool:
        return self.invalid_at is None

    def scope(self) -> Scope:
        return Scope(user_id=self.user_id, agent_id=self.agent_id, run_id=self.run_id)


class MemoryEvent(BaseModel):
    """Audit-trail entry for a memory mutation."""

    id: str = Field(default_factory=new_id)
    memory_id: str
    event: EventType
    old_content: str | None = None
    new_content: str | None = None
    reason: str | None = None
    actor: str = "system"  # "system" | "user" | "decay" | ...
    created_at: str = Field(default_factory=utcnow)


class CandidateFact(BaseModel):
    """A fact proposed by the extraction stage, before reconciliation."""

    content: str
    memory_type: MemoryType = "semantic"
    importance: float = 0.5
    categories: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    entity_types: dict[str, str] = Field(default_factory=dict)  # name_lower -> type
    # (subject_surface, predicate, object_surface) triples between this fact's
    # entities; resolved to typed Relation edges after entity linking.
    relations: list[dict[str, str]] = Field(default_factory=list)
    # Carried onto the stored memory. Set to {"pending_distillation": True}
    # when extraction is deferred or skipped, so a managed worker or explicit
    # distillation can process the active verbatim memory later.
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddAction(BaseModel):
    """What the reconciler decided to do with one candidate fact."""

    event: EventType
    memory_id: str | None = None
    content: str | None = None
    reason: str | None = None


class AddResult(BaseModel):
    episode_ids: list[str] = Field(default_factory=list)
    actions: list[AddAction] = Field(default_factory=list)
    # Non-fatal degradations the caller should surface to the user
    # (e.g. "extraction failed; stored verbatim").
    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.actions:
            counts[a.event] = counts.get(a.event, 0) + 1
        return counts


class SearchResult(BaseModel):
    memory: Memory
    score: float
    signals: dict[str, float] = Field(default_factory=dict)


class Topic(BaseModel):
    """A canonical classification label within one memory scope."""

    id: str = Field(default_factory=new_id)
    name: str
    normalized: str = ""
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    provenance: str = "memory"
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)


class TopicRelation(BaseModel):
    """A taxonomy edge: ``broader`` contains the narrower topic."""

    id: str = Field(default_factory=new_id)
    broader_topic_id: str
    narrower_topic_id: str
    user_id: str | None = None
    provenance: str = "synthetic"
    created_at: str = Field(default_factory=utcnow)

class Entity(BaseModel):
    """A distinct real-world thing (person/org/place/...) referenced by memories.

    Disambiguation is conservative: two mentions of "Jonas" are separate
    entities until evidence shows they are clearly the same. Merging sets
    ``merged_into`` on the losing entity; nothing is deleted.
    """

    id: str = Field(default_factory=new_id)
    name: str
    normalized: str = ""
    entity_type: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    description: str | None = None
    description_updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    merged_into: str | None = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

    @property
    def is_active(self) -> bool:
        return self.merged_into is None


class EntityMention(BaseModel):
    """One memory referring to one entity (by some surface text)."""

    id: str = Field(default_factory=new_id)
    entity_id: str
    memory_id: str
    surface: str
    created_at: str = Field(default_factory=utcnow)


ProposalStatus = Literal["proposed", "confirmed", "rejected"]


class MergeProposal(BaseModel):
    """A suspected same-entity pair awaiting confirmation (by the system once
    evidence is clear, or by the user)."""

    id: str = Field(default_factory=new_id)
    entity_a: str
    entity_b: str
    user_id: str | None = None
    status: ProposalStatus = "proposed"
    confidence: float = 0.5
    reason: str | None = None
    created_at: str = Field(default_factory=utcnow)
    decided_at: str | None = None


class Collection(BaseModel):
    """A titled, summarized cluster of memories - the coarse navigation layer.

    Built by clustering memory embeddings and having an LLM name and summarize
    each cluster from the actual memories (not from tags), so the label is
    grounded and specific. A memory can belong to several collections."""

    id: str = Field(default_factory=new_id)
    title: str
    summary: str = ""
    memory_ids: list[str] = Field(default_factory=list)
    user_id: str | None = None
    created_at: str = Field(default_factory=utcnow)


class Relation(BaseModel):
    """A typed edge between two entities, grounded in the memory that stated it.

    Relations are what make multi-hop recall work: "what tool does Ada use?"
    is answered by traversing ``Ada -works_on-> project -uses-> tool``, never by
    similarity (the answer memory names neither Ada nor "tool"). Bi-temporal like
    memories, so a relation can be superseded without being destroyed."""

    id: str = Field(default_factory=new_id)
    subject: str  # entity id
    predicate: str  # normalized short verb phrase, e.g. "works_on", "uses"
    object: str  # entity id
    user_id: str | None = None
    memory_id: str | None = None  # the memory that stated it (provenance)
    created_at: str = Field(default_factory=utcnow)
    valid_from: str = Field(default_factory=utcnow)
    invalid_at: str | None = None


class SyntheticTag(BaseModel):
    """A higher-level tag an LLM proposed to cluster several existing tags.

    Recorded so the system remembers which tags it invented (vs. tags that came
    from the user/extraction), can avoid re-proposing them, and can show or undo
    them later. The label is NOT copied onto member memories: it exists only as
    a ``TopicRelation`` edge, and query-time hierarchy expansion makes a filter
    on the parent reach the memories tagged with its children."""

    id: str = Field(default_factory=new_id)
    tag: str
    source_tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    created_at: str = Field(default_factory=utcnow)


class ContextResult(BaseModel):
    text: str
    memory_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
