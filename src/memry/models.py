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
    # Carried onto the stored memory. Set to {"pending_distillation": True}
    # when extraction was requested but skipped (no LLM / LLM failure), so the
    # memory can be distilled later.
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


class ContextResult(BaseModel):
    text: str
    memory_ids: list[str] = Field(default_factory=list)
    token_estimate: int = 0
