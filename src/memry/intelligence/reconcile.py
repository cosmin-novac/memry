"""Reconciliation: decide what to do with each candidate fact.

This is the Mem0-paper phase 2 (ADD / UPDATE / DELETE / NOOP), upgraded with
Zep-style temporal semantics: a contradicted memory is *invalidated and
superseded* (kept for audit + time-travel), never destroyed.

Decisions:
- ADD      - genuinely new information -> new memory
- UPDATE   - refines/extends an existing memory -> rewrite it in place
- DELETE   - contradicts an existing memory -> invalidate old, add new,
             link old.superseded_by -> new.id  (temporal supersede)
- NONE     - duplicate / already known -> skip

Without an LLM, reconciliation degrades to exact-duplicate detection.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..backends.base import MemoryBackend
from ..config import RetrievalConfig
from ..models import (
    AddAction,
    CandidateFact,
    Memory,
    MemoryEvent,
    Scope,
    SearchResult,
    utcnow,
)
from ..providers.embeddings import Embedder
from ..providers.llm import LLM
from .extraction import parse_lenient_json

RECONCILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NONE"]},
        "target": {"type": ["integer", "null"]},
        "content": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "target", "content", "reason"],
    "additionalProperties": False,
}

RECONCILE_SYSTEM = """You maintain an AI assistant's long-term memory store.
Given a NEW fact and the most similar EXISTING memories, decide one action:

- "ADD": the new fact is new information not covered by any existing memory.
- "UPDATE": the new fact refines, extends, or corrects wording of an existing
  memory without contradicting it (e.g. adds detail). Set "target" to that
  memory's index and "content" to the merged, self-contained replacement text.
- "DELETE": the new fact contradicts an existing memory, which is no longer
  true (e.g. user moved cities, changed jobs, reversed a preference). Set
  "target" to the outdated memory's index. The old memory will be archived and
  the new fact stored.
- "NONE": the new fact is already fully captured by an existing memory.

Prefer UPDATE over ADD when the information overlaps. Prefer DELETE over
UPDATE when the old statement would now be false.
When writing UPDATE content, the replacement must preserve EVERY concrete
detail from both texts - numbers, dates, prices, names, versions, file
formats, tool names, constraints and their reasons. Never drop a detail to
make the merged text shorter.
Respond with JSON only:
{"action": "ADD"|"UPDATE"|"DELETE"|"NONE", "target": int|null,
 "content": str|null, "reason": short str}"""

_WS_RE = re.compile(r"\W+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def reconcile_candidate(
    *,
    candidate: CandidateFact,
    scope: Scope,
    similar: list[SearchResult],
    backend: MemoryBackend,
    embedder: Embedder,
    llm: LLM,
    episode_ids: list[str],
    retrieval_cfg: RetrievalConfig | None = None,
    prepare_update: Callable[[str, str], dict[str, Any]] | None = None,
) -> AddAction:
    """Apply one candidate fact against the store and return what happened."""

    # Fast path: exact duplicate needs no LLM round-trip.
    norm = _normalize(candidate.content)
    for result in similar:
        if _normalize(result.memory.content) == norm:
            return AddAction(
                event="NONE",
                memory_id=result.memory.id,
                content=result.memory.content,
                reason="exact duplicate",
            )

    decision: dict[str, Any] = {"action": "ADD", "target": None, "content": None, "reason": "new information"}
    if similar and llm.available:
        listing = "\n".join(
            f"[{i}] {r.memory.content}" for i, r in enumerate(similar)
        )
        raw = llm.complete(
            RECONCILE_SYSTEM,
            f"EXISTING memories:\n{listing}\n\nNEW fact:\n{candidate.content}",
            json_schema=RECONCILE_SCHEMA,
        )
        parsed = parse_lenient_json(raw)
        if isinstance(parsed, dict) and parsed.get("action") in ("ADD", "UPDATE", "DELETE", "NONE"):
            decision = parsed

    action = decision.get("action", "ADD")
    target_idx = decision.get("target")
    target: Memory | None = None
    if isinstance(target_idx, int) and 0 <= target_idx < len(similar):
        target = similar[target_idx].memory
    if action in ("UPDATE", "DELETE") and target is None:
        action = "ADD"  # malformed decision -> safest fallback

    reason = str(decision.get("reason") or "")

    if action == "NONE":
        return AddAction(
            event="NONE",
            memory_id=target.id if target else None,
            content=target.content if target else None,
            reason=reason or "already known",
        )

    if action == "UPDATE" and target is not None:
        new_content = str(decision.get("content") or candidate.content)
        embedding = _embed_or_none(embedder, new_content)
        merged_sources = list(dict.fromkeys(target.source_episode_ids + episode_ids))
        prepared = prepare_update(target.id, new_content) if prepare_update else {}
        backend.update_memory(
            target.id,
            content=new_content,
            embedding=embedding,
            embedding_model=embedder.model_id if embedding else None,
            importance=max(target.importance, candidate.importance),
            source_episode_ids=merged_sources,
            **prepared,
        )
        backend.add_event(
            MemoryEvent(
                memory_id=target.id,
                event="UPDATE",
                old_content=target.content,
                new_content=new_content,
                reason=reason or "refined by new information",
            )
        )
        return AddAction(event="UPDATE", memory_id=target.id, content=new_content, reason=reason)

    # ADD (possibly preceded by a supersede when action == DELETE)
    new_memory = Memory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        importance=candidate.importance,
        categories=candidate.categories,
        entities=candidate.entities,
        metadata=candidate.metadata,
        source_episode_ids=episode_ids,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    embedding = _embed_or_none(embedder, candidate.content)
    if embedding:
        new_memory.embedding_model = embedder.model_id
    stored = backend.insert_memory(new_memory, embedding)
    backend.add_event(
        MemoryEvent(
            memory_id=stored.id,
            event="ADD",
            new_content=stored.content,
            reason=reason or "new information",
        )
    )

    if action == "DELETE" and target is not None:
        backend.invalidate_memory(target.id, superseded_by=stored.id)
        backend.add_event(
            MemoryEvent(
                memory_id=target.id,
                event="SUPERSEDE",
                old_content=target.content,
                new_content=stored.content,
                reason=reason or "contradicted by new information",
            )
        )
        return AddAction(
            event="DELETE",
            memory_id=stored.id,
            content=stored.content,
            reason=reason or f"superseded memory {target.id}",
        )

    return AddAction(event="ADD", memory_id=stored.id, content=stored.content, reason=reason)


def _embed_or_none(embedder: Embedder, text: str) -> list[float] | None:
    if not embedder.dimensions:
        return None
    try:
        vectors = embedder.embed([text])
        return vectors[0] if vectors and vectors[0] else None
    except Exception:
        return None
