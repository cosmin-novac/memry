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
from ..providers.decisions import Choice, Decider
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


ACTION_QUESTION = Choice(
    instructions="What should happen to the store, given the NEW fact?",
    criteria={
        "ADD": "The new fact is new information. Keep the existing memories and store it too.",
        "UPDATE": "The new fact replaces one existing memory that is now out of date.",
        "DELETE": "The new fact says one existing memory was wrong and it should go.",
        "NONE": "An existing memory already says this. Store nothing.",
    },
)


def _decide_action(
    decider: Decider, state: str, count: int
) -> dict[str, Any] | None:
    """Which action, and against which memory, as two typed questions in one call.

    Only the action and the target come from here. Writing the merged sentence
    for an UPDATE is a writing task and stays with the text model, so this is a
    cheaper call in front of a rarer expensive one rather than a replacement.
    Returns None when the provider abstains, leaving the old path in charge.
    """
    if not decider.available or count <= 0:
        return None
    questions: dict[str, Choice] = {"action": ACTION_QUESTION}
    if count > 1:
        questions["target"] = Choice(
            instructions="Which existing memory does the NEW fact act on?",
            criteria={str(i): f"memory [{i}]" for i in range(count)},
        )
    answers = decider.decide(state, questions)
    action = answers["action"]
    if not action.available:
        return None
    if count == 1:
        target: int | None = 0
    else:
        chosen = answers["target"]
        target = int(chosen.value) if chosen.available else None
    return {
        "action": action.value,
        "target": target,
        "content": None,            # an UPDATE still needs prose written for it
        "reason": f"{decider.name}: {action.value} at {action.confidence:.2f}",
        "confidence": action.confidence,
    }


def reconcile_candidate(
    *,
    candidate: CandidateFact,
    scope: Scope,
    similar: list[SearchResult],
    backend: MemoryBackend,
    embedder: Embedder,
    llm: LLM,
    episode_ids: list[str],
    decider: Decider | None = None,
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
    if similar:
        listing = "\n".join(
            f"[{i}] {r.memory.content}" for i, r in enumerate(similar)
        )
        state = f"EXISTING memories:\n{listing}\n\nNEW fact:\n{candidate.content}"
        judged = _decide_action(decider, state, len(similar)) if decider else None
        if judged is not None:
            decision = judged
        elif llm.available:
            raw = llm.complete(
                RECONCILE_SYSTEM, state, json_schema=RECONCILE_SCHEMA,
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
        # A rewrite that says when the thing happens sets the occurrence time;
        # one that says nothing about time leaves the stored one alone, because
        # the merged text still describes the same event.
        when = (candidate.metadata or {}).get("when")
        extra: dict[str, Any] = {}
        if when:
            extra["metadata"] = {**(target.metadata or {}), "when": when}
        backend.update_memory(
            target.id,
            content=new_content,
            embedding=embedding,
            embedding_model=embedder.model_id if embedding else None,
            importance=max(target.importance, candidate.importance),
            source_episode_ids=merged_sources,
            **extra,
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
