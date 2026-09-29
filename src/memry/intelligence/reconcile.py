"""Reconciliation: decide what to do with each candidate fact.

This is the Mem0-paper phase 2 (ADD / UPDATE / DELETE / NOOP), upgraded with
Zep-style temporal semantics: a contradicted memory is *invalidated and
superseded* (kept for audit + time-travel), never destroyed.

Decisions:
- ADD      - genuinely new information -> new memory
- UPDATE   - refines/extends an existing memory -> rewrite it in place with
             the merged text the text model writes. When nothing wrote that
             text (no text model), the existing memory is kept and superseded
             by the new one, so no text is lost; held back as a DELETE is.
- DELETE   - contradicts an existing memory -> invalidate old, add new,
             link old.superseded_by -> new.id  (temporal supersede).
             Only where little is at stake: see ``held_back``. Otherwise both
             stay in use, the new one marked as conflicting, and a person
             decides under Upkeep.
- NONE     - duplicate / already known -> skip

The memories compared are the user's across runs (``store._across_runs``).
Where the one acted on belongs to another run than the save, a DELETE still
supersedes it, but a NONE or an UPDATE leaves it alone and adds the new fact
to the save's run: a search of that run must find what was said in it, and
the consolidation pass merges such duplicates later.

Without an LLM, reconciliation degrades to exact-duplicate detection.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from ..backends.base import MemoryBackend
from ..config import RetrievalConfig, SupersedeConfig
from ..models import (
    AddAction,
    CandidateFact,
    Memory,
    MemoryEvent,
    Scope,
    SearchResult,
    later_ts,
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


#: Appended to the reconcile prompt when a decision provider has already
#: chosen UPDATE: the text model only writes the merged sentence.
MERGE_REQUEST = ('The action is decided: UPDATE memory [0]. Reply with action "UPDATE", '
                 "target 0 and, as content, the merged replacement text.")


def write_merged(llm: LLM, existing: str, new: str) -> str | None:
    """The merged text for an UPDATE a decision provider chose, written by
    the text model with the prompt the no-provider path uses. None when there
    is no text model, it failed, or it wrote nothing."""
    if not llm.available:
        return None
    state = f"EXISTING memories:\n[0] {existing}\n\nNEW fact:\n{new}\n\n{MERGE_REQUEST}"
    try:
        raw = llm.complete(RECONCILE_SYSTEM, state, json_schema=RECONCILE_SCHEMA)
    except Exception:  # a provider hiccup must not cost the save its text
        return None
    parsed = parse_lenient_json(raw)
    content = parsed.get("content") if isinstance(parsed, dict) else None
    return content.strip() if isinstance(content, str) and content.strip() else None


#: Metadata key on a memory that was kept beside the one it contradicts.
CONFLICT_KEY = "conflict"

#: How the SUPERSEDE event of an UPDATE nobody could write a merged text for
#: begins. The newer memory adds to the old one; it does not contradict it, so
#: the Archive's undo brings the old one back beside it (``MemoryStore.
#: undo_replacement``) instead of forgetting it.
UPDATE_SUPERSEDE_REASON = "updated by new information, with no merged text written"


def held_back(
    target: Memory, decision: dict[str, Any], cfg: SupersedeConfig
) -> str | None:
    """Why this replacement has to be asked about, or None when it may go ahead.

    A contradiction is one model's reading of one text. That is enough to retire
    a passing remark, and a wrong call there is undone from the review list. It
    is not enough to retire what the store has the most reason to believe: a
    memory rated important, or one that several separate saves have stated.
    """
    if target.importance >= cfg.protect_importance:
        return (
            f"the memory it would replace is rated important "
            f"({target.importance:.2f})"
        )
    sources = len(target.source_episode_ids or [])
    if sources >= cfg.protect_sources:
        return f"the memory it would replace was stated in {sources} separate saves"
    confidence = decision.get("confidence")
    if isinstance(confidence, (int, float)) and confidence < cfg.confidence:
        return f"the judgement was only {confidence:.2f} sure"
    return None


def in_save_scope(memory: Memory, scope: Scope) -> bool:
    """Whether a memory belongs to the save's own scope (its run, agent and
    user; a field the save leaves None matches any). NONE and UPDATE act only
    on such a memory (``reconcile_candidate``)."""
    return all(getattr(scope, field) is None or getattr(memory, field) == getattr(scope, field)
               for field in ("user_id", "agent_id", "run_id"))


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
    supersede_cfg: SupersedeConfig | None = None,
    prepare_update: Callable[[str, str], dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> AddAction:
    """Apply one candidate fact against the store and return what happened.

    ``similar`` may hold memories of other runs of the user than the save's
    (``scope``): a DELETE supersedes one of them, a NONE or an UPDATE of one
    adds the new fact to the save's run and leaves it alone
    (``in_save_scope``).

    ``created_at`` is the time of the save (``MemoryStore.add``): a new
    memory's ``created_at``, ``updated_at`` and ``valid_from``, the
    ``updated_at`` of a memory an UPDATE rewrites (never moved back), the
    ``invalid_at`` of one it supersedes, and the time of the events recorded.
    The clock when None."""
    stamped: dict[str, Any] = {"created_at": created_at} if created_at else {}

    # Fast path: an exact duplicate needs no model call. One in the save's
    # own run is the fact already known there (NONE); one of another run is
    # added to this run by rule, as a NONE of it would be below, so a search
    # of the run finds it and that memory is left alone.
    norm = _normalize(candidate.content)
    duplicates = [r.memory for r in similar if _normalize(r.memory.content) == norm]
    for memory in duplicates:
        if in_save_scope(memory, scope):
            return AddAction(
                event="NONE",
                memory_id=memory.id,
                content=memory.content,
                reason="exact duplicate",
            )

    decision: dict[str, Any] = {"action": "ADD", "target": None, "content": None, "reason": "new information"}
    judged: dict[str, Any] | None = None
    if duplicates:
        decision["reason"] = (f"exact duplicate of memory {duplicates[0].id} of another run, "
                              "left as it is; added to this run")
    elif similar:
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
    if action in ("NONE", "UPDATE") and target is not None and not in_save_scope(target, scope):
        # another run's memory says it already: the fact is added to this
        # run, so a search of the run finds it, and that memory is left alone
        reason = (f"{action} of memory {target.id} of another run, left as it is; "
                  f"added to this run. {reason}").strip()
        action, target = "ADD", None

    if action == "NONE":
        return AddAction(
            event="NONE",
            memory_id=target.id if target else None,
            content=target.content if target else None,
            reason=reason or "already known",
        )

    # An UPDATE rewrites the target with merged text. A decision provider
    # chooses the action only, and the text model's own reply can leave the
    # text out as well, so whenever the decision carries none the text model
    # is asked to write it; when nothing wrote it, overwriting the target with
    # the new fact alone would lose what only the target said, so the new
    # memory supersedes it instead (held back as a contradiction would be).
    superseding = False
    new_content = ""
    if action == "UPDATE" and target is not None:
        new_content = str(decision.get("content") or "").strip()
        if not new_content:
            new_content = write_merged(llm, target.content, candidate.content) or ""
        superseding = not new_content

    if action == "UPDATE" and target is not None and not superseding:
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
            touch=created_at is None,
            **extra,
            **prepared,
        )
        if created_at is not None:
            # a replayed save older than the memory's last change does not
            # move its updated_at back
            backend.set_memory_timestamp(target.id, later_ts(target.updated_at, created_at))
        backend.add_event(
            MemoryEvent(
                memory_id=target.id,
                event="UPDATE",
                old_content=target.content,
                new_content=new_content,
                reason=reason or "refined by new information",
                **stamped,
            )
        )
        return AddAction(event="UPDATE", memory_id=target.id, content=new_content, reason=reason)

    # ADD, possibly preceded by a supersede: of a contradicted target (DELETE)
    # or of one an UPDATE nobody wrote the merged text for. Both take the
    # target out of use, so both are held back where that needs asking
    # (``held_back``); the marker's ``kind`` says which it was.
    held = (
        held_back(target, decision, supersede_cfg or SupersedeConfig())
        if (action == "DELETE" or superseding) and target is not None
        else None
    )
    metadata = dict(candidate.metadata or {})
    if held and target is not None:
        metadata[CONFLICT_KEY] = {
            "with": target.id,
            "reason": reason or ("updated by new information" if superseding
                                 else "contradicted by new information"),
            "held": held,
            "at": utcnow(),
            **({"kind": "update"} if superseding else {}),
        }
    new_memory = Memory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        importance=(max(target.importance, candidate.importance)
                    if superseding and target is not None else candidate.importance),
        categories=candidate.categories,
        entities=candidate.entities,
        metadata=metadata,
        source_episode_ids=episode_ids,
        created_at=created_at or utcnow(),
        updated_at=created_at or utcnow(),
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
            reason=(
                f"kept beside memory {target.id}, which it "
                f"{'updates' if superseding else 'contradicts'}, "
                f"because {held}. {reason}".strip()
                if held and target is not None
                else f"updates memory {target.id}, which it supersedes. {reason}".strip()
                if superseding and target is not None
                else reason or "new information"
            ),
            **stamped,
        )
    )

    if held and target is not None:
        return AddAction(
            event="ADD",
            memory_id=stored.id,
            content=stored.content,
            reason=(f"{'updates' if superseding else 'conflicts with'} memory "
                    f"{target.id}; waiting for review ({held})"),
            conflicts_with=target.id,
        )

    if (action == "DELETE" or superseding) and target is not None:
        # out of use when the save happened: a replay's time, not the clock
        backend.invalidate_memory(target.id, superseded_by=stored.id, at=created_at)
        backend.add_event(
            MemoryEvent(
                memory_id=target.id,
                event="SUPERSEDE",
                old_content=target.content,
                new_content=stored.content,
                reason=(
                    f"{UPDATE_SUPERSEDE_REASON}: kept and superseded. {reason}".strip()
                    if superseding else reason or "contradicted by new information"
                ),
                kind="update" if superseding else "contradiction",
                **stamped,
            )
        )
        return AddAction(
            event="SUPERSEDE" if superseding else "DELETE",
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
