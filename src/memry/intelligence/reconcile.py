"""Reconciliation: what a new fact does to the memories already stored.

Each fact a save yields is compared with the most similar current memories of
the same user, across all of the user's runs (``store._across_runs``), and is
judged one of five things. The judgement acts the same way whichever run the
memory it names belongs to; the save's run decides only where a new memory is
stored.

- NEW      new information, including another occurrence of the same kind of
           event (two yoga classes stay two): the fact is added.
- SAME     it says nothing the memory does not: no second live copy. The save
           is recorded on that memory as evidence (``_restated``): its
           episodes join the memory's ``source_episode_ids``, and a NONE
           event at the save's time says when it was last said. The episodes
           carry the save's run, so a search restricted to that run finds the
           memory: a memory is in a run when it is the run's own or when its
           evidence includes an episode of the run (``LocalBackend``'s search
           scope). Nothing but the memory's evidence records it.
- MORE     it adds detail to a memory that stays true: the text model writes
           one text of both (``write_merged``), stored as a new memory dated
           at the save, which supersedes the old one as an update. With no
           merged text written, the new fact itself supersedes the old one as
           an update, held back as a CHANGED is.
- CHANGED  the memory was true and is no longer: the new memory is added, and
           the old one's validity ends at the new one's date (``invalid_at``,
           ``superseded_by``), superseded as an update.
- WRONG    the memory was never true (a correction): superseded as a
           contradiction.

A memory superseded as an update stays retrievable as history
(``models.HISTORY_KINDS``): search returns it after the memory that replaced
it, and the answer context shows until when it held ("[until 2026-04-13]",
``context.until_note``); hiding such memories lost questions about the past.
One superseded as a contradiction leaves search, as before.

SAME, MORE, CHANGED and WRONG act only at or above the decision provider's
bar for that answer (``bar_for``, ``Decider.reconcile_bars``). Below it a SAME
or a MORE is stored as NEW; a doubtful CHANGED or WRONG keeps both memories
in use, the new one marked as a conflict for a person to settle under Upkeep.
CHANGED, WRONG and a MORE with no merged text also wait for a person where
much is at stake (``held_back``): a memory rated important, or stated in
several separate saves.

The judge is the decision provider (``ACTION_QUESTION``), or, where it
abstains, the text model (``RECONCILE_SYSTEM``), whose answers carry no
confidence: they act, within ``held_back``'s protections. Both see each memory
with the date it was said and the new fact with the save's: without dates a
second trip cannot be told from a changed plan. An exact duplicate is SAME
with no model asked, unless it is an event said on another day
(``another_occurrence``). Without an LLM, that is all reconciliation does.
"""

from __future__ import annotations

import re
from datetime import timezone
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
    new_id,
    parse_ts,
    utcnow,
)
from ..providers.embeddings import Embedder
from ..providers.llm import LLM
from ..providers.decisions import Choice, Decider
from .extraction import parse_lenient_json

#: The five answers, in the order the questions list them.
ACTIONS: tuple[str, ...] = ("NEW", "SAME", "MORE", "CHANGED", "WRONG")

RECONCILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "target": {"type": ["integer", "null"]},
        "content": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "target", "content", "reason"],
    "additionalProperties": False,
}

RECONCILE_SYSTEM = """You maintain an AI assistant's long-term memory store.
Given a NEW fact and the most similar EXISTING memories, each with the date it
was said, decide what the NEW fact is:

- "NEW": new information that no existing memory says. This includes another
  occurrence of something that happens repeatedly (another class, trip, visit
  or meal, on a later date). It is stored beside the existing memories.
- "SAME": one existing memory already says everything the new fact says, in
  the same or other words. Set "target" to it. Nothing new is stored.
- "MORE": the new fact adds detail to one existing memory, which is still true
  as it stands. Set "target" to it and "content" to one self-contained text
  that says everything both say. It replaces that memory.
- "CHANGED": one existing memory was true when it was said, but is no longer
  true now (moved, changed jobs, a new price or count, a plan that happened or
  changed, a status or preference that moved on). Set "target" to it. It is
  kept as history, and the new fact is stored.
- "WRONG": the new fact says one existing memory was never true (a
  correction). Set "target" to it. It is retracted, and the new fact stored.

When writing MORE content, the text must preserve EVERY concrete detail from
both texts - numbers, dates, prices, names, versions, file formats, tool names,
constraints and their reasons. Never drop a detail to make it shorter.
Respond with JSON only:
{"action": "NEW"|"SAME"|"MORE"|"CHANGED"|"WRONG", "target": int|null,
 "content": str|null, "reason": short str}"""

_WS_RE = re.compile(r"\W+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


ACTION_QUESTION = Choice(
    instructions=("Each EXISTING memory was said on the date shown; the NEW fact was just "
                  "said, on its date. What is the NEW fact, compared with the existing "
                  "memories?"),
    criteria={
        "NEW": ("New information that no existing memory says. This includes another "
                "occurrence of something that happens repeatedly, such as another class, "
                "trip, visit or meal on a later date."),
        "SAME": ("One existing memory already says everything the new fact says, in the same "
                 "or other words. The new fact adds nothing to it."),
        "MORE": ("It adds detail to one existing memory, and that memory is still true as it "
                 "stands."),
        "CHANGED": ("One existing memory was true when it was said, but the new fact says it "
                    "is no longer true now: a place, job, price, amount, plan, status or "
                    "preference has changed since."),
        "WRONG": "It corrects one existing memory: that memory was never true.",
    },
)

TARGET_INSTRUCTIONS = ("Which existing memory does the NEW fact restate, add to, change or "
                       "correct? If it is new information, the most similar one.")


def said_on(stamp: str | None) -> str | None:
    """A time as the reconcile state writes it ("2 March 2026"), None when it
    does not parse."""
    if not stamp:
        return None
    try:
        moment = parse_ts(stamp)
    except (TypeError, ValueError):
        return None
    return f"{moment.day} {moment.strftime('%B')} {moment.year}"


def reconcile_state(memories: list[Memory], new: str, new_said: str | None) -> str:
    """What the judge reads: each memory with the date it was said, the new
    fact with the save's date."""
    lines = []
    for i, memory in enumerate(memories):
        said = said_on(memory.created_at)
        lines.append(f"[{i}] " + (f"(said {said}) " if said else "") + memory.content)
    when = said_on(new_said)
    return ("EXISTING memories:\n" + "\n".join(lines)
            + f"\n\nNEW fact{f' (said {when})' if when else ''}:\n{new}")


def _decide_action(
    decider: Decider, state: str, count: int
) -> dict[str, Any] | None:
    """Which answer, and about which memory, as two typed questions in one call.

    Only the answer and the target come from here. Writing the merged text of
    a MORE is a writing task and stays with the text model, so this is a
    cheaper call in front of a rarer expensive one rather than a replacement.
    Returns None when the provider abstains, leaving the text model in charge.
    """
    if not decider.available or count <= 0:
        return None
    questions: dict[str, Choice] = {"action": ACTION_QUESTION}
    if count > 1:
        questions["target"] = Choice(
            instructions=TARGET_INSTRUCTIONS,
            criteria={str(i): f"memory [{i}]" for i in range(count)},
        )
    answers = decider.decide(state, questions)
    action = answers["action"]
    if not action.available or action.value not in ACTIONS:
        return None
    if count == 1:
        target: int | None = 0
    else:
        chosen = answers["target"]
        target = int(chosen.value) if chosen.available else None
    return {
        "action": action.value,
        "target": target,
        "content": None,            # a MORE still needs its text written
        "reason": f"{decider.name}: {action.value} at {action.confidence:.2f}",
        "confidence": action.confidence,
        "probabilities": action.probabilities,
    }


#: Appended to the reconcile prompt when a decision provider has already
#: answered MORE: the text model only writes the merged text.
MERGE_REQUEST = ('The answer is decided: MORE of memory [0]. Reply with action "MORE", '
                 "target 0 and, as content, the merged text.")


def write_merged(llm: LLM, existing: str, new: str) -> str | None:
    """The merged text of a MORE a decision provider chose, written by the
    text model with the prompt the no-provider path uses. None when there is
    no text model, it failed, or it wrote nothing."""
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


#: Metadata key on a memory that was kept beside the one it would replace.
CONFLICT_KEY = "conflict"

#: How the SUPERSEDE event of a MORE nobody could write a merged text for
#: begins. The newer memory adds to the old one; it does not contradict it, so
#: the Archive's undo brings the old one back beside it (``MemoryStore.
#: undo_replacement``) instead of forgetting it.
UPDATE_SUPERSEDE_REASON = "updated by new information, with no merged text written"

#: How the ADD event of a MORE's merged text begins.
MERGED_REASON = "merges memory"

#: The SUPERSEDE kind (``models.SUPERSEDE_KINDS``) each answer that takes a
#: memory out of use records: a changed value and a merged detail update the
#: memory (it stays retrievable as history); a correction contradicts it.
SUPERSEDE_KIND = {"MORE": "update", "CHANGED": "update", "WRONG": "contradiction"}

#: What the new memory does to the old one, as the events and Upkeep say it.
_VERB = {"update": "updates", "contradiction": "contradicts"}

#: Metadata a merged text does not take over from the memory it replaces:
#: markers of that memory's own state.
_NOT_CARRIED = (CONFLICT_KEY, "pending_distillation", "_enrichment")


def bar_for(decider: Decider | None, action: str, cfg: SupersedeConfig) -> float:
    """The confidence from which ``action`` acts: the decision provider's
    measured bar for it (``Decider.reconcile_bars``), or ``cfg.confidence``
    for a provider nobody measured."""
    bars = getattr(decider, "reconcile_bars", None) or {}
    return float(bars.get(action, cfg.confidence))


def _save_time(stamp: str) -> str:
    try:
        return parse_ts(stamp).astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return stamp


def saves_of(backend: MemoryBackend, memory: Memory) -> int:
    """How many separate saves stated a memory: the saves behind its evidence,
    the episodes it was made from and those of every save recorded on it as
    saying it again (SAME). A save is its run and its time, which every
    message of one save shares (``MemoryStore.add``), so a save of a whole
    conversation counts once however many of its messages a memory rests
    on. An episode the backend cannot find counts as a save of its own."""
    ids = list(dict.fromkeys(memory.source_episode_ids or []))
    if not ids:
        return 0
    try:
        found = backend.episodes_by_id(ids)
    except Exception:
        found = {}
    saves = {(e.run_id, e.agent_id, _save_time(e.created_at)) for e in found.values()}
    return len(saves) + sum(1 for i in ids if i not in found)


def held_back(
    target: Memory, decision: dict[str, Any], cfg: SupersedeConfig,
    *, bar: float | None = None, saves: int | None = None,
) -> str | None:
    """Why this replacement has to be asked about, or None when it may go ahead.

    A change or a contradiction is one model's reading of one text. That is
    enough to retire a passing remark, and a wrong call there is undone from
    the Archive. It is not enough to retire what the store has the most
    reason to believe: a memory rated important, or one that several separate
    saves have stated (``saves``; the source episodes when not given). Nor
    does a typed judgement under its ``bar`` (``cfg.confidence`` when not
    given) act on its own.
    """
    if target.importance >= cfg.protect_importance:
        return (
            f"the memory it would replace is rated important "
            f"({target.importance:.2f})"
        )
    sources = saves if saves is not None else len(target.source_episode_ids or [])
    if sources >= cfg.protect_sources:
        return f"the memory it would replace was stated in {sources} separate saves"
    confidence = decision.get("confidence")
    limit = cfg.confidence if bar is None else bar
    if isinstance(confidence, (int, float)) and confidence < limit:
        return f"the judgement was only {confidence:.2f} sure"
    return None


def _day(stamp: str | None) -> str:
    try:
        return parse_ts(str(stamp)).date().isoformat()
    except (TypeError, ValueError):
        return str(stamp or "")[:10]


def another_occurrence(memory: Memory, candidate: CandidateFact, said_at: str) -> bool:
    """Whether two texts that read the same may still be two things: events
    (an episodic memory, or one with an occurrence time) whose occurrence
    times differ, or, without both, that were said on different days. "I went
    to a yoga class this morning" said a week apart is two classes."""
    old_when = (memory.metadata or {}).get("when")
    new_when = (candidate.metadata or {}).get("when")
    if old_when and new_when:
        return old_when != new_when
    event = ("episodic" in (memory.memory_type, candidate.memory_type)
             or bool(old_when or new_when))
    return event and _day(memory.created_at) != _day(said_at)


def _judge(
    candidate: CandidateFact, similar: list[SearchResult], decider: Decider | None,
    llm: LLM, said_at: str,
) -> dict[str, Any]:
    """The decision provider's answer, or the text model's where it abstains,
    or NEW."""
    state = reconcile_state([r.memory for r in similar], candidate.content, said_at)
    judged = _decide_action(decider, state, len(similar)) if decider else None
    if judged is not None:
        return judged
    if llm.available:
        parsed = parse_lenient_json(
            llm.complete(RECONCILE_SYSTEM, state, json_schema=RECONCILE_SCHEMA))
        if isinstance(parsed, dict) and parsed.get("action") in ACTIONS:
            target = parsed.get("target")
            if isinstance(target, str) and target.strip().isdigit():
                parsed["target"] = int(target)
            return parsed
    return {"action": "NEW", "target": None, "content": None, "reason": "new information"}


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

    ``similar`` holds current memories of the user, of any run: the answer
    acts on the one it names whatever its run, and a new memory goes to the
    save's scope (``scope``).

    ``created_at`` is the time of the save (``MemoryStore.add``): a new
    memory's ``created_at``, ``updated_at`` and ``valid_from``, the
    ``invalid_at`` of one it supersedes, and the time of the events recorded
    (the NONE event of one it restates). The clock when None.

    The AddAction's event names what happened: "ADD" (NEW, a SAME or MORE
    under its bar, or a replacement held for review, ``conflicts_with`` set),
    "NONE" (SAME: ``memory_id`` is the memory restated), "UPDATE" (MORE: the
    new merged memory), "SUPERSEDE" (CHANGED, or a MORE with no merged text),
    "DELETE" (WRONG)."""
    cfg = supersede_cfg or SupersedeConfig()
    said = created_at or utcnow()
    stamped: dict[str, Any] = {"created_at": created_at} if created_at else {}

    # An exact duplicate needs no model call, whatever its run: the fact said
    # again. An event said on another day may be another occurrence, and is
    # left to the judge, who sees the dates.
    norm = _normalize(candidate.content)
    for result in similar:
        memory = result.memory
        if _normalize(memory.content) == norm and not another_occurrence(memory, candidate, said):
            return _restated(backend, memory, candidate, scope, episode_ids, stamped,
                             "exact duplicate")

    decision = _judge(candidate, similar, decider, llm, said) if similar else {
        "action": "NEW", "target": None, "content": None, "reason": "new information"}
    action = str(decision.get("action") or "NEW")
    target_idx = decision.get("target")
    target: Memory | None = None
    if isinstance(target_idx, int) and 0 <= target_idx < len(similar):
        target = similar[target_idx].memory
    reason = str(decision.get("reason") or "")
    if action not in ACTIONS or (action != "NEW" and target is None):
        action, target = "NEW", None  # malformed decision -> safest fallback

    bar = bar_for(decider, action, cfg) if action != "NEW" else None
    confidence = decision.get("confidence")
    doubtful = bar is not None and isinstance(confidence, (int, float)) and confidence < bar
    if action in ("SAME", "MORE") and doubtful and target is not None:
        reason = (f"{action} of memory {target.id} at {confidence:.2f}, under its bar of "
                  f"{bar:.2f}: added as new. {reason}").strip()
        action, target = "NEW", None

    if action == "SAME" and target is not None:
        return _restated(backend, target, candidate, scope, episode_ids, stamped, reason)

    if action == "MORE" and target is not None:
        merged = str(decision.get("content") or "").strip()
        if not merged:
            merged = write_merged(llm, target.content, candidate.content) or ""
        if merged:
            return _merged(backend, embedder, target, candidate, merged, scope, episode_ids,
                           said, stamped, reason, prepare_update)
        # Nothing wrote the merged text. Overwriting the old memory with the
        # new fact alone would lose what only the old one said, so the new
        # memory supersedes it instead, held back as a change would be.

    kind = SUPERSEDE_KIND.get(action) if target is not None else None
    held = (held_back(target, decision, cfg, bar=bar, saves=saves_of(backend, target))
            if kind and target is not None else None)
    metadata = dict(candidate.metadata or {})
    if held and target is not None and kind:
        metadata[CONFLICT_KEY] = {
            "with": target.id,
            "reason": reason or f"{_VERB[kind]} a memory",
            "held": held,
            "at": utcnow(),
            # absent for a contradiction, as it always was
            **({"kind": kind} if kind != "contradiction" else {}),
        }
    new_memory = Memory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        # a MORE with no merged text stands for both
        importance=(max(target.importance, candidate.importance)
                    if action == "MORE" and target is not None else candidate.importance),
        categories=candidate.categories,
        entities=candidate.entities,
        metadata=metadata,
        source_episode_ids=episode_ids,
        created_at=said,
        updated_at=said,
    )
    stored = _insert(backend, embedder, new_memory)
    backend.add_event(
        MemoryEvent(
            memory_id=stored.id,
            event="ADD",
            new_content=stored.content,
            reason=(
                f"kept beside memory {target.id}, which it {_VERB[kind]}, "
                f"because {held}. {reason}".strip()
                if held and target is not None and kind
                else f"{_VERB[kind]} memory {target.id}, which it supersedes. {reason}".strip()
                if kind and target is not None
                else reason or "new information"
            ),
            **stamped,
        )
    )

    if held and target is not None and kind:
        return AddAction(
            event="ADD",
            memory_id=stored.id,
            content=stored.content,
            reason=f"{_VERB[kind]} memory {target.id}; waiting for review ({held})",
            conflicts_with=target.id,
        )

    if kind and target is not None:
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
                    if action == "MORE"
                    else f"no longer true: {reason}".strip() if action == "CHANGED"
                    else reason or "contradicted by new information"
                ),
                kind=kind,
                **stamped,
            )
        )
        return AddAction(
            event="DELETE" if kind == "contradiction" else "SUPERSEDE",
            memory_id=stored.id,
            content=stored.content,
            reason=reason or f"superseded memory {target.id}",
        )

    return AddAction(event="ADD", memory_id=stored.id, content=stored.content, reason=reason)


def _restated(
    backend: MemoryBackend, target: Memory, candidate: CandidateFact, scope: Scope,
    episode_ids: list[str], stamped: dict[str, Any], reason: str,
) -> AddAction:
    """SAME: the save is recorded on the memory it restates as evidence, and
    nothing new is stored. Its episodes join the memory's sources: they carry
    the save's run, so a search restricted to that run finds the memory, and
    the latest of them says when it was last said, as the NONE event recorded
    at the save's time does. Its ``updated_at`` stays: the memory did not
    change."""
    backend.update_memory(
        target.id,
        source_episode_ids=list(dict.fromkeys([*target.source_episode_ids, *episode_ids])),
        touch=False,
    )
    where = f" in run {scope.run_id}" if scope.run_id and scope.run_id != target.run_id else ""
    backend.add_event(
        MemoryEvent(
            memory_id=target.id,
            event="NONE",
            new_content=candidate.content,
            reason=f"said again{where}. {reason}".strip(),
            **stamped,
        )
    )
    return AddAction(event="NONE", memory_id=target.id, content=target.content,
                     reason=reason or "already known")


def _merged(
    backend: MemoryBackend, embedder: Embedder, target: Memory, candidate: CandidateFact,
    merged: str, scope: Scope, episode_ids: list[str], said: str, stamped: dict[str, Any],
    reason: str, prepare_update: Callable[[str, str], dict[str, Any]] | None,
) -> AddAction:
    """MORE: the merged text is stored as a new memory dated at the save, in
    the save's scope, and supersedes the old one as an update. It keeps the
    old one's tags and sources (whose episodes keep it in the old one's run
    for a search restricted to that run) and the higher importance. Its names
    are read from the merged text as an edit of the old memory
    (``prepare_update``), so a name the old memory named keeps its entity and
    nothing is compared for it; that runs before anything is written, and the
    mentions it returns are attached to the new memory as a save attaches
    them."""
    prepared = prepare_update(target.id, merged) if prepare_update else {}
    metadata = {key: value for key, value in (target.metadata or {}).items()
                if key not in _NOT_CARRIED}
    metadata.update(candidate.metadata or {})
    memory = Memory(
        content=merged,
        memory_type=target.memory_type,
        user_id=scope.user_id,
        agent_id=scope.agent_id,
        run_id=scope.run_id,
        importance=max(target.importance, candidate.importance),
        categories=list(dict.fromkeys([*target.categories, *candidate.categories])),
        entities=list(prepared.get("entities") or candidate.entities),
        metadata=metadata,
        source_episode_ids=list(dict.fromkeys([*target.source_episode_ids, *episode_ids])),
        created_at=said,
        updated_at=said,
    )
    stored = _insert(backend, embedder, memory)
    # attached as a save attaches them, beside the mentions its tags made
    for mention in prepared.get("mentions") or []:
        backend.add_mention(mention.model_copy(update={"id": new_id(), "memory_id": stored.id}))
    backend.add_event(MemoryEvent(
        memory_id=stored.id, event="ADD", new_content=merged,
        reason=f"{MERGED_REASON} {target.id} with a detail it adds. {reason}".strip(), **stamped))
    backend.invalidate_memory(target.id, superseded_by=stored.id, at=stamped.get("created_at"))
    backend.add_event(MemoryEvent(
        memory_id=target.id, event="SUPERSEDE", old_content=target.content, new_content=merged,
        reason=f"merged with a detail it did not have into memory {stored.id}. {reason}".strip(),
        kind="update", **stamped))
    return AddAction(event="UPDATE", memory_id=stored.id, content=merged, reason=reason)


def _insert(backend: MemoryBackend, embedder: Embedder, memory: Memory) -> Memory:
    embedding = _embed_or_none(embedder, memory.content)
    if embedding:
        memory.embedding_model = embedder.model_id
    return backend.insert_memory(memory, embedding)


def _embed_or_none(embedder: Embedder, text: str) -> list[float] | None:
    if not embedder.dimensions:
        return None
    try:
        vectors = embedder.embed([text])
        return vectors[0] if vectors and vectors[0] else None
    except Exception:
        return None
