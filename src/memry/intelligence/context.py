"""Context reconstruction: turn search results into a token-budgeted block an
agent can drop straight into its prompt. This is the read-side counterpart of
extraction - selecting *which* memories fit the budget, most valuable first.

It is the one place that renders memories for a model (``memory_lines``):
each fact with the date its event happens, when known, and the date it was
said, both labelled, and for a memory kept as history the day it held until,
then the source turns shown as their evidence (``MemoryStore.evidence``), each
with the date it was said and its speaker.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..models import ContextResult, EvidenceTurn, Memory, SearchResult, parse_ts
from .when import describe_when

_HEADER = "## Relevant long-term memories (memry)\n"
_EVIDENCE_HEADER = "\n\nWhat was said, in the order it was said:\n"
_FOOTER = "\n(Use these silently as background knowledge; they may be incomplete.)"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def said_date(stamp: str | None) -> str:
    """A stored time as the day it names, "8 May 2023"."""
    try:
        moment = parse_ts(str(stamp))
    except (TypeError, ValueError):
        return str(stamp or "")[:10]
    return f"{moment.day} {moment:%B %Y}"


def said_at(memory: Memory) -> str:
    """When a memory was said, as stored: when it was recorded (its last
    change). For a memory out of use, such as one kept as history
    (``models.HISTORY_KINDS``), when it began to hold (``valid_from``, else
    ``created_at``): taking it out of use moved its ``updated_at`` to the day
    it ended, which ``until_note`` says."""
    if memory.invalid_at:
        return memory.valid_from or memory.created_at
    return memory.updated_at or memory.created_at


def until_note(memory: Memory) -> str:
    """The end of a memory's validity as a model reads it after the date it
    was said ("[until 15 July 2023]"): a memory kept as history held until
    then (``models.HISTORY_KINDS``), the day the memory that replaced it was
    said. Empty for a memory in use."""
    return f"[until {said_date(memory.invalid_at)}]" if memory.invalid_at else ""


def memory_line(memory: Memory, now: Any = None) -> str:
    """One memory as a model reads it: "[happened 2023-05-07] <text> (said 8
    May 2023)". The date it was said is when it was recorded (``said_at``),
    which is not when the thing it tells happened: that is written only where
    the memory knows it (``metadata["when"]``). Without both labels a model
    reads a birthday or a dated plan as something that happened on the day it
    was written down. A memory kept as history ends in the day it held until,
    "(said 8 May 2023) [until 15 July 2023]" (``until_note``), so it is not
    read as current."""
    occurs = describe_when((memory.metadata or {}).get("when"), now)
    until = until_note(memory)
    return (f"{f'[{occurs}] ' if occurs else ''}{memory.content} "
            f"(said {said_date(said_at(memory))}){f' {until}' if until else ''}")


def turn_line(turn: EvidenceTurn) -> str:
    """One source turn as a model reads it: "8 May 2023: Ada: <text>"."""
    return f"{said_date(turn.said_at)}: {turn.speaker}: {turn.content}"


def memory_lines(
    memories: Iterable[Memory], evidence: Iterable[EvidenceTurn] = (), now: Any = None
) -> list[str]:
    """The memories, then the turns shown as their evidence, one line each."""
    return [*(memory_line(m, now) for m in memories), *(turn_line(t) for t in evidence)]


def fitting(results: Sequence[SearchResult], token_budget: int) -> list[SearchResult]:
    """The results that fit ``token_budget`` as ``build_context`` packs them:
    in rank order, stopping at the first that does not fit (a first result
    too long on its own is skipped)."""
    kept: list[SearchResult] = []
    used = estimate_tokens(_HEADER) + estimate_tokens(_FOOTER)
    for result in results:
        cost = estimate_tokens(memory_line(result.memory)) + 1
        if used + cost > token_budget and kept:
            break
        if used + cost > token_budget:
            continue  # single over-budget item: skip, try the next
        kept.append(result)
        used += cost
    return kept


def build_context(
    results: list[SearchResult],
    *,
    token_budget: int = 1200,
    evidence: Sequence[EvidenceTurn] = (),
) -> ContextResult:
    """The memories that fit, then ``evidence``, the turns already chosen for
    them within their own budget (``MemoryStore.evidence``)."""
    turns_cost = (estimate_tokens(_EVIDENCE_HEADER)
                  + sum(estimate_tokens(turn_line(t)) + 1 for t in evidence)) if evidence else 0
    shown = fitting(results, token_budget - turns_cost)
    if not shown:
        return ContextResult(text="", memory_ids=[], token_estimate=0)
    kept = {r.memory.id for r in shown}
    turns = [t for t in evidence if kept.intersection(t.memory_ids)]
    lines = memory_lines([r.memory for r in shown])
    text = _HEADER + "\n".join(f"- {line}" for line in lines)
    if turns:
        text += _EVIDENCE_HEADER + "\n".join(f"- {turn_line(t)}" for t in turns)
    text += _FOOTER
    return ContextResult(text=text, memory_ids=[r.memory.id for r in shown],
                         token_estimate=estimate_tokens(text),
                         episode_ids=[t.episode_id for t in turns])
