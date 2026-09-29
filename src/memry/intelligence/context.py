"""Context reconstruction: turn search results into a token-budgeted block an
agent can drop straight into its prompt. This is the read-side counterpart of
extraction - selecting *which* memories fit the budget, most valuable first.
"""

from __future__ import annotations

from ..models import ContextResult, Memory, SearchResult
from .when import describe_when

_HEADER = "## Relevant long-term memories (memry)\n"
_FOOTER = "\n(Use these silently as background knowledge; they may be incomplete.)"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def until_note(memory: Memory) -> str:
    """The end of a memory's validity as the answer context shows it after
    its text ("[until 2026-04-13]"): a memory kept as history held until
    then (``models.HISTORY_KINDS``). Empty for a memory in use."""
    return f"[until {memory.invalid_at[:10]}]" if memory.invalid_at else ""


def build_context(
    results: list[SearchResult],
    *,
    token_budget: int = 1200,
) -> ContextResult:
    if not results:
        return ContextResult(text="", memory_ids=[], token_estimate=0)

    lines: list[str] = []
    ids: list[str] = []
    used = estimate_tokens(_HEADER) + estimate_tokens(_FOOTER)
    for result in results:
        memory = result.memory
        # The bracketed date says when this was recorded; for a memory kept
        # as history, from when it held (its update moved ``updated_at``), and
        # the note after the text until when. When the fact itself happens at
        # a time, say so too: without it an agent reads a stored birthday or a
        # dated plan as something that was merely written down.
        date = ((memory.valid_from or memory.created_at) if memory.invalid_at
                else (memory.updated_at or memory.created_at))[:10]
        occurs = describe_when((memory.metadata or {}).get("when"))
        line = f"- [{memory.memory_type} · {date}] {memory.content}"
        if occurs:
            line += f" ({occurs})"
        if memory.invalid_at:
            line += f" {until_note(memory)}"
        cost = estimate_tokens(line) + 1
        if used + cost > token_budget and lines:
            break
        if used + cost > token_budget:
            continue  # single over-budget item: skip, try the next
        lines.append(line)
        ids.append(memory.id)
        used += cost

    if not lines:
        return ContextResult(text="", memory_ids=[], token_estimate=0)
    text = _HEADER + "\n".join(lines) + _FOOTER
    return ContextResult(text=text, memory_ids=ids, token_estimate=estimate_tokens(text))
