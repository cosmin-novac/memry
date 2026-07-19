"""Context reconstruction: turn search results into a token-budgeted block an
agent can drop straight into its prompt. This is the read-side counterpart of
extraction - selecting *which* memories fit the budget, most valuable first.
"""

from __future__ import annotations

from ..models import ContextResult, SearchResult

_HEADER = "## Relevant long-term memories (memry)\n"
_FOOTER = "\n(Use these silently as background knowledge; they may be incomplete.)"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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
        date = (memory.updated_at or memory.created_at)[:10]
        line = f"- [{memory.memory_type} · {date}] {memory.content}"
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
