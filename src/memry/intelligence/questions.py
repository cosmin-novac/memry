"""Question keys: the questions a memory answers, written when the memory is
and kept beside it as extra search keys (``retrieval.question_keys``).

A memory is a fact in the third person ("Ada moved to Amsterdam in March
2024"); a question asked later shares few words with it ("what city is Ada
based in?"). On LoCoMo, three questions written per record and indexed
beside its text raised the share of questions whose answer reaches the top
8 by 11 to 15 points for every writing model, and by 7 to 17 as extra
vectors next to the record's own (PhD notes, dreaming-questions, E196 to
E198). Appended, never in place of the text: questions alone pay only with
the strongest writer.

Here: the rule the extractor is given, the shape of the answer, and the
cleaning every source of questions goes through (extraction, the backfill,
an agent's own). The store writes them (``MemoryStore._write_questions``),
the backend keeps them (``memory_questions``) and the search reads them as
two more candidate lists (``retrieval.hybrid_search``).
"""

from __future__ import annotations

from typing import Any

#: Questions kept per memory from one source, and in all.
QUESTIONS_PER_MEMORY = 3
QUESTIONS_LIMIT = 9

#: What the extractor is asked, placed among its rules. One question in
#: other words than the fact is where the gain is: the questions sharing
#: few words with the record gained most (C3 in the PhD notes).
QUESTIONS_RULE = """- questions: 2 or 3 questions this fact answers, as the person would ask them
  later in another conversation. Each stands on its own: it names the thing
  or person it asks about (never "it", "he" or "they" alone) and asks for one
  thing the fact says. At least one uses other words than the fact does. []
  when the fact answers no question anyone would ask.
"""

#: The answer field, a list of strings.
QUESTIONS_FIELD: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

_MAX_CHARS = 200


def clean_questions(raw: Any, limit: int = QUESTIONS_LIMIT) -> list[str]:
    """The questions of a model's or an agent's answer as the store keeps
    them: strings only, stripped, on one line, at most ``_MAX_CHARS`` each,
    each once (ignoring case), at most ``limit``. Anything else is read as
    none given."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if not text or len(text) > _MAX_CHARS:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def merge_questions(*lists: list[str], limit: int = QUESTIONS_LIMIT) -> list[str]:
    """The questions of several lists, first list first, each once (ignoring
    case), at most ``limit``: a merged text answers what both its parts
    answered."""
    merged: list[str] = []
    for items in lists:
        merged.extend(items)
    return clean_questions(merged, limit=limit)
