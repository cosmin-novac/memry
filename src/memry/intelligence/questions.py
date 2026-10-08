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

#: Metadata mark on a memory the backfill asked about and got no question
#: for, so it is not asked again (``MemoryStore.backfill_questions``).
QUESTIONS_CHECKED_KEY = "questions_checked"

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


#: The backfill's prompt: the same rule as the extractor's, for memories
#: saved before there were question keys, many in one call.
BACKFILL_SYSTEM = """You write the questions that saved memories answer, so that a
question asked later in another conversation finds the memory that answers it.

For each numbered memory, write 2 or 3 questions it answers, as the person
would ask them later. Each question stands on its own: it names the thing or
person it asks about (never "it", "he" or "they" alone) and asks for one thing
the memory says. At least one uses other words than the memory does. Give []
for a memory that answers no question anyone would ask.

Answer as JSON: {"items": [{"n": <the memory's number>, "questions": [str]}]},
one item per memory."""

#: The backfill's answer: the questions of each memory, by its number.
BACKFILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "questions": QUESTIONS_FIELD,
                },
                "required": ["n", "questions"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


def write_questions(llm: Any, memories: list[str]) -> list[list[str]]:
    """The questions each of ``memories`` answers, asked in one call: one list
    per memory, in order, cleaned as every source is, [] where the model gave
    none or an answer that cannot be read. The memories are shown numbered
    from 1, and an item whose number names no memory is ignored. A failing
    call raises, so the caller can tell a batch it could not ask from one the
    model found nothing in."""
    out: list[list[str]] = [[] for _ in memories]
    if not memories:
        return out
    # Imported here: extraction.py imports this module at its top, so a
    # top-level import in this direction would be a cycle.
    from .extraction import parse_lenient_json

    listing = "\n".join(
        f"{n}. {' '.join(str(text or '').split())}" for n, text in enumerate(memories, 1))
    raw = llm.complete(
        BACKFILL_SYSTEM,
        f"Memories:\n{listing}\n\nAnswer for every number as JSON.",
        json_schema=BACKFILL_SCHEMA,
    )
    parsed = parse_lenient_json(raw)
    items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(memories):
            continue
        out[n - 1] = clean_questions(item.get("questions"), limit=QUESTIONS_PER_MEMORY)
    return out
