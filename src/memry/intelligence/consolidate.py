"""Consolidation: collapse memories that say the same thing.

Write-time reconciliation only ever compares a new fact against the handful of
memories similar to it at that moment. Anything that arrives in a different
session, or just outside that candidate window, stays duplicated for good. Over
a long life a store accumulates families like:

    "User is Marcus Vandenberg"
    "The user's name is Marc."
    "User is Marcus Vandenberg (goes by Marc)."

They are not textual duplicates, so exact matching misses them, and each one is
individually true, so nothing contradicts. They simply crowd retrieval: three of
the ten result slots spent on one fact.

The decision is deliberately split in two, so that similarity alone can never
destroy information:

- **grouping is geometric** - connected components over the vectors already
  stored, above a similarity floor. No model involved, reproducible, free.
- **merging is judged** - one LLM call per group decides whether the group is
  genuinely one fact, and if so writes the single memory that preserves every
  detail from all of them. Without an LLM only exact textual duplicates
  collapse, because guessing here would silently lose information.

Nothing is deleted. The merged text becomes a NEW memory and every original is
invalidated with ``superseded_by`` pointing at it, so history still resolves.
Rewriting one of the originals in place would leave an arbitrary member wearing
the merged text along with its own creation date and history; a fresh record
says plainly that this came from consolidating several.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Memory
from .extraction import parse_lenient_json

CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_fact": {"type": "boolean"},
        "content": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["same_fact", "content", "reason"],
    "additionalProperties": False,
}

CONSOLIDATE_SYSTEM = """You are consolidating an AI assistant's long-term memory.

You are given several stored memories that look nearly identical. Decide whether
they are all recording THE SAME underlying fact about the user.

Set same_fact=true only when every memory is about the same single fact, so that
keeping one merged version loses nothing. Then write that merged version:

- preserve EVERY specific detail from every memory: names, nicknames, numbers,
  dates, amounts, model or product identifiers, and any stated qualifier;
- prefer the fullest phrasing. "User is Marcus Vandenberg" plus "The user's
  name is Marc." becomes "User is Marcus Vandenberg (goes by Marc).";
- write one self-contained sentence, resolving pronouns;
- never invent anything that is not in the inputs.

Set same_fact=false when they are DIFFERENT facts that merely share wording -
different people, different events, different dates, different measurements, or
a general rule beside a specific instance. When in doubt, false: splitting is
recoverable, merging is not.

Return JSON: {"same_fact": bool, "content": str, "reason": str}"""


def similarity_groups(
    vectors: list[tuple[str, "np.ndarray"]],
    *,
    threshold: float,
    max_group: int = 6,
) -> list[list[str]]:
    """Connected components of the "closer than threshold" graph.

    Components, not pairs: the example family above is a chain where the first
    and last are less alike than either is to the middle, and pairwise handling
    would collapse it in two passes instead of one coherent merge.

    ``max_group`` stops a dense region from being swallowed whole - beyond a
    handful of memories, "all mutually similar" stops meaning "all one fact".
    """
    if len(vectors) < 2:
        return []
    ids = [mid for mid, _ in vectors]
    matrix = np.array([np.asarray(vec, dtype=float) for _, vec in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    similarity = matrix @ matrix.T
    rows, cols = np.where(np.triu(similarity, k=1) >= threshold)
    for i, j in zip(rows.tolist(), cols.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    grouped: dict[int, list[str]] = {}
    for index, mid in enumerate(ids):
        grouped.setdefault(find(index), []).append(mid)
    return [g for g in grouped.values() if 2 <= len(g) <= max_group]


def representative(memories: list[Memory]) -> Memory:
    """The member whose wording best stands for the group.

    Used when every member says the same thing textually, so there is nothing
    for a model to merge and one of them can supply the text directly. Highest
    importance first, then the most detailed, then the oldest.
    """
    return sorted(
        memories,
        key=lambda m: (-m.importance, -len(m.content or ""), m.created_at),
    )[0]


def judge_group(llm, memories: list[Memory]) -> dict[str, Any]:
    """Ask whether the group is one fact, and get the merged text if so."""
    listing = "\n".join(f"[{i}] {m.content}" for i, m in enumerate(memories))
    try:
        raw = llm.complete(
            CONSOLIDATE_SYSTEM,
            f"Stored memories:\n{listing}\n\nDecide and return JSON.",
            json_schema=CONSOLIDATE_SCHEMA,
        )
    except Exception as exc:  # a provider hiccup must never mutate the store
        return {"same_fact": False, "content": "", "reason": f"llm error: {exc}"}
    parsed = parse_lenient_json(raw)
    if not isinstance(parsed, dict) or not parsed.get("same_fact"):
        return {
            "same_fact": False,
            "content": "",
            "reason": str((parsed or {}).get("reason") or "not the same fact"),
        }
    content = str(parsed.get("content") or "").strip()
    if not content:
        return {"same_fact": False, "content": "", "reason": "empty merged content"}
    return {"same_fact": True, "content": content,
            "reason": str(parsed.get("reason") or "same fact")}
