"""Forgetting: importance decays with time since last touch.

``effective_importance`` never hard-deletes anything - a decay *sweep*
invalidates memories whose decayed importance falls below a threshold
(soft-forget: they leave retrieval but remain in the audit trail and can be
inspected or restored). Inspired by Recall's STRONG→MEDIUM→WEAK GC and by
what Mem0 ships only in its managed platform.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..backends.base import MemoryBackend
from ..config import DecayConfig
from ..models import Memory, MemoryEvent, parse_ts
from ..providers.decisions import Decider, Score


#: Half-life multipliers for the three durability levels. A passing detail
#: fades in a fifth of the default; something stable about a person lasts four
#: times as long.
DURABILITY_FACTORS = (0.2, 1.0, 4.0)
DURABILITY_KEY = "durability"


def durability_factor(memory: Memory) -> float | None:
    """The recorded durability as a half-life multiplier, or None if absent.

    Stored as a 0-2 score, so a value between levels interpolates between the
    multipliers rather than snapping to one.
    """
    raw = (memory.metadata or {}).get(DURABILITY_KEY)
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    top = len(DURABILITY_FACTORS) - 1
    score = min(max(score, 0.0), float(top))
    low = int(score)
    if low >= top:
        return DURABILITY_FACTORS[top]
    return DURABILITY_FACTORS[low] + (score - low) * (
        DURABILITY_FACTORS[low + 1] - DURABILITY_FACTORS[low]
    )


def effective_importance(
    memory: Memory, cfg: DecayConfig, now: datetime | None = None
) -> float:
    if not cfg.enabled:
        return memory.importance
    now = now or datetime.now(timezone.utc)
    try:
        age_days = max(0.0, (now - parse_ts(memory.updated_at)).total_seconds() / 86400.0)
    except ValueError:
        return memory.importance
    # Per-fact durability when one was recorded, type otherwise. The type is a
    # crude proxy: "the train was delayed" and "allergic to penicillin" are both
    # semantic and fade at the same rate, which is wrong for both of them.
    factor = durability_factor(memory)
    if factor is None:
        # type-aware half-life: episodic events fade faster, procedural rules persist
        factor = cfg.half_life_by_type.get(memory.memory_type, 1.0)
    half_life = max(cfg.half_life_days * factor, 0.01)
    decay = math.pow(0.5, age_days / half_life)
    return memory.importance * (cfg.floor + (1.0 - cfg.floor) * decay)


def decay_sweep(
    backend: MemoryBackend,
    cfg: DecayConfig,
    *,
    threshold: float = 0.1,
    now: datetime | None = None,
) -> list[str]:
    """Invalidate active memories whose decayed importance dropped below
    ``threshold``. Returns the invalidated memory ids."""
    if not cfg.enabled:
        return []
    now = now or datetime.now(timezone.utc)
    forgotten: list[str] = []
    for memory in backend.all_memories_iter(include_invalid=False):
        score = effective_importance(memory, cfg, now)
        if score < threshold:
            backend.invalidate_memory(memory.id)
            backend.add_event(
                MemoryEvent(
                    memory_id=memory.id,
                    event="DELETE",
                    old_content=memory.content,
                    reason=f"decay sweep (effective importance {score:.3f} < {threshold})",
                    actor="decay",
                )
            )
            forgotten.append(memory.id)
    return forgotten


DURABILITY_QUESTION_LEVELS = [
    "Days. A passing detail that stops mattering almost immediately.",
    "Months. Relevant for a while, then stale.",
    "Years. A stable fact about the person, their work, their health or their "
    "relationships.",
]


def score_durability(decider: Decider, contents: list[str]) -> dict[int, float]:
    """How long each fact stays worth keeping, all in one call.

    Returns {index: 0-2 score}, leaving out anything the provider declined to
    answer so the caller falls back to the per-type half-life for those.
    """
    if not decider.available or not contents:
        return {}
    questions = {
        f"d{i}": Score(
            instructions=f'How long will this stay worth remembering? "{text}"',
            levels=DURABILITY_QUESTION_LEVELS,
        )
        for i, text in enumerate(contents)
    }
    answers = decider.decide(
        "Facts stored in a personal long-term memory store.", questions
    )
    out: dict[int, float] = {}
    for i in range(len(contents)):
        answer = answers[f"d{i}"]
        if answer.available:
            out[i] = float(answer.value)
    return out
