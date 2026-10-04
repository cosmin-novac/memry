"""Importance over time, and how long each fact stays worth keeping.

Nothing forgets by decay now. The forgetting sweep that invalidated memories
whose decayed importance fell below a threshold (``memry sweep``) was retired
in 0.2.44: a fact must never leave search for its age alone, and relevance is
to be measured per entity first. ``effective_importance`` and
``durability_factor`` stay as library functions that nothing in the product
calls; ``score_durability`` is the durability pass, which records an estimate
per fact that nothing acts on yet.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..config import DecayConfig
from ..models import Memory, parse_ts
from ..providers.decisions import Decider, Score


#: Half-life multipliers for the three durability levels. A passing detail
#: fades in a fifth of the default; something stable about a person lasts four
#: times as long.
DURABILITY_FACTORS = (0.2, 1.0, 4.0)
DURABILITY_KEY = "durability"


def durability_factor(memory: Memory) -> float | None:
    """The recorded durability as a half-life multiplier, or None if absent.
    A library function: nothing forgets by decay now.

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
    """``memory``'s importance decayed by its age on the half-life of its
    durability, else of its type. A library function: nothing forgets or
    ranks by it now."""
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


DURABILITY_QUESTION_LEVELS = [
    "Days. A passing detail that stops mattering almost immediately.",
    "Months. Relevant for a while, then stale.",
    "Years. A stable fact about the person, their work, their health or their "
    "relationships.",
]


def score_durability(decider: Decider, contents: list[str]) -> dict[int, float]:
    """How long each fact stays worth keeping, all in one call.

    Returns {index: 0-2 score}, leaving out anything the provider declined to
    answer.
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
