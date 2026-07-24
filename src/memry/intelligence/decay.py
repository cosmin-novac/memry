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
