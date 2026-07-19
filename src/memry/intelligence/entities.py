"""Entity resolution with conservative disambiguation.

The rule (set by design, not tunable prompts): **same name is never enough to
merge.** When a new memory mentions "Jonas" and one or more entities named
Jonas already exist, an identity judgment runs per candidate:

- ``same`` (confident)  -> the mention attaches to the existing entity
- ``unsure``            -> a NEW entity is created and a merge *proposal* is
                           recorded for later - automatic confirmation when the
                           evidence becomes clear, or a human decision
- ``different``         -> a new entity, no proposal

Without an LLM the safe default applies: keep separate + propose. Wrongly kept
apart is recoverable (merge later); wrongly merged is corrupted memory.

``resolve_open_proposals`` re-judges proposed pairs as evidence accumulates and
auto-confirms only clear, high-confidence matches.
"""

from __future__ import annotations

from typing import Any

from ..backends.base import MemoryBackend
from ..models import Entity, EntityMention, MergeProposal, Scope, utcnow
from ..providers.llm import LLM
from .extraction import parse_lenient_json

IDENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "unsure", "different"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

IDENTITY_SYSTEM = """You resolve entity identity for a long-term memory system.
You are given an EXISTING entity (with facts that mention it) and a NEW fact
that mentions the same name. Decide whether they refer to the same real-world
person/organization/place/thing.

- "same": clearly the same entity (strongly consistent roles, relationships,
  or context)
- "different": clearly a different entity (conflicting roles, relationships,
  ages, locations, or types)
- "unsure": the name matches but the evidence is insufficient either way

Be conservative: prefer "unsure" over "same" unless the evidence is strong.
A wrong merge corrupts memory; a deferred merge is harmless.
Respond with JSON only:
{"verdict": "same"|"unsure"|"different", "confidence": 0..1, "reason": short}"""

AUTO_CONFIRM_CONFIDENCE = 0.9


def _judge(
    llm: LLM, existing: Entity, existing_facts: list[str], new_fact: str, surface: str
) -> dict[str, Any]:
    if not llm.available:
        return {"verdict": "unsure", "confidence": 0.5, "reason": "no LLM: same name only"}
    facts = "\n".join(f"- {f}" for f in existing_facts) or "- (no facts recorded)"
    raw = llm.complete(
        IDENTITY_SYSTEM,
        f'EXISTING entity "{existing.name}", known facts:\n{facts}\n\n'
        f'NEW fact mentioning "{surface}":\n- {new_fact}',
        json_schema=IDENTITY_SCHEMA,
    )
    parsed = parse_lenient_json(raw)
    if isinstance(parsed, dict) and parsed.get("verdict") in ("same", "unsure", "different"):
        try:
            parsed["confidence"] = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            parsed["confidence"] = 0.5
        return parsed
    return {"verdict": "unsure", "confidence": 0.5, "reason": "unparseable judgment"}


def resolve_mentions(
    *,
    backend: MemoryBackend,
    llm: LLM,
    scope: Scope,
    memory_id: str,
    memory_content: str,
    surfaces: list[str],
) -> list[Entity]:
    """Attach a memory's entity mentions, creating/reusing entities per the
    conservative policy. Returns the entities the mentions attached to."""
    attached: list[Entity] = []
    seen: set[str] = set()
    for surface in surfaces:
        surface = surface.strip()
        normalized = surface.lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        candidates = backend.find_entities(normalized, scope)
        target: Entity | None = None
        proposals: list[tuple[Entity, dict[str, Any]]] = []
        for candidate in candidates:
            facts = [m.content for m in backend.entity_memories(candidate.id, limit=3)]
            judgment = _judge(llm, candidate, facts, memory_content, surface)
            if judgment["verdict"] == "same" and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE:
                target = candidate
                break
            if judgment["verdict"] in ("same", "unsure"):
                proposals.append((candidate, judgment))

        if target is None:
            target = backend.insert_entity(
                Entity(
                    name=surface,
                    normalized=normalized,
                    user_id=scope.user_id,
                    agent_id=scope.agent_id,
                    run_id=scope.run_id,
                )
            )
            for candidate, judgment in proposals:
                if backend.find_proposal(candidate.id, target.id) is None:
                    backend.add_proposal(
                        MergeProposal(
                            entity_a=candidate.id,
                            entity_b=target.id,
                            user_id=scope.user_id,
                            confidence=judgment["confidence"],
                            reason=judgment.get("reason"),
                        )
                    )

        backend.add_mention(
            EntityMention(entity_id=target.id, memory_id=memory_id, surface=surface)
        )
        attached.append(target)
    return attached


def resolve_open_proposals(
    *,
    backend: MemoryBackend,
    llm: LLM,
    scope: Scope,
    auto_confirm: bool = True,
) -> dict[str, int]:
    """Re-judge open proposals with the evidence accumulated since. Only clear,
    high-confidence "same" verdicts auto-merge; everything else stays for the
    user. Returns counts per outcome."""
    outcome = {"confirmed": 0, "rejected": 0, "kept": 0}
    for proposal in backend.list_proposals(scope, status="proposed", limit=1000):
        entity_a = backend.get_entity(proposal.entity_a)
        entity_b = backend.get_entity(proposal.entity_b)
        if entity_a is None or entity_b is None or not entity_a.is_active or not entity_b.is_active:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
            continue
        if not llm.available:
            outcome["kept"] += 1
            continue
        facts_a = [m.content for m in backend.entity_memories(entity_a.id, limit=5)]
        facts_b = [m.content for m in backend.entity_memories(entity_b.id, limit=5)]
        judgment = _judge(
            llm, entity_a, facts_a,
            " / ".join(facts_b) or f"(entity named {entity_b.name}, no facts)",
            entity_b.name,
        )
        if (
            auto_confirm
            and judgment["verdict"] == "same"
            and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE
        ):
            backend.merge_entities(entity_a.id, entity_b.id)
            backend.set_proposal_status(proposal.id, "confirmed")
            outcome["confirmed"] += 1
        elif judgment["verdict"] == "different" and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
        else:
            outcome["kept"] += 1
    return outcome
