"""Entity resolution with evidence-based disambiguation.

A shared short name is never enough to merge. An exact multi-part name plus
meaningful contextual overlap is treated as the same identity unless known types
conflict or the model finds a concrete contradiction. Otherwise an identity
judgment runs per candidate:

- ``same`` (confident)  -> the mention attaches to the existing entity
- ``unsure``            -> a new entity is created and a merge proposal is
                           recorded for later automatic or human resolution
- ``different``         -> a new entity, no proposal

Without an LLM, deterministic full-name-and-context matches still collapse; less
certain matches stay separate and recoverable. ``resolve_open_proposals`` follows
prior merge chains and auto-confirms only deterministic or high-confidence matches.
"""

from __future__ import annotations

import re
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

- "same": clearly the same entity (matching full name plus compatible context,
  or strongly consistent roles, relationships, or identifiers)
- "different": clearly a different entity (concrete conflicting ages, locations,
  relationships, identifiers, or types)
- "unsure": the name matches but the evidence is insufficient either way

A person can have several jobs, hobbies, purchases, projects, or public roles. Different
activities are not a contradiction. In a personal memory store, an exact first-and-last
name in overlapping context strongly favors "same" unless concrete evidence conflicts.
Do not demand a public profile or unique identifier when the stored context already aligns.
Be conservative when only a short/common name matches.
Respond with JSON only:
{"verdict": "same"|"unsure"|"different", "confidence": 0..1, "reason": short}"""

AUTO_CONFIRM_CONFIDENCE = 0.9

DESCRIPTION_MAX_CHARS = 1200
DESCRIPTION_MAX_WORDS = 300
DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
    "additionalProperties": False,
}
DESCRIPTION_SYSTEM = """Write a compact, evidence-grounded description of one entity
for a long-term memory system. Use only the supplied active memories. Preserve
concrete dates, numbers, preferences, constraints, and negations. If evidence
conflicts, state the conflict instead of choosing a side. Do not infer missing
facts. Aim for 100-300 tokens. Respond with JSON only: {"description": string}."""


def _bound_description(value: str) -> str:
    text = " ".join(value.split()).strip()
    words = text.split()
    if len(words) > DESCRIPTION_MAX_WORDS:
        text = " ".join(words[:DESCRIPTION_MAX_WORDS])
    if len(text) > DESCRIPTION_MAX_CHARS:
        text = text[: DESCRIPTION_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return text


def synthesize_entity_description(
    llm: LLM,
    entity: Entity,
    facts: list[str],
    aliases: list[str] | None = None,
) -> str:
    """Build a bounded cache from active evidence; degrade to a factual excerpt."""
    clean_facts = [" ".join(fact.split()).strip() for fact in facts if fact.strip()]
    if not clean_facts:
        return ""
    fallback = _bound_description(" ".join(clean_facts[:6]))
    if not llm.available:
        return fallback
    aliases = aliases or [entity.name]
    evidence = "\n".join(f"- {fact}" for fact in clean_facts[:40])
    prompt = (
        f"Entity: {entity.name}\n"
        f"Type: {entity.entity_type or 'unknown'}\n"
        f"Aliases: {', '.join(aliases[:20])}\n\n"
        f"Active evidence:\n{evidence}"
    )
    try:
        raw = llm.complete(DESCRIPTION_SYSTEM, prompt, json_schema=DESCRIPTION_SCHEMA)
        parsed = parse_lenient_json(raw)
        if isinstance(parsed, dict):
            description = parsed.get("description")
            if isinstance(description, str) and description.strip():
                return _bound_description(description)
    except Exception:
        pass
    return fallback


_NAME_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CONTEXT_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "been", "before",
    "being", "but", "can", "does", "existing", "fact", "for", "from",
    "had", "has", "have", "into", "its", "new", "not", "person", "same",
    "that", "the", "their", "them", "then", "they", "this", "user", "was",
    "were", "with", "work", "works", "would",
}


def _name_words(value: str) -> tuple[str, ...]:
    return tuple(_NAME_WORD_RE.findall(value.casefold()))


def _context_stem(token: str) -> str:
    if len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 3 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 5 and token.endswith("ied"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
        token = token[:-1]
    return token


def _context_words(value: str, name_words: tuple[str, ...]) -> set[str]:
    return {
        stem
        for raw in _NAME_WORD_RE.findall(value.casefold())
        if raw not in name_words and raw not in _CONTEXT_STOPWORDS and len(raw) >= 3
        for stem in [_context_stem(raw)]
        if len(stem) >= 3 and stem not in _CONTEXT_STOPWORDS
    }


def _same_name_and_no_evidence(
    existing: Entity,
    existing_facts: list[str],
    other_name: str,
    other_type: str | None = None,
) -> bool:
    """Exact name match against an entity that carries no evidence at all.

    Such a record has no identity to differ from: there are no facts, no
    description, nothing that could belong to a *different* thing of the same
    name. Forking a second entity there is strictly worse than reusing it - it
    fragments the graph and emits a merge proposal that nobody can adjudicate,
    because the question it asks ("are these the same?") has no evidence on
    either side. Real stores fill up with exactly that: 4x "sehr geehrte",
    3x "the father of photography", 206 of 519 entities with zero mentions.

    The usual worry, two different people sharing a common name, needs the
    existing record to actually say something about the first person. Once this
    reuse attaches evidence, later mentions go through the normal judged path.
    """
    if existing.normalized != (other_name or "").strip().lower():
        return False
    if existing.entity_type and other_type and existing.entity_type != other_type:
        return False
    return not existing_facts and not (existing.description or "").strip()


def _obvious_same_entity(
    existing: Entity,
    existing_facts: list[str],
    other_name: str,
    other_facts: list[str],
    other_type: str | None = None,
) -> bool:
    """Deterministic high-confidence identity match.

    Exact multi-part names are not enough by themselves. They become an automatic
    match when the two evidence sets also share meaningful context and their known
    types do not conflict. This catches repeated first+last-name memories without
    conflating unrelated people who happen to share a common full name.
    """
    existing_name = _name_words(existing.name)
    if len(existing_name) < 2 or existing_name != _name_words(other_name):
        return False
    if existing.entity_type and other_type and existing.entity_type != other_type:
        return False
    left = _context_words(" ".join(existing_facts), existing_name)
    right = _context_words(" ".join(other_facts), existing_name)
    if not left or not right:
        return False
    shared = left & right
    if len(shared) >= 2:
        return True
    return bool(shared) and max(map(len, shared)) >= 6 and (
        len(shared) / min(len(left), len(right)) >= 0.12
    )

def _judge(
    llm: LLM, existing: Entity, existing_facts: list[str], new_fact: str, surface: str
) -> dict[str, Any]:
    if not llm.available:
        return {"verdict": "unsure", "confidence": 0.5, "reason": "no LLM: same name only"}
    facts = "\n".join(f"- {f}" for f in existing_facts) or "- (no facts recorded)"
    description = existing.description or "(no synthesized description yet)"
    raw = llm.complete(
        IDENTITY_SYSTEM,
        f'EXISTING entity "{existing.name}"\nDescription: {description}\n'
        f'Recent evidence:\n{facts}\n\n'
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


_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["types"],
    "additionalProperties": False,
}

_TYPE_SYSTEM = """Classify each entity name into one type: person, organization,
project, product, place, event, document, code, concept, or other.

- document: a contract, invoice, certificate, form, report, or the reference
  number that identifies one ("HRB 110232", "TÜV Kaufvertrag")
- code: a file, function, table, endpoint, or config key ("lib/sync.ts",
  "canUserSync", "BILDY_AWS_S3_BUCKET")

Use "other" only when none fit.
JSON only: {"types": [{"name": str, "type": str}]}."""


def classify_entity_types(llm: LLM, names: list[str]) -> dict[str, str]:
    """One call classifies a whole batch of entity names -> type. Cheap: many
    entities per call, used to backfill entities that were linked before typing."""
    from .extraction import ENTITY_TYPES, parse_lenient_json

    if not names:
        return {}
    raw = llm.complete(
        _TYPE_SYSTEM,
        "Entities:\n" + "\n".join(f"- {n}" for n in names) + "\n\nTypes as JSON.",
        json_schema=_TYPE_SCHEMA,
    )
    data = parse_lenient_json(raw)
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for item in data.get("types", []):
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip().lower()
                etype = str(item.get("type", "")).strip().lower()
                if name and etype in ENTITY_TYPES:
                    out[name] = etype
    return out


_TEMPORAL_RE = re.compile(
    r"^(?:(?:19|20)\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?)?"      # 2019, 2026-07-24
    r"|(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december|januar|februar|märz|april|mai|juni|juli|august"
    r"|september|oktober|november|dezember)\s+(?:19|20)\d{2}"    # July 2026
    r"|(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2})$",               # 2026-2029
    re.IGNORECASE,
)
_QUANTITY_UNITS = {
    "gb", "tb", "mb", "kb", "ram", "gpu", "cpu", "ghz", "mhz", "cores",
    "eur", "usd", "kg", "km", "mg", "ml", "kpa", "kwh", "kw", "watt", "%",
}
_URL_EMAIL_RE = re.compile(r"://|^www\.|^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SALUTATIONS = (
    "sehr geehrte", "dear sir", "dear madam", "dear sir or madam",
    "mit freundlichen grüßen", "best regards", "kind regards",
)


def non_referent_reason(name: str) -> str | None:
    """Why a name mechanically cannot be an entity, or ``None`` if it might be.

    An entity is a named referent you could later ask a question about. A bare
    date, an amount, a URL, a template placeholder or a salutation is an
    attribute VALUE or a text fragment - real stores fill with them ("2019",
    "$149", "192 GB", "Sehr geehrte ..."), and no accumulation of evidence will
    ever make "2027" a thing with an identity. Only mechanically certain cases
    belong here; anything needing judgement goes to the LLM review instead.
    """
    text = " ".join((name or "").split()).strip().lower()
    if not text:
        return "empty name"
    if _TEMPORAL_RE.match(text):
        return "a date or time span, not a referent"
    if _URL_EMAIL_RE.search(text):
        return "a URL or email address"
    if re.fullmatch(r"[\[\](){}<>._\-\s]*|\[.*\]", text):
        return "a placeholder or punctuation fragment"
    if any(text.startswith(s) for s in _SALUTATIONS):
        return "a salutation, not a referent"
    words = text.replace("/", " ").split()
    if words and re.match(r"^[^a-zäöüß]*\d", words[0]) and all(
        re.fullmatch(r"[\d\W]+", w) or w in _QUANTITY_UNITS for w in words
    ):
        return "an amount or measurement, not a referent"
    return None


_REFERENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"junk": {"type": "array", "items": {"type": "string"}}},
    "required": ["junk"],
    "additionalProperties": False,
}

_REFERENT_SYSTEM = """You review the entity list of a personal memory system.

An entity must be a NAMED, REFERRING thing the user could later ask a question
about: a person, organization, place, named project, product, standard, method,
or defined domain term (tax rules, index names, technologies all qualify).

List as junk ONLY names that are clearly not referents:
- instructions or style preferences ("avoid parentheses", "casual but not
  choppy tone", "confirm understanding before drafting")
- descriptions of a writing task or its output ("corrected version",
  "2-3 improved versions", "shorter answers", "grammar feedback list")
- sentence fragments, generic role words ("note", "article", "assistant",
  "value", "jobs"), or epithets standing in for an unnamed thing
  ("the father of photography")

Keep every real-world term, even niche ones. When unsure, keep it: deleting an
entity loses its links, keeping a mediocre one costs nothing.
Return JSON: {"junk": ["name", ...]}"""


def judge_entity_referents(llm: LLM, names: list[str]) -> list[str]:
    """One call reviews a batch of names; returns the ones judged not to be
    entities. Names outside the input are discarded, so a model cannot condemn
    what it was not shown."""
    if not names or not llm.available:
        return []
    raw = llm.complete(
        _REFERENT_SYSTEM,
        "Entity names:\n" + "\n".join(f"- {n}" for n in names) + "\n\nJSON.",
        json_schema=_REFERENT_SCHEMA,
    )
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return []
    offered = {n.strip().lower(): n for n in names}
    return [offered[j] for j in
            (str(x).strip().lower() for x in data.get("junk", []))
            if j in offered]


def resolve_mentions(
    *,
    backend: MemoryBackend,
    llm: LLM,
    scope: Scope,
    memory_id: str,
    memory_content: str,
    surfaces: list[str],
    types: dict[str, str] | None = None,
    attach: bool = True,
) -> dict[str, Entity]:
    """Attach a memory's entity mentions, creating/reusing entities per the
    conservative policy. Returns a map of normalized surface -> entity, so the
    caller can resolve relation triples to the entities they linked to. Pass
    ``attach=False`` when the caller will replace all mentions atomically."""
    types = types or {}
    resolved: dict[str, Entity] = {}
    for surface in surfaces:
        surface = surface.strip()
        normalized = surface.lower()
        if not normalized or normalized in resolved:
            continue

        candidates = backend.find_entity_candidates(normalized, scope)
        target: Entity | None = None
        proposals: list[tuple[Entity, dict[str, Any]]] = []
        for candidate in candidates:
            facts = [m.content for m in backend.entity_memories(candidate.id, limit=5)]
            # An identically-named record with no evidence at all cannot be a
            # different thing. Reuse it before spending an LLM call on a
            # question that has nothing to answer with.
            if _same_name_and_no_evidence(
                candidate, facts, surface, types.get(normalized)
            ):
                target = candidate
                break
            judgment = _judge(llm, candidate, facts, memory_content, surface)
            high_conflict = (
                judgment["verdict"] == "different"
                and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE
            )
            if (
                judgment["verdict"] == "same"
                and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE
            ) or (
                not high_conflict
                and _obvious_same_entity(
                    candidate,
                    facts,
                    surface,
                    [memory_content],
                    types.get(normalized),
                )
            ):
                target = candidate
                break
            if judgment["verdict"] in ("same", "unsure"):
                proposals.append((candidate, judgment))

        if target is None:
            target = backend.insert_entity(
                Entity(
                    name=surface,
                    normalized=normalized,
                    entity_type=types.get(normalized),
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

        if attach:
            backend.add_mention(
                EntityMention(entity_id=target.id, memory_id=memory_id, surface=surface)
            )
        resolved[normalized] = target
    return resolved


def propose_same_name_duplicates(
    *, backend: MemoryBackend, scope: Scope, limit: int = 50
) -> int:
    """Raise proposals for active entities that share a normalized name.

    Proposals are otherwise only ever created at write time, so duplicates that
    predate a fix - or whose judgement once came back "different" - sit in the
    graph forever with nothing scheduled to look at them again. This gives
    maintenance a way to reconsider them as evidence accumulates.
    """
    groups: dict[str, list[Entity]] = {}
    for entity in backend.list_entities(scope, limit=10_000):
        if entity.merged_into is None:
            groups.setdefault(entity.normalized or entity.name.lower(), []).append(entity)
    created = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            if created >= limit:
                return created
            if backend.find_proposal(anchor.id, other.id) is None:
                backend.add_proposal(
                    MergeProposal(
                        entity_a=anchor.id,
                        entity_b=other.id,
                        user_id=scope.user_id,
                        confidence=0.5,
                        reason="same name, not yet compared",
                    )
                )
                created += 1
    return created


def resolve_open_proposals(
    *,
    backend: MemoryBackend,
    llm: LLM,
    scope: Scope,
    auto_confirm: bool = True,
) -> dict[str, int]:
    """Resolve obvious/stale proposals and re-judge the remaining pairs.

    Entity IDs are first followed through merge history, making maintenance safe
    for proposals created before another merge changed either endpoint.
    """
    outcome = {"confirmed": 0, "rejected": 0, "kept": 0}
    for proposal in backend.list_proposals(scope, status="proposed", limit=1000):
        entity_a_id = backend.resolve_entity_id(proposal.entity_a)
        entity_b_id = backend.resolve_entity_id(proposal.entity_b)
        if entity_a_id is None or entity_b_id is None:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
            continue
        if entity_a_id == entity_b_id:
            backend.set_proposal_status(proposal.id, "confirmed")
            outcome["confirmed"] += 1
            continue
        entity_a = backend.get_entity(entity_a_id)
        entity_b = backend.get_entity(entity_b_id)
        if entity_a is None or entity_b is None:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
            continue
        facts_a = [m.content for m in backend.entity_memories(entity_a.id, limit=8)]
        facts_b = [m.content for m in backend.entity_memories(entity_b.id, limit=8)]
        # Same name, and one side carries no evidence: nothing distinguishes
        # them and no reviewer could. Settle it instead of asking again forever.
        if auto_confirm and (
            _same_name_and_no_evidence(entity_a, facts_a, entity_b.name,
                                       entity_b.entity_type)
            or _same_name_and_no_evidence(entity_b, facts_b, entity_a.name,
                                          entity_a.entity_type)
        ):
            # Keep the record that actually has evidence attached to it.
            keep, drop = ((entity_a, entity_b) if facts_a or not facts_b
                          else (entity_b, entity_a))
            if backend.merge_entities(keep.id, drop.id):
                backend.set_proposal_status(proposal.id, "confirmed")
                outcome["confirmed"] += 1
                continue
        judgment = _judge(
            llm,
            entity_a,
            facts_a,
            " / ".join(facts_b) or f"(entity named {entity_b.name}, no facts)",
            entity_b.name,
        )
        high_conflict = (
            judgment["verdict"] == "different"
            and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE
        )
        obvious = _obvious_same_entity(
            entity_a,
            facts_a,
            entity_b.name,
            facts_b,
            entity_b.entity_type,
        )
        should_merge = auto_confirm and (
            (
                judgment["verdict"] == "same"
                and judgment["confidence"] >= AUTO_CONFIRM_CONFIDENCE
            )
            or (obvious and not high_conflict)
        )
        if should_merge and backend.merge_entities(entity_a.id, entity_b.id):
            backend.set_proposal_status(proposal.id, "confirmed")
            outcome["confirmed"] += 1
        elif high_conflict:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
        else:
            outcome["kept"] += 1
    return outcome
