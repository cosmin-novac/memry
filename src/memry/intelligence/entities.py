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
from typing import Any, Callable

from ..backends.base import MemoryBackend
from ..models import Entity, EntityMention, Memory, MergeProposal, Scope, utcnow
from ..providers.decisions import (
    MEASURED_MERGE_GATES,
    Answer,
    Choice,
    Decider,
    merge_gate_for,
)
from ..providers.llm import LLM
from .extraction import parse_lenient_json
from .identity import (
    NAME_CHECKS_PER_PASS,
    Mention,
    NameIndex,
    closest_people,
    compare,
    judges_pairs,
    merge_pair,
    name_vectors,
    pair_reason,
    parallel,
    worth_comparing,
)

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

# Measured over 56 labelled identity cases: at 0.9 a text model merges two
# entities that should stay apart, including a partner and a vendor architect
# who share a first name - the exact confusion the entity handling exists to
# prevent, waved through at 0.85. Its wrong answers score as high as its right
# ones, so the only threshold that lets nothing through is 0.95. Fewer merges
# happen without asking; the ones that do are the ones that should.
#: That 0.95 is gpt-5-mini's number. Other text models get their own from
#: ``MEASURED_MERGE_GATES``, and one nobody has measured never merges on its own.
AUTO_CONFIRM_CONFIDENCE = MEASURED_MERGE_GATES["gpt-5-mini"]


def _gate(decider: Decider | None, llm: LLM | None = None) -> float:
    """How confident a "same" has to be before it merges without asking.

    Each provider carries its own, because the number only means something
    relative to how that provider's confidence is distributed. When the
    provider cannot answer, the text model is reporting on itself, and that
    gate depends on which text model it is.
    """
    if decider is not None and decider.available:
        return decider.auto_confirm_confidence
    if decider is not None:
        return decider.fallback_gate
    return merge_gate_for(getattr(llm, "model", None))


def _conflict_bar(judgment: dict[str, Any]) -> float:
    """How confident a "different" has to be to block an obvious-looking merge.

    Never higher than 0.95: a model whose "same" may not merge on its own can
    still veto one, otherwise raising its gate would make merging *easier*.
    """
    return min(judgment.get("gate", AUTO_CONFIRM_CONFIDENCE), AUTO_CONFIRM_CONFIDENCE)

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


def _merges_on_gate(judgment: dict[str, Any]) -> bool:
    """Without a calibrated judge, only a "same" at the provider's gate merges."""
    return (
        judgment["verdict"] == "same"
        and judgment["confidence"] >= judgment.get("gate", AUTO_CONFIRM_CONFIDENCE)
    )


IDENTITY_QUESTION = Choice(
    instructions=(
        "Do the EXISTING entity and the NEW fact refer to the same real-world "
        "person, organization, place or thing?"
    ),
    criteria={
        "same": (
            "Clearly the same entity: a matching full name with compatible context, "
            "or strongly consistent roles, relationships or identifiers. Different "
            "jobs, hobbies or projects are not a contradiction."
        ),
        "different": (
            "Clearly a different entity: concretely conflicting ages, locations, "
            "relationships, identifiers or types."
        ),
        "unsure": "The name matches but the evidence does not settle it either way.",
    },
)


def _identity_state(
    existing: Entity, existing_facts: list[str], new_fact: str, surface: str
) -> str:
    facts = "\n".join(f"- {f}" for f in existing_facts) or "- (no facts recorded)"
    description = existing.description or "(no synthesized description yet)"
    return (
        f'EXISTING entity "{existing.name}"\nDescription: {description}\n'
        f"Recent evidence:\n{facts}\n\n"
        f'NEW fact mentioning "{surface}":\n- {new_fact}'
    )


def _judge_via_decider(
    decider: Decider,
    existing: Entity,
    existing_facts: list[str],
    new_fact: str,
    surface: str,
) -> dict[str, Any] | None:
    """Identity as a typed choice. Returns None when the provider abstained, so
    the caller can fall back rather than treat "no answer" as a verdict."""
    answers = decider.decide(
        _identity_state(existing, existing_facts, new_fact, surface),
        {"identity": IDENTITY_QUESTION},
    )
    answer: Answer = answers["identity"]
    if not answer.available:
        return None
    return {
        "verdict": answer.value,
        "confidence": answer.confidence,
        "reason": f"{decider.name}: {answer.value}",
        "gate": decider.auto_confirm_confidence,
        "probabilities": answer.probabilities,
    }


def _judge(
    llm: LLM,
    existing: Entity,
    existing_facts: list[str],
    new_fact: str,
    surface: str,
    decider: Decider | None = None,
) -> dict[str, Any]:
    if decider is not None and decider.available:
        judged = _judge_via_decider(decider, existing, existing_facts, new_fact, surface)
        if judged is not None:
            return judged
        # fall through: an abstaining or failing provider must not decide by default
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
        parsed["gate"] = _gate(decider, llm)
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


TYPE_CRITERIA = {
    "person": "A human being.",
    "organization": "A company, team or institution.",
    "project": "A named piece of work.",
    "product": "A tool, service, library, brand or product.",
    "place": "A city, country, building or region.",
    "event": "Something that happens at a point in time.",
    "document": ("A contract, invoice, certificate, form, report, or the "
                 "reference number that identifies one."),
    "code": "A file, function, table, endpoint or config key.",
    "concept": "An abstract idea.",
    "other": "None of the others fit.",
}


def _classify_via_decider(decider: Decider, names: list[str]) -> dict[str, str] | None:
    """One question per name, all in one call.

    The name has to travel in the question rather than the state: identical
    questions over a state that does not name them get identical answers, at a
    confidence that looks fine.
    """
    questions = {
        f"n{i}": Choice(instructions=f'What kind of thing is "{name}"?',
                        criteria=TYPE_CRITERIA)
        for i, name in enumerate(names)
    }
    answers = decider.decide(
        "Entity names extracted from a personal long-term memory store.", questions
    )
    out: dict[str, str] = {}
    for i, name in enumerate(names):
        answer = answers[f"n{i}"]
        if answer.available and answer.value in TYPE_CRITERIA:
            out[name.strip().lower()] = answer.value
    return out or None


def classify_entity_types(
    llm: LLM, names: list[str], decider: Decider | None = None
) -> dict[str, str]:
    """One call classifies a whole batch of entity names -> type. Cheap: many
    entities per call, used to backfill entities that were linked before typing."""
    from .extraction import ENTITY_TYPES, parse_lenient_json

    if not names:
        return {}
    if decider is not None and decider.available:
        typed = _classify_via_decider(decider, names)
        if typed is not None:
            return typed
    if not llm.available:
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
# A number welded to a unit, a rate or a range of them: "250 ms", "34px",
# "$0.4304/s", "900-450 ms", "10^5". Matched against the original spelling,
# because single letters only count as units in lower case: "1.3m" is a
# length, "3M" is a company.
_MEASURE_UNIT = (
    r"(?:ms|s|sec|secs|min|mins|h|hrs?|px|pt|em|rem|mm|cm|m|km|kb|mb|gb|tb|"
    r"fps|hz|khz|mhz|ghz|k|x|%|tokens?|eur|usd|kg|g|mg|ml|l|kw|kwh|w|v)"
)
_MEASURE_RE = re.compile(
    r"^[~<>=≤≥]*[$€£]?\d[\d.,^]*\s?" + _MEASURE_UNIT + r"?"
    r"(?:\s?[-–—]\s?[$€£]?\d[\d.,]*\s?" + _MEASURE_UNIT + r"?)?"
    r"(?:/(?:s|sec|min|h|hr|day|mo|month|yr|year|" + _MEASURE_UNIT + r"))?$"
)
# A count of ordinary things: "22 tests", "40 opponents", "450 ms floor". The
# words after the number must all be lower case in the original, which is what
# keeps "50 Cent" and "7 Wonders" out.
_COUNT_RE = re.compile(r"^\d[\d.,^]*(?:\s?[-–]\s?\d[\d.,]*)?\s+[a-z][a-z/\- ]*$")
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
    original = " ".join((name or "").split()).strip()
    if _MEASURE_RE.match(original):
        return "an amount or measurement, not a referent"
    if _COUNT_RE.match(original):
        return "a count of things, not a referent"
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


#: What a name can be, asked of the decision provider for names the store has
#: never seen. Roles and values are not made entities; topics still are, and
#: earn hub status or not like anything else.
SCREEN_CRITERIA = {
    # The wording was measured. A first draft listed "path" among the values
    # and the provider dutifully screened out source files, repositories and
    # street addresses; naming them as things took the harm from 6 names in
    # 360 to none at every gate from 0.70 up.
    "named_thing": ("A specific person, organization, product, project, place, "
                    "street address, document, source file, folder, repository "
                    "path, API endpoint, named feature or other thing that has "
                    "a name or an identifier of its own, which one could later "
                    "ask questions about."),
    "generic_topic": ("An ordinary noun, activity or topic with no identity of "
                      "its own, such as billing, content, conversation or "
                      "syntax validation."),
    "role": ("A role or function that someone or something holds, such as "
             "creator, client, landlord or assistant."),
    "value_or_fragment": ("A measurement, number, price, amount, count, date, "
                          "duration, quoted sentence, user-interface message, "
                          "instruction, or a fragment of a sentence. Not a "
                          "file, address or endpoint."),
}
#: Verdicts that keep a name from becoming an entity.
SCREEN_SKIPS = frozenset({"role", "value_or_fragment"})
#: Probability the provider must put on a skip verdict before it is believed.
#: Measured in evals/entity_structure_benchmark.py; see docs/self-hosting.md.
#: Lives with the structure rules, which read the same number for hub status.
from .structure import SCREEN_GATE  # noqa: E402


def screen_names(
    decider: Decider | None, memory_content: str, names: list[str]
) -> dict[str, dict[str, Any]]:
    """Ask what each name is, in the memory it came from.

    Returns ``{normalized name: {"verdict", "probability"}}`` for the names the
    provider answered. The name travels in the question and the memory is the
    state, because a question that carries no information still gets a
    confident-looking answer. Never raises, never blocks a write: no provider
    or no answer means nothing is screened out.
    """
    if decider is None or not decider.available or not names:
        return {}
    questions = {
        f"s{i}": Choice(instructions=f'In this memory, what is "{name}"?',
                        criteria=SCREEN_CRITERIA)
        for i, name in enumerate(names)
    }
    try:
        answers = decider.decide(
            "A memory from a personal long-term memory store: " + memory_content,
            questions,
        )
    except Exception:  # a provider hiccup must never cost a write its entities
        return {}
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names):
        answer = answers[f"s{i}"]
        if answer.available and answer.value in SCREEN_CRITERIA:
            probability = (answer.probabilities or {}).get(answer.value, answer.confidence)
            out[name.strip().lower()] = {
                "verdict": answer.value, "probability": round(float(probability), 3)}
    return out


def screened_out(verdict: dict[str, Any] | None, gate: float = SCREEN_GATE) -> bool:
    return bool(
        verdict
        and verdict.get("verdict") in SCREEN_SKIPS
        and float(verdict.get("probability") or 0.0) >= gate
    )


def resolve_mentions(
    *,
    backend: MemoryBackend,
    llm: LLM,
    decider: Decider | None = None,
    scope: Scope,
    memory_id: str,
    memory_content: str,
    surfaces: list[str],
    types: dict[str, str] | None = None,
    attach: bool = True,
    owner: Entity | None = None,
) -> dict[str, Entity]:
    """Attach a memory's entity mentions, creating/reusing entities per the
    conservative policy. Returns a map of normalized surface -> entity, so the
    caller can resolve relation triples to the entities they linked to. Pass
    ``attach=False`` when the caller will replace all mentions atomically.

    ``owner`` is the store owner's entity. The extractor was told to list the
    owner under that entity's name, so that name attaches to it directly.

    With a calibrated judge (``identity.judges_pairs``) each candidate is
    decided by ``identity.compare``: merge, keep apart, or wait for evidence.
    Both kept-apart and waiting pairs are recorded with the funnel step they
    were compared at, so neither is compared again on the same evidence;
    nobody is asked. A name the store already has is the exception: the
    mention joins the likeliest entity of that name unless the judge says
    "different" at the apart bar. Without one, a "same" at the provider's gate merges and
    anything short of "different" is recorded for a person."""
    types = types or {}
    resolved: dict[str, Entity] = {}
    # Names are looked up across the person's whole namespace, as the weekly
    # pass does. Looked up within the save's run, the same name saved in two
    # sessions became two entities that were never compared.
    lookup = Scope(user_id=scope.user_id) if scope.user_id is not None else scope
    # A name the store has never seen is screened before it becomes an entity:
    # mechanically first (free, certain), then one typed question per name. A
    # name that already has an entity is left alone; upkeep reviews those.
    cleaned = [s.strip() for s in surfaces if s and s.strip()]
    owner_name = owner.name.strip().casefold() if owner is not None else None
    unseen = [
        s for s in dict.fromkeys(cleaned)
        if s.casefold() != owner_name
        and not non_referent_reason(s)
        and not backend.find_entity_candidates(s.lower(), lookup)
    ]
    verdicts = screen_names(decider, memory_content, unseen)
    # A calibrated judge also compares names that are not spelled the same
    # ("Fundation" and "Fundation GmbH"); without one, only exact names meet.
    judge = decider if judges_pairs(decider) else None
    index: NameIndex | None = None
    saved = backend.get_memory(memory_id) if judge is not None else None
    for surface in cleaned:
        normalized = surface.lower()
        if not normalized or normalized in resolved:
            continue
        if owner is not None and surface.casefold() == owner_name:
            if attach:
                backend.add_mention(
                    EntityMention(entity_id=owner.id, memory_id=memory_id, surface=surface)
                )
            resolved[normalized] = owner
            continue
        if non_referent_reason(surface) or screened_out(verdicts.get(normalized)):
            continue

        candidates = backend.find_entity_candidates(normalized, lookup)
        same_name = {c.id for c in candidates}
        if judge is not None:
            if index is None:
                index = NameIndex(backend.list_entities(lookup, limit=100_000))
            candidates += index.candidates(surface, exclude={c.id for c in candidates})
        target: Entity | None = None
        proposals: list[MergeProposal] = []
        mention = Mention(surface, types.get(normalized),
                          saved or Memory(id=memory_id, content=memory_content))
        # A name the store already has: the mention belongs to the likeliest
        # entity of that name unless the evidence says it is something else.
        # Attaching one memory can be undone; merging entities cannot. Holding
        # it to the merge bar instead left 88 of 431 mentions of a known name
        # as new one-memory entities in a replayed store, which never gained
        # the evidence to be compared again.
        likely: list[tuple[float, Entity]] = []
        for candidate in candidates:
            if likely and candidate.id not in same_name:
                break  # a known name found its entity; no need to try others
            facts = [m.content for m in backend.entity_memories(candidate.id, limit=5)]
            # An identically-named record with no evidence at all cannot be a
            # different thing. Reuse it before spending an LLM call on a
            # question that has nothing to answer with.
            if _same_name_and_no_evidence(
                candidate, facts, surface, types.get(normalized)
            ):
                target = candidate
                break
            if judge is not None:
                verdict = compare(judge, backend, candidate, mention)
                probabilities = verdict.probabilities
                if candidate.id in same_name and (
                    verdict.action == "merge"
                    or (probabilities is not None
                        and probabilities["different"] < judge.pair_apart_probability)
                ):
                    likely.append((probabilities["same"] if probabilities else 1.0, candidate))
                    continue
                if verdict.action == "merge":
                    target = candidate
                    break
                proposals.append(MergeProposal(
                    entity_a=candidate.id, entity_b="", user_id=scope.user_id,
                    confidence=probabilities["same"] if probabilities else 0.5,
                    reason=pair_reason(judge, probabilities) if probabilities else None,
                    status="rejected" if verdict.action == "apart" else "proposed",
                    decided_at=utcnow() if verdict.action == "apart" else None,
                    compared_step=verdict.step,
                ))
                continue
            judgment = _judge(
                llm, candidate, facts, memory_content, surface, decider
            )
            if _merges_on_gate(judgment):
                target = candidate
                break
            if judgment["verdict"] in ("same", "unsure"):
                proposals.append(MergeProposal(
                    entity_a=candidate.id, entity_b="", user_id=scope.user_id,
                    confidence=judgment["confidence"], reason=judgment.get("reason"),
                ))

        if target is None and likely:
            target = max(likely, key=lambda option: option[0])[1]
            proposals = []
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
            for proposal in proposals:
                if backend.find_proposal(proposal.entity_a, target.id) is None:
                    backend.add_proposal(proposal.model_copy(update={"entity_b": target.id}))

        if attach:
            backend.add_mention(
                EntityMention(entity_id=target.id, memory_id=memory_id, surface=surface)
            )
        resolved[normalized] = target
    return resolved


def propose_same_name_duplicates(
    *,
    backend: MemoryBackend,
    scope: Scope,
    limit: int = 50,
    decider: Decider | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    owner: Entity | None = None,
) -> int:
    """Raise pairs of existing entities for the identity judge to compare.

    Pairs are otherwise only raised at write time, so duplicates that predate a
    fix sit in the graph with nothing scheduled to look at them again. With a
    calibrated judge, every name is paired with the names worth comparing
    (``identity.NameIndex``: a rare shared word, a similar spelling, an
    acronym, or a name close in meaning when ``embed`` is a semantic embedder),
    and pairs already decided either way are not raised again. The store
    ``owner`` is also paired with the people whose memories are closest to its
    own (``identity.closest_people``), since its name may be no name. Without one,
    only identical names are paired, and a pair whose members live under
    different homes ("privacy policy" in two projects) is left alone, since no
    judge could answer it and no person should be asked.
    """
    entities = [e for e in backend.list_entities(scope, limit=10_000) if e.merged_into is None]
    pairs: list[tuple[Entity, Entity]] = []
    if judges_pairs(decider):
        index = NameIndex(entities, name_vectors(embed, entities))
        vectors = index.vectors
        looked: set[frozenset[str]] = set()
        checks = 0
        for entity in entities:
            found = index.candidates(entity.name, vector=vectors.get(entity.id),
                                     exclude={entity.id})
            for other in found:
                if entity.id < other.id:
                    pairs.append((entity, other))
            # Names that only share a word in capitals ("PR #42" and "the Dutch
            # address PR"): the judge looks at the two names first, and a pair
            # it rules out is recorded, so it is not looked at again.
            if checks >= NAME_CHECKS_PER_PASS:
                continue
            loose = [
                other for other in index.loose_candidates(
                    entity.name, exclude={entity.id, *(o.id for o in found)})
                if frozenset((entity.id, other.id)) not in looked
                and backend.find_proposal(entity.id, other.id) is None
            ]
            if not loose:
                continue
            checks += 1
            looked.update(frozenset((entity.id, other.id)) for other in loose)
            kept, ruled_out = worth_comparing(decider, entity, loose)
            pairs += [(entity, other) for other in kept]
            for other, different in ruled_out:
                backend.add_proposal(MergeProposal(
                    entity_a=entity.id, entity_b=other.id, user_id=scope.user_id,
                    confidence=round(1 - different, 3), status="rejected",
                    reason=f"the names alone rule it out: P(different) {different:.2f}",
                    decided_at=utcnow(),
                ))
        if owner is not None:
            pairs[:0] = [(owner, person)
                         for person in closest_people(backend, scope, owner, entities)]
    else:
        groups: dict[str, list[Entity]] = {}
        for entity in entities:
            groups.setdefault(entity.normalized or entity.name.lower(), []).append(entity)

        def home_of(entity: Entity) -> str | None:
            home = (entity.metadata or {}).get("home")
            return home.get("id") if isinstance(home, dict) else None

        for members in groups.values():
            for other in members[1:]:
                home_a, home_b = home_of(members[0]), home_of(other)
                if not (home_a and home_b and home_a != home_b):
                    pairs.append((members[0], other))
    created = 0
    for a, b in pairs:
        if created >= limit:
            break
        if backend.find_proposal(a.id, b.id) is None:
            backend.add_proposal(MergeProposal(
                entity_a=a.id, entity_b=b.id, user_id=scope.user_id,
                confidence=0.5, reason="not yet compared",
            ))
            created += 1
    return created


def resolve_open_proposals(
    *,
    backend: MemoryBackend,
    llm: LLM,
    decider: Decider | None = None,
    scope: Scope,
    auto_confirm: bool = True,
    proposal_ids: set[str] | None = None,
) -> dict[str, int]:
    """Resolve obvious/stale proposals and re-judge the remaining pairs.

    Entity IDs are first followed through merge history, making maintenance safe
    for proposals created before another merge changed either endpoint.
    ``proposal_ids`` limits the pass to those proposals, for the re-check a save
    runs when a new memory mentions one side of a pair.

    A pair that stays open keeps the latest answer, so the list shows how sure
    the provider is now, not how sure it was when the pair was first raised.
    With a calibrated judge every pair goes through ``identity.compare``, which
    asks nothing unless the pair has reached a new step of the funnel since it
    was last compared.
    """
    outcome = {"confirmed": 0, "rejected": 0, "kept": 0}
    judge = decider if judges_pairs(decider) else None
    pending: list[tuple[MergeProposal, Entity, Entity]] = []
    for proposal in backend.list_proposals(scope, status="proposed", limit=1000):
        if proposal_ids is not None and proposal.id not in proposal_ids:
            continue
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
        if judge is not None:
            pending.append((proposal, entity_a, entity_b))
            continue
        judgment = _judge(
            llm,
            entity_a,
            facts_a,
            " / ".join(facts_b) or f"(entity named {entity_b.name}, no facts)",
            entity_b.name,
            decider,
        )
        high_conflict = (
            judgment["verdict"] == "different"
            and judgment["confidence"] >= _conflict_bar(judgment)
        )
        if auto_confirm and _merges_on_gate(judgment) and merge_pair(
            backend, entity_a, entity_b
        ):
            backend.set_proposal_status(proposal.id, "confirmed")
            outcome["confirmed"] += 1
        elif high_conflict:
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
        else:
            backend.update_proposal_judgement(
                proposal.id,
                confidence=judgment["confidence"],
                reason=judgment.get("reason"),
            )
            outcome["kept"] += 1
    # The judge's calls are independent, so they run side by side; the store
    # is changed one pair at a time afterwards.
    decided = parallel(
        lambda item: compare(judge, backend, item[1], item[2], item[0].compared_step), pending
    )
    for (proposal, entity_a, entity_b), verdict in zip(pending, decided):
        if (verdict.action == "merge" and auto_confirm
                and merge_pair(backend, entity_a, entity_b)):
            backend.set_proposal_status(proposal.id, "confirmed")
            outcome["confirmed"] += 1
        elif verdict.action == "apart":
            backend.set_proposal_status(proposal.id, "rejected")
            outcome["rejected"] += 1
        else:
            if verdict.probabilities is not None:
                backend.update_proposal_judgement(
                    proposal.id, confidence=verdict.probabilities["same"],
                    reason=pair_reason(judge, verdict.probabilities),
                    compared_step=verdict.step,
                )
            elif verdict.step != proposal.compared_step:  # a step with nothing to ask
                backend.update_proposal_judgement(
                    proposal.id, confidence=proposal.confidence, reason=proposal.reason,
                    compared_step=verdict.step,
                )
            outcome["kept"] += 1
    return outcome
