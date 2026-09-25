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
from ..models import Entity, EntityMention, MergeProposal, Scope
from ..providers.decisions import (
    MEASURED_MERGE_GATES,
    Answer,
    Choice,
    Decider,
    merge_gate_for,
)
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
    if _BARE_REFERENCE_RE.match(existing.name.strip()):
        return False  # "PR #92" is two words, and every repository has one
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

# Legal forms name the same company whether or not they are written out:
# "Fundation" and "Fundation GmbH" are one company.
_LEGAL_FORMS = frozenset({
    "gmbh", "mbh", "ug", "haftungsbeschränkt", "haftungsbeschrankt", "ag", "se",
    "kg", "kgaa", "ohg", "gbr", "ev", "inc", "ltd", "llc", "llp", "plc", "corp",
    "corporation", "limited", "sa", "sarl", "sas", "srl", "spa", "bv", "nv", "ab",
    "oy", "aps", "pty", "co",
})
_TLDS = ("ai", "app", "co", "com", "de", "dev", "eu", "io", "me", "net", "org", "so", "xyz")
_DOMAIN_RE = re.compile(rf"^([\w-]+)\.(?:{'|'.join(_TLDS)})$", re.I)
# How the legal forms above are written after a name, for looking up the other
# spelling of a company in the store.
_WRITTEN_LEGAL_FORMS = (
    "gmbh", "ug", "ug (haftungsbeschränkt)", "ag", "se", "kg", "gmbh & co. kg",
    "e.v.", "inc", "inc.", "ltd", "ltd.", "llc", "plc", "corp", "corp.", "bv", "sa",
)
# A name that starts like this describes a role, not a thing with a name.
_GENERIC_LEADS = frozenset({
    "the", "a", "an", "my", "our", "your", "his", "her", "their", "this", "that",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "mein",
    "meine", "unser", "unsere",
})
# A number that is only unique inside something the name leaves out: every
# repository has a PR #92, every tracker an issue 12.
_BARE_REFERENCE_RE = re.compile(
    r"^\W*(?:pr|pull request|mr|issue|ticket|bug|task|step|phase|stage|room|"
    r"version|v|release|sprint|chapter|section|page|table|figure|fig|item|order|"
    r"case|no|nr|number|nummer)?\W*#?\s*\d+[\w.]*\W*$",
    re.I,
)
# Types whose names are proper names. A document or a piece of code is usually
# named by what it is ("privacy policy", "config.py"), and every project has one.
_NAMED_TYPES = frozenset({
    "person", "organization", "project", "product", "place", "event", "concept",
})


def identity_key(name: str) -> str:
    """The name with its legal form and web domain taken off, for comparing
    two spellings of one company or product: "Fundation GmbH" and "Fundation"
    give "fundation", "bildy.ai" and "Bildy" give "bildy"."""
    raw = (name or "").strip()
    domain = _DOMAIN_RE.match(raw)
    if domain:
        raw = domain.group(1)
    words = list(_name_words(raw))
    while len(words) > 1 and words[-1] in _LEGAL_FORMS:
        words.pop()
    return " ".join(words)


def identity_variants(name: str) -> list[str]:
    """Other lowercased spellings of the same company or product, with the
    legal form or web domain written out or left off, to look up in the store.

    Entities are found by their exact lowercased name, so "Fundation" never
    met "Fundation GmbH" and the two were never compared.
    """
    raw = (name or "").strip()
    domain = _DOMAIN_RE.match(raw)
    tokens = (domain.group(1) if domain else raw).split()
    while len(tokens) > 1 and set(_name_words(tokens[-1])) <= _LEGAL_FORMS:
        tokens.pop()
    base = " ".join(tokens).lower()
    if not base:
        return []
    variants = {base, *(f"{base} {form}" for form in _WRITTEN_LEGAL_FORMS)}
    if len(tokens) == 1:
        variants |= {f"{base}.{tld}" for tld in _TLDS}
    variants.discard(raw.lower())
    return sorted(variants)


def name_is_specific(name: str, entity_type: str | None) -> bool:
    """Whether two mentions of exactly this name are, as a rule, one thing.

    True for a full personal name, a company, a product, a project, a place or
    a named programme ("Bochra Saffar", "Fundation GmbH", "Docker",
    "Forschungszulage"), and for a document or code reference that carries its
    own number ("Ronin Dash PR #41"). False for a first name on its own, a role
    ("the consultant"), an initial ("R. Patel"), a number that needs its
    repository or tracker ("PR #92"), a description in lowercase ("landing
    page") and a single word of unknown type.
    """
    raw = (name or "").strip()
    words = _name_words(raw)
    if not words or words[0] in _GENERIC_LEADS or _BARE_REFERENCE_RE.match(raw):
        return False
    if entity_type in ("document", "code"):
        # A reference with its namespace ("Ronin Dash PR #41") or a long
        # number ("BH258636489") names one thing; "config.py" does not.
        return any(ch.isdigit() for ch in raw) and (
            len(words) >= 3 or re.search(r"\d{4,}", raw) is not None
        )
    full_words = sum(len(word) > 1 for word in words)
    if entity_type in (None, "other"):
        # Untyped: a legal form makes a company, and two capitalised words make
        # a full name or a named thing. One word alone could be a first name.
        return (len(words) > 1 and words[-1] in _LEGAL_FORMS) or (
            full_words >= 2 and any(ch.isupper() for ch in raw)
        )
    if entity_type not in _NAMED_TYPES:
        return False
    # Lowercase words are a description ("landing page"), not a name.
    if not any(ch.isupper() or ch.isdigit() for ch in raw) and not _DOMAIN_RE.match(raw):
        return False
    if entity_type == "person":
        return full_words >= 2
    return True


def _home_id(entity: Entity) -> str | None:
    home = (entity.metadata or {}).get("home")
    return home.get("id") if isinstance(home, dict) else None


def specific_same_name(
    existing: Entity,
    other_name: str,
    other_type: str | None = None,
    other_home: str | None = None,
) -> bool:
    """Two mentions whose names match exactly (legal form and domain aside),
    whose names are specific, whose known types agree and whose homes agree.

    A new mention has no home yet, so an existing entity with a home is left to
    the normal rules: "Settings" under two products is two things.
    """
    key = identity_key(existing.name)
    if not key or key != identity_key(other_name):
        return False
    if existing.entity_type and other_type and existing.entity_type != other_type:
        return False
    if _home_id(existing) != other_home:
        return False
    entity_type = existing.entity_type or other_type
    return name_is_specific(existing.name, entity_type) and name_is_specific(
        other_name, entity_type
    )


#: No rule merges two mentions after a "different" at this confidence or
#: higher. The bar used to be the provider's merge gate, 0.95 for gpt-5-mini,
#: so two people with one full name and a "different" at 0.90 were merged
#: because both facts mentioned employee numbers (identity_v1 case d12).
DIFFERENT_VETO = 0.5


def _should_merge(
    existing: Entity,
    existing_facts: list[str],
    other_name: str,
    other_facts: list[str],
    other_type: str | None,
    judgment: dict[str, Any],
    other_home: str | None = None,
) -> bool:
    """Whether two mentions are one thing, given the judge's answer.

    A "same" at the provider's gate merges. After a "different" from 0.5
    nothing merges. Otherwise two rules merge without a sure "same":

    * a full name with shared context words (``_obvious_same_entity``);
    * the same specific name (``specific_same_name``) with a "same" at any
      confidence. Facts about a company's insurance and its tax number share no
      words, and asked whether they are clearly the same company, Jev answered
      "same" at 38-69% on a real store, under its 70% gate. On 101 labelled
      cases, over two runs, this rule took gpt-5-mini from 9-11 to 43-44 of 51
      correct merges and added no wrong one. Merging on anything short of
      "different" added 3-4 wrong ones, most of them "unsure" on two people
      who share a full name. See evals/identity_policy_benchmark.py.
    """
    verdict, confidence = judgment["verdict"], judgment["confidence"]
    if verdict == "same" and confidence >= judgment.get("gate", AUTO_CONFIRM_CONFIDENCE):
        return True
    if verdict == "different" and confidence >= DIFFERENT_VETO:
        return False
    if _obvious_same_entity(existing, existing_facts, other_name, other_facts, other_type):
        return True
    return verdict == "same" and specific_same_name(
        existing, other_name, other_type, other_home
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
) -> dict[str, Entity]:
    """Attach a memory's entity mentions, creating/reusing entities per the
    conservative policy. Returns a map of normalized surface -> entity, so the
    caller can resolve relation triples to the entities they linked to. Pass
    ``attach=False`` when the caller will replace all mentions atomically."""
    types = types or {}
    resolved: dict[str, Entity] = {}
    # A name the store has never seen is screened before it becomes an entity:
    # mechanically first (free, certain), then one typed question per name. A
    # name that already has an entity is left alone; upkeep reviews those.
    cleaned = [s.strip() for s in surfaces if s and s.strip()]
    unseen = [
        s for s in dict.fromkeys(cleaned)
        if not non_referent_reason(s)
        and not backend.find_entity_candidates(s.lower(), scope)
    ]
    verdicts = screen_names(decider, memory_content, unseen)
    for surface in cleaned:
        normalized = surface.lower()
        if not normalized or normalized in resolved:
            continue
        if non_referent_reason(surface) or screened_out(verdicts.get(normalized)):
            continue

        candidates = backend.find_entity_candidates(normalized, scope)
        found = {candidate.id for candidate in candidates}
        candidates += [
            entity
            for entity in backend.find_entities_by_aliases(identity_variants(surface), scope)
            if entity.id not in found
        ]
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
            judgment = _judge(
                llm, candidate, facts, memory_content, surface, decider
            )
            if _should_merge(
                candidate, facts, surface, [memory_content], types.get(normalized), judgment
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

    A pair whose members live under different homes ("privacy policy" in two
    projects) is two things by construction, so it is never raised: nobody can
    answer that question, and nobody should be asked it.
    """
    groups: dict[str, list[Entity]] = {}
    for entity in backend.list_entities(scope, limit=10_000):
        if entity.merged_into is None:
            key = identity_key(entity.name) or entity.normalized or entity.name.lower()
            groups.setdefault(key, []).append(entity)

    def home_of(entity: Entity) -> str | None:
        home = (entity.metadata or {}).get("home")
        return home.get("id") if isinstance(home, dict) else None

    created = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            if created >= limit:
                return created
            home_a, home_b = home_of(anchor), home_of(other)
            if home_a and home_b and home_a != home_b:
                continue
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
    """
    outcome = {"confirmed": 0, "rejected": 0, "kept": 0}
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
        should_merge = auto_confirm and _should_merge(
            entity_a, facts_a, entity_b.name, facts_b, entity_b.entity_type,
            judgment, other_home=_home_id(entity_b),
        )
        if should_merge and backend.merge_entities(entity_a.id, entity_b.id):
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
    return outcome
