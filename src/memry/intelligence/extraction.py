"""Fact extraction: turn raw conversation into candidate memories.

With an LLM configured, extraction distills discrete, self-contained facts
(the Mem0-paper phase 1). Without one, memry falls back to *verbatim*
mode - each message is stored as an episodic memory - so the system stays
useful with zero API keys.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..models import MEMORY_TYPES, CandidateFact
from ..providers.llm import LLM

# `document` and `code` were added after reviewing what a real store dumped into
# "other": contracts, invoices and registration numbers on one side, files,
# symbols and tables on the other. Both are common enough to be worth naming,
# and a named type keeps a document from being merged with a person who happens
# to share its name. Types are deliberately few - each extra one is another way
# for the model to mis-sort, and the type does not affect search ranking.
ENTITY_TYPES: tuple[str, ...] = (
    "person", "organization", "project", "product", "place", "event",
    "document", "code", "concept", "other",
)

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["semantic", "episodic", "procedural"],
                    },
                    "importance": {"type": "number"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ENTITY_TYPES,
                                },
                            },
                            "required": ["name", "type"],
                            "additionalProperties": False,
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                            },
                            "required": ["subject", "predicate", "object"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "content", "type", "importance", "categories", "entities", "relations",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM = """You are the long-term memory extraction system of an AI assistant.
Given a conversation, extract discrete facts worth remembering in future,
unrelated conversations. Today's date is {today}.

Extract:
- stable facts about the user (identity, role, location, relationships)
- preferences, opinions, goals, constraints
- decisions made and commitments/plans (convert relative dates to absolute)
- important entities, each with a type: person, organization, project, product,
  place, event, document (a contract, invoice, certificate, form, report or
  reference number), code (a file, function, table, endpoint or config key),
  concept, or other (use "other" only when none fit).
  An entity is a NAMED, REFERRING thing you could later ask a question about:
  a person, a company, a place, a named project, product, or event.
  It is NOT: a salutation or greeting ("Sehr geehrte", "Dear Sir"), a template
  placeholder ("[date]", "bracketed placeholders"), a style or tone descriptor
  ("casual variant", "casual but not choppy tone"), a generic role word
  ("user", "article", "adverbs"), a sentence fragment, or a description of the
  task you were asked to do ("corrected full version", "2-3 improved versions").
  If it has no name of its own, leave it out. An empty entity list is fine and
  is much better than a wrong one.
- procedural learnings (how the user wants things done)

Do NOT extract:
- small talk, transient context ("I'm tired today"), or assistant boilerplate
- secrets or credentials (passwords, API keys, tokens) - never store these
- information the user asked to keep out of memory

Rules:
- each fact must be fully self-contained: resolve pronouns and references
- one fact per item; keep each under ~200 characters, but NEVER shorten a fact
  by dropping specifics
- NEVER drop: numeric values, dates, prices, model/version identifiers, file
  formats, library/tool names, negative constraints ("not X", "rather than X",
  "must not"), or the stated reason for a constraint. These carry the
  operational weight; carry them into the fact verbatim.
- a constraint buried mid-sentence is still its own fact when it changes
  future behavior; extract it as a separate item rather than summarizing over it
- prefer several precise facts over one compressed summary
- importance in [0,1]: 0.9+ identity/hard constraints, ~0.7 preferences and
  decisions, ~0.4 minor details
- type: "semantic" (stable fact/preference), "episodic" (dated event/plan),
  "procedural" (how-to / workflow rule)
- categories: 1-3 retrieval tags, lowercase, words separated by single spaces
  (write "liver health", never "liver_health" or "LiverHealth"). A tag names the
  smallest RECURRING subject a future conversation would open with. Good shapes:
  domain + object ("liver health"), cadence + activity ("weekly gym"), artifact
  + domain ("health documents"), project + activity ("memry deployment"),
  bounded period + task ("2026 taxes"), event stream ("doctor appointments").
  Two failure modes, both of which destroy retrieval:
  * too broad - NEVER emit a bare domain like "health", "work", "personal",
    "finance", "medical", "misc" or "other". Those collect memories that share a
    subject area but would never be wanted in the same conversation.
  * too narrow - NEVER put a date, a measurement, or a one-off identifier in a
    tag ("2026-04-02 imaging", "paris sep 3-10 trip"). A tag used once is a tag
    that can never group anything. Tag the recurring concern, not the instance.
- relations: for each fact, list typed edges BETWEEN two of its entities as
  {{"subject", "predicate", "object"}}. Subject and object MUST be entity
  strings from this fact's "entities". The predicate is a short snake_case verb
  phrase describing how they relate (works_on, uses, located_in, manages,
  part_of, member_of, married_to, reports_to, depends_on). Only emit a relation
  when the fact actually states a link between two entities; return [] otherwise.
  These edges are what let later queries hop from one entity to another, so
  prefer the specific, durable relationship over a vague one.

Respond with JSON only: {{"facts": [{{"content": str, "type": str,
"importance": number, "categories": [str],
"entities": [{{"name": str, "type": str}}],
"relations": [{{"subject": str, "predicate": str, "object": str}}]}}]}}.
Return {{"facts": []}} if nothing is worth remembering."""


VOCABULARY_LIMIT = 120  # bounded so a large store cannot inflate every call


def extract_facts(
    llm: LLM,
    messages: list[dict[str, str]],
    *,
    now: datetime | None = None,
    vocabulary: list[str] | None = None,
) -> list[CandidateFact]:
    """LLM extraction (phase 1). Raises if the LLM is unavailable.

    ``vocabulary`` is the tags this namespace already uses. Offering them is
    what keeps tagging convergent: extraction that cannot see the existing
    labels coins a fresh near-synonym every session ("liver bloods" beside
    "liver lab results"), and no amount of later clustering recovers the
    distinction it split.
    """
    now = now or datetime.now(timezone.utc)
    transcript = "\n".join(
        f"{m.get('role', 'user')}: {m['content'].strip()}"
        for m in messages
        if (m.get("content") or "").strip()
    )
    if not transcript:
        return []
    known = ", ".join(sorted(vocabulary)[:VOCABULARY_LIMIT]) if vocabulary else ""
    offer = (
        f"\n\nTags this user already has. REUSE one verbatim whenever it fits; "
        f"only coin a new tag when nothing here covers the subject:\n{known}"
        if known
        else ""
    )
    raw = llm.complete(
        EXTRACTION_SYSTEM.format(today=now.date().isoformat()),
        f"Conversation:\n{transcript}{offer}\n\nExtract the facts as JSON.",
        json_schema=EXTRACTION_SCHEMA,
    )
    return _parse_facts(raw)


COVERAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"missing": {"type": "array", "items": {"type": "string"}}},
    "required": ["missing"],
    "additionalProperties": False,
}

COVERAGE_SYSTEM = """You audit what a memory system stored against what it was told.
Compare the INPUT with the STORED facts. Report substantive details that appear
in the input but in none of the stored facts: numbers, dates, prices, names,
model/version identifiers, file formats, library/tool names, constraints
(especially negations like "not" or "rather than"), and the reasons given for
constraints. Ignore phrasing differences, small talk, and anything a stored
fact already captures in different words.
Respond with JSON only: {"missing": ["<short description>", ...]}.
Return {"missing": []} when nothing substantive was lost."""


def verify_coverage(
    llm: LLM, messages: list[dict[str, str]], stored: list[str]
) -> list[str]:
    """One extra LLM pass after a write: which operational details from the
    input made it into none of the stored facts? Extraction is lossy and
    non-deterministic; this turns silent loss into a reportable warning."""
    transcript = "\n".join(
        f"{m.get('role', 'user')}: {m['content'].strip()}"
        for m in messages
        if (m.get("content") or "").strip()
    )
    if not transcript or not stored:
        return []
    listing = "\n".join(f"- {s}" for s in stored)
    raw = llm.complete(
        COVERAGE_SYSTEM,
        f"INPUT:\n{transcript}\n\nSTORED FACTS:\n{listing}",
        json_schema=COVERAGE_SCHEMA,
    )
    parsed = parse_lenient_json(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("missing"), list):
        return [str(m).strip() for m in parsed["missing"] if str(m).strip()][:8]
    return []


def verbatim_candidates(messages: list[dict[str, str]]) -> list[CandidateFact]:
    """Zero-LLM fallback: store each message as an episodic memory."""
    out: list[CandidateFact] = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role", "user")
        out.append(
            CandidateFact(
                content=content if role == "user" else f"{role}: {content}",
                memory_type="episodic",
                importance=0.5,
            )
        )
    return out


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _parse_facts(raw: str) -> list[CandidateFact]:
    data = parse_lenient_json(raw)
    if data is None:
        return []
    items = data.get("facts", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    facts: list[CandidateFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        mtype = item.get("type", "semantic")
        if mtype not in MEMORY_TYPES:
            mtype = "semantic"
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        facts.append(
            CandidateFact(
                content=content,
                memory_type=mtype,  # type: ignore[arg-type]
                importance=min(max(importance, 0.0), 1.0),
                categories=[str(c) for c in item.get("categories", []) if c],
                **_parse_entities(item.get("entities", [])),
                relations=_parse_relations(item.get("relations", [])),
            )
        )
    return facts


def _parse_entities(raw: Any) -> dict[str, Any]:
    """Accept both the typed form [{name,type}] and the legacy [str] form.
    Returns kwargs {entities: [name], entity_types: {name_lower: type}}."""
    names: list[str] = []
    types: dict[str, str] = {}
    if not isinstance(raw, list):
        return {"entities": names, "entity_types": types}
    for e in raw:
        if isinstance(e, str):
            name = e.strip()
        elif isinstance(e, dict):
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "")).strip().lower()
            if name and etype in ENTITY_TYPES:
                types[name.lower()] = etype
        else:
            continue
        if name:
            names.append(name)
    return {"entities": names, "entity_types": types}


RELATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}

RELATION_SYSTEM = """Given a statement and the entities in it, list the typed
edges that hold BETWEEN those entities. Subject and object must both be from the
given entity list, spelled exactly. Predicate is a short snake_case verb phrase
(works_on, uses, located_in, manages, part_of, member_of, reports_to). Only emit
an edge the statement actually asserts; return [] otherwise. JSON only:
{"relations": [{"subject": str, "predicate": str, "object": str}]}."""


def extract_relations(llm: LLM, content: str, entities: list[str]) -> list[dict[str, str]]:
    """Focused, cheap relation extraction for backfilling existing memories.

    Deliberately small (one short prompt, only for memories that already have
    two or more entities), so re-processing a store costs a fraction of a full
    re-extraction."""
    if len(entities) < 2:
        return []
    raw = llm.complete(
        RELATION_SYSTEM,
        f"Statement: {content}\nEntities: {', '.join(entities)}\nRelations as JSON.",
        json_schema=RELATION_SCHEMA,
    )
    data = parse_lenient_json(raw)
    return _parse_relations(data.get("relations", []) if isinstance(data, dict) else [])


def _parse_relations(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for r in raw:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subject", "")).strip()
        pred = str(r.get("predicate", "")).strip().lower().replace(" ", "_")
        obj = str(r.get("object", "")).strip()
        if subj and pred and obj and subj.lower() != obj.lower():
            out.append({"subject": subj, "predicate": pred, "object": obj})
    return out


def parse_lenient_json(raw: str) -> Any:
    """Parse JSON out of LLM output: tolerates code fences and leading prose."""
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None
