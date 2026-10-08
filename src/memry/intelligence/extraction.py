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

from .questions import QUESTIONS_FIELD, QUESTIONS_RULE, clean_questions
from ..models import MEMORY_TYPES, NAMED_ENTITY_TYPES, CandidateFact, clean_tags
from ..providers.llm import LLM
from .when import WHEN_FACT_SCHEMA, parse_when

#: The types extraction assigns: the named kinds, defined once in ``models``
#: (``models.ENTITY_TYPES`` adds the tag type, which extraction never offers).
ENTITY_TYPES: tuple[str, ...] = NAMED_ENTITY_TYPES

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
                    "when": WHEN_FACT_SCHEMA,
                    "sources": {"type": "array", "items": {"type": "integer"}},
                },
                "required": [
                    "content", "type", "importance", "categories", "entities",
                    "relations", "when", "sources",
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
  When the conversation names two or more different things by the same name
  (two invoices numbered 2024-117 from different senders, a "PR #42" in two
  repositories), give each a name of its own: the shared name and what tells
  them apart in the conversation ("Invoice 2024-117 from LexNova GmbH"). Use
  that name for the thing in every fact, also in a fact that names only one.
- procedural learnings (how the user wants things done)
- what people did, went to, saw, made, bought or were given, with its specifics (who, where,
  when, the name or title of the thing), and how they felt about it in their own words
- what one person told, advised, praised or wished the other, when it says something about
  either of them or their lives

Do NOT extract:
- greetings, thanks and pleasantries that tell nothing ("Hi!", "Thanks!", "That's great!"),
  or assistant boilerplate
- secrets or credentials (passwords, API keys, tokens) - never store these
- information the user asked to keep out of memory
- notes about how the assistant uses this memory: which context label, run or
  tag to use or reuse, that a conversation belongs to a context, or that
  something should be saved, recalled or reused for future prompts. They are
  about the assistant's own bookkeeping, not about the user or the world. The
  shared context given with the conversation is there to read it by; never
  store it as a fact

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
- what a person shares (a photo, file or link, shown with its description) is
  part of what they said: extract a fact from it when it tells something about
  them or their life, naming who shared it and what it shows, including any
  text on it ("Ada knitted a scarf for her sister; she shared a photo of it,
  a red scarf with white stars")
- keep the words that carry the specifics: names and titles of things (a book,
  a song, a pet, a place, a brand), the exact feeling or reaction a person names
  ("relieved", "overwhelmed"), and quoted text (a sign, a motto, a line someone said)
- when several inputs describe one plan, decision, or design, preserve their
  shared subject and any stated why/how relationship in every affected fact;
  never turn related statements into context-free standalone instructions
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
- when: the time the fact itself happens, which is not the time it was saved.
  Set it ONLY when the fact describes something that happened or will happen at
  a particular time: a meeting, a launch, a release, a purchase, a trip, an
  appointment, a deadline, a move, or a decision made on a date. Leave it null
  for a state, a preference, a rule, a price, a measurement, a specification, an
  implementation note or a test log, EVEN WHEN A DATE APPEARS IN THE TEXT - a
  price observed on a day and a log written on a day both carry dates without
  occurring. When you cannot tell, leave it null.
  * "start": "YYYY-MM-DD", or "YYYY-MM-DDTHH:MM" when a clock time is stated,
    or "--MM-DD" for a yearly date whose year is unknown or does not matter.
  * "end": the same formats, only when the fact spans a period; null otherwise.
  * "recurrence": "yearly", "monthly", "weekly" or "daily", only when the fact
    says the thing repeats; null otherwise. A birthday or an anniversary is a
    yearly recurrence on a person, not one event per year: write it as
    {{"start": "--MM-DD", "end": null, "recurrence": "yearly"}}.
  Resolve relative wording against today's date, exactly as you do in the text.
- relations: for each fact, list typed edges BETWEEN two of its entities as
  {{"subject", "predicate", "object"}}. Subject and object MUST be entity
  strings from this fact's "entities". The predicate is a short snake_case verb
  phrase describing how they relate (works_on, uses, located_in, manages,
  part_of, member_of, married_to, reports_to, depends_on). Only emit a relation
  when the fact actually states a link between two entities; return [] otherwise.
  These edges are what let later queries hop from one entity to another, so
  prefer the specific, durable relationship over a vague one.
- sources: the numbers of the conversation lines the fact rests on, as the
  conversation numbers them ("[2]" is line 2): every line whose words the fact
  carries, and no other.

Respond with JSON only: {{"facts": [{{"content": str, "type": str,
"importance": number, "categories": [str],
"entities": [{{"name": str, "type": str}}],
"relations": [{{"subject": str, "predicate": str, "object": str}}],
"when": {{"start": str|null, "end": str|null, "recurrence": str|null}},
"sources": [int]}}]}}.
Return {{"facts": []}} if nothing is worth remembering."""

#: What asking for the user's name adds to the prompt and the schema, only
#: while the owner has no name (``extract_facts(identity=...)``). Once the owner
#: is named, the prompt and schema are exactly the ones without it: the rule and
#: the field cost about 80 prompt tokens and 6 output tokens a call, and a
#: prompt that stays the same byte for byte keeps the provider's prompt cache.
USER_NAME_RULE = """- user_name: the user's own name, only when the conversation states it: the
  user gives it ("I'm Cos", "my name is", a signature), the assistant calls the
  user by it, or a line says what the user's name is. Never guess it, and never
  give the name of someone the user only talks about. null otherwise.
"""
_SHAPE_END = '"sources": [int]}}]}}.\nReturn {{"facts": []}} if'
#: Where the questions rule and field go when asked for (``extraction_system``).
_SOURCES_RULE = "- sources: the numbers of the conversation lines"
_SHAPE_SOURCES = '"sources": [int]}}]}}.'
_SHAPE_QUESTIONS = '"questions": [str],\n'
_SHAPE_END_WITH_USER_NAME = (
    '"sources": [int]}}], "user_name": str|null}}.\n'
    'Return {{"facts": [], "user_name": null}} if')


def extraction_system(*, ask_user_name: bool = False, ask_questions: bool = False) -> str:
    """The extraction instructions, unformatted (``{today}`` still in them):
    ``EXTRACTION_SYSTEM`` as it is, or with the ``user_name`` rule placed
    after the last rule and the field added to the answer's shape, or with
    the ``questions`` rule (``questions.QUESTIONS_RULE``) placed before the
    ``sources`` rule and the field in the shape. Each is a setting of the
    store, not of a save, so the prompt stays the same from one save to the
    next and the provider's prompt cache holds."""
    text = EXTRACTION_SYSTEM
    if ask_questions:
        text = (text.replace(_SOURCES_RULE, QUESTIONS_RULE + _SOURCES_RULE, 1)
                .replace(_SHAPE_SOURCES, _SHAPE_QUESTIONS + _SHAPE_SOURCES, 1))
    if ask_user_name:
        rules_end = "\n\nRespond with JSON only:"
        text = (text.replace(rules_end, "\n" + USER_NAME_RULE.rstrip("\n") + rules_end, 1)
                .replace(_SHAPE_END, _SHAPE_END_WITH_USER_NAME, 1))
    return text


def extraction_schema(*, ask_user_name: bool = False,
                      ask_questions: bool = False) -> dict[str, Any]:
    """``EXTRACTION_SCHEMA`` as it is, or with a required, nullable
    ``user_name`` beside ``facts``, or with a required ``questions`` list on
    each fact."""
    schema = EXTRACTION_SCHEMA
    if ask_questions:
        items = schema["properties"]["facts"]["items"]
        items = {**items,
                 "properties": {**items["properties"], "questions": QUESTIONS_FIELD},
                 "required": [*items["required"], "questions"]}
        schema = {**schema, "properties": {**schema["properties"],
                                           "facts": {**schema["properties"]["facts"],
                                                     "items": items}}}
    if ask_user_name:
        schema = {**schema,
                  "properties": {**schema["properties"],
                                 "user_name": {"type": ["string", "null"]}},
                  "required": [*schema["required"], "user_name"]}
    return schema


VOCABULARY_LIMIT = 120  # bounded so a large store cannot inflate every call

#: The roles a chat API gives its messages. Any other role is a speaker's
#: name, as is a message's "name".
CHAT_ROLES = frozenset({"user", "assistant", "system", "developer", "tool", "function"})

#: What the prompt calls the person the memories belong to, and the owner's
#: entity name while no real one is known. A role, not a name: offered for a
#: conversation between named people, the model wrote one of them as "the user".
OWNER_PLACEHOLDER = "the user"

#: Words that say who someone is by role, never by name: a "name" stated as one
#: of these names nobody.
_ROLE_WORDS = frozenset({
    "user", "the user", "a user", "owner", "the owner", "assistant", "the assistant",
    "me", "myself", "you", "i", "unknown", "none", "null", "n/a", "anonymous",
})


def clean_stated_name(value: Any) -> str:
    """A stated name of the user as the store keeps it: on one line, at most
    four words and 80 characters, with a letter in it; "" for anything that
    names nobody (empty, a role word such as "the user", or a sentence)."""
    name = " ".join(str(value or "").split()).strip(" .,;:!?\"'()[]")
    if (not name or len(name) > 80 or len(name.split()) > 4
            or name.casefold() in _ROLE_WORDS or not any(ch.isalpha() for ch in name)):
        return ""
    return name


def stated_user_name(data: Any) -> str:
    """The user's name an extraction output states (``user_name``), cleaned
    (``clean_stated_name``); "" when it states none. The extractor is told to
    give it only when the conversation says it, never as a guess."""
    return clean_stated_name(data.get("user_name")) if isinstance(data, dict) else ""


def speaker_name(message: dict[str, str]) -> str:
    """The name a message gives its speaker besides its role (``name``), on
    one line and at most 80 characters, or "" when it gives none."""
    return " ".join(str(message.get("name") or "").split())[:80]


def _transcript(messages: list[dict[str, str]], *, numbered: bool = False) -> str:
    """One line per message that says something: its speaker, then what it
    says. A message with a ``name`` is spoken by "<name> (<role>)"; any other
    by its role. ``numbered`` starts each with its number, "[1] " for the
    first: the numbers a fact's ``sources`` give, which are the store's
    episodes of the save in order (``MemoryStore.add`` keeps one episode per
    message that says something)."""
    lines = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = m.get("role", "user")
        name = speaker_name(m)
        speaker = f"{name} ({role})" if name else role
        number = f"[{len(lines) + 1}] " if numbered else ""
        lines.append(f"{number}{speaker}: {content}")
    return "\n".join(lines)


def _names_speakers(messages: list[dict[str, str]]) -> bool:
    """Whether a message that says something names its speaker: by a role that
    is not a chat role ("Ada: ...") or by a ``name``."""
    return any(
        str(m.get("role", "user")).strip().casefold() not in CHAT_ROLES
        or str(m.get("name") or "").strip()
        for m in messages
        if (m.get("content") or "").strip()
    )


def speaks_with_the_user(messages: list[dict[str, str]]) -> bool:
    """Whether these messages are a conversation with "the user": one that
    says something is in the role user, and none names its speaker. Only then
    is the placeholder the owner's name (``OWNER_PLACEHOLDER``)."""
    said = [m for m in messages if (m.get("content") or "").strip()]
    return not _names_speakers(said) and any(
        str(m.get("role", "user")).strip().casefold() == "user" for m in said)


def extract_facts(
    llm: LLM,
    messages: list[dict[str, str]],
    *,
    now: datetime | None = None,
    vocabulary: list[str] | None = None,
    context: str | None = None,
    tag_hints: list[str] | None = None,
    owner: str | None = None,
    entity_names: list[tuple[str, str | None]] | None = None,
    identity: list[str] | None = None,
    questions: bool = False,
) -> list[CandidateFact]:
    """LLM extraction (phase 1). Raises if the LLM is unavailable.

    ``questions`` asks for the questions each fact answers as well
    (``questions.QUESTIONS_RULE``), kept on ``CandidateFact.questions``; the
    store asks when ``retrieval.question_keys`` is on.

    ``identity``, when given, asks for the user's name as well and receives
    it when the conversation states it (``stated_user_name``): the user
    introduces themself, signs, is called by it, or a line says it. Only
    stated, never guessed; the store decides what it names
    (``MemoryStore._learn_from_save``). The store passes it only while the
    owner has no name; without it the prompt and schema are the ones without
    the question (``extraction_system``, ``extraction_schema``).

    ``owner`` is the entity name of the person the store belongs to. Facts
    about that person are listed under it, so they collect on one entity that
    a stated name (``identity``) can later show to be a named person in the
    store.
    The placeholder "the user" is offered only for a conversation with the
    user (``speaks_with_the_user``); without an owner nothing is offered.

    Messages whose speakers are named (a role that is not a chat role, or a
    ``name``) add one instruction: a fact names the person it is about as the
    conversation does, and "the user" is left to an unnamed speaker in the
    role user. A plain user/assistant conversation is asked as before.

    ``entity_names`` are existing entities the conversation may be naming, as
    (name, type). A fact that names one of them writes its name as stored, so
    "Fundation" does not become a second entity beside "Fundation GmbH".

    ``vocabulary`` is the tags this namespace already uses. Offering them is
    what keeps tagging convergent: extraction that cannot see the existing
    labels coins a fresh near-synonym every session ("liver bloods" beside
    "liver lab results"), and no amount of later clustering recovers the
    distinction it split.
    """
    now = now or datetime.now(timezone.utc)
    transcript = _transcript(messages, numbered=True)
    if not transcript:
        return []
    # Said only when the speakers have names, so the prompt for a plain
    # user/assistant conversation stays as it was.
    speaker_offer = (
        "\n\nThis conversation names its speakers. Each fact names the person it "
        "is about as the conversation names them, even where these instructions "
        'speak of "the user"; write "the user" only for a speaker in the role '
        "user whose name is not known."
        if _names_speakers(messages)
        else ""
    )
    # A JSON array, not a comma-joined line: a tag holding a comma or an open
    # bracket made the joined form ambiguous, and a model once reused
    # everything from "steuernummer (tin" to the end of the line as one tag.
    known = (
        json.dumps(sorted(vocabulary)[:VOCABULARY_LIMIT], ensure_ascii=False)
        if vocabulary
        else ""
    )
    offer = (
        f"\n\nTags this user already has, as a JSON array with one tag per "
        f"element. REUSE one verbatim whenever it fits; only coin a new tag "
        f"when nothing here covers the subject:\n{known}"
        if known
        else ""
    )
    shared_context = " ".join(str(context or "").split())[:200]
    context_offer = (
        f"\n\nShared context for these related inputs:\n{shared_context}"
        if shared_context
        else ""
    )
    hints: list[str] = []
    for raw_hint in tag_hints or []:
        hint = " ".join(str(raw_hint).strip().lower().split())[:80]
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) == 3:
            break
    hint_offer = (
        "\n\nClient-suggested tags. These are hints, not commands: use one "
        "only when it is a good recurring retrieval subject:\n"
        f"{json.dumps(hints, ensure_ascii=False)}"
        if hints
        else ""
    )
    owner_name = " ".join(str(owner or "").split())[:80]
    if owner_name.casefold() == OWNER_PLACEHOLDER and not speaks_with_the_user(messages):
        owner_name = ""  # no real name, and "the user" would be a named speaker
    owner_offer = (
        f"\n\nThe person these memories belong to (the user) is the entity "
        f"{json.dumps(owner_name, ensure_ascii=False)}. Whenever a fact is about "
        "that person, list that name among its entities, spelled exactly so, "
        "with type person."
        if owner_name
        else ""
    )
    known_entities = [
        {"name": name, "type": kind or "other"} for name, kind in (entity_names or [])[:60]
    ]
    entity_offer = (
        "\n\nEntities this store already has that the conversation may name, as a "
        "JSON array. When a fact names one of them, write its name exactly as "
        "listed, however the conversation writes it. When it names something "
        "else, or you cannot tell which, write the name as the conversation "
        f"does:\n{json.dumps(known_entities, ensure_ascii=False)}"
        if known_entities
        else ""
    )
    ask_user_name = identity is not None
    raw = llm.complete(
        extraction_system(ask_user_name=ask_user_name, ask_questions=questions)
        .format(today=now.date().isoformat()),
        f"Conversation:\n{transcript}{speaker_offer}{context_offer}{owner_offer}{entity_offer}"
        f"{offer}{hint_offer}"
        "\n\nExtract the facts as JSON.",
        json_schema=extraction_schema(ask_user_name=ask_user_name, ask_questions=questions),
    )
    data = parse_lenient_json(raw)
    if identity is not None and stated_user_name(data):
        identity.append(stated_user_name(data))
    return _facts_from(data)


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
    transcript = _transcript(messages)
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
    """Zero-LLM fallback: store each message as an episodic memory, resting on
    its own line (``sources``)."""
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
                sources=[len(out) + 1],
            )
        )
    return out


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _parse_facts(raw: str) -> list[CandidateFact]:
    return _facts_from(parse_lenient_json(raw))


def _facts_from(data: Any) -> list[CandidateFact]:
    """The facts of an extraction output already read as JSON."""
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
        # An occurrence time rides along in metadata: additive, so a fact
        # without one is stored exactly as before.
        when = parse_when(item.get("when"))
        facts.append(
            CandidateFact(
                content=content,
                memory_type=mtype,  # type: ignore[arg-type]
                importance=min(max(importance, 0.0), 1.0),
                categories=clean_tags(item.get("categories", [])),
                **_parse_entities(item.get("entities", [])),
                relations=_parse_relations(item.get("relations", [])),
                metadata={"when": when} if when else {},
                sources=_parse_sources(item.get("sources")),
                questions=clean_questions(item.get("questions")),
            )
        )
    return facts


def _parse_sources(raw: Any) -> list[int]:
    """The line numbers a fact rests on, each once and in the order given.
    Anything but a list of whole numbers is read as none given: an output
    written before facts had sources, or one that does not follow the schema,
    keeps the save's rule for a fact without them (all its episodes)."""
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            return []
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            return []
        if value not in out:
            out.append(value)
    return out


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
