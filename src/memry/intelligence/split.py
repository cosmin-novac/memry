"""Splitting a memory that holds several facts into single facts.

A memory should hold one fact: one claim, event or attribute with the details
that belong to it. Merges before reconcile kept to that (``reconcile.
MERGE_REQUEST``) folded each new claim about one subject into the memory
before it, one save at a time, until one memory held five claims about a
thesis: one vector for five claims matches none of them well. The repair
(``MemoryStore.split_memories``, ``memry split-memories``) asks the text model
to split such a memory, and replaces it with the facts.

Which memories are asked (``is_candidate``): a memory in use whose text has
more than one sentence, since a text that joins facts writes them as
sentences ("... The thesis further argues ... The thesis explicitly rejects
..."). One sentence is one statement, and it is left alone without a call.
The model then decides: one fact back means the memory states one fact (a
fact and its details may take two sentences) and it is left alone. A memory
waiting for its extraction, or kept beside another for a person to settle,
is not asked. ``min_words`` narrows a run to longer texts; it is a cost
limit a person may set, not the rule.

A fact read alone later has nothing around it, so each must state its
subject. The prompt asks for it, and ``without_subject`` checks it before
anything is written: of twelve two-fact splits of a real store made by the
first prompt, two had a fact that no longer said which project it was about
("Merge the first change first"), which is worse than the memory it came
from. A list stays one fact: asked without that rule, the model split one
memory's list of skills into 23 facts that each named one skill.

The same call says what each fact is about, among the entities the memory
is linked to (named things and tags, numbered under the memory). Given the
entities whose names its text states, a fact lost the ones it does not
spell out: "The tallest bulls in Etosha stand 4 m" lost Elephant, and the
linked search did not find it for a question about elephants; and each fact
took all the memory's tags, so the five facts of a five-topic summary each
ranked for all five. A fact keeps what the model gives it and what its text
names (``fact_homes``), and a split that would leave a fact with none of the
memory's entities, or an entity on no fact, is not made (``entity_gaps``).

A dry run only previews: asked again, the model answers a little
differently. ``make_plan`` keeps the splits a dry run proposed, and a later
run with that plan (``plan_entries``) makes exactly those, without the model.

Each fact keeps the memory's dates, so a date heading the whole memory
("Decision (2026-09-12): ...") is kept once, in the first fact: written in
front of every fact, as a real store's split did, it read as the date of
each. Its tense is kept too: the model put a present state of that store
("Reporting ... are included in every tier") in the past.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..models import TOPIC_TYPE, Entity, Memory, utcnow
from ..providers.llm import LLM
from .extraction import OWNER_PLACEHOLDER, parse_lenient_json
from .reconcile import CONFLICT_KEY

SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"},
                       "about": {"type": "array", "items": {"type": "integer"}}},
        "required": ["text", "about"],
        "additionalProperties": False,
    }}},
    "required": ["facts"],
    "additionalProperties": False,
}

SPLIT_SYSTEM = """You keep an AI assistant's long-term memory. One stored memory may
state several facts that were joined over time. Split it into single facts.

A single fact is one claim, event or attribute, with the details that belong
to it: who, what kind, when, where, how, how much, why, how sure. A detail
stays in the fact it belongs to, even when the memory gives it a sentence of
its own: "Tom has a dog named Rex. Rex is a three-year-old German shepherd."
is one fact, and so is "Maria quit coffee. She says it kept her awake." (the
reason for it). A list is one fact too: "Ada's skills include Python, SQL and
Go." stays whole, and so does "The review found the cells tau, delta and w=25
unmeasured." An item becomes a fact of its own only when it carries details of
its own (its own price, date or reason). Another claim, position, finding,
opinion, decision or event about the same subject is a fact of its own.

Rules:
- Every fact must stand alone, read on its own months later with nothing
  around it. It names its subject by the name the memory uses for it: the
  person, project, product, repository, document or place it is about. Never
  write "he", "she", "it", "they", "this", "the thesis", "the project" or
  "the repository" without the name, and never leave an instruction or a
  decision without the thing it is about: "PR 12 builds on PR 11 in the Tern
  repository. Merge PR 11 first." splits into "PR 12 in the Tern repository
  builds on PR 11." and "In the Tern repository, merge PR 11 before PR 12."
- Keep every concrete detail of the memory (numbers, dates, prices, names,
  titles, versions, constraints and their reasons), each in the fact it
  belongs to, in the memory's own words where you can. Never add anything the
  memory does not say, not even a date, and never merge two of its facts
  into one.
- Keep each time and tense as the memory writes it: every fact keeps the
  date the memory was said, so a relative time ("last year") stays as it
  is, and what the memory states in the present stays in the present
  ("Reports are included in every tier" never becomes "were included").
- A date that heads the whole memory, as in "Decision (2026-09-12): ...",
  dates the memory, not each fact: keep it once, in the first fact, and do
  not put it in front of the others. A fact about an event with a date of
  its own keeps that date.
- If the memory states one fact, return it as the only item, unchanged.
- "About" under the memory numbers the things it is linked to. Give each
  fact the numbers of those it is about, as many as apply, also one it does
  not spell out: in a memory about elephants, "The tallest bulls in Etosha
  stand 4 m." is about Elephant as well as Etosha. Every fact is about one
  of them at least, and each of them stays with one fact at least. With no
  list, "about" is empty.

Respond with JSON only: {"facts": [{"text": str, "about": [int, ...]}, ...]}"""

#: A sentence ends at ".", "!" or "?" before another that starts with a
#: capital, a digit or a quote, and at every ";", which merged texts join
#: facts with ("chose oak cabinets; the budget is $18,000"). Titles and common
#: abbreviations do not end one.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|(?<=;)\s+")
_ABBREVIATION = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|St|Mt|Jr|Sr|vs|etc|e\.g|i\.e|approx|No|Inc|Ltd|Co)\.$", re.I)


def sentences(text: str) -> list[str]:
    """The sentences of a text, by the boundary above."""
    parts = _BOUNDARY.split(" ".join(text.split()))
    out: list[str] = []
    for part in parts:
        if out and _ABBREVIATION.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return [s for s in out if s.strip()]


#: A date heading a whole memory at its very start: a label with an ISO date
#: in brackets, or the date alone, then a colon ("Decision (2026-09-12): ...",
#: "2026-09-12: ..."). "Note (draft):" is no date.
_HEADING = re.compile(
    r"^\s*(?:[^\W\d][^():\n]{0,48}\((\d{4}-\d{2}-\d{2})\)|(\d{4}-\d{2}-\d{2}))\s*:")


def heading_date(text: str) -> str | None:
    """The date a memory's heading gives the whole memory, as a time the way
    ``valid_from`` is stored ("2026-09-12T00:00:00+00:00"), or None. The
    split keeps that date once, in the first fact (the prompt), so each fact
    takes it as the time it holds from."""
    found = _HEADING.match(text)
    if found is None:
        return None
    day = found.group(1) or found.group(2)
    try:
        date.fromisoformat(day)
    except ValueError:
        return None
    return f"{day}T00:00:00+00:00"


def is_candidate(memory: Memory, *, min_words: int = 0) -> bool:
    """Whether a memory in use is asked about: its text has more than one
    sentence and at least ``min_words`` words, and it is neither waiting for
    extraction nor kept beside another for a person to settle."""
    metadata = memory.metadata or {}
    if memory.invalid_at is not None or metadata.get("pending_distillation"):
        return False
    if metadata.get(CONFLICT_KEY):
        return False
    if len(memory.content.split()) < min_words:
        return False
    return len(sentences(memory.content)) > 1


def entity_label(entity: Entity) -> str:
    """An entity as the model and a person read it: its name with its type
    ("Elephant (animal)"), a tag's being "tag"."""
    kind = "tag" if entity.entity_type == TOPIC_TYPE else entity.entity_type
    return f"{entity.name} ({kind})" if kind else entity.name


def split_prompt(memory: Memory, linked: list[Entity] | None = None) -> str:
    """The memory and the things it is linked to, numbered (``linked``: its
    named things, then its tags), without its dates. Shown with the date it
    was said, the model wrote that date into every fact, where it reads as
    the time of the fact; each fact keeps the memory's dates anyway."""
    about = "".join(f"\n{n}. {entity_label(entity)}"
                    for n, entity in enumerate(linked or [], 1))
    return (f"Memory:\n{memory.content}\n\n"
            + (f"About:{about}\n\n" if about else "") + "Split it as JSON.")


def split_facts(
    llm: LLM, memory: Memory, linked: list[Entity] | None = None,
) -> list[dict[str, Any]]:
    """The single facts the text model reads in a memory, in its order, empty
    ones and repeats left out, each with the ids of the ``linked`` entities
    the model says it is about (``about``): a number off the list, or one
    given twice, is left out. One fact back means the memory states one."""
    linked = list(linked or [])
    parsed = parse_lenient_json(llm.complete(SPLIT_SYSTEM, split_prompt(memory, linked),
                                             json_schema=SPLIT_SCHEMA))
    facts = parsed.get("facts") if isinstance(parsed, dict) else None
    if not isinstance(facts, list):
        raise ValueError("the text model answered no list of facts")
    out: list[dict[str, Any]] = []
    for fact in facts:
        # a bare string, as answers before "about" were, is a fact about nothing
        raw = fact.get("text") if isinstance(fact, dict) else fact
        text = " ".join(str(raw or "").split())
        if not text or any(text == kept["text"] for kept in out):
            continue
        about: list[str] = []
        for n in (fact.get("about") if isinstance(fact, dict) else None) or []:
            if (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(linked)
                    and linked[n - 1].id not in about):
                about.append(linked[n - 1].id)
        out.append({"text": text, "about": about})
    return out


def fact_homes(facts: list[str], about: list[list[str]],
               linked: list[tuple[str, list[str]]]) -> list[list[str]]:
    """The entities each fact keeps, in the order of ``linked`` (each entity's
    id with its names): those the model said it is about, and any its text
    names, should the model have missed one. An id not linked is left out."""
    return [[entity_id for entity_id, names in linked
             if entity_id in (about[i] if i < len(about) else []) or names_in(fact, names)]
            for i, fact in enumerate(facts)]


def entity_gaps(homes: list[list[str]], linked: list[str]) -> tuple[list[int], list[str]]:
    """What a split would lose of a memory's links (``linked``: its entity
    ids): the facts that would keep none of them, and the entities no fact
    would keep. Either holds the split back: a fact linked to nothing is
    found by no linked search, and an entity on no fact takes what the
    memory said out of that entity's searches."""
    if not linked:
        return [], []
    kept = {entity_id for home in homes for entity_id in home}
    return ([i for i, home in enumerate(homes) if not home],
            [entity_id for entity_id in linked if entity_id not in kept])


def names_in(text: str, names: list[str]) -> bool:
    """Whether a text names a thing by any of ``names`` (whole words, any
    case, a genitive "s" allowed: "Cosmins Umsatz")."""
    lowered = text.casefold()
    return any(re.search(rf"(?<!\w){re.escape(name.casefold())}s?(?!\w)", lowered)
               for name in names if name.strip())


#: A fact that opens on one of these points back at something it does not
#: name.
_POINTER = re.compile(r"^(?:he|she|it|they|this|that|these|those|his|her|its|their|them)\b",
                      re.I)
#: A capitalised word, or a run of them, inside a sentence: a name the text
#: states (a word that opens a sentence may be any word).
_NAME = re.compile(r"(?<=[^\s.!?;:]\s)[A-Z][\w'’&.-]*(?:\s+[A-Z][\w'’&.-]*)*")


#: A name of one word this short is mostly an abbreviation ("PR 11", "the
#: API", "AI"), which says what kind of thing a fact is about, not which one.
_ABBREVIATION_LENGTH = 3


def proper_names(text: str) -> list[str]:
    """The names a text states: capitalised words inside its sentences, the
    pointing words above and "I" left out."""
    names = []
    for sentence in sentences(text):
        for found in _NAME.findall(sentence):
            name = re.sub(r"['’]s$", "", found.rstrip(".'’"))
            if name and name != "I" and not _POINTER.match(name) and name not in names:
                names.append(name)
    return names


def without_subject(facts: list[str], memory: Memory, linked: list[str],
                    owner: list[str] | None = None) -> list[str]:
    """The facts of a split that do not state their subject, which the split
    must not be made with: a fact alone, months later, has to say what it is
    about. A fact states it when it names one of the things the memory is
    linked to (``linked``: names and aliases of its named entities), one of
    its tags the text names, or a name the text states (``proper_names``)
    other than a one-word abbreviation ("Merge PR 11 first." names no
    project). The owner counts too (``owner``: their names, and "the user"), since a
    memory about the owner often names nobody ("Prefers tea. Dislikes
    coffee.") and its facts then say "The user prefers tea.". A fact that
    opens on "he", "it", "this" or the like fails whatever it names later. A
    memory that names nothing and is linked to nothing is not checked beyond
    that."""
    subjects = [s for s in linked if s.strip()]
    subjects += [s for s in (memory.categories or [])
                 if names_in(memory.content, [s]) and s not in subjects]
    subjects += [name for name in proper_names(memory.content)
                 if (" " in name or len(name) > _ABBREVIATION_LENGTH) and name not in subjects]
    if subjects:
        subjects += [OWNER_PLACEHOLDER, *(owner or [])]
    missing = []
    for fact in facts:
        if _POINTER.match(fact) or (subjects and not names_in(fact, subjects)):
            missing.append(fact)
    return missing


#: What a split plan file is (``make_plan``), and the one version read.
PLAN_FORMAT = "memry-split-plan"
PLAN_VERSION = 2


def make_plan(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """The splits that dry runs of ``MemoryStore.split_memories`` proposed,
    as a plan to make later as they are: each memory's id and namespace, the
    text it was judged on and its facts, each with the ids of the entities
    it keeps (version 2; the facts of version 1 had none). A memory the dry
    run kept whole is not in it."""
    return {
        "format": PLAN_FORMAT, "version": PLAN_VERSION, "made_at": utcnow(),
        "splits": [{"memory_id": entry["memory_id"], "user": entry.get("user"),
                    "content": entry["content"],
                    "facts": [{"text": text, "about": list(about)}
                              for text, about in zip(entry["facts"], entry["about"])]}
                   for summary in summaries for entry in summary["splits"]
                   if not entry.get("not_split")],
    }


def plan_entries(plan: Any) -> list[dict[str, Any]]:
    """The splits of a plan ``make_plan`` wrote, checked before any is made,
    each with its facts' texts (``facts``) and their entity ids (``about``):
    a file of another format or version, or a split without its memory's id
    and text, or with fewer than two facts each a text with a list of ids,
    refuses the whole plan (ValueError) rather than a part of it being
    made."""
    if not isinstance(plan, dict) or plan.get("format") != PLAN_FORMAT:
        raise ValueError("not a split plan (memry split-memories --dry-run --plan-out)")
    if plan.get("version") != PLAN_VERSION:
        raise ValueError(f"split plan version {plan.get('version')!r}; this Memry reads "
                         f"version {PLAN_VERSION}: make the plan again with --dry-run "
                         "--plan-out")
    entries = plan.get("splits")
    if not isinstance(entries, list):
        raise ValueError("the split plan holds no list of splits")
    out: list[dict[str, Any]] = []
    for n, entry in enumerate(entries, 1):
        facts = entry.get("facts") if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or not isinstance(entry.get("memory_id"), str)
                or not isinstance(entry.get("content"), str)
                or not isinstance(entry.get("user"), (str, type(None)))
                or not isinstance(facts, list) or len(facts) < 2
                or not all(isinstance(f, dict) and isinstance(f.get("text"), str)
                           and f["text"].strip() and isinstance(f.get("about"), list)
                           and all(isinstance(i, str) for i in f["about"])
                           for f in facts)):
            raise ValueError(f"split {n} of the plan needs a memory_id, its content, "
                             "its user and two facts or more, each a text and the ids "
                             "of the entities it is about")
        out.append({"memory_id": entry["memory_id"], "user": entry["user"],
                    "content": entry["content"],
                    "facts": [" ".join(f["text"].split()) for f in facts],
                    "about": [list(dict.fromkeys(f["about"])) for f in facts]})
    return out
