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
"""

from __future__ import annotations

import re
from typing import Any

from ..models import Memory
from ..providers.llm import LLM
from .extraction import OWNER_PLACEHOLDER, parse_lenient_json
from .reconcile import CONFLICT_KEY

SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
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
- Keep each time as the memory writes it: every fact keeps the date the
  memory was said, so a relative time ("last year") stays as it is.
- If the memory states one fact, return it as the only item, unchanged.

Respond with JSON only: {"facts": [str, ...]}"""

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


def split_prompt(memory: Memory) -> str:
    """The memory alone. Shown with the date it was said, the model wrote
    that date into every fact, where it reads as the time of the fact; each
    fact keeps the memory's dates anyway."""
    return f"Memory:\n{memory.content}\n\nSplit it as JSON."


def split_facts(llm: LLM, memory: Memory) -> list[str]:
    """The single facts the text model reads in a memory, in its order, empty
    ones and repeats left out. One fact back means the memory states one."""
    parsed = parse_lenient_json(llm.complete(SPLIT_SYSTEM, split_prompt(memory),
                                             json_schema=SPLIT_SCHEMA))
    facts = parsed.get("facts") if isinstance(parsed, dict) else None
    if not isinstance(facts, list):
        raise ValueError("the text model answered no list of facts")
    out: list[str] = []
    for fact in facts:
        text = " ".join(str(fact or "").split())
        if text and text not in out:
            out.append(text)
    return out


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
