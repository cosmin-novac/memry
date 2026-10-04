"""Who the store owner is: learned from what was said, never judged.

Each namespace has one owner entity, the person the memories belong to.
Without an account name it is called "the user" (``OWNER_PLACEHOLDER``), and
"the user" is a role, not a name. Asked whether that entity and a named person
are one, the identity judge compares two sets of memories, and on a real store
it read the owner (61 memories) and "Cosmin" (363), the person it was, as two
people at P(different) 0.94-0.95; the pair was kept apart for good and the
owner stayed "the user". Who the owner is gets stated instead: the user says
their name, signs a message, is called by it, or a memory says it ("The
user's name is Cos."). This module holds what reads such statements; the
store decides what they name (``MemoryStore.learn_owner_name``).

Three readers, cheapest first:

* extraction reports the user's name when the messages it reads state it
  (``extraction.stated_user_name``), at no extra call;
* a turn in role user that carries a speaker's name names the owner;
* for what was saved before either existed, patterns pick out the stored
  memories and saved turns that may state it (``statements_in``), and one
  text-model call per namespace reads them against the namespace's people
  (``choose_owner``). Without a text model only the patterns that say the
  name outright count, by the matching rule below.

**Which person a stated name is** (``person_for``): the person who carries
that name or alias, ignoring case and accents, when exactly one does. Failing
that, for a stated name of one word: the one person whose name or alias starts
with that word ("Dan" is "Dan Popescu", also beside a "Dana"), else, from three
letters on, the one person with a name or alias whose first word begins with
it ("Cos" is "Cosmin" when no other person's name starts with "Cos"). Two
people who fit at the step that decides are no answer: the owner then takes
the stated name itself and is compared as any named entity is.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..providers.llm import LLM
from .extraction import OWNER_PLACEHOLDER, clean_stated_name, parse_lenient_json

#: Shortest stated name that may stand for a longer one ("Cos" for "Cosmin").
#: Two letters began too many first names to point at one.
SHORT_NAME_LETTERS = 3
#: Most statements one text-model call reads, and most people it is offered.
STATEMENTS_ASKED = 30
PEOPLE_OFFERED = 40
#: Characters of a statement shown or kept as evidence.
STATEMENT_CHARS = 240

# A name as it follows a cue: up to three words, read on until the first word
# that does not start with a capital letter (``_leading_name``).
_NAME = r"([^\W\d_][\w'\-]*(?:[ ][^\W\d_][\w'\-]*){0,3})"
#: Cues after which the text says outright what the user's name is.
_STRONG = [re.compile(cue + _NAME) for cue in (
    r"(?i:\b(?:my|the user'?s|the user\u2019s|user'?s|the owner'?s)\s+"
    r"(?:first\s+|full\s+|real\s+|preferred\s+|nick)?name(?:\s+is|:)\s+)",
    r"(?i:\bthe user\s+(?:is\s+)?(?:called|named|goes by)\s+)",
    r"(?i:\b(?:call me|my name'?s|i go by)\s+)",
)]
#: Cues that may introduce the user's name or someone else's ("I'm Cos", "I'm
#: Romanian", "the user is Cosmin", "the user is German"): read by the text
#: model only.
_WEAK = [re.compile(cue + _NAME) for cue in (
    r"(?i:\b(?:i'?m|i am|this is|the user is)\s+)",
    r"(?i:\bthe user\s*[(,]\s*)",
)] + [re.compile(_NAME + r"(?i:\s*\(\s*the user\s*\))")]
#: An assistant turn that greets or thanks someone by name.
_GREETING = re.compile(
    r"(?i:^\s*(?:hi|hello|hey|dear|thanks|thank you|good (?:morning|afternoon|evening))"
    r"[ ,]+)" + _NAME)


def fold(text: str) -> str:
    """Case and accents left out, as names are compared here."""
    text = unicodedata.normalize("NFKD", (text or "").casefold())
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def _leading_name(raw: str) -> str:
    """The words of ``raw`` up to the first one that does not start with a
    capital letter: "Cos and I" is "Cos"."""
    words: list[str] = []
    for word in raw.split():
        if not word[:1].isupper():
            break
        words.append(word)
    return clean_stated_name(" ".join(words))


def statements_in(text: str, *, role: str = "user") -> list[tuple[str, bool]]:
    """The names ``text`` may state for the user, each with whether it says so
    outright (a strong cue such as "my name is" or "the user's name is").
    ``role`` is who said it: a stored memory or a turn in role user reads
    every cue; an assistant turn is read only for a greeting by name ("Hi
    Cos,"); any other speaker states nobody's name for the user."""
    text = text or ""
    found: list[tuple[str, bool]] = []
    if role == "assistant":
        cues: list[tuple[re.Pattern[str], bool]] = [(_GREETING, False)]
    elif role in ("user", "memory"):
        cues = [(cue, True) for cue in _STRONG] + [(cue, False) for cue in _WEAK]
    else:
        return []
    for cue, strong in cues:
        for match in cue.finditer(text):
            name = _leading_name(match.group(1))
            if name and fold(name) != OWNER_PLACEHOLDER:
                found.append((name, strong))
    return found


def is_correction(text: str, earlier: str, later: str) -> bool:
    """Whether a statement of ``later`` says that ``earlier`` was wrong: it
    names both and says so ("my name is Cosima, not Cosmin", "actually it's
    Cosima"). A later name stated without that is a conflict, not a
    correction, and changes nothing."""
    folded = fold(text)
    return (bool(earlier) and fold(earlier) in folded and fold(later) in folded
            and re.search(r"\b(?:not|actually|correct\w*|wrong|misspel\w*|rather|instead)\b",
                          folded) is not None)


def person_for(name: str, people: Iterable[tuple[str, list[str]]],
               *, short_names: bool = True) -> str | None:
    """The id of the one person ``name`` is, of ``people`` given as (id, names
    and aliases), by the rule in the module docstring; None when no one or
    more than one fits. ``short_names`` False matches whole names only."""
    wanted = fold(name)
    if not wanted:
        return None
    people = [(pid, [fold(n) for n in names if fold(n)]) for pid, names in people]
    exact = [pid for pid, names in people if wanted in names]
    if exact:
        return exact[0] if len(set(exact)) == 1 else None
    if not short_names or len(wanted.split()) != 1:
        return None
    first = {pid for pid, names in people if any(n.split()[0] == wanted for n in names)}
    if first:
        return next(iter(first)) if len(first) == 1 else None
    if len(re.sub(r"\W", "", wanted)) < SHORT_NAME_LETTERS:
        return None
    starting = {pid for pid, names in people
                if any(n.split()[0].startswith(wanted) for n in names)}
    return next(iter(starting)) if len(starting) == 1 else None


@dataclass
class Statement:
    """Something stored or said that may state who the user is."""

    text: str
    #: The name it states by pattern, "" when only the text model can tell.
    name: str = ""
    #: Whether it says outright what the user's name is.
    strong: bool = False
    memory_id: str | None = None
    episode_id: str | None = None
    #: A memory no longer in use: a forgotten duplicate still states the fact.
    forgotten: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items()
                if value not in (None, "", False)}


OWNER_QUESTION = """You read statements from one person's long-term memory store.
"The user" is the person the memories belong to. Some statements may say what
the user's name is: the user gives it, signs with it, is called by it, or a
statement says it. Decide which of the listed people the statements say the
user is.

Answer with "person": a name exactly as listed, or null when the statements do
not say the user is one of them (they name someone else, or a name none of the
listed people carries). Answer with "name": the user's name as the statements
give it, or null when they give none. Never guess from what the people do;
only what the statements say counts.
JSON only: {"person": str|null, "name": str|null}."""

OWNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "person": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
    },
    "required": ["person", "name"],
    "additionalProperties": False,
}


def owner_prompt(statements: list[Statement], people: list[dict[str, Any]]) -> str:
    """The statements and the people (name, other names, memories), as asked."""
    lines = ["Statements:"]
    for statement in statements[:STATEMENTS_ASKED]:
        flag = " (forgotten)" if statement.forgotten else ""
        lines.append(f"- {statement.text[:STATEMENT_CHARS]}{flag}")
    lines.append("")
    lines.append("People in this store:")
    for person in people[:PEOPLE_OFFERED]:
        others = [a for a in person.get("aliases", []) if fold(a) != fold(person["name"])]
        also = f" (also: {', '.join(others[:5])})" if others else ""
        lines.append(f'- {json.dumps(person["name"], ensure_ascii=False)}{also}: '
                     f'{person.get("memories", 0)} memories')
    return "\n".join(lines)


def choose_owner(
    llm: LLM, statements: list[Statement], people: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """One text-model call: which of ``people`` the statements say the user
    is, and the name they give. Returns {"person_id", "person", "name"}
    (person keys None when it names none of them), or None when the model
    gave no usable answer. A person must be named exactly as listed."""
    raw = llm.complete(OWNER_QUESTION, owner_prompt(statements, people),
                       json_schema=OWNER_SCHEMA)
    data = parse_lenient_json(raw)
    if not isinstance(data, dict):
        return None
    chosen = fold(str(data.get("person") or ""))
    person = next((p for p in people[:PEOPLE_OFFERED] if fold(p["name"]) == chosen), None)
    if chosen and person is None:
        return None
    return {"person_id": person["id"] if person else None,
            "person": person["name"] if person else None,
            "name": clean_stated_name(data.get("name"))}


def rule_choice(
    statements: list[Statement], people: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Without a text model: the name the statements that say it outright give
    most often, and the person it is by ``person_for``. None when no
    statement says a name outright."""
    names = Counter(s.name for s in statements if s.strong and s.name)
    if not names:
        return None
    by_fold: dict[str, str] = {}
    for name, _ in names.most_common():
        by_fold.setdefault(fold(name), name)
    counted = Counter()
    for name, count in names.items():
        counted[fold(name)] += count
    name = by_fold[counted.most_common(1)[0][0]]
    person_id = person_for(name, [(p["id"], p.get("aliases", [p["name"]])) for p in people])
    person = next((p for p in people if p["id"] == person_id), None)
    return {"person_id": person_id, "person": person["name"] if person else None,
            "name": name}
