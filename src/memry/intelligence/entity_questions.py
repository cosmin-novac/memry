"""Entity questions: the questions the owner would ask about an entity related
to them, in their own words and without its name ("Where does my sister
work?"), kept beside the entity as search keys (``retrieval.entity_questions``).

A question about the owner's sister contains no name, so no hub is in it.
It is in the first person, so Memry starts the search from the owner, and
the owner's own fact ("Ilva Marsh works at Nordlicht GmbH") comes before
the sister's. With entity questions, Memry starts from the sister when two
conditions hold. The question contains a role word after "my" ("sister")
that only the sister's questions contain. Its best cosine similarity with
her questions is at least ``retrieval.entity_question_bar``. The question
is then read with "my sister" as "it", as a question about a named entity
is read with the name as "it".

Here: the rule for the role words, the masking, and the writer's prompt.
Writing: ``MemoryStore.write_entity_questions``. Storage: the table
``entity_questions``. Reading: ``MemoryStore._role_seed``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .questions import QUESTIONS_PER_MEMORY, clean_questions

#: The owner as a question speaks of them before a role: "my sister", "the
#: user's manager".
_OWNERS = r"(?:my|the\s+user['’]s|user['’]s)"
_ROLE = re.compile(_OWNERS + r"\s+([^\W\d_]+)(?:['’]s)?\b", re.IGNORECASE)


def role_words(question: str) -> set[str]:
    """The words right after "my" (or "the user's") in a question, lower
    case, a possessive taken off: {"sister"} for "When is my sister's
    birthday?". "Where do I work?" has none."""
    return {m.group(1).casefold() for m in _ROLE.finditer(question)}


def mask_role(question: str, word: str) -> str:
    """The question with "my <word>" read as "it" and "my <word>'s" as "its",
    as a memory about that entity is read once its name is masked: "When is
    my sister's birthday?" -> "When is its birthday?"."""
    pattern = re.compile(_OWNERS + r"\s+" + re.escape(word) + r"(['’]s)?\b", re.IGNORECASE)
    return pattern.sub(lambda m: "its" if m.group(1) else "it", question)


def distinct_holders(rows: list[tuple[str, str]]) -> dict[str, str]:
    """Role word -> the one entity whose questions use it, from (entity id,
    question) rows. A word in the questions of two entities is left out."""
    holders: dict[str, set[str]] = {}
    for entity_id, text in rows:
        for word in role_words(text):
            holders.setdefault(word, set()).add(entity_id)
    return {word: next(iter(ids)) for word, ids in holders.items() if len(ids) == 1}


#: What the text model is asked: for each entity, the questions its owner
#: would ask about it by its role.
ENTITY_QUESTIONS_SYSTEM = """You write the questions a person would ask about \
someone or something in their life without saying its name.

The person is {owner}. For each numbered entity you get its name, its \
description and how it relates to {owner}. Write 2 or 3 questions {owner} \
would ask about it later, in the first person, by its role and never by its \
name: "Who is my sister?", "Where does my manager live?". Put the role word \
right after "my". Each question asks for one thing the description says. Give \
[] for an entity {owner} would always call by its name.

Answer as JSON: {{"items": [{{"n": <the entity's number>, "questions": [str]}}]}}, \
one item per entity."""

ENTITY_QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"},
                       "questions": {"type": "array", "items": {"type": "string"}}},
        "required": ["n", "questions"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


#: Entity metadata key: a hash of what the writer was given for the entity
#: (``entity_questions_input``) the last time it was asked about it, with or
#: without questions as the answer. A different hash means its description or
#: relations changed since, so upkeep asks about it again.
ENTITY_QUESTIONS_CHECKED_KEY = "questions_checked"

#: Output tokens a call is estimated at per entity: up to 3 short questions in
#: JSON.
ANSWER_TOKENS_PER_ENTITY = 60


def entity_questions_input(owner: str, entity: dict[str, Any]) -> str:
    """A hash of what the writer is given for one entity: the owner's name,
    its name, description and relations. Equal hashes mean an equal prompt."""
    payload = json.dumps([owner, entity["name"],
                          " ".join(str(entity.get("description") or "").split()),
                          sorted(entity.get("relations") or [])], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def entity_questions_prompt(owner: str, entities: list[dict[str, Any]]) -> tuple[str, str]:
    """The system and user text of one call about ``entities``."""
    listing = "\n\n".join(
        f"{n}. {e['name']}\nDescription: {' '.join(str(e.get('description') or '').split())}\n"
        f"Relations: {'; '.join(e.get('relations') or []) or 'none'}"
        for n, e in enumerate(entities, 1))
    return (ENTITY_QUESTIONS_SYSTEM.format(owner=owner),
            f"Entities:\n{listing}\n\nAnswer for every number as JSON.")


def estimate_call_tokens(owner: str, entities: list[dict[str, Any]]) -> dict[str, int]:
    """Estimated tokens of one call about ``entities``: the prompt at four
    characters a token, the answer at ``ANSWER_TOKENS_PER_ENTITY`` each."""
    from .context import estimate_tokens

    system, user = entity_questions_prompt(owner, entities)
    return {"input": estimate_tokens(system) + estimate_tokens(user),
            "output": ANSWER_TOKENS_PER_ENTITY * len(entities)}


def write_entity_questions(llm: Any, owner: str, entities: list[dict[str, Any]]) -> list[list[str]]:
    """The questions ``owner`` would ask about each of ``entities`` (each a
    dict with "name", "description" and "relations", a list of lines such as
    "Ilva Marsh has_sister Mira Lund"), in one call: one list per entity, in
    order. A question without a role word after "my", or with the entity's
    name in it, is dropped. A failing call raises."""
    out: list[list[str]] = [[] for _ in entities]
    if not entities:
        return out
    from .extraction import parse_lenient_json  # extraction imports questions

    system, user = entity_questions_prompt(owner, entities)
    raw = llm.complete(system, user, json_schema=ENTITY_QUESTIONS_SCHEMA)
    parsed = parse_lenient_json(raw)
    items = parsed.get("items") if isinstance(parsed, dict) else parsed
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(entities):
            continue
        name = str(entities[n - 1]["name"]).casefold()
        out[n - 1] = [q for q in clean_questions(item.get("questions"))
                      if role_words(q) and name not in q.casefold()][:QUESTIONS_PER_MEMORY]
    return out
