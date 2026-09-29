"""Relational retrieval: reach the memories that similarity cannot.

Experiments (see evals) showed pure hybrid retrieval scores a flat zero on
multi-hop questions - "what tool does Ada use?" is answered by a memory that
names neither "Ada" nor "tool", so no embedder can find it. Following the links
from the query's entities does find it: the "linked" search
(``store._search_linked``) walks them directed and weighted
(``activation_paths``), takes the memories of the entities they reach as
candidates, and scores each by how well it states the property asked times how
strongly it is about the entity the query names (``aboutness``).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from ..backends.base import MemoryBackend
from ..models import Scope

_MIN_SURFACE = 3  # ignore 1-2 char "entities" that would match everything
_POSSESSIVE_END = re.compile("['’]s$")
#: A question in the first person, which is about the store's owner when it
#: names nobody else: "Where do I live?", "the user's shoe size".
_FIRST_PERSON = re.compile(
    r"\b(?:i|me|my|mine|myself)\b|\bthe user\b|\buser['’]s\b", re.IGNORECASE)
_FIRST_PERSON_AS_IT = [
    (re.compile(r"\bmyself\b", re.IGNORECASE), "itself"),
    (re.compile(r"\b(?:my|mine)\b", re.IGNORECASE), "its"),
    (re.compile(r"\b(?:i|me)\b", re.IGNORECASE), "it"),
    (re.compile(r"\bthe user['’]s\b|\buser['’]s\b", re.IGNORECASE), "its"),
    (re.compile(r"\bthe user\b", re.IGNORECASE), "it"),
]


def speaks_in_first_person(query: str) -> bool:
    """Whether a question speaks of the store's owner as "I", "my" or "the
    user"."""
    return bool(_FIRST_PERSON.search(query))


def mask_first_person(query: str) -> str:
    """The question with the owner spoken of as "it", as a memory reads once
    the owner's name is masked: "Where do I live?" -> "Where do it live?"."""
    for pattern, word in _FIRST_PERSON_AS_IT:
        query = pattern.sub(word, query)
    return query


def detect_query_entities(
    backend: MemoryBackend, scope: Scope, query: str, *, cap: int = 128,
    longest: bool = False,
) -> list[str]:
    """Resolve bounded query phrases through canonical and alias candidates.

    Work scales with the query length, not the number of stored entities.
    Longest phrases are tried first so multi-word names remain precise.
    ``longest`` keeps only the longest names found: "bildy v4" and not also
    "bildy", which it contains. A search that starts at "bildy" reaches every
    version below it, so a question about v4 would take v3's memories too.
    """
    # a dot inside a word stays: "VW ID.3", "Node.js"; a full stop does not
    tokens = re.findall(r"[^\W_]+(?:[-'’.][^\W_]+)*", query.lower(), re.UNICODE)
    # "Ilva Marsh's cat" names Ilva Marsh; "McDonald's" names itself. Each
    # phrase is tried as written and without a trailing possessive.
    bare = [_POSSESSIVE_END.sub("", token) for token in tokens]
    phrases: list[str] = []
    seen: set[str] = set()
    for width in range(min(6, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - width + 1):
            for words in (tokens, bare):
                phrase = " ".join(words[start : start + width]).strip()
                if len(phrase) < _MIN_SURFACE or phrase in seen:
                    continue
                seen.add(phrase)
                phrases.append(phrase)
            if len(phrases) >= cap:
                break
        if len(phrases) >= cap:
            break
    if not phrases:
        return []
    found = backend.find_entities_by_aliases(phrases, scope, limit=50)
    if longest:
        names = {entity.id: (entity.normalized or entity.name.lower()) for entity in found}
        found = [entity for entity in found if not any(
            names[entity.id] != other and names[entity.id] in other
            for other in names.values())]
    return [entity.id for entity in found]


# -- links of every kind, weighted --------------------------------------------
# A store can hold a graded answer for every pair it compared: how likely two
# entities are one thing, and how likely one is a version or a part of the
# other (``identity.BELONGS_QUESTION``). Search can then spread from the
# entities a query names the way recall spreads through memory (Collins and
# Loftus 1975): along each link by how much of the linked entity holds for
# this one. What holds depends on the kind and the direction of the link:
#
# * a version's thing: what is true of "bildy" is true of "bildy v4" unless a
#   memory says otherwise, so a question about v4 takes bildy's memories;
# * a thing's versions and parts: their memories are about the thing, so a
#   question about bildy takes them too;
# * a part's whole: context only, since what is true of a company is not true
#   of its dye lab;
# * up to a thing and down again reaches a sibling: v3's memories answer
#   little about v4, so such a path is held down.
#
# Weights multiply along a path, so how far a search reaches follows from the
# links themselves rather than from a fixed number of hops.

#: Factor per kind and direction of a link.
UP_KIND = 0.8
DOWN_KIND = 0.7
UP_PART = 0.5
DOWN_PART = 0.7
#: A relation carries little of what is true of an entity: Ada works on Project
#: X, but "Project X is written in Rust" says nothing about Ada. Its memories are
#: still candidates (the tools Ada uses are among Project X's), at this weight.
LINKED_RELATION = 0.5
#: Extra factor for a step down after a step up: a sibling.
TURN = 0.25
#: Activation under this is not followed further.
FLOOR = 0.05
#: How much a memory that names only entities the links do not reach is about
#: what the query names (``aboutness``): a sibling version, another product or
#: another person may match the text, but it is about something else.
LOW = 0.3


@dataclass(frozen=True)
class Link:
    """One link: for "kind" and "part" ``child`` belongs to ``parent``; for
    "same" and "relation" the two ends are equal."""

    child: str
    parent: str
    kind: str
    p: float


def links_of(backend: MemoryBackend, entity_ids: list[str]) -> list[Link]:
    """Extracted relations and compared pairs touching these entities, every
    answer with its probability.

    None reaches a tag (a topic entity): a tag is never what the linked
    search is about, as it is never a seed (``MemoryStore._is_hub``), so an
    open pair of a thing and the tag of its name ("Groceries" and
    "groceries", a "same" link at 0.5 until judged) does not draw the tag's
    memories into every question naming the thing."""
    relations = backend.relations_of(entity_ids)
    pairs = []
    for proposal in backend.proposals_of(entity_ids):
        a = backend.resolve_entity_id(proposal.entity_a)
        b = backend.resolve_entity_id(proposal.entity_b)
        if a is not None and b is not None and a != b:
            pairs.append((proposal, a, b))
    tags = backend.topic_ids({end for r in relations for end in (r.subject, r.object)}
                             | {end for _, a, b in pairs for end in (a, b)})
    links = [Link(r.subject, r.object, "relation", 1.0)
             for r in relations if r.subject not in tags and r.object not in tags]
    for proposal, a, b in pairs:
        if a in tags or b in tags:
            continue
        belongs = proposal.belongs or {}
        # A version scores P(same) of about 0.75: the names and facts are
        # close because one belongs to the other, which the belongs answer
        # already says. Only the share of "neither" is left for "same".
        same = proposal.confidence * belongs.get("neither", 1.0)
        if proposal.status == "proposed" and same >= FLOOR:
            links.append(Link(a, b, "same", same))
        for child, parent, side in ((a, b, "a"), (b, a, "b")):
            other = "b" if side == "a" else "a"
            for kind in ("kind", "part"):
                p = belongs.get(f"{side}_{kind}_of_{other}", 0.0)
                if p >= FLOOR:
                    links.append(Link(child, parent, kind, p))
    return links


def _steps(link: Link, node: str):
    """(other end, factor, step up, step down) for following ``link`` from ``node``."""
    if node not in (link.child, link.parent):
        return
    other = link.parent if node == link.child else link.child
    if link.kind in ("relation", "same"):
        factor = LINKED_RELATION if link.kind == "relation" else 1.0
        yield other, factor * link.p, False, False
    elif node == link.child:
        yield other, (UP_KIND if link.kind == "kind" else UP_PART) * link.p, True, False
    else:
        yield other, (DOWN_KIND if link.kind == "kind" else DOWN_PART) * link.p, False, True


def activation_paths(
    backend: MemoryBackend, seeds: list[str], *, depth: int = 1,
) -> tuple[dict[str, float], set[str]]:
    """How strongly each entity within ``depth`` links of the seeds bears on a
    query about the seeds (1.0 for a seed, the product of the factors along
    the best path for the rest), and the entities whose best path took a step
    up (the thing or the whole a seed belongs to, and siblings through them):
    what is true of those holds for a seed only where the seed says nothing
    else."""
    best: dict[str, float] = {seed: 1.0 for seed in seeds}
    up: set[str] = set()
    # A path is a state as well as a place: whether it has gone up to a thing
    # decides whether a step down reaches a sibling.
    seen: dict[tuple[str, bool], float] = {(seed, False): 1.0 for seed in seeds}
    frontier = dict(seen)
    for _ in range(max(depth, 0)):
        if not frontier:
            break
        links = links_of(backend, sorted({node for node, _ in frontier}))
        reached: dict[tuple[str, bool], float] = {}
        for (node, went_up), act in frontier.items():
            for link in links:
                for other, factor, step_up, step_down in _steps(link, node):
                    if step_down and went_up:
                        factor *= TURN
                    value = act * factor
                    state = (other, went_up or step_up)
                    if value >= FLOOR and value > reached.get(state, 0.0):
                        reached[state] = value
        frontier = {}
        for state, value in reached.items():
            if value > seen.get(state, 0.0):
                seen[state] = value
                frontier[state] = value
                strongest = best.get(state[0], 0.0)
                # of two paths as strong, the one without a step up counts,
                # whichever was found first
                if value > strongest or (value == strongest and not state[1]):
                    best[state[0]] = value
                    if state[1]:
                        up.add(state[0])
                    else:
                        up.discard(state[0])
    return best, up - set(seeds)


# -- "linked" search: which entity from the links, which property from the text --
# A memory answers a question when it is about an entity the answer can come
# from and it states the property asked about. The links say the first: how
# strongly the memory's entity is linked to the one the query names. The text
# says the second, once the entity names are out of it: "Where did Tovel Forum
# 2024 take place?" was closest to "Tovel Forum 2024 had 420 attendees" only
# because both say "Tovel Forum 2024". With names replaced by "it", "Where did
# it take place?" is closest to "It takes place in Lisbon", the forum's memory,
# which the 2024 edition inherits. See the PhD notes, relative-retrieval.

#: Entities linked at least this strongly have their memories that best state
#: the property asked searched as well: ``FAMILY_TOP`` of them each, chosen
#: among their ``FAMILY_SCAN`` newest.
FAMILY_MIN = 0.3
FAMILY_TOP = 10
FAMILY_SCAN = 500
#: A memory that names only entities the links do not reach is about something
#: else (``LOW``), and so, as measured, is one that names none: at 0.5, notes
#: naming nothing outranked a product's own parts on "What do I know about
#: it?" (roll-up recall 0.78 against 1.00). A memory about the entity a query
#: names names it.
NO_ENTITY = LOW

_POSSESSIVE = "(?:'s|\u2019s)?"


#: Judging (``store._judge_ranking``). A question whose answer is one memory
#: is answered from the first call: reading on while nothing scored 0.5 (Jev:
#: answers 0.54 to 0.86, non-answers 0.02 to 0.13) found nothing 5 of 5 times
#: it ran. One that needs several (SET_BAR on its "needs several memories"
#: answer: sets 0.73 to 0.90, one-answer 0.09 to 0.47) has one more call, on
#: the memories filed under the topics the first ones share
#: (``store._set_pool``); its members are the top tier of both calls' scores
#: (``set_members``), returned past the limit up to SET_RESULT_CAP.
SET_BAR = 0.5
MEMBER_FLOOR = 0.07  # non-members of the traced sets scored 0.06 or less
SET_RESULT_CAP = 100
#: A topic (a tag) is shared by the first judged memories when at least this
#: many of them carry it; each shared topic's newest ``SET_SCAN`` memories are
#: candidates for the second call.
SET_SHARED = 2
SET_SCAN = 500
#: Of the candidates tied at the cut of the second call's budget, as many as
#: places are left and this many more are scored to break the tie; the rest
#: are cut first by the share and the newest (``store._set_pool``).
SET_TIE_MARGIN = 20
#: Where the topics leave places in the second call, the memories nearest the
#: members found fill them: half by memory vector, half by property vector
#: among this many nearest by memory vector (``store._nearest_unjudged``).
SET_NEAREST = 200


def set_members(judged: dict[str, float]) -> set[str]:
    """The memories of a set question that belong to the set, from Jev's scores
    alone. Its scale differs by question: a car's price scores 0.08 to 0.16
    for "Which car is the cheapest?" beside insurance costs at 0.02 to 0.06; a
    liked restaurant 0.42 to 0.64 for "Which restaurants did I like?" beside
    "had lunch at" at 0.15 to 0.25 and noise at 0.02 to 0.05. The members are
    the top tier: the scores split in two where they separate best on a log
    scale (Otsu's threshold), the upper group is kept while it stands at
    least twice above the lower, and split again. Scores with no such split
    are all members or all noise, by ``MEMBER_FLOOR`` (a first call can hold
    nothing but the set: 20 of 21 test drives)."""
    import math

    tier = sorted((max(v, 0.01), mid) for mid, v in judged.items())
    while len(tier) > 1:
        logs = [math.log(v) for v, _ in tier]
        best, cut = -1.0, None
        for i in range(1, len(logs)):
            low, high = logs[:i], logs[i:]
            gap = sum(high) / len(high) - sum(low) / len(low)
            spread = len(low) * len(high) * gap ** 2
            if spread > best:
                best, cut = spread, (i, gap)
        if cut[1] < math.log(2):
            break
        tier = tier[cut[0]:]
    return {mid for v, mid in tier if v >= MEMBER_FLOOR}


#: A thing an entity more likely than not belongs to reads "it" in that
#: entity's memories too ("The first release of bildy" in bildy v1's).
HOME_P = 0.5


def homes_of(backend: MemoryBackend, entity_ids: list[str]) -> dict[str, set[str]]:
    """The entities each of these is a kind or a part of at ``HOME_P`` or more."""
    homes: dict[str, set[str]] = {entity_id: set() for entity_id in entity_ids}
    ids = sorted(homes)
    for start in range(0, len(ids), 400):
        for link in links_of(backend, ids[start:start + 400]):
            if link.kind in ("kind", "part") and link.p >= HOME_P and link.child in homes:
                homes[link.child].add(link.parent)
    return homes


def parts_of(backend: MemoryBackend, entity_ids: list[str]) -> set[str]:
    """The entities that are a kind or a part of any of these at ``HOME_P``
    or more (``homes_of`` the other way round): their memories read these
    names as "it" too."""
    wholes = set(entity_ids)
    return {link.child for link in links_of(backend, sorted(wholes))
            if link.kind in ("kind", "part") and link.p >= HOME_P and link.parent in wholes}


def mask_names(text: str, names: Iterable[str], keep: Iterable[str] = ()) -> str:
    """``text`` with each of ``names`` replaced by "it" ("its" for a
    possessive), matched as whole words in any case, the longest name first.
    A name in ``keep`` stays as written, and so does a shorter name inside it
    ("Bildy Bakery" keeps its "bildy")."""
    masked = {n.strip().lower() for n in names if n and n.strip()}
    every = masked | {n.strip().lower() for n in keep if n and n.strip()}
    if not masked:
        return text
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(every, key=len, reverse=True))
        + r")(?!\w)(" + _POSSESSIVE + ")", re.IGNORECASE)

    def swap(match: re.Match) -> str:
        if match.group(1).lower() not in masked:
            return match.group(0)
        return "its" if match.group(2) else "it"

    return pattern.sub(swap, text)


def aboutness(activations: list[float | None]) -> float:
    """How strongly a memory is about what the query names, from the
    activation of each entity it names (None for an entity the links do not
    reach): the strongest linked one, ``LOW`` if it names only unlinked ones,
    ``NO_ENTITY`` if it names none. A link, however weak, never ranks below no
    link: a version Jev linked at 0.4 (0.28 on the way down) is still more
    likely about the product than a memory about something else."""
    linked = [a for a in activations if a is not None]
    if linked:
        return max(max(linked), LOW)
    return LOW if activations else NO_ENTITY
