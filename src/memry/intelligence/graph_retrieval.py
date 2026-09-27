"""Relational retrieval: reach the memories that similarity cannot.

Experiments (see evals) showed pure hybrid retrieval scores a flat zero on
multi-hop questions - "what tool does Ada use?" is answered by a memory that
names neither "Ada" nor "tool", so no embedder can find it. Following typed
relations from the query's entities does find it, reliably, at any store size.

The move is deliberately ranked by graph distance, not by text similarity: the
multi-hop answer is relevant *because* it is two typed hops from the query, even
though it shares no words with it. Those candidates are then fused (RRF) with the
hybrid results, so direct lookups keep their strong lexical/semantic ranking and
relational questions gain the hop-reachable memories on top.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from ..backends.base import MemoryBackend
from ..models import Scope

_MIN_SURFACE = 3  # ignore 1-2 char "entities" that would match everything


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
    tokens = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", query.lower(), re.UNICODE)
    phrases: list[str] = []
    seen: set[str] = set()
    for width in range(min(6, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[start : start + width]).strip()
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


def expand_entities(
    backend: MemoryBackend, seeds: list[str], *, hops: int = 2
) -> list[str]:
    """Entities reachable from the seeds over typed relations, nearest first.

    Falls back to co-occurrence (entities sharing a memory) only if no typed
    relation touches the seeds, mirroring the PPR-as-fallback result: typed
    edges when present, structural proximity otherwise."""
    reached: set[str] = set(seeds)
    order: list[str] = []
    frontier = set(seeds)
    for _ in range(hops):
        rels = backend.relations_of(list(frontier))
        nxt: set[str] = set()
        for r in rels:
            for endpoint in (r.subject, r.object):
                if endpoint not in reached:
                    nxt.add(endpoint)
        for e in nxt:
            reached.add(e)
            order.append(e)
        frontier = nxt
        if not frontier:
            break
    if not order:
        # No typed edges (e.g. memories predating relation extraction): fall back
        # to co-occurrence proximity. The benchmark showed localized PageRank over
        # co-occurrence is the reliable relation-free option (~0.90, scale-stable),
        # so weight neighbours by how strongly they co-occur with the seeds rather
        # than dumping a flat pool.
        order = _cooccurrence_expand(backend, seeds, hops=hops)
    return order


def _cooccurrence_expand(
    backend: MemoryBackend, seeds: list[str], *, hops: int, per_entity: int = 25
) -> list[str]:
    """Entities near the seeds by shared-memory co-occurrence, ranked by a
    localized PageRank so the strongest structural neighbours come first."""
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reached: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: set[str] = set()
        for e in frontier:
            for mem in backend.entity_memories(e, limit=per_entity):
                others = [x.id for x in backend.entities_of_memory(mem.id)]
                for a in others:
                    for b in others:
                        if a != b:
                            adj[a][b] += 1.0
                    if a not in reached:
                        nxt.add(a)
        reached |= nxt
        frontier = nxt
        if not frontier:
            break
    nodes = list(reached | set(adj))
    if not nodes:
        return []
    # weighted personalized PageRank, localized to this neighbourhood
    idx = {n: i for i, n in enumerate(nodes)}
    seed_mass = 1.0 / max(len(seeds), 1)
    rank = {n: (seed_mass if n in seeds else 0.0) for n in nodes}
    alpha = 0.85
    for _ in range(20):
        nxt_rank = {n: (1 - alpha) * (seed_mass if n in seeds else 0.0) for n in nodes}
        for n in nodes:
            out = adj.get(n)
            if not out:
                continue
            total = sum(out.values())
            share = alpha * rank[n] / total
            for m, w in out.items():
                if m in idx:
                    nxt_rank[m] += share * w
        rank = nxt_rank
    ranked = sorted((n for n in nodes if n not in seeds), key=lambda n: -rank[n])
    return ranked


def relational_memory_ids(
    backend: MemoryBackend,
    scope: Scope,
    query: str,
    *,
    hops: int = 2,
    per_entity: int = 25,
) -> list[str]:
    """Ordered memory ids reached by traversing relations from the query's
    entities. Empty when the query names no known entity."""
    seeds = detect_query_entities(backend, scope, query)
    if not seeds:
        return []
    expanded = expand_entities(backend, seeds, hops=hops)
    ids: list[str] = []
    seen: set[str] = set()
    # expanded (hop-reachable) entities first - those carry the non-obvious,
    # multi-hop answers; the seed's own memories come after (hybrid already has
    # them covered).
    for entity_id in expanded + seeds:
        for mem in backend.entity_memories(entity_id, limit=per_entity):
            if mem.id not in seen:
                seen.add(mem.id)
                ids.append(mem.id)
    return ids


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
RELATION = 0.9
#: Extra factor for a step down after a step up: a sibling.
TURN = 0.25
#: Activation under this is not followed further.
FLOOR = 0.05
#: "weighted" fusion: a memory whose entities the query's links reach at this
#: activation or more keeps its text score, so the text decides between a
#: version's own facts and its thing's. A memory about entities the links do
#: not reach (another product, a sibling version, another person) drops
#: towards ``LOW``: its text may match, but it is about something else. A
#: memory that names no entity keeps its score. Lifting the linked memories
#: instead measured worse: text scores are close together, so any lift put
#: every memory of the named entity above the one that answered.
STRONG = 0.5
LOW = 0.3


def specificity(activations: list[float]) -> float:
    """How specific a memory is to what the query names: 1.0 for the entity
    it names, the activation for a thing it belongs to, ``LOW`` and up for
    entities the links do not reach. "gated" fusion puts the most specific of
    the memories that answer the question first: a version's own fact
    overrides its thing's, as a default does in an inheritance hierarchy, and
    only where the version has a fact on what was asked."""
    if not activations:
        return LOW
    strongest = max(activations)
    return strongest if strongest >= STRONG else LOW + (1.0 - LOW) * strongest / STRONG


def link_factor(activations: list[float]) -> float:
    """What a memory's text score is multiplied by in "weighted" fusion, from
    the activations of the entities it names (empty: it names none)."""
    if not activations:
        return 1.0
    strongest = max(activations)
    if strongest >= STRONG:
        return 1.0
    return LOW + (1.0 - LOW) * strongest / STRONG
#: The modes ``relational_mode`` takes besides "typed".
LINK_MODES = ("undirected", "directed")


@dataclass(frozen=True)
class Link:
    """One link: for "kind" and "part" ``child`` belongs to ``parent``; for
    "same" and "relation" the two ends are equal."""

    child: str
    parent: str
    kind: str
    p: float


def links_of(backend: MemoryBackend, entity_ids: list[str], *, graded: bool) -> list[Link]:
    """Extracted relations and compared pairs touching these entities. Graded:
    every answer with its probability. Otherwise only a version or a part at
    ``identity.BELONGS_BAR``, as the structure pass reads it."""
    from .identity import BELONGS_BAR, belonging

    links = [Link(r.subject, r.object, "relation", 1.0)
             for r in backend.relations_of(entity_ids)]
    for proposal in backend.proposals_of(entity_ids):
        a = backend.resolve_entity_id(proposal.entity_a)
        b = backend.resolve_entity_id(proposal.entity_b)
        if a is None or b is None or a == b:
            continue
        belongs = proposal.belongs or {}
        if graded:
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
            continue
        side, p = belonging(belongs)
        if side is None or p < BELONGS_BAR:
            continue
        child, parent = (a, b) if side == "a" else (b, a)
        other = "b" if side == "a" else "a"
        kind = ("kind" if belongs.get(f"{side}_kind_of_{other}", 0.0)
                >= belongs.get(f"{side}_part_of_{other}", 0.0) else "part")
        links.append(Link(child, parent, kind, 1.0))
    return links


def _steps(link: Link, node: str, directed: bool, relation: float = RELATION):
    """(other end, factor, step up, step down) for following ``link`` from ``node``."""
    if node not in (link.child, link.parent):
        return
    other = link.parent if node == link.child else link.child
    if not directed:
        yield other, RELATION, False, False
    elif link.kind in ("relation", "same"):
        yield other, (relation if link.kind == "relation" else 1.0) * link.p, False, False
    elif node == link.child:
        yield other, (UP_KIND if link.kind == "kind" else UP_PART) * link.p, True, False
    else:
        yield other, (DOWN_KIND if link.kind == "kind" else DOWN_PART) * link.p, False, True


def activation(
    backend: MemoryBackend, seeds: list[str], *, depth: int = 2, mode: str = "directed",
    relation: float = RELATION,
) -> dict[str, float]:
    """How strongly each entity near the seeds bears on a query about the
    seeds: 1.0 for a seed, the product of the factors along the best path for
    the rest. "undirected" follows relations and version and part links at the
    belongs bar alike, 0.9 a step, as relations were always followed."""
    directed = mode == "directed"
    best: dict[str, float] = {seed: 1.0 for seed in seeds}
    # A path is a state as well as a place: whether it has gone up to a thing
    # decides whether a step down reaches a sibling.
    seen: dict[tuple[str, bool], float] = {(seed, False): 1.0 for seed in seeds}
    frontier = dict(seen)
    for _ in range(max(depth, 0)):
        if not frontier:
            break
        links = links_of(backend, sorted({node for node, _ in frontier}), graded=directed)
        reached: dict[tuple[str, bool], float] = {}
        for (node, went_up), act in frontier.items():
            for link in links:
                for other, factor, up, down in _steps(link, node, directed, relation):
                    if down and went_up:
                        factor *= TURN
                    value = act * factor
                    state = (other, went_up or up)
                    if value >= FLOOR and value > reached.get(state, 0.0):
                        reached[state] = value
        frontier = {}
        for state, value in reached.items():
            if value > seen.get(state, 0.0):
                seen[state] = value
                frontier[state] = value
                best[state[0]] = max(best.get(state[0], 0.0), value)
    return best


def linked_memories(
    backend: MemoryBackend, scope: Scope, query: str, *, depth: int = 2,
    mode: str = "directed", per_entity: int = 25,
) -> tuple[list[str], dict[str, float]]:
    """Memory ids reached from the query's entities, strongest link first and
    the query's own entities last (the text ranking has those), and the
    activation of every entity reached. Empty when the query names none."""
    seeds = detect_query_entities(backend, scope, query, longest=True)
    if not seeds:
        return [], {}
    act = activation(backend, seeds, depth=depth, mode=mode)
    order = sorted((e for e in act if e not in seeds), key=lambda e: -act[e]) + seeds
    ids: list[str] = []
    seen: set[str] = set()
    for entity_id in order:
        for memory in backend.entity_memories(entity_id, limit=per_entity):
            if memory.id not in seen:
                seen.add(memory.id)
                ids.append(memory.id)
    return ids, act


def inherited_questions(
    backend: MemoryBackend, scope: Scope, query: str
) -> list[tuple[str, str]]:
    """(the query asked of a thing, that thing's id) for every thing an entity
    the query names is a version, occurrence or item of: "Which platforms does
    bildy v4 run on?" is also "Which platforms does bildy run on?", since what
    is true of bildy holds for v4 unless a memory says otherwise. People
    answer "can a canary fly?" by way of "bird" (Collins and Quillian 1969).
    A part inherits nothing, so a part's whole is not asked."""
    seeds = detect_query_entities(backend, scope, query, longest=True)
    asked: list[tuple[str, str]] = []
    for link in links_of(backend, seeds, graded=True):
        if link.kind != "kind" or link.child not in seeds or UP_KIND * link.p < STRONG:
            continue
        child, parent = backend.get_entity(link.child), backend.get_entity(link.parent)
        if child is None or parent is None:
            continue
        found = re.search(re.escape(child.name), query, re.IGNORECASE)
        if found:
            asked.append((query[:found.start()] + parent.name + query[found.end():], parent.id))
    return asked


# -- "linked" search: which entity from the links, which property from the text --
# A memory answers a question when it is about an entity the answer can come
# from and it states the property asked about. The links say the first: how
# strongly the memory's entity is linked to the one the query names. The text
# says the second, once the entity names are out of it: "Where did Tovel Forum
# 2024 take place?" was closest to "Tovel Forum 2024 had 420 attendees" only
# because both say "Tovel Forum 2024". With names replaced by "it", "Where did
# it take place?" is closest to "It takes place in Lisbon", the forum's memory,
# which the 2024 edition inherits. See the PhD notes, relative-retrieval.

#: A relation carries little of what is true of an entity: Ada works on Project
#: X, but "Project X is written in Rust" says nothing about Ada. Its memories are
#: still candidates (the tools Ada uses are among Project X's), at this weight.
LINKED_RELATION = 0.5
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


def mask_names(text: str, names: Iterable[str]) -> str:
    """``text`` with each of ``names`` replaced by "it" ("its" for a
    possessive), longest first, matched as whole words in any case."""
    for name in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)" + _POSSESSIVE, re.IGNORECASE)
        text = pattern.sub(lambda m: "its" if m.group(0)[len(name):] else "it", text)
    return text


def aboutness(activations: list[float | None]) -> float:
    """How strongly a memory is about what the query names, from the
    activation of each entity it names (None for an entity the links do not
    reach): the strongest linked one, ``LOW`` if it names only unlinked ones,
    ``NO_ENTITY`` if it names none."""
    linked = [a for a in activations if a is not None]
    if linked:
        return max(linked)
    return LOW if activations else NO_ENTITY
