"""Entity structure: which names are hubs, where a part belongs, and what a
shared name means.

Measured on a real store: 3,314 entities over 985 memories, of which 2,073
appeared in exactly one memory. Most extracted "entities" are one-off noun
phrases, and treating every one as a hub is what filled the map with noise and
the queue with questions nobody can answer ("round 1 and Round 1?").

Three rules, all computed from what the store already holds, none of which
deletes anything:

* **Hub status is earned.** A name is a hub when it is an anchor type, or two
  memories mention it, or a relation involves it. Everything else stays a
  phrase on its memory: searchable, linked, and promoted on second sighting
  because the status is recomputed, never stored.
* **Home is one level, and derived.** A part belongs to the project, product or
  organization it keeps appearing with. An explicit ``part_of`` relation wins;
  otherwise one anchor has to share most of the part's memories.
* **A shared name is read through home.** Same name under the same home is the
  same thing. Same name under different homes is two things and no question.
  People are never merged on a name alone.

Everything here is pure: lists in, plans out. The store applies the plans, so
a dry run is the same code with the last step left out.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

#: Types that are hubs on their own: the things a store is about.
ANCHOR_TYPES = frozenset({"person", "organization", "project", "product", "place"})
#: Types a part can belong to. A person or a place is never a home.
HOME_TYPES = frozenset({"project", "product", "organization"})
#: Predicates that state membership outright.
PART_PREDICATES = frozenset({
    "part_of", "belongs_to", "feature_of", "component_of", "module_of",
    "section_of", "included_in", "subproject_of", "version_of",
})

#: Share of a part's memories one anchor must appear in to be its home.
HOME_MIN_SHARE = 0.7
#: An anchor seen fewer times than this is too thin to be anyone's home.
HOME_MIN_ANCHOR_MEMORIES = 3


@dataclass(frozen=True)
class Node:
    """The little the structure rules need to know about one entity."""

    id: str
    name: str
    normalized: str
    entity_type: str | None
    memories: int = 0
    relations: int = 0
    created_at: str = ""


def is_hub(entity_type: str | None, memories: int, relations: int) -> bool:
    return (entity_type in ANCHOR_TYPES and memories >= 1) or memories >= 2 or relations >= 1


def hub_reason(entity_type: str | None, memories: int, relations: int) -> str:
    if entity_type in ANCHOR_TYPES and memories >= 1:
        return f"a {entity_type}"
    if memories >= 2:
        return f"in {memories} memories"
    if relations >= 1:
        return "in a relation"
    return ""


def derive_homes(
    nodes: Iterable[Node],
    links: Iterable[tuple[str, str]],
    relations: Iterable[tuple[str, str, str]],
    *,
    min_share: float = HOME_MIN_SHARE,
    min_anchor_memories: int = HOME_MIN_ANCHOR_MEMORIES,
) -> dict[str, dict[str, Any]]:
    """Where each part belongs: ``{entity_id: {"id", "share", "source"}}``.

    ``links`` are (entity_id, memory_id) over active memories; ``relations``
    are (subject, predicate, object). One level only: a home has no home, so a
    chain cannot form and nothing needs a tree.
    """
    by_id = {node.id: node for node in nodes}
    entity_memories: dict[str, set[str]] = defaultdict(set)
    memory_entities: dict[str, set[str]] = defaultdict(set)
    for entity_id, memory_id in links:
        if entity_id in by_id:
            entity_memories[entity_id].add(memory_id)
            memory_entities[memory_id].add(entity_id)

    def can_be_home(entity_id: str) -> bool:
        node = by_id.get(entity_id)
        return (
            node is not None
            and node.entity_type in HOME_TYPES
            and len(entity_memories[entity_id]) >= min_anchor_memories
        )

    homes: dict[str, dict[str, Any]] = {}
    # 1. a stated membership wins over anything inferred
    for subject, predicate, obj in relations:
        if predicate in PART_PREDICATES and subject in by_id and subject != obj:
            target = by_id.get(obj)
            if target is not None and target.entity_type in HOME_TYPES:
                homes.setdefault(subject, {"id": obj, "share": 1.0, "source": "relation"})

    # 2. otherwise the anchor it keeps appearing with
    for node in by_id.values():
        if node.id in homes or node.entity_type in ANCHOR_TYPES:
            continue
        memories = entity_memories[node.id]
        if not memories:
            continue
        counts: Counter[str] = Counter()
        for memory_id in memories:
            for other in memory_entities[memory_id]:
                if other != node.id and can_be_home(other):
                    counts[other] += 1
        if not counts:
            continue
        ranked = counts.most_common(2)
        top, shared = ranked[0]
        share = shared / len(memories)
        if share < min_share:
            continue
        # two anchors tied for the top cannot both be home; leave it unhomed
        if len(ranked) > 1 and ranked[1][1] == shared:
            continue
        homes[node.id] = {"id": top, "share": round(share, 2), "source": "co-mention"}

    # one level only: whoever is a home keeps none of its own
    used = {home["id"] for home in homes.values()}
    return {entity_id: home for entity_id, home in homes.items() if entity_id not in used}


def same_name_plan(
    nodes: Iterable[Node], homes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """What to do with entities that share a normalized name.

    Returns one step per pair, against the member kept:
    ``{"action": "merge"|"ask"|"separate", "keep", "other", "reason"}``.
    """
    groups: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        groups[node.normalized or node.name.strip().lower()].append(node)

    def home_of(node: Node) -> str | None:
        home = homes.get(node.id)
        return home["id"] if home else None

    plan: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # the best-evidenced member is the one kept; oldest breaks ties
        members = sorted(members, key=lambda n: (-n.memories, -n.relations, n.created_at, n.id))
        keep = members[0]
        for other in members[1:]:
            step = {"keep": keep.id, "other": other.id, "name": keep.name}
            if "person" in (keep.entity_type, other.entity_type):
                plan.append({**step, "action": "ask",
                             "reason": "a person is never merged on a name alone"})
                continue
            home_a, home_b = home_of(keep), home_of(other)
            if home_a and home_b and home_a != home_b:
                plan.append({**step, "action": "separate",
                             "reason": "same name under different homes"})
                continue
            if home_a != home_b:  # one has a home, the other does not
                plan.append({**step, "action": "ask",
                             "reason": "same name, but only one of them has a home"})
                continue
            anchor_a = keep.entity_type in ANCHOR_TYPES
            anchor_b = other.entity_type in ANCHOR_TYPES
            if not home_a and anchor_a != anchor_b:
                plan.append({**step, "action": "ask",
                             "reason": "same name, but typed as different kinds of thing"})
                continue
            plan.append({**step, "action": "merge",
                         "reason": "same name under the same home" if home_a
                         else "same name and nothing sets them apart"})
    return plan
