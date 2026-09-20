"""Entity structure: which names are hubs, where a part belongs, and what a
shared name means.

Measured on a real store: 3,314 entities over 985 memories, of which 2,073
appeared in exactly one memory. Most extracted "entities" are one-off noun
phrases, and treating every one as a hub is what filled the map with noise and
the queue with questions nobody can answer ("round 1 and Round 1?").

Three rules, all computed from what the store already holds, none of which
deletes anything:

* **Hub status is earned.** With a decision provider, a name is a hub when the
  provider called it a named thing, or it is an anchor type the provider did
  not call a value or a role. Without one, it is an anchor type or a name two
  memories mention. Everything else stays a phrase on its memory: searchable,
  linked, and never stored as "not a hub", so the answer changes as soon as
  the evidence does.
* **Home is one level, and derived.** A part belongs to the project or product
  it keeps appearing with. A stated ``part_of`` relation wins, and is the only
  way an organization becomes a home: a company co-occurs with everything its
  owner does, which says nothing about what belongs to it.

* **A shared name is read through home.** Same name under the same home is the
  same thing. Same name under different homes is two things and no question.
  People are never merged on a name alone.

Each rule was scored against 360 names, 141 homes and 310 past merge decisions
from a real store, labelled by an independent reader (see
``evals/entity_structure_benchmark.py``). The first draft of the hub rule
("anchor, or two memories, or a relation") was right about 52% of the names it
promoted: recurrence finds topics like "billing", not things. Using the
provider's verdict it is 72% at 98% recall. Homes from co-mention alone were
68% right; restricted as described above, 86%.

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
#: Types that can become a home from co-mention alone. An organization needs a
#: stated relation: as a co-mentioned home it measured 47% right.
COMENTION_HOME_TYPES = frozenset({"project", "product"})
#: Verdicts of the name screen (intelligence/entities.py) that are not things.
SCREEN_SKIPS = frozenset({"role", "value_or_fragment"})
#: Probability a skip verdict needs before anything believes it: the write
#: path, the upkeep queue and hub status all read this one number, so a name
#: can never lose its place on a guess that raised no question for anyone.
SCREEN_GATE = 0.80
#: Probability on "named_thing" from which the verdict alone makes a hub.
NAMED_THING_MIN = 0.6
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


def hub_reason(
    entity_type: str | None, memories: int, relations: int,
    screen: dict[str, Any] | None = None,
) -> str:
    """Why a name is a hub, in words; empty when it is not one."""
    if memories < 1 and relations < 1:
        return ""
    verdict = (screen or {}).get("verdict")
    if screen and screen.get("kept"):
        return "kept by you"
    probability = float((screen or {}).get("probability") or 0.0)
    if verdict in SCREEN_SKIPS and probability >= SCREEN_GATE:
        return ""
    if verdict == "named_thing" and probability >= NAMED_THING_MIN:
        return "a named thing"
    if entity_type in ANCHOR_TYPES:
        return f"a {entity_type}"
    if (verdict is None or verdict in SCREEN_SKIPS) and memories >= 2:
        return f"in {memories} memories"
    return ""


def is_hub(
    entity_type: str | None, memories: int, relations: int,
    screen: dict[str, Any] | None = None,
) -> bool:
    return bool(hub_reason(entity_type, memories, relations, screen))


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
        if share < min_share or by_id[top].entity_type not in COMENTION_HOME_TYPES:
            continue
        # Two anchors tied for the top cannot both be home; leave it unhomed.
        # This is also what keeps a part seen once honest: with one memory
        # every count is 1, so any second anchor in it, a company included,
        # is a tie, and that reading measured a coin toss (48% right).
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
