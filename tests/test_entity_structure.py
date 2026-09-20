"""Entity structure: which names are hubs, where a part belongs, what a shared
name means, and what the screen refuses to make an entity of.

The rules in ``intelligence/structure.py`` are pure, so most of this file is
lists in, plans out. The store tests then check that the pass applies exactly
those plans, that a dry run applies none of them, and that everything a
provider merely vouched for lands in the upkeep queue instead of happening.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.intelligence.entities import (
    SCREEN_CRITERIA,
    SCREEN_GATE,
    non_referent_reason,
    resolve_mentions,
    screen_names,
    screened_out,
)
from memry.intelligence.structure import (
    NAMED_THING_MIN,
    Node,
    derive_homes,
    hub_reason,
    is_hub,
    same_name_plan,
)
from memry.models import Entity, EntityMention, Memory, Relation, Scope
from memry.providers.decisions import Answer, Answers, Decider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


# --------------------------------------------------------------- fake provider
_ASKED_RE = re.compile(r'In this memory, what is "(.+)"\?$')


class FakeScreener(Decider):
    """Answers the name screen from a script; abstains on anything else.

    Keyed by the lower-cased name, because that is what the screen question
    carries. ``asked`` records every name it was shown, which is how the tests
    check that an already-known name is never sent to a provider.
    """

    name = "fake-screener"
    available = True
    auto_confirm_confidence = 0.95

    def __init__(self, verdicts: dict[str, tuple[str, float]] | None = None) -> None:
        self.verdicts = {k.lower(): v for k, v in (verdicts or {}).items()}
        self.asked: list[str] = []
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        out: dict[str, Answer] = {}
        for key, question in questions.items():
            match = _ASKED_RE.match(getattr(question, "instructions", "") or "")
            if match is None:
                out[key] = Answer()  # not a screen question: abstain
                continue
            asked = match.group(1)
            self.asked.append(asked)
            scripted = self.verdicts.get(asked.strip().lower())
            if scripted is None:
                out[key] = Answer()
                continue
            verdict, probability = scripted
            rest = (1.0 - probability) / (len(SCREEN_CRITERIA) - 1)
            probabilities = {
                option: (probability if option == verdict else round(rest, 3))
                for option in SCREEN_CRITERIA
            }
            out[key] = Answer(value=verdict, probabilities=probabilities,
                              confidence=probability, available=True)
        return Answers(out)


class BrokenDecider(Decider):
    """A provider that raises. A write must survive it untouched."""

    name = "broken"
    available = True

    def decide(self, state, questions):
        raise RuntimeError("provider down")


# --------------------------------------------------- non_referent_reason (free)
@pytest.mark.parametrize("name", [
    "250 ms", "34px", "1.3m", "$0.4304/s", "22 tests", "40 opponents",
    "10^5 run", "450 ms floor",
])
def test_a_measurement_or_a_count_is_never_a_referent(name):
    assert non_referent_reason(name), f"{name!r} should be screened mechanically"


@pytest.mark.parametrize("name", [
    "3M", "50 Cent", "7 Wonders", "1Password", "Office 365", "S&P 500",
    "4chan", "Windows 11", "GPT-5", "5G", "Round 47", "/register/",
    "Hot Chips 2026",
])
def test_a_name_that_merely_contains_a_number_survives(name):
    assert non_referent_reason(name) is None, f"{name!r} is a real name"


# ---------------------------------------------------------------- hub_reason
def test_an_anchor_type_is_a_hub_on_its_own():
    assert hub_reason("project", 1, 0) == "a project"
    assert is_hub("person", 1, 0)


def test_two_memories_earn_hub_status_when_nothing_was_screened():
    assert hub_reason("concept", 2, 0) == "in 2 memories"
    assert not is_hub("concept", 1, 0)


def test_a_relation_alone_does_not_make_a_hub():
    assert hub_reason("concept", 0, 3) == ""
    assert not is_hub("concept", 1, 5)


@pytest.mark.parametrize("verdict", ["value_or_fragment", "role"])
def test_a_skip_verdict_takes_hub_status_away_from_an_anchor(verdict):
    screen = {"verdict": verdict, "probability": 0.9}
    assert hub_reason("project", 9, 4, screen) == ""
    assert hub_reason("concept", 9, 0, screen) == ""


def test_a_skip_verdict_below_the_gate_is_believed_by_nobody():
    """One gate for the write path, the queue and hub status.

    A 0.30 "role" guess once took a typed project off the map while raising no
    queue row, so nobody could say "kept". Hub status now reads the same 0.80
    gate as everything else: what is not sure enough to ask about is not sure
    enough to act on.
    """
    unsure = {"verdict": "role", "probability": 0.3}
    assert not screened_out(unsure)
    assert hub_reason("project", 5, 2, unsure) == "a project"
    assert hub_reason("concept", 2, 0, unsure) == "in 2 memories"
    sure = {"verdict": "role", "probability": 0.9}
    assert hub_reason("project", 5, 2, sure) == ""


def test_a_named_thing_verdict_promotes_a_single_memory_above_the_minimum():
    assert hub_reason("concept", 1, 0,
                      {"verdict": "named_thing",
                       "probability": NAMED_THING_MIN}) == "a named thing"
    assert hub_reason("concept", 1, 0,
                      {"verdict": "named_thing", "probability": 0.59}) == ""


def test_keeping_a_name_wins_over_every_other_verdict():
    kept = {"verdict": "value_or_fragment", "probability": 0.99, "kept": True}
    assert hub_reason("concept", 1, 0, kept) == "kept by you"


def test_a_name_with_no_evidence_at_all_is_never_a_hub():
    assert hub_reason("project", 0, 0) == ""
    assert hub_reason("concept", 0, 0,
                      {"verdict": "named_thing", "probability": 1.0}) == ""


# --------------------------------------------------------------- derive_homes
def _node(node_id: str, entity_type: str | None = None, **kw) -> Node:
    return Node(id=node_id, name=node_id, normalized=node_id.lower(),
                entity_type=entity_type, **kw)


def _links(*pairs: tuple[str, str]) -> list[tuple[str, str]]:
    return list(pairs)


PART_AND_PROJECT = [_node("part"), _node("nimbus", "project")]
THREE_SHARED = _links(("part", "m1"), ("part", "m2"), ("part", "m3"),
                      ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m3"))


def test_a_stated_part_of_wins_over_the_anchor_it_co_occurs_with():
    nodes = [*PART_AND_PROJECT, _node("atlas", "project")]
    links = [*THREE_SHARED, ("atlas", "m7"), ("atlas", "m8"), ("atlas", "m9")]

    homes = derive_homes(nodes, links, [("part", "part_of", "atlas")])

    assert homes == {"part": {"id": "atlas", "share": 1.0, "source": "relation"}}


def test_co_mention_needs_a_high_share_and_an_anchor_with_some_history():
    assert derive_homes(PART_AND_PROJECT, THREE_SHARED, []) == {
        "part": {"id": "nimbus", "share": 1.0, "source": "co-mention"}}
    # the anchor itself is too thin: two memories is not a home
    thin = _links(("part", "m1"), ("part", "m2"), ("nimbus", "m1"), ("nimbus", "m2"))
    assert derive_homes(PART_AND_PROJECT, thin, []) == {}
    # present in half the part's memories: below HOME_MIN_SHARE
    half = _links(*[("part", f"m{i}") for i in (1, 2, 3, 4)],
                  ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m9"))
    assert derive_homes(PART_AND_PROJECT, half, []) == {}


def test_an_organization_is_a_home_only_when_a_relation_says_so():
    nodes = [_node("part"), _node("acme", "organization")]
    links = _links(("part", "m1"), ("part", "m2"), ("part", "m3"),
                   ("acme", "m1"), ("acme", "m2"), ("acme", "m3"))

    assert derive_homes(nodes, links, []) == {}
    assert derive_homes(nodes, links, [("part", "part_of", "acme")]) == {
        "part": {"id": "acme", "share": 1.0, "source": "relation"}}


def test_two_anchors_tied_for_the_top_leave_the_part_unhomed():
    nodes = [_node("part"), _node("nimbus", "project"), _node("atlas", "project")]
    links = _links(("part", "m1"), ("part", "m2"),
                   ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m3"),
                   ("atlas", "m1"), ("atlas", "m2"), ("atlas", "m3"))

    assert derive_homes(nodes, links, []) == {}


def test_a_part_seen_once_needs_the_anchor_to_itself():
    # NOTE: the explicit once-seen guard at structure.py:170 is unreachable -
    # with one memory every count is 1, so a second home-capable anchor always
    # trips the tie check three lines above it first. The outcome the rule
    # wanted is what happens, so this pins the behaviour, not the branch.
    nodes = [_node("part"), _node("nimbus", "project"), _node("acme", "organization")]
    shared_memory = _links(("part", "m1"),
                           ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m3"),
                           ("acme", "m1"), ("acme", "m7"), ("acme", "m8"))
    assert derive_homes(nodes, shared_memory, []) == {}

    alone = _links(("part", "m1"),
                   ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m3"))
    assert derive_homes(nodes, alone, []) == {
        "part": {"id": "nimbus", "share": 1.0, "source": "co-mention"}}


def test_an_anchor_typed_node_never_gets_a_home_from_co_mention():
    nodes = [_node("ada", "person"), _node("nimbus", "project")]
    links = _links(("ada", "m1"), ("ada", "m2"), ("ada", "m3"),
                   ("nimbus", "m1"), ("nimbus", "m2"), ("nimbus", "m3"))

    assert derive_homes(nodes, links, []) == {}


def test_home_is_one_level_so_a_home_keeps_none_of_its_own():
    nodes = [_node("part"), _node("nimbus", "project"), _node("atlas", "project")]

    homes = derive_homes(nodes, THREE_SHARED, [("nimbus", "part_of", "atlas")])

    assert set(homes) == {"part"}, "nimbus is a home, so it loses its own"
    assert homes["part"]["id"] == "nimbus"


# ------------------------------------------------------------- same_name_plan
def _pair(type_a=None, type_b=None, memories_a=3, memories_b=1) -> list[Node]:
    return [
        Node(id="a", name="privacy policy", normalized="privacy policy",
             entity_type=type_a, memories=memories_a),
        Node(id="b", name="privacy policy", normalized="privacy policy",
             entity_type=type_b, memories=memories_b),
    ]


def test_the_same_home_merges_and_keeps_the_better_evidenced_member():
    plan = same_name_plan(_pair(), {"a": {"id": "nimbus"}, "b": {"id": "nimbus"}})

    assert [step["action"] for step in plan] == ["merge"]
    assert plan[0]["keep"] == "a" and plan[0]["other"] == "b"
    assert plan[0]["reason"] == "same name under the same home"


def test_different_homes_are_two_things_and_no_question():
    plan = same_name_plan(_pair(), {"a": {"id": "nimbus"}, "b": {"id": "atlas"}})

    assert [step["action"] for step in plan] == ["separate"]


def test_a_person_is_never_merged_on_a_name_alone():
    plan = same_name_plan(_pair(type_a="person"), {})
    assert [step["action"] for step in plan] == ["ask"]
    plan = same_name_plan(_pair(type_b="person"),
                          {"a": {"id": "nimbus"}, "b": {"id": "nimbus"}})
    assert [step["action"] for step in plan] == ["ask"]


def test_one_homed_and_one_not_is_a_question():
    plan = same_name_plan(_pair(), {"a": {"id": "nimbus"}})

    assert [step["action"] for step in plan] == ["ask"]
    assert plan[0]["reason"] == "same name, but only one of them has a home"


def test_an_unhomed_anchor_against_a_non_anchor_is_a_question():
    plan = same_name_plan(_pair(type_a="product", type_b="concept"), {})

    assert [step["action"] for step in plan] == ["ask"]
    assert plan[0]["reason"] == "same name, but typed as different kinds of thing"


def test_two_unhomed_names_of_the_same_kind_merge():
    plan = same_name_plan(_pair(type_a="concept", type_b="concept"), {})

    assert [step["action"] for step in plan] == ["merge"]
    assert plan[0]["reason"] == "same name and nothing sets them apart"


# -------------------------------------------------------- screen_names / gate
def test_the_screen_reads_the_probability_the_provider_put_on_its_answer():
    decider = FakeScreener({"deadline": ("value_or_fragment", 0.91),
                            "nimbus": ("named_thing", 0.77)})

    out = screen_names(decider, "Nimbus ships past the deadline", ["Deadline", "Nimbus"])

    assert out["deadline"] == {"verdict": "value_or_fragment", "probability": 0.91}
    assert out["nimbus"] == {"verdict": "named_thing", "probability": 0.77}
    assert decider.asked == ["Deadline", "Nimbus"]
    assert screened_out(out["deadline"]) and not screened_out(out["nimbus"])


def test_a_skip_verdict_below_the_gate_is_not_believed():
    below = {"verdict": "role", "probability": round(SCREEN_GATE - 0.01, 3)}
    assert not screened_out(below)
    assert screened_out({**below, "probability": SCREEN_GATE})
    assert not screened_out(None)


def test_no_provider_and_no_names_screen_nothing():
    assert screen_names(None, "some memory", ["Deadline"]) == {}
    assert screen_names(FakeScreener(), "some memory", []) == {}


def test_a_provider_that_raises_costs_the_write_nothing():
    assert screen_names(BrokenDecider(), "some memory", ["Deadline"]) == {}


# ------------------------------------------------------- resolve_mentions
@pytest.fixture
def backend_store():
    s = MemoryStore(Config(db_path=":memory:", dedup_entities=False),
                    llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def _write(store, content: str, user_id: str = "ada") -> Memory:
    return store.backend.insert_memory(Memory(content=content, user_id=user_id))


def _resolve(store, memory, surfaces, decider):
    return resolve_mentions(
        backend=store.backend, llm=NoneLLM(), decider=decider,
        scope=Scope(user_id="ada"), memory_id=memory.id,
        memory_content=memory.content, surfaces=surfaces,
    )


def test_a_mechanical_non_referent_never_becomes_an_entity(backend_store):
    memory = _write(backend_store, "The p95 settled at 250 ms after the fix")
    decider = FakeScreener()

    resolved = _resolve(backend_store, memory, ["250 ms"], decider)

    assert resolved == {}
    assert backend_store.entities(user_id="ada") == []
    assert decider.asked == [], "a name the rules already reject costs no call"


def test_a_value_verdict_above_the_gate_creates_no_entity(backend_store):
    memory = _write(backend_store, "The p95 settled just under the deadline")
    decider = FakeScreener({"deadline": ("value_or_fragment", 0.95)})

    resolved = _resolve(backend_store, memory, ["deadline"], decider)

    assert resolved == {}
    assert backend_store.entities(user_id="ada") == []
    assert decider.asked == ["deadline"]


def test_the_same_verdict_below_the_gate_still_creates_the_entity(backend_store):
    memory = _write(backend_store, "The p95 settled just under the deadline")
    decider = FakeScreener({"deadline": ("value_or_fragment", 0.5)})

    resolved = _resolve(backend_store, memory, ["deadline"], decider)

    assert list(resolved) == ["deadline"]
    assert [e.name for e in backend_store.entities(user_id="ada")] == ["deadline"]


def test_a_name_the_store_already_knows_is_not_screened_again(backend_store):
    existing = backend_store.backend.insert_entity(
        Entity(name="Nimbus", normalized="nimbus", user_id="ada")
    )
    memory = _write(backend_store, "Nimbus shipped on Friday")
    decider = FakeScreener({"nimbus": ("value_or_fragment", 0.99)})

    resolved = _resolve(backend_store, memory, ["Nimbus"], decider)

    assert decider.asked == [], "upkeep reviews known names, the write path does not"
    assert resolved["nimbus"].id == existing.id


# ------------------------------------------------------------------- the store
@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:", dedup_entities=False),
                    llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def _entity(store, name: str, entity_type: str | None = None,
            user_id: str = "ada") -> Entity:
    return store.backend.insert_entity(
        Entity(name=name, normalized=name.strip().lower(),
               entity_type=entity_type, user_id=user_id)
    )


def _mention(store, entity: Entity, memory: Memory) -> None:
    store.backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface=entity.name)
    )


def _seed_part_and_home(store, user_id: str = "ada") -> dict[str, Entity]:
    """A project seen three times, a part that rides along, and a duplicate
    pair that shares a name in memories no anchor touches."""
    nimbus = _entity(store, "Nimbus", "project", user_id)
    worker = _entity(store, "sync worker", None, user_id)
    for i in range(3):
        memory = _write(store, f"Nimbus sync worker note {i}", user_id)
        _mention(store, nimbus, memory)
        _mention(store, worker, memory)
    policy_a = _entity(store, "privacy policy", "concept", user_id)
    policy_b = _entity(store, "privacy policy", "concept", user_id)
    for entity in (policy_a, policy_b):
        _mention(store, entity, _write(store, f"A note about {entity.id}", user_id))
    return {"nimbus": nimbus, "worker": worker, "a": policy_a, "b": policy_b}


def _duplicate_pair(store, seeded) -> tuple[Entity, Entity]:
    """(kept, folded away) for the seeded same-name pair.

    Both members carry one memory and no relations and were inserted in the
    same second, so which one wins the tie-break is down to the generated id;
    what the pass promises is that exactly one of them survives.
    """
    states = [store.backend.get_entity(seeded[key].id) for key in ("a", "b")]
    folded = [e for e in states if e.merged_into is not None]
    assert len(folded) == 1, f"expected exactly one merge, got {states}"
    kept = next(e for e in states if e.merged_into is None)
    assert folded[0].merged_into == kept.id
    return kept, folded[0]


def test_a_dry_run_reports_the_whole_plan_and_applies_none_of_it(store):
    seeded = _seed_part_and_home(store)

    outcome = store.run_structure_pass(user_id="ada", dry_run=True)

    assert outcome["dry_run"] is True
    assert outcome["homes"] == 1 and outcome["homes_changed"] == 1
    assert outcome["merged"] == 1
    assert [(step["action"], step["name"]) for step in outcome["plan"]] == [
        ("merge", "privacy policy")]
    assert outcome["home_list"] == [("sync worker", "Nimbus", 1.0, "co-mention")]
    # nothing was written: no metadata, no merge
    for entity in store.entities(user_id="ada", limit=100):
        assert not (entity.metadata or {}).get("home")
        assert entity.merged_into is None
    assert store.backend.get_entity(seeded["b"].id).merged_into is None


def test_the_real_pass_records_the_home_merges_the_duplicate_and_settles(store):
    seeded = _seed_part_and_home(store)

    outcome = store.run_structure_pass(user_id="ada")

    assert outcome["homes_changed"] == 1 and outcome["merged"] == 1
    home = store.backend.get_entity(seeded["worker"].id).metadata["home"]
    assert home == {"id": seeded["nimbus"].id, "name": "Nimbus",
                    "share": 1.0, "source": "co-mention"}
    _duplicate_pair(store, seeded)
    # a home has no home of its own, and the anchor is not a part
    assert not (store.backend.get_entity(seeded["nimbus"].id).metadata or {}).get("home")

    again = store.run_structure_pass(user_id="ada")
    assert again["homes_changed"] == 0 and again["merged"] == 0


def test_entity_structure_reports_hub_status_and_the_home_it_derived(store):
    seeded = _seed_part_and_home(store)

    structure = store.entity_structure(user_id="ada")

    assert structure[seeded["nimbus"].id]["why"] == "a project"
    worker = structure[seeded["worker"].id]
    assert worker["hub"] is True and worker["why"] == "in 3 memories"
    assert worker["home"]["name"] == "Nimbus"
    assert structure[seeded["a"].id]["home"] is None


# ------------------------------------------------------------- the name screen
def _seed_role_and_value(store, user_id: str = "ada") -> dict[str, Entity]:
    """Three memories about one project, one person and a role-ish phrase, plus
    a value-ish phrase that has nothing to do with them."""
    nimbus = _entity(store, "Nimbus", "project", user_id)
    ada = _entity(store, "Ada", "person", user_id)
    contact = _entity(store, "billing contact", None, user_id)
    for i in range(3):
        memory = _write(store, f"Ada is the billing contact for Nimbus ({i})", user_id)
        for entity in (nimbus, ada, contact):
            _mention(store, entity, memory)
    deadline = _entity(store, "the deadline", None, user_id)
    _mention(store, deadline, _write(store, "It slipped past the deadline", user_id))
    return {"nimbus": nimbus, "ada": ada, "contact": contact, "deadline": deadline}


@pytest.fixture
def screened(store):
    seeded = _seed_role_and_value(store)
    store.decider = FakeScreener({
        "billing contact": ("role", 0.93),
        "the deadline": ("value_or_fragment", 0.91),
        "nimbus": ("named_thing", 0.88),
        "ada": ("named_thing", 0.95),
    })
    store.run_structure_pass(user_id="ada")  # the role needs a home to point at
    return seeded


def test_the_screen_stores_a_verdict_and_never_asks_about_a_name_twice(store, screened):
    outcome = store.run_name_screen(user_id="ada")

    assert outcome == {"screened": 4, "queued": 2, "skipped": 0}
    stored = store.backend.get_entity(screened["contact"].id).metadata["screen"]
    assert stored["verdict"] == "role" and stored["probability"] == 0.93
    assert stored["at"]

    asked = len(store.decider.asked)
    assert store.run_name_screen(user_id="ada") == {
        "screened": 0, "queued": 0, "skipped": 0}
    assert len(store.decider.asked) == asked


def test_a_role_and_a_value_reach_the_queue_as_their_own_kinds(store, screened):
    store.run_name_screen(user_id="ada")

    queue = {item["kind"]: item for item in store.upkeep_queue(user_id="ada")}

    assert queue["role"]["id"] == screened["contact"].id
    assert queue["role"]["title"] == "billing contact"
    # the section explains what a role is once; a row repeats none of it
    assert queue["role"]["detail"] == ""
    assert queue["role"]["accept"] == "remove"
    assert queue["entity_review"]["id"] == screened["deadline"].id
    assert queue["entity_review"]["accept"] == "remove"
    # a named thing is not a question for anybody
    assert screened["nimbus"].id not in {item["id"] for item in
                                         store.upkeep_queue(user_id="ada")}


def test_declining_keeps_the_name_and_clears_the_row_for_good(store, screened):
    store.run_name_screen(user_id="ada")

    assert store.decide_upkeep("role", screened["contact"].id, "decline", user_id="ada")

    kept = store.backend.get_entity(screened["contact"].id)
    assert kept is not None and kept.metadata["screen"]["kept"] is True
    assert "role" not in {item["kind"] for item in store.upkeep_queue(user_id="ada")}
    # and a kept name is a hub again, whatever the verdict said
    assert store.entity_structure(user_id="ada")[kept.id]["why"] == "kept by you"


def test_accepting_a_value_retires_the_entity_and_it_can_be_brought_back(store, screened):
    store.run_name_screen(user_id="ada")
    deadline = screened["deadline"].id

    assert store.decide_upkeep("entity_review", deadline, "accept", user_id="ada")

    assert store.backend.get_entity(deadline) is None
    assert deadline in {row["entity_id"] for row in store.retired_entities(user_id="ada")}
    assert "It slipped past the deadline" in {
        m.content for m in store.get_all(user_id="ada", limit=50)}

    assert store.restore_entities([deadline]) == 1
    assert store.backend.get_entity(deadline) is not None


def test_accepting_a_role_removes_the_name_and_invents_nothing(store, screened):
    """A role is a word in a memory, not a thing. Guessing its holder from who
    else the memory mentions was wrong about half the time on a real store (a
    tax memory listing profile types is not a list of what its owner is), so
    accepting retires the name and records no relation in its place."""
    store.run_name_screen(user_id="ada")
    contact = screened["contact"].id

    assert store.decide_upkeep("role", contact, "accept", user_id="ada")

    assert store.backend.get_entity(contact) is None
    assert store.relations(user_id="ada") == []
    assert contact in {row["entity_id"] for row in store.retired_entities(user_id="ada")}
    assert store.restore_entities([contact]) == 1


def test_the_screen_is_skipped_entirely_without_a_provider(store):
    _seed_role_and_value(store)
    assert store.run_name_screen(user_id="ada") == {
        "screened": 0, "queued": 0, "skipped": -1}
    assert store.upkeep_queue(user_id="ada") == []


def test_a_name_with_no_memory_behind_it_is_not_screened(store):
    _entity(store, "orphan")
    store.decider = FakeScreener({"orphan": ("value_or_fragment", 0.99)})

    assert store.run_name_screen(user_id="ada")["skipped"] == 1
    assert "screen" not in (store.entities(user_id="ada")[0].metadata or {})


def test_a_pair_under_different_homes_is_never_proposed(store):
    from memry.intelligence.entities import propose_same_name_duplicates

    nimbus = _entity(store, "Nimbus", "project")
    atlas = _entity(store, "Atlas", "project")
    for anchor in (nimbus, atlas):
        policy = _entity(store, "privacy policy")
        store.backend.set_entity_metadata(
            policy.id, {"home": {"id": anchor.id, "name": anchor.name}})
        _mention(store, policy, _write(store, f"the {anchor.name} privacy policy"))

    created = propose_same_name_duplicates(
        backend=store.backend, scope=Scope(user_id="ada"))

    assert created == 0
    assert store.merge_proposals(user_id="ada") == []


# ------------------------------------------------------------------------ REST
@pytest.fixture
def client():
    # The app starts the scheduler, whose first cycle would run passes while a
    # test is still inserting entities; keep that pass off here.
    s = MemoryStore(Config(db_path=":memory:", dedup_entities=False),
                    llm=NoneLLM(), embedder=HashEmbedder(64))
    app = create_app(s)
    with TestClient(app) as c:
        c.store = s
        yield c
    s.close()


def test_the_entity_list_carries_hub_status_home_and_memory_count(client):
    seeded = _seed_part_and_home(client.store, user_id="u")

    payload = client.get("/api/v1/entities?user_id=u&limit=100").json()
    by_id = {row["id"]: row for row in payload}

    assert by_id[seeded["nimbus"].id]["hub"] is True
    assert by_id[seeded["nimbus"].id]["memories"] == 3
    assert by_id[seeded["nimbus"].id]["home"] is None
    worker = by_id[seeded["worker"].id]
    assert worker["hub"] is True and worker["memories"] == 3
    assert worker["home"]["name"] == "Nimbus" and worker["home"]["share"] == 1.0
    assert by_id[seeded["a"].id]["hub"] is False, "seen once, untyped, unscreened"


def test_the_map_shows_hubs_only_and_hangs_a_part_on_its_home(client):
    seeded = _seed_part_and_home(client.store, user_id="u")
    phrase = _entity(client.store, "some passing phrase", None, "u")
    _mention(client.store, phrase, _write(client.store, "a one-off remark", "u"))

    data = client.get("/api/v1/map?user_id=u").json()
    planets = {node["entity_id"]: node for node in data["entities"]}

    assert phrase.id not in planets, "a once-seen untyped phrase earns nothing"
    assert seeded["worker"].id not in planets, "a homed part is not a planet"
    nimbus = planets[seeded["nimbus"].id]
    assert [part["label"] for part in nimbus["parts"]] == ["sync worker"]
    assert nimbus["part_count"] == 1
    assert data["entity_names"] > len(planets)


def test_running_the_structure_pass_dry_over_rest_changes_nothing(client):
    seeded = _seed_part_and_home(client.store, user_id="u")

    result = client.post("/api/v1/maintenance/run/structure",
                         json={"user_id": "u", "dry_run": True})

    assert result.status_code == 200
    body = result.json()
    assert body["dry_run"] is True
    assert [step["action"] for step in body["plan"]] == ["merge"]
    assert body["home_list"] == [["sync worker", "Nimbus", 1.0, "co-mention"]]
    assert client.store.backend.get_entity(seeded["b"].id).merged_into is None
    assert not (client.store.backend.get_entity(seeded["worker"].id).metadata or {})

    # and the same call without the flag does apply it
    applied = client.post("/api/v1/maintenance/run/structure", json={"user_id": "u"})
    assert applied.json()["merged"] == 1
    _duplicate_pair(client.store, seeded)


def test_a_relation_makes_no_hub_but_a_second_memory_does(store):
    """Guards the rule the first draft got wrong: recurrence finds topics, a
    relation alone finds nothing at all."""
    linked = _entity(store, "billing")
    nimbus = _entity(store, "Nimbus", "project")
    _mention(store, linked, _write(store, "billing is handled in Nimbus"))
    _mention(store, nimbus, _write(store, "Nimbus shipped"))
    store.backend.add_relation(
        Relation(subject=linked.id, predicate="handled_in", object=nimbus.id,
                 user_id="ada")
    )

    structure = store.entity_structure(user_id="ada")
    assert structure[linked.id]["relations"] == 1
    assert structure[linked.id]["hub"] is False

    _mention(store, linked, _write(store, "billing came up again"))
    assert store.entity_structure(user_id="ada")[linked.id]["hub"] is True


def test_the_name_screen_runs_on_its_own_over_rest_and_says_so_without_a_provider():
    s = MemoryStore(Config(db_path=":memory:", dedup_entities=False),
                    llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        with TestClient(create_app(s)) as c:
            done = c.post("/api/v1/maintenance/run/screen", json={"user_id": "u"})
            assert done.status_code == 200
            assert done.json()["skipped"] == -1, "no provider: nothing asked, nothing changed"
    finally:
        s.close()
