"""Upkeep: what runs on its own, and the queue of what needs a person.

The rule behind every test here: a pass may do on its own only what was
measured safe (entity identity above the gate, word-for-word duplicates,
mechanical non-entities). Everything a model merely vouched for waits in the
queue, is asked about once, and is never proposed again after a "no".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import FakeLLM
from starlette.testclient import TestClient

from memry.config import Config
from memry.providers.decisions import Answer, Answers, Decider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


class FakeDecider(Decider):
    """Answers every score question with the same value."""

    name = "fake-decider"
    available = True
    auto_confirm_confidence = 0.95

    def __init__(self, value: float = 1.5) -> None:
        self.value = value
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        return Answers({
            key: Answer(value=self.value, confidence=0.9, available=True)
            for key in questions
        })


def _verdict(same: bool, content: str = "", reason: str = "r") -> str:
    return json.dumps({"same_fact": same, "content": content, "reason": reason})


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    # hash vectors put restatements nowhere near 0.90; the grouping itself is
    # covered in test_consolidation, this file is about what happens after
    s.UPKEEP_CONSOLIDATION_THRESHOLD = 0.25
    yield s
    s.close()


def _seed(store, *contents: str, user_id: str = "ada") -> None:
    """Bulk import is the route by which duplicates reach a real store;
    ``store.add`` would have caught them at write time."""
    store.import_verbatim(
        [{"content": c, "user_id": user_id} for c in contents], dedup=False
    )


def _live(store, user_id="ada"):
    return {m.content for m in store.get_all(user_id=user_id, limit=100)}


def _entity(store, name: str, content: str, user_id: str = "ada"):
    """A memory mentioning one entity, without a model to extract it."""
    from memry.models import Entity, EntityMention, Memory

    entity = store.backend.insert_entity(Entity(name=name, user_id=user_id))
    memory = store.backend.insert_memory(
        Memory(content=content, entities=[name], user_id=user_id)
    )
    store.backend.add_mention(
        EntityMention(entity_id=entity.id, memory_id=memory.id, surface=name)
    )
    return entity


# ----------------------------------------------------------- consolidation
def test_identical_text_merges_on_its_own_without_a_model(store):
    _seed(store, "User is Marcus Vandenberg", "User is  Marcus Vandenberg.")
    store.llm = FakeLLM()  # no scripted answer: asking it would raise

    outcome = store.run_consolidation_pass(user_id="ada")

    assert outcome["merged"] == 1, "word-for-word duplicates collapse on sight"
    assert outcome["queued"] == 0
    assert len(_live(store)) == 1
    assert not store.upkeep_queue(user_id="ada")


def test_a_restatement_the_model_vouches_for_waits_in_the_queue(store):
    _seed(store, "User is Marcus Vandenberg", "The user's name is Marc.")
    # the fake model vouches for the merge; nobody measured that judgement
    store.llm = FakeLLM([_verdict(True, "User is Marcus Vandenberg, who goes by Marc.")])

    outcome = store.run_consolidation_pass(user_id="ada")

    assert outcome["merged"] == 0 and outcome["queued"] == 1
    queue = [q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "consolidation"]
    assert len(queue) == 1
    assert queue[0]["title"] == "User is Marcus Vandenberg, who goes by Marc."
    assert queue[0]["accept"] == "merge" and queue[0]["decline"] == "keep all"
    assert _live(store) == {"User is Marcus Vandenberg", "The user's name is Marc."}


def test_accepting_a_queued_merge_applies_it_without_asking_the_model_again(store):
    _seed(store, "User is Marcus Vandenberg", "The user's name is Marc.")
    store.llm = FakeLLM([_verdict(True, "User is Marcus Vandenberg (Marc).")])
    store.run_consolidation_pass(user_id="ada")
    item = next(q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "consolidation")

    assert store.decide_upkeep("consolidation", item["id"], "accept", user_id="ada")

    assert _live(store) == {"User is Marcus Vandenberg (Marc)."}
    assert store.llm.calls and len(store.llm.calls) == 1, "the stored verdict was reused"
    assert not [q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "consolidation"]
    # a second pass has nothing left to ask about
    assert store.run_consolidation_pass(user_id="ada")["judged"] == 0


def test_declining_a_queued_merge_is_remembered(store):
    _seed(store, "User is Marcus Vandenberg", "The user's name is Marc.")
    store.llm = FakeLLM([_verdict(True, "merged")])
    store.run_consolidation_pass(user_id="ada")
    item = next(q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "consolidation")

    assert store.decide_upkeep("consolidation", item["id"], "decline", user_id="ada")

    assert _live(store) == {"User is Marcus Vandenberg", "The user's name is Marc."}
    assert not [q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "consolidation"]
    # the model is not asked about the same group twice: FakeLLM has no
    # scripted answer left, and would raise if it were called
    assert store.run_consolidation_pass(user_id="ada") == {
        "scanned": 2, "judged": 0, "merged": 0, "queued": 0}


def test_a_group_the_model_rejected_is_not_asked_again(store):
    _seed(store, "User is Marcus Vandenberg", "The user's name is Marc.")
    store.llm = FakeLLM([_verdict(False, reason="different people")])
    assert store.run_consolidation_pass(user_id="ada")["judged"] == 1
    assert store.run_consolidation_pass(user_id="ada")["judged"] == 0
    assert not store.upkeep_queue(user_id="ada")


# ------------------------------------------------------------ entity review
def test_entity_review_queues_verdicts_and_a_kept_name_stays_kept(store):
    _entity(store, "formal tone", "Remember to always answer in a formal tone")
    _entity(store, "Acme", "Ada works at Acme")
    store.llm = FakeLLM([json.dumps({"junk": ["formal tone"]})])

    outcome = store.run_entity_review(user_id="ada")
    assert outcome["queued"] == 1
    item = next(q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "entity_review")
    assert item["title"] == "formal tone"

    assert store.decide_upkeep("entity_review", item["id"], "decline", user_id="ada")
    assert store.backend.get_entity(item["id"]) is not None
    # the next review reaches the same verdict, and the queue stays empty
    store.llm = FakeLLM([json.dumps({"junk": ["formal tone"]})])
    assert store.run_entity_review(user_id="ada")["queued"] == 0


def test_accepting_an_entity_review_removes_only_the_entity(store):
    _entity(store, "formal tone", "Remember to always answer in a formal tone")
    store.llm = FakeLLM([json.dumps({"junk": ["formal tone"]})])
    store.run_entity_review(user_id="ada")
    item = next(q for q in store.upkeep_queue(user_id="ada") if q["kind"] == "entity_review")

    assert store.decide_upkeep("entity_review", item["id"], "accept", user_id="ada")
    assert store.backend.get_entity(item["id"]) is None
    assert "Remember to always answer in a formal tone" in _live(store)


# --------------------------------------------------------------- the cycle
def test_cycle_scores_durability_when_a_decider_is_configured(store):
    store.decider = FakeDecider(1.5)
    store.add("Allergic to penicillin", user_id="ada", infer=False)

    ran = store.run_upkeep_cycle(user_id="ada")

    assert ran["durability"]["scored"] == 1
    assert store.last_pass_run("durability", "ada")["result"]["scored"] == 1
    # nothing left to score: the next tick is silent and leaves the record alone
    assert "durability" not in store.run_upkeep_cycle(user_id="ada")
    assert store.decider.calls == 1


def test_pausing_stops_every_pass(store):
    store.decider = FakeDecider()
    store.add("Allergic to penicillin", user_id="ada", infer=False)
    store.set_upkeep_paused(True)
    assert store.run_upkeep_cycle(user_id="ada") == {}
    assert store.decider.calls == 0
    store.set_upkeep_paused(False)
    assert "durability" in store.run_upkeep_cycle(user_id="ada")


def test_passes_run_on_their_interval_and_remember_what_they_did(store):
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    first = store.run_upkeep_cycle(user_id="ada", now=now)
    assert "dedup_entities" in first
    assert store.last_pass_run("dedup_entities", "ada")["result"] == first["dedup_entities"]
    assert "dedup_entities" not in store.run_upkeep_cycle(user_id="ada", now=now + timedelta(days=1))
    assert "dedup_entities" in store.run_upkeep_cycle(user_id="ada", now=now + timedelta(days=8))


# ------------------------------------------------------------------ REST
@pytest.fixture
def client():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    app = create_app(s)
    with TestClient(app) as c:
        c.store = s
        yield c
    s.close()


def test_status_carries_the_queue_and_the_pause_switch(client):
    info = client.get("/api/v1/maintenance?user_id=u").json()
    assert info["queue"] == []
    assert info["paused"] is False
    keys = {p["key"] for p in info["passes"]}
    assert keys == {"dedup_entities", "tag_abstraction", "durability", "consolidation"}
    assert all(p["run_url"] == f"/api/v1/maintenance/run/{p['key']}" for p in info["passes"])

    assert client.post("/api/v1/maintenance/pause", json={"paused": True}).json() == {"paused": True}
    assert client.get("/api/v1/maintenance?user_id=u").json()["paused"] is True


def test_run_now_uses_the_same_pass_as_the_scheduler(client):
    result = client.post("/api/v1/maintenance/run/dedup_entities", json={"user_id": "u"})
    assert result.status_code == 200
    assert "purged" in result.json()
    assert client.post("/api/v1/maintenance/run/nonsense", json={}).status_code == 404


def test_deciding_a_queue_row_over_rest(client):
    s = client.store
    s.UPKEEP_CONSOLIDATION_THRESHOLD = 0.25
    s.import_verbatim([{"content": c, "user_id": "u"} for c in
                       ("User is Marcus Vandenberg", "The user's name is Marc.")], dedup=False)
    s.llm = FakeLLM([_verdict(True, "User is Marcus Vandenberg (Marc).")])
    s.run_consolidation_pass(user_id="u")
    queue = client.get("/api/v1/maintenance?user_id=u").json()["queue"]
    assert [q["kind"] for q in queue] == ["consolidation"]

    ok = client.post("/api/v1/maintenance/decide",
                     json={"kind": "consolidation", "id": queue[0]["id"],
                           "decision": "accept", "user_id": "u"})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert client.get("/api/v1/maintenance?user_id=u").json()["queue"] == []
    # a row that is gone cannot be decided twice
    gone = client.post("/api/v1/maintenance/decide",
                       json={"kind": "consolidation", "id": queue[0]["id"],
                             "decision": "accept", "user_id": "u"})
    assert gone.status_code == 409
    assert client.post("/api/v1/maintenance/decide",
                       json={"kind": "consolidation", "id": "x", "decision": "maybe"}
                       ).status_code == 400


def test_many_rows_of_one_kind_are_decided_in_one_call(client):
    s = client.store
    for name in ("formal tone", "morning routine", "Acme"):
        _entity(s, name, f"A memory about {name}", user_id="u")
    s.llm = FakeLLM([json.dumps({"junk": ["formal tone", "morning routine"]})])
    s.run_entity_review(user_id="u")
    queue = client.get("/api/v1/maintenance?user_id=u").json()["queue"]
    ids = [q["id"] for q in queue if q["kind"] == "entity_review"]
    assert len(ids) == 2

    done = client.post("/api/v1/maintenance/decide",
                       json={"kind": "entity_review", "ids": ids,
                             "decision": "accept", "user_id": "u"})
    assert done.status_code == 200 and done.json()["done"] == 2
    assert client.get("/api/v1/maintenance?user_id=u").json()["queue"] == []
    assert {e.name for e in s.entities(user_id="u", limit=100)} == {"Acme"}
