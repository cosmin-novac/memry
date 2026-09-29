"""Undoing a merge of two entities.

A merge folds one entity into another: its mentions, relations and pairs
point at the one kept, and it stays behind as a tombstone. What the merge
moved is recorded when it happens (``LocalBackend._merge_snapshot_locked``),
so that a person who finds two things were merged can take it back, the way
a removed name is restored.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import (
    CandidateFact, Entity, EntityMention, Memory, MergeProposal, Relation, Scope, utcnow,
)
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore

ADA = Scope(user_id="ada")


class _Judge(NoneDecider):
    """A calibrated judge whose answer to every pair is ``same``."""

    name = "stub"
    available = True
    calibrated = True
    pair_merge_probability = 0.95

    def __init__(self, same: float = 0.6, different: float = 0.2) -> None:
        self.same, self.different = same, different

    def decide(self, state, questions):
        if "pair" not in questions:
            return Answers({})
        probabilities = {"same": self.same, "different": self.different,
                         "unsure": max(0.0, 1 - self.same - self.different)}
        return Answers({"pair": Answer(max(probabilities, key=probabilities.get),
                                       probabilities, 0.9, True)})


def _store(judge: NoneDecider | None = None) -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64),
                       decider=judge)


def _entity(store, name, entity_type="organization"):
    return store.backend.insert_entity(Entity(
        name=name, normalized=name.lower(), entity_type=entity_type, user_id="ada"))


def _memory(store, content, *mentions, categories=()):
    memory = store.backend.insert_memory(
        Memory(content=content, user_id="ada", categories=list(categories),
               embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([content])[0])
    for entity, surface in mentions:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=surface))
    return memory


def _world(store) -> dict:
    """ "Kessler Bau GmbH" and "Kessler Bau", with memories, a tag, an
    alias, relations to a place and between the two, and an open pair of
    "Kessler Bau" with "Kessler Roofing" compared at step 3."""
    gmbh, bau = _entity(store, "Kessler Bau GmbH"), _entity(store, "Kessler Bau")
    lane, roofing = _entity(store, "Harrow Lane 14", "place"), _entity(store, "Kessler Roofing")
    memories = [
        _memory(store, "Kessler Bau GmbH poured the foundation", (gmbh, "Kessler Bau GmbH")),
        _memory(store, "Kessler Bau GmbH is based at Harrow Lane 14",
                (gmbh, "Kessler Bau GmbH"), (lane, "Harrow Lane 14")),
        _memory(store, "Kessler Bau sent the roofers", (bau, "Kessler Bau"),
                categories=["roofing"]),
        _memory(store, "Kessler Bau, KB to the crew, fixed the gutter", (bau, "Kessler Bau")),
        _memory(store, "Kessler Roofing quoted the attic", (roofing, "Kessler Roofing")),
    ]
    store.backend.add_entity_alias(bau.id, "KB")
    for subject, predicate, obj, memory in ((gmbh, "based_at", lane, memories[1]),
                                            (bau, "worked_at", lane, memories[2]),
                                            (gmbh, "owns", bau, memories[0])):
        store.backend.add_relation(Relation(subject=subject.id, predicate=predicate,
                                            object=obj.id, user_id="ada", memory_id=memory.id))
    store.backend.add_proposal(MergeProposal(
        entity_a=bau.id, entity_b=roofing.id, user_id="ada", confidence=0.4, different=0.3,
        reason="stub: unsure", compared_step=3))
    store.refresh_property_vectors(user_id="ada")
    return {"gmbh": gmbh, "bau": bau, "lane": lane, "roofing": roofing, "memories": memories}


def _state(store, world) -> dict:
    """Everything a merge moves, and its undo must put back."""
    backend = store.backend
    entities = [world[key] for key in ("gmbh", "bau", "lane", "roofing")]
    ids = [m.id for m in backend.list_memories(ADA, limit=100)]
    both = {world["gmbh"].id, world["bau"].id}
    return {
        "entities": {e.id: (lambda row: (row.name, row.entity_type, row.merged_into,
                                         row.metadata))(backend.get_entity(e.id))
                     for e in entities},
        "aliases": {e.id: sorted(backend.entity_aliases(e.id)) for e in entities},
        "memories": {e.id: sorted(m.id for m in backend.entity_memories(e.id, limit=100))
                     for e in entities},
        "mentions": sorted((m.id, m.entity_id, m.surface) for e in entities
                           for m in backend.entity_mentions(e.id)),
        "relations": sorted((r.id, r.subject, r.object) for r in backend.list_relations(ADA)),
        "pairs": sorted((p.id, p.entity_a, p.entity_b, p.status, p.compared_step)
                        for p in backend.list_proposals(ADA, status=None)
                        if {p.entity_a, p.entity_b} != both),
        "tags": {mid: sorted(e.name for e in backend.entities_of_memory(mid, kind="topic"))
                 for mid in ids},
        "vectors": backend.property_vector_hashes(ids),
    }


def test_undoing_a_merge_leaves_both_entities_as_they_were():
    store = _store()
    try:
        world = _world(store)
        before = _state(store, world)
        gmbh, bau = world["gmbh"], world["bau"]
        assert store.merge_entities(gmbh.id, bau.id)
        merged = _state(store, world)
        assert merged["memories"][gmbh.id] == sorted(m.id for m in world["memories"][:4])
        assert merged["relations"] != before["relations"]  # the loop is closed, ends moved
        assert merged["vectors"] != before["vectors"]  # its memories mask the kept names now
        [listed] = store.merges(user_id="ada")
        assert (listed["entity_id"], listed["keep_id"], listed["name"], listed["keep_name"],
                listed["decided"]) == (bau.id, gmbh.id, "Kessler Bau", "Kessler Bau GmbH",
                                       "merged by you")
        assert store.undo_merge(bau.id) == {"undone": True, "entity_id": bau.id,
                                            "keep_id": gmbh.id}
        assert _state(store, world) == before
        [pair] = [p for p in store.backend.list_proposals(ADA, status=None)
                  if {p.entity_a, p.entity_b} == {gmbh.id, bau.id}]
        assert (pair.status, pair.reason) == ("rejected", "undone by you")
        assert store.merges(user_id="ada") == []
        assert store.undo_merge(bau.id)["undone"] is False  # nothing left on record
    finally:
        store.close()


@pytest.mark.parametrize("status, created_at, stays", [
    ("rejected", None, "the kept one's"),  # a decision over an open pair
    ("proposed", "2026-01-01T00:00:00+00:00", "the merged one's"),  # the later answer
])
def test_a_merge_leaves_one_row_a_pair_and_the_undo_puts_the_other_back(
        status, created_at, stays):
    """Kessler Bau and Kessler Bau GmbH were each compared with Kessler
    Roofing. The merge points Bau's pair at GmbH, which had a pair with
    Roofing already: before, one pair had two rows, two answers. One row
    stays, the more decided, of two open ones the later answer, and the undo
    puts the other back."""
    store = _store()
    try:
        world = _world(store)
        gmbh, bau, roofing = world["gmbh"], world["bau"], world["roofing"]
        [bau_row] = store.backend.proposals_of([bau.id])
        gmbh_row = store.backend.add_proposal(MergeProposal(
            entity_a=roofing.id, entity_b=gmbh.id, user_id="ada", status=status,
            confidence=0.1, different=0.8, reason="stub: different",
            decided_at=utcnow() if status == "rejected" else None,
            **({"created_at": created_at} if created_at else {})))
        before = _state(store, world)
        assert store.merge_entities(gmbh.id, bau.id)
        [row] = [p for p in store.backend.list_proposals(ADA, status=None)
                 if roofing.id in (p.entity_a, p.entity_b)]
        assert {row.entity_a, row.entity_b} == {gmbh.id, roofing.id}
        assert row.id == (gmbh_row.id if stays == "the kept one's" else bau_row.id)
        assert row.status == status
        assert store.undo_merge(bau.id)["undone"]
        assert _state(store, world) == before
    finally:
        store.close()


def test_a_memory_saved_since_goes_with_the_name_it_calls_the_entity_by():
    """After the merge two saves attach to Kessler Bau GmbH: one calls it
    "Kessler Bau", a name only the merged entity had, and states where it
    worked; the other calls it by the kept name. Undone, the first memory and
    its relation go to Kessler Bau and the second stays."""
    store = _store(_Judge())
    try:
        world = _world(store)
        gmbh, bau, lane = world["gmbh"], world["bau"], world["lane"]
        assert store.merge_entities(gmbh.id, bau.id)
        [by_old_name] = store._apply_candidates([CandidateFact(
            content="Kessler Bau repaired the fence at Harrow Lane 14",
            entities=["Kessler Bau", "Harrow Lane 14"],
            relations=[{"subject": "Kessler Bau", "predicate": "repaired_at",
                        "object": "Harrow Lane 14"}])], ADA, [])
        [by_kept_name] = store._apply_candidates([CandidateFact(
            content="Kessler Bau GmbH paid for the fence", entities=["Kessler Bau GmbH"])],
            ADA, [])
        assert {e.id for e in store.backend.entities_of_memory(by_old_name.memory_id)} == {
            gmbh.id, lane.id}
        assert store.undo_merge(bau.id)["undone"]

        def of(memory_id):
            return {e.id for e in store.backend.entities_of_memory(memory_id)}

        assert of(by_old_name.memory_id) == {bau.id, lane.id}
        assert of(by_kept_name.memory_id) == {gmbh.id}
        [stated] = [r for r in store.backend.list_relations(ADA) if r.predicate == "repaired_at"]
        assert (stated.subject, stated.object) == (bau.id, lane.id)
    finally:
        store.close()


def test_a_merge_is_refused_while_the_entity_kept_was_merged_again_since():
    store = _store()
    try:
        world = _world(store)
        gmbh, bau, roofing = world["gmbh"], world["bau"], world["roofing"]
        assert store.merge_entities(gmbh.id, bau.id)
        assert store.merge_entities(roofing.id, gmbh.id)
        refused = store.undo_merge(bau.id)
        assert refused == {"undone": False, "reason": '"Kessler Bau GmbH" was merged into '
                                                      '"Kessler Roofing" since: undo that merge first'}
        assert store.backend.resolve_entity_id(bau.id) == roofing.id  # nothing changed
        assert store.undo_merge(gmbh.id)["undone"]
        assert store.undo_merge(bau.id)["undone"]
        assert {store.backend.resolve_entity_id(e.id) for e in (gmbh, bau, roofing)} == {
            gmbh.id, bau.id, roofing.id}
    finally:
        store.close()


def test_a_merge_undone_is_not_made_again_on_the_same_evidence():
    """The judge would merge the two at once; their pair, kept apart by the
    undo, is not compared again, not even by the weekly pass."""
    judge = _Judge(same=0.99, different=0.0)
    store = _store(judge)
    try:
        world = _world(store)
        gmbh, bau = world["gmbh"], world["bau"]
        store.backend.add_proposal(MergeProposal(
            entity_a=gmbh.id, entity_b=bau.id, user_id="ada", reason="not yet compared"))
        assert store.resolve_entities(user_id="ada")["confirmed"] >= 1
        assert store.backend.resolve_entity_id(bau.id) == gmbh.id
        assert store.merges(user_id="ada")[0]["decided"] == "stub: same"
        assert store.undo_merge(bau.id)["undone"]
        store.resolve_entities(user_id="ada")
        store.resolve_entities(user_id="ada")
        assert store.backend.resolve_entity_id(bau.id) == bau.id
        assert store.backend.find_proposal(gmbh.id, bau.id).reason == "undone by you"
    finally:
        store.close()


def test_a_tag_folded_into_a_thing_is_a_tag_again():
    """"kessler" the tag folded into Kessler Bau GmbH: its memory mentioned
    the thing. Undone, the tag is a topic again and files its memory."""
    store = _store()
    try:
        gmbh = _entity(store, "Kessler Bau GmbH")
        tagged = _memory(store, "The site office opens at seven", categories=["kessler"])
        topic = store.backend.topic_entity("kessler", ADA, create=False)
        assert store.merge_entities(gmbh.id, topic.id)
        assert [e.id for e in store.backend.entities_of_memory(tagged.id, kind="any")] == [gmbh.id]
        assert store.undo_merge(topic.id)["undone"]
        assert [e.id for e in store.backend.entities_of_memory(tagged.id, kind="any")] == [topic.id]
        assert store.backend.get_entity(topic.id).merged_into is None
        assert store.backend.get_memory(tagged.id).categories == ["kessler"]
    finally:
        store.close()


def test_the_owner_merged_into_a_person_is_the_owner_again():
    store = _store()
    try:
        owner = _entity(store, "the user", "person")
        store.backend.set_entity_metadata(owner.id, {"owner": True})
        store._upkeep_set("owner_entity", "ada", owner.id)
        _memory(store, "the user sails on weekends", (owner, "the user"))
        mira = _entity(store, "Mira Holt", "person")
        _memory(store, "Mira Holt tiled the bathroom", (mira, "Mira Holt"))
        from memry.intelligence.identity import merge_pair

        assert merge_pair(store.backend, store.backend.get_entity(owner.id), mira)
        assert store.owner_entity("ada").id == mira.id
        assert store.undo_merge(owner.id)["undone"]
        assert store.owner_entity("ada").id == owner.id
        assert not (store.backend.get_entity(mira.id).metadata or {}).get("owner")
    finally:
        store.close()


def test_merges_are_listed_and_undone_over_rest_and_the_cli(monkeypatch, tmp_path, capsys):
    store = _store()
    client = TestClient(create_app(store))
    world = _world(store)
    gmbh, bau = world["gmbh"], world["bau"]
    assert client.post("/api/v1/entities/merge",
                       json={"keep_id": gmbh.id, "merge_id": bau.id}).json()["merged"]
    [row] = client.get("/api/v1/entities/merges").json()
    assert (row["entity_id"], row["keep_name"]) == (bau.id, "Kessler Bau GmbH")
    response = client.post("/api/v1/entities/unmerge", json={"ids": [bau.id, "nope"]}).json()
    assert response["undone"] == 1 and list(response["refused"]) == ["nope"]
    assert client.get("/api/v1/entities/merges").json() == []
    store.close()

    from memry.cli import main

    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    cli = MemoryStore(Config.load())
    world = _world(cli)
    assert cli.merge_entities(world["gmbh"].id, world["bau"].id)
    cli.close()
    assert main(["entities", "merges"]) == 0
    assert [row["entity_id"] for row in json.loads(capsys.readouterr().out)] == [world["bau"].id]
    assert main(["entities", "unmerge", world["bau"].id]) == 0
    assert json.loads(capsys.readouterr().out)["undone"] is True
    assert main(["entities", "unmerge", world["bau"].id]) == 1
    assert json.loads(capsys.readouterr().out)["undone"] is False
