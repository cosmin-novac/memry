"""Removing an entity has to be a decision the user can take back.

Memories have had this for a while: a delete invalidates the record and the
Forgotten tab can bring it back. Entities used to be the exception - the row
and everything pointing at it went for good, so one wrong click on a name with
years of mentions behind it was unrecoverable. These tests pin the way back.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from memry.config import Config
from memry.models import Entity, EntityMention, Memory, Relation
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(
        Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64)
    )


def _seed(store: MemoryStore, user_id: str = "ada") -> dict:
    """One entity with mentions, a metadata alias, a tombstone and an edge."""
    backend = store.backend
    first = backend.insert_memory(Memory(content="Ada runs the workshop",
                                         user_id=user_id))
    second = backend.insert_memory(Memory(content="Ada bought a lathe",
                                          user_id=user_id))
    ada = backend.insert_entity(
        Entity(name="Ada", entity_type="person", user_id=user_id)
    )
    workshop = backend.insert_entity(
        Entity(name="Workshop", entity_type="place", user_id=user_id)
    )
    duplicate = backend.insert_entity(
        Entity(name="Ada L.", entity_type="person", user_id=user_id)
    )
    for memory, surface in ((first, "Ada"), (second, "Ada")):
        backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory.id,
                                          surface=surface))
    backend.add_entity_alias(ada.id, "Adalovelace")
    backend.merge_entities(ada.id, duplicate.id)  # leaves a tombstone alias
    relation = backend.add_relation(Relation(
        subject=ada.id, predicate="runs", object=workshop.id,
        user_id=user_id, memory_id=first.id,
    ))
    return {
        "ada": ada, "workshop": workshop, "duplicate": duplicate,
        "first": first, "second": second, "relation": relation,
    }


def test_retire_hides_the_entity_and_lists_it_as_removed():
    store = _store()
    try:
        seeded = _seed(store)
        assert store.remove_entities([seeded["ada"].id]) == 1

        assert [e.name for e in store.entities(user_id="ada")] == ["Workshop"]
        assert store.backend.get_entity(seeded["ada"].id) is None
        assert store.backend.entity_mentions(seeded["ada"].id) == []
        assert store.relations(user_id="ada") == []

        retired = store.retired_entities(user_id="ada")
        assert [row["name"] for row in retired] == ["Ada"]
        assert retired[0]["entity_id"] == seeded["ada"].id
        assert retired[0]["entity_type"] == "person"
        assert retired[0]["reason"] == "removed by you"
        assert retired[0]["retired_at"]
        # the memories are untouched: an entity is an index over them
        assert len(store.get_all(user_id="ada")) == 2
    finally:
        store.close()


def test_restore_brings_back_entity_mentions_aliases_and_relations():
    store = _store()
    try:
        seeded = _seed(store)
        entity_id = seeded["ada"].id
        aliases_before = set(store.backend.entity_aliases(entity_id))
        store.remove_entities([entity_id])

        assert store.restore_entities([entity_id]) == 1

        entity = store.backend.get_entity(entity_id)
        assert entity is not None and entity.name == "Ada"
        assert entity.entity_type == "person"
        assert {m.memory_id for m in store.backend.entity_mentions(entity_id)} == {
            seeded["first"].id, seeded["second"].id
        }
        assert set(store.backend.entity_aliases(entity_id)) == aliases_before
        assert "Ada L." in aliases_before  # the merge tombstone came back too
        assert "Adalovelace" in aliases_before
        relations = store.relations(user_id="ada")
        assert [(r.subject, r.predicate, r.object) for r in relations] == [
            (entity_id, "runs", seeded["workshop"].id)
        ]
        # and it is no longer in the trash
        assert store.retired_entities(user_id="ada") == []
    finally:
        store.close()


def test_restore_skips_mentions_of_memories_that_are_gone():
    store = _store()
    try:
        seeded = _seed(store)
        entity_id = seeded["ada"].id
        store.remove_entities([entity_id])
        # the memory itself is deleted for good while the entity sits in trash
        assert store.backend.delete_memory(seeded["second"].id) is True

        assert store.restore_entities([entity_id]) == 1

        mentions = store.backend.entity_mentions(entity_id)
        assert [m.memory_id for m in mentions] == [seeded["first"].id]
        # the surface it was known by is not lost with its evidence
        assert "Ada" in store.backend.entity_aliases(entity_id)
    finally:
        store.close()


def test_restore_skips_edges_whose_other_end_is_gone():
    store = _store()
    try:
        seeded = _seed(store)
        entity_id = seeded["ada"].id
        store.remove_entities([entity_id])
        store.remove_entities([seeded["workshop"].id])

        assert store.restore_entities([entity_id]) == 1
        assert store.relations(user_id="ada") == []
        assert store.backend.get_entity(entity_id) is not None
    finally:
        store.close()


def test_restore_refuses_when_the_id_is_in_use_again():
    store = _store()
    try:
        seeded = _seed(store)
        entity_id = seeded["ada"].id
        store.remove_entities([entity_id])
        store.backend.insert_entity(
            Entity(id=entity_id, name="Ada again", user_id="ada")
        )

        assert store.restore_entities([entity_id]) == 0
        assert store.backend.get_entity(entity_id).name == "Ada again"
        # the snapshot is still there, untouched, rather than silently dropped
        assert [row["name"] for row in store.retired_entities(user_id="ada")] == ["Ada"]
    finally:
        store.close()


def test_preserving_tag_removal_is_recoverable_too():
    store = _store()
    try:
        seeded = _seed(store)
        result = store.remove_entity_preserving_tag(seeded["ada"].id)
        assert result["removed"] == 1
        assert result["tag"] == "ada"

        retired = store.retired_entities(user_id="ada")
        assert [row["name"] for row in retired] == ["Ada"]
        assert retired[0]["reason"] == "removed by you, name kept as a tag"
        assert store.restore_entities([seeded["ada"].id]) == 1
    finally:
        store.close()


def test_orphan_purge_and_mechanical_junk_are_recoverable():
    store = _store()
    try:
        store.backend.insert_entity(
            Entity(name="Nothing points here", entity_type="concept", user_id="ada")
        )
        memory = store.backend.insert_memory(
            Memory(content="the invoice was 2019", user_id="ada")
        )
        junk = store.backend.insert_entity(
            Entity(name="2019", entity_type="concept", user_id="ada")
        )
        store.backend.add_mention(EntityMention(
            entity_id=junk.id, memory_id=memory.id, surface="2019"
        ))

        outcome = store.resolve_entities(user_id="ada")
        assert outcome["purged"] == 1
        assert outcome["junk_removed"] == 1
        assert store.entities(user_id="ada") == []

        reasons = {
            row["name"]: row["reason"]
            for row in store.retired_entities(user_id="ada")
        }
        assert reasons["Nothing points here"] == "nothing referenced it"
        assert reasons["2019"] and reasons["2019"] != "nothing referenced it"
        assert store.restore_entities([junk.id]) == 1
        assert [e.name for e in store.entities(user_id="ada")] == ["2019"]
    finally:
        store.close()


def test_owner_prefix_gates_removal_listing_and_restore():
    store = _store()
    try:
        mine = _seed(store, user_id="acme::default")
        theirs = _seed(store, user_id="globex::default")

        # a foreign caller can neither remove nor restore
        assert store.remove_entities([mine["ada"].id], owner_prefix="globex::") == 0
        assert store.backend.get_entity(mine["ada"].id) is not None

        assert store.remove_entities([mine["ada"].id], owner_prefix="acme::") == 1
        assert store.remove_entities([theirs["ada"].id], owner_prefix="globex::") == 1

        assert [row["entity_id"] for row in
                store.retired_entities(user_id="acme::default")] == [mine["ada"].id]
        assert store.restore_entities(
            [mine["ada"].id], owner_prefix="globex::"
        ) == 0
        assert store.backend.get_entity(mine["ada"].id) is None
        assert store.restore_entities([mine["ada"].id], owner_prefix="acme::") == 1
        assert store.backend.get_entity(mine["ada"].id) is not None
    finally:
        store.close()


def test_rest_round_trip_removes_lists_and_restores():
    store = _store()
    try:
        seeded = _seed(store)
        with TestClient(create_app(store)) as client:
            removed = client.post(
                "/api/v1/entities/remove", json={"ids": [seeded["ada"].id]}
            )
            listed = client.get("/api/v1/entities/retired")
            empty = client.post("/api/v1/entities/restore", json={"ids": []})
            restored = client.post(
                "/api/v1/entities/restore", json={"ids": [seeded["ada"].id]}
            )
            listed_after = client.get("/api/v1/entities/retired")
            entities = client.get(
                "/api/v1/entities", params={"user_id": "ada"}
            ).json()

        assert removed.json() == {"removed": 1}
        # the server may retire other names of its own accord (an orphan pass
        # runs on startup), so look for this one rather than the whole list
        row = next(r for r in listed.json() if r["entity_id"] == seeded["ada"].id)
        assert row["name"] == "Ada"
        assert row["reason"] == "removed by you"
        assert empty.status_code == 400
        assert restored.json() == {"restored": 1}
        assert seeded["ada"].id not in {r["entity_id"] for r in listed_after.json()}
        assert "Ada" in {row["name"] for row in entities}
    finally:
        store.close()
