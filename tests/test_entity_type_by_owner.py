"""A type the owner gives an entity on the dashboard stays.

The type of a named thing is the one most of its mentions give, recounted on
every save that types it otherwise and on every merge. A thing the extractor
called a person ("armored samurai", a game's enemy) and the owner then called
a concept would turn back into a person on the next save, so a type the owner
chose is marked and the recounts leave it alone.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from memry.config import Config
from memry.models import TYPE_SET_BY_OWNER, Entity, EntityMention
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


def _store() -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))


def _mention(store: MemoryStore, entity: Entity, text: str, entity_type: str) -> None:
    memory_id = store.add(text, user_id="default", infer=False).actions[0].memory_id
    store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory_id,
                                            surface=entity.name, entity_type=entity_type))


def test_a_type_the_owner_chose_outlasts_the_mentions_that_say_otherwise():
    store = _store()
    try:
        samurai = store.backend.insert_entity(Entity(
            name="armored samurai", entity_type="person", user_id="default"))
        for text in ("ONE CUT has armored samurai", "armored samurai parry"):
            _mention(store, samurai, text, "person")
        typed = store.set_entity_type(samurai.id, "concept")
        assert typed.entity_type == "concept"
        assert typed.metadata[TYPE_SET_BY_OWNER] is True
        # a later save calls it a person again; without the mark the recount
        # of three person mentions would make it a person
        _mention(store, samurai, "an armored samurai blocks", "person")
        assert store.backend.get_entity(samurai.id).entity_type == "concept"
    finally:
        store.close()


def test_without_the_owners_choice_the_mentions_decide_as_before():
    store = _store()
    try:
        shop = store.backend.insert_entity(Entity(
            name="the shop", entity_type="product", user_id="default"))
        for text in ("the shop sells shirts", "the shop ships today"):
            _mention(store, shop, text, "project")
        assert store.backend.get_entity(shop.id).entity_type == "project"
    finally:
        store.close()


def test_a_merge_keeps_the_type_the_owner_chose_on_either_side():
    store = _store()
    try:
        kept = store.backend.insert_entity(Entity(
            name="Rex", entity_type="person", user_id="default"))
        merged = store.backend.insert_entity(Entity(
            name="Rex the dog", entity_type="person", user_id="default"))
        for text in ("Rex barked", "Rex ran", "Rex slept"):
            _mention(store, kept, text, "person")
        _mention(store, merged, "Rex the dog ate", "person")
        store.set_entity_type(merged.id, "other")
        store.merge_entities(kept.id, merged.id)
        assert store.backend.get_entity(kept.id).entity_type == "other"
    finally:
        store.close()


def test_a_tag_is_not_given_a_named_type_and_an_unknown_type_is_refused():
    store = _store()
    try:
        store.add("booked flights", user_id="default", infer=False, categories=["travel"])
        from memry.models import Scope

        tag = store.backend.topic_entity("travel", Scope(user_id="default"), create=False)
        assert store.set_entity_type(tag.id, "concept") is None
        person = store.backend.insert_entity(Entity(
            name="Ada", entity_type="person", user_id="default"))
        try:
            store.set_entity_type(person.id, "topic")
        except ValueError:
            pass
        else:
            raise AssertionError("a named thing does not become a tag here")
    finally:
        store.close()


def test_the_entity_route_changes_the_type_and_says_what_it_is_now():
    store = _store()
    try:
        samurai = store.backend.insert_entity(Entity(
            name="armored samurai", entity_type="person", user_id="default"))
        with TestClient(create_app(store)) as client:
            ok = client.patch(f"/api/v1/entities/{samurai.id}", json={"entity_type": "concept"})
            bad = client.patch(f"/api/v1/entities/{samurai.id}", json={"entity_type": "wizard"})
            none = client.patch(f"/api/v1/entities/{samurai.id}", json={})
            missing = client.patch("/api/v1/entities/nope", json={"entity_type": "concept"})
        assert ok.status_code == 200, ok.text
        assert ok.json()["entity"]["entity_type"] == "concept"
        assert ok.json()["entity"]["name"] == "armored samurai"
        assert bad.status_code == 400 and none.status_code == 400
        assert missing.status_code == 404
    finally:
        store.close()


def test_the_map_marks_the_owner():
    store = _store()
    try:
        store.set_owner_name("default", "Ada Lind")
        ada = store.backend.insert_entity(Entity(
            name="Ada Lind", entity_type="person", user_id="default"))
        store._upkeep_set("owner_entity", "default", ada.id)
        lisbon = store.backend.insert_entity(Entity(
            name="Lisbon", entity_type="place", user_id="default"))
        for text in ("Ada Lind flew to Lisbon", "Ada Lind loved Lisbon"):
            memory_id = store.add(text, user_id="default", infer=False).actions[0].memory_id
            for entity in (ada, lisbon):
                store.backend.add_mention(EntityMention(
                    entity_id=entity.id, memory_id=memory_id, surface=entity.name))
        nodes = {node["label"]: node for node in store.knowledge_map(user_id="default")["entities"]}
        assert nodes["Ada Lind"].get("owner") is True
        assert not nodes["Lisbon"].get("owner")
        # read across accounts (a dashboard without login), the entity's own mark
        nodes = {node["label"]: node for node in store.knowledge_map()["entities"]}
        assert nodes["Ada Lind"].get("owner") is True
        assert not nodes["Lisbon"].get("owner")
    finally:
        store.close()
