"""The map can draw every memory on its own (entity bundling off): the map
data lists each memory, without its text, with the day it was said, its type
and the drawn entities it mentions, a part counted under its home."""

from __future__ import annotations

from memry.config import Config
from memry.models import Entity, EntityMention
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


def test_each_memory_is_listed_with_its_day_type_and_entities_but_no_text():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        ada = store.backend.insert_entity(Entity(name="Ada", entity_type="person", user_id="default"))
        lis = store.backend.insert_entity(Entity(name="Lisbon", entity_type="place", user_id="default"))
        ids = []
        for text, entities in (("Ada flew to Lisbon", (ada, lis)), ("Ada likes tea", (ada,)),
                               ("Lisbon is sunny", (lis,)), ("Ada met Lisbon friends", (ada, lis))):
            memory_id = store.add(text, user_id="default", infer=False).actions[0].memory_id
            ids.append(memory_id)
            for entity in entities:
                store.backend.add_mention(EntityMention(
                    entity_id=entity.id, memory_id=memory_id, surface=entity.name))
        data = store.knowledge_map(user_id="default")
        points = {point["id"]: point for point in data["memory_points"]}
        assert set(points) == set(ids)
        first = points[ids[0]]
        assert set(first["entities"]) == {f"entity:{ada.id}", f"entity:{lis.id}"}
        assert first["said"] and first["type"]
        assert "content" not in first and "Lisbon" not in str(first)
        shown = {node["key"] for node in data["entities"]}
        assert all(key in shown for point in points.values() for key in point["entities"])
    finally:
        store.close()
