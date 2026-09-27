"""Property vectors: each memory embedded with the names of its entities, and
of what those belong to, read as "it", so the linked search compares what a
memory says rather than whom it names (``store.refresh_property_vectors``)."""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.models import CandidateFact, Entity, EntityMention, MergeProposal, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore, _text_hash

NEITHER_BUT = {"a_kind_of_b": 0.9, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
               "b_part_of_a": 0.0, "neither": 0.1}


class _Recording(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(32)
        self.texts: list[str] = []

    def embed(self, texts):
        self.texts.extend(texts)
        return super().embed(texts)


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=_Recording())
    s.config.retrieval.relational_fusion = "linked"
    yield s
    s.close()


def _entity(store, name):
    return store.backend.insert_entity(Entity(name=name, normalized=name.lower(), user_id="ada"))


def _memory(store, content, entities):
    memory = store.backend.insert_memory(
        Memory(content=content, user_id="ada", embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([content])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def test_a_saved_memory_gets_its_property_vector_once_its_mentions_are_attached(store):
    actions = store._apply_candidates(
        [CandidateFact(content="bildy stores its data in SQLite", entities=["bildy"])],
        Scope(user_id="ada"), [])
    memory_id = actions[0].memory_id
    assert "it stores its data in SQLite" in store.embedder.texts
    assert store.backend.property_vector_hashes([memory_id]) == {
        memory_id: (_text_hash("it stores its data in SQLite"), store._property_label())}


def test_nothing_is_computed_while_the_linked_search_is_off(store):
    store.config.retrieval.relational_fusion = "rescue"
    actions = store._apply_candidates(
        [CandidateFact(content="bildy stores its data in SQLite", entities=["bildy"])],
        Scope(user_id="ada"), [])
    assert store.backend.property_vector_hashes([actions[0].memory_id]) == {}


def test_a_memory_whose_text_names_no_entity_keeps_its_ordinary_vector(store):
    note = _memory(store, "The sprint review moved to Friday", [])
    assert store.refresh_property_vectors(user_id="ada") == 0
    assert store.backend.property_vector_hashes([note.id]) == {}


def test_a_new_home_masks_the_things_name_at_the_next_refresh(store):
    """A version's memory often names its product. Once the provider answers
    that bildy v3 is a version of bildy, "bildy" reads "it" there too; a second
    refresh with nothing changed embeds nothing."""
    bildy, v3 = _entity(store, "bildy"), _entity(store, "bildy v3")
    memory = _memory(store, "The third release of bildy added offline mode", [v3])
    assert store.refresh_property_vectors(user_id="ada") == 0  # nothing to mask yet
    store.backend.add_proposal(MergeProposal(
        entity_a=v3.id, entity_b=bildy.id, user_id="ada", confidence=0.7,
        different=0.3, belongs=NEITHER_BUT, compared_step=1))
    assert store.refresh_property_vectors(user_id="ada") == 1
    assert store.embedder.texts[-1] == "The third release of it added offline mode"
    assert store.refresh_property_vectors(user_id="ada") == 0
    assert memory.id in store.backend.property_vectors_of([memory.id], store._property_label())


def test_a_vector_from_another_embedding_model_is_not_read_and_is_replaced(store):
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.backend.set_property_vectors({memory.id: [0.0] * 32}, "old-model",
                                       {memory.id: _text_hash("it runs on Linux")})
    assert store.backend.property_vectors_of([memory.id], store._property_label()) == {}
    assert store.refresh_property_vectors(user_id="ada") == 1
    assert memory.id in store.backend.property_vectors_of([memory.id], store._property_label())


def test_deleting_a_memory_deletes_its_property_vector(store):
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.refresh_property_vectors(user_id="ada")
    assert store.backend.property_vector_hashes([memory.id])
    store.backend.delete_memory(memory.id)
    assert store.backend.property_vector_hashes([memory.id]) == {}


def test_the_weekly_upkeep_refreshes_them(store):
    bildy = _entity(store, "bildy")
    _memory(store, "bildy runs on Linux", [bildy])
    ran = store.run_upkeep_cycle(user_id="ada")
    assert ran.get("property_vectors") == {"embedded": 1}


def test_property_vectors_can_be_stored_short(store):
    """With ``property_dimensions`` a property vector keeps its first numbers,
    at length 1; a vector stored at another length is not read and is
    re-embedded."""
    bildy = _entity(store, "bildy")
    memory = _memory(store, "bildy runs on Linux", [bildy])
    store.refresh_property_vectors(user_id="ada")
    store.config.retrieval.property_dimensions = 8
    assert store.backend.property_vectors_of([memory.id], store._property_label()) == {}
    assert store.refresh_property_vectors(user_id="ada") == 1
    vector = store.backend.property_vectors_of([memory.id], store._property_label())[memory.id]
    assert vector.shape == (8,)
    assert float((vector ** 2).sum()) == pytest.approx(1.0)
