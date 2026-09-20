"""The Forgotten list says what made each memory go, in words."""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.models import MemoryEvent
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def _add(store, content):
    store.add(content, user_id="ada", infer=False)
    return next(m for m in store.get_all(user_id="ada", limit=50) if m.content == content)


def _trigger(store, memory_id):
    return next(
        row["trigger"] for row in store.forgotten(user_id="ada")
        if row["memory"].id == memory_id
    )


def test_a_memory_you_deleted_says_so(store):
    memory = _add(store, "Ada likes tea")
    store.delete(memory.id)
    assert _trigger(store, memory.id) == "You deleted it."


def test_the_forgetting_sweep_gives_its_numbers(store):
    memory = _add(store, "The train was late")
    store.backend.invalidate_memory(memory.id)
    store.backend.add_event(MemoryEvent(
        memory_id=memory.id, event="DELETE", actor="decay",
        reason="decay sweep (effective importance 0.043 < 0.1)"))
    text = _trigger(store, memory.id)
    assert "forgetting sweep" in text and "0.043" in text and "0.1" in text


def test_a_raw_message_that_distilled_into_nothing_explains_itself(store):
    memory = _add(store, "ok thanks")
    store.backend.invalidate_memory(memory.id)
    store.backend.add_event(MemoryEvent(
        memory_id=memory.id, event="SUPERSEDE", actor="system",
        reason="distilled with its context into 0 fact(s)"))
    assert "distilling it produced nothing new" in _trigger(store, memory.id)
