"""PostgresBackend contract tests.

Run against a real Postgres with pgvector by setting:

    MEMRY_TEST_POSTGRES_DSN=postgresql://postgres:pw@localhost:5432/memry_test

e.g.  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg16

Skipped otherwise (no Docker on CI-less dev boxes). The same contract is
exercised for LocalBackend throughout the rest of the suite.
"""

from __future__ import annotations

import os

import pytest

from memry.models import Entity, Memory, MemoryEvent, Scope
from memry.providers.embeddings import HashEmbedder

DSN = os.environ.get("MEMRY_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="MEMRY_TEST_POSTGRES_DSN not set (needs Postgres + pgvector)"
)


@pytest.fixture
def backend():
    from memry.backends.postgres import PostgresBackend

    class Cfg:
        postgres_dsn = DSN

    b = PostgresBackend(Cfg())
    b.reset()
    yield b
    b.reset()
    b.close()


def test_memory_crud_and_temporal(backend):
    memory = backend.insert_memory(Memory(content="lives in munich", user_id="ada"))
    assert backend.get_memory(memory.id).content == "lives in munich"

    backend.update_memory(memory.id, content="lives in amsterdam", importance=0.9)
    updated = backend.get_memory(memory.id)
    assert updated.content == "lives in amsterdam"
    assert updated.importance == 0.9

    backend.invalidate_memory(memory.id, superseded_by="next-id")
    assert backend.list_memories(Scope(user_id="ada")) == []
    archived = backend.list_memories(Scope(user_id="ada"), include_invalid=True)[0]
    assert archived.superseded_by == "next-id"


def test_keyword_and_vector_search(backend):
    emb = HashEmbedder(64)
    texts = ["the user lives in berlin", "the user has a cat named miso"]
    for text in texts:
        backend.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=emb.model_id),
            emb.embed([text])[0],
        )
    kw = backend.keyword_search("berlin", Scope(user_id="ada"))
    assert kw and kw[0][0].content == "the user lives in berlin"

    query = emb.embed(["which city does the user live in"])[0]
    vec = backend.vector_search(query, emb.model_id, Scope(user_id="ada"), limit=2)
    assert vec[0][0].content == "the user lives in berlin"


def test_categories_filter(backend):
    backend.insert_memory(
        Memory(content="vegetarian", user_id="ada", categories=["diet"])
    )
    backend.insert_memory(
        Memory(content="typescript", user_id="ada", categories=["tooling"])
    )
    hits = backend.list_memories(Scope(user_id="ada"), categories=["DIET"])
    assert [m.content for m in hits] == ["vegetarian"]


def test_events_and_entities(backend):
    backend.add_event(MemoryEvent(memory_id="m1", event="ADD", new_content="x"))
    assert [e.event for e in backend.history("m1")] == ["ADD"]

    entity_a = backend.insert_entity(Entity(name="Jonas", user_id="ada"))
    entity_b = backend.insert_entity(Entity(name="Jonas", user_id="ada"))
    assert len(backend.find_entities("jonas", Scope(user_id="ada"))) == 2

    from memry.models import MergeProposal

    proposal = backend.add_proposal(
        MergeProposal(entity_a=entity_a.id, entity_b=entity_b.id, user_id="ada")
    )
    assert backend.find_proposal(entity_b.id, entity_a.id) is not None

    assert backend.merge_entities(entity_a.id, entity_b.id)
    backend.set_proposal_status(proposal.id, "confirmed")
    assert len(backend.list_entities(Scope(user_id="ada"))) == 1
    assert backend.list_proposals(Scope(user_id="ada"), status="proposed") == []

    stats = backend.stats()
    assert stats["backend"] == "postgres"
    assert stats["entities"] == 1
