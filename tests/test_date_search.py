"""Searching and listing by tag and by date window."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import Entity, EntityMention
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore, _within


def _seed(store):
    # created_at is set by the backend to "now"; to test date windows we write
    # then backdate via the backend's update path is not exposed, so we assert
    # on _within directly for date logic and on tag/browse via the store.
    store.add("berlin trip", user_id="ada", infer=False, categories=["travel"])
    store.add("amsterdam move", user_id="ada", infer=False, categories=["travel", "home"])
    store.add("dark mode", user_id="ada", infer=False, categories=["prefs"])
    berlin = next(m for m in store.get_all(user_id="ada") if m.content == "berlin trip")
    entity = store.backend.insert_entity(Entity(
        name="Berlin", normalized="berlin", entity_type="place", user_id="ada"
    ))
    store.backend.add_mention(EntityMention(
        entity_id=entity.id, memory_id=berlin.id, surface="Berlin"
    ))
    return entity.id


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    _seed(s)
    yield s
    s.close()


# ---------------------------------------------------------------- _within unit
def test_within_date_window():
    ts = "2026-07-22T12:30:00+00:00"
    assert _within(ts, None, None) is True
    assert _within(ts, "2026-07-01", None) is True
    assert _within(ts, "2026-08-01", None) is False          # since after
    assert _within(ts, None, "2026-07-22") is True           # until date inclusive of the day
    assert _within(ts, None, "2026-07-21") is False          # day before
    assert _within(ts, "2026-07-22", "2026-07-22") is True   # same-day window
    # unparseable created_at is never filtered out
    assert _within("garbage", "2026-07-01", "2026-07-02") is True


def test_within_handles_naive_created_at():
    assert _within("2026-07-22T12:30:00", "2026-07-22", "2026-07-22") is True


# ---------------------------------------------------------------- browse by tag
def test_search_with_empty_query_browses_by_tag(store):
    hits = store.search("", user_id="ada", categories=["travel"])
    assert {h.memory.content for h in hits} == {"berlin trip", "amsterdam move"}
    # a ranked query still works
    hits = store.search("amsterdam", user_id="ada")
    assert hits and hits[0].memory.content == "amsterdam move"


def test_get_all_filters_by_tag(store):
    travel = store.get_all(user_id="ada", categories=["travel"])
    assert len(travel) == 2
    prefs = store.get_all(user_id="ada", categories=["prefs"])
    assert [m.content for m in prefs] == ["dark mode"]


def test_entity_filter_uses_exact_mentions_and_intersects_with_topics(store):
    entity_id = store.entities(user_id="ada")[0].id
    assert [m.content for m in store.get_all(user_id="ada", entity_id=entity_id)] == [
        "berlin trip"
    ]
    hits = store.search("trip", user_id="ada", entity_id=entity_id)
    assert [hit.memory.content for hit in hits] == ["berlin trip"]
    assert store.search(
        "", user_id="ada", entity_id=entity_id, categories=["prefs"]
    ) == []
    assert store.get_all(user_id="ada", entity_id="missing") == []


def test_get_all_date_window_excludes_everything_in_the_future(store):
    assert store.get_all(user_id="ada", since="2999-01-01") == []
    assert len(store.get_all(user_id="ada", since="2000-01-01")) == 3


# ---------------------------------------------------------------- over REST + MCP
def test_rest_search_accepts_date_and_tag_filters():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    entity_id = _seed(store)
    with TestClient(create_app(store)) as client:
        # tag browse with empty query
        r = client.post("/api/v1/search", json={"query": "", "categories": ["travel"]})
        assert {h["memory"]["content"] for h in r.json()} == {"berlin trip", "amsterdam move"}
        # future date window returns nothing
        r = client.post("/api/v1/search", json={"query": "", "since": "2999-01-01"})
        assert r.json() == []
        # list endpoint takes the same filters as query params
        r = client.get("/api/v1/memories", params={"categories": "prefs"})
        assert [m["content"] for m in r.json()] == ["dark mode"]
        # entity IDs use exact mention links and intersect with topic filters
        r = client.post("/api/v1/search", json={"query": "", "entity_id": entity_id})
        assert [h["memory"]["content"] for h in r.json()] == ["berlin trip"]
        r = client.get("/api/v1/memories", params={
            "entity_id": entity_id, "categories": "prefs"
        })
        assert r.json() == []
        assert client.post(
            "/api/v1/search", json={"query": "", "entity_id": "missing"}
        ).status_code == 404
        page = client.get("/").text
        assert all(marker in page for marker in (
            'id="filter-date"', 'id="filter-topic"', 'id="filter-entity"'
        ))
