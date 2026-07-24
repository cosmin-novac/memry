"""Manual tag curation: rename, merge, delete tags across memories."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    s.add("a", user_id="ada", infer=False, categories=["finance", "budget"])
    s.add("b", user_id="ada", infer=False, categories=["financial"])
    s.add("c", user_id="ada", infer=False, categories=["budget", "travel"])
    yield s
    s.close()


def _tags(store):
    return {c["category"]: c["count"] for c in store.categories(user_id="ada")}


def test_rename_tag(store):
    assert store.rename_tag("budget", "money", user_id="ada") == 2
    tags = _tags(store)
    assert "budget" not in tags and tags["money"] == 2


def test_merge_tags_combines_and_dedups(store):
    # finance + financial -> finance, across both memories, no duplicate tag
    changed = store.merge_tags(["finance", "financial"], "finance", user_id="ada")
    assert changed == 2
    tags = _tags(store)
    assert "financial" not in tags and tags["finance"] == 2
    a = next(m for m in store.get_all(user_id="ada", limit=10) if m.content == "a")
    assert a.categories.count("finance") == 1  # not doubled


def test_delete_tag_leaves_memories(store):
    before = len(store.get_all(user_id="ada", limit=10))
    assert store.delete_tag("travel", user_id="ada") == 1
    assert "travel" not in _tags(store)
    assert len(store.get_all(user_id="ada", limit=10)) == before  # memory kept


def test_curation_is_namespaced(store):
    store.add("bob", user_id="bob", infer=False, categories=["budget"])
    store.rename_tag("budget", "money", user_id="ada")
    # bob's identical tag is untouched
    assert store.get_all(user_id="bob", limit=5)[0].categories == ["budget"]


def test_curating_drops_the_synthetic_marker():
    from memry.models import SyntheticTag

    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    s.add("x", user_id="ada", infer=False, categories=["running"])
    s.backend.record_synthetic_tag(SyntheticTag(tag="running", source_tags=["a"], user_id="ada"))
    assert [t.tag for t in s.synthetic_tags(user_id="ada")] == ["running"]
    s.rename_tag("running", "jogging", user_id="ada")
    # once the user renames it, it is no longer a system-owned synthetic tag
    assert s.synthetic_tags(user_id="ada") == []
    s.close()


def test_rest_tag_edit_endpoint():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    store.add("a", user_id="default", infer=False, categories=["finance", "budget"])
    store.add("b", user_id="default", infer=False, categories=["financial"])
    with TestClient(create_app(store)) as client:
        r = client.post("/api/v1/tags/edit",
                        json={"op": "merge", "tags": ["finance", "financial"], "to": "finance"})
        assert r.status_code == 200 and r.json()["memories_changed"] == 2
        cats = {c["category"] for c in client.get("/api/v1/categories").json()}
        assert "financial" not in cats and "finance" in cats

        r = client.post("/api/v1/tags/edit", json={"op": "delete", "tag": "budget"})
        assert r.json()["memories_changed"] == 1

        assert client.post("/api/v1/tags/edit", json={"op": "bogus"}).status_code == 400
