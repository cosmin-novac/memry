"""Canonicalization suggestions and RAPTOR-lite collection summaries."""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def seeded():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(96))
    for content, cats in [("a", ["finance", "budget"]), ("b", ["financial"]),
                          ("c", ["projects"]), ("d", ["project"]), ("e", ["running"])]:
        s.add(content, user_id="u", infer=False, categories=cats)
    yield s
    s.close()


# ---------------------------------------------------------------- canonicalize
def test_suggest_merges_keeps_only_real_variant_groups(seeded):
    seeded.llm = FakeLLM()
    seeded.llm.queue(json.dumps({"groups": [
        {"canonical": "finance", "variants": ["finance", "financial"]},
        {"canonical": "project", "variants": ["project", "projects"]},
        {"canonical": "x", "variants": ["ghost"]},            # not real -> dropped
        {"canonical": "running", "variants": ["running"]},    # size 1 -> dropped
    ]}))
    groups = seeded.suggest_tag_merges(user_id="u")
    assert groups == [
        {"canonical": "finance", "variants": ["finance", "financial"]},
        {"canonical": "project", "variants": ["project", "projects"]},
    ]


def test_suggest_merges_no_llm_is_empty():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    s.add("a", user_id="u", infer=False, categories=["x", "y"])
    assert s.suggest_tag_merges(user_id="u") == []
    s.close()


# ---------------------------------------------------------------- collections
def test_build_collections_clusters_and_bounds_llm_calls():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(128))
    for i in range(6):
        s.add(f"The Helios project uses Postgres for storage, note {i}",
              user_id="u", infer=False)
    for i in range(6):
        s.add(f"Ada follows a low-carb diet for her health, entry {i}",
              user_id="u", infer=False)
    s.llm = FakeLLM()
    for _ in range(8):
        s.llm.queue(json.dumps({"title": "Cluster", "summary": "A grounded summary."}))

    res = s.build_collections(user_id="u")
    assert res["collections"] == 2
    cols = s.collections(user_id="u")
    assert {len(c.memory_ids) for c in cols} == {6}
    assert len(s.llm.calls) == 2  # one summarize call per cluster, no more
    s.close()


def test_build_collections_rebuilds_idempotently():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(128))
    for i in range(6):
        s.add(f"Shared topic memory number {i} about the same thing", user_id="u", infer=False)
    s.llm = FakeLLM()
    for _ in range(16):
        s.llm.queue(json.dumps({"title": "T", "summary": "S"}))
    s.build_collections(user_id="u")
    n1 = len(s.collections(user_id="u"))
    s.build_collections(user_id="u")  # rebuild
    assert len(s.collections(user_id="u")) == n1  # cleared + rebuilt, not doubled
    s.close()


def test_build_collections_no_llm_is_safe_noop():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    for i in range(6):
        s.add(f"note {i}", user_id="u", infer=False)
    res = s.build_collections(user_id="u")
    assert res["collections"] == 0 and "no LLM" in res["skipped"]
    s.close()
