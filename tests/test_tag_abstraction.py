"""Tag abstraction: the LLM clusters existing tags into higher-level ones,
those get written onto member memories, and the synthetic ones are remembered.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

from memry.config import Config
from memry.intelligence.clustering import propose_synthetic_tags
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import _tag_run_due
from memry.store import MemoryStore


def _dt(iso: str):
    from datetime import datetime, timezone

    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


@pytest.fixture
def seeded():
    """A store seeded (verbatim, no LLM) with tags that should cluster, plus a
    scripted LLM swapped in for the abstraction step."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    for content, cats in [
        ("ran 10k", ["running"]), ("ate salad", ["diet"]), ("slept 8h", ["sleep"]),
        ("saw doctor", ["doctor"]), ("got promoted", ["promotion"]),
        ("did interview", ["interview"]), ("one-off", ["random"]),
    ]:
        store.add(content, user_id="ada", infer=False, categories=cats)
    store.llm = FakeLLM()
    yield store
    store.close()


# ---------------------------------------------------------------- unit: propose
def test_propose_filters_bad_clusters():
    llm = FakeLLM()
    llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["running", "diet", "sleep"]},   # good
        {"tag": "running", "members": ["diet", "sleep"]},             # dup of existing tag
        {"tag": "thin", "members": ["running"]},                      # < min_cluster
        {"tag": "ghost", "members": ["nonexistent", "alsofake"]},     # members not real
    ]}))
    tags = [{"category": c, "count": 1} for c in ["running", "diet", "sleep", "doctor"]]
    out = propose_synthetic_tags(llm, tags, existing_synthetic=[], max_new=5, min_cluster=2)
    assert out == [{"tag": "health", "members": ["running", "diet", "sleep"]}]


def test_propose_skips_already_synthetic():
    llm = FakeLLM()
    llm.queue(json.dumps({"clusters": [{"tag": "health", "members": ["running", "diet"]}]}))
    tags = [{"category": c, "count": 1} for c in ["running", "diet"]]
    out = propose_synthetic_tags(
        llm, tags, existing_synthetic=["health"], max_new=5, min_cluster=2
    )
    assert out == []  # 'health' already exists as a synthetic tag


# ---------------------------------------------------------------- store flow
def test_abstract_tags_applies_and_records(seeded):
    store = seeded
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["running", "diet", "sleep", "doctor"]},
        {"tag": "career", "members": ["promotion", "interview"]},
    ]}))
    result = store.abstract_tags(user_id="ada")

    applied = {a["tag"]: a["relations_added"] for a in result["applied"]}
    assert applied == {"health": 4, "career": 2}

    # Parent topics are not copied onto memories; hierarchy expansion is read-time.
    run_mem = next(m for m in store.get_all(user_id="ada", limit=50) if m.content == "ran 10k")
    assert run_mem.categories == ["running"]
    health = store.get_all(user_id="ada", categories=["health"], limit=50)
    assert {memory.content for memory in health} == {
        "ran 10k", "ate salad", "slept 8h", "saw doctor",
    }

    # and it is remembered as synthetic
    recorded = {t.tag: sorted(t.source_tags) for t in store.synthetic_tags(user_id="ada")}
    assert recorded == {
        "health": ["diet", "doctor", "running", "sleep"],
        "career": ["interview", "promotion"],
    }
    # the untouched tag stays untouched
    misc = next(m for m in store.get_all(user_id="ada", limit=50) if m.content == "one-off")
    assert misc.categories == ["random"]


def test_abstract_tags_is_namespaced(seeded):
    store = seeded
    store.add("bob run", user_id="bob", infer=False, categories=["running"])
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["running", "diet", "sleep", "doctor"]},
    ]}))
    store.abstract_tags(user_id="ada")
    # bob's namespace is untouched: no synthetic tags, no rewritten memory
    assert store.synthetic_tags(user_id="bob") == []
    bob_mem = store.get_all(user_id="bob", limit=10)[0]
    assert bob_mem.categories == ["running"]


def test_abstract_tags_no_llm_is_a_safe_noop():
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    for c in ["a", "b", "c", "d", "e", "f", "g"]:
        store.add(c, user_id="ada", infer=False, categories=[c])
    result = store.abstract_tags(user_id="ada")
    assert result["applied"] == [] and "no LLM" in result["skipped"]
    store.close()


def test_abstract_tags_skips_when_too_few_tags(seeded):
    store = MemoryStore(Config(db_path=":memory:"), llm=FakeLLM(), embedder=HashEmbedder(64))
    store.add("x", user_id="ada", infer=False, categories=["only-one"])
    result = store.abstract_tags(user_id="ada")
    assert result["applied"] == [] and "tags" in result["skipped"]
    # no LLM call was needed
    assert store.llm.calls == []
    store.close()


def test_abstract_tags_records_run_and_is_idempotent_on_reruns(seeded):
    store = seeded
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["running", "diet", "sleep", "doctor"]},
    ]}))
    store.abstract_tags(user_id="ada")
    assert store.last_tag_run("ada") is not None

    run_mem = next(m for m in store.get_all(user_id="ada", limit=50) if m.content == "ran 10k")
    tags_before = sorted(run_mem.categories)
    # a second run that re-proposes the same parent must not rewrite memories
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["running", "diet", "sleep", "doctor"]},
    ]}))
    store.abstract_tags(user_id="ada")
    run_mem = next(m for m in store.get_all(user_id="ada", limit=50) if m.content == "ran 10k")
    assert sorted(run_mem.categories) == tags_before  # no duplicate 'health'


# ------------------------------------------------- no recursive abstraction
def test_synthetic_parents_are_not_offered_back_to_abstraction(seeded):
    """A parent must never become a member of a broader parent.

    ``topic_counts`` rolls descendants up, so a synthetic parent carries a
    memory count and looks exactly like an ordinary tag. Feeding that histogram
    back in lets run two cluster 'liver health' and 'weekly gym' into 'health',
    losing the useful level one run at a time.
    """
    store = seeded
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "liver health", "members": ["running", "diet"]},
        {"tag": "weekly gym", "members": ["sleep", "doctor"]},
    ]}))
    store.abstract_tags(user_id="ada")

    # the parents exist and are reachable by filter ...
    assert {t.tag for t in store.synthetic_tags(user_id="ada")} == {
        "liver health", "weekly gym",
    }
    rolled = {row["category"] for row in store.categories(user_id="ada")}
    assert {"liver health", "weekly gym"} <= rolled

    # ... but abstraction's own input never lists them
    direct = {row["category"] for row in store.direct_categories(user_id="ada")}
    assert not ({"liver health", "weekly gym"} & direct)

    # and even if a model names one anyway, it is rejected as a member
    store.llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["liver health", "weekly gym"]},
    ]}))
    result = store.abstract_tags(user_id="ada")
    assert result["applied"] == []
    assert "health" not in {t.tag for t in store.synthetic_tags(user_id="ada")}


def test_propose_rejects_synthetic_members():
    llm = FakeLLM()
    llm.queue(json.dumps({"clusters": [
        {"tag": "health", "members": ["liver health", "weekly gym", "diet"]},
    ]}))
    tags = [{"category": c, "count": 3}
            for c in ["liver health", "weekly gym", "diet"]]
    out = propose_synthetic_tags(
        llm, tags, existing_synthetic=["liver health", "weekly gym"],
        max_new=5, min_cluster=2,
    )
    assert out == []  # only 'diet' survives filtering, below min_cluster


# ---------------------------------------------------------------- scheduler due
def test_tag_run_due():
    now = _dt("2026-07-24T00:00:00+00:00")
    assert _tag_run_due(None, 7.0, now) is True                       # never run
    assert _tag_run_due("2026-07-16T00:00:00+00:00", 7.0, now) is True   # 8 days
    assert _tag_run_due("2026-07-20T00:00:00+00:00", 7.0, now) is False  # 4 days
    assert _tag_run_due("not-a-date", 7.0, now) is True                  # unparseable
