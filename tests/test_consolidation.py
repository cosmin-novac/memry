"""Consolidation: collapse memories that record the same fact more than once.

The motivating case is the identity family a long-lived store drifts into:

    "User is Marcus Vandenberg"
    "The user's name is Marc."
    "User is Marcus Vandenberg (goes by Marc)."

No two are textual duplicates and none contradicts another, so write-time
reconciliation leaves all three. They then spend three of ten result slots on
one fact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import FakeLLM

from memry.config import Config
from memry.intelligence.consolidate import representative, similarity_groups
from memry.models import Memory
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

IDENTITY = [
    ("User is Marcus Vandenberg", 0.95),
    ("The user's name is Marc.", 0.50),
    ("User is Marcus Vandenberg (goes by Marc).", 0.95),
]


def _verdict(same: bool, content: str = "", reason: str = "r") -> str:
    return json.dumps({"same_fact": same, "content": content, "reason": reason})


@pytest.fixture
def store():
    """Seeded without an LLM so write-time reconciliation cannot pre-merge the
    duplicates, which is exactly the state a real store drifts into."""
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def _seed_identity(store):
    for content, importance in IDENTITY:
        store.add(content, user_id="ada", infer=False,
                  categories=["identity"], importance=importance)
    store.llm = FakeLLM()


# ---------------------------------------------------------------- grouping
def test_grouping_is_transitive():
    """A chain merges as one group even when the ends are not alike enough.

    'User is Marcus Vandenberg' and "The user's name is Marc." are far apart; both are
    close to the combined phrasing. Pairwise handling would need two passes.
    """
    vectors = [
        ("a", np.array([1.0, 0.0, 0.0])),
        ("b", np.array([0.8, 0.6, 0.0])),   # close to a, not to c
        ("c", np.array([0.0, 1.0, 0.0])),   # close to b only
        ("far", np.array([0.0, 0.0, 1.0])),
    ]
    groups = similarity_groups(vectors, threshold=0.55)
    assert len(groups) == 1
    assert set(groups[0]) == {"a", "b", "c"}


def test_oversized_groups_are_left_alone():
    vectors = [(str(i), np.array([1.0, 0.01 * i])) for i in range(10)]
    assert similarity_groups(vectors, threshold=0.5, max_group=6) == []


def test_representative_is_the_most_important_then_most_detailed():
    memories = [
        Memory(content="short", importance=0.5, created_at="2026-01-01T00:00:00+00:00"),
        Memory(content="a much longer and more detailed record",
               importance=0.95, created_at="2026-02-01T00:00:00+00:00"),
        Memory(content="brief", importance=0.95, created_at="2026-01-01T00:00:00+00:00"),
    ]
    assert representative(memories).content == "a much longer and more detailed record"


# ---------------------------------------------------------------- store flow
def test_identity_family_consolidates_into_one(store):
    _seed_identity(store)
    assert len(store.get_all(user_id="ada", limit=50)) == 3

    merged = "User is Marcus Vandenberg (goes by Marc)."
    store.llm.queue(_verdict(True, merged, "same person, merged nickname"))
    result = store.consolidate_memories(user_id="ada", threshold=0.25)

    assert result["merged"] == 1
    assert result["superseded"] == 3  # every original is forgotten
    active = store.get_all(user_id="ada", limit=50)
    assert [m.content for m in active] == [merged]
    assert active[0].importance == 0.95
    assert active[0].categories == ["identity"]
    # a genuinely new record, not one of the originals wearing the merged text
    assert active[0].id not in {m.id for m in store.get_all(
        user_id="ada", limit=50, include_invalid=True) if m.invalid_at}
    assert active[0].metadata["consolidated_from"]


def test_superseded_memories_are_kept_and_linked(store):
    _seed_identity(store)
    store.llm.queue(_verdict(True, "User is Marcus Vandenberg (goes by Marc).", "same"))
    store.consolidate_memories(user_id="ada", threshold=0.25)

    merged_memory = store.get_all(user_id="ada", limit=50)[0]
    everything = store.get_all(user_id="ada", limit=50, include_invalid=True)
    dropped = [m for m in everything if m.invalid_at is not None]
    assert len(dropped) == 3
    assert {m.superseded_by for m in dropped} == {merged_memory.id}
    # the merged record inherits the earliest creation date of the family
    assert merged_memory.created_at == min(m.created_at for m in dropped)


def test_different_facts_are_not_merged(store):
    store.add("Meeting with Jonas on Tuesday", user_id="ada", infer=False)
    store.add("Meeting with Jonas on Thursday", user_id="ada", infer=False)
    store.llm = FakeLLM()
    store.llm.queue(_verdict(False, "", "different dates"))
    result = store.consolidate_memories(user_id="ada", threshold=0.25)

    assert result["merged"] == 0
    assert len(store.get_all(user_id="ada", limit=50)) == 2


def test_dry_run_changes_nothing(store):
    _seed_identity(store)
    store.llm.queue(_verdict(True, "User is Marcus Vandenberg (goes by Marc).", "same"))
    preview = store.consolidate_memories(user_id="ada", threshold=0.25, apply=False)

    assert preview["groups"] and preview["groups"][0]["same_fact"] is True
    assert preview["merged"] == 0
    assert len(store.get_all(user_id="ada", limit=50)) == 3


def test_without_an_llm_identical_text_still_collapses():
    """Bulk import skips reconciliation, so restatements land unchecked.

    ``store.add`` would have caught this pair at write time; ``import_verbatim``
    is the route by which textual duplicates actually reach a store.
    """
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        s.import_verbatim(
            [{"content": "User is Marcus Vandenberg", "user_id": "ada"},
             {"content": "User is  Marcus Vandenberg.", "user_id": "ada"}],
            dedup=False,
        )
        assert len(s.get_all(user_id="ada", limit=50)) == 2
        result = s.consolidate_memories(user_id="ada", threshold=0.25)
        assert result["merged"] == 1
        assert len(s.get_all(user_id="ada", limit=50)) == 1
    finally:
        s.close()


def test_without_an_llm_a_restatement_is_never_guessed_away():
    """Similarity alone must not merge: dropping the nickname loses information
    that no later pass can recover."""
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        s.add("User is Marcus Vandenberg", user_id="ada", infer=False)
        s.add("User is Marcus Vandenberg (goes by Marc).", user_id="ada", infer=False)
        result = s.consolidate_memories(user_id="ada", threshold=0.25)
        assert result["merged"] == 0
        contents = {m.content for m in s.get_all(user_id="ada", limit=50)}
        assert "User is Marcus Vandenberg (goes by Marc)." in contents
    finally:
        s.close()


def test_consolidation_is_namespaced(store):
    store.add("User is Marcus Vandenberg", user_id="bob", infer=False)
    _seed_identity(store)
    store.llm.queue(_verdict(True, "User is Marcus Vandenberg (goes by Marc).", "same"))
    store.consolidate_memories(user_id="ada", threshold=0.25)
    assert len(store.get_all(user_id="bob", limit=50)) == 1


def test_only_applies_the_chosen_groups(store):
    """Accepting one proposal is not accepting all of them."""
    store.import_verbatim([
        {"content": "User is Marcus Vandenberg", "user_id": "ada"},
        {"content": "User is  Marcus Vandenberg.", "user_id": "ada"},
        {"content": "Meeting with Jonas on Tuesday", "user_id": "ada"},
        {"content": "Meeting with Jonas  on Tuesday.", "user_id": "ada"},
    ], dedup=False)

    preview = store.consolidate_memories(user_id="ada", threshold=0.9, apply=False)
    groups = [g for g in preview["groups"] if g["same_fact"]]
    assert len(groups) == 2

    result = store.consolidate_memories(
        user_id="ada", threshold=0.9, only=[groups[0]["memory_ids"]]
    )
    assert result["merged"] == 1
    # the group that was not ticked is untouched, both members still active
    active = {m.id for m in store.get_all(user_id="ada", limit=50)}
    assert set(groups[1]["memory_ids"]) <= active
    assert not set(groups[0]["memory_ids"]) & active
