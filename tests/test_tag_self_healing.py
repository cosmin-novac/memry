"""Self-healing tag vocabulary: detect splits the deterministic pass cannot see.

``obvious_canonical_merges`` catches "constraint"/"constraints". It cannot catch
"liver bloods" beside "liver lab results", which is the split that actually
costs recall: filtering to a fragmented tag drops memories the question needed,
and no ranking recovers them afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from memry.config import Config
from memry.intelligence.clustering import semantic_duplicate_tags
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


def _unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=float)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------- unit
def test_detects_a_split_subject():
    centroids = {
        "liver lab results": _unit(1.0, 0.0, 0.0),
        "liver bloods": _unit(0.99, 0.14, 0.0),   # same region, different words
        "weekly gym": _unit(0.0, 0.0, 1.0),       # unrelated
    }
    counts = {"liver lab results": 9, "liver bloods": 3, "weekly gym": 7}
    pairs = semantic_duplicate_tags(centroids, counts, {})
    assert len(pairs) == 1
    assert pairs[0]["variants"] == ["liver bloods", "liver lab results"]
    # the better-established label is proposed as the survivor
    assert pairs[0]["canonical"] == "liver lab results"


def test_co_applied_tags_are_left_alone():
    """Tags a user puts on the same memory are a deliberate distinction.

    'kitchen remodel' and 'bathroom remodel' sit close in vector space, but
    co-application says the user means different things by them.
    """
    centroids = {"kitchen remodel": _unit(1.0, 0.0), "bathroom remodel": _unit(0.99, 0.14)}
    counts = {"kitchen remodel": 8, "bathroom remodel": 8}
    assert semantic_duplicate_tags(centroids, counts, {}) != []
    together = {("bathroom remodel", "kitchen remodel"): 6}
    assert semantic_duplicate_tags(centroids, counts, together) == []


def test_distant_tags_and_thin_tags_are_ignored():
    centroids = {"a": _unit(1.0, 0.0), "b": _unit(0.0, 1.0)}
    assert semantic_duplicate_tags(centroids, {"a": 5, "b": 5}, {}) == []
    # a tag with a single memory has no reliable centroid
    close = {"a": _unit(1.0, 0.0), "b": _unit(0.999, 0.045)}
    assert semantic_duplicate_tags(close, {"a": 5, "b": 1}, {}) == []


# ---------------------------------------------------------------- store
@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def test_store_finds_duplicates_from_stored_vectors(store):
    for text in ("liver enzyme panel came back high",
                 "liver enzyme panel repeated in May",
                 "liver enzyme panel reviewed with the doctor"):
        store.add(text, user_id="ada", infer=False, categories=["liver lab results"])
    for text in ("liver enzyme panel came back high again",
                 "liver enzyme panel repeated once more"):
        store.add(text, user_id="ada", infer=False, categories=["liver bloods"])
    for text in ("squat session at the gym", "treadmill intervals at the gym"):
        store.add(text, user_id="ada", infer=False, categories=["weekly gym"])

    pairs = store.semantic_tag_duplicates(user_id="ada", threshold=0.80)
    variants = [p["variants"] for p in pairs]
    assert ["liver bloods", "liver lab results"] in variants
    assert not any("weekly gym" in v for v in variants)


def test_duplicates_are_namespaced(store):
    for text in ("liver enzyme panel high", "liver enzyme panel repeated"):
        store.add(text, user_id="ada", infer=False, categories=["liver lab results"])
    for text in ("liver enzyme panel high", "liver enzyme panel repeated"):
        store.add(text, user_id="bob", infer=False, categories=["liver bloods"])
    # each namespace has one tag, so there is no pair to propose
    assert store.semantic_tag_duplicates(user_id="ada", threshold=0.80) == []
    assert store.semantic_tag_duplicates(user_id="bob", threshold=0.80) == []


def test_repair_uses_the_existing_merge_primitive(store):
    for text in ("liver enzyme panel high", "liver enzyme panel repeated",
                 "liver enzyme panel reviewed"):
        store.add(text, user_id="ada", infer=False, categories=["liver lab results"])
    for text in ("liver enzyme panel high again", "liver enzyme panel once more"):
        store.add(text, user_id="ada", infer=False, categories=["liver bloods"])

    pair = store.semantic_tag_duplicates(user_id="ada", threshold=0.80)[0]
    remove = [v for v in pair["variants"] if v != pair["canonical"]]
    store.merge_tags(remove, pair["canonical"], user_id="ada")

    tags = {c["category"]: c["count"] for c in store.categories(user_id="ada")}
    assert tags == {"liver lab results": 5}
    assert store.semantic_tag_duplicates(user_id="ada", threshold=0.80) == []


# ------------------------------------- vocabulary offered back to extraction
def test_vocabulary_is_relevance_selected_once_it_exceeds_the_budget(store):
    """A rare but on-topic tag must still be offered.

    Sending only the most-used tags works until a store passes the budget. After
    that the long tail stops being offered, and an unoffered tag is exactly the
    one that gets a near-synonym coined for it next time its subject comes up.
    """
    from memry.models import Scope

    # one rare, highly specific tag ...
    store.add("liver enzyme panel came back elevated in April", user_id="ada",
              infer=False, categories=["liver lab results"])
    # ... buried under many more-used, unrelated ones
    for i in range(40):
        for j in range(3):
            store.add(f"unrelated note {i}-{j} about logistics", user_id="ada",
                      infer=False, categories=[f"filler topic {i}"])

    scope = Scope(user_id="ada")
    frequent = store._tag_vocabulary(scope, limit=10)
    assert "liver lab results" not in frequent  # frequency alone loses it

    relevant = store._tag_vocabulary(
        scope, text="my liver enzyme results from the hepatology clinic", limit=10
    )
    assert "liver lab results" in relevant
    assert len(relevant) <= 10


def test_vocabulary_stays_within_budget_and_is_unique(store):
    from memry.models import Scope

    for i in range(30):
        store.add(f"note {i}", user_id="ada", infer=False, categories=[f"tag {i}"])
    vocab = store._tag_vocabulary(Scope(user_id="ada"), text="note", limit=8)
    assert len(vocab) == 8
    assert len(set(vocab)) == 8


def test_small_stores_are_unaffected(store):
    """Below the budget every tag is offered, with no embedding work."""
    from memry.models import Scope

    for name in ("liver health", "weekly gym", "2026 taxes"):
        store.add(f"note about {name}", user_id="ada", infer=False, categories=[name])
    vocab = store._tag_vocabulary(Scope(user_id="ada"), text="anything", limit=120)
    assert sorted(vocab) == ["2026 taxes", "liver health", "weekly gym"]
