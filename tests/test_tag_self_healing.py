"""The tag vocabulary offered back to extraction, so tagging stays convergent.

Tags that name one subject twice are entity pairs, compared by the tag
question (``identity.compare_topics``; tests in test_topics.py).
"""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


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
