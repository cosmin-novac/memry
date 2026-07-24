"""updated_at hygiene: housekeeping must not bump it; a repair rebuilds it from
the audit trail. Plus the maintenance-autorun due check for entity dedup.
"""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.models import Memory
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    yield s
    s.close()


def test_housekeeping_does_not_bump_updated_at(store):
    store.add("Ada works here", user_id="u", infer=False, categories=["work"])
    m = store.get_all(user_id="u")[0]
    # backdate it, then do housekeeping (tag ops) - updated_at must be preserved
    store.backend.set_memory_timestamp(m.id, "2020-01-01T00:00:00+00:00")
    store.rename_tag("work", "job", user_id="u")
    store.merge_tags(["job"], "career", user_id="u")
    store.delete_tag("career", user_id="u")
    assert store.get(m.id).updated_at == "2020-01-01T00:00:00+00:00"


def test_reembed_does_not_bump_updated_at(store):
    store.add("Ada works here", user_id="u", infer=False)
    m = store.get_all(user_id="u")[0]
    store.backend.set_memory_timestamp(m.id, "2020-01-01T00:00:00+00:00")
    store.reindex()
    assert store.get(m.id).updated_at == "2020-01-01T00:00:00+00:00"


def test_content_edit_does_bump_updated_at(store):
    store.add("Ada works here", user_id="u", infer=False)
    m = store.get_all(user_id="u")[0]
    store.backend.set_memory_timestamp(m.id, "2020-01-01T00:00:00+00:00")
    store.update(m.id, content="Ada leads here now")
    assert store.get(m.id).updated_at != "2020-01-01T00:00:00+00:00"


def test_repair_reconstructs_updated_at_from_audit(store):
    store.add("a memory", user_id="u", infer=False)
    m = store.get_all(user_id="u")[0]
    created = m.created_at
    # simulate a past corruption (a backfill that wrongly bumped it forward)
    store.backend.set_memory_timestamp(m.id, "2099-12-31T00:00:00+00:00")
    res = store.repair_updated_at(user_id="u")
    assert res["fixed"] == 1
    # no content edit ever happened, so the true updated_at is created_at
    assert store.get(m.id).updated_at == created


def test_repair_keeps_a_real_edit_time(store):
    store.add("a memory", user_id="u", infer=False)
    m = store.get_all(user_id="u")[0]
    store.update(m.id, content="edited")  # a real UPDATE event exists
    edited = store.get(m.id).updated_at
    store.repair_updated_at(user_id="u")
    # the genuine edit time is preserved (>= created), not reset to creation
    assert store.get(m.id).updated_at == edited


def test_maintenance_due_check():
    from datetime import datetime, timezone
    from memry.rest import _tag_run_due
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    assert _tag_run_due(None, 7.0, now) is True
    assert _tag_run_due("2026-07-10T00:00:00+00:00", 7.0, now) is True
    assert _tag_run_due("2026-07-22T00:00:00+00:00", 7.0, now) is False
