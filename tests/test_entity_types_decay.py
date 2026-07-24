"""Entity types (extracted + backfilled) and type-aware decay behaviour."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import FakeLLM

from memry.config import Config, DecayConfig
from memry.intelligence.decay import effective_importance
from memry.models import Entity, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


# ---------------------------------------------------------------- entity types
def test_entity_type_set_on_save():
    llm = FakeLLM()
    s = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64))
    llm.queue(json.dumps({"facts": [{
        "content": "Ada works on Helios.", "type": "episodic", "importance": 0.7,
        "categories": ["work"],
        "entities": [{"name": "Ada", "type": "person"},
                     {"name": "Helios", "type": "project"}],
        "relations": []}]}))
    llm.queue(json.dumps({"missing": []}))
    s.add("Ada works on Helios", user_id="u")
    types = {e.name: e.entity_type for e in s.backend.list_entities(Scope(user_id="u"))}
    assert types == {"Ada": "person", "Helios": "project"}
    s.close()


def test_legacy_string_entities_still_parse():
    from memry.intelligence.extraction import _parse_entities
    assert _parse_entities(["Ada"]) == {"entities": ["Ada"], "entity_types": {}}


def test_backfill_entity_types_is_batched_and_gated():
    s = MemoryStore(Config(db_path=":memory:"), llm=FakeLLM(), embedder=HashEmbedder(64))
    s.backend.insert_entity(Entity(name="Cosmin", normalized="cosmin", user_id="u"))
    s.backend.insert_entity(Entity(name="Vienna", normalized="vienna", user_id="u"))
    s.backend.insert_entity(Entity(name="Rust", normalized="rust", entity_type="product",
                                   user_id="u"))  # already typed -> ignored
    s.llm.queue(json.dumps({"types": [{"name": "Cosmin", "type": "person"},
                                      {"name": "Vienna", "type": "place"}]}))
    res = s.backfill_entity_types(user_id="u")
    assert res["typed"] == 2
    assert len(s.llm.calls) == 1  # one batched call for both untyped entities
    got = {e.name: e.entity_type for e in s.backend.list_entities(Scope(user_id="u"))}
    assert got == {"Cosmin": "person", "Vienna": "place", "Rust": "product"}
    # rerun: nothing untyped -> no call
    before = len(s.llm.calls)
    s.backfill_entity_types(user_id="u")
    assert len(s.llm.calls) == before
    s.close()


# ---------------------------------------------------------------- typed decay
def test_type_aware_decay_orders_by_persistence():
    cfg = DecayConfig()
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    old = (now - timedelta(days=90)).isoformat()

    def score(t):
        return effective_importance(
            Memory(content="x", importance=1.0, memory_type=t, updated_at=old), cfg, now)

    # procedural persists longest, then semantic, then episodic, then working
    assert score("procedural") > score("semantic") > score("episodic") > score("working")


def test_decay_disabled_is_identity():
    cfg = DecayConfig(enabled=False)
    m = Memory(content="x", importance=0.8, memory_type="episodic",
               updated_at="2020-01-01T00:00:00+00:00")
    assert effective_importance(m, cfg) == 0.8
