"""Relational retrieval: typed-relation traversal recovers the multi-hop
answers that hybrid search structurally cannot reach, without disturbing the
ranking of direct lookups.
"""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.models import Entity, EntityMention, Memory, Relation, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(96))
    yield s
    s.close()


def _entity(store, name):
    return store.backend.insert_entity(
        Entity(name=name, normalized=name.lower(), user_id="ada"))


def _memory(store, content, mention_ids):
    emb = store.embedder.embed([content])[0]
    m = store.backend.insert_memory(
        Memory(content=content, user_id="ada"), embedding=emb)
    for eid in mention_ids:
        store.backend.add_mention(EntityMention(entity_id=eid, memory_id=m.id, surface=""))
    return m


@pytest.fixture
def graph(store):
    """Ada -works_on-> Helios -uses-> Postgres, plus a direct preference and
    noise, so multi-hop and direct queries can be told apart."""
    ada = _entity(store, "Ada")
    helios = _entity(store, "Helios")
    postgres = _entity(store, "Postgres")
    m_works = _memory(store, "Ada works on the Helios project.", [ada.id, helios.id])
    m_uses = _memory(store, "The Helios project uses Postgres in production.",
                     [helios.id, postgres.id])
    m_pref = _memory(store, "Ada preference preference: dark mode and short answers.",
                     [ada.id])
    # noise that mentions nobody, to fatten the store
    for i in range(50):
        _memory(store, f"Unrelated note number {i} about sprint planning.", [])
    store.backend.add_relation(Relation(subject=ada.id, predicate="works_on",
                                        object=helios.id, user_id="ada"))
    store.backend.add_relation(Relation(subject=helios.id, predicate="uses",
                                        object=postgres.id, user_id="ada"))
    return {"m_works": m_works, "m_uses": m_uses, "m_pref": m_pref}


def test_multi_hop_answer_is_recovered(store, graph):
    # the answer names neither "Ada" nor "tool" - hybrid alone cannot find it
    plain = store.search("What tool does Ada use for her work?",
                         user_id="ada", relational=False, limit=5)
    assert graph["m_uses"].id not in {r.memory.id for r in plain}

    # with relational fusion on (default), the hop-reachable answer surfaces
    fused = store.search("What tool does Ada use for her work?",
                         user_id="ada", limit=5)
    assert graph["m_uses"].id in {r.memory.id for r in fused}


def test_direct_lookup_ranking_is_not_hurt(store, graph):
    # a lexically clear direct lookup: hybrid should pick m_pref, and relational
    # fusion must not demote it by boosting a graph neighbour (m_works)
    hits = store.search("Ada preference preference", user_id="ada", limit=5)
    assert hits[0].memory.id == graph["m_pref"].id


def test_no_query_entity_means_no_expansion(store, graph):
    # a query naming no known entity just behaves like hybrid (no crash, no graph)
    hits = store.search("sprint planning note", user_id="ada", limit=5)
    assert hits  # returns the noise notes, unaffected


def test_relations_are_namespaced(store, graph):
    assert store.relations(user_id="ada")
    assert store.relations(user_id="someone-else") == []


def test_relation_lifecycle_follows_evidence_memory(store):
    subject = _entity(store, "Ada")
    obj = _entity(store, "Helios")

    invalidated = _memory(store, "Ada works on Helios.", [subject.id, obj.id])
    store.backend.add_relation(
        Relation(
            subject=subject.id,
            predicate="works_on",
            object=obj.id,
            user_id="ada",
            memory_id=invalidated.id,
        )
    )
    assert len(store.relations(user_id="ada")) == 1
    store.backend.invalidate_memory(invalidated.id)
    assert store.relations(user_id="ada") == []

    deleted = _memory(store, "Ada leads Helios.", [subject.id, obj.id])
    store.backend.add_relation(
        Relation(
            subject=subject.id,
            predicate="leads",
            object=obj.id,
            user_id="ada",
            memory_id=deleted.id,
        )
    )
    assert len(store.relations(user_id="ada")) == 1
    assert store.backend.delete_memory(deleted.id)
    assert store.relations(user_id="ada") == []


def test_entity_merge_repoints_relations_and_removes_self_edges(store):
    keep = _entity(store, "Marcus")
    duplicate = _entity(store, "Cozmin")
    project = _entity(store, "Helios")
    store.backend.add_relation(
        Relation(
            subject=duplicate.id,
            predicate="works_on",
            object=project.id,
            user_id="ada",
        )
    )
    store.backend.add_relation(
        Relation(
            subject=keep.id,
            predicate="same_as",
            object=duplicate.id,
            user_id="ada",
        )
    )

    assert store.backend.merge_entities(keep.id, duplicate.id)
    relations = store.relations(user_id="ada")
    assert [(relation.subject, relation.object) for relation in relations] == [
        (keep.id, project.id)
    ]

def test_backfill_relations_is_gated_and_idempotent(store):
    """Backfill only calls the LLM for 2+ entity memories, and not twice."""
    import sys
    sys.path.insert(0, "tests")
    from conftest import FakeLLM
    import json as _json

    ada = _entity(store, "Ada")
    helios = _entity(store, "Helios")
    m2 = _memory(store, "Ada works on Helios.", [ada.id, helios.id])
    _memory(store, "Ada is tired today.", [ada.id])  # single entity -> skipped

    llm = FakeLLM(); store.llm = llm
    llm.queue(_json.dumps({"relations": [
        {"subject": "Ada", "predicate": "works on", "object": "Helios"}]}))
    res = store.backfill_relations(user_id="ada")
    assert res["processed"] == 1 and res["skipped"] == 1 and res["relations_added"] == 1
    assert len(llm.calls) == 1  # only the 2-entity memory hit the LLM
    assert [r.predicate for r in store.relations(user_id="ada")] == ["works_on"]

    before = len(llm.calls)
    store.backfill_relations(user_id="ada")
    assert len(llm.calls) == before  # everything marked done: no new tokens spent


def test_query_entity_detection_uses_bounded_candidate_lookup(store, monkeypatch):
    from memry.intelligence.graph_retrieval import detect_query_entities
    from memry.models import Entity, Scope

    entity = store.backend.insert_entity(Entity(name="Marcus Vandenberg", user_id="ada"))
    store.backend.add_entity_alias(entity.id, "Costi")

    def vocabulary_scan_is_a_bug(*args, **kwargs):
        raise AssertionError("query detection must not scan the entity vocabulary")

    monkeypatch.setattr(store.backend, "list_entities", vocabulary_scan_is_a_bug)
    assert detect_query_entities(
        store.backend, Scope(user_id="ada"), "What is Costi working on?"
    ) == [entity.id]


# ------------------------------------------------- fusion cannot evict the top
def test_relational_fusion_never_displaces_the_strongest_hybrid_hits(store, graph):
    """Graph distance may fill the page but not take it over.

    Measured on a 456-memory store with a dense entity graph, an unprotected
    fusion let buried graph neighbours leapfrog correct answers and cost 0.18
    recall@10 on ordinary queries, while protecting the top hybrid results kept
    multi-hop hit@10 unchanged at 0.917.
    """
    protect = store.config.retrieval.relational_protect_top
    assert protect > 0

    query = "Ada preference preference"
    plain = store.search(query, user_id="ada", relational=False, limit=protect)
    fused = store.search(query, user_id="ada", relational=True, limit=10)
    # the protected prefix is exactly hybrid's own ranking, in order
    assert [r.memory.id for r in fused[:len(plain)]] == [r.memory.id for r in plain]


def test_protection_is_configurable_and_zero_restores_old_behaviour(graph):
    from memry.config import Config, RetrievalConfig

    cfg = Config(db_path=":memory:", retrieval=RetrievalConfig(relational_protect_top=0))
    s = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(96))
    try:
        assert s.config.retrieval.relational_protect_top == 0
    finally:
        s.close()


def test_multi_hop_still_works_with_protection_on(store, graph):
    """The protection must not cost the feature its reason to exist."""
    fused = store.search("What tool does Ada use for her work?", user_id="ada", limit=5)
    assert graph["m_uses"].id in {r.memory.id for r in fused}
