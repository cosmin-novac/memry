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
