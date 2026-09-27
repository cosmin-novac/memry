"""Relational retrieval: typed-relation traversal recovers the multi-hop
answers that hybrid search structurally cannot reach, without disturbing the
ranking of direct lookups.
"""

from __future__ import annotations

import pytest

from memry.config import Config
from memry.models import Entity, EntityMention, Memory, Relation, Scope
from memry.providers.embeddings import Embedder, HashEmbedder
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


# ------------------------------------------- versions, parts and siblings
def _belongs(store, child, parent, kind="kind", p=0.9, same=0.3):
    """A compared pair: ``child`` is a version ("kind") or a part of ``parent``."""
    from memry.models import MergeProposal

    answer = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
              "b_part_of_a": 0.0, "neither": 1.0 - p}
    answer[f"a_{kind}_of_b"] = p
    store.backend.add_proposal(MergeProposal(
        entity_a=child.id, entity_b=parent.id, user_id="ada", confidence=same,
        belongs=answer))


@pytest.fixture
def family(store):
    """bildy with two versions and a part, and a namesake that shares a word."""
    names = ["bildy", "bildy v3", "bildy v4", "bildy sync service", "Bildy Bakery"]
    e = {name: _entity(store, name) for name in names}
    _memory(store, "bildy stores its data in SQLite", [e["bildy"].id])
    _memory(store, "bildy runs on Linux and macOS", [e["bildy"].id])
    _memory(store, "bildy v3 added offline mode", [e["bildy v3"].id])
    _memory(store, "bildy v4 stores its data in Postgres", [e["bildy v4"].id])
    _memory(store, "bildy v4 added a timeline view", [e["bildy v4"].id])
    _memory(store, "The bildy sync service is maintained by Omar", [e["bildy sync service"].id])
    _memory(store, "Bildy Bakery sells sourdough", [e["Bildy Bakery"].id])
    _belongs(store, e["bildy v3"], e["bildy"])
    _belongs(store, e["bildy v4"], e["bildy"])
    _belongs(store, e["bildy sync service"], e["bildy"], kind="part")
    _belongs(store, e["Bildy Bakery"], e["bildy"], p=0.0, same=0.02)
    return {name: entity.id for name, entity in e.items()}


def test_a_version_takes_its_things_memories_and_little_of_its_siblings(store, family):
    from memry.intelligence.graph_retrieval import DOWN_KIND, TURN, UP_KIND, activation

    act = activation(store.backend, [family["bildy v4"]], depth=2, mode="directed")
    assert act[family["bildy v4"]] == 1.0
    assert act[family["bildy"]] == pytest.approx(UP_KIND * 0.9)
    assert act[family["bildy v3"]] == pytest.approx(UP_KIND * 0.9 * DOWN_KIND * 0.9 * TURN)
    assert act[family["bildy v3"]] < 0.2
    assert family["Bildy Bakery"] not in act  # P(same) 0.02 is under the floor


def test_a_thing_takes_its_versions_and_parts(store, family):
    from memry.intelligence.graph_retrieval import activation

    act = activation(store.backend, [family["bildy"]], depth=1, mode="directed")
    assert {name for name, eid in family.items() if act.get(eid, 0) >= 0.5} == {
        "bildy", "bildy v3", "bildy v4", "bildy sync service"}


def test_undirected_reaches_siblings_as_strongly_as_the_thing(store, family):
    from memry.intelligence.graph_retrieval import activation

    act = activation(store.backend, [family["bildy v4"]], depth=2, mode="undirected")
    assert act[family["bildy v3"]] > 0.5
    assert family["Bildy Bakery"] not in act


def test_weighted_fusion_puts_a_versions_own_memory_first(store, family):
    store.config.retrieval.relational_mode = "directed"
    store.config.retrieval.relational_fusion = "weighted"
    top = store.search("Where does bildy v4 store its data?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy v4 stores its data in Postgres"
    contents = [r.memory.content for r in top]
    assert contents.index("bildy stores its data in SQLite") < len(contents)


def test_search_keeps_its_old_path_by_default(store):
    cfg = store.config.retrieval
    assert (cfg.relational_mode, cfg.relational_depth, cfg.relational_fusion) == (
        "typed", 2, "rescue")


def test_a_question_about_a_version_is_also_asked_of_its_thing(store, family):
    from memry.intelligence.graph_retrieval import inherited_questions
    from memry.models import Scope

    assert inherited_questions(store.backend, Scope(user_id="ada"),
                               "Which systems does bildy v4 run on?") == [
        ("Which systems does bildy run on?", family["bildy"])]
    # a part inherits nothing: its whole is not asked
    assert inherited_questions(store.backend, Scope(user_id="ada"),
                               "Who maintains the bildy sync service?") == []
    store.config.retrieval.relational_mode = "directed"
    store.config.retrieval.relational_fusion = "inherit"
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=3)
    assert "bildy runs on Linux and macOS" in [r.memory.content for r in top[:2]]


def test_how_a_memory_is_weighed_by_the_entities_it_names():
    from memry.intelligence.graph_retrieval import LOW, link_factor, specificity

    # names nothing: its text alone decides
    assert link_factor([]) == 1.0
    # the entity asked about, or its thing: the text decides between them
    assert link_factor([1.0]) == link_factor([0.72]) == 1.0
    # a sibling version or an unrelated entity falls back
    assert link_factor([0.0]) == LOW
    assert LOW < link_factor([0.11]) < 1.0
    # gated: among memories that answer, the version's own before its thing's
    assert specificity([1.0]) > specificity([0.72]) > specificity([0.11]) > specificity([])


def _with_property_vectors(store):
    """Every memory's property vector: its text with its entities' names
    replaced by "it"."""
    from memry.intelligence.graph_retrieval import mask_names
    from memry.models import Scope

    memories = store.backend.list_memories(Scope(user_id="ada"), limit=1000)
    texts = {m.id: mask_names(m.content, [e.name for e in store.backend.entities_of_memory(m.id)])
             for m in memories}
    vectors = store.embedder.embed(list(texts.values()))
    store.backend.set_property_vectors(dict(zip(texts, vectors)), store.embedder.model_id)


class _ConceptEmbedder(Embedder):
    """Words to a few properties, so a test controls what "states the property
    asked" means: storage, platforms, features, bread. The real embedder's
    quality is the benchmark's to measure, not a unit test's."""

    name, _model, dimensions = "concept", "v1", 5
    CONCEPTS = [{"store", "stores", "data", "database"},
                {"run", "runs", "systems", "linux", "macos"},
                {"added", "add", "feature", "view", "mode"},
                {"sells", "bread", "sourdough"}]

    def embed(self, texts):
        import re

        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([float(len(words & c)) for c in self.CONCEPTS] + [0.1])
        return out


def _linked(store):
    store.embedder = _ConceptEmbedder()
    store.config.retrieval.relational_fusion = "linked"
    store.config.retrieval.relational_depth = 1
    _with_property_vectors(store)


def test_linked_search_takes_the_versions_own_fact_over_its_things(store, family):
    _linked(store)
    top = store.search("Where does bildy v4 store its data?", user_id="ada", limit=3)
    contents = [r.memory.content for r in top]
    assert contents[0] == "bildy v4 stores its data in Postgres"
    assert contents.index("bildy stores its data in SQLite") > 0


def test_linked_search_takes_the_things_fact_where_the_version_has_none(store, family):
    _linked(store)
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy runs on Linux and macOS"
    assert top[0].signals["about"] == pytest.approx(0.72)


def test_linked_search_keeps_the_text_ranking_when_the_query_names_no_hub(store, family):
    before = [r.memory.id for r in store.search("offline mode", user_id="ada", limit=5)]
    _linked(store)
    assert [r.memory.id for r in store.search("offline mode", user_id="ada", limit=5)] == before


def test_names_are_masked_as_whole_words():
    from memry.intelligence.graph_retrieval import aboutness, mask_names

    assert mask_names("Where does bildy v3 store its data?", ["bildy", "bildy v3"]) == \
        "Where does it store its data?"
    assert mask_names("bildy's database moved; rebuildy stays", ["bildy"]) == \
        "its database moved; rebuildy stays"
    assert aboutness([None, 0.72]) == 0.72  # the strongest linked entity
    assert aboutness([None]) == 0.3          # only entities the links do not reach
    assert aboutness([]) == 0.5              # no entity: no evidence either way
