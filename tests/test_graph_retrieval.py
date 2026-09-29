"""Relational retrieval: the linked search follows the links from the
query's entities and recovers the multi-hop answers that hybrid search
structurally cannot reach, without disturbing the ranking of direct lookups.
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

    # with the linked search on (default), the answer one relation away surfaces
    linked = store.search("What tool does Ada use for her work?",
                          user_id="ada", limit=5)
    assert graph["m_uses"].id in {r.memory.id for r in linked}


def test_direct_lookup_ranking_is_not_hurt(store, graph):
    # a lexically clear direct lookup: hybrid should pick m_pref, and the linked
    # search must not demote it below a graph neighbour (m_works)
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
    _memory(store, "Ada works on Helios.", [ada.id, helios.id])
    _memory(store, "Ada is tired today.", [ada.id])  # single entity -> skipped

    llm = FakeLLM()
    store.llm = llm
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


# ------------------------------------------- the links cannot evict the top
def test_the_linked_search_never_displaces_the_strongest_hybrid_hits(store, graph):
    """Graph distance may fill the page but not take it over.

    Measured on a 456-memory store with a dense entity graph, an earlier
    fusion of graph neighbours into the text ranking let buried ones leapfrog
    correct answers and cost 0.18 recall@10 on ordinary queries. The linked
    search weighs a memory by how strongly it is about the entity asked, so a
    neighbour reached through a relation (``LINKED_RELATION``) fills the page
    after hybrid's own top hits about Ada, which keep their order.
    """
    query = "Ada preference preference"
    plain = store.search(query, user_id="ada", relational=False, limit=5)
    linked = store.search(query, user_id="ada", relational=True, limit=10)
    assert [r.memory.id for r in linked[:len(plain)]] == [r.memory.id for r in plain]
    assert graph["m_uses"].id in {r.memory.id for r in linked[len(plain):]}


def test_multi_hop_works_at_the_default_depth(store, graph):
    """The linked search goes one link deep by default: the multi-hop answer
    names Helios, one relation from Ada, and is about her question at the
    relation's weight."""
    from memry.intelligence.graph_retrieval import LINKED_RELATION

    assert store.config.retrieval.relational_depth == 1
    linked = store.search("What tool does Ada use for her work?", user_id="ada", limit=5)
    found = {r.memory.id: r for r in linked}
    assert graph["m_uses"].id in found
    assert found[graph["m_uses"].id].signals["about"] == pytest.approx(LINKED_RELATION)


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
    from memry.intelligence.graph_retrieval import DOWN_KIND, TURN, UP_KIND, activation_paths

    act, _ = activation_paths(store.backend, [family["bildy v4"]], depth=2)
    assert act[family["bildy v4"]] == 1.0
    assert act[family["bildy"]] == pytest.approx(UP_KIND * 0.9)
    assert act[family["bildy v3"]] == pytest.approx(UP_KIND * 0.9 * DOWN_KIND * 0.9 * TURN)
    assert act[family["bildy v3"]] < 0.2
    assert family["Bildy Bakery"] not in act  # P(same) 0.02 is under the floor


def test_a_thing_takes_its_versions_and_parts(store, family):
    from memry.intelligence.graph_retrieval import activation_paths

    act, _ = activation_paths(store.backend, [family["bildy"]], depth=1)
    assert {name for name, eid in family.items() if act.get(eid, 0) >= 0.5} == {
        "bildy", "bildy v3", "bildy v4", "bildy sync service"}


def _with_property_vectors(store):
    """Every memory's property vector: its text with its entities' names
    replaced by "it"."""
    from memry.intelligence.graph_retrieval import mask_names
    from memry.models import Scope

    memories = store.backend.list_memories(Scope(user_id="ada"), limit=1000)
    texts = {m.id: mask_names(m.content, [e.name for e in store.backend.entities_of_memory(m.id)])
             for m in memories}
    vectors = store.embedder.embed(list(texts.values()))
    store.backend.set_property_vectors(dict(zip(texts, vectors)), store._property_label())


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


def test_the_linked_search_lifts_a_linked_memory_the_text_ranking_buried(store, family):
    """Notes that repeat the question's words but are about nothing the store
    knows bury the thing's answer in the text ranking; the linked search lifts
    it back to the top, since it is about bildy v4's thing and states the
    property asked."""
    for i in range(30):
        _memory(store, f"Question {i}: which systems does bildy v4 run on? Ask again", [])
    _linked(store)
    plain = [r.memory.content for r in
             store.search("Which systems does bildy v4 run on?", user_id="ada", limit=10,
                          relational=False)]
    assert "bildy runs on Linux and macOS" not in plain
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy runs on Linux and macOS"


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
    assert mask_names("Bildy Bakery sells bildy stickers", ["bildy"], keep=["Bildy Bakery"]) == \
        "Bildy Bakery sells it stickers"
    assert aboutness([None, 0.72]) == 0.72  # the strongest linked entity
    assert aboutness([None]) == 0.3          # only entities the links do not reach
    assert aboutness([0.1]) == 0.3           # a weak link is never below no link
    assert aboutness([]) == 0.3              # no entity: about something else too


def test_linked_search_keeps_the_names_of_a_memory_the_links_do_not_reach(store, family):
    """Only names the links account for are masked. A memory about something
    else is compared by its ordinary vector: with every name masked, "Lena Blum
    works on Project Ekmibo" reads "It works on it", as empty as a question
    that asks no property, and would outscore the family on a roll-up."""
    _linked(store)
    kaven = _entity(store, "Kaven planner")
    other = _memory(store, "Kaven planner is popular with bildy fans", [kaven.id])
    store.backend.set_property_vectors({other.id: [0.0, 1.0, 0.0, 0.0, 0.1]},
                                       store._property_label())  # as if it ran on systems
    results = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=20)
    stranger = next(r for r in results if r.memory.id == other.id)
    assert stranger.signals["property"] < 0.2  # its own vector, not the masked one
    assert results[0].memory.content == "bildy runs on Linux and macOS"


def test_linked_search_can_ask_the_decision_provider_what_answers(store, family):
    """With ``relational_relevance = "jev"`` the provider's probability that a
    memory answers the question replaces the vector similarity for the
    shortlist, and the search does not ask it a second time to re-rank. The
    names the links reach read "it"; another entity's name stays."""
    from memry.providers.decisions import Answer, Answers, NoneDecider

    seen, asked = [], []

    class Judge(NoneDecider):
        available = True
        may_rerank = reranks_by_default = True

        def decide(self, state, questions):
            seen.append(state)
            asked.extend(q.instructions for q in questions.values())
            return Answers({key: Answer(1.0 if key == "property" else 0.9 if "runs on" in
                                        q.instructions else 0.1, {}, 0.9, True)
                            for key, q in questions.items()})

    _linked(store)
    store.decider = Judge()
    store.config.retrieval.relational_relevance = "jev"
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=2)
    assert top[0].memory.content == "bildy runs on Linux and macOS"
    assert top[0].signals["judged"] == pytest.approx(0.9 * (1 - 0.1))  # v4's own: no answer
    assert seen == ["QUESTION: Which systems does it run on?"]  # one call, no re-rank
    ask = "Someone who reads only this memory can answer the question. Memory: "
    assert ask + "it runs on Linux and macOS" in asked
    assert ask + "Bildy Bakery sells sourdough" in asked


class _JevLike:
    """A decision provider that re-ranks by default, as Jev does, and answers
    every memory question 0.5; counts its calls."""

    name, available = "jev", True
    may_rerank = reranks_by_default = True

    def __init__(self):
        self.calls = 0

    def decide(self, state, questions):
        from memry.providers.decisions import Answer, Answers

        self.calls += 1
        return Answers({key: Answer(1.0 if key == "property" else 0.0 if key == "several"
                                    else 0.5, {}, 0.9, True) for key in questions})

    def close(self):
        pass


def test_the_decision_provider_that_reranks_judges_the_linked_search_by_default(store, family):
    """``relational_relevance`` is "auto" unless set: with a provider that
    re-ranks (Jev) a question naming a hub is judged; with none it is not, and
    "vector" set explicitly keeps the provider out of it."""
    assert Config().retrieval.relational_relevance == "auto"
    _linked(store)
    question = "Which systems does bildy v4 run on?"
    plain = store.search(question, user_id="ada", limit=3)  # no decision provider
    assert store.relevance_mode() == "vector"
    assert all("about" in r.signals and "judged" not in r.signals for r in plain)

    store.decider = jev = _JevLike()
    assert store.relevance_mode() == "jev"
    judged = store.search(question, user_id="ada", limit=3)
    assert jev.calls == 1 and all("judged" in r.signals for r in judged)

    store.config.retrieval.relational_relevance = "vector"
    assert store.relevance_mode() == "vector"
    kept = store.search(question, user_id="ada", limit=3)
    assert jev.calls == 1  # neither judged nor re-ranked after the linked search
    assert not any("judged" in r.signals for r in kept)


def test_auto_relevance_follows_the_rerank_setting():
    """"auto" is "jev" exactly where the provider re-ranks: Jev unless
    ``decision.rerank`` is off, a text model measured to help only when it is
    on, a provider never measured never."""
    from memry.config import DecisionConfig

    def mode(rerank, reranks_by_default, may_rerank):
        cfg = Config(db_path=":memory:")
        cfg.decision = DecisionConfig(rerank=rerank)
        decider = _JevLike()
        decider.reranks_by_default, decider.may_rerank = reranks_by_default, may_rerank
        s = MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(16), decider=decider)
        try:
            return s.relevance_mode()
        finally:
            s.close()

    assert mode(None, True, True) == "jev"      # Jev
    assert mode(False, True, True) == "vector"  # Jev with re-ranking turned off
    assert mode(None, False, True) == "vector"  # a text model measured to help...
    assert mode(True, False, True) == "jev"     # ...once re-ranking is turned on
    assert mode(True, False, False) == "vector"  # never measured: refused


def test_a_versions_own_answer_overrides_its_things_with_the_decision_provider(store, family):
    """An answer reached by a step up (the thing's) counts only as far as the
    version's own memories do not answer: the version's change wins even when
    the provider is surer of the thing's plainer wording."""
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        available = True

        def decide(self, state, questions):
            def value(key, text):
                if key == "property":
                    return 1.0
                return 0.6 if "Postgres" in text else 0.9 if "SQLite" in text else 0.05
            return Answers({key: Answer(value(key, q.instructions), {}, 0.9, True)
                            for key, q in questions.items()})

    _linked(store)
    store.decider = Judge()
    store.config.retrieval.relational_relevance = "jev"
    top = store.search("Where does bildy v4 store its data?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy v4 stores its data in Postgres"
    thing = next(r for r in top if r.memory.content == "bildy stores its data in SQLite")
    assert thing.signals["overridden"] == pytest.approx(0.6)
    assert thing.signals["judged"] == pytest.approx(0.9 * 0.4)


def test_only_steps_up_count_as_inherited(store, family):
    """The thing and siblings through it are reached by a step up from a
    version; a thing's versions and parts are reached by steps down."""
    from memry.intelligence.graph_retrieval import activation_paths

    _, above = activation_paths(store.backend, [family["bildy v4"]], depth=2)
    assert {family["bildy"], family["bildy v3"]} <= above
    assert family["bildy v4"] not in above
    _, above = activation_paths(store.backend, [family["bildy"]], depth=1)
    assert above == set()


def test_it_stands_for_one_entity_in_what_the_decision_provider_reads(store, family):
    """"it" is the entity a memory is about and what that entity belongs to;
    someone linked by a relation keeps their name where the memory is about
    the query's entity ("Kai Lund works on it", not "it works on it")."""
    from memry.providers.decisions import Answer, Answers, NoneDecider

    asked = []

    class Judge(NoneDecider):
        available = True

        def decide(self, state, questions):
            asked.extend(q.instructions.split("Memory: ", 1)[1] for key, q in questions.items()
                         if key.startswith("m"))
            return Answers({key: Answer(0.5, {}, 0.9, True) for key in questions})

    kai = _entity(store, "Kai Lund")
    store.backend.add_relation(Relation(subject=kai.id, predicate="works_on",
                                        object=family["bildy"], user_id="ada"))
    _memory(store, "Kai Lund works on bildy", [kai.id, family["bildy"]])
    _memory(store, "Kai Lund prefers tea with bildy stickers", [kai.id])
    _memory(store, "With its third release, bildy added sync", [family["bildy v3"]])
    _linked(store)
    store.decider = Judge()
    store.config.retrieval.relational_relevance = "jev"
    store.search("What do I know about bildy?", user_id="ada", limit=5)
    assert "Kai Lund works on it" in asked
    assert "it prefers tea with bildy stickers" in asked  # about Kai, who belongs to nothing
    assert "With its third release, it added sync" in asked  # v3 and the product it is of


def test_a_question_about_everything_is_ordered_by_aboutness(store, family):
    """Relevance and the override are per property. When the provider says
    the question asks for no particular property, the version's own memories
    come first and the product's are not pushed down by them."""
    from memry.providers.decisions import Answer, Answers, NoneDecider

    class Judge(NoneDecider):
        available = True

        def decide(self, state, questions):
            def value(key, text):
                if key == "property":
                    return 0.05
                return 0.9 if "SQLite" in text or "Linux" in text else 0.2
            return Answers({key: Answer(value(key, q.instructions), {}, 0.9, True)
                            for key, q in questions.items()})

    _linked(store)
    store.decider = Judge()
    store.config.retrieval.relational_relevance = "jev"
    top = store.search("Show everything about bildy v4", user_id="ada", limit=4)
    assert {r.memory.content for r in top[:2]} == {"bildy v4 stores its data in Postgres",
                                                   "bildy v4 added a timeline view"}
    thing = next(r for r in top if r.memory.content == "bildy stores its data in SQLite")
    assert thing.signals["judged"] == pytest.approx((0.9 * 0.8) ** 0.05, abs=1e-3)


def test_a_possessive_still_names_the_entity(store):
    """"Ilva Marsh's cat" names Ilva Marsh; a name that ends in 's itself
    ("McDonald's") still matches as written."""
    from memry.intelligence.graph_retrieval import detect_query_entities

    ilva = _entity(store, "Ilva Marsh")
    shop = _entity(store, "McDonald's")
    scope = Scope(user_id="ada")
    assert detect_query_entities(store.backend, scope, "What is Ilva Marsh's cat called?") == [ilva.id]
    assert detect_query_entities(store.backend, scope, "Is McDonald's open late?") == [shop.id]
    assert detect_query_entities(store.backend, scope, "When does Ilva Marsh’s gym open?") == [ilva.id]
    car = _entity(store, "VW ID.3")
    assert detect_query_entities(store.backend, scope, "How much does the VW ID.3 cost?") == [car.id]
    assert detect_query_entities(store.backend, scope, "Ilva Marsh. Where is she?") == [ilva.id]


def test_a_question_in_the_first_person_is_about_the_owner(store, family):
    """"Where do I live?" names nobody: the linked search starts at the store's
    owner. A first-person question that names someone else keeps them."""
    _linked(store)
    owner = _entity(store, "Ilva Marsh")
    lives = _memory(store, "Ilva Marsh lives in Lisbon", [owner.id])
    _memory(store, "Ilva Marsh's sister lives in Porto", [owner.id])
    _with_property_vectors(store)
    store._upkeep_set("owner_entity", "ada", owner.id)
    results = store.search("Where do I live?", user_id="ada", limit=10)
    about = {r.memory.id: r.signals.get("about") for r in results}
    assert about[lives.id] == 1.0
    results = store.search("What do I know about bildy?", user_id="ada", limit=10)
    assert lives.id not in {r.memory.id for r in results if r.signals.get("about") == 1.0}


class _CallJudge:
    """A decision provider for the judging calls: counts its calls, answers
    the meta questions as told and scores a memory by the words it contains."""

    available = True
    may_rerank = reranks_by_default = False

    def close(self):
        pass

    def __init__(self, *, specific, several, scores):
        self.specific, self.several, self.scores, self.calls = specific, several, scores, 0

    def decide(self, state, questions):
        from memry.providers.decisions import Answer, Answers

        self.calls += 1

        def value(key, text):
            if key == "property":
                return self.specific
            if key == "several":
                return self.several
            return next((v for word, v in self.scores.items() if word in text), 0.02)
        return Answers({key: Answer(value(key, q.instructions), {}, 0.9, True)
                        for key, q in questions.items()})


class _KindEmbedder(Embedder):
    """Vectors by kind of fact (a price, an insurance cost, a test drive), as a
    real embedder places them; the car's name does not move them."""

    name, _model, dimensions = "kind", "v1", 4
    KINDS = ["costs", "insurance", "drove"]

    def embed(self, texts):
        return [[float(k in t.lower()) for k in self.KINDS] + [0.1] for t in texts]


def _shopping(store):
    store.embedder = _KindEmbedder()
    store.config.retrieval.relational_relevance = "jev"
    store.config.decision.rerank_pool = 4

    def add(text):
        return store.backend.insert_memory(
            Memory(content=text, user_id="ada", embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([text])[0])

    prices = [add(f"The Carmodel{i} costs {14000 + 900 * i} euros.") for i in range(12)]
    for i in range(12):
        add(f"Insurance for the Carmodel{i} would be {300 + 20 * i} euros a year.")
        add(f"Ada test drove the Carmodel{i} on a rainy day.")
    return prices


def test_an_untagged_set_question_judges_the_memories_nearest_its_members(store):
    """"Which car is the cheapest?" needs every price. Nothing in this store is
    tagged, so the memories of the first call share no topic: the second call
    judges the memories nearest the members it found, up to
    ``retrieval.set_pool``. Every price is a member, and all of them are
    returned, more than the limit."""
    prices = _shopping(store)
    judge = _CallJudge(specific=0.9, several=0.9, scores={"costs": 0.12})
    store.decider = judge
    results = store.search("Which car is the cheapest?", user_id="ada", limit=5)
    members = [r for r in results if r.signals.get("member")]
    assert {r.memory.id for r in members} == {m.id for m in prices}
    assert len(results) >= len(prices) > 5
    assert results[0].signals["calls"] == judge.calls == 2
    assert results[0].signals["pool"] == 36 - 4  # every memory past the first four


def test_a_one_answer_question_is_answered_from_the_first_call(store):
    """Reading on while nothing answered found nothing 5 of 5 times it ran: a
    question with one answer makes one call, answered or not."""
    _shopping(store)
    store.decider = _CallJudge(specific=0.9, several=0.1, scores={"Carmodel3 costs": 0.9})
    store.search("How much does the Carmodel3 cost?", user_id="ada", limit=5)
    assert store.decider.calls == 1  # the answer was in the first call
    store.decider = _CallJudge(specific=0.9, several=0.1, scores={})
    results = store.search("How much does the Carmodel99 cost?", user_id="ada", limit=5)
    assert store.decider.calls == 1  # nothing scores 0.5, and still no second call
    assert results[0].signals["calls"] == 1 and results[0].signals["pool"] == 0


def test_set_members_split_where_the_scores_separate():
    """The scale differs by question; the members are the upper of two groups
    that separate at least twofold, or all of a call's scores above the noise."""
    from memry.intelligence.graph_retrieval import set_members

    cheap = dict(zip("abcdef", [0.16, 0.13, 0.11, 0.09, 0.09, 0.08]))
    noise = dict(zip("uvwxyz", [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]))
    assert set_members({**cheap, **noise}) == set(cheap)
    liked = dict(zip("abc", [0.64, 0.55, 0.42]))
    lunch = dict(zip("uvw", [0.25, 0.22, 0.15]))
    assert set_members({**liked, **lunch}) == set(liked)
    assert set_members(dict(zip("abcd", [0.12] * 4))) == set("abcd")   # a call of members only
    assert set_members(noise) == set()                                 # a call of noise only
    # three tiers: the liked, the merely visited, noise; the top tier is the set
    assert set_members({**liked, **lunch, **dict(zip("pqrst", [0.05, 0.04, 0.03, 0.02, 0.02]))}) \
        == set(liked)


class _Reading(_CallJudge):
    """``_CallJudge`` that keeps each question it was asked and each memory as
    it read it (the questions keyed "m...")."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.states, self.read = [], []

    def decide(self, state, questions):
        self.states.append(state)
        self.read += [q.instructions.split("Memory: ", 1)[1] for key, q in questions.items()
                      if key.startswith("m")]
        return super().decide(state, questions)


def test_a_question_naming_several_things_keeps_their_names(store):
    """"Did Ilva Marsh like Olive Kitchen?" names two things, and "it" could be
    either: the question and the memories the provider reads keep both names
    ("Did it like it?" asks nothing). A question naming one thing reads "it"
    for it, and another entity's name stays."""
    import re

    ilva, olive = _entity(store, "Ilva Marsh"), _entity(store, "Olive Kitchen")
    _memory(store, "Ilva Marsh liked the food at Olive Kitchen", [ilva.id, olive.id])
    _memory(store, "Ilva Marsh had lunch at Olive Kitchen with Kai", [ilva.id, olive.id])
    _memory(store, "Ilva Marsh lives in Lisbon", [ilva.id])
    _linked(store)
    store.config.retrieval.relational_relevance = "jev"

    store.decider = judge = _Reading(specific=0.9, several=0.1, scores={"liked": 0.9})
    top = store.search("Did Ilva Marsh like Olive Kitchen?", user_id="ada", limit=5)
    assert top[0].signals["about"] == 1.0  # the linked search ran, from both
    assert judge.states == ["QUESTION: Did Ilva Marsh like Olive Kitchen?"]
    assert "Ilva Marsh liked the food at Olive Kitchen" in judge.read
    assert "Ilva Marsh had lunch at Olive Kitchen with Kai" in judge.read
    assert not any(re.search(r"\bits?\b", text) for text in judge.read)

    store.decider = judge = _Reading(specific=0.9, several=0.1, scores={"lives": 0.9})
    store.search("Where does Ilva Marsh live?", user_id="ada", limit=5)
    assert judge.states == ["QUESTION: Where does it live?"]
    assert "it lives in Lisbon" in judge.read
    assert "it liked the food at Olive Kitchen" in judge.read


def test_only_a_hub_starts_the_linked_search(store, monkeypatch):
    """A common word stored as an entity with one memory ("budget") is no hub:
    a question containing the word does not start the linked search there. A
    named thing two memories mention ("Harlow") is one and does."""
    import memry.store as store_module

    budget, harlow = _entity(store, "budget"), _entity(store, "Harlow")
    _memory(store, "The budget for the offsite is 4,000 euros", [budget.id])
    _memory(store, "Harlow runs on Linux", [harlow.id])
    _memory(store, "Harlow stores its data in SQLite", [harlow.id])
    _linked(store)
    assert not store._is_hub(budget.id) and store._is_hub(harlow.id)
    seeded = []
    spread = store_module.activation_paths

    def seeds_of(backend, seeds, **kwargs):
        seeded.append(set(seeds))
        return spread(backend, seeds, **kwargs)

    monkeypatch.setattr(store_module, "activation_paths", seeds_of)
    results = store.search("What is the budget?", user_id="ada", limit=10)
    assert results and not any("about" in r.signals for r in results)  # the text ranking
    assert seeded == []
    results = store.search("What is the budget for Harlow?", user_id="ada", limit=10)
    assert seeded == [{harlow.id}]
    about = {r.memory.content: r.signals["about"] for r in results}
    assert about["Harlow runs on Linux"] == 1.0


def test_a_relation_counts_half_in_the_linked_search(store):
    """Ada works on Helios, but "Helios is written in Rust" says nothing about
    Ada: the linked search follows an extracted relation at
    ``LINKED_RELATION`` (0.5). A memory about Helios is about Ada's question
    at 0.5; Ada's own at 1.0."""
    from memry.intelligence.graph_retrieval import (
        LINKED_RELATION, aboutness, activation_paths)

    ada, helios = _entity(store, "Ada"), _entity(store, "Helios")
    store.backend.add_relation(Relation(subject=ada.id, predicate="works_on",
                                        object=helios.id, user_id="ada"))
    own = _memory(store, "Ada prefers dark mode", [ada.id])
    _memory(store, "Ada likes short answers", [ada.id])
    rust = _memory(store, "Helios is written in Rust", [helios.id])

    assert LINKED_RELATION == 0.5
    act, above = activation_paths(store.backend, [ada.id], depth=1)
    assert act == {ada.id: 1.0, helios.id: pytest.approx(0.5)}
    assert above == set()  # a relation is no step up

    def about(memory):
        return aboutness([act.get(e.id) for e in store.backend.entities_of_memory(memory.id)])

    assert about(rust) == pytest.approx(0.5)
    assert about(own) == 1.0
    # and so the store's linked search weighs them
    _linked(store)
    results = store.search("What does Ada prefer?", user_id="ada", limit=10)
    signals = {r.memory.id: r.signals["about"] for r in results}
    assert signals[rust.id] == pytest.approx(0.5)
    assert signals[own.id] == 1.0


def test_only_a_calibrated_answer_makes_a_pair_a_same_link(store):
    """A pair nobody judged (raised "not yet compared" at 0.5, or holding a
    text model's confidence) says nothing about whether the two are one, so
    the linked search gives it no "same" link; before, it drew the other
    entity's memories in at that weight. A calibrated judge's answer, which
    stores P(different) beside P(same), does."""
    from memry.intelligence.graph_retrieval import activation_paths, links_of
    from memry.models import MergeProposal

    kettle, other = _entity(store, "Kettlebay"), _entity(store, "Kettle Bay")
    pair = store.backend.add_proposal(MergeProposal(
        entity_a=kettle.id, entity_b=other.id, user_id="ada", reason="not yet compared"))
    store.backend.update_proposal_judgement(pair.id, confidence=0.9, reason="a guess")
    assert links_of(store.backend, [kettle.id]) == []
    assert activation_paths(store.backend, [kettle.id])[0] == {kettle.id: 1.0}

    store.backend.update_proposal_judgement(
        pair.id, confidence=0.8, reason="stub: same", different=0.1)
    [link] = links_of(store.backend, [kettle.id])
    assert (link.kind, link.p) == ("same", pytest.approx(0.8))


def test_the_linked_search_keeps_to_the_run_searched(store):
    """The memories an entity's family brings into the linked search come from
    the scope searched, as the text ranking's do: a search of run "s2" does
    not return Ada's memory saved under run "s1"; a search of the user does."""
    ada = store.backend.insert_entity(
        Entity(name="Ada", normalized="ada", user_id="ada", run_id="s2"))

    def remember(content, run_id):
        memory = store.backend.insert_memory(
            Memory(content=content, user_id="ada", run_id=run_id),
            embedding=store.embedder.embed([content])[0])
        store.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory.id,
                                                surface="Ada"))
        return memory

    tea = remember("Ada likes green tea", "s1")
    remember("Ada lives in Lisbon", "s2")
    remember("Ada works at Acme", "s2")
    _linked(store)
    in_run = store.search("What does Ada like?", user_id="ada", run_id="s2", limit=10)
    assert in_run and all("about" in r.signals for r in in_run)  # the linked search ran
    assert {r.memory.run_id for r in in_run} == {"s2"}
    assert tea.id not in {r.memory.id for r in in_run}
    everywhere = store.search("What does Ada like?", user_id="ada", limit=10)
    assert tea.id in {r.memory.id for r in everywhere}


def test_a_runs_memories_of_a_large_entity_reach_the_linked_search(store):
    """Ada has 600 memories; the run searched owns only the oldest 50, which
    the text ranking cannot find (no word of the question, no vector). The
    family candidates keep to the run before the newest ``FAMILY_SCAN`` are
    taken, so the search of the run still finds them."""
    from memry.intelligence.graph_retrieval import FAMILY_SCAN

    ada = store.backend.insert_entity(
        Entity(name="Ada", normalized="ada", user_id="ada", run_id="s1"))

    def remember(content, run_id, day):
        stamp = f"2023-{1 + day // 28:02d}-{1 + day % 28:02d}T00:00:00+00:00"
        memory = store.backend.insert_memory(Memory(
            content=content, user_id="ada", run_id=run_id, created_at=stamp,
            updated_at=stamp))
        store.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory.id,
                                                surface="Ada"))
        return memory

    own = {remember(f"Enjoys sourdough bread number {i}", "s1", i // 10).id for i in range(50)}
    for i in range(550):
        remember(f"Other session note {i}", "s2", 10 + i // 10)
    assert 550 > FAMILY_SCAN - 50
    _linked(store)
    assert {m.id for m in store.backend.entity_memories(
        ada.id, limit=FAMILY_SCAN, scope=Scope(user_id="ada", run_id="s1"))} == own
    found = store.search("What does Ada like?", user_id="ada", run_id="s1", limit=10)
    assert len(found) == 10 and {r.memory.id for r in found} <= own


# ------------------------------------ what the judgement does to the order
def _versions_of_the_database(store, *, builds_on):
    """bildy stores its data in SQLite, bildy v3 moved its data to Postgres,
    bildy v4 says nothing about its data; both versions belong to bildy, and
    with ``builds_on`` the provider also answered that v4 is a version of v3.
    The provider is surer of the thing's plainer wording (0.9) than of v3's
    change (0.8)."""
    e = {name: _entity(store, name) for name in ["bildy", "bildy v3", "bildy v4"]}
    _memory(store, "bildy stores its data in SQLite", [e["bildy"].id])
    _memory(store, "bildy runs on Linux and macOS", [e["bildy"].id])
    _memory(store, "bildy v3 moved its data to Postgres", [e["bildy v3"].id])
    _memory(store, "bildy v3 added offline mode", [e["bildy v3"].id])
    _memory(store, "bildy v4 added a timeline view", [e["bildy v4"].id])
    _memory(store, "bildy v4 added dark mode", [e["bildy v4"].id])
    _belongs(store, e["bildy v3"], e["bildy"])
    _belongs(store, e["bildy v4"], e["bildy"])
    if builds_on:
        _belongs(store, e["bildy v4"], e["bildy v3"])
    _linked(store)
    store.config.retrieval.relational_relevance = "jev"
    store.decider = _CallJudge(specific=1.0, several=0.0,
                               scores={"SQLite": 0.9, "Postgres": 0.8})


def test_a_version_takes_the_change_of_the_version_it_builds_on(store):
    """bildy v4 is a version of bildy v3 (depth 1, one judged link) and of
    bildy. Both answers are reached by a step up at 0.72, and v3's change
    overrides bildy's default for v4 as it does for v3 itself: bildy's answer
    counts as far as v3's does not (0.9 x 0.2), and v3's as far as none of
    v4's own memories answers."""
    _versions_of_the_database(store, builds_on=True)
    top = store.search("Where does bildy v4 store its data?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy v3 moved its data to Postgres"
    assert top[0].signals["about"] == pytest.approx(0.72)
    assert top[0].signals["overridden"] == pytest.approx(0.02)  # v4's own say nothing
    thing = next(r for r in top if r.memory.content == "bildy stores its data in SQLite")
    assert thing.signals["overridden"] == pytest.approx(0.8)
    assert thing.signals["judged"] == pytest.approx(0.9 * 0.2)


def test_a_version_linked_only_to_its_thing_takes_the_things_answer(store):
    """Without the link to v3, v3 is a sibling two links away: at depth 1 the
    search does not reach it, its change is about something else (0.3) and
    does not override bildy's default, which answers v4's question."""
    _versions_of_the_database(store, builds_on=False)
    top = store.search("Where does bildy v4 store its data?", user_id="ada", limit=3)
    assert top[0].memory.content == "bildy stores its data in SQLite"
    assert top[0].signals["overridden"] == pytest.approx(0.02)
    sibling = next(r for r in top if r.memory.content == "bildy v3 moved its data to Postgres")
    assert sibling.signals["about"] == pytest.approx(0.3)
    assert "overridden" not in sibling.signals


def test_a_false_yes_on_the_versions_own_memory_leaves_the_inherited_answer_first(store):
    """The provider says yes (0.3) to one of v4's own memories that does not
    answer. That takes 30% off the right answer v4 inherits from bildy, which
    still ranks first: 0.8 x 0.77 x 0.7 = 0.43 against 0.3 x 1.0."""
    bildy, v4 = _entity(store, "bildy"), _entity(store, "bildy v4")
    _memory(store, "bildy runs on Linux and macOS", [bildy.id])
    _memory(store, "bildy stores its data in SQLite", [bildy.id])
    _memory(store, "bildy v4 added a timeline view", [v4.id])
    _memory(store, "bildy v4 added dark mode", [v4.id])
    _belongs(store, v4, bildy, p=0.77 / 0.8)
    _linked(store)
    store.config.retrieval.relational_relevance = "jev"
    store.decider = _CallJudge(specific=1.0, several=0.0,
                               scores={"runs on": 0.8, "timeline": 0.3})
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=3)
    assert [r.memory.content for r in top[:2]] == ["bildy runs on Linux and macOS",
                                                   "bildy v4 added a timeline view"]
    answer, false_yes = top[0].signals, top[1].signals
    assert (answer["about"], answer["overridden"]) == (pytest.approx(0.77), pytest.approx(0.3))
    assert answer["judged"] * answer["about"] == pytest.approx(0.8 * 0.77 * 0.7)
    assert false_yes["judged"] * false_yes["about"] == pytest.approx(0.3)


@pytest.mark.parametrize("specific, first", [
    (0.54, "bildy runs on Linux and macOS"),
    (0.1, "bildy v4"),
])
def test_how_far_a_property_question_read_as_about_everything_keeps_its_answer(
        store, family, specific, first):
    """Relevance and the override count to the power of P(the question asks
    for one property). At 0.54 the answer v4 inherits still ranks
    above v4's own non-answers, (0.8 x 0.95) ** 0.54 x 0.72 = 0.62 against
    0.05 ** 0.54 = 0.20. Read as a question about everything (0.1), a
    property question would lose it to them: 0.70 against 0.74, the limit the
    weighting accepts."""
    _linked(store)
    store.config.retrieval.relational_relevance = "jev"
    store.decider = _CallJudge(specific=specific, several=0.0,
                               scores={"runs on": 0.8, "": 0.05})
    top = store.search("Which systems does bildy v4 run on?", user_id="ada", limit=3)
    assert top[0].memory.content.startswith(first)
    answer = next(r for r in top if r.memory.content == "bildy runs on Linux and macOS")
    assert answer.signals["judged"] * answer.signals["about"] == pytest.approx(
        (0.8 * 0.95) ** specific * 0.72, abs=1e-3)


class _Recording(_CallJudge):
    """``_CallJudge`` that keeps how many memories each call judged."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizes: list[int] = []

    def decide(self, state, questions):
        self.sizes.append(sum(key.startswith("m") for key in questions))
        return super().decide(state, questions)


def _planner(store):
    """Kaven planner's five facts and 16 of Kaven planner v2's own, with v2 a
    version of the planner: more than the 20 the provider reads."""
    planner, v2 = _entity(store, "Kaven planner"), _entity(store, "Kaven planner v2")
    for fact in ("runs on Windows only", "is written in Go", "stores its data in SQLite",
                 "is led by Mara Ruiz", "is released under the MIT license"):
        _memory(store, f"Kaven planner {fact}", [planner.id])
    for i in range(16):
        _memory(store, f"Kaven planner v2 added feature {i} to the timeline", [v2.id])
    _belongs(store, v2, planner)
    _linked(store)
    store.config.retrieval.relational_relevance = "jev"


def test_a_lone_answer_judged_low_still_ranks_first_among_non_answers(store):
    """Jev scored the one right answer "It runs on Windows only" 0.16 to 0.32
    across wordings. At 0.16, with the other 19 memories of the call judged
    near zero (0.02), it still ranks first: through the linked search, where
    the planner's answer counts 0.16 x 0.98 x 0.72 against 0.02 x 1.0 for
    v2's own, and through the re-rank blend of a question naming no hub,
    where 0.16 clears the floor (``decision.rerank_floor`` 0.15) that pushes
    the rest back."""
    _planner(store)
    store.decider = judge = _Recording(specific=1.0, several=0.0,
                                       scores={"Windows only": 0.16})
    top = store.search("Which platforms does Kaven planner v2 run on?", user_id="ada", limit=5)
    assert judge.sizes == [20]
    assert top[0].memory.content == "Kaven planner runs on Windows only"
    assert top[0].signals["judged"] * top[0].signals["about"] == pytest.approx(
        0.16 * 0.98 * 0.72)
    store.decider = judge = _Recording(specific=1.0, several=0.0,
                                       scores={"Windows only": 0.16})
    top = store.search("Which platforms does the planner run on?", user_id="ada", limit=5)
    assert judge.sizes == [20] and "about" not in top[0].signals  # no hub named
    assert top[0].memory.content == "Kaven planner runs on Windows only"


def test_a_non_answer_judged_above_a_lone_answer_comes_first(store):
    """The limit of judging each memory once: where Jev reads a non-answer
    higher than the answer ("It is written in Go" 0.34 against "It runs on
    Windows only" 0.16), the non-answer comes first and the answer second.
    Nothing combines the judgement with the vector to steady it."""
    _planner(store)
    store.decider = _CallJudge(specific=1.0, several=0.0,
                               scores={"Windows only": 0.16, "written in Go": 0.34})
    top = store.search("Which platforms does Kaven planner v2 run on?", user_id="ada", limit=5)
    assert [r.memory.content for r in top[:2]] == ["Kaven planner is written in Go",
                                                   "Kaven planner runs on Windows only"]


# ---------------------------- what only the words of a memory tell the search
class _TopicEmbedder(Embedder):
    """Vectors by what a text is about in a few words ("paid", "decided",
    "pricing"): an identifier ("invoice 2024-117"), a name and "settled"
    move nothing, so an invoice that was settled is near no question about
    paying."""

    name, _model, dimensions = "topic", "v1", 4
    TOPICS = [{"pay", "paid"}, {"decide", "decided"}, {"pricing", "prices"}]

    def embed(self, texts):
        import re

        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([float(len(words & topic)) for topic in self.TOPICS] + [0.01])
        return out


def _remember(store, text, entities=()):
    memory = store.backend.insert_memory(
        Memory(content=text, user_id="ada", embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([text])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


@pytest.mark.parametrize("relevance", ["vector", "jev"])
def test_an_identifier_only_the_words_match_reaches_what_the_provider_reads(store, relevance):
    """"Did Harlow pay invoice 2024-117?" is answered by a memory linked to
    nothing that the vectors do not bring near the question ("settled", and an
    identifier they cannot see); only the keyword search matches it. Harlow's
    30 memories about paying are each nearer and about Harlow (1.0 against
    0.3), and would fill the first 20 by themselves. The keyword search's best
    match keeps a place among the first ``decision.rerank_pool``: with the
    property similarity alone it is in the first 20, and judged it comes
    first (0.9 x 0.3 against 0.05 x 1.0 for Harlow's non-answers)."""
    store.embedder = _TopicEmbedder()
    store.config.retrieval.relational_relevance = relevance
    harlow = _entity(store, "Harlow")
    for i in range(30):
        _remember(store, f"Harlow paid the rent for flat {i} in cash", [harlow])
    invoice = _remember(store, "Invoice 2024-117 was settled by bank transfer on 3 March")
    for i in range(10):
        _remember(store, f"Invoice 2024-{200 + i} was settled late")
    question, scope = "Did Harlow pay invoice 2024-117?", Scope(user_id="ada")
    assert store.backend.keyword_search(question, scope, 5)[0][0].id == invoice.id
    nearest = store.backend.vector_search(store.embedder.embed([question])[0],
                                          store.embedder.model_id, scope, limit=30)
    assert invoice.id not in {memory.id for memory, _ in nearest}
    if relevance == "jev":
        store.decider = _CallJudge(specific=1.0, several=0.0,
                                   scores={"2024-117": 0.9, "": 0.05})
    results = store.search(question, user_id="ada", limit=20)
    ids = [r.memory.id for r in results]
    assert "about" in results[0].signals  # the linked search ran
    assert invoice.id in ids[:store.config.decision.rerank_pool]
    if relevance == "jev":
        assert ids[0] == invoice.id
        assert results[0].signals["about"] == pytest.approx(0.3)


def test_a_first_person_answer_about_something_else_outranks_the_owners_non_answers(store):
    """"What did I decide about the pricing?" starts at the store's owner,
    whose 30 memories about deciding are as near the question as "The team
    raised Pro pricing" and about the owner (1.0), while the answer is not
    (0.3). Judged, the answer counts 0.8 x 0.3 = 0.24 against 0.05 x 1.0 for
    each of the owner's non-answers, and comes first. It shares "pricing"
    with the question: as the keyword search's best match it is among the 20
    the provider reads, which the owner's memories would fill otherwise."""
    store.embedder = _TopicEmbedder()
    store.config.retrieval.relational_relevance = "jev"
    owner = _entity(store, "Ilva Marsh")
    for i in range(30):
        _remember(store, f"Ilva Marsh decided to repaint room {i} and decided on blue", [owner])
    answer = _remember(store, "The team raised Pro pricing")
    store._upkeep_set("owner_entity", "ada", owner.id)
    store.decider = judge = _Reading(specific=1.0, several=0.0,
                                     scores={"pricing": 0.8, "": 0.05})
    results = store.search("What did I decide about the pricing?", user_id="ada", limit=5)
    assert judge.states == ["QUESTION: What did it decide about the pricing?"]
    assert results[0].memory.id == answer.id
    assert results[0].signals["judged"] * results[0].signals["about"] == pytest.approx(0.8 * 0.3)
    assert all(r.signals["about"] == 1.0 and r.signals["judged"] == pytest.approx(0.05)
               for r in results[1:])
