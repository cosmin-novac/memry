"""Questions that need several memories ("Which car is the cheapest?"): the
decision provider judges the first ``decision.rerank_pool`` of the ranking and
then, in one more call, the memories filed under the topics those share, up
to ``retrieval.set_pool`` (``MemoryStore._set_pool``). A question with one
answer, or about everything, makes one call. Also: the linked search is the
only one, and a removed mode is refused."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from memry.config import Config, RetrievalConfig
from memry.models import EntityMention, Memory
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

from test_graph_retrieval import _KindEmbedder, _RoundJudge, _entity

ROOT = Path(__file__).resolve().parent.parent
PRICED = "The Carmodel{i} is priced at {n} euros."
QUOTED = "Dealer quote: the Carmodel{i} costs {n} euros."
QUESTION = "Which car is priced lowest?"


class _Batches(_RoundJudge):
    """``_RoundJudge`` that keeps the memories of each call, as it read them."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batches: list[list[str]] = []

    def decide(self, state, questions):
        self.batches.append([q.instructions.split("Memory: ", 1)[1]
                             for key, q in questions.items() if key.startswith("m")])
        return super().decide(state, questions)


@pytest.fixture
def store():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=_KindEmbedder())
    s.config.retrieval.relational_relevance = "jev"
    yield s
    s.close()


def _add(store, text, tags=(), entities=()):
    memory = store.backend.insert_memory(
        Memory(content=text, user_id="ada", embedding_model=store.embedder.model_id,
               categories=list(tags)),
        embedding=store.embedder.embed([text])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def _prices(store, *, broad=False, tagged=True):
    """30 car prices: 20 the question's words find ("priced"), 10 dealer quotes
    they do not, all filed under "car prices"; 60 other facts about the same
    cars under "insurance" and "test drives". ``broad`` files every memory but
    the dealer quotes under "cars" (80 memories) and only half of the 20
    under "car prices" too (20 memories with the quotes)."""
    def tags(*topics):
        return list(topics) if tagged else []

    wide = ("cars",) if broad else ()
    priced = [_add(store, PRICED.format(i=i, n=14000 + 900 * i),
                   tags(*(() if broad and i >= 10 else ("car prices",)), *wide))
              for i in range(20)]
    quoted = [_add(store, QUOTED.format(i=i, n=14000 + 900 * i), tags("car prices"))
              for i in range(20, 30)]
    for i in range(30):
        _add(store, f"Insurance for the Carmodel{i} would be {300 + 20 * i} a year.",
             tags("insurance", *wide))
        _add(store, f"Ada test drove the Carmodel{i} on a rainy day.", tags("test drives", *wide))
    return priced, quoted


def _set_judge():
    return _Batches(specific=0.9, several=0.9,
                    scores={"is priced at": 0.12, "Dealer quote": 0.12})


def test_a_set_whose_members_share_a_topic_takes_two_calls_and_every_member(store):
    """The first 20 judged are the prices the question's words find; they
    share the topic "car prices", so the second call judges the ten dealer
    quotes filed under it, which no word of the question matches. Every price
    is a member and returned, well past the limit."""
    priced, quoted = _prices(store)
    store.decider = judge = _set_judge()
    results = store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2
    first, second = judge.batches
    assert set(first) == {m.content for m in priced}
    assert set(second) == {m.content for m in quoted}  # the topic's, not the ranking's
    members = [r for r in results if r.signals.get("member")]
    assert {r.memory.id for r in members} == {m.id for m in priced + quoted}
    assert {m.id for m in quoted} <= {r.memory.id for r in results}  # outside the first 20
    assert results[0].signals["rounds"] == 2 and results[0].signals["pool"] == 10
    assert all(r.signals["judged"] for r in members)


def test_the_second_call_respects_the_budget_and_counts_small_topics_first(store):
    """Of the first 20, all carry "cars" (80 memories) and 10 "car prices" (20
    memories). A dealer quote, under the second, scores 10/20; an insurance
    fact, under the first, 20/80: at a budget of four the four judged are
    dealer quotes. A small topic half of them share says more about what is
    asked than a broad one all of them do."""
    _, quoted = _prices(store, broad=True)
    store.config.retrieval.set_pool = 4
    store.decider = judge = _set_judge()
    results = store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2
    assert len(judge.batches[1]) == 4
    assert set(judge.batches[1]) <= {m.content for m in quoted}
    assert results[0].signals["pool"] == 4
    store.config.retrieval.set_pool = 80
    store.decider = judge = _set_judge()
    store.search(QUESTION, user_id="ada", limit=5)
    assert len(judge.batches[1]) == 70  # every memory under a shared topic, within 80


def test_the_second_call_orders_each_candidate_once(store, monkeypatch):
    """Ten dealer quotes tie for four places: the property ranking that breaks
    the tie also orders the batch, so no memory is scored twice."""
    _prices(store, broad=True)
    store.config.retrieval.set_pool = 4
    store.decider = judge = _set_judge()
    scored: list[str] = []
    linked_scores = store._linked_scores

    def counted(asked, memory_ids, act, entities):
        scored.extend(memory_ids)
        return linked_scores(asked, memory_ids, act, entities)

    monkeypatch.setattr(store, "_linked_scores", counted)
    store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2 and len(judge.batches[1]) == 4
    assert len(scored) == len(set(scored)) == 10


def test_a_set_whose_first_share_no_topic_reads_the_ranking_past_them(store):
    """Nothing is tagged: the second call judges the ranking past the first 20
    instead, and the set question still makes two calls."""
    _prices(store, tagged=False)
    ranking = [r.memory.content for r in
               store.search(QUESTION, user_id="ada", limit=40, relational=False)]
    assert len(ranking) == 40  # the text ranking a limit-5 search reads
    store.decider = judge = _set_judge()
    results = store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2
    assert set(judge.batches[0]) == set(ranking[:20])
    assert set(judge.batches[1]) == set(ranking[20:])
    assert results[0].signals["rounds"] == 2 and results[0].signals["pool"] == 20


def test_a_set_question_about_a_named_thing_pools_by_topic_too(store):
    """The linked search runs (the question names a hub): "it" reads for the
    dealer in what the provider reads, and the second call still comes from
    the topic the first share."""
    harlow = _entity(store, "Harlow Motors")
    sold = [_add(store, f"Harlow Motors sells the Carmodel{i} for {14000 + 900 * i} euros.",
                 ["car prices"], [harlow]) for i in range(30)]
    for i in range(30):
        _add(store, f"Insurance for the Carmodel{i} would be {300 + 20 * i} a year.",
             ["insurance"])
    store.decider = judge = _Batches(specific=0.9, several=0.9, scores={"sells": 0.12})
    results = store.search("Which car at Harlow Motors is the cheapest?", user_id="ada",
                           limit=5)
    assert judge.calls == 2
    assert all(text.startswith("it sells") for text in judge.batches[0] + judge.batches[1])
    members = {r.memory.id for r in results if r.signals.get("member")}
    assert members == {m.id for m in sold}
    assert results[0].signals["about"] == 1.0 and results[0].signals["rounds"] == 2


def test_a_one_answer_question_makes_one_call_even_when_nothing_answers(store):
    _prices(store)
    store.decider = judge = _Batches(specific=0.9, several=0.1, scores={})
    results = store.search("How much is the Carmodel99 priced at?", user_id="ada", limit=5)
    assert judge.calls == 1  # nothing scores 0.5, and no second call
    assert results[0].signals["rounds"] == 1 and results[0].signals["pool"] == 0
    assert not any(r.signals.get("member") for r in results)


def test_a_question_about_everything_makes_one_call(store):
    _prices(store)
    store.decider = judge = _Batches(specific=0.1, several=0.9, scores={"priced": 0.5})
    results = store.search("Tell me about the cars priced so far", user_id="ada", limit=5)
    assert judge.calls == 1
    assert not any(r.signals.get("member") for r in results)
    assert len(results) == 5


# --------------------------------------------------------------- re-ranking
class _Reranker(_RoundJudge):
    """A provider that re-ranks by default and counts the questions it gets."""

    may_rerank = reranks_by_default = True

    def __init__(self):
        super().__init__(specific=0.9, several=0.1, scores={"Linux": 0.9})
        self.states: list[str] = []

    def decide(self, state, questions):
        self.states.append(state)
        return super().decide(state, questions)


def test_rerank_does_not_run_after_the_linked_search(store):
    """With the property vectors alone (``relational_relevance = "vector"``)
    the linked order stands: the provider is not asked to re-rank it. A search
    the linked search did not order (no hub named, or ``relational=False``)
    is re-ranked as before."""
    store.config.retrieval.relational_relevance = "vector"
    harlow = _entity(store, "Harlow")
    for text in ("Harlow runs on Linux", "Harlow stores its data in SQLite",
                 "Harlow costs nothing"):
        _add(store, text, entities=[harlow])
    for i in range(5):
        _add(store, f"Unrelated note {i} about Linux")
    linked = [r.memory.id for r in store.search("What does Harlow run on?", user_id="ada",
                                                limit=5)]
    store.decider = reranker = _Reranker()
    results = store.search("What does Harlow run on?", user_id="ada", limit=5)
    assert all("about" in r.signals for r in results)
    assert reranker.states == []
    assert [r.memory.id for r in results] == linked  # the linked order stands
    store.search("Which notes mention Linux?", user_id="ada", limit=5)
    assert reranker.states == ["QUESTION: Which notes mention Linux?"]
    store.search("What does Harlow run on?", user_id="ada", limit=5, relational=False)
    assert len(reranker.states) == 2


def test_the_benchmark_searches_deep_where_the_first_search_asked_no_decider(monkeypatch):
    """``relative_retrieval_benchmark.score`` reads a question's full ranking
    from a second search at limit 100, except where the first search asked
    the decision provider (it would be asked again). Under --rerank a linked
    search that ran is not re-ranked, so it is searched again; a question
    naming no hub is re-ranked, as is a search without links, and the linked
    search judging relevance asks on every search."""
    sys.path.insert(0, str(ROOT))
    from evals import relative_retrieval_benchmark as bench
    from memry.models import Entity

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=_KindEmbedder())
    harlow = store.backend.insert_entity(
        Entity(name="Harlow", normalized="harlow", user_id=bench.USER))
    ids = []
    for text in ("Harlow runs on Linux", "Harlow stores its data in SQLite",
                 "Harlow costs nothing", *(f"Unrelated note {i} about Linux" for i in range(5))):
        memory = store.backend.insert_memory(
            Memory(content=text, user_id=bench.USER, embedding_model=store.embedder.model_id),
            embedding=store.embedder.embed([text])[0])
        if text.startswith("Harlow"):
            store.backend.add_mention(EntityMention(entity_id=harlow.id, memory_id=memory.id,
                                                    surface="Harlow"))
        ids.append(memory.id)
    store.decider = reranker = _Reranker()
    limits: list[int] = []
    search = store.search

    def counted(query, **kwargs):
        limits.append(kwargs["limit"])
        return search(query, **kwargs)

    monkeypatch.setattr(store, "search", counted)
    hub = {"direct": [("What does Harlow run on?", [0], [])]}
    linked = next(mode for mode in bench.MODES if mode[0] == "linked k1")
    bench.score(store, ids, hub, linked)
    assert (reranker.states, limits) == ([], [10, 100])
    limits.clear()
    bench.score(store, ids, {"direct": [("Which notes mention Linux?", [3], [])]}, linked)
    assert (len(reranker.states), limits) == (1, [10])
    limits.clear()
    bench.score(store, ids, hub, next(mode for mode in bench.MODES if mode[0] == "hybrid"))
    assert (len(reranker.states), limits) == (2, [10])
    limits.clear()
    bench.score(store, ids, hub, next(mode for mode in bench.MODES if mode[0] == "linked jev"))
    assert (len(reranker.states), limits) == (3, [10])
    store.close()


# ----------------------------------------------------------- the one search
def test_the_linked_search_is_the_default():
    cfg = Config().retrieval
    assert (cfg.relational_fusion, cfg.relational_mode, cfg.relational_depth) == (
        "linked", "directed", 1)
    assert cfg.set_pool == 80
    assert not hasattr(cfg, "relational_protect_top")


@pytest.mark.parametrize("field,value", [
    ("relational_mode", "typed"), ("relational_mode", "undirected"),
    ("relational_fusion", "rescue"), ("relational_fusion", "weighted"),
    ("relational_fusion", "gated"), ("relational_fusion", "inherit"),
])
def test_a_removed_mode_is_refused_by_the_config(field, value, tmp_path):
    with pytest.raises(ValidationError, match="was removed"):
        RetrievalConfig(**{field: value})
    cfg = RetrievalConfig()
    with pytest.raises(ValidationError, match="was removed"):
        setattr(cfg, field, value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"retrieval": {field: value}}))
    with pytest.raises(ValidationError, match=f"retrieval.{field} '{value}' was removed"):
        Config.load(path)


def test_a_removed_mode_is_refused_by_the_benchmark(monkeypatch, capsys):
    sys.path.insert(0, str(ROOT))
    from evals import relative_retrieval_benchmark as bench

    labels = [mode[0] for mode in bench.MODES]
    assert {"hybrid", "linked k1", "linked jev"} <= set(labels)
    assert not set(labels) & set(bench.REMOVED_MODES)
    assert [m[0] for m in bench.select_modes(["linked k1"])] == ["linked k1"]
    with pytest.raises(ValueError, match="'typed d2 \\(today\\)' was removed"):
        bench.select_modes(["linked k1", "typed d2 (today)"])
    with pytest.raises(ValueError, match="unknown mode 'nope'"):
        bench.select_modes(["nope"])
    monkeypatch.setattr(sys, "argv", ["bench", "--modes", "directed d1 weighted"])
    with pytest.raises(SystemExit) as exited:
        bench.main()
    assert exited.value.code == 2
    assert "'directed d1 weighted' was removed" in capsys.readouterr().err
