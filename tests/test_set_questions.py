"""Questions that need several memories ("Which car is the cheapest?"): the
decision provider judges the first ``decision.rerank_pool`` of the ranking and
then, in one more call, the memories filed under the topics those share, and
past those the memories nearest the members found, up to
``retrieval.set_pool`` (``MemoryStore._set_pool``). A question with one
answer, or about everything, makes one call. Also: the linked search is the
only one, and a removed mode is refused."""

from __future__ import annotations

import json
import sys
from collections import Counter
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


def _add(store, text, tags=(), entities=(), run_id=None):
    memory = store.backend.insert_memory(
        Memory(content=text, user_id="ada", run_id=run_id,
               embedding_model=store.embedder.model_id, categories=list(tags)),
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
    is a member and returned, well past the limit. (At a budget of ten the
    topic fills the call; a larger one is filled from the memories nearest
    the members, below.)"""
    priced, quoted = _prices(store)
    store.config.retrieval.set_pool = 10
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


def test_a_large_tie_is_cut_before_it_is_scored(store, monkeypatch):
    """500 dealer quotes under the one topic the first 20 share tie for 80
    places. The tie is cut by a key that costs nothing (the share, then the
    newest first) to the places left plus a margin of 20 before the property
    ranking scores it, so about 100 are scored, not 500."""
    for i in range(20):
        _add(store, PRICED.format(i=i, n=14000 + 900 * i), ["car prices"])
    for i in range(20, 520):
        _add(store, QUOTED.format(i=i, n=14000 + 900 * i), ["car prices"])
    store.config.retrieval.set_pool = 80
    store.decider = judge = _set_judge()
    scored: list[str] = []
    linked_scores = store._linked_scores

    def counted(asked, memory_ids, act, entities):
        scored.extend(memory_ids)
        return linked_scores(asked, memory_ids, act, entities)

    monkeypatch.setattr(store, "_linked_scores", counted)
    store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2 and len(judge.batches[1]) == 80
    assert len(scored) == len(set(scored)) <= 100


class _Counting:
    """A backend's stand-in that counts the calls made to it."""

    def __init__(self, backend):
        self.backend, self.calls = backend, Counter()

    def __getattr__(self, name):
        attr = getattr(self.backend, name)
        if not callable(attr):
            return attr

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return attr(*args, **kwargs)
        return counted


def test_the_second_call_reads_its_hundred_candidates_in_a_handful_of_queries(
        store, monkeypatch):
    """A hundred candidates under two shared topics: their entities, the
    topics' sizes and the candidates' vectors are each read at once, not once
    per candidate or per topic."""
    for i in range(120):
        text = (PRICED if i < 20 else QUOTED).format(i=i, n=14000 + 900 * i)
        _add(store, text, ["car prices", "cars"])
    store.config.retrieval.set_pool = 100
    counting = _Counting(store.backend)
    set_pool = store._set_pool

    def counted(*args, **kwargs):
        store.backend = counting
        try:
            return set_pool(*args, **kwargs)
        finally:
            store.backend = counting.backend

    monkeypatch.setattr(store, "_set_pool", counted)
    store.decider = judge = _set_judge()
    store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2 and len(judge.batches[1]) == 100
    assert sum(counting.calls.values()) <= 8, dict(counting.calls)


def test_a_topics_size_is_counted_where_the_search_looks(store):
    """Searched in run r2, "car prices" files 4 memories there (and 400 in
    r1), "cars" 60. All 20 of the first carry "cars" and two of them "car
    prices" too: 2 of 4 counts more than 20 of 60, so at a budget of two the
    two dealer quotes under "car prices" in r2 are judged. Counted over every
    run, "car prices" had 2 of 404 and the broad topic won."""
    for i in range(20):
        _add(store, PRICED.format(i=i, n=14000 + 900 * i),
             ["cars", "car prices"] if i >= 18 else ["cars"], run_id="r2")
    quoted = [_add(store, QUOTED.format(i=i, n=14000 + 900 * i), ["car prices"], run_id="r2")
              for i in range(20, 22)]
    for i in range(40):
        _add(store, f"Ada test drove the Carmodel{i} on a rainy day.", ["cars"], run_id="r2")
    for i in range(400):
        _add(store, QUOTED.format(i=100 + i, n=9000 + i), ["car prices"], run_id="r1")
    store.config.retrieval.set_pool = 2
    store.decider = judge = _set_judge()
    store.search(QUESTION, user_id="ada", run_id="r2", limit=5)
    assert judge.calls == 2
    assert set(judge.batches[1]) == {m.content for m in quoted}


class _PriceEmbedder(_KindEmbedder):
    """Every price near every other however it is worded ("is priced at",
    "costs"), as a real embedder places them; the question near none."""

    KINDS = ["euros", "insurance", "drove"]


def test_a_set_whose_first_share_no_topic_reads_the_memories_nearest_its_members(store):
    """Nothing is tagged (memories saved with infer=False, or imported
    verbatim): the second call judges the unjudged memories nearest the
    members the first call found, by vector, not the ranking past the first
    20. Here those are the ten dealer quotes, which no word of the question
    matches and which the ranking does not reach next."""
    store.embedder = _PriceEmbedder()
    priced = [_add(store, PRICED.format(i=i, n=14000 + 900 * i)) for i in range(20)]
    for i in range(30):
        _add(store, f"Insurance for the Carmodel{i} would be {300 + 20 * i} a year.")
        _add(store, f"Ada test drove the Carmodel{i} on a rainy day.")
    quoted = [_add(store, QUOTED.format(i=i, n=14000 + 900 * i)) for i in range(20, 30)]
    store.config.retrieval.set_pool = 10
    ranking = [r.memory.content for r in
               store.search(QUESTION, user_id="ada", limit=40, relational=False)]
    assert len(ranking) == 40  # the text ranking a limit-5 search reads
    assert set(ranking[:20]) == {m.content for m in priced}
    quotes = {m.content for m in quoted}
    assert not set(ranking[20:30]) & quotes  # what the second call judged before
    store.decider = judge = _set_judge()
    results = store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2
    assert set(judge.batches[0]) == set(ranking[:20])
    assert set(judge.batches[1]) == quotes
    members = {r.memory.id for r in results if r.signals.get("member")}
    assert members == {m.id for m in priced + quoted}
    assert results[0].signals["rounds"] == 2 and results[0].signals["pool"] == 10


def test_a_topic_smaller_than_the_budget_is_filled_from_the_nearest_to_the_members(store):
    """Four of the dealer quotes are filed under "car prices", six were saved
    untagged: the second call judges the four the topic gives and fills its
    budget of ten with the memories nearest the members, the other six
    quotes, not the insurance costs or test drives."""
    store.embedder = _PriceEmbedder()
    priced = [_add(store, PRICED.format(i=i, n=14000 + 900 * i), ["car prices"])
              for i in range(20)]
    quoted = [_add(store, QUOTED.format(i=i, n=14000 + 900 * i),
                   ["car prices"] if i < 24 else []) for i in range(20, 30)]
    for i in range(30):
        _add(store, f"Insurance for the Carmodel{i} would be {300 + 20 * i} a year.",
             ["insurance"])
        _add(store, f"Ada test drove the Carmodel{i} on a rainy day.", ["test drives"])
    store.config.retrieval.set_pool = 10
    store.decider = judge = _set_judge()
    results = store.search(QUESTION, user_id="ada", limit=5)
    assert judge.calls == 2
    assert set(judge.batches[0]) == {m.content for m in priced}
    assert set(judge.batches[1]) == {m.content for m in quoted}
    assert results[0].signals["pool"] == 10


def test_a_set_question_about_a_named_thing_pools_by_topic_too(store):
    """The linked search runs (the question names a hub): "it" reads for the
    dealer in what the provider reads, and the second call still comes from
    the topic the first share (whose ten fill a budget of ten)."""
    harlow = _entity(store, "Harlow Motors")
    sold = [_add(store, f"Harlow Motors sells the Carmodel{i} for {14000 + 900 * i} euros.",
                 ["car prices"], [harlow]) for i in range(30)]
    for i in range(30):
        _add(store, f"Insurance for the Carmodel{i} would be {300 + 20 * i} a year.",
             ["insurance"])
    store.config.retrieval.set_pool = 10
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


def _parking(store, entities=()):
    """30 memories the question's words find, each with a word of its own."""
    return [_add(store, f"Ada parked the car at garage marker{i:02d} today.",
                 entities=entities) for i in range(30)]


def test_a_question_naming_no_hub_blends_the_judgement_with_the_text_ranking(store):
    """R-117. With the judging call on by default, a question naming no hub
    was ordered by the judged score alone, which measured worse than the
    0.35 blend of ``_rerank`` (recall@3 0.933 against 0.844 on
    distractors_v1). A one-answer question naming no hub is now ordered by
    that blend of the judged score and the text ranking's position: the
    memory judged 0.6 at the bottom of the first 20, against 0.5 for the
    rest, stays below the top hits and is not moved to first. One call is
    made; its meta questions still decide the set path."""
    _parking(store)
    question = "Where did Ada park the car?"
    ranking = [r.memory.id for r in store.search(question, user_id="ada", limit=40,
                                                 relational=False)]
    assert len(ranking) >= 20
    bottom = store.get(ranking[19])
    marker = bottom.content.split("garage ", 1)[1].split(" ", 1)[0]
    store.decider = judge = _Batches(specific=0.9, several=0.1,
                                     scores={marker: 0.6, "marker": 0.5})
    results = store.search(question, user_id="ada", limit=20)
    assert judge.calls == 1
    order = [r.memory.id for r in results]
    assert order[0] == ranking[0] and order[:5] == ranking[:5]
    assert order.index(bottom.id) > 5
    assert results[order.index(bottom.id)].signals["judged"] > results[0].signals["judged"]

    # a set question naming no hub still gets its second call
    store.decider = judge = _Batches(specific=0.9, several=0.9,
                                     scores={marker: 0.6, "marker": 0.5})
    store.search("Which garages did Ada park at?", user_id="ada", limit=5)
    assert judge.calls == 2


def test_a_question_naming_a_hub_is_ordered_by_the_judgement_as_before(store):
    """The blend is for questions naming no hub: after the linked search the
    judged score times aboutness orders, so the memory judged 0.6 comes
    first wherever the ranking had it."""
    harlow = _entity(store, "Harlow")
    memories = [_add(store, f"Harlow parked the car at garage marker{i:02d} today.",
                     entities=[harlow]) for i in range(30)]
    question = "Where did Harlow park the car?"
    store.decider = _Batches(specific=0.9, several=0.1, scores={"marker": 0.5})
    ranking = [r.memory.id for r in store.search(question, user_id="ada", limit=20)]
    assert len(ranking) == 20
    bottom = next(m for m in memories if m.id == ranking[19])
    marker = bottom.content.split("garage ", 1)[1].split(" ", 1)[0]
    store.decider = judge = _Batches(specific=0.9, several=0.1,
                                     scores={marker: 0.6, "marker": 0.5})
    results = store.search(question, user_id="ada", limit=5)
    assert judge.calls == 1
    assert results[0].memory.id == bottom.id and "about" in results[0].signals


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
