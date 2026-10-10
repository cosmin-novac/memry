"""Turn search (``retrieval.turn_search``, on by default at bar 0.8): when the judge
gives no fact found a relevance at the bar, the turns said in the scope
searched are searched as well, by their words and their vectors, each with
the turn said before it and the two said after it; the judge reads these
excerpts in one call, and those it judges at the keep threshold come after
the facts and their evidence, within their own token budget, in the order
they were said. A turn of a deleted or forgotten memory never comes back,
nor one whose every memory is out of use. A turn no memory rests on may come
back, unless extraction kept nothing of its save, a delete touched its save,
or it looks like a secret. Only the scope searched is read."""

from __future__ import annotations

import json
import re

import pytest

from memry.config import Config
from memry.intelligence.context import estimate_tokens, turn_line
from memry.models import MemoryEvent, SearchResult
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import Embedder
from memry.providers.llm import LLM
from memry.store import MemoryStore, fill_excerpts, judged_best

_EXCERPT = "Someone who reads only this part of a conversation can answer the question."


class ScriptedLLM(LLM):
    """Extraction answers from a queue of fact lists; a reconcile decision is
    always ADD and the coverage audit finds nothing missing."""

    name = "scripted"
    available = True

    def __init__(self) -> None:
        self.facts: list[list[dict]] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            return json.dumps({"facts": self.facts.pop(0)})
        if "decide one action" in system:
            return json.dumps({"action": "ADD", "target": None, "content": None,
                               "reason": "new"})
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        return "{}"


class WordEmbedder(Embedder):
    """A scripted embedder: one dimension per word of a fixed vocabulary, so
    a text is near the texts that share its words, and nothing else."""

    name, _model = "scripted", "words"
    VOCABULARY = ("tokyo", "wedding", "sister", "shibuya", "crossing", "times", "square",
                  "crowds", "temple", "kitchen", "green", "paint", "cat", "trip", "otters")
    dimensions = len(VOCABULARY) + 1

    def embed(self, texts):
        out = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            out.append([1.0 if w in words else 0.0 for w in self.VOCABULARY] + [0.1])
        return out


class ScriptedJudge(NoneDecider):
    """A scripted decision provider that re-ranks as Jev does: a memory is
    judged ``facts`` (one value for every fact), an excerpt by the first of
    ``excerpts`` (substring, value) whose text it contains, else 0.05. Every
    call's questions are kept in ``calls``."""

    available = True
    may_rerank = reranks_by_default = True

    def __init__(self, facts: float = 0.2,
                 excerpts: tuple[tuple[str, float], ...] = (("Times Square", 0.9),),
                 several: float = 0.0, complete: float = 1.0) -> None:
        self.facts, self.excerpts = facts, excerpts
        self.several, self.complete = several, complete
        self.calls: list[list[str]] = []
        self.keys: list[dict[str, str]] = []

    def decide(self, state, questions):
        self.calls.append([q.instructions for q in questions.values()])
        self.keys.append({key: q.instructions for key, q in questions.items()})

        def value(key: str, text: str) -> float:
            if key == "property":
                return 1.0
            if key == "several":
                return self.several
            if key == "complete":
                return self.complete
            if text.startswith(_EXCERPT):
                return next((v for part, v in self.excerpts if part in text), 0.05)
            return self.facts

        return Answers({key: Answer(value(key, q.instructions), {}, 0.9, True)
                        for key, q in questions.items()})

    def excerpt_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0].startswith(_EXCERPT)]

    def fact_calls(self) -> list[dict[str, str]]:
        return [k for k in self.keys if k and not next(iter(k.values())).startswith(_EXCERPT)]


_TALK = [
    ("Ada", "We went to Tokyo in March for my sister's wedding."),
    ("Bea", "How was the trip? Did you see Shibuya?"),
    ("Ada", "Shibuya Crossing is like Tokyo's Times Square, the crowds never stop."),
    ("Bea", "Ha, I can imagine those crowds."),
    ("Ada", "The wedding itself was in a small temple."),
    ("Bea", "I am repainting my kitchen a pale green this week."),
    ("Ada", "Pale green sounds calm."),
    ("Bea", "My cat knocked over the paint tin."),
]
_ASKED = "What did Ada say Shibuya Crossing is like?"


def _fact(content: str, *sources: int) -> dict:
    return {"content": content, "type": "episodic", "importance": 0.6, "categories": [],
            "entities": [], "relations": [], "when": None, "sources": list(sources)}


def _store(judge: ScriptedJudge | None = None, *, on: bool = True) -> MemoryStore:
    llm = ScriptedLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=WordEmbedder(),
                        decider=judge or ScriptedJudge())
    store.llm_script = llm
    store.config.retrieval.relational_relevance = "jev"
    store.config.retrieval.turn_search = on
    return store


def _save(store, *, user_id="ada", run_id="r1", created_at="2026-05-02T10:00:00+00:00",
          facts=None):
    store.llm_script.facts.append(facts if facts is not None else [
        _fact("Ada went to Tokyo in March for her sister's wedding", 1, 5),
        _fact("Bea is repainting her kitchen pale green", 6, 7),
    ])
    return store.add([{"role": who, "content": text} for who, text in _TALK],
                     user_id=user_id, run_id=run_id, created_at=created_at)


def _found(store, query=_ASKED, *, user_id="ada", run_id=None, note=None):
    results = store.search(query, user_id=user_id, run_id=run_id, limit=5, evidence=False)
    shown = store.evidence(query, results, user_id=user_id, run_id=run_id)
    turns = store.turn_search(query, results, user_id=user_id, run_id=run_id, shown=shown,
                              note=note)
    return results, shown, turns


def test_it_is_on_by_default_at_bar_0_8_and_off_it_judges_nothing_more():
    assert Config().retrieval.turn_search is True
    assert Config().retrieval.turn_search_bar == 0.8
    judge = ScriptedJudge()
    store = _store(judge, on=False)
    try:
        _save(store)
        results, _, turns = _found(store)
        assert turns == [] and judge.excerpt_calls() == []
        assert judged_best(results) == pytest.approx(0.2)  # judged, only not searched
        context = store.reconstruct_context(_ASKED, user_id="ada", token_budget=800)
        assert "More of what was said" not in context.text
    finally:
        store.close()


def test_when_no_fact_reaches_the_bar_the_turns_are_searched_and_judged_in_one_call():
    judge = ScriptedJudge(facts=0.2)
    store = _store(judge)
    try:
        saved = _save(store)
        note: dict = {}
        _, shown, turns = _found(store, note=note)
        assert note["due"] and note["best"] == pytest.approx(0.2)
        assert len(judge.excerpt_calls()) == 1  # every excerpt in one call
        ids = [t.episode_id for t in turns]
        # the turn no fact kept is shown, with the turn before it and those after
        assert saved.episode_ids[2] in ids
        assert {saved.episode_ids[1], saved.episode_ids[3]} <= set(ids)
        # in the order said, none of the facts' evidence again, resting on no result
        assert ids == [e for e in saved.episode_ids if e in set(ids)]
        assert not set(ids) & {t.episode_id for t in shown}
        assert all(t.memory_ids == [] and t.score == pytest.approx(0.9) for t in turns)
        assert [t.speaker for t in turns if t.episode_id == saved.episode_ids[2]] == ["Ada"]
    finally:
        store.close()


def test_a_fact_at_the_bar_or_a_search_not_judged_searches_no_turns():
    judge = ScriptedJudge(facts=0.9)
    store = _store(judge)
    try:
        _save(store)
        _, _, turns = _found(store)
        assert turns == [] and judge.excerpt_calls() == []
        store.config.retrieval.turn_search_bar = 0.95
        assert _found(store)[2]
        # no decision provider: nothing is judged, so nothing is due
        store.decider = NoneDecider()
        results, _, turns = _found(store)
        assert judged_best(results) is None and turns == []
    finally:
        store.close()


def test_the_window_is_one_turn_before_and_two_after_within_its_run_and_day():
    store = _store()
    try:
        saved = _save(store)
        other = _save(store, run_id="r2")  # the same day, another run
        later = _save(store, run_id="r1", created_at="2026-05-09T10:00:00+00:00")
        ids = saved.episode_ids

        def window(i, ep=ids):
            return [e.id for e in store.backend.episode_neighbours(ep[i], 1, 2)]

        assert window(2) == ids[1:5]
        assert window(0) == ids[0:3]
        assert window(7) == ids[6:8]  # the run's next save is another day
        assert window(0, other.episode_ids) == other.episode_ids[0:3]
        assert window(0, later.episode_ids) == later.episode_ids[0:3]
    finally:
        store.close()


def test_excerpts_are_taken_best_judged_first_within_the_budget_and_from_the_keep_bar():
    excerpts = [{"score": 0.6, "tokens": 30}, {"score": 0.9, "tokens": 50},
                {"score": 0.4, "tokens": 10}, {"score": 0.8, "tokens": 60},
                {"score": 0.7, "tokens": 0}]
    assert fill_excerpts(excerpts, 100, 0.5) == [1, 0]  # 0.8 does not fit after 0.9
    assert fill_excerpts(excerpts, 200, 0.5) == [1, 3, 0]
    assert fill_excerpts(excerpts, 200, 0.0) == [1, 3, 0, 2]
    assert fill_excerpts(excerpts, 40, 0.5) == [0]
    judge = ScriptedJudge(facts=0.2, excerpts=(("Times Square", 0.4),))
    store = _store(judge)
    try:
        _save(store)
        assert _found(store)[2] == []  # judged under the keep bar
        store.config.retrieval.turn_search_keep = 0.3
        _, _, turns = _found(store)
        assert turns
        one = estimate_tokens(turn_line(turns[0])) + 1
        results = store.search(_ASKED, user_id="ada", limit=5, evidence=False)
        for budget in (0, one, 3 * one, 40):
            got = store.turn_search(_ASKED, results, user_id="ada", token_budget=budget)
            assert sum(estimate_tokens(turn_line(t)) + 1 for t in got) <= budget
    finally:
        store.close()


def test_a_turn_of_a_deleted_or_forgotten_memory_never_comes_back():
    store = _store()
    try:
        saved = _save(store, facts=[
            _fact("Ada went to Tokyo in March for her sister's wedding", 1, 5),
            _fact("Ada compared Shibuya Crossing to a busy square", 3),
        ])
        memories = {m.content: m for m in store.get_all(user_id="ada")}
        shibuya = memories["Ada compared Shibuya Crossing to a busy square"]
        turn = saved.episode_ids[2]

        def in_context() -> tuple[bool, bool]:
            _, shown, turns = _found(store)
            return (turn in {t.episode_id for t in shown},
                    turn in {t.episode_id for t in turns})

        assert in_context() == (True, False)  # its fact's evidence, not repeated
        store.delete(shibuya.id)  # forgotten: out of use with nothing in its place
        assert in_context() == (False, False)
        store.unforget(shibuya.id)
        assert in_context() == (True, False)
        store.delete(shibuya.id, hard=True)  # deleted for good: withheld
        assert in_context() == (False, False)
        assert store.backend.episodes_by_id([turn])[turn].withheld_at
        # a delete touched the save: its turns no memory rests on are not read
        # either; the turns its other memory rests on still may be
        note: dict = {}
        _found(store, note=note)
        read = {e for x in note.get("excerpts", []) for e in x["episode_ids"]}
        assert not read & {saved.episode_ids[1], saved.episode_ids[3], turn}
        allowed = {e.id for e in store.backend.turn_search_episodes(saved.episode_ids)}
        assert allowed == {saved.episode_ids[0], saved.episode_ids[4]}
    finally:
        store.close()


def test_a_turn_whose_every_memory_is_out_of_use_is_not_shown_and_history_is():
    store = _store()
    try:
        saved = _save(store, facts=[
            _fact("Ada went to Tokyo in March for her sister's wedding", 1, 5),
            _fact("Ada compared Shibuya Crossing to a busy square", 3),
        ])
        backend = store.backend
        shibuya = {m.content: m for m in store.get_all(user_id="ada")}[
            "Ada compared Shibuya Crossing to a busy square"]
        replacement = store.add("Ada never went to Shibuya.", user_id="ada",
                                infer=False).actions[0].memory_id
        backend.invalidate_memory(shibuya.id, superseded_by=replacement)
        assert saved.episode_ids[2] not in [
            e.id for e in backend.turn_search_episodes(saved.episode_ids)]
        assert saved.episode_ids[2] not in [t.episode_id for t in _found(store)[2]]
        # kept as history (an update): what was said while it held may be shown
        backend.add_event(MemoryEvent(memory_id=shibuya.id, event="SUPERSEDE", kind="update",
                                      new_content="Ada never went to Shibuya."))
        assert backend.history_ids([shibuya.id]) == {shibuya.id}
        assert saved.episode_ids[2] in [
            e.id for e in backend.turn_search_episodes(saved.episode_ids)]
        # a turn no memory rests on (extraction kept nothing of it) may be shown
        assert saved.episode_ids[3] in [
            e.id for e in backend.turn_search_episodes(saved.episode_ids)]
    finally:
        store.close()


def test_only_the_scope_searched_is_read():
    store = _store()
    try:
        mine = _save(store, user_id="ada", run_id="r1")
        theirs = _save(store, user_id="kai", run_id="r1")
        later = _save(store, user_id="ada", run_id="r2", created_at="2026-06-01T10:00:00+00:00")
        got = {t.episode_id for t in _found(store)[2]}
        assert got and not got & set(theirs.episode_ids)
        in_r1 = {t.episode_id for t in _found(store, run_id="r1")[2]}
        assert in_r1 and in_r1 <= set(mine.episode_ids)
        in_r2 = {t.episode_id for t in _found(store, run_id="r2")[2]}
        assert in_r2 and in_r2 <= set(later.episode_ids)
        assert not {e for e, _ in store.backend.episode_keyword_search(
            "Shibuya", _scope("kai", "r1"), 50)} & set(mine.episode_ids)
    finally:
        store.close()


def _scope(user_id, run_id):
    from memry.models import Scope

    return Scope(user_id=user_id, run_id=run_id)


def test_a_turn_a_result_says_whole_or_already_shown_is_not_repeated():
    store = _store()
    try:
        saved = _save(store)
        results, shown, turns = _found(store)
        assert {t.episode_id for t in shown} & set(saved.episode_ids)
        verbatim = store.add(_TALK[2][1], user_id="ada", infer=False).actions[0].memory_id
        results = [SearchResult(memory=store.get(verbatim), score=1.0,
                                signals={"relevance": 0.1})]
        got = store.turn_search(_ASKED, results, user_id="ada")
        assert saved.episode_ids[2] not in [t.episode_id for t in got]
    finally:
        store.close()


def test_the_context_shows_the_turns_found_after_the_facts_and_their_evidence():
    store = _store()
    try:
        saved = _save(store)
        context = store.reconstruct_context(_ASKED, user_id="ada", token_budget=800)
        facts, rest = context.text.split("\n\nWhat was said, in the order it was said:\n")
        evidence, quoted = rest.split(
            "\n\nMore of what was said that matches the question, in the order it was said:\n")
        assert "Ada went to Tokyo in March for her sister's wedding" in facts
        assert f"- 2 May 2026: Ada: {_TALK[2][1]}" in quoted
        assert _TALK[2][1] not in evidence
        assert saved.episode_ids[2] in context.episode_ids
        assert context.token_estimate <= 800
        # a fact at the bar: the same context has no such part
        store.decider = ScriptedJudge(facts=0.9)
        assert "More of what was said" not in store.reconstruct_context(
            _ASKED, user_id="ada", token_budget=800).text
    finally:
        store.close()


@pytest.mark.parametrize("off", ["0", "false", "off", "no"])
def test_the_environment_turns_it_off(monkeypatch, tmp_path, off):
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    assert Config.load().retrieval.turn_search is True
    monkeypatch.setenv("MEMRY_TURN_SEARCH", off)
    assert Config.load().retrieval.turn_search is False


@pytest.mark.parametrize("name, key, default", [
    ("MEMRY_TURN_SEARCH_SEVERAL", "turn_search_several", 0.5),
    ("MEMRY_TURN_SEARCH_COMPLETE", "turn_search_complete", 0.7)])
def test_the_environment_sets_or_turns_off_each_trigger(monkeypatch, tmp_path, name, key,
                                                        default):
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    assert getattr(Config.load().retrieval, key) == default
    monkeypatch.setenv(name, "0.6")
    assert getattr(Config.load().retrieval, key) == 0.6
    for off in ("0", "off", "false", "no", "none"):
        monkeypatch.setenv(name, off)
        assert getattr(Config.load().retrieval, key) is None
    monkeypatch.setenv(name, "not a number")  # ignored, the default stands
    assert getattr(Config.load().retrieval, key) == default


def test_a_save_extraction_kept_nothing_of_is_never_shown():
    store = _store()
    try:
        kept = _save(store)
        nothing = _save(store, run_id="r2", facts=[])  # extraction refused or found nothing
        allowed = {e.id for e in store.backend.turn_search_episodes(
            [*kept.episode_ids, *nothing.episode_ids])}
        assert allowed == set(kept.episode_ids)
        assert not {t.episode_id for t in _found(store)[2]} & set(nothing.episode_ids)
    finally:
        store.close()


def test_a_turn_no_memory_rests_on_that_looks_like_a_secret_is_never_shown():
    from memry.intelligence.extraction import looks_secret

    assert looks_secret("My wifi password is otter-tokyo-77.")
    assert looks_secret("the key is sk-live-4f9a8b7c6d5e4f3a2b1c")
    assert not looks_secret("Shibuya Crossing is like Tokyo's Times Square.")
    store = _store()
    try:
        talk = [*_TALK[:3], ("Ada", "My wifi password at the Tokyo hotel was otter-77."),
                *_TALK[3:]]
        store.llm_script.facts.append([
            _fact("Ada went to Tokyo in March for her sister's wedding", 1, 6)])
        saved = store.add([{"role": who, "content": text} for who, text in talk],
                          user_id="ada", run_id="r1", created_at="2026-05-02T10:00:00+00:00")
        allowed = {e.id for e in store.backend.turn_search_episodes(saved.episode_ids)}
        assert saved.episode_ids[3] not in allowed and saved.episode_ids[2] in allowed
        note: dict = {}
        _found(store, "What was the wifi password at the Tokyo hotel?", note=note)
        assert not any(saved.episode_ids[3] in x["episode_ids"] for x in note["excerpts"])
    finally:
        store.close()


# -- the two triggers beside the relevance bar (PhD notes, completeness-trigger-plan) --


def test_the_several_answer_runs_turn_search_at_its_bar_with_no_more_calls():
    """With ``turn_search_several``, turn search also runs when the judge's
    "several" answer, asked in every judged search, is at least that bar,
    though a fact is judged above the relevance bar."""
    judge = ScriptedJudge(facts=0.9, several=0.5)
    store = _store(judge)
    try:
        saved = _save(store)
        assert Config().retrieval.turn_search_several == 0.5  # on by default
        store.config.retrieval.turn_search_several = None
        store.config.retrieval.turn_search_complete = None  # A alone here
        note: dict = {}
        assert _found(store, note=note)[2] == [] and note["why"] == []  # not read when unset
        store.config.retrieval.turn_search_several = 0.5
        before = len(judge.fact_calls())
        note = {}
        _, _, turns = _found(store, note=note)
        assert note["why"] == ["several"] and note["due"]
        assert saved.episode_ids[2] in [t.episode_id for t in turns]
        # the facts' call is the search's own: one, as without the trigger
        assert len(judge.fact_calls()) - before == 1
        assert all("complete" not in keys for keys in judge.fact_calls())
        judge.several = 0.49
        note = {}
        assert _found(store, note=note)[2] == [] and note["why"] == []
    finally:
        store.close()


def test_the_completeness_question_is_asked_in_the_judge_call_and_fires_under_its_bar():
    """With ``turn_search_complete``, the judge's first call also asks
    whether the facts, read together, contain everything asked, with the
    question as asked and each fact as a model reads it; turn search runs
    when the answer is under the bar."""
    judge = ScriptedJudge(facts=0.9, complete=0.6)
    store = _store(judge)
    try:
        saved = _save(store)
        assert Config().retrieval.turn_search_complete == 0.7  # on by default
        store.config.retrieval.turn_search_complete = None
        _found(store)
        assert all("complete" not in keys for keys in judge.fact_calls())  # not asked unset
        store.config.retrieval.turn_search_complete = 0.7
        before = len(judge.fact_calls())
        note: dict = {}
        results, _, turns = _found(store, note=note)
        calls = judge.fact_calls()[before:]
        assert len(calls) == 1  # in the same call as the relevance of each fact
        asked = calls[0]["complete"]
        assert asked.startswith("These facts, read together, contain everything the question "
                                "asks for, about the person or thing it asks about.")
        assert _ASKED in asked
        assert "- Ada went to Tokyo in March for her sister's wedding (said 2 May 2026)" in asked
        assert {f"m{i}" for i in range(len(results))} <= set(calls[0])
        assert {"property", "several"} <= set(calls[0])
        assert all(r.signals["complete"] == pytest.approx(0.6) for r in results)
        assert note["why"] == ["incomplete"]
        assert saved.episode_ids[2] in [t.episode_id for t in turns]
        store.config.retrieval.turn_search_complete = 0.5  # 0.6 is not under it
        note = {}
        assert _found(store, note=note)[2] == [] and note["why"] == []
        # the relevance bar and both triggers, each named
        store.config.retrieval.turn_search_complete = 0.7
        store.config.retrieval.turn_search_several = 0.5
        store.config.retrieval.turn_search_bar = 0.95
        judge.several = 0.8
        note = {}
        _found(store, note=note)
        assert note["why"] == ["relevance", "several", "incomplete"]
    finally:
        store.close()


def test_neither_trigger_fires_for_a_search_not_judged():
    judge = ScriptedJudge(facts=0.9, several=0.9, complete=0.1)
    store = _store(judge)
    store.config.retrieval.turn_search_several = 0.5
    store.config.retrieval.turn_search_complete = 0.7
    try:
        _save(store)
        assert _found(store)[2]  # judged: both fire
        store.config.retrieval.relational_relevance = "vector"  # no search is judged
        calls = len(judge.calls)
        results, _, turns = _found(store)
        assert turns == [] and len(judge.calls) == calls
        assert judged_best(results) is None
        store.decider = NoneDecider()
        store.config.retrieval.relational_relevance = "jev"
        results, _, turns = _found(store)
        assert turns == [] and judged_best(results) is None
        # signals without a relevance were not judged by this search's call
        unjudged = [SearchResult(memory=r.memory, score=1.0,
                                 signals={"several": 0.9, "complete": 0.1}) for r in results]
        assert store.turn_search_reasons(unjudged) == []
        assert not store.turn_search_due(unjudged)
    finally:
        store.close()


def test_with_turn_search_off_the_triggers_change_nothing():
    """The settings are read only with ``retrieval.turn_search`` on: off, the
    judge is asked what it was asked before, and nothing more is read."""

    def run(several, complete):
        judge = ScriptedJudge(facts=0.9, several=0.9, complete=0.1)
        store = _store(judge, on=False)
        store.config.retrieval.turn_search_several = several
        store.config.retrieval.turn_search_complete = complete
        try:
            _save(store)
            note: dict = {}
            results, shown, turns = _found(store, note=note)
            context = store.reconstruct_context(_ASKED, user_id="ada", token_budget=800).text
            # recency moves with the clock between the two runs
            return (judge.keys, [(r.memory.content, {k: v for k, v in r.signals.items()
                                                     if k != "recency"}) for r in results],
                    [t.episode_id for t in shown], turns, note, context)
        finally:
            store.close()

    unset = run(None, None)
    keys, results, _, turns, note, context = run(0.5, 0.7)
    assert keys == unset[0] and results == unset[1] and turns == unset[3] == []
    assert all("complete" not in k for k in keys)
    assert not note["due"] and note["why"] == []
    assert context == unset[5] and "More of what was said" not in context
