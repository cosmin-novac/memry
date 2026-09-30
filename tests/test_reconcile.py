"""Reconcile acts on its answer the same way whatever run the memory belongs to.

Each test runs in two layouts with the same expected outcome: every save in
one run ("same"), and each save in a run of its own ("diff"), as a benchmark
that gives every session its own run saves. A stub decision provider gives
the answer; the text model, where one is needed, writes a MORE's merged text.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM, decision, facts_response

from memry.config import Config
from memry.intelligence.reconcile import (
    ACTION_QUESTION,
    ACTIONS,
    CONFLICT_KEY,
    reconcile_state,
    saves_of,
)
from memry.models import Memory
from memry.providers.decisions import Answer, Answers, JevDecider, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

LAYOUTS = ("same", "diff")
USER = "tom"
FIRST = "2026-03-02T09:00:00+00:00"
LATER = "2026-04-13T09:00:00+00:00"
BARS = {"SAME": 0.85, "MORE": 0.8, "CHANGED": 0.5, "WRONG": 0.5}


class Judge(NoneDecider):
    """Answers every reconcile question with ``answer`` at ``confidence``,
    about memory [0], and keeps the states it was shown."""

    name = "stub"
    available = True
    reconcile_bars = BARS

    def __init__(self, answer: str = "NEW", confidence: float = 0.95) -> None:
        self.answer, self.confidence = answer, confidence
        self.states: list[str] = []

    def decide(self, state, questions):
        answers = {}
        if "action" in questions:
            self.states.append(state)
            answers["action"] = Answer(self.answer, {self.answer: self.confidence},
                                       self.confidence, True)
        if "target" in questions:
            answers["target"] = Answer("0", {"0": 1.0}, 1.0, True)
        return Answers(answers)


def _store(judge: Judge, llm=None) -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=llm or NoneLLM(),
                       embedder=HashEmbedder(64), decider=judge)


def _run(layout: str, save: int) -> str:
    return "r1" if layout == "same" else f"r{save + 1}"


def _save(store: MemoryStore, text: str, layout: str, save: int, at: str, **kwargs):
    result = store.add(text, user_id=USER, run_id=_run(layout, save), infer=False,
                       created_at=at, **kwargs)
    [action] = result.actions
    return action


def _live(store: MemoryStore) -> list[str]:
    return sorted(m.id for m in store.get_all(user_id=USER))


def _supersede_kinds(store: MemoryStore, memory_id: str) -> list[str | None]:
    return [e.kind for e in store.history(memory_id) if e.event == "SUPERSEDE"]


# ------------------------------------------------------------------ CHANGED
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_changed_value_supersedes_the_old_one_which_stays_as_history(layout):
    """The old value ends at the new one's date and stays retrievable: search
    returns it after the current value, even for a question worded like the
    old one, and the answer context shows until when it held."""
    store = _store(Judge("CHANGED", 0.9))
    old = _save(store, "Tom's gym membership costs $40 a month", layout, 0, FIRST)
    new = _save(store, "Tom's gym membership costs $55 a month", layout, 1, LATER)

    assert new.event == "SUPERSEDE" and new.memory_id != old.memory_id
    retired = store.get(old.memory_id)
    assert (retired.invalid_at, retired.superseded_by) == (LATER, new.memory_id)
    assert _supersede_kinds(store, old.memory_id) == ["update"]
    assert store.get(new.memory_id).run_id == _run(layout, 1)
    assert _live(store) == [new.memory_id]
    for question in ("How much does Tom's gym membership cost?", "$40 a month"):
        found = store.search(question, user_id=USER, limit=5)
        assert [r.memory.id for r in found] == [new.memory_id, old.memory_id]
    # the one rendering for a model (``context.memory_line``): the old value
    # said the day it began to hold, and held until the day of the new one
    context = store.reconstruct_context("gym membership price", user_id=USER).text
    assert ("- Tom's gym membership costs $55 a month (said 13 April 2026)\n"
            "- Tom's gym membership costs $40 a month (said 2 March 2026) "
            "[until 13 April 2026]") in context
    assert "$55 a month (said 13 April 2026) [until" not in context
    store.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_memory_kept_as_history_is_forgotten_by_a_normal_delete(layout):
    """Kept as history, the old value is still searchable, so a person who
    deletes it expects it gone as any memory goes: search, the answer context
    and the archive of replacements no longer show it, it is listed as
    forgotten, and from there it can be purged."""
    store = _store(Judge("CHANGED", 0.9))
    old = _save(store, "Tom's gym membership costs $40 a month", layout, 0, FIRST)
    new = _save(store, "Tom's gym membership costs $55 a month", layout, 1, LATER)
    question = "How much does Tom's gym membership cost?"
    assert [r.memory.id for r in store.search(question, user_id=USER, limit=5)] == \
        [new.memory_id, old.memory_id]

    assert store.delete(old.memory_id)
    assert [r.memory.id for r in store.search(question, user_id=USER, limit=5)] == \
        [new.memory_id]
    assert "$40" not in store.reconstruct_context("gym membership price", user_id=USER).text
    assert old.memory_id not in {row["memory"].id for row in store.replaced(user_id=USER)}
    [gone] = store.forgotten(user_id=USER)
    assert (gone["memory"].id, gone["trigger"]) == (old.memory_id, "You deleted it.")
    assert [e.event for e in store.history(old.memory_id)][-1] == "DELETE"
    assert _live(store) == [new.memory_id]
    assert store.purge(old.memory_id) and store.get(old.memory_id) is None
    store.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_doubtful_change_keeps_both_and_asks(layout):
    store = _store(Judge("CHANGED", 0.4))
    old = _save(store, "Tom's gym membership costs $40 a month", layout, 0, FIRST)
    new = _save(store, "Tom's gym membership costs $55 a month", layout, 1, LATER)

    assert (new.event, new.conflicts_with) == ("ADD", old.memory_id)
    assert _live(store) == sorted([old.memory_id, new.memory_id])
    mark = store.get(new.memory_id).metadata[CONFLICT_KEY]
    assert (mark["with"], mark["kind"]) == (old.memory_id, "update")
    assert "0.40 sure" in mark["held"]
    [item] = [i for i in store.upkeep_queue(user_id=USER) if i["kind"] == "conflict"]
    assert item["id"] == new.memory_id
    # confirmed, the old one becomes history as an unheld change would have
    assert store.decide_upkeep("conflict", new.memory_id, "accept", user_id=USER)
    assert _supersede_kinds(store, old.memory_id) == ["update"]
    found = store.search("gym membership", user_id=USER, limit=5)
    assert [r.memory.id for r in found] == [new.memory_id, old.memory_id]
    store.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_protection_counts_saves_one_of_many_turns_once_and_two_saves_twice(layout):
    """"Stated in two or more separate saves" counts the saves behind a
    memory's evidence, not its episodes. A memory made from a save of three
    messages was stated once, so a change may replace it without asking
    (counted by episode it had been "stated in 3 separate saves", and no
    change could retire a fact of a conversation saved whole). Said again by
    a second save (SAME), it was stated twice, and a change waits for a
    person."""
    judge = Judge("SAME", 0.95)
    store = _store(judge)
    first = store.add([{"role": "user", "content": part} for part in
                       ("Tom's gym", "membership costs", "$40 a month")],
                      user_id=USER, run_id=_run(layout, 0), infer=False,
                      created_at=FIRST).actions[0]
    memory = store.get(first.memory_id)
    assert len(memory.source_episode_ids) == 3
    assert saves_of(store.backend, memory) == 1

    again = _save(store, "Tom pays $40 a month for the gym", layout, 1,
                  "2026-03-20T09:00:00+00:00")
    assert (again.event, again.memory_id) == ("NONE", first.memory_id)
    assert saves_of(store.backend, store.get(first.memory_id)) == 2

    judge.answer, judge.confidence = "CHANGED", 0.9
    new = _save(store, "Tom's gym membership costs $55 a month", layout, 2, LATER)
    assert (new.event, new.conflicts_with) == ("ADD", first.memory_id)
    assert "stated in 2 separate saves" in store.get(new.memory_id).metadata[CONFLICT_KEY]["held"]
    store.close()


def test_one_save_of_many_turns_is_one_save_a_change_may_replace():
    store = _store(Judge("CHANGED", 0.9))
    first = store.add([{"role": "user", "content": part} for part in
                       ("Tom's gym", "membership costs", "$40 a month")],
                      user_id=USER, infer=False).actions[0]  # the clock's time
    assert saves_of(store.backend, store.get(first.memory_id)) == 1
    new = _save(store, "Tom's gym membership costs $55 a month", "same", 1, LATER)
    assert new.event == "SUPERSEDE" and new.conflicts_with is None
    store.close()


# -------------------------------------------------------------------- WRONG
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_correction_supersedes_as_a_contradiction_out_of_search(layout):
    store = _store(Judge("WRONG", 0.9))
    old = _save(store, "Tom's dentist appointment is on Monday at 3pm", layout, 0, FIRST)
    new = _save(store, "Tom's dentist appointment is on Tuesday at 3pm", layout, 1, LATER)

    assert new.event == "DELETE"
    retired = store.get(old.memory_id)
    assert (retired.invalid_at, retired.superseded_by) == (LATER, new.memory_id)
    assert _supersede_kinds(store, old.memory_id) == ["contradiction"]
    found = store.search("When is Tom's dentist appointment on Monday?", user_id=USER, limit=5)
    assert [r.memory.id for r in found] == [new.memory_id]
    assert "Monday" not in store.reconstruct_context("dentist", user_id=USER).text
    store.close()


# --------------------------------------------------------------------- SAME
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("confidence", [BARS["SAME"], BARS["SAME"] - 0.01])
def test_a_rewording_at_the_bar_is_recorded_on_the_memory_and_below_it_added(
        layout, confidence):
    """At the bar the save is recorded on the memory it restates: one live
    memory, whose sources now hold the save's episode, found by a search of
    the save's run. Below the bar the fact is added as new."""
    store = _store(Judge("SAME", confidence))
    first = _save(store, "Tom's brother Carlos lives in Madrid", layout, 0, FIRST)
    again = store.add("Carlos, Tom's brother, is based in Madrid", user_id=USER,
                      run_id=_run(layout, 1), infer=False, created_at=LATER)
    [action] = again.actions
    in_run = [r.memory.id for r in store.search(
        "Where does Carlos live?", user_id=USER, run_id=_run(layout, 1), limit=5)]
    if confidence >= BARS["SAME"]:
        assert (action.event, action.memory_id) == ("NONE", first.memory_id)
        assert _live(store) == [first.memory_id]
        memory = store.get(first.memory_id)
        assert again.episode_ids[0] in memory.source_episode_ids
        assert memory.updated_at == FIRST        # the memory did not change
        [said_again] = [e for e in store.history(first.memory_id) if e.event == "NONE"]
        assert said_again.created_at == LATER     # when it was last said
        assert in_run == [first.memory_id]
    else:
        assert action.event == "ADD" and action.memory_id != first.memory_id
        assert "under its bar" in action.reason
        assert _live(store) == sorted([first.memory_id, action.memory_id])
        assert action.memory_id in in_run
    store.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_exact_duplicate_is_recorded_as_same_without_asking(layout):
    judge = Judge("NEW", 0.99)
    store = _store(judge)
    first = _save(store, "Tom speaks fluent Portuguese", layout, 0, FIRST)
    again = _save(store, "Tom speaks fluent Portuguese.", layout, 1, LATER)
    assert (again.event, again.memory_id, again.reason) == (
        "NONE", first.memory_id, "exact duplicate")
    assert judge.states == []
    assert _live(store) == [first.memory_id]
    found = store.search("Portuguese", user_id=USER, run_id=_run(layout, 1), limit=5)
    assert [r.memory.id for r in found] == [first.memory_id]
    store.close()


# --------------------------------------------------------------------- MORE
@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_added_detail_leaves_one_merged_memory_dated_at_the_save(layout):
    """The merged text is a new memory of the save's time and run; the old
    one keeps its text and date and ends at the save, superseded as an
    update. Before, within one run the old memory was rewritten in place and
    kept its old date, and in another run the detail was added beside it."""
    llm = FakeLLM()
    store = _store(Judge("MORE", 0.95), llm)
    first = _save(store, "Tom is learning Spanish", layout, 0, FIRST)
    merged = "Tom is learning Spanish with a tutor twice a week"
    llm.queue(decision("MORE", target=0, content=merged), facts_response())
    more = _save(store, "Tom has a Spanish tutor twice a week", layout, 1, LATER)

    assert more.event == "UPDATE" and more.memory_id != first.memory_id
    new = store.get(more.memory_id)
    assert (new.content, new.run_id) == (merged, _run(layout, 1))
    assert new.created_at == new.updated_at == new.valid_from == LATER
    old = store.get(first.memory_id)
    assert (old.content, old.created_at) == ("Tom is learning Spanish", FIRST)
    assert (old.invalid_at, old.superseded_by) == (LATER, more.memory_id)
    assert _supersede_kinds(store, first.memory_id) == ["update"]
    assert _live(store) == [more.memory_id]
    for save in (0, 1):  # its sources keep it in both runs' search
        found = store.search("Spanish", user_id=USER, run_id=_run(layout, save), limit=5)
        assert found[0].memory.id == more.memory_id
    assert llm.responses == []
    store.close()


# ---------------------------------------------------------------------- NEW
@pytest.mark.parametrize("layout", LAYOUTS)
def test_two_identical_yoga_class_texts_on_different_dates_stay_two(layout):
    """Two events that read alike are not an exact duplicate when said on
    different days: the judge sees both dates, and another occurrence is NEW.
    Said again the same day, the text is the same event."""
    judge = Judge("NEW", 0.99)
    store = _store(judge)
    text = "Tom went to a yoga class this morning"
    first = _save(store, text, layout, 0, FIRST, memory_type="episodic")
    week = "2026-03-09T09:00:00+00:00"
    second = _save(store, text, layout, 1, week, memory_type="episodic")
    assert second.event == "ADD" and second.memory_id != first.memory_id
    assert _live(store) == sorted([first.memory_id, second.memory_id])
    [state] = judge.states
    assert f"[0] (said 2 March 2026) {text}" in state
    assert f"NEW fact (said 9 March 2026):\n{text}" in state
    same_day = _save(store, text, layout, 1, "2026-03-09T18:00:00+00:00",
                     memory_type="episodic")
    assert (same_day.event, same_day.memory_id) == ("NONE", second.memory_id)
    assert len(judge.states) == 1
    store.close()


# ------------------------------------------------------- every answer, acted on
OUTCOMES = {
    # answer: (the save's event, the old memory in use, the old one's supersede
    # kind, the old one in a search of the user)
    "NEW": ("ADD", True, [], True),
    "SAME": ("NONE", True, [], True),
    "MORE": ("UPDATE", False, ["update"], True),
    "CHANGED": ("SUPERSEDE", False, ["update"], True),
    "WRONG": ("DELETE", False, ["contradiction"], False),
}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("answer", ACTIONS)
def test_each_answer_maps_to_its_action(layout, answer):
    llm = FakeLLM()
    store = _store(Judge(answer, 0.95), llm)
    old = _save(store, "Tom works at Corlan", layout, 0, FIRST)
    if answer == "MORE":
        llm.queue(decision("MORE", target=0, content="Tom works at Corlan on payments"),
                  facts_response())
    new = _save(store, "Tom works on the payments team", layout, 1, LATER)

    event, in_use, kinds, searchable = OUTCOMES[answer]
    assert new.event == event
    assert (store.get(old.memory_id).invalid_at is None) is in_use
    assert _supersede_kinds(store, old.memory_id) == kinds
    found = [r.memory.id for r in store.search("Corlan", user_id=USER, limit=5)]
    assert (old.memory_id in found) is searchable
    assert new.memory_id in found
    live = _live(store)
    assert len(live) == (2 if answer == "NEW" else 1)
    assert llm.responses == []
    store.close()


# ------------------------------------------------------------ the question
def test_the_question_offers_the_five_answers_and_shows_dates():
    assert tuple(ACTION_QUESTION.criteria) == ACTIONS
    assert "another occurrence" in ACTION_QUESTION.criteria["NEW"]
    assert "still true" in ACTION_QUESTION.criteria["MORE"]
    assert "no longer true" in ACTION_QUESTION.criteria["CHANGED"]
    assert "never true" in ACTION_QUESTION.criteria["WRONG"]
    state = reconcile_state([Memory(content="Maria is in Paris for work this week",
                                    created_at=FIRST)],
                            "Maria is in Rome this week", LATER)
    assert state == ("EXISTING memories:\n[0] (said 2 March 2026) Maria is in Paris for work "
                     "this week\n\nNEW fact (said 13 April 2026):\nMaria is in Rome this week")


def test_jev_has_a_measured_bar_for_every_answer_that_acts():
    assert set(JevDecider.reconcile_bars) == {"SAME", "MORE", "CHANGED", "WRONG"}
    assert NoneDecider.reconcile_bars is None  # unmeasured: SupersedeConfig.confidence


# ---------------------------------------------------------- the benchmark
def test_the_update_benchmark_grades_a_store_by_rule():
    """``evals/reconcile_benchmark.py`` grades the final store: an older value
    kept as history is fine, one still live as current is stale, and a
    history memory above the current value is misranked; a restatement that
    left a second live copy is a duplicate."""
    from evals.reconcile_benchmark import CASES, bars, grade

    case = {c["id"]: c for c in CASES}

    def row(id, content, invalid=None, history=False):
        return {"id": id, "content": content, "invalid_at": invalid, "history": history,
                "conflict": None}

    old = row("o", "Tom's gym membership costs $40 a month.", LATER, True)
    new = row("n", "Tom's gym membership costs $55 a month.")
    first = {"saves": [{}, {"before": [row("o", old["content"])]}]}
    right = {**first, "final": [old, new], "search": [new, old]}
    assert grade(case["A3"], right) == ("right", "")
    assert grade(case["A3"], {**right, "search": [old, new]})[0] == "misranked"
    stale = {**first, "final": [dict(old, invalid_at=None, history=False), new],
             "search": [new]}
    assert grade(case["A3"], stale)[0] == "stale"
    copy = row("c", "Tom speaks fluent Portuguese.")
    twice = {"saves": [{}, {"before": [copy]}], "final": [copy, row("d", copy["content"])],
             "search": [copy]}
    assert grade(case["D4"], twice)[0] == "duplicate"
    answers = [{"action": "SAME", "conf": 0.9, "ok": ["SAME"]},
               {"action": "SAME", "conf": 0.6, "ok": ["MORE"]}]
    assert bars(answers)["SAME"]["bar_with_no_wrong"] == pytest.approx(0.61)
