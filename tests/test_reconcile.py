"""Reconcile acts on its answer the same way whatever run the memory belongs to.

Each test runs in two layouts with the same expected outcome: every save in
one run ("same"), and each save in a run of its own ("diff"), as a benchmark
that gives every session its own run saves. A stub decision provider gives
the answer; the text model, where one is needed, writes a MORE's merged text.
"""

from __future__ import annotations

import re

import pytest
from conftest import FakeLLM, decision, facts_response

from memry.config import Config
from memry.intelligence.reconcile import (
    ACTION_QUESTION,
    ACTIONS,
    CONFLICT_KEY,
    MERGE_REQUEST,
    RECONCILE_SYSTEM,
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


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_merge_writer_reads_each_fact_with_its_dates_and_keeps_each_time_with_it(layout):
    """The merged text is stored as said on the new fact's date, so a
    relative time in either text would be read against that date and next to
    the other fact's dates. The writer therefore reads both as Memry renders
    a memory for a model (``context.memory_line``: when it happened, where
    known, and when it was said) and is told to write each time as the date
    it names, read against its own text's date, with the event it belongs
    to. Before, it read the two texts with no date at all, and "a road trip
    last year" said in April 2023 became "the previous year's road trip"
    beside a trip of 16 December 2022 (LoCoMo conv-41)."""
    from memry.intelligence.reconcile import MERGE_REQUEST, RECONCILE_SYSTEM

    llm = FakeLLM()
    store = _store(Judge("MORE", 0.95), llm)
    _save(store, "Tom returned from a family road trip on 2026-03-01 and said it was fun",
          layout, 0, FIRST, memory_type="episodic",
          memory_metadata={"when": {"start": "2026-03-01"}})
    llm.queue(decision("MORE", target=0, content="merged"), facts_response())
    _save(store, "Tom said a road trip he took last year explored the coast", layout, 1, LATER)

    system, prompt = llm.calls[0]
    assert system == RECONCILE_SYSTEM and MERGE_REQUEST in prompt
    assert ("[0] [happened 2026-03-01] Tom returned from a family road trip on 2026-03-01 and "
            "said it was fun (said 2 March 2026)") in prompt
    assert ("NEW fact:\nTom said a road trip he took last year explored the coast "
            "(said 13 April 2026)") in prompt
    rule = " ".join(system.split())
    assert "The merged text is stored as said on the NEW fact's date." in rule
    assert "reading a relative time" in rule and "against the date its own text was said" in rule
    assert "keep each date with the event it belongs to" in rule
    store.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_more_the_merge_writer_reads_as_two_things_is_added_as_new(layout):
    """Jev's MORE on a second event of one kind (a trip "last year" beside a
    trip of March) stood at 0.80 to 0.97 in the LoCoMo stores, as high as its
    right merges, so no bar tells them apart. The writer reads both texts with
    their dates; when it answers that the new fact is another event or thing
    of the same kind, nothing is merged and the fact is stored beside the
    memory, as a NEW is. Before, a writer that wrote no text left the new fact
    superseding the old one."""
    from memry.intelligence.reconcile import MERGE_REQUEST

    llm = FakeLLM()
    store = _store(Judge("MORE", 0.95), llm)
    old = _save(store, "Tom returned from a family road trip on 2026-03-01", layout, 0, FIRST)
    llm.queue(decision("NEW"))
    new = _save(store, "Tom said a road trip he took last year explored the coast", layout, 1,
                LATER)

    assert MERGE_REQUEST in llm.calls[0][1]
    assert "another event or thing" in llm.calls[0][1]
    assert new.event == "ADD" and new.memory_id != old.memory_id
    assert new.conflicts_with is None
    assert _live(store) == sorted([old.memory_id, new.memory_id])
    assert store.get(old.memory_id).invalid_at is None
    assert _supersede_kinds(store, old.memory_id) == []
    assert store.get(new.memory_id).content == (
        "Tom said a road trip he took last year explored the coast")
    assert "another event or thing" in new.reason and old.memory_id in new.reason
    assert llm.responses == []
    store.close()


# ------------------------------------------------------- one fact per memory
#: Saves about one thesis, each stating another claim, then (the last) a
#: detail of the fourth claim: how strongly it is held. The shape of the
#: memories a store saved under 0.2.39 grew into, one save at a time: "The
#: central claim of X's thesis is ... The thesis further argues ... The thesis
#: explicitly rejects ... X explicitly accepts that ... X holds this
#: prediction very strongly."
CLAIMS = [
    "The central claim of Ana's thesis is that small models can match large ones on narrow tasks",
    "Ana's thesis further argues that benchmark contamination explains most reported gains",
    "Ana's thesis explicitly rejects the idea that scale alone produces reasoning",
    "Ana's thesis predicts that open models will match closed ones by 2028",
    "Ana explicitly accepts that large models write better open-ended prose",
]
DETAIL = "Ana holds her prediction that open models will match closed ones by 2028 very strongly"
SAVED = ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23", "2026-03-30", "2026-04-06"]


def _words(text: str) -> set[str]:
    return {w.strip(".,").lower() for w in text.split() if len(w) > 3}


class ChainJudge(Judge):
    """Answers MORE at 0.95 to every save, about the memory that shares the
    most words with the new fact: a judge that reads every claim about the
    thesis as detail added to the thesis memory."""

    def __init__(self) -> None:
        super().__init__("MORE", 0.95)

    def decide(self, state, questions):
        answers = super().decide(state, questions).answers
        if "target" in questions:
            listed = [line.split(") ", 1)[-1] for line in state.split("\n")
                      if line.startswith("[")]
            new = state.rsplit(":\n", 1)[-1]
            best = max(range(len(listed)), key=lambda i: len(_words(listed[i]) & _words(new)))
            answers["target"] = Answer(str(best), {str(best): 1.0}, 1.0, True)
        return Answers(answers)


class InstructedWriter(FakeLLM):
    """A text model that does what its instructions allow, for the reconcile
    prompt (as the judge where no decision provider answers, and as the
    merge writer after a MORE): a new fact the test marks as another claim
    (``claims``) is kept apart (NEW) only when the instructions name another
    claim as a reason not to merge; anything else is merged as the prompt
    asks, one text that says everything both say. Its merged text is the two
    texts joined, which is what every merge of a claim into an essay was.
    The judge's prompt names no other claim, so as the judge it answers MORE
    to every save; as the writer it keeps another claim apart."""

    def __init__(self, claims: list[str]) -> None:
        super().__init__()
        self.claims = set(claims)

    def complete(self, system, user, *, json_schema=None):
        self.calls.append((system, user))
        if "Conversation:" in user:  # the no-provider path extracts nothing new
            return facts_response()
        def text(line: str) -> str:  # a listed memory or the new fact, without its dates
            return re.sub(r"^\(said [^)]*\) | \(said [^)]*\)$", "", line)

        listed = [text(line.split("] ", 1)[1]) for line in user.split("\n")
                  if line.startswith("[")]
        new = text(user.split("NEW fact", 1)[1].split(":\n", 1)[1].split("\n", 1)[0])
        instructions = " ".join((system + " " + user).split())
        if new in self.claims and "another claim" in instructions:
            return decision("NEW")
        target = max(range(len(listed)), key=lambda i: len(_words(listed[i]) & _words(new)))
        return decision("MORE", target=target, content=f"{listed[target]}. {new}")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("judge", ["decision provider", "text model"])
def test_saves_of_other_claims_about_one_subject_stay_one_memory_per_claim(layout, judge):
    """Each save states another claim about one thesis, and every one reads
    as MORE: to the decision provider (the stub answers MORE for every save),
    or to the text model deciding alone. A merge adds a detail to the same
    fact; another claim about the same subject is a memory of its own, so the
    five claims stay five memories, and the last save, a detail of the
    fourth claim, merges into that one. The merge writer writes every merged
    text, after either judge's MORE, and keeps another claim apart. Before,
    nothing in the writer's instructions named another claim, and a text
    model judging alone wrote its merged text itself: each save folded its
    claim into the memory before it, and one memory grew with every save
    until it held all five."""
    llm = InstructedWriter(CLAIMS[1:])
    decider = ChainJudge() if judge == "decision provider" else NoneDecider()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(64),
                        decider=decider)
    for i, text in enumerate([*CLAIMS, DETAIL]):
        _save(store, text, layout, i, f"{SAVED[i]}T09:00:00+00:00")

    live = [m.content for m in store.get_all(user_id=USER)]
    holding = {claim: [m for m in live if claim in m] for claim in CLAIMS}
    assert len(live) == 5, f"{len(live)} memories in use: {live}"
    assert all(len(found) == 1 for found in holding.values()), holding
    [prediction] = holding[CLAIMS[3]]
    assert prediction == f"{CLAIMS[3]}. {DETAIL}"
    # the other claims were kept apart by the merge writer, which says why (the
    # fourth one's memory then took the detail and is history);
    # the text model judging alone answered MORE, and the writer, asked after
    # it as after a decision provider's MORE, kept them apart
    kept = [e.reason for m in store.get_all(user_id=USER) for e in store.history(m.id)
            if e.event == "ADD" and "another claim" in (e.reason or "")]
    assert len(kept) == 3
    writer = [user for _, user in llm.calls if MERGE_REQUEST in user]
    judged = [user for _, user in llm.calls
              if "EXISTING memories" in user and MERGE_REQUEST not in user]
    assert len(writer) == 5 and len(judged) == (0 if judge == "decision provider" else 5)
    store.close()


def test_the_merge_writer_is_told_that_a_merge_keeps_one_fact():
    """The merge writer, which writes every merged text, may answer NEW for a
    new fact that states another claim about the memory's subject (another
    argument, position, finding, opinion or decision), not only for another
    event or thing of one kind. Before, its only exit was another event or
    thing. The rule is in the writer's request alone: the same rule in the
    judge's system prompt made the text model answer NEW to restatements and
    changed values, and MORE worded as "the same statement, with more detail"
    lowered Jev's MORE on right merges, so both are as they were."""
    request = " ".join(MERGE_REQUEST.split())
    assert "another event or thing than memory [0], of the same kind" in request
    assert ("or states another claim about the same subject than memory [0] does (another "
            "argument, position, finding, opinion or decision, not a detail of the one memory "
            "[0] states)") in request
    assert "another claim" not in RECONCILE_SYSTEM
    assert ACTION_QUESTION.criteria["MORE"] == (
        "It adds detail to one existing memory, and that memory is still true as it stands.")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_text_models_more_is_written_by_the_merge_writer(layout):
    """With no decision provider, the text model answers MORE with a merged
    text of its own; the merge writer is asked all the same, and its text is
    the one stored. Where the writer writes nothing, the judge's text
    stands."""
    llm = FakeLLM()
    store = _store(NoneDecider(), llm)
    first = _save(store, "Tom is learning Spanish", layout, 0, FIRST)
    llm.queue(decision("MORE", target=0, content="the judge's text"),
              decision("MORE", target=0, content="Tom is learning Spanish with a tutor"),
              facts_response())
    more = _save(store, "Tom has a Spanish tutor", layout, 1, LATER)
    assert more.event == "UPDATE"
    assert store.get(more.memory_id).content == "Tom is learning Spanish with a tutor"
    assert MERGE_REQUEST in llm.calls[1][1] and MERGE_REQUEST not in llm.calls[0][1]

    llm.queue(decision("MORE", target=0, content="Tom is learning Spanish with a tutor on "
                       "Tuesdays"), "not json", facts_response())
    again = _save(store, "Tom sees his Spanish tutor on Tuesdays", layout, 2, LATER)
    assert store.get(again.memory_id).content == (
        "Tom is learning Spanish with a tutor on Tuesdays")
    assert first.memory_id not in _live(store) and llm.responses == []
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


def test_changed_carries_the_measured_worked_examples_and_more_none():
    """The wording measured with Jev (``ACTION_QUESTION``'s comment): CHANGED
    shows a plan that then happened and a status that moved on, so a plan
    followed by its outcome is not read as a detail added to the plan. MORE
    shows no example; one there lowered MORE on real added details."""
    changed = ACTION_QUESTION.criteria["CHANGED"]
    assert changed.endswith(
        'For example: "plans to visit Oslo in May", then "visited Oslo last week" (the plan '
        'happened); "is building a shed", then "finished the shed" (the status moved on).')
    assert ACTION_QUESTION.criteria["MORE"] == (
        "It adds detail to one existing memory, and that memory is still true as it stands.")
    for option in ("NEW", "SAME", "MORE", "WRONG"):
        assert "For example" not in ACTION_QUESTION.criteria[option]


def test_jev_has_a_measured_bar_for_every_answer_that_acts():
    assert set(JevDecider.reconcile_bars) == {"SAME", "MORE", "CHANGED", "WRONG"}
    assert NoneDecider.reconcile_bars is None  # unmeasured: SupersedeConfig.confidence


def test_jevs_reconcile_bars_are_the_measured_values():
    """Jev's bars are the values measured on its recorded answers to the
    reconcile question, in both wordings: ``evals/reconcile_benchmark.py
    bars`` recomputes them from the recorded answers and labels kept in the
    PhD repository's ``data/reconcile`` folder. A bar moved without a new
    measurement fails here, and the stub these tests act through
    (``BARS``) holds the same values, so what they test is what Jev does."""
    measured = {"SAME": 0.85, "MORE": 0.8, "CHANGED": 0.5, "WRONG": 0.5}
    assert JevDecider.reconcile_bars == measured
    assert BARS == measured


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


def test_the_update_benchmark_keeps_two_of_one_kind_apart_and_grades_a_moved_time():
    """Two events or things of one kind must stay two (NEW is the only
    answer that fits), and a merged memory written at a later save must not
    state a time relative to another day: the conv-41 regression (R1) fails
    as "misdated" when the April text holding both trips puts the coast trip
    in "the previous year" beside the December date, and passes when it
    gives the coast trip its year. A memory of the coast trip alone, which
    says "last year" on the day it was said, is not checked."""
    from evals.reconcile_benchmark import CASES, grade

    case = {c["id"]: c for c in CASES}
    kinds = {c["kind"] for c in CASES}
    assert {"two of one kind", "relative time"} <= kinds
    for id in ("L1", "L2", "L3", "L4", "L5"):
        assert (case[id]["expect"], case[id]["ok"], case[id]["n"]) == ("separate", [()], 2)
    assert case["R1"]["ok"] == [()] and case["M1"]["ok"] == [("MORE",)]

    def row(id, content, at, invalid=None):
        return {"id": id, "content": content, "created_at": at, "invalid_at": invalid,
                "history": bool(invalid), "conflict": None}

    december, april = "2026-12-17T09:00:00+00:00", "2027-04-10T09:00:00+00:00"
    old = row("o", "Tom returned from a family road trip on 2026-12-16 and said it was fun.",
              december, invalid=april)
    saves = [{"at": december}, {"at": april, "before": [dict(old, invalid_at=None)]}]

    def result(text):
        merged = row("m", text, april)
        return {"saves": saves, "final": [old, merged], "search": [merged]}

    wrong = result("Tom returned from a family road trip on 2026-12-16 and said it was fun. He "
                   "said the previous year's road trip explored the coast up north.")
    assert grade(case["R1"], wrong)[0] == "misdated"
    right = result("Tom returned from a family road trip on 2026-12-16 and said it was fun. On "
                   "10 April 2027 he said a road trip he took in 2026 explored the coast up "
                   "north.")
    assert grade(case["R1"], right) == ("right", "")
    alone = row("n", "Tom said a road trip he took last year explored the coast up north.", april)
    kept_apart = {"saves": saves, "final": [dict(old, invalid_at=None), alone],
                  "search": [alone]}
    assert grade(case["R1"], kept_apart) == ("right", "")


def test_the_merge_writer_pairs_grade_conv41_by_its_times():
    """``reconcile_benchmark.py merges`` asks the merge writer to join fixed
    pairs, the texts as extraction wrote them. P1 is conv-41: the text of
    3517519's writer, "the previous year's road trip" after the December
    date, is misdated; one that gives the coast trip its year is not, and it
    still joins two trips; declining is right."""
    from evals.reconcile_benchmark import MERGE_PAIRS, grade_merge

    pairs = {p["id"]: p for p in MERGE_PAIRS}
    p1 = pairs["P1"]
    assert (p1["expect"], p1["old"][0], p1["new"][0]) == ("apart", 290, 404)
    assert "previous year" in p1["new"][1]

    def written(answer, text=None):
        return {"id": "P1", "answer": answer, "text": text}

    recorded = ("Tom returned from a family road trip on 2026-12-16 and said it was fun. He said "
                "the previous year's road trip explored the coast up north.")
    assert grade_merge(p1, written("merged", recorded)) == "misdated"
    dated = ("Tom returned from a family road trip on 2026-12-16 and said it was fun. On 10 April "
             "2027 he said a road trip he took in 2026 explored the coast up north.")
    assert grade_merge(p1, written("merged", dated)) == "joined"
    assert grade_merge(p1, written("apart")) == "right"
    assert grade_merge(pairs["P10"], written("apart")) == "apart"
    assert {p["kind"] for p in MERGE_PAIRS} >= {"relative time", "two of one kind",
                                               "added detail"}


def test_the_update_benchmark_keeps_claims_about_one_subject_apart():
    """Saves that each state another claim about one subject must end as one
    memory per claim: a live memory that holds two of a case's ``claims`` is
    "joined". The claim pairs label a new claim NEW (the writer must keep it
    apart) and a detail of the same claim MORE (it must merge), and Jev's
    answers are graded as memry acts on them at its bars."""
    from evals.reconcile_benchmark import (
        CASES, CLAIM_PAIRS, MERGE_PAIRS, acted, claim_question_pairs, grade, grade_merge)

    case = {c["id"]: c for c in CASES}
    s1 = case["S1"]
    assert (s1["expect"], s1["n"], s1["ok"]) == ("separate", 3, [(), ()])

    def row(id, content):
        return {"id": id, "content": content, "invalid_at": None, "history": False,
                "conflict": None, "created_at": FIRST}

    claims = ["Maria's thesis claims small language models can match large ones on narrow "
              "tasks.", "Maria's thesis argues benchmark contamination explains most gains.",
              "Maria rejects the idea that scale alone produces reasoning."]
    apart = [row(str(i), text) for i, text in enumerate(claims)]
    saves = [{"at": FIRST}, {"at": LATER, "before": apart[:1]}]
    assert grade(s1, {"saves": saves, "final": apart, "search": apart[2:]}) == ("right", "")
    essay = row("e", " ".join(claims))
    assert grade(s1, {"saves": saves, "final": [essay], "search": [essay]})[0] == "joined"
    assert {c["kind"] for c in CASES} >= {"claims about one subject", "claim detail"}

    kinds = [p["kind"] for p in CLAIM_PAIRS]
    assert (kinds.count("new claim"), kinds.count("claim detail")) == (16, 15)
    assert all(p in MERGE_PAIRS for p in CLAIM_PAIRS)
    labelled = {p["id"]: p for p in claim_question_pairs()}
    assert (labelled["Q1"]["ok"], labelled["D1"]["ok"]) == ([], ["MORE"])
    q1 = next(p for p in CLAIM_PAIRS if p["id"] == "Q1")
    d1 = next(p for p in CLAIM_PAIRS if p["id"] == "D1")
    assert grade_merge(q1, {"answer": "merged", "text": "both claims"}) == "joined"
    assert grade_merge(q1, {"answer": "apart", "text": None}) == "right"
    assert grade_merge(d1, {"answer": "apart", "text": None}) == "apart"
    assert acted({"action": "MORE", "conf": 0.7}) == "NEW"
    assert acted({"action": "MORE", "conf": 0.9}) == "MORE"
    assert acted({"action": "CHANGED", "conf": 0.4}) == "held"
