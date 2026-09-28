"""evals/external_benchmarks.py on small fixtures in the LoCoMo and
LongMemEval formats; the runs on the real files skip when the data is absent."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import external_benchmarks as xb  # noqa: E402
from memry.config import Config  # noqa: E402
from memry.providers.embeddings import HashEmbedder  # noqa: E402
from memry.providers.llm import LLM, NoneLLM  # noqa: E402
from memry.store import MemoryStore  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
LOCOMO = FIXTURES / "locomo_mini.json"
LONGMEMEVAL = FIXTURES / "longmemeval_mini.json"


def verbatim_store() -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(128))


@pytest.fixture
def no_models(monkeypatch, tmp_path):
    """Config.load() finds no model: no keys, no config file."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY",
                 "MEMRY_LLM_PROVIDER", "MEMRY_LLM_API_KEY", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "no-config.json"))


# --------------------------------------------------------------------------
# loaders


def test_parse_bench_date_reads_both_formats():
    assert xb.parse_bench_date("1:56 pm on 8 May, 2023").isoformat() == "2023-05-08T13:56:00+00:00"
    assert xb.parse_bench_date("12:09 am on 13 September, 2023").hour == 0
    assert xb.parse_bench_date("2023/05/20 (Sat) 02:21").isoformat() == "2023-05-20T02:21:00+00:00"
    assert xb.parse_bench_date("8 May 2023").date().isoformat() == "2023-05-08"
    assert xb.parse_bench_date("2023-05-08").date().isoformat() == "2023-05-08"
    assert xb.parse_bench_date("sometime in spring") is None
    assert xb.parse_bench_date(None) is None


def test_load_locomo_fixture():
    convs = xb.load_locomo(LOCOMO)
    assert [c.conv_id for c in convs] == ["conv-mini-1", "conv-mini-2"]
    first = convs[0]
    assert first.warnings == []
    assert first.label == "conversation between Maya and Theo"
    assert [s.session_id for s in first.sessions] == ["session_1", "session_2"]
    assert first.sessions[1].date.isoformat() == "2023-05-25T19:30:00+00:00"
    assert [t.key for t in first.sessions[0].turns] == ["D1:1", "D1:2", "D1:3", "D1:4"]
    photo = first.sessions[0].turns[2]
    assert photo.role == "Maya"
    assert photo.text == ("Maya: She is shy but she loves the soft blanket by the window. "
                          "[shares a photo: a dog lying on a blue blanket]")
    q = {q.qid: q for c in convs for q in c.questions}
    assert len(q) == 7
    assert {x.category for x in q.values()} == {1, 2, 3, 4, 5}
    assert q["conv-mini-1/q0"].category_name == xb.LOCOMO_CATEGORIES[4]
    assert q["conv-mini-1/q0"].level == "turn"
    # "D1:4; D2:1" in one evidence string is two turns
    assert q["conv-mini-1/q2"].evidence == ["D1:4", "D2:1"]
    # an adversarial question keeps no misleading gold answer
    adversarial = q["conv-mini-1/q3"]
    assert adversarial.abstain and adversarial.answer == xb.ABSTAIN_ANSWER
    assert adversarial.extra == {"adversarial_answer": "saxophone"}
    assert q["conv-mini-2/q0"].answer == "2019"  # a number in the file
    assert all(e in {t.key for t in c.turns} for c in convs for x in c.questions
               for e in x.evidence)


def test_locomo_loader_tolerates_small_faults_and_reports_them(tmp_path):
    sample = {
        "sample_id": "odd",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_2_date_time": "not a date",
            "session_2": [{"speaker": "A", "text": "second session first"}],
            "session_1": [{"speaker": "B", "text": "", "blip_caption": "a cake"},
                          {"speaker": "A", "dia_id": "D1:2", "text": ""}],
        },
        "qa": [{"question": "What was shown?", "answer": "a cake",
                "evidence": ["D1:1", "D9:9", "nonsense", "D01:01"], "category": "4"}],
    }
    path = tmp_path / "odd.json"
    path.write_text(json.dumps({"data": [sample]}))
    (conv,) = xb.load_locomo(path)
    assert [s.session_id for s in conv.sessions] == ["session_1", "session_2"]
    assert [t.key for t in conv.turns] == ["D1:1", "D2:1"]  # dia_ids made up; empty turn dropped
    assert conv.turns[0].text == "B: [shares a photo: a cake]"
    assert conv.questions[0].evidence == ["D1:1"]  # "D01:01" is the same turn
    assert conv.questions[0].category == 4
    joined = " | ".join(conv.warnings)
    assert "D9:9" in joined and "nonsense" in joined
    assert "no readable date" in joined and "D1:2: empty turn" in joined


@pytest.mark.parametrize("broken, message", [
    ({"sample_id": "x", "conversation": {"speaker_a": "A"}, "qa": []}, "no session_N"),
    ({"sample_id": "x", "conversation": {"session_1": [{"speaker": "A", "text": "hi"}]},
      "qa": [{"answer": "hi", "evidence": []}]}, "no question"),
    ({"sample_id": "x", "conversation": {"session_1": [{"speaker": "A", "text": "hi"}]},
      "qa": {"question": "?"}}, "'qa' is not a list"),
    ({"sample_id": "x", "conversation": {"session_1": [
        {"speaker": "A", "dia_id": "D1:1", "text": "hi"},
        {"speaker": "B", "dia_id": "D1:1", "text": "hello"}]}, "qa": []}, "appears twice"),
])
def test_locomo_loader_rejects_what_it_cannot_read(tmp_path, broken, message):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps([broken]))
    with pytest.raises(xb.FormatError, match=message):
        xb.load_locomo(path)


def test_load_longmemeval_fixture():
    convs = xb.load_longmemeval(LONGMEMEVAL)
    assert len(convs) == 3
    types = [c.questions[0].category for c in convs]
    assert types == ["single-session-user", "knowledge-update", "multi-session"]
    for conv in convs:
        assert conv.warnings == []
        assert len(conv.sessions) == 2
        assert len(conv.questions) == 1 and conv.questions[0].level == "session"
    dog = convs[0]
    assert dog.sessions[0].date.isoformat() == "2023-05-20T02:21:00+00:00"
    assert [t.key for t in dog.sessions[0].turns] == ["s-dog#0", "s-dog#1"]
    assert dog.sessions[0].turns[0].has_answer
    assert dog.sessions[0].turns[1].text.startswith("assistant: Keep sessions short")
    assert dog.questions[0].evidence == ["s-dog"]
    assert dog.questions[0].question_date.date().isoformat() == "2023-05-30"
    books = convs[2].questions[0]
    assert books.answer == "5" and books.evidence == ["s-books-march", "s-books-april"]


def test_longmemeval_loader_rejects_and_reports(tmp_path):
    item = json.loads(LONGMEMEVAL.read_text())[0]
    short = dict(item, haystack_dates=item["haystack_dates"][:1])
    path = tmp_path / "lme.json"
    path.write_text(json.dumps([short]))
    with pytest.raises(xb.FormatError, match="2 sessions but 2 session ids and 1 dates"):
        xb.load_longmemeval(path)
    path.write_text(json.dumps([{k: v for k, v in item.items() if k != "answer"}]))
    with pytest.raises(xb.FormatError, match="no 'answer'"):
        xb.load_longmemeval(path)
    path.write_text(json.dumps([dict(item, answer_session_ids=["s-dog", "s-gone"],
                                     question_id="x_abs")]))
    (conv,) = xb.load_longmemeval(path)
    assert conv.questions[0].evidence == ["s-dog"] and conv.questions[0].abstain
    assert "s-gone" in conv.warnings[0]


# --------------------------------------------------------------------------
# ingestion


def test_verbatim_ingest_one_memory_per_turn_with_session_time_and_turn():
    conv = xb.load_locomo(LOCOMO)[0]
    store = verbatim_store()
    ingested = xb.ingest(store, conv, dataset="locomo")
    memories = store.get_all(user_id=xb.BENCH_USER, limit=100)
    assert len(memories) == len(conv.turns) == 8
    by_turn = {m.metadata["bench"]["turns"][0]: m for m in memories}
    assert set(by_turn) == {t.key for t in conv.turns}
    photo = by_turn["D1:3"]
    assert photo.content == conv.sessions[0].turns[2].text
    assert photo.run_id == "session_1" and photo.memory_type == "episodic"
    # the session's date, turns a second apart; D1:3 is the third turn
    assert photo.created_at == photo.updated_at == "2023-05-08T13:56:02+00:00"
    assert photo.metadata["bench"] == {
        "dataset": "locomo", "conversation": "conv-mini-1", "session_id": "session_1",
        "session_date": "1:56 pm on 8 May, 2023", "sessions": ["session_1"], "turns": ["D1:3"]}
    assert photo.valid_from == photo.created_at
    assert "when" not in photo.metadata  # --when never, the default
    last = by_turn["D2:4"]
    assert last.run_id == "session_2" and last.created_at == "2023-05-25T19:30:03+00:00"
    # the save's episode carries the session's time too
    [episode] = [e for e in store.episodes(user_id=xb.BENCH_USER, limit=100)
                 if e.id in photo.source_episode_ids]
    assert episode.created_at == "2023-05-08T13:56:02+00:00"
    assert ingested.turns_of(photo) == {"D1:3"}
    assert ingested.units_of(last, "session") == {"session_2"}
    store.close()


def test_the_harness_writes_through_the_store_api_only():
    """``add(created_at=, memory_metadata=, now=)`` replaced the harness's own
    writes: no private database access, no clock patching."""
    import inspect

    source = inspect.getsource(xb)
    for gone in ("_set_created_at", "extraction_clock", "._db", "._lock",
                 "set_memory_timestamp", "update_memory"):
        assert gone not in source, gone


def test_when_always_and_never():
    conv = xb.load_locomo(LOCOMO)[1]
    for policy, expected in (("always", {"2022-06-03", "2022-06-20"}), ("never", {None})):
        store = verbatim_store()
        xb.ingest(store, conv, when=policy)
        got = {(m.metadata.get("when") or {}).get("start")
               for m in store.get_all(user_id=xb.BENCH_USER, limit=100)}
        assert got == expected
        store.close()


# --------------------------------------------------------------------------
# retrieval metrics


def test_evidence_recall_and_mrr_by_hand():
    units = [{"D1:2"}, {"D1:1", "D1:3"}, set(), {"D2:1"}]
    assert xb.evidence_recall(units, ["D1:1", "D2:1"], 1) == 0.0
    assert xb.evidence_recall(units, ["D1:1", "D2:1"], 2) == 0.5
    assert xb.evidence_recall(units, ["D1:1", "D2:1"], 5) == 1.0
    assert xb.reciprocal_rank(units, ["D1:1", "D2:1"]) == 0.5
    assert xb.reciprocal_rank(units, ["D9:9"]) == 0.0
    assert xb.evidence_recall(units, [], 5) is None and xb.reciprocal_rank(units, []) is None


def test_evidence_recall_is_full_when_the_question_uses_the_evidence_words():
    conv = xb.load_locomo(LOCOMO)[0]
    ingested = xb.ingest(verbatim_store(), conv)
    question = conv.questions[0]  # the words of D1:1
    row = xb.ask(ingested, question)
    assert row["evidence"] == ["D1:1"]
    assert row["recall@5"] == row["recall@10"] == row["recall@20"] == 1.0
    assert row["mrr"] == 1.0 and row["evidence_ranks"][0] == 1
    assert row["retrieved"][0] == ["D1:1"]
    # a question without evidence has no recall
    empty = xb.ask(ingested, conv.questions[3])
    assert empty["recall@5"] is None and empty["mrr"] is None
    ingested.store.close()


def test_evidence_recall_is_lower_when_the_evidence_shares_no_words():
    turns = [xb.Turn(key=f"D1:{i + 1}", role="Priya", raw=text, text=f"Priya: {text}")
             for i, text in enumerate(
                 [f"The kayak club met by the lake for meeting number {n}." for n in range(30)]
                 + ["Bright orange, with black stripes."])]
    session = xb.Session("session_1", xb.parse_bench_date("8 May 2023"), "8 May 2023", turns)
    unrelated = xb.Question("q-unrelated", "What colour is the kayak Priya bought?", "orange",
                            2, "temporal", ["D1:31"], "turn")
    related = xb.Question("q-related", "When did the kayak club meet for meeting number 7?",
                          "by the lake", 2, "temporal", ["D1:8"], "turn")
    conv = xb.Conversation("synthetic", "conversation", [session], [unrelated, related])
    ingested = xb.ingest(verbatim_store(), conv)
    far, near = xb.ask(ingested, unrelated), xb.ask(ingested, related)
    assert near["recall@5"] == 1.0 and near["mrr"] == 1.0
    assert far["recall@5"] == 0.0 < near["recall@5"]
    assert far["mrr"] < near["mrr"]
    ingested.store.close()


def test_longmemeval_evidence_is_counted_by_session():
    conv = xb.load_longmemeval(LONGMEMEVAL)[2]  # two answer sessions
    ingested = xb.ingest(verbatim_store(), conv)
    row = xb.ask(ingested, conv.questions[0])
    assert row["level"] == "session"
    assert row["recall@20"] == 1.0  # both sessions among the four memories
    assert set(ingested.units_of(ingested.store.get_all(user_id=xb.BENCH_USER)[0], "session")) \
        <= {"s-books-march", "s-books-april"}
    ingested.store.close()


# --------------------------------------------------------------------------
# answer scoring


def test_normalization_and_token_f1():
    assert xb.normalize_answer("  The Red, CAR! ") == "red car"
    assert xb.normalize_answer("An apple and a pear.") == "apple and pear"
    # articles are dropped on both sides, so these are the same answer
    assert xb.token_f1("the red car", "a red car") == 1.0
    assert xb.token_f1("the red car parked", "red car") == pytest.approx(0.8)
    assert xb.token_f1("red car", "blue car") == pytest.approx(0.5)
    assert xb.token_f1("Paris", "London") == 0.0
    assert xb.token_f1("", "") == 1.0 and xb.token_f1("something", "") == 0.0
    assert xb.token_f1(2019, "2019") == 1.0
    assert xb.exact_match("The Red car.", "red car")
    assert not xb.exact_match("red cars", "red car")
    assert xb.answer_contained("She drives a red car daily", "The Red Car")
    assert not xb.answer_contained("she drives a redcar", "red car")
    assert not xb.answer_contained("anything", "")


def test_locomo_rules_for_multi_hop_open_domain_and_adversarial():
    assert xb.parts_f1("Kyoto", "Kyoto, Japan") == pytest.approx(0.5)
    assert xb.parts_f1("Japan, Kyoto", "Kyoto, Japan") == 1.0
    assert xb.is_abstention("No information available.")
    assert xb.is_abstention("That is not mentioned in the conversation")
    assert xb.is_abstention("You did not mention this information.")
    assert not xb.is_abstention("Pepper")
    convs = xb.load_locomo(LOCOMO)
    q = {x.qid: x for c in convs for x in c.questions}
    multi = xb.score_answer("Kyoto", q["conv-mini-2/q1"])
    assert multi["f1"] == 0.5 and multi["em"] == 0.0 and not multi["judge"]
    open_domain = xb.score_answer("likely yes", q["conv-mini-2/q2"])
    assert open_domain["f1"] == 1.0 and open_domain["em"] == 1.0  # the first alternative
    assert xb.score_answer("No information available", q["conv-mini-1/q3"]) == {
        "f1": 1.0, "em": 1.0, "contains": 1.0, "judge": True}
    assert xb.score_answer("the saxophone", q["conv-mini-1/q3"])["f1"] == 0.0


def test_judge_hook_default_and_plugged_in():
    assert xb.containment_judge("q", "golden retriever", "A golden retriever puppy.")
    assert not xb.containment_judge("q", "golden retriever", "a labrador")
    assert xb.containment_judge("q", "You did not mention this.", "No information available")
    assert xb.load_judge(None) is xb.containment_judge
    assert xb.load_judge("evals.external_benchmarks:containment_judge") is xb.containment_judge
    with pytest.raises(ValueError):
        xb.load_judge("no_colon_here")
    question = xb.load_longmemeval(LONGMEMEVAL)[0].questions[0]
    calls = []

    def strict(q, gold, prediction):
        calls.append((q, gold, prediction))
        return False

    assert xb.score_answer("golden retriever", question, strict)["judge"] is False
    assert calls == [("What breed is my dog?", "golden retriever", "golden retriever")]


def test_the_judge_reads_the_gold_the_scores_read():
    """An open-domain gold answer is scored by its first ";" alternative, and
    the judge is handed that same gold, not the whole string."""
    question = {x.qid: x for c in xb.load_locomo(LOCOMO) for x in c.questions}["conv-mini-2/q2"]
    assert question.answer == "Likely yes; he works in solar energy"
    golds = []

    def judge(q, gold, prediction):
        golds.append(gold)
        return True

    xb.score_answer("likely yes", question, judge)
    assert golds == ["Likely yes"]
    assert xb.score_answer("likely yes", question) == {
        "f1": 1.0, "em": 1.0, "contains": 1.0, "judge": True}


class ScriptedLLM(LLM):
    """Answers each question from a script; keeps the prompts it was sent."""

    name = "scripted"
    available = True

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        self.prompts.append(user)
        question = re.search(r"Question: (.*)\n", user).group(1)
        return self.answers[question]


def test_answers_are_scored_and_grouped_by_category():
    conv = xb.load_locomo(LOCOMO)[0]
    llm = ScriptedLLM({
        conv.questions[0].question: "Pepper",
        conv.questions[1].question: "on 24 May, 2023",
        conv.questions[2].question: "a cracked bowl",
        conv.questions[3].question: "No information available",
    })
    result = xb.run_benchmark([conv], dataset="locomo", answer_llm=llm, k=5,
                              embedder=HashEmbedder(128), log=lambda _: None)
    rows = {r["qid"]: r for r in result["rows"]}
    assert rows["conv-mini-1/q0"]["f1"] == 1.0 and rows["conv-mini-1/q0"]["judge"]
    assert rows["conv-mini-1/q1"]["f1"] == pytest.approx(0.8571, abs=1e-4)
    assert rows["conv-mini-1/q1"]["contains"] == 1.0
    assert rows["conv-mini-1/q2"]["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert rows["conv-mini-1/q3"]["f1"] == 1.0
    # the model saw the top memories with their session dates
    first = llm.prompts[0]
    assert "[2023-05-08] Maya: Hey Theo! I finally adopted a rescue greyhound" in first
    assert first.count("\n- [") == 5  # k=5 memories
    by = {r["category"]: r for r in result["tables"]["by_category"]}
    assert set(by) == {1, 2, 4, 5}
    assert by[2]["f1"] == pytest.approx(0.8571, abs=1e-4)
    assert result["tables"]["overall"]["judge"] == 1.0
    assert "| 4 | single-hop | 1 |" in xb.markdown_table(result["tables"])


def test_answering_from_reconstructed_context():
    conv = xb.load_longmemeval(LONGMEMEVAL)[0]
    llm = ScriptedLLM({conv.questions[0].question: "a golden retriever"})
    result = xb.run_benchmark([conv], dataset="longmemeval", answer_llm=llm, use_context=True,
                              embedder=HashEmbedder(128), log=lambda _: None)
    (row,) = result["rows"]
    assert row["judge"] and row["contains"] == 1.0 and row["context_recall"] == 1.0
    assert "Relevant long-term memories" in llm.prompts[0]
    assert "Today is 30 May 2023." in llm.prompts[0]


def test_aggregate_groups_by_category():
    def row(category, name, evidence, r5, mrr, ms):
        return {"category": category, "category_name": name, "evidence": evidence,
                "recall@5": r5, "recall@10": r5, "recall@20": r5, "mrr": mrr, "search_ms": ms}

    rows = [row(2, "temporal", ["D1:1"], 1.0, 1.0, 2.0),
            row(2, "temporal", ["D1:2"], 0.0, 0.5, 4.0),
            row(1, "multi-hop", ["D1:1", "D2:1"], 0.5, 0.25, 1.0),
            row(5, "adversarial", [], None, None, 3.0)]
    tables = xb.aggregate(rows)
    assert [r["category"] for r in tables["by_category"]] == [1, 2, 5]
    by = {r["category"]: r for r in tables["by_category"]}
    assert by[2]["name"] == "temporal" and by[2]["n"] == 2
    assert by[2]["recall@5"] == 0.5 and by[2]["mrr"] == 0.75 and by[2]["search_ms"] == 3.0
    assert by[5]["recall@5"] is None and by[5]["with_evidence"] == 0
    overall = tables["overall"]
    assert overall["n"] == 4 and overall["with_evidence"] == 3
    assert overall["recall@5"] == 0.5 and overall["mrr"] == pytest.approx(0.5833, abs=1e-4)
    assert "f1" not in overall  # nothing was answered
    table = xb.markdown_table(tables)
    assert table.splitlines()[0] == ("| category | name | n | recall@5 | recall@10 | recall@20 "
                                     "| mrr | search ms |")
    assert "| 5 | adversarial | 1 | - | - | - | - | 3.0 |" in table
    assert "| all | overall | 4 | 0.500 |" in table
    lme = xb.aggregate([row("multi-session", "multi-session", ["s1"], 1.0, 1.0, 1.0),
                        row("knowledge-update", "knowledge-update", ["s2"], 0.0, 0.0, 1.0)])
    assert [r["category"] for r in lme["by_category"]] == ["knowledge-update", "multi-session"]


# --------------------------------------------------------------------------
# extract mode, with a stand-in for the LLM


class RuleLLM(LLM):
    """Extraction keeps each line of the transcript as a fact; every other
    question gets the answer that changes nothing."""

    name = "rule"
    available = True

    def __init__(self) -> None:
        self.todays: list[str] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            self.todays.append(re.search(r"Today's date is (\S+?)\.", system).group(1))
            transcript = user.split("Conversation:\n", 1)[1].split("\n\n", 1)[0]
            return json.dumps({"facts": [
                {"content": line, "type": "episodic", "importance": 0.5, "categories": [],
                 "entities": [], "relations": [], "when": None}
                for line in transcript.splitlines() if line.strip()]})
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        if "decide one action" in system:
            return json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new"})
        return "{}"


def test_extract_mode_traces_memories_to_their_session_and_dates_extraction():
    conv = xb.load_locomo(LOCOMO)[0]
    llm = RuleLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(128))
    ingested = xb.ingest(store, conv, mode="extract", unit="session", dataset="locomo")
    # extraction resolved dates against each session's day (add's ``now``)
    assert llm.todays == ["2023-05-08", "2023-05-25"]
    memories = store.get_all(user_id=xb.BENCH_USER, limit=100)
    assert len(memories) == 8
    session_1 = {"D1:1", "D1:2", "D1:3", "D1:4"}
    first = next(m for m in memories if m.content.startswith("Maya: Hey Theo!"))
    assert ingested.turns_of(first) == session_1  # every turn of its session
    assert set(first.metadata["bench"]["turns"]) == session_1
    assert first.created_at.startswith("2023-05-08T13:56") and first.run_id == "session_1"
    row = xb.ask(ingested, conv.questions[0])
    assert row["recall@5"] == 1.0
    store.close()
    result = xb.run_benchmark([conv], dataset="locomo", mode="extract", questions=2,
                              store_factory=lambda: MemoryStore(
                                  Config(db_path=":memory:"), llm=RuleLLM(),
                                  embedder=HashEmbedder(128)),
                              log=lambda _: None)
    assert len(result["rows"]) == 2
    assert result["config"]["extract_unit"] == "session"
    assert any("session-level" in note for note in result["notes"])


def test_extract_mode_refuses_to_run_without_an_llm():
    conv = xb.load_locomo(LOCOMO)[0]
    with pytest.raises(SystemExit, match="needs a configured LLM"):
        xb.run_benchmark([conv], dataset="locomo", mode="extract", store_factory=verbatim_store,
                         log=lambda _: None)


# --------------------------------------------------------------------------
# embeddings cache


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(64)
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return super().embed(texts)


def test_embedding_cache_embeds_each_text_once_across_runs(tmp_path):
    base = CountingEmbedder()
    cache = xb.SqliteEmbeddingCache(base, tmp_path / "cache" / "vectors.sqlite")
    assert cache.model_id == base.model_id and cache.dimensions == 64
    vectors = cache.embed(["red car", "blue car", "red car"])
    assert sorted(base.seen) == ["blue car", "red car"]
    assert np.allclose(vectors[0], vectors[2])
    cache.close()  # a store closing it leaves it usable
    assert np.allclose(cache.embed(["red car"])[0], HashEmbedder(64).embed(["red car"])[0],
                       atol=1e-6)
    cache.release()
    again = CountingEmbedder()
    reopened = xb.SqliteEmbeddingCache(again, tmp_path / "cache" / "vectors.sqlite")
    assert np.allclose(reopened.embed(["blue car"])[0], vectors[1])
    assert again.seen == [] and reopened.base_calls == 0
    reopened.release()


# --------------------------------------------------------------------------
# the command line


def test_cli_runs_locomo_end_to_end(monkeypatch, tmp_path, capsys, no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    out = tmp_path / "locomo.json"
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["file"] == str(LOCOMO)
    assert len(result["rows"]) == 7 and [s["memories"] for s in result["stores"]] == [8, 6]
    assert {r["category"] for r in result["tables"]["by_category"]} == {1, 2, 3, 4, 5}
    assert result["config"]["ingest"] == "verbatim"
    assert result["config"]["embedder"] == "hash:v1-256"
    assert result["config"]["locomo_categories"] == {str(k): v
                                                     for k, v in xb.LOCOMO_CATEGORIES.items()}
    printed = capsys.readouterr().out
    assert "| category | name | n | recall@5 |" in printed
    assert f"results: {out}" in printed


def test_cli_writes_default_results_file_and_limits(monkeypatch, tmp_path, no_models):
    shutil.copy(LONGMEMEVAL, tmp_path / "longmemeval_s.json")
    monkeypatch.setenv(xb.DATA_ENV, str(tmp_path))
    assert xb.main(["--dataset", "longmemeval", "--limit", "2", "--seed", "3"]) == 0
    (written,) = (tmp_path / "results").glob("longmemeval_*.json")
    result = json.loads(written.read_text())
    assert len(result["rows"]) == 2 and result["config"]["limit"] == 2
    assert result["config"]["variant"] == "s"
    assert all(r["level"] == "session" for r in result["rows"])


def test_cli_questions_limit_and_answer_skipped_without_llm(monkeypatch, tmp_path, no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    out = tmp_path / "r.json"
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--limit", "1",
                    "--questions", "2", "--answer", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert [r["qid"] for r in result["rows"]] == ["conv-mini-1/q0", "conv-mini-1/q1"]
    assert any("--answer skipped" in note for note in result["notes"])
    assert "f1" not in result["rows"][0]


def test_cli_without_data_says_so(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(xb.DATA_ENV, raising=False)
    assert xb.main(["--dataset", "locomo"]) == 2
    assert "MEMRY_BENCH_DATA is not set" in capsys.readouterr().err
    monkeypatch.setenv(xb.DATA_ENV, str(tmp_path))
    assert xb.main(["--dataset", "longmemeval"]) == 2


# --------------------------------------------------------------------------
# the real data, when it is there


def real_data(dataset: str) -> Path:
    data_dir = os.environ.get(xb.DATA_ENV, "").strip()
    if not data_dir:
        pytest.skip(f"{xb.DATA_ENV} is not set")
    path = xb.find_dataset(dataset, data_dir)
    if path is None:
        pytest.skip(f"no {dataset} file under {data_dir}")
    return path


def test_real_data_is_skipped_without_it(monkeypatch, tmp_path):
    monkeypatch.delenv(xb.DATA_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="is not set"):
        real_data("locomo")
    monkeypatch.setenv(xb.DATA_ENV, str(tmp_path))  # a directory without the file
    with pytest.raises(pytest.skip.Exception, match="no longmemeval file"):
        real_data("longmemeval")


@pytest.mark.parametrize("dataset", xb.DATASETS)
def test_real_data_smoke_run(dataset, tmp_path, no_models):
    path = real_data(dataset)
    assert xb.load(dataset, path)
    out = tmp_path / f"{dataset}.json"
    assert xb.main(["--dataset", dataset, "--limit", "1", "--questions", "10",
                    "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["rows"] and result["stores"][0]["memories"] > 0
    assert result["tables"]["overall"]["n"] == len(result["rows"])
