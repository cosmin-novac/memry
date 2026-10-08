"""evals/external_benchmarks.py on small fixtures in the LoCoMo and
LongMemEval formats; the runs on the real files skip when the data is absent."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import external_benchmarks as xb  # noqa: E402
from memry.config import Config  # noqa: E402
from memry.models import Memory  # noqa: E402
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


def core(scores):
    """The scores by LoCoMo's rules and the judge's verdict."""
    return {key: scores[key] for key in ("f1", "em", "contains", "judge")}


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
    assert core(xb.score_answer("No information available", q["conv-mini-1/q3"])) == {
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
    assert core(xb.score_answer("likely yes", question)) == {
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
    # the model saw the top memories as Memry renders them, dated by their session
    first = llm.prompts[0]
    assert ("- Maya: Hey Theo! I finally adopted a rescue greyhound last weekend, her name is "
            "Pepper. (said 8 May 2023)") in first
    assert first.count(" (said ") == 5  # k=5 memories; a verbatim turn is its own evidence
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
    overall = tables["overall"]  # without the adversarial category
    assert overall["n"] == 3 and overall["with_evidence"] == 3
    assert overall["recall@5"] == 0.5 and overall["mrr"] == pytest.approx(0.5833, abs=1e-4)
    assert "f1" not in overall  # nothing was answered
    table = xb.markdown_table(tables)
    assert table.splitlines()[0] == ("| category | name | n | recall@5 | recall@10 | recall@20 "
                                     "| mrr | search ms |")
    assert "| 5 | adversarial | 1 | - | - | - | - | 3.0 |" in table
    assert "| all | overall | 3 | 0.500 |" in table
    lme = xb.aggregate([row("multi-session", "multi-session", ["s1"], 1.0, 1.0, 1.0),
                        row("knowledge-update", "knowledge-update", ["s2"], 0.0, 0.0, 1.0)])
    assert [r["category"] for r in lme["by_category"]] == ["knowledge-update", "multi-session"]


# --------------------------------------------------------------------------
# extract mode, with a stand-in for the LLM


class RuleLLM(LLM):
    """Extraction keeps each numbered line of the transcript as a fact resting
    on that line; every other question gets the answer that changes nothing."""

    name = "rule"
    available = True

    def __init__(self) -> None:
        self.todays: list[str] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            self.todays.append(re.search(r"Today's date is (\S+?)\.", system).group(1))
            transcript = user.split("Conversation:\n", 1)[1].split("\n\n", 1)[0]
            lines = [re.match(r"\[(\d+)\] (.*)", line) for line in transcript.splitlines()]
            return json.dumps({"facts": [
                {"content": line.group(2), "type": "episodic", "importance": 0.5,
                 "categories": [], "entities": [], "relations": [], "when": None,
                 "sources": [int(line.group(1))]}
                for line in lines if line]})
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


class QuestionRuleLLM(RuleLLM):
    """``RuleLLM`` that also answers the question keys rule when asked: each
    fact gets one question made of its own line number, which no turn's
    words hold."""

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        raw = super().complete(system, user, json_schema=json_schema)
        if system.startswith("You are the long-term memory extraction system") \
                and "- questions:" in system:
            data = json.loads(raw)
            for item in data["facts"]:
                item["questions"] = [f"Which turn is number {item['sources'][0]}?"]
            return json.dumps(data)
        return raw


def test_question_keys_both_asks_each_pass_twice_from_one_store():
    """--question-keys both: the store writes question keys at ingest (the
    extractor is asked for them) and every pass is asked once reading them
    and once without, from the same memories."""
    conv = xb.load_locomo(LOCOMO)[0]
    llms: list[QuestionRuleLLM] = []

    def factory():
        llms.append(QuestionRuleLLM())
        return MemoryStore(Config(db_path=":memory:"), llm=llms[-1], embedder=HashEmbedder(128))

    result = xb.run_benchmark([conv], dataset="locomo", mode="extract", questions=2,
                              store_factory=factory, log=lambda _: None, question_keys="both")
    assert [p["search_decider"] for p in result["passes"]] == ["store", "store:text-only"]
    assert result["config"]["question_keys"] == "both"
    (entry,) = result["stores"]
    assert entry["question_keys"] is True and entry["memories_with_questions"] == 8
    by_pass = {}
    for row in result["rows"]:
        by_pass.setdefault(row["search_decider"], []).append(row)
    assert set(by_pass) == {"store", "store:text-only"} and len(by_pass["store"]) == 2
    # the text-only pass read no question key
    assert all("question_keyword" not in r.get("signals", {}) for r in by_pass["store:text-only"])
    with pytest.raises(ValueError, match="config, on or both"):
        xb.run_benchmark([conv], dataset="locomo", store_factory=verbatim_store,
                         log=lambda _: None, question_keys="maybe")


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


# --------------------------------------------------------------------------
# the decision provider at question time


from memry.providers.decisions import Answer, Answers, Decider, NoneDecider  # noqa: E402


class StubDecider(Decider):
    """Judges a memory relevant when it holds ``word``; every question asks
    for one property with one answer. Keeps the states it was asked about."""

    name = "stub"
    available = True
    may_rerank = True
    reranks_by_default = True

    def __init__(self, word: str) -> None:
        self.word = word
        self.states: list[str] = []
        self.closed = False

    def decide(self, state, questions):
        self.states.append(state)
        answers = {}
        for key, question in questions.items():
            if key == "property":
                value = 0.9
            elif key == "several":
                value = 0.1
            else:
                value = 0.95 if self.word in question.instructions.lower() else 0.05
            answers[key] = Answer(value=value, confidence=abs(value - 0.5) * 2, available=True)
        return Answers(answers)

    def close(self) -> None:
        self.closed = True


class NamingLLM(RuleLLM):
    """RuleLLM whose facts name the people they mention as entities."""

    PEOPLE = ("Maya", "Theo", "Pepper")

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            transcript = user.split("Conversation:\n", 1)[1].split("\n\n", 1)[0]
            return json.dumps({"facts": [
                {"content": line, "type": "episodic", "importance": 0.5, "categories": [],
                 "entities": [{"name": p, "type": "person"} for p in self.PEOPLE if p in line],
                 "relations": [], "when": None}
                for line in transcript.splitlines() if line.strip()]})
        return super().complete(system, user, json_schema=json_schema)


def test_each_question_is_asked_once_per_search_decider_on_the_same_store():
    conv = xb.load_locomo(LOCOMO)[0]
    stubs, stores = [], []

    def stub():
        stubs.append(StubDecider("pepper"))
        return stubs[-1]

    def factory():
        stores.append(MemoryStore(Config(db_path=":memory:"), llm=NamingLLM(),
                                  embedder=HashEmbedder(128)))
        return stores[-1]

    result = xb.run_benchmark([conv], dataset="locomo", mode="extract", questions=2,
                              store_factory=factory, log=lambda _: None,
                              search_deciders={"none": NoneDecider, "stub": stub})
    assert len(stores) == 1  # one store, loaded once, asked twice
    rows = result["rows"]
    assert [(r["search_decider"], r["qid"]) for r in rows] == [
        ("none", "conv-mini-1/q0"), ("none", "conv-mini-1/q1"),
        ("stub", "conv-mini-1/q0"), ("stub", "conv-mini-1/q1")]
    # the question names Maya, a person the store holds: the linked search ran both times
    for row in (rows[0], rows[2]):
        assert row["named"] == ["Maya"] and row["named_entities"] >= 1 and row["linked"]
    # only the stub pass was judged, one call a question, the question as the state
    assert [r["judge_calls"] for r in rows] == [0, 0, 1, 1]
    (used,) = stubs
    assert [state.split(" ", 1)[0] for state in used.states] == ["QUESTION:", "QUESTION:"]
    assert used.states[0].endswith("adopted last weekend?") and used.closed
    # the store got its own provider back
    assert isinstance(stores[0].decider, NoneDecider)
    assert [p["search_decider"] for p in result["passes"]] == ["none", "stub"]
    assert result["tables"] == result["passes"][0]["tables"]
    assert result["passes"][1]["tables"]["overall"]["n"] == 2
    assert result["config"]["search_deciders"] == ["none", "stub"]
    store = result["stores"][0]
    assert store["decider_failures_stub"] == 0 and store["seconds_stub"] >= 0
    assert store["actions"]["ADD"] >= 1 and store["speaker_entities"]["Maya"]
    assert {"entities", "same_name_entities", "open_proposals"} <= set(store)
    assert result["complete"] and result["stopped"] is None


def test_the_judgement_moves_the_evidence_up():
    turns = [xb.Turn(key=f"D1:{i + 1}", role="Priya", raw=text, text=f"Priya: {text}")
             for i, text in enumerate(
                 [f"The kayak club met by the lake for meeting number {n}." for n in range(10)]
                 + ["Bright orange, with black stripes."])]
    session = xb.Session("session_1", xb.parse_bench_date("8 May 2023"), "8 May 2023", turns)
    question = xb.Question("q", "What colour is the kayak Priya bought?", "orange", 4,
                           "single-hop", ["D1:11"], "turn")
    conv = xb.Conversation("synthetic", "conversation", [session], [question])
    result = xb.run_benchmark([conv], dataset="locomo", store_factory=verbatim_store,
                              log=lambda _: None,
                              search_deciders={"none": NoneDecider,
                                               "stub": lambda: StubDecider("orange")})
    none, judged = result["rows"]
    assert none["mrr"] < 1.0 and none["judge_calls"] == 0 and not none["linked"]
    assert judged["mrr"] == 1.0 and judged["judge_calls"] == 1


def test_search_decider_factories(monkeypatch):
    assert xb.search_decider("store")() is None
    assert isinstance(xb.search_decider("none")(), NoneDecider)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        xb.search_decider("jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-key")
    jev = xb.search_decider("jev")()
    assert jev.name == "jev" and jev.may_rerank and jev.failures == 0
    jev.close()
    with pytest.raises(ValueError):
        xb.search_decider("gpt")


def test_a_decider_that_answers_nothing_is_asked_again(monkeypatch):
    monkeypatch.setattr(xb.time, "sleep", lambda seconds: None)

    class Flaky(StubDecider):
        def __init__(self, fails: int) -> None:
            super().__init__("x")
            self.fails = fails

        def decide(self, state, questions):
            if self.fails:
                self.fails -= 1
                self.states.append(state)
                return Answers({key: Answer() for key in questions})
            return super().decide(state, questions)

    from memry.providers.decisions import Noul

    question = {"m0": Noul(instructions="x")}
    flaky = xb.retrying_decider(Flaky(fails=2))
    assert flaky.decide("s", question)["m0"].available and flaky.failures == 0
    assert len(flaky.states) == 3
    down = xb.retrying_decider(Flaky(fails=10))
    assert not down.decide("s", question)["m0"].available and down.failures == 1
    assert len(down.states) == xb.ATTEMPTS


def test_the_store_model_is_tried_again_after_a_transient_error(monkeypatch):
    monkeypatch.setattr(xb.time, "sleep", lambda seconds: None)
    request = httpx.Request("POST", "https://api.example/v1/chat/completions")

    class Failing(LLM):
        name = "openai"
        model = "some-model"

        def __init__(self, errors):
            self.errors = list(errors)
            self.calls = 0

        def complete(self, system, user, *, json_schema=None):
            self.calls += 1
            if self.errors:
                status = self.errors.pop(0)
                raise httpx.HTTPStatusError("failed", request=request,
                                            response=httpx.Response(status, request=request))
            return "ok"

    base = Failing([429, 503])
    llm = xb.RetryingLLM(base)
    assert llm.model == "some-model" and llm.name == "openai"
    assert llm.complete("s", "u") == "ok" and base.calls == 3
    with pytest.raises(httpx.HTTPStatusError):
        xb.RetryingLLM(Failing([400])).complete("s", "u")  # not worth another try
    with pytest.raises(httpx.HTTPStatusError):
        xb.RetryingLLM(Failing([500] * 9)).complete("s", "u")


# --------------------------------------------------------------------------
# Mem0's answer prompt and judge


from evals import mem0_judge  # noqa: E402


class ScriptedChat(LLM):
    """A chat model that replies from a script and keeps what it was sent."""

    name = "openai"
    model = "scripted-chat"
    available = True

    def __init__(self, reply) -> None:
        self.reply = reply
        self.sent: list[tuple[list, bool]] = []

    def chat(self, messages, *, json_object=False):
        self.sent.append((messages, json_object))
        return self.reply(messages) if callable(self.reply) else self.reply

    def complete(self, system, user, *, json_schema=None):
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}])


def test_mem0_prompts_are_mem0s_own():
    import hashlib

    # evaluation/prompts.py ANSWER_PROMPT_ZEP and evaluation/metrics/llm_judge.py
    # ACCURACY_PROMPT at mem0ai/mem0 b3ede5b7c0ac0e847b03786a603c107ac943b3ee
    assert hashlib.sha256(mem0_judge.ANSWER_PROMPT.encode()).hexdigest() == (
        "95b8170c30bb86f0e6819cd4846e6e3cd24c9a82030f90dbe36c880b522b92fc")
    assert hashlib.sha256(mem0_judge.ACCURACY_PROMPT.encode()).hexdigest() == (
        "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0")


def test_mem0_answer_messages_hold_memrys_lines_and_the_question():
    from memry.intelligence.context import memory_lines
    from memry.models import EvidenceTurn, Memory

    assert mem0_judge.locomo_time("2023-05-08T13:56:00+00:00") == "1:56 pm on 8 May, 2023"
    assert mem0_judge.locomo_time("2023-05-08T00:05:00+00:00") == "12:05 am on 8 May, 2023"
    assert mem0_judge.locomo_time("2023-05-08T12:00:00+00:00") == "12:00 pm on 8 May, 2023"
    memories = [Memory(content='Maya adopted "Pepper".', updated_at="2023-05-08T13:56:00+00:00",
                       metadata={"when": {"start": "2023-05-06"}}),
                Memory(content="Maya ran a 5k.", updated_at="2023-05-25T19:30:02+00:00"),
                # kept as history: an update replaced it, and moved its updated_at
                Memory(content="Maya lives in Denver.", created_at="2023-05-08T13:56:00+00:00",
                       valid_from="2023-05-08T13:56:00+00:00",
                       updated_at="2023-07-15T10:00:00+00:00",
                       invalid_at="2023-07-15T10:00:00+00:00", superseded_by="m9")]
    turn = EvidenceTurn(episode_id="e1", content="Her name is Pepper!", speaker="Maya",
                        said_at="2023-05-08T13:56:00+00:00", memory_ids=[memories[0].id])
    lines = memory_lines(memories, [turn])
    assert lines == ['[happened 2023-05-06] Maya adopted "Pepper". (said 8 May 2023)',
                     "Maya ran a 5k. (said 25 May 2023)",
                     "Maya lives in Denver. (said 8 May 2023) [until 15 July 2023]",
                     "8 May 2023: Maya: Her name is Pepper!"]
    (message,) = mem0_judge.answer_messages("Who is Pepper?", lines)
    assert message["role"] == "system"
    text = message["content"]
    assert "{{" not in text and "Question: Who is Pepper?\n    Answer:" in text
    listed = json.dumps(lines, indent=4)
    assert f"Memories:\n\n    {listed}\n\n    Question:" in text
    # the prompt's own text is Mem0's
    assert text == mem0_judge._render(mem0_judge.ANSWER_PROMPT,
                                      {"memories": listed, "question": "Who is Pepper?"})


def test_mem0_judge_reads_the_label(monkeypatch):
    chat = ScriptedChat('{"label": "CORRECT"}')
    assert mem0_judge.judge_with(chat, "When?", "7 May 2023", "On May 7th")
    (messages, json_object), = chat.sent
    assert json_object and [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == mem0_judge.ACCURACY_PROMPT.format(
        question="When?", gold_answer="7 May 2023", generated_answer="On May 7th")
    assert not mem0_judge.judge_with(ScriptedChat('```json\n{"label": "WRONG"}\n```'), "q", "g", "p")
    assert mem0_judge.judge_with(ScriptedChat('Reason. {"label": "CORRECT"}'), "q", "g", "p")
    with pytest.raises(ValueError, match="no label"):
        mem0_judge.judge_with(ScriptedChat("CORRECT"), "q", "g", "p")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mem0_judge, "_judge_model", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        mem0_judge.judge("q", "g", "p")


def test_a_judge_may_read_the_full_answer_and_a_failing_one_is_recorded():
    question = {x.qid: x for c in xb.load_locomo(LOCOMO) for x in c.questions}["conv-mini-2/q2"]
    golds = []

    def full(q, gold, prediction):
        golds.append(gold)
        return True

    full.reads_full_answer = True
    xb.score_answer("likely yes", question, full)
    assert golds == ["Likely yes; he works in solar energy"]
    assert mem0_judge.judge.reads_full_answer

    def broken(q, gold, prediction):
        raise ValueError("no label in the judge's reply")

    scores = xb.score_answer("likely yes", question, broken)
    assert scores["judge"] is None and "no label" in scores["judge_error"]
    assert scores["f1"] == 1.0
    assert xb.aggregate([{"category": 3, "category_name": "open-domain", "judge": None},
                         {"category": 3, "category_name": "open-domain",
                          "judge": True}])["overall"]["judge"] == 1.0


def test_answering_with_mem0s_prompt_through_the_runner():
    conv = xb.load_locomo(LOCOMO)[0]
    chat = ScriptedChat("Pepper")
    verdicts = []

    def judge(q, gold, prediction):
        verdicts.append((q, gold, prediction))
        return prediction == gold

    result = xb.run_benchmark([conv], dataset="locomo", answer_llm=chat, judge=judge, k=3,
                              questions=1, embedder=HashEmbedder(128), log=lambda _: None,
                              answer_prompt=mem0_judge.answer_messages)
    (row,) = result["rows"]
    assert row["prediction"] == "Pepper" and row["judge"] and row["answer_k"] == 3
    ((messages, _),) = chat.sent
    assert messages[0]["role"] == "system" and "Question: What name did Maya" in messages[0]["content"]
    assert messages[0]["content"].count("(said 8 May 2023)") >= 1
    assert result["config"]["answer_prompt"] == "evals.mem0_judge:answer_messages"
    assert result["config"]["answer_model"] == "scripted-chat"


# --------------------------------------------------------------------------
# counting, timing and capping the calls


from evals import api_usage  # noqa: E402
from memry.intelligence.context import context_lines, memory_lines  # noqa: E402
from memry.intelligence.entities import DESCRIPTION_SYSTEM  # noqa: E402


def _mock_api(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                      "prompt_tokens_details": {"cached_tokens": 4},
                      "completion_tokens_details": {"reasoning_tokens": 2}}})
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [], "usage": {"prompt_tokens": 7,
                                                               "total_tokens": 7}})
    return httpx.Response(200, json={"answers": {}})  # no usage, as a provider may send


def test_the_meter_records_every_call_and_stops_at_a_cap(tmp_path):
    from memry.intelligence.extraction import EXTRACTION_SYSTEM

    ledger = tmp_path / "usage.sqlite"
    client = httpx.Client(transport=httpx.MockTransport(_mock_api))
    extraction = {"model": "text-model", "messages": [
        {"role": "system", "content": EXTRACTION_SYSTEM.format(today="2023-05-08")},
        {"role": "user", "content": f"Conversation:\nMaya: hi\n\n{xb.SHARED_CONTEXT}:\nx"}]}
    on_update = {"model": "text-model", "messages": [
        {"role": "system", "content": EXTRACTION_SYSTEM.format(today="2023-05-08")},
        {"role": "user", "content": "Conversation:\nuser: Maya has a dog."}]}
    send = httpx.Client.send
    with api_usage.UsageMeter(str(ledger), caps={"chat": 2}, refine=xb.memry_stage) as meter:
        with api_usage.labelled("conv-1"), api_usage.stage("ingest"):
            client.post("https://api.openai.com/v1/chat/completions", json=extraction)
            client.post("https://api.openai.com/v1/embeddings",
                        json={"model": "emb", "input": ["a", "b"]})
            client.post("https://api.openai.com/v1/chat/completions", json=on_update)
        with api_usage.stage("search:jev"):
            client.post("https://api.typesafe.ai/v1/systemone", json={"state": "QUESTION: x"})
        try:
            client.post("https://api.openai.com/v1/chat/completions", json=extraction)
        except Exception:  # what a store does after a failed call
            pytest.fail("a refused call must not be an ordinary exception")
        except api_usage.CapReached:
            pass
        else:
            pytest.fail("the third chat call was sent")
        assert meter.capped == "chat" and meter.calls("chat") == 2 and meter.calls() == 4
    assert httpx.Client.send is send  # uninstalled
    rows = {row["stage"]: row for row in api_usage.summarize(str(ledger), ("stage",))}
    assert set(rows) == {"ingest", "ingest:extraction", "ingest:extraction_on_update",
                         "search:jev"}
    first = rows["ingest:extraction"]
    assert (first["calls"], first["input_tokens"], first["output_tokens"],
            first["cached_tokens"], first["reasoning_tokens"]) == (1, 10, 3, 4, 2)
    assert rows["ingest"]["input_tokens"] == 7 and rows["ingest"]["items"] == 2
    jev = rows["search:jev"]
    assert jev["without_usage"] == 1 and jev["request_chars"] > 0 and jev["response_chars"] > 0
    assert jev["median_seconds"] is not None
    by_label = api_usage.summarize(str(ledger), ("label", "grp"))
    assert {(r["label"], r["grp"], r["calls"]) for r in by_label} == {
        ("conv-1", "chat", 2), ("conv-1", "embeddings", 1), ("", "jev", 1)}
    client.close()


def test_memry_stage_names_the_prompt():
    from memry.intelligence.entities import IDENTITY_SYSTEM
    from memry.intelligence.extraction import COVERAGE_SYSTEM
    from memry.intelligence.reconcile import MERGE_REQUEST, RECONCILE_SYSTEM

    def body(system, user="x"):
        return {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}

    assert xb.memry_stage("ingest", "chat", body(RECONCILE_SYSTEM)) == "ingest:reconcile"
    assert xb.memry_stage("ingest", "chat", body(RECONCILE_SYSTEM, MERGE_REQUEST)) == \
        "ingest:reconcile_merge"
    assert xb.memry_stage("ingest", "chat", body(IDENTITY_SYSTEM)) == "ingest:identity"
    assert xb.memry_stage("ingest", "chat", body(COVERAGE_SYSTEM)) == "ingest:audit"
    assert xb.memry_stage("ingest", "chat", body("Something else")) == "ingest:other"
    assert xb.memry_stage("answer", "chat", body(RECONCILE_SYSTEM)) is None
    assert xb.memry_stage("ingest", "embeddings", {"input": ["a"]}) is None


def test_a_run_stops_cleanly_at_a_cap():
    convs = xb.load_locomo(LOCOMO)
    asked = []

    def judge(q, gold, prediction):
        asked.append(q)
        if len(asked) == 2:
            raise api_usage.CapReached("chat: 2 calls, cap 2")
        return True

    result = xb.run_benchmark(convs, dataset="locomo", answer_llm=ScriptedChat("x"), judge=judge,
                              embedder=HashEmbedder(128), log=lambda _: None)
    assert not result["complete"] and result["stopped"].startswith("conv-mini-1: chat")
    assert [r["qid"] for r in result["rows"]] == ["conv-mini-1/q0"]
    assert [s["conversation"] for s in result["stores"]] == ["conv-mini-1"]
    assert result["stores"][0]["stopped"] == "chat: 2 calls, cap 2"
    assert xb.parse_caps(["chat=15000", "jev=2000"]) == {"chat": 15000, "jev": 2000}
    with pytest.raises(ValueError):
        xb.parse_caps(["tokens=5"])


# --------------------------------------------------------------------------
# parallel, resumable runs


def test_cli_runs_conversations_in_processes_and_resumes(monkeypatch, tmp_path, no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    parts = tmp_path / "parts"
    out = tmp_path / "all.json"
    ledger = tmp_path / "usage.sqlite"
    argv = ["--dataset", "locomo", "--file", "locomo_mini.json", "--search-decider", "none",
            "--search-decider", "store", "--results-dir", str(parts), "--out", str(out),
            "--usage-db", str(ledger), "--max-calls", "chat=10"]
    assert xb.main([*argv, "--jobs", "2"]) == 0
    result = json.loads(out.read_text())
    assert result["complete"] and len(result["rows"]) == 14
    assert [p["search_decider"] for p in result["passes"]] == ["none", "store"]
    assert result["passes"][0]["tables"]["overall"]["n"] == 6  # no adversarial question
    assert sorted(s["conversation"] for s in result["stores"]) == ["conv-mini-1", "conv-mini-2"]
    assert (parts / "conv-mini-1.log").exists()
    assert result["usage"]["by_stage"] == []  # no model was called
    first = json.loads((parts / "conv-mini-1.json").read_text())
    assert first["complete"] and first["config"]["options"]["search_decider"] == ["none", "store"]
    # a run again reuses what is complete and runs what is missing
    (parts / "conv-mini-2.json").unlink()
    assert xb.main([*argv, "--jobs", "1"]) == 0
    assert json.loads((parts / "conv-mini-1.json").read_text())["created"] == first["created"]
    assert json.loads((parts / "conv-mini-2.json").read_text())["complete"]
    assert len(json.loads(out.read_text())["rows"]) == 14
    # other options: nothing is reused
    assert xb.main([*argv, "--k", "5", "--conversation", "conv-mini-1"]) == 0
    again = json.loads((parts / "conv-mini-1.json").read_text())
    assert again["config"]["k"] == 5 and again["complete"]
    assert len(json.loads(out.read_text())["rows"]) == 8


def test_cli_rejects_jobs_without_a_results_dir(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        xb.parse_args(["--dataset", "locomo", "--jobs", "2"])
    with pytest.raises(SystemExit):
        xb.parse_args(["--dataset", "locomo", "--max-calls", "chat=5"])
    assert xb.parse_args(["--dataset", "locomo", "--usage-db", "u", "--max-calls",
                          "jev=3"]).caps == {"jev": 3}


# --------------------------------------------------------------------------
# the published protocol: categories, several k, judge runs, Mem0's scores


class CountingChat(ScriptedChat):
    """Answers with how many memories it was shown; reports usage like OpenAI."""

    def __init__(self) -> None:
        super().__init__(None)
        self._usage = None

    def chat(self, messages, *, json_object=False):
        self.sent.append((messages, json_object))
        shown = messages[0]["content"].count('",\n') + 1 if '"' in messages[0]["content"] else 0
        self._usage = {"prompt_tokens": 100 + shown, "completion_tokens": 3}
        return f"shown {shown}"

    def last_usage(self):
        return self._usage


def test_one_search_answered_at_several_k_and_judged_three_times():
    conv = xb.load_locomo(LOCOMO)[0]
    searches = []

    def factory():
        store = verbatim_store()
        search = store.search

        def counted(*args, **kwargs):
            searches.append(args[0])
            return search(*args, **kwargs)

        store.search = counted
        return store

    runs = []

    def judge(q, gold, prediction):
        runs.append(prediction)
        return len(runs) % 3 != 0  # every third verdict is wrong

    chat = CountingChat()
    result = xb.run_benchmark([conv], dataset="locomo", store_factory=factory, answer_llm=chat,
                              judge=judge, k=5, ks=[3, 5], judge_runs=3,
                              categories={"1", "2", "3", "4"}, log=lambda _: None,
                              answer_prompt=mem0_judge.answer_messages)
    rows = result["rows"]
    assert [r["qid"] for r in rows] == ["conv-mini-1/q0", "conv-mini-1/q1", "conv-mini-1/q2"]
    assert len(searches) == 3  # one search a question, whatever the number of k
    assert len(chat.sent) == 6 and len(runs) == 18  # two answers, three verdicts each
    row = rows[0]
    assert set(row["answers"]) == {"3", "5"} and row["answer_k"] == 5
    assert row["answers"]["3"]["prediction"] == "shown 3" and row["prediction"] == "shown 5"
    assert len(row["judges"]) == 3 and row["judge"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["answers"]["5"]["input_tokens"] == 105
    assert row["answer_seconds"] >= 0 and row["answers"]["3"]["k"] == 3
    # the tables: the headline k at the top, each k apart, the runs and their spread
    overall = result["tables"]["overall"]
    assert overall["n"] == 3 and len(overall["judge_runs"]) == 3 and overall["judge_std"] >= 0
    by_k = result["passes"][0]["tables_by_k"]
    assert set(by_k) == {"3", "5"} and by_k["5"]["overall"]["judge"] == overall["judge"]
    assert result["config"]["ks"] == [3, 5] and result["config"]["judge_runs"] == 3
    assert result["config"]["categories"] == ["1", "2", "3", "4"]


def test_rows_keep_the_memories_shown_and_the_reference_date():
    conv = xb.load_locomo(LOCOMO)[0]
    result = xb.run_benchmark([conv], dataset="locomo", store_factory=verbatim_store,
                              answer_llm=CountingChat(), k=3, questions=1, log=lambda _: None,
                              answer_prompt=mem0_judge.answer_messages)
    (row,) = result["rows"]
    assert row["reference_date"] == "2023-05-25T19:30:00+00:00"  # the last session
    first = row["memories"][0]
    assert first["memory"].startswith("Maya: Hey Theo!") and first["turns"] == ["D1:1"]
    assert first["created_at"] == "2023-05-08T13:56:00+00:00" and "score" in first
    assert len(row["memories"]) == len(conv.turns)  # all of them, up to the search depth
    tokens = xb.count_tokens(mem0_judge.memories_json(memory_lines(
        [Memory(content=m["memory"], updated_at=m["updated_at"]) for m in row["memories"][:3]])))
    assert row["context_tokens"] == tokens


class DescribingLLM(RuleLLM):
    """A fact keeps the first words of its line (the turn says more) and
    names Maya where it says her name; an entity's description is its first
    fact."""

    descriptions = 0

    def complete(self, system, user, *, json_schema=None):
        if system == DESCRIPTION_SYSTEM:
            self.descriptions += 1
            first = user.split("Active evidence:\n- ", 1)[1].split("\n", 1)[0]
            return json.dumps({"description": f"Known for: {first}"})
        raw = super().complete(system, user, json_schema=json_schema)
        if not system.startswith("You are the long-term memory extraction system"):
            return raw
        facts = json.loads(raw)["facts"]
        for fact in facts:
            fact["content"] = " ".join(fact["content"].split()[:5])
            if "Maya" in fact["content"]:
                fact["entities"] = [{"name": "Maya", "type": "person"}]
        return json.dumps({"facts": facts})


def _described_run(descriptions=True):
    conv = xb.load_locomo(LOCOMO)[0]
    chat = ScriptedChat("Pepper")
    llm = DescribingLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(128))
    try:
        ingested = xb.ingest(store, conv, mode="extract", unit="session", dataset="locomo")
        question = conv.questions[0]
        before = llm.descriptions
        row = xb.ask(ingested, question, k=3, answer_llm=chat,
                     answer_prompt=mem0_judge.answer_messages, descriptions=descriptions)
        row["descriptions built"] = llm.descriptions - before
        results = store.search(question.question, user_id=xb.BENCH_USER, limit=xb.DEPTH,
                               evidence=False)[:3]
        turns = store.evidence(question.question, results, user_id=xb.BENCH_USER)
        described = store.described_entities(question.question, user_id=xb.BENCH_USER)
        agent = store.reconstruct_context(question.question, user_id=xb.BENCH_USER,
                                          limit=3, token_budget=4000)
    finally:
        store.close()
    ((messages, _),) = chat.sent
    return question, row, messages, [r.memory for r in results], turns, described, agent, ingested


def test_the_runners_memory_list_is_memrys_rendering_with_the_evidence():
    """The answer prompt's memory list is what Memry's context builder gives
    an agent for the question (``context_lines``): the description of Maya,
    whom the question names, then the top k memories with their dates, then
    their source turns that best match the question, as
    ``MemoryStore.evidence`` chooses them. An agent's context
    (``reconstruct_context``) holds the same lines."""
    question, row, messages, memories, turns, described, agent, ingested = _described_run()
    expected = context_lines(described, memories, turns)
    assert messages == mem0_judge.answer_messages(question.question, expected)
    assert [e.name for e in described] == row["described"] == ["Maya"]
    assert row["descriptions built"] == 1  # built at the question, then kept
    assert expected[0].startswith("Maya (person): Known for: Maya:")
    assert turns and len(expected) == 1 + 3 + len(turns)
    assert all(f"- {line}" in agent.text for line in expected)
    # the row names the turns shown
    assert row["answers"]["3"]["evidence"] == [
        ingested.turn_of_episode[t.episode_id] for t in turns]


def test_no_descriptions_leaves_the_entities_out_of_the_memory_list():
    """``descriptions=False`` (``--no-descriptions``, an ablation): the list
    holds the memories and their turns only, and no description is built."""
    question, row, messages, memories, turns, _, _, _ = _described_run(descriptions=False)
    assert messages == mem0_judge.answer_messages(question.question,
                                                  memory_lines(memories, turns))
    assert row["described"] == [] and row["descriptions built"] == 0


def test_a_compared_answer_model_answers_from_the_same_memory_list():
    """--compare-answer-model has a second model answer each k from the one
    search made and the same memory list, turns and descriptions included,
    judged as the others under stages of their own."""
    judged: list[tuple[str, str]] = []

    def judge(question, gold, prediction):
        judged.append((api_usage.current_stage(), prediction))
        return prediction == "Pepper"

    conv = xb.load_locomo(LOCOMO)[0]
    first, second = ScriptedChat("Unknown"), ScriptedChat("Pepper")
    second.model = "second-chat"
    store = MemoryStore(Config(db_path=":memory:"), llm=RuleLLM(), embedder=HashEmbedder(128))
    searches = []
    try:
        ingested = xb.ingest(store, conv, mode="extract", unit="session", dataset="locomo")
        search = store.search

        def counted(*args, **kwargs):
            searches.append(args)
            return search(*args, **kwargs)

        store.search = counted
        row = xb.ask(ingested, conv.questions[0], k=3, ks=[3, 5], answer_llm=first,
                     judge=judge, answer_prompt=mem0_judge.answer_messages, judge_runs=2,
                     compare_answer_llm=second)
    finally:
        store.close()
    assert len(searches) == 1
    # the same messages, each k once, to each model
    assert [m for m, _ in first.sent] == [m for m, _ in second.sent] and len(first.sent) == 2
    assert set(row["answers"]) == set(row["answers_compared"]) == {"3", "5"}
    for at in ("3", "5"):
        assert row["answers"][at]["prediction"] == "Unknown"
        assert row["answers_compared"][at]["prediction"] == "Pepper"
        assert row["answers_compared"][at]["evidence"] == row["answers"][at]["evidence"]
    assert sum(stage == "judge:compared" for stage, _ in judged) == 4
    assert sum(stage == "judge" for stage, _ in judged) == 4
    tables = xb.pass_tables([row], ["store"])[0]
    assert tables["compared_by_k"]["3"]["overall"]["judge"] == 1.0
    assert tables["tables_by_k"]["3"]["overall"]["judge"] == 0.0


def test_compared_evidence_answers_come_from_the_same_search():
    """--compare-evidence-tokens 0 answers each k again from the memories of
    the one search made, without their turns, and judges those answers as
    the others, counting their calls under stages of their own."""
    class SummaryLLM(RuleLLM):
        def complete(self, system, user, *, json_schema=None):
            raw = super().complete(system, user, json_schema=json_schema)
            if not system.startswith("You are the long-term memory extraction system"):
                return raw
            facts = json.loads(raw)["facts"]
            for fact in facts:
                fact["content"] = " ".join(fact["content"].split()[:5])
            return json.dumps({"facts": facts})

    judged: list[tuple[str, str]] = []

    def judge(question, gold, prediction):
        judged.append((api_usage.current_stage(), prediction))
        return prediction == "Pepper"

    conv = xb.load_locomo(LOCOMO)[0]
    # the reply says whether the memory list held a turn ("8 May 2023: Maya: ...")
    chat = ScriptedChat(lambda messages: "Pepper" if re.search(
        r'"\d{1,2} [A-Z][a-z]+ \d{4}: ', messages[0]["content"]) else "Unknown")
    store = MemoryStore(Config(db_path=":memory:"), llm=SummaryLLM(),
                        embedder=HashEmbedder(128))
    searches = []
    try:
        ingested = xb.ingest(store, conv, mode="extract", unit="session", dataset="locomo")
        search = store.search

        def counted(*args, **kwargs):
            searches.append(args)
            return search(*args, **kwargs)

        store.search = counted
        question = conv.questions[0]
        row = xb.ask(ingested, question, k=3, ks=[3, 5], answer_llm=chat, judge=judge,
                     answer_prompt=mem0_judge.answer_messages, judge_runs=2,
                     compare_evidence_tokens=0)
        results = search(question.question, user_id=xb.BENCH_USER, limit=xb.DEPTH,
                         evidence=False)
    finally:
        store.close()
    assert len(searches) == 1
    assert set(row["answers"]) == set(row["answers_compared"]) == {"3", "5"}
    sent = [messages for messages, _ in chat.sent]
    for at in (3, 5):
        facts_only = mem0_judge.answer_messages(
            question.question, memory_lines([r.memory for r in results[:at]]))
        assert facts_only in sent
        compared = row["answers_compared"][str(at)]
        assert compared["evidence"] == [] and compared["k"] == at
        assert len(compared["judges"]) == 2 and "f1_mem0" in compared
    # the headline answers were shown their turns, the compared ones none
    assert row["answers"]["3"]["evidence"] and row["answers"]["3"]["prediction"] == "Pepper"
    assert row["answers_compared"]["3"]["prediction"] == "Unknown"
    assert len(sent) == 4 and len(judged) == 8
    assert sorted({stage for stage, _ in judged}) == ["judge", "judge:compared"]
    assert sum(stage == "judge:compared" for stage, _ in judged) == 4
    tables = xb.pass_tables([row], ["store"])[0]
    assert set(tables["compared_by_k"]) == {"3", "5"}
    assert tables["compared_by_k"]["3"]["overall"]["judge"] == 0.0
    assert tables["tables_by_k"]["3"]["overall"]["judge"] == 1.0


def test_mem0_f1_and_bleu1_are_mem0s():
    # Mem0's calculate_metrics: token sets after simple_tokenize, the whole gold
    assert xb.mem0_f1("The Red car.", "red car") == pytest.approx(0.8)
    assert xb.mem0_f1("red red car", "red car") == 1.0  # sets, not counts
    assert xb.mem0_f1("", "red car") == 0.0 and xb.mem0_f1("x", "") == 0.0
    assert xb.mem0_f1("Kyoto", "Kyoto, Japan") == pytest.approx(2 / 3)
    nltk = pytest.importorskip("nltk")
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    try:
        nltk.word_tokenize("a")
    except LookupError:
        pytest.skip("NLTK's punkt_tab data is not installed")
    expected = sentence_bleu([nltk.word_tokenize("on 7 may 2023")],
                             nltk.word_tokenize("she went on 7 may, 2023."),
                             weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    assert xb.bleu1("She went on 7 May, 2023.", "On 7 May 2023") == pytest.approx(expected)
    assert xb.bleu1("", "red car") == 0.0


def test_full_context_answers_from_the_whole_conversation_as_mem0():
    import hashlib

    assert hashlib.sha256(mem0_judge.FULL_CONTEXT_PROMPT.encode()).hexdigest() == (
        "744495b77f2955d437017fd33a0b7156ef41426b7ae8277e5efb92382f234b78")
    assert hashlib.sha256(mem0_judge.FULL_CONTEXT_SYSTEM.encode()).hexdigest() == (
        "0c6b92630ba4c22fd29e718d095abb2d6ffba10c04d00962e94bca4a65b23249")
    # as jinja2 3.1.6 renders the template: one newline at the very end dropped
    turns = [Memory(content="b: c", created_at="2023-05-08T13:56:00+00:00")]
    (system, user) = mem0_judge.full_context_messages("Q?", turns)
    assert system == {"role": "system", "content": mem0_judge.FULL_CONTEXT_SYSTEM}
    assert user["content"] == ("\n# Question: \nQ?\n\n# Context: \n1:56 pm on 8 May, 2023 | b: c"
                               "\n\n\n# Short answer:")
    conv = xb.load_locomo(LOCOMO)[0]
    chat = ScriptedChat("Pepper")

    def no_store():
        raise AssertionError("full context needs no store")

    result = xb.run_benchmark([conv], dataset="locomo", full_context=True, answer_llm=chat,
                              judge=lambda q, g, p: True, store_factory=no_store,
                              categories={"4"}, log=lambda _: None,
                              answer_prompt=mem0_judge.full_context_messages)
    (row,) = result["rows"]
    assert row["search_decider"] == "full-context" and row["prediction"] == "Pepper"
    assert "memories" not in row and row["answer_k"] == len(conv.turns)
    ((messages, _),) = chat.sent
    text = messages[1]["content"]
    # every turn, its session's time, the file's text without the photo's caption
    assert "1:56 pm on 8 May, 2023 | Maya: She is shy but she loves the soft blanket by the " \
           "window.\n" in text and "shares a photo" not in text
    assert text.count(" | ") == len(conv.turns) and text.endswith("# Short answer:")
    assert result["config"]["full_context"] and result["config"]["embedder"] is None


def test_corrected_answers_are_judged_too(tmp_path):
    errors = [
        {"question_id": "locomo_0_qa0", "error_type": "HALLUCINATION", "category": 4,
         "golden_answer": "Pepper", "correct_answer": "Pepper the greyhound"},
        {"question_id": "locomo_1_qa0", "error_type": "WRONG_CITATION", "category": 2,
         "golden_answer": "2019", "correct_answer": "2019"},
    ]
    path = tmp_path / "errors.json"
    path.write_text(json.dumps(errors))
    corrections = xb.load_corrections(path, LOCOMO)
    assert set(corrections) == {"conv-mini-1/q0"}  # a wrong citation changes no score
    conv = xb.load_locomo(LOCOMO)[0]

    def judge(q, gold, prediction):
        return gold == prediction

    result = xb.run_benchmark([conv], dataset="locomo", store_factory=verbatim_store,
                              answer_llm=ScriptedChat("Pepper the greyhound"), judge=judge,
                              questions=2, corrections=corrections, log=lambda _: None,
                              answer_prompt=mem0_judge.answer_messages)
    audited, other = result["rows"]
    assert audited["audit"] == {"error_type": "HALLUCINATION",
                                "correct_answer": "Pepper the greyhound"}
    assert audited["judge"] is False and audited["judge_corrected"] is True
    assert audited.get("judge_clean") is None
    assert "audit" not in other and other["judge_corrected"] == other["judge_clean"] is False
    overall = result["tables"]["overall"]
    assert overall["judge"] == 0.0 and overall["judge_corrected"] == 0.5
    assert overall["judge_clean"] == 0.0
    with pytest.raises(xb.FormatError):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"question_id": "locomo_9_qa0", "error_type": "AMBIGUOUS",
                                    "correct_answer": "x"}]))
        xb.load_corrections(bad, LOCOMO)


def test_the_mem0_export_has_mem0s_shape():
    convs = xb.load_locomo(LOCOMO)
    result = xb.run_benchmark(convs, dataset="locomo", store_factory=verbatim_store,
                              answer_llm=ScriptedChat("Pepper"), judge=lambda q, g, p: True,
                              k=2, ks=[2, 4], categories={"1", "2", "3", "4"},
                              log=lambda _: None, answer_prompt=mem0_judge.answer_messages)
    exported = mem0_judge.export_results(result, LOCOMO)
    assert list(exported) == ["0", "1"]
    raw = json.loads(LOCOMO.read_text())
    first = exported["0"][0]
    assert set(first) == {
        "question", "answer", "category", "evidence", "response", "adversarial_answer",
        "speaker_1_memories", "speaker_2_memories", "num_speaker_1_memories",
        "num_speaker_2_memories", "speaker_1_memory_time", "speaker_2_memory_time",
        "speaker_1_graph_memories", "speaker_2_graph_memories", "response_time"}
    assert first["question"] == raw[0]["qa"][0]["question"] and first["response"] == "Pepper"
    assert first["answer"] == raw[0]["qa"][0]["answer"] and first["category"] == 4
    assert first["num_speaker_1_memories"] == 2 and first["speaker_2_memories"] == []
    assert first["speaker_1_memories"][0]["timestamp"] == "1:56 pm on 8 May, 2023"
    assert exported["1"][0]["answer"] == 2019  # the dataset's own value, a number here
    assert [len(v) for v in exported.values()] == [3, 3]  # no adversarial question asked
    at_four = mem0_judge.export_results(result, LOCOMO, k=4)
    assert at_four["0"][0]["num_speaker_1_memories"] == 4


def test_the_store_gets_the_decider_asked_for(monkeypatch, tmp_path, no_models):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key")  # a text model; nothing is called
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        xb.make_store("extract", HashEmbedder(64), decider="jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-key")
    store = xb.make_store("extract", HashEmbedder(64), decider="jev",
                          db_path=xb.fresh_db(tmp_path / "s" / "conv.sqlite"))
    assert store.decider.name == "jev" and store.decider.failures == 0
    assert store.config.decision.provider == "jev" and store.relevance_mode() == "jev"
    assert isinstance(store.llm, xb.RetryingLLM) and store.llm.model == "gpt-6-luna"
    store.close()
    assert (tmp_path / "s" / "conv.sqlite").exists()
    none = xb.make_store("extract", HashEmbedder(64), decider="none")
    assert none.decider.name == "none" and none.relevance_mode() == "vector"
    none.close()
    (tmp_path / "s" / "conv.sqlite-wal").write_text("left over")
    xb.fresh_db(tmp_path / "s" / "conv.sqlite")
    assert list((tmp_path / "s").iterdir()) == []


def test_jev_calls_at_the_save_are_named_by_their_question():
    def body(state, *keys):
        return {"state": state, "questions": dict.fromkeys(keys, {})}

    assert xb.memry_stage("ingest", "jev", body("EXISTING", "action", "target")) == \
        "ingest:jev:reconcile"
    assert xb.memry_stage("ingest", "jev", body("Two entries", "pair", "belongs")) == \
        "ingest:jev:identity_pair"
    assert xb.memry_stage("ingest", "jev", body("A memory from a personal long-term memory "
                                                "store: x", "s0", "s1")) == "ingest:jev:name_screen"
    assert xb.memry_stage("ingest", "jev", body("A memory from a personal long-term memory "
                                                "store: x", "k")) == "ingest:jev:when"
    assert xb.memry_stage("ingest", "jev", body("A name from one person's long-term memory "
                                                "store: x", "n0")) == "ingest:jev:name_check"
    assert xb.memry_stage("search:store", "jev", body("QUESTION: x", "m0")) is None


def test_a_stage_reaches_calls_made_from_a_store_s_own_threads(tmp_path):
    import threading

    client = httpx.Client(transport=httpx.MockTransport(_mock_api))
    with api_usage.UsageMeter(str(tmp_path / "u.sqlite")):
        with api_usage.labelled("conv-9"), api_usage.stage("ingest"):
            worker = threading.Thread(target=lambda: client.post(
                "https://api.typesafe.ai/v1/systemone", json={"state": "x"}))
            worker.start()
            worker.join()
    (row,) = api_usage.summarize(str(tmp_path / "u.sqlite"), ("label", "stage"))
    assert (row["label"], row["stage"]) == ("conv-9", "ingest")
    client.close()


def test_cli_asks_only_the_categories_given(monkeypatch, tmp_path, no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    out = tmp_path / "r.json"
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--categories",
                    "1,2,3,4", "--ks", "5,10", "--k", "10", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert {r["category"] for r in result["rows"]} == {1, 2, 3, 4}
    assert result["config"]["ks"] == [5, 10] and result["config"]["options"]["categories"]
    with pytest.raises(SystemExit):
        xb.parse_args(["--dataset", "locomo", "--ks", "0,5"])


def test_ledgers_opened_at_the_same_moment_all_open(tmp_path):
    """Worker processes open one usage ledger at the same moment. Switching a
    new file to write-ahead mode can meet another opener's lock without the
    busy timeout applying, and a worker failed with "database is locked". Each
    opener now tries again until the file is in write-ahead mode."""
    import threading

    failures: list[str] = []
    for attempt in range(40):
        ledger = tmp_path / f"usage-{attempt}.sqlite"
        barrier = threading.Barrier(4)
        meters: list = []

        def open_one() -> None:
            barrier.wait()
            try:
                meters.append(api_usage.UsageMeter(str(ledger)))
            except Exception as exc:  # noqa: BLE001 - the failure is the finding
                failures.append(str(exc))

        threads = [threading.Thread(target=open_one) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for meter in meters:
            assert meter._db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            meter.close()
    assert failures == []
