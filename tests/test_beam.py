"""BEAM: the loader in evals/external_benchmarks.py and the grader in
evals/beam_eval.py, on a small chat in BEAM's format, with no model call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import beam_eval as be  # noqa: E402
from evals import external_benchmarks as xb  # noqa: E402


def message(mid, role, content, qtype=None, anchor=None):
    return {"id": mid, "role": role, "content": content, "question_type": qtype,
            "index": None, "time_anchor": anchor}


def beam_row(**overrides):
    questions = {
        "abstention": [{"question": "Who reviewed my thesis?",
                        "ideal_response": "There is no information about that.",
                        "rubric": ["States that the chat does not say"]}],
        "knowledge_update": [{"question": "How fast is the page now?", "answer": "250 ms",
                              "rubric": ["LLM response should contain: 250 ms"],
                              "source_chat_ids": {"original_info": [2], "updated_info": [6]}}],
        "event_ordering": [{"question": "In which order did I bring these up?",
                            "answer": "budget, login, deploy",
                            "rubric": ["Budget", "Login", "Deploy"],
                            "source_chat_ids": [[0, 2], [6]]}],
    }
    row = {
        "conversation_id": "7",
        "chat": [
            [message(0, "user", "I start the budget app today.", "main_question", "March-15-2024"),
             message(1, "assistant", "Good plan."),
             message(2, "user", "The page loads in 900 ms.", "main_question"),
             message(3, "assistant", "x" * 6000 + " " * 500 + "end")],
            [message(4, "user", "Now the login.", "main_question", "April-05-2024"),
             message(5, "assistant", "Sure."),
             message(6, "user", "Caching took the page to 250 ms.", "main_question"),
             message(7, "assistant", "Nice.")],
        ],
        "probing_questions": repr(questions),
    }
    row.update(overrides)
    return row


@pytest.fixture
def beam_file(tmp_path):
    path = tmp_path / "beam_100K.json"
    path.write_text(json.dumps([beam_row()]))
    return path


def test_beam_anchor_dates_parse():
    assert xb.parse_bench_date("March-15-2024").date().isoformat() == "2024-03-15"
    assert xb.parse_bench_date("Sept-3-2024").date().isoformat() == "2024-09-03"


def test_load_beam_reads_batches_questions_and_evidence(beam_file, monkeypatch):
    monkeypatch.setattr(xb, "BEAM_SESSION_CHARS", 1000)
    [conv] = xb.load_beam(beam_file)
    assert conv.conv_id == "beam-7"
    # the long reply opens a session of its own; the padding is one space
    assert [s.session_id for s in conv.sessions] == ["b1s1", "b1s2", "b1s3", "b2s1"]
    assert [[t.key for t in s.turns] for s in conv.sessions][:3] == [
        ["beam-7#0", "beam-7#1"], ["beam-7#2"], ["beam-7#3"]]
    assert conv.sessions[2].turns[0].raw.endswith("x end")
    assert {s.date.date().isoformat() for s in conv.sessions[:3]} == {"2024-03-15"}
    assert conv.sessions[3].date.date().isoformat() == "2024-04-05"
    assert conv.sessions[0].turns[1].text == "assistant: Good plan."
    by_kind = {q.category: q for q in conv.questions}
    assert [q.category for q in conv.questions] == ["abstention", "event_ordering",
                                                     "knowledge_update"]
    update = by_kind["knowledge_update"]
    assert update.qid == "beam-7/knowledge_update/0"
    assert update.answer == "250 ms"
    assert update.evidence == ["beam-7#2", "beam-7#6"]
    assert update.extra["rubric"] == ["LLM response should contain: 250 ms"]
    assert update.extra["beam_ability"] == "knowledge_update"
    assert update.reference_date.startswith("2024-04-05")
    assert by_kind["event_ordering"].evidence == ["beam-7#0", "beam-7#2", "beam-7#6"]
    assert by_kind["abstention"].abstain and not update.abstain
    assert conv.warnings == []


def test_beam_sessions_keep_exchanges_whole_under_the_cap(beam_file):
    [conv] = xb.load_beam(beam_file)
    assert [len(s.turns) for s in conv.sessions] == [4, 4]


def test_beam_reads_a_datasets_server_page(tmp_path):
    path = tmp_path / "page.json"
    path.write_text(json.dumps({"rows": [{"row_idx": 0, "row": beam_row()}]}))
    assert [c.conv_id for c in xb.load_beam(path)] == ["beam-7"]


def test_beam_without_rubric_is_a_format_error(tmp_path):
    path = tmp_path / "beam_100K.json"
    path.write_text(json.dumps([beam_row(probing_questions=repr(
        {"abstention": [{"question": "Who?", "rubric": []}]}))]))
    with pytest.raises(xb.FormatError):
        xb.load_beam(path)


def test_beam_variant_defaults_and_mismatches():
    assert xb.parse_args(["--dataset", "beam"]).variant == "100K"
    assert xb.parse_args(["--dataset", "longmemeval"]).variant == "s"
    assert xb.parse_args(["--dataset", "beam", "--variant", "1M"]).variant == "1M"
    for argv in (["--dataset", "beam", "--variant", "s"],
                 ["--dataset", "longmemeval", "--variant", "1M"]):
        with pytest.raises(SystemExit):
            xb.parse_args(argv)
    assert xb.dataset_files("beam", "500K")[0] == "beam_500K.json"


def test_answer_messages_put_memrys_lines_in_beams_prompt():
    [msg] = be.answer_messages("When did I start?", ["15 March 2024: started the app",
                                                     "the user likes Flask"])
    assert msg["role"] == "user"
    assert "CONTEXT:\n15 March 2024: started the app\nthe user likes Flask\n" in msg["content"]
    assert "QUESTION:\nWhen did I start?\n" in msg["content"]
    assert be.answer_messages.context_of(["a", "b"]) == "a\nb"


class FakeJudge:
    """Scores a rubric item 1 when the answer has its last word, else 0;
    calls two snippets one event when they match in lower case."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if messages[0]["role"] == "system":
            user = messages[1]["content"]
            first = user.split("First snippet: ", 1)[1].split(" \n", 1)[0]
            second = user.split("Second snippet: ", 1)[1].rstrip()
            same = first.strip().lower() == second.strip().lower()
            return ("YES" if same else "NO"), {"prompt_tokens": 50, "completion_tokens": 1}
        prompt = messages[0]["content"]
        item = prompt.split("RUBRIC CRITERION (what to check): ", 1)[1].split("\n", 1)[0]
        response = prompt.split("RESPONSE TO EVALUATE: ", 1)[1].split("\n\n## EVALUATION", 1)[0]
        score = 1.0 if item.split()[-1].lower() in response.lower() else 0.0
        return '```json\n{"score": %s, "reason": "r"}\n```' % score, \
            {"prompt_tokens": 1000, "completion_tokens": 60}


def test_grade_answer_means_the_rubric_items():
    judge = FakeJudge()
    out = be.grade_answer(judge, be.Budget("gpt-4.1-mini", None), "summarization",
                          ["has budget", "has login"], "We did the budget.")
    assert out["item_scores"] == [1.0, 0.0]
    assert out["llm_judge_score"] == 0.5
    assert "tau_norm" not in out


def test_event_order_is_scored_as_beam_scores_it():
    judge = FakeJudge()
    budget = be.Budget("gpt-4.1-mini", None)
    out = be.event_order(judge, budget, ["Budget", "Login", "Deploy"], "budget\ndeploy\nlogin")
    # every line matched: f1 1; one pair of three swapped: tau_b 1/3, normalised 2/3
    assert out["f1"] == 1.0
    assert out["tau_norm"] == pytest.approx(2 / 3)
    out = be.event_order(judge, budget, ["Budget", "Login", "Deploy"], "Budget\nLogin")
    assert out["precision"] == 1.0 and out["recall"] == pytest.approx(2 / 3)
    assert be.kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0
    assert be.kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert be.kendall_tau_b([1, 1], [1, 1]) is None


def test_parse_score_reads_the_judges_replies():
    assert be.parse_score('{"score": 0.5, "reason": "x"}') == 0.5
    assert be.parse_score('```json\n{"score": 1.0, "reason": "x"}\n```') == 1.0
    assert be.parse_score('Here: {"score": 0.0, "reason": "a {b}"}') == 0.0
    assert be.parse_score('{"score": [1.0], "reason": "x"') == 1.0
    assert be.parse_score("no json") is None


def test_budget_stops_past_its_cap():
    budget = be.Budget("gpt-4.1-mini", 0.001)
    budget.charge({"prompt_tokens": 1000, "completion_tokens": 0})
    with pytest.raises(be.CapReached):
        budget.charge({"prompt_tokens": 2000, "completion_tokens": 0})


def graded(conversation, ability, score, tau=None, items=None):
    g = {"qid": f"{conversation}/{ability}/{score}", "conversation": conversation,
         "ability": ability, "variant": "answer", "llm_judge_score": score,
         "item_scores": items or [score]}
    if tau is not None:
        g["tau_norm"] = tau
    return g


def test_tables_average_per_chat_then_over_chats():
    rows = [graded("a", "abstention", 1.0), graded("a", "abstention", 0.0),
            graded("b", "abstention", 1.0),
            graded("a", "event_ordering", 0.2, tau=1.0, items=[0.0, 0.5]),
            graded("b", "event_ordering", 1.0, tau=0.5, items=[1.0, 1.0])]
    t = be.tables(rows)["answer"]
    # chat a: 0.5, chat b: 1.0 -> 0.75, as BEAM's report averages
    assert t["official"]["abilities"]["abstention"] == 0.75
    assert t["official"]["abilities"]["event_ordering"] == 0.75
    assert t["rubric_average"]["abilities"]["event_ordering"] == pytest.approx(0.6)
    assert t["official"]["overall"] == 0.75
    assert t["strict"] == pytest.approx(3 / 5)
    assert t["pass"] == pytest.approx(4 / 5)


def results_file(path, rows, config=None):
    path.write_text(json.dumps({"config": config or {"answer_model": "gpt-4o-mini",
                                                      "compare_answer_model": "gpt-6-luna"},
                                "rows": rows}))
    return path


def test_grade_writes_scores_and_resumes(tmp_path):
    row = {"conversation": "beam-7", "qid": "beam-7/summarization/0",
           "beam_ability": "summarization", "rubric": ["has budget", "has login"],
           "prediction": "budget and login",
           "answers_compared": {"20": {"prediction": "budget only"}}}
    broken = {"conversation": "beam-7", "qid": "beam-7/summarization/1",
              "beam_ability": "summarization", "rubric": ["has budget"],
              "prediction": "", "answer_error": "400 Bad Request"}
    results = results_file(tmp_path / "r.json",
                           [row, broken, {"qid": "x", "conversation": "c"}])
    out = tmp_path / "scores.json"
    judge = FakeJudge()
    result = be.grade([str(results)], out, chat=judge)
    assert result["complete"] and result["judge_calls"] == 4
    assert result["failed_answers"] == ["beam-7/summarization/1 answer"]
    assert result["answer_model"] == "gpt-4o-mini"
    scores = {g["variant"]: g["llm_judge_score"] for g in result["graded"]}
    assert scores == {"answer": 1.0, "compared@20": 0.5}
    assert result["judge_dollars"] == pytest.approx(4 * (1000 * 0.4 + 60 * 1.6) / 1e6)
    again = FakeJudge()
    assert be.grade([str(results)], out, chat=again)["complete"]
    assert again.calls == []


def test_grade_stops_at_the_cap_and_keeps_what_it_graded(tmp_path):
    rows = [{"conversation": "beam-7", "qid": f"beam-7/abstention/{i}",
             "beam_ability": "abstention", "rubric": ["says none"], "prediction": "none"}
            for i in range(5)]
    out = tmp_path / "scores.json"
    result = be.grade([str(results_file(tmp_path / "r.json", rows))], out, chat=FakeJudge(),
                      max_dollars=0.0015, workers=1)
    assert not result["complete"] and result["stopped"]
    assert 1 <= len(result["graded"]) < 5


def test_judge_files_keep_the_calls_of_every_results_file_until_replied(tmp_path):
    def rows(qid, prediction):
        return [{"conversation": "beam-7", "qid": qid, "beam_ability": "summarization",
                 "rubric": ["has budget"], "prediction": prediction}]
    first = results_file(tmp_path / "a.json", rows("beam-7/summarization/0", "budget"))
    second = results_file(tmp_path / "b.json", rows("beam-7/summarization/1", "login"))
    folder = tmp_path / "judge"
    args = ["--out", str(tmp_path / "s.json"), "--judge-files", str(folder)]
    assert be.main(["grade", str(first), *args]) == 4
    assert be.main(["grade", str(second), *args]) == 4
    asked = json.loads((folder / "asked.json").read_text())
    assert sorted(asked) == ["beam-7/summarization/0", "beam-7/summarization/1"]
    key = next(iter(asked["beam-7/summarization/0"]))
    (folder / "replies").mkdir()
    (folder / "replies" / "x.json").write_text(json.dumps({key: {"score": 1.0, "reason": "ok"}}))
    assert be.main(["grade", str(second), *args]) == 4
    assert sorted(json.loads((folder / "asked.json").read_text())) == ["beam-7/summarization/1"]
    assert be.main(["grade", str(first), *args]) == 0


def test_fetch_pages_and_refuses_shortened_cells(tmp_path):
    pages = []

    def get(params):
        pages.append(params["offset"])
        rows = [{"row_idx": i, "row": {"conversation_id": str(i)}, "truncated_cells": []}
                for i in range(params["offset"], min(params["offset"] + params["length"], 3))]
        return {"num_rows_total": 3, "rows": rows}

    info = be.fetch("100K", tmp_path / "beam_100K.json", page=2, get=get)
    assert pages == [0, 2] and info["rows"] == 3
    assert [r["conversation_id"] for r in json.loads((tmp_path / "beam_100K.json").read_text())] \
        == ["0", "1", "2"]

    def short(params):
        return {"num_rows_total": 1, "rows": [{"row_idx": 0, "row": {},
                                               "truncated_cells": ["chat"]}]}

    with pytest.raises(RuntimeError):
        be.fetch("100K", tmp_path / "x.json", get=short)


def test_openai_chat_drops_a_temperature_the_model_refuses():
    import httpx

    from evals.mem0_judge import OpenAIChat

    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unsupported value: 'temperature' does not support 0 with this "
                           "model.", "param": "temperature", "code": "unsupported_value"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "blue"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    # a model that takes temperatures by its name, yet refuses this one
    llm = OpenAIChat("gpt-4o-mini-later", api_key="k")
    llm._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert llm.chat([{"role": "user", "content": "colour?"}]) == "blue"
    assert llm.chat([{"role": "user", "content": "again?"}]) == "blue"
    assert ["temperature" in b for b in bodies] == [True, False, False]
    assert llm.temperature is None and llm.last_usage()["prompt_tokens"] == 5


def test_openai_chat_keeps_temperature_zero_where_it_is_taken():
    import httpx

    from evals.mem0_judge import OpenAIChat

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAIChat("gpt-4o-mini", api_key="k")
    llm._client = httpx.Client(transport=httpx.MockTransport(handler))
    llm.chat([{"role": "user", "content": "hi"}])
    assert bodies[0]["temperature"] == 0.0


def kept_row(qid="beam-7/summarization/0", lines=("15 March 2024: started the app",)):
    return {"conversation": "beam-7", "qid": qid, "beam_ability": "summarization",
            "rubric": ["has budget"], "question": "What did I start?", "answer": "the app",
            "reference_date": "2024-04-25T00:00:00+00:00", "answer_k": 20,
            "prediction": "old answer",
            "answers": {"20": {"prediction": "old answer",
                               **({"context_lines": list(lines)} if lines else {})}}}


def test_prompts_build_each_answering_call():
    lines = ["15 March 2024: started the app"]
    [beam] = be.PROMPTS["beam"]("What did I start?", lines, "2024-04-25T00:00:00+00:00")
    assert beam["role"] == "user" and "CONTEXT:\n15 March 2024: started the app" in beam["content"]
    system, user = be.PROMPTS["abilities"]("What did I start?", lines, "2024-04-25T00:00:00+00:00")
    assert system["content"] == be.ABILITIES_SYSTEM
    assert user["content"].startswith("What did I start?\n\nToday is 25 April 2024.\n")
    assert user["content"].endswith("15 March 2024: started the app")
    system, user = be.PROMPTS["memry"]("What did I start?", lines, None)
    assert system["content"] == xb.ANSWER_SYSTEM
    assert user["content"] == ("Memories:\n- 15 March 2024: started the app\n\n"
                               "Question: What did I start?\nShort answer:")


def test_the_context_of_an_answer_that_read_every_member_is_found():
    # --all-members: the headline answer is filed under the k asked for (20) and read 58
    row = kept_row(lines=("a line",))
    row["answer_k"] = 58
    row["answers"] = {"5": {"k": 5, "context_lines": ["five"]},
                      "20": {"k": 58, "context_lines": ["a line", "another"]}}
    assert be.context_lines_of(row) == ["a line", "another"]
    assert be.context_lines_of(kept_row()) == ["15 March 2024: started the app"]
    assert be.context_lines_of(kept_row(lines=None)) is None


def test_answer_writes_a_results_file_grade_reads(tmp_path):
    rows = [kept_row(), kept_row("beam-7/summarization/1", lines=None)]
    src = results_file(tmp_path / "r.json", rows)
    seen = []

    def chat(messages):
        seen.append(messages)
        return " the app ", {"prompt_tokens": 1000, "completion_tokens": 100}

    out = tmp_path / "opus.json"
    result = be.answer([str(src)], out, prompt="abilities", model="anthropic/claude-opus-5.5",
                       provider="openrouter", chat=chat, workers=1)
    assert result["config"]["answer_prompt"] == "abilities" and result["failed"] == 1
    assert result["dollars"] == pytest.approx((1000 * 4 + 100 * 20) / 1e6)
    good, bad = sorted(result["rows"], key=lambda r: r["qid"])
    assert good["prediction"] == "the app" and good["rubric"] == ["has budget"]
    assert "keep-context" in bad["answer_error"]
    assert seen[0][0]["content"] == be.ABILITIES_SYSTEM
    rows_back, config = be.read_rows([str(out)])
    assert config["answer_model"] == "anthropic/claude-opus-5.5"
    assert [v for r in rows_back for v, _ in be.answers_of(r)] == ["answer"]
    # a second pass keeps the good answer and asks only for the failed one
    seen.clear()
    be.answer([str(src)], out, prompt="abilities", model="anthropic/claude-opus-5.5",
              chat=chat, workers=1)
    assert seen == []


def test_answer_stops_at_its_cap(tmp_path):
    rows = [kept_row(f"beam-7/summarization/{i}") for i in range(5)]
    src = results_file(tmp_path / "r.json", rows)

    def chat(messages):
        return "x", {"prompt_tokens": 1000, "completion_tokens": 0}

    result = be.answer([str(src)], tmp_path / "o.json", prompt="beam", model="gpt-4o-mini",
                       chat=chat, workers=1, max_dollars=0.0004)
    assert result["stopped"] and not result["complete"] and len(result["rows"]) == 2
