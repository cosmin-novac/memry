"""The metadata a benchmark run writes beside its results (``<out>.meta.json``,
evals/external_benchmarks.py ``run_meta``) and prints first. A tiny scripted
LoCoMo run (tests/fixtures/locomo_mini.json, verbatim, hash embedder) is
answered by two stub models whose replies come from an httpx mock transport,
so the usage ledger counts their calls and tokens as it counts real ones, and
no call leaves the process."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import api_usage  # noqa: E402
from evals import external_benchmarks as xb  # noqa: E402
from evals import mem0_judge  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
LOCOMO = FIXTURES / "locomo_mini.json"

#: input, cached and output tokens each stub reply reports
USAGE = {"gpt-4o-mini": (1000, 200, 10), "gpt-6-luna": (800, 0, 50)}
#: prices of the test, USD per million tokens; gpt-6-luna is priced through an alias
PRICES = {"models": {"gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60,
                                     "retrieved": "2026-09-29"},
                     "luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50,
                              "retrieved": "2026-09-30"}},
          "aliases": {"gpt-6-luna": "luna"}}


@pytest.fixture
def no_models(monkeypatch, tmp_path):
    """Config.load() finds no model: no keys, no config file."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "TYPESAFE_API_KEY",
                 "MEMRY_LLM_PROVIDER", "MEMRY_LLM_API_KEY", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "no-config.json"))


def _gold() -> dict[str, str]:
    return {q.question: str(q.answer) for c in xb.load_locomo(LOCOMO) for q in c.questions}


@pytest.fixture
def stub_models(monkeypatch):
    """--answer-model and --compare-answer-model answer through a mock
    transport: gpt-4o-mini with the gold answer, gpt-6-luna with nothing
    right. Every request is kept."""
    gold, sent = _gold(), []

    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((request.url.host, body["model"]))
        text = json.dumps(body["messages"])
        asked = next(q for q in gold if json.dumps(q)[1:-1] in text)
        tokens_in, cached, tokens_out = USAGE[body["model"]]
        return httpx.Response(200, json={
            "model": f"{body['model']}-2026-01-01",
            "choices": [{"message": {"content": gold[asked] if body["model"] == "gpt-4o-mini"
                                     else "I could not say."}}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                      "prompt_tokens_details": {"cached_tokens": cached}}})

    class MockedChat(mem0_judge.OpenAIChat):
        def __init__(self, model: str, **kwargs) -> None:
            super().__init__(model, api_key="no-key", **kwargs)
            self._client.close()
            self._client = httpx.Client(transport=httpx.MockTransport(reply))

    monkeypatch.setattr(mem0_judge, "OpenAIChat", MockedChat)
    return sent


def _run(tmp_path, monkeypatch, *extra: str) -> tuple[int, Path]:
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps(PRICES))
    out = tmp_path / "run.json"
    code = xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--categories", "1,2,3,4",
                    "--k", "3", "--judge-runs", "2", "--answer-model", "gpt-4o-mini",
                    "--compare-answer-model", "gpt-6-luna",
                    "--answer-prompt", "evals.mem0_judge:answer_messages",
                    "--usage-db", str(tmp_path / "usage.sqlite"), "--prices", str(prices),
                    "--out", str(out), *extra])
    return code, out


def test_a_run_writes_its_metadata_beside_the_results_and_prints_it_first(
        tmp_path, monkeypatch, capsys, no_models, stub_models):
    code, out = _run(tmp_path, monkeypatch)
    assert code == 0
    meta_file = tmp_path / "run.meta.json"
    meta = json.loads(meta_file.read_text())
    result = json.loads(out.read_text())
    assert result["meta_file"] == str(meta_file)
    # no call left the process: the 12 answers went to the mock, nothing else was sent
    assert sorted(set(stub_models)) == [("api.openai.com", "gpt-4o-mini"),
                                        ("api.openai.com", "gpt-6-luna")]
    assert len(stub_models) == 12

    assert set(meta) == {"dataset", "file", "file_sha256", "questions", "memry", "config",
                         "command", "time", "prices", "usage", "results", "out"}
    assert meta["dataset"] == "locomo" and meta["file"] == str(LOCOMO)
    assert meta["file_sha256"] == hashlib.sha256(LOCOMO.read_bytes()).hexdigest()
    assert meta["out"] == str(out)
    assert meta["command"][3:] == ["--dataset", "locomo", "--file", "locomo_mini.json",
                                   "--categories", "1,2,3,4", "--k", "3", "--judge-runs", "2",
                                   "--answer-model", "gpt-4o-mini", "--compare-answer-model",
                                   "gpt-6-luna", "--answer-prompt",
                                   "evals.mem0_judge:answer_messages", "--usage-db",
                                   str(tmp_path / "usage.sqlite"), "--prices",
                                   str(tmp_path / "prices.json"), "--out", str(out)]

    questions = meta["questions"]
    assert (questions["asked"], questions["scored"], questions["conversations"]) == (6, 6, 2)
    assert questions["selection"] == {
        "limit": None, "sample": None, "seed": None, "categories": ["1", "2", "3", "4"],
        "questions_per_conversation": None, "conversation": None, "variant": None,
        "conversation_ids": ["conv-mini-1", "conv-mini-2"]}

    assert set(meta["memry"]) == {"commit", "branch", "changed"}
    assert meta["memry"]["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", meta["memry"]["commit"])

    config = meta["config"]
    assert set(config) == {
        "ingest", "extract_unit", "embedder", "text_model", "text_effort", "decider",
        "decider_in_stores", "decider_model", "search_deciders", "k", "ks", "depth",
        "evidence_tokens", "evidence_tokens_in_stores", "compare_evidence_tokens",
        "answer_models", "answer_prompt", "judge", "judge_runs", "audit", "full_context",
        "context", "when", "descriptions", "question_keys", "question_keys_in_stores",
        "entity_questions_in_stores", "workers", "jobs", "max_calls"}
    assert (config["ingest"], config["embedder"], config["text_model"]) == (
        "verbatim", "hash:v1-256", None)
    assert (config["decider"], config["k"], config["ks"], config["judge_runs"]) == (
        "config", 3, [3], 2)
    assert config["answer_models"] == ["gpt-4o-mini", "gpt-6-luna"]
    assert config["answer_prompt"] == "evals.mem0_judge:answer_messages"
    assert config["judge"] == "evals.external_benchmarks:containment_judge"
    assert config["question_keys"] == "config" and config["audit"] is None
    assert config["evidence_tokens_in_stores"] == 600
    # the results file names the store's decider option, not the pass's provider
    assert result["config"]["decider"] == "config"

    assert set(meta["time"]) == {"started", "finished", "wall_seconds"}
    assert meta["time"]["started"] <= meta["time"]["finished"]
    assert meta["time"]["wall_seconds"] >= 0
    assert meta["prices"]["file"] == str(tmp_path / "prices.json")
    assert meta["prices"]["retrieved"] == {"gpt-4o-mini": "2026-09-29", "luna": "2026-09-30"}

    results = meta["results"]
    assert set(results) == {"pass", "answers", "recall@5", "recall@10", "recall@20", "mrr",
                            "search_ms_p50", "search_ms_p95", "context_tokens_mean",
                            "complete", "stopped"}
    first, compared = results["answers"]
    assert (first["answers"], first["model"], first["k"], first["n"], first["judge"]) == (
        "answer", "gpt-4o-mini", 3, 6, 1.0)
    assert (compared["answers"], compared["model"], compared["judge"]) == (
        "answer:compared", "gpt-6-luna", 0.0)
    assert first["judge_runs"] == [1.0, 1.0] and first["evidence_tokens"] == 600
    overall = result["tables"]["overall"]
    for key in ("recall@5", "recall@10", "recall@20", "mrr"):
        assert results[key] == overall[key]
    assert results["context_tokens_mean"] == overall["context_tokens"]
    times = sorted(r["search_ms"] for r in result["rows"])
    assert times[0] <= results["search_ms_p50"] <= results["search_ms_p95"] <= times[-1]
    assert results["complete"] is True

    # the dollars: the ledger's tokens at the test's prices
    usage = meta["usage"]
    assert set(usage) == {"ledger", "calls", "first_call", "last_call", "span_seconds", "usd",
                          "unpriced", "by_model", "by_stage", "served_models"}
    assert usage["calls"] == {"chat": 12, "all": 12} and usage["unpriced"] == []
    models = {m["model"]: m for m in usage["by_model"]}
    mini, luna = models["gpt-4o-mini"], models["gpt-6-luna"]
    assert (mini["calls"], mini["input_tokens"], mini["cached_tokens"], mini["output_tokens"]) == (
        6, 6000, 1200, 60)
    # 6 x (800 x 0.15 + 200 x 0.075 + 10 x 0.60) / 1e6, and 6 x (800 x 0.10 + 50 x 0.50) / 1e6
    assert mini["usd"] == pytest.approx(0.000846) and mini["priced_as"] == "gpt-4o-mini"
    assert luna["usd"] == pytest.approx(0.00063) and luna["priced_as"] == "luna"
    assert usage["usd"] == {"openai": pytest.approx(0.001476), "total": pytest.approx(0.001476)}
    assert {(s["stage"], s["model"], s["calls"]) for s in usage["by_stage"]} == {
        ("answer", "gpt-4o-mini", 6), ("answer:compared", "gpt-6-luna", 6)}
    assert {(s["model"], s["served_model"], s["calls"]) for s in usage["served_models"]} == {
        ("gpt-4o-mini", "gpt-4o-mini-2026-01-01", 6), ("gpt-6-luna", "gpt-6-luna-2026-01-01", 6)}

    # the same block opens the printed results
    printed = capsys.readouterr().out.splitlines()
    lines = xb.meta_lines(meta)
    assert printed[:len(lines)] == lines
    assert lines[0].startswith("## run: locomo, 6 questions scored of 6 asked")
    assert any("gpt-4o-mini: 6 calls" in line and "$0.0008" in line for line in lines)
    assert "| category | name | n |" in "\n".join(printed[len(lines):])


def test_a_missing_prices_file_stops_the_run_before_any_call(tmp_path, monkeypatch, capsys,
                                                             no_models, stub_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--answer-model",
                    "gpt-4o-mini", "--prices", str(tmp_path / "none.json"),
                    "--out", str(tmp_path / "r.json")]) == 2
    assert "--prices" in capsys.readouterr().err
    assert stub_models == [] and not (tmp_path / "r.json").exists()


def test_the_repository_prices_give_the_first_locomo_run_its_published_cost():
    """phd data/locomo/usage.json, the first full LoCoMo run (2026-09-29): the
    tokens per model, and $3.12 in all. Jev is priced on input tokens only,
    a cached token at the cached price."""
    prices = xb.load_prices(xb.PRICES)
    tokens = {"gpt-4o-mini": (11_723_500, 453_632, 115_505, 1.79381),
              "gpt-6-luna": (1_097_084, 131_563, 823_029, 0.50938),
              "text-embedding-3-small": (147_811, 0, 0, 0.00296),
              "jev-latest": (19_368_946, 0, 2_025_554, 0.81350)}
    total = 0.0
    for model, (tokens_in, cached, tokens_out, published) in tokens.items():
        name, price = xb.price_of(model, prices)
        assert price is not None, model
        cost = xb.dollars(price, tokens_in, cached, tokens_out)
        assert cost == pytest.approx(published, abs=6e-6), model
        total += cost
    assert total == pytest.approx(3.1196, abs=1e-4) and round(total, 2) == 3.12
    assert xb.price_of("jev-latest", prices)[0] == "jev"
    assert xb.price_of("gpt-4o-2024-08-06", prices)[0] == "gpt-4o"
    assert all(entry.get("retrieved") for entry in prices["models"].values())
    assert xb.price_of("a-model-nobody-priced", prices) == (None, None)


def test_the_ledger_names_the_served_model_and_an_older_ledger_gets_the_column(tmp_path):
    ledger = tmp_path / "old.sqlite"
    db = sqlite3.connect(ledger)
    db.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY, label TEXT, grp TEXT, host TEXT, "
               "path TEXT, model TEXT, stage TEXT, started REAL, seconds REAL, status INTEGER, "
               "ok INTEGER, input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER, "
               "reasoning_tokens INTEGER, has_usage INTEGER, items INTEGER, request_chars INTEGER, "
               "response_chars INTEGER, error TEXT)")
    db.execute("INSERT INTO calls (grp, host, model, stage, started, seconds, ok, input_tokens, "
               "output_tokens) VALUES ('jev', 'api.typesafe.ai', 'jev-latest', 'search', 100.0, "
               "0.5, 1, 1000000, 50000), ('chat', 'llm.example', 'mystery', 'answer', 101.0, "
               "1.0, 0, 10, 2)")
    db.commit()
    db.close()
    before = xb.ledger_usage(ledger, xb.load_prices(xb.PRICES))
    assert before["usd"] == {"jev": pytest.approx(0.042), "total": pytest.approx(0.042)}
    assert before["unpriced"] == [{"model": "mystery", "calls": 1}]
    assert before["span_seconds"] == 2.0 and before["calls"] == {"jev": 1, "chat": 1, "all": 2}
    assert {r["served_model"] for r in before["served_models"]} == {None}

    def jev(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {},
                                         "usage": {"input_tokens": 7, "output_tokens": 1}})

    client = httpx.Client(transport=httpx.MockTransport(jev))
    with api_usage.UsageMeter(str(ledger)):
        client.post("https://api.typesafe.ai/v1/systemone", json={"model": "jev-latest"})
    client.close()
    after = xb.ledger_usage(ledger, xb.load_prices(xb.PRICES))
    assert {(r["model"], r["served_model"], r["calls"]) for r in after["served_models"]} == {
        ("jev-latest", None, 1), ("jev-latest", "jev-1.13.0", 1), ("mystery", None, 1)}


def test_the_metadata_file_sits_beside_the_results():
    assert xb.meta_path("runs/locomo_full.json") == Path("runs/locomo_full.meta.json")
    assert xb.meta_path("runs/locomo.out") == Path("runs/locomo.out.meta.json")
