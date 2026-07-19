from __future__ import annotations

import json

import pytest

from memry.cli import main

pytestmark = pytest.mark.usefixtures("cli_env")


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)


def run(capsys, *argv: str):
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_add_search_list_roundtrip(capsys):
    code, out = run(capsys, "add", "Ada joined ASML in Amsterdam", "-u", "ada")
    assert code == 0
    assert json.loads(out)["actions"][0]["event"] == "ADD"

    code, out = run(capsys, "search", "where does ada work", "-u", "ada")
    assert code == 0
    results = json.loads(out)
    assert results and "ASML" in results[0]["content"]

    code, out = run(capsys, "list", "-u", "ada")
    assert len(json.loads(out)) == 1

    code, out = run(capsys, "context", "commute planning", "-u", "ada")
    assert "ASML" in out


def test_get_history_delete(capsys):
    _, out = run(capsys, "add", "temp note", "-u", "ada")
    memory_id = json.loads(out)["actions"][0]["memory_id"]

    code, out = run(capsys, "get", memory_id)
    assert json.loads(out)["content"] == "temp note"

    code, out = run(capsys, "delete", memory_id)
    assert json.loads(out)["deleted"] is True

    code, out = run(capsys, "history", memory_id)
    assert [e["event"] for e in json.loads(out)] == ["ADD", "DELETE"]

    code, out = run(capsys, "list", "-u", "ada")
    assert json.loads(out) == []


def test_get_missing_returns_error(capsys):
    code, _ = run(capsys, "get", "nope")
    assert code == 1


def test_stats_and_config(capsys):
    _, out = run(capsys, "stats")
    assert json.loads(out)["backend"] == "local"

    _, out = run(capsys, "config")
    cfg = json.loads(out)
    assert cfg["llm"]["provider"] == "none"
    assert cfg["embedding"]["provider"] == "hash"


def test_export_import_roundtrip(capsys, tmp_path):
    run(capsys, "add", "fact one", "-u", "ada")
    run(capsys, "add", "fact two", "-u", "ada")
    _, out = run(capsys, "export", "-u", "ada")
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2

    dump = tmp_path / "dump.jsonl"
    dump.write_text("\n".join(lines), encoding="utf-8")

    # import into a different user scope via records' user_id (kept from export)
    _, out = run(capsys, "import", str(dump))
    assert json.loads(out)["imported"] == 2


def test_reindex_and_sweep(capsys):
    run(capsys, "add", "sweep me", "-u", "ada")
    _, out = run(capsys, "reindex")
    assert json.loads(out)["reindexed"] >= 1
    _, out = run(capsys, "sweep", "--threshold", "0.0")
    assert json.loads(out)["count"] == 0  # fresh memories survive a 0-threshold sweep


def test_eval_command(capsys):
    code, out = run(capsys, "eval", "--dataset", "evals/datasets/synthetic_v1.jsonl",
                    "-k", "5", "--json")
    assert code == 0
    report = json.loads(out)
    assert report["questions"] > 0
    assert report["recall_at_k"] >= 0.6


def test_no_command_shows_help(capsys):
    code, _ = run(capsys)
    assert code == 1
