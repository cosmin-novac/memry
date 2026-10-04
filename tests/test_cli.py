from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

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
    backup = json.loads(out)
    assert backup["format"] == "memry-backup"
    assert len(backup["tables"]["memories"]) == 2

    dump = tmp_path / "backup.json"
    dump.write_text(json.dumps(backup), encoding="utf-8")

    # Re-importing the same backup preserves identities and is idempotent.
    _, out = run(capsys, "import", str(dump))
    result = json.loads(out)
    assert result["inserted"] == 0
    assert result["unchanged"] > 0


def test_reindex_and_sweep(capsys):
    run(capsys, "add", "sweep me", "-u", "ada")
    _, out = run(capsys, "reindex")
    assert json.loads(out)["reindexed"] >= 1
    _, out = run(capsys, "sweep", "--threshold", "0.0")
    assert json.loads(out)["count"] == 0  # fresh memories survive a 0-threshold sweep


def test_backfill_property_vectors_embeds_what_is_missing_in_every_namespace(capsys):
    """The command fills the property vectors the linked search reads, one
    namespace at a time or all of them, and a second run finds nothing to do."""
    from memry.config import Config
    from memry.models import Entity, EntityMention, Memory, Scope
    from memry.store import MemoryStore, _text_hash

    store = MemoryStore(Config.load())
    saved = {}
    for user, name, text in (("ada", "Quillon", "Quillon runs on Linux"),
                             ("bo", "Tessel Works", "Tessel Works meets on Mondays")):
        entity = store.backend.insert_entity(Entity(name=name, normalized=name.lower(),
                                                    user_id=user))
        memory = store.backend.insert_memory(Memory(content=text, user_id=user))
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=name))
        saved[user] = memory.id
    store.backend.insert_memory(Memory(content="The sprint review moved to Friday",
                                       user_id="ada"))
    label = store._property_label()
    store.close()

    code, out = run(capsys, "backfill-property-vectors", "-u", "ada")
    assert code == 0 and json.loads(out) == [{"user": "ada", "embedded": 1}]
    code, out = run(capsys, "backfill-property-vectors")
    assert sorted(json.loads(out), key=lambda row: row["user"]) == [
        {"user": "ada", "embedded": 0}, {"user": "bo", "embedded": 1}]
    code, out = run(capsys, "backfill-property-vectors")
    assert all(row["embedded"] == 0 for row in json.loads(out))

    store = MemoryStore(Config.load())
    assert store.backend.property_vector_hashes(list(saved.values())) == {
        saved["ada"]: (_text_hash("it runs on Linux"), label),
        saved["bo"]: (_text_hash("it meets on Mondays"), label)}
    assert len(store.backend.list_memories(Scope(user_id="ada"))) == 2
    store.close()


class _SentenceSplitter(FakeLLM):
    """A text model for split-memories: a memory splits at its sentences,
    and the audit misses nothing. ``refuse`` makes any call fail, for a run
    that must ask nothing."""

    refuse = False

    def complete(self, system, user, *, json_schema=None):
        from memry.intelligence.split import SPLIT_SYSTEM, sentences

        if self.refuse:
            raise AssertionError("the text model was asked")
        self.calls.append(user)
        if system == SPLIT_SYSTEM:
            text = user.split("Memory:\n", 1)[1].split("\n\nSplit it", 1)[0]
            return json.dumps({"facts": sentences(text)})
        return json.dumps({"missing": []})


def _split_cli(monkeypatch, llm):
    """The CLI's store with ``llm`` and a small embedder; a new one each run,
    as the CLI closes it."""
    from memry.config import Config
    from memry.providers.embeddings import HashEmbedder
    from memry.store import MemoryStore

    def make():
        return MemoryStore(Config.load(), llm=llm, embedder=HashEmbedder(64))

    monkeypatch.setattr("memry.cli._store", make)
    return make


def test_maintenance_commands_go_through_each_namespace_once(capsys, monkeypatch):
    """No user means every user's memories: the pass for the memories
    without one took in every namespace, and each named one was done again
    after it (in production 1446 memories asked, then the 1440 of "default"
    once more)."""
    from memry.models import Memory

    llm = _SentenceSplitter()
    store = _split_cli(monkeypatch, llm)()
    for user in (None, "a", "b"):
        memory = store.backend.insert_memory(Memory(
            content=f"Project {user} uses Go. Project {user} runs on one VPS.", user_id=user))
        store.backend.set_memory_timestamp(memory.id, "2030-01-01T00:00:00+00:00")
    store.close()

    code, out = run(capsys, "repair-dates")
    assert code == 0 and [row["fixed"] for row in json.loads(out)] == [1, 1, 1]
    code, out = run(capsys, "split-memories", "--dry-run", "--json")
    assert code == 0
    assert sorted((r["user"] or "", r["in_use"], r["split"]) for r in json.loads(out)) == [
        ("", 1, 1), ("a", 1, 1), ("b", 1, 1)]
    assert len(llm.calls) == 6  # a split and an audit for each memory, once


def test_split_memories_makes_exactly_the_reviewed_plan(capsys, monkeypatch, tmp_path):
    """--dry-run --plan-out writes the proposed splits; --plan-in makes those,
    asking the model nothing, each in its own namespace, and undo works on
    them."""
    from memry.models import Memory

    llm = _SentenceSplitter()
    store = _split_cli(monkeypatch, llm)()
    texts = {None: "Project Kite uses Go. Project Kite runs on one VPS.",
             "a": "Ada likes tea. Ada dislikes coffee."}
    ids = {user: store.backend.insert_memory(Memory(content=text, user_id=user)).id
           for user, text in texts.items()}
    store.close()
    path = tmp_path / "plan.json"

    code, _ = run(capsys, "split-memories", "--plan-out", str(path))
    assert code == 1 and not path.exists()  # a plan comes from a dry run
    code, out = run(capsys, "split-memories", "--dry-run", "--plan-out", str(path))
    assert code == 0 and "would be split" in out
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["format"] == "memry-split-plan" and plan["version"] == 2
    assert sorted((e["user"] or "", e["memory_id"]) for e in plan["splits"]) == [
        ("", ids[None]), ("a", ids["a"])]

    llm.refuse = True
    code, out = run(capsys, "split-memories", "--plan-in", str(path), "--json")
    assert code == 0
    reports = json.loads(out)
    assert sorted((r["user"] or "", r["split"], r["facts"], r["stale"]) for r in reports) == [
        ("", 1, 2, 0), ("a", 1, 2, 0)]
    store = _split_cli(monkeypatch, llm)()
    assert sorted(m.content for m in store.get_all(limit=100)) == sorted(
        ["Project Kite uses Go.", "Project Kite runs on one VPS.",
         "Ada likes tea.", "Ada dislikes coffee."])
    store.close()
    # applied again, the memories are out of use: skipped, nothing written twice
    code, out = run(capsys, "split-memories", "--plan-in", str(path))
    assert code == 0 and "0 split into 0 facts as planned, 1 skipped" in out
    code, out = run(capsys, "split-memories", "--undo", ids["a"])
    assert code == 0 and json.loads(out)["undone"] is True


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


def test_direct_http_mcp_launcher_is_removed():
    with pytest.raises(SystemExit):
        main(["mcp", "--transport", "http"])
