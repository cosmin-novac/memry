"""The search log (``retrieval.search_log``) and keys from traffic
(``retrieval.traffic_keys``).

With the log on, every search is kept as one row: when, the namespace and
run, the query, whether it was about a known entity (its seeds), the mode,
how many results and how long it took. Upkeep deletes rows older than 90
days, ``memry search-stats`` counts them, and neither an export nor a
snapshot contains them unless asked. With traffic keys on, a save after a
search gives the saved memory that the search would now rank in its first
20 the query as a question key (source "traffic")."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from memry import store as store_module
from memry.cli import format_search_stats, main
from memry.config import Config
from memry.models import Entity, EntityMention, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.snapshot import take_snapshot
from memry.store import SEARCH_LOG_DAYS, MemoryStore


def _store(search_log: bool = True, traffic_keys: bool = False, db_path: str = ":memory:",
           question_keys: bool = True) -> MemoryStore:
    cfg = Config(db_path=db_path)
    cfg.retrieval.search_log = search_log
    cfg.retrieval.traffic_keys = traffic_keys
    cfg.retrieval.question_keys = question_keys
    return MemoryStore(cfg, llm=NoneLLM(), embedder=HashEmbedder(64))


@pytest.fixture
def store():
    s = _store()
    yield s
    s.close()


def _remember(store, text, entities=(), user_id="ada"):
    memory = store.backend.insert_memory(
        Memory(content=text, user_id=user_id, embedding_model=store.embedder.model_id),
        embedding=store.embedder.embed([text])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def _rows(store, **kw):
    return store.backend.search_log_rows(**kw)


def _ago(**delta) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat(timespec="seconds")


def _keep(store, query, at, user_id="ada", run_id=None, seeds=0, mode=None, results=1,
          judged=0, best_judged=None):
    return store.backend.log_search({
        "at": at, "user_id": user_id, "agent_id": None, "run_id": run_id, "query": query,
        "mode": mode or ("linked" if seeds else "text"), "seeds": seeds, "first_person": 0,
        "filtered": 0, "judged": judged, "best_judged": best_judged, "results": results,
        "latency_ms": 5.0})


def _keys(store, memory_id):
    return [(q["text"], q["source"]) for q in store.backend.questions_of([memory_id])
            .get(memory_id, [])]


# ------------------------------------------------------------------ the log

def test_off_by_default_and_nothing_is_kept():
    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        assert s.config.retrieval.search_log is False
        assert s.config.retrieval.traffic_keys is False
        _remember(s, "Ada lives in Lisbon")
        s.search("where does ada live", user_id="ada")
        assert _rows(s) == []
    finally:
        s.close()


def test_a_search_is_kept_as_one_row(store):
    _remember(store, "Ada lives in Lisbon")
    _remember(store, "The boiler was serviced in May")
    found = store.search("  where is the   boiler ", user_id="ada", limit=1)
    store.search("boiler", user_id="ada", agent_id="claude", run_id="r1")
    rows = _rows(store)
    assert len(rows) == 2
    assert (rows[1]["run_id"], rows[1]["agent_id"], rows[1]["results"]) == ("r1", "claude", 0)
    row = rows[0]
    assert row["query"] == "where is the boiler"  # spacing folded
    assert (row["user_id"], row["run_id"], row["agent_id"]) == ("ada", None, None)
    assert row["mode"] == "text" and row["seeds"] == 0 and row["first_person"] == 0
    assert row["results"] == len(found) == 1
    assert row["latency_ms"] >= 0 and row["judged"] == 0 and row["best_judged"] is None
    assert row["filtered"] == 0 and row["keyed"] == 0
    datetime.fromisoformat(row["at"])


def test_a_search_without_a_namespace_is_kept_under_the_default_one(store):
    store.search("anything", user_id=None)
    assert _rows(store)[0]["user_id"] == store.config.default_user_id


def test_a_search_about_a_known_entity_is_kept_with_its_seeds(store):
    harlow = store.backend.insert_entity(Entity(
        name="Harlow", normalized="harlow", user_id="ada", entity_type="person"))
    _remember(store, "Harlow paid the rent in cash", [harlow])
    _remember(store, "Harlow moved to Ghent", [harlow])
    results = store.search("Where did Harlow move?", user_id="ada")
    assert "about" in results[0].signals  # the linked search ran
    row = _rows(store)[0]
    assert row["mode"] == "linked" and row["seeds"] == 1


def test_browsing_and_filters_are_marked(store):
    _remember(store, "Ada lives in Lisbon")
    store.search("", user_id="ada")
    store.search("lisbon", user_id="ada", since="2000-01-01")
    browse, filtered = _rows(store)
    assert browse["mode"] == "browse" and browse["query"] == ""
    assert filtered["filtered"] == 1 and filtered["mode"] == "text"


def test_the_judges_best_score_is_kept(store):
    _remember(store, "Ada lives in Lisbon")
    _remember(store, "Ada likes figs")

    class Judge:
        available = True
        reranks_by_default = True

        def decide(self, state, questions):
            raise AssertionError("not used")

        def close(self):
            return None

    store.decider = Judge()
    store.config.retrieval.relational_relevance = "jev"
    store._judged_relevance = lambda question, items, meta=False: (
        {mid: 0.3 for mid, _ in items}, 1.0, 0.0)
    store.search("where does ada live", user_id="ada")
    row = _rows(store)[0]
    assert row["judged"] == 1 and row["best_judged"] == pytest.approx(0.3)


def test_a_failing_log_never_fails_the_search(store, monkeypatch):
    _remember(store, "Ada lives in Lisbon")

    def broken(row):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store.backend, "log_search", broken)
    assert store.search("lisbon", user_id="ada")


def test_the_environment_turns_the_log_and_the_keys_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("MEMRY_SEARCH_LOG", "1")
    monkeypatch.setenv("MEMRY_TRAFFIC_KEYS", "true")
    monkeypatch.setenv("MEMRY_QUESTION_KEYS", "on")
    cfg = Config.load()
    assert cfg.retrieval.search_log and cfg.retrieval.traffic_keys and cfg.retrieval.question_keys


# ------------------------------------------------------- retention, deletion

def test_upkeep_deletes_searches_older_than_90_days(store):
    old = _keep(store, "old question", _ago(days=SEARCH_LOG_DAYS + 1))
    other = _keep(store, "old question of bea", _ago(days=SEARCH_LOG_DAYS + 2), user_id="bea")
    kept = _keep(store, "recent question", _ago(days=SEARCH_LOG_DAYS - 1))
    store.set_upkeep_paused(True)  # retention holds even while upkeep is paused
    ran = store.run_upkeep_cycle(user_id="ada")
    assert ran["search_log"] == {"deleted": 2}
    assert [r["id"] for r in _rows(store)] == [kept]
    assert old != other


def test_deleting_a_namespace_deletes_its_searches(store):
    _remember(store, "Ada lives in Lisbon")
    store.search("lisbon", user_id="ada")
    store.search("lisbon", user_id="bea")
    store.delete_all(user_id="ada", hard=True)
    assert [r["user_id"] for r in _rows(store)] == ["bea"]


# ------------------------------------------------------------- the counts

def test_search_stats_counts_per_namespace_and_the_common_unseeded_queries(store):
    now = _ago(minutes=1)
    for query in ["What is left to do?", "what is  LEFT to do?", "Any news?"]:
        _keep(store, query, now)
    _keep(store, "Where does Harlow live?", now, seeds=1)
    _keep(store, "", now, mode="browse", results=5)
    _keep(store, "nothing here", now, user_id="bea", results=0, judged=1, best_judged=0.1)
    _keep(store, "too old", _ago(days=40))
    stats = store.search_stats(days=30)
    total = stats["total"]
    assert (total["searches"], total["with_seeds"], total["without_seeds"]) == (5, 1, 4)
    assert total["share_without_seeds"] == pytest.approx(0.8)
    assert total["browses"] == 1 and total["no_results"] == 1
    assert total["judged"] == 1 and total["judged_none_answering"] == 1
    assert stats["namespaces"]["ada"]["searches"] == 4
    assert stats["namespaces"]["ada"]["share_without_seeds"] == pytest.approx(0.75)
    assert stats["namespaces"]["bea"]["without_seeds"] == 1
    assert stats["top_without_seeds"][0] == {"query": "What is left to do?", "count": 2}
    assert {item["query"] for item in stats["top_without_seeds"]} == {
        "What is left to do?", "Any news?", "nothing here"}
    assert store.search_stats(days=30, user_id="bea")["total"]["searches"] == 1
    assert store.search_stats(days=60)["total"]["searches"] == 6
    text = format_search_stats(stats)
    assert "share without" in text and "80.0%" in text and "2  What is left to do?" in text


def test_search_stats_on_an_empty_log(store):
    stats = store.search_stats()
    assert stats["total"]["searches"] == 0 and stats["total"]["share_without_seeds"] is None
    assert "No search without seeds." in format_search_stats(stats, logging_on=False)
    assert "The search log is off" in format_search_stats(stats, logging_on=False)


def test_the_command_prints_the_counts(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("MEMRY_SEARCH_LOG", "1")
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    assert main(["add", "Ada lives in Lisbon", "-u", "ada", "--no-infer"]) == 0
    assert main(["search", "where is the boiler", "-u", "ada"]) == 0
    capsys.readouterr()
    assert main(["search-stats", "--days", "7"]) == 0
    out = capsys.readouterr().out
    assert "last 7 days" in out and "where is the boiler" in out
    assert main(["search-stats", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["without_seeds"] == 1


# ------------------------------------------------- exports and snapshots

def test_an_export_leaves_the_log_out_unless_asked(store):
    _remember(store, "Ada lives in Lisbon")
    store.search("lisbon", user_id="ada")
    store.search("porto", user_id="bea")
    plain = store.export_backup(user_id="ada")
    assert "search_log" not in plain and "search_log" not in plain["tables"]
    assert "lisbon" not in json.dumps(plain).replace("Lisbon", "")
    asked = store.export_backup(user_id="ada", search_log=True)
    assert [r["query"] for r in asked["search_log"]] == ["lisbon"]

    other = _store()
    try:
        result = other.import_backup(asked)
        assert result["search_log"] == 1
        other.import_backup(asked)  # twice: kept once
        assert [r["query"] for r in _rows(other)] == ["lisbon"]
        other.import_backup(plain)  # a backup without the log restores without it
        assert len(_rows(other)) == 1
    finally:
        other.close()


def test_an_import_refuses_another_accounts_searches(store):
    store.search("lisbon", user_id="acct::ada")
    backup = store.export_backup(search_log=True)
    backup["search_log"][0]["user_id"] = "other::ada"
    with pytest.raises(ValueError, match="outside this account"):
        store.import_backup(backup, owner_prefix="acct::")


def test_a_snapshot_leaves_the_log_out_unless_asked(tmp_path):
    s = _store(db_path=str(tmp_path / "data" / "memry.db"))
    try:
        _remember(s, "Ada lives in Lisbon")
        s.search("a very particular question about the boiler", user_id="ada")
        target = tmp_path / "snap"
        assert take_snapshot(s.config, target, offsite=False)["ok"]
        copy = (target / "memry.db").read_bytes()
        assert b"particular question" not in copy
        conn = sqlite3.connect(target / "memry.db")
        assert conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        conn.close()
        assert len(_rows(s)) == 1  # the live log is untouched
        s.config.snapshot.include_search_log = True
        assert take_snapshot(s.config, target, offsite=False)["ok"]
        conn = sqlite3.connect(target / "memry.db")
        assert conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0] == 1
        conn.close()
    finally:
        s.close()


# ------------------------------------------------------ keys from traffic

@pytest.fixture
def keyed():
    s = _store(traffic_keys=True)
    for n in range(12):
        _remember(s, f"Note {n} on the quarterly roadmap review")
    yield s
    s.close()


def test_a_search_followed_by_a_save_keys_the_saved_memory(keyed):
    keyed.search("when was the boiler serviced", user_id="ada", run_id="r1")
    saved = keyed.add("The boiler was serviced on 3 May by Vela Heating",
                      user_id="ada", run_id="r1", infer=False)
    memory_id = saved.actions[0].memory_id
    assert _keys(keyed, memory_id) == [("when was the boiler serviced", "traffic")]
    row = _rows(keyed)[0]
    assert row["keyed"] == 1
    # the key has a vector, so the question's meaning matches too
    stored = keyed.backend._db.execute(
        "SELECT embedding_model FROM memory_questions WHERE memory_id = ?",
        (memory_id,)).fetchone()[0]
    assert stored == keyed.embedder.model_id


def test_no_key_without_the_flag_or_without_the_log():
    for search_log, traffic_keys in [(True, False), (False, True)]:
        s = _store(search_log=search_log, traffic_keys=traffic_keys)
        try:
            if not search_log:  # a search kept before the log was turned off
                _keep(s, "when was the boiler serviced", _ago(minutes=1))
            s.search("when was the boiler serviced", user_id="ada")
            saved = s.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
            assert _keys(s, saved.actions[0].memory_id) == []
        finally:
            s.close()


def test_traffic_keys_without_question_keys_warn_once_and_stay_off(caplog):
    with caplog.at_level("WARNING", logger=store_module.log.name):
        s = _store(traffic_keys=True, question_keys=False)
    try:
        warned = [r for r in caplog.records if "traffic_keys" in r.getMessage()]
        assert len(warned) == 1
        s.search("when was the boiler serviced", user_id="ada")
        saved = s.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
        assert _keys(s, saved.actions[0].memory_id) == []
        assert len([r for r in caplog.records if "traffic_keys" in r.getMessage()]) == 1
    finally:
        s.close()


def test_traffic_keys_with_question_keys_open_without_a_warning(caplog):
    with caplog.at_level("WARNING", logger=store_module.log.name):
        s = _store(traffic_keys=True, question_keys=True)
    s.close()
    assert not [r for r in caplog.records if "traffic_keys" in r.getMessage()]


def test_the_windows_the_run_for_an_hour_the_namespace_for_ten_minutes(keyed):
    _keep(keyed, "who serviced the boiler", _ago(minutes=40), run_id="r1")
    _keep(keyed, "what does the boiler cost", _ago(minutes=40), run_id="r2")
    _keep(keyed, "is the boiler under warranty", _ago(minutes=5))
    _keep(keyed, "boiler service date", _ago(minutes=90), run_id="r1")
    _keep(keyed, "who fixed the boiler", _ago(minutes=1), user_id="bea")
    saved = keyed.add("The boiler was serviced by Vela Heating, under warranty, at no cost",
                      user_id="ada", run_id="r1", infer=False)
    keys = {text for text, _ in _keys(keyed, saved.actions[0].memory_id)}
    assert keys == {"who serviced the boiler", "is the boiler under warranty"}


def test_a_query_that_would_not_rank_the_memory_gives_no_key(keyed):
    for n in range(30):
        _remember(keyed, f"The garden plan for bed {n} has tomatoes and garden beans")
    keyed.search("garden beans tomatoes plan", user_id="ada")
    saved = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    assert _keys(keyed, saved.actions[0].memory_id) == []
    assert _rows(keyed)[0]["keyed"] == 0


def test_a_query_keys_one_memory_once(keyed):
    keyed.search("when was the boiler serviced", user_id="ada")
    first = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    second = keyed.add("The boiler was serviced again on 9 June", user_id="ada", infer=False)
    assert _keys(keyed, first.actions[0].memory_id) == [
        ("when was the boiler serviced", "traffic")]
    assert _keys(keyed, second.actions[0].memory_id) == []


def test_browses_and_other_namespaces_give_no_key(keyed):
    keyed.search("", user_id="ada")
    keyed.search("when was the boiler serviced", user_id="bea")
    saved = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    assert _keys(keyed, saved.actions[0].memory_id) == []


def test_a_failed_embedding_keeps_the_words(keyed, monkeypatch):
    keyed.search("when was the boiler serviced", user_id="ada")
    real = keyed.embedder.embed

    def embed(texts):
        if texts == ["when was the boiler serviced"]:
            raise RuntimeError("embedding service down")
        return real(texts)

    monkeypatch.setattr(keyed.embedder, "embed", embed)
    saved = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    memory_id = saved.actions[0].memory_id
    assert _keys(keyed, memory_id) == [("when was the boiler serviced", "traffic")]
    assert keyed.backend.questions_without_vectors(Scope(user_id="ada"),
                                                   keyed.embedder.model_id)
    found = keyed.backend.question_keyword_search("boiler serviced", Scope(user_id="ada"))
    assert [memory.id for memory, _ in found] == [memory_id]


def test_a_failure_never_fails_the_save(keyed, monkeypatch):
    keyed.search("when was the boiler serviced", user_id="ada")

    def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(keyed, "_unjudged_order", broken)
    saved = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    assert saved.actions[0].event == "ADD"


def test_the_ordering_again_is_not_kept_in_the_log(keyed):
    keyed.search("when was the boiler serviced", user_id="ada")
    keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    assert len(_rows(keyed)) == 1


def test_the_key_counts_toward_the_limit(keyed, monkeypatch):
    saved = keyed.add("The boiler was serviced on 3 May", user_id="ada", infer=False)
    memory_id = saved.actions[0].memory_id
    keyed._write_questions(memory_id, [f"question {n}?" for n in range(9)], "backfill")
    keyed.search("when was the boiler serviced", user_id="ada")
    result = keyed._traffic_keys(Scope(user_id="ada"), [memory_id])
    assert result == {"searches": 1, "keys": 0}
    assert len(_keys(keyed, memory_id)) == 9


def test_keys_per_query_is_a_setting(keyed, monkeypatch):
    first = _remember(keyed, "The boiler was serviced on 3 May")
    second = _remember(keyed, "The boiler service cost 90 euros")
    keyed.search("boiler service", user_id="ada")
    monkeypatch.setattr(store_module, "TRAFFIC_KEYS_PER_QUERY", 2)
    result = keyed._traffic_keys(Scope(user_id="ada"), [first.id, second.id])
    assert result == {"searches": 1, "keys": 2}
