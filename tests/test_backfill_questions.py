"""The question-key backfill: memories saved before there were question keys
get theirs written by the text model, in batches, and only those."""

from __future__ import annotations

import json

import pytest
from conftest import FakeLLM

from memry.config import Config
from memry.intelligence.questions import BACKFILL_SYSTEM, write_questions
from memry.models import Memory
from memry.providers.embeddings import HashEmbedder
from memry.store import MemoryStore


def _answer(*items: tuple[int, list[str]]) -> str:
    return json.dumps({"items": [{"n": n, "questions": qs} for n, qs in items]})


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(llm):
    """A store saved without question keys: the backfill is for such stores."""
    cfg = Config(db_path=":memory:")
    cfg.retrieval.question_keys = False
    s = MemoryStore(cfg, llm=llm, embedder=HashEmbedder(64))
    yield s
    s.close()


def _insert(store, content, user_id):
    """A memory saved one minute after the one before, so the backfill's
    order (oldest first) is the order of insertion."""
    _insert.minute += 1
    return store.backend.insert_memory(Memory(
        content=content, user_id=user_id,
        created_at=f"2024-01-01T{_insert.minute // 60:02d}:{_insert.minute % 60:02d}:00+00:00"))


_insert.minute = 0


def _texts(store, memory_id):
    return [q["text"] for q in store.backend.questions_of([memory_id]).get(memory_id, [])]


def test_write_questions_maps_numbers_back_and_cleans():
    llm = FakeLLM([_answer(
        (2, ["Where does Ada live?", "where does ada live?", "Which city is Ada in?",
             "Did Ada move?", "One too many?"]),
        (7, ["Unknown number"]),
        (1, "not a list"),
    )])
    out = write_questions(llm, ["Bo likes tea", "Ada lives in Amsterdam"])
    assert out == [[], ["Where does Ada live?", "Which city is Ada in?", "Did Ada move?"]]
    system, user = llm.calls[0]
    assert system == BACKFILL_SYSTEM
    assert "1. Bo likes tea" in user and "2. Ada lives in Amsterdam" in user


def test_backfill_writes_questions_for_memories_without_any(store, llm):
    assert store.config.retrieval.question_keys is False  # works with the setting off
    ada = _insert(store, "Ada moved to Amsterdam", "ada")
    tea = _insert(store, "Ada drinks green tea", "ada")
    llm.queue(_answer((1, ["Where does Ada live?", "Which city did Ada move to?"]),
                      (2, ["What does Ada drink?"])))

    summary = store.backfill_questions(user_id="ada")

    assert summary == {"checked": 2, "written": 2}
    assert _texts(store, ada.id) == ["Where does Ada live?", "Which city did Ada move to?"]
    assert _texts(store, tea.id) == ["What does Ada drink?"]
    rows = store.backend.questions_of([ada.id])[ada.id]
    assert {row["source"] for row in rows} == {"backfill"}
    assert all(row["embedding_model"] for row in rows)


def test_memory_with_questions_is_not_asked_again(store, llm):
    done = _insert(store, "Bo plays chess", "bo")
    store._write_questions(done.id, ["What game does Bo play?"], "save")
    due = _insert(store, "Bo lives in Oslo", "bo")
    llm.queue(_answer((1, ["Where does Bo live?"])))

    summary = store.backfill_questions(user_id="bo")

    assert summary == {"checked": 1, "written": 1}
    assert len(llm.calls) == 1 and "Bo plays chess" not in llm.calls[0][1]
    assert _texts(store, done.id) == ["What game does Bo play?"]
    assert _texts(store, due.id) == ["Where does Bo live?"]
    # a second run finds nothing to ask (FakeLLM would raise if asked)
    assert store.backfill_questions(user_id="bo") == {"checked": 0, "written": 0}


def test_memory_the_model_found_no_question_for_is_not_asked_again(store, llm):
    """A memory the model gives [] for is marked (``questions_checked``) and
    left out of the next run; a dry run marks nothing."""
    plain = _insert(store, "Thanks!", "bo")
    llm.queue(_answer((1, [])))
    assert store.backfill_questions(user_id="bo", dry_run=True) == {
        "checked": 1, "written": 0, "proposals": []}
    assert store.backend.get_memory(plain.id).metadata == {}
    llm.queue(_answer((1, [])))
    assert store.backfill_questions(user_id="bo") == {"checked": 1, "written": 0}
    assert store.backend.get_memory(plain.id).metadata == {"questions_checked": True}
    assert _texts(store, plain.id) == []
    # asked once: the next run has nothing to ask (FakeLLM would raise)
    assert store.backfill_questions(user_id="bo") == {"checked": 0, "written": 0}


def test_batches_and_limit(store, llm):
    ids = [_insert(store, f"Fact number {i}", "ada").id
           for i in range(5)]
    llm.queue(_answer((1, ["Q0?"]), (2, ["Q1?"])), _answer((1, ["Q2?"])))

    summary = store.backfill_questions(user_id="ada", batch=2, limit=3)

    assert summary == {"checked": 3, "written": 3}
    assert len(llm.calls) == 2
    assert [_texts(store, i) for i in ids] == [["Q0?"], ["Q1?"], ["Q2?"], [], []]


def test_dry_run_writes_nothing(store, llm):
    memory = _insert(store, "Ada moved to Amsterdam", "ada")
    llm.queue(_answer((1, ["Where does Ada live?"])))

    summary = store.backfill_questions(user_id="ada", dry_run=True)

    assert summary["written"] == 0 and summary["checked"] == 1
    assert summary["proposals"] == [{"id": memory.id, "content": "Ada moved to Amsterdam",
                                     "questions": ["Where does Ada live?"]}]
    assert store.backend.questions_of([memory.id]) == {}


def test_garbage_answer_writes_nothing_and_does_not_raise(store, llm):
    memory = _insert(store, "Ada moved to Amsterdam", "ada")
    llm.queue("I am not JSON at all")

    summary = store.backfill_questions(user_id="ada")

    assert summary == {"checked": 1, "written": 0}
    assert store.backend.questions_of([memory.id]) == {}


def test_failing_call_skips_the_batch_and_goes_on(store, llm, caplog):
    first = _insert(store, "Ada moved to Amsterdam", "ada")
    second = _insert(store, "Ada drinks green tea", "ada")
    calls = iter([RuntimeError("provider down"), _answer((1, ["What does Ada drink?"]))])

    def complete(system, user, *, json_schema=None):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    llm.complete = complete
    with caplog.at_level("WARNING", logger="memry"):
        summary = store.backfill_questions(user_id="ada", batch=1)

    assert summary == {"checked": 1, "written": 1, "failed_batches": 1}
    assert "provider down" in caplog.text
    assert store.backend.questions_of([first.id]) == {}
    assert _texts(store, second.id) == ["What does Ada drink?"]


def test_no_llm_is_skipped():
    from memry.providers.llm import NoneLLM

    s = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        s.backend.insert_memory(Memory(content="Ada moved to Amsterdam", user_id="ada"))
        assert s.backfill_questions() == {"skipped": "no LLM configured"}
    finally:
        s.close()


def test_cli_backfill_questions(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("MEMRY_DB_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "missing.json"))
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY",
                "MEMRY_LLM_PROVIDER", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    from memry.cli import main

    llm = FakeLLM()

    def make():
        return MemoryStore(Config.load(), llm=llm, embedder=HashEmbedder(64))

    monkeypatch.setattr("memry.cli._store", make)
    store = make()
    memory = _insert(store, "Ada moved to Amsterdam", "ada")
    store.close()

    llm.queue(_answer((1, ["Where does Ada live?"])))
    assert main(["backfill-questions", "-u", "ada", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["user"] == "ada" and out[0]["written"] == 0
    assert out[0]["proposals"][0]["questions"] == ["Where does Ada live?"]

    llm.queue(_answer((1, ["Where does Ada live?"])))
    assert main(["backfill-questions", "--limit", "5"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"user": "ada", "checked": 1, "written": 1}]

    store = make()
    assert _texts(store, memory.id) == ["Where does Ada live?"]
    store.close()
