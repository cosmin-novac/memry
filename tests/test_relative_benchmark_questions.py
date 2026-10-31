"""The question keys in ``evals/relative_retrieval_benchmark.py``: a store built
with ``questions`` keeps them as Memry keeps a backfill's and searches them,
``score`` runs each arm with the keys on or off and sets them back, and the
batched call that writes them reads the model's answer as the store wants it.
Offline: hash vectors, a small world, no model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import relative_retrieval_benchmark as bench  # noqa: E402
from memry.providers.embeddings import HashEmbedder  # noqa: E402
from memry.providers.llm import LLM  # noqa: E402

FAMILIES = ("single_fact_para", "inherit_para")


def _marker(k: int) -> str:
    """A word no memory and no other question holds."""
    return f"qzx{k}vb"


def _questions(world: dict) -> dict[str, list[str]]:
    """Two questions a memory, made without a model: its text asked about,
    and one in rare words only that memory's question holds."""
    return {str(k): [f"What about {m['text'].rstrip('.')}?",
                     f"Which record holds marker {_marker(k)}?"]
            for k, m in enumerate(world["memories"])}


@pytest.fixture(scope="module")
def built():
    world = bench.build_world_dense(250)
    world["queries"] = {f: world["queries"][f][:6] for f in FAMILIES}
    answers = json.loads((ROOT / "evals" / "datasets" / "belongs_answers.json").read_text())
    store, ids = bench.build_store(world, HashEmbedder(64), "none", answers["answers"],
                                   questions=_questions(world))
    yield world, store, ids
    store.close()


def test_the_questions_are_stored_as_a_backfill_stores_them(built):
    world, store, ids = built
    assert store.config.retrieval.question_keys is True
    kept = store.backend.questions_of(ids)
    assert len(kept) == len(world["memories"])
    first = kept[ids[0]]
    assert [q["text"] for q in first] == _questions(world)["0"]
    assert {q["source"] for q in first} == {"backfill"}
    assert all(q["embedding_model"] == store.embedder.model_id for q in first)


def test_a_question_in_rare_words_finds_its_memory_through_the_question_keys(built):
    _, store, ids = built
    k = 17
    query = f"Which record holds marker {_marker(k)}?"
    results = store.search(query, user_id=bench.USER, limit=10)
    found = {r.memory.id: r for r in results}
    assert ids[k] in found
    assert "question_keyword" in found[ids[k]].signals
    # its question matches best by words (hash vectors of the near alike
    # questions of other memories are close, so the fused order may differ)
    by_words = max(results, key=lambda r: r.signals.get("question_keyword", 0.0))
    assert by_words.memory.id == ids[k]
    # with the keys off, no result comes through a question
    store.config.retrieval.question_keys = False
    try:
        off = store.search(query, user_id=bench.USER, limit=10)
    finally:
        store.config.retrieval.question_keys = True
    assert not any("question_keyword" in r.signals for r in off)


def test_score_runs_each_arm_and_sets_the_keys_back(built):
    world, store, ids = built
    mode = next(m for m in bench.MODES if m[0] == "hybrid")
    arms = {}
    for keys in (True, False):
        arms[keys] = bench.score(store, ids, world["queries"], mode, question_keys=keys)
        assert store.config.retrieval.question_keys is True
    for res in arms.values():
        assert set(res) == set(FAMILIES)
        for family in FAMILIES:
            assert {"n", "mrr", "recall", "r20", "linked"} <= set(res[family])
            assert res[family]["n"] == len(world["queries"][family])
    # with no arm named, the store's own setting is read and left
    store.config.retrieval.question_keys = False
    try:
        bench.score(store, ids, {"single_fact_para": world["queries"]["single_fact_para"][:1]},
                    mode)
        assert store.config.retrieval.question_keys is False
    finally:
        store.config.retrieval.question_keys = True


class _Writer(LLM):
    """A text model that answers from a script: each call returns the next
    answer and keeps what it was asked."""

    name = "script"

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.asked = list(answers), []

    def complete(self, system, user, *, json_schema=None):
        self.asked.append(user)
        return self.answers.pop(0)


def test_ask_questions_reads_the_numbered_answer_and_cleans_it():
    llm = _Writer(['Here: {"items": [{"n": 2, "questions": ["Where does Ada live?", '
                   '" where does ada  live? ", "Which city?", "What else?"]}, '
                   '{"n": 9, "questions": ["Out of range?"]}, {"n": "1", "questions": ["x"]}]}'])
    got = bench.ask_questions(llm, ["Bo likes tea.", "Ada lives in Graz."])
    assert llm.asked == ["1. Bo likes tea.\n2. Ada lives in Graz."]
    assert got == [[], ["Where does Ada live?", "Which city?", "What else?"]]
    assert bench.ask_questions(_Writer(["not json"]), ["a", "b"]) == [[], []]


def test_write_world_questions_asks_again_for_a_skipped_memory_and_saves():
    texts = ["Bo likes tea.", "Ada lives in Graz.", "Kai runs."]
    llm = _Writer(['{"items": [{"n": 1, "questions": ["What does Bo like?"]}]}',
                   '{"items": []}'])
    saved = []
    got = bench.write_world_questions(llm, texts, {"1": ["Where does Ada live?"]},
                                      lambda q: saved.append(dict(q)), batch=5, workers=1)
    assert llm.asked == ["1. Bo likes tea.\n2. Kai runs.", "1. Kai runs."]
    assert got == {"0": ["What does Bo like?"], "1": ["Where does Ada live?"], "2": []}
    assert saved[-1] == got


def test_a_questions_file_is_read_by_size_or_whole():
    one = {"0": ["What?"]}
    assert bench.questions_for_size({"250": one, "500": {}}, 250) == one
    assert bench.questions_for_size({"250": one}, 500) is None
    assert bench.questions_for_size(one, 250) == one
