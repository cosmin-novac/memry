from __future__ import annotations

import importlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from memry.config import Config
from memry.evals.harness import load_dataset, run_eval
from memry.models import Scope
from memry.providers.decisions import Answer, Answers, NoneDecider
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "evals" / "datasets" / "synthetic_v1.jsonl"


def _bench(name: str):
    """One of the scripts under ``evals/``, imported as the tests of the
    benchmarks import them."""
    sys.path.insert(0, str(ROOT))
    return importlib.import_module(f"evals.{name}")


def test_dataset_loads():
    cases = load_dataset(DATASET)
    assert len(cases) >= 6
    for case in cases:
        assert case.get("questions")
        assert case.get("conversation") or case.get("sessions")


def test_harness_runs_zero_key_mode():
    def factory():
        return MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(128))

    report = run_eval(DATASET, k=5, store_factory=factory)
    assert report["questions"] > 0
    assert report["memories_stored"] > 0
    # verbatim + hybrid keyword retrieval should already do reasonably well
    assert report["recall_at_k"] >= 0.6, report
    assert 0.0 <= report["mrr"] <= 1.0


# ------------------------------------------- the relative retrieval benchmark
def test_the_benchmark_gives_a_person_two_different_languages():
    """A trace of the dense world showed a person who "speaks Spanish and
    Spanish": the two languages were drawn one at a time. The generator draws
    them together, so a person's two languages always differ."""
    bench = _bench("relative_retrieval_benchmark")
    spoken = re.compile(r"(?:speaks|is fluent in) (\w+) and (\w+)\.")
    pairs = []
    for seed in range(200):
        for text in bench._person_fillers("Kai Osei", random.Random(seed)):
            found = spoken.search(text)
            if found:
                pairs.append(found.groups())
    assert len(pairs) == 200
    assert all(first != second for first, second in pairs)


def test_the_benchmark_stores_every_memory_under_the_embedders_model():
    """The first relative benchmark runs stored memories with no embedding
    model, so the vector search matched none of them and every run measured
    the keyword search alone. ``build_store`` stores each memory under the
    embedder's model: the vector search finds each by its own text, and a
    search's results carry the vector signal."""
    bench = _bench("relative_retrieval_benchmark")
    world = bench.build_world(60)
    answers = json.loads((ROOT / "evals" / "datasets" / "belongs_answers.json").read_text())
    embedder = HashEmbedder(64)
    store, memory_ids = bench.build_store(world, embedder, "measured", answers["answers"])
    scope = Scope(user_id=bench.USER)
    for k in range(0, len(memory_ids), 25):
        text = world["memories"][k]["text"]
        found = store.backend.vector_search(embedder.embed([text])[0], embedder.model_id,
                                            scope, limit=5)
        assert found and found[0][0].id == memory_ids[k], text
    question, _, _ = world["queries"]["single_fact"][0]
    results = store.search(question, user_id=bench.USER, limit=10)
    assert any("vector" in r.signals for r in results)
    store.close()


class _Jev(NoneDecider):
    """What the benchmark's ``jev_judge`` builds, standing in for Jev: every
    one made is kept, it answers while open and, as a closed HTTP client
    does, not at all once closed (``fails`` makes it never answer)."""

    name, available, calibrated = "jev", True, True
    may_rerank = reranks_by_default = True
    made: list[_Jev] = []
    fails = False

    def __init__(self, cfg) -> None:
        self.closed = False
        self.asked: list[bool] = []  # closed or not, at each call
        _Jev.made.append(self)

    def decide(self, state, questions):
        self.asked.append(self.closed)
        up = not (self.closed or self.fails)
        return Answers({key: Answer(0.2, {}, 0.9 if up else 0.0, up) for key in questions})

    def close(self) -> None:
        self.closed = True


def _run_with_jev(monkeypatch, tmp_path, *, fails: bool) -> int | None:
    """The benchmark's ``main`` with ``--jev`` on a small simple world, both
    compared links, the Jev mode only, one question a family, and ``_Jev`` in
    place of Jev (no call leaves the machine). Returns the code the run
    exited with, None if it ran to the end."""
    import memry.providers.decisions as decisions

    bench = _bench("relative_retrieval_benchmark")
    _Jev.made, _Jev.fails = [], fails
    monkeypatch.setattr(decisions, "JevDecider", _Jev)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-key")
    monkeypatch.setenv("TMPDIR", str(tmp_path))  # the embedding cache

    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", exit_)
    monkeypatch.setattr(sys, "argv", [
        "bench", "--jev", "--sizes", "60", "--links", "oracle", "measured",
        "--modes", "linked jev", "--per-family", "1",
        "--families", "inherit", "override", "event_inherit"])
    try:
        bench.main()
    except SystemExit as exited:
        return exited.code
    return None


def test_the_benchmark_asks_a_fresh_jev_for_every_store(monkeypatch, tmp_path, capsys):
    """One Jev decider was shared by the benchmark's stores, and closing the
    first store closed it: every Jev call after that failed, and the search
    fell back to vectors unnoticed. Each store gets its own, and none is
    asked once closed."""
    assert _run_with_jev(monkeypatch, tmp_path, fails=False) is None
    assert len(_Jev.made) == 2  # one a store: oracle and measured links
    # one call a question, three questions a store, none once closed
    assert [judge.asked for judge in _Jev.made] == [[False] * 3, [False] * 3]
    assert "stopped" not in capsys.readouterr().out


def test_the_benchmark_stops_after_three_calls_jev_does_not_answer(
        monkeypatch, tmp_path, capsys):
    """A Jev that does not answer stops the run after three failed calls
    (exit code 2), so no result is measured on vectors while it claims Jev."""
    assert _run_with_jev(monkeypatch, tmp_path, fails=True) == 2
    assert [len(judge.asked) for judge in _Jev.made] == [3]
    assert "stopped: Jev is not answering" in capsys.readouterr().out


def test_the_benchmarks_answer_keys_hold_what_the_questions_ask():
    """Two answer keys were wrong: "Which cars cost less than 30,000
    euros?" counted every price, and the owner's favourite restaurant, one
    they like too, was scored a wrong member of "Which restaurants did I
    like?". The first key is the prices under 30,000 alone, and the
    favourite is in both liked sets."""
    bench = _bench("relative_retrieval_benchmark")
    world = bench.build_world_dense(60, owner=True)
    texts = [m["text"] for m in world["memories"]]
    sets = {question: gold for question, gold, _ in world["queries"]["set"]}
    price = re.compile(r"(?:costs|quoted) ([\d,]+) euros")
    prices = {k: int(price.search(texts[k]).group(1).replace(",", ""))
              for k in sets["Which car is the cheapest?"]}
    assert len(prices) == 40
    under = sets["Which cars cost less than 30,000 euros?"]
    assert sorted(under) == sorted(k for k, euros in prices.items() if euros < 30_000)
    assert 0 < len(under) < 40
    favourite = next(k for k, text in enumerate(texts)
                     if text.startswith(f"{bench.OWNER}'s favourite restaurant is"))
    for question in (f"Which restaurants did {bench.OWNER} like?", "Which restaurants did I like?"):
        assert favourite in sets[question]
        assert len(sets[question]) == 13  # the 12 liked and the favourite


def test_multi_hop_answers_do_not_name_the_person_asked_about():
    """An honest multi-hop question has no lexical shortcut: the memory that
    answers "What tools does Priya Nair use for their work?" names her
    project and its tool, never her, and is reached by a relation from her to
    that project. Both retrieval benchmarks build it so."""
    bench = _bench("relative_retrieval_benchmark")
    for world in (bench.build_world(60), bench.build_world_dense(60)):
        works_on = {(s, o) for s, predicate, o in world["relations"] if predicate == "works_on"}
        uses = {(s, o) for s, predicate, o in world["relations"] if predicate == "uses"}
        questions = world["queries"]["multi_hop"]
        assert questions
        for question, gold, _ in questions:
            who = re.match(r"What tools does (.+) use for their work\?", question).group(1)
            assert gold
            for k in gold:
                memory = world["memories"][k]
                assert who not in memory["text"], (question, memory["text"])
                project, tool = memory["entities"]
                assert (who, project) in works_on and (project, tool) in uses
    plain = _bench("retrieval_benchmark").build_store(40, 12, 8, 400)
    for person, projects in plain["person_projects"].items():
        gold = [k for project in projects for k in plain["project_uses_mems"][project]]
        assert gold
        for k in gold:
            memory = plain["mems"][k]
            assert person not in memory["text"] and person not in memory["ents"]
            assert memory["ents"] & set(projects)


def test_the_identity_comparison_runs_from_the_committed_dataset(monkeypatch, capsys):
    """The Jev and text-model comparison of the identity gate was published
    with no dataset, and two runs differed (53 and 52 of 56). The 56 cases
    are committed where ``identity_benchmark.py`` reads them (22 one
    entity, 22 two, 12 that nothing settles), and the script scores them
    offline with the decider swapped out: a judge that is unsure of
    everything is safe on the 34 that are not one entity."""
    bench = _bench("identity_benchmark")
    cases = bench.load_cases()
    assert bench.DATASET == ROOT / "evals" / "datasets" / "identity_v1.jsonl"
    assert Counter(case["truth"] for case in cases) == {
        "same": 22, "not-same": 22, "ambiguous": 12}
    assert len({case["id"] for case in cases}) == 56

    class Unsure(NoneDecider):
        available = True

        def __init__(self, cfg) -> None:
            pass

        def decide(self, state, questions):
            return Answers({key: Answer("unsure", {"unsure": 1.0}, 0.9, True)
                            for key in questions})

    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-key")
    monkeypatch.setattr(bench, "JevDecider", Unsure)
    bench.report("unsure", bench.run_jev(cases))
    out = capsys.readouterr().out
    assert "safe verdicts   34/56" in out
    assert "same       0/22" in out and "not-same   22/22" in out and "ambiguous  12/12" in out
