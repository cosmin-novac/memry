from __future__ import annotations

from pathlib import Path

from memry.config import Config
from memry.evals.harness import load_dataset, run_eval
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore

DATASET = Path(__file__).parent.parent / "evals" / "datasets" / "synthetic_v1.jsonl"


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
