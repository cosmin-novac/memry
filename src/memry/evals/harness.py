"""Retrieval evaluation harness.

Datasets are JSONL; each line is one case:

    {"id": "case-1",
     "sessions": [[{"role": "user", "content": "..."}, ...], ...],
     "questions": [{"q": "...", "expected_keywords": ["berlin"],
                    "answer": "optional gold answer"}]}

(``"conversation": [...]`` is accepted as shorthand for a single session.)

For every case the harness ingests each session through the full write path
(extraction + reconciliation with whatever LLM/embedder is configured - or the
zero-key verbatim path), then asks each question against retrieval and scores:

- recall@k - a hit is a retrieved memory containing any expected keyword
- MRR      - reciprocal rank of the first hit
- latency  - search p50/p95 in ms

This measures the *retrieval* half end-to-end and is deliberately judge-free
so it runs offline and deterministically in CI. Answer-quality scoring with an
LLM judge (LoCoMo/LongMemEval protocol) layers on top: format those datasets
into this schema and compare backends/configs under identical conditions.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from ..config import Config
from ..store import MemoryStore


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_eval(
    dataset_path: str | Path,
    *,
    k: int = 5,
    store_factory=None,
) -> dict[str, Any]:
    cases = load_dataset(dataset_path)
    if store_factory is None:
        def store_factory():
            cfg = Config.load(db_path=":memory:")
            return MemoryStore(cfg)

    hits = 0
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    total_questions = 0
    total_memories = 0
    llm_name = embedder_name = ""

    for case in cases:
        store = store_factory()
        llm_name = store.llm.name
        embedder_name = store.embedder.model_id
        user_id = str(case.get("id", "case"))
        sessions = case.get("sessions") or [case.get("conversation", [])]
        for session in sessions:
            if session:
                store.add(session, user_id=user_id)
        total_memories += len(store.get_all(user_id=user_id, limit=100_000))

        for question in case.get("questions", []):
            total_questions += 1
            keywords = [kw.lower() for kw in question.get("expected_keywords", [])]
            started = time.perf_counter()
            results = store.search(question["q"], user_id=user_id, limit=k)
            latencies.append((time.perf_counter() - started) * 1000)

            rank = None
            for i, result in enumerate(results):
                text = result.memory.content.lower()
                if any(kw in text for kw in keywords):
                    rank = i + 1
                    break
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        store.close()

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return values[idx]

    return {
        "dataset": str(dataset_path),
        "cases": len(cases),
        "questions": total_questions,
        "memories_stored": total_memories,
        "k": k,
        "recall_at_k": hits / total_questions if total_questions else 0.0,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "latency_ms_p50": pct(latencies, 0.50),
        "latency_ms_p95": pct(latencies, 0.95),
        "llm": llm_name,
        "embedder": embedder_name,
    }
