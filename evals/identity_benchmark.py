"""Entity identity: which provider's confidence can an automatic merge trust?

Memry merges two entities without asking when a "same" verdict clears
``Decider.auto_confirm_confidence``. That number is only meaningful relative to
how a provider's confidence is distributed, so it is measured per provider here
rather than guessed once.

``datasets/identity_v1.jsonl`` holds 56 labelled cases - 22 that are one entity,
22 that are two, and 12 that nothing in the store settles - covering the
confusions that actually happen: a partner and a colleague sharing a first name,
a nickname, a person and a project sharing a name, a role change, a house move,
conflicting employee numbers, a bare first name with no evidence.

What the numbers in docs/self-hosting.md came from:

  jev-1.13.0      53/56 safe verdicts, median 211 ms. Its worst wrong "same"
                  scored 0.50 while correct ones ran to a median of 0.89, so the
                  two separate and a gate at 0.70 merges 20 of 22 correct pairs
                  and nothing it should not.
  gpt-5-mini      49/56, median 2,535 ms. A wrong "same" scored 0.90, the same
                  as the median of its right ones, so nothing separates them:
                  the lowest gate that lets nothing through is 0.95, and that
                  automates 4 of 22.

The conclusion that matters is not which model is better at answering. Both
found all 22 genuine matches. It is that one of them reports useful uncertainty
and the other does not, and only the first can be allowed to act alone.

Run:
    TYPESAFE_API_KEY=... python evals/identity_benchmark.py jev
    OPENAI_API_KEY=...   python evals/identity_benchmark.py llm --model gpt-5-mini
    ... python evals/identity_benchmark.py both
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memry.config import DecisionConfig, LLMConfig  # noqa: E402
from memry.intelligence.entities import (  # noqa: E402
    IDENTITY_QUESTION,
    _identity_state,
    _judge,
)
from memry.models import Entity  # noqa: E402
from memry.providers.decisions import JevDecider  # noqa: E402
from memry.providers.llm import build_llm  # noqa: E402

DATASET = pathlib.Path(__file__).parent / "datasets" / "identity_v1.jsonl"
GATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def load_cases() -> list[dict]:
    with open(DATASET, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _entity(name: str) -> Entity:
    return Entity(id=name.lower(), name=name, user_id="bench")


def run_jev(cases: list[dict]) -> list[dict]:
    decider = JevDecider(DecisionConfig(provider="jev",
                                        api_key=os.environ["TYPESAFE_API_KEY"]))

    def one(case: dict) -> dict:
        started = time.time()
        answer = decider.decide(
            _identity_state(_entity(case["existing_name"]), case["existing_facts"],
                            case["new_fact"], case["surface"]),
            {"identity": IDENTITY_QUESTION},
        )["identity"]
        return {**case, "verdict": answer.value, "confidence": answer.confidence,
                "ms": (time.time() - started) * 1000}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, cases))


def run_llm(cases: list[dict], model: str) -> list[dict]:
    def one(case: dict) -> dict:
        llm = build_llm(LLMConfig(provider="openai", model=model))
        started = time.time()
        try:
            judged = _judge(llm, _entity(case["existing_name"]), case["existing_facts"],
                            case["new_fact"], case["surface"])
            row = {"verdict": judged.get("verdict"),
                   "confidence": float(judged.get("confidence", 0.5))}
        except Exception as exc:  # a provider hiccup is a result, not a crash
            row = {"verdict": None, "confidence": 0.0, "error": str(exc)[:120]}
        finally:
            llm.close()
        return {**case, **row, "ms": (time.time() - started) * 1000}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, cases))


def safe(truth: str, verdict: str | None) -> bool:
    """Would this verdict leave the store intact? Anything but a wrong merge."""
    if truth == "same":
        return verdict == "same"
    if truth == "not-same":
        return verdict in ("different", "unsure")
    return verdict == "unsure"


def report(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label}")
    total = len(rows)
    print(f"safe verdicts   {sum(safe(r['truth'], r['verdict']) for r in rows)}/{total}")
    for truth in ("same", "not-same", "ambiguous"):
        group = [r for r in rows if r["truth"] == truth]
        got = sum(safe(truth, r["verdict"]) for r in group)
        print(f"  {truth:<10} {got}/{len(group)}")
    print(f"latency         median {statistics.median(r['ms'] for r in rows):.0f} ms")

    same = [r for r in rows if r["truth"] == "same"]
    risky = [r for r in rows if r["truth"] != "same"]
    print(f"\n{'gate':>5} {'auto-merged':>12} {'of true-same':>13} {'WRONG':>7}")
    best = None
    for gate in GATES:
        merged = [r for r in same if r["verdict"] == "same" and r["confidence"] >= gate]
        wrong = [r for r in risky if r["verdict"] == "same" and r["confidence"] >= gate]
        flag = "  <- " + ",".join(r["id"] for r in wrong) if wrong else ""
        print(f"{gate:>5.2f} {len(merged) + len(wrong):>12} "
              f"{len(merged)}/{len(same):>11} {len(wrong):>7}{flag}")
        if not wrong and (best is None or len(merged) > best[1]):
            best = (gate, len(merged))
    if best:
        print(f"lowest safe gate {best[0]:.2f}: {best[1]}/{len(same)} correct merges "
              f"happen automatically, none wrong")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("jev", "llm", "both"), default="both",
                        nargs="?")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    cases = load_cases()
    print(f"{len(cases)} labelled cases from {DATASET.name}")
    if args.provider in ("jev", "both"):
        report("jev", run_jev(cases))
    if args.provider in ("llm", "both"):
        report(args.model, run_llm(cases, args.model))
