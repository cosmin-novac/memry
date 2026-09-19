"""Is there anything in this message worth remembering? A gate before extraction.

Every message sent with ``infer=True`` goes to the text model to have its facts
pulled out, and most messages to an assistant hold nothing worth keeping:
"thanks", "what's the capital of Australia", "summarise this". A single cheap
typed question in front of that call (see the ``extract_facts`` calls in
``store.add``) would skip the expensive one for those.

Whether it is safe to wire depends on one number: how many messages that *do*
hold something worth keeping it would skip. Skipping "I'm allergic to
penicillin" costs far more than one wasted extraction call, so the threshold
to pick is the highest one that skips nothing storable in
``datasets/storable_v1.jsonl`` (22 storable, 22 not), with headroom.

This stage is not wired. Run this first, on your own messages too.

Run:
    TYPESAFE_API_KEY=... python evals/storable_benchmark.py jev
    OPENAI_API_KEY=...   python evals/storable_benchmark.py llm --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memry.config import DecisionConfig, LLMConfig  # noqa: E402
from memry.providers.decisions import JevDecider, LLMDecider, Noul  # noqa: E402
from memry.providers.llm import build_llm  # noqa: E402

DATASET = pathlib.Path(__file__).parent / "datasets" / "storable_v1.jsonl"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
STATE = ("A message a person sent to their assistant. The assistant keeps a "
         "long-term memory of facts, preferences, decisions, plans and "
         "corrections about that person.")


def question(text: str) -> Noul:
    return Noul(instructions="This message contains something worth remembering "
                             "about the person later: a fact about them or someone "
                             "close to them, a preference, a decision, a plan, or a "
                             f"correction to something already known. Message: {text!r}")


def load_cases() -> list[dict]:
    with open(DATASET, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(decider, cases: list[dict]) -> list[dict]:
    started = time.time()
    answers = decider.decide(STATE, {c["id"]: question(c["text"]) for c in cases})
    ms = (time.time() - started) * 1000
    rows = []
    for c in cases:
        a = answers[c["id"]]
        rows.append({**c, "value": a.value if a.available else None, "available": a.available})
    print(f"{len(cases)} questions in one call, {ms:.0f} ms")
    return rows


def report(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label}")
    missing = [r["id"] for r in rows if not r["available"]]
    if missing:
        print(f"no answer for {len(missing)}: {', '.join(missing)}")
    print(f"\n{'threshold':>9} {'skipped':>8} {'of not-storable':>16} {'SKIPPED BUT STORABLE':>22}")
    best = None
    for t in THRESHOLDS:
        skip = [r for r in rows if r["available"] and r["value"] < t]
        right = [r for r in skip if not r["storable"]]
        wrong = [r for r in skip if r["storable"]]
        flag = "  <- " + ", ".join(r["id"] for r in wrong) if wrong else ""
        not_storable = sum(1 for r in rows if not r["storable"])
        print(f"{t:>9.2f} {len(skip):>8} {len(right)}/{not_storable:>13} {len(wrong):>22}{flag}")
        if not wrong:
            best = (t, len(right), not_storable)
    if best:
        t, right, total = best
        print(f"\nhighest threshold that skips nothing storable: {t:.2f}, which skips "
              f"{right} of the {total} messages with nothing in them")
    else:
        print("\nno threshold skips only what should be skipped; do not wire this")
    for r in rows:
        if r["available"]:
            mark = "storable" if r["storable"] else "not     "
            print(f"  {mark} {r['value']:.2f}  {r['text'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("jev", "llm"))
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    cases = load_cases()
    if args.provider == "jev":
        decider = JevDecider(DecisionConfig(provider="jev",
                                            api_key=os.environ["TYPESAFE_API_KEY"]))
        report("jev", run(decider, cases))
    else:
        llm = build_llm(LLMConfig(provider="openai", model=args.model))
        try:
            report(args.model, run(LLMDecider(llm), cases))
        finally:
            llm.close()
