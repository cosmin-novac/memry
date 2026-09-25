"""Entity identity: which merge rule makes the obvious merges and no wrong ones?

``identity_benchmark.py`` measured how far a "same" verdict can be trusted.
This measures the rule built on top of it. On a real store the pairs left for
a person were mostly obvious: "Fundation GmbH" beside "Fundation GmbH", a full
name beside the same full name. Their facts were about unrelated topics (a
company's insurance, its tax number, an employee's contract dates), and asked
whether that is clearly the same company, Jev answered "same" at 40-69%, under
its 70% gate.

Two datasets:

* ``identity_v1.jsonl``, the 56 cases the gate was measured on;
* ``identity_v2.jsonl``, 45 cases shaped like the real misses: specific names
  whose facts share no topic (English and German mixed, as real stores are),
  a legal form or web domain written or left out, references like "PR #92"
  that are only unique inside a repository, first names of full names, and
  two different things that share a specific name.

Rules compared, per provider:

* ``today``: merge when "same" clears the provider's gate, or when the
  deterministic full-name rule sees shared context words;
* ``specific: unless 'different'``: also merge two mentions of the same
  specific name (``specific_same_name``) unless the judge answers "different";
* ``specific: unless either says so``: the same, with a second question that
  asks only for concrete evidence of two different things as a second veto;
* ``specific: any 'same'``: merge two mentions of the same specific name when
  the judge answers "same" at any confidence;
* ``shipped``: ``_should_merge`` as Memry runs it, which is the rule above.

Every rule except ``today`` also lets a "different" from 0.5 stop the
word-overlap rule. Today that takes the gate, which is 0.95 for gpt-5-mini.

Each rule is scored twice: with both mentions typed as an extractor types them,
and with only the existing entity's type known, which is how much of the
safety comes from the judge rather than from the type check.

Judgements are cached per provider, so the rules can be changed and re-scored
without calling the provider again.

Run:
    OPENAI_API_KEY=...   python evals/identity_policy_benchmark.py llm --model gpt-5-mini
    TYPESAFE_API_KEY=... python evals/identity_policy_benchmark.py jev
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memry.config import DecisionConfig, LLMConfig  # noqa: E402
from memry.intelligence.entities import (  # noqa: E402
    IDENTITY_QUESTION,
    _identity_state,
    _judge,
    _obvious_same_entity,
    _should_merge,
    specific_same_name,
)
from memry.models import Entity  # noqa: E402
from memry.providers.decisions import (  # noqa: E402
    Choice,
    JevDecider,
    LLMDecider,
    merge_gate_for,
)
from memry.providers.llm import build_llm  # noqa: E402

HERE = pathlib.Path(__file__).parent
DATASETS = (HERE / "datasets" / "identity_v1.jsonl", HERE / "datasets" / "identity_v2.jsonl")

CONFLICT_QUESTION = Choice(
    instructions=(
        "The EXISTING entity and the NEW fact use the same name. Is there concrete "
        "evidence that they are two different things?"
    ),
    criteria={
        "different": (
            "Yes: the facts contradict each other on age, employer, location, size, "
            "dates, identifiers, relationships or the kind of thing, or the new fact "
            "is about something else that only shares the name."
        ),
        "compatible": (
            "No: nothing in the facts contradicts one thing, even when the facts "
            "are about unrelated topics."
        ),
    },
)


def load_cases() -> list[dict]:
    cases = []
    for path in DATASETS:
        with open(path, encoding="utf-8") as fh:
            cases += [json.loads(line) for line in fh if line.strip()]
    return cases


def _entity(case: dict, *, typed: bool = True) -> Entity:
    return Entity(id=case["existing_name"].lower(), name=case["existing_name"],
                  normalized=case["existing_name"].lower(), user_id="bench",
                  entity_type=case.get("existing_type") if typed else None)


def judge_all(cases: list[dict], provider: str, model: str) -> list[dict]:
    """The identity question Memry asks today, and the conflict question."""
    if provider == "jev":
        decider = JevDecider(DecisionConfig(provider="jev",
                                            api_key=os.environ["TYPESAFE_API_KEY"]))
    else:
        decider = None

    def one(case: dict) -> dict:
        state = _identity_state(_entity(case), case["existing_facts"],
                                case["new_fact"], case["surface"])
        started = time.time()
        row: dict = {"id": case["id"]}
        try:
            if decider is not None:
                answers = decider.decide(state, {"identity": IDENTITY_QUESTION,
                                                 "conflict": CONFLICT_QUESTION})
                identity, conflict = answers["identity"], answers["conflict"]
                row.update(verdict=identity.value, confidence=identity.confidence,
                           probabilities=identity.probabilities,
                           conflict=conflict.value, conflict_confidence=conflict.confidence,
                           conflict_probabilities=conflict.probabilities)
            else:
                llm = build_llm(LLMConfig(provider="openai", model=model))
                try:
                    judged = _judge(llm, _entity(case), case["existing_facts"],
                                    case["new_fact"], case["surface"])
                    conflict = LLMDecider(llm).decide(state, {"conflict": CONFLICT_QUESTION})
                finally:
                    llm.close()
                row.update(verdict=judged.get("verdict"),
                           confidence=float(judged.get("confidence", 0.5)),
                           conflict=conflict["conflict"].value,
                           conflict_confidence=conflict["conflict"].confidence)
        except Exception as exc:  # a provider hiccup is a result, not a crash
            row.update(verdict=None, confidence=0.0, conflict=None,
                       conflict_confidence=0.0, error=str(exc)[:160])
        row["ms"] = (time.time() - started) * 1000
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, cases))


VETO = 0.5


def decide(case: dict, row: dict, rule: str, gate: float, *, typed: bool) -> bool:
    """Would Memry merge this pair under ``rule``?"""
    existing = _entity(case, typed=True)
    new_type = case.get("new_type") if typed else None
    verdict, confidence = row["verdict"], row["confidence"]
    if rule == "shipped":
        judgment = {"verdict": verdict, "confidence": confidence, "gate": gate}
        return _should_merge(existing, case["existing_facts"], case["surface"],
                             [case["new_fact"]], new_type, judgment)
    # today: only a "different" at the gate (at most 0.95) stops the word-overlap rule
    bar = min(gate, 0.95) if rule == "today" else VETO
    different = verdict == "different" and confidence >= bar
    merged = (
        (verdict == "same" and confidence >= gate)
        or (not different and _obvious_same_entity(
            existing, case["existing_facts"], case["surface"], [case["new_fact"]], new_type))
    )
    if merged or rule == "today":
        return merged
    if not specific_same_name(existing, case["surface"], new_type):
        return False
    if rule == "unless different":
        return not different
    if rule == "unless different, either question":
        return not different and not (
            row["conflict"] == "different" and row["conflict_confidence"] >= VETO)
    return verdict == "same"


RULES = (
    ("today", "today"),
    ("specific: unless 'different'", "unless different"),
    ("specific: unless either says so", "unless different, either question"),
    ("specific: any 'same'", "any same"),
    ("shipped (_should_merge)", "shipped"),
)


def report(label: str, cases: list[dict], rows: dict[str, dict], gate: float) -> None:
    print(f"\n=== {label} (gate {gate:.2f})")
    errors = [r for r in rows.values() if r.get("error")]
    if errors:
        print(f"{len(errors)} errors, e.g. {errors[0]['error']}")
    same = [c for c in cases if c["truth"] == "same"]
    other = [c for c in cases if c["truth"] != "same"]
    print(f"{'rule':<34} {'types':<9} {'merged of same':>15} {'WRONG merges':>13}")
    for name, rule in RULES:
        for typed in (True, False):
            merged = [c for c in same if decide(c, rows[c["id"]], rule, gate, typed=typed)]
            wrong = [c for c in other if decide(c, rows[c["id"]], rule, gate, typed=typed)]
            flag = "  <- " + ",".join(c["id"] for c in wrong) if wrong else ""
            print(f"{name:<34} {'both' if typed else 'existing':<9} "
                  f"{len(merged):>6}/{len(same):<8} {len(wrong):>13}{flag}")
    missed = [c["id"] for c in same
              if not decide(c, rows[c["id"]], "any same", gate, typed=True)]
    print(f"\nstill left for a person under the 'any same' rule: {', '.join(missed)}")
    print("\nper case (verdict / conflict):")
    for case in cases:
        row = rows[case["id"]]
        print(f"  {case['id']:<4} {case['truth']:<9} {case['surface'][:26]:<26} "
              f"{str(row['verdict']):<9} {row['confidence']:.2f}   "
              f"{str(row['conflict']):<10} {row['conflict_confidence']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("provider", choices=("jev", "llm"))
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="judgements file; reused when it exists")
    args = parser.parse_args()

    cases = load_cases()
    label = "jev" if args.provider == "jev" else args.model
    cache = args.cache or HERE / "results" / f"identity_policy_{label}.jsonl"
    if cache.exists():
        judged = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    else:
        judged = judge_all(cases, args.provider, args.model)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in judged) + "\n",
                         encoding="utf-8")
    rows = {r["id"]: r for r in judged}
    gate = 0.70 if args.provider == "jev" else merge_gate_for(args.model)
    print(f"{len(cases)} labelled cases, judgements from {cache}")
    report(label, cases, rows, gate)
