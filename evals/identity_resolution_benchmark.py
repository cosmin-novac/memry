"""Entity identity from first principles: which question, asked of Jev, decides
most pairs without a person and merges nothing wrongly?

Every case is two entities from one person's store, each with a name, a type
and its facts. The judge sees both profiles side by side. Three designs:

* ``today``: the question Memry asked until now. The existing entity's facts on
  one side, the other side's facts joined into one "new fact".
* ``one order``: both profiles, one question with three answers (one thing,
  two things, nothing settles it), decided on the probabilities;
* ``both orders``: the same question asked with A and B swapped as well, the
  probabilities averaged. This is what Memry ships
  (``memry.intelligence.identity``).
* ``three questions``: both profiles, three questions a person asks in turn.
  Are the two names one name written differently? Does that name mean one
  thing in a person's life, or many things (a first name, a bare number)? Do
  the facts link the two, contradict one thing, or neither? Merge when the
  names are one name and the facts do not contradict, and, for a name many
  things carry, only when the facts link them.

Nothing in any design lists suffixes, prefixes or forms of names: the judge
decides whether two names are one name.

Datasets: ``identity_v1`` (56), ``identity_v2`` (45) and ``identity_v3`` (55,
both sides with several facts, name variants from many languages and legal
systems, and near-identical names of two things). For each design the report
sweeps the merge threshold and the keep-apart threshold and prints the
operating point with no wrong merge: how many pairs are decided without a
person, and which are left. The last lines score the shipped thresholds.

Run:
    TYPESAFE_API_KEY=... python evals/identity_resolution_benchmark.py
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

from memry.config import DecisionConfig  # noqa: E402
from memry.intelligence.entities import (  # noqa: E402
    IDENTITY_QUESTION,
    _identity_state,
)
from memry.models import Entity  # noqa: E402
from memry.intelligence.identity import PAIR_QUESTION, Profile, pair_state  # noqa: E402
from memry.providers.decisions import Choice, JevDecider  # noqa: E402

HERE = pathlib.Path(__file__).parent
DATA = HERE / "datasets"

NAME_QUESTION = Choice(
    instructions="Are the names of entity A and entity B one name for one thing?",
    criteria={
        "one_name": (
            "Yes: the same name, as written or written differently (a legal form, "
            "title, abbreviation, acronym, nickname, translation, domain or "
            "spelling added or left out)."
        ),
        "two_names": (
            "No: the names only share a word, a surname or a number, or one of "
            "them names a part, member, version, product or owner of the other."
        ),
    },
)

NAME_KIND_QUESTION = Choice(
    instructions=(
        "Take the shorter of the two names. In one person's life, does that name "
        "usually mean exactly one thing?"
    ),
    criteria={
        "one_thing": (
            "Yes: a full personal name, a company, a product, a named project, "
            "programme or place, or a reference that carries its own context."
        ),
        "many_things": (
            "No: a first name alone, a surname alone, a role, a generic word or a "
            "bare number, which many different things carry."
        ),
    },
)

EVIDENCE_QUESTION = Choice(
    instructions="What do the facts about entity A and entity B say about them being one thing?",
    criteria={
        "link": (
            "They connect: the same role, relationship, employer, place, project, "
            "identifier or event appears on both sides."
        ),
        "contradict": (
            "They conflict: different people, ages, roles, places, dates, sizes, "
            "owners, versions or kinds of thing."
        ),
        "neutral": (
            "Neither: the facts are about unrelated topics and fit one thing."
        ),
    },
)


def load_cases() -> list[dict]:
    """All three datasets as pairs of profiles."""
    cases = []
    for name in ("identity_v1.jsonl", "identity_v2.jsonl"):
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append({
                "id": row["id"], "set": name[:-6], "category": row["category"],
                "truth": row["truth"],
                "a_name": row["existing_name"], "a_type": row.get("existing_type"),
                "a_facts": row["existing_facts"],
                "b_name": row["surface"], "b_type": row.get("new_type"),
                "b_facts": [row["new_fact"]],
            })
    for line in (DATA / "identity_v3.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append({**json.loads(line), "set": "identity_v3"})
    return cases


def profiles(case: dict) -> tuple[Profile, Profile]:
    """Both sides with dated facts, as Memry sends them. One rule for every
    case, so the dates carry no hint of the label: entity A's facts are the
    older ones (January onwards), entity B's the newer ones (July onwards)."""
    return (
        Profile(case["a_name"], case["a_type"], case["a_facts"],
                dates=[f"2025-{1 + i:02d}-10" for i in range(len(case["a_facts"]))]),
        Profile(case["b_name"], case["b_type"], case["b_facts"],
                dates=[f"2025-{7 + i:02d}-10" for i in range(len(case["b_facts"]))]),
    )


def judge(cases: list[dict], decider: JevDecider) -> list[dict]:
    def one(case: dict) -> dict:
        started = time.time()
        a, b = profiles(case)
        today = decider.decide(
            _identity_state(Entity(id="a", name=case["a_name"], user_id="bench"),
                            case["a_facts"], " / ".join(case["b_facts"]), case["b_name"]),
            {"identity": IDENTITY_QUESTION},
        )["identity"]
        forward = decider.decide(pair_state(a, b), {
            "pair": PAIR_QUESTION, "name": NAME_QUESTION,
            "name_kind": NAME_KIND_QUESTION, "evidence": EVIDENCE_QUESTION,
        })
        backward = decider.decide(pair_state(b, a), {"pair": PAIR_QUESTION})
        row = {"id": case["id"], "ms": (time.time() - started) * 1000,
               "today": today.probabilities,
               "pair": forward["pair"].probabilities, "pair_rev": backward["pair"].probabilities}
        for key in ("name", "name_kind", "evidence"):
            row[key] = forward[key].probabilities if forward[key].available else None
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, cases))


def scores(row: dict, design: str) -> tuple[float, float] | None:
    """(probability of one thing, probability of two things) under a design."""
    if design == "today":
        probs = row["today"] or {}
        return probs.get("same", 0.0), probs.get("different", 0.0)
    if design == "one order":
        probs = row["pair"]
        return (probs.get("same", 0.0), probs.get("different", 0.0)) if probs else None
    if design == "both orders":
        a, b = row["pair"], row["pair_rev"]
        if not (a and b):
            return None
        return ((a.get("same", 0.0) + b.get("same", 0.0)) / 2,
                (a.get("different", 0.0) + b.get("different", 0.0)) / 2)
    name, kind, evidence = row["name"], row["name_kind"], row["evidence"]
    if not (name and kind and evidence):
        return None
    one_name = name.get("one_name", 0.0)
    one_thing = kind.get("one_thing", 0.0)
    contradict = evidence.get("contradict", 0.0)
    link = evidence.get("link", 0.0)
    merge = one_name * (one_thing * (1 - contradict) + (1 - one_thing) * link)
    apart = 1 - one_name * (1 - contradict)
    return merge, apart


THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def operating_point(cases: list[dict], rows: dict[str, dict], design: str,
                    merge_at: float | None = None, apart_at: float | None = None) -> dict:
    """Lowest merge threshold with no wrong merge and lowest keep-apart threshold
    that separates no true pair, unless given; then what each pair gets."""
    scored = [(c, scores(rows[c["id"]], design)) for c in cases]
    scored = [(c, s) for c, s in scored if s is not None]
    if merge_at is None:
        merge_at = next((t for t in THRESHOLDS if not any(
            s[0] >= t for c, s in scored if c["truth"] != "same")), 1.01)
    if apart_at is None:
        apart_at = next((t for t in THRESHOLDS if not any(
            s[1] >= t and s[0] < merge_at for c, s in scored if c["truth"] == "same")), 1.01)
    out = {"merge_at": merge_at, "apart_at": apart_at, "merged": [], "apart": [],
           "left": [], "wrong_merge": [], "wrong_apart": []}
    for case, (merge, apart) in scored:
        if merge >= merge_at:
            out["merged" if case["truth"] == "same" else "wrong_merge"].append(case["id"])
        elif apart >= apart_at:
            out["apart" if case["truth"] != "same" else "wrong_apart"].append(case["id"])
        else:
            out["left"].append(case["id"])
    out["n"] = len(scored)
    return out


def report(cases: list[dict], rows: dict[str, dict]) -> None:
    designs = ("today", "one order", "both orders", "three questions")
    subsets = {
        "all 156": cases,
        "v1+v2 (101)": [c for c in cases if c["set"] != "identity_v3"],
        "v3 (55)": [c for c in cases if c["set"] == "identity_v3"],
    }
    print(f"\n{'design':<17} {'cases':<12} {'merge at':>8} {'apart at':>8} "
          f"{'merged':>7} {'kept apart':>10} {'left':>5} {'wrong merge':>11} "
          f"{'wrong apart':>11} {'decided':>8}")
    for design in designs:
        for label, subset in subsets.items():
            point = operating_point(subset, rows, design)
            decided = len(point["merged"]) + len(point["apart"]) + len(point["wrong_apart"])
            print(f"{design:<17} {label:<12} {point['merge_at']:>8.2f} {point['apart_at']:>8.2f} "
                  f"{len(point['merged']):>7} {len(point['apart']):>10} {len(point['left']):>5} "
                  f"{len(point['wrong_merge']):>11} {len(point['wrong_apart']):>11} "
                  f"{decided / point['n']:>7.0%}")
    print("\nthresholds chosen on v1+v2, applied to v3 unseen:")
    for design in designs:
        fitted = operating_point(subsets["v1+v2 (101)"], rows, design)
        held = operating_point(subsets["v3 (55)"], rows, design,
                               fitted["merge_at"], fitted["apart_at"])
        decided = len(held["merged"]) + len(held["apart"]) + len(held["wrong_apart"])
        print(f"  {design:<16} merge at {fitted['merge_at']:.2f}, apart at "
              f"{fitted['apart_at']:.2f}: merged {len(held['merged'])}, kept apart "
              f"{len(held['apart'])}, left {len(held['left'])}, wrong merges "
              f"{held['wrong_merge'] or 0}, true pairs kept apart {held['wrong_apart'] or 0}, "
              f"decided {decided / held['n']:.0%}")
    shipped = operating_point(cases, rows, "both orders", 0.95, 0.5)
    print(f"\nshipped (both orders, merge at 0.95, apart at 0.50): merged "
          f"{len(shipped['merged'])} of 81 true pairs, kept apart {len(shipped['apart'])} of 75, "
          f"waiting {len(shipped['left'])}, wrong merges {shipped['wrong_merge'] or 0}, "
          f"true pairs kept apart {shipped['wrong_apart'] or 0}")
    print(f"waiting: {shipped['left']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=pathlib.Path,
                        default=HERE / "results" / "identity_resolution_jev.jsonl")
    args = parser.parse_args()
    cases = load_cases()
    if args.cache.exists():
        judged = [json.loads(line) for line in args.cache.read_text(encoding="utf-8").splitlines()]
    else:
        decider = JevDecider(DecisionConfig(provider="jev",
                                            api_key=os.environ["TYPESAFE_API_KEY"]))
        judged = judge(cases, decider)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in judged)
                              + "\n", encoding="utf-8")
    rows = {r["id"]: r for r in judged}
    missing = [r["id"] for r in judged if not r.get("pair")]
    print(f"{len(cases)} labelled pairs; {len(missing)} without an answer {missing[:5]}")
    report(cases, rows)
