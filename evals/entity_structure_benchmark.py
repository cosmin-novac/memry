"""Entity structure: which names deserve to be hubs, and where parts belong?

Most of what an extractor calls an "entity" is not one. On the store this was
measured against, 3,314 names came out of 985 memories and 2,073 of them
appeared exactly once - counts, quoted fragments, ordinary nouns. Treating
every one as a hub is what fills a knowledge map with noise and an upkeep
queue with questions nobody can answer. ``intelligence/structure.py`` and the
name screen in ``intelligence/entities.py`` are the rules that sort them out,
and this is what measured those rules.

The datasets are private, because they are somebody's memories. The harness is
not: point ``--data`` at a directory of your own and every number below is
reproducible on your own store.

``--data DIR`` holds, one JSON object per line:

``d1_referents.jsonl``  ``{"id", "name", "type", "mentions", "snippet"}``
``labels_d1.jsonl``     ``{"id", "label"}`` with label in
                        thing | generic | value | role | event
``d2_homes.jsonl``      ``{"id", "name", "home", "home_type", "share",
                        "mentions", "other_anchors"}`` - one candidate home per
                        line, as ``derive_homes`` would have proposed it
``labels_d2.jsonl``     ``{"id", "home_correct"}`` with yes | no | unsure

and optionally ``screen.json``, a list of ``{"id", "value", "probabilities",
"confidence"}`` - what a decision provider answered for each d1 name. The
``ask`` subcommand writes that file; the rest only read it.

A label is one reader's judgement of what a name is, made without seeing what
any rule said about it. "thing" means a named referent you could later ask a
question about; "generic" an ordinary noun or topic; "value" a number, amount,
date or fragment.

Subcommands:

``mechanical``  ``non_referent_reason`` against the "value" label. This rule
                deletes nothing and costs nothing, so its recall matters less
                than its precision: a name it rejects never becomes an entity,
                so any "thing" it fires on is real harm and gets listed.
``screen``      the provider's verdict against the label, as a confusion
                table, plus a sweep of the gate from 0.50 to 0.95 counting how
                many "thing" names each threshold screens out. Pick the lowest
                gate that harms nothing, with headroom. ``SCREEN_GATE`` is 0.80
                because of this sweep.
``hubs``        ``is_hub`` against the "thing" label, with and without the
                provider's verdict. The verdict is what took the rule from 52%
                precision to 72% at 98% recall.
``homes``       precision of a proposed home, split by share threshold, by the
                type of the home, and by whether the part was seen once or
                more. ``HOME_MIN_SHARE`` and the organization rule come from
                these splits.
``ask``         runs ``SCREEN_CRITERIA`` over every d1 name through the
                configured decision provider and writes ``screen.json``. Needs
                ``MEMRY_DECISION_PROVIDER=jev`` and an API key; everything else
                here is offline.

Run:
    python evals/entity_structure_benchmark.py mechanical --data local/d
    python evals/entity_structure_benchmark.py ask        --data local/d
    python evals/entity_structure_benchmark.py screen     --data local/d
    python evals/entity_structure_benchmark.py hubs       --data local/d
    python evals/entity_structure_benchmark.py homes      --data local/d

Measured on a 985-memory store, 360 labelled names and 141 labelled homes:
the mechanical rule fires on 12 of the labelled names, 83% of them values and
none of them things; Jev agrees with the reader on 78% of names and screens
out no "thing" from a gate of 0.60 up; the hub rule goes from 60% precision at
78% recall without a verdict to 72% at 98% with one; and a home restricted to
a project or product at share >= 0.7, sole anchor when the part was seen once,
is 85% right against 68% unrestricted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from memry.config import Config  # noqa: E402
from memry.intelligence.entities import (  # noqa: E402
    SCREEN_CRITERIA,
    SCREEN_GATE,
    SCREEN_SKIPS,
    non_referent_reason,
)
from memry.intelligence.structure import (  # noqa: E402
    COMENTION_HOME_TYPES,
    HOME_MIN_SHARE,
    NAMED_THING_MIN,
    is_hub,
)
from memry.providers.decisions import Choice, JevDecider, LLMDecider  # noqa: E402
from memry.providers.llm import build_llm  # noqa: E402

#: Reader labels, and the screen verdict that means the same thing. "event" has
#: no verdict of its own: the screen was never asked to find events.
LABEL_TO_VERDICT = {
    "thing": "named_thing",
    "generic": "generic_topic",
    "value": "value_or_fragment",
    "role": "role",
}
LABELS = ("thing", "generic", "value", "role", "event")
VERDICTS = tuple(SCREEN_CRITERIA)
GATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


# ------------------------------------------------------------------ loading
def _jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path}. See the module docstring for the file layout.")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def load_d1(data: pathlib.Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Names with the evidence behind them, and one label each."""
    names = _by_id(_jsonl(data / "d1_referents.jsonl"))
    labels = {row["id"]: row["label"] for row in _jsonl(data / "labels_d1.jsonl")}
    shared = [i for i in names if i in labels]
    if not shared:
        sys.exit("no id appears in both d1_referents.jsonl and labels_d1.jsonl")
    return {i: names[i] for i in shared}, {i: labels[i] for i in shared}


def load_d2(data: pathlib.Path) -> tuple[dict[str, dict], dict[str, str]]:
    homes = _by_id(_jsonl(data / "d2_homes.jsonl"))
    labels = {row["id"]: row["home_correct"] for row in _jsonl(data / "labels_d2.jsonl")}
    shared = [i for i in homes if i in labels]
    if not shared:
        sys.exit("no id appears in both d2_homes.jsonl and labels_d2.jsonl")
    return {i: homes[i] for i in shared}, {i: labels[i] for i in shared}


def load_screen(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        sys.exit(f"missing {path}. Run the `ask` subcommand first, or pass --screen.")
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return {row["id"]: row for row in rows if row.get("value")}


def probability(row: dict) -> float:
    """What the provider put on the answer it gave, not on its favourite."""
    return float((row.get("probabilities") or {}).get(
        row["value"], row.get("confidence") or 0.0))


# --------------------------------------------------------------- mechanical
def run_mechanical(names: dict[str, dict], labels: dict[str, str]) -> None:
    print(f"\n=== mechanical rule (non_referent_reason) over {len(names)} names")
    fired = {i: reason for i in names
             if (reason := non_referent_reason(names[i]["name"]))}
    values = [i for i in names if labels[i] == "value"]
    hit_labels = Counter(labels[i] for i in fired)
    correct = hit_labels["value"]

    print(f"fires on        {len(fired)} of {len(names)}")
    print(f"precision       {correct}/{len(fired) or 1} = "
          f"{correct / max(len(fired), 1):.0%} against label 'value'")
    print(f"recall          {correct}/{len(values)} = "
          f"{correct / max(len(values), 1):.0%} of labelled values")
    print("by label       ", ", ".join(f"{label} {hit_labels[label]}"
                                       for label in LABELS if hit_labels[label]))
    print("by reason      ", dict(Counter(fired.values())))

    harm = [i for i in fired if labels[i] == "thing"]
    if harm:
        print(f"HARM: {len(harm)} name(s) labelled 'thing' are rejected outright:")
        for i in harm:
            print(f"  {names[i]['name']!r}  -> {fired[i]}")
    else:
        print("harm            none: no name labelled 'thing' is rejected")


# ------------------------------------------------------------------- screen
def run_screen(names: dict[str, dict], labels: dict[str, str],
               screen: dict[str, dict]) -> None:
    answered = [i for i in names if i in screen]
    print(f"\n=== name screen: {len(answered)} of {len(names)} names answered")

    table: dict[tuple[str, str], int] = Counter(
        (labels[i], screen[i]["value"]) for i in answered)
    width = max(len(v) for v in VERDICTS) + 2
    print("label".ljust(10) + "".join(v.ljust(width) for v in VERDICTS))
    for label in LABELS:
        row = [str(table[(label, verdict)]).ljust(width) for verdict in VERDICTS]
        print(label.ljust(10) + "".join(row))

    comparable = [i for i in answered if labels[i] in LABEL_TO_VERDICT]
    agree = sum(1 for i in comparable if LABEL_TO_VERDICT[labels[i]] == screen[i]["value"])
    print(f"exact agreement {agree}/{len(comparable)} = "
          f"{agree / max(len(comparable), 1):.0%} (events excluded: no verdict fits)")

    print(f"\ngate sweep over the skip verdicts {sorted(SCREEN_SKIPS)}")
    print(f"{'gate':>5} {'screened':>9} {'value/role':>11} {'generic':>8} "
          f"{'event':>6} {'THING':>6}  harm")
    for gate in GATES:
        out = [i for i in answered
               if screen[i]["value"] in SCREEN_SKIPS and probability(screen[i]) >= gate]
        counted = Counter(labels[i] for i in out)
        harmed = [names[i]["name"] for i in out if labels[i] == "thing"]
        flag = ", ".join(repr(n) for n in harmed[:6]) if harmed else ""
        mark = " <-" if gate == SCREEN_GATE else "   "
        print(f"{gate:>5.2f}{mark}{len(out):>6} {counted['value'] + counted['role']:>11} "
              f"{counted['generic']:>8} {counted['event']:>6} "
              f"{counted['thing']:>6}  {flag}")
    print(f"(<- is SCREEN_GATE, currently {SCREEN_GATE:.2f})")


# --------------------------------------------------------------------- hubs
def _hub_report(label: str, names: dict[str, dict], labels: dict[str, str],
                chosen: list[str]) -> None:
    things = [i for i in names if labels[i] == "thing"]
    true_positive = sum(1 for i in chosen if labels[i] == "thing")
    missed = Counter(labels[i] for i in chosen if labels[i] != "thing")
    print(f"  {label:<34} hubs {len(chosen):>3}  precision "
          f"{true_positive / max(len(chosen), 1):>4.0%}  recall "
          f"{true_positive / max(len(things), 1):>4.0%}   "
          + " ".join(f"{k} {v}" for k, v in sorted(missed.items())))


def run_hubs(names: dict[str, dict], labels: dict[str, str],
             screen: dict[str, dict] | None) -> None:
    things = sum(1 for i in names if labels[i] == "thing")
    print(f"\n=== hub rule (is_hub) over {len(names)} names, {things} labelled 'thing'")
    print("the dataset carries no relation counts, so every name is scored with "
          "relations=0;\nis_hub ignores them except as evidence that a name exists "
          "at all.")

    def mentions(i: str) -> int:
        return int(names[i].get("mentions") or 0)

    print("\nwithout the provider's verdict:")
    _hub_report("is_hub(type, mentions)", names, labels,
                [i for i in names if is_hub(names[i]["type"], mentions(i), 0)])
    _hub_report("anchor type only", names, labels,
                [i for i in names if is_hub(names[i]["type"], 0, 1)])
    _hub_report("two or more memories only", names, labels,
                [i for i in names if mentions(i) >= 2])

    if not screen:
        print("\n(no screen.json: run `ask`, or pass --screen, for the rest)")
        return
    answered = {i: names[i] for i in names if i in screen}
    answered_labels = {i: labels[i] for i in answered}
    print(f"\nwith the provider's verdict ({len(answered)} answered):")
    _hub_report("is_hub(type, mentions, screen)", answered, answered_labels,
                [i for i in answered
                 if is_hub(answered[i]["type"], mentions(i), 0,
                           {"verdict": screen[i]["value"],
                            "probability": probability(screen[i])})])
    _hub_report("is_hub, same names, no verdict", answered, answered_labels,
                [i for i in answered if is_hub(answered[i]["type"], mentions(i), 0)])
    _hub_report(f"verdict named_thing >= {NAMED_THING_MIN} only", answered,
                answered_labels,
                [i for i in answered if screen[i]["value"] == "named_thing"
                 and probability(screen[i]) >= NAMED_THING_MIN])


# -------------------------------------------------------------------- homes
def _precision(labels: dict[str, str], ids: list[str]) -> str:
    counted = Counter(labels[i] for i in ids)
    decided = counted["yes"] + counted["no"]
    return (f"n={len(ids):>3}  precision {counted['yes'] / max(decided, 1):>4.0%}  "
            f"(yes {counted['yes']:>3}, no {counted['no']:>3}, "
            f"unsure {counted['unsure']:>2})")


def run_homes(homes: dict[str, dict], labels: dict[str, str]) -> None:
    print(f"\n=== proposed homes: {len(homes)} labelled candidates")
    print(f"  every candidate                    {_precision(labels, list(homes))}")

    print("\nby share of the part's memories the anchor appears in:")
    for threshold in (0.5, 0.6, HOME_MIN_SHARE, 0.8, 0.9, 1.0):
        chosen = [i for i in homes if float(homes[i].get("share") or 0.0) >= threshold]
        mark = " <-" if threshold == HOME_MIN_SHARE else "   "
        print(f"  share >= {threshold:.2f}{mark}                     "
              f"{_precision(labels, chosen)}")
    print(f"(<- is HOME_MIN_SHARE, currently {HOME_MIN_SHARE:.2f})")

    print("\nby the type of the proposed home:")
    by_type: dict[str, list[str]] = defaultdict(list)
    for i in homes:
        by_type[str(homes[i].get("home_type") or "untyped")].append(i)
    for home_type, ids in sorted(by_type.items()):
        note = "" if home_type in COMENTION_HOME_TYPES else "  (needs a stated relation)"
        print(f"  home is a {home_type:<24}{_precision(labels, ids)}{note}")

    print("\nby how often the part itself was seen:")
    once = [i for i in homes if int(homes[i].get("mentions") or 0) <= 1]
    repeated = [i for i in homes if int(homes[i].get("mentions") or 0) >= 2]
    print(f"  seen once                          {_precision(labels, once)}")
    print(f"  seen twice or more                 {_precision(labels, repeated)}")
    alone = [i for i in once if not homes[i].get("other_anchors")]
    crowded = [i for i in once if homes[i].get("other_anchors")]
    print(f"  seen once, sole anchor             {_precision(labels, alone)}")
    print(f"  seen once, other anchors present   {_precision(labels, crowded)}")

    shipped = [
        i for i in homes
        if str(homes[i].get("home_type")) in COMENTION_HOME_TYPES
        and float(homes[i].get("share") or 0.0) >= HOME_MIN_SHARE
        and (int(homes[i].get("mentions") or 0) >= 2 or not homes[i].get("other_anchors"))
    ]
    print(f"\n  what the shipped rule keeps        {_precision(labels, shipped)}")


# ---------------------------------------------------------------------- ask
def run_ask(names: dict[str, dict], out_path: pathlib.Path, workers: int) -> None:
    """Put the shipped screen question to the configured provider, once per name."""
    config = Config.load()
    llm = build_llm(config.llm)
    decider = (JevDecider(config.decision) if config.decision.provider == "jev"
               else LLMDecider(llm))
    if not decider.available:
        sys.exit(f"decision provider {decider.name!r} is not available; set "
                 "MEMRY_DECISION_PROVIDER and its API key")
    print(f"asking {decider.name} about {len(names)} names, {workers} at a time")

    def one(entity_id: str) -> dict:
        row = names[entity_id]
        started = time.time()
        question = Choice(instructions=f'In this memory, what is "{row["name"]}"?',
                          criteria=SCREEN_CRITERIA)
        try:
            answer = decider.decide(
                "A memory from a personal long-term memory store: "
                + str(row.get("snippet") or ""),
                {"s": question},
            )["s"]
        except Exception as exc:  # a provider hiccup is a result, not a crash
            return {"id": entity_id, "available": False, "value": None,
                    "error": str(exc)[:120], "ms": (time.time() - started) * 1000}
        return {
            "id": entity_id,
            "available": answer.available,
            "value": answer.value,
            "probabilities": answer.probabilities,
            "confidence": round(float(answer.confidence), 3),
            "ms": round((time.time() - started) * 1000),
        }

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, list(names)))
    finally:
        decider.close()
        llm.close()

    out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    answered = [r for r in rows if r.get("available")]
    print(f"wrote {out_path} - {len(answered)} answered, "
          f"{len(rows) - len(answered)} abstained or failed")
    print("verdicts:", dict(Counter(r["value"] for r in answered)))


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=("mechanical", "screen", "hubs", "homes", "ask"))
    parser.add_argument("--data", required=True, type=pathlib.Path,
                        help="directory holding the jsonl datasets and labels")
    parser.add_argument("--screen", type=pathlib.Path, default=None,
                        help="provider answers (default: <data>/screen.json)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel provider calls for `ask`")
    args = parser.parse_args()

    screen_path = args.screen or (args.data / "screen.json")
    if args.command == "homes":
        run_homes(*load_d2(args.data))
        return

    names, labels = load_d1(args.data)
    if args.command == "mechanical":
        run_mechanical(names, labels)
    elif args.command == "ask":
        run_ask(names, screen_path, args.workers)
    elif args.command == "screen":
        run_screen(names, labels, load_screen(screen_path))
    elif args.command == "hubs":
        run_hubs(names, labels,
                 load_screen(screen_path) if screen_path.exists() else None)


if __name__ == "__main__":
    main()
