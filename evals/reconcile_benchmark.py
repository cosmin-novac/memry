"""Reconcile on updates: does a later save retire, merge or keep what an earlier
one said, whichever run each save belongs to?

**The cases.** 60 synthetic cases (``CASES``) of 2 or 3 saves by two invented
speakers, Maria and Tom: a value that changes, a correction, the same fact
reworded, an exact restatement, a contradiction nobody announces, an added
detail, a plan that happened, a time-bound fact, a recurring event, a
preference that changes, a project's status, two events or things of one
kind (two trips, injuries, deals, paintings, games), a relative time in a
fact that is merged (with LoCoMo conv-41's two road trips in fictional form,
R1), claims about one subject that must stay one memory each (S1 to S3,
graded "joined" when one memory holds two of them), and a detail of the same
claim that must merge (T1, T2). Each case runs twice on fresh
in-memory stores through ``MemoryStore.add(infer=True)``, saved as the LoCoMo
runner saves a session: one message whose role is the speaker, with the
context "conversation between Maria and Tom, <date>" and the save's date as
``created_at`` and ``now``; later saves come 1 to 180 days after the first.
Layout "same": every save in run r1. Layout "diff": save i in run r<i>, one
run per save, as a benchmark that gives every session its own run does.

**The grade** (``grade``) reads the final store and one search for the latest
value, by rule:

* ``replace``: the latest value is in a live memory that the search finds
  among its first five, above any memory stating only the older value; no
  live memory states only an older value (it is superseded, kept as history,
  or merged into a text that also says the new one). "held" when both are
  live and the new one waits for a person as a conflict.
* ``one``: the later saves left no second live copy (no more live memories
  of the fact than the first save made), and the search finds it.
* ``merge``: the same, and a live memory holds the added detail.
* ``separate``: at least ``n`` live memories (events that recur stay apart).
* ``keep``: the new fact is live and found, and the older one is live or kept
  as history (a time-bound fact, an event that stays true).

``stale`` names a present-tense wording of an older value that a merged text
must not keep; ``dated_ok`` lets a live older memory that carries its own
date count as history (an event that stays true on its date). ``misdated``
names a time that a memory holding both facts and written at a later save
must not state, unless ``anchored`` finds the date it names beside it: such a
memory is dated at its own save, so "this week" or "the previous year" there
reads as another time. It is checked first, whatever the rule, and fails the
case as "misdated".

**Labelled answers.** Each case also names the reconcile answers that fit each
later save (``ok``), and a set of labelled pairs from a conversation
benchmark (``pairs``, read from a path: those texts are not synthetic and
stay outside this repository) does the same for single pairs. ``bars`` reads
the answers of both and prints, for each of SAME, MORE, CHANGED and WRONG,
the confidence from which no answer of that kind was wrong: the measured bars
of ``JevDecider.reconcile_bars``.

Commands (``TYPESAFE_API_KEY`` for Jev, ``OPENAI_API_KEY`` for extraction and
embeddings; every call counted in ``--ledger LEDGER.sqlite`` and capped per
group, ``--jev-cap`` and ``--chat-cap``):

    python evals/reconcile_benchmark.py cases OUT.json --ledger L [--only A1,B2] [--layouts same,diff]
    python evals/reconcile_benchmark.py replay CASES.json OUT.json --ledger L [--judge text]
    python evals/reconcile_benchmark.py pairs PAIRS.json OUT.json --ledger L [--judge text]
    python evals/reconcile_benchmark.py claims OUT.json --ledger L
    python evals/reconcile_benchmark.py claims-text OUT.json --ledger L
    python evals/reconcile_benchmark.py merges OUT.json --ledger L [--runs 3]
    python evals/reconcile_benchmark.py merge-table OUT.json [OUT.json ...]
    python evals/reconcile_benchmark.py claim-table OUT.json [OUT.json ...]
    python evals/reconcile_benchmark.py table CASES.json [CASES.json ...]
    python evals/reconcile_benchmark.py bars ANSWERS.json [ANSWERS.json ...]

``replay`` asks the reconcile question alone, with the current wording, on
every state a ``cases`` run logged, for measuring a wording without saving
anything. It and ``pairs`` write answers; ``cases`` writes stores, grades and
the answers given on the way. ``merges`` asks the installed memry's merge
writer (the text model, ``OPENAI_API_KEY``) to join each fixed pair of
``MERGE_PAIRS`` (texts as extraction wrote them, with their dates), and
grades the text it writes: two things of one kind must not be joined, and
no time may move (``misdated``). LoCoMo conv-41 is P1. ``CLAIM_PAIRS`` (Q, a
new claim about the memory's subject, which must stay apart; D, a detail of
the same claim, which must merge) are among those pairs, and ``claims`` asks
Jev the reconcile question on them (``claims-text``: the text model as judge,
as memry asks it where no decision provider answers; ``--judge text`` does
the same for ``replay`` and ``pairs``).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

BASE = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)  # a Monday
USER = "u1"
SPEAKERS = "conversation between Maria and Tom"
ACTIONS = ("NEW", "SAME", "MORE", "CHANGED", "WRONG")
#: The answers of the question asked before this benchmark, as the five read.
OLD_WORDS = {"ADD": "NEW", "NONE": "SAME", "UPDATE": "CHANGED", "DELETE": "WRONG"}

# ok sets by kind of case (per later save, unless the case lists its own)
_CHANGE = ("CHANGED", "WRONG")
_SAME = ("SAME", "MORE")


def case(id: str, kind: str, expect: str, saves: list, *, ok: list | None = None,
         **fields: Any) -> dict[str, Any]:
    return {"id": id, "kind": kind, "expect": expect,
            "saves": [(day, speaker, text) for day, speaker, text in saves],
            "ok": ok, **fields}


CASES: list[dict[str, Any]] = [
    # -- a value that changes ------------------------------------------------------
    case("A1", "value change", "replace", [
        (0, "Maria", "I just signed the lease on a flat on Elm Street in Boston, I live there now."),
        (42, "Maria", "Big news, I moved to Denver last weekend! My new place is on Pine Avenue.")],
        new=r"denver|pine", old=r"boston|elm", dated_ok=True,
        topic=r"live|lives|moved|flat|place|lease|denver|boston",
        query="Where does Maria live?", ok=[_CHANGE]),
    case("A2", "value change", "replace", [
        (0, "Maria", "I work as a nurse at Northgate Hospital."),
        (42, "Maria", "I left Northgate Hospital last month; I'm now a nursing instructor at the community college.")],
        new=r"instructor|community college",
        old=r"nurse at northgate|works? (as a nurse )?at northgate",
        topic=r"northgate|nurs|instructor", query="What is Maria's job?", ok=[_CHANGE]),
    case("A3", "value change", "replace", [
        (0, "Tom", "My gym membership costs $40 a month."),
        (42, "Tom", "The gym raised its price, my membership is $55 a month now.")],
        new=r"\$?55", old=r"\$?40", topic=r"gym|membership",
        query="How much does Tom's gym membership cost?", ok=[_CHANGE]),
    case("A4", "value change", "replace", [
        (0, "Maria", "I drive a 2012 Arden hatchback."),
        (42, "Maria", "I sold the Arden and bought a used Tamber SUV."),
        (80, "Maria", "The Tamber was a lemon, so I traded it in for a Brisa wagon.")],
        new=r"brisa", old=r"arden|tamber",
        stale=r"\bdrives\b.*\b(arden|tamber)\b|\b(owns|has)\b.*\b(arden|tamber)\b",
        topic=r"arden|tamber|brisa|car|drive",
        query="What car does Maria drive?", ok=[_CHANGE, _CHANGE]),
    case("A5", "value change", "replace", [
        (0, "Tom", "My salary at Corlan is $70,000 a year."),
        (42, "Tom", "I got a raise, I make $78,000 a year at Corlan now.")],
        new=r"78,?000", old=r"70,?000", topic=r"salary|corlan|\$7",
        query="What is Tom's salary?", ok=[_CHANGE]),
    # -- a correction ------------------------------------------------------------
    case("B1", "correction", "replace", [
        (0, "Maria", "My sister's wedding is on Saturday, June 13."),
        (3, "Maria", "Correction about my sister's wedding: it's actually on Sunday, June 14, not Saturday.")],
        new=r"june 14|06-14|sunday", old=r"june 13|06-13|saturday", topic=r"wedding",
        query="When is Maria's sister's wedding?", ok=[_CHANGE]),
    case("B2", "correction", "replace", [
        (0, "Tom", "I met my business partner Raj at a conference in Chicago."),
        (3, "Tom", "Actually I got it wrong earlier: I met Raj at a conference in Detroit, not Chicago.")],
        new=r"detroit", old=r"chicago", topic=r"raj|conference", query="Where did Tom meet Raj?",
        ok=[_CHANGE]),
    case("B3", "correction", "replace", [
        (0, "Maria", "My daughter Lily is 7."),
        (3, "Maria", "Oops, I said Lily is 7, but she's actually 8, she turned 8 in January.")],
        new=r"\b8\b|eight", old=r"\b7\b|seven", topic=r"lily", query="How old is Maria's daughter Lily?",
        ok=[_CHANGE]),
    case("B4", "correction", "replace", [
        (0, "Tom", "My dentist appointment is on Monday at 3pm."),
        (1, "Tom", "Actually the dentist appointment is on Tuesday at 3pm, I mixed up the days.")],
        new=r"tuesday|03-03|03-10", old=r"monday|03-02|03-09", topic=r"dentist",
        query="When is Tom's dentist appointment?", ok=[_CHANGE]),
    # -- the same fact reworded --------------------------------------------------
    case("C1", "reworded", "one", [
        (0, "Maria", "I'm a vegetarian, I haven't eaten meat in ten years."),
        (42, "Maria", "Just so you remember, I don't eat meat; I've been vegetarian for about a decade.")],
        topic=r"vegetarian|meat", query="Does Maria eat meat?", new=r"vegetarian|meat", ok=[_SAME]),
    case("C2", "reworded", "one", [
        (0, "Tom", "I'm allergic to penicillin."),
        (42, "Tom", "Reminder: penicillin gives me a bad allergic reaction, so I can't take it.")],
        topic=r"penicillin", query="What is Tom allergic to?", new=r"penicillin", ok=[_SAME]),
    case("C3", "reworded", "one", [
        (0, "Maria", "My brother Carlos lives in Madrid."),
        (42, "Maria", "Carlos, my brother, is based in Madrid.")],
        topic=r"carlos", query="Where does Maria's brother live?", new=r"madrid", ok=[_SAME]),
    case("C4", "reworded", "one", [
        (0, "Tom", "I play the cello in a community orchestra."),
        (42, "Tom", "I'm a cellist in our local community orchestra."),
        (80, "Tom", "Playing cello in the community orchestra is my big hobby.")],
        topic=r"cell|orchestra", query="What instrument does Tom play?", new=r"cell",
        ok=[_SAME, _SAME]),
    # -- an exact restatement ----------------------------------------------------
    case("D1", "exact restatement", "one", [
        (0, "Maria", "I have a golden retriever named Biscuit."),
        (42, "Maria", "I have a golden retriever named Biscuit.")],
        topic=r"biscuit|golden retriever", query="What dog does Maria have?", new=r"biscuit",
        ok=[_SAME]),
    case("D2", "exact restatement", "one", [
        (0, "Tom", "My favorite band is the Glass Harbors."),
        (42, "Tom", "My favorite band is the Glass Harbors.")],
        topic=r"glass harbors|favorite band", query="What is Tom's favorite band?",
        new=r"glass harbors",
        ok=[_SAME]),
    case("D3", "exact restatement", "one", [
        (0, "Maria", "I work from home on Fridays."),
        (42, "Maria", "I work from home on Fridays.")],
        topic=r"from home|fridays", query="Which day does Maria work from home?", new=r"friday",
        ok=[_SAME]),
    case("D4", "exact restatement", "one", [
        (0, "Tom", "I speak fluent Portuguese."),
        (42, "Tom", "I speak fluent Portuguese."),
        (80, "Tom", "I speak fluent Portuguese.")],
        topic=r"portuguese", query="What languages does Tom speak?", new=r"portuguese",
        ok=[_SAME, _SAME]),
    # -- a contradiction nobody announces ----------------------------------------
    case("E1", "contradiction", "replace", [
        (0, "Maria", "I have two kids."),
        (42, "Maria", "All three of my kids go to the same school now.")],
        new=r"three", old=r"\btwo\b", topic=r"kid|child", query="How many kids does Maria have?",
        ok=[_CHANGE]),
    case("E2", "contradiction", "replace", [
        (0, "Tom", "My mother's name is Helen."),
        (42, "Tom", "My mom, Margaret, is visiting next week.")],
        new=r"margaret", old=r"helen", topic=r"mother|mom|helen|margaret",
        query="What is the name of Tom's mother?", ok=[_CHANGE]),
    case("E3", "contradiction", "replace", [
        (0, "Maria", "I've never been to Japan."),
        (42, "Maria", "When I was in Tokyo back in 2019, I loved the food markets.")],
        new=r"tokyo", old=r"never been", topic=r"japan|tokyo", query="Has Maria been to Japan?",
        ok=[_CHANGE]),
    case("E4", "contradiction", "replace", [
        (0, "Tom", "I'm an only child."),
        (42, "Tom", "My older brother Sam is coming to stay with me.")],
        new=r"brother|sam", old=r"only child", topic=r"only child|brother|sibling|sam",
        query="Does Tom have siblings?", ok=[_CHANGE]),
    # -- an added detail ---------------------------------------------------------
    case("F1", "added detail", "merge", [
        (0, "Maria", "I'm learning Spanish."),
        (42, "Maria", "I'm learning Spanish with a tutor twice a week, on Tuesdays and Thursdays.")],
        new=r"tutor|twice|tuesdays", old=r"spanish", topic=r"spanish",
        query="How is Maria learning Spanish?", ok=[("MORE",)]),
    case("F2", "added detail", "merge", [
        (0, "Tom", "I have a dog named Rex."),
        (42, "Tom", "Rex, my dog, is a three-year-old German shepherd.")],
        new=r"german shepherd|three-year", old=r"rex", topic=r"rex", query="What breed is Tom's dog?",
        ok=[("MORE",)]),
    case("F3", "added detail", "merge", [
        (0, "Maria", "I'm writing a novel."),
        (42, "Maria", "I'm writing a novel, a mystery set in 1920s Lisbon.")],
        new=r"mystery|lisbon", old=r"novel", topic=r"novel", query="What is Maria's novel about?",
        ok=[("MORE",)]),
    case("F4", "added detail", "merge", [
        (0, "Tom", "I work at Corlan."),
        (42, "Tom", "I work at Corlan as a data engineer on the payments team.")],
        new=r"data engineer|payments", old=r"corlan", topic=r"corlan",
        query="What does Tom do at Corlan?", ok=[("MORE",)]),
    # -- a plan that happened ----------------------------------------------------
    case("G1", "plan happened", "replace", [
        (0, "Maria", "I'm planning to run the Harbor City Marathon in April."),
        (50, "Maria", "I ran the Harbor City Marathon on Monday and finished in 4 hours 10 minutes!")],
        new=r"\bran\b|finished|4 hours", old=r"plan|will run|intend", topic=r"marathon",
        query="Did Maria run the Harbor City Marathon?", ok=[("CHANGED", "MORE")]),
    case("G2", "plan happened", "replace", [
        (0, "Tom", "We're thinking about adopting a cat this spring."),
        (42, "Tom", "We adopted a cat last weekend, her name is Luna.")],
        new=r"adopted|luna", old=r"thinking|consider|plan", topic=r"cat|luna",
        query="Does Tom have a cat?", ok=[("CHANGED", "MORE")]),
    case("G3", "plan happened", "replace", [
        (0, "Maria", "I'm going to apply for a master's program in public health."),
        (42, "Maria", "I got accepted into the MPH program at Linden University, starting in the fall."),
        (180, "Maria", "I started my MPH classes at Linden University this week.")],
        new=r"started", old=r"apply|accepted|will start|starting in the fall",
        topic=r"mph|public health|linden|master", query="Where is Maria studying?",
        ok=[("CHANGED", "MORE"), ("CHANGED", "MORE")]),
    case("G4", "plan happened", "replace", [
        (0, "Tom", "I plan to repaint the kitchen green next month."),
        (42, "Tom", "Finished repainting the kitchen; I went with blue in the end instead of green.")],
        new=r"blue", old=r"green", topic=r"kitchen", query="What color is Tom's kitchen?",
        ok=[("CHANGED", "MORE")]),
    # -- a time-bound fact -------------------------------------------------------
    case("H1", "time-bound", "keep", [
        (0, "Maria", "I'm in Paris this week for work."),
        (42, "Maria", "I'm in Rome this week visiting friends.")],
        topic=r"paris|rome", new=r"rome", old=r"paris", query="Where is Maria this week?",
        ok=[("CHANGED",)]),
    case("H2", "time-bound", "keep", [
        (0, "Tom", "I'm working from the Berlin office this month."),
        (42, "Tom", "I'm back at the London office for good now.")],
        new=r"london", old=r"berlin", topic=r"berlin|london|office",
        query="Which office does Tom work from?", ok=[("CHANGED",)]),
    case("H3", "time-bound", "replace", [
        (0, "Maria", "I'm in Paris until Friday."),
        (2, "Maria", "I extended my Paris trip until next Tuesday.")],
        new=r"tuesday|03-10|extended", old=r"friday|03-06", topic=r"paris",
        query="Until when is Maria in Paris?", ok=[("CHANGED", "WRONG", "MORE")]),
    case("H4", "time-bound", "one", [
        (0, "Tom", "I'm on vacation in Lisbon this week."),
        (2, "Tom", "Still on vacation in Lisbon this week, and the weather has been great.")],
        topic=r"lisbon", new=r"lisbon", query="Where is Tom on vacation?", ok=[_SAME]),
    # -- a recurring event (must stay separate) ----------------------------------
    case("I1", "recurring event", "separate", [
        (0, "Maria", "I went to a yoga class this morning."),
        (7, "Maria", "I went to a yoga class this morning.")],
        n=2, topic=r"yoga", new=r"yoga", query="When did Maria go to yoga?", ok=[()]),
    case("I2", "recurring event", "separate", [
        (0, "Tom", "I had dinner with my parents yesterday."),
        (28, "Tom", "I had dinner with my parents again yesterday, and my mom made lasagna.")],
        n=2, topic=r"dinner", new=r"lasagna", query="When did Tom have dinner with his parents?",
        ok=[()]),
    case("I3", "recurring event", "separate", [
        (0, "Maria", "I ran a 5K race in Central Park today."),
        (35, "Maria", "I ran another 5K race in Central Park today and beat my time by a minute.")],
        n=2, topic=r"5k", new=r"beat|another", query="How many 5K races has Maria run in Central Park?",
        ok=[()]),
    case("I4", "recurring event", "separate", [
        (0, "Tom", "I had a doctor's appointment today about my knee."),
        (30, "Tom", "I had a follow-up appointment with the doctor about my knee today.")],
        n=2, topic=r"knee", new=r"follow-up", query="When did Tom see the doctor about his knee?",
        ok=[()]),
    case("I5", "recurring event", "separate", [
        (0, "Maria", "I went to a yoga class this morning."),
        (7, "Maria", "I went to a yoga class this morning."),
        (14, "Maria", "I went to a yoga class this morning.")],
        n=3, topic=r"yoga", new=r"yoga", query="How often does Maria go to yoga?", ok=[(), ()]),
    # -- a preference that changes -----------------------------------------------
    case("J1", "preference change", "replace", [
        (0, "Maria", "I love coffee, I drink three cups a day."),
        (42, "Maria", "I quit coffee last month; I only drink green tea now.")],
        new=r"quit|green tea", old=r"loves? coffee|three cups",
        stale=r"\bloves coffee|\bdrinks three cups",
        topic=r"coffee|tea", query="What does Maria drink?", ok=[_CHANGE]),
    case("J2", "preference change", "replace", [
        (0, "Tom", "My favorite TV show is Night Ledger."),
        (42, "Tom", "My new favorite show is Paper Orchard, it's even better than Night Ledger.")],
        new=r"paper orchard", old=r"favorite (tv )?show is night ledger|night ledger is",
        topic=r"favorite|show", query="What is Tom's favorite TV show?", ok=[_CHANGE]),
    case("J3", "preference change", "replace", [
        (0, "Maria", "I prefer working late at night."),
        (42, "Maria", "I've become a morning person; I do my best work at 6am now.")],
        new=r"morning|6", old=r"late at night|night owl", topic=r"night|morning|work",
        query="When does Maria prefer to work?", ok=[_CHANGE]),
    case("J4", "preference change", "replace", [
        (0, "Tom", "I hate running."),
        (42, "Tom", "I've really come to enjoy running; I go three times a week now.")],
        new=r"enjoy|three times", old=r"hates?|dislike", topic=r"run",
        query="Does Tom like running?", ok=[_CHANGE]),
    # -- the status of a project -------------------------------------------------
    case("K1", "project status", "replace", [
        (0, "Maria", "My startup's app is in private beta."),
        (42, "Maria", "We launched the app publicly last week."),
        (80, "Maria", "Our app just passed 10,000 users.")],
        new=r"10,?000|launched", old=r"private beta", topic=r"app|beta|launch",
        query="What is the status of Maria's app?", ok=[("CHANGED", "MORE"), ("CHANGED", "MORE")]),
    case("K2", "project status", "replace", [
        (0, "Tom", "I've started renovating the bathroom."),
        (42, "Tom", "The bathroom renovation is finished.")],
        new=r"finished|complete", old=r"started|renovating|is renovating", dated_ok=True,
        topic=r"bathroom", query="Is Tom's bathroom renovation done?", ok=[("CHANGED", "MORE")]),
    case("K3", "project status", "replace", [
        (0, "Maria", "I'm on chapter 3 of my thesis."),
        (42, "Maria", "I'm on chapter 5 of my thesis now.")],
        new=r"chapter 5|fifth", old=r"chapter 3|third", topic=r"thesis|chapter",
        query="Which chapter of her thesis is Maria on?", ok=[_CHANGE]),
    case("K4", "project status", "replace", [
        (0, "Tom", "The Harbor Bridge project I manage is behind schedule."),
        (42, "Tom", "The Harbor Bridge project caught up and is on schedule now.")],
        new=r"on schedule|caught up", old=r"behind schedule", topic=r"harbor bridge",
        query="Is the Harbor Bridge project on schedule?", ok=[_CHANGE]),
    case("K5", "project status", "keep", [
        (0, "Maria", "I submitted my grant proposal to the Science Fund."),
        (42, "Maria", "The Science Fund rejected my grant proposal.")],
        new=r"reject", old=r"submitted", topic=r"grant|science fund",
        query="What happened to Maria's Science Fund grant proposal?", ok=[("CHANGED", "MORE")]),
    # -- two events or things of one kind (must stay separate) -------------------
    # Each pair is two trips, injuries, deals, paintings or games, the second
    # told later and often about an earlier time. In the LoCoMo stores Jev
    # answered such pairs MORE at 0.80 to 0.97 and one text of both was
    # written, the second event taking the first one's date.
    case("L1", "two of one kind", "separate", [
        (0, "Maria", "We took the kids camping at Lake Arden last weekend; they loved the canoe."),
        (40, "Maria", "I'll never forget our camping trip last summer, when we watched the "
                      "northern lights from the tent.")],
        n=2, topic=r"camp", new=r"northern lights", query="When did Maria see the northern lights?",
        ok=[()]),
    case("L2", "two of one kind", "separate", [
        (0, "Tom", "I sprained my wrist playing volleyball yesterday; the doctor says it's not "
                   "serious."),
        (5, "Tom", "Last season I broke my ankle and needed six weeks of physical therapy before "
                   "I could play again.")],
        n=2, topic=r"wrist|ankle|injur|sprain", new=r"ankle",
        query="When did Tom break his ankle?", ok=[()]),
    case("L3", "two of one kind", "separate", [
        (0, "Tom", "I just signed a sponsorship deal with Arvo Sports for running shoes."),
        (150, "Tom", "Last week I signed a deal with Pinecrest, an outdoor gear company; they sent "
                     "me a tent and hiking boots.")],
        n=2, topic=r"deal|sponsor|arvo|pinecrest", new=r"pinecrest",
        query="When did Tom sign with Pinecrest?", ok=[()]),
    case("L4", "two of one kind", "separate", [
        (0, "Maria", "I finished a painting of the harbor at sunset for the art fair."),
        (30, "Maria", "Here's another painting I made, 'Quiet Morning': a woman reading by a "
                      "window.")],
        n=2, topic=r"paint", new=r"quiet morning", query="What paintings has Maria made?",
        ok=[()]),
    case("L5", "two of one kind", "separate", [
        (0, "Tom", "Last night I scored 30 points, my career high, and we beat the Hawks."),
        (150, "Tom", "Friday's game against our rivals was a memorable night: I had twelve "
                     "assists and the arena was electric.")],
        n=2, topic=r"game|points|assists|hawks|rival", new=r"assists",
        query="When did Tom have twelve assists?", ok=[()]),
    # -- a relative time in a merged fact ----------------------------------------
    # One thing with a detail added later (MORE is right), where one text
    # carries a time relative to the day it was said. The merged memory is
    # dated at the later save, so a memory written then may keep the older
    # text's relative time (``misdated``) only beside the date it names
    # (``anchored``).
    case("M1", "relative time", "merge", [
        (0, "Maria", "I just started aerial yoga this week, it's great!"),
        (180, "Maria", "My favorite part of aerial yoga is the upside-down poses; they make me "
                       "feel free and light.")],
        new=r"upside|free and light", old=r"aerial yoga", topic=r"aerial yoga",
        misdated=r"\b(this week|recently|just started)\b",
        anchored=r"\bmarch 2026\b|\b2026-03|\b2 march\b|\bmarch 2\b",
        query="When did Maria start aerial yoga?", ok=[("MORE",)]),
    case("M2", "relative time", "merge", [
        (0, "Tom", "I'm getting ready for the Riverside chess tournament next month."),
        (30, "Tom", "For the Riverside chess tournament I've been practicing endgames every "
                    "night.")],
        new=r"endgame", old=r"riverside|tournament", topic=r"chess|riverside",
        misdated=r"\bnext month\b", anchored=r"\bapril\b|\b2026-04",
        query="When is the Riverside chess tournament?", ok=[("MORE",)]),
    # -- regression: LoCoMo conv-41 in fictional form ----------------------------
    # A family road trip that ended the day before a December save, then, in
    # April, "a road trip we took last year" up the coast. Jev answered MORE at
    # 0.96 and the merged text read "returned from a family road trip on
    # 2022-12-16 ... the previous year's road trip explored the coast", so a
    # model dated the coast trip a year too early. Whatever the answer, a
    # memory written in April that holds both trips must give the coast trip
    # its year, 2026, and not a year relative to the December trip.
    case("R1", "relative time", "keep", [
        (290, "Tom", "Hey Maria! Just got back from a family road trip yesterday, it was fun!"),
        (404, "Tom", "This photo reminds me of a road trip we took last year; we explored the "
                     "coast up north and hit some cool national parks.")],
        new=r"coast", old=r"family road trip", topic=r"road trip|coast",
        misdated=r"\b(previous|prior|last) year\b|\byear before\b|\b2025\b|\byesterday\b",
        anchored=r"(?<![\d-])2026(?![\d-])",
        query="When did Tom take the road trip up the coast?", ok=[()]),
    # -- claims about one subject (must stay separate) ---------------------------
    # Each save states another claim, position or decision about one thing (a
    # thesis, a book, a project). A merge folds each new one into the memory
    # before it, so one memory ends up holding every claim, which no one vector
    # matches well. Each claim must stay a memory of its own: no live memory
    # may hold two of ``claims`` (graded "joined").
    case("S1", "claims about one subject", "separate", [
        (0, "Maria", "The central claim of my thesis is that small language models can match "
                     "large ones on narrow tasks."),
        (14, "Maria", "My thesis also argues that benchmark contamination explains most of the "
                      "reported gains of large models."),
        (30, "Maria", "In my thesis I explicitly reject the idea that scale alone produces "
                      "reasoning.")],
        n=3, claims=[r"narrow tasks", r"contamination", r"scale alone"],
        topic=r"thesis|narrow tasks|contamination|scale alone", new=r"scale alone",
        query="What does Maria's thesis argue?", ok=[(), ()]),
    case("S2", "claims about one subject", "separate", [
        (0, "Tom", "I'm reading 'Slow Rivers'; its main argument is that 20th-century dams did "
                   "more harm than good."),
        (10, "Tom", "'Slow Rivers' also claims that beavers restore wetlands faster than "
                    "engineered projects do."),
        (20, "Tom", "The author of 'Slow Rivers' rejects fish ladders as a fix for dams.")],
        n=3, claims=[r"more harm than good", r"beaver", r"fish ladder"],
        topic=r"slow rivers|dams?\b|beaver|fish ladder", new=r"fish ladder",
        query="What does the book Slow Rivers argue?", ok=[(), ()]),
    case("S3", "claims about one subject", "separate", [
        (0, "Maria", "For our app Orbit we decided to write the backend in Go."),
        (7, "Maria", "For Orbit we also decided to host everything on a single VPS instead of "
                     "Kubernetes."),
        (21, "Maria", "We decided that Orbit ships without user accounts in its first version.")],
        n=3, claims=[r"\bgo\b", r"\bvps\b|kubernetes", r"user accounts"],
        topic=r"orbit", new=r"user accounts",
        query="What did Maria's team decide about Orbit?", ok=[(), ()]),
    # -- a detail of the same claim (must merge) ---------------------------------
    # The later save states the same claim again with a detail it did not have
    # (how strongly it is held, why): one statement, merged (MORE).
    case("T1", "claim detail", "merge", [
        (0, "Maria", "My thesis predicts that open models will match closed ones on most "
                     "benchmarks by 2028."),
        (20, "Maria", "I hold my prediction that open models will match closed ones by 2028 "
                      "very strongly.")],
        new=r"strongly", old=r"2028", topic=r"open models|2028|prediction",
        query="How strongly does Maria hold her prediction about open models?",
        ok=[("MORE",)]),
    case("T2", "claim detail", "merge", [
        (0, "Tom", "Our team decided to write the Orbit backend in Go."),
        (14, "Tom", "We chose Go for the Orbit backend because everyone on the team already "
                    "knows it.")],
        new=r"already know|everyone", old=r"\bgo\b", topic=r"orbit|backend",
        query="Why did Tom's team choose Go for the Orbit backend?", ok=[("MORE",)]),
]
KINDS = list(dict.fromkeys(c["kind"] for c in CASES))
LAYOUTS = ("same", "diff")


def merge_pair(id: str, kind: str, expect: str, old: tuple, new: tuple,
               **fields: Any) -> dict[str, Any]:
    """A fixed pair for the merge writer: (day said, text, when or None) of
    the old memory and of the new fact, as extraction wrote them."""
    return {"id": id, "kind": kind, "expect": expect, "old": old, "new": new, **fields}


#: Pairs for the merge writer alone (``merges``), the texts fixed as the
#: extraction wrote them, so that a run tests the writer whatever extraction
#: does that day. "merge": one thing, the writer must write one text, and a
#: time must not move (``misdated``, unless ``anchored`` finds its date);
#: "apart": two events or things of one kind, which the writer should not
#: join. P1 is LoCoMo conv-41 in fictional form: extraction kept "the
#: previous year", and the merged text put it after the December date.
MERGE_PAIRS: list[dict[str, Any]] = [
    merge_pair("P1", "relative time", "apart",
               (290, "Tom returned from a family road trip on 2026-12-16 and said it was fun.",
                {"start": "2026-12-16"}),
               (404, "Tom shared a photo of a mountain at sunset; it reminded him of a road trip "
                     "from the previous year, when they explored the coast up north and visited "
                     "some national parks.", None),
               misdated=r"\b(previous|prior|last) year\b|\byear before\b|\b2025\b",
               anchored=r"(?<![\d-])2026(?![\d-])"),
    merge_pair("P2", "relative time", "merge",
               (0, "Maria keeps fit and recently started doing aerial yoga; she says it is great.",
                None),
               (180, "Maria said she really enjoys the upside-down poses in aerial yoga because "
                     "they make her feel free and light.", None),
               misdated=r"\b(recently|just) started\b|\bthis week\b",
               anchored=r"\bmarch 2026\b|\b2026-03|\b2 march\b|\bmarch 2\b"),
    merge_pair("P3", "relative time", "merge",
               (0, "Dave went to a classic car show last weekend and said the restored cars were "
                   "amazing.", {"start": "2026-02-28", "end": "2026-03-01"}),
               (11, "Dave said the best car at the classic car show was a restored 1967 Mustang.",
                None),
               misdated=r"\blast weekend\b",
               anchored=r"\bfebruary 28\b|\b28 february\b|\bmarch 1\b|\b1 march\b|\b2026-02-28\b"
                        r"|\b2026-03-01\b"),
    merge_pair("P4", "relative time", "merge",
               (0, "Tom is getting ready for the Riverside chess tournament next month.", None),
               (30, "Tom has been practicing endgames every night for the Riverside chess "
                    "tournament.", None),
               misdated=r"\bnext month\b", anchored=r"\bapril\b|\b2026-04"),
    merge_pair("P5", "two of one kind", "apart",
               (119, "Audrey had a doggy playdate with her dogs on Friday, 2026-06-26; it was a "
                     "bit crazy but lots of fun.", {"start": "2026-06-26"}),
               (187, "Audrey recently organized a doggy playdate with the neighbors' dogs; their "
                     "joy made her heart feel so full.", None)),
    merge_pair("P6", "two of one kind", "apart",
               (100, "Tom has a leg injury; he said it is rough, but the doctor says it is not "
                     "serious.", None),
               (105, "Last season, Tom hurt his ankle and needed physical therapy before he could "
                     "play again.", None)),
    merge_pair("P7", "two of one kind", "apart",
               (133, "Tom scored his career-high 40 points in a win during the week of "
                     "2026-07-06.", {"start": "2026-07-06", "end": "2026-07-12"}),
               (283, "Last Friday, Tom had a career high in assists in a big game against his "
                     "rivals; the arena was electric.", {"start": "2026-12-04"})),
    merge_pair("P8", "two of one kind", "apart",
               (0, "Tom signed a basketball shoe deal with Arvo Sports.", None),
               (150, "Last week, Tom got a deal with an outdoor gear company and received top "
                     "hiking gear.", {"start": "2026-07-20", "end": "2026-07-26"})),
    merge_pair("P9", "two of one kind", "apart",
               (112, "Melanie took her family camping in the mountains during the week of "
                     "2026-06-15.", {"start": "2026-06-15", "end": "2026-06-21"}),
               (135, "Melanie will always remember the family camping trip last year when they "
                     "saw the Perseid meteor shower.", None)),
    merge_pair("P10", "added detail", "merge",
               (0, "Tom has a dog named Rex.", None),
               (42, "Tom's dog Rex is a three-year-old German shepherd.", None)),
    merge_pair("P11", "added detail", "merge",
               (0, "Maria is writing a novel.", None),
               (42, "Maria's novel is a mystery set in 1920s Lisbon.", None)),
    merge_pair("P12", "plan happened", "merge",
               (95, "Nate planned a gaming party for the weekend of 2026-06-20, inviting his "
                    "tournament friends.", {"start": "2026-06-20", "end": "2026-06-21"}),
               (116, "Seven people came to Nate's gaming party, and six said they want to do it "
                     "again next month.", None)),
]

# -- claims about one subject -------------------------------------------------
# A labelled set for the rule that a merge keeps a memory to one fact: "new
# claim" pairs, where the new fact states another claim, position, finding or
# decision about the memory's subject (it must be its own memory: NEW, and the
# writer must not join it), and "claim detail" pairs, where it states the same
# claim with a detail the memory lacks (a condition, a reason, the evidence,
# how strongly it is held, who or when: MORE, one merged text). Q4, Q5 and Q15
# start from a memory that earlier merges already joined, the step by which one
# memory grows with every save. Every pair: the memory said on day 0, the new
# fact on day 21.
_THESIS = ("The central claim of Ana Reyes's thesis is that small language models can match "
           "large ones on narrow tasks.")
_CONTAMINATION = ("Ana Reyes's thesis argues that benchmark contamination explains most of the "
                  "reported gains of large models.")
_SCALE = "Ana Reyes's thesis explicitly rejects the idea that scale alone produces reasoning."
_PROSE = "Ana Reyes accepts that large models still write better open-ended prose than small ones."
_REMOTE = "Jonas Berg believes remote work makes teams more productive."
_BATTERY = "The Vela X2 laptop's battery lasts about 14 hours."
_GO = "The Orbit team decided to write the backend in Go."
_NET_ZERO = "The Harbor Council's climate plan sets a goal of net zero emissions by 2040."
_GAS = "The Harbor Council's climate plan bans new gas heating in public buildings from 2028."
_DAMS = "The book 'Slow Rivers' argues that 20th-century dams did more harm than good."
_FUSION = ("One idea for Lena's AGI article is to compare AGI forecasts with past forecasts for "
           "fusion power.")
_SLEEP = "Dr. Okafor holds that sleep debt cannot be repaid by sleeping in on weekends."
_WALKS = "The 2024 Lindqvist study found that daily walks lowered blood pressure."
_PASTA = "Maria thinks the pasta at Nonna's is the best in town."
_KNEE = "Tom's doctor said his knee pain comes from weak quadriceps."
_PREDICTION = ("Ana Reyes's thesis predicts that open models will match closed ones on most "
               "benchmarks by 2028.")


def _claim(id: str, old: str, new: str) -> dict[str, Any]:
    return merge_pair(id, "new claim", "apart", (0, old, None), (21, new, None))


def _detail(id: str, old: str, new: str) -> dict[str, Any]:
    return merge_pair(id, "claim detail", "merge", (0, old, None), (21, new, None))


CLAIM_PAIRS: list[dict[str, Any]] = [
    _claim("Q1", _THESIS, _CONTAMINATION),
    _claim("Q2", _THESIS, _SCALE),
    _claim("Q3", _CONTAMINATION, _PROSE),
    _claim("Q4", f"{_THESIS} The thesis further argues that benchmark contamination explains "
                 "most of the reported gains of large models.", _PREDICTION),
    _claim("Q5", f"{_THESIS} The thesis further argues that benchmark contamination explains "
                 "most of the reported gains of large models. The thesis explicitly rejects the "
                 "idea that scale alone produces reasoning.", _PROSE),
    _claim("Q6", _REMOTE, "Jonas Berg thinks open-plan offices should be phased out."),
    _claim("Q7", _BATTERY, "The Vela X2 laptop's keyboard is too shallow for long typing "
                           "sessions."),
    _claim("Q8", _GO, "The Orbit team decided to host everything on a single VPS instead of "
                      "Kubernetes."),
    _claim("Q9", _NET_ZERO, _GAS),
    _claim("Q10", _DAMS, "'Slow Rivers' claims that beavers restore wetlands faster than "
                         "engineered projects do."),
    _claim("Q11", _FUSION, "Lena wants her AGI article to argue that current benchmarks measure "
                           "memorization rather than general ability."),
    _claim("Q12", _SLEEP, "Dr. Okafor recommends no screens for an hour before bed."),
    _claim("Q13", _WALKS, "The 2024 Lindqvist study found no effect of daily walks on "
                          "cholesterol."),
    _claim("Q14", _PASTA, "Maria finds Nonna's too loud for a quiet conversation."),
    _claim("Q15", f"{_REMOTE} He also thinks open-plan offices should be phased out.",
           "Jonas Berg argues that four-day weeks reduce burnout."),
    _claim("Q16", _KNEE, "Tom's doctor told him to stop running on concrete."),
    _detail("D1", _THESIS, "Ana Reyes's thesis claims that small language models match large "
                           "ones on narrow tasks once they are fine-tuned on fewer than 10,000 "
                           "examples."),
    _detail("D2", _PREDICTION, "Ana Reyes holds her prediction that open models will match "
                               "closed ones by 2028 very strongly."),
    _detail("D3", _REMOTE, "Jonas Berg believes remote work makes teams more productive because "
                           "it cuts down on interruptions."),
    _detail("D4", _BATTERY, "The Vela X2's 14-hour battery life was measured on video playback "
                            "at half brightness."),
    _detail("D5", _GO, "The Orbit team chose Go for the backend because everyone on the team "
                       "already knows it."),
    _detail("D6", _GAS, "The Harbor Council's ban on new gas heating from 2028 also covers "
                        "schools and libraries."),
    _detail("D7", _DAMS, "'Slow Rivers' backs its case against 20th-century dams with the "
                         "collapse of salmon runs on the Columbia River."),
    _detail("D8", _FUSION, "Lena wants the comparison of AGI and fusion forecasts in her article "
                           "to go back to the 1950s."),
    _detail("D9", _WALKS, "In the 2024 Lindqvist study, daily 30-minute walks lowered systolic "
                          "blood pressure by 5 mmHg over 12 weeks."),
    _detail("D10", _PASTA, "Maria says the cacio e pepe at Nonna's is the best pasta in town."),
    _detail("D11", _KNEE, "Dr. Hale, Tom's doctor, told him on Monday that his knee pain comes "
                          "from weak quadriceps."),
    _detail("D12", _PROSE, "Ana Reyes concedes in chapter 4 of her thesis that large models still "
                           "write better open-ended prose."),
    _detail("D13", _SLEEP, "Dr. Okafor holds that sleep debt cannot be repaid on weekends, citing "
                           "her 2023 study of 400 night-shift nurses."),
    _detail("D14", "Tom is learning the cello.", "Tom is learning the cello with weekly lessons "
                                                 "at the Riverside music school."),
    _detail("D15", "The Orbit team moved the launch to May.", "The Orbit team moved the launch "
                                                              "to 12 May 2026 to finish the "
                                                              "security audit first."),
]
MERGE_PAIRS += CLAIM_PAIRS


def claim_question_pairs() -> list[dict[str, Any]]:
    """``CLAIM_PAIRS`` as labelled pairs for the reconcile question
    (``pair_answers``): a new claim fits no answer that acts (it is stored as
    new), a detail of the same claim fits MORE."""
    return [{"id": p["id"], "set": "claims", "label": "NEW" if p["expect"] == "apart" else "MORE",
             "ok": [] if p["expect"] == "apart" else ["MORE"],
             "old": p["old"][1], "old_said": _day_stamp(p["old"][0]),
             "new": p["new"][1], "new_said": _day_stamp(p["new"][0])} for p in CLAIM_PAIRS]

_DATED = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(january|february|march|april|may|june|july|"
                    r"august|september|october|november|december) \d{1,2}\b", re.I)


# ---------------------------------------------------------------- the grade
def _says(pattern: str | None, text: str) -> bool:
    return bool(pattern) and re.search(pattern, text, re.I) is not None


#: A text that says an older value was before: history in its own words.
_PAST = re.compile(r"\b(previously|used to|formerly|former|no longer|sold|left|quit|stopped)\b", re.I)


def _moment(stamp: Any) -> datetime:
    moment = datetime.fromisoformat(str(stamp))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def misdated(case: dict[str, Any], result: dict[str, Any]) -> str | None:
    """The text of a live memory that holds both facts (it matches ``old``
    and ``new``), was written at a later save than the first, and states a
    time wrong: ``misdated`` finds it and ``anchored`` (the date that time
    names) does not. Such a memory is dated at its own save, so a time
    relative to an earlier day, or to the other fact's date, now reads as
    another one. None when the case sets no ``misdated`` or none does."""
    if not case.get("misdated"):
        return None
    first = _moment(result["saves"][0]["at"])
    for row in result["final"]:
        text = row["content"]
        if (row["invalid_at"] or _moment(row["created_at"]) <= first
                or not (_says(case.get("old"), text) and _says(case.get("new"), text))):
            continue
        if _says(case["misdated"], text) and not _says(case.get("anchored"), text):
            return text
    return None


def grade(case: dict[str, Any], result: dict[str, Any]) -> tuple[str, str]:
    """(verdict, note) of one case's final store: "right", "held" (both live,
    the new one waiting for a person), or what went wrong: "stale" (a live
    memory states only an older value, as current), "duplicate" (a later save
    left a second live copy), "redundant" (a detail beside the fact it
    refines), "merged" (recurring events became one), "joined" (one live
    memory holds two of a case's separate ``claims``), "lost" (the new fact is
    not live), "retracted" (a memory that stays true was taken out of
    search), "not found" (not among the search's first five), "misranked" (an
    older value ranks above the current one), "misdated" (a memory written
    later states one fact's time wrong, ``misdated``)."""
    rows, search = result["final"], result["search"]
    wrong_time = misdated(case, result)
    if wrong_time:
        return "misdated", wrong_time
    topic = [r for r in rows if _says(case["topic"], r["content"])]
    live = [r for r in topic if not r["invalid_at"]]
    history = [r for r in topic if r["invalid_at"] and r.get("history")]
    new, old = case.get("new"), case.get("old")
    held = any(r.get("conflict") for r in live)
    first5 = search[:5]
    found = any(h["id"] in {r["id"] for r in live} and _says(new, h["content"]) for h in first5)
    # what the first save left: later saves of one fact must not add to it
    first = [r for r in result["saves"][1]["before"]
             if not r["invalid_at"] and _says(case["topic"], r["content"])]
    expect = case["expect"]

    def dated(text: str) -> bool:
        return bool(case.get("dated_ok") and _DATED.search(text))

    def is_stale(text: str) -> bool:
        if "stale" in case:
            return _says(case["stale"], text)
        return (_says(old, text) and not _says(new, text) and not _PAST.search(text)
                and not dated(text))

    if expect == "replace":
        if not any(_says(new, r["content"]) for r in live):
            return "lost", "no live memory holds the new value"
        stale = [r for r in live if is_stale(r["content"])]
        if stale and held:
            return "held", ""
        if stale:
            return "stale", stale[0]["content"]
        if not found:
            return "not found", ""
        # an older value kept as history shows only below the current one
        ranked = [h for h in first5
                  if (not h["invalid_at"] and _says(new, h["content"]))
                  or (h["invalid_at"] and _says(old, h["content"]) and not _says(new, h["content"]))]
        if ranked and ranked[0]["invalid_at"]:
            return "misranked", ranked[0]["content"]
        return "right", ""
    if expect == "one":
        if not live:
            return "lost", ""
        if len(live) > max(len(first), 1):
            return ("held" if held else "duplicate"), ""
        return ("right" if found else "not found"), ""
    if expect == "merge":
        if not live:
            return "lost", ""
        if len(live) > max(len(first), 1):
            return ("held" if held else "redundant"), ""
        if not any(_says(new, r["content"]) for r in live):
            return "lost", "the detail"
        return ("right" if found else "not found"), ""
    if expect == "separate":
        joined = next((r["content"] for r in live
                       if sum(_says(claim, r["content"]) for claim in case.get("claims", ())) > 1),
                      None)
        if joined:
            return "joined", joined
        if len(live) >= case["n"]:
            return "right", ""
        return ("merged" if live else "lost"), f"{len(live)} live of {case['n']}"
    if expect == "keep":
        if not any(_says(new, r["content"]) for r in live):
            return "lost", "the new fact"
        if not any(_says(old, r["content"]) for r in live + history):
            return "retracted", "the older fact"
        if held:
            return "held", ""
        return ("right" if found else "not found"), ""
    raise ValueError(expect)


# ------------------------------------------------------------ one case, run
def _date_text(moment: datetime) -> str:
    hour = moment.strftime("%I").lstrip("0")
    return (f"{hour}:{moment.strftime('%M')} {moment.strftime('%p').lower()} on {moment.day} "
            f"{moment.strftime('%B')}, {moment.year}")


def _row(store: Any, memory: Any) -> dict[str, Any]:
    metadata = memory.metadata or {}
    kinds = [e.kind for e in store.backend.history(memory.id) if e.event == "SUPERSEDE"]
    return {"id": memory.id, "content": memory.content, "run": memory.run_id,
            "created_at": memory.created_at, "updated_at": memory.updated_at,
            "invalid_at": memory.invalid_at, "superseded_by": memory.superseded_by,
            # superseded as an update: searchable as history (models.HISTORY_KINDS)
            "history": bool(memory.invalid_at and memory.superseded_by and kinds
                            and kinds[-1] == "update"),
            "kind": kinds[-1] if kinds else None,
            "conflict": metadata.get("conflict"), "when": metadata.get("when")}


def run_case(case: dict[str, Any], layout: str, make: Any) -> dict[str, Any]:
    """Save a case's messages into a fresh store (``make()``) in ``layout`` and
    return what happened: every save's actions, every reconcile question and
    its answer, the memories before each save, the final store (superseded
    memories included) and the search for the latest value."""
    out: dict[str, Any] = {"case": case["id"], "kind": case["kind"], "layout": layout,
                           "saves": [], "decisions": [], "error": None}
    store = make()
    ask = store.decider.decide
    current = {"save": -1}

    def logged(state: str, questions: dict[str, Any]) -> Any:
        answers = ask(state, questions)
        if "action" in questions:
            answer = answers["action"]
            record = {"save": current["save"], "state": state, "action": answer.value,
                      "p": answer.probabilities, "conf": answer.confidence,
                      "available": answer.available}
            if "target" in questions:
                record["target"] = answers["target"].value
            out["decisions"].append(record)
        return answers

    store.decider.decide = logged  # type: ignore[method-assign]
    try:
        for i, (day, speaker, text) in enumerate(case["saves"]):
            current["save"] = i
            moment = BASE + timedelta(days=day)
            run_id = "r1" if layout == "same" else f"r{i + 1}"
            before = [_row(store, m) for m in
                      store.get_all(user_id=USER, include_invalid=True, limit=1000)]
            result = store.add([{"role": speaker, "content": text}], user_id=USER, run_id=run_id,
                               metadata={"context": f"{SPEAKERS}, {_date_text(moment)}"},
                               infer=True, now=moment,
                               created_at=moment.isoformat(timespec="seconds"))
            out["saves"].append({"i": i, "run": run_id, "at": moment.isoformat(), "text": text,
                                 "before": before,
                                 "actions": [a.model_dump() for a in result.actions],
                                 "warnings": result.warnings})
        out["final"] = [_row(store, m) for m in
                        store.get_all(user_id=USER, include_invalid=True, limit=1000)]
        hits = store.search(case["query"], user_id=USER, limit=5)
        out["search"] = [{"id": h.memory.id, "content": h.memory.content,
                          "invalid_at": h.memory.invalid_at, "score": h.score} for h in hits]
        last_run = out["saves"][-1]["run"]
        hits = store.search(case["query"], user_id=USER, run_id=last_run, limit=5)
        out["search_in_run"] = [{"id": h.memory.id, "content": h.memory.content,
                                 "invalid_at": h.memory.invalid_at} for h in hits]
        # what the last save said, found by a search of its run alone
        out["found_in_run"] = any(not h["invalid_at"] and _says(case.get("new"), h["content"])
                                  for h in out["search_in_run"])
        out["verdict"], out["note"] = grade(case, out)
    except BaseException as exc:  # a reached cap is a BaseException
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-2000:]
    finally:
        try:
            store.decider.close()
        except Exception:
            pass
        store.close()
    return out


# ------------------------------------------------------- asking the question
def _said(stamp: str | None) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def render_state(candidates: list[tuple[str, str | None]], new: str, new_said: str | None) -> str:
    """The reconcile state as the installed memry writes it: with the dates
    where it shows them (``reconcile.reconcile_state``), as before otherwise."""
    from memry.intelligence import reconcile
    from memry.models import Memory

    if hasattr(reconcile, "reconcile_state"):
        memories = [Memory(content=text, **({"created_at": said} if said else {}))
                    for text, said in candidates]
        return reconcile.reconcile_state(memories, new, new_said)
    listing = "\n".join(f"[{i}] {text}" for i, (text, _) in enumerate(candidates))
    return f"EXISTING memories:\n{listing}\n\nNEW fact:\n{new}"


def ask_text(llm: Any, state: str) -> dict[str, Any]:
    """The reconcile question asked of the text model alone
    (``reconcile.RECONCILE_SYSTEM``), as memry asks it where no decision
    provider answers. Its answer carries no confidence and acts as given
    (read here as 1.0); a MORE carries the merged text it wrote."""
    from memry.intelligence import reconcile
    from memry.intelligence.extraction import parse_lenient_json

    parsed = parse_lenient_json(llm.complete(reconcile.RECONCILE_SYSTEM, state,
                                             json_schema=reconcile.RECONCILE_SCHEMA))
    parsed = parsed if isinstance(parsed, dict) else {}
    action, target = parsed.get("action"), parsed.get("target")
    if isinstance(target, str) and target.strip().isdigit():
        target = int(target)
    return {"state": state, "raw": action, "action": action if action in ACTIONS else None,
            "conf": 1.0 if action in ACTIONS else None,
            "target": target if isinstance(target, int) else None,
            "content": parsed.get("content"), "reason": parsed.get("reason")}


def ask(decider: Any, candidates: list[tuple[str, str | None]], new: str,
        new_said: str | None) -> dict[str, Any]:
    """One reconcile question as the installed memry asks it, and the answer
    read in the five answers of the redesign: of the decision provider, or of
    the text model when ``decider`` is one (``ask_text``)."""
    from memry.intelligence import reconcile

    state = render_state(candidates, new, new_said)
    if not hasattr(decider, "decide"):
        return ask_text(decider, state)
    decided = reconcile._decide_action(decider, state, len(candidates))
    if decided is None:
        return {"state": state, "action": None}
    action = decided["action"]
    return {"state": state, "raw": action, "action": OLD_WORDS.get(action, action),
            "conf": decided.get("confidence"), "target": decided.get("target"),
            "p": decided.get("probabilities")}


_LISTING = re.compile(r"^\[(\d+)\] (?:\(said [^)]*\) )?(.*)$", re.M)


def parse_state(state: str) -> tuple[list[str], str]:
    head, _, rest = state.partition("\n\nNEW fact")
    new = rest.split(":\n", 1)[1].split("\n\n", 1)[0].strip() if ":\n" in rest else ""
    return [text for _, text in _LISTING.findall(head)], new


def replay_answers(results: list[dict[str, Any]], decider: Any) -> list[dict[str, Any]]:
    """Every logged reconcile question of a cases run, asked again with the
    installed wording, with the dates the memories were said (read from the
    store before the save) and the case's labels (``ok``)."""
    cases = {c["id"]: c for c in CASES}
    out = []
    for result in results:
        spec = cases[result["case"]]
        for decision in result.get("decisions", []):
            i = decision["save"]
            if i < 1 or i >= len(result["saves"]):
                continue
            save = result["saves"][i]
            said = {row["content"]: row["created_at"] for row in save["before"]}
            texts, new = parse_state(decision["state"])
            candidates = [(text, said.get(text)) for text in texts]
            answer = ask(decider, candidates, new, save["at"])
            target = answer.get("target")
            out.append({"source": f"{result['case']}:{result['layout']}:{i}", "set": "cases",
                        "kind": spec["kind"], "ok": list((spec["ok"] or [()] * 9)[i - 1]),
                        "on_topic": (target is not None and target < len(texts)
                                     and _says(spec["topic"], texts[target])),
                        "was": [decision["action"], decision["conf"]], **answer})
    return out


def pair_answers(pairs: list[dict[str, Any]], decider: Any) -> list[dict[str, Any]]:
    """The reconcile question on each labelled pair: one existing memory, the
    new fact, both dated. An exact duplicate is SAME by rule, as in the store."""
    from memry.intelligence.reconcile import _normalize

    out = []
    for pair in pairs:
        if _normalize(pair["old"]) == _normalize(pair["new"]):
            answer: dict[str, Any] = {"action": "SAME", "conf": None, "rule": True}
        else:
            answer = ask(decider, [(pair["old"], pair["old_said"])], pair["new"], pair["new_said"])
        out.append({"source": pair["id"], "set": pair["set"], "label": pair["label"],
                    "ok": pair["ok"], "on_topic": True, **answer})
    return out


# ------------------------------------------------------ the merge writer
def _day_stamp(day: int) -> str:
    return (BASE + timedelta(days=day)).isoformat(timespec="seconds")


def write_pair(llm: Any, pair: dict[str, Any]) -> dict[str, Any]:
    """The installed memry's merge writer on one fixed pair (``MERGE_PAIRS``),
    asked as that memry asks it: with both facts dated where it shows the
    dates (``reconcile.merge_state``), with the two texts alone before. The
    answer is "merged" (with its text), "apart" (the writer read two things)
    or "none" (it wrote nothing)."""
    from memry.intelligence import reconcile
    from memry.models import Memory

    (old_day, old_text, old_when), (new_day, new_text, new_when) = pair["old"], pair["new"]
    if hasattr(reconcile, "merge_state"):
        target = Memory(content=old_text, created_at=_day_stamp(old_day),
                        updated_at=_day_stamp(old_day),
                        metadata={"when": old_when} if old_when else {})
        written = reconcile.write_merged(llm, target, new_text, said=_day_stamp(new_day),
                                         when=new_when)
        text, apart = written.content, written.apart
    else:
        text, apart = reconcile.write_merged(llm, old_text, new_text), False
    return {"id": pair["id"], "kind": pair["kind"], "expect": pair["expect"],
            "answer": "merged" if text else "apart" if apart else "none", "text": text}


def grade_merge(pair: dict[str, Any], written: dict[str, Any]) -> str:
    """"right", or what went wrong: "misdated" (the merged text states a time
    wrong, ``misdated`` without ``anchored``), "joined" (two things of one
    kind became one text), "apart" (one thing kept apart: the fact would be
    stored beside the memory it adds to), "none" (nothing written)."""
    if written["answer"] == "none":
        return "none"
    if written["answer"] == "apart":
        return "right" if pair["expect"] == "apart" else "apart"
    text = written["text"] or ""
    if _says(pair.get("misdated"), text) and not _says(pair.get("anchored"), text):
        return "misdated"
    return "right" if pair["expect"] == "merge" else "joined"


def merge_table(results: list[dict[str, Any]]) -> str:
    pairs = {p["id"]: p for p in MERGE_PAIRS}
    runs = sorted({r["run"] for r in results})
    kinds = list(dict.fromkeys(p["kind"] for p in MERGE_PAIRS))
    lines = ["| kind | pairs | " + " | ".join(f"run {n}: right / other" for n in runs) + " |",
             "|---|---|" + "---|" * len(runs)]
    for kind in kinds + ["all"]:
        ids = [p["id"] for p in MERGE_PAIRS if kind in ("all", p["kind"])]
        cells = []
        for n in runs:
            verdicts = collections.Counter(grade_merge(pairs[r["id"]], r) for r in results
                                           if r["run"] == n and r["id"] in ids)
            other = ", ".join(f"{c} {v}" for v, c in sorted(verdicts.items()) if v != "right")
            cells.append(f"{verdicts['right']} / {other or '-'}")
        lines.append(f"| {kind} | {len(ids)} | " + " | ".join(cells) + " |")
    lines.append("")
    for r in results:
        lines.append(f"{r['id']} run {r['run']}: {grade_merge(pairs[r['id']], r)} "
                     f"({r['answer']}) {r['text'] or ''}")
    return "\n".join(lines)


#: Jev's measured bars (``JevDecider.reconcile_bars``), read when an answer is
#: graded as memry acts on it.
JEV_BARS = {"SAME": 0.85, "MORE": 0.8, "CHANGED": 0.5, "WRONG": 0.5}


def acted(answer: dict[str, Any], bars_: dict[str, float] = JEV_BARS) -> str:
    """What memry does with an answer at the bars: the answer itself at or
    above its bar, "NEW" for a SAME or a MORE under it (stored as new),
    "held" for a CHANGED or a WRONG under it (both kept, a person asked)."""
    action, conf = answer.get("action"), answer.get("conf")
    if action in (None, "NEW") or conf is None:
        return action or "none"
    if conf >= bars_.get(action, 0.0):
        return action
    return "NEW" if action in ("SAME", "MORE") else "held"


def claim_table(answers: list[dict[str, Any]]) -> str:
    """Per label of ``CLAIM_PAIRS`` (a new claim, NEW; a detail of the same
    claim, MORE): how the answers act at Jev's bars, and MORE's confidences."""
    lines = ["| label | pairs | acted as | MORE confidences |", "|---|---|---|---|"]
    for label in ("NEW", "MORE"):
        rows = [a for a in answers if a.get("label") == label]
        counts = collections.Counter(acted(a) for a in rows)
        confs = sorted((a["conf"] for a in rows if a.get("action") == "MORE"
                        and a.get("conf") is not None), reverse=True)
        lines.append(f"| {label} | {len(rows)} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                     + " | " + (", ".join(f"{c:.2f}" for c in confs) or "-") + " |")
    lines.append("")
    for a in answers:
        lines.append(f"{a['source']} {a.get('label')}: {a.get('action')} "
                     f"{a.get('conf') if a.get('conf') is None else round(a['conf'], 2)} "
                     f"-> {acted(a)}")
    return "\n".join(lines)


def bars(answers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each acting answer, the lowest confidence from which none of that
    answer was wrong (outside its ``ok``, or aimed at a memory off the
    case's topic), and how many right ones that bar lets act."""
    out: dict[str, dict[str, Any]] = {}
    for action in ACTIONS[1:]:
        given = [a for a in answers if a.get("action") == action and a.get("conf") is not None]
        wrong = [a["conf"] for a in given if action not in a["ok"] or not a.get("on_topic", True)]
        right = [a["conf"] for a in given if action in a["ok"] and a.get("on_topic", True)]
        worst = max(wrong, default=None)
        bar = None if worst is None else worst + 0.01
        out[action] = {"answers": len(given), "wrong": len(wrong), "worst_wrong": worst,
                       "right": len(right),
                       "right_from_bar": sum(1 for c in right if bar is None or c >= bar),
                       "bar_with_no_wrong": bar,
                       "wrong_confidences": sorted(wrong, reverse=True)[:8]}
    return out


# ------------------------------------------------------------------- report
def regraded(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The results with each verdict read again by ``grade`` from the store
    and search they hold, so a table of an older run reads as a new one."""
    cases = {c["id"]: c for c in CASES}
    out = []
    for r in results:
        if r.get("final") is not None and r.get("search") is not None:
            verdict, note = grade(cases[r["case"]], r)
            r = {**r, "verdict": verdict, "note": note}
        if r.get("search_in_run") is not None:
            r["found_in_run"] = any(not h.get("invalid_at")
                                    and _says(cases[r["case"]].get("new"), h["content"])
                                    for h in r["search_in_run"])
        out.append(r)
    return out


def table(results: list[dict[str, Any]]) -> str:
    by = {(r["case"], r["layout"]): r for r in results}
    lines = ["| kind | cases | same run: right / held / other | one run per save: right / held / other |",
             "|---|---|---|---|"]
    totals = {layout: collections.Counter() for layout in LAYOUTS}
    for kind in KINDS:
        ids = [c["id"] for c in CASES if c["kind"] == kind]
        cells = []
        for layout in LAYOUTS:
            verdicts = collections.Counter(by[(i, layout)].get("verdict") or "error"
                                           for i in ids if (i, layout) in by)
            totals[layout].update(verdicts)
            other = ", ".join(f"{n} {v}" for v, n in sorted(verdicts.items())
                              if v not in ("right", "held"))
            cells.append(f"{verdicts['right']} / {verdicts['held']} / {other or '-'}")
        lines.append(f"| {kind} | {len(ids)} | {cells[0]} | {cells[1]} |")
    cells = []
    for layout in LAYOUTS:
        v = totals[layout]
        other = ", ".join(f"{n} {k}" for k, n in sorted(v.items()) if k not in ("right", "held"))
        cells.append(f"{v['right']} / {v['held']} / {other or '-'}")
    lines.append(f"| all | {len(CASES)} | {cells[0]} | {cells[1]} |")
    agree = sum(1 for c in CASES if (c["id"], "same") in by and (c["id"], "diff") in by
                and by[(c["id"], "same")].get("verdict") == by[(c["id"], "diff")].get("verdict"))
    lines.append(f"\nThe two layouts end alike in {agree} of {len(CASES)} cases.")
    found = [r.get("found_in_run") for r in results if r["layout"] == "diff" and "found_in_run" in r]
    lines.append(f"One run per save: a search of the last save's run finds what it said in "
                 f"{sum(1 for f in found if f)} of {len(found)} cases.")
    return "\n".join(lines)


def case_lines(results: list[dict[str, Any]]) -> str:
    by = {(r["case"], r["layout"]): r for r in results}
    lines = []
    for c in CASES:
        cells = []
        for layout in LAYOUTS:
            r = by.get((c["id"], layout))
            if r is None:
                cells.append("-")
                continue
            said = "; ".join(f"{d['action']} {d['conf']:.2f}" for d in r.get("decisions", [])
                             if d["save"] >= 1 and d.get("available"))
            acts = "; ".join(",".join(a["event"] for a in s["actions"]) for s in r["saves"][1:])
            cells.append(f"{r.get('verdict')} ({acts}{' | ' + said if said else ''})")
        lines.append(f"{c['id']:3s} {c['kind']:18s} same: {cells[0]}  ||  diff: {cells[1]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- run
def _store_factory() -> Any:
    from evals.external_benchmarks import make_store
    from memry.config import EmbeddingConfig
    from memry.providers.embeddings import OpenAIEmbedder

    def make() -> Any:
        return make_store("extract", OpenAIEmbedder(EmbeddingConfig(provider="openai")),
                          decider="jev", db_path=":memory:")
    return make


def _text_model() -> Any:
    from evals.external_benchmarks import RetryingLLM
    from memry.config import Config
    from memry.providers.llm import build_llm

    llm = build_llm(Config.load(db_path=":memory:").llm)
    if not llm.available:
        raise SystemExit("needs a text model (OPENAI_API_KEY)")
    return RetryingLLM(llm)


def _jev() -> Any:
    from evals.external_benchmarks import retrying_decider
    from memry.config import DecisionConfig
    from memry.providers.decisions import JevDecider

    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise SystemExit("needs TYPESAFE_API_KEY")
    return retrying_decider(JevDecider(DecisionConfig(provider="jev", api_key=key)))


def _meter(args: argparse.Namespace) -> Any:
    from evals import api_usage
    from evals.external_benchmarks import memry_stage

    return api_usage.UsageMeter(args.ledger, label=args.command,
                                caps={"jev": args.jev_cap, "chat": args.chat_cap},
                                refine=memry_stage).install()


def _load(path: str) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _dump(path: str, data: Any) -> None:
    pathlib.Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cases", "replay", "pairs", "claims", "claims-text",
                                            "merges", "table", "bars", "merge-table",
                                            "claim-table"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--only", default="", help="case ids, comma-separated")
    parser.add_argument("--layouts", default="same,diff")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ledger", help="SQLite ledger of every model call (evals/api_usage.py); "
                        "needed by cases, replay and pairs")
    parser.add_argument("--jev-cap", type=int, default=3000)
    parser.add_argument("--chat-cap", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=1, help="merges: runs over the pairs")
    parser.add_argument("--judge", choices=("jev", "text"), default="jev",
                        help="replay, pairs: ask Jev, or the text model alone as memry asks "
                             "it where no decision provider answers")
    args = parser.parse_args(argv)

    if args.command == "table":
        results = regraded([r for path in args.paths for r in _load(path)])
        print(table(results))
        print()
        print(case_lines(results))
        return
    if args.command == "bars":
        answers = [a for path in args.paths for a in _load(path)]
        print(json.dumps(bars(answers), indent=1))
        return
    if args.command == "merge-table":
        print(merge_table([r for path in args.paths for r in _load(path)]))
        return
    if args.command == "claim-table":
        print(claim_table([a for path in args.paths for a in _load(path)]))
        return

    if not args.ledger:
        parser.error(f"{args.command} needs --ledger")
    meter = _meter(args)
    try:
        if args.command == "pairs":
            pairs = _load(args.paths[0])["pairs"]
            decider = _text_model() if args.judge == "text" else _jev()
            _dump(args.paths[1], pair_answers(pairs, decider))
        elif args.command == "claims":
            answers = pair_answers(claim_question_pairs(), _jev())
            _dump(args.paths[0], answers)
            print(claim_table(answers))
        elif args.command == "claims-text":
            answers = pair_answers(claim_question_pairs(), _text_model())
            _dump(args.paths[0], answers)
            print(claim_table(answers))
        elif args.command == "replay":
            decider = _text_model() if args.judge == "text" else _jev()
            _dump(args.paths[1], replay_answers(_load(args.paths[0]), decider))
        elif args.command == "merges":
            from evals import api_usage

            llm = _text_model()
            with api_usage.stage("ingest"), ThreadPoolExecutor(max_workers=args.workers) as pool:
                written = list(pool.map(lambda job: {**write_pair(llm, job[1]), "run": job[0]},
                                        [(n + 1, pair) for n in range(args.runs)
                                         for pair in MERGE_PAIRS]))
            _dump(args.paths[0], written)
            print(merge_table(written))
        else:
            out_path = args.paths[0]
            wanted = {i for i in args.only.split(",") if i}
            layouts = [layout for layout in args.layouts.split(",") if layout]
            results: list[dict[str, Any]] = _load(out_path) if os.path.exists(out_path) else []
            done = {(r["case"], r["layout"]) for r in results if not r.get("error")}
            results = [r for r in results if not r.get("error")]
            make, lock, started = _store_factory(), threading.Lock(), time.time()

            def job(spec: dict[str, Any], layout: str) -> None:
                if (spec["id"], layout) in done:
                    return
                result = run_case(spec, layout, make)
                with lock:
                    results.append(result)
                    _dump(out_path, results)
                print(f"{spec['id']}:{layout} {result.get('verdict')} err={result['error']} "
                      f"jev={meter.calls('jev')} t={time.time() - started:.0f}s", flush=True)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(job, spec, layout) for spec in CASES
                           if not wanted or spec["id"] in wanted for layout in layouts]
                for future in futures:
                    future.result()
            print(table(results))
    finally:
        meter.close()


if __name__ == "__main__":
    main()
