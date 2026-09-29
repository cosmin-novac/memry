"""End to end: a long, realistic history through the whole save path.

The unit tests check each mechanism alone, on inputs picked by hand. Two bugs
got past them and showed only when real data went through ``MemoryStore.add``
and the weekly upkeep: an UPDATE that re-read an edited memory's names made a
second entity of a name it already held, and without a calibrated judge every
fact naming a known person made another entity. This test replays a synthetic
world instead, and checks what must hold after every save and every upkeep
pass.

**The world.** A store owner and 20 people and things (``ENTITIES``), a few in
most saves and many rarely; two different Johnnys (an electrician and a
climbing partner); things written several ways (case, "the Orbit Café", the
nickname "Tomi", "Nordvik" for "Nordvik Labs", "Vessa" for "Vessa Holm"); a
shop the extractor types as a project except in two kinds of sentence, where
it is a product (its listing safety text); versions and a part of one product
("Plotwise v3", "Plotwise v4", "Plotwise sync engine"); tags with case, plural
and synonym variants, and tags that are an entity's name. About 220 saves over
ten simulated weeks, a third of them through the MCP server's deferred path
(``add_deferred``, distilled once the conversation is quiet): new facts, one-off
facts that make the frequent names gather memories, refinements (UPDATE),
exact and reworded restatements (NONE), contradictions (supersede), and a
person's own actions through the store's API: edits (a value, a name added, a
name taken out), deletes (one for good), a retag, a merge, a mistaken merge
and its undo, renames, a removed place and person brought back a week later,
tag merges, renames and deletes, a pair kept apart. Every upkeep pass runs at
the end of each week. Everything is seeded; ids and the clock are made
deterministic for the run.

**The models.** ``WorldLLM`` is the text model: it extracts the world's facts
with names and types as a real extractor does (it writes a stored name the
store offers for another way of writing the same name, keeps "Tomi" or an
unlisted "the Orbit Café" as written, and types the shop a product in its
safety sentences), reconciles from the world's truth, and answers the identity
prompt at random: without a calibrated judge its answers must not matter.
``WorldJudge`` is a calibrated judge: it answers the pair question from the
world's truth with noise (P(same) 0.9-1 for one thing, 0.7-0.95 for one thing
typed two ways, mostly 0-0.2 for two things, some answers in the middle), the
belongs question for versions and parts, and every other typed question Memry
asks. The embedder is the hash embedder, counted. Every call is counted by
kind and by operation.

**The invariants** (``Checker``), after every save and every upkeep pass:

1. one active entity per normalized name, unless that pair was kept apart by
   the calibrated judge (P(different) at its apart bar, or its "apart") or by
   a person (a pair kept apart, a merge undone); a rejection on the text
   model's answer does not count;
2. a memory is linked to each known entity whose name or alias it contains.
   The documented exceptions (counted and reported): a name inside a longer
   name the memory is linked to ("Plotwise" in "Plotwise v3":
   ``graph_retrieval.mask_names``, ``detect_query_entities(longest=True)``);
   a name that is not a referent (``entities.non_referent_reason``) or was
   screened out (``entities.screened_out``); another spelling of a name the
   memory is linked to, where only exact names meet (no calibrated judge,
   ``entities.resolve_mentions``) or the two are a pair in the funnel; a link
   that went with an entity a person removed (``remove_entities``); a memory
   waiting to be distilled (``add_deferred``);
   2b. and the other way round, not linked to a thing its text does not name
   (except the owner, a tag folded into the thing, and the stored name written
   for another spelling);
3. no identity question (a pair, a name check, the identity choice, the text
   model's identity prompt) about a name the memory is already linked to, and
   without a calibrated judge none at all, at save or in upkeep;
4. model calls per save within the bound the design gives
   (``Checker.save_budget``);
5. a memory's tag column, its topic index and its topic mentions agree (a tag
   folded into the thing of its name is mentioned by the thing, under the
   tag's name: ``LocalBackend._mention_tags_locked``), and the column names no
   tag merged away (``tag_filing``);
6. every memory naming an entity has a current property vector, unless its
   masked text is its text (``MemoryStore.refresh_property_vectors``); checked
   after each save and each person's action, and after each whole upkeep
   cycle, whose last step computes what its passes left missing;
7. no orphan rows: mentions, relations, proposals, merge records, property
   vectors, topic links or supersede pointers to rows that are gone, or to an
   entity merged away (a confirmed proposal may keep a tombstone); no relation
   in use whose memory is out of use; one row per pair of entities;
8. after each weekly cycle, the memory answering each of 20 questions about
   the world is in the top 10 of the linked search with a stub judge.

A violation of the store's state (1, 2, 2b, 5, 6, 7) is reported with the
operation that made it and the one after which it was gone, if it went.

Three setups (``SETUPS``): a calibrated judge; no decision provider, the text
model only; and the text model answering the decision questions too
(``MEMRY_DECISION_PROVIDER=llm``). ``MEMRY_E2E_REPORT=<path>`` writes every
violation, the call counts and the operation log as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import threading
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import memry.intelligence.decay as decay_mod
import memry.intelligence.entities as entities_mod
import memry.intelligence.identity as identity_mod
import memry.models as models_mod
import memry.retrieval as retrieval_mod
import memry.store as store_mod
from memry.config import Config
from memry.intelligence.entities import non_referent_reason, screened_out
from memry.intelligence.identity import CANDIDATES_PER_NAME, Mention
from memry.models import TOPIC_TYPE, Scope
from memry.providers.decisions import Answer, Answers, Choice, Decider, Noul, Score
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import LLM
from memry.store import MemoryStore, _text_hash

USER = "ilka"
OWNER = "Ilka Maren"
AGENT = "notes-app"
SEED = 20260105
START = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)  # a Monday
WEEKS = 10


# ---------------------------------------------------------------- the world
#: world id -> (canonical name, extractor type, how often it comes up)
ENTITIES: dict[str, tuple[str, str, int]] = {
    "owner": (OWNER, "person", 6),
    "quirk": ("Quirkwear", "project", 6),
    "lumen": ("Lumen Print", "organization", 1),
    "client": ("Tallow & Pine", "organization", 1),
    "plot": ("Plotwise", "product", 5),
    "plot3": ("Plotwise v3", "product", 2),
    "plot4": ("Plotwise v4", "product", 2),
    "sync": ("Plotwise sync engine", "product", 1),
    "nordvik": ("Nordvik Labs", "organization", 2),
    "tomas": ("Tomas Vell", "person", 3),
    "johnny_e": ("Johnny", "person", 3),
    "johnny_c": ("Johnny", "person", 2),
    "gym": ("Brightline Gym", "place", 1),
    "orbit": ("Orbit Café", "place", 2),
    "pell": ("Dr. Pell", "person", 1),
    "vessa": ("Vessa Holm", "person", 1),
    "oskar": ("Oskar Brem", "person", 1),
    "kestrel": ("Kestrel", "other", 1),
    "ember": ("Ember Grid", "project", 1),
    "trip": ("Ferrow Valley trip", "event", 1),
}
#: Other ways a thing is written, with their weight against the canonical
#: name (1.0), and the week from which they are used.
VARIANTS: dict[str, list[tuple[str, float, int]]] = {
    "quirk": [("QuirkWear", 0.25, 0)],
    "orbit": [("the Orbit Café", 0.45, 0)],
    "tomas": [("Tomi", 0.45, 2)],
    "nordvik": [("Nordvik", 0.3, 1)],
    "vessa": [("Vessa", 0.35, 1)],
    "oskar": [("Oskar", 0.35, 1)],
    "pell": [("Dr Pell", 0.3, 0)],
}
#: child -> (parent, "kind" for a version, "part" for a part)
BELONGS = {"plot3": ("plot", "kind"), "plot4": ("plot", "kind"), "sync": ("plot", "part")}
#: Tags that name one subject written two ways the obvious rules do not catch.
TAG_SYNONYMS = {frozenset({"2026 taxes", "taxes 2026"})}
#: Topic groups: one conversation stays in one.
GROUPS = {
    "shop": ["quirk", "lumen", "client"],
    "work": ["plot", "plot3", "plot4", "sync", "nordvik", "tomas"],
    "home": ["johnny_e", "vessa", "kestrel", "ember"],
    "climbing": ["johnny_c", "gym"],
    "personal": ["owner", "orbit", "pell", "oskar", "trip"],
}


@dataclass(frozen=True)
class Slot:
    """One property of one thing: a sentence with a value that can be stated,
    refined, restated and contradicted."""

    id: str
    subject: str
    template: str
    values: tuple[str, ...]
    tags: tuple[str, ...]
    importance: float = 0.6
    question: str | None = None
    #: extractor types that differ from the entity's usual one in this sentence
    types: tuple[tuple[str, str], ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    start_week: int = 0
    #: how often it comes up, against the others of its thing
    weight: float = 1.0


SLOTS: list[Slot] = [
    # the owner
    Slot("coffee", "owner", "{E} drinks {v} every morning",
         ("oat flat whites", "cardamom lattes", "black filter coffee"), ("morning routine",)),
    Slot("shoes", "owner", "{E}'s shoe size is {v}", ("EU 39", "EU 40"), ("clothing sizes",)),
    Slot("allergy", "owner", "{E} is allergic to {v}", ("penicillin",), ("allergies",),
         importance=0.95, question="What is Ilka Maren allergic to?"),
    Slot("notes", "owner", "{E} keeps work notes in {v}",
         ("plain markdown files", "a paper notebook"), ("work habits",)),
    Slot("race", "owner", "{E} wants to run a {v} race in the spring",
         ("10 km", "half marathon"), ("running goals",)),
    # the shop and its partners
    Slot("ship_day", "quirk", "The {E} shop ships orders through {lumen} every {v}",
         ("Tuesday", "Thursday"), ("quirkwear orders",),
         relations=(("quirk", "uses", "lumen"),)),
    Slot("designs", "quirk", "{E} lists {v} new T-shirt designs each week",
         ("seven", "nine"), ("quirkwear listings",)),
    Slot("safety", "quirk", "{E} listing safety text must say {v}",
         ("'wash cold, inside out'", "'do not tumble dry the print'"), ("quirkwear listings",),
         question="What must the Quirkwear listing safety text say?",
         types=(("quirk", "product"),), start_week=1, weight=3.0),
    Slot("shipping", "quirk", "{E} charges {v} for standard shipping",
         ("4.90 euros", "5.50 euros"), ("quirkwear pricing",),
         question="How much does Quirkwear charge for standard shipping?"),
    Slot("bestseller", "quirk", "The {E} bestseller this month is the {v} design",
         ("tidal owl", "copper fern", "night heron"), ("quirkwear listings",)),
    Slot("banner", "quirk", "The {E} banner uses a {v} palette",
         ("teal and rust", "ink and sand"), ("quirkwear branding", "quirkwear")),
    Slot("cotton", "lumen", "{E} prints the {quirk} shirts on {v} cotton",
         ("organic ring-spun", "recycled"), ("quirkwear orders",),
         question="What cotton does Lumen Print use for the Quirkwear shirts?",
         relations=(("lumen", "supplies", "quirk"),)),
    Slot("order", "client", "{E} ordered {v} custom shirts from {quirk}",
         ("forty", "sixty-five"), ("quirkwear orders",),
         question="How many custom shirts did Tallow & Pine order?",
         relations=(("client", "customer_of", "quirk"),)),
    # work
    Slot("storage", "plot", "{E} stores its boards in {v}", ("PostgreSQL", "SQLite"),
         ("plotwise architecture",), question="Where does Plotwise store its boards?"),
    Slot("price", "plot", "{E} pricing starts at {v} a month", ("9 euros", "12 euros"),
         ("plotwise pricing",)),
    Slot("frontend", "plot", "{E} is built with {v} on the frontend", ("Svelte", "Solid"),
         ("plotwise architecture",)),
    Slot("team", "nordvik", "{E} develops {plot} with a team of {v} engineers", ("six", "eight"),
         ("nordvik work",), relations=(("nordvik", "develops", "plot"),)),
    Slot("export", "plot3", "{E} shipped the {v} export", ("CSV", "PDF"),
         ("plotwise releases",), question="Which export did Plotwise v3 ship?"),
    Slot("notes3", "plot3", "The {E} release notes mention {v}",
         ("offline boards", "faster uploads"), ("plotwise release",)),
    Slot("feature4", "plot4", "{E} will add {v}", ("shared templates", "a calendar view"),
         ("plotwise releases",)),
    Slot("beta", "plot4", "{tomas} leads the {E} beta, which starts in {v}", ("April", "May"),
         ("plotwise releases",), question="When does the Plotwise v4 beta start?",
         relations=(("tomas", "leads", "plot4"),)),
    Slot("retries", "sync", "The {E} retries failed uploads {v} times", ("four", "eleven"),
         ("plotwise architecture",),
         question="How many times does the Plotwise sync engine retry failed uploads?"),
    Slot("salary", "nordvik", "{E} pays salaries on the {v} of each month", ("25th", "28th"),
         ("nordvik work",), question="On which day does Nordvik Labs pay salaries?"),
    Slot("office", "nordvik", "The {E} office moved to {v}", ("Kellerweg 12", "Hafenkai 3"),
         ("nordvik work",)),
    Slot("role", "tomas", "{E} works as {v} at {nordvik}", ("tech lead", "engineering manager"),
         ("nordvik work",), relations=(("tomas", "works_at", "nordvik"),)),
    Slot("reviews", "tomas", "{E} prefers code reviews {v}", ("in the morning", "after lunch"),
         ("nordvik work",), question="When does Tomas Vell prefer code reviews?"),
    Slot("owes", "tomas", "{E} owes {owner} {v} for the team lunch", ("23 euros", "31 euros"),
         ("money owed",)),
    # home
    Slot("rate", "johnny_e", "{E} the electrician charges {v} an hour", ("45 euros", "60 euros"),
         ("kitchen renovation",),
         question="How much does Johnny the electrician charge per hour?"),
    Slot("rewire", "johnny_e", "{E} is rewiring the kitchen on {v}", ("Friday", "Saturday"),
         ("kitchen renovation",)),
    Slot("sockets", "johnny_e", "{E} recommended {v} sockets for the kitchen island",
         ("pop-up", "flush"), ("kitchen renovation",)),
    Slot("rent", "vessa", "{E} raised the rent to {v}", ("980 euros", "1,020 euros"),
         ("apartment",), importance=0.85, question="What did Vessa Holm raise the rent to?"),
    Slot("balcony", "vessa", "{E} will fix the balcony door by {v}",
         ("mid-February", "the end of March"), ("apartment",)),
    Slot("food", "kestrel", "{E} the cat eats {v} twice a day", ("salmon kibble", "duck pâté"),
         ("cat care",), question="What does Kestrel the cat eat?"),
    Slot("vet", "kestrel", "{E} has a vet check-up on {v}", ("February 20", "March 3"),
         ("cat care",)),
    Slot("tracks", "ember", "{E} tracks {v} for the solar panels",
         ("battery charge", "grid export"), ("ember grid",)),
    Slot("runs", "ember", "{E} runs on {v}", ("a single-board computer", "an old laptop"),
         ("ember grid",), question="What does Ember Grid run on?"),
    # climbing: the second Johnny arrives in week 2
    Slot("belay", "johnny_c", "{E} belays {owner} at {gym} on {v} evenings",
         ("Monday", "Wednesday"), ("climbing",), start_week=2,
         relations=(("johnny_c", "climbs_at", "gym"),)),
    Slot("grade", "johnny_c", "{E} is training for a {v} bouldering grade", ("7A", "7B+"),
         ("climbing",), question="Which bouldering grade is Johnny training for?",
         start_week=2),
    Slot("rope", "johnny_c", "{E} lent {owner} his {v} rope", ("sixty-metre", "seventy-metre"),
         ("Climbing",), start_week=2),
    Slot("fee", "gym", "{E} costs {v} a month", ("49 euros", "55 euros"), ("climbing",),
         question="How much does Brightline Gym cost per month?"),
    Slot("opens", "gym", "{E} opens at {v} on weekends", ("8 am", "9 am"), ("climbing",)),
    # personal
    Slot("buns", "orbit", "{E} serves {v} on Sundays", ("cardamom buns", "rye waffles"),
         ("cafes",), question="What does the Orbit Café serve on Sundays?"),
    Slot("meet", "orbit", "{owner} meets {tomas} at {E} at {v}",
         ("half past eight", "ten sharp"), ("cafes",)),
    Slot("cleaning", "pell", "{E} scheduled the next dental cleaning for {v}",
         ("March 12", "March 19"), ("dental care",),
         question="When is the next dental cleaning with Dr. Pell?"),
    Slot("floss", "pell", "{E} recommends {v} floss", ("waxed", "PTFE"), ("dental care",)),
    Slot("deadline", "oskar", "{E} files {owner}'s tax return by {v}", ("May 31", "July 31"),
         ("2026 taxes",), question="By when does Oskar Brem file the tax return?"),
    Slot("fee_tax", "oskar", "{E} charges {v} for the tax return", ("380 euros", "420 euros"),
         ("taxes 2026",)),
    Slot("when_trip", "trip", "The {E} is planned for {v}",
         ("the first week of June", "mid-July"), ("travel plans",)),
    Slot("stay", "trip", "{owner} booked a guesthouse in {v} for the {E}", ("Kellbach", "Orrin"),
         ("travel plans",)),
]
SLOT = {slot.id: slot for slot in SLOTS}

#: Families of one-off facts ("episodes"): each save of one adds a fact the
#: store has not seen, so the frequent names gather memories week by week, as
#: they do in a real store, and the funnel's later steps are reached.
EPISODES: list[Slot] = [
    Slot("run", "owner", "{E} went for a {v} run on {d}",
         ("5 km", "8 km", "hill", "interval", "12 km"), ("running goals",)),
    Slot("read", "owner", "{E} finished reading {v} on {d}",
         ("a novel about lighthouses", "a book on typography", "a field guide to mosses"),
         ("reading",)),
    Slot("sold", "quirk", "{E} sold {v} shirts on {d}", ("12", "17", "23", "31", "8"),
         ("quirkwear orders",)),
    Slot("passed", "quirk", "{E} listing for the {v} shirt passed the safety review on {d}",
         ("tidal owl", "copper fern", "night heron", "salt marsh"), ("quirkwear listings",),
         types=(("quirk", "product"),), start_week=1, weight=2.0),
    Slot("review", "quirk", "{E} got a {v} review on {d}",
         ("five-star", "four-star", "glowing", "lukewarm"), ("quirkwear listings",)),
    Slot("restock", "lumen", "{E} restocked blank shirts for {quirk} on {d}", ("",),
         ("quirkwear orders",), relations=(("lumen", "supplies", "quirk"),)),
    Slot("deploy", "plot", "{E} deployed build {v} on {d}",
         ("1.8.2", "1.8.3", "1.9.0", "1.9.1", "2.0.0-rc"), ("plotwise releases",)),
    Slot("bug", "plot", "{tomas} fixed a {v} bug in {E} on {d}",
         ("login", "sync", "export", "billing", "search"), ("plotwise bugs",),
         relations=(("tomas", "works_on", "plot"),)),
    Slot("allhands", "nordvik", "{E} held a {v} all-hands on {d}",
         ("quarterly", "short", "remote"), ("nordvik work",)),
    Slot("pair", "tomas", "{E} and {owner} paired on {v} on {d}",
         ("the sync engine", "the billing page", "release notes"), ("nordvik work",)),
    Slot("fix", "johnny_e", "{E} replaced the {v} in the kitchen on {d}",
         ("fuse box", "ceiling light", "oven socket", "extractor fan"), ("kitchen renovation",)),
    Slot("climb", "johnny_c", "{E} and {owner} climbed the {v} wall at {gym} on {d}",
         ("overhang", "slab", "competition", "cave"), ("climbing",), start_week=2,
         relations=(("johnny_c", "climbs_at", "gym"),)),
    Slot("coffee_at", "orbit", "{owner} had {v} at {E} on {d}",
         ("a flat white", "lunch", "a slice of plum cake", "breakfast"), ("cafes",)),
    Slot("rent_paid", "vessa", "{owner} paid {E} the rent for {d}", ("",), ("apartment",)),
]

#: Refinements an UPDATE appends; some name something the memory did not.
REFINEMENTS = (", confirmed by email", ", as {owner} noted", ", which {tomas} double-checked")
#: A manual edit that adds a name to a memory.
EDIT_ADDS = " (recommended by {tomas})"
#: What a manual edit that takes the subject's name out writes instead.
WITHOUT_NAME = {"person": "The contractor", "project": "The project", "product": "The app",
                "organization": "The company", "place": "The place", "other": "The pet",
                "event": "The trip"}


def norm(text: str) -> str:
    """As reconcile compares texts (``reconcile._normalize``)."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def contains_name(text: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


def stable_rng(*parts: Any) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


@dataclass
class Ref:
    surface: str
    wid: str
    etype: str


@dataclass
class TextInfo:
    """What a text says, as the world knows it: the slot and value it states
    and every name in it, with the thing that name is."""

    text: str
    slot: str | None
    value: str | None
    refs: list[Ref]
    relations: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Plan:
    """One fact the extractor returns, and what the user meant by it."""

    info: TextInfo
    op: str  # new | restate | reword | update | contradict
    tags: list[str]
    importance: float


@dataclass
class SlotState:
    stated: bool = False
    index: int = 0
    text: str = ""
    #: the last statement that was not a rewording of another
    said: str = ""
    refined: int = 0
    live: bool = False


class World:
    """The truth the stubs answer from, and the history the test replays."""

    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.texts: dict[str, TextInfo] = {}
        self.messages: dict[str, list[Plan]] = {}
        self.plans: dict[str, Plan] = {}
        self.aliases: dict[str, set[str]] = {
            wid: {name.casefold()} | {v.casefold() for v, _, _ in VARIANTS.get(wid, [])}
            for wid, (name, _, _) in ENTITIES.items()
        }
        self.names = {wid: name for wid, (name, _, _) in ENTITIES.items()}
        self.state = {slot.id: SlotState() for slot in SLOTS}
        self.episodes: dict[str, int] = {}
        self.week = 0
        values = [v.strip("'") for slot in SLOTS for v in slot.values]
        assert len(values) == len(set(values)), "slot values must be unique"

    # -- names ----------------------------------------------------------
    def owners_of(self, name: str) -> set[str]:
        key = name.strip().casefold()
        return {wid for wid, names in self.aliases.items() if key in names}

    def rename(self, wid: str, name: str) -> None:
        self.names[wid] = name
        self.aliases[wid].add(name.casefold())

    def surface(self, wid: str, rng: random.Random) -> str:
        options = [(self.names[wid], 1.0)] + [
            (name, weight) for name, weight, week in VARIANTS.get(wid, []) if self.week >= week]
        if wid == "ember" and self.names[wid] != ENTITIES["ember"][0]:
            options.append((ENTITIES["ember"][0], 0.4))  # the old name is still used
        names, weights = zip(*options)
        return rng.choices(names, weights)[0]

    def etype(self, slot: Slot | None, wid: str) -> str:
        return dict(slot.types).get(wid, ENTITIES[wid][1]) if slot else ENTITIES[wid][1]

    # -- texts ------------------------------------------------------------
    def render(self, slot: Slot, value: str, rng: random.Random, suffix: str = "",
               day: str = "") -> TextInfo:
        template = slot.template + suffix
        fields = [f for _, f, _, _ in string.Formatter().parse(template) if f]
        surfaces = {"v": value, "d": day}
        refs: list[Ref] = []
        for name in fields:
            if name in ("v", "d") or name in surfaces:
                continue
            wid = slot.subject if name == "E" else name
            surfaces[name] = self.surface(wid, rng)
            refs.append(Ref(surfaces[name], wid, self.etype(slot, wid)))
        text = template.format(**surfaces) + "."
        text = text[0].upper() + text[1:]
        return self.register(TextInfo(text, slot.id, value, refs, list(slot.relations)))

    def register(self, info: TextInfo) -> TextInfo:
        self.texts[norm(info.text)] = info
        return info

    def info(self, text: str) -> TextInfo | None:
        return self.texts.get(norm(text))

    def scan(self, text: str) -> TextInfo:
        """What an extractor reads in a text it was not planned for: the
        world's names in it, the longest first."""
        known = self.info(text)
        if known is not None:
            return known
        refs: list[Ref] = []
        taken: list[tuple[int, int]] = []
        names = sorted({(name, wid) for wid, names in self.aliases.items() for name in names},
                       key=lambda item: -len(item[0]))
        for name, wid in names:
            for match in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
                if any(a < match.end() and match.start() < b for a, b in taken):
                    continue
                taken.append((match.start(), match.end()))
                refs.append(Ref(match.group(0), wid, ENTITIES[wid][1]))
        return self.register(TextInfo(text, None, None, refs))

    def plan(self, slot: Slot, op: str, rng: random.Random) -> Plan:
        state = self.state[slot.id]
        if op in ("new", "contradict"):
            state.index = 0 if op == "new" and not state.stated else (
                (state.index + 1) % len(slot.values))
            info = self.render(slot, slot.values[state.index], rng)
            state.refined = 0
        elif op == "restate":
            info = self.info(state.said)
        elif op == "reword":
            info = self.register(TextInfo(
                "As mentioned before: " + state.said, slot.id, slot.values[state.index],
                self.info(state.said).refs, list(slot.relations)))
        elif op == "update":
            suffix = REFINEMENTS[state.refined % len(REFINEMENTS)]
            state.refined += 1
            base = self.info(state.said)
            extra = self.render(Slot("x", slot.subject, suffix.lstrip(", "), ("",), ()), "", rng)
            text = state.said.rstrip(".") + ", " + extra.text[0].lower() + extra.text[1:]
            info = self.register(TextInfo(text, slot.id, slot.values[state.index],
                                          base.refs + extra.refs, list(slot.relations)))
        else:
            raise ValueError(op)
        state.stated, state.live, state.text = True, True, info.text
        if op != "reword":
            state.said = info.text
        tags = list(slot.tags)
        if rng.random() < 0.15 and tags[0].endswith("s"):
            tags[0] = tags[0][:-1]  # a plural written singular
        plan = Plan(info, op, tags, slot.importance)
        self.plans[norm(info.text)] = plan
        return plan

    def episode(self, family: Slot, rng: random.Random, at: datetime) -> Plan:
        """A new fact of a family, dated the day it is saved."""
        n = self.episodes[family.id] = self.episodes.get(family.id, 0) + 1
        day = f"{at:%B} {at.day}"
        for i in range(len(family.values)):
            value = family.values[(n + i) % len(family.values)]
            info = self.render(family, value, rng, day=day)
            if norm(info.text) not in self.plans:
                break
        info.slot, info.value = f"{family.id}#{n}", None
        plan = Plan(info, "new", list(family.tags), 0.5)
        self.plans[norm(info.text)] = plan
        return plan

    def value_of(self, slot_id: str) -> str | None:
        state = self.state[slot_id]
        return SLOT[slot_id].values[state.index].strip("'") if state.live else None

    # -- identity truth -----------------------------------------------------
    def side(self, name: str, facts: list[str]) -> str:
        """The world thing one side of a comparison is: the one its facts
        call by its name, by majority."""
        key = name.strip().casefold()
        votes: Counter[str] = Counter()
        for fact in facts:
            info = self.info(fact)
            if info is None:
                continue
            ids = {r.wid for r in info.refs if key in self.aliases[r.wid]}
            ids = ids or {r.wid for r in info.refs if r.surface.casefold() == key}
            ids = ids or {r.wid for r in info.refs
                          if key in r.surface.casefold() or r.surface.casefold() in key}
            votes.update(ids)
        if votes:
            best = max(votes.values())
            return sorted(w for w, n in votes.items() if n == best)[0]
        owners = sorted(self.owners_of(name))
        return owners[0] if len(owners) == 1 else f"?{key}"

    def relation(self, a: str, b: str) -> str | None:
        """"a_kind_of_b" and the like, when one belongs to the other."""
        if BELONGS.get(a, (None,))[0] == b:
            return f"a_{BELONGS[a][1]}_of_b"
        if BELONGS.get(b, (None,))[0] == a:
            return f"b_{BELONGS[b][1]}_of_a"
        return None


# ------------------------------------------------------------ call counting
class Calls:
    """Every model call, by kind, attributed to the operation running."""

    IDENTITY = {"pair", "name_check", "identity_choice", "identity_text"}

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.op: str = "setup"
        self.by_op: dict[str, Counter[str]] = defaultdict(Counter)
        self.total: Counter[str] = Counter()
        #: the save's resolution in progress: memory id, and the name compared
        self.resolving: dict[str, Any] | None = None
        self.comparing: dict[str, Any] | None = None
        self.identity_events: list[dict[str, Any]] = []
        self.budget: Counter[str] = Counter()
        self.question: tuple[str, str] | None = None

    def count(self, kind: str, detail: str = "") -> None:
        with self.lock:
            self.by_op[self.op][kind] += 1
            self.total[kind] += 1
            if kind in self.IDENTITY:
                self.identity_events.append({
                    "op": self.op, "kind": kind, "detail": detail[:160],
                    "resolving": dict(self.resolving) if self.resolving else None,
                    "comparing": dict(self.comparing) if self.comparing else None,
                })


class CountingEmbedder(HashEmbedder):
    def __init__(self, calls: Calls) -> None:
        super().__init__(128)
        self.calls = calls

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self.calls.lock:
            self.calls.by_op[self.calls.op]["embedded_texts"] += len(texts)
        return super().embed(texts)


def _json_after(marker: str, text: str) -> Any:
    start = text.find(marker)
    if start < 0:
        return None
    rest = text[start + len(marker):]
    depth, end = 0, None
    for i, ch in enumerate(rest):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(rest[:end]) if end else None


ENTITY_OFFER = ("write the name as the conversation does:\n")
OWNER_OFFER = re.compile(
    r'The person these memories belong to \(the user\) is the entity "([^"]*)"')


class WorldLLM(LLM):
    """The text model."""

    name = "world-text"
    available = True

    def __init__(self, world: World, calls: Calls) -> None:
        self.world = world
        self.calls = calls
        self.store: MemoryStore | None = None
        self.judged = False  # set when a calibrated judge decides identity
        self.typed: WorldJudge | None = None

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            return self._extract(user)
        if system.startswith("You audit what a memory system stored"):
            self.calls.count("coverage")
            return json.dumps({"missing": []})
        if system.startswith("You maintain an AI assistant's long-term memory store"):
            return self._reconcile(user)
        if system.startswith("You resolve entity identity"):
            name = re.search(r'NEW fact mentioning "([^"]*)"', user)
            self.calls.count("identity_text", name.group(1) if name else user[:80])
            rng = stable_rng("identity", user)
            return json.dumps({"verdict": rng.choice(["same", "unsure", "different"]),
                               "confidence": round(rng.uniform(0.5, 0.99), 2),
                               "reason": "a guess"})
        if system.startswith("You answer typed questions about a piece of state"):
            return self._typed(user)
        if system.startswith("You review the entity list"):
            self.calls.count("referent_review")
            return json.dumps({"junk": []})
        if system.startswith("You organize a personal memory system's tags"):
            self.calls.count("synthetic_tags")
            known = re.findall(r"^- (.*) \(\d+\)$", user, re.MULTILINE)
            members = [t for t in ("kitchen renovation", "apartment", "cat care") if t in known]
            clusters = [{"tag": "home life", "members": members}] if len(members) >= 2 else []
            return json.dumps({"clusters": clusters})
        if system.startswith("You are consolidating an AI assistant's long-term memory"):
            self.calls.count("consolidate")
            texts = re.findall(r"^\[\d+\] (.*)$", user, re.MULTILINE)
            infos = [self.world.info(t) for t in texts]
            same = (all(infos) and len({(i.slot, i.value) for i in infos}) == 1
                    and infos[0].slot is not None)
            return json.dumps({"same_fact": bool(same),
                               "content": max(texts, key=len) if same else "",
                               "reason": "one fact" if same else "different facts"})
        if system.startswith("Write a compact, evidence-grounded description"):
            self.calls.count("description")
            return json.dumps({"description": "A thing from the world."})
        if system.startswith("You de-duplicate a tag vocabulary"):
            self.calls.count("canonicalize")
            return json.dumps({"groups": []})
        if system.startswith("You read stored memories and say when"):
            self.calls.count("when")
            return json.dumps({"items": []})
        if system.startswith("Given a statement and the entities in it"):
            self.calls.count("relations")
            return json.dumps({"relations": []})
        if system.startswith("Classify each entity name"):
            self.calls.count("types")
            return json.dumps({"types": []})
        raise AssertionError(f"the text model got a prompt the test does not know: {system[:80]}")

    # -- typed questions ------------------------------------------------------
    def _typed(self, user: str) -> str:
        """The decision questions, when an operator sends them to the text
        model (``providers.decisions.LLMDecider``): answered as the judge
        would, except identity, which the text model guesses."""
        head, spec = user.split("\n\nQUESTIONS:\n", 1)
        state = head.split("STATE:\n", 1)[1]
        spec = json.loads(spec.rsplit("\n\nJSON only.", 1)[0])
        questions: dict[str, Any] = {}
        for key, q in spec.items():
            if q["type"] == "choice":
                questions[key] = Choice(q["instructions"], q["options"])
            elif q["type"] == "score":
                questions[key] = Score(q["instructions"], q["levels"])
            else:
                questions[key] = Noul(q["instructions"])
        if self.typed is None:
            self.typed = WorldJudge(self.world, self.calls, guesses_identity=True)
        answers = self.typed.decide(state, questions)
        out = {}
        for key, question in questions.items():
            answer = answers[key]
            if not answer.available:
                continue
            value = round(answer.value) if isinstance(question, Score) else answer.value
            out[key] = {"answer": value, "confidence": round(answer.confidence, 3)}
        return json.dumps(out)

    # -- extraction ---------------------------------------------------------
    def _extract(self, user: str) -> str:
        self.calls.count("extract")
        transcript = user.split("Conversation:\n", 1)[1].split("\n\n", 1)[0]
        offered = _json_after(ENTITY_OFFER, user) or []
        owner = OWNER_OFFER.search(user)
        plans = []
        # one numbered line per message ("[1] user: ..."): a deferred save's
        # group is read together
        for line in transcript.splitlines():
            line = re.sub(r"^\[\d+\] ", "", line)
            message = line.split(": ", 1)[1] if line.startswith("user: ") else line
            planned = self.world.messages.get(message)
            # else a memory's own text, read again after an edit
            plans += planned or [Plan(self.world.scan(message), "reread", [], 0.6)]
        facts = []
        for plan in plans:
            rng = stable_rng("names", plan.info.text)
            written: dict[str, str] = {}
            entities = []
            for ref in plan.info.refs:
                name = self._written(ref, offered, owner.group(1) if owner else None, rng)
                written.setdefault(ref.wid, name)
                entities.append({"name": name, "type": ref.etype})
                self._budget(name)
            relations = [{"subject": written[s], "predicate": p, "object": written[o]}
                         for s, p, o in plan.info.relations if s in written and o in written]
            facts.append({"content": plan.info.text, "type": "semantic",
                          "importance": plan.importance, "categories": plan.tags,
                          "entities": entities, "relations": relations, "when": None})
        return json.dumps({"facts": facts})

    def _written(self, ref: Ref, offered: list[dict], owner: str | None,
                 rng: random.Random) -> str:
        """The name a real extractor writes: the owner's as offered; a stored
        name the store offers for another way of writing this one, most of
        the time; otherwise the name as the text writes it."""
        if ref.wid == "owner" and owner:
            return owner
        words = set(re.findall(r"\w{3,}", ref.surface.casefold()))
        # a listed name a reader can tell is this one: it names this thing
        # and nothing else, and shares a word with the text's ("Nordvik" and
        # "Nordvik Labs"; not "Tomi" and "Tomas Vell")
        mine = [o["name"] for o in offered if self.world.owners_of(o["name"]) == {ref.wid}
                and words & set(re.findall(r"\w{3,}", o["name"].casefold()))]
        if any(name.casefold() == ref.surface.casefold() for name in mine):
            return ref.surface  # listed as the text writes it
        if mine and rng.random() < 0.8:
            return mine[0]
        return ref.surface

    def _budget(self, name: str) -> None:
        """What the design lets a save ask about this name (``Checker.save_budget``)."""
        if self.store is None or not self.judged:
            return
        backend = self.store.backend
        found = backend.find_entity_candidates(name.strip().lower(), Scope(user_id=USER))
        ids = {e.id for e in found}
        open_pairs = [p for p in backend.proposals_of(sorted(ids)) if p.status == "proposed"]
        with self.calls.lock:
            self.calls.budget["compare"] += len(found) + CANDIDATES_PER_NAME
            self.calls.budget["recheck"] += len(open_pairs)

    # -- reconciliation -----------------------------------------------------
    def _reconcile(self, user: str) -> str:
        listing, new = parse_reconcile_state(user)
        if "The action is decided: UPDATE memory [0]" in user:
            self.calls.count("merge_text")
            plan = self.world.plans.get(norm(new))
            return json.dumps({"action": "UPDATE", "target": 0,
                               "content": plan.info.text if plan else f"{listing[0]} {new}",
                               "reason": "merged"})
        self.calls.count("reconcile_text")
        action, target = decide_action(self.world, listing, new)
        content = self.world.plans[norm(new)].info.text if action == "UPDATE" else None
        return json.dumps({"action": action, "target": target, "content": content,
                           "reason": "world"})


def parse_reconcile_state(state: str) -> tuple[list[str], str]:
    head, new = state.split("\n\nNEW fact:\n", 1)
    new = new.split("\n\n", 1)[0].strip()
    listing = re.findall(r"^\[\d+\] (.*)$", head, re.MULTILINE)
    return listing, new


def decide_action(world: World, listing: list[str], new: str) -> tuple[str, int | None]:
    plan = world.plans.get(norm(new))
    if plan is None or plan.op == "new":
        return "ADD", None
    slots = [(world.info(text).slot if world.info(text) else None) for text in listing]
    target = next((i for i, slot in enumerate(slots) if slot == plan.info.slot), None)
    if target is None:
        return "ADD", None  # the model cannot act on what it was not shown
    return {"restate": "NONE", "reword": "NONE", "update": "UPDATE",
            "contradict": "DELETE"}[plan.op], target


# ------------------------------------------------------------ the judges
SIDE = re.compile(r'ENTITY ([AB]): "([^\n]*?)" \(([^)\n]*)\)\n(.*?)(?=\n\nENTITY [AB]: |\Z)',
                  re.DOTALL)
FACT = re.compile(r"^- (?:\[[^\]]*\] )?(.*)$")


def parse_sides(state: str) -> dict[str, tuple[str, str, list[str]]]:
    sides = {}
    for label, name, kind, body in SIDE.findall(state):
        facts, reading = [], False
        for line in body.splitlines():
            if line == "Facts:":
                reading = True
                continue
            if reading and line.startswith("- ") and line != "- (no facts)":
                facts.append(FACT.match(line).group(1))
            elif reading and not line.startswith("- "):
                break
        sides[label] = (name, kind, facts)
    return sides


class WorldJudge(Decider):
    """A calibrated judge that answers from the world's truth, with noise."""

    name = "world-judge"
    available = True
    calibrated = True
    reranks_by_default = True
    may_rerank = True
    rejudges_on_new_evidence = True
    auto_confirm_confidence = 0.7
    pair_merge_probability = 0.95
    pair_merge_by_step = {1: 0.97, 3: 0.96, 10: 0.85, 50: 0.80}
    tag_merge_probability = 0.55

    def __init__(self, world: World, calls: Calls, *, guesses_identity: bool = False) -> None:
        self.world = world
        self.calls = calls
        self.guesses_identity = guesses_identity

    def decide(self, state: str, questions: dict) -> Answers:
        keys = set(questions)
        if "pair" in keys:
            return self._pair(state, questions)
        if "action" in keys:
            return self._action(state, questions)
        if "identity" in keys:
            name = re.search(r'NEW fact mentioning "([^"]*)"', state)
            self.calls.count("identity_choice", name.group(1) if name else state[:120])
            if self.guesses_identity:
                rng = stable_rng("identity", state)
                return Answers({"identity": _choice(rng.choice(IDENTITY_OPTIONS),
                                                    round(rng.uniform(0.5, 0.99), 2),
                                                    IDENTITY_OPTIONS)})
            return Answers({"identity": _choice("unsure", 0.5, IDENTITY_OPTIONS)})
        if "tag" in keys:
            self.calls.count("tag_pair")
            tags = re.findall(r'^TAG [AB]: "(.*)" \(on', state, re.MULTILINE)
            same = frozenset(tags) in TAG_SYNONYMS
            p = 0.9 if same else 0.08
            return Answers({"tag": Answer("same" if same else "different",
                                          {"same": p, "different": 1 - p}, max(p, 1 - p), True)})
        if "same_fact" in keys:
            self.calls.count("same_fact")
            texts = re.findall(r"^- (.*)$", state, re.MULTILINE)
            infos = [self.world.info(t) for t in texts]
            same = all(infos) and len({(i.slot, i.value) for i in infos}) == 1
            value = 0.9 if same else 0.1
            return Answers({"same_fact": Answer(value, {}, abs(value - 0.5) * 2, True)})
        first = questions[sorted(keys)[0]]
        if isinstance(first, Score):
            self.calls.count("durability")
            return Answers({k: Answer(1.5, {}, 0.8, True) for k in keys})
        if isinstance(first, Noul):
            return self._relevance(state, questions)
        if isinstance(first, Choice) and "named_thing" in first.criteria:
            self.calls.count("screen")
            out = {}
            for key, q in questions.items():
                name = re.search(r'what is "(.*)"\?', q.instructions).group(1)
                known = bool(self.world.owners_of(name))
                verdict = "named_thing" if known else "generic_topic"
                out[key] = _choice(verdict, 0.96 if known else 0.6, list(first.criteria))
            return Answers(out)
        if isinstance(first, Choice) and "possible" in first.criteria:
            return self._name_check(state, questions)
        if isinstance(first, Choice) and "person" in first.criteria:
            self.calls.count("types")
            out = {}
            for key, q in questions.items():
                name = re.search(r'What kind of thing is "(.*)"\?', q.instructions).group(1)
                owners = self.world.owners_of(name)
                kind = ENTITIES[sorted(owners)[0]][1] if owners else "other"
                out[key] = _choice(kind, 0.9, list(first.criteria))
            return Answers(out)
        if isinstance(first, Choice) and "event" in first.criteria:
            self.calls.count("when_confirm")
            return Answers({k: _choice("event", 0.7, list(first.criteria)) for k in keys})
        raise AssertionError(f"the judge got a question the test does not know: {keys}")

    # -- identity -------------------------------------------------------------
    def _pair(self, state: str, questions: dict) -> Answers:
        sides = parse_sides(state)
        (name_a, kind_a, facts_a), (name_b, kind_b, facts_b) = sides["A"], sides["B"]
        a, b = self.world.side(name_a, facts_a), self.world.side(name_b, facts_b)
        self.calls.count("pair", f"{name_a} [{a}] / {name_b} [{b}]")
        rng = stable_rng("pair", state)
        related = self.world.relation(a, b)
        if a == b and not a.startswith("?") and kind_a != kind_b and TOPIC_TYPE not in (
                kind_a, kind_b):
            # one thing typed two ways: a reader of both hesitates
            same, different = rng.uniform(0.7, 0.95), rng.uniform(0.02, 0.25)
            different = min(different, 1 - same)
        elif a == b and not a.startswith("?"):
            same = rng.uniform(0.9, 1.0) if rng.random() < 0.85 else rng.uniform(0.55, 0.9)
            different = rng.uniform(0.0, min(0.05, 1 - same))
        elif related:
            same = rng.uniform(0.6, 0.8)
            different = rng.uniform(0.1, 0.2)
        else:
            if rng.random() < 0.85:
                same, different = rng.uniform(0.0, 0.2), rng.uniform(0.6, 0.95)
            else:
                same, different = rng.uniform(0.2, 0.45), rng.uniform(0.3, 0.5)
            different = min(different, 1 - same)
        pair = {"same": same, "different": different, "unsure": max(0.0, 1 - same - different)}
        belongs = dict.fromkeys(BELONGS_OPTIONS, 0.01)
        if related:
            belongs[related] = rng.uniform(0.85, 0.95)
        else:
            belongs["neither"] = rng.uniform(0.9, 0.97)
        total = sum(belongs.values())
        belongs = {k: v / total for k, v in belongs.items()}
        out = {"pair": Answer(max(pair, key=pair.get), pair, max(pair.values()), True)}
        if "belongs" in questions:
            out["belongs"] = Answer(max(belongs, key=belongs.get), belongs,
                                    max(belongs.values()), True)
        return Answers(out)

    def _name_check(self, state: str, questions: dict) -> Answers:
        entity = re.search(r'store: "(.*)" \(', state).group(1)
        out = {}
        for key, q in questions.items():
            other = re.search(r'Could "(.*)" \(', q.instructions).group(1)
            self.calls.count("name_check", f"{entity} / {other}")
            same = self.world.owners_of(entity) & self.world.owners_of(other)
            p = 0.1 if same else 0.95
            out[key] = Answer("different" if p > 0.5 else "possible",
                              {"possible": 1 - p, "different": p}, max(p, 1 - p), True)
        return Answers(out)

    # -- reconciliation ---------------------------------------------------------
    def _action(self, state: str, questions: dict) -> Answers:
        self.calls.count("reconcile_action")
        listing, new = parse_reconcile_state(state)
        action, target = decide_action(self.world, listing, new)
        out = {"action": _choice(action, 0.95, list(questions["action"].criteria))}
        if "target" in questions:
            out["target"] = _choice(str(target or 0), 0.95, list(questions["target"].criteria))
        return Answers(out)

    # -- search -----------------------------------------------------------------
    def _relevance(self, state: str, questions: dict) -> Answers:
        self.calls.count("relevance")
        wanted = self.calls.question[1] if self.calls.question else None
        out = {}
        for key, q in questions.items():
            if key == "property":
                value = 0.9
            elif key == "several":
                value = 0.1
            else:
                memory = q.instructions.split("Memory: ", 1)[-1]
                value = 0.92 if wanted and contains_name(memory, wanted) else 0.04
            out[key] = Answer(value, {}, abs(value - 0.5) * 2, True)
        return Answers(out)


class SearchJudge(WorldJudge):
    """The relevance judge alone, swapped in for the search checks of a store
    that has no decision provider. It decides nothing else."""

    calibrated = False
    rejudges_on_new_evidence = False

    def decide(self, state: str, questions: dict) -> Answers:
        first = questions[sorted(questions)[0]]
        if not isinstance(first, Noul):
            raise AssertionError(f"the search judge was asked {sorted(questions)}")
        return self._relevance(state, questions)


IDENTITY_OPTIONS = ["same", "different", "unsure"]
BELONGS_OPTIONS = ["a_kind_of_b", "a_part_of_b", "b_kind_of_a", "b_part_of_a", "neither"]


def _choice(value: str, p: float, options: list[str]) -> Answer:
    rest = (1 - p) / max(len(options) - 1, 1)
    probabilities = {o: (p if o == value else rest) for o in options}
    return Answer(value, probabilities, p, True)


# ------------------------------------------------------------ determinism
class _Clock:
    now = START


CLOCK = _Clock()


class _FakeDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102
        return CLOCK.now if tz is not None else CLOCK.now.replace(tzinfo=None)


class _Ids:
    def __init__(self) -> None:
        self.n = 0
        self.lock = threading.Lock()

    def __call__(self) -> uuid.UUID:
        with self.lock:
            self.n += 1
            return uuid.UUID(int=(0x5EED << 100) | self.n)


# ------------------------------------------------------------ the checks
@dataclass
class Violation:
    invariant: str
    kind: str
    op: str
    detail: dict[str, Any]
    #: for a violation of the store's state: the check that no longer found
    #: it, None while it lasts
    resolved: str | None = None


class Checker:
    """The invariants, read straight from the database."""

    def __init__(self, store: MemoryStore, world: World, calls: Calls, judged: bool) -> None:
        self.store = store
        self.world = world
        self.calls = calls
        self.judged = judged
        #: whether a decision provider answers the typed questions at all
        self.decides = store.decider.available
        self.db = store.backend._db
        self.violations: list[Violation] = []
        self.seen: dict[tuple, Violation] = {}
        self.found: set[tuple] | None = None
        self.identity_checked = 0
        #: how often each documented exception was taken
        self.notes: Counter[str] = Counter()
        #: merges a person undid: (entity merged away, merged at, undone at)
        self.undone: list[tuple[str, str, str]] = []
        #: pairs a person kept apart (``MemoryStore.reject_merge``)
        self.rejected_by_person: set[str] = set()

    #: The invariants about the store's state, whose violations can go away.
    STATE = ("1 ", "2 ", "2b", "5 ", "6 ", "7 ")

    def add(self, invariant: str, kind: str, key: tuple, detail: dict[str, Any]) -> None:
        """Record a violation once per state it describes (``key``)."""
        full = (invariant, kind, key)
        if self.found is not None:
            self.found.add(full)
        if full in self.seen:
            self.seen[full].resolved = None
            return
        self.seen[full] = Violation(invariant, kind, self.calls.op, detail)
        self.violations.append(self.seen[full])

    # -- helpers ----------------------------------------------------------------
    def rows(self, sql: str, *params: Any) -> list[Any]:
        return self.db.execute(sql, params).fetchall()

    def world_of_memory(self, content: str) -> list[str]:
        info = self.world.info(content)
        return sorted({r.wid for r in info.refs}) if info else ["?"]

    def purity(self, entity_id: str) -> dict[str, int]:
        entity = self.store.backend.get_entity(entity_id)
        out: Counter[str] = Counter()
        for memory in self.store.backend.entity_memories(entity_id, limit=10_000):
            out[self.world.side(entity.name, [memory.content])] += 1
        return dict(out)

    # -- all of them --------------------------------------------------------------
    def check(self, *, vectors: bool = True) -> None:
        self.found = set()
        try:
            self.one_entity_per_name()
            self.linked_to_named()
            self.names_in_text()
            self.identity_questions()
            self.tags_agree()
            if vectors:
                self.property_vectors()
            self.no_orphans()
        finally:
            ran = [p for p in self.STATE if vectors or p != "6 "]
            for full, violation in self.seen.items():
                if (full[0][:2] in ran and full not in self.found
                        and violation.resolved is None):
                    violation.resolved = self.calls.op
            self.found = None

    def type_drift(self) -> str:
        """Where the memories went whose text the extractor typed the shop a
        product (it is a project otherwise), and how many product entities
        of its name were made."""
        joined: Counter[str] = Counter()
        for row in self.rows(
                "SELECT m.content, e.entity_type, em.decided FROM memories m "
                "JOIN entity_mentions em ON em.memory_id = m.id "
                "JOIN entities e ON e.id = em.entity_id WHERE e.normalized = 'quirkwear' "
                "AND e.entity_type != ? AND m.invalid_at IS NULL", TOPIC_TYPE):
            info = self.world.info(row["content"])
            if info and any(r.wid == "quirk" and r.etype == "product" for r in info.refs):
                reason = json.loads(row["decided"])["reason"] if row["decided"] else "made it"
                joined[f"{row['entity_type']}: {reason}"] += 1
        made = self.rows("SELECT COUNT(*) FROM entities WHERE normalized = 'quirkwear' "
                         "AND entity_type = 'product'")[0][0]
        return (f"type drift: {made} 'Quirkwear' product entities made; the memories "
                "typed so are on " + (", ".join(f"{k} ({n})" for k, n in joined.most_common())
                                      or "nothing"))

    def entity_summary(self) -> list[str]:
        """Every named entity at the end: type, memories, and which world
        things its memories are about."""
        out = []
        for entity in sorted(self.store.backend.list_entities(Scope(user_id=USER), limit=10_000),
                             key=lambda e: (e.normalized, e.created_at)):
            purity = self.purity(entity.id)
            out.append(f"{entity.name} ({entity.entity_type}) "
                       + ", ".join(f"{w}:{n}" for w, n in sorted(purity.items())))
        return out

    # 1 --------------------------------------------------------------------------
    def one_entity_per_name(self) -> None:
        backend = self.store.backend
        entities = self.rows(
            "SELECT id, name, normalized, entity_type, created_at FROM entities "
            "WHERE merged_into IS NULL AND IFNULL(entity_type, '') != ? AND user_id = ?",
            TOPIC_TYPE, USER)
        groups: dict[str, list[Any]] = defaultdict(list)
        for row in entities:
            groups[row["normalized"]].append(row)
        pairs: dict[frozenset, list[Any]] = defaultdict(list)
        for p in self.rows("SELECT * FROM entity_proposals WHERE user_id = ?", USER):
            a = backend.resolve_entity_id(p["entity_a"])
            b = backend.resolve_entity_id(p["entity_b"])
            if a and b and a != b:
                pairs[frozenset((a, b))].append(p)
        bar = self.store.decider.pair_apart_probability
        for name, members in groups.items():
            if len(members) < 2:
                continue
            for i, first in enumerate(members):
                for second in members[i + 1:]:
                    decided = pairs.get(frozenset((first["id"], second["id"])), [])
                    if any(self._kept_apart(p, bar) for p in decided):
                        continue
                    if self._made_while_undone(first, second):
                        # A merge was undone: the entity it brought back, and
                        # one made while the merge stood (the judge compared
                        # the new name with the two merged), have no answer
                        # of their own until the weekly pass pairs them.
                        self.notes["1: made while a merge later undone stood"] += 1
                        continue
                    detail = {
                        "name": name,
                        "types": [first["entity_type"], second["entity_type"]],
                        "memories": [backend.count_entity_memories(first["id"]),
                                     backend.count_entity_memories(second["id"])],
                        "world": [self.purity(first["id"]), self.purity(second["id"])],
                        "pairs": [{k: p[k] for k in ("status", "confidence", "different",
                                                      "compared_step", "reason")}
                                  for p in decided],
                        "made_by": self._made_by(second["id"]),
                    }
                    kind = ("type conflict" if first["entity_type"] and second["entity_type"]
                            and first["entity_type"] != second["entity_type"]
                            else "same type")
                    self.add("1 one entity per name", kind,
                             (first["id"], second["id"]), detail)

    def _kept_apart(self, proposal: Any, bar: float) -> bool:
        """Whether a pair's row says the two are two things: the calibrated
        judge's "different" (at the apart bar, or ruled out on the names), or
        a person's (kept apart, or a merge undone). A pair the text model
        rejected is not: without a calibrated judge its answer must not
        matter."""
        reason = proposal["reason"] or ""
        judge = self.store.decider.name if self.judged else None
        if proposal["status"] == "proposed":
            return bool(judge and proposal["different"] is not None
                        and proposal["different"] >= bar)
        if proposal["status"] != "rejected":
            return False
        if proposal["id"] in self.rejected_by_person or reason == "undone by you":
            return True
        if reason.startswith("kept apart: the tag was unfolded"):
            return True
        return bool(judge and (reason.startswith(judge)
                               or reason.startswith("the names alone rule it out")))

    def _made_while_undone(self, first: Any, second: Any) -> bool:
        for merged, since, until in self.undone:
            for back, other in ((first, second), (second, first)):
                if back["id"] == merged and since <= other["created_at"] <= until:
                    return True
        return False

    def _made_by(self, entity_id: str) -> list[str]:
        """What joined or made the entity's mentions (the reasons kept on them)."""
        reasons = Counter()
        for row in self.rows("SELECT decided FROM entity_mentions WHERE entity_id = ?", entity_id):
            reasons[json.loads(row["decided"])["reason"] if row["decided"] else "made it"] += 1
        return [f"{n} x {r}" for r, n in reasons.most_common()]

    # 2 --------------------------------------------------------------------------
    def linked_to_named(self) -> None:
        backend = self.store.backend
        entities = backend.list_entities(Scope(user_id=USER), limit=100_000)
        alias_of: dict[str, set[str]] = defaultdict(set)
        aliases: dict[str, list[str]] = {}
        for entity in entities:
            aliases[entity.id] = backend.entity_aliases(entity.id)
            for alias in aliases[entity.id]:
                if len(alias) >= 3:
                    alias_of[alias.casefold()].add(entity.id)
        memories = backend.list_memories(Scope(user_id=USER), limit=100_000)
        linked = backend.entities_of_memories([m.id for m in memories])
        screened = {e.id for e in entities if screened_out((e.metadata or {}).get("screen"))}
        # names whose link a person's removal of an entity took off a memory
        unlinked: dict[str, set[str]] = defaultdict(set)
        for row in self.rows("SELECT snapshot FROM retired_entities WHERE user_id = ?", USER):
            snapshot = json.loads(row["snapshot"])
            names = {a.casefold() for a in snapshot.get("aliases", [])}
            for mention in snapshot.get("mentions", []):
                unlinked[mention["memory_id"]] |= names
        for memory in memories:
            if (memory.metadata or {}).get("pending_distillation"):
                # stored verbatim by a deferred save, read for names when it
                # is distilled (``MemoryStore.add_deferred``)
                self.notes["2: waiting to be distilled"] += 1
                continue
            text = memory.content.casefold()
            mine = {e.id for e in linked[memory.id]}
            longer = [a for eid in mine for a in aliases.get(eid, [])]
            for alias, holders in alias_of.items():
                if alias not in text or not contains_name(memory.content, alias):
                    continue
                if holders & mine:
                    continue
                if alias in unlinked.get(memory.id, ()):
                    # ``MemoryStore.remove_entities``: the memories stay, their
                    # links go with the entity (and come back with a restore)
                    self.notes["2: its link went with an entity a person removed"] += 1
                    continue
                if non_referent_reason(alias) or holders <= screened:
                    self.notes["2: not a referent, or screened out"] += 1
                    continue
                if any(len(other) > len(alias) and contains_name(other, alias)
                       and contains_name(memory.content, other) for other in longer):
                    self.notes["2: inside a longer name it is linked to"] += 1
                    continue
                look_alike = [e for e in mine if any(
                    contains_name(alias, a) or contains_name(a, alias) for a in aliases.get(e, []))]
                if look_alike and (not self.judged or any(
                        backend.find_proposal(e, h) for e in look_alike for h in holders)):
                    # Two spellings of one name, two entities: without a judge
                    # only exact names meet (``entities.resolve_mentions``);
                    # with one, a pair of them waits in the funnel or was
                    # decided (``identity.compare``).
                    self.notes["2: linked to another spelling of the name"] += 1
                    continue
                kind = "linked to a look-alike instead" if look_alike else "not linked"
                self.add("2 linked to each name it contains", kind, (memory.id, alias), {
                    "memory": memory.content, "name": alias,
                    "entities_of_name": [backend.get_entity(e).name for e in sorted(holders)],
                    "linked_to": sorted(backend.get_entity(e).name for e in mine),
                    "history": [e.event for e in backend.history(memory.id)],
                })

    def names_in_text(self) -> None:
        """2b, the other way round: a memory is not linked to a thing its text
        does not name. Exceptions: the store owner (extraction lists the
        owner for any fact about the user), a mention a tag folded into the
        thing brought (its surface is one of the memory's tags), and the
        stored name extraction writes for another spelling of it ("Nordvik"
        written "Nordvik Labs", ``extraction.extract_facts``' entity offer):
        a word of the name is in the text."""
        backend = self.store.backend
        memories = backend.list_memories(Scope(user_id=USER), limit=100_000)
        linked = backend.entities_of_memories([m.id for m in memories])
        aliases: dict[str, list[str]] = {}
        for memory in memories:
            tags = {str(t).casefold() for t in memory.categories}
            for entity in linked[memory.id]:
                if (entity.metadata or {}).get("owner"):
                    continue
                if entity.id not in aliases:
                    aliases[entity.id] = backend.entity_aliases(entity.id)
                names = aliases[entity.id]
                if any(contains_name(memory.content, a) for a in names):
                    continue
                surfaces = {r["surface"].casefold() for r in self.rows(
                    "SELECT surface FROM entity_mentions WHERE entity_id = ? AND memory_id = ?",
                    entity.id, memory.id)}
                if surfaces <= tags:
                    self.notes["2b: a tag folded into the thing"] += 1
                    continue
                words = {w for a in names for w in re.findall(r"\w{4,}", a)}
                if any(contains_name(memory.content, w) for w in words):
                    self.notes["2b: the stored name written for another spelling"] += 1
                    continue
                self.add("2b linked only to names it contains", "linked to a name it does not "
                         "contain", (memory.id, entity.id), {
                             "memory": memory.content, "entity": entity.name,
                             "aliases": names, "surfaces": sorted(surfaces),
                             "history": [e.event for e in backend.history(memory.id)]})

    # 3 --------------------------------------------------------------------------
    def identity_questions(self) -> None:
        events = self.calls.identity_events[self.identity_checked:]
        self.identity_checked = len(self.calls.identity_events)
        for event in events:
            comparing = event["comparing"] or {}
            if not self.judged:
                where = "at save" if event["resolving"] else "in upkeep or a person's action"
                self.add("3 identity questions", f"without a judge, {where}",
                         (event["op"], event["kind"], event["detail"]), event)
            elif comparing.get("already_linked"):
                self.add("3 identity questions", "about a name the memory is linked to",
                         (event["op"], event["kind"], event["detail"]), event)

    # 5 --------------------------------------------------------------------------
    def tags_agree(self) -> None:
        backend = self.store.backend
        for row in self.rows("SELECT id, content, categories, invalid_at, user_id, agent_id, "
                             "run_id FROM memories WHERE user_id = ?", USER):
            column = [str(t) for t in json.loads(row["categories"])]
            wanted = sorted({t.strip().casefold() for t in column if t.strip()})
            mentions = sorted(r["normalized"] for r in self.rows(
                "SELECT e.normalized FROM entity_mentions em JOIN entities e "
                "ON e.id = em.entity_id WHERE em.memory_id = ? AND e.entity_type = ?",
                row["id"], TOPIC_TYPE))
            # A tag folded into the thing of its name is mentioned by the
            # thing, under the tag's name (``LocalBackend._mention_tags_locked``).
            folded = sorted(r["normalized"] for r in self.rows(
                "SELECT DISTINCT t.normalized FROM entity_mentions em JOIN entities t "
                "ON t.merged_into = em.entity_id AND t.entity_type = ? "
                "WHERE em.memory_id = ? AND lower(trim(em.surface)) = t.normalized",
                TOPIC_TYPE, row["id"]) if r["normalized"] in wanted)
            if folded:
                self.notes["5: a tag folded into the thing of its name"] += 1
                mentions = sorted(set(mentions) | set(folded))
            index = sorted(r["normalized"] for r in self.rows(
                "SELECT t.normalized FROM memory_topics mt JOIN topics t ON t.id = mt.topic_id "
                "WHERE mt.memory_id = ?", row["id"]))
            filed = backend.tag_filing(wanted, Scope(user_id=USER))
            retired = sorted(t for t in wanted if filed.get(t, t) != t)
            state = "invalid" if row["invalid_at"] else "active"
            if mentions != wanted or index != wanted or retired:
                self.add("5 tags agree", f"{state} memory",
                         (row["id"], tuple(wanted), tuple(mentions), tuple(index)), {
                             "memory": row["content"], "column": column,
                             "topic_mentions": mentions, "topic_index": index,
                             "names_merged_away": retired})

    # 6 --------------------------------------------------------------------------
    def property_vectors(self) -> None:
        backend, store = self.store.backend, self.store
        memories = backend.list_memories(Scope(user_id=USER), limit=100_000)
        linked = backend.entities_of_memories([m.id for m in memories])
        named = {m.id: [e.id for e in linked[m.id]] for m in memories if linked[m.id]}
        contents = {m.id: m.content for m in memories if m.id in named}
        masked = store._masked_texts(contents, named)
        stored = backend.property_vector_hashes(list(contents))
        label = store._property_label()
        for memory_id, text in masked.items():
            if text == contents[memory_id]:
                continue  # nothing masked: search reads the ordinary vector
            if stored.get(memory_id) != (_text_hash(text), label):
                kind = "missing" if memory_id not in stored else "stale"
                self.add("6 property vector", kind, (memory_id, _text_hash(text)), {
                    "memory": contents[memory_id], "masked": text,
                    "entities": [backend.get_entity(e).name for e in named[memory_id]]})

    # 7 --------------------------------------------------------------------------
    ORPHANS = {
        "mention of a memory that is gone":
            "SELECT em.id, em.surface FROM entity_mentions em "
            "LEFT JOIN memories m ON m.id = em.memory_id WHERE m.id IS NULL",
        "mention of an entity that is gone":
            "SELECT em.id, em.surface FROM entity_mentions em "
            "LEFT JOIN entities e ON e.id = em.entity_id WHERE e.id IS NULL",
        "mention of an entity merged away":
            "SELECT em.id, em.surface FROM entity_mentions em "
            "JOIN entities e ON e.id = em.entity_id WHERE e.merged_into IS NOT NULL",
        "relation of a memory that is gone":
            "SELECT r.id, r.predicate FROM relations r LEFT JOIN memories m "
            "ON m.id = r.memory_id WHERE r.memory_id IS NOT NULL AND m.id IS NULL",
        "relation to an entity that is gone or merged away":
            "SELECT r.id, r.predicate FROM relations r LEFT JOIN entities s ON s.id = r.subject "
            "LEFT JOIN entities o ON o.id = r.object WHERE s.id IS NULL OR o.id IS NULL "
            "OR s.merged_into IS NOT NULL OR o.merged_into IS NOT NULL",
        "relation in use whose memory is out of use":
            "SELECT r.id, r.predicate FROM relations r JOIN memories m ON m.id = r.memory_id "
            "WHERE r.invalid_at IS NULL AND m.invalid_at IS NOT NULL",
        "proposal naming an entity that is gone":
            "SELECT p.id, p.status FROM entity_proposals p "
            "LEFT JOIN entities a ON a.id = p.entity_a LEFT JOIN entities b ON b.id = p.entity_b "
            "WHERE a.id IS NULL OR b.id IS NULL",
        "property vector of a memory that is gone":
            "SELECT v.memory_id, '' FROM memory_property_vectors v "
            "LEFT JOIN memories m ON m.id = v.memory_id WHERE m.id IS NULL",
        "topic link of a memory or topic that is gone":
            "SELECT mt.memory_id, mt.topic_id FROM memory_topics mt "
            "LEFT JOIN memories m ON m.id = mt.memory_id LEFT JOIN topics t ON t.id = mt.topic_id "
            "WHERE m.id IS NULL OR t.id IS NULL",
        "supersede pointer to a memory that is gone":
            "SELECT m.id, m.superseded_by FROM memories m LEFT JOIN memories s "
            "ON s.id = m.superseded_by WHERE m.superseded_by IS NOT NULL AND s.id IS NULL",
        "merge record of an entity that is gone":
            "SELECT g.id, g.merge_id FROM entity_merges g LEFT JOIN entities k ON k.id = g.keep_id "
            "LEFT JOIN entities m ON m.id = g.merge_id WHERE k.id IS NULL OR m.id IS NULL",
        "tombstone pointing at an entity that is gone":
            "SELECT e.id, e.name FROM entities e LEFT JOIN entities t ON t.id = e.merged_into "
            "WHERE e.merged_into IS NOT NULL AND t.id IS NULL",
    }

    def no_orphans(self) -> None:
        for kind, sql in self.ORPHANS.items():
            for row in self.rows(sql):
                self.add("7 no orphan rows", kind, (row[0],), {"row": [row[0], row[1]]})
        # A pair has one row: a merge points the merged entity's pairs at the
        # kept one, and a pair the kept one had already must not become two.
        # (A confirmed row keeps the ends it was decided on, by design.)
        rows: dict[frozenset, list[Any]] = defaultdict(list)
        for p in self.rows("SELECT * FROM entity_proposals WHERE status != 'confirmed'"):
            rows[frozenset((p["entity_a"], p["entity_b"]))].append(p)
        backend = self.store.backend
        for pair, found in rows.items():
            if len(found) > 1:
                names = sorted((backend.get_entity(e).name if backend.get_entity(e) else e)
                               for e in pair)
                self.add("7 no orphan rows", "two rows for one pair",
                         tuple(sorted(p["id"] for p in found)), {
                             "entities": names,
                             "rows": [{k: p[k] for k in ("status", "confidence", "different",
                                                          "compared_step", "reason",
                                                          "created_at")} for p in found]})

    # 4 --------------------------------------------------------------------------
    def save_budget(self, facts: int, updates: int, counts: Counter[str],
                    messages: int = 1) -> None:
        """The calls a save may make, from the design:

        * text model: 1 extraction and 1 coverage audit (per group of
          messages a deferred save distills together), and per fact 1
          reconcile (only without a decision provider), and per UPDATE 1
          merged text (only with one: it decides the action, the text model
          writes) and 1 re-extraction of the new text;
        * the typed questions go to the decision provider, the text model
          when an operator sends them there (``LLMDecider``), and count as
          the provider's;
        * decision provider: per fact 1 action and 1 screen of names new to
          the store, and 1 more screen per UPDATE;
        * identity: none without a calibrated judge. With one, per name
          compared, at most 6 questions per candidate (2 orders, at a step
          of 10 memories and one of 50, and once in context), the candidates
          being the entities of that name and ``CANDIDATES_PER_NAME`` more,
          and the same 6 for each open pair of those entities the save
          compares again.
        """
        budget = self.calls.budget
        text = {k: counts[k] for k in ("extract", "coverage", "reconcile_text", "merge_text")}
        text_bound = 2 * messages + (0 if self.decides else facts) + 2 * updates
        decider = {k: counts[k] for k in ("reconcile_action", "screen", "when_confirm")}
        decider_bound = (2 * facts + updates) if self.decides else 0
        identity = sum(counts[k] for k in Calls.IDENTITY)
        identity_bound = 6 * (budget["compare"] + budget["recheck"]) if self.judged else 0
        other = {k: n for k, n in counts.items()
                 if k not in text and k not in decider and k not in Calls.IDENTITY
                 and k != "embedded_texts"}
        for label, got, bound in (("text model", sum(text.values()), text_bound),
                                  ("decision provider", sum(decider.values()), decider_bound),
                                  ("identity", identity, identity_bound)):
            if got > bound:
                self.add("4 calls per save", label, (self.calls.op, label), {
                    "calls": got, "bound": bound, "facts": facts, "updates": updates,
                    "by_kind": {k: n for k, n in counts.items() if n}})
        if other:
            self.add("4 calls per save", "a kind of call a save should not make",
                     (self.calls.op,), {"calls": other})

    # 8 --------------------------------------------------------------------------
    def search(self, week: int, judge: Decider | None) -> dict[str, int]:
        store = self.store
        own = store.decider
        if judge is not None:
            store.decider = judge
        found = asked = 0
        try:
            for slot in SLOTS:
                value = self.world.value_of(slot.id)
                if not slot.question or value is None:
                    continue
                asked += 1
                self.calls.question = (slot.question, value)
                results = store.search(slot.question, user_id=USER, limit=10)
                rank = next((i for i, r in enumerate(results)
                             if contains_name(r.memory.content, value)), None)
                if rank is not None and rank < 10:
                    found += 1
                    continue
                holders = [m.content for m in store.get_all(user_id=USER, limit=10_000)
                           if contains_name(m.content, value)]
                self.add("8 search", "answer not in the top 10", (week, slot.id), {
                    "week": week, "question": slot.question, "answer": value,
                    "rank": rank, "stored_as": holders[:3],
                    "top": [r.memory.content for r in results[:5]]})
        finally:
            self.calls.question = None
            store.decider = own
        return {"asked": asked, "found": found}


# ------------------------------------------------------------ the replay
@dataclass
class Replay:
    store: MemoryStore
    world: World
    calls: Calls
    checker: Checker
    judged: bool
    search_judge: Decider | None
    ops: int = 0
    saves: int = 0
    log: list[str] = field(default_factory=list)
    searches: list[dict[str, int]] = field(default_factory=list)
    pending: list[Plan] = field(default_factory=list)
    pending_saves: int = 0
    #: the person's actions that span weeks: entities removed, to bring back;
    #: the entity a spelling was merged into; a mistaken merge to undo
    removed: list[str] = field(default_factory=list)
    removed_later: list[str] = field(default_factory=list)
    cleaned: str | None = None
    mistake: str | None = None
    namesake: str | None = None

    def begin(self, label: str, at: datetime) -> None:
        self.ops += 1
        CLOCK.now = at
        self.calls.op = f"#{self.ops} w{self.world.week} {label}"
        self.calls.budget = Counter()
        self.log.append(self.calls.op)

    # -- operations ------------------------------------------------------------
    def save(self, plans: list[Plan], context: str, at: datetime) -> None:
        message = " ".join(plan.info.text for plan in plans)
        self.world.messages[message] = plans
        self.saves += 1
        ops = "+".join(plan.op for plan in plans)
        self.begin(f"save {self.saves} ({ops}): {message[:70]}", at)
        result = self.store.add(message, user_id=USER, agent_id=AGENT,
                                metadata={"context": context}, created_at=at.isoformat(),
                                now=at)
        updates = sum(1 for a in result.actions if a.event == "UPDATE")
        self.checker.save_budget(len(plans), updates, self.calls.by_op[self.calls.op])
        self.checker.check()

    def save_deferred(self, plans: list[Plan], context: str, at: datetime) -> None:
        """A save as the MCP server makes it: stored verbatim now, distilled
        with the rest of its conversation once it has gone quiet."""
        message = " ".join(plan.info.text for plan in plans)
        self.world.messages[message] = plans
        self.saves += 1
        self.pending += plans
        self.pending_saves += 1
        self.begin(f"deferred save {self.saves}: {message[:70]}", at)
        self.store.add_deferred(message, user_id=USER, agent_id=AGENT,
                                metadata={"context": context}, created_at=at.isoformat(),
                                now=at)
        self.checker.save_budget(0, 0, self.calls.by_op[self.calls.op])
        self.checker.check()

    def distill(self, at: datetime) -> None:
        if not self.pending:
            return
        self.begin(f"distill {len(self.pending)} facts", at)
        db = self.store.backend._db
        updates_sql = "SELECT COUNT(*) FROM memory_events WHERE event = 'UPDATE'"
        before = db.execute(updates_sql).fetchone()[0]
        outcome = self.store.process_pending_enrichments(limit=50, quiet_seconds=120, now=at)
        assert not outcome["failed"], outcome
        updates = db.execute(updates_sql).fetchone()[0] - before
        self.checker.save_budget(len(self.pending), updates, self.calls.by_op[self.calls.op],
                                 messages=self.pending_saves)
        self.pending, self.pending_saves = [], 0
        self.checker.check()

    def memory_of(self, slot_id: str) -> Any | None:
        for memory in self.store.get_all(user_id=USER, limit=10_000):
            info = self.world.info(memory.content)
            if info is not None and info.slot == slot_id:
                return memory
        return None

    def edit(self, slot_id: str, at: datetime, *, mode: str = "value") -> None:
        """A person's edit of a memory's text: a corrected value, a name
        added ("add"), or the subject's name taken out ("drop")."""
        memory = self.memory_of(slot_id)
        if memory is None:
            return
        slot, state = SLOT[slot_id], self.world.state[slot_id]
        rng = stable_rng("edit", slot_id, self.ops)
        base = self.world.info(memory.content)
        if mode == "add":
            surface = self.world.surface("tomas", rng)
            text = memory.content.rstrip(".") + EDIT_ADDS.format(tomas=surface) + "."
            info = self.world.register(TextInfo(
                text, slot_id, base.value, base.refs + [Ref(surface, "tomas", "person")],
                base.relations))
        elif mode == "drop":
            subject = next((r for r in base.refs if r.wid == slot.subject), None)
            if subject is None or not memory.content.startswith(subject.surface):
                return
            text = WITHOUT_NAME[ENTITIES[slot.subject][1]] + memory.content[len(subject.surface):]
            info = self.world.register(TextInfo(
                text, slot_id, base.value, [r for r in base.refs if r.wid != slot.subject], []))
        else:
            state.index = (state.index + 1) % len(slot.values)
            info = self.world.render(slot, slot.values[state.index], rng)
        state.text = state.said = info.text
        state.live = True
        self.world.plans[norm(info.text)] = Plan(info, "edit", list(slot.tags), slot.importance)
        self.begin(f"edit {slot_id}: {info.text[:70]}", at)
        self.store.update(memory.id, content=info.text)
        self.checker.check()

    def delete(self, slot_id: str, at: datetime, *, hard: bool = False) -> None:
        memory = self.memory_of(slot_id)
        if memory is None:
            return
        self.begin(f"{'hard ' if hard else ''}delete {slot_id}: {memory.content[:60]}", at)
        self.store.delete(memory.id, hard=hard)
        self.world.state[slot_id] = SlotState()  # said again later, it is new
        self.checker.check()

    def retag(self, slot_id: str, tags: list[str], at: datetime) -> None:
        memory = self.memory_of(slot_id)
        if memory is None:
            return
        self.begin(f"retag {slot_id}: {tags}", at)
        self.store.update(memory.id, categories=tags)
        self.checker.check()

    def act(self, label: str, at: datetime, action) -> Any:
        """Any other action of a person's, through the store's API."""
        self.begin(label, at)
        result = action()
        self.checker.check()
        return result

    def entity_named(self, name: str, wid: str | None = None) -> list[Any]:
        found = [e for e in self.store.entities(user_id=USER, limit=10_000)
                 if e.name.casefold() == name.casefold()]
        if wid is not None:
            found = [e for e in found
                     if max(self.checker.purity(e.id).items(), key=lambda kv: kv[1],
                            default=("", 0))[0] == wid]
        return found

    def merge(self, keep: str, other: str, at: datetime, label: str) -> str | None:
        keeps, others = self.entity_named(keep), self.entity_named(other)
        if not keeps or not others or keeps[0].id == others[0].id:
            return None
        self.begin(f"{label}: merge {other} into {keep}", at)
        self.store.merge_entities(keeps[0].id, others[0].id)
        self.checker.check()
        return others[0].id

    def spellings(self) -> list[tuple[Any, Any]]:
        """Pairs of entities that are one world thing under two names (as a
        person reading them knows), the one with more memories first."""
        by_thing: dict[str, list[Any]] = defaultdict(list)
        for entity in self.store.entities(user_id=USER, limit=10_000):
            purity = self.checker.purity(entity.id)
            if purity and not (entity.metadata or {}).get("owner"):
                by_thing[max(purity, key=purity.get)].append(entity)
        out = []
        for wid, entities in sorted(by_thing.items()):
            entities.sort(key=lambda e: (-self.store.backend.count_entity_memories(e.id),
                                         e.name))
            for other in entities[1:]:
                if other.normalized != entities[0].normalized:
                    out.append((entities[0], other))
        return out

    def merge_ids(self, keep_id: str, other_id: str, at: datetime, label: str) -> None:
        self.begin(f"{label}: merge {other_id[-4:]} into {keep_id[-4:]}", at)
        self.store.merge_entities(keep_id, other_id)
        self.checker.check()

    def undo(self, entity_id: str | None, at: datetime) -> None:
        if entity_id is None:
            return
        self.begin(f"undo merge of {entity_id[-4:]}", at)
        record = self.store.backend.merge_record(entity_id)
        outcome = self.store.undo_merge(entity_id)
        assert outcome["undone"], outcome
        self.checker.undone.append((entity_id, record["merged_at"], at.isoformat()))
        self.checker.check()

    def rename(self, entity_id: str, wid: str, name: str, at: datetime) -> None:
        self.begin(f"rename {wid} to {name}", at)
        self.world.rename(wid, name)
        self.store.rename_entity(entity_id, name)
        self.checker.check()

    def upkeep(self, at: datetime) -> None:
        self.begin("upkeep cycle", at)
        store, checker, calls = self.store, self.checker, self.calls
        cycle_op = calls.op
        run_pass = store.run_upkeep_pass

        def checked(key: str, **kwargs: Any) -> dict[str, Any]:
            calls.op = f"{cycle_op}: {key}"
            try:
                return run_pass(key, **kwargs)
            finally:
                # the cycle ends by computing the property vectors its passes
                # left missing, so they are checked after it
                checker.check(vectors=False)

        store.run_upkeep_pass = checked  # type: ignore[method-assign]
        try:
            store.run_upkeep_cycle(user_id=USER, now=at)
        finally:
            del store.run_upkeep_pass
        calls.op = f"{cycle_op}: end"
        checker.check()
        calls.op = f"{cycle_op}: search"
        self.searches.append(checker.search(self.world.week, self.search_judge))


def generate_and_replay(replay: Replay) -> None:
    """The history: sessions of saves, the person's own actions, and upkeep
    at the end of each week."""
    world = replay.world
    rng = world.rng
    weights = {wid: w for wid, (_, _, w) in ENTITIES.items()}
    for week in range(WEEKS):
        world.week = week
        monday = START + timedelta(weeks=week)
        sessions = []
        for day in range(6):
            for _ in range(rng.choice([0, 1, 1, 2, 2])):
                sessions.append(monday + timedelta(days=day, hours=rng.randint(0, 12)))
        for s, begun in enumerate(sorted(sessions)):
            group = rng.choices(sorted(GROUPS), [
                sum(weights[w] for w in GROUPS[g]) for g in sorted(GROUPS)])[0]
            context = f"{group} chat {week}.{s}"
            slots = [slot for slot in SLOTS if slot.subject in GROUPS[group]
                     and slot.start_week <= week]
            families = [f for f in EPISODES if f.subject in GROUPS[group]
                        and f.start_week <= week]
            at = begun
            # a third of the conversations save through the MCP server's path
            deferred = rng.random() < 0.35
            for _ in range(rng.choice([1, 2, 2, 3, 3, 4])):
                plans: list[Plan] = []
                used: set[str] = set()
                for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
                    if families and rng.random() < 0.5:
                        family = rng.choices(families, [weights[f.subject] * f.weight
                                                        for f in families])[0]
                        plans.append(world.episode(family, rng, at))
                        continue
                    choice = _pick(world, slots, weights, used, rng)
                    if choice is not None:
                        used.add(choice[0].id)
                        plans.append(world.plan(choice[0], choice[1], rng))
                if plans and deferred:
                    replay.save_deferred(plans, context, at)
                elif plans:
                    replay.save(plans, context, at)
                # a client saving through the MCP server sends one thought in
                # calls under two minutes apart, which are distilled together
                at += (timedelta(seconds=rng.randint(20, 100)) if deferred
                       else timedelta(minutes=rng.randint(2, 25)))
            if deferred:
                replay.distill(at + timedelta(minutes=10))
        # the person's own actions, on Friday evening, in order
        friday = monday + timedelta(days=4, hours=18)
        steps = person_actions(replay, week, rng)
        for hours, step in sorted(steps, key=lambda item: item[0]):
            step(friday + timedelta(hours=hours))
        replay.upkeep(monday + timedelta(days=6, hours=22))


def person_actions(replay: Replay, week: int, rng: random.Random) -> list[tuple[float, Any]]:
    """What the person does through the store's API this week: (hours after
    Friday 18:00, the action given its time)."""
    world, store = replay.world, replay.store
    steps: list[tuple[float, Any]] = []
    live = [s for s in SLOTS if world.state[s.id].live]
    if live:
        slot = rng.choice(live).id
        mode = "add" if week in (2, 5, 8) else "value"
        steps.append((0.0, lambda at: replay.edit(slot, at, mode=mode)))
    unasked = [s for s in SLOTS if world.state[s.id].live and not s.question
               and s.subject != "owner" and s.template.startswith("{E}")]
    if unasked and week in (3, 7):
        dropped = rng.choice(unasked).id
        steps.append((0.05, lambda at: replay.edit(dropped, at, mode="drop")))
    deletable = [s for s in SLOTS if world.state[s.id].live and not s.question]
    if deletable and week % 2 == 1 and week != 5:
        doomed = rng.choice(deletable).id
        steps.append((0.1, lambda at: replay.delete(doomed, at)))
    if week == 5 and deletable:
        def hard_delete(at: datetime) -> None:
            # for good: the newest version of a fact, which replaced an older one
            replaced = {row["superseded_by"] for row in store.backend._db.execute(
                "SELECT superseded_by FROM memories WHERE superseded_by IS NOT NULL")}
            newest = [s for s in deletable
                      if (m := replay.memory_of(s.id)) and m.id in replaced]
            replay.delete((newest or deletable)[0].id, at, hard=True)
        steps.append((0.1, hard_delete))

    def act(label: str, action) -> Any:
        return lambda at: replay.act(label, at, action)

    if week == 3:
        steps.append((1, lambda at: replay.retag("rate", ["Kitchen Renovation", "electrics"], at)))

        def remove(at: datetime) -> None:
            # a place and a person removed by mistake, named again before
            # they are brought back a week later
            for name, family_id in (("Brightline Gym", "climb"), ("Vessa Holm", "rent_paid")):
                found = replay.entity_named(name)
                if not found:
                    continue
                replay.removed.append(found[0].id)
                replay.act(f"remove {name}", at, lambda: store.remove_entities([found[0].id]))
            for i, (name, family_id) in enumerate((("Brightline Gym", "climb"),
                                                    ("Vessa Holm", "rent_paid"))):
                family = next(f for f in EPISODES if f.id == family_id)
                saturday = at + timedelta(days=1, hours=i)
                replay.save([world.episode(family, rng, saturday)], f"{name} chat {week}",
                            saturday)
        steps.append((2, remove))
    if week == 4:
        def tomi(at: datetime) -> None:
            if replay.merge("Tomas Vell", "Tomi", at, "a person's merge"):
                replay.cleaned = replay.entity_named("Tomas Vell")[0].id
        steps.append((1, tomi))
        if replay.removed:
            steps.append((2, act("restore what was removed",
                                 lambda: store.restore_entities(replay.removed))))
        steps.append((3, act("merge the tag taxes 2026 into 2026 taxes",
                             lambda: store.merge_tags(["taxes 2026"], "2026 taxes",
                                                      user_id=USER))))
    if week == 5:
        def rename_ember(at: datetime) -> None:
            ember = replay.entity_named("Ember Grid")
            if ember:
                replay.rename(ember[0].id, "ember", "Ember Grid Planner", at)
        steps.append((2, rename_ember))

        def mistake(at: datetime) -> None:
            replay.mistake = replay.merge("Plotwise v3", "Plotwise v4", at, "a mistaken merge")
        steps.append((3, mistake))

        def keep_apart(at: datetime) -> None:
            def major(entity_id: str) -> str:
                purity = replay.checker.purity(entity_id)
                return max(purity, key=purity.get) if purity else "?"

            apart = [p for p in store.merge_proposals(user_id=USER)
                     if major(p.entity_a) != major(p.entity_b)]
            if apart:  # a pair of two things, which the person knows
                replay.checker.rejected_by_person.add(apart[0].id)
                replay.act("a person keeps a pair apart", at,
                           lambda: store.reject_merge(apart[0].id))
        steps.append((5, keep_apart))

        def spelling(at: datetime) -> None:
            # a person folds a spelling the store kept apart into the name
            spellings = replay.spellings()
            if spellings:
                keep, other = spellings[0]
                replay.cleaned = keep.id
                replay.merge_ids(keep.id, other.id, at,
                                 f"a person's merge of {other.name} into {keep.name}")
        steps.append((6, spelling))
    if week == 6:
        steps.append((3, lambda at: replay.undo(replay.mistake, at)))

        def namesakes(at: datetime) -> None:
            electrician = replay.entity_named("Johnny", "johnny_e")
            climber = replay.entity_named("Johnny", "johnny_c")
            if electrician and climber and electrician[0].id != climber[0].id:
                replay.merge_ids(electrician[0].id, climber[0].id, at,
                                 "a mistaken merge of namesakes")
                replay.namesake = climber[0].id
        steps.append((4, namesakes))

        def rename_tag(at: datetime) -> None:
            cafes = store.backend.topic_entity("cafes", Scope(user_id=USER), create=False)
            if cafes is not None:
                replay.act("rename the tag cafes", at,
                           lambda: store.rename_entity(cafes.id, "coffee places"))
        steps.append((5, rename_tag))
    if week == 7:
        steps.append((3, lambda at: replay.undo(replay.namesake, at)))

        def rename_climber(at: datetime) -> None:
            climber = replay.entity_named("Johnny", "johnny_c")
            electrician = replay.entity_named("Johnny", "johnny_e")
            if climber and electrician and climber[0].id != electrician[0].id:
                replay.rename(climber[0].id, "johnny_c", "Johnny Brask", at)
        steps.append((4, rename_climber))

        def remove_cleaned(at: datetime) -> None:
            # the entity a name was merged into, removed by mistake, and
            # brought back the week after
            cleaned = store.backend.resolve_entity_id(replay.cleaned) if replay.cleaned else None
            if cleaned:
                replay.removed_later = [cleaned]
                replay.act("remove the entity a name was merged into", at,
                           lambda: store.remove_entities([cleaned]))
        steps.append((6, remove_cleaned))
    if week == 8:
        steps.append((5, act("delete the tag reading",
                             lambda: store.delete_tag("reading", user_id=USER))))
        if replay.removed_later:
            steps.append((6, act("restore it",
                                 lambda: store.restore_entities(replay.removed_later))))
    return steps


def _pick(world: World, slots: list[Slot], weights: dict[str, int], used: set[str],
          rng: random.Random) -> tuple[Slot, str] | None:
    options = [s for s in slots if s.id not in used]
    if not options:
        return None
    fresh = [s for s in options if not world.state[s.id].stated]
    if fresh and rng.random() < 0.45:
        slot = rng.choices(fresh, [weights[s.subject] * s.weight for s in fresh])[0]
        return slot, "new"
    said = [s for s in options if world.state[s.id].stated]
    if not said:
        slot = rng.choices(options, [weights[s.subject] * s.weight for s in options])[0]
        return slot, "new"
    slot = rng.choices(said, [weights[s.subject] * s.weight for s in said])[0]
    state = world.state[slot.id]
    ops = ["restate", "reword", "update", "contradict"]
    odds = [0.18, 0.12, 0.35 if state.refined < 2 else 0.0,
            0.35 if len(slot.values) > 1 else 0.0]
    return slot, rng.choices(ops, odds)[0]


# ------------------------------------------------------------ the test
def _instrument(monkeypatch, calls: Calls, store: MemoryStore) -> None:
    """Deterministic ids and clock; and the save's resolution and each
    comparison recorded, so an identity question can be traced to the name
    and the memory it was about."""
    monkeypatch.setattr(uuid, "uuid4", _Ids())
    for module in (models_mod, retrieval_mod, identity_mod, decay_mod, store_mod):
        monkeypatch.setattr(module, "datetime", _FakeDateTime)

    resolve = store_mod.resolve_mentions

    def resolving(**kwargs: Any):
        calls.resolving = {"memory": kwargs["memory_id"], "attach": kwargs.get("attach", True),
                           "names": list(kwargs["surfaces"])}
        try:
            return resolve(**kwargs)
        finally:
            calls.resolving = None

    monkeypatch.setattr(store_mod, "resolve_mentions", resolving)

    def linked_names(memory_id: str) -> set[str]:
        return {alias.casefold() for e in store.backend.entities_of_memory(memory_id)
                for alias in store.backend.entity_aliases(e.id)}

    compare = entities_mod.compare

    def comparing(decider, backend, a, b, *args, **kwargs):
        before = calls.comparing
        if isinstance(b, Mention):
            calls.comparing = {"name": b.name, "with": a.name, "memory": b.memory.id,
                               "already_linked": b.name.casefold() in linked_names(b.memory.id)}
        else:
            calls.comparing = {"name": b.name, "with": a.name}
        try:
            return compare(decider, backend, a, b, *args, **kwargs)
        finally:
            calls.comparing = before

    monkeypatch.setattr(entities_mod, "compare", comparing)

    judge = entities_mod._judge

    def judging(llm, existing, facts, new_fact, surface, decider=None):
        before = calls.comparing
        memory = (calls.resolving or {}).get("memory")
        calls.comparing = {"name": surface, "with": existing.name, "memory": memory,
                           "already_linked": bool(memory)
                           and surface.casefold() in linked_names(memory)}
        try:
            return judge(llm, existing, facts, new_fact, surface, decider)
        finally:
            calls.comparing = before

    monkeypatch.setattr(entities_mod, "_judge", judging)


def _report(setup: str, replay: Replay) -> str:
    violations = replay.checker.violations
    groups: dict[tuple[str, str], list[Violation]] = defaultdict(list)
    for v in violations:
        groups[(v.invariant, v.kind)].append(v)
    lines = [f"[{setup}] {replay.saves} saves, {replay.ops} operations; "
             f"{len(violations)} violations in {len(groups)} groups",
             "calls: " + ", ".join(f"{k} {n}" for k, n in sorted(replay.calls.total.items())),
             "search (found/asked per week): " + " ".join(
                 f"{s['found']}/{s['asked']}" for s in replay.searches),
             "documented exceptions taken: " + ", ".join(
                 f"{k} ({n})" for k, n in sorted(replay.checker.notes.items())),
             replay.checker.type_drift()]
    for (invariant, kind), found in sorted(groups.items()):
        lasting = [v for v in found if invariant[:2] in Checker.STATE and v.resolved is None]
        lines.append(f"\n== {invariant} / {kind}: {len(found)}"
                     + (f", {len(lasting)} still there at the end"
                        if invariant[:2] in Checker.STATE else ""))
        for v in (lasting or found)[:3]:
            gone = f" (gone at {v.resolved})" if v.resolved else ""
            lines.append(f"  at {v.op}{gone}\n    "
                         f"{json.dumps(v.detail, ensure_ascii=False)[:700]}")
    return "\n".join(lines)


#: The decision setups: a calibrated judge; no decision provider, the text
#: model only; and the text model answering the decision questions too, as an
#: operator sets ``MEMRY_DECISION_PROVIDER=llm``.
SETUPS = ("judge", "text_only", "text_decider")


def build(setup: str, monkeypatch) -> Replay:
    world, calls = World(), Calls()
    judged = setup == "judge"
    config = Config(db_path=":memory:")
    config.tags.enabled = True                       # synthetic parent tags, weekly
    config.decay.durability = setup != "text_only"   # needs a decision provider
    if setup == "text_decider":
        config.decision.provider = "llm"
    llm = WorldLLM(world, calls)
    llm.judged = judged
    store = MemoryStore(config, llm=llm, decider=WorldJudge(world, calls) if judged else None,
                        embedder=CountingEmbedder(calls))
    llm.store = store
    _instrument(monkeypatch, calls, store)
    CLOCK.now = START
    store.set_owner_name(USER, OWNER)
    checker = Checker(store, world, calls, judged)
    # a store whose provider does not judge relevance is searched with a stub
    # judge swapped in for the question
    search_judge = None if store.relevance_mode() == "jev" else SearchJudge(world, calls)
    return Replay(store, world, calls, checker, judged, search_judge)


@pytest.mark.parametrize("setup", SETUPS)
def test_a_long_history_through_the_save_path_keeps_every_invariant(setup, monkeypatch):
    replay = build(setup, monkeypatch)
    store, calls, checker = replay.store, replay.calls, replay.checker
    try:
        generate_and_replay(replay)
    finally:
        report = _report(setup, replay)
        path = os.environ.get("MEMRY_E2E_REPORT")
        if path:
            with open(f"{path}.{setup}.json", "w", encoding="utf-8") as out:
                json.dump({"summary": report, "log": replay.log,
                           "calls": dict(calls.total),
                           "violations": [v.__dict__ for v in checker.violations]},
                          out, ensure_ascii=False, indent=1, default=str)
        store.close()
    assert 150 <= replay.saves <= 300, replay.saves
    assert not checker.violations, report
