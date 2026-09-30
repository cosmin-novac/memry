"""The stages of a search, each tested at a scale where it decides the outcome.

Every search runs one pipeline (``MemoryStore.search``), its stages in a fixed
order, each rule in one stage: the seeds (``_seeds``: the hubs the question
names, the longest names among them, else the owner for a question in the
first person); the candidates, every filter applied as they are gathered (the
text ranking, ``_text_ranking``, keyword and vector fused by
``retrieval.hybrid_search``, and with seeds the linked pool: for every entity
the links reach, the ``FAMILY_TOP`` of its memories that best state the
property asked); the order (with seeds the linked order, property similarity
times aboutness, ``_search_linked``; else the text ranking's); the judged pool
(the first ``decision.rerank_pool``, the keyword search's best match kept
among them, ``_with_the_keyword_place``); the judge (``_judge_ranking``, in
``_judged_relevance``'s wording, "it" for the names of a single seed); the set
question's second call (``_set_pool``); the final order and the limit
(``_final_order``).

A rule that must hold at a stage is only tested there when the store is
larger than what the later stages read: with 20 memories or fewer, the judge
reads everything, and a stage before it cannot lose an answer. Every store
here holds more than the judge reads, has distractors that outrank the answer
where the rule is missing, a judge that scores only what it is given (an
answer it never reads is lost), and vectors under which the property
comparison matters.

Tests named ``test_gap_*`` failed at ce2b49f, where a search took one of two
routes (a question naming a hub through the linked search, any other, and a
filtered search, through the text ranking and ``_rerank``) and the rule they
state was missing on one of them or at the wrong stage. Each has a companion
that tests the rule at its stage, running the stages as ``search`` runs them
(``_through_the_judged_pool``) or reading what the judge read; the filtered set
question's companion runs the same question unfiltered on the same store. The
other tests passed at ce2b49f too (but the weak-link tests, which failed at
a5eecd8), and each has a companion (``*_fails_without_the_rule``) that takes
the rule away at its stage and shows the answer lost, so they are tests that
can fail. The judges are stubs, the
vectors ``_SenseEmbedder``: what a real embedder or Jev does with these texts
is the benchmark's to measure.
"""

from __future__ import annotations

import re
import zlib
from datetime import datetime, timedelta, timezone

import pytest

import memry.intelligence.graph_retrieval as graph_retrieval
import memry.store as store_module
from memry.backends.local import LocalBackend
from memry.config import Config
from memry.intelligence.graph_retrieval import detect_query_entities
from memry.models import Entity, EntityMention, Episode, Memory, Relation, Scope
from memry.providers.decisions import Answer, Answers
from memry.providers.embeddings import Embedder
from memry.providers.llm import NoneLLM
from memry.store import MemoryStore, _Reads

USER = "ada"
NOW = datetime.now(timezone.utc)


def _stamp(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


# ------------------------------------------------------------------ vectors
class _SenseEmbedder(Embedder):
    """What a sentence embedder does that these tests rest on, in the open.

    - A word counts toward its sense, so a paraphrase is near ("platforms"
      and "Windows", "think" and "loved").
    - "it" and "its" count as the word they are: a text emptied of its names
      ("it met it at the gym") is near another emptied text ("What did it
      think of it?"). Measured with OpenAI vectors, masked "it works on it."
      and "it liked the food at it." outranked the answer to "What does Raj
      Osei like?" (registry R-113), and "Did it like it?" put every liked
      restaurant above the one asked about (R-35).
    - A name's words weigh ``name_weight`` times a word: an embedder follows
      names.
    - A number is a number: the vectors cannot tell 2024-117 from 2024-118
      (registry R-21), only the keyword search can.
    - Every other word lands in one of ``FILLER`` dimensions, so a longer text
      is less near a short question, as with real vectors.
    """

    name, _model = "sense", "v1"
    FILLER = 97
    STOP = frozenset(
        "a an the at of to in on for and or by with from as is was were be been "
        "did does do what where which when how would not too much many we our "
        "that this there her his their".split())
    SENSES = {
        "opinion": "think thought found loved liked like likes enjoyed hated disliked",
        "contact": "met meet called texted visited saw",
        "pay": "pay paid paying payment",
        "invoice": "invoice invoices",
        "live": "live lives lived living",
        "park": "park parked parking",
        "car": "car",
        "sync": "sync syncs synced",
        "work": "work works working",
        "offline": "offline network online connection",
        "platform": "platform platforms windows linux macos systems",
        "run": "run runs",
        "spend": "spend spent",
        "grocery": "groceries grocery",
        "tool": "tool tools",
        "use": "use uses used",
        "number": "",
    }

    def __init__(self, names=(), name_weight: float = 1.0) -> None:
        self.sense_of = {w: s for s, words in self.SENSES.items() for w in words.split()}
        senses = sorted(self.SENSES)
        self.name_words = sorted({w for n in names for w in self.words(n)})
        self.name_weight = name_weight
        self.index = {s: i for i, s in enumerate(senses)}
        self.index["it"] = len(self.index)
        for word in self.name_words:
            self.index["name:" + word] = len(self.index)
        self.filler_start = len(self.index)
        self.dimensions = self.filler_start + self.FILLER + 1

    @staticmethod
    def words(text: str) -> list[str]:
        return [re.sub("['’]s$", "", w) for w in re.findall(r"[a-z0-9]+(?:['’][a-z]+)?",
                                                             text.lower())]

    def embed(self, texts):
        out = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for word in self.words(text):
                if word in self.STOP:
                    continue
                if word in ("it", "its"):
                    vector[self.index["it"]] += 1.0
                elif word.isdigit():
                    vector[self.index["number"]] += 1.0
                elif word in self.name_words:
                    vector[self.index["name:" + word]] += self.name_weight
                elif word in self.sense_of:
                    vector[self.index[self.sense_of[word]]] += 1.0
                else:
                    vector[self.filler_start + zlib.crc32(word.encode()) % self.FILLER] += 1.0
            vector[-1] = 0.01
            out.append(vector)
        return out


# ------------------------------------------------------------------- judges
class _Judge:
    """A decision provider that scores only the memories it is sent, by the
    first of ``scores`` (word -> probability) that the memory's text as sent
    contains, else ``rest``; the meta questions as told. It keeps each
    question and each memory as it read them. An answer never sent is never
    scored, so an answer a stage before the judge loses stays lost."""

    name, available = "judge", True
    may_rerank = reranks_by_default = False

    def __init__(self, *, scores, rest=0.03, specific=1.0, several=0.0):
        self.scores, self.rest = scores, rest
        self.specific, self.several = specific, several
        self.states: list[str] = []
        self.batches: list[list[str]] = []

    def close(self):
        pass

    def decide(self, state, questions):
        self.states.append(state)
        batch = [q.instructions.split("Memory: ", 1)[1] for key, q in questions.items()
                 if key.startswith("m")]
        self.batches.append(batch)

        def value(key, text):
            if key == "property":
                return self.specific
            if key == "several":
                return self.several
            return next((v for word, v in self.scores.items() if word in text), self.rest)
        return Answers({key: Answer(value(key, q.instructions), {}, 0.9, True)
                        for key, q in questions.items()})

    @property
    def read(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


class _Reranker(_Judge):
    """``_Judge`` for a provider that re-ranks by default, as Jev does."""

    may_rerank = reranks_by_default = True


# -------------------------------------------------------------------- store
def _store(embedder: Embedder) -> MemoryStore:
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=embedder)
    store.config.retrieval.relational_relevance = "jev"
    return store


@pytest.fixture
def stores():
    made: list[MemoryStore] = []

    def make(embedder: Embedder) -> MemoryStore:
        made.append(_store(embedder))
        return made[-1]

    yield make
    for store in made:
        store.close()


def _entity(store, name, entity_type=None):
    return store.backend.insert_entity(Entity(
        name=name, normalized=name.lower(), entity_type=entity_type, user_id=USER))


def _remember(store, text, entities=(), *, days_ago=0.0, tags=(), memory_id=None):
    stamp = _stamp(days_ago)
    memory = store.backend.insert_memory(
        Memory(**({"id": memory_id} if memory_id else {}), content=text, user_id=USER,
               embedding_model=store.embedder.model_id, categories=list(tags),
               created_at=stamp, updated_at=stamp),
        embedding=store.embedder.embed([text])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def _ids(results) -> list[str]:
    return [r.memory.id for r in results]


def _text(store, query, limit=5) -> list[str]:
    """Stage 2's text ranking, as deep as a search at ``limit`` reads it."""
    return _ids(store._text_ranking(query, _Reads(Scope(user_id=USER)), limit))


def _through_the_judged_pool(store, query, *, limit=5, **filters):
    """Stages 1 to 4 as ``MemoryStore.search`` runs them: the plan (the seeds,
    the question as read, whether it is judged), the text ranking, and the
    order with its judged pool."""
    reads = _Reads(Scope(user_id=USER), **filters)
    plan = store._plan(query, reads, True)
    results = store._text_ranking(query, reads, limit)
    return plan, results, store._search_linked(query, reads.scope, results, False, plan=plan)


# ================================================================== R-35
# "Did Ilva Marsh like Olive Kitchen?" must not be read "Did it like it?".
# With several hubs named, the question the order and the judge read keeps
# their names (``MemoryStore._plan``), and the linked order compares it with
# each memory's ordinary vector, names kept (``_linked_order``). At ce2b49f
# the judge kept the names but the linked order still compared every memory
# with the question masked over all seeds, and that order decides which 20
# memories the judge reads, which of each entity's memories join the pool, and
# how a set question's second call is cut.
FRIENDS = ["Kai Lund", "Pia Holt", "Tomas Brenn", "Edda Voss", "Rafe Oduya", "Lio Marten",
           "Sanne Kiro", "Ottar Bell", "Yara Quill", "Nils Farrow"]
PLACES = ["Saffron Kitchen", "Blue Fig", "Casa Verde", "Harbor Noodle", "Mint Garden",
          "Copper Pot", "Rye House", "Lemon Tree", "Salt Barn", "Fern Cafe", "Oak Table",
          "Plum Diner"]
DISHES = ["curry", "ramen", "paella", "dumplings", "tacos", "pho", "gnocchi", "falafel",
          "risotto", "bibimbap", "moussaka", "pierogi"]
OPINION_QUESTION = "What did Ilva Marsh think of Olive Kitchen?"
OPINION = "Ilva Marsh found the tasting menu at Olive Kitchen too salty and would not go back"


def _two_things_named(stores):
    """Ilva Marsh's store: 40 memories of meeting, calling and visiting her
    friends, 12 of what she thought of other restaurants, 6 about Olive
    Kitchen, 3 about her and Olive Kitchen (one of them the answer) and 30
    notes naming nobody: 91 memories. Every memory names the entities it is
    about, and its property vector reads those names "it", as the save path
    stores it (``refresh_property_vectors``)."""
    everyone = ["Ilva Marsh", "Olive Kitchen", *FRIENDS, *PLACES]
    store = stores(_SenseEmbedder(names=everyone))
    ilva = _entity(store, "Ilva Marsh", "person")
    olive = _entity(store, "Olive Kitchen", "organization")
    friends = [_entity(store, name, "person") for name in FRIENDS]
    places = [_entity(store, name, "organization") for name in PLACES]
    for i, friend in enumerate(friends):
        _remember(store, f"Ilva Marsh met {friend.name} at the gym", [ilva, friend])
        _remember(store, f"Ilva Marsh called {friend.name} on Sunday", [ilva, friend])
        _remember(store, f"{friend.name} visited Ilva Marsh", [friend, ilva])
        _remember(store, f"Ilva Marsh texted {friend.name}", [ilva, friend])
    for place, dish in zip(places, DISHES):
        _remember(store, f"Ilva Marsh loved the {dish} at {place.name}", [ilva, place])
    _remember(store, "Olive Kitchen is run by Pia Holt", [olive, friends[1]])
    for text in ("Olive Kitchen opened in 2019", "Olive Kitchen is closed on Mondays",
                 "Olive Kitchen moved to the old harbour", "Olive Kitchen takes no cards",
                 "Olive Kitchen has a garden terrace"):
        _remember(store, text, [olive])
    answer = _remember(store, OPINION, [ilva, olive])
    _remember(store, "Ilva Marsh booked Olive Kitchen for Friday", [ilva, olive])
    _remember(store, "Ilva Marsh had lunch at Olive Kitchen with Kai Lund",
              [ilva, olive, friends[0]])
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    assert store.refresh_property_vectors(user_id=USER) > 60
    assert store._is_hub(ilva.id) and store._is_hub(olive.id)
    return store, answer


def _several_things_are_answered(store, answer):
    # the answer is in the pool: the text ranking has it third, and it is
    # not the keyword search's best match, which keeps a place in the 20
    assert answer.id in _text(store, OPINION_QUESTION)[:3]
    best = store.backend.keyword_search(OPINION_QUESTION, Scope(user_id=USER), 1)
    assert best[0][0].id != answer.id
    judge = _Judge(scores={"tasting menu": 0.9, "Olive Kitchen": 0.2})
    store.decider = judge
    results = store.search(OPINION_QUESTION, user_id=USER, limit=5)
    # the judge stage keeps the names already (8a07222)
    assert judge.states == [f"QUESTION: {OPINION_QUESTION}"]
    assert "about" in results[0].signals  # the linked search ran, from both
    assert answer.content in judge.read, "the answer never reached the judge"
    assert results[0].memory.id == answer.id


def test_gap_a_question_naming_several_things_keeps_their_names_in_the_linked_order(stores):
    """R-35 / S16-S18 at the linked order. "What did Ilva Marsh think of
    Olive Kitchen?" names two hubs. Masked over both it reads "What did it
    think of it?": nearer that are her 12 opinions of other restaurants
    ("it loved the curry at it") and her 40 meetings ("it met it at the
    gym") than the answer ("it found the tasting menu at it too salty ...").
    Third in the text ranking, the answer was 57th of 91 in the linked order
    at ce2b49f: the judge never read it, and it was lost. 8a07222's test used
    3 memories, where the judge reads everything."""
    store, answer = _two_things_named(stores)
    _several_things_are_answered(store, answer)


def test_a_question_naming_several_things_is_ordered_as_written(stores):
    """Stages 1 and 3: the question naming two hubs is read as written, and
    in the linked order the answer is among the first 20, which the judge
    reads."""
    store, answer = _two_things_named(stores)
    plan, _, ranked = _through_the_judged_pool(store, OPINION_QUESTION)
    assert len(plan.seeds) == 2 and plan.question == OPINION_QUESTION
    assert answer.id in _ids(ranked)[:store.config.decision.rerank_pool]


# ================================================================== R-21
# An identifier only the keyword search matches ("invoice 2024-117") keeps a
# place among the 20 the judge reads, on every search, judged or not
# (``MemoryStore._with_the_keyword_place``). a06db33 built it into the linked
# search only, so at ce2b49f it held for a question naming a hub and not for
# the registry's scenario, which names none ("invoice 2024-117 untagged"):
# judged on the text ranking as it stood, by ``_judge_ranking`` or ``_rerank``.
INVOICE_QUESTION = "Did we pay invoice 2024-117?"


def _invoices(stores):
    """45 invoices paid by card this month and 20 bills paid, all near a
    question about paying an invoice; the answer, settled by transfer two
    months ago, shares only the identifier with it, which the vectors cannot
    see: 66 memories, no entity."""
    store = stores(_SenseEmbedder())
    for i in range(45):
        _remember(store, f"Invoice 2024-{200 + i} was paid by card", days_ago=i % 10)
    for i, bill in enumerate(["electricity", "water", "phone", "internet", "rent"] * 4):
        _remember(store, f"We paid the {bill} bill number {i}", days_ago=i % 10)
    answer = _remember(store, "Invoice 2024-117 was settled by bank transfer on 3 March",
                       days_ago=60)
    scope = Scope(user_id=USER)
    assert store.backend.keyword_search(INVOICE_QUESTION, scope, 1)[0][0].id == answer.id
    return store, answer


def _the_identifier_is_answered(store, answer, route):
    text = _text(store, INVOICE_QUESTION)
    assert answer.id in text and text.index(answer.id) >= 20  # the words alone see it
    if route == "judge":  # relational_relevance "jev"
        judge = _Judge(scores={"2024-117": 0.9}, rest=0.05)
    else:  # "auto" with a provider that re-ranks by default, as Jev does: "jev"
        store.config.retrieval.relational_relevance = "auto"
        judge = _Reranker(scores={"2024-117": 0.9}, rest=0.05)
    store.decider = judge
    results = store.search(INVOICE_QUESTION, user_id=USER, limit=5)
    assert not any("about" in r.signals for r in results)  # no seeds: the text order
    assert len(judge.batches) == 1 and len(judge.batches[0]) == 20
    assert answer.content in judge.read, "the identifier never reached the judge"
    assert results[0].memory.id == answer.id


@pytest.mark.parametrize("route", ["judge", "rerank"])
def test_gap_an_identifier_only_the_words_match_reaches_the_judge_on_the_text_route(
        stores, route):
    """R-21 for a question naming no hub. 65 memories about paying invoices
    and bills are nearer "Did we pay invoice 2024-117?" by vector, and the
    invoices match two of its words; the answer matches three (the
    identifier), by keyword alone, and is two months older. The fused ranking
    puts it below 20, where at ce2b49f neither the judging call ("judge") nor
    the re-rank ("rerank") read it. Both are one judge now, and "rerank" was
    "vector" relevance with a provider that re-ranks, which judges nothing
    under the one judging rule: it sets "auto", which that provider resolves
    to "jev", the setting a store with Jev has."""
    store, answer = _invoices(stores)
    _the_identifier_is_answered(store, answer, route)


@pytest.mark.parametrize("judged", [True, False])
def test_the_best_keyword_match_keeps_a_place_in_the_judged_pool(stores, judged):
    """Stage 4, on every search, judged or not: below the judged pool in the
    text ranking, the identifier's memory keeps the last of its places."""
    store, answer = _invoices(stores)
    if judged:
        store.decider = _Judge(scores={"2024-117": 0.9}, rest=0.05)
    plan, results, ranked = _through_the_judged_pool(store, INVOICE_QUESTION)
    assert plan.judges is judged and not plan.seeds
    assert _ids(results).index(answer.id) >= 20
    assert _ids(ranked).index(answer.id) == store.config.decision.rerank_pool - 1


# ================================================================== R-12
# Only a hub starts the linked search: a stray phrase stored as an entity does
# not decide what a search is about (``MemoryStore._is_hub``). The hubs are
# kept first and the longest-name rule runs among them (``_seeds``,
# ``graph_retrieval.longest_names``). At ce2b49f the longest-name rule ran
# first (``detect_query_entities(longest=True)``): a stray entity whose name
# holds a hub's ("bildy sync", one memory, no type) hid the hub, and a question
# about bildy was searched as naming nothing. The test of 063cb97 used
# "budget", which holds no hub's name, in a store of 3 memories.
SYNC_QUESTION = "Does bildy sync work offline?"
SYNC_ANSWER = "bildy keeps working without a network since v3"
APPS = ["Kaven planner", "Luma notes", "Orbit mail", "Tessel tasks", "Quill docs",
        "Brio calendar", "Nimbo drive", "Sable chat", "Vanta boards", "Pico wiki",
        "Rook sheets", "Dune photos", "Fable reader", "Mosaic crm", "Umber todo"]


def _a_stray_name_holds_a_hub(stores):
    """bildy, a product with 20 memories and the answer, and a stray entity
    "bildy sync" that extraction took from one of them; 60 notes on whether
    other apps sync offline, which match the question's words better and
    name nothing the store knows: 81 memories."""
    store = stores(_SenseEmbedder(names=["bildy"]))
    bildy = _entity(store, "bildy", "product")
    stray = _entity(store, "bildy sync")
    answer = _remember(store, SYNC_ANSWER, [bildy], days_ago=30)
    _remember(store, "bildy sync keeps failing on Mondays", [bildy, stray])
    for day in ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        _remember(store, f"bildy crashed on {day} after the update", [bildy])
    for fact in ["is written in Go", "stores its data in SQLite", "is led by Mara Ruiz",
                 "is released under the MIT license", "has a dark theme",
                 "exports notes as Markdown", "ships every two weeks",
                 "has 4,000 users", "started in 2021", "has a public roadmap",
                 "charges 5 euros a month", "runs a Discord server", "has a web clipper"]:
        _remember(store, f"bildy {fact}", [bildy])
    for app in APPS:
        for tail in ["sync works offline", "sync does not work offline on the train",
                     "sync works offline since the update", "sync needs work offline"]:
            _remember(store, f"The {app} {tail}")
    assert store._is_hub(bildy.id) and not store._is_hub(stray.id)
    return store, bildy, answer


def _the_hub_is_searched(store, bildy, answer):
    scope = Scope(user_id=USER)
    assert bildy.id in detect_query_entities(store.backend, scope, SYNC_QUESTION)
    assert answer.id not in _text(store, SYNC_QUESTION)[:20]  # the words alone do not find it
    store.decider = judge = _Judge(scores={"without a network": 0.9}, rest=0.05)
    results = store.search(SYNC_QUESTION, user_id=USER, limit=5)
    assert "about" in results[0].signals, "the linked search did not start at bildy"
    assert answer.content.replace("bildy", "it") in judge.read
    assert results[0].memory.id == answer.id


def test_gap_a_stray_name_holding_a_hubs_name_does_not_hide_the_hub(stores):
    """R-12 at the seeds. "Does bildy sync work offline?" names bildy, a hub,
    and "bildy sync", a stray entity of one memory. At ce2b49f the longer
    stray name dropped bildy, and then the stray was dropped as no hub: the
    question named nothing, the text ranking was judged, and the answer,
    which shares only "bildy" with the question, was not among its first
    20."""
    store, bildy, answer = _a_stray_name_holds_a_hub(stores)
    _the_hub_is_searched(store, bildy, answer)


def test_the_seeds_are_the_longest_names_among_the_hubs(stores):
    """Stage 1: of the names the question holds, the hub bildy and the stray
    "bildy sync", the hubs are kept first and the longest name is chosen
    among them, so bildy seeds the search. The longest over every name found
    is the stray's, which is no hub."""
    store, bildy, _ = _a_stray_name_holds_a_hub(stores)
    scope = Scope(user_id=USER)
    assert store._seeds(SYNC_QUESTION, scope) == ([bildy.id], False)
    longest = detect_query_entities(store.backend, scope, SYNC_QUESTION, longest=True)
    assert longest and not any(store._is_hub(e) for e in longest)


# ================================================== date filters and the judge
# ``since`` and ``until`` are kept to as the candidates are gathered, the
# linked pool's included (``_Reads``, stage 2), before anything is ordered or
# judged. At ce2b49f they were applied after the linked search, which had
# already judged its first 20: the judge read memories the date window then
# dropped, and none of those the search returned. The re-rank of the other
# route ran after the filter. No registry row covers it; it is the route
# difference the audit found.
PARK_QUESTION = "Where did Harlow park the car?"
PARK_ANSWER = "Harlow left the car at the Nordpark lot on Tuesday"
STREETS = ["Linden", "Birch", "Mill", "Canal", "Market", "Church", "Station", "Bridge",
           "Garden", "Park", "Castle", "River", "Hill", "Lake", "Forest", "Meadow",
           "Harbour", "School", "Chapel", "Orchard"]


def _parked(stores):
    """Harlow parked the car on 40 streets over the last three months; this
    week he parked the bike and the van, and left the car at the Nordpark
    lot: 47 memories about Harlow, and 20 notes naming nobody."""
    store = stores(_SenseEmbedder(names=["Harlow"]))
    harlow = _entity(store, "Harlow", "person")
    for i in range(40):
        _remember(store, f"Harlow parked the car on {STREETS[i % 20]} street",
                  [harlow], days_ago=30 + i)
    for text in ("Harlow parked the bike at the station", "Harlow parked the van at work",
                 "Harlow parked the scooter by the gym", "Harlow parked the trailer at home",
                 "Harlow parked the bike at the market", "Harlow parked the van downtown"):
        _remember(store, text, [harlow], days_ago=2)
    answer = _remember(store, PARK_ANSWER, [harlow], days_ago=1)
    for i in range(20):
        _remember(store, f"Note {i}: water the plants on the balcony", days_ago=5)
    store.refresh_property_vectors(user_id=USER)
    return store, answer


def _this_weeks_answer(store, answer, since):
    store.decider = judge = _Judge(scores={"Nordpark": 0.9, "parked the car": 0.8}, rest=0.05)
    results = store.search(PARK_QUESTION, user_id=USER, limit=5, since=since)
    assert results and all(r.memory.created_at >= since for r in results)
    assert answer.content.replace("Harlow", "it") in judge.read, \
        "the judge read only memories the date window dropped"
    assert results[0].memory.id == answer.id


def test_gap_the_judge_reads_what_the_date_window_keeps(stores):
    """"Where did Harlow park the car?" since a week ago. At ce2b49f the 20
    the judge read were Harlow's 40 older parkings, which the window then
    dropped; of this week's, none was judged and the linked order stood, with
    six parkings of the bike, the van, the scooter and the trailer above the
    answer, which was seventh."""
    store, answer = _parked(stores)
    _this_weeks_answer(store, answer, _stamp(7))


def test_every_candidate_is_inside_the_date_window_before_it_is_ordered(stores):
    """Stage 2: the window is kept to as the candidates are gathered, the
    linked pool's included, so the judged pool holds this week's memories
    and the answer. Without the window Harlow's older parkings fill it."""
    store, answer = _parked(stores)
    since, size = _stamp(7), store.config.decision.rerank_pool
    plan, _, ranked = _through_the_judged_pool(store, PARK_QUESTION, since=since)
    assert plan.seeds
    assert all(r.memory.created_at >= since for r in ranked)
    assert answer.id in _ids(ranked)[:size]
    _, _, unfiltered = _through_the_judged_pool(store, PARK_QUESTION)
    assert answer.id not in _ids(unfiltered)[:size]
    assert all(r.memory.created_at < since for r in unfiltered[:size])


# ============================================= a search filtered by tag or entity
# A tag or entity filter keeps the candidates to it and nothing else: the
# search reads as deep, its judge reads ``decision.rerank_pool``, and a
# question needing several memories gets its second call, on memories the
# filter keeps too (``_set_pool``), and returns every member. At ce2b49f
# ``categories`` and ``entity_id`` turned the linked search off and fetched
# only ``limit`` candidates: the judge (``_rerank``) read ``limit`` memories,
# and there was no second call and no member past the limit (only
# ``_judge_ranking`` marked members). R-26 and R-39 say one call reads 20 "on
# every search, linked or not"; R-82 and R-97 that a set question returns
# every member. Both held only unfiltered.
GROCERY_QUESTION = "How much did I spend on groceries?"
SHOPS = ["Lidl", "Aldi", "Rewe", "Edeka", "Penny", "Netto", "Kaufland", "Spar"]


def _groceries(stores):
    """40 grocery purchases filed under "groceries", 40 other expenses under
    "expenses": 80 memories."""
    store = stores(_SenseEmbedder())
    bought = [_remember(store, f"Spent {12 + i} euros on groceries at {SHOPS[i % 8]}",
                        tags=["groceries"], days_ago=i) for i in range(40)]
    for i in range(40):
        _remember(store, f"Spent {30 + i} euros on fuel at station {i}", tags=["expenses"],
                  days_ago=i)
    return store, bought


def _every_purchase(store, bought, **filters):
    store.decider = judge = _Reranker(scores={"groceries": 0.12}, rest=0.02, several=0.9)
    results = store.search(GROCERY_QUESTION, user_id=USER, limit=5, **filters)
    assert len(judge.batches[0]) == 20, f"the judge read {len(judge.batches[0])}"
    assert {r.memory.id for r in results} == {m.id for m in bought}


def test_gap_a_set_question_filtered_by_its_tag_returns_every_member(stores):
    """"How much did I spend on groceries?" with the tag "groceries" given, as
    a caller does for a vague question (``test_date_search``): at ce2b49f the
    judge read five purchases and five were returned, of 40."""
    store, bought = _groceries(stores)
    _every_purchase(store, bought, categories=["groceries"])


def test_a_set_question_returns_every_member_unfiltered(stores):
    """The same question on the same store, unfiltered: the judging call reads
    20, the second call the other 60 (the rest of the topic, then the nearest
    to the members), and all 40 are returned, as they are filtered."""
    store, bought = _groceries(stores)
    _every_purchase(store, bought)


# ======================================================= a tie read by write order
# "it" in what the judge reads is the memory's entity the links reach most
# strongly, a tie by entity id (``subject`` in ``MemoryStore._judge_ranking``).
# The same choice decides the override (``subject(mid) in above``) and the
# best answer about each entity. At ce2b49f a tie went to the entity whose
# mention was written first (``max`` over ``entities_of_memories``, ``ORDER BY
# em.created_at, em.id``): two entities reached as strongly (two projects a
# person works on, each at ``LINKED_RELATION``) were told apart by write
# order. 3623c71 broke every ranked read's tie by id (R-121); its test writes
# mentions with fixed ids at one time, so this one never differs there.
def _two_projects(stores, *, helios_first):
    store = stores(_SenseEmbedder(names=["Ada", "Helios", "Orion"]))

    def entity(name, entity_type):  # the same id in both builds
        return store.backend.insert_entity(Entity(
            id=f"entity-{name.lower()}", name=name, normalized=name.lower(),
            entity_type=entity_type, user_id=USER))

    ada = entity("Ada", "person")
    projects = {name: entity(name, "project") for name in ("Helios", "Orion")}
    for name, project in projects.items():
        store.backend.add_relation(Relation(subject=ada.id, predicate="works_on",
                                            object=project.id, user_id=USER))
        _remember(store, f"{name} ships every Friday", [project])
    _remember(store, "Ada prefers dark mode", [ada])
    shared = _remember(store, "Helios and Orion share one Postgres cluster",
                       memory_id="memory-shared-cluster")
    order = ["Helios", "Orion"] if helios_first else ["Orion", "Helios"]
    for second, name in enumerate(order):  # the same mention ids, written in turn
        store.backend.add_mention(EntityMention(
            id=f"mention-{name.lower()}", entity_id=projects[name].id, memory_id=shared.id,
            surface=name, created_at=_stamp(0.001 - second / 86400)))
    return store


def test_gap_what_the_judge_reads_does_not_depend_on_which_mention_was_written_first(stores):
    """Two builds of one store, the same ids, the two mentions of "Helios and
    Orion share one Postgres cluster" written a second apart, in the other
    order. (Written in the same second, as one save writes them, the random
    id of each mention decided.) Ada's question reaches both projects at
    0.5; at ce2b49f the judge read "it and Orion share ..." in one build and
    "Helios and it share ..." in the other."""
    read = []
    for helios_first in (True, False):
        store = _two_projects(stores, helios_first=helios_first)
        store.decider = judge = _Judge(scores={"Postgres": 0.9})
        store.search("Which database does Ada use?", user_id=USER, limit=5)
        read.append(next(text for text in judge.read if "Postgres" in text))
    assert read[0] == read[1]


def test_the_entity_read_as_it_is_the_one_reached_most_strongly_a_tie_by_id(stores):
    """Stage 5: Helios and Orion are reached as strongly, and the one of the
    smaller id ("entity-helios") reads "it", in either build."""
    for helios_first in (True, False):
        store = _two_projects(stores, helios_first=helios_first)
        store.decider = judge = _Judge(scores={"Postgres": 0.9})
        store.search("Which database does Ada use?", user_id=USER, limit=5)
        assert "it and Orion share one Postgres cluster" in judge.read


# ============================================ rules that hold, tested where they decide
# R-78 / OF1. "Where do I live?" names nobody; the owner seeds the linked
# search (``MemoryStore._seeds``). Its test (TG
# ``test_a_question_in_the_first_person_is_about_the_owner``) reads the
# "about" signal in a store of 13 memories. Here the owner's answer is not
# in the first 20 of the text ranking, which 30 friends' homes fill.
def _an_owner_and_friends(stores):
    """Ilva Marsh, the owner, with her address and 40 other memories; 30
    friends she is linked to, each with where they live: 71 memories."""
    names = ["Ilva Marsh", *FRIENDS, *[f"{first} {last}" for first in ("Aino", "Bo", "Cato")
                                       for last in ("Dahl", "Eske", "Falk", "Grue", "Holm",
                                                    "Isak", "Juhl")]]
    store = stores(_SenseEmbedder(names=names))
    owner = _entity(store, "Ilva Marsh", "person")
    store._upkeep_set("owner_entity", USER, owner.id)
    answer = _remember(store, "Ilva Marsh lives in Lisbon, in a flat near the river", [owner])
    for place, dish in zip(PLACES, DISHES):
        _remember(store, f"Ilva Marsh loved the {dish} at {place}", [owner])
    for i in range(28):
        _remember(store, f"Ilva Marsh ran {3 + i} km along the coast", [owner])
    cities = ["Porto", "Brno", "Leipzig", "Gent", "Turku", "Split"]
    for i, name in enumerate(names[1:31]):
        friend = _entity(store, name, "person")
        store.backend.add_relation(Relation(subject=owner.id, predicate="friend_of",
                                            object=friend.id, user_id=USER))
        _remember(store, f"{name} lives in {cities[i % 6]}", [friend])
    store.refresh_property_vectors(user_id=USER)
    return store, answer


def _where_the_owner_lives(store, answer):
    # 30 friends' homes fill the text ranking's 20
    assert answer.id not in _text(store, "Where do I live?")[:20]
    store.decider = judge = _Judge(scores={"lives in": 0.9})
    return judge, store.search("Where do I live?", user_id=USER, limit=5)


def test_a_first_person_question_starts_at_the_owner_at_scale(stores):
    """The owner seeds the search, the question reads "Where do it live?",
    and the owner's answer (about 1.0) comes before the friends' homes, which
    the judge scores as high but which are about someone linked (0.5)."""
    store, answer = _an_owner_and_friends(stores)
    judge, results = _where_the_owner_lives(store, answer)
    assert judge.states == ["QUESTION: Where do it live?"]
    assert results[0].memory.id == answer.id and results[0].signals["about"] == 1.0


def test_a_first_person_question_starts_at_the_owner_at_scale_fails_without_the_rule(
        stores, monkeypatch):
    """Without the rule at the seeds, the question names nothing: the text
    ranking's 20 are judged, the answer is not among them, and it is lost."""
    store, answer = _an_owner_and_friends(stores)
    monkeypatch.setattr(store_module, "speaks_in_first_person", lambda query: False)
    judge, results = _where_the_owner_lives(store, answer)
    assert "about" not in results[0].signals
    assert answer.content not in judge.read
    assert answer.id not in _ids(results)


# R-1 / R-58 / R-73 / R-10. The multi-hop answer ("Helios uses Postgres in
# production" for "What tools does Ada use for her work?") reaches the
# linked search only as one of the ``FAMILY_TOP`` of Helios's memories that
# best state the property asked (``MemoryStore._linked_order``). Its tests (TG
# ``test_multi_hop_answer_is_recovered``, ``test_multi_hop_works_at_the_
# default_depth``) use 53 memories, all inside the text ranking's 80, so the
# family pick never decided there.
def _a_project_among_many(stores):
    """Ada works on Helios (a relation, 0.5); Helios has 60 memories and one
    names the tool; 40 notes about tools for work name nothing; Ada has 30
    of her own: 131 memories."""
    store = stores(_SenseEmbedder(names=["Ada", "Helios", "Postgres"]))
    ada = _entity(store, "Ada", "person")
    helios = _entity(store, "Helios", "project")
    postgres = _entity(store, "Postgres", "product")
    store.backend.add_relation(Relation(subject=ada.id, predicate="works_on",
                                        object=helios.id, user_id=USER))
    answer = _remember(store, "Helios uses Postgres in production", [helios, postgres],
                       days_ago=90)
    for i in range(59):
        _remember(store, f"Helios had its retro number {i} on {STREETS[i % 20]} day", [helios],
                  days_ago=i % 30)
    for i in range(40):
        _remember(store, f"Tip {i}: use the new tools for work tickets", days_ago=i % 30)
    for place, dish in zip(PLACES, DISHES):
        _remember(store, f"Ada loved the {dish} at {place}", [ada])
    for i in range(18):
        _remember(store, f"Ada ran {3 + i} km before work", [ada])
    store.refresh_property_vectors(user_id=USER)
    return store, answer


def _which_tools(store, answer):
    question = "What tools does Ada use for her work?"
    # not in the text ranking the linked search starts from
    assert answer.id not in _text(store, question)
    store.decider = judge = _Judge(scores={"Postgres": 0.9}, rest=0.05)
    return judge, store.search(question, user_id=USER, limit=5)


def test_a_projects_answer_joins_the_pool_by_what_it_says_at_scale(stores):
    """Of Helios's 60 memories, the one that says what the question asks
    joins the pool ("it uses Postgres in production" against "What tools
    does it use for work?"), though it is the oldest; judged, it comes
    first (0.9 x 0.5 against 0.05 x 1.0 for Ada's own)."""
    store, answer = _a_project_among_many(stores)
    judge, results = _which_tools(store, answer)
    assert "it uses Postgres in production" in judge.read
    assert results[0].memory.id == answer.id


def test_a_projects_answer_joins_the_pool_by_what_it_says_at_scale_fails_without_the_rule(
        stores, monkeypatch):
    """Without the family's best memories in the pool, the answer never
    reaches the linked order, and is lost."""
    store, answer = _a_project_among_many(stores)
    monkeypatch.setattr(store_module, "FAMILY_TOP", 0)
    judge, results = _which_tools(store, answer)
    assert "about" in results[0].signals
    assert "it uses Postgres in production" not in judge.read
    assert answer.id not in _ids(results)


# ================================================================== R-128
# A question names an entity by one word of a longer name: "Arvel" for the
# place stored as "Mount Arvel". The word seeds the search when it is a word
# of that entity's names alone and the store uses it for that entity more
# likely than not (``graph_retrieval.named_by_a_word``, stage 1). At
# fe094fc a question named an entity only by a name or alias in full: the
# person asked about was the one seed, the question read "When did it do yoga
# at Arvel?", and the answer, whose names read "it" in its property vector
# ("it did yoga on top of it"), lost to the person's other yoga memories.
ARVEL_QUESTION = "When did Harlow do yoga at Arvel?"
ARVEL_ANSWER = ("On 5 June Harlow did yoga on top of Mount Arvel in the morning and shared "
                "a photo of the view")


def _a_place_named_by_a_word(stores):
    """Harlow did yoga in 40 parks; once on top of Mount Arvel, a place with
    one other memory. One memory matches more of the question's words than
    the answer does ("When Harlow has nothing to do, Harlow does yoga at
    home"), and the store holds no turns that would weigh those words down
    (R-129): it takes the keyword place. 30 notes name nobody: 73 memories.
    A name's words weigh twice a word in the vectors: an embedder follows a
    rare name."""
    store = stores(_SenseEmbedder(names=["Harlow", "Mount Arvel"], name_weight=2.0))
    harlow = _entity(store, "Harlow", "person")
    arvel = _entity(store, "Mount Arvel", "place")
    for i in range(40):
        _remember(store, f"Harlow did yoga in {STREETS[i % 20]} park", [harlow], days_ago=i)
    answer = _remember(store, ARVEL_ANSWER, [harlow, arvel], days_ago=100)
    _remember(store, "Mount Arvel has a hut below the summit", [arvel], days_ago=100)
    thief = _remember(store, "When Harlow has nothing to do, Harlow does yoga at home", [harlow])
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    store.refresh_property_vectors(user_id=USER)
    scope = Scope(user_id=USER)
    assert store.backend.keyword_search(ARVEL_QUESTION, scope, 1)[0][0].id == thief.id
    return store, harlow, arvel, answer


def _yoga_on_arvel(store, answer):
    store.decider = judge = _Judge(scores={"photo of the view": 0.9}, rest=0.05)
    return judge, store.search(ARVEL_QUESTION, user_id=USER, limit=5)


def test_a_word_of_one_name_seeds_the_search_at_scale(stores):
    """R-128. "Arvel" is a word of Mount Arvel's name alone, and both memories
    that say it are Mount Arvel's: it seeds the search beside Harlow. Naming
    two hubs, the question is read as written, the answer meets it on both
    names, and judged, it comes first. It failed at fe094fc."""
    store, harlow, arvel, answer = _a_place_named_by_a_word(stores)
    judge, results = _yoga_on_arvel(store, answer)
    assert judge.states == [f"QUESTION: {ARVEL_QUESTION}"]
    assert answer.content in judge.read
    assert results[0].memory.id == answer.id


def test_a_word_of_one_name_seeds_the_search_at_scale_fails_without_the_rule(
        stores, monkeypatch):
    """Without the rule at the seeds, Harlow is the one seed, the question
    reads "When did it do yoga at Arvel?", 40 parks come before the answer,
    the keyword place holds another memory, and the answer is lost."""
    store, _, _, answer = _a_place_named_by_a_word(stores)
    monkeypatch.setattr(graph_retrieval, "named_by_a_word", lambda *args, **kwargs: [])
    judge, results = _yoga_on_arvel(store, answer)
    assert judge.states == ["QUESTION: When did it do yoga at Arvel?"]
    assert answer.content not in judge.read
    assert answer.id not in _ids(results)


def test_the_seeds_hold_the_entity_a_word_names(stores):
    """Stage 1: the seeds are Harlow, named in full, and Mount Arvel, named by
    a word of its name."""
    store, harlow, arvel, _ = _a_place_named_by_a_word(stores)
    seeds, _ = store._seeds(ARVEL_QUESTION, Scope(user_id=USER))
    assert set(seeds) == {harlow.id, arvel.id}


def _words_that_name_nothing(stores):
    """Harlow did yoga in 20 parks the store knows by name ("Linden Park",
    ...), two memories each; Lotus Studio, where Harlow bought a class pass,
    and six memories of the studio at home that do not name it: 47
    memories."""
    store = stores(_SenseEmbedder(names=["Harlow", "Lotus Studio",
                                         *[f"{s} Park" for s in STREETS]]))
    harlow = _entity(store, "Harlow", "person")
    parks = {name: _entity(store, f"{name} Park", "place") for name in STREETS}
    for name, park in parks.items():
        _remember(store, f"Harlow did yoga in {name} Park at dawn", [harlow, park])
        _remember(store, f"{name} Park has a pond and a cafe", [park])
    lotus = _entity(store, "Lotus Studio", "organization")
    _remember(store, "Harlow bought a class pass at Lotus Studio", [harlow, lotus])
    for thing in ("a new mirror", "a heater", "cork blocks", "a speaker", "a plant", "a rug"):
        _remember(store, f"Harlow's home studio got {thing}", [harlow])
    return store, harlow, parks


def test_a_word_many_names_share_seeds_nothing(stores):
    """"park" is a word of 20 names: it names none of them, and Harlow alone
    seeds the search. "Linden", a word of one name that the store says of
    that park only, does seed it."""
    store, harlow, parks = _words_that_name_nothing(stores)
    scope = Scope(user_id=USER)
    assert store._seeds("When did Harlow do yoga in the park?", scope) == ([harlow.id], False)
    seeds, _ = store._seeds("When did Harlow do yoga in Linden?", scope)
    assert set(seeds) == {harlow.id, parks["Linden"].id}


def test_a_word_one_name_holds_that_the_store_says_of_other_things_seeds_nothing(stores):
    """"studio" is a word of Lotus Studio's name alone, but of the seven
    memories that say it, one is Lotus Studio's: the store says it of other
    things, and it names nothing."""
    store, harlow, _ = _words_that_name_nothing(stores)
    scope = Scope(user_id=USER)
    assert store._seeds("When did Harlow do yoga in the studio?", scope) == ([harlow.id], False)


# ================================================================== R-129
# The keyword search weighs each word of the question by how rare it is in
# everything the store holds: its memories and the turns they were said in
# (``LocalBackend.keyword_search``). At fe094fc it weighed a word by the
# memories alone. In a store of third-person facts "did" and "we" are rare
# there and common in what was said, and the one keyword match a search keeps
# in its judged pool (R-21) went to a memory that shares only those words.
PAID_QUESTION = "Did we pay invoice 2024-117?"


def _invoices_and_turns(stores, *, turns=True):
    """R-21's 66 invoices and bills (``_invoices``), and one memory more that
    says "we did": "We did the quarterly tax filing with the accountant".
    With ``turns``, each memory rests on the turn it was said in, in the
    first person ("Did we pay invoice 2024-201? We did, by card."): 67
    memories and 67 turns."""
    store = stores(_SenseEmbedder())
    said: list[tuple[str, str, float]] = []
    for i in range(45):
        said.append((f"Invoice 2024-{200 + i} was paid by card",
                     f"Did we pay invoice 2024-{200 + i}? We did, by card.", i % 10))
    for i, bill in enumerate(["electricity", "water", "phone", "internet", "rent"] * 4):
        said.append((f"We paid the {bill} bill number {i}",
                     f"Did we pay the {bill} bill? We did, number {i}.", i % 10))
    said.append(("Invoice 2024-117 was settled by bank transfer on 3 March",
                 "Invoice 2024-117 went out by bank transfer on 3 March.", 60))
    said.append(("We did the quarterly tax filing with the accountant",
                 "We did the quarterly tax filing with the accountant today.", 5))
    memories = []
    for fact, turn, days_ago in said:
        memory = _remember(store, fact, days_ago=days_ago)
        if turns:
            episode = Episode(content=turn, user_id=USER, created_at=_stamp(days_ago))
            store.backend.add_episodes([episode])
            store.backend.update_memory(memory.id, source_episode_ids=[episode.id], touch=False)
        memories.append(memory)
    answer, thief = memories[-2], memories[-1]
    return store, answer, thief


def _paid(store, answer):
    text = _text(store, PAID_QUESTION)
    assert answer.id not in text[:20]  # below the judged pool in the text ranking
    store.decider = judge = _Judge(scores={"2024-117": 0.9}, rest=0.05)
    return judge, store.search(PAID_QUESTION, user_id=USER, limit=5)


def test_the_keyword_place_weighs_words_by_what_was_said_at_scale(stores):
    """R-129. "did" and "we" are in one memory and in most turns; the
    identifier is in one memory and one turn. Weighed over both, the answer
    is the keyword search's best match, keeps its place among the 20 judged,
    and comes first. It failed at fe094fc."""
    store, answer, thief = _invoices_and_turns(stores)
    best = store.backend.keyword_search(PAID_QUESTION, Scope(user_id=USER), 1)
    assert best[0][0].id == answer.id
    judge, results = _paid(store, answer)
    assert answer.content in judge.read
    assert results[0].memory.id == answer.id


def test_the_keyword_place_weighs_words_by_what_was_said_at_scale_fails_without_the_rule(
        stores, monkeypatch):
    """Weighed over the memories alone, as ``bm25()`` weighs them, "did" and
    "we" outweigh the identifier: the keyword place goes to the tax filing,
    and the answer is lost."""
    store, answer, thief = _invoices_and_turns(stores)
    monkeypatch.setattr(LocalBackend, "_word_weights",
                        lambda self, words, counts=None: {word: 1.0 for word in words})
    assert store.backend.keyword_search(PAID_QUESTION, Scope(user_id=USER), 1)[0][0].id == \
        thief.id
    judge, results = _paid(store, answer)
    assert answer.content not in judge.read
    assert answer.id not in _ids(results)


def test_without_turns_a_word_weighs_as_bm25_weighs_it(stores):
    """Stage 2: a store that holds no turns ranks as ``bm25()`` over the
    memories: the same memories without their turns give the tax filing the
    keyword search's first place, in the order of an OR query of the words."""
    store, answer, thief = _invoices_and_turns(stores, turns=False)
    scope = Scope(user_id=USER)
    ranked = [m.id for m, _ in store.backend.keyword_search(PAID_QUESTION, scope, 40)]
    assert ranked[0] == thief.id
    match = " OR ".join(f'"{word}"' for word in re.findall(r"[A-Za-z0-9]+", PAID_QUESTION))
    rows = store.backend._db.execute(
        "SELECT m.id FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
        "WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts), m.id LIMIT 40", (match,))
    assert ranked == [row["id"] for row in rows]


# ================================================ rows the benchmark measured
# The relative retrieval benchmark (``evals/relative_retrieval_benchmark.py``)
# measures these families with real vectors and Jev. The rules they rest on
# are stubs' business: how the links weigh a version, an occurrence and a
# sibling, the sharpness of the property comparison, the judge's shortlist,
# and a set question starting at the owner. Each is tested in a store of
# around a hundred memories or more, where the rule decides, and each has a
# companion that takes the rule away.
class _ReleaseEmbedder(_SenseEmbedder):
    """``_SenseEmbedder`` with the senses of releases, events and purchases."""

    SENSES = {**_SenseEmbedder.SENSES,
              "storage": "store stores stored data database keeps",
              "add": "add adds added introduced",
              "venue": "take takes took place held hosted",
              "attend": "attendees attended attendance",
              "car": "car cars",
              "cheap": "cheapest cheap costs cost quoted price euros",
              "drive": "drive drove driven",
              "dine": "restaurants restaurant food dinner lunch"}


def _belongs(store, child, parent, p=0.9, kind="kind"):
    """A compared pair: ``child`` is a version or an occurrence ("kind") or a
    part of ``parent``, and not the same thing."""
    from memry.models import MergeProposal

    belongs = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
               "b_part_of_a": 0.0, "neither": round(1.0 - p, 6)}
    belongs[f"a_{kind}_of_b"] = p
    store.backend.add_proposal(MergeProposal(
        entity_a=child.id, entity_b=parent.id, user_id=USER, confidence=0.1,
        different=0.9, belongs=belongs))


def _links_weigh_nothing(monkeypatch, store):
    """The rule taken away: the links reach every entity of the store as the
    one the question names (1.0, none by a step up), so what a memory is
    tagged with no longer tells it from another's: each is read with its
    names as "it", counts 1.0, and no answer yields to a nearer one."""
    everyone = {e.id: 1.0 for e in store.backend.list_entities(Scope(user_id=USER), limit=1000)}
    monkeypatch.setattr(store_module, "activation_paths",
                        lambda backend, seeds, depth=1: (dict(everyone), set()))


# -------------------------------------------- the sharpness of the property
# The linked order is property similarity to the power
# ``relational_sharpness``, times aboutness (``_linked_order``). At 1.0 a
# version's own answer comes before its thing's where it states the property
# at least 0.72 as well as the thing's does (UP_KIND x 0.9); at 2.0 or 3.0
# the worded overrides of the benchmark fell to 0.06 and 0.00.
OVERRIDE = "With its third release, bildy moved its data to CouchDB"


def _a_worded_override(stores):
    """bildy, a product with 25 memories, "bildy stores its data in SQLite"
    among them; its versions v1 to v3, 11 memories each, and v3's worded
    override; 30 notes: 89 memories, searched by vectors alone."""
    store = stores(_ReleaseEmbedder(names=["bildy", "bildy v1", "bildy v2", "bildy v3"]))
    bildy = _entity(store, "bildy", "product")
    thing = _remember(store, "bildy stores its data in SQLite", [bildy])
    for i in range(24):
        _remember(store, f"bildy had its sprint review number {i} on {STREETS[i % 20]} day",
                  [bildy])
    versions = {}
    for n in (1, 2, 3):
        version = versions[n] = _entity(store, f"bildy v{n}", "product")
        _belongs(store, version, bildy)
        for i in range(11):
            _remember(store, f"bildy v{n} fixed bug number {100 * n + i}", [version])
    answer = _remember(store, OVERRIDE, [versions[3]])
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    store.refresh_property_vectors(user_id=USER)
    store.config.retrieval.relational_relevance = "vector"
    return store, answer, thing


def test_a_versions_worded_override_comes_first_at_the_default_sharpness(stores):
    """"Where does bildy v3 store its data?" reads "Where does it
    store its data?". v3's own answer ("With its third release, it moved
    its data to CouchDB") states it at 0.8 of bildy's ("it stores its data
    in SQLite"), whose memories count 0.72: at the default sharpness of 1.0
    v3's answer comes first."""
    assert Config().retrieval.relational_sharpness == 1.0
    store, answer, thing = _a_worded_override(stores)
    results = store.search("Where does bildy v3 store its data?", user_id=USER, limit=5)
    assert "about" in results[0].signals
    assert _ids(results)[:2] == [answer.id, thing.id]


def test_a_versions_worded_override_comes_first_at_the_default_sharpness_fails_without_the_rule(
        stores):
    """At a sharpness of 2.0 the similarities are squared, 0.8 becomes 0.64,
    under 0.72, and bildy's default comes before v3's own answer."""
    store, answer, thing = _a_worded_override(stores)
    store.config.retrieval.relational_sharpness = 2.0
    results = store.search("Where does bildy v3 store its data?", user_id=USER, limit=5)
    assert _ids(results)[:2] == [thing.id, answer.id]


# ------------------------------------------------------ versions in words
# Worded versions: "The first release of bildy added ..." says no "v1", so
# the text cannot tell the releases apart; the entity each memory is tagged
# with does (aboutness: the version asked 1.0, another release, which the
# links do not reach, ``LOW``).
FEATURES = ["a timeline view", "dark mode", "CSV export", "two-factor login", "a plugin API",
            "shared folders", "voice notes", "calendar sync", "an audit log", "bulk editing",
            "keyboard shortcuts", "a public API", "tagging", "comments", "a mobile widget",
            "PDF export", "search filters", "offline maps", "a web clipper", "emoji reactions",
            "a kanban board", "spell check", "read receipts", "a dark icon"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth"]
FIRST_RELEASE = ("The first release of bildy added a timeline view, which the team had "
                 "promised for a year")


def _worded_releases(stores):
    """bildy (20 memories) and nine releases, each a version of bildy, each
    tagged on its own memories. The first release: the answer, worded as
    people write it, and five more. The other eight: three features each,
    "The second release of bildy added dark mode", shorter than the
    answer. 30 notes: 80 memories."""
    names = ["bildy", *[f"bildy v{n}" for n in range(1, 10)]]
    store = stores(_ReleaseEmbedder(names=names))
    bildy = _entity(store, "bildy", "product")
    for i in range(20):
        _remember(store, f"bildy had its sprint review number {i} on {STREETS[i % 20]} day",
                  [bildy])
    releases = {}
    for n in range(1, 10):
        releases[n] = _entity(store, f"bildy v{n}", "product")
        _belongs(store, releases[n], bildy)
    answer = _remember(store, FIRST_RELEASE, [releases[1]], days_ago=400)
    for i in range(5):
        _remember(store, f"The first release of bildy fixed bug number {i}", [releases[1]],
                  days_ago=400)
    for n in range(2, 10):
        for k in range(3):
            _remember(store, f"The {ORDINALS[n - 1]} release of bildy added "
                             f"{FEATURES[(3 * n + k) % len(FEATURES)]}", [releases[n]],
                      days_ago=10 - n)
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    store.refresh_property_vectors(user_id=USER)
    return store, answer


def _what_v1_added(store):
    store.decider = judge = _Judge(scores={"added": 0.9}, rest=0.05)
    return judge, store.search("What did bildy v1 add?", user_id=USER, limit=5)


def test_only_the_entity_tells_worded_releases_apart_at_scale(stores):
    """"What did bildy v1 add?" reads "What did it add?". The answer says
    no "v1" ("The first release of bildy added a timeline view, ..."), and
    the 24 features of the eight other releases say as much ("The second
    release of bildy added dark mode"). What tells them apart is the entity
    each is tagged with: bildy v1, the one asked (1.0, its names and
    bildy's read "it"), or a release the links do not reach (0.3, read as
    written). Judged alike, the answer comes first."""
    store, answer = _worded_releases(stores)
    judge, results = _what_v1_added(store)
    assert judge.states == ["QUESTION: What did it add?"]
    assert answer.content.replace("bildy", "it") in judge.read
    assert results[0].memory.id == answer.id and results[0].signals["about"] == 1.0


def test_only_the_entity_tells_worded_releases_apart_at_scale_fails_without_the_rule(
        stores, monkeypatch):
    """Where the links reach every release alike, each is read as the one
    asked ("The second release of it added dark mode", shorter than the
    answer and nearer "What did it add?") and counts 1.0: the 24 features
    of the other releases fill the 20 the judge reads, and the first
    release's answer is lost."""
    store, answer = _worded_releases(stores)
    _links_weigh_nothing(monkeypatch, store)
    judge, results = _what_v1_added(store)
    assert answer.content.replace("bildy", "it") not in judge.read
    assert answer.id not in _ids(results)


# ---------------------------------------------- an event and its editions
# An occurrence of an event series takes what is true of the series where it
# says nothing itself (the series 0.72, UP_KIND x 0.9), its own memory over
# the series' (1.0, and the series' answer yields to it), and its own over a
# sibling occurrence's (``LOW``: the links do not reach a sibling at depth 1).
SERIES_CITY = "Tovel Forum takes place in Lyon each year"
MOVED = "Tovel Forum 2025 was held in Graz for one year while the Lyon hall was rebuilt"
ATTENDED = ("Tovel Forum 2023 was attended by 356 people, fewer than planned because of "
            "the rail strike")
TOPICS = ["open data", "edge computing", "soil health", "typography", "accessibility",
          "robotics", "urban farming", "privacy", "energy", "maps"]


def _an_event_series(stores, *, p=0.9, city=SERIES_CITY,
                     parts=("opening", "party", "workshop day")):
    """Tovel Forum (21 memories, ``city`` among them) and its editions 2019
    to 2025 (10 memories each), each an occurrence of the series at ``p``.
    2025 moved to Graz; the other editions say where their ``parts`` took
    place ("The Tovel Forum 2023 opening took place in Turin", shorter than
    the series' city); every edition but 2023 says "had N attendees", and
    2023 says it in more words. 30 notes: 121 memories."""
    names = ["Tovel Forum", *[f"Tovel Forum {year}" for year in range(2019, 2026)]]
    store = stores(_ReleaseEmbedder(names=names))
    series = _entity(store, "Tovel Forum", "event")
    found = {"city": _remember(store, city, [series])}
    for i in range(20):
        _remember(store, f"Tovel Forum has a track on {TOPICS[i % 10]} number {i}", [series])
    cities = ["Turin", "Porto", "Graz", "Riga", "Brno", "Ghent"]
    for year in range(2019, 2026):
        edition = _entity(store, f"Tovel Forum {year}", "event")
        _belongs(store, edition, series, p)
        o = f"Tovel Forum {year}"
        if year == 2023:
            found["attended"] = _remember(store, ATTENDED, [edition])
        else:
            _remember(store, f"{o} had {300 + year % 100} attendees", [edition])
        if year == 2025:
            found["moved"] = _remember(store, MOVED, [edition])
        elif year != 2024:
            for part in parts:
                _remember(store, f"The {o} {part} took place in {cities[year % 6]}", [edition])
        while store.backend.count_entity_memories(edition.id) < 10:
            k = store.backend.count_entity_memories(edition.id)
            _remember(store, f"The keynote at {o} was about {TOPICS[(year + k) % 10]}",
                      [edition])
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    store.refresh_property_vectors(user_id=USER)
    return store, found


def _asked(store, question, scores):
    store.decider = judge = _Judge(scores=scores, rest=0.03)
    return judge, store.search(question, user_id=USER, limit=5)


WHERE = {"place": 0.9, "Graz": 0.9}
HOW_MANY = {"attend": 0.9}


def test_an_occurrence_takes_its_series_fact_over_a_siblings_own_at_scale(stores):
    """"Where did Tovel Forum 2024 take place?": 2024 says nothing of
    where. The judge scores the series' city and the other editions' "took
    place" alike; the series counts 0.72 for 2024, an edition the links do
    not reach 0.3, and the series' city comes first."""
    store, found = _an_event_series(stores)
    judge, results = _asked(store, "Where did Tovel Forum 2024 take place?", WHERE)
    assert "it takes place in Lyon each year" in judge.read
    assert results[0].memory.id == found["city"].id


def test_an_occurrences_own_answer_overrides_its_series_at_scale(stores):
    """"Where did Tovel Forum 2025 take place?": its own memory says Graz
    (1.0), the series' says Lyon (0.72, and yielding to the edition's own
    answer); judged alike, Graz comes first, though the series' city is
    nearer the question."""
    store, found = _an_event_series(stores)
    judge, results = _asked(store, "Where did Tovel Forum 2025 take place?", WHERE)
    assert found["moved"].content.replace("Tovel Forum 2025", "it") in judge.read
    assert results[0].memory.id == found["moved"].id


def test_an_occurrences_own_fact_comes_before_its_siblings_at_scale(stores):
    """"How many attendees did Tovel Forum 2023 have?": six other
    editions say "had N attendees", in fewer words than 2023's own answer;
    judged alike, 2023's (1.0) comes before theirs (0.3)."""
    store, found = _an_event_series(stores)
    judge, results = _asked(store, "How many attendees did Tovel Forum 2023 have?", HOW_MANY)
    assert found["attended"].content.replace("Tovel Forum 2023", "it") in judge.read
    assert results[0].memory.id == found["attended"].id


@pytest.mark.parametrize("question,scores,answer", [
    ("Where did Tovel Forum 2024 take place?", WHERE, "city"),
    ("Where did Tovel Forum 2025 take place?", WHERE, "moved"),
    ("How many attendees did Tovel Forum 2023 have?", HOW_MANY, "attended"),
])
def test_an_occurrence_is_told_from_its_series_and_siblings_fails_without_the_rule(
        stores, monkeypatch, question, scores, answer):
    """Where the links reach every edition and the series alike, what the
    judge scores alike goes by the order it was judged in, the nearer text
    first: the other editions' "took place" come before the series' city
    and before 2025's own, their "had N attendees" before 2023's."""
    store, found = _an_event_series(stores)
    _links_weigh_nothing(monkeypatch, store)
    _, results = _asked(store, question, scores)
    assert results[0].memory.id != found[answer].id


# ------------------------------------------ an edition linked weakly to its series
# A link, however weak, never ranks below no link (``aboutness``). An edition
# Jev linked to its series at 0.3 or 0.2 reaches the series at 0.24 or 0.16
# (UP_KIND x p), under ``LOW``: the series' memories count as much as a
# sibling edition's, which the links do not reach, and no less. Both tests
# failed at a5eecd8, each at its own stage: the order put the series' answer
# below the siblings' (the override's discount came after the floor), and
# the candidates left it out (only an entity reached at 0.3 or more added
# its best memories).
WEAK = [0.3, 0.2]
HOSTED = "Tovel Forum is hosted in Lyon"
NINE_PARTS = ("opening", "party", "workshop day", "gala", "hackathon", "closing",
              "poster session", "panel", "career fair")


@pytest.mark.parametrize("p", WEAK)
def test_an_occurrence_linked_weakly_takes_its_series_fact_over_a_siblings_own_at_scale(
        stores, p):
    """R-52's store with each edition linked to the forum at ``p``. The
    judge reads the series' city and scores it and the other editions'
    "took place" alike (0.9). The city yields to 2024's own memories, none
    of which answers (0.03 off), and what the override leaves is floored
    as the link is: at 0.3, 0.9 x max(0.97 x 0.24, 0.3) = 0.27, as much as
    an unreached edition's 0.9 x 0.3. The tie keeps the order judged, where
    the city ("it takes place in Lyon each year") states the property
    better than a sibling's, read with its names. At a5eecd8 the discount
    came after the floor, 0.9 x 0.97 x 0.3 = 0.26, and every sibling's
    "took place" the judge read came first."""
    store, found = _an_event_series(stores, p=p)
    judge, results = _asked(store, "Where did Tovel Forum 2024 take place?", WHERE)
    assert "it takes place in Lyon each year" in judge.read
    assert results[0].memory.id == found["city"].id
    assert results[0].signals["overridden"] == pytest.approx(0.03)


def test_an_occurrence_linked_weakly_takes_its_series_fact_fails_without_the_rule(
        stores, monkeypatch):
    """With the floor taken off what the override leaves (``LOW`` 0 in the
    final order), the series' city counts 0.9 x 0.97 x 0.3 against a
    sibling's 0.9 x 0.3: another edition's "took place" comes first, and
    the city is not among the first five."""
    store, found = _an_event_series(stores, p=0.3)
    monkeypatch.setattr(store_module, "LOW", 0.0)
    judge, results = _asked(store, "Where did Tovel Forum 2024 take place?", WHERE)
    assert "it takes place in Lyon each year" in judge.read
    assert "took place" in results[0].memory.content
    assert found["city"].id not in _ids(results)


def _hosted(store, found):
    question = "Where did Tovel Forum 2024 take place?"
    # not in the text ranking the linked search starts from
    assert found["city"].id not in _text(store, question)
    return _asked(store, question, {"place": 0.9, "hosted": 0.9})


@pytest.mark.parametrize("p", WEAK)
def test_a_weakly_linked_series_answer_joins_the_pool_by_what_it_says_at_scale(stores, p):
    """The series' city worded "Tovel Forum is hosted in Lyon", and each
    edition 2019 to 2023 saying where nine of its parts took place: 45
    memories say "place" and a year, and the city is not in the text
    ranking. It joins the pool as the series' memory that best states the
    property ("it is hosted in Lyon"), as a linked entity's best memories
    do however weak the link, and comes first in the linked order (0.3 x
    0.77 against 1.0 x 0.22 for 2024's own and 0.3 x 0.60 for a sibling's
    "took place"); judged alike with the siblings', it stays first. At
    a5eecd8 an entity reached under 0.3 added no candidates, and the city
    never reached the judge."""
    store, found = _an_event_series(stores, p=p, city=HOSTED, parts=NINE_PARTS)
    judge, results = _hosted(store, found)
    assert "it is hosted in Lyon" in judge.read
    assert results[0].memory.id == found["city"].id


def test_a_weakly_linked_series_answer_joins_the_pool_fails_without_the_rule(
        stores, monkeypatch):
    """Without the linked entities' best memories in the pool, the city
    never reaches the judge, and an edition's "took place" comes first."""
    store, found = _an_event_series(stores, p=0.3, city=HOSTED, parts=NINE_PARTS)
    monkeypatch.setattr(store_module, "FAMILY_TOP", 0)
    judge, results = _hosted(store, found)
    assert "it is hosted in Lyon" not in judge.read
    assert found["city"].id not in _ids(results)


# ------------------------------------------- a thing's answer and a version
# The dense world's "Which platforms does bildy v2 run on?": the answer is
# bildy's (0.72 for v2), and v2's own memories (1.0), none of which answers,
# compete. The judge reads the first 20 of the linked order and orders them
# by its judgement times aboutness: 0.9 x 0.72 for the answer against 0.05 x
# 1.0 for each of v2's own. By vectors alone v2's own near miss comes first.
PLATFORMS = "bildy runs on Linux and macOS"
NEAR_MISS = "bildy v2 runs its Windows installer silently"


def _a_version_among_its_own(stores):
    """bildy (40 memories: the answer, two near misses of the dense world,
    37 more) and its versions v1 to v3, each with 40 memories of its own;
    v2's hold a near miss that says "runs" and "Windows". 30 notes: 190
    memories."""
    store = stores(_ReleaseEmbedder(names=["bildy", "bildy v1", "bildy v2", "bildy v3"]))
    bildy = _entity(store, "bildy", "product")
    answer = _remember(store, PLATFORMS, [bildy], days_ago=300)
    _remember(store, "the bildy sync backend is deployed on Fly.io", [bildy])
    _remember(store, "bildy's CI builds run on Ubuntu runners", [bildy])
    for i in range(37):
        _remember(store, f"bildy had its sprint review number {i} on {STREETS[i % 20]} day",
                  [bildy])
    near = None
    for n in (1, 2, 3):
        version = _entity(store, f"bildy v{n}", "product")
        _belongs(store, version, bildy)
        if n == 2:
            near = _remember(store, NEAR_MISS, [version])
        while store.backend.count_entity_memories(version.id) < 40:
            k = store.backend.count_entity_memories(version.id)
            _remember(store, f"bildy v{n} fixed bug number {100 * n + k}", [version])
    for i in range(30):
        _remember(store, f"Note {i}: renew the parking permit before the month ends")
    store.refresh_property_vectors(user_id=USER)
    return store, answer, near


def _which_platforms(store):
    store.decider = judge = _Judge(scores={"Linux and macOS": 0.9}, rest=0.05)
    return judge, store.search("Which platforms does bildy v2 run on?", user_id=USER, limit=5)


def test_a_things_answer_comes_before_the_versions_own_at_scale(stores):
    """"Which platforms does bildy v2 run on?" Of 190 memories, the judge
    reads 20, bildy's answer among them ("it runs on Linux and macOS"), and
    it comes first past v2's 40 of its own."""
    store, answer, _ = _a_version_among_its_own(stores)
    judge, results = _which_platforms(store)
    assert len(judge.batches[0]) == 20
    assert "it runs on Linux and macOS" in judge.read
    assert results[0].memory.id == answer.id


def test_a_things_answer_comes_before_the_versions_own_at_scale_fails_without_the_rule(
        stores):
    """Without the judge picking from the first 20 (relevance by vectors
    alone), v2's own near miss ("it runs its Windows installer silently",
    1.0) comes before bildy's answer (0.72)."""
    store, answer, near = _a_version_among_its_own(stores)
    store.config.retrieval.relational_relevance = "vector"
    _, results = _which_platforms(store)
    assert results[0].memory.id == near.id
    assert _ids(results).index(answer.id) > 0


# -------------------------------------------------------- sets of the owner
# A set question in the first person or naming the owner starts at the
# owner, an entity of hundreds of memories: the judge reads the first 20 of
# the linked order, the second call the memories nearest the members found
# (the store has no tags), and every member comes back past the limit.
# "Which car is the cheapest?" names nobody and takes the text route
# (``test_a_set_question_returns_every_member_unfiltered``).
CAR_NAMES = [f"{brand} {model} {trim}" for brand, model in [
    ("Skoda", "Octavia"), ("Renault", "Zoe"), ("Toyota", "Yaris"), ("Fiat", "Panda"),
    ("Kia", "Niro"), ("Hyundai", "Kona"), ("Seat", "Leon"), ("Dacia", "Spring"),
    ("Mazda", "Mizu"), ("Volvo", "Ekko"), ("Opel", "Corsa"), ("Citroen", "Ami"),
    ("Honda", "Jazz"), ("Nissan", "Leaf"), ("Ford", "Puma"), ("Cupra", "Born"),
    ("Audi", "Sella"), ("Mini", "Aceman"), ("Lancia", "Ypsilon"), ("Smart", "Hashtag")]
    for trim in ("Base", "Sport")]
LIKED = ["Olive Kitchen", "Harbour Bistro", "Linden Trattoria", "Copper Kitchen",
         "Saffron Bistro", "Juniper Kitchen", "Maple Trattoria", "Fig Bistro", "Cedar Kitchen",
         "Pepper Trattoria", "Clover Bistro", "Amber Kitchen"]
DISLIKED = ["Sage Bistro", "Quince Kitchen", "Ember Trattoria", "Birch Bistro", "Hazel Kitchen",
            "Plum Trattoria", "Rowan Bistro", "Thyme Kitchen"]
LUNCH = ["Willow Bistro", "Fennel Kitchen", "Basil Trattoria", "Nutmeg Bistro", "Laurel Kitchen"]
MONTHS = ["January", "March", "May", "July", "September", "November"]


def _the_owners_store(stores):
    """Ilva Marsh, the store's owner: 120 everyday memories; 40 cars, a
    price each (half quoted to her, half about the car alone), 21 test
    drives, 19 ranges and 20 insurance quotes; 40 grocery purchases among 40
    other expenses and refunds and "wants to spend less on groceries"; 12
    restaurants she liked, her favourite, 8 she did not like and 9 lunches
    (5 elsewhere, 4 at places she liked). 331 memories."""
    names = ["Ilva Marsh", *CAR_NAMES, *LIKED, *DISLIKED, *LUNCH, *FRIENDS]
    store = stores(_ReleaseEmbedder(names=names))
    owner = _entity(store, "Ilva Marsh", "person")
    store._upkeep_set("owner_entity", USER, owner.id)
    gold: dict[str, list] = {}
    everyday = ["ran {n} km along the coast", "met {f} at the gym", "fixed the tap at home {n}",
                "watched a film with {f}", "called {f} about the trip", "read {n} pages"]
    for i in range(120):
        _remember(store, f"Ilva Marsh {everyday[i % 6].format(n=i + 2, f=FRIENDS[i % 10])}",
                  [owner], days_ago=i % 50)
    cars = [_entity(store, name, "product") for name in CAR_NAMES]
    gold["prices"], gold["drives"] = [], []
    for i, car in enumerate(cars):
        price = f"{14_000 + 1_100 * i:,}"
        if i % 2:
            gold["prices"].append(_remember(store, f"The {car.name} costs {price} euros", [car]))
        else:
            gold["prices"].append(_remember(
                store, f"Ilva Marsh was quoted {price} euros for the {car.name}", [car, owner]))
        if i <= 20:
            gold["drives"].append(_remember(
                store, f"Ilva Marsh test drove the {car.name} in {MONTHS[i % 6]}", [car, owner]))
        else:
            _remember(store, f"The {car.name} has a range of {250 + 10 * i} km", [car])
        if i % 2 == 0:
            _remember(store, f"Insurance for the {car.name} would be {300 + 20 * i} euros a year",
                      [car])
    shops = ["Lidl", "Aldi", "Rewe", "Edeka", "the farmers' market"]
    gold["groceries"] = [_remember(
        store, f"Ilva Marsh spent {8 + 3 * i} euros on groceries at {shops[i % 5]} on "
               f"{1 + i % 28} {MONTHS[i % 6]}", [owner]) for i in range(40)]
    for i in range(40):
        text = (f"Ilva Marsh spent {5 + 7 * i} euros on fuel on {1 + i % 28} {MONTHS[i % 6]}"
                if i % 2 else
                f"Ilva Marsh got a {5 + 7 * i} euro refund from {shops[i % 5]}")
        _remember(store, text, [owner])
    _remember(store, "Ilva Marsh wants to spend less on groceries", [owner])
    places = {name: _entity(store, name, "organization") for name in LIKED + DISLIKED + LUNCH}
    gold["liked"] = [_remember(
        store, (f"Ilva Marsh liked the food at {name}" if i % 2 else
                f"Ilva Marsh loved dinner at {name}"), [owner, places[name]])
        for i, name in enumerate(LIKED)]
    gold["liked"].append(_remember(store, "Ilva Marsh's favourite restaurant is Trattoria Sole",
                                   [owner]))
    for i, name in enumerate(DISLIKED):
        _remember(store, (f"Ilva Marsh did not like the food at {name}" if i % 2 else
                          f"Ilva Marsh found {name} disappointing"), [owner, places[name]])
    for i, name in enumerate(LUNCH + LIKED[:4]):
        _remember(store, f"Ilva Marsh had lunch at {name} with {FRIENDS[i]}",
                  [owner, places[name]])
    store.refresh_property_vectors(user_id=USER)
    return store, gold


RESTAURANTS = {"did not like": 0.03, "disappointing": 0.03, "liked the food": 0.5,
               "loved dinner": 0.5, "favourite restaurant": 0.5, "had lunch": 0.2}
GROCERIES = {"wants to spend less": 0.03, "on groceries": 0.6}
OWNER_SETS = [
    ("Which of the cars I looked at is the cheapest?",
     {"test drove": 0.02, "quoted": 0.12, "costs": 0.12}, "prices"),
    ("Which cars did I test drive?", {"test drove": 0.6}, "drives"),
    ("How much did Ilva Marsh spend on groceries?", GROCERIES, "groceries"),
    ("How much did I spend on groceries?", GROCERIES, "groceries"),
    ("Which restaurants did Ilva Marsh like?", RESTAURANTS, "liked"),
    ("Which restaurants did I like?", RESTAURANTS, "liked"),
]


def _the_set(store, question, scores):
    store.decider = judge = _Judge(scores=scores, rest=0.02, several=0.9)
    return judge, store.search(question, user_id=USER, limit=5)


@pytest.mark.parametrize("question,scores,members", OWNER_SETS)
def test_a_set_question_about_the_owner_returns_every_member_at_scale(
        stores, question, scores, members):
    """Set questions about the owner, named or in the first person. The
    owner seeds the search, the judge reads 20 of her 331 memories, the
    second call the rest of the set, and every member comes back, none
    else: all 40 prices of "the cars I looked at", half of them about a car
    she is not linked to (and none of her test drives); 21 test drives; 40
    grocery purchases (not the wish to spend less); 13 restaurants she
    liked, her favourite with them (not the dislikes, and not the lunches,
    which score between)."""
    store, gold = _the_owners_store(stores)
    judge, results = _the_set(store, question, scores)
    read = question.replace("Ilva Marsh", "it").replace(" I ", " it ")
    assert judge.states == [f"QUESTION: {read}"] * 2 and "about" in results[0].signals
    assert len(judge.batches) == 2 and len(judge.batches[0]) == 20
    assert {r.memory.id for r in results} == {m.id for m in gold[members]}
    assert all(r.signals.get("member") for r in results)


@pytest.mark.parametrize("question,scores,members", OWNER_SETS)
def test_a_set_question_about_the_owner_returns_every_member_at_scale_fails_without_the_rule(
        stores, question, scores, members):
    """Without the second call (``retrieval.set_pool`` 0) only the members
    among the 20 judged come back, and the rest of each set is lost: most
    of the prices (the first 20 hold ten of them), some test drives, half
    the purchases, a restaurant she liked or more."""
    store, gold = _the_owners_store(stores)
    store.config.retrieval.set_pool = 0
    judge, results = _the_set(store, question, scores)
    assert len(judge.batches) == 1
    assert {m.id for m in gold[members]} - {r.memory.id for r in results}
