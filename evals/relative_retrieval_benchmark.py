"""Relative relations in retrieval: how should search follow links between a
thing, its versions, its dated occurrences and its parts?

A synthetic store with exact labels, searched through Memry's own
``MemoryStore.search``:

* products with three versions and a part: "Kaven planner", "Kaven planner
  v1" to "v3", "Kaven planner sync service". The thing's memories hold what is
  true of every version (platform, language, database); each version adds a
  feature and a release date, and v3 moves the database (an override);
* events with three dated occurrences: "Tovel Forum", "Tovel Forum 2023" to
  "2025". The event's memories say where it takes place; 2025 moved;
* namesakes sharing a word with a product: "Kaven Bakery";
* the families of ``retrieval_benchmark.py`` (people, projects, tools and the
  typed relations between them), to check that nothing it measured gets worse;
* notes as noise, up to the store size.

Query families, each with the memories that answer it:

  inherit     "Which platforms does Kaven planner v2 run on?"  the thing's memory
  override    "Where does Kaven planner v3 store its data?"   v3's own memory
  sibling     "What did Kaven planner v1 add?"                v1's, not v2's or v3's
  rollup      "What do I know about Kaven planner?"           thing, versions, part
  part        "Who maintains the Kaven planner sync service?" the part's memory
  event_inherit   "Where did Tovel Forum 2024 take place?"    the event's memory
  event_override  "Where did Tovel Forum 2025 take place?"    2025's own memory
  event_sibling   "How many attendees did Tovel Forum 2023 have?"  2023's
  namesake    "What does Kaven Bakery sell?"                  the bakery's
  single_fact, multi_hop, by_entity                           as retrieval_benchmark

Links between the entities, as Memry keeps them on compared pairs:

  none      no compared pairs (the store as it was before the belongs question)
  oracle    the true relation with probability 1
  measured  one of Jev's measured answers for that kind of pair, drawn at
            random (evals/datasets/belongs_answers.json): graded, noisy, with a
            "same" answer on every pair

Search modes (``RetrievalConfig``): hybrid alone; "typed" (today: extracted
relations both ways, 2 hops, rescue fusion); "undirected" (also version and
part links at the belongs bar, unweighted); "directed" (every link weighted by
kind, direction and probability, a step down after a step up held down), each
at depth 1, 2 and 3, with "rescue" or "weighted" fusion.

Run:
    OPENAI_API_KEY=... python evals/relative_retrieval_benchmark.py      # real embeddings
    python evals/relative_retrieval_benchmark.py --sizes 1500             # hash, offline
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from memry.config import Config, EmbeddingConfig  # noqa: E402
from memry.models import Entity, EntityMention, Memory, MergeProposal, Relation  # noqa: E402
from memry.providers.embeddings import Embedder, HashEmbedder, OpenAIEmbedder  # noqa: E402
from memry.providers.llm import NoneLLM  # noqa: E402
from memry.store import MemoryStore  # noqa: E402

USER = "bench"
NOUNS = ["planner", "editor", "tracker", "ledger", "studio", "console", "reader", "scanner"]
EVENT_NOUNS = ["Forum", "Summit", "Days", "Meetup", "Festival", "Conference"]
PLATFORMS = ["Linux and macOS", "Windows only", "iOS and Android", "the web browser",
             "Linux only", "macOS and Windows", "Android only", "every desktop system"]
LANGS = ["Rust", "Go", "Python", "TypeScript", "Kotlin", "Swift", "Elixir", "C#", "Java", "Zig"]
DBS = ["SQLite", "Postgres", "MySQL", "DuckDB", "Redis", "MongoDB", "CouchDB", "Firebird"]
LICENSES = ["MIT", "Apache 2.0", "GPL 3", "BSD", "MPL 2.0", "proprietary"]
FEATURES = ["offline mode", "a timeline view", "dark mode", "CSV export", "two-factor login",
            "a plugin API", "shared folders", "voice notes", "calendar sync", "an audit log",
            "bulk editing", "keyboard shortcuts", "a public API", "tagging", "comments",
            "a mobile widget", "PDF export", "search filters"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
CITIES = ["Lisbon", "Porto", "Graz", "Tallinn", "Ghent", "Bergen", "Turin", "Leipzig",
          "Krakow", "Aarhus", "Bilbao", "Brno", "Riga", "Lyon", "Malmo", "Split"]
TOPICS = ["open data", "edge computing", "soil health", "typography", "accessibility",
          "robotics", "bookbinding", "urban trees", "privacy", "wind power", "fermentation"]
ORGS = ["the city library", "a university lab", "a volunteer group", "a regional bank",
        "a design school", "a farmers' union"]
BREADS = ["sourdough", "rye bread", "croissants", "bagels", "focaccia", "pretzels", "baguettes"]
FIRST = ["Priya", "Jonas", "Mara", "Wei", "Ada", "Tom", "Lena", "Omar", "Sofia", "Kai",
         "Nina", "Raj", "Elsa", "Yuki", "Bea", "Ivan", "Zoe", "Hugo", "Mila", "Theo"]
LAST = ["Nair", "Berg", "Ruiz", "Chen", "Novak", "Diaz", "Kraus", "Sato", "Meyer", "Osei",
        "Popov", "Haas", "Lund", "Ferro", "Blum", "Costa", "Weber", "Reid", "Falk", "Roy"]
SYLLABLES = ["ka", "lo", "mi", "ren", "sa", "tor", "vel", "dri", "ne", "qua", "bra", "fen",
             "gil", "hor", "is", "jun", "kel", "lum", "mor", "nix", "os", "pra", "ril", "sel",
             "tam", "ul", "vor", "wen", "xa", "yor", "zel", "bo", "ce", "du", "ek"]
ORDINALS = ["first", "second", "third"]
TOPICWORDS = ["roadmap", "budget", "latency", "hiring", "design", "billing", "outage",
              "review", "demo", "migration", "release", "research", "pricing", "support"]


class CachedEmbedder(Embedder):
    """One vector per distinct text, fetched in batches and kept on disk."""

    def __init__(self, base: Embedder, path: pathlib.Path) -> None:
        self.base, self.path = base, path
        self.name, self._model, self.dimensions = base.name, base._model, base.dimensions
        self.cache: dict[str, list[float]] = {}
        if path.exists():
            self.cache = json.loads(path.read_text())

    def warm(self, texts: list[str]) -> None:
        missing = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(missing), 256):
            batch = missing[i:i + 256]
            for text, vector in zip(batch, self.base.embed(batch)):
                self.cache[text] = [round(float(x), 6) for x in vector]
            print(f"  embedded {min(i + 256, len(missing))}/{len(missing)}", flush=True)
        if missing:
            self.path.write_text(json.dumps(self.cache))

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.warm(texts)
        return [self.cache[t] for t in texts]


def words(n: int, rnd: random.Random) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add((rnd.choice(SYLLABLES) + rnd.choice(SYLLABLES) + rnd.choice(SYLLABLES)[:2])
                .capitalize())
    return sorted(out)


def person(rnd: random.Random) -> str:
    return f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"


def build_world(size: int, seed: int = 7) -> dict:
    """Memories (text and the entity names they mention), relations, pairs
    to link, and queries with their gold memories."""
    rnd = random.Random(seed)
    n_products, n_events = max(8, size // 60), max(6, size // 90)
    base_words = words(n_products + n_events, rnd)
    memories: list[dict] = []
    relations: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str, str]] = []  # (child or either, parent or other, kind)
    queries: dict[str, list[tuple[str, list[int], list[int]]]] = defaultdict(list)
    types: dict[str, str] = {}

    def add(text: str, *names: str) -> int:
        memories.append({"text": text, "entities": list(names)})
        return len(memories) - 1

    for i in range(n_products):
        w = base_words[i]
        p = f"{w} {NOUNS[i % len(NOUNS)]}"
        platform, lang, license_ = rnd.choice(PLATFORMS), rnd.choice(LANGS), rnd.choice(LICENSES)
        db, db3 = rnd.sample(DBS, 2)
        types[p] = "product"
        m_platform = add(f"{p} runs on {platform}.", p)
        add(f"{p} is written in {lang}.", p)
        m_db = add(f"{p} stores its data in {db}.", p)
        add(f"{person(rnd)} leads the {p} team.", p)
        add(f"{p} is released under the {license_} license.", p)
        family = [i for i in range(len(memories) - 5, len(memories))]
        feats = rnd.sample(FEATURES, 3)
        versions = [f"{p} v{n}" for n in (1, 2, 3)]
        # every other product has its versions written out in words, as people
        # often write them: the extractor still tags the version, but no text
        # says "v1", so only the entity tells the versions apart
        worded = i % 2 == 1
        feature_mem, own = {}, defaultdict(list)
        for n, (v, feat) in enumerate(zip(versions, feats), start=1):
            types[v] = "product"
            said = f"The {ORDINALS[n - 1]} release of {p}" if worded else v
            feature_mem[v] = add(f"{said} added {feat}.", v)
            own[v].append(feature_mem[v])
            own[v].append(add(f"{said} came out in {rnd.choice(MONTHS)} {2021 + n}.", v))
            family += own[v]
            pairs.append((v, p, "version"))
        m_override = add(
            f"With its third release, {p} moved its data to {db3}." if worded
            else f"{versions[2]} stores its data in {db3}.", versions[2])
        own[versions[2]].append(m_override)
        family.append(m_override)
        for a in range(3):
            for b in range(a + 1, 3):
                pairs.append((versions[a], versions[b], "siblings"))
        s = f"{p} sync service"
        types[s] = "code"
        m_part = add(f"The {s} is maintained by {person(rnd)}.", s)
        family += [m_part, add(f"The {s} runs every {rnd.choice([5, 10, 15, 30])} minutes.", s)]
        pairs.append((s, p, "component"))
        tag = "_worded" if worded else ""
        queries["inherit"].append((f"Which platforms does {versions[1]} run on?", [m_platform], []))
        queries["override" + tag].append((f"Where does {versions[2]} store its data?",
                                          [m_override], [m_db]))
        queries["sibling" + tag].append((f"What did {versions[0]} add?", [feature_mem[versions[0]]],
                                         [feature_mem[versions[1]], feature_mem[versions[2]]]))
        queries["rollup"].append((f"What do I know about {p}?", family, []))
        # a version's own memories and what it inherits from its product
        queries["version_rollup"].append((f"What do I know about {versions[2]}?",
                                          own[versions[2]] + family[:5], []))
        queries["part"].append((f"Who maintains the {s}?", [m_part], []))
        if i % 2 == 0:
            bakery = f"{w} Bakery"
            types[bakery] = "organization"
            m_sells = add(f"{bakery} sells {rnd.choice(BREADS)}.", bakery)
            add(f"{bakery} is in {rnd.choice(CITIES)}.", bakery)
            add(f"{bakery} opens at {rnd.choice([6, 7, 8])} a.m.", bakery)
            pairs.append((bakery, p, "namesake"))
            queries["namesake"].append((f"What does {bakery} sell?", [m_sells], family))

    for j in range(n_events):
        w = base_words[n_products + j]
        e = f"{w} {EVENT_NOUNS[j % len(EVENT_NOUNS)]}"
        city, city25 = rnd.sample(CITIES, 2)
        types[e] = "event"
        m_city = add(f"{e} takes place in {city}.", e)
        add(f"{e} is organised by {rnd.choice(ORGS)}.", e)
        add(f"{e} has a track on {rnd.choice(TOPICS)}.", e)
        occ = [f"{e} {y}" for y in (2023, 2024, 2025)]
        attendees = {}
        for o in occ:
            types[o] = "event"
            attendees[o] = add(f"{o} had {rnd.randint(80, 900)} attendees.", o)
            add(f"The keynote at {o} was about {rnd.choice(TOPICS)}.", o)
            pairs.append((o, e, "occurrence"))
        m_moved = add(f"{occ[2]} took place in {city25}.", occ[2])
        for a in range(3):
            for b in range(a + 1, 3):
                pairs.append((occ[a], occ[b], "siblings"))
        queries["event_inherit"].append((f"Where did {occ[1]} take place?", [m_city], [m_moved]))
        queries["event_override"].append((f"Where did {occ[2]} take place?", [m_moved], [m_city]))
        queries["event_sibling"].append((f"How many attendees did {occ[0]} have?",
                                         [attendees[occ[0]]],
                                         [attendees[occ[1]], attendees[occ[2]]]))

    # the families of retrieval_benchmark.py, with their typed relations
    # unique full names: a query names its person the way the memory does
    people = rnd.sample([f"{f} {last}" for f in FIRST for last in LAST],
                        min(len(FIRST) * len(LAST), max(20, size // 40)))
    projects = [f"Project {w}" for w in words(max(8, size // 120), random.Random(seed + 1))]
    tools = ["Postgres", "Redis", "Kafka", "Docker", "Terraform", "Grafana", "Numpy", "Caddy",
             "Nginx", "Deno"]
    project_tool_mems: dict[str, list[int]] = defaultdict(list)
    for pr in projects:
        types[pr] = "project"
        for t in rnd.sample(tools, rnd.choice([1, 2])):
            types[t] = "product"
            project_tool_mems[pr].append(add(f"{pr} uses {t} in production.", pr, t))
            relations.append((pr, "uses", t))
    for who in people:
        types[who] = "person"
        mine = rnd.sample(projects, rnd.choice([1, 2]))
        for pr in mine:
            add(f"{who} works on {pr}.", who, pr)
            relations.append((who, "works_on", pr))
        pref = add(f"{who} prefers {rnd.choice(['dark mode', 'short answers', 'metric units', 'async updates', 'vim keybindings'])}.", who)
        add(f"{who} joined the team and focuses on {rnd.choice(TOPICWORDS)}.", who)
        queries["single_fact"].append((f"What does {who} prefer?", [pref], []))
        gold = [m for pr in mine for m in project_tool_mems[pr]]
        queries["multi_hop"].append((f"What tools does {who} use for their work?", gold, []))
    for pr in rnd.sample(projects, min(len(projects), 20)):
        gold = [k for k, m in enumerate(memories) if pr in m["entities"]]
        queries["by_entity"].append((f"Show everything about {pr}.", gold, []))

    for family, items in queries.items():  # at most 120 a family, drawn at random
        if len(items) > 120:
            queries[family] = random.Random(seed + 2).sample(items, 120)
    while len(memories) < size:
        tw = rnd.choice(TOPICWORDS)
        add(f"Note on {tw}: the {tw} for {rnd.choice(CITIES)} needs attention next sprint.")
    return {"memories": memories, "relations": relations, "pairs": pairs,
            "queries": dict(queries), "types": types}


def build_store(world: dict, embedder: Embedder, links: str, answers: dict, seed: int = 3,
                decider=None):
    """The world in a fresh store, with compared pairs as ``links`` says.
    ``decider`` re-ranks every search when given (Jev in production)."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=embedder,
                        decider=decider)
    ids: dict[str, str] = {}
    for name, entity_type in world["types"].items():
        ids[name] = store.backend.insert_entity(Entity(
            name=name, normalized=name.lower(), entity_type=entity_type, user_id=USER)).id
    vectors = embedder.embed([m["text"] for m in world["memories"]])
    memory_ids = []
    stamp = "2026-06-01T09:00:00+00:00"
    for m, vector in zip(world["memories"], vectors):
        memory = store.backend.insert_memory(
            Memory(content=m["text"], user_id=USER, created_at=stamp, updated_at=stamp,
                   embedding_model=embedder.model_id),
            embedding=vector)
        memory_ids.append(memory.id)
        for name in m["entities"]:
            store.backend.add_mention(EntityMention(entity_id=ids[name], memory_id=memory.id,
                                                    surface=name))
    for subject, predicate, obj in world["relations"]:
        store.backend.add_relation(Relation(subject=ids[subject], predicate=predicate,
                                            object=ids[obj], user_id=USER))
    rnd = random.Random(seed)
    for child, parent, kind in world["pairs"] if links != "none" else []:
        if links == "oracle":
            related = kind in ("version", "occurrence", "component")
            belongs = {"a_kind_of_b": 0.0, "a_part_of_b": 0.0, "b_kind_of_a": 0.0,
                       "b_part_of_a": 0.0, "neither": 0.0 if related else 1.0}
            if related:
                belongs["a_part_of_b" if kind == "component" else "a_kind_of_b"] = 1.0
            same, different = 0.0, 1.0
        else:
            row = rnd.choice(answers[kind])
            belongs, same, different = row["belongs"], row["same"], row["different"]
        store.backend.add_proposal(MergeProposal(
            entity_a=ids[child], entity_b=ids[parent], user_id=USER, confidence=same,
            different=different, belongs=belongs, compared_step=1))
    # as Memry computes them: each memory's entities and what those belong to
    # (at the links just stored) read "it"
    store.refresh_property_vectors(user_id=USER)
    return store, memory_ids


MODES = [
    # (label, relational, mode, depth, fusion)
    ("hybrid", False, "typed", 2, "rescue"),
    ("typed d1", True, "typed", 1, "rescue"),
    ("typed d2 (today)", True, "typed", 2, "rescue"),
    ("undirected d1", True, "undirected", 1, "rescue"),
    ("undirected d2", True, "undirected", 2, "rescue"),
    ("undirected d3", True, "undirected", 3, "rescue"),
    ("directed d1", True, "directed", 1, "rescue"),
    ("directed d2", True, "directed", 2, "rescue"),
    ("directed d3", True, "directed", 3, "rescue"),
    ("undirected d2 weighted", True, "undirected", 2, "weighted"),
    ("directed d1 weighted", True, "directed", 1, "weighted"),
    ("directed d2 weighted", True, "directed", 2, "weighted"),
    ("directed d3 weighted", True, "directed", 3, "weighted"),
    ("directed d1 inherit", True, "directed", 1, "inherit"),
    ("directed d1 gated", True, "directed", 1, "gated"),
    ("directed d2 gated", True, "directed", 2, "gated"),
    ("linked k1", True, "directed", 1, "linked", 1.0),
    ("linked k2", True, "directed", 1, "linked", 2.0),
    ("linked k3", True, "directed", 1, "linked", 3.0),
    # the same, with the decision provider judging relevance (--rerank off,
    # --jev on): its answer is a probability, so no sharpening
    ("linked jev", True, "directed", 1, "linked", 1.0, "jev"),
]


def score(store: MemoryStore, memory_ids: list[str], queries: dict, mode) -> dict:
    label, relational, rmode, depth, fusion = mode[:5]
    cfg = store.config.retrieval
    cfg.relational_mode, cfg.relational_depth, cfg.relational_fusion = rmode, depth, fusion
    if len(mode) > 5:
        cfg.relational_sharpness = mode[5]
    cfg.relational_relevance = mode[6] if len(mode) > 6 else "vector"
    out: dict[str, dict[str, float]] = {}
    for family, items in queries.items():
        mrr, recall, wrong_first, ms = [], [], [], []
        for query, gold, wrong in items:
            started = time.perf_counter()
            got = [r.memory.id for r in store.search(query, user_id=USER, limit=10,
                                                     relational=relational)]
            ms.append((time.perf_counter() - started) * 1000)
            gold_ids = {memory_ids[k] for k in gold}
            wrong_ids = {memory_ids[k] for k in wrong}
            first_gold = next((r for r, m in enumerate(got) if m in gold_ids), None)
            mrr.append(1.0 / (first_gold + 1) if first_gold is not None else 0.0)
            recall.append(len(gold_ids & set(got)) / min(len(gold_ids), 10))
            if wrong_ids:
                first_wrong = next((r for r, m in enumerate(got) if m in wrong_ids), None)
                wrong_first.append(first_wrong is not None and (
                    first_gold is None or first_wrong < first_gold))
        out[family] = {"mrr": statistics.mean(mrr), "recall": statistics.mean(recall),
                       "wrong_first": statistics.mean(wrong_first) if wrong_first else None,
                       "ms": statistics.median(ms), "n": len(items)}
    return out


#: Families scored by recall@10 (many memories answer them); the rest by MRR.
RECALL = {"rollup": "recall", "version_rollup": "recall", "multi_hop": "recall",
          "by_entity": "recall"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--modes", nargs="*", default=None,
                        help="only these mode labels (all by default)")
    parser.add_argument("--jev", action="store_true",
                        help="give the store Jev as its decision provider without re-ranking "
                             "(TYPESAFE_API_KEY), for the linked jev mode")
    parser.add_argument("--links", nargs="*", default=["none", "oracle", "measured"],
                        help="which compared pairs to build stores with")
    parser.add_argument("--rerank", action="store_true",
                        help="Jev re-ranks each search (TYPESAFE_API_KEY), on the families "
                             "where the text ranking and the links disagree, fewer modes")
    args = parser.parse_args()
    decider = None
    modes = [m for m in MODES if not args.modes or m[0] in args.modes]
    def jev_judge():
        """A fresh Jev decider per store (closing a store closes its decider)
        that counts its calls and stops the run after three failed ones, so a
        search never falls back to vectors unnoticed."""
        from memry.config import DecisionConfig
        from memry.providers.decisions import JevDecider

        judge = JevDecider(DecisionConfig(provider="jev", api_key=os.environ["TYPESAFE_API_KEY"]))
        judge.reranks_by_default = False
        judge.calls, judge.failed = 0, 0
        ask = judge.decide

        def counted(state, questions):
            judge.calls += 1
            answers = ask(state, questions)
            if not any(answers[key].available for key in questions):
                judge.failed += 1
                if judge.failed >= 3:
                    print("stopped: Jev is not answering", flush=True)
                    os._exit(2)
            return answers

        judge.decide = counted
        return judge
    if args.rerank:
        from memry.config import DecisionConfig
        from memry.providers.decisions import JevDecider

        decider = JevDecider(DecisionConfig(provider="jev", rerank=True,
                                            api_key=os.environ["TYPESAFE_API_KEY"]))
        modes = [m for m in MODES if m[0] in (
            "hybrid", "typed d2 (today)", "directed d1 weighted", "directed d1 inherit")]
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        base: Embedder = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key=key))
        sizes = args.sizes or [2000, 10000]
    else:
        base = HashEmbedder(128)
        sizes = args.sizes or [1500, 6000]
    cache = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"memry_relative_{base.name}.json"
    embedder = CachedEmbedder(base, cache)
    answers = json.loads((HERE / "datasets" / "belongs_answers.json").read_text())["answers"]
    results = {}
    for size in sizes:
        world = build_world(size)
        texts = [m["text"] for m in world["memories"]]
        texts += [q for items in world["queries"].values() for q, _, _ in items]
        embedder.warm(texts)
        if args.rerank or args.jev:  # 25 queries a family: a Jev call each
            world["queries"] = {f: q[:25] for f, q in world["queries"].items()}
        print(f"\n===== {len(world['memories'])} memories, embedder {embedder.model_id} =====",
              flush=True)
        for links in args.links:
            if args.jev:
                decider = jev_judge()
            store, memory_ids = build_store(world, embedder, links, answers, decider=decider)
            for mode in modes:
                if links == "none" and mode[2] != "typed":
                    continue  # without compared pairs the link modes see only relations
                if links != "none" and mode[2] == "typed" and mode[0] != "hybrid":
                    continue  # "typed" reads no compared pairs: same as without links
                if links != "none" and mode[0] == "hybrid":
                    continue
                calls_before = getattr(decider, "calls", 0)
                res = score(store, memory_ids, world["queries"], mode)
                asked = sum(v["n"] for v in res.values())
                res["_jev_calls_per_search"] = (getattr(decider, "calls", 0) - calls_before) / asked
                results[f"{size}|{links}|{mode[0]}"] = res
                print(f"{links:9} {mode[0]:24} " + "  ".join(
                    f"{f[:12]} {v[RECALL.get(f, 'mrr')]:.2f}"
                    + (f"/{v['wrong_first']:.2f}" if v["wrong_first"] is not None else "")
                    for f, v in res.items() if not f.startswith("_"))
                    + f"  jev/search {res['_jev_calls_per_search']:.2f}", flush=True)
            store.close()
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
