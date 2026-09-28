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
        # the same questions in words the answers do not use
        queries["inherit_para"].append(
            (f"Which operating systems does {versions[1]} support?", [m_platform], []))
        queries["override_para"].append(
            (f"What database does {versions[2]} use?", [m_override], [m_db]))
        queries["rollup"].append((f"What do I know about {p}?", family, []))
        # a version's own memories and what it inherits from its product
        queries["version_rollup"].append((f"What do I know about {versions[2]}?",
                                          own[versions[2]] + family[:5], []))
        queries["part"].append((f"Who maintains the {s}?", [m_part], []))
        queries["part_para"].append((f"Who is in charge of the {s}?", [m_part], []))
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
        queries["event_inherit_para"].append((f"Where was {occ[1]} held?", [m_city], [m_moved]))
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
        queries["single_fact_para"].append((f"What does {who} like?", [pref], []))
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


# --------------------------------------------------------------------------
# A denser world: entities with 15 to 50 memories, most about something other
# than what a question asks, some close to it. The answer has to be picked
# out of its entity's memories, not merely reached through the links.

HOSTS = ["Fly.io", "Hetzner", "AWS", "Render", "DigitalOcean"]
CI_OS = ["Ubuntu", "Debian", "Alpine", "Fedora"]
CACHES = ["Redis", "Memcached", "Valkey"]
SEARCHES = ["Meilisearch", "Typesense", "Elasticsearch", "Tantivy"]
EXPORTS = ["CSV", "JSON", "Markdown", "XLSX"]
INTEGRATIONS = ["Slack", "Notion", "Google Drive", "Dropbox", "Zapier"]
DOCS = ["Read the Docs", "GitBook", "its own website"]
AUTHS = ["Auth0", "Clerk", "Keycloak"]
MAILERS = ["Postmark", "Amazon SES", "Mailgun", "Resend"]
REPOS = ["GitHub", "GitLab", "Codeberg"]
TRANSLATIONS = ["Weblate", "Crowdin", "Transifex"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CAUSES = ["a failed migration", "an expired certificate", "a DNS mistake", "a full disk"]
BUGS = ["the sync conflicts", "a crash on startup", "slow search", "broken PDF links",
        "a memory leak", "wrong time zones"]
OLD_FEATURES = ["the old importer", "the legacy theme", "the XML export", "the beta API",
                "the classic editor"]
BROKE = ["printing", "the Windows installer", "drag and drop", "the zoom setting"]
CHANNELS = ["the blog", "Mastodon", "the newsletter", "Hacker News"]
LOGS = ["Loki", "Papertrail", "CloudWatch"]
QUEUES = ["RabbitMQ", "NATS", "a Postgres table"]
VENUES = ["Old Mill", "Harbour Hall", "Blue Door", "Glasshouse"]
STREAMS = ["YouTube", "Twitch", "PeerTube"]
CATERERS = ["a local co-op", "a food truck", "the venue's kitchen"]
PETS = ["dog", "cat", "parrot", "tortoise"]
PET_NAMES = ["Biscuit", "Luna", "Pixel", "Olive", "Mochi"]
TEAMS = ["platform", "growth", "design", "data", "support"]
PREFS = ["dark mode", "short answers", "metric units", "async updates", "vim keybindings",
         "written summaries", "morning meetings"]
CUSTOMERS = ["a regional bank", "the city council", "a logistics firm", "a hospital group"]
RISKS = ["the data migration", "hiring", "a vendor delay", "unclear scope"]
STATUSES = ["on track", "behind schedule", "paused", "in its final phase"]
LANGUAGES_SPOKEN = ["English", "German", "Spanish", "French", "Hindi", "Japanese"]


def _pick(rnd: random.Random, wordings: list[str], **values) -> str:
    text = rnd.choice(wordings).format(**values)
    return text[0].upper() + text[1:]


def _product_fillers(p: str, rnd: random.Random) -> list[str]:
    """Things said about a product that no question here asks; several sit
    close to one (where it is deployed, what its CI runs on, what it caches)."""
    who = lambda: person(rnd)  # noqa: E731
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["the {p} sync backend is deployed on {v}.",
                    "{p}'s backend servers are hosted on {v}."], p=p, v=rnd.choice(HOSTS)),
        _pick(rnd, ["{p}'s CI builds run on {v} runners.",
                    "the {p} test pipeline runs on {v} runners."], p=p, v=rnd.choice(CI_OS)),
        _pick(rnd, ["{p}'s interface is translated into {v} languages.",
                    "{p} can be used in {v} languages."], p=p, v=n(3, 30)),
        _pick(rnd, ["{p} caches sessions in {v}.",
                    "session data for {p} is cached in {v}."], p=p, v=rnd.choice(CACHES)),
        _pick(rnd, ["{p}'s search index is built with {v}.",
                    "{p} uses {v} for search."], p=p, v=rnd.choice(SEARCHES)),
        _pick(rnd, ["{p} keeps backups for {v} days.",
                    "backups of {p} are kept for {v} days."], p=p, v=n(7, 90)),
        _pick(rnd, ["{p} exports data as {v}.",
                    "users can export their {p} data as {v}."], p=p, v=rnd.choice(EXPORTS)),
        _pick(rnd, ["{v} reviews most {p} pull requests.",
                    "most {p} code reviews are done by {v}."], p=p, v=who()),
        _pick(rnd, ["{v} designed the {p} onboarding.",
                    "the {p} onboarding was designed by {v}."], p=p, v=who()),
        _pick(rnd, ["users asked {p} for {v}.",
                    "several customers requested {v} in {p}."], p=p, v=rnd.choice(FEATURES)),
        _pick(rnd, ["{p} plans to add {v} next quarter.",
                    "{v} is on the {p} roadmap."], p=p, v=rnd.choice(FEATURES)),
        _pick(rnd, ["{p} has about {v} weekly active users.",
                    "around {v} people use {p} every week."], p=p, v=n(200, 90000)),
        _pick(rnd, ["hosting {p} costs {v} euros a month.",
                    "{p}'s monthly hosting bill is {v} euros."], p=p, v=n(10, 900)),
        _pick(rnd, ["{p}'s paid plan costs {v} euros a month.",
                    "a {p} subscription is {v} euros a month."], p=p, v=n(3, 40)),
        _pick(rnd, ["{p} was down for {v} hours in {m} after {c}.",
                    "in {m}, {c} took {p} offline for {v} hours."],
              p=p, v=n(1, 9), m=rnd.choice(MONTHS), c=rnd.choice(CAUSES)),
        _pick(rnd, ["{p} integrates with {v}.",
                    "{p} has a {v} integration."], p=p, v=rnd.choice(INTEGRATIONS)),
        _pick(rnd, ["{p}'s documentation is hosted on {v}.",
                    "the {p} docs live on {v}."], p=p, v=rnd.choice(DOCS)),
        _pick(rnd, ["{p} starts in about {v} seconds.",
                    "{p}'s startup time is around {v} seconds."], p=p, v=n(1, 9)),
        _pick(rnd, ["{p} uses {v} for sign-in.",
                    "sign-in for {p} goes through {v}."], p=p, v=rnd.choice(AUTHS)),
        _pick(rnd, ["{p}'s test suite has {v} tests.",
                    "{p} has {v} automated tests."], p=p, v=n(80, 4000)),
        _pick(rnd, ["{p} sends email through {v}.",
                    "{p}'s emails go out via {v}."], p=p, v=rnd.choice(MAILERS)),
        _pick(rnd, ["{p}'s code is hosted on {v}.",
                    "the {p} repository lives on {v}."], p=p, v=rnd.choice(REPOS)),
        _pick(rnd, ["{p} was started in {v}.",
                    "work on {p} began in {v}."], p=p, v=n(2015, 2023)),
        _pick(rnd, ["{p}'s support inbox gets about {v} emails a week.",
                    "about {v} support emails reach {p} each week."], p=p, v=n(5, 400)),
        _pick(rnd, ["a customer asked whether {p} works without internet.",
                    "someone asked if {p} can be used offline."], p=p),
        _pick(rnd, ["{p}'s API allows {v} requests a minute.",
                    "the {p} API is rate limited to {v} requests a minute."], p=p, v=n(30, 1200)),
        _pick(rnd, ["volunteers translate {p} on {v}.",
                    "{p}'s translations are managed on {v}."], p=p, v=rnd.choice(TRANSLATIONS)),
        _pick(rnd, ["the {p} team meets every {v}.",
                    "{p}'s weekly meeting is on {v}."], p=p, v=rnd.choice(DAYS)),
        _pick(rnd, ["{p}'s logo was redesigned in {v}.",
                    "{p} got a new logo in {v}."], p=p, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{p}'s installer is about {v} MB.",
                    "the {p} download is roughly {v} MB."], p=p, v=n(20, 600)),
        _pick(rnd, ["{p} uses about {v} MB of memory.",
                    "{p} needs around {v} MB of RAM."], p=p, v=n(80, 2000)),
        _pick(rnd, ["{v} wrote the {p} user guide.", "the {p} user guide is by {v}."],
              p=p, v=who()),
        _pick(rnd, ["{p} won a design award in {v}.", "in {v}, {p} won a design award."],
              p=p, v=n(2018, 2025)),
        _pick(rnd, ["{p}'s changelog is published on {v}.", "{p} posts its changelog on {v}."],
              p=p, v=rnd.choice(CHANNELS)),
        _pick(rnd, ["{p} has {v} open issues.", "there are {v} open issues for {p}."],
              p=p, v=n(10, 900)),
        _pick(rnd, ["{p}'s pages load in about {v} ms.", "a {p} page takes about {v} ms to load."],
              p=p, v=n(80, 1500)),
        _pick(rnd, ["{p} was featured in {v}.", "{v} wrote about {p}."],
              p=p, v=rnd.choice(["a tech newsletter", "a podcast", "a design magazine"])),
        _pick(rnd, ["{p}'s free plan allows {v} projects.", "on the free plan, {p} allows {v} projects."],
              p=p, v=n(1, 10)),
        _pick(rnd, ["{p} refunds purchases within {v} days.", "{p}'s refund window is {v} days."],
              p=p, v=rnd.choice([14, 30, 60])),
        _pick(rnd, ["{p} supports single sign-on through SAML.", "{p} offers SAML single sign-on."],
              p=p),
        _pick(rnd, ["{p} runs a user survey every spring.", "every spring {p} runs a user survey."],
              p=p),
        _pick(rnd, ["{p} runs a beta program with {v} members.",
                    "{v} people are in the {p} beta program."], p=p, v=n(20, 3000)),
        _pick(rnd, ["{p} ships a release every {v} weeks.", "a new {p} release comes out every {v} weeks."],
              p=p, v=rnd.choice([2, 4, 6])),
        _pick(rnd, ["{p}'s status page shows its uptime.", "{p} has a public status page."], p=p),
        _pick(rnd, ["{p} is used by about {v} companies.", "about {v} companies use {p}."],
              p=p, v=n(5, 4000)),
        _pick(rnd, ["an accessibility audit of {p} found {v} issues.",
                    "{p}'s accessibility audit listed {v} problems."], p=p, v=n(3, 60)),
        _pick(rnd, ["{p}'s mascot is a {v}.", "{p} has a {v} as its mascot."],
              p=p, v=rnd.choice(["fox", "owl", "whale", "hedgehog"])),
    ]


def _version_fillers(said: str, rnd: random.Random) -> list[str]:
    """Release facts that are not the feature a version added nor where it
    keeps its data; "removed" and "users asked for" sit close to "added"."""
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["{s} fixed {v}.", "{s} fixed {v}, which users had reported for months."],
              s=said, v=rnd.choice(BUGS)),
        _pick(rnd, ["{s} removed {v}.", "{s} dropped {v}."], s=said, v=rnd.choice(OLD_FEATURES)),
        _pick(rnd, ["{s} took {v} weeks to build.", "building {s} took {v} weeks."],
              s=said, v=n(3, 30)),
        _pick(rnd, ["{v} wrote the release notes for {s}.",
                    "the release notes for {s} were written by {v}."], s=said, v=person(rnd)),
        _pick(rnd, ["{s} had {v} beta testers.", "{v} people beta tested {s}."],
              s=said, v=n(10, 900)),
        _pick(rnd, ["{s} was delayed by {v} weeks.", "{s} shipped {v} weeks late."],
              s=said, v=n(1, 8)),
        _pick(rnd, ["after {s}, users asked for {v}.", "users wanted {v} in {s}."],
              s=said, v=rnd.choice(FEATURES)),
        _pick(rnd, ["{s} was downloaded {v} times in its first week.",
                    "{s} had {v} downloads in its first week."], s=said, v=n(100, 50000)),
        _pick(rnd, ["{s} broke {v} for some users.", "some users found that {s} broke {v}."],
              s=said, v=rnd.choice(BROKE)),
        _pick(rnd, ["{s} was announced on {v}.", "the team announced {s} on {v}."],
              s=said, v=rnd.choice(CHANNELS)),
        _pick(rnd, ["{s} has a rating of {v} out of 5.", "reviewers gave {s} {v} out of 5."],
              s=said, v=f"{n(2, 4)}.{n(0, 9)}"),
        _pick(rnd, ["{s} is {v} MB to download.", "the download for {s} is {v} MB."],
              s=said, v=n(20, 600)),
    ]


def _part_fillers(s: str, rnd: random.Random) -> list[str]:
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["the {s} runs every {v} minutes.", "the {s} wakes up every {v} minutes."],
              s=s, v=rnd.choice([5, 10, 15, 30])),
        _pick(rnd, ["the {s} is written in {v}.", "the {s} code is {v}."],
              s=s, v=rnd.choice(LANGS)),
        _pick(rnd, ["the {s} logs to {v}.", "logs from the {s} go to {v}."],
              s=s, v=rnd.choice(LOGS)),
        _pick(rnd, ["the {s} failed {v} times last month.", "last month the {s} failed {v} times."],
              s=s, v=n(1, 9)),
        _pick(rnd, ["the {s} processes about {v} jobs a day.",
                    "about {v} jobs a day go through the {s}."], s=s, v=n(100, 90000)),
        _pick(rnd, ["the {s} was rewritten in {v}.", "the team rewrote the {s} in {v}."],
              s=s, v=n(2018, 2025)),
        _pick(rnd, ["{v} asked to speed up the {s}.", "{v} wants the {s} to be faster."],
              s=s, v=person(rnd)),
        _pick(rnd, ["the {s} retries failed jobs {v} times.",
                    "failed jobs in the {s} are retried {v} times."], s=s, v=n(2, 6)),
        _pick(rnd, ["the {s} queue is kept in {v}.", "the {s} uses {v} as its queue."],
              s=s, v=rnd.choice(QUEUES)),
        _pick(rnd, ["the {s} times out after {v} seconds.",
                    "the {s}'s timeout is {v} seconds."], s=s, v=n(10, 120)),
        _pick(rnd, ["an alert fires when the {s} is {v} minutes behind.",
                    "the {s} pages someone when it falls {v} minutes behind."], s=s, v=n(5, 60)),
        _pick(rnd, ["{v} reported a bug in the {s}.", "{v} found a bug in the {s}."],
              s=s, v=person(rnd)),
        _pick(rnd, ["the {s} has a dashboard in Grafana.", "a Grafana dashboard tracks the {s}."],
              s=s),
        _pick(rnd, ["the {s} uses about {v} MB of memory.", "the {s} needs {v} MB of RAM."],
              s=s, v=n(50, 800)),
    ]


def _series_fillers(e: str, rnd: random.Random) -> list[str]:
    """Facts about an event series; the partner hotel and the afterparty name
    other places than the one it takes place in."""
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["{e} is organised by {v}.", "{v} organises {e}."], e=e, v=rnd.choice(ORGS)),
        _pick(rnd, ["{e} has a track on {v}.", "one {e} track covers {v}."],
              e=e, v=rnd.choice(TOPICS)),
        _pick(rnd, ["{e} tickets cost {v} euros.", "a ticket for {e} is {v} euros."],
              e=e, v=n(20, 400)),
        _pick(rnd, ["{e} has {v} sponsors.", "{v} companies sponsor {e}."], e=e, v=n(2, 30)),
        _pick(rnd, ["the {e} call for papers opens in {v}.", "{e} starts taking talk proposals in {v}."],
              e=e, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{e}'s partner hotel is in {v}.", "most {e} guests stay at a hotel in {v}."],
              e=e, v=rnd.choice(CITIES)),
        _pick(rnd, ["the {e} afterparty is at the {v}.", "{e} ends with a party at the {v}."],
              e=e, v=rnd.choice(VENUES)),
        _pick(rnd, ["{e} is streamed on {v}.", "talks at {e} are streamed on {v}."],
              e=e, v=rnd.choice(STREAMS)),
        _pick(rnd, ["{e}'s mailing list has {v} subscribers.",
                    "{v} people subscribe to the {e} mailing list."], e=e, v=n(100, 20000)),
        _pick(rnd, ["{e} started in {v}.", "the first {e} was in {v}."], e=e, v=n(2005, 2020)),
        _pick(rnd, ["{e} relies on about {v} volunteers.", "about {v} volunteers help at {e}."],
              e=e, v=n(10, 200)),
        _pick(rnd, ["talks at {e} are in {v}.", "{e} talks are given in {v}."],
              e=e, v=rnd.choice(["English", "German", "Spanish"])),
        _pick(rnd, ["{e} lasts {v} days.", "{e} runs for {v} days."], e=e, v=n(1, 4)),
        _pick(rnd, ["{e} usually happens in {v}.", "{e} is usually in {v}."],
              e=e, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{e} offers childcare for attendees.", "there is childcare at {e}."], e=e),
        _pick(rnd, ["{e} gives {v} travel grants a year.", "{v} travel grants are given at {e}."],
              e=e, v=n(2, 40)),
        _pick(rnd, ["lunch at {e} is vegetarian.", "{e} serves vegetarian lunch."], e=e),
        _pick(rnd, ["{v} takes the photos at {e}.", "photos at {e} are taken by {v}."],
              e=e, v=person(rnd)),
        _pick(rnd, ["the main hall at {e} seats {v} people.", "{e}'s main hall has {v} seats."],
              e=e, v=n(100, 2000)),
        _pick(rnd, ["{e} has its own schedule app.", "there is a schedule app for {e}."], e=e),
        _pick(rnd, ["{e}'s badges are made of wood.", "{e} hands out wooden badges."], e=e),
        _pick(rnd, ["{e} has a code of conduct team of {v} people.",
                    "{v} people are on the {e} code of conduct team."], e=e, v=n(2, 9)),
    ]


def _occurrence_fillers(o: str, rnd: random.Random) -> list[str]:
    """Facts about one occurrence; none says where it took place, though one
    counts the countries its speakers came from."""
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["{o} sold out in {v} days.", "tickets for {o} were gone in {v} days."],
              o=o, v=n(1, 60)),
        _pick(rnd, ["the catering at {o} came from {v}.", "{v} did the catering at {o}."],
              o=o, v=rnd.choice(CATERERS)),
        _pick(rnd, ["{o} got a feedback score of {v} out of 5.",
                    "attendees rated {o} {v} out of 5."], o=o, v=f"{n(3, 4)}.{n(0, 9)}"),
        _pick(rnd, ["the budget for {o} was {v} euros.", "{o} cost {v} euros to run."],
              o=o, v=n(5000, 400000)),
        _pick(rnd, ["{o} was in {v}.", "{o} happened in {v}."], o=o, v=rnd.choice(MONTHS)),
        _pick(rnd, ["the best-rated talk at {o} was about {v}.",
                    "attendees liked the talk on {v} at {o} most."], o=o, v=rnd.choice(TOPICS)),
        _pick(rnd, ["speakers at {o} came from {v} countries.",
                    "{o} had speakers from {v} countries."], o=o, v=n(3, 40)),
        _pick(rnd, ["travel to {o} cost the team {v} euros.",
                    "the team spent {v} euros getting to {o}."], o=o, v=n(200, 9000)),
        _pick(rnd, ["it rained during most of {o}.", "{o} had rainy weather."], o=o),
        _pick(rnd, ["talks from {o} were posted on {v}.", "{o} recordings are on {v}."],
              o=o, v=rnd.choice(STREAMS)),
        _pick(rnd, ["{o} had {v} speakers.", "{v} people spoke at {o}."], o=o, v=n(10, 200)),
        _pick(rnd, ["{o} had {v} workshops.", "there were {v} workshops at {o}."],
              o=o, v=n(1, 20)),
    ]


def _person_fillers(who: str, rnd: random.Random) -> list[str]:
    """Facts about a person that are neither a preference nor a tool."""
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["{w} lives in {v}.", "{w} is based in {v}."], w=who, v=rnd.choice(CITIES)),
        _pick(rnd, ["{w} is in the {v} time zone.", "{w} works on {v} time."],
              w=who, v=rnd.choice(["CET", "UTC", "EST", "IST", "JST"])),
        _pick(rnd, ["{w} has a {v} called {x}.", "{w}'s {v} is called {x}."],
              w=who, v=rnd.choice(PETS), x=rnd.choice(PET_NAMES)),
        _pick(rnd, ["{w} is on holiday in {v}.", "{w} takes time off in {v}."],
              w=who, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{w}'s birthday is in {v}.", "{w} was born in {v}."],
              w=who, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{w} is on the {v} team.", "{w} belongs to the {v} team."],
              w=who, v=rnd.choice(TEAMS)),
        _pick(rnd, ["{w} reports to {v}.", "{w}'s manager is {v}."], w=who, v=person(rnd)),
        _pick(rnd, ["{w} speaks {v} and {x}.", "{w} is fluent in {v} and {x}."],
              w=who, **dict(zip("vx", rnd.sample(LANGUAGES_SPOKEN, 2)))),
        _pick(rnd, ["{w} cycles to work.", "{w} takes the train to the office."], w=who),
        _pick(rnd, ["{w} is in the office on {v}s.", "{w} comes in on {v}s."],
              w=who, v=rnd.choice(DAYS)),
        _pick(rnd, ["{w} started in {v} {x}.", "{w} joined in {v} {x}."],
              w=who, v=rnd.choice(MONTHS), x=n(2016, 2025)),
        _pick(rnd, ["{w} mentors {v}.", "{v} is mentored by {w}."], w=who, v=person(rnd)),
        _pick(rnd, ["{w} is on call in week {v}.", "{w} has the on-call shift in week {v}."],
              w=who, v=n(1, 52)),
        _pick(rnd, ["{w} sits on the {v} floor.", "{w}'s desk is on the {v} floor."],
              w=who, v=rnd.choice(["second", "third", "fourth"])),
        _pick(rnd, ["{w} is allergic to {v}.", "{w} cannot eat {v}."],
              w=who, v=rnd.choice(["nuts", "gluten", "shellfish"])),
        _pick(rnd, ["{w} gave a talk on {v} last year.", "{w} spoke about {v} last year."],
              w=who, v=rnd.choice(TOPICS)),
    ]


def _project_fillers(pr: str, rnd: random.Random) -> list[str]:
    """Facts about a project that name no tool."""
    n = lambda a, b: rnd.randint(a, b)  # noqa: E731
    return [
        _pick(rnd, ["{p} is due in {v}.", "the deadline for {p} is in {v}."],
              p=pr, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{p}'s budget is {v} euros.", "{p} has a budget of {v} euros."],
              p=pr, v=n(10, 900) * 1000),
        _pick(rnd, ["{p} is for {v}.", "the client for {p} is {v}."], p=pr, v=rnd.choice(CUSTOMERS)),
        _pick(rnd, ["the {p} stand-up is at {v}.", "{p} has its stand-up at {v}."],
              p=pr, v=rnd.choice(["9:00", "9:30", "10:00"])),
        _pick(rnd, ["{p} kicked off in {v} {x}.", "{p} started in {v} {x}."],
              p=pr, v=rnd.choice(MONTHS), x=n(2023, 2026)),
        _pick(rnd, ["{p} is {v}.", "right now {p} is {v}."], p=pr, v=rnd.choice(STATUSES)),
        _pick(rnd, ["the biggest risk for {p} is {v}.", "{v} is the main risk for {p}."],
              p=pr, v=rnd.choice(RISKS)),
        _pick(rnd, ["{v} is the main stakeholder of {p}.", "{p}'s main stakeholder is {v}."],
              p=pr, v=person(rnd)),
        _pick(rnd, ["the last {p} demo went well.", "the latest demo of {p} went badly."], p=pr),
        _pick(rnd, ["{p} is reviewed every {v} weeks.", "{p} has a review every {v} weeks."],
              p=pr, v=n(2, 6)),
        _pick(rnd, ["{p} has used {v} hours so far.", "{v} hours have gone into {p}."],
              p=pr, v=n(100, 9000)),
        _pick(rnd, ["{p}'s scope was cut in {v}.", "the scope of {p} shrank in {v}."],
              p=pr, v=rnd.choice(MONTHS)),
        _pick(rnd, ["{p} has {v} people.", "{v} people work on {p} full time."], p=pr, v=n(2, 12)),
        _pick(rnd, ["the {p} team sits in {v}.", "{p}'s team is based in {v}."],
              p=pr, v=rnd.choice(CITIES)),
        _pick(rnd, ["the last {p} retrospective was about {v}.",
                    "{p}'s last retro focused on {v}."], p=pr, v=rnd.choice(TOPICWORDS)),
        _pick(rnd, ["the {p} contract ends in {v}.", "{p}'s contract runs until {v}."],
              p=pr, v=n(2026, 2030)),
        _pick(rnd, ["{p} sends a weekly update email.", "a weekly email reports on {p}."], p=pr),
        _pick(rnd, ["{p} tracks {v}.", "the main metric for {p} is {v}."],
              p=pr, v=rnd.choice(["churn", "response time", "sign-ups", "cost per order"])),
    ]


def _some(fillers: list[str], lo: int, hi: int, rnd: random.Random) -> list[str]:
    return rnd.sample(fillers, min(len(fillers), rnd.randint(lo, hi)))


def build_world_dense(size: int, seed: int = 11) -> dict:
    """Like ``build_world``, with entities of realistic size: a product has
    35 to 50 memories, each version 10 to 15, its sync service 11 to 15, an event
    series 19 to 23, each occurrence 10 to 14, a person 14 to 20, a project 15
    to 22 plus its members' "works on"."""
    rnd = random.Random(seed)
    n_products, n_events = max(6, size // 250), max(4, size // 400)
    n_people, n_projects = max(12, size // 100), max(6, size // 250)
    base_words = words(n_products + n_events + 8, rnd)
    memories: list[dict] = []
    relations: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str, str]] = []
    queries: dict[str, list[tuple[str, list[int], list[int]]]] = defaultdict(list)
    types: dict[str, str] = {}

    def add(text: str, *names: str) -> int:
        memories.append({"text": text, "entities": list(names)})
        return len(memories) - 1

    for i in range(n_products):
        w = base_words[i]
        p = f"{w} {NOUNS[i % len(NOUNS)]}"
        types[p] = "product"
        platform, lang, license_ = rnd.choice(PLATFORMS), rnd.choice(LANGS), rnd.choice(LICENSES)
        db, db3 = rnd.sample(DBS, 2)
        m_platform = add(_pick(rnd, ["{p} runs on {v}.", "{p} is available on {v}."],
                               p=p, v=platform), p)
        m_lang = add(_pick(rnd, ["{p} is written in {v}.", "the {p} codebase is in {v}."],
                           p=p, v=lang), p)
        m_db = add(_pick(rnd, ["{p} stores its data in {v}.", "{p} keeps its data in {v}."],
                         p=p, v=db), p)
        m_license = add(_pick(rnd, ["{p} is released under the {v} license.",
                                    "{p} ships under the {v} license."], p=p, v=license_), p)
        m_lead = add(_pick(rnd, ["{v} leads the {p} team.", "the {p} team is led by {v}."],
                           p=p, v=person(rnd)), p)
        thing = [m_platform, m_lang, m_db, m_license, m_lead]
        thing += [add(text, p) for text in _some(_product_fillers(p, rnd), 30, 45, rnd)]
        family = list(thing)
        worded = i % 2 == 1
        versions = [f"{p} v{n}" for n in (1, 2, 3)]
        feats = rnd.sample(FEATURES, 3)
        own: dict[str, list[int]] = defaultdict(list)
        feature_mem = {}
        for n, (v, feat) in enumerate(zip(versions, feats), start=1):
            types[v] = "product"
            said = f"the {ORDINALS[n - 1]} release of {p}" if worded else v
            feature_mem[v] = add(_pick(rnd, ["{s} added {f}.", "{s} introduced {f}."],
                                       s=said, f=feat), v)
            own[v].append(feature_mem[v])
            own[v].append(add(_pick(rnd, ["{s} came out in {m} {y}.", "{s} was released in {m} {y}."],
                                    s=said, m=rnd.choice(MONTHS), y=2021 + n), v))
            own[v] += [add(text, v) for text in _some(_version_fillers(said, rnd), 8, 12, rnd)]
            pairs.append((v, p, "version"))
        m_override = add(
            f"With its third release, {p} moved its data to {db3}." if worded
            else _pick(rnd, ["{v} stores its data in {d}.", "{v} keeps its data in {d}."],
                       v=versions[2], d=db3), versions[2])
        own[versions[2]].append(m_override)
        for v in versions:
            family += own[v]
        for a in range(3):
            for b in range(a + 1, 3):
                pairs.append((versions[a], versions[b], "siblings"))
        s = f"{p} sync service"
        types[s] = "code"
        m_part = add(_pick(rnd, ["the {s} is maintained by {v}.", "{v} maintains the {s}."],
                           s=s, v=person(rnd)), s)
        part = [m_part] + [add(text, s) for text in _some(_part_fillers(s, rnd), 10, 14, rnd)]
        family += part
        pairs.append((s, p, "component"))
        tag = "_worded" if worded else ""
        for v, (question, gold) in zip(rnd.sample(versions, 3), [
                ("Which platforms does {v} run on?", m_platform),
                ("What language is {v} written in?", m_lang),
                ("What license is {v} released under?", m_license)]):
            queries["inherit"].append((question.format(v=v), [gold], []))
        queries["inherit_para"].append(
            (f"Which operating systems does {versions[1]} support?", [m_platform], []))
        queries["inherit_para"].append(
            (f"What programming language is {versions[0]} built with?", [m_lang], []))
        queries["override" + tag].append((f"Where does {versions[2]} store its data?",
                                          [m_override], [m_db]))
        queries["override_para"].append((f"What database does {versions[2]} use?",
                                         [m_override], [m_db]))
        queries["sibling" + tag].append((f"What did {versions[0]} add?", [feature_mem[versions[0]]],
                                         [feature_mem[versions[1]], feature_mem[versions[2]]]))
        queries["thing_fact"].append((f"Who leads the {p} team?", [m_lead], []))
        queries["thing_fact"].append((f"Where does {p} store its data?", [m_db], [m_override]))
        queries["part"].append((f"Who maintains the {s}?", [m_part], []))
        queries["part_para"].append((f"Who is in charge of the {s}?", [m_part], []))
        queries["rollup"].append((f"What do I know about {p}?", family, []))
        queries["version_rollup"].append((f"What do I know about {versions[2]}?",
                                          own[versions[2]] + thing, []))
        if i % 2 == 0:
            bakery = f"{w} Bakery"
            types[bakery] = "organization"
            m_sells = add(f"{bakery} sells {rnd.choice(BREADS)}.", bakery)
            add(f"{bakery} is in {rnd.choice(CITIES)}.", bakery)
            add(f"{bakery} opens at {rnd.choice([6, 7, 8])} a.m.", bakery)
            add(f"{bakery} closes on {rnd.choice(DAYS)}s.", bakery)
            add(f"{bakery} was founded in {rnd.randint(1950, 2015)}.", bakery)
            pairs.append((bakery, p, "namesake"))
            queries["namesake"].append((f"What does {bakery} sell?", [m_sells], family))

    for j in range(n_events):
        w = base_words[n_products + j]
        e = f"{w} {EVENT_NOUNS[j % len(EVENT_NOUNS)]}"
        types[e] = "event"
        city, city25 = rnd.sample(CITIES, 2)
        m_city = add(_pick(rnd, ["{e} takes place in {v}.", "{e} is hosted in {v}."],
                           e=e, v=city), e)
        for text in _some(_series_fillers(e, rnd), 18, 22, rnd):
            add(text, e)
        occ = [f"{e} {y}" for y in (2023, 2024, 2025)]
        attendees = {}
        for o in occ:
            types[o] = "event"
            attendees[o] = add(_pick(rnd, ["{o} had {v} attendees.", "{v} people attended {o}."],
                                     o=o, v=rnd.randint(80, 900)), o)
            add(_pick(rnd, ["the keynote at {o} was about {v}.", "{o} opened with a keynote on {v}."],
                      o=o, v=rnd.choice(TOPICS)), o)
            for text in _some(_occurrence_fillers(o, rnd), 8, 12, rnd):
                add(text, o)
            pairs.append((o, e, "occurrence"))
        m_moved = add(_pick(rnd, ["{o} took place in {v}.", "{o} was hosted in {v}."],
                            o=occ[2], v=city25), occ[2])
        for a in range(3):
            for b in range(a + 1, 3):
                pairs.append((occ[a], occ[b], "siblings"))
        queries["event_inherit"].append((f"Where did {occ[1]} take place?", [m_city], [m_moved]))
        queries["event_inherit_para"].append((f"Where was {occ[1]} held?", [m_city], [m_moved]))
        queries["event_override"].append((f"Where did {occ[2]} take place?", [m_moved], [m_city]))
        queries["event_sibling"].append((f"How many attendees did {occ[0]} have?",
                                         [attendees[occ[0]]],
                                         [attendees[occ[1]], attendees[occ[2]]]))

    people = rnd.sample([f"{f} {last}" for f in FIRST for last in LAST], n_people)
    projects = [f"Project {w}" for w in base_words[n_products + n_events:]][:n_projects]
    if len(projects) < n_projects:
        projects += [f"Project {w}" for w in words(n_projects, random.Random(seed + 1))]
        projects = list(dict.fromkeys(projects))[:n_projects]
    tools = ["Postgres", "Redis", "Kafka", "Docker", "Terraform", "Grafana", "Numpy", "Caddy",
             "Nginx", "Deno"]
    project_tool_mems: dict[str, list[int]] = defaultdict(list)
    for pr in projects:
        types[pr] = "project"
        for t in rnd.sample(tools, rnd.choice([1, 2])):
            types[t] = "product"
            project_tool_mems[pr].append(add(_pick(
                rnd, ["{p} uses {t} in production.", "{p} runs {t} in production."], p=pr, t=t),
                pr, t))
            relations.append((pr, "uses", t))
        for text in _some(_project_fillers(pr, rnd), 14, 18, rnd):
            add(text, pr)
    for who in people:
        types[who] = "person"
        mine = rnd.sample(projects, rnd.choice([1, 2]))
        for pr in mine:
            add(f"{who} works on {pr}.", who, pr)
            relations.append((who, "works_on", pr))
        pref = add(_pick(rnd, ["{w} prefers {v}.", "{w} always asks for {v}."],
                         w=who, v=rnd.choice(PREFS)), who)
        add(f"{who} joined the team and focuses on {rnd.choice(TOPICWORDS)}.", who)
        for text in _some(_person_fillers(who, rnd), 11, 16, rnd):
            add(text, who)
        queries["single_fact"].append((f"What does {who} prefer?", [pref], []))
        queries["single_fact_para"].append((f"What does {who} like?", [pref], []))
        gold = [m for pr in mine for m in project_tool_mems[pr]]
        queries["multi_hop"].append((f"What tools does {who} use for their work?", gold, []))
    for pr in projects:
        gold = [k for k, m in enumerate(memories) if pr in m["entities"]]
        queries["by_entity"].append((f"Show everything about {pr}.", gold, []))

    while len(memories) < size:
        tw = rnd.choice(TOPICWORDS)
        add(f"Note on {tw}: the {tw} for {rnd.choice(CITIES)} needs attention next sprint.")
    return {"memories": memories, "relations": relations, "pairs": pairs,
            "queries": dict(queries), "types": types}


def build_store(world: dict, embedder: Embedder, links: str, answers: dict, seed: int = 3,
                decider=None, property_dimensions: int | None = None):
    """The world in a fresh store, with compared pairs as ``links`` says.
    ``decider`` re-ranks every search when given (Jev in production)."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=embedder,
                        decider=decider)
    store.config.retrieval.property_dimensions = property_dimensions
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
    parser.add_argument("--world", choices=["simple", "dense"], default="simple",
                        help="dense: entities of 15 to 50 memories with near misses")
    parser.add_argument("--out", default=None)
    parser.add_argument("--modes", nargs="*", default=None,
                        help="only these mode labels (all by default)")
    parser.add_argument("--jev", action="store_true",
                        help="give the store Jev as its decision provider without re-ranking "
                             "(TYPESAFE_API_KEY), for the linked jev mode")
    parser.add_argument("--per-family", type=int, default=25,
                        help="questions a family with --jev or --rerank (a Jev call each)")
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
        world = build_world_dense(size) if args.world == "dense" else build_world(size)
        texts = [m["text"] for m in world["memories"]]
        texts += [q for items in world["queries"].values() for q, _, _ in items]
        embedder.warm(texts)
        if args.rerank or args.jev:  # a Jev call a question
            world["queries"] = {f: q[:args.per_family] for f, q in world["queries"].items()}
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
