"""Retrieval benchmark: why Memry organizes memory the way it does.

A controlled, offline-runnable experiment behind the anchors + typed-relations
design. It builds a synthetic store with a known entity/relation structure and
compares retrieval strategies on three query families:

  single_fact : direct lookup (hybrid is enough)
  multi_hop   : answer names none of the query terms (needs relation traversal)
  by_entity   : recall everything about an entity (anchor index is exact)

Findings that shaped the code:
  - hybrid scores a flat zero on multi_hop, at every store size and even with
    real embeddings - the gap is structural, not semantic;
  - naive co-occurrence-graph expansion helps at 1k but decays to zero at scale;
  - typed-relation 2-hop is exact and scale-stable (1.00), PPR is a robust
    relation-free fallback (~0.90) but must never be used for direct lookups.

Run:  python evals/retrieval_benchmark.py        # hash embeddings, offline
      OPENAI_API_KEY=sk-... python evals/retrieval_benchmark.py   # real embeddings
"""

import math
import os
import pathlib
import random
import re
import time
from collections import defaultdict

import numpy as np

from memry.config import EmbeddingConfig
from memry.providers.embeddings import HashEmbedder, OpenAIEmbedder

WORD = re.compile(r"[A-Za-z0-9]+")

# Real embeddings when OPENAI_API_KEY is set, else deterministic hash vectors.
_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if _KEY:
    EMB = OpenAIEmbedder(EmbeddingConfig(provider="openai", api_key=_KEY))
    print(f"[embedder] OpenAI {EMB._model}")
    SIZES = (2000, 10000)
else:
    EMB = HashEmbedder(128)
    print("[embedder] hash (deterministic, no API key)")
    SIZES = (1000, 5000, 20000)


FIRST = ["Priya", "Jonas", "Mara", "Wei", "Ada", "Tom", "Lena", "Omar", "Sofia", "Kai",
         "Nina", "Raj", "Elsa", "Yuki", "Cosmin", "Bea", "Ivan", "Zoe", "Hugo", "Mila"]
LAST = ["Nair", "Berg", "Ruiz", "Chen", "Novak", "Diaz", "Kraus", "Sato", "Meyer", "Osei",
        "Popov", "Haas", "Lund", "Ferro", "Blum", "Costa", "Weber", "Reid", "Falk", "Roy"]
PROJECTS = ["Helios", "Nimbus", "Kestrel", "Orion", "Tessera", "Vantage", "Cobalt", "Marlin",
            "Aster", "Pallas", "Quill", "Vireo", "Solace", "Zephyr", "Drift", "Halcyon"]
TOOLS = ["Postgres", "Redis", "Kafka", "Docker", "Terraform", "Grafana", "Rust", "Numpy",
         "Caddy", "usearch", "SQLite", "Starlette", "pgvector", "Nginx", "Bun", "Deno"]
PREFS = ["dark mode", "short answers", "metric units", "tabs over spaces", "morning meetings",
         "async updates", "no jargon", "euros for currency", "vim keybindings", "minimal deps"]
TOPICWORDS = ["roadmap", "budget", "latency", "hiring", "design", "billing", "outage",
              "review", "demo", "migration", "release", "research", "pricing", "support"]


def build_store(n_people, n_projects, n_tools, target_total, seed=7):
    rnd = random.Random(seed)
    people = []
    for i in range(n_people):
        people.append(f"{FIRST[i % len(FIRST)]} {LAST[(i * 7 + 3) % len(LAST)]} #{i}")
    projects = [f"{PROJECTS[i % len(PROJECTS)]}-{i}" for i in range(n_projects)]
    tools = [f"{TOOLS[i % len(TOOLS)]}" for i in range(n_tools)]

    mems = []  # each: dict(id, text, entities(set), gold_key)
    def add(text, ents):
        mems.append({"id": len(mems), "text": text, "ents": set(ents)})
        return len(mems) - 1

    person_projects = {}
    project_tools = {}
    person_pref_mem = {}
    project_uses_mems = defaultdict(list)

    for p in projects:
        ts = rnd.sample(tools, k=rnd.choice([1, 2]))
        project_tools[p] = ts
        for t in ts:
            mid = add(f"Project {p} uses {t} in production.", [p, t])
            project_uses_mems[p].append(mid)
    for person in people:
        js = rnd.sample(projects, k=rnd.choice([1, 2]))
        person_projects[person] = js
        for j in js:
            add(f"{person} works on project {j}.", [person, j])
        pref = rnd.choice(PREFS)
        person_pref_mem[person] = add(f"{person} prefers {pref}.", [person])
        add(f"{person} joined the team and focuses on {rnd.choice(TOPICWORDS)}.", [person])

    # pad with noise so the signal memories are needles in a big haystack
    while len(mems) < target_total:
        tw = rnd.choice(TOPICWORDS)
        who = rnd.choice(people) if rnd.random() < 0.5 else None
        if who:
            add(f"Note on {tw}: {who} flagged a {tw} item for follow-up.", [who])
        else:
            add(f"Note on {tw}: the {tw} needs attention next sprint.", [])

    return {
        "mems": mems, "people": people, "projects": projects,
        "person_projects": person_projects, "project_tools": project_tools,
        "person_pref_mem": person_pref_mem, "project_uses_mems": project_uses_mems,
    }


_CACHE = pathlib.Path(os.environ.get("TMP", ".")) / "memry_emb_cache"
_CACHE.mkdir(exist_ok=True)


def _embed_cached(texts, tag):
    key = _CACHE / f"{tag}_{getattr(EMB, '_model', 'hash')}_{len(texts)}.npy"
    if key.exists():
        return np.load(key)
    vecs = []
    for i in range(0, len(texts), 256):
        vecs.extend(EMB.embed(texts[i:i + 256]))
        print(f"  embedding {min(i+256,len(texts))}/{len(texts)}", flush=True)
    arr = np.array(vecs, dtype=np.float32)
    np.save(key, arr)
    return arr


def index(mems):
    texts = [m["text"] for m in mems]
    E = _embed_cached(texts, "store")
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    # BM25
    docs = [WORD.findall(t.lower()) for t in texts]
    df = defaultdict(int)
    for d in docs:
        for w in set(d):
            df[w] += 1
    N = len(docs)
    idf = {w: math.log(1 + (N - f + 0.5) / (f + 0.5)) for w, f in df.items()}
    dl = np.array([len(d) for d in docs]); avgdl = dl.mean()
    tf = [defaultdict(int) for _ in docs]
    for i, d in enumerate(docs):
        for w in d:
            tf[i][w] += 1
    # entity inverted index + co-occurrence graph
    ent_posting = defaultdict(list)
    cooc = defaultdict(set)
    for m in mems:
        es = list(m["ents"])
        for e in es:
            ent_posting[e].append(m["id"])
        for a in range(len(es)):
            for b in range(a + 1, len(es)):
                cooc[es[a]].add(es[b]); cooc[es[b]].add(es[a])
    return {"E": E, "idf": idf, "dl": dl, "avgdl": avgdl, "tf": tf,
            "ent_posting": ent_posting, "cooc": cooc, "k1": 1.5, "b": 0.75}


def bm25_scores(ix, qtokens, cand):
    out = {}
    for i in cand:
        s = 0.0
        for w in qtokens:
            if w in ix["tf"][i]:
                f = ix["tf"][i][w]
                s += ix["idf"].get(w, 0.0) * f * (ix["k1"] + 1) / (
                    f + ix["k1"] * (1 - ix["b"] + ix["b"] * ix["dl"][i] / ix["avgdl"]))
        out[i] = s
    return out


def rrf(rank_lists, k=60):
    score = defaultdict(float)
    for rl in rank_lists:
        for r, i in enumerate(rl):
            score[i] += 1.0 / (k + r)
    return score


def detect_entities(query, all_entities):
    ql = query.lower()
    return [e for e in all_entities if e.lower() in ql]


WORKS = re.compile(r"^(.+) works on project (.+)\.$")
USES = re.compile(r"^Project (.+) uses (.+) in production\.$")


def extract_typed_edges(mems):
    """Realistic relation extraction from the memory text (not the generator):
    works_on(person->project) and uses(project->memory-id)."""
    works_on = defaultdict(set)      # person -> projects
    project_uses = defaultdict(list)  # project -> memory ids that state a tool
    for m in mems:
        w = WORKS.match(m["text"])
        if w:
            works_on[w.group(1)].add(w.group(2))
        u = USES.match(m["text"])
        if u:
            project_uses[u.group(1)].append(m["id"])
    return works_on, project_uses


def ppr_scores(ix, seeds, hops=3, alpha=0.85, iters=25):
    """Personalized PageRank on the co-occurrence graph, localized to a few hops
    around the seeds so it stays cheap on a huge store."""
    cooc = ix["cooc"]
    nodes, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for e in frontier:
            nxt |= cooc.get(e, set())
        frontier = nxt - nodes
        nodes |= nxt
    nodes = list(nodes)
    idx = {e: i for i, e in enumerate(nodes)}
    r = np.zeros(len(nodes)); s = np.zeros(len(nodes))
    for e in seeds:
        if e in idx:
            s[idx[e]] = 1.0 / len(seeds)
    r[:] = s
    for _ in range(iters):
        nr = (1 - alpha) * s
        for e in nodes:
            neigh = [n for n in cooc.get(e, ()) if n in idx]
            if neigh:
                share = alpha * r[idx[e]] / len(neigh)
                for n in neigh:
                    nr[idx[n]] += share
        r = nr
    return {e: r[idx[e]] for e in nodes}


_QEMB = {}


def q_embed(query):
    if query not in _QEMB:
        v = np.array(EMB.embed([query])[0], dtype=np.float32)
        _QEMB[query] = v / (np.linalg.norm(v) + 1e-9)
    return _QEMB[query]


def hybrid_rank(ix, query, cand):
    q = q_embed(query)
    sims = ix["E"][cand] @ q
    vec_order = [cand[i] for i in np.argsort(-sims)]
    bm = bm25_scores(ix, WORD.findall(query.lower()), cand)
    bm_order = sorted(cand, key=lambda i: -bm[i])
    fused = rrf([vec_order, bm_order])
    return sorted(cand, key=lambda i: -fused[i]), fused


def retrieve(strategy, ix, store, query, qents, limit=10):
    allids = list(range(len(ix["E"])))
    if strategy == "hybrid":
        cand = allids
        order, _ = hybrid_rank(ix, query, cand)
        return order[:limit]
    if strategy == "hybrid+anchor":
        order, fused = hybrid_rank(ix, query, allids)
        boosted = sorted(allids, key=lambda i: -(fused[i] + 0.02 * len(
            store["mems"][i]["ents"] & set(qents))))
        return boosted[:limit]
    if strategy == "graph":
        # 1-hop expansion over the co-occurrence graph, then hybrid-rank the pool
        pool = set()
        expanded = set(qents)
        for e in qents:
            expanded |= ix["cooc"].get(e, set())
        for e in expanded:
            pool.update(ix["ent_posting"].get(e, []))
        if not pool:
            return retrieve("hybrid", ix, store, query, qents, limit)
        cand = list(pool)
        order, _ = hybrid_rank(ix, query, cand)
        return order[:limit]
    if strategy == "graph+hybrid":
        g = retrieve("graph", ix, store, query, qents, limit)
        h = retrieve("hybrid", ix, store, query, qents, limit)
        seen, out = set(), []
        for i in g + h:            # graph first, then fill from hybrid
            if i not in seen:
                seen.add(i); out.append(i)
        return out[:limit]
    if strategy == "ppr":
        if not qents:
            return retrieve("hybrid", ix, store, query, qents, limit)
        mass = ppr_scores(ix, qents)
        # score each memory by summed PPR mass of its entities
        cand = set()
        for e in mass:
            cand.update(ix["ent_posting"].get(e, []))
        cand = list(cand)
        score = {i: sum(mass.get(e, 0.0) for e in store["mems"][i]["ents"]) for i in cand}
        return sorted(cand, key=lambda i: -score[i])[:limit]
    if strategy == "typed2hop":
        # follow typed relations: person -works_on-> project -uses-> memory
        works_on, project_uses = ix["typed"]
        gold = []
        for person in qents:
            for proj in works_on.get(person, ()):  # 1st hop
                gold += project_uses.get(proj, [])   # 2nd hop
        if not gold:  # not a multi-hop-shaped query -> fall back
            return retrieve("hybrid", ix, store, query, qents, limit)
        # rank the typed candidates by hybrid, fill from hybrid if short
        order, _ = hybrid_rank(ix, query, gold) if len(gold) > 1 else (gold, None)
        out = list(dict.fromkeys(order))
        if len(out) < limit:
            for i in retrieve("hybrid", ix, store, query, qents, limit):
                if i not in out:
                    out.append(i)
        return out[:limit]
    raise ValueError(strategy)


def evaluate(store, ix, n_queries=80, seed=11):
    rnd = random.Random(seed)
    people = rnd.sample(store["people"], min(n_queries, len(store["people"])))
    all_ents = list(ix["ent_posting"].keys())
    families = {"single_fact": [], "multi_hop": [], "by_entity": []}
    for person in people:
        families["single_fact"].append(
            (f"What does {person} prefer?", [store["person_pref_mem"][person]]))
        gold = []
        for j in store["person_projects"][person]:
            gold += store["project_uses_mems"][j]
        if gold:
            families["multi_hop"].append(
                (f"What tools does {person} use for their work?", gold))
    for proj in rnd.sample(store["projects"], min(n_queries, len(store["projects"]))):
        gold = [m["id"] for m in store["mems"] if proj in m["ents"]]
        families["by_entity"].append((f"Show everything about project {proj}.", gold))

    strategies = ["hybrid", "graph", "ppr", "typed2hop"]
    results = {}
    for fam, queries in families.items():
        for strat in strategies:
            recalls, mrrs = [], []
            for q, gold in queries:
                qents = detect_entities(q, all_ents)
                got = retrieve(strat, ix, store, q, qents, limit=10)
                goldset = set(gold)
                hit = [i for i in got if i in goldset]
                recalls.append(len(set(got) & goldset) / len(goldset))
                rr = 0.0
                for r, i in enumerate(got):
                    if i in goldset:
                        rr = 1.0 / (r + 1); break
                mrrs.append(rr)
            results[(fam, strat)] = (np.mean(recalls), np.mean(mrrs))
    return results, strategies, list(families)


def latency(store, ix, strat, reps=40, seed=5):
    rnd = random.Random(seed)
    people = rnd.sample(store["people"], min(reps, len(store["people"])))
    all_ents = list(ix["ent_posting"].keys())
    t0 = time.perf_counter()
    for person in people:
        q = f"What tools does {person} use for their work?"
        retrieve(strat, ix, store, q, detect_entities(q, all_ents), limit=10)
    return (time.perf_counter() - t0) / len(people) * 1000  # ms/query


if __name__ == "__main__":
    for total in SIZES:
        store = build_store(n_people=120, n_projects=40, n_tools=16, target_total=total)
        ix = index(store["mems"])
        ix["typed"] = extract_typed_edges(store["mems"])
        results, strategies, fams = evaluate(store, ix)
        print(f"\n===== N = {len(store['mems'])} memories =====")
        header = f"{'family':<12}" + "".join(f"{s:>16}" for s in strategies)
        print(header + "   (recall@10 / MRR)")
        for fam in fams:
            row = f"{fam:<12}"
            for s in strategies:
                rc, mr = results[(fam, s)]
                row += f"{rc:>7.2f}/{mr:<8.2f}"
            print(row)
        lat = {s: latency(store, ix, s) for s in ("hybrid", "graph+hybrid")}
        print(f"latency multi_hop  hybrid={lat['hybrid']:.2f}ms  "
              f"graph+hybrid={lat['graph+hybrid']:.2f}ms/query")
