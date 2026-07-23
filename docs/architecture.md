# Memry architecture - the plain-language edition

This document explains how Memry works, why it's shaped the way it is, and where its
limits are - written to be understandable without reading the code. The competitive
research that motivated the design is in
[research/competitive-analysis.md](research/competitive-analysis.md).

---

## 1. The one-paragraph version

Memry sits between an agent and its conversations. Everything the agent tells it is first
written down verbatim (**episodes**), then distilled by an LLM into small, self-contained
facts (**memories**), each of which is checked against what's already known: duplicates are
skipped, refinements merge, and contradictions retire the old fact and link it to its
replacement. When the agent asks a question, Memry searches those facts by keywords *and*
meaning, prefers recent and important ones, and hands back either a ranked list or a
ready-to-paste context block. Every step leaves a paper trail.

## 2. The layer cake

```
Agent / App  ──── MCP · REST · CLI · Python import
      │
MemoryStore ──── the only public API (add, search, context, update, delete, history…)
      │
Intelligence ─── extraction · reconciliation · decay · context building   ← the "brain"
      │
Retrieval ────── BM25 + vectors, fused and boosted, explainable scores
      │
Providers ────── LLMs (Anthropic/OpenAI/Ollama) · embeddings (OpenAI/Voyage/Ollama/hash)
      │
Backend API ──── storage contract (backends/base.py)
      ├── LocalBackend:    one SQLite file (episodes, memories, FTS index,
      │                    entities, events) + optional usearch ANN sidecar
      ├── PostgresBackend: multi-writer scale-out (pgvector + tsvector)
      └── Mem0Backend:     optional adapter for interop + benchmarking
```

The contract between layers is the point: **everything below `MemoryStore` is swappable**
without callers noticing. That is also the mechanism for moving research into production
(section 7).

## 3. What exactly gets stored

Four kinds of records, all in one SQLite file:

| Record | What it is | Mutability |
|---|---|---|
| **Episode** | A raw message/event, exactly as received | Immutable, kept forever |
| **Memory** | One distilled fact ("User lives in Amsterdam") | Content can be updated; retired by *invalidation*, not deletion |
| **Event** | One audit entry (ADD / UPDATE / SUPERSEDE / DELETE / NONE) | Append-only |
| **FTS row / embedding** | Derived search indexes for a memory | Rebuilt automatically (`memry reindex`) |

A memory carries: content, a **type** (`semantic` = stable fact/preference, `episodic` =
dated event/plan, `procedural` = how-to rule, `working` = reserved for short-lived state),
an **importance** score (0-1, set by the extractor), **categories**, **entities**,
free-form metadata, scope ids (`user_id`/`agent_id`/`run_id`), timestamps, and three
temporal fields that make it *bi-temporal*: `valid_from`, `invalid_at`, `superseded_by`.

### "Do we tag memories?" - yes, three ways

1. **`memory_type`** - the coarse taxonomy above. Stored on every memory, shown in every
   result, and the hook for future "routing" research (send working-memory queries one way,
   procedural another).
2. **`categories`** - free-form labels the extractor assigns ("location", "diet",
   "project-phoenix"). Stored, returned with every result, and **filterable everywhere**:
   `search(categories=["diet"])`, `get_all(categories=...)`, `memry search -c diet`,
   `?categories=diet` on the REST API, and a `categories` argument on the MCP search tool.
   Matching is case-insensitive and pushed down into SQL, not post-filtered.
3. **`entities`** - names of people/places/things mentioned in the fact ("Jonas", "ASML").
   These are now first-class objects, not just strings; see the next section.

### "Do we store unique entities permanently?"

Yes. Every mention creates or attaches to a row in the `entities` table (plus an
`entity_mentions` row linking memory to entity), and entities are never deleted: a merge
sets `merged_into` on the losing entity and repoints its mentions, keeping the full record.

**Disambiguation is deliberately conservative.** A name match is never enough to merge:

- When a new memory mentions "Jonas" and a Jonas already exists in scope, an LLM identity
  judgment compares the new fact against the existing entity's known facts.
- Verdict "same" with high confidence (>= 0.9): the mention attaches to the existing Jonas.
- Verdict "unsure": a **new, separate Jonas** is created, plus a **merge proposal**
  recording the suspicion and its confidence. Three ambiguous Jonases stay three Jonases.
- Verdict "different": a new Jonas, no proposal, nothing to clean up later.
- No LLM configured: same-name mentions always stay separate and a proposal is recorded.

Proposals are the human-in-the-loop hook you asked for: `memry entities proposals` lists
them, `memry entities confirm <id>` merges (the user says "same person"), `memry entities
reject <id>` keeps them apart for good, and `memry entities resolve` re-judges open
proposals with whatever evidence has accumulated since, auto-confirming only clear,
high-confidence matches. The same operations exist on the REST API
(`/api/v1/entities/...`) and the Python API (`store.confirm_merge`, `store.reject_merge`,
`store.merge_entities`).

The wrong-merge asymmetry drives the whole design: a deferred merge costs one later
confirmation; a wrong merge silently corrupts every future recall about both entities.
A `relations` table (subject-predicate-object edges between entities, Mem0-graph/Zep
territory) remains the next step, and the episode log still means any better future
extractor can be replayed **retroactively over the entire history**.

## 4. The write path, step by step

`store.add("I moved to Amsterdam", user_id="ada")`:

1. **Record** - the raw text becomes an Episode. This happens before anything can fail.
2. **Extract** - the LLM turns the conversation into candidate facts with type, importance,
   categories, entities. Relative dates become absolute; pronouns get resolved; secrets are
   excluded by prompt. No LLM configured → the text is stored verbatim instead.
3. **Reconcile** - for each candidate, Memry retrieves the 5 most similar existing memories
   in the same scope and decides:
   - *exact duplicate* → *NONE* (skipped, no LLM call, no cost)
   - *new information* → **ADD**
   - *refines an existing fact* → **UPDATE** (rewritten in place, old text kept in the event)
   - *contradicts an existing fact* → **SUPERSEDE**: the old memory gets `invalid_at` set
     and `superseded_by` pointing at the new one. "User lives in Munich" is still in the
     database, marked as no-longer-true since 09:17:41, with the reason recorded.
4. **Link entities** - each extracted entity mention attaches to an existing entity only
   when identity is clear; otherwise a separate entity plus a merge proposal is created
   (section 3).
5. **Audit** - every decision appends an event. `memry history <id>` replays a memory's life.

Malformed LLM output never corrupts state: unparseable decisions fall back to ADD, and every
JSON response is parsed leniently (code fences, leading prose, etc.).

## 5. The read path

`store.search("where does ada live?", user_id="ada")`:

1. Two candidate lists: **BM25 keyword** matches (FTS5) and **cosine-similarity** matches
   over embeddings (only vectors produced by the currently configured embedding model).
2. **Reciprocal Rank Fusion** merges them - a memory ranked high by either signal scores
   well; one ranked high by both scores best.
3. The fused score is blended: `0.70·relevance + 0.15·recency + 0.15·importance`
   (recency halves every 30 days; all weights configurable). Invalidated memories are
   excluded unless you ask for them.
4. Every result exposes its component signals - so when ranking is wrong, you can see *why*.

`reconstruct_context(...)` runs the same search and greedily packs results into a markdown
block that fits a token budget - the "just give me something to put in the prompt" call.

**Forgetting** is a separate, deliberate act: effective importance decays on a half-life,
and `memry sweep` invalidates (never deletes) memories that decayed below threshold.

## 6. Zero-key mode

Every stage has a keyless fallback: verbatim storage instead of extraction, exact-duplicate
detection instead of LLM reconciliation, and BM25 + deterministic **hash embeddings**
(feature-hashed n-grams - fuzzy lexical similarity, not semantics) instead of embedding
APIs. This isn't a demo mode; on the synthetic eval it reaches recall@5 = 1.0 with 0.3 ms
searches. Keys upgrade quality; they are never required for function.

## 7. Promoting experimental components into production

The system was shaped for exactly this workflow. The seams:

- **`MemoryBackend`** (storage), **provider interfaces** (LLM/embedder), and the
  **intelligence functions** (`extract_facts`, `reconcile_candidate`, `decay`, retrieval
  weights) are all injectable - `MemoryStore(config, backend=…, llm=…, embedder=…)`.
- The **episode log** means new pipelines can be replayed over old data: a better extractor
  can rebuild the memory index from raw history without data loss.
- The **eval harness** is the promotion gate: same datasets, same metrics, swap one
  component.

Recommended loop:

1. An experimental repo `pip install`s memry and imports the interfaces - experiments live
   there as `ExperimentalBackend(MemoryBackend)` or an alternative `extract_facts`, without
   forking.
2. Benchmark against the shipped implementation (and the Mem0 adapter) with
   `memry eval` on shared datasets; keep results with the experiments.
3. When an idea wins, it graduates: merged into `src/memry/` behind a config flag
   (`Config.extraction_strategy = "v2"`), default-off, then default-on once the
   harness says so. The public `MemoryStore` API never changes, so downstream applications
   and every MCP client are untouched.
4. Results stay reproducible, because the production repo pins the exact configs an
   experiment ran.

## 8. Honest assessment - is this state of the art?

**Where it genuinely leads (design-level):** the *combination* is rare - bi-temporal
supersession + per-fact provenance + full audit history are platform-only or absent in
Mem0 OSS; a zero-key local mode is absent in Mem0 and Zep; and no competitor ships a
deterministic eval harness in the box. Architecture-wise this is state of the art for a
self-hosted memory layer.

**Scale and operations (addressed in v0.2):**

- *ANN index.* With `pip install memry[ann]`, vector search runs through a usearch HNSW
  sidecar above a configurable row threshold (below it, exact brute force is faster
  anyway). ANN candidates are always exact-rescored, the sidecar is a rebuildable cache
  (never the source of truth), and restrictive filters fall back to the exact scan so
  recall is never silently sacrificed.
- *Multi-writer backend.* `MEMRY_BACKEND=postgres` + `MEMRY_POSTGRES_DSN` runs the full
  contract (episodes, bi-temporal memories, events, entities, categories) on PostgreSQL
  with pgvector cosine search and tsvector keyword search: real concurrent writers,
  replication, and managed-HA options. SQLite remains the default for single-node.
- *Multi-tenant auth.* Named tenants each get their own API key; the server transparently
  namespaces every tenant request (`acme::u1`), refuses cross-tenant access to memories,
  entities, and proposals (404, no existence leak), and scopes stats per tenant. The admin
  key keeps a global view. The same confinement covers MCP-over-HTTP: tools take their
  namespace from the authenticated principal, never from their own `user_id` argument,
  and every id-addressed store method re-checks ownership behind the same door as the
  data, so a new endpoint or tool cannot forget the check.

**Where it is honestly not (yet):**

- *Unproven extraction quality.* SOTA claims in this field are benchmark claims
  (LoCoMo/LongMemEval). Memry hasn't run them yet; until it does, "state of the art" is a
  design statement, not a measured one. Running those benchmarks is the very next step.
- *No relations graph, no hierarchical summaries, no connectors.* Known roadmap items,
  not accidents. Entities exist and disambiguate; edges between them are next.
- *Auth: config keys, runtime accounts, or OAuth.* Config tenants and admin keys for the
  simple case; runtime accounts (hashed API keys, scrypt passwords) plus a built-in OAuth
  2.1 authorization server (DCR, PKCE, refresh rotation, revocation) for self-service
  multiuser. Still out of scope: per-key rate limits, and delegating login to an external
  IdP (SSO) - Memry is its own authorization server today.

**Fast?** Retrieval: yes, sub-millisecond locally, and with the ANN sidecar the vector
path stays fast into the millions of rows; the latency budget is entirely the LLM calls on
the *write* path (and those are per-save, not per-recall). **Reliable/robust?** For its
intended deployments (one process + one file, or many processes + Postgres), yes: nothing
is ever silently destroyed, every failure path degrades instead of crashing, and 85+ tests
cover the contracts on both backends. The honest one-line verdict: *state-of-the-art
architecture and trust model for self-hosted agent memory, now with the scale/ops story
(ANN, Postgres, multi-tenant) in place; extraction quality still has to be earned on the
public benchmarks.*
