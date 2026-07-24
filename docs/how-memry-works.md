# How Memry works: the pieces and how they fit

This describes exactly what Memry stores, how each piece is created, and how they
combine on read, as the code actually behaves today. Where something is a label
rather than a behaviour, or infrastructure that exists but is not yet populated,
this document says so plainly, so you can tell the real structure from the
roadmap.

## The objects

Memry keeps six kinds of things. Only the first two are core; the rest are
indexes and organization layered on top.

| Object | What it is | Where it lives | Created by |
|---|---|---|---|
| **Episode** | A raw message, stored verbatim, immutable. The source of truth. | `episodes` table | every `add()`; never edited |
| **Memory** | One distilled, self-contained fact/event/statement. Bi-temporal. | `memories` table | extraction + reconciliation |
| **Entity** | A distinct thing a memory mentions (a person, org, project…). | `entities` + `entity_mentions` | entity linking on save |
| **Relation** | A typed edge between two entities (`Ada -works_on-> Helios`). | `relations` table | relation extraction on save |
| **Category (tag)** | A free-form label on a memory. A filter, not structure. | `categories` JSON on each memory | extraction, or you |
| **Collection** | A titled, summarized cluster of memories. Navigation only. | `collections` table | `build_collections` (on demand) |

Two important properties of a **Memory**:

- **Bi-temporal.** Each memory has `valid_from`, `invalid_at`, and
  `superseded_by`. Nothing is ever hard-deleted by the system: a contradicted or
  forgotten memory is *invalidated* (kept, marked no longer valid) and optionally
  points at the memory that replaced it. Every change is also written to
  `memory_events` as an audit trail.
- **Derived, with provenance.** A memory links back to the episode(s) it came
  from (`source_episode_ids`), so you can always re-run a better extraction over
  the original text.

## The write path (what happens on `save`)

```
message ─▶ episode (verbatim, immutable)
        ─▶ extract_facts (LLM)  →  candidate facts
        ─▶ for each fact: reconcile against similar existing memories
                             ADD / UPDATE / DELETE / NONE
        ─▶ store the memory (with embedding + categories)
        ─▶ link entities  (resolve_mentions, conservative disambiguation)
        ─▶ extract relations between those entities  (typed edges)
        ─▶ coverage audit  (verify_coverage → warnings for dropped details)
```

**Reconciliation** is the step that keeps the store from bloating. Each new fact
is compared to the most similar existing memories and the LLM decides:

- **ADD** – genuinely new → a new memory.
- **UPDATE** – refines/corrects an existing one → rewritten in place, and the
  rewrite must preserve every concrete detail from both versions.
- **DELETE** – the old statement is now false → the old memory is invalidated and
  superseded by the new one.
- **NONE** – already known → skipped.

`infer=false` skips extraction and reconciliation entirely and stores the text
verbatim as one memory (the "just save this exactly" path).

## The read path (what happens on `search`)

Retrieval is a **fusion of three moves**, because the experiments (see
`evals/retrieval_benchmark.py`) showed no single index answers every kind of
question:

1. **Hybrid relevance** — the workhorse. A vector k-NN search and a BM25 keyword
   search are combined with Reciprocal Rank Fusion, then each candidate's score
   is adjusted by recency and importance:

   ```
   final = fused_weight·RRF(vector, keyword) + recency_weight·recency + importance_weight·importance
   ```

   This nails direct lookups ("what does Ada prefer?") and "about X" queries.

2. **Relational traversal** — for questions whose answer shares no words with the
   query ("what tool does Ada use for work?", answered by a memory naming neither
   "Ada" nor "tool"). The query's entities are detected, typed relations are
   followed up to two hops, and the reached memories are fused in. When a
   namespace has no typed relations yet, this falls back to localized PageRank
   over entity co-occurrence (no LLM). A **rescue threshold** means relational
   candidates only recover memories hybrid *buried or missed*; they never demote
   a strong direct hit.

3. **Filters** — an optional `categories` (tag) filter and a `since`/`until` date
   window. An empty query with just a tag or date *browses* instead of ranking.

## Memory types: semantic / episodic / procedural

These are **descriptive labels only, today.** The extractor assigns one per fact:

- **semantic** — a stable fact or preference ("Ada lives in Berlin").
- **episodic** — a dated event or plan ("Ada launched Helios on 2026-03-01").
- **procedural** — a how-to or workflow rule ("always send Ada currency in EUR").

The type is stored and shown in the context block label (`[semantic · 2026-…]`),
but it does **not** currently change retrieval ranking, decay, or storage. It is a
lens for you and the agent, not a behaviour. (Making episodic memories decay
faster, or procedural ones rank higher for how-to queries, would be a natural
future use of the field; it does not happen yet.)

## Entities and their types

Entities are **extracted and disambiguated but not yet typed.**

- On save, the extractor lists the entity names in a fact. `resolve_mentions`
  either reuses an existing entity (only when the LLM is confident they are the
  same, to avoid conflating two people named "Jonas") or creates a new one and
  files a **merge proposal** for the ambiguous case, which you confirm or reject.
- The `entity_type` field (person / org / project / place / event) **exists in
  the schema but is always `null`**: nothing populates it today. So entities are
  real and linked, but untyped.
- You can see them now via **`GET /api/v1/entities`** (and one entity with its
  mentions and memories via `/api/v1/entities/{id}`). There is no dashboard view
  yet, and types will read as null until entity typing is built.

## Tags: a filter, not the structure

Tags (`categories`) are free-form labels. They are the **weakest** organizing
primitive and are deliberately kept that way. The rule the design follows:

> Put specificity in tags and entities; put abstraction in a layer *above*, never
> by collapsing the specific away.

Concretely:

- **Canonicalization ("Suggest merges")** only de-duplicates the *same* label
  written differently (`writing preferences` = `writing-preference`, `project` =
  `projects`). It must never merge distinct-but-related tags (`writing-style` and
  `response-style` are different and stay separate). Merging loses a distinction
  permanently, so it is conservative by design; when a tag is ambiguous, the fix
  is to make it *more specific*, not to fold it into a neighbour.
- **The Tag manager** (dashboard, "tags" link) lists every tag A→Z with its count
  and lets you rename, combine true duplicates, or delete a tag across all
  memories.
- **Synthetic tags** and **collections** are the "abstraction above" layer:
  higher-level groupings that sit on top of the specific tags without replacing
  them. Both are opt-in and off by default (synthetic-tag auto-abstraction turned
  out to produce vague labels, so it is disabled; collections are on-demand).

## How the layers fit together

```
        collections            ← coarse navigation (titled clusters, on demand)
            │
   entities ── relations       ← the retrieval backbone: specific, typed, graph
            │
        memories               ← the atoms: distilled, bi-temporal facts
            │
        episodes               ← immutable source of truth
            │
          tags                 ← a clean cross-cutting filter over memories
```

- **Memories** are the atoms; **episodes** are what they came from.
- **Entities + relations** are where retrieval intelligence lives: they turn a
  bag of facts into a graph you can traverse, which is the only thing that makes
  multi-hop questions answerable.
- **Tags** cut across memories as a filter; keep them specific and de-duplicated.
- **Collections** (and synthetic tags) are an optional map on top, not a place
  facts live.

## Keeping it manageable

- Reconciliation already prevents duplicate facts on write.
- Run **`memry backfill-relations`** once to extract relations from memories that
  predate the feature (cheap: only multi-entity memories, marked done so re-runs
  are free).
- Use the **Tag manager** + conservative "Suggest merges" to keep the tag
  vocabulary clean; prefer specific tags.
- Run **`memry build-collections`** on demand when you want a fresh map; it never
  runs on a schedule, so it spends no tokens unasked.
- Nothing the system does destroys data: forgetting is invalidation, and every
  mutation is in `memory_events`.

## What is real vs roadmap (as of this writing)

| Capability | Status |
|---|---|
| Episodes, memories, bi-temporal, audit trail | real |
| Extraction + reconciliation (ADD/UPDATE/SUPERSEDE/NONE) | real |
| Hybrid retrieval (vector + BM25 + recency/importance) | real |
| Entity extraction + conservative disambiguation + merge proposals | real |
| Typed relations + relational retrieval (+ PPR fallback) | real |
| Tag canonicalization, synthetic tags, collections | real (opt-in where noted) |
| **Entity *types*** (person/event/…) | schema exists, **not populated** |
| **Memory-type-driven behaviour** (decay/ranking by type) | **label only, no behaviour** |
| Dashboard views for entities / relations / collections | API only, **no UI yet** |
