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
| **Entity** | A stable referent hub with aliases, a derived description, and linked evidence. | `entities` + `entity_mentions` | entity linking on save; description on first use |
| **Relation** | A typed edge between two entities (`Ada -works_on-> Helios`). | `relations` table | relation extraction on save |
| **Topic** | A scoped classification and filter, with optional parent/child hierarchy. | `topics` + `memory_topics` + `topic_relations` | extraction, user, or abstraction |
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
- **`updated_at` tracks content, not housekeeping.** It moves only on a genuine
  content change (a user edit, or a reconciliation UPDATE), because it drives
  recency ranking and decay *age*. Tagging, relation backfill, and re-embedding
  update the row with `touch=False` and leave `updated_at` alone. `created_at`
  never changes after creation.

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

## Memory types: semantic / episodic / procedural / working

The extractor assigns one per fact, and the type now **shapes how fast a memory
fades** (via `half_life_by_type` in `DecayConfig`):

- **semantic** — a stable fact or preference ("Ada lives in Berlin"). Base rate.
- **episodic** — a dated event or plan ("Ada launched Helios on 2026-03-01").
  Fades about twice as fast: events lose relevance as they age.
- **procedural** — a how-to or workflow rule ("always send Ada currency in EUR").
  Persists about three times as long: rules should stick.
- **working** — short-lived scratch; fades fastest.

So over time an old dated event decays out of retrieval sooner than a standing
rule, even at equal starting importance. The type is also shown in the context
block label (`[procedural · 2026-…]`). It does not (yet) change ranking within a
single query, only how importance decays with age.

## Entities and their types

Entities are **extracted, disambiguated, and typed.**

- On save, the extractor lists each entity in a fact with a `type` (person,
  organization, project, product, place, event, concept, other), so typing costs
  no extra call. `resolve_mentions` reuses an existing entity when the model is
  confident or when an exact multi-part name has meaningful contextual overlap.
  A shared short name or full name without supporting context stays separate and
  creates a **merge proposal** that can be confirmed or rejected.
- Entities linked before typing existed can be classified with
  **`memry backfill-entity-types`** (or `POST /api/v1/entities/backfill-types`) -
  batched, so a whole namespace is a handful of calls, and only untyped entities
  are touched.
- See them under **Knowledge -> People and things**, grouped by type with aliases,
  descriptions, active evidence, and merge controls. Relations have their own Knowledge
  tab. The same data is available through **`GET /api/v1/entities`** and
  `/api/v1/relations`.

## Topics: indexed classification, not identity

Public APIs still call the topic list `categories` for compatibility. Internally, topics are
canonical scoped rows linked to memories through an indexed many-to-many table. They never
enter entity disambiguation: `health` is a classification, while `Jonas` may refer to several
people.

- The **Topics** tab in the dashboard Knowledge area lists topics A-to-Z with counts and
  supports rename, combine, and delete operations.
- Separator and conservative singular/plural duplicates such as `food`/`foods` merge
  automatically. "Suggest merges" proposes semantic synonyms for review; distinct related
  topics remain separate.
- Synthetic abstraction creates hierarchy edges such as `health` broader than
  `liver health`. The parent is not copied onto the child memories. Filtering by `health`
  expands through the hierarchy at query time. It is off by default and meant for
  browsing: a filter that names the specific tag retrieves better than its parent.
- Consolidation merges memories that record the same fact more than once. Grouping is
  geometric over the stored vectors; the merge itself is judged by an LLM and written to
  preserve every detail. Originals are superseded, never deleted. Review it under
  Knowledge > Upkeep before applying.
- Collections remain separate generated maps over memory clusters.
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
     topic links              ← indexed cross-cutting filters and hierarchy
```

- **Memories** are the atoms; **episodes** are what they came from.
- **Entities + relations** are where retrieval intelligence lives: they turn a
  bag of facts into a graph you can traverse, which is the only thing that makes
  multi-hop questions answerable.
- **Topics** cut across memories as indexed filters; hierarchy provides abstraction without copying labels.
- **Collections** and synthetic topic parents are optional maps on top, not places
  facts live.

## Keeping it manageable

- Reconciliation already prevents duplicate *facts* on write.
- A weekly **maintenance autorun** de-duplicates entities and mechanical topic variants.
  It follows prior merge chains, auto-confirms deterministic full-name/context matches,
  and re-judges remaining open proposals (`dedup_entities`, on by default; bounded by the
  number of open proposals). Trigger entity resolution with `POST /api/v1/entities/resolve`.
- If a past run ever bumped dates, **`memry repair-dates`** recomputes every
  `updated_at` from the audit trail (token-free, idempotent).
- Run **`memry backfill-relations`** once to extract relations from memories that
  predate the feature (cheap: only multi-entity memories, marked done so re-runs
  are free).
- Use **Knowledge → Topics** and conservative "Suggest merges" to keep the
  classification vocabulary clean; prefer specific topics.
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
| Normalized topics, hierarchy expansion, canonicalization, collections | real (abstraction/collections opt-in) |
| Entity types (person/project/place/…) + typing backfill | real |
| Memory-type-driven decay (episodic fades, procedural persists) | real |
| Unified Knowledge dashboard for topics, entity hubs, relations, and collections | real |
| Memory-type effect on *ranking* (not just decay) | not yet |
