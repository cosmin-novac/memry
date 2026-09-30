# How Memry works: the pieces and how they fit

This describes exactly what Memry stores, how each piece is created, and how they
combine on read, as the code actually behaves today. Where something is a label
rather than a behaviour, or infrastructure that exists but is not yet populated,
this document says so plainly, so you can tell the real structure from the
roadmap.

## The objects

Memry keeps five kinds of things. Only the first two are core; the rest are
indexes and organization layered on top.

| Object | What it is | Where it lives | Created by |
|---|---|---|---|
| **Episode** | A raw message, stored verbatim, immutable. The source of truth. | `episodes` table | every `add()`; never edited |
| **Memory** | One distilled, self-contained fact/event/statement. Bi-temporal. | `memories` table | extraction + reconciliation |
| **Entity** | A stable referent hub with aliases, a derived description, and linked evidence. | `entities` + `entity_mentions` | entity linking on save; description on first use |
| **Relation** | A typed edge between two entities (`Ada -works_on-> Helios`). | `relations` table | relation extraction on save |
| **Tag** | A classification and filter: an entity of type `topic`, one per user and name, mentioned by every memory filed under it. | `entities` + `entity_mentions`; the `categories` column and the `topics`/`memory_topics` filter index; `topic_relations` for optional parents | extraction, user, or abstraction |

Two important properties of a **Memory**:

- **Bi-temporal.** Each memory has `valid_from`, `invalid_at`, and
  `superseded_by`. Nothing is ever hard-deleted by the system: a contradicted or
  forgotten memory is *invalidated* (kept, marked no longer valid) and optionally
  points at the memory that replaced it. Every change is also written to
  `memory_events` as an audit trail.
- **Derived, with provenance.** A memory links back to the episode(s) it came
  from (`source_episode_ids`), so you can always re-run a better extraction over
  the original text.
- **A "when" is separate from the record's own dates.** A memory whose fact
  happens at a time carries `metadata["when"]` with a `start` (`YYYY-MM-DD`,
  `YYYY-MM-DDTHH:MM`, or `--MM-DD` for a yearly date), an optional `end`, and an
  optional `recurrence`. Extraction sets it only for something that happened or
  will happen (a meeting, a launch, a trip, a deadline), never for a state, a
  price or a log, even when those carry a date. `when_since`/`when_until` search
  on it, so "what is on this weekend" reaches events rather than everything
  saved that weekend.
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

For MCP saves with `infer=true`, the raw text remains immediately searchable
while enrichment waits for two minutes of quiet. Related calls in the same
user/agent/run scope and with the same optional `context` label are extracted
together. Clients should preferably send related facts in one concise multiline
call; optional `tags` help classification but do not define the ingestion group.

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

2. **The linked search** — for questions whose answer shares no words with the
   query ("what tool does Ada use for work?", answered by a memory naming neither
   "Ada" nor "tool"). The query's entities are detected, their relations and
   version and part links are followed (directed and weighted, one link deep by
   default), and every candidate, the text ranking's and the linked entities'
   best, is ordered by how well it states the property asked times how strongly
   it is about the entity named. With Jev, the decision model then judges the
   first 20 of every search in one call, whether it names anything or not, and a
   question whose answer is a set gets one more call (see architecture.md, read
   path, for the stages every search runs in order).

3. **Filters** — an optional `categories` (tag) or entity filter and a
   `since`/`until` date window, applied to every candidate before anything is
   ordered or judged. An empty query with just a tag or date *browses* instead
   of ranking.

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
rule, even at equal starting importance. The type is not shown in the context
block: a memory reads `[happened 2023-05-07] <text> (said 8 May 2023)`, with the
day its event happens where known and the day it was said
(`context.memory_lines`). It does not (yet) change ranking within a single
query, only how importance decays with age.

## Entities and their types

Entities are **extracted, disambiguated, and typed.**

- On save, the extractor lists each entity in a fact with a `type` (person,
  organization, project, product, place, event, document, code, concept,
  other), so typing costs no extra call. The type keeps unrelated same-named
  things apart during disambiguation and groups the Knowledge list; it does
  **not** influence search ranking (see architecture.md, "Entity types and what
  they actually do"). `resolve_mentions` reuses an existing entity when the model is
  confident or when an exact multi-part name has meaningful contextual overlap.
  A shared short name or full name without supporting context stays separate and
  creates a **merge proposal** that can be confirmed or rejected.
- Entities linked before typing existed can be classified with
  **`memry backfill-entity-types`** (or `POST /api/v1/entities/backfill-types`) -
  batched, so a whole namespace is a handful of calls, and only untyped entities
  are touched.
- See them under **Upkeep > Entities**, grouped by type with aliases,
  descriptions, active evidence, merge controls, and the entity's own relations.
  The same data is available through **`GET /api/v1/entities`** and
  `/api/v1/relations`.

## Tags: topic entities

Public APIs still call a memory's tags `categories`. Each tag is an entity of type `topic`,
one per user and normalized name, and every memory filed under it mentions it. The
`categories` column and the legacy `topics`/`memory_topics` index, which the filters read,
are written from the same tags. Two tags merge through the tag question (each shown with
its 10 most recent memories); a tag and a named thing of the same name ("bildy" and the
product Bildy) through the entity pair question. A tag is never a hub and never what a
search is about.

- The **Tags** tab in the dashboard's Upkeep area lists topics A-to-Z with counts and
  supports rename, combine, and delete operations.
- Separator and conservative singular/plural duplicates such as `food`/`foods` merge
  automatically. "Suggest merges" proposes semantic synonyms for review; distinct related
  topics remain separate.
- Synthetic abstraction creates hierarchy edges such as `health` broader than
  `liver health`. The parent is not copied onto the child memories. Filtering by `health`
  expands through the hierarchy at query time. It is off by default and meant for
  browsing: a filter that names the specific tag retrieves better than its parent.
- Entity structure: whether a name is a hub is computed when asked, from its type, its
  memories and relations and the name screen's verdict. The structure pass records each
  part's home (a stated `part_of` relation, or the judge's answer that it is a version or
  a part of another entity; appearing in the same memories gives none) and merges names
  that are the same thing under the same home. It deletes nothing; a removed name is
  retired and can be restored under Upkeep > Archive.
- Consolidation merges memories that record the same fact more than once. Grouping is
  geometric over the stored vectors; the merge itself is judged by an LLM and written to
  preserve every detail. Originals are superseded, never deleted. Word-for-word duplicates
  merge on their own; a merge the LLM proposed waits for a yes under Upkeep.
## How the layers fit together

```
   entities ── relations       ← the retrieval backbone: specific, typed, graph
            │
        memories               ← the atoms: distilled, bi-temporal facts
            │
        episodes               ← immutable source of truth
            │
     tags (topic entities)    ← cross-cutting filters, optional hierarchy
```

- **Memories** are the atoms; **episodes** are what they came from.
- **Entities + relations** are where retrieval intelligence lives: they turn a
  bag of facts into a graph you can traverse, which is the only thing that makes
  multi-hop questions answerable.
- **Tags** cut across memories as filters; hierarchy provides abstraction without copying labels.
- Synthetic tag parents are an optional map on top, off by default, not places
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
- Use **Upkeep > Tags** and conservative "Suggest merges" to keep the
  classification vocabulary clean; prefer specific topics.
- Nothing the system does destroys data: forgetting is invalidation, and every
  mutation is in `memory_events`.

## What is real vs roadmap (as of this writing)

| Capability | Status |
|---|---|
| Episodes, memories, bi-temporal, audit trail | real |
| Extraction + reconciliation (ADD/UPDATE/SUPERSEDE/NONE) | real |
| Hybrid retrieval (vector + BM25 + recency/importance) | real |
| Entity extraction + conservative disambiguation + merge proposals | real |
| Typed relations + the linked search | real |
| Tags as topic entities, hierarchy expansion, canonicalization | real (abstraction opt-in) |
| Entity types (person/project/place/…) + typing backfill | real |
| Memory-type-driven decay (episodic fades, procedural persists) | real |
| Unified Upkeep area: what needs you, entity hubs with their relations, tags, and the archive of what was removed | real |
| Memory-type effect on *ranking* (in addition to decay) | not yet |
