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
| **Tag** | A classification and filter: an entity of type `topic`, one per user and name, mentioned by every memory filed under it. | `entities` + `entity_mentions`; the `categories` column and the `topics`/`memory_topics` filter index | extraction or user |

Two important properties of a **Memory**:

- **Bi-temporal.** Each memory has `valid_from`, `invalid_at`, and
  `superseded_by`. Nothing is ever hard-deleted by the system: a contradicted or
  forgotten memory is *invalidated* (kept, marked no longer valid) and optionally
  points at the memory that replaced it. Every change is also written to
  `memory_events` as an audit trail.
- **Derived, with provenance.** A memory links back to the episode(s) it came
  from (`source_episode_ids`), so you can always re-run a better extraction over
  the original text. A search returns each memory with the episodes it rests on
  (`evidence`: the day each was said, the speaker, the text), and the context
  block lists them under "What was said".
- **The day it was said.** A memory's `created_at` is the time it was saved,
  or the `said_at` day a caller gives for content said on another day.
  Relative times in the text ("last Friday", "next month") count from that day.
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
  recency ranking. Tagging, relation backfill, and re-embedding
  update the row with `touch=False` and leave `updated_at` alone. `created_at`
  never changes after creation.

## The write path (what happens on `save`)

```
message ─▶ episode (verbatim, immutable)
        ─▶ extract_facts (LLM)  →  candidate facts, each naming the lines it rests on
        ─▶ for each fact: reconcile against similar existing memories
                             NEW / SAME / MORE / CHANGED / WRONG
        ─▶ store the memory (with embedding + categories)
        ─▶ link entities  (resolve_mentions, conservative disambiguation)
        ─▶ extract relations between those entities  (typed edges)
        ─▶ coverage audit  (verify_coverage → warnings for dropped details)
```

**Reconciliation** is the step that keeps the store from bloating. Each new fact
is compared to the most similar existing memories, each shown with the day it
was said, and the decision model gives one of five answers:

- **NEW**: new information, including a second event of the same kind (two yoga
  classes stay two). Memry adds a new memory.
- **SAME**: the memory already holds everything in the fact. Memry stores no
  second copy and records the save as more evidence for the memory.
- **MORE**: the fact has a detail the memory lacks, and the memory is still
  true. Memry writes one text of both, dated at the save, and keeps the old
  memory as history.
- **CHANGED**: the old memory was true and no longer is. Memry adds the new one
  and keeps the old one as history.
- **WRONG**: the old memory was never true (a correction). Memry retires it.

A memory kept as history is still found by search, shown with the day it was
said and `[until <date>]`, the day it stopped holding. When the old memory is
rated important or was said in two or more saves, or the decision model is
unsure of a CHANGED or a WRONG, Memry keeps both memories in use and lists the
pair under Upkeep for you to decide. Section 4 of
[architecture.md](architecture.md) has the details.

`infer=false` skips extraction and reconciliation entirely and stores the text
verbatim as one memory (the "just save this exactly" path).

### What a client should send when it saves

Memry keeps the saved text as the source turns of the memories it extracts, and
extraction only has what that text says. Clients should send what was said in
words close to the original, one statement per line. If a client sends a
summary, a later search shows that summary as the source, and the feelings,
advice, event details or photo descriptions it dropped are lost.

Over MCP, `save_memories` stores its `content` as one turn by the user, so a
client writes the name of anyone else who spoke into the text ("Ada: I got the
job"). Over REST, `POST /api/v1/memories` also takes a `messages` list, one turn
each, where a `role` other than a chat role (`user`, `assistant` and the like)
or a `name` field is the speaker's name. Extraction then names each person as
the conversation does, and writes "the user" only for an unnamed speaker in the
role `user`.

For content said on another day, such as an import or an earlier conversation,
the client passes `said_at` (`YYYY-MM-DD`, or an ISO date and time, read in
UTC). Memry dates the save and its memories that day, and "yesterday" or "next
month" in the text counts from it. A value that is not an ISO date, or a day
after today, is refused.

When something changes or the user corrects a fact, the client saves the new
statement as it was said, and Memry keeps or retires the old memory.
`update_memory` rewrites a memory in place and dates it today, which suits a
memory Memry wrote wrong. `delete_memory` forgets a memory when the user asks
for that.

For MCP saves with `infer=true`, the raw text remains immediately searchable
while enrichment waits for two minutes of quiet. Related calls in the same
user/agent/run scope, with the same optional `context` label and given the same
`said_at` day, are extracted together. Clients should send related statements in
one call. Optional `tags` are classification hints, and each tag becomes a topic
in the user's tag list. The `context` label is also shown to the decision model
when it compares two names that may be one person or thing.

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

The extractor assigns one per fact. Nothing forgets a memory for its age (the
forgetting sweep was retired in 0.2.44), so the type no longer shapes how a memory
fades; `half_life_by_type` in `DecayConfig` only feeds the library function
`decay.effective_importance`, which nothing in the product calls:

- **semantic**: a stable fact or preference ("Ada lives in Berlin").
- **episodic**: a dated event or plan ("Ada launched Helios on 2026-03-01").
- **procedural**: a how-to or workflow rule ("always send Ada currency in EUR").
- **working**: short-lived scratch.

The type is not shown in the context
block: a memory reads `[happened 2023-05-07] <text> (said 8 May 2023)`, with the
day its event happens where known and the day it was said
(`context.memory_lines`). It does not (yet) change ranking.

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
are written from the same tags. Two tags are a merge proposal like two names, compared by
the tag question (each shown with its 10 most recent memories); a tag and a named thing of
the same name ("bildy" and the product Bildy) by the entity pair question. A tag is never a hub and never what a
search is about.

- The dashboard's Upkeep > Entities lists tags with the people and things, filtered by
  type (a tag reads "tag"), with counts, and supports rename, combine, and delete; the map
  draws tags once their type is turned on, and the memory list's About filter picks any
  of them. A memory card shows its tags as chips beside the people and things it
  mentions, each chip with its type, and a click on one filters by it.
- Separator and conservative singular/plural duplicates such as `food`/`foods` merge
  automatically. Other tags that may be one (`qa`/`quality assurance`) are merge proposals:
  a calibrated judge decides them, and without one they wait under Upkeep with the other
  pairs. Distinct related topics remain separate.
- There are no parent tags. A tag filter finds the memories filed under that tag, and a
  question whose answer is a set reads the tags its first answers share.
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
     tags (topic entities)    ← cross-cutting filters
```

- **Memories** are the atoms; **episodes** are what they came from.
- **Entities + relations** are where retrieval intelligence lives: they turn a
  bag of facts into a graph you can traverse, which is the only thing that makes
  multi-hop questions answerable.
- **Tags** cut across memories as filters, and are things a memory is about.

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
- Upkeep > Entities lists tags with the people and things, to rename, combine or delete;
  prefer specific topics.
- Nothing the system does destroys data: forgetting is invalidation, and every
  mutation is in `memory_events`.

## What is real vs roadmap (as of this writing)

| Capability | Status |
|---|---|
| Episodes, memories, bi-temporal, audit trail | real |
| Extraction + reconciliation (NEW/SAME/MORE/CHANGED/WRONG) | real |
| Hybrid retrieval (vector + BM25 + recency/importance) | real |
| Entity extraction + conservative disambiguation + merge proposals | real |
| Typed relations + the linked search | real |
| Tags as topic entities, canonicalization | real |
| Entity types (person/project/place/…) + typing backfill | real |
| Forgetting by age (the decay sweep) | retired in 0.2.44 |
| Unified Upkeep area: what needs you, entity hubs with their relations, tags, and the archive of what was removed | real |
| Memory-type effect on *ranking* | not yet |
