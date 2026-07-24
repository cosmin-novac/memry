# Knowledge model plan and implementation record

This plan gives Memry one coherent user-facing "Knowledge" model while keeping
its internal behavior small, evidence-grounded, fast, and cheap. It covers
entities, topics (today's tags/categories), relations, synthetic topics, and
collections.

A reusable one-feature-per-row catalog lives in
[knowledge-model-features.md](knowledge-model-features.md).

The central decision is simple:

> An entity is a named, typed memory hub with aliases, a synthesized description,
> and links to the atomic memories that describe it.

The entity is memory-like to the user, but it is not an ordinary fact-memory.
Its identity is stable; its description is a replaceable cache; linked memories
and their source episodes retain the evidence and history.

## Decisions

1. **One UI concept, separate behavioral paths.** The dashboard presents one
   "Knowledge" area, but topics and entities do not share identity resolution or
   graph-retrieval hot paths.
2. **Keep the entity model minimal.** Extend the existing `Entity` with a
   synthesized `description` and `description_updated_at`. Reuse the existing
   `EntityMention` links. Do not introduce an `EntityProfile`, generic
   `MemoryAnchorLink`, evidence-version object, role taxonomy, or span model now.
3. **Aliases are first-class behavior, not necessarily another table.** Derive
   aliases from observed mention surfaces and names of entities merged into the
   survivor. Allow explicit user aliases through existing entity metadata until
   measured lookup needs justify a dedicated indexed alias table.
4. **Descriptions summarize evidence; they never replace it.** Generate them
   from active linked memories, lazily and in bounded form. Every detail remains
   recoverable through the existing entity-to-memory links.
5. **Normalize topics for indexed filtering.** Move categories behind a topic
   table and an indexed memory-topic join while preserving the current
   `categories` API during migration.
6. **Start topic normalization early.** It is independent of entity-description
   work and has the clearest measured performance upside. A 2026-07-24 run of the checked-in
   50,000-row, 300-query microbenchmark measured 45.7563 ms for legacy JSON scans
   and 0.0109 ms for indexed links on this machine. Treat that as implementation evidence,
   not a public product
   performance claim or regression threshold.
7. **Do not physically merge topic and entity tables unless benchmarks later
   justify it.** Fewer tables are not automatically a simpler or faster system.

## What exists today

Memry already has the core entity-as-hub structure:

- `Entity`: stable ID, canonical name, normalized name, type, scope, metadata,
  merge history, and timestamps.
- `EntityMention`: an entity ID linked to a memory ID, with the observed surface
  text and timestamp.
- `entity_memories(entity_id)`: exact retrieval of memories linked to an entity.
- `merge_entities(keep_id, merge_id)`: repoints mentions and keeps the losing
  entity as a `merged_into` record rather than deleting it.
- `Relation`: typed, evidence-linked edges between entities.

The current hierarchy of truth remains:

```text
Episode                      immutable raw source
   -> Memory                 derived, reconciled, bi-temporal fact
      -> EntityMention       exact identity reference
         -> Entity           stable hub + synthesized description
```

Episodes, not entity descriptions, are the immutable source of truth. Memories
are the curated evidence layer and retain provenance back to episodes. Entity
descriptions are derived navigation and retrieval aids.

## Minimal target model

```text
Entity
  id                         existing
  name                       existing canonical display name
  normalized                 existing lookup form
  entity_type                existing: person, organization, project, ...
  description                NEW: bounded synthesis of active linked memories
  description_updated_at     NEW: when that synthesis was generated
  metadata                   existing; may hold explicit user aliases initially
  merged_into                existing merge history
  scope + timestamps         existing

EntityMention
  entity_id                  existing
  memory_id                  existing
  surface                    existing observed name/alias
  created_at                 existing

Memory
  unchanged                  derived, bi-temporal evidence with episode provenance
```

There is no separate profile row and no duplicated `based_on` list. The
`EntityMention` join already answers which memories support the description.
There is no separate evidence version: existing entity, mention, and memory
timestamps provide a sufficient staleness watermark.

## Entity example

The stored memory remains plain text:

```text
Memory m1: "Cosmin is a good student."
Entity e1: name="Cosmin", type="person"
EntityMention: entity_id=e1, memory_id=m1, surface="Cosmin"
```

The UI should always expose `Cosmin` as a safe entity chip or link beside the
memory. Exact inline rendering is optional and best-effort. Opening the entity
gives:

```text
Cosmin
  "Cosmin is a physics student at TU Berlin..."   <- synthesized description

Related memories
  - Cosmin is a good student.
  - Cosmin studies physics.
  - Cosmin moved to Berlin in 2025.
```

Do not write `[Cosmin]` or an internal entity ID into `Memory.content`. Reference
syntax in stored content would leak into embeddings, full-text search, exports,
and the stored memory representation. The existing `surface` field is enough
for reliable entity chips, but it cannot always identify an exact inline span.
For example, `Cosmin met Cosmin's advisor` repeats the same surface. Keep inline
linking explicitly best-effort and add character spans only if real UI tests show
that exact inline references are worth the extra model and storage complexity.

## Synthesized description

The description exists to provide a compact orientation and a coarse retrieval
entry point. It is not a belief and must not become canonical truth.

Rules:

- Synthesize only from **active** linked memories by default.
- Keep it bounded (target roughly 100-300 tokens).
- Preserve concrete names, dates, numbers, versions, and negative constraints.
- Represent conflicting or changing facts honestly instead of silently choosing
  one when temporal evidence is unclear.
- Rebuild after an entity merge because the evidence sets have changed.
- Invalidation, hard deletion, and merge must touch each affected entity's
  `updated_at` (or clear its description timestamp) so lazy refresh cannot miss it.
- Exclude descriptions from normal memory reconciliation and decay.
- Return the description with a small set of relevant linked memories so answers
  remain grounded in exact evidence.
- Do not add an entity-description embedding initially. Measure whether it
  improves entity-focused or multi-hop retrieval before paying its storage and
  re-embedding cost.

Descriptions should be lazy so normal saves stay cheap. When an entity is opened
or selected during retrieval, compare `description_updated_at` with the latest of:

- the entity's `updated_at`;
- its mention timestamps; and
- the `updated_at` values of all linked memories, including invalidated ones.

If newer evidence or an invalidation exists, regenerate from the currently active
linked memories. This avoids a separate dirty flag or evidence-version counter.
A later maintenance job may refresh frequently used entities in batches, but it
must not add an LLM call to every memory write.

### Required correctness invariant

`entity_memories()` excludes invalidated memories by default and exposes
`include_invalid=True` only for history/audit callers. Invalidation, deletion,
content changes, new mentions, alias/type changes, and merges clear the description
watermark and advance the entity evidence clock. Synthesis and ordinary entity
retrieval therefore use active evidence only.

## Aliases and conservative merging

Aliases improve candidate discovery; they do not prove identity.

For an active entity, its alias set is:

```text
canonical name
+ distinct EntityMention.surface values
+ canonical names of entities whose merged_into points to it
+ explicit aliases supplied by the user (optional, initially in metadata)
```

This reuses evidence Memry already has. When entities merge, mention surfaces are
repointed and the losing entity remains stored, so its names are not lost.

Merge resolution follows two steps:

1. **Candidate discovery:** find entities whose canonical name or aliases match
   the incoming surface. Use normalized, indexed lookup rather than scanning the
   full entity vocabulary.
2. **Identity decision:** compare type, synthesized description, and a small set
   of linked memories. Alias overlap raises a candidate; it never auto-merges by
   itself. Duplicate normalized names remain legal because two people can both
   be called Cosmin.

This separation should reduce LLM work: deterministic name/alias lookup narrows
the candidate set, and the bounded descriptions provide compact context for the
remaining ambiguous comparison. Existing high-confidence and human-confirmed
merge safeguards remain in place.

A dedicated `entity_aliases` table is deliberately deferred. Add it only if
manual aliases become common or benchmarks show that indexed lookup cannot be
implemented adequately from canonical names, observed surfaces, and merge
history. If introduced, aliases must not be globally unique.

## Entity-focused retrieval

For a query about a known entity:

```text
query surface
  -> indexed canonical/observed/merged-name lookup, then manual-alias fallback
  -> entity ID candidate(s)
  -> synthesized description
  -> rank active linked memories against the query
  -> optionally traverse typed relations
  -> return bounded context
```

This replaces the current per-query scan of the full entity vocabulary. Direct
entity lookup should not require an LLM. Graph traversal remains conditional: it
runs when a referent is detected and the query can benefit from relations.

High-degree entities must not flood context. Always cap linked memories and graph
neighbors, rank them for the current query, and retain exact provenance in the
response.

## Topics (today's tags/categories)

A topic is a canonical classification label such as `health` or `finance`. It is
not an identity-bearing referent and must never enter entity disambiguation.

Minimal normalized storage:

```text
Topic
  id, name, normalized, scope, provenance

MemoryTopic
  memory_id, topic_id
  unique(memory_id, topic_id)
```

Required indexes:

- unique normalized topic per scope;
- `(topic_id, memory_id)` for filtering and counts; and
- `(memory_id, topic_id)` for reading a memory's topics.

During migration, keep `Memory.categories` and all REST/MCP/Python category
arguments as a compatibility projection. Backfill topics and links, dual-write,
compare counts and filter results, then switch reads to the indexed links. Do not
force callers to adopt new names in the same release as the storage migration.

Synthetic topics are hierarchy edges such as `health broader_than running`, not
umbrella strings copied onto every member memory. Query expansion follows those edges
without rewriting all affected memories.

## Relations and collections

- **Relations stay separate.** They are typed, evidence-grounded world edges
  between referents, such as `Cosmin studies_at TU Berlin`.
- **Topic hierarchy stays separate from world relations.** `broader_than` is a
  taxonomy operation, not an entity relationship.
- **Collections stay separate.** They are generated navigation summaries over
  clusters of memories, not identities or classification labels.

A unified UI does not require a unified persistence table.

## Knowledge UI

Replace the separate top-level Tags and Entities entry points with one
**Knowledge** area containing:

- **Topics:** counts, filter, rename, combine, delete, synthetic marker.
- **People & things:** canonical name, type, aliases, synthesized description,
  linked memories, merge proposals, and merge/reject actions.
- **Relations:** typed referent-to-referent relationships with supporting memory.
- **Collections:** generated cluster summaries and their member memories.

Actions remain role-specific. A topic can be renamed or combined freely; an
entity merge remains conservative because a false merge contaminates future
recall.

## Implementation phases

The topic and entity workstreams are independent. The correctness prerequisite
is small and should land first, but topic normalization does not need to wait for
entity descriptions or the entity UI.

0. **Entity correctness prerequisite**
   - Make `entity_memories()` active-only by default.
   - Make invalidation, hard deletion, and merge touch every affected entity's
     `updated_at` (or clear its description timestamp).
   - Preserve SQLite migration safety and tenant scoping.

1. **Normalized topics behind compatibility APIs**
   - Add `Topic` and `MemoryTopic` storage and indexes.
   - Backfill and dual-write.
   - Switch filters, histograms, and tag management to indexed operations after
     parity checks.
   - Check in a reproducible filter benchmark and keep its machine-specific result
     separate from external performance claims or release thresholds.

2. **Indexed entity lookup and alias candidate discovery**
   - Add indexed canonical-name and observed-surface lookup.
   - Replace the full-vocabulary scan in query entity detection.
   - Use aliases to generate candidates, never as automatic merge proof.

3. **Descriptions and alias behavior**
   - Add `description` and `description_updated_at` to `Entity` and SQLite with
     additive migration support.
   - Expose derived aliases and the description through entity APIs.
   - Add lazy, bounded description synthesis and refresh after merges or stale
     evidence.
   - Continue returning linked memories as evidence.

4. **Entity hub retrieval and UI**
   - Resolve entity-focused queries through the hub before graph expansion.
   - Add the entity detail view with description, aliases, and active memories.
   - Render reliable entity chips and only best-effort inline references without
     rewriting memory content.

5. **Topic hierarchy and unified Knowledge UI**
   - Represent synthetic abstraction through topic hierarchy edges.
   - Migrate existing synthetic markers without losing provenance.
   - Unite Topics, People & things, Relations, and Collections in the UI.

6. **Optional optimizations, only if earned**
   - Entity-description embeddings.
   - Exact character spans or semantic mention roles.
   - Dedicated manual-alias table.
   - A shared physical anchor table.


## Implementation snapshot (2026-07-24)

Phases 0 through 5 are implemented in the current repository state: active-only entity
evidence, mutation-aware description freshness, normalized indexed topics and migration,
indexed canonical/observed/merged-name discovery with metadata-alias fallback, lazy
bounded descriptions, entity-focused context, entity chips, hierarchy edges, legacy
synthetic-tag migration, and the unified Knowledge dashboard.
Phase 6 remains explicitly deferred. Focused tests and the reproducible topic-filter
benchmark are checked in; public end-to-end memory benchmarks remain outstanding.
## Validation gates

| Metric | Required evidence |
|---|---|
| entity lookup p50/p95 | indexed lookup improves over vocabulary scan across realistic entity counts |
| alias candidate recall | known aliases find the correct entity without increasing false merges |
| false-merge rate | no regression; alias overlap alone never triggers a merge |
| description faithfulness | sampled descriptions preserve active facts and do not resurrect invalidated ones |
| description freshness | invalidation, content update, new mention, and merge all trigger lazy refresh |
| entity-focused retrieval | improved answer quality and bounded latency for "what do I know about X?" |
| tokens per saved memory | unchanged by lazy description synthesis |
| description token cost | measured per refresh and amortized per entity access |
| topic-filter p50/p95 | indexed topic links improve filtering and histograms |
| migration/tenant safety | upgraded SQLite files preserve scoped results and ID-addressed APIs remain tenant-confined |
| storage growth | descriptions and normalized links remain proportionate |

Public LoCoMo/LongMemEval-style evaluation remains necessary before calling the
system best-in-class. Schema cleanliness is not a substitute for measured recall,
identity precision, latency, and token cost.

## Explicit non-goals

- Do not turn entities into ordinary `Memory` rows.
- Do not copy every entity fact into its description.
- Do not store internal reference markup in memory text.
- Do not make aliases globally unique or use alias overlap as merge proof.
- Do not run description synthesis or identity LLM calls on every save.
- Do not let topics enter referent identity resolution or graph seeding.
- Do not perform a big-bang physical merge of tags and entities.
- Do not add optional profile, role, span, alias-table, or embedding structures
  before measurements justify them.

## Evidence behind the design

- Zep separates summarized entity nodes, raw episodic nodes, and fact-bearing
  entity edges: <https://help.getzep.com/v2/understanding-the-graph>
- Microsoft GraphRAG gives entities synthesized descriptions while retaining
  links to source text units: <https://microsoft.github.io/graphrag/index/outputs/>
- HippoRAG uses entities as an associative index into source passages rather
  than replacing those passages: <https://arxiv.org/abs/2405.14831>
- APEX-MEM's entity-event model keeps facts on temporally grounded events so
  summaries do not erase contradictions or history:
  <https://arxiv.org/abs/2604.14362>
