# Knowledge model feature catalog

Status date: 2026-07-24

This is the decision register for the knowledge model. Each row is one decided feature or
boundary and explains why it exists. Status describes the current repository state:

- **Implemented**: the decided behavior exists on the supported production path and has
  focused validation.
- **Partial**: useful code exists, but a stated completion condition is still missing.
- **Not implemented**: accepted work remains to be built.
- **Deferred**: deliberately excluded until a measured trigger justifies it.
- **Superseded**: an older decision was explicitly replaced and its implementation removed.

Implementation status is not a quality claim. Public end-to-end benchmarks remain the gate
for any "best in class" statement.

## Architecture and product boundaries

| Feature or decision | Status | Decided behavior | Explanation and justification |
|---|---|---|---|
| Two production SQL backends | Superseded | Do not maintain parallel SQLite and PostgreSQL implementations. | The second implementation was not tied to a demonstrated customer deployment and made every feature require duplicate schema, migration, behavior, and test work. PostgreSQL code, configuration, dependency, tests, and deployment claims were removed. |
| SQLite-only production persistence | Implemented | Use SQLite as the sole production memory database. | The shipped product is a low-operations service. One engine minimizes installation cost, support surface, parity defects, and time-to-feature. The explicit disadvantage is no supported horizontally scaled write cluster. |
| One server process, many clients | Implemented | Let many agents and devices share one Memry server; keep one process as the database write owner. | Customers get shared memory without running a database service. Multiple clients are not multiple database writers, so the operating model stays simple and testable. |
| Optional Mem0 comparison adapter | Partial | Keep Mem0 only as a reduced interop/evaluation adapter, not as a feature-equivalent production promise. | It is useful for comparisons, but it currently remains selectable as the whole runtime backend and drives silent optional methods in `MemoryBackend`. The duplication audit proposes isolating it from the product path. |
| Explicit backend capability failures | Partial | Unsupported adapter operations should be explicit rather than inherited silent no-ops. | Silent success can discard topics, entities, relations, events, or descriptions. The risk is documented, but the base interface still contains no-op defaults for the reduced Mem0 adapter. |
| Separate authentication SQLite file | Implemented | Runtime accounts and OAuth state currently live in `auth.db`, adjacent to `memry.db`. | This isolated identity lifecycle from the old selectable-backend design. With SQLite now the sole production store, the extra backup/transaction lifecycle may no longer earn its cost; consolidation is listed for a later decision, not silently performed. |
| Complete technology inventory | Implemented | Keep the actual runtime, provider, protocol, build, test, and deployment technologies in `architecture.md`. | Operators and product owners should know what they are paying to run and maintain; hidden transitive technologies or implied infrastructure make decisions impossible to evaluate. |
| Architecture decision discipline | Implemented | Record consequential decisions with product need, benefit, cost, limits, migration, and validation before building them. | Technical capability alone is not a product reason. This catalog and the architecture limits are the review record; future architecture still needs explicit approval. |
| Unified Knowledge area, separate internals | Implemented | Present Topics, People and things, Relations, and Collections in one dashboard area without forcing them into one physical model. | Users need one place to inspect knowledge. Topics and referents have different invariants and hot paths, so UI unity should not create persistence or resolution complexity. |

## Evidence and entity model

| Feature or decision | Status | Decided behavior | Explanation and justification |
|---|---|---|---|
| Immutable episode source | Partial | Capture raw episodes before inference and keep them as source evidence. | This preserves what was actually received when extraction changes. Storage exists; a complete replay/rebuild workflow over all derived structures is still not shipped. |
| Bi-temporal memory evidence | Implemented | Keep atomic memories with validity, invalidation, supersession, source episode IDs, and audit events. | Old claims remain inspectable instead of being silently overwritten, and derived entity views can always point back to exact evidence. |
| Stable entity identity | Implemented | Give every referent a durable ID, canonical/normalized name, type, scope, merge state, and timestamps. | IDs remain stable when display names, aliases, descriptions, or evidence change. Duplicate names are valid because names do not establish identity. |
| Entity as a memory hub | Implemented | Opening an entity returns its identity, bounded description, aliases, mentions, active memories, and relation navigation. | This provides the useful "entity memory" experience without pretending a mutable profile is an atomic claim. |
| Minimal entity extension | Implemented | Add only `description` and `description_updated_at`; reuse `EntityMention`. | Two nullable cache fields avoid `EntityProfile`, `MemoryAnchorLink`, `based_on` lists, evidence counters, role taxonomies, and another synchronization lifecycle. |
| Description as derived cache | Implemented | Treat the description as replaceable synthesis; memories and episodes remain authoritative. | Summaries can omit or compress facts. A cache can be regenerated safely, while a second truth store would become unauditable. |
| Bounded entity descriptions | Implemented | Bound synthesized descriptions to at most 300 words and 1,200 characters while instructing the model to preserve dates, numbers, constraints, negation, and conflicts. | Hard bounds make prompt size and cost predictable. Preserving high-information details avoids the most damaging summary failures. |
| Lazy description synthesis | Implemented | Generate or refresh only when an entity is opened or selected for reconstructed context. | Normal writes incur no description LLM call. Cost is paid only for entities that are used; the first stale read may be slower. |
| Keyless description fallback | Implemented | When no LLM is available or synthesis fails, build a bounded factual excerpt from active memories. | Entity hubs remain useful and deterministic in zero-key mode without inventing facts or failing the read. |
| Timestamp-based freshness | Implemented | Compare `description_updated_at` with the entity evidence watermark; do not add a dirty flag or version counter. | Existing timestamps are enough when every evidence mutation reliably touches the entity or clears the description watermark. This removes another state machine. |
| Mutation-aware refresh | Implemented | New mentions, content edits, invalidation, hard deletion, type changes, aliases, and merges advance the entity clock and clear description freshness. | Missing any of these transitions could serve obsolete evidence as current. Focused tests cover active-only invalidation, deletion, merge, and regeneration. |
| Active-only entity memories | Implemented | `entity_memories()` excludes invalidated evidence by default; audit callers opt in with `include_invalid=True`. | Obsolete claims must not poison descriptions, entity views, graph traversal, or normal answers, while history remains available. |
| Exact entity-to-memory links | Implemented | Use `EntityMention` as the authoritative entity evidence link and retain its observed surface text. | The join already answers which memories support an entity and avoids duplicated profile provenance structures. |
| Evidence-backed entity answers | Implemented | Pair compact entity orientation with bounded active supporting memories in detail and context results. | The description is fast to consume; exact memories make the answer auditable and recover details omitted by compression. |
| Derived aliases | Implemented | Derive aliases from canonical names, observed mention surfaces, and names of merged entities. | These signals already exist in evidence and merge history, so they do not require a new write lifecycle. |
| Optional user aliases in metadata | Implemented | Let users add aliases through Python, CLI, REST, and the Knowledge UI; store them in entity metadata. | Manual correction is valuable now. A separate alias table is not justified until alias volume or latency measurements show the metadata trade-off is material. |
| Alias candidate signals, not proof | Implemented | Use canonical or alias matches to discover candidates, then apply conservative contextual identity judgment. | Different people can share names and nicknames. Automatically merging on alias overlap would trade cheap ambiguity for expensive corruption. |
| Indexed canonical and observed-surface lookup | Implemented | Resolve canonical names and mention surfaces through SQLite indexes and generate bounded phrases from the query. | Query work scales with the query and match set instead of loading the complete entity vocabulary into Python. |
| Metadata-alias lookup performance | Partial | User metadata aliases are discoverable, but their JSON fallback is not separately indexed. | This is the explicit cost of deferring an alias table. Promote aliases to dedicated non-unique indexed storage only after measurement shows a product latency problem. |
| Compact disambiguation context | Implemented | Give identity comparison the cached description plus a few active memories. | The LLM sees enough evidence to judge ambiguity without repeatedly sending an entity's full history. The description remains supporting context, not merge authority. |
| Duplicate-name safety | Implemented | Allow several active entities with the same normalized name. | Ambiguous names are normal. A uniqueness constraint would force unrelated referents together. |
| Conservative merging | Implemented | Auto-merge only high-confidence matches; keep unsure cases separate with confirm/reject proposals. | A missed merge is recoverable. A false merge contaminates every future recall about both identities. |
| Non-destructive merge history | Implemented | Mark the loser with `merged_into`, repoint mentions and relation endpoints, retain the losing name as alias evidence, and stale the survivor description. | Identity transitions remain inspectable and no evidence is deleted merely because two hubs were unified. |
| Indexed query entity detection | Implemented | Generate at most 128 query phrases and ask indexed canonical, observed-surface, and merged-name lookup for at most 50 candidates, with a metadata-alias fallback. | The previous full-vocabulary scan became slower with every entity. Bounded indexed discovery protects common read latency. |
| Entity-focused retrieval | Implemented | Resolve known query entities, include bounded hub context, and add graph-reachable evidence only when the query names a referent. | Direct entity questions should not start with an unbounded global graph search. Hub-first context is cheaper and less noisy. |
| Bounded entity neighborhoods | Implemented | Cap linked memories per entity, relation hops, fallback co-occurrence memories, detected entities, and description context. | High-degree entities cannot flood the context window or make query cost grow without a product-controlled bound. |
| Reliable entity chips | Implemented | Return entity IDs/names with memory API payloads and render clickable chips in the dashboard. | Chips use exact stored links and do not depend on fragile character matching. |
| Plain stored memory text | Implemented | Keep `Cosmin` as ordinary content; never persist bracket markup or entity IDs inside memory text. | UI syntax would pollute embeddings, FTS, exports, and external clients and would couple durable evidence to one renderer. |
| Exact inline entity highlighting | Deferred | Do not highlight inline until unambiguous character spans are demonstrably needed. | `EntityMention.surface` alone is ambiguous when text repeats a name. Chips are correct today; a span model would add extraction and migration complexity. |

## Topic model

| Feature or decision | Status | Decided behavior | Explanation and justification |
|---|---|---|---|
| Separate topic and entity behavior | Implemented | Keep deterministic classifications out of referent disambiguation and graph seeding. | `health` has one classification meaning; `Jonas` may refer to several identities. Combining their behavior would add ambiguity and slow both hot paths. |
| Canonical scoped topics | Implemented | Store a normalized `Topic` once per user/agent/run scope. | Stable topic identity removes repeated string comparisons from management operations while respecting namespace isolation. |
| Normalized topic links | Implemented | Store indexed many-to-many `MemoryTopic` links and backfill existing category JSON. | Joins make exact filtering, histograms, edits, and hierarchy expansion efficient and structurally clear. |
| Bidirectional topic indexes | Implemented | Index both `(topic_id, memory_id)` and `(memory_id, topic_id)`. | One direction serves filters/counts; the other loads or replaces a memory's assignments efficiently. |
| Category API compatibility | Implemented | Keep `categories` fields and arguments across Python, CLI, REST, MCP, and exports while topic links back them. | Existing integrations should not break merely because storage became normalized. Product wording can move to Topics without forcing an API migration today. |
| Safe topic migration | Partial | Backfill and dual-write JSON plus normalized links, use links for reads, and retain a repair/rollback window. | The live migration is additive and tested. The final cleanup release that removes the JSON column/write has not been chosen and is listed in the duplication audit. |
| Indexed topic filtering | Implemented | Push category filters and histograms through normalized joins rather than per-row JSON scans. | On a 2026-07-24 run of the checked-in 50,000-memory, 300-query microbenchmark, the local median was 45.7563 ms for legacy JSON scanning versus 0.0109 ms for indexed links. That supports the implementation choice but is not an end-to-end product benchmark. |
| Reproducible topic benchmark | Implemented | Check in dataset generation, sizes, query count, SQL, seed, and timing method in `evals/topic_filter_benchmark.py`. | Anyone can reproduce or challenge the speed measurement instead of trusting an unexplained ratio. |
| Synthetic topic hierarchy | Implemented | Represent `health` broader than `running` as `TopicRelation` edges, not copied parent labels. | Taxonomy changes no longer rewrite every memory, and assigned topics remain distinguishable from inferred parent recall. |
| Legacy synthetic-tag migration | Implemented | Convert recorded synthetic parent/source strings to edges once and remove copied umbrella labels only from matching legacy child memories. | Existing data gains the new semantics without discarding proposal provenance or deleting unrelated manual labels. |
| Bounded query-time topic expansion | Implemented | Expand parent filters through at most eight hierarchy levels using indexed edges. | Hierarchical recall works without corpus rewrites, while a depth cap prevents malformed or very deep taxonomies from creating unbounded work. |
| Separate taxonomy edges | Implemented | Store topic hierarchy separately from factual entity relations. | `health broader_than running` is a classification rule, not a claim between real-world referents; mixing them would corrupt traversal semantics. |

## Relations, collections, UI, and safety

| Feature or decision | Status | Decided behavior | Explanation and justification |
|---|---|---|---|
| Evidence-grounded entity relations | Implemented | Store typed subject-predicate-object edges with an optional supporting memory. | Multi-hop recall can reach facts sharing no query words while retaining an evidence pointer. Invalidating/deleting the evidence invalidates/removes the edge. |
| Conditional graph traversal | Implemented | Traverse relations only when indexed query resolution finds an entity; otherwise use normal hybrid retrieval. | Most lookups do not need graph work. Conditional traversal protects latency and reduces noisy expansions. |
| Collections as navigation | Implemented | Keep generated cluster titles/summaries over memory IDs and expose member navigation. | Collections help browse emergent themes without becoming another canonical identity or fact store. |
| Role-specific Knowledge controls | Implemented | Topics support rename/combine/delete; entities expose aliases, evidence and merge proposals; relations open evidence; collections open members. | A unified area remains understandable because each concept exposes operations matching its invariants. |
| Tenant-safe knowledge operations | Implemented | Apply principal ownership and user/agent/run scoping to entity detail, aliases, proposals, topics, relations, collections, and memory IDs. | Knowledge links can leak private data just as easily as memory text. ID-addressed alias/detail/evidence calls stay behind the same ownership gate. |
| Cheap normal write path | Implemented | Do not require description synthesis, taxonomy rebuilding, or extra graph scans on every memory write. | Raw evidence and deterministic links land first; lazy or scheduled derivation keeps ingestion latency and provider spend predictable. |

## Deliberate deferrals and remaining proof

| Feature or decision | Status | Decided behavior | Explanation and justification |
|---|---|---|---|
| Entity-description embeddings | Deferred | Do not embed descriptions until an evaluation shows material recall improvement. | They add storage, provider cost, re-embedding work, and another stale-index lifecycle. Current entity links and descriptions work without them. |
| Dedicated alias table | Deferred | Keep derived aliases and user metadata until measured lookup latency or alias volume earns another table. | The current model avoids schema complexity; the trade-off is the explicitly partial metadata-alias hot path above. |
| Physical topic/entity anchor merge | Deferred | Do not combine topic and entity tables just to reduce table count. | A shared UI concept does not justify sparse fields, conditional invariants, or merging a deterministic filter path with ambiguous identity resolution. |
| Generic anchor links, roles, and spans | Deferred | Do not add `MemoryAnchorLink`, role taxonomies, or character spans without a demonstrated feature that needs them. | `EntityMention` and topic links cover current evidence and navigation with far less lifecycle complexity. |
| Measurement promotion gates | Partial | Track identity precision, false merges, description faithfulness/freshness, retrieval quality, p50/p95 latency, token cost, storage growth, migration safety, and isolation. | Focused correctness tests and a topic microbenchmark exist. A complete repeatable scorecard and release thresholds do not. |
| Public long-memory evaluation | Not implemented | Run LoCoMo/LongMemEval-style evaluation before calling the system best in class. | Architecture and synthetic tests cannot establish end-to-end memory quality or fair competitive performance. |