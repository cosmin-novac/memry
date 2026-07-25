# Memry architecture

This document records the architecture that exists in the repository, the product reason
for each consequential choice, and the limits that follow from it. It is descriptive, not
a roadmap.

## 1. Product topology

A normal deployment is one Memry Python process backed by one SQLite memory database.
Many agents, browsers, and API clients can connect to that process over MCP or REST, but
they are not independent database writers: the server owns the database connection and
serializes writes inside the process.

```text
Agents and applications
  |-- MCP over stdio (local client)
  |-- MCP streamable HTTP at /mcp
  |-- REST/JSON at /api/v1
  |-- dashboard in a browser
  `-- Python API / CLI
              |
         MemoryStore
          |       |
          |   managed enrichment worker
          |   (pending rows, bounded batches)
          |
    intelligence and retrieval
              |
       LocalBackend (SQLite)
        |-- memry.db: knowledge data and search indexes
        `-- *.usearch: optional rebuildable ANN sidecars

    account and OAuth store
              |
        auth.db: accounts, sessions, clients, and tokens
```

The VPS bundle adds Caddy in front of the single Memry process for HTTPS and reverse
proxying. There is no supported horizontally scaled or active-active write topology.

### Why SQLite is the only production store

The actual product requirement is a self-hosted memory service that is cheap to install,
backup, and operate. SQLite satisfies that requirement without an external database
service and is already used by the shipped Docker and VPS deployments. Maintaining a
second SQL implementation
did not serve a demonstrated customer deployment and made every schema or knowledge
feature require two implementations, two migrations, and two test paths. PostgreSQL was
therefore removed.

Business benefit:

- installation needs no database service, credentials, migration operator, or managed
  database bill;
- a small customer can back up and restore the durable memory state as files;
- every product feature has one production persistence path, reducing delay and parity
  defects;
- the supported operating model is easy to explain: run one Memry server and let all
  clients connect to it.

The cost is explicit: Memry does not currently support several server replicas or several
machines writing the same store, database-native high availability, or zero-downtime
horizontal write scaling. If a real product requirement later needs those capabilities,
that is a new, reviewed architecture decision with a migration plan. It is not a hidden
alternative backend.

Decision provenance: the PostgreSQL path first entered the original Git history in commit
`162b4eee3652752f70b301e7a330aa18b701492a` on 2026-07-17. The commit was authored by
Cosmin Novac and carries `Co-Authored-By: Claude Fable 5`; its stated reason was the generic
phrase "multi-writer deployments." The repository contained no named customer, deployment,
load target, or product requirement that needed that topology. A later history rewrite
folded the change into root commit `6e74c9c902f1198ef135777b3f642d74d2519e68`. The
SQLite-only decision reverses that unsupported architecture choice explicitly.

### Why Mem0 is comparison/import-only

Mem0 runtime selection was removed on 2026-07-24. Its reduced adapter cannot preserve
Memry episodes, invalidation history, normalized topics, entity links, relations,
descriptions, or collections. Offering it as a runtime setting therefore made the same
product silently behave differently and lose features depending on one environment
variable. No user or deployment depended on that mode.

The business decision is one complete runtime product on SQLite. The optional `mem0ai`
dependency and adapter remain available only to explicit comparison/import code, where
those limits are visible and intentional.

## 2. Main layers

| Layer | Responsibility | Important rule |
|---|---|---|
| Public surfaces | Python API, CLI, REST, dashboard, MCP | They call `MemoryStore`; they do not implement memory behavior independently. |
| `MemoryStore` | Ownership checks, write/read workflows, knowledge operations | It is the product facade and the main invariant boundary. |
| Intelligence | Extraction, reconciliation, entity resolution, relation extraction, topic abstraction, descriptions, decay, collections | Derived outputs remain rebuildable from evidence. |
| Retrieval | FTS5/BM25, vectors, reciprocal-rank fusion, recency/importance, entity graph expansion | Invalidated evidence is excluded by default and work is bounded. |
| Providers | LLM and embedding integrations | External providers are optional; zero-key fallbacks remain functional. |
| Production persistence | `LocalBackend` | SQLite is the sole production source of truth. |

The `MemoryBackend` interface isolates persistence and permits explicit test fixtures.
`MemoryStore` always constructs `LocalBackend` in normal operation. The optional Mem0
adapter can only be instantiated directly by comparison/import code; there is no config,
environment variable, CLI flag, REST option, or server option that selects it.

## 3. Stored knowledge model

### Evidence records

| Record | Purpose | Lifecycle |
|---|---|---|
| `Episode` | Raw input captured before derived processing | Append-only source evidence. |
| `Memory` | One derived or verbatim claim | May be updated, invalidated, or superseded; hard deletion is explicit. |
| `MemoryEvent` | Audit event for add/update/supersede/delete decisions | Append-only audit trail. |
| Embedding and FTS row | Search representation of a memory | Derived and rebuildable. |

A memory has content, one memory type, importance, public `categories`, compatibility
`entities`, metadata, scope (`user_id`, `agent_id`, `run_id`), timestamps, source episode
IDs, and validity fields (`valid_from`, `invalid_at`, `superseded_by`). The validity fields
preserve old claims instead of pretending that the latest claim erased history.

### Tags (backend names: categories and topics)

The product and dashboard call deterministic classification labels such as `health` or
`finance` **tags**. The existing Python/REST field remains `categories`. Internally, the
normalized `topics` table plus the indexed `memory_topics` join provides canonicalization,
hierarchy, counts, and filtering. The memory's JSON `categories` list is the public projection
returned with each memory. Both are updated together deliberately; they are not competing
knowledge concepts and no rename or data migration is planned.

Mechanical separator and singular/plural duplicates are merged deterministically once two
real stored labels map to the same form. Semantic synonym merges remain reviewable.
Synthetic umbrella topics are hierarchy edges, for example `health` broader than
`running`. The parent label is not copied onto every child memory. Filters expand the
hierarchy in SQLite at query time, so taxonomy changes do not rewrite the memory corpus.
Topic hierarchy edges are separate from real-world entity relations.

### Entities

An entity is a stable identity hub for a person, organization, project, product, place,
event, concept, or other referent. `EntityMention` is the authoritative link from a memory
to an entity and keeps the observed surface text.

Names and aliases discover identity candidates. Candidate lookup uses indexed canonical
names, observed mention surfaces, and merged names. Optional user aliases stored in entity
metadata use a fallback scan only when indexed evidence finds nothing. An exact multi-part
name plus meaningful contextual overlap is a deterministic identity match unless known types
conflict or the model finds a concrete contradiction. A shared short name or full name
without contextual overlap remains separate and creates a merge proposal. Proposal actions
resolve each endpoint through `merged_into`, so already-satisfied and stale proposals are
idempotent. Confirmed merges retain the losing entity, repoint mentions and relations, and
preserve its name as alias evidence.

Each entity has two derived profile fields only:

- `description`: a bounded synthesis of active linked memories;
- `description_updated_at`: the synthesis watermark.

The description is a cache, not a fact store. It is generated lazily when the entity is
opened or selected for context, costs nothing on the normal write path, and is rebuilt
when mentions, linked memory content, invalidation, deletion, type, alias, or merge state
changes. Active linked memories remain the evidence returned with the hub.

### Relations and collections

Entity relations are typed subject-predicate-object edges with an optional evidence memory.
Invalidating or deleting that evidence also invalidates or removes the relation. Search
traverses a bounded relation neighborhood only when the query resolves to a known entity.

Collections are generated titles and summaries over groups of memory IDs. They are a
navigation layer, not identities, topics, or authoritative claims.

## 4. Write path

### Durable MCP save and managed enrichment

The default `save_memories(infer=true)` path is intentionally split at the safe boundary:

1. Commit the exact input as both an immutable episode and an active, searchable memory.
2. Mark that memory `pending_distillation` in its existing SQLite metadata and return the
   MCP acknowledgement. No LLM or embedding request runs before this response.
3. Wake one in-process worker. It selects at most eight due pending memories per database
   pass, but sends every memory through extraction separately. Text, user scope,
   provenance, and failure handling are never combined across payloads.
4. On success, reconcile the extracted facts and supersede the raw pending memory. If
   extraction finds no facts, keep the raw memory and clear the pending marker.
5. On provider or processing failure, keep the raw memory active, record the error, and
   retry with exponential backoff capped at five minutes. After a process restart, the
   worker discovers the same pending rows, including work interrupted while processing.

The active pending memory is both usable knowledge and the recovery marker. This avoids a
second queue database or broker and ensures acknowledgement never means "accepted only in
RAM." Status is visible on MCP memory rows and in aggregate statistics.

### Synchronous library and REST write

`store.add(...)` and the REST write route retain the synchronous workflow:

1. Store raw input as episodes before inference.
2. With an LLM, extract small candidate memories, types, importance, topics, entities, and
   possible relations. Without an LLM, store the input verbatim.
3. Retrieve similar active memories in the same scope and reconcile each candidate as add,
   update, supersede, or no-op.
4. Store or update the memory, normalized topic links, embedding, and FTS row.
5. Resolve entity mentions conservatively. Alias matches only narrow the candidates.
6. Store evidence-grounded relations whose endpoints resolved in that memory.
7. Append audit events.

When existing memory text is edited manually or rewritten by reconciliation, Memry analyzes
the final text before committing the change and replaces that memory's entity-name snapshot
and authoritative mention links together. A failed LLM analysis leaves the old text and links
unchanged. In zero-key mode, existing links are retained or removed by exact known-alias
matching; discovering a brand-new entity still requires an LLM.

Entity descriptions and synthetic topic hierarchy are not mandatory write-path work. This
keeps ingestion latency and provider cost bounded.

## 5. Read path

For a normal text query:

1. FTS5 produces BM25 keyword candidates.
2. The configured embedder produces vector candidates. Small stores use exact NumPy cosine
   scoring; the optional usearch HNSW sidecar supplies candidates above its threshold.
3. Reciprocal Rank Fusion combines the candidate lists.
4. Relevance is blended with recency and importance according to configuration.
5. If canonical or alias candidate lookup resolves a query entity, bounded typed-relation
   traversal can add otherwise unreachable multi-hop evidence.
6. Context reconstruction may prepend a bounded, lazily refreshed entity description and
   then packs exact memories into the remaining token budget.

The ANN file is a cache. SQLite remains authoritative, ANN candidates are exact-rescored,
and the index can be rebuilt. Invalidated memories are excluded unless a caller explicitly
requests history.

## 6. Product surfaces and security

- The Python API and `memry` CLI expose the same store workflows.
- `memry mcp` runs only the local stdio transport.
- Remote streamable HTTP/HTTPS MCP is available at `/mcp` only through `memry serve`, which
  hosts REST, the dashboard, OAuth endpoints when enabled, and MCP in one Starlette/Uvicorn
  process. Caddy supplies HTTPS in the VPS bundle.
- The standalone `memry mcp --transport http` launcher was removed on 2026-07-24. Removing it
  does not affect local stdio clients or remote clients pointed at a `memry serve` URL. It
  removes an MCP-only network server that bypassed Memry's account, OAuth, and bearer-key
  middleware and otherwise acted as the global administrator.
- A single operator bearer key, configured tenants, or runtime accounts can authenticate
  network calls. The operator key is the explicit global credential. The oldest/first runtime
  account is persisted as bootstrap administrator but is memory-confined to the existing
  `default` space; every later account is confined to one `<name>::default` space. The
  dashboard exposes no storage namespace selector. Administrator role and memory ownership
  are independent, and knowledge merges never cross account boundaries.
- Runtime API keys are stored as SHA-256 hashes; human passwords use scrypt with a random
  salt; comparisons are constant-time.
- OAuth uses the MCP SDK authorization-server interfaces with dynamic client registration,
  PKCE, short-lived authorization codes, access/refresh tokens, refresh rotation, and
  revocation.
- The dashboard uses an HTTP-only session cookie. OAuth discovery is mounted at the domain
  root; MCP is mounted at `/mcp`.

### Why memory and login data use separate files

This separation was kept deliberately on 2026-07-24. `memry.db` contains knowledge and
search data. `auth.db` contains accounts, password hashes, sessions, OAuth clients, and
tokens. As a result, knowledge export, import, or reset cannot overwrite who can log in or
invalidate credentials accidentally. That is the product benefit.

The cost is operational: a complete server backup must include both `memry.db` and
`auth.db` from the same point in time. They live in the same directory by default and in
the same Docker data volume, so one coordinated directory or volume snapshot captures
both. `memry export` is a lossless knowledge backup only; it does not contain login data.

## 7. Technologies used

This inventory lists technologies actually imported, executed, or shipped by the
repository. "Optional" means the default installation or deployment can function without
it.

### Runtime and data

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Python 3.11+ | Yes | Application language, CLI, servers, intelligence, providers, and evals. The Docker image currently uses Python 3.12 slim. |
| SQLite through Python `sqlite3` | Yes | Sole production persistence for memories; also the current runtime account/OAuth store. |
| SQLite WAL | Yes for file databases | Permits reads while the single server process serializes writes. |
| SQLite FTS5 | Yes | Content index and BM25 keyword retrieval. |
| SQLite JSON1 | Yes | Reads the public `categories` projection and metadata aliases; normalized topic links are the indexed path. |
| NumPy | Yes | Float32 embeddings, exact cosine scoring, clustering, and vector math. |
| Pydantic 2 | Yes | Configuration and typed domain/API models. |
| JSON and JSONL | Yes | Configuration values, REST payloads, exports/imports, provider structured output, and eval datasets. |
| Python standard library cryptography primitives | Yes for accounts | `hashlib`, scrypt, SHA-256, HMAC comparison, and `secrets` for credentials and tokens. |

### Servers, protocols, and UI

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Model Context Protocol Python SDK / FastMCP | Yes | MCP tools, stdio transport, streamable HTTP, and OAuth server interfaces. |
| AnyIO | Yes | Moves synchronous store/provider work off MCP event-loop tasks. |
| Starlette / ASGI | Yes for `memry serve` | REST routes, dashboard, middleware, sessions, OAuth routes, and the mounted MCP app. |
| Uvicorn | Yes for `memry serve` | Runs the combined ASGI application. |
| HTTPX | Yes | Reusable OpenAI, Voyage, and Ollama HTTP clients keep connections warm across background enrichment calls; also used by tests. |
| python-multipart | Yes for account/OAuth forms | Parses dashboard and OAuth login form bodies through Starlette. |
| REST over HTTP with JSON | Yes for network API | Application integration and dashboard data access. |
| MCP stdio | Optional surface | Zero-port local agent connection. |
| MCP streamable HTTP | Optional surface | Remote agents and multiple client devices connecting to one Memry server. |
| OAuth 2.1-style flows, DCR, and PKCE | Optional | Account sign-in for OAuth-capable MCP clients when `MEMRY_PUBLIC_URL` is configured. |
| HTML5, CSS, vanilla JavaScript, Canvas 2D | Yes for dashboard | Server-embedded dashboard and the topic galaxy visualization; no frontend build tool or framework. |

### Provider integrations and optional accelerators

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Deterministic hash embeddings | Built-in fallback | Offline, zero-key fuzzy lexical vectors. |
| Anthropic Python SDK | Optional extra `memry[anthropic]` | Anthropic LLM completion and structured output. |
| OpenAI Chat Completions HTTP API | Optional | LLM extraction, reconciliation, summaries, and descriptions. |
| OpenAI Embeddings HTTP API | Optional | Semantic memory embeddings. |
| Voyage embeddings HTTP API | Optional | Alternative semantic embeddings. |
| Ollama HTTP API | Optional | Local LLM and embedding provider. |
| usearch HNSW | Optional extra `memry[ann]` | Persistent approximate-nearest-neighbor candidate sidecars for larger stores. |
| Mem0 / `mem0ai` | Optional extra `memry[mem0]` | Used only when comparison/import code explicitly instantiates the reduced adapter; never selected by the running product. |

### Build, test, and deployment

| Technology | Required? | Where and why it is used |
|---|---:|---|
| Hatchling | Build-time | Builds the Python wheel and source distribution. |
| pytest | Development | Unit, integration, tenant, server, retrieval, and migration tests. |
| GitHub Actions | Release-time | Runs tests and publishes package artifacts. |
| Docker | Optional deployment | Builds the packaged single-process server image. |
| Docker Compose | Optional deployment | Runs the Memry container and, in the VPS bundle, Caddy. |
| Caddy 2 | Optional VPS deployment | TLS termination, gzip, and reverse proxying to the single Memry process. |
| Bash and curl | Optional VPS installer | Installs and operates the bundled Docker deployment on a Linux VPS. |

## 8. Current limits and non-promises

- One Memry process owns a production database. Multiple write replicas are unsupported.
- Complete backups must capture `memry.db` and `auth.db` together; ANN sidecars may be discarded and
  rebuilt.
- The Mem0 adapter is comparison/import-only and is not a supported runtime persistence path.
- There is no external IdP/SSO integration, per-key rate limiter, external queue
  service, separate vector database, or distributed cache.
- Description faithfulness and end-to-end memory quality still require public evaluation;
  a clean schema and synthetic benchmarks do not establish "best in class" quality.
- Exact inline entity highlighting is deferred because mention surfaces do not provide
  unambiguous character spans. Reliable entity chips are the shipped navigation path.

## 9. Decision record

| Decision | Product reason | Implemented? |
|---|---|---:|
| SQLite is the only runtime database | One complete, cheap self-hosted product is more valuable than maintaining an unused second SQL implementation. | Yes |
| Mem0 is comparison/import-only | Its adapter cannot preserve the complete Memry knowledge model and no runtime user depended on it. | Yes |
| Knowledge and login data remain in `memry.db` and `auth.db` | Knowledge restore/reset cannot overwrite credentials; a complete server backup must capture both files together. | Yes |
| Local MCP uses `memry mcp`; remote MCP uses `/mcp` from `memry serve` | This preserves local zero-port use and one network server where configured authentication is applied. The separate unauthenticated HTTP launcher added risk without a used product case. | Yes |
| Edited memory text is re-analyzed for entity links | Entity chips and entity filters must describe the current text, not names left behind by an older version. | Yes |
| The UI says tags; the public backend field remains `categories`; normalized storage remains `topics`/`memory_topics` | Users get one familiar word without a breaking API/schema rename. | Yes |
| MCP saves persist raw text before acknowledgement and enrich it in one managed worker | Agent calls return after a cheap SQLite commit instead of waiting on several provider calls, while the active pending row prevents data loss and enables restart recovery without another queue system. | Yes |
| Background work uses bounded database batches but separate prompts per memory | Bounded draining improves throughput; separate prompts preserve each user scope, provenance, retry, and failure boundary. | Yes |
| Anthropic defaults to claude-haiku-4-5 | Memory extraction is frequent background work, so the lower-cost, lower-latency model is the useful default; operators can explicitly select a larger model when quality justifies the extra cost. | Yes |
| Provider HTTP clients are reused for the store lifetime | Reusing connections removes repeated connection setup from enrichment latency without adding a service or a second execution path. | Yes |

Any future consequential architecture change must be added here with its product reason and
implementation status before it is treated as decided work.
