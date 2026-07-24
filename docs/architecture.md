# Memry architecture

This document records the architecture that exists in the repository, the product reason
for each consequential choice, and the limits that follow from it. It is descriptive, not
a roadmap. Planned knowledge-model work is tracked separately in
[knowledge-model-features.md](knowledge-model-features.md).

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
              |
    intelligence and retrieval
              |
       LocalBackend (SQLite)
        |-- memry.db: product data and search indexes
        |-- auth.db: runtime accounts and OAuth state, when used
        `-- *.usearch: optional rebuildable ANN sidecars
```

The VPS bundle adds Caddy in front of the single Memry process for HTTPS and reverse
proxying. There is no supported horizontally scaled or active-active write topology.

### Why SQLite is the only production store

The actual product requirement is a self-hosted memory service that is cheap to install,
backup, and operate. One SQLite file satisfies that requirement and is already the storage
used by the shipped Docker and VPS deployments. Maintaining a second SQL implementation
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

## 2. Main layers

| Layer | Responsibility | Important rule |
|---|---|---|
| Public surfaces | Python API, CLI, REST, dashboard, MCP | They call `MemoryStore`; they do not implement memory behavior independently. |
| `MemoryStore` | Ownership checks, write/read workflows, knowledge operations | It is the product facade and the main invariant boundary. |
| Intelligence | Extraction, reconciliation, entity resolution, relation extraction, topic abstraction, descriptions, decay, collections | Derived outputs remain rebuildable from evidence. |
| Retrieval | FTS5/BM25, vectors, reciprocal-rank fusion, recency/importance, entity graph expansion | Invalidated evidence is excluded by default and work is bounded. |
| Providers | LLM and embedding integrations | External providers are optional; zero-key fallbacks remain functional. |
| Production persistence | `LocalBackend` | SQLite is the sole production source of truth. |
| Evaluation adapter | `Mem0Backend` | Optional and reduced; it is for interop/benchmarking, not a second production contract. |

The `MemoryBackend` interface remains useful for isolating persistence in tests and for the
Mem0 comparison adapter. It must not be interpreted as a promise that every adapter
supports every Memry feature. Silent default no-ops in that interface are listed as a
cleanup risk in [duplication-audit.md](duplication-audit.md).

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

### Topics

Topics are deterministic classification labels such as `health` or `finance`. Their
canonical storage is the normalized `topics` table plus the indexed `memory_topics` join.
The public field and API name remains `categories` for compatibility. During the migration
window, the JSON category list is also written as a compatibility projection; it is not a
second knowledge concept.

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

For `store.add(...)`:

1. Store raw input as episodes before inference.
2. With an LLM, extract small candidate memories, types, importance, topics, entities, and
   possible relations. Without an LLM, store the input verbatim.
3. Retrieve similar active memories in the same scope and reconcile each candidate as add,
   update, supersede, or no-op.
4. Store or update the memory, normalized topic links, embedding, and FTS row.
5. Resolve entity mentions conservatively. Alias matches only narrow the candidates.
6. Store evidence-grounded relations whose endpoints resolved in that memory.
7. Append audit events.

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
- MCP runs locally over stdio or remotely over streamable HTTP.
- `memry serve` hosts REST, the dashboard, OAuth endpoints when enabled, and `/mcp` in one
  Starlette/Uvicorn process.
- A single admin bearer key, configured tenants, or runtime accounts can authenticate
  network calls. The oldest/first runtime account is persisted as bootstrap administrator
  and uses the existing unconfined/default namespace; later accounts and tenant principals
  map public user IDs into confined namespaces.
- Runtime API keys are stored as SHA-256 hashes; human passwords use scrypt with a random
  salt; comparisons are constant-time.
- OAuth uses the MCP SDK authorization-server interfaces with dynamic client registration,
  PKCE, short-lived authorization codes, access/refresh tokens, refresh rotation, and
  revocation.
- The dashboard uses an HTTP-only session cookie. OAuth discovery is mounted at the domain
  root; MCP is mounted at `/mcp`.

Runtime accounts and OAuth state currently live in `auth.db`, separate from `memry.db`.
That is current behavior, not an assertion that two database files are ideal; the
consolidation option is recorded in the duplication audit.

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
| SQLite JSON1 | Yes during compatibility migration | Reads legacy category JSON and metadata aliases; normalized topic links are the indexed path. |
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
| HTTPX | Yes | OpenAI, Voyage, and Ollama provider HTTP calls; also used by tests. |
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
| Mem0 / `mem0ai` | Optional extra `memry[mem0]` | Reduced interop and comparison adapter, not a full production backend. |

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
- SQLite and `auth.db` backups must be coordinated; ANN sidecars may be discarded and
  rebuilt.
- The Mem0 adapter does not implement the complete knowledge contract.
- There is no external IdP/SSO integration, per-key rate limiter, queue, separate vector
  database, or distributed cache.
- Description faithfulness and end-to-end memory quality still require public evaluation;
  a clean schema and synthetic benchmarks do not establish "best in class" quality.
- Exact inline entity highlighting is deferred because mention surfaces do not provide
  unambiguous character spans. Reliable entity chips are the shipped navigation path.

## 9. Decision discipline

Consequential architecture changes must be recorded with product need, alternatives,
business benefit, costs, migration, and validation before implementation. Accepted
knowledge-model decisions and their live status are maintained in
[knowledge-model-features.md](knowledge-model-features.md). Potential simplifications are
listed without silently applying them in [duplication-audit.md](duplication-audit.md).