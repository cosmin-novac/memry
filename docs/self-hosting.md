# Self-hosting Memry

Memry is one Python process with SQLite and no external database, queue, or vector service.
Knowledge lives in `memry.db`; runtime accounts and OAuth live in the adjacent `auth.db`.
Keeping them separate prevents knowledge restore/reset operations from changing login data.
A complete server backup must include both files.

## Option 1 - bare (recommended for personal use)

```bash
pip install memry
memry serve --host 0.0.0.0 --port 8787
```

- Dashboard: `http://<host>:8787/`
- REST API: `http://<host>:8787/api/v1/...`
- MCP (streamable HTTP): `http://<host>:8787/mcp`
- Knowledge data: `~/.memry/memry.db` (override with `MEMRY_DB_PATH`)
- Login data: `~/.memry/auth.db` when accounts/OAuth are used (override with `MEMRY_AUTH_DB_PATH`)

## Option 2 - Docker

```bash
docker compose up -d
```

The compose file mounts a named volume at `/data` and reads the same `MEMRY_*`
environment variables. See [`docker-compose.yml`](../docker-compose.yml).

## Option 3 - VPS, one command (Docker + automatic HTTPS)

On a fresh Ubuntu/Debian server at any provider (Contabo, Hetzner,
DigitalOcean, ...):

```bash
curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
  | MEMRY_DOMAIN=memory.example.com bash
```

Installs Docker, builds Memry, puts Caddy in front for automatic HTTPS, and
generates a `MEMRY_API_KEY`. Re-run the same command to update. Full
walkthrough (cloud-init, DNS, backups, uninstall): [deploy-vps.md](deploy-vps.md).

## Securing the server

1. **Set an API key** - `MEMRY_API_KEY=<random>` requires
   `Authorization: Bearer <key>` on `/api/*`. Create the first runtime account for the human
   administrator. Every dashboard user signs in at `/login` with an account name and password;
   the HttpOnly session cookie remains confined to that account's memories. Programmatic and
   recovery clients keep using the operator bearer key.
2. **Bind privately** - without a key, keep `--host 127.0.0.1` or terminate TLS + auth in a
   reverse proxy (Caddy/Traefik/nginx).
3. **Backups** - a complete server backup must capture `memry.db` and `auth.db`
   together, including any live SQLite `-wal`/`-shm` files. A directory/volume snapshot
   does that. `memry export` is a lossless knowledge backup, but it does not include
   accounts, sessions, OAuth clients, or tokens from `auth.db`.

## Multi-tenant mode

Serve several teams or customers from one Memry server, each with their own API key and an
isolated memory space:

```bash
export MEMRY_API_KEY="admin-key-with-global-access"     # optional but recommended
export MEMRY_TENANTS='[{"name":"acme","api_key":"acme-secret"},
                       {"name":"globex","api_key":"globex-secret"}]'
memry serve --host 0.0.0.0
```

(or the same under `"tenants": [...]` in `~/.memry/config.json`.)

How isolation works:

- A tenant's requests are transparently namespaced: user `u1` under key `acme-secret`
  reads and writes `acme::u1`. Tenants never see, guess, or address each other's data;
  cross-tenant access by memory/entity id returns 404.
- `GET /api/v1/stats` returns per-tenant counts for tenant keys, global stats for the
  admin key.
- MCP over HTTP (`/mcp`) accepts tenant keys too, with the same confinement: a tool's
  `user_id` argument selects a namespace *under* the calling tenant, so asking for
  another tenant's namespace lands in your own rather than reaching theirs. `/mcp/<key>`
  works for tenant keys as well as the admin key. Local stdio MCP (`memry mcp`) has no
  auth and stays single-user.

Keys live in config on your infrastructure; treat the config file like a secret.

Tenants are fixed in config. For a server where people sign themselves up and connect from
any MCP client, use **accounts** instead.

## Accounts and OAuth

Accounts are runtime-managed identities (not config). The first account is the bootstrap
administrator and keeps only the server's existing `default` memory space. Every later
account gets one private `<account>::default` space. The administrator role does not grant
access to other accounts' memories. Accounts are reachable either with an API key or through
a real OAuth login from clients like Claude Code, Cursor, or VS Code. `MEMRY_API_KEY` remains
the separate operator credential with global access and should be protected accordingly.

Create accounts with the CLI:

```bash
memry account add alice --password s3cret   # prints an API key (shown once)
memry account list
memry account issue-key alice --label laptop
memry account disable alice                 # keys and tokens stop working immediately
```

Accounts live in `auth.db` next to your memory database (override with `MEMRY_AUTH_DB_PATH`).
Back it up together with `memry.db`; a knowledge export alone cannot restore accounts or
OAuth state. An account's API key works on both `/api` and `/mcp`, including the
`/mcp/<key>` URL form.

**OAuth.** Set a public URL and Memry becomes an OAuth 2.1 authorization server for its own
accounts:

```bash
export MEMRY_PUBLIC_URL="https://memory.example.com"
memry serve --host 0.0.0.0
```

That turns on, at the domain root:

- `/.well-known/oauth-authorization-server` and
  `/.well-known/oauth-protected-resource/mcp` - discovery documents clients read first.
- `/register` (Dynamic Client Registration), `/authorize`, `/token`, `/revoke` - the flow,
  with PKCE required and refresh-token rotation.
- `/oauth/login` - Memry's own sign-in and consent page, where the account name and password
  are entered.

A client pointed at `https://memory.example.com/mcp` with no key now discovers the
authorization server (via the `WWW-Authenticate` header on the 401), registers itself, sends
the user through login, and receives a token scoped to that account. No key to copy by hand.
Memry verifies the human against its own accounts, so no third-party IdP is required.

## Scaling up

| Situation | Setting |
|---|---|
| Faster vector search past ~5k memories | `pip install "memry[ann]"` - a usearch HNSW sidecar supplies candidates above the configured threshold; `memry reindex` rebuilds it |
| Many agents or devices sharing memory | Point every client at the same `memry serve` URL. They share one server process and one SQLite store. |
| Several server replicas or machines writing one store | Unsupported. Do not point multiple Memry processes at the same database file. This would require a separately reviewed storage architecture. |

## Connecting agents to a shared server

Multiple agents on multiple machines can share one memory server over MCP HTTP:

```jsonc
{
  "mcpServers": {
    "memry": {
      "type": "http",
      "url": "https://memory.example.com/mcp",
      "headers": { "Authorization": "Bearer <MEMRY_API_KEY>" }
    }
  }
}
```

Clients that cannot send headers (claude.ai custom connectors) may embed the
admin key in the URL instead: `https://memory.example.com/mcp/<MEMRY_API_KEY>`
(or `/mcp?key=...`). See [connect-claude-ai.md](connect-claude-ai.md).

For local single-machine use, prefer stdio (`memry mcp`) - no port, no auth surface.

## Provider configuration

| Goal | Setting |
|---|---|
| Best extraction quality | `ANTHROPIC_API_KEY` + `pip install "memry[anthropic]"` (default model `claude-opus-4-8`) |
| Cheaper extraction | `MEMRY_LLM_MODEL=claude-haiku-4-5` |
| OpenAI end-to-end | `OPENAI_API_KEY` (LLM `gpt-5-mini`, embeddings `text-embedding-3-small`) |
| Fully offline | `MEMRY_LLM_PROVIDER=ollama` + `MEMRY_EMBEDDING_PROVIDER=ollama` (e.g. `llama3.1`, `nomic-embed-text`) |
| Zero keys, zero model downloads | nothing - verbatim writes + BM25/hash retrieval |

After switching embedding providers, run `memry reindex` once to re-embed the store.

## Maintenance

```bash
memry sweep --threshold 0.1   # soft-forget stale, low-importance memories
memry stats                   # counts, providers, db path
memry export > backup.json    # knowledge only: IDs, provenance, entities, relations, history
memry abstract-tags           # LLM clusters tags into higher-level ones now
```

When accounts or OAuth are enabled, also back up `auth.db` with `memry.db`. The JSON export
does not contain login data.

A weekly `sweep` in cron/Task Scheduler keeps long-running stores lean; forgotten memories
are invalidated (auditable, recoverable), never destroyed.

## Managing topics and entities

The dashboard's **Knowledge** area contains Topics, People and things, Relations, and
Collections. Topics show memory counts and can be renamed, combined, or deleted under the
current user filter. The same topic operations remain available at
`POST /api/v1/tags/edit` for API compatibility.

An optional, off-by-default LLM pass proposes higher-level parents such as `health` over
`running` and `sleep` (`MEMRY_TAG_ABSTRACTION=on`,
`MEMRY_TAG_ABSTRACTION_INTERVAL_DAYS=7`, or `memry abstract-tags`). Memry stores hierarchy
edges and expands a parent filter at query time; it does not copy the parent label onto each
memory. Synthetic parents remain visible through `/api/v1/categories` and
`GET /api/v1/tags/synthetic`.

People and things open as entity hubs with aliases, a bounded description, and active
supporting memories. Relations can open their evidence memory, and collections link to their
members.

## Searching by tag and date

Beyond relevance search, both `search_memories`/`POST /api/v1/search` and
`list_memories`/`GET /api/v1/memories` accept a `categories` (tag) filter and a `since`/
`until` date window (`YYYY-MM-DD`, the `until` day inclusive). Pass an empty query with just
a tag or date to browse rather than rank, e.g. "everything tagged `travel` since 2026-01-01".
