# Self-hosting Memry

Memry is a single Python process with a single SQLite file. There is nothing else to
operate - no vector database, no queue, no Postgres.

## Option 1 - bare (recommended for personal use)

```bash
pip install memry
memry serve --host 0.0.0.0 --port 8787
```

- Dashboard: `http://<host>:8787/`
- REST API: `http://<host>:8787/api/v1/...`
- MCP (streamable HTTP): `http://<host>:8787/mcp`
- Data: `~/.memry/memry.db` (override with `MEMRY_DB_PATH`)

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
   `Authorization: Bearer <key>` on `/api/*`. The dashboard prompts for it once and stores
   it in localStorage.
2. **Bind privately** - without a key, keep `--host 127.0.0.1` or terminate TLS + auth in a
   reverse proxy (Caddy/Traefik/nginx).
3. **Backups** - the entire state is one SQLite file; snapshot it (plus `-wal`/`-shm`) or
   use `memry export` for JSONL.

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
- MCP over HTTP (`/mcp`) accepts only the admin key while auth is configured; tenant keys
  are REST-scoped. Local stdio MCP (`memry mcp`) is single-user and unaffected.

Keys live in config on your infrastructure; treat the config file like a secret.

## Scaling up

| Situation | Setting |
|---|---|
| Faster vector search past ~5k memories | `pip install "memry[ann]"` - a usearch HNSW sidecar kicks in automatically (threshold configurable via `ann.min_rows`); `memry reindex` rebuilds it |
| Multiple server processes / machines writing one store | `pip install "memry[postgres]"`, then `MEMRY_BACKEND=postgres` and `MEMRY_POSTGRES_DSN=postgresql://...` (needs pgvector; use the `pgvector/pgvector` Docker image or any managed Postgres) |
| One process, one machine | keep the default SQLite backend - it is the simplest and plenty fast |

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
memry export > backup.jsonl   # JSONL backup (memories incl. invalidated)
```

A weekly `sweep` in cron/Task Scheduler keeps long-running stores lean; forgotten memories
are invalidated (auditable, recoverable), never destroyed.
