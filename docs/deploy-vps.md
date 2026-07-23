# Deploying Memry on a VPS

One command on any fresh Ubuntu/Debian server (Contabo, Hetzner, DigitalOcean,
Linode, a home box...) gives you Memry behind Caddy with automatic HTTPS:

```bash
curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
  | MEMRY_DOMAIN=memory.example.com ANTHROPIC_API_KEY=sk-ant-... bash
```

The installer:

1. installs Docker if missing (via get.docker.com),
2. downloads Memry into `/opt/memry/app`,
3. generates a random `MEMRY_API_KEY` into `/opt/memry/.env` (chmod 600),
4. starts two containers - `memry` (the server) and `caddy` (TLS termination),
5. waits for `/health` and prints your URLs and API key.

Both env vars are optional: without `MEMRY_DOMAIN` it serves plain HTTP on
port 80 (fine for a first look, not for real use); without an LLM key Memry
stores verbatim and retrieves with BM25 + hash embeddings.

**Re-running the same command updates Memry in place.** Code in
`/opt/memry/app` is disposable; your config (`/opt/memry/.env`) and data
(Docker volumes) survive updates.

## Walkthrough: Contabo

1. **Create the VPS** - any plan works; the smallest (4 GB RAM) is plenty.
   Pick **Ubuntu 24.04** as the image.
2. **Optional, zero-SSH path**: in the order form, expand **Cloud-Init** and
   paste [`deploy/cloud-init.yaml`](../deploy/cloud-init.yaml) (edit the domain
   line first). The server boots straight into a running Memry.
3. **DNS**: create an A record for your domain/subdomain pointing at the VPS IP
   (Contabo shows it in the panel once provisioned). Do this before or right
   after install; Caddy retries certificate issuance automatically.
4. **Or install over SSH**:

   ```bash
   ssh root@<vps-ip>
   curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
     | MEMRY_DOMAIN=memory.example.com bash
   ```

5. **Verify**:

   ```bash
   curl https://memory.example.com/health
   # {"status": "ok", "service": "memry"}
   ```

The same steps work on any provider; only the panel names differ.

## Connecting your agents

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

For claude.ai (web/desktop/mobile custom connectors, which cannot send
headers) embed the key in the URL instead:
`https://memory.example.com/mcp/<MEMRY_API_KEY>` - full walkthrough in
[connect-claude-ai.md](connect-claude-ai.md).

The dashboard at `https://memory.example.com/` has a sign-in page: accounts use their
name and password, the admin uses the API key ("Sign in as admin").

## Changing configuration

Edit `/opt/memry/.env` (domain, LLM keys, `MEMRY_TENANTS` for multi-tenant
mode - see [self-hosting.md](self-hosting.md)), then apply:

```bash
docker compose --env-file /opt/memry/.env \
  -f /opt/memry/app/deploy/vps/docker-compose.yml up -d
```

## Multi-user & OAuth

To let other people connect as isolated accounts (each confined to its own
namespace) instead of sharing the admin key:

1. Set your public URL so OAuth turns on, then re-apply:

   ```bash
   echo 'MEMRY_PUBLIC_URL=https://memory.example.com' >> /opt/memry/.env
   docker compose --env-file /opt/memry/.env \
     -f /opt/memry/app/deploy/vps/docker-compose.yml up -d
   ```

2. Create an account (accounts live in `auth.db` on the same data volume, so
   they survive redeploys):

   ```bash
   docker compose --env-file /opt/memry/.env \
     -f /opt/memry/app/deploy/vps/docker-compose.yml \
     exec memry memry account add alice --password 's3cret'
   ```

   That prints an API key (shown once). The account holder can now either use
   that key directly (`Authorization: Bearer <key>`, or `/mcp/<key>`) or sign in
   through OAuth from a client that supports it, using the account name and
   password. There is no public signup: only keys and accounts you create here
   grant access. See [self-hosting.md](self-hosting.md#accounts-and-oauth).

Your existing admin-key connections keep working unchanged; enabling OAuth only
adds a second way in.

## Backups

All state is one SQLite file in the `memry_memry-data` volume:

```bash
docker compose --env-file /opt/memry/.env \
  -f /opt/memry/app/deploy/vps/docker-compose.yml \
  exec memry sh -c "memry export" > backup.jsonl
```

or snapshot the volume itself. A nightly cron line on the VPS is enough:

```bash
0 3 * * * docker compose --env-file /opt/memry/.env -f /opt/memry/app/deploy/vps/docker-compose.yml exec -T memry memry export > /root/memry-backup-$(date +\%F).jsonl
```

## Uninstall

```bash
docker compose --env-file /opt/memry/.env \
  -f /opt/memry/app/deploy/vps/docker-compose.yml down -v   # -v deletes data!
rm -rf /opt/memry
```

## Without the installer (manual / non-root / other distros)

The installer is convenience, not magic. Equivalent manual steps:

```bash
git clone https://github.com/cosmin-novac/memry && cd memry/deploy/vps
cp .env.example .env    # set MEMRY_API_KEY (and MEMRY_DOMAIN etc.)
docker compose up -d --build
```
