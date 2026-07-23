# Connecting Memry to claude.ai (custom connector)

This guide connects a self-hosted Memry to Claude on the web, desktop, and
mobile apps, so every Claude chat can recall and save long-term memories.
For Claude Code, Cursor, and other clients that support MCP config files, see
[self-hosting.md](self-hosting.md) - they connect with a Bearer header instead.

Requirements:

- Memry `>= 0.2.2` deployed behind HTTPS (see [deploy-vps.md](deploy-vps.md)).
- Your `MEMRY_API_KEY`. On a VPS install: `grep MEMRY_API_KEY /opt/memry/.env`
- A claude.ai plan with custom connectors (Pro, Max, Team, or Enterprise).

## Why the key goes in the URL

The claude.ai **Add custom connector** dialog accepts a URL plus optional
OAuth credentials. There is no field for an `Authorization` header, which is
how Memry normally authenticates MCP over HTTP. Memry therefore also accepts
the admin key embedded in the MCP URL:

```
https://memory.example.com/mcp/<MEMRY_API_KEY>
```

A query-param form (`/mcp?key=<MEMRY_API_KEY>`) works too; prefer the path
form, as some clients strip query strings. Both are equivalent to sending
`Authorization: Bearer <MEMRY_API_KEY>`.

## Steps

1. In claude.ai open **Settings → Connectors** (on desktop:
   Claude menu → Settings → Connectors).
2. Click **Add custom connector**.
3. Fill the dialog:
   - **Name**: `Memry`
   - **Remote MCP server URL**: `https://memory.example.com/mcp/<MEMRY_API_KEY>`
   - Leave **OAuth Client ID** and **OAuth Client Secret** empty.
4. Click **Add**. Claude performs the MCP handshake; the connector should show
   as connected.
5. In a chat, open the search-and-tools menu (sliders icon) and make sure
   **Memry** is enabled. Tool permissions can be adjusted per tool.

## First run

Try these in a chat:

- "Remember that I prefer short answers and I deploy on Contabo."
  Claude calls `save_memories`; facts are extracted and reconciled.
- In a NEW chat: "What do you know about my preferences?"
  Claude calls `search_memories` or `get_memory_context` and answers from
  your memory store.

The connector exposes the full toolset: `save_memories`, `search_memories`,
`get_memory_context`, `list_memories`, `update_memory`, `delete_memory`,
`memory_history`, `memory_stats`. Everything lands in the same store your
other agents use, so what Claude Code learned yesterday, claude.ai knows today.

## Security notes

- The URL now contains your admin key. Treat the connector URL like a
  password: don't paste it in shared chats or screenshots.
- Keys in URLs can end up in intermediary logs. Memry's bundled Caddy does not
  log requests by default; if you front Memry with your own proxy, disable
  access-log query/path capture for `/mcp` or accept the tradeoff.
- To rotate the key: edit `MEMRY_API_KEY` in `/opt/memry/.env`, re-apply
  (`docker compose --env-file /opt/memry/.env -f
  /opt/memry/app/deploy/vps/docker-compose.yml up -d`), then update the
  connector URL in claude.ai.
- Memories are namespaced per user id; MCP writes under the admin key default to
  `MEMRY_DEFAULT_USER_ID`. If several people share one server, give each their
  own tenant key and hand out `https://memory.example.com/mcp/<TENANT_KEY>`:
  tenant keys are accepted on `/mcp` and confined to their own namespace, so
  one deployment can serve several people without them seeing each other.

## Troubleshooting

- **"Could not connect" right after clicking Add**: confirm
  `curl https://memory.example.com/health` returns ok, and that the URL ends
  with `/mcp/<key>` (no trailing slash needed).
- **401 unauthorized**: wrong key, or the server predates URL-key support.
  Update in place by re-running the installer command from
  [deploy-vps.md](deploy-vps.md) - config and data survive.
- **421 Misdirected Request**: server predates v0.2.1 (DNS-rebinding
  protection rejecting proxied Host headers). Update as above.
- **Tools never fire**: check the connector is enabled in the chat's tools
  menu, and ask explicitly ("remember this", "what do you remember about me")
  the first few times.
