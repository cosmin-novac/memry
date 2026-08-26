# Connecting Memry to ChatGPT (custom connector / app)

This guide connects a self-hosted Memry to ChatGPT, so chats there recall and
save the same long-term memories your other agents use. ChatGPT signs in with
**OAuth** against Memry's own accounts - there is no key to paste.

For claude.ai see [connect-claude-ai.md](connect-claude-ai.md); for Claude
Code, Cursor, and other config-file clients see
[self-hosting.md](self-hosting.md).

Requirements:

- Memry deployed behind HTTPS (see [deploy-vps.md](deploy-vps.md)).
- `MEMRY_PUBLIC_URL` set to that HTTPS origin, e.g.
  `MEMRY_PUBLIC_URL=https://memory.example.com`. Without it Memry serves no
  OAuth endpoints at all and ChatGPT has nothing to sign in against.
- At least one account: `memry account add alice --password s3cret`.
- A ChatGPT plan that offers connectors / developer mode.

## Steps

1. In ChatGPT open **Settings → Connectors** (developer mode: **Settings →
   Plugins → Create**).
2. Give it a name (`Memry`) and enter the MCP server URL:

   ```
   https://memory.example.com/mcp
   ```

3. Choose **OAuth** as the authentication method. Leave client ID and secret
   empty: Memry supports Dynamic Client Registration, so ChatGPT registers
   itself.
4. Click **Connect**. A window opens on Memry's own sign-in page; enter the
   account name and password and approve.
5. The connector shows as connected, and the tools appear in chat.

## First run

- "Remember that I prefer short answers and deploy on Contabo." - ChatGPT
  calls `save_memories`.
- In a NEW chat: "What do you know about my preferences?" - it calls
  `search_memories` or `get_memory_context` and answers from your store.

Memories written here land in the same store as every other client, namespaced
to the account that signed in.

## Troubleshooting

**"There was a problem connecting Memry. Try again later."** - the browser
console shows a `424 Failed Dependency` from
`chatgpt.com/backend-api/aip/connectors/links/oauth/callback`. The 424 means
ChatGPT's own callback failed because something it depended on did: the OAuth
part usually succeeded, and the MCP handshake right after it did not. Check,
in order:

1. **The URL is the MCP endpoint.** `https://memory.example.com/mcp`. Memry
   `>= 0.2.28` also answers the handshake on the bare origin
   (`https://memory.example.com`), so an older server is the common cause -
   there, the bare origin serves the dashboard, which answers ChatGPT's POST
   with `405 Method Not Allowed`. Confirm with:

   ```bash
   curl -i -X POST https://memory.example.com/mcp \
     -H 'content-type: application/json' \
     -H 'accept: application/json, text/event-stream' \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
   ```

   A `401` with a `WWW-Authenticate: Bearer resource_metadata=...` header is
   the correct answer here - it is what starts the OAuth flow. A `405` means
   the URL is not an MCP endpoint on this server version.

2. **`MEMRY_PUBLIC_URL` is set, and matches the URL you typed** (scheme and
   host, no trailing slash). It is what the discovery documents advertise:

   ```bash
   curl https://memory.example.com/.well-known/oauth-authorization-server
   ```

   404 here means the variable is unset - set it in `/opt/memry/.env`, re-apply
   the compose file, and retry.

3. **An account exists.** `memry account list`. There is nothing to sign in as
   otherwise; the sign-in page rejects every attempt with "Wrong account or
   password".

**The sign-in page says the link expired** - pending authorization requests are
dropped after 15 minutes. Start the connection again from ChatGPT.

**Connected, but tools never fire** - make sure the connector is enabled for
the chat, and ask explicitly ("remember this", "what do you remember about
me") the first few times.
