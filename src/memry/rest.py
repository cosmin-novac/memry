"""Self-hosting server: REST API + minimal dashboard + MCP over HTTP.

    memry serve --host 0.0.0.0 --port 8787

Endpoints:
    GET  /                       dashboard
    GET  /health
    GET  /api/v1/memories        ?user_id=&limit=&include_invalid=
    POST /api/v1/memories        {content|messages, user_id?, infer?, ...}
    GET  /api/v1/memories/{id}
    PATCH  /api/v1/memories/{id} {content?, importance?, categories?}
    DELETE /api/v1/memories/{id} ?hard=true
    GET  /api/v1/memories/{id}/history
    POST /api/v1/search          {query, user_id?, limit?}
    POST /api/v1/context         {query, user_id?, token_budget?}
    GET  /api/v1/stats
    /mcp                         MCP streamable-HTTP endpoint

Auth: set MEMRY_API_KEY to require ``Authorization: Bearer <key>`` on /api
and /mcp. Without it the server is open - bind to localhost or a private net.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from .mcp_server import create_server
from .store import MemoryStore

_DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memry dashboard</title>
<style>
:root{--bg:#0b0e14;--panel:#141a24;--line:#232c3b;--text:#dbe4f0;--dim:#8494ab;--accent:#5eead4;--warn:#f0a35e;font-size:15px}
@media (prefers-color-scheme: light){:root{--bg:#f5f7fa;--panel:#ffffff;--line:#dde4ee;--text:#1a2333;--dim:#5c6b82;--accent:#0d9488;--warn:#b45309}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:2rem 1rem}
h1{font-size:1.3rem;margin:.2rem 0 1rem}h1 span{color:var(--accent)}
.bar{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
input,button,textarea{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.5rem .7rem}
input:focus,textarea:focus{outline:2px solid var(--accent);outline-offset:-1px}
button{cursor:pointer}button.primary{background:var(--accent);color:#04211c;border-color:transparent;font-weight:600}
#stats{color:var(--dim);font-size:.85rem;margin-bottom:1rem}
.mem{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin-bottom:.5rem}
.mem .meta{color:var(--dim);font-size:.78rem;margin-top:.35rem;display:flex;gap:.8rem;flex-wrap:wrap}
.mem .del{float:right;border:none;background:none;color:var(--dim)}.mem .del:hover{color:var(--warn)}
.tag{border:1px solid var(--line);border-radius:999px;padding:0 .5rem}
textarea{width:100%;min-height:70px;margin-bottom:.4rem}
.empty{color:var(--dim);padding:2rem;text-align:center}
</style></head><body><main>
<h1><span>Mem</span>ry <small style="color:var(--dim);font-weight:400">memory dashboard</small></h1>
<div id="stats">loading…</div>
<div class="bar">
  <input id="user" placeholder="user_id (default)" style="width:11rem">
  <input id="q" placeholder="search memories…" style="flex:1;min-width:12rem">
  <button class="primary" onclick="search()">Search</button>
  <button onclick="loadAll()">All</button>
</div>
<textarea id="newmem" placeholder="Add to memory… (extraction runs if an LLM is configured)"></textarea>
<div class="bar"><button class="primary" onclick="add(true)">Add (infer)</button>
<button onclick="add(false)">Add verbatim</button></div>
<div id="list"></div>
</main><script>
const key = localStorage.getItem('memry_key') || '';
const H = key ? {'Authorization':'Bearer '+key,'Content-Type':'application/json'} : {'Content-Type':'application/json'};
const uid = () => document.getElementById('user').value.trim() || 'default';
async function api(path, opts={}){
  const r = await fetch(path,{headers:H,...opts});
  if(r.status===401){const k=prompt('API key required');if(k){localStorage.setItem('memry_key',k);location.reload();}throw new Error('unauthorized');}
  return r.json();
}
function esc(s){return (s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function render(items){
  const el=document.getElementById('list');
  if(!items.length){el.innerHTML='<div class="empty">No memories yet.</div>';return}
  el.innerHTML=items.map(m=>`<div class="mem"><button class="del" title="forget" onclick="del('${m.id}')">✕</button>
   <div>${esc(m.content)}</div>
   <div class="meta"><span class="tag">${m.memory_type||m.type||'semantic'}</span>
   <span>imp ${(m.importance??0.5).toFixed(2)}</span>
   ${m.score!==undefined?`<span>score ${m.score.toFixed(3)}</span>`:''}
   <span>${(m.updated_at||m.created_at||'').slice(0,10)}</span>
   ${m.invalid_at?'<span style="color:var(--warn)">invalidated</span>':''}</div></div>`).join('');
}
async function loadAll(){render(await api('/api/v1/memories?user_id='+encodeURIComponent(uid())+'&limit=100'))}
async function search(){
  const q=document.getElementById('q').value.trim(); if(!q)return loadAll();
  const rs=await api('/api/v1/search',{method:'POST',body:JSON.stringify({query:q,user_id:uid(),limit:20})});
  render(rs.map(r=>({...r.memory,score:r.score})));
}
async function add(infer){
  const t=document.getElementById('newmem').value.trim(); if(!t)return;
  await api('/api/v1/memories',{method:'POST',body:JSON.stringify({content:t,user_id:uid(),infer})});
  document.getElementById('newmem').value=''; loadAll(); loadStats();
}
async function del(id){await api('/api/v1/memories/'+id,{method:'DELETE'}); loadAll(); loadStats();}
async function loadStats(){
  const s=await api('/api/v1/stats');
  document.getElementById('stats').textContent=
    `${s.active_memories??'?'} active memories · ${s.invalidated_memories??0} invalidated · `+
    `${s.episodes??0} episodes · backend ${s.backend} · llm ${s.llm} · embeddings ${s.embedder}`;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
loadStats(); loadAll();
</script></body></html>"""


def create_app(store: MemoryStore | None = None) -> Starlette:
    store = store or MemoryStore()
    api_key = store.config.api_key
    tenants = {t.api_key: t.name for t in store.config.tenants}
    default_user = store.config.default_user_id
    mcp = create_server(store)
    mcp.settings.streamable_http_path = "/"
    mcp_app = mcp.streamable_http_app()

    def _unauthorized() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    def _resolve_auth(request: Request) -> tuple[bool, str | None]:
        """Returns (authorized, tenant_name). tenant_name None = admin/open."""
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not api_key and not tenants:
            return True, None  # open mode: bind privately or set keys
        if api_key and token == api_key:
            return True, None
        if token in tenants:
            return True, tenants[token]
        return False, None

    def _ns(tenant: str | None, user_id: str | None) -> str | None:
        """Tenant requests are transparently namespaced: they can only ever
        read or write user ids of the form ``<tenant>::<user>``."""
        if tenant is None:
            return user_id
        return f"{tenant}::{user_id or default_user}"

    def _owns(tenant: str | None, owner_user_id: str | None) -> bool:
        if tenant is None:
            return True
        return bool(owner_user_id) and owner_user_id.startswith(f"{tenant}::")

    def guarded(handler):
        async def wrapper(request: Request) -> Response:
            ok, tenant = _resolve_auth(request)
            if not ok:
                return _unauthorized()
            request.state.tenant = tenant
            return await handler(request)

        return wrapper

    # -- handlers ---------------------------------------------------------
    async def dashboard(request: Request) -> Response:
        return HTMLResponse(_DASHBOARD)

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "memry"})

    def _memory_or_error(request: Request) -> tuple[Any, Response | None]:
        memory = store.get(request.path_params["memory_id"])
        if memory is None or not _owns(request.state.tenant, memory.user_id):
            return None, JSONResponse({"error": "not found"}, status_code=404)
        return memory, None

    def _parse_categories(value: Any) -> list[str] | None:
        if not value:
            return None
        if isinstance(value, str):
            return [c.strip() for c in value.split(",") if c.strip()]
        return [str(c) for c in value]

    async def list_memories(request: Request) -> Response:
        q = request.query_params
        memories = store.get_all(
            user_id=_ns(request.state.tenant, q.get("user_id")),
            agent_id=q.get("agent_id"),
            run_id=q.get("run_id"),
            include_invalid=q.get("include_invalid") == "true",
            limit=int(q.get("limit", "100")),
            offset=int(q.get("offset", "0")),
            categories=_parse_categories(q.get("categories")),
        )
        return JSONResponse([m.model_dump() for m in memories])

    async def create_memory(request: Request) -> Response:
        body = await request.json()
        content = body.get("messages") or body.get("content")
        if not content:
            return JSONResponse({"error": "content or messages required"}, status_code=400)
        result = store.add(
            content,
            user_id=_ns(request.state.tenant, body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            metadata=body.get("metadata"),
            infer=bool(body.get("infer", True)),
            memory_type=body.get("memory_type", "semantic"),
            importance=float(body.get("importance", 0.5)),
            categories=body.get("categories"),
        )
        return JSONResponse(result.model_dump(), status_code=201)

    async def get_memory(request: Request) -> Response:
        memory, error = _memory_or_error(request)
        return error or JSONResponse(memory.model_dump())

    async def patch_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        body = await request.json()
        memory = store.update(
            request.path_params["memory_id"],
            content=body.get("content"),
            importance=body.get("importance"),
            categories=body.get("categories"),
            metadata=body.get("metadata"),
        )
        return JSONResponse(memory.model_dump())

    async def delete_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        hard = request.query_params.get("hard") == "true"
        store.delete(request.path_params["memory_id"], hard=hard)
        return JSONResponse({"deleted": True, "hard": hard})

    async def memory_history(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        events = store.history(request.path_params["memory_id"])
        return JSONResponse([e.model_dump() for e in events])

    async def search(request: Request) -> Response:
        body = await request.json()
        results = store.search(
            body.get("query", ""),
            user_id=_ns(request.state.tenant, body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            limit=int(body.get("limit", 10)),
            include_invalid=bool(body.get("include_invalid", False)),
            categories=_parse_categories(body.get("categories")),
        )
        return JSONResponse(
            [
                {"memory": r.memory.model_dump(), "score": r.score, "signals": r.signals}
                for r in results
            ]
        )

    async def context(request: Request) -> Response:
        body = await request.json()
        ctx = store.reconstruct_context(
            body.get("query", ""),
            user_id=_ns(request.state.tenant, body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            token_budget=int(body.get("token_budget", 1200)),
        )
        return JSONResponse(ctx.model_dump())

    async def stats(request: Request) -> Response:
        data = store.stats()
        if request.state.tenant is not None:
            prefix = f"{request.state.tenant}::"
            mine = [
                m for m in store.get_all(include_invalid=True, limit=100_000)
                if (m.user_id or "").startswith(prefix)
            ]
            data = {
                "backend": data.get("backend"),
                "tenant": request.state.tenant,
                "active_memories": sum(1 for m in mine if m.invalid_at is None),
                "invalidated_memories": sum(1 for m in mine if m.invalid_at is not None),
            }
        return JSONResponse(json.loads(json.dumps(data, default=str)))

    # -- entities ---------------------------------------------------------
    async def list_entities(request: Request) -> Response:
        q = request.query_params
        entities = store.entities(
            user_id=_ns(request.state.tenant, q.get("user_id")),
            include_merged=q.get("include_merged") == "true",
            limit=int(q.get("limit", "100")),
        )
        return JSONResponse([e.model_dump() for e in entities])

    async def get_entity(request: Request) -> Response:
        detail = store.entity(request.path_params["entity_id"])
        if detail is None or not _owns(request.state.tenant, detail["entity"].user_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "entity": detail["entity"].model_dump(),
                "mentions": [m.model_dump() for m in detail["mentions"]],
                "memories": [m.model_dump() for m in detail["memories"]],
            }
        )

    async def list_proposals(request: Request) -> Response:
        q = request.query_params
        proposals = store.merge_proposals(
            user_id=_ns(request.state.tenant, q.get("user_id")),
            status=q.get("status", "proposed"),
            limit=int(q.get("limit", "100")),
        )
        return JSONResponse([p.model_dump() for p in proposals])

    def _proposal_guard(request: Request):
        proposal = store.backend.get_proposal(request.path_params["proposal_id"])
        if proposal is None or not _owns(request.state.tenant, proposal.user_id):
            return None, JSONResponse({"error": "not found"}, status_code=404)
        return proposal, None

    async def confirm_proposal(request: Request) -> Response:
        proposal, error = _proposal_guard(request)
        if error:
            return error
        ok = store.confirm_merge(proposal.id)
        return JSONResponse({"confirmed": ok, "proposal_id": proposal.id})

    async def reject_proposal(request: Request) -> Response:
        proposal, error = _proposal_guard(request)
        if error:
            return error
        ok = store.reject_merge(proposal.id)
        return JSONResponse({"rejected": ok, "proposal_id": proposal.id})

    async def resolve_entities_route(request: Request) -> Response:
        body = await request.json() if await request.body() else {}
        outcome = store.resolve_entities(
            user_id=_ns(request.state.tenant, body.get("user_id"))
        )
        return JSONResponse(outcome)

    async def guarded_mcp_app(scope, receive, send):
        """MCP over HTTP: open in open mode; admin-key-only once any auth is
        configured (per-tenant scoping inside MCP tool calls isn't enforced
        yet, so tenant keys are REST-only by design)."""
        if api_key or tenants:
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            token = headers.get("authorization", "")
            token = token[7:].strip() if token.lower().startswith("bearer ") else ""
            if not (api_key and token == api_key):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(
                    scope, receive, send
                )
                return
        await mcp_app(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            yield

    routes = [
        Route("/", dashboard),
        Route("/health", health),
        Route("/api/v1/memories", guarded(list_memories), methods=["GET"]),
        Route("/api/v1/memories", guarded(create_memory), methods=["POST"]),
        Route("/api/v1/memories/{memory_id}", guarded(get_memory), methods=["GET"]),
        Route("/api/v1/memories/{memory_id}", guarded(patch_memory), methods=["PATCH"]),
        Route("/api/v1/memories/{memory_id}", guarded(delete_memory), methods=["DELETE"]),
        Route("/api/v1/memories/{memory_id}/history", guarded(memory_history), methods=["GET"]),
        Route("/api/v1/search", guarded(search), methods=["POST"]),
        Route("/api/v1/context", guarded(context), methods=["POST"]),
        Route("/api/v1/stats", guarded(stats), methods=["GET"]),
        Route("/api/v1/entities", guarded(list_entities), methods=["GET"]),
        Route("/api/v1/entities/proposals", guarded(list_proposals), methods=["GET"]),
        Route(
            "/api/v1/entities/proposals/{proposal_id}/confirm",
            guarded(confirm_proposal), methods=["POST"],
        ),
        Route(
            "/api/v1/entities/proposals/{proposal_id}/reject",
            guarded(reject_proposal), methods=["POST"],
        ),
        Route("/api/v1/entities/resolve", guarded(resolve_entities_route), methods=["POST"]),
        Route("/api/v1/entities/{entity_id}", guarded(get_entity), methods=["GET"]),
        Mount("/mcp", app=guarded_mcp_app),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def main(host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
