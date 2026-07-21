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
Clients that cannot send headers (claude.ai custom connectors) may embed the
admin key in the MCP URL instead: ``/mcp/<key>`` or ``/mcp?key=<key>``.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any
from urllib.parse import parse_qs

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
button.toggle[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
#stats{color:var(--dim);font-size:.85rem;margin-bottom:1rem}
.mem{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin-bottom:.5rem}
.mem .meta{color:var(--dim);font-size:.78rem;margin-top:.35rem;display:flex;gap:.8rem;flex-wrap:wrap}
.mem .del,.mem .edit{float:right;border:none;background:none;color:var(--dim)}.mem .del:hover{color:var(--warn)}.mem .edit:hover{color:var(--accent)}
.tag{border:1px solid var(--line);border-radius:999px;padding:0 .5rem}
.meta button.distill{border:1px solid var(--warn);border-radius:999px;padding:0 .5rem;background:none;color:var(--warn);font-size:inherit}
.meta button.distill:hover{background:var(--warn);color:var(--bg)}
textarea{width:100%;min-height:70px;margin-bottom:.4rem}
.empty{color:var(--dim);padding:2rem;text-align:center}
#mapwrap{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.4rem .4rem 0;margin-bottom:.7rem}
#map{display:block;width:100%}
.maphint{color:var(--dim);font-size:.75rem;text-align:center;margin:.15rem 0 .35rem}
</style></head><body><main>
<h1><span>Mem</span>ry <small style="color:var(--dim);font-weight:400">memory dashboard</small></h1>
<div id="stats">loading…</div>
<div class="bar">
  <input id="user" placeholder="user_id (all)" style="width:11rem">
  <input id="q" placeholder="search memories…" style="flex:1;min-width:12rem">
  <button class="primary" onclick="search()">Search</button>
  <button onclick="activeCat=null;loadAll()">All</button>
  <button class="toggle" id="addbtn" onclick="togglePanel('add')">+ Add</button>
  <button class="toggle" id="mapbtn" onclick="togglePanel('map')">Map</button>
  <button onclick="exportMemories()" title="Download memories as JSONL (respects the user_id filter; same format as `memry export`)">Export</button>
  <button id="importbtn" onclick="document.getElementById('importfile').click()" title="Additive import from a JSON/JSONL export; nothing is deleted">Import</button>
</div>
<input type="file" id="importfile" accept=".json,.jsonl,.txt,application/json" hidden onchange="importMemories(this.files[0]);this.value=''">
<div id="addpanel" hidden>
<textarea id="newmem" placeholder="Add to memory… (extraction runs if an LLM is configured)"></textarea>
<div class="bar">
  <input id="newcats" placeholder="categories, comma separated (optional)" style="flex:1;min-width:10rem">
  <button class="primary" onclick="add(true)">Add (infer)</button>
  <button onclick="add(false)">Add verbatim</button>
</div>
</div>
<div id="mapwrap" hidden><canvas id="map"></canvas><p class="maphint" id="maphint"></p></div>
<div id="list"></div>
</main><script>
const key = localStorage.getItem('memry_key') || '';
const H = key ? {'Authorization':'Bearer '+key,'Content-Type':'application/json'} : {'Content-Type':'application/json'};
// Empty box = no user filter: list EVERY namespace, so the list always
// agrees with the stats line. (Tenant keys are namespaced server-side.)
const uid = () => document.getElementById('user').value.trim();
let askedKey=false;
async function api(path, opts={}){
  const r = await fetch(path,{headers:H,...opts});
  if(r.status===401){
    // Concurrent boot requests (stats + list) may both 401; prompt only once.
    if(!askedKey){askedKey=true;const k=prompt('API key required');if(k){localStorage.setItem('memry_key',k);location.reload();}}
    throw new Error('unauthorized');
  }
  return r.json();
}
function esc(s){return (s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
// Add form and category map are opt-in panels; the choice sticks per browser.
const panels={add:localStorage.getItem('memry_show_add')==='1',map:localStorage.getItem('memry_show_map')==='1'};
function syncPanels(){
  document.getElementById('addpanel').hidden=!panels.add;
  document.getElementById('addbtn').setAttribute('aria-pressed',panels.add);
  document.getElementById('mapbtn').setAttribute('aria-pressed',panels.map);
  drawMap(current);
}
function togglePanel(name){
  panels[name]=!panels[name];
  localStorage.setItem('memry_show_'+name,panels[name]?'1':'0');
  syncPanels();
}
let current=[],activeCat=null,mapGraph=null,haveMore=false,editingId=null;
const cats=m=>((m.categories&&m.categories.length)?m.categories:['(untagged)']).map(c=>String(c).toLowerCase());
function render(items){
  current=items; drawMap(items);
  const shown=activeCat?items.filter(m=>cats(m).includes(activeCat)):items;
  const el=document.getElementById('list');
  if(!shown.length&&!haveMore){el.innerHTML='<div class="empty">'+(activeCat?'No memories under #'+esc(activeCat)+'.':'No memories yet.')+'</div>';return}
  el.innerHTML=shown.map(m=>m.id===editingId?editCard(m):viewCard(m)).join('')
   +(haveMore?'<div class="bar"><button onclick="loadAll(true)">Load more</button></div>':'');
}
function viewCard(m){
  return `<div class="mem"><button class="del" title="forget" onclick="del('${m.id}')">✕</button>
   <button class="edit" title="edit" onclick="startEdit('${m.id}')">✎</button>
   <div>${esc(m.content)}</div>
   <div class="meta"><span class="tag">${m.memory_type||m.type||'semantic'}</span>
   ${(m.categories||[]).map(c=>`<span class="tag">#${esc(String(c))}</span>`).join('')}
   <span>@${esc(m.user_id||'(no user)')}</span>
   <span>imp ${(m.importance??0.5).toFixed(2)}</span>
   ${m.score!==undefined?`<span>score ${m.score.toFixed(3)}</span>`:''}
   <span>${(m.updated_at||m.created_at||'').slice(0,10)}</span>
   ${m.invalid_at?'<span style="color:var(--warn)">invalidated</span>':''}
   ${m.metadata&&m.metadata.pending_distillation&&!m.invalid_at?`<button class="distill" onclick="distill('${m.id}')" title="Saved verbatim because extraction was skipped (no LLM or LLM error). Distill into discrete facts now.">not distilled ↻</button>`:''}</div></div>`;
}
function editCard(m){
  return `<div class="mem">
   <textarea id="edit-note">${esc(m.content)}</textarea>
   <div class="bar" style="margin-bottom:0">
     <input id="edit-cats" value="${esc((m.categories||[]).join(', '))}" placeholder="tags, comma separated (optional)" style="flex:1;min-width:8rem">
     <button class="primary" onclick="saveEdit('${m.id}')">Save</button>
     <button onclick="cancelEdit()">Cancel</button>
   </div></div>`;
}

// ---- category map: each bubble is a category, dots are its memories ------
// (Ported from the cory-orb memory panel; deterministic layout - golden-angle
// spiral seeded by category hash, then a short force relaxation - so the map
// stays familiar between visits.)
const PALETTE=['#c96442','#246f59','#8a6d3b','#4a7d9f','#7d6b8f','#b3402e','#5c8a5c','#a05c7b'];
const hashCode=s=>{let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h+s.charCodeAt(i))|0;return Math.abs(h)};
function buildGraph(items){
  const counts=new Map();
  items.forEach(m=>cats(m).forEach(c=>counts.set(c,(counts.get(c)||0)+1)));
  const nodes=[...counts.keys()].sort().map(t=>({tag:t,count:counts.get(t),
    radius:Math.min(30,11+5*Math.sqrt(counts.get(t))),color:PALETTE[hashCode(t)%PALETTE.length]}));
  // Two categories are linked when a memory carries both (strong signal), or
  // when a memory under one merely mentions the other in its text (weak).
  const index=new Map(nodes.map((n,i)=>[n.tag,i]));
  const edgeMap=new Map();
  const bump=(a,b,w)=>{const k=a<b?a+':'+b:b+':'+a;edgeMap.set(k,(edgeMap.get(k)||0)+w)};
  for(const m of items){
    const t=cats(m);
    for(let i=0;i<t.length;i++)for(let j=i+1;j<t.length;j++)bump(index.get(t[i]),index.get(t[j]),2);
    const text=String(m.content||'').toLowerCase();
    for(const n of nodes){
      if(t.includes(n.tag)||n.tag.length<3||n.tag==='(untagged)')continue;
      if(text.includes(n.tag))bump(index.get(t[0]),index.get(n.tag),1);
    }
  }
  const edges=[...edgeMap.entries()].map(([k,w])=>{const[a,b]=k.split(':').map(Number);return{a,b,weight:w}});
  return{nodes,edges};
}
function layoutMap(nodes,edges,width,height){
  nodes.forEach((n,i)=>{
    const angle=(hashCode(n.tag)%360)*(Math.PI/180)+i*2.39996;
    const spread=0.32*Math.min(width,height)*Math.sqrt((i+1)/nodes.length);
    n.x=width/2+spread*Math.cos(angle); n.y=height/2+spread*Math.sin(angle);
  });
  const iterations=nodes.length>1?130:0;
  for(let it=0;it<iterations;it++){
    for(let i=0;i<nodes.length;i++){
      const a=nodes[i];
      let fx=(width/2-a.x)*0.005, fy=(height/2-a.y)*0.005;
      for(let j=0;j<nodes.length;j++){
        if(i===j)continue;
        const b=nodes[j], dx=a.x-b.x, dy=a.y-b.y;
        const dist=Math.max(Math.hypot(dx,dy),1);
        const minGap=a.radius+b.radius+34;
        const push=(dist<minGap?2.2:1)*900/(dist*dist);
        fx+=dx/dist*push; fy+=dy/dist*push;
      }
      a.fx=fx; a.fy=fy;
    }
    for(const e of edges){
      const a=nodes[e.a], b=nodes[e.b], dx=b.x-a.x, dy=b.y-a.y;
      const dist=Math.max(Math.hypot(dx,dy),1);
      const pull=0.02*(dist-(a.radius+b.radius+60));
      a.fx+=dx/dist*pull; a.fy+=dy/dist*pull;
      b.fx-=dx/dist*pull; b.fy-=dy/dist*pull;
    }
    for(const n of nodes){
      n.x=Math.min(width-n.radius-8,Math.max(n.radius+8,n.x+n.fx));
      n.y=Math.min(height-n.radius-22,Math.max(n.radius+8,n.y+n.fy));
    }
  }
}
function drawMap(items){
  const wrap=document.getElementById('mapwrap'), canvas=document.getElementById('map');
  const tagCount=new Set(items.flatMap(cats)).size;
  if(!panels.map||!items.length||!tagCount){wrap.hidden=true;mapGraph=null;return}
  wrap.hidden=false;
  document.getElementById('maphint').textContent=activeCat
    ?`Filtering #${activeCat} - click its bubble again (or All) to clear.`
    :'Each bubble is a category; dots are its memories. Click a bubble to filter.';
  const width=Math.max(300,wrap.clientWidth-14);
  const height=Math.min(420,Math.max(200,150+tagCount*16));
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
  canvas.style.width=width+'px'; canvas.style.height=height+'px';
  const g=buildGraph(items);
  layoutMap(g.nodes,g.edges,width,height);
  mapGraph=g;
  const rootStyle=getComputedStyle(document.documentElement);
  const textColor=rootStyle.getPropertyValue('--text').trim()||'#dbe4f0';
  const dimColor=rootStyle.getPropertyValue('--dim').trim()||'#8494ab';
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,width,height);
  const alphaOf=n=>activeCat&&n.tag!==activeCat?0.25:1;
  for(const e of g.edges){
    const a=g.nodes[e.a], b=g.nodes[e.b];
    ctx.globalAlpha=0.35*Math.min(alphaOf(a),alphaOf(b));
    ctx.strokeStyle=dimColor;
    ctx.lineWidth=Math.min(3,1+e.weight*0.6);
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }
  for(const n of g.nodes){
    const alpha=alphaOf(n);
    const satellites=Math.min(n.count,10);
    for(let i=0;i<satellites;i++){
      const angle=i/satellites*Math.PI*2-Math.PI/2;
      ctx.fillStyle=n.color; ctx.globalAlpha=alpha*0.55;
      ctx.beginPath();
      ctx.arc(n.x+(n.radius+7)*Math.cos(angle),n.y+(n.radius+7)*Math.sin(angle),2.4,0,Math.PI*2);
      ctx.fill();
    }
    ctx.globalAlpha=alpha; ctx.fillStyle=n.color;
    ctx.beginPath(); ctx.arc(n.x,n.y,n.radius,0,Math.PI*2); ctx.fill();
    if(activeCat===n.tag){ctx.strokeStyle=textColor;ctx.lineWidth=2;ctx.stroke()}
    ctx.fillStyle='#ffffff';
    ctx.font=`700 ${Math.max(10,Math.min(13,n.radius*0.6))}px system-ui,sans-serif`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(String(n.count),n.x,n.y);
    ctx.fillStyle=textColor;
    ctx.font='600 12px system-ui,sans-serif'; ctx.textBaseline='top';
    const label=n.tag.length>18?n.tag.slice(0,17)+'…':n.tag;
    ctx.fillText('#'+label,n.x,n.y+n.radius+12);
  }
  ctx.globalAlpha=1;
}
function hitNode(event){
  if(!mapGraph)return null;
  const rect=document.getElementById('map').getBoundingClientRect();
  const x=event.clientX-rect.left, y=event.clientY-rect.top;
  return mapGraph.nodes.find(n=>Math.hypot(n.x-x,n.y-y)<=n.radius+6)||null;
}
document.getElementById('map').addEventListener('click',e=>{
  const n=hitNode(e);
  if(n){activeCat=activeCat===n.tag?null:n.tag;render(current)}
});
document.getElementById('map').addEventListener('mousemove',e=>{
  e.target.style.cursor=hitNode(e)?'pointer':'default';
});
window.addEventListener('resize',()=>drawMap(current));
const PAGE=100; let offset=0;
async function loadAll(more){
  if(!more){offset=0;current=[]}
  const u=uid()?'&user_id='+encodeURIComponent(uid()):'';
  const items=await api('/api/v1/memories?limit='+PAGE+'&offset='+offset+u);
  offset+=items.length; haveMore=items.length===PAGE;
  render(current.concat(items));
}
async function search(){
  const q=document.getElementById('q').value.trim(); if(!q)return loadAll();
  const body={query:q,limit:20}; if(uid())body.user_id=uid();
  const rs=await api('/api/v1/search',{method:'POST',body:JSON.stringify(body)});
  haveMore=false;
  render(rs.map(r=>({...r.memory,score:r.score})));
}
async function add(infer){
  const t=document.getElementById('newmem').value.trim(); if(!t)return;
  const cs=document.getElementById('newcats').value.split(',').map(s=>s.trim()).filter(Boolean);
  const res=await api('/api/v1/memories',{method:'POST',
    body:JSON.stringify({content:t,user_id:uid()||undefined,infer,categories:cs.length?cs:undefined})});
  if(res.warnings&&res.warnings.length)alert(res.warnings.join('\\n'));
  document.getElementById('newmem').value=''; document.getElementById('newcats').value='';
  loadAll(); loadStats();
}
async function distill(id){
  const r=await fetch('/api/v1/memories/'+id+'/distill',{method:'POST',headers:H});
  const data=await r.json().catch(()=>null);
  if(!r.ok){alert((data&&data.error)||('Distillation failed ('+r.status+').'));return}
  if(data.warnings&&data.warnings.length)alert(data.warnings.join('\\n'));
  loadAll(); loadStats();
}
async function del(id){await api('/api/v1/memories/'+id,{method:'DELETE'}); loadAll(); loadStats();}
function startEdit(id){editingId=id;render(current)}
function cancelEdit(){editingId=null;render(current)}
async function saveEdit(id){
  const content=document.getElementById('edit-note').value.trim(); if(!content)return;
  const cats=document.getElementById('edit-cats').value.split(',').map(s=>s.trim()).filter(Boolean);
  const updated=await api('/api/v1/memories/'+id,{method:'PATCH',body:JSON.stringify({content,categories:cats})});
  editingId=null;
  current=current.map(m=>m.id===id?{...m,...updated}:m);
  render(current);
}
// Same JSONL shape as `memry export`, so files work with the CLI and back.
async function exportMemories(){
  const u=uid()?'&user_id='+encodeURIComponent(uid()):'';
  let all=[],off=0,page;
  do{
    page=await api('/api/v1/memories?limit=500&offset='+off+u);
    all=all.concat(page); off+=page.length;
  }while(page.length===500);
  if(!all.length){alert('Nothing to export.');return}
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([all.map(m=>JSON.stringify(m)).join('\\n')+'\\n'],{type:'application/json'}));
  a.download='memry-export-'+new Date().toISOString().slice(0,10)+'.jsonl';
  a.click(); URL.revokeObjectURL(a.href);
}
// Additive: every row becomes a new verbatim memory (same field mapping as
// `memry import`); nothing is deleted or deduplicated.
async function importMemories(file){
  if(!file)return;
  const text=((await file.text())||'').trim(); if(!text)return;
  let rows;
  try{rows=text.startsWith('[')?JSON.parse(text):text.split('\\n').map(l=>l.trim()).filter(Boolean).map(l=>JSON.parse(l));}
  catch{alert('Not a valid JSON or JSONL export file.');return}
  if(!Array.isArray(rows))rows=[rows];
  const btn=document.getElementById('importbtn');
  let ok=0,failed=0;
  for(const r of rows){
    const content=String(r.content||'').trim();
    if(!content){failed++;continue}
    const body={content,infer:false,
      user_id:r.user_id||uid()||undefined,
      memory_type:r.memory_type||'semantic',
      importance:r.importance??0.5,
      categories:(r.categories&&r.categories.length)?r.categories:undefined};
    try{await api('/api/v1/memories',{method:'POST',body:JSON.stringify(body)});ok++}
    catch{failed++}
    btn.textContent='Import '+(ok+failed)+'/'+rows.length;
  }
  btn.textContent='Import';
  alert('Imported '+ok+' of '+rows.length+(failed?' ('+failed+' failed or empty)':'')+'.');
  loadAll(); loadStats();
}
async function loadStats(){
  const s=await api('/api/v1/stats');
  document.getElementById('stats').textContent=
    `${s.active_memories??'?'} active memories · ${s.invalidated_memories??0} invalidated · `+
    `${s.episodes??0} episodes · backend ${s.backend} · llm ${s.llm} · embeddings ${s.embedder}`;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
syncPanels(); loadStats(); loadAll();
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
        # Writes always land in a concrete namespace: an omitted user_id
        # defaults to config.default_user_id instead of storing NULL, which
        # no scoped read (dashboard, clients) would ever find again. Reads
        # keep None = "all users" as the admin view.
        result = store.add(
            content,
            user_id=_ns(request.state.tenant, body.get("user_id")) or default_user,
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

    async def distill_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        try:
            result = store.distill(request.path_params["memory_id"])
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # LLM/provider failure: report, don't 500
            return JSONResponse(
                {"error": f"distillation failed: {exc}"}, status_code=502
            )
        if result is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(result.model_dump())

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
        yet, so tenant keys are REST-only by design).

        The admin key may arrive as ``Authorization: Bearer <key>`` or, for
        clients that cannot send headers (claude.ai custom connectors),
        embedded in the URL as ``/mcp/<key>`` or ``/mcp?key=<key>``."""
        if api_key or tenants:
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            token = headers.get("authorization", "")
            token = token[7:].strip() if token.lower().startswith("bearer ") else ""
            if not token:
                # scope["path"] is the full path; the mount prefix lives in
                # root_path (ASGI spec), so look at the part after /mcp.
                root = scope.get("root_path", "")
                path = scope.get("path", "")
                sub = path[len(root):] if path.startswith(root) else path
                segments = [s for s in sub.split("/") if s]
                if api_key and segments and segments[0] == api_key:
                    token = segments[0]
                    # Strip the key segment so the MCP app sees its root path.
                    scope = dict(scope)
                    scope["path"] = root + "/" + "/".join(segments[1:])
                else:
                    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                    token = (query.get("key") or [""])[0]
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
        Route("/api/v1/memories/{memory_id}/distill", guarded(distill_memory), methods=["POST"]),
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
