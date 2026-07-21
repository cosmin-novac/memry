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
from functools import partial
from typing import Any
from urllib.parse import parse_qs

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from .mcp_server import create_server
from .store import MemoryStore

_DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memry dashboard</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50' fill='none' stroke='%2314b8a6' stroke-width='7' stroke-linecap='round'/%3E%3Ccircle cx='47' cy='10.5' r='4.5' fill='%2314b8a6'/%3E%3Ccircle cx='56' cy='20' r='3.2' fill='%2314b8a6' opacity='.85'/%3E%3Ccircle cx='57.5' cy='30' r='2.2' fill='%2314b8a6' opacity='.7'/%3E%3C/svg%3E">
<style>
:root{--bg:#0b0e14;--panel:#141a24;--line:#232c3b;--text:#dbe4f0;--dim:#8494ab;--accent:#5eead4;--warn:#f0a35e;font-size:15px}
@media (prefers-color-scheme: light){:root{--bg:#f5f7fa;--panel:#ffffff;--line:#dde4ee;--text:#1a2333;--dim:#5c6b82;--accent:#0d9488;--warn:#b45309}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:2rem 1rem}
h1{font-size:1.3rem;margin:.2rem 0 1rem}h1 span{color:var(--accent)}
h1 .datalinks{float:right;font-size:.75rem;font-weight:400;color:var(--dim)}
h1 .datalinks a{color:var(--dim);text-decoration:none;border-bottom:1px dotted var(--dim);cursor:help}
h1 .datalinks a:hover{color:var(--accent);border-bottom-color:var(--accent)}
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
#mapwrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.4rem;margin-bottom:.7rem;overflow:hidden}
#map{display:block;width:100%;border-radius:8px}
#mapwrap:fullscreen,#mapwrap.maxed{padding:0;border:0;border-radius:0;background:var(--bg)}
#mapwrap:fullscreen #map,#mapwrap.maxed #map{border-radius:0}
#mapwrap.maxed{position:fixed;inset:0;z-index:99999}
.gx-ctrl{position:absolute;top:.6rem;right:.6rem;display:flex;gap:.4rem;z-index:3}
.gx-ctrl button{background:color-mix(in srgb,var(--panel) 68%,transparent);border:1px solid var(--line);color:var(--dim);border-radius:7px;padding:.3rem .5rem;font-size:.72rem;cursor:pointer;backdrop-filter:blur(5px);line-height:1}
.gx-ctrl button:hover{color:var(--accent);border-color:var(--accent)}
.gx-read{position:absolute;top:.6rem;left:.6rem;z-index:3;font-size:.75rem;color:var(--dim);background:color-mix(in srgb,var(--panel) 60%,transparent);border:1px solid var(--line);border-radius:7px;padding:.32rem .6rem;backdrop-filter:blur(5px);max-width:62%;pointer-events:none;opacity:0;transition:opacity .15s}
.gx-read.on{opacity:1}.gx-read b{color:var(--text)}
.gx-stat{position:absolute;bottom:.5rem;left:.7rem;z-index:3;font-size:.68rem;color:var(--dim);opacity:.55;pointer-events:none}
</style></head><body><main>
<h1><svg viewBox="0 0 64 64" width="22" height="22" aria-hidden="true" style="color:var(--accent);vertical-align:-3px;margin-right:.35rem"><path d="M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50" fill="none" stroke="currentColor" stroke-width="7" stroke-linecap="round"/><circle cx="47" cy="10.5" r="4.5" fill="currentColor"/><circle cx="56" cy="20" r="3.2" fill="currentColor" opacity=".85"/><circle cx="57.5" cy="30" r="2.2" fill="currentColor" opacity=".7"/></svg><span>Mem</span>ry <small style="color:var(--dim);font-weight:400">memory dashboard</small>
<span class="datalinks"><a href="#" onclick="exportMemories();return false" title="Download memories as a .jsonl file: one JSON object per line with content, categories, user_id, type and importance. Same format as the CLI command memry export. Respects the user_id filter box.">export</a> · <a href="#" id="importbtn" onclick="document.getElementById('importfile').click();return false" title="Additive import from a memry export: a .jsonl file (one JSON object per line) or a JSON array. Each row needs a content field; categories, user_id, memory_type and importance are optional. Nothing is deleted or overwritten.">import</a></span></h1>
<div id="stats">loading…</div>
<div class="bar">
  <input id="user" placeholder="user_id (all)" style="width:11rem">
  <input id="q" placeholder="search memories…" style="flex:1;min-width:12rem">
  <button class="primary" onclick="search()">Search</button>
  <button onclick="activeCat=null;loadAll()">All</button>
  <button class="toggle" id="addbtn" onclick="togglePanel('add')">+ Add</button>
  <button class="toggle" id="mapbtn" onclick="togglePanel('map')">Map</button>
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
<div id="mapwrap" hidden><canvas id="map"></canvas>
<div class="gx-ctrl"><button id="fsBtn" title="Fullscreen" aria-label="Fullscreen">⤢</button></div>
<div class="gx-read" id="mapread"></div><div class="gx-stat" id="mapstat"></div></div>
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
// Add form is opt-in, map is opt-out; the choice sticks per browser.
const panels={add:localStorage.getItem('memry_show_add')==='1',map:localStorage.getItem('memry_show_map')!=='0'};
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
let current=[],activeCat=null,haveMore=false,editingId=null,hoverTag=null;
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

// ---- galaxy map: tags as planets in three sd-based orbital zones ----------
// core: count >= mean+2sd (largest tags when none qualify), rim: count <=
// max(1, mean-2sd), belt: between. Deterministic per tag (seeded angles),
// slow counter-rotating drift, temperature palette: gold core, teal belt,
// ice-violet rim. Links = co-occurrence (strong) or text mention (weak).
const hashCode=s=>{let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h+s.charCodeAt(i))|0;return Math.abs(h)};
const mulberry=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
let G=null,gRAF=0,gPulses=[],gMaxed=false;
const gStars=Array.from({length:230},(_,i)=>{const r=mulberry(i*2654435761+11);const k=r();
  return{x:r(),y:r(),s:.3+r()*1.4,a:.2+r()*.7,ph:r()*6.28,sp:.3+r()*.7,hue:k>.94?36:(k>.84?176:(k>.78?252:null)),big:r()>.965};});
const gDust=Array.from({length:150},(_,i)=>{const r=mulberry(i*97+5);
  return{a:r()*Math.PI*2,rad:0.12+r()*0.95,off:(r()-0.5)*0.3,s:0.35+r()*0.75,al:0.05+r()*0.10,teal:r()>0.8};});
const gTone=(n,dark)=>{const j=((n.seed%1000)/1000-0.5);
  if(n.zone==='core')return{h:36+j*16,s:dark?95:80,l:dark?66:42};
  if(n.zone==='belt')return{h:172+j*34,s:dark?75:65,l:dark?60:36};
  return{h:248+j*40,s:dark?60:50,l:dark?74:44};};
const hsla=(c,a,dl)=>`hsla(${c.h},${c.s}%,${Math.max(4,Math.min(96,c.l+(dl||0)))}%,${a})`;
const hexA=(hex,a)=>{const v=parseInt(hex.slice(1),16);return`rgba(${(v>>16)&255},${(v>>8)&255},${v&255},${a})`};
function buildGalaxy(items){
  const counts=new Map();
  items.forEach(m=>cats(m).forEach(c=>counts.set(c,(counts.get(c)||0)+1)));
  if(!counts.size)return null;
  const vals=[...counts.values()];
  const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
  const sd=Math.sqrt(vals.reduce((a,c)=>a+(c-mean)**2,0)/vals.length);
  const coreMin=mean+2*sd,rimMax=Math.max(1,mean-2*sd),maxC=Math.max(...vals);
  const fb=!vals.some(c=>c>=coreMin);
  // belt and rim planets get a size boost so the outer zones read clearly
  const ZF={core:1.0,belt:1.28,rim:1.55};
  const nodes=[...counts.keys()].sort().map(tag=>{const count=counts.get(tag);
    const zone=(fb?count===maxC:count>=coreMin)?'core':(count<=rimMax?'rim':'belt');
    return{tag,count,zone,radius:Math.min(34,(9+5*Math.sqrt(count))*ZF[zone]),seed:hashCode(tag),h:0};});
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
  const edges=[...edgeMap.entries()].map(([k,w])=>{const[a,b]=k.split(':').map(Number);return{a,b,weight:Math.min(3,w)}});
  const neigh={};
  edges.forEach(e=>{const ta=nodes[e.a].tag,tb=nodes[e.b].tag;
    (neigh[ta]??=new Set()).add(tb);(neigh[tb]??=new Set()).add(ta);});
  const BANDS={core:[0.02,0.16],belt:[0.30,0.62],rim:[0.66,0.99]};
  const PHASE={core:0,belt:0.7,rim:1.4},PACK={core:4,belt:16,rim:22},GOLDEN=2.399963229728653;
  for(const zone of['core','belt','rim']){
    const ring=nodes.filter(n=>n.zone===zone);
    ring.sort((a,b)=>b.count-a.count||a.tag.localeCompare(b.tag));
    const N=ring.length,lo=BANDS[zone][0],hi=BANDS[zone][1];
    // planets shrink as a ring fills so a long tail of tags still fits
    const shrink=Math.max(0.4,Math.min(1,1/Math.sqrt(Math.max(1,N)/PACK[zone])));
    const dense=N>PACK[zone]*0.8;
    ring.forEach((n,k)=>{const rnd=mulberry(n.seed);
      n.radius*=shrink;
      if(zone==='core'){n.rFrac=N===1?0:0.13;n.ang=PHASE.core+k*(Math.PI*2/N);}
      // dense rings fill the annulus with a phyllotaxis (sunflower) pattern -
      // even area density at any count, no clumps; sparse ones use an even ring
      else if(dense){n.rFrac=lo+(hi-lo)*Math.sqrt((k+0.5)/N);n.ang=PHASE[zone]+k*GOLDEN;}
      else{n.ang=PHASE[zone]+k*(Math.PI*2/N)+(rnd()-0.5)*0.22;n.rFrac=lo+(hi-lo)*(0.35+0.5*rnd());}});
  }
  return{nodes,edges,neigh,byTag:Object.fromEntries(nodes.map(n=>[n.tag,n])),fb,total:items.length};
}
function drawMap(items){
  const wrap=document.getElementById('mapwrap');
  G=panels.map&&items.length?buildGalaxy(items):null;
  if(!G){wrap.hidden=true;if(gRAF){cancelAnimationFrame(gRAF);gRAF=0}return}
  wrap.hidden=false;
  sizeGalaxy();
  galaxyRead();
  if(reducedMotion)galaxyFrame(performance.now());
  else if(!gRAF)gRAF=requestAnimationFrame(galaxyFrame);
}
function sizeGalaxy(){
  const wrap=document.getElementById('mapwrap'),canvas=document.getElementById('map');
  const big=document.fullscreenElement===wrap||gMaxed;
  let width,height;
  if(big){width=wrap.clientWidth;height=wrap.clientHeight;}
  else{width=Math.max(300,wrap.clientWidth-10);height=Math.max(380,Math.min(620,width*0.56));}
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  canvas.style.width=width+'px';canvas.style.height=height+'px';
  canvas.getContext('2d').setTransform(dpr,0,0,dpr,0,0);
  G.W=width;G.H=height;G.CX=width/2;G.CY=height/2;G.RX=width/2-46;G.RY=height/2-42;
}
// hover/filter info as a corner overlay; a faint tag/memory count sits bottom-left
function galaxyRead(){
  const readEl=document.getElementById('mapread'),statEl=document.getElementById('mapstat');
  if(!G){readEl.classList.remove('on');return}
  const f=activeCat?G.byTag[activeCat]:(hoverTag?G.byTag[hoverTag]:null);
  if(f){
    const linked=G.neigh[f.tag]?[...G.neigh[f.tag]].map(x=>'#'+x).join(', '):'none';
    readEl.innerHTML=`<b>#${f.tag}</b> · ${f.count} memor${f.count===1?'y':'ies'} · ${f.zone}`
      +(activeCat===f.tag?' · filtering':'')+`<br><span style="opacity:.8">linked: ${linked}</span>`;
    readEl.classList.add('on');
  }else readEl.classList.remove('on');
  statEl.textContent=`${G.nodes.length} tags · ${G.total} memories`+(G.fb?' · core = largest':'');
}
function galaxyFrame(now){
  if(!G){gRAF=0;return}
  const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
  const W=G.W,H=G.H,RX=G.RX,RY=G.RY,Rm=Math.max(RX,RY),CX=G.CX,CY=G.CY,t=now;
  const rootStyle=getComputedStyle(document.documentElement);
  const bg=(rootStyle.getPropertyValue('--bg').trim()||'#0b0e14');
  const dark=parseInt(bg.slice(5,7)||'14',16)<120;
  const TEXT=rootStyle.getPropertyValue('--text').trim()||'#dbe4f0';
  const DIM=rootStyle.getPropertyValue('--dim').trim()||'#8494ab';
  const WARM=rootStyle.getPropertyValue('--warn').trim()||'#f0a35e';
  const STAR=dark?'#c9d6ea':'#33415c';
  const still=reducedMotion||!!hoverTag||!!activeCat;
  ctx.clearRect(0,0,W,H);
  // deep space ground
  const g0=ctx.createRadialGradient(CX,CY-H*0.05,Rm*0.12,CX,CY,Rm*1.45);
  g0.addColorStop(0,dark?'#0b1020':'#f0f4f9');g0.addColorStop(1,dark?'#04060c':'#e3e9f2');
  ctx.fillStyle=g0;ctx.fillRect(0,0,W,H);
  // nebula washes + distant sunlight
  for(const wsh of[[W*0.22,H*0.2,Rm*1.0,252],[W*0.8,H*0.84,Rm*0.95,176],[W*0.62,H*0.2,Rm*0.7,318]]){
    const neb=ctx.createRadialGradient(wsh[0],wsh[1],0,wsh[0],wsh[1],wsh[2]);
    neb.addColorStop(0,`hsla(${wsh[3]},70%,${dark?55:70}%,${dark?0.09:0.11})`);
    neb.addColorStop(1,'transparent');
    ctx.fillStyle=neb;ctx.fillRect(0,0,W,H);
  }
  const sun=ctx.createRadialGradient(W*0.1,H*0.06,0,W*0.1,H*0.06,Rm*0.9);
  sun.addColorStop(0,hexA(WARM,0.10));sun.addColorStop(1,'transparent');
  ctx.fillStyle=sun;ctx.fillRect(0,0,W,H);
  // galactic dust band with stellar specks
  ctx.save();ctx.translate(CX,CY);ctx.rotate(-0.28);ctx.scale(1,0.34);
  const band=ctx.createRadialGradient(0,0,0,0,0,Rm*1.4);
  band.addColorStop(0,hexA(STAR,0.07));band.addColorStop(1,'transparent');
  ctx.fillStyle=band;ctx.beginPath();ctx.arc(0,0,Rm*1.4,0,Math.PI*2);ctx.fill();
  for(const d of gDust){
    ctx.globalAlpha=d.al*(dark?1:0.55);
    ctx.fillStyle=d.teal?`hsla(176,60%,${dark?70:40}%,1)`:STAR;
    ctx.beginPath();ctx.arc(Math.cos(d.a)*Rm*1.3*d.rad,Math.sin(d.a)*Rm*1.3*d.rad+d.off*Rm,d.s*2,0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1;ctx.restore();
  // starfield
  for(const s of gStars){
    const tw=reducedMotion?0.7:0.5+0.5*Math.sin(s.ph+t*0.00025*s.sp);
    ctx.globalAlpha=s.a*tw*(dark?0.9:0.5);
    ctx.fillStyle=s.hue?`hsla(${s.hue},80%,${dark?75:40}%,1)`:STAR;
    const x=s.x*W,y=s.y*H;
    ctx.fillRect(x,y,s.s,s.s);
    if(s.big){ctx.globalAlpha*=0.4;
      const halo=ctx.createRadialGradient(x,y,0,x,y,7);
      halo.addColorStop(0,hexA(STAR,0.5));halo.addColorStop(1,'transparent');
      ctx.fillStyle=halo;ctx.fillRect(x-7,y-7,14,14);}
  }
  ctx.globalAlpha=1;
  // gravity well + lens streak
  const well=ctx.createRadialGradient(CX,CY,0,CX,CY,Rm*0.30);
  well.addColorStop(0,hexA(WARM,0.11));well.addColorStop(1,'transparent');
  ctx.fillStyle=well;ctx.fillRect(CX-Rm*0.32,CY-Rm*0.32,Rm*0.64,Rm*0.64);
  const streak=ctx.createLinearGradient(CX-RX*0.9,CY,CX+RX*0.9,CY);
  streak.addColorStop(0,'transparent');streak.addColorStop(0.5,hexA(WARM,0.15));streak.addColorStop(1,'transparent');
  ctx.fillStyle=streak;ctx.fillRect(CX-RX*0.9,CY-1,RX*1.8,2);
  ctx.strokeStyle=hexA(WARM,0.45);ctx.lineWidth=0.9;
  ctx.beginPath();ctx.ellipse(CX,CY,RX*0.05,RY*0.05,0,0,Math.PI*2);ctx.stroke();
  // zone hairlines (elliptical)
  ctx.lineWidth=1.2;
  for(const fr of[0.28,0.72,0.99]){
    ctx.strokeStyle=hexA(DIM,0.4);
    ctx.beginPath();ctx.ellipse(CX,CY,RX*fr,RY*fr,0,0,Math.PI*2);ctx.stroke();
  }
  // positions + focus
  const pts={};
  for(const n of G.nodes){
    n.ang+=still?0:0.00010*(n.zone==='core'?1:(n.zone==='belt'?-0.5:0.3));
    pts[n.tag]={x:CX+n.rFrac*RX*Math.cos(n.ang),y:CY+n.rFrac*RY*Math.sin(n.ang)};
  }
  const sel=activeCat?G.byTag[activeCat]:null;
  const hov=hoverTag?G.byTag[hoverTag]:null;
  const foc=hov||sel;
  const emph=n=>{
    if(!foc)return 1;
    if(n===foc)return 1;
    return(G.neigh[foc.tag]&&G.neigh[foc.tag].has(n.tag))?0.92:0.16;
  };
  // filaments: soft underglow + bright core line; particles on locked links
  for(const e of G.edges){
    const na=G.nodes[e.a],nb=G.nodes[e.b];
    const p=pts[na.tag],q=pts[nb.tag];
    const ca=gTone(na,dark),cb=gTone(nb,dark);
    const touches=foc&&(na===foc||nb===foc);
    const A=touches?1:(foc?0.06:1);
    const mxp=(p.x+q.x)/2,myp=(p.y+q.y)/2;
    const cpx=mxp+(CX-mxp)*0.13,cpy=myp+(CY-myp)*0.13;
    const grad=ctx.createLinearGradient(p.x,p.y,q.x,q.y);
    grad.addColorStop(0,hsla(ca,1));grad.addColorStop(1,hsla(cb,1));
    if(dark)ctx.globalCompositeOperation='lighter';
    ctx.globalAlpha=(touches?0.34:0.15)*A*(dark?1:0.9);
    ctx.strokeStyle=grad;ctx.lineWidth=(touches?7:4.5)+e.weight*0.6;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo(cpx,cpy,q.x,q.y);ctx.stroke();
    ctx.globalAlpha=(touches?0.95:0.5)*A;
    ctx.lineWidth=touches?1.8:1.1;
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo(cpx,cpy,q.x,q.y);ctx.stroke();
    if(touches&&foc===sel&&!reducedMotion){
      for(let i=0;i<2+e.weight;i++){
        const tt=((t*0.00022*(0.7+0.15*i))+i/(2+e.weight))%1;
        const u=1-tt;
        const sx=u*u*p.x+2*u*tt*cpx+tt*tt*q.x,sy=u*u*p.y+2*u*tt*cpy+tt*tt*q.y;
        const cc=tt<0.5?ca:cb;
        ctx.globalAlpha=0.9;
        const spark=ctx.createRadialGradient(sx,sy,0,sx,sy,4.5);
        spark.addColorStop(0,hsla(cc,0.95,18));spark.addColorStop(1,'transparent');
        ctx.fillStyle=spark;ctx.beginPath();ctx.arc(sx,sy,4.5,0,Math.PI*2);ctx.fill();
      }
    }
    ctx.globalCompositeOperation='source-over';
  }
  ctx.globalAlpha=1;
  // click pulses
  for(let i=gPulses.length-1;i>=0;i--){
    const pl=gPulses[i],age=(now-pl.start)/700;
    if(age>1){gPulses.splice(i,1);continue}
    ctx.globalAlpha=(1-age)*0.5;
    ctx.strokeStyle=hsla(pl.tone,1,10);ctx.lineWidth=1.5;
    ctx.beginPath();ctx.arc(pl.x,pl.y,pl.r+age*46,0,Math.PI*2);ctx.stroke();
  }
  ctx.globalAlpha=1;
  // planets: flat crisp discs, varied memory dots, gated counts and labels
  for(const n of G.nodes){
    const p=pts[n.tag],x=p.x,y=p.y,A=emph(n),c=gTone(n,dark);
    n.h+=(((n===hov||activeCat===n.tag)?1:0)-n.h)*(reducedMotion?1:0.14);
    ctx.globalAlpha=A;
    if(n.h>0.03){
      const len=n.radius*(2.4+1.0*n.h);
      for(const rot of[0,Math.PI/2]){
        const sx=Math.cos(rot),sy=Math.sin(rot);
        const sp=ctx.createLinearGradient(x-sx*len,y-sy*len,x+sx*len,y+sy*len);
        sp.addColorStop(0,'transparent');sp.addColorStop(0.5,hsla(c,0.55*n.h*A,15));sp.addColorStop(1,'transparent');
        ctx.strokeStyle=sp;ctx.lineWidth=0.8;
        ctx.beginPath();ctx.moveTo(x-sx*len,y-sy*len);ctx.lineTo(x+sx*len,y+sy*len);ctx.stroke();
      }
    }
    ctx.shadowColor=hsla(c,0.5,6);ctx.shadowBlur=(5+5*n.h)*(dark?1:0.5);
    const body=ctx.createRadialGradient(x,y,0,x,y,n.radius);
    body.addColorStop(0,hsla(c,1,3));body.addColorStop(0.8,hsla(c,1));body.addColorStop(1,hsla(c,1,-3));
    ctx.fillStyle=body;
    ctx.beginPath();ctx.arc(x,y,n.radius,0,Math.PI*2);ctx.fill();
    ctx.shadowBlur=0;
    ctx.strokeStyle=hsla(c,0.85+0.15*n.h,dark?16:-14);ctx.lineWidth=1.2;
    ctx.beginPath();ctx.arc(x,y,n.radius,0,Math.PI*2);ctx.stroke();
    const sats=Math.min(n.count,10);
    for(let i=0;i<sats;i++){
      const angle=i/sats*Math.PI*2-Math.PI/2+(reducedMotion?0:t*0.00008);
      const ds=0.9+(((n.seed>>3)+i*37)%10)/12;
      ctx.fillStyle=hsla(c,0.8*A,10);
      ctx.beginPath();ctx.arc(x+(n.radius+6)*Math.cos(angle),y+(n.radius+6)*Math.sin(angle),ds,0,Math.PI*2);ctx.fill();
    }
    if(activeCat===n.tag){
      ctx.strokeStyle=hsla(c,0.95,18);ctx.lineWidth=1.3;
      ctx.beginPath();ctx.arc(x,y,n.radius+5,0,Math.PI*2);ctx.stroke();
    }
    if(n.radius>=9){
      ctx.globalAlpha=Math.min(1,A+0.05);
      ctx.font=`700 ${Math.max(8,Math.min(15,n.radius*0.55))}px ui-sans-serif,system-ui`;
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.shadowColor=dark?'rgba(0,0,0,0.7)':'rgba(255,255,255,0.8)';ctx.shadowBlur=4;
      ctx.fillStyle=dark?'#ffffff':'#0c1524';
      ctx.fillText(n.count,x,y+0.5);
      ctx.shadowBlur=0;
    }
    const lit=n.h>0.4||(foc&&G.neigh[foc.tag]&&G.neigh[foc.tag].has(n.tag));
    if(n.radius>=19||n.h>0.05||lit){
      ctx.globalAlpha=Math.min(1,A+0.05);
      ctx.font='500 9.5px ui-sans-serif,system-ui';
      if('letterSpacing'in ctx)ctx.letterSpacing='1.5px';
      ctx.textAlign='center';ctx.textBaseline='top';
      const label=n.tag.length>18?n.tag.slice(0,17)+'…':n.tag;
      ctx.fillStyle=lit?TEXT:hexA(DIM.length===7?DIM:'#8494ab',0.95);
      ctx.fillText(label.toUpperCase()+' · '+n.count,x,y+n.radius+8+3*n.h);
      if('letterSpacing'in ctx)ctx.letterSpacing='0px';
    }
  }
  ctx.globalAlpha=1;
  if(!reducedMotion&&panels.map)gRAF=requestAnimationFrame(galaxyFrame);else gRAF=0;
}
function hitNode(event){
  if(!G)return null;
  const rect=document.getElementById('map').getBoundingClientRect();
  const x=event.clientX-rect.left,y=event.clientY-rect.top;
  let best=null,bd=1e9;
  for(const n of G.nodes){
    const px=G.CX+n.rFrac*G.RX*Math.cos(n.ang),py=G.CY+n.rFrac*G.RY*Math.sin(n.ang);
    const d=Math.hypot(px-x,py-y);
    if(d<Math.max(n.radius+8,14)&&d<bd){bd=d;best=n}
  }
  return best;
}
document.getElementById('map').addEventListener('click',e=>{
  const n=hitNode(e);
  if(n){
    activeCat=activeCat===n.tag?null:n.tag;
    const rootStyle=getComputedStyle(document.documentElement);
    const dark=parseInt((rootStyle.getPropertyValue('--bg').trim()||'#0b0e14').slice(5,7)||'14',16)<120;
    gPulses.push({x:G.CX+n.rFrac*G.RX*Math.cos(n.ang),y:G.CY+n.rFrac*G.RY*Math.sin(n.ang),
      r:n.radius,start:performance.now(),tone:gTone(n,dark)});
    render(current);
  }
});
document.getElementById('map').addEventListener('mousemove',e=>{
  const n=hitNode(e);
  e.target.style.cursor=n?'pointer':'default';
  const tg=n?n.tag:null;
  if(tg!==hoverTag){hoverTag=tg;if(G)galaxyRead();if(reducedMotion&&G)galaxyFrame(performance.now())}
});
document.getElementById('map').addEventListener('mouseleave',()=>{
  if(hoverTag){hoverTag=null;if(G){galaxyRead();if(reducedMotion)galaxyFrame(performance.now())}}
});
// Fullscreen: real API where allowed, CSS-maximize fallback otherwise.
function setMaxed(v){gMaxed=v;document.getElementById('mapwrap').classList.toggle('maxed',v);
  document.documentElement.style.overflow=v?'hidden':'';
  if(G){sizeGalaxy();if(reducedMotion)galaxyFrame(performance.now())}}
document.getElementById('fsBtn').addEventListener('click',()=>{
  const wrap=document.getElementById('mapwrap');
  if(document.fullscreenElement){document.exitFullscreen();return}
  if(gMaxed){setMaxed(false);return}
  let p;try{p=wrap.requestFullscreen&&wrap.requestFullscreen();}catch(e){}
  if(p&&p.then)p.then(()=>{},()=>setMaxed(true));
  else if(!document.fullscreenElement)setMaxed(true);
});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&gMaxed)setMaxed(false);});
document.addEventListener('fullscreenchange',()=>{if(G){sizeGalaxy();if(reducedMotion)galaxyFrame(performance.now())}});
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
// Additive: every row becomes a new verbatim memory in ONE bulk request
// (server batches the embedding calls); nothing is deleted or deduplicated.
async function importMemories(file){
  if(!file)return;
  const text=((await file.text())||'').trim(); if(!text)return;
  let rows;
  try{rows=text.startsWith('[')?JSON.parse(text):text.split('\\n').map(l=>l.trim()).filter(Boolean).map(l=>JSON.parse(l));}
  catch{alert('Not a valid JSON or JSONL export file.');return}
  if(!Array.isArray(rows))rows=[rows];
  const btn=document.getElementById('importbtn');
  btn.textContent='importing…';
  try{
    const res=await api('/api/v1/import',{method:'POST',
      body:JSON.stringify({memories:rows,user_id:uid()||undefined})});
    if(res&&res.imported!==undefined)
      alert('Imported '+res.imported+' of '+rows.length+(res.skipped?' ('+res.skipped+' empty rows skipped)':'')+'.');
    else alert((res&&res.error)||'Import failed.');
  }catch{alert('Import failed.')}
  btn.textContent='import';
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
        # Store calls that hit LLM/embedding providers run in the threadpool:
        # blocking the event loop would stall every other request for the
        # duration of a provider round-trip.
        result = await run_in_threadpool(partial(
            store.add,
            content,
            user_id=_ns(request.state.tenant, body.get("user_id")) or default_user,
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            metadata=body.get("metadata"),
            infer=bool(body.get("infer", True)),
            memory_type=body.get("memory_type", "semantic"),
            importance=float(body.get("importance", 0.5)),
            categories=body.get("categories"),
        ))
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

    async def list_categories_route(request: Request) -> Response:
        q = request.query_params
        cats = await run_in_threadpool(partial(
            store.categories,
            user_id=_ns(request.state.tenant, q.get("user_id")),
            agent_id=q.get("agent_id"),
            run_id=q.get("run_id"),
        ))
        return JSONResponse(cats)

    async def import_memories_route(request: Request) -> Response:
        """Bulk verbatim import: a JSON array of rows, or {memories: [...],
        user_id?}. One request instead of one POST per memory."""
        body = await request.json()
        rows = body if isinstance(body, list) else body.get("memories")
        if not isinstance(rows, list) or not rows:
            return JSONResponse(
                {"error": "provide a JSON array of rows or {\"memories\": [...]}"},
                status_code=400,
            )
        default_uid = None if isinstance(body, list) else body.get("user_id")
        tenant = request.state.tenant
        sanitized = [
            {**row, "user_id": _ns(tenant, row.get("user_id") or default_uid)}
            for row in rows
            if isinstance(row, dict)
        ]
        result = await run_in_threadpool(partial(
            store.import_verbatim,
            sanitized,
            user_id=_ns(tenant, default_uid) or default_user,
        ))
        return JSONResponse(result, status_code=201)

    async def distill_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        try:
            result = await run_in_threadpool(
                partial(store.distill, request.path_params["memory_id"])
            )
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
        results = await run_in_threadpool(partial(
            store.search,
            body.get("query", ""),
            user_id=_ns(request.state.tenant, body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            limit=int(body.get("limit", 10)),
            include_invalid=bool(body.get("include_invalid", False)),
            categories=_parse_categories(body.get("categories")),
        ))
        return JSONResponse(
            [
                {"memory": r.memory.model_dump(), "score": r.score, "signals": r.signals}
                for r in results
            ]
        )

    async def context(request: Request) -> Response:
        body = await request.json()
        ctx = await run_in_threadpool(partial(
            store.reconstruct_context,
            body.get("query", ""),
            user_id=_ns(request.state.tenant, body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            token_budget=int(body.get("token_budget", 1200)),
        ))
        return JSONResponse(ctx.model_dump())

    async def stats(request: Request) -> Response:
        data = await run_in_threadpool(store.stats)
        if request.state.tenant is not None:
            prefix = f"{request.state.tenant}::"
            everything = await run_in_threadpool(
                partial(store.get_all, include_invalid=True, limit=100_000)
            )
            mine = [
                m for m in everything
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
        outcome = await run_in_threadpool(partial(
            store.resolve_entities,
            user_id=_ns(request.state.tenant, body.get("user_id")),
        ))
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
        Route("/api/v1/categories", guarded(list_categories_route), methods=["GET"]),
        Route("/api/v1/import", guarded(import_memories_route), methods=["POST"]),
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
