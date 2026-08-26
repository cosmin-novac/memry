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
key in the MCP URL instead: ``/mcp/<key>`` or ``/mcp?key=<key>``.

Tenant keys (MEMRY_TENANTS) work on both transports and are confined to their
own ``<tenant>::<user>`` namespace; the admin key is unconfined. See
memry.principal for how that identity is derived and enforced.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any
from urllib.parse import parse_qs

from mcp.server.auth.provider import construct_redirect_uri
from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Mount, Route

from .accounts import SESSION_TTL, AccountStore, default_auth_db_path
from .enrichment import EnrichmentWorker
from .mcp_server import PRINCIPAL_SCOPE_KEY, create_server
from .oauth import MEMRY_SCOPE, MemryOAuthProvider
from .principal import ADMIN, Principal
from .store import MemoryStore

SESSION_COOKIE = "memry_session"

_DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memry dashboard</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50' fill='none' stroke='%2314b8a6' stroke-width='7' stroke-linecap='round'/%3E%3Ccircle cx='47' cy='10.5' r='4.5' fill='%2314b8a6'/%3E%3Ccircle cx='56' cy='20' r='3.2' fill='%2314b8a6' opacity='.85'/%3E%3Ccircle cx='57.5' cy='30' r='2.2' fill='%2314b8a6' opacity='.7'/%3E%3C/svg%3E">
<style>
:root{--bg:#0b0e14;--panel:#141a24;--line:#232c3b;--text:#dbe4f0;--dim:#8494ab;--accent:#5eead4;--warn:#f0a35e;--semantic:#5eead4;--procedural:#60a5fa;--episodic:#fbbf24;--working:#c084fc;font-size:15px}
@media (prefers-color-scheme: light){:root{--bg:#f5f7fa;--panel:#ffffff;--line:#dde4ee;--text:#1a2333;--dim:#5c6b82;--accent:#0d9488;--warn:#b45309;--semantic:#0d9488;--procedural:#2563eb;--episodic:#b45309;--working:#7c3aed}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:2rem 1rem}
h1{font-size:1.3rem;margin:.2rem 0 1rem}h1 span{color:var(--accent)}
h1 .datalinks{float:right;font-size:.75rem;font-weight:400;color:var(--dim)}
h1 .datalinks a{color:var(--dim);text-decoration:none;border-bottom:1px dotted var(--dim);cursor:help}
h1 .datalinks a:hover{color:var(--accent);border-bottom-color:var(--accent)}
h1 .datalinks .knowledge-link{display:inline-block;background:var(--accent);color:#04211c;border:1px solid transparent;border-radius:999px;padding:.22rem .55rem;font-weight:700;cursor:pointer;box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 28%,transparent)}
h1 .datalinks .knowledge-link:hover{color:#04211c;border-color:transparent;filter:brightness(1.08)}
.bar{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem}
input,button,textarea,select{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:.5rem .7rem}
.qwrap{position:relative;flex:1;min-width:12rem;display:flex}
.qwrap input{flex:1;padding-right:1.9rem}
#qclear{position:absolute;right:.3rem;top:50%;transform:translateY(-50%);border:none;background:none;color:var(--dim);padding:.2rem .4rem;font-size:.9rem;line-height:1;display:none}
#qclear:hover{color:var(--accent)}
.search-filters{display:grid;grid-template-columns:minmax(12rem,1fr) minmax(10rem,1fr) minmax(12rem,1.4fr);gap:.5rem;margin:-.45rem 0 1rem}
/* `display:grid` above beats the UA rule behind the `hidden` attribute, so the
   panel has to be hidden explicitly or it is never actually collapsed. */
.search-filters[hidden]{display:none}
.search-filters .range{display:flex;gap:.35rem;align-items:center}
.search-filters .range input{min-width:0}
.search-filters select[multiple]{height:5.2rem;padding:.15rem}
.search-filters .picked{color:var(--accent)}
.search-filters label{display:flex;flex-direction:column;gap:.2rem;color:var(--dim);font-size:.72rem}
.search-filters input,.search-filters select{width:100%;min-width:0;color:var(--text)}
@media(max-width:650px){.search-filters{grid-template-columns:1fr}}
html.knowledge-open,body.knowledge-open{overflow:hidden}
.modal{position:fixed;inset:0;z-index:99998;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;padding:4rem 1rem;overflow:auto;overscroll-behavior:contain}
.modal.on{display:flex}
.modal .sheet{background:var(--panel);border:1px solid var(--line);border-radius:12px;width:min(97vw,96rem);padding:1.2rem 1.3rem}
/* Entities: the selected one gets its own column beside the list, so choosing
   an entity does not push its detail below the merge proposals. Below the
   breakpoint it stacks and moves to the TOP, where a selection belongs. */
.entity-split{display:grid;grid-template-columns:1fr;gap:1.1rem;align-items:start}
@media(min-width:62rem){.entity-split{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}}
.entity-side{min-width:0}
.entity-side:empty{display:none}
@media(max-width:62rem){.entity-side{order:-1}}
@media(min-width:62rem){.entity-side{position:sticky;top:.25rem;max-height:78vh;overflow:auto}}
.entity-side .detail{border:1px solid var(--line);border-radius:10px;padding:.75rem .85rem}
.modal h2{margin:.1rem 0 .2rem;font-size:1.05rem}.modal h2 .x{float:right;cursor:pointer;color:var(--dim);border:none;background:none;font-size:1rem}
.modal .hint{color:var(--dim);font-size:.8rem;margin:0 0 .9rem}
.tagrow{display:flex;align-items:center;gap:.5rem;padding:.32rem .1rem;border-bottom:1px solid var(--line)}
.tagrow input[type=checkbox]{width:auto;flex:none}
.tagrow .name{flex:1;min-width:0}
.tagrow .name b{font-weight:600}
.tagrow .cnt{color:var(--dim);font-size:.78rem}
.tagrow .syn{border:1px solid var(--accent);color:var(--accent);border-radius:999px;padding:0 .45rem;font-size:.68rem}
.tagrow .entity-type{flex:0 0 6.5rem;text-align:center;white-space:nowrap}
.tagrow .act{border:none;background:none;color:var(--dim);cursor:pointer;padding:.15rem .35rem;font-size:.8rem}
.tagrow .act:hover{color:var(--accent)}.tagrow .act.del:hover{color:var(--warn)}
.tagbar{display:flex;gap:.5rem;flex-wrap:wrap;margin:.8rem 0;align-items:center}
.tagbar .sel{color:var(--dim);font-size:.82rem;margin-right:auto}
input:focus,textarea:focus,select:focus{outline:2px solid var(--accent);outline-offset:-1px}
button{cursor:pointer}button.primary{background:var(--accent);color:#04211c;border-color:transparent;font-weight:600}
button.toggle[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
/* A filter left on behind a collapsed panel would silently narrow every result,
   so the button carries a dot whenever one is active. */
button.toggle.active{border-color:var(--accent);color:var(--accent)}
#filterbtn svg{width:.85em;height:.85em;vertical-align:-.08em}
#filterdot{color:var(--accent);font-size:1.1em;line-height:0}
#addbtn{font-size:1.15rem;font-weight:650;line-height:1;padding:.42rem .72rem}
.knowledge-tabs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.9rem 0}
.knowledge-tabs button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
.kpanel[hidden]{display:none}.entity-link,.entity-chip{border:1px solid var(--line);background:none;color:var(--accent);border-radius:999px;padding:.05rem .45rem;font-size:.78rem}
.entity-link{border:none;padding:.1rem .2rem}.entity-link:hover,.entity-chip:hover{border-color:var(--accent)}
.detail{border:1px solid var(--line);border-radius:9px;padding:.75rem;margin:.7rem 0;background:color-mix(in srgb,var(--bg) 35%,transparent)}
.detail h3{margin:0 0 .35rem;font-size:1rem}.detail .description{line-height:1.45}.alias-list{display:flex;gap:.35rem;flex-wrap:wrap;margin:.4rem 0}.alias-list span{border:1px solid var(--line);border-radius:999px;padding:.05rem .45rem;font-size:.75rem;color:var(--dim)}
#stats{color:var(--dim);font-size:.85rem;margin-bottom:1rem}
.mem{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem;margin-bottom:.5rem}
.mem .meta{color:var(--dim);font-size:.78rem;margin-top:.35rem;display:flex;gap:.8rem;flex-wrap:wrap}
.mem .del,.mem .edit{float:right;border:none;background:none;color:var(--dim)}.mem .del:hover{color:var(--warn)}.mem .edit:hover{color:var(--accent)}
.tag{border:1px solid var(--line);border-radius:999px;padding:0 .5rem}
.memory-type{display:inline-flex;align-items:center;gap:.3rem;border-color:currentColor;font-weight:600}
.memory-type.semantic{color:var(--semantic)}.memory-type.procedural{color:var(--procedural)}
.memory-type.episodic{color:var(--episodic)}.memory-type.working{color:var(--working)}
.type-symbol{display:inline-block;width:.48rem;height:.48rem;background:currentColor;flex:none}
.memory-type.semantic .type-symbol{border-radius:50%}
.memory-type.episodic .type-symbol{width:0;height:0;background:none;border-left:.28rem solid transparent;border-right:.28rem solid transparent;border-bottom:.5rem solid currentColor}
.memory-type.working .type-symbol{background:none;border:1px solid currentColor;transform:rotate(45deg)}
button.tagfilter{background:none;color:inherit;font:inherit;cursor:pointer}
.about-tabs{display:flex;gap:.4rem;flex-wrap:wrap;margin:.9rem 0}
.about-tabs button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent)}
.apanel .step{padding:.7rem 0;border-bottom:1px solid var(--line)}
.apanel .step:last-child{border-bottom:none}
.apanel .step h3{margin:0 0 .25rem;font-size:.95rem;font-weight:600}
.apanel .step p{margin:0;line-height:1.5;color:var(--dim)}
.glossary{margin:.2rem 0;display:grid;grid-template-columns:1fr;gap:0}
@media(min-width:44rem){.glossary{grid-template-columns:11rem 1fr;column-gap:1rem}}
.glossary dt{font-weight:600;padding:.45rem 0 0}
.glossary dd{margin:0;padding:.45rem 0;color:var(--dim);line-height:1.5;border-bottom:1px solid var(--line)}
@media(max-width:44rem){.glossary dd{padding-top:.1rem}}
button.tagfilter:hover{border-color:var(--accent);color:var(--accent)}
.meta button.distill{border:1px solid var(--warn);border-radius:999px;padding:0 .5rem;background:none;color:var(--warn);font-size:inherit}
.meta button.distill:hover{background:var(--warn);color:var(--bg)}
textarea{width:100%;min-height:70px;margin-bottom:.4rem}
.empty{color:var(--dim);padding:2rem;text-align:center}
#mapwrap{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.4rem;margin-bottom:.7rem;overflow:hidden}
#map{display:block;width:100%;border-radius:8px}
#mapwrap:fullscreen,#mapwrap.maxed{padding:0;border:0;border-radius:0;background:var(--bg)}
#mapwrap:fullscreen #map,#mapwrap.maxed #map{border-radius:0}
#mapwrap.maxed{position:fixed;inset:0;z-index:99999}
.gx-ctrl{position:absolute;top:.6rem;right:.6rem;display:flex;gap:.4rem;align-items:flex-start;z-index:3}
.gx-ctrl button,.gx-types summary{background:color-mix(in srgb,var(--panel) 68%,transparent);border:1px solid var(--line);color:var(--dim);border-radius:7px;padding:.3rem .5rem;font-size:.72rem;cursor:pointer;backdrop-filter:blur(5px);line-height:1;list-style:none}
.gx-ctrl button:hover,.gx-types summary:hover{color:var(--accent);border-color:var(--accent)}
.gx-ctrl button[aria-pressed="true"],.gx-types summary[aria-pressed="true"],.gx-types[open] summary{color:var(--accent);border-color:var(--accent);background:color-mix(in srgb,var(--accent) 11%,var(--panel))}
.gx-types{position:relative}.gx-types summary::-webkit-details-marker{display:none}
.gx-type-menu{position:absolute;right:0;top:1.9rem;width:15rem;max-height:min(25rem,70vh);overflow:auto;background:color-mix(in srgb,var(--panel) 96%,transparent);border:1px solid var(--line);border-radius:9px;padding:.55rem;box-shadow:0 .7rem 2rem rgba(0,0,0,.28);backdrop-filter:blur(8px)}
.gx-type-option{display:flex;align-items:center;gap:.45rem;padding:.22rem .1rem;color:var(--text);font-size:.75rem;white-space:nowrap}.gx-type-option input{width:auto;margin:0}.gx-type-option .cnt{margin-left:auto}
.gx-type-actions{display:flex;gap:.35rem;margin-top:.45rem;padding-top:.45rem;border-top:1px solid var(--line)}
.gx-type-actions button{flex:1}
.gx-read{position:absolute;top:.6rem;left:.6rem;z-index:3;font-size:.75rem;color:var(--dim);background:color-mix(in srgb,var(--panel) 60%,transparent);border:1px solid var(--line);border-radius:7px;padding:.32rem .6rem;backdrop-filter:blur(5px);max-width:62%;pointer-events:none;opacity:0;transition:opacity .15s}
.gx-read.on{opacity:1}.gx-read b{color:var(--text)}
.gx-stat{position:absolute;bottom:.5rem;left:.7rem;z-index:3;font-size:.68rem;color:var(--dim);opacity:.55;pointer-events:none}
.gx-legend{position:absolute;right:.7rem;bottom:1.35rem;z-index:3;display:flex;gap:.45rem .7rem;align-items:center;justify-content:flex-end;flex-wrap:wrap;max-width:calc(100% - 1.4rem);color:var(--dim);font-size:.66rem;pointer-events:none;opacity:.8}
.gx-legend span{display:inline-flex;gap:.28rem;align-items:center}.gx-shape{display:inline-block;width:.48rem;height:.48rem;background:var(--accent)}
.gx-shape.semantic{border-radius:50%;background:var(--semantic)}.gx-shape.procedural{border-radius:1px;background:var(--procedural)}.gx-shape.episodic{width:0;height:0;background:none;border-left:.28rem solid transparent;border-right:.28rem solid transparent;border-bottom:.5rem solid var(--episodic)}
.gx-shape.working{transform:rotate(45deg);background:transparent;border:1px solid var(--working)}
.gx-empty{position:absolute;inset:3rem 1rem 2rem;display:grid;place-items:center;color:var(--dim);font-size:.82rem;text-align:center;pointer-events:none}
.gx-empty[hidden]{display:none}
.map-entity-detail{margin:-.15rem 0 .8rem;padding:.75rem .85rem;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.map-entity-detail[hidden]{display:none}.map-entity-detail h3{margin:0 0 .35rem;font-size:1rem}.map-entity-actions{display:grid;grid-template-columns:minmax(10rem,1fr) auto auto;gap:.45rem;align-items:center;margin-top:.7rem;padding-top:.65rem;border-top:1px solid var(--line)}
.map-entity-actions .danger{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 55%,var(--line))}
@media(max-width:44rem){.map-entity-actions{grid-template-columns:1fr}.map-entity-actions button{width:100%}}
</style></head><body><main>
<h1><svg viewBox="0 0 64 64" width="22" height="22" aria-hidden="true" style="color:var(--accent);vertical-align:-3px;margin-right:.35rem"><path d="M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50" fill="none" stroke="currentColor" stroke-width="7" stroke-linecap="round"/><circle cx="47" cy="10.5" r="4.5" fill="currentColor"/><circle cx="56" cy="20" r="3.2" fill="currentColor" opacity=".85"/><circle cx="57.5" cy="30" r="2.2" fill="currentColor" opacity=".7"/></svg><span>Mem</span>ry <small style="color:var(--dim);font-weight:400">memory dashboard</small>
<span class="datalinks"><a class="knowledge-link" href="#" onclick="openKnowledge();return false" title="Open Knowledge to browse and maintain tags, people, things, and forgotten memories.">Knowledge</a> · <a href="#" onclick="exportMemories();return false" title="Download a lossless Memry backup containing memories, entity links, provenance, relations, timestamps, IDs, and history for this account.">export</a> · <a href="#" id="importbtn" onclick="document.getElementById('importfile').click();return false" title="Restore a lossless Memry backup exactly. Legacy memory-only JSON and JSONL files remain supported as additive imports.">import</a> · <a href="#" onclick="openAbout();return false" title="What Memry does with what you tell it, in plain words.">about</a> <span class="account-links">· <span title="signed-in account">@__WHOAMI__</span> · <a href="/logout" title="Sign out of this Memry dashboard.">sign out</a></span></span></h1>
<div id="stats">loading…</div>
<div class="bar">
  <span class="qwrap"><input id="q" placeholder="search memories…" oninput="toggleClear()">
    <button id="qclear" type="button" title="clear search and show all" onclick="clearSearch()">✕</button></span>
  <button class="primary" onclick="search()" title="Search memories using the text and filters above.">Search</button>
  <button class="toggle" id="filterbtn" onclick="togglePanel('filters')" title="Filter by date, tag, or person/thing." aria-label="Filters"><svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M1 2.5A.5.5 0 0 1 1.5 2h13a.5.5 0 0 1 .38.82L10 8.7V13a.5.5 0 0 1-.72.45l-3-1.5A.5.5 0 0 1 6 11.5V8.7L1.12 2.82A.5.5 0 0 1 1 2.5Z"/></svg><span id="filterdot" hidden>•</span></button>
  <button class="toggle" id="addbtn" onclick="togglePanel('add')" title="Add a memory." aria-label="Add a memory">+</button>
  <button class="toggle" id="mapbtn" onclick="togglePanel('map')" title="Show or hide the memory map.">Map</button>
</div>
<div class="search-filters" id="filterpanel" aria-label="Search filters" hidden>
  <label>Date range<span class="range">
    <input id="filter-date" type="date" title="from" onchange="toggleClear()">
    <span class="cnt">to</span>
    <input id="filter-date-to" type="date" title="to" onchange="toggleClear()">
  </span></label>
  <label>Tags <span class="picked" id="topiccount"></span>
    <select id="filter-topic" multiple size="4" title="ctrl/cmd-click for several" onchange="toggleClear()"></select></label>
  <label>People or things <span class="picked" id="entitycount"></span>
    <select id="filter-entity" multiple size="4" title="ctrl/cmd-click for several" onchange="toggleClear()"></select></label>
</div>
<input type="file" id="importfile" accept=".json,.jsonl,.txt,application/json" hidden onchange="importMemories(this.files[0]);this.value=''">
<div id="addpanel" hidden>
<textarea id="newmem" placeholder="Add to memory… (extraction runs if an LLM is configured)"></textarea>
<div class="bar">
  <input id="newcats" placeholder="tags, comma separated (optional)" style="flex:1;min-width:10rem">
  <button class="primary" onclick="add(true)">Add (infer)</button>
  <button onclick="add(false)">Add verbatim</button>
</div>
</div>
<div id="mapwrap" hidden><canvas id="map"></canvas>
<div class="gx-ctrl">
  <button id="mapTagsBtn" onclick="setMapMode('tags')" title="Group every active memory by tag.">Tags</button>
  <details class="gx-types" id="mapEntityFilter">
    <summary id="mapEntitiesBtn" onclick="setMapMode('entities')" title="Group memories by entity and choose which entity types appear.">Entities</summary>
    <div class="gx-type-menu">
      <div id="mapEntityTypeOptions"></div>
      <div class="gx-type-actions">
        <button type="button" onclick="setMapEntityTypes('defaults')" title="Show all entity types except concept and other.">defaults</button>
        <button type="button" onclick="setMapEntityTypes('all')" title="Show every entity type.">all</button>
        <button type="button" onclick="setMapEntityTypes('none')" title="Hide every entity type.">none</button>
      </div>
    </div>
  </details>
  <button id="fsBtn" title="Open the map fullscreen." aria-label="Fullscreen">⤢</button>
</div>
<div class="gx-read" id="mapread"></div><div class="gx-stat" id="mapstat"></div>
<div class="gx-legend" aria-label="Memory type shapes">
  <span><i class="gx-shape semantic"></i>semantic</span><span><i class="gx-shape procedural"></i>procedural</span>
  <span><i class="gx-shape episodic"></i>episodic</span><span><i class="gx-shape working"></i>working</span>
</div>
<div class="gx-empty" id="mapempty" hidden></div></div>
<section class="map-entity-detail" id="mapentitydetail" aria-live="polite" hidden></section>
<div id="list"></div>
<div class="modal" id="aboutmodal"><div class="sheet" style="width:min(96vw,54rem)">
<h2><button class="x" onclick="closeAbout()" title="close">x</button>About Memry</h2>
<div class="about-tabs">
  <button id="atab-how" onclick="showAbout('how')">How it works</button>
  <button id="atab-words" onclick="showAbout('words')">Glossary</button>
  <button id="atab-server" onclick="showAbout('server')">This server</button>
</div>
<section class="apanel" id="apanel-how">
  <div class="step"><h3>Nothing you say is thrown away</h3>
  <p>Every message you save is kept exactly as you wrote it, and that copy is never edited. Everything else here is worked out from it, so if a later step gets something wrong, the original is still there to redo it from.</p></div>

  <div class="step"><h3>Long messages get split into single facts</h3>
  <p>Tell it three things in one paragraph and you get three memories, not one blob. Each one can then be found, corrected or deleted on its own, without disturbing the other two.</p></div>

  <div class="step"><h3>New facts are checked against what is already known</h3>
  <p>Say the same thing twice and the second one is dropped. Add a detail and the existing memory is filled in. Contradict yourself and the old version is retired in favour of the new one - kept, but marked as no longer true, so you can see what changed and when.</p></div>

  <div class="step"><h3>Every memory gets tags</h3>
  <p>Tags are subjects, and they are deliberately narrow: <i>liver health</i> rather than <i>health</i>. A tag as broad as "health" would put your gym routine and your blood tests in the same bucket, which helps nobody. Click any tag on a memory to pull up everything filed under it.</p></div>

  <div class="step"><h3>People and things get their own pages</h3>
  <p>When a name keeps coming up - someone you work with, a company, a project - it gets an entry that gathers everything known about it in one place, with the memories that back each part. Two people with the same name stay separate until there is real reason to think they are one person.</p></div>

  <div class="step"><h3>It also remembers how things connect</h3>
  <p>"Ada works on Helios" and "Helios runs on Postgres" are stored as links rather than sentences. That is why asking what database Ada's project uses can find the answer, even though the memory that holds it never mentions Ada.</p></div>

  <div class="step"><h3>Search does three things at once</h3>
  <p>It matches on meaning, so "blood test" finds a memory about liver results. It matches on exact words, so codes, names and numbers still work. And it follows those links between things. The three sets of results are combined, with newer and more important memories nudged up.</p></div>

  <div class="step"><h3>Old events sink; standing instructions do not</h3>
  <p>A note about a meeting last spring gradually stops crowding your results. "Always answer briefly" keeps its weight, because a rule does not expire the way an appointment does. Nothing disappears on its own - it just stops coming first.</p></div>

  <div class="step"><h3>It tidies up on a schedule</h3>
  <p>Duplicate people get merged once it is clear they are the same. Names that were never really things - a bare year, an amount, a greeting - get removed. Tags that have drifted into two spellings get flagged. Anything less than obvious waits for you to confirm it, under Knowledge &gt; Upkeep, where you can also switch each of these off.</p></div>

  <div class="step"><h3>Deleting takes two steps, on purpose</h3>
  <p>Deleting a memory takes it out of search but keeps the record, in Knowledge &gt; Forgotten. From there you can put it back, or delete it for good. Only that second step is permanent.</p></div>

  <div class="step"><h3>You can take everything with you</h3>
  <p>Export downloads the whole store - memories, people, links, history, timestamps - in a form that restores exactly. Every change ever made to a memory is recorded, so you can always see what happened to it.</p></div>
</section>
<section class="apanel" id="apanel-words" hidden>
  <p class="hint">The words that show up around the app, in plain terms.</p>
  <dl class="glossary">
    <dt>Memory</dt><dd>One fact, kept on its own. The thing everything else here is about.</dd>
    <dt>Episode</dt><dd>The raw message you originally sent, stored word for word and never edited. Memories are worked out from episodes; if extraction ever needs redoing, this is what it is redone from. One message can produce several memories, which is why the two numbers differ.</dd>
    <dt>Tag</dt><dd>A subject a memory is filed under, like <i>2026 taxes</i>. A memory can have a few. Clicking one filters to everything under it.</dd>
    <dt>Entity (a "person or thing")</dt><dd>Someone or something that keeps coming up, with its own page collecting what is known about it.</dd>
    <dt>Entity type</dt><dd>What kind of thing it is: person, organization, project, product, place, event, document, code, or concept. Used to keep unrelated things with the same name apart, and to group the list - it does not change search ranking.</dd>
    <dt>Relation</dt><dd>A link between two entities, like "Ada works on Helios". Search follows these to reach answers that share no words with your question.</dd>
    <dt>Invalidated</dt><dd>A memory that is no longer treated as true, but is still on file. Happens when you delete it, or when something you said later contradicted it. It stops appearing in search; it does not stop existing.</dd>
    <dt>Superseded</dt><dd>An invalidated memory that was replaced by a specific newer one - the old version of a fact you updated. It stays attached to its replacement as history, which is why it is not listed under Forgotten.</dd>
    <dt>Forgotten</dt><dd>An invalidated memory that nothing replaced - you deleted it, or it faded out. These are listed on their own tab, where you can restore one or delete it permanently.</dd>
    <dt>Importance</dt><dd>How much weight a memory carries in results, from 0 to 1. Set when it is saved.</dd>
    <dt>Decay</dt><dd>The slow drop in a memory's pull on results as it ages. Dated events fade fastest, standing rules barely at all.</dd>
    <dt>Consolidation</dt><dd>Merging several memories that say the same thing into one that keeps every detail. The originals become forgotten, not deleted.</dd>
    <dt>Merge proposal</dt><dd>Two entities that might be the same, waiting for you to say yes or no.</dd>
    <dt>Synthetic tag</dt><dd>A broader tag Memry grouped others under, for browsing. Off by default, because searching under the narrower tag works better.</dd>
    <dt>Embedding</dt><dd>A memory turned into numbers so that similar meanings sit near each other. This is what makes "blood test" find "liver results".</dd>
    <dt>Distillation</dt><dd>Turning a raw saved message into separate facts. Usually happens moments after saving; a memory says "not distilled" if it is still waiting.</dd>
    <dt>Namespace</dt><dd>Whose memories these are. Yours are separate from every other account's.</dd>
  </dl>
</section>
<section class="apanel" id="apanel-server" hidden>
  <p class="hint">What this particular Memry is running.</p>
  <div id="serverinfo"></div>
</section>
</div></div>
<div class="modal" id="knowmodal"><div class="sheet">
<h2><button class="x" onclick="closeKnowledge()" title="close">x</button>Knowledge</h2>
<p class="hint">Tags classify memories. People and things are stable entity hubs with aliases, a bounded description, supporting memories, and their evidence-linked relations.</p>
<div class="knowledge-tabs">
  <button id="ktab-topics" onclick="showKnowledge('topics')">Tags</button>
  <button id="ktab-entities" onclick="showKnowledge('entities')">People &amp; things</button>
  <button id="ktab-forgotten" onclick="showKnowledge('forgotten')">Forgotten</button>
  <button id="ktab-maintenance" onclick="showKnowledge('maintenance')">Upkeep</button>
</div>
<section class="kpanel" id="kpanel-topics">
  <div class="tagbar">
    <span class="sel" id="tagsel">none selected</span>
    <button onclick="suggestMerges()" title="let the LLM propose duplicate or variant tags to merge">Suggest merges</button>
    <button onclick="mergeTags()" title="combine the checked tags into one">Combine selected...</button>
    <input id="tagsearch" type="search" placeholder="filter tags..." oninput="renderTags()"
           title="show only tags whose name contains this" style="flex:1;min-width:7rem">
  </div>
  <div id="tagsuggest"></div><div id="taglist"></div>
</section>
<section class="kpanel" id="kpanel-entities" hidden>
  <div class="entity-split">
    <div class="entity-main">
      <div class="tagbar"><span class="sel" id="entcount"></span>
        <button onclick="backfillTypes()" title="classify entities that have no type yet">Backfill types</button></div>
      <div id="entlist"></div>
      <h2 style="font-size:.95rem;margin-top:1.1rem">Merge proposals</h2><div id="proplist"></div>
    </div>
    <aside class="entity-side" id="entitydetail"></aside>
  </div>
</section>
<section class="kpanel" id="kpanel-forgotten" hidden>
  <p class="hint">Deleting a memory hides it from search but keeps the record, so nothing is lost by accident. This is where those land. Permanent deletion is only possible from here, and only for memories that are already forgotten.</p>
  <div id="forgottenlist"></div>
</section>
<section class="kpanel" id="kpanel-maintenance" hidden>
  <h2 style="font-size:.95rem;margin-top:.2rem">Automatic passes</h2>
  <p class="hint">What Memry does to your memories on its own. Switch any of them off, or run one right now.</p>
  <div id="upkeeplist"></div>
  <h2 style="font-size:.95rem;margin-top:1.1rem">Tag health</h2>
  <p class="hint">A tag that has quietly split in two caps what any search can find under it, because filtering drops the rest of the evidence before ranking starts.</p>
  <div id="taghealth"></div>
  <h2 style="font-size:.95rem;margin-top:1.1rem">Entity health</h2>
  <p class="hint">Some extracted names are not really things - bare dates, amounts, style instructions. Obvious cases are removed automatically by the self-healing pass; the rest can be reviewed here. Removing an entity never touches the memories behind it.</p>
  <div id="entityjunk"></div>
  <h2 style="font-size:.95rem;margin-top:1.1rem">Consolidate duplicate memories</h2>
  <p class="hint">Finds memories that record the same fact more than once and merges them into one that keeps every detail. The originals are kept and linked, never deleted.</p>
  <div class="tagbar">
    <label>Similarity
      <select id="conthresh">
        <option value="0.95">very close only (0.95)</option>
        <option value="0.90" selected>close (0.90)</option>
        <option value="0.85">looser (0.85)</option>
      </select>
    </label>
    <button onclick="previewConsolidation()" title="show what would be merged, without changing anything">Preview</button>
    <button id="conapply" onclick="applyConsolidation()" disabled title="apply the merges shown above">Apply shown merges</button>
  </div>
  <div id="conresult"></div>
</section>
</div></div>
</main><script>
// Auth rides the session cookie set at /login; no key to paste anymore.
const H = {'Content-Type':'application/json'};

async function api(path, opts={}){
  const r = await fetch(path,{headers:H,...opts});
  if(r.status===401){ location.href='/login'; throw new Error('unauthorized'); }
  return r.json();
}
function esc(s){return (s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
// Add form is opt-in, map is opt-out; the choice sticks per browser.
const panels={add:localStorage.getItem('memry_show_add')==='1',
              map:localStorage.getItem('memry_show_map')!=='0',
              filters:localStorage.getItem('memry_show_filters')==='1'};
function syncPanels(){
  document.getElementById('addpanel').hidden=!panels.add;
  document.getElementById('addbtn').setAttribute('aria-pressed',panels.add);
  document.getElementById('mapbtn').setAttribute('aria-pressed',panels.map);
  syncMapModeButtons();
  // Filters are collapsed by default; an active one is still shown as a dot on
  // the button, so a filter can never be silently applied behind a closed panel.
  document.getElementById('filterpanel').hidden=!panels.filters;
  document.getElementById('filterbtn').setAttribute('aria-pressed',panels.filters);
  drawMap();
}
function togglePanel(name){
  panels[name]=!panels[name];
  localStorage.setItem('memry_show_'+name,panels[name]?'1':'0');
  syncPanels();
}
let current=[],activeMapKey=null,haveMore=false,editingId=null,hoverMapKey=null,searchActive=false;
let hoverFocusTag=null,hoverFocusMix=0,hoverFadeStarted=0;
const HOVER_FADE_MS=500;
const cats=m=>((m.categories&&m.categories.length)?m.categories:['(untagged)']).map(c=>String(c).toLowerCase());
const moreBar=()=>haveMore
  ? '<div class="bar" id="morebar"><button onclick="loadAll(true)">Load more</button></div>' : '';
// `appendFrom` renders only the newly arrived tail. Rebuilding the whole list
// on every "load more" is quadratic: reaching 10k memories a hundred at a time
// would re-render half a million cards. Appending costs only what arrived.
function render(items,appendFrom){
  current=items;drawMap();
  const el=document.getElementById('list');
  if(!items.length&&!haveMore){
    el.innerHTML='<div class="empty">'+(searchActive?'No memories match this search.':'No memories yet.')+'</div>';return;
  }
  const card=m=>m.id===editingId?editCard(m):viewCard(m);
  if(appendFrom!==undefined&&appendFrom>0){
    document.getElementById('morebar')?.remove();
    el.insertAdjacentHTML('beforeend',
      items.slice(appendFrom).map(card).join('')+moreBar());
    return;
  }
  el.innerHTML=items.map(card).join('')+moreBar();
}
function normalizedMemoryType(m){
  const type=String(m.memory_type||m.type||'semantic').toLowerCase();
  return ['semantic','procedural','episodic','working'].includes(type)?type:'semantic';
}
function memoryTypeBadge(m){
  const type=normalizedMemoryType(m);
  return `<span class="tag memory-type ${type}"><i class="type-symbol" aria-hidden="true"></i>${type}</span>`;
}
function viewCard(m){
  return `<div class="mem"><button class="del" title="forget" onclick="del('${m.id}')">✕</button>
   <button class="edit" title="edit" onclick="startEdit('${m.id}')">✎</button>
   <div>${esc(m.content)}</div>
   <div class="meta">${memoryTypeBadge(m)}
   ${(m.categories||[]).map(c=>`<button class="tag tagfilter" title="show everything tagged #${esc(String(c))}" onclick='filterByTag(${JSON.stringify(String(c))})'>#${esc(String(c))}</button>`).join('')}
   ${(m.entity_links||[]).map(entity=>`<button class="entity-chip" onclick='openEntity(${JSON.stringify(entity.id)})'>${esc(entity.name)}</button>`).join('')}
   <span>@${esc(m.user_id||'(no user)')}</span>
   <span>imp ${(m.importance??0.5).toFixed(2)}</span>
   ${m.score!==undefined?`<span>score ${m.score.toFixed(3)}</span>`:''}
   <span>${(m.updated_at||m.created_at||'').slice(0,10)}</span>
   ${m.invalid_at?'<span style="color:var(--warn)">invalidated</span>':''}
   ${m.saving?'<span class="cnt" title="the edit is saved; entity links are being refreshed">saving...</span>':''}
   ${m.metadata&&m.metadata.pending_distillation&&!m.invalid_at?(m.metadata._enrichment?`<span title="The exact text is saved and searchable while background enrichment runs.">${esc(m.metadata._enrichment.status||'enrichment pending')}</span>`:`<button class="distill" onclick="distill('${m.id}')" title="Saved verbatim because extraction was skipped. Distill into discrete facts now.">not distilled ↻</button>`):''}</div></div>`;
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

// ---- galaxy map: memories grouped as tag or entity planets ----------------
// Planet size reflects how many loaded memories belong to a group. The small
// orbiting markers are those memories: circle = semantic, square = procedural,
// triangle = episodic, diamond = working. Links connect groups that co-occur.
const hashCode=s=>{let h=0;for(let i=0;i<s.length;i++)h=((h<<5)-h+s.charCodeAt(i))|0;return Math.abs(h)};
const mulberry=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
let mapMode=localStorage.getItem('memry_map_mode')==='entities'?'entities':'tags';
let mapVisible=true;
if('IntersectionObserver'in window){
  new IntersectionObserver(entries=>{
    mapVisible=entries.some(e=>e.isIntersecting);
    if(mapVisible&&!gRAF&&panels.map&&G&&!reducedMotion){
      gRAF=requestAnimationFrame(galaxyFrame);
    }
  },{rootMargin:'120px'}).observe(document.getElementById('mapwrap'));
}
let G=null,gRAF=0,gPulses=[],gMaxed=false;
const gStars=Array.from({length:230},(_,i)=>{const r=mulberry(i*2654435761+11);const k=r();
  return{x:r(),y:r(),s:.3+r()*1.4,a:.2+r()*.7,ph:r()*6.28,sp:.3+r()*.7,hue:k>.94?36:(k>.84?176:(k>.78?252:null)),big:r()>.965};});
const gDust=Array.from({length:150},(_,i)=>{const r=mulberry(i*97+5);
  return{a:r()*Math.PI*2,rad:0.12+r()*0.95,off:(r()-0.5)*0.3,s:0.35+r()*0.75,al:0.05+r()*0.10,teal:r()>0.8};});
const gTone=(n,dark)=>{const j=((n.seed%1000)/1000-0.5);
  if(n.zone==='core')return{h:36+j*16,s:dark?95:80,l:dark?66:42};
  if(n.zone==='belt')return{h:172+j*34,s:dark?75:65,l:dark?60:36};
  return{h:248+j*40,s:dark?60:50,l:dark?74:44};};
const hsla=(c,a,dl)=>'hsla('+c.h+','+c.s+'%,'+Math.max(4,Math.min(96,c.l+(dl||0)))+'%,'+a+')';
const hexA=(hex,a)=>{const v=parseInt(hex.slice(1),16);return'rgba('+((v>>16)&255)+','+((v>>8)&255)+','+(v&255)+','+a+')'};
const MAX_IDLE_EDGES=400;
let mapData=null,mapEntityTypes=null;
function knownEntityTypes(){
  if(!mapData)return[];
  return [...new Set(mapData.entities.map(node=>node.entity_type||'untyped'))].sort();
}
function defaultEntityTypes(){
  return knownEntityTypes().filter(type=>type!=='concept'&&type!=='other');
}
function saveMapEntityTypes(){
  localStorage.setItem('memry_map_entity_types',JSON.stringify([...mapEntityTypes].sort()));
}
function initializeMapEntityTypes(){
  if(mapEntityTypes!==null)return;
  let saved=null;
  try{saved=JSON.parse(localStorage.getItem('memry_map_entity_types')||'null')}catch(error){}
  mapEntityTypes=new Set(Array.isArray(saved)?saved:defaultEntityTypes());
}
function renderMapEntityTypes(){
  const filter=document.getElementById('mapEntityFilter');
  if(mapMode!=='entities')filter.open=false;
  if(!mapData)return;
  initializeMapEntityTypes();
  const counts={};
  mapData.entities.forEach(node=>{
    const type=node.entity_type||'untyped';counts[type]=(counts[type]||0)+1;
  });
  document.getElementById('mapEntityTypeOptions').innerHTML=knownEntityTypes().map(type=>
    '<label class="gx-type-option"><input type="checkbox" data-entity-type="'+esc(type)+'"'
      +(mapEntityTypes.has(type)?' checked':'')+'>'
      +'<span>'+esc(type)+'</span><span class="cnt">'+counts[type]+'</span></label>'
  ).join('')||'<div class="hint">No entity types yet.</div>';
}
function toggleMapEntityType(type,checked){
  if(checked)mapEntityTypes.add(type);else mapEntityTypes.delete(type);
  if(activeMapKey&&G&&G.byKey[activeMapKey]&&G.byKey[activeMapKey].entityType===type){
    activeMapKey=null;clearMapEntityDetail();
  }
  saveMapEntityTypes();drawMap();
}
function handleMapEntityTypeChange(event){
  const input=event.target;
  if(!input||!input.matches('input[data-entity-type]'))return;
  toggleMapEntityType(input.dataset.entityType,input.checked);
}
document.getElementById('mapEntityTypeOptions').addEventListener(
  'change',handleMapEntityTypeChange
);
function setMapEntityTypes(mode){
  const types=mode==='all'?knownEntityTypes():(mode==='none'?[]:defaultEntityTypes());
  mapEntityTypes=new Set(types);activeMapKey=null;clearMapEntityDetail();saveMapEntityTypes();
  renderMapEntityTypes();drawMap();
}
async function loadMapData(){
  if(knowledgeMapSuspended){mapData=null;return}
  const data=await api('/api/v1/map');
  if(knowledgeMapSuspended){mapData=null;return}
  mapData=data;
  if(mapEntityTypes===null)initializeMapEntityTypes();
  if(activeMapKey){
    const keys=new Set([...data.tags,...data.entities].map(node=>node.key));
    if(!keys.has(activeMapKey)){activeMapKey=null;clearMapEntityDetail()}
  }
  renderMapEntityTypes();drawMap();
}
function syncMapModeButtons(){
  document.getElementById('mapTagsBtn').setAttribute('aria-pressed',mapMode==='tags');
  document.getElementById('mapEntitiesBtn').setAttribute('aria-pressed',mapMode==='entities');
  renderMapEntityTypes();
}
function setMapMode(mode){
  if(mode!=='tags'&&mode!=='entities')return;
  mapMode=mode;localStorage.setItem('memry_map_mode',mode);
  activeMapKey=null;clearMapEntityDetail();updateHover(null);syncMapModeButtons();drawMap();
}
function buildGalaxy(data){
  let source=mapMode==='entities'?data.entities:data.tags;
  if(mapMode==='entities'){
    initializeMapEntityTypes();
    source=source.filter(node=>mapEntityTypes.has(node.entity_type||'untyped'));
  }
  if(!source.length)return null;
  const vals=source.map(node=>node.count);
  const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
  const sd=Math.sqrt(vals.reduce((a,c)=>a+(c-mean)**2,0)/vals.length);
  const coreMin=mean+2*sd,rimMax=Math.max(1,mean-2*sd),maxC=Math.max(...vals);
  const fb=!vals.some(count=>count>=coreMin);
  const ZF={core:1.0,belt:1.28,rim:1.55};
  const nodes=source.map(raw=>{
    const zone=(fb?raw.count===maxC:raw.count>=coreMin)?'core':(raw.count<=rimMax?'rim':'belt');
    return{...raw,typeCounts:raw.type_counts||{},zone,
      entityType:raw.entity_type||'untyped',
      radius:Math.min(34,(9+5*Math.sqrt(raw.count))*ZF[zone]),
      seed:hashCode(raw.key),h:0};
  }).sort((a,b)=>a.label.localeCompare(b.label));
  const index=new Map(nodes.map((node,i)=>[node.key,i]));
  const rawEdges=mapMode==='entities'?data.entity_edges:data.tag_edges;
  const edges=rawEdges
    .filter(edge=>index.has(edge.a)&&index.has(edge.b))
    .map(edge=>({a:index.get(edge.a),b:index.get(edge.b),weight:Math.min(3,edge.weight)}))
    .sort((a,b)=>b.weight-a.weight||a.a-b.a||a.b-b.b);
  const neigh={},edgesByNode={};
  edges.forEach(edge=>{
    const a=nodes[edge.a].key,b=nodes[edge.b].key;
    (neigh[a]??=new Set()).add(b);(neigh[b]??=new Set()).add(a);
    (edgesByNode[a]??=[]).push(edge);(edgesByNode[b]??=[]).push(edge);
  });
  const BANDS={core:[0.02,0.16],belt:[0.30,0.62],rim:[0.66,0.99]};
  const PHASE={core:0,belt:0.7,rim:1.4},PACK={core:4,belt:16,rim:22},GOLDEN=2.399963229728653;
  for(const zone of['core','belt','rim']){
    const ring=nodes.filter(node=>node.zone===zone);
    ring.sort((a,b)=>b.count-a.count||a.label.localeCompare(b.label));
    const N=ring.length,lo=BANDS[zone][0],hi=BANDS[zone][1];
    const shrink=Math.max(0.4,Math.min(1,1/Math.sqrt(Math.max(1,N)/PACK[zone])));
    const dense=N>PACK[zone]*0.8;
    ring.forEach((node,k)=>{const rnd=mulberry(node.seed);
      node.radius*=shrink;
      if(zone==='core'){node.rFrac=N===1?0:0.13;node.ang=PHASE.core+k*(Math.PI*2/N);}
      else if(dense){node.rFrac=lo+(hi-lo)*Math.sqrt((k+0.5)/N);node.ang=PHASE[zone]+k*GOLDEN;}
      else{node.ang=PHASE[zone]+k*(Math.PI*2/N)+(rnd()-0.5)*0.22;node.rFrac=lo+(hi-lo)*(0.35+0.5*rnd());}});
  }
  return{
    nodes,edges,neigh,edgesByNode,idleEdges:edges.slice(0,MAX_IDLE_EDGES),
    byKey:Object.fromEntries(nodes.map(node=>[node.key,node])),fb,
    total:mapMode==='entities'?(data.entity_memories??data.memories):data.memories,
    mode:mapMode,
  };
}
function displayedGalaxyEdges(graph,selected,hovered){
  if(selected)return graph.edgesByNode[selected.key]||[];
  if(!hovered)return graph.idleEdges;
  const displayed=[...graph.idleEdges],seen=new Set(displayed);
  for(const edge of graph.edgesByNode[hovered.key]||[]){
    if(!seen.has(edge)){seen.add(edge);displayed.push(edge)}
  }
  return displayed;
}
function drawMap(){
  const wrap=document.getElementById('mapwrap'),empty=document.getElementById('mapempty');
  const visible=panels.map&&!knowledgeMapSuspended&&mapData&&mapData.memories;
  wrap.hidden=!visible;syncMapEntityDetailVisibility();
  if(!visible){G=null;empty.hidden=true;if(gRAF){cancelAnimationFrame(gRAF);gRAF=0}return}
  G=buildGalaxy(mapData);sizeGalaxy();syncMapModeButtons();
  empty.hidden=!!G;
  if(!G){
    empty.textContent=mapMode==='entities'
      ?'No entities match the selected types.'
      :'No tags are available.';
    const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
    if(gRAF){cancelAnimationFrame(gRAF);gRAF=0}return;
  }
  galaxyRead();
  if(reducedMotion)galaxyFrame(performance.now());
  else if(!gRAF)gRAF=requestAnimationFrame(galaxyFrame);
}function sizeGalaxy(){
  const wrap=document.getElementById('mapwrap'),canvas=document.getElementById('map');
  const big=document.fullscreenElement===wrap||gMaxed;
  let width,height;
  if(big){width=wrap.clientWidth;height=wrap.clientHeight;}
  else{width=Math.max(300,wrap.clientWidth-10);height=Math.max(380,Math.min(620,width*0.56));}
  const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  canvas.style.width=width+'px';canvas.style.height=height+'px';
  canvas.getContext('2d').setTransform(dpr,0,0,dpr,0,0);
  if(G){G.W=width;G.H=height;G.CX=width/2;G.CY=height/2;G.RX=width/2-46;G.RY=height/2-42;}
}
// hover/filter info as a corner overlay; a faint tag/memory count sits bottom-left
function galaxyRead(){
  const readEl=document.getElementById('mapread'),statEl=document.getElementById('mapstat');
  if(!G){readEl.classList.remove('on');return}
  const node=activeMapKey?G.byKey[activeMapKey]:(hoverMapKey?G.byKey[hoverMapKey]:null);
  if(node){
    const types=Object.entries(node.typeCounts).map(([type,count])=>count+' '+type).join(', ');
    const heading=node.kind==='tag'?'#'+node.label:node.label+' · '+node.entityType;
    readEl.innerHTML='<b>'+esc(heading)+'</b> · '+node.count+' memor'+(node.count===1?'y':'ies')
      +(types?' · '+types:'')+(activeMapKey===node.key?' · filtering':'');
    readEl.classList.add('on');
  }else readEl.classList.remove('on');
  const noun=G.mode==='entities'?'entities':'tags',linked=G.mode==='entities'?' linked':'';
  const selectedNode=activeMapKey?G.byKey[activeMapKey]:null;
  const hoveredNode=!selectedNode&&hoverMapKey?G.byKey[hoverMapKey]:null;
  const shownLinks=displayedGalaxyEdges(G,selectedNode,hoveredNode).length;
  const linkNote=G.edges.length?' · '+shownLinks+'/'+G.edges.length+' links shown':'';
  statEl.textContent=G.nodes.length+' '+noun+' · '+G.total+linked+' memories'+linkNote
    +(G.fb?' · core = largest':'');
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
  if(hoverMapKey){hoverFocusTag=hoverMapKey;hoverFocusMix=1;hoverFadeStarted=0}
  else if(hoverFocusTag){
    hoverFocusMix=reducedMotion||!hoverFadeStarted?0:Math.max(0,1-(now-hoverFadeStarted)/HOVER_FADE_MS);
    if(!hoverFocusMix)hoverFocusTag=null;
  }
  const still=reducedMotion||!!hoverFocusTag||!!activeMapKey;
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
    pts[n.key]={x:CX+n.rFrac*RX*Math.cos(n.ang),y:CY+n.rFrac*RY*Math.sin(n.ang)};
  }
  const sel=activeMapKey?G.byKey[activeMapKey]:null;
  const hov=hoverFocusTag?G.byKey[hoverFocusTag]:null;
  const hoverMix=hov?hoverFocusMix:0;
  const focusEmph=(n,f)=>{
    if(!f||n===f)return 1;
    return(G.neigh[f.key]&&G.neigh[f.key].has(n.key))?0.92:0.16;
  };
  const emph=n=>{
    let A=focusEmph(n,sel);
    if(hov)A+=(focusEmph(n,hov)-A)*hoverMix;
    return A;
  };
  // At rest only the strongest links are drawn. Hover and selection both use
  // the complete per-node index, so focusing a node never hides its links.
  const displayedEdges=displayedGalaxyEdges(G,sel,hov);
  for(const e of displayedEdges){
    const na=G.nodes[e.a],nb=G.nodes[e.b];
    const p=pts[na.key],q=pts[nb.key];
    const ca=gTone(na,dark),cb=gTone(nb,dark);
    const selTouches=!!sel&&(na===sel||nb===sel);
    let touchMix=selTouches?1:0;
    let A=selTouches?1:(sel?0.06:1);
    if(hov){
      const hovTouches=na===hov||nb===hov;
      A+=((hovTouches?1:0.06)-A)*hoverMix;
      touchMix+=((hovTouches?1:0)-touchMix)*hoverMix;
    }
    const mxp=(p.x+q.x)/2,myp=(p.y+q.y)/2;
    const cpx=mxp+(CX-mxp)*0.13,cpy=myp+(CY-myp)*0.13;
    const grad=ctx.createLinearGradient(p.x,p.y,q.x,q.y);
    grad.addColorStop(0,hsla(ca,1));grad.addColorStop(1,hsla(cb,1));
    if(dark)ctx.globalCompositeOperation='lighter';
    ctx.globalAlpha=(0.15+0.19*touchMix)*A*(dark?1:0.9);
    ctx.strokeStyle=grad;ctx.lineWidth=4.5+2.5*touchMix+e.weight*0.6;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo(cpx,cpy,q.x,q.y);ctx.stroke();
    ctx.globalAlpha=(0.5+0.45*touchMix)*A;
    ctx.lineWidth=1.1+0.7*touchMix;
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.quadraticCurveTo(cpx,cpy,q.x,q.y);ctx.stroke();
    const sparkMix=selTouches&&(hov&&hov!==sel?1-hoverMix:1);
    if(sparkMix>0.01&&!reducedMotion){
      for(let i=0;i<2+e.weight;i++){
        const tt=((t*0.00022*(0.7+0.15*i))+i/(2+e.weight))%1;
        const u=1-tt;
        const sx=u*u*p.x+2*u*tt*cpx+tt*tt*q.x,sy=u*u*p.y+2*u*tt*cpy+tt*tt*q.y;
        const cc=tt<0.5?ca:cb;
        ctx.globalAlpha=0.9*sparkMix;
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
  // Draw the hovered planet last so its disc and label are always in front.
  const planetOrder=hov?[...G.nodes.filter(node=>node!==hov),hov]:G.nodes;
  for(const n of planetOrder){
    const p=pts[n.key],x=p.x,y=p.y,A=emph(n),c=gTone(n,dark);
    const glowTarget=Math.max(activeMapKey===n.key?1:0,n===hov?hoverMix:0);
    n.h+=(glowTarget-n.h)*(reducedMotion?1:0.14);
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
    const satelliteFocus=sel||hov;
    const focusedNeighbor=satelliteFocus&&(n===satelliteFocus
      ||(G.neigh[satelliteFocus.key]&&G.neigh[satelliteFocus.key].has(n.key)));
    const showSatellites=satelliteFocus?focusedNeighbor:n.zone!=='rim';
    const satelliteTypes=showSatellites?memoryMarkerTypes(n.typeCounts,Math.min(n.count,10)):[];
    for(let i=0;i<satelliteTypes.length;i++){
      const angle=i/satelliteTypes.length*Math.PI*2-Math.PI/2+(reducedMotion?0:t*0.00008);
      const ds=1.55+(((n.seed>>3)+i*37)%10)/15;
      ctx.fillStyle=hsla(c,0.9*A,13);
      drawMemoryMarker(ctx,satelliteTypes[i],x+(n.radius+7)*Math.cos(angle),y+(n.radius+7)*Math.sin(angle),ds);
    }
    if(activeMapKey===n.key){
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
    const selLinked=sel&&(n===sel||(G.neigh[sel.key]&&G.neigh[sel.key].has(n.key)));
    const hovLinked=hov&&(n===hov||(G.neigh[hov.key]&&G.neigh[hov.key].has(n.key)));
    const lit=n.h>0.4||selLinked||(hovLinked&&hoverMix>0.04);
    if(n.radius>=19||n.h>0.05||lit){
      ctx.globalAlpha=Math.min(1,A+0.05);
      ctx.font='500 9.5px ui-sans-serif,system-ui';
      if('letterSpacing'in ctx)ctx.letterSpacing='1.5px';
      ctx.textAlign='center';ctx.textBaseline='top';
      const label=n.label.length>18?n.label.slice(0,17)+'…':n.label;
      const labelText=label.toUpperCase()+' · '+n.count,labelY=y+n.radius+8+3*n.h;
      ctx.fillStyle=lit?TEXT:hexA(DIM.length===7?DIM:'#8494ab',0.95);
      if(n===hov&&hoverMix>0.04){
        ctx.lineWidth=5;ctx.lineJoin='round';
        ctx.strokeStyle=dark?'rgba(4,6,12,0.94)':'rgba(245,247,250,0.96)';
        ctx.strokeText(labelText,x,labelY);
      }
      ctx.fillText(labelText,x,labelY);
      if('letterSpacing'in ctx)ctx.letterSpacing='0px';
    }
  }
  ctx.globalAlpha=1;
  // Only keep animating while the canvas is actually on screen. Scrolling a
  // long list pushes the map out of view, and redrawing every star and dust
  // particle at 60fps behind the viewport is what made scrolling stutter.
  if(!reducedMotion&&panels.map&&mapVisible)gRAF=requestAnimationFrame(galaxyFrame);
  else gRAF=0;
}
function memoryMarkerTypes(typeCounts,limit){
  const order=['semantic','procedural','episodic','working'];
  const entries=Object.entries(typeCounts||{}).sort((a,b)=>{
    const ai=order.indexOf(a[0]),bi=order.indexOf(b[0]);
    return(ai<0?99:ai)-(bi<0?99:bi)||a[0].localeCompare(b[0]);
  });
  const total=entries.reduce((sum,entry)=>sum+entry[1],0);
  if(!total||!limit)return[];
  return Array.from({length:limit},(_,index)=>{
    const target=(index+0.5)*total/limit;let cumulative=0;
    for(const [type,count] of entries){
      cumulative+=count;if(target<=cumulative)return type;
    }
    return entries[entries.length-1][0];
  });
}function drawMemoryMarker(ctx,type,x,y,size){
  ctx.beginPath();
  if(type==='procedural')ctx.rect(x-size,y-size,size*2,size*2);
  else if(type==='episodic'){
    ctx.moveTo(x,y-size*1.25);ctx.lineTo(x+size*1.1,y+size);ctx.lineTo(x-size*1.1,y+size);ctx.closePath();
  }else if(type==='working'){
    ctx.moveTo(x,y-size*1.3);ctx.lineTo(x+size*1.3,y);ctx.lineTo(x,y+size*1.3);ctx.lineTo(x-size*1.3,y);ctx.closePath();
  }else ctx.arc(x,y,size,0,Math.PI*2);
  ctx.fill();
}
function hitNode(event){
  if(!G)return null;
  const rect=document.getElementById('map').getBoundingClientRect();
  const x=event.clientX-rect.left,y=event.clientY-rect.top;
  let best=null,bd=1e9;
  for(const node of G.nodes){
    const px=G.CX+node.rFrac*G.RX*Math.cos(node.ang),py=G.CY+node.rFrac*G.RY*Math.sin(node.ang);
    const d=Math.hypot(px-x,py-y);
    if(d<Math.max(node.radius+8,14)&&d<bd){bd=d;best=node}
  }
  return best;
}
let mapEntityDetailRequest=0;
function syncMapEntityDetailVisibility(){
  const panel=document.getElementById('mapentitydetail');
  panel.hidden=!(panels.map&&!knowledgeMapSuspended&&mapMode==='entities'&&panel.dataset.entityId
    &&activeMapKey==='entity:'+panel.dataset.entityId);
}
function clearMapEntityDetail(){
  mapEntityDetailRequest++;
  const panel=document.getElementById('mapentitydetail');
  panel.hidden=true;panel.innerHTML='';delete panel.dataset.entityId;
}
function mapEntityTargetOptions(entityId){
  return (mapData?.entities||[])
    .filter(node=>node.entity_id&&node.entity_id!==entityId)
    .sort((a,b)=>a.label.localeCompare(b.label))
    .map(node=>`<option value="${esc(node.entity_id)}">${esc(node.label)} · ${esc(node.entity_type||'untyped')}</option>`)
    .join('');
}
async function showMapEntityDetail(entityId){
  const panel=document.getElementById('mapentitydetail'),request=++mapEntityDetailRequest;
  panel.dataset.entityId=entityId;panel.hidden=false;
  panel.innerHTML='<div class="hint">loading entity...</div>';
  try{
    const detail=await api('/api/v1/entities/'+encodeURIComponent(entityId));
    if(request!==mapEntityDetailRequest||activeMapKey!=='entity:'+entityId)return;
    const entity=detail.entity,aliases=detail.aliases||[];
    panel.innerHTML=`<h3><span id="mapentityname">${esc(entity.name)}</span> ${entity.entity_type?`<span class="syn">${esc(entity.entity_type)}</span>`:''} <button class="act" onclick='renameEntity(${JSON.stringify(entityId)})' title="Change this entity's canonical name; the old name remains an alias.">rename</button></h3>
      <div id="mapentityidentity">${entityIdentityBlock(entity,aliases)}</div>
      <div class="bar"><input id="mapaliasinput" placeholder="add an alias"><button onclick='addMapAlias(${JSON.stringify(entityId)})' title="Add another name for this entity.">Add alias</button></div>
      <div class="map-entity-actions">
        <select id="mapduplicatetarget" onchange="document.getElementById('mapduplicatebtn').disabled=!this.value" title="Choose the entity this is a duplicate of.">
          <option value="">is duplicate of...</option>${mapEntityTargetOptions(entityId)}
        </select>
        <button id="mapduplicatebtn" disabled onclick='mergeMapEntity(${JSON.stringify(entityId)})' title="Combine this entity into the selected entity; memories are preserved.">Combine</button>
        <button class="danger" onclick='removeMapEntity(${JSON.stringify(entityId)})' title="Remove this derived entity; if it has multiple memories, keep its name as a tag.">Not an entity</button>
      </div>`;
    syncMapEntityDetailVisibility();
  }catch(error){
    if(request===mapEntityDetailRequest){panel.innerHTML='<div class="hint">Could not load this entity.</div>'}
  }
}
function syncEntityIdentity(entityId,result){
  const entity=result.entity,aliases=result.aliases||[];
  const mapPanel=document.getElementById('mapentitydetail');
  if(mapPanel.dataset.entityId===entityId){
    const name=document.getElementById('mapentityname'),identity=document.getElementById('mapentityidentity');
    if(name)name.textContent=entity.name;if(identity)identity.innerHTML=entityIdentityBlock(entity,aliases);
  }
  const knowledgePanel=document.getElementById('entitydetail');
  if(knowledgePanel.dataset.entityId===entityId){
    const name=document.getElementById('knowledgeentityname'),identity=document.getElementById('knowledgeentityidentity');
    if(name)name.textContent=entity.name;if(identity)identity.innerHTML=entityIdentityBlock(entity,aliases);
  }
  const mapNode=(mapData?.entities||[]).find(node=>node.entity_id===entityId);
  if(mapNode)mapNode.label=entity.name;
  const graphNode=G&&G.byKey['entity:'+entityId];if(graphNode)graphNode.label=entity.name;
  const filterOption=[...document.getElementById('filter-entity').options].find(option=>option.value===entityId);
  if(filterOption)filterOption.textContent=entity.name;
  knowledgeNames[entityId]=entity.name;galaxyRead();
}
async function renameEntity(entityId){
  const current=(mapData?.entities||[]).find(node=>node.entity_id===entityId)?.label||knowledgeNames[entityId]||'';
  const entered=prompt(`Rename "${current}" to:`,current);
  const name=(entered||'').trim();if(!name||name===current)return;
  const result=await api('/api/v1/entities/'+encodeURIComponent(entityId),{method:'PATCH',body:JSON.stringify({name})});
  if(result.error){alert(result.error);return}
  syncEntityIdentity(entityId,result);
  await loadEntities();
}
async function addMapAlias(entityId){
  const input=document.getElementById('mapaliasinput'),alias=input.value.trim();if(!alias)return;
  const result=await api('/api/v1/entities/'+encodeURIComponent(entityId)+'/aliases',{method:'POST',body:JSON.stringify({alias})});
  input.value='';syncEntityIdentity(entityId,result);
  await loadEntities();
}
async function refreshAfterMapEntityCleanup(){
  clearMapEntityDetail();activeMapKey=null;
  [...document.getElementById('filter-entity').options].forEach(option=>option.selected=false);
  toggleClear();
  await Promise.all([loadMapData(),loadSearchFilters(),loadEntities()]);
  await search();
}
async function mergeMapEntity(entityId){
  const entityName=(mapData?.entities||[]).find(node=>node.entity_id===entityId)?.label||'this entity';
  const targetId=document.getElementById('mapduplicatetarget').value;if(!targetId)return;
  const target=mapData.entities.find(node=>node.entity_id===targetId);
  if(!target||!confirm(`Combine ${entityName} into ${target.label}? Memories and aliases will be preserved.`))return;
  await api('/api/v1/entities/merge',{method:'POST',body:JSON.stringify({keep_id:targetId,merge_id:entityId})});
  await refreshAfterMapEntityCleanup();
}
async function removeMapEntity(entityId){
  const node=(mapData?.entities||[]).find(candidate=>candidate.entity_id===entityId);
  const entityName=node?.label||'this entity';
  const fallback=(node?.count||0)>1
    ?` Its name will be kept as a tag on ${node.count} memories.`
    :' Its memories will stay untouched.';
  if(!confirm(`Mark ${entityName} as not an entity?${fallback}`))return;
  await api('/api/v1/entities/remove',{method:'POST',body:JSON.stringify({ids:[entityId],preserve_as_tag:true})});
  await refreshAfterMapEntityCleanup();
}
async function applyMapNodeFilter(node){
  const same=activeMapKey===node.key;
  for(const id of['filter-topic','filter-entity'])
    [...document.getElementById(id).options].forEach(option=>option.selected=false);
  if(same){
    activeMapKey=null;clearMapEntityDetail();toggleClear();await search();return;
  }
  const selectId=node.kind==='tag'?'filter-topic':'filter-entity';
  const value=node.kind==='tag'?node.label:node.entity_id;
  const select=document.getElementById(selectId);
  let option=[...select.options].find(candidate=>candidate.value===value);
  if(!option){option=new Option(node.label,value);select.add(option)}
  option.selected=true;activeMapKey=node.key;
  if(node.kind==='entity')showMapEntityDetail(node.entity_id);
  else clearMapEntityDetail();
  // Keep the filter panel in its current state; the active dot still shows it.
  toggleClear();await search();galaxyRead();
}
document.getElementById('map').addEventListener('click',event=>{
  const node=hitNode(event);
  if(node){
    const rootStyle=getComputedStyle(document.documentElement);
    const dark=parseInt((rootStyle.getPropertyValue('--bg').trim()||'#0b0e14').slice(5,7)||'14',16)<120;
    gPulses.push({x:G.CX+node.rFrac*G.RX*Math.cos(node.ang),y:G.CY+node.rFrac*G.RY*Math.sin(node.ang),
      r:node.radius,start:performance.now(),tone:gTone(node,dark)});
    applyMapNodeFilter(node).catch(()=>alert('Could not filter memories from the map.'));
  }
});function updateHover(key){
  if(key===hoverMapKey)return;
  hoverMapKey=key;
  if(key){hoverFocusTag=key;hoverFocusMix=1;hoverFadeStarted=0}
  else if(hoverFocusTag){
    hoverFadeStarted=performance.now();
    if(reducedMotion){hoverFocusTag=null;hoverFocusMix=0}
  }
  if(G){galaxyRead();if(reducedMotion)galaxyFrame(performance.now())}
}
document.getElementById('map').addEventListener('mousemove',event=>{
  const node=hitNode(event);
  event.target.style.cursor=node?'pointer':'default';
  updateHover(node?node.key:null);
});
document.getElementById('map').addEventListener('mouseleave',()=>updateHover(null));
// Fullscreen: real API where allowed, CSS-maximize fallback otherwise.
function setMaxed(v){gMaxed=v;document.getElementById('mapwrap').classList.toggle('maxed',v);
  document.documentElement.style.overflow=v?'hidden':'';drawMap()}
document.getElementById('fsBtn').addEventListener('click',()=>{
  const wrap=document.getElementById('mapwrap');
  if(document.fullscreenElement){document.exitFullscreen();return}
  if(gMaxed){setMaxed(false);return}
  let p;try{p=wrap.requestFullscreen&&wrap.requestFullscreen();}catch(e){}
  if(p&&p.then)p.then(()=>{},()=>setMaxed(true));
  else if(!document.fullscreenElement)setMaxed(true);
});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&gMaxed)setMaxed(false);});
document.addEventListener('fullscreenchange',()=>drawMap());
window.addEventListener('resize',()=>drawMap());
const PAGE=100; let offset=0;
// One click from a memory to everything sharing its tag. This goes through the
// server-side filter, not a client-side hide, so hierarchy expansion applies and
// the whole store is searched rather than the page already loaded. Tag filtering
// is where the measured retrieval gain actually is: the user supplies the tag.
function filterByTag(tag){
  const select=document.getElementById('filter-topic');
  const value=String(tag).toLowerCase();
  let option=[...select.options].find(o=>o.value===value);
  if(!option){option=new Option(value,value);select.add(option)}
  option.selected=!option.selected;  // clicking an active tag removes it again
  // Reveal the panel, so a filter set from a chip is visible and clearable
  // rather than applied behind a collapsed row.
  if(option.selected&&!panels.filters)togglePanel('filters');
  toggleClear();
  activeMapKey=null;clearMapEntityDetail();
  search();
}
const picked=id=>[...document.getElementById(id).selectedOptions]
  .map(o=>o.value).filter(Boolean);
function searchFilters(){
  return {
    since:document.getElementById('filter-date').value,
    until:document.getElementById('filter-date-to').value,
    topics:picked('filter-topic'),
    entities:picked('filter-entity')
  };
}
function anyFilter(f){return !!(f.since||f.until||f.topics.length||f.entities.length)}
async function loadSearchFilters(){
  const topicSelect=document.getElementById('filter-topic');
  const entitySelect=document.getElementById('filter-entity');
  // multi-select: keep every current choice across a reload, not just one
  const keepTopics=new Set(picked('filter-topic'));
  const keepEntities=new Set(picked('filter-entity'));
  const [topics,entities]=await Promise.all([
    api('/api/v1/categories'),api('/api/v1/entities?limit=10000')]);
  topicSelect.innerHTML=topics
    .sort((a,b)=>a.category.localeCompare(b.category))
    .map(topic=>`<option value="${esc(topic.category)}"${keepTopics.has(topic.category)?' selected':''}>${esc(topic.category)} (${topic.count})</option>`).join('');
  entitySelect.innerHTML=entities
    .sort((a,b)=>a.name.localeCompare(b.name))
    .map(entity=>`<option value="${esc(entity.id)}"${keepEntities.has(entity.id)?' selected':''}>${esc(entity.name)}${entity.entity_type?' · '+esc(entity.entity_type):''}</option>`).join('');
  toggleClear();
}
async function loadAll(more){
  if(!more){offset=0;current=[];searchActive=false}
  const previous=current.length;
  const items=await api('/api/v1/memories?limit='+PAGE+'&offset='+offset);
  offset+=items.length; haveMore=items.length===PAGE;
  render(current.concat(items),more?previous:undefined);
}
async function search(){
  const q=document.getElementById('q').value.trim(),f=searchFilters();
  if(!q&&!anyFilter(f))return loadAll();
  const body={query:q,limit:100};
  // an open-ended range is still a range: one bound is enough
  if(f.since)body.since=f.since;
  if(f.until)body.until=f.until;
  if(f.topics.length)body.categories=f.topics;
  if(f.entities.length)body.entity_id=f.entities;
  const rs=await api('/api/v1/search',{method:'POST',body:JSON.stringify(body)});
  haveMore=false;searchActive=true;
  render(rs.map(r=>({...r.memory,score:r.score})));
}
function toggleClear(){
  const f=searchFilters(),on=anyFilter(f);
  document.getElementById('qclear').style.display =
    document.getElementById('q').value||on ? 'block' : 'none';
  document.getElementById('filterdot').hidden=!on;
  document.getElementById('filterbtn').classList.toggle('active',on);
  const label=n=>n?`(${n})`:'';
  document.getElementById('topiccount').textContent=label(f.topics.length);
  document.getElementById('entitycount').textContent=label(f.entities.length);
}
function clearSearch(){
  document.getElementById('q').value='';
  for(const id of['filter-date','filter-date-to'])document.getElementById(id).value='';
  for(const id of['filter-topic','filter-entity'])
    [...document.getElementById(id).options].forEach(o=>o.selected=false);
  activeMapKey=null;clearMapEntityDetail();toggleClear();loadAll();
}

// -- unified knowledge area -------------------------------------------------
let knowledgeTab='topics',knowledgeNames={},allTags=[];
let knowledgeMapSuspended=false,knowledgeMapWasOpen=false;
function suspendMapForKnowledge(){
  knowledgeMapWasOpen=panels.map;knowledgeMapSuspended=knowledgeMapWasOpen;
  if(!knowledgeMapSuspended)return;
  if(gRAF){cancelAnimationFrame(gRAF);gRAF=0}
  G=null;mapData=null;gPulses=[];hoverMapKey=null;
  document.getElementById('mapwrap').hidden=true;
  document.getElementById('mapentitydetail').hidden=true;
  const canvas=document.getElementById('map');canvas.width=1;canvas.height=1;
}
function resumeMapAfterKnowledge(){
  if(!knowledgeMapSuspended)return;
  const restore=knowledgeMapWasOpen&&panels.map;
  knowledgeMapSuspended=false;knowledgeMapWasOpen=false;
  if(restore)loadMapData();else drawMap();
}
function setKnowledgeOpen(open){
  const modal=document.getElementById('knowmodal'),wasOpen=modal.classList.contains('on');
  if(open&&!wasOpen)suspendMapForKnowledge();
  modal.classList.toggle('on',open);
  document.documentElement.classList.toggle('knowledge-open',open);
  document.body.classList.toggle('knowledge-open',open);
  if(!open&&wasOpen)resumeMapAfterKnowledge();
}
async function openKnowledge(tab='topics'){
  setKnowledgeOpen(true);
  showKnowledge(tab);
  await Promise.all([loadTags(),loadEntities()]);
}
function closeKnowledge(){setKnowledgeOpen(false)}
function openAbout(){
  document.getElementById('aboutmodal').classList.add('on');
  showAbout('how');
}
function closeAbout(){document.getElementById('aboutmodal').classList.remove('on')}
function showAbout(tab){
  for(const name of['how','words','server']){
    document.getElementById('apanel-'+name).hidden=name!==tab;
    document.getElementById('atab-'+name).setAttribute('aria-pressed',name===tab);
  }
  if(tab==='server')renderServerInfo();
}
function renderServerInfo(){
  const s=serverInfo||{};
  const rows=[
    ['Your memories',`${s.active_memories??0} active`
      +(s.forgotten_memories?`, ${s.forgotten_memories} forgotten`:'')],
    ['Raw messages stored',s.episodes],
    ['Language model',s.llm,'Reads your messages to split them into facts and decide what is new. Without one, messages are stored whole.'],
    ['Embeddings',s.embedder,'Turns text into numbers so search can match on meaning.'],
    ['Storage',s.backend,'Everything lives in one file on this server.'],
  ];
  document.getElementById('serverinfo').innerHTML=rows
    .filter(r=>r[1]!==undefined&&r[1]!==null&&r[1]!=='')
    .map(([k,v,note])=>`<div class="tagrow"><span class="name"><b>${esc(k)}</b>
      <div class="hint">${esc(String(v))}${note?' — '+esc(note):''}</div></span></div>`)
    .join('')||'<div class="empty">No server details available.</div>';
}
function openTags(){return openKnowledge('topics')}
function openEntities(){return openKnowledge('entities')}
function showKnowledge(tab){
  knowledgeTab=tab;
  for(const name of['topics','entities','forgotten','maintenance']){
    document.getElementById('kpanel-'+name).hidden=name!==tab;
    document.getElementById('ktab-'+name).setAttribute('aria-pressed',name===tab);
  }
  if(tab==='maintenance')loadUpkeep();
  if(tab==='forgotten')loadForgotten();
}

// -- forgotten: deleted, but still recoverable until purged -----------------
async function loadForgotten(){
  const rows=await api('/api/v1/memories/forgotten');
  const el=document.getElementById('forgottenlist');
  if(!rows.length){el.innerHTML='<div class="empty">Nothing forgotten.</div>';return}
  el.innerHTML=rows.map(row=>`<div class="tagrow"><span class="name">
    ${esc(row.memory.content)}
    <div class="hint">forgotten ${esc((row.forgotten_at||'').slice(0,10))}
      by ${esc(row.actor||'system')}${row.reason?' · '+esc(row.reason):''}</div></span>
    <button class="act" title="bring this memory back into search"
      onclick='unforgetMemory(${JSON.stringify(row.memory.id)})'>restore</button>
    <button class="act del" title="delete permanently - this cannot be undone"
      onclick='purgeMemory(${JSON.stringify(row.memory.id)})'>delete for good</button></div>`).join('');
}
async function unforgetMemory(id){
  const result=await api('/api/v1/memories/'+encodeURIComponent(id)+'/unforget',
    {method:'POST',body:'{}'});
  if(result.error){alert(result.error);return}
  await Promise.all([loadForgotten(),loadStats(),loadMapData()]);
  loadAll();
}
async function purgeMemory(id){
  if(!confirm('Permanently delete this memory? This cannot be undone.'))return;
  const result=await api('/api/v1/memories/'+encodeURIComponent(id)+'/purge',
    {method:'POST',body:'{}'});
  if(result.error){alert(result.error);return}
  await Promise.all([loadForgotten(),loadStats()]);
}

// -- upkeep: what runs on its own, and consolidation under review -----------
async function loadUpkeep(){
  const info=await api('/api/v1/maintenance');
  const el=document.getElementById('upkeeplist');
  el.innerHTML=info.passes.map(p=>{
    const blocked=p.needs_llm&&!info.llm_available;
    const state=blocked?'<span class="cnt">needs an LLM</span>'
      :p.automatic?'<span class="syn">on</span>':'<span class="cnt">off</span>';
    const every=p.automatic&&p.interval_days?` Runs every ${p.interval_days} days.`:'';
    const last=p.last_run?` Last run ${esc(String(p.last_run).slice(0,16).replace('T',' '))}.`:'';
    const toggle=p.toggleable&&!blocked
      ? `<button class="act" onclick='togglePass(${JSON.stringify(p.key)},${!p.automatic})'
           title="${p.automatic?'stop running this automatically':'run this automatically from now on'}">turn ${p.automatic?'off':'on'}</button>`
      : '';
    const run=p.run_url&&!blocked
      ? `<button class="act" onclick='runPass(${JSON.stringify(p.run_url)},this)'
           title="run this pass right now">run now</button>`
      : '';
    return `<div class="tagrow"><span class="name"><b>${esc(p.label)}</b> ${state}
      <div class="hint">${esc(p.detail)}${every}${last}</div></span>${run}${toggle}</div>`;
  }).join('');
  renderTagHealth(info.tag_health||{});
  renderEntityJunk(info.entity_junk||{});
}
// Fragmentation caps recall silently: filtering to a tag that has split its
// subject drops the memories the question needed. Show it rather than wait for
// someone to go looking.
function renderTagHealth(h){
  const el=document.getElementById('taghealth');
  if(!el)return;
  const splits=(h.splits||[]);
  const rows=splits.map(s=>{
    const [a,b]=s.variants;
    return `<div class="tagrow"><span class="name">
      <b>${esc(a)}</b> and <b>${esc(b)}</b> <span class="cnt">${s.similarity}</span>
      <div class="hint">look like one subject split in two</div></span>
      <button onclick='healSplit(${JSON.stringify(s.variants)},${JSON.stringify(s.canonical)})'
        title="combine these two tags">Combine into "${esc(s.canonical)}"</button></div>`;
  }).join('');
  el.innerHTML=`<div class="hint">
      ${h.tags} tags over ${h.memories} memories · ${h.untagged} untagged ·
      ${h.single_use_tags} used once (${Math.round((h.single_use_share||0)*100)}%) ·
      ${h.suspected_splits} suspected split${h.suspected_splits===1?'':'s'}
    </div>`+(rows||'<div class="empty">No split tags detected.</div>');
}
async function healSplit(variants,canonical){
  const drop=variants.filter(v=>v!==canonical);
  if(!confirm(`Combine ${drop.join(', ')} into "${canonical}"?`))return;
  await api('/api/v1/tags/edit',{method:'POST',
    body:JSON.stringify({op:'merge',tags:drop,to:canonical})});
  await Promise.all([loadTags(),loadUpkeep()]);
}
async function togglePass(key,enabled){
  await api('/api/v1/maintenance/toggle',{method:'POST',
    body:JSON.stringify({key,enabled})});
  await loadUpkeep();
}
async function runPass(url,button){
  const label=button.textContent;
  button.disabled=true;button.textContent='running...';
  try{ await api(url,{method:'POST',body:'{}'}); }
  finally{ button.disabled=false;button.textContent=label; }
  await Promise.all([loadUpkeep(),loadTags(),loadEntities(),loadStats(),loadMapData()]);
}
// Obvious non-entities (dates, amounts, URLs) are cleaned automatically; the
// judgement cases (style instructions vs. real niche terms) need a reader, so
// the AI proposes and the user confirms with checkboxes.
function renderEntityJunk(junk){
  const el=document.getElementById('entityjunk');
  if(!el)return;
  const mech=junk.mechanical||[];
  const parts=[];
  if(mech.length){
    parts.push(`<div class="hint">${mech.length} obvious non-entit${mech.length===1?'y':'ies'} found - these are removed by the next self-healing run, or now:</div>`
      +mech.map(j=>`<div class="tagrow"><span class="name"><b>${esc(j.name)}</b>
        <div class="hint">${esc(j.reason)}</div></span></div>`).join('')
      +`<div class="bar"><button onclick="removeMechanicalJunk()">Remove ${mech.length} now</button></div>`);
  }else{
    parts.push('<div class="empty">No obvious non-entities.</div>');
  }
  parts.push(`<div class="bar"><button onclick="reviewEntities(this)"
    title="one AI call proposes which concept-type names are not really entities; nothing is removed until you confirm">
    Review ${junk.reviewable||0} concept names with AI</button></div>
    <div id="entityreview"></div>`);
  el.innerHTML=parts.join('');
  window._mechJunk=mech.map(j=>j.id);
}
async function removeMechanicalJunk(){
  const ids=window._mechJunk||[];
  if(!ids.length)return;
  await api('/api/v1/entities/remove',{method:'POST',body:JSON.stringify({ids})});
  await Promise.all([loadUpkeep(),loadEntities(),loadMapData()]);
}
async function reviewEntities(button){
  button.disabled=true;button.textContent='Reviewing...';
  const res=await api('/api/v1/maintenance/entity-review',{method:'POST',body:'{}'});
  const judged=res.judged||[];
  const el=document.getElementById('entityreview');
  if(!judged.length){
    el.innerHTML='<div class="empty">The review found nothing to remove.</div>';
    button.textContent='Review again';button.disabled=false;return;
  }
  el.innerHTML=`<div class="hint">${judged.length} name${judged.length===1?'':'s'} judged not to be entities. Untick any you want to keep.</div>`
    +judged.map(j=>`<div class="tagrow">
      <input type="checkbox" class="junkpick" checked value="${esc(j.id)}">
      <span class="name"><b>${esc(j.name)}</b></span></div>`).join('')
    +`<div class="bar"><button onclick="removeReviewedJunk()">Remove selected</button></div>`;
  button.textContent='Review again';button.disabled=false;
}
async function removeReviewedJunk(){
  const ids=[...document.querySelectorAll('.junkpick:checked')].map(c=>c.value);
  if(!ids.length)return;
  if(!confirm(`Remove ${ids.length} entit${ids.length===1?'y':'ies'}? Their memories are untouched.`))return;
  await api('/api/v1/entities/remove',{method:'POST',body:JSON.stringify({ids})});
  await Promise.all([loadUpkeep(),loadEntities(),loadMapData()]);
}
let consolidationPreview=null;
async function previewConsolidation(){
  const el=document.getElementById('conresult');
  el.innerHTML='<div class="empty">Looking for duplicates...</div>';
  const threshold=parseFloat(document.getElementById('conthresh').value);
  const res=await api('/api/v1/maintenance/consolidate',
    {method:'POST',body:JSON.stringify({threshold,apply:false})});
  consolidationPreview=threshold;
  const merges=(res.groups||[]).filter(g=>g.same_fact);
  document.getElementById('conapply').disabled=!merges.length;
  if(!merges.length){
    el.innerHTML=`<div class="empty">Scanned ${res.scanned} memories. Nothing to merge.</div>`;
    return;
  }
  // each proposal is ticked individually: accepting one is not accepting all
  el.innerHTML=`<div class="hint">${merges.length} group${merges.length===1?'':'s'} found. Tick the ones to merge.</div>`
   +merges.map(g=>`<div class="tagrow">
    <input type="checkbox" class="congroup" checked
           value="${esc(JSON.stringify(g.memory_ids))}" onchange="updateConsolidationCount()">
    <span class="name">
      <b>${esc(g.merged_content)}</b>
      <div class="hint">replaces ${g.memory_ids.length}: ${g.contents.map(c=>esc(c)).join(' · ')}</div>
      <div class="hint">${esc(g.reason)}</div></span></div>`).join('');
  updateConsolidationCount();
}
function chosenGroups(){
  return [...document.querySelectorAll('.congroup:checked')].map(c=>JSON.parse(c.value));
}
function updateConsolidationCount(){
  const n=chosenGroups().length,button=document.getElementById('conapply');
  button.disabled=!n;
  button.textContent=n?`Merge ${n} selected`:'Merge selected';
}
async function applyConsolidation(){
  const only=chosenGroups();
  if(!only.length)return;
  const count=only.reduce((n,g)=>n+g.length,0);
  if(!confirm(`Merge ${only.length} group(s)? ${count} memories become ${only.length} new one(s); the originals are forgotten and stay listed under Forgotten.`))return;
  const res=await api('/api/v1/maintenance/consolidate',
    {method:'POST',body:JSON.stringify({threshold:consolidationPreview,apply:true,only})});
  document.getElementById('conresult').innerHTML=
    `<div class="empty">Merged ${res.merged} group(s); ${res.superseded} memories forgotten.</div>`;
  document.getElementById('conapply').disabled=true;
  load();
}
function tagSel(){return[...document.querySelectorAll('.tagrow input:checked')].map(c=>c.value)}
function updateSel(){
  const n=tagSel().length;
  document.getElementById('tagsel').textContent=n?`${n} selected`:'none selected';
}
async function loadTags(){
  allTags=(await api('/api/v1/categories'))
    .sort((a,b)=>a.category.localeCompare(b.category));
  renderTags();
}
// Filtering redraws from the cached list: a store with hundreds of tags is
// unusable as one long scroll, and refetching on every keystroke is wasteful.
function renderTags(){
  const el=document.getElementById('taglist');
  const needle=(document.getElementById('tagsearch')?.value||'').trim().toLowerCase();
  if(!allTags.length){el.innerHTML='<div class="empty">No tags yet.</div>';return}
  const shown=needle?allTags.filter(t=>t.category.toLowerCase().includes(needle)):allTags;
  if(!shown.length){
    el.innerHTML=`<div class="empty">No tag matches "${esc(needle)}".</div>`;return;
  }
  el.innerHTML=(needle?`<div class="hint">${shown.length} of ${allTags.length} tags</div>`:'')
   +shown.map(topic=>`<div class="tagrow">
    <input type="checkbox" value="${esc(topic.category)}" onchange="updateSel()">
    <span class="name"><b>${esc(topic.category)}</b> <span class="cnt">${topic.count}</span>
      ${topic.synthetic?'<span class="syn">synthetic parent</span>':''}</span>
    <button class="act" title="rename this tag everywhere" onclick='renameTag(${JSON.stringify(topic.category)})'>rename</button>
    <button class="act del" title="delete this tag from all memories" onclick='deleteTag(${JSON.stringify(topic.category)})'>delete</button>
  </div>`).join('');
  updateSel();
}
async function tagOp(body){
  const result=await api('/api/v1/tags/edit',{method:'POST',body:JSON.stringify(body)});
  await Promise.all([loadTags(),loadSearchFilters(),loadMapData()]);activeMapKey=null;clearMapEntityDetail();loadAll();return result;
}
async function renameTag(tag){
  const to=prompt('Rename tag "'+tag+'" to:',tag);if(!to||to.trim()===tag)return;
  await tagOp({op:'rename',tag,to:to.trim()});
}
async function deleteTag(tag){
  if(!confirm('Delete tag "'+tag+'" from all memories? The memories stay.'))return;
  await tagOp({op:'delete',tag});
}
async function mergeTags(){
  const selected=tagSel();if(selected.length<2)return alert('Check at least two tags to combine.');
  const to=prompt('Combine '+selected.length+' tags into one named:',selected[0]);
  if(!to||!to.trim())return;
  await tagOp({op:'merge',tags:selected,to:to.trim()});
}
async function suggestMerges(){
  const box=document.getElementById('tagsuggest');box.innerHTML='<div class="hint">thinking...</div>';
  const groups=await api('/api/v1/tags/suggest-merges');
  await loadTags();
  if(!groups.length){box.innerHTML='<div class="hint">Obvious plural/format duplicates were merged automatically. No other variants found.</div>';return}
  box.innerHTML=groups.map((group,index)=>`<div class="tagrow" id="sg${index}">
    <span class="name">merge <b>${group.variants.map(esc).join('</b>, <b>')}</b> into <b>${esc(group.canonical)}</b></span>
    <button class="act" onclick='applyMerge(${JSON.stringify(group)},${index})'>apply</button>
    <button class="act del" onclick="document.getElementById('sg${index}').remove()">dismiss</button>
  </div>`).join('');
}
async function applyMerge(group,index){
  await tagOp({op:'merge',tags:group.variants,to:group.canonical});
  const row=document.getElementById('sg'+index);if(row)row.remove();
}
async function loadEntities(){
  const [entities,relations,proposals]=await Promise.all([
    api('/api/v1/entities?limit=100000&include_merged=true'),
    api('/api/v1/relations?limit=2000'),
    api('/api/v1/entities/proposals?limit=1000')]);
  knowledgeNames={};entities.forEach(entity=>knowledgeNames[entity.id]=entity.name);
  const active=entities.filter(entity=>!entity.merged_into);
  document.getElementById('entcount').textContent=active.length+' entities, '+relations.length+' relations';
  const byType={};
  active.forEach(entity=>(byType[entity.entity_type||'untyped']??=[]).push(entity));
  entityGroups=byType;
  renderEntityGroups();
  document.getElementById('proplist').innerHTML=proposals.length?proposals.map(proposal=>`<div class="tagrow"><span class="name">
    <b>${esc(knowledgeNames[proposal.entity_a]||proposal.entity_a)}</b> and <b>${esc(knowledgeNames[proposal.entity_b]||proposal.entity_b)}</b>
    <span class="cnt">${esc(proposal.reason||'identity is uncertain')}</span></span>
    <button class="act" onclick='decideProposal(${JSON.stringify(proposal.id)},"confirm",this)'>merge</button>
    <button class="act del" onclick='decideProposal(${JSON.stringify(proposal.id)},"reject",this)'>keep separate</button></div>`).join(''):'<div class="empty">No open merge proposals.</div>';
}
function entityIdentityBlock(entity,aliases){
  return `<div class="description">${esc(entity.description||'No active evidence to summarize yet.')}</div>
    <div class="alias-list">${aliases.map(alias=>`<span>${esc(alias)}</span>`).join('')||'<span>No aliases yet.</span>'}</div>`;
}
async function openEntity(id){
  setKnowledgeOpen(true);showKnowledge('entities');
  const box=document.getElementById('entitydetail');box.dataset.entityId=id;box.innerHTML='<div class="hint">loading entity...</div>';
  const detail=await api('/api/v1/entities/'+encodeURIComponent(id));
  const entity=detail.entity,aliases=detail.aliases||[];
  box.innerHTML=`<div class="detail"><h3><button class="x" style="float:right;border:none;background:none;color:var(--dim);cursor:pointer" title="close" onclick="closeEntity()">x</button><span id="knowledgeentityname">${esc(entity.name)}</span> ${entity.entity_type?`<span class="syn">${esc(entity.entity_type)}</span>`:''} <button class="act" onclick='renameEntity(${JSON.stringify(id)})' title="Change this entity's canonical name; the old name remains an alias.">rename</button></h3>
    <div id="knowledgeentityidentity">${entityIdentityBlock(entity,aliases)}</div>
    <div class="bar"><input id="aliasinput" placeholder="add an alias"><button onclick='addAlias(${JSON.stringify(id)})' title="Add another name for this entity.">Add alias</button></div>
    ${relationsBlock(id,detail)}
    <div class="hint">${detail.memories.length} active supporting memor${detail.memories.length===1?'y':'ies'}</div>
    ${detail.memories.map(memory=>`<div class="tagrow"><span class="name">${esc(memory.content)}</span><button class="act" onclick='showMemory(${JSON.stringify(memory.id)})'>open</button></div>`).join('')||'<div class="empty">No active supporting memories.</div>'}</div>`;
}
// Relations read as "this entity -> predicate -> that one", so they belong next
// to the entity they describe. A flat list of every edge in the store had no
// subject to be about. These are also the edges relational search traverses, so
// this doubles as the explanation for why a search reached a given memory.
function relationsBlock(id,detail){
  const rels=detail.relations||[],names=detail.relation_names||{};
  if(!rels.length)return '<div class="hint">No relations recorded yet.</div>';
  const name=eid=>esc(names[eid]||'?');
  const rows=rels.map(r=>{
    const outgoing=r.subject===id;
    const other=outgoing?r.object:r.subject;
    return `<div class="tagrow"><span class="name">
      <span class="cnt">${outgoing?'':'&larr; '}${esc(r.predicate)}${outgoing?' &rarr;':''}</span>
      <button class="entity-link" onclick='openEntity(${JSON.stringify(other)})'>${name(other)}</button>
      </span>${r.memory_id?`<button class="act" onclick='showMemory(${JSON.stringify(r.memory_id)})'>evidence</button>`:''}</div>`;
  }).join('');
  return `<div class="hint">${rels.length} relation${rels.length===1?'':'s'}</div>${rows}`;
}
function closeEntity(){const box=document.getElementById('entitydetail');box.innerHTML='';delete box.dataset.entityId}
// A type like "concept" can hold hundreds of entities. Listing them all turns
// the tab into one long scroll, so each type is capped until asked to expand.
const ENTITY_ROW_CAP=12;
let entityGroups={},entityExpanded=new Set();
function toggleEntityType(type){
  entityExpanded.has(type)?entityExpanded.delete(type):entityExpanded.add(type);
  renderEntityGroups();
}
function renderEntityGroups(){
  const el=document.getElementById('entlist');
  const types=Object.keys(entityGroups).sort();
  if(!types.length){el.innerHTML='<div class="empty">No entities yet.</div>';return}
  el.innerHTML=types.map(type=>{
    const all=entityGroups[type].slice().sort((a,b)=>a.name.localeCompare(b.name));
    const open=entityExpanded.has(type);
    const shown=open?all:all.slice(0,ENTITY_ROW_CAP);
    const hidden=all.length-shown.length;
    const links=shown.map(e=>`<button class="entity-link" onclick='openEntity(${JSON.stringify(e.id)})'>${esc(e.name)}</button>`).join(', ');
    let more='';
    if(hidden>0)more=` <button class="act" onclick='toggleEntityType(${JSON.stringify(type)})'>show ${hidden} more</button>`;
    else if(open&&all.length>ENTITY_ROW_CAP)more=` <button class="act" onclick='toggleEntityType(${JSON.stringify(type)})'>show less</button>`;
    return `<div class="tagrow">
      <span class="syn entity-type">${esc(type)}</span>
      <span class="name">${links}${more}</span>
      <span class="cnt">${all.length}</span></div>`;
  }).join('');
}
async function addAlias(id){
  const input=document.getElementById('aliasinput'),alias=input.value.trim();if(!alias)return;
  const result=await api('/api/v1/entities/'+encodeURIComponent(id)+'/aliases',{method:'POST',body:JSON.stringify({alias})});
  input.value='';syncEntityIdentity(id,result);
  await loadEntities();
}
async function decideProposal(id,decision,button){
  if(button)button.disabled=true;
  try{
    const result=await api('/api/v1/entities/proposals/'+encodeURIComponent(id)+'/'+decision,{method:'POST',body:'{}'});
    const ok=decision==='confirm'?result.confirmed:result.rejected;
    if(!ok)alert('That proposal changed while this view was open. The list has been refreshed.');
  }catch(error){alert('Could not update that merge proposal.');}
  await Promise.all([loadEntities(),loadMapData()]);
}
async function showMemory(id){
  const memory=await api('/api/v1/memories/'+encodeURIComponent(id));
  closeKnowledge();activeMapKey=null;clearMapEntityDetail();haveMore=false;render([memory]);
}
async function backfillTypes(){
  document.getElementById('entcount').textContent='classifying...';
  await api('/api/v1/entities/backfill-types',{method:'POST',body:'{}'});
  await Promise.all([loadEntities(),loadMapData()]);
}
async function add(infer){
  const t=document.getElementById('newmem').value.trim(); if(!t)return;
  const cs=document.getElementById('newcats').value.split(',').map(s=>s.trim()).filter(Boolean);
  // Clear the form first: the text is durably stored before the response comes
  // back, so making the user watch extraction finish buys them nothing.
  document.getElementById('newmem').value=''; document.getElementById('newcats').value='';
  const res=await api('/api/v1/memories',{method:'POST',
    body:JSON.stringify({content:t,infer,defer:infer,categories:cs.length?cs:undefined})});
  if(res.warnings&&res.warnings.length)alert(res.warnings.join('\\n'));
  loadAll(); loadStats(); loadMapData();
}
async function distill(id){
  const r=await fetch('/api/v1/memories/'+id+'/distill',{method:'POST',headers:H});
  const data=await r.json().catch(()=>null);
  if(!r.ok){alert((data&&data.error)||('Distillation failed ('+r.status+').'));return}
  if(data.warnings&&data.warnings.length)alert(data.warnings.join('\\n'));
  loadAll(); loadStats(); loadMapData();
}
async function del(id){await api('/api/v1/memories/'+id,{method:'DELETE'}); loadAll(); loadStats(); loadMapData();}
function startEdit(id){editingId=id;render(current)}
function cancelEdit(){editingId=null;render(current)}
async function saveEdit(id){
  const content=document.getElementById('edit-note').value.trim(); if(!content)return;
  const cats=document.getElementById('edit-cats').value.split(',').map(s=>s.trim()).filter(Boolean);
  // Editing text re-runs entity analysis, which is several provider calls. Show
  // the saved row at once and let that finish behind it; if it fails, say so and
  // put the editor back rather than pretending the edit landed.
  const before=current.find(m=>m.id===id);
  editingId=null;
  current=current.map(m=>m.id===id?{...m,content,categories:cats,saving:true}:m);
  render(current);
  try{
    const updated=await api('/api/v1/memories/'+id,{method:'PATCH',
      body:JSON.stringify({content,categories:cats})});
    if(updated.error)throw new Error(updated.error);
    current=current.map(m=>m.id===id?{...m,...updated,saving:false}:m);
  }catch(error){
    alert('Could not save that edit: '+error.message);
    current=current.map(m=>m.id===id?before:m);
    editingId=id;
  }
  render(current);
  await loadMapData();
}
async function exportMemories(){
  const backup=await api('/api/v1/export');
  if(backup.error){alert(backup.error);return}
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(backup,null,2)+'\\n'],{type:'application/json'}));
  a.download='memry-backup-'+new Date().toISOString().slice(0,10)+'.json';
  a.click(); URL.revokeObjectURL(a.href);
}
async function importMemories(file){
  if(!file)return;
  const text=((await file.text())||'').trim(); if(!text)return;
  let payload;
  try{payload=JSON.parse(text)}
  catch{
    try{payload=text.split('\\n').map(line=>line.trim()).filter(Boolean).map(line=>JSON.parse(line))}
    catch{alert('Not a valid Memry JSON or JSONL file.');return}
  }
  const isBackup=payload&&payload.format==='memry-backup';
  const rows=Array.isArray(payload)?payload:[payload];
  const body=isBackup?payload:{memories:rows};
  const btn=document.getElementById('importbtn');
  btn.textContent='importing…';
  try{
    const res=await api('/api/v1/import',{method:'POST',body:JSON.stringify(body)});
    if(res&&res.error)alert(res.error);
    else if(isBackup)alert('Backup restored: '+res.inserted+' records added, '+res.unchanged+' already identical.');
    else if(res&&res.imported!==undefined){
      const notes=[];
      if(res.deduplicated)notes.push(res.deduplicated+' duplicates skipped');
      if(res.skipped)notes.push(res.skipped+' empty rows skipped');
      alert('Imported '+res.imported+' of '+rows.length+(notes.length?' ('+notes.join(', ')+')':'')+'.');
    }
    else alert('Import failed.');
  }catch{alert('Import failed.')}
  btn.textContent='import';
  loadAll();loadStats();loadSearchFilters();loadMapData();
}
let serverInfo={};
async function loadStats(){
  const s=await api('/api/v1/stats');
  serverInfo=s;
  // Only what a person reading their own memory list cares about. The model
  // and backend names live under About, where there is room to say what they
  // mean; here they were just noise, and half of them read "undefined".
  const forgotten=s.forgotten_memories??s.invalidated_memories??0;
  const bits=[`${s.active_memories??'?'} memories`];
  if(forgotten)bits.push(`${forgotten} forgotten`);
  document.getElementById('stats').textContent=bits.join(' · ');
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search()});
syncPanels(); loadStats(); loadSearchFilters(); loadMapData(); loadAll();
</script></body></html>"""


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to Memry</title>
<style>
:root{{--bg:#0b0e14;--panel:#141a24;--line:#232c3b;--text:#dbe4f0;--dim:#8494ab;
--accent:#5eead4;--warn:#f0a35e;font-size:15px}}
@media (prefers-color-scheme: light){{:root{{--bg:#f5f7fa;--panel:#fff;--line:#dde4ee;
--text:#1a2333;--dim:#5c6b82;--accent:#0d9488;--warn:#b45309}}}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);
color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:1.8rem;width:min(94vw,26rem)}}
h1{{font-size:1.15rem;margin:.2rem 0 .3rem}}h1 span{{color:var(--accent)}}
p.sub{{color:var(--dim);font-size:.85rem;margin:0 0 1.2rem;line-height:1.45}}
label{{display:block;font-size:.8rem;color:var(--dim);margin:.7rem 0 .25rem}}
input{{width:100%;font:inherit;color:inherit;background:var(--bg);
border:1px solid var(--line);border-radius:8px;padding:.55rem .7rem}}
input:focus{{outline:2px solid var(--accent);outline-offset:-1px}}
.row{{display:flex;gap:.6rem;margin-top:1.3rem}}
button{{flex:1;font:inherit;cursor:pointer;border-radius:8px;padding:.6rem;
border:1px solid var(--line);background:var(--panel);color:inherit}}
button.primary{{background:var(--accent);color:#04211c;border-color:transparent;
font-weight:600}}
.err{{background:color-mix(in srgb,var(--warn) 16%,transparent);border:1px solid var(--warn);
color:var(--warn);border-radius:8px;padding:.5rem .7rem;font-size:.85rem;margin-bottom:1rem}}
.scope{{font-size:.78rem;color:var(--dim);margin-top:1.1rem;border-top:1px solid var(--line);
padding-top:.8rem;line-height:1.5}}
</style></head><body>
<form class="card" method="post" action="/oauth/login">
<h1><span>Mem</span>ry</h1>
<p class="sub"><b>{client}</b> is asking to read and write your memories.</p>
{error}
<input type="hidden" name="request" value="{request_id}">
<label for="account">Account</label>
<input id="account" name="account" autocomplete="username" autofocus required>
<label for="password">Password</label>
<input id="password" name="password" type="password"
       autocomplete="current-password" required>
<div class="row">
  <button type="submit" name="decision" value="deny">Deny</button>
  <button class="primary" type="submit" name="decision" value="approve">Approve</button>
</div>
<p class="scope">Approving lets this client save, search and delete memories in your
namespace only. You can revoke it at any time with
<code>memry account revoke-keys</code> or by disabling the account.</p>
</form></body></html>"""


# Same visual language as the OAuth login, but this one opens a dashboard
# session cookie instead of issuing an authorization code.
_DASH_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to Memry</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50' fill='none' stroke='%2314b8a6' stroke-width='7' stroke-linecap='round'/%3E%3Ccircle cx='47' cy='10.5' r='4.5' fill='%2314b8a6'/%3E%3Ccircle cx='56' cy='20' r='3.2' fill='%2314b8a6' opacity='.85'/%3E%3Ccircle cx='57.5' cy='30' r='2.2' fill='%2314b8a6' opacity='.7'/%3E%3C/svg%3E">
<style>
:root{{--bg:#0b0e14;--panel:#141a24;--line:#232c3b;--text:#dbe4f0;--dim:#8494ab;
--accent:#5eead4;--warn:#f0a35e;font-size:15px}}
@media (prefers-color-scheme: light){{:root{{--bg:#f5f7fa;--panel:#fff;--line:#dde4ee;
--text:#1a2333;--dim:#5c6b82;--accent:#0d9488;--warn:#b45309}}}}
*{{box-sizing:border-box}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);
color:var(--text);font-family:ui-sans-serif,system-ui,Segoe UI,sans-serif}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:1.8rem;width:min(94vw,26rem)}}
h1{{font-size:1.15rem;margin:.2rem 0 1rem}}h1 span{{color:var(--accent)}}
h1 svg{{vertical-align:-4px;margin-right:.3rem}}
label{{display:block;font-size:.8rem;color:var(--dim);margin:.7rem 0 .25rem}}
input{{width:100%;font:inherit;color:inherit;background:var(--bg);
border:1px solid var(--line);border-radius:8px;padding:.55rem .7rem}}
input:focus{{outline:2px solid var(--accent);outline-offset:-1px}}
button{{width:100%;margin-top:1.2rem;font:inherit;cursor:pointer;border-radius:8px;
padding:.6rem;border:1px solid transparent;background:var(--accent);color:#04211c;
font-weight:600}}
.err{{background:color-mix(in srgb,var(--warn) 16%,transparent);border:1px solid var(--warn);
color:var(--warn);border-radius:8px;padding:.5rem .7rem;font-size:.85rem;margin-bottom:1rem}}
</style></head><body>
<form class="card" method="post" action="/login">
<h1><svg viewBox="0 0 64 64" width="20" height="20"><path d="M12,50 L12,30 Q12,20 21,20 Q30,20 30,30 L30,50 M30,30 Q30,20 39,20 Q48,20 48,30 L48,50" fill="none" stroke="#5eead4" stroke-width="7" stroke-linecap="round"/><circle cx="47" cy="10.5" r="4.5" fill="#5eead4"/><circle cx="56" cy="20" r="3.2" fill="#5eead4" opacity=".85"/><circle cx="57.5" cy="30" r="2.2" fill="#5eead4" opacity=".7"/></svg><span>Mem</span>ry dashboard</h1>
{error}
<label for="account">Account</label>
<input id="account" name="account" autocomplete="username" autofocus>
<label for="password">Password</label>
<input id="password" name="password" type="password" autocomplete="current-password">
<button type="submit">Sign in</button>
</form></body></html>"""


def _tag_run_due(last_run: str | None, interval_days: float, now: datetime) -> bool:
    """Has ``interval_days`` elapsed since the last tag-abstraction run?

    A namespace never run before (``None``) is due. An unparseable stamp is
    treated as due rather than wedging the scheduler forever.
    """
    if last_run is None:
        return True
    try:
        last = datetime.fromisoformat(last_run)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return True
    return (now - last) >= timedelta(days=max(interval_days, 0.0))


MCP_ORIGIN_KEY = "memry.mcp_origin"


def _looks_like_mcp(scope: dict) -> bool:
    """Is this Streamable HTTP, or a browser asking for the dashboard?

    The dashboard root is GET-only, so POST (a JSON-RPC message) and DELETE
    (session teardown) at the root can only be MCP. A GET is MCP only when it
    opens the server-to-client event stream, which browsers never ask for.
    """
    method = scope.get("method", "")
    if method in ("POST", "DELETE"):
        return True
    if method != "GET":
        return False
    accept = ""
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == "accept":
            accept = value.decode("latin-1").lower()
            break
    return "text/event-stream" in accept and "text/html" not in accept


class _NormalizeMcpPath:
    """``/mcp``, ``/mcp/`` and the domain root are one endpoint.

    Left alone, Starlette's router answers ``/mcp`` with a 307 to ``/mcp/``.
    Clients that drop the Authorization header across a redirect (VS Code and
    other SDK MCP clients do, especially when a proxy makes it cross-scheme)
    then arrive unauthenticated and see a 401. Rewrite instead of redirect.

    The root rewrite covers a second field failure. Connector UIs ask for a
    server URL and people paste the site they already have open, without the
    ``/mcp`` suffix. OAuth then completes - the metadata documents live at the
    root - and the handshake that follows lands on the dashboard route, which
    answers POST with 405. The client reports only that connecting failed
    (ChatGPT turns it into a 424 on its own callback), with nothing in the
    error naming the missing path. Serving the handshake from the root makes
    the URL people actually paste work.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path")
            if path == "/mcp":
                scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
            elif path in ("", "/") and _looks_like_mcp(scope):
                scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
                # remembered so an unauthenticated reply can point at the
                # metadata document for the URL the client actually configured
                scope[MCP_ORIGIN_KEY] = "/"
        await self.app(scope, receive, send)


def create_app(
    store: MemoryStore | None = None, *, accounts: AccountStore | None = None
) -> Starlette:
    store = store or MemoryStore()
    api_key = store.config.api_key
    tenants = {t.api_key: t.name for t in store.config.tenants}
    default_user = store.config.default_user_id
    accounts = accounts or AccountStore(
        store.config.auth_db_path or default_auth_db_path(store.config.db_path)
    )
    # OAuth needs a public issuer URL to put in its metadata; without one there
    # is nothing coherent to advertise, so the endpoints simply do not exist.
    public_url = (store.config.public_url or "").rstrip("/")
    oauth = MemryOAuthProvider(accounts, public_url=public_url) if public_url else None
    enrichment_worker = EnrichmentWorker(store)
    mcp = create_server(
        store,
        enrichment_worker=enrichment_worker,
        manage_enrichment_worker=False,
    )
    mcp.settings.streamable_http_path = "/"
    mcp_app = mcp.streamable_http_app()

    def _unauthorized() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    def _account_principal(account) -> Principal:
        return Principal(
            name=account.name,
            default_user=default_user,
            admin=account.is_admin,
            fixed_user=(
                default_user
                if account.is_admin
                else f"{account.name}::{default_user}"
            ),
        )

    def resolve_principal(token: str) -> Principal | None:
        """Map a bearer token to who it acts as, or None to reject it.

        Shared by REST and MCP so both transports agree on identity; there is
        no second implementation to drift.
        """
        if not api_key and not tenants and accounts.is_empty():
            return ADMIN  # open mode: bind privately or set keys
        if api_key and token == api_key:
            return ADMIN
        if token in tenants:
            return Principal(name=tenants[token], default_user=default_user)
        account = accounts.account_for_key(token)
        if account is not None:
            return _account_principal(account)
        if oauth is not None:
            granted = oauth.verify_access_token(token)
            if granted is not None and granted.subject:
                account = accounts.get_by_name(granted.subject)
                if account is not None and not account.disabled:
                    return _account_principal(account)
        return None

    def _is_configured_key(value: str) -> bool:
        """Is this URL segment actually one of our keys?

        Deliberately stricter than resolve_principal, which returns admin for
        anything in open mode: without that distinction an open server would
        strip the first path segment of every /mcp request as if it were a key.
        """
        if not value:
            return False
        return (
            value == api_key
            or value in tenants
            or accounts.account_for_key(value) is not None
        )

    def _bearer(request: Request) -> str:
        header = request.headers.get("authorization", "")
        return header[7:].strip() if header.lower().startswith("bearer ") else ""

    def _session_principal(request: Request) -> Principal | None:
        """Principal from a dashboard session cookie, if any.

        This is the browser's path: humans log in once at /login and ride a
        cookie, instead of pasting an API key into every session. Programmatic
        clients keep using the bearer header and never touch this.
        """
        row = accounts.resolve_session(request.cookies.get(SESSION_COOKIE, ""))
        if row is None:
            return None
        _, name = row
        account = accounts.get_by_name(name) if name else None
        return _account_principal(account) if account is not None else None

    def _authenticate(request: Request) -> Principal | None:
        """Who this request acts as: bearer token first, then session cookie."""
        if not api_key and not tenants and accounts.is_empty():
            return ADMIN  # open mode
        token = _bearer(request)
        if token:
            principal = resolve_principal(token)
            if principal is not None:
                return principal
        return _session_principal(request)

    def _p(request: Request) -> Principal:
        return request.state.principal

    def guarded(handler):
        async def wrapper(request: Request) -> Response:
            principal = _authenticate(request)
            if principal is None:
                return _unauthorized()
            request.state.principal = principal
            return await handler(request)

        return wrapper

    # -- dashboard login / session ---------------------------------------
    def _set_session(response: Response, request: Request, account: str) -> None:
        token = accounts.create_session(account)
        response.set_cookie(
            SESSION_COOKIE, token,
            max_age=SESSION_TTL, httponly=True, samesite="lax",
            secure=request.url.scheme == "https",
        )

    def _login_error(message: str) -> HTMLResponse:
        return HTMLResponse(
            _DASH_LOGIN_PAGE.format(error=f'<div class="err">{html.escape(message)}</div>'),
            status_code=401,
        )

    async def login_form(request: Request) -> Response:
        if _authenticate(request) is not None:
            return RedirectResponse("/", status_code=302)
        return HTMLResponse(_DASH_LOGIN_PAGE.format(error=""))

    async def login_submit(request: Request) -> Response:
        form = await request.form()

        name = str(form.get("account", "")).strip()
        password = str(form.get("password", ""))
        account = accounts.get_by_name(name) if name else None
        if account is None or account.disabled or not account.check_password(password):
            # one message for any failure: no probing which accounts exist
            return _login_error("Wrong account or password.")
        resp = RedirectResponse("/", status_code=302)
        _set_session(resp, request, account.name)
        return resp

    async def logout(request: Request) -> Response:
        accounts.delete_session(request.cookies.get(SESSION_COOKIE, ""))
        resp = RedirectResponse("/login", status_code=302)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    # -- handlers ---------------------------------------------------------
    async def dashboard(request: Request) -> Response:
        principal = _authenticate(request)
        if principal is None:
            return RedirectResponse("/login", status_code=302)
        who = principal.name or "admin"
        return HTMLResponse(_DASHBOARD.replace("__WHOAMI__", html.escape(who)))

    async def health(request: Request) -> Response:
        from . import __version__

        # Version is public on purpose: it lets anyone (and the deploy script)
        # confirm which release a server actually runs.
        return JSONResponse(
            {"status": "ok", "service": "memry", "version": __version__}
        )

    def _memory_or_error(request: Request) -> tuple[Any, Response | None]:
        memory = store.get(
            request.path_params["memory_id"], owner_prefix=_p(request).prefix
        )
        if memory is None:
            return None, JSONResponse({"error": "not found"}, status_code=404)
        return memory, None

    def _parse_categories(value: Any) -> list[str] | None:
        if not value:
            return None
        if isinstance(value, str):
            return [c.strip() for c in value.split(",") if c.strip()]
        return [str(c) for c in value]

    def _resolve_entity_filter(
        request: Request, value: Any
    ) -> tuple[str | list[str] | None, Response | None]:
        """Resolve one entity filter or several, checking ownership on each.

        A list must not weaken the check: every id is verified, so a caller
        cannot smuggle another account's entity in beside one of their own.
        """
        if not value:
            return None, None
        many = isinstance(value, (list, tuple))
        raw = [str(v) for v in value if v] if many else [str(value)]
        if not raw:
            return None, None
        resolved: list[str] = []
        for candidate in raw:
            entity_id = store.backend.resolve_entity_id(candidate)
            entity = store.backend.get_entity(entity_id) if entity_id else None
            if entity is None or not _p(request).owns(entity.user_id):
                return None, JSONResponse(
                    {"error": "entity not found"}, status_code=404
                )
            resolved.append(entity.id)
        return (resolved if many else resolved[0]), None

    def _memory_payload(memory) -> dict[str, Any]:
        data = memory.model_dump()
        data["entity_links"] = [
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
            }
            for entity in store.backend.entities_of_memory(memory.id)
        ]
        return data

    async def list_memories(request: Request) -> Response:
        q = request.query_params
        entity_id, error = _resolve_entity_filter(request, q.get("entity_id"))
        if error:
            return error
        memories = store.get_all(
            user_id=_p(request).namespace(q.get("user_id")),
            agent_id=q.get("agent_id"),
            run_id=q.get("run_id"),
            include_invalid=q.get("include_invalid") == "true",
            limit=int(q.get("limit", "100")),
            offset=int(q.get("offset", "0")),
            categories=_parse_categories(q.get("categories")),
            entity_id=entity_id,
            since=q.get("since") or None,
            until=q.get("until") or None,
        )
        return JSONResponse([_memory_payload(memory) for memory in memories])

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
        user_id = _p(request).namespace(body.get("user_id")) or default_user
        infer = bool(body.get("infer", True))
        # `defer` returns as soon as the text is durably stored and lets the
        # enrichment worker extract, reconcile and link afterwards. Extraction
        # is several provider round-trips, so a caller that waits for it sits
        # there for seconds to save one note. The text is searchable either way.
        if infer and bool(body.get("defer")) and store.llm.available:
            result = await run_in_threadpool(partial(
                store.add_deferred,
                content if isinstance(content, str) else json.dumps(content),
                user_id=user_id,
                agent_id=body.get("agent_id"),
                run_id=body.get("run_id"),
                metadata=body.get("metadata"),
                importance=float(body.get("importance", 0.5)),
                categories=body.get("categories"),
            ))
            return JSONResponse(result.model_dump(), status_code=202)
        result = await run_in_threadpool(partial(
            store.add,
            content,
            user_id=user_id,
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            metadata=body.get("metadata"),
            infer=infer,
            memory_type=body.get("memory_type", "semantic"),
            importance=float(body.get("importance", 0.5)),
            categories=body.get("categories"),
        ))
        return JSONResponse(result.model_dump(), status_code=201)

    async def get_memory(request: Request) -> Response:
        memory, error = _memory_or_error(request)
        return error or JSONResponse(_memory_payload(memory))

    async def patch_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        body = await request.json()
        try:
            memory = await run_in_threadpool(partial(
                store.update,
                request.path_params["memory_id"],
                content=body.get("content"),
                importance=body.get("importance"),
                categories=body.get("categories"),
                metadata=body.get("metadata"),
                owner_prefix=_p(request).prefix,
            ))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        return JSONResponse(_memory_payload(memory))

    async def delete_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        hard = request.query_params.get("hard") == "true"
        store.delete(
            request.path_params["memory_id"], hard=hard, owner_prefix=_p(request).prefix
        )
        return JSONResponse({"deleted": True, "hard": hard})

    async def forgotten_memories(request: Request) -> Response:
        rows = await run_in_threadpool(partial(
            store.forgotten,
            user_id=_p(request).namespace(request.query_params.get("user_id")),
            limit=int(request.query_params.get("limit", 200)),
        ))
        return JSONResponse([
            {
                "memory": _memory_payload(row["memory"]),
                "forgotten_at": row["forgotten_at"],
                "actor": row["actor"],
                "reason": row["reason"],
            }
            for row in rows
        ])

    async def unforget_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        try:
            restored = await run_in_threadpool(partial(
                store.unforget,
                request.path_params["memory_id"],
                owner_prefix=_p(request).prefix,
            ))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"restored": restored})

    async def purge_memory(request: Request) -> Response:
        """Permanently delete a memory that was already forgotten.

        Separate from DELETE on purpose: that one is reversible in the sense
        that the record survives and stays inspectable, this one is not.
        """
        _, error = _memory_or_error(request)
        if error:
            return error
        try:
            purged = await run_in_threadpool(partial(
                store.purge,
                request.path_params["memory_id"],
                owner_prefix=_p(request).prefix,
            ))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"purged": purged})

    async def memory_history(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        events = store.history(
            request.path_params["memory_id"], owner_prefix=_p(request).prefix
        )
        return JSONResponse([e.model_dump() for e in events])

    async def list_categories_route(request: Request) -> Response:
        q = request.query_params
        user_id = _p(request).namespace(q.get("user_id"))
        cats = await run_in_threadpool(partial(
            store.categories,
            user_id=user_id,
            agent_id=q.get("agent_id"),
            run_id=q.get("run_id"),
        ))
        synthetic = {
            t.tag for t in await run_in_threadpool(
                partial(store.synthetic_tags, user_id=user_id)
            )
        }
        for c in cats:
            if c["category"] in synthetic:
                c["synthetic"] = True
        return JSONResponse(cats)

    async def knowledge_map_route(request: Request) -> Response:
        """All active map aggregates, without memory text or card pagination."""
        q = request.query_params
        data = await run_in_threadpool(partial(
            store.knowledge_map,
            user_id=_p(request).namespace(q.get("user_id")),
            agent_id=q.get("agent_id"),
            run_id=q.get("run_id"),
        ))
        return JSONResponse(data)

    async def synthetic_tags_route(request: Request) -> Response:
        tags = await run_in_threadpool(partial(
            store.synthetic_tags,
            user_id=_p(request).namespace(request.query_params.get("user_id")),
        ))
        return JSONResponse([t.model_dump() for t in tags])

    async def abstract_tags_route(request: Request) -> Response:
        """Run tag abstraction now for the caller's namespace (also runs on a
        weekly schedule; this is the manual trigger)."""
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.abstract_tags,
            user_id=_p(request).namespace(body.get("user_id")),
        ))
        return JSONResponse(result)

    async def consolidate_route(request: Request) -> Response:
        """Preview or apply memory consolidation for the caller's namespace.

        Defaults to a dry run: the dashboard shows every proposed merge, with
        the exact text that would replace the group, before anything changes.
        """
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.consolidate_memories,
            user_id=_p(request).namespace(body.get("user_id")),
            threshold=float(body.get("threshold") or 0.90),
            apply=bool(body.get("apply")),
            # the dashboard sends back only the groups the user ticked
            only=body.get("only") or None,
        ))
        return JSONResponse(result)

    async def maintenance_toggle_route(request: Request) -> Response:
        """Switch an automatic pass on or off, effective from the next cycle.

        Persisted in the store's meta table, so a dashboard toggle survives
        restarts instead of silently reverting to the env-var default.
        """
        body = await request.json()
        key = str(body.get("key", ""))
        ok = await run_in_threadpool(partial(
            store.set_maintenance_enabled, key, bool(body.get("enabled"))
        ))
        if not ok:
            return JSONResponse({"error": "unknown pass"}, status_code=400)
        return JSONResponse({"key": key, "enabled": store.maintenance_enabled(key)})

    async def entity_review_route(request: Request) -> Response:
        """AI review of concept-type entity names: which are not referents.

        A separate POST because it spends LLM tokens; the mechanical tier ships
        free with the maintenance status.
        """
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.entity_junk,
            user_id=_p(request).namespace(body.get("user_id")),
            judge=True,
        ))
        return JSONResponse(result)

    async def remove_entities_route(request: Request) -> Response:
        body = await request.json()
        ids = [str(i) for i in body.get("ids", []) if i]
        if not ids:
            return JSONResponse({"error": "ids required"}, status_code=400)
        if body.get("preserve_as_tag") is True:
            if len(ids) != 1:
                return JSONResponse(
                    {"error": "tag preservation requires exactly one entity"},
                    status_code=400,
                )
            result = await run_in_threadpool(partial(
                store.remove_entity_preserving_tag,
                ids[0],
                owner_prefix=_p(request).prefix,
            ))
            return JSONResponse(result)
        removed = await run_in_threadpool(partial(
            store.remove_entities, ids, owner_prefix=_p(request).prefix,
        ))
        return JSONResponse({"removed": removed})

    async def maintenance_status_route(request: Request) -> Response:
        """What the automatic passes are, when they last ran, and their settings.

        Background work that silently rewrites a user's memory is not
        acceptable; this is the window into it.
        """
        user_id = _p(request).namespace(request.query_params.get("user_id"))
        tcfg = store.config.tags
        return JSONResponse({
            "namespace": user_id,
            "passes": [
                {
                    "key": "dedup_entities",
                    "label": "Entity self-healing",
                    "detail": "Merges duplicate people and things once evidence "
                              "is clear, drops entities nothing references, and "
                              "removes names that cannot be entities at all - "
                              "bare dates, amounts, URLs, salutations.",
                    "automatic": store.maintenance_enabled("dedup_entities"),
                    "interval_days": store.config.dedup_interval_days,
                    "needs_llm": False,
                    "toggleable": True,
                    "run_url": "/api/v1/entities/resolve",
                },
                {
                    "key": "tag_abstraction",
                    "label": "Tag abstraction",
                    "detail": "Groups tags under broader parents for browsing. "
                              "Off by default: measured retrieval is best at the "
                              "specific tag level, not the broad one.",
                    "automatic": store.maintenance_enabled("tag_abstraction"),
                    "interval_days": tcfg.interval_days,
                    "last_run": store.last_tag_run(user_id),
                    "needs_llm": True,
                    "toggleable": True,
                    "run_url": "/api/v1/tags/abstract",
                },
                {
                    "key": "consolidation",
                    "label": "Memory consolidation",
                    "detail": "Merges memories that record the same fact more than "
                              "once. Manual: review each group before applying.",
                    "automatic": False,
                    "needs_llm": True,
                },
            ],
            "llm_available": store.llm.available,
            "embedding_model": store.embedder.model_id,
            "tag_health": await run_in_threadpool(partial(
                store.tag_health, user_id=user_id)),
            "entity_junk": await run_in_threadpool(partial(
                store.entity_junk, user_id=user_id)),
        })

    async def suggest_merges_route(request: Request) -> Response:
        user_id = _p(request).namespace(request.query_params.get("user_id"))
        await run_in_threadpool(partial(store.merge_obvious_topics, user_id=user_id))
        groups = await run_in_threadpool(partial(
            store.suggest_tag_merges,
            user_id=user_id,
        ))
        return JSONResponse(groups)

    async def relations_route(request: Request) -> Response:
        rels = await run_in_threadpool(partial(
            store.relations,
            user_id=_p(request).namespace(request.query_params.get("user_id")),
            limit=int(request.query_params.get("limit", "500")),
        ))
        return JSONResponse([r.model_dump() for r in rels])

    async def backfill_relations_route(request: Request) -> Response:
        """One-time: extract typed relations from existing memories. Cheap and
        resumable (only 2+ entity memories, one small call each, marked done)."""
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.backfill_relations,
            user_id=_p(request).namespace(body.get("user_id")),
        ))
        return JSONResponse(result)

    async def backfill_entity_types_route(request: Request) -> Response:
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.backfill_entity_types,
            user_id=_p(request).namespace(body.get("user_id")),
        ))
        return JSONResponse(result)

    async def repair_dates_route(request: Request) -> Response:
        body = await request.json() if await request.body() else {}
        result = await run_in_threadpool(partial(
            store.repair_updated_at,
            user_id=_p(request).namespace(body.get("user_id")),
        ))
        return JSONResponse(result)

    async def edit_tags_route(request: Request) -> Response:
        """Manual tag curation: rename, merge, or delete a tag across memories.

        {op: "rename", tag, to} | {op: "merge", tags: [...], to} |
        {op: "delete", tag}
        """
        body = await request.json()
        op = body.get("op")
        user_id = _p(request).namespace(body.get("user_id"))
        try:
            if op == "rename":
                changed = await run_in_threadpool(partial(
                    store.rename_tag, body["tag"], body["to"], user_id=user_id))
            elif op == "merge":
                changed = await run_in_threadpool(partial(
                    store.merge_tags, body["tags"], body["to"], user_id=user_id))
            elif op == "delete":
                changed = await run_in_threadpool(partial(
                    store.delete_tag, body["tag"], user_id=user_id))
            else:
                return JSONResponse({"error": "op must be rename|merge|delete"},
                                    status_code=400)
        except (KeyError, TypeError):
            return JSONResponse({"error": "missing fields for op"}, status_code=400)
        return JSONResponse({"op": op, "memories_changed": changed})

    async def export_memories_route(request: Request) -> Response:
        q = request.query_params
        try:
            backup = await run_in_threadpool(partial(
                store.export_backup,
                user_id=_p(request).namespace(q.get("user_id")),
                agent_id=q.get("agent_id"),
                run_id=q.get("run_id"),
            ))
        except NotImplementedError as exc:
            return JSONResponse({"error": str(exc)}, status_code=501)
        return JSONResponse(backup)
    async def import_memories_route(request: Request) -> Response:
        """Restore a lossless backup or accept legacy additive memory rows."""
        body = await request.json()
        principal = _p(request)
        if isinstance(body, dict) and body.get("format") == "memry-backup":
            try:
                result = await run_in_threadpool(partial(
                    store.import_backup, body, owner_prefix=principal.prefix
                ))
            except NotImplementedError as exc:
                return JSONResponse({"error": str(exc)}, status_code=501)
            except ValueError as exc:
                status = 409 if "conflict" in str(exc).lower() else 400
                return JSONResponse({"error": str(exc)}, status_code=status)
            return JSONResponse(result, status_code=201 if result["inserted"] else 200)
        rows = body if isinstance(body, list) else body.get("memories")
        if not isinstance(rows, list) or not rows:
            return JSONResponse(
                {"error": "provide a JSON array of rows or {\"memories\": [...]}"},
                status_code=400,
            )
        default_uid = None if isinstance(body, list) else body.get("user_id")
        sanitized = [
            {**row, "user_id": principal.namespace(row.get("user_id") or default_uid)}
            for row in rows
            if isinstance(row, dict)
        ]
        result = await run_in_threadpool(partial(
            store.import_verbatim,
            sanitized,
            user_id=principal.namespace(default_uid) or default_user,
        ))
        return JSONResponse(result, status_code=201)

    async def distill_memory(request: Request) -> Response:
        _, error = _memory_or_error(request)
        if error:
            return error
        try:
            result = await run_in_threadpool(
                partial(
                    store.distill,
                    request.path_params["memory_id"],
                    owner_prefix=_p(request).prefix,
                )
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
        entity_id, error = _resolve_entity_filter(request, body.get("entity_id"))
        if error:
            return error
        results = await run_in_threadpool(partial(
            store.search,
            body.get("query", ""),
            user_id=_p(request).namespace(body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            limit=int(body.get("limit", 10)),
            include_invalid=bool(body.get("include_invalid", False)),
            categories=_parse_categories(body.get("categories")),
            entity_id=entity_id,
            since=body.get("since") or None,
            until=body.get("until") or None,
        ))
        return JSONResponse(
            [
                {"memory": _memory_payload(r.memory), "score": r.score, "signals": r.signals}
                for r in results
            ]
        )

    async def context(request: Request) -> Response:
        body = await request.json()
        ctx = await run_in_threadpool(partial(
            store.reconstruct_context,
            body.get("query", ""),
            user_id=_p(request).namespace(body.get("user_id")),
            agent_id=body.get("agent_id"),
            run_id=body.get("run_id"),
            token_budget=int(body.get("token_budget", 1200)),
        ))
        return JSONResponse(ctx.model_dump())

    async def stats(request: Request) -> Response:
        data = await run_in_threadpool(store.stats)
        principal = _p(request)
        if principal.prefix is not None:
            everything = await run_in_threadpool(
                partial(store.get_all, include_invalid=True, limit=100_000)
            )
            mine = [
                m for m in everything
                if principal.owns(m.user_id)
            ]
            # Server-wide facts (which models are configured) are not another
            # account's data, and the About panel needs them; omitting them was
            # what rendered "llm undefined" in the dashboard.
            data = {
                "backend": data.get("backend"),
                "tenant": principal.name,
                "active_memories": sum(1 for m in mine if m.invalid_at is None),
                "invalidated_memories": sum(1 for m in mine if m.invalid_at is not None),
                "forgotten_memories": sum(
                    1 for m in mine if m.invalid_at is not None and not m.superseded_by
                ),
                "llm": data.get("llm"),
                "embedder": data.get("embedder"),
            }
        return JSONResponse(json.loads(json.dumps(data, default=str)))

    # -- entities ---------------------------------------------------------
    async def list_entities(request: Request) -> Response:
        q = request.query_params
        entities = store.entities(
            user_id=_p(request).namespace(q.get("user_id")),
            include_merged=q.get("include_merged") == "true",
            limit=int(q.get("limit", "100")),
        )
        return JSONResponse([e.model_dump() for e in entities])

    async def get_entity(request: Request) -> Response:
        detail = await run_in_threadpool(partial(
            store.entity,
            request.path_params["entity_id"],
            owner_prefix=_p(request).prefix,
        ))
        if detail is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "entity": detail["entity"].model_dump(),
                "aliases": detail["aliases"],
                "mentions": [mention.model_dump() for mention in detail["mentions"]],
                "memories": [_memory_payload(memory) for memory in detail["memories"]],
                "relations": [r.model_dump() for r in detail.get("relations", [])],
                "relation_names": detail.get("relation_names", {}),
            }
        )

    async def add_entity_alias(request: Request) -> Response:
        body = await request.json()
        alias = str(body.get("alias", "")).strip()
        if not alias:
            return JSONResponse({"error": "alias required"}, status_code=400)
        entity = store.add_entity_alias(
            request.path_params["entity_id"],
            alias,
            owner_prefix=_p(request).prefix,
        )
        if entity is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "entity": entity.model_dump(),
                "aliases": store.backend.entity_aliases(entity.id),
            }
        )

    async def rename_entity_route(request: Request) -> Response:
        body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)
        entity = store.rename_entity(
            request.path_params["entity_id"],
            name,
            owner_prefix=_p(request).prefix,
        )
        if entity is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "entity": entity.model_dump(),
                "aliases": store.backend.entity_aliases(entity.id),
            }
        )

    async def merge_entities_route(request: Request) -> Response:
        """User-confirmed direct merge: fold merge_id into keep_id."""
        body = await request.json()
        keep_id = str(body.get("keep_id", "")).strip()
        merge_id = str(body.get("merge_id", "")).strip()
        if not keep_id or not merge_id:
            return JSONResponse(
                {"error": "keep_id and merge_id required"}, status_code=400
            )
        if keep_id == merge_id:
            return JSONResponse(
                {"error": "entities must be different"}, status_code=400
            )
        merged = await run_in_threadpool(partial(
            store.merge_entities,
            keep_id,
            merge_id,
            owner_prefix=_p(request).prefix,
        ))
        if not merged:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {"merged": True, "keep_id": keep_id, "merge_id": merge_id}
        )

    async def list_proposals(request: Request) -> Response:
        q = request.query_params
        proposals = store.merge_proposals(
            user_id=_p(request).namespace(q.get("user_id")),
            status=q.get("status", "proposed"),
            limit=int(q.get("limit", "100")),
        )
        return JSONResponse([p.model_dump() for p in proposals])

    def _proposal_guard(request: Request):
        proposal = store.backend.get_proposal(request.path_params["proposal_id"])
        if proposal is None or not _p(request).owns(proposal.user_id):
            return None, JSONResponse({"error": "not found"}, status_code=404)
        return proposal, None

    async def confirm_proposal(request: Request) -> Response:
        proposal, error = _proposal_guard(request)
        if error:
            return error
        ok = store.confirm_merge(proposal.id, owner_prefix=_p(request).prefix)
        return JSONResponse(
            {"confirmed": ok, "proposal_id": proposal.id},
            status_code=200 if ok else 409,
        )

    async def reject_proposal(request: Request) -> Response:
        proposal, error = _proposal_guard(request)
        if error:
            return error
        ok = store.reject_merge(proposal.id, owner_prefix=_p(request).prefix)
        return JSONResponse(
            {"rejected": ok, "proposal_id": proposal.id},
            status_code=200 if ok else 409,
        )

    async def resolve_entities_route(request: Request) -> Response:
        body = await request.json() if await request.body() else {}
        outcome = await run_in_threadpool(partial(
            store.resolve_entities,
            user_id=_p(request).namespace(body.get("user_id")),
        ))
        return JSONResponse(outcome)

    # -- OAuth login / consent -------------------------------------------
    def _login_page(request_id: str, client_name: str, error: str = "") -> HTMLResponse:
        return HTMLResponse(_LOGIN_PAGE.format(
            request_id=html.escape(request_id),
            client=html.escape(client_name),
            error=f'<div class="err">{html.escape(error)}</div>' if error else "",
        ))

    async def _client_label(client_id: str) -> str:
        client = await oauth.get_client(client_id)
        name = getattr(client, "client_name", None) if client else None
        return name or client_id

    async def oauth_login_form(request: Request) -> Response:
        request_id = request.query_params.get("request", "")
        pending = oauth.load_pending(request_id)
        if pending is None:
            return HTMLResponse(
                "<p>This sign-in link has expired. Start again from your client.</p>",
                status_code=400,
            )
        return _login_page(request_id, await _client_label(pending[0]))

    async def oauth_login_submit(request: Request) -> Response:
        form = await request.form()
        request_id = str(form.get("request", ""))
        pending = oauth.load_pending(request_id)
        if pending is None:
            return HTMLResponse(
                "<p>This sign-in link has expired. Start again from your client.</p>",
                status_code=400,
            )
        client_id, params = pending
        if form.get("decision") != "approve":
            return RedirectResponse(
                construct_redirect_uri(
                    str(params.redirect_uri),
                    error="access_denied",
                    error_description="the user denied the request",
                    state=params.state,
                ),
                status_code=302,
            )

        account = accounts.get_by_name(str(form.get("account", "")).strip())
        password = str(form.get("password", ""))
        if account is None or account.disabled or not account.check_password(password):
            # one message for every failure: no probing for which accounts exist
            return _login_page(
                request_id,
                await _client_label(client_id),
                "Wrong account or password.",
            )

        redirect = oauth.complete_authorization(request_id, account.name)
        if redirect is None:  # pragma: no cover - consumed concurrently
            return HTMLResponse("<p>Sign-in expired, please retry.</p>", status_code=400)
        return RedirectResponse(redirect, status_code=302)

    async def guarded_mcp_app(scope, receive, send):
        """MCP over HTTP, authenticated exactly like the REST API.

        Tenant keys work here too: the tools derive their namespace from the
        principal published below rather than from their own ``user_id``
        argument, so a tenant is confined on this transport the same way it is
        on /api.

        The key may arrive as ``Authorization: Bearer <key>`` or, for clients
        that cannot send headers (claude.ai custom connectors), embedded in the
        URL as ``/mcp/<key>`` or ``/mcp?key=<key>``."""
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
            if segments and _is_configured_key(segments[0]):
                token = segments[0]
                # Strip the key segment so the MCP app sees its root path.
                scope = dict(scope)
                scope["path"] = root + "/" + "/".join(segments[1:])
            else:
                query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                token = (query.get("key") or [""])[0]
        principal = resolve_principal(token)
        if principal is None:
            # RFC 9728: point the client at the protected-resource document so
            # it can discover the authorization server. Without this header a
            # client falls back to guessing /.well-known/oauth-authorization-server
            # and reports the 404 instead of starting the OAuth flow.
            unauthorized_headers = {}
            if public_url:
                # a client configured with the bare site URL treats that as the
                # resource, so point it at the document describing that URL
                suffix = "" if scope.get(MCP_ORIGIN_KEY) == "/" else "/mcp"
                unauthorized_headers["WWW-Authenticate"] = (
                    'Bearer resource_metadata='
                    f'"{public_url}/.well-known/oauth-protected-resource{suffix}"'
                )
            await JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers=unauthorized_headers,
            )(scope, receive, send)
            return
        scope = dict(scope)
        scope[PRINCIPAL_SCOPE_KEY] = principal
        await mcp_app(scope, receive, send)

    async def _maintenance_scheduler() -> None:
        """Periodic per-namespace maintenance: deterministic topic cleanup,
        entity de-duplication (resolve stale/obvious proposals), and optional tag
        abstraction. Each task runs on its own interval; last-run times are
        persisted (backend meta) so restarts don't re-run, and per-cycle work is
        capped so a many-account server spreads LLM cost across cycles.
        """
        tcfg = store.config.tags
        tag_interval = max(tcfg.interval_days, 0.001)
        dedup_interval = max(store.config.dedup_interval_days, 0.001)
        check_every = max(min(min(tag_interval, dedup_interval) * 86400, 6 * 3600), 60)
        max_per_cycle = 25

        def dedup_key(uid: str | None) -> str:
            return f"entity_dedup:v2:last_run:{uid or ''}"

        while True:
            try:
                now = datetime.now(timezone.utc)
                stamp = now.isoformat(timespec="seconds")
                processed = 0
                for uid in store.backend.distinct_user_ids() or [None]:
                    if processed >= max_per_cycle:
                        break
                    did = False
                    if store.maintenance_enabled("dedup_entities") and _tag_run_due(
                        store.backend.get_meta(dedup_key(uid)), dedup_interval, now
                    ):
                        await run_in_threadpool(store.merge_obvious_topics, user_id=uid)
                        await run_in_threadpool(store.resolve_entities, user_id=uid)
                        store.backend.set_meta(dedup_key(uid), stamp)
                        did = True
                    if store.maintenance_enabled("tag_abstraction") and _tag_run_due(
                        store.last_tag_run(uid), tag_interval, now
                    ):
                        await run_in_threadpool(store.abstract_tags, user_id=uid)
                        did = True
                    processed += 1 if did else 0
            except Exception:  # a scheduler hiccup must never take the server down
                pass
            await asyncio.sleep(check_every)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        maintenance_task: asyncio.Task | None = None
        enrichment_task = asyncio.create_task(enrichment_worker.run())
        # Always started: each cycle consults the runtime switches, so a pass
        # toggled on from the dashboard begins running without a restart.
        maintenance_task = asyncio.create_task(_maintenance_scheduler())
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            for task in (enrichment_task, maintenance_task):
                if task is not None:
                    task.cancel()
            for task in (enrichment_task, maintenance_task):
                if task is not None:
                    with contextlib.suppress(BaseException):
                        await task

    routes = [
        Route("/", dashboard),
        Route("/health", health),
        Route("/login", login_form, methods=["GET"]),
        Route("/login", login_submit, methods=["POST"]),
        Route("/logout", logout, methods=["GET", "POST"]),
        Route("/api/v1/memories", guarded(list_memories), methods=["GET"]),
        Route("/api/v1/memories", guarded(create_memory), methods=["POST"]),
        # before /{memory_id}, or "forgotten" is read as an id and 404s
        Route("/api/v1/memories/forgotten", guarded(forgotten_memories), methods=["GET"]),
        Route("/api/v1/memories/{memory_id}/purge", guarded(purge_memory), methods=["POST"]),
        Route("/api/v1/memories/{memory_id}/unforget", guarded(unforget_memory), methods=["POST"]),
        Route("/api/v1/memories/{memory_id}", guarded(get_memory), methods=["GET"]),
        Route("/api/v1/memories/{memory_id}", guarded(patch_memory), methods=["PATCH"]),
        Route("/api/v1/memories/{memory_id}", guarded(delete_memory), methods=["DELETE"]),
        Route("/api/v1/memories/{memory_id}/history", guarded(memory_history), methods=["GET"]),
        Route("/api/v1/memories/{memory_id}/distill", guarded(distill_memory), methods=["POST"]),
        Route("/api/v1/categories", guarded(list_categories_route), methods=["GET"]),
        Route("/api/v1/map", guarded(knowledge_map_route), methods=["GET"]),
        Route("/api/v1/tags/synthetic", guarded(synthetic_tags_route), methods=["GET"]),
        Route("/api/v1/tags/abstract", guarded(abstract_tags_route), methods=["POST"]),
        Route("/api/v1/tags/edit", guarded(edit_tags_route), methods=["POST"]),
        Route("/api/v1/tags/suggest-merges", guarded(suggest_merges_route), methods=["GET"]),
        Route("/api/v1/maintenance", guarded(maintenance_status_route), methods=["GET"]),
        Route("/api/v1/maintenance/consolidate", guarded(consolidate_route), methods=["POST"]),
        Route("/api/v1/maintenance/toggle", guarded(maintenance_toggle_route), methods=["POST"]),
        Route("/api/v1/maintenance/entity-review", guarded(entity_review_route), methods=["POST"]),
        Route("/api/v1/entities/remove", guarded(remove_entities_route), methods=["POST"]),
        Route("/api/v1/relations", guarded(relations_route), methods=["GET"]),
        Route("/api/v1/relations/backfill", guarded(backfill_relations_route), methods=["POST"]),
        Route("/api/v1/entities/backfill-types", guarded(backfill_entity_types_route), methods=["POST"]),
        Route("/api/v1/memories/repair-dates", guarded(repair_dates_route), methods=["POST"]),
        Route("/api/v1/export", guarded(export_memories_route), methods=["GET"]),
        Route("/api/v1/import", guarded(import_memories_route), methods=["POST"]),
        Route("/api/v1/search", guarded(search), methods=["POST"]),
        Route("/api/v1/context", guarded(context), methods=["POST"]),
        Route("/api/v1/stats", guarded(stats), methods=["GET"]),
        Route("/api/v1/entities", guarded(list_entities), methods=["GET"]),
        Route("/api/v1/entities/merge", guarded(merge_entities_route), methods=["POST"]),
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
        Route(
            "/api/v1/entities/{entity_id}/aliases",
            guarded(add_entity_alias),
            methods=["POST"],
        ),
        Route("/api/v1/entities/{entity_id}", guarded(rename_entity_route), methods=["PATCH"]),
        Route("/api/v1/entities/{entity_id}", guarded(get_entity), methods=["GET"]),
    ]
    if oauth is not None:
        # These MUST live at the domain root. FastMCP would add them inside its
        # own app, which is mounted at /mcp, putting the metadata documents at
        # /mcp/.well-known/... where no client looks for them.
        routes += create_auth_routes(
            provider=oauth,
            issuer_url=AnyHttpUrl(public_url),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=[MEMRY_SCOPE],
                default_scopes=[MEMRY_SCOPE],
            ),
            revocation_options=RevocationOptions(enabled=True),
        )
        # Two documents, one per URL a client may have been given: /mcp, and
        # the bare site URL that connector UIs invite people to paste. RFC 9728
        # keys the document by the resource's path, and a client that asked for
        # one URL will not read the other's document.
        for resource in (f"{public_url}/mcp", public_url):
            routes += create_protected_resource_routes(
                resource_url=AnyHttpUrl(resource),
                authorization_servers=[AnyHttpUrl(public_url)],
                scopes_supported=[MEMRY_SCOPE],
                resource_name="Memry",
            )
        routes += [
            Route("/oauth/login", oauth_login_form, methods=["GET"]),
            Route("/oauth/login", oauth_login_submit, methods=["POST"]),
        ]
    routes.append(Mount("/mcp", app=guarded_mcp_app))
    return Starlette(
        routes=routes, lifespan=lifespan, middleware=[Middleware(_NormalizeMcpPath)]
    )


_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_open_mode(store: MemoryStore, accounts: AccountStore) -> bool:
    """True when every request would be treated as admin (no key, tenants or accounts)."""
    return not store.config.api_key and not store.config.tenants and accounts.is_empty()


def check_bind_safety(store: MemoryStore, accounts: AccountStore, host: str) -> None:
    """Refuse to expose an unauthenticated server beyond loopback.

    resolve_principal() grants ADMIN to everyone in open mode, which is fine on
    127.0.0.1 and a public read/write memory store on 0.0.0.0. Setting
    MEMRY_API_KEY (or creating an account) is the fix; MEMRY_ALLOW_OPEN=1 is
    the explicit escape hatch for a port that a private network or a
    reverse proxy really does protect.
    """
    if not is_open_mode(store, accounts):
        return
    log = logging.getLogger("memry")
    if host in _LOOPBACK_HOSTS:
        log.warning(
            "no MEMRY_API_KEY, tenants or accounts configured: serving WITHOUT "
            "authentication on %s (loopback only)", host,
        )
        return
    if os.environ.get("MEMRY_ALLOW_OPEN") == "1":
        log.warning(
            "MEMRY_ALLOW_OPEN=1: serving WITHOUT authentication on %s. Anyone who "
            "can reach this port can read and write every memory.", host,
        )
        return
    lines = [
        f"memry serve: refusing to bind {host} without authentication - anyone "
        "who could reach this port would be able to read and write every memory.",
        "Fix one of:",
        "  - set MEMRY_API_KEY=<secret> (bearer token for REST and MCP), or",
        "  - create an account: memry account add <name> --password ..., or",
        "  - bind loopback only: memry serve --host 127.0.0.1, or",
        "  - set MEMRY_ALLOW_OPEN=1 if a private network or reverse proxy really "
        "protects this port.",
    ]
    raise SystemExit("\n".join(lines))


def main(host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    store = MemoryStore()
    accounts = AccountStore(
        store.config.auth_db_path or default_auth_db_path(store.config.db_path)
    )
    check_bind_safety(store, accounts, host)

    # Behind a TLS-terminating proxy (the bundled Caddy), uvicorn must trust
    # X-Forwarded-Proto or it builds redirects as http:// - e.g. /mcp -> /mcp/.
    # Clients drop the Authorization header on such cross-scheme redirects and
    # then see a 401. Only the proxy can reach this port, so trust any peer.
    uvicorn.run(
        create_app(store, accounts=accounts),
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
