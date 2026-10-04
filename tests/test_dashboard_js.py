"""The dashboard's JavaScript must actually parse.

The dashboard is one big inline script built by string concatenation in
rest.py, so an editing slip can leave a stray brace and ship a page that loads,
returns 200, passes every API test, and then dies on the first line of script -
memories stuck on "loading" with a single console error. That happened. Python
tests cannot see it because the JS is opaque text to them.

`node --check` parses without executing. Where node is unavailable the test
skips rather than pretending to have checked.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import Entity, EntityMention, MergeProposal
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to parse the dashboard JS"
)


def _dashboard_html() -> str:
    store = MemoryStore(
        Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64)
    )
    try:
        with TestClient(create_app(store)) as client:
            return client.get("/").text
    finally:
        store.close()


def _scripts(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def test_dashboard_javascript_parses(tmp_path):
    blocks = _scripts(_dashboard_html())
    assert blocks, "the dashboard should serve at least one inline script"
    for index, source in enumerate(blocks):
        path = tmp_path / f"dashboard_{index}.js"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"dashboard script block {index} is not valid JavaScript:\n"
            f"{result.stderr}"
        )


def test_map_uses_complete_aggregates_entity_types_and_rendering_bounds():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert '<details class="gx-types" id="mapEntityFilter">' in html
    assert '<summary id="mapEntitiesBtn"' in html
    assert '>Types</summary>' not in html
    # the type filter sits over the map, so it has all three ways out
    assert '<div class="gx-type-head"><span>Entity types</span>' in html
    assert 'class="x" onclick="closeMapEntityFilter()"' in html
    assert (
        "function closeMapEntityFilter()"
        "{document.getElementById('mapEntityFilter').open=false}" in source
    )
    assert "if(event.target.closest&&event.target.closest('.gx-types'))return" in source
    assert "if(mapEntityFilterOpen()){closeMapEntityFilter();return}" in source
    assert 'aria-label="Memory type shapes"' in html
    assert "const data=await api(mapDataPath(tags));" in source
    assert "const MAX_IDLE_EDGES=400" in source
    assert "const displayedEdges=displayedGalaxyEdges(G,sel,hov)" in source
    assert "A+=((hovTouches?1:0.06)-A)*hoverMix" in source
    assert "const satelliteFocus=sel||hov" in source
    assert (
        "const showSatellites=satelliteFocus?focusedNeighbor:"
        "(G.lod?n.zone==='core':n.zone!=='rim')" in source
    )
    assert "const LOD_NODES=400,LOD_FRAME_MS=32" in source
    assert "function planetSprite(n,c,dark,dpr)" in source
    assert "ctx.drawImage(galaxyBackdrop(W,H,dark,WARM,STAR,dpr),0,0,W,H)" in source
    assert "function drawMemoryMarker(ctx,type,x,y,size)" in source
    assert 'data-entity-type="' in source
    assert "handleMapEntityTypeChange" in source

    type_label = source[source.index("function typeLabel(") :].split("\n", 1)[0]
    map_source = source[
        source.index("const hashCode=") : source.index("function drawMap(){")
    ]
    contract = type_label + """
const window={};
const nodes={};
const document={getElementById:id=>(nodes[id]??={setAttribute:()=>{},addEventListener:()=>{}})};
const stored={};
const localStorage={getItem:key=>stored[key]??null,setItem:(key,value)=>{stored[key]=value}};
const matchMedia=()=>({matches:true});
let activeMapKey=null,hoverMapKey=null,hoverFocusTag=null,redraws=0;
let knowledgeMapSuspended=false;
const updateHover=()=>{},clearMapEntityDetail=()=>{};
const drawMap=()=>{redraws++};
const esc=value=>value;
const asked=[];
const api=async path=>{asked.push(path);return {memories:1,entity_memories:1,
  entities:[{key:'entity:t1',label:'work',kind:'entity',entity_id:'t1',entity_type:'topic',count:3,type_counts:{semantic:3}}],
  entity_edges:[]}};
""" + map_source + """
function check(condition,message){if(!condition)throw new Error(message)}
const edges=Array.from({length:430},(_,index)=>({
  a:'entity:hub',b:'entity:t'+index,weight:1
}));
const data={
  memories:432,entity_memories:431,withTags:true,
  entities:[
    {key:'entity:hub',label:'Ada',kind:'entity',entity_id:'hub',entity_type:'person',
     count:2,type_counts:{semantic:1,procedural:1}},
    ...Array.from({length:430},(_,index)=>({
      key:'entity:t'+index,label:'t'+index,kind:'entity',entity_id:'t'+index,
      entity_type:'person',count:1,type_counts:{episodic:1}
    })),
    {key:'entity:rag-1',label:'RAG',kind:'entity',entity_id:'rag-1',
     entity_type:'concept',count:1,type_counts:{semantic:1}},
    {key:'entity:work',label:'work',kind:'entity',entity_id:'work',
     entity_type:'topic',count:5,type_counts:{semantic:5}}
  ],
  entity_edges:[...edges,{a:'entity:hub',b:'entity:rag-1',weight:1},
    {a:'entity:work',b:'entity:hub',weight:2}]
};
mapData=data;
const big=buildGalaxy(data);
check(big.total===431,'linked memory total');
check(big.byKey['entity:hub'].count===2,'entity count');
check(big.byKey['entity:hub'].typeCounts.procedural===1,'type counts');
check(big.idleEdges.length===400,'idle edge cap');
check(big.lod===true,'431 planets is above the detail threshold');
check(big.byKey['entity:hub'].satTypes.length===2,'orbit marker types are precomputed');
const hoverEdges=displayedGalaxyEdges(big,null,big.byKey['entity:hub']);
check(hoverEdges.length===430,'hover shows every node edge');
check(big.idleEdges.every(edge=>hoverEdges.includes(edge)),'hover preserves every idle edge');
check(displayedGalaxyEdges(big,big.byKey['entity:hub'],null).length===430,'selection shows every node edge');
check(!big.byKey['entity:rag-1'],'concept should default off');
check(!big.byKey['entity:work'],'tags start off');
check(!mapWantsTags(),'so the map does not ask for them');
check(mapDataPath(false)==='/api/v1/map?kind=named','people and things only');
check(mapDataPath(true)==='/api/v1/map?kind=any','tags when asked for');
check(knownEntityTypes().join()==='concept,person,topic','the tag type is offered: '+knownEntityTypes().join());
renderMapEntityTypes();
const menu=nodes.mapEntityTypeOptions.innerHTML;
check(menu.includes('<span>tag</span>')&&!menu.includes('<span>topic</span>'),'tags read "tag"');
mapEntityTypes.add('concept');
check(buildGalaxy(data).byKey['entity:rag-1'].entityType==='concept','concept opt-in');
handleMapEntityTypeChange({target:{
  matches:selector=>selector==='input[data-entity-type]',
  dataset:{entityType:'concept'},checked:false
}});
check(!mapEntityTypes.has('concept'),'checkbox updates selected entity types');
check(redraws===1,'checkbox redraws map immediately');
check(asked.length===0,'no reload for a type that is loaded');
toggleMapEntityType('topic',true);
const withTag=buildGalaxy(data);
check(withTag.byKey['entity:work'].entityType==='topic','a tag is a planet once its type is on');
check(withTag.edges.some(edge=>withTag.nodes[edge.a].key==='entity:work'||withTag.nodes[edge.b].key==='entity:work'),
  'and its co-mention edges are drawn');
check(JSON.parse(stored.memry_map_entity_types).includes('topic'),'the choice is remembered');
check(asked.length===0,'tags already loaded are not asked for again');
// a map loaded without tags asks for them when their type is turned on
mapEntityTypes.delete('topic');data.withTags=false;
toggleMapEntityType('topic',true);
check(asked.join()==='/api/v1/map?kind=any','turning tags on loads them: '+asked.join());
// a later page load remembers it
mapEntityTypes=null;
check(mapWantsTags(),'a remembered tag type is asked for at once');
stored.memry_map_entity_types=JSON.stringify(['person']);
check(!mapWantsTags(),'and one left off is not');
mapEntityTypes=null;
setMapEntityTypes('defaults');
check(!mapEntityTypes.has('topic')&&!mapEntityTypes.has('concept')&&mapEntityTypes.has('person'),'defaults leave tags off');
setMapEntityTypes('all');
check(mapEntityTypes.has('topic'),'all means tags too');
setMapEntityTypes('none');
check(mapEntityTypes.size===0,'none');
// A long tail: the twos leave the over-packed belt for the rim.
mapEntityTypes=new Set(['person']);
const crowd=(counts,edgeRows=[])=>({memories:1,entities:counts.map((count,index)=>({
  key:'entity:c'+index,label:'c'+index,kind:'entity',entity_id:'c'+index,entity_type:'person',
  count,type_counts:{semantic:count}
})),entity_edges:edgeRows});
const zones=graph=>graph.nodes.reduce((seen,node)=>{
  (seen[node.zone]??=new Set()).add(node.count);return seen},{});
const tail=[200,...Array(20).fill(1),...Array(60).fill(2),...Array(40).fill(3),
  ...Array(30).fill(5),...Array(20).fill(8)];
const long=zones(buildGalaxy(crowd(tail)));
check(long.rim.has(1)&&long.rim.has(2),'ones and twos sit on the rim');
check(!long.rim.has(3)&&long.belt.has(3),'threes stay in the belt');
check(!long.belt.has(2),'no two is left in the belt');
// A small store never had a crowded belt, so it keeps the two-sigma split.
const small=zones(buildGalaxy(crowd([12,7,5,4,3,3,2,2,1,1,1])));
check(small.rim.has(1)&&!small.rim.has(2)&&small.belt.has(2),
  'a belt with room keeps its twos');
// The belt must stay at least as dense as the rim, so it hands over a share of
// its twos and keeps the rest: 62 in the belt against 1 on the rim leaves room
// for 35 of them.
const packed=buildGalaxy(crowd([200,1,...Array(60).fill(2),3,4]));
const twosIn=zone=>packed.nodes.filter(node=>node.count===2&&node.zone===zone).length;
check(twosIn('rim')===35,'the rim takes the twos that fit');
check(twosIn('belt')===25,'the belt keeps the rest of its twos');
// Least-linked first: the one two with edges is last in line and stays put.
const linked=buildGalaxy(crowd([200,1,...Array(60).fill(2),3,4],[
  {a:'entity:c11',b:'entity:c0',weight:1},{a:'entity:c11',b:'entity:c62',weight:1},
  {a:'entity:c11',b:'entity:c63',weight:1}]));
check(linked.byKey['entity:c11'].zone==='belt','a well-linked two keeps its place');
check(linked.byKey['entity:c10'].zone==='rim','an unlinked two goes out to the rim');
// Tags turned on while a load without them is on its way: they are asked
// for, and the later load is the one kept.
(async()=>{
  mapEntityTypes=new Set(['person']);mapData=null;asked.length=0;
  const first=loadMapData();
  toggleMapEntityType('topic',true);
  await first;await new Promise(resolve=>setTimeout(resolve,0));
  check(asked.join()==='/api/v1/map?kind=named,/api/v1/map?kind=any',
        'both loads asked: '+asked.join());
  check(mapData&&mapData.withTags===true,'the load with tags is kept');
})().catch(e=>{console.error(e.message);process.exit(1)});"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

def test_knowledge_modal_releases_and_restores_the_map():
    source = "\n".join(_scripts(_dashboard_html()))
    modal_source = source[
        source.index("let knowledgeNames={};") : source.index("function openAbout(){")
    ]
    contract = r"""
const classList=()=>({
  values:new Set(),
  toggle(name,on){on?this.values.add(name):this.values.delete(name)},
  contains(name){return this.values.has(name)}
});
const nodes={
  knowmodal:{classList:classList()},
  mapwrap:{hidden:false},
  mapentitydetail:{hidden:false},
  map:{width:900,height:500}
};
const document={
  getElementById:id=>nodes[id],
  documentElement:{classList:classList()},
  body:{classList:classList()}
};
const panels={map:true};
let G={heavy:true},mapData={heavy:true},gPulses=[1],hoverMapKey='entity:x',gRAF=7;
let cancelled=0,loads=0,draws=0;
const cancelAnimationFrame=()=>cancelled++;
const loadMapData=()=>loads++;
const drawMap=()=>draws++;
function check(condition,message){if(!condition)throw new Error(message)}
""" + modal_source + r"""
setKnowledgeOpen(true);
check(knowledgeMapSuspended,'map should be suspended');
check(G===null&&mapData===null&&gPulses.length===0,'heavy map state released');
check(gRAF===0&&cancelled===1,'animation cancelled');
check(nodes.map.width===1&&nodes.map.height===1,'canvas buffer released');
check(nodes.mapwrap.hidden&&nodes.mapentitydetail.hidden,'map UI hidden');
setKnowledgeOpen(true);
check(knowledgeMapWasOpen,'opening twice must retain restore state');
setKnowledgeOpen(false);
check(!knowledgeMapSuspended&&loads===1,'open map restored once');
panels.map=false;G={closed:true};mapData={closed:true};
setKnowledgeOpen(true);setKnowledgeOpen(false);
check(G.closed&&mapData.closed&&loads===1,'closed map stays closed');
"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "if(knowledgeMapSuspended){mapData=null;return}" in source
    assert (
        "const visible=panels.map&&!knowledgeMapSuspended&&mapData&&mapData.memories"
        in source
    )


def test_selected_map_entity_shows_identity_and_cleanup_actions():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert 'id="mapentitydetail"' in html
    assert "function entityIdentityBlock(entity,aliases)" in source
    assert source.count("${entityIdentityBlock(entity,aliases)}") == 2
    assert "function showMapEntityDetail(entityId)" in source
    assert "is duplicate of..." in source
    # the picker stays folded until asked for, so the four actions read as a row
    assert 'id="mapduplicatepicker" hidden' in source
    assert "function toggleDuplicatePicker(button,panel='map')" in source
    assert 'id="mapaliasinput"' not in source
    assert source.count('<div class="entity-actions">') == 2
    assert "function mergeMapEntity(entityId)" in source
    assert "api('/api/v1/entities/merge'" in source
    assert "function removeMapEntity(entityId)" in source
    assert "api('/api/v1/entities/remove'" in source
    assert "preserve_as_tag:true" in source
    assert "Its name will be kept as a tag" in source
    assert "Its memories will stay untouched." in source
    assert "function renameEntity(entityId)" in source
    assert "{method:'PATCH',body:JSON.stringify({name})}" in source
    # the three things you can do to a name sit side by side under it, and both
    # panels ask the same question before retiring one
    assert '<div class="entity-actions">' in source
    for label in (">rename</button>", ">add alias</button>", ">not an entity</button>"):
        assert label in source, label
    assert 'id="aliasinput"' not in source, "the knowledge panel asks, like rename"
    assert "async function removeEntity(id,memories)" in source
    # a name like O'Brien inside the single-quoted onclick ended it early
    assert "JSON.stringify(entity.name)" not in source
    assert source.count("async function confirmNotAnEntity(") == 1
    assert source.count("preserve_as_tag:true") == 1
    map_alias = source.split("async function addMapAlias(entityId){", 1)[1].split(
        "async function refreshAfterMapEntityCleanup", 1,
    )[0]
    knowledge_alias = source.split("async function addAlias(id){", 1)[1].split(
        "async function removeEntity", 1,
    )[0]
    assert "showMapEntityDetail" not in map_alias
    assert "openEntity" not in knowledge_alias
    assert "syncEntityIdentity" in map_alias
    assert "syncEntityIdentity" in knowledge_alias


def test_forgotten_panel_lists_removed_names_with_a_way_back():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert 'id="retiredlist"' in html
    assert ">Removed names</h2>" in html
    assert ("if(tab==='forgotten'){loadForgotten();loadReplaced();"
            "loadRetiredEntities();loadMerges()}") in source
    assert "async function loadRetiredEntities()" in source
    assert "api('/api/v1/entities/retired')" in source
    assert "async function restoreEntity(id)" in source
    assert "api('/api/v1/entities/restore'" in source


def test_the_archive_lists_merged_names_with_a_way_back():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert 'id="mergedlist"' in html and ">Merged names</h2>" in html
    assert "async function loadMerges()" in source
    assert "api('/api/v1/entities/merges')" in source
    assert "async function undoMerge(id)" in source
    assert "api('/api/v1/entities/unmerge'" in source


def test_memory_cards_show_colored_type_symbols():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    for memory_type in ("semantic", "procedural", "episodic", "working"):
        assert f".memory-type.{memory_type}" in html
        assert f"--{memory_type}:" in html
    assert ".memory-type.episodic .type-symbol" in html
    assert ".memory-type.working .type-symbol" in html
    assert "${memoryTypeBadge(m)}" in source

    badge_source = source[
        source.index("function normalizedMemoryType") : source.index(
            "function viewCard"
        )
    ]
    contract = badge_source + """
function check(condition,message){if(!condition)throw new Error(message)}
for(const type of ['semantic','procedural','episodic','working']){
  const badge=memoryTypeBadge({memory_type:type});
  check(badge.includes('memory-type '+type),type+' color class');
  check(badge.includes('type-symbol'),type+' symbol');
}
check(memoryTypeBadge({memory_type:'unknown'}).includes('semantic'),'safe fallback');
"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_memory_cards_show_when_the_fact_happens():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert ".when-chip" in html
    # the "happened" date sits under the "said" date in the date column
    assert "${said.slice(0,10)}</span>${whenChip(m)}" in source

    chip_source = source[
        source.index("const WHEN_UNITS=") : source.index("function viewCard")
    ]
    contract = chip_source + """
function esc(value){return value}
function check(condition,message){if(!condition)throw new Error(message)}
check(occursText({})==='','no when, no chip');
check(whenChip({})==='','no when, no chip');
check(occursText({when:{start:'2026-10-03'},next_occurrence:'2026-10-03'})
  ==='happens 2026-10-03','a one-off still ahead');
check(occursText({when:{start:'2026-09-10'},next_occurrence:null})
  ==='happened 2026-09-10','a one-off that has passed');
check(occursText({when:{start:'2026-10-03',end:'2026-10-07'},next_occurrence:'2026-10-03'})
  ==='happens 2026-10-03 to 2026-10-07','a span');
check(occursText({when:{start:'--03-03',recurrence:'yearly'},next_occurrence:'2027-03-03'})
  ==='every year on 03-03, next 2027-03-03','a yearly recurrence');
check(occursText({when:{start:'2026-09-01',recurrence:'daily'},next_occurrence:'2026-09-20'})
  ==='every day, next 2026-09-20','a daily recurrence');
check(whenChip({when:{start:'2026-10-03'},next_occurrence:'2026-10-03'})
  .includes('when-chip'),'the chip carries its class');
"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_primary_dashboard_controls_have_tooltips_and_compact_add():
    html = _dashboard_html()

    assert 'class="knowledge-link"' in html
    assert 'title="What needs you, plus entities' in html
    assert 'class="account-links"' in html
    assert html.index('class="knowledge-link"') < html.index('class="account-links"')
    # export, import, about and sign out now live behind the account button,
    # where about still comes before the sign-out link.
    assert html.index('>about</button>') < html.index('class="account-links"')
    assert 'id="addbtn"' in html
    assert 'aria-label="Add a memory">+</button>' in html
    assert 'title="Show or hide the memory map."' in html


def test_map_click_does_not_expand_filter_panel():
    source = "\n".join(_scripts(_dashboard_html()))
    handler = source[
        source.index("async function applyMapNodeFilter") : source.index(
            "document.getElementById('map').addEventListener('click'"
        )
    ]

    assert "togglePanel('filters')" not in handler
    assert "await search()" in handler


def test_every_onclick_handler_is_defined(tmp_path):
    """An `onclick` naming a function that does not exist is a dead button.

    Removing a feature is the usual way this happens: the handler goes, the
    markup that calls it stays, and nothing fails until someone clicks.
    """
    html = _dashboard_html()
    source = "\n".join(_scripts(html))
    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", source))
    defined |= set(re.findall(r"(?:const|let|var)\s+(\w+)\s*=", source))

    called = set(re.findall(r'on(?:click|change|input)="(\w+)\(', html))
    called |= set(re.findall(r"on(?:click|change|input)='(\w+)\(", html))
    missing = sorted(name for name in called if name not in defined)
    assert not missing, f"markup calls handlers that no longer exist: {missing}"


def test_the_entities_page_can_declare_a_selected_entity_a_duplicate():
    """The map panel could say "this is a duplicate of...", the entities page
    could not, so a duplicate found while browsing the list had to be hunted
    down again on the map."""
    source = "\n".join(_scripts(_dashboard_html()))
    panel = source[source.index("async function openEntity(id)") : source.index("function relationsBlock(")]

    assert "toggleKnowledgeDuplicatePicker(this," in panel
    assert ">is duplicate of...</button>" in panel
    # folded until asked for, and nothing to combine until a target is picked
    assert 'id="knowledgeduplicatepicker" data-memories="${detail.memories.length}" hidden' in panel
    assert 'id="knowledgeduplicatebtn" disabled' in panel
    # the name reaches the merge from the page, not through a quoted attribute
    assert "mergeKnowledgeEntity(${JSON.stringify(id)})'" in panel


def test_duplicate_picker_on_the_entities_page_merges_this_into_the_chosen_one():
    source = "\n".join(_scripts(_dashboard_html()))
    esc_start = source.index("function esc(s)")
    esc_line = source[esc_start : source.index("\n", esc_start)]
    label_start = source.index("function typeLabel(")
    label_line = source[label_start : source.index("\n", label_start)]
    picker = source[
        source.index("function knowledgeEntityTargetOptions(") : source.index("async function addAlias(")
    ]
    contract = esc_line + "\n" + label_line + "\n" + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const entities=[
  {id:'self',name:'Jonas',entity_type:'person',memories:2},
  {id:'quiet',name:'Jonas',entity_type:'person',memories:1},
  {id:'busy',name:'Jonas',entity_type:'person',memories:9},
  {id:'gone',name:'Ada',entity_type:'person',memories:4,merged_into:'x'},
  {id:'tag',name:'<b>Ada</b>',entity_type:null,memories:0},
  {id:'topic',name:'work',entity_type:'topic',memories:3},
];
""" + picker + r"""
const html=knowledgeEntityTargetOptions(entities,'self');
const values=[...html.matchAll(/value="([^"]+)"/g)].map(m=>m[1]);
check(!values.includes('self'),'an entity cannot duplicate itself');
check(!values.includes('gone'),'an already-merged entity is not offered');
check(values.join()==='tag,busy,quiet,topic','same names sort busiest first: '+values.join());
check(html.includes('work · tag · 3 memories')&&!html.includes('topic ·'),'a tag reads "tag"');
check(html.includes('Jonas · person · 9 memories'),'type and count tell twins apart');
check(html.includes('1 memory<'),'singular');
check(html.includes('&lt;b&gt;Ada&lt;/b&gt; · untyped · 0 memories'),'names are escaped, no type reads untyped');

// mergeKnowledgeEntity: posts keep=target merge=this, then opens the target
let posted=null,opened=null,confirmed=true,alerted=null,reloaded=0;
const nodes={
  knowledgeduplicatetarget:{value:'busy',selectedOptions:[{dataset:{name:'Jonas',memories:'9'}}]},
  knowledgeentityname:{textContent:'Jonas'},
  knowledgeduplicatepicker:{dataset:{memories:'1'}},
};
const document={getElementById:id=>nodes[id]};
let asked=null;
const confirm=m=>{asked=m;return confirmed};
const alert=m=>alerted=m;
let reply={merged:true};
const api=async(path,opts)=>{posted={path,body:JSON.parse(opts.body)};return reply};
const loadEntities=async()=>reloaded++,loadStats=async()=>reloaded++,loadMapData=async()=>reloaded++,
  loadSearchFilters=async()=>reloaded++;
const openEntity=async id=>opened=id;
(async()=>{
  await mergeKnowledgeEntity('self');
  check(posted.path==='/api/v1/entities/merge','uses the merge endpoint');
  check(posted.body.keep_id==='busy'&&posted.body.merge_id==='self','this one folds into the chosen one');
  check(opened==='busy'&&reloaded===4,'lists and the About filter refresh and the result opens');
  check(asked==='Combine Jonas (1 memory) into Jonas (9 memories)? Memories and aliases will be preserved.',
        'the confirm says which Jonas is which: '+asked);

  posted=null;opened=null;confirmed=false;
  await mergeKnowledgeEntity('self');
  check(posted===null&&opened===null,'cancelling the confirm does nothing');

  confirmed=true;reply={error:'not found'};
  await mergeKnowledgeEntity('self');
  check(alerted==='not found'&&opened===null,'an error is shown and nothing opens');

  nodes.knowledgeduplicatetarget.value='';posted=null;
  await mergeKnowledgeEntity('self');
  check(posted===null,'nothing is sent before a target is picked');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    result = subprocess.run(["node", "-"], input=contract, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_free_text_passed_through_a_single_quoted_onclick_survives_an_apostrophe():
    """A tag like "mum's health" ended onclick='filterByTag("mum's health")'
    at the apostrophe, so its chip, rename, delete and merge buttons did
    nothing. jsArg escapes for the attribute; the browser decodes it back."""
    source = "\n".join(_scripts(_dashboard_html()))
    helpers = "\n".join(
        source[source.index(name) : source.index("\n", source.index(name))]
        for name in ("function esc(s)", "function jsArg(v)")
    )
    contract = helpers + r"""
function check(condition,message){if(!condition)throw new Error(message)}
// what the HTML parser does to an attribute value before the handler runs
const decode=s=>s.replace(/&#39;/g,"'").replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
for(const value of ["mum's health",'say "hi"','<b>&amp;</b>',"it's & <that>",
                    {canonical:"it's",variants:["it's","its"]},["O'Brien"]]){
  const out=jsArg(value);
  check(!out.includes("'"),'no raw apostrophe may reach the attribute: '+out);
  check(JSON.stringify(JSON.parse(decode(out)))===JSON.stringify(value),'round trip: '+out);
}
"""
    result = subprocess.run(["node", "-"], input=contract, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for call in ("filterByEntity(${jsArg(entity)})", "renameEntity(${jsArg(id)})",
                 "deleteTagEntity(${jsArg(id)})",
                 "toggleEntityType(${jsArg(group.type)})", "setEntityType(${jsArg(chip.type)})",
                 "pickEntity(${jsArg(id)},this.checked)", "openEntity(${jsArg(id)})"):
        assert call in source, call
    for unsafe in ("JSON.stringify(String(c))", "JSON.stringify(topic.category)",
                   "JSON.stringify(group)", "JSON.stringify(type)", "JSON.stringify(entity.name)"):
        assert unsafe not in source, unsafe


def test_the_entity_panel_shows_the_entity_clicked_last_not_the_one_answered_last():
    source = "\n".join(_scripts(_dashboard_html()))
    esc_start = source.index("function esc(s)")
    helpers = source[esc_start : source.index("\n", source.index("function jsArg(v)"))]
    panel = source[source.index("function placeBlock(") : source.index("function closeEntity(")]
    contract = helpers + "\n" + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const box={dataset:{},innerHTML:''};
const document={getElementById:id=>id==='entitydetail'?box:null};
const setKnowledgeOpen=()=>{},showKnowledge=()=>{};
const renameTitle=()=>'',DELETE_TAG_TITLE='',rememberTag=()=>{};
const pending={};
const api=path=>new Promise(resolve=>{pending[path.split('/').pop()]=resolve});
const reply=name=>({entity:{name,description:name+' facts'},aliases:[],memories:[],relations:[],relation_names:{},hub:true});
""" + panel + r"""
(async()=>{
  const first=openEntity('slow'),second=openEntity('fast');
  pending.fast(reply('Fast'));await second;
  pending.slow(reply('Slow'));await first;
  check(box.dataset.entityId==='fast','the last click is the open entity');
  check(box.innerHTML.includes('Fast')&&!box.innerHTML.includes('Slow'),'and it is what the panel shows');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    result = subprocess.run(["node", "-"], input=contract, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_rename_that_lands_in_another_entity_moves_the_panel_to_it():
    """Renaming a tag to a name merged away folds it into that name's
    survivor, and the rename answers with the survivor. The panels, the map
    selection and the About filter follow it to its id and its name, and
    each panel is drawn again for it (its buttons carry the id); the memory
    list and the About filter are read again, since a tag is renamed on its
    memories. A plain rename keeps the id and only updates the name."""
    source = "\n".join(_scripts(_dashboard_html()))
    rename = source[
        source.index("function syncEntityIdentity(") : source.index("async function addMapAlias(")
    ]
    contract = r"""
function check(condition,message){if(!condition)throw new Error(message)}
function option(value,selected,tag){
  const o={value,textContent:value,selected,dataset:{tag},remove(){options.splice(options.indexOf(o),1)}};
  return o;
}
const options=[option('tag-old',true,'tax')];
const nodes={
  mapentitydetail:{dataset:{entityId:'tag-old'}},
  entitydetail:{dataset:{entityId:'tag-old'}},
  mapentityname:{textContent:'tax'},mapentityidentity:{innerHTML:''},
  knowledgeentityname:{textContent:'tax'},knowledgeentityidentity:{innerHTML:''},
  'filter-about':{get options(){return options}},
};
const document={getElementById:id=>nodes[id]};
let activeMapKey='entity:tag-old',mapData={entities:[{entity_id:'tag-old',label:'tax'}]},G=null;
const knowledgeNames={},TAG_TYPE='topic',tagNames={'tag-old':'tax'};
const tagKey=entity=>String(entity.normalized||entity.name).toLowerCase();
const rememberTag=entity=>{if(entity.entity_type===TAG_TYPE)tagNames[entity.id]=tagKey(entity)};
const galaxyRead=()=>{},entityIdentityBlock=(entity,aliases)=>'identity of '+entity.name;
const shown=[],opened=[];let reloaded=[];
const showMapEntityDetail=id=>shown.push(id),openEntity=id=>opened.push(id);
const loadEntities=async()=>reloaded.push('entities'),loadMapData=async()=>reloaded.push('map');
const loadSearchFilters=async()=>reloaded.push('filters'),search=async()=>reloaded.push('search');
const alert=message=>{throw new Error(message)};
let answer='levies',reply=null,patched=null;
const prompt=()=>answer;
const api=async(path,opts)=>{patched=path;return reply};
""" + rename + r"""
(async()=>{
  reply={entity_id:'survivor',entity:{id:'survivor',name:'levies',normalized:'levies',entity_type:'topic'},aliases:[]};
  await renameEntity('tag-old');
  check(patched==='/api/v1/entities/tag-old','the rename is sent for the entity shown');
  check(nodes.mapentitydetail.dataset.entityId==='survivor','the map panel shows the survivor');
  check(nodes.entitydetail.dataset.entityId==='survivor','the entities panel shows the survivor');
  check(activeMapKey==='entity:survivor','the map selection follows it');
  check(shown.join()==='survivor'&&opened.join()==='survivor','both panels are drawn for it');
  check(options.length===1&&options[0].value==='survivor'&&options[0].selected,'the filter follows it');
  check(options[0].dataset.tag==='levies','and filters by the new tag name');
  check(tagNames.survivor==='levies','the tag is known under its new id');
  check(reloaded.includes('filters')&&reloaded.at(-1)==='search','the About filter and the memory list are read again');
  check(knowledgeNames.survivor==='levies','its name is known under its id');
  check(reloaded.includes('map')&&reloaded.includes('entities'),'the map and the list reload');

  // a plain rename keeps the id and draws nothing again
  shown.length=0;opened.length=0;reloaded=[];answer='duties';delete tagNames.survivor;
  reply={entity_id:'survivor',entity:{id:'survivor',name:'duties'},aliases:[]};
  mapData.entities[0].entity_id='survivor';mapData.entities[0].label='levies';
  await renameEntity('survivor');
  check(nodes.mapentitydetail.dataset.entityId==='survivor'&&activeMapKey==='entity:survivor','same id');
  check(nodes.mapentityname.textContent==='duties'&&nodes.knowledgeentityname.textContent==='duties',
        'the name is updated in place');
  check(shown.length===0&&opened.length===0&&reloaded.join()==='entities','nothing is drawn again');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    result = subprocess.run(["node", "-"], input=contract, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_rename_route_answers_with_the_entity_the_tag_went_into(tmp_path):
    """Renaming a tag to a name merged away answers with that name's
    survivor, and says its id."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        for content, tag in (("a", "tax"), ("b", "levies"), ("c", "duties")):
            store.add(content, user_id="default", infer=False, categories=[tag])
        store.merge_tags(["tax"], "levies", user_id="default")
        from memry.models import Scope

        scope = Scope(user_id="default")
        duties = store.backend.topic_entity("duties", scope, create=False)
        levies = store.backend.topic_entity("levies", scope, create=False)
        with TestClient(create_app(store)) as client:
            response = client.patch(f"/api/v1/entities/{duties.id}", json={"name": "tax"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["entity_id"] == levies.id == body["entity"]["id"]
    finally:
        store.close()


# ------------------------------------------------ one Entities view, one filter
def _lines(source: str, *starts: str) -> str:
    """The one line of the dashboard script that begins with each of ``starts``."""
    return "\n".join(
        source[source.index(start) : source.index("\n", source.index(start))]
        for start in starts
    )


def _region(source: str, start: str, end: str) -> str:
    """From ``start`` up to the first ``end`` after it."""
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def _run_node(contract: str, *args: str) -> str:
    result = subprocess.run(
        ["node", "-", *args], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_upkeep_has_one_entities_view_and_the_map_one_mode():
    """Tags are entities of the type "topic": they are listed, filtered and
    drawn with the people and things, not on a Tags tab or in a Tags mode."""
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    tabs = re.findall(r'<button id="ktab-(\w+)"[^>]*>([^<]+)</button>', html)
    assert tabs == [("maintenance", "Upkeep"), ("entities", "Entities"), ("forgotten", "Archive")]
    for gone in ('id="kpanel-topics"', 'id="taglist"', 'id="tagsearch"',
                 'id="mapTagsBtn"', 'id="filter-topic"', 'id="filter-entity"'):
        assert gone not in html, gone
    for gone in ("setMapMode", "mapMode", "renderTags", "loadTags", "Tags page",
                 "mergeTags", "into one named", "legacy_tags", "tag_edges",
                 "suggestMerges", "suggest-merges", "tag_split", "tag_health"):
        assert gone not in source, gone
    # the one combine path: keep one, merge the rest into it
    combine = _region(source, "async function combineSelected(", "function closeCombine(")
    assert "/api/v1/tags/edit" not in combine and "prompt(" not in combine
    assert "api('/api/v1/entities/merge'" in _region(source, "async function applyCombine(", "// Where this")
    panel = _region(html, '<section class="kpanel" id="kpanel-entities"', "</section>")
    for control in ('id="entsearch"', 'id="enttypes"', 'id="entsel"',
                    ">Combine selected...</button>", 'id="entitydetail"'):
        assert control in panel, control
    # merge proposals are asked under Upkeep, and Backfill types runs from there
    for gone in (">Merge proposals</h2>", 'id="proplist"', "Backfill types"):
        assert gone not in panel, gone
    # two tags that may be one are a merge proposal like any pair
    assert "Suggest merges" not in html and 'id="tagsuggest"' not in html
    # the map offers tags as a type, off until turned on
    assert 'title="Show every type except concept, other and tag.">defaults</button>' in html
    assert 'title="Show every type, tags included.">all</button>' in html
    # the memory list has one About filter
    assert html.count('<select id="filter-about" multiple') == 1
    # a tag reads "tag", never "topic"
    assert "function typeLabel(type){return type==='topic'?'tag':(type||'untyped')}" in source
    assert "or topic." not in html and "A topic is a tag" not in html
    # no dashes in the words this change wrote
    for text in (panel, _region(html, '<div class="search-filters"', "</div>"),
                 _region(html, '<div class="gx-ctrl">', '<div class="gx-read"')):
        assert "\u2014" not in text and "\u2013" not in text  # no em or en dash


def test_the_entities_list_holds_tags_beside_people_and_things_with_a_type_filter():
    source = "\n".join(_scripts(_dashboard_html()))
    contract = _lines(source, "function esc(s)", "function typeLabel(", "function jsArg(v)",
                      "const TAG_TYPE=") + r"""
let knowledgeNames={};
const nodes={};
const document={getElementById:id=>(nodes[id]??={innerHTML:'',textContent:'',value:'',dataset:{}})};
const concepts=Array.from({length:13},(_,i)=>({id:'c'+i,name:'idea '+String(i).padStart(2,'0'),
  entity_type:'concept',hub:true,memories:1}));
const rows=[
  {id:'ada',name:'Ada',entity_type:'person',hub:true,memories:4},
  {id:'widget',name:'Widget',entity_type:'product',hub:false,memories:1},
  {id:'t-travel',name:'travel',normalized:'travel',entity_type:'topic',hub:false,memories:3},
  {id:'t-life',name:'life',normalized:'life',entity_type:'topic',hub:false,memories:1},
  {id:'gone',name:'Old',entity_type:'person',hub:true,memories:1,merged_into:'ada'},
  ...concepts];
const reads={
  '/api/v1/entities?limit=100000&include_merged=true&kind=any':rows,
  '/api/v1/relations?limit=2000':[]};
const api=async path=>reads[path];
""" + _region(source, "// -- tags: deleted on every memory", "// Where this entity belongs") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const chips=view=>view.chips.map(chip=>chip.label+'='+chip.count).join();
let view=entityListView(rows);
check(chips(view)==='all=17,concept=13,person=1,product=1,tag=2','every live entity, tags included: '+chips(view));
check(view.groups.find(g=>g.type==='concept').rows.length===12,'a type is capped in the all view');
view=entityListView(rows,{type:'topic'});
check(view.groups.length===1&&view.groups[0].rows.map(e=>e.name).join()==='life,travel','the tag filter shows the tags');
check(entityListView(rows,{type:'concept'}).groups[0].rows.length===13,'one type shows its rows');
check(chips(entityListView(rows,{hubs:true}))==='all=16,concept=13,person=1,tag=2','hubs only keeps the tags');
check(chips(entityListView(rows,{needle:'TRA'}))==='all=1,tag=1','the name filter');
check(chips(entityListView(rows,{type:'event'})).endsWith('event=0'),'a type picked stays a chip');
(async()=>{
  await loadEntities();
  const list=nodes.entlist.innerHTML,types=nodes.enttypes.innerHTML;
  check(types.includes('>tag<span class="cnt">2</span>')&&!types.includes('topic<'),'the chip reads "tag"');
  check(list.includes('<div class="ent-group"><span>tag</span>'),'the tag group reads "tag"');
  check(!/>topic</.test(list),'never "topic"');
  // 12 of the 13 concepts, Ada, Widget and the two tags
  check((list.match(/type="checkbox"/g)||[]).length===16,'a checkbox a row: '+(list.match(/type="checkbox"/g)||[]).length);
  check(list.includes("deleteTagEntity(&quot;t-travel&quot;)"),'a tag row deletes');
  check(!list.includes("deleteTagEntity(&quot;ada&quot;)"),'a person row does not');
  check(list.includes("renameEntity(&quot;ada&quot;)")&&list.includes("renameEntity(&quot;t-travel&quot;)"),'every row renames');
  check(list.includes('life</button>')&&!list.includes('synthetic'),'a tag carries no synthetic badge');
  check(list.includes('show 1 more'),'the capped type offers the rest');
  check(nodes.entsel.textContent==='none selected','nothing checked yet');
  check(nodes.entcombinebtn.hidden===true,'no combine with nothing checked');
  pickEntity('t-travel',true);
  check(nodes.entcombinebtn.hidden===true,'no combine with one checked');
  pickEntity('ada',true);
  check(nodes.entsel.textContent==='2 selected','the checked are counted');
  check(nodes.entcombinebtn.hidden===false,'combine shows from the second check');
  pickEntity('ada',false);
  check(nodes.entcombinebtn.hidden===true,'and goes again below two');
  pickEntity('ada',true);
  setEntityType('topic');
  check(!nodes.entlist.innerHTML.includes('Ada<')&&nodes.entlist.innerHTML.includes('travel<'),'filtered to tags');
  check(nodes.entlist.innerHTML.includes('value="t-travel" checked'),'a check survives the filter');
  check(nodes.entcount.innerHTML.startsWith('17 names, 0 relations'),'the count line: '+nodes.entcount.innerHTML);
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract)


def test_a_tag_is_renamed_deleted_and_combined_from_the_entities_list():
    """What the Entities list sends for a tag, replayed on a real store: a
    rename renames it on its memories, a delete takes it off them (the
    memories stay), and Combine keeps the one picked and merges the rest into
    it (``/entities/merge``): two tags become the tag kept, and a tag combined
    with a person goes into the person, the only one offered to keep."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        for content, tags in (("Trip to Lisbon with the kids", ["travel", "family"]),
                              ("Flights to Oslo are booked", ["trips"]),
                              ("Pack light this time", ["packing"]),
                              ("Jonas runs every Saturday", ["running"])):
            added = store.add(content, user_id="default", infer=False, categories=tags)
        jonas = store.backend.insert_entity(
            Entity(name="Jonas", entity_type="person", user_id="default"))
        store.backend.add_mention(EntityMention(
            entity_id=jonas.id, memory_id=added.actions[0].memory_id, surface="Jonas"))
        reads_paths = ("/api/v1/entities?limit=100000&include_merged=true&kind=any",
                       "/api/v1/relations?limit=2000")
        with TestClient(create_app(store)) as client:
            html = client.get("/").text
            reads = {path: client.get(path).json() for path in reads_paths}
            ids = {row["name"]: row["id"] for row in reads[reads_paths[0]]}
            source = "\n".join(_scripts(html))
            contract = _lines(source, "function esc(s)", "function typeLabel(",
                              "function jsArg(v)", "const TAG_TYPE=") + r"""
let knowledgeNames={},activeMapKey=null,G=null,mapData=null;
const nodes={};
const document={getElementById:id=>(nodes[id]??={innerHTML:'',textContent:'',value:'',dataset:{},options:[]})};
const reads=JSON.parse(process.argv[2]),ids=JSON.parse(process.argv[3]);
const writes=[],asked=[];
const api=async(path,opts={})=>{
  const method=opts.method||'GET';
  if(method==='GET')return reads[path];
  const body=opts.body?JSON.parse(opts.body):null;
  writes.push({method,path,body});
  if(method==='PATCH'){const id=path.split('/').pop();
    return {entity_id:id,entity:{id,name:body.name,normalized:body.name,entity_type:'topic'},aliases:[]}}
  return {};
};
let answer='';
const prompt=(question,value)=>{asked.push(question+' ['+value+']');return answer};
const confirm=question=>{asked.push(question);return true};
const alert=message=>{throw new Error(message)};
const noop=async()=>{};
const loadSearchFilters=noop,loadMapData=noop,loadStats=noop,search=noop,openEntity=noop,
  closeEntity=()=>{},clearMapEntityDetail=()=>{},galaxyRead=()=>{},showMapEntityDetail=()=>{},
  entityIdentityBlock=()=>'';
""" + _region(source, "// -- tags: deleted on every memory", "// Where this entity belongs") \
                + _region(source, "function syncEntityIdentity(", "async function addMapAlias(") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
(async()=>{
  await loadEntities();
  answer='journeys';
  await renameEntity(ids.travel);
  await deleteTagEntity(ids.packing);
  pickEntity(ids.trips,true);pickEntity(ids.family,true);
  await combineSelected();
  let panel=nodes.entcombine.innerHTML;
  check(panel.includes('value="'+ids.trips+'"')&&panel.includes('value="'+ids.family+'"'),
        'of two tags either can be kept');
  document.getElementById('entcombinekeep').value=ids.trips;
  await applyCombine();
  check(nodes.entsel.textContent==='none selected','combined tags leave nothing checked');
  check(nodes.entcombinebtn.hidden===true,'and no combine button');
  pickEntity(ids.Jonas,true);pickEntity(ids.running,true);
  await combineSelected();
  panel=nodes.entcombine.innerHTML;
  check(!nodes.entcombine.hidden,'a person among them asks which one to keep');
  check(panel.includes('value="'+ids.Jonas+'"')&&!panel.includes('value="'+ids.running+'"'),
        'only the person can be kept: a tag goes into the person');
  document.getElementById('entcombinekeep').value=ids.Jonas;
  await applyCombine();
  console.log(JSON.stringify({writes,asked}));
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
            out = json.loads(_run_node(contract, json.dumps(reads), json.dumps(ids)))
            for write in out["writes"]:
                response = client.request(write["method"], write["path"], json=write["body"])
                assert response.status_code == 200, (write, response.text)
            tags = client.get("/api/v1/categories").json()
            listed = client.get("/api/v1/entities", params={"kind": "any", "limit": 100}).json()
            person = client.get(f"/api/v1/entities/{jonas.id}").json()
            memories = client.get("/api/v1/memories").json()
    finally:
        store.close()

    writes = [(w["method"], w["path"], w["body"]) for w in out["writes"]]
    assert writes[0] == ("PATCH", f"/api/v1/entities/{ids['travel']}", {"name": "journeys"})
    assert writes[1] == ("POST", "/api/v1/tags/edit", {"op": "delete", "tag": "packing"})
    assert writes[2] == ("POST", "/api/v1/entities/merge",
                         {"keep_id": ids["trips"], "merge_id": ids["family"]})
    assert writes[3] == ("POST", "/api/v1/entities/merge",
                         {"keep_id": ids["Jonas"], "merge_id": ids["running"]})
    assert len(writes) == 4
    assert out["asked"] == [
        'Rename tag "travel" on every memory to: [travel]',
        'Delete tag "packing" from all memories? The memories stay.',
        "Combine family into trips? Memories and names are kept.",
        "Combine running into Jonas? Memories and names are kept."]
    # what the store holds afterwards
    assert sorted((tag["category"], tag["count"]) for tag in tags) == [("journeys", 1), ("trips", 2)]
    assert sorted(row["name"] for row in listed) == ["Jonas", "journeys", "trips"]
    assert "running" in [alias.lower() for alias in person["aliases"]]
    assert len(person["memories"]) == 1
    assert len(memories) == 4, "no memory is lost"


def test_the_about_filter_maps_picks_onto_the_api_parameters():
    """A tag picked goes to ``categories``, anything else to ``entity_id``,
    and the two filters hold together: a tag with a person finds the
    memories that have both. A tag filter reaches the memories filed under
    that tag and no other."""
    from memry.models import Scope

    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        added = {}
        for content, tags in (("Ada's standup notes", ["work"]), ("Quarterly planning", ["work"]),
                              ("Sleep eight hours", ["life"]), ("Ada's running plan", ["health"])):
            added[content] = store.add(content, user_id="default", infer=False,
                                       categories=tags).actions[0].memory_id
        ada = store.backend.insert_entity(Entity(name="Ada", entity_type="person", user_id="default"))
        for content in ("Ada's standup notes", "Ada's running plan"):
            store.backend.add_mention(EntityMention(
                entity_id=ada.id, memory_id=added[content], surface="Ada"))
        scope = Scope(user_id="default")
        life = store.backend.topic_entity("life", scope, create=False)
        with TestClient(create_app(store)) as client:
            html = client.get("/").text

            def found(**body):
                rows = client.post("/api/v1/search", json={"query": "", **body}).json()
                return sorted(row["memory"]["content"] for row in rows)

            by_tag = found(categories=["life"])
            by_topic_id = found(entity_id=[life.id])
            both = found(categories=["work"], entity_id=[ada.id])
    finally:
        store.close()
    assert by_tag == by_topic_id == ["Sleep eight hours"]
    # a tag and a person together: the memories that have both
    assert both == ["Ada's standup notes"]

    source = "\n".join(_scripts(html))
    contract = _lines(source, "function esc(s)", "function typeLabel(", "const TAG_TYPE=",
                      "function tagKey(", "function rememberTag(") + r"""
const tagNames={};
class Option{constructor(text,value){this.textContent=text;this.value=value;this.dataset={};this.selected=false}}
const option=(value,tag)=>{const o=new Option(value,value);if(tag)o.dataset.tag=tag;return o};
const select={options:[option('ada'),option('t-work','work'),option('t-travel','travel')],
  get selectedOptions(){return this.options.filter(o=>o.selected)},
  add(o){this.options.push(o)},appendChild(o){this.options.push(o)},querySelector:()=>null,innerHTML:''};
const nodes={'filter-about':select,'filter-date':{value:''},'filter-date-to':{value:''},q:{value:''}};
const document={getElementById:id=>nodes[id]};
const panels={filters:true};
let activeMapKey=null,haveMore=false,searchActive=false,current=[],offset=0;
const PAGE=100,sent=[];
const api=async(path,opts={})=>{sent.push(opts.body?JSON.parse(opts.body):path);return []};
const render=()=>{},togglePanel=()=>{},toggleClear=()=>{},clearMapEntityDetail=()=>{},
  showMapEntityDetail=()=>{},galaxyRead=()=>{};
""" + _region(source, "function filterByEntity(entity){", "function toggleClear(){") \
        + _region(source, "// A click on a planet makes it", "document.getElementById('map').addEventListener('click'") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const pick=(...values)=>select.options.forEach(o=>{o.selected=values.includes(o.value)});
(async()=>{
  // the list: grouped by type, tags labelled "tag", only what a memory is about
  const html=aboutOptions([
    {id:'t-work',name:'work',normalized:'work',entity_type:'topic',memories:2},
    {id:'ada',name:'Ada',entity_type:'person',memories:2},
    {id:'lis',name:'Lisbon',entity_type:'place',memories:1},
    {id:'idle',name:'Idle',entity_type:'place',memories:0}],new Set(['ada']));
  const groups=[...html.matchAll(/<optgroup label="([^"]+)"/g)].map(m=>m[1]);
  check(groups.join()==='person,place,tag','groups: '+groups.join());
  check(html.includes('value="t-work" data-tag="work">work (2)'),'a tag option names its tag');
  check(html.includes('value="ada" selected>Ada (2)'),'a pick is kept');
  check(!html.includes('Idle'),'nothing no memory is about');
  // picks onto parameters
  const params=aboutParams([option('ada'),option('t-work','work'),option('tag:x','x')]);
  check(JSON.stringify(params)==='{"categories":["work","x"],"entities":["ada"]}',JSON.stringify(params));
  pick('ada','t-work');await search();
  check(JSON.stringify(sent.at(-1))==='{"query":"","limit":100,"categories":["work"],"entity_id":["ada"]}',
        'a tag and a person: '+JSON.stringify(sent.at(-1)));
  pick();
  // a map click fills the one filter: a tag by its name, a person by its id
  await applyMapNodeFilter({key:'entity:t-home',entity_id:'t-home',label:'home',entity_type:'topic'});
  check(JSON.stringify(sent.at(-1))==='{"query":"","limit":100,"categories":["home"]}','a tag planet: '+JSON.stringify(sent.at(-1)));
  check(select.selectedOptions.length===1&&select.selectedOptions[0].dataset.tag==='home','its option is added and picked');
  await applyMapNodeFilter({key:'entity:t-home',entity_id:'t-home',label:'home',entity_type:'topic'});
  check(select.selectedOptions.length===0&&activeMapKey===null,'a second click clears it');
  await applyMapNodeFilter({key:'entity:ada',entity_id:'ada',label:'Ada',entity_type:'person'});
  check(JSON.stringify(sent.at(-1))==='{"query":"","limit":100,"entity_id":["ada"]}','a person planet');
  check(select.selectedOptions.map(o=>o.value).join()==='ada','only the planet clicked is picked');
  // a chip on a memory card toggles its thing, and adds one the list lacks
  pick();filterByEntity({id:'t-travel',name:'travel',entity_type:'topic'});await Promise.resolve();
  check(select.selectedOptions.map(o=>o.value).join()==='t-travel','the listed tag is picked');
  filterByEntity({id:'t-packing',name:'packing',entity_type:'topic'});
  const added=select.options.find(o=>o.value==='t-packing');
  check(added&&added.selected&&added.dataset.tag==='packing','an unlisted tag gets an option');
  check(JSON.stringify(searchFilters().topics)==='["travel","packing"]','both are tag filters');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract)

def test_a_memory_card_shows_what_it_is_about_in_one_row_of_typed_chips():
    """A memory's tags are things it is about: the card shows them beside
    the people and things, in one row of chips, each typed as the Entities
    list types it ("person", "tag"), and no "#tag" row. A chip filters by its
    thing through the About filter (a tag as a tag filter, a person by id),
    and a second click takes it off again. The API keeps ``categories``."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        memory_id = store.add("Ada booked the flights to Lisbon", user_id="default",
                              infer=False, categories=["travel", "lisbon trip"]
                              ).actions[0].memory_id
        ada = store.backend.insert_entity(Entity(name="Ada", entity_type="person",
                                                 user_id="default"))
        store.backend.add_mention(EntityMention(entity_id=ada.id, memory_id=memory_id,
                                                surface="Ada"))
        with TestClient(create_app(store)) as client:
            html = client.get("/").text
            [memory] = client.get("/api/v1/memories").json()
    finally:
        store.close()
    assert memory["categories"] == ["travel", "lisbon trip"]
    assert [(link["name"], link["entity_type"]) for link in memory["entity_links"]] == [
        ("Ada", "person"), ("travel", "topic"), ("lisbon trip", "topic")]
    source = "\n".join(_scripts(html))
    assert "tagfilter" not in html and "filterByTag" not in source
    contract = _lines(source, "function esc(s)", "function typeLabel(", "function jsArg(v)",
                      "const TAG_TYPE=", "function tagKey(") + r"""
class Option{constructor(text,value){this.textContent=text;this.value=value;this.dataset={};this.selected=false}}
const select={options:[],get selectedOptions(){return this.options.filter(o=>o.selected)},
  add(o){this.options.push(o)}};
const nodes={'filter-about':select,'filter-date':{value:''},'filter-date-to':{value:''},q:{value:''}};
const document={getElementById:id=>nodes[id]};
const panels={filters:false};
let activeMapKey=null,editingId=null,opened=0;
const sent=[];
const togglePanel=()=>{opened++},toggleClear=()=>{},clearMapEntityDetail=()=>{};
const search=async()=>{sent.push(searchFilters())};
""" + _region(source, "function normalizedMemoryType", "function editCard(") \
        + _region(source, "function filterByEntity(entity){", "async function loadSearchFilters(") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const decode=s=>s.replace(/&#39;/g,"'").replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
const memory=JSON.parse(process.argv[2]);
const card=viewCard(memory);
check(!card.includes('#travel')&&!card.includes('tagfilter'),'no #tag row: '+card);
const chips=[...card.matchAll(/<button class="entity-chip"[^>]*onclick='filterByEntity\(([^)]*)\)'>([^<]*)<span class="chip-type">([^<]*)<\/span>/g)];
check(chips.map(m=>m[2].trim()+':'+m[3]).join()==='Ada:person,travel:tag,lisbon trip:tag',
      'one row of chips, typed: '+chips.map(m=>m[2]+':'+m[3]).join());
const clicked=chips.map(m=>JSON.parse(decode(m[1])));
(async()=>{
  filterByEntity(clicked[1]);
  check(opened===1,'the filters open to show the pick');
  filterByEntity(clicked[0]);
  check(JSON.stringify(sent.at(-1))==='{"since":"","until":"","topics":["travel"],"entities":["'+clicked[0].id+'"]}',
        'a tag chip filters by its tag, a person chip by its id: '+JSON.stringify(sent.at(-1)));
  filterByEntity(clicked[1]);
  check(JSON.stringify(sent.at(-1).topics)==='[]','a second click takes the tag off');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract, json.dumps(memory))


def test_a_tag_panel_renames_and_deletes_but_offers_no_alias():
    """A name lookup never finds a tag, so its panels, on the Entities tab and
    under the map, have no "add alias"; a person's keep it."""
    source = "\n".join(_scripts(_dashboard_html()))
    contract = _lines(source, "function esc(s)", "function typeLabel(", "function jsArg(v)",
                      "const TAG_TYPE=") + r"""
const nodes={};
const document={getElementById:id=>(nodes[id]??={innerHTML:'',dataset:{},hidden:true})};
const panels={map:true};let knowledgeMapSuspended=false,activeMapKey=null,mapData={entities:[]};
const setKnowledgeOpen=()=>{},showKnowledge=()=>{},rememberTag=()=>{};
const renameTitle=tag=>tag?'tag rename':'rename',DELETE_TAG_TITLE='delete this tag';
const replies={
  't1':{entity:{id:'t1',name:'travel',entity_type:'topic'},aliases:[],memories:[],relations:[],relation_names:{},hub:false},
  'p1':{entity:{id:'p1',name:'Ada',entity_type:'person'},aliases:[],memories:[],relations:[],relation_names:{},hub:true}};
const api=async path=>replies[path.split('/').pop()];
""" + _region(source, "let mapEntityDetailRequest=0;", "// A rename can answer") \
        + _region(source, "function placeBlock(", "function closeEntity(") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
(async()=>{
  await openEntity('t1');
  let html=nodes.entitydetail.innerHTML;
  check(html.includes('>rename</button>')&&html.includes('>delete tag</button>'),'a tag renames and deletes');
  check(!html.includes('add alias')&&!html.includes('not an entity'),'and nothing else');
  check(html.includes('<span class="syn">tag</span>'),'it reads "tag"');
  await openEntity('p1');
  html=nodes.entitydetail.innerHTML;
  check(html.includes('>add alias</button>')&&html.includes('>not an entity</button>'),'a person keeps both');
  activeMapKey='entity:t1';await showMapEntityDetail('t1');
  html=nodes.mapentitydetail.innerHTML;
  check(html.includes('>delete tag</button>')&&!html.includes('add alias'),'the map panel of a tag too');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract)


def test_every_element_the_script_looks_up_by_id_is_on_the_page():
    """A lookup of an id that is gone returns null, and the next line throws:
    the Merge proposals list moved to Upkeep, so nothing may still ask for it."""
    html = _dashboard_html()
    looked_up = set(re.findall(r"getElementById\('([\w-]+)'\)", html))
    assert looked_up, "the script should look elements up by id"
    missing = sorted(i for i in looked_up if f'id="{i}"' not in html)
    assert not missing, missing
    for gone in ("proplist", "decideProposal", "/api/v1/entities/proposals"):
        assert gone not in html, gone


def test_entity_merge_proposals_are_asked_under_upkeep():
    """The Entities tab no longer lists merge proposals; Upkeep, Needs you,
    Entities does, with merge and keep separate, from the same proposals."""
    store = MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        ada = store.backend.insert_entity(Entity(name="Ada", entity_type="person", user_id="default"))
        ada_l = store.backend.insert_entity(
            Entity(name="Ada L.", entity_type="person", user_id="default"))
        proposal = store.backend.add_proposal(MergeProposal(
            entity_a=ada.id, entity_b=ada_l.id, user_id="default", reason="similar names"))
        with TestClient(create_app(store)) as client:
            asked = client.get("/api/v1/entities/proposals?asked=true").json()
            queue = client.get("/api/v1/maintenance").json()["queue"]
    finally:
        store.close()
    assert [p["id"] for p in asked] == [proposal.id]
    rows = [item for item in queue if item["kind"] == "proposal"]
    assert [row["id"] for row in rows] == [proposal.id]
    assert rows[0]["title"] == "Ada and Ada L."
    assert (rows[0]["accept"], rows[0]["decline"]) == ("merge", "keep separate")
    source = "\n".join(_scripts(_dashboard_html()))
    assert "{kind:'proposal',label:'Entities'," in source


def test_combine_keeps_a_real_name_over_the_owner_placeholder():
    """Combining "the user" with Cosmin must not preselect "the user", even
    with more memories: a real name comes first, then the most memories, and
    the placeholder is still offered, last."""
    from memry.intelligence.extraction import OWNER_PLACEHOLDER

    source = "\n".join(_scripts(_dashboard_html()))
    assert f"const OWNER_PLACEHOLDER='{OWNER_PLACEHOLDER}';" in source
    contract = _lines(source, "function esc(s)", "function typeLabel(", "function jsArg(v)",
                      "const TAG_TYPE=") + r"""
let knowledgeNames={};
const nodes={};
const document={getElementById:id=>(nodes[id]??={innerHTML:'',textContent:'',value:'',dataset:{}})};
const api=async()=>[];
""" + _region(source, "// -- tags: deleted on every memory", "// Where this entity belongs") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const names=rows=>combineKeepOptions(rows).map(e=>e.name).join();
const owner={id:'u',name:'the user',entity_type:'person',memories:40};
const cosmin={id:'c',name:'Cosmin',entity_type:'person',memories:3};
const cos={id:'k',name:'Cos',entity_type:'person',memories:9};
const tag={id:'t',name:'work',entity_type:'topic',memories:50};
check(names([owner,cosmin])==='Cosmin,the user','a real name before the placeholder: '+names([owner,cosmin]));
check(names([{...owner,name:' The User '},cosmin])==='Cosmin, The User ','in any case');
check(names([owner,cosmin,cos])==='Cos,Cosmin,the user','then the most memories');
check(names([owner,tag])==='the user','the placeholder can still be kept');
check(names([tag,{...tag,id:'t2',name:'job',memories:60}])==='job,work','two tags: the most memories');
entityRows=[owner,cosmin];
pickEntity('u',true);pickEntity('c',true);
(async()=>{
  await combineSelected();
  const panel=nodes.entcombine.innerHTML;
  check(panel.indexOf('value="c"')>-1&&panel.indexOf('value="c"')<panel.indexOf('value="u"'),
        'Cosmin is the first option, so the one picked to keep');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract)


def test_backfill_types_runs_by_hand_from_upkeep_done_on_its_own():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))
    auto = _region(html, '<details id="upkeepauto"', "</details>")
    assert 'id="upkeeplist"' in auto
    contract = _lines(source, "function esc(s)") + "\n" + _region(
        source, "const PASS_WORDS={", "// The queue is an inbox") + r"""
const nodes={};
const document={getElementById:id=>(nodes[id]??={innerHTML:'',textContent:'',dataset:{},
  setAttribute(){},classList:{add(c){this.added=c}}})};
const writes=[];let reply={typed:2};
const api=async(path,opts={})=>{writes.push([opts.method||'GET',path]);return reply};
const noop=async()=>{};const loadEntities=noop,loadMapData=noop;
""" + _region(source, "function renderUpkeepPasses(", "async function togglePause(") \
        + _region(source, "async function backfillTypes(", "async function add(") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const info={passes:[],llm_available:true,decider_available:false,merge_gate:2,paused:false};
renderUpkeepPasses(info);
let list=nodes.upkeeplist.innerHTML;
check(list.includes('<b>Backfill types</b>')&&list.includes('onclick="backfillTypes(this)"'),'listed with a run button');
check(list.includes('title="Ask the language model for the type of every person or thing that has none yet.">run now</button>'),'the tooltip it had');
renderUpkeepPasses({...info,llm_available:false});
list=nodes.upkeeplist.innerHTML;
check(list.includes('needs a language model')&&!list.includes('backfillTypes('),'no model, no button');
(async()=>{
  const button={textContent:'run now',disabled:false};
  await backfillTypes(button);
  check(JSON.stringify(writes)==='[["POST","/api/v1/entities/backfill-types"]]','posts once: '+JSON.stringify(writes));
  check(nodes['passlog-backfill-types'].textContent.endsWith(' - typed 2 names'),'says what it did');
  check(button.textContent==='run now'&&!button.disabled,'the button is back');
  reply={typed:0,skipped:'no LLM configured'};
  await backfillTypes(button);
  check(nodes['passlog-backfill-types'].textContent.endsWith(' - not run: no LLM configured'),'says why not');
})().catch(e=>{console.error(e.message);process.exit(1)});
"""
    _run_node(contract)


def test_the_memory_list_is_a_table_sorted_by_a_header_click():
    """The list reads as a table: a header row, then a row per memory with the
    text, what it is about, its tags, the dates and the numbers in their own
    cells. A header click sorts the loaded rows (importance highest first, then
    lowest first, then back to the order they came in) and the choice sticks
    per browser. Chips still filter, edit and forget still work, and the
    namespace column shows only when more than one namespace is listed."""
    html = _dashboard_html()
    source = "\n".join(_scripts(html))
    assert ".mrow" in html and ".mhead" in html and "@media(max-width:56rem)" in html
    contract = _lines(source, "function esc(s)", "function typeLabel(", "function jsArg(v)",
                      "const TAG_TYPE=") + r"""
const store={memry_memory_sort:JSON.stringify({key:'nope',dir:1})};
const localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=String(v)},removeItem:k=>{delete store[k]}};
const list={innerHTML:'',vars:{},classList:{on:new Set(),add(c){this.on.add(c)},remove(c){this.on.delete(c)}},
  style:{setProperty(k,v){list.vars[k]=v}},insertAdjacentHTML(_,h){list.innerHTML+=h}};
const document={getElementById:id=>id==='list'?list:null};
let current=[],haveMore=false,searchActive=false,editingId=null;
const moreBar=()=>'',drawMap=()=>{};
""" + _region(source, "function render(items,appendFrom){", "// ---- galaxy map") + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const rows=()=>[...list.innerHTML.matchAll(/<div class="mrow">[\s\S]*?<div class="c-text"><div>([^<]*)<\/div>/g)].map(m=>m[1]);
const cell=(row,name)=>{const m=row.match(new RegExp('<div class="'+name+'"[^>]*>([\\s\\S]*?)</div>'));return m?m[1]:null};
const items=[
  {id:'a',content:'beta',importance:0.5,user_id:'default',created_at:'2026-09-02T10:00:00',
   entity_links:[{id:'e1',name:'Ada',entity_type:'person'},{id:'t1',name:'travel',entity_type:'topic'}]},
  {id:'b',content:'alpha',importance:0.9,user_id:'default',created_at:'2026-09-03T10:00:00',
   when:{start:'2026-09-10'},next_occurrence:null,entity_links:[]},
  {id:'c',content:'gamma',importance:0.2,user_id:'default',created_at:'2026-09-01T10:00:00',entity_links:[]}];
check(memSort===null,'a saved sort for an unknown column is ignored');
render(items);
check(list.classList.on.has('memtable'),'the list is a table');
check(list.innerHTML.startsWith('<div class="mhead">'),'a header row comes first');
for(const key of ['text','type','said','happened','imp'])
  check(list.innerHTML.includes(`onclick="sortMemories('${key}')"`),'a sortable header for '+key);
check(!list.innerHTML.includes("sortMemories('score')")&&!list.innerHTML.includes('c-score'),'no score column when browsing');
check(!list.innerHTML.includes('c-ns')&&!list.innerHTML.includes('@default'),'one namespace, no namespace column');
check(rows().join()==='beta,alpha,gamma','the order the list came in by default');
const first=list.innerHTML.split('<div class="mrow">')[1];
const unq=s=>s.replace(/&quot;/g,'"');
check(cell(first,'c-ent').includes('Ada')&&!cell(first,'c-ent').includes('travel'),'people and things in their own cell');
check(cell(first,'c-tags').includes('travel')&&!cell(first,'c-tags').includes('Ada'),'tags in their own cell');
check(/onclick='filterByEntity\([^)]*Ada/.test(unq(cell(first,'c-ent'))),'an entity chip still filters');
check(/onclick='filterByEntity\([^)]*travel/.test(unq(cell(first,'c-tags'))),'a tag chip still filters');
check(first.includes(`onclick="startEdit('a')"`)&&first.includes(`onclick="del('a')"`),'edit and forget stay on the row');
check(list.innerHTML.split('<div class="mrow">')[2].includes('happened 2026-09-10'),'the happened date shows in the date cell');
check(cell(first,'c-imp')==='0.50','importance has its own cell');
sortMemories('imp');
check(rows().join()==='alpha,beta,gamma','first click: highest importance first');
check(list.innerHTML.includes('imp ↓'),'the header shows the direction');
check(JSON.parse(store.memry_memory_sort).dir===-1,'the sort sticks per browser');
sortMemories('imp');
check(rows().join()==='gamma,beta,alpha','second click: lowest importance first');
check(list.innerHTML.includes('imp ↑'),'the header shows the reversed direction');
sortMemories('imp');
check(rows().join()==='beta,alpha,gamma'&&!('memry_memory_sort' in store),'third click: back to the order it came in');
sortMemories('text');
check(rows().join()==='alpha,beta,gamma','the text sorts A to Z');
sortMemories('happened');
check(rows()[0]==='alpha','a memory with no happened date goes last');
render(items.map((m,i)=>({...m,score:1-i/10,user_id:i?'default':'work'})));
check(list.innerHTML.includes("sortMemories('score')")&&list.innerHTML.includes('0.900'),'search results get a score column');
check(list.innerHTML.includes('@work')&&list.vars['--memcols'].split(' ').length===8,'two namespaces add a namespace column');
render([]);
check(!list.classList.on.has('memtable')&&list.innerHTML.includes('No memories yet.'),'an empty list is no table');
"""
    _run_node(contract)
