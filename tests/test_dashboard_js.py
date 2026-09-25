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

import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient

from memry.config import Config
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

    assert 'id="mapTagsBtn"' in html
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
    assert "api('/api/v1/map')" in source
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

    map_source = source[
        source.index("const hashCode=") : source.index("function drawMap(){")
    ]
    contract = """
const window={};
const document={getElementById:()=>({setAttribute:()=>{},addEventListener:()=>{}})};
const localStorage={getItem:()=>null,setItem:()=>{}};
const matchMedia=()=>({matches:true});
let activeMapKey=null,hoverMapKey=null,hoverFocusTag=null,redraws=0;
const updateHover=()=>{};
const drawMap=()=>{redraws++};
const esc=value=>value;
""" + map_source + """
function check(condition,message){if(!condition)throw new Error(message)}
const tagEdges=Array.from({length:430},(_,index)=>({
  a:'tag:work',b:'tag:t'+index,weight:1
}));
const data={
  memories:432,entity_memories:2,
  tags:[
    {key:'tag:work',label:'work',kind:'tag',count:2,
     type_counts:{semantic:1,procedural:1}},
    ...Array.from({length:430},(_,index)=>({
      key:'tag:t'+index,label:'t'+index,kind:'tag',count:1,
      type_counts:{episodic:1}
    }))
  ],
  tag_edges:tagEdges,
  entities:[
    {key:'entity:ada-1',label:'Ada',kind:'entity',entity_id:'ada-1',
     entity_type:'person',count:2,type_counts:{semantic:1,procedural:1}},
    {key:'entity:rag-1',label:'RAG',kind:'entity',entity_id:'rag-1',
     entity_type:'concept',count:1,type_counts:{semantic:1}}
  ],
  entity_edges:[{a:'entity:ada-1',b:'entity:rag-1',weight:1}]
};
mapData=data;
mapMode='tags';
const tags=buildGalaxy(data);
check(tags.total===432,'tag total');
check(tags.byKey['tag:work'].count===2,'tag count');
check(tags.byKey['tag:work'].typeCounts.procedural===1,'type counts');
check(tags.idleEdges.length===400,'idle edge cap');
check(tags.lod===true,'431 planets is above the detail threshold');
check(tags.byKey['tag:work'].satTypes.length===2,'orbit marker types are precomputed');
const hoverEdges=displayedGalaxyEdges(tags,null,tags.byKey['tag:work']);
check(hoverEdges.length===430,'hover shows every node edge');
check(tags.idleEdges.every(edge=>hoverEdges.includes(edge)),'hover preserves every idle edge');
check(displayedGalaxyEdges(tags,tags.byKey['tag:work'],null).length===430,'selection shows every node edge');
mapMode='entities';
mapEntityTypes=null;
const defaultEntities=buildGalaxy(data);
check(defaultEntities.total===2,'linked memory total');
check(defaultEntities.lod===false,'small graphs keep full detail');
check(defaultEntities.byKey['entity:ada-1'].count===2,'entity count');
check(!defaultEntities.byKey['entity:rag-1'],'concept should default off');
mapEntityTypes.add('concept');
const allEntities=buildGalaxy(data);
check(allEntities.byKey['entity:rag-1'].entityType==='concept','concept opt-in');
handleMapEntityTypeChange({target:{
  matches:selector=>selector==='input[data-entity-type]',
  dataset:{entityType:'concept'},checked:false
}});
check(!mapEntityTypes.has('concept'),'checkbox updates selected entity types');
check(redraws===1,'checkbox redraws map immediately');
// A long tail: the twos leave the over-packed belt for the rim.
const crowd=(counts,edges=[])=>({memories:1,tags:counts.map((count,index)=>({
  key:'tag:c'+index,label:'c'+index,kind:'tag',count,type_counts:{semantic:count}
})),tag_edges:edges,entities:[],entity_edges:[]});
const zones=graph=>graph.nodes.reduce((seen,node)=>{
  (seen[node.zone]??=new Set()).add(node.count);return seen},{});
mapMode='tags';
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
  {a:'tag:c11',b:'tag:c0',weight:1},{a:'tag:c11',b:'tag:c62',weight:1},
  {a:'tag:c11',b:'tag:c63',weight:1}]));
check(linked.byKey['tag:c11'].zone==='belt','a well-linked two keeps its place');
check(linked.byKey['tag:c10'].zone==='rim','an unlinked two goes out to the rim');"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

def test_knowledge_modal_releases_and_restores_the_map():
    source = "\n".join(_scripts(_dashboard_html()))
    modal_source = source[
        source.index("let knowledgeTab=") : source.index("function openAbout(){")
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
        "async function decideProposal", 1,
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
            "loadRetiredEntities()}") in source
    assert "async function loadRetiredEntities()" in source
    assert "api('/api/v1/entities/retired')" in source
    assert "async function restoreEntity(id)" in source
    assert "api('/api/v1/entities/restore'" in source


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
    assert "${memoryTypeBadge(m)}${whenChip(m)}" in source

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
    picker = source[
        source.index("function knowledgeEntityTargetOptions(") : source.index("const ENTITY_ROW_CAP")
    ]
    contract = esc_line + "\n" + r"""
function check(condition,message){if(!condition)throw new Error(message)}
const entities=[
  {id:'self',name:'Jonas',entity_type:'person',memories:2},
  {id:'quiet',name:'Jonas',entity_type:'person',memories:1},
  {id:'busy',name:'Jonas',entity_type:'person',memories:9},
  {id:'gone',name:'Ada',entity_type:'person',memories:4,merged_into:'x'},
  {id:'tag',name:'<b>Ada</b>',entity_type:null,memories:0},
];
""" + picker + r"""
const html=knowledgeEntityTargetOptions(entities,'self');
const values=[...html.matchAll(/value="([^"]+)"/g)].map(m=>m[1]);
check(!values.includes('self'),'an entity cannot duplicate itself');
check(!values.includes('gone'),'an already-merged entity is not offered');
check(values.join()==='tag,busy,quiet','same names sort busiest first: '+values.join());
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
const loadEntities=async()=>reloaded++,loadStats=async()=>reloaded++,loadMapData=async()=>reloaded++;
const openEntity=async id=>opened=id;
(async()=>{
  await mergeKnowledgeEntity('self');
  check(posted.path==='/api/v1/entities/merge','uses the merge endpoint');
  check(posted.body.keep_id==='busy'&&posted.body.merge_id==='self','this one folds into the chosen one');
  check(opened==='busy'&&reloaded===3,'lists refresh and the result opens');
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
    for call in ("filterByTag(${jsArg(String(c))})", "renameTag(${jsArg(topic.category)})",
                 "deleteTag(${jsArg(topic.category)})", "applyMerge(${jsArg(group)},${index})",
                 "toggleEntityType(${jsArg(type)})"):
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
