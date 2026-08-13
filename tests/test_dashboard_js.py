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
    assert 'id="mapEntitiesBtn"' in html
    assert 'id="mapEntityFilter"' in html
    assert 'aria-label="Memory type shapes"' in html
    assert "api('/api/v1/map')" in source
    assert "const MAX_IDLE_EDGES=400" in source
    assert "const displayedEdges=displayedGalaxyEdges(G,sel,hov)" in source
    assert "const satelliteFocus=sel||hov" in source
    assert "const showSatellites=satelliteFocus?focusedNeighbor:n.zone!=='rim'" in source
    assert "function drawMemoryMarker(ctx,type,x,y,size)" in source

    map_source = source[
        source.index("const hashCode=") : source.index("function drawMap(){")
    ]
    contract = """
const window={};
const document={getElementById:()=>({setAttribute:()=>{}})};
const localStorage={getItem:()=>null,setItem:()=>{}};
const matchMedia=()=>({matches:true});
let activeMapKey=null,hoverMapKey=null,hoverFocusTag=null;
const updateHover=()=>{};
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
check(displayedGalaxyEdges(tags,null,tags.byKey['tag:work']).length===430,'hover shows every node edge');
check(displayedGalaxyEdges(tags,tags.byKey['tag:work'],null).length===430,'selection shows every node edge');
mapMode='entities';
mapEntityTypes=null;
const defaultEntities=buildGalaxy(data);
check(defaultEntities.total===2,'linked memory total');
check(defaultEntities.byKey['entity:ada-1'].count===2,'entity count');
check(!defaultEntities.byKey['entity:rag-1'],'concept should default off');
mapEntityTypes.add('concept');
const allEntities=buildGalaxy(data);
check(allEntities.byKey['entity:rag-1'].entityType==='concept','concept opt-in');
"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

def test_primary_dashboard_controls_have_tooltips_and_compact_add():
    html = _dashboard_html()

    assert 'class="knowledge-link"' in html
    assert 'title="Open Knowledge' in html
    assert 'id="addbtn"' in html
    assert 'aria-label="Add a memory">+</button>' in html
    assert 'title="Show or hide the memory map."' in html


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
