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


def test_map_groups_memories_by_tags_or_entities_and_draws_type_shapes():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert 'id="mapTagsBtn"' in html
    assert 'id="mapEntitiesBtn"' in html
    assert 'aria-label="Memory type shapes"' in html
    assert "function drawMemoryMarker(ctx,type,x,y,size)" in source

    map_source = source[
        source.index("const hashCode=") : source.index("function drawMap(items){")
    ]
    contract = """
const window={};
const document={getElementById:()=>({})};
const localStorage={getItem:()=>null,setItem:()=>{}};
const matchMedia=()=>({matches:true});
const cats=m=>((m.categories&&m.categories.length)?m.categories:['(untagged)'])
  .map(c=>String(c).toLowerCase());
""" + map_source + """
function check(condition,message){if(!condition)throw new Error(message)}
const items=[
  {content:'Ada works on Helios',categories:['Work'],memory_type:'semantic',
   entity_links:[{id:'ada-1',name:'Ada',entity_type:'person'},
                 {id:'helios-1',name:'Helios',entity_type:'project'}]},
  {content:'Ask Ada before release',categories:['Work'],memory_type:'procedural',
   entity_links:[{id:'ada-1',name:'Ada',entity_type:'person'}]},
  {content:'Went hiking',categories:['Life'],memory_type:'episodic',entity_links:[]}
];
mapMode='tags';
const tags=buildGalaxy(items);
check(tags.total===3,'tag total');
check(tags.byKey['tag:work'].count===2,'tag count');
check(tags.byKey['tag:work'].memoryTypes.join(',')==='semantic,procedural','types');
mapMode='entities';
const entities=buildGalaxy(items);
check(entities.total===2,'linked memory total');
check(entities.byKey['entity:ada-1'].count===2,'entity count');
check(entities.byKey['entity:helios-1'].entityType==='project','entity type');
check(!entities.byKey['entity:undefined'],'fake entity');
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
