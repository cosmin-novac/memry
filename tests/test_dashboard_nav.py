"""The dashboard header: two buttons, one account menu, one shared wording.

The header is hand-written HTML inside one Python string, so the order of the
controls and the place of the moved links are easy to disturb by accident.
These pin what a person sees: Upkeep, then Timeline, then the account button
with export, import, about and sign out under it.

The four memory-type sentences come from one JS map. Pinning that each one
appears exactly once is what stops a second copy being pasted into the map
legend, where it would quietly drift from the badge wording.
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
    shutil.which("node") is None, reason="node is needed to run the dashboard JS"
)

TYPE_SENTENCES = (
    "A fact that holds for a while: who someone is, what you prefer, "
    "how something works.",
    "Something tied to a particular time: a meeting, a decision, an incident.",
    "How to do something, or a rule to follow: steps, conventions, instructions.",
    "A short-lived note for a task in progress. It fades fastest of the four.",
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


def _menu(html: str) -> str:
    match = re.search(r'<div class="menu" id="usermenu"[^>]*>(.*?)</div>', html, re.S)
    assert match, "the account menu container should be in the header"
    return match.group(1)


def test_header_shows_upkeep_then_timeline_then_the_account_button():
    html = _dashboard_html()

    upkeep = html.index(">Upkeep<span class=\"badge\"")
    timeline = html.index(">Timeline</a>")
    account = html.index('id="usermenubtn"')
    assert upkeep < timeline < account
    # Timeline is the same pill as Upkeep, not a plain link.
    assert '<a class="knowledge-link" href="#" onclick="openTimeline()' in html


def test_the_moved_links_live_in_the_account_menu_only():
    html = _dashboard_html()
    menu = _menu(html)
    header = re.search(r"<h1>.*?</h1>", html, re.S).group(0)
    outside = header.replace(menu, "")

    for label in (">export</button>", ">import</button>", ">about</button>"):
        assert label in menu
        assert label not in outside
    assert 'href="/logout"' in menu
    assert 'href="/logout"' not in outside
    assert ">sign out</a>" in menu
    # The tooltips came along with the links.
    assert "Download a lossless Memry backup" in menu
    assert "Restore a lossless Memry backup exactly." in menu
    assert "What Memry does with what you tell it, in plain words." in menu
    assert "Sign out of this Memry dashboard." in menu
    # Import still goes through the hidden file input.
    assert 'id="importbtn" onclick="chooseImportFile()"' in menu
    assert 'id="importfile"' in html


def test_the_account_button_is_a_keyboard_reachable_menu_button():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    button = re.search(r"<button class=\"menubtn\"[^>]*>", html).group(0)
    assert 'aria-haspopup="menu"' in button
    assert 'aria-expanded="false"' in button
    assert "function toggleUserMenu()" in source
    assert "function closeUserMenu()" in source
    # Escape and a click outside both close it.
    assert "if(userMenuOpen()){closeUserMenu();return}" in source
    assert "if(event.target.closest&&event.target.closest('.menuwrap'))return;" in source
    # With no account the button says what it is and sign out goes away.
    assert "button.textContent=name?'@'+name:'menu';" in source
    assert "if(!name&&links)links.hidden=true;" in source


def test_each_memory_type_sentence_is_written_once():
    html = _dashboard_html()

    for sentence in TYPE_SENTENCES:
        assert html.count(sentence) == 1, f"{sentence!r} should live in one map"
    assert 'title="${MEMORY_TYPE_HELP[type]}"' in html
    # The legend takes its wording from the same map rather than its own copy.
    assert 'data-memory-type="semantic"' in html
    assert "function labelMemoryTypes()" in html


def test_the_timeline_orders_groups_and_marks_today():
    html = _dashboard_html()
    source = "\n".join(_scripts(html))

    assert 'id="timemodal"' in html
    assert ">Timeline</h2>" in html
    assert "No memories carry a time yet." in source
    assert "when_since=1900-01-01" in source
    assert "openTimelineMemory(" in source
    assert "function openTimelineMemory(id){closeTimeline();showMemory(id)}" in source

    pure = source[
        source.index("const MONTH_NAMES=") : source.index("function timelineRepeat(")
    ]
    contract = pure + """
function check(condition,message){if(!condition)throw new Error(message)}
const rows=[
  {id:'past',content:'a',memory_type:'episodic',
   when:{start:'2026-08-05'},next_occurrence:null},
  {id:'yearly',content:'b',memory_type:'semantic',
   when:{start:'--03-03',recurrence:'yearly'},next_occurrence:'2027-03-03'},
  {id:'soon',content:'c',memory_type:'episodic',
   when:{start:'2026-09-25T09:30'},next_occurrence:'2026-09-25T09:30'},
  {id:'stopped',content:'d',
   when:{start:'--12-01',recurrence:'yearly'},next_occurrence:null},
  {id:'undated',content:'e'}
];
const entries=timelineEntries(rows,'2026-09-20');
const placed=entries.filter(entry=>entry.kind==='row');
check(placed.map(entry=>entry.memory.id).join(',')==='past,soon,yearly',
  'ascending, and only memories that can be placed');
check(placed.find(entry=>entry.memory.id==='yearly').at==='2027-03-03',
  'a recurring memory sits at its next occurrence');
const shape=entries.map(entry=>entry.kind+':'
  +(entry.label||entry.at||'')).join('|');
check(shape==='month:August 2026|row:2026-08-05|month:September 2026'
  +'|today:2026-09-20|row:2026-09-25T09:30|month:March 2027|row:2027-03-03',
  'month headings, the today line and the rows in order, got '+shape);
const allPast=timelineEntries([rows[0]],'2026-09-20');
check(allPast[allPast.length-1].kind==='today','a past-only timeline ends at today');
const allAhead=timelineEntries([rows[2]],'2026-09-20');
check(allAhead[0].kind==='month'&&allAhead[1].kind==='today',
  'a future-only timeline starts at today');
check(timelineEntries([],'2026-09-20').every(entry=>entry.kind!=='row'),
  'nothing dated, nothing placed');
"""
    result = subprocess.run(
        ["node", "-"], input=contract, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
