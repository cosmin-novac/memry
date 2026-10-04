"""Search filters: stated by the caller, applied before anything is ranked.

The bug these pin: an agent asked "Was habe ich am 01. April 2025 gemacht?",
a memory dated April 2025 existed, and the agent heard there was nothing.
get_memory_context took no filter at all, search_memories took dates only as
full days on two different parameters, and the ranking preferred memories of
April 2027 that shared more words. Agents now pass ``when`` and ``about`` (a
quoted phrase in the query is matched exactly); REST takes the finer set. A
filtered-out memory never appears, however well it matches, and an empty
result says what was asked and shows what lies nearest.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import date

import pytest
from mcp.types import CallToolResult
from starlette.testclient import TestClient

from memry.config import Config
from memry.filters import Filters, parse_period, quoted_phrases
from memry.mcp_server import create_server
from memry.models import Entity, EntityMention, Memory, Scope
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore, _Reads

APRIL = {"start": "2025-04-01", "end": "2025-04-30"}


def make_store() -> MemoryStore:
    return MemoryStore(Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64))


def remember(store, text, *, when=None, said="2026-01-01T10:00:00+00:00", user="u",
             tags=(), memory_type="semantic", entities=()):
    memory = store.backend.insert_memory(
        Memory(content=text, user_id=user, memory_type=memory_type,
               embedding_model=store.embedder.model_id, categories=list(tags),
               metadata={"when": when} if when else {}, created_at=said, updated_at=said),
        embedding=store.embedder.embed([text])[0])
    for entity in entities:
        store.backend.add_mention(EntityMention(entity_id=entity.id, memory_id=memory.id,
                                                surface=entity.name))
    return memory


def entity(store, name, entity_type="person", user="u"):
    return store.backend.insert_entity(Entity(
        name=name, normalized=name.lower(), entity_type=entity_type, user_id=user))


def ids(results) -> list[str]:
    return [getattr(r, "memory", r).id for r in results]


def call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    blocks = result.content if isinstance(result, CallToolResult) else (
        result[0] if isinstance(result, tuple) else result)
    text = blocks[0].text
    return json.loads(text) if text.startswith(("{", "[")) else text


@pytest.fixture
def store():
    s = make_store()
    yield s
    s.close()


# ------------------------------------------------------------------ parsing
def _parts(period):
    return period.start, period.end, period.precision


def test_periods_take_a_day_a_month_a_year_and_ranges():
    assert _parts(parse_period("2025-04-01", "when")) == (date(2025, 4, 1), date(2025, 4, 1), "day")
    assert _parts(parse_period("2025-04", "when")) == (date(2025, 4, 1), date(2025, 4, 30), "month")
    assert parse_period("2024-02", "when").end == date(2024, 2, 29)
    assert _parts(parse_period("2025", "when")) == (date(2025, 1, 1), date(2025, 12, 31), "year")
    whole = parse_period("2025-04..2025-06", "when")
    assert (whole.start, whole.end) == (date(2025, 4, 1), date(2025, 6, 30))
    assert parse_period("2025-04-01..2025-06-30", "when").end == date(2025, 6, 30)
    assert parse_period("2025-04..", "when").end is None
    assert parse_period("", "when") is None
    for bad in ("April 2025", "2025-13", "2025-06..2025-04", "..", "2025-02-30"):
        with pytest.raises(ValueError, match="is not a day"):
            parse_period(bad, "when")


def test_quoted_phrases_and_rejected_values():
    assert quoted_phrases('the "Blue Fig" dinner and „Café 50%“') == ("Blue Fig", "Café 50%")
    assert Filters.parse(query='was "x_y" said').contains == ("x_y",)
    with pytest.raises(ValueError, match="tag="):
        Filters.parse(entity_type="topic")
    with pytest.raises(ValueError, match="not one of"):
        Filters.parse(entity_type="dragon")
    with pytest.raises(ValueError, match="memory_type"):
        Filters.parse(memory_type="factual")
    assert not Filters.parse().active


# ------------------------------------------------------------------- periods
def test_a_day_matches_a_month_and_a_month_matches_a_day_inside_it(store):
    month = remember(store, "In April 2025 AI-Flow built a tool-calling stack", when=APRIL)
    day = remember(store, "Ada flew to Lisbon", when={"start": "2025-04-12"})
    remember(store, "Ada flew to Porto", when={"start": "2025-05-02"})
    undated = remember(store, "Ada likes tool-calling stacks")  # said 2026-01-01
    on_day = Filters.parse(when="2025-04-01")
    assert ids(store.get_all(user_id="u", filters=on_day)) == [month.id]
    in_month = store.get_all(user_id="u", filters=Filters.parse(when="2025-04"))
    assert set(ids(in_month)) == {month.id, day.id}
    # `when` falls back to the day said; `happened` never matches a memory without one
    assert ids(store.get_all(user_id="u", filters=Filters.parse(when="2026-01-01"))) == [undated.id]
    assert store.get_all(user_id="u", filters=Filters.parse(happened="2026")) == []
    assert undated.id not in ids(store.get_all(user_id="u", filters=Filters.parse(happened="2025")))


def test_said_reads_the_day_the_rows_show(store):
    old = remember(store, "first", said="2025-03-04T09:00:00+00:00")
    remember(store, "second", said="2025-06-01T09:00:00+00:00")
    assert ids(store.get_all(user_id="u", filters=Filters.parse(said="2025-03"))) == [old.id]


def test_browse_orders_by_the_time_asked_about_newest_first(store):
    early = remember(store, "a", when={"start": "2025-01-05"}, said="2026-03-01T00:00:00+00:00")
    late = remember(store, "b", when={"start": "2025-09-05"}, said="2025-01-01T00:00:00+00:00")
    assert ids(store.search("", user_id="u", filters=Filters.parse(when="2025"))) == [late.id, early.id]


# ---------------------------------------------------------------- pre-filter
def test_the_filter_applies_before_ranking_however_many_better_matches(store):
    """80 memories of April 2027 match the question better; the one of April
    2025 is still found, and none of 2027 is returned."""
    for i in range(80):
        remember(store, f"AI-Flow tool-calling stack tool-calling stack note {i}",
                 when={"start": "2027-04-01", "end": "2027-04-30"})
    answer = remember(store, "In April 2025 AI-Flow developed an agent stack", when=APRIL)
    found = store.search("AI-Flow tool-calling stack", user_id="u", limit=3,
                         filters=Filters.parse(when="2025-04-01"))
    assert ids(found) == [answer.id]


def test_linked_search_and_context_keep_to_the_filter(store):
    harlow = entity(store, "Harlow")
    kept = [remember(store, f"Harlow parked the van at work {i}", entities=[harlow],
                     when={"start": "2025-04-0" + str(i + 1)}) for i in range(3)]
    best = remember(store, "Where did Harlow park the car? Harlow parked the car here",
                    entities=[harlow], when={"start": "2027-04-01"})
    for i in range(10):
        remember(store, f"Harlow parked the bike {i}", entities=[harlow],
                 when={"start": "2027-05-01"})
    filters = Filters.parse(when="2025-04")
    reads = _Reads(Scope(user_id="u"), among=frozenset(m.id for m in kept))
    plan = store._plan("Where did Harlow park the car?", reads, True)
    assert plan.seeds == [harlow.id], "the linked search runs"
    assert set(ids(reads.entity_memories(store.backend, harlow.id, 50))) == set(ids(kept))
    found = store.search("Where did Harlow park the car?", user_id="u", limit=10,
                         filters=filters)
    assert found and set(ids(found)) <= set(ids(kept)) and best.id not in ids(found)
    ctx = store.reconstruct_context("Where did Harlow park the car?", user_id="u",
                                    filters=filters)
    assert set(ctx.memory_ids) <= set(ids(kept)) and "car here" not in ctx.text
    assert ctx.text.startswith("Filters applied: when=2025-04.")


# -------------------------------------------------------------- names, tags
def test_about_resolves_names_aliases_merges_and_tags(store):
    bochra = entity(store, "Bochra Saffar")
    store.add_entity_alias(bochra.id, "Bo")
    hers = remember(store, "Bochra built the eval harness", entities=[bochra])
    other = entity(store, "B. Saffar")
    merged = remember(store, "B. Saffar reviewed the paper", entities=[other])
    tagged = remember(store, "Bought milk", tags=["groceries"])
    remember(store, "Unrelated note")
    assert ids(store.get_all(user_id="u", filters=Filters.parse(about="bochra saffar"))) == [hers.id]
    assert ids(store.get_all(user_id="u", filters=Filters.parse(about="BO"))) == [hers.id]
    assert store.merge_entities(bochra.id, other.id)
    after = store.get_all(user_id="u", filters=Filters.parse(about="B. Saffar"))
    assert set(ids(after)) == {hers.id, merged.id}
    both = store.get_all(user_id="u", filters=Filters.parse(about="Groceries, Bo"))
    assert set(ids(both)) == {hers.id, merged.id, tagged.id}
    # REST's halves: entity names, and tags
    assert set(ids(store.get_all(user_id="u", filters=Filters.parse(entity="Bo")))) == {
        hers.id, merged.id}
    assert ids(store.get_all(user_id="u", filters=Filters.parse(tag="groceries"))) == [tagged.id]


def test_an_unknown_name_says_so_with_close_names(store):
    entity(store, "Bochra Saffar")
    found = store.search_filtered("eval", Filters.parse(about="Bochra Safar"), user_id="u")
    assert found.results == []
    assert 'named "Bochra Safar"' in found.note and "Bochra Saffar" in found.note
    partly = store.search_filtered("", Filters.parse(entity="Bochra Saffar, Zed"),
                                   user_id="u")
    assert 'named "Zed"' in partly.note and not partly.filters.matches_nothing


def test_names_resolve_only_in_the_callers_namespace(store):
    theirs = entity(store, "Mira", user="other")
    remember(store, "Mira moved to Oslo", user="other", entities=[theirs])
    found = store.search_filtered("Mira", Filters.parse(about="Mira"), user_id="u")
    assert found.results == [] and 'named "Mira"' in found.note
    assert "Close names" not in found.note, "another namespace's names are not suggested"
    mine = store.search_filtered("Mira", Filters.parse(about="Mira"), user_id="other")
    assert len(mine.results) == 1


def test_entity_type_contains_and_memory_type(store):
    flow = entity(store, "AI-Flow", "project")
    ada = entity(store, "Ada")
    project = remember(store, "AI-Flow started in 2025", entities=[flow], when={"start": "2025-02-01"})
    remember(store, "Ada started yoga in 2025", entities=[ada], when={"start": "2025-03-01"})
    odd = remember(store, 'Paid 50% of "Café_X" (100%) bill', memory_type="episodic")
    remember(store, "Paid 50 of CafeX bill", memory_type="procedural")
    started = store.get_all(user_id="u", filters=Filters.parse(entity_type="project", happened="2025"))
    assert ids(started) == [project.id]
    for phrase in ('"café_x" (100%)', "50% OF"):
        assert ids(store.get_all(user_id="u", filters=Filters.parse(contains=phrase))) == [odd.id]
    quoted = store.search('bill "Café_X"', user_id="u", filters=Filters.parse(query='bill "Café_X"'))
    assert ids(quoted) == [odd.id]
    assert ids(store.get_all(user_id="u", filters=Filters.parse(memory_type="episodic"))) == [odd.id]
    both = Filters.parse(memory_type="episodic", contains="cafex")
    assert store.get_all(user_id="u", filters=both) == []


# ----------------------------------------------------------- honest emptiness
def test_nothing_on_that_day_shows_the_nearest_dated(store):
    april = remember(store, "In April 2025 AI-Flow built a stack", when=APRIL)
    march = remember(store, "Ada visited Lisbon", when={"start": "2025-03-02"})
    later = remember(store, "Ada planned a trip", when={"start": "2027-04-01"})
    found = store.search_filtered("what did I do", Filters.parse(when="2025-05-10"), user_id="u")
    assert found.results == [] and "when=2025-05-10" in found.note
    assert [(m.id, days, side) for m, days, side in found.nearest] == [
        (april.id, 10, "before"), (march.id, 69, "before"), (later.id, 691, "after")]


# ----------------------------------------------------------------- MCP tools
def test_mcp_tools_expose_when_and_about_only():
    server = create_server(make_store(), manage_enrichment_worker=False)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    for name in ("search_memories", "get_memory_context", "list_memories"):
        props = set(tools[name].inputSchema["properties"])
        assert {"when", "about"} <= props, name
        assert not props & {"happened", "said", "entity", "entity_type", "tag", "contains",
                            "memory_type"}, name
    for name in ("search_memories", "get_memory_context"):
        assert 'when="2025-04-01"' in tools[name].description, name
    assert 'about="Bochra Saffar"' in tools["search_memories"].description
    assert "in double quotes" in tools["search_memories"].description


def test_mcp_marks_a_coarse_date_and_answers_empty_honestly(store):
    remember(store, "In April 2025 AI-Flow built a tool-calling stack", when=APRIL)
    remember(store, "In April 2027 AI-Flow plans a tool-calling stack rewrite",
             when={"start": "2027-04-01", "end": "2027-04-30"})
    server = create_server(store, manage_enrichment_worker=False)
    hit = call(server, "search_memories", {
        "query": "Was habe ich am 01. April 2025 gemacht?", "user_id": "u", "when": "2025-04-01"})
    assert hit["filters"] == {"when": "2025-04-01"}
    assert [row["happened"] for row in hit["memories"]] == ["happened 2025-04 (month)"]
    in_month = call(server, "search_memories", {"query": "stack", "user_id": "u", "when": "2025-04"})
    assert in_month["memories"][0]["happened"] == "happened 2025-04-01 to 2025-04-30"
    empty = call(server, "search_memories", {"query": "", "user_id": "u", "when": "2025-05-10"})
    assert empty["memories"] == [] and "No memory matches" in empty["note"]
    assert empty["nearest"][0]["side"] == "before" and empty["nearest"][0]["days"] == 10
    ctx = call(server, "get_memory_context", {"query": "", "user_id": "u", "when": "2025-05-10"})
    assert "Nearest memories by time" in ctx and "[happened 2025-04 (month)]" in ctx
    listed = call(server, "list_memories", {"user_id": "u", "about": "Nobody"})
    assert listed["memories"] == [] and 'named "Nobody"' in listed["note"]
    plain = call(server, "search_memories", {"query": "stack", "user_id": "u"})
    assert isinstance(plain, list) and len(plain) == 2


# ---------------------------------------------------------------------- REST
def test_rest_endpoints_take_the_full_filter_set(store):
    flow = entity(store, "AI-Flow", "project")
    april = remember(store, "In April 2025 AI-Flow built a stack", when=APRIL, entities=[flow],
                     memory_type="episodic", tags=["work"])
    remember(store, "In April 2027 AI-Flow plans a rewrite", entities=[flow],
             when={"start": "2027-04-01", "end": "2027-04-30"})
    with TestClient(create_app(store)) as client:
        body = {"query": "AI-Flow stack", "user_id": "u", "happened": "2025-04-01",
                "entity": "ai-flow", "entity_type": "project", "tag": "work",
                "contains": "built a", "memory_type": "episodic"}
        found = client.post("/api/v1/search", json=body).json()
        assert [r["memory"]["id"] for r in found["results"]] == [april.id]
        assert found["results"][0]["memory"]["happened"] == "happened 2025-04 (month)"
        assert found["filters"]["entity_type"] == "project"
        empty = client.post("/api/v1/search", json={"query": "", "user_id": "u",
                                                    "when": "2026"}).json()
        assert empty["results"] == [] and empty["nearest"]
        listed = client.get("/api/v1/memories?user_id=u&about=AI-Flow&said=2026-01").json()
        assert len(listed) == 2
        unknown = client.get("/api/v1/memories?user_id=u&entity=Nobody")
        assert unknown.status_code == 404 and "Nobody" in unknown.json()["error"]
        bad = client.post("/api/v1/search", json={"query": "x", "when": "April"})
        assert bad.status_code == 400 and "is not a day" in bad.json()["error"]
        ctx = client.post("/api/v1/context", json={"query": "stack", "user_id": "u",
                                                   "when": "2025-04-01"}).json()
        assert ctx["memory_ids"] == [april.id] and "2025-04 (month)" in ctx["text"]
        legacy = client.post("/api/v1/search", json={"query": "stack", "user_id": "u"}).json()
        assert isinstance(legacy, list)


# ----------------------------------------------------------- dashboard timeline
@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the dashboard JS")
def test_the_timeline_labels_a_whole_month_or_year_and_nothing_else(store):
    import re

    with TestClient(create_app(store)) as client:
        html = client.get("/").text
    source = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    pure = source[source.index("const MONTH_NAMES="): source.index("function timelineRepeat(")]
    contract = pure + """
function check(c,m){if(!c)throw new Error(m)}
const month={id:'m',when:{start:'2025-04-01',end:'2025-04-30'}};
const year={id:'y',when:{start:'2025-01-01',end:'2025-12-31'}};
const leap={id:'f',when:{start:'2024-02-01',end:'2024-02-29'}};
const day={id:'d',when:{start:'2025-04-01'}};
const timed={id:'t',when:{start:'2025-04-01T09:30'}};
const span={id:'s',when:{start:'2025-04-01',end:'2025-04-29'}};
const repeat={id:'r',when:{start:'2025-04-01',end:'2025-04-30',recurrence:'weekly'}};
check(timelineLabel(month,'2025-04-01')==='April 2025','a whole month: '+timelineLabel(month,'2025-04-01'));
check(timelineLabel(year,'2025-01-01')==='2025','a whole year');
check(timelineLabel(leap,'2024-02-01')==='February 2024','a leap February');
check(timelineLabel(day,'2025-04-01')==='2025-04-01','a day shows as before');
check(timelineLabel(timed,'2025-04-01T09:30')==='2025-04-01 09:30','a time shows as before');
check(timelineLabel(span,'2025-04-01')==='2025-04-01','a range short of the month');
check(timelineLabel(repeat,'2025-04-08')==='2025-04-08','a repeating one');
const placed=timelineEntries([month,day],'2026-01-01').filter(e=>e.kind==='row');
check(placed.every(e=>e.at==='2025-04-01'),'a month sits at its start');
"""
    result = subprocess.run(["node", "-"], input=contract, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
