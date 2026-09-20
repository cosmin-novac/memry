"""Occurrence time: when the thing a memory describes happens.

Covers the pure functions (formats, recurrence, windows, wording), the batched
read used by the backfill, and the paths that carry a "when" into the store and
back out through the store, REST and MCP filters.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from starlette.testclient import TestClient

from memry.config import Config
from memry.models import Memory
from memry.intelligence.when import (
    describe_when,
    extract_when,
    next_occurrence,
    overlaps,
    parse_when,
)
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.rest import create_app
from memry.store import MemoryStore

from conftest import FakeLLM, mcp_call

TODAY = date(2026, 9, 20)


# ---------------------------------------------------------------- parse_when


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-10-03", {"start": "2026-10-03"}),
        ({"start": "2026-10-03"}, {"start": "2026-10-03"}),
        ({"start": "2026-10-03T19:00"}, {"start": "2026-10-03T19:00"}),
        ({"start": "2026-10-03 19:00:30"}, {"start": "2026-10-03T19:00"}),
        ({"start": "2026-1-3"}, {"start": "2026-01-03"}),
        ({"start": "--03-03"}, {"start": "--03-03", "recurrence": "yearly"}),
        ({"start": "XXXX-03-03"}, {"start": "--03-03", "recurrence": "yearly"}),
        (
            {"start": "2026-10-03", "end": "2026-10-07"},
            {"start": "2026-10-03", "end": "2026-10-07"},
        ),
        (
            {"start": "2026-10-03", "recurrence": "WEEKLY"},
            {"start": "2026-10-03", "recurrence": "weekly"},
        ),
    ],
)
def test_parse_when_normalises(raw, expected):
    assert parse_when(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "soon",
        "next tuesday",
        42,
        {"start": None},
        {"start": "2026-02-30"},          # no such day
        {"start": "2026-13-01"},          # no such month
        {"start": "2026-10-03T25:00"},    # no such hour
        {"start": "--02-30"},             # no such day in any February
        {"start": "2026-10-07", "end": "2026-10-03"},   # end before start
        {"start": "--03-05", "end": "--03-01"},
    ],
)
def test_parse_when_rejects_garbage(raw):
    assert parse_when(raw) is None


def test_parse_when_drops_an_unusable_end_and_recurrence():
    # A yearless start with a dated end cannot be compared, so the end goes and
    # the rest survives; an unknown recurrence word is dropped the same way.
    assert parse_when({"start": "--03-03", "end": "2026-03-04"}) == {
        "start": "--03-03", "recurrence": "yearly",
    }
    assert parse_when({"start": "2026-10-03", "recurrence": "fortnightly"}) == {
        "start": "2026-10-03",
    }


def test_an_end_on_the_start_day_is_kept():
    assert parse_when({"start": "2026-10-03T19:00", "end": "2026-10-03"}) == {
        "start": "2026-10-03T19:00", "end": "2026-10-03",
    }


# ----------------------------------------------------------- next_occurrence


def test_one_off_in_the_future_and_in_the_past():
    assert next_occurrence({"start": "2026-10-03"}, TODAY) == date(2026, 10, 3)
    assert next_occurrence({"start": "2026-09-20"}, TODAY) == TODAY
    assert next_occurrence({"start": "2026-09-10"}, TODAY) is None


def test_yearly_recurrence_steps_to_the_next_turn():
    when = {"start": "--03-03", "recurrence": "yearly"}
    assert next_occurrence(when, TODAY) == date(2027, 3, 3)
    assert next_occurrence(when, date(2026, 3, 3)) == date(2026, 3, 3)
    assert next_occurrence(when, date(2026, 3, 2)) == date(2026, 3, 3)


def test_february_29_lands_on_the_28th_in_a_common_year():
    when = {"start": "--02-29", "recurrence": "yearly"}
    assert next_occurrence(when, date(2026, 1, 1)) == date(2026, 2, 28)
    assert next_occurrence(when, date(2024, 1, 1)) == date(2024, 2, 29)
    # past the clamped day, the next turn is the following year
    assert next_occurrence(when, date(2026, 3, 1)) == date(2027, 2, 28)


def test_monthly_recurrence_is_month_end_safe():
    when = {"start": "2026-01-31", "recurrence": "monthly"}
    assert next_occurrence(when, date(2026, 2, 1)) == date(2026, 2, 28)
    assert next_occurrence(when, date(2026, 3, 1)) == date(2026, 3, 31)
    assert next_occurrence(when, date(2026, 4, 1)) == date(2026, 4, 30)
    assert next_occurrence(when, date(2026, 12, 31)) == date(2026, 12, 31)
    assert next_occurrence(when, date(2027, 1, 1)) == date(2027, 1, 31)


def test_weekly_and_daily_recurrence():
    weekly = {"start": "2026-09-15", "recurrence": "weekly"}  # a Tuesday
    assert next_occurrence(weekly, TODAY) == date(2026, 9, 22)
    assert next_occurrence(weekly, date(2026, 9, 22)) == date(2026, 9, 22)
    daily = {"start": "2026-09-01", "recurrence": "daily"}
    assert next_occurrence(daily, TODAY) == TODAY


def test_a_recurrence_never_starts_before_its_start():
    when = {"start": "2027-05-01", "recurrence": "yearly"}
    assert next_occurrence(when, TODAY) == date(2027, 5, 1)


def test_a_recurrence_with_a_last_day_runs_out():
    when = {"start": "2026-01-05", "end": "2026-06-05", "recurrence": "monthly"}
    assert next_occurrence(when, date(2026, 3, 1)) == date(2026, 3, 5)
    assert next_occurrence(when, TODAY) is None


def test_next_occurrence_of_garbage_is_nothing():
    assert next_occurrence({"start": "whenever"}, TODAY) is None
    assert next_occurrence(None, TODAY) is None


# ------------------------------------------------------------------ overlaps


def test_one_off_overlaps_a_window():
    when = {"start": "2026-09-26"}
    assert overlaps(when, "2026-09-25", "2026-09-27") is True
    assert overlaps(when, "2026-09-26", "2026-09-26") is True   # inclusive
    assert overlaps(when, "2026-09-27", "2026-09-30") is False
    assert overlaps(when, None, "2026-09-30") is True
    assert overlaps(when, "2026-10-01", None) is False


def test_a_span_counts_where_it_reaches():
    trip = {"start": "2026-09-01", "end": "2026-09-30"}
    assert overlaps(trip, "2026-09-26", "2026-09-27") is True
    assert overlaps(trip, "2026-08-01", "2026-08-31") is False


def test_a_recurrence_matches_any_of_its_occurrences():
    birthday = {"start": "--03-03", "recurrence": "yearly"}
    assert overlaps(birthday, "2027-03-01", "2027-03-05") is True
    assert overlaps(birthday, "2027-04-01", "2027-04-05") is False
    weekly = {"start": "2026-09-15", "recurrence": "weekly"}
    assert overlaps(weekly, "2026-09-26", "2026-09-27") is False  # Sat/Sun
    assert overlaps(weekly, "2026-09-21", "2026-09-27") is True   # the Tuesday


def test_a_recurring_span_covers_the_days_in_between():
    holiday = {"start": "--12-24", "end": "--12-26", "recurrence": "yearly"}
    assert overlaps(holiday, "2026-12-25", "2026-12-25") is True


def test_a_memory_without_a_when_never_matches():
    assert overlaps(None, "2026-09-01", "2026-09-30") is False
    assert overlaps({}, "2026-09-01", None) is False


def test_an_unreadable_bound_matches_nothing():
    assert overlaps({"start": "2026-09-26"}, "this weekend", None) is False


# -------------------------------------------------------------- describe_when


def test_describe_when_reads_as_a_short_phrase():
    assert describe_when({"start": "2026-09-10"}, TODAY) == "happened 2026-09-10"
    assert describe_when({"start": "2026-10-03"}, TODAY) == "happens 2026-10-03"
    assert describe_when({"start": "2026-10-03T19:00"}, TODAY) == "happens 2026-10-03 19:00"
    assert describe_when(
        {"start": "2026-10-03", "end": "2026-10-07"}, TODAY
    ) == "happens 2026-10-03 to 2026-10-07"
    assert describe_when(
        {"start": "--03-03", "recurrence": "yearly"}, TODAY
    ) == "every year on 03-03, next 2027-03-03"
    assert describe_when(
        {"start": "2026-01-31", "recurrence": "monthly"}, TODAY
    ) == "every month on 01-31, next 2026-09-30"
    assert describe_when(
        {"start": "2026-09-15", "recurrence": "weekly"}, TODAY
    ) == "every week on Tuesday, next 2026-09-22"
    assert describe_when(
        {"start": "2026-09-01", "recurrence": "daily"}, TODAY
    ) == "every day, next 2026-09-20"
    assert describe_when(None, TODAY) == ""


def test_a_recurrence_that_has_run_out_drops_the_next_part():
    when = {"start": "2026-01-05", "end": "2026-06-05", "recurrence": "monthly"}
    assert describe_when(when, TODAY) == "every month on 01-05"


# ---------------------------------------------------------------- extract_when


def _when_response(*rows: dict) -> str:
    return json.dumps({"items": list(rows)})


def test_extract_when_reads_a_batch_in_one_call():
    llm = FakeLLM([
        _when_response(
            {"index": 0, "start": "2026-10-03", "end": None, "recurrence": None},
            {"index": 1, "start": None, "end": None, "recurrence": None},
            {"index": 2, "start": "--03-03", "end": None, "recurrence": "yearly"},
        )
    ])
    found = extract_when(llm, [
        {"content": "Team offsite in Lisbon on 2026-10-03", "recorded_at": "2026-09-01"},
        {"content": "The EUR price on 2026-09-01 was 41.20", "recorded_at": "2026-09-01"},
        {"content": "Ada's birthday is on 3 March", "recorded_at": "2026-09-01"},
    ])
    assert len(llm.calls) == 1
    assert found == [
        {"start": "2026-10-03"},
        None,
        {"start": "--03-03", "recurrence": "yearly"},
    ]


def test_extract_when_shows_the_model_the_recorded_date():
    llm = FakeLLM([_when_response()])
    extract_when(llm, [{"content": "Signed the lease yesterday", "recorded_at": "2026-09-19T08:00:00+00:00"}])
    _system, user = llm.calls[0]
    assert "(recorded 2026-09-19)" in user
    assert "[0]" in user


def test_extract_when_never_raises():
    class Broken(FakeLLM):
        def complete(self, system, user, *, json_schema=None):
            raise RuntimeError("provider down")

    items = [{"content": "a", "recorded_at": "2026-09-01"}] * 3
    assert extract_when(Broken(), items) == [None, None, None]
    assert extract_when(FakeLLM(["not json at all"]), items) == [None, None, None]
    assert extract_when(FakeLLM(['{"items": "nonsense"}']), items) == [None, None, None]
    assert extract_when(FakeLLM([_when_response()]), []) == []


def test_extract_when_ignores_rows_it_cannot_place():
    llm = FakeLLM([
        _when_response(
            {"index": 7, "start": "2026-10-03", "end": None, "recurrence": None},
            {"index": 0, "start": "not a date", "end": None, "recurrence": None},
            {"index": 1, "start": "2026-10-04", "end": None, "recurrence": None},
        )
    ])
    items = [{"content": "a", "recorded_at": "2026-09-01"}] * 2
    assert extract_when(llm, items) == [None, {"start": "2026-10-04"}]


# ------------------------------------------------------------- the write path


def _fact(content: str, when=None, **kw) -> dict:
    return {
        "content": content,
        "type": kw.get("type", "episodic"),
        "importance": kw.get("importance", 0.7),
        "categories": kw.get("categories", []),
        "entities": [],
        "relations": [],
        "when": when,
    }


def test_extraction_stores_the_when_on_the_memory(store, fake_llm):
    fake_llm.queue(json.dumps({"facts": [
        _fact("Ada's flight to Lisbon leaves on 2026-10-03",
              {"start": "2026-10-03T07:40", "end": None, "recurrence": None}),
    ]}), json.dumps({"missing": []}))
    store.add("I fly to Lisbon on 3 October at 07:40", user_id="u")
    memory = store.get_all(user_id="u")[0]
    assert memory.metadata["when"] == {"start": "2026-10-03T07:40"}


def test_a_dated_fact_that_is_not_an_event_stays_without_a_when(store, fake_llm):
    fake_llm.queue(json.dumps({"facts": [
        _fact("The EUR price of the plan on 2026-09-01 was 41.20", None, type="semantic"),
    ]}), json.dumps({"missing": []}))
    store.add("the plan cost 41.20 EUR on 2026-09-01", user_id="u")
    memory = store.get_all(user_id="u")[0]
    assert "when" not in memory.metadata


def test_an_update_keeps_the_stored_when(store, fake_llm):
    fake_llm.queue(json.dumps({"facts": [
        _fact("The Helios launch is on 2026-10-03",
              {"start": "2026-10-03", "end": None, "recurrence": None}),
    ]}), json.dumps({"missing": []}))
    store.add("Helios launches on 3 October", user_id="u")
    original = store.get_all(user_id="u")[0]

    # A refinement that says nothing about time: the occurrence time stands.
    fake_llm.queue(
        json.dumps({"facts": [_fact("The Helios launch is in Lisbon", None)]}),
        json.dumps({"action": "UPDATE", "target": 0,
                    "content": "The Helios launch is on 2026-10-03 in Lisbon",
                    "reason": "adds the place"}),
        json.dumps({"facts": []}),
        json.dumps({"missing": []}),
    )
    store.add("the Helios launch is in Lisbon", user_id="u")
    kept = store.get(original.id)
    assert kept.content.endswith("in Lisbon")
    assert kept.metadata["when"] == {"start": "2026-10-03"}


def test_an_update_that_carries_a_when_replaces_it(store, fake_llm):
    fake_llm.queue(json.dumps({"facts": [
        _fact("The Helios launch is on 2026-10-03",
              {"start": "2026-10-03", "end": None, "recurrence": None}),
    ]}), json.dumps({"missing": []}))
    store.add("Helios launches on 3 October", user_id="u")
    original = store.get_all(user_id="u")[0]

    fake_llm.queue(
        json.dumps({"facts": [
            _fact("The Helios launch moved to 2026-10-10",
                  {"start": "2026-10-10", "end": None, "recurrence": None}),
        ]}),
        json.dumps({"action": "UPDATE", "target": 0,
                    "content": "The Helios launch is on 2026-10-10",
                    "reason": "the date moved"}),
        json.dumps({"facts": []}),
        json.dumps({"missing": []}),
    )
    store.add("Helios now launches on 10 October", user_id="u")
    assert store.get(original.id).metadata["when"] == {"start": "2026-10-10"}


# ----------------------------------------------------------------- backfilling


def _seed(store, *contents, memory_type="episodic") -> list[str]:
    """Put memories in without going through reconciliation, which would want
    scripted answers of its own and has nothing to do with the backfill."""
    ids = []
    for content in contents:
        memory = Memory(content=content, memory_type=memory_type, user_id="u")
        store.backend.insert_memory(memory)
        ids.append(memory.id)
    return ids


def test_backfill_when_dry_run_writes_nothing(store, fake_llm):
    first, second = _seed(
        store,
        "Team offsite in Lisbon on 2026-10-03",
        "The EUR price on 2026-09-01 was 41.20",
    )
    fake_llm.queue(_when_response(
        {"index": 0, "start": "2026-10-03", "end": None, "recurrence": None},
        {"index": 1, "start": None, "end": None, "recurrence": None},
    ))
    outcome = store.backfill_when(user_id="u", dry_run=True)
    assert outcome["checked"] == 2
    assert outcome["found"] == 1
    assert [p["id"] for p in outcome["proposals"]] == [first]
    assert outcome["proposals"][0]["when"] == {"start": "2026-10-03"}
    assert store.get(first).metadata.get("when") is None
    assert store.get(second).metadata.get("when_checked") is None


def test_backfill_when_writes_and_a_re_run_costs_nothing(store, fake_llm):
    first, second = _seed(
        store,
        "Team offsite in Lisbon on 2026-10-03",
        "The EUR price on 2026-09-01 was 41.20",
    )
    before = store.get(first).updated_at
    fake_llm.queue(_when_response(
        {"index": 0, "start": "2026-10-03", "end": None, "recurrence": None},
        {"index": 1, "start": None, "end": None, "recurrence": None},
    ))
    assert store.backfill_when(user_id="u") == {"checked": 2, "found": 1}
    assert store.get(first).metadata["when"] == {"start": "2026-10-03"}
    assert store.get(second).metadata["when_checked"] is True
    assert store.get(first).updated_at == before  # housekeeping, not an edit

    # Nothing is left to look at, so the second run makes no provider call.
    assert store.backfill_when(user_id="u") == {"checked": 0, "found": 0}
    assert fake_llm.responses == []


def test_backfill_when_respects_limit_and_types(store, fake_llm):
    _seed(store, "one on 2026-10-03", "two on 2026-10-04")
    _seed(store, "Ada prefers TypeScript", memory_type="procedural")
    fake_llm.queue(_when_response(
        {"index": 0, "start": "2026-10-03", "end": None, "recurrence": None},
    ))
    outcome = store.backfill_when(user_id="u", limit=1)
    assert outcome["checked"] == 1


def test_backfill_when_without_an_llm_says_so(verbatim_store):
    verbatim_store.add("something on 2026-10-03", user_id="u", infer=False)
    assert verbatim_store.backfill_when(user_id="u") == {"skipped": "no LLM configured"}


# --------------------------------------------------------------- the filters


@pytest.fixture
def when_store():
    store = MemoryStore(
        Config(db_path=":memory:"), llm=NoneLLM(), embedder=HashEmbedder(64)
    )
    saved = store.add(
        "Team offsite in Lisbon on 2026-10-03", user_id="u", infer=False,
        memory_type="episodic",
    )
    offsite = saved.actions[0].memory_id
    store.backend.update_memory(
        offsite, metadata={"when": {"start": "2026-10-03"}}, touch=False
    )
    saved = store.add("Ada's birthday is 3 March", user_id="u", infer=False)
    birthday = saved.actions[0].memory_id
    store.backend.update_memory(
        birthday,
        metadata={"when": {"start": "--03-03", "recurrence": "yearly"}},
        touch=False,
    )
    saved = store.add(
        "The EUR price on 2026-10-03 was 41.20", user_id="u", infer=False
    )
    priced = saved.actions[0].memory_id
    yield store, {"offsite": offsite, "birthday": birthday, "priced": priced}
    store.close()


def test_store_filters_on_occurrence_time(when_store):
    store, ids = when_store
    found = store.get_all(user_id="u", when_since="2026-10-01", when_until="2026-10-31")
    assert [m.id for m in found] == [ids["offsite"]]

    found = store.get_all(user_id="u", when_since="2027-03-01", when_until="2027-03-31")
    assert [m.id for m in found] == [ids["birthday"]]

    # A memory with a date in its text but no occurrence time never matches.
    everything = store.get_all(user_id="u", when_since="1900-01-01")
    assert ids["priced"] not in [m.id for m in everything]


def test_search_filters_on_occurrence_time(when_store):
    store, ids = when_store
    results = store.search(
        "Lisbon", user_id="u", when_since="2026-10-01", when_until="2026-10-31"
    )
    assert [r.memory.id for r in results] == [ids["offsite"]]
    assert store.search(
        "Lisbon", user_id="u", when_since="2026-11-01", when_until="2026-11-30"
    ) == []
    # An empty query browses through the same filter.
    browsed = store.search("", user_id="u", when_since="2026-10-01", when_until="2026-10-31")
    assert [r.memory.id for r in browsed] == [ids["offsite"]]


def test_context_line_says_when_the_fact_happens(when_store):
    store, _ids = when_store
    context = store.reconstruct_context("Lisbon offsite", user_id="u")
    assert "(happened 2026-10-03)" in context.text or "(happens 2026-10-03)" in context.text


# ----------------------------------------------------------- REST and the MCP


@pytest.fixture
def when_client(when_store):
    store, ids = when_store
    with TestClient(create_app(store)) as client:
        yield client, store, ids


def test_rest_payload_carries_when_and_next_occurrence(when_client):
    client, _store, ids = when_client
    rows = {row["id"]: row for row in client.get("/api/v1/memories").json()}
    birthday = rows[ids["birthday"]]
    assert birthday["when"] == {"start": "--03-03", "recurrence": "yearly"}
    assert birthday["next_occurrence"].endswith("-03-03")
    priced = rows[ids["priced"]]
    assert priced["when"] is None
    assert priced["next_occurrence"] is None


def test_rest_filters_on_occurrence_time(when_client):
    client, _store, ids = when_client
    listed = client.get(
        "/api/v1/memories?when_since=2026-10-01&when_until=2026-10-31"
    ).json()
    assert [row["id"] for row in listed] == [ids["offsite"]]
    found = client.post("/api/v1/search", json={
        "query": "Lisbon", "when_since": "2026-10-01", "when_until": "2026-10-31",
    }).json()
    assert [row["memory"]["id"] for row in found] == [ids["offsite"]]


def test_rest_maintenance_runs_the_backfill(when_client):
    client, _store, _ids = when_client
    # No LLM in this fixture, so the pass reports why it did nothing rather
    # than 404ing like an unknown pass would.
    assert client.post("/api/v1/maintenance/run/when", json={"dry_run": True}).json() == {
        "skipped": "no LLM configured"
    }
    assert client.post("/api/v1/maintenance/run/nonsense", json={}).status_code == 404


def test_mcp_tools_filter_on_occurrence_time(when_client, monkeypatch):
    client, _store, ids = when_client
    listed = json.loads(mcp_call(client, "", "list_memories", {
        "user_id": "u", "when_since": "2026-10-01", "when_until": "2026-10-31",
    }))
    assert [row["id"] for row in listed] == [ids["offsite"]]
    searched = json.loads(mcp_call(client, "", "search_memories", {
        "query": "Lisbon", "user_id": "u",
        "when_since": "2026-10-01", "when_until": "2026-10-31",
    }))
    assert [row["id"] for row in searched] == [ids["offsite"]]
