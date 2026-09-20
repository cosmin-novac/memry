"""A proposed "when" is not believed on the text model's word alone.

Measured on 160 labelled memories: a text model dates work logs (gpt-5-mini was
right about 56% of the "when"s it gave, gpt-5.6-luna about 85%). Dropping write
dates read back, and letting the decision provider veto what it is sure is a
record, takes gpt-5.6-luna to 97%. These tests pin the two checks, not the
numbers.
"""

from __future__ import annotations

from datetime import date

from memry.intelligence.when import confirm_whens, describe_when, is_write_date
from memry.providers.decisions import Answer, Answers, Decider


class EventJudge(Decider):
    name = "event-judge"
    available = True

    def __init__(self, verdicts: dict[str, tuple[str, float]]):
        self.verdicts = verdicts
        self.asked: list[str] = []

    def decide(self, state, questions):
        self.asked.append(state)
        for needle, (value, probability) in self.verdicts.items():
            if needle in state:
                return Answers({key: Answer(value=value, probabilities={value: probability},
                                            confidence=probability, available=True)
                                for key in questions})
        return Answers({key: Answer() for key in questions})


WHEN = {"start": "2026-09-11"}


def test_a_when_on_the_recording_day_with_no_date_in_the_text_is_the_write_date():
    assert is_write_date(WHEN, "Git main was pushed with commit 334d4c", "2026-09-11T10:00:00+00:00")
    assert not is_write_date(WHEN, "Pushed on 2026-09-11 after review", "2026-09-11T10:00:00+00:00")
    assert not is_write_date(WHEN, "We met the landlord yesterday", "2026-09-11T10:00:00+00:00")
    assert not is_write_date({"start": "2026-10-03"}, "Dentist appointment", "2026-09-11T10:00:00+00:00")
    assert not is_write_date({"start": "--03-03", "recurrence": "yearly"}, "Raluca's birthday",
                             "2026-03-03T10:00:00+00:00")


def test_without_a_provider_only_the_write_date_check_applies():
    items = [{"content": "Git main was pushed", "recorded_at": "2026-09-11"},
             {"content": "Dentist on 2026-10-03", "recorded_at": "2026-09-11"}]
    assert confirm_whens(None, items, [WHEN, {"start": "2026-10-03"}]) == [None, {"start": "2026-10-03"}]


def test_the_provider_vetoes_what_it_is_sure_is_a_record():
    items = [{"content": "27 tests passed on 2026-09-11", "recorded_at": "2026-09-12"},
             {"content": "Product Hunt launch on 2026-10-03", "recorded_at": "2026-09-12"},
             {"content": "Reviewed the signup terms on 2026-10-09", "recorded_at": "2026-09-12"},
             {"content": "No date here", "recorded_at": "2026-09-12"}]
    found = [{"start": "2026-09-11"}, {"start": "2026-10-03"}, {"start": "2026-10-09"}, None]
    # sure it is a record: dropped. An event: kept. A record it is not sure
    # about: kept, because requiring an "event" verdict lost a fifth of the
    # real events on the labelled set for no gain in precision.
    judge = EventJudge({"tests passed": ("record", 0.9), "Product Hunt": ("event", 0.8),
                        "signup terms": ("record", 0.6)})

    kept = confirm_whens(judge, items, found)

    assert kept == [None, {"start": "2026-10-03"}, {"start": "2026-10-09"}, None]
    assert len(judge.asked) == 3, "a memory with no proposed when is not asked about"


def test_a_provider_that_fails_or_stays_silent_costs_no_when():
    class Broken(Decider):
        name, available = "broken", True

        def decide(self, state, questions):
            raise RuntimeError("down")

    items = [{"content": "Launch on 2026-10-03", "recorded_at": "2026-09-12"}]
    assert confirm_whens(Broken(), items, [{"start": "2026-10-03"}]) == [{"start": "2026-10-03"}]
    assert confirm_whens(EventJudge({}), items, [{"start": "2026-10-03"}]) == [{"start": "2026-10-03"}]


def test_a_span_that_has_started_and_not_ended_is_happening_now():
    trip = {"start": "2026-09-01", "end": "2026-09-30"}
    assert describe_when(trip, date(2026, 9, 15)).startswith("happening now")
    assert describe_when(trip, date(2026, 8, 1)).startswith("happens")
    assert describe_when(trip, date(2026, 10, 5)).startswith("happened")
