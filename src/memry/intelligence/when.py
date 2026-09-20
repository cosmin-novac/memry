"""Occurrence time: when the thing a memory describes happens.

A memory already carries transaction time (``created_at``/``updated_at``) and
validity (``valid_from``/``invalid_at``). Neither answers "what is on this
weekend", because both record when the store learned something, not when the
thing itself takes place. A birthday has nowhere to live at all.

The occurrence time lives in ``metadata["when"]``:

    {"start": "2026-10-03" | "2026-10-03T19:00" | "--03-03",
     "end": same formats, optional,
     "recurrence": "yearly" | "monthly" | "weekly" | "daily", optional}

``--MM-DD`` is a yearly date whose year is unknown or does not matter, which is
what a birthday is: one recurring fact about a person, not one event per year.

A date in the text is NOT a "when". Of 160 labelled episodic memories from a
real store, only 44 described an event, while 46 non-events carried a calendar
date anyway: prices observed on a day, test logs, specifications. Reading every
date as an occurrence would therefore be wrong about as often as it was right,
so a "when" is set only when the fact describes something that happened or will
happen at a particular time. The same store recurred once in 160 memories,
which is why recurrence is a plain field rather than a rule engine.

Everything above ``extract_when`` is pure: no store, no provider, no clock
except the ``now`` the caller passes in.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

RECURRENCES: tuple[str, ...] = ("yearly", "monthly", "weekly", "daily")

#: The three shapes a point in time can take. ``yearless`` carries no year.
_FULL = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_FULL_TIME = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})[T ](\d{1,2}):(\d{2})(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?$"
)
# "--03-03" is the ISO 8601 yearless form; "XXXX-03-03" is what models write
# when asked for a date they do not have a year for, so both are accepted.
_YEARLESS = re.compile(r"^(?:--|[xX]{4}-)(\d{1,2})-(\d{1,2})$")

_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)
_EVERY = {"yearly": "year", "monthly": "month", "weekly": "week", "daily": "day"}


def _point(raw: Any) -> tuple[str, str] | None:
    """Normalise one point in time. Returns (text, kind) or None.

    kind is "date", "datetime" or "yearless". A February 29 anchor is valid in
    the yearless form even though most years have no such day; reading it is
    where that is resolved.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    match = _FULL_TIME.match(text)
    if match:
        year, month, day, hour, minute = (int(g) for g in match.groups())
        if hour > 23 or minute > 59:
            return None
        try:
            date(year, month, day)
        except ValueError:
            return None
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}", "datetime"
    match = _FULL.match(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        try:
            date(year, month, day)
        except ValueError:
            return None
        return f"{year:04d}-{month:02d}-{day:02d}", "date"
    match = _YEARLESS.match(text)
    if match:
        month, day = (int(g) for g in match.groups())
        if not 1 <= month <= 12:
            return None
        longest = 29 if month == 2 else calendar.monthrange(2001, month)[1]
        if not 1 <= day <= longest:
            return None
        return f"--{month:02d}-{day:02d}", "yearless"
    return None


def _month_day(text: str, kind: str) -> tuple[int, int]:
    if kind == "yearless":
        return int(text[2:4]), int(text[5:7])
    return int(text[5:7]), int(text[8:10])


def _day_of(text: str, kind: str) -> date | None:
    """The calendar day of a point, or None for a yearless one."""
    if kind == "yearless":
        return None
    return date(int(text[0:4]), int(text[5:7]), int(text[8:10]))


def _clamp(year: int, month: int, day: int) -> date:
    """A real day in that month: the 31st of April is its 30th, and a February
    29 anchor lands on the 28th in a common year."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def parse_when(raw: Any) -> dict[str, Any] | None:
    """Validate and normalise a "when", or return None.

    Rejected: anything without a readable start, a date that does not exist
    (2026-02-30), and an end before its start. A recurrence that is not one of
    the four known words is dropped rather than failing the whole value, and a
    yearless start is marked yearly, which is the only thing it can mean.
    """
    if isinstance(raw, str):
        raw = {"start": raw}
    if not isinstance(raw, dict):
        return None
    start = _point(raw.get("start"))
    if start is None:
        return None
    start_text, start_kind = start
    when: dict[str, Any] = {"start": start_text}

    end_raw = raw.get("end")
    if end_raw not in (None, ""):
        end = _point(end_raw)
        if end is not None:
            end_text, end_kind = end
            yearless = (start_kind == "yearless", end_kind == "yearless")
            if yearless[0] != yearless[1]:
                pass  # one has a year and the other does not: unusable, drop it
            elif _before(end_text, end_kind, start_text, start_kind):
                return None  # an end before its start describes nothing
            else:
                when["end"] = end_text

    recurrence = raw.get("recurrence")
    if isinstance(recurrence, str) and recurrence.strip().lower() in RECURRENCES:
        when["recurrence"] = recurrence.strip().lower()
    elif start_kind == "yearless":
        when["recurrence"] = "yearly"
    return when


def _before(a_text: str, a_kind: str, b_text: str, b_kind: str) -> bool:
    """Is point a strictly before point b? Compares days first, so a date-only
    end on the start's own day is not read as earlier than it."""
    if a_kind == "yearless":
        return _month_day(a_text, a_kind) < _month_day(b_text, b_kind)
    a_day, b_day = _day_of(a_text, a_kind), _day_of(b_text, b_kind)
    if a_day != b_day:
        return a_day < b_day  # type: ignore[operator]
    if a_kind == "datetime" and b_kind == "datetime":
        return a_text[11:] < b_text[11:]
    return False


def _today(now: Any = None) -> date:
    if isinstance(now, datetime):
        return now.date()
    if isinstance(now, date):
        return now
    if isinstance(now, str):
        parsed = _point(now)
        if parsed is not None and parsed[1] != "yearless":
            return _day_of(*parsed)  # type: ignore[return-value]
    return datetime.now(timezone.utc).date()


def next_occurrence(when: Any, now: Any = None) -> date | None:
    """The first day on or after ``now`` on which this happens.

    A one-off that has passed returns None: it is history, not a date to plan
    around. A recurring one steps forward to its next turn, month-end safe in
    both directions (the 31st, and February 29).
    """
    data = parse_when(when)
    if data is None:
        return None
    today = _today(now)
    start_kind = _point(data["start"])[1]  # type: ignore[index]
    start_day = _day_of(data["start"], start_kind)
    recurrence = data.get("recurrence")

    if not recurrence:
        return start_day if start_day is not None and start_day >= today else None

    floor = today if start_day is None else max(today, start_day)
    month, day = _month_day(data["start"], start_kind)

    if recurrence == "daily":
        candidate = floor
    elif recurrence == "weekly":
        weekday = (start_day or _clamp(floor.year, month, day)).weekday()
        candidate = floor + timedelta(days=(weekday - floor.weekday()) % 7)
    elif recurrence == "monthly":
        candidate = _clamp(floor.year, floor.month, day)
        if candidate < floor:
            year = floor.year + (1 if floor.month == 12 else 0)
            candidate = _clamp(year, 1 if floor.month == 12 else floor.month + 1, day)
    else:  # yearly
        candidate = _clamp(floor.year, month, day)
        if candidate < floor:
            candidate = _clamp(floor.year + 1, month, day)

    end = data.get("end")
    if end:
        end_day = _day_of(end, _point(end)[1])  # type: ignore[index]
        if end_day is not None and candidate > end_day:
            return None  # the repetition has a last day and it is behind us
    return candidate


def _span_days(data: dict[str, Any]) -> int:
    """How many days one occurrence covers, beyond its first."""
    end = data.get("end")
    if not end:
        return 0
    start_kind = _point(data["start"])[1]  # type: ignore[index]
    end_kind = _point(end)[1]  # type: ignore[index]
    if start_kind == "yearless":
        # Both are yearless here (parse_when drops a mismatched end). Measure
        # the span in a leap year so a February 29 anchor still counts.
        first = _clamp(2024, *_month_day(data["start"], start_kind))
        last = _clamp(2024, *_month_day(end, end_kind))
        return max((last - first).days, 0)
    first_day = _day_of(data["start"], start_kind)
    last_day = _day_of(end, end_kind)
    if first_day is None or last_day is None:
        return 0
    return max((last_day - first_day).days, 0)


def overlaps(when: Any, since: str | None = None, until: str | None = None) -> bool:
    """Does this "when" fall inside the [since, until] window? Both bounds are
    inclusive days, and either may be omitted. A recurring "when" matches when
    any of its occurrences lands in the window."""
    data = parse_when(when)
    if data is None:
        return False
    lo = _bound(since)
    hi = _bound(until)
    if since and lo is None:
        return False
    if until and hi is None:
        return False
    span = _span_days(data)

    if not data.get("recurrence"):
        start_kind = _point(data["start"])[1]  # type: ignore[index]
        first = _day_of(data["start"], start_kind)
        if first is None:
            return False
        last = first + timedelta(days=span)
        return not (hi is not None and first > hi) and not (lo is not None and last < lo)

    # Recurring: the first occurrence that could still reach the window. An
    # occurrence starting up to `span` days before `lo` is still inside it.
    try:
        floor = date.min if lo is None else lo - timedelta(days=span)
    except OverflowError:
        floor = date.min
    occurrence = next_occurrence(data, floor)
    if occurrence is None:
        return False
    return hi is None or occurrence <= hi


def _bound(value: str | None) -> date | None:
    if not value:
        return None
    parsed = _point(value)
    if parsed is None or parsed[1] == "yearless":
        return None
    return _day_of(*parsed)


def describe_when(when: Any, now: Any = None) -> str:
    """One short phrase for a memory line or a dashboard chip."""
    data = parse_when(when)
    if data is None:
        return ""
    start = data["start"].replace("T", " ")
    upcoming = next_occurrence(data, now)
    recurrence = data.get("recurrence")

    if recurrence:
        unit = _EVERY.get(recurrence, recurrence)
        if recurrence == "daily":
            phrase = "every day"
        elif recurrence == "weekly":
            start_kind = _point(data["start"])[1]  # type: ignore[index]
            day = _day_of(data["start"], start_kind)
            weekday = _WEEKDAYS[day.weekday()] if day else ""
            phrase = f"every week on {weekday}" if weekday else "every week"
        else:
            anchor = start[2:] if start.startswith("--") else start[5:]
            phrase = f"every {unit} on {anchor}"
        return f"{phrase}, next {upcoming.isoformat()}" if upcoming else phrase

    end = data.get("end")
    span = f"{start} to {end.replace('T', ' ')}" if end else start
    if upcoming:
        return f"happens {span}"
    if end and not start.startswith("--"):
        end_kind = _point(end)[1]  # type: ignore[index]
        last = _day_of(end, end_kind)
        if last is not None and last >= _today(now):
            return f"happening now, {span}"
    return f"happened {span}"


# ----------------------------------------------------------------------
# backfill: reading a "when" out of memories written before it existed
# ----------------------------------------------------------------------

#: The per-fact shape, reused by the extraction schema on the write path.
WHEN_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
        "recurrence": {"type": ["string", "null"]},
    },
    "required": ["start", "end", "recurrence"],
    "additionalProperties": False,
}

WHEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                    "recurrence": {"type": ["string", "null"]},
                },
                "required": ["index", "start", "end", "recurrence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

#: The rule, in the words the model reads. Written from 160 labelled memories
#: of a real store: a date in the text is common in non-events, so the test is
#: whether the fact describes something that occurs, not whether it has a date.
WHEN_SYSTEM = """You read stored memories and say when the thing each one
describes happens, if it happens at a time at all.

Give a memory a "when" only when the fact describes something that happened or
will happen at a particular time: a meeting, a launch, a release, a purchase, a
trip, an appointment, a deadline, a move, or a decision made on a date.

Leave "when" empty for a state, a preference, a rule, a price, a measurement, a
specification, an implementation note or a test log, EVEN WHEN A DATE APPEARS
IN THE TEXT. Most dated memories are not events: a price observed on a day, a
log written on a day, and a document dated a day all carry dates without
occurring. The date something was written down is never its "when". When you
cannot tell whether the fact occurs or merely holds, leave it empty; a wrong
"when" is worse than none.

A birthday or an anniversary is a yearly recurrence on a person, not one event
per year: write it as start "--MM-DD" with recurrence "yearly".

Fields:
- start: "YYYY-MM-DD", or "YYYY-MM-DDTHH:MM" when the text states a clock time,
  or "--MM-DD" for a yearly date whose year is unknown or does not matter.
- end: the same formats, only when the fact spans a period (a trip, a stay, a
  window). Leave it null otherwise.
- recurrence: "yearly", "monthly", "weekly" or "daily", only when the fact says
  the thing repeats. Leave it null otherwise.

Each memory is given with the date it was recorded. Resolve relative wording
("yesterday", "last Tuesday", "next month", "in two weeks") against that
recorded date and write the absolute date you arrive at. Leave "when" empty
when no date can be worked out, including vague wording such as "soon", "one
day" or "later this year".

Answer for every memory, by its index. Return JSON only:
{"items": [{"index": int, "start": str|null, "end": str|null,
"recurrence": str|null}]}
Set start to null for a memory that has no "when"."""


# ----------------------------------------------------------------------
# confirming a "when": the text model proposes a date, and is not believed
# on its own
# ----------------------------------------------------------------------
#
# Measured on 160 labelled memories of a real store, 44 of them events. How
# much a text model can be trusted here depends on the model: gpt-5-mini gave a
# "when" to 61 memories and was right about 56% of them, dating work logs after
# being told in capitals not to; gpt-5.6-luna gave 39 and was right about 85%,
# four times faster. Two checks sit on top of either.
#
# * A "when" equal to the day the memory was recorded, in a memory whose text
#   names no date, is the write date read back.
# * The decision provider is asked the one thing it is good at, whether the
#   memory is an event or a record, and holds a veto: a "when" is dropped when
#   it calls the memory a record with at least 0.80 probability. Requiring an
#   "event" verdict instead cost a fifth of the real events for nothing. With
#   gpt-5.6-luna the veto gives 97% precision with 66% of events found (86% and
#   68% without a provider); with gpt-5-mini, 86% and 68%.

_NAMES_A_DATE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\b(?:today|yesterday|tomorrow|tonight|this (?:morning|afternoon|evening|week|month)"
    r"|last|next|ago|heute|gestern|morgen|letzte[nrs]?|n\u00e4chste[nrs]?)\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)

EVENT_CRITERIA = {
    "event": ("Something that happened or will happen at a particular time and "
              "that a person would put on a calendar or a timeline: a meeting, "
              "an appointment, a launch, a public release, a purchase, a "
              "payment, a trip, a move, a deadline, an incident, a decision or "
              "a request made on a day."),
    "record": ("A record of work done, a state, a fact, a preference, a rule, a "
               "price, a measurement, a specification, a test or verification "
               "result, a commit or a deployment log. It may carry a date, but "
               "nothing in it is an occasion."),
}
#: Probability on "record" from which the provider's veto drops a "when".
RECORD_VETO = 0.80


def is_write_date(when: Any, content: str, recorded_at: str | None) -> bool:
    """A one-off "when" on the very day the memory was recorded, in a text
    that names no date: the model read the write date back."""
    data = parse_when(when)
    if data is None or data.get("recurrence") or not recorded_at:
        return False
    return (
        data["start"][:10] == str(recorded_at)[:10]
        and not _NAMES_A_DATE.search(content or "")
    )


def confirm_whens(
    decider: Any, items: list[dict[str, Any]], found: list[dict[str, Any] | None]
) -> list[dict[str, Any] | None]:
    """Keep only the proposed "when"s that survive both checks.

    ``items`` are {"content", "recorded_at"}, aligned with ``found``. Never
    raises: a provider that fails or does not answer leaves the mechanical
    check as the only one, exactly as when no provider is configured.
    """
    kept = [
        None if (when and is_write_date(when, item.get("content") or "", item.get("recorded_at")))
        else when
        for item, when in zip(items, found)
    ]
    asked = [i for i, when in enumerate(kept) if when]
    if not asked or decider is None or not getattr(decider, "available", False):
        return kept
    try:
        from ..providers.decisions import Choice

        def ask(index: int):
            answer = decider.decide(
                "A memory from a personal long-term memory store: "
                + str(items[index].get("content") or ""),
                {"k": Choice(instructions="Is this memory an event or a record?",
                             criteria=EVENT_CRITERIA)},
            )["k"]
            return index, answer

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=6) as pool:
            answers = list(pool.map(ask, asked))
    except Exception:
        return kept
    for index, answer in answers:
        if not getattr(answer, "available", False):
            continue
        probability = (answer.probabilities or {}).get(answer.value, answer.confidence)
        if answer.value == "record" and float(probability or 0.0) >= RECORD_VETO:
            kept[index] = None
    return kept


def extract_when(llm: Any, items: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Read a "when" out of each memory. One call for the whole batch.

    ``items`` are {"content", "recorded_at"}. The result is aligned with them,
    holding a validated when-dict or None. Any failure - provider error, bad
    JSON, a missing or unusable answer - is a None, never an exception: this
    runs over a whole store, and one malformed batch must not stop a backfill.
    """
    if not items:
        return []
    out: list[dict[str, Any] | None] = [None] * len(items)
    listing = "\n".join(
        f"[{index}] (recorded {str(item.get('recorded_at') or 'unknown')[:10]}) "
        f"{' '.join(str(item.get('content') or '').split())}"
        for index, item in enumerate(items)
    )
    try:
        # Imported here: extraction.py reads the schemas above, so a top-level
        # import in either direction would be a cycle.
        from .extraction import parse_lenient_json

        raw = llm.complete(
            WHEN_SYSTEM,
            f"Memories:\n{listing}\n\nAnswer for every index as JSON.",
            json_schema=WHEN_SCHEMA,
        )
        parsed = parse_lenient_json(raw)
    except Exception:
        return out
    rows = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(items):
            continue
        out[index] = parse_when(row)
    return out
