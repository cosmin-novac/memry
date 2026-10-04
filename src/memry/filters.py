"""Search filters: what a caller states to narrow a search before anything
is ranked.

A question such as "Was habe ich am 01. April 2025 gemacht?" carries its
filter in its words, but Memry does not guess dates or names from question
text: an agent holding the conversation knows them better, in any language,
and says them as parameters (``when="2025-04-01"``). Every filter is a hard
pre-filter. ``MemoryStore.search`` keeps every candidate of every stage (text
ranking, linked pool, judged pool, set call, context) to what the filters
admit, so a memory filtered out never comes back as a close match.

Agents get two filters and a convention, so a tool call stays easy to get
right: ``when`` (the time a question is about: when the thing happened where
a memory knows it, else the day it was said), ``about`` (names of people,
projects, things or tags) and a phrase in double quotes inside the query,
matched exactly. The REST endpoints read the same object with the finer
filters a dashboard or a script wants (``happened``, ``said``, ``entity``,
``entity_type``, ``tag``, ``contains``, ``memory_type``).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from .intelligence.context import said_at
from .intelligence.when import overlaps, parse_when
from .models import MEMORY_TYPES, NAMED_ENTITY_TYPES, TOPIC_TYPE

_DAY = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_YEAR = re.compile(r"^(\d{4})$")
#: A phrase in double quotes, straight or typographic ("...", “...”, „...“).
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”|„([^“”]+)[“”]')

PERIOD_FORMS = ('a day "2025-04-01", a month "2025-04", a year "2025" or a range '
                '"2025-04-01..2025-06-30"')


@dataclass(frozen=True)
class Period:
    """A period a filter asks for: its first and last day (either open in a
    range written "2025-04.." or "..2025-06"), as written, and how it was
    written ("day", "month", "year" or "range")."""

    text: str
    start: date | None
    end: date | None
    precision: str

    def contains(self, day: date) -> bool:
        return (self.start is None or self.start <= day) and (self.end is None or day <= self.end)

    def bounds(self) -> tuple[str | None, str | None]:
        """The first and last day as ``intelligence.when.overlaps`` reads them."""
        return (self.start.isoformat() if self.start else None,
                self.end.isoformat() if self.end else None)


def _endpoint(text: str) -> tuple[date, date, str]:
    """The first and last day of a day, a month or a year as written."""
    try:
        if match := _DAY.match(text):
            day = date(*(int(g) for g in match.groups()))
            return day, day, "day"
        if match := _MONTH.match(text):
            year, month = (int(g) for g in match.groups())
            return (date(year, month, 1),
                    date(year, month, calendar.monthrange(year, month)[1]), "month")
        if match := _YEAR.match(text):
            year = int(match.group(1))
            return date(year, 1, 1), date(year, 12, 31), "year"
    except ValueError:
        pass
    raise ValueError(text)


def parse_period(value: str, name: str) -> Period | None:
    """A period filter as written, or None when empty. A range's ends may be
    days, months or years, each taken whole ("2025-04..2025-06" is 1 April to
    30 June), and either end may be left open. Anything else is an error
    that says the accepted forms, so an agent can correct its call."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        if ".." in text:
            low, high = (part.strip() for part in text.split("..", 1))
            if not (low or high):
                raise ValueError(text)
            start = _endpoint(low)[0] if low else None
            end = _endpoint(high)[1] if high else None
            if start and end and end < start:
                raise ValueError(text)
            return Period(text, start, end, "range")
        start, end, precision = _endpoint(text)
        return Period(text, start, end, precision)
    except ValueError:
        raise ValueError(f'{name}="{text}" is not {PERIOD_FORMS}') from None


def _names(value: Any) -> tuple[str, ...]:
    """Comma-separated text, or a list, as distinct trimmed names in order."""
    if not value:
        return ()
    parts = value.split(",") if isinstance(value, str) else [str(v) for v in value]
    out: list[str] = []
    for part in parts:
        name = " ".join(str(part).split())
        if name and name.casefold() not in {n.casefold() for n in out}:
            out.append(name)
    return tuple(out)


def fold(text: str) -> str:
    """Text as an exact-phrase filter compares it: case folded, so "strasse"
    finds "Straße" and "ADA" finds "Ada", and runs of whitespace read as one
    space. A plain substring test, so quotes, %, _ and other characters a
    LIKE or FTS query would read as syntax are matched as written."""
    return " ".join(text.casefold().split())


def quoted_phrases(query: str) -> tuple[str, ...]:
    """The phrases in double quotes inside a query, each matched exactly
    (``fold``) as a hard filter: 'the "blue fig" dinner' keeps the memories
    that say "blue fig"."""
    found = (" ".join(next(g for g in match.groups() if g).split())
             for match in _QUOTED.finditer(query or ""))
    return tuple(dict.fromkeys(phrase for phrase in found if phrase))


@dataclass(frozen=True)
class Filters:
    """One set of search filters, as the tools and endpoints take them.

    ``when`` matches a memory's occurrence time where it has one and the day
    it was said (``created_at``) otherwise, so the caller never chooses
    between the two; ``happened`` reads the occurrence time alone (a memory
    without one never matches) and ``said`` the day the result rows show as
    "said" (``context.said_at``). Every period matches by overlap: a memory
    dated "April 2025" matches a question about 1 April.

    ``about`` names people, projects, things or tags: each name is resolved
    by ``MemoryStore.resolve_filters`` through entity names, aliases and
    merges, and tags, and a memory matches any of them. ``entity`` (named
    things only) and ``tag`` are its two halves; ``contains`` holds the
    exact phrases, a query's quoted ones included.

    The legacy parameters stay as they were: ``since``/``until`` on the day
    a memory was first saved (``created_at``), ``when_since``/``when_until``
    on when it happens, ``categories`` as tags and ``entity_id`` as entity
    ids.

    ``MemoryStore.resolve_filters`` fills the second group: the entity ids
    the names stand for, followed through merges, the memories ``about``
    reaches, the notes a result should carry (an unknown name and the names
    close to it) and whether the filters can match nothing at all."""

    when: Period | None = None
    about: tuple[str, ...] = ()
    happened: Period | None = None
    said: Period | None = None
    entities: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    memory_type: str = ""
    since: str = ""
    until: str = ""
    when_since: str = ""
    when_until: str = ""
    # filled by MemoryStore.resolve_filters
    resolved: bool = False
    resolved_entity_ids: tuple[str, ...] = ()
    about_ids: frozenset[str] | None = None
    notes: tuple[str, ...] = ()
    matches_nothing: bool = False

    @classmethod
    def parse(
        cls, *, query: str = "", when: Any = "", about: Any = "", happened: Any = "",
        said: Any = "", entity: Any = "", entity_id: Any = "", entity_type: Any = "",
        tag: Any = "", contains: Any = "", memory_type: Any = "", since: Any = "",
        until: Any = "", when_since: Any = "", when_until: Any = "", categories: Any = "",
    ) -> Filters:
        """The filters as a tool or an endpoint received them, the phrases
        quoted in ``query`` among the exact ones. A malformed value is a
        ValueError that names the parameter and what it accepts: a filter
        read wrong must not silently become no filter."""
        types = tuple(t.casefold() for t in _names(entity_type))
        for kind in types:
            if kind in (TOPIC_TYPE, "tag", "tags", "category"):
                raise ValueError('entity_type="topic": tags are filtered with tag=, '
                                 "not entity_type")
            if kind not in NAMED_ENTITY_TYPES:
                raise ValueError(f'entity_type="{kind}" is not one of '
                                 f'{", ".join(NAMED_ENTITY_TYPES)}')
        kind = str(memory_type or "").strip().casefold()
        if kind and kind not in MEMORY_TYPES:
            raise ValueError(f'memory_type="{kind}" is not one of {", ".join(MEMORY_TYPES)}')
        phrases = quoted_phrases(query)
        stated = " ".join(str(contains or "").split())
        return cls(
            when=parse_period(str(when or ""), "when"),
            about=_names(about),
            happened=parse_period(str(happened or ""), "happened"),
            said=parse_period(str(said or ""), "said"),
            entities=_names(entity),
            entity_ids=_names(entity_id),
            entity_types=types,
            tags=_names([*_names(tag), *_names(categories)]),
            contains=tuple(dict.fromkeys([*phrases, *([stated] if stated else [])])),
            memory_type=kind,
            since=str(since or "").strip(),
            until=str(until or "").strip(),
            when_since=str(when_since or "").strip(),
            when_until=str(when_until or "").strip(),
        )

    @property
    def active(self) -> bool:
        """Whether any filter is set."""
        return bool(self.applied())

    @property
    def scans(self) -> bool:
        """Whether a filter is read from each memory rather than in SQL: the
        periods, the exact phrases, the memory and entity types, ``about``.
        Those are computed once over the scope searched
        (``MemoryStore._admitted``) and the set they admit is what every
        stage then reads."""
        return bool(self.when or self.about or self.happened or self.said or self.contains
                    or self.memory_type or self.entity_types or self.since or self.until
                    or self.when_since or self.when_until)

    @property
    def period(self) -> Period | None:
        """The time asked about (``when``, else ``happened``): what a row's
        coarse date is measured against and what an empty search shows the
        nearest memories to."""
        return self.when or self.happened

    def applied(self) -> dict[str, str]:
        """The filters set, as the caller wrote them: what a result says it
        applied."""
        values = {
            "when": self.when.text if self.when else "",
            "about": ", ".join(self.about),
            "happened": self.happened.text if self.happened else "",
            "said": self.said.text if self.said else "",
            "entity": ", ".join(self.entities),
            "entity_id": ", ".join(self.entity_ids),
            "entity_type": ", ".join(self.entity_types),
            "tag": ", ".join(self.tags),
            "exact text": ", ".join(f'"{p}"' for p in self.contains),
            "memory_type": self.memory_type,
            "since": self.since,
            "until": self.until,
            "when_since": self.when_since,
            "when_until": self.when_until,
        }
        return {key: value for key, value in values.items() if value}

    def without_period(self) -> Filters:
        """These filters but the time asked about: what the nearest dated
        memories of an empty search are chosen from."""
        return replace(self, when=None, happened=None, when_since="", when_until="")

    def describe(self) -> str:
        """The filters applied, in one line: 'when=2025-04-01, about=Bochra
        Saffar'."""
        return ", ".join(f"{key}={value}" for key, value in self.applied().items())


def admits(filters: Filters, memory: Any) -> bool:
    """Whether a memory passes the filters read from the memory itself: the
    periods, the exact phrases and the memory type. Tags, entities, entity
    types and ``about`` are read by the store, in SQL or from the links."""
    if filters.memory_type and memory.memory_type != filters.memory_type:
        return False
    if filters.contains:
        text = fold(memory.content)
        if any(fold(phrase) not in text for phrase in filters.contains):
            return False
    when = (memory.metadata or {}).get("when")
    if filters.happened and not overlaps(when, *filters.happened.bounds()):
        return False
    if filters.when:
        if parse_when(when) is not None:
            if not overlaps(when, *filters.when.bounds()):
                return False
        elif not _day_in(memory.created_at, filters.when):
            return False
    if filters.said:
        if not _day_in(said_at(memory), filters.said):
            return False
    return True


def _day_in(timestamp: str | None, period: Period) -> bool:
    try:
        return period.contains(date.fromisoformat((timestamp or "")[:10]))
    except ValueError:
        return False
