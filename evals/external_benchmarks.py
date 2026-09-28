"""Memry on published long-term-memory benchmarks: LoCoMo and LongMemEval.

LoCoMo (Maharana et al. 2024, ``locomo10.json``): long conversations between
two people over many dated sessions, each with a few hundred questions whose
evidence is named by turn ("D3:7"). LongMemEval (Wu et al. 2024,
``longmemeval_s.json``, ``_m``, ``_oracle``): questions that each come with
their own haystack of dated chat sessions; the evidence is named by session.

Each conversation (a LoCoMo sample; a LongMemEval question with its haystack)
goes into a fresh in-memory store through ``MemoryStore.add``:

  verbatim  one add(infer=False) a turn: no model, each turn one memory
  extract   add(infer=True) with the configured LLM, one call a session
            (--extract-unit turn: one a turn). A memory is traced to its turns
            through the episodes it came from, so with whole sessions a
            memory counts for every turn of its session.

Every memory gets its session's date as its time: created_at and updated_at
(``add`` takes no time, so both are written after the save, see
``_set_created_at``), and ``metadata["when"] = {"start": "YYYY-MM-DD"}`` when
created_at cannot be written (--when always: on every new memory).
``metadata["bench"]`` names the conversation, session and turns it came from.

Every question is asked through ``MemoryStore.search`` (user "bench", the top
20) and scored without a model:

  recall@k  evidence_recall@k: of the question's evidence units (LoCoMo:
            turns, from "evidence"; LongMemEval: sessions, from
            "answer_session_ids"), the share that some memory among the top k
            comes from; k = 5, 10, 20. A question without evidence is left
            out of the mean (its row says null).
  mrr       1 / rank of the first memory that comes from an evidence unit,
            0 when none of the top 20 does
  search ms median time of one search

With --answer the configured LLM answers each question from the top --k
memories (--context: from ``MemoryStore.reconstruct_context``) and the
answer is scored against the gold one:

  f1        token F1 after SQuAD normalisation (lower case, punctuation and
            the articles a/an/the removed); by LoCoMo's rules, a multi-hop
            gold answer is scored by its comma-separated parts and an
            open-domain one by its first ";"-separated alternative
  em        normalised exact match
  contains  the normalised gold answer occurs, as whole words, in the answer
  judge     ``Judge(question, gold, prediction) -> bool``; the default
            (``containment_judge``) is "contains", or both abstaining.
            LongMemEval's LLM judge plugs in here: --judge module:function

A question whose right answer is that the conversation does not say
(LoCoMo's adversarial ones, LongMemEval's "_abs") scores 1 on f1, em and
contains when the answer abstains ("No information available").

Run (the data directory holds locomo10.json and longmemeval_s.json):

    MEMRY_BENCH_DATA=/data .venv/bin/python -m evals.external_benchmarks --dataset locomo
    MEMRY_BENCH_DATA=/data .venv/bin/python -m evals.external_benchmarks \\
        --dataset longmemeval --limit 30 --seed 1
    # with models (costs money): extraction, OpenAI embeddings, answers
    ... --ingest extract --embedder openai --answer

Results go to --out (default $MEMRY_BENCH_DATA/results/<dataset>_<time>.json):
per-question rows and the tables, which are also printed as markdown.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib
import json
import os
import pathlib
import random
import re
import sqlite3
import statistics
import string
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from memry.config import Config, EmbeddingConfig  # noqa: E402
from memry.models import AddAction, Memory  # noqa: E402
from memry.providers.embeddings import Embedder, HashEmbedder, OpenAIEmbedder  # noqa: E402
from memry.providers.llm import LLM, NoneLLM, build_llm  # noqa: E402
from memry.store import MemoryStore  # noqa: E402

DATA_ENV = "MEMRY_BENCH_DATA"
DATASETS = ("locomo", "longmemeval")
BENCH_USER = "bench"
KS = (5, 10, 20)
DEPTH = max(KS)
#: Token budget of the context block with --context.
CONTEXT_TOKENS = 4000

#: LoCoMo's "category" number -> name. The paper names five kinds of question
#: (single-hop, multi-hop, temporal, open-domain, adversarial) but the file
#: holds only numbers, and harnesses built on it disagree about which is
#: which. This is the reading of the LoCoMo repository's own scorer; edit it
#: here if yours differs. Rows and tables always keep the number as well.
LOCOMO_CATEGORIES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
#: Category names whose right answer is that the conversation does not say.
ABSTAIN_CATEGORIES = {"adversarial"}
#: The gold answer given to such a question when the file has none (LoCoMo
#: keeps only a misleading "adversarial_answer" for them).
ABSTAIN_ANSWER = "Not mentioned in the conversation"

INGEST_MODES = ("verbatim", "extract")
EXTRACT_UNITS = ("session", "turn")
WHEN_POLICIES = ("fallback", "always", "never")


class FormatError(ValueError):
    """A benchmark file that does not have the shape its loader reads."""


# --------------------------------------------------------------------------
# the data


@dataclass
class Turn:
    key: str      # LoCoMo: the dia_id ("D1:3"); LongMemEval: "<session id>#<index from 0>"
    role: str     # LoCoMo: the speaker's name; LongMemEval: "user" or "assistant"
    raw: str      # what was said (LoCoMo: with a shared photo's caption)
    text: str     # the memory it becomes verbatim ("Maya: ...", "assistant: ...")
    has_answer: bool = False


@dataclass
class Session:
    session_id: str
    date: datetime | None
    date_text: str
    turns: list[Turn]


@dataclass
class Question:
    qid: str
    question: str
    answer: str
    category: int | str       # LoCoMo: the number; LongMemEval: the question type
    category_name: str
    evidence: list[str]       # turn keys (level "turn") or session ids (level "session")
    level: str
    question_date: datetime | None = None
    abstain: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    conv_id: str
    label: str                # said to the extractor with each save
    sessions: list[Session]
    questions: list[Question]
    warnings: list[str] = field(default_factory=list)

    @property
    def turns(self) -> list[Turn]:
        return [t for s in self.sessions for t in s.turns]


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
# "1:56 pm on 8 May, 2023" (LoCoMo)
_LOCOMO_DATE = re.compile(
    r"^(\d{1,2})[:.](\d{2})\s*([ap])\.?\s*m\.?\s+on\s+(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$",
    re.IGNORECASE)
# "8 May, 2023", "8 May 2023"
_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})$")
# "May 8, 2023"
_MONTH_DAY_YEAR = re.compile(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$")
# "2023/05/20 (Sat) 02:21" (LongMemEval)
_SLASH_DATE = re.compile(
    r"^(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\([A-Za-z]+\))?(?:\s+(\d{1,2}):(\d{2}))?$")


def parse_bench_date(text: Any) -> datetime | None:
    """A benchmark's session or question date as a UTC datetime, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    text = " ".join(text.split())
    try:
        if m := _LOCOMO_DATE.match(text):
            hour, minute, half, day, month, year = m.groups()
            hour_24 = int(hour) % 12 + (12 if half.lower() == "p" else 0)
            return datetime(int(year), _MONTHS[month.lower()], int(day), hour_24, int(minute),
                            tzinfo=timezone.utc)
        if m := _SLASH_DATE.match(text):
            year, month, day, hour, minute = m.groups()
            return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0),
                            tzinfo=timezone.utc)
        if m := _DAY_MONTH_YEAR.match(text):
            day, month, year = m.groups()
            return datetime(int(year), _MONTHS[month.lower()], int(day), tzinfo=timezone.utc)
        if m := _MONTH_DAY_YEAR.match(text):
            month, day, year = m.groups()
            return datetime(int(year), _MONTHS[month.lower()], int(day), tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(text)
    except (KeyError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read_json(path: str | os.PathLike[str]) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path}: not JSON ({exc})") from exc


def _items(data: Any, marker: str) -> list[Any]:
    """The list of samples in a file: the file itself, a list under a usual
    key, or one sample on its own."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "samples", "questions", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        if marker in data:
            return [data]
    raise FormatError(f"expected a JSON list of samples, got {type(data).__name__}")


def _text(value: Any) -> str:
    """An answer as text: LoCoMo has numbers (2022) among its answers."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, list):
        return ", ".join(_text(v) for v in value)
    return " ".join(str(value).split())


_SESSION_KEY = re.compile(r"^session_(\d+)$")
_DIA_ID = re.compile(r"D\d+:\d+")


def _canonical_dia(text: str) -> str:
    """"D01:05" and "D1:5" name the same turn."""
    m = re.fullmatch(r"\s*D(\d+):(\d+)\s*", text)
    return f"D{int(m.group(1))}:{int(m.group(2))}" if m else text.strip()


def load_locomo(path: str | os.PathLike[str]) -> list[Conversation]:
    """LoCoMo samples: turns keyed by dia_id, questions with their evidence.
    Raises FormatError on a shape it cannot read; smaller faults (an evidence
    id that names no turn, an unreadable date) go to each conversation's
    warnings."""
    return [_locomo_sample(item, i) for i, item in enumerate(_items(_read_json(path), "qa"))]


def _locomo_sample(sample: Any, index: int) -> Conversation:
    if not isinstance(sample, dict):
        raise FormatError(f"sample {index}: not an object")
    sample_id = str(sample.get("sample_id") or f"sample-{index}")
    conv = sample.get("conversation")
    if not isinstance(conv, dict):
        raise FormatError(f"{sample_id}: 'conversation' is not an object")
    numbers = sorted(int(m.group(1)) for key, value in conv.items()
                     if (m := _SESSION_KEY.match(key)) and isinstance(value, list))
    if not numbers:
        raise FormatError(f"{sample_id}: 'conversation' has no session_N list of turns")
    warnings: list[str] = []
    sessions: list[Session] = []
    known: set[str] = set()
    for n in numbers:
        sid = f"session_{n}"
        date_text = str(conv.get(f"{sid}_date_time") or conv.get(f"{sid}_date") or "").strip()
        date = parse_bench_date(date_text)
        if date is None:
            warnings.append(f"{sample_id} {sid}: no readable date ({date_text!r})")
        turns: list[Turn] = []
        for i, raw in enumerate(conv[sid], start=1):
            if not isinstance(raw, dict):
                raise FormatError(f"{sample_id} {sid} turn {i}: not an object")
            speaker = " ".join(str(raw.get("speaker") or "").split())
            said = " ".join(str(raw.get("text") or "").split())
            caption = " ".join(str(raw.get("blip_caption") or "").split())
            if caption:
                said = f"{said} [shares a photo: {caption}]".strip()
            dia_id = _canonical_dia(str(raw.get("dia_id") or f"D{n}:{i}"))
            if dia_id in known:
                raise FormatError(f"{sample_id} {sid}: dia_id {dia_id} appears twice")
            known.add(dia_id)
            if not said:
                warnings.append(f"{sample_id} {dia_id}: empty turn left out")
                continue
            turns.append(Turn(key=dia_id, role=speaker or "user", raw=said,
                              text=f"{speaker}: {said}" if speaker else said))
        sessions.append(Session(sid, date, date_text, turns))
    qa = sample.get("qa", [])
    if not isinstance(qa, list):
        raise FormatError(f"{sample_id}: 'qa' is not a list")
    questions = []
    for j, item in enumerate(qa):
        if not isinstance(item, dict) or not _text(item.get("question")):
            raise FormatError(f"{sample_id} qa {j}: no question")
        category: int | str = item.get("category", 0)
        try:
            category = int(category)
        except (TypeError, ValueError):
            category = str(category)
        name = LOCOMO_CATEGORIES.get(category, f"category {category}")  # type: ignore[arg-type]
        gold = _text(item.get("answer"))
        abstain = name in ABSTAIN_CATEGORIES or (not gold and "adversarial_answer" in item)
        if not gold:
            if not abstain:
                raise FormatError(f"{sample_id} qa {j}: no answer")
            gold = ABSTAIN_ANSWER
        evidence: list[str] = []
        unknown: list[str] = []
        raw_evidence = item.get("evidence") or []
        if isinstance(raw_evidence, str):
            raw_evidence = [raw_evidence]
        for entry in raw_evidence:
            ids = [_canonical_dia(found) for found in _DIA_ID.findall(str(entry))]
            if not ids:
                unknown.append(str(entry))
            for dia_id in ids:
                (evidence if dia_id in known else unknown).append(dia_id)
        if unknown:
            warnings.append(f"{sample_id} qa {j}: evidence {unknown} names no turn, left out")
        extra = {"adversarial_answer": _text(item["adversarial_answer"])} \
            if "adversarial_answer" in item else {}
        questions.append(Question(
            qid=f"{sample_id}/q{j}", question=_text(item["question"]), answer=gold,
            category=category, category_name=name, evidence=list(dict.fromkeys(evidence)),
            level="turn", abstain=abstain, extra=extra))
    speakers = [str(conv.get(k) or "").strip() for k in ("speaker_a", "speaker_b")]
    label = " and ".join(s for s in speakers if s)
    return Conversation(sample_id, f"conversation between {label}" if label else "conversation",
                        sessions, questions, warnings)


def load_longmemeval(path: str | os.PathLike[str]) -> list[Conversation]:
    """LongMemEval questions, each with its haystack as a conversation of its
    own; the evidence is the answer sessions. Raises FormatError on a shape it
    cannot read; smaller faults go to each conversation's warnings."""
    return [_longmemeval_item(item, i)
            for i, item in enumerate(_items(_read_json(path), "haystack_sessions"))]


def _longmemeval_item(item: Any, index: int) -> Conversation:
    if not isinstance(item, dict):
        raise FormatError(f"question {index}: not an object")
    qid = str(item.get("question_id") or f"question-{index}")
    for key in ("question", "answer", "haystack_sessions"):
        if key not in item:
            raise FormatError(f"{qid}: no '{key}'")
    raw_sessions = item["haystack_sessions"]
    if not isinstance(raw_sessions, list) or not all(isinstance(s, list) for s in raw_sessions):
        raise FormatError(f"{qid}: 'haystack_sessions' is not a list of sessions")
    n = len(raw_sessions)
    ids = item.get("haystack_session_ids") or [f"session_{i + 1}" for i in range(n)]
    dates = item.get("haystack_dates") or [""] * n
    if len(ids) != n or len(dates) != n:
        raise FormatError(f"{qid}: {n} sessions but {len(ids)} session ids and {len(dates)} dates")
    warnings: list[str] = []
    sessions: list[Session] = []
    seen: set[str] = set()
    for sid, date_text, raw_turns in zip(ids, dates, raw_sessions):
        sid = str(sid)
        if sid in seen:
            raise FormatError(f"{qid}: session id {sid} appears twice")
        seen.add(sid)
        date = parse_bench_date(date_text)
        if date is None:
            warnings.append(f"{qid} {sid}: no readable date ({date_text!r})")
        turns = []
        for t, raw in enumerate(raw_turns):
            if not isinstance(raw, dict):
                raise FormatError(f"{qid} {sid} turn {t}: not an object")
            role = str(raw.get("role") or "user").strip().lower()
            if role not in ("user", "assistant"):
                warnings.append(f"{qid} {sid} turn {t}: role {role!r} read as user")
                role = "user"
            content = str(raw.get("content") or "").strip()
            if not content:
                continue
            turns.append(Turn(key=f"{sid}#{t}", role=role, raw=content,
                              text=content if role == "user" else f"{role}: {content}",
                              has_answer=bool(raw.get("has_answer"))))
        sessions.append(Session(sid, date, str(date_text or ""), turns))
    evidence, unknown = [], []
    for sid in item.get("answer_session_ids") or []:
        (evidence if str(sid) in seen else unknown).append(str(sid))
    if unknown:
        warnings.append(f"{qid}: answer sessions {unknown} are not in the haystack, left out")
    qtype = str(item.get("question_type") or "unknown")
    question = Question(
        qid=qid, question=_text(item["question"]), answer=_text(item["answer"]),
        category=qtype, category_name=qtype, evidence=list(dict.fromkeys(evidence)),
        level="session", question_date=parse_bench_date(item.get("question_date")),
        abstain=qid.endswith("_abs"))
    if not question.question:
        raise FormatError(f"{qid}: empty question")
    return Conversation(qid, "chat between the user and an assistant", sessions, [question],
                        warnings)


LOADERS: dict[str, Callable[[Any], list[Conversation]]] = {
    "locomo": load_locomo, "longmemeval": load_longmemeval,
}


def dataset_files(dataset: str, variant: str = "s") -> list[str]:
    """Where a dataset's file may sit under the data directory, in order."""
    if dataset == "locomo":
        names = ["locomo10.json"]
        dirs = ["", "locomo/", "locomo/data/", "data/"]
    else:
        names = [f"longmemeval_{variant}.json", f"longmemeval_{variant}_cleaned.json",
                 f"longmemeval_{variant}"]
        dirs = ["", "longmemeval/", "LongMemEval/", "LongMemEval/data/", "data/"]
    return [d + n for d in dirs for n in names]


def find_dataset(dataset: str, data_dir: str | os.PathLike[str] | None, *, variant: str = "s",
                 file: str | os.PathLike[str] | None = None) -> pathlib.Path | None:
    """The dataset file: ``file`` (relative to the data directory), or the
    first of ``dataset_files`` there; None when there is none."""
    root = pathlib.Path(data_dir) if data_dir else None
    if file:
        path = pathlib.Path(file)
        if not path.is_absolute() and root is not None:
            path = root / path
        return path if path.is_file() else None
    if root is None or not root.is_dir():
        return None
    return next((root / name for name in dataset_files(dataset, variant)
                 if (root / name).is_file()), None)


# --------------------------------------------------------------------------
# into a store


@dataclass
class Ingested:
    """A conversation in its store, and how to trace a memory to its turns."""

    store: MemoryStore
    conversation: Conversation
    session_of_turn: dict[str, str]
    turn_of_episode: dict[str, str] = field(default_factory=dict)
    turns_of_memory: dict[str, set[str]] = field(default_factory=dict)
    created_at_set: bool = True
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def turns_of(self, memory: Memory) -> set[str]:
        """The turns a memory came from: the saves that landed on it, and the
        episodes it lists as its sources (which survive merges)."""
        turns = set(self.turns_of_memory.get(memory.id, ()))
        turns.update(self.turn_of_episode[e] for e in memory.source_episode_ids
                     if e in self.turn_of_episode)
        return turns

    def units_of(self, memory: Memory, level: str) -> set[str]:
        turns = self.turns_of(memory)
        return turns if level == "turn" else {self.session_of_turn[t] for t in turns}


def make_store(mode: str, embedder: Embedder, *, llm: LLM | None = None) -> MemoryStore:
    """Verbatim: a store with no model at all, default settings. Extract: the
    configured Memry (``Config.load``: LLM, decision model, retrieval
    settings), in memory, with the benchmark's embedder."""
    if mode == "verbatim":
        return MemoryStore(Config(db_path=":memory:"), llm=llm or NoneLLM(), embedder=embedder)
    return MemoryStore(Config.load(db_path=":memory:"), llm=llm, embedder=embedder)


def _set_created_at(store: MemoryStore, memory_id: str, stamp: str) -> bool:
    """Write a memory's created_at. Neither ``MemoryStore.add`` nor the
    backend's ``update_memory`` takes one (add stamps the wall clock), so on a
    LocalBackend the row is written directly; False on any other backend."""
    backend = store.backend
    db, lock = getattr(backend, "_db", None), getattr(backend, "_lock", None)
    if not isinstance(db, sqlite3.Connection) or lock is None:
        return False
    with lock:
        cur = db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (stamp, memory_id))
        db.commit()
    return cur.rowcount == 1


@contextmanager
def extraction_clock(moment: datetime | None) -> Iterator[None]:
    """While it is open, extraction's "today" is ``moment``. Extraction
    resolves "yesterday" against ``datetime.now()`` and ``add`` takes no
    reference time, so without this a 2023 session's dates would come out in
    the year the benchmark runs. Does nothing when the module has no clock to
    replace."""
    from memry.intelligence import extraction

    real = getattr(extraction, "datetime", None)
    if moment is None or not (isinstance(real, type) and issubclass(real, datetime)):
        yield
        return

    class SessionClock(real):  # type: ignore[misc,valid-type]
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz) if tz else moment.replace(tzinfo=None)

    extraction.datetime = SessionClock
    try:
        yield
    finally:
        extraction.datetime = real


def _record(ingested: Ingested, action: AddAction, session: Session, keys: list[str],
            moment: datetime | None, dataset: str, when: str) -> None:
    """Trace the memory an action landed on to its turns and give it its
    session's time."""
    store = ingested.store
    ingested.turns_of_memory.setdefault(action.memory_id, set()).update(keys)
    memory = store.backend.get_memory(action.memory_id)
    if memory is None:
        return
    meta = dict(memory.metadata or {})
    bench = dict(meta.get("bench") or {})
    bench.setdefault("dataset", dataset)
    bench.setdefault("conversation", ingested.conversation.conv_id)
    bench.setdefault("session_id", session.session_id)
    bench.setdefault("session_date", session.date_text)
    bench["sessions"] = list(dict.fromkeys([*bench.get("sessions", []), session.session_id]))
    bench["turns"] = list(dict.fromkeys([*bench.get("turns", []), *keys]))
    meta["bench"] = bench
    new = action.event in ("ADD", "DELETE")  # DELETE: a new memory replaced an old one
    stamped = False
    if moment is not None and action.event != "NONE":
        stamp = moment.isoformat(timespec="seconds")
        if new:
            stamped = _set_created_at(store, memory.id, stamp)
            ingested.created_at_set = ingested.created_at_set and stamped
        store.backend.set_memory_timestamp(memory.id, stamp)  # updated_at
    wants_when = when == "always" or (when == "fallback" and not stamped)
    if moment is not None and new and wants_when and not meta.get("when"):
        meta["when"] = {"start": moment.date().isoformat()}
    store.backend.update_memory(memory.id, metadata=meta, touch=False)


def ingest(store: MemoryStore, conversation: Conversation, *, mode: str = "verbatim",
           unit: str = "session", when: str = "fallback", dataset: str = "") -> Ingested:
    """Save a conversation session by session, as ``mode`` says (see the
    module docstring). The turns of a session are a second apart."""
    if mode not in INGEST_MODES or unit not in EXTRACT_UNITS or when not in WHEN_POLICIES:
        raise ValueError(f"mode {mode!r}, unit {unit!r}, when {when!r}")
    ingested = Ingested(store, conversation, session_of_turn={
        t.key: s.session_id for s in conversation.sessions for t in s.turns})
    started = time.perf_counter()
    for session in conversation.sessions:
        whole = mode == "extract" and unit == "session"
        groups = [session.turns] if whole and session.turns else [[t] for t in session.turns]
        offset = 0
        for group in groups:
            moment = session.date + timedelta(seconds=offset) if session.date else None
            offset += len(group)
            keys = [t.key for t in group]
            meta = {"bench": {"dataset": dataset, "conversation": conversation.conv_id,
                              "session_id": session.session_id,
                              "session_date": session.date_text, "turns": keys}}
            if mode == "verbatim":
                result = store.add(group[0].text, user_id=BENCH_USER, run_id=session.session_id,
                                   metadata=meta, infer=False, memory_type="episodic")
            else:
                meta["context"] = f"{conversation.label}, {session.date_text}".strip(", ")
                with extraction_clock(moment):
                    result = store.add([{"role": t.role, "content": t.raw} for t in group],
                                       user_id=BENCH_USER, run_id=session.session_id,
                                       metadata=meta, infer=True)
            if len(result.episode_ids) != len(group):
                ingested.warnings.append(
                    f"{conversation.conv_id} {keys[0]}: {len(group)} turns but "
                    f"{len(result.episode_ids)} episodes; turns traced by save only")
            else:
                ingested.turn_of_episode.update(zip(result.episode_ids, keys))
            for action in result.actions:
                if action.memory_id:
                    _record(ingested, action, session, keys, moment, dataset, when)
            ingested.warnings.extend(f"{conversation.conv_id} {keys[0]}: {w}"
                                     for w in result.warnings)
    ingested.seconds = time.perf_counter() - started
    return ingested


# --------------------------------------------------------------------------
# scoring


def evidence_recall(units: list[set[str]], evidence: list[str], k: int) -> float | None:
    """Share of the evidence units that some memory among the first k (each
    given as the units it comes from) comes from; None without evidence."""
    wanted = set(evidence)
    if not wanted:
        return None
    found: set[str] = set().union(*units[:k])
    return len(wanted & found) / len(wanted)


def reciprocal_rank(units: list[set[str]], evidence: list[str]) -> float | None:
    """1 / rank of the first memory that comes from an evidence unit, 0 when
    none does; None without evidence."""
    wanted = set(evidence)
    if not wanted:
        return None
    return next((1.0 / rank for rank, got in enumerate(units, start=1) if got & wanted), 0.0)


_PUNCTUATION = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: Any) -> str:
    """SQuAD's normalisation: lower case, no punctuation, no articles, single
    spaces."""
    text = _ARTICLES.sub(" ", _text(text).lower().translate(_PUNCTUATION))
    return " ".join(text.split())


def token_f1(prediction: Any, gold: Any) -> float:
    """Token F1 between normalised answers (SQuAD, LoCoMo). Two empty answers
    agree; one empty answer scores 0."""
    pred, true = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not true:
        return float(pred == true)
    common = sum((Counter(pred) & Counter(true)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(true)
    return 2 * precision * recall / (precision + recall)


def parts_f1(prediction: Any, gold: Any) -> float:
    """LoCoMo's multi-hop F1: each comma-separated part of the gold answer
    scores its best part of the prediction, averaged over the gold parts."""
    preds = [p.strip() for p in _text(prediction).split(",")]
    golds = [g.strip() for g in _text(gold).split(",") if g.strip()] or [""]
    return statistics.mean(max(token_f1(p, g) for p in preds) for g in golds)


def exact_match(prediction: Any, gold: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def answer_contained(prediction: Any, gold: Any) -> bool:
    """The normalised gold answer occurs as whole words in the prediction."""
    true = normalize_answer(gold)
    return bool(true) and f" {true} " in f" {normalize_answer(prediction)} "


_ABSTAINS = re.compile(
    r"no information|not enough information|information provided is not enough"
    r"|not mentioned|no mention|never mentioned"
    r"|(?:do not|don't|does not|doesn't|did not|didn't) (?:know|say|mention|specify)"
    r"|cannot (?:be )?(?:answer|determine|tell)|can't (?:answer|determine|tell)"
    r"|unanswerable|not (?:specified|stated|provided|available)",
    re.IGNORECASE)


def is_abstention(text: Any) -> bool:
    """Says that the conversation does not tell (LoCoMo checks "no information
    available" and "not mentioned"; this reads a few more wordings)."""
    return bool(_ABSTAINS.search(_text(text)))


#: A judge reads (question, gold answer, prediction) and says whether the
#: prediction is right: LongMemEval's metric is an LLM judge's accuracy.
Judge = Callable[[str, str, str], bool]


def containment_judge(question: str, gold: str, prediction: str) -> bool:
    """The default judge, needing no model: the gold answer is contained in
    the prediction, or both say the conversation does not tell."""
    if is_abstention(gold):
        return is_abstention(prediction)
    return answer_contained(prediction, gold)


def load_judge(spec: str | None) -> Judge:
    """``module:function`` -> that function; None -> ``containment_judge``."""
    if not spec:
        return containment_judge
    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise ValueError(f"--judge wants module:function, got {spec!r}")
    judge = getattr(importlib.import_module(module_name), name)
    if not callable(judge):
        raise ValueError(f"{spec} is not callable")
    return judge


def score_answer(prediction: str, question: Question,
                 judge: Judge = containment_judge) -> dict[str, Any]:
    """f1, em, contains and the judge's verdict for one answer."""
    gold = question.answer
    if question.abstain:
        right = float(is_abstention(prediction))
        f1, em, contains = right, right, right
    else:
        if question.category_name == "open-domain":
            gold = gold.split(";")[0].strip()  # LoCoMo scores its first alternative
        f1 = parts_f1(prediction, gold) if question.category_name == "multi-hop" \
            else token_f1(prediction, gold)
        em, contains = float(exact_match(prediction, gold)), float(answer_contained(prediction, gold))
    verdict = bool(judge(question.question, question.answer, prediction))
    return {"f1": round(f1, 4), "em": em, "contains": contains, "judge": verdict}


ANSWER_SYSTEM = """You answer a question about past conversations from memories \
retrieved for it. Each memory starts with the date it was said. Use only the \
memories. Resolve relative dates ("yesterday", "last week") against the date of \
the memory that uses them. Answer with a short phrase, using the memories' own \
words where you can, not a full sentence. If the memories do not say, answer \
exactly: No information available."""


def memories_text(memories: list[Memory]) -> str:
    lines = []
    for memory in memories:
        when = ((memory.metadata or {}).get("when") or {}).get("start")
        lines.append(f"- [{when or (memory.created_at or '')[:10]}] {memory.content}")
    return "Memories:\n" + "\n".join(lines) if lines else "Memories: (none found)"


def answer_question(llm: LLM, question: Question, context: str) -> str:
    today = f"Today is {question.question_date:%d %B %Y}.\n" if question.question_date else ""
    raw = llm.complete(ANSWER_SYSTEM, f"{context}\n\n{today}Question: {question.question}\n"
                                      "Short answer:")
    return " ".join(str(raw or "").split())


def ask(ingested: Ingested, question: Question, *, k: int = 10, answer_llm: LLM | None = None,
        judge: Judge = containment_judge, use_context: bool = False) -> dict[str, Any]:
    """Search for one question, score what came back, answer when asked."""
    store = ingested.store
    started = time.perf_counter()
    results = store.search(question.question, user_id=BENCH_USER, limit=max(DEPTH, k))
    ms = (time.perf_counter() - started) * 1000
    results = results[:max(DEPTH, k)]
    units = [ingested.units_of(r.memory, question.level) for r in results]
    row: dict[str, Any] = {
        "conversation": ingested.conversation.conv_id, "qid": question.qid,
        "category": question.category, "category_name": question.category_name,
        "question": question.question, "answer": question.answer, "level": question.level,
        "evidence": question.evidence, "abstain": question.abstain,
        **question.extra,
        "search_ms": round(ms, 3),
        "evidence_ranks": [rank for rank, got in enumerate(units, start=1)
                           if got & set(question.evidence)],
        "retrieved": [sorted(ingested.turns_of(r.memory)) for r in results[:k]],
    }
    for depth in KS:
        row[f"recall@{depth}"] = evidence_recall(units, question.evidence, depth)
    row["mrr"] = reciprocal_rank(units, question.evidence)
    if answer_llm is None:
        return row
    if use_context:
        context = store.reconstruct_context(question.question, user_id=BENCH_USER, limit=k,
                                            token_budget=CONTEXT_TOKENS)
        text = context.text or "Memories: (none found)"
        in_context = [store.get(mid) for mid in context.memory_ids]
        row["context_recall"] = evidence_recall(
            [ingested.units_of(m, question.level) for m in in_context if m],
            question.evidence, len(in_context))
    else:
        text = memories_text([r.memory for r in results[:k]])
    try:
        row["prediction"] = answer_question(answer_llm, question, text)
    except Exception as exc:  # one failed call must not end a long run
        row["prediction"], row["answer_error"] = "", str(exc)[:300]
    row.update(score_answer(row["prediction"], question, judge))
    return row


METRICS = ("recall@5", "recall@10", "recall@20", "mrr", "context_recall",
           "f1", "em", "contains", "judge")


def _mean(values: Any) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return round(statistics.mean(kept), 4) if kept else None


def _summary(category: Any, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"category": category, "name": name, "n": len(rows),
                           "with_evidence": sum(1 for r in rows if r.get("evidence"))}
    for metric in METRICS:
        if any(metric in r for r in rows):
            out[metric] = _mean(r.get(metric) for r in rows)
    times = [r["search_ms"] for r in rows if r.get("search_ms") is not None]
    out["search_ms"] = round(statistics.median(times), 3) if times else None
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The metrics' means per category (number and name) and overall, and the
    median search time. A metric a row does not have (null) is left out of
    its mean."""
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["category"], row["category_name"]), []).append(row)
    order = sorted(groups, key=lambda key: (not isinstance(key[0], int), str(key[0]).zfill(6)))
    return {"by_category": [_summary(c, n, groups[(c, n)]) for c, n in order],
            "overall": _summary("all", "overall", rows)}


def markdown_table(tables: dict[str, Any]) -> str:
    rows = [*tables["by_category"], tables["overall"]]
    metrics = [m for m in METRICS if any(r.get(m) is not None for r in rows)]
    header = ["category", "name", "n", *metrics, "search ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(row["category"]), row["name"], str(row["n"])]
        cells += ["-" if row.get(m) is None else f"{row[m]:.3f}" for m in metrics]
        cells.append("-" if row.get("search_ms") is None else f"{row['search_ms']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# embeddings


class SqliteEmbeddingCache(Embedder):
    """One vector per distinct text, kept in a SQLite file, fetched in batches
    of 256: a re-run, and the many LongMemEval haystacks that share sessions,
    embed each text once. ``close`` keeps it open, because every store of a
    run closes its embedder; ``release`` closes it."""

    def __init__(self, base: Embedder, path: str | os.PathLike[str]) -> None:
        self.base = base
        self.name, self._model, self.dimensions = base.name, base._model, base.dimensions
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, "
                        "vector BLOB NOT NULL)")
        self.base_calls = 0

    @property
    def model_id(self) -> str:
        return self.base.model_id

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.base.model_id}\0{text}".encode()).hexdigest()

    def _vectors(self, texts: list[str]) -> dict[str, list[float]]:
        keys = {self._key(t): t for t in texts}
        found: dict[str, list[float]] = {}
        wanted = list(keys)
        for i in range(0, len(wanted), 500):
            chunk = wanted[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for key, blob in self.db.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({marks})", chunk):
                found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        missing = [(key, text) for key, text in keys.items() if key not in found]
        for i in range(0, len(missing), 256):
            batch = missing[i:i + 256]
            vectors = self.base.embed([text for _, text in batch])
            self.base_calls += 1
            self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", [
                (key, np.asarray(v, dtype=np.float32).tobytes())
                for (key, _), v in zip(batch, vectors)])
            self.db.commit()
            found.update({key: [float(x) for x in v] for (key, _), v in zip(batch, vectors)})
        return found

    def warm(self, texts: list[str]) -> None:
        self._vectors(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        found = self._vectors(texts)
        return [found[self._key(t)] for t in texts]

    def close(self) -> None:
        return None

    def release(self) -> None:
        self.db.close()
        self.base.close()


def build_embedder(kind: str, data_dir: str | os.PathLike[str] | None) -> Embedder:
    """hash: the local hashing embedder (costs nothing). openai: the configured
    OpenAI embedding model (``OPENAI_API_KEY``), cached under
    <data>/cache/."""
    if kind == "hash":
        return HashEmbedder(256)
    if kind != "openai":
        raise ValueError(f"embedder {kind!r}")
    cfg = Config.load().embedding
    if cfg.provider != "openai":
        cfg = EmbeddingConfig(provider="openai")
    if not (cfg.api_key or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("--embedder openai needs OPENAI_API_KEY")
    base = OpenAIEmbedder(cfg)
    root = pathlib.Path(data_dir) if data_dir else pathlib.Path(tempfile.gettempdir())
    safe = re.sub(r"[^A-Za-z0-9.@-]+", "_", base.model_id)
    return SqliteEmbeddingCache(base, root / "cache" / f"embeddings_{safe}.sqlite")


# --------------------------------------------------------------------------
# a run


def run_benchmark(conversations: list[Conversation], *, dataset: str, mode: str = "verbatim",
                  unit: str = "session", embedder: Embedder | None = None, k: int = 10,
                  questions: int | None = None, answer_llm: LLM | None = None,
                  judge: Judge = containment_judge, use_context: bool = False,
                  when: str = "fallback",
                  store_factory: Callable[[], MemoryStore] | None = None,
                  log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Ingest each conversation into a fresh store, ask its questions, and
    return {config, stores, rows, tables, warnings, notes}."""
    embedder = embedder or HashEmbedder(256)
    log = log or (lambda text: print(text, file=sys.stderr, flush=True))
    rows: list[dict[str, Any]] = []
    stores: list[dict[str, Any]] = []
    warnings: list[str] = []
    notes: list[str] = []
    for conv in conversations:
        warnings.extend(conv.warnings)
        asked = conv.questions[:questions] if questions else conv.questions
        if mode == "verbatim" and hasattr(embedder, "warm"):
            embedder.warm([t.text for t in conv.turns] + [q.question for q in asked])
        store = store_factory() if store_factory else make_store(mode, embedder)
        try:
            if mode == "extract" and not store.llm.available:
                raise SystemExit("--ingest extract needs a configured LLM "
                                 "(OPENAI_API_KEY, ANTHROPIC_API_KEY or MEMRY_LLM_PROVIDER)")
            ingested = ingest(store, conv, mode=mode, unit=unit, when=when, dataset=dataset)
            memories = len(store.get_all(user_id=BENCH_USER, limit=1_000_000))
            stores.append({"conversation": conv.conv_id, "sessions": len(conv.sessions),
                           "turns": len(conv.turns), "memories": memories,
                           "questions": len(asked), "ingest_seconds": round(ingested.seconds, 2),
                           "created_at_set": ingested.created_at_set})
            warnings.extend(ingested.warnings)
            for question in asked:
                rows.append(ask(ingested, question, k=k, answer_llm=answer_llm, judge=judge,
                                use_context=use_context))
        finally:
            store.close()
        log(f"  {conv.conv_id}: {len(conv.turns)} turns -> {memories} memories in "
            f"{ingested.seconds:.1f}s, {len(asked)} questions")
    if mode == "extract" and unit == "session":
        notes.append("extract by session: a memory counts for every turn of the session it "
                     "came from, so turn-level recall is session-level recall")
    if stores and not all(s["created_at_set"] for s in stores):
        notes.append("created_at could not be written on this backend: the session date is "
                     "in updated_at and metadata['when'] only")
    return {
        "dataset": dataset,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"ingest": mode, "extract_unit": unit if mode == "extract" else None,
                   "embedder": embedder.model_id, "k": k, "depth": max(DEPTH, k),
                   "conversations": len(conversations), "questions_per_conversation": questions,
                   "answer_llm": getattr(answer_llm, "name", None) if answer_llm else None,
                   "judge": getattr(judge, "__qualname__", repr(judge)),
                   "context": use_context, "when": when,
                   "locomo_categories": LOCOMO_CATEGORIES if dataset == "locomo" else None},
        "stores": stores,
        "tables": aggregate(rows),
        "rows": rows,
        "warnings": warnings,
        "notes": notes,
    }


def load(dataset: str, path: str | os.PathLike[str]) -> list[Conversation]:
    if dataset not in LOADERS:
        raise ValueError(f"dataset {dataset!r}: one of {', '.join(DATASETS)}")
    return LOADERS[dataset](path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.external_benchmarks",
        description=f"Run Memry on LoCoMo or LongMemEval; the data directory is ${DATA_ENV}.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--file", default=None,
                        help=f"the dataset file (relative paths start at ${DATA_ENV}); "
                             "default: the usual file name there")
    parser.add_argument("--variant", default="s", choices=["s", "m", "oracle"],
                        help="LongMemEval file: longmemeval_<variant>.json (default s)")
    parser.add_argument("--ingest", choices=INGEST_MODES, default="verbatim")
    parser.add_argument("--extract-unit", choices=EXTRACT_UNITS, default="session",
                        help="with --ingest extract: one save a session (default) or a turn")
    parser.add_argument("--embedder", choices=["hash", "openai"], default="hash")
    parser.add_argument("--k", type=int, default=10,
                        help="memories given to the answering model (default 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N conversations (LoCoMo samples, LongMemEval questions)")
    parser.add_argument("--questions", type=int, default=None,
                        help="first N questions of each conversation")
    parser.add_argument("--seed", type=int, default=None,
                        help="shuffle the conversations with this seed before --limit")
    parser.add_argument("--answer", action="store_true",
                        help="answer with the configured LLM and score F1, EM, contains, judge")
    parser.add_argument("--context", action="store_true",
                        help="with --answer: answer from reconstruct_context, not the top k")
    parser.add_argument("--judge", default=None,
                        help="module:function(question, gold, prediction) -> bool "
                             "(default: containment)")
    parser.add_argument("--when", choices=WHEN_POLICIES, default="fallback",
                        help="write metadata['when'] = session date: when created_at cannot "
                             "be written (default), on every new memory, or never")
    parser.add_argument("--out", default=None,
                        help=f"results file (default ${DATA_ENV}/results/<dataset>_<time>.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = os.environ.get(DATA_ENV, "").strip() or None
    path = find_dataset(args.dataset, data_dir, variant=args.variant, file=args.file)
    if path is None:
        where = f"{DATA_ENV}={data_dir}" if data_dir else f"{DATA_ENV} is not set"
        wanted = args.file or " or ".join(dataset_files(args.dataset, args.variant)[:2])
        print(f"no {args.dataset} data ({where}): expected {wanted}", file=sys.stderr)
        return 2
    conversations = load(args.dataset, path)
    if args.seed is not None:
        random.Random(args.seed).shuffle(conversations)
    if args.limit:
        conversations = conversations[:args.limit]
    judge = load_judge(args.judge)
    answer_llm, notes = None, []
    if args.answer:
        llm = build_llm(Config.load().llm)
        if llm.available:
            answer_llm = llm
        else:
            notes.append("--answer skipped: no LLM configured (OPENAI_API_KEY, "
                         "ANTHROPIC_API_KEY or MEMRY_LLM_PROVIDER)")
            print(notes[-1], file=sys.stderr)
    embedder = build_embedder(args.embedder, data_dir)
    print(f"{args.dataset}: {path} ({len(conversations)} conversations), ingest {args.ingest}, "
          f"embedder {embedder.model_id}", file=sys.stderr, flush=True)
    try:
        result = run_benchmark(
            conversations, dataset=args.dataset, mode=args.ingest, unit=args.extract_unit,
            embedder=embedder, k=args.k, questions=args.questions, answer_llm=answer_llm,
            judge=judge, use_context=args.context, when=args.when)
    finally:
        if isinstance(embedder, SqliteEmbeddingCache):
            embedder.release()
        if answer_llm is not None:
            answer_llm.close()
    result["file"] = str(path)
    result["config"].update(limit=args.limit, seed=args.seed, variant=args.variant
                            if args.dataset == "longmemeval" else None)
    result["notes"] = notes + result["notes"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = pathlib.Path(data_dir) if data_dir else path.parent
    out = pathlib.Path(args.out) if args.out else base / "results" / f"{args.dataset}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"\n## {args.dataset}: {len(result['rows'])} questions, "
          f"{sum(s['memories'] for s in result['stores'])} memories\n")
    print(markdown_table(result["tables"]))
    for note in result["notes"]:
        print(f"\nnote: {note}")
    if result["warnings"]:
        print(f"\n{len(result['warnings'])} warnings (in the results file); first: "
              f"{result['warnings'][0]}")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
