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

In both modes a save reconciled as NONE (a restated turn) credits its turns to
the memory it landed on, which counts as retrieved for them; every results
file says so in its notes.

Every save carries its session's date: ``add(created_at=...)`` makes it the
episodes' and new memories' created_at, updated_at and valid_from (and the
updated_at of a memory a save rewrites), and ``add(now=...)`` makes it the day
extraction resolves "yesterday" against. With --when always a new memory
without a "when" of its own gets ``metadata["when"] = {"start": "YYYY-MM-DD"}``.
``metadata["bench"]`` (``add(memory_metadata=...)``) names the conversation,
session and turns a new memory came from.

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
answer is scored against the gold one. The model is shown what Memry's
context builder gives an agent for the question (``intelligence.context.
context_lines``): the descriptions of the entities the question names
(``MemoryStore.described_entities``, built by the store's text model where
stale and counted under the stage "describe"; --no-descriptions leaves them
out), then the memories, each with the date its event happened, where known,
and the date it was said, then their source turns that best match the
question (``MemoryStore.evidence``, within the store's
``retrieval.evidence_tokens``; --evidence-tokens sets it, 0 shows none).
--compare-evidence-tokens N answers
each question a second time from the same search, its turns chosen within N
tokens (0: the memories alone), with no further search: those answers are
scored and judged as the others, kept under the row's "answers_compared" and
counted under the stages "answer:compared" and "judge:compared".
--compare-answer-model MODEL has MODEL write that second answer, from the
same memory list (the turns within N tokens where both are given), the way
published LoCoMo rows compare answer models over one memory:

  f1        token F1 after SQuAD normalisation (lower case, punctuation and
            the articles a/an/the removed); by LoCoMo's rules, a multi-hop
            gold answer is scored by its comma-separated parts and an
            open-domain one by its first ";"-separated alternative
  em        normalised exact match
  contains  the normalised gold answer occurs, as whole words, in the answer
  judge     ``Judge(question, gold, prediction) -> bool``, given the gold f1
            reads (an open-domain one's first alternative); the default
            (``containment_judge``) is "contains", or both abstaining.
            LongMemEval's LLM judge plugs in here: --judge module:function

  f1_mem0   Mem0's token F1 (its evaluation/metrics/utils.py): sets of
            lower-case tokens split on white space and . , ! ?, the gold whole
  bleu1     Mem0's BLEU-1: NLTK's sentence_bleu of lower-cased word_tokenize
            tokens, weights (1, 0, 0, 0), smoothing method 1 (memry[eval])

A question whose right answer is that the conversation does not say
(LoCoMo's adversarial ones, LongMemEval's "_abs") scores 1 on f1, em and
contains when the answer abstains ("No information available"). The overall
row leaves LoCoMo's adversarial category out, as the published results do;
--categories 1,2,3,4 does not ask it at all.

--ks 10,20,30 answers each question from the top 10, 20 and 30 of the one
search it made; --k names the headline k of the tables, and every answer is
kept under the row's "answers" with its seconds and the cl100k_base tokens
of the memories it was shown (tiktoken, memry[eval]). --judge-runs 3 judges
each answer three times: the tables give each run's share right
("judge_runs") and their standard deviation. A row keeps the text, times
and score of every memory the search returned and the question's reference
date. --audit errors.json (dial481/locomo-audit's corrected answer key)
judges the questions it corrects against the corrected answer as well:
"judge_corrected" is the score with the corrections, "judge_clean" the score
on the questions it leaves alone. --full-context answers from the whole
conversation, with no store and no search (Mem0's full-context baseline with
evals.mem0_judge:full_context_messages). --decider jev makes Jev the store's
decision provider at the save and at the question (TYPESAFE_API_KEY);
--store-dir keeps each store in a file. --workers N makes a question's answer
and judge calls N at a time. --export-mem0 writes the headline answers as
Mem0's per-question results file.

--answer-model answers through that OpenAI chat model at temperature 0 in
place of the configured LLM, and --answer-prompt module:function writes the
answering call's messages from the question and the top k memories.
``evals/mem0_judge.py`` holds Mem0's answer prompt and LLM judge:

    ... --answer-model gpt-4o-mini --answer-prompt evals.mem0_judge:answer_messages \\
        --judge evals.mem0_judge:judge

--search-decider says what decides at question time, once the conversation
is loaded: "store" (the default) the store's own decision provider, "none"
no provider (the text and linked ranking without a judgement), "jev" Jev
(``TYPESAFE_API_KEY``), so that the linked search is judged. Given more than
once, every question is asked once per value on the same store; each row
names its pass ("search_decider") and ``passes`` holds each pass's tables.

--jobs N runs the conversations in N processes, one store each, and
--results-dir DIR keeps one results file per conversation there
(<conversation>.json): a conversation whose file is complete is not run
again, so an interrupted run resumes where it stopped.

--usage-db FILE counts, times and records every model call in a SQLite
ledger the processes share (``evals/api_usage.py``): the tokens each
response reports, per stage (extraction, reconcile, audit, search, answer,
judge, ...) and conversation. --max-calls GROUP=N stops the run before a
call that would pass N calls of that group (chat, embeddings, jev). The
harness tries a failed call again (a timeout, a rate limit, a server
error), up to four calls in all; every try is counted.

Every run writes beside its results file (x.json) a metadata file
(x.meta.json, ``run_meta``) and prints the same as the first lines of its
tables (``meta_lines``): the data file and its sha256, the questions asked
and how they were chosen, the memry commit and the tracked files that differ
from it, the settings (ingest, text model, embedder, decider, k, evidence
tokens, answer models and prompt, judge, question keys, entity questions),
the start, end and wall seconds, and the headline numbers: J of each answer
model at the headline k, recall@5, 10 and 20, the median and 95th
percentile of a search's time, and the mean context tokens. With
--usage-db it adds the ledger's calls and tokens per stage and model, the
model each reply named (a snapshot, or the version behind "jev-latest"),
and the dollars per model and provider at the prices of --prices (default
``evals/prices.json``, USD per million tokens with the day each was read).
The ledger is the whole ledger, so a resumed run counts every segment.

LongMemEval: each question is a conversation of its own, so each question has
its own store, its haystack saved session by session in time order, each
session dated with its time (``load_longmemeval``). ``evals/longmemeval_judge.py``
contains the official reading prompt and judge (one prompt per question type
and one for the abstention questions). --sample N: N questions drawn in
proportion to the question types (``stratified_sample``, seeded with --seed).
--export-longmemeval PATH: the answers in the jsonl form of the official
evaluate_qa.py. The tables have a row for the abstention questions and the
task-averaged judge score, as in print_qa_metrics.py. The store's owner is
"the user": Memry's own rule for a conversation between the user and an
assistant (``MemoryStore.owner_name``), so no name is set. With
``retrieval.entity_questions`` on, the store first describes each hub related
to the owner that has no description yet, at most 20 a question
(``describe_related_hubs_step``, counted under the stage "entity_descriptions"),
and then writes its entity questions once the haystack is loaded and before
the question is asked (``write_entity_questions_step``, counted under the stage
"entity_questions"). Both are capped with the chat calls. Each row has the store's memories with
question keys, its entities with entity questions, and where the search
started (``search_start``):

    ... --dataset longmemeval --file longmemeval_s_cleaned.json --sample 100 --seed 1 \
        --ingest extract --embedder openai --decider jev --k 20 \
        --answer-model gpt-4o-mini --compare-answer-model gpt-6-luna \
        --answer-prompt evals.longmemeval_judge:answer_messages \
        --judge evals.longmemeval_judge:judge

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
import contextvars
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
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))

from evals import api_usage  # noqa: E402
from memry.config import Config, DecisionConfig, EmbeddingConfig  # noqa: E402
from memry.intelligence.context import context_lines  # noqa: E402
from memry.intelligence.entities import DESCRIPTION_MIN_MEMORIES  # noqa: E402
from memry.intelligence.graph_retrieval import detect_query_entities  # noqa: E402
from memry.models import Memory, Scope  # noqa: E402
from memry.providers.decisions import (  # noqa: E402
    Decider,
    JevDecider,
    NoneDecider,
    build_decider,
)
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
WHEN_POLICIES = ("never", "always")
#: What decides at question time (--search-decider; see the module docstring).
SEARCH_DECIDERS = ("store", "none", "jev")
#: A provider call is tried again after these replies, and after a timeout
#: or a dropped connection, up to ``ATTEMPTS`` calls in all.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
ATTEMPTS = 4


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
    plain: str | None = None  # LoCoMo: the text as the file has it, no photo caption

    @property
    def said(self) -> str:
        """What was said, as the file has it, without a shared photo's caption."""
        return self.raw if self.plain is None else self.plain


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
    #: the date the question is asked at (LoCoMo: the last session's, as
    #: memory-benchmarks takes it; LongMemEval: the question's), ISO 8601
    reference_date: str | None = None


@dataclass
class Conversation:
    conv_id: str
    label: str                # said to the extractor with each save
    sessions: list[Session]
    questions: list[Question]
    warnings: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)  # LoCoMo: the two people talking

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
                              text=f"{speaker}: {said}" if speaker else said,
                              plain=str(raw.get("text") or "")))
        sessions.append(Session(sid, date, date_text, turns))
    dated = [s.date for s in sessions if s.date is not None]
    reference = max(dated).isoformat() if dated else None
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
            level="turn", abstain=abstain, extra=extra, reference_date=reference))
    speakers = [str(conv.get(k) or "").strip() for k in ("speaker_a", "speaker_b")]
    label = " and ".join(s for s in speakers if s)
    return Conversation(sample_id, f"conversation between {label}" if label else "conversation",
                        sessions, questions, warnings, [s for s in speakers if s])


def load_longmemeval(path: str | os.PathLike[str]) -> list[Conversation]:
    """LongMemEval questions, each with its haystack as a conversation of its
    own; the evidence is the answer sessions. The sessions are in time order
    (the file's order where two have the same time): the authors write in the
    README that the files are sorted, but the haystacks of the
    temporal-reasoning and knowledge-update questions are not. A session id
    that comes again (the same filler session at another date) is kept as
    "<id>~2", "<id>~3", ... An abstention question (its id ends in "_abs")
    has no evidence, because the authors leave those questions out of the
    retrieval scores; its answer sessions are kept under
    ``extra["answer_session_ids"]``. ``extra["question_date"]``
    is the question's date as the file writes it. Raises FormatError on a
    shape it cannot read; smaller faults go to each conversation's
    warnings."""
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
    again: Counter = Counter()
    for sid, date_text, raw_turns in zip(ids, dates, raw_sessions):
        sid = str(sid)
        if sid in seen:
            again[sid] += 1
            kept = f"{sid}~{again[sid] + 1}"
            warnings.append(f"{qid}: session id {sid} comes again, kept as {kept}")
            sid = kept
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
    in_time = sorted(sessions, key=lambda s: s.date or datetime.max.replace(tzinfo=timezone.utc))
    moved = sum(a is not b for a, b in zip(sessions, in_time))
    if moved:
        warnings.append(f"{qid}: sessions saved in time order, {moved} of {n} "
                        "not where the file has them")
    sessions = in_time
    evidence, unknown = [], []
    for sid in item.get("answer_session_ids") or []:
        (evidence if str(sid) in seen else unknown).append(str(sid))
    if unknown:
        warnings.append(f"{qid}: answer sessions {unknown} are not in the haystack, left out")
    qtype = str(item.get("question_type") or "unknown")
    asked_on = parse_bench_date(item.get("question_date"))
    abstain = qid.endswith("_abs")
    extra: dict[str, Any] = {"question_date": str(item.get("question_date") or "")}
    if abstain:
        extra["answer_session_ids"] = list(dict.fromkeys(evidence))
        evidence = []
    question = Question(
        qid=qid, question=_text(item["question"]), answer=_text(item["answer"]),
        category=qtype, category_name=qtype, evidence=list(dict.fromkeys(evidence)),
        level="session", question_date=asked_on, abstain=abstain, extra=extra,
        reference_date=asked_on.isoformat() if asked_on else None)
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


def stratum(conversation: Conversation) -> str:
    """The stratum ``stratified_sample`` draws a one-question conversation
    from: its question's category, with "/abstention" for a question whose
    right answer is that the conversation does not say."""
    if len(conversation.questions) != 1:
        raise ValueError(f"{conversation.conv_id}: {len(conversation.questions)} questions; "
                         "a stratified sample draws conversations of one question "
                         "(LongMemEval)")
    question = conversation.questions[0]
    return question.category_name + ("/abstention" if question.abstain else "")


def stratified_sample(conversations: list[Conversation], n: int, seed: int = 0
                      ) -> list[Conversation]:
    """``n`` of the one-question ``conversations``, each stratum (``stratum``)
    given its share of ``n`` (largest remainder; ties to the larger stratum,
    then by name), drawn with ``random.Random(seed)`` from each stratum's
    conversations sorted by id, the strata in order of name. The sample is
    the same whatever order the conversations come in; it keeps theirs."""
    if n < 1:
        raise ValueError("a sample of at least 1")
    groups: dict[str, list[Conversation]] = {}
    for conv in conversations:
        groups.setdefault(stratum(conv), []).append(conv)
    total = len(conversations)
    if n >= total:
        return list(conversations)
    shares = {name: n * len(group) / total for name, group in groups.items()}
    quota = {name: int(share) for name, share in shares.items()}
    for name in sorted(groups, key=lambda g: (-(shares[g] - quota[g]), -len(groups[g]), g)
                       )[:n - sum(quota.values())]:
        quota[name] += 1
    rng = random.Random(seed)
    chosen: set[str] = set()
    for name in sorted(groups):
        ids = sorted(c.conv_id for c in groups[name])
        chosen.update(rng.sample(ids, quota[name]))
    return [c for c in conversations if c.conv_id in chosen]


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
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    actions: Counter = field(default_factory=Counter)  # the saves' actions (ADD, UPDATE, ...)

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


def transient(exc: BaseException) -> bool:
    """A provider error worth another try: a timeout, a dropped connection,
    or a reply in ``RETRY_STATUSES``."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRY_STATUSES


def retried(call: Callable[[], Any], attempts: int = ATTEMPTS) -> Any:
    """``call()``, tried again after a ``transient`` error, waiting 1, 2, 4 s."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if attempt + 1 >= attempts or not transient(exc):
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("the loop returns or raises")


class RetryingLLM(LLM):
    """A text model whose calls are tried again after a ``transient`` error.
    A long run meets such errors, and the store would otherwise store a
    session verbatim (a failed extraction) or lose the save (a failed
    reconcile). The model's name and settings read through, so the store
    decides exactly as with the model itself."""

    def __init__(self, base: LLM, attempts: int = ATTEMPTS) -> None:
        self.base = base
        self.name, self.available = base.name, base.available
        self.model = getattr(base, "model", None)
        self.effort = getattr(base, "effort", None)
        self.attempts = attempts

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        return retried(lambda: self.base.complete(system, user, json_schema=json_schema),
                       self.attempts)

    def close(self) -> None:
        self.base.close()


#: The store's decision provider (--decider): the configured one
#: (``Config.load``, none unless MEMRY_DECISION_PROVIDER says), none, or Jev
#: (``TYPESAFE_API_KEY``) at the save and at the question alike.
STORE_DECIDERS = ("config", "none", "jev")


def fresh_db(path: str | os.PathLike[str]) -> str:
    """A database path with nothing at it: an earlier store there, and its
    write-ahead log and side files, are removed."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{path.name}*"):
        if old.is_file():
            old.unlink()
    return str(path)


def make_store(mode: str, embedder: Embedder, *, llm: LLM | None = None,
               decider: str = "config", db_path: str = ":memory:") -> MemoryStore:
    """Verbatim: a store with no model at all, default settings. Extract: the
    configured Memry (``Config.load``: LLM, decision model, retrieval
    settings) with the benchmark's embedder and the decision provider
    ``decider`` (``STORE_DECIDERS``), in memory or at ``db_path``. The text
    model's failed calls are tried again (``RetryingLLM``), and so are the
    decision provider's calls that answered nothing (``retrying_decider``)."""
    if mode == "verbatim":
        return MemoryStore(Config(db_path=db_path), llm=llm or NoneLLM(), embedder=embedder)
    overrides: dict[str, Any] = {}
    if decider == "jev":
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise SystemExit("--decider jev needs TYPESAFE_API_KEY")
        overrides["decision"] = {"provider": "jev", "api_key": key}
    elif decider == "none":
        overrides["decision"] = {"provider": "none"}
    elif decider != "config":
        raise ValueError(f"decider {decider!r}: one of {', '.join(STORE_DECIDERS)}")
    config = Config.load(db_path=db_path, **overrides)
    llm = llm or build_llm(config.llm)
    llm = RetryingLLM(llm) if llm.available else llm
    judge = build_decider(config.decision, llm)
    return MemoryStore(config, llm=llm, embedder=embedder,
                       decider=retrying_decider(judge) if judge.available else judge)


def ingest(store: MemoryStore, conversation: Conversation, *, mode: str = "verbatim",
           unit: str = "session", when: str = "never", dataset: str = "") -> Ingested:
    """Save a conversation session by session, as ``mode`` says (see the
    module docstring). The turns of a session are a second apart; each save
    carries its time (``created_at``, ``now``) and its memories' bench
    metadata (``memory_metadata``). A memory is traced to the turns of every
    save that landed on it, a save reconciled as NONE included: a turn that
    restates what a memory already says is credited to that memory, so it
    counts as retrieved whenever the original memory is. That is kept on
    purpose (the memory does hold the answer) and said in the results' notes
    (``DEDUP_NOTE``)."""
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
            bench = {"dataset": dataset, "conversation": conversation.conv_id,
                     "session_id": session.session_id, "session_date": session.date_text,
                     "sessions": [session.session_id], "turns": keys}
            meta: dict[str, Any] = {"bench": bench}
            own: dict[str, Any] = {"bench": bench}
            if moment is not None and when == "always":
                own["when"] = {"start": moment.date().isoformat()}
            dated = {"created_at": moment.isoformat(timespec="seconds") if moment else None,
                     "memory_metadata": own}
            if mode == "verbatim":
                result = store.add(group[0].text, user_id=BENCH_USER, run_id=session.session_id,
                                   metadata=meta, infer=False, memory_type="episodic", **dated)
            else:
                meta["context"] = f"{conversation.label}, {session.date_text}".strip(", ")
                result = store.add([{"role": t.role, "content": t.raw} for t in group],
                                   user_id=BENCH_USER, run_id=session.session_id,
                                   metadata=meta, infer=True, now=moment, **dated)
            if len(result.episode_ids) != len(group):
                ingested.warnings.append(
                    f"{conversation.conv_id} {keys[0]}: {len(group)} turns but "
                    f"{len(result.episode_ids)} episodes; turns traced by save only")
            else:
                ingested.turn_of_episode.update(zip(result.episode_ids, keys))
            ingested.actions.update(action.event for action in result.actions)
            for action in result.actions:
                if action.memory_id:  # a NONE too: the memory it landed on (DEDUP_NOTE)
                    ingested.turns_of_memory.setdefault(action.memory_id, set()).update(keys)
            ingested.warnings.extend(f"{conversation.conv_id} {keys[0]}: {w}"
                                     for w in result.warnings)
    ingested.seconds = time.perf_counter() - started
    return ingested


#: Said in every results file: how a save that added nothing is traced.
DEDUP_NOTE = ("a save reconciled as NONE (already known) credits its turns to the memory "
              "it landed on, so a turn restating an earlier one counts as retrieved when "
              "that memory is; the memory does hold the answer")


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


def mem0_tokens(text: Any) -> set[str]:
    """Mem0's ``simple_tokenize`` (evaluation/metrics/utils.py): lower case,
    ".", ",", "!" and "?" read as spaces, split on white space; as a set."""
    text = str(text).lower()
    for mark in ".,!?":
        text = text.replace(mark, " ")
    return set(text.split())


def mem0_f1(prediction: Any, gold: Any) -> float:
    """Mem0's F1 (``calculate_metrics``): 0 when either string is empty, else
    the F1 of the two token sets, the gold always whole."""
    prediction, gold = str(prediction).strip(), str(gold).strip()
    if not prediction or not gold:
        return 0.0
    pred, true = mem0_tokens(prediction), mem0_tokens(gold)
    if not pred or not true:
        return 0.0
    common = pred & true
    precision, recall = len(common) / len(pred), len(common) / len(true)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu1(prediction: Any, gold: Any) -> float | None:
    """BLEU-1 as Mem0's ``calculate_bleu_scores`` computes it: NLTK's
    ``sentence_bleu`` of the lower-cased ``word_tokenize`` tokens, weights
    (1, 0, 0, 0), smoothing method 1, 0 where NLTK fails. None without NLTK
    or its "punkt_tab" data (``pip install memry[eval]``)."""
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        pred = nltk.word_tokenize(str(prediction).lower())
        true = [nltk.word_tokenize(str(gold).lower())]
    except (ImportError, LookupError):
        return None
    try:
        return float(sentence_bleu(true, pred, weights=(1, 0, 0, 0),
                                   smoothing_function=SmoothingFunction().method1))
    except Exception:
        return 0.0


def lexical_scores(prediction: str, question: Question) -> dict[str, Any]:
    """The scores that need no model: f1, em and contains by LoCoMo's rules
    (the gold an open-domain question's first ";" alternative, a multi-hop
    one's comma-separated parts), and Mem0's f1 and BLEU-1 on the whole gold."""
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
    scores: dict[str, Any] = {"f1": round(f1, 4), "em": em, "contains": contains,
                              "f1_mem0": round(mem0_f1(prediction, question.answer), 4)}
    blue = bleu1(prediction, question.answer)
    if blue is not None:
        scores["bleu1"] = round(blue, 4)
    return scores


def judged_gold(question: Question, judge: Judge) -> str:
    """The gold a judge reads: the one the lexical scores read (an
    open-domain question's first ";" alternative), or the file's whole answer
    for a judge with a true ``reads_full_answer`` attribute."""
    if getattr(judge, "reads_full_answer", False) or question.abstain:
        return question.answer
    if question.category_name == "open-domain":
        return question.answer.split(";")[0].strip()
    return question.answer


def call_judge(judge: Judge, question: Question, gold: str, prediction: str) -> bool:
    """One verdict of ``judge`` on ``prediction``: ``judge(question text,
    gold, prediction)``, and ``asked=question`` as well for a judge with a
    true ``reads_question`` attribute (LongMemEval's judge prompt depends on
    the question's type and on whether it is an abstention question)."""
    if getattr(judge, "reads_question", False):
        return bool(judge(question.question, gold, prediction, asked=question))
    return bool(judge(question.question, gold, prediction))


def verdicts(judge: Judge, question: Question, gold: str, prediction: str, runs: int = 1,
             pool: Any = None) -> tuple[list[bool | None], str | None]:
    """``runs`` verdicts of ``judge`` on one answer to ``question`` (in
    ``pool``, a thread pool, when given) and the first error: a failed run
    is None."""
    def once() -> tuple[bool | None, str | None]:
        try:
            return call_judge(judge, question, gold, prediction), None
        except Exception as exc:  # one failed judgement must not end a long run
            return None, str(exc)[:300]

    if pool is not None and runs > 1:
        results = [f.result() for f in [pool.submit(contextvars.copy_context().run, once)
                                        for _ in range(runs)]]
    else:
        results = [once() for _ in range(runs)]
    return [v for v, _ in results], next((e for _, e in results if e), None)


def judge_fields(found: list[bool | None], error: str | None) -> dict[str, Any]:
    """A row's judge fields: one run gives ``judge`` (True, False or None);
    several give ``judges`` and ``judge``, the mean of those that answered."""
    fields: dict[str, Any] = {}
    if len(found) == 1:
        fields["judge"] = found[0]
    else:
        answered = [float(v) for v in found if v is not None]
        fields["judges"] = found
        fields["judge"] = round(statistics.mean(answered), 4) if answered else None
    if error:
        fields["judge_error"] = error
    return fields


def score_answer(prediction: str, question: Question, judge: Judge = containment_judge,
                 runs: int = 1, pool: Any = None) -> dict[str, Any]:
    """f1, em, contains, Mem0's f1 and BLEU-1 (``lexical_scores``) and the
    judge's verdict for one answer, ``runs`` times (``judge_fields``). The
    judge reads ``judged_gold``. A judge that fails leaves its verdict None
    ("judge_error" says why), which the means leave out."""
    scores = lexical_scores(prediction, question)
    scores.update(judge_fields(*verdicts(judge, question, judged_gold(question, judge),
                                         prediction, runs, pool)))
    return scores


ANSWER_SYSTEM = """You answer a question about past conversations from memories \
retrieved for it. Each memory says the date it was said and, where known, the \
date what it tells happened; the lines after the memories are what was said, \
each after its date and speaker. Use only the memories. Resolve relative dates \
("yesterday", "last week") against the date the words were said. Answer with a \
short phrase, using the memories' own words where you can, not a full sentence. \
If the memories do not say, answer exactly: No information available."""


def memories_text(items: list[Any]) -> str:
    """The harness's own list: Memry's lines as its context builder renders
    them (``answer_with``), or, answering from the whole conversation, each
    turn as "<date>: <speaker>: <text>"."""
    lines = [item if isinstance(item, str) else f"{(item.created_at or '')[:10]}: {item.content}"
             for item in items]
    return "Memories:\n" + "\n".join(f"- {line}" for line in lines) if lines \
        else "Memories: (none found)"


def answer_question(llm: LLM, question: Question, context: str) -> str:
    today = f"Today is {question.question_date:%d %B %Y}.\n" if question.question_date else ""
    raw = llm.complete(ANSWER_SYSTEM, f"{context}\n\n{today}Question: {question.question}\n"
                                      "Short answer:")
    return " ".join(str(raw or "").split())


#: --answer-prompt: (question, the memory list) -> the answering call's
#: messages ([{"role", "content"}, ...]). The memory list is Memry's lines as
#: its context builder renders them (``answer_with``); answering from the whole
#: conversation, the turns (``full_context_memories``). A function with a true
#: ``reads_question`` attribute is also given ``asked=`` the ``Question`` (its
#: date: LongMemEval's prompt contains the day the question is asked).
AnswerPrompt = Callable[[str, list[Any]], list[dict[str, str]]]


def chat(llm: LLM, messages: list[dict[str, str]]) -> str:
    """One answering call with ``messages``: through the model's own ``chat``
    where it has one, else as ``complete(system, user)``."""
    if hasattr(llm, "chat"):
        return llm.chat(messages)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    return llm.complete(system, user)


def search_signals(store: MemoryStore, question: str, results: list[Any]) -> dict[str, Any]:
    """How the search went for one question: the names of the entities it
    names (``named``; ``named_entities`` counts the entities, several can
    share a name), whether the linked search ran from them (``linked``: a
    result carries its "about" signal), how many calls judged the ranking
    (``judge_calls``: 0 unjudged, 1, or 2 for a question needing several
    memories) and how many results were found to be members of such a set."""
    ids = detect_query_entities(store.backend, Scope(user_id=BENCH_USER), question, longest=True)
    names = {entity.name for entity in map(store.backend.get_entity, ids) if entity is not None}
    signals = [r.signals or {} for r in results]
    return {"named": sorted(names), "named_entities": len(ids),
            "linked": any("about" in s for s in signals),
            "judge_calls": max((int(s.get("calls") or 0) for s in signals), default=0),
            "set_members": sum(1 for s in signals if s.get("member")),
            **search_start(store, question)}


def search_start(store: MemoryStore, question: str) -> dict[str, Any]:
    """The entities the search of ``question`` starts from (stage 1 of
    ``MemoryStore.search``, its seeds): their names (``start``), whether the
    start is the owner for a question in the first person
    (``start_first_person``), whether it is an entity whose entity questions
    match the question by its role (``start_by_role``), and whether it
    contains an entity other than the owner (``start_other_than_owner``).
    The plan is made again as the search made it; the question's vector
    comes from the embedder's cache in a benchmark run."""
    from memry.store import _Reads

    scope = Scope(user_id=BENCH_USER)
    plan = store._plan(question, _Reads(scope), True)
    named, _ = store._seeds(question, scope)
    owner = store.owner_entity(BENCH_USER)
    names = [entity.name for entity in map(store.backend.get_entity, plan.seeds)
             if entity is not None]
    return {"start": names, "start_first_person": plan.first_person,
            "start_by_role": bool(plan.seeds) and plan.seeds != named,
            "start_other_than_owner": any(owner is None or seed != owner.id
                                          for seed in plan.seeds)}


_ENCODING: list[Any] = []


def count_tokens(text: str) -> int | None:
    """The ``cl100k_base`` tokens of ``text`` (tiktoken), as the Mem0 paper
    counts memory tokens; None without tiktoken (``pip install memry[eval]``)."""
    if not _ENCODING:
        try:
            import tiktoken

            _ENCODING.append(tiktoken.get_encoding("cl100k_base"))
        except Exception:
            _ENCODING.append(None)
    encoding = _ENCODING[0]
    return len(encoding.encode(text, disallowed_special=())) if encoding is not None else None


def context_text(answer_prompt: AnswerPrompt | None, items: list[Any]) -> str:
    """The memory list as the answering call shows it: the prompt function's
    ``context_of`` where it has one, else the harness's own list."""
    shown = getattr(answer_prompt, "context_of", None)
    return shown(items) if shown else memories_text(items)


def memory_record(result: Any, ingested: Ingested) -> dict[str, Any]:
    """One retrieved memory as a row keeps it: its text, times, score and turns."""
    memory = result.memory
    return {"id": memory.id, "memory": memory.content, "created_at": memory.created_at,
            "updated_at": memory.updated_at, "score": round(float(result.score), 6),
            "turns": sorted(ingested.turns_of(memory))}


def answer_with(answer_llm: LLM, question: Question, memories: list[Memory],
                answer_prompt: AnswerPrompt | None, evidence: list[Any] = (),
                entities: list[Any] = ()) -> dict[str, Any]:
    """One answering call from what Memry gives a model for the question:
    the descriptions of the ``entities`` it names, then ``memories`` and
    their ``evidence`` turns, as its context builder renders them
    (``context_lines``). That list is the memory list of the answer prompt
    (``answer_from``); ``k`` counts the memories."""
    record = answer_from(answer_llm, question, context_lines(entities, memories, evidence),
                         answer_prompt)
    record["k"] = len(memories)
    return record


def answer_from(answer_llm: LLM, question: Question, items: list[Any],
                answer_prompt: AnswerPrompt | None) -> dict[str, Any]:
    """One answering call from the memory list ``items``: the prediction, the
    call's seconds, the ``cl100k_base`` tokens of the list shown
    (``context_tokens``) and the input tokens the reply reports, if any.
    With ``answer_prompt`` the prediction is kept as the model wrote it, as
    Mem0's evaluation keeps it."""
    record: dict[str, Any] = {"k": len(items)}
    started = time.perf_counter()
    try:
        if answer_prompt is not None:
            messages = answer_prompt(question.question, items, asked=question) \
                if getattr(answer_prompt, "reads_question", False) \
                else answer_prompt(question.question, items)
            prediction = str(chat(answer_llm, messages) or "").strip()
        else:
            prediction = answer_question(answer_llm, question, memories_text(items))
    except Exception as exc:  # one failed call must not end a long run
        prediction, record["answer_error"] = "", str(exc)[:300]
    record["answer_seconds"] = round(time.perf_counter() - started, 3)
    record["prediction"] = prediction
    usage = getattr(answer_llm, "last_usage", lambda: None)()
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        record["input_tokens"] = usage["prompt_tokens"]
    record["context_tokens"] = count_tokens(context_text(answer_prompt, items))
    return record


#: The fields of the headline answer a row carries at its top level.
ANSWER_FIELDS = ("prediction", "answer_error", "answer_seconds", "context_tokens",
                 "input_tokens", "f1", "em", "contains", "f1_mem0", "bleu1", "judge", "judges",
                 "judge_error", "judge_corrected", "judges_corrected", "judge_clean")


def judge_answers(answers: dict[Any, dict[str, Any]], question: Question, judge: Judge,
                  runs: int, pool: Any, correction: dict[str, Any] | None,
                  stages: dict[Any, str] | None = None) -> None:
    """Score each answer (``lexical_scores``) and judge it ``runs`` times,
    every call in ``pool`` at once when given. With ``correction`` (an
    audited question's corrected answer) each answer is judged against it
    too: ``judge_corrected`` is that verdict; without, ``judge_corrected``
    is the verdict against the file's answer and ``judge_clean`` the same
    (the mean over the questions the audit left alone). ``stages`` (key of
    ``answers`` -> stage) counts an answer's judge calls under a stage of
    their own (``api_usage.stage``; default: the caller's)."""
    gold = judged_gold(question, judge)
    golds = [("", gold)] + ([("_corrected", correction["correct_answer"])] if correction else [])
    stages = stages or {}

    def once(stage: str | None, truth: str, prediction: str) -> tuple[bool | None, str | None]:
        try:
            if stage:
                with api_usage.stage(stage):
                    return call_judge(judge, question, truth, prediction), None
            return call_judge(judge, question, truth, prediction), None
        except Exception as exc:  # one failed judgement must not end a long run
            return None, str(exc)[:300]

    pending = {}
    for k, record in answers.items():
        record.update(lexical_scores(record["prediction"], question))
        for suffix, truth in golds:
            for run in range(runs):
                args = (stages.get(k), truth, record["prediction"])
                pending[(k, suffix, run)] = (pool.submit(contextvars.copy_context().run, once, *args)
                                             if pool is not None else None, args)
    for (k, suffix, run), (future, args) in pending.items():
        pending[(k, suffix, run)] = future.result() if future is not None else once(*args)
    for k, record in answers.items():
        for suffix, _ in golds:
            found = [pending[(k, suffix, run)][0] for run in range(runs)]
            error = next((pending[(k, suffix, run)][1] for run in range(runs)
                          if pending[(k, suffix, run)][1]), None)
            fields = judge_fields(found, error if not suffix else None)
            record.update({f"{key}{suffix}" if key != "judge_error" else key: value
                           for key, value in fields.items()})
        if correction is None:
            record["judge_corrected"] = record.get("judge")
            if "judges" in record:
                record["judges_corrected"] = record["judges"]
            record["judge_clean"] = record.get("judge")


def ask(ingested: Ingested, question: Question, *, k: int = 10, answer_llm: LLM | None = None,
        judge: Judge = containment_judge, use_context: bool = False,
        answer_prompt: AnswerPrompt | None = None, search_stage: str = "search",
        ks: list[int] | None = None, judge_runs: int = 1, pool: Any = None,
        correction: dict[str, Any] | None = None,
        compare_evidence_tokens: int | None = None,
        compare_answer_llm: LLM | None = None,
        descriptions: bool = True) -> dict[str, Any]:
    """Search for one question once, score what came back, and answer when
    asked: from the top k of that search for each k of ``ks`` (default
    ``[k]``), after the descriptions of the entities the question names
    (``MemoryStore.described_entities``, built where stale under the stage
    "describe"; none with ``descriptions`` false), each answer judged
    ``judge_runs`` times (``judge_answers``).
    The row carries the answer at ``k`` at its top level and every answer
    under "answers". The model calls are counted under ``search_stage``,
    "answer" and "judge"; with ``pool`` (a thread pool) a question's answer
    calls, and then its judge calls, run at once. With
    ``compare_evidence_tokens`` each k is answered a second time from the
    same memories, their turns chosen within that many tokens (0: none),
    kept under "answers_compared" and counted under "answer:compared" and
    "judge:compared"; ``compare_answer_llm`` writes that second answer (the
    turns within the store's own budget unless ``compare_evidence_tokens``
    is given too)."""
    ks = sorted(set(ks or [k]) | {k})
    depth = max(DEPTH, *ks)
    store = ingested.store
    started = time.perf_counter()
    with api_usage.stage(search_stage):
        results = store.search(question.question, user_id=BENCH_USER, limit=depth,
                               evidence=False)
    ms = (time.perf_counter() - started) * 1000
    results = results[:depth]
    units = [ingested.units_of(r.memory, question.level) for r in results]
    row: dict[str, Any] = {
        "conversation": ingested.conversation.conv_id, "qid": question.qid,
        "category": question.category, "category_name": question.category_name,
        "question": question.question, "answer": question.answer, "level": question.level,
        "evidence": question.evidence, "abstain": question.abstain,
        **question.extra,
        "reference_date": question.reference_date,
        "search_ms": round(ms, 3),
        "evidence_ranks": [rank for rank, got in enumerate(units, start=1)
                           if got & set(question.evidence)],
        "retrieved": [sorted(ingested.turns_of(r.memory)) for r in results[:k]],
        "memories": [memory_record(r, ingested) for r in results],
    }
    for at in KS:
        row[f"recall@{at}"] = evidence_recall(units, question.evidence, at)
    row["mrr"] = reciprocal_rank(units, question.evidence)
    row.update(search_signals(store, question.question, results))
    if answer_llm is None:
        return row
    if correction is not None:
        row["audit"] = {key: correction.get(key) for key in ("error_type", "correct_answer")}
    answers: dict[int, dict[str, Any]] = {}
    compared: dict[int, dict[str, Any]] = {}
    if use_context:
        with api_usage.stage(search_stage):
            context = store.reconstruct_context(question.question, user_id=BENCH_USER, limit=k,
                                                token_budget=CONTEXT_TOKENS)
        in_context = [store.get(mid) for mid in context.memory_ids]
        row["context_recall"] = evidence_recall(
            [ingested.units_of(m, question.level) for m in in_context if m],
            question.evidence, len(in_context))
        with api_usage.stage("answer"):
            started = time.perf_counter()
            record: dict[str, Any] = {"k": k}
            try:
                record["prediction"] = answer_question(
                    answer_llm, question, context.text or "Memories: (none found)")
            except Exception as exc:  # one failed call must not end a long run
                record["prediction"], record["answer_error"] = "", str(exc)[:300]
            record["answer_seconds"] = round(time.perf_counter() - started, 3)
            answers[k] = record
    else:
        top = [r.memory for r in results]
        # the entities the question names, described as an agent's context describes them
        described: list[Any] = []
        if descriptions:
            with api_usage.stage("describe"):
                described = store.described_entities(question.question, user_id=BENCH_USER)
        row["described"] = [entity.name for entity in described]
        # each variant's evidence budget (None: the store's own), as its answer stage names it
        budgets = {"answer": None}
        if compare_evidence_tokens is not None or compare_answer_llm is not None:
            budgets["answer:compared"] = compare_evidence_tokens
        # each variant's model: the compared answer's own where one is given
        writers = {"answer": answer_llm, "answer:compared": compare_answer_llm or answer_llm}
        # the source turns of each k's memories, as a search of that depth chooses them,
        # chosen once for each budget
        with api_usage.stage(search_stage):
            chosen = {(budget, at): store.evidence(question.question, results[:at],
                                                   user_id=BENCH_USER, token_budget=budget)
                      for budget in dict.fromkeys(budgets.values()) for at in ks}
        turns = {(name, at): chosen[(budget, at)] for name, budget in budgets.items()
                 for at in ks}
        if pool is not None and len(turns) > 1:
            futures = {key: pool.submit(contextvars.copy_context().run, staged, key[0],
                                        answer_with, writers[key[0]], question, top[:key[1]],
                                        answer_prompt, shown, described)
                       for key, shown in turns.items()}
            done = {key: future.result() for key, future in futures.items()}
        else:
            done = {key: staged(key[0], answer_with, writers[key[0]], question, top[:key[1]],
                                answer_prompt, shown, described)
                    for key, shown in turns.items()}
        for key, record in done.items():
            record["evidence"] = [ingested.turn_of_episode.get(t.episode_id, t.episode_id)
                                  for t in turns[key]]
        answers = {at: done[("answer", at)] for at in ks}
        compared = {at: done[("answer:compared", at)] for at in ks
                    if ("answer:compared", at) in done}
    everything: dict[Any, dict[str, Any]] = {**answers}
    everything.update({("compared", at): record for at, record in compared.items()})
    with api_usage.stage("judge"):
        judge_answers(everything, question, judge, judge_runs, pool, correction,
                      stages={("compared", at): "judge:compared" for at in compared})
    row["answer_k"] = answers[k]["k"]
    row.update({key: answers[k][key] for key in ANSWER_FIELDS if key in answers[k]})
    row["answers"] = {str(at): answer for at, answer in answers.items()}
    if compared:
        row["answers_compared"] = {str(at): answer for at, answer in compared.items()}
    return row


def staged(stage: str, function: Callable[..., Any], *args: Any) -> Any:
    """``function(*args)`` with its calls counted under ``stage``."""
    with api_usage.stage(stage):
        return function(*args)


METRICS = ("recall@5", "recall@10", "recall@20", "mrr", "context_recall",
           "f1", "f1_mem0", "bleu1", "em", "contains", "judge", "judge_corrected", "judge_clean",
           "context_tokens")


def _mean(values: Any) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return round(statistics.mean(kept), 4) if kept else None


def judge_runs(rows: list[dict[str, Any]]) -> list[float]:
    """The share judged right in each judge run (``judges``, one verdict per
    run), over the rows that have that run's verdict."""
    runs = max((len(r.get("judges") or []) for r in rows), default=0)
    out = []
    for i in range(runs):
        verdicts = [r["judges"][i] for r in rows
                    if len(r.get("judges") or []) > i and r["judges"][i] is not None]
        if verdicts:
            out.append(round(statistics.mean(float(v) for v in verdicts), 4))
    return out


def _summary(category: Any, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"category": category, "name": name, "n": len(rows),
                           "with_evidence": sum(1 for r in rows if r.get("evidence"))}
    for metric in METRICS:
        if any(metric in r for r in rows):
            out[metric] = _mean(r.get(metric) for r in rows)
    runs = judge_runs(rows)
    if len(runs) > 1:
        out["judge_runs"] = runs
        out["judge_std"] = round(statistics.stdev(runs), 4)
    times = [r["search_ms"] for r in rows if r.get("search_ms") is not None]
    out["search_ms"] = round(statistics.median(times), 3) if times else None
    answered = [r["answer_seconds"] for r in rows if r.get("answer_seconds") is not None]
    if answered:
        out["answer_seconds"] = round(statistics.median(answered), 3)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The metrics' means per category (number and name) and overall, and the
    median search and answer times. A metric a row does not have (null) is
    left out of its mean. The overall leaves out the questions whose right
    answer is that the conversation does not say (LoCoMo's adversarial
    category 5, ``ABSTAIN_CATEGORIES``), as the published LoCoMo results do.
    With several judge runs a row's ``judge`` is its mean verdict, and
    ``judge_runs`` holds each run's share right with ``judge_std`` their
    sample standard deviation.

    For LongMemEval's rows (level "session") the tables also have the two
    other numbers of the official print_qa_metrics.py: a last row
    "abstention" for the questions whose right answer is that the haystack
    does not contain it (these questions are also in their types' rows and
    in the overall), and the overall's ``judge_task_averaged``, the mean of
    the types' judge scores."""
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["category"], row["category_name"]), []).append(row)
    order = sorted(groups, key=lambda key: (not isinstance(key[0], int), str(key[0]).zfill(6)))
    scored = [r for r in rows if r.get("category_name") not in ABSTAIN_CATEGORIES]
    tables = {"by_category": [_summary(c, n, groups[(c, n)]) for c, n in order],
              "overall": _summary("all", "overall", scored)}
    if any(r.get("level") == "session" for r in rows):
        types = [entry["judge"] for entry in tables["by_category"]
                 if entry.get("judge") is not None]
        if types:
            tables["overall"]["judge_task_averaged"] = round(statistics.mean(types), 4)
        abstained = [r for r in rows if r.get("abstain")]
        if abstained:
            tables["by_category"].append(_summary("abstention", "abstention", abstained))
    return tables


def markdown_table(tables: dict[str, Any]) -> str:
    rows = [*tables["by_category"], tables["overall"]]
    metrics = [m for m in METRICS if any(r.get(m) is not None for r in rows)]
    header = ["category", "name", "n", *metrics, "search ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(row["category"]), row["name"], str(row["n"])]
        cells += ["-" if row.get(m) is None else f"{row[m]:.1f}" if m == "context_tokens"
                  else f"{row[m]:.3f}" for m in metrics]
        cells.append("-" if row.get("search_ms") is None else f"{row['search_ms']:.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    if tables["overall"].get("judge_task_averaged") is not None:
        lines.append(f"\njudge, task-averaged: {tables['overall']['judge_task_averaged']:.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# embeddings


class SqliteEmbeddingCache(Embedder):
    """One vector per distinct text, kept in a SQLite file, fetched in batches
    of 256: a re-run, and the many LongMemEval haystacks that share sessions,
    embed each text once. The processes of a parallel run share the file. A
    failed batch is tried again (``retried``). ``close`` keeps it open,
    because every store of a run closes its embedder; ``release`` closes it."""

    def __init__(self, base: Embedder, path: str | os.PathLike[str]) -> None:
        self.base = base
        self.name, self._model, self.dimensions = base.name, base._model, base.dimensions
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
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
            vectors = retried(lambda: self.base.embed([text for _, text in batch]))
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


#: A question pass's decision provider; None keeps the store's own.
DeciderFactory = Callable[[], Decider | None]


def retrying_decider(decider: Decider, attempts: int = ATTEMPTS) -> Decider:
    """``decider``, asking again when a call came back with no answer at all:
    a provider that fails answers nothing, and search then keeps the order it
    had without saying so. ``decider.failures`` counts the calls that still
    answered nothing."""
    ask_once = decider.decide
    decider.failures = 0  # type: ignore[attr-defined]

    def decide(state: str, questions: dict[str, Any]) -> Any:
        for attempt in range(attempts):
            answers = ask_once(state, questions)
            if not questions or any(answers[key].available for key in questions):
                return answers
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
        decider.failures += 1  # type: ignore[attr-defined]
        return answers

    decider.decide = decide  # type: ignore[method-assign]
    return decider


def search_decider(name: str) -> DeciderFactory:
    """What --search-decider ``name`` asks with (``SEARCH_DECIDERS``): the
    store's own provider, none, or Jev (``TYPESAFE_API_KEY``), whose calls
    are tried again (``retrying_decider``)."""
    if name == "store":
        return lambda: None
    if name == "none":
        return NoneDecider
    if name == "jev":
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise SystemExit("--search-decider jev needs TYPESAFE_API_KEY")
        return lambda: retrying_decider(JevDecider(DecisionConfig(provider="jev", api_key=key)))
    raise ValueError(f"search decider {name!r}: one of {', '.join(SEARCH_DECIDERS)}")


def store_stats(store: MemoryStore, conversation: Conversation) -> dict[str, Any]:
    """The loaded store's named entities: how many, how many share their
    name with another (``same_name_entities``, in ``same_name_groups``
    names), the open merge proposals, and for each speaker how many memories
    each entity of that name holds, most first."""
    scope = Scope(user_id=BENCH_USER)
    entities = store.backend.list_entities(scope, limit=1_000_000)

    def name_of(entity: Any) -> str:
        return entity.normalized or entity.name.strip().lower()

    names = Counter(name_of(e) for e in entities)
    shared = {name: n for name, n in names.items() if n > 1}
    speakers = {speaker: sorted((store.backend.count_entity_memories(e.id) for e in entities
                                 if name_of(e) == speaker.strip().lower()), reverse=True)
                for speaker in conversation.speakers}
    memories = store.get_all(user_id=BENCH_USER, limit=1_000_000)
    with_questions = sum(1 for mid, rows in store.backend.questions_of(
        [m.id for m in memories]).items() if rows)
    entities_asked = sum(1 for rows in store.backend.entity_questions_of(
        [e.id for e in entities]).values() if rows)
    owner = store.owner_entity(BENCH_USER)
    return {"entities": len(entities), "same_name_entities": sum(shared.values()),
            "same_name_groups": len(shared),
            "open_proposals": len(store.backend.list_proposals(scope, limit=1_000_000)),
            "speaker_entities": speakers,
            "owner": owner.name if owner is not None else None,
            "memories_with_questions": with_questions,
            "entities_with_questions": entities_asked}


#: The stage the entity question writer's calls are counted under.
ENTITY_QUESTIONS_STAGE = "entity_questions"

#: The stage the description writer's calls for the entity questions are counted under.
ENTITY_DESCRIPTIONS_STAGE = "entity_descriptions"

#: Hubs related to the owner described at most for one question's store.
DESCRIBE_CAP = 20

#: Store counts copied into each of its questions' rows.
ROW_STORE_FIELDS = ("owner", "memories_with_questions", "entities_with_questions",
                    "entities_described")


def describe_related_hubs_step(store: MemoryStore, *, cap: int | None = None,
                               log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """With ``retrieval.entity_questions`` on, a description for each hub
    related to the owner that has none yet, so the entity question writer
    has something to ask from. Memry's own description writer writes it
    (``MemoryStore._refresh_entity_description``, the one an opened entity
    and a question about it use). A hub with fewer than
    ``DESCRIPTION_MIN_MEMORIES`` memories gets no description, so it is not
    taken. The most remembered hubs go first, at most ``cap``; the rest are
    counted in "skipped_by_cap" and logged. One call a hub, counted under
    ``ENTITY_DESCRIPTIONS_STAGE``; a call refused at a cap ends the run.
    Off, no call and {"skipped": ...}. ``cap`` defaults to ``DESCRIBE_CAP``."""
    if not store.config.retrieval.entity_questions:
        return {"skipped": "retrieval.entity_questions is off"}
    cap = DESCRIBE_CAP if cap is None else cap
    owner = store.owner_entity(BENCH_USER)
    if owner is None:
        return {"skipped": "no owner"}
    related = {relation.object if relation.subject == owner.id else relation.subject
               for relation in store.backend.relations_of([owner.id])} - {owner.id}
    due: list[tuple[int, str]] = []
    for entity_id in sorted(related):
        entity = store.backend.get_entity(entity_id)
        if (entity is None or entity.merged_into or entity.description
                or not store._is_hub(entity_id)):
            continue
        memories = store.backend.count_entity_memories(entity_id)
        if memories >= DESCRIPTION_MIN_MEMORIES:
            due.append((memories, entity_id))
    due.sort(key=lambda item: (-item[0], item[1]))
    taken = due[: max(int(cap), 0)]
    skipped = len(due) - len(taken)
    if skipped and log:
        log(f"  entity descriptions: {skipped} of {len(due)} hubs skipped by the cap of {cap}")
    described = 0
    with api_usage.stage(ENTITY_DESCRIPTIONS_STAGE):
        for _, entity_id in taken:
            entity = store._refresh_entity_description(entity_id)
            if entity is not None and entity.description:
                described += 1
    return {"due": len(due), "calls": len(taken) if store.llm.available else 0,
            "described": described, "skipped_by_cap": skipped}


def write_entity_questions_step(store: MemoryStore) -> dict[str, Any]:
    """With ``retrieval.entity_questions`` on, the store's entity questions,
    written once for the loaded haystack (``MemoryStore.write_entity_questions``),
    its calls counted under ``ENTITY_QUESTIONS_STAGE``. A call refused at a
    cap (``api_usage.CapReached``) ends the run as any other. Off, no call
    and {"skipped": ...}. The writer asks only about described entities
    related to the owner; a store loaded just now has no description yet,
    so the runner describes them first (``describe_related_hubs_step``)."""
    if not store.config.retrieval.entity_questions:
        return {"skipped": "retrieval.entity_questions is off"}
    with api_usage.stage(ENTITY_QUESTIONS_STAGE):
        return store.write_entity_questions(user_id=BENCH_USER)


def _qualname(function: Any) -> str | None:
    if function is None:
        return None
    return f"{getattr(function, '__module__', '?')}:{getattr(function, '__qualname__', function)}"


def pass_tables(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Each question pass's tables, from its rows; with answers at several k,
    ``tables_by_k`` holds the tables of the answers at each k, and with
    answers under another evidence budget (``answers_compared``),
    ``compared_by_k`` holds theirs."""
    out = []
    for name in names:
        mine = [r for r in rows if r.get("search_decider", name) == name]
        entry: dict[str, Any] = {"search_decider": name, "tables": aggregate(mine)}
        ks = sorted({int(at) for r in mine for at in r.get("answers") or {}})
        if len(ks) > 1:
            entry["tables_by_k"] = {
                str(at): aggregate([{**r, **r["answers"][str(at)]} for r in mine
                                    if str(at) in (r.get("answers") or {})])
                for at in ks}
        compared = sorted({int(at) for r in mine for at in r.get("answers_compared") or {}})
        if compared:
            entry["compared_by_k"] = {
                str(at): aggregate([{**r, **r["answers_compared"][str(at)]} for r in mine
                                    if str(at) in (r.get("answers_compared") or {})])
                for at in compared}
        out.append(entry)
    return out


def full_context_memories(conversation: Conversation) -> list[Memory]:
    """The whole conversation as full-context answering reads it: one item a
    turn, "<speaker>: <text>" as the file writes the text (no photo caption,
    as Mem0's ``locomo10_rag.json``), dated with its session's time."""
    return [Memory(content=f"{turn.role}: {turn.said}",
                   created_at=session.date.isoformat() if session.date else session.date_text)
            for session in conversation.sessions for turn in session.turns]


def answer_in_full(conversation: Conversation, question: Question, context: list[Memory], *,
                   answer_llm: LLM, judge: Judge, answer_prompt: AnswerPrompt | None,
                   judge_runs: int, pool: Any, correction: dict[str, Any] | None,
                   ) -> dict[str, Any]:
    """One question answered from the whole conversation (``context``, from
    ``full_context_memories``), judged ``judge_runs`` times, with no store and
    no search."""
    row: dict[str, Any] = {
        "conversation": conversation.conv_id, "qid": question.qid,
        "category": question.category, "category_name": question.category_name,
        "question": question.question, "answer": question.answer, "level": question.level,
        "evidence": question.evidence, "abstain": question.abstain, **question.extra,
        "reference_date": question.reference_date}
    if correction is not None:
        row["audit"] = {key: correction.get(key) for key in ("error_type", "correct_answer")}
    with api_usage.stage("answer"):
        record = answer_from(answer_llm, question, context, answer_prompt)
    with api_usage.stage("judge"):
        judge_answers({0: record}, question, judge, judge_runs, pool, correction)
    row["answer_k"] = record["k"]
    row.update({key: record[key] for key in ANSWER_FIELDS if key in record})
    return row


def run_benchmark(conversations: list[Conversation], *, dataset: str, mode: str = "verbatim",
                  unit: str = "session", embedder: Embedder | None = None, k: int = 10,
                  questions: int | None = None, answer_llm: LLM | None = None,
                  judge: Judge = containment_judge, use_context: bool = False,
                  when: str = "never",
                  store_factory: Callable[[], MemoryStore] | None = None,
                  log: Callable[[str], None] | None = None,
                  search_deciders: dict[str, DeciderFactory] | None = None,
                  answer_prompt: AnswerPrompt | None = None,
                  ks: list[int] | None = None, judge_runs: int = 1,
                  categories: set[str] | None = None, workers: int = 1,
                  corrections: dict[str, dict[str, Any]] | None = None,
                  decider: str = "config", store_dir: str | os.PathLike[str] | None = None,
                  full_context: bool = False,
                  evidence_tokens: int | None = None,
                  compare_evidence_tokens: int | None = None,
                  compare_answer_llm: LLM | None = None,
                  descriptions: bool = True,
                  question_keys: str = "config") -> dict[str, Any]:
    """Ingest each conversation into a fresh store and ask its questions once
    per pass of ``search_deciders`` (name -> the decision provider the store
    asks with, given it once the conversation is loaded; default: the store's
    own, as "store"). Only questions of ``categories`` (their numbers or
    names as text) are asked, then the first ``questions`` of those. Each
    question is searched once and answered from the top k for each k of
    ``ks`` (default ``[k]``; ``k`` is the headline), each answer judged
    ``judge_runs`` times, a question's calls ``workers`` at a time.
    ``corrections`` (qid -> an audit's corrected answer) has those questions
    judged against the correction too. ``decider`` and ``store_dir`` go to
    ``make_store`` (a store in memory, or <store_dir>/<conversation>.sqlite).
    ``full_context`` answers from the whole conversation instead, with no
    store and no search. ``evidence_tokens`` sets each store's
    ``retrieval.evidence_tokens`` (None keeps its own);
    ``compare_evidence_tokens`` answers each k again from the same search
    with the turns chosen within that many tokens, and ``compare_answer_llm``
    with that model (``ask``); ``descriptions``
    false leaves out the descriptions of the entities a question names, for
    an ablation. ``question_keys`` "on" or "both" has the store write
    question keys at ingest (``retrieval.question_keys``; "config" keeps the
    store's setting); "both" then asks every pass twice, once with the keys
    read and once without ("<pass>:text-only"), from the one store, so the
    keys' gain is measured on the same memories. Returns {config, stores, passes, tables (the first
    pass's), rows, warnings, notes, complete}. A call refused at a cap
    (``api_usage.CapReached``) ends the run where it is: what was done is
    kept, "complete" is false and "stopped" says where."""
    embedder = embedder or HashEmbedder(256)
    log = log or (lambda text: print(text, file=sys.stderr, flush=True))
    passes = search_deciders or {"store": search_decider("store")}
    if full_context:
        passes = {"full-context": search_decider("store")}
    if question_keys not in ("config", "on", "both"):
        raise ValueError(f"question_keys {question_keys!r}: config, on or both")
    # which passes read the question keys; "both" adds a text-only twin of each
    reads_keys: dict[str, bool | None] = {name: None for name in passes}
    if question_keys == "both" and not full_context:
        reads_keys = {}
        for name, factory in list(passes.items()):
            reads_keys[name] = True
            passes[f"{name}:text-only"] = factory
            reads_keys[f"{name}:text-only"] = False
    ks = sorted(set(ks or [k]) | {k})
    corrections = corrections or {}
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    rows: list[dict[str, Any]] = []
    stores: list[dict[str, Any]] = []
    warnings: list[str] = []
    notes: list[str] = []
    stopped: str | None = None
    for conv in conversations:
        warnings.extend(conv.warnings)
        asked = [q for q in conv.questions
                 if not categories or {str(q.category), q.category_name} & categories]
        asked = asked[:questions] if questions else asked
        entry: dict[str, Any] = {"conversation": conv.conv_id, "sessions": len(conv.sessions),
                                 "turns": len(conv.turns), "questions": len(asked)}
        stores.append(entry)
        if full_context:
            context = full_context_memories(conv)
            started = time.perf_counter()
            try:
                with api_usage.labelled(conv.conv_id):
                    for question in asked:
                        rows.append({"search_decider": "full-context", **answer_in_full(
                            conv, question, context, answer_llm=answer_llm, judge=judge,
                            answer_prompt=answer_prompt, judge_runs=judge_runs, pool=pool,
                            correction=corrections.get(question.qid))})
            except api_usage.CapReached as exc:
                stopped = f"{conv.conv_id}: {exc}"
                entry["stopped"] = str(exc)
            entry["seconds_full-context"] = round(time.perf_counter() - started, 2)
            log(f"  {conv.conv_id}: {len(asked)} questions from the whole conversation"
                + (f"; stopped: {stopped}" if stopped else ""))
            if stopped:
                break
            continue
        if mode == "verbatim" and hasattr(embedder, "warm"):
            embedder.warm([t.text for t in conv.turns] + [q.question for q in asked])
        if store_factory:
            store = store_factory()
        else:
            db_path = fresh_db(pathlib.Path(store_dir) / f"{conv.conv_id}.sqlite") \
                if store_dir else ":memory:"
            store = make_store(mode, embedder, decider=decider, db_path=db_path)
        if evidence_tokens is not None:
            store.config.retrieval.evidence_tokens = evidence_tokens
        if question_keys in ("on", "both"):
            store.config.retrieval.question_keys = True
        entry["question_keys"] = store.config.retrieval.question_keys
        entry["decider"] = store.decider.name
        entry["decider_model"] = getattr(store.decider, "model", None)
        entry["text_model"] = getattr(store.llm, "model", None) if store.llm.available else None
        entry["text_effort"] = getattr(store.llm, "effort", None) if store.llm.available else None
        entry["evidence_tokens"] = store.config.retrieval.evidence_tokens
        try:
            with api_usage.labelled(conv.conv_id):
                if mode == "extract" and not store.llm.available:
                    raise SystemExit("--ingest extract needs a configured LLM "
                                     "(OPENAI_API_KEY, ANTHROPIC_API_KEY or MEMRY_LLM_PROVIDER)")
                with api_usage.stage("ingest"):
                    ingested = ingest(store, conv, mode=mode, unit=unit, when=when,
                                      dataset=dataset)
                entry["entity_questions"] = store.config.retrieval.entity_questions
                entry["entities_described"] = 0
                if dataset == "longmemeval":
                    entry["entity_descriptions"] = describe_related_hubs_step(store, log=log)
                    entry["entities_described"] = entry["entity_descriptions"].get("described", 0)
                    entry["entity_question_writer"] = write_entity_questions_step(store)
                entry.update(memories=len(store.get_all(user_id=BENCH_USER, limit=1_000_000)),
                             ingest_seconds=round(ingested.seconds, 2),
                             actions=dict(ingested.actions), **store_stats(store, conv))
                counts = {key: entry[key] for key in ROW_STORE_FIELDS}
                warnings.extend(ingested.warnings)
                for name, factory in passes.items():
                    asked_with, kept = factory(), store.decider
                    if asked_with is not None:
                        store.decider = asked_with
                    kept_keys = store.config.retrieval.question_keys
                    if reads_keys.get(name) is not None:
                        store.config.retrieval.question_keys = bool(reads_keys[name])
                    started = time.perf_counter()
                    try:
                        for question in asked:
                            rows.append({"search_decider": name, **counts, **ask(
                                ingested, question, k=k, answer_llm=answer_llm, judge=judge,
                                use_context=use_context, answer_prompt=answer_prompt,
                                search_stage=f"search:{name}", ks=ks, judge_runs=judge_runs,
                                pool=pool, correction=corrections.get(question.qid),
                                compare_evidence_tokens=compare_evidence_tokens,
                                compare_answer_llm=compare_answer_llm,
                                descriptions=descriptions)})
                    finally:
                        entry[f"seconds_{name}"] = round(time.perf_counter() - started, 2)
                        store.config.retrieval.question_keys = kept_keys
                        if asked_with is not None:
                            store.decider = kept
                            entry[f"decider_failures_{name}"] = getattr(asked_with, "failures", 0)
                            asked_with.close()
        except api_usage.CapReached as exc:
            stopped = f"{conv.conv_id}: {exc}"
            entry["stopped"] = str(exc)
        finally:
            store.close()
        log(f"  {conv.conv_id}: {len(conv.turns)} turns -> {entry.get('memories', '?')} memories "
            f"in {entry.get('ingest_seconds', '?')}s, {len(asked)} questions"
            + (f"; stopped: {stopped}" if stopped else ""))
        if stopped:
            break
    if pool is not None:
        pool.shutdown()
    if not full_context:
        notes.append(DEDUP_NOTE)
    if mode == "extract" and unit == "session" and not full_context:
        notes.append("extract by session: a memory counts for every turn of the session it "
                     "came from, so turn-level recall is session-level recall")
    names = list(passes)
    tables = pass_tables(rows, names)
    return {
        "dataset": dataset,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"ingest": "none (full context)" if full_context else mode,
                   "extract_unit": unit if mode == "extract" and not full_context else None,
                   "embedder": None if full_context else embedder.model_id, "k": k, "ks": ks,
                   "depth": max(DEPTH, *ks), "judge_runs": judge_runs,
                   "categories": sorted(categories) if categories else None,
                   "decider": decider, "full_context": full_context,
                   "audited_questions": len(corrections),
                   "conversations": len(conversations), "questions_per_conversation": questions,
                   "answer_llm": getattr(answer_llm, "name", None) if answer_llm else None,
                   "answer_model": getattr(answer_llm, "model", None) if answer_llm else None,
                   "answer_prompt": _qualname(answer_prompt),
                   "judge": getattr(judge, "__qualname__", repr(judge)),
                   "judge_function": _qualname(judge),
                   "search_deciders": names,
                   "context": use_context, "when": when, "evidence_tokens": evidence_tokens,
                   "compare_evidence_tokens": compare_evidence_tokens,
                   "compare_answer_model": (getattr(compare_answer_llm, "model", None)
                                            if compare_answer_llm else None),
                   "descriptions": descriptions,
                   "question_keys": question_keys,
                   "locomo_categories": LOCOMO_CATEGORIES if dataset == "locomo" else None},
        "stores": stores,
        "passes": tables,
        "tables": tables[0]["tables"],
        "rows": rows,
        "warnings": warnings,
        "notes": notes,
        "complete": stopped is None,
        "stopped": stopped,
    }


def merge_results(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """One result from per-conversation results of the same run: their rows,
    stores, warnings and notes together, the tables computed again."""
    if not parts:
        raise ValueError("no results to merge")
    rows = [row for part in parts for row in part["rows"]]
    names = parts[0]["config"].get("search_deciders") or ["store"]
    tables = pass_tables(rows, names)
    stopped = [part["stopped"] for part in parts if part.get("stopped")]
    return {
        **{key: parts[0][key] for key in ("dataset", "file") if key in parts[0]},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {**parts[0]["config"], "conversations": len(parts)},
        "stores": [store for part in parts for store in part["stores"]],
        "passes": tables,
        "tables": tables[0]["tables"],
        "rows": rows,
        "warnings": [w for part in parts for w in part["warnings"]],
        "notes": list(dict.fromkeys(n for part in parts for n in part["notes"])),
        "complete": all(part.get("complete", True) for part in parts),
        "stopped": "; ".join(stopped) or None,
    }


#: Audit error types that leave the gold answer right (only the evidence
#: cited is wrong): a score does not change.
AUDIT_KEEPS = frozenset({"WRONG_CITATION"})


def load_corrections(path: str | os.PathLike[str],
                     dataset_path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """A corrected LoCoMo answer key in dial481/locomo-audit's ``errors.json``
    form (``question_id`` "locomo_<conversation index>_qa<question index>",
    ``error_type``, ``correct_answer``, ...), as {qid: entry} for the entries
    whose error changes a score (every type but ``AUDIT_KEEPS``). The
    conversation index is the position in ``dataset_path``."""
    entries = _read_json(path)
    if not isinstance(entries, list):
        raise FormatError(f"{path}: expected a JSON list of audit entries")
    samples = _items(_read_json(dataset_path), "qa")
    ids = [str(s.get("sample_id") or f"sample-{i}") for i, s in enumerate(samples)]
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("error_type") in AUDIT_KEEPS or not entry.get("correct_answer"):
            continue
        found = re.fullmatch(r"locomo_(\d+)_qa(\d+)", str(entry.get("question_id")))
        if not found or int(found.group(1)) >= len(ids):
            raise FormatError(f"{path}: question_id {entry.get('question_id')!r} names no question")
        out[f"{ids[int(found.group(1))]}/q{int(found.group(2))}"] = entry
    return out


# --------------------------------------------------------------------------
# what a run was: <out>.meta.json


#: The prices read by default (--prices): USD per million tokens per model.
PRICES = HERE / "prices.json"
#: The provider a call is paid to, by the host it went to.
PROVIDERS = {"api.openai.com": "openai", "api.typesafe.ai": "jev"}


def meta_path(out: str | os.PathLike[str]) -> pathlib.Path:
    """Where a results file's metadata goes: beside it, ``x.json`` ->
    ``x.meta.json`` (another name gets ``.meta.json`` added)."""
    out = pathlib.Path(out)
    stem = out.name[:-len(".json")] if out.name.endswith(".json") else out.name
    return out.with_name(f"{stem}.meta.json")


def file_sha256(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def memry_commit(root: str | os.PathLike[str] = HERE.parent) -> dict[str, Any]:
    """The commit the code runs from, its branch, and the tracked files that
    differ from it (``changed``); None each where git cannot say."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()

    try:
        commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
        changed = [line[3:] for line in git("status", "--porcelain",
                                            "--untracked-files=no").splitlines()]
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "changed": None}
    return {"commit": commit, "branch": branch, "changed": changed}


def load_prices(path: str | os.PathLike[str]) -> dict[str, Any]:
    """A prices file: {"models": {name: {"input", "cached_input", "output",
    "retrieved", ...}}, "aliases": {name asked for: name priced}}, in USD per
    million tokens."""
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise FormatError(f"{path}: expected {{\"models\": {{name: prices}}}}")
    return data


def price_of(model: str | None, prices: dict[str, Any]) -> tuple[str | None, dict | None]:
    """The name a model is priced under and its prices: the model itself,
    else its alias; (None, None) when the file has neither."""
    models = prices.get("models") or {}
    name = model if model in models else (prices.get("aliases") or {}).get(model or "")
    return (name, models[name]) if name in models else (None, None)


def dollars(price: dict[str, Any] | None, input_tokens: int | None, cached_tokens: int | None,
            output_tokens: int | None) -> float | None:
    """USD for these tokens: input tokens less the cached ones at "input",
    cached ones at "cached_input" ("input" where none is given), output
    tokens at "output" (none given: not priced, as Jev's single price is
    applied to input tokens only). Reasoning tokens are part of the output
    tokens. None without a price."""
    if price is None:
        return None
    total_in = input_tokens or 0
    cached = min(cached_tokens or 0, total_in)
    rate_in = float(price.get("input") or 0.0)
    rate_cached = float(price.get("cached_input", rate_in))
    rate_out = float(price.get("output") or 0.0)
    return ((total_in - cached) * rate_in + cached * rate_cached
            + (output_tokens or 0) * rate_out) / 1_000_000


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def ledger_usage(path: str | os.PathLike[str], prices: dict[str, Any]) -> dict[str, Any]:
    """Calls, tokens and dollars of a whole ledger (``api_usage``): per stage,
    per model and per provider, the models the replies named
    (``served_models``), and the first and last call. A resumed run's ledger
    holds every segment of it, so this is the run's whole spend."""
    db = sqlite3.connect(str(path), timeout=120)
    try:
        columns = {row[1] for row in db.execute("PRAGMA table_info(calls)")}
        served = "served_model" if "served_model" in columns else "NULL"
        grouped = db.execute(
            "SELECT grp, host, model, stage, count(*), sum(ok = 0), sum(input_tokens), "
            "sum(output_tokens), sum(cached_tokens), sum(reasoning_tokens), sum(seconds) "
            "FROM calls GROUP BY grp, host, model, stage ORDER BY grp, model, stage").fetchall()
        named = db.execute(f"SELECT model, {served}, count(*) FROM calls GROUP BY 1, 2 "
                           "ORDER BY 1, 2").fetchall()
        first, last = db.execute("SELECT min(started), max(started + seconds) FROM calls"
                                 ).fetchone()
    finally:
        db.close()
    by_stage: list[dict[str, Any]] = []
    by_model: dict[tuple[str, str, str], dict[str, Any]] = {}
    for grp, host, model, stage, calls, failed, tin, tout, tcached, treason, secs in grouped:
        priced_as, price = price_of(model, prices)
        row = {"grp": grp, "provider": PROVIDERS.get(host or "", "jev" if grp == "jev" else host),
               "model": model, "stage": stage, "calls": calls, "failed": failed or 0,
               "input_tokens": tin or 0, "output_tokens": tout or 0, "cached_tokens": tcached or 0,
               "reasoning_tokens": treason or 0, "seconds": round(secs or 0.0, 3)}
        cost = dollars(price, tin, tcached, tout)
        by_stage.append({**row, "usd": None if cost is None else round(cost, 6)})
        total = by_model.setdefault((grp, row["provider"], model), {
            "grp": grp, "provider": row["provider"], "model": model, "priced_as": priced_as,
            "price": price, "calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "cached_tokens": 0, "reasoning_tokens": 0, "usd": None if price is None else 0.0})
        for key in ("calls", "failed", "input_tokens", "output_tokens", "cached_tokens",
                    "reasoning_tokens"):
            total[key] += row[key]
        if cost is not None:
            total["usd"] += cost
    models = list(by_model.values())
    providers: dict[str, float] = {}
    for entry in models:
        if entry["usd"] is not None:
            entry["usd"] = round(entry["usd"], 6)
            providers[entry["provider"]] = providers.get(entry["provider"], 0.0) + entry["usd"]
    calls: dict[str, int] = {}
    for entry in models:
        calls[entry["grp"]] = calls.get(entry["grp"], 0) + entry["calls"]
    served_models = [{"model": model, "served_model": name, "calls": count}
                     for model, name, count in named]
    return {
        "ledger": str(path),
        "calls": {**calls, "all": sum(calls.values())},
        "first_call": _iso(first), "last_call": _iso(last),
        "span_seconds": round(last - first, 1) if first is not None and last is not None else None,
        "usd": {**{name: round(value, 6) for name, value in sorted(providers.items())},
                "total": round(sum(providers.values()), 6)},
        "unpriced": [{"model": e["model"], "calls": e["calls"]} for e in models
                     if e["usd"] is None],
        "by_model": models,
        "by_stage": by_stage,
        "served_models": served_models,
    }


def _percentile(values: list[float], at: float) -> float | None:
    return round(float(np.percentile(values, at)), 3) if values else None


def _answer_accuracy(table: dict[str, Any], **about: Any) -> dict[str, Any]:
    keep = ("n", "judge", "judge_std", "judge_runs", "judge_corrected", "judge_clean",
            "judge_task_averaged", "f1", "context_tokens")
    return {**about, **{key: table[key] for key in keep if key in table}}


def run_headline(result: dict[str, Any]) -> dict[str, Any]:
    """The numbers a run is cited by, from its first question pass: J (the
    judge's share right) of each answer model at the headline k, recall@5,
    10 and 20, MRR, the median and 95th percentile of a search's
    milliseconds, and the mean context tokens of an answer. The questions
    scored are those of the tables' overall row."""
    config = result["config"]
    part = (result.get("passes") or [{"search_decider": None, "tables": result["tables"]}])[0]
    overall = part["tables"]["overall"]
    name = part.get("search_decider")
    rows = [r for r in result["rows"] if r.get("search_decider", name) == name]
    scored = [r for r in rows if r.get("category_name") not in ABSTAIN_CATEGORIES]
    k = config.get("k")
    evidence = sorted({s.get("evidence_tokens") for s in result.get("stores") or []
                       if s.get("evidence_tokens") is not None})
    store_budget = config.get("evidence_tokens")
    if store_budget is None and len(evidence) == 1:
        store_budget = evidence[0]
    answers = []
    if any("prediction" in r for r in rows):
        model = config.get("answer_model") or config.get("answer_llm")
        answers.append(_answer_accuracy(overall, answers="answer", model=model, k=k,
                                        evidence_tokens=store_budget))
        compared = (part.get("compared_by_k") or {}).get(str(k))
        if compared:
            budget = config.get("compare_evidence_tokens")
            answers.append(_answer_accuracy(
                compared["overall"], answers="answer:compared",
                model=config.get("compare_answer_model") or model, k=k,
                evidence_tokens=store_budget if budget is None else budget))
    times = [r["search_ms"] for r in scored if r.get("search_ms") is not None]
    return {
        "pass": name,
        "answers": answers,
        "recall@5": overall.get("recall@5"), "recall@10": overall.get("recall@10"),
        "recall@20": overall.get("recall@20"), "mrr": overall.get("mrr"),
        "search_ms_p50": _percentile(times, 50), "search_ms_p95": _percentile(times, 95),
        "context_tokens_mean": overall.get("context_tokens"),
    }


def _seen(stores: list[dict[str, Any]], key: str) -> Any:
    """A store setting as the stores had it: one value when all agree, the
    list of values when they differ, None without stores."""
    values: list[Any] = []
    for store in stores:
        if store.get(key) not in values:
            values.append(store.get(key))
    return values[0] if len(values) == 1 else (values or None)


def run_meta(result: dict[str, Any], args: argparse.Namespace, *, argv: list[str],
             dataset_path: str | os.PathLike[str], out: str | os.PathLike[str],
             started: float, finished: float, prices: dict[str, Any],
             prices_path: str | os.PathLike[str]) -> dict[str, Any]:
    """What a run was and what it gave, for ``<out>.meta.json``: the data
    and its sha256, the questions and how they were chosen, the memry commit,
    the settings, the times, the ledger's calls, tokens and dollars at
    ``prices``, and the headline numbers (``run_headline``)."""
    config, stores = result["config"], result.get("stores") or []
    headline = run_headline(result)
    part = (result.get("passes") or [{}])[0]
    asked = sum(1 for r in result["rows"]
                if r.get("search_decider", part.get("search_decider")) == part.get("search_decider"))
    answer_models = [m for m in (config.get("answer_model") or config.get("answer_llm"),
                                 config.get("compare_answer_model")) if m]
    meta: dict[str, Any] = {
        "dataset": result.get("dataset"),
        "file": str(dataset_path),
        "file_sha256": file_sha256(dataset_path),
        "questions": {
            "asked": asked,
            "scored": (part.get("tables") or result["tables"])["overall"]["n"],
            "conversations": len(stores),
            "selection": {
                "limit": args.limit, "sample": args.sample, "seed": args.seed,
                "categories": sorted(args.category_set) if args.category_set else None,
                "questions_per_conversation": args.questions,
                "conversation": args.conversation,
                "variant": args.variant if args.dataset == "longmemeval" else None,
                "conversation_ids": [s.get("conversation") for s in stores],
            },
        },
        "memry": memry_commit(),
        "config": {
            "ingest": config.get("ingest"), "extract_unit": config.get("extract_unit"),
            "embedder": config.get("embedder"),
            "text_model": _seen(stores, "text_model"), "text_effort": _seen(stores, "text_effort"),
            "decider": args.decider, "decider_in_stores": _seen(stores, "decider"),
            "decider_model": _seen(stores, "decider_model"),
            "search_deciders": config.get("search_deciders"),
            "k": config.get("k"), "ks": config.get("ks"), "depth": config.get("depth"),
            "evidence_tokens": config.get("evidence_tokens"),
            "evidence_tokens_in_stores": _seen(stores, "evidence_tokens"),
            "compare_evidence_tokens": config.get("compare_evidence_tokens"),
            "answer_models": answer_models,
            "answer_prompt": config.get("answer_prompt"),
            "judge": config.get("judge_function") or config.get("judge"),
            "judge_runs": config.get("judge_runs"),
            "audit": ({"file": args.audit, "sha256": file_sha256(args.audit),
                       "questions": config.get("audited_questions")} if args.audit else None),
            "full_context": config.get("full_context"), "context": config.get("context"),
            "when": config.get("when"), "descriptions": config.get("descriptions"),
            "question_keys": args.question_keys,
            "question_keys_in_stores": _seen(stores, "question_keys"),
            "entity_questions_in_stores": _seen(stores, "entity_questions"),
            "workers": args.workers, "jobs": args.jobs, "max_calls": args.caps or None,
        },
        "command": ["python", "-m", "evals.external_benchmarks", *argv],
        "time": {"started": _iso(started), "finished": _iso(finished),
                 "wall_seconds": round(finished - started, 1)},
        "prices": {"file": str(prices_path), "sha256": file_sha256(prices_path),
                   "retrieved": {name: entry.get("retrieved")
                                 for name, entry in (prices.get("models") or {}).items()}},
        "usage": (ledger_usage(args.usage_db, prices)
                  if args.usage_db and pathlib.Path(args.usage_db).exists() else None),
        "results": {**headline, "complete": result.get("complete"),
                    "stopped": result.get("stopped")},
        "out": str(out),
    }
    return meta


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _listed(value: Any) -> str:
    return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)


def meta_lines(meta: dict[str, Any]) -> list[str]:
    """The metadata as the first lines of the printed results."""
    config, results, usage = meta["config"], meta["results"], meta.get("usage")
    memry = meta["memry"]
    commit = (memry.get("commit") or "unknown")[:12]
    if memry.get("changed"):
        commit += f" with {len(memry['changed'])} changed files"
    selection = meta["questions"]["selection"]
    chosen = ", ".join(f"{key} {_listed(value)}" for key, value in selection.items()
                       if value is not None and key != "conversation_ids") or "all"
    lines = [
        f"## run: {meta['dataset']}, {meta['questions']['scored']} questions scored of "
        f"{meta['questions']['asked']} asked, {meta['questions']['conversations']} conversations"
        f" ({chosen})",
        f"data: {meta['file']} (sha256 {meta['file_sha256']})",
        f"memry: {commit} on {memry.get('branch')}",
        f"time: {meta['time']['started']} to {meta['time']['finished']}, "
        f"{meta['time']['wall_seconds']:.0f} s",
        f"config: ingest {config['ingest']}, text model {config['text_model']}, embedder "
        f"{config['embedder']}, decider {_listed(config['decider_in_stores'] or config['decider'])}"
        + (f" {_listed(config['decider_model'])}" if config["decider_model"] else "")
        + f", k {config['k']} (ks {_listed(config['ks'])}), evidence tokens "
        f"{_listed(config['evidence_tokens_in_stores'])}, question keys "
        f"{config['question_keys_in_stores']}, entity questions "
        f"{config['entity_questions_in_stores']}, descriptions {config['descriptions']}",
        f"answers: {', '.join(config['answer_models']) or 'none'}, prompt "
        f"{config['answer_prompt']}; judge {config['judge']} x{config['judge_runs']}",
    ]
    for entry in results["answers"]:
        lines.append(f"  J {_number(entry.get('judge'))} at k {entry['k']}: {entry['model']} "
                     f"({entry['answers']}, evidence tokens {entry['evidence_tokens']}, "
                     f"n {entry.get('n')}, context tokens {_number(entry.get('context_tokens'), 0)})")
    lines.append(f"search: recall@5 {_number(results['recall@5'])}, recall@10 "
                 f"{_number(results['recall@10'])}, recall@20 {_number(results['recall@20'])}, "
                 f"p50 {_number(results['search_ms_p50'], 0)} ms, p95 "
                 f"{_number(results['search_ms_p95'], 0)} ms, context tokens "
                 f"{_number(results['context_tokens_mean'], 0)}")
    if usage is None:
        lines.append("usage: no ledger (--usage-db)")
    else:
        groups = ", ".join(f"{grp} {n}" for grp, n in usage["calls"].items() if grp != "all")
        spend = ", ".join(f"{name} ${value:.4f}" for name, value in usage["usd"].items())
        lines.append(f"usage: {usage['calls']['all']} calls ({groups or 'none'}), {spend}; "
                     f"ledger {usage['ledger']}")
        for entry in usage["by_model"]:
            served = ", ".join(row["served_model"] for row in usage["served_models"]
                               if row["model"] == entry["model"] and row["served_model"])
            lines.append(f"  {entry['model']}: {entry['calls']} calls, {entry['input_tokens']} in "
                         f"({entry['cached_tokens']} cached), {entry['output_tokens']} out, "
                         + ("no price" if entry["usd"] is None else f"${entry['usd']:.4f}")
                         + (f", served {served}" if served else ""))
    lines.append(f"prices: {meta['prices']['file']}; meta: {meta_path(meta['out'])}")
    if not results.get("complete", True):
        lines.append(f"stopped: {results.get('stopped')}")
    return lines


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
                        help="memories given to the answering model (default 10); with "
                             "--ks, the headline k of the tables")
    parser.add_argument("--ks", default=None, metavar="K,K,...",
                        help="answer from the top k of the same search for each of these "
                             "(e.g. 10,20,30); every answer is scored")
    parser.add_argument("--judge-runs", type=int, default=1,
                        help="judge each answer this many times (default 1)")
    parser.add_argument("--categories", default=None, metavar="C,C,...",
                        help="ask only questions of these categories (numbers or names), "
                             "e.g. 1,2,3,4")
    parser.add_argument("--decider", choices=STORE_DECIDERS, default="config",
                        help="the store's decision provider at the save and the question: "
                             "the configured one (default), none, or Jev (TYPESAFE_API_KEY)")
    parser.add_argument("--store-dir", default=None,
                        help="keep each conversation's store in <dir>/<conversation>.sqlite "
                             "(default: in memory)")
    parser.add_argument("--full-context", action="store_true",
                        help="answer from the whole conversation: no store, no search")
    parser.add_argument("--audit", default=None, metavar="ERRORS_JSON",
                        help="a corrected answer key (dial481/locomo-audit errors.json): "
                             "its questions are judged against the correction too")
    parser.add_argument("--workers", type=int, default=1,
                        help="a question's answer and judge calls made at once (default 1)")
    parser.add_argument("--export-mem0", default=None, metavar="PATH",
                        help="also write the headline answers as Mem0's per-question "
                             "results file (evals/mem0_judge.export_results)")
    parser.add_argument("--write-selected", default=None, metavar="PATH",
                        help="write the file's items of the questions selected (--sample, "
                             "--seed, --limit, --conversation) to PATH as they are, and stop "
                             "(LongMemEval): a small file for a run in several processes")
    parser.add_argument("--export-longmemeval", default=None, metavar="PATH",
                        help="also write the headline answers as the jsonl LongMemEval's "
                             "evaluate_qa.py reads (and the compared answers to "
                             "PATH.compared.jsonl, if any)")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N conversations (LoCoMo samples, LongMemEval questions)")
    parser.add_argument("--conversation", action="append", default=None, metavar="ID",
                        help="only this conversation (a LoCoMo sample_id, a LongMemEval "
                             "question_id); may be given several times")
    parser.add_argument("--questions", type=int, default=None,
                        help="first N questions of each conversation")
    parser.add_argument("--seed", type=int, default=None,
                        help="shuffle the conversations with this seed before --limit; "
                             "with --sample, also the seed of the draw (default 0)")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="N conversations of one question (LongMemEval) drawn in "
                             "proportion to the question types, abstention questions apart "
                             "(stratified_sample), before --seed's shuffle and --limit")
    parser.add_argument("--answer", action="store_true",
                        help="answer with the configured LLM and score F1, EM, contains, judge")
    parser.add_argument("--answer-model", default=None, metavar="MODEL",
                        help="answer with this OpenAI chat model at temperature 0 "
                             "(OPENAI_API_KEY); implies --answer")
    parser.add_argument("--answer-prompt", default=None, metavar="MODULE:FUNCTION",
                        help="function(question, memories) -> the answering call's messages "
                             "(default: the harness's own prompt)")
    parser.add_argument("--context", action="store_true",
                        help="with --answer: answer from reconstruct_context, not the top k")
    parser.add_argument("--evidence-tokens", type=int, default=None, metavar="N",
                        help="the source turns shown with the memories, at most N tokens "
                             "(retrieval.evidence_tokens; 0 shows none; default: the "
                             "store's own)")
    parser.add_argument("--compare-evidence-tokens", type=int, default=None, metavar="N",
                        help="answer each question again from the same search with the "
                             "source turns within N tokens (0: the memories alone), "
                             "kept under the row's answers_compared; no further search")
    parser.add_argument("--compare-answer-model", default=None, metavar="MODEL",
                        help="answer each question again from the same memory list with "
                             "this OpenAI chat model, kept under the row's answers_compared; "
                             "no further search")
    parser.add_argument("--no-descriptions", action="store_true",
                        help="leave out the descriptions of the entities a question names "
                             "from the memory list (an ablation; default: shown, as an "
                             "agent's context shows them)")
    parser.add_argument("--judge", default=None,
                        help="module:function(question, gold, prediction) -> bool "
                             "(default: containment)")
    parser.add_argument("--search-decider", action="append", choices=SEARCH_DECIDERS,
                        default=None,
                        help="what decides at question time: the store's provider (store, "
                             "the default), none, or Jev (TYPESAFE_API_KEY); given several "
                             "times, every question is asked once per value")
    parser.add_argument("--question-keys", choices=["config", "on", "both"], default="config",
                        help="question keys (retrieval.question_keys): on writes them at "
                             "ingest and reads them; both also asks every pass a second "
                             "time without reading them (<pass>:text-only), from the same "
                             "store; config keeps the store's setting (the default)")
    parser.add_argument("--when", choices=WHEN_POLICIES, default="never",
                        help="write metadata['when'] = session date on every new memory "
                             "without one of its own (always), or not (never, the default)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="conversations run at once, each in its own process "
                             "(needs --results-dir)")
    parser.add_argument("--results-dir", default=None,
                        help="one results file per conversation here; a complete one is "
                             "not run again")
    parser.add_argument("--usage-db", default=None,
                        help="SQLite ledger of every model call (tokens, seconds, stage)")
    parser.add_argument("--max-calls", action="append", default=None, metavar="GROUP=N",
                        help="stop before the call that would pass N calls of GROUP "
                             "(chat, embeddings, jev, other) in the ledger; needs --usage-db")
    parser.add_argument("--prices", default=str(PRICES), metavar="PATH",
                        help="USD per million tokens per model, for the dollars in "
                             "<out>.meta.json (default: evals/prices.json)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None,
                        help=f"results file (default ${DATA_ENV}/results/<dataset>_<time>.json)")
    args = parser.parse_args(argv)
    if args.jobs > 1 and not args.results_dir:
        parser.error("--jobs needs --results-dir")
    if args.max_calls and not args.usage_db:
        parser.error("--max-calls needs --usage-db")
    try:
        args.caps = parse_caps(args.max_calls)
        args.k_list = sorted({int(x) for x in args.ks.split(",") if x.strip()} | {args.k}) \
            if args.ks else [args.k]
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.k_list) < 1 or args.judge_runs < 1 or args.workers < 1:
        parser.error("--k, --ks, --judge-runs and --workers want numbers from 1")
    args.category_set = {c.strip() for c in args.categories.split(",") if c.strip()} \
        if args.categories else None
    return args


def parse_caps(specs: list[str] | None) -> dict[str, int]:
    """["chat=15000", "jev=2000"] -> {"chat": 15000, "jev": 2000}."""
    caps: dict[str, int] = {}
    for spec in specs or []:
        group, _, number = spec.partition("=")
        if group not in ("chat", "embeddings", "jev", "other") or not number.isdigit():
            raise ValueError(f"--max-calls wants GROUP=N (chat, embeddings, jev, other), "
                             f"got {spec!r}")
        caps[group] = int(number)
    return caps


def load_function(spec: str, option: str) -> Callable[..., Any]:
    """``module:function`` -> that function."""
    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise ValueError(f"{option} wants module:function, got {spec!r}")
    function = getattr(importlib.import_module(module_name), name)
    if not callable(function):
        raise ValueError(f"{spec} is not callable")
    return function


#: Said in the prompt of an extraction that a save makes (the harness gives
#: every save its context); the extraction an UPDATE makes to read the
#: rewritten memory's names has none.
SHARED_CONTEXT = "Shared context for these related inputs"


def _memry_prompts() -> list[tuple[str, str]]:
    from memry.intelligence import entities, extraction, reconcile, when

    prompts = [(extraction.EXTRACTION_SYSTEM, "extraction"), (extraction.COVERAGE_SYSTEM, "audit"),
               (extraction.RELATION_SYSTEM, "relations"), (reconcile.RECONCILE_SYSTEM, "reconcile"),
               (entities.IDENTITY_SYSTEM, "identity"),
               (entities.DESCRIPTION_SYSTEM, "entity_description"), (when.WHEN_SYSTEM, "when")]
    return [(text.split("{", 1)[0][:60], name) for text, name in prompts]


def jev_stage(body: dict[str, Any]) -> str:
    """Which memry question a decision call asks, by its question keys and
    state: "reconcile" (the action), "identity_pair" (the pair and belongs
    questions), "identity", "name_screen", "name_check", "entity_types",
    "when" (event or record), "tag", "durability", "relevance" or "other"."""
    keys = set(body.get("questions") or {})
    state = str(body.get("state") or "")
    if "action" in keys:
        return "reconcile"
    if "pair" in keys:
        return "identity_pair"
    for key, name in (("identity", "identity"), ("tag", "tag"), ("k", "when")):
        if key in keys:
            return name
    if state.startswith("A name from"):
        return "name_check"
    if state.startswith("Entity names extracted"):
        return "entity_types"
    prefixes = {key[:1] for key in keys}
    if "s" in prefixes and state.startswith("A memory"):
        return "name_screen"
    if "d" in prefixes:
        return "durability"
    if "m" in prefixes:
        return "relevance"
    return "other"


def memry_stage(stage: str, group: str, body: Any) -> str | None:
    """A call made while a conversation is loaded, counted by what it asks: a
    chat call by the memry prompt it carries ("ingest:extraction",
    "ingest:extraction_on_update", "ingest:reconcile", "ingest:reconcile_merge"
    (writing an UPDATE's text), "ingest:identity", "ingest:audit", ... or
    "ingest:other"), a decision call by its question (``jev_stage``:
    "ingest:jev:reconcile", ...)."""
    if not stage.startswith("ingest") or not isinstance(body, dict):
        return None
    if group == "jev":
        return f"{stage}:jev:{jev_stage(body)}"
    if group != "chat":
        return None
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    system = next((str(m.get("content") or "") for m in messages if m.get("role") == "system"), "")
    user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    from memry.intelligence.reconcile import MERGE_REQUEST

    for prefix, name in _memry_prompts():
        if system.startswith(prefix):
            if name == "extraction" and SHARED_CONTEXT not in user:
                name = "extraction_on_update"
            elif name == "reconcile" and MERGE_REQUEST in user:
                name = "reconcile_merge"
            return f"{stage}:{name}"
    return f"{stage}:other"


def _read_json_file(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    partial.replace(path)


#: Options that do not change a conversation's results: a per-conversation
#: file written under other values of these is still reused.
_RUN_ONLY = ("jobs", "results_dir", "out", "worker", "usage_db", "max_calls", "caps", "prices",
             "conversation", "limit", "seed", "sample", "workers", "store_dir", "export_mem0",
             "export_longmemeval", "k_list", "category_set")


def run_options(args: argparse.Namespace) -> dict[str, Any]:
    """The options a per-conversation results file must match to be reused."""
    return {key: value for key, value in sorted(vars(args).items()) if key not in _RUN_ONLY}


def reusable(path: pathlib.Path, options: dict[str, Any]) -> bool:
    """A complete per-conversation results file written with these options."""
    part = _read_json_file(path)
    return bool(part and part.get("complete") and part.get("config", {}).get("options") == options)


def run_workers(argv: list[str], conversation_ids: list[str], results_dir: pathlib.Path,
                jobs: int, log: Callable[[str], None]) -> None:
    """Run each conversation in a process of its own (this module with
    ``--worker``), ``jobs`` at a time, each logging to <conversation>.log
    beside its results file. Once one stops unfinished (a cap), no more are
    started."""
    queue = list(conversation_ids)
    running: dict[str, tuple[subprocess.Popen, Any]] = {}
    while queue or running:
        while queue and len(running) < jobs:
            conv_id = queue.pop(0)
            handle = open(results_dir / f"{conv_id}.log", "a", encoding="utf-8")
            command = [sys.executable, "-m", "evals.external_benchmarks", *argv,
                       "--worker", conv_id]
            running[conv_id] = (subprocess.Popen(command, cwd=str(HERE.parent), stdout=handle,
                                                 stderr=subprocess.STDOUT), handle)
            log(f"  {conv_id}: started")
        time.sleep(1)
        for conv_id, (process, handle) in list(running.items()):
            if process.poll() is None:
                continue
            handle.close()
            del running[conv_id]
            part = _read_json_file(results_dir / f"{conv_id}.json") or {}
            log(f"  {conv_id}: exit {process.returncode}, "
                f"{'complete' if part.get('complete') else 'not complete'}")
            if not part.get("complete"):
                queue.clear()


def write_selected(dataset: str, path: pathlib.Path, conversations: list[Conversation],
                   target: str | os.PathLike[str]) -> int:
    """Write the items of ``path`` whose question is one of ``conversations``,
    unchanged and in the file's order, to ``target`` (LongMemEval). Each
    process of a run loads its whole data file, and the 500 questions of
    longmemeval_s take about 2.4 GB in memory; a file of the questions to run
    keeps ten processes in a few GB."""
    if dataset != "longmemeval":
        print("--write-selected: LongMemEval only", file=sys.stderr)
        return 2
    keep = {c.conv_id for c in conversations}
    items = [item for i, item in enumerate(_items(_read_json(path), "haystack_sessions"))
             if str(item.get("question_id") or f"question-{i}") in keep]
    out = pathlib.Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{len(items)} questions of {path} written to {out} (sha256 {digest})")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    started = time.time()
    try:
        prices = load_prices(args.prices)
    except (OSError, FormatError) as exc:
        print(f"--prices: {exc}", file=sys.stderr)
        return 2
    data_dir = os.environ.get(DATA_ENV, "").strip() or None
    path = find_dataset(args.dataset, data_dir, variant=args.variant, file=args.file)
    if path is None:
        where = f"{DATA_ENV}={data_dir}" if data_dir else f"{DATA_ENV} is not set"
        wanted = args.file or " or ".join(dataset_files(args.dataset, args.variant)[:2])
        print(f"no {args.dataset} data ({where}): expected {wanted}", file=sys.stderr)
        return 2
    conversations = load(args.dataset, path)
    if args.worker:
        conversations = [c for c in conversations if c.conv_id == args.worker]
    else:
        if args.sample:
            try:
                conversations = stratified_sample(conversations, args.sample, args.seed or 0)
            except ValueError as exc:
                print(f"--sample: {exc}", file=sys.stderr)
                return 2
        if args.seed is not None:
            random.Random(args.seed).shuffle(conversations)
        if args.limit:
            conversations = conversations[:args.limit]
        if args.conversation:
            known = {c.conv_id for c in conversations}
            unknown = [c for c in args.conversation if c not in known]
            if unknown:
                print(f"--conversation: no {', '.join(unknown)} in {path}", file=sys.stderr)
                return 2
            conversations = [c for c in conversations if c.conv_id in set(args.conversation)]
        if args.write_selected:
            return write_selected(args.dataset, path, conversations, args.write_selected)
    judge = load_judge(args.judge)
    answer_prompt = load_function(args.answer_prompt, "--answer-prompt") \
        if args.answer_prompt else None
    corrections = load_corrections(args.audit, path) if args.audit else None
    deciders = {name: search_decider(name)
                for name in dict.fromkeys(args.search_decider or ["store"])}
    options = run_options(args)

    def log(text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    results_dir = pathlib.Path(args.results_dir) if args.results_dir else None
    notes: list[str] = []
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in conversations
                if not reusable(results_dir / f"{c.conv_id}.json", options)]
        log(f"{args.dataset}: {len(conversations)} conversations, {len(todo)} to run "
            f"in {results_dir}")
        if args.jobs > 1 and len(todo) > 1 and not args.worker:
            run_workers(argv, [c.conv_id for c in todo], results_dir, args.jobs, log)
            todo = []
    else:
        todo = conversations
    meter = api_usage.UsageMeter(args.usage_db, label=args.worker or "", caps=args.caps,
                                 refine=memry_stage).install() if args.usage_db and todo else None
    answer_llm = compare_answer_llm = embedder = None
    try:
        if todo:
            if args.answer or args.answer_model:
                if args.answer_model:
                    from evals.mem0_judge import OpenAIChat

                    llm: LLM = OpenAIChat(args.answer_model)
                else:
                    llm = build_llm(Config.load().llm)
                if llm.available:
                    answer_llm = llm
                else:
                    notes.append("--answer skipped: no LLM configured (OPENAI_API_KEY, "
                                 "ANTHROPIC_API_KEY or MEMRY_LLM_PROVIDER)")
                    print(notes[-1], file=sys.stderr)
            if args.compare_answer_model and answer_llm is not None:
                from evals.mem0_judge import OpenAIChat

                compare_answer_llm = OpenAIChat(args.compare_answer_model)
            if answer_llm is not None and (bleu1("a", "a") is None or count_tokens("a") is None):
                notes.append("BLEU-1 or context tokens not computed: install memry[eval] "
                             "(nltk with its punkt_tab data, tiktoken)")
                print(notes[-1], file=sys.stderr)
            if not args.full_context:
                embedder = build_embedder(args.embedder, data_dir)
            log(f"{args.dataset}: {path} ({len(todo)} conversations), ingest "
                f"{'none (full context)' if args.full_context else args.ingest}, embedder "
                f"{embedder.model_id if embedder else None}")
        settings = dict(dataset=args.dataset, mode=args.ingest, unit=args.extract_unit,
                        embedder=embedder, k=args.k, questions=args.questions,
                        answer_llm=answer_llm, judge=judge, use_context=args.context,
                        when=args.when, search_deciders=deciders, answer_prompt=answer_prompt,
                        ks=args.k_list, judge_runs=args.judge_runs,
                        categories=set(args.category_set) if args.category_set else None,
                        workers=args.workers, corrections=corrections, decider=args.decider,
                        store_dir=args.store_dir, full_context=args.full_context,
                        evidence_tokens=args.evidence_tokens,
                        compare_evidence_tokens=args.compare_evidence_tokens,
                        compare_answer_llm=compare_answer_llm,
                        descriptions=not args.no_descriptions,
                        question_keys=args.question_keys)

        def finish(part: dict[str, Any]) -> dict[str, Any]:
            part["file"] = str(path)
            part["config"].update(limit=args.limit, seed=args.seed, options=options,
                                  variant=args.variant if args.dataset == "longmemeval" else None)
            part["notes"] = notes + part["notes"]
            return part

        if results_dir is None:
            result = finish(run_benchmark(conversations, **settings))
        else:
            for conv in todo:
                part = finish(run_benchmark([conv], **settings))
                _write_json(results_dir / f"{conv.conv_id}.json", part)
                if not part["complete"]:
                    break
            if args.worker:
                return 0
            parts = [part for c in conversations
                     if (part := _read_json_file(results_dir / f"{c.conv_id}.json"))]
            if not parts:
                print(f"no results in {results_dir}", file=sys.stderr)
                return 1
            result = merge_results(parts)
            missing = sorted({c.conv_id for c in conversations}
                             - {s["conversation"] for s in result["stores"]})
            if missing:
                result["complete"] = False
                result["notes"].append(f"no results for {', '.join(missing)}")
    finally:
        if isinstance(embedder, SqliteEmbeddingCache):
            embedder.release()
        if answer_llm is not None:
            answer_llm.close()
        if meter is not None:
            meter.close()
    if args.usage_db and pathlib.Path(args.usage_db).exists():
        result["usage"] = {
            "ledger": str(args.usage_db),
            "by_stage": api_usage.summarize(args.usage_db, ("grp", "model", "stage")),
            "by_conversation": api_usage.summarize(args.usage_db,
                                                   ("label", "grp", "model", "stage")),
        }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = pathlib.Path(data_dir) if data_dir else path.parent
    out = pathlib.Path(args.out) if args.out else base / "results" / f"{args.dataset}_{stamp}.json"
    meta = run_meta(result, args, argv=argv, dataset_path=path, out=out, started=started,
                    finished=time.time(), prices=prices, prices_path=args.prices)
    result["meta_file"] = str(meta_path(out))
    _write_json(meta_path(out), meta)
    _write_json(out, result)
    print("\n".join(meta_lines(meta)))
    if args.export_mem0:
        from evals.mem0_judge import export_results

        _write_json(pathlib.Path(args.export_mem0), export_results(result, path))
        print(f"Mem0's results file: {args.export_mem0}")
    if args.export_longmemeval:
        from evals.longmemeval_judge import export_hypotheses

        target = pathlib.Path(args.export_longmemeval)
        for compared, where in ((False, target), (True, target.with_suffix(".compared.jsonl"))):
            lines = export_hypotheses(result, compared=compared)
            if lines or not compared:
                where.parent.mkdir(parents=True, exist_ok=True)
                where.write_text("".join(json.dumps(line) + "\n" for line in lines),
                                 encoding="utf-8")
                print(f"LongMemEval's hypothesis file: {where}")
    print(f"\n## {args.dataset}: {len(result['rows'])} rows, "
          f"{sum(s.get('memories', 0) for s in result['stores'])} memories")
    for part in result["passes"]:
        if len(result["passes"]) > 1:
            print(f"\n### questions asked with search decider {part['search_decider']}")
        print("\n" + markdown_table(part["tables"]))
        for at, tables in (part.get("tables_by_k") or {}).items():
            print(f"\n#### answered from the top {at}\n\n" + markdown_table(tables))
        config = result["config"]
        compared_as = ", ".join(
            [f"the turns within {config['compare_evidence_tokens']} tokens"]
            * (config.get("compare_evidence_tokens") is not None)
            + [f"answered by {config['compare_answer_model']}"]
            * bool(config.get("compare_answer_model")))
        for at, tables in (part.get("compared_by_k") or {}).items():
            print(f"\n#### answered from the top {at}, {compared_as}\n\n"
                  + markdown_table(tables))
    for row in (result.get("usage") or {}).get("by_stage", []):
        print(f"\nusage: {row['grp']} {row['model']} {row['stage']}: {row['calls']} calls, "
              f"{row['input_tokens']} in, {row['output_tokens']} out, {row['seconds']:.0f} s",
              end="")
    for note in result["notes"]:
        print(f"\nnote: {note}")
    if result.get("stopped"):
        print(f"\nstopped: {result['stopped']}")
    if result["warnings"]:
        print(f"\n{len(result['warnings'])} warnings (in the results file); first: "
              f"{result['warnings'][0]}")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
