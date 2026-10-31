"""BEAM (Tavakoli et al., "Beyond a Million Tokens", ICLR 2026) for Memry.

``evals.external_benchmarks --dataset beam`` loads each chat into a store and
answers its 20 questions; this module holds what is BEAM's own:

  fetch    the chats and questions of one size from the Hugging Face datasets
           server, written as $MEMRY_BENCH_DATA/beam_<size>.json; a cell the
           server shortened stops the fetch, so the file is the data whole
  answer   ``answer_messages``: BEAM's answering prompt for retrieval
           (``answer_generation_for_rag``), Memry's context in its place of
           the retrieved chunks; pass it as --answer-prompt
  answer   every question again, from the lines its first answer was shown
           (a harness run with --keep-context), with another prompt
           (``beam``: BEAM's; ``memry``: the harness's own short-answer
           prompt; ``abilities``: ``ABILITIES_SYSTEM``) or another model,
           through OpenAI or OpenRouter; no store is loaded again
  grade    BEAM's grader over a results file: every rubric item of a
           question is scored 0, 0.5 or 1 by the judge model with BEAM's
           ``unified_llm_judge_base_prompt``, and the question scores their
           mean; an event-ordering answer is also scored by Kendall's tau-b
           against the rubric's order, its lines matched to the rubric's
           items by the judge model (``llm_equivalence``), as BEAM's
           ``event_ordering_score`` does

The tables follow BEAM's ``report_results.py``: per chat and ability the mean
question score (event ordering: the normalised tau, ``tau_norm``), then the
mean over chats, and the overall score is the mean of the ten abilities.
Beside it: the same with event ordering by its rubric (as SurrealDB's
"rubric average"), the share of questions with every rubric item at 1
("strict"), and the share scoring at least 0.5 ("pass").

The prompts and the scoring are BEAM's (github.com/mohammadtavakoli78/BEAM,
src/prompts.py and src/evaluation/compute_metrics.py, MIT License, Copyright
(c) 2025 Mohammad Tavakoli). BEAM's grader uses gpt-4.1-mini at temperature 0;
so does this one unless --judge-model says otherwise.

    python -m evals.beam_eval fetch --size 100K
    MEMRY_BENCH_DATA=/data .venv/bin/python -m evals.external_benchmarks --dataset beam \\
        --variant 100K --ingest extract --embedder openai --decider jev --k 20 \\
        --answer-model gpt-4o-mini --compare-answer-model gpt-6-luna \\
        --answer-prompt evals.beam_eval:answer_messages ...
    python -m evals.beam_eval grade RESULTS.json [more ...] --out beam_scores.json \\
        --max-dollars 2
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

ABILITIES = ("abstention", "contradiction_resolution", "event_ordering",
             "information_extraction", "instruction_following", "knowledge_update",
             "multi_session_reasoning", "preference_following", "summarization",
             "temporal_reasoning")
SIZES = ("100K", "500K", "1M")
DATASET = "Mohammadta/BEAM"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
JUDGE_MODEL = "gpt-4.1-mini"
#: USD per million tokens, input and output (OpenAI's list prices).
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60), "gpt-6-luna": (0.10, 0.50),
          "anthropic/claude-opus-5.5": (4.00, 20.00), "anthropic/claude-sonnet-5.5": (2.00, 10.00)}
OPENROUTER_URL = "https://openrouter.ai/api"

# ---------------------------------------------------------------- BEAM's prompts, verbatim

ANSWER_PROMPT = (
    '\n'
    'You are an assistant that MUST answer questions using ONLY the information provided in the context below. \n'
    '\n'
    'STRICT INSTRUCTIONS:\n'
    '1. Answer ONLY based on the provided context\n'
    '2. Do NOT use your internal knowledge\n'
    '\n'
    'CONTEXT:\n'
    '<context>\n'
    '\n'
    'QUESTION:\n'
    '<question>\n'
    '\n'
    'ANSWER REQUIREMENTS:\n'
    '- Be direct and concise\n'
    '- Only output the answer to the question without any explanation \n'
    '\n'
    'RESPONSE:\n'
)

JUDGE_PROMPT = (
    '\n'
    "You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.\n"
    '\n'
    '## EVALUATION INPUTS\n'
    '- RUBRIC CRITERION (what to check): <rubric_item>\n'
    '- RESPONSE TO EVALUATE: <llm_response>\n'
    '\n'
    '## EVALUATION RUBRIC:\n'
    'The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. \n'
    '\n'
    '**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n'
    '- **Positive requirements** (things the response SHOULD include/do)\n'
    '- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")\n'
    '\n'
    '## RESPONSIVENESS REQUIREMENT\n'
    'A compliant response must be **on-topic** and attempt to answer it.\n'
    '- If the response does not address the QUESTION, score **0.0** and stop.\n'
    '- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.\n'
    '\n'
    '## SEMANTIC TOLERANCE RULES:\n'
    'Judge by meaning, not exact wording.\n'
    '- Accept **paraphrases** and **synonyms** that preserve intent.\n'
    '- **Case/punctuation/whitespace** differences must be ignored.\n'
    '- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n'
    '- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.\n'
    '\n'
    '## STYLE NEUTRALITY (prevents style contamination):\n'
    'Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).\n'
    '- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n'
    '- Only evaluate format when the rubric **explicitly** mandates it.\n'
    '\n'
    '## SCORING SCALE:\n'
    '- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n'
    '  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n'
    '  - Negative: prohibited element **absent** AND response is **responsive**.\n'
    '  \n'
    '- **0.5 (Partial Compliance)**: Partially complies.\n'
    '  - Positive: element present but minor inaccuracies/incomplete execution.\n'
    '  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n'
    '  \n'
    '- **0.0 (No Compliance)**: Fails to comply.\n'
    '  - Positive: required element missing or incorrect.\n'
    '  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.\n'
    '\n'
    '## EVALUATION INSTRUCTIONS:\n'
    '1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).\n'
    '\n'
    '2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:\n'
    '   - **All elements** must be present for full compliance (1.0)\n'
    '   - **Some elements** present indicates partial compliance (0.5)\n'
    '   - **No elements** present indicates no compliance (0.0)\n'
    '   \n'
    '3. **Check Compliance**: \n'
    '   - For positive requirements: Look for the presence and quality of the required element\n'
    '   - For negative constraints: Look for the absence of the prohibited element\n'
    '\n'
    '4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.\n'
    '\n'
    '5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n'
    '\n'
    '## OUTPUT FORMAT:\n'
    'Return your evaluation in JSON format with two fields:\n'
    '\n'
    '{\n'
    '   "score": [your score: 1.0, 0.5, or 0.0],\n'
    '   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"\n'
    '}\n'
    '\n'
    'NOTE: ONLY output the json object, without any explanation before or after that\n'
)

EQUIVALENCE_SYSTEM = """
            You are a binary classifier.
            If the TWO snippets describe the SAME event/fact, reply **YES**
            Otherwise reply **NO**. No extra words.
            DO NOT provide any exaplanation.
        """


#: BEAM's user turn of the equivalence call (an f-string there)
EQUIVALENCE_USER = (
    'First snippet: {first} \n'
    '\n'
    '                       Second snippet: {second}\n'
    '                    '
)


# ---------------------------------------------------------------- answering

def context_of(items: list[Any]) -> str:
    """Memry's lines for the question, as its context builder renders them."""
    return "\n".join(item if isinstance(item, str) else str(getattr(item, "content", item))
                     for item in items)


def answer_messages(question: str, items: list[Any]) -> list[dict[str, str]]:
    """BEAM's answering call for retrieval: one user message, its context Memry's."""
    prompt = ANSWER_PROMPT.replace("<context>", context_of(items)).replace("<question>", question)
    return [{"role": "user", "content": prompt}]


answer_messages.context_of = context_of  # type: ignore[attr-defined]


#: An assistant answering from Memry's context, told what BEAM's README says
#: each of its ten abilities asks of a memory (abstention, contradiction
#: resolution, event ordering, information extraction, instruction following,
#: knowledge update, multi-session reasoning, preference following,
#: summarization, temporal reasoning). Written from that list after the first
#: trial, when a few of BEAM's questions and rubrics had been seen. The text
#: a model reads never names BEAM.
ABILITIES_SYSTEM = """You are a personal assistant with a long-term memory of your \
past conversations with this user. Below the user's message is what your memory \
returned for it: memories with the date each was said, and quoted parts of the \
conversations they came from. Answer the user from this context.

- If the context does not hold what the user asks about, say that you do not \
know it. Do not guess.
- If the user said contradictory things at different times and the context does \
not show which one holds now, point out the contradiction, say both statements, \
and ask which one is correct.
- If a fact changed over time, answer with the latest value.
- Follow the instructions and preferences the user gave earlier about how you \
should answer (format, style, length, tools, level of detail), even when this \
message does not repeat them.
- When asked in what order things happened or were discussed, list them in that \
order.
- When asked for a summary, cover the main points across all the conversations \
in the context.
- For questions about time, work out the dates and durations from the dates in \
the context.
- When the answer needs facts from several conversations, combine them.

Answer the way the user asked, completely but without padding."""


def _today(reference_date: str | None) -> str:
    try:
        return f"Today is {datetime.fromisoformat(str(reference_date)):%d %B %Y}.\n"
    except (TypeError, ValueError):
        return ""


def abilities_messages(question: str, items: list[Any],
                       reference_date: str | None = None) -> list[dict[str, str]]:
    """``ABILITIES_SYSTEM``, then the user's message with Memry's context."""
    return [{"role": "system", "content": ABILITIES_SYSTEM},
            {"role": "user", "content": f"{question}\n\n{_today(reference_date)}"
                                        f"What your memory returned:\n{context_of(items)}"}]


def memry_messages(question: str, items: list[Any],
                   reference_date: str | None = None) -> list[dict[str, str]]:
    """The harness's own answering call (``external_benchmarks.answer_question``),
    which Memry's LoCoMo and LongMemEval runs use: a short phrase from the
    memories, or "No information available"."""
    from evals.external_benchmarks import ANSWER_SYSTEM, memories_text

    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{memories_text(items)}\n\n{_today(reference_date)}"
                                        f"Question: {question}\nShort answer:"}]


#: --prompt of ``answer``: (question, the lines Memry gave, reference date) -> messages.
PROMPTS: dict[str, Callable[..., list[dict[str, str]]]] = {
    "beam": lambda question, items, reference_date=None: answer_messages(question, items),
    "memry": memry_messages,
    "abilities": abilities_messages,
}


# ---------------------------------------------------------------- fetching

def fetch(size: str, out: pathlib.Path, *, page: int = 5,
          get: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every row of one size, whole, as a JSON list at ``out``."""
    if get is None:
        import httpx

        client = httpx.Client(timeout=300)

        def get(params: dict[str, Any]) -> dict[str, Any]:
            resp = client.get(ROWS_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    rows: list[dict[str, Any]] = []
    total = None
    while total is None or len(rows) < total:
        data = get({"dataset": DATASET, "config": "default", "split": size,
                    "offset": len(rows), "length": page})
        total = int(data["num_rows_total"])
        batch = data.get("rows") or []
        if not batch:
            raise RuntimeError(f"the server stopped at row {len(rows)} of {total}")
        for item in batch:
            if item.get("truncated_cells"):
                raise RuntimeError(f"row {item.get('row_idx')}: the server shortened "
                                   f"{item['truncated_cells']}; read the parquet file instead")
            rows.append(item["row"])
    text = json.dumps(rows, ensure_ascii=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return {"rows": len(rows), "path": str(out),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


# ---------------------------------------------------------------- grading

class Budget:
    """Dollars spent by the judge model, from the usage each reply reports;
    ``charge`` raises once the total passes ``cap``."""

    def __init__(self, model: str, cap: float | None) -> None:
        self.model = model
        self.price = PRICES.get(model)
        self.cap = cap
        self.dollars = 0.0
        self.calls = 0
        self.tokens = [0, 0]
        self._lock = threading.Lock()

    def charge(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        with self._lock:
            self.calls += 1
            self.tokens[0] += tin
            self.tokens[1] += tout
            if self.price:
                self.dollars += (tin * self.price[0] + tout * self.price[1]) / 1e6
            if self.cap is not None and self.dollars > self.cap:
                raise CapReached(f"${self.dollars:.2f} of {self.model} calls passes the cap of "
                                 f"${self.cap:.2f}")


class CapReached(RuntimeError):
    pass


def parse_score(text: str) -> float | None:
    """The "score" of the judge's JSON reply, as BEAM's parser and its
    json_repair fallback read it; None when there is none."""
    text = text.strip()
    if text.startswith("```"):
        found = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
        text = found.group(1) if found else text
    for candidate in (text, *re.findall(r"\{.*?\}", text, re.S)):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and "score" in value:
                return float(value["score"])
        except (ValueError, TypeError):
            continue
    found = re.search(r'"?score"?\s*:\s*\[?\s*"?([01](?:\.\d+)?|0?\.5)', text)
    return float(found.group(1)) if found else None


Chat = Callable[[list[dict[str, str]]], tuple[str, dict[str, Any] | None]]


def openai_chat(model: str, provider: str = "openai") -> Chat:
    """A chat call through OpenAI, or through OpenRouter (``OPENROUTER_API_KEY``)
    for the models it serves, such as anthropic/claude-opus-5.5."""
    from evals.mem0_judge import OpenAIChat

    if provider == "openrouter":
        llm = OpenAIChat(model, api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                         base_url=OPENROUTER_URL, timeout=600)
    else:
        llm = OpenAIChat(model)

    def chat(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any] | None]:
        return llm.chat(messages), llm.last_usage()

    return chat


#: The question a judge call is made for, so a ``FileJudge`` files it under it.
ASKED_FOR: contextvars.ContextVar[str] = contextvars.ContextVar("asked_for", default="")


def judge_key(messages: list[dict[str, str]]) -> str:
    """A judge call's key: the hash of its messages, so the same call made again,
    for another arm or another pass, has the same reply."""
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


class FileJudge:
    """A judge whose replies are written outside this process (--judge-files),
    such as by a model with no API key. A call whose key (``judge_key``) is in
    ``replies`` gets that reply. Any other call gets an empty reply and is kept
    in ``asked`` (question -> key -> messages), to be answered before grading
    again. An empty reply has no score and calls no two lines the same event, so
    the first pass asks every rubric item and every pair an event-ordering answer
    can need."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.asked: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
        self._lock = threading.Lock()

    def __call__(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any] | None]:
        key = judge_key(messages)
        if key in self.replies:
            return self.replies[key], None
        with self._lock:
            self.asked[ASKED_FOR.get()][key] = messages
        return "", None


def read_replies(folder: pathlib.Path) -> dict[str, str]:
    """Every reply in <folder>/replies/*.json (key -> reply text)."""
    replies: dict[str, str] = {}
    for file in sorted((folder / "replies").glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        replies.update({str(k): v if isinstance(v, str) else json.dumps(v)
                        for k, v in data.items()})
    return replies


def judge_item(chat: Chat, budget: Budget, item: str, response: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", response)
    for attempt in range(2):
        text, usage = chat([{"role": "user", "content": prompt}])
        budget.charge(usage)
        score = parse_score(text or "")
        if score is not None:
            return {"score": score, "reply": text}
    return {"score": 0.0, "reply": text, "error": "no score in the reply"}


def equivalent(chat: Chat, budget: Budget, first: str, second: str) -> bool:
    text, usage = chat([{"role": "system", "content": EQUIVALENCE_SYSTEM},
                        {"role": "user", "content": EQUIVALENCE_USER.format(first=first, second=second)}])
    budget.charge(usage)
    return "yes" in (text or "").lower()


def kendall_tau_b(x: list[int], y: list[int]) -> float | None:
    """Kendall's tau-b of two rankings (scipy.stats.kendalltau, variant "b")."""
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)) ** 0.5
    return (concordant - discordant) / denominator if denominator else None


def event_order(chat: Chat, budget: Budget, reference: list[str], response: str) -> dict[str, Any]:
    """BEAM's ``event_ordering_score`` with ``align_type="llm"``: each line of
    the answer takes the first unused rubric item the judge model calls the
    same event, and the order is scored over the union of both lists."""
    system = response.split("\n")
    used: set[int] = set()
    canon: list[str] = []
    for line in system:
        match = next((i for i, ref in enumerate(reference)
                      if i not in used and equivalent(chat, budget, ref, line)), None)
        if match is None:
            canon.append(line)
        else:
            canon.append(reference[match])
            used.add(match)
    tp = len(set(reference) & set(canon))
    fp = len([x for x in canon if x not in reference])
    fn = len([x for x in reference if x not in canon])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    union = list(dict.fromkeys(reference + canon))
    tie = len(union) + 1

    def ranks(seq: list[str]) -> list[int]:
        rank = {item: i + 1 for i, item in enumerate(seq)}
        return [rank.get(u, tie) for u in union]

    tau = kendall_tau_b(ranks(reference), ranks(canon))
    tau_norm = (tau + 1) / 2 if tau is not None else 0
    return {"precision": precision, "recall": recall, "f1": f1, "tau_norm": tau_norm,
            "final_score": tau_norm * f1}


def grade_answer(chat: Chat, budget: Budget, ability: str, rubric: list[str],
                 response: str) -> dict[str, Any]:
    items = [judge_item(chat, budget, item, response) for item in rubric]
    scores = [i["score"] for i in items]
    out: dict[str, Any] = {"llm_judge_score": sum(scores) / len(scores), "item_scores": scores,
                           "replies": [i["reply"] for i in items]}
    if any("error" in i for i in items):
        out["errors"] = sum("error" in i for i in items)
    if ability == "event_ordering":
        out.update(event_order(chat, budget, rubric, response))
    return out


def official(grade: dict[str, Any], ability: str) -> float:
    """The question's score in BEAM's report: event ordering by tau_norm."""
    return grade["tau_norm"] if ability == "event_ordering" else grade["llm_judge_score"]


def answers_of(row: dict[str, Any], *, failed: bool = False) -> Iterable[tuple[str, str]]:
    """(variant, prediction) of each answer a results row carries; with
    ``failed``, the answers whose call failed (``answer_error``) instead,
    which are never graded: an empty answer there is no answer at all."""
    answers = [("answer", row)] + [(f"compared@{at}", a) for at, a in
                                   sorted((row.get("answers_compared") or {}).items())]
    for variant, answer in answers:
        if answer.get("prediction") is not None and bool(answer.get("answer_error")) == failed:
            yield variant, str(answer["prediction"])


def read_rows(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    for spec in paths:
        path = pathlib.Path(spec)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            data = json.loads(file.read_text(encoding="utf-8"))
            config = config or data.get("config") or {}
            rows.extend(r for r in data.get("rows") or [] if r.get("rubric"))
    return rows, config


def tables(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """Per answer variant: BEAM's report (per chat and ability, then over
    chats; the overall mean of the abilities), with event ordering by its
    rubric beside it, and the strict and pass shares."""
    out: dict[str, Any] = {}
    for variant in sorted({g["variant"] for g in graded}):
        mine = [g for g in graded if g["variant"] == variant]
        per: dict[str, dict[str, Any]] = {}
        for kind, score_of in (("official", lambda g: official(g, g["ability"])),
                               ("rubric", lambda g: g["llm_judge_score"])):
            chats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for g in mine:
                chats[g["conversation"]][g["ability"]].append(score_of(g))
            abilities = {}
            for ability in ABILITIES:
                means = [statistics.mean(c[ability]) for c in chats.values() if c.get(ability)]
                if means:
                    abilities[ability] = statistics.mean(means)
            per[kind] = {"abilities": abilities,
                         "overall": statistics.mean(abilities.values()) if abilities else None}
        out[variant] = {
            "n": len(mine), "chats": len({g["conversation"] for g in mine}),
            "official": per["official"], "rubric_average": per["rubric"],
            "strict": statistics.mean(all(s == 1.0 for s in g["item_scores"]) for g in mine),
            "pass": statistics.mean(official(g, g["ability"]) >= 0.5 for g in mine),
            "judge_errors": sum(g.get("errors", 0) for g in mine)}
    return out


def grade(paths: list[str], out: pathlib.Path, *, judge_model: str = JUDGE_MODEL,
          max_dollars: float | None = None, workers: int = 8,
          chat: Chat | None = None) -> dict[str, Any]:
    """Grade every answer of the results files; graded answers already in
    ``out`` are kept, so an interrupted grading resumes."""
    rows, config = read_rows(paths)
    chat = chat or openai_chat(judge_model)
    budget = Budget(judge_model, max_dollars)
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if out.exists():
        for g in json.loads(out.read_text(encoding="utf-8")).get("graded") or []:
            done[(g["qid"], g["variant"])] = g
    todo = [(row, variant, prediction) for row in rows
            for variant, prediction in answers_of(row) if (row["qid"], variant) not in done]
    lock = threading.Lock()
    stopped: list[str] = []

    def one(task: tuple[dict[str, Any], str, str]) -> None:
        row, variant, prediction = task
        if stopped:
            return
        ASKED_FOR.set(row["qid"])
        try:
            result = grade_answer(chat, budget, row["beam_ability"], row["rubric"], prediction)
        except CapReached as exc:
            stopped.append(str(exc))
            return
        with lock:
            done[(row["qid"], variant)] = {
                "qid": row["qid"], "conversation": row["conversation"],
                "ability": row["beam_ability"], "variant": variant, **result}

    started = time.time()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    asked = getattr(chat, "asked", None)
    if asked:
        # a file judge without every reply: nothing is graded until they are written
        return {"judge_model": judge_model, "complete": False, "asked": dict(asked),
                "stopped": f"{sum(map(len, asked.values()))} judge calls need replies"}
    graded = sorted(done.values(), key=lambda g: (g["qid"], g["variant"]))
    failed = sorted(f"{row['qid']} {variant}" for row in rows
                    for variant, _ in answers_of(row, failed=True))
    result = {"judge_model": judge_model, "answer_model": config.get("answer_model"),
              "failed_answers": failed,
              "compare_answer_model": config.get("compare_answer_model"),
              "complete": not stopped and len(graded) == sum(1 for r in rows
                                                              for _ in answers_of(r)),
              "stopped": stopped[0] if stopped else None,
              "judge_calls": budget.calls, "judge_tokens": budget.tokens,
              "judge_dollars": round(budget.dollars, 6), "seconds": round(time.time() - started),
              "tables": tables(graded), "graded": graded}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


#: --context of ``answer``: which of the kept lines are answered from. "full":
#: what the answer was shown, turn search's turns included; "bare": the same
#: memories and evidence without turn search's turns (the lines shown when it
#: added none); "evidence:N": the same memories with the evidence turns chosen
#: within N tokens (the harness's --keep-evidence-tokens N), no turn search;
#: "turns:N": the bare lines, then the turns of the excerpts turn search judged
#: at least ``TURN_KEEP``, the best first, while they fit N tokens (rebuilt from
#: the judged excerpts the harness keeps; the memories stay those of "bare").
CONTEXTS = ("full", "bare", "evidence:N", "turns:N")

#: The judged score at which turn search shows an excerpt (``retrieval.turn_search_keep``).
TURN_KEEP = 0.5


def context_lines_of(row: dict[str, Any], variant: str = "full") -> list[str] | None:
    """The lines the row's headline answer was shown (``--keep-context``),
    or another ``variant`` of them (``CONTEXTS``). The harness files that
    answer under the k it asked for, and with ``--all-members`` the answer
    read more memories than that: its ``k`` is the row's ``answer_k``. So the
    answer is found by that count first, then under the key."""
    answers = row.get("answers") or {}
    answer = next((a for a in answers.values()
                   if isinstance(a, dict) and a.get("k") == row.get("answer_k")), None)
    if answer is None:
        answer = answers.get(str(row.get("answer_k")))
    answer = answer or {}
    if variant == "full":
        lines = answer.get("context_lines")
    elif variant == "bare":
        lines = answer.get("context_lines_bare") or answer.get("context_lines")
    elif variant.startswith("turns:"):
        # the bare lines, then turn search's turns as a budget of N tokens shows them
        from memry.store import fill_excerpts

        excerpts = answer.get("turn_search_excerpts") or []
        bare = answer.get("context_lines_bare") or answer.get("context_lines")
        if bare is None:
            return None
        chosen = fill_excerpts(excerpts, int(variant.split(":", 1)[1]), TURN_KEEP)
        shown = sorted({tuple(turn) for i in chosen for turn in excerpts[i]["turns"]})
        lines = list(bare) + [line for _, line in shown]
    elif variant.startswith("evidence:"):
        lines = (answer.get("context_lines_evidence") or {}).get(variant.split(":", 1)[1])
    else:
        raise ValueError(f"context {variant!r}: full, bare or evidence:N")
    return list(lines) if isinstance(lines, list) else None


def answer(paths: list[str], out: pathlib.Path, *, prompt: str, model: str,
           provider: str = "openai", max_dollars: float | None = None, workers: int = 8,
           chat: Chat | None = None, context: str = "full") -> dict[str, Any]:
    """Answer every question of the results files again, from the lines its
    first answer was shown, with ``prompt`` and ``model``; written as a
    results file ``grade`` reads. Answers already in ``out`` without an
    error are kept, so an interrupted pass resumes. ``context`` picks the
    variant of the kept lines (``CONTEXTS``)."""
    rows, config = read_rows(paths)
    build = PROMPTS[prompt]
    chat = chat or openai_chat(model, provider)
    budget = Budget(model, max_dollars)
    done: dict[str, dict[str, Any]] = {}
    if out.exists():
        for row in json.loads(out.read_text(encoding="utf-8")).get("rows") or []:
            if not row.get("answer_error"):
                done[row["qid"]] = row
    keep = ("conversation", "qid", "category", "category_name", "beam_ability", "beam_index",
            "difficulty", "question", "answer", "rubric", "evidence", "reference_date",
            "recall@5", "recall@10", "recall@20", "answer_k", "context_tokens")
    lock = threading.Lock()
    stopped: list[str] = []

    def one(row: dict[str, Any]) -> None:
        if stopped or row["qid"] in done:
            return
        record = {key: row.get(key) for key in keep}
        lines = context_lines_of(row, context)
        started = time.perf_counter()
        if lines is None:
            record.update(prediction="", answer_error="no context_lines in the row "
                                                      "(run the harness with --keep-context)")
        else:
            try:
                text, usage = chat(build(row["question"], lines, row.get("reference_date")))
                budget.charge(usage)
                record.update(prediction=(text or "").strip(), usage=usage)
            except CapReached as exc:
                stopped.append(str(exc))
                return
            except Exception as exc:  # one failed call must not end the pass
                record.update(prediction="", answer_error=f"{type(exc).__name__}: {exc}"[:300])
        record["answer_seconds"] = round(time.perf_counter() - started, 3)
        with lock:
            done[row["qid"]] = record

    started = time.time()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, rows))
    answered = sorted(done.values(), key=lambda r: r["qid"])
    result = {"config": {"answer_model": model, "answer_prompt": prompt, "provider": provider,
                         "context": context, "from": [str(p) for p in paths],
                         "loaded_with": {k: config.get(k) for k in ("ingest", "decider", "k",
                                                                    "embedder", "variant")}},
              "complete": not stopped and len(answered) == len(rows),
              "stopped": stopped[0] if stopped else None,
              "failed": sum(bool(r.get("answer_error")) for r in answered),
              "calls": budget.calls, "tokens": budget.tokens,
              "dollars": round(budget.dollars, 6), "seconds": round(time.time() - started),
              "rows": answered}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def markdown(result: dict[str, Any]) -> str:
    lines = [f"judge {result['judge_model']}, ${result['judge_dollars']} this pass, "
             f"{len(result.get('failed_answers') or [])} failed answers not graded"]
    for variant, t in result["tables"].items():
        lines += ["", f"### {variant} ({t['n']} answers, {t['chats']} chats)", "",
                  "| ability | BEAM's score | by rubric |", "|---|---|---|"]
        for ability in ABILITIES:
            a = t["official"]["abilities"].get(ability)
            b = t["rubric_average"]["abilities"].get(ability)
            if a is not None:
                lines.append(f"| {ability} | {a:.3f} | {b:.3f} |")
        lines += [f"| overall | {t['official']['overall']:.3f} | "
                  f"{t['rubric_average']['overall']:.3f} |", "",
                  f"strict {t['strict']:.3f}, pass {t['pass']:.3f}, "
                  f"judge errors {t['judge_errors']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.beam_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download one size of BEAM")
    f.add_argument("--size", choices=SIZES, default="100K")
    f.add_argument("--out", default=None, help="default $MEMRY_BENCH_DATA/beam_<size>.json")
    a = sub.add_parser("answer", help="answer again from the lines a run kept (--keep-context)")
    a.add_argument("results", nargs="+", help="results files or a --results-dir")
    a.add_argument("--out", required=True)
    a.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--provider", choices=("openai", "openrouter"), default="openai")
    a.add_argument("--max-dollars", type=float, default=None,
                   help="stop once the answers have cost more than this many dollars")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--context", default="full", metavar="full|bare|evidence:N",
                   help="which kept lines to answer from: what was shown (full), the same "
                        "without turn search's turns (bare), or the evidence within N tokens "
                        "(evidence:N, kept with the harness's --keep-evidence-tokens N)")
    g = sub.add_parser("grade", help="grade a results file with BEAM's rubric judge")
    g.add_argument("results", nargs="+", help="results files or a --results-dir")
    g.add_argument("--out", required=True)
    g.add_argument("--judge-model", default=JUDGE_MODEL)
    g.add_argument("--max-dollars", type=float, default=None,
                   help="stop once the judge calls have cost more than this many dollars")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--judge-files", default=None, metavar="DIR",
                   help="judge from replies written to DIR/replies/*.json (key -> reply) "
                        "instead of calling a model; the calls still needing replies are "
                        "written to DIR/asked.json, grouped by question, and nothing is "
                        "graded until every call has its reply")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        root = os.environ.get("MEMRY_BENCH_DATA", "").strip()
        if not args.out and not root:
            parser.error("--out or MEMRY_BENCH_DATA is needed")
        out = pathlib.Path(args.out or pathlib.Path(root) / f"beam_{args.size}.json")
        print(json.dumps(fetch(args.size, out)))
        return 0
    if args.command == "answer":
        result = answer(args.results, pathlib.Path(args.out), prompt=args.prompt,
                        model=args.model, provider=args.provider,
                        max_dollars=args.max_dollars, workers=args.workers,
                        context=args.context)
        print(json.dumps({k: result[k] for k in ("complete", "stopped", "failed", "calls",
                                                 "tokens", "dollars", "seconds")}))
        return 3 if result["stopped"] else 0
    chat: Chat | None = None
    if args.judge_files:
        folder = pathlib.Path(args.judge_files)
        chat = FileJudge(read_replies(folder))
    result = grade(args.results, pathlib.Path(args.out), judge_model=args.judge_model,
                   max_dollars=args.max_dollars, workers=args.workers, chat=chat)
    if result.get("asked") is not None:
        # added to the calls other results files still need, those answered since left out
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "asked.json"
        asked = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for qid, calls in result["asked"].items():
            asked.setdefault(qid, {}).update(calls)
        replied = read_replies(folder)
        asked = {qid: {k: m for k, m in calls.items() if k not in replied}
                 for qid, calls in asked.items()}
        path.write_text(json.dumps({q: c for q, c in asked.items() if c}, ensure_ascii=False),
                        encoding="utf-8")
        print(f"{result['stopped']}: written to {folder / 'asked.json'}", file=sys.stderr)
        return 4
    print(markdown(result))
    if result["stopped"]:
        print(f"stopped: {result['stopped']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
