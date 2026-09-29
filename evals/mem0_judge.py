"""Mem0's LoCoMo protocol: its answer prompt and its LLM judge.

Chhikara et al. 2025, "Mem0: Building Production-Ready AI Agents with Scalable
Long-Term Memory", published its evaluation code in github.com/mem0ai/mem0,
``evaluation/``. The two prompts below are copied from it unchanged, at commit
b3ede5b7c0ac0e847b03786a603c107ac943b3ee, the last one that had the code in
the repository (the next moved it to mem0ai/memory-benchmarks). Mem0 is licensed
under the Apache License 2.0; the prompts are reproduced here under its terms:

  ANSWER_PROMPT    ``ANSWER_PROMPT_ZEP`` of evaluation/prompts.py, the variant
                   Mem0 wrote for a memory system that returns one list of
                   memories. Its main prompt has a list per speaker because
                   Mem0 keeps a store per speaker; a Memry store holds both.
  ACCURACY_PROMPT  evaluation/metrics/llm_judge.py

Answering (``answer_messages``) follows evaluation/src/memzero/search.py: the
prompt is the system message, and the memories are a JSON list of
"<timestamp>: <memory>" strings, the timestamp written as LoCoMo writes a
session's ("1:56 pm on 8 May, 2023"). The model is called at temperature 0.

Judging (``judge``) follows llm_judge.py: the prompt as the user message to
gpt-4o-mini, JSON response format, temperature 0, and the answer is right
when the label is CORRECT.

    ... --answer-model gpt-4o-mini --answer-prompt evals.mem0_judge:answer_messages \\
        --judge evals.mem0_judge:judge
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time
from datetime import datetime
from typing import Any

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from memry.providers.llm import LLM  # noqa: E402

#: The model Mem0's judge calls.
JUDGE_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com"
#: Replies worth another try: a timeout, a conflict, a rate limit, a server error.
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})

ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """


ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

_FIELD = re.compile(r"\{\{(memories|question)\}\}")


class OpenAIChat(LLM):
    """An OpenAI chat model called as Mem0's evaluation calls one: the
    messages as given, at temperature 0 (``OPENAI_API_KEY``). A timeout, a
    dropped connection, a rate limit or a server error is tried again, up to
    ``attempts`` calls in all."""

    name = "openai"

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None,
                 temperature: float = 0.0, timeout: float = 120.0, attempts: int = 4) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.available = bool(self.api_key)
        self.base_url = (base_url or OPENAI_URL).rstrip("/")
        self.temperature = temperature
        self.attempts = max(attempts, 1)
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def chat(self, messages: list[dict[str, str]], *, json_object: bool = False) -> str:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": self.temperature}
        if json_object:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(self.attempts):
            last = attempt + 1 == self.attempts
            try:
                resp = self._client.post(f"{self.base_url}/v1/chat/completions",
                                         headers={"Authorization": f"Bearer {self.api_key}"},
                                         json=body)
            except httpx.TransportError:
                if last:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in RETRY_STATUSES and not last:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        raise AssertionError("the loop returns or raises")

    def complete(self, system: str, user: str, *, json_schema: dict[str, Any] | None = None) -> str:
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}])


def locomo_time(stamp: Any) -> str:
    """An ISO time as LoCoMo writes a session's: "1:56 pm on 8 May, 2023"."""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except ValueError:
        return str(stamp or "")
    half = "am" if moment.hour < 12 else "pm"
    return (f"{moment.hour % 12 or 12}:{moment.minute:02d} {half} on {moment.day} "
            f"{moment:%B}, {moment.year}")


def answer_messages(question: str, memories: list[Any]) -> list[dict[str, str]]:
    """The answering call's messages: ``ANSWER_PROMPT`` holding the memories
    (each timestamped with its ``created_at``) and the question, as the system
    message."""
    values = {
        "memories": json.dumps([f"{locomo_time(m.created_at)}: {m.content}" for m in memories],
                               indent=4),
        "question": question,
    }
    return [{"role": "system",
             "content": _FIELD.sub(lambda match: values[match.group(1)], ANSWER_PROMPT)}]


def parse_label(raw: str) -> str:
    """The "label" of the judge's JSON reply, in a code fence or not."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    try:
        data = json.loads(text)
    except ValueError:
        found = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(found.group(0)) if found else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("label"), str):
        raise ValueError(f"no label in the judge's reply: {(raw or '')[:200]!r}")
    return data["label"].strip()


def judge_with(model: LLM, question: str, gold: str, prediction: str) -> bool:
    """Mem0's judgement of one answer by ``model`` (an ``OpenAIChat``)."""
    prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold,
                                    generated_answer=prediction)
    raw = model.chat([{"role": "user", "content": prompt}], json_object=True)  # type: ignore[attr-defined]
    return parse_label(raw) == "CORRECT"


_judge_model: OpenAIChat | None = None


def judge(question: str, gold: str, prediction: str) -> bool:
    """Mem0's LLM judge with ``JUDGE_MODEL``: the prediction is right."""
    global _judge_model
    if _judge_model is None:
        _judge_model = OpenAIChat(JUDGE_MODEL)
        if not _judge_model.available:
            raise RuntimeError("the judge needs OPENAI_API_KEY")
    return judge_with(_judge_model, question, gold, prediction)


#: Mem0's judge reads the file's whole answer, an open-domain one's reason
#: after the ";" included (``external_benchmarks.score_answer``).
judge.reads_full_answer = True  # type: ignore[attr-defined]
