"""LongMemEval's official protocol: its reading prompt and its judge.

Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
Memory" (ICLR 2025), github.com/xiaowu0162/LongMemEval. The prompts below are
copied unchanged from commit 9e0b455f4ef0e2ab8f2e582289761153549043fc (MIT
license):

  JUDGE_*        ``get_anscheck_prompt`` of src/evaluation/evaluate_qa.py: one
                 prompt for single-session-user, single-session-assistant and
                 multi-session, one each for temporal-reasoning,
                 knowledge-update and single-session-preference, and one for
                 every abstention question (a question id ending in "_abs"),
                 whatever its type
  ANSWER_PROMPT  src/generation/run_generation.py, the template for history
                 chats shown with the user facts extracted from them, with
                 chain-of-thought reading ("con", the README's
                 recommendation): Memry shows its facts and then the turns
                 they rest on

Judging (``judge``) follows evaluate_qa.py: the prompt as the only user
message to ``JUDGE_MODEL`` (gpt-4o-2024-08-06, the model of the README's
command and of print_qa_metrics.py) at temperature 0 with at most 10 tokens,
and the answer is right when the reply contains "yes". ``judge_mini`` is the same with
gpt-4o-mini-2024-07-18, the other OpenAI model of the script's model list.

Answering (``answer_messages``) follows run_generation.py: the filled template
as the only user message. The history is Memry's own list, one line each, as
its context builder renders it for the question (``external_benchmarks.
answer_with``); the current date is the question's date as the file writes it
("2023/05/30 (Tue) 23:40"). In run_generation.py the model is called at
temperature 0 with at most 800 tokens for this prompt. Here gpt-4o-mini is
called at temperature 0 and gpt-6-luna at its own temperature (``mem0_judge.
OpenAIChat``), with no cap.

    ... --dataset longmemeval --answer-model gpt-4o-mini \\
        --answer-prompt evals.longmemeval_judge:answer_messages \\
        --judge evals.longmemeval_judge:judge

``export_hypotheses`` writes the answers as the jsonl that evaluate_qa.py
reads ({"question_id", "hypothesis"} a line).
"""

from __future__ import annotations

import pathlib
import sys
import threading
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from evals.mem0_judge import OpenAIChat  # noqa: E402

#: The model of the official judge (evaluate_qa.py's "gpt-4o").
JUDGE_MODEL = "gpt-4o-2024-08-06"
#: The official script's other OpenAI judge ("gpt-4o-mini").
JUDGE_MODEL_MINI = "gpt-4o-mini-2024-07-18"
#: evaluate_qa.py's reply cap.
JUDGE_MAX_TOKENS = 10

JUDGE_FACTUAL = (
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'
)
JUDGE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
JUDGE_KNOWLEDGE_UPDATE = (
    'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.'
)
JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
JUDGE_ABSTENTION = (
    'I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.'
)
ANSWER_PROMPT = (
    'I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
)

#: The question types and the judge prompt each is asked with.
JUDGE_PROMPTS = {
    "single-session-user": JUDGE_FACTUAL,
    "single-session-assistant": JUDGE_FACTUAL,
    "multi-session": JUDGE_FACTUAL,
    "temporal-reasoning": JUDGE_TEMPORAL,
    "knowledge-update": JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference": JUDGE_PREFERENCE,
}


def judge_prompt(question_type: str, question: str, answer: str, response: str,
                 abstention: bool = False) -> str:
    """``get_anscheck_prompt``: the judge's prompt for one answer."""
    if abstention:
        return JUDGE_ABSTENTION.format(question, answer, response)
    if question_type not in JUDGE_PROMPTS:
        raise ValueError(f"no judge prompt for question type {question_type!r}")
    return JUDGE_PROMPTS[question_type].format(question, answer, response)


def judge_with(model: Any, question_type: str, question: str, gold: str, prediction: str,
               abstention: bool = False) -> bool:
    """The official judgement of one answer by ``model`` (an ``OpenAIChat``):
    right when the reply contains "yes", in any case."""
    prompt = judge_prompt(question_type, question, gold, prediction, abstention)
    reply = model.chat([{"role": "user", "content": prompt}])
    return "yes" in str(reply or "").strip().lower()


_models: dict[str, OpenAIChat] = {}
_lock = threading.Lock()


def _model(name: str) -> OpenAIChat:
    with _lock:
        if name not in _models:
            model = OpenAIChat(name, temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
            if not model.available:
                raise RuntimeError("the judge needs OPENAI_API_KEY")
            _models[name] = model
    return _models[name]


def _asked(asked: Any) -> tuple[str, bool]:
    if asked is None:
        raise ValueError("the LongMemEval judge needs the question's type (asked=Question)")
    return str(asked.category_name), bool(asked.abstain)


def judge(question: str, gold: str, prediction: str, *, asked: Any = None) -> bool:
    """The official judge (``JUDGE_MODEL``) on one answer to ``asked``, the
    runner's ``Question``: its type and whether it is an abstention question
    choose the prompt."""
    question_type, abstention = _asked(asked)
    return judge_with(_model(JUDGE_MODEL), question_type, question, gold, prediction, abstention)


def judge_mini(question: str, gold: str, prediction: str, *, asked: Any = None) -> bool:
    """``judge`` with gpt-4o-mini (``JUDGE_MODEL_MINI``)."""
    question_type, abstention = _asked(asked)
    return judge_with(_model(JUDGE_MODEL_MINI), question_type, question, gold, prediction,
                      abstention)


for _function in (judge, judge_mini):
    #: The judge gets the question (``external_benchmarks.call_judge``) and
    #: the file's whole answer (``judged_gold``).
    _function.reads_question = True  # type: ignore[attr-defined]
    _function.reads_full_answer = True  # type: ignore[attr-defined]


def history_text(lines: list[Any]) -> str:
    """The history as it is in the prompt: Memry's lines, one a line."""
    return "\n".join(str(line) for line in lines)


def answer_messages(question: str, memories: list[Any], *, asked: Any = None
                    ) -> list[dict[str, str]]:
    """The answering call's messages: ``ANSWER_PROMPT`` with Memry's lines,
    the question's date as the file writes it and the question, as the only
    user message."""
    date = ""
    if asked is not None:
        date = str((asked.extra or {}).get("question_date") or asked.reference_date or "")
    return [{"role": "user",
             "content": ANSWER_PROMPT.format(history_text(memories), date, question)}]


answer_messages.reads_question = True  # type: ignore[attr-defined]
#: What the context tokens count: the history block (``external_benchmarks.context_text``).
answer_messages.context_of = history_text  # type: ignore[attr-defined]


def export_hypotheses(result: dict[str, Any], *, compared: bool = False) -> list[dict[str, str]]:
    """The headline answers of a results file (``compared``: the compared
    answers at the headline k) as evaluate_qa.py reads them, one
    {"question_id", "hypothesis"} per question."""
    k = str(result["config"]["k"])
    out = []
    for row in result["rows"]:
        if compared:
            record = (row.get("answers_compared") or {}).get(k)
            if record is None:
                continue
            hypothesis = record.get("prediction", "")
        else:
            hypothesis = row.get("prediction", "")
        out.append({"question_id": row["qid"], "hypothesis": hypothesis})
    return out
