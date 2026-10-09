"""The LongMemEval path of evals/external_benchmarks.py on a tiny scripted
dataset (tests/fixtures/longmemeval_tiny.json) with scripted models: how a
haystack goes into its store, how its one question is asked, answered and
judged with the official prompts, the stratified sample, and resuming. No
call leaves the process."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import api_usage  # noqa: E402
from evals import external_benchmarks as xb  # noqa: E402
from evals import longmemeval_judge as lj  # noqa: E402
from evals import mem0_judge  # noqa: E402
from memry.config import Config  # noqa: E402
from memry.providers.embeddings import HashEmbedder  # noqa: E402
from memry.providers.llm import LLM  # noqa: E402
from memry.store import MemoryStore  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "longmemeval_tiny.json"


@pytest.fixture
def no_models(monkeypatch, tmp_path):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY",
                 "MEMRY_LLM_PROVIDER", "MEMRY_LLM_API_KEY", "MEMRY_EMBEDDING_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMRY_CONFIG", str(tmp_path / "no-config.json"))


class RuleLLM(LLM):
    """Extraction keeps each numbered line of the transcript as a fact resting
    on that line, and records the day and context it was given; every other
    question gets the answer that changes nothing."""

    name = "rule"
    available = True

    def __init__(self) -> None:
        self.saves: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        if system.startswith("You are the long-term memory extraction system"):
            today = re.search(r"Today's date is (\S+?)\.", system).group(1)
            context = user.split("Shared context for these related inputs:\n", 1)[-1]
            self.saves.append((today, context.split("\n", 1)[0]))
            transcript = user.split("Conversation:\n", 1)[1].split("\n\n", 1)[0]
            lines = [re.match(r"\[(\d+)\] (.*)", line) for line in transcript.splitlines()]
            return json.dumps({"facts": [
                {"content": line.group(2), "type": "episodic", "importance": 0.5,
                 "categories": [], "entities": [], "relations": [], "when": None,
                 "sources": [int(line.group(1))]}
                for line in lines if line]})
        if system.startswith("You audit"):
            return json.dumps({"missing": []})
        if "decide one action" in system:
            return json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new"})
        return "{}"


class ScriptedChat(LLM):
    """A chat model that replies from a script and keeps what it was sent,
    with the stage each call was counted under."""

    name = "openai"
    available = True

    def __init__(self, reply, model: str = "scripted-chat") -> None:
        self.reply = reply
        self.model = model
        self.sent: list[tuple[list, str | None]] = []

    def chat(self, messages, *, json_object=False):
        self.sent.append((messages, api_usage.current_stage()))
        return self.reply(messages) if callable(self.reply) else self.reply

    def complete(self, system, user, *, json_schema=None):
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}])


def tiny() -> dict[str, xb.Conversation]:
    return {c.conv_id: c for c in xb.load_longmemeval(TINY)}


# --------------------------------------------------------------------------
# the data


def test_sessions_are_in_time_order_and_a_repeated_id_is_kept():
    convs = tiny()
    gym = convs["tiny-ku"]
    # the file has the new gym day first; the store must get the old one first
    assert [s.session_id for s in gym.sessions] == ["s-gym-old", "f-bread", "s-gym-new",
                                                    "f-bread~2"]
    assert [s.date.date().isoformat() for s in gym.sessions] == [
        "2023-03-02", "2023-04-01", "2023-06-15", "2023-06-20"]
    assert [t.key for t in gym.sessions[3].turns] == ["f-bread~2#0", "f-bread~2#1"]
    joined = " | ".join(gym.warnings)
    assert "session id f-bread comes again, kept as f-bread~2" in joined
    assert "sessions saved in time order, 2 of 4 not where the file has them" in joined
    assert gym.questions[0].evidence == ["s-gym-new", "s-gym-old"]
    assert convs["tiny-ms"].warnings == []  # in order already: nothing said
    dog = convs["tiny-ssu"].questions[0]
    assert dog.extra == {"question_date": "2023/05/30 (Tue) 23:40"}
    assert dog.reference_date == "2023-05-30T23:40:00+00:00"
    # an abstention question has no evidence, as the official retrieval scores skip it
    cat = convs["tiny-ssu_abs"].questions[0]
    assert cat.abstain and cat.evidence == [] and cat.category_name == "single-session-user"
    assert cat.extra["answer_session_ids"] == ["s-dog"]


def test_a_haystack_is_saved_session_by_session_in_time_order_with_its_dates():
    conv = tiny()["tiny-ku"]
    llm = RuleLLM()
    store = MemoryStore(Config(db_path=":memory:"), llm=llm, embedder=HashEmbedder(128))
    try:
        ingested = xb.ingest(store, conv, mode="extract", unit="session", dataset="longmemeval")
        # one extraction a session, each resolving dates against its own session's day
        assert [day for day, _ in llm.saves] == ["2023-03-02", "2023-04-01", "2023-06-15",
                                                 "2023-06-20"]
        assert llm.saves[0][1] == ("chat between the user and an assistant, "
                                   "2023/03/02 (Thu) 18:10")
        memories = store.get_all(user_id=xb.BENCH_USER, limit=100)
        old = next(m for m in memories if "every Monday" in m.content)
        new = next(m for m in memories if "from Monday to Thursday" in m.content)
        assert old.created_at.startswith("2023-03-02T18:10")
        assert new.created_at.startswith("2023-06-15T07:45")
        assert ingested.units_of(new, "session") == {"s-gym-new"}
        assert old.metadata["bench"]["session_date"] == "2023/03/02 (Thu) 18:10"
    finally:
        store.close()


def test_the_stratified_sample_is_proportional_seeded_and_order_free():
    def conv(i: int, kind: str, abstain: bool = False) -> xb.Conversation:
        qid = f"q{i:03d}" + ("_abs" if abstain else "")
        question = xb.Question(qid=qid, question="?", answer="!", category=kind,
                               category_name=kind, evidence=[], level="session",
                               abstain=abstain)
        return xb.Conversation(qid, "chat", [], [question])

    convs = ([conv(i, "temporal-reasoning") for i in range(60)]
             + [conv(100 + i, "multi-session") for i in range(30)]
             + [conv(200 + i, "knowledge-update") for i in range(7)]
             + [conv(300 + i, "multi-session", abstain=True) for i in range(3)])
    sample = xb.stratified_sample(convs, 20, seed=7)
    kinds = [xb.stratum(c) for c in sample]
    assert len(sample) == 20
    assert {k: kinds.count(k) for k in set(kinds)} == {
        "temporal-reasoning": 12, "multi-session": 6, "knowledge-update": 1,
        "multi-session/abstention": 1}
    assert [c.conv_id for c in sample] == [c.conv_id for c in convs if c in sample]
    again = xb.stratified_sample(list(reversed(convs)), 20, seed=7)
    assert {c.conv_id for c in again} == {c.conv_id for c in sample}
    assert {c.conv_id for c in xb.stratified_sample(convs, 20, seed=8)} != \
        {c.conv_id for c in sample}
    assert xb.stratified_sample(convs, 500, seed=7) == convs
    two = xb.Conversation("two", "chat", [], [convs[0].questions[0]] * 2)
    with pytest.raises(ValueError, match="2 questions"):
        xb.stratified_sample([two], 1)


# --------------------------------------------------------------------------
# the official prompts and judge


def test_the_prompts_are_the_official_ones():
    # src/evaluation/evaluate_qa.py and src/generation/run_generation.py at
    # xiaowu0162/LongMemEval 9e0b455f4ef0e2ab8f2e582289761153549043fc
    official = {
        "JUDGE_FACTUAL": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
        "JUDGE_TEMPORAL": "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
        "JUDGE_KNOWLEDGE_UPDATE":
            "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
        "JUDGE_PREFERENCE": "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
        "JUDGE_ABSTENTION": "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b",
        "ANSWER_PROMPT": "aa840fef33fe11024fd541dfcc9b2be22b1a5ac036a8e64401719fabebc69786",
    }
    for name, digest in official.items():
        assert hashlib.sha256(getattr(lj, name).encode()).hexdigest() == digest, name
    assert lj.JUDGE_MODEL == "gpt-4o-2024-08-06" and lj.JUDGE_MAX_TOKENS == 10


def test_each_question_type_is_judged_with_its_own_prompt():
    assert lj.judge_prompt("multi-session", "Q", "A", "R") == lj.JUDGE_FACTUAL.format("Q", "A", "R")
    assert lj.judge_prompt("temporal-reasoning", "Q", "A", "R").count("off-by-one") == 2
    assert lj.judge_prompt("knowledge-update", "Q", "A", "R") == \
        lj.JUDGE_KNOWLEDGE_UPDATE.format("Q", "A", "R")
    assert "Rubric: A" in lj.judge_prompt("single-session-preference", "Q", "A", "R")
    # an abstention question gets the abstention prompt whatever its type
    assert lj.judge_prompt("temporal-reasoning", "Q", "A", "R", abstention=True) == \
        lj.JUDGE_ABSTENTION.format("Q", "A", "R")
    with pytest.raises(ValueError, match="no judge prompt"):
        lj.judge_prompt("open-domain", "Q", "A", "R")
    assert lj.judge_with(ScriptedChat("Yes."), "multi-session", "Q", "A", "R")
    assert not lj.judge_with(ScriptedChat("no"), "multi-session", "Q", "A", "R")
    with pytest.raises(ValueError, match="needs the question's type"):
        lj.judge("Q", "A", "R")


def _replying(request: httpx.Request, bodies: list) -> httpx.Response:
    bodies.append(json.loads(request.content))
    return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}],
                                     "usage": {"prompt_tokens": 5, "completion_tokens": 1}})


def test_the_judge_and_the_answer_models_are_called_as_the_official_script_calls(monkeypatch):
    bodies: list = []
    transport = httpx.MockTransport(lambda request: _replying(request, bodies))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(lj, "_models", {})
    question = tiny()["tiny-tr"].questions[0]
    lj._model(lj.JUDGE_MODEL)._client = httpx.Client(transport=transport)
    assert lj.judge(question.question, question.answer, "10 days", asked=question)
    (judge_body,) = bodies
    assert judge_body["model"] == "gpt-4o-2024-08-06"
    assert judge_body["temperature"] == 0 and judge_body["max_tokens"] == 10
    assert judge_body["messages"] == [{"role": "user", "content": lj.JUDGE_TEMPORAL.format(
        question.question, question.answer, "10 days")}]
    luna = mem0_judge.OpenAIChat("gpt-6-luna")
    luna._client = httpx.Client(transport=transport)
    luna.chat([{"role": "user", "content": "x"}])
    assert "temperature" not in bodies[-1] and "max_tokens" not in bodies[-1]
    mini = mem0_judge.OpenAIChat("gpt-4o-mini")
    mini._client = httpx.Client(transport=transport)
    mini.chat([{"role": "user", "content": "x"}])
    assert bodies[-1]["temperature"] == 0 and "max_tokens" not in bodies[-1]


# --------------------------------------------------------------------------
# the question path


def test_each_question_is_searched_once_at_20_and_answered_and_judged_officially(monkeypatch):
    """One store per question; one search of 20; gpt-4o-mini and gpt-6-luna
    (scripted) answer from the same memory list with the official reading
    prompt; the official judge (scripted) reads each answer with the prompt
    of the question's type."""
    judge_model = ScriptedChat("yes", model=lj.JUDGE_MODEL)
    monkeypatch.setattr(lj, "_models", {lj.JUDGE_MODEL: judge_model})
    searches: list[tuple[str, int]] = []
    stores: list[MemoryStore] = []

    def store_factory() -> MemoryStore:
        store = MemoryStore(Config(db_path=":memory:"), llm=RuleLLM(), embedder=HashEmbedder(128))
        search = store.search

        def counted(query, **kwargs):
            searches.append((query, kwargs["limit"]))
            return search(query, **kwargs)

        store.search = counted
        stores.append(store)
        return store

    convs = tiny()
    asked = [convs[q] for q in ("tiny-ssu", "tiny-ku", "tiny-ssu_abs")]
    mini = ScriptedChat("Step 1 ... so the answer is a golden retriever.", model="gpt-4o-mini")
    luna = ScriptedChat("Thursdays", model="gpt-6-luna")
    result = xb.run_benchmark(asked, dataset="longmemeval", mode="extract", k=20,
                              store_factory=store_factory, answer_llm=mini,
                              compare_answer_llm=luna, judge=lj.judge,
                              answer_prompt=lj.answer_messages, log=lambda _: None)
    assert result["complete"] and len(stores) == 3
    assert searches == [(c.questions[0].question, 20) for c in asked]
    # both models got the same messages: the official prompt, one user message
    assert [m for m, _ in mini.sent] == [m for m, _ in luna.sent] and len(mini.sent) == 3
    ((message,), _) = mini.sent[0]
    assert message["role"] == "user"
    assert message["content"].startswith("I will give you several history chats between you "
                                         "and a user, as well as the relevant user facts")
    assert message["content"].endswith("\n\nCurrent Date: 2023/05/30 (Tue) 23:40\n"
                                       "Question: What breed is my dog?\nAnswer (step by step):")
    assert "My golden retriever Biscuit pulls on the leash" in message["content"]
    assert "(said 20 May 2023)" in message["content"]
    rows = {r["qid"]: r for r in result["rows"]}
    assert rows["tiny-ssu"]["answer_k"] <= 20 and rows["tiny-ssu"]["recall@20"] == 1.0
    assert rows["tiny-ssu_abs"]["recall@20"] is None  # no evidence for an abstention question
    assert rows["tiny-ssu"]["question_date"] == "2023/05/30 (Tue) 23:40"
    # the judge saw each answer once, with its type's prompt and the file's whole answer
    prompts = [(m[0]["content"], stage) for m, stage in judge_model.sent]
    assert len(prompts) == 6
    assert sum(stage == "judge" for _, stage in prompts) == 3
    assert sum(stage == "judge:compared" for _, stage in prompts) == 3
    texts = [p for p, _ in prompts]
    gym = convs["tiny-ku"].questions[0]
    assert lj.JUDGE_KNOWLEDGE_UPDATE.format(gym.question, gym.answer, "Thursdays") in texts
    cat = convs["tiny-ssu_abs"].questions[0]
    assert lj.JUDGE_ABSTENTION.format(cat.question, cat.answer, "Thursdays") in texts
    assert rows["tiny-ku"]["answers_compared"]["20"]["prediction"] == "Thursdays"
    tables = result["tables"]
    by = {r["category"]: r for r in tables["by_category"]}
    assert by["abstention"]["n"] == 1 and by["single-session-user"]["n"] == 2
    assert tables["overall"]["n"] == 3 and tables["overall"]["judge"] == 1.0
    assert tables["overall"]["judge_task_averaged"] == 1.0
    assert "judge, task-averaged: 1.000" in xb.markdown_table(tables)
    hypotheses = lj.export_hypotheses(result)
    assert hypotheses[1] == {"question_id": "tiny-ku",
                             "hypothesis": "Step 1 ... so the answer is a golden retriever."}
    assert lj.export_hypotheses(result, compared=True)[1]["hypothesis"] == "Thursdays"


def test_the_overall_scores_follow_print_qa_metrics():
    def row(kind, right, abstain=False):
        return {"category": kind, "category_name": kind, "level": "session", "judge": right,
                "abstain": abstain, "evidence": [] if abstain else ["s"]}

    rows = ([row("multi-session", True)] * 3 + [row("multi-session", False, True)]
            + [row("temporal-reasoning", False)])
    tables = xb.aggregate(rows)
    assert tables["overall"]["judge"] == 0.6  # all five questions
    assert tables["overall"]["judge_task_averaged"] == 0.375  # (0.75 + 0) / 2
    assert tables["by_category"][-1]["category"] == "abstention"
    assert tables["by_category"][-1]["judge"] == 0.0


def test_a_run_stops_at_a_cap_and_keeps_what_was_done():
    calls = []

    def judge(question, gold, prediction, *, asked=None):
        calls.append(asked.qid)
        if len(calls) == 2:
            raise api_usage.CapReached("chat: 2 calls, cap 2")
        return True

    judge.reads_question = True
    convs = tiny()
    result = xb.run_benchmark([convs["tiny-ms"], convs["tiny-ku"], convs["tiny-tr"]],
                              dataset="longmemeval", answer_llm=ScriptedChat("x"), judge=judge,
                              embedder=HashEmbedder(128), log=lambda _: None)
    assert not result["complete"] and result["stopped"].startswith("tiny-ku: chat")
    assert [r["qid"] for r in result["rows"]] == ["tiny-ms"]
    assert [s["conversation"] for s in result["stores"]] == ["tiny-ms", "tiny-ku"]


def test_cli_runs_a_sample_and_then_the_rest_reusing_it(monkeypatch, tmp_path, no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    parts = tmp_path / "parts"
    argv = ["--dataset", "longmemeval", "--file", "longmemeval_tiny.json", "--k", "20",
            "--results-dir", str(parts), "--usage-db", str(tmp_path / "usage.sqlite"),
            "--max-calls", "chat=10", "--max-calls", "jev=10"]
    assert xb.main([*argv, "--sample", "4", "--seed", "3", "--out", str(tmp_path / "a.json"),
                    "--export-longmemeval", str(tmp_path / "hyp.jsonl")]) == 0
    first = json.loads((tmp_path / "a.json").read_text())
    assert first["complete"] and len(first["rows"]) == 4
    sampled = sorted(s["conversation"] for s in first["stores"])
    assert sampled == sorted(c.conv_id for c in xb.stratified_sample(
        xb.load_longmemeval(TINY), 4, seed=3))
    created = {c: json.loads((parts / f"{c}.json").read_text())["created"] for c in sampled}
    assert len((tmp_path / "hyp.jsonl").read_text().splitlines()) == 4
    # the whole file: the sampled questions' results are reused, the others run
    assert xb.main([*argv, "--out", str(tmp_path / "b.json")]) == 0
    whole = json.loads((tmp_path / "b.json").read_text())
    assert whole["complete"] and len(whole["rows"]) == 8
    assert all(json.loads((parts / f"{c}.json").read_text())["created"] == created[c]
               for c in sampled)
    assert sum(r["abstain"] for r in whole["rows"]) == 2
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json", "--sample", "1"]) == 2


def test_the_selected_questions_are_written_as_the_file_has_them(monkeypatch, tmp_path,
                                                                   no_models):
    monkeypatch.setenv(xb.DATA_ENV, str(FIXTURES))
    target = tmp_path / "data" / "tiny_sample4_seed3.json"
    assert xb.main(["--dataset", "longmemeval", "--file", "longmemeval_tiny.json",
                    "--sample", "4", "--seed", "3", "--write-selected", str(target)]) == 0
    written = json.loads(target.read_text())
    source = json.loads(TINY.read_text())
    chosen = {c.conv_id for c in xb.stratified_sample(xb.load_longmemeval(TINY), 4, seed=3)}
    assert written == [item for item in source if item["question_id"] in chosen]
    # a run from the small file asks the same questions in the same order
    small = xb.load_longmemeval(target)
    assert [c.conv_id for c in small] == [c.conv_id for c in xb.stratified_sample(
        xb.load_longmemeval(TINY), 4, seed=3)]
    assert xb.main(["--dataset", "locomo", "--file", "locomo_mini.json",
                    "--write-selected", str(tmp_path / "x.json")]) == 2


# --------------------------------------------------------------------------
# the owner and the entity questions


FAMILY = [{
    "question_id": "fam-1", "question_type": "single-session-user",
    "question": "Where does my sister work?", "answer": "Kestrel Labs",
    "question_date": "2023/06/01 (Thu) 10:00",
    "haystack_session_ids": ["s-job", "s-hike"],
    "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/05/07 (Sun) 18:00"],
    "haystack_sessions": [
        [{"role": "user", "content": "My sister Mira started at Kestrel Labs.",
          "has_answer": True},
         {"role": "assistant", "content": "Congratulations to her."}],
        [{"role": "user", "content": "Mira and I went hiking today."},
         {"role": "assistant", "content": "That sounds lovely."}]],
    "answer_session_ids": ["s-job"]}]


#: FAMILY with a brother, Tomas, in two more sessions and Mira in a third, so
#: the user has two related people and Mira has the more memories.
FAMILY_TWO = [{**FAMILY[0], "question_id": "fam-2",
               "haystack_session_ids": FAMILY[0]["haystack_session_ids"]
               + ["s-move", "s-call", "s-photos"],
               "haystack_dates": FAMILY[0]["haystack_dates"]
               + ["2023/05/10 (Wed) 09:00", "2023/05/12 (Fri) 09:00",
                  "2023/05/14 (Sun) 09:00"],
               "haystack_sessions": FAMILY[0]["haystack_sessions"] + [
                   [{"role": "user", "content": "My brother Tomas moved to Lisbon."},
                    {"role": "assistant", "content": "A big change."}],
                   [{"role": "user", "content": "Tomas called me about his flat."},
                    {"role": "assistant", "content": "Good to hear."}],
                   [{"role": "user", "content": "Mira sent me photos of the lake."},
                    {"role": "assistant", "content": "How nice."}]]}]

#: The relation a line's family word gives, and the person it is about.
KIN = {"sister": ("has_sister", "Mira"), "brother": ("has_brother", "Tomas")}


class FamilyLLM(RuleLLM):
    """``RuleLLM`` whose facts name the user as "the user" (the owner's name
    in a chat between the user and an assistant), Mira or Tomas where a line
    names them, and the user's sister or brother relation where a line says
    "sister" or "brother". Each fact has one question key. The description
    writer writes one sentence for each entity it is asked about. The entity
    question writer gets two questions by role for each entity. Both writers'
    calls are kept with their stage. With ``cap`` the entity question
    writer's call is refused as at a --max-calls cap."""

    def __init__(self, cap: bool = False) -> None:
        super().__init__()
        self.cap = cap
        self.entity_question_calls: list[str] = []
        self.description_calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        from memry.intelligence.entities import DESCRIPTION_SCHEMA
        from memry.intelligence.entity_questions import ENTITY_QUESTIONS_SCHEMA

        if json_schema is DESCRIPTION_SCHEMA:
            name = user.split("\n", 1)[0].removeprefix("Entity: ")
            self.description_calls.append((api_usage.current_stage(), name))
            return json.dumps({"description": f"{name} is family of the user."})
        if json_schema is ENTITY_QUESTIONS_SCHEMA:
            self.entity_question_calls.append(api_usage.current_stage())
            if self.cap:
                raise api_usage.CapReached("chat: 2 calls, cap 2")
            entities = [line for line in user.splitlines() if line.startswith("Description:")]
            return json.dumps({"items": [
                {"n": n + 1, "questions": ["Who is my sister?", "Where does my sister work?"]}
                for n in range(len(entities))]})
        out = super().complete(system, user, json_schema=json_schema)
        if not system.startswith("You are the long-term memory extraction system"):
            return out
        facts = json.loads(out)["facts"]
        for fact in facts:
            fact["entities"] = [{"name": "the user", "type": "person"}]
            for person in ("Mira", "Tomas"):
                if person in fact["content"]:
                    fact["entities"].append({"name": person, "type": "person"})
            for word, (predicate, person) in KIN.items():
                if word in fact["content"]:
                    fact["relations"] = [{"subject": "the user", "predicate": predicate,
                                          "object": person}]
            fact["questions"] = [f"What did I say about {fact['content'][:20]}?"]
        return json.dumps({"facts": facts})


def _family_run(monkeypatch, tmp_path, *, flag: bool, look: bool = True, cap: bool = False,
                data: list | None = None, logs: list | None = None):
    """The invented one-question dataset (``data``, FAMILY by default) run
    with ``FamilyLLM``, entity questions on or off. With ``look`` someone
    opens Mira's page once the haystack is loaded, so she has a description
    before the runner's description step; a store loaded just now has none.
    ``logs`` keeps the run's log lines."""
    path = tmp_path / "family.json"
    path.write_text(json.dumps(data or FAMILY))
    llms: list[FamilyLLM] = []

    def store_factory() -> MemoryStore:
        cfg = Config(db_path=":memory:")
        cfg.retrieval.entity_questions = flag
        llms.append(FamilyLLM(cap=cap))
        return MemoryStore(cfg, llm=llms[-1], embedder=HashEmbedder(128))

    loaded = xb.ingest

    def ingest_and_look(store, conversation, **kwargs):
        ingested = loaded(store, conversation, **kwargs)
        if look:
            from memry.models import Scope

            for entity in store.backend.list_entities(Scope(user_id=xb.BENCH_USER), limit=50):
                if entity.name == "Mira":
                    store.entity(entity.id)  # writes her description
        return ingested

    monkeypatch.setattr(xb, "ingest", ingest_and_look)
    result = xb.run_benchmark(xb.load_longmemeval(path), dataset="longmemeval",
                              mode="extract", k=20, store_factory=store_factory,
                              log=(logs.append if logs is not None else lambda _: None))
    return result, llms[0]


def test_the_owner_is_the_user_and_entity_questions_are_written_before_the_question(
        monkeypatch, tmp_path):
    result, llm = _family_run(monkeypatch, tmp_path, flag=True)
    assert result["complete"]
    store = result["stores"][0]
    assert store["owner"] == "the user"
    assert store["entity_questions"] is True
    writer = store["entity_question_writer"]
    assert (writer["entities"], writer["calls"], writer["written"]) == (1, 1, 1)
    # one call, counted under its own stage
    assert llm.entity_question_calls == [xb.ENTITY_QUESTIONS_STAGE]
    # Mira was described when her page was opened, so the runner describes no one
    assert llm.description_calls == [("ingest", "Mira")]
    assert store["entity_descriptions"]["due"] == 0 and store["entities_described"] == 0
    (row,) = result["rows"]
    assert row["owner"] == "the user"
    assert row["entities_with_questions"] == 1
    assert row["memories_with_questions"] == store["memories_with_questions"] > 0
    # the question by role starts from Mira, not from the owner
    assert row["start"] == ["Mira"]
    assert row["start_by_role"] and row["start_other_than_owner"]
    assert not row["start_first_person"]


def test_with_entity_questions_off_the_step_makes_no_call_and_the_owner_is_the_start(
        monkeypatch, tmp_path):
    result, llm = _family_run(monkeypatch, tmp_path, flag=False, look=False)
    store = result["stores"][0]
    assert store["entity_question_writer"] == {"skipped": "retrieval.entity_questions is off"}
    assert store["entity_descriptions"] == {"skipped": "retrieval.entity_questions is off"}
    assert llm.entity_question_calls == []
    assert [stage for stage, _ in llm.description_calls
            if stage == xb.ENTITY_DESCRIPTIONS_STAGE] == []
    (row,) = result["rows"]
    assert row["entities_with_questions"] == 0
    assert row["entities_described"] == 0
    assert row["start"] == ["the user"] and row["start_first_person"]
    assert not (row["start_by_role"] or row["start_other_than_owner"])


def test_a_related_person_without_a_description_is_described_and_then_asked_about(
        monkeypatch, tmp_path):
    result, llm = _family_run(monkeypatch, tmp_path, flag=True, look=False)
    store = result["stores"][0]
    assert store["entity_descriptions"] == {"due": 1, "calls": 1, "described": 1,
                                            "skipped_by_cap": 0}
    writer = store["entity_question_writer"]
    assert (writer["entities"], writer["calls"], writer["written"]) == (1, 1, 1)
    # one call in each stage, the description first
    assert llm.description_calls == [(xb.ENTITY_DESCRIPTIONS_STAGE, "Mira")]
    assert llm.entity_question_calls == [xb.ENTITY_QUESTIONS_STAGE]
    (row,) = result["rows"]
    assert row["entities_described"] == 1
    assert row["entities_with_questions"] == 1
    assert row["start"] == ["Mira"]


def test_the_description_step_stops_at_its_cap_and_logs_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(xb, "DESCRIBE_CAP", 1)
    logs: list[str] = []
    result, llm = _family_run(monkeypatch, tmp_path, flag=True, look=False,
                              data=FAMILY_TWO, logs=logs)
    store = result["stores"][0]
    # Mira has the more memories, so she goes first; Tomas waits
    assert store["entity_descriptions"] == {"due": 2, "calls": 1, "described": 1,
                                            "skipped_by_cap": 1}
    assert llm.description_calls == [(xb.ENTITY_DESCRIPTIONS_STAGE, "Mira")]
    assert any("1 of 2 hubs skipped by the cap of 1" in line for line in logs)
    writer = store["entity_question_writer"]
    assert (writer["entities"], writer["calls"]) == (1, 1)
    assert result["rows"][0]["entities_described"] == 1


def test_the_entity_question_call_stops_at_a_cap(monkeypatch, tmp_path):
    result, llm = _family_run(monkeypatch, tmp_path, flag=True, cap=True)
    assert llm.entity_question_calls == [xb.ENTITY_QUESTIONS_STAGE]
    assert not result["complete"] and result["stopped"].startswith("fam-1: chat")
    assert result["rows"] == []
