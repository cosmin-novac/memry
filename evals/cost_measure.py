"""Measure what Memry costs at ingestion and saves at read time.

Run:  python evals/cost_measure.py [--scale]
Results feed docs/cost-benefit.md. Writes cost_measure*.json/.db next to this file.

Runs a scripted 4-session conversation through the real MemoryStore pipeline
with a fake LLM that answers every prompt type plausibly, so we can count
LLM calls, prompt/output tokens (len//4, same estimator as memry) and
embedding calls per phase. Then compares the read side against
"paste the whole history" and "verbatim/no-LLM" alternatives.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memry.config import Config
from memry.intelligence import clustering, consolidate, entities, extraction, reconcile
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import LLM
from memry.store import MemoryStore

TOK = lambda s: max(1, len(s) // 4)  # memry.intelligence.context.estimate_tokens

# --------------------------------------------------------------------------
# Scenario: 4 sessions, 14 user messages, realistic length (35-80 words),
# with duplicates, one contradiction, recurring entities.
# --------------------------------------------------------------------------
SESSIONS = [
    [  # session 1
        ("Hi, I'm Ada. I work as a data engineer at Northwind, we're a logistics company "
         "based in Munich. Most of my day is Airflow DAGs and a lot of dbt models on BigQuery.",
         [("Ada works as a data engineer at Northwind", "semantic", 0.8, ["work"], ["Ada", "Northwind"]),
          ("Northwind is a logistics company based in Munich", "semantic", 0.6, ["work"], ["Northwind", "Munich"]),
          ("Ada mainly works with Airflow DAGs and dbt models on BigQuery", "semantic", 0.7, ["tools"], ["Ada", "Airflow", "dbt", "BigQuery"])]),
        ("I live in Munich too, in Schwabing, and I bike to the office most days unless it's raining.",
         [("Ada lives in Munich (Schwabing)", "semantic", 0.8, ["personal"], ["Ada", "Munich"]),
          ("Ada bikes to the office most days unless it rains", "semantic", 0.5, ["personal"], ["Ada"])]),
        ("One thing to remember: I strongly prefer uv over pip for Python projects, and I use ruff. "
         "Please don't suggest poetry.",
         [("Ada prefers uv over pip and uses ruff; does not want poetry suggested", "preference", 0.7, ["tools"], ["Ada", "uv", "ruff"])]),
        ("Can you help me write a DAG that loads yesterday's shipments parquet into a staging table? "
         "Nothing to remember here, just a coding question.",
         []),
    ],
    [  # session 2
        ("Quick one: I prefer uv over pip, remember that.",  # duplicate of session 1
         [("Ada prefers uv over pip", "preference", 0.7, ["tools"], ["Ada", "uv"])]),
        ("My manager is Jonas Weber, he leads the data platform team. We do sprint planning on Mondays "
         "and I usually present the pipeline health dashboard there.",
         [("Ada's manager is Jonas Weber, who leads the data platform team at Northwind", "semantic", 0.7, ["work"], ["Ada", "Jonas Weber", "Northwind"]),
          ("Northwind data platform team does sprint planning on Mondays; Ada presents the pipeline health dashboard", "semantic", 0.5, ["work"], ["Ada", "Northwind"])]),
        ("We're migrating from BigQuery to Snowflake in Q4, so a lot of the dbt models need to be ported.",
         [("Northwind is migrating from BigQuery to Snowflake in Q4; dbt models need porting", "semantic", 0.7, ["work"], ["Northwind", "BigQuery", "Snowflake", "dbt"])]),
        ("Also, how do I set up incremental models in dbt for Snowflake with merge strategy?",
         []),
    ],
    [  # session 3 (contradiction: moved)
        ("Big news - I moved to Amsterdam last month. Same job, Northwind opened a hub here and I'm "
         "remote most of the week. Still biking though, that part of Munich I kept.",
         [("Ada lives in Amsterdam (moved from Munich)", "semantic", 0.9, ["personal"], ["Ada", "Amsterdam", "Munich"]),
          ("Ada now works remotely most of the week from Northwind's Amsterdam hub", "semantic", 0.7, ["work"], ["Ada", "Northwind", "Amsterdam"])]),
        ("I need to plan my commute for the two office days, Tuesday and Thursday. The hub is near Zuid.",
         [("Ada goes to the Northwind Amsterdam hub (near Zuid) on Tuesdays and Thursdays", "semantic", 0.6, ["personal"], ["Ada", "Northwind", "Amsterdam"])]),
        ("Please always answer in metric units and keep code comments in English.",
         [("Ada wants metric units and English code comments", "preference", 0.6, ["preferences"], ["Ada"])]),
    ],
    [  # session 4
        ("Jonas approved the Snowflake budget, migration kickoff is on the 3rd of next month.",
         [("Jonas Weber approved the Snowflake migration budget; kickoff on the 3rd of next month", "event", 0.6, ["work"], ["Jonas Weber", "Snowflake"])]),
        ("Reminder that I use uv, not pip.",  # duplicate again
         [("Ada uses uv, not pip", "preference", 0.7, ["tools"], ["Ada", "uv"])]),
        ("Can you draft the migration checklist for the dbt models? Just the outline.",
         []),
    ],
]
# facts that supersede an earlier fact (new content substring -> old content substring)
SUPERSEDES = {
    "Ada lives in Amsterdam": "Ada lives in Munich",
}
QUERIES = [
    "where does ada live",
    "what does ada prefer for python packaging",
    "who is ada's manager and what team",
    "plan my commute to the office",
    "what is the snowflake migration status",
]


class CountingLLM(LLM):
    name = "counting-fake"
    available = True

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.unknown: Counter = Counter()

    def _kind(self, system: str) -> str:
        head = system[:60]
        if head == extraction.EXTRACTION_SYSTEM[:60]:
            return "extraction"
        if head == extraction.COVERAGE_SYSTEM[:60]:
            return "coverage-audit"
        if head == extraction.RELATION_SYSTEM[:60]:
            return "relations"
        if head == reconcile.RECONCILE_SYSTEM[:60]:
            return "reconcile"
        if head == entities.IDENTITY_SYSTEM[:60]:
            return "entity-identity"
        if head == entities.DESCRIPTION_SYSTEM[:60]:
            return "entity-description"
        if head == entities._TYPE_SYSTEM[:60]:
            return "entity-type"
        if head == entities._REFERENT_SYSTEM[:60]:
            return "entity-referent"
        if head == clustering.SYNTHETIC_TAG_SYSTEM[:60] or head == clustering.CANONICALIZE_SYSTEM[:60]:
            return "tag-clustering"
        if head == consolidate.CONSOLIDATE_SYSTEM[:60]:
            return "consolidate"
        return "unknown"

    def complete(self, system: str, user: str, *, json_schema=None) -> str:
        kind = self._kind(system)
        out = self._answer(kind, user)
        self.calls.append({
            "kind": kind,
            "in_tokens": TOK(system) + TOK(user),
            "out_tokens": TOK(out),
            "system_tokens": TOK(system),
            "user_tokens": TOK(user),
        })
        if kind == "unknown":
            self.unknown[system[:80]] += 1
        return out

    # ---- plausible answers ------------------------------------------------
    def _answer(self, kind: str, user: str) -> str:
        if kind == "extraction":
            facts = []
            for session in SESSIONS:
                for text, fs in session:
                    if text[:40] in user:
                        for c, t, imp, cats, ents in fs:
                            facts.append({"content": c, "type": t, "importance": imp,
                                          "categories": cats,
                                          "entities": [{"name": e, "type": _etype(e)} for e in ents],
                                          "relations": []})
            return json.dumps({"facts": facts})
        if kind == "reconcile":
            m = re.search(r"NEW fact:\n(.*)$", user, re.S)
            new = m.group(1).strip() if m else ""
            existing = re.findall(r"^\[(\d+)\] (.*)$", user, re.M)
            for key, old in SUPERSEDES.items():
                if key in new:
                    for idx, content in existing:
                        if old in content:
                            return json.dumps({"action": "DELETE", "target": int(idx), "content": None,
                                               "reason": "contradiction: moved"})
            # near-duplicate preference restatements -> NONE
            for idx, content in existing:
                if _norm(content) == _norm(new) or (
                    "uv" in new and "pip" in new and "uv" in content and "pip" in content):
                    return json.dumps({"action": "NONE", "target": int(idx), "content": None,
                                       "reason": "already known"})
            return json.dumps({"action": "ADD", "target": None, "content": None, "reason": "new information"})
        if kind == "coverage-audit":
            return json.dumps({"missing": []})
        if kind == "relations":
            return json.dumps({"relations": []})
        if kind == "entity-identity":
            ex = re.search(r'EXISTING entity "([^"]+)"', user)
            new = re.search(r'NEW fact mentioning "([^"]+)"', user)
            same = ex and new and ex.group(1).lower() == new.group(1).lower()
            return json.dumps({"verdict": "same" if same else "different",
                               "confidence": 0.95 if same else 0.9, "reason": "name match"})
        if kind == "entity-description":
            return json.dumps({"description": "Entity known from Ada's work and personal context."})
        if kind == "entity-type":
            names = re.findall(r"^- (.*)$", user, re.M)
            return json.dumps({"types": [{"name": n, "type": _etype(n)} for n in names]})
        if kind == "entity-referent":
            return json.dumps({"entities": []})
        return "{}"


def _etype(name: str) -> str:
    return {"Ada": "person", "Jonas Weber": "person", "Northwind": "organization",
            "Munich": "place", "Amsterdam": "place"}.get(name, "other")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.texts = 0
        self.chars = 0

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        self.chars += sum(len(t) for t in texts)
        return super().embed(texts)


def build(llm, embedder, path: Path) -> MemoryStore:
    if path.exists():
        path.unlink()
    cfg = Config(db_path=str(path))
    return MemoryStore(cfg, llm=llm, embedder=embedder)


def phase_summary(calls: list[dict]) -> dict:
    by = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for c in calls:
        b = by[c["kind"]]
        b["calls"] += 1; b["in"] += c["in_tokens"]; b["out"] += c["out_tokens"]
    return dict(by)


def main() -> None:
    out_dir = Path(__file__).parent
    scratch = out_dir / "cost_measure.db"
    report: dict = {}

    # ---------------- ingestion, per-message add(infer=True) -------------
    llm, emb = CountingLLM(), CountingEmbedder()
    store = build(llm, emb, scratch)
    per_episode = []
    all_msgs = []
    for s_idx, session in enumerate(SESSIONS):
        for text, _ in session:
            all_msgs.append(text)
            before_calls = len(llm.calls); before_emb = emb.calls
            t0 = time.perf_counter()
            res = store.add(text, user_id="ada")
            dt = time.perf_counter() - t0
            calls = llm.calls[before_calls:]
            per_episode.append({
                "session": s_idx + 1,
                "episode_tokens": TOK(text),
                "llm_calls": len(calls),
                "in_tokens": sum(c["in_tokens"] for c in calls),
                "out_tokens": sum(c["out_tokens"] for c in calls),
                "embed_calls": emb.calls - before_emb,
                "actions": Counter(a.event for a in res.actions),
                "local_ms": round(dt * 1000, 1),
            })
    report["per_message_add"] = {
        "episodes": len(per_episode),
        "episode_tokens_total": sum(e["episode_tokens"] for e in per_episode),
        "llm_calls_total": sum(e["llm_calls"] for e in per_episode),
        "in_tokens_total": sum(e["in_tokens"] for e in per_episode),
        "out_tokens_total": sum(e["out_tokens"] for e in per_episode),
        "embed_calls_total": emb.calls, "embed_texts_total": emb.texts,
        "by_phase": phase_summary(llm.calls),
        "per_episode": per_episode,
        "unknown_prompts": dict(llm.unknown),
    }
    # system prompt share (fixed overhead per call, cacheable)
    sys_tok = sum(c["system_tokens"] for c in llm.calls)
    report["per_message_add"]["system_prompt_tokens_total"] = sys_tok

    # ---------------- read side ---------------------------------------
    pass
    active = [m for m in store.get_all(user_id="ada", limit=500)]
    report["store_state"] = {
        "active_memories": len(active),
        "active_memory_tokens": sum(TOK(m.content) for m in active),
    }
    history_tokens = sum(TOK(t) for t in all_msgs)
    reads = []
    for q in QUERIES:
        lat = []
        for _ in range(7):
            t0 = time.perf_counter(); ctx = store.reconstruct_context(q, user_id="ada", token_budget=1200)
            lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); hits = store.search(q, user_id="ada", limit=5)
        s_ms = (time.perf_counter() - t0) * 1000
        reads.append({
            "query": q,
            "ctx_tokens": ctx.token_estimate,
            "ctx_memories": len(ctx.memory_ids),
            "ctx_ms_median": round(statistics.median(lat), 2),
            "search_top5_tokens": sum(TOK(h.memory.content) for h in hits),
            "search_ms": round(s_ms, 2),
            "top_hit": hits[0].memory.content if hits else None,
        })
    report["reads"] = {
        "full_history_tokens": history_tokens,
        "queries": reads,
        "ctx_tokens_mean": round(statistics.mean(r["ctx_tokens"] for r in reads)),
    }

    # ---------------- alternative: verbatim / no LLM ---------------------
    class NoLLM(LLM):
        name = "none"; available = False
        def complete(self, *a, **k): raise RuntimeError
    emb2 = CountingEmbedder()
    store2 = build(NoLLM(), emb2, out_dir / "cost_measure_verbatim.db")
    for text in all_msgs:
        store2.add(text, user_id="ada")
    vreads = []
    for q in QUERIES:
        ctx = store2.reconstruct_context(q, user_id="ada", token_budget=1200)
        hits = store2.search(q, user_id="ada", limit=5)
        vreads.append({"query": q, "ctx_tokens": ctx.token_estimate,
                       "top_hit": hits[0].memory.content[:80] if hits else None})
    report["verbatim_alternative"] = {"embed_calls": emb2.calls, "reads": vreads}

    # ---------------- deferred/burst path (what MCP save_memories uses) ---
    llm3, emb3 = CountingLLM(), CountingEmbedder()
    store3 = build(llm3, emb3, out_dir / "cost_measure_deferred.db")
    for session in SESSIONS:
        for text, _ in session:
            store3.add_deferred(text, user_id="ada")
        # worker drains after the quiet period; here: immediately, whole session as one burst
        store3.process_pending_enrichments(limit=64, quiet_seconds=0.0)
    report["deferred_burst_add"] = {
        "llm_calls_total": len(llm3.calls),
        "in_tokens_total": sum(c["in_tokens"] for c in llm3.calls),
        "out_tokens_total": sum(c["out_tokens"] for c in llm3.calls),
        "by_phase": phase_summary(llm3.calls),
        "embed_calls_total": emb3.calls,
        "active_memories": len(store3.get_all(user_id="ada", limit=500)),
        "unknown_prompts": dict(llm3.unknown),
    }

    (out_dir / "cost_measure.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_message_add"}, indent=1, default=str)[:6000])
    pm = report["per_message_add"]
    print("PER-MESSAGE ADD:", {k: v for k, v in pm.items() if k != "per_episode"})
    for e in pm["per_episode"]:
        print("  ", e)




def scaling() -> None:
    """Grow the history with generic project notes and watch read-side tokens/latency."""
    import random
    random.seed(7)
    out_dir = Path(__file__).parent
    llm, emb = CountingLLM(), CountingEmbedder()
    # extend the fake extraction with a generic rule: "Note: X" -> fact "X"
    orig = llm._answer
    def answer(kind, user):
        if kind == "extraction" and "Note:" in user:
            facts = [{"content": m.group(1).strip(), "type": "semantic", "importance": 0.5,
                      "categories": ["project"], "entities": [], "relations": []}
                     for m in re.finditer(r"Note: (.*?)(?:\n|$)", user)]
            return json.dumps({"facts": facts})
        return orig(kind, user)
    llm._answer = answer
    store = build(llm, emb, out_dir / "cost_measure_scale.db")
    history = []
    for session in SESSIONS:
        for text, _ in session:
            store.add(text, user_id="ada"); history.append(text)
    topics = ["invoice pipeline", "warehouse schema", "kafka topic", "airflow scheduler", "dbt test",
              "snowflake warehouse", "cost report", "on-call rota", "shipment api", "data catalog"]
    verbs = ["needs a retry policy", "is owned by the platform team", "runs at 03:00 UTC",
             "was migrated last sprint", "has a flaky integration test", "is documented in confluence",
             "uses parquet on gcs", "must keep 30 days retention", "is blocked on legal review",
             "should move to the new cluster"]
    rows = []
    def snapshot(n):
        hist_tok = sum(TOK(t) for t in history)
        lat = []; ctxs = []
        for q in QUERIES:
            t0 = time.perf_counter(); ctx = store.reconstruct_context(q, user_id="ada", token_budget=1200)
            lat.append((time.perf_counter() - t0) * 1000); ctxs.append(ctx.token_estimate)
        rows.append({"messages": n, "history_tokens": hist_tok,
                     "ctx_tokens_mean": round(statistics.mean(ctxs)),
                     "ctx_ms_median": round(statistics.median(lat), 1),
                     "active_memories": len(store.get_all(user_id="ada", limit=5000))})
    snapshot(len(history))
    for i in range(1, 91):
        t = random.choice(topics); v = random.choice(verbs)
        text = (f"Note: the {t} {v}, ticket DP-{1000+i}. Also, for context, we discussed this in "
                f"the {random.choice(['standup','retro','planning','1:1'])} and agreed to revisit it "
                f"{random.choice(['next week','after the migration','in Q4','when Jonas is back'])}.")
        store.add(text, user_id="ada"); history.append(text)
        if len(history) in (34, 54, 74, 104):
            snapshot(len(history))
    print("SCALING:")
    for r in rows: print("  ", r)
    print("  ingestion so far: calls", len(llm.calls), "in", sum(c["in_tokens"] for c in llm.calls),
          "out", sum(c["out_tokens"] for c in llm.calls), "per msg in",
          round(sum(c["in_tokens"] for c in llm.calls) / len(history)))
    json.dump(rows, open(out_dir / "cost_scaling.json", "w"), indent=1)


if __name__ == "__main__":
    if "--scale" in sys.argv:
        scaling()
    else:
        main()
