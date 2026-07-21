"""Memry command-line interface.

    memry mcp                     run the MCP server (stdio)
    memry mcp --transport http    run the MCP server (streamable HTTP)
    memry serve                   REST API + dashboard + /mcp
    memry add "text" -u ada       add a memory
    memry search "query" -u ada   search memories
    memry list -u ada             list memories
    memry context "task" -u ada   build a context block
    memry history <memory_id>     audit trail for one memory
    memry stats                   store statistics
    memry sweep                   decay sweep (soft-forget stale memories)
    memry reindex                 re-embed all memories
    memry export / import         JSONL export/import of memories
    memry config                  print resolved configuration
    memry eval --dataset <path>   run the retrieval eval harness
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .config import Config


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _store():
    from .store import MemoryStore

    return MemoryStore(Config.load())


def _scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=None, help="user_id scope")
    parser.add_argument("-a", "--agent", default=None, help="agent_id scope")
    parser.add_argument("-r", "--run", default=None, help="run_id scope")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memry", description=__doc__)
    parser.add_argument("--version", action="version", version=f"memry {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("mcp", help="run the MCP server")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)

    p = sub.add_parser("serve", help="run REST API + dashboard + /mcp")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)

    p = sub.add_parser("add", help="add a memory")
    p.add_argument("text")
    _scope_args(p)
    p.add_argument("--no-infer", action="store_true", help="store verbatim (skip extraction)")
    p.add_argument("-c", "--category", action="append", default=None,
                   help="category label for verbatim adds (repeatable)")

    p = sub.add_parser("search", help="search memories")
    p.add_argument("query")
    _scope_args(p)
    p.add_argument("-n", "--limit", type=int, default=10)
    p.add_argument("-c", "--category", action="append", default=None,
                   help="restrict to a category (repeatable)")

    p = sub.add_parser("list", help="list memories")
    _scope_args(p)
    p.add_argument("-n", "--limit", type=int, default=50)
    p.add_argument("--all", action="store_true", help="include invalidated memories")
    p.add_argument("-c", "--category", action="append", default=None,
                   help="restrict to a category (repeatable)")

    p = sub.add_parser("entities", help="inspect and disambiguate entities")
    entity_sub = p.add_subparsers(dest="entities_command")
    ep = entity_sub.add_parser("list", help="list entities")
    _scope_args(ep)
    ep.add_argument("-n", "--limit", type=int, default=50)
    ep = entity_sub.add_parser("show", help="show one entity with its memories")
    ep.add_argument("entity_id")
    ep = entity_sub.add_parser("proposals", help="list merge proposals")
    _scope_args(ep)
    ep.add_argument("--status", default="proposed", choices=["proposed", "confirmed", "rejected"])
    ep = entity_sub.add_parser("confirm", help="confirm a merge proposal (same entity)")
    ep.add_argument("proposal_id")
    ep = entity_sub.add_parser("reject", help="reject a merge proposal (different entities)")
    ep.add_argument("proposal_id")
    ep = entity_sub.add_parser("merge", help="merge entity MERGE_ID into KEEP_ID directly")
    ep.add_argument("keep_id")
    ep.add_argument("merge_id")
    ep = entity_sub.add_parser("resolve", help="re-judge open proposals with the LLM")
    _scope_args(ep)

    p = sub.add_parser("context", help="build a context block for a task")
    p.add_argument("query")
    _scope_args(p)
    p.add_argument("--budget", type=int, default=1200)

    p = sub.add_parser("get", help="show one memory")
    p.add_argument("memory_id")

    p = sub.add_parser("delete", help="forget a memory (soft delete)")
    p.add_argument("memory_id")
    p.add_argument("--hard", action="store_true")

    p = sub.add_parser("history", help="audit trail for one memory")
    p.add_argument("memory_id")

    sub.add_parser("stats", help="store statistics")

    p = sub.add_parser("sweep", help="decay sweep: soft-forget stale memories")
    p.add_argument("--threshold", type=float, default=0.1)

    sub.add_parser("reindex", help="re-embed all memories with the current embedder")

    p = sub.add_parser("export", help="export memories as JSONL to stdout")
    _scope_args(p)

    p = sub.add_parser("import", help="import memories from a JSONL file (verbatim)")
    p.add_argument("path")

    sub.add_parser("config", help="print resolved configuration (keys redacted)")

    p = sub.add_parser("eval", help="run the retrieval eval harness")
    p.add_argument("--dataset", required=True)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main(transport=args.transport, host=args.host, port=args.port)
        return 0

    if args.command == "serve":
        from .rest import main as serve_main

        serve_main(host=args.host, port=args.port)
        return 0

    if args.command == "config":
        _print(Config.load().redacted())
        return 0

    if args.command == "eval":
        from .evals.harness import run_eval

        report = run_eval(args.dataset, k=args.k)
        if args.as_json:
            _print(report)
        else:
            print(format_eval_report(report))
        return 0

    store = _store()
    try:
        if args.command == "entities":
            sub_command = getattr(args, "entities_command", None)
            if sub_command == "list" or sub_command is None:
                entities = store.entities(
                    user_id=getattr(args, "user", None),
                    agent_id=getattr(args, "agent", None),
                    run_id=getattr(args, "run", None),
                    limit=getattr(args, "limit", 50),
                )
                _print([e.model_dump(exclude={"metadata"}) for e in entities])
            elif sub_command == "show":
                detail = store.entity(args.entity_id)
                if detail is None:
                    print("not found", file=sys.stderr)
                    return 1
                _print(
                    {
                        "entity": detail["entity"].model_dump(),
                        "mentions": [m.model_dump() for m in detail["mentions"]],
                        "memories": [
                            {"id": m.id, "content": m.content} for m in detail["memories"]
                        ],
                    }
                )
            elif sub_command == "proposals":
                proposals = store.merge_proposals(
                    user_id=getattr(args, "user", None), status=args.status
                )
                _print([p.model_dump() for p in proposals])
            elif sub_command == "confirm":
                _print({"confirmed": store.confirm_merge(args.proposal_id)})
            elif sub_command == "reject":
                _print({"rejected": store.reject_merge(args.proposal_id)})
            elif sub_command == "merge":
                _print({"merged": store.merge_entities(args.keep_id, args.merge_id)})
            elif sub_command == "resolve":
                _print(store.resolve_entities(user_id=getattr(args, "user", None)))
        elif args.command == "add":
            result = store.add(
                args.text, user_id=args.user, agent_id=args.agent, run_id=args.run,
                infer=not args.no_infer, categories=args.category,
            )
            _print(result.model_dump())
        elif args.command == "search":
            results = store.search(
                args.query, user_id=args.user, agent_id=args.agent, run_id=args.run,
                limit=args.limit, categories=args.category,
            )
            _print(
                [
                    {"score": round(r.score, 4), "id": r.memory.id, "content": r.memory.content,
                     "type": r.memory.memory_type, "signals": {k: round(v, 4) for k, v in r.signals.items()}}
                    for r in results
                ]
            )
        elif args.command == "list":
            memories = store.get_all(
                user_id=args.user, agent_id=args.agent, run_id=args.run,
                include_invalid=args.all, limit=args.limit, categories=args.category,
            )
            _print([m.model_dump(exclude={"metadata", "source_episode_ids"}) for m in memories])
        elif args.command == "context":
            ctx = store.reconstruct_context(
                args.query, user_id=args.user, agent_id=args.agent, run_id=args.run,
                token_budget=args.budget,
            )
            print(ctx.text or "(no relevant memories)")
        elif args.command == "get":
            memory = store.get(args.memory_id)
            if memory is None:
                print("not found", file=sys.stderr)
                return 1
            _print(memory.model_dump())
        elif args.command == "delete":
            ok = store.delete(args.memory_id, hard=args.hard)
            _print({"deleted": ok})
        elif args.command == "history":
            _print([e.model_dump() for e in store.history(args.memory_id)])
        elif args.command == "stats":
            _print(store.stats())
        elif args.command == "sweep":
            forgotten = store.decay_sweep(threshold=args.threshold)
            _print({"forgotten": forgotten, "count": len(forgotten)})
        elif args.command == "reindex":
            count = store.reindex()
            _print({"reindexed": count, "embedder": store.embedder.model_id})
        elif args.command == "export":
            for memory in store.get_all(
                user_id=args.user, agent_id=args.agent, run_id=args.run,
                include_invalid=True, limit=1_000_000,
            ):
                print(memory.model_dump_json())
        elif args.command == "import":
            with open(args.path, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            result = store.import_verbatim(rows)
            _print({k: v for k, v in result.items() if k != "memory_ids"})
    finally:
        store.close()
    return 0


def format_eval_report(report: dict[str, Any]) -> str:
    lines = [
        f"dataset:      {report['dataset']}",
        f"cases:        {report['cases']}  questions: {report['questions']}",
        f"memories:     {report['memories_stored']}",
        f"recall@{report['k']}:     {report['recall_at_k']:.3f}",
        f"MRR:          {report['mrr']:.3f}",
        f"search p50:   {report['latency_ms_p50']:.1f} ms   p95: {report['latency_ms_p95']:.1f} ms",
        f"llm:          {report['llm']}   embedder: {report['embedder']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
