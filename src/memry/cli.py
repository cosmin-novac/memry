"""Memry command-line interface.

    memry mcp                     run the MCP server (stdio)
    memry serve                   REST API + dashboard + /mcp
    memry add "text" -u ada       add a memory
    memry search "query" -u ada   search memories
    memry list -u ada             list memories
    memry context "task" -u ada   build a context block
    memry history <memory_id>     audit trail for one memory
    memry stats                   store statistics
    memry sweep                   decay sweep (soft-forget stale memories)
    memry reindex                 re-embed all memories
    memry backfill-property-vectors  property vectors for the linked search
    memry export / import         lossless backup/restore; legacy JSON imports
    memry tags-to-things          give existing tags their topic entities (first open does it)
    memry split-memories          split memories that hold several facts (--dry-run first)
    memry adopt-unscoped          give memories without a namespace one (--dry-run first)
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


def _accounts():
    from .accounts import AccountStore, default_auth_db_path

    cfg = Config.load()
    return AccountStore(cfg.auth_db_path or default_auth_db_path(cfg.db_path)), cfg


def _account_command(args: argparse.Namespace) -> int:
    accounts, cfg = _accounts()
    command = getattr(args, "account_command", None) or "list"
    name = getattr(args, "name", None)
    try:
        if command == "add":
            if any(t.name == name for t in cfg.tenants):
                print(
                    f"error: {name!r} is already a configured tenant; it would share "
                    "that namespace. Pick another name.",
                    file=sys.stderr,
                )
                return 1
            account = accounts.create(name, password=args.password)
            out = {"account": name, "created": True, "admin": account.is_admin}
            if not args.no_key:
                # printed once and never recoverable: only the hash is stored
                out["api_key"] = accounts.issue_key(name, label="initial")
            _print(out)
            return 0

        if command == "list":
            _print([
                {
                    "name": a.name,
                    "admin": a.is_admin,
                    "disabled": a.disabled,
                    "has_password": a.has_password,
                    "keys": len(accounts.keys_for(a.name)),
                    "created_at": a.created_at,
                }
                for a in accounts.list()
            ])
            return 0

        if command == "issue-key":
            _print({"account": name, "api_key": accounts.issue_key(name, label=args.label)})
            return 0

        if command == "revoke-keys":
            _print({"account": name, "revoked": accounts.revoke_keys(name)})
            return 0

        if command == "passwd":
            if not accounts.set_password(name, args.password):
                print(f"no such account: {name}", file=sys.stderr)
                return 1
            _print({"account": name, "password_set": True})
            return 0

        if command in ("disable", "enable"):
            if not accounts.set_disabled(name, command == "disable"):
                print(f"no such account: {name}", file=sys.stderr)
                return 1
            _print({"account": name, "disabled": command == "disable"})
            return 0

        if command == "delete":
            if not accounts.delete(name):
                print(f"no such account: {name}", file=sys.stderr)
                return 1
            _print({"account": name, "deleted": True})
            return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        accounts.close()

    print(f"unknown account command: {command}", file=sys.stderr)
    return 1


def _namespaces(store: Any, user: str | None) -> list[str | None]:
    """The namespaces a maintenance command goes through: the one asked for,
    else each in the store. None among them is the memories without a user;
    the command is called with ``exact_user=True`` so that its pass for None
    does not take in every namespace, each named one then done a second
    time."""
    return [user] if user else (store.backend.distinct_user_ids() or [None])


def _scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=None, help="user_id scope")
    parser.add_argument("-a", "--agent", default=None, help="agent_id scope")
    parser.add_argument("-r", "--run", default=None, help="run_id scope")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memry", description=__doc__)
    parser.add_argument("--version", action="version", version=f"memry {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("mcp", help="run the local stdio MCP server")


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
    p.add_argument("--entity", default=None, help="restrict to an exact entity ID")
    p.add_argument("--since", default=None, help="created on/after YYYY-MM-DD")
    p.add_argument("--until", default=None, help="created on/before YYYY-MM-DD")

    p = sub.add_parser("list", help="list memories")
    _scope_args(p)
    p.add_argument("-n", "--limit", type=int, default=50)
    p.add_argument("--all", action="store_true", help="include invalidated memories")
    p.add_argument("-c", "--category", action="append", default=None,
                   help="restrict to a category (repeatable)")
    p.add_argument("--entity", default=None, help="restrict to an exact entity ID")
    p.add_argument("--since", default=None, help="created on/after YYYY-MM-DD")
    p.add_argument("--until", default=None, help="created on/before YYYY-MM-DD")

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
    ep = entity_sub.add_parser("merges", help="list merges that can be undone")
    _scope_args(ep)
    ep = entity_sub.add_parser(
        "unmerge", help="undo the merge of ENTITY_ID (the one merged away); the two stay apart")
    ep.add_argument("entity_id")
    ep = entity_sub.add_parser("alias", help="add a user-supplied alias to an entity")
    ep.add_argument("entity_id")
    ep.add_argument("alias")
    ep = entity_sub.add_parser("resolve", help="re-judge open proposals with the LLM")
    _scope_args(ep)

    p = sub.add_parser("account", help="manage multiuser accounts")
    account_sub = p.add_subparsers(dest="account_command")
    ap = account_sub.add_parser("add", help="create an account and mint its API key")
    ap.add_argument("name")
    ap.add_argument("--password", default=None,
                    help="password for dashboard/OAuth login (optional)")
    ap.add_argument("--no-key", action="store_true",
                    help="create the account without minting an API key")
    ap = account_sub.add_parser("list", help="list accounts")
    ap = account_sub.add_parser("issue-key", help="mint another API key for an account")
    ap.add_argument("name")
    ap.add_argument("--label", default=None, help="what this key is for")
    ap = account_sub.add_parser("revoke-keys", help="revoke every API key of an account")
    ap.add_argument("name")
    ap = account_sub.add_parser("passwd", help="set an account password")
    ap.add_argument("name")
    ap.add_argument("password")
    ap = account_sub.add_parser("disable", help="disable an account (keys stop working)")
    ap.add_argument("name")
    ap = account_sub.add_parser("enable", help="re-enable a disabled account")
    ap.add_argument("name")
    ap = account_sub.add_parser("delete", help="delete an account (memories are kept)")
    ap.add_argument("name")

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

    p = sub.add_parser(
        "backfill-relations",
        help="extract typed relations from existing memories (one-time, cheap)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")

    p = sub.add_parser(
        "backfill-entity-types",
        help="classify existing untyped entities (one-time, batched/cheap)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")

    p = sub.add_parser(
        "repair-dates",
        help="recompute updated_at from the audit trail (token-free)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")

    p = sub.add_parser(
        "restore-context",
        help="give memories back the context label of the saves they came from (token-free)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")
    p.add_argument("--dry-run", action="store_true", help="count without writing")

    p = sub.add_parser(
        "split-memories",
        help="split each memory in use that holds several facts into one memory per fact "
             "(asks the text model; undo under Archive or with --undo)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")
    p.add_argument("--dry-run", action="store_true",
                   help="ask the model and print each split, writing nothing")
    p.add_argument("--min-words", type=int, default=0,
                   help="ask only about memories of at least this many words")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print the summary as JSON")
    p.add_argument("--undo", metavar="MEMORY_ID", default=None,
                   help="bring back a memory that was split and forget its facts")
    p.add_argument("--plan-out", metavar="PATH", default=None,
                   help="with --dry-run: write the proposed splits to PATH as a JSON plan")
    p.add_argument("--plan-in", metavar="PATH", default=None,
                   help="make exactly the splits of a plan from --plan-out, asking no "
                        "model; a memory that left use or changed since is skipped")

    p = sub.add_parser(
        "adopt-unscoped",
        help="move every memory without a namespace, with its entities, relations and "
             "upkeep state, into one (default: the default namespace), in one transaction",
    )
    p.add_argument("--into", default=None, metavar="NAMESPACE",
                   help="the namespace they go to (default: the configured default user)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would move and fold, writing nothing")

    sub.add_parser("reindex", help="re-embed all memories with the current embedder")

    p = sub.add_parser(
        "tags-to-things",
        help="make every existing tag a topic entity and each tagged memory a "
             "mention of it (token-free, idempotent; the legacy tag tables are "
             "only read)",
    )
    p.add_argument("-u", "--user", default=None,
                   help="namespace to migrate (default: every namespace)")
    p.add_argument("--dry-run", action="store_true", help="count without writing")

    p = sub.add_parser(
        "backfill-property-vectors",
        help="embed each memory with its entity names masked, for the linked search "
             "(only what is missing or changed)",
    )
    p.add_argument("-u", "--user", default=None, help="namespace (default: every namespace)")

    p = sub.add_parser("export", help="export a lossless JSON backup to stdout")
    _scope_args(p)

    p = sub.add_parser("import", help="restore a Memry backup or import legacy JSON/JSONL")
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

        mcp_main()
        return 0

    if args.command == "serve":
        from .rest import main as serve_main

        serve_main(host=args.host, port=args.port)
        return 0

    if args.command == "config":
        from .config import model_requirements

        cfg = Config.load()
        _print(cfg.redacted())
        for item in model_requirements(cfg):
            print(f"not set yet for memry serve / memry mcp: {item}", file=sys.stderr)
        return 0

    if args.command == "account":
        return _account_command(args)

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
                        "aliases": detail["aliases"],
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
            elif sub_command == "merges":
                _print(store.merges(user_id=getattr(args, "user", None)))
            elif sub_command == "unmerge":
                result = store.undo_merge(args.entity_id)
                _print(result)
                if not result["undone"]:
                    return 1
            elif sub_command == "alias":
                entity = store.add_entity_alias(args.entity_id, args.alias)
                if entity is None:
                    print("not found", file=sys.stderr)
                    return 1
                _print({"entity": entity.model_dump(), "aliases": store.backend.entity_aliases(entity.id)})
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
                limit=args.limit, categories=args.category, entity_id=args.entity,
                since=args.since, until=args.until,
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
                entity_id=args.entity, since=args.since, until=args.until,
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
        elif args.command == "backfill-relations":
            if not store.llm.available:
                print("no LLM configured; relation backfill needs one", file=sys.stderr)
                return 1
            namespaces = _namespaces(store, args.user)
            _print([store.backfill_relations(user_id=uid, exact_user=True)
                    for uid in namespaces])
        elif args.command == "backfill-entity-types":
            if not store.llm.available:
                print("no LLM configured; entity typing needs one", file=sys.stderr)
                return 1
            namespaces = _namespaces(store, args.user)
            _print([store.backfill_entity_types(user_id=uid, exact_user=True)
                    for uid in namespaces])
        elif args.command == "repair-dates":
            namespaces = _namespaces(store, args.user)
            _print([store.repair_updated_at(user_id=uid, exact_user=True)
                    for uid in namespaces])
        elif args.command == "restore-context":
            namespaces = _namespaces(store, args.user)
            _print([store.restore_context_labels(user_id=uid, dry_run=args.dry_run,
                                                 exact_user=True)
                    for uid in namespaces])
        elif args.command == "split-memories":
            if args.undo:
                try:
                    undone = store.undo_replacement(args.undo)
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                _print({"undone": undone, "memory_id": args.undo})
                return 0 if undone else 1
            from .intelligence.split import make_plan, plan_entries

            if args.plan_out and (args.plan_in or not args.dry_run):
                print("error: --plan-out goes with --dry-run, and not with --plan-in",
                      file=sys.stderr)
                return 1
            if args.plan_in:
                # the plan's splits as they were read; no model is asked
                try:
                    with open(args.plan_in, encoding="utf-8") as fh:
                        plan = plan_entries(json.load(fh))
                except (OSError, ValueError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
                # each namespace's splits in that namespace alone (exact_user)
                namespaces = ([args.user] if args.user
                              else list(dict.fromkeys(entry["user"] for entry in plan)))
                left_out = sum(entry["user"] not in namespaces for entry in plan)
                if left_out:
                    print(f"{left_out} planned splits of other namespaces left out",
                          file=sys.stderr)
                reports = [store.split_memories(
                    user_id=uid, dry_run=args.dry_run, exact_user=True,
                    plan=[entry for entry in plan if entry["user"] == uid])
                    for uid in namespaces]
            else:
                if not store.llm.available:
                    print("no LLM configured; splitting memories needs one", file=sys.stderr)
                    return 1
                reports = [store.split_memories(user_id=uid, dry_run=args.dry_run,
                                                min_words=args.min_words, exact_user=True)
                           for uid in _namespaces(store, args.user)]
            if args.as_json:
                _print(reports)
            else:
                print(format_split_report(reports))
            if args.plan_out:
                # after the report: a plan that cannot be written leaves it shown
                plan_file = make_plan(reports)
                try:
                    with open(args.plan_out, "w", encoding="utf-8") as fh:
                        json.dump(plan_file, fh, indent=2, ensure_ascii=False)
                except OSError as exc:
                    print(f"error: the plan was not written: {exc}", file=sys.stderr)
                    return 1
                print(f"plan of {len(plan_file['splits'])} splits written to {args.plan_out}; "
                      f"make exactly these with: memry split-memories --plan-in {args.plan_out}",
                      file=sys.stderr)
        elif args.command == "adopt-unscoped":
            _print(store.adopt_unscoped(into=args.into, dry_run=args.dry_run))
        elif args.command == "backfill-property-vectors":
            namespaces = _namespaces(store, args.user)
            _print([{"user": uid,
                     "embedded": store.refresh_property_vectors(user_id=uid, exact_user=True)}
                    for uid in namespaces])
        elif args.command == "tags-to-things":
            scopes = store.tags_to_topics(
                user_id=args.user, all_users=args.user is None, dry_run=args.dry_run)
            totals = {
                key: sum(row[key] for row in scopes)
                for key in ("topics", "entities_created",
                            "entities_existing", "mentions_created", "mentions_existing")
            }
            _print({"dry_run": args.dry_run, "scopes": scopes, "total": totals})
        elif args.command == "reindex":
            count = store.reindex()
            _print({"reindexed": count, "embedder": store.embedder.model_id})
        elif args.command == "export":
            backup = store.export_backup(
                user_id=args.user, agent_id=args.agent, run_id=args.run,
            )
            print(json.dumps(backup, ensure_ascii=False))
        elif args.command == "import":
            with open(args.path, encoding="utf-8") as fh:
                text = fh.read().strip()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = [json.loads(line) for line in text.splitlines() if line.strip()]
            if isinstance(payload, dict) and payload.get("format") == "memry-backup":
                result = store.import_backup(payload)
            else:
                rows = payload if isinstance(payload, list) else [payload]
                result = store.import_verbatim(rows)
                result = {k: v for k, v in result.items() if k != "memory_ids"}
            _print(result)
    finally:
        store.close()
    return 0


def format_split_report(reports: list[dict[str, Any]]) -> str:
    """``split-memories`` for a person to read: the counts per namespace, then
    each memory split (or that would be), with its facts and under each the
    entities it keeps, for the owner to read before the split is made."""
    lines: list[str] = []
    for report in reports:
        verb = "would be split" if report["dry_run"] else "split"
        if "planned" in report:  # a run of a plan (--plan-in)
            lines.append(
                f"namespace {report['user'] or '(none)'}: {report['planned']} splits planned, "
                f"{report['split']} {verb} into {report['facts']} facts as planned, "
                f"{report['stale']} skipped because the memory left use or changed since "
                f"the plan, {report['failed']} failed"
                + (" (dry run: nothing written)" if report["dry_run"] else ""))
        else:
            lines.append(
                f"namespace {report['user'] or '(none)'}: {report['in_use']} memories in use, "
                f"{report['candidates']} with more than one sentence asked, "
                f"{report['one_fact']} one fact (left alone), {report['split']} {verb} into "
                f"{report['facts']} facts, {report['no_entity']} left because a fact would "
                f"keep none of the memory's entities, {report['lost_entity']} left because "
                f"an entity would be lost, {report['no_subject']} left because a fact would "
                f"not state its subject, {report['lossy']} left because the facts would lose "
                f"a detail, {report['failed']} failed"
                + (" (dry run: nothing written)" if report["dry_run"] else ""))
        for entry in report["splits"]:
            lines.append("")
            lines.append(f"memory {entry['memory_id']}:")
            lines.append(f"  {entry['content']}")
            if entry.get("not_split"):
                lines.append(f"  not split: {entry['not_split']}")
            labels = entry.get("labels") or {}
            for i, fact in enumerate(entry["facts"], 1):
                made = entry.get("memory_ids")
                lines.append(f"  {i}. {fact}" + (f"  [{made[i - 1]}]" if made else ""))
                if labels:  # a memory linked to nothing has nothing to show
                    kept = entry["about"][i - 1]
                    lines.append("     about: " + (", ".join(labels.get(e, e) for e in kept)
                                                   or "nothing"))
        lines.append("")
    if reports and not reports[0]["dry_run"]:
        lines.append("Undo one: memry split-memories --undo MEMORY_ID, or undo under Archive "
                     "in the dashboard.")
    return "\n".join(lines).rstrip() + "\n"


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
