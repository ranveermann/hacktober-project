"""Command-line interface: `gittrace <command>`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .ai import DEFAULT_HOST, DEFAULT_MODEL, AIError, OllamaClient
from .analysis import Analyzer
from .git import Git, GitError, NotARepositoryError
from .provenance import ProvenanceIndex
from .render import Out, dump_json, render_report, render_tests, render_trace
from .resolve import CHOICES, ResolveError, commit, resolve_hunk
from .status import deploy_status, test_status
from .store import ENVIRONMENTS, RESULTS, Store, ValidationError


def _meta(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValidationError(f"--meta expects key=value, got '{p}'")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gittrace", description="Explain Git merge conflicts with change lineage, testing and deployment provenance.")
    p.add_argument("--version", action="version", version=f"gittrace {__version__}")
    p.add_argument("-C", dest="repo", default=None, help="run as if started in this directory")
    p.add_argument("--plain", action="store_true", help="plain text output (no Rich formatting)")
    p.add_argument("--history-limit", type=int, default=5000, help="commits scanned for lineage (default 5000)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", aliases=["status"], help="analyze unresolved conflicts in the repository")
    a.add_argument("path", nargs="?", help="only this conflicted path")
    a.add_argument("--json", action="store_true")
    a.add_argument("--no-lineage", action="store_true", help="skip lineage/similarity analysis (faster)")
    a.add_argument("--ai", action="store_true", help="ask the LOCAL Ollama model about Side A vs Side B")
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--ollama-host", default=DEFAULT_HOST)

    ac = sub.add_parser("ai-check", help="check the local Ollama server and model")
    ac.add_argument("--model", default=DEFAULT_MODEL)
    ac.add_argument("--ollama-host", default=DEFAULT_HOST)

    t = sub.add_parser("trace", help="show lineage, testing and deployment provenance of a commit")
    t.add_argument("commit")
    t.add_argument("--json", action="store_true")

    rt = sub.add_parser("record-test", help="record a test result for an exact commit snapshot")
    rt.add_argument("--commit", required=True)
    rt.add_argument("--environment", "-e", required=True, help="/".join(ENVIRONMENTS))
    rt.add_argument("--result", "-r", required=True, help="/".join(RESULTS))
    rt.add_argument("--suite")
    rt.add_argument("--run-id")
    rt.add_argument("--source", default="manual", help="evidence source, e.g. ci, jenkins, manual")
    rt.add_argument("--meta", action="append", metavar="KEY=VALUE")

    ts = sub.add_parser("tests", help="show testing provenance for a commit")
    ts.add_argument("commit")
    ts.add_argument("--json", action="store_true")

    rd = sub.add_parser("record-deploy", help="record that a commit was deployed to an environment")
    rd.add_argument("--commit", required=True)
    rd.add_argument("--environment", "-e", required=True)
    rd.add_argument("--reference", help="release tag / deployment id / pipeline URL")
    rd.add_argument("--source", default="manual")
    rd.add_argument("--meta", action="append", metavar="KEY=VALUE")

    eb = sub.add_parser("env-branch", help="map environments to long-lived branches")
    ebs = eb.add_subparsers(dest="action", required=True)
    s = ebs.add_parser("set")
    s.add_argument("environment")
    s.add_argument("branch")
    u = ebs.add_parser("unset")
    u.add_argument("environment")
    ebs.add_parser("list")

    ch = sub.add_parser("cherry", help="which commits of HEAD are already in UPSTREAM (git cherry + lineage)")
    ch.add_argument("upstream")
    ch.add_argument("head", nargs="?", default="HEAD")
    ch.add_argument("--json", action="store_true")

    rs = sub.add_parser("resolve", help="apply YOUR chosen resolution to one conflict hunk")
    rs.add_argument("path")
    rs.add_argument("--hunk", type=int, required=True)
    rs.add_argument("--take", required=True, choices=CHOICES)
    rs.add_argument("--stage", action="store_true", help="git add the file once no conflict markers remain")

    cm = sub.add_parser("commit", help="create a normal Git commit with GitTrace-Resolution trailers")
    cm.add_argument("-m", "--message")
    cm.add_argument("--no-trailers", action="store_true")

    ex = sub.add_parser("export", help="export testing/deployment records to a JSON file")
    ex.add_argument("file")
    im = sub.add_parser("import", help="merge testing/deployment records from a JSON file")
    im.add_argument("file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Out(plain=args.plain)
    err = Out(plain=True, stream=sys.stderr)
    try:
        git = Git(args.repo)
        git.ensure_repo()
        store = Store(git)
        return _dispatch(args, git, store, out)
    except (NotARepositoryError, ValidationError, ResolveError, AIError, ValueError) as exc:
        err.line(f"gittrace: error: {exc}")
        return 2
    except GitError as exc:
        err.line(f"gittrace: git error: {exc}")
        return 3
    except BrokenPipeError:  # output piped into `head` etc.
        try:  # silence the final flush at interpreter exit
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0


def _dispatch(args, git: Git, store: Store, out: Out) -> int:
    cmd = args.cmd
    if cmd in ("analyze", "status"):
        client = OllamaClient(args.ollama_host, args.model, cache_path=store.dir / "ai-cache.json") if args.ai else None
        report = Analyzer(git, store, args.history_limit, lineage=not args.no_lineage, ai_client=client).analyze(args.path)
        if args.json:
            print(dump_json(report))
        else:
            render_report(out, report)
        return 0

    if cmd == "ai-check":
        h = OllamaClient(args.ollama_host, args.model).health()
        if not h["reachable"]:
            out.line(f"Ollama NOT reachable at {args.ollama_host}: {h['error']}  ->  run `ollama serve`")
            return 1
        out.line(f"Ollama reachable at {args.ollama_host}. Installed models: {', '.join(h['models']) or 'none'}")
        if not h["model_available"]:
            out.line(f"Model '{args.model}' missing  ->  run `ollama pull {args.model}`")
            return 1
        out.line(f"Model '{args.model}' is available. No cloud AI is used.")
        return 0

    if cmd == "trace":
        idx = ProvenanceIndex(git, args.history_limit)
        info = idx.ensure(args.commit)
        if not info:
            raise ValidationError(f"unknown commit '{args.commit}'")
        data = dict(
            commit=info,
            references=idx.references(info.hash),
            reverts=idx.reverts(info.hash),
            lineage=idx.lineage(info.hash),
            tests=test_status(git, store, idx, info.hash),
            deploy=deploy_status(git, store, idx, info.hash),
        )
        if args.json:
            print(dump_json(data))
        else:
            render_trace(out, info, data["references"], data["lineage"], data["tests"], data["deploy"], data["reverts"])
        return 0

    if cmd == "record-test":
        rec = store.add_test(args.commit, args.environment, args.result, args.suite, args.run_id, args.source, _meta(args.meta))
        out.line(f"recorded test {rec.id}: {rec.commit[:10]} (tree {rec.tree[:10]}) {rec.environment} = {rec.result}")
        return 0

    if cmd == "tests":
        idx = ProvenanceIndex(git, args.history_limit)
        st = test_status(git, store, idx, args.commit)
        if args.json:
            print(dump_json(st))
        else:
            render_tests(out, st)
        return 0

    if cmd == "record-deploy":
        rec = store.add_deploy(args.commit, args.environment, args.reference, args.source, _meta(args.meta))
        out.line(f"recorded deployment {rec.id}: {rec.commit[:10]} -> {rec.environment}")
        return 0

    if cmd == "env-branch":
        if args.action == "set":
            store.set_env_branch(args.environment, args.branch)
            if not git.branch_exists(args.branch):
                out.line(f"warning: branch '{args.branch}' does not exist (yet)", "yellow")
        elif args.action == "unset":
            store.unset_env_branch(args.environment)
        for env in ENVIRONMENTS:
            out.line(f"{env:12} -> {store.env_branches.get(env, '(not set)')}")
        return 0

    if cmd == "cherry":
        idx = ProvenanceIndex(git, args.history_limit)
        rows = []
        for sign, c in idx.cherry(args.upstream, args.head):
            info = idx.ensure(c)
            rels = [r for r in idx.lineage(c) if git.is_ancestor(r.target, args.upstream)]
            best = rels[0] if rels else None
            rows.append({
                "commit": c, "subject": info.subject if info else "",
                "in_upstream": "equivalent patch" if sign == "-" else (best.relation.value if best else "missing"),
                "upstream_commit": best.target if best else None,
                "confidence": best.confidence if best else ("high" if sign == "-" else None),
            })
        if args.json:
            print(dump_json(rows))
        else:
            out.table(f"Commits on {args.head} vs {args.upstream}", ["commit", "subject", "status in upstream", "upstream commit", "confidence"],
                      [[r["commit"][:10], r["subject"][:50], r["in_upstream"], (r["upstream_commit"] or "-")[:10], r["confidence"] or "-"] for r in rows])
        return 0

    if cmd == "resolve":
        res = resolve_hunk(git, store, args.path, args.hunk, args.take, args.stage)
        out.line(f"{res['path']} hunk {res['hunk']}: took {res['choice']}. {res['remaining_hunks']} conflict hunk(s) left" + ("; staged." if res["staged"] else "."))
        return 0

    if cmd == "commit":
        h = commit(git, store, args.message, trailers=not args.no_trailers)
        out.line(f"created commit {h[:10]}")
        return 0

    if cmd == "export":
        n = store.export(Path(args.file))
        out.line(f"exported {n} record(s) to {args.file}")
        return 0

    if cmd == "import":
        added, skipped = store.import_(Path(args.file))
        out.line(f"imported {added} record(s), skipped {skipped} duplicate(s)")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
