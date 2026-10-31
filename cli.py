#!/usr/bin/env python3
"""GitTrace: AI-powered Git conflict resolution in your terminal.

Usage
-----
    python cli.py                                   # analyse the current directory
    python cli.py path/to/repo                      # analyse a local clone
    python cli.py https://github.com/user/repo.git  # clone to a temp dir, then analyse
    python cli.py <url> --merge production uat      # clone, start `git merge uat` into production

For each conflict you see Gemma's local analysis, then Side A and Side B side by side with
syntax highlighting. Type  A  to keep Side A,  B  to keep Side B,  S  to skip,  D  for details,
or  Q  to abort. Your choice is written to the file, staged, and committed with a standard
`git commit` (a normal two-parent merge commit) whose message records the provenance.

Dependencies (pip): rich, requests. The AI runs locally in Ollama (default model gemma4:e4b).
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))  # run straight from a checkout, no install needed

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover
    sys.exit("GitTrace needs the 'rich' package:  pip install rich requests")

from gittrace.ai import DEFAULT_HOST, DEFAULT_MODEL, AIAssessment, NonLocalEndpointError, OllamaClient  # noqa: E402
from gittrace.analysis import Analyzer, Badge, HunkReport, SideReport, describe_choice, side_badge  # noqa: E402
from gittrace.conflicts import Operation  # noqa: E402
from gittrace.git import Git, GitError  # noqa: E402
from gittrace.provenance import Relationship  # noqa: E402
from gittrace.repolink import RepoLinkError, branches, start_merge  # noqa: E402
from gittrace.resolve import ResolveError, resolve_and_finish  # noqa: E402
from gittrace.store import Store  # noqa: E402

PUMPKIN = "#FF7518"
PHANTOM = "#A78BFA"
GHOST = "#4ADE80"
console = Console(highlight=False)

LEVEL_STYLE = {"success": f"bold {GHOST}", "warning": f"bold {PUMPKIN}", "error": "bold red", "info": f"bold {PHANTOM}"}
LEVEL_ICON = {"success": "✔", "warning": "▲", "error": "✖", "info": "●"}
VERDICT_UI = {
    "SAME_FIX": ("Logically the SAME fix", GHOST, "green"),
    "PARTIALLY_SAME": ("PARTIALLY the same", "yellow", "yellow"),
    "DIFFERENT_CHANGES": ("DIFFERENT changes", "red", "red"),
    "UNCLEAR": ("Not enough evidence", "bright_black", "bright_black"),
}


class Abort(Exception):
    """The user chose to quit."""


# =====================================================================================
# 1. Locating the repository: local directory or remote URL (cloned to a temp dir)
# =====================================================================================
def is_remote_url(target: str) -> bool:
    t = target.strip()
    if t.startswith(("http://", "https://", "ssh://", "git://", "file://", "git@")):
        return True
    return t.endswith(".git") and not Path(t).exists()


def clone_to_temp(url: str) -> Path:
    """`git clone` a remote repository into a fresh temporary directory."""
    tmp = Path(tempfile.mkdtemp(prefix="gittrace-"))
    dest = tmp / (url.rstrip("/").split("/")[-1].removesuffix(".git") or "repo")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # never hang on a password prompt
    with console.status(f"[bold]Cloning[/] {url} …", spinner="dots"):
        proc = subprocess.run(["git", "clone", "--quiet", url, str(dest)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        reason = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise RepoLinkError(f"git clone failed: {reason}\n(For private repositories, clone it yourself and run `python cli.py <path>`.)")
    console.print(f"[{GHOST}]✔[/] Cloned into [bold]{dest}[/]")
    return dest


def open_repo(target: str | None) -> tuple[Git, Path | None]:
    """Return (Git for the repo, temp dir to clean up or None)."""
    if target and is_remote_url(target):
        dest = clone_to_temp(target)
        return Git(dest), dest.parent
    path = Path(target or os.getcwd()).expanduser()
    if not path.is_dir():
        raise RepoLinkError(f"'{path}' is not a directory or a repository URL.")
    g = Git(path)
    if not g.ok("rev-parse", "--git-dir"):
        raise RepoLinkError(f"'{path}' is not inside a Git repository.")
    return g, None


# =====================================================================================
# 2. Rendering
# =====================================================================================
def header(git: Git, report, total: int) -> Panel:
    ctx = report.context
    title = Text.assemble(("Git", f"bold {PUMPKIN}"), ("Trace", "bold white"), ("  ·  AI-powered conflict resolution", "dim"))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()
    grid.add_row("repository", Text(str(git.toplevel), style="bold"))
    grid.add_row("operation", Text(ctx.operation.value, style=f"bold {PHANTOM}"))
    grid.add_row("Side A", Text(ctx.current.label, style=PUMPKIN))
    grid.add_row("Side B", Text(ctx.incoming.label, style=PHANTOM))
    grid.add_row("conflicts", Text(f"{total} hunk(s) in {len(report.files)} file(s) remaining", style="bold"))
    return Panel(grid, title=title, title_align="left", border_style="bright_black", box=box.ROUNDED, padding=(1, 2))


def confidence_bar(conf: float, color: str, width: int = 30) -> Text:
    filled = round(conf * width)
    return Text.assemble(("█" * filled, color), ("░" * (width - filled), "bright_black"), (f"  {conf:.0%}", f"bold {color}"))


def ai_panel(assessment: AIAssessment | None, rel: Relationship | None, client: OllamaClient | None) -> Panel:
    if client is None:
        body = Text("Local AI is turned off (--no-ai). Deterministic provenance is shown below.", style="dim")
        return Panel(body, title="🤖 Gemma analysis", title_align="left", border_style="bright_black", box=box.HEAVY)
    if assessment is None or not assessment.ok:
        msg = assessment.error if assessment is not None else "unavailable"
        body = Group(
            Text("Local AI is offline: provenance below is still deterministic.", style="bold yellow"),
            Text(msg or "", style="yellow"),
            Text(f"\nStart it with:  ollama serve   ·   ollama pull {client.model}", style="dim"),
        )
        return Panel(body, title="⚠  Gemma analysis unavailable", title_align="left", border_style="yellow", box=box.HEAVY)

    label, color, border = VERDICT_UI.get(assessment.verdict, VERDICT_UI["UNCLEAR"])
    parts: list = [
        Text.assemble(("Verdict  ", "dim"), (label, f"bold {color}")),
        Text.assemble(("Confidence  ", "dim"), confidence_bar(assessment.confidence, color)),
        Text(""),
        Text(assessment.explanation, style="white"),
    ]
    if assessment.key_differences:
        parts.append(Text(""))
        parts.append(Text("Key behavioural differences", style="bold"))
        parts += [Text.assemble(("  • ", color), (d, "white")) for d in assessment.key_differences[:6]]
    if rel is not None:
        mod = " (modified copy)" if rel.modified else ""
        parts.append(Text(""))
        parts.append(Text.assemble(("Lineage  ", "dim"), (f"{rel.relation.value}{mod}", "bold"), (f" · confidence {rel.confidence}", "dim")))
    meta = f"{assessment.model} · on-device" + (" · cached" if assessment.cached else f" · {assessment.duration_s:.1f}s")
    parts.append(Text(f"\n{meta} · advisory only: patch-id / reference evidence takes precedence, and you decide.", style="dim italic"))
    return Panel(Group(*parts), title="🤖 Gemma analysis (local)", title_align="left", subtitle="provenance & logic",
                 subtitle_align="right", border_style=border, box=box.HEAVY, padding=(1, 2))


def changed_lines(mine: list[str], other: list[str], start: int) -> set[int]:
    out: set[int] = set()
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, mine, other, autojunk=False).get_opcodes():
        if op != "equal":
            out.update(range(start + i1, start + i2))
    return out


def side_panel(key: str, side: SideReport, other_lines: list[str], badge: Badge, path: str, start: int, color: str) -> Panel:
    top = Table.grid(padding=(0, 1))
    top.add_column()
    top.add_row(Text.assemble((f" {LEVEL_ICON.get(badge.level, '●')} {badge.text} ", LEVEL_STYLE.get(badge.level, "bold"))))
    if side.commits:
        cc = side.commits[0]
        top.add_row(Text.assemble((cc.commit[:7] + " ", "dim"), (cc.subject, "white"), (f" · {cc.author}", "dim")))
    for d in (badge.details[1:3] if len(badge.details) > 1 else []):
        top.add_row(Text(f"  {d}", style="dim"))
    if side.lines:
        code = "\n".join(side.lines)
        lexer = Syntax.guess_lexer(path, code=code)
        body = Syntax(code, lexer, theme="monokai", line_numbers=True, start_line=start, word_wrap=True,
                      highlight_lines=changed_lines(side.lines, other_lines, start), background_color="default")
    else:
        body = Text("(this side deletes these lines)", style="dim italic")
    title = Text.assemble((f" {key} ", f"bold black on {color}"), (f"  Side {key}", f"bold {color}"))
    return Panel(Group(top, Rule(style="bright_black"), body), title=title, title_align="left",
                 subtitle=Text(side.label, style="dim"), subtitle_align="right", border_style=color, box=box.ROUNDED,
                 padding=(0, 1))


def code_view(hr: HunkReport, path: str, badges: dict) -> Table | Group:
    start = hr.hunk.start_line + 1
    a = side_panel("A", hr.current, hr.hunk.incoming_lines, badges["current"], path, start, PUMPKIN)
    b = side_panel("B", hr.incoming, hr.hunk.current_lines, badges["incoming"], path, start, PHANTOM)
    if console.width < 110:  # narrow terminal: stack instead of side by side
        return Group(a, b)
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(a, b)
    return grid


def details_view(hr: HunkReport) -> None:
    console.print(Rule("[bold]Provenance details", style="bright_black"))
    for key, side in (("A", hr.current), ("B", hr.incoming)):
        t = Table(title=f"Side {key} · originating commits", box=box.SIMPLE_HEAVY, title_style="bold", expand=True)
        for col in ("commit", "subject", "lines", "dev", "uat", "prod", "deployment"):
            t.add_column(col, overflow="fold")
        for cc in side.commits:
            st = cc.tests.by_environment
            dep = ", ".join(f"{f.environment}: {f.state}" for f in cc.deploy.findings if f.state not in ("absent", "unconfigured")) or "-"
            t.add_row(cc.commit[:10], cc.subject[:48], str(cc.lines or "-"), st["development"], st["uat"], st["production"], dep)
        console.print(t)
        for cc in side.commits[:2]:
            for r in cc.lineage[:3]:
                console.print(f"   [dim]{cc.commit[:10]} ~ {r.target[:10]}[/]  [bold]{r.relation.value}[/] [dim](confidence {r.confidence})[/]")
                for ev in r.evidence[:4]:
                    console.print(f"      [dim]· {ev}[/]")
    if hr.advisories:
        console.print(Panel("\n".join(f"• {a}" for a in hr.advisories), title="Context for your decision",
                            border_style="bright_black", box=box.ROUNDED))


# =====================================================================================
# 3. Interaction
# =====================================================================================
def ask(prompt: str, valid: dict[str, str]) -> str:
    """Case-insensitive single-key prompt. EOF (piped input ran out) counts as Quit."""
    while True:
        try:
            raw = console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "q"
        choice = raw.strip().lower()[:1]
        if choice in valid:
            return choice
        console.print(f"[red]Please type one of: {', '.join(k.upper() for k in valid)}[/]")


ACTION_PROMPT = (f"\n[bold black on {PUMPKIN}] A [/] Keep Side A   [bold black on {PHANTOM}] B [/] Keep Side B   "
                 "[bold black on white] S [/] Skip   [bold black on white] D [/] Details   [bold white on red] Q [/] Abort\n[bold]› [/]")


def gemma_for(analyzer: Analyzer, client: OllamaClient | None, hr: HunkReport, path: str):
    if client is None:
        return None, None
    with console.status(f"[bold {PHANTOM}]Gemma[/] is reading both sides locally ({client.model})…", spinner="dots12"):
        return analyzer.compare_hunk(client, hr, path)


def maybe_start_merge(git: Git, merge: list[str] | None, interactive: bool) -> bool:
    """If nothing is conflicted, optionally start a standard `git merge`. Returns True if conflicts now exist."""
    local, remote = branches(git)
    names = local + [r.split("/", 1)[1] for r in remote if r.split("/", 1)[1] not in local]
    if merge:
        target, incoming = merge
    elif interactive and len(names) >= 2:
        console.print(Panel(Text.assemble(("No merge conflict in progress.\n", "bold"),
                                          ("Branches: ", "dim"), (", ".join(names), "white"),
                                          ("\n\nStart a standard `git merge` between two branches to look for conflicts?", "dim")),
                            border_style="bright_black", box=box.ROUNDED))
        if ask("[bold]Start a merge? [y/N] › [/]", {"y": "yes", "n": "no", "q": "quit"}) != "y":
            return False
        target = pick_branch("Merge INTO (target)", names, git.current_branch())
        incoming = pick_branch("Bring IN (incoming)", [n for n in names if n != target], None)
    else:
        return False
    try:
        with console.status(f"Running [bold]git merge {incoming}[/] into [bold]{target}[/] …"):
            ref = incoming if incoming in local else next((r for r in remote if r.split("/", 1)[1] == incoming), incoming)
            res = start_merge(git, target, ref)  # fresh clones only have origin/<branch> for most branches
    except RepoLinkError as exc:
        console.print(f"[red]✖ {exc}[/]")
        return False
    if not res["conflicts"]:
        console.print(f"[{GHOST}]✔ Merged cleanly, nothing to resolve.[/]")
        return False
    return True


def pick_branch(label: str, names: list[str], default: str | None) -> str:
    t = Table(box=None, show_header=False, padding=(0, 1))
    for i, n in enumerate(names, 1):
        t.add_row(f"[bold {PUMPKIN}]{i}[/]", n + ("  [dim](current)[/]" if n == default else ""))
    console.print(Text(label, style="bold"))
    console.print(t)
    while True:
        try:
            raw = console.input("[bold]number › [/]").strip()
        except (EOFError, KeyboardInterrupt):
            raise Abort()
        if not raw and default in names:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw in names:
            return raw
        console.print("[red]Pick a number from the list.[/]")


def resolve_loop(git: Git, client: OllamaClient | None, interactive: bool) -> dict | None:
    """Walk through every conflict hunk. Returns the final `finish` result, or None if aborted/skipped."""
    store = Store(git)
    if client is not None:
        client.cache_path = store.dir / "ai-cache.json"
    skipped: set[str] = set()
    last_finish = None
    while True:
        with console.status("Tracing history, tests and deployments…", spinner="dots"):
            analyzer = Analyzer(git, store)
            report = analyzer.analyze()
        hunks = [(fr, hr) for fr in report.files for hr in fr.hunks]
        pending = [(fr, hr) for fr, hr in hunks if f"{fr.file.path}:{hr.hunk.current_lines}:{hr.hunk.incoming_lines}" not in skipped]
        structural = [fr for fr in report.files if not fr.hunks]
        if not report.files:
            return last_finish
        if not pending and not structural:
            console.print(f"[yellow]{len(hunks)} conflict(s) skipped and still unresolved. Run again to continue.[/]")
            return None
        if interactive and console.is_terminal:
            console.clear()
        console.print(header(git, report, len(hunks)))

        if not pending:  # only structural conflicts (deleted/binary files) left
            fr = structural[0]
            console.print(Panel(f"[bold]{fr.file.path}[/]  ·  {fr.file.kind}\n[dim]Structural conflict: choose which side's whole file to keep.[/]",
                                border_style="yellow", box=box.HEAVY))
            choice = ask(ACTION_PROMPT, {"a": "A", "b": "B", "q": "quit"}) if interactive else "q"
            if choice == "q":
                raise Abort()
            pick = "current" if choice == "a" else "incoming"
            last_finish = apply(git, store, fr.file.path, 0, pick, f"kept Side {choice.upper()} (whole file)", whole_file=True)
            continue

        fr, hr = pending[0]
        path = fr.file.path
        badges = {"current": side_badge(hr.current), "incoming": side_badge(hr.incoming)}
        assessment, rel = gemma_for(analyzer, client, hr, path)
        console.print(ai_panel(assessment, rel, client))
        console.print(Rule(Text.assemble((path, "bold"), (f"  ·  lines {hr.hunk.start_line}–{hr.hunk.end_line}", "dim"),
                                         (f"  ·  conflict 1 of {len(pending)}", "dim")), style="bright_black"))
        console.print(code_view(hr, path, badges))
        if not interactive:
            return None
        while True:
            choice = ask(ACTION_PROMPT, {"a": "keep A", "b": "keep B", "s": "skip", "d": "details", "q": "abort"})
            if choice == "d":
                details_view(hr)
                continue
            break
        if choice == "q":
            raise Abort()
        if choice == "s":
            skipped.add(f"{path}:{hr.hunk.current_lines}:{hr.hunk.incoming_lines}")
            continue
        pick = "current" if choice == "a" else "incoming"
        name = "Side A" if pick == "current" else "Side B"
        side = hr.current if pick == "current" else hr.incoming
        last_finish = apply(git, store, path, hr.hunk.index, pick, describe_choice(name, side, badges[pick], assessment))
        if last_finish and last_finish.get("committed"):
            return last_finish


def apply(git: Git, store: Store, path: str, hunk: int, pick: str, provenance: str, whole_file: bool = False) -> dict | None:
    """Call the backend: write lines -> git add -> commit once nothing is left unmerged."""
    name = "Side A" if pick == "current" else "Side B"
    try:
        with console.status(f"Keeping [bold]{name}[/]: writing, staging, committing…"):
            out = resolve_and_finish(git, store, path, hunk, pick, provenance=provenance, whole_file=whole_file)
    except (ResolveError, GitError, OSError, UnicodeDecodeError) as exc:
        console.print(f"[red]✖ Could not resolve: {exc}[/]")
        return None
    fin = out["finish"]
    if fin["action"] == "pending":
        left = len(fin.get("remaining", []))
        console.print(f"[{GHOST}]✔ Kept {name}[/] in [bold]{path}[/]. [dim]{left} file(s) still unmerged.[/]")
    return fin


def show_final(git: Git, fin: dict) -> None:
    if fin["action"] == "commit":
        parents = ", ".join(p[:7] for p in fin.get("parents", []))
        kind = "merge commit" if len(fin.get("parents", [])) > 1 else "commit"
        body = Group(
            Align.center(Text("✔  Merge complete", style=f"bold {GHOST}")),
            Align.center(Text(f"standard Git {kind} {fin['commit'][:10]}  ·  parents {parents}", style="dim")),
            Text(""),
            Syntax(fin.get("message", ""), "text", theme="ansi_dark", word_wrap=True, background_color="default"),
        )
        console.print(Panel(body, border_style="green", box=box.DOUBLE, padding=(1, 2), title="Resolved", title_align="left"))
    else:
        console.print(Panel(f"[{GHOST}]✔ Done:[/] ran [bold]git {fin['action'].replace('-', ' --')}[/]; HEAD is {(fin.get('commit') or '')[:10]}",
                            border_style="green", box=box.ROUNDED))


# =====================================================================================
# 4. Entry point
# =====================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py", description="GitTrace: resolve Git merge conflicts in your terminal with local AI provenance.")
    p.add_argument("target", nargs="?", help="local repository path (default: current directory) or a remote URL to clone")
    p.add_argument("--merge", nargs=2, metavar=("TARGET", "INCOMING"), help="start `git merge INCOMING` into TARGET first")
    p.add_argument("--no-ai", action="store_true", help="skip the local Gemma analysis")
    p.add_argument("--model", default=os.environ.get("GITTRACE_OLLAMA_MODEL", DEFAULT_MODEL), help="Ollama model (default: %(default)s)")
    p.add_argument("--ollama-host", default=os.environ.get("GITTRACE_OLLAMA_HOST", DEFAULT_HOST), help="local Ollama URL (default: %(default)s)")
    p.add_argument("--keep", action="store_true", help="keep the temporary clone even if nothing was committed")
    p.add_argument("--report", action="store_true", help="show the first conflict and exit without prompting")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interactive = not args.report
    client = None
    if not args.no_ai:
        try:
            client = OllamaClient(args.ollama_host, args.model)
        except NonLocalEndpointError as exc:
            console.print(f"[red]✖ {exc}[/]")
            return 2

    try:
        git, temp_dir = open_repo(args.target)
    except RepoLinkError as exc:
        console.print(f"[red]✖ {exc}[/]")
        return 2

    committed = False
    code = 0
    try:
        report = Analyzer(git, Store(git), lineage=False).analyze()
        if not report.files and report.context.operation is Operation.NONE:
            if not maybe_start_merge(git, args.merge, interactive):
                console.print(Panel(f"[{GHOST}]✔[/] No merge conflicts in [bold]{git.toplevel}[/]. Nothing to resolve.",
                                    border_style="bright_black", box=box.ROUNDED))
                return 0
        fin = resolve_loop(git, client, interactive)
        if fin and fin.get("committed"):
            committed = True
            show_final(git, fin)
        elif interactive and fin is None:
            code = 1
    except Abort:
        console.print(Panel("Aborted. The repository was left exactly as it was: nothing written or committed for the current conflict.\n"
                            "[dim]To undo the whole merge:  git merge --abort[/]", border_style="red", box=box.ROUNDED))
        code = 130
    finally:
        if temp_dir is not None:
            if committed or args.keep:
                console.print(f"[dim]Clone kept at[/] [bold]{git.toplevel}[/] [dim](push or inspect it from there)[/]")
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                console.print("[dim]Temporary clone removed.[/]")
    return code


if __name__ == "__main__":
    sys.exit(main())
