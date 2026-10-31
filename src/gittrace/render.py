"""Output: Rich terminal rendering (if installed), plain text fallback, and JSON.

Rendering is kept separate from analysis so a web dashboard can consume the same
report objects (or the JSON form) later.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from enum import Enum
from typing import Any

from .analysis import CommitContext, Report, SideReport
from .provenance import Relationship
from .status import DeployStatus, TestStatus

STATUS_STYLE = {"passed": "green", "failed": "bold red", "unknown": "yellow", "untested": "dim"}


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = [to_jsonable(v) for v in obj]
        return sorted(items) if isinstance(obj, set) else items
    return obj


def dump_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=2)


class Out:
    """Minimal output facade over Rich or plain stdout."""

    def __init__(self, plain: bool = False, stream=None):
        self.stream = stream or sys.stdout
        self.rich = None
        if not plain:
            try:
                from rich.console import Console

                self.rich = Console(file=self.stream, highlight=False, soft_wrap=False)
            except ImportError:
                self.rich = None

    def _p(self, text: str = "") -> None:
        print(text, file=self.stream)

    def heading(self, text: str) -> None:
        if self.rich:
            self.rich.rule(f"[bold]{_esc(text)}")
        else:
            self._p()
            self._p(f"== {text} ==")

    def sub(self, text: str) -> None:
        if self.rich:
            self.rich.print(f"[bold cyan]{_esc(text)}[/]")
        else:
            self._p(f"-- {text}")

    def line(self, text: str = "", style: str | None = None) -> None:
        if self.rich:
            self.rich.print(f"[{style}]{_esc(text)}[/]" if style else _esc(text))
        else:
            self._p(text)

    def kv(self, pairs: list[tuple[str, str]]) -> None:
        width = max((len(k) for k, _ in pairs), default=0)
        for k, v in pairs:
            if self.rich:
                self.rich.print(f"  [bold]{_esc(k.ljust(width))}[/]  {_esc(v)}")
            else:
                self._p(f"  {k.ljust(width)}  {v}")

    def table(self, title: str | None, headers: list[str], rows: list[list[str]]) -> None:
        if self.rich:
            from rich.table import Table

            t = Table(title=title, show_lines=False, header_style="bold")
            for h in headers:
                t.add_column(h, overflow="fold")
            for r in rows:
                t.add_row(*[_style_cell(c) for c in r])
            self.rich.print(t)
        else:
            if title:
                self._p(title)
            widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
            self._p("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
            self._p("  " + "-+-".join("-" * w for w in widths))
            for r in rows:
                self._p("  " + " | ".join(c.ljust(w) for c, w in zip(r, widths)))

    def code(self, title: str, lines: list[str], style: str) -> None:
        if self.rich:
            from rich.panel import Panel
            from rich.text import Text

            body = Text("\n".join(lines) if lines else "(empty - lines removed on this side)")
            self.rich.print(Panel(body, title=_esc(title), title_align="left", border_style=style, expand=True))
        else:
            self._p(f"  +--- {title}")
            for ln in lines or ["(empty - lines removed on this side)"]:
                self._p(f"  | {ln}")
            self._p("  +---")


def _esc(text: str) -> str:
    return text.replace("[", r"\[")


def _style_cell(c: str) -> str:
    style = STATUS_STYLE.get(c.split(" ")[0])
    return f"[{style}]{_esc(c)}[/]" if style else _esc(c)


# ----------------------------------------------------------------------- reports
def render_report(out: Out, report: Report) -> None:
    ctx = report.context
    out.heading("GitTrace conflict report")
    out.kv([
        ("Operation", ctx.operation.value),
        ("Current  (stage 2)", ctx.current.label),
        ("Incoming (stage 3)", ctx.incoming.label),
        ("Base     (stage 1)", ctx.base.label),
        ("Conflicted files", str(len(report.files))),
    ])
    for n in report.notes:
        out.line(f"  note: {n}", "dim")
    if not report.files:
        out.line("")
        out.line("No unresolved conflicts found.", "green")
        return
    for key, st in report.side_tip_tests.items():
        out.line(f"  {key} tip {st.commit[:10]} testing: " + _env_summary(st), None)

    for fr in report.files:
        f = fr.file
        out.heading(f"{f.path}  [{f.kind}]")
        out.kv([(f"stage {s}", f"{e.sha[:10]} mode {e.mode}") for s, e in sorted(f.stages.items())])
        if fr.current_file_commits or fr.incoming_file_commits:
            out.kv([
                ("current-side commits", ", ".join(c[:10] for c in fr.current_file_commits[:6]) or "-"),
                ("incoming-side commits", ", ".join(c[:10] for c in fr.incoming_file_commits[:6]) or "-"),
            ])
        for a in fr.advisories:
            out.line(f"  ! {a}", "yellow")
        for hr in fr.hunks:
            h = hr.hunk
            out.sub(f"Hunk {h.index}/{len(fr.hunks)}  (lines {h.start_line}-{h.end_line})")
            out.code(f"CURRENT  - {hr.current.label}", h.current_lines, "blue")
            if h.base_lines is not None:
                out.code("BASE", h.base_lines, "white")
            out.code(f"INCOMING - {hr.incoming.label}", h.incoming_lines, "magenta")
            _render_side(out, hr.current)
            _render_side(out, hr.incoming)
            if hr.cross_side:
                out.line("  Cross-side lineage:", "bold")
                for r in hr.cross_side:
                    _render_rel(out, r, indent="    ")
            if hr.advisories:
                out.line("  Context for your decision (advisory - GitTrace does not pick a side):", "bold")
                for a in hr.advisories:
                    out.line(f"   * {a}", "red" if a.startswith("WARNING") else None)
            out.line(f"  resolve: gittrace resolve {f.path} --hunk {h.index} --take current|incoming|both", "dim")


def _env_summary(st: TestStatus) -> str:
    return ", ".join(f"{e}={s}" for e, s in st.by_environment.items())


def _render_side(out: Out, side: SideReport) -> None:
    a = side.attribution
    how = {"blame": "git blame", "deleted": "region deleted", "unlocated": "not located", "unavailable": "n/a"}[a.method]
    out.line(f"  {side.key.upper()} origin ({how}){': ' + a.note if a.note else ''}", "bold")
    if not side.commits:
        out.line("    no originating commit identified", "dim")
        return
    rows = []
    for cc in side.commits:
        rows.append(_commit_row(cc))
    out.table(None, ["commit", "lines", "subject", "dev", "uat", "prod", "deployment"], rows)
    for cc in side.commits:
        for r in cc.lineage[:4]:
            _render_rel(out, r, indent="    ")


def _commit_row(cc: CommitContext) -> list[str]:
    t = cc.tests.by_environment
    dep = [f"{f.environment}:{f.state}" for f in cc.deploy.findings if f.state not in ("absent", "unconfigured")]
    return [cc.commit[:10], str(cc.lines or "-"), cc.subject[:48], t["development"], t["uat"], t["production"], ", ".join(dep) or "-"]


def _render_rel(out: Out, r: Relationship, indent: str = "") -> None:
    score = f", score {r.score:.2f}" if r.score is not None and r.relation.name in ("PROBABLE", "REFERENCED") else ""
    out.line(f"{indent}{r.source[:10]} ~ {r.target[:10]}: {r.relation.value} (confidence {r.confidence}{score})")
    for ev in r.evidence:
        out.line(f"{indent}    - {ev}", "dim")


def render_trace(out: Out, info, refs, lineage: list[Relationship], tests: TestStatus, deploy: DeployStatus, reverts: list[str]) -> None:
    out.heading(f"Commit {info.hash[:12]}")
    out.kv([("subject", info.subject), ("author", f"{info.author_name} <{info.author_email}>"), ("date", info.author_date),
            ("tree", info.tree), ("parents", ", ".join(p[:10] for p in info.parents) or "(root)")])
    if refs:
        out.sub("Explicit references in message")
        for r in refs:
            out.line(f"  {r.kind}: {r.raw}" + ("" if r.resolved else "  (NOT found in this repository)"))
    for rv in reverts:
        out.line(f"  reverts {rv[:10]}")
    out.sub("Lineage")
    if not lineage:
        out.line("  unknown origin - no references, equivalent patches or similar changes found", "dim")
    for r in lineage:
        _render_rel(out, r, "  ")
    render_tests(out, tests)
    out.sub("Deployment")
    out.table(None, ["environment", "state", "evidence", "detail"], [[f.environment, f.state, f.strength, f.detail] for f in deploy.findings])


def render_tests(out: Out, st: TestStatus) -> None:
    out.sub("Testing provenance (exact snapshot)")
    out.table(None, ["environment", "status"], [[e, s] for e, s in st.by_environment.items()])
    rows = [[r.environment, r.result, r.suite or "-", r.run_id or "-", r.timestamp, "this commit", r.source] for r in st.exact]
    rows += [[r.environment, r.result, r.suite or "-", r.run_id or "-", r.timestamp, f"same tree via {r.commit[:10]}", r.source] for r in st.same_snapshot]
    if rows:
        out.table("Records", ["env", "result", "suite", "run", "timestamp", "applies via", "source"], rows)
    if st.related:
        out.sub("Related evidence (does NOT certify this commit)")
        out.table(None, ["env", "result", "tested commit", "relation", "confidence"],
                  [[rt.record.environment, rt.record.result, rt.record.commit[:10], rt.relationship.relation.value, rt.relationship.confidence] for rt in st.related])
