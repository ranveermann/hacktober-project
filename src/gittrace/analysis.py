"""Combine conflict detection, provenance, testing and deployment into one report.

The advisory notes describe evidence; they never choose a side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conflicts import (
    Attribution,
    ConflictFile,
    ConflictHunk,
    Operation,
    OperationContext,
    attribute_lines,
    commits_touching,
    detect_conflicts,
    detect_operation,
)
from .ai import AIAssessment, OllamaClient, SideEvidence
from .git import Git
from .provenance import ProvenanceIndex, Relation, Relationship, apply_ai
from .status import DeployStatus, TestStatus, deploy_status, test_status
from .store import Store


@dataclass
class CommitContext:
    commit: str
    subject: str
    author: str
    date: str
    lines: int  # conflicting lines this commit introduced on this side (0 for file-level)
    tests: TestStatus
    deploy: DeployStatus
    lineage: list[Relationship]
    reverts: list[str] = field(default_factory=list)


@dataclass
class SideReport:
    key: str
    label: str
    rev: str | None
    lines: list[str]
    attribution: Attribution
    commits: list[CommitContext]


@dataclass
class HunkReport:
    hunk: ConflictHunk
    current: SideReport
    incoming: SideReport
    cross_side: list[Relationship]
    advisories: list[str]


@dataclass
class FileReport:
    file: ConflictFile
    current_file_commits: list[str]
    incoming_file_commits: list[str]
    hunks: list[HunkReport]
    advisories: list[str]


@dataclass
class Report:
    context: OperationContext
    files: list[FileReport]
    side_tip_tests: dict[str, TestStatus]
    notes: list[str]


class Analyzer:
    def __init__(self, git: Git, store: Store | None = None, history_limit: int = 5000, lineage: bool = True,
                 max_commits_per_side: int = 4, ai_client: OllamaClient | None = None):
        self.git = git
        self.ai_client = ai_client
        self.store = store or Store(git)
        self.lineage_enabled = lineage
        self.index = ProvenanceIndex(git, history_limit, ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD"])
        self.max_commits = max_commits_per_side
        self._ctx_cache: dict[str, CommitContext] = {}
        self._current_path: str | None = None

    # ----------------------------------------------------------------- commits
    def commit_context(self, commit: str, lines: int = 0) -> CommitContext:
        if commit not in self._ctx_cache:
            info = self.index.ensure(commit)
            assert info, commit
            lin = self.index.lineage(commit) if self.lineage_enabled else []
            self._ctx_cache[commit] = CommitContext(
                commit=info.hash,
                subject=info.subject,
                author=info.author_name,
                date=info.author_date,
                lines=0,
                tests=test_status(self.git, self.store, self.index if self.lineage_enabled else None, info.hash),
                deploy=deploy_status(self.git, self.store, self.index if self.lineage_enabled else None, info.hash),
                lineage=lin,
                reverts=self.index.reverts(info.hash),
            )
        base = self._ctx_cache[commit]
        return CommitContext(**{**base.__dict__, "lines": lines})

    def _side(self, key: str, label: str, rev: str | None, path: str, lines: list[str], fallback: list[str]) -> SideReport:
        attr = attribute_lines(self.git, rev, path, lines)
        commits: list[CommitContext] = []
        if attr.commits:
            for c, n in list(attr.commits.items())[: self.max_commits]:
                if set(c) == {"0"}:
                    continue  # uncommitted lines
                commits.append(self.commit_context(c, n))
        else:
            for c in fallback[: min(2, self.max_commits)]:
                commits.append(self.commit_context(c, 0))
        return SideReport(key, label, rev, lines, attr, commits)

    # --------------------------------------------------------------- analysis
    def analyze(self, only_path: str | None = None) -> Report:
        ctx = detect_operation(self.git)
        files = detect_conflicts(self.git, only_path)
        side_tips: dict[str, TestStatus] = {}
        for side in (ctx.current, ctx.incoming):
            if side.rev:
                side_tips[side.key] = test_status(self.git, self.store, self.index if self.lineage_enabled else None, side.rev)
        reports = [self._file_report(ctx, f, side_tips) for f in files]
        notes = list(ctx.notes)
        if self.lineage_enabled:
            notes.append(
                "Lineage uses explicit references, git patch-id and heuristics. Patch IDs only match unmodified "
                "diffs; 'probably related' is a similarity estimate, not proof."
            )
        return Report(ctx, reports, side_tips, notes)

    def _file_report(self, ctx: OperationContext, f: ConflictFile, tips: dict[str, TestStatus]) -> FileReport:
        cur_rev, inc_rev = ctx.current.rev, ctx.incoming.rev
        if ctx.operation in (Operation.MERGE, Operation.REBASE) and cur_rev and inc_rev:
            mb = self.git.merge_base(cur_rev, inc_rev)
            cur_commits = commits_touching(self.git, mb, cur_rev, f.path)
            inc_commits = commits_touching(self.git, mb, inc_rev, f.path) if ctx.operation is Operation.MERGE else [inc_rev]
            if ctx.operation is Operation.REBASE and ctx.subject_commit:
                inc_commits = [ctx.subject_commit]
        elif ctx.operation in (Operation.CHERRY_PICK, Operation.REVERT) and ctx.subject_commit and cur_rev:
            mb = self.git.merge_base(cur_rev, ctx.subject_commit)
            cur_commits = commits_touching(self.git, mb, cur_rev, f.path)
            inc_commits = [ctx.subject_commit]
        else:
            cur_commits = commits_touching(self.git, None, cur_rev, f.path, limit=5)
            inc_commits = []

        hunks = []
        self._current_path = f.path
        for h in f.hunks:
            cur = self._side("current", ctx.current.label, cur_rev, f.path, h.current_lines, cur_commits)
            inc = self._side("incoming", ctx.incoming.label, inc_rev, f.path, h.incoming_lines, inc_commits)
            cross = self._cross(cur, inc)
            hunks.append(HunkReport(h, cur, inc, cross, self._advise(ctx, cur, inc, cross, tips)))
        file_adv: list[str] = []
        if not f.hunks:
            file_adv.extend(f.notes)
            if f.kind.startswith("deleted"):
                file_adv.append(f"Structural conflict ({f.kind}). Decide whether the file should exist, e.g. `git rm` or `git add`.")
        return FileReport(f, cur_commits, inc_commits, hunks, file_adv)

    def _cross(self, cur: SideReport, inc: SideReport) -> list[Relationship]:
        if not self.lineage_enabled:
            return []
        rels = []
        judge = self.ai_judge(path=self._current_path) if self.ai_client else None
        for a in cur.commits[:2]:
            for b in inc.commits[:2]:
                r = self.index.classify(a.commit, b.commit, ai_judge=judge)
                if r.relation is not Relation.UNKNOWN:
                    rels.append(r)
        return rels

    # --------------------------------------------------------------- advisory
    def _advise(self, ctx: OperationContext, cur: SideReport, inc: SideReport, cross: list[Relationship],
                tips: dict[str, TestStatus]) -> list[str]:
        out: list[str] = []
        names = {"current": "Current", "incoming": "Incoming"}
        for r in cross:
            if r.relation is Relation.EXACT:
                out.append(f"Both sides' lines come from the same commit {r.source[:10]}; the conflict is caused by surrounding edits.")
            elif r.relation is Relation.REFERENCED:
                mod = "a MODIFIED copy" if r.modified else "an unmodified copy"
                out.append(f"Same change lineage on both sides: {r.target[:10]} and {r.source[:10]} are linked by explicit reference ({mod}). Compare the differences rather than treating them as unrelated work.")
            elif r.relation is Relation.EQUIVALENT:
                out.append(f"{r.source[:10]} and {r.target[:10]} are equivalent patches (identical patch-id) applied on both sides; the conflict comes from context, not from the change itself.")
            elif r.relation is Relation.PROBABLE:
                out.append(f"{r.source[:10]} and {r.target[:10]} look related ({r.confidence} confidence, score {r.score:.2f}) - heuristic, verify manually.")

        for side in (cur, inc):
            label = names[side.key]
            if side.attribution.method == "deleted":
                out.append(f"{label} side deletes this region.")
            for cc in side.commits:
                env_pass = [e for e, s in cc.tests.by_environment.items() if s == "passed"]
                env_fail = [e for e, s in cc.tests.by_environment.items() if s == "failed"]
                if env_fail:
                    out.append(f"WARNING: {label} lines from {cc.commit[:10]} - its exact snapshot FAILED in {', '.join(env_fail)}.")
                if env_pass:
                    out.append(f"{label} lines from {cc.commit[:10]} - exact snapshot passed in {', '.join(env_pass)}.")
                elif not env_fail:
                    strong = [rt for rt in cc.tests.related if rt.relationship.confidence in ("high", "medium")]
                    rel_pass = {rt.record.environment for rt in strong if rt.record.result == "passed"}
                    if rel_pass:
                        srcs = sorted({f"{rt.record.commit[:10]} ({rt.relationship.relation.value}"
                                       + (", modified" if rt.relationship.modified else "") + ")"
                                       for rt in strong if rt.record.result == "passed"})
                        out.append(
                            f"{label} lines from {cc.commit[:10]} have NO test record of their own. Related commit(s) "
                            f"{', '.join(srcs)} passed in {', '.join(sorted(rel_pass))} - this does not certify {cc.commit[:10]}."
                        )
                    else:
                        out.append(f"{label} lines from {cc.commit[:10]} have no testing record.")
                for env in ("production", "uat"):
                    hit = cc.deploy.reached(env)
                    if hit:
                        out.append(f"{label} lines from {cc.commit[:10]} already reached {env}: {hit.detail}.")
                        break
                for rv in cc.reverts:
                    out.append(f"{cc.commit[:10]} is a revert of {rv[:10]}.")
        tip = tips.get("incoming")
        if tip and ctx.operation is Operation.MERGE:
            passed = [e for e, s in tip.by_environment.items() if s == "passed"]
            if passed:
                out.append(f"The incoming branch tip {tip.commit[:10]} passed in {', '.join(passed)} as a whole snapshot. Your merge result will be a NEW snapshot that is not yet tested.")
        if ctx.operation is Operation.REVERT:
            out.append("Incoming side restores older code (revert). Keeping 'incoming' removes the reverted change in this region.")
        # de-duplicate, keep order
        seen: set[str] = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    # ------------------------------------------------------------- local AI
    def commit_diff(self, commit: str, path: str | None = None) -> str:
        args = ["show", "--format=", "--no-color", "--no-ext-diff", "-U3", commit]
        if path:
            args += ["--", path]
        return self.git.run(*args, check=False)

    def side_evidence(self, name: str, side: SideReport, path: str) -> SideEvidence:
        """Commit message + diff + conflicting lines for one side (what the model sees)."""
        commit = side.commits[0].commit if side.commits else side.rev
        info = self.index.ensure(commit) if commit else None
        return SideEvidence(
            name=name,
            label=side.label,
            commit=info.hash if info else None,
            subject=info.subject if info else "",
            message=info.message if info else "",
            diff=self.commit_diff(info.hash, path) if info else "",
            conflict_lines=side.lines,
        )

    def ai_judge(self, path: str | None = None):
        """Return a judge(a, b) callable for ProvenanceIndex.classify, backed by the local model."""
        client = self.ai_client
        assert client is not None

        def judge(a: str, b: str) -> AIAssessment:
            ev = []
            for name, c in (("Side A", a), ("Side B", b)):
                info = self.index.ensure(c)
                ev.append(SideEvidence(name, "commit", c, info.subject if info else "", info.message if info else "",
                                       self.commit_diff(c, path), []))
            return client.compare(ev[0], ev[1])

        return judge

    def compare_hunk(self, client: OllamaClient, hr: HunkReport, path: str, use_cache: bool = True) -> tuple[AIAssessment, Relationship | None]:
        """Ask the local model about Side A vs Side B of one hunk; merge into the lineage verdict."""
        a = self.side_evidence("Side A", hr.current, path)
        b = self.side_evidence("Side B", hr.incoming, path)
        assessment = client.compare(a, b, use_cache=use_cache)
        rel = None
        if a.commit and b.commit and a.commit != b.commit:
            rel = apply_ai(self.index.classify(a.commit, b.commit), assessment)
        return assessment, rel


# ------------------------------------------------------------------ badges
ENV_SHORT = {"development": "DEV", "uat": "UAT", "production": "PROD"}
ENV_ORDER = ("production", "uat", "development")  # highest first


@dataclass
class Badge:
    text: str  # e.g. "Tested in PROD", "Unverified DEV change"
    level: str  # success | warning | error | info
    details: list[str] = field(default_factory=list)


def side_badge(side: SideReport) -> Badge:
    """One-line provenance verdict for a conflict side, based on its originating commit."""
    if not side.commits:
        if side.attribution.method == "deleted":
            return Badge("Removes these lines (no originating commit)", "info", [side.attribution.note])
        return Badge("Unknown origin", "info", [side.attribution.note or "no commit could be attributed"])
    cc = side.commits[0]
    details = [f"origin {cc.commit[:10]} \"{cc.subject}\" by {cc.author}"]
    if len(side.commits) > 1:
        details.append(f"+{len(side.commits) - 1} more contributing commit(s)")
    st = cc.tests.by_environment
    failed = [e for e in ENV_ORDER if st.get(e) == "failed"]
    passed = [e for e in ENV_ORDER if st.get(e) == "passed"]
    reached = next((e for e in ENV_ORDER if cc.deploy.reached(e)), None)
    if reached:
        details.append(f"deployment: {cc.deploy.reached(reached).detail}")
    for r in cc.lineage[:2]:
        details.append(f"{r.relation.value} of {r.target[:10]} ({r.confidence})")
    if failed:
        return Badge(f"FAILED in {ENV_SHORT[failed[0]]}", "error", details)
    if passed:
        extra = f" (also {', '.join(ENV_SHORT[e] for e in passed[1:])})" if len(passed) > 1 else ""
        return Badge(f"Tested in {ENV_SHORT[passed[0]]}{extra}", "success", details)
    strong = [rt for rt in cc.tests.related if rt.relationship.confidence in ("high", "medium") and rt.record.result == "passed"]
    if strong:
        env = next(e for e in ENV_ORDER if any(rt.record.environment == e for rt in strong))
        rt = next(rt for rt in strong if rt.record.environment == env)
        phrase = {Relation.REFERENCED: "its referenced original", Relation.EQUIVALENT: "an equivalent patch",
                  Relation.PROBABLE: "a similar change"}.get(rt.relationship.relation, "a related commit")
        mod = " (this copy was modified)" if rt.relationship.modified else ""
        details.append(f"related {rt.record.commit[:10]} passed {ENV_SHORT[env]} - does not certify this snapshot")
        where = f" {ENV_SHORT[reached]}" if reached else ""
        return Badge(f"Untested{where} change · {phrase} passed {ENV_SHORT[env]}{mod}", "warning", details)
    if reached:
        return Badge(f"Unverified {ENV_SHORT[reached]} change", "warning", details)
    return Badge("Unverified change (no test or deployment record)", "warning", details)


def describe_choice(choice_name: str, side: SideReport, badge: Badge, ai: AIAssessment | None = None) -> str:
    """Multi-line provenance text that goes into the merge commit message."""
    lines = [f"kept {choice_name} ({side.label})"]
    if side.commits:
        cc = side.commits[0]
        lines.append(f"Origin: {cc.commit[:10]} \"{cc.subject}\" by {cc.author}")
        for r in cc.lineage[:2]:
            mod = " (modified)" if r.modified else ""
            lines.append(f"Lineage: {r.relation.value}{mod} of {r.target[:10]}, confidence {r.confidence}")
    lines.append(f"Provenance: {badge.text}")
    if ai is not None and ai.ok:
        expl = " ".join(ai.explanation.split())
        expl = expl if len(expl) <= 220 else expl[:217] + "..."
        lines.append(f"Local AI ({ai.model}): {ai.verdict} ({ai.confidence:.0%}) - {expl}")
    return "\n".join(lines)
