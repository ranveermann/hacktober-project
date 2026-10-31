"""Evaluate testing and deployment provenance for a commit.

Key rule: a test result certifies a *snapshot* (tree). It applies to a commit only if
that commit is the tested commit or has an identical tree. Results on related commits
(backports, equivalent patches, similar changes) are shown as *related evidence* and are
never promoted to the commit's own status.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .git import Git
from .provenance import ProvenanceIndex, Relation, Relationship
from .store import ENVIRONMENTS, Store, TestRecord


@dataclass
class RelatedTest:
    relationship: Relationship
    record: TestRecord


@dataclass
class TestStatus:
    __test__ = False

    commit: str
    tree: str
    exact: list[TestRecord] = field(default_factory=list)  # recorded against this commit
    same_snapshot: list[TestRecord] = field(default_factory=list)  # other commit, identical tree
    related: list[RelatedTest] = field(default_factory=list)  # evidence only
    by_environment: dict[str, str] = field(default_factory=dict)  # env -> passed/failed/unknown/untested
    related_by_environment: dict[str, str] = field(default_factory=dict)


def _latest_per_env(records: list[TestRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for env in ENVIRONMENTS:
        recs = sorted((r for r in records if r.environment == env), key=lambda r: r.timestamp)
        if recs:
            # failure in any suite at the latest timestamp wins over pass
            latest_ts = recs[-1].timestamp
            latest = [r.result for r in recs if r.timestamp == latest_ts]
            out[env] = "failed" if "failed" in latest else latest[-1]
    return out


def test_status(git: Git, store: Store, index: ProvenanceIndex | None, commit: str, with_related: bool = True) -> TestStatus:
    full = git.rev_parse(commit)
    if not full:
        raise ValueError(f"unknown commit {commit}")
    tree = git.tree_of(full) or ""
    st = TestStatus(commit=full, tree=tree)
    for rec in store.tests():
        if rec.commit == full:
            st.exact.append(rec)
        elif rec.tree == tree:
            st.same_snapshot.append(rec)
    own = st.exact + st.same_snapshot
    summary = _latest_per_env(own)
    st.by_environment = {env: summary.get(env, "untested") for env in ENVIRONMENTS}

    if with_related and index is not None and store.tests():
        tested = {r.commit for r in store.tests()} - {full}
        own_ids = {r.id for r in own}
        for target in tested:
            if not index.ensure(target):
                continue
            rel = index.classify(full, target)
            if rel.relation in (Relation.UNKNOWN, Relation.EXACT):
                continue
            for rec in store.tests_for(commit=target):
                if rec.id not in own_ids:
                    st.related.append(RelatedTest(rel, rec))
        rel_summary = _latest_per_env([rt.record for rt in st.related])
        st.related_by_environment = rel_summary
    return st


# --------------------------------------------------------------------- deployments
@dataclass
class DeployFinding:
    environment: str
    state: str  # "deployed" | "present" | "equivalent-present" | "backport-present" | "absent" | "unconfigured"
    detail: str
    strength: str  # "recorded" | "exact" | "equivalent" | "referenced" | "none"


@dataclass
class DeployStatus:
    commit: str
    findings: list[DeployFinding]

    def reached(self, env: str) -> DeployFinding | None:
        for f in self.findings:
            if f.environment == env and f.state not in ("absent", "unconfigured"):
                return f
        return None


def deploy_status(git: Git, store: Store, index: ProvenanceIndex | None, commit: str) -> DeployStatus:
    full = git.rev_parse(commit)
    if not full:
        raise ValueError(f"unknown commit {commit}")
    tree = git.tree_of(full)
    findings: list[DeployFinding] = []
    branches = store.env_branches
    deploys = store.deployments()
    lineage: list[Relationship] | None = None

    for env in ENVIRONMENTS:
        # 1. explicit deployment records (strongest evidence)
        recs = [d for d in deploys if d.environment == env]
        exact = [d for d in recs if d.commit == full or d.tree == tree]
        if exact:
            d = sorted(exact, key=lambda d: d.timestamp)[-1]
            findings.append(DeployFinding(env, "deployed", f"deployment recorded {d.timestamp}" + (f" ({d.reference})" if d.reference else ""), "recorded"))
            continue
        contained = [d for d in recs if git.is_ancestor(full, d.commit)]
        if contained:
            d = sorted(contained, key=lambda d: d.timestamp)[-1]
            findings.append(DeployFinding(env, "deployed", f"included in deployed commit {d.commit[:10]} ({d.timestamp})", "recorded"))
            continue

        branch = branches.get(env)
        if not branch:
            findings.append(DeployFinding(env, "unconfigured", "no branch mapped (gittrace env-branch set ...)", "none"))
            continue
        if not git.branch_exists(branch):
            findings.append(DeployFinding(env, "unconfigured", f"mapped branch '{branch}' does not exist", "none"))
            continue
        # 2. ancestry: exact commit merged into the environment branch
        if git.is_ancestor(full, branch):
            findings.append(DeployFinding(env, "present", f"commit is reachable from '{branch}'", "exact"))
            continue
        # 3. git cherry: equivalent patch already on the environment branch
        info = git.commit_info(full)
        limit = info.parents[0] if info and info.parents else None
        rows = index.cherry(branch, full, limit) if index else []
        if any(sign == "-" and c == full for sign, c in rows):
            findings.append(DeployFinding(env, "equivalent-present", f"an equivalent patch (same patch-id) is on '{branch}' (git cherry)", "equivalent"))
            continue
        # 4. an explicitly referenced (possibly modified) backport reachable from the branch
        if index is not None:
            if lineage is None:
                lineage = index.lineage(full)
            hit = next((r for r in lineage if r.relation is Relation.REFERENCED and git.is_ancestor(r.target, branch)), None)
            if hit:
                mod = " - MODIFIED copy" if hit.modified else ""
                findings.append(DeployFinding(env, "backport-present", f"referenced backport {hit.target[:10]} is on '{branch}'{mod}", "referenced"))
                continue
        findings.append(DeployFinding(env, "absent", f"not found on '{branch}'", "none"))
    return DeployStatus(full, findings)
