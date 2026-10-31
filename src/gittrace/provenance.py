"""Change-origin / lineage detection.

Relationship classes, strongest first:

* EXACT       - the very same commit object.
* REFERENCED  - one commit explicitly names the other ("cherry picked from commit ...",
                Backport-of: trailer, ...), possibly through a chain of references.
* EQUIVALENT  - `git patch-id --stable` is identical: the diffs are the same apart from
                whitespace and line numbers. Says nothing about surrounding code.
* PROBABLE    - heuristic: overlapping changed lines, files and subjects. Never proof.
* UNKNOWN     - no evidence found.

Patch IDs stop matching as soon as a backport is modified, even trivially. GitTrace
therefore never upgrades a similarity match to "equivalent", and always reports evidence.
"""

from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from .git import LOG_FORMAT, CommitInfo, Git, parse_log_record


class Relation(str, Enum):
    EXACT = "exact original commit"
    REFERENCED = "explicitly referenced backport"
    EQUIVALENT = "equivalent patch"
    PROBABLE = "probably related change"
    UNKNOWN = "unknown origin"


RELATION_RANK = {Relation.EXACT: 0, Relation.REFERENCED: 1, Relation.EQUIVALENT: 2, Relation.PROBABLE: 3, Relation.UNKNOWN: 4}


@dataclass
class Relationship:
    source: str  # commit we asked about
    target: str  # related commit
    relation: Relation
    confidence: str  # certain | high | medium | low | none
    score: float | None = None
    evidence: list[str] = field(default_factory=list)
    modified: bool | None = None  # for REFERENCED: True if patch differs from the referenced commit
    ai: Any = None  # AIAssessment from the local model, when one was consulted


@dataclass
class Reference:
    target: str  # full hash (or the unresolved abbreviation)
    kind: str  # e.g. "cherry-pick -x line", "Backport-of trailer"
    raw: str
    resolved: bool


# "(cherry picked from commit X)" is written by `git cherry-pick -x`.
_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\(cherry picked from commit ([0-9a-fA-F]{7,64})\)"), "cherry-pick -x line"),
    (re.compile(r"\(backported from commit ([0-9a-fA-F]{7,64})\)", re.I), "backport note"),
    (
        re.compile(
            r"^[ \t]*(Cherry-picked-from|Cherry-pick-of|Backport-of|Backported-from|Original-commit|Upstream-commit)[ \t]*:[ \t]*([0-9a-fA-F]{7,64})\b",
            re.I | re.M,
        ),
        "trailer",
    ),
]
_REVERT_RE = re.compile(r"This reverts commit ([0-9a-fA-F]{7,64})")
_SUBJECT_NOISE = re.compile(
    r"^\s*((\[[^\]]{1,30}\]|\((?:backport|cherry[- ]?pick)[^)]*\))\s*|(backport|cherry[- ]?pick(ed)?|hotfix|uat|prod|release)\s*[:/-]\s*)+",
    re.I,
)

PROBABLE_THRESHOLD = 0.45
MEDIUM_THRESHOLD = 0.70
MIN_LINE_OVERLAP = 0.30

AI_GRAY_ZONE = 0.20  # heuristic scores at or above this (but below PROBABLE) are sent to the AI too
AI_CONFIRM_MIN = 0.60
AI_REJECT_MIN = 0.70

PATCH_ID_CAVEAT = (
    "patch-id compares diff text only (ignoring whitespace and line numbers); it does not prove the "
    "surrounding code or runtime behaviour is the same, and any edit to the patch breaks the match"
)


def parse_references(message: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for pattern, kind in _REF_PATTERNS:
        for m in pattern.finditer(message):
            if kind == "trailer":
                refs.append((m.group(2).lower(), f"{m.group(1)} trailer"))
            else:
                refs.append((m.group(1).lower(), kind))
    return refs


def parse_reverts(message: str) -> list[str]:
    return [m.group(1).lower() for m in _REVERT_RE.finditer(message)]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class ProvenanceIndex:
    """Lazily built index over the repository's recent history (all refs)."""

    def __init__(self, git: Git, history_limit: int = 5000, extra_revs: list[str] | None = None):
        self.git = git
        self.history_limit = history_limit
        revs = ["--all", "HEAD"]
        for name in extra_revs or []:
            if git.rev_parse(name):
                revs.append(name)
        self._revs = revs
        self._commits: dict[str, CommitInfo] | None = None
        self._patch_ids: dict[str, str] | None = None
        self._by_patch: dict[str, set[str]] = {}
        self._refs: dict[str, list[Reference]] = {}
        self._reverts: dict[str, list[str]] = {}
        self._graph: dict[str, set[str]] = {}
        self._by_file: dict[str, list[str]] = {}
        self._sig_cache: dict[str, tuple[set[str], set[str]]] = {}

    # ---------------------------------------------------------------- building
    @property
    def commits(self) -> dict[str, CommitInfo]:
        if self._commits is None:
            self._build()
        return self._commits  # type: ignore[return-value]

    def _build(self) -> None:
        out = self.git.run(
            "log", "--no-merges", f"-n{self.history_limit}", f"--format={LOG_FORMAT}%x00", "--name-only", *self._revs, check=False
        )
        commits: dict[str, CommitInfo] = {}
        for rec in out.split("\x1e"):
            info = parse_log_record(rec, with_files=True)
            if info:
                commits[info.hash] = info
        self._commits = commits
        for info in commits.values():
            self._index_commit(info)

    def _index_commit(self, info: CommitInfo) -> None:
        for path in info.files:
            self._by_file.setdefault(path, []).append(info.hash)
        refs: list[Reference] = []
        for abbrev, kind in parse_references(info.message):
            full = self.resolve(abbrev)
            refs.append(Reference(full or abbrev, kind, abbrev, full is not None))
            if full and full != info.hash:
                self._graph.setdefault(info.hash, set()).add(full)
                self._graph.setdefault(full, set()).add(info.hash)
        self._refs[info.hash] = refs
        self._reverts[info.hash] = [self.resolve(r) or r for r in parse_reverts(info.message)]

    def resolve(self, abbrev: str) -> str | None:
        """Resolve a hash prefix or any revision expression (branch, HEAD~2, tag...)."""
        if not re.fullmatch(r"[0-9a-fA-F]{4,64}", abbrev):
            return self.git.rev_parse(abbrev)  # ref names are case-sensitive
        abbrev = abbrev.lower()
        if self._commits is not None:
            if abbrev in self._commits:
                return abbrev
            hits = [h for h in self._commits if h.startswith(abbrev)]
            if len(hits) == 1:
                return hits[0]
        return self.git.rev_parse(abbrev)

    def ensure(self, commit: str) -> CommitInfo | None:
        """Make sure a commit (possibly older than the history limit) is indexed."""
        full = self.resolve(commit)
        if not full:
            return None
        if full not in self.commits:
            info = self.git.commit_info(full, with_files=True)
            if not info:
                return None
            self.commits[full] = info
            self._index_commit(info)
            pid = self._patch_id_single(full)
            if pid:
                self.patch_ids[full] = pid
                self._by_patch.setdefault(pid, set()).add(full)
        return self.commits[full]

    @property
    def patch_ids(self) -> dict[str, str]:
        if self._patch_ids is None:
            _ = self.commits
            log = self.git.run_bytes(
                "log", "--no-merges", f"-n{self.history_limit}", "-p", "--no-ext-diff", "--no-color",
                "--format=commit %H", *self._revs, check=False,
            )
            self._patch_ids = self._parse_patch_ids(self.git.run("patch-id", "--stable", input=log, check=False))
            for c, pid in self._patch_ids.items():
                self._by_patch.setdefault(pid, set()).add(c)
        return self._patch_ids

    @staticmethod
    def _parse_patch_ids(out: str) -> dict[str, str]:
        ids: dict[str, str] = {}
        for ln in out.splitlines():
            parts = ln.split()
            if len(parts) == 2:
                ids[parts[1]] = parts[0]
        return ids

    def _patch_id_single(self, commit: str) -> str | None:
        diff = self.git.run_bytes("show", "--no-ext-diff", "--no-color", "--format=commit %H", commit, check=False)
        return self._parse_patch_ids(self.git.run("patch-id", "--stable", input=diff, check=False)).get(commit)

    def patch_id(self, commit: str) -> str | None:
        return self.patch_ids.get(commit)

    # -------------------------------------------------------------- lookups
    def references(self, commit: str) -> list[Reference]:
        self.ensure(commit)
        return self._refs.get(commit, [])

    def reverts(self, commit: str) -> list[str]:
        self.ensure(commit)
        return self._reverts.get(commit, [])

    def reference_path(self, a: str, b: str) -> list[str] | None:
        """Shortest chain of explicit references connecting a and b (inclusive), if any."""
        _ = self.commits
        if a not in self._graph or b not in self._graph:
            return None
        prev: dict[str, str | None] = {a: None}
        q = deque([a])
        while q:
            cur = q.popleft()
            if cur == b:
                path = [b]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])  # type: ignore[arg-type]
                return list(reversed(path))
            for nxt in self._graph.get(cur, ()):
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        return None

    def reference_component(self, commit: str) -> set[str]:
        _ = self.commits
        seen = {commit}
        q = deque([commit])
        while q:
            for nxt in self._graph.get(q.popleft(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return seen

    def _describe_link(self, x: str, y: str) -> str:
        for ref in self._refs.get(y, []):
            if ref.target == x:
                return f"{y[:10]} names {x[:10]} via {ref.kind} ('{ref.raw}')"
        for ref in self._refs.get(x, []):
            if ref.target == y:
                return f"{x[:10]} names {y[:10]} via {ref.kind} ('{ref.raw}')"
        return f"{x[:10]} <-> {y[:10]}"

    # ------------------------------------------------------------- similarity
    def _signature(self, commit: str) -> tuple[set[str], set[str]]:
        if commit not in self._sig_cache:
            out = self.git.run("show", "--format=", "--no-color", "--no-ext-diff", "-U0", "--no-renames", commit, check=False)
            changed: set[str] = set()
            for ln in out.splitlines():
                if ln.startswith(("+++", "---")):
                    continue
                if ln[:1] in "+-":
                    body = ln[1:].strip()
                    if len(body) > 2:
                        changed.add(ln[0] + body)
            info = self.commits.get(commit)
            self._sig_cache[commit] = (set(info.files) if info else set(), changed)
        return self._sig_cache[commit]

    @staticmethod
    def _fuzzy_overlap(xs: set[str], ys: set[str], cutoff: float = 0.8) -> float:
        """Share of lines in the larger set that have a near-identical partner in the other set."""
        if not xs or not ys:
            return 0.0
        small, large = (xs, ys) if len(xs) <= len(ys) else (ys, xs)
        pool = list(large)
        matched = 0
        for ln in small:
            if ln in large:
                matched += 1
                continue
            sm = difflib.SequenceMatcher(None, "", ln, autojunk=False)  # seq2 fixed, seq1 varies
            for cand in pool:
                sm.set_seq1(cand)
                if sm.real_quick_ratio() >= cutoff and sm.quick_ratio() >= cutoff and sm.ratio() >= cutoff:
                    matched += 1
                    break
        return matched / len(large)

    def similarity(self, a: str, b: str) -> tuple[float, float, list[str]]:
        """Return (score, changed-line overlap, evidence).

        Added lines dominate: two commits that delete the same old line but write
        different replacements are competing changes, not the same change.
        """
        fa, la = self._signature(a)
        fb, lb = self._signature(b)
        add_a = {x[1:] for x in la if x[0] == "+"}
        add_b = {x[1:] for x in lb if x[0] == "+"}
        rem_a = {x[1:] for x in la if x[0] == "-"}
        rem_b = {x[1:] for x in lb if x[0] == "-"}
        if add_a or add_b:
            line_sim = 0.75 * self._fuzzy_overlap(add_a, add_b) + 0.25 * _jaccard(rem_a, rem_b)
        else:  # pure deletions
            line_sim = _jaccard(rem_a, rem_b)
        file_sim = _jaccard(fa, fb)
        sa = _SUBJECT_NOISE.sub("", self.commits[a].subject).lower()
        sb = _SUBJECT_NOISE.sub("", self.commits[b].subject).lower()
        subj_sim = difflib.SequenceMatcher(None, sa, sb).ratio() if sa and sb else 0.0
        score = 0.6 * line_sim + 0.25 * file_sim + 0.15 * subj_sim
        evidence = [
            f"changed-line overlap {line_sim:.0%}",
            f"file overlap {file_sim:.0%}" + (f" ({', '.join(sorted(fa & fb)[:3])})" if fa & fb else ""),
            f"subject similarity {subj_sim:.0%}",
        ]
        ia, ib = self.commits[a], self.commits[b]
        if a in ib.parents or b in ia.parents:
            score *= 0.8
            evidence.append("direct parent/child: more likely a follow-up edit than the same change (score reduced)")
        return score, line_sim, evidence

    # ------------------------------------------------------------- classification
    def classify(self, a: str, b: str, ai_judge: Callable[[str, str], Any] | None = None) -> Relationship:
        """Classify the relationship between two commits.

        `ai_judge(a, b) -> AIAssessment` (optional) is consulted only where the
        deterministic evidence runs out: heuristic PROBABLE matches and gray-zone
        UNKNOWNs. It can confirm, upgrade or reject those, never the stronger classes.
        """
        rel = self._classify_deterministic(a, b)
        if ai_judge is None or rel.relation not in (Relation.PROBABLE, Relation.UNKNOWN):
            return rel
        if rel.relation is Relation.UNKNOWN and (rel.score or 0) < AI_GRAY_ZONE:
            return rel
        return apply_ai(rel, ai_judge(rel.source, rel.target))

    def _classify_deterministic(self, a: str, b: str) -> Relationship:
        ia, ib = self.ensure(a), self.ensure(b)
        if not ia or not ib:
            return Relationship(a, b, Relation.UNKNOWN, "none", evidence=["commit not found"])
        a, b = ia.hash, ib.hash
        if a == b:
            return Relationship(a, b, Relation.EXACT, "certain", 1.0, ["same commit object"])
        pa, pb = self.patch_id(a), self.patch_id(b)
        same_patch = bool(pa) and pa == pb
        path = self.reference_path(a, b)
        if path:
            evidence = [self._describe_link(x, y) for x, y in zip(path, path[1:])]
            if len(path) > 2:
                evidence.append(f"reference chain of {len(path) - 1} hops")
            if same_patch:
                evidence.append(f"patch-id also identical ({pa[:12]}): transferred without modification")
                return Relationship(a, b, Relation.REFERENCED, "high", 1.0, evidence, modified=False)
            score, _, sim_ev = self.similarity(a, b)
            evidence.append("patch-id differs: the change was MODIFIED during transfer (conflict fix, adaptation, or extra edits)")
            evidence.extend(sim_ev)
            return Relationship(a, b, Relation.REFERENCED, "high", round(score, 3), evidence, modified=True)
        if same_patch:
            return Relationship(
                a, b, Relation.EQUIVALENT, "high", 1.0,
                [f"git patch-id --stable identical ({pa[:12]})", "no explicit reference between the commits", PATCH_ID_CAVEAT],
            )
        score, line_sim, sim_ev = self.similarity(a, b)
        if score >= PROBABLE_THRESHOLD and line_sim >= MIN_LINE_OVERLAP:
            conf = "medium" if score >= MEDIUM_THRESHOLD else "low"
            return Relationship(
                a, b, Relation.PROBABLE, conf, round(score, 3),
                sim_ev + ["heuristic similarity only - not proof that the changes are the same"],
            )
        return Relationship(a, b, Relation.UNKNOWN, "none", round(score, 3), ["no reference, patch-id differs, similarity below threshold"])

    def lineage(self, commit: str, max_probable: int = 5, max_candidates: int = 400) -> list[Relationship]:
        # Deterministic only: the AI is consulted for specific pairs (e.g. Side A vs Side B),
        # not for hundreds of lineage candidates.
        """All commits related to `commit`, strongest first (excluding UNKNOWN)."""
        info = self.ensure(commit)
        if not info:
            return []
        c = info.hash
        strong: set[str] = (self.reference_component(c) | self._by_patch.get(self.patch_id(c) or "", set())) - {c}
        results = [self.classify(c, t) for t in strong]
        # similarity candidates: commits touching at least one of the same files
        cand: list[str] = []
        seen = set(strong) | {c}
        for path in sorted(info.files):
            for h in self._by_file.get(path, []):
                if h not in seen:
                    seen.add(h)
                    cand.append(h)
                if len(cand) >= max_candidates:
                    break
        probable = [r for r in (self.classify(c, t) for t in cand) if r.relation is Relation.PROBABLE]
        probable.sort(key=lambda r: -(r.score or 0))
        results.extend(probable[:max_probable])
        results.sort(key=lambda r: (RELATION_RANK[r.relation], -(r.score or 0)))
        return results

    def related_set(self, commit: str) -> dict[str, Relationship]:
        return {r.target: r for r in self.lineage(commit)}

    # ------------------------------------------------------------- git cherry
    def cherry(self, upstream: str, head: str = "HEAD", limit: str | None = None) -> list[tuple[str, str]]:
        """Wrap `git cherry -v`: list (sign, commit). '-' = an equivalent patch is already in upstream."""
        args = ["cherry", upstream, head] + ([limit] if limit else [])
        out = self.git.run(*args, check=False)
        rows = []
        for ln in out.splitlines():
            parts = ln.split()
            if len(parts) >= 2 and parts[0] in "+-":
                rows.append((parts[0], parts[1]))
        return rows


def apply_ai(rel: Relationship, assessment: Any) -> Relationship:
    """Combine a deterministic relationship with a local-AI assessment.

    * EXACT / REFERENCED / EQUIVALENT are evidence-backed and never changed; the AI
      opinion is only attached.
    * SAME_FIX (confidence >= 0.6) turns UNKNOWN into PROBABLE, or strengthens PROBABLE.
    * PARTIALLY_SAME keeps or creates a low-confidence PROBABLE.
    * DIFFERENT_CHANGES (confidence >= 0.7) rejects a heuristic PROBABLE match.
    * UNCLEAR or an error leaves the deterministic result unchanged.
    The AI can never produce EQUIVALENT: that still requires identical patch-ids.
    """
    if assessment is None:
        return rel
    model = getattr(assessment, "model", "local model")
    verdict = getattr(assessment, "verdict", "UNCLEAR")
    conf = float(getattr(assessment, "confidence", 0.0) or 0.0)
    note = f"{model} (local AI): {verdict} with confidence {conf:.0%}"
    out = replace(rel, evidence=list(rel.evidence), ai=assessment)
    if getattr(assessment, "error", None):
        out.evidence.append(f"{model} (local AI) unavailable: {assessment.error}")
        return out
    if rel.relation not in (Relation.PROBABLE, Relation.UNKNOWN):
        out.evidence.append(note + " - informational only; deterministic evidence takes precedence")
        return out
    if verdict == "SAME_FIX" and conf >= AI_CONFIRM_MIN:
        out.relation = Relation.PROBABLE
        out.confidence = "high" if conf >= 0.85 and rel.relation is Relation.PROBABLE else "medium"
        out.evidence = [e for e in out.evidence if not e.startswith("no reference, patch-id differs")]
        out.evidence.append(note + " - judged logically the same fix despite different bytes")
    elif verdict == "PARTIALLY_SAME":
        out.relation = Relation.PROBABLE
        out.confidence = "low"
        out.evidence.append(note + " - same intent, but behaviour differs in part")
    elif verdict == "DIFFERENT_CHANGES" and conf >= AI_REJECT_MIN:
        if rel.relation is Relation.PROBABLE:
            out.evidence.append(note + " - heuristic similarity REJECTED: the changes behave differently")
        else:
            out.evidence.append(note)
        out.relation = Relation.UNKNOWN
        out.confidence = "none"
    else:
        out.evidence.append(note + " - not decisive; heuristic result kept")
    out.evidence.append("AI assessments are advisory and can be wrong")
    return out
