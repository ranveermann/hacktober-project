"""Conflict detection: operation context, index stages, conflict hunks, blame.

Nothing in this module writes to the working tree or the index.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .git import Git, GitError


class Operation(str, Enum):
    MERGE = "merge"
    CHERRY_PICK = "cherry-pick"
    REVERT = "revert"
    REBASE = "rebase"
    AM = "am"
    UNKNOWN = "unknown"  # e.g. `git stash pop`, `git checkout -m`, apply --3way
    NONE = "none"


@dataclass
class Side:
    """One side of a conflict, described semantically for the current operation."""

    key: str  # "current" | "incoming" | "base"
    stage: int  # index stage that holds this side's blob
    label: str  # human description, e.g. "current branch (HEAD = main)"
    rev: str | None  # commit to blame / inspect for this side's content
    note: str = ""


@dataclass
class OperationContext:
    operation: Operation
    head: str | None
    current: Side
    incoming: Side
    base: Side
    subject_commit: str | None = None  # commit being merged/picked/reverted/replayed
    notes: list[str] = field(default_factory=list)


@dataclass
class StageEntry:
    mode: str
    sha: str


@dataclass
class ConflictHunk:
    index: int  # 1-based position within the file
    start_line: int  # 1-based line of the <<<<<<< marker in the working file
    end_line: int  # line of the >>>>>>> marker
    current_lines: list[str]
    incoming_lines: list[str]
    base_lines: list[str] | None  # only with diff3/zdiff3 conflict style
    current_marker: str
    incoming_marker: str


@dataclass
class ConflictFile:
    path: str
    stages: dict[int, StageEntry]
    kind: str
    binary: bool
    worktree_exists: bool
    hunks: list[ConflictHunk]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- context
def _read_text(path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _short(git: Git, rev: str | None) -> str:
    return rev[:10] if rev else "?"


def detect_operation(git: Git) -> OperationContext:
    """Work out which Git operation produced the conflict and what stages 2/3 mean.

    Stage 2 is always "the version Git had checked out" and stage 3 "the version being
    brought in", but what that means for a human differs: during a rebase, stage 2 is
    the *upstream* and stage 3 is *your own* commit being replayed.
    """
    gd = git.git_dir
    head = git.rev_parse("HEAD")
    branch = git.current_branch()
    head_desc = f"HEAD = {branch}" if branch else f"detached HEAD {_short(git, head)}"
    notes: list[str] = []

    rebase_merge = gd / "rebase-merge"
    rebase_apply = gd / "rebase-apply"
    rebase_head = (git.pseudo_ref("REBASE_HEAD") or [None])[0]

    if rebase_merge.is_dir() or (rebase_apply.is_dir() and not (rebase_apply / "applying").exists()):
        state = rebase_merge if rebase_merge.is_dir() else rebase_apply
        onto = _read_text(state / "onto") or None
        head_name = _read_text(state / "head-name").replace("refs/heads/", "") or "detached"
        replay = rebase_head or (git.pseudo_ref("MERGE_HEAD") or [None])[0]
        notes.append(
            "Rebase: stage 2 ('ours') is the UPSTREAM you are rebasing onto plus commits already "
            "replayed; stage 3 ('theirs') is YOUR commit being replayed. Git's ours/theirs are swapped "
            "relative to a merge."
        )
        parent = None
        if replay:
            info = git.commit_info(replay)
            parent = info.parents[0] if info and info.parents else None
        return OperationContext(
            operation=Operation.REBASE,
            head=head,
            current=Side("current", 2, f"upstream being rebased onto (onto {_short(git, onto)}, {head_desc})", head),
            incoming=Side("incoming", 3, f"your commit being replayed from '{head_name}' ({_short(git, replay)})", replay),
            base=Side("base", 1, f"parent of the replayed commit ({_short(git, parent)})", parent),
            subject_commit=replay,
            notes=notes,
        )

    if rebase_apply.is_dir() and (rebase_apply / "applying").exists():
        notes.append("git am: stage 3 is the patch being applied; it may not correspond to any local commit.")
        return OperationContext(
            operation=Operation.AM,
            head=head,
            current=Side("current", 2, f"current branch ({head_desc})", head),
            incoming=Side("incoming", 3, "mailbox patch being applied", None),
            base=Side("base", 1, "pre-image recorded in the patch (blob ids)", None),
            notes=notes,
        )

    picked = (git.pseudo_ref("CHERRY_PICK_HEAD") or [None])[0]
    if picked:
        info = git.commit_info(picked)
        parent = info.parents[0] if info and info.parents else None
        if info and len(info.parents) > 1:
            notes.append("The cherry-picked commit is a merge; the base depends on the -m mainline chosen.")
        notes.append(
            "Cherry-pick: the base is the PARENT of the picked commit, not the merge-base of the two "
            "branches. Only the picked commit's own diff is being transferred."
        )
        return OperationContext(
            operation=Operation.CHERRY_PICK,
            head=head,
            current=Side("current", 2, f"target branch ({head_desc})", head),
            incoming=Side("incoming", 3, f"cherry-picked commit {_short(git, picked)}" + (f" \"{info.subject}\"" if info else ""), picked),
            base=Side("base", 1, f"parent of the picked commit ({_short(git, parent)})", parent),
            subject_commit=picked,
            notes=notes,
        )

    reverted = (git.pseudo_ref("REVERT_HEAD") or [None])[0]
    if reverted:
        info = git.commit_info(reverted)
        parent = info.parents[0] if info and info.parents else None
        notes.append(
            "Revert: stage 3 is the content from BEFORE the reverted commit (its parent), and the base "
            "is the reverted commit itself. The incoming side removes a change rather than adding one."
        )
        return OperationContext(
            operation=Operation.REVERT,
            head=head,
            current=Side("current", 2, f"current branch ({head_desc})", head),
            incoming=Side("incoming", 3, f"state before reverted commit {_short(git, reverted)} (i.e. its parent)", parent),
            base=Side("base", 1, f"the reverted commit {_short(git, reverted)}", reverted),
            subject_commit=reverted,
            notes=notes,
        )

    merge_heads = git.pseudo_ref("MERGE_HEAD")
    if merge_heads:
        incoming_rev = merge_heads[0]
        if len(merge_heads) > 1:
            notes.append(f"Octopus merge with {len(merge_heads)} incoming heads; showing the first.")
        bases = git.merge_bases_all(head, incoming_rev) if head else []
        if len(bases) > 1:
            notes.append(
                f"Criss-cross history: {len(bases)} merge-bases exist, so Git merged them into a virtual "
                "base. Stage 1 may not match any single real commit."
            )
        merge_msg = _read_text(gd / "MERGE_MSG").splitlines()
        m = re.match(r"Merge (?:remote-tracking )?branch(?:es)? '([^']+)'", merge_msg[0]) if merge_msg else None
        if m:
            incoming_desc = f"incoming branch '{m.group(1)}'"
        elif merge_msg and merge_msg[0].startswith("Merge commit"):
            incoming_desc = "incoming commit"
        else:
            incoming_desc = merge_msg[0] if merge_msg else "incoming commit"
        base = bases[0] if bases else None
        return OperationContext(
            operation=Operation.MERGE,
            head=head,
            current=Side("current", 2, f"current branch ({head_desc})", head),
            incoming=Side("incoming", 3, f"{incoming_desc} ({_short(git, incoming_rev)})", incoming_rev),
            base=Side("base", 1, f"merge-base {_short(git, base)}" if base else "no common ancestor", base),
            subject_commit=incoming_rev,
            notes=notes,
        )

    # No operation in progress. Unmerged entries can still exist (stash pop, checkout -m, apply -3).
    op = Operation.UNKNOWN if list_unmerged(git) else Operation.NONE
    if op is Operation.UNKNOWN:
        notes.append(
            "Unmerged paths exist but no merge/cherry-pick/revert/rebase is in progress (e.g. after "
            "`git stash pop` or `git checkout -m`). Stage 3 cannot be tied to a commit reliably."
        )
    return OperationContext(
        operation=op,
        head=head,
        current=Side("current", 2, f"working copy ({head_desc})", head),
        incoming=Side("incoming", 3, "incoming change (origin unknown)", None),
        base=Side("base", 1, "base (origin unknown)", None),
        notes=notes,
    )


# ---------------------------------------------------------------------------- stages
def list_unmerged(git: Git) -> dict[str, dict[int, StageEntry]]:
    """Parse `git ls-files -u -z` into {path: {stage: StageEntry}}."""
    out = git.run_bytes("ls-files", "-u", "-z").decode("utf-8", "replace")
    result: dict[str, dict[int, StageEntry]] = {}
    for entry in out.split("\x00"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        mode, sha, stage = meta.split()
        result.setdefault(path, {})[int(stage)] = StageEntry(mode, sha)
    return result


def classify_stages(stages: set[int]) -> str:
    return {
        frozenset({1, 2, 3}): "both modified",
        frozenset({2, 3}): "both added",
        frozenset({1, 2}): "deleted on incoming side, modified on current side",
        frozenset({1, 3}): "deleted on current side, modified on incoming side",
        frozenset({1}): "deleted on both sides",
        frozenset({2}): "added on current side only (rename/directory conflict)",
        frozenset({3}): "added on incoming side only (rename/directory conflict)",
    }.get(frozenset(stages), f"unusual stage combination {sorted(stages)}")


# --------------------------------------------------------------------------- markers
def parse_conflict_markers(text: str) -> list[ConflictHunk]:
    """Parse merge / diff3 / zdiff3 style conflict blocks from working-tree text."""
    lines = text.splitlines()
    hunks: list[ConflictHunk] = []
    i = 0
    open_re = re.compile(r"^(<{7,})(?: (.*))?$")
    while i < len(lines):
        m = open_re.match(lines[i])
        if not m:
            i += 1
            continue
        size = len(m.group(1))
        start = i
        cur_marker = m.group(2) or ""
        current: list[str] = []
        base: list[str] | None = None
        incoming: list[str] = []
        section = "current"
        j = i + 1
        closed = False
        inc_marker = ""
        while j < len(lines):
            ln = lines[j]
            if ln.startswith("|" * size) and not ln.startswith("|" * (size + 1)) and section == "current":
                section, base = "base", []
            elif ln == "=" * size and section in ("current", "base"):
                section = "incoming"
            elif ln.startswith(">" * size) and not ln.startswith(">" * (size + 1)) and section == "incoming":
                inc_marker = ln[size:].strip()
                closed = True
                break
            else:
                {"current": current, "base": base, "incoming": incoming}[section].append(ln)  # type: ignore[union-attr]
            j += 1
        if not closed:
            break  # malformed / truncated block: stop rather than guess
        hunks.append(
            ConflictHunk(
                index=len(hunks) + 1,
                start_line=start + 1,
                end_line=j + 1,
                current_lines=current,
                incoming_lines=incoming,
                base_lines=base,
                current_marker=cur_marker,
                incoming_marker=inc_marker,
            )
        )
        i = j + 1
    return hunks


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8000]


def detect_conflicts(git: Git, only_path: str | None = None) -> list[ConflictFile]:
    files: list[ConflictFile] = []
    for path, stages in sorted(list_unmerged(git).items()):
        if only_path and path != only_path.replace("\\", "/"):
            continue
        notes: list[str] = []
        binary = any(_is_binary(git.cat_blob(e.sha)) for e in stages.values() if e.mode != "160000")
        if any(e.mode == "160000" for e in stages.values()):
            notes.append("Submodule (gitlink) conflict: compare the recorded submodule commits instead of text.")
        if len({e.mode for e in stages.values()}) > 1:
            notes.append("File mode differs between sides: " + ", ".join(f"stage {s}={e.mode}" for s, e in sorted(stages.items())))
        wt = git.toplevel / path
        exists = wt.is_file()
        hunks: list[ConflictHunk] = []
        if exists and not binary:
            text = wt.read_bytes().decode("utf-8", "replace")
            hunks = parse_conflict_markers(text)
            if not hunks and {2, 3} <= set(stages):
                notes.append("No conflict markers in the working file - it may already be hand-resolved but not yet `git add`-ed.")
        elif binary:
            notes.append("Binary content: Git cannot produce line-level conflict hunks.")
        files.append(
            ConflictFile(
                path=path,
                stages=stages,
                kind=classify_stages(set(stages)),
                binary=binary,
                worktree_exists=exists,
                hunks=hunks,
                notes=notes,
            )
        )
    return files


# ------------------------------------------------------------------------ attribution
@dataclass
class Attribution:
    rev: str | None
    method: str  # "blame" | "deleted" | "unlocated" | "unavailable"
    commits: dict[str, int]  # commit hash -> number of conflicting lines it introduced
    note: str = ""


_BLAME_HDR = re.compile(r"^([0-9a-f]{40}|[0-9a-f]{64}) \d+ \d+")


def _find_block(haystack: list[str], needle: list[str]) -> int | None:
    n = len(needle)
    norm_h = [h.rstrip("\r") for h in haystack]
    norm_n = [x.rstrip("\r") for x in needle]
    for i in range(0, len(norm_h) - n + 1):
        if norm_h[i : i + n] == norm_n:
            return i
    return None


def attribute_lines(git: Git, rev: str | None, path: str, lines: list[str]) -> Attribution:
    """Find which commits introduced `lines` as they exist in `path` at `rev` (via git blame)."""
    if not rev:
        return Attribution(None, "unavailable", {}, "no commit is associated with this side")
    if not lines:
        return Attribution(rev, "deleted", {}, "this side removes the lines in this region")
    content = git.show_file(rev, path)
    if content is None:
        return Attribution(rev, "unavailable", {}, f"{path} does not exist at {rev[:10]} (renamed or added elsewhere?)")
    start = _find_block(content.decode("utf-8", "replace").splitlines(), lines)
    if start is None:
        return Attribution(rev, "unlocated", {}, "conflict text not found verbatim at that commit (edited after conflict?)")
    try:
        out = git.run("blame", "--porcelain", "-L", f"{start + 1},{start + len(lines)}", rev, "--", path)
    except GitError as exc:
        return Attribution(rev, "unavailable", {}, f"git blame failed: {exc.stderr.strip()}")
    counts: Counter[str] = Counter()
    for ln in out.splitlines():
        m = _BLAME_HDR.match(ln)
        if m:
            counts[m.group(1)] += 1
    return Attribution(rev, "blame", dict(counts.most_common()))


def commits_touching(git: Git, since: str | None, until: str | None, path: str, limit: int = 15) -> list[str]:
    """Commits on `until` (not on `since`) that changed `path`, newest first."""
    if not until:
        return []
    rng = f"{since}..{until}" if since else until
    out = git.run("log", f"-n{limit}", "--format=%H", rng, "--", path, check=False)
    return out.split()
