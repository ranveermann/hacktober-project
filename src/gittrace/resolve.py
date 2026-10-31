"""Conflict resolution engine: explicit, user-driven, standard Git output.

Flow for one click in the UI (or one CLI call):
  1. write the chosen side's lines into the conflicting file (other hunks untouched);
  2. `git add` the file once no conflict markers remain;
  3. when no unmerged paths remain, finish the operation the standard way:
     merge / cherry-pick / revert -> `git commit` (a normal merge commit has 2 parents),
     rebase -> `git rebase --continue`, am -> `git am --continue`.
The commit message explains the provenance of every chosen side, plus
machine-readable `GitTrace-Resolution:` trailers.

All Git calls go through subprocess (gittrace.git.Git).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone

from .conflicts import Operation, detect_operation, list_unmerged, parse_conflict_markers
from .git import Git
from .store import Store

CHOICES = ("current", "incoming", "base", "both", "both-incoming-first")
SIDE_NAMES = {"current": "Side A", "incoming": "Side B", "base": "base", "both": "both sides (A then B)",
              "both-incoming-first": "both sides (B then A)"}


class ResolveError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_hunk(git: Git, store: Store, path: str, hunk_no: int, choice: str, stage: bool = False,
                 provenance: str | None = None) -> dict:
    if choice not in CHOICES:
        raise ResolveError(f"choice must be one of {', '.join(CHOICES)}")
    path = path.replace("\\", "/")
    if path not in list_unmerged(git):
        raise ResolveError(f"{path} is not an unmerged path")
    full = git.toplevel / path
    text = full.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"
    trailing_eol = text.endswith(("\n", "\r\n"))
    hunks = parse_conflict_markers(text)
    if not 1 <= hunk_no <= len(hunks):
        raise ResolveError(f"{path} has {len(hunks)} conflict hunk(s); hunk {hunk_no} is out of range")
    h = hunks[hunk_no - 1]
    if choice == "current":
        replacement = h.current_lines
    elif choice == "incoming":
        replacement = h.incoming_lines
    elif choice == "base":
        if h.base_lines is None:
            raise ResolveError("base text is not in the conflict markers; enable it with `git config merge.conflictStyle zdiff3` and re-run `git checkout --conflict=zdiff3 -- <path>`")
        replacement = h.base_lines
    elif choice == "both":
        replacement = h.current_lines + h.incoming_lines
    else:
        replacement = h.incoming_lines + h.current_lines

    lines = text.splitlines()
    new_lines = lines[: h.start_line - 1] + replacement + lines[h.end_line :]
    new_text = eol.join(new_lines) + (eol if trailing_eol and new_lines else "")
    _atomic_write_bytes(full, new_text.encode("utf-8"))

    ctx = detect_operation(git)
    hunk_hash = hashlib.sha1("\n".join(h.current_lines + ["="] + h.incoming_lines).encode()).hexdigest()[:12]
    entry = {
        "path": path,
        "hunk": hunk_no,
        "line": h.start_line,
        "choice": choice,
        "operation": ctx.operation.value,
        "head": ctx.head,
        "incoming_rev": ctx.incoming.rev,
        "hunk_text_hash": hunk_hash,
        "provenance": provenance,
        "timestamp": _now(),
    }
    store.append_journal(entry)
    remaining = len(parse_conflict_markers(new_text))
    staged = False
    if stage and remaining == 0:
        git.run("add", "--", path)
        staged = True
    return {**entry, "remaining_hunks": remaining, "staged": staged}


def resolve_whole_file(git: Git, store: Store, path: str, choice: str, provenance: str | None = None) -> dict:
    """Resolve a structural/binary conflict by taking one side's whole file (or its deletion)."""
    stage_no = {"current": 2, "incoming": 3}.get(choice)
    if stage_no is None:
        raise ResolveError("whole-file resolution supports 'current' or 'incoming'")
    path = path.replace("\\", "/")
    stages = list_unmerged(git).get(path)
    if not stages:
        raise ResolveError(f"{path} is not an unmerged path")
    full = git.toplevel / path
    if stage_no in stages:
        data = git.cat_blob(stages[stage_no].sha)
        full.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(full, data, new=not full.exists())
        git.run("add", "--", path)
        action = "kept"
    else:  # the chosen side deleted the file
        git.run("rm", "-q", "--cached", "--", path)
        if full.exists():
            full.unlink()
        action = "deleted"
    ctx = detect_operation(git)
    store.append_journal({
        "path": path, "hunk": 0, "line": 0, "choice": choice, "operation": ctx.operation.value,
        "head": ctx.head, "incoming_rev": ctx.incoming.rev, "hunk_text_hash": f"file:{path}",
        "provenance": provenance, "timestamp": _now(), "file_action": action,
    })
    return {"path": path, "choice": choice, "action": action, "staged": True, "remaining_hunks": 0}


def build_commit_message(git: Git, store: Store, message: str | None = None, trailers: bool = True) -> str:
    msg = message
    if msg is None:
        merge_msg = git.git_dir / "MERGE_MSG"
        if merge_msg.is_file():
            msg = "\n".join(
                ln for ln in merge_msg.read_text(encoding="utf-8", errors="replace").splitlines() if not ln.startswith("#")
            ).strip()
        if not msg:
            raise ResolveError("no commit message: pass one explicitly")
    journal = store.journal()
    explained = [e for e in journal if e.get("provenance")]
    if explained:
        body = ["", "Conflict resolution provenance (GitTrace):"]
        for e in explained:
            where = f"{e['path']}" + (f" hunk {e['hunk']} (line {e.get('line', '?')})" if e.get("hunk") else " (whole file)")
            prov = e["provenance"].splitlines()
            body.append(f"- {where}: {prov[0]}")
            body.extend(f"  {ln}" for ln in prov[1:])
        msg = msg.rstrip() + "\n" + "\n".join(body)
    if trailers and journal:
        lines = [f"GitTrace-Resolution: {e['path']}#{e['hunk']}={e['choice']}" for e in journal]
        msg = msg.rstrip() + "\n\n" + "\n".join(lines)
    return msg


def finish(git: Git, store: Store, message: str | None = None, trailers: bool = True) -> dict:
    """Complete the in-progress operation with a standard Git command, if nothing is left unmerged."""
    ctx = detect_operation(git)
    remaining = sorted(list_unmerged(git))
    if remaining:
        return {"action": "pending", "committed": False, "remaining": remaining}
    if ctx.operation is Operation.REBASE:
        git.run("-c", "core.editor=true", "rebase", "--continue")
        store.clear_journal()
        return {"action": "rebase-continue", "committed": True, "commit": git.rev_parse("HEAD"), "remaining": []}
    if ctx.operation is Operation.AM:
        git.run("am", "--continue")
        store.clear_journal()
        return {"action": "am-continue", "committed": True, "commit": git.rev_parse("HEAD"), "remaining": []}
    if ctx.operation in (Operation.NONE, Operation.UNKNOWN):
        return {"action": "staged", "committed": False, "remaining": [],
                "note": "no merge/cherry-pick/revert in progress - changes are staged; commit them yourself"}
    msg = build_commit_message(git, store, message, trailers)
    fd, tmp = tempfile.mkstemp(prefix="gittrace-msg-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(msg + "\n")
        git.run("commit", "--cleanup=strip", "-F", tmp)
    finally:
        os.unlink(tmp)
    store.clear_journal()
    head = git.rev_parse("HEAD") or ""
    parents = git.run("rev-list", "--parents", "-n1", head).split()[1:]
    return {"action": "commit", "committed": True, "commit": head, "parents": parents, "message": msg, "remaining": []}


def commit(git: Git, store: Store, message: str | None = None, trailers: bool = True) -> str:
    """CLI helper: like finish(), but strict (raises instead of returning a pending state)."""
    ctx = detect_operation(git)
    if list_unmerged(git):
        raise ResolveError("unmerged paths remain; resolve them and `git add` before committing")
    if ctx.operation in (Operation.REBASE, Operation.AM):
        raise ResolveError(f"a {ctx.operation.value} is in progress: run `git {ctx.operation.value} --continue` instead (Git creates the commit)")
    if ctx.operation in (Operation.NONE, Operation.UNKNOWN) and message is None:
        raise ResolveError("no merge/cherry-pick/revert in progress: pass -m to commit staged changes")
    msg = build_commit_message(git, store, message, trailers)
    fd, tmp = tempfile.mkstemp(prefix="gittrace-msg-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(msg + "\n")
        git.run("commit", "--cleanup=strip", "-F", tmp)
    finally:
        os.unlink(tmp)
    store.clear_journal()
    return git.rev_parse("HEAD") or ""


def resolve_and_finish(git: Git, store: Store, path: str, hunk_no: int, choice: str,
                       provenance: str | None = None, whole_file: bool = False) -> dict:
    """One UI click: write lines -> stage -> commit (once everything is resolved)."""
    if whole_file:
        res = resolve_whole_file(git, store, path, choice, provenance)
    else:
        res = resolve_hunk(git, store, path, hunk_no, choice, stage=True, provenance=provenance)
    done = finish(git, store) if res["remaining_hunks"] == 0 else {"action": "pending", "committed": False, "remaining": sorted(list_unmerged(git))}
    return {"resolution": res, "finish": done}


def _atomic_write_bytes(path, data: bytes, new: bool = False) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".gittrace-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if not new:
            try:
                os.chmod(tmp, path.stat().st_mode)
            except OSError:
                pass
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
