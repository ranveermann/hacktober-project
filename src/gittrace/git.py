"""Thin, cross-platform wrapper around the Git CLI.

All Git access in GitTrace goes through :class:`Git`. Commands are run with an
argument list (never a shell string), so paths with spaces work identically on
Windows, Linux and macOS. Output is captured as bytes and decoded explicitly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HEX_RE = re.compile(r"^[0-9a-f]{7,64}$")

# %x1e separates records, %x00 separates fields. %B is the raw message body.
LOG_FIELDS = "%H%x00%P%x00%T%x00%an%x00%ae%x00%aI%x00%B"
LOG_FORMAT = "%x1e" + LOG_FIELDS


class GitError(RuntimeError):
    """A Git command exited with a non-zero status."""

    def __init__(self, args: list[str], returncode: int, stderr: str):
        self.git_args = args
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"git {' '.join(args)} failed ({returncode}): {stderr.strip()}")


class NotARepositoryError(RuntimeError):
    """The working directory is not inside a Git repository."""


@dataclass
class CommitInfo:
    hash: str
    parents: list[str]
    tree: str
    author_name: str
    author_email: str
    author_date: str
    message: str
    files: set[str] = field(default_factory=set)

    @property
    def short(self) -> str:
        return self.hash[:10]

    @property
    def subject(self) -> str:
        return self.message.strip().splitlines()[0] if self.message.strip() else ""


def parse_log_record(rec: str, with_files: bool = False) -> CommitInfo | None:
    rec = rec.lstrip("\n")
    if not rec.strip():
        return None
    parts = rec.split("\x00", 7 if with_files else 6)
    if len(parts) < 7:
        return None
    files: set[str] = set()
    if with_files and len(parts) == 8:
        files = {ln for ln in parts[7].splitlines() if ln.strip()}
    return CommitInfo(
        hash=parts[0].strip(),
        parents=parts[1].split(),
        tree=parts[2],
        author_name=parts[3],
        author_email=parts[4],
        author_date=parts[5],
        message=parts[6],
        files=files,
    )


class Git:
    """Run Git commands against one repository."""

    def __init__(self, cwd: str | os.PathLike | None = None, executable: str | None = None):
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.exe = executable or shutil.which("git") or "git"
        env = dict(os.environ)
        env.update(
            {
                "LC_ALL": "C",
                "LANG": "C",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_PAGER": "cat",
                "PAGER": "cat",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        self._env = env
        self._cache: dict[str, object] = {}

    # ------------------------------------------------------------------ runners
    def run_bytes(self, *args: str, input: bytes | None = None, check: bool = True) -> bytes:
        cmd = [self.exe, "-c", "core.quotepath=off", "-c", "color.ui=never", *args]
        try:
            proc = subprocess.run(cmd, cwd=self.cwd, input=input, capture_output=True, env=self._env)
        except FileNotFoundError as exc:  # git not installed
            raise RuntimeError("Git executable not found. Install Git and make sure it is on PATH.") from exc
        if check and proc.returncode != 0:
            raise GitError(list(args), proc.returncode, proc.stderr.decode("utf-8", "replace"))
        return proc.stdout

    def run(self, *args: str, input: bytes | None = None, check: bool = True) -> str:
        return self.run_bytes(*args, input=input, check=check).decode("utf-8", "replace")

    def ok(self, *args: str) -> bool:
        cmd = [self.exe, "-c", "core.quotepath=off", *args]
        proc = subprocess.run(cmd, cwd=self.cwd, capture_output=True, env=self._env)
        return proc.returncode == 0

    # --------------------------------------------------------------- locations
    def ensure_repo(self) -> None:
        if not self.ok("rev-parse", "--git-dir"):
            raise NotARepositoryError(f"{self.cwd} is not inside a Git repository")

    @property
    def toplevel(self) -> Path:
        if "top" not in self._cache:
            self.ensure_repo()
            self._cache["top"] = Path(self.run("rev-parse", "--show-toplevel").strip())
        return self._cache["top"]  # type: ignore[return-value]

    @property
    def git_dir(self) -> Path:
        """Per-worktree git dir (holds MERGE_HEAD, rebase state, ...)."""
        if "gitdir" not in self._cache:
            self.ensure_repo()
            self._cache["gitdir"] = Path(self.run("rev-parse", "--absolute-git-dir").strip())
        return self._cache["gitdir"]  # type: ignore[return-value]

    @property
    def common_dir(self) -> Path:
        """Shared git dir (same for all worktrees) - where GitTrace keeps its data."""
        if "common" not in self._cache:
            self.ensure_repo()
            raw = self.run("rev-parse", "--git-common-dir").strip()
            p = Path(raw)
            if not p.is_absolute():
                p = (self.cwd / p).resolve()
            self._cache["common"] = p
        return self._cache["common"]  # type: ignore[return-value]

    # ------------------------------------------------------------------ objects
    def rev_parse(self, rev: str) -> str | None:
        out = self.run("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False).strip()
        return out or None

    def pseudo_ref(self, name: str) -> list[str]:
        """Read a pseudo-ref file such as MERGE_HEAD (may hold several hashes)."""
        path = self.git_dir / name
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [ln.split()[0] for ln in text.splitlines() if ln.strip() and HEX_RE.match(ln.split()[0])]

    def commit_info(self, rev: str, with_files: bool = False) -> CommitInfo | None:
        full = self.rev_parse(rev)
        if not full:
            return None
        args = ["log", "-1", "--no-walk", f"--format={LOG_FORMAT}" + ("%x00" if with_files else "")]
        if with_files:
            args += ["--name-only", "-m", "--first-parent"]
        out = self.run(*args, full)
        for rec in out.split("\x1e"):
            info = parse_log_record(rec, with_files=with_files)
            if info:
                return info
        return None

    def tree_of(self, commit: str) -> str | None:
        out = self.run("rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}", check=False).strip()
        return out or None

    def merge_base(self, a: str, b: str) -> str | None:
        out = self.run("merge-base", a, b, check=False).strip()
        return out or None

    def merge_bases_all(self, a: str, b: str) -> list[str]:
        return self.run("merge-base", "--all", a, b, check=False).split()

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self.ok("merge-base", "--is-ancestor", ancestor, descendant)

    def show_file(self, rev: str, path: str) -> bytes | None:
        try:
            return self.run_bytes("show", f"{rev}:{path}")
        except GitError:
            return None

    def cat_blob(self, sha: str) -> bytes:
        return self.run_bytes("cat-file", "blob", sha)

    def branch_exists(self, name: str) -> bool:
        return self.rev_parse(name) is not None

    def current_branch(self) -> str | None:
        out = self.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).strip()
        return out or None
