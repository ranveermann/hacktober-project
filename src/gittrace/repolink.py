"""Turn what the user pastes (a GitHub link or a local path) into a local Git working copy.

GitTrace works on a LOCAL clone, because that is where the merge state lives (.git/MERGE_HEAD,
index stages, ...). Resolution order for a GitHub link:
  1. the repository the app was started in (or GITTRACE_REPO), if its remote matches the link;
  2. an existing GitTrace workspace clone (~/.gittrace/repos/<owner>/<repo>);
  3. a fresh `git clone` into that workspace (standard Git; uses your own credentials).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .git import Git, GitError

WORKSPACE = Path(os.environ.get("GITTRACE_WORKSPACE", Path.home() / ".gittrace" / "repos"))

_URL_PATTERNS = [
    re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?/?$", re.I),
    re.compile(r"^git@github\.com:([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", re.I),
    re.compile(r"^ssh://git@github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", re.I),
]


class RepoLinkError(ValueError):
    pass


@dataclass
class RepoRef:
    owner: str
    name: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}".lower()

    @property
    def https_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


@dataclass
class Located:
    path: Path
    how: str  # "current" | "workspace" | "cloned" | "path"
    ref: RepoRef | None = None


def parse_github_url(text: str) -> RepoRef | None:
    text = (text or "").strip()
    for pat in _URL_PATTERNS:
        m = pat.match(text)
        if m:
            return RepoRef(m.group(1), m.group(2))
    return None


def remote_slugs(git: Git) -> set[str]:
    out = git.run("remote", "-v", check=False)
    slugs = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            ref = parse_github_url(parts[1])
            if ref:
                slugs.add(ref.slug)
    return slugs


def _is_repo(path: Path) -> Git | None:
    if not path.is_dir():
        return None
    g = Git(path)
    return g if g.ok("rev-parse", "--git-dir") else None


def locate(text: str, start_dirs: list[Path] | None = None, allow_clone: bool = True) -> Located:
    """Resolve a GitHub link or local path to a local repository."""
    text = (text or "").strip().strip('"').strip("'")
    if not text:
        raise RepoLinkError("Paste a GitHub repository link (or a local path).")

    ref = parse_github_url(text)
    if ref is None:
        p = Path(os.path.expanduser(text))
        g = _is_repo(p)
        if g:
            return Located(g.toplevel, "path")
        if "github" in text.lower() or "://" in text:
            raise RepoLinkError("That doesn't look like a GitHub repository link. Expected https://github.com/<owner>/<repo>.")
        raise RepoLinkError(f"'{text}' is neither a GitHub link nor a local Git repository.")

    for d in start_dirs or []:
        g = _is_repo(Path(d))
        if g and ref.slug in remote_slugs(g):
            return Located(g.toplevel, "current", ref)

    dest = WORKSPACE / ref.owner / ref.name
    g = _is_repo(dest)
    if g:
        return Located(g.toplevel, "workspace", ref)
    if not allow_clone:
        raise RepoLinkError(f"No local clone of {ref.slug} found. Start GitTrace inside your clone of it.")
    clone(ref, dest)
    return Located(dest, "cloned", ref)


def clone(ref: RepoRef, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(["git", "clone", "--quiet", ref.https_url, str(dest)], capture_output=True, text=True,
                          env=env, timeout=timeout)
    if proc.returncode != 0:
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        raise RepoLinkError(f"Could not clone {ref.slug}: {msg}. For private repos, clone it yourself and start GitTrace inside it.")


# ----------------------------------------------------------------- starting a merge
def branches(git: Git) -> tuple[list[str], list[str]]:
    """(local branches, remote-tracking branches such as origin/uat)."""
    local = [b for b in git.run("for-each-ref", "--format=%(refname:short)", "refs/heads", check=False).split() if b]
    remote = [b for b in git.run("for-each-ref", "--format=%(refname:short)", "refs/remotes", check=False).split()
              if b and "/" in b and not b.endswith("/HEAD")]  # origin/HEAD shortens to plain "origin"
    return local, remote


def is_clean(git: Git) -> bool:
    return git.run("status", "--porcelain", "--untracked-files=no", check=False).strip() == ""


def start_merge(git: Git, target: str, incoming: str) -> dict:
    """Check out `target` and run a standard `git merge incoming` (conflicts are expected)."""
    if not is_clean(git):
        raise RepoLinkError("Your working tree has uncommitted changes. Commit or stash them first.")
    if target == incoming:
        raise RepoLinkError("Pick two different branches.")
    current = git.current_branch()
    if current != target:
        try:
            git.run("switch", target)
        except GitError as exc:
            raise RepoLinkError(f"Could not switch to '{target}': {exc.stderr.strip()}") from exc
    proc = subprocess.run(["git", "merge", "--no-ff", "--no-edit", incoming], cwd=git.toplevel, capture_output=True,
                          text=True, env={**os.environ, "GIT_EDITOR": "true", "LC_ALL": "C"})
    conflicted = bool(git.run("ls-files", "-u", check=False).strip())
    if proc.returncode != 0 and not conflicted:
        raise RepoLinkError(f"git merge failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return {"conflicts": conflicted, "output": (proc.stdout + proc.stderr).strip()}
