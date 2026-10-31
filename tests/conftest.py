from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gittrace.git import Git
from gittrace.store import Store


class Repo:
    """A throwaway Git repository for integration tests."""

    def __init__(self, path: Path):
        self.path = path
        self.git_cmd("init", "-q", "-b", "development")
        self.git_cmd("config", "user.email", "dev@example.com")
        self.git_cmd("config", "user.name", "Dev")
        self.git_cmd("config", "commit.gpgsign", "false")
        self.git_cmd("config", "core.autocrlf", "false")
        self.git_cmd("config", "merge.conflictStyle", "merge")
        self.git = Git(path)
        self.store = Store(self.git)

    def git_cmd(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_EDITOR": "true", "LC_ALL": "C"}
        return subprocess.run(["git", *args], cwd=self.path, capture_output=True, text=True, check=check, env=env)

    def write(self, name: str, content: str) -> None:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")

    def read(self, name: str) -> str:
        return (self.path / name).read_text(encoding="utf-8")

    def commit(self, message: str, **files: str) -> str:
        for name, content in files.items():
            self.write(name.replace("__", "/"), content)
        self.git_cmd("add", "-A")
        self.git_cmd("commit", "-q", "-m", message)
        return self.head()

    def head(self) -> str:
        return self.git_cmd("rev-parse", "HEAD").stdout.strip()

    def checkout(self, branch: str, create: bool = False) -> None:
        self.git_cmd("checkout", "-q", *(["-b"] if create else []), branch)


BASE = "def price(x):\n    return x * 1.0\n\n\ndef tax(x):\n    return x * 0.1\n"


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    r = Repo(tmp_path / "repo")
    return r


@pytest.fixture
def base_repo(tmp_path: Path) -> Repo:
    """development/uat/production branches sharing one initial commit."""
    path = tmp_path / "repo"
    path.mkdir()
    r = Repo(path)
    r.commit("initial", **{"app.py": BASE})
    # Give each long-lived branch its own history, so cherry-picks create distinct commits
    # (picking onto an identical parent in the same second would recreate the same hash).
    for env in ("uat", "production"):
        r.checkout(env, create=True)
        r.commit(f"{env} config", **{f"{env}.cfg": f"env={env}\n"})
    r.checkout("development")
    r.commit("development config", **{"development.cfg": "env=development\n"})
    return r
