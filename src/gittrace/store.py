"""Testing and deployment provenance storage.

Records live in `<git-common-dir>/gittrace/records.json`: inside `.git`, so they are
never committed by accident, shared by all worktrees, and untouched by checkouts.
Every record stores both the commit hash and its tree hash; the tree hash identifies
the exact tested snapshot (two commits with the same tree have byte-identical content).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .git import Git

SCHEMA_VERSION = 1
ENVIRONMENTS = ("development", "uat", "production")
RESULTS = ("passed", "failed", "unknown")

_ENV_ALIASES = {"dev": "development", "develop": "development", "development": "development",
                "uat": "uat", "staging": "uat", "test": "uat",
                "prod": "production", "production": "production", "live": "production"}
_RESULT_ALIASES = {"pass": "passed", "passed": "passed", "ok": "passed", "success": "passed",
                   "fail": "failed", "failed": "failed", "failure": "failed",
                   "unknown": "unknown", "skipped": "unknown"}


class ValidationError(ValueError):
    pass


def normalize_env(value: str) -> str:
    env = _ENV_ALIASES.get(value.strip().lower())
    if not env:
        raise ValidationError(f"unknown environment '{value}' (expected one of: {', '.join(ENVIRONMENTS)})")
    return env


def normalize_result(value: str) -> str:
    res = _RESULT_ALIASES.get(value.strip().lower())
    if not res:
        raise ValidationError(f"unknown result '{value}' (expected one of: {', '.join(RESULTS)})")
    return res


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TestRecord:
    __test__ = False  # keep pytest from collecting this class

    commit: str
    tree: str
    environment: str
    result: str
    suite: str | None = None
    run_id: str | None = None
    timestamp: str = field(default_factory=_now)
    source: str = "manual"
    metadata: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class DeployRecord:
    commit: str
    tree: str
    environment: str
    reference: str | None = None  # release tag, deployment id, pipeline URL...
    timestamp: str = field(default_factory=_now)
    source: str = "manual"
    metadata: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class Store:
    def __init__(self, git: Git):
        self.git = git
        self.dir = git.common_dir / "gittrace"
        self.path = self.dir / "records.json"
        self.journal_path = self.dir / "resolutions.json"
        self._data: dict | None = None

    # ------------------------------------------------------------------ io
    @property
    def data(self) -> dict:
        if self._data is None:
            if self.path.is_file():
                with open(self.path, encoding="utf-8") as fh:
                    self._data = json.load(fh)
            else:
                self._data = {"schema": SCHEMA_VERSION, "tests": [], "deployments": [], "config": {"env_branches": {}}}
            self._data.setdefault("config", {}).setdefault("env_branches", {})
        return self._data

    def save(self) -> None:
        _atomic_write_json(self.path, self.data)

    # -------------------------------------------------------------- helpers
    def _resolve(self, commit: str) -> tuple[str, str]:
        full = self.git.rev_parse(commit)
        if not full:
            raise ValidationError(f"'{commit}' is not a commit in this repository")
        tree = self.git.tree_of(full)
        assert tree
        return full, tree

    # ---------------------------------------------------------------- tests
    def add_test(self, commit: str, environment: str, result: str, suite: str | None = None,
                 run_id: str | None = None, source: str = "manual", metadata: dict[str, str] | None = None,
                 timestamp: str | None = None) -> TestRecord:
        full, tree = self._resolve(commit)
        rec = TestRecord(full, tree, normalize_env(environment), normalize_result(result), suite, run_id,
                         source=source, metadata=metadata or {})
        if timestamp:
            rec.timestamp = timestamp
        self.data["tests"].append(asdict(rec))
        self.save()
        return rec

    def tests(self) -> list[TestRecord]:
        return [TestRecord(**r) for r in self.data["tests"]]

    def tests_for(self, commit: str | None = None, tree: str | None = None) -> list[TestRecord]:
        return [r for r in self.tests() if (commit and r.commit == commit) or (tree and r.tree == tree)]

    # ---------------------------------------------------------- deployments
    def add_deploy(self, commit: str, environment: str, reference: str | None = None, source: str = "manual",
                   metadata: dict[str, str] | None = None) -> DeployRecord:
        full, tree = self._resolve(commit)
        rec = DeployRecord(full, tree, normalize_env(environment), reference, source=source, metadata=metadata or {})
        self.data["deployments"].append(asdict(rec))
        self.save()
        return rec

    def deployments(self) -> list[DeployRecord]:
        return [DeployRecord(**r) for r in self.data["deployments"]]

    # --------------------------------------------------------------- config
    @property
    def env_branches(self) -> dict[str, str]:
        return dict(self.data["config"]["env_branches"])

    def set_env_branch(self, environment: str, branch: str) -> None:
        self.data["config"]["env_branches"][normalize_env(environment)] = branch
        self.save()

    def unset_env_branch(self, environment: str) -> None:
        self.data["config"]["env_branches"].pop(normalize_env(environment), None)
        self.save()

    # ------------------------------------------------------- import / export
    def export(self, dest: Path) -> int:
        _atomic_write_json(dest, self.data)
        return len(self.data["tests"]) + len(self.data["deployments"])

    def import_(self, src: Path) -> tuple[int, int]:
        with open(src, encoding="utf-8") as fh:
            incoming = json.load(fh)
        added = skipped = 0
        for key in ("tests", "deployments"):
            known = {r["id"] for r in self.data[key]}
            for r in incoming.get(key, []):
                if r.get("id") in known:
                    skipped += 1
                    continue
                self.data[key].append(r)
                added += 1
        self.save()
        return added, skipped

    # ----------------------------------------------------- resolution journal
    def journal(self) -> list[dict]:
        if self.journal_path.is_file():
            with open(self.journal_path, encoding="utf-8") as fh:
                return json.load(fh)
        return []

    def append_journal(self, entry: dict) -> None:
        entries = [e for e in self.journal() if not (e["path"] == entry["path"] and e["hunk_text_hash"] == entry["hunk_text_hash"])]
        entries.append(entry)
        _atomic_write_json(self.journal_path, entries)

    def clear_journal(self) -> None:
        if self.journal_path.exists():
            self.journal_path.unlink()


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gittrace-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
