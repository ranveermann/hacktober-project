"""Tests for GitHub-link resolution and starting merges (gittrace.repolink)."""

import subprocess
import sys
from pathlib import Path

import pytest

from gittrace import repolink
from gittrace.git import Git
from gittrace.repolink import RepoLinkError, locate, parse_github_url, start_merge


ROOT = Path(__file__).resolve().parents[1]


def make_demo(tmp_path, merge=True, name="demo"):
    path = tmp_path / name
    args = [sys.executable, str(ROOT / "demo" / "make_demo_repo.py"), str(path)] + ([] if merge else ["--no-merge"])
    subprocess.run(args, check=True, capture_output=True)
    return path


def git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------- repo links
@pytest.mark.parametrize("url,slug", [
    ("https://github.com/acme/billing", "acme/billing"),
    ("https://github.com/acme/billing.git", "acme/billing"),
    ("github.com/Acme/Billing-Service/tree/main/src", "acme/billing-service"),
    ("git@github.com:acme/billing.git", "acme/billing"),
    ("ssh://git@github.com/acme/billing", "acme/billing"),
])
def test_parse_github_url(url, slug):
    assert parse_github_url(url).slug == slug


@pytest.mark.parametrize("bad", ["", "https://gitlab.com/a/b", "not a url", "https://github.com/onlyowner"])
def test_parse_github_url_rejects(bad):
    assert parse_github_url(bad) is None


def test_locate_local_path_and_errors(tmp_path):
    repo = make_demo(tmp_path, merge=False)
    assert locate(str(repo)).how == "path"
    assert locate(str(repo / "billing.py").replace("billing.py", "")).path == repo.resolve()
    with pytest.raises(RepoLinkError):
        locate("")
    with pytest.raises(RepoLinkError, match="GitHub"):
        locate("https://gitlab.com/x/y")
    with pytest.raises(RepoLinkError):
        locate(str(tmp_path / "nope"))


def test_locate_github_link_matches_current_clone(tmp_path):
    repo = make_demo(tmp_path, merge=False)
    git(repo, "remote", "add", "origin", "git@github.com:acme/billing.git")
    loc = locate("https://github.com/acme/billing", start_dirs=[repo])
    assert loc.how == "current" and loc.path == repo.resolve()


def test_locate_uses_workspace_clone(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    make_demo(ws / "acme", merge=False, name="billing")
    monkeypatch.setattr(repolink, "WORKSPACE", ws)
    loc = locate("https://github.com/acme/billing", start_dirs=[], allow_clone=False)
    assert loc.how == "workspace"
    with pytest.raises(RepoLinkError, match="No local clone"):
        locate("https://github.com/acme/other", start_dirs=[], allow_clone=False)


def test_start_merge_creates_conflict(tmp_path):
    repo = make_demo(tmp_path, merge=False)
    g = Git(repo)
    local, _ = repolink.branches(g)
    assert {"development", "uat", "production"} <= set(local)
    res = start_merge(g, "production", "uat")
    assert res["conflicts"] and git(repo, "status", "--short") == "UU billing.py"


def test_start_merge_refuses_dirty_tree(tmp_path):
    repo = make_demo(tmp_path, merge=False)
    (repo / "billing.py").write_text("dirty\n")
    with pytest.raises(RepoLinkError, match="uncommitted"):
        start_merge(Git(repo), "production", "uat")
