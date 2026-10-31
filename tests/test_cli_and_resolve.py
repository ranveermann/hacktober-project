import json

import pytest

from gittrace.cli import main
from gittrace.resolve import ResolveError, commit, resolve_hunk

from conftest import BASE

FIXED = BASE.replace("return x * 0.1", "return round(x * 0.18, 2)")


def scenario(r):
    """production hotfix vs a modified uat backport of a UAT-tested development fix."""
    orig = r.commit("Fix GST rate", **{"app.py": FIXED})
    r.store.add_test(orig, "uat", "passed", suite="regression")
    r.checkout("uat")
    bp = r.commit(f"Fix GST rate\n\n(cherry picked from commit {orig})", **{"app.py": FIXED.replace("2)", "2)  # uat")})
    r.checkout("production")
    hot = r.commit("Hotfix tax 12%", **{"app.py": BASE.replace("x * 0.1", "x * 0.12")})
    r.store.set_env_branch("production", "production")
    r.store.set_env_branch("uat", "uat")
    r.git_cmd("merge", "uat", check=False)
    return orig, bp, hot


def run(r, capsys, *args):
    code = main(["-C", str(r.path), "--plain", *args])
    return code, capsys.readouterr()


def test_analyze_json(base_repo, capsys):
    orig, bp, hot = scenario(base_repo)
    code, cap = run(base_repo, capsys, "analyze", "--json")
    assert code == 0
    data = json.loads(cap.out)
    assert data["context"]["operation"] == "merge"
    hunk = data["files"][0]["hunks"][0]
    inc = hunk["incoming"]["commits"][0]
    assert inc["commit"] == bp
    assert inc["tests"]["by_environment"]["uat"] == "untested"
    assert inc["lineage"][0]["relation"] == "explicitly referenced backport"
    assert inc["lineage"][0]["modified"] is True
    adv = " ".join(hunk["advisories"])
    assert "does not certify" in adv and "reached production" in adv


def test_analyze_plain_text(base_repo, capsys):
    scenario(base_repo)
    code, cap = run(base_repo, capsys, "analyze")
    assert code == 0
    assert "Operation           merge" in cap.out
    assert "GitTrace does not pick a side" in cap.out
    assert "gittrace resolve app.py --hunk 1" in cap.out


def test_record_and_trace(base_repo, capsys):
    r = base_repo
    c = r.commit("Fix GST", **{"app.py": FIXED})
    code, cap = run(r, capsys, "record-test", "--commit", c, "--environment", "uat", "--result", "passed", "--suite", "smoke", "--run-id", "42", "--meta", "build=7")
    assert code == 0 and "uat = passed" in cap.out
    code, cap = run(r, capsys, "trace", c, "--json")
    data = json.loads(cap.out)
    assert data["tests"]["exact"][0]["run_id"] == "42"
    assert data["tests"]["by_environment"]["uat"] == "passed"


def test_cli_errors(base_repo, capsys, tmp_path):
    code, cap = run(base_repo, capsys, "record-test", "--commit", "nope", "-e", "uat", "-r", "passed")
    assert code == 2 and "not a commit" in cap.err
    notrepo = tmp_path / "plain"
    notrepo.mkdir()
    assert main(["-C", str(notrepo), "analyze"]) == 2


def test_resolve_and_commit_produce_normal_merge(base_repo):
    r = base_repo
    orig, bp, hot = scenario(r)
    with pytest.raises(ResolveError):
        resolve_hunk(r.git, r.store, "app.py", 5, "incoming")
    res = resolve_hunk(r.git, r.store, "app.py", 1, "incoming", stage=True)
    assert res["remaining_hunks"] == 0 and res["staged"]
    assert "round(x * 0.18, 2)  # uat" in r.read("app.py") and "<<<<<<<" not in r.read("app.py")
    h = commit(r.git, r.store)
    parents = r.git_cmd("rev-list", "--parents", "-n1", h).stdout.split()[1:]
    assert parents == [hot, bp]  # a standard two-parent merge commit
    msg = r.git_cmd("log", "-1", "--format=%B").stdout
    assert "GitTrace-Resolution: app.py#1=incoming" in msg
    assert r.git_cmd("status", "--porcelain").stdout.strip() == ""
    assert r.git_cmd("fsck", "--no-progress").returncode == 0


def test_commit_refuses_with_unmerged_paths(base_repo):
    scenario(base_repo)
    with pytest.raises(ResolveError):
        commit(base_repo.git, base_repo.store)


def test_resolve_preserves_crlf(base_repo):
    r = base_repo
    r.commit("crlf", **{"w.txt": "a\r\nb\r\nc\r\n"})
    r.git_cmd("branch", "-f", "production")
    r.commit("dev", **{"w.txt": "a\r\nDEV\r\nc\r\n"})
    r.checkout("production")
    r.commit("prod", **{"w.txt": "a\r\nPROD\r\nc\r\n"})
    r.git_cmd("merge", "development", check=False)
    resolve_hunk(r.git, r.store, "w.txt", 1, "both")
    assert (r.path / "w.txt").read_bytes() == b"a\r\nPROD\r\nDEV\r\nc\r\n"


def test_cherry_command(base_repo, capsys):
    r = base_repo
    orig = r.commit("Fix GST", **{"app.py": FIXED})
    r.checkout("uat")
    r.git_cmd("cherry-pick", orig)
    r.checkout("development")
    code, cap = run(r, capsys, "cherry", "uat", "development", "--json")
    rows = {row["commit"]: row for row in json.loads(cap.out)}
    assert rows[orig]["in_upstream"] == "equivalent patch"
    assert any(row["in_upstream"] == "missing" for row in rows.values())
