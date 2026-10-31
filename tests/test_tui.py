"""End-to-end tests for the Rich terminal UI (cli.py), driven through piped keyboard input."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli.py"


def make_demo(tmp_path, merge=True, name="demo"):
    path = tmp_path / name
    args = [sys.executable, str(ROOT / "demo" / "make_demo_repo.py"), str(path)] + ([] if merge else ["--no-merge"])
    subprocess.run(args, check=True, capture_output=True)
    return path


def git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True).stdout.strip()


def run(args, keys="", cwd=None, env_extra=None, timeout=120):
    env = {**os.environ, "COLUMNS": "150", "NO_COLOR": "1", **(env_extra or {})}
    proc = subprocess.run([sys.executable, str(CLI), *args], input=keys, capture_output=True, text=True,
                          cwd=cwd, env=env, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


def test_no_args_uses_cwd_and_commits(tmp_path):
    repo = make_demo(tmp_path)
    code, out = run(["--no-ai"], "a\nb\n", cwd=repo)
    assert code == 0, out
    assert "Side A" in out and "Side B" in out and "Merge complete" in out
    assert len(git(repo, "rev-list", "--parents", "-n1", "HEAD").split()) == 3  # two-parent merge commit
    msg = git(repo, "log", "-1", "--format=%B")
    assert "kept Side A" in msg and "kept Side B" in msg and "GitTrace-Resolution" in msg
    assert git(repo, "status", "--porcelain") == ""


def test_gemma_panel_and_side_by_side(tmp_path, fake):
    repo = make_demo(tmp_path)
    code, out = run([str(repo), "--ollama-host", fake.url], "a\nb\n")
    assert code == 0, out
    assert "Gemma analysis (local)" in out
    assert "Logically the SAME fix" in out and "DIFFERENT changes" in out
    assert "Tested in PROD" in out and "Untested UAT change" in out
    # side-by-side: both panel titles on the same terminal line at 150 columns
    assert any("Side A" in ln and "Side B" in ln for ln in out.splitlines())
    assert "Local AI (gemma4:e4b): SAME_FIX" in git(repo, "log", "-1", "--format=%B")


def test_quit_leaves_repo_untouched(tmp_path):
    repo = make_demo(tmp_path)
    before = (repo / "billing.py").read_text()
    code, out = run(["--no-ai"], "q\n", cwd=repo)
    assert code == 130 and "Aborted" in out and "git merge --abort" in out
    assert (repo / "billing.py").read_text() == before
    assert git(repo, "status", "--short") == "UU billing.py"


def test_eof_counts_as_quit(tmp_path):
    repo = make_demo(tmp_path)
    code, out = run(["--no-ai"], "", cwd=repo)
    assert code == 130


def test_invalid_key_reprompts_and_details(tmp_path):
    repo = make_demo(tmp_path)
    code, out = run(["--no-ai"], "x\nd\nA\nB\n", cwd=repo)
    assert code == 0, out
    assert "Please type one of" in out and "Provenance details" in out
    assert "Merge complete" in out


def test_skip_leaves_hunk_unresolved(tmp_path):
    repo = make_demo(tmp_path)
    code, out = run(["--no-ai"], "s\na\n", cwd=repo)
    assert code == 1 and "skipped" in out
    text = (repo / "billing.py").read_text()
    assert text.count("<<<<<<<") == 1 and "total >= 500" in text  # hunk 2 resolved with Side A, hunk 1 still open


def test_report_mode_changes_nothing(tmp_path):
    repo = make_demo(tmp_path)
    code, out = run(["--no-ai", "--report"], "", cwd=repo)
    assert code == 0 and "Side A" in out
    assert git(repo, "status", "--short") == "UU billing.py"


def test_remote_url_is_cloned_to_temp_and_merged(tmp_path):
    src = make_demo(tmp_path, merge=False)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(src), str(bare)], check=True)
    code, out = run([bare.as_uri(), "--merge", "production", "uat", "--no-ai"], "a\nb\n", cwd=tmp_path)
    assert code == 0, out
    assert "Cloned into" in out and "Merge complete" in out
    kept = re.search(r"Clone kept at (\S+)", out.replace("\n", " ")).group(1)
    assert Path(kept).is_dir() and Path(kept).parent.name.startswith("gittrace-")
    assert git(kept, "log", "-1", "--format=%s") == "Merge remote-tracking branch 'origin/uat' into production"
    assert len(git(kept, "rev-list", "--parents", "-n1", "HEAD").split()) == 3


def test_remote_url_without_conflicts_is_cleaned_up(tmp_path):
    src = make_demo(tmp_path, merge=False)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "--quiet", "--bare", str(src), str(bare)], check=True)
    code, out = run([bare.as_uri(), "--no-ai"], "n\n", cwd=tmp_path)
    assert code == 0 and "No merge conflicts" in out and "Temporary clone removed" in out


def test_interactive_merge_start_in_clean_repo(tmp_path):
    repo = make_demo(tmp_path, merge=False)
    # branches listed alphabetically: development, production, uat  ->  target 2 (production), incoming 2 (uat)
    code, out = run(["--no-ai"], "y\n2\n2\na\nb\n", cwd=repo)
    assert code == 0, out
    assert "Start a merge?" in out and "Merge complete" in out
    assert git(repo, "log", "-1", "--format=%s") == "Merge branch 'uat' into production"


def test_errors(tmp_path):
    code, out = run(["--no-ai", str(tmp_path)])
    assert code == 2 and "not inside a Git repository" in out
    code, out = run(["--no-ai", (tmp_path / "missing.git").as_uri()])
    assert code == 2 and "git clone failed" in out
    code, out = run(["--ollama-host", "https://api.openai.com", str(tmp_path)])
    assert code == 2 and "LOCAL Ollama" in out


def test_is_remote_url():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gt_cli", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for u in ("https://github.com/user/repo.git", "git@github.com:user/repo.git", "ssh://git@host/x", "file:///tmp/x.git", "user/repo.git"):
        assert mod.is_remote_url(u), u
    for u in (".", "/tmp", "relative/path"):
        assert not mod.is_remote_url(u), u
