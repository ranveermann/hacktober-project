import subprocess
import sys
from pathlib import Path

import pytest

from gittrace.ai import OllamaClient
from gittrace.analysis import Analyzer, describe_choice, side_badge
from gittrace.git import Git
from gittrace.provenance import Relation
from gittrace.resolve import resolve_and_finish
from gittrace.store import Store

from fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "demo"
    subprocess.run([sys.executable, str(ROOT / "demo" / "make_demo_repo.py"), str(path)], check=True, capture_output=True)
    return path


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


def git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True).stdout.strip()


def test_demo_badges(demo):
    rep = Analyzer(Git(demo)).analyze()
    h1, h2 = rep.files[0].hunks
    assert side_badge(h1.current).text == "Tested in PROD"
    b = side_badge(h1.incoming)
    assert b.level == "warning" and "passed UAT" in b.text and b.text.startswith("Untested UAT change")
    assert side_badge(h2.current).text == "Unverified PROD change"
    assert side_badge(h2.incoming).text == "Unverified UAT change"


def test_gemma_augments_cross_side_lineage(demo, fake):
    g = Git(demo)
    analyzer = Analyzer(g)
    rep = analyzer.analyze()
    h1, h2 = rep.files[0].hunks
    client = OllamaClient(fake.url)
    # Hunk 1: heuristics can't link the two GST fixes; Gemma says same fix -> probably related.
    a1, rel1 = analyzer.compare_hunk(client, h1, "billing.py")
    assert a1.verdict == "SAME_FIX" and rel1.relation is Relation.PROBABLE
    assert any("judged logically the same fix" in e for e in rel1.evidence)
    # Hunk 2: heuristics call the discount rules 'probably related'; Gemma rejects it.
    assert h2.cross_side and h2.cross_side[0].relation is Relation.PROBABLE
    a2, rel2 = analyzer.compare_hunk(client, h2, "billing.py")
    assert a2.verdict == "DIFFERENT_CHANGES" and rel2.relation is Relation.UNKNOWN
    prompt = fake.requests[0]["body"]["prompt"]
    assert "HOTFIX: customers were charged 10% GST" in prompt and "cherry picked from commit" in prompt


def test_analyzer_with_ai_client_uses_judge(demo, fake):
    rep = Analyzer(Git(demo), ai_client=OllamaClient(fake.url)).analyze()
    h2 = rep.files[0].hunks[1]
    assert all(r.relation is not Relation.PROBABLE for r in h2.cross_side)  # rejected by the local model
    assert fake.requests


def test_click_flow_produces_standard_merge_commit_with_provenance(demo, fake):
    g = Git(demo)
    store = Store(g)
    analyzer = Analyzer(g, store)
    rep = analyzer.analyze()
    h1 = rep.files[0].hunks[0]
    ai, _ = analyzer.compare_hunk(OllamaClient(fake.url), h1, "billing.py")
    prov = describe_choice("Side A", h1.current, side_badge(h1.current), ai)
    first = resolve_and_finish(g, store, "billing.py", 1, "current", provenance=prov)
    assert first["finish"]["action"] == "pending" and first["resolution"]["remaining_hunks"] == 1

    h2 = Analyzer(g, store).analyze().files[0].hunks[0]  # former hunk 2 is now hunk 1
    prov2 = describe_choice("Side B", h2.incoming, side_badge(h2.incoming))
    second = resolve_and_finish(g, store, "billing.py", 1, "incoming", provenance=prov2)
    fin = second["finish"]
    assert fin["action"] == "commit" and len(fin["parents"]) == 2

    msg = git(demo, "log", "-1", "--format=%B")
    assert msg.startswith("Merge branch 'uat' into production")
    assert "Conflict resolution provenance (GitTrace):" in msg
    assert "kept Side A" in msg and "Provenance: Tested in PROD" in msg
    assert "Local AI (gemma4:e4b): SAME_FIX (88%)" in msg
    assert "kept Side B" in msg and "Provenance: Unverified UAT change" in msg
    assert "GitTrace-Resolution: billing.py#1=current" in msg
    content = (demo / "billing.py").read_text()
    assert 'return float(f"{gst:.2f}")' in content and "total > 1000" in content and "<<<<<<<" not in content
    assert git(demo, "status", "--porcelain") == ""
    assert subprocess.run(["git", "fsck", "--no-progress"], cwd=demo).returncode == 0
    assert git(demo, "rev-parse", "HEAD^1") == fin["parents"][0]


# ------------------------------------------------------------------ Streamlit UI
def _app(monkeypatch, demo, fake=None):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GITTRACE_REPO", str(demo))
    if fake:
        monkeypatch.setenv("GITTRACE_OLLAMA_HOST", fake.url)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    return at.run()


def _texts(at):
    out = []
    for kind in ("markdown", "success", "warning", "error", "info", "caption", "code"):
        out += [e.value for e in getattr(at, kind)]
    return "\n".join(str(t) for t in out)


def test_app_shows_conflict_side_by_side(monkeypatch, demo, fake):
    at = _app(monkeypatch, demo, fake)
    assert not at.exception
    text = _texts(at)
    assert "Side A" in text and "Side B" in text
    assert "**Tested in PROD**" in text
    assert "Untested UAT change" in text
    assert "Logically the SAME fix" in text and "DIFFERENT changes" in text
    assert "Both compute 18% GST" in text
    labels = [b.label for b in at.button]
    assert labels.count("✅ Keep Side A") == 2 and labels.count("✅ Keep Side B") == 2
    assert len(at.columns) >= 4  # st.columns(2) for sides + buttons, per hunk


def test_app_buttons_resolve_and_commit(monkeypatch, demo, fake):
    at = _app(monkeypatch, demo, fake)
    next(b for b in at.button if b.label == "✅ Keep Side A").click().run()
    assert not at.exception
    assert "1 hunk(s) left" in _texts(at)
    next(b for b in at.button if b.label == "✅ Keep Side B").click().run()
    assert not at.exception
    text = _texts(at)
    assert "Created standard Git merge commit" in text
    assert "No merge conflict in progress" in text
    assert len(git(demo, "rev-list", "--parents", "-n1", "HEAD").split()) == 3


def test_app_without_ollama_still_works(monkeypatch, demo):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    monkeypatch.setenv("GITTRACE_OLLAMA_HOST", f"http://127.0.0.1:{s.getsockname()[1]}")
    s.close()
    at = _app(monkeypatch, demo)
    assert not at.exception
    text = _texts(at)
    assert "Local AI unavailable" in text and "ollama serve" in text
    assert "**Tested in PROD**" in text


def test_app_no_conflict(monkeypatch, tmp_path):
    repo = tmp_path / "clean"
    subprocess.run([sys.executable, str(ROOT / "demo" / "make_demo_repo.py"), str(repo), "--no-merge"], check=True, capture_output=True)
    at = _app(monkeypatch, repo)
    assert not at.exception and "No merge conflict in progress" in _texts(at)


def test_app_rejects_cloud_endpoint(monkeypatch, demo):
    monkeypatch.setenv("GITTRACE_OLLAMA_HOST", "https://api.openai.com")
    at = _app(monkeypatch, demo)
    assert not at.exception
    assert any("only uses a LOCAL Ollama" in e.value for e in at.sidebar.error)
