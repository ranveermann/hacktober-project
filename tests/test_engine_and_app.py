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
