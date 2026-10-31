import json
import re
import socket
from pathlib import Path

import pytest

from gittrace.ai import (
    AIAssessment,
    NonLocalEndpointError,
    OllamaClient,
    SideEvidence,
    build_prompt,
    ensure_local,
    parse_assessment,
)
from gittrace.provenance import Relation, Relationship, apply_ai

from fake_ollama import FakeOllama

A = SideEvidence("Side A", "production", "a" * 40, "HOTFIX gst", "HOTFIX gst\n\nbody A", "+    gst = amount * 18 / 100", ["    gst = amount * 18 / 100"])
B = SideEvidence("Side B", "uat", "b" * 40, "Fix GST", "Fix GST\n\nbody B", "+    return round(amount * 0.18, 2)", ["    return round(amount * 0.18, 2)"])


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


@pytest.mark.parametrize("host", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434", "localhost:11434"])
def test_local_hosts_accepted(host):
    assert ensure_local(host).startswith("http")


@pytest.mark.parametrize("host", ["https://api.openai.com", "https://api.anthropic.com", "https://generativelanguage.googleapis.com",
                                  "http://192.168.1.20:11434", "http://localhost.evil.com:11434", "ftp://localhost"])
def test_non_local_hosts_refused(host):
    with pytest.raises(NonLocalEndpointError):
        OllamaClient(host)


def test_generate_request_shape(fake):
    client = OllamaClient(fake.url, "gemma4:e4b")
    res = client.compare(A, B)
    assert res.ok and res.verdict == "SAME_FIX" and res.confidence == pytest.approx(0.88)
    req = fake.requests[-1]
    assert req["path"] == "/api/generate"
    body = req["body"]
    assert body["model"] == "gemma4:e4b" and body["stream"] is False
    assert body["format"]["properties"]["verdict"]["enum"][0] == "SAME_FIX"
    p = body["prompt"]
    for needle in ("HOTFIX gst", "body B", "gst = amount * 18 / 100", "round(amount * 0.18, 2)", "LOGICALLY THE SAME FIX", "Do NOT recommend"):
        assert needle in p


def test_health(fake):
    assert OllamaClient(fake.url, "gemma4:e4b").health() == {"reachable": True, "models": ["gemma4:e4b"], "model_available": True}
    assert OllamaClient(fake.url, "llama3").health()["model_available"] is False


def test_unreachable_server_never_raises():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    res = OllamaClient(f"http://127.0.0.1:{port}", timeout=3).compare(A, B)
    assert not res.ok and res.verdict == "ERROR" and "ollama serve" in res.explanation


def test_missing_model_message(fake):
    fake.responder = lambda p: (404, {"error": "model not found"})
    res = OllamaClient(fake.url, "gemma4:e4b").compare(A, B)
    assert "ollama pull gemma4:e4b" in res.error


def test_schema_fallback_to_json_mode(fake):
    def responder(p):
        if isinstance(p.get("format"), dict):
            return 400, {"error": "invalid format"}
        return 200, {"response": json.dumps({"verdict": "UNCLEAR", "confidence": 0.3, "explanation": "x", "key_differences": []})}
    fake.responder = responder
    res = OllamaClient(fake.url).compare(A, B)
    assert res.verdict == "UNCLEAR" and fake.requests[-1]["body"]["format"] == "json"


def test_cache(fake, tmp_path):
    client = OllamaClient(fake.url, cache_path=tmp_path / "c.json")
    first = client.compare(A, B)
    second = client.compare(A, B)
    assert not first.cached and second.cached and len(fake.requests) == 1
    client.compare(A, B, use_cache=False)
    assert len(fake.requests) == 2


@pytest.mark.parametrize("text,verdict,conf", [
    ('```json\n{"verdict":"SAME_FIX","confidence":0.7,"explanation":"e","key_differences":[]}\n```', "SAME_FIX", 0.7),
    ('Sure! {"verdict":"different changes","confidence":85,"explanation":"e","key_differences":"x"} hope it helps', "DIFFERENT_CHANGES", 0.85),
    ('{"verdict":"MAYBE","confidence":"high","explanation":""}', "UNCLEAR", 0.0),
    ("I think they are the same.", "UNCLEAR", 0.0),
])
def test_parse_assessment_is_tolerant(text, verdict, conf):
    a = parse_assessment(text, "m")
    assert a.verdict == verdict and a.confidence == pytest.approx(conf)


def test_prompt_truncates_huge_diffs():
    big = SideEvidence("Side A", "x", None, "s", "m", "+" + "x" * 50000, [])
    assert "[truncated" in build_prompt(big, B) and len(build_prompt(big, B)) < 20000


def _rel(relation, conf="low", score=0.5):
    return Relationship("a" * 40, "b" * 40, relation, conf, score, ["heuristic"])


def _ai(verdict, conf):
    return AIAssessment(verdict, conf, "because", [], "gemma4:e4b")


def test_apply_ai_rules():
    up = apply_ai(_rel(Relation.UNKNOWN, "none", 0.25), _ai("SAME_FIX", 0.9))
    assert up.relation is Relation.PROBABLE and up.confidence == "medium" and up.ai.verdict == "SAME_FIX"
    strong = apply_ai(_rel(Relation.PROBABLE, "medium", 0.8), _ai("SAME_FIX", 0.9))
    assert strong.confidence == "high"
    rej = apply_ai(_rel(Relation.PROBABLE, "medium", 0.8), _ai("DIFFERENT_CHANGES", 0.9))
    assert rej.relation is Relation.UNKNOWN and any("REJECTED" in e for e in rej.evidence)
    weak = apply_ai(_rel(Relation.PROBABLE, "medium"), _ai("DIFFERENT_CHANGES", 0.5))
    assert weak.relation is Relation.PROBABLE
    part = apply_ai(_rel(Relation.UNKNOWN, "none"), _ai("PARTIALLY_SAME", 0.6))
    assert part.relation is Relation.PROBABLE and part.confidence == "low"
    err = apply_ai(_rel(Relation.PROBABLE), AIAssessment.failed("m", "down"))
    assert err.relation is Relation.PROBABLE and err.confidence == "low"
    for strong_rel in (Relation.EXACT, Relation.REFERENCED, Relation.EQUIVALENT):
        out = apply_ai(_rel(strong_rel, "high"), _ai("DIFFERENT_CHANGES", 0.99))
        assert out.relation is strong_rel  # AI never overrides evidence-backed classes
    # AI can never produce EQUIVALENT
    assert apply_ai(_rel(Relation.UNKNOWN), _ai("SAME_FIX", 1.0)).relation is not Relation.EQUIVALENT


def test_no_cloud_ai_in_codebase():
    """Hackathon rule: no OpenAI / Anthropic / Gemini SDKs or endpoints anywhere in the shipped code."""
    root = Path(__file__).resolve().parents[1]
    banned = re.compile(r"openai|anthropic|generativelanguage|google\.genai|google-generativeai|api\.mistral|cohere", re.I)
    for f in list((root / "src").rglob("*.py")) + [root / "app.py", root / "pyproject.toml"]:
        assert not banned.search(f.read_text(encoding="utf-8")), f"cloud AI reference in {f}"
