"""Local-only AI assessment through Ollama (default model: gemma4:e4b).

Hackathon rule: no cloud AI. This module talks ONLY to an Ollama server on the
loopback interface. `ensure_local()` rejects any non-localhost endpoint, and
redirects are not followed, so a misconfiguration can't silently send code to a
remote service.

The model answers one question: are Side A and Side B *logically the same fix*,
even if their bytes differ? Its answer is advisory. It can strengthen or weaken a
heuristic "probably related" match, but it can never produce an "equivalent patch"
(that still requires identical git patch-ids) or choose a side.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_HOST = os.environ.get("GITTRACE_OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("GITTRACE_OLLAMA_MODEL", "gemma4:e4b")
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}
VERDICTS = ("SAME_FIX", "PARTIALLY_SAME", "DIFFERENT_CHANGES", "UNCLEAR")
MAX_DIFF_CHARS = 4000
MAX_MESSAGE_CHARS = 1500
PROMPT_VERSION = "v1"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
        "key_differences": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "explanation", "key_differences"],
}


class AIError(RuntimeError):
    """Ollama could not produce an assessment (unreachable, model missing, bad reply...)."""


class NonLocalEndpointError(AIError):
    """Refused: the configured endpoint is not on this machine."""


@dataclass
class SideEvidence:
    name: str  # "Side A" / "Side B"
    label: str  # semantic label, e.g. "current branch (HEAD = production)"
    commit: str | None
    subject: str
    message: str
    diff: str  # diff of the originating commit, limited to the conflicted file
    conflict_lines: list[str]


@dataclass
class AIAssessment:
    verdict: str  # one of VERDICTS, or "ERROR"
    confidence: float
    explanation: str
    key_differences: list[str] = field(default_factory=list)
    model: str = ""
    duration_s: float = 0.0
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @staticmethod
    def failed(model: str, message: str) -> "AIAssessment":
        return AIAssessment("ERROR", 0.0, message, [], model, error=message)


def ensure_local(host: str) -> str:
    """Return the normalised base URL, or raise if it doesn't point at this machine."""
    parsed = urlparse(host if "://" in host else f"http://{host}")
    if parsed.scheme not in ("http", "https"):
        raise NonLocalEndpointError(f"unsupported scheme in '{host}'")
    if (parsed.hostname or "").lower() not in LOCAL_HOSTNAMES:
        raise NonLocalEndpointError(
            f"refusing to call '{host}': GitTrace only uses a LOCAL Ollama server "
            f"(localhost / 127.0.0.1 / ::1). Cloud AI endpoints are not allowed."
        )
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def build_prompt(a: SideEvidence, b: SideEvidence) -> str:
    def block(s: SideEvidence) -> str:
        code = "\n".join(s.conflict_lines) if s.conflict_lines else "(this side deletes the lines)"
        return (
            f"### {s.name} - {s.label}\n"
            f"Commit: {s.commit[:12] if s.commit else 'unknown'}  {s.subject}\n"
            f"Full commit message:\n{_clip(s.message.strip(), MAX_MESSAGE_CHARS) or '(none)'}\n\n"
            f"Conflicting lines on {s.name}:\n```\n{code}\n```\n\n"
            f"Diff of the commit that introduced them (conflicted file only):\n```diff\n{_clip(s.diff, MAX_DIFF_CHARS) or '(not available)'}\n```\n"
        )

    return (
        "You are a senior software engineer reviewing a Git merge conflict between two long-lived "
        "branches (for example development, UAT and production).\n"
        "Decide whether Side A and Side B are LOGICALLY THE SAME FIX: the same intent AND the same "
        "behavioural effect, even if the bytes differ (renamed variables, comments, formatting, "
        "reordered statements, equivalent refactoring).\n\n"
        "Rules:\n"
        "- Judge runtime behaviour, not textual similarity.\n"
        "- Different constants, thresholds, conditions or return values mean different behaviour "
        "(DIFFERENT_CHANGES or PARTIALLY_SAME), even when the commit messages are similar.\n"
        "- Comment-only, whitespace-only or logging-only differences do NOT change behaviour.\n"
        "- PARTIALLY_SAME: same intent, but one side does more or something slightly different.\n"
        "- If the information is insufficient, answer UNCLEAR. Never guess.\n"
        "- Do NOT recommend which side to keep; a human decides.\n\n"
        f"{block(a)}\n{block(b)}\n"
        "Answer with JSON only, exactly in this shape:\n"
        '{"verdict": "SAME_FIX | PARTIALLY_SAME | DIFFERENT_CHANGES | UNCLEAR", '
        '"confidence": <number 0.0-1.0>, '
        '"explanation": "<at most 120 words, mention the concrete code that matters>", '
        '"key_differences": ["<short behavioural difference>", ...]}'
    )


def parse_assessment(text: str, model: str = "") -> AIAssessment:
    """Parse the model reply; tolerate code fences or prose around the JSON."""
    data = None
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        c = re.sub(r"^```(?:json)?|```$", "", c.strip()).strip()
        try:
            data = json.loads(c)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if not isinstance(data, dict):
        snippet = text.strip()[:600] or "(empty reply)"
        return AIAssessment("UNCLEAR", 0.0, f"Model reply was not valid JSON: {snippet}", [], model)
    verdict = str(data.get("verdict", "UNCLEAR")).strip().upper().replace(" ", "_")
    if verdict not in VERDICTS:
        verdict = "UNCLEAR"
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if conf > 1.0:  # some models answer in percent
        conf = conf / 100.0
    diffs = data.get("key_differences") or []
    if isinstance(diffs, str):
        diffs = [diffs]
    return AIAssessment(
        verdict=verdict,
        confidence=max(0.0, min(1.0, conf)),
        explanation=str(data.get("explanation", "")).strip() or "(no explanation given)",
        key_differences=[str(d) for d in diffs][:8],
        model=model,
    )


class OllamaClient:
    """Minimal client for Ollama's REST API (POST /api/generate) using `requests`."""

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL, timeout: float = 240.0,
                 cache_path: Path | None = None, session: requests.Session | None = None):
        self.base = ensure_local(host)
        self.model = model
        self.timeout = timeout
        self.cache_path = cache_path
        self.session = session or requests.Session()
        self.session.trust_env = False  # never route localhost traffic through an HTTP(S)_PROXY

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=5, allow_redirects=False)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException as exc:
            return {"reachable": False, "error": str(exc), "models": [], "model_available": False}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return {"reachable": True, "models": names, "model_available": wanted in names or self.model in names}

    # ---------------------------------------------------------------- generate
    def generate(self, prompt: str, json_schema: dict | None = None) -> str:
        """POST /api/generate (non-streaming) and return the model's text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
            "keep_alive": "15m",
        }
        if json_schema is not None:
            payload["format"] = json_schema
        url = f"{self.base}/api/generate"
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout, allow_redirects=False)
            if r.status_code == 400 and json_schema is not None:
                # Older Ollama builds don't accept a JSON schema in "format"; fall back to plain JSON mode.
                payload["format"] = "json"
                r = self.session.post(url, json=payload, timeout=self.timeout, allow_redirects=False)
        except requests.ConnectionError as exc:
            raise AIError(f"Ollama is not reachable at {self.base}. Start it with `ollama serve`.") from exc
        except requests.Timeout as exc:
            raise AIError(f"Ollama did not answer within {self.timeout:.0f}s (model still loading?).") from exc
        except requests.RequestException as exc:
            raise AIError(f"request to Ollama failed: {exc}") from exc
        if r.status_code == 404:
            raise AIError(f"model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if r.status_code >= 300:
            raise AIError(f"Ollama returned HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
        except ValueError as exc:
            raise AIError("Ollama returned a non-JSON response") from exc
        if body.get("error"):
            raise AIError(f"Ollama error: {body['error']}")
        return body.get("response", "")

    # ----------------------------------------------------------------- compare
    def compare(self, a: SideEvidence, b: SideEvidence, use_cache: bool = True) -> AIAssessment:
        """Ask the model whether Side A and Side B are logically the same fix. Never raises."""
        prompt = build_prompt(a, b)
        key = hashlib.sha256(f"{PROMPT_VERSION}\0{self.model}\0{prompt}".encode()).hexdigest()
        if use_cache:
            hit = self._cache_get(key)
            if hit:
                hit.cached = True
                return hit
        start = time.monotonic()
        try:
            text = self.generate(prompt, RESPONSE_SCHEMA)
        except AIError as exc:
            return AIAssessment.failed(self.model, str(exc))
        result = parse_assessment(text, self.model)
        result.duration_s = round(time.monotonic() - start, 2)
        if result.verdict != "UNCLEAR" or result.confidence > 0:
            self._cache_put(key, result)
        return result

    # ------------------------------------------------------------------- cache
    def _load_cache(self) -> dict:
        if not self.cache_path or not self.cache_path.is_file():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _cache_get(self, key: str) -> AIAssessment | None:
        raw = self._load_cache().get(key)
        if not raw:
            return None
        try:
            return AIAssessment(**raw)
        except TypeError:
            return None

    def _cache_put(self, key: str, value: AIAssessment) -> None:
        if not self.cache_path:
            return
        data = self._load_cache()
        data[key] = asdict(value)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_path.parent, prefix=".ai-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, self.cache_path)
