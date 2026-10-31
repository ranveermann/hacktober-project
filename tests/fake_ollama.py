"""A tiny stand-in for Ollama's REST API, served on 127.0.0.1 for tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def default_responder(payload: dict) -> tuple[int, dict]:
    prompt = payload.get("prompt", "")
    if "total * 0.95" in prompt or "total * 0.9" in prompt:
        answer = {"verdict": "DIFFERENT_CHANGES", "confidence": 0.9,
                  "explanation": "Side A gives 5% off from 500; Side B gives 10% off above 1000. Different thresholds and rates.",
                  "key_differences": ["threshold 500 vs 1000", "rate 5% vs 10%"]}
    else:
        answer = {"verdict": "SAME_FIX", "confidence": 0.88,
                  "explanation": "Both compute 18% GST rounded to two decimals; only formatting differs.",
                  "key_differences": []}
    return 200, {"model": payload.get("model"), "response": json.dumps(answer), "done": True}


class FakeOllama:
    def __init__(self, responder=default_responder, models=("gemma4:e4b",)):
        self.responder = responder
        self.models = list(models)
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in fake.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.requests.append({"path": self.path, "body": body})
                if self.path != "/api/generate":
                    return self._send(404, {"error": "not found"})
                code, resp = fake.responder(body)
                self._send(code, resp)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
