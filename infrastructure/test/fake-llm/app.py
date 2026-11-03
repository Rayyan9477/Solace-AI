"""Fake LLM test double (MVP-039, D-04). Standard library only.

Serves two routes from one scenario state and one request log:
  POST /v1/chat/completions        OpenAI chat.completion shape
  POST /model/{model_id}/converse  bedrock-runtime Converse shape (SigV4 ignored, never recorded)
Control: POST /__control/scenario, POST /__control/reset, GET /__control/requests, GET /health.
TEST ONLY: never referenced by docker-compose.yml or docker-compose.prod.yml.
"""
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

FAKE_REPLY = "Thank you for telling me. I'm here with you."
CRISIS_INTENT = (
    '{"intent":"crisis_disclosure","is_crisis":true,"crisis_level":"HIGH","risk_level":"HIGH",'
    '"confidence":0.95,"categories":["suicidal_ideation"]}'
)
DEFAULT_TIMEOUT_SECONDS = 65.0
_CONVERSE_RE = re.compile(r"^/model/([^/]+)/converse$")


class _State:
    def __init__(self, timeout_seconds):
        self.lock = threading.Lock()
        self.timeout_seconds = timeout_seconds
        self.scenario = {"mode": "ok", "match": None, "content": None}
        self.requests = []


def _tokens(text):
    return max(1, len(text.split()))


def _decide(state, last_user):
    """Return (mode, content) for this request under the current scenario."""
    with state.lock:
        sc = dict(state.scenario)
    mode, match = sc["mode"], sc["match"]
    if match and match.lower() not in last_user.lower():
        mode = "ok"
    if mode == "echo":
        return mode, "ECHO: " + last_user
    if mode == "crisis_intent":
        return mode, CRISIS_INTENT
    if mode == "reply":
        return mode, sc["content"] or ""
    if mode in ("error", "timeout"):
        return mode, ""
    return "ok", FAKE_REPLY


def _openai_last_user(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if isinstance(p, dict))
            return "" if c is None else str(c)
    return ""


def _converse_last_user(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return "".join(b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict))
    return ""


def make_server(host, port, *, timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("FAKE_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    state = _State(float(timeout_seconds))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence; never log headers
            pass

        def _send(self, status, body, headers=None):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/__control/requests":
                with state.lock:
                    snap = list(state.requests)
                self._send(200, snap)
            else:
                self._send(404, {"error": {"type": "not_found", "message": self.path}})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/__control/scenario":
                body = self._read_json()
                with state.lock:
                    state.scenario = {
                        "mode": body.get("mode", "ok"),
                        "match": body.get("match"),
                        "content": body.get("content"),
                    }
                self._send(200, {"status": "ok"})
            elif path == "/__control/reset":
                with state.lock:
                    state.scenario = {"mode": "ok", "match": None, "content": None}
                    state.requests = []
                self._send(200, {"status": "ok"})
            elif path == "/v1/chat/completions":
                self._chat(path)
            else:
                m = _CONVERSE_RE.match(path)
                if m:
                    self._converse(path, unquote(m.group(1)))
                else:
                    self._send(404, {"error": {"type": "not_found", "message": path}})

        def _record(self, path, model, messages):
            with state.lock:
                state.requests.append({"ts": time.time(), "path": path, "model": model, "messages": messages})

        def _chat(self, path):
            body = self._read_json()
            messages = body.get("messages") or []
            model = body.get("model", "fake-model")
            self._record(path, model, messages)
            last = _openai_last_user(messages)
            mode, content = _decide(state, last)
            if mode == "error":
                self._send(500, {"error": {"type": "server_error", "message": "fake-llm forced error"}})
                return
            if mode == "timeout":
                time.sleep(state.timeout_seconds)
                self._send(504, {"error": {"type": "timeout", "message": "fake-llm forced timeout"}})
                return
            pt, ct = _tokens(last), _tokens(content)
            self._send(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
            })

        def _converse(self, path, model_id):
            body = self._read_json()
            messages = body.get("messages") or []
            self._record(path, model_id, messages)
            last = _converse_last_user(messages)
            mode, content = _decide(state, last)
            if mode == "error":
                self._send(500, {"message": "fake-llm forced error"}, {"x-amzn-ErrorType": "InternalServerException"})
                return
            if mode == "timeout":
                time.sleep(state.timeout_seconds)
                self._send(504, {"message": "fake-llm forced timeout"}, {"x-amzn-ErrorType": "ModelTimeoutException"})
                return
            pt, ct = _tokens(last), _tokens(content)
            self._send(200, {
                "output": {"message": {"role": "assistant", "content": [{"text": content}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": pt, "outputTokens": ct, "totalTokens": pt + ct},
                "metrics": {"latencyMs": 1},
            })

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    port = int(os.environ.get("FAKE_LLM_PORT", "8080"))
    server = make_server("0.0.0.0", port)  # noqa: S104 - test-only container
    server.serve_forever()


if __name__ == "__main__":
    main()
