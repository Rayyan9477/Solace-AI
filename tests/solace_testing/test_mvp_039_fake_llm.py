"""MVP-039 / D-04: the fake LLM test double (OpenAI chat route + Bedrock Converse route)."""
import importlib.util
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

APP = Path(__file__).resolve().parents[2] / "infrastructure" / "test" / "fake-llm" / "app.py"


@pytest.fixture
def fake():
    spec = importlib.util.spec_from_file_location("fake_llm_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = mod.make_server("127.0.0.1", 0, timeout_seconds=0.5)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield mod, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _chat(base, text):
    return httpx.post(f"{base}/v1/chat/completions", json={"model": "claude-test", "messages": [{"role": "user", "content": text}]}, timeout=5)


def test_mvp_039_ok_mode_returns_openai_shape(fake):
    mod, base = fake
    body = _chat(base, "hello").json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"] == {"role": "assistant", "content": mod.FAKE_REPLY}


def test_mvp_039_echo_error_and_timeout_modes(fake):
    _, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "echo"})
    assert _chat(base, "abc").json()["choices"][0]["message"]["content"] == "ECHO: abc"
    httpx.post(f"{base}/__control/scenario", json={"mode": "error"})
    assert _chat(base, "abc").status_code == 500
    httpx.post(f"{base}/__control/scenario", json={"mode": "timeout"})
    t0 = time.monotonic()
    r = _chat(base, "abc")
    assert r.status_code == 504 and time.monotonic() - t0 >= 0.5


def test_mvp_039_crisis_intent_is_scoped_by_match(fake):
    mod, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "crisis_intent", "match": "bridge downtown"})
    hit = json.loads(_chat(base, "I keep thinking about the Bridge Downtown").json()["choices"][0]["message"]["content"])
    assert hit["intent"] == "crisis_disclosure" and hit["categories"] == ["suicidal_ideation"]
    assert _chat(base, "nice weather").json()["choices"][0]["message"]["content"] == mod.FAKE_REPLY


def test_mvp_039_reply_mode_and_request_recording(fake):
    _, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "reply", "content": "{\"risk_level\": \"NONE\"}"})
    assert _chat(base, "x").json()["choices"][0]["message"]["content"] == "{\"risk_level\": \"NONE\"}"
    assert [r["messages"][-1]["content"] for r in httpx.get(f"{base}/__control/requests").json()] == ["x"]
    httpx.post(f"{base}/__control/reset")
    assert httpx.get(f"{base}/__control/requests").json() == []


def test_d04_bedrock_converse_route_answers_the_real_botocore_client(fake):
    # D-04: the client WS-09 uses (bedrock-runtime Converse through botocore) parses the fake's bodies and errors.
    # Unsigned and pointed at the fake, so no credentials are needed and no AWS endpoint is ever called.
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
    mod, base = fake
    client = boto3.client("bedrock-runtime", region_name="us-east-1", endpoint_url=base,
                          config=Config(signature_version=UNSIGNED, retries={"total_max_attempts": 1, "mode": "standard"}))
    msgs = [{"role": "user", "content": [{"text": "hello"}]}]
    out = client.converse(modelId="fake-e2e-model:0", messages=msgs)          # synthetic id; ':' exercises path encoding
    assert out["output"]["message"] == {"role": "assistant", "content": [{"text": mod.FAKE_REPLY}]} and out["stopReason"] == "end_turn"
    httpx.post(f"{base}/__control/scenario", json={"mode": "echo"})
    assert client.converse(modelId="fake-e2e-model:0", messages=msgs)["output"]["message"]["content"][0]["text"] == "ECHO: hello"
    httpx.post(f"{base}/__control/scenario", json={"mode": "error"})
    with pytest.raises(ClientError) as err:
        client.converse(modelId="fake-e2e-model:0", messages=msgs)
    assert err.value.response["Error"]["Code"] == "InternalServerException"
    first = httpx.get(f"{base}/__control/requests").json()[0]
    assert first["path"].startswith("/model/") and first["model"] == "fake-e2e-model:0"
