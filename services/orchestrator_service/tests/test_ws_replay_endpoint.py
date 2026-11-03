"""REV-23: end-to-end WS replay-on-reconnect through the real handler.

Drives the actual websocket_chat_endpoint via TestClient with a mocked JWT
manager and graph_builder. Proves:
  * each outbound response carries a monotonic per-session seq;
  * reconnecting with ?last_seq=k replays exactly k+1..N in order (no dupes/gaps)
    BEFORE resuming live streaming;
  * a too-old cursor yields a resync signal;
  * no cursor => normal live behaviour (no replay frames injected).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.replay_buffer import InMemoryReplayBufferStore

BASE = "/api/v1/orchestrator"


def _make_app(replay_store) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    gb = MagicMock()

    async def _invoke(state, thread_id=None):
        # echo the inbound message back as the final response
        return {
            "final_response": f"echo:{state.get('current_message', '')}",
            "intent": "general_chat",
            "intent_confidence": 0.5,
            "safety_flags": {},
        }

    gb.invoke = AsyncMock(side_effect=_invoke)
    app.state.graph_builder = gb
    app.state.active_connections = {}
    app.state.replay_buffer = replay_store
    return app


def _jwt_patch():
    manager = MagicMock()
    manager.validate_access_token_async = AsyncMock(
        return_value=MagicMock(success=True, payload={"sub": "user-1"}, user_id="user-1")
    )
    return patch("solace_security.middleware._get_jwt_manager", return_value=manager)


def _send_and_read_response(ws, text: str) -> dict:
    ws.send_json({"type": "message", "message": text})
    frame = ws.receive_json()
    assert frame["type"] == "processing"
    resp = ws.receive_json()
    assert resp["type"] == "response"
    return resp


def test_responses_carry_monotonic_seq_and_replay_on_reconnect() -> None:
    store = InMemoryReplayBufferStore(max_per_session=100)
    app = _make_app(store)
    client = TestClient(app)
    session = "sess-replay"
    with _jwt_patch():
        # Connection 1: send N=5 messages, collect their seqs.
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x") as ws:
            assert ws.receive_json()["type"] == "connected"
            seqs = []
            for i in range(5):
                resp = _send_and_read_response(ws, f"m{i}")
                seqs.append(resp["seq"])
            assert seqs == [1, 2, 3, 4, 5]
        # Client dropped after seq=2. Reconnect with last_seq=2.
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x&last_seq=2") as ws:
            assert ws.receive_json()["type"] == "connected"
            replayed = [ws.receive_json() for _ in range(3)]
            assert [m["seq"] for m in replayed] == [3, 4, 5]
            assert [m["response"] for m in replayed] == ["echo:m2", "echo:m3", "echo:m4"]
            # after replay, live streaming resumes with the next seq
            live = _send_and_read_response(ws, "m5")
            assert live["seq"] == 6


def test_too_old_cursor_yields_resync_signal() -> None:
    store = InMemoryReplayBufferStore(max_per_session=3)
    app = _make_app(store)
    client = TestClient(app)
    session = "sess-gap"
    with _jwt_patch():
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x") as ws:
            assert ws.receive_json()["type"] == "connected"
            for i in range(5):  # seqs 1..5, buffer keeps 3,4,5
                _send_and_read_response(ws, f"m{i}")
        # last_seq=1 -> message 2 evicted -> gap
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x&last_seq=1") as ws:
            assert ws.receive_json()["type"] == "connected"
            frame = ws.receive_json()
            assert frame["type"] == "resync"
            assert frame.get("oldest_seq") == 3


def test_no_cursor_no_replay_normal_live_behaviour() -> None:
    store = InMemoryReplayBufferStore(max_per_session=100)
    app = _make_app(store)
    client = TestClient(app)
    session = "sess-live"
    with _jwt_patch():
        # prime the buffer with prior messages
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x") as ws:
            assert ws.receive_json()["type"] == "connected"
            _send_and_read_response(ws, "prior")
        # reconnect WITHOUT a cursor -> no replay/resync frames injected
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x") as ws:
            assert ws.receive_json()["type"] == "connected"
            # next frame must be the live response to our new message, not a replay
            resp = _send_and_read_response(ws, "fresh")
            assert resp["response"] == "echo:fresh"


def test_replay_disabled_when_store_absent() -> None:
    """Backward-compat: with no replay_buffer on app.state the handler behaves
    exactly as before (no seq, no replay) and existing WS flows stay green."""
    app = _make_app(replay_store=None)
    client = TestClient(app)
    session = "sess-off"
    with _jwt_patch():
        with client.websocket_connect(f"{BASE}/ws/{session}?token=x&last_seq=2") as ws:
            assert ws.receive_json()["type"] == "connected"
            # cursor is ignored when replay is off; normal live behaviour
            resp = _send_and_read_response(ws, "hi")
            assert resp["response"] == "echo:hi"
            assert "seq" not in resp
