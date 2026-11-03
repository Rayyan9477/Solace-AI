"""MVP-211 probe app (not a test module): the query-string token shapes the real services accept.

`test_mvp_211_container_entry.py` serves this app with the orchestrator image's uvicorn flags and checks that
neither token reaches the server's output: the WebSocket `?token=` that D-27 keeps for browser clients, and the
`POST /auth/verify-email?token=` link token. The app itself never logs anything.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/verify-email")
async def verify_email(token: str, user_id: str) -> dict[str, bool]:
    return {"verified": bool(token and user_id)}


@app.websocket("/ws/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str, token: str = "") -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_text(await websocket.receive_text())
    except WebSocketDisconnect:
        return
