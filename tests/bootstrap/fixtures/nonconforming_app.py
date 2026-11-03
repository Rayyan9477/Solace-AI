"""A service module that bypasses the runtime: the conformance probe's negative control (WS-04.5).

It builds a plain ``FastAPI()``, installs its own catch-all exception handler and serves its own
``/health``, which is the pre-IF-01 pattern the conformance suite must reject.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="nonconforming")


@app.exception_handler(Exception)
async def handle_any_exception(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": "internal error"}, status_code=500)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
