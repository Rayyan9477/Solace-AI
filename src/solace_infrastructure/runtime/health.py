"""Health routes for every service: ``/live``, ``/ready`` and ``/health`` (IF-01).

- ``GET /live`` answers whether the process is healthy: 200 ``{"status": "alive"}``, or 503
  ``{"status": "dead", "task": <name>}`` once a registered background task has exited.
  It never touches a dependency, so a datastore outage never makes a container restart.
- ``GET /ready`` (and ``GET /health``, the same answer) runs the registered readiness checks:
  200 with ``status`` ``"ready"``, or 503 with ``"not_ready"`` when a critical check fails, the
  runtime has not started yet, or ``stop_service_runtime`` has begun (the drain: no check runs)
  or finished (it removes the runtime from ``app.state``).

The routes read ``app.state.runtime`` (a ``ServiceRuntime``) and ``app.state.spec`` at request
time; this module does not import the bootstrap, so it has no import cycle with it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from solace_infrastructure.runtime.metrics import SERVICE_READY

LIVE_PATH = "/live"
READY_PATH = "/ready"
HEALTH_PATH = "/health"
HEALTH_PATHS: frozenset[str] = frozenset({LIVE_PATH, READY_PATH, HEALTH_PATH})


@dataclass(frozen=True)
class CheckResult:
    """One readiness check outcome. ``detail`` is fixed text or an exception class name."""

    ok: bool
    critical: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    """The outcome of every readiness check. ``ready`` is False when a critical check failed."""

    ready: bool
    checks: Mapping[str, CheckResult]


async def live_endpoint(request: Request) -> JSONResponse:
    """Liveness: 503 naming the first registered background task that has exited."""
    runtime = getattr(request.app.state, "runtime", None)
    dead = runtime.dead_background_task() if runtime is not None else None
    if dead is not None:
        return JSONResponse({"status": "dead", "task": dead}, status_code=503)
    return JSONResponse({"status": "alive"})


async def ready_endpoint(request: Request) -> JSONResponse:
    """Readiness: 200 when every critical check passes, else 503."""
    spec = getattr(request.app.state, "spec", None)
    service = getattr(spec, "service_name", "unknown")
    body: dict[str, Any] = {
        "status": "not_ready",
        "service": service,
        "version": getattr(spec, "version", "unknown"),
        "checks": {},
    }
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        # The lifespan has not started the runtime yet, or stop_service_runtime has detached it.
        SERVICE_READY.labels(service=service).set(0)
        return JSONResponse(body, status_code=503)

    report: ReadinessReport = await runtime.evaluate_readiness()
    body["status"] = "ready" if report.ready else "not_ready"
    body["checks"] = {
        name: {"ok": result.ok, "critical": result.critical, "detail": result.detail}
        for name, result in report.checks.items()
    }
    return JSONResponse(body, status_code=200 if report.ready else 503)


def register_health_routes(app: FastAPI) -> None:
    """Add ``GET /live``, ``GET /ready`` and ``GET /health`` to ``app`` (not in OpenAPI)."""
    app.add_api_route(
        LIVE_PATH, live_endpoint, methods=["GET"], include_in_schema=False, name="runtime_live"
    )
    app.add_api_route(
        READY_PATH, ready_endpoint, methods=["GET"], include_in_schema=False, name="runtime_ready"
    )
    app.add_api_route(
        HEALTH_PATH,
        ready_endpoint,
        methods=["GET"],
        include_in_schema=False,
        name="runtime_health",
    )


__all__ = [
    "HEALTH_PATH",
    "HEALTH_PATHS",
    "LIVE_PATH",
    "READY_PATH",
    "CheckResult",
    "ReadinessReport",
    "register_health_routes",
]
