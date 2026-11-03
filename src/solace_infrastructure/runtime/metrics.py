"""Standard Prometheus metrics for every service and the ``/metrics`` route (IF-01, MVP-036).

The metric objects live on the default ``prometheus_client.REGISTRY`` at module level, so any
code in the service process (the runtime, the safety pipeline, the LLM client, the audit
writer) increments the same series that ``GET /metrics`` renders.

Label values are always bounded: ``route`` is a route template (never a raw path), ``method``
is one of a fixed set, and ``service``/``check``/``task`` are names the service declares.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

METRICS_PATH = "/metrics"


def _metric(cls: type, name: str, documentation: str, labelnames: tuple[str, ...]) -> Any:
    """Create a metric on ``REGISTRY``, or return the one an earlier import registered.

    A second import of this module (``importlib.reload`` in a test, or the same file reached
    through a second package path) would otherwise fail with "Duplicated timeseries".
    """
    try:
        return cls(name, documentation, labelnames)
    except ValueError:
        names_to_collectors = getattr(REGISTRY, "_names_to_collectors", {})
        existing = names_to_collectors.get(name)
        if isinstance(existing, cls) and tuple(getattr(existing, "_labelnames", ())) == labelnames:
            return existing
        raise


HTTP_REQUESTS = _metric(
    Counter,
    "solace_http_requests_total",
    "HTTP requests served, by route template, method and status code.",
    ("service", "route", "method", "status"),
)
HTTP_DURATION = _metric(
    Histogram,
    "solace_http_request_duration_seconds",
    "HTTP request duration in seconds, by route template.",
    ("service", "route"),
)
READINESS_CHECK_OK = _metric(
    Gauge,
    "solace_readiness_check_ok",
    "1 when the named readiness check passed at its last evaluation, else 0.",
    ("service", "check"),
)
SERVICE_READY = _metric(
    Gauge,
    "solace_service_ready",
    "1 when every critical readiness check passed at the last evaluation, else 0.",
    ("service",),
)
BACKGROUND_TASK_UP = _metric(
    Gauge,
    "solace_background_task_up",
    "1 while the named background task is running, 0 once it has exited.",
    ("service", "task"),
)
CRISIS_DETECTED = _metric(
    Counter,
    "solace_crisis_detected_total",
    "Crisis detections, by detecting source and crisis level.",
    ("source", "level"),
)
LLM_FALLBACK = _metric(
    Counter,
    "solace_llm_fallback_total",
    "LLM calls answered by a fallback instead of the primary provider, by reason.",
    ("service", "reason"),
)
AUDIT_WRITE_FAILURES = _metric(
    Counter,
    "solace_audit_write_failures_total",
    "Audit records that could not be written.",
    ("service",),
)


async def metrics_endpoint() -> Response:
    """Render every metric on the default registry in the Prometheus text format."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def register_metrics_route(app: FastAPI) -> None:
    """Add ``GET /metrics`` to ``app`` (kept out of the OpenAPI schema)."""
    app.add_api_route(
        METRICS_PATH,
        metrics_endpoint,
        methods=["GET"],
        include_in_schema=False,
        name="runtime_metrics",
    )


__all__ = [
    "AUDIT_WRITE_FAILURES",
    "BACKGROUND_TASK_UP",
    "CRISIS_DETECTED",
    "HTTP_DURATION",
    "HTTP_REQUESTS",
    "LLM_FALLBACK",
    "METRICS_PATH",
    "READINESS_CHECK_OK",
    "SERVICE_READY",
    "register_metrics_route",
]
