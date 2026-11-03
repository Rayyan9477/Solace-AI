"""The service runtime declaration (IF-01) and the startup failure type.

Each MVP service declares one frozen ``ServiceRuntimeSpec``. The runtime bootstrap reads the
spec's flags to decide what to start (Postgres, Redis, the event plane, audit, rate limiting)
and how strictly to treat a failed dependency. This module imports only the standard library,
so declaring a spec never pulls in FastAPI, asyncpg or any client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

EVENT_PLANE_MODES: tuple[str, ...] = ("none", "publish", "publish_consume")

# Any "scheme://..." run. A reason must never carry a DSN or URL, because a DSN can embed a
# password; this pattern is the last line of defence when a caller gets that wrong.
_URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S*")
_URL_PLACEHOLDER = "<redacted-url>"


@dataclass(frozen=True)
class ServiceRuntimeSpec:
    """What one service needs from the shared runtime (IF-01).

    ``environment`` is the service's own resolved setting (for example
    ``settings.environment``). ``docs_enabled=None`` means ``/docs`` and ``/openapi.json``
    are served only outside strict environments. ``max_request_body_bytes`` caps every HTTP
    request body (1 MiB by default); a larger one is refused with 413 ``REQUEST_TOO_LARGE``.
    """

    service_name: str
    environment: str
    version: str = "1.0.0"
    log_level: str = "INFO"
    debug: bool = False
    uses_postgres: bool = False
    uses_redis: bool = False
    event_plane: Literal["none", "publish", "publish_consume"] = "none"
    handles_phi: bool = False
    audited: bool = False
    rate_limited: bool = False
    edge: bool = False
    docs_enabled: bool | None = None
    max_request_body_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if self.event_plane not in EVENT_PLANE_MODES:
            raise ValueError(
                f"event_plane must be one of {EVENT_PLANE_MODES}, got {self.event_plane!r}"
            )
        limit = self.max_request_body_bytes
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(
                f"max_request_body_bytes must be a positive int, got {limit!r}"
            )


class StartupDependencyError(RuntimeError):
    """A dependency the service needs at startup is missing or unusable.

    ``component`` names the dependency (for example ``"logging"`` or ``"postgres"``) and
    ``reason`` says what is wrong in fixed words. Callers never put a secret, DSN or URL into
    ``reason``; any ``scheme://...`` text that slips through is replaced with
    ``<redacted-url>`` before it is stored.
    """

    def __init__(self, component: str, reason: str) -> None:
        self.component = component
        self.reason = _URL_PATTERN.sub(_URL_PLACEHOLDER, reason)
        super().__init__(self.component, self.reason)

    def __str__(self) -> str:
        return f"{self.component}: {self.reason}"


__all__ = ["EVENT_PLANE_MODES", "ServiceRuntimeSpec", "StartupDependencyError"]
