"""Request context, uniform error bodies and edge CORS for every service (IF-01, MVP-036).

``RequestContextMiddleware`` is pure ASGI (not ``BaseHTTPMiddleware``). Per request it:

- accepts ``X-Request-ID``, ``X-Correlation-ID`` and ``X-Turn-ID`` when they match
  ``[A-Za-z0-9._:-]{1,128}`` and mints a ``uuid4()`` for any that is missing or malformed, so
  a CR/LF, control character or oversized value never reaches a log line or a header (an id
  the PHI log sanitizer would flag is replaced too, see ``_PHI_PATTERNS``, and a DEBUG
  ``request_id_replaced`` line names the header, never the value);
- stores the ids on ``scope["state"]`` (``request.state.request_id`` and friends), which is
  what lets the 500 handler in Starlette's outermost ``ServerErrorMiddleware`` see them;
- sets them in ``solace_common.request_context`` (``ServiceRuntime.outbound_headers()``) and
  binds them into the structlog contextvars, starting from a cleared context;
- echoes them as response headers;
- records ``solace_http_requests_total`` and ``solace_http_request_duration_seconds`` by
  route template (never the raw path) and writes one ``http_request`` line with method,
  route template, status and ``duration_ms`` (DEBUG for the probe routes), never the query
  string;
- clears the ids again when the request ends.

``RequestBodyLimitMiddleware`` (pure ASGI, inside ``RequestContextMiddleware``) refuses a body
over ``spec.max_request_body_bytes`` with 413 ``REQUEST_TOO_LARGE``: a ``Content-Length`` over the
limit before anything is read, a chunked or length-less body as soon as it crosses the limit.

``install_exception_handlers`` gives every error the same body, and never the exception text:
``{"error": {"code", "message", "request_id"}}`` (422 adds ``details``). ``HTTPException`` keeps
FastAPI's ``{"detail"}`` body until WS-04.9 (D-11). ``install_cors`` is used for edge services.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

import structlog
from fastapi import FastAPI, Request, WebSocket
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.status import WS_1008_POLICY_VIOLATION
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from solace_common.request_context import clear_correlation_ids, set_correlation_ids
from solace_infrastructure.runtime.health import HEALTH_PATHS

# One strictness rule for the whole runtime: strict when the spec's environment OR the bare
# ENVIRONMENT is strict. The logging module owns it; it is re-exported here for the factory.
from solace_infrastructure.runtime.logging import _is_strict as spec_is_strict

# One rule for summarising a pydantic error without its input, shared by the 422 body and the
# log lines (the logging module owns it, so the logging chain applies the same rule).
from solace_infrastructure.runtime.logging import (
    _find_validation_error,
    _masked_field_path,
    _raise_site,
    _validation_error_fields,
)
from solace_infrastructure.runtime.metrics import HTTP_DURATION, HTTP_REQUESTS, METRICS_PATH
from solace_infrastructure.runtime.spec import ServiceRuntimeSpec, StartupDependencyError

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"
TURN_ID_HEADER = "X-Turn-ID"

# (state / contextvar key, header name, lower-case ASGI header name)
_ID_FIELDS: tuple[tuple[str, str, bytes], ...] = (
    ("request_id", REQUEST_ID_HEADER, b"x-request-id"),
    ("correlation_id", CORRELATION_ID_HEADER, b"x-correlation-id"),
    ("turn_id", TURN_ID_HEADER, b"x-turn-id"),
)
_ID_HEADER_NAMES: frozenset[bytes] = frozenset(raw for _, _, raw in _ID_FIELDS)

# fullmatch only: "$" would also accept a value that ends in "\n".
ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_MAX_MINT_ATTEMPTS = 16

UNMATCHED_ROUTE = "__unmatched__"
# Probe and scrape routes log their access line at DEBUG so they do not flood the logs.
QUIET_ROUTES: frozenset[str] = HEALTH_PATHS | {METRICS_PATH}
# Any other method shares one label value, so a client cannot grow the metric's cardinality.
_METRIC_METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
)
_OTHER_METHOD = "OTHER"
# Response status recorded when the client went away before any response started.
_CLIENT_CLOSED_STATUS = 499

CORS_ORIGINS_ENV = "CORS_ALLOWED_ORIGINS"
CORS_ALLOW_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
CORS_EXPOSE_HEADERS: tuple[str, ...] = (
    REQUEST_ID_HEADER,
    CORRELATION_ID_HEADER,
    TURN_ID_HEADER,
    "X-Session-ID",
    "X-Thread-ID",
    "Retry-After",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
)
_FORBIDDEN_STRICT_ORIGINS: frozenset[str] = frozenset({"*", "null"})

INTERNAL_ERROR_CODE = "INTERNAL_ERROR"
INTERNAL_ERROR_MESSAGE = "An unexpected error occurred"
BAD_REQUEST_CODE = "BAD_REQUEST"
BAD_REQUEST_MESSAGE = "Invalid request"
VALIDATION_ERROR_CODE = "VALIDATION_ERROR"
VALIDATION_ERROR_MESSAGE = "Request validation failed"
REQUEST_TOO_LARGE_CODE = "REQUEST_TOO_LARGE"
REQUEST_TOO_LARGE_MESSAGE = "Request body too large"
# A Content-Length with more digits than this is over any limit (and int() of a very long digit
# string raises), so it is read as this many bytes.
_MAX_LENGTH_DIGITS = 18
_UNREADABLY_LARGE = 10**_MAX_LENGTH_DIGITS

# Pydantic error types whose built-in message is built from the schema only, never from the
# input, so it is safe to show. Every other type (including union_tag_invalid and
# uuid_parsing, which quote the input, value_error/assertion_error, whose text a validator
# author wrote, custom types and any type a later pydantic adds) gets the generic message.
_SAFE_MESSAGE_ERROR_TYPES: frozenset[str] = frozenset(
    {
        "missing",
        "extra_forbidden",
        "string_type",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "int_type",
        "int_parsing",
        "int_from_float",
        "float_type",
        "float_parsing",
        "bool_type",
        "bool_parsing",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "multiple_of",
        "too_short",
        "too_long",
        "list_type",
        "dict_type",
        "model_type",
        "model_attributes_type",
        "literal_error",
        "enum",
        "json_invalid",
        "json_type",
        "uuid_type",
        "missing_argument",
    }
)
_GENERIC_FIELD_MESSAGE = "Invalid value"
_MAX_VALIDATION_DETAILS = 50

_log = structlog.get_logger("solace_infrastructure.runtime.http")


# --------------------------------------------------------------------------- request ids


def _load_phi_patterns() -> tuple[re.Pattern[str], ...]:
    """The log sanitizer's PHI patterns, compiled the way its detector compiles them."""
    try:
        from solace_security.phi_protection import DEFAULT_PATTERNS
    except ImportError:  # no sanitizer (development only): nothing to stay clear of
        return ()
    return tuple(re.compile(p.pattern, re.IGNORECASE) for p in DEFAULT_PATTERNS)


# The ids are bound into every log line of the request, so they must pass the PHI log
# sanitizer untouched: a masked id no longer correlates, and with the current sanitizer a
# flagged context value makes every log call of the request recurse until RecursionError.
_PHI_PATTERNS = _load_phi_patterns()


def _looks_like_phi(value: str) -> bool:
    return any(pattern.search(value) for pattern in _PHI_PATTERNS)


def _mint_id() -> str:
    """A fresh ``uuid4`` string the PHI log sanitizer does not flag (about 1 in 60 is)."""
    candidate = str(uuid.uuid4())
    for _ in range(_MAX_MINT_ATTEMPTS):
        if not _looks_like_phi(candidate):
            break
        candidate = str(uuid.uuid4())
    return candidate


def _accept_or_mint(raw: bytes | None) -> tuple[str, bool]:
    """Return ``(id, phi_like_replaced)``: the inbound id when it is well formed and not
    PHI-like, else a minted one. The flag is True only when a PHI-like value was replaced."""
    if raw is None:
        return _mint_id(), False
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        return _mint_id(), False
    if not ID_PATTERN.fullmatch(value):
        return _mint_id(), False
    if _looks_like_phi(value):
        return _mint_id(), True
    return value, False


def _inbound_ids(scope: Scope) -> tuple[dict[str, str], list[str]]:
    """Read the three id headers (first occurrence wins), minting any that are unusable.

    Returns the ids and the names of the headers whose PHI-like value was replaced.
    """
    found: dict[bytes, bytes] = {}
    for name, value in scope.get("headers") or ():
        lowered = bytes(name).lower()
        if lowered in _ID_HEADER_NAMES and lowered not in found:
            found[lowered] = bytes(value)
    ids: dict[str, str] = {}
    replaced: list[str] = []
    for key, header, raw in _ID_FIELDS:
        ids[key], phi_like = _accept_or_mint(found.get(raw))
        if phi_like:
            replaced.append(header)
    return ids, replaced


def _enter_request_context(ids: Mapping[str, str]) -> None:
    structlog.contextvars.clear_contextvars()
    clear_correlation_ids()
    set_correlation_ids(
        request_id=ids["request_id"],
        correlation_id=ids["correlation_id"],
        turn_id=ids["turn_id"],
    )
    structlog.contextvars.bind_contextvars(**ids)


def _leave_request_context() -> None:
    clear_correlation_ids()
    structlog.contextvars.clear_contextvars()


def _state_ids(request: Request) -> dict[str, str | None]:
    """The ids the middleware stored for this request (``None`` when it did not run)."""
    return {key: getattr(request.state, key, None) for key, _, _ in _ID_FIELDS}


def _id_response_headers(ids: Mapping[str, str | None]) -> dict[str, str]:
    return {header: value for key, header, _ in _ID_FIELDS if (value := ids.get(key))}


def route_template(scope: Scope) -> str:
    """The matched route's path template, or ``__unmatched__``. Never the raw request path.

    FastAPI routes put themselves on ``scope["route"]``. Plain Starlette routes (``/docs``,
    ``/openapi.json``) only set ``scope["endpoint"]``, so their template is looked up among
    the app's routes by endpoint.
    """
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path
    endpoint = scope.get("endpoint")
    if endpoint is not None:
        router = getattr(scope.get("app"), "router", None)
        for candidate in getattr(router, "routes", ()):
            if getattr(candidate, "endpoint", None) is endpoint:
                path = getattr(candidate, "path", None)
                if isinstance(path, str) and path:
                    return path
    return UNMATCHED_ROUTE


def _metric_method(method: Any) -> str:
    upper = str(method or "").upper()
    return upper if upper in _METRIC_METHODS else _OTHER_METHOD


class RequestContextMiddleware:
    """Pure-ASGI request context: ids, response id headers, HTTP metrics and the access line."""

    def __init__(self, app: ASGIApp, *, service_name: str) -> None:
        self.app = app
        self.service_name = service_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        ids, replaced = _inbound_ids(scope)
        scope.setdefault("state", {}).update(ids)
        _enter_request_context(ids)
        try:
            for header in replaced:
                # The header name only: the replaced value is the PHI-like text itself.
                _log.debug("request_id_replaced", header=header)
            if scope["type"] == "http":
                await self._serve_http(scope, receive, send, ids)
            else:
                await self.app(scope, receive, send)
        finally:
            _leave_request_context()

    async def _serve_http(
        self, scope: Scope, receive: Receive, send: Send, ids: Mapping[str, str]
    ) -> None:
        started = time.perf_counter()
        status: int | None = None
        id_headers = [
            (raw, ids[key].encode("ascii")) for key, _, raw in _ID_FIELDS
        ]

        async def send_with_ids(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                kept = [
                    (name, value)
                    for name, value in message.get("headers") or ()
                    if bytes(name).lower() not in _ID_HEADER_NAMES
                ]
                message = {**message, "headers": kept + id_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_ids)
        except asyncio.CancelledError:
            if status is None:
                status = _CLIENT_CLOSED_STATUS
            raise
        except BaseException:
            if status is None:
                status = 500
            raise
        finally:
            self._record(scope, 500 if status is None else status, time.perf_counter() - started)

    def _record(self, scope: Scope, status: int, seconds: float) -> None:
        route = route_template(scope)
        method = _metric_method(scope.get("method"))
        HTTP_REQUESTS.labels(
            service=self.service_name, route=route, method=method, status=str(status)
        ).inc()
        HTTP_DURATION.labels(service=self.service_name, route=route).observe(seconds)
        write = _log.debug if route in QUIET_ROUTES else _log.info
        write(
            "http_request",
            service=self.service_name,
            method=method,
            route=route,
            status=status,
            duration_ms=round(seconds * 1000.0, 2),
        )


# --------------------------------------------------------------------------- body limit


def _declared_body_length(scope: Scope) -> int | None:
    """The request's ``Content-Length``, or None when it has none or one that is not a number."""
    for name, value in scope.get("headers") or ():
        if bytes(name).lower() == b"content-length":
            digits = bytes(value).strip()
            if not digits.isdigit():
                return None  # the server rejects a malformed length; counting still applies
            return int(digits) if len(digits) <= _MAX_LENGTH_DIGITS else _UNREADABLY_LARGE
    return None


async def _receive_nothing() -> Message:
    return {"type": "http.disconnect"}


async def _send_request_too_large(scope: Scope, send: Send) -> None:
    """The 413 envelope: a fixed message and the request id, never the body or a header."""
    state = scope.get("state") or {}
    body = _error_body(REQUEST_TOO_LARGE_CODE, REQUEST_TOO_LARGE_MESSAGE, state.get("request_id"))
    # Connection: close, so the server neither reads nor waits for the rest of the body.
    response = JSONResponse(body, status_code=413, headers={"Connection": "close"})
    await response(scope, _receive_nothing, send)


class RequestBodyLimitMiddleware:
    """Pure-ASGI request body cap (HTT-2): 413 ``REQUEST_TOO_LARGE`` over ``max_body_bytes``.

    ``create_service_app`` installs it inside ``RequestContextMiddleware``, so a refusal carries
    the request ids and is counted and logged like any other response. The 413 body is the
    uniform error envelope with a fixed message: neither the body nor a header is echoed.

    - A ``Content-Length`` over the limit is refused before anything is read.
    - A chunked or length-less body is counted as it is received. Once it passes the limit the
      app gets ``http.disconnect`` instead of the rest, whatever the app answers to the cut-off
      body (or the exception the cut-off causes) is dropped, and the 413 is sent, unless the app
      had already started its response.

    So the app never receives more than ``max_body_bytes`` of body. Other scope types (WebSocket,
    lifespan) pass through unchanged.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_body_bytes
        declared = _declared_body_length(scope)
        if declared is not None and declared > limit:
            await _send_request_too_large(scope, send)
            return

        received = 0
        cut_off = False
        app_started = False

        async def limited_receive() -> Message:
            nonlocal received, cut_off
            if cut_off:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    cut_off = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal app_started
            if cut_off and not app_started:
                return  # the app's answer to a cut-off body: the 413 replaces it
            if message["type"] == "http.response.start":
                app_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not cut_off or app_started:
                raise
            # The app failed on the cut-off body (a disconnect mid-read): the 413 answers it.
        if cut_off and not app_started:
            await _send_request_too_large(scope, send)


# --------------------------------------------------------------------------- error bodies


def _error_body(
    code: str, message: str, request_id: str | None, *, details: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    error["request_id"] = request_id
    return {"error": error}


def _validation_detail(error: Mapping[str, Any]) -> dict[str, str]:
    """One error as ``{field, message, type}``; never the ``input`` or ``ctx`` values.

    The built-in message is shown only for allowlisted types; anything else fails closed to
    "Invalid value". The field path is masked as in every log summary (``_masked_field_path``):
    a non-identifier location part, and the key an ``extra_forbidden`` error names, are
    ``[key]``.
    """
    error_type = str(error.get("type", "value_error"))
    if error_type in _SAFE_MESSAGE_ERROR_TYPES:
        message = str(error.get("msg", _GENERIC_FIELD_MESSAGE))
    else:
        message = _GENERIC_FIELD_MESSAGE
    return {"field": _masked_field_path(error), "message": message, "type": error_type}


def _log_server_fault(request: Request, exc: Exception, ids: Mapping[str, str | None]) -> None:
    """One ``unhandled_exception`` ERROR line for a 500, with the request ids.

    When ``exc`` is a ``pydantic.ValidationError``, or one is linked to it (its cause, its
    context or an ExceptionGroup member), the line carries the error types, masked field paths
    and raise site only: a ValidationError's message quotes every input value
    (``input_value=...``), and the regex PHI sanitizer cannot recognise names or free text, so
    neither the message nor the traceback is logged. Any other exception is logged with its
    traceback (MVP-180).
    """
    common: dict[str, Any] = {
        **ids,
        "method": _metric_method(request.scope.get("method")),
        "route": route_template(request.scope),
        "error_type": type(exc).__name__,
    }
    validation = _find_validation_error(exc)
    if validation is not None:
        _log.error(
            "unhandled_exception",
            **common,
            validation_errors=_validation_error_fields(validation),
            origin=_raise_site(exc),
        )
    else:
        _log.error("unhandled_exception", **common, exc_info=exc)


def _internal_error_response(ids: Mapping[str, str | None]) -> JSONResponse:
    body = _error_body(INTERNAL_ERROR_CODE, INTERNAL_ERROR_MESSAGE, ids.get("request_id"))
    return JSONResponse(body, status_code=500, headers=_id_response_headers(ids))


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    details = [_validation_detail(error) for error in errors[:_MAX_VALIDATION_DETAILS]]
    body = _error_body(
        VALIDATION_ERROR_CODE,
        VALIDATION_ERROR_MESSAGE,
        getattr(request.state, "request_id", None),
        details=details,
    )
    return JSONResponse(body, status_code=422)


async def _websocket_validation_error_handler(websocket: WebSocket, exc: Exception) -> None:
    # FastAPI's default puts the errors, input values included, into the close reason.
    await websocket.close(code=WS_1008_POLICY_VIOLATION, reason=VALIDATION_ERROR_MESSAGE)


async def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if request.scope.get("type") != "http":
        # A WebSocket has no response body to send; keep the default handling.
        raise exc
    if isinstance(exc, PydanticValidationError):
        # A ValueError subclass, but raised by the service's own models (stored data that no
        # longer validates, say): a server fault, so a 500. It is answered here instead of
        # re-raised: a re-raised exception reaches the ASGI server, whose own "Exception in
        # ASGI application" log line prints str(exc), input values included.
        ids = _state_ids(request)
        _log_server_fault(request, exc, ids)
        return _internal_error_response(ids)
    # The message is not logged: domain code quotes the rejected input ("unknown mood: <text>")
    # and the PHI sanitizer cannot recognise free text. The raise site is enough to find it.
    _log.info("request_value_error", error_type=type(exc).__name__, origin=_raise_site(exc))
    body = _error_body(BAD_REQUEST_CODE, BAD_REQUEST_MESSAGE, getattr(request.state, "request_id", None))
    return JSONResponse(body, status_code=400)


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Runs in ServerErrorMiddleware, outside RequestContextMiddleware: the contextvars are
    # already cleared, so the ids come from request.state (they live on scope["state"]).
    ids = _state_ids(request)
    _log_server_fault(request, exc, ids)
    return _internal_error_response(ids)


def install_exception_handlers(app: FastAPI, spec: ServiceRuntimeSpec) -> None:
    """Install the uniform error bodies. No body ever carries ``str(exc)``.

    - ``RequestValidationError`` -> 422 ``VALIDATION_ERROR`` with ``details`` of
      ``{field, message, type}`` (no input values);
    - ``ValueError`` -> 400 ``BAD_REQUEST``, logged with its raise site only (never the
      message, which may quote patient text);
    - ``pydantic.ValidationError`` (a ``ValueError`` subclass, raised by the service's own
      models) -> 500 ``INTERNAL_ERROR``, logged as error types, field paths and raise site;
    - any other exception -> 500 ``INTERNAL_ERROR`` (logged with its traceback);
    - ``WebSocketRequestValidationError`` -> close 1008 with a fixed reason.

    ``HTTPException`` keeps FastAPI's ``{"detail"}`` body; WS-04.9 (W2) adds the envelope for
    ``spec.edge`` services.
    """
    del spec  # used from W2 (WS-04.9): edge services get the HTTPException envelope
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(WebSocketRequestValidationError, _websocket_validation_error_handler)
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)


# --------------------------------------------------------------------------- edge CORS


def cors_allowed_origins(spec: ServiceRuntimeSpec) -> list[str]:
    """Origins from ``CORS_ALLOWED_ORIGINS`` (comma-separated).

    Raises:
        StartupDependencyError: in a strict environment, when the list is empty or holds
            ``*`` or ``null``.
    """
    raw = os.environ.get(CORS_ORIGINS_ENV, "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if spec_is_strict(spec):
        if not origins:
            raise StartupDependencyError(
                "cors", f"{CORS_ORIGINS_ENV} must list the allowed origins in strict environments"
            )
        if any(origin in _FORBIDDEN_STRICT_ORIGINS for origin in origins):
            raise StartupDependencyError(
                "cors", f"{CORS_ORIGINS_ENV} must not contain '*' or 'null' in strict environments"
            )
    return origins


def install_cors(app: FastAPI, origins: list[str]) -> None:
    """Add Starlette's CORS middleware for ``origins`` (call it last so it is outermost).

    Credentials are allowed only for an explicit origin list: with ``*`` Starlette would
    reflect any caller's origin on credentialed requests. An empty list allows no origin.
    """
    wildcard = "*" in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else list(origins),
        allow_credentials=not wildcard,
        allow_methods=list(CORS_ALLOW_METHODS),
        allow_headers=["*"],
        expose_headers=list(CORS_EXPOSE_HEADERS),
    )


__all__ = [
    "CORRELATION_ID_HEADER",
    "CORS_ORIGINS_ENV",
    "ID_PATTERN",
    "REQUEST_ID_HEADER",
    "REQUEST_TOO_LARGE_CODE",
    "TURN_ID_HEADER",
    "UNMATCHED_ROUTE",
    "RequestBodyLimitMiddleware",
    "RequestContextMiddleware",
    "cors_allowed_origins",
    "install_cors",
    "install_exception_handlers",
    "route_template",
    "spec_is_strict",
]
