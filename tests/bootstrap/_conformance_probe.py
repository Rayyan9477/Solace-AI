"""Conformance probe for one service module (MVP-036 / IF-01); run in a fresh interpreter.

    python _conformance_probe.py <module>

``tests/bootstrap/test_service_conformance.py`` runs it with ``PYTHONPATH`` set to the service's
import roots, ``ENVIRONMENT=development`` and synthetic secrets. It imports ``<module>``, reads
``module.SPEC`` and ``module.app``, and prints exactly one JSON line, the last line on stdout:

- ``has_spec``: ``module.SPEC`` is a ``ServiceRuntimeSpec``;
- ``spec``: ``dataclasses.asdict(SPEC)``, or ``{}``;
- ``created_by_runtime``: ``app.state.spec is SPEC`` (only ``create_service_app`` sets it);
- ``runtime_routes``: paths whose endpoint ``__module__`` starts with
  ``solace_infrastructure.runtime``;
- ``foreign_exception_handlers``: the handlers in ``app.exception_handlers`` whose
  ``__module__`` is not ``solace_infrastructure.runtime.http``, FastAPI's own three defaults
  excepted;
- ``shadowed_runtime_routes``: which of ``/live``, ``/ready``, ``/health`` and ``/metrics`` a GET
  would reach on a route other than the runtime's;
- ``startup``: ``None``, unless the runtime built the app. Then the probe applies uvicorn's default
  logging configuration (as the uvicorn CLI does before the lifespan), enters the app's lifespan
  and stops it as soon as ``start_service_runtime`` returns, so no service-specific start-up code
  runs. It reports the logging state at that point (``uvicorn.access`` disabled, the handlers
  uvicorn's loggers still own, the runtime handlers on the root logger, ``logging_chain_names()``)
  and calls ``/live``, ``/ready``, ``/health``, ``/metrics`` and an unknown path in process:
  their status codes, whether any secret-like environment value appears in a body or header, and
  whether the unknown path or its query value became a metric label. It then stops the runtime.

Hermetic by construction:

- the import runs from a fresh, empty working directory (removed again before exit), so no
  settings class reads a developer's ``.env`` (most of them use ``env_file=".env"``), as in a
  container image, which has none;
- every outbound connection (``socket.create_connection``, ``socket.connect``, the asyncio
  loop's ``create_connection``/``sock_connect``) is refused, so the module and the runtime never
  reach a real Postgres, Redis, Kafka or provider, and a development start degrades at once;
- everything the module and the runtime write to stdout goes to stderr, and the report is
  written to the original stdout last, then the process exits with ``os._exit``, so a lingering
  thread cannot add a line after it or keep the process alive;
- the value of every secret-like environment variable is replaced with ``<redacted>`` in all
  stderr output, tracebacks included;
- the report holds names, booleans and status codes only: never an environment value. An import
  failure exits 1 with the traceback on stderr, of which the test shows the last 2000 characters.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from typing import Any

# Environment variables whose values count as secrets: redacted from stderr, and looked for in
# the runtime routes' responses.
_SECRET_NAME = re.compile(r"KEY|SECRET|PASSWORD|PASSWD|TOKEN|SALT|CREDENTIAL", re.IGNORECASE)
_URL_NAME = re.compile(r"(_URL|_DSN|_URI)$", re.IGNORECASE)
_MIN_SECRET_LENGTH = 8
_REDACTED = "<redacted>"


def _secret_values() -> list[str]:
    values = {
        value
        for name, value in os.environ.items()
        if len(value) >= _MIN_SECRET_LENGTH
        and (_SECRET_NAME.search(name) or (_URL_NAME.search(name) and "@" in value))
    }
    return sorted(values, key=len, reverse=True)  # longest first, so no value is half-replaced


class _RedactingStream:
    """A text stream that replaces every secret-like environment value before writing it."""

    def __init__(self, stream: Any, secrets: list[str]) -> None:
        self._stream = stream
        self._secrets = secrets

    def write(self, text: str) -> int:
        for value in self._secrets:
            text = text.replace(value, _REDACTED)
        return self._stream.write(text)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# Redirect first, before any import can bind sys.stdout: the report must be the last stdout line.
_REPORT_STREAM = sys.stdout
for _stream in (_REPORT_STREAM, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
sys.stderr = _RedactingStream(sys.stderr, _secret_values())
sys.stdout = sys.stderr

# A fresh, empty working directory: no .env is found, and nothing the import writes stays behind.
_ORIGINAL_CWD = os.getcwd()
_WORKDIR = tempfile.mkdtemp(prefix="solace-conformance-probe-")
os.chdir(_WORKDIR)

import asyncio  # noqa: E402
import asyncio.base_events  # noqa: E402
import asyncio.proactor_events  # noqa: E402
import asyncio.selector_events  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import socket  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

# The probe's own directory (tests/bootstrap) is not one of the service's import roots.
_PROBE_DIR = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != _PROBE_DIR]

RUNTIME_PACKAGE = "solace_infrastructure.runtime"
RUNTIME_HTTP_MODULE = "solace_infrastructure.runtime.http"
RUNTIME_PATHS = ("/live", "/ready", "/health", "/metrics")
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
RUNTIME_ROOT_HANDLER = "solace-runtime-stdout"
STARTUP_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 15.0


# --------------------------------------------------------------------------- network off


class _NetworkDisabledError(ConnectionRefusedError):
    """Every outbound connection attempt in the probe fails with this (a ConnectionRefusedError)."""


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise _NetworkDisabledError("conformance probe: outbound connections are disabled")


async def _refuse_async(*_args: Any, **_kwargs: Any) -> Any:
    raise _NetworkDisabledError("conformance probe: outbound connections are disabled")


def _disable_network() -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    socket_file = socket.__file__

    def _from_socketpair() -> bool:
        # On Windows, socket.socketpair() is emulated with a loopback listen/connect, and the
        # asyncio event loop needs it for its self-pipe. That one stdlib caller is allowed.
        caller = sys._getframe(2).f_code
        return caller.co_filename == socket_file and "socketpair" in caller.co_name

    def connect(self: socket.socket, address: Any) -> None:
        if _from_socketpair():
            return real_connect(self, address)
        _refuse()

    def connect_ex(self: socket.socket, address: Any) -> int:
        if _from_socketpair():
            return real_connect_ex(self, address)
        _refuse()
        return 1  # unreachable

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.create_connection = _refuse  # type: ignore[assignment]
    asyncio.base_events.BaseEventLoop.create_connection = _refuse_async  # type: ignore[method-assign]
    asyncio.proactor_events.BaseProactorEventLoop.sock_connect = _refuse_async  # type: ignore[method-assign]
    asyncio.selector_events.BaseSelectorEventLoop.sock_connect = _refuse_async  # type: ignore[method-assign]


# --------------------------------------------------------------------------- static report


def _qualified_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None) or type(obj).__module__
    name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{module}.{name}"


def _key_name(key: Any) -> str:
    return key.__name__ if isinstance(key, type) else str(key)


def _fastapi_default_handlers() -> set[tuple[Any, Any]]:
    from fastapi.exception_handlers import (
        http_exception_handler,
        request_validation_exception_handler,
        websocket_request_validation_exception_handler,
    )
    from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    return {
        (StarletteHTTPException, http_exception_handler),
        (RequestValidationError, request_validation_exception_handler),
        (WebSocketRequestValidationError, websocket_request_validation_exception_handler),
    }


def _foreign_exception_handlers(app: Any) -> list[str]:
    defaults = _fastapi_default_handlers()
    foreign = []
    for key, handler in getattr(app, "exception_handlers", {}).items():
        if (key, handler) in defaults:
            continue
        if getattr(handler, "__module__", None) == RUNTIME_HTTP_MODULE:
            continue
        foreign.append(f"{_key_name(key)} -> {_qualified_name(handler)}")
    return sorted(foreign)


def _is_runtime_endpoint(endpoint: Any) -> bool:
    module = getattr(endpoint, "__module__", None) or ""
    return module == RUNTIME_PACKAGE or module.startswith(RUNTIME_PACKAGE + ".")


def _runtime_routes(app: Any) -> list[str]:
    return sorted(
        {
            route.path
            for route in getattr(app, "routes", ())
            if isinstance(getattr(route, "path", None), str)
            and _is_runtime_endpoint(getattr(route, "endpoint", None))
        }
    )


def _shadowed_runtime_routes(app: Any) -> list[str]:
    """The runtime paths a GET would serve from a route other than the runtime's own."""
    from starlette.routing import Match

    shadowed = []
    for path in RUNTIME_PATHS:
        scope = {"type": "http", "path": path, "root_path": "", "method": "GET", "headers": []}
        for route in getattr(app, "routes", ()):
            try:
                match, _ = route.matches(scope)
            except Exception:  # a route that needs more of the scope (a Host route) cannot match
                match = Match.NONE
            if match is Match.FULL:
                if not _is_runtime_endpoint(getattr(route, "endpoint", None)):
                    shadowed.append(path)
                break
    return shadowed


# --------------------------------------------------------------------------- start-up report


class _StopAfterRuntimeStart(BaseException):
    """Raised right after start_service_runtime returns, so no service-specific start-up runs.

    A BaseException, so an ``except Exception`` in the lifespan cannot swallow it.
    """


def _logging_state() -> dict[str, Any]:
    from solace_infrastructure.runtime.logging import logging_chain_names

    access = logging.getLogger("uvicorn.access")
    return {
        "uvicorn_access_disabled": access.disabled,
        "uvicorn_access_propagates": access.propagate,
        "uvicorn_own_handlers": {name: len(logging.getLogger(name).handlers) for name in UVICORN_LOGGERS},
        "root_runtime_handlers": sum(
            1 for handler in logging.getLogger().handlers if handler.get_name() == RUNTIME_ROOT_HANDLER
        ),
        "logging_chain": logging_chain_names(),
    }


async def _route_report(app: Any, service_name: str) -> dict[str, Any]:
    import httpx

    secrets_in_env = _secret_values()
    marker_path = f"conformance-probe-{uuid.uuid4().hex}"
    marker_query = uuid.uuid4().hex
    status: dict[str, int] = {}
    texts: list[str] = []
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://conformance-probe", timeout=REQUEST_TIMEOUT_SECONDS
    ) as client:
        unknown = await client.get(f"/{marker_path}/item", params={"token": marker_query})
        status["unknown_path"] = unknown.status_code
        texts.append(unknown.text)
        texts.extend(f"{name}: {value}" for name, value in unknown.headers.items())
        metrics_text = ""
        for path in RUNTIME_PATHS:
            response = await client.get(path)
            status[path] = response.status_code
            texts.append(response.text)
            texts.extend(f"{name}: {value}" for name, value in response.headers.items())
            if path == "/metrics":
                metrics_text = response.text
    return {
        "route_status": status,
        "secret_in_responses": any(secret in text for secret in secrets_in_env for text in texts),
        "raw_path_in_metrics": marker_path in metrics_text or marker_query in metrics_text,
        "service_label_in_metrics": f'service="{service_name}"' in metrics_text,
    }


def _apply_uvicorn_logging(app: Any) -> None:
    """What ``uvicorn <module>:app`` does to logging before it runs the lifespan."""
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    uvicorn.Config(app, log_config=copy.deepcopy(LOGGING_CONFIG))


async def _startup_report(app: Any, spec: Any) -> dict[str, Any]:
    import solace_infrastructure.runtime as runtime_package
    from solace_infrastructure.runtime import bootstrap

    report: dict[str, Any] = {"ran": False, "error": None}
    real_start = bootstrap.start_service_runtime
    started: dict[str, Any] = {}

    async def start_then_stop(app_: Any, spec_: Any) -> Any:
        runtime = await real_start(app_, spec_)
        started["runtime"] = runtime
        raise _StopAfterRuntimeStart

    # The lifespan reads the name from its module's globals at call time; the package and the
    # bootstrap module are patched too, for any other import style.
    bootstrap.start_service_runtime = start_then_stop
    vars(runtime_package)["start_service_runtime"] = start_then_stop
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if isinstance(namespace, dict) and namespace.get("start_service_runtime") is real_start:
            namespace["start_service_runtime"] = start_then_stop

    _apply_uvicorn_logging(app)
    try:
        async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
            async with app.router.lifespan_context(app):
                pass
    except _StopAfterRuntimeStart:
        pass
    except BaseException as exc:  # the class name is the report
        report["error"] = type(exc).__name__
    runtime = started.get("runtime")
    if runtime is None:
        if report["error"] is None:
            report["error"] = "start_service_runtime was not called by the lifespan"
        return report

    report["ran"] = True
    report.update(_logging_state())
    try:
        report.update(await _route_report(app, spec.service_name))
    except Exception as exc:
        report["error"] = f"route check failed: {type(exc).__name__}"
    finally:
        await bootstrap.stop_service_runtime(runtime)
    return report


# --------------------------------------------------------------------------- main


def _build_report(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)

    from solace_infrastructure.runtime.spec import ServiceRuntimeSpec

    spec = getattr(module, "SPEC", None)
    has_spec = isinstance(spec, ServiceRuntimeSpec)
    app = getattr(module, "app", None)
    state = getattr(app, "state", None)
    created_by_runtime = has_spec and state is not None and getattr(state, "spec", None) is spec
    report: dict[str, Any] = {
        "module": module_name,
        "has_spec": has_spec,
        "spec": dataclasses.asdict(spec) if has_spec else {},
        "has_app": app is not None,
        "created_by_runtime": created_by_runtime,
        "runtime_routes": _runtime_routes(app) if app is not None else [],
        "foreign_exception_handlers": _foreign_exception_handlers(app) if app is not None else [],
        "shadowed_runtime_routes": _shadowed_runtime_routes(app) if app is not None else [],
        "startup": None,
    }
    if created_by_runtime:
        report["startup"] = asyncio.run(_startup_report(app, spec))
    return report


def _exit(code: int) -> None:
    """Remove the scratch working directory, flush, and leave without running exit hooks."""
    try:
        os.chdir(_ORIGINAL_CWD)
    except OSError:
        pass
    shutil.rmtree(_WORKDIR, ignore_errors=True)
    sys.stderr.flush()
    os._exit(code)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        sys.stderr.write("usage: python _conformance_probe.py <module>\n")
        _exit(2)
    _disable_network()
    try:
        report = _build_report(argv[1])
    except BaseException:  # the traceback goes to stderr (redacted), never into the report
        traceback.print_exc(file=sys.stderr)
        _exit(1)
    _REPORT_STREAM.write(json.dumps(report, sort_keys=True) + "\n")
    _REPORT_STREAM.flush()
    _exit(0)


if __name__ == "__main__":
    main(sys.argv)
