"""The shared service app factory and the per-process runtime registry (IF-01, MVP-036).

``create_service_app(spec, lifespan=..., routers=..., title=...)`` builds every service's
FastAPI app the same way:

- ``app.state.spec = spec``;
- ``/docs``, ``/redoc`` and ``/openapi.json`` only when ``spec.docs_enabled`` is True, or when
  it is None and the environment is not strict;
- the uniform exception handlers, ``/live``, ``/ready``, ``/health`` and ``/metrics`` (the
  runtime routes are registered before the service routers, so a router cannot shadow them);
- middleware, outer to inner: CORS when ``spec.edge``, ``RequestContextMiddleware``, (from W1,
  WS-04.6) the rate limiter, then ``RequestBodyLimitMiddleware``, which answers 413
  ``REQUEST_TOO_LARGE`` for a body over ``spec.max_request_body_bytes`` (1 MiB by default).

``ServiceRuntime`` is the registry the lifespan fills: datastore clients, readiness checks,
background tasks and shutdown closers.

Every lifespan is bracketed by the runtime (MVP-332). The canonical shape is
``service_runtime``, which starts the runtime on entry and stops it on every exit: after the
normal shutdown, and also when composition raises before the ``yield`` or the lifespan is
cancelled::

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with service_runtime(app, SPEC) as runtime:
            ...  # build the service from runtime.pg_pool / runtime.redis, register checks,
            ...  # and register every service-owned stop with runtime.register_shutdown
            yield

The ``async with`` is the first and only statement of the lifespan: all composition and the
``yield`` are inside it, and no service stop code follows the ``yield``. Service-owned stops are
closers registered with ``runtime.register_shutdown``; they run in reverse registration order,
before the runtime's own (refresher, Redis, Postgres), and a closer that raises or hangs does
not stop the rest. The only other accepted shape is ``runtime = await
start_service_runtime(app, SPEC)`` immediately followed by a ``try`` (composition and the
``yield`` inside) whose ``finally`` is exactly ``await stop_service_runtime(runtime)``;
``tests/bootstrap/test_service_conformance.py`` accepts these two shapes and nothing else.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import APIRouter, FastAPI

from solace_common.request_context import get_correlation_headers
from solace_infrastructure.runtime.datastores import start_postgres, start_redis
from solace_infrastructure.runtime.health import (
    CheckResult,
    ReadinessReport,
    register_health_routes,
)
from solace_infrastructure.runtime.http import (
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
    cors_allowed_origins,
    install_cors,
    install_exception_handlers,
    spec_is_strict,
)
from solace_infrastructure.runtime.logging import configure_service_logging
from solace_infrastructure.runtime.metrics import (
    BACKGROUND_TASK_UP,
    READINESS_CHECK_OK,
    SERVICE_READY,
    register_metrics_route,
)
from solace_infrastructure.runtime.spec import ServiceRuntimeSpec
from solace_security.production_guards import ProductionGuards

ReadinessCheck = Callable[[], Awaitable[tuple[bool, str]]]
ShutdownCloser = Callable[[], Awaitable[None]]
Lifespan = Callable[[FastAPI], Any]

# A report younger than this is served from the cache; registering a check invalidates it.
READINESS_CACHE_SECONDS = 2.0
# Each readiness check gets this long; a check that takes longer reports "TimeoutError".
READINESS_CHECK_TIMEOUT_SECONDS = 2.0
# The readiness refresher re-evaluates readiness this often, so solace_service_ready and
# solace_readiness_check_ok stay current without /ready traffic.
READINESS_REFRESH_SECONDS = 15.0
# stop_service_runtime gives each closer this long, then moves on to the next one.
SHUTDOWN_STEP_TIMEOUT_SECONDS = 5.0
# A closer cancelled for overrunning its step gets this long to finish its cancellation
# clean-up (its finally blocks) before the next closer starts.
SHUTDOWN_CANCEL_GRACE_SECONDS = 1.0

READINESS_REFRESHER_TASK = "readiness_refresher"

# A check's returned detail is shown on /ready, so it is capped and any URL is removed.
_MAX_DETAIL_LENGTH = 200
_URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S*")
_URL_PLACEHOLDER = "<redacted-url>"

_log = structlog.get_logger("solace_infrastructure.runtime.bootstrap")


@dataclass(frozen=True)
class _RegisteredCheck:
    check: ReadinessCheck
    critical: bool


def _safe_detail(detail: Any) -> str:
    text = "" if detail is None else str(detail)
    return _URL_PATTERN.sub(_URL_PLACEHOLDER, text)[:_MAX_DETAIL_LENGTH]


class ServiceRuntime:
    """What one service process started, and what ``/live`` and ``/ready`` report on.

    The component attributes start as ``None`` and are set by ``start_service_runtime``:
    ``pg_pool`` (asyncpg pool) and ``redis`` (redis.asyncio client), each left ``None`` when
    the spec does not use it or it failed to start outside strict; from W1 to W3 also
    ``event_plane``, ``service_tokens`` and ``phi_codec``.
    """

    def __init__(self, spec: ServiceRuntimeSpec) -> None:
        self.spec = spec
        self.pg_pool: Any = None
        self.redis: Any = None
        self.event_plane: Any = None
        self.service_tokens: Any = None
        self.phi_codec: Any = None
        self._readiness_checks: dict[str, _RegisteredCheck] = {}
        self._background_tasks: dict[str, asyncio.Task[Any]] = {}
        self._shutdown_closers: list[tuple[str, ShutdownCloser]] = []
        self._readiness_cache: tuple[float, ReadinessReport] | None = None
        self._readiness_generation = 0
        # Concurrent /ready calls share one evaluation instead of each probing every check.
        self._readiness_lock = asyncio.Lock()
        # The app whose state holds this runtime (set by start_service_runtime), and whether
        # stop_service_runtime has begun: from then on /ready answers 503.
        self._app: FastAPI | None = None
        self._stopping = False

    # ----------------------------------------------------------------- registration

    def register_readiness_check(
        self, name: str, check: ReadinessCheck, *, critical: bool = True
    ) -> None:
        """Register (or replace) a readiness check. A failing critical check makes /ready 503.

        Critical checks cover the service's own datastores, broker and event plane; peer
        services are registered with ``critical=False``.
        """
        if not name:
            raise ValueError("a readiness check needs a name")
        self._readiness_checks[name] = _RegisteredCheck(check=check, critical=critical)
        self._invalidate_readiness()

    def register_background_task(self, name: str, task: asyncio.Task[Any]) -> None:
        """Register a long-running task; ``/live`` reports 503 once it has exited.

        ``stop_service_runtime`` cancels the task and waits for it, as a shutdown step named
        ``name``, so the task never outlives the datastores it uses. The step is registered
        here, so it runs before every closer registered earlier (the runtime's Redis and
        Postgres closers included) and after every closer registered later: register a
        graceful stop for the task (``register_shutdown``) after the task, so the graceful stop
        runs first and the cancel only mops up.

        Raises:
            TypeError: when ``task`` is not an ``asyncio.Future`` (an ``asyncio.Task`` is
                one), for example ``None`` from awaiting the task or a bare coroutine.

        A task that has already exited is reported dead and logged straight away.
        """
        if not name:
            raise ValueError("a background task needs a name")
        if not isinstance(task, asyncio.Future):
            raise TypeError(
                f"register_background_task({name!r}) needs an asyncio.Task, "
                f"got {type(task).__name__}"
            )
        self._background_tasks[name] = task
        gauge = BACKGROUND_TASK_UP.labels(service=self.spec.service_name, task=name)
        done_callback = self._on_background_task_done(name)
        if task.done():
            done_callback(task)  # sets the gauge to 0 and logs the exit
            return
        gauge.set(1)
        task.add_done_callback(done_callback)
        self.register_shutdown(name, _task_canceller(task))

    def register_shutdown(self, name: str, closer: ShutdownCloser) -> None:
        """Register a closer; ``stop_service_runtime`` runs closers in reverse order."""
        if not name:
            raise ValueError("a shutdown closer needs a name")
        self._shutdown_closers.append((name, closer))

    def _pop_shutdown_closer(self) -> tuple[str, ShutdownCloser] | None:
        """Remove and return the last-registered closer, or None when none is left.

        Taking one closer at a time keeps the ones not yet reached registered, so a stop that
        is cancelled part-way can be finished by a later call.
        """
        return self._shutdown_closers.pop() if self._shutdown_closers else None

    def outbound_headers(self) -> dict[str, str]:
        """The correlation headers (request, correlation and turn id) for an outbound call."""
        return get_correlation_headers()

    # ----------------------------------------------------------------- liveness

    def dead_background_task(self) -> str | None:
        """Name of the first registered background task that has exited, else ``None``.

        Also refreshes ``solace_background_task_up`` for every registered task.
        """
        dead: str | None = None
        for name, task in self._background_tasks.items():
            up = not task.done()
            BACKGROUND_TASK_UP.labels(service=self.spec.service_name, task=name).set(
                1 if up else 0
            )
            if not up and dead is None:
                dead = name
        return dead

    def _on_background_task_done(self, name: str) -> Callable[[asyncio.Task[Any]], None]:
        def _done(task: asyncio.Task[Any]) -> None:
            if self._background_tasks.get(name) is not task:
                return  # replaced by a later registration under the same name
            BACKGROUND_TASK_UP.labels(service=self.spec.service_name, task=name).set(0)
            if task.cancelled():
                _log.info("background_task_exited", task=name, outcome="cancelled")
                return
            exc = task.exception()
            if exc is None:
                _log.warning("background_task_exited", task=name, outcome="returned")
            else:
                _log.error(
                    "background_task_exited",
                    task=name,
                    outcome="error",
                    error_type=type(exc).__name__,
                    exc_info=exc,
                )

        return _done

    # ----------------------------------------------------------------- readiness

    async def evaluate_readiness(self) -> ReadinessReport:
        """Run every readiness check concurrently (each with a timeout) and report.

        A report is cached for ``READINESS_CACHE_SECONDS``. A failed check's ``detail`` is
        the exception class name only. Sets ``solace_readiness_check_ok`` and
        ``solace_service_ready``. Once ``stop_service_runtime`` has begun, no check runs: the
        report is not ready (no checks) and ``solace_service_ready`` is 0.
        """
        if self._stopping:
            return self._stopping_report()
        cached = self._cached_report()
        if cached is not None:
            return cached
        async with self._readiness_lock:
            cached = self._cached_report()
            if cached is not None:
                return cached
            generation = self._readiness_generation
            checks = list(self._readiness_checks.items())
            results = await asyncio.gather(*(self._run_check(entry) for _, entry in checks))
            by_name = {name: result for (name, _), result in zip(checks, results, strict=True)}
            report = ReadinessReport(
                ready=all(result.ok for result in results if result.critical),
                checks=by_name,
            )
            self._publish(report)
            if generation == self._readiness_generation:
                self._readiness_cache = (time.monotonic(), report)
            return report

    def _cached_report(self) -> ReadinessReport | None:
        cached = self._readiness_cache
        if cached is None:
            return None
        stored_at, report = cached
        if time.monotonic() - stored_at >= READINESS_CACHE_SECONDS:
            return None
        return report

    def _invalidate_readiness(self) -> None:
        self._readiness_generation += 1
        self._readiness_cache = None

    @staticmethod
    async def _run_check(entry: _RegisteredCheck) -> CheckResult:
        try:
            ok, detail = await asyncio.wait_for(
                entry.check(), timeout=READINESS_CHECK_TIMEOUT_SECONDS
            )
        except Exception as exc:  # the class name only: the text may hold a DSN or secret
            return CheckResult(ok=False, critical=entry.critical, detail=type(exc).__name__)
        return CheckResult(ok=bool(ok), critical=entry.critical, detail=_safe_detail(detail))

    def _publish(self, report: ReadinessReport) -> None:
        service = self.spec.service_name
        for name, result in report.checks.items():
            READINESS_CHECK_OK.labels(service=service, check=name).set(1 if result.ok else 0)
        SERVICE_READY.labels(service=service).set(1 if report.ready else 0)

    def _stopping_report(self) -> ReadinessReport:
        SERVICE_READY.labels(service=self.spec.service_name).set(0)
        return ReadinessReport(ready=False, checks={})

    # ----------------------------------------------------------------- lifecycle

    def _attach(self, app: FastAPI) -> None:
        """Publish this runtime as ``app.state.runtime`` (what /live and /ready read)."""
        self._app = app
        app.state.runtime = self

    def _begin_stopping(self) -> None:
        """Mark the runtime stopping: /ready answers 503 from here on (the drain)."""
        self._stopping = True
        self._invalidate_readiness()
        SERVICE_READY.labels(service=self.spec.service_name).set(0)

    def _detach(self) -> None:
        """Remove this runtime from ``app.state`` (only if it is still the one published there)."""
        app, self._app = self._app, None
        if app is not None and getattr(app.state, "runtime", None) is self:
            del app.state.runtime


def _task_canceller(task: asyncio.Future[Any]) -> ShutdownCloser:
    """The shutdown step of a registered background task: cancel it and wait for it to end."""

    async def _cancel_and_wait() -> None:
        task.cancel()
        await asyncio.wait({task})

    return _cancel_and_wait


def _docs_enabled(spec: ServiceRuntimeSpec) -> bool:
    if spec.docs_enabled is not None:
        return spec.docs_enabled
    return not spec_is_strict(spec)


def create_service_app(
    spec: ServiceRuntimeSpec,
    *,
    lifespan: Lifespan,
    routers: Sequence[tuple[APIRouter, str]],
    title: str,
) -> FastAPI:
    """Build a service's FastAPI app with the shared routes, handlers and middleware.

    Raises:
        StartupDependencyError: for an edge service in a strict environment when
            ``CORS_ALLOWED_ORIGINS`` is empty or holds ``*``.
    """
    # Resolve the CORS policy first, so a bad strict configuration fails before any app exists.
    cors_origins = cors_allowed_origins(spec) if spec.edge else None
    docs = _docs_enabled(spec)

    # FastAPI's debug flag stays off in every environment: it renders tracebacks to clients.
    app = FastAPI(
        title=title,
        version=spec.version,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    app.state.spec = spec

    install_exception_handlers(app, spec)
    register_health_routes(app)
    register_metrics_route(app)
    for router, prefix in routers:
        app.include_router(router, prefix=prefix)

    # Starlette puts each added middleware outside the ones added before it, so add them
    # innermost first. The body limit is innermost: inside the request context, so its 413
    # carries the request ids. WS-04.6 (W1): install_rate_limiting(app, spec) when
    # spec.rate_limited goes between the two, so the limiter also runs inside the request context.
    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=spec.max_request_body_bytes)
    app.add_middleware(RequestContextMiddleware, service_name=spec.service_name)
    if cors_origins is not None:
        install_cors(app, cors_origins)
    return app


def required_secrets_for(spec: ServiceRuntimeSpec) -> tuple[str, ...]:
    """The variables a strict environment must provide for this spec (ProductionGuards).

    ``AUTH_SECRET_KEY`` always; ``POSTGRES_PASSWORD`` when the spec uses Postgres; ``REDIS_URL``
    when it uses Redis; ``ENCRYPTION_MASTER_KEY`` and ``ENCRYPTION_SEARCH_HASH_SALT`` when it
    handles PHI.
    """
    names = ["AUTH_SECRET_KEY"]
    if spec.uses_postgres:
        names.append("POSTGRES_PASSWORD")
    if spec.uses_redis:
        names.append("REDIS_URL")
    if spec.handles_phi:
        names.extend(("ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT"))
    return tuple(names)


async def start_service_runtime(app: FastAPI, spec: ServiceRuntimeSpec) -> ServiceRuntime:
    """Start what ``spec`` declares and return the runtime; the first statement of a lifespan.

    Order:

    1. ``ProductionGuards.assert_startup(spec.environment, service_name=...,
       required_vars=required_secrets_for(spec))``;
    2. ``configure_service_logging(spec)``;
    3. the Postgres ``default`` pool when ``spec.uses_postgres`` (plus the schema-head check in
       strict), then Redis from ``REDIS_URL`` when ``spec.uses_redis`` (``datastores``);
    4. the readiness refresher: a registered background task that re-evaluates readiness every
       ``READINESS_REFRESH_SECONDS``, so its death shows on ``/live``.

    Sets ``app.state.runtime`` once everything has started. In a strict environment a component
    that fails to start raises ``StartupDependencyError``, so the process exits non-zero;
    elsewhere the component is logged as degraded and its readiness check fails (``/ready``
    503). If start-up fails part-way, whatever already started is stopped before the error
    propagates. That clean-up is shielded: when start-up is cancelled, the cancellation
    propagates at once and the clean-up still runs to the end.

    Raises:
        ProductionSecurityError: when the strict start-up guards find a violation.
        StartupDependencyError: from ``configure_service_logging``, or in strict when Postgres
            (pool or schema head) or Redis fails to start.
    """
    ProductionGuards.assert_startup(
        spec.environment,
        service_name=spec.service_name,
        required_vars=required_secrets_for(spec),
    )
    configure_service_logging(spec)
    runtime = ServiceRuntime(spec)
    try:
        if spec.uses_postgres:
            runtime.pg_pool = await start_postgres(spec, runtime)
        if spec.uses_redis:
            runtime.redis = await start_redis(spec, runtime)
        _start_readiness_refresher(runtime)
    except BaseException:
        # No caller holds this runtime, so a later stop could never finish an interrupted
        # clean-up: shield it, so the closers all run even when start-up itself is cancelled.
        await asyncio.shield(stop_service_runtime(runtime))
        raise
    runtime._attach(app)
    _log.info(
        "service_runtime_started",
        service=spec.service_name,
        environment=spec.environment,
        strict=spec_is_strict(spec),
    )
    return runtime


def _start_readiness_refresher(runtime: ServiceRuntime) -> None:
    task = asyncio.create_task(
        _refresh_readiness(runtime),
        name=f"{runtime.spec.service_name}:{READINESS_REFRESHER_TASK}",
    )
    # Also registers the refresher's shutdown step (cancel it and wait for it).
    runtime.register_background_task(READINESS_REFRESHER_TASK, task)


async def _refresh_readiness(runtime: ServiceRuntime) -> None:
    # A failed evaluation is logged and the loop goes on: the refresher's death means /live 503
    # and a container restart, which a readiness bug must not cause.
    while True:
        await asyncio.sleep(READINESS_REFRESH_SECONDS)
        try:
            await runtime.evaluate_readiness()
        except Exception as exc:
            _log.warning("readiness_refresh_failed", error=type(exc).__name__)


async def stop_service_runtime(runtime: ServiceRuntime) -> None:
    """Stop the runtime: mark it stopping, run the closers in reverse order, then detach it.

    ``service_runtime`` calls this on every exit of the lifespan. From the moment it begins,
    ``/ready`` answers 503 and ``solace_service_ready`` is 0 (the drain). Registered background
    tasks are cancelled and awaited as steps of their own (``register_background_task``). Once
    every closer has run, the runtime is removed from ``app.state``, so ``/ready`` stays 503.

    Each closer gets ``SHUTDOWN_STEP_TIMEOUT_SECONDS``. A closer that raises or runs out of time
    is logged (step name and exception class, never the text) and skipped, and the next closer
    still runs. A timed-out closer is cancelled and given ``SHUTDOWN_CANCEL_GRACE_SECONDS`` to
    finish its cancellation clean-up before the next closer starts; one that ignores the
    cancellation is left behind after that.

    Each closer runs at most once. Closers are taken one at a time, so when the caller is
    cancelled part-way, the closer that was running is cancelled and the ones not yet reached
    stay registered: a later call runs them. A call after a complete stop does nothing. Only
    cancellation of the caller propagates.
    """
    runtime._begin_stopping()
    failed: list[str] = []
    while (entry := runtime._pop_shutdown_closer()) is not None:
        name, closer = entry
        error = await _run_closer(closer, SHUTDOWN_STEP_TIMEOUT_SECONDS)
        if error is not None:
            failed.append(name)
            _log.warning("runtime_shutdown_step_failed", step=name, error=error)
    runtime._detach()
    _log.info("service_runtime_stopped", service=runtime.spec.service_name, failed_steps=failed)


@asynccontextmanager
async def service_runtime(app: FastAPI, spec: ServiceRuntimeSpec) -> AsyncIterator[ServiceRuntime]:
    """Bracket a lifespan with the runtime: ``async with service_runtime(app, SPEC) as runtime:``.

    Awaits ``start_service_runtime(app, spec)`` on entry and yields the runtime. On exit,
    ``stop_service_runtime(runtime)`` runs in a ``finally``: after the lifespan's normal
    shutdown, and also when the composition inside the block raises (before or after the
    lifespan's ``yield``) or the lifespan is cancelled. The exception or cancellation then
    propagates unchanged. If start-up itself fails, ``start_service_runtime`` has already
    stopped what it started, so there is nothing to stop here.

    Register every service-owned stop with ``runtime.register_shutdown`` instead of writing it
    after the ``yield``: closers are isolated, so one that raises or hangs never stops the
    Redis and Postgres closers from running.
    """
    runtime = await start_service_runtime(app, spec)
    try:
        yield runtime
    finally:
        await stop_service_runtime(runtime)


async def _run_closer(closer: ShutdownCloser, timeout: float) -> str | None:
    """Run one closer with a hard time bound; return the failure's class name, else None."""
    try:
        task = asyncio.ensure_future(closer())
    except Exception as exc:  # the closer raised before returning, or returned no awaitable
        return type(exc).__name__
    task.add_done_callback(_consume_outcome)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not done:
        # Cancel, then give the cancellation a bounded grace, so the closer's own clean-up (its
        # finally blocks, which may still need a datastore a later closer shuts) finishes before
        # the next closer starts. A closer that ignores the cancellation holds up the rest by
        # no more than the grace.
        task.cancel()
        await asyncio.wait({task}, timeout=SHUTDOWN_CANCEL_GRACE_SECONDS)
        return "TimeoutError"
    if task.cancelled():
        return "CancelledError"
    exc = task.exception()
    return None if exc is None else type(exc).__name__


def _consume_outcome(task: asyncio.Future[Any]) -> None:
    # Retrieve the outcome so an abandoned closer never logs "exception was never retrieved",
    # whose text could carry a DSN or secret.
    if not task.cancelled():
        task.exception()


__all__ = [
    "READINESS_CACHE_SECONDS",
    "READINESS_CHECK_TIMEOUT_SECONDS",
    "READINESS_REFRESHER_TASK",
    "READINESS_REFRESH_SECONDS",
    "SHUTDOWN_CANCEL_GRACE_SECONDS",
    "SHUTDOWN_STEP_TIMEOUT_SECONDS",
    "ServiceRuntime",
    "create_service_app",
    "required_secrets_for",
    "service_runtime",
    "start_service_runtime",
    "stop_service_runtime",
]
