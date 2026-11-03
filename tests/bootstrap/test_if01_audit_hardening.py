"""WS-04.F1: IF-01 hardening from the integrated audit (MVP-036, MVP-180, MVP-332).

One group of tests per surviving finding of the IF-01 integrated audit: LIF-1 to LIF-5, PHI-1
to PHI-4 and HTT-2. Each reproduces the audit's failure scenario in process, with fake
datastores (the ``fakes`` fixture) or an app built by ``create_service_app`` and driven with
``TestClient``, ``httpx.ASGITransport`` or raw ASGI messages. Every value is synthetic.
"""
from __future__ import annotations

import ast
import asyncio
import enum
import functools
import gc
import importlib.util
import json
import logging
import textwrap
import time
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from pydantic import BaseModel, ConfigDict, ValidationError

from solace_infrastructure.runtime import (
    ServiceRuntimeSpec,
    create_service_app,
    start_service_runtime,
    stop_service_runtime,
)
from solace_infrastructure.runtime import bootstrap as bs
from solace_infrastructure.runtime.bootstrap import ServiceRuntime
from solace_infrastructure.runtime.logging import configure_service_logging

BOOTSTRAP_TESTS = Path(__file__).resolve().parent

# Patient free text the regex PHI sanitizer cannot recognise, and shapes it can.
PATIENT_TEXT = "Margaret Hollis reports chest pain"
PATIENT_WORDS = ("Margaret", "Hollis", "chest pain")
EMAIL = "jane.doe@example.com"
PHONE = "555-867-5309"

DEFAULT_BODY_LIMIT = 1_048_576


# --------------------------------------------------------------------------- shared helpers


def _spec(env: str = "development", **overrides: Any) -> ServiceRuntimeSpec:
    fields: dict[str, Any] = {"uses_postgres": True, "uses_redis": True}
    fields.update(overrides)
    return ServiceRuntimeSpec(service_name="probe-service", environment=env, **fields)


def _strict_spec() -> ServiceRuntimeSpec:
    return ServiceRuntimeSpec(service_name="safety-service", environment="production")


def _service_runtime():
    # Imported per test, not at module level, so that before LIF-1 lands only its cases fail.
    from solace_infrastructure.runtime import service_runtime

    return service_runtime


def _live_refreshers() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks()
            if t.get_name().endswith(bs.READINESS_REFRESHER_TASK) and not t.done()]


def _closer_calls(calls: list) -> list:
    return [c for c in calls if c in ("svc.stop", "redis.aclose", "pg.close_all")]


def _json_lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _events(out: str, event: str) -> list[dict[str, Any]]:
    return [line for line in _json_lines(out) if line.get("event") == event]


def _only(lines: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(lines) == 1, lines
    return lines[0]


def _assert_no_patient_text(captured: Any) -> None:
    text = captured.out + captured.err
    for word in PATIENT_WORDS:
        assert word not in text, f"patient text {word!r} reached the log output"


# --------------------------------------------------------------------------- LIF-1
# Once start_service_runtime returned, nothing rolled the runtime back: a composition failure
# before the yield, a raising service stop or a cancelled lifespan skipped every closer.


class _ConsumerStartFailed(Exception):
    """A service component that fails between start-up and the yield (a Kafka consumer, say)."""


async def _start_consumer(runtime: Any) -> None:
    raise _ConsumerStartFailed("consumer could not join its group")


async def test_lif1_composition_failure_inside_service_runtime_runs_every_closer(fakes, calls):
    service_runtime = _service_runtime()
    spec = _spec()

    @asynccontextmanager
    async def lifespan(app):
        async with service_runtime(app, spec) as runtime:
            assert app.state.runtime is runtime
            await _start_consumer(runtime)
            yield

    app = create_service_app(spec, lifespan=lifespan, routers=[], title="probe")
    with pytest.raises(_ConsumerStartFailed):
        async with app.router.lifespan_context(app):
            pass
    assert _closer_calls(calls) == ["redis.aclose", "pg.close_all"]
    assert _live_refreshers() == []


async def test_lif1_a_failing_service_stop_does_not_skip_the_datastore_closers(fakes, calls):
    service_runtime = _service_runtime()
    spec = _spec()

    async def consumer_stop():
        calls.append("svc.stop")
        raise ConnectionError("the broker is going away")

    @asynccontextmanager
    async def lifespan(app):
        async with service_runtime(app, spec) as runtime:
            runtime.register_shutdown("consumer", consumer_stop)
            yield

    app = create_service_app(spec, lifespan=lifespan, routers=[], title="probe")
    async with app.router.lifespan_context(app):
        assert _live_refreshers() != []
    assert _closer_calls(calls) == ["svc.stop", "redis.aclose", "pg.close_all"]
    assert _live_refreshers() == []


async def test_lif1_a_cancelled_lifespan_still_runs_every_closer(fakes, calls):
    service_runtime = _service_runtime()
    entered = asyncio.Event()

    async def serve():
        async with service_runtime(FastAPI(), _spec()):
            entered.set()
            await asyncio.sleep(30)  # serving; cancelled at loop teardown (a dev Ctrl-C)

    task = asyncio.create_task(serve())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _closer_calls(calls) == ["redis.aclose", "pg.close_all"]
    assert _live_refreshers() == []


def test_lif1_service_runtime_is_exported_by_the_runtime_package():
    import solace_infrastructure.runtime as runtime_package

    assert "service_runtime" in runtime_package.__all__
    assert _service_runtime() is bs.service_runtime


@functools.cache
def _conformance():
    """tests/bootstrap/test_service_conformance.py, loaded by path (it is not a package)."""
    path = BOOTSTRAP_TESTS / "test_service_conformance.py"
    spec = importlib.util.spec_from_file_location("_ws04f1_service_conformance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SNIPPET_HEADER = (
    "from contextlib import asynccontextmanager\n"
    "from solace_infrastructure.runtime import (\n"
    "    create_service_app, service_runtime, start_service_runtime, stop_service_runtime)\n\n"
)
_SNIPPET_FOOTER = '\napp = create_service_app(SPEC, lifespan=lifespan, routers=[], title="svc")\n'

# The shape the bootstrap docstring used to show: service stop code inline, no try/finally.
BARE_LIFESPAN = """
@asynccontextmanager
async def lifespan(app):
    runtime = await start_service_runtime(app, SPEC)
    consumer = await start_consumer(runtime)
    yield
    await consumer.stop()
    await stop_service_runtime(runtime)
"""

# Form (a), the canonical lifespan.
ASYNC_WITH_LIFESPAN = '''
@asynccontextmanager
async def lifespan(app):
    """Form (a)."""
    async with service_runtime(app, SPEC) as runtime:
        consumer = await start_consumer(runtime)
        runtime.register_shutdown("consumer", consumer.stop)
        yield
'''

# Form (b).
TRY_FINALLY_LIFESPAN = """
@asynccontextmanager
async def lifespan(app):
    runtime = await start_service_runtime(app, SPEC)
    try:
        consumer = await start_consumer(runtime)
        runtime.register_shutdown("consumer", consumer.stop)
        yield
    finally:
        await stop_service_runtime(runtime)
"""

NEAR_MISSES = {
    "composition_before_the_try": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            consumer = await start_consumer(runtime)
            try:
                yield
            finally:
                await stop_service_runtime(runtime)
        """,
    "service_stop_in_the_finally": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                consumer = await start_consumer(runtime)
                yield
            finally:
                await consumer.stop()
                await stop_service_runtime(runtime)
        """,
    "yield_outside_the_try": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                consumer = await start_consumer(runtime)
            finally:
                await stop_service_runtime(runtime)
            yield
        """,
    "yield_after_the_async_with": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime:
                consumer = await start_consumer(runtime)
            yield
        """,
    "statement_after_the_async_with": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime:
                yield
            await flush_metrics()
        """,
    "second_context_manager_in_the_same_with": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime, consumer_group() as group:
                yield
        """,
    # WS-04.F1 fix round 1 (WS04F1-LIF-1): the bare yield is the bracket body's last statement.
    "stop_code_after_the_yield_in_form_a": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime:
                consumer = await start_consumer(runtime)
                yield
                await consumer.stop()
        """,
    "stop_code_after_the_yield_in_form_b": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                consumer = await start_consumer(runtime)
                yield
                await consumer.stop()
            finally:
                await stop_service_runtime(runtime)
        """,
    "yield_inside_a_nested_block_in_form_a": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime:
                if runtime is not None:
                    yield
        """,
    "yield_with_a_value_in_form_a": """
        @asynccontextmanager
        async def lifespan(app):
            async with service_runtime(app, SPEC) as runtime:
                yield {"consumer": None}
        """,
    "two_yields_in_form_b": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                yield
                yield
            finally:
                await stop_service_runtime(runtime)
        """,
    # WS-04.F1 fix round 2 (WS04F1F1-LIF-1): form (b)'s try has no except and no else clause.
    "stop_code_in_the_else_of_form_b": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                consumer = await start_consumer(runtime)
                yield
            except Exception:
                raise
            else:
                await consumer.stop()
            finally:
                await stop_service_runtime(runtime)
        """,
    "stop_code_in_an_except_of_form_b": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                consumer = await start_consumer(runtime)
                yield
            except BaseException:
                await consumer.stop()
                raise
            finally:
                await stop_service_runtime(runtime)
        """,
    "yield_in_an_except_of_form_b": """
        @asynccontextmanager
        async def lifespan(app):
            runtime = await start_service_runtime(app, SPEC)
            try:
                yield
            except Exception:
                yield
            finally:
                await stop_service_runtime(runtime)
        """,
}


def _check_lifespan(tmp_path: Path, lifespan_source: str) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        _SNIPPET_HEADER + textwrap.dedent(lifespan_source) + _SNIPPET_FOOTER, encoding="utf-8"
    )
    _conformance()._assert_lifespan_bracketed(main_py)


def test_lif1_conformance_rejects_the_bare_lifespan_shape(tmp_path):
    with pytest.raises(AssertionError):
        _check_lifespan(tmp_path, BARE_LIFESPAN)


@pytest.mark.parametrize("source", [ASYNC_WITH_LIFESPAN, TRY_FINALLY_LIFESPAN],
                         ids=["form_a_async_with", "form_b_try_finally"])
def test_lif1_conformance_accepts_the_two_bracketed_forms(tmp_path, source):
    _check_lifespan(tmp_path, source)


@pytest.mark.parametrize("source", list(NEAR_MISSES.values()), ids=list(NEAR_MISSES))
def test_lif1_conformance_rejects_near_misses(tmp_path, source):
    with pytest.raises(AssertionError):
        _check_lifespan(tmp_path, source)


def test_lif1_reference_app_and_bootstrap_docstring_show_form_a():
    tree = ast.parse((BOOTSTRAP_TESTS / "fixtures" / "reference_app.py").read_text(encoding="utf-8"))
    lifespan = _conformance()._lifespan_function(tree)
    assert len(lifespan.body) == 1 and isinstance(lifespan.body[0], ast.AsyncWith)
    assert ast.unparse(lifespan.body[0].items[0]) == "service_runtime(app, SPEC) as runtime"
    assert "async with service_runtime(app, SPEC) as runtime:" in bs.__doc__


# --------------------------------------------------------------------------- LIF-2
# register_background_task only tracked a task for /live: stop_service_runtime left it running
# after the Redis and Postgres closers had run.


async def test_lif2_a_registered_task_is_cancelled_and_awaited_before_the_datastore_closers(
        fakes, calls):
    rt = await start_service_runtime(FastAPI(), _spec())

    async def consumer():
        try:
            while True:
                calls.append("consumer.tick")
                await asyncio.sleep(0.005)
        finally:
            await asyncio.sleep(0.02)  # clean-up that still needs the pool (commit offsets, say)
            calls.append("consumer.cleanup_done")

    task = asyncio.create_task(consumer(), name="probe-consumer")
    try:
        rt.register_background_task("consumer", task)
        await asyncio.sleep(0.03)
        await stop_service_runtime(rt)
        assert task.done()
        ticks = calls.count("consumer.tick")
        await asyncio.sleep(0.05)
        assert calls.count("consumer.tick") == ticks
        tail = [c for c in calls if c in ("consumer.cleanup_done", "redis.aclose", "pg.close_all")]
        assert tail == ["consumer.cleanup_done", "redis.aclose", "pg.close_all"]
    finally:
        task.cancel()
        await asyncio.wait({task})


# --------------------------------------------------------------------------- LIF-3
# A closer that overran SHUTDOWN_STEP_TIMEOUT_SECONDS was cancelled and not awaited, so its
# cancellation clean-up ran alongside the later closers, even after stop had returned.


async def test_lif3_a_timed_out_closer_finishes_its_cleanup_before_the_next_closer(
        fakes, calls, monkeypatch):
    monkeypatch.setattr(bs, "SHUTDOWN_STEP_TIMEOUT_SECONDS", 0.2)
    rt = await start_service_runtime(FastAPI(), _spec())

    async def event_plane_stop():
        calls.append("plane.stop.begin")
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.05)  # releasing outbox claims needs the pool
            calls.append("plane.stop.cleanup_done")

    rt.register_shutdown("event_plane", event_plane_stop)
    await stop_service_runtime(rt)
    calls.append("stop_returned")
    order = [c for c in calls if c in ("plane.stop.begin", "plane.stop.cleanup_done",
                                       "redis.aclose", "pg.close_all", "stop_returned")]
    assert order == ["plane.stop.begin", "plane.stop.cleanup_done", "redis.aclose",
                     "pg.close_all", "stop_returned"]


async def test_lif3_a_closer_that_ignores_cancellation_is_left_after_a_bounded_grace(
        fakes, calls, monkeypatch):
    assert bs.SHUTDOWN_CANCEL_GRACE_SECONDS == 1.0
    monkeypatch.setattr(bs, "SHUTDOWN_STEP_TIMEOUT_SECONDS", 0.2)
    rt = await start_service_runtime(FastAPI(), _spec())
    release = asyncio.Event()

    async def stubborn():
        while not release.is_set():
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                continue  # swallows the cancellation

    rt.register_shutdown("stubborn", stubborn)
    started = time.monotonic()
    await stop_service_runtime(rt)
    elapsed = time.monotonic() - started
    release.set()
    assert 0.2 + 1.0 - 0.05 <= elapsed < 0.2 + 1.0 + 0.8
    assert _closer_calls(calls) == ["redis.aclose", "pg.close_all"]
    await asyncio.sleep(0.1)  # let the stubborn closer see the release and end


# --------------------------------------------------------------------------- LIF-4
# A Postgres close cut off by the step timeout left the closed pool in ConnectionPoolManager's
# process-wide registry, so the next lifespan in the same process got it back from get_pool.


class _CheckedOutPool:
    """Closes like asyncpg's Pool while a connection is checked out.

    ``close()`` waits for the connection to be released (it never is); a cancelled close
    terminates the pool, as asyncpg's does.
    """

    def __init__(self, calls: list, *, hangs: bool = True) -> None:
        self.calls = calls
        self.hangs = hangs
        self.closed = False

    async def close(self) -> None:
        self.calls.append("pool.close")
        if not self.hangs:
            self.closed = True
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.terminate()
            raise

    def terminate(self) -> None:
        if self.closed:  # as asyncpg's: a closed pool returns at once
            return
        self.closed = True
        self.calls.append("pool.terminate")

    async def fetchval(self, sql: str) -> int:
        if self.closed:
            raise RuntimeError("pool is closed")
        return 1


async def test_lif4_a_timed_out_postgres_close_leaves_no_closed_pool_in_the_registry(
        monkeypatch, tmp_path, calls):
    from solace_infrastructure.database import connection_manager as cm

    manager = cm.ConnectionPoolManager
    monkeypatch.setattr(bs.ProductionGuards, "assert_startup",
                        classmethod(lambda cls, env, **kw: None))
    monkeypatch.setattr(bs, "configure_service_logging", lambda spec: None)
    monkeypatch.setattr(bs, "SHUTDOWN_STEP_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(manager, "_pools", {})
    monkeypatch.setattr(manager, "_pool_configs", {})
    monkeypatch.setattr(manager, "_locks", {})
    spec = ServiceRuntimeSpec(service_name="memory-service", environment="development",
                              uses_postgres=True)

    first = _CheckedOutPool(calls)
    manager._pools["default"] = first  # what the first lifespan's get_pool created
    manager._pool_configs["default"] = object()
    rt = await start_service_runtime(FastAPI(), spec)
    assert rt.pg_pool is first
    await stop_service_runtime(rt)
    await asyncio.sleep(0.05)  # LIF-3 aside: let the cut-off close finish its cancellation
    assert first.closed and calls.count("pool.terminate") == 1
    assert "default" not in manager._pools
    assert manager._pool_configs == {} and manager._locks == {}

    # A second lifespan in the same process builds a new pool instead of reusing the closed one.
    second = _CheckedOutPool(calls, hangs=False)

    async def create_pool(**kwargs):
        calls.append("create_pool")
        return second

    monkeypatch.chdir(tmp_path)  # PostgresSettings also reads .env from the working directory
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-only-pw")
    monkeypatch.setattr(cm.asyncpg, "create_pool", create_pool)
    rt2 = await start_service_runtime(FastAPI(), spec)
    try:
        assert rt2.pg_pool is second and "create_pool" in calls
    finally:
        await stop_service_runtime(rt2)


# --------------------------------------------------------------------------- LIF-5
# stop_service_runtime neither marked nor detached the runtime, so /ready answered 200 with
# postgres and redis "connected" during the drain and after both were closed.


async def test_lif5_ready_is_503_while_stopping_and_after_stop(fakes, calls):
    from solace_infrastructure.runtime.metrics import SERVICE_READY

    spec = _spec()

    @asynccontextmanager
    async def lifespan(app):
        yield

    app = create_service_app(spec, lifespan=lifespan, routers=[], title="probe")
    rt = await start_service_runtime(app, spec)
    during: list[int] = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://probe") as client:
        assert (await client.get("/ready")).status_code == 200
        assert REGISTRY.get_sample_value("solace_service_ready", {"service": "probe-service"}) == 1.0

        async def drain_probe():
            during.append((await client.get("/ready")).status_code)
            during.append(int(REGISTRY.get_sample_value(
                "solace_service_ready", {"service": "probe-service"})))

        rt.register_shutdown("drain_probe", drain_probe)
        await stop_service_runtime(rt)
        after = await client.get("/ready")
    assert during == [503, 0]
    assert after.status_code == 503 and after.json()["status"] == "not_ready"
    assert getattr(app.state, "runtime", None) is None
    SERVICE_READY.labels(service="probe-service").set(0)


def test_lif5_ready_after_a_testclient_lifespan_is_503(fakes, calls):
    spec = _spec()

    @asynccontextmanager
    async def lifespan(app):
        runtime = await start_service_runtime(app, spec)
        try:
            yield
        finally:
            await stop_service_runtime(runtime)

    app = create_service_app(spec, lifespan=lifespan, routers=[], title="probe")
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
    response = TestClient(app).get("/ready")  # no lifespan: the runtime was torn down
    assert response.status_code == 503 and response.json()["checks"] == {}
    assert getattr(app.state, "runtime", None) is None


# --------------------------------------------------------------------------- PHI-1
# A pydantic ValidationError was kept out of the log only as the top-level exception of a 500;
# in a cause, context or ExceptionGroup member its input_value text was logged.

phi_router = APIRouter()


class Age(BaseModel):
    age: int


class NoteIn(BaseModel):
    text: str


class StrictRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int


# A stored row whose extra key is client-chosen text shaped like an identifier.
STORED_ROW = {"score": 1, "Margaret_Hollis_HIV_positive": True}


def _parse_age(text: str) -> Age:
    return Age(age=text)  # user text that is not a number


@phi_router.post("/chained")
async def chained(body: NoteIn):
    try:
        _parse_age(body.text)
    except ValidationError as exc:
        raise RuntimeError("repository row could not be loaded") from exc


@phi_router.post("/implicit")
async def implicit(body: NoteIn):
    try:
        _parse_age(body.text)
    except ValidationError:
        raise KeyError("row")  # noqa: B904 - the implicit __context__ is the point


@phi_router.post("/group")
async def group(body: NoteIn):
    async def load():
        _parse_age(body.text)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(load())


@phi_router.get("/stored-extra")
async def stored_extra():
    return StrictRow.model_validate(STORED_ROW)


@phi_router.get("/stored-extra-chained")
async def stored_extra_chained():
    try:
        StrictRow.model_validate(STORED_ROW)
    except ValidationError as exc:
        raise RuntimeError("stored row rejected") from exc


def _phi_app() -> FastAPI:
    spec = _strict_spec()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = ServiceRuntime(spec)
        yield

    return create_service_app(spec, lifespan=lifespan, routers=[(phi_router, "/api/v1/p")],
                              title="phi")


AGE_ERRORS = [{"type": "int_parsing", "field": "age"}]


@pytest.mark.parametrize("path", ["/chained", "/implicit", "/group"])
def test_phi1_a_500_whose_exception_chain_holds_a_validation_error_logs_no_input(capsys, path):
    configure_service_logging(_strict_spec())
    with TestClient(_phi_app(), raise_server_exceptions=False) as client:
        response = client.post(f"/api/v1/p{path}", json={"text": PATIENT_TEXT},
                               headers={"X-Request-ID": "req-phi1"})
    assert response.status_code == 500 and response.json()["error"]["code"] == "INTERNAL_ERROR"
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only(_events(captured.out, "unhandled_exception"))
    assert line["request_id"] == "req-phi1" and "exception" not in line
    assert line["validation_errors"] == AGE_ERRORS and line["origin"]


@pytest.mark.parametrize("chained_error", [False, True], ids=["direct", "chained"])
async def test_phi1_a_dead_background_task_logs_no_input(capsys, chained_error):
    configure_service_logging(_strict_spec())

    async def consume_one():  # a W1 consumer that validates a PHI-bearing payload
        if not chained_error:
            _parse_age(PATIENT_TEXT)
        try:
            _parse_age(PATIENT_TEXT)
        except ValidationError as exc:
            raise RuntimeError("payload rejected") from exc

    task = asyncio.create_task(consume_one())
    ServiceRuntime(_strict_spec()).register_background_task("probe_worker", task)
    await asyncio.wait({task})
    await asyncio.sleep(0)
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only([l for l in _events(captured.out, "background_task_exited")
                  if l.get("task") == "probe_worker"])
    assert line["outcome"] == "error" and "exception" not in line
    assert line["error_type"] == ("RuntimeError" if chained_error else "ValidationError")
    assert line["validation_errors"] == AGE_ERRORS


@pytest.mark.parametrize("via", ["stdlib", "structlog"])
def test_phi1_logger_exception_with_a_chained_validation_error_logs_no_input(capsys, via):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    try:
        try:
            _parse_age(PATIENT_TEXT)
        except ValidationError as exc:
            raise RuntimeError("load failed") from exc
    except RuntimeError:
        if via == "stdlib":
            logging.getLogger("svc.repo").exception("load failed")
        else:
            structlog.get_logger("svc.repo").exception("load_failed")
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _json_lines(captured.out)[-1]
    assert line["level"] == "error" and "exception" not in line
    assert line["error_type"] == "RuntimeError" and line["validation_errors"] == AGE_ERRORS
    assert "test_if01_audit_hardening.py" in line["origin"]


def test_phi1_traceback_text_quoting_pydantic_input_is_dropped(capsys):
    # The ValidationError is gone (its text was copied into another exception's message), so
    # only the rendered traceback shows pydantic's input_value= rendering.
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    try:
        _parse_age(PATIENT_TEXT)
    except ValidationError as exc:
        detail = str(exc)
    try:
        raise RuntimeError(f"row rejected: {detail}")
    except RuntimeError:
        structlog.get_logger("svc.repo").exception("row_rejected")
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only(_events(captured.out, "row_rejected"))
    assert "exception" not in line and line["error_type"] == "RuntimeError"
    assert line["validation_errors"] == []


def test_phi1_an_exception_without_pydantic_input_keeps_its_traceback(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    try:
        raise RuntimeError("cache warm-up failed")
    except RuntimeError:
        structlog.get_logger("svc").exception("warm_up_failed")
    line = _only(_events(capsys.readouterr().out, "warm_up_failed"))
    assert "Traceback (most recent call last)" in line["exception"]
    assert "validation_errors" not in line


# --------------------------------------------------------------------------- PHI-2
# warnings.warn output bypassed the runtime chain: raw text on stderr, no scrubber, and pydantic
# serializer warnings quote the offending value (input_value=...).


class _Mood(enum.Enum):
    LOW = "low"


class _Checkin(BaseModel):
    mood: _Mood


def _capture_warning_output(capsys, emit) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        configure_service_logging(_strict_spec())
        capsys.readouterr()
        try:
            emit()
        finally:
            logging.captureWarnings(False)
    return capsys.readouterr()


def test_phi2_a_warning_is_rendered_by_the_runtime_chain(capsys):
    captured = _capture_warning_output(
        capsys, lambda: warnings.warn(f"could not reach {EMAIL} at {PHONE}", UserWarning,
                                      stacklevel=1))
    for raw in (EMAIL, PHONE):
        assert raw not in captured.out + captured.err
    assert captured.err == ""
    line = _json_lines(captured.out)[-1]
    assert line["level"] == "warning" and "UserWarning" in line["event"]


def test_phi2_a_pydantic_serializer_warning_is_rewritten_without_its_input(capsys):
    captured = _capture_warning_output(
        capsys, lambda: _Checkin.model_construct(mood=PATIENT_TEXT).model_dump())
    _assert_no_patient_text(captured)
    assert captured.err == ""
    line = _only(_events(captured.out, "python_warning_redacted"))
    assert line["warning_category"] == "UserWarning" and line["level"] == "warning"


def test_phi2_py_warnings_is_routed_to_the_runtime_handler():
    configure_service_logging(_strict_spec())
    try:
        routed = logging.getLogger("py.warnings")
        assert routed.propagate is True and routed.disabled is False
        assert warnings.showwarning is logging._showwarning  # captureWarnings(True) is in effect
    finally:
        logging.captureWarnings(False)


# --------------------------------------------------------------------------- PHI-3
# Non-str values passed the PHI sanitizer untouched and were turned into text by JSONRenderer
# afterwards; dict keys were never scanned.


@dataclass
class _Patient:
    email: str
    note: str


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("error", ValueError(f"bad {EMAIL} {PHONE}")),
        ("contacts", (EMAIL, PHONE)),
        ("contacts", frozenset({EMAIL, PHONE})),
        ("contacts", {EMAIL, PHONE}),
        ("raw", f"{EMAIL} {PHONE}".encode()),
        ("data", {EMAIL: 1, PHONE: 2}),
        ("nested", {"rows": [(EMAIL,), {PHONE: {EMAIL}}]}),
        ("patient", _Patient(email=EMAIL, note=f"call back on {PHONE}")),
    ],
    ids=["exception", "tuple", "frozenset", "set", "bytes", "dict_keys", "nested", "object"],
)
def test_phi3_non_str_values_are_sanitized_before_rendering(capsys, field, value):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    structlog.get_logger("svc").info("value_logged", **{field: value})
    captured = capsys.readouterr()
    for raw in (EMAIL, PHONE):
        assert raw not in captured.out + captured.err, f"{raw!r} reached the log line"
    line = _only(_events(captured.out, "value_logged"))
    assert field in line


def test_phi3_stdlib_extra_values_are_sanitized_too(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    logging.getLogger("svc").warning("lookup for %s", (EMAIL, PHONE))
    captured = capsys.readouterr()
    for raw in (EMAIL, PHONE):
        assert raw not in captured.out + captured.err


# --------------------------------------------------------------------------- PHI-4
# The 500 log summary of an extra_forbidden ValidationError logged the client-chosen key that
# the 422 body masks.


@pytest.mark.parametrize("path", ["/stored-extra", "/stored-extra-chained"],
                         ids=["top_level", "chained"])
def test_phi4_a_500_summary_masks_the_extra_forbidden_key(capsys, path):
    configure_service_logging(_strict_spec())
    with TestClient(_phi_app(), raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/p{path}")
    assert response.status_code == 500
    captured = capsys.readouterr()
    assert "Margaret_Hollis" not in captured.out + captured.err
    line = _only(_events(captured.out, "unhandled_exception"))
    assert line["validation_errors"] == [{"type": "extra_forbidden", "field": "[key]"}]


def test_phi4_a_lifespan_summary_masks_the_extra_forbidden_key(capsys):
    try:
        StrictRow.model_validate(STORED_ROW)
    except ValidationError as exc:
        error = exc
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    logging.getLogger("uvicorn.error").error("Exception in 'lifespan' protocol\n", exc_info=error)
    captured = capsys.readouterr()
    assert "Margaret_Hollis" not in captured.out + captured.err
    line = _only(_events(captured.out, "lifespan_startup_failed"))
    assert line["validation_errors"] == [{"type": "extra_forbidden", "field": "[key]"}]


# --------------------------------------------------------------------------- HTT-2
# No request-body size limit anywhere in the HTTP stack: a huge Content-Length or chunked body
# was buffered into memory.

body_router = APIRouter()


@body_router.post("/stream")
async def stream_body(request: Request):
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        request.app.state.seen_max = max(getattr(request.app.state, "seen_max", 0), seen)
    return {"seen": seen}


@body_router.post("/json")
async def json_body(body: NoteIn):
    return {"length": len(body.text)}


def _body_app(**overrides: Any) -> FastAPI:
    spec = ServiceRuntimeSpec(service_name="safety-service", environment="development",
                              **overrides)

    @asynccontextmanager
    async def lifespan(app):
        yield

    return create_service_app(spec, lifespan=lifespan, routers=[(body_router, "/api/v1/b")],
                              title="body")


async def _asgi_post(app: FastAPI, path: str, *, headers: list, chunks: list[bytes]):
    """One POST driven through ``app`` the way an ASGI server would; nothing is buffered here."""
    pulled = {"receive_calls": 0, "bytes": 0}
    pending = list(chunks)

    async def receive():
        pulled["receive_calls"] += 1
        if not pending:
            return {"type": "http.disconnect"}
        body = pending.pop(0)
        pulled["bytes"] += len(body)
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": headers, "client": ("127.0.0.1", 50000),
        "server": ("probe", 80),
    }
    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    response_headers = {bytes(k).decode().lower(): bytes(v).decode() for k, v in start["headers"]}
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], response_headers, body, pulled


def _assert_too_large(status: int, headers: dict, body: bytes) -> None:
    assert status == 413
    assert json.loads(body) == {"error": {"code": "REQUEST_TOO_LARGE",
                                          "message": "Request body too large",
                                          "request_id": headers["x-request-id"]}}


async def test_htt2_an_oversized_content_length_is_refused_before_the_body_is_read():
    app = _body_app()
    status, headers, body, pulled = await _asgi_post(
        app, "/api/v1/b/stream",
        headers=[(b"content-type", b"application/octet-stream"),
                 (b"content-length", str(DEFAULT_BODY_LIMIT + 1).encode())],
        chunks=[b"x" * (DEFAULT_BODY_LIMIT + 1)])
    _assert_too_large(status, headers, body)
    assert pulled["receive_calls"] == 0
    assert getattr(app.state, "seen_max", None) is None  # the endpoint never ran


@pytest.mark.parametrize("path", ["/api/v1/b/stream", "/api/v1/b/json"])
async def test_htt2_a_chunked_body_over_the_limit_is_refused_while_it_is_received(path):
    app = _body_app()
    chunks = [b"x" * 65_536] * 16 + [b"x"]  # 1_048_577 bytes, no Content-Length
    status, headers, body, pulled = await _asgi_post(
        app, path, headers=[(b"content-type", b"application/json"),
                            (b"transfer-encoding", b"chunked")], chunks=chunks)
    _assert_too_large(status, headers, body)
    assert getattr(app.state, "seen_max", 0) <= DEFAULT_BODY_LIMIT
    assert pulled["bytes"] == DEFAULT_BODY_LIMIT + 1  # nothing after the chunk that crossed it


async def test_htt2_a_chunked_body_of_exactly_the_limit_is_served():
    app = _body_app()
    status, _, body, _ = await _asgi_post(
        app, "/api/v1/b/stream", headers=[(b"transfer-encoding", b"chunked")],
        chunks=[b"x" * 65_536] * 16)
    assert status == 200 and json.loads(body) == {"seen": DEFAULT_BODY_LIMIT}


def test_htt2_testclient_bodies_at_and_over_the_limit():
    marker_body = b"SENTINELbody" + b"x" * DEFAULT_BODY_LIMIT
    with TestClient(_body_app()) as client:
        at_limit = client.post("/api/v1/b/stream", content=b"x" * DEFAULT_BODY_LIMIT)
        declared = client.post("/api/v1/b/json", content=marker_body,
                               headers={"content-type": "application/json",
                                        "X-Probe": "SENTINELheader"})
        chunked = client.post("/api/v1/b/json", content=(part for part in [marker_body]),
                              headers={"content-type": "application/json"})
    assert at_limit.status_code == 200 and at_limit.json() == {"seen": DEFAULT_BODY_LIMIT}
    for response in (declared, chunked):
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
        assert "SENTINEL" not in response.text


def test_htt2_the_spec_limit_overrides_the_default():
    assert ServiceRuntimeSpec(service_name="x", environment="development"
                              ).max_request_body_bytes == DEFAULT_BODY_LIMIT
    with TestClient(_body_app(max_request_body_bytes=16)) as client:
        assert client.post("/api/v1/b/stream", content=b"x" * 16).status_code == 200
        assert client.post("/api/v1/b/stream", content=b"x" * 17).status_code == 413


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1048576"])
def test_htt2_the_spec_rejects_an_unusable_limit(bad):
    with pytest.raises(ValueError):
        ServiceRuntimeSpec(service_name="x", environment="development", max_request_body_bytes=bad)


# =========================================================================== fix round 1
# The WS-04.F1 checkpoint review (batch 25): WS04F1-PHI-1, WS04F1-PHI-2, WS04F1-HTT-1 and
# WS04F1-LIF-1 (the last one is covered by the NEAR_MISSES cases above).


def test_fix1_htt1_the_413_code_is_upper_case_like_every_envelope_code():
    from solace_infrastructure.runtime import http

    assert http.REQUEST_TOO_LARGE_CODE == "REQUEST_TOO_LARGE"
    assert all(code.isupper() for code in (http.INTERNAL_ERROR_CODE, http.BAD_REQUEST_CODE,
                                           http.VALIDATION_ERROR_CODE, http.REQUEST_TOO_LARGE_CODE))


# --- WS04F1-PHI-1: FastAPI's ValidationException family is not a pydantic ValidationError, and
# its message lists each error dict with 'input': <value>.


@phi_router.get("/response-model", response_model=Age)
async def response_model_route():
    return {"age": PATIENT_TEXT}  # a stored row that no longer fits the response model


@phi_router.get("/return-annotation")
async def return_annotation_route() -> Age:
    return {"age": PATIENT_TEXT}  # type: ignore[return-value]


@pytest.mark.parametrize("path", ["/response-model", "/return-annotation"])
def test_fix1_phi1_a_response_model_mismatch_logs_no_input(capsys, path):
    configure_service_logging(_strict_spec())
    with TestClient(_phi_app(), raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/p{path}", headers={"X-Request-ID": "req-rv"})
    assert response.status_code == 500 and response.json()["error"]["code"] == "INTERNAL_ERROR"
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only(_events(captured.out, "unhandled_exception"))
    assert line["error_type"] == "ResponseValidationError" and "exception" not in line
    assert line["validation_errors"] == [{"type": "int_parsing", "field": "response.age"}]
    assert line["request_id"] == "req-rv" and line["origin"]


def _fastapi_errors() -> list[dict[str, Any]]:
    return [{"type": "int_parsing", "loc": ("body", "age"), "msg": "Input should be a valid integer",
             "input": PATIENT_TEXT},
            {"type": "extra_forbidden", "loc": ("body", "Margaret_Hollis_HIV_positive"),
             "msg": "Extra inputs are not permitted", "input": True}]


@pytest.mark.parametrize("via", ["stdlib", "structlog"])
def test_fix1_phi1_a_chained_fastapi_validation_error_logs_masked_fields(capsys, via):
    from fastapi.exceptions import RequestValidationError

    configure_service_logging(_strict_spec())
    capsys.readouterr()
    try:
        try:
            raise RequestValidationError(_fastapi_errors())
        except RequestValidationError as exc:
            raise RuntimeError("replayed request rejected") from exc
    except RuntimeError:
        if via == "stdlib":
            logging.getLogger("svc.replay").exception("replay failed")
        else:
            structlog.get_logger("svc.replay").exception("replay_failed")
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    assert "Margaret_Hollis" not in captured.out + captured.err
    line = _json_lines(captured.out)[-1]
    assert line["error_type"] == "RuntimeError" and "exception" not in line
    assert line["validation_errors"] == [{"type": "int_parsing", "field": "body.age"},
                                         {"type": "extra_forbidden", "field": "body.[key]"}]


def test_fix1_phi1_traceback_text_quoting_fastapi_input_is_dropped(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    try:
        raise RuntimeError(f"rows rejected: {_fastapi_errors()}")
    except RuntimeError:
        structlog.get_logger("svc.repo").exception("rows_rejected")
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only(_events(captured.out, "rows_rejected"))
    assert "exception" not in line and line["error_type"] == "RuntimeError"
    assert line["validation_errors"] == []


def test_fix1_phi1_a_lifespan_response_validation_error_logs_fields_not_values(capsys):
    from fastapi.exceptions import ResponseValidationError

    error = ResponseValidationError(_fastapi_errors())
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    logging.getLogger("uvicorn.error").error("Exception in 'lifespan' protocol\n", exc_info=error)
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    line = _only(_events(captured.out, "lifespan_startup_failed"))
    assert line["error_type"] == "ResponseValidationError"
    assert line["validation_errors"] == [{"type": "int_parsing", "field": "body.age"},
                                         {"type": "extra_forbidden", "field": "body.[key]"}]


# --- WS04F1-PHI-2: validation input in rendered text outside the 'exception' field.


def test_fix1_phi2_a_task_exception_never_retrieved_logs_no_input(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()

    async def scenario():
        async def fire_and_forget():  # not registered with the runtime, never awaited
            _parse_age(PATIENT_TEXT)

        task = asyncio.get_running_loop().create_task(fire_and_forget())
        await asyncio.sleep(0.01)
        assert task.done()
        del task
        gc.collect()  # the finalizer reports "Task exception was never retrieved"

    asyncio.run(scenario())
    gc.collect()
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    assert "input_value=" not in captured.out + captured.err
    line = _only([l for l in _json_lines(captured.out)
                  if str(l.get("event", "")).startswith("Task exception was never retrieved")])
    assert line["event"] == "Task exception was never retrieved"
    assert line["error_type"] == "ValidationError" and line["validation_errors"] == AGE_ERRORS


def _validation_error() -> ValidationError:
    try:
        _parse_age(PATIENT_TEXT)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")  # pragma: no cover


def _chained_validation_error() -> RuntimeError:
    try:
        try:
            _parse_age(PATIENT_TEXT)
        except ValidationError as exc:
            raise RuntimeError("payload rejected") from exc
    except RuntimeError as outer:
        return outer
    raise AssertionError("expected a RuntimeError")  # pragma: no cover


def test_fix1_phi2_an_exception_value_is_summarised_not_rendered(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    log = structlog.get_logger("svc.consumer")
    log.error("consumer_failed", error=_validation_error())
    log.error("consumer_failed_chained", error=_chained_validation_error())
    log.error("consumer_failed_list", errors=[_validation_error()])
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    assert "input_value=" not in captured.out
    summary = {"error_type": "ValidationError", "validation_errors": AGE_ERRORS}
    assert _only(_events(captured.out, "consumer_failed"))["error"] == summary
    assert _only(_events(captured.out, "consumer_failed_chained"))["error"] == {
        "error_type": "RuntimeError", "validation_errors": AGE_ERRORS}
    assert _only(_events(captured.out, "consumer_failed_list"))["errors"] == [summary]


def test_fix1_phi2_text_quoting_validation_input_is_cut_in_every_string(capsys):
    from fastapi.exceptions import ResponseValidationError

    configure_service_logging(_strict_spec())
    capsys.readouterr()
    log = structlog.get_logger("svc.repo")
    log.warning(f"row rejected: {_validation_error()}")  # the event text itself
    log.warning("row_rejected", detail=str(_validation_error()))
    log.warning("rows_rejected", detail=str(_fastapi_errors()), nested={"why": [repr(_fastapi_errors())]})
    log.warning("response_rejected", detail=str(ResponseValidationError(_fastapi_errors())))
    logging.getLogger("svc.repo").warning("stdlib row rejected: %s", _validation_error())
    captured = capsys.readouterr()
    _assert_no_patient_text(captured)
    for marker in ("input_value=", "'input':", "Margaret_Hollis"):
        assert marker not in captured.out + captured.err
    lines = _json_lines(captured.out)
    assert lines[0]["event"] == "row rejected: 1 validation error for Age"  # cut to its first line
    assert _only(_events(captured.out, "row_rejected"))["detail"] == "1 validation error for Age"
    rows = _only(_events(captured.out, "rows_rejected"))
    assert rows["detail"] == rows["nested"]["why"][0] != ""  # one-line text: a fixed text instead
    assert _only(_events(captured.out, "response_rejected"))["detail"] == "2 validation errors:"
    assert lines[-1]["event"] == "stdlib row rejected: 1 validation error for Age"


# =========================================================================== fix round 2
# The fix-round-1 re-review (batch 29): WS04F1F1-PHI-1, WS04F1F1-PHI-2 and WS04F1F1-LIF-1 (the
# last one is covered by the NEAR_MISSES cases above).

# Free text with an apostrophe: repr() then escapes the quotes around 'input', and JSON
# re-encoding escapes the ones around "input".
APOSTROPHE_TEXT = "Margaret can't stop crying"
APOSTROPHE_WORDS = ("Margaret", "crying")


def _assert_no_apostrophe_text(captured: Any) -> None:
    text = captured.out + captured.err
    for word in APOSTROPHE_WORDS:
        assert word not in text, f"patient text {word!r} reached the log output"


def _apostrophe_errors() -> list[dict[str, Any]]:
    """pydantic's errors() for patient text with an apostrophe (input, url and all)."""
    try:
        Age(age=APOSTROPHE_TEXT)
    except ValidationError as exc:
        return exc.errors()
    raise AssertionError("expected a ValidationError")  # pragma: no cover


def _unlinked_batch_error() -> RuntimeError:
    # Stored errors() raised later, outside the except block: nothing links a ValidationError.
    return RuntimeError(f"batch rejected: {_apostrophe_errors()}")


def _response_error_text() -> str:
    from fastapi.exceptions import ResponseValidationError

    return str(ResponseValidationError(_apostrophe_errors()))


@dataclass
class _Holder:
    detail: str


def _emit_exception_value() -> None:
    structlog.get_logger("svc.batch").error("batch_failed", error=_unlinked_batch_error())


def _emit_unretrieved_task() -> None:
    async def scenario():
        async def fire_and_forget():
            raise _unlinked_batch_error()

        task = asyncio.get_running_loop().create_task(fire_and_forget())
        await asyncio.sleep(0.01)
        del task
        gc.collect()

    asyncio.run(scenario())
    gc.collect()


def _emit_stdlib_percent_r() -> None:
    logging.getLogger("svc.batch").warning("rejected %r", _response_error_text())


def _emit_structlog_percent_r() -> None:
    structlog.get_logger("svc.batch").warning("rejected %r", _response_error_text())


def _emit_object_repr() -> None:
    structlog.get_logger("svc.batch").info("holder_logged", holder=_Holder(str(_apostrophe_errors())))


def _emit_double_encoded_json() -> None:
    payload = json.dumps(json.dumps({"input": APOSTROPHE_TEXT, "type": "int_parsing"}))
    structlog.get_logger("svc.batch").info("payload_logged", body=payload)


@pytest.mark.parametrize(
    "emit",
    [_emit_exception_value, _emit_unretrieved_task, _emit_stdlib_percent_r,
     _emit_structlog_percent_r, _emit_object_repr, _emit_double_encoded_json],
    ids=["unlinked_exception_value", "unretrieved_task", "stdlib_percent_r",
         "structlog_percent_r", "dataclass_repr", "double_encoded_json"],
)
def test_fix2_phi1_escaped_input_markers_are_still_recognised(capsys, emit):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    emit()
    captured = capsys.readouterr()
    _assert_no_apostrophe_text(captured)
    assert _json_lines(captured.out), "the emitter produced no log line"


def test_fix2_phi1_an_escaped_warning_text_is_rewritten(capsys):
    captured = _capture_warning_output(
        capsys, lambda: warnings.warn(repr(_response_error_text()), UserWarning, stacklevel=1))
    _assert_no_apostrophe_text(captured)
    line = _only(_events(captured.out, "python_warning_redacted"))
    assert line["warning_category"] == "UserWarning"


@pytest.mark.parametrize(
    "text",
    [
        "'input': 'x'", '"input": "x"', "\\'input\\': 'x'", '\\"input\\": "x"',
        '\\\\\\"input\\\\\\": "x"', "{\\u0022input\\u0022: 1}", "'input' : 1", "input_value='x'",
    ],
    ids=["single", "double", "repr_escaped", "json_escaped", "json_twice", "unicode_escape",
         "space_before_colon", "pydantic"],
)
def test_fix2_phi1_the_marker_matches_every_escaping(text):
    from solace_infrastructure.runtime import logging as runtime_logging

    assert runtime_logging._quotes_validation_input(text)


@pytest.mark.parametrize("payload", [
    "\\" * 65536,                      # one long backslash run (quadratic with an unbounded quantifier)
    "\\" * 65536 + "'inpu",            # a run that almost reaches a marker
    ("\\" * 64 + "'") * 1000,          # many shorter runs, each ending in a quote
], ids=["backslash-run", "run-then-near-marker", "short-runs-with-quotes"])
def test_fix2_review_marker_is_linear_on_long_backslash_runs(payload):
    """WS04F1F2-PHI-1: the escape-tolerant marker must not be quadratic (one request body could
    otherwise block a service's event loop while its log line is checked)."""
    import time

    from solace_infrastructure.runtime import logging as runtime_logging
    from solace_infrastructure.runtime.logging import redact_credentials

    started = time.perf_counter()
    runtime_logging._quotes_validation_input(payload)
    redact_credentials(None, "info", {"event": "x", "query": payload})
    assert time.perf_counter() - started < 0.5


def test_fix2_review_marker_still_matches_four_levels_of_escaping():
    from solace_infrastructure.runtime import logging as runtime_logging

    assert runtime_logging._quotes_validation_input("\\\\\\\\\\\\\\'input\\\\\\\\\\\\\\': 'x'")


@pytest.mark.parametrize("text", ["user input was empty", "inputs: 3", "'inputs': 2", "input_type"])
def test_fix2_phi1_the_marker_ignores_ordinary_text(text):
    from solace_infrastructure.runtime import logging as runtime_logging

    assert not runtime_logging._quotes_validation_input(text)


# --- WS04F1F1-PHI-2: pydantic and FastAPI error dicts logged as values.


def _value_error_errors() -> list[dict[str, Any]]:
    """errors() of a value_error whose ctx.error quotes the input (a validator author's text)."""
    from pydantic import field_validator

    class Checkin(BaseModel):
        mood: str

        @field_validator("mood")
        @classmethod
        def known_mood(cls, value: str) -> str:
            raise ValueError(f"unknown mood: {value}")

    try:
        Checkin(mood=APOSTROPHE_TEXT)
    except ValidationError as exc:
        return exc.errors()
    raise AssertionError("expected a ValidationError")  # pragma: no cover


def test_fix2_phi2_error_dicts_logged_as_structlog_values_are_reduced(capsys):
    from fastapi.exceptions import RequestValidationError

    configure_service_logging(_strict_spec())
    capsys.readouterr()
    log = structlog.get_logger("svc.validation")
    log.warning("validation_failed", errors=_apostrophe_errors())
    log.warning("validation_failed_tuple", errors=tuple(_value_error_errors()))
    log.warning("request_validation_failed",
                errors=RequestValidationError(_fastapi_errors()).errors())
    captured = capsys.readouterr()
    _assert_no_apostrophe_text(captured)
    _assert_no_patient_text(captured)
    assert "Margaret_Hollis" not in captured.out + captured.err
    assert _only(_events(captured.out, "validation_failed"))["errors"] == [
        {"type": "int_parsing", "field": "age"}]
    assert _only(_events(captured.out, "validation_failed_tuple"))["errors"] == [
        {"type": "value_error", "field": "mood"}]
    assert _only(_events(captured.out, "request_validation_failed"))["errors"] == [
        {"type": "int_parsing", "field": "body.age"}, {"type": "extra_forbidden", "field": "body.[key]"}]


def test_fix2_phi2_stdlib_extra_validation_errors_are_reduced(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    logging.getLogger("svc.validation").warning(
        "validation failed", extra={"validation_errors": _apostrophe_errors() + _value_error_errors()})
    captured = capsys.readouterr()
    _assert_no_apostrophe_text(captured)
    line = _only(_events(captured.out, "validation failed"))
    assert line["validation_errors"] == [{"type": "int_parsing", "field": "age"},
                                         {"type": "value_error", "field": "mood"}]


def test_fix2_phi2_a_dict_that_only_looks_alike_is_kept(capsys):
    configure_service_logging(_strict_spec())
    capsys.readouterr()
    structlog.get_logger("svc").info("routing", rule={"type": "prefix", "loc": "/api"},
                                     job={"type": "export", "input": "s3-bucket-a"})
    line = _only(_events(capsys.readouterr().out, "routing"))
    assert line["rule"] == {"type": "prefix", "loc": "/api"}
    assert line["job"] == {"type": "export", "input": "s3-bucket-a"}
