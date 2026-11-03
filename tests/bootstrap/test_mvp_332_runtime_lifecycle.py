"""MVP-332: teardown closes pools, Redis and registered clients in reverse order with a timeout;
strict start-up failures are fatal, development failures degrade /ready."""
import asyncio, json, time
import pytest
from fastapi import FastAPI
from solace_infrastructure.runtime import (ServiceRuntimeSpec, StartupDependencyError,
                                          start_service_runtime, stop_service_runtime)
from solace_infrastructure.runtime import bootstrap as bs, datastores as ds
from solace_infrastructure.runtime.logging import configure_service_logging as real_configure_logging

def _spec(env="production", **kw):
    base = dict(uses_postgres=True, uses_redis=True, handles_phi=True)
    base.update(kw)
    return ServiceRuntimeSpec(service_name="memory-service", environment=env, **base)

async def test_start_order_is_guards_logging_datastores(fakes, calls):
    app = FastAPI()
    rt = await start_service_runtime(app, _spec())
    assert calls[0] == ("guards", ("AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "POSTGRES_PASSWORD", "REDIS_URL"))
    assert calls[1] == "logging"
    assert calls.index("get_pool:default") < calls.index("schema_head") < calls.index("redis.ping")
    assert app.state.runtime is rt and rt.pg_pool is fakes and rt.redis is not None
    report = await rt.evaluate_readiness()
    assert report.ready and report.checks["postgres"].critical and report.checks["redis"].critical
    await stop_service_runtime(rt)

async def test_dev_skips_schema_head_check(fakes, calls):
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    assert "schema_head" not in calls
    await stop_service_runtime(rt)

async def test_strict_postgres_failure_is_fatal(fakes, monkeypatch):
    async def boom(name="default"):
        raise OSError("connect to postgres:5432 failed password=s3cret")
    monkeypatch.setattr(ds.ConnectionPoolManager, "get_pool", boom)
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("staging"))
    assert exc.value.component == "postgres" and "s3cret" not in str(exc.value)

async def test_dev_postgres_failure_degrades_ready(fakes, monkeypatch):
    async def boom(name="default"):
        raise OSError("refused")
    monkeypatch.setattr(ds.ConnectionPoolManager, "get_pool", boom)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    report = await rt.evaluate_readiness()
    assert report.ready is False and report.checks["postgres"].ok is False and rt.pg_pool is None
    await stop_service_runtime(rt)

async def test_strict_requires_redis_url(fakes, monkeypatch):
    monkeypatch.delenv("REDIS_URL")
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("production"))
    assert exc.value.component == "redis"

async def test_redis_failure_warning_never_logs_the_url(fakes, monkeypatch, capsys):
    monkeypatch.setattr(bs, "configure_service_logging", real_configure_logging)
    class BrokenRedis:
        async def ping(self):
            raise ConnectionError("redis://:p4ss@redis:6379/0 refused")
        async def aclose(self):
            pass
    monkeypatch.setattr(ds.redis_asyncio, "from_url", lambda url, **kw: BrokenRedis())
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    await stop_service_runtime(rt)
    out = capsys.readouterr().out
    lines = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    assert any(l.get("event") == "runtime_component_degraded" and l.get("component") == "redis"
               and l.get("error") == "ConnectionError" for l in lines)
    assert "p4ss" not in out

async def test_stop_closes_in_reverse_order_with_timeout(fakes, calls, monkeypatch):
    monkeypatch.setattr(bs, "SHUTDOWN_STEP_TIMEOUT_SECONDS", 0.2)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    async def closer_a():
        calls.append("a")
    async def closer_hang():
        await asyncio.sleep(30)
    async def closer_b():
        calls.append("b")
    rt.register_shutdown("a", closer_a)
    rt.register_shutdown("hang", closer_hang)
    rt.register_shutdown("b", closer_b)
    t0 = time.monotonic()
    await stop_service_runtime(rt)
    assert time.monotonic() - t0 < 2.0
    tail = [c for c in calls if c in ("a", "b", "redis.aclose", "pg.close_all")]
    assert tail == ["b", "a", "redis.aclose", "pg.close_all"]

def test_required_secrets_follow_the_spec():
    spec = ServiceRuntimeSpec(service_name="analytics-service", environment="production", uses_postgres=True)
    assert bs.required_secrets_for(spec) == ("AUTH_SECRET_KEY", "POSTGRES_PASSWORD")


# --- WS-04.4 fix round 1 (review R1 to R3) ------------------------------------------------------
# Readiness turns red when a started datastore goes down; a failed or cancelled start-up closes
# what it started; the readiness refresher runs on its own; a cancelled stop leaves the closers it
# has not reached for the next stop; a strict Postgres configuration error is fatal before any
# connection attempt. The fakes fixture's Pool has no methods, so tests that probe Postgres give
# the returned pool instance an async fetchval.

_REAL_SCHEMA_HEAD_CHECK = ds._schema_head_check
_REAL_GET_POOL = ds.ConnectionPoolManager.__dict__["get_pool"]


async def _eventually(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _live_refreshers():
    return [t for t in asyncio.all_tasks()
            if t.get_name().endswith(bs.READINESS_REFRESHER_TASK) and not t.done()]


def _probe_every_evaluation(monkeypatch):
    monkeypatch.setattr(ds, "DATASTORE_PROBE_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(bs, "READINESS_CACHE_SECONDS", 0.0)


async def test_readiness_turns_red_when_a_started_postgres_goes_down(fakes, monkeypatch):
    _probe_every_evaluation(monkeypatch)
    state = {"down": False}

    async def fetchval(sql):
        if state["down"]:
            raise ConnectionResetError("lost postgresql://solace_app:s3cret@db:5432/solace")
        return 1

    fakes.fetchval = fetchval
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    try:
        assert (await rt.evaluate_readiness()).ready
        state["down"] = True
        report = await rt.evaluate_readiness()
        assert report.ready is False
        assert report.checks["postgres"].ok is False
        assert report.checks["postgres"].detail == "ConnectionResetError"
        state["down"] = False
        assert (await rt.evaluate_readiness()).ready
    finally:
        await stop_service_runtime(rt)


async def test_readiness_turns_red_when_a_started_redis_stops_answering(fakes, monkeypatch):
    _probe_every_evaluation(monkeypatch)
    fakes.fetchval = lambda sql: asyncio.sleep(0, result=1)
    state = {"pong": True}

    class Redis:
        async def ping(self):
            return state["pong"]

        async def aclose(self):
            pass

    monkeypatch.setattr(ds.redis_asyncio, "from_url", lambda url, **kw: Redis())
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    try:
        assert (await rt.evaluate_readiness()).ready
        state["pong"] = False
        report = await rt.evaluate_readiness()
        assert report.ready is False
        assert (report.checks["redis"].ok, report.checks["redis"].detail) == (False, "ConnectionError")
        state["pong"] = True
        assert (await rt.evaluate_readiness()).ready
    finally:
        await stop_service_runtime(rt)


async def test_strict_partial_start_up_failure_closes_what_started(fakes, calls, monkeypatch):
    monkeypatch.delenv("REDIS_URL")
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("production"))
    assert (exc.value.component, exc.value.reason) == ("redis", "REDIS_URL is not set")
    assert "pg.close_all" in calls
    assert _live_refreshers() == []


async def test_strict_redis_ping_failure_closes_the_client_and_the_pool(fakes, calls, monkeypatch):
    class DeadRedis:
        async def ping(self):
            raise ConnectionRefusedError("redis://:p4ss@redis:6379/0 refused")

        async def aclose(self):
            calls.append("dead_redis.aclose")

    monkeypatch.setattr(ds.redis_asyncio, "from_url", lambda url, **kw: DeadRedis())
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("staging"))
    assert (exc.value.component, exc.value.reason) == ("redis", "ConnectionRefusedError")
    assert "p4ss" not in str(exc.value)
    assert "dead_redis.aclose" in calls and "pg.close_all" in calls
    assert _live_refreshers() == []


async def test_readiness_refresher_probes_and_publishes_without_ready_traffic(fakes, calls, monkeypatch):
    from prometheus_client import REGISTRY
    from solace_infrastructure.runtime.metrics import SERVICE_READY

    monkeypatch.setattr(bs, "READINESS_REFRESH_SECONDS", 0.01)
    monkeypatch.setattr(ds, "DATASTORE_PROBE_INTERVAL_SECONDS", 0.0)

    async def fetchval(sql):
        calls.append("pg.fetchval")
        return 1

    fakes.fetchval = fetchval
    SERVICE_READY.labels(service="memory-service").set(0)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    task = rt._background_tasks[bs.READINESS_REFRESHER_TASK]
    assert not task.done() and rt.dead_background_task() is None
    # No evaluate_readiness call from the test: only the refresher can probe and publish.
    assert await _eventually(lambda: "pg.fetchval" in calls and calls.count("redis.ping") >= 2)
    assert await _eventually(lambda: REGISTRY.get_sample_value(
        "solace_service_ready", {"service": "memory-service"}) == 1.0)
    await stop_service_runtime(rt)
    assert task.cancelled()


async def test_readiness_refresher_survives_a_failed_evaluation(fakes, monkeypatch):
    monkeypatch.setattr(bs, "READINESS_REFRESH_SECONDS", 0.01)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    seen = []
    real_evaluate = rt.evaluate_readiness

    async def flaky():
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("readiness bug")
        return await real_evaluate()

    rt.evaluate_readiness = flaky
    try:
        assert await _eventually(lambda: len(seen) >= 2)
        assert rt.dead_background_task() is None
    finally:
        await stop_service_runtime(rt)


async def test_a_second_stop_runs_no_closer_again(fakes, calls):
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    await stop_service_runtime(rt)
    before = list(calls)
    assert "redis.aclose" in before and "pg.close_all" in before
    await stop_service_runtime(rt)
    assert calls == before


async def test_strict_schema_mismatch_is_fatal_and_closes_the_pool(fakes, calls, monkeypatch):
    # Only the spec is strict here (bare ENVIRONMENT=test), so assert_schema_head alone would warn.
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setattr(ds, "_schema_head_check", _REAL_SCHEMA_HEAD_CHECK)

    async def fetchval(sql):
        calls.append("pg.fetchval")
        return "020_an_older_revision"

    fakes.fetchval = fetchval
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("staging"))
    assert (exc.value.component, exc.value.reason) == ("postgres", "SchemaVersionMismatch")
    assert "pg.fetchval" in calls and "pg.close_all" in calls


async def test_strict_schema_at_head_starts(fakes, monkeypatch):
    from solace_infrastructure.database.schema_check import SCHEMA_HEAD

    monkeypatch.setattr(ds, "_schema_head_check", _REAL_SCHEMA_HEAD_CHECK)
    fakes.fetchval = lambda sql: asyncio.sleep(0, result=SCHEMA_HEAD)
    rt = await start_service_runtime(FastAPI(), _spec("production"))
    try:
        assert rt.pg_pool is fakes
    finally:
        await stop_service_runtime(rt)


async def test_a_cancelled_stop_leaves_unrun_closers_for_the_next_stop(fakes, calls):
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    refresher = rt._background_tasks[bs.READINESS_REFRESHER_TASK]
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(5)
        calls.append("slow.done")

    rt.register_shutdown("slow", slow)
    stopping = asyncio.create_task(stop_service_runtime(rt))
    await asyncio.wait_for(started.wait(), 2)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert not refresher.done() and "redis.aclose" not in calls
    await stop_service_runtime(rt)
    # The closer that was running when the stop was cancelled is not run again.
    assert [c for c in calls if c in ("slow.done", "redis.aclose", "pg.close_all")] == [
        "redis.aclose", "pg.close_all"]
    assert refresher.cancelled()


async def test_a_cancelled_start_up_cleanup_still_closes_what_started(fakes, calls, monkeypatch):
    monkeypatch.delenv("REDIS_URL")
    closing = asyncio.Event()

    async def slow_close_all():
        closing.set()
        await asyncio.sleep(0.2)
        calls.append("pg.close_all")

    monkeypatch.setattr(ds.ConnectionPoolManager, "close_all_pools", slow_close_all)
    starting = asyncio.create_task(start_service_runtime(FastAPI(), _spec("production")))
    await asyncio.wait_for(closing.wait(), 2)
    starting.cancel()
    with pytest.raises((asyncio.CancelledError, StartupDependencyError)):
        await starting
    assert await _eventually(lambda: "pg.close_all" in calls)


async def test_strict_postgres_config_error_is_fatal_before_any_connection(
        fakes, monkeypatch, tmp_path, capsys):
    import traceback

    from solace_infrastructure.database import connection_manager as cm

    dialled = []

    async def no_connection(*args, **kwargs):
        dialled.append(1)
        raise AssertionError("a connection was attempted")

    monkeypatch.chdir(tmp_path)  # PostgresSettings also reads .env from the working directory
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_USER", "solace_app")
    monkeypatch.setenv("POSTGRES_PASSWORD", "Sentinel-pw-0x51")
    monkeypatch.delenv("POSTGRES_SSL_ROOT_CERT", raising=False)
    monkeypatch.setattr(cm.asyncpg, "create_pool", no_connection)
    monkeypatch.setattr(ds.ConnectionPoolManager, "_pools", {})
    monkeypatch.setattr(ds.ConnectionPoolManager, "_pool_configs", {})
    monkeypatch.setattr(ds.ConnectionPoolManager, "_locks", {})
    monkeypatch.setattr(ds.ConnectionPoolManager, "get_pool", _REAL_GET_POOL)
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("production"))
    assert (exc.value.component, exc.value.reason) == ("postgres", "DatabaseConfigError")
    assert dialled == []
    rendered = "".join(traceback.format_exception(exc.value))
    captured = capsys.readouterr()
    for text in (str(exc.value), rendered, captured.out, captured.err):
        assert "Sentinel-pw-0x51" not in text
