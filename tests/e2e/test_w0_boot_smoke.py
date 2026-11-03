"""W0 exit-criterion suite: every MVP service boots on the compose test stack and serves the IF-01 contract
(MVP-039, WS-05.5). Dev and strict profiles (SOLACE_E2E_PROFILE / SOLACE_E2E_ENVIRONMENT); see tests/e2e/conftest.py.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

import pytest

from solace_infrastructure.database.schema_check import SCHEMA_HEAD
from solace_testing.e2e import MVP_SERVICES, ComposeError, make_user
from solace_testing.e2e_manifest import hop

pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.timeout(1200, func_only=True)]

DB_SERVICES = ("orchestrator-service", "user-service", "safety-service", "notification-service", "diagnosis-service",
               "memory-service", "therapy-service", "personality-service", "analytics-service")
FALLBACK_PATTERNS = [r"publisher_using_in_memory_outbox", r"dlq_using_in_memory_store", r"in_memory_fallback",
                     r"event_outbox_pool_not_available", r"localhost:(5432|6379|9092|29092)"]
PUBLISHERS_ALLOWED = {"caddy", "mailpit", "fake-llm"}


def _poll(fetch: Callable, *, until: Callable[[object], bool], timeout: float, interval: float = 3.0):
    """Call ``fetch`` until ``until(result)`` holds or ``timeout`` seconds pass; return the last result.

    A fetch that raises ConnectionError or ComposeError (service not up yet, as in e2e._wait_live) counts as not
    ready. On timeout with no result ever obtained, the last exception is re-raised.
    """
    deadline = time.monotonic() + timeout
    result, last_error = None, None
    while True:
        try:
            result = fetch()
            if until(result):
                return result
        except (ConnectionError, ComposeError) as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            if result is None and last_error is not None:
                raise last_error
            return result
        time.sleep(interval)


def _assert_ready(r, service: str) -> None:
    """The IF-01 /ready body: ready, named, with at least one critical check and every critical check ok."""
    assert r.status == 200 and r.json["status"] == "ready" and r.json["service"] == service
    checks = r.json["checks"]
    assert checks, f"{service}: /ready reports no checks"
    critical = [c for c in checks.values() if c["critical"]]
    assert critical, f"{service}: /ready reports no critical check"
    assert all(c["ok"] for c in critical)


def _assert_only_edge_published(ps: list[dict]) -> None:
    """ps lists every MVP service and caddy, and only the allowed services publish a host port."""
    listed = {p["Service"] for p in ps}
    missing = (set(MVP_SERVICES) | {"caddy"}) - listed
    assert not missing, f"ps does not list: {sorted(missing)}"
    published = {p["Service"] for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort")}
    assert published <= PUBLISHERS_ALLOWED, sorted(published - PUBLISHERS_ALLOWED)


def _ps_rows(stack, *args: str) -> list[dict]:
    out = stack.compose("ps", *args, "--format", "json").stdout
    rows: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        rows.extend(parsed if isinstance(parsed, list) else [parsed])   # old compose: one array; new: one object/line
    return rows


class HttpResultStub:
    def __init__(self, status, json=None):
        self.status, self.json = status, json


def test_w0_poll_returns_the_first_match_and_the_last_result_on_timeout():
    seen = iter([1, 2, 3])
    assert _poll(lambda: next(seen), until=lambda r: r == 2, timeout=5, interval=0) == 2
    assert _poll(lambda: 7, until=lambda r: False, timeout=0, interval=0) == 7
    with pytest.raises(ConnectionError):
        _poll(lambda: (_ for _ in ()).throw(ConnectionError("down")), until=lambda r: True, timeout=0, interval=0)


def test_w0_poll_retries_compose_errors_and_reraises_the_last_one_on_timeout():
    calls = iter([ComposeError("starting"), HttpResultStub(200)])

    def fetch():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    assert _poll(fetch, until=lambda r: r.status == 200, timeout=5, interval=0).status == 200
    with pytest.raises(ComposeError, match="never up"):
        _poll(lambda: (_ for _ in ()).throw(ComposeError("never up")), until=lambda r: True, timeout=0, interval=0)


def test_w0_ready_check_rejects_empty_or_noncritical_checks():
    def ok(checks):
        return HttpResultStub(200, {"status": "ready", "service": "svc", "checks": checks})

    _assert_ready(ok({"db": {"ok": True, "critical": True, "detail": ""}}), "svc")
    for checks in ({}, {"cache": {"ok": True, "critical": False, "detail": ""}}):
        with pytest.raises(AssertionError):
            _assert_ready(ok(checks), "svc")
    with pytest.raises(AssertionError):
        _assert_ready(ok({"db": {"ok": False, "critical": True, "detail": ""}}), "svc")


def test_w0_published_check_requires_the_expected_services_in_ps():
    def row(svc, port=None):
        return {"Service": svc, "Publishers": [{"PublishedPort": port}] if port else []}

    full = [row(s) for s in MVP_SERVICES] + [row("caddy", 8080), row("mailpit", 8025)]
    _assert_only_edge_published(full)
    with pytest.raises(AssertionError):
        _assert_only_edge_published([])                      # an empty ps must not pass
    with pytest.raises(AssertionError):
        _assert_only_edge_published([r for r in full if r["Service"] != "caddy"])
    with pytest.raises(AssertionError):
        _assert_only_edge_published(full + [row("postgres", 5432)])


def test_w0_fallback_patterns_match_the_known_warnings_only():
    bad = ["publisher_using_in_memory_outbox", "dlq_using_in_memory_store x", "in_memory_fallback=1",
           "event_outbox_pool_not_available", "dsn=localhost:5432", "redis localhost:6379", "localhost:29092"]
    assert all(any(re.search(p, line) for p in FALLBACK_PATTERNS) for line in bad)
    assert not any(re.search(p, "postgres:5432 kafka:9092 connected") for p in FALLBACK_PATTERNS)


def test_w0_migrate_job_completed(e2e_stack):
    with hop("migrate_completed"):
        (job,) = _ps_rows(e2e_stack, "-a", "migrate")
        assert job["State"] == "exited" and job["ExitCode"] == 0
        assert e2e_stack.sql("SELECT version_num AS v FROM alembic_version")[0]["v"] == SCHEMA_HEAD


@pytest.mark.parametrize("service", MVP_SERVICES)
def test_w0_service_ready_within_180s(e2e_stack, service):
    with hop("service_ready"):
        r = _poll(lambda: e2e_stack.call(service, "GET", "/ready"), until=lambda r: r.status == 200, timeout=180)
        _assert_ready(r, service)


@pytest.mark.parametrize("service", MVP_SERVICES)
def test_w0_service_serves_metrics(e2e_stack, service):
    with hop("metrics_served"):
        r = e2e_stack.call(service, "GET", "/metrics")
        assert r.status == 200 and "solace_readiness_check_ok" in r.text


@pytest.mark.profile("strict")
def test_w0_strict_services_connect_as_solace_app_over_tls(e2e_stack):
    with hop("app_role_tls"):
        rows = e2e_stack.sql("SELECT a.usename AS u, s.ssl AS ssl FROM pg_stat_activity a JOIN pg_stat_ssl s ON s.pid = a.pid "
                             "WHERE a.datname = 'solace' AND a.backend_type = 'client backend' AND a.pid <> pg_backend_pid()")
        assert rows and {r["u"] for r in rows} == {"solace_app"} and all(r["ssl"] for r in rows)
        for svc in DB_SERVICES:
            assert e2e_stack.compose("exec", "-T", svc, "printenv", "POSTGRES_SSL_MODE").stdout.strip() == "verify-full"


@pytest.mark.profile("strict")
def test_w0_no_internal_port_published(e2e_stack):
    with hop("no_internal_ports"):
        _assert_only_edge_published(_ps_rows(e2e_stack))


@pytest.mark.profile("strict")
@pytest.mark.parametrize("service", MVP_SERVICES)
def test_w0_no_fallback_warning_in_logs(e2e_stack, service):
    with hop("no_fallback_logs"):
        text = e2e_stack.logs(service)
        hits = [p for p in FALLBACK_PATTERNS if re.search(p, text)]   # patterns only: never print log text
        assert hits == []


@pytest.mark.profile("dev")
async def test_w0_register_and_login_through_the_edge(e2e_stack):
    with hop("register_login"):
        user = await make_user(e2e_stack)
        assert user.access_token and user.id
