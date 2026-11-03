# tests/bootstrap/test_service_conformance.py
"""MVP-036/IF-01: each of the nine MVP services builds its app with create_service_app, declares SPEC,
serves /live /ready /health /metrics from the runtime, and brackets its lifespan with the runtime.

The nine ``test_service_conforms[...]`` and ``test_lifespan_is_bracketed_by_the_runtime[...]``
parameters are red by design until each service's W0 phase-B adoption task lands; each
adoption task runs ``-k <service>`` as its acceptance. The probe-control and reference-app
tests are green from WS-04.5 on.

``_conformance_probe.py`` imports one module in a fresh interpreter with synthetic secrets
(``_probe_env``) and all outbound connections refused, and prints one JSON report. For a
module whose app the runtime built, it also enters the app's lifespan after applying
uvicorn's default logging configuration (as the uvicorn CLI does) and stops right after
``start_service_runtime`` returns, so no service-specific start-up code runs. The report then
covers the logging state after start-up (``uvicorn.access`` disabled, no uvicorn handlers of
its own, the runtime chain) and what ``/live``, ``/ready``, ``/health`` and ``/metrics``
answer: no secret in any body or header, and no raw request path in a metric label.
"""
import ast, copy, json, os, secrets, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name("_conformance_probe.py")
FIXTURES = Path(__file__).with_name("fixtures")
ALL = dict(uses_postgres=True, uses_redis=True, event_plane="publish_consume")
PHI = dict(ALL, handles_phi=True, audited=True)

@dataclass(frozen=True)
class Svc:
    paths: tuple
    module: str
    main_py: str
    flags: dict = field(default_factory=dict)

SERVICES = {
    "orchestrator-service": Svc((REPO, REPO / "src"), "services.orchestrator_service.src.main", "services/orchestrator_service/src/main.py", dict(PHI, rate_limited=True, edge=True)),
    "safety-service": Svc((REPO, REPO / "src"), "services.safety_service.src.main", "services/safety_service/src/main.py", dict(PHI, rate_limited=False, edge=False)),
    "therapy-service": Svc((REPO, REPO / "src"), "services.therapy_service.src.main", "services/therapy_service/src/main.py", dict(PHI, rate_limited=False, edge=False)),
    "diagnosis-service": Svc((REPO, REPO / "src"), "services.diagnosis_service.src.main", "services/diagnosis_service/src/main.py", dict(PHI, rate_limited=False, edge=False)),
    "memory-service": Svc((REPO, REPO / "src"), "services.memory_service.src.main", "services/memory_service/src/main.py", dict(PHI, rate_limited=False, edge=False)),
    "personality-service": Svc((REPO, REPO / "src"), "services.personality_service.src.main", "services/personality_service/src/main.py", dict(PHI, rate_limited=False, edge=False)),
    "user-service": Svc((REPO / "services" / "user-service", REPO, REPO / "src"), "src.main", "services/user-service/src/main.py", dict(PHI, rate_limited=True, edge=True)),
    "notification-service": Svc((REPO / "services" / "notification-service", REPO, REPO / "src"), "src.main", "services/notification-service/src/main.py", dict(ALL, handles_phi=False, audited=True, rate_limited=False, edge=False)),
    "analytics-service": Svc((REPO / "services" / "analytics-service" / "src", REPO, REPO / "src"), "main", "services/analytics-service/src/main.py", dict(ALL, handles_phi=False, audited=False, rate_limited=False, edge=False)),
}

# The structlog chain configure_service_logging installs (WS-04.2), as logging_chain_names() reports it.
RUNTIME_CHAIN = ["merge_contextvars", "add_log_level", "TimeStamper", "StackInfoRenderer",
                 "format_exc_info", "redact_credentials", "phi_sanitizer_processor", "JSONRenderer"]
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _probe_env() -> dict:
    """Synthetic, per-run values for every variable a service's settings require at import.

    The names are the union of what the runtime guards read and what the services'
    tests/conftest.py files set (user-service sets the USER_SERVICE_*, SECURITY_*, USER_DB_* and
    FERNET_* ones). No value is a real secret, and none is ever printed.
    """
    from cryptography.fernet import Fernet
    return {
        "AUTH_SECRET_KEY": secrets.token_urlsafe(48),
        "USER_SERVICE_JWT_SECRET_KEY": secrets.token_urlsafe(48),
        "SECURITY_JWT_SECRET": secrets.token_urlsafe(48),
        "USER_SERVICE_FIELD_ENCRYPTION_KEY": secrets.token_urlsafe(48),
        "USER_DB_PASSWORD": "probe_db_credential_only",
        "POSTGRES_PASSWORD": "probe_db_credential_only",
        "FERNET_TOKEN_KEY": Fernet.generate_key().decode(),
        "FERNET_FIELD_KEY": Fernet.generate_key().decode(),
        "ENCRYPTION_MASTER_KEY": secrets.token_hex(32),
        "ENCRYPTION_SEARCH_HASH_SALT": secrets.token_urlsafe(32),
        "PERSONALITY_REDIS_PASSWORD": secrets.token_urlsafe(24),
    }


def _probe(paths, module) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME") if k in os.environ}
    env.update(_probe_env(), PYTHONPATH=os.pathsep.join(str(p) for p in paths), ENVIRONMENT="development")
    out = subprocess.run([sys.executable, str(PROBE), module], cwd=REPO, env=env, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _assert_runtime_started_cleanly(report) -> None:
    """After start_service_runtime: logging is the runtime's alone and the runtime routes leak nothing."""
    startup = report["startup"]
    assert startup is not None, "the probe did not start the runtime (the app is not the runtime's)"
    assert startup["error"] is None, f"start-up failed: {startup['error']}"
    assert startup["ran"] is True, "the lifespan never called start_service_runtime"
    # Review hand-off (a): uvicorn's access log is off and uvicorn keeps no handler of its own,
    # even though uvicorn's default logging configuration was applied before the lifespan.
    assert startup["uvicorn_access_disabled"] is True
    assert startup["uvicorn_own_handlers"] == {name: 0 for name in UVICORN_LOGGERS}
    assert startup["root_runtime_handlers"] == 1
    expected_chain = RUNTIME_CHAIN if not report["spec"].get("debug") else [*RUNTIME_CHAIN[:-1], "ConsoleRenderer"]
    assert startup["logging_chain"] == expected_chain
    status = startup["route_status"]
    assert status["/live"] == 200 and status["/metrics"] == 200
    assert status["/ready"] in (200, 503) and status["/health"] == status["/ready"]
    assert startup["secret_in_responses"] is False, "a synthetic secret appeared in a runtime route response"
    assert startup["raw_path_in_metrics"] is False, "a raw request path or query value became a metric label"
    assert startup["service_label_in_metrics"] is True, "/metrics has no series labelled with this service"


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_service_conforms(service):
    svc = SERVICES[service]
    report = _probe(svc.paths, svc.module)
    assert report["has_spec"], "main.py must define SPEC = ServiceRuntimeSpec(...) at import"
    spec = report["spec"]
    assert spec["service_name"] == service
    for flag, expected in svc.flags.items():
        assert spec[flag] == expected, flag
    assert report["created_by_runtime"] is True
    assert {"/live", "/ready", "/health", "/metrics"} <= set(report["runtime_routes"])
    assert report["foreign_exception_handlers"] == []
    assert report["shadowed_runtime_routes"] == []
    _assert_runtime_started_cleanly(report)


def _assert_lifespan_bracketed(main_py: Path) -> None:
    """The lifespan is bracketed by the runtime in one of the two accepted forms (WS-04.F1, LIF-1).

    (a) ``async with service_runtime(app, SPEC) as runtime:`` is the lifespan's only statement
        and holds all composition, ending with a bare ``yield``;
    (b) ``runtime = await start_service_runtime(app, SPEC)`` is followed by a ``try`` (the last
        statement) that holds the composition, ends with a bare ``yield``, has no ``except``
        and no ``else`` clause, and whose ``finally`` is exactly
        ``await stop_service_runtime(runtime)``.

    The ``yield`` is the bracket body's last statement and its only ``yield``: service stops go
    through ``runtime.register_shutdown``, never after the ``yield``.

    Anything else is rejected, including the bare shape whose last statement is the stop: there,
    a composition failure before the ``yield`` or a raising stop after it skips every closer.
    """
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    src = ast.unparse(tree)
    assert "structlog.configure(" not in src and "ProductionGuards.assert_startup(" not in src
    assert "exception_handler" not in src
    lifespan = _lifespan_function(tree)
    assert _bracket_form(lifespan) is not None, (
        "the lifespan must be `async with service_runtime(app, SPEC) as runtime:` holding all "
        "composition and ending with a bare `yield`, or `runtime = await start_service_runtime(app, "
        "SPEC)` followed by a try (composition, then a bare `yield` last) whose finally is exactly "
        "`await stop_service_runtime(runtime)`; nothing after the yield: register service stops "
        "with runtime.register_shutdown"
    )


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_lifespan_is_bracketed_by_the_runtime(service):
    _assert_lifespan_bracketed(REPO / SERVICES[service].main_py)


def test_lifespan_check_accepts_the_reference_app():
    _assert_lifespan_bracketed(FIXTURES / "reference_app.py")


def test_probe_accepts_the_reference_app():
    report = _probe((Path(__file__).parent / "fixtures", REPO, REPO / "src"), "reference_app")
    assert report["has_spec"] and report["created_by_runtime"] and report["foreign_exception_handlers"] == []
    assert {"/live", "/ready", "/health", "/metrics"} <= set(report["runtime_routes"])
    assert report["shadowed_runtime_routes"] == []
    _assert_runtime_started_cleanly(report)


def test_probe_rejects_a_nonconforming_app():
    report = _probe((Path(__file__).parent / "fixtures", REPO, REPO / "src"), "nonconforming_app")
    assert report["created_by_runtime"] is False and report["foreign_exception_handlers"] != []
    assert report["has_spec"] is False and report["runtime_routes"] == []
    assert report["startup"] is None  # the probe never runs the lifespan of an app the runtime did not build


# --------------------------------------------------------------------------- AST helpers


def _is_create_service_app(func: ast.expr) -> bool:
    return (isinstance(func, ast.Name) and func.id == "create_service_app") or (
        isinstance(func, ast.Attribute) and func.attr == "create_service_app"
    )


def _lifespan_function(tree: ast.Module):
    """The function passed as ``lifespan=`` to ``create_service_app`` in this module.

    A leading docstring is not counted as the function's first statement.
    """
    names = [
        keyword.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_create_service_app(node.func)
        for keyword in node.keywords
        if keyword.arg == "lifespan" and isinstance(keyword.value, ast.Name)
    ]
    assert names, "main.py must call create_service_app(SPEC, lifespan=<function>, ...)"
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == names[0]
    ]
    assert len(functions) == 1, f"the lifespan {names[0]!r} must be defined once in main.py"
    lifespan = functions[0]
    if ast.get_docstring(lifespan, clean=False) is not None:
        lifespan = copy.copy(lifespan)
        lifespan.body = lifespan.body[1:]
    assert lifespan.body, f"the lifespan {names[0]!r} has no statements"
    return lifespan


_RUNTIME_WITH_ITEM = "service_runtime(app, SPEC) as runtime"
_RUNTIME_START = "runtime = await start_service_runtime(app, SPEC)"
_RUNTIME_STOP = "await stop_service_runtime(runtime)"


def _bracket_form(fn) -> str | None:
    """``"async_with"`` (form a), ``"try_finally"`` (form b), or None for any other shape.

    In both forms the bracket body (the ``async with`` body, or the ``try`` body) must end with
    a bare ``yield`` statement and hold no other ``yield``: no service stop code may follow the
    ``yield``, because a raising or cancelled shutdown would skip it (WS-04.F1 fix round 1).
    """
    body = fn.body
    if (
        len(body) == 1
        and isinstance(body[0], ast.AsyncWith)
        and [ast.unparse(item) for item in body[0].items] == [_RUNTIME_WITH_ITEM]
        and _ends_with_the_only_bare_yield(body[0].body)
    ):
        return "async_with"
    if (
        len(body) == 2
        and ast.unparse(body[0]) == _RUNTIME_START
        and isinstance(body[1], ast.Try | ast.TryStar)
        and not body[1].handlers  # no except clause: stop code or a second yield could sit there
        and not body[1].orelse  # no else clause: it runs after the yield, skipped on a failure
        and [ast.unparse(statement) for statement in body[1].finalbody] == [_RUNTIME_STOP]
        and _ends_with_the_only_bare_yield(body[1].body)
    ):
        return "try_finally"
    return None


def _ends_with_the_only_bare_yield(statements) -> bool:
    """The last statement is ``yield`` (no value) and no earlier statement yields."""
    if not statements:
        return False
    last = statements[-1]
    return (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Yield)
        and last.value.value is None
        and not _contains_yield(statements[:-1])
    )


def _contains_yield(statements) -> bool:
    """Whether a ``yield`` runs in ``statements`` (one inside a nested function does not count)."""
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Yield | ast.YieldFrom):
            return True
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        pending.extend(ast.iter_child_nodes(node))
    return False
