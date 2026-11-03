"""MVP-036 (part 1): every service serves /metrics; /live and /ready tell the truth; error bodies never leak exception text."""
import asyncio, json, uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Union
import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict, Field
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError, create_service_app
from solace_infrastructure.runtime.bootstrap import ServiceRuntime

router = APIRouter()

# Patient free text that the regex PHI sanitizer cannot recognise (no e-mail or phone shape).
PATIENT_TEXT = "Margaret Hollis feels hopeless about her divorce"

class EchoIn(BaseModel):
    message: str

class StrictIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str

class Mood:
    """A domain value object that rejects text it cannot parse, quoting it (as real code does)."""
    def __init__(self, value: str):
        raise ValueError(f"unknown mood: {value}")

@router.post("/mood")
async def mood(body: EchoIn):
    Mood(body.message)

@router.post("/strict")
async def strict(body: StrictIn):
    return {"ok": True}

class _Cat(BaseModel):
    kind: Literal["cat"]

class _Dog(BaseModel):
    kind: Literal["dog"]

class PetIn(BaseModel):
    pet: Annotated[Union[_Cat, _Dog], Field(discriminator="kind")]
    ref: uuid.UUID | None = None

class StoredRow(BaseModel):
    score: int

@router.post("/pets")
async def pets(body: PetIn):
    return {"ok": True}

@router.get("/corrupt")
async def corrupt():
    # Server-side data that fails validation: a pydantic.ValidationError (a ValueError subclass).
    return StoredRow.model_validate({"score": PATIENT_TEXT})

@router.get("/items/{item_id}")
async def get_item(item_id: str):
    return {"id": item_id}

@router.get("/boom")
async def boom():
    raise RuntimeError("secret detail jane@example.com")

@router.get("/bad")
async def bad():
    raise ValueError("user 123 not found")

@router.post("/echo")
async def echo(body: EchoIn):
    return body

@router.get("/headers")
async def headers(request: Request):
    return request.app.state.runtime.outbound_headers()

async def _ok():
    return True, "ok"

async def _fail():
    raise ConnectionError("redis://:p4ss@redis:6379 refused")

def _app(env="development", **kw) -> FastAPI:
    spec = ServiceRuntimeSpec(service_name="safety-service", environment=env, **kw)
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = ServiceRuntime(spec)
        yield
    return create_service_app(spec, lifespan=lifespan, routers=[(router, "/api/v1/x")], title="t")

def _samples(text: str, name: str) -> list:
    return [s for fam in text_string_to_metric_families(text) for s in fam.samples if s.name == name]

def test_metrics_route_counts_requests_by_route_template():
    with TestClient(_app()) as c:
        assert c.get("/api/v1/x/items/abc").status_code == 200
        r = c.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    hits = [s for s in _samples(r.text, "solace_http_requests_total")
            if s.labels == {"service": "safety-service", "route": "/api/v1/x/items/{item_id}", "method": "GET", "status": "200"}]
    assert hits and hits[0].value >= 1
    assert 'item_id="abc"' not in r.text and '"/api/v1/x/items/abc"' not in r.text

def test_live_ready_and_health():
    with TestClient(_app()) as c:
        assert c.get("/live").json() == {"status": "alive"}
        rt = c.app.state.runtime
        rt.register_readiness_check("postgres", _ok)
        r = c.get("/ready")
        assert r.status_code == 200 and r.json()["status"] == "ready"
        assert r.json()["checks"]["postgres"] == {"ok": True, "critical": True, "detail": "ok"}
        rt.register_readiness_check("redis", _fail)
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["status"] == "not_ready"
        assert r.json()["checks"]["redis"] == {"ok": False, "critical": True, "detail": "ConnectionError"}
        assert "p4ss" not in r.text
        assert c.get("/health").status_code == 503

def test_noncritical_failure_keeps_ready():
    with TestClient(_app()) as c:
        c.app.state.runtime.register_readiness_check("therapy-service", _fail, critical=False)
        assert c.get("/ready").status_code == 200

def test_ready_is_503_before_runtime_starts():
    app = _app()
    with TestClient(app) as c:
        del c.app.state.runtime
        assert c.get("/ready").status_code == 503

async def _finished_task():
    # Returned from a coroutine, so the portal hands back the Task itself (it awaits only the coroutine).
    return asyncio.ensure_future(asyncio.sleep(0))

async def _crashing():
    raise RuntimeError("consumer crashed jane@example.com")

def test_live_reports_a_dead_background_task():
    with TestClient(_app()) as c:
        # c.portal is the TestClient's anyio portal into the app's event loop
        loop_task = c.portal.call(_finished_task)
        assert isinstance(loop_task, asyncio.Task)
        c.portal.call(asyncio.sleep, 0.05)
        assert loop_task.done()
        c.app.state.runtime.register_background_task("consumer", loop_task)
        r = c.get("/live")
    assert r.status_code == 503 and r.json() == {"status": "dead", "task": "consumer"}

def test_register_background_task_rejects_a_non_task():
    rt = ServiceRuntime(ServiceRuntimeSpec(service_name="safety-service", environment="development"))
    with pytest.raises(TypeError):
        rt.register_background_task("consumer", None)
    coro = asyncio.sleep(0)
    try:
        with pytest.raises(TypeError):
            rt.register_background_task("consumer", coro)
    finally:
        coro.close()

def test_task_registered_after_it_exited_is_logged(capsys):
    from solace_infrastructure.runtime.logging import configure_service_logging
    spec = ServiceRuntimeSpec(service_name="safety-service", environment="development")
    configure_service_logging(spec)
    async def scenario():
        task = asyncio.ensure_future(_crashing())
        await asyncio.sleep(0.01)
        assert task.done()
        ServiceRuntime(spec).register_background_task("consumer", task)
    asyncio.run(scenario())
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert any(l.get("event") == "background_task_exited" and l.get("task") == "consumer"
               and l.get("outcome") == "error" and l.get("error_type") == "RuntimeError" for l in lines)

def test_unhandled_exception_body_has_no_exception_text():
    with TestClient(_app(), raise_server_exceptions=False) as c:
        r = c.get("/api/v1/x/boom", headers={"X-Request-ID": "req-123"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred", "request_id": "req-123"}}

def test_value_error_is_400_without_text():
    with TestClient(_app()) as c:
        r = c.get("/api/v1/x/bad", headers={"X-Request-ID": "req-9"})
    assert r.status_code == 400
    assert r.json() == {"error": {"code": "BAD_REQUEST", "message": "Invalid request", "request_id": "req-9"}}

def test_server_side_validation_error_is_500_not_400():
    with TestClient(_app(), raise_server_exceptions=False) as c:
        r = c.get("/api/v1/x/corrupt", headers={"X-Request-ID": "req-sv"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred", "request_id": "req-sv"}}

def _json_lines(out: str) -> list:
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]

def test_value_error_log_names_the_raise_site_not_the_text(capsys):
    from solace_infrastructure.runtime.logging import configure_service_logging
    configure_service_logging(ServiceRuntimeSpec(service_name="safety-service", environment="development"))
    with TestClient(_app()) as c:
        r = c.post("/api/v1/x/mood", json={"message": PATIENT_TEXT})
    assert r.status_code == 400
    out = capsys.readouterr().out
    assert "Margaret" not in out and "Hollis" not in out and "hopeless" not in out
    logged = [l for l in _json_lines(out) if l.get("event") == "request_value_error"]
    assert logged and "exception" not in logged[0]
    assert "test_mvp_036_service_app.py" in logged[0]["origin"] and logged[0]["origin"].endswith(":__init__")

def test_server_side_validation_error_logs_no_stored_text(capsys):
    from solace_infrastructure.runtime.logging import configure_service_logging
    configure_service_logging(ServiceRuntimeSpec(service_name="safety-service", environment="development"))
    with TestClient(_app(), raise_server_exceptions=False) as c:
        r = c.get("/api/v1/x/corrupt", headers={"X-Request-ID": "req-sv2"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR"
    out = capsys.readouterr().out
    assert "Margaret" not in out and "Hollis" not in out and "hopeless" not in out
    logged = [l for l in _json_lines(out) if l.get("event") == "unhandled_exception"]
    assert logged and logged[0]["request_id"] == "req-sv2" and "exception" not in logged[0]
    assert logged[0]["validation_errors"] == [{"type": "int_parsing", "field": "score"}] and logged[0]["origin"]

def test_extra_forbidden_field_never_echoes_the_client_key():
    with TestClient(_app()) as c:
        r = c.post("/api/v1/x/strict", json={"message": "hi", "SENTINELjanedoe": 1})
    assert r.status_code == 422 and "SENTINELjanedoe" not in r.text
    extra = [d for d in r.json()["error"]["details"] if d["type"] == "extra_forbidden"]
    assert extra and extra[0]["field"] == "body.[key]"

def test_validation_error_shape_has_no_input_echo():
    with TestClient(_app()) as c:
        r = c.post("/api/v1/x/echo", json={"message": 7, "x": "jane@example.com"}, headers={"X-Request-ID": "req-v"})
    body = r.json()["error"]
    assert r.status_code == 422 and body["code"] == "VALIDATION_ERROR" and body["request_id"] == "req-v"
    assert set(body["details"][0]) == {"field", "message", "type"}
    assert "jane@example.com" not in r.text

def test_validation_messages_outside_the_allowlist_never_echo_input():
    with TestClient(_app()) as c:
        r = c.post("/api/v1/x/pets", json={"pet": {"kind": "jane@example.com"}, "ref": "jane-not-a-uuid"})
    assert r.status_code == 422
    assert "jane@example.com" not in r.text and "`j`" not in r.text
    by_type = {d["type"]: d["message"] for d in r.json()["error"]["details"]}
    assert by_type.get("union_tag_invalid") == "Invalid value" and by_type.get("uuid_parsing") == "Invalid value"

def test_phi_like_inbound_id_is_replaced_and_logged_without_its_value(capsys):
    from solace_infrastructure.runtime.logging import configure_service_logging
    configure_service_logging(ServiceRuntimeSpec(service_name="safety-service", environment="development", log_level="DEBUG"))
    with TestClient(_app()) as c:
        r = c.get("/live", headers={"X-Request-ID": "123456789"})
    minted = r.headers["X-Request-ID"]
    assert minted != "123456789" and len(minted) == 36
    out = capsys.readouterr().out
    lines = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    replaced = [l for l in lines if l.get("event") == "request_id_replaced"]
    assert replaced and replaced[0]["header"] == "X-Request-ID" and replaced[0]["level"] == "debug"
    assert "123456789" not in out

def test_request_ids_are_echoed_or_minted():
    with TestClient(_app()) as c:
        r1 = c.get("/live", headers={"X-Request-ID": "abc-1", "X-Correlation-ID": "cor-1", "X-Turn-ID": "turn-1"})
        r2 = c.get("/live", headers={"X-Request-ID": "bad\nid"})
    assert (r1.headers["X-Request-ID"], r1.headers["X-Correlation-ID"], r1.headers["X-Turn-ID"]) == ("abc-1", "cor-1", "turn-1")
    assert r2.headers["X-Request-ID"] != "bad\nid" and len(r2.headers["X-Request-ID"]) == 36

def test_outbound_headers_carry_the_inbound_ids():
    with TestClient(_app()) as c:
        r = c.get("/api/v1/x/headers", headers={"X-Request-ID": "r-1", "X-Correlation-ID": "c-1", "X-Turn-ID": "t-1"})
    assert r.json() == {"X-Request-ID": "r-1", "X-Correlation-ID": "c-1", "X-Turn-ID": "t-1"}

def test_access_line_has_route_template_and_no_query(capsys):
    from solace_infrastructure.runtime.logging import configure_service_logging
    configure_service_logging(ServiceRuntimeSpec(service_name="safety-service", environment="development"))
    with TestClient(_app()) as c:
        c.get("/api/v1/x/items/abc?token=s3cr3t")
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    access = [l for l in lines if l.get("event") == "http_request" and l.get("route") == "/api/v1/x/items/{item_id}"]
    assert access and access[-1]["status"] == 200 and access[-1]["method"] == "GET" and "duration_ms" in access[-1]
    assert all("s3cr3t" not in json.dumps(l) and "/items/abc" not in json.dumps(l) for l in lines)

def test_docs_hidden_in_strict():
    with TestClient(_app("production")) as c:
        assert c.get("/openapi.json").status_code == 404 and c.get("/docs").status_code == 404
    with TestClient(_app("development")) as c:
        assert c.get("/openapi.json").status_code == 200

def test_cors_only_for_edge(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.test")
    pre = {"Origin": "https://app.example.test", "Access-Control-Request-Method": "POST"}
    with TestClient(_app(edge=True)) as c:
        assert c.options("/api/v1/x/echo", headers=pre).headers.get("access-control-allow-origin") == "https://app.example.test"
    with TestClient(_app(edge=False)) as c:
        assert "access-control-allow-origin" not in c.options("/api/v1/x/echo", headers=pre).headers

def test_strict_edge_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(StartupDependencyError):
        _app("production", edge=True)
