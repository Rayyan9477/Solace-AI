"""MVP-016: the orchestrator boots through IF-01 and its limiter uses REDIS_URL, never a missing setting."""
from __future__ import annotations
import ast
import asyncio
import pathlib
from contextlib import asynccontextmanager
import pytest

MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "main.py"
SRC = MAIN.parent
ORIGIN = "https://app.example.test"


def _strict_env(monkeypatch, environment="production", redis_url="redis://:test-only@redis:6379/0"):
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", environment)
    # An edge app in a strict environment must name its origins (create_service_app, MVP-164).
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", ORIGIN)
    if redis_url is None:
        monkeypatch.delenv("REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("REDIS_URL", redis_url)


def test_mvp_016_production_app_builds_a_redis_backed_limiter(monkeypatch):
    from services.orchestrator_service.src import main
    _strict_env(monkeypatch)
    app = main.create_application()
    limiters = [m for m in app.user_middleware if m.cls.__name__ == "RateLimitMiddleware"]
    assert len(limiters) == 1
    assert type(limiters[0].kwargs["limiter"]._store).__name__ == "RedisRateLimitStore"


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_mvp_016_strict_app_without_redis_url_fails_closed(monkeypatch, environment):
    """D-14: staging is as strict as production; neither falls back to a per-process counter."""
    from solace_infrastructure.runtime import StartupDependencyError
    from services.orchestrator_service.src import main
    _strict_env(monkeypatch, environment, redis_url=None)
    with pytest.raises(StartupDependencyError) as info:
        main.create_application()
    assert info.value.component == "rate_limiting"


def test_mvp_016_limiter_runs_inside_edge_cors_and_the_request_context(monkeypatch):
    """A 429 still carries the CORS headers a browser needs to read it, and the request ids."""
    from services.orchestrator_service.src import main
    _strict_env(monkeypatch)
    names = [m.cls.__name__ for m in main.create_application().user_middleware]  # outermost first
    assert names.index("CORSMiddleware") < names.index("RequestContextMiddleware") < names.index("RateLimitMiddleware")


def test_mvp_016_spec_declares_every_platform_flag():
    from services.orchestrator_service.src.main import SPEC
    assert SPEC.service_name == "orchestrator-service" and SPEC.event_plane == "publish_consume"
    assert (SPEC.uses_postgres, SPEC.uses_redis, SPEC.handles_phi, SPEC.audited, SPEC.rate_limited, SPEC.edge) == (True,) * 6


def test_mvp_016_lifespan_uses_the_runtime_and_swallows_nothing():
    """WS-04.F1 form (a): the async with is the only statement and nothing follows the yield."""
    fn = next(n for n in ast.walk(ast.parse(MAIN.read_text(encoding="utf-8")))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.AsyncWith)
    assert [ast.unparse(item) for item in body[0].items] == ["service_runtime(app, SPEC) as runtime"]
    assert ast.unparse(body[0].body[-1]) == "yield"
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try | ast.TryStar)]


class _FakeRuntime:
    def __init__(self):
        self.pg_pool = object()
        self.redis = None
        self.closers: list[str] = []

    def register_shutdown(self, name, closer):
        assert callable(closer)
        self.closers.append(name)


class _FakeLLM:
    is_available = False

    def __init__(self, settings):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def shutdown(self):
        pass


class _FakeBridge:
    async def stop(self):
        pass


def test_mvp_016_lifespan_composes_from_the_runtime_and_registers_every_stop(monkeypatch):
    from fastapi import FastAPI
    from services.orchestrator_service.src import main
    from services.orchestrator_service.src.agents import chat_agent
    from services.orchestrator_service.src.langgraph import supervisor
    # The lifespan sets these module globals; restore them for the rest of the suite.
    monkeypatch.setattr(chat_agent, "_llm_client", chat_agent._llm_client)
    monkeypatch.setattr(supervisor, "_supervisor_llm_client", supervisor._supervisor_llm_client)
    runtime, seen = _FakeRuntime(), {}

    @asynccontextmanager
    async def fake_service_runtime(app, spec):
        assert spec is main.SPEC
        yield runtime

    async def fake_initialize_event_bridge(*, event_bus, postgres_pool):
        seen["postgres_pool"] = postgres_pool
        return _FakeBridge()

    monkeypatch.setattr(main, "service_runtime", fake_service_runtime)
    monkeypatch.setattr(main, "initialize_event_bridge", fake_initialize_event_bridge)
    monkeypatch.setattr(main, "UnifiedLLMClient", _FakeLLM)
    monkeypatch.setattr(main, "LLMClientSettings", lambda: None)
    app = FastAPI()

    async def run():
        async with main.lifespan(app):
            seen["graph"] = app.state.compiled_graph
            seen["llm"] = app.state.llm_client
            assert app.state.graph_builder is not None and app.state.active_connections == {}

    asyncio.run(run())
    assert seen["graph"] is not None and seen["llm"].initialized
    assert seen["postgres_pool"] is runtime.pg_pool  # the outbox uses the runtime's pool, never a private one
    assert {"llm_client", "event_bridge"} <= set(runtime.closers)


class _RaisingRedis:
    """A Redis that is mid-failover: every command raises."""

    async def eval(self, *args, **kwargs):
        from redis.exceptions import ConnectionError as RedisConnectionError
        raise RedisConnectionError("synthetic outage")

    async def ttl(self, *args, **kwargs):
        from redis.exceptions import ConnectionError as RedisConnectionError
        raise RedisConnectionError("synthetic outage")

    async def aclose(self):
        pass


class _OverLimitRedis:
    """A healthy Redis whose counter is far past every policy's limit."""

    async def eval(self, *args, **kwargs):
        return 10_000

    async def ttl(self, *args, **kwargs):
        return 30

    async def aclose(self):
        pass


class _Graph:
    async def invoke(self, state, thread_id=None):
        return {"final_response": "synthetic reply", "intent": "general_chat"}


class _RecordingLogger:
    def __init__(self):
        self.records: list[tuple[str, str, dict]] = []

    def __getattr__(self, level):
        return lambda event, **fields: self.records.append((level, event, fields))


_CHAT = {"message": "synthetic message", "user_id": "user-1", "session_id": "session-1"}


def _chat_client(monkeypatch, fake_redis=None):
    """The real app (development, so the limiter is mounted) with auth and the graph stubbed."""
    from types import SimpleNamespace
    import redis.asyncio as redis_asyncio
    from fastapi.testclient import TestClient
    from services.orchestrator_service.src import main
    from services.orchestrator_service.src.api import get_graph_builder
    from solace_security.middleware import get_current_user
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", "development")
    monkeypatch.setenv("REDIS_URL", "redis://:test-only@redis:6379/0")
    if fake_redis is not None:
        monkeypatch.setattr(redis_asyncio, "from_url", lambda url, **kwargs: fake_redis)
    app = main.create_application()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id="user-1")
    app.dependency_overrides[get_graph_builder] = lambda: _Graph()
    return app, TestClient(app, raise_server_exceptions=False)


def test_mvp_016_limiter_backend_outage_fails_open_on_the_chat_route(monkeypatch):
    """WS111-CRI-1: a Redis blip must not turn a (possibly crisis) chat message into a 500."""
    from solace_security import rate_limit
    recorder = _RecordingLogger()
    monkeypatch.setattr(rate_limit, "logger", recorder)
    _, client = _chat_client(monkeypatch, _RaisingRedis())
    response = client.post("/api/v1/orchestrator/chat", json=_CHAT)
    assert response.status_code == 200
    assert response.json()["response"] == "synthetic reply"
    # The exception class only: no key, IP, user or message text.
    assert [r for r in recorder.records if r[1] == "rate_limiter_backend_error"] == [
        ("warning", "rate_limiter_backend_error", {"error_type": "ConnectionError"})
    ]


def test_mvp_016_a_real_over_limit_still_returns_429(monkeypatch):
    """Fail-open covers backend errors only: a healthy limiter's deny is still a 429."""
    _, client = _chat_client(monkeypatch, _OverLimitRedis())
    response = client.post("/api/v1/orchestrator/chat", json=_CHAT)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert int(response.headers["Retry-After"]) >= 1


def test_mvp_016_limiter_redis_client_has_socket_timeouts(monkeypatch):
    """WS111-SPE-1: a hung Redis costs a request at most the timeout, never an unbounded wait."""
    app, _ = _chat_client(monkeypatch)
    kwargs = app.state.rate_limit_redis.connection_pool.connection_kwargs
    assert kwargs["socket_connect_timeout"] == 0.5
    assert kwargs["socket_timeout"] == 0.5


def test_mvp_016_no_runtime_ddl_or_dead_state_store():
    assert not (SRC / "infrastructure" / "state.py").exists()
    assert not [p for p in SRC.rglob("*.py") if "CREATE TABLE" in p.read_text(encoding="utf-8")]
