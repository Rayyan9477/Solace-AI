# WS-11: Orchestrator edge: REST/WS API, agent clients and provider contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The orchestrator's public REST and WebSocket API is authenticated, thread-owned, bounded and documented as IF-12, and every agent and memory call reaches the right peer with an obo token, the provider's own schema, a per-hop timeout and correlation headers, surfacing failures as degraded reasons, metrics and readiness, writing each turn to memory and erasing threads and checkpoints when a user is deleted.

**Architecture:** `main.py` adopts the IF-01 runtime (W0) and from W1 builds the graph with `OrchestratorGraphBuilder.from_runtime()`; `api.py` maps WS-10's `run_turn()`/`TurnResult` (IF-10) onto IF-12 REST bodies and WS frames, minting the turn id server-side. A pooled `AgentHttpClient` per provider (W2) carries obo tokens, `Idempotency-Key` and X-Request-ID/X-Correlation-ID/X-Turn-ID, never retries an unsafe POST, and each agent adapts to its provider's canonical schema under JSON contracts (IF-11, IF-20). Thread ownership, provider session ids and the last turn id live in `orchestrator_threads` (slot 016, RLS), which also drives erasure through IF-16 and WS-10's `delete_threads`.

**Covers:** 24 register entries (blocker 4, high 6, medium 9, low 5). **Waves:** W0 (phase B: WS-11.1); W1 (phase B: WS-11.2 to WS-11.5); W2 (phase A: WS-11.6; phase B: WS-11.7 to WS-11.15); W3 (phase B: WS-11.16 to WS-11.21). **Depends on:** WS-04 (IF-01 runtime and readiness, W1 rate-limit wiring), WS-07 (IF-04 obo `service_auth_headers`, `acting_as_user`, the `cns` claim on `AuthenticatedUser`), WS-10 (IF-10 `run_turn`, `TurnResult`, `from_runtime`, `delete_threads`, `get_thread_messages`, `session_uuid`, `screen_local`, the `degraded`/`safety_flags`/`active_treatment` state contract), WS-14 (IF-16 erasure consumer helper; consent inputs), WS-15/WS-16/WS-17/WS-18 (provider schemas and provider-side contract verification, IF-11), WS-05 (IF-19 harness, IF-20 contract framework), WS-08 (IF-17 `cns`, session claims), WS-02 (slot 016 scaffold, the `default` pool with RLS), WS-01 (lazy `infrastructure.api_gateway`, uvicorn `--ws-max-size`), WS-03 (env registry entries). **Owned paths:** `services/orchestrator_service/src/{api,main,config,websocket,replay_buffer,__init__}.py`, `services/orchestrator_service/src/infrastructure/{clients,state,__init__,thread_registry}.py`, `services/orchestrator_service/src/agents/{therapy_agent,diagnosis_agent,personality_agent,assessment_agent,__init__}.py`, `services/orchestrator_service/src/langgraph/memory_node.py`, `services/orchestrator_service/tests` (steward of the shared suite), `migrations/versions/016_orchestrator_threads.py`, `docs/API-HANDOFF.md`, `docs/api/orchestrator.md`; new tests under `tests/realinfra/orchestrator-edge-contracts/`; consumer contract files under `tests/contracts/orchestrator__{therapy,diagnosis,personality,memory}/` (IF-20: the consumer writes the contract file; `orchestrator__safety/` is WS-10.16's because the `/check` consumer is WS-10 code, and `client__orchestrator/` is WS-05.10's, which WS-11 verifies as the provider). **Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

All commands run from `D:\Repo\Contextual-Chatbot` in PowerShell 5.1. Test data is synthetic; never print `.env`, tokens or key material.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/orchestrator_service/src/main.py` | Modify | IF-01 adoption, SPEC, lifespan, `from_runtime`, readiness checks, registry/factory/writer/erasure wiring | 11.1, 11.3, 11.7, 11.8, 11.12, 11.17, 11.18, 11.19 |
| `services/orchestrator_service/src/api.py` | Modify | IF-12 REST (`/chat`, `/sessions`, history, `/batch`) and WS handler over `run_turn`; ownership; limits; token lifecycle; consent gate; the token locale passed to `run_turn` (D-29) | 11.1, 11.4, 11.5, 11.7, 11.8, 11.10, 11.12, 11.14, 11.16, 11.17, 11.18, 11.19, 11.20, 11.21 |
| `services/orchestrator_service/src/websocket.py` | Modify | IF-12 frame parsing/building; `WsLimiter` (per-user cap, frame size, message rate); dead `ConnectionManager` removed | 11.5, 11.17 |
| `services/orchestrator_service/src/infrastructure/clients.py` | Modify | `assert_peer_endpoints`, `AgentHttpClient`, pooled `ServiceClientFactory`, single-probe circuit breaker, fallback metric; dead per-service clients removed | 11.2, 11.8, 11.13 |
| `services/orchestrator_service/src/infrastructure/thread_registry.py` | Create | `ThreadRegistry` (Postgres under RLS, in-memory for dev/test), `validate_thread_id`, `build_orchestrator_erasure` | 11.7, 11.18 |
| `services/orchestrator_service/src/infrastructure/state.py` | Modify (delete) | dead `StatePersistenceManager` and its runtime DDL removed (D-01) | 11.1 |
| `services/orchestrator_service/src/infrastructure/__init__.py` | Modify | exports | 11.1, 11.7 |
| `services/orchestrator_service/src/agents/{therapy,diagnosis,personality,assessment}_agent.py` | Modify | compose URLs, obo, pooled transport, provider schemas and lifecycles, degraded reasons; therapy relays the turn locale (D-29) | 11.2, 11.8, 11.9, 11.10, 11.11, 11.13, 11.21 |
| `services/orchestrator_service/src/agents/__init__.py` | Modify | `configure_agent_runtime` | 11.8 |
| `services/orchestrator_service/src/langgraph/memory_node.py` | Modify | obo, `session_uuid`, timeout, pooled transport, `MemoryTurnWriter` | 11.2, 11.8, 11.12, 11.13 |
| `migrations/versions/016_orchestrator_threads.py` | Modify | `orchestrator_threads` table with RLS | 11.6 |
| `tests/contracts/orchestrator__{therapy,diagnosis,personality,memory}/*.json` | Create | IF-11 consumer contract files (the IF-12 client contracts are WS-05.10's; WS-11.15 verifies them as the provider) | 11.9, 11.10, 11.11, 11.12, 11.21 (therapy `message.json` gains `locale`) |
| `docs/API-HANDOFF.md`, `docs/api/orchestrator.md` | Modify / Create | IF-12 documentation | 11.15, 11.19, 11.21 |
| `services/orchestrator_service/tests/test_mvp_*.py`, `tests/realinfra/orchestrator-edge-contracts/*.py` | Create | per-task tests | all |
| `services/orchestrator_service/tests/{test_api,test_ws_replay_endpoint,test_history_idor,test_infrastructure,test_memory_node,test_agent_url_paths,test_personality_endpoints,test_sprint6_regression}.py`; `test_rate_limit_wiring.py`, `test_trusted_proxy_hops.py` (deleted in 11.3) | Modify | existing tests of WS-11 code updated to the new contracts | 11.1, 11.3, 11.4, 11.5, 11.7, 11.8, 11.17 |

## Tasks

### Task WS-11.1: Adopt the IF-01 runtime; limiter Redis from REDIS_URL; CORS from CORS_ALLOWED_ORIGINS
**Covers:** MVP-016, MVP-164
**Severity:** blocker
**Wave:** 0
**Depends on:** WS-04.4, WS-01.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: staging is as strict as production; every guard uses `is_strict_environment`); D-01 (default assumed: no runtime DDL, so the dead `orchestrator_checkpoints` DDL goes)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/main.py` (module-level `SPEC`; lifespan starts with `start_service_runtime(app, SPEC)` and ends with `stop_service_runtime(runtime)`; `create_application()` returns `create_service_app(SPEC, lifespan=lifespan, routers=[(router, "/api/v1/orchestrator")], title=...)`; W0-only `_register_rate_limiting(app)` builds its Redis client from `REDIS_URL`; delete `configure_logging`, the ProductionGuards call, the PHI ORM block, service-auth/`ServiceClientFactory` construction, `request_tracking_middleware`, the exception handlers, the `/`, `/ready`, `/live`, `/health` routes and the `cors_origins` setting)
- Modify: `services/orchestrator_service/src/api.py` (delete `/health/detailed` and `HealthResponse`)
- Modify: `services/orchestrator_service/src/infrastructure/state.py` (delete the file)
- Modify: `services/orchestrator_service/src/infrastructure/__init__.py` (drop the `state` exports)
- Test: `services/orchestrator_service/tests/test_mvp_016_runtime_adoption.py`, `services/orchestrator_service/tests/test_mvp_164_cors_allowed_origins.py` (new); `services/orchestrator_service/tests/test_infrastructure.py` (remove `TestStatePersistence`), `services/orchestrator_service/tests/test_api.py` (remove `TestHealthResponse`)
**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `create_service_app` (WS-04.2, WS-04.3), `start_service_runtime`, `stop_service_runtime` (WS-04.4); `infrastructure.api_gateway.rate_limiting.create_solace_rate_limiter(redis_client=)` (W0 only; importable without Kong or jose after WS-01.3); IF-02 `REDIS_URL`, `CORS_ALLOWED_ORIGINS`.
- Produces: `main.SPEC = ServiceRuntimeSpec(service_name="orchestrator-service", environment=<ORCHESTRATOR_ENVIRONMENT>, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=True, edge=True)`; `main.create_application() -> FastAPI`; `main.app`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_016_runtime_adoption.py`
```python
"""MVP-016: the orchestrator boots through IF-01 and its limiter uses REDIS_URL, never a missing setting."""
from __future__ import annotations
import ast
import pathlib
import pytest

MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "main.py"
SRC = MAIN.parent

def test_mvp_016_production_app_builds_a_redis_backed_limiter(monkeypatch):
    from services.orchestrator_service.src import main
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", "redis://:test-only@redis:6379/0")
    app = main.create_application()
    limiters = [m for m in app.user_middleware if m.cls.__name__ == "RateLimitMiddleware"]
    assert len(limiters) == 1
    assert type(limiters[0].kwargs["limiter"]._store).__name__ == "RedisRateLimitStore"

def test_mvp_016_spec_declares_every_platform_flag():
    from services.orchestrator_service.src.main import SPEC
    assert SPEC.service_name == "orchestrator-service" and SPEC.event_plane == "publish_consume"
    assert (SPEC.uses_postgres, SPEC.uses_redis, SPEC.handles_phi, SPEC.audited, SPEC.rate_limited, SPEC.edge) == (True,) * 6

def test_mvp_016_lifespan_uses_the_runtime_and_swallows_nothing():
    fn = next(n for n in ast.walk(ast.parse(MAIN.read_text(encoding="utf-8")))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert ast.unparse(body[0]) == "runtime = await start_service_runtime(app, SPEC)"
    assert ast.unparse(body[-1]) == "await stop_service_runtime(runtime)"
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]

def test_mvp_016_no_runtime_ddl_or_dead_state_store():
    assert not (SRC / "infrastructure" / "state.py").exists()
    assert not [p for p in SRC.rglob("*.py") if "CREATE TABLE" in p.read_text(encoding="utf-8")]
```
and in `services/orchestrator_service/tests/test_mvp_164_cors_allowed_origins.py`
```python
"""MVP-164: browser clients on the configured origin pass CORS preflight; others do not."""
from fastapi.testclient import TestClient

def _preflight(monkeypatch, origin):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.test")
    monkeypatch.delenv("ORCHESTRATOR_CORS_ORIGINS", raising=False)
    from services.orchestrator_service.src.main import create_application
    return TestClient(create_application()).options(
        "/api/v1/orchestrator/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})

def test_mvp_164_cors_allowed_origins_drives_preflight(monkeypatch):
    assert _preflight(monkeypatch, "https://app.example.test").headers.get("access-control-allow-origin") == "https://app.example.test"

def test_mvp_164_unlisted_origin_is_refused(monkeypatch):
    assert "access-control-allow-origin" not in _preflight(monkeypatch, "https://evil.example.test").headers
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_016_runtime_adoption.py -o "addopts=" -p no:logging -q` and the same command with `test_mvp_164_cors_allowed_origins.py`
Expected: FAIL: `create_application()` raises `RuntimeError: Rate limiting is required in production` because `settings.redis_url` does not exist on `OrchestratorAppSettings` (the `AttributeError` is swallowed, so `redis_client=None`); `SPEC` is missing; `state.py` holds `CREATE TABLE IF NOT EXISTS orchestrator_checkpoints`; the preflight has no `access-control-allow-origin` because the app reads `ORCHESTRATOR_CORS_ORIGINS` (empty).
- [ ] **Step 3: Implement** the adoption as listed. Lifespan order: `runtime = await start_service_runtime(app, SPEC)`; create `UnifiedLLMClient(LLMClientSettings())` and `await initialize()` (IF-15 never raises outside strict for missing keys); `configure_chat_agent_llm`/`configure_supervisor_llm`; build the graph exactly as today (`from_runtime` arrives in WS-11.3); replay buffer as today; `event_bridge = await initialize_event_bridge(event_bus=get_event_bus(), postgres_pool=runtime.pg_pool)`; register closers with `runtime.register_shutdown`; `yield`; `await stop_service_runtime(runtime)`. `_register_rate_limiting(app)` keeps the existing middleware but builds `redis.asyncio.from_url(os.environ["REDIS_URL"], decode_responses=True)` when `REDIS_URL` is set and never logs the URL; it is deleted in WS-11.3 once the bootstrap installs `install_rate_limiting`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/bootstrap` (WS-04's conformance test imports this app). Expected: all pass. Boot in the dev and strict profiles is proven by WS-05's `tests/e2e/test_w0_boot_smoke.py` at the W0 barrier; record the test-count drop from the removed `TestStatePersistence`/`TestHealthResponse` in the ledger.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.2: Agent and memory calls use the compose URLs and obo tokens
**Covers:** MVP-020, MVP-021
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-10.3, WS-10.4, WS-07.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: short-lived obo service tokens for every user-scoped call; user JWTs are never forwarded or stored)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/therapy_agent.py`, `diagnosis_agent.py`, `personality_agent.py`, `assessment_agent.py` (`service_url = Field(validation_alias="SERVICE_<NAME>_SERVICE_URL")` with `populate_by_name=True`; legacy `ORCHESTRATOR_*_AGENT_SERVICE_URL` no longer read; every request carries `await service_auth_headers("<target>-service", on_behalf_of=user_id)`)
- Modify: `services/orchestrator_service/src/langgraph/memory_node.py` (obo headers; `session_uuid(session_id)` instead of `UUID(session_id)`)
- Modify: `services/orchestrator_service/src/infrastructure/clients.py` (`MemoryServiceClient.assemble_context(..., headers=None)`; new `assert_peer_endpoints(environment: str) -> None`)
- Create: `services/orchestrator_service/tests/test_mvp_020_agent_service_urls.py` (includes the client-module prefix lock moved from WS-05.7, WS-05 request 15), `services/orchestrator_service/tests/test_mvp_021_obo_headers.py`
- Test: `services/orchestrator_service/tests/test_mvp_020_agent_service_urls.py`, `services/orchestrator_service/tests/test_mvp_021_obo_headers.py`; `services/orchestrator_service/tests/test_memory_node.py` (patch `service_auth_headers`; invalid-session case now maps through `session_uuid`)
**Interfaces:**
- Consumes: IF-04 `service_auth_headers(target_service, *, on_behalf_of=None)`; IF-02 `SERVICE_{SAFETY,THERAPY,DIAGNOSIS,MEMORY,PERSONALITY}_SERVICE_URL`; IF-01 `StartupDependencyError`; WS-10 `events.session_uuid`.
- Produces: `assert_peer_endpoints(environment: str) -> None` (strict: raises `StartupDependencyError` naming any of the five peer URLs that is unset, `localhost` or `127.0.0.1`).
- Produces for requesting tasks: the versioned-prefix lock for this workstream's client module, `test_orchestrator_clients_use_versioned_provider_prefixes` (every `/api/...` string constant in `clients.py`, f-string heads included, starts with `/api/v1/<provider>/`), moved out of WS-05.7's alignment test at WS-05's request 15 because the module is WS-11's; it stays green through WS-11.8 and WS-11.13, and WS-05.10's contract-coverage guard checks that it exists.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_020_agent_service_urls.py`
```python
"""MVP-020: agents read SERVICE_*_SERVICE_URL (what compose sets); strict boots refuse localhost peers."""
import ast
import pathlib
import re
import pytest
from services.orchestrator_service.src.agents.assessment_agent import AssessmentAgentSettings
from services.orchestrator_service.src.agents.diagnosis_agent import DiagnosisAgentSettings
from services.orchestrator_service.src.agents.personality_agent import PersonalityAgentSettings
from services.orchestrator_service.src.agents.therapy_agent import TherapyAgentSettings

PEERS = {"SERVICE_SAFETY_SERVICE_URL": "http://safety-service:8002", "SERVICE_THERAPY_SERVICE_URL": "http://therapy-service:8006",
         "SERVICE_DIAGNOSIS_SERVICE_URL": "http://diagnosis-service:8004", "SERVICE_MEMORY_SERVICE_URL": "http://memory-service:8005",
         "SERVICE_PERSONALITY_SERVICE_URL": "http://personality-service:8007"}

@pytest.fixture
def compose_env(monkeypatch):
    for k, v in PEERS.items():
        monkeypatch.setenv(k, v)
    for k in ("THERAPY", "DIAGNOSIS", "PERSONALITY", "ASSESSMENT"):
        monkeypatch.delenv(f"ORCHESTRATOR_{k}_AGENT_SERVICE_URL", raising=False)

def test_mvp_020_agents_read_compose_service_urls(compose_env):
    assert TherapyAgentSettings().service_url == "http://therapy-service:8006"
    assert DiagnosisAgentSettings().service_url == "http://diagnosis-service:8004"
    assert AssessmentAgentSettings().service_url == "http://diagnosis-service:8004"
    assert PersonalityAgentSettings().service_url == "http://personality-service:8007"

def test_mvp_020_legacy_agent_url_names_are_retired(monkeypatch):
    monkeypatch.delenv("SERVICE_THERAPY_SERVICE_URL", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_THERAPY_AGENT_SERVICE_URL", "http://legacy:1")
    assert TherapyAgentSettings().service_url == "http://localhost:8006"
    assert TherapyAgentSettings(service_url="http://therapy:8006").service_url == "http://therapy:8006"

@pytest.mark.parametrize("env", ["production", "staging"])
def test_mvp_020_strict_boot_refuses_localhost_peers(monkeypatch, env):
    for k in PEERS:
        monkeypatch.delenv(k, raising=False)
    from solace_infrastructure.runtime import StartupDependencyError
    from services.orchestrator_service.src.infrastructure.clients import assert_peer_endpoints
    with pytest.raises(StartupDependencyError):
        assert_peer_endpoints(env)

def test_mvp_020_strict_boot_accepts_compose_peers(compose_env):
    from services.orchestrator_service.src.infrastructure.clients import assert_peer_endpoints
    assert assert_peer_endpoints("production") is None

def test_orchestrator_clients_use_versioned_provider_prefixes():
    # moved here from WS-05.7's tests/alignment check (WS-05 request 15): the client module is WS-11's
    source = pathlib.Path(__file__).resolve().parents[1] / "src" / "infrastructure" / "clients.py"
    paths = [n.value for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
             if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("/api/")]
    assert paths, "no /api/ path constants found"
    assert [p for p in paths if not re.match(r"^/api/v1/(safety|therapy|diagnosis|memory|personality)/", p)] == []
```
and in `services/orchestrator_service/tests/test_mvp_021_obo_headers.py`
```python
"""MVP-021: every agent and memory call carries an obo service token for the user."""
from __future__ import annotations
import httpx
import pytest
from services.orchestrator_service.src.agents import assessment_agent, diagnosis_agent, personality_agent, therapy_agent
from services.orchestrator_service.src.langgraph import memory_node
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
SESSION = "5c1d2e3f-0000-4000-8000-000000000004"
LONG = "I have been feeling very low and tired for weeks and I do not know why this keeps happening to me"

@pytest.fixture
def wire(monkeypatch):
    seen, calls = [], []
    real = httpx.AsyncClient
    def handler(req):
        seen.append(req)
        return httpx.Response(503, json={"detail": "unavailable"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    async def fake(target, *, on_behalf_of=None):
        calls.append((target, on_behalf_of))
        return {"Authorization": f"Bearer obo-{target}"}
    for mod in (therapy_agent, diagnosis_agent, personality_agent, assessment_agent, memory_node):
        monkeypatch.setattr(mod, "service_auth_headers", fake, raising=False)
    return seen, calls

@pytest.mark.asyncio
@pytest.mark.parametrize("run,target", [
    (lambda s: therapy_agent.TherapyAgent(therapy_agent.TherapyAgentSettings(service_url="http://therapy-service:8006")).process(s), "therapy-service"),
    (lambda s: diagnosis_agent.DiagnosisAgent(diagnosis_agent.DiagnosisAgentSettings(service_url="http://diagnosis-service:8004")).process(s), "diagnosis-service"),
    (lambda s: personality_agent.PersonalityAgent(personality_agent.PersonalityAgentSettings(service_url="http://personality-service:8007")).process(s), "personality-service"),
    (lambda s: assessment_agent.assessment_agent_node(s), "diagnosis-service"),
    (lambda s: memory_node.MemoryRetrievalNode(memory_node.MemoryNodeSettings(enable_retrieval=True)).process(s), "memory-service"),
])
async def test_mvp_021_calls_carry_an_obo_token(wire, run, target):
    seen, calls = wire
    state = create_initial_state(USER, SESSION, LONG)
    state["intent"] = "emotional_support"
    await run(state)
    assert seen, "no request was made"
    assert all(r.headers.get("Authorization") == f"Bearer obo-{target}" for r in seen)
    assert (target, USER) in calls

@pytest.mark.asyncio
async def test_mvp_021_memory_maps_a_client_session_id(wire):
    seen, _ = wire
    await memory_node.MemoryRetrievalNode(memory_node.MemoryNodeSettings(enable_retrieval=True)).process(
        create_initial_state(USER, "chat-2026-10-18", "hello"))
    assert seen, "a non-UUID session id must be mapped, not dropped"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_020_agent_service_urls.py -o "addopts=" -p no:logging -q` and the same command with `test_mvp_021_obo_headers.py`
Expected: FAIL: settings return `http://localhost:8006` etc. (they read `ORCHESTRATOR_*_AGENT_SERVICE_URL`); `assert_peer_endpoints` is missing; requests carry no `Authorization` header (MVP-021); the memory node makes no request for `chat-2026-10-18` (`UUID()` raises). `test_orchestrator_clients_use_versioned_provider_prefixes` is a lock and passes today (every constant is already versioned, for example `clients.py:249`, `:361`, `:408`); it guards this task's and WS-11.8's rewrites of the module.
- [ ] **Step 3: Implement** the settings aliases, the obo header call in each client method (headers merged per request), `session_uuid` in the memory node, and `assert_peer_endpoints(environment)` over a fresh `ServiceEndpoints()` (not the cached `get_config()`), raising when `is_strict_environment(environment)` and any peer host is unset or loopback.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.3: main.py W1 wiring: graph from the runtime, honest readiness, peer validation, bootstrap-owned rate limiting
**Covers:** MVP-160
**Severity:** medium
**Wave:** 1
**Depends on:** WS-11.2, WS-10.5, WS-04.6, WS-09.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-26 (default assumed: the bootstrap's `install_rate_limiting` is the only limiter)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/main.py` (lifespan: `assert_peer_endpoints(SPEC.environment)`, `graph_builder = await OrchestratorGraphBuilder.from_runtime(runtime)`, `register_orchestrator_readiness(...)`; delete `_register_rate_limiting`, `_trusted_proxy_hops`, `_make_principal_resolver`, `_rate_limiting_disabled`)
- Test: `services/orchestrator_service/tests/test_mvp_160_readiness.py` (new); delete `services/orchestrator_service/tests/test_rate_limit_wiring.py` and `services/orchestrator_service/tests/test_trusted_proxy_hops.py` (their behaviour moves to the bootstrap's tests; record the count drop in the ledger)
**Interfaces:**
- Consumes: IF-01 `ServiceRuntime.register_readiness_check(name, check, *, critical=True)`; the W1 bootstrap wiring of rate limiting and the event plane (WS-04.6); IF-15 `async UnifiedLLMClient.readiness() -> tuple[bool, str]` (WS-09.4); IF-10 `OrchestratorGraphBuilder.from_runtime` (async); `assert_peer_endpoints` (WS-11.2).
- Produces: `register_orchestrator_readiness(runtime, *, graph_builder, llm_client, http: httpx.AsyncClient, endpoints: ServiceEndpoints) -> None` registering `graph` (critical), `llm` (non-critical; the client's own `llm_client.readiness` when a client exists, which is the orchestrator's `/ready` half of WS-09's CR-7 for WS-09.4; `(False, "llm: disabled")` when no client was built) and `peer:<service>` for safety, therapy, diagnosis, memory and personality (non-critical, `GET {url}/live` with a 1 s timeout). The event-plane check stays the bootstrap's.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_160_readiness.py`
```python
"""MVP-160: readiness reflects the graph and reports the LLM and peers without making them fatal."""
from __future__ import annotations
import ast
import pathlib
import httpx
import pytest
from services.orchestrator_service.src.config import ServiceEndpoints

MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "main.py"

class FakeRuntime:
    def __init__(self):
        self.checks = {}
    def register_readiness_check(self, name, check, *, critical=True):
        self.checks[name] = (check, critical)

class CompiledStub:
    def get_compiled_graph(self):
        return object()

@pytest.mark.asyncio
async def test_mvp_160_graph_is_critical_llm_and_peers_are_not():
    from services.orchestrator_service.src.main import register_orchestrator_readiness
    rt = FakeRuntime()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    register_orchestrator_readiness(rt, graph_builder=CompiledStub(), llm_client=None, http=http, endpoints=ServiceEndpoints())
    assert rt.checks["graph"][1] is True and rt.checks["llm"][1] is False
    peers = {n for n in rt.checks if n.startswith("peer:")}
    assert peers == {"peer:safety-service", "peer:therapy-service", "peer:diagnosis-service", "peer:memory-service", "peer:personality-service"}
    assert all(rt.checks[n][1] is False for n in peers) and "event_plane" not in rt.checks
    assert await rt.checks["llm"][0]() == (False, "llm: disabled")
    assert (await rt.checks["peer:safety-service"][0]())[0] is False
    assert (await rt.checks["graph"][0]())[0] is True

class GatedLLM:
    async def readiness(self):
        return (False, "disabled: LLM_PHI_EGRESS_APPROVED is false")

@pytest.mark.asyncio
async def test_mvp_160_llm_check_is_the_clients_own_readiness():
    from services.orchestrator_service.src.main import register_orchestrator_readiness
    rt, llm = FakeRuntime(), GatedLLM()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    register_orchestrator_readiness(rt, graph_builder=CompiledStub(), llm_client=llm, http=http, endpoints=ServiceEndpoints())
    assert rt.checks["llm"] == (llm.readiness, False)
    assert await rt.checks["llm"][0]() == (False, "disabled: LLM_PHI_EGRESS_APPROVED is false")

def test_mvp_160_bootstrap_owns_rate_limiting(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    import services.orchestrator_service.src.main as m
    assert not hasattr(m, "_register_rate_limiting") and not hasattr(m, "_trusted_proxy_hops")
    assert sum(1 for mw in m.create_application().user_middleware if mw.cls.__name__ == "RateLimitMiddleware") == 1

def test_mvp_160_lifespan_builds_the_graph_from_the_runtime():
    src = MAIN.read_text(encoding="utf-8")
    assert "await OrchestratorGraphBuilder.from_runtime(runtime)" in src
    assert "register_orchestrator_readiness(" in src and "assert_peer_endpoints(SPEC.environment)" in src
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_160_readiness.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `ImportError: cannot import name 'register_orchestrator_readiness'` (readiness only checks that a graph object exists, so a dead LLM or peer never shows); two `RateLimitMiddleware` entries (the W1 bootstrap plus the local one); the lifespan still constructs the builder directly.
- [ ] **Step 3: Implement** `register_orchestrator_readiness` and the lifespan changes; the `llm` check is `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)` when a client was built and a constant `(False, "llm: disabled")` check otherwise; the peer probe client is one pooled `httpx.AsyncClient` closed through `runtime.register_shutdown`; delete the local limiter helpers and the two superseded test files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/bootstrap`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.4: REST chat runs through run_turn and returns the IF-12 body; identity comes from the token
**Covers:** MVP-297
**Supports:** MVP-079 and MVP-017 (crisis-safe REST surface)
**Severity:** low
**Wave:** 1
**Depends on:** WS-10.5, WS-10.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-11 (default assumed: the code is the contract, IF-12 shapes); D-22 (default assumed: resources never depend on the graph)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py` (`ChatMessageRequest` = `{message: 1..4000, thread_id?, session_id?}` with any body `user_id` ignored; `ChatMessageResponse` = IF-12 fields; `/chat` and `/batch` call `run_turn(user_id=current_user.user_id, ..., turn_id=uuid4(), channel="rest")`; `/sessions` ignores the body `user_id`; a `run_turn` exception is logged with `exc_info` and returns 200 with `degraded=["internal_error"]` and, when `screen_local(message)` is ELEVATED or above, the crisis object and `crisis_reply_text`)
- Test: `services/orchestrator_service/tests/test_mvp_297_rest_chat_contract.py` (new); `services/orchestrator_service/tests/test_api.py` (model tests on the IF-12 fields)
**Interfaces:**
- Consumes: IF-10 `run_turn`, `TurnResult`, `CrisisInfo.to_dict()`; WS-10 `crisis_screen.screen_local`, `response.safety_copy.crisis_reply_text`, `SAFE_FALLBACK_TEXT`; IF-07 `crisis_resources`, `RESOURCES_VERSION`.
- Produces: IF-12 REST response `{response, thread_id, session_id, turn_id, crisis, degraded, agents, request_id}`; `turn_id` minted per message server-side (any inbound `X-Turn-ID` is ignored).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_297_rest_chat_contract.py`
```python
"""MVP-297 / IF-12: REST chat takes identity from the token and returns the IF-12 body via run_turn."""
from __future__ import annotations
from uuid import UUID
from fastapi import FastAPI
from fastapi.testclient import TestClient
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.langgraph.graph_builder import CrisisInfo, TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"
CRISIS = CrisisInfo(level="HIGH", categories=["suicidal_ideation"],
                    resources=[{"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"}],
                    resources_version="test-1")

class FakeBuilder:
    def __init__(self, crisis=None, exc=None):
        self.calls, self.crisis, self.exc = [], crisis, exc
    async def run_turn(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return TurnResult(response_text="I'm here with you.", crisis=self.crisis, degraded=[], agents=["chat"],
                          turn_id=kw["turn_id"], safety_flags={}, crisis_watch_until=None)

def _client(builder):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder = builder
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS)
    return TestClient(app)

def test_mvp_297_body_user_id_is_ignored_and_the_body_is_if12():
    b = FakeBuilder()
    r = _client(b).post(f"{BASE}/chat", json={"message": "hello", "user_id": USER.upper()})
    assert r.status_code == 200
    assert b.calls[0]["user_id"] == USER and b.calls[0]["channel"] == "rest"
    body = r.json()
    assert set(body) == {"response", "thread_id", "session_id", "turn_id", "crisis", "degraded", "agents", "request_id"}
    assert body["turn_id"] == str(b.calls[0]["turn_id"]) and body["crisis"] is None

def test_mvp_297_crisis_object_is_returned():
    assert _client(FakeBuilder(crisis=CRISIS)).post(f"{BASE}/chat", json={"message": "x"}).json()["crisis"] == CRISIS.to_dict()

def test_mvp_297_turn_id_is_minted_server_side():
    b = FakeBuilder()
    forged = "11111111-1111-4111-8111-111111111111"
    _client(b).post(f"{BASE}/chat", json={"message": "hello"}, headers={"X-Turn-ID": forged})
    assert str(b.calls[0]["turn_id"]) != forged and isinstance(b.calls[0]["turn_id"], UUID)

def test_mvp_297_message_length_is_bounded():
    assert _client(FakeBuilder()).post(f"{BASE}/chat", json={"message": "a" * 4001}).status_code == 422

def test_mvp_297_runner_failure_still_shows_resources_for_crisis_text():
    r = _client(FakeBuilder(exc=RuntimeError("bug"))).post(f"{BASE}/chat", json={"message": "I want to kill myself"})
    body = r.json()
    assert r.status_code == 200 and body["degraded"] == ["internal_error"]
    assert body["crisis"]["level"] in ("HIGH", "CRITICAL") and "988" in body["response"]
    assert "bug" not in r.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_297_rest_chat_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL: 422 for the missing `session_id`, then 403 "Cannot send messages as another user" for the re-cased UUID (exact string compare, MVP-297); the handler calls `invoke`, not `run_turn`; a runner exception gives a bare 500 with no resources.
- [ ] **Step 3: Implement** the request/response models and the `run_turn` mapping; `session_id` and `thread_id` default to `str(uuid4())`; `request_id` comes from the bound structlog contextvar (IF-01 middleware) or a fresh uuid4.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (5 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.5: The WebSocket speaks IF-12 (frames, safety.alert first, errors, subprotocol token)
**Covers:** MVP-072
**Severity:** high
**Wave:** 1
**Depends on:** WS-11.4, WS-10.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-11 (default assumed: documented frame names with the legacy `{type:'message', message}` alias); D-27 (default assumed: token via `Sec-WebSocket-Protocol` `['solace.v1', 'bearer.<jwt>']`, `?token=` kept and deprecated)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/websocket.py` (`parse_client_frame(raw: str) -> ClientFrame`, `safety_alert_frame(crisis)`, `chat_response_frame(result, *, thread_id, client_message_id)`, `error_frame(code, message)`)
- Modify: `services/orchestrator_service/src/api.py` (WS handler: token from the subprotocol or `?token=`; close 4401 on bad token; per message mint `turn_id`, call `run_turn(..., channel="ws")`, send `safety.alert` before `chat.response` when `crisis` is set; `UNSUPPORTED_FRAME`, `INVALID_FRAME`, `INTERNAL` error frames; the `processing` frame removed; replay buffer appends the new frames)
- Test: `services/orchestrator_service/tests/test_mvp_072_ws_protocol.py` (new); `services/orchestrator_service/tests/test_ws_replay_endpoint.py` (fake builder exposes `run_turn`; frames are `chat.response`)
**Interfaces:**
- Consumes: IF-10 `run_turn`, `TurnResult`, `CrisisInfo`; `solace_security.middleware._get_jwt_manager().validate_access_token_async`.
- Produces: IF-12 frames: server `{type:'safety.alert', level, categories, resources, resources_version}`, `{type:'chat.response', content, thread_id, turn_id, client_message_id, degraded, agents}`, `{type:'pong'}`, `{type:'error', code, message}`; informational `{type:'connected', session_id, thread_id, connection_id}`, always the first server frame after `accept` (an IF-12 addition, listed under New interfaces and documented by WS-11.15; clients and WS-05.6's E2E read it before the first `chat.message`, so "safety.alert first" means first after the client's message, crisis-review F5); close 4401 for authentication failure.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_072_ws_protocol.py`
```python
"""MVP-072: documented WS frames are handled, unknown frames get an error, and crisis resources come first."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.langgraph.graph_builder import CrisisInfo, TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"
CRISIS = CrisisInfo(level="HIGH", categories=["suicidal_ideation"],
                    resources=[{"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"}],
                    resources_version="test-1")

class FakeBuilder:
    def __init__(self, crisis=None):
        self.calls, self.crisis = [], crisis
    async def run_turn(self, **kw):
        self.calls.append(kw)
        return TurnResult(response_text=f"echo:{kw['message']}", crisis=self.crisis, degraded=[], agents=["chat"],
                          turn_id=kw["turn_id"], safety_flags={}, crisis_watch_until=None)

def _jwt(ok=True):
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(return_value=MagicMock(success=ok, payload={"sub": USER} if ok else None,
                                                                     user_id=USER if ok else None))
    return patch("solace_security.middleware._get_jwt_manager", return_value=m), m

def _client(builder):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.active_connections, app.state.replay_buffer = builder, {}, None
    return TestClient(app)

@pytest.mark.timeout(10)
def test_mvp_072_chat_message_gets_safety_alert_then_chat_response():
    b = FakeBuilder(crisis=CRISIS)
    p, _ = _jwt()
    with p, _client(b).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "chat.message", "content": "I want to kill myself", "client_message_id": "c1"})
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "safety.alert", "level": "HIGH", "categories": ["suicidal_ideation"],
                                     "resources": CRISIS.resources, "resources_version": "test-1"}
        resp = ws.receive_json()
        assert set(resp) == {"type", "content", "thread_id", "turn_id", "client_message_id", "degraded", "agents"}
        assert (resp["type"], resp["content"], resp["client_message_id"]) == ("chat.response", "echo:I want to kill myself", "c1")
        assert resp["turn_id"] == str(b.calls[0]["turn_id"]) and b.calls[0]["channel"] == "ws"
        assert ws.receive_json()["type"] == "pong"

@pytest.mark.timeout(10)
@pytest.mark.parametrize("send,code", [("json-bogus", "UNSUPPORTED_FRAME"), ("text-garbage", "INVALID_FRAME")])
def test_mvp_072_bad_frames_get_error_frames(send, code):
    p, _ = _jwt()
    with p, _client(FakeBuilder()).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        ws.receive_json()
        ws.send_json({"type": "bogus"}) if send == "json-bogus" else ws.send_text("{not json")
        ws.send_json({"type": "ping"})
        frame = ws.receive_json()
        assert frame["type"] == "error" and frame["code"] == code

@pytest.mark.timeout(10)
def test_mvp_072_legacy_message_alias_still_works():
    p, _ = _jwt()
    with p, _client(FakeBuilder()).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        ws.receive_json()
        ws.send_json({"type": "message", "message": "hello"})
        assert ws.receive_json()["content"] == "echo:hello"

@pytest.mark.timeout(10)
def test_mvp_072_token_via_subprotocol():
    p, m = _jwt()
    with p, _client(FakeBuilder()).websocket_connect(f"{BASE}/ws/s1", subprotocols=["solace.v1", "bearer.sub-token"]) as ws:
        assert ws.accepted_subprotocol == "solace.v1"
        m.validate_access_token_async.assert_awaited_with("sub-token")

@pytest.mark.timeout(10)
def test_mvp_072_bad_token_closes_4401():
    p, _ = _jwt(ok=False)
    with p, pytest.raises(WebSocketDisconnect) as e:
        with _client(FakeBuilder()).websocket_connect(f"{BASE}/ws/s1?token=bad"):
            pass
    assert e.value.code == 4401
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_072_ws_protocol.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the `chat.message` and `bogus` frames are dropped silently, so the first frame received is `pong`; invalid JSON gives `{"type":"error","message":"Invalid JSON"}` with no `code`; the connection without `?token=` is refused because the query parameter is required; a bad token closes with 4001.
- [ ] **Step 3: Implement** the frame helpers in `websocket.py` and the handler; the subprotocol item `bearer.<jwt>` is read from `websocket.scope["subprotocols"]` and `accept(subprotocol="solace.v1")` echoes only `solace.v1`; the turn's `run_turn` call runs per frame, and an unexpected exception gives `{type:'error', code:'INTERNAL'}` plus, when `screen_local(content)` is ELEVATED or above, a `safety.alert` with `crisis_resources()` first.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.6: Slot 016 creates orchestrator_threads under RLS
**Covers:** MVP-070
**Severity:** high
**Wave:** 2
**Depends on:** WS-02.3, WS-14.11, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record); D-02 (default assumed: services run as `solace_app`, NOBYPASSRLS)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/016_orchestrator_threads.py` (`upgrade()`/`downgrade()` only)
- Test: `tests/realinfra/orchestrator-edge-contracts/test_mvp_070_threads_schema.py` (new)
**Interfaces:**
- Consumes: IF-03 slot scaffold `016_orchestrator_threads` and the NULLIF policy form (WS-02.3); the W2 schema lane order, in which slot 012's W2 body (WS-14.11) runs first and slot 016 follows it; IF-19 `pg_migrated` (WS-05.2).
- Produces: table `orchestrator_threads(thread_id VARCHAR(200) PRIMARY KEY, user_id UUID NOT NULL, session_id VARCHAR(200) NOT NULL, channel TEXT NOT NULL CHECK (channel IN ('rest','ws')), therapy_session_id UUID NULL, diagnosis_session_id UUID NULL, diagnosis_turns INTEGER NOT NULL DEFAULT 0, last_turn_id UUID NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), last_turn_at TIMESTAMPTZ NOT NULL DEFAULT now())`, index on `user_id`, `ENABLE` and `FORCE ROW LEVEL SECURITY`, policy `user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid` for USING and WITH CHECK.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/orchestrator-edge-contracts/test_mvp_070_threads_schema.py`
```python
"""MVP-070: orchestrator_threads isolates thread ownership per user under RLS as solace_app."""
from __future__ import annotations
from uuid import uuid4
import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
INSERT = "INSERT INTO orchestrator_threads (thread_id, user_id, session_id, channel) VALUES ($1, $2, 's1', 'rest')"

@pytest.mark.asyncio
async def test_mvp_070_thread_rows_are_isolated_per_user(pg_migrated):
    a, b = str(uuid4()), str(uuid4())
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", a)
        await conn.execute(INSERT, f"t-{a}", a)
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await conn.execute(INSERT, f"t-{b}", b)
        assert await conn.fetchval("SELECT count(*) FROM orchestrator_threads WHERE thread_id=$1", f"t-{a}") == 1
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", b)
        assert await conn.fetchval("SELECT count(*) FROM orchestrator_threads") == 0
        assert await conn.fetchval(f"{INSERT} ON CONFLICT (thread_id) DO NOTHING RETURNING user_id", f"t-{a}", b) is None
        await conn.execute("SELECT set_config('app.current_user_id', '', false)")
        assert await conn.fetchval("SELECT count(*) FROM orchestrator_threads") == 0
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/orchestrator-edge-contracts/test_mvp_070_threads_schema.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `asyncpg.exceptions.UndefinedTableError: relation "orchestrator_threads" does not exist` (no thread-to-user binding exists anywhere, MVP-070).
- [ ] **Step 3: Implement** `upgrade()` with the DDL above and `downgrade()` dropping the policy and the table.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q`. Expected: all pass (chain upgrade/downgrade/upgrade; every policy uses NULLIF)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.7: Thread ownership on chat, history, batch and WebSocket paths
**Covers:** MVP-070
**Severity:** high
**Wave:** 2
**Depends on:** WS-11.6, WS-11.5, WS-10.10, WS-02.6, WS-07.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `services/orchestrator_service/src/infrastructure/thread_registry.py`
- Modify: `services/orchestrator_service/src/api.py` (`/chat`, `/batch` items and the WS handshake call `bind_or_verify`; `GET /sessions/{session_id}/history` checks ownership then calls `graph_builder.get_thread_messages(thread_id, limit=limit)`; foreign or malformed thread ids give 404 `{error:{code:'THREAD_NOT_FOUND', message, request_id}}`, WS closes 4403; `record_turn` after each turn)
- Modify: `services/orchestrator_service/src/main.py` (`app.state.thread_registry = await create_thread_registry(runtime)`)
- Modify: `services/orchestrator_service/src/infrastructure/__init__.py` (exports)
- Test: `services/orchestrator_service/tests/test_mvp_070_thread_ownership.py`, `tests/realinfra/orchestrator-edge-contracts/test_mvp_070_thread_registry.py` (new); `services/orchestrator_service/tests/test_history_idor.py` (registry-based ownership; channel_values.user_id is no longer trusted); `test_mvp_297_rest_chat_contract.py`, `test_mvp_072_ws_protocol.py`, `test_ws_replay_endpoint.py` (set `app.state.thread_registry = InMemoryThreadRegistry()`)
**Interfaces:**
- Consumes: IF-03 `ConnectionPoolManager.acquire(pool_name)` on the one `default` pool (WS-02.6; sets the RLS GUC from the request context and clears it on release); WS-10 `get_thread_messages`, `CheckpointAccessError` (WS-10.10); IF-04 `acting_as_user` (WS-07.1; WS frames).
- Produces: `@dataclass(frozen=True) ThreadBinding(thread_id, user_id, session_id, channel, therapy_session_id: UUID | None = None, diagnosis_session_id: UUID | None = None, diagnosis_turns: int = 0, last_turn_id: UUID | None = None)`; `class ThreadNotOwned(Exception)`; protocol `ThreadRegistry` with `async bind_or_verify(*, thread_id, user_id, session_id, channel) -> ThreadBinding`, `async get_binding(thread_id, user_id) -> ThreadBinding | None`, `async set_provider_session(thread_id, user_id, provider: Literal["therapy", "diagnosis"], session_id: UUID | None, *, diagnosis_turns: int | None = None) -> None`, `async record_turn(thread_id, user_id, turn_id: UUID) -> None`, `async list_threads(user_id) -> list[str]`, `async delete_user_threads(user_id) -> int`; `PostgresThreadRegistry(pool_name: str = "default")`, `InMemoryThreadRegistry()` (refused in strict), `async create_thread_registry(runtime) -> ThreadRegistry`; `validate_thread_id(thread_id: str, user_id: str) -> bool` (a UUID string or `f"{user_id}:<session>"` only).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_070_thread_ownership.py`
```python
"""MVP-070: a user can neither write to nor read another user's thread."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.testclient import TestClient
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult

VICTIM = "8a3f0d4e-0000-4000-8000-000000000001"
ATTACKER = "9b4e1f5a-0000-4000-8000-000000000009"
BASE = "/api/v1/orchestrator"

class FakeBuilder:
    def __init__(self):
        self.calls = []
    async def run_turn(self, **kw):
        self.calls.append(kw)
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"],
                          safety_flags={}, crisis_watch_until=None)
    async def get_thread_messages(self, thread_id, *, limit=50):
        return [{"role": "user", "content": "lithium dose question"}]

def _client(user, registry, builder):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry = builder, registry
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=user, token_type=TokenType.ACCESS)
    return TestClient(app)

def test_mvp_070_foreign_thread_is_not_found_and_owner_keeps_access():
    reg, b = InMemoryThreadRegistry(), FakeBuilder()
    victim, attacker = _client(VICTIM, reg, b), _client(ATTACKER, reg, b)
    tid = victim.post(f"{BASE}/chat", json={"message": "hello"}).json()["thread_id"]
    r = attacker.post(f"{BASE}/chat", json={"message": "hi", "thread_id": tid})
    assert r.status_code == 404 and r.json()["error"]["code"] == "THREAD_NOT_FOUND"
    assert [c["user_id"] for c in b.calls] == [VICTIM]
    hist = attacker.get(f"{BASE}/sessions/s1/history", params={"thread_id": tid})
    assert hist.status_code == 404 and "lithium" not in hist.text
    assert victim.get(f"{BASE}/sessions/s1/history", params={"thread_id": tid}).status_code == 200

def test_mvp_070_rest_cannot_claim_another_users_ws_thread_id():
    r = _client(ATTACKER, InMemoryThreadRegistry(), FakeBuilder()).post(
        f"{BASE}/chat", json={"message": "hi", "thread_id": f"{VICTIM}:s1"})
    assert r.status_code == 404
```
and in `tests/realinfra/orchestrator-edge-contracts/test_mvp_070_thread_registry.py`
```python
"""MVP-070: the Postgres registry enforces ownership as solace_app under RLS."""
from __future__ import annotations
from urllib.parse import urlparse
from uuid import uuid4
import pytest
from pydantic import SecretStr
from solace_common.request_context import acting_as_user
from solace_infrastructure.database.connection_manager import ConnectionPoolManager
from solace_infrastructure.postgres import PostgresSettings
from services.orchestrator_service.src.infrastructure.thread_registry import PostgresThreadRegistry, ThreadNotOwned

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

@pytest.fixture
async def registry(pg_migrated):
    u = urlparse(pg_migrated.dsn_app)
    await ConnectionPoolManager.register_pool("orchestrator_threads_test", PostgresSettings(
        host=u.hostname, port=u.port, database=u.path.lstrip("/"), user=u.username, password=SecretStr(u.password)))
    yield PostgresThreadRegistry(pool_name="orchestrator_threads_test")
    await ConnectionPoolManager.close_pool("orchestrator_threads_test")

@pytest.mark.asyncio
async def test_mvp_070_registry_binds_verifies_lists_and_deletes(registry):
    a, b, tid = str(uuid4()), str(uuid4()), str(uuid4())
    with acting_as_user(a):
        bound = await registry.bind_or_verify(thread_id=tid, user_id=a, session_id="s1", channel="rest")
        assert bound.user_id == a and await registry.list_threads(a) == [tid]
    with acting_as_user(b), pytest.raises(ThreadNotOwned):
        await registry.bind_or_verify(thread_id=tid, user_id=b, session_id="s1", channel="rest")
    with acting_as_user(a):
        assert await registry.delete_user_threads(a) == 1 and await registry.list_threads(a) == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_070_thread_ownership.py -o "addopts=" -p no:logging -q` and `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/orchestrator-edge-contracts/test_mvp_070_thread_registry.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: ...thread_registry`; with only that stubbed, the attacker's `/chat` on the victim's thread returns 200 and runs a turn (MVP-070: no thread-to-user binding).
- [ ] **Step 3: Implement** the registry: Postgres `bind_or_verify` runs `INSERT ... ON CONFLICT (thread_id) DO NOTHING RETURNING user_id` then `SELECT` under RLS, raising `ThreadNotOwned` when the row exists but is invisible; `validate_thread_id` runs first; `InMemoryThreadRegistry` mirrors the semantics keyed by `(thread_id)` with the owner recorded; `create_thread_registry(runtime)` returns the Postgres registry when `runtime.pg_pool` exists, else in-memory with a WARNING, and raises `StartupDependencyError` in strict without Postgres. WS frames run inside `acting_as_user(user_id)`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/realinfra/orchestrator-edge-contracts` with `$env:SOLACE_REQUIRE_REALINFRA="1"`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.8: Pooled agent transport: per-hop timeout, no unsafe retries, correlation headers on every hop
**Covers:** MVP-159, MVP-162, MVP-293, MVP-356
**Severity:** medium
**Wave:** 2
**Depends on:** WS-11.7, WS-11.2, WS-10.5
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/infrastructure/clients.py` (`AgentHttpClient`, `AgentCallTimeout`, `AgentTransportSettings`; `ServiceClientFactory(transport=None)` with `agent_client(service)` cached per provider and `close_all()`; `CircuitBreaker` admits one HALF_OPEN probe at a time; delete the unused `PersonalityServiceClient`, `DiagnosisServiceClient`, `TherapyServiceClient`, `TreatmentServiceClient`, `SafetyServiceClient` subclasses)
- Modify: `services/orchestrator_service/src/agents/__init__.py` (`configure_agent_runtime(*, clients: ServiceClientFactory, threads: ThreadRegistry | None = None) -> None`)
- Modify: `services/orchestrator_service/src/agents/therapy_agent.py`, `diagnosis_agent.py`, `personality_agent.py`, `assessment_agent.py`, `services/orchestrator_service/src/langgraph/memory_node.py` (use `agent_client(...)`; memory node applies `MemoryNodeSettings.timeout_seconds` and returns `degraded=["memory_unavailable"]` on failure)
- Modify: `services/orchestrator_service/src/main.py` (factory and `configure_agent_runtime` in the lifespan; `register_shutdown("agent_clients", factory.close_all)`)
- Modify: `services/orchestrator_service/src/api.py` (each WS frame binds `request_id` (new uuid4), `correlation_id` (one per connection) and `turn_id` via `structlog.contextvars.bound_contextvars` before `run_turn`)
- Test: `services/orchestrator_service/tests/test_mvp_159_agent_transport.py`, `services/orchestrator_service/tests/test_mvp_162_memory_deadline.py`, `services/orchestrator_service/tests/test_mvp_356_correlation_ids.py` (new); `services/orchestrator_service/tests/test_agent_url_paths.py`, `test_personality_endpoints.py`, `test_sprint6_regression.py` (H-43 removed with the dead client; H-25 kept), `test_infrastructure.py` (`TestServiceClients` on the new factory)
**Interfaces:**
- Consumes: IF-04 `service_auth_headers` (carries X-Request-ID, X-Correlation-ID, X-Turn-ID from context); IF-02 `ORCHESTRATOR_AGENT_TIMEOUT_SECONDS` (8).
- Produces: `AgentHttpClient(*, service: str, base_url: str, timeout_seconds: float, transport: httpx.AsyncBaseTransport | None = None, honours_idempotency_key: bool = False)` with `async post_json(path: str, body: dict, *, user_id: str, idempotency_key: str | None = None) -> httpx.Response` (one attempt bounded by `asyncio.wait_for(timeout_seconds)`, raising `AgentCallTimeout`; one retry only on 5xx or transport error and only when `honours_idempotency_key` and a key is given; never on 4xx) and `async aclose()`; `ServiceClientFactory.agent_client(service: str) -> AgentHttpClient`; `configure_agent_runtime(...)`; Idempotency-Key format `f"{turn_id}:{provider}:{operation}"`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_159_agent_transport.py`
```python
"""MVP-159 / MVP-293 / MVP-356: pooled clients, per-hop timeout, no unsafe retries, correlation headers."""
from __future__ import annotations
import asyncio
import httpx
import pytest
from services.orchestrator_service.src.agents.therapy_agent import TherapyAgent, TherapyAgentSettings
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
TURN = "4b6f1c2a-0000-4000-8000-000000000002"

@pytest.fixture(autouse=True)
def headers(monkeypatch):
    async def fake(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "req-1", "X-Correlation-ID": "corr-1", "X-Turn-ID": TURN}
    monkeypatch.setattr(cl, "service_auth_headers", fake, raising=False)

def _client(handler, timeout=8.0, idem=False):
    return cl.AgentHttpClient(service="therapy-service", base_url="http://therapy-service:8006", timeout_seconds=timeout,
                              transport=httpx.MockTransport(handler), honours_idempotency_key=idem)

@pytest.mark.asyncio
async def test_mvp_159_therapy_agent_never_reposts_a_failed_message(monkeypatch):
    posts = []
    real = httpx.AsyncClient
    def handler(req):
        posts.append(req)
        return httpx.Response(503)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    state = create_initial_state(USER, "s1", "I need coping strategies")
    state["intent"] = "coping_strategy"
    await TherapyAgent(TherapyAgentSettings(service_url="http://therapy-service:8006")).process(state)
    assert len([p for p in posts if p.url.path.endswith("/message")]) <= 1

@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 409, 500, 503])
async def test_mvp_159_post_without_idempotency_support_is_single_shot(status):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(status, json={})
    resp = await _client(handler).post_json("/api/v1/therapy/sessions/x/message", {"a": 1}, user_id=USER,
                                            idempotency_key=f"{TURN}:therapy:message")
    assert resp.status_code == status and len(calls) == 1
    assert calls[0].headers["Idempotency-Key"] == f"{TURN}:therapy:message"

@pytest.mark.asyncio
async def test_mvp_159_timeout_is_bounded_and_not_retried():
    calls = []
    async def handler(req):
        calls.append(req)
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={})
    with pytest.raises(cl.AgentCallTimeout):
        await _client(handler, timeout=0.2).post_json("/x", {}, user_id=USER)
    assert len(calls) == 1

@pytest.mark.asyncio
async def test_mvp_159_honouring_provider_gets_one_retry_on_5xx_only():
    seq = iter([httpx.Response(503), httpx.Response(200, json={})])
    calls = []
    def handler(req):
        calls.append(req)
        return next(seq)
    assert (await _client(handler, idem=True).post_json("/x", {}, user_id=USER, idempotency_key="k")).status_code == 200
    assert len(calls) == 2

@pytest.mark.asyncio
async def test_mvp_356_every_hop_carries_correlation_headers():
    seen = []
    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={})
    await _client(handler).post_json("/x", {}, user_id=USER)
    h = seen[0].headers
    assert (h["X-Request-ID"], h["X-Correlation-ID"], h["X-Turn-ID"], h["Authorization"]) == ("req-1", "corr-1", TURN, "Bearer obo")

def test_mvp_293_factory_pools_one_client_per_provider():
    f = cl.ServiceClientFactory()
    assert f.agent_client("therapy-service") is f.agent_client("therapy-service")
    assert f.agent_client("therapy-service") is not f.agent_client("diagnosis-service")
```
and in `services/orchestrator_service/tests/test_mvp_162_memory_deadline.py`
```python
"""MVP-162: a hung memory service costs at most the node timeout; HALF_OPEN admits one probe."""
from __future__ import annotations
import asyncio
import time
import httpx
import pytest
from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker
from services.orchestrator_service.src.langgraph import memory_node
from services.orchestrator_service.src.langgraph.memory_node import MemoryNodeSettings, MemoryRetrievalNode
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
SESSION = "5c1d2e3f-0000-4000-8000-000000000004"

@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_mvp_162_hung_memory_costs_at_most_the_node_timeout(monkeypatch):
    async def hang(req):
        await asyncio.sleep(30)
        return httpx.Response(200, json={})
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(hang)}))
    async def fake(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo"}
    monkeypatch.setattr(memory_node, "service_auth_headers", fake, raising=False)
    t0 = time.perf_counter()
    out = await MemoryRetrievalNode(MemoryNodeSettings(timeout_seconds=1.0)).process(create_initial_state(USER, SESSION, "hello"))
    assert time.perf_counter() - t0 < 1.5
    assert out["memory_context"]["fallback_mode"] is True and out["degraded"] == ["memory_unavailable"]

@pytest.mark.asyncio
async def test_mvp_162_half_open_admits_a_single_probe():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
    await cb.record_failure()
    assert sorted(await asyncio.gather(*(cb.allow_request() for _ in range(3)))) == [False, False, True]
```
and in `services/orchestrator_service/tests/test_mvp_356_correlation_ids.py`
```python
"""MVP-356: WS turns bind request, correlation and turn ids so every downstream hop can be traced."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"

@pytest.mark.timeout(10)
def test_mvp_356_ws_frames_bind_ids():
    seen = []
    class Builder:
        async def run_turn(self, **kw):
            seen.append(dict(structlog.contextvars.get_contextvars()))
            return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"],
                              safety_flags={}, crisis_watch_until=None)
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry, app.state.active_connections, app.state.replay_buffer = \
        Builder(), InMemoryThreadRegistry(), {}, None
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(return_value=MagicMock(success=True, payload={"sub": USER}, user_id=USER))
    with patch("solace_security.middleware._get_jwt_manager", return_value=m), \
            TestClient(app).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat.message", "content": "hi"})
        first = ws.receive_json()
        ws.send_json({"type": "chat.message", "content": "again"})
        ws.receive_json()
    assert seen[0]["turn_id"] == first["turn_id"] and seen[0]["request_id"] and seen[0]["correlation_id"]
    assert seen[0]["request_id"] != seen[1]["request_id"] and seen[0]["correlation_id"] == seen[1]["correlation_id"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_159_agent_transport.py -o "addopts=" -p no:logging -q`, then the same command with `test_mvp_162_memory_deadline.py` and with `test_mvp_356_correlation_ids.py`
Expected: FAIL: the therapy agent POSTs the same message 3 times on 503 (MVP-159); `AgentHttpClient` is missing; the memory node ignores `timeout_seconds` and waits on a 30 s client with 3 retries until pytest-timeout fires (MVP-162); all three HALF_OPEN calls are admitted; the WS handler binds no ids (MVP-356).
- [ ] **Step 3: Implement** the transport and factory as specified, then move each agent and the memory node onto `agent_client(...)`; agents send `Idempotency-Key` on every POST and keep `honours_idempotency_key=False` for every provider until a provider contract says otherwise.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass; record the removed H-43 and dead-client tests in the ledger
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.9: Therapy provider contract and session lifecycle
**Covers:** MVP-022, MVP-078
**Severity:** blocker
**Wave:** 2
**Depends on:** WS-11.8, WS-11.7, WS-05.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-30 (default assumed: the orchestrator sends no diagnosis in the start context and never asserts one)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/therapy_agent.py` (start a session once per thread via `POST /api/v1/therapy/sessions/start` with `treatment_plan_id = uuid5(NAMESPACE_URL, "solace:therapy-plan:" + user_id)` and `context={}`; store it with `set_provider_session(..., "therapy", sid)`; message the stored session; on 404 restart once and retry with the same Idempotency-Key; `conversation_history` as `[{role, content}]`; lowercase enums parsed; the provider's risk mapped to a `safety_flags` update as for diagnosis in WS-11.10 (crisis-review F12): `risk_level = max_level("ELEVATED" if safety_alerts or risk_categories carry a CATEGORIES code else "NONE", the response's risk_level)` (WS-15.5's `TherapyResponse.risk_level`, parsed case-insensitively, absent means NONE; never lowered), `categories` = the `CATEGORIES` codes of `risk_categories` and `safety_alerts`; emitted only at ELEVATED or above; `active_treatment` returned)
- Create: `tests/contracts/orchestrator__therapy/session_start.json`, `tests/contracts/orchestrator__therapy/message.json`
- Test: `services/orchestrator_service/tests/test_mvp_078_therapy_contract.py` (new); `services/orchestrator_service/tests/test_agents.py` (TechniqueDTO/TherapyResponse fixtures in lowercase)
**Interfaces:**
- Consumes: IF-11 therapy calls; `ThreadRegistry` (WS-11.7); `AgentHttpClient` (WS-11.8); IF-20 `ContractFile` and `assert_request_matches(contract: ContractFile, request)` (WS-05.7); WS-10 state contract (`safety_flags` partial update, `active_treatment`); WS-15.5 `TherapyResponse.{risk_level, risk_categories, degraded}` (optional on the wire until WS-15.5 lands in the same wave).
- Produces: therapy agent state update `{"agent_results": [...], "active_treatment": {"therapy_session_id": str, "treatment_plan_id": str}, "safety_flags"?: {"risk_level": <the provider's level, at least ELEVATED>, "categories": [...], "signal_source": "therapy"}}` (WS-10.5 publishes a graph-raised HIGH/CRITICAL once per turn, so a therapy-only HIGH reaches the user as a HIGH crisis object, crisis-review F12); `message.json`'s 200 `body_schema` declares `risk_level`, `risk_categories` and `degraded` as optional properties, without `additionalProperties: false`; contract files verified by WS-15, and the `orchestrator__therapy` consumer directory of WS-05's request #10 (needed by WS-05.10).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_078_therapy_contract.py`
```python
"""MVP-078 / MVP-022 (therapy): start a session once per thread and speak the provider's schema."""
from __future__ import annotations
import json
import pathlib
from uuid import NAMESPACE_URL, UUID, uuid5
import httpx
import pytest
from solace_common.request_context import acting_as_user
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.agents import configure_agent_runtime
from services.orchestrator_service.src.agents.therapy_agent import TherapyAgent, TherapyResponse
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
SID = "7d2b9c1e-0000-4000-8000-000000000003"
PLAN = str(uuid5(NAMESPACE_URL, f"solace:therapy-plan:{USER}"))
CONTRACTS = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__therapy"
REPLY = {"session_id": SID, "user_id": USER, "response_text": "Let's try box breathing.", "current_phase": "working",
         "technique_applied": {"technique_id": "6f1e2d3c-0000-4000-8000-000000000005", "name": "Box breathing",
                               "modality": "mindfulness", "category": "relaxation", "description": "d"},
         "homework_assigned": [], "safety_alerts": ["self_harm"], "next_steps": [], "processing_time_ms": 12}

@pytest.fixture
def wire(monkeypatch):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    log, statuses = [], {"message": [200]}
    def handler(req):
        log.append(req)
        if req.url.path == "/api/v1/therapy/sessions/start":
            return httpx.Response(201, json={"session_id": SID, "user_id": USER, "treatment_plan_id": PLAN, "session_number": 1,
                                             "current_phase": "opening", "initial_message": "Hi", "suggested_agenda": []})
        status = statuses["message"].pop(0) if len(statuses["message"]) > 1 else statuses["message"][0]
        return httpx.Response(status, json=REPLY if status == 200 else {"error": {"code": "SESSION_NOT_FOUND"}})
    reg = InMemoryThreadRegistry()
    configure_agent_runtime(clients=ServiceClientFactory(transport=httpx.MockTransport(handler)), threads=reg)
    return log, statuses, reg

async def _turn(reg, text="I need coping strategies"):
    state = create_initial_state(USER, "s1", text, thread_id="t1")
    state["intent"] = "coping_strategy"
    with acting_as_user(USER):
        await reg.bind_or_verify(thread_id="t1", user_id=USER, session_id="s1", channel="rest")
        return await TherapyAgent().process(state)

def test_mvp_022_lowercase_provider_enums_parse():
    assert TherapyResponse.from_dict(REPLY).current_phase.value == "working"

@pytest.mark.asyncio
async def test_mvp_078_first_turn_starts_a_session_then_messages_it(wire):
    log, _, reg = wire
    out = await _turn(reg)
    assert [r.url.path for r in log] == ["/api/v1/therapy/sessions/start", f"/api/v1/therapy/sessions/{SID}/message"]
    start, msg = json.loads(log[0].content), json.loads(log[1].content)
    assert start["user_id"] == USER and start["treatment_plan_id"] == PLAN and "diagnosis" not in start.get("context", {})
    assert all(set(m) == {"role", "content"} for m in msg["conversation_history"])
    assert_request_matches(ContractFile.model_validate(json.loads((CONTRACTS / "session_start.json").read_text(encoding="utf-8"))), log[0])
    assert_request_matches(ContractFile.model_validate(json.loads((CONTRACTS / "message.json").read_text(encoding="utf-8"))), log[1])
    assert out["agent_results"][0]["response_content"] == "Let's try box breathing."
    assert out["active_treatment"] == {"therapy_session_id": SID, "treatment_plan_id": PLAN}
    assert out["safety_flags"] == {"risk_level": "ELEVATED", "categories": ["self_harm"], "signal_source": "therapy"}

@pytest.mark.asyncio
async def test_mvp_078_later_turns_reuse_the_session(wire):
    log, _, reg = wire
    await _turn(reg)
    await _turn(reg, "and another thing")
    assert [r.url.path for r in log].count("/api/v1/therapy/sessions/start") == 1

@pytest.mark.asyncio
async def test_mvp_078_lost_session_is_restarted_once(wire):
    log, statuses, reg = wire
    await _turn(reg)
    statuses["message"] = [404, 200]
    log.clear()
    out = await _turn(reg, "still there?")
    assert [r.url.path for r in log] == [f"/api/v1/therapy/sessions/{SID}/message", "/api/v1/therapy/sessions/start",
                                         f"/api/v1/therapy/sessions/{SID}/message"]
    assert log[0].headers["Idempotency-Key"] == log[2].headers["Idempotency-Key"]
    assert out["agent_results"][0]["response_content"] == "Let's try box breathing."

@pytest.mark.asyncio
async def test_crisis_review_f12_provider_risk_level_is_honoured_and_never_lowered(wire, monkeypatch):
    _, _, reg = wire
    monkeypatch.setitem(REPLY, "risk_level", "HIGH")
    monkeypatch.setitem(REPLY, "risk_categories", ["self_harm"])
    out = await _turn(reg)
    assert out["safety_flags"] == {"risk_level": "HIGH", "categories": ["self_harm"], "signal_source": "therapy"}
    monkeypatch.setitem(REPLY, "risk_level", "NONE")
    assert (await _turn(reg, "and again"))["safety_flags"]["risk_level"] == "ELEVATED"   # safety_alerts still carry a code
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_078_therapy_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `TherapyResponse.from_dict` raises `ValueError: 'working' is not a valid SessionPhase` (MVP-022); the agent POSTs `/sessions/s1/message` for a session therapy never created (MVP-078); the contract files do not exist; no provider risk ever reaches `safety_flags`, so a therapy-detected HIGH is invisible to `run_turn` (crisis-review F12).
- [ ] **Step 3: Implement** the lifecycle and schema mapping in `therapy_agent.py` (enums use the provider's lowercase values; the old uppercase enums go), and write the two contract JSON files (`headers_required`: Authorization, X-Request-ID, X-Correlation-ID, X-Turn-ID, Idempotency-Key; `body_schema` from `services/therapy_service/src/schemas.py` `SessionStartResponse` and `TherapyResponse`, with synthetic examples).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.10: Diagnosis provider contract and session lifecycle (signals only)
**Covers:** MVP-022, MVP-170
**Severity:** blocker
**Wave:** 2
**Depends on:** WS-11.9
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-30 (default assumed: diagnosis contributes symptom and risk signals only; hypotheses and diagnosis text are never shown to users and are not kept in graph state)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/diagnosis_agent.py` (start once per thread via `/api/v1/diagnosis/session/start`; `/assess` with the stored session id, lowercase `current_phase` from the previous response's `phase`, `[{role, content}]` history; parse `phase` and lowercase symptom types; `response_content=None`; the provider's `risk_level`/`risk_categories` (NI-16.2) become a `safety_flags` update at the provider's level and never below ELEVATED, falling back to the `CATEGORIES` codes in `safety_flags` at ELEVATED when `risk_level` is absent; a 404 `SESSION_NOT_FOUND` starts a new session and retries once; a 403 `USER_ID_MISMATCH` is a failed result with no retry and no new session; end via `/session/end` after 12 assessments (`ORCHESTRATOR_DIAGNOSIS_SESSION_MAX_TURNS`) or phase `closure`; new `end_provider_sessions(thread_id, user_id)`)
- Modify: `services/orchestrator_service/src/agents/assessment_agent.py` (same session and schema; no user-facing diagnosis text)
- Modify: `services/orchestrator_service/src/api.py` (WS disconnect calls `end_provider_sessions` within 2 s, best effort)
- Create: `tests/contracts/orchestrator__diagnosis/session_start.json`, `assess.json`, `session_end.json`, `assess_user_mismatch.json`, `assess_session_not_found.json` with WS-16's seed ids (R6): user `00000000-0000-4000-a000-000000000001`; `assess.json` session `00000000-0000-4000-d000-000000000001`; `session_end.json` session `00000000-0000-4000-d000-000000000002`; `assess_user_mismatch.json` session `00000000-0000-4000-d000-000000000003` (owned by `00000000-0000-4000-a000-000000000002`), response 403 with `detail.code` const `USER_ID_MISMATCH`; `assess_session_not_found.json` session `00000000-0000-4000-d000-0000000000ff`, response 404 with `detail.code` const `SESSION_NOT_FOUND`; turn `00000000-0000-4000-e000-000000000001`. Every assess file lists `Authorization` and `X-Turn-ID` in `headers_required`; the 200 `body_schema` declares `risk_level`, `risk_categories` and `degraded` as optional properties and does not set `additionalProperties: false`.
- Test: `services/orchestrator_service/tests/test_mvp_170_diagnosis_lifecycle.py` (new); `services/orchestrator_service/tests/test_agents.py` (DTO fixtures lowercase)
**Interfaces:**
- Consumes: IF-11 diagnosis calls; WS-16 NI-16.2 (`risk_level`, `risk_categories`, `degraded` on `AssessmentResponse`) and NI-16.4 (404 `{"detail": {"code": "SESSION_NOT_FOUND"}}`, 403 `{"detail": {"code": "USER_ID_MISMATCH"}}`); `ThreadRegistry.set_provider_session(..., "diagnosis", ..., diagnosis_turns=)`; `AgentHttpClient`; IF-20 `ContractFile`, `assert_request_matches` (WS-05.7).
- Produces: `async end_provider_sessions(thread_id: str, user_id: str) -> None`; diagnosis state update `{"agent_results": [AgentResult(DIAGNOSIS, response_content=None, metadata={symptoms_count, phase})], "metadata": {"diagnosis_phase", "existing_symptoms"}, "safety_flags"?: {"risk_level": <provider level, at least ELEVATED>, "categories": [...], "signal_source": "diagnosis"}}`; the `orchestrator__diagnosis` contract files with the fixed seed ids and the 403/404 interactions that WS-16 requested (R6, needed by WS-16.12) and the consumer directory of WS-05's request #10 (needed by WS-05.10); WS-16's non-blocking R7 consumer behaviour (X-Turn-ID through WS-11.8's hop headers, restart once on 404, no retry on 403, provider `risk_level` honoured).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_170_diagnosis_lifecycle.py`
```python
"""MVP-170 / MVP-022 (diagnosis): a real diagnosis session per thread, provider schema, signals only."""
from __future__ import annotations
import json
import pathlib
import httpx
import pytest
from solace_common.request_context import acting_as_user
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.agents import configure_agent_runtime
from services.orchestrator_service.src.agents.diagnosis_agent import DiagnosisAgent, end_provider_sessions
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
DSID = "3e4f5a6b-0000-4000-8000-000000000006"
CONTRACTS = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__diagnosis"
ASSESS = {"assessment_id": "1a2b3c4d-0000-4000-8000-000000000007", "user_id": USER, "session_id": DSID, "phase": "history",
          "extracted_symptoms": [{"symptom_id": "2b3c4d5e-0000-4000-8000-000000000008", "name": "low mood", "description": "d",
                                  "symptom_type": "emotional", "severity": "MILD", "confidence": 0.7}],
          "differential": {"primary": {"hypothesis_id": "4c5d6e7f-0000-4000-8000-000000000010", "name": "HYPOTHESIS-X", "confidence": 0.5}},
          "reasoning_chain": [], "next_question": "How long?", "response_text": "It sounds like HYPOTHESIS-X",
          "confidence_score": 0.6, "safety_flags": ["suicidal_ideation"], "risk_level": "HIGH",
          "risk_categories": ["suicidal_ideation"], "degraded": [], "processing_time_ms": 5}
SEED_USER = "00000000-0000-4000-a000-000000000001"
ERRORS = {404: "SESSION_NOT_FOUND", 403: "USER_ID_MISMATCH"}

@pytest.fixture
def wire(monkeypatch):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    log, phase = [], {"value": "history", "assess": [200]}
    def handler(req):
        log.append(req)
        if req.url.path.endswith("/session/start"):
            return httpx.Response(201, json={"session_id": DSID, "user_id": USER, "session_number": 1, "initial_phase": "rapport",
                                             "greeting": "Hi", "loaded_context": False})
        if req.url.path.endswith("/session/end"):
            return httpx.Response(200, json={"session_id": DSID, "user_id": USER, "duration_minutes": 1, "messages_exchanged": 2})
        status = phase["assess"].pop(0) if len(phase["assess"]) > 1 else phase["assess"][0]
        if status != 200:
            return httpx.Response(status, json={"detail": {"code": ERRORS[status]}})
        return httpx.Response(200, json={**ASSESS, "phase": phase["value"]})
    reg = InMemoryThreadRegistry()
    configure_agent_runtime(clients=ServiceClientFactory(transport=httpx.MockTransport(handler)), threads=reg)
    return log, phase, reg

async def _turn(reg):
    with acting_as_user(USER):
        await reg.bind_or_verify(thread_id="t1", user_id=USER, session_id="s1", channel="ws")
        return await DiagnosisAgent().process(create_initial_state(USER, "s1", "I have felt low for weeks", thread_id="t1"))

@pytest.mark.asyncio
async def test_mvp_170_session_is_started_assessed_and_signals_kept(wire):
    log, _, reg = wire
    out = await _turn(reg)
    assert [r.url.path for r in log] == ["/api/v1/diagnosis/session/start", "/api/v1/diagnosis/assess"]
    body = json.loads(log[1].content)
    assert body["session_id"] == DSID and body["current_phase"] == "rapport"
    assert_request_matches(ContractFile.model_validate(json.loads((CONTRACTS / "assess.json").read_text(encoding="utf-8"))), log[1])
    assert out["agent_results"][0]["response_content"] is None
    assert "HYPOTHESIS-X" not in json.dumps(out)
    assert out["metadata"]["diagnosis_phase"] == "history"
    assert out["metadata"]["existing_symptoms"][0]["symptom_type"] == "emotional"
    assert out["safety_flags"] == {"risk_level": "HIGH", "categories": ["suicidal_ideation"], "signal_source": "diagnosis"}

def test_mvp_022_diagnosis_contracts_use_the_provider_seed_ids():
    files = {p.stem: ContractFile.model_validate(json.loads(p.read_text(encoding="utf-8"))) for p in CONTRACTS.glob("*.json")}
    assert {"session_start", "assess", "session_end", "assess_user_mismatch", "assess_session_not_found"} <= set(files)
    sid = lambda name: files[name].request.body_example["session_id"]
    assert sid("assess") == "00000000-0000-4000-d000-000000000001" and sid("session_end") == "00000000-0000-4000-d000-000000000002"
    assert (files["assess_user_mismatch"].response.status, sid("assess_user_mismatch")) == (403, "00000000-0000-4000-d000-000000000003")
    assert (files["assess_session_not_found"].response.status, sid("assess_session_not_found")) == (404, "00000000-0000-4000-d000-0000000000ff")
    assert all(f.request.body_example.get("user_id", SEED_USER) == SEED_USER for f in files.values())
    assert {"Authorization", "X-Turn-ID"} <= set(files["assess"].request.headers_required)
    assert {"risk_level", "risk_categories", "degraded"} <= set(files["assess"].response.body_schema["properties"])

@pytest.mark.asyncio
async def test_mvp_170_lost_session_restarts_once_and_a_foreign_session_is_not_retried(wire):
    log, knobs, reg = wire
    await _turn(reg)
    knobs["assess"] = [404, 200]
    log.clear()
    out = await _turn(reg)
    assert [r.url.path for r in log] == ["/api/v1/diagnosis/assess", "/api/v1/diagnosis/session/start", "/api/v1/diagnosis/assess"]
    assert out["metadata"]["diagnosis_phase"] == "history"
    knobs["assess"] = [403]
    log.clear()
    out = await _turn(reg)
    assert [r.url.path for r in log] == ["/api/v1/diagnosis/assess"] and out["agent_results"][0]["success"] is False

@pytest.mark.asyncio
async def test_mvp_170_closure_ends_the_session_and_disconnect_ends_an_open_one(wire):
    log, phase, reg = wire
    phase["value"] = "closure"
    await _turn(reg)
    assert "/api/v1/diagnosis/session/end" in [r.url.path for r in log]
    phase["value"] = "history"
    log.clear()
    await _turn(reg)
    with acting_as_user(USER):
        await end_provider_sessions("t1", USER)
        assert (await reg.get_binding("t1", USER)).diagnosis_session_id is None
    assert [r.url.path for r in log][-1] == "/api/v1/diagnosis/session/end"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_170_diagnosis_lifecycle.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the agent calls `/assess` first with the orchestrator's session id and `current_phase: None` (MVP-170, MVP-022 422); `AssessmentResult.from_dict` reads `current_phase` and raises on `'emotional'`; `response_text` with the hypothesis is shown; `end_provider_sessions` does not exist; `test_mvp_022_diagnosis_contracts_use_the_provider_seed_ids` fails because `tests/contracts/orchestrator__diagnosis/` has no files; a 404 is never followed by a new session.
- [ ] **Step 3: Implement** the lifecycle, schema mapping and the five contract files; `ORCHESTRATOR_DIAGNOSIS_SESSION_MAX_TURNS` (default 12) is read by `DiagnosisAgentSettings`; the differential is dropped from graph state. The risk mapping only raises: `risk_level = max_level("ELEVATED", provider risk_level)` when the provider reports ELEVATED or above or any `CATEGORIES` code, never a lower level than the provider's. Contract examples are synthetic and use only the seed ids above.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.11: Personality provider contract; style comes from the provider
**Covers:** MVP-022, MVP-169
**Severity:** blocker
**Wave:** 2
**Depends on:** WS-11.8
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (default assumed: the orchestrator requests `ensemble`; whether an LLM is used is decided by the personality service's IF-15 egress gate)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/personality_agent.py` (detect with `sources: ["ensemble"]`, lowercase enums, `session_id = str(session_uuid(...))`; style always from `POST /api/v1/personality/style`; delete `_scores_to_style`/`_determine_style_type`; cached style reused when `custom_params.confidence > 0.7`)
- Create: `tests/contracts/orchestrator__personality/detect.json`, `tests/contracts/orchestrator__personality/style.json`
- Test: `services/orchestrator_service/tests/test_mvp_169_personality_contract.py` (new); `services/orchestrator_service/tests/test_agents.py` (drop `_scores_to_style` tests; DTO fixtures lowercase)
**Interfaces:**
- Consumes: IF-11 personality calls; `AgentHttpClient`; WS-10 `session_uuid`; IF-20 `ContractFile`, `assert_request_matches` (WS-05.7).
- Produces: personality state update `{"personality_style": <provider StyleParametersDTO dict incl. custom_params.confidence>, "agent_results": [...]}`; the `orchestrator__personality` consumer directory of WS-05's request #10 (needed by WS-05.10).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_169_personality_contract.py`
```python
"""MVP-169 / MVP-022 (personality): provider schema in, provider style out, no local inverse mapping."""
from __future__ import annotations
import json
import pathlib
import httpx
import pytest
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.agents import configure_agent_runtime
from services.orchestrator_service.src.agents.personality_agent import PersonalityAgent
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
TEXT = "I worry about everything all the time and I get upset really easily when plans change"
CONTRACTS = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__personality"

@pytest.fixture
def wire(monkeypatch):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    log = []
    def handler(req):
        log.append(req)
        if req.url.path.endswith("/detect"):
            return httpx.Response(200, json={"user_id": USER, "ocean_scores": {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.4,
                                             "agreeableness": 0.4, "neuroticism": 0.9}, "assessment_source": "ensemble",
                                             "confidence": 0.7, "evidence": [], "processing_time_ms": 3})
        return httpx.Response(200, json={"user_id": USER, "style_parameters": {"warmth": 0.85, "validation_level": 0.9, "structure": 0.5,
                                         "complexity": 0.5, "directness": 0.4, "energy": 0.4, "style_type": "amiable",
                                         "custom_params": {"confidence": 0.8}}, "recommendations": [], "profile_confidence": 0.8})
    configure_agent_runtime(clients=ServiceClientFactory(transport=httpx.MockTransport(handler)), threads=None)
    return log

@pytest.mark.asyncio
async def test_mvp_022_detect_request_uses_provider_enums(wire):
    await PersonalityAgent().process(create_initial_state(USER, "s1", TEXT))
    body = json.loads(wire[0].content)
    assert body["sources"] == ["ensemble"]
    assert_request_matches(ContractFile.model_validate(json.loads((CONTRACTS / "detect.json").read_text(encoding="utf-8"))), wire[0])

@pytest.mark.asyncio
async def test_mvp_169_style_is_the_providers_not_an_inverse_local_mapping(wire):
    out = await PersonalityAgent().process(create_initial_state(USER, "s1", TEXT))
    assert [r.url.path for r in wire] == ["/api/v1/personality/detect", "/api/v1/personality/style"]
    assert out["personality_style"]["warmth"] == 0.85 and out["personality_style"]["style_type"] == "amiable"
    assert not hasattr(PersonalityAgent, "_scores_to_style")

@pytest.mark.asyncio
async def test_mvp_169_confident_cached_style_skips_the_network(wire):
    state = create_initial_state(USER, "s1", TEXT)
    state["personality_style"] = {"warmth": 0.85, "style_type": "amiable", "custom_params": {"confidence": 0.8}}
    await PersonalityAgent().process(state)
    assert wire == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_169_personality_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `sources == ["TEXT_ANALYSIS", "LLM_ZERO_SHOT"]` (422 at the provider); `AssessmentSource("ensemble")` raises; warmth is computed locally as `(0.4 + (1 - 0.9)) / 2 = 0.25` with no `/style` call (MVP-169); the cached-style shortcut reads a top-level `confidence` that never exists.
- [ ] **Step 3: Implement** the request/response mapping, the `/style` call after detection (and for short messages), the cache rule, and the two contract files from `services/personality_service/src/schemas.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.12: Memory read/write contracts; every user and assistant turn is written
**Covers:** MVP-065
**Severity:** high
**Wave:** 2
**Depends on:** WS-11.8, WS-11.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/memory_node.py` (`MemoryTurnWriter.record_turn(...)`; read path request per contract)
- Modify: `services/orchestrator_service/src/api.py` (REST schedules `record_turn` with FastAPI `BackgroundTasks` after `run_turn`; WS awaits it after sending `chat.response`)
- Modify: `services/orchestrator_service/src/main.py` (`app.state.memory_writer = MemoryTurnWriter(factory.agent_client("memory-service"))`)
- Create: `tests/contracts/orchestrator__memory/context.json`, `tests/contracts/orchestrator__memory/session_message.json`
- Test: `services/orchestrator_service/tests/test_mvp_065_memory_turn_writes.py` (new)
**Interfaces:**
- Consumes: IF-11 `POST /api/v1/memory/context`, `POST /api/v1/memory/session/message` (`{user_id, session_id, role, content, metadata}`); `AgentHttpClient`; WS-10 `session_uuid`; IF-20 `ContractFile`, `assert_request_matches` (WS-05.7).
- Produces: `MemoryTurnWriter(client: AgentHttpClient)` with `async record_turn(*, user_id: str, session_id: str, thread_id: str, turn_id: UUID, user_text: str, assistant_text: str) -> None` (never raises; counts failures in `solace_orchestrator_memory_write_failures_total`); `metadata` holds only `turn_id` and `thread_id`; the `orchestrator__memory` consumer directory of WS-05's request #10 (needed by WS-05.10).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_065_memory_turn_writes.py`
```python
"""MVP-065: each user and assistant turn is written to the memory service, and a write failure is visible."""
from __future__ import annotations
import json
import pathlib
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.events import session_uuid
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult
from services.orchestrator_service.src.langgraph.memory_node import MemoryTurnWriter

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"
CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__memory/session_message.json"

class FakeBuilder:
    async def run_turn(self, **kw):
        return TurnResult(response_text="I'm here with you.", crisis=None, degraded=[], agents=["chat"], turn_id=kw["turn_id"],
                          safety_flags={}, crisis_watch_until=None)

def _client(monkeypatch, status=201):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    log = []
    def handler(req):
        log.append(req)
        return httpx.Response(status, json={"session_id": str(session_uuid("s1")), "storage_time_ms": 1})
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry = FakeBuilder(), InMemoryThreadRegistry()
    app.state.memory_writer = MemoryTurnWriter(cl.AgentHttpClient(service="memory-service", base_url="http://memory-service:8005",
                                                                  timeout_seconds=2.0, transport=httpx.MockTransport(handler)))
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS)
    return TestClient(app), log

def test_mvp_065_rest_turn_writes_user_and_assistant_messages(monkeypatch):
    client, log = _client(monkeypatch)
    r = client.post(f"{BASE}/chat", json={"message": "hello there", "session_id": "s1"})
    assert r.status_code == 200
    bodies = [json.loads(x.content) for x in log]
    assert [(b["role"], b["content"]) for b in bodies] == [("user", "hello there"), ("assistant", "I'm here with you.")]
    assert all(b["session_id"] == str(session_uuid("s1")) and b["user_id"] == USER for b in bodies)
    assert set(bodies[0]["metadata"]) == {"turn_id", "thread_id"}
    assert_request_matches(ContractFile.model_validate(json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))), log[0])

def test_mvp_065_memory_write_failure_is_counted_not_fatal(monkeypatch):
    before = REGISTRY.get_sample_value("solace_orchestrator_memory_write_failures_total") or 0.0
    client, _ = _client(monkeypatch, status=503)
    assert client.post(f"{BASE}/chat", json={"message": "hello there", "session_id": "s1"}).status_code == 200
    assert REGISTRY.get_sample_value("solace_orchestrator_memory_write_failures_total") == before + 2
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_065_memory_turn_writes.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `ImportError: cannot import name 'MemoryTurnWriter'` (nothing ever writes turns to memory, MVP-065), and the contract file is missing.
- [ ] **Step 3: Implement** `MemoryTurnWriter`, the REST/WS scheduling and the two contract files from `services/memory_service/src/schemas.py` (`ContextAssemblyRequest`/`Response`, `AddMessageRequest`/`Response`).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.13: Agent failures surface as degraded reasons and a fallback metric
**Covers:** MVP-076
**Severity:** high
**Wave:** 2
**Depends on:** WS-11.9, WS-11.10, WS-11.11, WS-11.12, WS-10.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/therapy_agent.py`, `diagnosis_agent.py`, `personality_agent.py`, `assessment_agent.py`, `services/orchestrator_service/src/langgraph/memory_node.py` (on provider failure keep the template content, set `metadata.fallback_mode=True`, return `degraded: ["<agent>_unavailable"]` and increment the metric)
- Modify: `services/orchestrator_service/src/infrastructure/clients.py` (`AGENT_FALLBACK_TOTAL = Counter("solace_orchestrator_agent_fallback_total", ..., ["agent", "reason"])`)
- Test: `services/orchestrator_service/tests/test_mvp_076_degraded_reasons.py` (new)
**Interfaces:**
- Consumes: WS-10 `degraded` channel and `TurnResult.agents` rule (fallbacks excluded); `run_turn`.
- Produces: degraded codes `therapy_unavailable`, `diagnosis_unavailable`, `personality_unavailable`, `assessment_unavailable`, `memory_unavailable`; metric `solace_orchestrator_agent_fallback_total{agent, reason}` with `reason` in `unavailable|timeout|bad_response`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_076_degraded_reasons.py`
```python
"""MVP-076: a failing provider is reported as degraded, counted, and never listed as a contributor."""
from __future__ import annotations
from uuid import UUID
import httpx
import pytest
from prometheus_client import REGISTRY
from services.orchestrator_service.src.agents import configure_agent_runtime
from services.orchestrator_service.src.agents.therapy_agent import TherapyAgent
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"

@pytest.fixture
def down(monkeypatch):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    def refuse(req):
        raise httpx.ConnectError("refused", request=req)
    configure_agent_runtime(clients=ServiceClientFactory(transport=httpx.MockTransport(refuse)), threads=InMemoryThreadRegistry())

def _sample():
    return REGISTRY.get_sample_value("solace_orchestrator_agent_fallback_total", {"agent": "therapy", "reason": "unavailable"}) or 0.0

@pytest.mark.asyncio
async def test_mvp_076_therapy_failure_is_degraded_and_counted(down):
    before = _sample()
    state = create_initial_state(USER, "s1", "I need coping strategies", thread_id="t1")
    state["intent"] = "coping_strategy"
    out = await TherapyAgent().process(state)
    assert out["degraded"] == ["therapy_unavailable"]
    assert out["agent_results"][0]["metadata"]["fallback_mode"] is True and out["agent_results"][0]["response_content"]
    assert _sample() == before + 1

@pytest.mark.asyncio
async def test_mvp_076_turn_reports_degraded_and_omits_the_agent(down):
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True), turn_settings=TurnSettings())
    b.compile()
    r = await b.run_turn(user_id=USER, thread_id="t2", session_id="s1", message="What coping strategies help when I feel overwhelmed?",
                         turn_id=UUID("4b6f1c2a-0000-4000-8000-000000000002"), channel="rest")
    assert "therapy_unavailable" in r.degraded and "therapy" not in r.agents
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_076_degraded_reasons.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `KeyError: 'degraded'` (the fallback is a silent `success=True` template, MVP-076), the metric does not exist, and `therapy` appears in `r.agents`.
- [ ] **Step 3: Implement** a shared `_fallback(agent, reason, content)` helper per agent module that returns the template result with `fallback_mode`, the degraded code and the metric increment; map `AgentCallTimeout` to `timeout`, transport errors and 5xx to `unavailable`, schema errors to `bad_response`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.14: /batch is bounded and never echoes exception text
**Covers:** MVP-168, MVP-296
**Severity:** medium
**Wave:** 2
**Depends on:** WS-11.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py` (`/batch`: at most 10 items (`Field(max_length=10)`), `asyncio.Semaphore(2)`, each item through `bind_or_verify` and `run_turn`; a failed item returns the safe reply with `degraded=["internal_error"]` and no exception text; a foreign thread item returns `{"error": {"code": "THREAD_NOT_FOUND", ...}}`)
- Test: `services/orchestrator_service/tests/test_mvp_168_batch_limits.py` (new)
**Interfaces:**
- Consumes: IF-10 `run_turn`; `ThreadRegistry`.
- Produces: IF-12 `/batch` semantics (10 items, concurrency 2, no exception text).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_168_batch_limits.py`
```python
"""MVP-168 / MVP-296: /batch caps size and concurrency and never returns exception text."""
from __future__ import annotations
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"

class FakeBuilder:
    def __init__(self, fail_on=None):
        self.active = self.peak = 0
        self.fail_on = fail_on
    async def run_turn(self, **kw):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        if kw["message"] == self.fail_on:
            raise RuntimeError("SELECT secret FROM internal_table -- password=hunter2")
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)

def _client(b):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry = b, InMemoryThreadRegistry()
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS)
    return TestClient(app)

def test_mvp_168_more_than_ten_items_is_rejected():
    assert _client(FakeBuilder()).post(f"{BASE}/batch", json=[{"message": f"m{i}"} for i in range(11)]).status_code == 422

def test_mvp_168_concurrency_is_at_most_two():
    b = FakeBuilder()
    assert _client(b).post(f"{BASE}/batch", json=[{"message": f"m{i}"} for i in range(10)]).status_code == 200
    assert b.peak == 2

def test_mvp_296_failed_item_has_no_exception_text():
    r = _client(FakeBuilder(fail_on="m1")).post(f"{BASE}/batch", json=[{"message": "m0"}, {"message": "m1"}])
    assert r.status_code == 200 and r.json()[1]["degraded"] == ["internal_error"]
    assert "hunter2" not in r.text and "SELECT" not in r.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_168_batch_limits.py -o "addopts=" -p no:logging -q`
Expected: FAIL: 11 items are accepted (cap 50), all items run at once (`peak == 10`), and the failed item's `metadata.error` carries `str(exception)` (MVP-296).
- [ ] **Step 3: Implement** the bounds and the per-item error handling in `api.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (3 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.15: API-HANDOFF and the client contract match the code (IF-12)
**Covers:** MVP-072
**Severity:** high
**Wave:** 2
**Depends on:** WS-11.4, WS-11.5, WS-11.7, WS-11.12, WS-11.14, WS-05.10, WS-08.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-11 (default assumed: the code is the contract and API-HANDOFF is rewritten from it)
**Real infra:** no
**Files:**
- Modify: `docs/API-HANDOFF.md` (index plus the IF-12 orchestrator contract; the auth section shrinks to a link to WS-08's auth contract document, so the stale Google OAuth "NOT IMPLEMENTED" section and the token-in-redirect-URL text go)
- Create: `docs/api/orchestrator.md` (REST bodies, errors, WS frames, close codes, limits, reconnect guidance)
- Create: `services/orchestrator_service/tests/test_mvp_072_client_contract.py`
- Test: `services/orchestrator_service/tests/test_mvp_072_client_contract.py`
**Interfaces:**
- Consumes: IF-20 synchronous `verify_provider(app, contracts_dir, *, auth_headers) -> list[VerificationResult]` (WS-05.7), called without `await` from a plain sync test (it runs its ASGI calls in a private loop on a worker thread; WS-05 request 13); `assert_request_matches` accepts a raw `json.loads(...)` dict or a `ContractFile`, but this provider test does not need it; WS-05.10's client contracts `tests/contracts/client__orchestrator/` (`chat.post.json`, `chat.foreign-thread.post.json`, `history.get.json`), whose examples use the seed ids requested below. This task creates and edits no file in that directory: WS-05.10 is its only writer, and a missing interaction (for example `/batch`) is a request to WS-05; WS-08's auth contract document, created by WS-08.7, as a link target only (WS-11 never edits it).
- Produces: the provider verification of `client__orchestrator` (WS-05 request #10, needed by WS-05.10's coverage guard, which looks for `verify_provider(` and `client__orchestrator` in this suite); the API-HANDOFF auth rewrite that WS-08 requested as R-13 (needed by WS-08.10), delivered in W2 rather than W3 because this task rewrites the whole document.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_072_client_contract.py`
```python
"""MVP-072: WS-05.10's client contract verifies against the real router, and the docs name the real frames and link auth."""
from __future__ import annotations
import asyncio
import pathlib
import pytest
from fastapi import FastAPI
from solace_common.request_context import acting_as_user
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from solace_testing.contracts import verify_provider
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult

REPO = pathlib.Path(__file__).resolve().parents[3]
CLIENT_CONTRACTS = REPO / "tests/contracts/client__orchestrator"
# seed ids used by WS-05.10's client__orchestrator examples (cross-workstream request to WS-05)
USER, OTHER = "00000000-0000-4000-a000-000000000001", "00000000-0000-4000-a000-000000000002"
OWNED_THREAD, FOREIGN_THREAD = "00000000-0000-4000-b000-000000000001", "00000000-0000-4000-b000-000000000002"
SESSION = "00000000-0000-4000-c000-000000000001"

class FakeBuilder:
    async def run_turn(self, **kw):
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)
    async def get_thread_messages(self, thread_id, *, limit=50):
        return []

async def _seed(registry):
    for thread_id, owner in ((OWNED_THREAD, USER), (FOREIGN_THREAD, OTHER)):
        with acting_as_user(owner):
            await registry.bind_or_verify(thread_id=thread_id, user_id=owner, session_id=SESSION, channel="rest")

@pytest.mark.contract
def test_mvp_072_client_contracts_verify_against_the_router():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/orchestrator")
    registry = InMemoryThreadRegistry()
    asyncio.run(_seed(registry))
    app.state.graph_builder, app.state.thread_registry = FakeBuilder(), registry
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS)
    # WS-05.10 writes these files; this suite only reads them
    assert {"chat.post.json", "chat.foreign-thread.post.json", "history.get.json"} <= {p.name for p in CLIENT_CONTRACTS.glob("*.json")}
    # IF-20: verify_provider is synchronous (never awaited); it raises AssertionError listing every mismatch
    verify_provider(app, CLIENT_CONTRACTS, auth_headers=lambda interaction: {"Authorization": "Bearer t"})

def test_mvp_072_docs_name_the_real_frames_and_link_the_auth_contract():
    doc = (REPO / "docs/api/orchestrator.md").read_text(encoding="utf-8")
    for term in ("chat.message", "safety.alert", "chat.response", "UNSUPPORTED_FRAME", "4401", "Sec-WebSocket-Protocol", "THREAD_NOT_FOUND"):
        assert term in doc, term
    handoff = (REPO / "docs/API-HANDOFF.md").read_text(encoding="utf-8")
    assert "chat.response.final" not in handoff
    assert "NOT IMPLEMENTED" not in handoff and "?access_token=" not in handoff   # WS-08 R-13
    assert "api/auth.md" in handoff and "api/orchestrator.md" in handoff
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_072_client_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `docs/api/orchestrator.md` does not exist (`FileNotFoundError`), and `docs/API-HANDOFF.md` still documents `chat.response.final` (line 130), the Google OAuth "NOT IMPLEMENTED" section (line 39) and a `?access_token=` redirect (line 64), the drift MVP-072 describes. The provider verification is the acceptance check of WS-05.10's contracts against the router of WS-11.4 to WS-11.14 and may already pass; any mismatch it lists is fixed in `api.py`, never by editing WS-05's files.
- [ ] **Step 3: Implement** the two docs from the code as it stands after WS-11.4 to WS-11.14 (synthetic examples only); the auth section of API-HANDOFF is a link to WS-08's document, not a copy (shared-files protocol for `docs/API-HANDOFF.md`). Write no contract example: the `client__orchestrator` interactions are WS-05.10's, so no second `chat.json` or `history.json` is created for the same routes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.16: Every WebSocket frame re-checks the token; re-auth and 4401 close
**Covers:** MVP-163
**Severity:** medium
**Wave:** 3
**Depends on:** WS-11.15
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-27 (default assumed: expiry and revocation checked on every frame; `{type:'auth', token}` replaces the token for the same subject; 4401 on expiry, revocation or deletion); D-22 (default assumed: a refusal never withholds in-band crisis resources, crisis-review F3)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py` (before handling each frame, `validate_access_token_async(current_token)`; failure on a `chat.message` (or legacy `message`) frame first calls `graph_builder.crisis_for_refused_message(user_id=<the socket's validated sub>, session_id, message=content, turn_id=uuid4(), channel="ws")` and, when it returns a `CrisisInfo`, sends `safety_alert_frame(crisis)`; then every failure sends `{type:'error', code:'AUTH_EXPIRED'}` and closes 4401; `auth` frames validate the new token and require the same `sub`, else 4401)
- Test: `services/orchestrator_service/tests/test_mvp_163_ws_token_lifecycle.py` (new)
**Interfaces:**
- Consumes: `_get_jwt_manager().validate_access_token_async` (revocation-aware, WS-07); IF-12 `auth` frame; IF-10 addition `OrchestratorGraphBuilder.crisis_for_refused_message(...) -> CrisisInfo | None` (WS-10.5, crisis-review F3).
- Produces: IF-12 token lifecycle behaviour; a crisis message on a socket whose token just expired still gets `safety.alert` before `AUTH_EXPIRED`, and HIGH/CRITICAL still pages (the user was authenticated on this socket).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_163_ws_token_lifecycle.py`
```python
"""MVP-163: an expired, revoked or swapped token ends the socket instead of being trusted for its lifetime."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import CrisisInfo, TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
OTHER = "9b4e1f5a-0000-4000-8000-000000000009"
BASE = "/api/v1/orchestrator"
CRISIS = CrisisInfo(level="HIGH", categories=["suicidal_ideation"],
                    resources=[{"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"}],
                    resources_version="test-1")

def _ok(sub=USER):
    return MagicMock(success=True, payload={"sub": sub}, user_id=sub)

def _bad():
    return MagicMock(success=False, payload=None, user_id=None, error_code="TOKEN_EXPIRED")

class FakeBuilder:
    def __init__(self):
        self.calls, self.refused = 0, []
    async def run_turn(self, **kw):
        self.calls += 1
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)
    async def crisis_for_refused_message(self, **kw):        # WS-10.5; the real one screens locally and pages HIGH+
        self.refused.append(kw)
        return CRISIS if "kill myself" in kw["message"] else None

def _run(results, frames, *, builder=None, tail_reads=1):
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(side_effect=results)
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry, app.state.active_connections, app.state.replay_buffer = \
        builder or FakeBuilder(), InMemoryThreadRegistry(), {}, None
    got = []
    with patch("solace_security.middleware._get_jwt_manager", return_value=m), pytest.raises(WebSocketDisconnect) as e:
        with TestClient(app).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
            got.append(ws.receive_json())
            for f in frames:
                ws.send_json(f)
                got.append(ws.receive_json())
            for _ in range(tail_reads):
                got.append(ws.receive_json())
    return got, e.value.code

@pytest.mark.timeout(10)
def test_mvp_163_expired_token_ends_the_socket_with_4401():
    got, code = _run([_ok(), _ok(), _bad()], [{"type": "chat.message", "content": "a"}, {"type": "chat.message", "content": "b"}])
    assert got[1]["type"] == "chat.response"
    assert got[2] == {"type": "error", "code": "AUTH_EXPIRED", "message": got[2]["message"]}
    assert code == 4401

@pytest.mark.timeout(10)
def test_mvp_163_reauth_with_another_subject_is_refused():
    _, code = _run([_ok(), _ok(), _ok(OTHER)], [{"type": "chat.message", "content": "a"}, {"type": "auth", "token": "other"}])
    assert code == 4401

@pytest.mark.timeout(10)
def test_crisis_review_f3_expired_token_still_gets_resources_for_a_crisis_frame():
    b = FakeBuilder()
    got, code = _run([_ok(), _bad()], [{"type": "chat.message", "content": "I want to kill myself"}], builder=b, tail_reads=2)
    assert [f["type"] for f in got[1:3]] == ["safety.alert", "error"] and got[2]["code"] == "AUTH_EXPIRED"
    assert got[1]["resources"] == CRISIS.resources and code == 4401
    assert b.calls == 0 and b.refused[0]["user_id"] == USER and b.refused[0]["channel"] == "ws"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_163_ws_token_lifecycle.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the token is validated only at the handshake, so the second frame gets another `chat.response` and the socket stays open until pytest-timeout (MVP-163). The F3 test then fails on its first frame: the crisis message gets a `chat.response` from `run_turn` instead of the refusal, and once per-frame validation exists without the screen it would get only `AUTH_EXPIRED`, with no resources and no page (crisis-review F3).
- [ ] **Step 3: Implement** per-frame validation and the `auth` frame in the handler (`ping` frames are validated too; validation is cheap and blacklist-aware). A failed validation of a chat frame calls `crisis_for_refused_message` first and sends the `safety.alert` frame before the error when it returns a crisis.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.17: WebSocket connection cap, frame size limit and per-user message rate
**Covers:** MVP-158
**Severity:** medium
**Wave:** 3
**Depends on:** WS-11.16
**Not a prerequisite:** the uvicorn `--ws-max-size` image flag (cross-workstream request, defence in depth; no task carries it yet). This task's tests enforce the 8 KB limit in the app and do not need it.
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: limits are Redis-backed in strict environments and in-memory only in development/test); D-22 (default assumed: a refusal never withholds in-band crisis resources, crisis-review F3)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/websocket.py` (`WsLimiter(redis | None, *, max_per_user: int, max_frame_bytes: int, messages_per_minute: int)` with `async register(user_id, connection_id) -> bool`, `async release(user_id, connection_id)`, `async allow_message(user_id) -> bool`, `frame_too_large(raw: str) -> bool`, `async close_user(user_id, code)`; delete the dead `ConnectionManager`, `WebSocketMessage`, `MessageType`)
- Modify: `services/orchestrator_service/src/api.py` (cap at handshake: close 4429; oversized frame: `INVALID_FRAME` error without calling `run_turn`; rate exceeded on a `chat.message` frame: first `graph_builder.crisis_for_refused_message(user_id, session_id, message=content, turn_id=uuid4(), channel="ws")` and, when it returns a crisis, the `safety.alert` frame; then the `RATE_LIMITED` error; a second breach in the window runs the same screen, sends its `safety.alert` when there is one, and closes 4429)
- Modify: `services/orchestrator_service/src/main.py` (`OrchestratorAppSettings`: `ws_max_per_user=3`, `ws_max_frame_bytes=8192`, `ws_messages_per_minute=30`; `app.state.ws_limiter` from `runtime.redis`)
- Test: `services/orchestrator_service/tests/test_mvp_158_ws_limits.py` (new); `services/orchestrator_service/tests/test_infrastructure.py` (remove the dead `TestWebSocketModule` cases)
**Interfaces:**
- Consumes: IF-02 `ORCHESTRATOR_WS_MAX_PER_USER`; IF-01 `runtime.redis`; IF-10 addition `OrchestratorGraphBuilder.crisis_for_refused_message` (WS-10.5, crisis-review F3).
- Produces: `WsLimiter` (also used by WS-11.18 to close a deleted user's sockets); env `ORCHESTRATOR_WS_MAX_FRAME_BYTES`, `ORCHESTRATOR_WS_MESSAGES_PER_MINUTE`; an over-rate crisis message still gets `safety.alert` before `RATE_LIMITED` and HIGH/CRITICAL still pages. The refused screen runs at most twice per socket per window (the second breach closes it), so it cannot be used to flood the outbox.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_158_ws_limits.py`
```python
"""MVP-158: one user cannot open unbounded sockets, send multi-megabyte frames or flood turns."""
from __future__ import annotations
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import CrisisInfo, TurnResult
from services.orchestrator_service.src.websocket import WsLimiter

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"
CRISIS_TEXT = "I want to kill myself"

class FakeBuilder:
    def __init__(self):
        self.calls, self.refused = 0, []
    async def run_turn(self, **kw):
        self.calls += 1
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)
    async def crisis_for_refused_message(self, **kw):        # WS-10.5
        self.refused.append(kw)
        if kw["message"] != CRISIS_TEXT:
            return None
        return CrisisInfo(level="HIGH", categories=["suicidal_ideation"],
                          resources=[{"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"}],
                          resources_version="test-1")

def _app(b, rate=30):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry, app.state.active_connections, app.state.replay_buffer = \
        b, InMemoryThreadRegistry(), {}, None
    app.state.ws_limiter = WsLimiter(None, max_per_user=3, max_frame_bytes=8192, messages_per_minute=rate)
    return TestClient(app)

def _jwt():
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(return_value=MagicMock(success=True, payload={"sub": USER}, user_id=USER))
    return patch("solace_security.middleware._get_jwt_manager", return_value=m)

@pytest.mark.timeout(10)
def test_mvp_158_fourth_socket_for_a_user_is_refused():
    client = _app(FakeBuilder())
    with _jwt(), ExitStack() as stack:
        for i in range(3):
            stack.enter_context(client.websocket_connect(f"{BASE}/ws/s{i}?token=t")).receive_json()
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect(f"{BASE}/ws/s9?token=t") as ws:
                ws.receive_json()
        assert e.value.code == 4429

@pytest.mark.timeout(10)
def test_mvp_158_oversized_frame_is_rejected_without_a_turn():
    b = FakeBuilder()
    with _jwt(), _app(b).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        ws.receive_json()
        ws.send_json({"type": "chat.message", "content": "x" * 9000})
        frame = ws.receive_json()
    assert frame["type"] == "error" and frame["code"] == "INVALID_FRAME" and b.calls == 0

@pytest.mark.timeout(10)
def test_mvp_158_message_rate_is_limited_then_closed():
    with _jwt(), pytest.raises(WebSocketDisconnect) as e:
        with _app(FakeBuilder(), rate=2).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
            ws.receive_json()
            frames = []
            for i in range(4):
                ws.send_json({"type": "chat.message", "content": f"m{i}"})
                frames.append(ws.receive_json())
    assert [f["type"] for f in frames[:2]] == ["chat.response", "chat.response"]
    assert frames[2] == {"type": "error", "code": "RATE_LIMITED", "message": frames[2]["message"]}
    assert e.value.code == 4429

@pytest.mark.timeout(10)
def test_crisis_review_f3_an_over_rate_crisis_message_still_gets_resources_first():
    b = FakeBuilder()
    with _jwt(), pytest.raises(WebSocketDisconnect) as e:
        with _app(b, rate=1).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
            ws.receive_json()
            ws.send_json({"type": "chat.message", "content": "m0"})
            first = ws.receive_json()
            ws.send_json({"type": "chat.message", "content": CRISIS_TEXT})
            alert, limited = ws.receive_json(), ws.receive_json()
            ws.send_json({"type": "chat.message", "content": CRISIS_TEXT})
            second_alert = ws.receive_json()                  # the second breach still screens, then closes
            ws.receive_json()
    assert first["type"] == "chat.response" and alert["type"] == "safety.alert" and limited["code"] == "RATE_LIMITED"
    assert second_alert["type"] == "safety.alert" and e.value.code == 4429
    assert b.calls == 1 and [r["channel"] for r in b.refused] == ["ws", "ws"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_158_ws_limits.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `ImportError: cannot import name 'WsLimiter'`; the live handler has no per-user cap, frame limit or message throttle (the only limits are in the dead `ConnectionManager`, MVP-158). The F3 test pins that a rate limit, once it exists, never withholds resources from a crisis message (crisis-review F3).
- [ ] **Step 3: Implement** `WsLimiter` (Redis `INCR`/`EXPIRE 60` per user per minute and a per-user connection set with a TTL refresh in strict; bounded dicts in development/test; strict without Redis raises `StartupDependencyError` at startup) and the handler checks; the over-rate path calls `crisis_for_refused_message` before it sends `RATE_LIMITED` or closes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.18: user.deleted erases the user's threads and checkpoints
**Covers:** MVP-071
**Severity:** high
**Wave:** 3
**Depends on:** WS-11.17, WS-11.7, WS-10.10, WS-14.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: immediate, irreversible deletion of every thread and checkpoint; nothing pseudonymized here)
**Real infra:** yes
**Files:**
- Modify: `services/orchestrator_service/src/infrastructure/thread_registry.py` (`build_orchestrator_erasure(*, registry, graph_builder) -> UserDataErasure` registering `orchestrator_checkpoints` then `orchestrator_threads`)
- Modify: `services/orchestrator_service/src/main.py` (`register_user_erasure_consumer(runtime.event_plane, service_name="orchestrator-service", erasure=build_orchestrator_erasure(...), purge_caches=...)` where the purge closes the user's sockets with 4401 via `WsLimiter.close_user` and drops the user's replay buffers)
- Test: `services/orchestrator_service/tests/test_mvp_071_erasure_registration.py`, `tests/realinfra/orchestrator-edge-contracts/test_mvp_071_erasure_consumer.py` (new)
**Interfaces:**
- Consumes: IF-16 `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None)`, `UserDataErasure`, `ErasureIncomplete` (WS-14.13, W3 phase A); IF-10 `delete_threads` (WS-10.10); `ThreadRegistry.list_threads`/`delete_user_threads`.
- Produces: erasure stores `orchestrator_checkpoints` and `orchestrator_threads` for the `solace-group-erasure-orchestrator-service` group.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_071_erasure_registration.py`
```python
"""MVP-071: the orchestrator erases a deleted user's threads and checkpoints, checkpoints first."""
from __future__ import annotations
import pytest
from solace_infrastructure.gdpr.erasure import ErasureIncomplete
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry, build_orchestrator_erasure

USER = "8a3f0d4e-0000-4000-8000-000000000001"

class FakeBuilder:
    def __init__(self, fail=False):
        self.deleted, self.fail = [], fail
    async def delete_threads(self, thread_ids):
        if self.fail:
            raise RuntimeError("db down")
        self.deleted.extend(thread_ids)
        return len(thread_ids)

@pytest.mark.asyncio
async def test_mvp_071_erasure_deletes_checkpoints_then_threads():
    reg, b = InMemoryThreadRegistry(), FakeBuilder()
    for tid in ("t1", f"{USER}:s1"):
        await reg.bind_or_verify(thread_id=tid, user_id=USER, session_id="s1", channel="rest")
    erasure = build_orchestrator_erasure(registry=reg, graph_builder=b)
    assert erasure.store_names == ("orchestrator_checkpoints", "orchestrator_threads")
    report = await erasure.erase(USER)
    assert sorted(b.deleted) == sorted(["t1", f"{USER}:s1"]) and report.deleted == {"orchestrator_checkpoints": 2, "orchestrator_threads": 2}
    assert await reg.list_threads(USER) == []

@pytest.mark.asyncio
async def test_mvp_071_failed_checkpoint_delete_is_incomplete_and_retryable():
    reg = InMemoryThreadRegistry()
    await reg.bind_or_verify(thread_id="t1", user_id=USER, session_id="s1", channel="rest")
    with pytest.raises(ErasureIncomplete):
        await build_orchestrator_erasure(registry=reg, graph_builder=FakeBuilder(fail=True)).erase(USER)
```
and in `tests/realinfra/orchestrator-edge-contracts/test_mvp_071_erasure_consumer.py`
```python
"""MVP-071: after erasure no thread row or checkpoint row of the user remains; other users are untouched."""
from __future__ import annotations
from urllib.parse import urlparse
from uuid import uuid4
import asyncpg
import pytest
from pydantic import SecretStr
from solace_common.request_context import acting_as_user
from solace_infrastructure.database.connection_manager import ConnectionPoolManager
from solace_infrastructure.postgres import PostgresSettings
from solace_security.phi_codec import get_phi_codec
from services.orchestrator_service.src.infrastructure.thread_registry import PostgresThreadRegistry, build_orchestrator_erasure
from services.orchestrator_service.src.langgraph.checkpointer import create_checkpointer
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

@pytest.mark.asyncio
async def test_mvp_071_user_threads_and_checkpoints_are_gone(monkeypatch, pg_migrated):
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "11" * 32)  # synthetic test key
    monkeypatch.setenv("ENCRYPTION_KEY_ID", "test-k1")
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", "22" * 16)
    u = urlparse(pg_migrated.dsn_app)
    await ConnectionPoolManager.register_pool("orch_erasure_test", PostgresSettings(host=u.hostname, port=u.port,
        database=u.path.lstrip("/"), user=u.username, password=SecretStr(u.password)))
    reg = PostgresThreadRegistry(pool_name="orch_erasure_test")
    handle = await create_checkpointer(environment="production", conninfo=pg_migrated.dsn_app, codec=get_phi_codec())
    b = OrchestratorGraphBuilder(GraphBuilderSettings(), checkpointer=handle.saver, turn_settings=TurnSettings())
    b.compile()
    a, other = str(uuid4()), str(uuid4())
    threads = {a: [str(uuid4()), f"{a}:s1"], other: [str(uuid4())]}
    for user, tids in threads.items():
        for tid in tids:
            with acting_as_user(user):
                await reg.bind_or_verify(thread_id=tid, user_id=user, session_id="s1", channel="rest")
            await b.run_turn(user_id=user, thread_id=tid, session_id="s1", message="hello there friend", turn_id=uuid4(), channel="rest")
    with acting_as_user(a):
        await build_orchestrator_erasure(registry=reg, graph_builder=b).erase(a)
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        assert await conn.fetchval("SELECT count(*) FROM orchestrator_threads WHERE user_id=$1::uuid", a) == 0
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table} WHERE thread_id = ANY($1)", threads[a]) == 0
        assert await conn.fetchval("SELECT count(*) FROM checkpoints WHERE thread_id = ANY($1)", threads[other]) >= 1
    finally:
        await conn.close()
        await handle.aclose()
        await ConnectionPoolManager.close_pool("orch_erasure_test")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_071_erasure_registration.py -o "addopts=" -p no:logging -q` and `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/orchestrator-edge-contracts/test_mvp_071_erasure_consumer.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'build_orchestrator_erasure'` (no erasure path, no `user.deleted` consumer, MVP-071).
- [ ] **Step 3: Implement** `build_orchestrator_erasure` (the checkpoint deleter lists the user's threads and calls `delete_threads`; the threads deleter calls `delete_user_threads`) and the `main.py` registration; a partial failure surfaces as `ErasureIncomplete`, which the IF-16 helper turns into `RetryableEventError`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then `tests/realinfra/orchestrator-edge-contracts` with `$env:SOLACE_REQUIRE_REALINFRA="1"`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.19: WebSocket reconnect recovery: history, not an in-memory replay buffer
**Covers:** MVP-294
**Severity:** low
**Wave:** 3
**Depends on:** WS-11.18
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none (decision recorded here: replay stays off; the durable history endpoint is the recovery path, because an in-memory buffer cannot survive the restarts that cause most drops)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py` (the `connected` frame adds `last_turn_id` from the registry binding and `recovery: "history"`)
- Modify: `services/orchestrator_service/src/main.py` (in strict environments `enable_ws_replay=True` raises `StartupDependencyError`)
- Modify: `services/orchestrator_service/src/replay_buffer.py` (module docstring: development/test only)
- Modify: `docs/api/orchestrator.md` (reconnect section)
- Test: `services/orchestrator_service/tests/test_mvp_294_ws_recovery.py` (new)
**Interfaces:**
- Consumes: `ThreadBinding.last_turn_id` (WS-11.7); `get_thread_messages` (WS-10.10).
- Produces: `connected` frame `{type:'connected', session_id, thread_id, connection_id, last_turn_id: str | null, recovery: 'history'}`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_294_ws_recovery.py`
```python
"""MVP-294: on reconnect the client learns the last completed turn and recovers it from durable history."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"

class FakeBuilder:
    async def run_turn(self, **kw):
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)

@pytest.mark.timeout(10)
def test_mvp_294_connected_frame_names_the_last_completed_turn():
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry, app.state.active_connections, app.state.replay_buffer = \
        FakeBuilder(), InMemoryThreadRegistry(), {}, None
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(return_value=MagicMock(success=True, payload={"sub": USER}, user_id=USER))
    client = TestClient(app)
    with patch("solace_security.middleware._get_jwt_manager", return_value=m):
        with client.websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
            assert ws.receive_json()["last_turn_id"] is None
            ws.send_json({"type": "chat.message", "content": "hi"})
            turn = ws.receive_json()["turn_id"]
        with client.websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
            hello = ws.receive_json()
    assert hello["last_turn_id"] == turn and hello["recovery"] == "history"

def test_mvp_294_strict_refuses_the_in_memory_replay_buffer(monkeypatch):
    from solace_infrastructure.runtime import StartupDependencyError
    from services.orchestrator_service.src.main import assert_replay_allowed
    with pytest.raises(StartupDependencyError):
        assert_replay_allowed(environment="production", enable_ws_replay=True)
    assert assert_replay_allowed(environment="development", enable_ws_replay=True) is None
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_294_ws_recovery.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `KeyError: 'last_turn_id'` (a reply produced while the socket was down is simply lost from the live stream, MVP-294) and `ImportError` for `assert_replay_allowed`.
- [ ] **Step 3: Implement** the frame fields (from `registry.get_binding(...).last_turn_id`, written by `record_turn`), `assert_replay_allowed(*, environment, enable_ws_replay) -> None` called in the lifespan, and the doc section.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.20: Consent gate at chat ingress (REST and WebSocket)
**Covers:** none (supporting task)
**Supports:** MVP-100 (owned and closed by WS-14)
**Severity:** high
**Wave:** 3
**Depends on:** WS-11.19, WS-07.2, WS-08.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-23 (default assumed: chat requires the `cns.ok` claim; a missing claim is refused in strict environments and allowed with a WARNING in development/test; a refused message is still screened, gets the static crisis resources at ELEVATED or above and is paged at HIGH/CRITICAL with category codes only, under the vital-interest reading of the crisis-handling disclosure, which the D-23 owner confirms by ledger Ruling; crisis-review F3); D-22 (a refusal never withholds in-band crisis resources)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py` (`consent_ok(user: AuthenticatedUser) -> bool`; before refusing, `/chat` calls `graph_builder.crisis_for_refused_message(user_id=current_user.user_id, session_id=body.session_id or a new id, message=body.message, turn_id=uuid4(), channel="rest")` and `/batch` does the same for each item; `/chat` and `/batch` return 403 `{error:{code:'CONSENT_REQUIRED', message, request_id}, crisis: <IF-06 object> | null}` with the highest-level crisis object of the refused messages; the WS handshake sends `{type:'error', code:'CONSENT_REQUIRED'}` then closes 4403, before any client message exists, so nothing is screened there)
- Modify: `docs/API-HANDOFF.md` (the `403 CONSENT_REQUIRED` body carries `crisis`, which clients render exactly as on a 200; written by WS-11.15 in W2)
- Test: `services/orchestrator_service/tests/test_mvp_100_consent_gate.py` (new)
**Interfaces:**
- Consumes: IF-08/IF-17 access-token claim `cns: {ok: bool, version: str}` minted at every issuance by WS-08.13 and surfaced as the field `AuthenticatedUser.cns: dict | None` (WS-07.2, mapped in `from_payload` too, so the WS path sees it); IF-01 `is_strict_environment`; IF-10 addition `OrchestratorGraphBuilder.crisis_for_refused_message` (WS-10.5, crisis-review F3).
- Produces: IF-12 `403 CONSENT_REQUIRED`, whose body carries the IF-06 `crisis` object (null below ELEVATED; an IF-12 addition for the API-HANDOFF, crisis-review F3); WS error code `CONSENT_REQUIRED` with close 4403 (extension of IF-12).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_100_consent_gate.py`
```python
"""Supports MVP-100: no chat without the required consents."""
from __future__ import annotations
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import CrisisInfo, TurnResult

USER = "8a3f0d4e-0000-4000-8000-000000000001"
BASE = "/api/v1/orchestrator"
CRISIS_TEXT = "I want to kill myself"
CRISIS = CrisisInfo(level="HIGH", categories=["suicidal_ideation"],
                    resources=[{"name": "988 Suicide & Crisis Lifeline", "contact": "Call or text 988", "kind": "phone"}],
                    resources_version="test-1")

class FakeBuilder:
    def __init__(self):
        self.calls, self.refused = 0, []
    async def run_turn(self, **kw):
        self.calls += 1
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)
    async def crisis_for_refused_message(self, **kw):        # WS-10.5; the real one screens locally and pages HIGH+
        self.refused.append(kw)
        return CRISIS if kw["message"] == CRISIS_TEXT else None

def _post(cns, b, message="hello"):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry = b, InMemoryThreadRegistry()
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS, cns=cns)
    return TestClient(app).post(f"{BASE}/chat", json={"message": message})

def test_mvp_100_revoked_consent_blocks_chat():
    b = FakeBuilder()
    r = _post({"ok": False, "version": "2026-10"}, b)
    assert r.status_code == 403 and r.json()["error"]["code"] == "CONSENT_REQUIRED" and b.calls == 0
    assert r.json()["crisis"] is None

def test_mvp_100_granted_consent_allows_chat():
    assert _post({"ok": True, "version": "2026-10"}, FakeBuilder()).status_code == 200

@pytest.mark.parametrize("env,status", [("production", 403), ("staging", 403), ("test", 200)])
def test_mvp_100_missing_claim_is_strict_only(monkeypatch, env, status):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("ORCHESTRATOR_ENVIRONMENT", env)
    assert _post(None, FakeBuilder()).status_code == status

def test_crisis_review_f3_refused_consent_still_returns_resources_for_a_crisis_message():
    b = FakeBuilder()
    r = _post({"ok": False, "version": "2026-10"}, b, message=CRISIS_TEXT)       # e.g. after a versioned consent bump
    body = r.json()
    assert r.status_code == 403 and body["error"]["code"] == "CONSENT_REQUIRED" and b.calls == 0
    assert body["crisis"] == CRISIS.to_dict()
    assert (b.refused[0]["user_id"], b.refused[0]["channel"], b.refused[0]["message"]) == (USER, "rest", CRISIS_TEXT)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_100_consent_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL: a revoked consent still gets 200 and a turn runs (consent is recorded but never enforced, MVP-100). The F3 test pins that the refusal, once it exists, still carries the crisis object and pages (crisis-review F3).
- [ ] **Step 3: Implement** `consent_ok` (true when `user.cns` is a dict whose `ok` is `True`; a missing claim (`cns is None`) is refused when `is_strict_environment()`), the REST 403 (screening the refused message or items through `crisis_for_refused_message` first and adding `crisis`) and the WS refusal before any turn.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (existing WS/REST tests run with `ENVIRONMENT=test`, so the missing claim is allowed there)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.20:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-11.21: The token's locale reaches run_turn, every refusal screen, the edge fallback and therapy, with no lookup (D-29)
**Covers:** none (supporting task for D-29)
**Supports:** MVP-079 (closed by WS-10.5; the in-band resources, including the edge fallback, follow the user's locale)
**Severity:** high
**Wave:** 3
**Depends on:** WS-11.20, WS-11.9, WS-10.5, WS-08.21, WS-07.12
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-29 (decided 2026-10-19: several countries, US first. The locale contract is the access-token claim `loc` (WS-08.21, from the profile locale, default `en-US`). The edge carries the raw claim from `AuthenticatedUser.locale` to `run_turn` and `crisis_for_refused_message` with no database or network call; WS-10.5 resolves it with `solace_clinical.resolve_locale`, so a missing claim means en-US and an unsupported one gets the generic fallback. No body field, query parameter or header supplies a locale in the MVP, so IF-12 is unchanged; a later request-supplied locale could only refine through `resolve_locale(claim, requested=...)`, within the supported set and never across countries)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/api.py`. `/chat` and `/batch` pass `locale=current_user.locale` to `run_turn` and to `crisis_for_refused_message`. The WS handler passes the locale of the frame's current principal (from the handshake, replaced by an `auth` re-auth frame, WS-11.16) to `run_turn` and to every refusal screen (WS-11.16, WS-11.17, WS-11.20). The WS-11.4 and WS-11.5 edge fallbacks build their crisis object and `crisis_reply_text` from one `crisis_resource_set(resolve_locale(current_user.locale))`.
- Modify: `services/orchestrator_service/src/agents/therapy_agent.py` (the IF-11 message body adds `"locale": state.get("locale") or DEFAULT_LOCALE`, the resolved locale WS-10.5 wrote to the graph state)
- Modify: `tests/contracts/orchestrator__therapy/message.json` (optional request property `locale` of type string; `"en-US"` in `body_example`)
- Modify: `docs/API-HANDOFF.md` (crisis resources follow the account's profile locale; a locale without populated resources gets the generic fallback with `resources_version` `fallback-1.0.0`; clients render `crisis.resources` exactly as sent, including `kind: "url"`)
- Test: `services/orchestrator_service/tests/test_mvp_079_locale_edge.py` (new)
**Interfaces:**
- Consumes: `AuthenticatedUser.locale: str | None`, the raw `loc` claim, also mapped in `from_payload` so the WS path sees it (WS-07.12, which lands the WS-07 request below); IF-10 `run_turn(..., locale=)`, `crisis_for_refused_message(..., locale=)` and the graph-state key `locale` (WS-10.5); `solace_clinical.crisis_resource_set`, `resolve_locale`, `DEFAULT_LOCALE` (WS-12.2); IF-17 claim `loc` (WS-08.21).
- Produces: the D-29 carrier from token to principal to `run_turn` to graph state to therapy; the IF-11 therapy message field `locale` (consumed by WS-15.23); the API-HANDOFF note. The IF-12 request and response shapes are unchanged.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_079_locale_edge.py`
```python
"""D-29: the access token's locale reaches run_turn, every refusal screen, the edge fallback and therapy, with no lookup."""
from __future__ import annotations
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import NAMESPACE_URL, uuid5
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from solace_clinical import FALLBACK_RESOURCES_VERSION, RESOURCES_VERSION, crisis_resources
from solace_common.request_context import acting_as_user
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.agents import configure_agent_runtime
from services.orchestrator_service.src.agents.therapy_agent import TherapyAgent
from services.orchestrator_service.src.api import router
from services.orchestrator_service.src.infrastructure import clients as cl
from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
from services.orchestrator_service.src.infrastructure.thread_registry import InMemoryThreadRegistry
from services.orchestrator_service.src.langgraph.graph_builder import TurnResult
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"
SID = "7d2b9c1e-0000-4000-8000-000000000003"
PLAN = str(uuid5(NAMESPACE_URL, f"solace:therapy-plan:{USER}"))
BASE = "/api/v1/orchestrator"
OK = {"ok": True, "version": "2026-10"}
CONTRACTS = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__therapy"
REPLY = {"session_id": SID, "user_id": USER, "response_text": "Let's try box breathing.", "current_phase": "working",
         "technique_applied": {"technique_id": "6f1e2d3c-0000-4000-8000-000000000005", "name": "Box breathing",
                               "modality": "mindfulness", "category": "relaxation", "description": "d"},
         "homework_assigned": [], "safety_alerts": [], "next_steps": [], "processing_time_ms": 12}

class FakeBuilder:
    def __init__(self, exc=None):
        self.turns, self.refused, self.exc = [], [], exc
    async def run_turn(self, **kw):
        self.turns.append(kw)
        if self.exc:
            raise self.exc
        return TurnResult(response_text="ok", crisis=None, degraded=[], agents=[], turn_id=kw["turn_id"], safety_flags={},
                          crisis_watch_until=None)
    async def crisis_for_refused_message(self, **kw):
        self.refused.append(kw)
        return None

def _app(b):
    app = FastAPI()
    app.include_router(router, prefix=BASE)
    app.state.graph_builder, app.state.thread_registry = b, InMemoryThreadRegistry()
    app.state.active_connections, app.state.replay_buffer = {}, None
    return app

def _rest(b, *, locale, cns=OK):
    app = _app(b)
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id=USER, token_type=TokenType.ACCESS,
                                                                           cns=cns, locale=locale)
    return TestClient(app)

@pytest.mark.parametrize("claim", [None, "en-US", "fr-FR"])
def test_d29_rest_chat_passes_the_token_locale_to_run_turn(claim):
    b = FakeBuilder()
    assert _rest(b, locale=claim).post(f"{BASE}/chat", json={"message": "hello"}).status_code == 200
    assert b.turns[0]["locale"] == claim

def test_d29_refused_rest_message_is_screened_with_the_token_locale():
    b = FakeBuilder()
    r = _rest(b, locale="fr-FR", cns={"ok": False, "version": "2026-10"}).post(f"{BASE}/chat", json={"message": "hello"})
    assert r.status_code == 403 and b.refused[0]["locale"] == "fr-FR" and b.turns == []

@pytest.mark.parametrize("claim,version", [(None, RESOURCES_VERSION), ("en-US", RESOURCES_VERSION),
                                           ("fr-FR", FALLBACK_RESOURCES_VERSION)])
def test_d29_edge_fallback_uses_the_token_locale(claim, version):
    r = _rest(FakeBuilder(exc=RuntimeError("bug")), locale=claim).post(f"{BASE}/chat",
                                                                        json={"message": "I want to kill myself"})
    crisis = r.json()["crisis"]
    assert crisis["resources"] == crisis_resources(claim) and crisis["resources_version"] == version
    assert ("988" in r.json()["response"]) is (version == RESOURCES_VERSION)

@pytest.mark.timeout(10)
@pytest.mark.parametrize("loc", [None, "de-DE"])
def test_d29_websocket_turn_passes_the_token_locale(loc):
    b = FakeBuilder()
    payload = {"sub": USER, "type": "access", "cns": OK, **({"loc": loc} if loc else {})}
    m = MagicMock()
    m.validate_access_token_async = AsyncMock(return_value=MagicMock(success=True, payload=payload, user_id=USER))
    with patch("solace_security.middleware._get_jwt_manager", return_value=m), \
            TestClient(_app(b)).websocket_connect(f"{BASE}/ws/s1?token=t") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "chat.message", "content": "hello", "client_message_id": "c1"})
        assert ws.receive_json()["type"] == "chat.response"
    assert b.turns[0]["locale"] == loc

@pytest.mark.asyncio
@pytest.mark.parametrize("state_locale", ["en-US", "fr-FR"])
async def test_d29_therapy_message_relays_the_turn_locale(monkeypatch, state_locale):
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cl, "service_auth_headers", headers, raising=False)
    log = []
    def handler(req):
        log.append(req)
        if req.url.path == "/api/v1/therapy/sessions/start":
            return httpx.Response(201, json={"session_id": SID, "user_id": USER, "treatment_plan_id": PLAN, "session_number": 1,
                                             "current_phase": "opening", "initial_message": "Hi", "suggested_agenda": []})
        return httpx.Response(200, json=REPLY)
    reg = InMemoryThreadRegistry()
    configure_agent_runtime(clients=ServiceClientFactory(transport=httpx.MockTransport(handler)), threads=reg)
    state = create_initial_state(USER, "s1", "I need coping strategies", thread_id="t1", locale=state_locale)
    state["intent"] = "coping_strategy"
    with acting_as_user(USER):
        await reg.bind_or_verify(thread_id="t1", user_id=USER, session_id="s1", channel="rest")
        await TherapyAgent().process(state)
    assert json.loads(log[-1].content)["locale"] == state_locale
    assert_request_matches(ContractFile.model_validate(json.loads((CONTRACTS / "message.json").read_text(encoding="utf-8"))),
                           log[-1])
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_079_locale_edge.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The REST and WS tests fail with `KeyError: 'locale'`: the edge never passes a locale to `run_turn` or `crisis_for_refused_message`. The `fr-FR` edge-fallback case gets the en-US set with `RESOURCES_VERSION` and "988". The therapy message body has no `locale`. `AuthenticatedUser(..., locale=...)` exists from WS-07.12 (a dependency); the edge never defaults the locale itself.
- [ ] **Step 3: Implement** the edge plumbing, the fallback switch to `crisis_resource_set(resolve_locale(current_user.locale))`, the therapy body field, the contract property and the API-HANDOFF note. The edge never validates, defaults or looks up the locale itself; `run_turn` and `crisis_for_refused_message` resolve it.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/contracts`. Expected: all pass (`test_mvp_297_rest_chat_contract.py` and `test_mvp_072_ws_protocol.py` still see the en-US resources, because their principals carry no `loc`)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-11.21:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Owner WS | Change | Files | Needed by |
|---|---|---|---|
| WS-03 | Env registry and compose: orchestrator gets `CORS_ALLOWED_ORIGINS`, `REDIS_URL`, `TRUSTED_PROXY_HOPS` (W0); mark retired `ORCHESTRATOR_CORS_ORIGINS` and `PERSISTENCE_*` (W0), `ORCHESTRATOR_{THERAPY,DIAGNOSIS,PERSONALITY,ASSESSMENT}_AGENT_SERVICE_URL` and `ORCHESTRATOR_TRUSTED_PROXY_HOPS` (W1); add `ORCHESTRATOR_DIAGNOSIS_SESSION_MAX_TURNS` (default 12, W2), `ORCHESTRATOR_WS_MAX_FRAME_BYTES` (8192) and `ORCHESTRATOR_WS_MESSAGES_PER_MINUTE` (30) (W3). | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-11.1, WS-11.2, WS-11.3, WS-11.10, WS-11.17 |
| WS-01 | Orchestrator image CMD adds `--ws-max-size 16384` so frames far above the 8 KB app limit are refused by uvicorn before they reach the app (W3). No WS-01 task carries this yet (WS-01.6 owns the uvicorn flags, W0); add it by ledger Ruling. Defence in depth only. | `services/orchestrator_service/Dockerfile` | none (hardens WS-11.17; not a prerequisite) |
| WS-07 | `AuthenticatedUser.cns` carries the access token's `cns` claim (`{ok, version}`) on REST and through `from_payload` for the WS path. Planned as WS-07.2 (W1); WS-11.20 reads the field. | `src/solace_security/middleware.py`, `src/solace_security/auth.py` | WS-11.20 |
| WS-07 | D-29: a field `AuthenticatedUser.locale` (a `str`, or `None`) carries the access token's `loc` claim (IF-17 addition by WS-08.21) unchanged, with no defaulting or validation (`None` when absent; `solace_clinical.resolve_locale` in WS-10.5 owns the policy), on REST and through `from_payload` for the WS path; obo SERVICE tokens yield `None`. Planned as WS-07.12 (W3 phase A), on which WS-11.21 depends; WS-11.21's tests construct `AuthenticatedUser(..., locale=...)`. | `src/solace_security/middleware.py`, `src/solace_security/auth.py` | WS-11.21 |
| WS-05 | D-29 E2E (the W3 suite): (1) a seeded user whose profile locale is set to `fr-FR` through `PUT /api/v1/users/me`, then refreshed, sends a crisis message over REST and over WS; (2) assert `crisis.resources == crisis_resources("fr-FR")` (the generic fallback), `resources_version == FALLBACK_RESOURCES_VERSION` and no "988" in the reply or the `safety.alert` frame; (3) assert exactly one page is delivered, as for en-US, and `solace_crisis_resources_fallback_total` rises on the orchestrator's `/metrics`. WS-05.6's en-US assertions stay unchanged, because `RESOURCES_VERSION` is still the en-US version and seed users default to `en-US`. | `tests/e2e/test_w3_locale_fallback.py`, `src/solace_testing/e2e.py` | WS-11.21 |
| WS-07 | Confirm `service_auth_headers` reads `request_id`, `correlation_id` and `turn_id` from `structlog.contextvars` (the keys the WS handler and `run_turn` bind). | `src/solace_security/service_auth.py` | WS-11.8 |
| WS-05 | IF-20 call shapes as WS-05.7 plans them (confirmed by WS-05's reply, its request 13), now used here: `assert_request_matches(contract, request)` accepts a `ContractFile` or the raw `json.loads(...)` dict, and an `httpx.Request` (WS-11.9 to WS-11.12 may keep wrapping with `ContractFile.model_validate`), and the synchronous `verify_provider(app, contracts_dir, *, auth_headers) -> list[VerificationResult]`, which WS-11.15 calls without `await`. Directory ownership as in WS-05.7's README: WS-11 writes `orchestrator__{therapy,diagnosis,personality,memory}`, WS-10.16 writes `orchestrator__safety`, and WS-05.10 writes `client__orchestrator`. | `src/solace_testing/contracts.py`, `tests/contracts/README.md` | WS-11.9, WS-11.10, WS-11.11, WS-11.12, WS-11.15 |
| WS-05 | WS-05.10's `client__orchestrator` examples use fixed seed ids, so the provider test can seed its registry. Authenticated user `00000000-0000-4000-a000-000000000001`. `chat.post.json` has no `thread_id` (a new thread). `chat.foreign-thread.post.json` uses thread `00000000-0000-4000-b000-000000000002`, owned by `00000000-0000-4000-a000-000000000002`. `history.get.json` uses session `00000000-0000-4000-c000-000000000001` and thread `00000000-0000-4000-b000-000000000001`, owned by the authenticated user. Optional: a `batch.post.json` from IF-12 `/batch` (WS-11.14). | `tests/contracts/client__orchestrator/*.json` | WS-11.15 |
| WS-15 | Provider verification of `tests/contracts/orchestrator__therapy/{session_start,message}.json`; `SESSION_NOT_FOUND` returns 404 (IF-11). | `services/therapy_service/tests/test_mvp_022_contract_orchestrator_therapy.py` | WS-11.9 |
| WS-16 | Provider verification of `tests/contracts/orchestrator__diagnosis/{session_start,assess,session_end,assess_user_mismatch,assess_session_not_found}.json`, seeded with the R6 ids that WS-11.10 writes (planned as WS-16.12). | `services/diagnosis_service/tests/test_mvp_022_contract_orchestrator_diagnosis.py` | WS-11.10 |
| WS-18 | Provider verification of `tests/contracts/orchestrator__personality/{detect,style}.json` (accepts an obo service token, `sources=["ensemble"]`). | `services/personality_service/tests/test_mvp_022_contract_orchestrator_personality.py` | WS-11.11 |
| WS-17 | Provider verification of `tests/contracts/orchestrator__memory/{context,session_message}.json` (accepts an obo token; a client session id arrives as a uuid5 UUID). | `services/memory_service/tests/test_mvp_065_contract_orchestrator_memory.py` | WS-11.12 |

## New interfaces

- D-29 (WS-11.21): the access token's `loc` claim, read from `AuthenticatedUser.locale`, is passed raw as `run_turn(locale=)` and `crisis_for_refused_message(locale=)` and drives the edge fallback's resources; the IF-11 therapy message body gains the optional `locale`. No IF-12 field is added.
- IF-12 additions: informational server frame `{type:'connected', session_id, thread_id, connection_id, last_turn_id, recovery:'history'}`; WS error code `CONSENT_REQUIRED` with close 4403; REST `/batch` per-item error object `{"error": {"code": "THREAD_NOT_FOUND"}}`; the `processing` frame is removed.
- `infrastructure/thread_registry.py`: `ThreadBinding`, `ThreadNotOwned`, `ThreadRegistry` protocol, `PostgresThreadRegistry(pool_name="default")`, `InMemoryThreadRegistry`, `create_thread_registry(runtime)`, `validate_thread_id`, `build_orchestrator_erasure(*, registry, graph_builder)`.
- `infrastructure/clients.py`: `AgentHttpClient`, `AgentCallTimeout`, `AgentTransportSettings` (`ORCHESTRATOR_AGENT_TIMEOUT_SECONDS`), `ServiceClientFactory(transport=None).agent_client(service)`, `assert_peer_endpoints(environment)`, metric `solace_orchestrator_agent_fallback_total{agent,reason}`.
- `agents/__init__.py`: `configure_agent_runtime(*, clients, threads=None)`; `agents/diagnosis_agent.end_provider_sessions(thread_id, user_id)`.
- `langgraph/memory_node.py`: `MemoryTurnWriter.record_turn(...)`; metric `solace_orchestrator_memory_write_failures_total`.
- `websocket.py`: `parse_client_frame`, `safety_alert_frame`, `chat_response_frame`, `error_frame`, `WsLimiter`.
- `main.py`: `SPEC`, `register_orchestrator_readiness(...)`, `assert_replay_allowed(*, environment, enable_ws_replay)`.
- Slot 016 table `orchestrator_threads` (columns in WS-11.6).
- Contract interactions `orchestrator__diagnosis/assess_user_mismatch` (403 `USER_ID_MISMATCH`) and `assess_session_not_found` (404 `SESSION_NOT_FOUND`) with WS-16's seed ids (WS-11.10).
- Env: `ORCHESTRATOR_DIAGNOSIS_SESSION_MAX_TURNS`, `ORCHESTRATOR_WS_MAX_FRAME_BYTES`, `ORCHESTRATOR_WS_MESSAGES_PER_MINUTE`.
- Degraded codes added by WS-11: `therapy_unavailable`, `diagnosis_unavailable`, `personality_unavailable`, `assessment_unavailable`, `memory_unavailable`, `internal_error`.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-016 | blocker | WS-11.1 |
| MVP-020 | blocker | WS-11.2 (safety-agent part delivered by WS-10.4) |
| MVP-021 | blocker | WS-11.2 (safety-agent part delivered by WS-10.4) |
| MVP-022 | blocker | WS-11.9, WS-11.10, WS-11.11 |
| MVP-065 | high | WS-11.12 |
| MVP-070 | high | WS-11.6, WS-11.7 |
| MVP-071 | high | WS-11.18 |
| MVP-072 | high | WS-11.5, WS-11.15 |
| MVP-076 | high | WS-11.13 |
| MVP-078 | high | WS-11.9 |
| MVP-158 | medium | WS-11.17 |
| MVP-159 | medium | WS-11.8 |
| MVP-160 | medium | WS-11.3 |
| MVP-162 | medium | WS-11.8 (turn deadline delivered by WS-10.5) |
| MVP-163 | medium | WS-11.16 |
| MVP-164 | medium | WS-11.1 |
| MVP-168 | medium | WS-11.14 |
| MVP-169 | medium | WS-11.11 |
| MVP-170 | medium | WS-11.10 |
| MVP-293 | low | WS-11.8 |
| MVP-294 | low | WS-11.19 |
| MVP-296 | low | WS-11.14 (the ValueError-handler half is removed by WS-11.1's IF-01 adoption) |
| MVP-297 | low | WS-11.4 |
| MVP-356 | low | WS-11.8 (turn id binding in `run_turn` delivered by WS-10.5) |

All 24 ids are covered; none are left uncovered. WS-11.20 supports MVP-100 (WS-14).

**Review focus** (the skeleton's five review_focus failure modes, numbered 1-5 in skeleton order)

REVIEW_FOCUS: 3=WS-11.6,WS-11.7,WS-11.18; 4=WS-11.12,WS-11.14; 5=WS-11.1,WS-11.2,WS-11.16,WS-11.19,WS-11.20
- 1 and 2 are outside WS-11's code. Crisis detection, ingress fusion and crisis-event publishing are WS-10's (`run_turn`) and WS-12's; alert delivery is WS-13's. WS-11 maps `TurnResult.crisis` to the REST body and to the `safety.alert` frame (WS-11.4, WS-11.5), and those tests pin resources even when the runner fails.
- 3: WS-11.6 pins RLS on `orchestrator_threads` as `solace_app` (foreign insert refused, other user and empty context see no rows); WS-11.7 pins another user's valid thread id as 404 `THREAD_NOT_FOUND` on REST chat and history, including a WS-format `<user_id>:<session>` id, and its realinfra registry test runs as `solace_app`; WS-11.18 pins that the user's threads and checkpoints are gone on real Postgres after `user.deleted`.
- 4: WS-11.12 pins memory-write metadata as only `turn_id` and `thread_id`; WS-11.14 pins no exception text in `/batch` item errors.
- 5: WS-11.1 pins the production limiter on `REDIS_URL`; WS-11.2 pins strict boots refusing unset or localhost peer URLs; WS-11.16 pins a token expiring mid-WebSocket closing 4401; WS-11.19 pins strict refusal of the in-memory replay buffer; WS-11.20 pins a missing consent claim as refused in production and staging.

**Dependencies**
- Every cross-workstream dependency names the producing task: WS-04.4 (IF-01 start/stop), WS-04.6 (W1 bootstrap wiring), WS-01.3 (lazy `infrastructure.api_gateway`), WS-07.1 (`acting_as_user`), WS-07.2 (`AuthenticatedUser.cns`), WS-07.3 (obo `service_auth_headers`), WS-09.4 (`UnifiedLLMClient.readiness`), WS-02.3 (slot-016 scaffold), WS-02.6 (`default` pool), WS-14.11 (slot 012 runs first in the W2 schema lane), WS-14.13 (IF-16), WS-05.2 (IF-19), WS-05.7 (IF-20 API), WS-05.10 (client contracts), WS-08.7 (auth document link target), WS-08.13 (`cns` minting), WS-08.21 (`loc` minting, D-29). WS-11.21's `AuthenticatedUser.locale` has no producing WS-07 task yet, so it stays a cross-workstream request and its Step 2 stops for a ledger Ruling if the field is missing. WS-11.6, the only phase-A task, depends only on earlier-wave or same-wave phase-A work (WS-14.11 runs first in the W2 schema lane). WS-11.15 depends on WS-08.7 and WS-11.20 depends on WS-08.13: both are same-wave phase-B tasks in user-service, with no file overlap, and the controller orders them first. The uvicorn `--ws-max-size` flag has no producing task, so it stays a cross-workstream request and is not a dependency.

**Interface consistency**
- IF-10 is consumed exactly as WS-10's plan produces it: `await OrchestratorGraphBuilder.from_runtime(runtime)` (async), `run_turn(*, user_id, thread_id, session_id, message, turn_id, channel, locale=None)` (D-29's `locale`, passed from WS-11.21 on), `TurnResult` fields, `CrisisInfo.to_dict()`, `get_thread_messages(thread_id, *, limit)`, `delete_threads(thread_ids)`, `session_uuid`, `screen_local`, the `degraded`/`safety_flags`/`active_treatment` state contract and the rule that fallbacks are excluded from `TurnResult.agents`.
- IF-12 shapes match the skeleton (REST body, error body, frames, close codes 4401/4403/4429, 8 KB, 30/min, 3 sockets); additions are listed above.
- IF-11 calls use the provider paths and lowercase enum values; every call carries an obo token and the three correlation headers; a POST is retried only when the provider honours `Idempotency-Key` (none today), never on 4xx.
- IF-01 adoption keeps no try/except around platform components; cross-cutting behaviour after W0 lives in the bootstrap, and `main.py` changes after W0 are WS-11's own composition.
- IF-16/IF-03: erasure runs under `acting_as_user` in the helper; the registry touches the RLS table only through `ConnectionPoolManager.acquire`.
- Shared test files: WS-11 edits only test files whose code under test is WS-11's; new files follow `test_mvp_<id>_<slug>.py`.

**Open risks**
- WS-11 W2 is long and serial (api.py and the agent files are shared by most tasks); it is the orchestrator's long pole behind WS-10.10.
- Provider-side contract verification depends on WS-15/16/17/18 tasks that may lag; until then the contract files prove only the consumer side.
- Ending diagnosis sessions on WS disconnect and every 12 assessments is a product choice; REST-only threads rely on the turn cap and phase `closure`.
- `assert_request_matches` and `verify_provider` signatures are assumed from IF-20; a different landed signature means small test edits under a ledger Ruling.
- Personality `ensemble` sends user text to the personality service's LLM only when that service's IF-15 egress gate allows it (D-04); the orchestrator cannot see that decision.
- The W1 REST/WS API accepts any thread id until WS-11.7 lands in W2 (MVP-070 remains open through W1, as scheduled).
- WS-10 and WS-11 share one service test suite; the controller should not run their orchestrator tasks concurrently.
- WS-11.15's provider verification depends on WS-05.10 using the requested seed ids; with different ids the seeding constants change under a ledger Ruling, and the router is never bent to fit.

## Review responses

- [Error] WS-11.15 edits `docs/api/auth.md`, which WS-08 owns: rejected as stated, fixed in wording. WS-11.15 never edited that file. Its only Files mention was "auth links to `docs/api/auth.md`" in the API-HANDOFF bullet, the link the shared-files protocol requires ("linked, not copied", skeleton shared_files `docs/API-HANDOFF.md`). The Files list now names only `docs/API-HANDOFF.md` and `docs/api/orchestrator.md`, and the auth document appears only as WS-08.7's link target.
- [Error] Wave 2 `docs/api/auth.md` touched by WS-08.7 and WS-11.15: fixed. WS-11.15 does not edit it (above), and it now depends on WS-08.7 so the link target exists before API-HANDOFF points at it.
- [Error] WS-05 request #10 (needed by WS-05.10): consumer contract directories and provider `verify_provider`: fixed. WS-11's consumer directories are WS-11.9 (therapy), WS-11.10 (diagnosis), WS-11.11 (personality) and WS-11.12 (memory), and each Produces line now names WS-05.10. `orchestrator__safety` is written by WS-10.16, because the `/check` consumer is WS-10's `crisis_screen.SafetyCheckClient`. As a provider, WS-11 verifies `client__orchestrator` in WS-11.15, which now consumes WS-05.10's files instead of writing its own.
- [Error] WS-09 CR-7 (needed by WS-09.4): `llm` readiness from `llm_client.readiness` and `LLMUnavailableError` at call sites: fixed. WS-11.3 now registers `llm_client.readiness` (non-critical) when a client exists, with a new test `test_mvp_160_llm_check_is_the_clients_own_readiness` and a dependency on WS-09.4. WS-11 owns no LLM call site (the agents call HTTP providers); the two call sites are WS-10's (WS-10.7, WS-10.11).
- [Error] WS-12 R-12-10 (needed by WS-12.10): write `tests/contracts/orchestrator__safety/check.json`: rejected for WS-11, fulfilled by WS-10.16. The `/check` client is WS-10 code (WS-10 plan, WS-10.4 `SafetyCheckClient`; skeleton WS-10 owns `langgraph/crisis_screen.py`), and IF-20 says the consumer workstream writes the contract. WS-10.16 (W2) writes the file, and its Produces line names WS-12.10.
- [Error] WS-16 R6 (needed by WS-16.12): fixed seed ids and the 403/404 interactions in `tests/contracts/orchestrator__diagnosis/`: fixed in WS-11.10. It writes five files with the exact ids and the `risk_level`/`risk_categories`/`degraded` schema allowance, adds `test_mvp_022_diagnosis_contracts_use_the_provider_seed_ids`, and its Produces line names WS-16.12. WS-16's non-blocking R7 (restart once on 404, no retry on 403, honour the provider's `risk_level`, never lowering it) joined WS-11.10 with `test_mvp_170_lost_session_restarts_once_and_a_foreign_session_is_not_retried`.
- [Warning] WS-11.1 (WS-04:IF-01; WS-01:lazy-api-gateway): fixed, now WS-04.4 and WS-01.3.
- [Warning] WS-11.2 (WS-07:IF-04): fixed, now WS-07.3.
- [Warning] WS-11.3 (WS-04:W1-bootstrap-wiring): fixed, now WS-04.6 (plus WS-09.4 for CR-7).
- [Warning] WS-11.6 (WS-02:slot-016-scaffold; WS-14:slot-012; WS-05:IF-19): fixed, now WS-02.3, WS-14.11 and WS-05.2.
- [Warning] WS-11.7 (WS-02:IF-03-pool): fixed, now WS-02.6 (plus WS-07.1 for `acting_as_user`).
- [Warning] WS-11.9 (WS-05:IF-20): fixed, now WS-05.7.
- [Warning] WS-11.15 (WS-05:IF-20): fixed, now WS-05.10 (and WS-08.7).
- [Warning] WS-11.17 (WS-01:ws-max-size): fixed, dependency removed. No WS-01 task produces the flag (no WS-01 plan line mentions `ws-max-size`), and WS-11.17's tests enforce the limit in the app. It stays a cross-workstream request marked "not a prerequisite".
- [Warning] WS-11.18 (WS-14:IF-16): fixed, now WS-14.13.
- [Warning] WS-11.20 (WS-07:cns-claim; WS-08:IF-17): fixed, now WS-07.2 and WS-08.13.
- [Warning] WS-11.15 edits `tests/contracts/client__orchestrator/chat.json` owned by WS-05: fixed. WS-05.7's README protocol gives `client__orchestrator/` to WS-05, and WS-05.10 creates `chat.post.json`, `chat.foreign-thread.post.json` and `history.get.json`. WS-11.15 no longer creates contract files; it verifies them as the provider. A seed-id request and an optional `batch.post.json` request went to WS-05.
- [Warning] WS-11.15 modifies `docs/api/auth.md`, which does not exist yet: fixed. It is not modified (see the first Error) and is created by WS-08.7, now a dependency.
- [Self-check] WS-11.20 read `AuthenticatedUser.metadata["cns"]`, but WS-07.2 plans a field `AuthenticatedUser.cns`: fixed in the test, Step 3 and the WS-07 request row.
- [Self-check] WS-05.7 plans `assert_request_matches(contract: ContractFile, request)` and a synchronous `verify_provider(...) -> list[...]`. WS-11.9 to WS-11.12 passed raw dicts, and WS-11.15 awaited `verify_provider`. Fixed: the tests wrap with `ContractFile.model_validate`, and WS-11.15's test is synchronous.
- [Self-check] WS-08 R-13 (needed by WS-08.10): API-HANDOFF drops the stale Google OAuth "NOT IMPLEMENTED" section and the token-in-URL redirect text and links WS-08's auth document. Joined WS-11.15 (W2, earlier than the requested W3) with assertions on both strings (today at `docs/API-HANDOFF.md:39` and `:64`).
- [Reconciliation 4] WS-05.7 named this workstream's `services/orchestrator_service/src/infrastructure/clients.py` in its Files line (as an AST input of an alignment test): fixed. The client-module half of that check moved here as WS-05's request 15. WS-11.2, which already modifies `clients.py` in W1, now creates `test_orchestrator_clients_use_versioned_provider_prefixes` in `test_mvp_020_agent_service_urls.py` (a lock: every `/api/...` constant in the module starts with `/api/v1/<provider>/`, true today and kept through WS-11.8 and WS-11.13). Its Produces line names WS-05.10, whose coverage guard checks the lock exists and which now depends on WS-11.2. WS-05.7 cannot be the requester, because WS-11.2 depends on WS-10.3, which depends on WS-05.7.
- [Reconciliation 11] WS-05's reply on IF-20 call shapes (its request 13): aligned. WS-11.15's test calls the synchronous `verify_provider` without `await`, checks that WS-05.10's three `client__orchestrator` files are present, and its Consumes and Step 3 say it creates no contract file (WS-05.10 is the only writer; no second `chat.json` or `history.json`). The request row to WS-05 now states that `assert_request_matches` takes the raw dict or a `ContractFile`; the test file moved to a Create line. WS-05.10 now writes the seed ids this test seeds (WS-11's row to WS-05), so the provider check runs against matching examples.
- crisis-review F3: fixed. Verified at WS-11.20 (403 `CONSENT_REQUIRED` before `run_turn`; its test asserted `b.calls == 0`), WS-11.16 (`AUTH_EXPIRED`, then 4401) and WS-11.17 (`RATE_LIMITED`, then 4429): none screened the refused message, so a crisis message got no resources and no page. Each path now calls WS-10.5's `crisis_for_refused_message` first: the REST 403 carries the IF-06 `crisis` object (API-HANDOFF updated in WS-11.20), the WS paths send `safety.alert` before the error or close, and HIGH/CRITICAL pages. One new test per path. The WS handshake consent refusal is unchanged, because no message exists then. D-23's reading (static resources always, a category-only page at HIGH/CRITICAL under the vital-interest basis) is recorded for the D-23 owner's Ruling. Residual outside the finding, raised for the D-22 sign-off: oversized frames (over 8 KB, WS-11.17) and REST bodies over 4000 characters (422, WS-11.4) are still refused unscreened.
- crisis-review F5: fixed. WS-11.5's Produces now says `connected` is always the first server frame after accept (already an IF-12 addition under New interfaces); WS-05.6 reads it before sending.
- crisis-review F7: fixed in WS-10.5. WS-11.5 is unchanged: it sends `safety.alert` as soon as `run_turn` returns, and `run_turn` is now bounded by one deadline from its entry, which the WS handler reaches right after authentication.
- crisis-review F12: fixed. WS-11.9 mapped therapy's signal to ELEVATED only and ignored WS-15.5's `TherapyResponse.risk_level`. It now maps it as WS-11.10 does for diagnosis (the provider's level, at least ELEVATED when a code is present, never lowered), declares `risk_level`, `risk_categories` and `degraded` as optional in `message.json`, and adds `test_crisis_review_f12_provider_risk_level_is_honoured_and_never_lowered`. WS-11.10 is unchanged: its update carries no layer or confidence by design, and WS-10.5 now supplies the defaults. WS-11.4 is unchanged: its fallback covers a `run_turn` exception, which WS-10.5's exception-safe step 4 no longer raises for a provider-raised level.
- decision 2026-10-19 D-29: new task WS-11.21 (W3 phase B, after WS-11.20) carries the user's locale with no lookup. The access token's `loc` claim, read from `AuthenticatedUser.locale`, goes raw to `run_turn(locale=)` and `crisis_for_refused_message(locale=)` on REST, `/batch`, every WS frame and every refusal path. The WS-11.4 and WS-11.5 edge fallbacks use `crisis_resource_set(resolve_locale(...))`. The therapy agent relays the turn's resolved locale in the IF-11 message body (`message.json` gains the optional `locale`), and API-HANDOFF documents the per-locale resources and the fallback. RED tests: a missing claim and en-US keep today's resources; `fr-FR` reaches `run_turn`, the refusal screen and the fallback; the WS frame carries `de-DE`; therapy receives the locale. New requests: WS-07 (`AuthenticatedUser.locale` from `loc`, no defaulting) and WS-05 (a W3 E2E proving the fallback, one page and the metric for a `fr-FR` user). No env setting is needed. IF-12 is unchanged, because no request-supplied locale is accepted in the MVP. The Waves line, file map, New interfaces and Interface consistency (IF-10) are updated.
