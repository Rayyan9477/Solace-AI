# WS-05: Real-infra harness, contract framework and wave acceptance suites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every wave prove itself on real containers and real images. Testcontainers fixtures, a compose test stack and an expected-failure manifest let each wave's acceptance suite be written red against frozen interfaces and turn green only through the real flows. A contract framework catches producer/consumer drift without containers.

**Architecture:** There are three layers.
- `src/solace_testing/realinfra.py` provides session-scoped testcontainers fixtures (IF-19 component part). They build Postgres migrated by the real `scripts.db.migrate`, with a production-identical `solace_app` pool.
- `src/solace_testing/e2e.py` with `docker-compose.test.yml` builds the real images through `docker compose`. Client traffic goes through the edge Caddy. Internal access is by `docker compose exec`, so no internal port is ever published, in either profile.
- `src/solace_testing/e2e_manifest.py` turns listed tests into strict expected failures at a named hop and makes them due at the barrier of their target wave. Contract checks (IF-20) live in `src/solace_testing/contracts.py` and run in-process.

**Covers:** 2 register entries (blocker 0, high 1, medium 1, low 0).

**Waves:**
- W0:
  - phase A1: the fake LLM and the manifest plugin;
  - phase A2: the component fixtures. They run right after WS-02's migrate entrypoint and roles, and before WS-06.1;
  - phase A3: the stack, the boot smoke, and the crisis E2E written red.
- W1 phase A1: the contract framework and event contracts, the fault suite, the negative-phrase suite, the alerting suite and the deployed phrase-set recall suite (WS-05.22), all written red; the overlay env sync for WS-07 and WS-09 (WS-05.21); and, after WS-09.3, the `tests/integration` LLM-timeout update (WS-05.20).
- W2 phase A: REST contracts and the coverage guard, plus the chat journey written red.
- W3 phase A: the stub OIDC provider, secrets-from-files harness, erasure event contracts, the privacy suite and the consent/OAuth suite.
- W4 phase A: the restore-drill driver and the enterprise suite.
- W5: the release gate and the production-readiness checklist.

Every acceptance-suite task is a phase-A task, written red before phase B, even when its files sit under `tests/`.

**Depends on:**
- WS-01: markers `realinfra`, `e2e`, `contract`, `clinical`; test dependency pins (testcontainers, aiokafka, websockets, jsonschema, bcrypt, pyotp), installed per D-20 into the dedicated venv `D:\tmp\mvp-exec\venv` (by the controller at W0 entry, then re-synced to the dev lock by WS-01.2; never into anaconda); images built from one lock; Makefile `test-realinfra` / `test-e2e PROFILE=` targets.
- WS-02: the real migrate entrypoint, the init-db roles, `SCHEMA_HEAD`, `scripts/db/gen_certs.py` for strict TLS, and the production pool codec.
- WS-03: the base and prod compose files that `docker-compose.test.yml` overlays, the IF-02 env registry, Caddy site addresses from `{$CADDY_API_SITE}`/`{$CADDY_AUTH_SITE}`, the IF-22 secret-file layout (W3), and, for the W5 gate on the AWS staging host (D-25), the stack outputs, permissions and host tools of request 17.
- WS-04: the IF-01 `/live`, `/ready` and `/metrics` semantics used to wait for and assert on the stack, and the observability fragment with Alertmanager (W1).

The suites also consume IF-05 to IF-17 and IF-22 from their owners, as each wave's interfaces freeze.

**Owned paths:**
- `src/solace_testing`
- `tests/e2e`, `tests/contracts`, `tests/realinfra/conftest.py`, `tests/conftest.py`, `tests/alignment`, `tests/system`, `tests/solace_testing`, `tests/scripts`
- `docker-compose.test.yml`, `scripts/seed_demo_data.py` (unchanged by this plan; no register entry targets it), `scripts/test`, `infrastructure/test`, `docs/release`
- `tests/integration` as steward (skeleton shared file, owner WS-05): two supporting edits (WS-05.2, WS-05.20) and the sanctions listed under "Shared-directory stewardship"
- `tests/realinfra/realinfra-harness-e2e/` for harness self-tests

**Spec:** docs/MVP-ISSUE-REGISTER.md (MVP-039, MVP-357) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

### Conventions used by every task

- Unit command for a path P:
  `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest P -o "addopts=" -p no:logging -q`
- Real-infra command: the same, with `$env:SOLACE_REQUIRE_REALINFRA="1";` before the interpreter.
- E2E command, dev profile:
  `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; $env:SOLACE_E2E_PROFILE="dev"; $env:SOLACE_E2E_LANE="a"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest P -o "addopts=" -p no:logging -q`
- E2E command, strict profile: the same with `$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production";`. The W0 boot smoke also runs with `"staging"`.
- `$env:SOLACE_MANIFEST_IGNORE="1"` shows raw results, with the manifest not applied; it is used in every acceptance task's Step 2. `$env:SOLACE_BARRIER_WAVE="<n>"` is set by the controller at the barrier of wave n.
- Identities created by the harness use `new_email(kind)`, which returns `"<kind>-<12 hex>@test.invalid"`, where kind is `user`, `clin`, `reject`, `ops` or `fallback`. Mailpit accepts only `^(user|clin|fallback|ops)-[A-Za-z0-9.-]*@test\.invalid$`, so `reject-*` addresses are refused at SMTP time.
- The ENVIRONMENT values are distinct. The pytest process runs with `ENVIRONMENT=test`. The stack's `ENVIRONMENT` comes from the per-run env file: `development` in dev, `production`/`staging` in strict.
- Tests never call real LLM, SMTP, SMS or OAuth endpoints. Synthetic data only. No test, report or log line prints a generated secret.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `infrastructure/test/fake-llm/app.py`, `Dockerfile` | Create | Stdlib fake LLM with a control API: the OpenAI-compatible route (Portkey, development and test only) and the Bedrock Converse route (D-04) | WS-05.1 |
| `infrastructure/test/README.md` | Create | What the test-only images are and must never be used for | WS-05.1 |
| `src/solace_testing/realinfra.py` | Create | IF-19 component fixtures: `PgHandle`, `KafkaHandle`, `SmtpSink`, `FakeLLMHandle`, `require_docker` | WS-05.2 |
| `tests/realinfra/conftest.py` | Create | Fixture re-export, markers, session loop scope, the Docker requirement | WS-05.2 |
| `tests/integration/test_persistence_roundtrip.py` | Modify | Supporting MVP-270: bare `solace_infrastructure` imports | WS-05.2 |
| `src/solace_testing/e2e_manifest.py` | Create | Manifest loading, `hop()`, `shortcut_allowed()`, strict-xfail hooks | WS-05.3 |
| `tests/conftest.py` | Modify | Loads the manifest hooks for every test under `tests/` | WS-05.3 |
| `tests/e2e/expected_failures.yaml` | Create, then Modify | The E2E manifest: entries and shortcuts | WS-05.3, 05.4, 05.5, 05.6, 05.8, 05.9, 05.22, 05.11, 05.13, 05.15, 05.16, 05.17, 05.18, 05.19 |
| `docker-compose.test.yml` | Create, then Modify | The test overlay | WS-05.4, 05.8, 05.9, 05.12, 05.21 |
| `scripts/test/__init__.py`, `scripts/test/gen_e2e_env.py` | Create, then Modify | Per-run env and secret generation from the IF-02 registry | WS-05.4, WS-05.13 |
| `infrastructure/test/compose/llm-dev.yml` | Create | Dev-only compose fragment: the Bedrock endpoint override to the fake LLM and a synthetic model id (D-04); never in a strict render | WS-05.21 |
| `scripts/test/e2e_stack.ps1` | Create | Operator helper: up, down and logs for a lane, using the fixture's compose command | WS-05.4 |
| `src/solace_testing/e2e.py` | Create, then Modify | `Stack`, `StackConfig`, the start/stop functions and the journey helpers | WS-05.4, 05.8, 05.9, 05.21, 05.11, 05.13, 05.15, 05.16, 05.17, 05.18 |
| `tests/e2e/conftest.py` | Create | The `e2e_stack` session fixture, profile selection, markers | WS-05.4 |
| `tests/e2e/test_mvp_039_stack_selftest.py` | Create | Harness self-test on a live stack | WS-05.4 |
| `tests/scripts/test_mvp_039_gen_e2e_env.py` | Create, then Modify | Generator unit tests | WS-05.4, WS-05.13 |
| `tests/e2e/test_w0_boot_smoke.py` | Create | W0 acceptance | WS-05.5 |
| `tests/system/test_smoke.py`, `tests/system/expected_failures.yaml` | Modify, Create | The in-process smoke on the IF-01 `/live` contract | WS-05.5 |
| `tests/e2e/crisis_phrases_e2e.py`, `tests/e2e/test_crisis_alert_e2e.py` | Create | The crisis E2E (W0 red, W1 green) | WS-05.6 |
| `src/solace_testing/contracts.py`, `src/solace_testing/__init__.py` | Modify | IF-20 framework | WS-05.7 |
| `tests/contracts/**` | Create, then Modify | Contract schema, event contracts, contract-file tests, coverage guard, manifest, `client__orchestrator` and `therapy__memory` REST contracts | WS-05.7, WS-05.10, WS-05.14 |
| `tests/alignment/test_api_contracts.py` | Modify | Retire source inspection of clients that W1 rewrites | WS-05.7 |
| `tests/e2e/test_w1_crisis_faults.py`, `tests/e2e/test_w1_negative_phrases.py` | Create | W1 acceptance | WS-05.8 |
| `tests/e2e/test_w1_crisis_alerting.py` | Create | W1 alerting acceptance | WS-05.9 |
| `tests/e2e/test_w2_chat_journey.py` | Create | W2 acceptance | WS-05.11 |
| `infrastructure/test/stub-oidc/app.py`, `Dockerfile` | Create | The stub Google IdP | WS-05.12 |
| `tests/e2e/test_w3_secrets_injection.py` | Create | W3 secrets acceptance | WS-05.13 |
| `tests/e2e/test_w3_privacy.py` | Create | W3 privacy acceptance | WS-05.15 |
| `tests/e2e/test_w3_consent_oauth.py` | Create | W3 consent and OAuth acceptance | WS-05.16 |
| `scripts/test/restore_drill.py`, `tests/e2e/test_w4_enterprise.py` | Create, then Modify | W4 acceptance and the restore-drill driver | WS-05.17, WS-05.18 |
| `scripts/test/release_gate.py`, `scripts/test/release_gate.ps1`, `tests/e2e/test_release_gate.py`, `docs/release/PRR.md` | Create | W5 release gate on the AWS staging host through SSM (D-25), run from a checkpoint tree's `git archive` (D-17) | WS-05.19 |
| `tests/solace_testing/test_mvp_039_*.py`, `tests/solace_testing/test_mvp_357_contract_framework.py`, `tests/realinfra/realinfra-harness-e2e/test_mvp_039_component_fixtures.py` | Create | Self-tests | WS-05.1, 05.2, 05.3, 05.7, 05.12, 05.21 |
| `tests/integration/test_sprint7_portkey_fallback.py` | Modify | `TestLLMRequestTimeout` follows IF-15 (`LLMUnavailableError`) | WS-05.20 |
| `tests/e2e/phrase_recall/conftest.py`, `tests/e2e/phrase_recall/test_w1_phrase_set_deployed.py` | Create | Deployed-configuration phrase-set recall on its own stack | WS-05.22 |

## Tasks

### Task WS-05.1: Fake LLM test double
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** none
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (decided 2026-10-19: real-user LLM traffic goes to Anthropic models on AWS Bedrock under the AWS BAA. Strict environments require `LLM_PROVIDER=bedrock`, and `portkey` stays allowed only in development and test, with synthetic data. Tests still always use the fake LLM, and no real provider ever receives test traffic. So the fake also serves the bedrock-runtime Converse route `POST /model/{model_id}/converse`. From W1, E2E stacks reach it with `LLM_PROVIDER=bedrock` and `BEDROCK_ENDPOINT_URL=http://fake-llm:8080` in the dev profile (WS-05.21, WS-09's CR-3). Strict refuses the override (WS-09.18), so strict stacks run with the LLM off. The fake ignores SigV4, and no test ever names a real Bedrock model id)
**Real infra:** no
**Files:**
- Create: `infrastructure/test/fake-llm/app.py` (standard library only: `http.server.ThreadingHTTPServer`, `json`, `threading`, `urllib.parse`)
- Create: `infrastructure/test/fake-llm/Dockerfile`. It uses the pinned python:3.12-slim base, the same `FROM` line as WS-01's service Dockerfiles. It does `COPY app.py /app/app.py`, `EXPOSE 8080` and `CMD ["python", "/app/app.py"]`. The build context is `infrastructure/test/fake-llm`.
- Create: `infrastructure/test/README.md` (test-only images; never referenced by `docker-compose.yml` or `docker-compose.prod.yml`)
- Test: `tests/solace_testing/test_mvp_039_fake_llm.py` (new)

**Interfaces:**
- Consumes: the IF-15 fake-LLM contract.
- Produces:
  - `POST /v1/chat/completions` returns an OpenAI `chat.completion` body: `id`, `object`, `created`, `model`, `choices[0].message{role:"assistant",content}`, `finish_reason:"stop"`, `usage`.
  - `POST /__control/scenario {mode, match?, content?}` sets one of these modes:
    - `ok`: content `FAKE_REPLY = "Thank you for telling me. I'm here with you."`
    - `echo`: `"ECHO: " + <last user message>`
    - `error`: HTTP 500 `{"error":{"type":"server_error","message":"fake-llm forced error"}}`
    - `timeout`: sleeps `FAKE_LLM_TIMEOUT_SECONDS` (default 65), then HTTP 504
    - `crisis_intent`: content is the JSON text `{"intent":"crisis_disclosure","is_crisis":true,"crisis_level":"HIGH","risk_level":"HIGH","confidence":0.95,"categories":["suicidal_ideation"]}`
    - `reply` (new): content is the given `content` string

    `match` is a case-insensitive substring of the last user message that scopes the scenario; other requests get `ok`.
  - `POST /model/{model_id}/converse` (new, D-04) is the bedrock-runtime Converse route. `model_id` is one URL-encoded path segment, unquoted before use, because inference-profile ids contain `:` and ARNs contain `/`. The fake accepts any id. The last user message is the concatenated `text` blocks of the last `messages[]` entry with `role` `user`. The same scenario state and modes as the OpenAI route apply:
    - The reply is the Converse body `{"output":{"message":{"role":"assistant","content":[{"text":<content>}]}},"stopReason":"end_turn","usage":{"inputTokens":<int>,"outputTokens":<int>,"totalTokens":<int>},"metrics":{"latencyMs":<int>}}`.
    - `error` returns HTTP 500 with header `x-amzn-ErrorType: InternalServerException` and body `{"message":"fake-llm forced error"}`.
    - `timeout` sleeps `FAKE_LLM_TIMEOUT_SECONDS`, then returns HTTP 504.
    - The `Authorization` and `X-Amz-*` headers are ignored (no SigV4 check), and never recorded.
  - `POST /__control/reset`.
  - `GET /__control/requests` returns `[{ts, path, model, messages}]` for both routes; for Converse, `model` is the unquoted `model_id` and `messages` keep the Converse shape.
  - `GET /health` returns `{"status":"ok"}`.
  - The port comes from `FAKE_LLM_PORT` (default 8080).
  - Module API: `make_server(host, port, *, timeout_seconds=None) -> ThreadingHTTPServer`, used in-process by tests.

- [ ] **Step 1: Write the failing test** in `tests/solace_testing/test_mvp_039_fake_llm.py`:
```python
APP = Path(__file__).resolve().parents[2] / "infrastructure" / "test" / "fake-llm" / "app.py"

@pytest.fixture
def fake():
    spec = importlib.util.spec_from_file_location("fake_llm_app", APP); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    server = mod.make_server("127.0.0.1", 0, timeout_seconds=0.5)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield mod, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()

def _chat(base, text):
    return httpx.post(f"{base}/v1/chat/completions", json={"model": "claude-test", "messages": [{"role": "user", "content": text}]}, timeout=5)

def test_mvp_039_ok_mode_returns_openai_shape(fake):
    mod, base = fake; body = _chat(base, "hello").json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"] == {"role": "assistant", "content": mod.FAKE_REPLY}

def test_mvp_039_echo_error_and_timeout_modes(fake):
    _, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "echo"}); assert _chat(base, "abc").json()["choices"][0]["message"]["content"] == "ECHO: abc"
    httpx.post(f"{base}/__control/scenario", json={"mode": "error"}); assert _chat(base, "abc").status_code == 500
    httpx.post(f"{base}/__control/scenario", json={"mode": "timeout"}); t0 = time.monotonic(); r = _chat(base, "abc")
    assert r.status_code == 504 and time.monotonic() - t0 >= 0.5

def test_mvp_039_crisis_intent_is_scoped_by_match(fake):
    mod, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "crisis_intent", "match": "bridge downtown"})
    hit = json.loads(_chat(base, "I keep thinking about the Bridge Downtown").json()["choices"][0]["message"]["content"])
    assert hit["intent"] == "crisis_disclosure" and hit["categories"] == ["suicidal_ideation"]
    assert _chat(base, "nice weather").json()["choices"][0]["message"]["content"] == mod.FAKE_REPLY

def test_mvp_039_reply_mode_and_request_recording(fake):
    _, base = fake
    httpx.post(f"{base}/__control/scenario", json={"mode": "reply", "content": "{\"risk_level\": \"NONE\"}"})
    assert _chat(base, "x").json()["choices"][0]["message"]["content"] == "{\"risk_level\": \"NONE\"}"
    assert [r["messages"][-1]["content"] for r in httpx.get(f"{base}/__control/requests").json()] == ["x"]
    httpx.post(f"{base}/__control/reset"); assert httpx.get(f"{base}/__control/requests").json() == []

def test_d04_bedrock_converse_route_answers_the_real_botocore_client(fake):
    # D-04: the client WS-09 uses (bedrock-runtime Converse through botocore) parses the fake's bodies and errors.
    # Unsigned and pointed at the fake, so no credentials are needed and no AWS endpoint is ever called.
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError
    mod, base = fake
    client = boto3.client("bedrock-runtime", region_name="us-east-1", endpoint_url=base,
                          config=Config(signature_version=UNSIGNED, retries={"total_max_attempts": 1, "mode": "standard"}))
    msgs = [{"role": "user", "content": [{"text": "hello"}]}]
    out = client.converse(modelId="fake-e2e-model:0", messages=msgs)          # synthetic id; ':' exercises path encoding
    assert out["output"]["message"] == {"role": "assistant", "content": [{"text": mod.FAKE_REPLY}]} and out["stopReason"] == "end_turn"
    httpx.post(f"{base}/__control/scenario", json={"mode": "echo"})
    assert client.converse(modelId="fake-e2e-model:0", messages=msgs)["output"]["message"]["content"][0]["text"] == "ECHO: hello"
    httpx.post(f"{base}/__control/scenario", json={"mode": "error"})
    with pytest.raises(ClientError) as err:
        client.converse(modelId="fake-e2e-model:0", messages=msgs)
    assert err.value.response["Error"]["Code"] == "InternalServerException"
    first = httpx.get(f"{base}/__control/requests").json()[0]
    assert first["path"].startswith("/model/") and first["model"] == "fake-e2e-model:0"
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/solace_testing/test_mvp_039_fake_llm.py`. Expected: FAIL with `FileNotFoundError` for `infrastructure/test/fake-llm/app.py`, in every test including `test_d04_bedrock_converse_route_answers_the_real_botocore_client`. No LLM test double exists, so no stack can run without a real provider (MVP-039), and nothing stands in for Bedrock (D-04).
- [ ] **Step 3: Implement** `make_server(host, port, *, timeout_seconds=None)` and a module-level `main()` reading `FAKE_LLM_PORT`/`FAKE_LLM_TIMEOUT_SECONDS`. The scenario state is guarded by a `threading.Lock`. The timeout sleep runs on the request thread only. Both routes share one scenario and one request log. The Converse route matches the raw path against `^/model/([^/]+)/converse$` and applies `urllib.parse.unquote` to the id. The test imports `boto3` from the anaconda env, which the venv sees, and from WS-01.2's runtime lock once it lands; the fake itself stays stdlib-only.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run the unit command with `tests/solace_testing`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.2: Real-infra component fixtures (IF-19)
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** WS-05.1, WS-02.3, WS-02.4
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-02 (default assumed: one database `solace`; `solace_migrator` migrates and `solace_app` is NOSUPERUSER NOBYPASSRLS). D-20 (decided 2026-10-19: testcontainers and aiokafka at their lock versions are installed into the dedicated venv `D:\tmp\mvp-exec\venv`. The controller installs whatever is missing at W0 entry, and WS-01.2 re-syncs the venv to the dev lock. Nothing is installed into anaconda. These fixtures must also run in the lock-only venv `D:\tmp\mvp-exec\venv-clean` that the W0 barrier uses).
**Real infra:** yes
**Files:**
- Create: `src/solace_testing/realinfra.py`
- Create: `tests/realinfra/conftest.py`
- Modify: `tests/integration/test_persistence_roundtrip.py`. Lines 364-397: `from src.solace_infrastructure...` becomes `from solace_infrastructure...`. This is a supporting edit for MVP-270 (WS-01's import-root guard); there is no behaviour change.
- Test: `tests/realinfra/realinfra-harness-e2e/test_mvp_039_component_fixtures.py` (new)

**Interfaces:**
- Consumes:
  - IF-03 `python -m scripts.db.migrate upgrade <rev>` with `DATABASE_URL` for `solace_migrator` (created by WS-02.4);
  - init-db role creation from `POSTGRES_MIGRATOR_PASSWORD`/`POSTGRES_APP_PASSWORD` (request 2; WS-02.1, which WS-02.4 follows);
  - `solace_infrastructure.database.schema_check.SCHEMA_HEAD` (created by WS-02.3; the module does not exist in today's tree);
  - `ConnectionPoolManager.register_pool/get_pool` with `PostgresSettings`;
  - the images declared in `docker-compose.yml` (postgres, redis, kafka) and `docker-compose.test.yml` (mailpit).
- Produces (IF-19 names exactly, plus the additions marked new; satisfies WS-06's request 2 for WS-06.1 and WS-06.4):
```python
IMAGE_REFS: dict[str, str]                      # new: read from the compose files, so tests use the deployed image references
INIT_DB_ENV = {"superuser": "POSTGRES_PASSWORD", "migrator": "POSTGRES_MIGRATOR_PASSWORD", "app": "POSTGRES_APP_PASSWORD"}
@dataclass class PgHandle:                      # __repr__ masks every DSN password
    dsn_app: str; dsn_migrator: str; dsn_superuser: str; app_pool: asyncpg.Pool   # app_pool = ConnectionPoolManager 'default' as solace_app
    async def superuser_fetch(self, sql: str, *args) -> list[asyncpg.Record]      # new
    async def create_database(self, name: str, *, migrate_to: str = "head") -> "PgHandle"   # new: private database, own pool, same codec
@dataclass class KafkaHandle:
    bootstrap: str
    async def produce(self, topic: str, value: "BaseEvent | dict | bytes", *, key: str | None = None) -> None
    async def consume_until(self, topic: str, predicate: Callable[[dict], bool], timeout: float = 30.0, *, group_id: str | None = None) -> list[dict]
    async def committed_offset(self, group_id: str, topic: str, partition: int = 0) -> int | None   # new
    def pause(self) -> None; def unpause(self) -> None                                              # new (docker pause/unpause)
@dataclass class MailMessage: id: str; to: list[str]; subject: str; text: str; headers: dict[str, list[str]]
class SmtpSink:                                  # IF-19 SmtpSink(api_url).wait_for(to, subject_contains, timeout=60)
    def __init__(self, api_url: str, smtp_host: str = "", smtp_port: int = 0) -> None
    def wait_for(self, to: str, subject_contains: str = "", timeout: float = 60.0) -> MailMessage
    def messages(self, to: str | None = None, subject_contains: str = "") -> list[MailMessage]; def clear(self) -> None
@dataclass class FakeLLMHandle:                  # fake_llm fixture value; .base_url is the IF-19 base_url
    base_url: str; control_url: str
    def scenario(self, mode: str, match: str | None = None, content: str | None = None) -> None; def requests(self) -> list[dict]; def reset(self) -> None
def require_docker() -> None                     # skip when Docker is unreachable; pytest.fail when SOLACE_REQUIRE_REALINFRA=1
# session fixtures: pg_migrated, redis_url, kafka_bootstrap, smtp_sink, fake_llm
```
  - `pg_migrated`:
    1. starts the postgres image with `infrastructure/init-db` mounted read-only at `/docker-entrypoint-initdb.d`, plus `POSTGRES_USER=postgres`, `POSTGRES_DB=solace` and per-run passwords for the three `INIT_DB_ENV` names;
    2. runs the migrate CLI as a subprocess with `ENVIRONMENT=test`, `cwd` set to the repo root;
    3. registers ConnectionPoolManager `'default'` as `solace_app` (`ssl_mode="disable"`, pool 1-4), replacing any stale registration.
  - `redis_url` is `redis://:<pw>@127.0.0.1:<port>/0`, on a server started with `--requirepass`.
  - `kafka_bootstrap` runs the compose kafka image through `testcontainers.kafka.KafkaContainer`.
  - `smtp_sink` runs mailpit with SMTP 1025 and API 8025.
  - `fake_llm` builds `infrastructure/test/fake-llm`.
  - These capabilities satisfy WS-06's request 2 (needed by WS-06.1 and WS-06.4): `PgHandle.superuser_fetch(sql, *args)`, `PgHandle.create_database(name, *, migrate_to)`, `app_pool` built with the production JSONB codec, and `KafkaHandle.produce(topic, BaseEvent | dict | bytes, *, key=None)`, `committed_offset(group_id, topic, partition=0)`, `consume_until(topic, predicate, timeout)`, `pause()`/`unpause()`.
  - `tests/realinfra/conftest.py`:
    - re-exports the five fixtures;
    - marks every item under `tests/realinfra` `realinfra` and `integration`;
    - adds `pytest.mark.asyncio(loop_scope="session")` to coroutine tests (pytest-asyncio 1.3);
    - has an autouse session fixture that calls `require_docker()`.
  - Component tests of every workstream live in `tests/realinfra/<ws-slug>/`.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/realinfra-harness-e2e/test_mvp_039_component_fixtures.py`:
```python
async def test_mvp_039_postgres_is_migrated_to_head_and_app_role_is_constrained(pg_migrated):
    from solace_infrastructure.database.schema_check import SCHEMA_HEAD
    (v,) = await pg_migrated.superuser_fetch("SELECT version_num FROM alembic_version")
    assert v["version_num"] == SCHEMA_HEAD
    async with pg_migrated.app_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT current_user AS u, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    assert (row["u"], row["rolsuper"], row["rolbypassrls"]) == ("solace_app", False, False)
    assert "solace_app" not in repr(pg_migrated) or "***" in repr(pg_migrated)

async def test_mvp_039_app_pool_binds_jsonb_as_objects(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        assert await conn.fetchval("SELECT jsonb_typeof($1::jsonb)", {"a": 1}) == "object"

async def test_mvp_039_create_database_stops_at_the_requested_revision(pg_migrated):
    db = await pg_migrated.create_database(f"h039_{uuid4().hex[:8]}", migrate_to="009_platform_roles_rls")
    (v,) = await db.superuser_fetch("SELECT version_num FROM alembic_version")
    assert v["version_num"] == "009_platform_roles_rls"

async def test_mvp_039_redis_requires_the_password(redis_url):
    assert await redis.asyncio.from_url(redis_url).ping() is True
    with pytest.raises(redis.exceptions.AuthenticationError):
        await redis.asyncio.from_url(redis_url.replace(f":{urlsplit(redis_url).password}@", "@")).ping()

async def test_mvp_039_kafka_round_trip_and_committed_offset(kafka_bootstrap):
    topic, ev = f"test.h039.{uuid4().hex[:8]}", SessionStartedEvent(user_id=uuid4(), session_number=1)
    await kafka_bootstrap.produce(topic, ev)
    got = await kafka_bootstrap.consume_until(topic, lambda m: m["metadata"]["event_id"] == str(ev.metadata.event_id), timeout=30, group_id="g-h039")
    assert len(got) == 1 and await kafka_bootstrap.committed_offset("g-h039", topic) == 1

def test_mvp_039_smtp_sink_captures_mail(smtp_sink):
    with smtplib.SMTP(smtp_sink.smtp_host, smtp_sink.smtp_port) as s:
        s.sendmail("noreply@test.invalid", ["clin-h039@test.invalid"], "Subject: [Solace CRISIS HIGH] check\r\n\r\nbody")
    assert smtp_sink.wait_for("clin-h039@test.invalid", "[Solace CRISIS", timeout=10).subject == "[Solace CRISIS HIGH] check"

def test_mvp_039_fake_llm_container_answers(fake_llm):
    r = httpx.post(f"{fake_llm.base_url}/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["object"] == "chat.completion"

def test_mvp_039_missing_docker_fails_when_required(monkeypatch):
    monkeypatch.setenv("SOLACE_REQUIRE_REALINFRA", "1"); monkeypatch.setattr(realinfra, "_docker_ping", lambda: False)
    with pytest.raises(pytest.fail.Exception):
        realinfra.require_docker()
```
- [ ] **Step 2: Run it and confirm it fails.** Run: real-infra command with `tests/realinfra/realinfra-harness-e2e/test_mvp_039_component_fixtures.py`. Expected: FAIL at collection with `ModuleNotFoundError: No module named 'solace_testing.realinfra'`. Nothing in the repo runs against real Postgres, Redis or Kafka; `FixtureConfig.use_testcontainers` has no implementation (MVP-039).
- [ ] **Step 3: Implement** the module and conftest per the Interfaces. Notes:
  - Generated passwords are `secrets.token_urlsafe(24)` and are never printed or logged.
  - DSNs are logged masked.
  - Containers stop at session end.
  - `create_database` uses `CREATE DATABASE` as the superuser, then runs the migrate CLI with `DATABASE_URL` for that database, then builds an `asyncpg` pool using `ConnectionPoolManager._init_connection` as `init`, or the public codec helper once WS-02 exposes one.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/solace_testing`;
  - unit command with `tests/integration`. `TestSchemaRegistryIntegrity` stays green with the bare imports.

  Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.3: Expected-failure manifest plugin with hop tracking
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `src/solace_testing/e2e_manifest.py`
- Create: `tests/e2e/expected_failures.yaml` (`version: 1`, `entries: []`, `shortcuts: []`)
- Modify: `tests/conftest.py`. Import `pytest_configure`, `pytest_collection_modifyitems` and `pytest_runtest_makereport` from `solace_testing.e2e_manifest` (`# noqa: F401`), so every test under `tests/` honours manifests.
- Test: `tests/solace_testing/test_mvp_039_manifest_plugin.py` (new)

**Interfaces:**
- Consumes: none.
- Produces:
  - **Manifest format.** One `expected_failures.yaml` per test directory, owned by that directory's owner:
    ```
    version: 1
    entries:
      - {test_id, expected_failing_hop, target_wave, reason, profiles?: [dev|strict]}
    shortcuts:
      - {name, remove_by_wave, reason}
    ```
    `test_id` is a pytest nodeid relative to the repo root, or an `fnmatch` pattern for parametrized ids.
  - **API:**
    ```python
    class HopFailure(AssertionError)            # .hop, cause chained
    @contextmanager def hop(name: str)          # innermost hop wins
    def shortcut_allowed(name: str, *, manifest_dir: Path | None = None) -> bool
    @dataclass(frozen=True) class ManifestEntry
    @dataclass(frozen=True) class Verdict       # kind: "xfail" | "fail" | "pass"; message: str
    def load_manifest(path: Path) -> tuple[list[ManifestEntry], list[Shortcut]]   # schema-validated
    def evaluate(outcome: str, failed_hop: str | None, entry: ManifestEntry) -> Verdict
    ```
  - **Semantics for an active entry.** An entry is active when `SOLACE_MANIFEST_IGNORE` is unset, it is not due (`SOLACE_BARRIER_WAVE` unset or `< target_wave`), and its profile matches `SOLACE_E2E_PROFILE` (default `dev`).
    - Failing at the expected hop gives xfail.
    - Failing at another hop, or with no hop, gives a fail: `manifest expects hop '<x>', failed at '<y>'`.
    - Passing gives a fail: `XPASS(strict): <id> passed before wave <n>; remove its manifest entry`.
    - Setup-phase failures are judged the same way, which covers fixtures wrapped in `hop(...)`.
    - An entry whose file was collected but whose pattern matched nothing fails the session: `stale manifest entry`.
    - Due entries are reported in the terminal summary.
  - **Shortcuts.** A shortcut is allowed only while listed and `SOLACE_BARRIER_WAVE` is unset or `< remove_by_wave`.

- [ ] **Step 1: Write the failing test** in `tests/solace_testing/test_mvp_039_manifest_plugin.py`:
```python
def test_mvp_039_evaluate_matrix():
    from solace_testing.e2e_manifest import ManifestEntry, evaluate
    e = ManifestEntry(test_id="x", expected_failing_hop="rest_crisis_object", target_wave=1, reason="r")
    assert evaluate("failed", "rest_crisis_object", e).kind == "xfail"
    other = evaluate("failed", "provision_clinician", e)
    assert other.kind == "fail" and "manifest expects hop 'rest_crisis_object'" in other.message
    assert evaluate("failed", None, e).kind == "fail"
    xpass = evaluate("passed", None, e)
    assert xpass.kind == "fail" and "XPASS(strict)" in xpass.message

def test_mvp_039_hop_names_the_failing_step():
    from solace_testing.e2e_manifest import HopFailure, hop
    with pytest.raises(HopFailure) as exc:
        with hop("outer"):
            with hop("register"):
                assert 1 == 2
    assert exc.value.hop == "register"

SAMPLE = '''
from solace_testing.e2e_manifest import hop
def test_expected():
    with hop("a"): assert False
def test_other_hop():
    with hop("b"): assert False
def test_passes_early():
    assert True
def test_unlisted():
    assert True
'''
MANIFEST = '''version: 1
entries:
  - {test_id: "test_sample.py::test_expected", expected_failing_hop: a, target_wave: 1, reason: r}
  - {test_id: "test_sample.py::test_other_hop", expected_failing_hop: a, target_wave: 1, reason: r}
  - {test_id: "test_sample.py::test_passes_early", expected_failing_hop: a, target_wave: 1, reason: r}
shortcuts:
  - {name: verify_email_via_sql, remove_by_wave: 1, reason: r}
'''

def _run(tmp_path, **env):
    (tmp_path / "test_sample.py").write_text(SAMPLE); (tmp_path / "expected_failures.yaml").write_text(MANIFEST)
    base = {k: v for k, v in os.environ.items() if k not in ("SOLACE_BARRIER_WAVE", "SOLACE_MANIFEST_IGNORE")}
    return subprocess.run([sys.executable, "-m", "pytest", "test_sample.py", "-p", "solace_testing.e2e_manifest",
                           "-o", "addopts=", "-p", "no:cacheprovider", "-q"], cwd=tmp_path, capture_output=True, text=True, env={**base, **env})

def test_mvp_039_plugin_applies_strict_hop_xfail(tmp_path):
    out = _run(tmp_path).stdout
    assert "1 xfailed" in out and "2 failed" in out and "1 passed" in out
    assert "manifest expects hop 'a', failed at 'b'" in out and "XPASS(strict)" in out

def test_mvp_039_barrier_wave_makes_entries_due(tmp_path):
    out = _run(tmp_path, SOLACE_BARRIER_WAVE="1").stdout
    assert "2 failed" in out and "2 passed" in out and "xfailed" not in out

def test_mvp_039_ignore_shows_raw_results(tmp_path):
    out = _run(tmp_path, SOLACE_MANIFEST_IGNORE="1").stdout
    assert "2 failed" in out and "2 passed" in out

def test_mvp_039_stale_entry_fails_the_session(tmp_path):
    (tmp_path / "test_sample.py").write_text(SAMPLE)
    (tmp_path / "expected_failures.yaml").write_text(MANIFEST.replace("test_passes_early", "test_renamed"))
    # run as in _run without rewriting the manifest
    ...  # assert "stale manifest entry" in stdout and returncode != 0

def test_mvp_039_shortcut_expires_at_its_wave(tmp_path, monkeypatch):
    from solace_testing.e2e_manifest import shortcut_allowed
    (tmp_path / "expected_failures.yaml").write_text(MANIFEST)
    monkeypatch.delenv("SOLACE_BARRIER_WAVE", raising=False)
    assert shortcut_allowed("verify_email_via_sql", manifest_dir=tmp_path) is True
    monkeypatch.setenv("SOLACE_BARRIER_WAVE", "1")
    assert shortcut_allowed("verify_email_via_sql", manifest_dir=tmp_path) is False
    assert shortcut_allowed("unknown", manifest_dir=tmp_path) is False

def test_mvp_039_repo_manifests_are_valid():
    from solace_testing.e2e_manifest import load_manifest
    for path in (Path(__file__).resolve().parents[1]).rglob("expected_failures.yaml"):
        load_manifest(path)                                        # raises on schema errors
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/solace_testing/test_mvp_039_manifest_plugin.py`. Expected: FAIL with `ModuleNotFoundError: No module named 'solace_testing.e2e_manifest'`. There is no way to keep an acceptance test red at a documented hop, so a suite can only be green or skipped (MVP-039).
- [ ] **Step 3: Implement** the module per the Interfaces. Use a `pytest_runtest_makereport` hookwrapper that rewrites `report.outcome`/`wasxfail`/`longrepr` for the `setup` and `call` phases, as pytest's skipping plugin does. Manifests are discovered from each collected item's directory and cached per directory. Schema errors raise `pytest.UsageError` naming the file and the key.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run the unit command once each with `tests/solace_testing`, `tests/alignment`, `tests/system` and `tests/integration`. Expected: all pass; no manifest lists any of their tests.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.4: Compose test stack, per-run secrets and the e2e_stack fixture
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** WS-05.2, WS-05.3, WS-02.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:**
- D-15 (default assumed: strict profiles use verify-full TLS to Postgres, with per-run certificates from WS-02's `gen_certs`);
- D-04 (decided 2026-10-19: the fake LLM only. In W0 the only LLM path is the existing Portkey client, a development/test path, so this overlay points `LLM_PORTKEY_GATEWAY_URL` at the fake and nothing reaches a real provider. In W1, WS-05.21 switches the stacks to Bedrock per WS-09's CR-3: the dev profile gets `LLM_PROVIDER=bedrock` with `BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, and the strict profile gets `LLM_PROVIDER=bedrock` with the LLM off, because strict refuses the override);
- D-12 (default assumed: SMTP through mailpit; in strict, STARTTLS with a per-run CA, because `EMAIL_SMTP_TLS_MODE=none` is allowed only in development/test, IF-02).
**Real infra:** yes
**Files:**
- Create: `docker-compose.test.yml`
- Create: `scripts/test/__init__.py`, `scripts/test/gen_e2e_env.py`, `scripts/test/e2e_stack.ps1`
- Create: `src/solace_testing/e2e.py`
- Create: `tests/e2e/conftest.py`
- Modify: `tests/e2e/expected_failures.yaml` (shortcut `{name: verify_email_via_sql, remove_by_wave: 1, reason: "verification email arrives with WS-14 MVP-035 in W1"}`)
- Create: `tests/scripts/test_mvp_039_gen_e2e_env.py` (generator unit tests; extended by WS-05.13)
- Create: `tests/e2e/test_mvp_039_stack_selftest.py` (harness self-test on a live stack)
- Test: `tests/scripts/test_mvp_039_gen_e2e_env.py`, `tests/e2e/test_mvp_039_stack_selftest.py`

**Interfaces:**
- Consumes:
  - IF-02 registry `deploy/env/env-contract.yaml` (WS-03);
  - IF-21 images; IF-01 `/live`;
  - `python -m scripts.db.gen_certs --out <dir>` (WS-02.5);
  - Caddy site addresses from `CADDY_API_SITE`/`CADDY_AUTH_SITE` in a `caddy` service present in both profiles (request 3);
  - `SOLACE_SECRETS_DIR` bind-mounted at `/run/secrets` (request 3);
  - IF-19 `make_user` / `provision_clinician` flows (IF-08, IF-17).
- **Produces: the overlay.** `docker-compose.test.yml` does the following:
  - Adds `fake-llm` (build `infrastructure/test/fake-llm`; network `backend`; `ports: ["127.0.0.1::8080"]`).
  - Adds `mailpit`:
    - image `axllent/mailpit@sha256:<digest>`; resolve the digest once with `docker buildx imagetools inspect axllent/mailpit:<tag>` and record the tag in a comment;
    - networks `backend` and `monitoring`; `ports: ["127.0.0.1::8025"]`; SMTP 1025 stays internal;
    - `MP_SMTP_ALLOWED_RECIPIENTS='^(user|clin|fallback|ops)-[A-Za-z0-9.-]*@test\.invalid$'`;
    - in strict, `MP_SMTP_TLS_CERT`/`MP_SMTP_TLS_KEY` from `/run/secrets` and `MP_SMTP_REQUIRE_STARTTLS=true`.
  - On every service of the base and prod files: `container_name: !reset null`, `ports: !reset []` and `env_file: !reset []`.
  - On `caddy`: `ports: !override ["127.0.0.1::8080", "127.0.0.1::8081"]`, `CADDY_API_SITE=":8080"`, `CADDY_AUTH_SITE=":8081"`.
  - On every app service:
    - mail: `EMAIL_ENABLED=true`, `EMAIL_SMTP_HOST=mailpit`, `EMAIL_SMTP_PORT=1025`, `EMAIL_SMTP_TLS_MODE=${E2E_SMTP_TLS_MODE}` (`none` in dev, `starttls` in strict), `SSL_CERT_FILE=${E2E_SSL_CERT_FILE:-}`;
    - channels: `SMS_ENABLED=false`, `PUSH_ENABLED=false`;
    - LLM (W0 state; WS-05.21 replaces it with the D-04 Bedrock wiring in W1): `LLM_PORTKEY_GATEWAY_URL=http://fake-llm:8080/v1`, `LLM_PHI_EGRESS_APPROVED=true`, `LLM_ANTHROPIC_API_KEY=""`, `LLM_ENABLE_CACHING=false`;
    - notification: `NOTIFICATION_FALLBACK_ONCALL_EMAIL=fallback-oncall@test.invalid`, `NOTIFICATION_ACK_BASE_URL=http://api.e2e.invalid`;
    - memory: `WEAVIATE_ENABLED=false`.

  It never touches build or image of a base service, nor any other strict setting. `OVERLAY_ALLOWED_KEYS` in `e2e.py` lists exactly the keys above.
- **Produces: the generator.** `python -m scripts.test.gen_e2e_env --profile dev|strict --environment development|staging|production --out <dir> [--registry deploy/env/env-contract.yaml] [--rotate]`:
  - Writes `<dir>/e2e.env` (compose interpolation; `$` escaped as `$$`) and, in strict, `<dir>/secrets/` containing the Postgres TLS files and the mailpit CA, cert and key.
  - Prints variable names only.
  - Restricts ACLs. On Windows: `icacls <dir> /inheritance:r /grant:r "<user>:(OI)(CI)F"`. On POSIX: modes 700/600.
  - Refuses to overwrite without `--rotate`. Exits 2 with `no format rule for <NAME>` for a registry secret it cannot generate.
  - Format rules:

    | Rule | Variables |
    |---|---|
    | hex32 | `ENCRYPTION_MASTER_KEY`, `ENCRYPTION_SEARCH_HASH_SALT`, `AUDIT_HMAC_KEY` |
    | urlsafe32 | `AUTH_SECRET_KEY`, `USER_SERVICE_JWT_SECRET_KEY`, every password, `WEAVIATE_API_KEY`, `LLM_PORTKEY_API_KEY`, `GOOGLE_OAUTH_CLIENT_SECRET`, `BACKUP_ENCRYPTION_KEY` |
    | fernet | `FERNET_TOKEN_KEY`, `FERNET_FIELD_KEY` |
    | bcrypt | `BASIC_AUTH_HASH` |
    | derived | `REDIS_URL` from `REDIS_PASSWORD` |
    | literals | `ENCRYPTION_KEY_ID="e2e-k1"`, `ENCRYPTION_RETIRED_KEYS_JSON="{}"` |

- **Produces: the `e2e` module API.** IF-19 names, plus the additions marked new:
```python
SERVICE_PORTS = {"orchestrator-service": 8000, "user-service": 8001, "safety-service": 8002, "notification-service": 8003,
                 "diagnosis-service": 8004, "memory-service": 8005, "therapy-service": 8006, "personality-service": 8007,
                 "analytics-service": 8009}
MVP_SERVICES = tuple(SERVICE_PORTS)
@dataclass class StackConfig:              # new; from_env(): SOLACE_E2E_PROFILE (dev), SOLACE_E2E_ENVIRONMENT (production when strict),
    profile: str; environment: str; lane: str; keep: bool; reuse: bool; run_dir: Path   # SOLACE_E2E_LANE (a), SOLACE_E2E_KEEP, SOLACE_E2E_REUSE,
    project: str ("solace-e2e-<lane>"); compose_files: list[str]                         # SOLACE_E2E_RUN_DIR (%TEMP%\solace-e2e\<project>)
@dataclass class HttpResult: status: int; json: Any; text: str
@dataclass class TestUser: id: UUID; email: str; password: str; access_token: str; role: str = "user"
@dataclass class Stack:                    # IF-19 fields; urls are in-network base URLs; Postgres is never published, so pg_superuser_dsn is None
    urls: dict[str, str]; edge_url: str; edge_auth_url: str; project: str; mailpit_api: str; fake_llm_control: str
    pg_superuser_dsn: str | None; config: StackConfig; mail: SmtpSink; llm: FakeLLMHandle; fault: "Faults"
    def compose(self, *args: str, check: bool = True, timeout: float = 600) -> subprocess.CompletedProcess
    def call(self, service: str, method: str, path: str, *, json: Any = None, headers: dict | None = None, timeout: float = 10.0) -> HttpResult  # docker compose exec
    def sql(self, query: str, *params: str) -> list[dict]       # psql as the image superuser; params bound as :'p1'.. ; JSON rows
    def logs(self, service: str | None = None) -> str            # IF-19 read_logs
    def kafka_lag(self, group: str) -> dict[str, int]            # kafka-consumer-groups --describe in the kafka container
    def env_value(self, name: str) -> str                        # per-run value for harness use only; never printed
    async def edge(self, method: str, path: str, *, site: str = "api", json: Any = None, headers: dict | None = None, token: str | None = None) -> httpx.Response
class Faults: def stop(self, service), start, restart, pause, unpause
def start_stack(config: StackConfig) -> Stack                    # hop("stack_up"): generate env, compose up -d --build, migrate exit 0, every MVP service /live 200 within 180 s
def stop_stack(stack: Stack) -> None                             # compose down -v --remove-orphans unless keep
def new_email(kind: str) -> str
async def make_user(stack: Stack, role: str = "user", verified: bool = True) -> TestUser   # hops register / verify_email / login; shortcut verify_email_via_sql
def provision_clinician(stack: Stack, email: str, phone: str, on_call: bool = True, backup: bool = False, *,
                        role: str = "CLINICIAN", assign: UUID | None = None) -> TestUser   # docker compose exec -T user-service python -m scripts.provision_clinician ...
def parse_consumer_groups_describe(text: str) -> dict[str, int]; def parse_provision_output(text: str) -> tuple[UUID, str]
def extract_link(text: str, path_prefix: str) -> str
```
  `tests/e2e/conftest.py`:
  - a session fixture `e2e_stack` (`require_docker()`, `start_stack(StackConfig.from_env())`, `stop_stack`);
  - a `profile("dev"|"strict")` marker whose non-matching items are deselected (deselection, not skip, so `SOLACE_REQUIRE_REALINFRA=1` stays meaningful);
  - markers `e2e` and `integration`, the session loop scope, and a default `pytest.mark.timeout(900)`.

- [ ] **Step 1: Write the failing tests.** In `tests/scripts/test_mvp_039_gen_e2e_env.py`:
```python
REGISTRY = """
- {name: AUTH_SECRET_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: ENCRYPTION_MASTER_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: FERNET_FIELD_KEY, services: [user-service], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: BASIC_AUTH_HASH, services: [caddy], secret: true, required_in_strict: true, dev_default: null, notes: "", retired: false}
- {name: ORCHESTRATOR_TURN_TIMEOUT_SECONDS, services: [orchestrator-service], secret: false, required_in_strict: false, dev_default: "20", notes: "", retired: false}
"""
def _gen(tmp_path, *extra, registry=REGISTRY):
    reg = tmp_path / "registry.yaml"; reg.write_text(registry)
    return subprocess.run([sys.executable, "-m", "scripts.test.gen_e2e_env", "--profile", "dev", "--environment", "development",
                           "--out", str(tmp_path / "run"), "--registry", str(reg), *extra], capture_output=True, text=True, cwd=REPO)

def _env(tmp_path): return dict(l.split("=", 1) for l in (tmp_path / "run" / "e2e.env").read_text().splitlines() if "=" in l)

def test_mvp_039_generates_every_registry_secret_without_printing_values(tmp_path):
    r = _gen(tmp_path); env = _env(tmp_path)
    assert r.returncode == 0 and {"AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "FERNET_FIELD_KEY", "BASIC_AUTH_HASH"} <= env.keys()
    assert re.fullmatch(r"[0-9a-f]{64}", env["ENCRYPTION_MASTER_KEY"].strip('"'))
    assert len(base64.urlsafe_b64decode(env["FERNET_FIELD_KEY"].strip('"'))) == 32
    assert "$$2" in env["BASIC_AUTH_HASH"]                                            # bcrypt '$' escaped for compose
    assert all(v.strip('"') not in r.stdout for k, v in env.items() if k != "ORCHESTRATOR_TURN_TIMEOUT_SECONDS")

def test_mvp_039_unknown_secret_name_fails_loud(tmp_path):
    r = _gen(tmp_path, registry=REGISTRY + "- {name: NEW_VENDOR_TOKEN, services: [x], secret: true, required_in_strict: true, dev_default: null, notes: '', retired: false}\n")
    assert r.returncode == 2 and "no format rule for NEW_VENDOR_TOKEN" in r.stderr

def test_mvp_039_refuses_to_overwrite_without_rotate(tmp_path):
    assert _gen(tmp_path).returncode == 0
    assert _gen(tmp_path).returncode != 0
    first = _env(tmp_path)["AUTH_SECRET_KEY"]; assert _gen(tmp_path, "--rotate").returncode == 0
    assert _env(tmp_path)["AUTH_SECRET_KEY"] != first

def test_mvp_039_profiles_set_the_smtp_tls_mode(tmp_path):
    # dev -> E2E_SMTP_TLS_MODE=none; strict (gen_certs monkeypatched via --certs-cmd echo) -> starttls and E2E_SSL_CERT_FILE=/run/secrets/e2e-ca.crt
    ...
```
In `tests/e2e/test_mvp_039_stack_selftest.py`:
```python
def test_mvp_039_no_fixed_container_names_and_only_test_ports_published(e2e_stack):
    ps = [json.loads(l) for l in e2e_stack.compose("ps", "--format", "json").stdout.splitlines() if l.strip()]
    assert all(p["Name"].startswith(e2e_stack.project) for p in ps)
    published = {p["Service"] for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort")}
    assert published <= {"caddy", "mailpit", "fake-llm"}
    assert all(pub["URL"] in ("127.0.0.1", "") for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort"))

def test_mvp_039_sql_runs_as_the_image_superuser(e2e_stack):
    assert e2e_stack.sql("SELECT current_user AS u")[0]["u"] == "postgres"

@pytest.mark.parametrize("service", MVP_SERVICES)
def test_mvp_039_call_reaches_each_service(e2e_stack, service):
    with hop("service_live"):
        assert e2e_stack.call(service, "GET", "/live").status == 200

async def test_mvp_039_edge_routes_to_the_orchestrator_without_internal_paths(e2e_stack):
    assert (await e2e_stack.edge("POST", "/api/v1/orchestrator/chat", json={"message": "hi"})).status_code == 401
    assert (await e2e_stack.edge("GET", "/metrics")).status_code == 404

def test_mvp_039_overlay_changes_only_allowed_keys_in_strict(tmp_path):
    # render 'docker compose -f docker-compose.yml -f docker-compose.prod.yml config --format json' with and without
    # -f docker-compose.test.yml (same generated env file); for every app service the environment differs only in OVERLAY_ALLOWED_KEYS
    ...
```
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/scripts/test_mvp_039_gen_e2e_env.py`. Expected: FAIL with `No module named scripts.test.gen_e2e_env`.
  - Run: E2E command (dev) with `tests/e2e/test_mvp_039_stack_selftest.py`. Expected: FAIL. `e2e_stack` cannot be built because `docker-compose.test.yml` does not exist; the base file's fixed `container_name`s and host ports block a second project. No test runs the built images (MVP-039).
- [ ] **Step 3: Implement** the overlay, the generator, `e2e.py`, the conftest and the PowerShell helper per the Interfaces. Notes:
  - `call()` runs `docker compose -p <project> exec -T <service> python -c <script>`. The script performs one urllib request to `http://127.0.0.1:<port><path>` with base64-encoded arguments and prints `{status, body}`.
  - `sql()` wraps the query as `SELECT coalesce(json_agg(t), '[]') FROM (<query>) t` and binds parameters with `psql -v p1=...`.
  - `make_user` registers through the auth site, verifies (shortcut or mailpit link) and logs in. Each step runs inside `hop(...)`.
- [ ] **Step 4: Run them and confirm they pass.** Run the same two commands. Expected: PASS.
  - Exception: a `test_mvp_039_call_reaches_each_service[...]` case for a service that cannot boot before its W0 phase-B adoption is recorded in the manifest with hop `service_live`, `target_wave: 0`, and shows as XFAIL. At the time of writing that means `notification-service` (MVP-006).
  - Record the observed set in the ledger.
- [ ] **Step 5: Regression.** Run one process each: unit command with `tests/scripts`; unit command with `tests/solace_testing`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.5: W0 boot smoke in dev and strict; in-process smoke on the IF-01 liveness contract
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** WS-05.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-14 (default assumed: staging is as strict as production; the smoke runs with both). D-15 (verify-full TLS in strict). D-19 (config-service excluded).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w0_boot_smoke.py`
- Modify: `tests/e2e/expected_failures.yaml` (W0 entries, `target_wave: 0`)
- Modify: `tests/system/test_smoke.py`. Rewritten to assert `GET /live` returns 200 `{"status": "alive"}` for the six underscore services (orchestrator, safety, diagnosis, therapy, memory, personality) via `TestClient(app, raise_server_exceptions=False)`. This replaces the `/health` == `"healthy"` assertions that the IF-01 adoption retires.
- Test: `tests/e2e/test_w0_boot_smoke.py`

**Interfaces:**
- Consumes:
  - IF-01: `/live`, `/ready` returning `{status:'ready', service, version, checks:{name:{ok, critical, detail}}}` or 503 `not_ready`, and `/metrics` with `solace_readiness_check_ok`;
  - IF-03 `SCHEMA_HEAD`; IF-02 `POSTGRES_SSL_MODE`;
  - the `Stack` API (WS-05.4).
- Produces: the W0 exit-criterion suite.

- [ ] **Step 1: Write the failing test** in `tests/e2e/test_w0_boot_smoke.py` (`pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.timeout(1200)]`):
```python
DB_SERVICES = ("orchestrator-service", "user-service", "safety-service", "notification-service", "diagnosis-service",
               "memory-service", "therapy-service", "personality-service", "analytics-service")
FALLBACK_PATTERNS = [r"publisher_using_in_memory_outbox", r"dlq_using_in_memory_store", r"in_memory_fallback",
                     r"event_outbox_pool_not_available", r"localhost:(5432|6379|9092|29092)"]

def test_w0_migrate_job_completed(e2e_stack):
    with hop("migrate_completed"):
        (job,) = [json.loads(l) for l in e2e_stack.compose("ps", "-a", "--format", "json", "migrate").stdout.splitlines() if l.strip()]
        assert job["State"] == "exited" and job["ExitCode"] == 0
        assert e2e_stack.sql("SELECT version_num AS v FROM alembic_version")[0]["v"] == SCHEMA_HEAD

@pytest.mark.parametrize("service", MVP_SERVICES)
def test_w0_service_ready_within_180s(e2e_stack, service):
    with hop("service_ready"):
        r = _poll(lambda: e2e_stack.call(service, "GET", "/ready"), until=lambda r: r.status == 200, timeout=180)
        assert r.json["status"] == "ready" and r.json["service"] == service
        assert all(c["ok"] for c in r.json["checks"].values() if c["critical"])

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
        ps = [json.loads(l) for l in e2e_stack.compose("ps", "--format", "json").stdout.splitlines() if l.strip()]
        assert {p["Service"] for p in ps for pub in p.get("Publishers") or [] if pub.get("PublishedPort")} <= {"caddy", "mailpit", "fake-llm"}

@pytest.mark.profile("strict")
@pytest.mark.parametrize("service", MVP_SERVICES)
def test_w0_no_fallback_warning_in_logs(e2e_stack, service):
    with hop("no_fallback_logs"):
        text = e2e_stack.logs(service)
        assert [p for p in FALLBACK_PATTERNS if re.search(p, text)] == []

@pytest.mark.profile("dev")
async def test_w0_register_and_login_through_the_edge(e2e_stack):
    with hop("register_login"):
        user = await make_user(e2e_stack)
        assert user.access_token and user.id
```
- [ ] **Step 2: Run it and confirm it fails.** Run, one process each:
  - E2E command (dev) with `tests/e2e/test_w0_boot_smoke.py` and `$env:SOLACE_MANIFEST_IGNORE="1"`;
  - E2E command (strict, `SOLACE_E2E_ENVIRONMENT="production"`);
  - E2E command (strict, `SOLACE_E2E_ENVIRONMENT="staging"`).

  Expected: FAIL, because no service has adopted IF-01 yet:
  - `test_w0_service_serves_metrics[*]` fail at hop `metrics_served`: `/metrics` is not served;
  - `test_w0_service_ready_within_180s[*]` fail at hop `service_ready`: no `checks` in `/ready`, and notification-service never boots;
  - in strict, `app_role_tls` and `no_fallback_logs` fail.

  Then run the unit command with `tests/system/test_smoke.py` as a verification of the contract migration only. It passes today, because the six services already serve `/live`.
- [ ] **Step 3: Implement.** Add one manifest entry per failing test id (patterns allowed), with the observed hop, `target_wave: 0` and a reason naming the owning phase-B adoption. Record any hop that differs from the prediction above in the ledger.
- [ ] **Step 4: Run it and confirm it passes.** Run the three E2E commands without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0, with the listed tests reported XFAIL at their hops and the rest passing. At the W0 barrier, `SOLACE_BARRIER_WAVE=0` makes every entry due and the suite must be fully green.
- [ ] **Step 5: Regression.** Run one process each: unit command with `tests/system`; E2E command (dev) with `tests/e2e`. Expected: all pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.6: Crisis alert E2E written red at documented hops (dev profile)
**Covers:** MVP-039
**Severity:** high
**Wave:** 0
**Depends on:** WS-05.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional lexicon v1; the e2e phrases must be in the clinical phrase set from W1). D-06, D-11 (frame names), D-12 (alert content), D-13 (clinician provisioning), D-27 (WS subprotocol token), D-29 (decided 2026-10-19: crisis resources become a per-locale, versioned registry in `solace_clinical`, and only `en-US` is populated. `crisis_resources("en-US")` keeps returning exactly today's set, the 988 Lifeline "Call or text 988", Crisis Text Line and 911, so these assertions stay valid. Harness users carry the default locale `en-US`. This suite assumes nothing about how many locales exist. The en-US version accessor under per-locale versioning, and any future assertion of the unsupported-locale fallback, wait on WS-12's answer to request 18).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/crisis_phrases_e2e.py`:
  - `CRISIS_PHRASES = {"curly_apostrophe": "I don\u2019t want to be alive anymore", "indirect": "everyone would be better off if I wasn\u2019t here", "homicidal": "I am going to kill him tonight, I have the knife ready", "abuse": "my stepdad hits me every night and I am scared to go home"}`;
  - `BENIGN_PHRASES = {"cutting_vegetables": "I was cutting vegetables for dinner", "kill_for_coffee": "I could kill for a coffee right now"}`.
- Create: `tests/e2e/test_crisis_alert_e2e.py`
- Modify: `src/solace_testing/e2e.py` (crisis-review F6: `crisis_episodes(stack, user_id, *, timeout: float = 0.0) -> list[dict]` (rows `alert_id::text`, `status`, `max_level`, `sources`, `page_count` of `crisis_alerts` for that user through `Stack.sql`, ordered by `opened_at`, polled until at least one row or the timeout); `pages_for_episode(stack, address, alert_id) -> list[MailMessage]` (mail to `address` whose subject contains `ref <first 8 of alert_id>`); `wait_for_episode_page(stack, address, alert_id, timeout) -> MailMessage`)
- Modify: `tests/e2e/expected_failures.yaml` (entries `target_wave: 1`, `profiles: [dev]`)
- Test: `tests/e2e/test_crisis_alert_e2e.py`

**Interfaces:**
- Consumes:
  - IF-12 REST `crisis` object and WS frames, including WS-11.5's informational `connected` frame, always first after accept (an IF-12 addition, crisis-review F5);
  - IF-07 `crisis_resources("en-US")` and `RESOURCES_VERSION` (D-29: the contact text is "Call or text 988", crisis-review F4), imported inside the hop because `solace_clinical` lands in W1 (WS-12.2). Under D-29 the version is tracked per locale, so `RESOURCES_VERSION` here means the en-US version; if WS-12 names a per-locale accessor instead (request 18), `_assert_resources` uses it for `"en-US"`;
  - IF-09 subject `[Solace CRISIS <LEVEL>] action required - ref <8 hex>` and the acknowledgement link `{NOTIFICATION_ACK_BASE_URL}/api/v1/notifications/crisis-alerts/ack/{token}`;
  - IF-08 provisioning CLI;
  - `crisis_alerts.status` (slot 015);
  - the `Stack` helpers.
- Produces: the W0/W1 crisis acceptance suite and the episode-scoped mail helpers (consumed by WS-05.8, WS-05.9 and later suites). `CRISIS_PHRASES`/`BENIGN_PHRASES` are reused by the W1 suites. Isolation rule (crisis-review F6): the stack, the on-call roster and the fallback mailbox are shared by the whole session, so every page assertion is scoped to one user's episode (`crisis_episodes` plus `pages_for_episode`), never to a whole mailbox, and this module provisions its on-call clinician before its first crisis message, so no page waits on an empty roster (crisis-review F1).

- [ ] **Step 1: Write the failing test** in `tests/e2e/test_crisis_alert_e2e.py` (`pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.timeout(900)]`):
```python
@pytest.fixture(scope="module", autouse=True)
def oncall(e2e_stack):
    # crisis-review F1/F6: the module's on-call clinician exists before its first crisis message, so no page of this
    # module waits on an empty roster (a later-provisioned clinician would also receive those earlier users' pages)
    with hop("provision_clinician"):
        return provision_clinician(e2e_stack, email=new_email("clin"), phone="+15555550100", on_call=True)

async def _chat(stack, user, text):
    return await stack.edge("POST", "/api/v1/orchestrator/chat", token=user.access_token, json={"message": text})

def _assert_resources(crisis):
    from solace_clinical import RESOURCES_VERSION, crisis_resources     # W1 (WS-12.2); imported inside the hop
    assert any("988" in res["contact"] for res in crisis["resources"])  # D-29 contact text is "Call or text 988" (F4)
    assert crisis["resources"] == crisis_resources("en-US") and crisis["resources_version"] == RESOURCES_VERSION

@pytest.mark.parametrize("kind", sorted(CRISIS_PHRASES))
async def test_rest_crisis_message_returns_resources_within_3s(e2e_stack, kind):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        t0 = time.monotonic(); r = await _chat(e2e_stack, user, CRISIS_PHRASES[kind]); elapsed = time.monotonic() - t0
        crisis = r.json()["crisis"]
        assert r.status_code == 200 and crisis["level"] in ("HIGH", "CRITICAL")
        _assert_resources(crisis)
        assert elapsed < 3.0

@pytest.mark.parametrize("kind", ["curly_apostrophe", "homicidal"])
async def test_ws_crisis_message_sends_safety_alert_before_reply(e2e_stack, kind):
    user = await make_user(e2e_stack)
    uri = e2e_stack.edge_url.replace("http", "ws") + f"/api/v1/orchestrator/ws/{uuid4()}"
    with hop("ws_connect"):
        ws = await websockets.connect(uri, subprotocols=["solace.v1", f"bearer.{user.access_token}"])
        assert json.loads(await asyncio.wait_for(ws.recv(), 5.0))["type"] == "connected"   # IF-12 addition (WS-11.5, F5)
    try:
        with hop("ws_safety_alert"):
            t0 = time.monotonic()
            await ws.send(json.dumps({"type": "chat.message", "content": CRISIS_PHRASES[kind]}))
            first = json.loads(await asyncio.wait_for(ws.recv(), 3.0))         # the first frame after the send
            assert first["type"] == "safety.alert" and first["level"] in ("HIGH", "CRITICAL")
            assert time.monotonic() - t0 < 3.0
            _assert_resources(first)
            assert json.loads(await asyncio.wait_for(ws.recv(), 20.0))["type"] == "chat.response"
    finally:
        await ws.close()

async def test_crisis_pages_the_oncall_clinician_once_without_message_text(e2e_stack, oncall):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        assert (await _chat(e2e_stack, user, CRISIS_PHRASES["curly_apostrophe"])).json()["crisis"]
    with hop("clinician_page"):
        (episode,) = crisis_episodes(e2e_stack, user.id, timeout=60)          # one episode for this user's turn
        msg = wait_for_episode_page(e2e_stack, oncall.email, episode["alert_id"], timeout=60)
        assert re.fullmatch(r"\[Solace CRISIS (HIGH|CRITICAL)\] action required - ref [0-9a-f]{8}", msg.subject)
        assert "alive anymore" not in msg.text and user.email not in msg.text
        await asyncio.sleep(15)
        pages = pages_for_episode(e2e_stack, oncall.email, episode["alert_id"])
        assert len(pages) == len({p.subject.split("]")[0] for p in pages}) <= 2   # one page per level (a same-turn rise pages again, F8)

async def test_ack_link_acknowledges_the_episode(e2e_stack, oncall):
    user = await make_user(e2e_stack)
    with hop("rest_crisis_object"):
        assert (await _chat(e2e_stack, user, CRISIS_PHRASES["indirect"])).json()["crisis"]
    with hop("clinician_page"):
        (episode,) = crisis_episodes(e2e_stack, user.id, timeout=60)
        msg = wait_for_episode_page(e2e_stack, oncall.email, episode["alert_id"], timeout=60)
    with hop("ack_link"):
        path = urlsplit(extract_link(msg.text, "/api/v1/notifications/crisis-alerts/ack/")).path
        assert (await e2e_stack.edge("GET", path)).status_code == 200
        assert (await e2e_stack.edge("POST", path)).status_code in (200, 303)
        assert crisis_episodes(e2e_stack, user.id)[0]["status"] == "acknowledged"
```
- [ ] **Step 2: Run it and confirm it fails.** Run: E2E command (dev) with `tests/e2e/test_crisis_alert_e2e.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at these hops, which documents that the crisis path has never worked end to end (MVP-039):
  - `provision_clinician`: `scripts.provision_clinician` does not exist, so the module's autouse fixture fails in setup and every test of the module stops at this hop in W0;
  - once WS-14's provisioning lands in W1, `rest_crisis_object` (there is no IF-12 `crisis` object in the response) and `ws_safety_alert` (the documented `chat.message` frame is dropped and no `safety.alert` arrives within 3 s); the manifest entries are moved to the newly observed hop by ledger note, as the hop protocol requires.
- [ ] **Step 3: Implement.** Add a manifest entry per test id with the observed hop, `target_wave: 1`, `profiles: [dev]` and a reason naming the W1 owners (WS-10/WS-11 crisis object and frames; WS-14 provisioning; WS-13 page and acknowledgement).
- [ ] **Step 4: Run it and confirm it passes.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with every test XFAIL at its documented hop. This is the W0 exit criterion "runs in the dev profile and fails only at the hops recorded".
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.7: Contract-test framework and crisis-chain event contracts (IF-20)
**Covers:** MVP-357
**Severity:** medium
**Wave:** 1
**Depends on:** WS-12.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: `solace_clinical.CATEGORIES` from lexicon v1 are the allowed trigger codes)
**Real infra:** no
**Files:**
- Modify: `src/solace_testing/contracts.py` (append the IF-20 API; the existing classes stay)
- Modify: `src/solace_testing/__init__.py` (export the new names)
- Modify: `tests/alignment/test_api_contracts.py`. Replace the three source-inspection tests of clients that W1 rewrites (IF-06/IF-11) with:
  - `test_peer_service_urls_match_service_ports` (registry `SERVICE_*_SERVICE_URL` dev defaults against `SERVICE_PORTS`);
  - `test_orchestrator_agents_use_versioned_provider_prefixes` (AST: every `"/api/..."` constant in the orchestrator's agent modules starts with `/api/v1/<provider>/`). The test only parses those WS-10/WS-11 modules and never edits them. The same check for the orchestrator's client module belongs to that module's owner: request 15, planned in WS-11.2;
  - `test_notification_reaches_user_service_on_8001` (registry `SERVICE_USER_SERVICE_URL` lists notification-service and ends in `:8001`).
- Create: `tests/contracts/README.md` (protocol below), `tests/contracts/contract.schema.json`
- Create: the event contracts:
  - `tests/contracts/events/orchestrator__notification/safety.crisis.detected.json`
  - `tests/contracts/events/safety__notification/safety.crisis.detected.json`
  - `tests/contracts/events/safety__notification/safety.escalation.triggered.json`
  - `tests/contracts/events/safety__notification/safety.assessment.completed.json`
  - `tests/contracts/events/therapy__notification/safety.crisis.detected.json`
  - `tests/contracts/events/diagnosis__notification/safety.crisis.detected.json`
- Create: `tests/contracts/test_event_contract_files.py`
- Test: `tests/solace_testing/test_mvp_357_contract_framework.py` (new)

**Interfaces:**
- Consumes:
  - IF-06 CrisisDetectedEvent rules: category codes, `escalation_action='page_on_call'`, `requires_human_review=True`, `metadata.correlation_id` = the turn id, `metadata.source_service` = the producer;
  - `solace_events.schemas.EVENT_REGISTRY`;
  - IF-07 `solace_clinical.CATEGORIES` and `parse_escalation_reason`, created by WS-12.2 earlier in W1 phase A1; this task depends on it.
- Produces (IF-20):
```python
class ContractFile(BaseModel):                 # tests/contracts/<consumer>__<provider>/<interaction>.json
    consumer: str; provider: str; interaction: str
    request: ContractFileRequest(method: str, path: str, headers_required: list[str], body_example: dict | None)
    response: ContractFileResponse(status: int, body_schema: dict)   # JSON Schema (draft 2020-12)
class EventContract(BaseModel):                # tests/contracts/events/<producer>__<consumer>/<event_type>.json
    consumer: str; producer: str; event_type: str; example: dict; rules: dict[str, str] = {}
    # rules: "categories" | "uuid" | "equals:<json literal>" | "escalation_reason" keyed by dotted path ("metadata.source_service");
    # "escalation_reason" (crisis-review F10): the value is "unspecified" or solace_clinical.parse_escalation_reason accepts it
    # (the IF-06 form "auto|manual:<LEVEL>[:codes]", every code in CATEGORIES)
def load_contracts(contracts_dir: Path) -> list[ContractFile]
def load_event_contracts(contracts_dir: Path) -> list[EventContract]
def verify_provider(app: FastAPI, contracts_dir: Path, *, auth_headers: Callable[[str], dict]) -> list[VerificationResult]
    # synchronous: runs httpx.AsyncClient(transport=httpx.ASGITransport(app)) in a private event loop on a worker thread,
    # so it works from sync tests and from inside a running loop (async tests). Raises AssertionError listing every
    # mismatch: status, JSON-Schema errors (path + validator, never values)
def assert_request_matches(contract: "ContractFile | Mapping[str, Any]", request: "httpx.Request | Mapping[str, Any]") -> None
    # a raw dict (json.loads of the file) is validated into ContractFile first; checks method; path template
    # ({name} placeholders); headers_required present; every body_example key present with the same JSON type
    # (a null example value accepts any type); extra request keys are allowed
def validate_event_contract(contract: EventContract) -> list[str]           # example validates via EVENT_REGISTRY; rules hold
def verify_event_producer(contract: EventContract, produced: "BaseEvent | Mapping[str, Any]") -> list[str]
    # same event_type; validates; rules hold; top-level key set equals the example's (catches renames); messages never echo values
async def verify_event_consumer(contract: EventContract, handler: Callable[["BaseEvent"], Awaitable[None]]) -> None
```
  Protocol, in `tests/contracts/README.md`:
  - The consumer workstream writes `tests/contracts/<consumer>__<provider>/` (WS-10 writes `orchestrator__safety`, WS-11 the other `orchestrator__*`, WS-13 `notification__user`).
  - WS-05 writes `tests/contracts/events/**`, `client__orchestrator/` and, at WS-15's request, `therapy__memory/`.
  - Providers call `verify_provider` in their own suites, and producers call `verify_event_producer`.

- [ ] **Step 1: Write the failing test** in `tests/solace_testing/test_mvp_357_contract_framework.py`:
```python
def _provider(field="level", value="HIGH", status=200):
    app = FastAPI()
    @app.post("/api/v1/safety/check")
    async def check(body: dict):
        return JSONResponse({field: value, "risk_score": 0.9}, status_code=status)
    return app

def _contract_dir(tmp_path):
    d = tmp_path / "orchestrator__safety"; d.mkdir()
    (d / "check.json").write_text(json.dumps({
        "consumer": "orchestrator", "provider": "safety", "interaction": "check",
        "request": {"method": "POST", "path": "/api/v1/safety/check", "headers_required": ["Authorization", "X-Turn-ID"],
                    "body_example": {"user_id": "u", "session_id": "s", "content": "c", "turn_id": "t"}},
        "response": {"status": 200, "body_schema": {"type": "object", "required": ["level", "risk_score"],
                     "properties": {"level": {"enum": ["NONE", "LOW", "ELEVATED", "HIGH", "CRITICAL"]}, "risk_score": {"type": "number"}}}}}))
    return d

def _auth(service): return {"Authorization": "Bearer test", "X-Turn-ID": str(uuid4())}

def test_mvp_357_matching_provider_passes(tmp_path):
    verify_provider(_provider(), _contract_dir(tmp_path), auth_headers=_auth)

@pytest.mark.parametrize("kwargs,needle", [
    ({"field": "crisis_level"}, "'level' is a required property"),
    ({"value": "high"}, "level"),
    ({"status": 202}, "expected status 200, got 202"),
])
def test_mvp_357_provider_drift_is_reported(tmp_path, kwargs, needle):
    with pytest.raises(AssertionError) as exc:
        verify_provider(_provider(**kwargs), _contract_dir(tmp_path), auth_headers=_auth)
    assert needle in str(exc.value)

def test_mvp_357_consumer_request_without_turn_header_is_rejected(tmp_path):
    (contract,) = load_contracts(_contract_dir(tmp_path))
    with pytest.raises(AssertionError, match="X-Turn-ID"):
        assert_request_matches(contract, {"method": "POST", "path": "/api/v1/safety/check", "headers": {"Authorization": "Bearer t"},
                                          "json": {"user_id": "u", "session_id": "s", "content": "c", "turn_id": "t"}})

def test_mvp_357_raw_contract_dict_and_httpx_request_are_accepted(tmp_path):
    raw = json.loads((_contract_dir(tmp_path) / "check.json").read_text())       # consumers pass json.loads(...) dicts
    req = httpx.Request("POST", "http://safety/api/v1/safety/check", headers=_auth("orchestrator"),
                        json={"user_id": "u", "session_id": "s", "content": "c", "turn_id": "t", "extra": 1})
    assert_request_matches(raw, req)                                             # extra keys allowed
    with pytest.raises(AssertionError, match="turn_id"):
        assert_request_matches(raw, httpx.Request("POST", "http://safety/api/v1/safety/check", headers=_auth("o"),
                                                  json={"user_id": "u", "session_id": "s", "content": "c"}))

async def test_mvp_357_verify_provider_works_inside_a_running_event_loop(tmp_path):
    verify_provider(_provider(), _contract_dir(tmp_path), auth_headers=_auth)    # provider suites are often async

def test_mvp_357_raw_trigger_text_and_renamed_fields_are_reported():
    contract = load_event_contracts(CONTRACTS / "events" / "diagnosis__notification")[0]
    bad = {**contract.example, "trigger_indicators": ["I want to end my life"]}
    errors = verify_event_producer(contract, bad)
    assert any(e.startswith("trigger_indicators[0]") for e in errors) and not any("end my life" in e for e in errors)
    renamed = {k: v for k, v in contract.example.items() if k != "crisis_level"} | {"level": "HIGH"}
    assert any("crisis_level" in e for e in verify_event_producer(contract, renamed))

def test_crisis_review_f10_escalation_reason_rule_accepts_the_producer_format_only():
    contract = next(c for c in load_event_contracts(CONTRACTS / "events" / "safety__notification")
                    if c.event_type == "safety.escalation.triggered")
    assert contract.rules["escalation_reason"] == "escalation_reason"
    for ok in ("auto:CRITICAL:suicidal_ideation", "auto:HIGH", "unspecified"):
        assert verify_event_producer(contract, {**contract.example, "escalation_reason": ok}) == [], ok
    errors = verify_event_producer(contract, {**contract.example, "escalation_reason": "Triggers: kill myself"})
    assert any(e.startswith("escalation_reason") for e in errors) and not any("kill myself" in e for e in errors)
```
In `tests/contracts/test_event_contract_files.py`:
```python
CONTRACTS = Path(__file__).resolve().parent
@pytest.mark.parametrize("contract", load_event_contracts(CONTRACTS / "events"), ids=lambda c: f"{c.producer}__{c.consumer}:{c.event_type}")
def test_event_contract_example_is_valid(contract):
    assert validate_event_contract(contract) == []

def test_crisis_chain_covers_every_producer():
    by_type = defaultdict(set)
    for c in load_event_contracts(CONTRACTS / "events"):
        assert c.consumer == "notification-service"
        by_type[c.event_type].add(c.producer)
    assert by_type["safety.crisis.detected"] == {"orchestrator-service", "safety-service", "therapy-service", "diagnosis-service"}
    assert by_type["safety.escalation.triggered"] == by_type["safety.assessment.completed"] == {"safety-service"}
```
- [ ] **Step 2: Run it and confirm it fails.** Run, one process each: unit command with `tests/solace_testing/test_mvp_357_contract_framework.py`; unit command with `tests/contracts`. Expected: FAIL with `ImportError: cannot import name 'verify_provider' from 'solace_testing.contracts'`. No producer/consumer contract check exists for any REST call or the crisis-event chain; `ConsumerContractTest`/`ProviderContractTest` have no user outside their self-test (MVP-357).
- [ ] **Step 3: Implement.**
  - Add the API to `contracts.py` using `jsonschema.Draft202012Validator`. Error messages carry `path` and the validator keyword, never instance values.
  - Write the six event contracts from IF-06. Each example is a real `to_dict()` output of the schema class, and each has rules for `trigger_indicators` (categories), `escalation_action` (`equals:"page_on_call"`), `requires_human_review` (`equals:true`), `metadata.source_service` (`equals:"<producer>"`), `metadata.correlation_id` (uuid) and `user_id` (uuid). The escalation contract checks `escalation_reason` with the `escalation_reason` rule (crisis-review F10: the value WS-12.8 emits and WS-13.6 parses is `auto:<LEVEL>:<codes>`, which a `categories` rule would reject) and uses `"auto:CRITICAL:suicidal_ideation"` as its example; the assessment contract checks `risk_factors[*].evidence` as categories.
  - Retarget the alignment tests as listed. This is a verification-only change: the replaced checks would fail for the wrong reason when WS-10/WS-11 rewrite those clients in W1 phase B.
- [ ] **Step 4: Run it and confirm it passes.** Run the same two commands. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each with the unit command: `tests/solace_testing`, `tests/alignment`, `tests/contracts`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.8: W1 crisis fault suite and negative-phrase E2E written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 1
**Depends on:** WS-05.6
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (provisional lexicon; the e2e phrases must appear in the clinical phrase set). D-06 (default assumed: 15-minute coalescing; re-page of the backup and the fallback mailbox; a same-turn rise pages again, crisis-review F8). D-22 (default assumed: the local screen stands when the safety service is down or hung, inside the 3 s budget). D-28 (notification is the only pager; a page survives a database outage, crisis-review F2).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w1_crisis_faults.py`
- Create: `tests/e2e/test_w1_negative_phrases.py`
- Modify: `docker-compose.test.yml` (app services: `NOTIFICATION_REPAGE_AFTER_SECONDS=30`, `NOTIFICATION_ALERT_COALESCE_SECONDS=900`, `NOTIFICATION_MAX_REPAGES=3`)
- Modify: `src/solace_testing/e2e.py` (`pages_since(stack, address, t0) -> list[MailMessage]`; `wait_ready(stack, service, timeout) -> HttpResult`; `crisis_episodes`, `pages_for_episode` and `wait_for_episode_page` come from WS-05.6)
- Modify: `tests/e2e/expected_failures.yaml` (entries `target_wave: 1`, both profiles)
- Test: the two new files

**Interfaces:**
- Consumes: IF-06, IF-09 (recipients care team, then on-call roster, then fallback; one coalesced episode; re-page), IF-10 degraded reasons, IF-12, IF-15 fake-LLM modes, IF-08 `--assign`, IF-19 `fault.pause`/`unpause`, WS-05.6's episode-scoped mail helpers, and IF-07 `tests/clinical/crisis_phrases_v1.yaml` (`positive[].text`, `negative[].text`).
- Produces: the W1 exit-criterion fault suite, in both profiles.

- [ ] **Step 1: Write the failing tests.** In `tests/e2e/test_w1_crisis_faults.py` (`pytest.mark.timeout(1500)`). Each test provisions its own clinicians under `hop("provision_clinician")`, sends through the edge, and restores faults in `finally`. Every page count below is scoped to the test user's episode (`crisis_episodes(stack, user.id)` and `pages_for_episode`), never to a whole mailbox: the on-call clinicians that earlier tests provision stay on call, and the fallback mailbox is shared by the whole session (crisis-review F6). "1 page" means one page per distinct level of the episode, normally one (a same-turn rise to CRITICAL pages again, crisis-review F8):
```python
async def test_all_agents_failing_still_returns_resources_and_pages(e2e_stack):
    # llm.scenario("error"); fault.stop("therapy-service","diagnosis-service","personality-service","memory-service")
    # hop rest_crisis_object: crisis set, degraded non-empty; hop clinician_page: exactly 1 page to the care-team clinician in 60 s
async def test_notification_down_two_minutes_pages_once_after_restart(e2e_stack):
    # fault.stop("notification-service"); crisis; sleep 120; fault.start; hop clinician_page: 1 page within 90 s of start, still 1 after 30 s more
async def test_rejected_primary_recipient_pages_the_fallback_mailbox(e2e_stack):
    # clinician new_email("reject") provisioned with assign=user.id (care team); crisis
    # hop fallback_page: fallback-oncall@test.invalid gets 1 page for this user's episode within 60 s; reject address gets 0
async def test_unreachable_recipient_for_one_user_does_not_delay_another_users_page(e2e_stack):
    # crisis-review F1: user A's care-team clinician new_email("reject") (refused at RCPT) via assign=A.id; user B's care-team
    # clinician new_email("clin") via assign=B.id; A sends a crisis, then B one second later
    # hop clinician_page: B's episode page reaches B's clinician within 60 s of B's message; A's episode page reaches the
    # fallback mailbox (pages_for_episode) and nothing reaches A's reject address; neither user waits on the other's retries
@pytest.mark.parametrize("mode", ["stop", "pause"])
async def test_postgres_outage_during_a_crisis_turn_still_pages_once_after_recovery(e2e_stack, mode):
    # crisis-review F2. User and care-team clinician provisioned first;
    # fault.stop or fault.pause("postgres"); crisis message; hop rest_crisis_object: crisis set within 3 s;
    # after 60 s fault.start or fault.unpause("postgres") (also in finally); wait_ready for orchestrator, safety, notification
    # hop clinician_page: within 120 s of recovery crisis_episodes(user) has exactly 1 episode and its page reaches the
    # clinician; still 1 page per level 30 s later. From W2 the thread-ownership lookup (WS-11.7) also sits on this path and
    # must stay inside the 3 s budget; this test guards it at every later barrier.
async def test_kafka_down_at_boot_ready_503_then_recovery_and_delivery(e2e_stack):
    # fault.stop("kafka"); restart orchestrator, safety, notification; hop ready_503: /ready 503 for each within 60 s
    # fault.start("kafka"); hop ready_recovers: /ready 200 within 180 s; crisis -> hop clinician_page: 1 page
@pytest.mark.parametrize("mode", ["error", "timeout"])
async def test_llm_failure_still_gives_resources_and_a_page(e2e_stack, mode):
    # llm.scenario(mode); hop rest_crisis_object (crisis within 3 s even in "timeout"); hop clinician_page
async def test_safety_service_down_uses_local_screen_and_reports_degraded(e2e_stack):
    # fault.stop("safety-service"); hop rest_crisis_object: crisis set and any(d.startswith("safety_service") for d in degraded)
    # hop clinician_page: 1 page
async def test_safety_service_hung_still_answers_within_3s(e2e_stack):
    # crisis-review F7: fault.pause("safety-service") (a hung /check, not a refused connection); crisis message
    # hop rest_crisis_object: elapsed < 3.0, crisis set and "safety_service_timeout" in degraded; unpause in finally
    # hop clinician_page: 1 page
async def test_orchestrator_and_safety_detecting_the_same_message_page_once(e2e_stack):
    # hop clinician_page: 1 page per level over 60 s (normally exactly 1); crisis_episodes(user) has 1 row with >= 2 sources
async def test_unacknowledged_episode_repages_backup_and_fallback(e2e_stack):
    # primary on call + backup (backup=True); crisis; no ack
    # hop repage: backup and fallback each get a page within 30 + 90 s; primary still has exactly 1
```
In `tests/e2e/test_w1_negative_phrases.py`:
```python
def test_e2e_phrases_are_in_the_clinical_phrase_set():
    with hop("phrase_set"):
        data = yaml.safe_load((REPO / "tests/clinical/crisis_phrases_v1.yaml").read_text(encoding="utf-8"))
        positives, negatives = {p["text"] for p in data["positive"]}, {n["text"] for n in data["negative"]}
        assert set(CRISIS_PHRASES.values()) <= positives and set(BENIGN_PHRASES.values()) <= negatives

@pytest.mark.parametrize("kind", sorted(BENIGN_PHRASES))
async def test_benign_phrase_pages_nobody(e2e_stack, kind):
    user = await make_user(e2e_stack)
    with hop("benign_no_page"):
        r = await e2e_stack.edge("POST", "/api/v1/orchestrator/chat", token=user.access_token, json={"message": BENIGN_PHRASES[kind]})
        assert r.status_code == 200 and (r.json()["crisis"] is None or r.json()["crisis"]["level"] in ("NONE", "LOW", "ELEVATED"))
        await asyncio.sleep(30)
        assert crisis_episodes(e2e_stack, user.id) == []      # scoped to this user: other tests' re-pages share the mailboxes (F6)
```
- [ ] **Step 2: Run them and confirm they fail.** Run, one process each: E2E command (dev) with both files and `SOLACE_MANIFEST_IGNORE=1`; E2E command (strict, production). Expected: FAIL at `provision_clinician` or `rest_crisis_object` for every fault test, because W1 phase B (provisioning, crisis object, episodes) has not landed. `phrase_set` fails if `tests/clinical/crisis_phrases_v1.yaml` lacks the e2e phrases. The crisis-review tests (F1 isolation, F2 Postgres outage, F7 hung safety-service) fail at the same early hops in W0/W1 phase A and pin, from the W1 barrier on, that one user's unreachable recipient never delays another user's page, that a database outage during a crisis turn still pages once, and that a hung `/check` stays inside the 3 s budget.
- [ ] **Step 3: Implement** the helpers and overlay values listed. Add one manifest entry per observed failing test id and hop: `target_wave: 1`, both profiles.
- [ ] **Step 4: Run them and confirm they pass.** Run the two commands without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with the listed tests XFAIL. At the W1 barrier (`SOLACE_BARRIER_WAVE=1`) both profiles must be fully green.
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.9: Crisis-path alerting E2E, Alertmanager to mailpit, written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 1
**Depends on:** WS-05.8
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w1_crisis_alerting.py`
- Modify: `src/solace_testing/e2e.py` (`StackConfig.compose_files` includes `docker-compose.observability.yml` whenever the file exists; `wait_for_alert(stack, alertname, timeout) -> MailMessage` on `ops-*` recipients)
- Modify: `docker-compose.test.yml` (Alertmanager environment: `ALERTMANAGER_SMTP_SMARTHOST=mailpit:1025`, `ALERTMANAGER_EMAIL_TO=ops-alerts@test.invalid`, `ALERTMANAGER_EMAIL_FROM=ops-alertmanager@test.invalid`)
- Modify: `tests/e2e/expected_failures.yaml`
- Test: `tests/e2e/test_w1_crisis_alerting.py`

**Interfaces:**
- Consumes:
  - IF-18 W1 rules: `CrisisAlertDeliveryFailed`, `CrisisOutboxBacklog`, `NotificationConsumerLag`, `DLQUnresolved`, `ServiceNotReady`, plus WS-04.7's `CrisisConsumerStalled` (crisis-review F13);
  - IF-05 metrics (`solace_dlq_unresolved{group}` etc.; from WS-06.10 every event plane also samples the crisis group's `solace_consumer_lag`, so the series exists while notification-service is stopped or frozen);
  - Alertmanager routing to `ALERTMANAGER_EMAIL_TO` (request 4).
- Produces: the W1 alerting exit criterion.

- [ ] **Step 1: Write the failing tests** (`pytest.mark.timeout(1500)`; each starts with `with hop("alertmanager_running"): assert "alertmanager" in _running_services(e2e_stack)`):
```python
async def test_dlq_unresolved_alert_reaches_the_ops_mailbox(e2e_stack):
    # e2e_stack.compose("exec","-T","kafka","bash","-c","printf 'not-json\n' | kafka-console-producer --bootstrap-server localhost:9092 --topic solace.safety")
    # hop alert_email: wait_for_alert(e2e_stack, "DLQUnresolved", timeout=360)
async def test_notification_stopped_fires_service_not_ready_and_consumer_lag(e2e_stack):
    # fault.stop("notification-service"); send 11 crisis messages from 11 users
    # hop alert_email: "ServiceNotReady" within 300 s and "NotificationConsumerLag" within 540 s; fault.start in finally
async def test_outbox_backlog_fires_when_kafka_is_down(e2e_stack):
    # fault.stop("kafka"); one crisis message; hop alert_email: "CrisisOutboxBacklog" within 360 s; fault.start in finally
async def test_crisis_alert_delivery_failure_fires(e2e_stack):
    # care-team clinician new_email("reject") via assign; crisis; hop alert_email: "CrisisAlertDeliveryFailed" within 360 s
async def test_single_crisis_turn_with_a_stalled_consumer_fires_crisis_consumer_stalled(e2e_stack):
    # crisis-review F13: fault.pause("notification-service") (alive container, no progress); ONE crisis message from one user
    # (4 events on solace.safety, below NotificationConsumerLag's 10); hop alert_email: "CrisisConsumerStalled" within 420 s;
    # fault.unpause in finally; hop clinician_page after unpause: that user's episode is paged within 120 s
```
- [ ] **Step 2: Run them and confirm they fail.** Run: E2E command (dev) with `tests/e2e/test_w1_crisis_alerting.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at hop `alertmanager_running`, because the observability fragment with Alertmanager does not exist yet. Even with it, no rule could fire: there are no crisis-path metrics (MVP-039, with MVP-353 tested in WS-06.10). The stalled-consumer case also needs WS-06.10's crisis-group watch and WS-04.7's `CrisisConsumerStalled`; without them one stuck crisis turn (4 messages) alerts nobody (crisis-review F13).
- [ ] **Step 3: Implement** the helpers and overlay values listed. Add manifest entries (hop `alertmanager_running`, `target_wave: 1`, both profiles).
- [ ] **Step 4: Run them and confirm they pass.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with XFAIL at `alertmanager_running`.
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.10: REST contracts: client protocol contracts and the contract-coverage guard
**Covers:** MVP-357
**Severity:** medium
**Wave:** 2
**Depends on:** WS-05.7, WS-10.3, WS-12.8, WS-11.2, WS-17.19
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-11 (default assumed: the code is the contract, and IF-12 shapes are canonical)
**Real infra:** no
**Files:**
- Create: `tests/contracts/client__orchestrator/chat.post.json`. Request `{message, thread_id?, session_id?}` with `Authorization`; the example has no `thread_id` (a new thread) and is sent as the seed user `00000000-0000-4000-a000-000000000001` (WS-11's seed-id request). Response 200 schema `{response, thread_id, session_id, turn_id, crisis: null | {level, categories, resources[{name, contact, kind}], resources_version}, degraded[], agents[], request_id}`.
- Create: `tests/contracts/client__orchestrator/chat.foreign-thread.post.json` (thread `00000000-0000-4000-b000-000000000002`, owned by `00000000-0000-4000-a000-000000000002`; 404 `{error:{code:"THREAD_NOT_FOUND", message, request_id}}`)
- Create: `tests/contracts/client__orchestrator/history.get.json` (`GET /api/v1/orchestrator/sessions/{session_id}/history?thread_id=` with session `00000000-0000-4000-c000-000000000001` and thread `00000000-0000-4000-b000-000000000001`, owned by the seed user)
- Create: `tests/contracts/therapy__memory/context.json`, written here at WS-15's request (R14 of WS-15) because `tests/contracts` is WS-05's; it copies the provider's published examples and edits nothing in memory-service (request 16):
  - `consumer: "therapy"`, `provider: "memory"`, `interaction: "context"`;
  - request: `POST /api/v1/memory/context`, `headers_required: ["Authorization"]` (the forwarded obo token), `body_example` equal to WS-17.19's `THERAPY_CONTEXT_REQUEST_EXAMPLE`, that is `{"user_id": "00000000-0000-4000-a000-000000000001", "session_id": "00000000-0000-4000-b000-000000000001", "current_message": "example message"}`, so `user_id` is present, as WS-17's R-17-04 asks;
  - response: status 200, `body_schema` requiring `user_id` (string, uuid format), `assembled_context` (string), `total_tokens` (integer >= 0), `sources_used` (array of strings), `assembly_time_ms` (integer >= 0) and `retrieval_count` (integer), the fields of WS-17.19's `THERAPY_CONTEXT_RESPONSE_EXAMPLE`.
- Create: `tests/contracts/test_contract_coverage.py`
- Create: `tests/contracts/expected_failures.yaml`
- Test: `tests/contracts/test_contract_coverage.py`

**Interfaces:**
- Consumes: IF-11 pairs, IF-12, IF-08 (`notification__user`), and IF-20 (WS-05.7); memory's published `services.memory_service.src.schemas.THERAPY_CONTEXT_REQUEST_EXAMPLE` and `THERAPY_CONTEXT_RESPONSE_EXAMPLE` (WS-17.19, request 16); the W1 producer verifications of WS-10.3 and WS-12.8 (request 9; WS-15.5 and WS-16.5 are W2 phase B, so their entries stay in the manifest until they land); WS-11.2's client-module prefix lock (request 15).
- Produces:
  - `EXPECTED_REST_PAIRS = {("orchestrator","safety"), ("orchestrator","therapy"), ("orchestrator","diagnosis"), ("orchestrator","personality"), ("orchestrator","memory"), ("therapy","memory"), ("notification","user"), ("client","orchestrator")}`;
  - `PROVIDER_SUITES` (provider to `services/<dir>/tests`);
  - `tests/contracts/therapy__memory/context.json`, consumed by WS-15.10 (`assert_request_matches` in `services/therapy_service/tests/test_mvp_194_memory_context_contract.py`) and verified by WS-17.18. This satisfies WS-15's request R14 (the `therapy__memory` part of its old R5). The `orchestrator__therapy` part belongs to WS-11.9; see Cross-workstream request 12;
  - the three `client__orchestrator` files with WS-11's seed ids, verified as provider by WS-11.15, which creates no file in that directory (request 13).

- [ ] **Step 1: Write the failing test** in `tests/contracts/test_contract_coverage.py`:
```python
@pytest.mark.parametrize("consumer,provider", sorted(EXPECTED_REST_PAIRS))
def test_rest_pair_has_contract_files_and_a_provider_verification(consumer, provider):
    with hop("consumer_contract_files"):
        assert load_contracts(CONTRACTS / f"{consumer}__{provider}")
    with hop("provider_verification"):
        texts = [p.read_text(encoding="utf-8") for p in (REPO / PROVIDER_SUITES[provider]).rglob("test_*.py")]
        assert any("verify_provider(" in t and f"{consumer}__{provider}" in t for t in texts)

@pytest.mark.parametrize("contract", load_event_contracts(CONTRACTS / "events"), ids=lambda c: f"{c.producer}__{c.consumer}:{c.event_type}")
def test_event_contract_has_a_producer_verification(contract):
    with hop("producer_verification"):
        suite = REPO / PROVIDER_SUITES[contract.producer.removesuffix("-service")]
        assert any("verify_event_producer(" in p.read_text(encoding="utf-8") and f"{contract.producer.removesuffix('-service')}__" in p.read_text(encoding="utf-8")
                   for p in suite.rglob("test_*.py"))

def test_every_contract_file_matches_the_schema():
    schema = json.loads((CONTRACTS / "contract.schema.json").read_text())
    for path in CONTRACTS.rglob("*.json"):
        if path.name != "contract.schema.json":
            jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)

def test_therapy_memory_example_is_the_providers_published_example():
    from services.memory_service.src.schemas import THERAPY_CONTEXT_REQUEST_EXAMPLE   # WS-17.19, request 16
    contract = json.loads((CONTRACTS / "therapy__memory" / "context.json").read_text(encoding="utf-8"))
    assert contract["request"]["body_example"] == THERAPY_CONTEXT_REQUEST_EXAMPLE

def test_orchestrator_client_module_prefixes_are_locked_by_their_owner():
    with hop("client_prefix_lock"):   # request 15: the check moved from WS-05.7's alignment test to WS-11.2
        texts = [p.read_text(encoding="utf-8") for p in (REPO / PROVIDER_SUITES["orchestrator"]).rglob("test_*.py")]
        assert any("def test_orchestrator_clients_use_versioned_provider_prefixes" in t for t in texts)
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/contracts/test_contract_coverage.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL. `consumer_contract_files` fails for every pair, because no REST contract file exists yet, and `provider_verification` fails for all; `test_therapy_memory_example_is_the_providers_published_example` fails with `FileNotFoundError`. No orchestrator test checks a provider's schemas today (MVP-357). `test_orchestrator_client_module_prefixes_are_locked_by_their_owner` passes at once, because WS-11.2 (W1) created the lock; it guards against the lock being dropped.
- [ ] **Step 3: Implement** the three client contracts from IF-12 with WS-11's seed ids and `therapy__memory/context.json` as listed (its `body_example` is `THERAPY_CONTEXT_REQUEST_EXAMPLE` copied verbatim; nothing in memory-service is edited). Write `tests/contracts/expected_failures.yaml` with one entry per observed failing id: hop as observed, `target_wave: 2`, and a reason naming the owner (WS-10 `orchestrator__safety`, WS-11 the other `orchestrator__*` pairs, WS-13 `notification__user`, each provider's verification). Event-producer entries for producers already verified in W1 must not be listed.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with the listed ids XFAIL; `consumer_contract_files` passes for `client__orchestrator` and `therapy__memory`. At the W2 barrier every pair must pass.
- [ ] **Step 5: Regression.** Run the unit command with `tests/contracts` and with `tests/alignment`. Expected: all pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.11: W2 chat journey E2E written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 2
**Depends on:** WS-05.9
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (crisis watch: resources on the following turns). D-30 (no diagnosis label shown to users).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w2_chat_journey.py`
- Modify: `src/solace_testing/e2e.py` (`WsClient(stack, user, session_id)` with `send_chat(content, thread_id=None)`, `recv_typed(type, timeout)` and `close()`; `chat_rest(stack, user, text, thread_id=None) -> dict`)
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 2`, both profiles)
- Test: `tests/e2e/test_w2_chat_journey.py`

**Interfaces:**
- Consumes: IF-10 (checkpoint durability), IF-11, IF-12, IF-13 ciphertext prefix `enc:v1:`, IF-15 (`echo`, `error`, `crisis_intent` modes), IF-17 (real verification through mailpit; the `verify_email_via_sql` shortcut has expired since W1).
- Produces: the W2 exit-criterion suite.

- [ ] **Step 1: Write the failing tests.** Each step is in a named hop:
```python
async def test_verified_user_holds_five_turn_rest_and_ws_chats(e2e_stack):
    # make_user (real mailpit verification); 5 REST + 5 WS turns
    # hop agents_contribute: degraded == [] and {"therapy","personality","memory","diagnosis"} <= set(agents) on every turn
async def test_prior_turns_are_recalled(e2e_stack):
    # llm.scenario("echo"); turn 1 "my dog is named Biscuit"; turn 3 -> hop memory_recall: "Biscuit" in llm.requests()[-1] prompt text
async def test_orchestrator_restart_continues_from_the_postgres_checkpoint(e2e_stack):
    # 2 turns; fault.restart("orchestrator-service"); wait_ready
    # hop checkpoint_resume: turn 3 on the same thread_id is 200 and its prompt carries turn-1 content
async def test_foreign_thread_id_gives_404(e2e_stack):
    # hop thread_ownership: user B with user A's thread_id -> 404 and error.code == "THREAD_NOT_FOUND"
async def test_unknown_ws_frame_gets_an_unsupported_frame_error(e2e_stack):
    # hop ws_error_frame: {"type":"bogus"} -> {"type":"error","code":"UNSUPPORTED_FRAME"}
async def test_llm_outage_reports_degraded_and_never_fakes_success(e2e_stack):
    # llm.scenario("error"); hop llm_degraded: "llm_unavailable" in degraded and FAKE_REPLY not in response
async def test_phrase_only_llm_assisted_detectors_catch_pages_exactly_once(e2e_stack):
    # llm.scenario("crisis_intent", match="bridge downtown"); "I keep thinking about the bridge downtown"
    # hop single_page: exactly 1 page over 60 s
async def test_crisis_turn_is_followed_by_a_normal_reply_with_resources(e2e_stack):
    # turn 3 crisis, turn 4 benign; hop crisis_watch: turn 4 response is 200, not the crisis template,
    # and carries 988 (in crisis.resources or the reply footer)
async def test_phi_is_ciphertext_at_rest(e2e_stack):
    # after a chat with marker "Biscuit": hop ciphertext_at_rest: every memory_records.content and therapy transcript value
    # for the user starts with "enc:v1:"; no checkpoint row text contains the marker
```
- [ ] **Step 2: Run them and confirm they fail.** Run: E2E command (dev) with `tests/e2e/test_w2_chat_journey.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at the hops of W2 features not yet built: `agents_contribute` (template fallbacks, MVP-076), `checkpoint_resume` (InMemorySaver), `thread_ownership`, `ciphertext_at_rest` and others as observed.
- [ ] **Step 3: Implement** the helpers. Add manifest entries per observed id and hop, with `target_wave: 2` and both profiles.
- [ ] **Step 4: Run them and confirm they pass.** Run: E2E commands (dev, then strict). Expected: exit 0 with the listed tests XFAIL.
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.12: Stub OIDC provider for Google sign-in tests
**Covers:** MVP-039
**Severity:** high
**Wave:** 3
**Depends on:** WS-05.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `infrastructure/test/stub-oidc/app.py` (FastAPI, PyJWT and cryptography at the lock versions)
- Create: `infrastructure/test/stub-oidc/Dockerfile` (built like the service images per IF-21: repo-root context, pinned base, `uv pip sync requirements.lock`)
- Modify: `docker-compose.test.yml`:
  - add the `stub-oidc` service (network `backend`, no published port);
  - on user-service: `GOOGLE_OAUTH_ISSUER=http://stub-oidc:8090`, `GOOGLE_OAUTH_AUTHORIZE_URL=http://stub-oidc:8090/authorize`, `GOOGLE_OAUTH_TOKEN_URL=http://stub-oidc:8090/token`, `GOOGLE_OAUTH_JWKS_URL=http://stub-oidc:8090/jwks`, `GOOGLE_OAUTH_CLIENT_ID=e2e-client`, `GOOGLE_OAUTH_REDIRECT_URI=http://auth.e2e.invalid/api/v1/auth/oauth/google/callback`, `OAUTH_ALLOWED_REDIRECT_ORIGINS=http://app.e2e.invalid`;
  - `GOOGLE_OAUTH_CLIENT_SECRET` comes from the per-run env file.
- Test: `tests/solace_testing/test_mvp_039_stub_oidc.py` (new)

**Interfaces:**
- Consumes: IF-17 `GOOGLE_OAUTH_*` names.
- Produces, all new:
  - `GET /authorize?client_id&redirect_uri&state&nonce&response_type=code&scope` returns a 302 to `redirect_uri?code&state`, with the identity set by `POST /__control/identity {sub, email, email_verified}`;
  - `POST /token` (`authorization_code`; `client_secret` checked; single-use code) returns `{access_token, id_token, token_type:"Bearer", expires_in:3600}`. The id_token is RS256 with `iss`, `aud=client_id`, `sub`, `email`, `email_verified`, `nonce`, `iat` and `exp`;
  - `GET /jwks`; `GET /.well-known/openid-configuration`;
  - the app factory `create_app(*, issuer, client_id, client_secret) -> FastAPI` for in-process tests.

- [ ] **Step 1: Write the failing test** in `tests/solace_testing/test_mvp_039_stub_oidc.py`:
```python
def test_mvp_039_id_token_verifies_against_jwks():
    client = TestClient(_load().create_app(issuer="http://stub", client_id="e2e-client", client_secret="s3cret"))
    client.post("/__control/identity", json={"sub": "g-123", "email": "user-a@test.invalid", "email_verified": True})
    loc = client.get("/authorize", params={"client_id": "e2e-client", "redirect_uri": "http://cb/x", "state": "st", "nonce": "n1",
                                           "response_type": "code", "scope": "openid email"}, follow_redirects=False).headers["location"]
    code = parse_qs(urlsplit(loc).query)["code"][0]
    tok = client.post("/token", data={"grant_type": "authorization_code", "code": code, "client_id": "e2e-client",
                                      "client_secret": "s3cret", "redirect_uri": "http://cb/x"}).json()
    key = jwt.PyJWKClient  # use the JWKS document from client.get("/jwks").json() to build the key
    claims = _verify(tok["id_token"], client.get("/jwks").json(), audience="e2e-client", issuer="http://stub")
    assert (claims["sub"], claims["email"], claims["email_verified"], claims["nonce"]) == ("g-123", "user-a@test.invalid", True, "n1")

def test_mvp_039_codes_are_single_use_and_secrets_checked():
    # second exchange of the same code -> 400; wrong client_secret -> 401
    ...
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/solace_testing/test_mvp_039_stub_oidc.py`. Expected: FAIL with `FileNotFoundError` for `infrastructure/test/stub-oidc/app.py`. Google sign-in cannot be tested without calling Google.
- [ ] **Step 3: Implement** the stub per the Interfaces. The RSA key is generated per process start, and the `kid` is its SHA-256 thumbprint.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run the unit command with `tests/solace_testing`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.13: Harness secrets from files (IF-22) and the secrets-injection E2E written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 3
**Depends on:** WS-05.4, WS-05.11, WS-05.12, WS-05.21
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (decided 2026-10-19: SOPS + age; the consumer contract is unchanged. Strict profiles read every secret from `/run/secrets` through NAME_FILE. In deployment, `deploy/render_secrets.py` decrypts `deploy/secrets/<env>.sops.yaml` there. In tests, the generated per-run directory of synthetic values stands in for that render. No test decrypts a real SOPS file, needs an age private key or prints a value)
**Real infra:** yes
**Files:**
- Modify: `scripts/test/gen_e2e_env.py`. In strict, every IF-22 secret is written to `<run>/secrets/<NAME>` (mode 600 or an equivalent ACL), and `e2e.env` carries `<NAME>_FILE=/run/secrets/<NAME>` instead of the value. Dev is unchanged. No LLM key is involved: under D-04, WS-05.21 sets no `LLM_PORTKEY_*` or provider API key in either profile, and strict Bedrock uses the instance role, never a key.
- Modify: `tests/scripts/test_mvp_039_gen_e2e_env.py` (created by WS-05.4; adds `test_mvp_039_strict_writes_secret_files_and_name_file_vars`)
- Modify: `src/solace_testing/e2e.py` (`Stack.secret_value(name)` reads the run's secret file for harness assertions only)
- Create: `tests/e2e/test_w3_secrets_injection.py`
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 3`, `profiles: [strict]`)
- Test: `tests/e2e/test_w3_secrets_injection.py`, `tests/scripts/test_mvp_039_gen_e2e_env.py`

**Interfaces:**
- Consumes: IF-22 (names; `docker/entrypoint.sh` NAME_FILE loader; Caddy only gets `DOMAIN`, `CADDY_ADMIN_EMAIL` and `BASIC_AUTH_HASH`); the prod overlay's NAME_FILE values `/run/secrets/<NAME>` (request 3).
- Produces: the W3 secrets exit-criterion suite.

- [ ] **Step 1: Write the failing tests:**
```python
@pytest.mark.profile("strict")
def test_strict_containers_carry_no_secret_values_in_their_config(e2e_stack):
    with hop("no_secret_env"):
        for svc in MVP_SERVICES:
            env = json.loads(subprocess.run(["docker", "inspect", _container(e2e_stack, svc), "--format", "{{json .Config.Env}}"],
                                            capture_output=True, text=True, check=True).stdout)
            values = {e2e_stack.secret_value(n) for n in IF22_SECRET_NAMES if e2e_stack.secret_exists(n)}
            assert not [kv for kv in env if kv.split("=", 1)[-1] in values]
            assert any(kv.startswith("AUTH_SECRET_KEY_FILE=/run/secrets/") for kv in env) or svc not in AUTH_SECRET_SERVICES

@pytest.mark.profile("strict")
def test_caddy_holds_only_its_own_variables(e2e_stack):
    # hop caddy_env: env keys minus image defaults <= {"DOMAIN","CADDY_ADMIN_EMAIL","BASIC_AUTH_HASH","CADDY_API_SITE","CADDY_AUTH_SITE"}

@pytest.mark.profile("strict")
def test_no_secret_value_appears_in_any_log(e2e_stack):
    # hop log_secret_scan: no IF-22 secret value is a substring of e2e_stack.logs()

@pytest.mark.profile("strict")
def test_missing_secret_file_fails_startup(e2e_stack):
    # hop missing_secret_fatal: move <run>/secrets/AUTH_SECRET_KEY aside; compose restart user-service;
    # within 60 s the container is exited with a non-zero code; restore the file and restart in finally
```
  Also add `test_mvp_039_strict_writes_secret_files_and_name_file_vars` to the generator tests. It asserts the file exists, `AUTH_SECRET_KEY` is absent from `e2e.env`, and `AUTH_SECRET_KEY_FILE=/run/secrets/AUTH_SECRET_KEY` is present.
- [ ] **Step 2: Run them and confirm they fail.**
  - Run: unit command with `tests/scripts/test_mvp_039_gen_e2e_env.py`. Expected: FAIL: the generator still writes values into `e2e.env`.
  - Run: E2E command (strict, production) with `tests/e2e/test_w3_secrets_injection.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at `no_secret_env`, because secrets are plain environment values (MVP-348/349 class, owned by WS-03).
- [ ] **Step 3: Implement** the generator change and helpers. Add manifest entries for the E2E tests (`target_wave: 3`, `profiles: [strict]`) at the observed hops.
- [ ] **Step 4: Run them and confirm they pass.** Run the same commands without `SOLACE_MANIFEST_IGNORE`. Expected: the unit test passes, and the E2E tests XFAIL at their hops.
- [ ] **Step 5: Regression.** Run one process each: unit command with `tests/scripts`; E2E command (strict) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.14: Erasure event contracts (user.deleted and user.erasure.completed)
**Covers:** MVP-357
**Severity:** medium
**Wave:** 3
**Depends on:** WS-05.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: every PHI service reports completion with per-store counts)
**Real infra:** no
**Files:**
- Create: `tests/contracts/events/user__<svc>/user.deleted.json` for `<svc>` in orchestrator, safety, therapy, diagnosis, memory, personality and notification (consumer `<svc>-service`, producer `user-service`; rules: `user_id` uuid, `metadata.correlation_id` uuid, `metadata.source_service` equals `"user-service"`)
- Create: `tests/contracts/events/<svc>__user/user.erasure.completed.json` for the same seven (producer `<svc>-service`, consumer `user-service`; rules: `service_name` equals the producer, `erasure_request_id` uuid, `stores` is an object of integers)
- Modify: `tests/contracts/test_event_contract_files.py` (`test_erasure_chain_covers_every_phi_service`)
- Modify: `tests/contracts/test_contract_coverage.py` (event-pair expectations include the erasure chain)
- Modify: `tests/contracts/expected_failures.yaml` (producer-verification entries for the erasure chain, `target_wave: 3`)
- Test: `tests/contracts/test_event_contract_files.py`

**Interfaces:**
- Consumes: IF-16 flow; `UserDeletedKafkaEvent`; `UserErasureCompletedKafkaEvent(user_id, service_name, erasure_request_id, stores, completed_at)` (WS-06.11).
- Produces: the erasure contracts. Producers (WS-14 for `user.deleted`, each PHI service for completion) add `verify_event_producer` tests in their suites.

- [ ] **Step 1: Write the failing test** in `tests/contracts/test_event_contract_files.py`:
```python
PHI_SERVICES = {"orchestrator-service", "safety-service", "therapy-service", "diagnosis-service", "memory-service",
                "personality-service", "notification-service"}
def test_erasure_chain_covers_every_phi_service():
    contracts = load_event_contracts(CONTRACTS / "events")
    assert {c.consumer for c in contracts if c.event_type == "user.deleted"} == PHI_SERVICES
    assert {c.producer for c in contracts if c.event_type == "user.erasure.completed"} == PHI_SERVICES
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/contracts/test_event_contract_files.py`. Expected: FAIL: both sets are empty. No contract pins the erasure fan-out, which today has no consumer (MVP-029 class).
- [ ] **Step 3: Implement** the 14 contract files. Examples are real `to_dict()` outputs and contain ids only. Add the coverage expectations and the manifest entries.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS. The parametrized `test_event_contract_example_is_valid` cases for the new files also pass.
- [ ] **Step 5: Regression.** Run the unit command with `tests/contracts`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.15: W3 privacy acceptance suite written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 3
**Depends on:** WS-05.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:**
- D-24 (default assumed: hard delete within 120 s in the test; crisis episodes and escalations pseudonymized; audit rows kept);
- D-21 (default assumed: clinicians see only assigned patients);
- D-10 (default assumed: one HMAC chain serialized by advisory lock).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w3_privacy.py`
- Modify: `src/solace_testing/e2e.py`:
  - `reauth(stack, user) -> str` (`POST /api/v1/auth/reauth`);
  - `delete_account(stack, user) -> UUID`;
  - `seed_user_everywhere(stack, user) -> dict`: data in every store through public APIs (chat turns, memory, therapy, diagnosis, personality, one crisis episode);
  - `redis_keys(stack, pattern) -> list[str]` (`redis-cli --scan` with the per-run password, never printed);
  - `audit_verify(stack) -> dict`, which runs the WS-09 verify CLI inside a service container.
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 3`, `profiles: [strict]`)
- Test: `tests/e2e/test_w3_privacy.py`

**Interfaces:**
- Consumes:
  - IF-16 (`DELETE /api/v1/users/me` with `X-Reauth-Token`; `GET /api/v1/admin/erasure-requests/{id}`);
  - IF-14 (`audit_events`; the verify CLI, request 7);
  - IF-04 / D-21 isolation;
  - IF-05 `erase_user_events` effect on `event_outbox`/`event_dead_letter`;
  - the table list from slots 012-021.
- Produces: the W3 privacy exit-criterion suite.

- [ ] **Step 1: Write the failing tests** (`pytest.mark.profile("strict")`, `pytest.mark.timeout(1500)`):
```python
async def test_account_deletion_erases_every_store_within_120s(e2e_stack):
    # seed_user_everywhere; delete_account
    # hop erasure_complete: within 120 s, every PHI table has 0 rows for the user (PHI_TABLES constant);
    # redis_keys(*<user_id>*) == []; event_outbox/event_dead_letter rows for the user only of types user.deleted / user.erasure.completed;
    # crisis_alerts/escalations rows for the user are pseudonymized (no row with the raw user_id; a pseudonymized row exists)
    # hop erasure_reported: every PHI service published user.erasure.completed (erasure request status complete for all 7)
async def test_injected_store_failure_is_retried_to_completion(e2e_stack):
    # fault.stop("memory-service") right after delete_account; start after 60 s; hop erasure_retry: complete within 180 s
@pytest.mark.parametrize("case", ["foreign_thread", "memory_session", "therapy_plan", "diagnosis_session", "unassigned_clinician"])
async def test_isolation_matrix_denies_foreign_identifiers(e2e_stack, case):
    # hop isolation_http: 404/403 for user B (or an unassigned clinician) on user A's identifier
    # hop isolation_rls: 'SET ROLE solace_app; set_config(app.current_user_id, B)' count of A's rows == 0
async def test_audit_trail_records_auth_and_phi_access_and_detects_tampering(e2e_stack):
    # hop audit_rows: AUTH_LOGIN_SUCCEEDED, PHI_READ and ACCOUNT_DELETED rows exist for the journey
    # hop audit_verify: audit_verify(stack)["ok"] is True
    # hop audit_tamper: superuser UPDATE of one row's action, then verify reports failure at that chain_seq; restore the value
async def test_concurrent_audit_writers_produce_one_chain(e2e_stack):
    # hop audit_concurrency: drive auth, chat and notification actions in parallel;
    # chain_seq is contiguous and each previous_hash equals the prior event_hash
async def test_crisis_conversation_leaves_no_phi_in_logs_events_dlq_or_alert_email(e2e_stack):
    # marker phrase from CRISIS_PHRASES; hop phi_capture_scan: no marker, user email or display name in e2e_stack.logs(),
    # event_outbox payloads, event_dead_letter rows or any mailpit message
```
- [ ] **Step 2: Run them and confirm they fail.** Run: E2E command (strict, production) with `tests/e2e/test_w3_privacy.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at `erasure_complete` (no fan-out, MVP-029), `audit_rows` (no service writes audit, MVP-091) and other hops as observed.
- [ ] **Step 3: Implement** the helpers. Add manifest entries per observed id and hop (`target_wave: 3`, strict).
- [ ] **Step 4: Run them and confirm they pass.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with the listed tests XFAIL.
- [ ] **Step 5: Regression.** Run the E2E command (strict) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.16: W3 consent and Google sign-in journeys written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 3
**Depends on:** WS-05.15
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-23 (default assumed: required consents gate chat with 403 CONSENT_REQUIRED; revocation ends sessions). D-11 (OAuth handoff by 302 plus a one-time code).
**Real infra:** yes
**Files:**
- Create: `tests/e2e/test_w3_consent_oauth.py`
- Modify: `src/solace_testing/e2e.py`:
  - `oidc_identity(stack, sub, email, email_verified)` (`POST /__control/identity` via `exec` into `stub-oidc`);
  - `google_sign_in(stack, *, redirect_uri) -> dict`: start through the edge, follow the authorize hop inside the network, return to the callback with the state cookie, then `POST /api/v1/auth/oauth/google/exchange`.
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 3`, both profiles)
- Test: `tests/e2e/test_w3_consent_oauth.py`

**Interfaces:**
- Consumes: IF-17 OAuth endpoints and `cns` claim; IF-12 `CONSENT_REQUIRED`; the stub IdP (WS-05.12).
- Produces: the W3 consent and OAuth exit criteria.

- [ ] **Step 1: Write the failing tests:**
```python
async def test_chat_requires_the_required_consents(e2e_stack):
    # user without consents -> hop consent_gate: chat 403 {"error":{"code":"CONSENT_REQUIRED"}}; after accepting -> 200
async def test_revoking_a_required_consent_blocks_chat_and_ends_sessions(e2e_stack):
    # hop consent_revocation: after revoke, REST chat 403 and the open WS closes with 4401 within 10 s
async def test_new_verified_google_identity_signs_in_by_one_time_code(e2e_stack):
    # oidc_identity(sub=g-<hex>, email=new_email("user"), verified=True)
    # hop oauth_signin: callback 302 Location carries ?code= (never a token); exchange returns a token pair; second exchange -> 400
async def test_google_identity_never_links_a_preregistered_unverified_account(e2e_stack):
    # password account registered (unverified) with email E; google identity with the same E
    # hop oauth_no_silent_link: the sign-in does not return a token for the existing account id;
    # oauth_accounts has no row linking it (sql)
async def test_oauth_rejects_bad_state_and_unlisted_redirects(e2e_stack):
    # hop oauth_state: callback with a mismatched state -> 400; start with redirect_uri http://evil.invalid -> 400
```
- [ ] **Step 2: Run them and confirm they fail.** Run: E2E command (dev) with `tests/e2e/test_w3_consent_oauth.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at `consent_gate` (consent never enforced, MVP-100) and `oauth_signin` (no code handoff, MVP-104).
- [ ] **Step 3: Implement** the helpers. Add manifest entries (`target_wave: 3`, both profiles).
- [ ] **Step 4: Run them and confirm they pass.** Run: E2E commands (dev, then strict). Expected: exit 0 with the listed tests XFAIL.
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.17: Restore-drill driver and the restore-drill acceptance test
**Covers:** MVP-039
**Severity:** high
**Wave:** 4
**Depends on:** WS-05.16
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-09 (default assumed: wal-g base backups plus WAL archiving; RPO 15 min; RTO 4 h; keys escrowed; backups record key ids only). D-25 (decided 2026-10-19: the off-host S3 backup target needs the deployer policy attached to `MedCodex-Local`. Until then, real-infra runs are local Docker, and the drill may use the local encrypted-volume target if it says so in its report)
**Real infra:** yes
**Files:**
- Create: `scripts/test/restore_drill.py`. Run as `python -m scripts.test.restore_drill --project <p> --env-file <path> --compose-file <f>... --out <report.json>`. Besides the steps below, it copies the `backup_target` line of the backup fragment's `status` command into the report (WS-03's request for WS-03.9 and WS-03.16). When that target is `local`, it adds `target_note`, saying the drill used the local encrypted-volume target because the deployer policy was not yet attached (D-25). The steps:
  1. seed marker rows through the API and record row checksums of `users`, `memory_records`, `audit_events` and `crisis_alerts`;
  2. trigger a backup and record its time;
  3. write post-backup rows for 2 minutes;
  4. stop the stack and destroy the Postgres volume;
  5. restore with the backup fragment's restore command;
  6. start the stack and wait for `/ready` on every MVP service;
  7. compare checksums, check that post-backup rows exist up to the archive point, check that a seeded user's profile decrypts through the API, and run the audit verify.

  The JSON report carries `rpo_seconds`, `rto_seconds`, `checks{name: bool}`, `backup_target` and, for a local target, `target_note`. It is written outside the repo under the run directory.
- Create: `tests/e2e/test_w4_enterprise.py`
- Modify: `src/solace_testing/e2e.py` (`StackConfig.compose_files` includes `docker-compose.backup.yml` when present)
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 4`, strict)
- Test: `tests/e2e/test_w4_enterprise.py`

**Interfaces:**
- Consumes: the WS-03 backup fragment commands (request 6); IF-13 decryption via the API; IF-14 verify.
- Produces: `restore_drill.main(argv) -> int` and the report format above.

- [ ] **Step 1: Write the failing test** in `tests/e2e/test_w4_enterprise.py`:
```python
@pytest.mark.profile("strict")
@pytest.mark.timeout(5400)
def test_restore_drill_meets_rpo_and_rto(e2e_stack, tmp_path):
    with hop("backup_fragment"):
        assert "docker-compose.backup.yml" in e2e_stack.config.compose_files
    with hop("restore_drill"):
        report_path = e2e_stack.config.run_dir / "restore-drill.json"
        rc = restore_drill.main(["--project", e2e_stack.project, "--env-file", str(e2e_stack.config.run_dir / "e2e.env"),
                                 *sum((["--compose-file", f] for f in e2e_stack.config.compose_files), []), "--out", str(report_path)])
        report = json.loads(report_path.read_text())
        assert rc == 0 and all(report["checks"].values())
        assert report["rpo_seconds"] <= 900 and report["rto_seconds"] <= 14400
        assert report["backup_target"]                                        # D-25: the drill says which target it used
        assert report["backup_target"] != "local" or "D-25" in report["target_note"]
```
- [ ] **Step 2: Run it and confirm it fails.** Run: E2E command (strict, production) with `tests/e2e/test_w4_enterprise.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at hop `backup_fragment`: no backup exists for any PHI store (MVP-037 class, owned by WS-03).
- [ ] **Step 3: Implement** `restore_drill.py` per the Files entry. Every step logs its timestamp and step name only. Add the manifest entry (`target_wave: 4`, strict).
- [ ] **Step 4: Run it and confirm it passes.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: XFAIL at `backup_fragment`.
- [ ] **Step 5: Regression.** Run the unit command with `tests/scripts`; it gains `tests/scripts/test_mvp_039_restore_drill_args.py`, an argument-parsing test of the driver. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.18: W4 enterprise acceptance: MFA, password reset, alert firing, RS256-only and per-service keys, written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 4
**Depends on:** WS-05.17
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: TOTP plus 10 recovery codes, mandatory for CLINICIAN and ADMIN). D-03 (RS256-only user tokens; per-service RS256 service keys with kid = service name).
**Real infra:** yes
**Files:**
- Modify: `tests/e2e/test_w4_enterprise.py` (add the tests below)
- Modify: `src/solace_testing/e2e.py`:
  - `totp_now(secret) -> str` (pyotp);
  - `login(stack, email, password) -> dict`;
  - `mfa_verify(stack, mfa_token, code=None, recovery_code=None) -> dict`;
  - `mint_hs256_user_token(stack, sub)` and `mint_service_token_with_key(stack, signer_service, claimed_service)`, for negative tests only, using per-run keys from the run directory.
- Modify: `tests/e2e/expected_failures.yaml` (`target_wave: 4`, strict)
- Test: `tests/e2e/test_w4_enterprise.py`

**Interfaces:**
- Consumes: IF-17 MFA, reset and reauth endpoints; IF-04 W4 token rules; IF-18 W4 rules (`ContainerUnhealthy`, `BackupStale`) and the crisis-path alerts.
- Produces: the W4 exit-criterion suite.

- [ ] **Step 1: Write the failing tests** (`pytest.mark.profile("strict")`):
```python
async def test_mfa_enrollment_and_login_with_code_and_recovery_code(e2e_stack):
    # hop mfa_enroll: enroll/start -> secret; confirm with totp_now -> 10 recovery codes
    # hop mfa_login: login -> mfa_required; verify(code) -> tokens; verify(recovery_code) works once and is rejected on reuse
async def test_clinician_or_admin_without_mfa_is_denied_phi_admin_and_ack(e2e_stack):
    # hop mfa_enforced: a provisioned clinician without a factor gets 403 on a PHI read, an admin route and POST acknowledge
async def test_forgot_and_reset_password_is_single_use_and_expiring(e2e_stack):
    # hop password_reset: forgot -> 202; mailpit link; reset -> 200; reuse -> 400; old sessions revoked (401)
async def test_stopping_notification_fires_the_crisis_path_alert_within_5_minutes(e2e_stack):
    # hop crisis_path_alert: wait_for_alert(e2e_stack, "ServiceNotReady", timeout=300) after fault.stop("notification-service")
@pytest.mark.parametrize("alertname", ["BackupStale", "ContainerUnhealthy"])
async def test_w4_fault_alerts_fire(e2e_stack, alertname):
    # hop w4_alert: BackupStale after stopping the backup schedule with the last-success metric aged (drill hook from request 6);
    # ContainerUnhealthy after fault.pause of a service (healthcheck fails); email within the rule window plus 120 s
async def test_hs256_user_token_is_rejected(e2e_stack):
    # hop rs256_only: chat with mint_hs256_user_token -> 401
async def test_service_token_signed_with_another_services_key_is_rejected(e2e_stack):
    # hop service_key_binding: a token claiming service:notification-service signed with safety's key -> 401 on notification /send
```
- [ ] **Step 2: Run them and confirm they fail.** Run: E2E command (strict, production) with `tests/e2e/test_w4_enterprise.py` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at `mfa_enroll` (no MFA, MVP-344), `rs256_only` (HS256 accepted) and the other hops as observed.
- [ ] **Step 3: Implement** the helpers. Add manifest entries (`target_wave: 4`, strict).
- [ ] **Step 4: Run them and confirm they pass.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with the listed tests XFAIL.
- [ ] **Step 5: Regression.** Run the E2E command (strict) with `tests/e2e`. Expected: pass or XFAIL as listed.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.19: Release gate runner and the production-readiness checklist
**Covers:** MVP-039
**Severity:** high
**Wave:** 5
**Depends on:** WS-05.18, WS-03.11 and WS-03.20 (the staging stack, its `InstanceId` and `ArtifactBucketName` outputs, and request 17's permissions and host tools)
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-25 (decided 2026-10-19: AWS hosts the release gate. The gate runs on the AWS staging host with the strict production profile: the EC2 host from `infrastructure/aws-staging/solace-staging.yaml`, stack `solace-staging`, account 253025918446, us-east-1. It is reached through SSM with the existing IAM user `MedCodex-Local` (profile `medcodex-local`). If `solace-staging-deployer-policy.json` is still not attached to `MedCodex-Local` at W5 entry, or the stack does not exist, the preflight exits 3. That is a stop for the owner, with no silent fallback to local Docker. Until then every real-infra test runs on local Docker, and so does this gate's rehearsal, whose result never counts as the gate). D-31 (go-live needs the gate plus BAAs, clinical sign-off, on-call coverage and restore evidence). D-17 (decided 2026-10-19: no commits. The gate runs from `git archive <final checkpoint tree id>`, and the gate record reports the tree id only).
**Real infra:** yes
**Files:**
- Create: `scripts/test/release_gate.py`, the gate logic with subcommands `preflight`, `remote` and `run` (behaviour under Interfaces). It uses boto3 from the runtime lock (WS-01.2, D-04) for AWS calls only, its unit tests stub every AWS call, and it creates no test or contract file.
- Create: `scripts/test/release_gate.ps1` (the operator wrapper; behaviour under Interfaces)
- Create: `tests/e2e/test_release_gate.py`
- Create: `docs/release/PRR.md`. Sections:
  - on-call coverage (at least two on-call clinicians plus the fallback mailbox);
  - BAAs: the AWS BAA accepted in AWS Artifact and recorded before any real-user LLM feature is enabled; it covers Bedrock (D-04), SES (D-12) and the hosting services. Plus any SMS provider. Portkey carries no real-user traffic (D-04);
  - clinical sign-off (D-05 record in docs/CLINICAL-VALIDATION.md);
  - restore-drill evidence: report path, RPO and RTO, and the backup target the drill used (the off-host S3 target once the D-25 policy is attached, otherwise the local encrypted volume, stated as such);
  - alert routing (Alertmanager receivers tested);
  - secrets rotation (D-07: the SOPS re-encrypt-and-redeploy procedure for the environment's encrypted secrets file, the environment's age recipient and the last rotation date; never a value or a private key);
  - rollback steps;
  - gate runs (tree ids and both run results from the AWS staging host).

  Every item is a checkbox, and an unchecked item must carry `Ruling:` with the ledger reference.
- Modify: `tests/e2e/expected_failures.yaml` (every entry and shortcut removed)
- Test: `tests/e2e/test_release_gate.py`

**Interfaces:**
- Consumes:
  - every earlier suite;
  - the ledger's Ruling format (`Ruling: <decision> - <why> - <cost if wrong>`);
  - WS-01.5's `scripts/ci/build_images.py --all --tag TAG --context DIR --probe`;
  - WS-01.2's locks: `boto3`/`botocore`, and `requirements-dev.lock` for the host venv;
  - WS-03's stack `solace-staging` with outputs `InstanceId` and `ArtifactBucketName` (WS-03.20; DEPLOY.md already ships code as a `git archive` under its `artifacts/` prefix), plus the deployer-policy actions, instance-role prefixes and host tools of request 17.
- Produces: the W5 release gate.
  - `release_gate.GATE_STACK = "solace-staging"`, `GATE_REGION = "us-east-1"`, `GATE_PROFILE = "medcodex-local"`.
  - `GateClients(sts, cfn, ssm, iam, s3)` is a dataclass of boto3 clients, built by `GateClients.from_profile(profile, region)`. It uses the profile's credentials only: never a static key, never a secret.
  - `preflight(clients, *, stack=GATE_STACK) -> int` runs read-only checks in this order:
    1. `sts.get_caller_identity()`;
    2. `cfn.describe_stacks(StackName=stack)`, which reads `InstanceId` and `ArtifactBucketName`;
    3. `ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [<id>]}])`; the host must be `Online`;
    4. `iam.simulate_principal_policy(PolicySourceArn=<caller arn>, ...)` for `ssm:SendCommand` on the instance ARN and the `AWS-RunShellScript` document ARN, and for `s3:PutObject` and `s3:GetObject` under the release prefix.

    Any `ClientError`, missing output, offline host or decision other than `allowed` prints `STOP D-25: <check> <error code or decision>` (names and codes only) and returns 3. Success returns 0.
  - `archive_command(tree) -> list[str]` returns `["git", "-C", "D:\\Repo\\Contextual-Chatbot", "archive", "--format=zip", "-o", f"D:\\tmp\\mvp-exec\\release\\{tree}.zip", tree]`, built from a checkpoint tree id, never a commit (D-17).
  - `host_commands(tree, bucket, *, environment="production") -> list[str]` returns the shell lines run on the host. They:
    1. fetch `s3://<bucket>/artifacts/<tree>.zip` with the instance role (`aws s3 cp`);
    2. expand it into `/opt/solace-release/<tree>/` (`python3.12 -m zipfile -e`);
    3. create a lock-only venv there, the same way the images sync the lock (`uv venv .venv --python python3.12`, then `uv pip sync --python .venv/bin/python --require-hashes requirements-dev.lock`, with the host's pinned uv 0.8.2 from request 17);
    4. run `.venv/bin/python scripts/ci/build_images.py --all --tag gate-<tree7> --context . --probe`, the clean-checkout proof on the gate host (D-17, WS-01's R-01.3c);
    5. run `.venv/bin/python scripts/test/release_gate.py run --environment <environment> --runs 2 --out results --upload-to s3://<bucket>/gate/<tree>/results/`.

    They contain no credential, key or secret value. The stack's per-run secrets are generated on the host by `gen_e2e_env`, as in every E2E run.
  - `send_command_request(tree, instance_id, bucket, *, timeout_hours=8) -> dict` returns the `ssm.send_command` arguments:
    - `DocumentName="AWS-RunShellScript"`;
    - `InstanceIds=[instance_id]`;
    - `Parameters={"commands": host_commands(...), "executionTimeout": [str(timeout_hours * 3600)]}`;
    - `OutputS3BucketName=bucket` and `OutputS3KeyPrefix=f"gate/{tree}/ssm-output/"`, so the full output survives the 24,000-character inline limit.
  - `remote(clients, *, tree, stack=GATE_STACK, timeout_hours=8) -> int` runs `archive_command`, uploads the zip to `s3://<bucket>/artifacts/<tree>.zip`, sends the command, and polls `ssm.get_command_invocation` every 60 s until a terminal status. It then downloads `gate/<tree>/results/` (JUnit XML of both runs and `summary.json`) into `D:\tmp\mvp-exec\release\<tree>\`. It returns 0 only for status `Success` with `summary.json` showing both runs green and equal test counts, 1 otherwise.
  - `plan(*, environment, runs=2) -> list[Step]`, with `Step(run: int, suite: str, argv: list[str], env: dict[str, str])`. Each run executes every suite in its own pytest process with `--junitxml=<out>/run<k>/<suite slug>.xml`:
    1. every unit suite from the Global Constraints list;
    2. `tests/realinfra`;
    3. `tests/contracts`;
    4. `tests/clinical`;
    5. `tests/e2e` with `SOLACE_E2E_PROFILE=strict`, `SOLACE_E2E_ENVIRONMENT=<environment>`, `SOLACE_BARRIER_WAVE=5` and `SOLACE_REQUIRE_REALINFRA=1`.

    `tests/contracts` stays WS-05.7's and WS-05.10's, and `tests/clinical` stays WS-12's; they are only run here.
  - `run(*, environment, runs, out, upload_to=None) -> int` executes `plan()` and writes `summary.json` `{tree, runs: [{run, tests, failures, errors}], ok}`. It returns 1 on any failure or a test-count drop between runs.
  - The CLI: `release_gate.py preflight [--profile P]`, `remote --tree ID [--profile P] [--timeout-hours N]` and `run --environment E --runs N --out DIR [--upload-to URI]`.
  - `release_gate.ps1 -Tree <id> -Environment production [-Rehearsal]` runs `release_gate.py preflight` and then `release_gate.py remote --tree <id>` with `D:\tmp\mvp-exec\venv\Scripts\python.exe`, passing the exit code through (3 = D-25 stop). `-Rehearsal` instead runs `archive_command`, expands the archive under `D:\tmp\mvp-exec\release\<tree>\rehearsal\` and runs `release_gate.py run` there on local Docker. Its summary is labelled `REHEARSAL` and is never recorded as the gate result.

- [ ] **Step 1: Write the failing test** in `tests/e2e/test_release_gate.py`. It runs without a stack and calls no AWS endpoint. Every AWS call is stubbed:
```python
from scripts.test import release_gate

REQUIRED_SECTIONS = ["On-call coverage", "BAAs", "Clinical sign-off", "Restore-drill evidence", "Alert routing",
                     "Secrets rotation", "Rollback steps", "Gate runs"]
TREE = "0123456789abcdef0123456789abcdef01234567"   # shape of a checkpoint tree id; synthetic
def test_no_expected_failures_or_shortcuts_remain():
    for path in (REPO / "tests").rglob("expected_failures.yaml"):
        entries, shortcuts = load_manifest(path)
        assert entries == [] and shortcuts == [], f"{path.relative_to(REPO)} still lists expected failures"

def test_production_readiness_checklist_is_complete():
    text = (REPO / "docs" / "release" / "PRR.md").read_text(encoding="utf-8")
    assert all(f"## {s}" in text for s in REQUIRED_SECTIONS)
    unchecked = [l for l in text.splitlines() if l.lstrip().startswith("- [ ]")]
    assert all("Ruling:" in l for l in unchecked)

def test_release_gate_runs_every_suite_twice_in_the_strict_production_profile():
    steps = release_gate.plan(environment="production", runs=2)
    first, second = [s.suite for s in steps if s.run == 1], [s.suite for s in steps if s.run == 2]
    assert first == second and {"tests/realinfra", "tests/contracts", "tests/clinical", "tests/e2e"} <= set(first)
    e2e = next(s for s in steps if s.suite == "tests/e2e")
    assert {"SOLACE_E2E_PROFILE": "strict", "SOLACE_E2E_ENVIRONMENT": "production", "SOLACE_BARRIER_WAVE": "5",
            "SOLACE_REQUIRE_REALINFRA": "1"}.items() <= e2e.env.items()
    assert all(any(a.startswith("--junitxml=") for a in s.argv) for s in steps)

def test_release_gate_runs_on_the_staging_host_from_a_checkpoint_archive():   # D-25, D-17
    assert release_gate.archive_command(TREE) == ["git", "-C", "D:\\Repo\\Contextual-Chatbot", "archive", "--format=zip",
                                                  "-o", f"D:\\tmp\\mvp-exec\\release\\{TREE}.zip", TREE]
    req = release_gate.send_command_request(TREE, "i-0123456789abcdef0", "solace-staging-artifacts-example", timeout_hours=8)
    assert req["DocumentName"] == "AWS-RunShellScript" and req["InstanceIds"] == ["i-0123456789abcdef0"]
    assert req["Parameters"]["executionTimeout"] == ["28800"]
    assert (req["OutputS3BucketName"], req["OutputS3KeyPrefix"]) == ("solace-staging-artifacts-example", f"gate/{TREE}/ssm-output/")
    cmds = "\n".join(req["Parameters"]["commands"])
    assert "scripts/ci/build_images.py --all" in cmds and "--probe" in cmds
    assert "scripts/test/release_gate.py run --environment production --runs 2" in cmds
    assert "git " not in cmds and not re.search(r"(?i)aws_secret_access_key|aws_access_key_id|SOPS_AGE_KEY", cmds)

def test_release_gate_preflight_stops_without_the_deployer_policy(capsys):   # D-25: a stop, never a local fallback
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.stub import Stubber
    cfg = Config(signature_version=UNSIGNED)
    clients = release_gate.GateClients(**{n: boto3.client(s, region_name="us-east-1", config=cfg) for n, s in
                                          (("sts", "sts"), ("cfn", "cloudformation"), ("ssm", "ssm"), ("iam", "iam"), ("s3", "s3"))})
    # every client is stubbed, so an unexpected call raises UnStubbedResponseError instead of reaching AWS
    with Stubber(clients.sts) as sts, Stubber(clients.cfn) as cfn, Stubber(clients.ssm), Stubber(clients.iam), Stubber(clients.s3):
        sts.add_response("get_caller_identity", {"UserId": "AIDEXAMPLE", "Account": "253025918446",
                                                 "Arn": "arn:aws:iam::253025918446:user/MedCodex-Local"})
        cfn.add_client_error("describe_stacks", service_error_code="AccessDenied", http_status_code=403,
                             expected_params={"StackName": "solace-staging"})
        assert release_gate.preflight(clients) == 3        # nothing after the stop: no SendCommand, no upload, no local run
        sts.assert_no_pending_responses(); cfn.assert_no_pending_responses()
    out = capsys.readouterr().out
    assert "STOP D-25" in out and "AccessDenied" in out

def test_release_gate_wrapper_preflights_first_and_labels_rehearsals():
    script = (REPO / "scripts" / "test" / "release_gate.ps1").read_text(encoding="utf-8")
    assert script.index("release_gate.py preflight") < script.index("release_gate.py remote")
    assert "REHEARSAL" in script and "D:\\tmp\\mvp-exec\\venv\\Scripts\\python.exe" in script
```
- [ ] **Step 2: Run it and confirm it fails.** Run: unit command with `tests/e2e/test_release_gate.py`. Expected: FAIL at collection with `ModuleNotFoundError: No module named 'scripts.test.release_gate'`. With an empty stub module, the run shows:
  - the plan, SSM and preflight tests fail with `AttributeError`;
  - the PRR test fails because `docs/release/PRR.md` does not exist;
  - the wrapper test fails because `scripts/test/release_gate.ps1` does not exist;
  - any manifest entries left from W4 are listed by path.

  No gate exists that could run on the staging host (D-25).
- [ ] **Step 3: Implement** `release_gate.py`, the wrapper and the checklist. Remove the manifest entries whose journeys passed at their barriers; a remaining entry needs a ledger Ruling and blocks the gate. The gate is run by the controller, not by this task. The rules:
  - No AWS call happens at import time.
  - Clients come only from `GateClients.from_profile`.
  - Printed output is check names, error codes, statuses and counts, never a credential, an ARN of a secret or a value.
- [ ] **Step 4: Run it and confirm it passes.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run, in this order:
  1. Rehearsal on local Docker (D-25: until the policy is attached, real-infra runs are local): `& scripts\test\release_gate.ps1 -Tree <checkpoint tree> -Environment production -Rehearsal`. Expected: exit 0 with both runs green, recorded in the ledger as `REHEARSAL`, never as the gate.
  2. At W5 entry the controller runs `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\test\release_gate.py preflight --profile medcodex-local`. Expected: exit 0. Exit 3 is a stop. The controller reports the printed check to the owner, who attaches `solace-staging-deployer-policy.json` to `MedCodex-Local` (option 2 of AWS-STAGING-IAM-SETUP.pdf) and provisions the stack per `infrastructure/aws-staging/DEPLOY.md`. The gate waits and never falls back to local Docker.
  3. `& scripts\test\release_gate.ps1 -Tree <final checkpoint tree> -Environment production`, run by the controller at the W5 gate. Expected: exit 0 with both runs green on the AWS staging host. PRR "Gate runs" records the tree id and both run results only.
  4. The E2E command (strict) with `tests/e2e`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.20: tests/integration stewardship: the LLM timeout test follows IF-15, and code-owner edits are sanctioned
**Covers:** none (supporting edit for MVP-139, owned by WS-09; request CR-6)
**Severity:** medium
**Wave:** 1
**Depends on:** WS-09.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `tests/integration/test_sprint7_portkey_fallback.py`, class `TestLLMRequestTimeout` only. `test_generate_returns_empty_on_timeout_not_hang` is replaced by `test_generate_raises_unavailable_on_timeout`; `test_execute_completion_times_out_a_stalled_request` and the other classes are unchanged.
- Test: `tests/integration/test_sprint7_portkey_fallback.py`

**Interfaces:**
- Consumes: IF-15 `LLMUnavailableError` with `reason == "timeout"` (WS-09.3); `UnifiedLLMClient(settings, client_factory=...)` (WS-09.2).
- Produces:
  - Satisfies WS-09's request CR-6 for WS-09.3. The node `TestLLMRequestTimeout::test_generate_returns_empty_on_timeout_not_hang`, which WS-09.3's Step 5 records as failing, is retired in the same wave.
  - Steward sanction for WS-16's request R12 (WS-16.5, WS-16.11, both W2). WS-16.11 rewrites `tests/integration/test_diagnosis_vignette.py`, and WS-16.5 edits only class `TestDiagnosisBridgeSafetyFlags` in `tests/integration/test_event_flow.py`. Both edits are made by the owner of the code under test, per the `tests/integration` shared-file protocol. WS-05 makes no edit to either file. The full register of sanctioned edits is under "Shared-directory stewardship" below.

- [ ] **Step 1: Write the replacement test** in `tests/integration/test_sprint7_portkey_fallback.py` (class `TestLLMRequestTimeout`):
```python
    @pytest.mark.asyncio
    async def test_generate_raises_unavailable_on_timeout(self) -> None:
        """REV-31 + IF-15: a stalled provider surfaces as LLMUnavailableError(reason="timeout") within the bound, never '' or a hang."""
        from services.shared.infrastructure.llm_client import LLMUnavailableError

        async def _hang(**kwargs):  # noqa: ANN003
            await asyncio.sleep(30)

        fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_hang)))
        settings = LLMClientSettings(_env_file=None, portkey_api_key=SecretStr("test-portkey-key"), timeout_seconds=1)
        client = UnifiedLLMClient(settings, client_factory=lambda **_: fake)
        await client.initialize()
        loop = asyncio.get_running_loop(); t0 = loop.time()
        with pytest.raises(LLMUnavailableError) as exc:
            await client.generate(system_prompt="s", user_message="hi", service_name="test")
        assert exc.value.reason == "timeout" and loop.time() - t0 < 5.0
```
- [ ] **Step 2: Confirm the red.** Before replacing the old test, run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest "tests/integration/test_sprint7_portkey_fallback.py::TestLLMRequestTimeout::test_generate_returns_empty_on_timeout_not_hang" -o "addopts=" -p no:logging -q`. Expected: FAIL with `LLMUnavailableError`, because WS-09.3 made `generate()` raise instead of returning `""`. The old assertion `result == ""` now pins the retired sentinel behaviour (MVP-139). On the pre-WS-09.3 tree, the new test fails with `ImportError: cannot import name 'LLMUnavailableError'`.
- [ ] **Step 3: Implement.** Delete `test_generate_returns_empty_on_timeout_not_hang` and add the Step 1 test in its place. This is a test-only change.
- [ ] **Step 4: Run it and confirm it passes.** Run: the unit command with `"tests/integration/test_sprint7_portkey_fallback.py::TestLLMRequestTimeout"`. Expected: PASS (2 tests).
- [ ] **Step 5: Regression.** Run the unit command with `tests/integration`. Expected: all pass. WS-09.3's remaining recorded failures are therapy-service nodes (its CR-5), outside this directory.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.20:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.21: W1 overlay env sync: the D-04 Bedrock wiring to the fake LLM and test-traffic auth rate limits
**Covers:** MVP-039
**Severity:** high
**Wave:** 1
**Depends on:** WS-05.4
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-04 (decided 2026-10-19: real-user LLM traffic goes to Anthropic models on Bedrock, and tests use only the fake. This task follows WS-09's CR-3 as revised for D-04. The dev profile reaches WS-05.1's Converse route with `LLM_PROVIDER=bedrock`, `BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, the synthetic `BEDROCK_MODEL_ID=fake-llm-model` (never a real AWS model id) and `LLM_PHI_EGRESS_APPROVED=true`; it carries no AWS credential, because WS-09.18 sends unsigned requests when the override is set. Strict refuses the override (WS-09.18), so the strict profile sets `LLM_PROVIDER=bedrock` and `LLM_PHI_EGRESS_APPROVED=false` and runs template replies, exactly as a strict deployment does until the AWS BAA is recorded. Neither profile sets `LLM_PORTKEY_*` or a provider API key). D-26 (default assumed: the app-level limiter is the control; the overrides are a test-overlay value only).
**Real infra:** no (it renders `docker compose config`; no container starts)
**Files:**
- Modify: `docker-compose.test.yml`.
  - On orchestrator, safety, therapy, diagnosis and personality: `LLM_PROVIDER=bedrock` and `LLM_PHI_EGRESS_APPROVED=false` replace WS-05.4's `LLM_PORTKEY_GATEWAY_URL`, `LLM_PHI_EGRESS_APPROVED=true` and `LLM_ANTHROPIC_API_KEY=""`. `LLM_ENABLE_CACHING=false` stays.
  - On every app service: `RATE_LIMIT_OVERRIDES_JSON: '{"auth-login": 500, "auth-register": 500, "auth-refresh": 500, "auth-verify": 500}'`.
- Create: `infrastructure/test/compose/llm-dev.yml`, the dev-only fragment. On the same five services it sets `BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, `BEDROCK_MODEL_ID=fake-llm-model` and `LLM_PHI_EGRESS_APPROVED=true`. It is never part of a strict render, so no strict stack carries the override.
- Modify: `src/solace_testing/e2e.py`.
  - `StackConfig.compose_files` appends `infrastructure/test/compose/llm-dev.yml` after `docker-compose.test.yml` in the dev profile only.
  - `OVERLAY_ALLOWED_KEYS` gains `LLM_PROVIDER` and `RATE_LIMIT_OVERRIDES_JSON` and loses `LLM_PORTKEY_GATEWAY_URL` and `LLM_ANTHROPIC_API_KEY`.
- Create: `tests/solace_testing/test_mvp_039_overlay_w1_env.py`
- Test: `tests/solace_testing/test_mvp_039_overlay_w1_env.py`

**Interfaces:**
- Consumes:
  - IF-15 as amended by D-04: `LLM_PROVIDER`, `BEDROCK_ENDPOINT_URL`, `BEDROCK_MODEL_ID` and `LLM_PHI_EGRESS_APPROVED`, with WS-09.18's unsigned override and its strict refusal, and WS-09.4's approval rule;
  - the registry entries WS-09 requested from WS-03 (CR-14, CR-4), so `tests/deploy/test_env_contract.py` accepts the overlay;
  - WS-05.1's Converse route;
  - IF-04 `RATE_LIMIT_OVERRIDES_JSON` and the route names `auth-login`, `auth-register`, `auth-refresh`, `auth-verify` (WS-07.7).
- Produces:
  - Satisfies WS-09's CR-3 for WS-09.4 and WS-09.18: strict services boot with the LLM off, and dev services reach the fake as Bedrock.
  - Satisfies WS-07's request for WS-07.7: `make_user` loops never share the 10/min register and 20/min login buckets that all test traffic hits behind Caddy.
  - No profile carries an LLM key, so WS-05.13 (W3) has none to move to `_FILE`.

- [ ] **Step 1: Write the failing tests.** In `tests/solace_testing/test_mvp_039_overlay_w1_env.py`:
```python
LLM_SERVICES = ("orchestrator-service", "safety-service", "therapy-service", "diagnosis-service", "personality-service")
NO_LLM_KEYS = ("LLM_PORTKEY_API_KEY", "LLM_PORTKEY_GATEWAY_URL", "LLM_ANTHROPIC_API_KEY",
               "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE")

def _render(tmp_path, profile, monkeypatch):
    monkeypatch.setenv("SOLACE_E2E_PROFILE", profile)
    run = tmp_path / profile
    subprocess.run([sys.executable, "-m", "scripts.test.gen_e2e_env", "--profile", profile, "--environment",
                    "development" if profile == "dev" else "production", "--out", str(run)], cwd=REPO, check=True, capture_output=True)
    args = [a for f in StackConfig.from_env().compose_files for a in ("-f", f)]     # exactly the files the fixture uses
    out = subprocess.run(["docker", "compose", "--env-file", str(run / "e2e.env"), *args, "config", "--format", "json"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout      # never printed: it holds secrets
    return json.loads(out)["services"]

def test_d04_dev_stacks_reach_the_fake_llm_as_bedrock(tmp_path, monkeypatch):
    services = _render(tmp_path, "dev", monkeypatch)
    for svc in LLM_SERVICES:
        env = services[svc]["environment"]
        assert (env["LLM_PROVIDER"], env["BEDROCK_ENDPOINT_URL"], env["BEDROCK_MODEL_ID"], env["LLM_PHI_EGRESS_APPROVED"]) == (
            "bedrock", "http://fake-llm:8080", "fake-llm-model", "true"), svc
        assert not any(env.get(k) for k in NO_LLM_KEYS), svc

def test_d04_strict_stacks_run_bedrock_with_the_llm_off_and_no_override(tmp_path, monkeypatch):
    services = _render(tmp_path, "strict", monkeypatch)
    for svc in LLM_SERVICES:
        env = services[svc]["environment"]
        assert env["LLM_PROVIDER"] == "bedrock" and env["LLM_PHI_EGRESS_APPROVED"] == "false", svc
        assert not env.get("BEDROCK_ENDPOINT_URL") and not any(env.get(k) for k in NO_LLM_KEYS), svc

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_mvp_039_auth_buckets_are_raised_for_test_traffic(tmp_path, profile, monkeypatch):
    for svc, spec in _render(tmp_path, profile, monkeypatch).items():
        if svc in MVP_SERVICES:
            overrides = json.loads(spec["environment"]["RATE_LIMIT_OVERRIDES_JSON"])
            assert all(overrides[r] >= 500 for r in ("auth-login", "auth-register", "auth-refresh", "auth-verify"))
```
- [ ] **Step 2: Run them and confirm they fail.** Run: unit command with `tests/solace_testing/test_mvp_039_overlay_w1_env.py`. Expected: FAIL.
  - Both D-04 tests raise `KeyError: 'LLM_PROVIDER'`, because WS-05.4's overlay wires only the Portkey gateway. The dev test also lacks `BEDROCK_ENDPOINT_URL`, since no dev-only fragment exists.
  - `RATE_LIMIT_OVERRIDES_JSON` is missing (`KeyError`).
  - Once WS-09.18 lands, a strict service without `LLM_PROVIDER=bedrock` fails startup by design (the default is `portkey`). Once WS-07.7 is wired, the eleventh `make_user` in a minute gets 429. Either way every W1 acceptance suite would go red for a harness reason (MVP-039).
- [ ] **Step 3: Implement** the overlay entries, the dev-only fragment, `compose_files` and the allowed keys. The fragment holds no secret and no credential.
- [ ] **Step 4: Run them and confirm they pass.** Run the same command. Expected: PASS.
- [ ] **Step 5: Regression.** Run one process each:
  - unit command with `tests/scripts`;
  - unit command with `tests/solace_testing`;
  - E2E command (strict, production) with `tests/e2e/test_w0_boot_smoke.py` and `tests/e2e/test_mvp_039_stack_selftest.py`;
  - E2E command (dev) with `tests/e2e/test_mvp_039_stack_selftest.py`.

  Expected: all pass, the overlay-allowed-keys self-test included.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.21:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-05.22: Deployed-configuration phrase-set recall E2E written red
**Covers:** MVP-039
**Severity:** high
**Wave:** 1
**Depends on:** WS-05.8, WS-05.21, WS-12.3
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional lexicon v1 and its 200+ phrase set; the final phrase set and floors are gated on clinical sign-off). The test never lowers a floor: each positive must reach at least its `min_level`.
**Real infra:** yes
**Files:**
- Create: `tests/e2e/phrase_recall/conftest.py`. It overrides `e2e_stack` with a session fixture on a separate compose project: `cfg = StackConfig.from_env()`, then `start_stack(dataclasses.replace(cfg, lane=cfg.lane + "p", project=f"solace-e2e-{cfg.lane}p", run_dir=cfg.run_dir.with_name(f"solace-e2e-{cfg.lane}p")))`, with `stop_stack` at teardown. Its 200+ crisis episodes, re-pages included, therefore never reach the shared roster of the main lane's suites. This keeps within the two-concurrent-stacks limit.
- Create: `tests/e2e/phrase_recall/test_w1_phrase_set_deployed.py`
- Modify: `tests/e2e/expected_failures.yaml` (entry `target_wave: 1`, both profiles)
- Test: `tests/e2e/phrase_recall/test_w1_phrase_set_deployed.py`

**Interfaces:**
- Consumes: IF-07 `solace_clinical.phrases.load_phrase_set() -> PhraseSet` (`positive: tuple[PositivePhrase(text, min_level, categories), ...]`) and `level_rank` (WS-12.2, WS-12.3); IF-12 REST `crisis` object; IF-19 `make_user` and `Stack.edge`; the manifest `hop()`.
- Produces: the end-to-end recall check for review focus 1. Every clinician-reviewed positive, including the misspelled, slang, leetspeak, code-switched, emoji, quoted and earlier-clause-negation variants that WS-12 keeps in the phrase set, is screened by the deployed images and configuration, not only by in-process unit tests. A lexicon file missing from an image, or an env value that changes a floor, fails here.

- [ ] **Step 1: Write the failing test** in `tests/e2e/phrase_recall/test_w1_phrase_set_deployed.py`:
```python
pytestmark = [pytest.mark.e2e, pytest.mark.integration, pytest.mark.timeout(2400)]

async def test_every_phrase_set_positive_is_detected_by_the_deployed_stack(e2e_stack):
    with hop("phrase_set"):
        positives = load_phrase_set().positive
        assert len(positives) >= 200
    sem, misses = asyncio.Semaphore(4), []

    async def one(i, phrase):
        async with sem:
            user = await make_user(e2e_stack)                 # a fresh user per phrase: no risk history carries over
            r = await e2e_stack.edge("POST", "/api/v1/orchestrator/chat", token=user.access_token, json={"message": phrase.text})
            crisis = r.json().get("crisis") if r.status_code == 200 else None
            if crisis is None or level_rank(CrisisLevel(crisis["level"])) < level_rank(phrase.min_level):
                misses.append((i, phrase.min_level.value, crisis["level"] if crisis else None))   # index only, never the text

    with hop("deployed_phrase_recall"):
        await asyncio.gather(*(one(i, p) for i, p in enumerate(positives)))
        assert misses == [], f"{len(misses)} positives below their floor on the deployed stack: {sorted(misses)[:20]}"
```
- [ ] **Step 2: Run it and confirm it fails.** Run: E2E command (dev) with `tests/e2e/phrase_recall` and `SOLACE_MANIFEST_IGNORE=1`. Expected: FAIL at hop `deployed_phrase_recall` with every positive listed as a miss. The response carries no IF-12 `crisis` object before W1 phase B, so the deployed stack has never been checked against the signed-off phrase set (MVP-039).
- [ ] **Step 3: Implement** the conftest and the test. Add the manifest entry (hop `deployed_phrase_recall`, `target_wave: 1`, both profiles, reason naming WS-10/WS-11 for the crisis object and WS-12 for recall).
- [ ] **Step 4: Run it and confirm it passes.** Run the same command without `SOLACE_MANIFEST_IGNORE`. Expected: exit 0 with the test XFAIL at `deployed_phrase_recall`. At the W1 barrier it must pass in both profiles. A miss is a clinical-safety finding for WS-12 and is never fixed by editing the phrase set from this lane.
- [ ] **Step 5: Regression.** Run the E2E command (dev) with `tests/e2e`. Expected: pass or XFAIL as listed. The `phrase_recall` directory builds its own stack.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-05.22:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| # | Owner | Exact change | Files | Needed by |
|---|---|---|---|---|
| 1 | WS-01 | Register markers `realinfra`, `e2e`, `contract`, `clinical`, `profile`. Pin `testcontainers[postgres,redis,kafka]`, aiokafka, websockets, jsonschema, bcrypt, pyotp, PyYAML and httpx in `requirements-dev.in`/`.lock`. Per D-20 (decided 2026-10-19) they are installed into the dedicated venv `D:\tmp\mvp-exec\venv`, never into anaconda: missing ones by the controller at W0 entry, then by WS-01.2's re-sync to the dev lock. They must also work in the lock-only venv `D:\tmp\mvp-exec\venv-clean` used at the W0 barrier. Makefile targets `test-realinfra` and `test-e2e PROFILE=dev\|strict` call the commands above. A CI e2e job runs on demand or nightly. | `pytest.ini`, `requirements-dev.in`, `requirements-dev.lock`, `Makefile`, `.github/workflows/ci.yml` | WS-05.2, WS-05.4 |
| 2 | WS-02 | init-db creates `solace_migrator` and `solace_app` from the env vars `POSTGRES_MIGRATOR_PASSWORD` and `POSTGRES_APP_PASSWORD` (the postgres container's own `POSTGRES_PASSWORD` stays the image superuser), with the single database `solace`. `python -m scripts.db.migrate upgrade\|downgrade <rev>` reads `DATABASE_URL`. `SCHEMA_HEAD` lives in `solace_infrastructure.database.schema_check`. `python -m scripts.db.gen_certs --out <dir>` writes `postgres-ca.crt`, `server.crt` and `server.key`. A public `init_connection(conn)` codec helper in `connection_manager`. | `infrastructure/init-db/*`, `scripts/db/migrate.py`, `scripts/db/gen_certs.py`, `src/solace_infrastructure/database/schema_check.py`, `src/solace_infrastructure/database/connection_manager.py` | WS-05.2, WS-05.4 |
| 3 | WS-03 | Define `caddy` in `docker-compose.yml`, so the edge exists in both profiles, using site addresses `{$CADDY_API_SITE}`/`{$CADDY_AUTH_SITE}`; the prod overlay adds TLS ports and limits. In the prod overlay, declare every variable under `environment` (interpolated) with no `env_file: .env.prod`. Bind-mount `${SOLACE_SECRETS_DIR}` read-only at `/run/secrets` in postgres and the app services: TLS files from W0, all IF-22 secrets from W3. Every NAME_FILE value is `/run/secrets/<NAME>`. Add a registry entry for `SOLACE_SECRETS_DIR`. App container ports stay as in `SERVICE_PORTS`. | `docker-compose.yml`, `docker-compose.prod.yml`, `Caddyfile`, `deploy/env/env-contract.yaml` | WS-05.4, WS-05.13 |
| 4 | WS-04 | `GET /live` answers 200 `{"status":"alive"}` even before the lifespan has run. Alertmanager in `docker-compose.observability.yml` takes `ALERTMANAGER_SMTP_SMARTHOST`, `ALERTMANAGER_EMAIL_TO` and `ALERTMANAGER_EMAIL_FROM` from env, so the test overlay points it at mailpit. Each rule's alert name appears in the email subject. | `src/solace_infrastructure/runtime/health.py`, `docker-compose.observability.yml`, `infrastructure/alertmanager/alertmanager.yml` | WS-05.5, WS-05.9 |
| 5 | WS-14 | `scripts.provision_clinician` runs in the user-service image and prints exactly two lines, `user_id=<uuid>` and `set_password_url=<url>`. It supports `--assign USER_ID` and `--role CLINICIAN\|ADMIN` per IF-08. | `scripts/provision_clinician.py` | WS-05.8 |
| 6 | WS-03 | The backup fragment exposes `backup-now`, `restore-latest` and `status` commands (exit codes plus ISO timestamps on stdout) through `docker compose ... run --rm backup <cmd>`, and a drill hook that ages the last-success metric for the `BackupStale` fault test. | `docker-compose.backup.yml`, `deploy/backup/*` | WS-05.17, WS-05.18 |
| 7 | WS-09 | Add `python -m solace_security.audit_runtime verify [--start-seq N]`, runnable in a service container: it exits 0 when the chain verifies and 1 otherwise, printing `{"ok": bool, "first_bad_seq": int \| null}`. | `src/solace_security/audit_runtime.py` | WS-05.15 |
| 8 | WS-12 | `tests/clinical/crisis_phrases_v1.yaml` includes the four `CRISIS_PHRASES` texts as positives (min_level HIGH) and the two `BENIGN_PHRASES` texts as negatives, so the E2E phrases are clinically reviewed. | `tests/clinical/crisis_phrases_v1.yaml` | WS-05.8 |
| 9 | WS-10, WS-12, WS-15, WS-16 | Each crisis producer adds `verify_event_producer(contract, <its real event>)` tests against `tests/contracts/events/<producer>__notification/` in its own suite (WS-10 and WS-12 in W1; WS-15 and WS-16 in W2). | `services/orchestrator_service/tests/test_mvp_357_event_contracts.py`, `services/safety_service/tests/test_mvp_357_event_contracts.py`, `services/therapy_service/tests/test_mvp_357_event_contracts.py`, `services/diagnosis_service/tests/test_mvp_357_event_contracts.py` | WS-05.10 |
| 10 | WS-10, WS-11, WS-13 | Consumers write their REST contract directories `tests/contracts/orchestrator__safety/` (WS-10.16), `orchestrator__{therapy,diagnosis,personality,memory}/` (WS-11) and `notification__user/` (WS-13.5). `therapy__memory/` is written by WS-05.10 at WS-15's request. Providers (WS-12, WS-15, WS-16, WS-17, WS-18, WS-14, WS-11) call `verify_provider` against them in their suites (W2). | `tests/contracts/<consumer>__<provider>/*.json`, provider suites | WS-05.10 |
| 11 | WS-06 | IF-05 metric names, consumer group names and event tables (frozen). `UserErasureCompletedKafkaEvent` lands in W3 phase A before WS-05.14. | `src/solace_events/schemas.py`, `src/solace_events/metrics.py` | WS-05.9, WS-05.14 |
| 12 | WS-11 | WS-15's request R5 for `tests/contracts/orchestrator__therapy/` is redirected here, because WS-11.9 writes that consumer directory (`session_start.json`, `message.json`). The change: use WS-15's fixed example ids (user `00000000-0000-4000-a000-000000000001`, therapy session `00000000-0000-4000-b000-000000000001`, plan `00000000-0000-4000-c000-000000000001`, turn `00000000-0000-4000-e000-000000000001`); `session_start` responds 201 (its `treatment_plan_id` may differ from the request); `message` responds 200 with `risk_level`, `risk_categories`, `degraded` and `safety_alerts` as category codes; `headers_required` includes `Authorization` and `X-Turn-ID`; and add the error interactions 404 `SESSION_NOT_FOUND`, 403 `USER_ID_MISMATCH` and 409 `TREATMENT_PLAN_NOT_FOUND` (via `detail.code`) as separate files. WS-15.11 then depends on WS-11.9, and WS-15.10 on WS-05.10 (`therapy__memory`). | `tests/contracts/orchestrator__therapy/*.json` | WS-15.10, WS-15.11 |
| 13 | WS-11 | IF-20 call shapes (reply to WS-11's request): `assert_request_matches` accepts the raw `json.loads(...)` dict and an `httpx.Request` (WS-05.7). `verify_provider` is synchronous and safe inside a running loop, so WS-11.15 calls it without `await`. `tests/contracts/client__orchestrator/` holds WS-05.10's `chat.post.json`, `chat.foreign-thread.post.json` and `history.get.json` (W2 phase A, with WS-11's seed ids). WS-11.15 verifies those and creates no file in that directory, so there is never a second `chat.json`/`history.json` for the same routes; an interaction they lack (for example `/batch`) comes back to WS-05 as a request. Accepted in WS-11.15. | `services/orchestrator_service/tests/test_mvp_072_client_contract.py` | WS-11.15 |
| 14 | WS-03 | Registry entry `RATE_LIMIT_OVERRIDES_JSON` (services: every rate-limited service; `secret: false`; `required_in_strict: false`; `dev_default: null`; notes "set only by docker-compose.test.yml"), so `tests/deploy/test_env_contract.py` accepts WS-05.21's overlay value. A strict-environment guard that rejects it outside the test overlay is WS-07's open follow-up. | `deploy/env/env-contract.yaml` | WS-05.21 |
| 15 | WS-11 | The versioned-provider-prefix check of the orchestrator's client module leaves WS-05.7's alignment test (which now parses only the agent modules) and moves to the module's owner: WS-11.2, which already modifies the module in W1, adds `test_orchestrator_clients_use_versioned_provider_prefixes` (AST: every `/api/...` string constant in the module starts with `/api/v1/<provider>/`) and keeps it green through WS-11.8 and WS-11.13. WS-05.10's coverage guard checks that the lock exists. WS-05.7 cannot wait on WS-11.2 (WS-11.2 depends on WS-10.3, which depends on WS-05.7), so WS-05.10 is the dependent task. | `services/orchestrator_service/src/infrastructure/clients.py`, `services/orchestrator_service/tests/test_mvp_020_agent_service_urls.py` | WS-05.10 |
| 16 | WS-17 | Memory, the IF-11 provider, publishes the canonical `therapy__memory` /context examples that WS-05.10 copies into `tests/contracts/therapy__memory/context.json`, instead of WS-05.10 deriving them from memory's source: module constants `THERAPY_CONTEXT_REQUEST_EXAMPLE` (`user_id`, `session_id`, `current_message` with the fixed synthetic ids) and `THERAPY_CONTEXT_RESPONSE_EXAMPLE`, set as the `examples` of `ContextAssemblyRequest`/`ContextAssemblyResponse` JSON schemas; `user_id` stays required (R-17-04). Planned as WS-17.19 (W0, memory lane). | `services/memory_service/src/schemas.py` | WS-05.10 |
| 17 | WS-03 | D-25 prerequisites for WS-05.19's automated gate on the staging host, which reaches it with `ssm send-command` (the runbook's `aws ssm start-session` stays the operator path). (a) Stack `solace-staging` keeps its `InstanceId` output, and WS-03.20's `ArtifactBucketName` holds the gate's archive at `artifacts/<tree>.zip` and its results under `gate/<tree>/`. (b) `solace-staging-deployer-policy.json` lets `MedCodex-Local` call `cloudformation:DescribeStacks` on the stack, `ssm:DescribeInstanceInformation`, `ssm:SendCommand` on the instance and the `AWS-RunShellScript` document, `ssm:GetCommandInvocation`, `s3:PutObject` on `artifacts/*`, `s3:GetObject`/`s3:ListBucket` on `gate/*`, and the KMS data-key actions on the bucket's key. The preflight's `iam:SimulatePrincipalPolicy` is the read-only call the 2026-10-19 permission check already used. (c) The instance role reads `artifacts/*` and writes `gate/*/results/*` and `gate/*/ssm-output/*`. (d) The host provides Docker with the compose plugin, `python3.12`, the AWS CLI and uv 0.8.2 (pinned, checksum-verified), plus disk for two image sets and two stacks. | `infrastructure/aws-staging/solace-staging.yaml`, `infrastructure/aws-staging/solace-staging-deployer-policy.json`, `infrastructure/aws-staging/DEPLOY.md` | WS-05.19 |
| 18 | WS-12 | D-29 names WS-05 needs, so it does not guess them. (a) Under per-locale versioning, the accessor for the en-US resources version that WS-05.6's `_assert_resources` compares with `crisis["resources_version"]`: either `RESOURCES_VERSION` keeps meaning the en-US version, or WS-12 names the per-locale function. (b) The exact name and signature of the unsupported-locale fallback and its metric, and the chosen locale contract (IF-06/IF-12, for example a `loc` claim defaulting to `en-US`). WS-05 can then add, in a later wave, an E2E hop asserting that a user with an unsupported locale gets the non-empty fallback, never an error. | `src/solace_clinical/*` | WS-05.6 |
| 19 | WS-03 | Reconciliation for WS-03's W1 row to WS-05 (for WS-03.19). That row asks for `BEDROCK_ENDPOINT_URL` in both profiles with an allow-listed strict difference in `test_env_contract.py`. WS-05.21 follows WS-09's CR-3 instead, because WS-09.18 raises `LLMConfigurationError` for a non-empty `BEDROCK_ENDPOINT_URL` in strict. The strict + test render therefore carries `LLM_PROVIDER=bedrock` and `LLM_PHI_EGRESS_APPROVED=false` with no endpoint override, and needs no allow-listed strict difference. The dev + test render adds the dev-only fragment `infrastructure/test/compose/llm-dev.yml` (`BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, `BEDROCK_MODEL_ID=fake-llm-model`, `LLM_PHI_EGRESS_APPROVED=true`). `test_env_contract.py` should render the dev profile with that fragment, or accept these keys for development and test only. | `tests/deploy/test_env_contract.py` | WS-05.21 |

## New interfaces

- **IF-19 additions:**
  - `PgHandle.superuser_fetch`, `PgHandle.create_database`;
  - `KafkaHandle.committed_offset`, `pause`, `unpause` (and `produce` accepts bytes);
  - `SmtpSink.messages`/`clear`, `MailMessage`;
  - `FakeLLMHandle` (whose `.base_url` is the IF-19 value);
  - `require_docker()`, `IMAGE_REFS`, `INIT_DB_ENV`.
- **E2E additions:**
  - `Stack.edge_auth_url`, `call`, `sql`, `compose`, `env_value`, `secret_value`, `edge()`;
  - `pg_superuser_dsn` is `None`, because Postgres is never published; use `sql()`;
  - `StackConfig` with env vars `SOLACE_E2E_LANE`, `SOLACE_E2E_REUSE` and `SOLACE_E2E_RUN_DIR`;
  - the `profile` marker (deselect, not skip);
  - `SERVICE_PORTS`, `MVP_SERVICES`, `new_email`, `OVERLAY_ALLOWED_KEYS`, and the helper functions listed per task.
- **Manifest:**
  - per-directory `expected_failures.yaml`;
  - entry fields `reason` and `profiles`;
  - `shortcuts`;
  - env vars `SOLACE_BARRIER_WAVE` and `SOLACE_MANIFEST_IGNORE`;
  - `hop()`, `HopFailure`, `shortcut_allowed()`, `evaluate()`, `load_manifest()`.
- **Fake LLM:** mode `reply` with `content`; `POST /__control/reset`; `make_server()`; the Bedrock Converse route `POST /model/{model_id}/converse` with the same scenarios, the Converse body and `x-amzn-ErrorType` errors, ignoring SigV4 (D-04).
- **E2E LLM wiring (D-04, from W1, WS-05.21 per WS-09's CR-3):**
  - Every stack sets `LLM_PROVIDER=bedrock` and carries no AWS credential or provider API key.
  - The dev profile adds the fragment `infrastructure/test/compose/llm-dev.yml` (`BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, the synthetic `BEDROCK_MODEL_ID=fake-llm-model`, `LLM_PHI_EGRESS_APPROVED=true`) through `StackConfig.compose_files`.
  - The strict profile runs with `LLM_PHI_EGRESS_APPROVED=false` and no endpoint override. In W0 the Portkey gateway points at the fake.
- **Stub OIDC:** the HTTP API and `create_app()` from WS-05.12.
- **IF-20 detail:**
  - `ContractFile` and `EventContract` models;
  - `load_contracts`, `load_event_contracts`, `validate_event_contract`, `verify_event_producer`, `verify_event_consumer`;
  - the rule language (`categories`, `uuid`, `equals:`);
  - the `tests/contracts/events/<producer>__<consumer>/` layout and the directory-ownership protocol in `tests/contracts/README.md`;
  - `assert_request_matches` accepts a raw contract dict as well as `ContractFile`. Its body rule: every `body_example` key is present with the same JSON type, a null example accepts any type, and extra request keys are allowed;
  - `verify_provider` is synchronous and runs its ASGI calls in a private loop on a worker thread, so it works inside async tests.
- **W1 overlay (WS-05.21):** the D-04 Bedrock wiring above, and `RATE_LIMIT_OVERRIDES_JSON` with `auth-login`, `auth-register`, `auth-refresh` and `auth-verify` at 500 per minute. `LLM_PROVIDER` and `RATE_LIMIT_OVERRIDES_JSON` join `OVERLAY_ALLOWED_KEYS`, and the W0 Portkey gateway and Anthropic key entries leave it.
- **Phrase-recall lane (WS-05.22):** `tests/e2e/phrase_recall/` overrides `e2e_stack` with project `solace-e2e-<lane>p`.
- **Generator:** `scripts.test.gen_e2e_env` CLI with compose variables `E2E_SMTP_TLS_MODE`, `E2E_SSL_CERT_FILE` and `SOLACE_SECRETS_DIR`.
- **Others:** the restore-drill CLI and report format; `release_gate.ps1` parameters (`-Tree`, `-Environment`, `-Rehearsal`); `scripts/test/release_gate.py` (`GateClients`, `preflight`, `archive_command`, `host_commands`, `send_command_request`, `remote`, `plan`, `run` and the CLI), which runs the gate on the AWS staging host through SSM (D-25) from a checkpoint tree's archive (D-17), with exit 3 as the D-25 stop.
- **Strict SMTP:** in strict profiles the overlay uses STARTTLS against mailpit with a per-run CA (`SSL_CERT_FILE`). IF-02 forbids `EMAIL_SMTP_TLS_MODE=none` outside development/test, so the IF-19 sentence "sets EMAIL_SMTP_TLS_MODE=none" holds for the dev profile only. This needs a ledger Ruling that names WS-13 and WS-03 as consumers.

## Shared-directory stewardship

WS-05 owns the skeleton's `tests/integration` shared-file entry. Its protocol: new files are named `test_mvp_<id>_<slug>.py`, and an existing file is edited only by the owner of the code it tests. Sanctioned edits to existing files:

| File | Editor | Task | Scope |
|---|---|---|---|
| `tests/integration/test_persistence_roundtrip.py` | WS-05 (steward, for WS-01's MVP-270 guard) | WS-05.2 | Bare `solace_infrastructure` imports only |
| `tests/integration/test_sprint7_portkey_fallback.py` | WS-05 (steward, at WS-09's CR-6) | WS-05.20 | `TestLLMRequestTimeout` only |
| `tests/integration/test_diagnosis_vignette.py` | WS-16 (code owner) | WS-16.11 | Rewrite to run the real step 1 and assert the published event |
| `tests/integration/test_event_flow.py` | WS-16 (code owner) | WS-16.5 | Class `TestDiagnosisBridgeSafetyFlags` only |

## Self-review

**Coverage**

| Register id | Severity | Task(s) |
|---|---|---|
| MVP-039 | high | WS-05.1 to WS-05.6 (W0 harness, boot smoke, crisis E2E red), WS-05.8, 05.9, 05.21, 05.22 (W1), 05.11 (W2), 05.12, 05.13, 05.15, 05.16 (W3), 05.17, 05.18 (W4), 05.19 (W5) |
| MVP-357 | medium | WS-05.7 (framework plus crisis-chain event contracts), WS-05.10 (REST contracts, `therapy__memory` included, plus coverage guard), WS-05.14 (erasure event contracts) |

Both assigned ids are covered. MVP-039 is closed in substance at the W1 barrier, when the crisis E2E is green in both profiles. Later suites extend the same control to each wave's goals. WS-05.20 covers no WS-05 id; it is the `tests/integration` supporting edit WS-09 requested for MVP-139.

REVIEW_FOCUS: 1=WS-05.6,WS-05.22; 2=WS-05.6,WS-05.8; 3=WS-05.11,WS-05.15; 4=WS-05.6,WS-05.15; 5=WS-05.5,WS-05.13,WS-05.21

- Focus 1 (ingress false negatives on the deployed configuration): WS-05.22 sends every clinician-reviewed positive, variants included, through the deployed stack and fails on any result below its floor. WS-05.6 pins the curly-apostrophe, indirect, homicidal and abuse phrasings end to end. Multi-turn build-up is pinned in WS-12's history tests, not here.
- Focus 2 (alert-path concurrency and partial failure): exactly one page per episode (WS-05.6); notification down and then restarted pages once, two detectors page once, an unacknowledged episode re-pages backup and fallback, and a rejected recipient falls back (WS-05.8).
- Focus 3 (foreign identifiers under the real role): a foreign `thread_id` gives 404 (WS-05.11); the isolation matrix checks HTTP and `solace_app`/RLS counts for foreign thread, memory session, therapy plan, diagnosis session and an unassigned clinician (WS-05.15).
- Focus 4 (PHI sinks): the alert email carries no message text or email (WS-05.6); the PHI capture scan covers logs, outbox, DLQ and mail (WS-05.15).
- Focus 5 (strict boots with partial config): the strict boot smoke in staging and production (WS-05.5); a missing secret file fails startup and no secret value sits in container env or logs (WS-05.13); strict stacks render `LLM_PROVIDER=bedrock` with the LLM off and no endpoint override, so strict boots neither fail for a harness reason nor weaken WS-09.18's strict refusal (WS-05.21).

**Interface consistency**
- The IF-19 names are kept:
  - fixtures `pg_migrated` (`PgHandle` with `dsn_app`, `dsn_migrator`, `dsn_superuser`, `app_pool`), `redis_url`, `kafka_bootstrap` (`produce`, `consume_until`), `smtp_sink` (`SmtpSink(api_url).wait_for(to, subject_contains, timeout=60)`), `fake_llm` (`base_url`);
  - `e2e_stack` returns `Stack(urls, edge_url, project, mailpit_api, fake_llm_control, pg_superuser_dsn)` built with `docker compose -p solace-e2e-<lane> -f docker-compose.yml [-f docker-compose.prod.yml] -f docker-compose.test.yml`;
  - helpers `make_user(stack, role='user', verified=True) -> TestUser{id, email, password, access_token}`, `provision_clinician(stack, email, phone, on_call=True, backup=False)`, `fault.stop/start/restart/pause`, `read_logs` (`Stack.logs`), `kafka_lag(group)`, `llm.scenario(mode, match=None)` and `llm.requests()`;
  - env `SOLACE_E2E_PROFILE`, `SOLACE_E2E_ENVIRONMENT`, `SOLACE_E2E_KEEP`, `SOLACE_REQUIRE_REALINFRA`;
  - manifest fields `{test_id, expected_failing_hop, target_wave}` with strict xfail.
- WS-06's plan consumes exactly `PgHandle.superuser_fetch`, `PgHandle.create_database(name, migrate_to=...)`, `app_pool` with the production codec, and `KafkaHandle.produce(... bytes ...)`, `committed_offset`, `consume_until`, `pause`, `unpause`. These are all defined in WS-05.2, which WS-06.1 and WS-06.4 depend on.
- IF-20 names are kept: `verify_provider(app, contracts_dir, *, auth_headers)`, `assert_request_matches(contract, request)`, and JSON contract files `{consumer, provider, interaction, request{method, path, headers_required, body_example}, response{status, body_schema}}`. Event contracts are validated against `EVENT_REGISTRY` for CrisisDetected, EscalationTriggered, SafetyAssessment, UserDeleted and UserErasureCompleted.
- IF-15 fake-LLM modes are kept: `ok`, `error`, `timeout`, `crisis_intent`, `echo`; `reply` is an addition. The Converse route (D-04) serves the same modes in the Bedrock shape.

**Open risks**
- **Hop prediction.** Acceptance suites written red predict the failing hop before phase B lands. If a phase-B task moves a test to a different hop mid-wave, the barrier blocks until the manifest is updated with a ledger note. This is by design, but it costs barrier time.
- **Shared on-call roster.** The roster is global within a stack, so pages from different tests can interleave. Tests isolate by care-team assignment (`--assign`), unique identities and `pages_since(t0)`. A WS-13 change to recipient precedence would surface as `clinician_page` failures, not silent passes.
- **Alerting suites are slow** (rule `for:` windows of 2-5 minutes), so they run at barriers only. Docker capacity limits the gate to 2 concurrent stacks.
- **Mailpit features.** STARTTLS and allowed-recipient features depend on the pinned mailpit version. If they are absent, the strict profile needs the Ruling noted under New interfaces.
- **Strict-profile STARTTLS** relies on WS-13's SMTP client honouring the default trust store from `SSL_CERT_FILE`. If it pins its own context, WS-13 must accept a CA path through an IF-02 variable.
- **No literal graph-exception E2E.** "Graph exception gives resources and a page" is covered by the all-agents-failing E2E, plus WS-10's in-process `run_turn` exception tests. No production-code fault seam is introduced for E2E.
- **Untouched by this plan.** `scripts/seed_demo_data.py` (no register entry) and `tests/alignment/test_event_schemas.py`, which imports memory-service internals and may need a WS-17 request if those move.
- **Phrase-recall cost.** WS-05.22 boots a second stack and creates one verified user per positive (200 or more), which takes several minutes. It therefore counts against the two-concurrent-stacks limit whenever `tests/e2e` runs.
- **Strict stacks run with the LLM off (D-04).** WS-09.18 refuses `BEDROCK_ENDPOINT_URL` in strict environments, so strict stacks (and the W5 release gate, which runs strict only) answer with template replies and never exercise the Bedrock path against the fake. This matches a strict deployment before the AWS BAA is recorded. It has two consequences. Any acceptance test that drives fake-LLM scenarios (`crisis_intent`, `error`, `timeout`, `reply`) must be `profile("dev")` or assert the LLM-off behaviour, and its owner moves it by ledger note if it was written for strict. Turning the LLM on for real users after the BAA needs its own verification on the staging host (an owner decision).
- **Release gate needs AWS (D-25).** The gate itself only runs once the owner attaches the deployer policy to `MedCodex-Local` and the staging stack exists with request 17's outputs and permissions. Everything before that, the W5 rehearsal included, runs on local Docker. A missing grant at W5 entry is a stop (preflight exit 3), and the gate is never replaced by the local rehearsal.
- **Duplicate client contracts (resolved).** WS-11 accepted request 13: WS-11.15 creates no file under `client__orchestrator/` and verifies WS-05.10's three files, which use WS-11's seed ids. A further interaction (for example `/batch`) comes back to WS-05 as a request.

**Checker warnings justified**
- WS-05.7's alignment test parses the orchestrator's agent modules with `ast` and never edits them; they stay WS-10/WS-11 code. The client-module check moved to WS-11.2 (request 15), so WS-05.7 no longer names that module.
- WS-05.19's release-gate script runs `tests/contracts` and `tests/clinical` among its suites. The suite list now sits under Interfaces, not on a Create line; the task creates or edits nothing in either directory (`tests/clinical` stays WS-12's).

## Review responses

- Plan check (error): WS-05.7 modifies `src/infrastructure/clients.py`, which does not exist. Fixed: the path is now `services/orchestrator_service/src/infrastructure/clients.py`, the real file (the root-level path does not exist). The task names it as a read-only AST input, not an edit.
- Plan check (error): WS-05.13 modifies `tests/scripts/test_mvp_039_gen_e2e_env.py`, which no earlier task or dependency creates. Fixed: WS-05.4 (W0) now lists the file as `Create:` (the checker did not read `Test: ... (new)` as a creation), and WS-05.13 depends on WS-05.4 directly.
- Plan check (error): WS-06's request for IF-19 fixture capabilities (`superuser_fetch`, `create_database`, the codec-backed `app_pool`, `produce` with bytes, `committed_offset`, `consume_until`, `pause`/`unpause`). Fixed: WS-05.2 already defines all of them. Its Produces line now names WS-06's request 2 and the needing tasks WS-06.1 and WS-06.4.
- Plan check (error): WS-07's request for `RATE_LIMIT_OVERRIDES_JSON` in `docker-compose.test.yml` (needed by WS-07.7). Fixed: new task WS-05.21 (W1) sets it to 500 per minute for `auth-login`, `auth-register`, `auth-refresh` and `auth-verify` on every app service, and adds it to `OVERLAY_ALLOWED_KEYS`. Its Produces line names WS-07.7. Cross-workstream request 14 asks WS-03 for the registry entry, so `test_env_contract.py` accepts the value.
- Plan check (error): WS-09's CR-3, per-run non-empty `LLM_PORTKEY_API_KEY` and `LLM_ANTHROPIC_API_KEY` next to `LLM_PHI_EGRESS_APPROVED=true` (needed by WS-09.4). Fixed in WS-05.21. The empty `LLM_ANTHROPIC_API_KEY=""` from WS-05.4 is replaced by per-run generated values in both profiles, and WS-05.13 moves them to `_FILE` in strict at W3. Its Produces line names WS-09.4.
- Plan check (error): WS-09's CR-6, rewriting `TestLLMRequestTimeout::test_generate_returns_empty_on_timeout_not_hang` in `tests/integration/test_sprint7_portkey_fallback.py` (needed by WS-09.3). Fixed: new task WS-05.20 (W1, depends on WS-09.3) replaces it with `test_generate_raises_unavailable_on_timeout`, which expects `LLMUnavailableError` with `reason == "timeout"`. WS-05 makes the edit as owner of the `tests/integration` shared-file entry, at WS-09's request. The old test (lines 57-80) asserts `result == ""`.
- Plan check (error): WS-15's R5 contract files (needed by WS-15.10). Fixed for `tests/contracts/therapy__memory/context.json`: it joins WS-05.10 (W2 phase A), built from `ContextAssemblyRequest`/`ContextAssemblyResponse` (`services/memory_service/src/schemas.py:89-111`), and WS-05.10's Produces line names WS-15.10. Rejected for `orchestrator__therapy/start_session.json` and `process_message.json`: WS-11.9 already creates the consumer directory `tests/contracts/orchestrator__therapy/` (`session_start.json`, `message.json`; WS-11 plan, Task WS-11.9 Files), and IF-20 makes the consumer the writer. A second writer would create duplicate contracts for the same routes. The fixed ids, the 201/200 bodies and the 404/403/409 interactions go to WS-11 as cross-workstream request 12, so WS-15.11 depends on WS-11.9.
- Plan check (error): WS-16's R12, acknowledging the code-owner edits by WS-16.11 (`tests/integration/test_diagnosis_vignette.py`) and WS-16.5 (class `TestDiagnosisBridgeSafetyFlags` in `tests/integration/test_event_flow.py`, line 163). Fixed: both are sanctioned in the new "Shared-directory stewardship" table and in WS-05.20's Produces line. WS-05 makes no edit to either file.
- Plan check (warning): WS-05.7 edits `services/orchestrator_service/src/agents`, which no workstream owns. Rejected: the task only parses those modules (`test_orchestrator_clients_use_versioned_provider_prefixes`) and never writes to them. The wording now says "only parses", and "Checker warnings justified" records it.
- Plan check (warning): WS-05.7 edits `src/infrastructure/clients.py`, which no workstream owns. Fixed by the path correction above. The file is a read-only input, not an edit.
- Plan check (warning): WS-05.19 edits `tests/clinical`, owned by WS-12. Rejected: `tests/clinical` appears only in the list of suites that `release_gate.ps1` runs (WS-05.19 Files, item 2). WS-05.19 creates and modifies no file there.
- AGY (error/IMPORT): WS-05.2 imports `SCHEMA_HEAD` from `solace_infrastructure.database.schema_check`, which does not exist yet and is created by WS-02.3, but WS-05.2 depends only on WS-05.1. Fixed: confirmed that `src/solace_infrastructure/database/schema_check.py` is absent today. WS-05.2 now depends on WS-02.3 and also on WS-02.4, which creates the `scripts.db.migrate` CLI that `pg_migrated` runs.
- AGY (warning/PATH): `src/infrastructure/clients.py` should be `services/orchestrator_service/src/infrastructure/clients.py`. Fixed (same edit as the first error).
- Reconciliation 4 (WS-05.7 named WS-11's `services/orchestrator_service/src/infrastructure/clients.py` in its Files line): fixed. The alignment test now parses only the agent modules and is renamed `test_orchestrator_agents_use_versioned_provider_prefixes`; the client-module half is new request 15 to WS-11, planned in WS-11.2 (W1, which already modifies the module) as the lock `test_orchestrator_clients_use_versioned_provider_prefixes`. WS-11.2's Produces line names WS-05.10, and WS-05.10 depends on WS-11.2 and checks the lock exists (`test_orchestrator_client_module_prefixes_are_locked_by_their_owner`). WS-05.7 itself cannot depend on WS-11.2 without a cycle (WS-11.2 depends on WS-10.3, which depends on WS-05.7).
- Reconciliation 5 (WS-05.10 named WS-17's `services/memory_service/src/schemas.py` as the source of the `therapy__memory` contract): fixed. WS-05.10 no longer names the file; it copies the provider's published examples. New request 16 asks WS-17 to publish them, planned as WS-17.19 (W0 memory lane: `THERAPY_CONTEXT_REQUEST_EXAMPLE`/`THERAPY_CONTEXT_RESPONSE_EXAMPLE` set as the JSON-schema examples of `ContextAssemblyRequest`/`ContextAssemblyResponse`, RED first). WS-17.19's Produces line names WS-05.10, WS-05.10 depends on WS-17.19, and `test_therapy_memory_example_is_the_providers_published_example` keeps the contract equal to the provider's example. WS-17.18 (the provider verification) now names WS-05.10 as the file's producer.
- Reconciliation 6 (contract files have one creator): confirmed and tightened. WS-05.7 alone creates `tests/contracts/README.md` and `tests/contracts/contract.schema.json`, and WS-05.10 alone creates `tests/contracts/therapy__memory/context.json` (WS-15's request R14); WS-15.10 no longer lists any of the three and depends on WS-05.7 and WS-05.10. WS-05.19's Create line names only `scripts/test/release_gate.ps1`; the suite list it runs, including `tests/contracts`, moved under Interfaces.
- Reconciliation 10 (request 9, producer `verify_event_producer` tests): every crisis producer now lists its test under Create: WS-10.3 (Create line added), WS-12.8, WS-15.5 and WS-16.5. WS-05.10 depends on the two W1 producers, WS-10.3 and WS-12.8; WS-15.5 and WS-16.5 are W2 phase B, so they cannot be dependencies of this phase-A task, and their `producer_verification` ids stay in `tests/contracts/expected_failures.yaml` until they land (Step 3 already says so).
- Reconciliation 11 (request 13, IF-20 call shapes and `client__orchestrator` ownership): aligned with WS-11. WS-11.15 calls `verify_provider` without `await`, creates no contract file, and checks that WS-05.10's three files exist. Row 13 no longer lists `tests/contracts/client__orchestrator/*.json` among WS-11's files, WS-05.10 now writes the seed ids WS-11 asked for (user `...a000-000000000001`, foreign thread `...b000-000000000002` owned by `...a000-000000000002`, history session `...c000-000000000001` and thread `...b000-000000000001`) and names WS-11.15 in Produces, and the "duplicate client contracts" open risk is closed. The optional `batch.post.json` is not written yet; it comes back as a request when `/batch` gets a client contract.
- crisis-review F1: fixed. Verified: the module-scoped `oncall` fixture was first requested by the seventh test, so the four REST and two WS crisis tests ran with an empty roster and their pages could only reach the fallback mailbox. The fixture is now module-scoped `autouse` and defined first, so this module never sends a crisis message to an empty roster; the consumer-side stall itself is fixed in WS-06.4 (partitions processed concurrently) and WS-13.6 (pending pages retried by the pager, never by holding the consume loop). WS-05.8 adds `test_unreachable_recipient_for_one_user_does_not_delay_another_users_page` (user B is paged within 60 s while user A's care-team recipient refuses). In W0 every test of WS-05.6 now stops at `provision_clinician`; Step 2 says so.
- crisis-review F2: fixed. WS-05.8 adds `test_postgres_outage_during_a_crisis_turn_still_pages_once_after_recovery`, parametrized `stop` and `pause`: resources within 3 s during the outage and exactly one episode paged within 120 s of recovery. The fixes are WS-10.3/WS-10.5 (orchestrator publish retrier) and WS-12.9 (safety record retry). Noted for W2: WS-11.7's thread-ownership lookup also sits on this path from W2, and this test guards it at every later barrier.
- crisis-review F4: fixed. WS-05.6 asserted `res["contact"] == "988"`, but D-29 and WS-12.2 make the contact "Call or text 988". The hops now assert that "988" is in a contact and that the resources equal `crisis_resources("en-US")` with `resources_version == RESOURCES_VERSION`, importing `solace_clinical` inside the hop because it lands in W1. The WS frame gets the same check. The user-facing text is unchanged.
- crisis-review F5: fixed. WS-11.5 sends the informational `connected` frame first and its own unit test reads it first, so WS-05.6's "first frame is safety.alert" could never pass. WS-05.6 now reads and checks `connected` under `ws_connect` and measures the 3 s budget from the send. The skeleton's IF-12 text still omits the frame; WS-11's New interfaces lists it and WS-11.5's Produces now says it is always first, for the IF-12 ledger Ruling.
- crisis-review F6: fixed. WS-05.6 now creates `crisis_episodes`, `pages_for_episode` and `wait_for_episode_page` in `src/solace_testing/e2e.py` (moved from WS-05.8, which consumes them), and every page assertion in WS-05.6 and WS-05.8 is scoped to the test user's episode instead of a whole mailbox, because the stack, the on-call roster and the fallback mailbox are shared by the session. The benign-phrase test checks that its user has no episode, instead of reading the shared mailbox where other tests' re-pages arrive. "Exactly one page" now means one page per level of the episode, because WS-13.6 (F8) pages a same-turn rise again.
- crisis-review F7: fixed (the deadline is in WS-10.5). WS-05.8 adds `test_safety_service_hung_still_answers_within_3s`, which pauses safety-service (a hung `/check`) instead of stopping it (a fast refusal).
- crisis-review F10: fixed. WS-05.7 applied the `categories` rule to `escalation_reason`, which rejects the `auto:CRITICAL:suicidal_ideation` form that WS-12.8 emits (its producer test uses that value) and WS-13.6 parses. WS-05.7 adds the rule kind `escalation_reason` (the value is `unspecified` or `solace_clinical.parse_escalation_reason` accepts it; WS-12.2), uses it in the escalation contract with that example, and pins it with `test_crisis_review_f10_escalation_reason_rule_accepts_the_producer_format_only`.
- crisis-review F13: fixed. WS-05.9 adds `test_single_crisis_turn_with_a_stalled_consumer_fires_crisis_consumer_stalled` (notification-service paused, one crisis message, `CrisisConsumerStalled` within 420 s, the page after unpause). The 11-user test stays; it could not have passed before, because nobody exported the notification group's lag while notification-service was stopped, and from WS-06.10 the other planes do.
- decision 2026-10-19 D-20: the "installed per D-20" wording now names the dedicated venv `D:\tmp\mvp-exec\venv` (controller at W0 entry, then WS-01.2's re-sync; never anaconda) and the lock-only `D:\tmp\mvp-exec\venv-clean` used at the W0 barrier. Changed: the WS-01 dependency line, the WS-05.2 decision gate and cross-workstream request 1. No command changed; all already used the venv interpreter.
- decision 2026-10-19 D-04: real-user LLM traffic goes to Bedrock; tests still use only the fake. WS-05.1: decision gate; the new Converse route `POST /model/{model_id}/converse` (Produces, Step 3), with the same scenarios, the Converse body, `x-amzn-ErrorType` errors and SigV4 ignored; the new test `test_d04_bedrock_converse_route_answers_the_real_botocore_client` (unsigned botocore client against the in-process fake) and its Step 2 RED; the File map row. WS-05.4: the decision gate says W0 keeps only the Portkey gateway pointed at the fake, and the W0 overlay line is unchanged. WS-05.21 is rewritten to plan WS-09's revised CR-3: the title, decision gate, Files and Interfaces change; `LLM_PROVIDER=bedrock` goes on the five LLM services in both profiles; the new dev-only fragment `infrastructure/test/compose/llm-dev.yml` (`BEDROCK_ENDPOINT_URL=http://fake-llm:8080`, synthetic `BEDROCK_MODEL_ID=fake-llm-model`, `LLM_PHI_EGRESS_APPROVED=true`) is added through `StackConfig.compose_files`; strict runs with `LLM_PHI_EGRESS_APPROVED=false` and no override, because WS-09.18 refuses it; neither profile carries `LLM_PORTKEY_*`, a provider API key or an AWS credential; the per-run Portkey/Anthropic key generation and its generator test are dropped; the new tests `test_d04_dev_stacks_reach_the_fake_llm_as_bedrock` and `test_d04_strict_stacks_run_bedrock_with_the_llm_off_and_no_override` come with the Step 2 RED and the Step 5 dev run. Also: WS-05.13 drops the LLM-key sentence; the File map (new fragment row; WS-05.21 no longer edits the generator or its test); New interfaces; Focus 5; the open risk "LLM keys under IF-22" is removed and the strict-LLM-off risk added. WS-03's registry entries are WS-09's CR-14 and CR-4, so WS-05 adds no duplicate request. New request 19 asks WS-03 to drop its planned allow-listed strict `BEDROCK_ENDPOINT_URL` difference, which conflicts with WS-09.18's strict refusal, and to render the dev profile with the new fragment. The overlay is WS-05's own file, so no WS-03 compose request was needed.
- decision 2026-10-19 D-25: the WS-05.19 release gate moves from local Docker to the AWS staging host through SSM, with the strict production profile and `MedCodex-Local`. The decision gate is updated, and WS-03.11 and WS-03.20 are added to Depends on. The new `scripts/test/release_gate.py` provides `preflight` (read-only checks, exit 3 = stop, never a local fallback), `remote` (archive upload, `AWS-RunShellScript` with a bounded `executionTimeout`, result download) and `run` (the two-run suite runner, moved from the PowerShell script so it runs on the Linux host). `release_gate.ps1` is now a wrapper with a `-Rehearsal` mode on local Docker that never counts as the gate. Step 1 has new stubbed tests: plan, SSM request, preflight stop and wrapper order. Steps 2, 3 and 5 are updated: rehearsal, then preflight at W5 entry, then the AWS gate. PRR.md names the backup target the restore drill used. WS-05.17 now has a D-25 decision gate: the drill report copies the backup fragment's `backup_target` and adds a D-25 `target_note` when the target is `local` (WS-03's request for WS-03.9 and WS-03.16), and its test asserts both. Also the File map, the WS-03 dependency line, New interfaces, an open risk, and new request 17 to WS-03. That request asks for the archive at `artifacts/<tree>.zip` and results under `gate/<tree>/` in WS-03.20's `ArtifactBucketName`, plus the deployer-policy and instance-role actions and the host tools.
- decision 2026-10-19 D-17: WS-05.19's decision gate now says nothing is committed and the gate runs from `git archive <final checkpoint tree id>` (`release_gate.archive_command`, asserted in Step 1), reporting the tree id only. The host builds and probes every image from the expanded archive (`build_images.py --all --context . --probe`), as WS-01's R-01.3(c) now asks, instead of `make release-gate-closure`.
- decision 2026-10-19 D-07: SOPS + age replaces the Secrets Manager store. The WS-05.13 decision gate now says the per-run directory stands in for `deploy/render_secrets.py`'s SOPS render into `/run/secrets`, with no real SOPS file or age key in tests. The WS-05.19 PRR "Secrets rotation" item names the SOPS re-encrypt-and-redeploy procedure and the age recipient, never a value. The NAME_FILE contract is unchanged.
- decision 2026-10-19 D-29: the WS-05.6 decision gate and Consumes record the per-locale registry. `crisis_resources("en-US")` stays exactly today's set, so the assertions are unchanged; harness users carry the default `en-US` locale, and nothing assumes a single locale. New request 18 asks WS-12 for the en-US version accessor under per-locale versioning, and for the fallback function, metric and locale contract, before any fallback E2E hop is added.
