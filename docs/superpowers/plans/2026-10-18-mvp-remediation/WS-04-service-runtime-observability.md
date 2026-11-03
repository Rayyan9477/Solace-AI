# WS-04: Service runtime bootstrap, health, metrics and alerting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every MVP service boots through one runtime bootstrap (IF-01) that fails loud in strict environments, logs PHI-safe JSON with tracebacks, serves truthful `/live`, `/ready`, `/health` and `/metrics`, tears down cleanly, and feeds promtool-tested Prometheus and Alertmanager rules that page on crisis-path, service, host and backup failures (IF-18).

**Architecture:** A new package `solace_infrastructure.runtime` (`spec.py`, `logging.py`, `http.py`, `health.py`, `metrics.py`, `datastores.py`, `bootstrap.py`) exposes `create_service_app`, `start_service_runtime` and `stop_service_runtime`; the nine MVP services adopt it once in W0 phase B, and every later component (service tokens, event plane, rate limiting, PHI codec, audit) is wired only in `bootstrap.py`, gated by a `ServiceRuntimeSpec` flag the services declared in W0. `ProductionGuards` becomes the single strict-environment gate keyed on `is_strict_environment()`. Observability ships as `infrastructure/prometheus` config and rules plus the `docker-compose.observability.yml` fragment (Alertmanager in W1; node-exporter, autoheal and log shipping in W4).

**Covers:** 6 register entries (blocker 1, high 1, medium 3, low 1): MVP-031, MVP-036, MVP-114, MVP-119, MVP-180, MVP-332. Supporting wiring tasks also carry IF-13/IF-14/D-11 work owned elsewhere (see WS-04.8, WS-04.9, WS-04.10).

**Waves:** 0, 1, 2, 3, 4 (every task is phase A: `src/**`, `infrastructure/**`, compose fragments).

**Depends on:**
- WS-01: `prometheus-client` in `requirements.lock`, the lazy (PEP 562) `src/solace_infrastructure/__init__.py`, the realinfra/e2e markers, and `docker/entrypoint.sh` (NAME_FILE loader) used from W3.
- WS-02: the single `'default'` pool built from `POSTGRES_*`, the strict-env pool policy, and `solace_infrastructure.database.schema_check.assert_schema_head(pool)` wrapped by `runtime/datastores.py`.
- WS-03: base compose services and networks that `docker-compose.observability.yml` overlays, registry rows for the Alertmanager variables and the Prometheus directory mount (WS-03.6), the strict NAME_FILE secret delivery (WS-03.7/WS-03.8), the backup textfile metric (WS-03.9) and autoheal labels (WS-03.13).
- WS-06: `start_service_event_plane` (IF-05) and its outbox/lag/DLQ metrics, wired in W1.
- WS-07: `ServiceTokenManager`, `install_rate_limiting(app, spec)` (IF-04) wired in W1, and the correlation-id helpers in `solace_common.request_context` (requested below, W0).
- WS-09: `get_phi_codec` (IF-13, W2) and `configure_service_audit` (IF-14, W3).

**Owned paths:** `src/solace_infrastructure/runtime/**`, `src/solace_infrastructure/health.py`, `src/solace_infrastructure/observability_core.py`, `src/solace_infrastructure/observability/**`, `src/solace_security/production_guards.py`, `infrastructure/prometheus/**`, `infrastructure/grafana/**`, `infrastructure/alertmanager/**`, `docker-compose.observability.yml`, `tests/bootstrap/**`, `docs/runbooks/alerting.md`. New e2e files are added under `tests/e2e/` as `test_mvp_<id>_<slug>.py` using only IF-19 fixtures. Under the shared-test-dir rule WS-04 also edits `tests/integration/test_phi_encryption_activation.py` (WS-04.15), because IF-01 moves the startup PHI activation it locks into the runtime bootstrap (CR-8 of WS-09).

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md. Interfaces IF-01 and IF-18 are produced here exactly as written in the skeleton; additions are listed under "New interfaces".

**Command conventions used below** (PowerShell 5.1, one pytest process per suite):
- `UNIT <path>` means: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`
- `DOCKER <path>` means the same command prefixed with `$env:SOLACE_RUN_DOCKER_TESTS="1"; ` (tests marked `integration` plus `realinfra` or `e2e` inside `tests/bootstrap` are skipped unless that variable or `SOLACE_REQUIRE_REALINFRA=1` is set; with either set, a missing `docker` CLI is a failure, not a skip).
- `BOOTSMOKE` means two runs of WS-05's acceptance file: `$env:SOLACE_E2E_PROFILE="dev"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_w0_boot_smoke.py` and then `$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_w0_boot_smoke.py`.

---

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `src/solace_security/production_guards.py` | Modify | Strict gate: `is_strict_environment`, required secrets per caller, value-free violations, verify-full floor; W3 dev-value deny-list, entropy floor and NAME_FILE provenance | WS-04.1, WS-04.11 |
| `src/solace_infrastructure/runtime/__init__.py` | Create | PEP 562 lazy exports of the IF-01 names | WS-04.2 |
| `src/solace_infrastructure/runtime/spec.py` | Create | `ServiceRuntimeSpec`, `StartupDependencyError` | WS-04.2 |
| `src/solace_infrastructure/runtime/logging.py` | Create | `configure_service_logging`, `logging_chain_names`; stdlib/uvicorn routing | WS-04.2 |
| `src/solace_infrastructure/runtime/http.py` | Create / Modify | Request-context ASGI middleware, access line, uniform error bodies, CORS for edge; W2 edge HTTPException envelope | WS-04.3, WS-04.9 |
| `src/solace_infrastructure/runtime/health.py` | Create | `/live`, `/ready`, `/health` routes, `CheckResult`, `ReadinessReport` | WS-04.3 |
| `src/solace_infrastructure/runtime/metrics.py` | Create | Standard metric objects on the default registry; `/metrics` route | WS-04.3 |
| `src/solace_infrastructure/runtime/bootstrap.py` | Create / Modify | `ServiceRuntime`, `create_service_app`, `start_service_runtime`, `stop_service_runtime`, `required_secrets_for`; W1/W2/W3 component wiring | WS-04.3, WS-04.4, WS-04.6, WS-04.8, WS-04.10 |
| `src/solace_infrastructure/runtime/datastores.py` | Create | Postgres pool and schema-head check, Redis client, readiness checks, closers | WS-04.4 |
| `infrastructure/prometheus/prometheus.yml` | Modify | One `solace-services` job for the nine services; rule files; Alertmanager; node job | WS-04.5, WS-04.7, WS-04.12 |
| `infrastructure/prometheus/rules/solace.yml` | Create / Modify | IF-18 alert rules (W1 crisis path, W4 host/backup/audit/container) | WS-04.7, WS-04.12, WS-04.13 |
| `infrastructure/prometheus/tests/solace_rules_test.yml` | Create / Modify | promtool unit tests for every rule | WS-04.7, WS-04.12, WS-04.13 |
| `infrastructure/alertmanager/alertmanager.yml.tmpl` | Create | Routing to `ALERTMANAGER_EMAIL_TO` via SMTP | WS-04.7 |
| `infrastructure/alertmanager/render-config.sh` | Create / Modify | Renders the template from env at container start; W3 reads the SMTP password from a file | WS-04.7, WS-04.11 |
| `docker-compose.observability.yml` | Create / Modify | Alertmanager (W1); node-exporter, socket proxy, autoheal, Loki, Alloy (W4) | WS-04.7, WS-04.11, WS-04.12, WS-04.13, WS-04.14 |
| `infrastructure/grafana/loki/loki-config.yml` | Create | Loki storage and 31-day retention | WS-04.14 |
| `infrastructure/grafana/alloy/config.alloy` | Create | Docker log discovery through the socket proxy, push to Loki | WS-04.14 |
| `infrastructure/grafana/provisioning/datasources/loki.yml` | Create | Grafana Loki datasource | WS-04.14 |
| `docs/runbooks/alerting.md` | Create / Modify | One section per alert (anchor = lower-case alert name), log access | WS-04.7, WS-04.12, WS-04.13, WS-04.14 |
| `tests/bootstrap/conftest.py` | Create | Docker opt-in gate; `make_spec` fixture; strict env fixture | WS-04.1 |
| `tests/bootstrap/test_mvp_031_production_guards.py` | Create | MVP-031 proofs | WS-04.1 |
| `tests/bootstrap/test_mvp_180_logging.py` | Create | MVP-180 proofs | WS-04.2 |
| `tests/bootstrap/test_mvp_036_service_app.py` | Create | `/metrics`, health, error bodies, request ids | WS-04.3 |
| `tests/bootstrap/test_mvp_332_runtime_lifecycle.py` | Create | Start order, strict/dev degradation, teardown | WS-04.4 |
| `tests/bootstrap/test_service_conformance.py` | Create | Nine services adopt IF-01 (subprocess probe + AST) | WS-04.5 |
| `tests/bootstrap/_conformance_probe.py` | Create | Subprocess import probe that prints a JSON report | WS-04.5 |
| `tests/bootstrap/fixtures/reference_app.py` | Create | Positive control for the probe | WS-04.5 |
| `tests/bootstrap/fixtures/nonconforming_app.py` | Create | Negative control for the probe | WS-04.5 |
| `tests/bootstrap/test_mvp_036_prometheus_scrape.py` | Create | Scrape config proofs (+ promtool check config) | WS-04.5 |
| `tests/bootstrap/test_mvp_332_w1_wiring.py` | Create | W1 component wiring proofs | WS-04.6 |
| `tests/bootstrap/test_mvp_036_alert_rules.py` | Create | Rule/Alertmanager structure + promtool/amtool | WS-04.7 |
| `tests/e2e/test_mvp_036_alert_delivery.py` | Create | ServiceNotReady reaches mailpit without PHI | WS-04.7 |
| `tests/bootstrap/test_mvp_055_codec_warmup.py` | Create | PHI codec warm-up wiring | WS-04.8 |
| `tests/bootstrap/test_d11_edge_error_envelope.py` | Create | Edge HTTPException envelope | WS-04.9 |
| `tests/bootstrap/test_mvp_091_audit_wiring.py` | Create | Audit wiring | WS-04.10 |
| `tests/bootstrap/test_mvp_349_secret_source_guards.py` | Create | Dev-value, entropy and NAME_FILE guards | WS-04.11 |
| `tests/bootstrap/test_mvp_044_host_backup_alerts.py` | Create | Host/disk/backup/audit rules | WS-04.12 |
| `tests/bootstrap/test_mvp_119_autoheal.py` | Create | Autoheal and socket-proxy confinement | WS-04.13 |
| `tests/e2e/test_mvp_119_autoheal_restart.py` | Create | A wedged service is restarted and alerted | WS-04.13 |
| `tests/bootstrap/test_mvp_114_log_shipping.py` | Create | Loki/Alloy config and retention | WS-04.14 |
| `tests/e2e/test_mvp_114_logs_survive_recreate.py` | Create | Logs survive container recreation | WS-04.14 |
| `tests/integration/test_phi_encryption_activation.py` | Modify | PHI-activation lock accepts a `handles_phi=True` SPEC in place of the retired call (shared test dir, CR-8) | WS-04.15 |

`src/solace_infrastructure/health.py` and `src/solace_infrastructure/observability_core.py` stay unchanged: the runtime does not reuse `HealthMonitor` (its messages embed `str(e)`), and no task edits them.

---

## Tasks

### Task WS-04.1: ProductionGuards require what services read, treat staging as strict and never echo values
**Covers:** MVP-031
**Severity:** blocker
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: the pilot runs ENVIRONMENT=production; staging is identically strict; every guard uses `is_strict_environment()`), D-15 (default assumed: strict Postgres clients use `verify-full`)
**Real infra:** no
**Cross-workstream prerequisites:** none (pure module)
**Files:**
- Modify: `src/solace_security/production_guards.py` (add module-level `is_strict_environment` and `DEFAULT_REQUIRED_IN_STRICT`; `REQUIRED_IN_PRODUCTION` becomes the four names the code reads; `FORBIDDEN_VALUES` keyed on those names; `validate_environment(*, required_vars=None)`; `assert_startup(..., required_vars=None)`; `STRICT_ENVIRONMENTS` checks replaced by `is_strict_environment`; violation strings carry names only; `POSTGRES_SSL_MODE` must be `verify-full` when `POSTGRES_PASSWORD` is required)
- Create: `tests/bootstrap/conftest.py` (docker opt-in gate hook; `make_spec` and `strict_secrets` fixtures used by later tasks)
- Test: `tests/bootstrap/test_mvp_031_production_guards.py`
**Interfaces:**
- Consumes: none
- Produces: `is_strict_environment(env: str | None = None) -> bool` (canonical home `solace_security.production_guards`; re-exported lazily as `solace_infrastructure.runtime.is_strict_environment` in WS-04.2); `DEFAULT_REQUIRED_IN_STRICT: tuple[str, ...] = ("AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "POSTGRES_PASSWORD")`; `ProductionGuards.validate_environment(*, required_vars: Collection[str] | None = None) -> None`; `ProductionGuards.assert_startup(service_environment: str | None, *, service_name: str | None = None, required_vars: Collection[str] | None = None) -> None` (None means `DEFAULT_REQUIRED_IN_STRICT`, so un-adopted mains keep working).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_031_production_guards.py`
```python
"""MVP-031: guards must require AUTH_SECRET_KEY, ENCRYPTION_MASTER_KEY, ENCRYPTION_SEARCH_HASH_SALT
and POSTGRES_PASSWORD (what the services read), not SECRET_KEY/ENCRYPTION_KEY; staging is strict;
violations never contain a value."""
import secrets
import pytest
from solace_security.production_guards import ProductionGuards, ProductionSecurityError

READ_NAMES = ("AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "POSTGRES_PASSWORD")

@pytest.fixture
def strict_ok(monkeypatch):
    for gone in ("SECRET_KEY", "ENCRYPTION_KEY", "JWT_SECRET", "API_KEY", "DEBUG", "TESTING", "MOCK_SERVICES", "USE_IN_MEMORY_DB"):
        monkeypatch.delenv(gone, raising=False)
    values = {"AUTH_SECRET_KEY": secrets.token_urlsafe(48), "ENCRYPTION_MASTER_KEY": secrets.token_hex(32),
              "ENCRYPTION_SEARCH_HASH_SALT": secrets.token_urlsafe(32), "POSTGRES_PASSWORD": secrets.token_urlsafe(24),
              "POSTGRES_USER": "solace_app", "POSTGRES_SSL_MODE": "verify-full"}
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    return values

@pytest.mark.parametrize("env", ["staging", "production", "Staging", " PRODUCTION "])
def test_is_strict_environment_true(env):
    from solace_security.production_guards import is_strict_environment
    assert is_strict_environment(env) is True

@pytest.mark.parametrize("env", ["development", "test", ""])
def test_is_strict_environment_false(env):
    from solace_security.production_guards import is_strict_environment
    assert is_strict_environment(env) is False

def test_is_strict_environment_reads_ENVIRONMENT_when_none(monkeypatch):
    from solace_security.production_guards import is_strict_environment
    monkeypatch.setenv("ENVIRONMENT", "staging")
    assert is_strict_environment() is True
    monkeypatch.delenv("ENVIRONMENT")
    assert is_strict_environment() is False

@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_passes_with_the_variables_services_read(monkeypatch, strict_ok, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)
    ProductionGuards.validate_environment()

@pytest.mark.parametrize("missing", READ_NAMES)
def test_each_read_variable_is_required_in_strict(monkeypatch, strict_ok, missing):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.delenv(missing)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith(f"{missing} ") for v in exc.value.violations)

def test_required_vars_follow_the_caller(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ENCRYPTION_MASTER_KEY")
    monkeypatch.delenv("ENCRYPTION_SEARCH_HASH_SALT")
    ProductionGuards.assert_startup("production", service_name="analytics-service",
                                    required_vars=("AUTH_SECRET_KEY", "POSTGRES_PASSWORD"))

def test_violation_never_contains_the_value(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "changeme")
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert "changeme" not in str(exc.value)
    assert all("changeme" not in v for v in exc.value.violations)

def test_strict_requires_verify_full(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("POSTGRES_SSL_MODE", "require")
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith("POSTGRES_SSL_MODE ") for v in exc.value.violations)

def test_dev_is_never_blocked(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    for name in READ_NAMES:
        monkeypatch.delenv(name, raising=False)
    ProductionGuards.validate_environment()
```
`tests/bootstrap/conftest.py` defines: a `pytest_collection_modifyitems` hook that skips items marked `realinfra` or `e2e` unless `SOLACE_RUN_DOCKER_TESTS=1` or `SOLACE_REQUIRE_REALINFRA=1` (and fails them when either is set but `shutil.which("docker")` is None); `make_spec` (factory returning `ServiceRuntimeSpec(service_name="safety-service", environment="development", **overrides)`, imported lazily so WS-04.1 runs before the runtime exists); `strict_secrets` (the `strict_ok` values above, reusable).
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_031_production_guards.py`
Expected: FAIL. `test_strict_passes_*` raise `ProductionSecurityError` listing "SECRET_KEY is not set" and "ENCRYPTION_KEY is not set" (the MVP-031 defect); `test_each_read_variable_is_required_in_strict[AUTH_SECRET_KEY]`, `[ENCRYPTION_MASTER_KEY]` and `[ENCRYPTION_SEARCH_HASH_SALT]` fail because no violation names them; `test_required_vars_follow_the_caller` raises `TypeError: ... unexpected keyword argument 'required_vars'`; `test_violation_never_contains_the_value` fails because the message contains `('changeme')`; `test_strict_requires_verify_full` does not raise; the three `is_strict_environment` tests raise `ImportError`.
- [ ] **Step 3: Implement** `is_strict_environment(env: str | None = None) -> bool`, `DEFAULT_REQUIRED_IN_STRICT`, `ProductionGuards.validate_environment(*, required_vars=None)` and `ProductionGuards.assert_startup(service_environment, *, service_name=None, required_vars=None)` in `src/solace_security/production_guards.py`. Approach: the required set is `tuple(required_vars) if required_vars is not None else DEFAULT_REQUIRED_IN_STRICT`; forbidden-value checks run only for names in that set; database checks run only when `POSTGRES_PASSWORD` is in the set and then require `POSTGRES_SSL_MODE == "verify-full"` and a non-superuser `POSTGRES_USER`; every violation string starts with the variable name followed by a space and never interpolates a value; `assert_environment_consistency` keeps its behaviour but uses `is_strict_environment` for both sides; the `checks_passed` log keeps names only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_security`; `UNIT D:\Repo\Contextual-Chatbot\services\memory_service\tests\test_env_consistency_startup.py`. Expected: all pass (`tests/solace_security/test_env_consistency.py` is unchanged and still green).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.2: Runtime spec and the PHI-safe logging chain with tracebacks
**Covers:** MVP-180
**Severity:** medium
**Wave:** 0
**Depends on:** WS-04.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Cross-workstream prerequisites:** WS-01 W0 phase A1 lazy `src/solace_infrastructure/__init__.py` (so importing `solace_infrastructure.runtime` does not import Kong, jose or every client); `solace_security.phi_protection.phi_sanitizer_processor` (exists today; WS-09 hardens it in W3 without renaming it).
**Files:**
- Create: `src/solace_infrastructure/runtime/__init__.py` (PEP 562 `__getattr__` mapping `ServiceRuntimeSpec`, `StartupDependencyError` to `.spec`; `ServiceRuntime`, `create_service_app`, `start_service_runtime`, `stop_service_runtime` to `.bootstrap`; `is_strict_environment` to `solace_security.production_guards`; any other name raises `AttributeError`, so `from solace_infrastructure.runtime import bootstrap` still imports the submodule; `__all__` lists exactly the IF-01 names)
- Create: `src/solace_infrastructure/runtime/spec.py` (`ServiceRuntimeSpec`, `StartupDependencyError`, `EVENT_PLANE_MODES = ("none", "publish", "publish_consume")`)
- Create: `src/solace_infrastructure/runtime/logging.py` (`configure_service_logging`, `logging_chain_names`)
- Test: `tests/bootstrap/test_mvp_180_logging.py`
**Interfaces:**
- Consumes: `is_strict_environment` (WS-04.1); `phi_sanitizer_processor(logger_, method_name, event_dict) -> dict` from `solace_security.phi_protection`.
- Produces: `@dataclass(frozen=True) ServiceRuntimeSpec` with exactly the IF-01 fields and defaults (`service_name: str`, `environment: str`, `version: str = "1.0.0"`, `log_level: str = "INFO"`, `debug: bool = False`, `uses_postgres: bool = False`, `uses_redis: bool = False`, `event_plane: Literal["none","publish","publish_consume"] = "none"`, `handles_phi: bool = False`, `audited: bool = False`, `rate_limited: bool = False`, `edge: bool = False`, `docs_enabled: bool | None = None`), whose `__post_init__` raises `ValueError` for an unknown `event_plane`; `class StartupDependencyError(RuntimeError)` with `__init__(self, component: str, reason: str)` and attributes `component`, `reason` (the reason never carries a secret or URL); `configure_service_logging(spec: ServiceRuntimeSpec) -> None`; `logging_chain_names() -> list[str]` (processor names of the active structlog config, used by WS-04.5).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_180_logging.py`
```python
"""MVP-180: logger.exception() must emit the traceback, and the traceback text must pass the PHI sanitizer."""
import dataclasses, json, logging, subprocess, sys
import pytest, structlog
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError
from solace_infrastructure.runtime.logging import configure_service_logging, logging_chain_names

def _spec(env="development", **kw):
    return ServiceRuntimeSpec(service_name="safety-service", environment=env, **kw)

def _last_json(out: str) -> dict:
    return json.loads(out.strip().splitlines()[-1])

def test_exception_log_carries_the_traceback(capsys):
    configure_service_logging(_spec())
    try:
        raise RuntimeError("boom-xyz")
    except RuntimeError:
        structlog.get_logger("t").exception("crisis_handler_failed")
    line = _last_json(capsys.readouterr().out)
    assert line["event"] == "crisis_handler_failed"
    assert "Traceback (most recent call last)" in line["exception"]
    assert "RuntimeError" in line["exception"]
    assert "exc_info" not in line

def test_traceback_text_is_phi_sanitized(capsys):
    configure_service_logging(_spec("production"))
    try:
        raise ValueError("lookup failed for jane.doe@example.com 555-867-5309")
    except ValueError:
        structlog.get_logger("t").exception("lookup_failed")
    out = capsys.readouterr().out
    assert "jane.doe@example.com" not in out
    assert "555-867-5309" not in out

def test_chain_order_puts_exc_info_before_sanitizer_before_renderer():
    configure_service_logging(_spec("production"))
    names = logging_chain_names()
    assert names.index("format_exc_info") < names.index("phi_sanitizer_processor") < names.index("JSONRenderer")
    assert names[:4] == ["merge_contextvars", "add_log_level", "TimeStamper", "StackInfoRenderer"]

def test_strict_requires_the_phi_sanitizer(monkeypatch):
    monkeypatch.setitem(sys.modules, "solace_security.phi_protection", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("staging"))
    assert exc.value.component == "logging"

def test_dev_without_sanitizer_still_configures(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "solace_security.phi_protection", None)
    configure_service_logging(_spec("development"))
    structlog.get_logger("t").info("hello")
    assert _last_json(capsys.readouterr().out)["event"] == "hello"

def test_console_renderer_never_in_strict(capsys):
    configure_service_logging(_spec("production", debug=True))
    structlog.get_logger("t").info("hello")
    assert _last_json(capsys.readouterr().out)["event"] == "hello"

def test_stdlib_and_uvicorn_records_are_json(capsys):
    configure_service_logging(_spec())
    logging.getLogger("uvicorn.error").error("bind failed")
    assert _last_json(capsys.readouterr().out)["event"] == "bind failed"
    assert logging.getLogger("uvicorn.access").disabled is True

def test_spec_is_frozen_and_validated():
    spec = _spec()
    assert (spec.event_plane, spec.docs_enabled, spec.uses_postgres) == ("none", None, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.debug = True
    with pytest.raises(ValueError):
        ServiceRuntimeSpec(service_name="x", environment="development", event_plane="consume")

def test_runtime_package_import_is_light():
    code = "import sys, solace_infrastructure.runtime as r; print('fastapi' in sys.modules, 'asyncpg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False False"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_180_logging.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'solace_infrastructure.runtime'`. There is no shared logging configuration today; each of the ten `configure_logging` functions (e.g. `services/safety_service/src/main.py:59-70`) omits `format_exc_info`, so no existing chain could satisfy `test_exception_log_carries_the_traceback`.
- [ ] **Step 3: Implement** `configure_service_logging(spec: ServiceRuntimeSpec) -> None` in `src/solace_infrastructure/runtime/logging.py`, plus `spec.py` and the lazy `__init__.py`. Approach: processors `merge_contextvars`, `add_log_level`, `TimeStamper(fmt="iso", utc=True)`, `StackInfoRenderer()`, `format_exc_info`, `phi_sanitizer_processor`, then `JSONRenderer()` (`ConsoleRenderer` only when `spec.debug` and not strict); `PrintLoggerFactory()` and `make_filtering_bound_logger(level from spec.log_level)`; the stdlib root logger gets one `StreamHandler(sys.stdout)` with `structlog.stdlib.ProcessorFormatter(foreign_pre_chain=[add_log_level, TimeStamper, StackInfoRenderer, format_exc_info], processors=[ProcessorFormatter.remove_processors_meta, phi_sanitizer_processor, JSONRenderer()])`; `uvicorn`, `uvicorn.error`, `aiokafka` and `asyncpg` loggers propagate to it; `uvicorn.access` is disabled; a repeated call replaces the handler it installed earlier instead of adding a second one. In strict environments an `ImportError` for the sanitizer raises `StartupDependencyError("logging", "PHI log sanitizer unavailable")`; outside strict it logs one WARNING and omits the processor. `logging_chain_names()` returns `type(p).__name__` or `p.__name__` for each configured processor.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`; `UNIT D:\Repo\Contextual-Chatbot\tests\test_syntax_and_imports.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.3: create_service_app with request context, uniform error bodies, health routes and /metrics
**Covers:** MVP-036
**Severity:** high
**Wave:** 0
**Depends on:** WS-04.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-11 (default assumed: error bodies are `{error:{code,message,request_id}}`; FastAPI `HTTPException` bodies keep `{detail}` until WS-04.9 in W2)
**Real infra:** no
**Cross-workstream prerequisites:** WS-07 W0 phase A1 adds `set_correlation_ids`, `get_correlation_headers`, `clear_correlation_ids` to `src/solace_common/request_context.py` (requested below); WS-01 W0 phase A1 `prometheus-client` in the lock.
**Files:**
- Create: `src/solace_infrastructure/runtime/http.py` (pure-ASGI `RequestContextMiddleware`; `install_exception_handlers(app, spec)`; CORS for `spec.edge`)
- Create: `src/solace_infrastructure/runtime/health.py` (`CheckResult`, `ReadinessReport`, `register_health_routes(app)`)
- Create: `src/solace_infrastructure/runtime/metrics.py` (module-level metric objects; `register_metrics_route(app)`)
- Create: `src/solace_infrastructure/runtime/bootstrap.py` (`ServiceRuntime` registry and `create_service_app`; start/stop arrive in WS-04.4)
- Test: `tests/bootstrap/test_mvp_036_service_app.py`
**Interfaces:**
- Consumes: `ServiceRuntimeSpec`, `StartupDependencyError` (WS-04.2); `is_strict_environment` (WS-04.1); `solace_common.request_context.set_correlation_ids(*, request_id: str, correlation_id: str, turn_id: str) -> None`, `get_correlation_headers() -> dict[str, str]`, `clear_correlation_ids() -> None` (WS-07).
- Produces: `create_service_app(spec: ServiceRuntimeSpec, *, lifespan, routers: Sequence[tuple[APIRouter, str]], title: str) -> FastAPI` (sets `app.state.spec = spec`; docs and OpenAPI only when `spec.docs_enabled` is True, or when it is None and the environment is not strict; middleware outer to inner: CORS when `spec.edge`, `RequestContextMiddleware`, then the W1 rate limiter); `class ServiceRuntime` (`__init__(self, spec)`, attributes `spec`, `pg_pool=None`, `redis=None`, `event_plane=None`, `service_tokens=None`, `phi_codec=None`; `register_readiness_check(name: str, check: Callable[[], Awaitable[tuple[bool, str]]], *, critical: bool = True) -> None`; `register_background_task(name: str, task: asyncio.Task) -> None`; `register_shutdown(name: str, closer: Callable[[], Awaitable[None]]) -> None`; `outbound_headers() -> dict[str, str]`; `async evaluate_readiness() -> ReadinessReport`); `@dataclass(frozen=True) CheckResult(ok: bool, critical: bool, detail: str)`; `@dataclass(frozen=True) ReadinessReport(ready: bool, checks: Mapping[str, CheckResult])`; metric objects on `prometheus_client.REGISTRY`: `HTTP_REQUESTS` (`solace_http_requests_total{service,route,method,status}`), `HTTP_DURATION` (`solace_http_request_duration_seconds{service,route}`), `READINESS_CHECK_OK` (`solace_readiness_check_ok{service,check}`), `SERVICE_READY` (`solace_service_ready{service}`, new), `BACKGROUND_TASK_UP` (`solace_background_task_up{service,task}`), `CRISIS_DETECTED` (`solace_crisis_detected_total{source,level}`), `LLM_FALLBACK` (`solace_llm_fallback_total{service,reason}`), `AUDIT_WRITE_FAILURES` (`solace_audit_write_failures_total{service}`).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_036_service_app.py`
```python
"""MVP-036 (part 1): every service serves /metrics; /live and /ready tell the truth; error bodies never leak exception text."""
import asyncio, json
from contextlib import asynccontextmanager
import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError, create_service_app
from solace_infrastructure.runtime.bootstrap import ServiceRuntime

router = APIRouter()

class EchoIn(BaseModel):
    message: str

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

def test_live_reports_a_dead_background_task():
    with TestClient(_app()) as c:
        # c.portal is the TestClient's anyio portal into the app's event loop
        loop_task = c.portal.call(lambda: asyncio.ensure_future(asyncio.sleep(0)))
        c.portal.call(asyncio.sleep, 0.05)
        c.app.state.runtime.register_background_task("consumer", loop_task)
        r = c.get("/live")
    assert r.status_code == 503 and r.json() == {"status": "dead", "task": "consumer"}

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

def test_validation_error_shape_has_no_input_echo():
    with TestClient(_app()) as c:
        r = c.post("/api/v1/x/echo", json={"message": 7, "x": "jane@example.com"}, headers={"X-Request-ID": "req-v"})
    body = r.json()["error"]
    assert r.status_code == 422 and body["code"] == "VALIDATION_ERROR" and body["request_id"] == "req-v"
    assert set(body["details"][0]) == {"field", "message", "type"}
    assert "jane@example.com" not in r.text

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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_service_app.py`
Expected: FAIL with `ImportError: cannot import name 'create_service_app'` (lazy export resolves to the missing `bootstrap` module). This is the MVP-036 gap: no shared app factory exists, and 9 of 10 services register no `/metrics` route today (the WS-04.5 conformance test proves it per service).
- [ ] **Step 3: Implement** `create_service_app(spec, *, lifespan, routers, title) -> FastAPI` and `ServiceRuntime` in `bootstrap.py`, `RequestContextMiddleware` and `install_exception_handlers(app, spec)` in `http.py`, `register_health_routes(app)` in `health.py`, `register_metrics_route(app)` and the metric objects in `metrics.py`. Approach: the middleware is pure ASGI (not `BaseHTTPMiddleware`) so its context is visible to the 500 handler; it accepts inbound ids matching `^[A-Za-z0-9._:-]{1,128}$` and mints `uuid4()` otherwise, stores them on `scope["state"]`, calls `set_correlation_ids(...)`, binds them into structlog contextvars, adds them to response headers, records the HTTP metrics with the route template from `scope["route"].path` (or `"__unmatched__"`), writes one `http_request` line (method, route, status, duration_ms; DEBUG for `/live`, `/ready`, `/health`, `/metrics`), and calls `clear_correlation_ids()` at the end. Handlers read `request.state.request_id`; detail strings in readiness results are the exception class name only. `evaluate_readiness` runs every check concurrently with a 2 s timeout each, caches the report for `READINESS_CACHE_SECONDS = 2.0` (registering a check invalidates the cache), and sets `READINESS_CHECK_OK` and `SERVICE_READY`. `/live` returns 503 `{"status":"dead","task":name}` for the first registered task whose `done()` is true and sets `BACKGROUND_TASK_UP`. `/metrics` renders `generate_latest(REGISTRY)`. Edge CORS reads `CORS_ALLOWED_ORIGINS` (comma-separated); in strict an empty value or `*` raises `StartupDependencyError("cors", ...)`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.4: start_service_runtime and stop_service_runtime with datastores and ordered teardown
**Covers:** MVP-332
**Severity:** low
**Wave:** 0
**Depends on:** WS-04.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: one 'solace' database, services connect only as solace_app through the single 'default' pool), D-14 (default assumed: strict = staging or production)
**Real infra:** no (fakes here; the real proof is WS-05's W0 boot smoke in dev and strict, run at the W0 barrier)
**Cross-workstream prerequisites:** WS-02 W0 phase A2 `ConnectionPoolManager.get_pool("default")` from `POSTGRES_*` with the strict `DatabaseConfigError` policy and `solace_infrastructure.database.schema_check.assert_schema_head(pool)`.
**Files:**
- Create: `src/solace_infrastructure/runtime/datastores.py` (`start_postgres`, `start_redis`, `_schema_head_check`)
- Modify: `src/solace_infrastructure/runtime/bootstrap.py` (add `start_service_runtime`, `stop_service_runtime`, `required_secrets_for`, `SHUTDOWN_STEP_TIMEOUT_SECONDS = 5.0`, `READINESS_REFRESH_SECONDS = 15.0`)
- Modify: `tests/bootstrap/conftest.py` (add the shared `calls` and `fakes` fixtures shown below; they import `solace_infrastructure.runtime.bootstrap`/`datastores` inside the fixture body so collection never breaks before this task)
- Test: `tests/bootstrap/test_mvp_332_runtime_lifecycle.py`
**Interfaces:**
- Consumes: `ProductionGuards.assert_startup(..., required_vars=...)` (WS-04.1); `configure_service_logging` (WS-04.2); `ServiceRuntime` (WS-04.3); `ConnectionPoolManager.get_pool("default")`, `ConnectionPoolManager.close_all_pools()`, `assert_schema_head(pool)` (WS-02); `redis.asyncio.from_url`.
- Produces: `async start_service_runtime(app: FastAPI, spec: ServiceRuntimeSpec) -> ServiceRuntime` (order: guards with `required_vars=required_secrets_for(spec)`, logging, Postgres pool plus strict schema-head check when `uses_postgres`, Redis from `REDIS_URL` when `uses_redis`, readiness refresher task; sets `app.state.runtime`); `async stop_service_runtime(runtime: ServiceRuntime) -> None` (closers in reverse registration order, `SHUTDOWN_STEP_TIMEOUT_SECONDS` each, failures logged by exception class and skipped); `required_secrets_for(spec: ServiceRuntimeSpec) -> tuple[str, ...]` (`AUTH_SECRET_KEY`, plus `POSTGRES_PASSWORD` if `uses_postgres`, `REDIS_URL` if `uses_redis`, `ENCRYPTION_MASTER_KEY` and `ENCRYPTION_SEARCH_HASH_SALT` if `handles_phi`); readiness checks `postgres` and `redis` (critical).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_332_runtime_lifecycle.py`, with the shared fixtures in `tests/bootstrap/conftest.py`
```python
# tests/bootstrap/conftest.py (added by this task)
@pytest.fixture
def calls():
    return []

@pytest.fixture
def fakes(monkeypatch, calls):
    from solace_infrastructure.runtime import bootstrap as bs, datastores as ds
    class Pool:
        pass
    pool = Pool()
    async def get_pool(name="default"):
        calls.append(f"get_pool:{name}")
        return pool
    async def close_all():
        calls.append("pg.close_all")
    async def head(p):
        calls.append("schema_head")
    class Redis:
        async def ping(self):
            calls.append("redis.ping"); return True
        async def aclose(self):
            calls.append("redis.aclose")
    monkeypatch.setattr(ds.ConnectionPoolManager, "get_pool", get_pool)
    monkeypatch.setattr(ds.ConnectionPoolManager, "close_all_pools", close_all)
    monkeypatch.setattr(ds, "_schema_head_check", head)
    monkeypatch.setattr(ds.redis_asyncio, "from_url", lambda url, **kw: Redis())
    monkeypatch.setenv("REDIS_URL", "redis://:p4ss@redis:6379/0")
    monkeypatch.setattr(bs.ProductionGuards, "assert_startup",
                        classmethod(lambda cls, env, **kw: calls.append(("guards", tuple(sorted(kw["required_vars"]))))))
    monkeypatch.setattr(bs, "configure_service_logging", lambda spec: calls.append("logging"))
    return pool
```
```python
# tests/bootstrap/test_mvp_332_runtime_lifecycle.py
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
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_332_runtime_lifecycle.py`
Expected: FAIL with `ImportError: cannot import name 'start_service_runtime'` and `ModuleNotFoundError: ... datastores`. This is the MVP-332 gap: no lifespan closes its Redis/httpx/pg resources (`services/user-service/src/main.py:451-460`, `services/orchestrator_service/src/main.py:209-218`) and `ConnectionPoolManager.close_all_pools` has no caller.
- [ ] **Step 3: Implement** `start_service_runtime`, `stop_service_runtime`, `required_secrets_for` in `bootstrap.py` and `start_postgres(spec, runtime) -> asyncpg.Pool | None`, `start_redis(spec, runtime) -> Redis | None` in `datastores.py`. Approach: each component start is wrapped once; in strict any exception becomes `StartupDependencyError(component, type(exc).__name__)` raised out of the lifespan (uvicorn exits non-zero); outside strict it logs one WARNING `runtime_component_degraded` with `component=<name>` and `error=<exception class name>` (never the exception text) and registers a failing readiness check for that component. `_schema_head_check(pool)` imports `assert_schema_head` from `solace_infrastructure.database.schema_check` at call time and is called only in strict. `datastores.py` imports `redis.asyncio as redis_asyncio` at module level so tests can patch `from_url`. Closers registered here: `redis.aclose`, `ConnectionPoolManager.close_all_pools`, and cancellation of the readiness refresher; the refresher is itself a registered background task, so its death shows on `/live`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.5: Service conformance suite and a corrected Prometheus scrape config
**Covers:** MVP-036
**Severity:** high
**Wave:** 0
**Depends on:** WS-04.4
**Implementer model:** opus
**Review:** standard
**Decision gate:** D-19 (default assumed: config-service is not an MVP service and is not scraped), D-18 (default assumed: analytics is a non-critical staff dashboard with no PHI)
**Real infra:** yes (`promtool check config` in the pinned Prometheus image)
**Cross-workstream prerequisites:** none to write it; the nine `test_service_conforms[...]` parameters turn green only as the W0 phase-B adoption tasks of WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17 and WS-18 land (requested below).
**Files:**
- Create: `tests/bootstrap/test_service_conformance.py`, `tests/bootstrap/_conformance_probe.py`, `tests/bootstrap/fixtures/reference_app.py`, `tests/bootstrap/fixtures/nonconforming_app.py`, `tests/bootstrap/test_mvp_036_prometheus_scrape.py`
- Modify: `infrastructure/prometheus/prometheus.yml` (one job `solace-services`, nine static targets each labelled `service`, `metrics_path: /metrics`; config-service:8010 and the exporter-less postgres/redis/kafka jobs removed)
- Test: `tests/bootstrap/test_service_conformance.py`, `tests/bootstrap/test_mvp_036_prometheus_scrape.py`
**Interfaces:**
- Consumes: IF-01 names (WS-04.2 to WS-04.4); `logging_chain_names()`.
- Produces: the conformance table `SERVICES` (service name, sys.path entries, import module, `main.py` path, expected SPEC flags) used as each adoption task's acceptance (`-k <service>`); the Prometheus job name `solace-services` with label `service` (consumed by WS-04.7 rules).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_service_conformance.py` and `tests/bootstrap/test_mvp_036_prometheus_scrape.py`
```python
# tests/bootstrap/test_service_conformance.py
"""MVP-036/IF-01: each of the nine MVP services builds its app with create_service_app, declares SPEC,
serves /live /ready /health /metrics from the runtime, and brackets its lifespan with the runtime."""
import ast, json, os, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name("_conformance_probe.py")
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

def _probe(paths, module) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "HOME") if k in os.environ}
    env.update(_probe_env(), PYTHONPATH=os.pathsep.join(str(p) for p in paths), ENVIRONMENT="development")
    out = subprocess.run([sys.executable, str(PROBE), module], cwd=REPO, env=env, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])

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

@pytest.mark.parametrize("service", sorted(SERVICES))
def test_lifespan_is_bracketed_by_the_runtime(service):
    tree = ast.parse((REPO / SERVICES[service].main_py).read_text(encoding="utf-8"))
    src = ast.unparse(tree)
    assert "structlog.configure(" not in src and "ProductionGuards.assert_startup(" not in src
    assert "exception_handler" not in src
    lifespan = _lifespan_function(tree)
    assert ast.unparse(lifespan.body[0]) == "runtime = await start_service_runtime(app, SPEC)"
    assert _last_statement(lifespan) == "await stop_service_runtime(runtime)"

def test_probe_accepts_the_reference_app():
    report = _probe((Path(__file__).parent / "fixtures", REPO, REPO / "src"), "reference_app")
    assert report["has_spec"] and report["created_by_runtime"] and report["foreign_exception_handlers"] == []

def test_probe_rejects_a_nonconforming_app():
    report = _probe((Path(__file__).parent / "fixtures", REPO, REPO / "src"), "nonconforming_app")
    assert report["created_by_runtime"] is False and report["foreign_exception_handlers"] != []
```
```python
# tests/bootstrap/test_mvp_036_prometheus_scrape.py
"""MVP-036: Prometheus scrapes the nine MVP services on their real ports and nothing that has no exporter."""
import subprocess
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {"orchestrator-service:8000": "orchestrator-service", "user-service:8001": "user-service",
            "safety-service:8002": "safety-service", "notification-service:8003": "notification-service",
            "diagnosis-service:8004": "diagnosis-service", "memory-service:8005": "memory-service",
            "therapy-service:8006": "therapy-service", "personality-service:8007": "personality-service",
            "analytics-service:8009": "analytics-service"}

def _cfg():
    return yaml.safe_load((REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8"))

def test_one_job_scrapes_the_nine_services():
    jobs = {j["job_name"]: j for j in _cfg()["scrape_configs"]}
    job = jobs["solace-services"]
    targets = {t: sc["labels"]["service"] for sc in job["static_configs"] for t in sc["targets"]}
    assert targets == EXPECTED
    assert job.get("metrics_path", "/metrics") == "/metrics"

def test_no_dead_jobs():
    names = {j["job_name"] for j in _cfg()["scrape_configs"]}
    assert names.isdisjoint({"config-service", "postgres", "redis", "kafka"})
    assert "8010" not in (REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8")

@pytest.mark.integration
@pytest.mark.realinfra
def test_promtool_accepts_the_config():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'prometheus'}:/etc/prometheus:ro",
                          "--entrypoint", "promtool", "prom/prometheus:v2.47.0", "check", "config", "/etc/prometheus/prometheus.yml"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
```
`_conformance_probe.py` imports the module named in `argv[1]`, reads `module.SPEC` (if present) and `module.app`, and prints one JSON line: `has_spec`, `spec` (`dataclasses.asdict(SPEC)` or `{}`), `created_by_runtime` (`getattr(app.state, "spec", None) is SPEC`), `runtime_routes` (paths whose endpoint `__module__` starts with `solace_infrastructure.runtime`), `foreign_exception_handlers` (qualified names of handlers in `app.exception_handlers` whose `__module__` is not `solace_infrastructure.runtime.http`, excluding FastAPI's own defaults). `_probe_env()` returns synthetic, per-run values for every variable a service's settings require at import: `AUTH_SECRET_KEY`, `USER_SERVICE_JWT_SECRET_KEY`, `SECURITY_JWT_SECRET`, `USER_SERVICE_FIELD_ENCRYPTION_KEY` (`secrets.token_urlsafe(48)`), `USER_DB_PASSWORD` and `POSTGRES_PASSWORD` (`"probe_db_credential_only"`), `FERNET_TOKEN_KEY`/`FERNET_FIELD_KEY` (`Fernet.generate_key()`), `ENCRYPTION_MASTER_KEY` (`token_hex(32)`), `ENCRYPTION_SEARCH_HASH_SALT`, `PERSONALITY_REDIS_PASSWORD`, plus anything else the services' `tests/conftest.py` files set (read them). `_lifespan_function(tree)` finds the function passed as `lifespan=` to `create_service_app`; `_last_statement(fn)` returns the unparsed last top-level statement, or the last statement of a trailing `try/finally` block's `finalbody`. `fixtures/reference_app.py` builds `SPEC` and `app = create_service_app(SPEC, lifespan=..., routers=[], title="ref")` with the bracketed lifespan; `fixtures/nonconforming_app.py` builds a plain `FastAPI()` with its own `@app.exception_handler(Exception)`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_prometheus_scrape.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_service_conformance.py`
Expected: FAIL. Scrape tests: `KeyError: 'solace-services'` and `config-service` present (prometheus.yml:46-63 scrapes config-service:8010 and exporter-less postgres/redis/kafka). Conformance: every `test_service_conforms[...]` fails with "main.py must define SPEC" and every `test_lifespan_is_bracketed_by_the_runtime[...]` fails on `structlog.configure(`; the two probe-control tests fail until the fixtures exist. The nine service parameters stay RED by design at the end of this task (they are the W0 phase-B acceptance gate).
- [ ] **Step 3: Implement** the probe, fixtures and the scrape config. Approach: one `solace-services` job with one `static_configs` entry per target carrying `labels: {service: <name>}`; global `scrape_interval: 15s`, `evaluation_interval: 15s`; no rule files yet (WS-04.7 adds them). The probe never prints environment values or stack traces of the imported module beyond the last 2000 characters of stderr.
- [ ] **Step 4: Run it and confirm it passes**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_prometheus_scrape.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_service_conformance.py -k "probe"`; `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_prometheus_scrape.py`. Expected: PASS (the nine service parameters remain RED until phase B; record that in the ledger).
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap -k "not test_service_conforms and not test_lifespan_is_bracketed_by_the_runtime"`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`. Expected: all pass. Design gate: before any W0 phase-B adoption task is dispatched, the controller runs the AGY adversarial review over WS-04.1 to WS-04.5; this checkpoint is its input.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.6: Wire service tokens, the event plane and rate limiting into the bootstrap
**Covers:** MVP-332
**Severity:** low
**Wave:** 1
**Depends on:** WS-04.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: HS256 service JWTs from `ServiceTokenManager` in W1-W3), D-26 (default assumed: app-level Redis limiter; auth routes fail closed on Redis failure)
**Real infra:** yes (BOOTSMOKE in dev and strict)
**Cross-workstream prerequisites:** WS-06 W1 phase A1 `solace_events.service_plane.start_service_event_plane` and `ServiceEventPlane.readiness()/stop()` (IF-05; `stop()` finishes the in-flight flush and releases claims); WS-07 W1 phase A1 `solace_security.rate_limit.install_rate_limiting(app, spec)` and `ServiceTokenManager`/`ServiceAuthSettings` (IF-04); WS-05's boot smoke.
**Files:**
- Modify: `src/solace_infrastructure/runtime/bootstrap.py` (step 4 of `start_service_runtime`; `create_service_app` calls `install_rate_limiting(app, spec)` when `spec.rate_limited`)
- Test: `tests/bootstrap/test_mvp_332_w1_wiring.py`
**Interfaces:**
- Consumes: `start_service_event_plane(service_name: str, *, pool: asyncpg.Pool | None, kafka_settings: KafkaSettings | None = None, strict: bool) -> ServiceEventPlane` (IF-05); `ServiceTokenManager(service_settings=ServiceAuthSettings(service_name=...))` and `.create_service_token(service_name)` (IF-04); `install_rate_limiting(app, spec)` (IF-04).
- Produces: `ServiceRuntime.service_tokens` (every service), `ServiceRuntime.event_plane` (when `event_plane != "none"`), readiness checks `service_tokens` and `event_plane` (critical); closer order guarantees `event_plane.stop()` runs before the Postgres pool closes.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_332_w1_wiring.py`
```python
"""MVP-332 (event plane): the plane is started by the runtime, gated by SPEC, and stopped before the pool closes."""
import pytest
from fastapi import FastAPI
from solace_infrastructure.runtime import (ServiceRuntimeSpec, StartupDependencyError,
                                          create_service_app, start_service_runtime, stop_service_runtime)
from solace_infrastructure.runtime import bootstrap as bs
# `calls` and `fakes` come from tests/bootstrap/conftest.py (WS-04.4)

class Plane:
    def __init__(self, calls):
        self.calls = calls
    def readiness(self):
        return True, "connected"
    async def stop(self):
        self.calls.append("plane.stop")

def _spec(env="staging", **kw):
    base = dict(uses_postgres=True, uses_redis=True, event_plane="publish_consume")
    base.update(kw)
    return ServiceRuntimeSpec(service_name="notification-service", environment=env, **base)

async def test_event_plane_started_with_pool_and_strictness(fakes, calls, monkeypatch):
    seen = {}
    async def fake_start(service_name, *, pool, kafka_settings=None, strict):
        seen.update(service_name=service_name, pool=pool, strict=strict)
        return Plane(calls)
    monkeypatch.setattr(bs, "start_service_event_plane", fake_start)
    rt = await start_service_runtime(FastAPI(), _spec())
    assert seen == {"service_name": "notification-service", "pool": fakes, "strict": True}
    report = await rt.evaluate_readiness()
    assert report.checks["event_plane"].critical and report.checks["event_plane"].ok
    await stop_service_runtime(rt)
    assert calls.index("plane.stop") < calls.index("pg.close_all")

async def test_no_event_plane_when_spec_says_none(fakes, monkeypatch):
    monkeypatch.setattr(bs, "start_service_event_plane", pytest.fail)
    rt = await start_service_runtime(FastAPI(), _spec(event_plane="none"))
    assert rt.event_plane is None
    await stop_service_runtime(rt)

async def test_strict_event_plane_failure_is_fatal(fakes, monkeypatch):
    async def boom(*a, **k):
        raise ConnectionError("kafka down")
    monkeypatch.setattr(bs, "start_service_event_plane", boom)
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("production"))
    assert exc.value.component == "event_plane"

async def test_dev_event_plane_failure_degrades_ready(fakes, monkeypatch):
    async def boom(*a, **k):
        raise ConnectionError("kafka down")
    monkeypatch.setattr(bs, "start_service_event_plane", boom)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    assert (await rt.evaluate_readiness()).checks["event_plane"].ok is False
    await stop_service_runtime(rt)

async def test_service_tokens_are_minted_at_startup(fakes, monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "x" * 48)
    monkeypatch.setattr(bs, "start_service_event_plane", lambda *a, **k: None)
    rt = await start_service_runtime(FastAPI(), _spec("development", event_plane="none"))
    assert rt.service_tokens is not None
    assert rt.service_tokens.create_service_token("notification-service").token
    await stop_service_runtime(rt)

def test_rate_limiting_installed_only_when_flagged(monkeypatch):
    seen = []
    monkeypatch.setattr(bs, "install_rate_limiting", lambda app, spec: seen.append(spec.service_name))
    for flag in (True, False):
        spec = ServiceRuntimeSpec(service_name="user-service", environment="development", rate_limited=flag)
        create_service_app(spec, lifespan=None, routers=[], title="t")
    assert seen == ["user-service"]
```
(`ServiceCredentials.token` is the existing attribute in `src/solace_security/service_auth.py`.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_332_w1_wiring.py`
Expected: FAIL with `AttributeError: module 'solace_infrastructure.runtime.bootstrap' has no attribute 'start_service_event_plane'` (and `install_rate_limiting`); the plane is not started, not checked and not stopped by any runtime, which leaves `OutboxPoller.stop` cancelling mid-flush with nothing ordering it before the pool close (MVP-332).
- [ ] **Step 3: Implement** step 4 of `start_service_runtime` in `bootstrap.py`. Approach: module-level imports `from solace_events.service_plane import start_service_event_plane`, `from solace_security.rate_limit import install_rate_limiting`, `from solace_security.service_auth import ServiceAuthSettings, ServiceTokenManager`; the token manager is built for every service and proven by minting one token (readiness `service_tokens`); the plane gets `pool=runtime.pg_pool` and `strict=is_strict_environment(spec.environment)`, its `readiness` is wrapped as an async check, and `plane.stop` is registered as a closer after the pool closer is registered (reverse order then stops the plane first). The rate limiter is installed inside `create_service_app` between CORS and the request-context middleware in the outer-to-inner order fixed in WS-04.3.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_events`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_security`; then BOOTSMOKE. Expected: all pass; in strict no container log contains an in-memory or localhost fallback warning.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.7: Alertmanager and the crisis-path alert rules, promtool-tested
**Covers:** MVP-036
**Severity:** high
**Wave:** 1
**Depends on:** WS-04.5, WS-03.6
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-12 (default assumed: email through an SMTP relay with STARTTLS; mailpit in tests)
**Real infra:** yes (promtool and amtool in pinned images; e2e delivery through the IF-19 stack)
**Cross-workstream prerequisites:** WS-03.6 registers the `ALERTMANAGER_*` variables and mounts `./infrastructure/prometheus` at `/etc/prometheus` in the base compose; WS-06 W1 metrics `solace_outbox_oldest_pending_seconds{service}`, `solace_consumer_lag{group,topic}` (from WS-06.10 sampled by every plane for the crisis pair, crisis-review F13), `solace_dlq_unresolved{group}`, `solace_event_publish_failures_total{service}`; WS-13 W1 metrics `solace_crisis_alert_delivery_total{outcome,channel}` (outcome values requested below) and `solace_crisis_alerts_unacknowledged`; WS-05 includes this fragment in the e2e stack and points Alertmanager at mailpit (requested below).
**Files:**
- Create: `docker-compose.observability.yml` (service `alertmanager`: image `prom/alertmanager:v0.27.0`, entrypoint `["/bin/sh", "/etc/alertmanager-src/render-config.sh"]`, volume `./infrastructure/alertmanager:/etc/alertmanager-src:ro` and named volume `alertmanager_data:/alertmanager`, tmpfs `/run/alertmanager`, environment from the six `ALERTMANAGER_*` registry variables with `${ALERTMANAGER_EMAIL_TO:?...}` and `${ALERTMANAGER_SMTP_SMARTHOST:?...}` (no default, so the fragment can never page a placeholder address), network `monitoring`, no published port; no base service is redefined)
- Create: `infrastructure/prometheus/rules/solace.yml`, `infrastructure/prometheus/tests/solace_rules_test.yml`, `infrastructure/alertmanager/alertmanager.yml.tmpl`, `infrastructure/alertmanager/render-config.sh`, `docs/runbooks/alerting.md`
- Modify: `infrastructure/prometheus/prometheus.yml` (`rule_files: ["/etc/prometheus/rules/*.yml"]`; `alerting.alertmanagers` → `alertmanager:9093`)
- Test: `tests/bootstrap/test_mvp_036_alert_rules.py`, `tests/e2e/test_mvp_036_alert_delivery.py`
**Interfaces:**
- Consumes: IF-05 and IF-09 metric names above; `solace_service_ready{service}` and `up{job="solace-services"}` (WS-04.3, WS-04.5).
- Produces: alerts `CrisisAlertDeliveryFailed` (`sum by (channel) (increase(solace_crisis_alert_delivery_total{outcome="failed"}[5m])) > 0`), `CrisisOutboxBacklog` (`max by (service) (solace_outbox_oldest_pending_seconds) > 120` for 1m), `NotificationConsumerLag` (`max by (group, topic) (solace_consumer_lag{group="solace-group-notifications"}) > 10` for 5m; `max`, not `sum`, because from WS-06.10 every event plane also samples the crisis group's lag, crisis-review F13), `CrisisConsumerStalled` (`max by (group, topic) (solace_consumer_lag{group="solace-group-notifications",topic="solace.safety"}) > 0` for 3m, crisis-review F13: one stuck crisis turn is only 4 messages, so the volume rule above never sees a consumer that is alive but makes no progress), `DLQUnresolved` (`sum by (group) (solace_dlq_unresolved) > 0` for 1m), `CrisisAlertsUnacknowledged` (`max(solace_crisis_alerts_unacknowledged) > 0` for 20m), `CrisisEventPublishFailures` (`sum by (service) (increase(solace_event_publish_failures_total[5m])) > 0`), `ServiceNotReady` (`(up{job="solace-services"} == 0) or (solace_service_ready == 0)` for 2m); every rule has `labels.severity: page`, crisis-path rules `labels.path: crisis`, and `annotations.runbook_url` pointing at `docs/runbooks/alerting.md#<lower-case alert name>`; Alertmanager routes everything to `ALERTMANAGER_EMAIL_TO`, grouping by `alertname` and `service`, `group_wait 30s`, `group_interval 5m`, `repeat_interval 1h` for `path="crisis"` and `4h` otherwise, `send_resolved: true`. `docs/runbooks/alerting.md` is created and edited only by WS-04 (owned path); WS-13.10's `docs/runbooks/crisis-escalation.md` links to it read-only for the WS-04 alerts, which this task creates in W1 phase A, ahead of WS-13.10 in phase B.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_036_alert_rules.py` and `tests/e2e/test_mvp_036_alert_delivery.py`
```python
# tests/bootstrap/test_mvp_036_alert_rules.py
"""MVP-036 (part 2): crisis-path alert rules exist, reference real metric names, carry no user data,
fire under promtool series fixtures, and Alertmanager routes them to the ops mailbox."""
import re, subprocess
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[2]
RULES = REPO / "infrastructure/prometheus/rules/solace.yml"
W1_ALERTS = {"CrisisAlertDeliveryFailed": "solace_crisis_alert_delivery_total", "CrisisOutboxBacklog": "solace_outbox_oldest_pending_seconds",
             "NotificationConsumerLag": "solace_consumer_lag", "CrisisConsumerStalled": "solace_consumer_lag", "DLQUnresolved": "solace_dlq_unresolved",
             "CrisisAlertsUnacknowledged": "solace_crisis_alerts_unacknowledged", "CrisisEventPublishFailures": "solace_event_publish_failures_total",
             "ServiceNotReady": "solace_service_ready"}

def _rules():
    return {r["alert"]: r for g in yaml.safe_load(RULES.read_text(encoding="utf-8"))["groups"] for r in g["rules"]}

def test_every_w1_alert_exists_and_uses_its_metric():
    rules = _rules()
    for alert, metric in W1_ALERTS.items():
        assert metric in rules[alert]["expr"], alert
        assert rules[alert]["labels"]["severity"] == "page"
    assert 'up{job="solace-services"} == 0' in rules["ServiceNotReady"]["expr"]

def test_crisis_review_f13_lag_rules_use_max_and_the_stall_rule_fires_on_any_lag():
    rules = _rules()
    for alert in ("NotificationConsumerLag", "CrisisConsumerStalled"):
        assert rules[alert]["expr"].lstrip().startswith("max by (group, topic)"), alert   # several planes export the pair
    stalled = rules["CrisisConsumerStalled"]
    assert 'topic="solace.safety"' in stalled["expr"] and stalled["expr"].rstrip().endswith("> 0")
    assert stalled["for"] == "3m" and stalled["labels"]["path"] == "crisis"

def test_annotations_never_template_user_data():
    for alert, rule in _rules().items():
        text = yaml.safe_dump(rule.get("annotations", {}))
        assert not re.search(r"\$labels\.(user_id|email|message|session_id)", text), alert

def test_every_alert_has_a_runbook_section():
    doc = (REPO / "docs/runbooks/alerting.md").read_text(encoding="utf-8").lower()
    for alert, rule in _rules().items():
        anchor = rule["annotations"]["runbook_url"].split("#")[1]
        assert anchor == alert.lower() and f"## {alert.lower()}" in doc, alert

def test_prometheus_loads_rules_and_alertmanager():
    cfg = yaml.safe_load((REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8"))
    assert "/etc/prometheus/rules/*.yml" in cfg["rule_files"]
    assert cfg["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"] == ["alertmanager:9093"]

def test_fragment_adds_alertmanager_without_redefining_base_services():
    frag = yaml.safe_load((REPO / "docker-compose.observability.yml").read_text(encoding="utf-8"))
    base = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    am = frag["services"]["alertmanager"]
    assert am["image"] == "prom/alertmanager:v0.27.0" and "ports" not in am
    for name, svc in frag["services"].items():
        if name in base["services"]:
            assert "image" not in svc and "build" not in svc, name

@pytest.mark.integration
@pytest.mark.realinfra
def test_promtool_rule_unit_tests_pass():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'prometheus'}:/etc/prometheus:ro",
                          "--entrypoint", "promtool", "prom/prometheus:v2.47.0", "test", "rules", "/etc/prometheus/tests/solace_rules_test.yml"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr

@pytest.mark.integration
@pytest.mark.realinfra
def test_rendered_alertmanager_config_is_valid(tmp_path):
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'alertmanager'}:/etc/alertmanager-src:ro",
                          "-e", "ALERTMANAGER_EMAIL_TO=ops@example.test", "-e", "ALERTMANAGER_EMAIL_FROM=alerts@example.test",
                          "-e", "ALERTMANAGER_SMTP_SMARTHOST=smtp.example.test:587", "-e", "ALERTMANAGER_SMTP_REQUIRE_TLS=true",
                          "--entrypoint", "/bin/sh", "prom/alertmanager:v0.27.0", "-c",
                          "RENDER_ONLY=1 /bin/sh /etc/alertmanager-src/render-config.sh && amtool check-config /run/alertmanager/alertmanager.yml"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr

@pytest.mark.integration
@pytest.mark.realinfra
def test_render_refuses_missing_recipient():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'alertmanager'}:/etc/alertmanager-src:ro",
                          "--entrypoint", "/bin/sh", "prom/alertmanager:v0.27.0", "-c", "RENDER_ONLY=1 /bin/sh /etc/alertmanager-src/render-config.sh"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 64 and "ALERTMANAGER_EMAIL_TO" in out.stderr
```
```python
# tests/e2e/test_mvp_036_alert_delivery.py
"""MVP-036: stopping notification-service pages the ops mailbox via Prometheus -> Alertmanager -> SMTP, with no PHI."""
import re, time
import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

def test_service_not_ready_reaches_mailpit(e2e_stack, fault):
    fault.stop("notification-service")
    try:
        deadline = time.monotonic() + 420
        hit = None
        while time.monotonic() < deadline and hit is None:
            msgs = httpx.get(f"{e2e_stack.mailpit_api}/api/v1/messages", timeout=10).json()["messages"]
            hit = next((m for m in msgs if "ServiceNotReady" in m["Subject"]), None)
            time.sleep(10)
        assert hit is not None, "ServiceNotReady alert email not delivered within 7 minutes"
        body = httpx.get(f"{e2e_stack.mailpit_api}/api/v1/message/{hit['ID']}", timeout=10).json()["Text"]
        assert "notification-service" in body
        emails = set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", body))
        assert emails <= {"ops-alerts@test.invalid", "alerts@test.invalid"}
    finally:
        fault.start("notification-service")
```
(`fault` is the IF-19 fault helper; if WS-05 exposes it as an attribute of `e2e_stack` instead of a fixture, the implementer uses that form.) `solace_rules_test.yml` contains, for each W1 alert, one firing case with input series at the threshold plus margin and the alert's `for` duration elapsed, and one non-firing case (`outcome="delivered"` increments, lag 10, oldest pending 119 s, `solace_service_ready` 1 with `up` 1). For `CrisisConsumerStalled` (crisis-review F13) the firing case is lag 4 held for 3 minutes, exported by two series that differ only in `service` (two planes sampling the same pair), and the non-firing case is lag 4 for one evaluation, then 0; the `NotificationConsumerLag` non-firing case also has lag 6 from each of two such series (which `sum` would push over 10). `render-config.sh` exits 64 with the missing variable's name on stderr when `ALERTMANAGER_EMAIL_TO` or `ALERTMANAGER_SMTP_SMARTHOST` is unset, writes `/run/alertmanager/alertmanager.yml` with `sed` substitutions, omits the auth lines when `ALERTMANAGER_SMTP_USERNAME` is empty, writes the SMTP password to `/run/alertmanager/smtp_password` and references it with `smtp_auth_password_file`, and execs `/bin/alertmanager --config.file=/run/alertmanager/alertmanager.yml --storage.path=/alertmanager` unless `RENDER_ONLY=1`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_alert_rules.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_alert_rules.py`
Expected: FAIL with `FileNotFoundError` for `infrastructure/prometheus/rules/solace.yml` and `docker-compose.observability.yml`, and `KeyError: 'rule_files'` (MVP-036: prometheus.yml has no rule_files or alerting block and no Alertmanager exists anywhere). The e2e test (`$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_mvp_036_alert_delivery.py`) fails with "ServiceNotReady alert email not delivered within 7 minutes".
- [ ] **Step 3: Implement** the rules, promtool tests, Alertmanager template and render script, the fragment and the runbook. Approach: rule expressions exactly as listed under Produces; the email template uses Alertmanager's default text body plus a subject `[Solace ALERT {{ .Status | toUpper }}] {{ .CommonLabels.alertname }} {{ .CommonLabels.service }}`; labels only ever hold service, group, topic, channel and store names; the runbook has one `## <alert name>` section per alert with meaning, first checks and escalation.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2, then the e2e command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy` (the strict render with the observability fragment stays registry-clean); BOOTSMOKE with the observability fragment. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.8: PHI codec warm-up in the bootstrap
**Covers:** none (IF-01 wiring for IF-13; supports MVP-055, owned by WS-09)
**Severity:** high
**Wave:** 2
**Depends on:** WS-04.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes (BOOTSMOKE in dev and strict)
**Cross-workstream prerequisites:** WS-09 W2 phase A `solace_security.phi_codec.get_phi_codec()` and `PHIConfigError` (IF-13).
**Files:**
- Modify: `src/solace_infrastructure/runtime/bootstrap.py` (step 5: `get_phi_codec()` when `spec.handles_phi`; readiness `phi_codec`)
- Test: `tests/bootstrap/test_mvp_055_codec_warmup.py`
**Interfaces:**
- Consumes: `get_phi_codec() -> PHICodec`, `PHIConfigError` (IF-13).
- Produces: `ServiceRuntime.phi_codec`; readiness check `phi_codec` (critical).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_055_codec_warmup.py`
```python
"""Supports MVP-055: PHI services load the codec at startup; strict key errors are fatal."""
import pytest
from fastapi import FastAPI
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError, start_service_runtime, stop_service_runtime
from solace_infrastructure.runtime import bootstrap as bs
# `calls` and `fakes` come from tests/bootstrap/conftest.py (WS-04.4)

def _spec(env, phi=True):
    return ServiceRuntimeSpec(service_name="therapy-service", environment=env, uses_postgres=True, uses_redis=True, handles_phi=phi)

async def test_codec_loaded_once_for_phi_services(fakes, monkeypatch):
    sentinel, seen = object(), []
    monkeypatch.setattr(bs, "get_phi_codec", lambda: seen.append(1) or sentinel)
    rt = await start_service_runtime(FastAPI(), _spec("production"))
    assert rt.phi_codec is sentinel and seen == [1]
    assert (await rt.evaluate_readiness()).checks["phi_codec"].ok
    await stop_service_runtime(rt)

async def test_codec_not_loaded_without_phi(fakes, monkeypatch):
    monkeypatch.setattr(bs, "get_phi_codec", pytest.fail)
    rt = await start_service_runtime(FastAPI(), _spec("production", phi=False))
    assert rt.phi_codec is None
    await stop_service_runtime(rt)

async def test_strict_codec_error_is_fatal(fakes, monkeypatch):
    def bad():
        raise bs.PHIConfigError("ENCRYPTION_MASTER_KEY invalid")
    monkeypatch.setattr(bs, "get_phi_codec", bad)
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("staging"))
    assert exc.value.component == "phi_codec"

async def test_dev_codec_error_degrades_ready(fakes, monkeypatch):
    def bad():
        raise bs.PHIConfigError("missing")
    monkeypatch.setattr(bs, "get_phi_codec", bad)
    rt = await start_service_runtime(FastAPI(), _spec("development"))
    assert (await rt.evaluate_readiness()).checks["phi_codec"].ok is False
    await stop_service_runtime(rt)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_055_codec_warmup.py`
Expected: FAIL with `AttributeError: ... has no attribute 'get_phi_codec'`; today PHI services only arm ORM listeners (`configure_phi_encryption`) and nothing proves at boot that the codec keys load (MVP-055).
- [ ] **Step 3: Implement** step 5 of `start_service_runtime`. Approach: module-level `from solace_security.phi_codec import PHIConfigError, get_phi_codec`; the call runs after the datastores and before audit; the readiness check reports `ok` with detail equal to the codec's `active_key_id` (an id, never key material).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_security`; then BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.9: Edge services render HTTPException as the uniform error envelope
**Covers:** none (IF-01/IF-12 contract per D-11; supports the orchestrator and user-service API contract owned by WS-11 and WS-08)
**Severity:** medium
**Wave:** 2
**Depends on:** WS-04.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-11 (default assumed: edge services return `{error:{code,message,request_id}}` from W2; internal services keep FastAPI's `{detail}` body)
**Real infra:** no
**Cross-workstream prerequisites:** none. Consumer impact: orchestrator and user-service tests that assert `{"detail": ...}` are updated in their own W2 phase-B lanes (WS-11, WS-08, WS-14).
**Files:**
- Modify: `src/solace_infrastructure/runtime/http.py` (`install_exception_handlers` adds a `StarletteHTTPException` handler when `spec.edge`)
- Test: `tests/bootstrap/test_d11_edge_error_envelope.py`
**Interfaces:**
- Consumes: `create_service_app` (WS-04.3).
- Produces: for `spec.edge`, `HTTPException(status, detail)` renders `{"error": {"code": <code>, "message": <message>, "request_id": <id>}}` where a dict detail with `code`/`message` keys is used as given, otherwise the code is `NOT_FOUND` (404), `UNAUTHORIZED` (401), `FORBIDDEN` (403), `CONFLICT` (409), `RATE_LIMITED` (429), `BAD_REQUEST` (400) or `HTTP_<status>`, and the message is the string detail; response headers (`WWW-Authenticate`, `Retry-After`) are preserved.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_d11_edge_error_envelope.py`
```python
"""D-11: edge HTTP errors use {error:{code,message,request_id}}; internal services keep {detail}."""
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient
from solace_infrastructure.runtime import ServiceRuntimeSpec, create_service_app
from solace_infrastructure.runtime.bootstrap import ServiceRuntime

router = APIRouter()

@router.get("/missing")
async def missing():
    raise HTTPException(404, "Thread not found")

@router.get("/consent")
async def consent():
    raise HTTPException(403, detail={"code": "CONSENT_REQUIRED", "message": "Consent required"})

@router.get("/auth")
async def auth():
    raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})

def _client(edge):
    spec = ServiceRuntimeSpec(service_name="orchestrator-service", environment="development", edge=edge)
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = ServiceRuntime(spec)
        yield
    return TestClient(create_service_app(spec, lifespan=lifespan, routers=[(router, "")], title="t"))

def test_edge_404_uses_the_envelope():
    with _client(True) as c:
        r = c.get("/missing", headers={"X-Request-ID": "r-404"})
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "NOT_FOUND", "message": "Thread not found", "request_id": "r-404"}}

def test_edge_dict_detail_keeps_its_code():
    with _client(True) as c:
        assert c.get("/consent").json()["error"]["code"] == "CONSENT_REQUIRED"

def test_edge_401_keeps_www_authenticate():
    with _client(True) as c:
        r = c.get("/auth")
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer" and r.json()["error"]["code"] == "UNAUTHORIZED"

def test_internal_service_keeps_detail():
    with _client(False) as c:
        assert c.get("/missing").json() == {"detail": "Thread not found"}
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_d11_edge_error_envelope.py`
Expected: FAIL: the three edge tests receive FastAPI's `{"detail": ...}` body (the W0 runtime leaves `HTTPException` untouched); `test_internal_service_keeps_detail` passes.
- [ ] **Step 3: Implement** the edge `StarletteHTTPException` handler in `install_exception_handlers(app, spec)`. Approach: register it only when `spec.edge`; read the request id from `request.state.request_id`; never include `str(exc)` beyond a string `detail` the route author wrote.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`. Expected: all pass. The orchestrator and user-service suites are re-run at the W2 barrier after their lanes adapt.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.10: Audit wiring in the bootstrap and AUDIT_HMAC_KEY as a required secret
**Covers:** none (IF-01 wiring for IF-14; supports MVP-091, owned by WS-09)
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-04.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: audit_events in the same database, HMAC chain keyed by AUDIT_HMAC_KEY from the secrets store)
**Real infra:** yes (BOOTSMOKE in dev and strict)
**Cross-workstream prerequisites:** WS-09 W3 phase A `solace_security.audit_runtime.configure_service_audit(service_name, pool)` (IF-14).
**Files:**
- Modify: `src/solace_infrastructure/runtime/bootstrap.py` (step 6; `required_secrets_for` adds `AUDIT_HMAC_KEY` when `spec.audited`)
- Test: `tests/bootstrap/test_mvp_091_audit_wiring.py`
**Interfaces:**
- Consumes: `async configure_service_audit(service_name: str, pool: asyncpg.Pool) -> None` (IF-14).
- Produces: readiness check `audit` (critical) for audited services; `required_secrets_for(spec)` returns `AUDIT_HMAC_KEY` for audited services.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_091_audit_wiring.py`
```python
"""Supports MVP-091: audited services configure the audit trail at startup; strict failures are fatal."""
import pytest
from fastapi import FastAPI
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError, start_service_runtime, stop_service_runtime
from solace_infrastructure.runtime import bootstrap as bs
# `calls` and `fakes` come from tests/bootstrap/conftest.py (WS-04.4)

def _spec(env, audited=True):
    return ServiceRuntimeSpec(service_name="user-service", environment=env, uses_postgres=True, uses_redis=True, audited=audited)

async def test_audit_configured_with_the_service_pool(fakes, calls, monkeypatch):
    async def cfg(name, pool):
        calls.append(("audit", name, pool is fakes))
    monkeypatch.setattr(bs, "configure_service_audit", cfg)
    rt = await start_service_runtime(FastAPI(), _spec("production"))
    assert ("audit", "user-service", True) in calls
    assert (await rt.evaluate_readiness()).checks["audit"].ok
    await stop_service_runtime(rt)

async def test_strict_audit_failure_is_fatal(fakes, monkeypatch):
    async def cfg(name, pool):
        raise RuntimeError("audit_events missing")
    monkeypatch.setattr(bs, "configure_service_audit", cfg)
    with pytest.raises(StartupDependencyError) as exc:
        await start_service_runtime(FastAPI(), _spec("staging"))
    assert exc.value.component == "audit"

async def test_unaudited_service_skips_audit(fakes, monkeypatch):
    monkeypatch.setattr(bs, "configure_service_audit", pytest.fail)
    rt = await start_service_runtime(FastAPI(), _spec("production", audited=False))
    await stop_service_runtime(rt)

def test_audited_services_require_the_hmac_key():
    assert "AUDIT_HMAC_KEY" in bs.required_secrets_for(_spec("production"))
    assert "AUDIT_HMAC_KEY" not in bs.required_secrets_for(_spec("production", audited=False))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_091_audit_wiring.py`
Expected: FAIL with `AttributeError: ... has no attribute 'configure_service_audit'` and `AUDIT_HMAC_KEY` missing from `required_secrets_for` (MVP-091: no service configures the audit logger).
- [ ] **Step 3: Implement** step 6 of `start_service_runtime`. Approach: module-level `from solace_security.audit_runtime import configure_service_audit`; runs after the codec step with `runtime.pg_pool`; the readiness check reports `ok` once configured.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_security`; then BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.11: Secret-source guards: known dev values, entropy floor and NAME_FILE provenance
**Covers:** MVP-349
**Severity:** high
**Wave:** 3
**Depends on:** WS-04.1, WS-04.7, WS-03.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (changed 2026-10-19 to SOPS + age: strict secrets come only from the environment's SOPS file `deploy/secrets/<env>.sops.yaml`, decrypted by WS-03.8's `deploy/render_secrets.py` with pinned sops/age into root-only 0400 files on the tmpfs `/run/secrets` and loaded through NAME_FILE; the NAME_FILE contract these guards check is unchanged)
**Real infra:** yes (BOOTSMOKE in strict proves every required secret arrives through NAME_FILE)
**Cross-workstream prerequisites:** WS-03.7 and WS-03.8 (strict compose delivers every secret as `NAME_FILE=/run/secrets/<source>`, rendered from the SOPS file; each file is owned by the uid that reads it, so the alertmanager password file belongs to the image user 65534, requested below); WS-01 `docker/entrypoint.sh` exports `NAME` from `NAME_FILE` and leaves `NAME_FILE` set.
**Files:**
- Modify: `src/solace_security/production_guards.py` (`KNOWN_DEV_VALUE_SHA256: frozenset[str]`, `MIN_SECRET_ENTROPY_BITS_PER_CHAR = 3.0`, `MIN_SECRET_BITS = 128`, `_check_secret_sources(required)` run in strict)
- Modify: `docker-compose.observability.yml` (alertmanager gets `ALERTMANAGER_SMTP_PASSWORD_FILE` and its secret mount instead of the plain variable)
- Modify: `infrastructure/alertmanager/render-config.sh` (reads the password from `ALERTMANAGER_SMTP_PASSWORD_FILE` when set)
- Test: `tests/bootstrap/test_mvp_349_secret_source_guards.py`
**Interfaces:**
- Consumes: the IF-22 rule "ProductionGuards rejects known dev values and low-entropy values in strict envs"; the list of formerly committed dev values from WS-03 (below).
- Produces: strict violations `"<NAME> matches a known development value"`, `"<NAME> has insufficient entropy"`, `"<NAME> must be supplied through <NAME>_FILE in strict environments"`.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_349_secret_source_guards.py`
```python
"""MVP-349 (guard half): strict boots reject committed dev secrets, placeholders, low-entropy values and
secrets that did not come from a NAME_FILE."""
import pytest
from solace_security.production_guards import ProductionGuards, ProductionSecurityError

KNOWN_DEV = ["dev-jwt-secret-key-minimum-32-chars", "dev-encryption-key-32-characters",
             "8kTaL3nRf9xQpZmVwYdE7gHjKbNcSuW2A5iO0tP4rXs=", "3mBvN7xRtYfW0pLqKjH8dS5gC2eA9iZuO6wTrXnMaQs=",
             "solace_dev_password", "solace-dev-key-change-in-production",
             "your-secure-jwt-secret-key-minimum-32-characters", "your-secure-auth-secret-key-32-chars",
             "your-secure-database-password"]

@pytest.fixture
def strict_files(monkeypatch, strict_secrets):
    monkeypatch.setenv("ENVIRONMENT", "production")
    for name in strict_secrets:
        if name not in ("POSTGRES_USER", "POSTGRES_SSL_MODE"):
            monkeypatch.setenv(f"{name}_FILE", f"/run/secrets/{name}")
    return strict_secrets

def test_file_delivered_random_secrets_pass(strict_files):
    ProductionGuards.validate_environment()

@pytest.mark.parametrize("value", KNOWN_DEV)
def test_known_dev_values_are_rejected(monkeypatch, strict_files, value):
    monkeypatch.setenv("AUTH_SECRET_KEY", value)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert "AUTH_SECRET_KEY matches a known development value" in exc.value.violations
    assert value not in str(exc.value)

def test_low_entropy_is_rejected(monkeypatch, strict_files):
    monkeypatch.setenv("AUTH_SECRET_KEY", "a" * 48)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert "AUTH_SECRET_KEY has insufficient entropy" in exc.value.violations

def test_plain_env_secret_is_rejected_in_strict(monkeypatch, strict_files):
    monkeypatch.delenv("AUTH_SECRET_KEY_FILE")
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert "AUTH_SECRET_KEY must be supplied through AUTH_SECRET_KEY_FILE in strict environments" in exc.value.violations

def test_dev_skips_source_checks(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("AUTH_SECRET_KEY", "dev-jwt-secret-key-minimum-32-chars")
    ProductionGuards.validate_environment()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_349_secret_source_guards.py`
Expected: FAIL: `test_known_dev_values_are_rejected[...]`, `test_low_entropy_is_rejected` and `test_plain_env_secret_is_rejected_in_strict` do not raise (MVP-349: the guards exact-match only trivial words and never inspect provenance); `test_file_delivered_random_secrets_pass` and `test_dev_skips_source_checks` pass.
- [ ] **Step 3: Implement** `_check_secret_sources(required: Sequence[str]) -> list[str]` in `production_guards.py` and call it from `validate_environment` in strict. Approach: compare `hashlib.sha256(value.encode()).hexdigest()` with `KNOWN_DEV_VALUE_SHA256` (the module stores digests, not literals, of every value in `KNOWN_DEV` above); Shannon entropy per character times length must reach `MIN_SECRET_BITS` and the per-character entropy must reach the floor; provenance requires `f"{name}_FILE"` in the environment for every required secret; violations carry names only. Update the observability fragment and render script so the Alertmanager SMTP password also arrives by file; alertmanager keeps running as its image user (65534), which owns that 0400 file in strict (WS-03.8), and `render-config.sh` reads it as that user.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\solace_security`; `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_alert_rules.py`; then BOOTSMOKE (strict boots only from files). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.12: Host, disk, backup and audit alerts with node-exporter
**Covers:** MVP-036, MVP-044
**Severity:** high
**Wave:** 4
**Depends on:** WS-04.11, WS-03.9
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-09 (default assumed: nightly base backup plus continuous WAL archiving, RPO 15 min, RTO 4 h)
**Real infra:** yes (promtool; WS-05's W4 BackupStale fault test)
**Cross-workstream prerequisites:** WS-03.9 writes `solace_backup_last_success_timestamp_seconds{store="postgres_base"|"postgres_wal"}` to `/textfile/solace_backup.prom` on the named volume `solace_textfile_metrics`; WS-09 writes `solace_audit_chain_verify_ok` to the same directory from its nightly verify job (requested below); `solace_audit_write_failures_total{service}` is incremented by WS-09 on the `AUDIT_WRITE_FAILURES` counter from WS-04.3.
**Files:**
- Modify: `docker-compose.observability.yml` (service `node-exporter`: image `prom/node-exporter:v1.8.2`, `--path.rootfs=/host`, `--collector.textfile.directory=/textfile`, read-only bind `/:/host:ro,rslave`, volume `solace_textfile_metrics:/textfile:ro`, network `monitoring`; top-level volume `solace_textfile_metrics`)
- Modify: `infrastructure/prometheus/prometheus.yml` (job `node` → `node-exporter:9100`)
- Modify: `infrastructure/prometheus/rules/solace.yml`, `infrastructure/prometheus/tests/solace_rules_test.yml`, `docs/runbooks/alerting.md`
- Test: `tests/bootstrap/test_mvp_044_host_backup_alerts.py`
**Interfaces:**
- Consumes: the backup textfile metric (WS-03.9); `solace_audit_chain_verify_ok` (WS-09).
- Produces: alerts `HostDiskUsage` (`max by (mountpoint) (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|overlay|squashfs"} / node_filesystem_size_bytes{fstype!~"tmpfs|overlay|squashfs"}) > 0.80` for 10m), `BackupStale` (`time() - max(solace_backup_last_success_timestamp_seconds{store="postgres_base"}) > 93600` or `absent(solace_backup_last_success_timestamp_seconds{store="postgres_base"})` for 30m), `WalArchiveStale` (new: `time() - max(solace_backup_last_success_timestamp_seconds{store="postgres_wal"}) > 1200` for 5m), `AuditWriteFailures` (`sum by (service) (increase(solace_audit_write_failures_total[10m])) > 0`), `AuditChainVerifyFailed` (`min(solace_audit_chain_verify_ok) == 0` or `absent(solace_audit_chain_verify_ok)` for 26h), `NodeExporterDown` (`up{job="node"} == 0` for 5m).

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_044_host_backup_alerts.py`
```python
"""MVP-036/MVP-044: host disk, backup freshness and audit integrity page the operator."""
import subprocess
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[2]

def _rules():
    doc = yaml.safe_load((REPO / "infrastructure/prometheus/rules/solace.yml").read_text(encoding="utf-8"))
    return {r["alert"]: r for g in doc["groups"] for r in g["rules"]}

@pytest.mark.parametrize("alert,needle", [("HostDiskUsage", "node_filesystem_avail_bytes"),
    ("BackupStale", 'store="postgres_base"'), ("WalArchiveStale", 'store="postgres_wal"'),
    ("AuditWriteFailures", "solace_audit_write_failures_total"), ("AuditChainVerifyFailed", "solace_audit_chain_verify_ok"),
    ("NodeExporterDown", 'up{job="node"}')])
def test_w4_alert_exists(alert, needle):
    rule = _rules()[alert]
    assert needle in rule["expr"] and rule["labels"]["severity"] == "page"

def test_backup_stale_threshold_is_26_hours():
    assert "93600" in _rules()["BackupStale"]["expr"]

def test_node_exporter_reads_the_textfile_volume():
    frag = yaml.safe_load((REPO / "docker-compose.observability.yml").read_text(encoding="utf-8"))
    ne = frag["services"]["node-exporter"]
    assert "--collector.textfile.directory=/textfile" in ne["command"]
    assert "solace_textfile_metrics:/textfile:ro" in ne["volumes"] and "ports" not in ne
    assert "solace_textfile_metrics" in frag["volumes"]

def test_prometheus_scrapes_node_exporter():
    cfg = yaml.safe_load((REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8"))
    node = {j["job_name"]: j for j in cfg["scrape_configs"]}["node"]
    assert node["static_configs"][0]["targets"] == ["node-exporter:9100"]

@pytest.mark.integration
@pytest.mark.realinfra
def test_promtool_rule_unit_tests_pass():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'prometheus'}:/etc/prometheus:ro",
                          "--entrypoint", "promtool", "prom/prometheus:v2.47.0", "test", "rules", "/etc/prometheus/tests/solace_rules_test.yml"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_044_host_backup_alerts.py`
Expected: FAIL with `KeyError: 'HostDiskUsage'` (and the other five), `KeyError: 'node-exporter'` and `KeyError: 'node'` (MVP-044: no disk/memory metrics and no host alerting exist).
- [ ] **Step 3: Implement** the node-exporter service, the scrape job, the six rules, promtool cases (firing and non-firing for each, including `absent` cases) and runbook sections. Approach: thresholds exactly as listed under Produces; node-exporter runs read-only with no published port.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command, then `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_044_host_backup_alerts.py`. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_036_alert_rules.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.13: Autoheal on /live behind a restricted socket proxy, plus ContainerUnhealthy
**Covers:** MVP-119
**Severity:** medium
**Wave:** 4
**Depends on:** WS-04.12, WS-03.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes (e2e restart test)
**Cross-workstream prerequisites:** WS-03.13 labels the nine app services `solace.autoheal: "true"` and keeps their healthchecks on `/live`.
**Files:**
- Modify: `docker-compose.observability.yml` (service `docker-socket-proxy`: image `tecnativa/docker-socket-proxy:0.3.0`, `/var/run/docker.sock:/var/run/docker.sock:ro`, env `CONTAINERS=1`, `ALLOW_RESTARTS=1`, `POST=0`, `EXEC=0`, `IMAGES=0`, `NETWORKS=0`, `VOLUMES=0`, `INFO=0`, internal network `socketproxy` only; service `autoheal`: image `willfarrell/autoheal:1.2.0`, env `AUTOHEAL_CONTAINER_LABEL=solace.autoheal`, `AUTOHEAL_INTERVAL=10`, `AUTOHEAL_START_PERIOD=90`, `DOCKER_SOCK=tcp://docker-socket-proxy:2375`, network `socketproxy`, no socket mount; both with `cap_drop: [ALL]`, `security_opt: ["no-new-privileges:true"]`, `read_only: true`, `pids_limit: 64`)
- Modify: `infrastructure/prometheus/rules/solace.yml`, `infrastructure/prometheus/tests/solace_rules_test.yml`, `docs/runbooks/alerting.md`
- Test: `tests/bootstrap/test_mvp_119_autoheal.py`, `tests/e2e/test_mvp_119_autoheal_restart.py`
**Interfaces:**
- Consumes: `solace.autoheal` labels and `/live` healthchecks (WS-03.13); `solace_background_task_up` (WS-04.3); `process_start_time_seconds` from prometheus_client's default process collector.
- Produces: alert `ContainerUnhealthy` (`(min by (service) (solace_background_task_up) == 0) or (changes(process_start_time_seconds{job="solace-services"}[15m]) > 0)` for 1m); only the proxy touches the Docker socket.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_119_autoheal.py` and `tests/e2e/test_mvp_119_autoheal_restart.py`
```python
# tests/bootstrap/test_mvp_119_autoheal.py
"""MVP-119: an unhealthy (hung) container is restarted and alerted; only a restart-only proxy sees the socket."""
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
FRAG = yaml.safe_load((REPO / "docker-compose.observability.yml").read_text(encoding="utf-8"))

def test_autoheal_watches_labelled_containers_through_the_proxy():
    env = FRAG["services"]["autoheal"]["environment"]
    assert env["AUTOHEAL_CONTAINER_LABEL"] == "solace.autoheal"
    assert env["DOCKER_SOCK"] == "tcp://docker-socket-proxy:2375"
    assert not any("docker.sock" in v for v in FRAG["services"]["autoheal"].get("volumes", []))

def test_only_the_proxy_mounts_the_socket_and_it_only_allows_restarts():
    mounts = {n: s.get("volumes", []) for n, s in FRAG["services"].items()}
    holders = [n for n, vols in mounts.items() if any("docker.sock" in v for v in vols)]
    assert holders == ["docker-socket-proxy"]
    env = FRAG["services"]["docker-socket-proxy"]["environment"]
    assert env["ALLOW_RESTARTS"] == "1" and env["POST"] == "0" and env["EXEC"] == "0"
    assert "ports" not in FRAG["services"]["docker-socket-proxy"]

def test_container_unhealthy_rule():
    doc = yaml.safe_load((REPO / "infrastructure/prometheus/rules/solace.yml").read_text(encoding="utf-8"))
    rule = {r["alert"]: r for g in doc["groups"] for r in g["rules"]}["ContainerUnhealthy"]
    assert "solace_background_task_up" in rule["expr"] and "process_start_time_seconds" in rule["expr"]
```
```python
# tests/e2e/test_mvp_119_autoheal_restart.py
"""MVP-119: a SIGSTOPped notification-service turns unhealthy, autoheal restarts it, /live recovers, an alert email arrives."""
import subprocess, time
import httpx, pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

def test_hung_service_is_restarted(e2e_stack):
    name = subprocess.run(["docker", "compose", "-p", e2e_stack.project, "ps", "-q", "notification-service"],
                          capture_output=True, text=True, check=True).stdout.strip()
    before = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}}", name], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["docker", "kill", "--signal=SIGSTOP", name], check=True)
    deadline, recovered = time.monotonic() + 300, False
    while time.monotonic() < deadline and not recovered:
        time.sleep(10)
        state = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}} {{.State.Health.Status}}", name], capture_output=True, text=True).stdout.split()
        recovered = len(state) == 2 and state[0] != before and state[1] == "healthy"
    assert recovered, "autoheal did not restart the hung container within 5 minutes"
    msgs = httpx.get(f"{e2e_stack.mailpit_api}/api/v1/messages", timeout=10).json()["messages"]
    assert any(("ContainerUnhealthy" in m["Subject"]) or ("ServiceNotReady" in m["Subject"]) for m in msgs)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_119_autoheal.py`; `$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_mvp_119_autoheal_restart.py`
Expected: FAIL with `KeyError: 'autoheal'` / `'ContainerUnhealthy'`; the e2e test fails with "autoheal did not restart the hung container within 5 minutes" (MVP-119: plain Docker restarts only on process exit, so the wedged container stays unhealthy).
- [ ] **Step 3: Implement** the proxy, autoheal, the rule, promtool cases and the runbook section. Approach: the proxy sits on a dedicated internal network shared only with autoheal (and Alloy in WS-04.14); the runbook notes that a deploy restart fires `ContainerUnhealthy` once and how to silence it during planned deploys.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2, then `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_044_host_backup_alerts.py` (promtool). Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy`; BOOTSMOKE with the observability fragment. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.14: Log shipping to Loki with retention
**Covers:** MVP-114
**Severity:** medium
**Wave:** 4
**Depends on:** WS-04.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: the 6-year audit retention lives in Postgres audit_events from WS-09, not in logs; operational logs are kept 31 days)
**Real infra:** yes (config validation in pinned images; e2e recreate test)
**Cross-workstream prerequisites:** WS-09 IF-14 audit trail (W3) closes the audit-retention half of MVP-114; the socket proxy from WS-04.13.
**Files:**
- Modify: `docker-compose.observability.yml` (service `loki`: image `grafana/loki:3.1.1`, config `/etc/loki/loki-config.yml`, volume `loki_data`, network `monitoring`; service `alloy`: image `grafana/alloy:v1.4.2`, config `./infrastructure/grafana/alloy/config.alloy`, networks `monitoring` and `socketproxy`, no socket mount; both confined like WS-04.13)
- Create: `infrastructure/grafana/loki/loki-config.yml` (filesystem storage, `limits_config.retention_period: 744h`, `compactor.retention_enabled: true`)
- Create: `infrastructure/grafana/alloy/config.alloy` (`discovery.docker` with `host = "tcp://docker-socket-proxy:2375"`, relabel `__meta_docker_container_label_com_docker_compose_service` to label `service`, `loki.source.docker` to `loki.write` at `http://loki:3100/loki/api/v1/push`; no JSON field is promoted to a label)
- Create: `infrastructure/grafana/provisioning/datasources/loki.yml`
- Modify: `docs/runbooks/alerting.md` (section "Finding logs")
- Test: `tests/bootstrap/test_mvp_114_log_shipping.py`, `tests/e2e/test_mvp_114_logs_survive_recreate.py`
**Interfaces:**
- Consumes: the socket proxy (WS-04.13); JSON logs from the runtime (WS-04.2).
- Produces: Loki at `loki:3100` with 31-day retention; label set limited to `service` and `container`.

- [ ] **Step 1: Write the failing test** in `tests/bootstrap/test_mvp_114_log_shipping.py` and `tests/e2e/test_mvp_114_logs_survive_recreate.py`
```python
# tests/bootstrap/test_mvp_114_log_shipping.py
"""MVP-114: container logs are shipped off the json-file driver with retention, through the restricted proxy."""
import re, subprocess
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[2]

def test_loki_retains_31_days():
    cfg = yaml.safe_load((REPO / "infrastructure/grafana/loki/loki-config.yml").read_text(encoding="utf-8"))
    assert cfg["limits_config"]["retention_period"] == "744h" and cfg["compactor"]["retention_enabled"] is True

def test_alloy_uses_the_proxy_and_promotes_no_payload_labels():
    alloy = (REPO / "infrastructure/grafana/alloy/config.alloy").read_text(encoding="utf-8")
    assert 'host = "tcp://docker-socket-proxy:2375"' in alloy and "docker.sock" not in alloy
    assert not re.search(r"(user_id|email|session_id|message)\s*=", alloy)

def test_fragment_wires_loki_and_alloy():
    frag = yaml.safe_load((REPO / "docker-compose.observability.yml").read_text(encoding="utf-8"))
    assert {"loki", "alloy"} <= set(frag["services"]) and "loki_data" in frag["volumes"]
    assert not any("docker.sock" in v for v in frag["services"]["alloy"].get("volumes", []))

@pytest.mark.integration
@pytest.mark.realinfra
def test_loki_config_is_valid():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'grafana' / 'loki'}:/etc/loki:ro",
                          "grafana/loki:3.1.1", "-config.file=/etc/loki/loki-config.yml", "-verify-config"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
```
```python
# tests/e2e/test_mvp_114_logs_survive_recreate.py
"""MVP-114: a log line written before a container is recreated is still queryable afterwards."""
import subprocess, time, uuid
import httpx, pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]

def test_logs_survive_container_recreate(e2e_stack):
    marker = f"logprobe-{uuid.uuid4().hex[:12]}"
    # an unmatched route is logged at INFO (route "__unmatched__") with the bound request id
    httpx.get(f"{e2e_stack.urls['safety-service']}/api/v1/safety/__logprobe__", headers={"X-Request-ID": marker}, timeout=10)
    subprocess.run(["docker", "compose", "-p", e2e_stack.project, "up", "-d", "--force-recreate", "--no-deps", "safety-service"], check=True)
    loki = e2e_stack.urls["loki"]
    deadline, found = time.monotonic() + 180, False
    while time.monotonic() < deadline and not found:
        time.sleep(5)
        r = httpx.get(f"{loki}/loki/api/v1/query_range", params={"query": f'{{service="safety-service"}} |= "{marker}"', "limit": 5}, timeout=10)
        found = bool(r.json()["data"]["result"])
    assert found
```
(`e2e_stack.urls["loki"]` requires WS-05 to publish Loki on an ephemeral loopback port in the test overlay, requested below.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_114_log_shipping.py`; `$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_mvp_114_logs_survive_recreate.py`
Expected: FAIL with `FileNotFoundError` for `infrastructure/grafana/loki/loki-config.yml` and `KeyError` for `loki`; the e2e test fails because no log store exists and recreate deletes the json-file logs (MVP-114).
- [ ] **Step 3: Implement** the Loki and Alloy configs, the datasource and the fragment services. Approach: Loki single-binary filesystem mode, compactor deletes beyond 744h; Alloy reads container logs only for containers of the compose project (relabel keep on `com_docker_compose_project`).
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2, then `DOCKER D:\Repo\Contextual-Chatbot\tests\bootstrap\test_mvp_114_log_shipping.py`. Expected: PASS
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy`; BOOTSMOKE with the observability fragment. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.15: The PHI-activation lock accepts the runtime SPEC in place of the retired configure_phi_encryption call
**Covers:** none (supporting change for IF-01 adoption: CR-8 of WS-09, R3 of WS-16, R-17-03 of WS-17, R-18-03 of WS-18)
**Severity:** high
**Wave:** 0
**Depends on:** WS-04.5
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** none
**Real infra:** no (AST scan of the service sources)
**Cross-workstream prerequisites:** none to write it. Later edits by the owner: WS-12.12 (W3) moves `safety_service` from `_PHI_ENCRYPTING` to `_PHI_VIA_RUNTIME` in this file (WS-12's own edit), and until then WS-09 keeps `configure_phi_encryption` importable (R-12-7).
**Files:**
- Modify: `tests/integration/test_phi_encryption_activation.py`. This is the shared test dir, so the shared-dir rule applies: IF-01 moves the startup PHI activation that this file locks into the runtime bootstrap, so WS-04 edits it, as WS-09's CR-8 assigns. The edits:
  - `_PHI_ENCRYPTING = {"safety_service"}` (the only service still on the legacy ORM listener until WS-12.12);
  - new `_PHI_VIA_RUNTIME = {"diagnosis_service", "memory_service", "orchestrator_service", "personality_service", "therapy_service"}`;
  - new helper `_declares_phi_spec(src: pathlib.Path) -> bool`;
  - the tests below, and a docstring that describes both activation routes;
  - unchanged: `_EXEMPT` (user-service stays exempt from the ClinicalBase listener; its `handles_phi=True` is pinned by WS-04.5's conformance table), `_imports_clinical_base`, `_calls_activation`, `test_exempt_services_do_not_import_clinical_base` and `test_detector_has_teeth`.
- Test: `tests/integration/test_phi_encryption_activation.py` (the same file)
**Interfaces:**
- Consumes: the IF-01 declaration `SPEC = ServiceRuntimeSpec(..., handles_phi=True, ...)` that each adopted `src/main.py` writes (WS-11.1, WS-15.1, WS-16.1, WS-17.1, WS-18.1); WS-04.5's `SERVICES` table, which already pins `handles_phi=True` for orchestrator, safety, therapy, diagnosis, memory, personality and user-service (the conformance half of CR-8).
- Produces: a PHI-activation lock that stays green when a W0 phase-B adoption replaces `configure_phi_encryption(...)` with a `handles_phi=True` SPEC, and turns red when a PHI service has neither. Consumed by WS-16.1 (WS-16 request R3), WS-17.1 (R-17-03), WS-18.1 and WS-18.2 (R-18-03), WS-11.1 and WS-15.1 (the same removal for orchestrator and therapy), and WS-12.12 (moves safety in W3).

- [ ] **Step 1: Write the failing test** in `tests/integration/test_phi_encryption_activation.py`
```python
# module level, after _EXEMPT
_PHI_ENCRYPTING = {"safety_service"}
_PHI_VIA_RUNTIME = {"diagnosis_service", "memory_service", "orchestrator_service", "personality_service", "therapy_service"}

class TestPhiEncryptionActivation:
    def test_every_service_is_explicitly_classified(self) -> None:
        buckets = (_PHI_ENCRYPTING, _PHI_VIA_RUNTIME, _EXEMPT)
        assert all(a.isdisjoint(b) for i, a in enumerate(buckets) for b in buckets[i + 1:])
        unclassified = [s for s in _service_names() if s not in set().union(*buckets)]
        assert not unclassified, f"Unclassified services {unclassified}"

    def test_runtime_phi_services_activate_through_spec_or_legacy_call(self) -> None:
        missing = [n for n in sorted(_PHI_VIA_RUNTIME)
                   if not (_declares_phi_spec(_SERVICES_DIR / n / "src") or _calls_activation(_SERVICES_DIR / n / "src"))]
        assert not missing, f"PHI services with neither a handles_phi=True SPEC nor configure_phi_encryption(): {missing}"

    def test_classification_is_non_vacuous(self) -> None:
        assert (_PHI_ENCRYPTING | _PHI_VIA_RUNTIME) <= set(_service_names())
    # test_phi_services_activate_encryption and test_exempt_services_do_not_import_clinical_base keep their bodies

def test_spec_detector_has_teeth(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    main = src / "main.py"
    main.write_text("from solace_infrastructure.runtime import ServiceRuntimeSpec\n"
                    "# SPEC = ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=True)\n"
                    "SPEC = ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=False)\n", encoding="utf-8")
    assert _declares_phi_spec(src) is False
    main.write_text("from solace_infrastructure import runtime\n"
                    "SPEC = runtime.ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=True)\n", encoding="utf-8")
    assert _declares_phi_spec(src) is True

def test_adoption_that_drops_phi_handling_is_caught(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("from solace_infrastructure.runtime import ServiceRuntimeSpec\n"
                                 "SPEC = ServiceRuntimeSpec(service_name='x', environment='test')\n", encoding="utf-8")
    assert not (_declares_phi_spec(src) or _calls_activation(src))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`
Expected: FAIL. `test_spec_detector_has_teeth` and `test_adoption_that_drops_phi_handling_is_caught` raise `NameError: name '_declares_phi_spec' is not defined`, and `test_runtime_phi_services_activate_through_spec_or_legacy_call` raises `NameError` for `_PHI_VIA_RUNTIME`. The file has no route for the IF-01 declaration: `test_phi_services_activate_encryption` requires a `configure_phi_encryption` call in all six services of `_PHI_ENCRYPTING` (test_phi_encryption_activation.py:37-44, 129-139). The W0 adoptions remove that call (for example `services/diagnosis_service/src/main.py:96-100` under WS-16.1), so without this change they would turn the lock red. If the lock were deleted instead, nothing would catch a PHI service that drops the call and declares no PHI handling.
- [ ] **Step 3: Implement** `_declares_phi_spec` and the set split. Approach: walk every `.py` under `src` with `ast` (skip `__pycache__`, ignore `SyntaxError` like the two existing helpers); a match is an `ast.Call` whose function is `ServiceRuntimeSpec` (a Name, or an Attribute with that `attr`) and that has a keyword `handles_phi` whose value is `ast.Constant(True)`. The adoption tasks write the keyword form, and a non-literal value does not count.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS. All five `_PHI_VIA_RUNTIME` services still pass through their legacy call today, and each turns to the SPEC route as its adoption lands.
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap -k "not test_service_conforms and not test_lifespan_is_bracketed_by_the_runtime"`; `UNIT D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`. Expected: all pass. Each W0 phase-B adoption task re-runs this file in its own regression, and it must stay green there.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-04.F1: IF-01 hardening from the integrated audit (lifespan shape, PHI paths, body limit, teardown)
**Covers:** MVP-036, MVP-180, MVP-332
**Severity:** high
**Wave:** 0
**Depends on:** WS-04.3, WS-04.4, WS-04.5, WS-09.F1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none. Added by controller Ruling on 2026-11-03 from the IF-01 integrated audit (D:\tmp\mvp-exec\reviews\batch12-if01-audit.json: 3 lenses, each finding put to a skeptic panel with majority-refute-to-kill). Surviving findings: important LIF-1, PHI-1, PHI-2, HTT-2; minor LIF-2, LIF-3, LIF-4, LIF-5, PHI-3, PHI-4. Gates W0 phase B: LIF-1 fixes the lifespan shape that all nine adoptions copy, so it lands first under its own checkpoint.
**Real infra:** no (probe apps under uvicorn and TestClient, fake datastores)
**Files:**
- Modify: `src/solace_infrastructure/runtime/bootstrap.py`, `src/solace_infrastructure/runtime/http.py`, `src/solace_infrastructure/runtime/logging.py`, `src/solace_infrastructure/runtime/datastores.py`, `src/solace_infrastructure/runtime/health.py`, `src/solace_infrastructure/runtime/spec.py`, `src/solace_infrastructure/runtime/__init__.py` (export `service_runtime`)
- Modify: `tests/bootstrap/test_service_conformance.py` (`test_lifespan_is_bracketed_by_the_runtime`), `tests/bootstrap/fixtures/reference_app.py`
- Create: `tests/bootstrap/test_if01_audit_hardening.py`
- Not touched: `src/solace_infrastructure/database/connection_manager.py` (WS-02.6 owns it; LIF-4 is fixed in the runtime closer).
**Interfaces:**
- Produces (LIF-1, consumed by every W0 phase-B adoption): `service_runtime(app, spec) -> AsyncContextManager[ServiceRuntime]` in `solace_infrastructure.runtime`. It awaits `start_service_runtime` on entry and runs `stop_service_runtime` in a `finally` on exit, including when composition raises or the lifespan is cancelled. The canonical lifespan is `async with service_runtime(app, SPEC) as runtime:` as the first statement of the lifespan body, with all composition and the `yield` inside it and every service-owned stop registered with `runtime.register_shutdown`. The conformance check accepts exactly two forms: (a) that `async with`, or (b) `runtime = await start_service_runtime(...)` immediately followed by a `try` whose `finalbody` is exactly `[await stop_service_runtime(runtime)]` and which contains the `yield`. A bare last statement no longer passes. The module docstring and `reference_app.py` show form (a).
- Produces (HTT-2): `ServiceRuntimeSpec.max_request_body_bytes: int = 1_048_576`. A pure ASGI middleware, installed inside the request-id middleware so a refusal still carries the request id, returns 413 with the uniform error envelope (code `request_too_large`, no echo of the body or headers) when `Content-Length` exceeds the limit, before reading anything, and also when a chunked or length-less body exceeds it while being received. It never buffers more than the limit.
- Produces (LIF-5): `/ready` answers 503 from the moment `stop_service_runtime` begins (drain) and after it returns, and a stopped runtime is detached from `app.state`.
- Unchanged: every other IF-01 name and the error envelope shape.

- [ ] **Step 1: Write the failing tests** in `tests/bootstrap/test_if01_audit_hardening.py`, one per finding, each reproducing the audit's failure scenario:
  - LIF-1 (a): a lifespan whose composition raises after `service_runtime` entry still runs the Postgres, Redis and refresher closers and leaves no pending runtime task. (b): service stop code registered with `register_shutdown` that raises does not stop the datastore closers. (c): `test_lifespan_is_bracketed_by_the_runtime` rejects the bare form and accepts forms (a) and (b), pinned with three inline source snippets.
  - LIF-2: a task given to `register_background_task` is cancelled and awaited by `stop_service_runtime` before the first datastore closer.
  - LIF-3: a closer that overruns `SHUTDOWN_STEP_TIMEOUT_SECONDS` is cancelled and its cancellation cleanup finishes (bounded grace of 1 s) before the next closer starts.
  - LIF-4: after the Postgres closer times out, the `ConnectionPoolManager` registry holds no entry for the closed pool, so a second lifespan in the same process gets a new pool.
  - LIF-5: `/ready` returns 503 during and after `stop_service_runtime`.
  - PHI-1: an unhandled exception whose `__cause__` or `__context__` chain contains a pydantic `ValidationError` built from user input logs no input value (assert on the rendered log line).
  - PHI-2: `warnings.warn("user@example.com ...")` after `configure_service_logging` reaches the runtime chain (`logging.captureWarnings(True)` and `py.warnings` routed) and the rendered line holds no email.
  - PHI-3: tuple, set, bytes, exception and arbitrary-object event values that hold an email or a phone number are sanitized before rendering.
  - PHI-4: the 500 log summary of an `extra_forbidden` ValidationError masks the client-chosen key exactly as the 422 body does.
  - HTT-2: `Content-Length: 1048577` gets 413 without the app reading the body; a chunked body of 1 048 577 bytes gets 413; a 1 048 576-byte body passes; `max_request_body_bytes` on the spec overrides the default.
- [ ] **Step 2: Run them and confirm they fail.** Run: `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap\test_if01_audit_hardening.py`. Expected: FAIL on every case. `service_runtime` does not exist (ImportError for LIF-1), and the other cases fail on the audit's recorded behaviour: closers skipped, task still alive, cleanup after return, closed pool returned, 200 after stop, values in the log line, no 413.
- [ ] **Step 3a: Implement LIF-1 first** (`service_runtime`, the conformance check, the docstring and `reference_app.py`). Run the LIF-1 cases and `tests/bootstrap/test_service_conformance.py -k lifespan`, then checkpoint `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.F1:shape"` and report the tree id to the controller. That checkpoint releases W0 phase B.
- [ ] **Step 3b: Implement the rest:** PHI-1 walks the exception chain; PHI-2 routes `py.warnings`; PHI-3 stringifies and sanitizes non-JSON-native values in a processor before the renderer; PHI-4 reuses `_validation_detail`'s masking; LIF-2 registers a cancel-and-await closer per task; LIF-3 adds the bounded grace; LIF-4 drops the registry entries in the runtime's Postgres closer, in a `finally`; LIF-5 marks the runtime stopping; HTT-2 adds the middleware and the spec field.
- [ ] **Step 4: Run them and confirm they pass.** Run: same command. Expected: PASS.
- [ ] **Step 5: Regression:** `UNIT D:\Repo\Contextual-Chatbot\tests\bootstrap` (the 18 per-service conformance cases stay red until phase B, as before), `UNIT D:\Repo\Contextual-Chatbot\tests\solace_infrastructure`, `UNIT D:\Repo\Contextual-Chatbot\tests\integration\test_phi_encryption_activation.py`, one process each. Expected: no new failures.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-04.F1:head"` and record the printed tree id in the ledger. No git commit.

---

## Cross-workstream requests

| Owner | Exact change | Files | Needed by |
|---|---|---|---|
| WS-07 | W0 phase A1, alongside `acting_as_user`: add `set_correlation_ids(*, request_id: str, correlation_id: str, turn_id: str) -> None`, `get_correlation_headers() -> dict[str, str]` (keys `X-Request-ID`, `X-Correlation-ID`, `X-Turn-ID`, only those set) and `clear_correlation_ids() -> None` on contextvars; W1 `service_auth_headers` and `forwarded_auth_headers` read `get_correlation_headers()` | `src/solace_common/request_context.py` | WS-04.3 |
| WS-07 | W1 phase A1: `install_rate_limiting(app, spec)` is callable at app construction (no running loop needed) and `ServiceTokenManager(service_settings=ServiceAuthSettings(service_name=...)).create_service_token(name)` works with only `AUTH_SECRET_KEY` set | `src/solace_security/rate_limit.py`, `src/solace_security/service_auth.py` | WS-04.6 |
| WS-01 | W0 phase A1: `prometheus-client` stays in `requirements.in`/lock; `src/solace_infrastructure/__init__.py` lazy so `import solace_infrastructure.runtime` imports neither FastAPI nor asyncpg | `requirements.in`, `src/solace_infrastructure/__init__.py` | WS-04.2 |
| WS-02 | W0 phase A2: `solace_infrastructure.database.schema_check.assert_schema_head(pool)` and the strict `'default'` pool policy, importable without side effects | `src/solace_infrastructure/database/schema_check.py` | WS-04.4 |
| WS-06 | W1 phase A1: `start_service_event_plane(...)` per IF-05 with `readiness() -> tuple[bool, str]` and `stop()` that finishes the in-flight flush before returning (MVP-332 outbox half) | `src/solace_events/service_plane.py` | WS-04.6 |
| WS-09 | W0 phase A1 (with MVP-199): `UnifiedLLMClient.shutdown()` closes the underlying Portkey/httpx client instead of only dropping the reference (MVP-332); from W1 it also closes the botocore `bedrock-runtime` client of the D-04 Bedrock provider, which strict environments use instead of Portkey | `services/shared/infrastructure/llm_client.py` | WS-04.4 |
| WS-09 | W2: `get_phi_codec()` and `PHIConfigError` per IF-13; W3: `configure_service_audit(service_name, pool)` per IF-14, increments `solace_infrastructure.runtime.metrics.AUDIT_WRITE_FAILURES` on failed writes, and the nightly `verify_chain` job writes `solace_audit_chain_verify_ok` to `/textfile/solace_audit.prom` on volume `solace_textfile_metrics` | `src/solace_security/phi_codec.py`, `src/solace_security/audit_runtime.py` | WS-04.8, WS-04.10, WS-04.12 |
| WS-13 | W1: `solace_crisis_alert_delivery_total{outcome,channel}` uses exactly the outcome values `delivered`, `failed`, `fallback_delivered` (CrisisAlertDeliveryFailed fires on `failed`) | `services/notification-service/src/consumers.py` | WS-04.7 |
| WS-05 | W1: the e2e stack includes `-f docker-compose.observability.yml`; `docker-compose.test.yml` sets `ALERTMANAGER_SMTP_SMARTHOST=mailpit:1025`, `ALERTMANAGER_EMAIL_TO=ops-alerts@test.invalid`, `ALERTMANAGER_EMAIL_FROM=alerts@test.invalid`, `ALERTMANAGER_SMTP_REQUIRE_TLS=false`; `test_w1_crisis_faults.py` asserts the matching alert email (CrisisAlertDeliveryFailed, CrisisOutboxBacklog, NotificationConsumerLag, DLQUnresolved) in its fault scenarios | `tests/e2e/conftest.py`, `docker-compose.test.yml`, `tests/e2e/test_w1_crisis_faults.py` | WS-04.7 |
| WS-05 | W4: publish Loki on an ephemeral loopback port as `Stack.urls["loki"]`; `test_w4_enterprise.py` stops `pg-backup` and asserts BackupStale/WalArchiveStale email delivery | `docker-compose.test.yml`, `tests/e2e/test_w4_enterprise.py` | WS-04.12, WS-04.14 |
| WS-03 | W1 (WS-03.6): registry rows `ALERTMANAGER_EMAIL_TO`, `ALERTMANAGER_EMAIL_FROM`, `ALERTMANAGER_SMTP_SMARTHOST`, `ALERTMANAGER_SMTP_USERNAME`, `ALERTMANAGER_SMTP_PASSWORD` (secret), `ALERTMANAGER_SMTP_REQUIRE_TLS` (services `[alertmanager]`), and the base `prometheus` service mounts `./infrastructure/prometheus:/etc/prometheus:ro` | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-04.7 |
| WS-03 | W4 (WS-03.9 and WS-03.13): the backup sidecar writes the textfile metric on volume `solace_textfile_metrics`; the nine app services carry label `solace.autoheal: "true"` and `/live` healthchecks | `docker-compose.backup.yml`, `docker-compose.yml` | WS-04.12, WS-04.13 |
| WS-03 | W3 (WS-03.8, D-07): `SECRET_READER_UIDS` maps `alertmanager` to 65534, so `render_secrets` gives its 0400 `ALERTMANAGER_SMTP_PASSWORD` file to the alertmanager image user; any later WS-04 fragment service that reads a secret file is added to that map in the same wave | `deploy/env/registry.py` | WS-04.11 |
| WS-11 | W0 phase B adoption: `SPEC` flags exactly as in the WS-04.5 table (orchestrator: postgres, redis, publish_consume, PHI, audited, rate_limited, edge); register LLM/httpx clients with `runtime.register_shutdown`; acceptance `-k orchestrator-service` | `services/orchestrator_service/src/main.py` | WS-04.5 |
| WS-12 | W0 phase B adoption per the WS-04.5 table (safety); acceptance `-k safety-service` | `services/safety_service/src/main.py` | WS-04.5 |
| WS-12 | W3 (WS-12.12): when `safety_service` leaves `_PHI_ENCRYPTING` in `tests/integration/test_phi_encryption_activation.py`, add it to `_PHI_VIA_RUNTIME` (the bucket WS-04.15 creates); otherwise `test_every_service_is_explicitly_classified` reports it unclassified | `tests/integration/test_phi_encryption_activation.py` | WS-04.15 |
| WS-13 | W0 phase B adoption per the WS-04.5 table (notification: not PHI, audited); background consumers registered with `runtime.register_background_task`; acceptance `-k notification-service` | `services/notification-service/src/main.py` | WS-04.5 |
| WS-14 | W0 phase B adoption per the WS-04.5 table (user-service: PHI, audited, rate_limited, edge); acceptance `-k user-service` | `services/user-service/src/main.py` | WS-04.5 |
| WS-15 | W0 phase B adoption per the WS-04.5 table (therapy); acceptance `-k therapy-service` | `services/therapy_service/src/main.py` | WS-04.5 |
| WS-16 | W0 phase B adoption per the WS-04.5 table (diagnosis); acceptance `-k diagnosis-service` | `services/diagnosis_service/src/main.py` | WS-04.5 |
| WS-17 | W0 phase B adoption per the WS-04.5 table (memory); the module-level memory-node client and Redis clients are registered as closers; acceptance `-k memory-service` | `services/memory_service/src/main.py` | WS-04.5 |
| WS-18 | W0 phase B adoption per the WS-04.5 table (personality: PHI; analytics: not PHI, not audited); analytics drops `prometheus_fastapi_instrumentator` because `/metrics` now comes from the runtime; acceptance `-k personality-service` and `-k analytics-service` | `services/personality_service/src/main.py`, `services/analytics-service/src/main.py` | WS-04.5 |

## New interfaces

- `ProductionGuards.validate_environment(*, required_vars=None)` and `ProductionGuards.assert_startup(service_environment, *, service_name=None, required_vars=None)`; `DEFAULT_REQUIRED_IN_STRICT`; `is_strict_environment` has its canonical definition in `solace_security.production_guards` and is re-exported lazily by `solace_infrastructure.runtime` (so the security package never imports the infrastructure package).
- `solace_infrastructure.runtime.bootstrap.required_secrets_for(spec) -> tuple[str, ...]` (`AUTH_SECRET_KEY`; `POSTGRES_PASSWORD` if `uses_postgres`; `REDIS_URL` if `uses_redis`; `ENCRYPTION_MASTER_KEY`, `ENCRYPTION_SEARCH_HASH_SALT` if `handles_phi`; `AUDIT_HMAC_KEY` if `audited`, from W3).
- `ServiceRuntime.evaluate_readiness() -> ReadinessReport`; `CheckResult(ok, critical, detail)`; `ReadinessReport(ready, checks)`; `StartupDependencyError(component, reason)` attributes.
- Metric `solace_service_ready{service}` (1 when `/ready` would return 200), used by `ServiceNotReady`.
- Standard metric objects importable from `solace_infrastructure.runtime.metrics` so producers never re-declare them: `CRISIS_DETECTED`, `LLM_FALLBACK`, `AUDIT_WRITE_FAILURES` (plus the HTTP, readiness and background-task metrics).
- Prometheus job `solace-services` with per-target label `service`; job `node`.
- Alert `WalArchiveStale` and alert `NodeExporterDown` (beyond IF-18's list); `ContainerUnhealthy` is defined as a dead registered background task or a process restart within 15 minutes.
- Alertmanager variables `ALERTMANAGER_EMAIL_TO`, `ALERTMANAGER_EMAIL_FROM`, `ALERTMANAGER_SMTP_SMARTHOST`, `ALERTMANAGER_SMTP_USERNAME`, `ALERTMANAGER_SMTP_PASSWORD` (`_FILE` from W3), `ALERTMANAGER_SMTP_REQUIRE_TLS`.
- Textfile-collector contract: named volume `solace_textfile_metrics`, mounted at `/textfile` by metric writers (backup sidecar, audit verify job) and read-only by node-exporter.
- `solace_common.request_context` correlation helpers (requested from WS-07).
- Crisis delivery outcome label values `delivered`, `failed`, `fallback_delivered` (requested from WS-13).
- Test-only switch `SOLACE_RUN_DOCKER_TESTS=1` for docker-running tests in `tests/bootstrap` (shared with WS-03's `tests/deploy`).

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-031 | blocker | WS-04.1 |
| MVP-036 | high | WS-04.3 (/metrics everywhere), WS-04.5 (conformance + scrape config), WS-04.7 (Alertmanager + crisis-path rules), WS-04.12 (host/disk/backup/audit alerts) |
| MVP-114 | medium | WS-04.14 (log shipping); the audit-retention half closes with WS-09's IF-14 trail (W3) |
| MVP-119 | medium | WS-04.13 |
| MVP-180 | medium | WS-04.2 (chain), WS-04.5 (every service uses it) |
| MVP-332 | low | WS-04.4 (pools, Redis, registered closers), WS-04.6 (event plane stopped before the pool) |

Supporting tasks without own register ids: WS-04.8 (IF-13 warm-up, MVP-055), WS-04.9 (D-11 envelope), WS-04.10 (IF-14 wiring, MVP-091), WS-04.15 (PHI-activation lock for the W0 adoptions, CR-8); WS-04.11 also carries the guard half of WS-03's MVP-349.

**Review focus** (numbered 1-5 in skeleton order)

REVIEW_FOCUS: 2=WS-04.7; 4=WS-04.2,WS-04.3,WS-04.7; 5=WS-04.1,WS-04.4,WS-04.11,WS-04.15

- 1 (ingress false negatives) and 3 (foreign identifiers under RLS): no WS-04 code.
- 2 (alert-path failures that lose or strand pages): WS-04.7's promtool cases in `test_promtool_rule_unit_tests_pass` pin that a failed delivery (`CrisisAlertDeliveryFailed`), a stranded outbox row (`CrisisOutboxBacklog`), consumer lag, an unresolved DLQ entry and an unacknowledged episode (`CrisisAlertsUnacknowledged`) each page the operator; the delivery logic itself is WS-06's and WS-13's.
- 4 (PHI through unplanned sinks): WS-04.2 `test_traceback_text_is_phi_sanitized` and `test_stdlib_and_uvicorn_records_are_json` (tracebacks and uvicorn records go through the sanitizer); WS-04.3 `test_unhandled_exception_body_has_no_exception_text`, `test_validation_error_shape_has_no_input_echo` and `test_access_line_has_route_template_and_no_query` (no exception text in error bodies, no query string, so no WS token, in access lines, route templates as metric labels); WS-04.7 `test_annotations_never_template_user_data` and the e2e `test_service_not_ready_reaches_mailpit` (alert emails carry no PHI).
- 5 (strict boots with partial or drifting config): WS-04.1 `test_each_read_variable_is_required_in_strict` and `test_strict_requires_verify_full`; WS-04.4 `test_strict_postgres_failure_is_fatal` and `test_strict_requires_redis_url`; WS-04.11 `test_known_dev_values_are_rejected`, `test_low_entropy_is_rejected` and `test_plain_env_secret_is_rejected_in_strict`; WS-04.15 `test_adoption_that_drops_phi_handling_is_caught` (a renamed or dropped PHI activation cannot pass silently).

**Interface consistency**
- IF-01 names, fields, defaults, route bodies, start order (guards, logging, datastores, [W1] tokens and event plane, [W2] codec, [W3] audit, readiness) and the 5 s teardown timeout match the skeleton; `create_service_app` installs `install_rate_limiting` from W1 when `spec.rate_limited`.
- IF-18 metric names and labels match; the W1 and W4 rule lists match, plus the two additions listed above.
- Env names consumed (`REDIS_URL`, `CORS_ALLOWED_ORIGINS`, `ENVIRONMENT`, `*_FILE`) match IF-02/IF-22; the Alertmanager names are registered by WS-03.6.
- WS-03 tasks this plan depends on (WS-03.6, WS-03.8, WS-03.9, WS-03.13) exist in the paired WS-03 plan with the same scope.

**Open risks**
- The conformance probe imports each service in a subprocess with synthetic env; if an adoption keeps a different import layout than the table (notably notification-service, whose layout WS-01/WS-13 normalize in W0), the table row is updated by ledger Ruling in this lane.
- Starlette's middleware ordering and the pure-ASGI context propagation are load-bearing for request ids in 500 bodies; WS-04.3's tests pin the behaviour.
- `smtp_auth_password_file` requires Alertmanager 0.26 or later; the pin is 0.27.0 and `amtool check-config` proves it.
- D-12 makes Amazon SES the clinician-alert relay. If `ALERTMANAGER_SMTP_SMARTHOST` also points at SES, an SES outage that makes `CrisisAlertDeliveryFailed` fire also blocks the email that reports it, so nobody is paged about the failed crisis alert. The operator should route Alertmanager through a different provider (D-12 already puts the fallback mailbox on a different provider). Raise it as a ledger Ruling at W1 entry if the owner wants this enforced rather than documented in `docs/runbooks/alerting.md`.
- The autoheal socket proxy is the only container with Docker socket access; a proxy CVE would widen it. It sits on an internal network shared only with autoheal and Alloy.
- `ContainerUnhealthy` fires once after every deploy restart; the runbook documents silencing during planned deploys.
- WS-05 helper names (`fault`, `Stack.urls`) are used as IF-19 describes them; the e2e files adapt to the final conftest without changing assertions.
- `test_phi_encryption_activation.py` is edited by two lanes (WS-04.15 in W0, WS-12.12 in W3). The waves keep them apart, and the request to WS-12 above keeps the classification complete when safety moves.

## Review responses

- Checker error: "Wave 1: `docs/runbooks/alerting.md` is edited by several workstreams in the same wave (WS-04.7, WS-13.10)": rejected. WS-13.10 does not edit that file. Its only docs entry is `- Create: docs/runbooks/crisis-escalation.md (... links to docs/runbooks/alerting.md for the WS-04 alerts)` (WS-13-notification-alert-delivery.md:1479), where `alerting.md` is a link target, not a file the task writes. WS-13's owned paths (WS-13-notification-alert-delivery.md:25) list only `docs/runbooks/crisis-escalation.md`, and `docs/runbooks/alerting.md` is a WS-04 owned path (this plan, line 21; skeleton `owned_paths` of WS-04). The order already holds as well: WS-04.7 creates the file in W1 phase A, ahead of WS-13.10 in phase B. WS-04.7's Produces line now states this.
- Checker error: "`docs/runbooks/alerting.md` is created by several tasks: ['WS-04.7', 'WS-13.10']": rejected, same evidence as above. Only WS-04.7 creates it; WS-04.12, WS-04.13 and WS-04.14 modify it, in order.
- Checker error: plan a task for WS-16's request R3 (retire `tests/integration/test_phi_encryption_activation.py` in W0, CR-8 of WS-09; needed by WS-16.1): fixed with new task WS-04.15 (W0, phase A, depends on WS-04.5), whose Produces line names WS-16.1, plus WS-17.1, WS-18.1, WS-18.2, WS-11.1, WS-15.1 and WS-12.12. The file is replaced, not deleted. CR-8 itself (WS-09-phi-codec-audit-llm.md:2060) says "keep the file and keep `safety_service` in `_PHI_ENCRYPTING` until WS-12.12 moves it out in W3", and WS-12.12 edits the file (WS-12-safety-detection-lexicon.md:1136). WS-13's regressions also run it (WS-13-notification-alert-delivery.md:140, 243, 572, 1951). WS-04.15 drops the `configure_phi_encryption` requirement for the five services whose W0 adoption removes the call (today they all call it: `services/{diagnosis,memory,orchestrator,personality,therapy}_service/src/main.py`). It accepts a `ServiceRuntimeSpec(..., handles_phi=True)` instead and keeps a detector that catches a PHI service with neither. The conformance half of CR-8 (`handles_phi` for the seven PHI services) was already pinned by WS-04.5's `SERVICES` table. A new request to WS-12 keeps safety classified when WS-12.12 moves it.
- Checker warnings: none were reported.
- crisis-review F11: fixed (in WS-12.9 and WS-06.10). WS-04.3 needs no change: it already exports the counter as `CRISIS_DETECTED` (New interfaces, standard metric objects). WS-12.9 now increments that object with `source="safety"` and depends on WS-04.3, and WS-12's R-12-5 is marked "Acknowledged, no task".
- crisis-review F13: fixed. Verified: `NotificationConsumerLag` fired only above 10 messages for 5 minutes, one stuck crisis turn is 4 messages, and `/ready` only reports that the consume task runs, so a consumer that is alive but makes no progress alerted nobody. WS-04.7 adds `CrisisConsumerStalled` (`max by (group, topic)` of the `solace-group-notifications`/`solace.safety` lag `> 0` for 3m, `path: crisis`) and switches `NotificationConsumerLag` from `sum` to `max`, because from WS-06.10 every event plane exports the crisis pair and a `sum` would multiply it. The rules test and the promtool cases pin both, including two exporters of the same pair.
- decision 2026-10-19 D-07: WS-04.11's gate and prerequisites now name the SOPS + age source. Strict secrets are rendered by WS-03.8 from `deploy/secrets/<env>.sops.yaml` into root-only 0400 tmpfs files, so the guards and their tests are unchanged, because they check NAME_FILE provenance, not the store. Step 3 notes that alertmanager reads its password file as its image user (65534), and a new WS-03 row asks WS-03.8's `SECRET_READER_UIDS` to map alertmanager to 65534.
- decision 2026-10-19 D-04: the WS-09 shutdown request (needed by WS-04.4) now also covers the botocore `bedrock-runtime` client of the Bedrock provider that strict environments use instead of Portkey. No WS-04 task routes LLM traffic, so nothing else changes.
- decision 2026-10-19 D-12: Open risks gains the shared-relay risk. If Alertmanager also sends through SES, an SES outage would hide its own `CrisisAlertDeliveryFailed` page, so the operator should use a different provider, or a W1 ledger Ruling enforces it. WS-04.7's rules and tests are unchanged.
