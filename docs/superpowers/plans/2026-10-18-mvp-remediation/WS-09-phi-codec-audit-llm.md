# WS-09: PHI protection: field codec, audit trail, log sanitizer and LLM gateway client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the PHI controls real: one fast AES-256-GCM field codec bound to the row owner and rotatable by key id, an append-only HMAC-chained audit trail that verifies under concurrent writers, a log sanitizer that redacts free text and never raises, and an LLM gateway client that fails loudly, never caches PHI across users, honours the PHI-egress decision and, in staging and production, sends real-user traffic only to Anthropic models on AWS Bedrock under the AWS BAA (D-04, decided 2026-10-19).

**Architecture:** `solace_security.phi_codec` (new, IF-13) derives one data key per key id with HKDF from `ENCRYPTION_MASTER_KEY` and `ENCRYPTION_RETIRED_KEYS_JSON`; the service lanes adopt it in their repositories in W2 phase B and WS-09 locks the adoption in W3. `solace_security.audit_runtime` (new, IF-14) writes `audit_events` (slot 011) under one `pg_advisory_xact_lock` chain and `verify_chain` recomputes every HMAC; `phi_protection.phi_sanitizer_processor` keeps its name but applies a key- and shape-based redaction policy; `services/shared/infrastructure/llm_client.py` (IF-15) raises `LLMUnavailableError`, returns `LLMResult` from the fallback path, keeps caching off and gates real providers on `LLM_PHI_EGRESS_APPROVED` in strict environments; strict environments accept only `LLM_PROVIDER=bedrock`, the Bedrock Converse provider (WS-09.18, D-04), and Portkey remains a development and test route with synthetic data. Every WS-09 production path is a platform path, so every task runs in phase A of its wave, except WS-09.15: it creates only a test file (no platform or service path) and runs in W3 phase B, after the service lanes' codec-adoption tasks it locks (WS-12.12 is itself a W3 phase-B task).

**Covers:** 16 register entries (blocker 2, high 6, medium 6, low 2).

**Waves:** 0 (phase A1: master-key format, LLM init), 1 (phase A1: LLM error surface, Bedrock provider, key names and egress gate, caching), 2 (phase A: history trimming, codec, key rotation), 3 (phase A: legacy audit model, slot 011, audit runtime, PHI-access dependency and verifier, verify CLI, crisis pseudonym, sanitizer; phase B: MVP-055 closure lock WS-09.15).

**Depends on:**
- WS-01: cryptography, portkey-ai, prometheus-client, pytest-asyncio and testcontainers pins (lock); `boto3` and `botocore` in the runtime lock (D-04, CR-16); `realinfra` marker registration.
- WS-02: slot `011_audit_trail` scaffold on the fixed chain, `solace_app`/`solace_migrator` roles and default privileges (009).
- WS-04: `solace_infrastructure.runtime.is_strict_environment` (W0 phase A2, used from W1); `solace_infrastructure.runtime.metrics.AUDIT_WRITE_FAILURES` (WS-04.3, W0); it wires `get_phi_codec` (WS-04.8, W2) and `configure_service_audit` (WS-04.10, W3) into the bootstrap.
- WS-05: IF-19 `pg_migrated` fixture and `tests/realinfra/conftest.py` for the slot 011 and audit-chain tests.
- WS-07: `solace_security.middleware.get_current_user` / `AuthenticatedUser` (with `via_service` from W1) used by `audit_phi_access`.
- WS-10, WS-12, WS-14, WS-15, WS-16, WS-17, WS-18: codec adoption in their repositories (W2 phase B: WS-10.10, WS-14.10, WS-15.8, WS-16.8, WS-16.9, WS-17.5, WS-18.4; W3 phase B: WS-12.12); MVP-055 closes only after all of them (WS-09.15).

**Owned paths:** `src/solace_security/encryption.py`, `src/solace_security/phi_protection.py`, `src/solace_security/audit.py`, `src/solace_security/phi_codec.py`, `src/solace_security/audit_runtime.py`, `services/shared/**`, `migrations/versions/011_audit_trail.py`. New tests go to `tests/solace_security/test_mvp_*.py`, `tests/integration/phi-codec-audit-llm/` and `tests/realinfra/phi-codec-audit-llm/`.

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Test-location note for reviewers:** `services/shared` has no test directory, so LLM-client tests live in `tests/integration/phi-codec-audit-llm/`. Existing test files that encode retired behaviour but live in other workstreams' paths (`services/therapy_service/tests/test_llm_client.py`, `tests/integration/test_sprint7_portkey_fallback.py`, `tests/integration/test_phi_encryption_activation.py`) are changed only through the cross-workstream requests below; WS-09 tasks name the exact assertions that are expected to fail until those supporting edits land in the same wave.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `src/solace_security/encryption.py` | Modify | Master-key parsing (`parse_master_key`, `master_key_bytes`), `EncryptionSettings.key_id` and `retired_keys_json`, strict-env helper | WS-09.1, WS-09.7, WS-09.8 |
| `src/solace_security/phi_codec.py` | Create | IF-13 codec: envelope, AAD, HKDF per key id, keyring, search hash, field helpers, process singleton; salt-only `crisis_pseudonym` | WS-09.7, WS-09.8, WS-09.17 |
| `src/solace_security/audit.py` | Modify | Legacy model: full-field hash, recomputing `verify_integrity`, shared tip, retired `PostgresAuditStore`, IF-14 `AuditEventType` values, `AuditSettings.hmac_retired_keys_json` | WS-09.9, WS-09.11 |
| `src/solace_security/audit_runtime.py` | Create | IF-14 runtime: `AuditKeyring`, `AuditWriter`, `configure_service_audit`, `audit`, `verify_chain`, `audit_phi_access`, `audit_chain_verifier`, gauges; `python -m solace_security.audit_runtime verify` CLI | WS-09.11, WS-09.12, WS-09.16 |
| `src/solace_security/phi_protection.py` | Modify | Log-event redaction policy, non-string handling, never-raise processor, token key matching, UUID protection | WS-09.13, WS-09.14 |
| `services/shared/infrastructure/llm_client.py` | Modify | IF-15: init semantics, `LLMConfigurationError`, `shutdown()` closes the SDK client, `LLMUnavailableError`, `LLMResult`, egress gate, `readiness()`, caching policy, history trimming; the `LLM_PROVIDER` switch and the Bedrock Converse provider (D-04) | WS-09.2 to WS-09.6, WS-09.18 |
| `services/shared/infrastructure/__init__.py` | Modify | Export the new LLM error/result types | WS-09.2, WS-09.3 |
| `services/shared/__init__.py` | Modify | Export the new LLM error/result types | WS-09.3 |
| `migrations/versions/011_audit_trail.py` | Modify | Slot 011 body: append-only `audit_events` | WS-09.10 |
| `tests/solace_security/test_mvp_092_master_key_format.py` | Create (test) | MVP-092 | WS-09.1 |
| `tests/integration/phi-codec-audit-llm/test_mvp_199_llm_init.py` | Create (test) | MVP-199 | WS-09.2 |
| `tests/integration/phi-codec-audit-llm/test_mvp_139_llm_unavailable.py` | Create (test) | MVP-139 | WS-09.3 |
| `tests/integration/phi-codec-audit-llm/test_mvp_057_llm_settings_egress.py` | Create (test) | MVP-057 | WS-09.4 |
| `tests/integration/phi-codec-audit-llm/test_mvp_056_llm_caching.py` | Create (test) | MVP-056 | WS-09.5 |
| `tests/integration/phi-codec-audit-llm/test_mvp_205_llm_history.py` | Create (test) | MVP-205 | WS-09.6 |
| `tests/integration/phi-codec-audit-llm/test_mvp_d04_bedrock_provider.py` | Create (test) | D-04 Bedrock provider: request shape, no caching, error mapping, strict refusals, client config | WS-09.18 |
| `tests/solace_security/test_mvp_055_phi_codec.py` | Create (test) | MVP-055 codec, MVP-187 | WS-09.7 |
| `tests/solace_security/test_mvp_351_codec_rotation.py` | Create (test) | MVP-351 codec part | WS-09.8 |
| `tests/solace_security/test_mvp_345_legacy_audit_integrity.py` | Create (test) | MVP-345/346 legacy model, IF-14 types | WS-09.9 |
| `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_events_schema.py` | Create (test) | Slot 011 append-only behaviour | WS-09.10 |
| `tests/realinfra/phi-codec-audit-llm/test_mvp_346_audit_chain_concurrency.py` | Create (test) | Chain under 3 writers, tamper, rotation, configured path | WS-09.11 |
| `tests/solace_security/test_mvp_091_audit_runtime.py` | Create (test) | Details allow-list, hash coverage, never-raise, retry policy | WS-09.11 |
| `tests/solace_security/test_mvp_091_audit_phi_access.py` | Create (test) | PHI-access dependency, verifier gauges | WS-09.12 |
| `tests/solace_security/test_mvp_093_log_sanitizer.py` | Create (test) | MVP-093, MVP-315, MVP-316 | WS-09.13 |
| `tests/solace_security/test_mvp_189_uuid_masking.py` | Create (test) | MVP-189 | WS-09.14 |
| `tests/solace_security/test_mvp_055_codec_adoption.py` | Create (test) | MVP-055 closure lock | WS-09.15 |
| `tests/solace_security/test_mvp_091_audit_verify_cli.py` | Create (test) | Verify CLI output, exit codes, `--start-seq`, `--textfile`, config errors | WS-09.16 |
| `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_verify_cli_pg.py` | Create (test) | `python -m solace_security.audit_runtime verify` as a subprocess against slot 011 | WS-09.16 |
| `tests/solace_security/test_mvp_029_crisis_pseudonym.py` | Create (test) | Salt-only crisis pseudonym equal to the codec's search hash | WS-09.17 |

## Tasks

### Task WS-09.1: Master key accepts the documented 64-hex and base64 formats

**Covers:** MVP-092
**Severity:** high
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_security/encryption.py` (new module function `parse_master_key(value: str, *, strict: bool) -> bytes` and private `_strict_environment() -> bool`; `EncryptionSettings.model_post_init` validates through `parse_master_key` and stores the decoded bytes in a private attribute exposed by a new read-only property `master_key_bytes`; `EncryptionSettings.for_development` raises in staging as well as production; `KeyManager.__init__` sets `self._master_key = self._settings.master_key_bytes` instead of `master_key.get_secret_value().encode()`)
- Test: `tests/solace_security/test_mvp_092_master_key_format.py` (create)

**Interfaces:**
- Consumes: none (phase A1 runs before IF-01 exists, so strictness is read locally: `os.environ.get("ENVIRONMENT", "development").strip().lower() in {"staging", "production"}`, the same rule as IF-01 `is_strict_environment`).
- Produces: `parse_master_key(value: str, *, strict: bool) -> bytes` returning exactly 32 bytes; raises `ValueError` whose message contains "exactly 32 bytes", "64 hex characters" and `secrets.token_hex(32)` and never contains the value. `EncryptionSettings.master_key_bytes: bytes`. WS-04's ProductionGuards calls `parse_master_key(value, strict=True)` (cross-workstream request CR-2); WS-09.7 builds the codec from `master_key_bytes`.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_092_master_key_format.py`
```python
"""MVP-092: ENCRYPTION_MASTER_KEY accepts the documented 64-hex recipe and 'base64:' keys."""
from __future__ import annotations

import base64
import secrets

import pytest

from solace_security.encryption import EncryptionSettings, Encryptor


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")


def test_mvp_092_documented_hex_recipe_is_accepted():
    key = secrets.token_bytes(32).hex()  # docs/STAGING-BRINGUP.md recipe: 64 hex characters
    settings = EncryptionSettings(master_key=key, _env_file=None)
    assert settings.master_key_bytes == bytes.fromhex(key)
    assert len(settings.master_key_bytes) == 32


def test_mvp_092_base64_prefixed_key_is_accepted():
    raw = bytes(range(32))
    settings = EncryptionSettings(master_key="base64:" + base64.b64encode(raw).decode(), _env_file=None)
    assert settings.master_key_bytes == raw


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_hex_key_accepted_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    assert EncryptionSettings(master_key="ab" * 32, _env_file=None).master_key_bytes == bytes.fromhex("ab" * 32)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_legacy_32_char_key_rejected_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(ValueError, match="64 hex characters"):
        EncryptionSettings(master_key="a" * 32, _env_file=None)


def test_mvp_092_legacy_32_char_key_still_accepted_in_test():
    assert EncryptionSettings(master_key="a" * 32, _env_file=None).master_key_bytes == b"a" * 32


@pytest.mark.parametrize("bad", ["short-key", "zz" * 32, "base64:" + base64.b64encode(b"x" * 31).decode()])
def test_mvp_092_error_message_names_working_formats(bad):
    with pytest.raises(ValueError) as exc:
        EncryptionSettings(master_key=bad, _env_file=None)
    message = str(exc.value)
    assert "exactly 32 bytes" in message
    assert "64 hex characters" in message
    assert "secrets.token_hex(32)" in message
    assert bad not in message  # never echo key material


def test_mvp_092_encryptor_round_trips_with_hex_key():
    encryptor = Encryptor(EncryptionSettings(master_key="cd" * 32, _env_file=None))
    assert encryptor.decrypt_to_string(encryptor.encrypt("phi-value", "field"), "field") == "phi-value"


def test_mvp_092_parse_master_key_strict_flag():
    from solace_security.encryption import parse_master_key

    assert parse_master_key("ef" * 32, strict=True) == bytes.fromhex("ef" * 32)
    with pytest.raises(ValueError):
        parse_master_key("a" * 32, strict=True)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_for_development_refused_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(RuntimeError):
        EncryptionSettings.for_development()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_092_master_key_format.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_092_documented_hex_recipe_is_accepted` raises `ValueError: ENCRYPTION_MASTER_KEY must be exactly 32 bytes, got 64 bytes (64 characters)` (the documented recipe is rejected); the staging legacy-key case fails with `DID NOT RAISE` (staging accepts a raw 32-character key); `test_mvp_092_error_message_names_working_formats` fails because the message recommends `secrets.token_bytes(32).hex()` and has no "64 hex characters"; `test_mvp_092_for_development_refused_in_strict_envs[staging]` fails with `DID NOT RAISE`; `test_mvp_092_parse_master_key_strict_flag` fails with `ImportError`.
- [ ] **Step 3: Implement** `parse_master_key(value: str, *, strict: bool) -> bytes` in `src/solace_security/encryption.py`. Accepted formats, checked in this order: a `base64:` prefix whose remainder decodes with `base64.b64decode(..., validate=True)` to exactly 32 bytes; `re.fullmatch(r"[0-9A-Fa-f]{64}", value)` decoded with `bytes.fromhex`; and, only when `strict` is false, a value whose UTF-8 encoding is exactly 32 bytes, used as-is (legacy development/test keys, so `for_development()` and existing suites keep working). Anything else raises `ValueError("ENCRYPTION_MASTER_KEY must be exactly 32 bytes of key material: 64 hex characters (generate with: python -c \"import secrets; print(secrets.token_hex(32))\") or 'base64:' followed by base64 of 32 random bytes; a raw 32-character value is accepted only in development and test.")`. `model_post_init` calls it with `strict=_strict_environment()`; `for_development` raises when `_strict_environment()` is true. KeyManager therefore derives from the same 32 bytes for legacy keys (byte-identical to today) and from the decoded bytes for hex/base64 keys.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suites. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_infrastructure -o "addopts=" -p no:logging -q`. Expected: all pass (existing `test_master_key_exact_length` and the multibyte test still match "exactly 32 bytes").
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.2: LLM client initialisation never raises for missing credentials

**Covers:** MVP-199 (plus the `shutdown()` half of MVP-332, owned by WS-04, requested for WS-04.4)
**Severity:** medium
**Wave:** 0
**Depends on:** none
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (decided 2026-10-19, changed: strict environments send real-user traffic only to Anthropic models on AWS Bedrock (WS-09.18), and Portkey is a development and test route with synthetic data; unchanged: without credentials the LLM is disabled and callers use their template fallbacks; tests use only fakes)
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (new `class LLMConfigurationError(Exception)`; `UnifiedLLMClient.__init__(self, settings=None, *, client_factory=None)`; `initialize()` rewritten; `shutdown()` closes the SDK client; private `_strict_environment() -> bool` with the IF-01 rule; module-level `_default_client_factory(**kwargs)` that imports `portkey_ai.AsyncPortkey` lazily)
- Modify: `services/shared/infrastructure/__init__.py` (export `LLMConfigurationError`)
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_199_llm_init.py` (create)

**Interfaces:**
- Consumes: none.
- Produces: `LLMConfigurationError`; `UnifiedLLMClient(settings: LLMClientSettings | None = None, *, client_factory: Callable[..., Any] | None = None)`, where `client_factory(**kwargs)` receives `api_key=<LLM_PORTKEY_API_KEY>`, `base_url=<LLM_PORTKEY_GATEWAY_URL>`, `config=<portkey config dict>` and returns an object with `chat.completions.create(**request)`. `initialize()` contract: an empty `LLM_PORTKEY_API_KEY` constructs no client and leaves `is_initialized=True`, `is_available=False` (WARNING `llm_disabled_missing_credentials` outside strict, ERROR inside strict; variable names only); portkey-ai not importable raises `LLMConfigurationError` in strict and disables outside; an exception from the factory raises `LLMConfigurationError` in strict and disables outside (logging only the exception type name). WS-09.4 adds the egress rule on top. `async shutdown() -> None` (for WS-04.4's ordered teardown, MVP-332): awaits the client's `close()` (or `aclose()`) when the client has one, logs WARNING `llm_client_close_failed` with the exception type name only if that raises, never raises, then drops the reference so `is_available` is False. From WS-09.18 (D-04), strict environments refuse `LLM_PROVIDER=portkey` before any of these rules run, so the Portkey rules here govern development and test, and the same SDK-import and factory rules apply to boto3 on the Bedrock route.

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_199_llm_init.py`
```python
"""MVP-199: UnifiedLLMClient.initialize() never raises for missing credentials."""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient


class _StrictSdk:
    """Mimics portkey-ai 1.9.x: AsyncPortkey(api_key='') with no PORTKEY_API_KEY raises ValueError."""

    def __init__(self, *, api_key, base_url=None, config=None, **_):
        if not api_key:
            raise ValueError("Portkey API Key Not Found")
        self.api_key = api_key
        self.config = config


@pytest.fixture
def fake_portkey(monkeypatch):
    module = types.ModuleType("portkey_ai")
    module.AsyncPortkey = _StrictSdk
    monkeypatch.setitem(sys.modules, "portkey_ai", module)
    for name in ("PORTKEY_API_KEY", "LLM_PROVIDER", "BEDROCK_MODEL_ID", "BEDROCK_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    return module


def _settings(**kwargs):
    return LLMClientSettings(_env_file=None, **kwargs)


@pytest.mark.parametrize("env", ["development", "test"])
async def test_mvp_199_missing_key_does_not_raise_outside_strict(fake_portkey, monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("")))
    await client.initialize()
    assert client.is_initialized is True
    assert client.is_available is False


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_199_missing_key_in_strict_disables_without_raising(fake_portkey, monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")  # D-04: the only strict route (WS-09.18); no setting reads it before WS-09.18
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("")))
    await client.initialize()
    assert client.is_available is False


async def test_mvp_199_configured_key_builds_client_via_factory(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return object()

    client = UnifiedLLMClient(
        _settings(portkey_api_key=SecretStr("pk-test"), anthropic_api_key=SecretStr("sk-test")),
        client_factory=factory,
    )
    await client.initialize()
    assert client.is_available is True
    assert calls[0]["api_key"] == "pk-test"


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_199_missing_sdk_is_a_startup_error_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setitem(sys.modules, "portkey_ai", None)  # makes the lazy import raise ImportError
    from services.shared.infrastructure.llm_client import LLMConfigurationError

    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("pk-test")))
    with pytest.raises(LLMConfigurationError):
        await client.initialize()


class _ClosableClient:
    def __init__(self, fail):
        self.closed = False
        self.fail = fail

    async def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("connection pool already closed")


@pytest.mark.parametrize("fail", [False, True])
async def test_mvp_332_shutdown_closes_the_underlying_client(monkeypatch, fail):
    monkeypatch.setenv("ENVIRONMENT", "test")
    inner = _ClosableClient(fail)
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("pk-test")), client_factory=lambda **_: inner)
    await client.initialize()
    await client.shutdown()  # never raises, even when close() does
    assert inner.closed is True
    assert client.is_available is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_199_llm_init.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Both missing-key tests raise `ValueError: Portkey API Key Not Found` out of `initialize()` (non-ImportError exceptions are re-raised at llm_client.py:124-127, the therapy/personality crash-loop in the register); the factory test fails with `TypeError: ... unexpected keyword argument 'client_factory'`; the strict SDK test fails with `ImportError: cannot import name 'LLMConfigurationError'`; both shutdown cases fail with the same `TypeError` today, and once the factory exists they would still fail on `inner.closed is True`, because `shutdown()` only drops the reference (llm_client.py:363-372) and leaks the SDK's HTTP connection pool (MVP-332).
- [ ] **Step 3: Implement** `async def initialize(self) -> None` in `services/shared/infrastructure/llm_client.py`: if `portkey_api_key` is empty, log and return disabled without calling the factory; otherwise call `self._client_factory(api_key=..., base_url=self._settings.portkey_gateway_url, config=self._build_portkey_config())` (default `_default_client_factory`), mapping `ImportError` and any other exception per the contract above; never call `AsyncPortkey` with an empty key, so the SDK's unprefixed `PORTKEY_API_KEY` fallback can never silently enable a client. Rewrite `shutdown()` per the produced contract (look up `close`, then `aclose`, with `getattr`; await the result when it is awaitable).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_initialize_without_portkey` still sees `_initialized is True` and `_client is None`; `TestUnifiedLLMClient::test_shutdown` still sees `_initialized is False` and `_client is None`, because a `MagicMock` client's `close()` result is not awaitable).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.3: generate() raises LLMUnavailableError and the fallback path returns LLMResult

**Covers:** MVP-139
**Severity:** medium
**Wave:** 1
**Depends on:** WS-09.2
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (new `class LLMUnavailableError(Exception)` with attributes `reason: str` and `service: str`; new `@dataclass(frozen=True) class LLMResult: text: str; degraded: bool; provider: str`; `generate()` raises instead of returning `"(LLM unavailable)"` or `""`; `generate_with_fallback(...) -> LLMResult`)
- Modify: `services/shared/infrastructure/__init__.py` and `services/shared/__init__.py` (export `LLMUnavailableError`, `LLMResult`)
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_139_llm_unavailable.py` (create)

**Interfaces:**
- Consumes: `UnifiedLLMClient(..., client_factory=...)` from WS-09.2.
- Produces (IF-15): `LLMUnavailableError.reason` in {"not_configured", "disabled", "timeout", "provider_error", "empty_response"}; `str(error) == f"LLM unavailable ({reason})"` and never contains prompt, response or provider error text; `generate_with_fallback(system_prompt, user_message, fallback_response, **kwargs) -> LLMResult` returns `LLMResult(text=fallback_response, degraded=True, provider="fallback")` on `LLMUnavailableError` and `LLMResult(text=<reply>, degraded=False, provider=settings.primary_provider)` otherwise. `_error_count` increments for timeout, provider_error and empty_response; the log line `llm_generation_failed` carries `reason`, `service` and `error_type` (exception class name) only.

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_139_llm_unavailable.py`
```python
"""MVP-139: generate() raises LLMUnavailableError; generate_with_fallback returns LLMResult."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")


def _client(create=None, *, key="pk-test", timeout=1):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = LLMClientSettings(_env_file=None, portkey_api_key=SecretStr(key), timeout_seconds=timeout)
    return UnifiedLLMClient(settings, client_factory=lambda **_: fake)


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


async def test_mvp_139_unavailable_client_never_returns_a_sentinel():
    client = _client(key="")
    await client.initialize()
    with pytest.raises(Exception) as exc:
        await client.generate(system_prompt="s", user_message="hello", service_name="t")
    assert type(exc.value).__name__ == "LLMUnavailableError"
    assert exc.value.reason == "not_configured"


async def test_mvp_139_provider_error_raises_without_echoing_text():
    from services.shared.infrastructure.llm_client import LLMUnavailableError

    async def boom(**_):
        raise RuntimeError("upstream 500: prompt was 'hello there'")

    client = _client(boom)
    await client.initialize()
    with pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="hello there", service_name="t")
    assert exc.value.reason == "provider_error"
    assert "hello" not in str(exc.value)
    assert client.get_statistics()["total_errors"] == 1


async def test_mvp_139_timeout_raises_with_reason_timeout():
    from services.shared.infrastructure.llm_client import LLMUnavailableError

    async def hang(**_):
        await asyncio.sleep(30)

    client = _client(hang)
    await client.initialize()
    with pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="u", service_name="t")
    assert exc.value.reason == "timeout"


async def test_mvp_139_empty_completion_raises():
    from services.shared.infrastructure.llm_client import LLMUnavailableError

    async def empty(**_):
        return SimpleNamespace(choices=[])

    client = _client(empty)
    await client.initialize()
    with pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="u", service_name="t")
    assert exc.value.reason == "empty_response"


async def test_mvp_139_fallback_returns_degraded_llmresult():
    from services.shared.infrastructure.llm_client import LLMResult

    client = _client(key="")
    await client.initialize()
    result = await client.generate_with_fallback(
        system_prompt="s", user_message="u", fallback_response="template reply", service_name="t"
    )
    assert result == LLMResult(text="template reply", degraded=True, provider="fallback")


async def test_mvp_139_fallback_success_is_not_degraded():
    from services.shared.infrastructure.llm_client import LLMResult

    async def ok(**_):
        return _reply("real reply")

    client = _client(ok)
    await client.initialize()
    result = await client.generate_with_fallback(
        system_prompt="s", user_message="u", fallback_response="template", service_name="t"
    )
    assert result == LLMResult(text="real reply", degraded=False, provider="anthropic")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_139_llm_unavailable.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_139_unavailable_client_never_returns_a_sentinel` fails with `DID NOT RAISE` because `generate()` returns `"(LLM unavailable)"` (llm_client.py:216-222); the other tests fail with `ImportError: cannot import name 'LLMUnavailableError'` / `'LLMResult'` (today a provider error is swallowed into `""`, llm_client.py:253-256).
- [ ] **Step 3: Implement** `async def generate(self, system_prompt: str, user_message: str, conversation_history: list[dict[str, str]] | None = None, context: str | None = None, service_name: str = "unknown", metadata: dict[str, Any] | None = None, max_tokens: int | None = None, temperature: float | None = None, task_type: str | None = None) -> str` raising `LLMUnavailableError` (`"disabled"` is reserved for WS-09.4's egress gate; until then a missing client means `"not_configured"`), and `async def generate_with_fallback(...) -> LLMResult`. `asyncio.TimeoutError` maps to `"timeout"`; an extracted empty string maps to `"empty_response"`; any other exception maps to `"provider_error"` raised `from None` so the provider text is not chained into logs.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass except exactly these cross-workstream-owned assertions, which fail with `LLMUnavailableError` or a non-`str` result until their supporting edits land in this wave: `services/therapy_service/tests/test_llm_client.py::TestUnifiedLLMClient::test_generate_no_client`, `::test_generate_error_handling`, `::test_generate_with_fallback`, `::test_generate_with_fallback_on_empty` (CR-5) and `tests/integration/test_sprint7_portkey_fallback.py::TestLLMRequestTimeout::test_generate_returns_empty_on_timeout_not_hang` (CR-6). Record the five node ids in the ledger.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.4: LLM settings read the documented keys and honour the PHI-egress decision

**Covers:** MVP-057
**Severity:** high
**Wave:** 1
**Depends on:** WS-09.18
**Implementer model:** sonnet
**Review:** standard (Opus reviewer per the WS-09 model policy)
**Decision gate:** D-04 (decided 2026-10-19, changed: strict environments run only `LLM_PROVIDER=bedrock` (WS-09.18) and the BAA that matters is the AWS BAA, accepted by the owner in AWS Artifact; unchanged: in strict environments `LLM_PHI_EGRESS_APPROVED=false` disables the real provider until that BAA is accepted and recorded; approved plus missing configuration, here `BEDROCK_MODEL_ID`, is a startup error)
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (`LLMClientSettings.phi_egress_approved: bool = False`, read as `LLM_PHI_EGRESS_APPROVED`; `initialize()` strict gating for both providers; new `async def readiness(self) -> tuple[bool, str]`; `_strict_environment()` now delegates to `solace_infrastructure.runtime.is_strict_environment` through a function-local import)
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_057_llm_settings_egress.py` (create)

**Interfaces:**
- Consumes: `is_strict_environment()` (IF-01, WS-04); WS-09.18 `LLMClientSettings.provider`, `BedrockSettings`, `bedrock_client_factory` and its strict routing refusals.
- Produces (IF-15): strict + not approved → no client is built for either provider, `is_available=False`, `readiness() == (False, "disabled: LLM_PHI_EGRESS_APPROVED is false")`, `generate()` raises `LLMUnavailableError(reason="disabled")`; strict + approved + missing configuration → `initialize()` raises `LLMConfigurationError` whose message lists the missing variable names and no values (strict environments run only `LLM_PROVIDER=bedrock` from WS-09.18, so the name is `BEDROCK_MODEL_ID`; the Bedrock route has no API key, its credentials come from the host's instance role); outside strict the approval flag is not required and missing configuration disables the client. `readiness()` returns `(True, "ok")` when available, `(False, "disabled: missing BEDROCK_MODEL_ID")` for an unconfigured Bedrock client and `(False, "disabled: missing LLM_PORTKEY_API_KEY")` for an unconfigured Portkey client (development and test). Service lanes register it with `runtime.register_readiness_check("llm", client.readiness, critical=False)` (CR-7).

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_057_llm_settings_egress.py`
```python
"""MVP-057: the LLM client reads only the documented names and honours LLM_PHI_EGRESS_APPROVED (D-04: strict runs Bedrock)."""
from __future__ import annotations

import pytest

from services.shared.infrastructure.llm_client import (
    BedrockSettings,
    LLMClientSettings,
    LLMConfigurationError,
    UnifiedLLMClient,
)

_NAMES = (
    "LLM_PORTKEY_API_KEY", "LLM_ANTHROPIC_API_KEY", "LLM_OPENAI_API_KEY", "LLM_PHI_EGRESS_APPROVED",
    "LLM_ENABLE_FALLBACK", "LLM_PROVIDER", "PORTKEY_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "BEDROCK_MODEL_ID", "BEDROCK_ENDPOINT_URL",
)
MODEL_ID = "test.synthetic-model-v1"  # synthetic; never a real AWS model id


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


def _factory(calls):
    def factory(**kwargs):
        calls.append(kwargs)
        return object()

    return factory


def _bedrock_client(calls, **bedrock):
    return UnifiedLLMClient(
        LLMClientSettings(_env_file=None),
        bedrock_settings=BedrockSettings(**bedrock),
        bedrock_client_factory=_factory(calls),
    )


async def test_mvp_057_undocumented_names_are_never_read(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-unprefixed-value")  # the name STAGING-BRINGUP documents
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-unprefixed-value")  # the name .env.example documents
    calls = []
    client = UnifiedLLMClient(LLMClientSettings(_env_file=None), client_factory=_factory(calls))
    await client.initialize()
    assert client.is_available is False
    assert calls == []
    assert await client.readiness() == (False, "disabled: missing LLM_PORTKEY_API_KEY")


async def test_mvp_057_strict_approved_requires_bedrock_model_id(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("LLM_PHI_EGRESS_APPROVED", "true")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-unprefixed-value")
    calls = []
    client = _bedrock_client(calls)
    with pytest.raises(LLMConfigurationError) as exc:
        await client.initialize()
    assert "BEDROCK_MODEL_ID" in str(exc.value)
    assert "pk-unprefixed-value" not in str(exc.value)
    assert calls == []


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_057_strict_not_approved_is_disabled_and_reported(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    calls = []
    client = _bedrock_client(calls, model_id=MODEL_ID)
    await client.initialize()
    assert client.is_available is False
    assert calls == []
    assert await client.readiness() == (False, "disabled: LLM_PHI_EGRESS_APPROVED is false")


async def test_mvp_057_strict_approved_bedrock_is_available(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("LLM_PHI_EGRESS_APPROVED", "true")
    calls = []
    client = _bedrock_client(calls, model_id=MODEL_ID)
    await client.initialize()
    assert client.is_available is True
    assert len(calls) == 1
    assert await client.readiness() == (True, "ok")


async def test_mvp_057_bedrock_without_model_id_is_reported(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    client = _bedrock_client([])
    await client.initialize()
    assert await client.readiness() == (False, "disabled: missing BEDROCK_MODEL_ID")


async def test_mvp_057_documented_names_reach_the_gateway(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LLM_PORTKEY_API_KEY", "pk-test")
    monkeypatch.setenv("LLM_ANTHROPIC_API_KEY", "sk-test")
    calls = []
    client = UnifiedLLMClient(LLMClientSettings(_env_file=None), client_factory=_factory(calls))
    await client.initialize()
    assert client.is_available is True
    assert calls[0]["api_key"] == "pk-test"
    assert calls[0]["config"]["targets"][0]["api_key"] == "sk-test"
    assert await client.readiness() == (True, "ok")


def test_mvp_057_egress_flag_defaults_to_false():
    assert LLMClientSettings(_env_file=None).phi_egress_approved is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_057_llm_settings_egress.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_057_strict_approved_requires_bedrock_model_id` fails with `DID NOT RAISE` (WS-09.18 only disables a Bedrock client without a model id, so an approved production deployment would start with the LLM silently off); `test_mvp_057_strict_not_approved_is_disabled_and_reported` fails on `assert client.is_available is False` (no egress gate exists, so with a model id set WS-09.18 builds the client in strict without approval); the other tests fail with `AttributeError` on `readiness` and `phi_egress_approved`.
- [ ] **Step 3: Implement** the gating in `initialize()` after WS-09.18's strict routing refusals and before any factory call, in this order: strict and not approved → disabled with reason "LLM_PHI_EGRESS_APPROVED is false"; collect the missing names for the selected provider (`bedrock`: `BEDROCK_MODEL_ID`; `portkey`, development and test only: `LLM_PORTKEY_API_KEY`); strict and approved and missing → `raise LLMConfigurationError("LLM configuration missing: " + ", ".join(names))`; non-strict and missing → disabled; else build the client through the selected provider's factory. Add `async def readiness(self) -> tuple[bool, str]`. The variable names come from the `LLM_` prefix of `LLMClientSettings` and the `BEDROCK_` prefix of `BedrockSettings`; no setting reads an unprefixed name.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass except the five node ids recorded in WS-09.3 Step 5 (CR-5, CR-6).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.5: Gateway caching off by default and never shared across users

**Covers:** MVP-056
**Severity:** high
**Wave:** 1
**Depends on:** WS-09.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (decided 2026-10-19; this part is unchanged from the default: `LLM_ENABLE_CACHING=false` for every request; a semantic cache is never used; strict environments refuse caching outright; the Bedrock route from WS-09.18 never caches, so `LLM_ENABLE_CACHING=true` is refused with `LLM_PROVIDER=bedrock` as well)
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (`LLMClientSettings.enable_caching` default `False`, `cache_mode` default `"simple"`; a `model_validator(mode="after")` rejecting `cache_mode != "simple"` and rejecting `enable_caching=True` in strict environments or with `provider == "bedrock"`; `_build_portkey_config(self, cache_namespace: str | None = None)` adds `config["cache"] = {"mode": "simple"}` only when caching is enabled and a namespace is given; per-namespace clients built through `client_factory(..., cache_namespace=ns)` and kept in an LRU of at most 256 entries; requests without `metadata["user_id"]` use the uncached client)
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_056_llm_caching.py` (create)

**Interfaces:**
- Consumes: `client_factory` from WS-09.2.
- Produces (IF-15): cache namespace format `f"solace:user:{metadata['user_id']}"`; validator errors mention "semantic" or "LLM_ENABLE_CACHING"; the default factory forwards `cache_namespace` to `AsyncPortkey` (portkey-ai 1.9.x constructor parameter, per the MVP-056 evidence `client.py:272`); per-namespace clients exist only on the Portkey route, which D-04 limits to development and test.

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_056_llm_caching.py`
```python
"""MVP-056: Portkey caching is off by default, never semantic and never shared across users."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("LLM_ENABLE_CACHING", "LLM_CACHE_MODE", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")


def test_mvp_056_caching_off_by_default():
    settings = LLMClientSettings(_env_file=None)
    assert settings.enable_caching is False
    assert settings.cache_mode == "simple"
    assert "cache" not in UnifiedLLMClient(settings)._build_portkey_config()


def test_mvp_056_semantic_cache_rejected():
    with pytest.raises(ValueError, match="semantic"):
        LLMClientSettings(_env_file=None, enable_caching=True, cache_mode="semantic")


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_056_caching_refused_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(ValueError, match="LLM_ENABLE_CACHING"):
        LLMClientSettings(_env_file=None, enable_caching=True, cache_mode="simple")


def test_mvp_056_caching_refused_with_bedrock():
    with pytest.raises(ValueError, match="LLM_ENABLE_CACHING"):
        LLMClientSettings(_env_file=None, provider="bedrock", enable_caching=True, cache_mode="simple")


async def test_mvp_056_cache_namespace_is_per_user():
    created = []

    def factory(**kwargs):
        created.append(kwargs)

        async def create(**_request):
            content = f"r:{kwargs.get('cache_namespace')}"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    settings = LLMClientSettings(
        _env_file=None, portkey_api_key=SecretStr("pk-test"), enable_caching=True, cache_mode="simple"
    )
    client = UnifiedLLMClient(settings, client_factory=factory)
    await client.initialize()
    a = await client.generate(system_prompt="s", user_message="hi", service_name="t", metadata={"user_id": "user-a"})
    b = await client.generate(system_prompt="s", user_message="hi", service_name="t", metadata={"user_id": "user-b"})
    anon = await client.generate(system_prompt="s", user_message="hi", service_name="t")
    assert (a, b, anon) == ("r:solace:user:user-a", "r:solace:user:user-b", "r:None")
    namespaced = [kw for kw in created if kw.get("cache_namespace")]
    assert {kw["cache_namespace"] for kw in namespaced} == {"solace:user:user-a", "solace:user:user-b"}
    assert all(kw["config"]["cache"] == {"mode": "simple"} for kw in namespaced)
    assert all("cache" not in kw["config"] for kw in created if not kw.get("cache_namespace"))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_056_llm_caching.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_056_caching_off_by_default` fails on `assert settings.enable_caching is False` (default True, mode "semantic", llm_client.py:37-38, and `config["cache"] == {"mode": "semantic"}` with no namespace); the validator tests fail with `DID NOT RAISE` (`test_mvp_056_caching_refused_with_bedrock` too: WS-09.18's Bedrock route ignores the caching settings but nothing refuses them); the namespace test fails because every request shares one client (`r:None` for all three).
- [ ] **Step 3: Implement** the settings validator, `_build_portkey_config(self, cache_namespace: str | None = None) -> dict[str, Any]` and `_client_for(self, metadata: Mapping[str, Any] | None)` used by `_execute_completion`. The uncached base client is the one built in `initialize()`; namespaced clients are created lazily with the same `api_key`/`base_url` and `config=self._build_portkey_config(cache_namespace=ns)`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass except the WS-09.3 node ids plus, in `services/therapy_service/tests/test_llm_client.py`, `TestLLMClientSettings::test_default_settings`, `TestUnifiedLLMClient::test_build_portkey_config_with_caching` and `TestUnifiedLLMClient::test_get_statistics` (they assert the semantic default; CR-5). Record the node ids in the ledger.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.6: History trimming keeps system messages and pinned context

**Covers:** MVP-205
**Severity:** medium
**Wave:** 2
**Depends on:** WS-09.5
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (module constants `MAX_HISTORY_MESSAGES = 10` and `_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})`; `_build_messages(self, system_prompt, user_message, conversation_history, context, pinned_context=None)`; `generate(..., pinned_context: list[dict[str, str]] | None = None)` forwards it)
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_205_llm_history.py` (create)

**Interfaces:**
- Consumes: none new.
- Produces (IF-15): message order `[system(system_prompt + context)] + pinned_context + leading system messages of the history + the last MAX_HISTORY_MESSAGES other history messages + [user(user_message)]`; any role outside `_ALLOWED_ROLES` becomes "user"; pinned items are never trimmed. Consumers may pass memory context as `pinned_context` (CR-5, optional for WS-15).
- Produces for requesting tasks: the last W2 change to `services/shared/infrastructure/llm_client.py`, so the client's final W2 shape (IF-15 errors and results from WS-09.3, cache defaults from WS-09.5, `pinned_context` here) is what WS-15.22's seven rewritten `services/therapy_service/tests/test_llm_client.py` assertions lock (CR-5). WS-15.22 depends on this task and edits no client code; any client change those tests seem to need is made here or in WS-09.3/WS-09.5, never in WS-15's lane.

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_205_llm_history.py`
```python
"""MVP-205: history trimming keeps leading system context and pinned context."""
from __future__ import annotations

from types import SimpleNamespace

from pydantic import SecretStr

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient


def _client():
    return UnifiedLLMClient(LLMClientSettings(_env_file=None))


def test_mvp_205_leading_system_context_survives_trimming():
    history = [{"role": "system", "content": "[User Context] memories"}]
    history += [{"role": "user", "content": f"m{i}"} for i in range(10)]
    messages = _client()._build_messages("SYS", "now", history, None)
    contents = [m["content"] for m in messages]
    assert "[User Context] memories" in contents
    assert contents[0] == "SYS" and contents[-1] == "now"
    assert len(messages) == 13


def test_mvp_205_history_trimmed_oldest_first():
    history = [{"role": "user", "content": f"m{i}"} for i in range(15)]
    contents = [m["content"] for m in _client()._build_messages("SYS", "now", history, None)]
    assert contents[1:-1] == [f"m{i}" for i in range(5, 15)]


def test_mvp_205_pinned_context_never_trimmed():
    history = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    messages = _client()._build_messages(
        "SYS", "now", history, None, pinned_context=[{"role": "system", "content": "PINNED"}]
    )
    assert messages[1] == {"role": "system", "content": "PINNED"}
    assert len(messages) == 13


def test_mvp_205_roles_restricted():
    history = [{"role": "tool", "content": "x"}, {"role": "developer", "content": "y"}]
    roles = [m["role"] for m in _client()._build_messages("SYS", "now", history, None)]
    assert roles == ["system", "user", "user", "user"]


async def test_mvp_205_generate_forwards_pinned_context(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    seen = {}

    async def create(**request):
        seen.update(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = UnifiedLLMClient(
        LLMClientSettings(_env_file=None, portkey_api_key=SecretStr("pk-test")), client_factory=lambda **_: fake
    )
    await client.initialize()
    await client.generate(
        system_prompt="SYS",
        user_message="now",
        conversation_history=[{"role": "user", "content": f"m{i}"} for i in range(30)],
        pinned_context=[{"role": "system", "content": "PINNED"}],
        service_name="t",
    )
    assert seen["messages"][1] == {"role": "system", "content": "PINNED"}
    assert len(seen["messages"]) == 13
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_205_llm_history.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_205_leading_system_context_survives_trimming` fails on `"[User Context] memories" in contents` (history[-10:] drops therapy's memory block, llm_client.py:272-277); `test_mvp_205_roles_restricted` fails because raw roles pass through; the pinned tests fail with `TypeError: ... unexpected keyword argument 'pinned_context'`.
- [ ] **Step 3: Implement** `_build_messages(...) -> list[dict[str, str]]` exactly in the produced order and add `pinned_context` to `generate()` and `generate_with_fallback()` (passed through `**kwargs`).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_build_messages_limits_history` still sees 12 messages), except the seven `test_llm_client.py` cases CR-5 lists, which stay red from WS-09.3/WS-09.5 until WS-15.22 rewrites them after this task (record them in the ledger as expected failures).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.7: PHI field codec (IF-13)

**Covers:** MVP-055, MVP-187
**Severity:** high
**Wave:** 2
**Depends on:** WS-09.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `src/solace_security/phi_codec.py` (`PHICodec`, `PHIConfigError`, `PHIDecryptError`, `get_phi_codec`, `reset_phi_codec`, module-level `encrypt_fields`/`decrypt_fields`, private `_derive_data_key(master: bytes, key_id: str) -> bytes`)
- Modify: `src/solace_security/encryption.py` (`EncryptionSettings.key_id: str = "k1"` read as `ENCRYPTION_KEY_ID` and validated with `^[A-Za-z0-9_-]{1,32}$`; `_strict_environment()` now calls `solace_infrastructure.runtime.is_strict_environment` through a function-local import)
- Test: `tests/solace_security/test_mvp_055_phi_codec.py` (create)

**Interfaces:**
- Consumes: `EncryptionSettings.master_key_bytes` (WS-09.1); `is_strict_environment()` (IF-01).
- Produces (IF-13): `PHICodec(*, active_key_id: str, keys: Mapping[str, bytes], search_salt: bytes, allow_plaintext: bool)` where `keys` maps key id to a 32-byte master key and every data key is derived once in the constructor with `HKDF(SHA256, length=32, salt=None, info=b"solace-phi:" + key_id)`; `PHICodec.from_settings(settings: EncryptionSettings, *, strict: bool | None = None) -> PHICodec` (strict default from `is_strict_environment()`; `allow_plaintext = not strict`); `encrypt(value: str | None, *, field: str, owner_id: str | UUID) -> str | None` returning `"enc:v1:<key_id>:<base64url(nonce||ciphertext||tag) without padding>"` with AES-256-GCM, 12-byte random nonce and AAD `f"{field}:{owner}".encode()` where `owner` is the canonical lower-case UUID string when `owner_id` parses as a UUID, else `str(owner_id)`; `decrypt(value, *, field, owner_id) -> str | None` (non-`enc:v1:` values returned as-is only when `allow_plaintext`, otherwise `PHIDecryptError`; unknown key id → `PHIDecryptError("unknown key id <id>")`; tag failure → `PHIDecryptError("ciphertext authentication failed for field <field>")`); `search_hash(value: str, *, purpose: str) -> str` = `hmac_sha256(search_salt, purpose + "\x1f" + value).hexdigest()`; `encrypt_fields(row, fields, *, owner_id) -> dict` / `decrypt_fields(...)` (copy of the row; the AAD field name is the column key; `None` stays `None`; non-`str` values raise `TypeError`); `active_key_id -> str`; `get_phi_codec() -> PHICodec` (process singleton from `EncryptionSettings()`; any settings or key error becomes `PHIConfigError` with no key material; strict requires a search salt of at least 16 bytes); `reset_phi_codec() -> None` (tests). Module-level `encrypt_fields(row, fields, *, owner_id)` and `decrypt_fields(...)` delegate to `get_phi_codec()`. WS-04 warms it at startup (CR-8); service lanes adopt it (CR-10).

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_055_phi_codec.py`
```python
"""MVP-055 / MVP-187: the IF-13 PHI codec encrypts with owner-bound AAD and one key derivation per key id."""
from __future__ import annotations

import time
import uuid

import pytest

from solace_security.phi_codec import PHICodec, PHIConfigError, PHIDecryptError, get_phi_codec, reset_phi_codec

KEY = bytes(range(32))
SALT = b"s" * 32
OWNER = uuid.UUID("11111111-2222-4333-8444-555555555555")


@pytest.fixture
def codec():
    return PHICodec(active_key_id="k1", keys={"k1": KEY}, search_salt=SALT, allow_plaintext=False)


def test_mvp_055_round_trip_and_envelope(codec):
    ciphertext = codec.encrypt("I took all my pills", field="safety_assessments.content", owner_id=OWNER)
    assert ciphertext.startswith("enc:v1:k1:")
    assert "pills" not in ciphertext
    assert codec.decrypt(ciphertext, field="safety_assessments.content", owner_id=OWNER) == "I took all my pills"


def test_mvp_055_nonce_is_random(codec):
    assert codec.encrypt("same", field="f", owner_id=OWNER) != codec.encrypt("same", field="f", owner_id=OWNER)


def test_mvp_055_aad_binds_owner_and_field(codec):
    ciphertext = codec.encrypt("x", field="users.email", owner_id=OWNER)
    with pytest.raises(PHIDecryptError):
        codec.decrypt(ciphertext, field="users.email", owner_id=uuid.uuid4())
    with pytest.raises(PHIDecryptError):
        codec.decrypt(ciphertext, field="users.bio", owner_id=OWNER)


def test_mvp_055_owner_id_forms_are_equivalent(codec):
    ciphertext = codec.encrypt("x", field="f", owner_id=OWNER)
    assert codec.decrypt(ciphertext, field="f", owner_id=str(OWNER).upper()) == "x"


def test_mvp_055_none_passthrough(codec):
    assert codec.encrypt(None, field="f", owner_id=OWNER) is None
    assert codec.decrypt(None, field="f", owner_id=OWNER) is None


def test_mvp_055_plaintext_rejected_when_strict(codec):
    with pytest.raises(PHIDecryptError):
        codec.decrypt("plain text", field="f", owner_id=OWNER)


def test_mvp_055_plaintext_tolerated_in_dev():
    dev = PHICodec(active_key_id="k1", keys={"k1": KEY}, search_salt=SALT, allow_plaintext=True)
    assert dev.decrypt("plain text", field="f", owner_id=OWNER) == "plain text"


def test_mvp_055_search_hash_is_deterministic_and_purpose_scoped(codec):
    first = codec.search_hash("alice@example.test", purpose="users.email")
    assert first == codec.search_hash("alice@example.test", purpose="users.email")
    assert len(first) == 64
    assert first != codec.search_hash("alice@example.test", purpose="other")


def test_mvp_055_field_helpers_round_trip(codec):
    row = {"id": OWNER, "email": "a@b.test", "bio": None, "role": "user"}
    encrypted = codec.encrypt_fields(row, ["email", "bio"], owner_id=OWNER)
    assert encrypted["email"].startswith("enc:v1:k1:") and encrypted["bio"] is None and encrypted["role"] == "user"
    assert codec.decrypt_fields(encrypted, ["email", "bio"], owner_id=OWNER) == row
    with pytest.raises(TypeError):
        codec.encrypt_fields({"n": 5}, ["n"], owner_id=OWNER)


def test_mvp_055_get_phi_codec_strict_without_key_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env file is read
    for name in ("ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "ENCRYPTION_KEY_ID", "ENCRYPTION_RETIRED_KEYS_JSON"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    reset_phi_codec()
    try:
        with pytest.raises(PHIConfigError):
            get_phi_codec()
    finally:
        reset_phi_codec()


def test_mvp_055_get_phi_codec_is_a_singleton_with_env_key_id(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "ab" * 32)
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", "search-salt-for-tests-0123456789")
    monkeypatch.setenv("ENCRYPTION_KEY_ID", "k7")
    monkeypatch.delenv("ENCRYPTION_RETIRED_KEYS_JSON", raising=False)
    reset_phi_codec()
    try:
        first = get_phi_codec()
        assert first is get_phi_codec()
        assert first.active_key_id == "k7"
        assert first.encrypt("x", field="f", owner_id=OWNER).startswith("enc:v1:k7:")
    finally:
        reset_phi_codec()


def test_mvp_187_one_key_derivation_per_key_id(monkeypatch):
    import solace_security.phi_codec as phi_codec

    calls = []
    real = phi_codec._derive_data_key
    monkeypatch.setattr(phi_codec, "_derive_data_key", lambda master, key_id: calls.append(key_id) or real(master, key_id))
    codec = PHICodec(active_key_id="k1", keys={"k1": KEY}, search_salt=SALT, allow_plaintext=False)
    for _ in range(500):
        codec.decrypt(codec.encrypt("note", field="f", owner_id=OWNER), field="f", owner_id=OWNER)
    assert calls == ["k1"]


def test_mvp_187_2000_round_trips_under_two_seconds(codec):
    start = time.perf_counter()
    for _ in range(2000):
        codec.decrypt(codec.encrypt("x" * 200, field="f", owner_id=OWNER), field="f", owner_id=OWNER)
    assert time.perf_counter() - start < 2.0
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_055_phi_codec.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'solace_security.phi_codec'`: the only encryption path today is the ORM-listener `FieldEncryptor`, which no repository calls (MVP-055) and which runs a 600k-iteration PBKDF2 on every field (MVP-187).
- [ ] **Step 3: Implement** `src/solace_security/phi_codec.py` to the produced contract with `cryptography.hazmat.primitives.ciphers.aead.AESGCM` and `cryptography.hazmat.primitives.kdf.hkdf.HKDF`. Envelope parsing splits on ":" at most 3 times and pads the base64url part before decoding; a malformed envelope raises `PHIDecryptError("malformed ciphertext for field <field>")`. Nothing in this module logs or raises with a plaintext, key or ciphertext value. Do not import it from `src/solace_security/__init__.py` (shared-files rule: full-path imports only).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.8: Key ids and retired keys: rotation without data loss

**Covers:** MVP-351
**Severity:** medium
**Wave:** 2
**Depends on:** WS-09.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (decided 2026-10-19, changed: SOPS + age replace the default's cloud secrets store; the consumer contract is unchanged: active and retired keys live in the per-environment SOPS file `deploy/secrets/<env>.sops.yaml`, which `deploy/render_secrets.py` renders into root-only tmpfs `/run/secrets` files read through `NAME_FILE`, and each environment's age private key is held off-host by the owner or operator; rotating means a new `ENCRYPTION_MASTER_KEY` with a new `ENCRYPTION_KEY_ID`, the old pair moves into `ENCRYPTION_RETIRED_KEYS_JSON`, and the file is re-encrypted and redeployed)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/phi_codec.py` (`PHICodec.from_settings` parses retired keys into the keyring)
- Modify: `src/solace_security/encryption.py` (`EncryptionSettings.retired_keys_json: SecretStr = SecretStr("{}")` read as `ENCRYPTION_RETIRED_KEYS_JSON`; docstrings state that `enable_key_rotation` and `key_rotation_days` are advisory and that rotation is by key id)
- Test: `tests/solace_security/test_mvp_351_codec_rotation.py` (create)

**Interfaces:**
- Consumes: WS-09.7 codec.
- Produces (IF-13): `ENCRYPTION_RETIRED_KEYS_JSON` is a JSON object `{key_id: master_key}` whose values use the `parse_master_key` formats (strict rules apply); retired keys decrypt only; invalid JSON, an invalid key id, an invalid key or an active key id that also appears in the retired map raises `PHIConfigError` without echoing values.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_351_codec_rotation.py`
```python
"""MVP-351: rotating ENCRYPTION_MASTER_KEY keeps old rows readable through key ids and retired keys."""
from __future__ import annotations

import json
import uuid

import pytest

from solace_security.encryption import EncryptionSettings
from solace_security.phi_codec import PHICodec, PHIConfigError, PHIDecryptError

M1 = "11" * 32
M2 = "22" * 32
SALT = "search-salt-for-tests-0123456789"
OWNER = uuid.UUID("11111111-2222-4333-8444-555555555555")
FIELD = "therapy_sessions.notes"


def _settings(**kwargs):
    return EncryptionSettings(_env_file=None, search_hash_salt=SALT, **kwargs)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    for name in ("ENCRYPTION_MASTER_KEY", "ENCRYPTION_KEY_ID", "ENCRYPTION_RETIRED_KEYS_JSON"):
        monkeypatch.delenv(name, raising=False)


def test_mvp_351_rotated_master_still_decrypts_old_rows():
    old = PHICodec.from_settings(_settings(master_key=M1, key_id="k1"), strict=True)
    ciphertext = old.encrypt("old note", field=FIELD, owner_id=OWNER)
    new = PHICodec.from_settings(
        _settings(master_key=M2, key_id="k2", retired_keys_json=json.dumps({"k1": M1})), strict=True
    )
    assert new.decrypt(ciphertext, field=FIELD, owner_id=OWNER) == "old note"
    assert new.encrypt("new note", field=FIELD, owner_id=OWNER).startswith("enc:v1:k2:")
    assert new.active_key_id == "k2"


def test_mvp_351_missing_retired_key_is_explicit():
    old = PHICodec.from_settings(_settings(master_key=M1, key_id="k1"), strict=True)
    ciphertext = old.encrypt("old note", field=FIELD, owner_id=OWNER)
    new = PHICodec.from_settings(_settings(master_key=M2, key_id="k2"), strict=True)
    with pytest.raises(PHIDecryptError, match="unknown key id"):
        new.decrypt(ciphertext, field=FIELD, owner_id=OWNER)


@pytest.mark.parametrize(
    "retired", ["not-json", json.dumps({"bad:id": M1}), json.dumps({"k1": "short"}), json.dumps({"k2": M1})]
)
def test_mvp_351_invalid_retired_keys_are_config_errors(retired):
    with pytest.raises(PHIConfigError) as exc:
        PHICodec.from_settings(_settings(master_key=M2, key_id="k2", retired_keys_json=retired), strict=True)
    assert M1 not in str(exc.value)


def test_mvp_351_retired_keys_never_rendered():
    settings = _settings(master_key=M2, key_id="k2", retired_keys_json=json.dumps({"k1": M1}))
    assert M1 not in repr(settings) and M1 not in str(settings.model_dump())
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_351_codec_rotation.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_351_rotated_master_still_decrypts_old_rows` raises `PHIDecryptError: unknown key id k1` (the `retired_keys_json` keyword is silently ignored by `extra="ignore"`, so after rotation old rows are undecryptable, the data-loss scenario in MVP-351); the invalid-config cases fail with `DID NOT RAISE`.
- [ ] **Step 3: Implement** the settings field and the keyring merge in `PHICodec.from_settings`: `keys = {settings.key_id: settings.master_key_bytes} | {kid: parse_master_key(value, strict=strict) for kid, value in json.loads(...).items()}` after validating every key id with the same pattern and refusing overlap with the active id.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.9: Legacy audit model: full-field hash, recomputing verification, shared tip and IF-14 event types

**Covers:** MVP-345, MVP-346
**Severity:** blocker
**Wave:** 3
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: the authoritative trail is Postgres `audit_events` written by `audit_runtime`; the legacy in-process classes remain only for tests and must not under-report tampering)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/audit.py` (`AuditEventType` gains the IF-14 members plus `ADMIN_ASSIGNMENT_CHANGED` and `CONSENT_CHANGED`; `AuditEvent.compute_hash` covers every field except `event_hash`; new `AuditStore.iter_range(start_id, end_id) -> list[AuditEvent]` implemented by `InMemoryAuditStore`; `AuditLogger.verify_integrity` recomputes every hash in range and checks links; `AuditLogger.log` reads `self._store.get_latest_hash()` under its lock for every event and keeps `_last_hash` updated; `AsyncAuditStore.iter_range` default raises `NotImplementedError` and `AsyncAuditLogger.verify_integrity` returns False when it cannot recompute; `PostgresAuditStore.initialize()` raises `RuntimeError` naming `solace_security.audit_runtime`)
- Test: `tests/solace_security/test_mvp_345_legacy_audit_integrity.py` (create)

**Interfaces:**
- Consumes: none.
- Produces (IF-14 event types; member name → value): AUTH_LOGIN_SUCCEEDED "auth_login_succeeded", AUTH_LOGIN_FAILED "auth_login_failed", AUTH_LOCKOUT "auth_lockout", AUTH_LOGOUT "auth_logout", AUTH_TOKEN_REFRESHED "auth_token_refreshed", AUTH_TOKEN_REVOKED "auth_token_revoked", AUTH_PASSWORD_CHANGED "auth_password_changed", AUTH_PASSWORD_RESET "auth_password_reset", AUTH_MFA_ENROLLED "auth_mfa_enrolled", AUTH_MFA_FAILED "auth_mfa_failed", AUTH_OAUTH_LOGIN "auth_oauth_login", ACCOUNT_DELETED "account_deleted", ACCOUNT_ERASED_IN_SERVICE "account_erased_in_service", PHI_READ "phi_read", PHI_WRITE "phi_write", PHI_DELETE "phi_delete", CRISIS_ALERT_SENT "crisis_alert_sent", CRISIS_ALERT_FAILED "crisis_alert_failed", CRISIS_ALERT_ACKNOWLEDGED "crisis_alert_acknowledged", ADMIN_CLINICIAN_PROVISIONED "admin_clinician_provisioned", ADMIN_ONCALL_CHANGED "admin_oncall_changed", plus new ADMIN_ASSIGNMENT_CHANGED "admin_assignment_changed" and CONSENT_CHANGED "consent_changed" (see New interfaces). Existing members are unchanged.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_345_legacy_audit_integrity.py`
```python
"""MVP-345 / MVP-346: the legacy audit model hashes every field, recomputes on verify and shares one tip."""
from __future__ import annotations

import pytest

from solace_security.audit import (
    AuditActor,
    AuditEventType,
    AuditLogger,
    AuditOutcome,
    AuditResource,
    AuditSettings,
    InMemoryAuditStore,
    PostgresAuditStore,
)

KEY = "k" * 32
IF14 = {
    "AUTH_LOGIN_SUCCEEDED": "auth_login_succeeded", "AUTH_LOGIN_FAILED": "auth_login_failed",
    "AUTH_LOCKOUT": "auth_lockout", "AUTH_LOGOUT": "auth_logout", "AUTH_TOKEN_REFRESHED": "auth_token_refreshed",
    "AUTH_TOKEN_REVOKED": "auth_token_revoked", "AUTH_PASSWORD_CHANGED": "auth_password_changed",
    "AUTH_PASSWORD_RESET": "auth_password_reset", "AUTH_MFA_ENROLLED": "auth_mfa_enrolled",
    "AUTH_MFA_FAILED": "auth_mfa_failed", "AUTH_OAUTH_LOGIN": "auth_oauth_login",
    "ACCOUNT_DELETED": "account_deleted", "ACCOUNT_ERASED_IN_SERVICE": "account_erased_in_service",
    "PHI_READ": "phi_read", "PHI_WRITE": "phi_write", "PHI_DELETE": "phi_delete",
    "CRISIS_ALERT_SENT": "crisis_alert_sent", "CRISIS_ALERT_FAILED": "crisis_alert_failed",
    "CRISIS_ALERT_ACKNOWLEDGED": "crisis_alert_acknowledged",
    "ADMIN_CLINICIAN_PROVISIONED": "admin_clinician_provisioned", "ADMIN_ONCALL_CHANGED": "admin_oncall_changed",
    "ADMIN_ASSIGNMENT_CHANGED": "admin_assignment_changed", "CONSENT_CHANGED": "consent_changed",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")


def _settings():
    return AuditSettings(hmac_key=KEY, _env_file=None)


def _log(logger, action):
    return logger.log(
        AuditEventType.DATA_ACCESS, action, AuditOutcome.SUCCESS,
        AuditActor(actor_id="clin-1", roles=["clinician"], ip_address="10.0.0.1"),
        AuditResource(resource_type="diagnosis", resource_id="d-1", owner_id="patient-1", contains_phi=True),
        details={"route": "/x"},
    )


def test_mvp_345_content_tamper_is_detected():
    store = InMemoryAuditStore()
    logger = AuditLogger(store, _settings())
    first, second = _log(logger, "read:1"), _log(logger, "read:2")
    store._events[0] = first.model_copy(update={"actor": first.actor.model_copy(update={"actor_id": "other"}), "action": "read:x"})
    assert logger.verify_integrity(first.event_id, second.event_id) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.model_copy(update={"resource": e.resource.model_copy(update={"owner_id": "patient-2"})}),
        lambda e: e.model_copy(update={"resource": e.resource.model_copy(update={"contains_phi": False})}),
        lambda e: e.model_copy(update={"details": {"route": "/other"}}),
        lambda e: e.model_copy(update={"actor": e.actor.model_copy(update={"ip_address": "10.9.9.9"})}),
        lambda e: e.model_copy(update={"actor": e.actor.model_copy(update={"roles": ["admin"]})}),
    ],
)
def test_mvp_345_every_stored_field_is_hashed(mutate):
    logger = AuditLogger(InMemoryAuditStore(), _settings())
    event = _log(logger, "read:1")
    assert mutate(event).compute_hash("sha256", KEY) != event.event_hash


def test_mvp_345_deleted_middle_event_with_relinked_successor_is_detected():
    store = InMemoryAuditStore()
    logger = AuditLogger(store, _settings())
    a, _b, c = _log(logger, "a"), _log(logger, "b"), _log(logger, "c")
    store._events = [a, c.model_copy(update={"previous_hash": a.event_hash})]
    store._index = {a.event_id: 0, c.event_id: 1}
    assert logger.verify_integrity(a.event_id, c.event_id) is False


def test_mvp_346_two_loggers_on_one_store_keep_one_chain():
    store = InMemoryAuditStore()
    service_a, service_b = AuditLogger(store, _settings()), AuditLogger(store, _settings())
    a1 = _log(service_a, "a1")
    b1 = _log(service_b, "b1")
    a2 = _log(service_a, "a2")
    assert b1.previous_hash == a1.event_hash
    assert a2.previous_hash == b1.event_hash
    assert store.verify_chain(a1.event_id, a2.event_id) is True
    assert service_a.verify_integrity(a1.event_id, a2.event_id) is True


def test_mvp_091_if14_event_types_exist():
    for name, value in IF14.items():
        assert AuditEventType[name].value == value


async def test_mvp_345_postgres_audit_store_is_retired():
    with pytest.raises(RuntimeError, match="audit_runtime"):
        await PostgresAuditStore("postgresql://u:p@127.0.0.1:1/x").initialize()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_345_legacy_audit_integrity.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_345_content_tamper_is_detected` and the relink test fail on `is False` because `verify_integrity` only compares stored links (audit.py:250-261, 700-701); every `test_mvp_345_every_stored_field_is_hashed` case fails because `compute_hash` covers 8 fields (audit.py:134-155); `test_mvp_346_two_loggers_on_one_store_keep_one_chain` fails on `b1.previous_hash == a1.event_hash` (per-process tip, audit.py:609-610); the IF-14 test fails with `KeyError: 'AUTH_LOGIN_SUCCEEDED'`; the Postgres test fails because `initialize()` tries to connect instead of refusing.
- [ ] **Step 3: Implement** in `src/solace_security/audit.py`: `compute_hash` canonicalises `self.model_dump(mode="json", exclude={"event_hash"})` with `json.dumps(sort_keys=True, separators=(",", ":"))` (roles sorted first); `verify_integrity` walks `iter_range` in order, recomputes with the logger's key and requires each `previous_hash` to equal the prior event's stored hash. `existing tests/integration/test_audit_chain.py` keeps passing because untampered chains still verify and `_last_hash` is still rehydrated in `__init__`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.10: Slot 011: append-only audit_events

**Covers:** MVP-091, MVP-345
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-09.9, WS-02.3, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: `audit_events` in the same `solace` database, INSERT and SELECT only for `solace_app`, append-only, no PHI, retained 6 years and included in backups)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/011_audit_trail.py` (`upgrade()` and `downgrade()` only; `revision = "011_audit_trail"` and `down_revision` stay as WS-02 scaffolded them)
- Test: `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_events_schema.py` (create)

**Interfaces:**
- Consumes: IF-03 roles, default privileges (slot 009) and the slot-011 scaffold with its `revision`/`down_revision` (WS-02.3), IF-19 `pg_migrated` (WS-05.2). Also satisfies WS-02's request R-02.6 (revoke UPDATE, DELETE and TRUNCATE from `solace_app`).
- Produces: table `audit_events(chain_seq BIGSERIAL PRIMARY KEY, event_id UUID NOT NULL UNIQUE, occurred_at TIMESTAMPTZ NOT NULL, writer_service VARCHAR(64) NOT NULL, event_type VARCHAR(64) NOT NULL, actor_id VARCHAR(128) NOT NULL, actor_type VARCHAR(16) NOT NULL CHECK (actor_type IN ('user','service','system')), action VARCHAR(128) NOT NULL, outcome VARCHAR(16) NOT NULL, resource_type VARCHAR(64) NOT NULL, resource_id VARCHAR(128) NULL, subject_user_id UUID NULL, request_id VARCHAR(64) NULL, correlation_id VARCHAR(64) NULL, details JSONB NOT NULL DEFAULT '{}'::jsonb, details_sha256 CHAR(64) NOT NULL, hmac_key_id VARCHAR(32) NOT NULL, previous_hash CHAR(64) NULL, event_hash CHAR(64) NOT NULL)`; indexes `ix_audit_events_subject (subject_user_id, occurred_at)`, `ix_audit_events_type_time (event_type, occurred_at)`, `ix_audit_events_writer (writer_service, chain_seq)`; `REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM solace_app`, `GRANT SELECT, INSERT ON audit_events TO solace_app`, `GRANT USAGE, SELECT ON SEQUENCE audit_events_chain_seq_seq TO solace_app`; rules `audit_events_no_update` and `audit_events_no_delete` (`DO INSTEAD NOTHING`); function `audit_events_reject_truncate()` and statement trigger `audit_events_no_truncate BEFORE TRUNCATE` raising `'audit_events is append-only'`; table comment naming IF-14 and D-10. `downgrade()` drops the trigger, the function and the table.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_events_schema.py`
```python
"""MVP-091 / MVP-345: slot 011 creates an append-only audit_events table on real PostgreSQL."""
from __future__ import annotations

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.asyncio(loop_scope="session")]

_INSERT = (
    "INSERT INTO audit_events (chain_seq, event_id, occurred_at, writer_service, event_type, actor_id, actor_type,"
    " action, outcome, resource_type, details, details_sha256, hmac_key_id, event_hash)"
    " VALUES (nextval('audit_events_chain_seq_seq'), gen_random_uuid(), now(), 'schema-test', 'phi_read', 'u-1',"
    " 'user', 'GET /x', 'success', 'test', '{}'::jsonb, repeat('0', 64), 'kid', repeat('a', 64))"
)


async def test_mvp_091_solace_app_may_only_insert_and_select(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute(_INSERT)
        assert await conn.fetchval("SELECT count(*) FROM audit_events WHERE writer_service = 'schema-test'") >= 1
        for statement in (
            "UPDATE audit_events SET actor_id = 'x'",
            "DELETE FROM audit_events",
            "TRUNCATE audit_events",
        ):
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(statement)


async def test_mvp_345_owner_cannot_silently_rewrite_or_truncate(pg_migrated):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        await su.execute(_INSERT)
        rows_before = await su.fetchval("SELECT count(*) FROM audit_events")
        await su.execute("UPDATE audit_events SET actor_id = 'tampered'")
        assert await su.fetchval("SELECT count(*) FROM audit_events WHERE actor_id = 'tampered'") == 0
        await su.execute("DELETE FROM audit_events")
        assert await su.fetchval("SELECT count(*) FROM audit_events") == rows_before
        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            await su.execute("TRUNCATE audit_events")
    finally:
        await su.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_events_schema.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL with `asyncpg.exceptions.UndefinedTableError: relation "audit_events" does not exist`: no migration creates the table today (the DDL lives only in the never-called `PostgresAuditStore.initialize()`, audit.py:328), so no audit row can be written.
- [ ] **Step 3: Implement** the slot body with `op.execute(...)` statements in the produced order (table, indexes, privileges, rules, function, trigger, comment) and the reverse in `downgrade()`. Run DDL only; insert no rows.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations/test_chain_real_pg.py -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass (upgrade head, downgrade base, upgrade head; solace_app cannot run DDL).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.11: Audit runtime: serialized HMAC chain, writer API and verify_chain

**Covers:** MVP-091, MVP-345, MVP-346, MVP-351
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-09.10, WS-04.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: one global chain serialized by `pg_advisory_xact_lock(hashtext('solace_audit_chain'))`; `AUDIT_HMAC_KEY` from the secrets store; rows hold ids and codes only)
**Real infra:** yes
**Files:**
- Create: `src/solace_security/audit_runtime.py` (`AuditConfigError`, `AuditKeyring`, `AuditWriter`, `ChainVerification`, `ALLOWED_DETAIL_KEYS`, `sanitize_details`, `compute_event_hash`, `configure_service_audit`, `audit`, `verify_chain`, `verify_chain_on`, `reset_audit_runtime`, gauges `solace_audit_chain_verify_ok` and `solace_audit_chain_tip_seq`; failures are counted on WS-04.3's `solace_infrastructure.runtime.metrics.AUDIT_WRITE_FAILURES` (`solace_audit_write_failures_total{service}`), imported inside the failure path and never re-declared here, so a service that loads both modules does not hit `Duplicated timeseries`)
- Modify: `src/solace_security/audit.py` (`AuditSettings.hmac_retired_keys_json: str = Field(default="[]")`, read as `AUDIT_HMAC_RETIRED_KEYS_JSON`)
- Test: `tests/realinfra/phi-codec-audit-llm/test_mvp_346_audit_chain_concurrency.py` (create)
- Test: `tests/solace_security/test_mvp_091_audit_runtime.py` (create)

**Interfaces:**
- Consumes: slot 011 table (WS-09.10); `AuditEventType`/`AuditOutcome` (audit.py); `is_strict_environment()` (IF-01); the `AUDIT_WRITE_FAILURES` counter object (WS-04.3, IF-18).
- Produces (IF-14): `async def configure_service_audit(service_name: str, pool: asyncpg.Pool) -> None` (reads `AuditSettings()`; `AUDIT_HMAC_KEY` shorter than 32 UTF-8 bytes or invalid retired JSON raises `AuditConfigError` without echoing keys; runs no query); `async def audit(event_type: AuditEventType, *, actor_id: str, actor_type: Literal["user", "service", "system"], action: str, outcome: AuditOutcome, resource_type: str, resource_id: str | None = None, subject_user_id: str | None = None, details: Mapping[str, str | int | bool] | None = None) -> None` (never raises; fills `request_id`/`correlation_id` from `structlog.contextvars.get_contextvars()`; strict: 4 attempts with 0.05/0.1/0.2 s backoff, then CRITICAL log `audit_write_failed` and the counter; non-strict: 1 attempt, WARNING and the counter; unconfigured: counter with `service="unconfigured"`); `async def verify_chain(start_seq: int | None = None, end_seq: int | None = None) -> ChainVerification` over the configured pool and keyring. Helpers used by tests and WS-09.12: `AuditKeyring.from_keys(active: str, retired: Sequence[str] = ()) -> AuditKeyring` with `key_id_for(key) = sha256(b"solace-audit-key-id:" + key.encode()).hexdigest()[:12]`; `AuditWriter(service_name: str, pool, keyring)` with `async write(event_type, *, actor_id, actor_type, action, outcome, resource_type, resource_id=None, subject_user_id=None, details=None, request_id=None, correlation_id=None) -> int` (returns `chain_seq`, raises on failure); `async def verify_chain_on(pool, keyring, *, start_seq=None, end_seq=None, batch_size=1000) -> ChainVerification`; `@dataclass(frozen=True) ChainVerification(ok: bool, checked: int, first_bad_seq: int | None, reason: str, tip_seq: int | None, tip_hash: str | None)` with `reason` in {"ok", "details_mismatch", "unknown_key", "hash_mismatch", "link_mismatch", "not_configured"}; `compute_event_hash(fields: Mapping[str, Any], previous_hash: str | None, key: bytes) -> str`; `sanitize_details(details) -> dict[str, str | int | bool]`; `reset_audit_runtime() -> None`.

- [ ] **Step 1: Write the failing tests** in `tests/solace_security/test_mvp_091_audit_runtime.py` and `tests/realinfra/phi-codec-audit-llm/test_mvp_346_audit_chain_concurrency.py`
```python
# tests/solace_security/test_mvp_091_audit_runtime.py
"""MVP-091 / MVP-345: audit() details allow-list, full-field hash, never-raise and strict retry policy."""
from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

import solace_security.audit_runtime as ar
from solace_security.audit import AuditEventType, AuditOutcome


def _failures(service):
    return REGISTRY.get_sample_value("solace_audit_write_failures_total", {"service": service}) or 0.0


def test_mvp_091_details_allow_list_drops_and_redacts():
    details = {"route": "/api/v1/x/{id}", "status_code": 200, "mfa": True,
               "message": "I want to die", "reason_code": "has a space"}
    assert ar.sanitize_details(details) == {
        "route": "/api/v1/x/{id}", "status_code": 200, "mfa": True, "reason_code": "[REDACTED]",
    }


def test_mvp_345_event_hash_covers_every_field():
    key = b"k" * 32
    base = {
        "chain_seq": 1, "event_id": "e", "occurred_at": "2026-10-18T00:00:00.000000+00:00",
        "writer_service": "s", "event_type": "phi_read", "actor_id": "a", "actor_type": "user",
        "action": "GET /x", "outcome": "success", "resource_type": "r", "resource_id": None,
        "subject_user_id": None, "request_id": None, "correlation_id": None,
        "details_sha256": "0" * 64, "hmac_key_id": "kid",
    }
    digest = ar.compute_event_hash(base, None, key)
    for field in base:
        changed = dict(base, **{field: 2 if field == "chain_seq" else "changed"})
        assert ar.compute_event_hash(changed, None, key) != digest, field
    assert ar.compute_event_hash(base, "f" * 64, key) != digest


async def test_mvp_091_audit_never_raises_when_unconfigured():
    ar.reset_audit_runtime()
    before = _failures("unconfigured")
    result = await ar.audit(AuditEventType.PHI_READ, actor_id="u", actor_type="user", action="GET /x",
                            outcome=AuditOutcome.SUCCESS, resource_type="t")
    assert result is None
    assert _failures("unconfigured") == before + 1


class _BrokenPool:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        self.calls += 1
        raise ConnectionError("db down")


@pytest.mark.parametrize(("env", "attempts"), [("production", 4), ("test", 1)])
async def test_mvp_091_failure_policy_by_environment(monkeypatch, env, attempts):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("AUDIT_HMAC_KEY", "k" * 40)
    monkeypatch.setenv("AUDIT_HMAC_RETIRED_KEYS_JSON", "[]")
    pool = _BrokenPool()
    ar.reset_audit_runtime()
    await ar.configure_service_audit("svc-x", pool)
    before = _failures("svc-x")
    await ar.audit(AuditEventType.PHI_READ, actor_id="u", actor_type="user", action="GET /x",
                   outcome=AuditOutcome.SUCCESS, resource_type="t")
    assert pool.calls == attempts
    assert _failures("svc-x") == before + 1
    ar.reset_audit_runtime()


async def test_mvp_091_short_hmac_key_is_a_config_error(monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", "short")
    with pytest.raises(ar.AuditConfigError) as exc:
        await ar.configure_service_audit("svc-x", _BrokenPool())
    assert "short" not in str(exc.value)
```
```python
# tests/realinfra/phi-codec-audit-llm/test_mvp_346_audit_chain_concurrency.py
"""MVP-091/345/346/351: one unforked chain under three writers; tamper and key-rotation verification."""
from __future__ import annotations

import asyncio

import asyncpg
import pytest

from solace_security.audit import AuditEventType, AuditOutcome
from solace_security.audit_runtime import AuditKeyring, AuditWriter, verify_chain_on

pytestmark = [pytest.mark.realinfra, pytest.mark.asyncio(loop_scope="session")]
KEY_A = "audit-key-A-" + "a" * 32
KEY_B = "audit-key-B-" + "b" * 32
_RULES = ("CREATE RULE audit_events_no_update AS ON UPDATE TO audit_events DO INSTEAD NOTHING;"
          " CREATE RULE audit_events_no_delete AS ON DELETE TO audit_events DO INSTEAD NOTHING")


@pytest.fixture
async def su(pg_migrated):
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    await conn.execute("ALTER TABLE audit_events DISABLE TRIGGER audit_events_no_truncate;"
                       " TRUNCATE audit_events RESTART IDENTITY;"
                       " ALTER TABLE audit_events ENABLE TRIGGER audit_events_no_truncate")
    yield conn
    await conn.close()


async def _tamper(su, statement, *args):
    await su.execute("DROP RULE audit_events_no_update ON audit_events; DROP RULE audit_events_no_delete ON audit_events")
    try:
        await su.execute(statement, *args)
    finally:
        await su.execute(_RULES)


async def _write(writer, n, actor):
    for i in range(n):
        await writer.write(AuditEventType.PHI_READ, actor_id=f"{actor}-{i}", actor_type="user",
                           action="GET /api/v1/memory/context", outcome=AuditOutcome.SUCCESS,
                           resource_type="memory_context",
                           details={"route": "/api/v1/memory/context", "status_code": 200})


async def test_mvp_346_three_writers_produce_one_unforked_chain(pg_migrated, su):
    keyring = AuditKeyring.from_keys(KEY_A)
    pools = [await asyncpg.create_pool(pg_migrated.dsn_app, min_size=1, max_size=3) for _ in range(3)]
    try:
        writers = [AuditWriter(f"svc-{i}", pool, keyring) for i, pool in enumerate(pools)]
        await asyncio.gather(*(_write(w, 50, w.service_name) for w in writers))
        result = await verify_chain_on(pools[0], keyring)
        assert (result.ok, result.checked, result.reason) == (True, 150, "ok")
        counts = dict(await su.fetch("SELECT writer_service, count(*) FROM audit_events GROUP BY writer_service"))
        assert counts == {"svc-0": 50, "svc-1": 50, "svc-2": 50}
        assert await su.fetchval("SELECT count(*) FROM audit_events WHERE previous_hash IS NULL") == 1
    finally:
        for pool in pools:
            await pool.close()


async def test_mvp_345_edited_row_is_detected(pg_migrated, su):
    keyring = AuditKeyring.from_keys(KEY_A)
    await _write(AuditWriter("svc-0", pg_migrated.app_pool, keyring), 20, "actor")
    await _tamper(su, "UPDATE audit_events SET actor_id = 'someone-else' WHERE chain_seq = 10")
    result = await verify_chain_on(pg_migrated.app_pool, keyring)
    assert (result.ok, result.first_bad_seq, result.reason) == (False, 10, "hash_mismatch")


async def test_mvp_345_deleted_row_with_relinked_successor_is_detected(pg_migrated, su):
    keyring = AuditKeyring.from_keys(KEY_A)
    await _write(AuditWriter("svc-0", pg_migrated.app_pool, keyring), 20, "actor")
    previous = await su.fetchval("SELECT event_hash FROM audit_events WHERE chain_seq = 9")
    await _tamper(su, "DELETE FROM audit_events WHERE chain_seq = 10")
    await _tamper(su, "UPDATE audit_events SET previous_hash = $1 WHERE chain_seq = 11", previous)
    result = await verify_chain_on(pg_migrated.app_pool, keyring)
    assert (result.ok, result.first_bad_seq, result.reason) == (False, 11, "hash_mismatch")


async def test_mvp_345_details_edit_is_detected(pg_migrated, su):
    keyring = AuditKeyring.from_keys(KEY_A)
    await _write(AuditWriter("svc-0", pg_migrated.app_pool, keyring), 10, "actor")
    await _tamper(su, "UPDATE audit_events SET details = '{\"route\": \"/other\"}'::jsonb WHERE chain_seq = 5")
    result = await verify_chain_on(pg_migrated.app_pool, keyring)
    assert (result.ok, result.first_bad_seq, result.reason) == (False, 5, "details_mismatch")


async def test_mvp_351_rotated_hmac_key_still_verifies_old_rows(pg_migrated, su):
    pool = pg_migrated.app_pool
    await _write(AuditWriter("svc-0", pool, AuditKeyring.from_keys(KEY_A)), 5, "a")
    rotated = AuditKeyring.from_keys(KEY_B, retired=[KEY_A])
    await _write(AuditWriter("svc-0", pool, rotated), 5, "b")
    assert (await verify_chain_on(pool, rotated)).ok is True
    assert await su.fetchval("SELECT count(DISTINCT hmac_key_id) FROM audit_events") == 2
    without_old = await verify_chain_on(pool, AuditKeyring.from_keys(KEY_B))
    assert (without_old.ok, without_old.first_bad_seq, without_old.reason) == (False, 1, "unknown_key")


async def test_mvp_091_configured_audit_writes_and_verifies(pg_migrated, su, monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", KEY_A)
    monkeypatch.setenv("AUDIT_HMAC_RETIRED_KEYS_JSON", "[]")
    from solace_security.audit_runtime import audit, configure_service_audit, reset_audit_runtime, verify_chain

    reset_audit_runtime()
    await configure_service_audit("user-service", pg_migrated.app_pool)
    await audit(AuditEventType.AUTH_LOGIN_SUCCEEDED, actor_id="u-1", actor_type="user", action="login",
                outcome=AuditOutcome.SUCCESS, resource_type="session",
                subject_user_id="11111111-2222-4333-8444-555555555555", details={"amr": "pwd"})
    row = await su.fetchrow("SELECT writer_service, event_type, subject_user_id FROM audit_events")
    assert (row["writer_service"], row["event_type"]) == ("user-service", "auth_login_succeeded")
    assert str(row["subject_user_id"]) == "11111111-2222-4333-8444-555555555555"
    assert (await verify_chain()).ok is True
    reset_audit_runtime()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_091_audit_runtime.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/phi-codec-audit-llm/test_mvp_346_audit_chain_concurrency.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL in both files with `ModuleNotFoundError: No module named 'solace_security.audit_runtime'`: no service-callable audit writer exists (MVP-091), the only chain is per-process with link-only verification (MVP-345/346) and no HMAC key id is recorded (MVP-351).
- [ ] **Step 3: Implement** `src/solace_security/audit_runtime.py`. `AuditWriter.write` runs one transaction on `pool.acquire()` (audit_events is not an RLS table): `SELECT pg_advisory_xact_lock(hashtext('solace_audit_chain'))`, `SELECT nextval('audit_events_chain_seq_seq')`, `SELECT event_hash FROM audit_events ORDER BY chain_seq DESC LIMIT 1`, then the INSERT with an explicit `chain_seq` and `details` bound as `$n::text::jsonb` from `json.dumps(sanitized, sort_keys=True, separators=(",", ":"))` so the pool's JSON codec never double-encodes. `details_sha256` is the sha256 of that canonical string; the HMAC input is `json.dumps(fields, sort_keys=True, separators=(",", ":"))` of the 16 fields named in `test_mvp_345_event_hash_covers_every_field` plus `previous_hash`, with `occurred_at` rendered by `.astimezone(UTC).isoformat(timespec="microseconds")`. `verify_chain_on` selects `details::text` in `chain_seq` order (fetching the predecessor row when `start_seq` is given) and checks, per row and in this order: details hash, key id known, HMAC, link to the predecessor (the first row of the table must have a NULL `previous_hash`). `ALLOWED_DETAIL_KEYS = frozenset({"route", "method", "status_code", "reason_code", "count", "stores", "service", "channel", "template_id", "level", "categories", "attempt", "mfa", "amr", "erasure_request_id", "alert_id", "kid", "role", "on_call", "backup", "consent_type", "consent_version", "granted", "via_service", "sid"})`; values of type `bool` or `int` are kept, `str` values are kept only when they match `^[A-Za-z0-9_.:/{}=,\-]{0,128}$` and are otherwise replaced by `"[REDACTED]"`, other types and other keys are dropped (WARNING `audit_details_key_dropped` with key names only).
- [ ] **Step 4: Run them and confirm they pass**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/phi-codec-audit-llm -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.12: audit_phi_access dependency and the chain verifier task

**Covers:** MVP-091
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-09.11, WS-07.2, WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: a nightly in-process verifier in user-service sets `solace_audit_chain_verify_ok`; WS-04 alerts on it)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/audit_runtime.py` (new `audit_phi_access(resource_type: str, *, subject: Callable[[Request], str | None] | None = None) -> Callable` and `async def audit_chain_verifier(*, interval_seconds: float = 86400.0, stop_event: asyncio.Event | None = None) -> None`)
- Test: `tests/solace_security/test_mvp_091_audit_phi_access.py` (create)

**Interfaces:**
- Consumes: `solace_security.middleware.get_current_user` and `AuthenticatedUser` (IF-04; the `via_service` field from WS-07.2 and obo-token acceptance from WS-07.4, falling back to `metadata["via_service"]`); `audit()` and `verify_chain()` from WS-09.11.
- Produces (IF-14): `audit_phi_access(...)` returns an async-generator FastAPI dependency `async def _dependency(request: Request, principal: AuthenticatedUser = Depends(get_current_user))` that yields, then records exactly one audit call: event type `PHI_READ` for GET/HEAD, `PHI_DELETE` for DELETE, `PHI_WRITE` otherwise; `actor_type="service"` and `actor_id=<via_service>` for on-behalf-of tokens, else `actor_type="user"` and `actor_id=principal.user_id`; `subject_user_id = subject(request) if subject else principal.user_id`; `action=f"{method} {route template}"`; `resource_type` as given; `details={"method", "route", "status_code"}` (+ `"via_service"` for on-behalf-of tokens); outcome `SUCCESS` below 400, `DENIED` for 401/403/404, `FAILURE` otherwise (status from the raised `HTTPException`, 500 for other exceptions, else the route's declared status code or 200); never request or response bodies. `audit_chain_verifier` loops until `stop_event` is set: `verify_chain()`, then sets `solace_audit_chain_verify_ok` to 1 or 0 and `solace_audit_chain_tip_seq` to the tip, logs CRITICAL `audit_chain_verify_failed` with `first_bad_seq` and `reason` on failure, then sleeps `interval_seconds`. WS-04 registers it as a user-service background task (CR-8); service lanes add `Depends(audit_phi_access(...))` to PHI routes (IF-14 consumers).

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_091_audit_phi_access.py`
```python
"""MVP-091: PHI access is audited per request (actor, subject, route template, status) and the chain is verified."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import solace_security.audit_runtime as ar
from solace_security.audit import AuditEventType, AuditOutcome
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user

PATIENT = "11111111-2222-4333-8444-555555555555"


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    async def fake_audit(event_type, **kwargs):
        calls.append((event_type, kwargs))

    monkeypatch.setattr(ar, "audit", fake_audit)
    return calls


def _app(principal):
    app = FastAPI()
    phi = ar.audit_phi_access("therapy_notes", subject=lambda request: request.path_params["user_id"])

    @app.get("/patients/{user_id}/notes")
    async def read_notes(user_id: str, _audit: None = Depends(phi)):
        return {"notes": "PHI BODY"}

    @app.delete("/patients/{user_id}/notes")
    async def delete_notes(user_id: str, _audit: None = Depends(phi)):
        raise HTTPException(status_code=403, detail="not assigned")

    app.dependency_overrides[get_current_user] = lambda: principal
    return app


def test_mvp_091_phi_read_is_audited_with_route_template(recorded):
    principal = AuthenticatedUser(user_id="clin-1", token_type=TokenType.ACCESS, roles=["clinician"])
    response = TestClient(_app(principal)).get(f"/patients/{PATIENT}/notes")
    assert response.status_code == 200
    [(event_type, kwargs)] = recorded
    assert event_type == AuditEventType.PHI_READ
    assert (kwargs["actor_id"], kwargs["actor_type"], kwargs["subject_user_id"]) == ("clin-1", "user", PATIENT)
    assert (kwargs["outcome"], kwargs["resource_type"]) == (AuditOutcome.SUCCESS, "therapy_notes")
    assert kwargs["details"] == {"method": "GET", "route": "/patients/{user_id}/notes", "status_code": 200}
    assert "PHI BODY" not in repr(kwargs)


def test_mvp_091_denied_access_is_audited_as_denied(recorded):
    principal = AuthenticatedUser(user_id="clin-2", token_type=TokenType.ACCESS, roles=["clinician"])
    response = TestClient(_app(principal)).delete(f"/patients/{PATIENT}/notes")
    assert response.status_code == 403
    [(event_type, kwargs)] = recorded
    assert (event_type, kwargs["outcome"], kwargs["details"]["status_code"]) == (
        AuditEventType.PHI_DELETE, AuditOutcome.DENIED, 403,
    )


def test_mvp_091_on_behalf_of_service_is_the_actor(recorded):
    principal = AuthenticatedUser(user_id=PATIENT, token_type=TokenType.SERVICE, roles=["user"],
                                  metadata={"via_service": "orchestrator-service"})
    TestClient(_app(principal)).get(f"/patients/{PATIENT}/notes")
    [(_, kwargs)] = recorded
    assert (kwargs["actor_type"], kwargs["actor_id"]) == ("service", "orchestrator-service")
    assert kwargs["details"]["via_service"] == "orchestrator-service"


async def test_mvp_091_chain_verifier_sets_gauges(monkeypatch):
    bad = ar.ChainVerification(ok=False, checked=3, first_bad_seq=2, reason="hash_mismatch", tip_seq=3, tip_hash="a" * 64)

    async def fake_verify(start_seq=None, end_seq=None):
        return bad

    monkeypatch.setattr(ar, "verify_chain", fake_verify)
    stop = asyncio.Event()
    task = asyncio.create_task(ar.audit_chain_verifier(interval_seconds=0.01, stop_event=stop))
    await asyncio.sleep(0.05)
    stop.set()
    await task
    assert REGISTRY.get_sample_value("solace_audit_chain_verify_ok") == 0.0
    assert REGISTRY.get_sample_value("solace_audit_chain_tip_seq") == 3.0
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_091_audit_phi_access.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `AttributeError: module 'solace_security.audit_runtime' has no attribute 'audit_phi_access'` (no route can record PHI access; the register scenario of an unattributable clinician read).
- [ ] **Step 3: Implement** both functions. The dependency resolves the route template from `request.scope["route"].path` (falling back to `request.url.path`), catches `HTTPException` and other exceptions around `yield` only to record the status and re-raises them unchanged, and calls `audit` through the module global so the test double applies.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.13: Log sanitizer redacts free text, handles nested and non-string values and never raises

**Covers:** MVP-093, MVP-315, MVP-316
**Severity:** high
**Wave:** 3
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_security/phi_protection.py` (new `class LogEventSanitizer` with module constants `SAFE_KEYS`, `ALWAYS_REDACT_KEYS`, `FREE_TEXT_KEYS`, `ERROR_KEYS`, `TRACEBACK_KEYS`, `SENSITIVE_KEY_TOKENS`, `PROSE_MIN_WORDS = 4`, `MAX_DEPTH = 8`; `phi_sanitizer_processor` delegates to it inside a never-raise wrapper; `PHISanitizer.sanitize_dict` coerces keys with `str(key)` and `PHISanitizer._is_sensitive_key` uses token matching)
- Test: `tests/solace_security/test_mvp_093_log_sanitizer.py` (create)

**Interfaces:**
- Consumes: none.
- Produces: `phi_sanitizer_processor(logger_, method_name, event_dict) -> dict` (name unchanged; WS-04's `configure_service_logging` keeps calling it after `format_exc_info`). Policy, applied per key with `k = str(key).lower()` and tokens `re.split(r"[^a-z0-9]+", k)`: `SAFE_KEYS = {"event", "level", "log_level", "timestamp", "logger", "logger_name", "service", "service_name", "method", "route", "path", "status", "status_code", "request_id", "correlation_id", "turn_id", "trace_id", "span_id", "duration_ms", "hint", "func_name", "lineno", "module", "event_type", "template", "template_id", "topic", "group_id", "consumer_group", "pool_name", "environment", "version", "reason_code", "error_type", "error_code", "violations", "missing", "fields", "checks"}` get PHI-pattern masking only; `ALWAYS_REDACT_KEYS = {"display_name", "name", "full_name", "first_name", "last_name", "bio", "password", "token", "access_token", "refresh_token", "authorization", "api_key", "secret"}` become `"[REDACTED]"`; `FREE_TEXT_KEYS = {"message", "msg", "content", "text", "body", "payload", "prompt", "response", "completion", "transcript", "query", "notes", "summary", "reason", "description", "input", "output", "user_message", "event_data", "raw", "data", "context", "history", "messages", "detail", "details", "args", "kwargs", "trigger_indicators", "evidence", "risk_factors", "symptoms", "diagnosis", "escalation_reason"}` keep only code-like strings (`^[a-z0-9_.:\-]{1,64}$`) and otherwise become `f"[REDACTED:free_text len={len}]"` (strings) or `f"[REDACTED:{type} items={n}]"` (containers); `ERROR_KEYS = {"error", "error_message", "err", "exc"}` go through `redact_error_text` (replace `DETAIL:` and everything after it with `DETAIL: [REDACTED]`, quoted substrings with `[REDACTED]`, `input_value=...` up to the next `,` or `]` with `input_value=[REDACTED]`, then PHI masking, then truncate to 300 characters); `TRACEBACK_KEYS = {"exception", "exc_info", "stack", "stack_info"}` keep `Traceback (most recent call last):`, chaining sentences, `File "...", line N, in f` lines and the indented source line after each, turn `Name: message` lines into `Name: [REDACTED]` and every other line into `[REDACTED]`; a key is sensitive when any token is in `SENSITIVE_KEY_TOKENS = {"ssn", "social", "phone", "email", "address", "dob", "birth", "birthdate", "mrn"}` (PHI masking, else first/last-character masking as today); any other string is PHI-masked and then becomes `f"[REDACTED:free_text len={len}]"` when it has at least `PROSE_MIN_WORDS` whitespace-separated words. Value types: `None`, `bool`, `int`, `float`, `Decimal`, `datetime`/`date`/`time`, `UUID`, `Enum` unchanged; mappings recurse (deeper than `MAX_DEPTH` → `"[REDACTED:depth]"`); `list`/`tuple`/`set`/`frozenset` become lists of sanitized items; `BaseException` → `f"{type(value).__name__}: [REDACTED]"`; `bytes`-like → `f"[REDACTED:bytes len={n}]"`; pydantic models → their `model_dump(mode="json")`; anything else → the string rule on `str(value)`, or `"[REDACTED:unrenderable]"` if `str()` raises. Any internal error returns `{"event": <event if it is a str else "log_event">, "sanitizer_error": <exception type name>}`. The detector used by the processor has `log_detections=False`, so sanitizing never logs.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_093_log_sanitizer.py`
```python
"""MVP-093 / MVP-315 / MVP-316: the structlog PHI processor redacts free text, handles any value and never raises."""
from __future__ import annotations

import uuid

import structlog

from solace_security.phi_protection import PHISanitizer, phi_sanitizer_processor


def _p(**event):
    return phi_sanitizer_processor(None, "info", {"event": "evt", **event})


def test_mvp_093_free_text_values_are_redacted():
    out = _p(message="I'm going to take all my pills tonight", content="I want to end it all")
    assert "pills" not in str(out) and "end it all" not in str(out)
    assert out["message"].startswith("[REDACTED")


def test_mvp_093_nested_event_payload_is_redacted():
    out = _p(event_data={"trigger_indicators": ["I'm going to take all my pills tonight"],
                         "metadata": {"source_service": "diagnosis-service"}})
    assert "pills" not in str(out)


def test_mvp_093_prose_under_an_unknown_key_is_redacted():
    out = _p(note_for_dev="the patient said she wants to die soon")
    assert "wants to die" not in str(out)


def test_mvp_093_db_error_detail_redacted_but_class_kept():
    error = ('duplicate key value violates unique constraint "users_email_key" '
             "DETAIL: Key (display_name)=(Jane Roe) already exists.")
    out = _p(error=error)
    assert "Jane Roe" not in out["error"]
    assert "duplicate key value violates unique constraint" in out["error"]


def test_mvp_093_exception_objects_reduced_to_type():
    assert _p(exc_obj=ValueError("I want to die"))["exc_obj"] == "ValueError: [REDACTED]"


def test_mvp_093_traceback_messages_redacted_frames_kept():
    traceback = ('Traceback (most recent call last):\n  File "/app/src/x.py", line 10, in handler\n'
                 "    raise ValueError(msg)\nValueError: I want to die tonight")
    out = _p(exception=traceback)["exception"]
    assert 'File "/app/src/x.py", line 10, in handler' in out
    assert "die tonight" not in out and "ValueError: [REDACTED]" in out


def test_mvp_093_tuple_values_are_sanitized():
    assert "life" not in str(_p(items=("I want to end my life now",))["items"])


def test_mvp_093_identifiers_and_codes_untouched():
    out = _p(user_id="u-123", status_code=403, route="/api/v1/users/{user_id}", reason="rehydrate_failed",
             hint="Install portkey-ai: pip install portkey-ai")
    assert out == {"event": "evt", "user_id": "u-123", "status_code": 403, "route": "/api/v1/users/{user_id}",
                   "reason": "rehydrate_failed", "hint": "Install portkey-ai: pip install portkey-ai"}


def test_mvp_315_non_string_keys_do_not_raise_and_render():
    out = _p(by_level={1: "a"}, keyed={uuid.UUID(int=7): "s"}, rows=[{2: "z"}])
    assert out["by_level"] == {"1": "a"}
    assert out["rows"] == [{"2": "z"}]
    structlog.processors.JSONRenderer()(None, "info", out)


def test_mvp_315_processor_never_raises():
    class Boom:
        def __str__(self):
            raise RuntimeError("cannot render")

    assert _p(weird=Boom())["weird"] == "[REDACTED:unrenderable]"


def test_mvp_316_substring_keys_are_not_masked():
    out = _p(health_status="unhealthy", account_id="acc-9f", card_layout="grid")
    assert out == {"event": "evt", "health_status": "unhealthy", "account_id": "acc-9f", "card_layout": "grid"}
    assert PHISanitizer().sanitize_dict({"health_status": "unhealthy"}) == {"health_status": "unhealthy"}


def test_mvp_316_contact_keys_are_still_masked():
    out = _p(clinician_email="dr.smith@clinic.org", ip_address="10.1.2.3", fallback_email="x@y.org")
    assert "dr.smith" not in out["clinician_email"]
    assert "10.1.2.3" not in out["ip_address"]
    assert "x@y.org" not in out["fallback_email"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_093_log_sanitizer.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The free-text, nested, prose, DB-detail, exception, traceback and tuple tests fail because the text comes back unchanged (only structured identifiers are matched, phi_protection.py:99-174, and non-str values are copied verbatim, :338-358); `test_mvp_315_non_string_keys_do_not_raise_and_render` raises `AttributeError: 'int' object has no attribute 'lower'` (:340); `test_mvp_316_substring_keys_are_not_masked` fails because `health_status` becomes `u*******y` (:382-397).
- [ ] **Step 3: Implement** `LogEventSanitizer.sanitize_event(event_dict: Mapping[Any, Any]) -> dict[str, Any]` exactly per the produced policy and route `phi_sanitizer_processor` through it. Keep `PHISanitizer`, `PHIDetector`, `detect_phi` and `mask_phi` public behaviour unchanged apart from the key coercion and token matching, so `tests/solace_security/test_phi_protection.py` and `test_security_suite.py` stay green.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.14: UUIDs survive the sanitizer

**Covers:** MVP-189
**Severity:** medium
**Wave:** 3
**Depends on:** WS-09.13
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_security/phi_protection.py` (`_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")`; `PHIDetector.detect` drops matches that overlap a UUID span; `LICENSE_NUMBER` and `HEALTH_PLAN` patterns compile without `re.IGNORECASE`; `LogEventSanitizer` returns a value that is exactly one UUID unchanged before any pattern runs)
- Test: `tests/solace_security/test_mvp_189_uuid_masking.py` (create)

**Interfaces:**
- Consumes: `LogEventSanitizer` (WS-09.13).
- Produces: UUIDs (alone or inside text) are never altered by `mask_phi` or `phi_sanitizer_processor`; licence and health-plan detection becomes case-sensitive (upper-case letters only), recorded under Open risks.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_189_uuid_masking.py`
```python
"""MVP-189 (absorbs MVP-313): UUIDs are never partially masked by the PHI patterns."""
from __future__ import annotations

import random
import uuid

from solace_security.phi_protection import mask_phi, phi_sanitizer_processor


def _uuids(n):
    rng = random.Random(189)
    return [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(n)]


def test_mvp_189_20000_uuids_survive_the_log_processor():
    altered = [u for u in _uuids(20000)
               if phi_sanitizer_processor(None, "info", {"event": "e", "user_id": u})["user_id"] != u]
    assert altered == []


def test_mvp_189_uuids_inside_text_survive_mask_phi():
    assert [u for u in _uuids(5000) if u not in mask_phi(f"escalation {u} created")] == []


def test_mvp_189_register_examples():
    for value in ("b1504008-d310-4e84-bacb-e41e216706e9", "07296153-9086-4418-a5c3-1f2e3d4c5b6a"):
        assert phi_sanitizer_processor(None, "info", {"event": "e", "escalation_id": value})["escalation_id"] == value


def test_mvp_189_real_licence_and_plan_numbers_still_masked():
    assert "D1234567" not in mask_phi("licence D1234567 on file")
    assert "ABC123456789" not in mask_phi("plan ABC123456789")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_189_uuid_masking.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The 20,000-UUID test lists about 320 altered values (case-insensitive licence pattern `\b[A-Z]\d{7,8}\b` plus the card and health-plan patterns, phi_protection.py:131-173, 241-243); `b1504008-...` comes back as `********-d310-...`.
- [ ] **Step 3: Implement** the UUID span exclusion inside `PHIDetector.detect` (computed once per string) and the per-pattern flags (`re.IGNORECASE` for every pattern except `LICENSE_NUMBER` and `HEALTH_PLAN`).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.15: MVP-055 closure: codec adoption lock

**Covers:** MVP-055
**Severity:** high
**Wave:** 3 (phase B: a test-only barrier lock that runs after the W3 phase-B adoption task WS-12.12; it edits no platform or service path)
**Depends on:** WS-09.8, WS-04.8, WS-10.10, WS-12.12, WS-14.10, WS-15.8, WS-16.8, WS-16.9, WS-17.5, WS-18.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Test: `tests/solace_security/test_mvp_055_codec_adoption.py` (create; no production file changes)

**Interfaces:**
- Consumes: the adoption of IF-13 by every column owner (CR-10): WS-10.10 (checkpointer), WS-14.10 (user PII), WS-15.8 (therapy repositories), WS-16.8 and WS-16.9 (diagnosis records and Redis blobs), WS-17.5 (memory writes), WS-18.4 (personality profiles), all W2 phase B, and WS-12.12 (escalation records, W3 phase B, which also removes safety's `configure_phi_encryption` call); WS-04.8's codec warm-up in `src/solace_infrastructure/runtime/bootstrap.py` (CR-8).
- Produces: a permanent lock that every IF-13 adopter imports `solace_security.phi_codec` and that no service code uses the retired ORM-listener path. The ciphertext-at-rest proof is the W2 exit criterion (per-service realinfra repository suites reading raw SQL); this lock keeps it from regressing.

- [ ] **Step 1: Write the acceptance test** in `tests/solace_security/test_mvp_055_codec_adoption.py`
```python
"""MVP-055 closure: every PHI repository uses the IF-13 codec; nothing uses the ORM-only listeners."""
from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
ADOPTERS = {
    "services/user-service/src/infrastructure/postgres_repository.py": "WS-14",
    "services/safety_service/src/domain/escalation.py": "WS-12",
    "services/diagnosis_service/src/infrastructure/postgres_repository.py": "WS-16",
    "services/diagnosis_service/src/infrastructure/redis_session_repository.py": "WS-16",
    "services/therapy_service/src/infrastructure/postgres_repository.py": "WS-15",
    "services/memory_service/src/infrastructure/postgres_repo.py": "WS-17",
    "services/personality_service/src/infrastructure/postgres_repository.py": "WS-18",
    "services/orchestrator_service/src/langgraph/checkpointer.py": "WS-10",
}
LEGACY = {"configure_phi_encryption", "get_phi_encryptor", "FieldEncryptor", "create_field_encryptor"}


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))


def test_mvp_055_every_phi_repository_imports_the_codec():
    missing = []
    for relative, owner in ADOPTERS.items():
        path = REPO / relative
        modules = {node.module for node in ast.walk(_tree(path)) if isinstance(node, ast.ImportFrom)} if path.is_file() else set()
        if "solace_security.phi_codec" not in modules:
            missing.append(f"{relative} ({owner})")
    assert missing == []


def test_mvp_055_no_service_uses_the_retired_orm_listener_path():
    offenders = []
    for path in (REPO / "services").rglob("*.py"):
        if {"tests", "__pycache__", "config_service"} & set(path.parts):
            continue
        tree = _tree(path)
        names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        names |= {getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                  for node in ast.walk(tree) if isinstance(node, ast.Call)}
        if names & LEGACY:
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


def test_mvp_055_bootstrap_warms_the_codec():
    bootstrap = REPO / "src" / "solace_infrastructure" / "runtime" / "bootstrap.py"
    assert "get_phi_codec" in bootstrap.read_text(encoding="utf-8")
```
- [ ] **Step 2: Run it against the pre-adoption tree and confirm it fails**
Run: `$t = (Select-String -Path D:\tmp\mvp-exec\ledger\progress.md -Pattern 'Wave 2 start tree=([0-9a-f]+)').Matches[0].Groups[1].Value; git -C D:\Repo\Contextual-Chatbot archive --format=zip -o D:\tmp\mvp-exec\snapshots\ws09-15-red.zip $t; Expand-Archive -Force D:\tmp\mvp-exec\snapshots\ws09-15-red.zip D:\tmp\mvp-exec\snapshots\ws09-15-red; Copy-Item tests\solace_security\test_mvp_055_codec_adoption.py D:\tmp\mvp-exec\snapshots\ws09-15-red\tests\solace_security\; $env:PYTHONPATH="D:\tmp\mvp-exec\snapshots\ws09-15-red;D:\tmp\mvp-exec\snapshots\ws09-15-red\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\tmp\mvp-exec\snapshots\ws09-15-red\tests\solace_security\test_mvp_055_codec_adoption.py -o "addopts=" -p no:logging -q` (the ledger line `Wave 2 start tree=<id>` is written by the controller at the W2 start; if it is named differently, use the W2 wave-start tree id from the ledger).
Expected: FAIL on the W2-start tree: `test_mvp_055_every_phi_repository_imports_the_codec` lists all eight adopters (`solace_security.phi_codec` does not exist before WS-09.7); `test_mvp_055_no_service_uses_the_retired_orm_listener_path` lists at least `services/diagnosis_service/src/infrastructure/postgres_repository.py` (it calls `get_phi_encryptor` at postgres_repository.py:207-208, the misleading `encryption_key_id='primary'` stamp in MVP-055) and `services/safety_service/src/main.py` (it imports `FieldEncryptor` and calls `configure_phi_encryption` at main.py:116-121 until WS-12.12 removes the block in W3); and the bootstrap test fails because `get_phi_codec` arrives with WS-04.8.
- [ ] **Step 3: Implement** nothing in production code. If Step 4 fails, each listed path is reported to its owner as a W3 fix task `<owner>.F<k>` in the ledger (the ADOPTERS map is the contract agreed in CR-10 and may be amended only by a ledger Ruling).
- [ ] **Step 4: Run it on the working tree and confirm it passes**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_055_codec_adoption.py -o "addopts=" -p no:logging -q`. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.15:done"` and record the printed tree id in the ledger. No git commit. Mark MVP-055 fixed only after the W2 exit criterion "PHI columns are ciphertext in raw SQL" is recorded green for every lane.

### Task WS-09.16: Audit chain verify CLI for the E2E suite and the nightly job

**Covers:** MVP-091
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-09.11, WS-09.12, WS-02.6, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: the chain is verified nightly and on demand; the nightly result reaches Prometheus through the node-exporter textfile `/textfile/solace_audit.prom` on volume `solace_textfile_metrics`, as WS-04.12 expects)
**Real infra:** yes
**Files:**
- Modify: `src/solace_security/audit_runtime.py` (`def main(argv: Sequence[str] | None = None, *, pool_factory: Callable[[], Awaitable[asyncpg.Pool]] | None = None) -> int`; private `async def _run_verify(args, pool_factory) -> int`; an `if __name__ == "__main__":` block as the last statement)
- Test: `tests/solace_security/test_mvp_091_audit_verify_cli.py` (create)
- Test: `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_verify_cli_pg.py` (create)

**Interfaces:**
- Consumes: `configure_service_audit`, `verify_chain`, `verify_chain_on`, `ChainVerification`, `AuditKeyring`, `AuditWriter`, `AuditConfigError` (WS-09.11); WS-09.12 only as the previous edit of the same module; `ConnectionPoolManager.get_pool("default")` and `ConnectionPoolManager.close_all_pools()` from `solace_infrastructure.database.connection_manager` (IF-03, WS-02.6: built from `POSTGRES_*`, verified TLS in strict); `AUDIT_HMAC_KEY` and `AUDIT_HMAC_RETIRED_KEYS_JSON`.
- Produces (IF-14 "scripts run nightly"): `python -m solace_security.audit_runtime verify [--start-seq N] [--textfile PATH]`, requested by WS-05 (request 7) for WS-05.15's `audit_verify(stack)` and by WS-04 for WS-04.12. stdout is exactly one line, `json.dumps({"ok": <bool>, "first_bad_seq": <int or null>})`. Exit 0 when the chain verifies; exit 1 for every other outcome: a broken chain, an unknown key id, a configuration error (missing or short `AUDIT_HMAC_KEY`, invalid retired JSON) or a database error, the last two printing `{"ok": false, "first_bad_seq": null}`; exit 2 only for an invalid command line (argparse). One diagnostic JSON line goes to stderr (`event="audit_chain_verify_cli"`, `reason`, `checked`, `tip_seq`, `error` as an exception type name), never key material or row content. `--textfile PATH` writes `solace_audit_chain_verify_ok` (1 or 0) and `solace_audit_chain_tip_seq` (tip, or 0 when unknown) from a private `CollectorRegistry` with `prometheus_client.write_to_textfile` (atomic rename), on failure as well. The CLI configures audit as writer `audit-verify-cli` and never writes a row. Scheduling the nightly run is CR-13.

- [ ] **Step 1: Write the failing tests** in `tests/solace_security/test_mvp_091_audit_verify_cli.py` and `tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_verify_cli_pg.py`
```python
# tests/solace_security/test_mvp_091_audit_verify_cli.py
"""MVP-091: the verify CLI reports the chain state with a stable stdout line and exit code."""
from __future__ import annotations

import json

import pytest

import solace_security.audit_runtime as ar

KEY = "audit-key-A-" + "a" * 32


async def _pool_factory():
    return object()  # verify_chain_on is replaced below, so no query runs


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", KEY)
    monkeypatch.setenv("AUDIT_HMAC_RETIRED_KEYS_JSON", "[]")
    ar.reset_audit_runtime()
    yield
    ar.reset_audit_runtime()


def _result(ok, first_bad_seq=None):
    return ar.ChainVerification(ok=ok, checked=7, first_bad_seq=first_bad_seq,
                                reason="ok" if ok else "hash_mismatch", tip_seq=7, tip_hash="a" * 64)


def _stub(monkeypatch, result, seen):
    async def fake(pool, keyring, *, start_seq=None, end_seq=None, batch_size=1000):
        seen.append(start_seq)
        return result

    monkeypatch.setattr(ar, "verify_chain_on", fake)


def _last_stdout_line(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_mvp_091_cli_intact_chain_exits_zero(monkeypatch, capsys):
    seen = []
    _stub(monkeypatch, _result(True), seen)
    assert ar.main(["verify"], pool_factory=_pool_factory) == 0
    assert _last_stdout_line(capsys) == {"ok": True, "first_bad_seq": None}
    assert seen == [None]


def test_mvp_091_cli_broken_chain_exits_one_and_names_the_row(monkeypatch, capsys):
    seen = []
    _stub(monkeypatch, _result(False, first_bad_seq=5), seen)
    assert ar.main(["verify", "--start-seq", "3"], pool_factory=_pool_factory) == 1
    assert _last_stdout_line(capsys) == {"ok": False, "first_bad_seq": 5}
    assert seen == [3]


def test_mvp_091_cli_config_error_exits_one_without_echoing_the_key(monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_HMAC_KEY", "short-key")
    assert ar.main(["verify"], pool_factory=_pool_factory) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip().splitlines()[-1]) == {"ok": False, "first_bad_seq": None}
    assert "short-key" not in captured.out + captured.err


def test_mvp_091_cli_database_error_exits_one(capsys):
    async def broken():
        raise ConnectionError("db down")

    assert ar.main(["verify"], pool_factory=broken) == 1
    assert _last_stdout_line(capsys) == {"ok": False, "first_bad_seq": None}


@pytest.mark.parametrize(("ok", "gauge"), [(True, "1.0"), (False, "0.0")])
def test_mvp_091_cli_textfile_holds_both_gauges(monkeypatch, tmp_path, ok, gauge):
    _stub(monkeypatch, _result(ok, first_bad_seq=None if ok else 2), [])
    target = tmp_path / "solace_audit.prom"
    ar.main(["verify", "--textfile", str(target)], pool_factory=_pool_factory)
    text = target.read_text(encoding="utf-8")
    assert f"solace_audit_chain_verify_ok {gauge}" in text
    assert "solace_audit_chain_tip_seq 7.0" in text


def test_mvp_091_cli_rejects_an_invalid_command_line():
    with pytest.raises(SystemExit) as exc:
        ar.main(["verify", "--start-seq", "x"], pool_factory=_pool_factory)
    assert exc.value.code == 2
```
```python
# tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_verify_cli_pg.py
"""MVP-091: `python -m solace_security.audit_runtime verify` as a real subprocess against slot 011."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import sys
from urllib.parse import urlsplit

import asyncpg
import pytest

from solace_security.audit import AuditEventType, AuditOutcome
from solace_security.audit_runtime import AuditKeyring, AuditWriter

pytestmark = [pytest.mark.realinfra, pytest.mark.asyncio(loop_scope="session")]
REPO = pathlib.Path(__file__).resolve().parents[3]
KEY = "audit-key-A-" + "a" * 32
OK = json.dumps({"ok": True, "first_bad_seq": None})


@pytest.fixture
async def su(pg_migrated):
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    await conn.execute("ALTER TABLE audit_events DISABLE TRIGGER audit_events_no_truncate;"
                       " TRUNCATE audit_events RESTART IDENTITY;"
                       " ALTER TABLE audit_events ENABLE TRIGGER audit_events_no_truncate")
    yield conn
    await conn.close()


async def _cli(pg_migrated, cwd, *args):
    dsn = urlsplit(pg_migrated.dsn_app)
    env = dict(os.environ, ENVIRONMENT="test", AUDIT_HMAC_KEY=KEY, AUDIT_HMAC_RETIRED_KEYS_JSON="[]",
               POSTGRES_HOST=dsn.hostname, POSTGRES_PORT=str(dsn.port), POSTGRES_USER=dsn.username,
               POSTGRES_PASSWORD=dsn.password, POSTGRES_DATABASE=dsn.path.lstrip("/"),
               POSTGRES_SSL_MODE="disable", POSTGRES_MIN_POOL_SIZE="1", POSTGRES_MAX_POOL_SIZE="2",
               PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "src")]))
    done = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-m", "solace_security.audit_runtime", "verify", *args],
        env=env, cwd=str(cwd), capture_output=True, text=True, timeout=120,
    )
    return done.returncode, done.stdout.strip().splitlines()


async def test_mvp_091_cli_verifies_detects_tamper_and_resumes(pg_migrated, su, tmp_path):
    writer = AuditWriter("svc-0", pg_migrated.app_pool, AuditKeyring.from_keys(KEY))
    for i in range(10):
        await writer.write(AuditEventType.PHI_READ, actor_id=f"a-{i}", actor_type="user", action="GET /x",
                           outcome=AuditOutcome.SUCCESS, resource_type="t")
    assert await _cli(pg_migrated, tmp_path) == (0, [OK])
    await su.execute("DROP RULE audit_events_no_update ON audit_events")
    try:
        await su.execute("UPDATE audit_events SET actor_id = 'someone-else' WHERE chain_seq = 5")
    finally:
        await su.execute("CREATE RULE audit_events_no_update AS ON UPDATE TO audit_events DO INSTEAD NOTHING")
    assert await _cli(pg_migrated, tmp_path) == (1, [json.dumps({"ok": False, "first_bad_seq": 5})])
    assert await _cli(pg_migrated, tmp_path, "--start-seq", "6") == (0, [OK])
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_091_audit_verify_cli.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/phi-codec-audit-llm/test_mvp_091_audit_verify_cli_pg.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. The unit tests fail with `AttributeError: module 'solace_security.audit_runtime' has no attribute 'main'`. The subprocess test fails on its first assertion with `(0, []) != (0, ['{"ok": true, "first_bad_seq": null}'])`: the module has no `__main__` entry, so running it prints nothing and exits 0, which a nightly job or the W3 privacy suite would read as success even for the tampered chain (nothing verifies the trail outside a test, MVP-091).
- [ ] **Step 3: Implement** `main()` with `argparse` (a required `verify` subcommand; `--start-seq` as `int`; `--textfile` as a path) and `asyncio.run(_run_verify(...))`. `_run_verify` takes the pool from `pool_factory`, or from `ConnectionPoolManager.get_pool("default")` through a function-local import (then `close_all_pools()` in `finally`), calls `configure_service_audit("audit-verify-cli", pool)` and then `verify_chain(start_seq=...)` through the module globals, catches `AuditConfigError` and any other `Exception` into the exit-1 result, writes the textfile when asked, prints the stderr diagnostic line, prints the stdout line last, and calls `reset_audit_runtime()` before returning. The `__main__` block routes structlog to stderr (`structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))`, so pool logs never reach stdout) and then runs `raise SystemExit(main())` on the same module object. Nothing reached from `main()` may import `solace_security.audit_runtime` by name: a second import would register the gauges twice (`Duplicated timeseries`), and the subprocess test would fail.
- [ ] **Step 4: Run them and confirm they pass**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/phi-codec-audit-llm -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.17: Salt-only crisis pseudonym for erased users' crisis records

**Covers:** none (supporting MVP-029, owned by WS-14, and MVP-089, owned by WS-12; requested by WS-13 as R-13-3 for WS-13.14)
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-09.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: crisis episodes and escalation records of an erased user are pseudonymized with a salted hash and kept for duty of care)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/phi_codec.py` (`def crisis_pseudonym(user_id: UUID | str) -> UUID`; private `_SearchSaltSettings(BaseSettings)` holding only `search_hash_salt: SecretStr = SecretStr("")` with the same `env_prefix="ENCRYPTION_", env_file=".env", extra="ignore"` as `EncryptionSettings`; the parsed salt is cached and `reset_phi_codec()` clears it)
- Test: `tests/solace_security/test_mvp_029_crisis_pseudonym.py` (create)

**Interfaces:**
- Consumes: `PHICodec.search_hash` (WS-09.7, for the equality test only); `is_strict_environment()` (IF-01).
- Produces (R-13-3 of WS-13): `crisis_pseudonym(user_id)` returns `UUID(hex=hmac_sha256(salt, "crisis_pseudonym" + "\x1f" + canonical).hexdigest()[:32])`, where `canonical = str(UUID(str(user_id)))` (a non-UUID input raises `ValueError`). For any UUID this equals WS-12.12's `UUID(hex=get_phi_codec().search_hash(str(user_id), purpose="crisis_pseudonym")[:32])`. It reads only `ENCRYPTION_SEARCH_HASH_SALT`, so it works in notification-service, which holds no `ENCRYPTION_MASTER_KEY`. Strict environments raise `PHIConfigError` (no salt in the message) when the salt is missing or shorter than 16 bytes; outside strict an empty salt is accepted, as in the codec. Consumers: WS-13.14; WS-12.12 may switch to it.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_029_crisis_pseudonym.py`
```python
"""Supports MVP-029 / MVP-089 (R-13-3): a salt-only crisis pseudonym equal to the codec's search hash."""
from __future__ import annotations

import uuid

import pytest

from solace_security.phi_codec import PHIConfigError, crisis_pseudonym, get_phi_codec, reset_phi_codec

USER = uuid.UUID("11111111-2222-4333-8444-555555555555")
SALT = "search-salt-for-tests-0123456789"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env file is read
    for name in ("ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "ENCRYPTION_KEY_ID", "ENCRYPTION_RETIRED_KEYS_JSON"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    reset_phi_codec()
    yield
    reset_phi_codec()


def test_mvp_029_pseudonym_equals_the_codec_search_hash(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "ab" * 32)
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", SALT)
    expected = uuid.UUID(hex=get_phi_codec().search_hash(str(USER), purpose="crisis_pseudonym")[:32])
    assert crisis_pseudonym(USER) == expected
    assert crisis_pseudonym(str(USER).upper()) == expected


def test_mvp_029_pseudonym_needs_only_the_salt(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", SALT)
    first = crisis_pseudonym(USER)
    assert isinstance(first, uuid.UUID)
    assert first == crisis_pseudonym(USER)
    assert first != USER
    assert first != crisis_pseudonym(uuid.UUID("99999999-2222-4333-8444-555555555555"))


@pytest.mark.parametrize("salt", [None, "short-salt"])
def test_mvp_029_strict_requires_a_16_byte_salt(monkeypatch, salt):
    if salt is not None:
        monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", salt)
    with pytest.raises(PHIConfigError) as exc:
        crisis_pseudonym(USER)
    assert "short-salt" not in str(exc.value)


def test_mvp_029_non_uuid_input_is_rejected(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", SALT)
    with pytest.raises(ValueError):
        crisis_pseudonym("not-a-uuid")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security/test_mvp_029_crisis_pseudonym.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'crisis_pseudonym' from 'solace_security.phi_codec'`. Without it, notification-service cannot compute WS-12.12's pseudonym: `get_phi_codec()` needs `EncryptionSettings`, whose `master_key` is required (encryption.py:49-52), and notification holds only the salt, so an erased user's crisis episodes could not be pseudonymized consistently with the safety escalations.
- [ ] **Step 3: Implement** `crisis_pseudonym` with `hmac.new(salt, message, hashlib.sha256)` per the produced formula, reading the salt through `_SearchSaltSettings` (never `EncryptionSettings`, which requires the master key) and applying the strict length rule with `is_strict_environment()`. Nothing logs or raises with the salt, the user id or the digest.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_security -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.18: Bedrock Converse provider, the only real-user LLM route in staging and production

**Covers:** none (implements D-04 as decided 2026-10-19; supports MVP-057 and MVP-199)
**Severity:** high
**Wave:** 1
**Depends on:** WS-09.3, WS-01.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (decided 2026-10-19, changed: real-user LLM traffic goes to Anthropic models on AWS Bedrock under the AWS BAA; strict environments require `LLM_PROVIDER=bedrock`; `LLM_PROVIDER=portkey` is allowed only in development and test with synthetic data; the LLM stays off for real users until the owner accepts the BAA in AWS Artifact and it is recorded through `LLM_PHI_EGRESS_APPROVED` (WS-09.4); unchanged: caching off, no semantic cache, minimum-necessary prompts, tests use only stubs and fakes)
**Real infra:** no
**Files:**
- Modify: `services/shared/infrastructure/llm_client.py` (`LLMClientSettings.provider: Literal["bedrock", "portkey"] = "portkey"`, read as `LLM_PROVIDER`; new `class BedrockSettings(BaseSettings)` with `model_config = SettingsConfigDict(env_prefix="BEDROCK_", extra="ignore")` (process environment only, no `.env` file) and fields `region: str = "us-east-1"`, `model_id: str = ""`, `endpoint_url: str = ""`, `timeout_seconds: float = Field(default=20.0, gt=0)`; `UnifiedLLMClient.__init__(self, settings=None, *, client_factory=None, bedrock_settings=None, bedrock_client_factory=None)`; module-level `_default_bedrock_client_factory(*, region_name, endpoint_url, config)` that imports boto3 lazily and returns `boto3.session.Session().client("bedrock-runtime", region_name=region_name, endpoint_url=endpoint_url, config=config)`; private `_bedrock_config(settings: BedrockSettings) -> botocore.config.Config`; pure helpers `_to_converse_request(messages, *, model_id, max_tokens, temperature) -> dict[str, Any]` and `_converse_text(response: Mapping[str, Any]) -> str`; the routing and strict refusals in `initialize()`; the Bedrock branch of `_execute_completion`; the botocore error mapping in `generate()`; boto3 and botocore are imported only on the Bedrock path, so the module still imports where they are absent)
- Create: `tests/integration/phi-codec-audit-llm/test_mvp_d04_bedrock_provider.py`
- Test: `tests/integration/phi-codec-audit-llm/test_mvp_d04_bedrock_provider.py`

**Interfaces:**
- Consumes: WS-09.2 `UnifiedLLMClient(..., client_factory=...)`, `LLMConfigurationError` and the local `_strict_environment()`; WS-09.3 `LLMUnavailableError`, `LLMResult`, `generate()` and `generate_with_fallback()`; WS-01.2's lock pins of `boto3` and `botocore` (CR-16; until the venv is re-synced to the lock, its system site packages provide boto3 and botocore 1.40.61).
- Produces (IF-15 as amended by D-04; the setting names are the decision's IF-02 names, registered by WS-03 through CR-14):
  - `LLM_PROVIDER` in {bedrock, portkey} (`LLMClientSettings.provider`; default `portkey` for development and test, so a strict deployment that omits it fails at startup). `BedrockSettings` reads `BEDROCK_REGION` (default `us-east-1`), `BEDROCK_MODEL_ID` (no default), `BEDROCK_ENDPOINT_URL` (test and E2E only) and `BEDROCK_TIMEOUT_SECONDS` (default 20). `UnifiedLLMClient(settings=None, *, client_factory=None, bedrock_settings=None, bedrock_client_factory=None)`; `bedrock_settings` defaults to `BedrockSettings()`, so existing `UnifiedLLMClient(settings)` call sites need no change.
  - `bedrock_client_factory(*, region_name, endpoint_url, config)` receives exactly these three keywords and never credentials (boto3's default chain resolves the host's instance role, or `AWS_PROFILE` in local development) and returns an object with `converse(**request) -> dict`. `config` is `botocore.config.Config(connect_timeout=5, read_timeout=BEDROCK_TIMEOUT_SECONDS, retries={"total_max_attempts": 2, "mode": "standard"})`, plus `signature_version=botocore.UNSIGNED` only when `BEDROCK_ENDPOINT_URL` is set, so the fake LLM needs no AWS credentials; `endpoint_url` is `None` when the setting is empty.
  - `initialize()`: in strict environments `LLM_PROVIDER=portkey` raises `LLMConfigurationError` naming `LLM_PROVIDER`, and a non-empty `BEDROCK_ENDPOINT_URL` raises `LLMConfigurationError` naming the variable and never its value; both checks run first, before any factory call and whatever the approval flag says. With `LLM_PROVIDER=bedrock`, an empty `BEDROCK_MODEL_ID` builds no client and leaves `is_available=False` (ERROR `llm_disabled_missing_configuration` with `missing=["BEDROCK_MODEL_ID"]` in strict, WARNING outside); WS-09.4 makes it a startup error in strict once `LLM_PHI_EGRESS_APPROVED=true`, the only state in which the model would be called. boto3 or botocore not importable, or a factory exception, raises `LLMConfigurationError` in strict and disables outside (WS-09.2's rule; only the exception type name is logged).
  - Request: `_to_converse_request` puts every system-role message, in order, into `system=[{"text": ...}]`; maps `assistant` to assistant and every other role to user; drops empty texts and any assistant turn before the first user turn; merges consecutive same-role turns into one message with several text blocks; sends `inferenceConfig={"maxTokens": ..., "temperature": ...}` and never `topP`, so a model that refuses both sampling parameters together cannot fail every task-preset call (presets keep only their temperature on this route); never sends `cachePoint`, `requestMetadata`, `additionalModelRequestFields`, `promptVariables` or `guardrailConfig`, so no prompt, user id or trace metadata reaches a cache or AWS-side metadata. The call is `await asyncio.wait_for(asyncio.to_thread(client.converse, **request), timeout=BEDROCK_TIMEOUT_SECONDS)`.
  - Reply: `_converse_text` joins the `text` blocks of `output.message.content`; `generate_with_fallback` returns `LLMResult(text=..., degraded=False, provider="bedrock")`.
  - Teardown: the Bedrock client is held where WS-09.2's `shutdown()` finds it, so `shutdown()` also calls the botocore `bedrock-runtime` client's synchronous `close()` and never raises (WS-04's W1 row, for WS-04.4's ordered teardown; pinned by `test_d04_shutdown_closes_the_bedrock_client`).
  - Errors, mapped to the existing IF-15 types only: `asyncio.TimeoutError`, `botocore.exceptions.ReadTimeoutError` and `botocore.exceptions.ConnectTimeoutError` become `LLMUnavailableError(reason="timeout")`; `botocore.exceptions.ClientError` and any other `BotoCoreError` become `reason="provider_error"`, raised `from None`; a reply without text becomes `reason="empty_response"`. The `llm_generation_failed` line carries `reason`, `service`, `error_type` and, for a `ClientError`, `error_code` (for example `ThrottlingException`); no log line, error message or cache ever holds prompt or reply text.
  - Consumers: WS-09.4 (the egress gate and `readiness()` over both providers), WS-09.5 (caching refused with `LLM_PROVIDER=bedrock`), WS-09.6 (the message order this converter hoists and merges); every service lane that builds a `UnifiedLLMClient` (the CR-7 set: WS-10, WS-11, WS-12, WS-15, WS-16, WS-18), with no call-site change; WS-05.1's fake LLM `POST /model/{model_id}/converse` route and the E2E stacks (CR-3, WS-05.21, which depends on this task); WS-03's env registry (CR-14) and the instance-role grant (CR-15); WS-01.2's lock pins (CR-16). It also answers WS-03's W1 row for WS-03.19 and WS-03.20 (exactly the five WS-03.19 names, credentials only from the default boto3 chain, never `AWS_*` key variables, and the strict refusals of `BEDROCK_ENDPOINT_URL` and `LLM_PROVIDER=portkey`; WS-03.19 depends on this task) and WS-04's W1 row for WS-04.4 (the teardown above).

- [ ] **Step 1: Write the failing test** in `tests/integration/phi-codec-audit-llm/test_mvp_d04_bedrock_provider.py`
```python
"""D-04 (2026-10-19): the Bedrock Converse provider is the only real-user LLM route in strict environments."""
from __future__ import annotations

import sys
import threading

import botocore.session
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError
from botocore.stub import Stubber

from services.shared.infrastructure import llm_client as llm_module
from services.shared.infrastructure.llm_client import (
    BedrockSettings,
    LLMClientSettings,
    LLMConfigurationError,
    LLMResult,
    LLMUnavailableError,
    UnifiedLLMClient,
)

MODEL_ID = "test.synthetic-model-v1"  # synthetic; never a real AWS model id
REPLY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Synthetic reply."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
    "metrics": {"latencyMs": 42},
}
HISTORY = [
    {"role": "system", "content": "[User Context] memories"},
    {"role": "assistant", "content": "Welcome back."},
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "b"},
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in (
        "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "LLM_PROVIDER", "LLM_PHI_EGRESS_APPROVED", "LLM_ENABLE_CACHING",
        "BEDROCK_REGION", "BEDROCK_MODEL_ID", "BEDROCK_ENDPOINT_URL", "BEDROCK_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "test")


def _stubbed():
    runtime = botocore.session.get_session().create_client(
        "bedrock-runtime", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
    )
    return runtime, Stubber(runtime)


def _client(runtime=None, calls=None, **bedrock):
    def factory(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return runtime if runtime is not None else object()

    return UnifiedLLMClient(
        LLMClientSettings(_env_file=None, provider="bedrock"),
        bedrock_settings=BedrockSettings(**{"model_id": MODEL_ID, **bedrock}),
        bedrock_client_factory=factory,
    )


class _RecordingLogger:
    def __init__(self):
        self.events = []

    def _record(self, event, **kwargs):
        self.events.append((event, kwargs))

    debug = info = warning = error = exception = _record


class _HangingRuntime:
    def __init__(self):
        self.release = threading.Event()

    def converse(self, **_):
        self.release.wait(5)
        return REPLY


class _TimeoutRuntime:
    def converse(self, **_):
        raise ReadTimeoutError(endpoint_url="http://fake-llm.invalid")


async def test_d04_converse_request_shape_and_reply():
    runtime, stubber = _stubbed()
    stubber.add_response("converse", REPLY, {
        "modelId": MODEL_ID,
        "system": [{"text": "SYS"}, {"text": "[User Context] memories"}],
        "messages": [
            {"role": "user", "content": [{"text": "a"}]},
            {"role": "assistant", "content": [{"text": "ok"}]},
            {"role": "user", "content": [{"text": "b"}, {"text": "now"}]},
        ],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.2},
    })
    client = _client(runtime)
    await client.initialize()
    with stubber:
        result = await client.generate_with_fallback(
            system_prompt="SYS", user_message="now", fallback_response="template",
            conversation_history=HISTORY, task_type="crisis", service_name="t",
            metadata={"user_id": "user-a", "request_id": "r1"},
        )
        stubber.assert_no_pending_responses()
    assert result == LLMResult(text="Synthetic reply.", degraded=False, provider="bedrock")


async def test_d04_identical_prompts_are_never_served_from_a_cache():
    runtime, stubber = _stubbed()
    expected = {
        "modelId": MODEL_ID,
        "system": [{"text": "s"}],
        "messages": [{"role": "user", "content": [{"text": "same"}]}],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.7},
    }
    stubber.add_response("converse", REPLY, expected)
    stubber.add_response("converse", REPLY, expected)
    client = _client(runtime)
    await client.initialize()
    with stubber:
        for _ in range(2):
            assert await client.generate(system_prompt="s", user_message="same", service_name="t") == "Synthetic reply."
        stubber.assert_no_pending_responses()


async def test_d04_client_error_is_provider_error_without_prompt_text(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(llm_module, "logger", recorder)
    runtime, stubber = _stubbed()
    stubber.add_client_error(
        "converse", service_error_code="ThrottlingException",
        service_message="Too many requests for: hello there", http_status_code=429,
    )
    client = _client(runtime)
    await client.initialize()
    with stubber, pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="hello there", service_name="t")
    assert exc.value.reason == "provider_error"
    assert "hello" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__ is True
    assert client.get_statistics()["total_errors"] == 1
    failed = [kwargs for event, kwargs in recorder.events if event == "llm_generation_failed"]
    assert failed and failed[0]["error_code"] == "ThrottlingException"
    assert "hello there" not in repr(recorder.events)


async def test_d04_slow_call_times_out_off_the_event_loop():
    runtime = _HangingRuntime()
    client = _client(runtime, timeout_seconds=0.2)
    await client.initialize()
    try:
        with pytest.raises(LLMUnavailableError) as exc:
            await client.generate(system_prompt="s", user_message="u", service_name="t")
    finally:
        runtime.release.set()
    assert exc.value.reason == "timeout"


async def test_d04_botocore_read_timeout_maps_to_timeout():
    client = _client(_TimeoutRuntime())
    await client.initialize()
    with pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="u", service_name="t")
    assert exc.value.reason == "timeout"


async def test_d04_reply_without_text_is_empty_response():
    runtime, stubber = _stubbed()
    stubber.add_response("converse", {**REPLY, "output": {"message": {"role": "assistant", "content": []}}})
    client = _client(runtime)
    await client.initialize()
    with stubber, pytest.raises(LLMUnavailableError) as exc:
        await client.generate(system_prompt="s", user_message="u", service_name="t")
    assert exc.value.reason == "empty_response"


async def test_d04_client_config_has_timeouts_bounded_retries_and_no_credentials():
    calls = []
    client = _client(calls=calls)
    await client.initialize()
    assert client.is_available is True
    assert set(calls[0]) == {"region_name", "endpoint_url", "config"}
    assert calls[0]["region_name"] == "us-east-1"
    assert calls[0]["endpoint_url"] is None
    config = calls[0]["config"]
    assert (config.connect_timeout, config.read_timeout) == (5, 20.0)
    assert config.retries == {"total_max_attempts": 2, "mode": "standard"}
    assert config.signature_version is not UNSIGNED


async def test_d04_test_endpoint_is_unsigned():
    calls = []
    client = _client(calls=calls, endpoint_url="http://fake-llm:8080")
    await client.initialize()
    assert calls[0]["endpoint_url"] == "http://fake-llm:8080"
    assert calls[0]["config"].signature_version is UNSIGNED


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_d04_strict_refuses_portkey(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    calls = []
    client = UnifiedLLMClient(
        LLMClientSettings(_env_file=None, provider="portkey"), client_factory=lambda **kw: calls.append(kw)
    )
    with pytest.raises(LLMConfigurationError, match="LLM_PROVIDER"):
        await client.initialize()
    assert calls == []


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_d04_strict_refuses_endpoint_override(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    calls = []
    client = _client(calls=calls, endpoint_url="http://fake-llm:8080")
    with pytest.raises(LLMConfigurationError) as exc:
        await client.initialize()
    assert "BEDROCK_ENDPOINT_URL" in str(exc.value)
    assert "fake-llm" not in str(exc.value)
    assert calls == []


@pytest.mark.parametrize("env", ["test", "staging"])
async def test_d04_missing_model_id_builds_no_client(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    calls = []
    client = _client(calls=calls, model_id="")
    await client.initialize()
    assert client.is_available is False
    assert calls == []


@pytest.mark.parametrize(("env", "raises"), [("test", False), ("production", True)])
async def test_d04_missing_boto3(monkeypatch, env, raises):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("LLM_PHI_EGRESS_APPROVED", "true")  # read from WS-09.4 on; strict approval is what reaches the SDK
    monkeypatch.setitem(sys.modules, "boto3", None)  # the lazy import raises ImportError
    client = UnifiedLLMClient(
        LLMClientSettings(_env_file=None, provider="bedrock"), bedrock_settings=BedrockSettings(model_id=MODEL_ID)
    )
    if raises:
        with pytest.raises(LLMConfigurationError):
            await client.initialize()
    else:
        await client.initialize()
        assert client.is_available is False


class _ClosableRuntime:
    def __init__(self):
        self.closed = False

    def close(self):  # botocore's BaseClient.close() is synchronous
        self.closed = True


async def test_d04_shutdown_closes_the_bedrock_client():
    runtime = _ClosableRuntime()
    client = _client(runtime)
    await client.initialize()
    assert client.is_available is True
    await client.shutdown()
    assert runtime.closed is True
    assert client.is_available is False


def test_d04_settings_defaults_and_provider_values():
    bedrock = BedrockSettings()
    assert (bedrock.region, bedrock.model_id, bedrock.endpoint_url, bedrock.timeout_seconds) == ("us-east-1", "", "", 20.0)
    assert LLMClientSettings(_env_file=None).provider == "portkey"
    with pytest.raises(ValueError):
        LLMClientSettings(_env_file=None, provider="openai")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration/phi-codec-audit-llm/test_mvp_d04_bedrock_provider.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'BedrockSettings' from 'services.shared.infrastructure.llm_client'`. No Bedrock route exists: `LLMClientSettings` has no `provider` field and `UnifiedLLMClient.__init__` accepts no `bedrock_client_factory` (`TypeError` once the import is fixed), so every real-user call would go through Portkey, the route D-04 forbids in strict environments, and nothing refuses Portkey or an endpoint override in staging or production.
- [ ] **Step 3: Implement.** First load the `amazon-bedrock` skill (Skill tool) and follow its Converse API guidance; check against it the request shape (system blocks, `user`/`assistant` roles starting with a user turn, the `inferenceConfig` names), the throttling and validation error codes and the retry configuration, and record any difference as a ledger Ruling before writing code. Then, in `services/shared/infrastructure/llm_client.py`: add `LLMClientSettings.provider` and `BedrockSettings`; in `initialize()`, apply the strict refusals (provider, endpoint override) before anything else, then for `bedrock` the missing-model-id rule, then build `_bedrock_config(...)` and call `self._bedrock_client_factory(region_name=..., endpoint_url=... or None, config=...)`, mapping `ImportError` and factory exceptions exactly as WS-09.2 does for Portkey; in `_execute_completion`, for `bedrock`, build `_to_converse_request(...)` from the `_build_messages` output and await `asyncio.wait_for(asyncio.to_thread(self._client.converse, **request), timeout=self._bedrock.timeout_seconds)`; map the errors in `generate()` per the produced contract (re-raise `from None`; take `error_code` from `exc.response["Error"]["Code"]` only). Import boto3 and botocore inside the Bedrock path only. Never log, cache or attach to an exception any message text, and never write a model id into code: `BEDROCK_MODEL_ID` is set at deploy after checking availability with `bedrock:ListFoundationModels` and the inference profiles (WS-03, CR-14), and the tests use the synthetic id above. The tests never call AWS: Stubber short-circuits every request, and the fixture empties the credential chain.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/therapy_service/tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/personality_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass except the five node ids recorded in WS-09.3 Step 5 (CR-5, CR-6). WS-09.2's suite stays green: its strict missing-key case sets `LLM_PROVIDER=bedrock`, and its strict missing-SDK case now raises `LLMConfigurationError` through the Portkey refusal.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-09.F1: The PHI log sanitizer never recurses into the logging chain
**Covers:** MVP-180
**Severity:** high
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none. Added by controller Ruling on 2026-11-02 from WS-04.3's concern 1. `PHIDetector.detect` (`src/solace_security/phi_protection.py:273-278`) logs `phi_detected` from inside the structlog chain. The chain re-binds the request context, so a bound value that looks like PHI (a 9-digit id, `a1234567`, about 1 in 61 uuid4 strings) makes every log call in that request raise `RecursionError`, the crisis path included. One crafted `X-Request-ID: 123456789` header triggers it. The probe is `D:\tmp\mvp-exec\ws04\ws043_phi_recursion_probe.py`.
**Real infra:** no
**Files:**
- Modify: `src/solace_security/phi_protection.py`
- Test: `tests/solace_security/test_mvp_180_sanitizer_reentrancy.py`
**Interfaces:**
- Consumes: the WS-04.2 chain, `solace_infrastructure.runtime.logging.configure_service_logging`.
- Produces: logging from inside the sanitizer (including the `phi_detected` counter line) never re-enters the sanitizer and never raises; a log call whose bound context contains PHI-looking values is sanitized once and emitted once.

- [ ] **Step 1: Write the failing test.** Configure logging with a production `ServiceRuntimeSpec`, then `structlog.contextvars.bind_contextvars(request_id="123456789", user_id="a1234567")`. Call `structlog.get_logger("t").info("hello")` and assert it neither raises nor recurses (wrap it in a recursion-limit guard), that exactly one JSON line is written, and that neither `123456789` nor `a1234567` appears unmasked. Second case: 200 calls with random `uuid4()` request ids bound, all of which complete.
- [ ] **Step 2: Run it and confirm it fails.** Expected: FAIL with `RecursionError` for the bound-PHI case.
- [ ] **Step 3: Implement** a re-entrancy guard: a module-level `contextvars.ContextVar("_phi_sanitizing", default=False)`. While it is set, the processor returns the event dict unchanged, and the `phi_detected` line goes to the stdlib logger at DEBUG with counts and types only, or is not emitted. Never let the guard leak across tasks: set it and reset it by token in `finally`.
- [ ] **Step 4: Run to pass.** Expected: PASS.
- [ ] **Step 5: Regression:** tests\solace_security, tests\bootstrap (the known `test_runtime_package_import_is_light` waits on WS-01.3) and tests\integration.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-09.F1:head"`.

## Cross-workstream requests

| # | Owner WS | Exact change | Files | Needed by |
|---|---|---|---|---|
| CR-1 | WS-03 | W0 phase-A2 env sync: the ENCRYPTION_MASTER_KEY recipe becomes `python -c "import secrets; print(secrets.token_hex(32))"` (64 hex characters; `base64:` alternative noted); the registry note for `ENCRYPTION_MASTER_KEY` states the accepted formats and that raw 32-character keys are dev/test only. | `docs/STAGING-BRINGUP.md`, `deploy/env/env-contract.yaml` | WS-09.1 |
| CR-2 | WS-04 | W0 phase A2: ProductionGuards validates `ENCRYPTION_MASTER_KEY` with `solace_security.encryption.parse_master_key(value, strict=True)` (no length-only check) and reports only the variable name. | `src/solace_security/production_guards.py` | WS-09.1 |
| CR-3 | WS-05 | W1 phase A1 (D-04, 2026-10-19): `docker-compose.test.yml` sets, on orchestrator, safety, therapy, diagnosis and personality: in the dev profile `LLM_PROVIDER=bedrock`, `BEDROCK_ENDPOINT_URL=http://fake-llm:8080` (WS-05.1's `POST /model/{model_id}/converse` route), a synthetic `BEDROCK_MODEL_ID` such as `fake-llm-model` (never a real AWS model id) and `LLM_PHI_EGRESS_APPROVED=true`, with no AWS credentials (the client sends unsigned requests when `BEDROCK_ENDPOINT_URL` is set, WS-09.18); in the strict profile `LLM_PROVIDER=bedrock`, `LLM_PHI_EGRESS_APPROVED=false` and no `BEDROCK_ENDPOINT_URL`, because strict refuses the override, so strict stacks run template replies. Neither profile sets `LLM_PORTKEY_*` or provider API keys. Planned as WS-05.21 (W1 phase A), which depends on WS-09.18 and WS-03.19; WS-09.4 and WS-09.18 need no reverse dependency, because their unit tests use stubs and the overlay is their E2E proof. | `docker-compose.test.yml` | WS-09.4, WS-09.18 (E2E proof) |
| CR-4 | WS-03 | W1 phase-A env sync: registry entries `LLM_PHI_EGRESS_APPROVED` (orchestrator, safety, therapy, diagnosis, personality; non-secret; dev_default false; false in strict until the owner accepts the AWS BAA in AWS Artifact and it is recorded, D-04), `LLM_ENABLE_CACHING` (same services; must be false in strict), `LLM_TIMEOUT_SECONDS`; `LLM_PORTKEY_API_KEY`, `LLM_PORTKEY_GATEWAY_URL` and `LLM_ANTHROPIC_API_KEY` stay development/test-only entries that the strict render never sets (D-04); mark `PORTKEY_API_KEY`, `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` retired; `docs/STAGING-BRINGUP.md` documents `LLM_PROVIDER=bedrock`, the `BEDROCK_*` settings (CR-14) and `LLM_PHI_EGRESS_APPROVED` instead of `PORTKEY_API_KEY`. | `deploy/env/env-contract.yaml`, `docs/STAGING-BRINGUP.md` | WS-09.4, WS-09.5 |
| CR-5 | WS-15 | W1 phase-B supporting task (WS-15 has no W1 lane; added by ledger Ruling): in `test_llm_client.py`, `TestLLMClientSettings::test_default_settings` expects `enable_caching is False` and `cache_mode == "simple"`; `TestUnifiedLLMClient::test_build_portkey_config_with_caching` uses `cache_mode="simple"` and asserts `_build_portkey_config(cache_namespace="solace:user:u")["cache"] == {"mode": "simple"}`; `::test_generate_no_client` and `::test_generate_error_handling` expect `LLMUnavailableError` (error count 1 for the provider error); `::test_generate_with_fallback` and `::test_generate_with_fallback_on_empty` expect `LLMResult(text="Fallback response here", degraded=True, provider="fallback")`; `::test_get_statistics` uses `cache_mode="simple"`. W2 (optional): pass the memory-context block as `pinned_context` from `context_assembler.py`. | `services/therapy_service/tests/test_llm_client.py`, `services/therapy_service/src/infrastructure/context_assembler.py` | WS-09.3, WS-09.5, WS-09.6 |
| CR-6 | WS-05 | W1 (tests/integration owner; by the shared-files rule the code owner may make the edit if the controller rules so): `TestLLMRequestTimeout::test_generate_returns_empty_on_timeout_not_hang` becomes `test_generate_raises_unavailable_on_timeout` and expects `LLMUnavailableError` with `reason == "timeout"`. | `tests/integration/test_sprint7_portkey_fallback.py` | WS-09.3 |
| CR-7 | WS-10, WS-11, WS-12, WS-15, WS-16, WS-18 | W1/W2 phase B in each service that builds a `UnifiedLLMClient`: `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)`; catch `LLMUnavailableError` (or use `generate_with_fallback(...)` and report `degraded`) instead of testing for `""`. | each service's `main.py` and LLM call sites | WS-09.4 |
| CR-8 | WS-04 | W0: replace the `configure_phi_encryption` requirement in `tests/integration/test_phi_encryption_activation.py` for every service whose W0 IF-01 adoption removes the call (it locks calls that IF-13 retires), but keep the file and keep `safety_service` in `_PHI_ENCRYPTING` until WS-12.12 moves it out in W3 (WS-12's R-12-7; WS-12.12 edits that file); and assert `SPEC.handles_phi` for orchestrator, safety, therapy, diagnosis, memory, personality and user-service in `tests/bootstrap/test_service_conformance.py`. W2 phase A: bootstrap step 5 calls `get_phi_codec()` and turns `PHIConfigError` into `StartupDependencyError` in strict. W3 phase A: `configure_service_audit(spec.service_name, runtime.pg_pool)` when `spec.audited`; register `audit_chain_verifier` as a background task in user-service only; ProductionGuards requires `AUDIT_HMAC_KEY` (at least 32 bytes, not a known dev value) for audited services. | `src/solace_infrastructure/runtime/bootstrap.py`, `src/solace_security/production_guards.py`, `tests/bootstrap/test_service_conformance.py`, `tests/integration/test_phi_encryption_activation.py` | WS-09.7, WS-09.11, WS-09.12, WS-09.15 |
| CR-9 | WS-03 | W3 phase A: registry entry `AUDIT_HMAC_RETIRED_KEYS_JSON` (secret; audited services; dev_default "[]"; not required in strict) and its IF-22 secret entry; `AUDIT_HMAC_KEY` delivered to every audited service. | `deploy/env/env-contract.yaml`, `docker-compose.prod.yml`, `deploy/render_secrets.py` | WS-09.11 |
| CR-10 | WS-10, WS-12, WS-14, WS-15, WS-16, WS-17, WS-18 | W2 phase B (WS-12: W3 phase B): adopt IF-13 in the repository modules listed in WS-09.15's `ADOPTERS` (each imports `solace_security.phi_codec`), stop calling `get_phi_encryptor`/`configure_phi_encryption`, write `codec.active_key_id` to `encryption_key_id`; confirm or amend the ADOPTERS paths at the W2 barrier (ledger Ruling). The tasks are WS-10.10, WS-14.10, WS-15.8, WS-16.8, WS-16.9, WS-17.5, WS-18.4 and WS-12.12 (safety's module is `services/safety_service/src/domain/escalation.py`, where `PostgresEscalationRepository` lives; WS-12.12 also removes the `configure_phi_encryption` block from safety's `main.py`). | the eight ADOPTERS files | WS-09.15 |
| CR-11 | WS-02 | Declined by owner (WS-02, its waves are W0 and W2 phase A, so no WS-02 task can depend on the W3 phase-B work this cleanup must follow; left to a W3 ledger Ruling that schedules it after WS-12.12, WS-16.8 and WS-09.15): Optional cleanup no earlier than the W3 barrier (ledger): delete `configure_phi_encryption`, `get_phi_encryptor` and the ClinicalBase listeners from `base_models.py` together with `tests/integration/test_phi_at_rest.py` only after WS-12.12 and WS-16.8 have removed the last callers and WS-09.15 is green. Until then they stay importable unchanged (WS-12's R-12-7: safety's W0 `main.py` still imports `configure_phi_encryption` until WS-12.12). | `src/solace_infrastructure/database/base_models.py` | none (WS-09.15 asserts services no longer call them) |
| CR-12 | WS-02 | W0 slot 009: grant nothing on `audit_events` (slot 011 creates it and revokes UPDATE/DELETE/TRUNCATE from `solace_app`); the default privileges must give `solace_app` USAGE and SELECT on new sequences so `audit_events_chain_seq_seq` is usable. | `migrations/versions/009_platform_roles_rls.py` | WS-09.10 |
| CR-13 | WS-03 | W4 (with the WS-03.9 backup schedule and the WS-03.11 host): run `docker compose exec -T user-service python -m solace_security.audit_runtime verify --textfile /textfile/solace_audit.prom` nightly from the host timer, and mount volume `solace_textfile_metrics` at `/textfile` in the strict user-service container, so `solace_audit_chain_verify_ok` reaches WS-04.12's node-exporter. | `infrastructure/aws-staging/solace-staging.yaml`, `docker-compose.prod.yml` | WS-04.12 (the `AuditChainVerifyFailed` alert's data) |
| CR-14 | WS-03 | W1 phase-A env sync (WS-03.6), D-04: registry entries on orchestrator, safety, therapy, diagnosis and personality: `LLM_PROVIDER` (non-secret; dev_default `portkey`; required in strict with the value `bedrock`), `BEDROCK_REGION` (non-secret; dev_default `us-east-1`), `BEDROCK_MODEL_ID` (non-secret; no default in any profile; set at deploy after checking availability with `bedrock:ListFoundationModels` and the inference profiles, never guessed; required in strict once `LLM_PHI_EGRESS_APPROVED=true`), `BEDROCK_ENDPOINT_URL` (non-secret; test and E2E only; `tests/deploy/test_env_contract.py` asserts the strict render does not set it), `BEDROCK_TIMEOUT_SECONDS` (non-secret; dev_default 20). No `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` or `AWS_SESSION_TOKEN` entry in any profile or in `deploy/secrets/<env>.sops.yaml`: credentials come from the host's instance role (`AWS_PROFILE` only for local development). Planned as WS-03.19 (W1 phase A), whose registry rows follow these names and rules exactly; WS-03.19 depends on WS-09.18 (the registry records what the provider reads), so WS-09.18 takes no reverse dependency. `docs/STAGING-BRINGUP.md` gets the names in WS-03.8 and WS-03.15. | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml`, `docs/STAGING-BRINGUP.md` | WS-09.18 (deployment; no reverse dependency) |
| CR-15 | WS-03 | With the pilot host (D-04): the instance role in `infrastructure/aws-staging` gets `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` only, scoped through a template parameter to the chosen foundation-model or inference-profile ARNs (an inference profile also needs the foundation-model ARNs it routes to; WS-03 confirms this with the amazon-bedrock skill), and nothing broader. The client calls only `Converse`, which `bedrock:InvokeModel` authorizes. | `infrastructure/aws-staging/solace-staging.yaml` | WS-09.18 (at runtime on the staging host; its unit tests need no AWS) |
| CR-16 | WS-01 | W0 phase A1 (WS-01.2), D-04: `boto3` and `botocore` in `requirements.in` and therefore in the runtime `requirements.lock`, not only the dev lock, pinned at the version installed in the anaconda environment (`pip show boto3`; 1.40.61 at planning time) unless an advisory forces a bump. | `requirements.in`, `requirements.lock` | WS-09.18 |

## New interfaces

Beyond IF-13, IF-14 and IF-15 as written in the skeleton:

- `solace_security.encryption.parse_master_key(value: str, *, strict: bool) -> bytes`; `EncryptionSettings.master_key_bytes: bytes`; `EncryptionSettings.key_id` (`ENCRYPTION_KEY_ID`, default "k1", `^[A-Za-z0-9_-]{1,32}$`) and `EncryptionSettings.retired_keys_json` (`ENCRYPTION_RETIRED_KEYS_JSON`, JSON object `{key_id: key}`).
- `solace_security.phi_codec`: constructor `PHICodec(*, active_key_id, keys, search_salt, allow_plaintext)`, `PHICodec.from_settings(settings, *, strict=None)`, `PHIDecryptError`, `reset_phi_codec()`; envelope `enc:v1:<key_id>:<base64url without padding>`; AAD `"<field>:<canonical owner id>"`; HKDF info `b"solace-phi:" + key_id`; search-hash input `purpose + "\x1f" + value`.
- `services.shared.infrastructure.llm_client`: `LLMConfigurationError`; `UnifiedLLMClient(..., *, client_factory=None)`; `LLMUnavailableError.reason` values {"not_configured", "disabled", "timeout", "provider_error", "empty_response"}; `async readiness() -> tuple[bool, str]` with the strings in WS-09.4; `LLMClientSettings.phi_egress_approved`; cache namespace `solace:user:<user_id>`; `_build_portkey_config(cache_namespace=None)`; `MAX_HISTORY_MESSAGES = 10`.
- `solace_security.audit.AuditEventType.ADMIN_ASSIGNMENT_CHANGED` ("admin_assignment_changed") and `.CONSENT_CHANGED` ("consent_changed") in addition to the IF-14 list (used by WS-14.18); values of the IF-14 members are the lower-snake names in WS-09.9.
- `solace_security.audit_runtime`: `AuditConfigError`, `AuditKeyring.from_keys(active, retired=())`, `AuditWriter(service_name, pool, keyring).write(...) -> int`, `verify_chain_on(pool, keyring, *, start_seq=None, end_seq=None, batch_size=1000)`, `ChainVerification(ok, checked, first_bad_seq, reason, tip_seq, tip_hash)`, `compute_event_hash(fields, previous_hash, key)`, `sanitize_details(details)`, `ALLOWED_DETAIL_KEYS`, `reset_audit_runtime()`, `audit_chain_verifier(*, interval_seconds=86400.0, stop_event=None)`, gauge `solace_audit_chain_tip_seq`; environment variable `AUDIT_HMAC_RETIRED_KEYS_JSON` (JSON array of retired keys; key ids derived as the first 12 hex characters of `sha256(b"solace-audit-key-id:" + key)`).
- `audit_events` schema and privileges as listed in WS-09.10.
- `solace_security.phi_protection.LogEventSanitizer` and its policy constants (WS-09.13).
- `python -m solace_security.audit_runtime verify [--start-seq N] [--textfile PATH]` and `solace_security.audit_runtime.main(argv=None, *, pool_factory=None) -> int`, with the stdout line, exit codes and textfile gauges of WS-09.16 (requested by WS-05 for WS-05.15 and by WS-04 for WS-04.12).
- `solace_security.phi_codec.crisis_pseudonym(user_id: UUID | str) -> UUID` (WS-09.17, R-13-3 of WS-13).
- The Bedrock route (WS-09.18, D-04 as decided 2026-10-19): `LLMClientSettings.provider` (`LLM_PROVIDER` in {bedrock, portkey}); `BedrockSettings` (`BEDROCK_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_ENDPOINT_URL`, `BEDROCK_TIMEOUT_SECONDS`); `UnifiedLLMClient(..., bedrock_settings=None, bedrock_client_factory=None)`; `_default_bedrock_client_factory(*, region_name, endpoint_url, config)`; `_bedrock_config(settings)`; `_to_converse_request(messages, *, model_id, max_tokens, temperature)`; `_converse_text(response)`; `LLMResult.provider == "bedrock"`; the log field `error_code` on `llm_generation_failed`; the readiness string `"disabled: missing BEDROCK_MODEL_ID"` (WS-09.4).
- `UnifiedLLMClient.shutdown()` closes the SDK client and never raises (WS-09.2, requested by WS-04 for WS-04.4). Audit write failures use WS-04.3's `AUDIT_WRITE_FAILURES` counter; `audit_runtime` declares only the two chain gauges.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-055 | high | WS-09.7 (codec), WS-09.15 (closure lock in W3 phase B, after CR-10's eight adoption tasks) |
| MVP-056 | high | WS-09.5 |
| MVP-057 | high | WS-09.4 (code); CR-4 (docs and registry) |
| MVP-091 | blocker | WS-09.10, WS-09.11, WS-09.12, WS-09.16 (verify CLI); closes at the W3 barrier with CR-8 wiring, the service lanes' `audit()`/`audit_phi_access` calls and WS-05's `test_w3_privacy.py` audit assertions |
| MVP-092 | high | WS-09.1; CR-1, CR-2; the "staging boots with encryption silently off" half closes when CR-8's W2 codec warm-up lands |
| MVP-093 | high | WS-09.13 |
| MVP-139 | medium | WS-09.3 (personality's 0.65-confidence path disappears because its `except` now receives `LLMUnavailableError`) |
| MVP-187 | medium | WS-09.7 |
| MVP-189 | medium | WS-09.14 (absorbs MVP-313) |
| MVP-199 | medium | WS-09.2 |
| MVP-205 | medium | WS-09.6 |
| MVP-315 | low | WS-09.13 |
| MVP-316 | low | WS-09.13 |
| MVP-345 | blocker | WS-09.9, WS-09.10, WS-09.11 |
| MVP-346 | high | WS-09.9 (legacy), WS-09.11 (real chain under three writers) |
| MVP-351 | medium | WS-09.8 (codec keyring), WS-09.11 (audit HMAC key ids) |

WS-09.18 closes no register entry by itself: it implements D-04 as decided on 2026-10-19 (the strict LLM route) and supports MVP-057 and MVP-199.

**Interface consistency check**
- IF-13 names and signatures (`get_phi_codec`, `PHICodec.encrypt/decrypt/search_hash`, `encrypt_fields`, `decrypt_fields`, `active_key_id`, `PHIConfigError`, the envelope and AAD) match the skeleton; WS-14.10 consumes exactly these.
- IF-14 `configure_service_audit(service_name, pool)`, `audit(...)` keyword set, `audit_phi_access(resource_type, *, subject=None)` and `verify_chain(start_seq=None, end_seq=None) -> ChainVerification` match the skeleton; the advisory lock key `hashtext('solace_audit_chain')`, `chain_seq BIGSERIAL` and the metric names `solace_audit_write_failures_total{service}` and `solace_audit_chain_verify_ok` match IF-14 and IF-18. WS-14.13 and WS-14.18 use `AuditEventType.ACCOUNT_ERASED_IN_SERVICE`, `ACCOUNT_DELETED`, `ADMIN_CLINICIAN_PROVISIONED`, `ADMIN_ONCALL_CHANGED`, `ADMIN_ASSIGNMENT_CHANGED` and `CONSENT_CHANGED` with details keys inside `ALLOWED_DETAIL_KEYS`.
- IF-15 `LLMUnavailableError`, `LLMResult(text, degraded, provider)`, `pinned_context`, the `LLM_` setting names and the strict egress rule match the skeleton.
- D-04 (2026-10-19) amends IF-15 and IF-02: `LLM_PROVIDER`, `BEDROCK_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_ENDPOINT_URL` and `BEDROCK_TIMEOUT_SECONDS` are used exactly as the decision file names them; WS-03 registers them (CR-14), WS-05.1's Converse route and the E2E stacks use them (CR-3), and the existing `LLM_PORTKEY_*` names stay for development and test only.
- IF-01: only `is_strict_environment()` is consumed; phase-A1 W0 tasks use a local copy of the same rule because the runtime package does not exist yet, and WS-09.4/WS-09.7 switch to the shared function.

**Incoming requests, each planned no later than the requesting task's wave**
- WS-04 (WS-04.4, W0): `shutdown()` closes the SDK client → WS-09.2 (W0 phase A1).
- WS-04 (WS-04.8, WS-04.10, WS-04.12): `get_phi_codec`/`PHIConfigError` → WS-09.7 (W2); `configure_service_audit` counting on `AUDIT_WRITE_FAILURES` → WS-09.11 (W3); the nightly textfile → WS-09.16 (W3), scheduled by WS-03 through CR-13.
- WS-05 request 7 (WS-05.15, W3): the verify CLI → WS-09.16 (W3 phase A).
- WS-13 R-13-3 (WS-13.14, W3): `crisis_pseudonym` → WS-09.17 (W3 phase A).
- WS-02 R-02.6 (slot 011 revokes UPDATE/DELETE/TRUNCATE) → WS-09.10 (W3).
- WS-12 R-12-7 (keep `configure_phi_encryption` importable until WS-12.12): WS-09 never retires the listeners and owns neither named file; honoured by the amended CR-8 and CR-11 and by WS-09.15 depending on WS-12.12.

**Review focus (skeleton order 1-5)**
- 1 (crisis phrasing false negatives) and 2 (alert-path partial failures): no WS-09 code is on those paths.
- 3: WS-09.10 (solace_app may only INSERT/SELECT `audit_events`); WS-09.16 (the CLI runs as solace_app and must detect a tampered row, so a CLI that silently reads nothing fails).
- 4: WS-09.3 (provider error text never echoed), WS-09.5 (cache namespace per user, caching off in strict), WS-09.11 (details allow-list), WS-09.12 (no bodies in audit rows), WS-09.13 (nested, non-string and exception values in logs), WS-09.18 (no prompt text in Bedrock requests' metadata, logs, errors or a cache; no user id sent to AWS).
- 5: WS-09.1 (master-key formats in strict), WS-09.4 (renamed LLM key names, the egress flag and the Bedrock model id in strict), WS-09.7 (strict codec without keys), WS-09.8 (rotation with old ciphertext), WS-09.11 (short HMAC key; rotated HMAC key), WS-09.16 (config error is exit 1, never a silent pass), WS-09.17 (strict salt length), WS-09.18 (strict refuses Portkey and the endpoint override; a missing SDK is a startup error).

REVIEW_FOCUS: 3=WS-09.10,WS-09.16; 4=WS-09.3,WS-09.5,WS-09.11,WS-09.12,WS-09.13,WS-09.18; 5=WS-09.1,WS-09.4,WS-09.7,WS-09.8,WS-09.11,WS-09.16,WS-09.17,WS-09.18

**Open risks**
- WS-15 has no W1 lane: CR-5 needs a ledger Ruling to add a supporting task in W1, or the W1 barrier keeps the listed therapy assertions red.
- `cache_namespace` as an `AsyncPortkey` constructor argument is taken from the MVP-056 evidence (portkey-ai 1.9.x `client.py:272`) and cannot be checked locally because portkey-ai is not installed; the tests use a fake factory. The implementer of WS-09.5 checks the pinned SDK in an image built from the lock before marking the task done.
- The audit chain is global and serialized: every audited write takes one advisory lock (expected 1 to 3 ms), which bounds audit throughput to a few hundred writes per second for the pilot. Tail truncation by the table owner remains detectable only through the nightly tip gauge and backups (`solace_audit_chain_tip_seq`), and a compromised service holding `AUDIT_HMAC_KEY` can still append forged rows; both are documented residual risks under D-10.
- The sanitizer's prose rule (four or more words) and the free-text key list will redact some developer log text (for example `reason=` sentences); that over-redaction is deliberate. Licence and health-plan detection becomes case-sensitive (WS-09.14), so lower-case licence numbers are no longer masked.
- WS-09.15's ADOPTERS map assumes the module paths in CR-10; a lane that stores PHI in another module must amend the map by ledger Ruling at the W2 barrier. Because safety adopts the codec only in W3 (WS-12.12), WS-09.15 is the one WS-09 task in a phase B; if WS-12.12 slips past the W3 barrier, WS-09.15 slips with it and MVP-055 stays open.
- The verify CLI (WS-09.16) relies on WS-02.6's `ConnectionPoolManager` reading `POSTGRES_*`; if a service container lacks `AUDIT_HMAC_KEY` (CR-9), the CLI exits 1 by design, which WS-05.15 reports as a failed audit check rather than a pass.
- The legacy `AuditLogger`, `AsyncAuditLogger` and in-memory stores remain importable for tests; `PostgresAuditStore.initialize()` refuses to run, so no service can fall back to the per-process chain.
- D-04 Bedrock route (WS-09.18): the model id is chosen at deploy, and the LLM stays off for real users until the AWS BAA is accepted and `LLM_PHI_EGRESS_APPROVED` records it, so the pilot may run template replies only; the plan never names a model id. `topP` is never sent, so task presets keep only their temperature on this route. Converse needs alternating roles that start with a user turn, so the converter merges consecutive same-role turns and drops an assistant turn that precedes the first user turn. A call that times out keeps its worker thread until botocore's own bound (two attempts, each with a 5 s connect and a `BEDROCK_TIMEOUT_SECONDS` read timeout); the caller already has `LLMUnavailableError(reason="timeout")`. The orchestrator's 8 s agent timeout is shorter than the 20 s Bedrock default, as it was with Portkey; this plan does not change it.

## Review responses

- Error: plan a task for WS-05's request 7 (`python -m solace_security.audit_runtime verify [--start-seq N]`, exit 0/1, JSON `{"ok", "first_bad_seq"}`, for WS-05.15): fixed. New task WS-09.16 (W3 phase A, depends on WS-09.11, WS-09.12 (the same module, serialized), WS-02.6 and WS-05.2) adds the CLI with a unit suite and a real subprocess test against slot 011; its Produces line names WS-05.15.
- Error: plan a task for WS-12's R-12-7 (keep `configure_phi_encryption` importable as a shim; keep `safety_service` in `_PHI_ENCRYPTING` until WS-12.12): rejected as a WS-09 task. WS-09 never retires the ORM listeners and owns neither named file: `src/solace_infrastructure/database/base_models.py` falls under WS-02's owned path `src/solace_infrastructure/database` (skeleton.json, WS-02 owned_paths), and `tests/integration/test_phi_encryption_activation.py` is outside WS-09's owned paths (plan line 21; skeleton WS-09 owned_paths). No WS-09 task edits either file (File map). The intent is honoured instead: CR-11 (the only deletion request, to WS-02) now reads "no earlier than the W3 barrier, after WS-12.12 and WS-16.8 removed the last callers"; CR-8 (to WS-04) now keeps the activation test and `safety_service` in `_PHI_ENCRYPTING` until WS-12.12 moves it; and WS-09.15 depends on WS-12.12 and runs in W3 phase B.
- Error: plan a task for WS-13's R-13-3 (`solace_security.phi_codec.crisis_pseudonym(user_id) -> UUID`, salt-only, strict `PHIConfigError` for a missing or short salt, for WS-13.14): fixed. New task WS-09.17 (W3 phase A, depends on WS-09.7); its test pins equality with WS-12.12's `search_hash(..., purpose="crisis_pseudonym")[:32]` formula, salt-only operation and the strict salt rule.
- Warning: WS-09.10 depends on a workstream (WS-02): fixed. Now `WS-09.9, WS-02.3, WS-05.2` (WS-02.3 scaffolds slot 011 and the 009 default privileges; WS-05.2 provides `pg_migrated`).
- Warning: WS-09.12 depends on a workstream (WS-07): fixed. Now `WS-09.11, WS-07.2, WS-07.4` (`AuthenticatedUser.via_service` from WS-07.2; obo-token acceptance in `get_current_user` from WS-07.4).
- Warning: WS-09.15 depends on workstreams (WS-10, WS-12, WS-15, WS-16, WS-17, WS-18, WS-04): fixed. Now `WS-09.8, WS-04.8, WS-10.10, WS-12.12, WS-14.10, WS-15.8, WS-16.8, WS-16.9, WS-17.5, WS-18.4`. Because WS-12.12 is a W3 phase-B task, WS-09.15 (test-only, no platform path) moves to W3 phase B, so no phase-A task depends on a same-wave phase-B task. The ADOPTERS entry for safety is corrected to `services/safety_service/src/domain/escalation.py`, where `PostgresEscalationRepository` lives (escalation.py:580) and where WS-12.12 adds the codec; WS-12's codec work does not touch `infrastructure/repository.py`. The RED expectation now also names safety's `configure_phi_encryption` block at `main.py:116-121`.
- Additional (found while resolving; not in the findings file): WS-04's requests to WS-09 (WS-04 plan, Cross-workstream requests) had no task. `shutdown()` closing the SDK client for WS-04.4 (W0) joins WS-09.2 with a failing test. The nightly textfile for WS-04.12 joins WS-09.16 (`--textfile`), with the host schedule and volume mount requested from WS-03 as CR-13 (WS-03 owns `docker-compose.prod.yml` and `infrastructure/aws-staging`). WS-09.11 now counts failures on WS-04.3's `AUDIT_WRITE_FAILURES` instead of declaring a second `solace_audit_write_failures_total`, which would have raised `Duplicated timeseries` in any service that imports both modules.
- Review focus: the one-line focus summary (areas 3, 4 and 5) is added to the Self-review. Focus areas 1 and 2 have no WS-09 code.
- Reconciliation 3 (CR-11, the optional `base_models.py` cleanup requested from WS-02): marked "Declined by owner (WS-02, ...)" in the table. WS-02 has only W0 and W2 phase-A tasks and cannot depend on the W3 phase-B work (WS-12.12, WS-16.8) that must remove the last callers first; WS-02's plan already records it under "Not planned". It is left to a W3 ledger Ruling after WS-09.15 is green, which may assign it to any lane; nothing in WS-09 waits on it (WS-09.15 scans only `services/`).
- Reconciliation 9 (WS-15.22 named `services/shared/infrastructure/llm_client.py`, WS-09's file, in the same wave as WS-09.6): resolved. WS-15.22 never changed the client; its Files line described it. The client side stays with WS-09: WS-09.6, the last W2 change to the client, now has a Produces line naming WS-15.22, WS-15.22 depends on WS-09.6 and no longer names the client path, and WS-09.6's regression step names the seven CR-5 cases as the expected failures until WS-15.22 lands.
- decision 2026-10-19 D-04: new task WS-09.18 (W1 phase A, Opus, security-critical; depends on WS-09.3 and WS-01.2) adds the Bedrock Converse provider to `services/shared/infrastructure/llm_client.py`: boto3 through `asyncio.to_thread` under `asyncio.wait_for`, an explicit botocore Config (5 s connect, `BEDROCK_TIMEOUT_SECONDS` read, two total attempts), the `LLM_PROVIDER` switch, `BedrockSettings`, strict refusals of Portkey and `BEDROCK_ENDPOINT_URL`, no prompt logging or caching, errors mapped to `LLMUnavailableError`/`LLMConfigurationError`, Stubber and fake-client tests that never call AWS, and Step 3 loads the amazon-bedrock skill. Its Produces line names the consumers. The W1 chain on the client is now WS-09.3, WS-09.18, WS-09.4, WS-09.5, so the egress gate and the caching validator are written once over both providers.
- decision 2026-10-19 D-04: WS-09.18 builds no client when `BEDROCK_MODEL_ID` is missing, and WS-09.4 makes that a startup error in strict once `LLM_PHI_EGRESS_APPROVED=true`. This follows the D-04 rule "approved plus missing configuration is a startup error" and lets a pre-BAA strict stack start with the LLM off, without a placeholder model id. The Portkey refusal and the endpoint-override refusal apply in strict whatever the approval flag says.
- decision 2026-10-19 D-04: WS-09.2: the gate wording is updated; the Produces line says strict environments refuse Portkey from WS-09.18 on, so its Portkey rules govern development and test; the strict missing-key test sets `LLM_PROVIDER=bedrock` (unread until WS-09.18) and the fixture clears `LLM_PROVIDER`, `BEDROCK_MODEL_ID` and `BEDROCK_ENDPOINT_URL`, so the test stays valid after WS-09.18 and WS-09.4.
- decision 2026-10-19 D-04: WS-09.4 now depends on WS-09.18 instead of WS-09.3. Its strict tests use the Bedrock route: approved without `BEDROCK_MODEL_ID` is a startup error, not approved is disabled, approved with a model id is available. The Portkey key-name check moves to a non-strict test (`test_mvp_057_undocumented_names_are_never_read`). The strict Portkey provider-key rule (`LLM_ANTHROPIC_API_KEY`, `LLM_OPENAI_API_KEY`) is removed, because strict never runs Portkey. The readiness string for Bedrock is added, and the gate, Produces, RED and Step 3 text are rewritten to match.
- decision 2026-10-19 D-04: WS-09.5 also refuses `LLM_ENABLE_CACHING=true` with `LLM_PROVIDER=bedrock` (new test `test_mvp_056_caching_refused_with_bedrock`); the fixture clears `LLM_PROVIDER`; the Produces line says per-namespace clients exist only on the development and test Portkey route. Caching off, no semantic cache and strict refusal are unchanged.
- decision 2026-10-19 D-04: cross-workstream requests: CR-3 (WS-05) now asks for `LLM_PROVIDER=bedrock` with `BEDROCK_ENDPOINT_URL=http://fake-llm:8080` and a synthetic model id in the dev profile, and for the LLM off in the strict profile, instead of Portkey keys. CR-4 (WS-03) records the AWS BAA in AWS Artifact, keeps the `LLM_PORTKEY_*` names development/test-only and documents the Bedrock settings in STAGING-BRINGUP. New CR-14 (WS-03 registry entries for `LLM_PROVIDER` and `BEDROCK_*`, with no static AWS keys), CR-15 (WS-03 instance-role `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`, scoped to parameterized ARNs) and CR-16 (WS-01.2 pins `boto3` and `botocore` in the runtime lock).
- decision 2026-10-19 D-04: the header (goal, architecture, waves, WS-01 dependency), the file map, New interfaces and the Self-review (coverage note, interface check, review focus, REVIEW_FOCUS line and an open-risk entry) now describe the Bedrock route and WS-09.18.
- decision 2026-10-19 D-07: WS-09.8's gate now names SOPS + age: keys live in `deploy/secrets/<env>.sops.yaml`, are rendered by `deploy/render_secrets.py` into `/run/secrets`, and rotation re-encrypts and redeploys. It no longer says "per-environment secrets store with escrow". The `NAME_FILE` consumer contract and the task's code are unchanged.
