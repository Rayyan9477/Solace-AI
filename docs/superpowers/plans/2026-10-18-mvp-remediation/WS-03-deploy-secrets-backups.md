# WS-03: Deploy topology, env contract, edge, secrets, backups and host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dev and strict compose profiles, the env registry, Caddy and the AWS template set exactly what the code reads, strict profiles boot on per-environment secrets delivered as files from a secrets store, nothing internal is exposed, every PHI store and the key material needed to read it is backed up with a measured restore, and the docs match reality.

**Architecture:** A machine-readable registry (`deploy/env/env-contract.yaml`, IF-02) drives `.env.example`, the synthetic env used by tests and the harness, and `tests/deploy/test_env_contract.py`, which renders `docker compose config` for both profiles and enforces the contract. Secrets move in W3 from interpolated environment values to one file per secret under `SOLACE_SECRETS_DIR`, mounted per service at `/run/secrets/<name>` and loaded through `NAME_FILE` (IF-22), generated per developer in dev and, in strict, decrypted from the environment's SOPS + age file (`deploy/secrets/<env>.sops.yaml`, D-07) by `deploy/render_secrets.py` with pinned `sops`/`age` into root-only 0400 files on the tmpfs `/run/secrets`. W4 adds `docker-compose.backup.yml` (WAL archiving, encrypted base backups, restore tooling), key escrow with key-id manifests, the hardened CloudFormation host, Caddy limits and container confinement.

**Covers:** 24 register entries (blocker 6, high 8, medium 4, low 6): MVP-002, MVP-003, MVP-004, MVP-037, MVP-040, MVP-042, MVP-043, MVP-044, MVP-045, MVP-115, MVP-116, MVP-117, MVP-118, MVP-225, MVP-231, MVP-242, MVP-247, MVP-250, MVP-308, MVP-348, MVP-349, MVP-350, MVP-352, MVP-354.

**Waves:** 0, 1, 3, 4, 5 (all tasks are phase A: compose files, `deploy/**`, `infrastructure/aws-staging/**`; WS-03 has no W2 wave, so W2 env requests from other plans land one wave early, in WS-03.6).

**Depends on:**
- WS-01: images built from the repo root, `docker/entrypoint.sh` (NAME_FILE loader), `docker/migrate.Dockerfile` (solace-migrate), `.dockerignore`, the realinfra/e2e markers, CI targets for cfn-lint/cfn-guard.
- WS-02: the migrate job command (`python -m scripts.db.migrate upgrade head`), the `solace_app`/`solace_migrator` roles created by `infrastructure/init-db`, Postgres TLS certificates from `scripts/db/gen_certs.py`, and a pg_hba policy that allows the backup sidecar's replication connection.
- WS-04 (paired): `/live` semantics (IF-18), Prometheus job names, the observability fragment that consumes the Alertmanager variables and the backup textfile metric, and the W3 secret-source guards.
- WS-05: the e2e harness and `docker-compose.test.yml`, which overlay these files and must never weaken the strict profile; its W4 restore drill and `BackupStale` fault test drive the backup CLI of WS-03.16.
- WS-02 (W4 use): `scripts.db.images.POSTGRES_IMAGE` (WS-02.1), the digest-pinned postgres reference WS-03.17 adopts.
- WS-06 (W5 use): create-if-missing of the `solace_events.config.TOPIC_CONFIGS` topics at event-plane start (request R-01.4(b), not yet a WS-06 task) before WS-03.18 turns broker auto-create off in strict.
- WS-09 and WS-05 (W1 use, D-04): the Bedrock provider reads WS-03.19's names and takes credentials from the instance role; the fake LLM's Converse route and the e2e overlay values for `LLM_PROVIDER`/`BEDROCK_ENDPOINT_URL`.
- Owner (D-25, D-04, D-12; external actions, no task): attaching the deployer policies to `MedCodex-Local`, accepting the AWS BAA in AWS Artifact, enabling Anthropic model access, verifying the SES identity and requesting SES production access, generating the per-environment and escrow age key pairs off-host, and rotating the working-tree `.env` keys.

**Owned paths:** `docker-compose.yml`, `docker-compose.prod.yml`, `docker-compose.backup.yml`, `.env.example`, `Caddyfile`, `deploy/**`, `infrastructure/aws-staging/**`, `services/config_service/**` (frozen), `tests/deploy/**`, `tests/config_service/**`, `README.md`, `system-design/**`, `docs/STAGING-BRINGUP.md`, `docs/SYSTEM-DESIGN-SUMMARY.md`, `docs/KNOWN-LIMITATIONS.md`, `docs/runbooks/deploy.md`, `docs/runbooks/secrets.md`, `docs/runbooks/backup-restore.md`. WS-03 also edits the compose-string assertions in `tests/security/test_config_secrets_hardening.py` because it owns the compose files they test (shared-test-dir rule).

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md. IF-02 and IF-22 are produced here; the registry-schema and secret-file additions are listed under "New interfaces".

**Command conventions used below** (PowerShell 5.1, one pytest process per suite):
- `UNIT <path>` means: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`
- `DOCKER <path>` means the same command prefixed with `$env:SOLACE_RUN_DOCKER_TESTS="1"; ` (tests in `tests/deploy` marked `integration` plus `realinfra` or `e2e` start containers and are skipped unless that variable or `SOLACE_REQUIRE_REALINFRA=1` is set; with either set, a missing `docker` CLI is a failure). Rendering `docker compose config` needs only the docker CLI and runs in the plain unit suite.
- `RENDER` means `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m deploy.env.render_env_example --write` (run from `D:\Repo\Contextual-Chatbot` with the PYTHONPATH above) after every registry edit.
- `BOOTSMOKE` (from W1 on, when WS-05's file exists) means `$env:SOLACE_E2E_PROFILE="dev"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_w0_boot_smoke.py` then `$env:SOLACE_E2E_PROFILE="strict"; $env:SOLACE_E2E_ENVIRONMENT="production"; UNIT D:\Repo\Contextual-Chatbot\tests\e2e\test_w0_boot_smoke.py`.
- Regression set `DEPLOYREG`: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy`; `UNIT D:\Repo\Contextual-Chatbot\tests\security\test_config_secrets_hardening.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\alignment`; `UNIT D:\Repo\Contextual-Chatbot\tests\config_service`.

---

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `deploy/__init__.py`, `deploy/env/__init__.py` | Create | Make the deploy tooling importable as `deploy.*` | WS-03.1 |
| `deploy/env/env-contract.yaml` | Create / Modify | The IF-02 registry | WS-03.1 to WS-03.7, WS-03.19, WS-03.8, WS-03.9, WS-03.14, WS-03.15 |
| `deploy/env/registry.py` | Create / Modify | `EnvVar`, `load_registry`, `generate_value`, `synthetic_values`, `write_synthetic_env`, `APP_SERVICES`; W1 `LLM_SERVICES`, `DENIED_NAMES`; W3 `derive`, `SECRET_READER_UIDS`, `secret_file_owner` | WS-03.1, WS-03.19, WS-03.7, WS-03.8 |
| `deploy/env/scan_env_file.py` | Create | Names-only scan (and in-place blanking) of secret values in an env file, for the working-tree `.env` rotation and the strict settings file | WS-03.7 |
| `deploy/env/render_env_example.py` | Create | Renders `.env.example`; `--write` / `--check` | WS-03.1 |
| `deploy/env/compose_render.py` | Create / Modify | Hermetic `docker compose config` rendering; W0 profiles; W3 placeholder secret files | WS-03.1, WS-03.4, WS-03.7 |
| `deploy/env/env_readers.py` | Create | AST scan of env names read by code (`os.getenv`, `os.environ`, `BaseSettings` prefixes) | WS-03.1 |
| `deploy/env/generate_dev_env.py` | Create / Modify | Per-developer / per-run secret files; W4 backup key, key ids, RSA keys | WS-03.7, WS-03.9, WS-03.10, WS-03.14 |
| `deploy/render_secrets.py` | Create / Modify | Strict secret files decrypted from `deploy/secrets/<env>.sops.yaml` with pinned sops/age into root-only 0400 tmpfs files (D-07); `--check` without decrypting; W4 key ids, escrow-recipient check and derived public keys | WS-03.8, WS-03.10, WS-03.14 |
| `deploy/sops/tools.json`, `deploy/sops/install_tools.sh`, `deploy/sops/Dockerfile` | Create | Pinned sops/age versions and checksums, their verified install, and the `solace/sops-tools` image the host renders in | WS-03.8 |
| `deploy/key_ids.py` | Create | One-way key fingerprints and `key-ids.json` | WS-03.10 |
| `deploy/key_escrow.py` | Create | Offline escrow bundle export/verify (X25519 + AES-GCM) | WS-03.10 |
| `deploy/backup/Dockerfile` | Create / Modify | `solace/pg-backup` image (same postgres image + gnupg + aws-cli); W4 CLI copy and digest-pinned base | WS-03.9, WS-03.16, WS-03.17 |
| `deploy/backup/solace-backup.sh` | Create / Modify | Agent loop: WAL push, nightly base backup, retention, textfile metric, S3 sync; `base-now`; W4 key-id manifest; W4 base-backup lock | WS-03.9, WS-03.10, WS-03.16 |
| `deploy/backup/solace-backup-cli.sh` | Create | Operator and drill commands `backup-now`, `restore-latest`, `status`, `age-metric` | WS-03.16 |
| `deploy/backup/solace-restore.sh` | Create / Modify | Restore + PITR, refuses mismatched key material | WS-03.9, WS-03.10 |
| `docker-compose.yml` | Modify | Dev profile and the base for strict | WS-03.1 to WS-03.7, WS-03.19, WS-03.9, WS-03.13, WS-03.14, WS-03.15, WS-03.17 |
| `docker-compose.prod.yml` | Modify | Strict overlay | WS-03.1 to WS-03.7, WS-03.19, WS-03.13, WS-03.14, WS-03.15, WS-03.17, WS-03.18 |
| `docker-compose.backup.yml` | Create / Modify | Backup and restore fragment | WS-03.9, WS-03.10, WS-03.13, WS-03.16, WS-03.17 |
| `.env.example` | Modify | Rendered from the registry (never hand-edited) | every task that edits the registry |
| `Caddyfile` | Modify | Edge routes, health, limits | WS-03.5, WS-03.6, WS-03.12 |
| `infrastructure/aws-staging/solace-staging.yaml` | Modify | W3 host access (D-04/D-07/D-25): scoped Bedrock invoke, no secret-store access, IMDSv2 for containers, artifact bucket, secrets tmpfs; W4 ingress, EIP, DNS, data volume, backup bucket and its retained KMS key, alarms | WS-03.20, WS-03.11 |
| `infrastructure/aws-staging/solace-staging-deployer-policy.json` | Modify | The self-sufficient least-privilege customer managed policy the owner attaches to `MedCodex-Local` for the W3 host (D-25) | WS-03.20 |
| `infrastructure/aws-staging/solace-staging-deployer-host-policy.json` | Create | The second customer managed policy with the W4 host resources (D-09 S3 + KMS, EIP, DNS, alarms), attached the same way | WS-03.11 |
| `infrastructure/aws-staging/guard/solace-staging.guard` | Create | cfn-guard rules | WS-03.11 |
| `infrastructure/aws-staging/DEPLOY.md` | Modify | Owner grant and owner actions, token-free tarball delivery, SOPS render, alarms, backups | WS-03.20, WS-03.8, WS-03.11 |
| `docs/runbooks/secrets.md`, `docs/runbooks/backup-restore.md`, `docs/runbooks/deploy.md` | Create / Modify | Operator runbooks | WS-03.8, WS-03.9, WS-03.10, WS-03.15, WS-03.16 |
| `docs/STAGING-BRINGUP.md`, `docs/KNOWN-LIMITATIONS.md`, `docs/SYSTEM-DESIGN-SUMMARY.md`, `README.md`, `system-design/README.md` | Modify | Docs truth | WS-03.5, WS-03.8, WS-03.15 |
| `tests/security/test_config_secrets_hardening.py` | Modify | Compose-string assertions follow the derived/file forms | WS-03.1, WS-03.8 |
| `tests/deploy/conftest.py` | Create / Modify | `registry`, `render`, `compose_project`, `caddy_adapt`, `cfn_template`, `sops_tools` fixtures; docker gate | WS-03.1, WS-03.2, WS-03.5, WS-03.7, WS-03.20, WS-03.8, WS-03.9, WS-03.10, WS-03.16 |
| `tests/deploy/test_env_contract.py` | Create | Registry-driven invariants (IF-02) | WS-03.1 |
| `tests/deploy/test_mvp_*.py` | Create | One file per covered issue (names in each task) | each task |
| `tests/deploy/test_d04_llm_provider_env.py`, `tests/deploy/test_d25_aws_access.py` | Create | Decision-driven contracts (D-04 LLM routing; D-25/D-04/D-07/D-12 AWS access) | WS-03.19, WS-03.20 |

`services/config_service/**` and `tests/config_service/**` are frozen (D-19): no task edits them; the config_service suite must stay green in every regression run.

---

## Tasks

### Task WS-03.1: Env registry, rendered .env.example, compose render harness and ENVIRONMENT-derived service gates
**Covers:** MVP-004, MVP-247, MVP-231
**Severity:** blocker
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: strict means ENVIRONMENT in {staging, production}; every service gate variable equals ENVIRONMENT), D-19 (default assumed: config-service is removed from the MVP compose files and its code is frozen)
**Real infra:** no (renders `docker compose config` with the docker CLI; no containers)
**Cross-workstream prerequisites:** none.
**Files:**
- Create: `deploy/__init__.py`, `deploy/env/__init__.py`, `deploy/env/registry.py`, `deploy/env/env-contract.yaml`, `deploy/env/render_env_example.py`, `deploy/env/compose_render.py`, `deploy/env/env_readers.py`
- Create: `tests/deploy/conftest.py`, `tests/deploy/test_env_contract.py` (the IF-02 contract test, extended by WS-03.6)
- Modify: `docker-compose.yml` (every app service sets `ENVIRONMENT: ${ENVIRONMENT:-development}` and its gate variable `${ENVIRONMENT:-development}`; the `USER_SERVICE_ENV: development` literal is gone; the `config-service` block is removed; every `container_name:` is removed so containers are scoped by compose project; interpolation sources use the IF-22 names: `AUTH_SECRET_KEY` replaces `JWT_SECRET_KEY` for `AUTH_SECRET_KEY`, `USER_SERVICE_JWT_SECRET_KEY` and `SECURITY_JWT_SECRET`, and `USER_SERVICE_FIELD_ENCRYPTION_KEY` replaces `FIELD_ENCRYPTION_KEY`)
- Modify: `docker-compose.prod.yml` (every gate variable is `${ENVIRONMENT:?ENVIRONMENT must be set}`; the `config-service` block is removed; every `env_file: .env.prod` is removed and replaced by explicit `${SOURCE:?...}` lines for exactly the secrets each service reads per the registry; Caddy gets only `DOMAIN`, `CADDY_ADMIN_EMAIL`, `BASIC_AUTH_HASH`; the header comment lists the derivation rule and points at the registry)
- Modify: `.env.example` (rendered)
- Modify: `tests/security/test_config_secrets_hardening.py` (the three P1-9 base assertions change from `${MEMORY_ENVIRONMENT:-development}`, `${NOTIFICATION_SERVICE_ENV:-development}`, `${ANALYTICS_SERVICE_ENV:-development}` to `${ENVIRONMENT:-development}`; all other assertions unchanged)
- Test: `tests/deploy/test_mvp_004_environment_derivation.py`, `tests/deploy/test_mvp_231_config_service_removed.py` (plus `tests/deploy/test_env_contract.py`, created above)
**Interfaces:**
- Consumes: none
- Produces: the IF-02 registry with the schema in "New interfaces" (one row per container variable and service set: `name`, `source`, `services`, `profiles`, `secret`, `format`, `required_in_strict`, `dev_default`, `strict_default`, `example`, `group`, `notes`, `retired`); `deploy.env.registry`: `APP_SERVICES: tuple[str, ...]` (the nine), `FORMATS: frozenset[str]`, `@dataclass(frozen=True) EnvVar`, `load_registry(path: Path = REGISTRY_PATH) -> list[EnvVar]`, `generate_value(fmt: str) -> str`, `synthetic_values(registry, *, profile: Literal["dev","strict"], environment: str, extra_sources: Iterable[str] = ()) -> dict[str, str]`, `write_synthetic_env(path: Path, *, profile, environment) -> None` (single-quoted lines; for WS-05's harness); `deploy.env.compose_render`: `class ComposeRenderError(RuntimeError)`, `render_compose_config(files: Sequence[str], env: Mapping[str, str], *, profiles: Sequence[str] = (), workdir: Path) -> dict`; `deploy.env.env_readers.collect_env_readers(roots: Sequence[Path]) -> dict[str, list[str]]`; `deploy.env.render_env_example.render_env_example(registry) -> str` and `main(argv) -> int`.

- [ ] **Step 1: Write the failing test** (tests plus the render harness and a registry describing what compose sets today; no compose edits yet)
```python
# tests/deploy/conftest.py
import os, shutil, tempfile
from dataclasses import dataclass
from pathlib import Path
import pytest
from deploy.env.registry import load_registry, synthetic_values
from deploy.env.compose_render import render_compose_config

REPO = Path(__file__).resolve().parents[2]
PROFILES = {"dev": ["docker-compose.yml"], "strict": ["docker-compose.yml", "docker-compose.prod.yml"]}
# interpolation names the pre-change files still use, so the RED run renders today's compose (removed in WS-03.15)
LEGACY_RENDER_SOURCES = ("JWT_SECRET_KEY", "FIELD_ENCRYPTION_KEY")

def pytest_collection_modifyitems(config, items):
    run = os.environ.get("SOLACE_RUN_DOCKER_TESTS") == "1" or os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1"
    for item in items:
        if item.get_closest_marker("realinfra") or item.get_closest_marker("e2e"):
            if not run:
                item.add_marker(pytest.mark.skip(reason="starts containers: set SOLACE_RUN_DOCKER_TESTS=1"))
            elif shutil.which("docker") is None:
                item.add_marker(pytest.mark.xfail(reason="docker CLI missing", run=False, strict=True))

@dataclass
class Rendered:
    config: dict
    env: dict
    @property
    def services(self) -> dict:
        return self.config["services"]

@pytest.fixture(scope="session")
def registry():
    return load_registry()

@pytest.fixture(scope="session")
def render(registry, tmp_path_factory):
    cache, base = {}, {}
    def _render(profile, *, environment="production", overrides=None, drop=(), fragments=(), profiles=()):
        key = (profile, environment, tuple(sorted((overrides or {}).items())), tuple(drop), tuple(fragments), tuple(profiles))
        if key not in cache:
            env = dict(base.setdefault((profile, environment), synthetic_values(
                registry, profile=profile, environment=environment, extra_sources=LEGACY_RENDER_SOURCES)))
            env.update(overrides or {})
            for k in drop:
                env.pop(k, None)
            workdir = Path(tempfile.mkdtemp(dir=tmp_path_factory.getbasetemp()))
            cache[key] = Rendered(render_compose_config(PROFILES[profile] + list(fragments), env, profiles=profiles, workdir=workdir), env)
        return cache[key]
    return _render
```
```python
# tests/deploy/test_mvp_004_environment_derivation.py
"""MVP-004/MVP-247: every service gate variable is derived from ENVIRONMENT in both profiles, never a literal."""
import re
import pytest
from deploy.env.compose_render import ComposeRenderError
from tests.deploy.conftest import REPO

GATES = {"orchestrator-service": "ORCHESTRATOR_ENVIRONMENT", "user-service": "USER_SERVICE_ENV",
         "safety-service": "SAFETY_ENVIRONMENT", "notification-service": "NOTIFICATION_SERVICE_ENV",
         "diagnosis-service": "DIAGNOSIS_ENVIRONMENT", "memory-service": "MEMORY_ENVIRONMENT",
         "therapy-service": "THERAPY_ENVIRONMENT", "personality-service": "PERSONALITY_ENVIRONMENT",
         "analytics-service": "ANALYTICS_SERVICE_ENV"}

@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_gates_follow_ENVIRONMENT(render, environment):
    r = render("strict", environment=environment)
    for service, gate in GATES.items():
        env = r.services[service]["environment"]
        assert env["ENVIRONMENT"] == environment, service
        assert env[gate] == environment, f"{service}:{gate}"

def test_dev_gates_follow_ENVIRONMENT(render):
    r = render("dev", environment="development")
    for service, gate in GATES.items():
        assert r.services[service]["environment"]["ENVIRONMENT"] == "development", service
        assert r.services[service]["environment"][gate] == "development", f"{service}:{gate}"

def test_strict_render_fails_without_ENVIRONMENT(render):
    with pytest.raises(ComposeRenderError, match="ENVIRONMENT"):
        render("strict", drop=["ENVIRONMENT"])

def test_no_gate_variable_is_a_literal():
    text = (REPO / "docker-compose.yml").read_text(encoding="utf-8") + (REPO / "docker-compose.prod.yml").read_text(encoding="utf-8")
    for gate in GATES.values():
        for m in re.finditer(rf"^\s*{gate}:\s*(.+)$", text, re.M):
            assert m.group(1).strip().strip('"').startswith("${ENVIRONMENT"), (gate, m.group(1))
```
```python
# tests/deploy/test_mvp_231_config_service_removed.py
"""MVP-231: config-service (zero consumers, permanently unhealthy) is not deployed in the MVP profiles."""
import pytest

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_config_service_not_deployed(render, profile):
    r = render(profile)
    assert "config-service" not in r.services
    assert all("config-service" not in (svc.get("depends_on") or {}) for svc in r.services.values())
```
```python
# tests/deploy/test_env_contract.py
"""IF-02: the registry is the deployment env contract; compose, .env.example and code agree with it."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from deploy.env.registry import APP_SERVICES, FORMATS
from deploy.env.render_env_example import render_env_example
from deploy.env.env_readers import collect_env_readers
from deploy.env.compose_render import ComposeRenderError
from tests.deploy.conftest import REPO

def _index(registry, profile):
    idx = {}
    for v in registry:
        if not v.retired and profile in v.profiles:
            for s in v.services:
                idx.setdefault(s, set()).add(v.name)
    return idx

def test_registry_entries_are_well_formed(registry):
    seen = set()
    for v in registry:
        assert v.format in FORMATS and v.services and v.group, v.name
        for s in v.services:
            assert (v.name, s) not in seen, (v.name, s)
            seen.add((v.name, s))
        if v.secret:
            assert v.strict_default is None and v.example is None, v.name

def test_env_example_is_rendered_from_registry(registry):
    assert (REPO / ".env.example").read_text(encoding="utf-8") == render_env_example(registry)

def test_env_example_holds_no_secret_value(registry):
    lines = dict(l.split("=", 1) for l in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
                 if l and not l.startswith("#"))
    for v in registry:
        if v.secret and v.source and not v.retired:
            assert lines.get(v.source, "") in ("", "''"), v.source

@pytest.mark.parametrize("profile,environment", [("dev", "development"), ("strict", "staging"), ("strict", "production")])
def test_registered_variables_are_set_for_their_services(render, registry, profile, environment):
    r = render(profile, environment=environment)
    idx = _index(registry, profile)
    missing = sorted((s, n) for s, names in idx.items() if s in r.services
                     for n in names if n not in r.services[s].get("environment", {}))
    assert not missing

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_app_variable_is_registered(render, registry, profile):
    r = render(profile)
    idx = _index(registry, profile)
    extra = {s: sorted(set(r.services[s].get("environment", {})) - idx.get(s, set()))
             for s in (*APP_SERVICES, "migrate", "caddy") if s in r.services}
    assert {s: e for s, e in extra.items() if e} == {}

def test_no_service_reads_an_env_file(render):
    assert [s for s, d in render("strict").services.items() if d.get("env_file")] == []

def test_containers_are_project_scoped(render):
    assert [s for s, d in render("dev").services.items() if d.get("container_name")] == []

def test_no_dev_default_survives_in_strict(render, registry):
    r = render("strict")
    leaks = sorted((s, v.name) for v in registry if v.dev_default is not None and v.strict_default is None
                   and not v.retired and "strict" in v.profiles for s in v.services
                   if s in r.services and r.services[s]["environment"].get(v.name) == v.dev_default)
    assert not leaks

def test_required_strict_sources_fail_fast(render, registry):
    sources = sorted({v.source for v in registry if v.required_in_strict and v.source and not v.retired})
    def silently_rendered(src):
        try:
            render("strict", drop=[src])
        except ComposeRenderError as exc:
            return None if src in str(exc) else src
        return src
    with ThreadPoolExecutor(max_workers=8) as pool:
        offenders = [s for s in pool.map(silently_rendered, sources) if s]
    assert offenders == []

def test_no_code_reads_a_retired_name(registry):
    readers = collect_env_readers([REPO / "src", REPO / "services"])
    assert {v.name: readers[v.name] for v in registry if v.retired and v.name in readers} == {}
```
The registry written in this step lists the variables compose sets today plus the IF-02 auth, PHI, audit, LLM, OAuth, orchestrator-timeout and memory rows that WS-03.1 adds in Step 3 (each with services, secret, format, required_in_strict and defaults). `compose_render.render_compose_config` copies the named compose files into `workdir`, writes `workdir/render.env` (every value single-quoted), creates an empty `workdir/.env.prod` placeholder (so pre-change files that still reference `env_file: .env.prod` render without ever reading a real one), runs `docker compose --project-directory <workdir> -p solace-render --env-file <workdir>/render.env -f ... [--profile p] config --format json` with a scrubbed process environment (only `PATH`, `SYSTEMROOT`, `TEMP`, `TMP`, `USERPROFILE`, `HOME`, `APPDATA`, `LOCALAPPDATA`, `ProgramData`, `DOCKER_HOST`, `DOCKER_CONFIG`, `DOCKER_CONTEXT`), and raises `ComposeRenderError(stderr)` on a non-zero exit. The developer's `.env` is never read and rendered output is never printed.
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy`
Expected: FAIL. `test_strict_gates_follow_ENVIRONMENT[staging]` fails with `user-service:USER_SERVICE_ENV` ('development' from the literal at docker-compose.yml:206, MVP-004) and, after that line, `therapy-service:THERAPY_ENVIRONMENT` ('development', MVP-247); `test_dev_gates_follow_ENVIRONMENT` fails with `KeyError: 'ENVIRONMENT'`; `test_no_gate_variable_is_a_literal` fails on `USER_SERVICE_ENV: development`; `test_config_service_not_deployed` fails (docker-compose.yml:507, MVP-231); `test_env_example_is_rendered_from_registry`, `test_no_service_reads_an_env_file` (prod services use `env_file: .env.prod`) and `test_containers_are_project_scoped` fail.
- [ ] **Step 3: Implement** the compose edits listed under Files, the full W0 registry rows for the edited variables, `render_env_example`, `collect_env_readers` and `generate_value`. Approach: `collect_env_readers` parses every `.py` under the roots (skipping any `tests` directory) with `ast`, collecting string literals passed to `os.getenv`, `os.environ.get`, `os.environ[...]`, and, for every class whose body assigns `model_config = SettingsConfigDict(env_prefix="X", ...)`, `X + FIELD.upper()` for each annotated field; `generate_value` returns `secrets.token_urlsafe(24|48)`, `secrets.token_hex(32)`, `Fernet.generate_key().decode()`, a `$2a$14$`-prefixed 53-character random string for `bcrypt` (render tests only; WS-03.7 makes real hashes), and the registry `example` for non-secret formats; `.env.example` groups rows under `# --- <group> ---` headings, writes one comment line per source (secret flag, required-in-strict flag, format, notes), writes secret sources with an empty value, and single-quotes any value containing `$`. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; and `git -C D:\Repo\Contextual-Chatbot diff --stat <the task's base tree id> -- services/config_service tests/config_service` prints nothing (D-19 frozen code). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.2: Data plane: migrate job, solace_app over verified TLS, Kafka and peer URLs everywhere, orchestrator on the data network
**Covers:** MVP-003, MVP-116
**Severity:** blocker
**Wave:** 0
**Depends on:** WS-03.1, WS-01.5, WS-01.6, WS-01.9
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: one database `solace`; services connect only as `solace_app`; `solace_migrator` only in the migrate job; the image superuser `postgres` only for init, ops and backups), D-15 (default assumed: strict uses `verify-full` against host `postgres` with `/run/secrets/postgres-ca.crt`; dev uses `prefer`)
**Real infra:** yes (dev stack: postgres, then the migrate job, then role assertions)
**Cross-workstream prerequisites:** WS-01.9 (W0 phase A2, after WS-02.4) `docker/migrate.Dockerfile`, WS-01.6 `docker/entrypoint.sh` and WS-01.5's image references `image_ref`/`COMPOSE_TAG`; WS-01.9 runs `clean_checkout.py --compose` against this task's compose sync at the W0 barrier (request R-01.1, landed here); WS-02 W0 phase A2 `scripts.db.migrate` (reads `DATABASE_URL`, `POSTGRES_SSL_MODE`, `POSTGRES_SSL_ROOT_CERT`), the init-db role script reading `SOLACE_APP_PASSWORD`/`SOLACE_MIGRATOR_PASSWORD`, and `scripts/db/gen_certs.py --out DIR` writing `postgres-ca.crt`, `server.crt` (SAN `DNS:postgres`) and `server.key` (all requested below).
**Files:**
- Modify: `docker-compose.yml` (postgres: `POSTGRES_USER: postgres`, `POSTGRES_DB: solace`, `POSTGRES_PASSWORD: ${POSTGRES_SUPERUSER_PASSWORD:-solace_superuser_dev_only}`, `SOLACE_APP_PASSWORD: ${POSTGRES_PASSWORD:-solace_app_dev_only}`, `SOLACE_MIGRATOR_PASSWORD: ${POSTGRES_MIGRATOR_PASSWORD:-solace_migrator_dev_only}`, healthcheck `pg_isready -U postgres -d solace`; new `migrate` service (`build: {context: ., dockerfile: docker/migrate.Dockerfile}`, `image: solace/migrate:${SOLACE_IMAGE_TAG:-dev}`, command `["python", "-m", "scripts.db.migrate", "upgrade", "head"]`, `DATABASE_URL: postgresql+asyncpg://solace_migrator:${POSTGRES_MIGRATOR_PASSWORD:-solace_migrator_dev_only}@postgres:5432/solace`, `restart: "no"`, network `data`, depends_on postgres healthy); all nine app services keep `build: {context: ., dockerfile: <their Dockerfile>}` and gain `image: solace/<service>:${SOLACE_IMAGE_TAG:-dev}` (the compose service name equals the WS-01 `IMAGES` name and the value is WS-01.5's `image_ref(<service>, COMPOSE_TAG)`, so `docker compose build` tags exactly what WS-01's image builder builds; this task edits no WS-01 file; `SOLACE_IMAGE_TAG` is a compose interpolation variable only, seen by no container, so it gets no registry row); all nine app services get `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`, `POSTGRES_DATABASE=solace`, `POSTGRES_USER=solace_app`, `POSTGRES_PASSWORD`, `POSTGRES_SSL_MODE: ${POSTGRES_SSL_MODE:-prefer}`, `POSTGRES_MIN_POOL_SIZE=1`, `POSTGRES_MAX_POOL_SIZE=8`, the legacy aliases with identical values (`USER_DB_HOST/PORT/NAME/USER/PASSWORD`, `DIAGNOSIS_DB_*`, `THERAPY_DB_HOST/PORT/DATABASE/USER/PASSWORD`, `MEMORY_DB_HOST/PORT/DATABASE/USER/PASSWORD`, `PERSONALITY_DB_*`, `SAFETY_REPO_STORAGE_BACKEND=postgres`, `SAFETY_REPO_POSTGRES_DSN=postgresql+asyncpg://solace_app:<same password>@postgres:5432/solace`), `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`, `EVENTS_ENABLED=true`, the legacy Kafka names (`SAFETY_KAFKA_ENABLED=true`, `SAFETY_KAFKA_BOOTSTRAP_SERVERS`, `SAFETY_EVENTS_KAFKA_BOOTSTRAP_SERVERS`, `THERAPY_KAFKA_ENABLED=true`, `THERAPY_KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_ENABLED=true` on notification only, `USER_KAFKA_ENABLE=true`, `USER_KAFKA_BOOTSTRAP_SERVERS`, `MEMORY_KAFKA_BOOTSTRAP_SERVERS`, `CONSUMER_KAFKA_ENABLED=true`, `CONSUMER_KAFKA_BOOTSTRAP_SERVERS`), the seven `SERVICE_*_SERVICE_URL` peers, the legacy peer names read today (`ESCALATION_NOTIFICATION_SERVICE_URL`, `CLINICIAN_REGISTRY_USER_SERVICE_URL`, `NOTIFICATION_USER_SERVICE_URL`, `THERAPY_CONTEXT_MEMORY_SERVICE_URL`, `USER_SERVICE_THERAPY_SERVICE_URL`, `USER_SERVICE_NOTIFICATION_SERVICE_URL`, `ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL`, `ORCHESTRATOR_THERAPY_AGENT_SERVICE_URL`, `ORCHESTRATOR_DIAGNOSIS_AGENT_SERVICE_URL`, `ORCHESTRATOR_PERSONALITY_AGENT_SERVICE_URL`, `ORCHESTRATOR_ASSESSMENT_AGENT_SERVICE_URL`, `SERVICE_TREATMENT_SERVICE_URL`), `depends_on: {migrate: {condition: service_completed_successfully}, kafka: {condition: service_healthy}}`, and healthchecks `["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:<port>/live', timeout=3)"]` (interval 15s, timeout 5s, retries 3, start_period 60s); orchestrator networks `backend`, `frontend`, `data`; the orchestrator's peer dependencies become `condition: service_started` with `required: false`)
- Modify: `docker-compose.prod.yml` (postgres TLS: an entrypoint wrapper copies `/run/secrets/server.key` and `server.crt` with owner `postgres` and mode 0600 into `/var/lib/postgresql/tls` and execs `docker-entrypoint.sh`, command adds `-c ssl=on -c ssl_cert_file=... -c ssl_key_file=...`; top-level file secrets `postgres-ca.crt`, `server.crt`, `server.key` from `${SOLACE_SECRETS_DIR:?...}`; every DB client and `migrate` gets `POSTGRES_SSL_MODE: verify-full`, `POSTGRES_SSL_ROOT_CERT: /run/secrets/postgres-ca.crt` and the `postgres-ca.crt` secret; `POSTGRES_PASSWORD`, `POSTGRES_MIGRATOR_PASSWORD`, `POSTGRES_SUPERUSER_PASSWORD` are `:?`)
- Modify: `deploy/env/env-contract.yaml` (the rows above, including `SOLACE_SECRETS_DIR` as a non-secret strict-required source), `.env.example` (RENDER)
- Modify: `tests/deploy/conftest.py` (add the `compose_project` fixture: unique project `solace-deploytest-<8 hex>`, `--project-directory D:\Repo\Contextual-Chatbot`, a temp `--env-file` from `write_synthetic_env(..., profile="dev", environment="development")`, helpers `up(services, *, profiles=(), wait=True)`, `run(service) -> CompletedProcess`, `exec(service, *cmd) -> str`, `psql(sql) -> list[tuple]` as the image superuser, teardown `down -v --remove-orphans`)
- Test: `tests/deploy/test_mvp_003_data_plane.py`, `tests/deploy/test_mvp_116_dev_db_password.py`
**Interfaces:**
- Consumes: `render`, `registry` fixtures and `write_synthetic_env` (WS-03.1); WS-02's migrate entrypoint, init-db variables and certificate names; WS-01.9's migrate image (`IMAGES` name `migrate`); WS-01.5's `scripts.ci.build_images.image_ref(name, tag)`, `COMPOSE_TAG` and `scripts.ci.images.IMAGES` (the reference format is WS-01's; this task only writes the compose values); WS-01.6's entrypoint.
- Produces: compose service `migrate`; the postgres-container init variables `SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD` (New interfaces); app healthchecks on `/live` (IF-18); `SOLACE_SECRETS_DIR` (host directory holding one file per secret, mounted per service under `/run/secrets/`); the W0 compose sync of request R-01.1 for WS-01.9 (`clean_checkout.py --compose`): every built service keeps `build: {context: ., ...}` and is tagged `image: solace/<name>:${SOLACE_IMAGE_TAG:-dev}` (the nine app services and `migrate`), the `migrate` one-shot is built from `docker/migrate.Dockerfile`, app healthchecks call `/live`, and config-service is gone (WS-03.1).

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_003_data_plane.py` and `tests/deploy/test_mvp_116_dev_db_password.py`
```python
# tests/deploy/test_mvp_003_data_plane.py
"""MVP-003: the orchestrator (and every app service) reaches Postgres, Kafka and its peers; app services start only after migrations."""
import pytest
from deploy.env.registry import APP_SERVICES

PEERS = {"SERVICE_SAFETY_SERVICE_URL": "http://safety-service:8002", "SERVICE_NOTIFICATION_SERVICE_URL": "http://notification-service:8003",
         "SERVICE_DIAGNOSIS_SERVICE_URL": "http://diagnosis-service:8004", "SERVICE_MEMORY_SERVICE_URL": "http://memory-service:8005",
         "SERVICE_THERAPY_SERVICE_URL": "http://therapy-service:8006", "SERVICE_PERSONALITY_SERVICE_URL": "http://personality-service:8007",
         "SERVICE_USER_SERVICE_URL": "http://user-service:8001"}

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_orchestrator_joins_the_data_network(render, profile):
    assert "data" in render(profile).services["orchestrator-service"]["networks"]

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_app_service_gets_the_data_plane(render, profile):
    r = render(profile)
    for svc in APP_SERVICES:
        env = r.services[svc]["environment"]
        assert (env["POSTGRES_HOST"], env["POSTGRES_USER"], env["POSTGRES_DATABASE"]) == ("postgres", "solace_app", "solace"), svc
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:9092" and env["EVENTS_ENABLED"] == "true", svc
        assert {k: env.get(k) for k in PEERS} == PEERS, svc
        assert r.services[svc]["depends_on"]["migrate"]["condition"] == "service_completed_successfully", svc
        assert "/live" in " ".join(r.services[svc]["healthcheck"]["test"]), svc

def test_strict_db_clients_verify_tls(render):
    r = render("strict")
    for svc in (*APP_SERVICES, "migrate"):
        env = r.services[svc]["environment"]
        assert env["POSTGRES_SSL_MODE"] == "verify-full", svc
        assert env["POSTGRES_SSL_ROOT_CERT"] == "/run/secrets/postgres-ca.crt", svc
        assert "postgres-ca.crt" in {s["source"] for s in r.services[svc]["secrets"]}, svc
    assert "ssl=on" in " ".join(r.services["postgres"]["command"])

def test_migrate_runs_as_solace_migrator(render):
    m = render("dev").services["migrate"]
    assert m["environment"]["DATABASE_URL"].startswith("postgresql+asyncpg://solace_migrator:")
    assert m["command"] == ["python", "-m", "scripts.db.migrate", "upgrade", "head"] and m["restart"] == "no"

def test_built_services_are_tagged_for_ci():
    # raw YAML, not the render: WS-01's image gate (WS-01.9 --compose, WS-01.10) reads the files as written
    import yaml
    from scripts.ci.build_images import COMPOSE_TAG, image_ref   # WS-01.5: the reference format is WS-01's
    from scripts.ci.images import IMAGES
    from tests.deploy.conftest import REPO
    services = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    built = {name for name, spec in services.items() if "build" in spec}
    assert built == {*APP_SERVICES, "migrate"} == {i.name for i in IMAGES}
    for name in built:
        assert services[name]["build"]["context"] == ".", name
        assert services[name]["image"] == image_ref(name, COMPOSE_TAG), name
    assert services["migrate"]["build"]["dockerfile"] == "docker/migrate.Dockerfile"

def test_app_password_is_not_the_superuser_password(render):
    r = render("strict")
    su = r.services["postgres"]["environment"]["POSTGRES_PASSWORD"]
    assert su == r.env["POSTGRES_SUPERUSER_PASSWORD"]
    assert all(r.services[s]["environment"]["POSTGRES_PASSWORD"] != su for s in APP_SERVICES)

@pytest.mark.integration
@pytest.mark.e2e
def test_migrate_creates_head_as_solace_migrator(compose_project):
    compose_project.up(["postgres"])
    assert compose_project.run("migrate").returncode == 0
    roles = compose_project.psql("SELECT rolname, rolsuper, rolbypassrls FROM pg_roles "
                                 "WHERE rolname IN ('solace_app','solace_migrator') ORDER BY 1")
    assert roles == [("solace_app", False, False), ("solace_migrator", False, False)]
    assert compose_project.psql("SELECT count(*) FROM alembic_version") == [(1,)]
```
```python
# tests/deploy/test_mvp_116_dev_db_password.py
"""MVP-116: the dev database password passes user-service's unsafe-secret check, so user-service boots in dev."""
import importlib.util
from tests.deploy.conftest import REPO

def _user_service_unsafe_check():
    spec = importlib.util.spec_from_file_location("user_service_config_probe", REPO / "services/user-service/src/config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._is_unsafe_secret

def test_dev_db_password_passes_user_service_validation(render):
    env = render("dev", environment="development").services["user-service"]["environment"]
    pw = env["USER_DB_PASSWORD"]
    assert pw == env["POSTGRES_PASSWORD"]
    assert "password" not in pw.lower()
    assert _user_service_unsafe_check()(pw) is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_003_data_plane.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_116_dev_db_password.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_003_data_plane.py`
Expected: FAIL. `test_orchestrator_joins_the_data_network` fails (networks are `backend`, `frontend` only, docker-compose.yml:474-476, MVP-003); the data-plane test fails with `KeyError: 'POSTGRES_HOST'` on orchestrator-service; `test_migrate_runs_as_solace_migrator` raises `KeyError: 'migrate'`; `test_built_services_are_tagged_for_ci` fails because no `migrate` service is built and the app services have `build:` but no `image:` (docker-compose.yml:200-203 and the eight like it); `test_dev_db_password_passes_user_service_validation` fails because `USER_DB_PASSWORD` is `solace_dev_password` and `_is_unsafe_secret` returns True (MVP-116); the docker test fails because there is no `migrate` service.
- [ ] **Step 3: Implement** the compose and registry edits listed under Files. Approach: the dev `POSTGRES_PASSWORD` default `solace_app_dev_only` contains none of user-service's unsafe substrings; the legacy aliases are sourced from the same interpolation keys so they can never diverge; Kafka enable flags stay `true` until WS-06/W1 retires them; peers are plain service-DNS URLs on every app container. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the three commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.3: Redis authentication everywhere, strict host-port lockdown and required Grafana credentials
**Covers:** MVP-118, MVP-040
**Severity:** high
**Wave:** 0
**Depends on:** WS-03.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes (redis container: healthcheck needs auth)
**Cross-workstream prerequisites:** none (WS-07 reads `REDIS_URL` for the blacklist and rate limiter from W1; legacy names keep current readers working).
**Files:**
- Modify: `docker-compose.yml` (redis: `REDIS_PASSWORD: ${REDIS_PASSWORD:-solace_redis_dev_only}`; command `["sh", "-c", "umask 077; printf 'appendonly yes\\nrequirepass %s\\n' \"$$REDIS_PASSWORD\" > /tmp/redis.conf; exec redis-server /tmp/redis.conf"]` so requirepass is always on and the password is not in argv; healthcheck `["CMD-SHELL", "REDISCLI_AUTH=\"$$REDIS_PASSWORD\" redis-cli ping | grep -qx PONG"]`; all nine app services get `REDIS_URL: redis://:${REDIS_PASSWORD:-solace_redis_dev_only}@redis:6379/0` and the legacy names read today (`USER_REDIS_HOST/PORT/PASSWORD`, `DIAGNOSIS_REDIS_HOST/PORT/PASSWORD`, `DIAGNOSIS_SESSION_REDIS_HOST/PORT/PASSWORD`, `THERAPY_REDIS_HOST/PORT/PASSWORD`, `THERAPY_SESSION_REDIS_HOST/PORT/PASSWORD`, `PERSONALITY_REDIS_HOST/PORT/PASSWORD`, `MEMORY_REDIS_HOST/PORT/PASSWORD`, `REDIS_HOST/REDIS_PORT/REDIS_PASSWORD` on memory-service, `SAFETY_REPO_REDIS_URL`); grafana: `GF_SECURITY_ADMIN_USER: ${GRAFANA_ADMIN_USER:-admin}`, `GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-}`; every dev published port binds `127.0.0.1`)
- Modify: `docker-compose.prod.yml` (`ports: !reset []` on every service except `caddy`; `REDIS_PASSWORD` and `GRAFANA_ADMIN_PASSWORD` are `:?`; `REDIS_PASSWORD` documented as URL-safe)
- Modify: `deploy/env/env-contract.yaml` (rows; `GRAFANA_USER`/`GRAFANA_PASSWORD` interpolation names dropped), `.env.example` (RENDER)
- Test: `tests/deploy/test_mvp_118_redis_auth.py`, `tests/deploy/test_mvp_040_strict_exposure.py`
**Interfaces:**
- Consumes: `render`, `compose_project` (WS-03.1, WS-03.2).
- Produces: `REDIS_URL=redis://:<REDIS_PASSWORD>@redis:6379/0` on every app container (IF-02); the strict invariant "only caddy publishes host ports".

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_118_redis_auth.py` and `tests/deploy/test_mvp_040_strict_exposure.py`
```python
# tests/deploy/test_mvp_118_redis_auth.py
"""MVP-118: Redis always requires a password, every client gets it, and the healthcheck fails on NOAUTH."""
import pytest
from deploy.env.compose_render import ComposeRenderError
from deploy.env.registry import APP_SERVICES

LEGACY = {"user-service": {"USER_REDIS_HOST": "redis", "USER_REDIS_PASSWORD": None},
          "therapy-service": {"THERAPY_SESSION_REDIS_HOST": "redis", "THERAPY_SESSION_REDIS_PASSWORD": None},
          "diagnosis-service": {"DIAGNOSIS_SESSION_REDIS_HOST": "redis", "DIAGNOSIS_SESSION_REDIS_PASSWORD": None},
          "memory-service": {"MEMORY_REDIS_HOST": "redis", "MEMORY_REDIS_PASSWORD": None, "REDIS_PASSWORD": None}}

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_redis_requires_a_password(render, profile):
    redis = render(profile).services["redis"]
    assert "requirepass" in " ".join(redis["command"]) and redis["environment"]["REDIS_PASSWORD"]
    hc = " ".join(redis["healthcheck"]["test"])
    assert "PONG" in hc and "REDISCLI_AUTH" in hc

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_redis_client_gets_the_password(render, profile):
    r = render(profile)
    pw = r.services["redis"]["environment"]["REDIS_PASSWORD"]
    for svc in APP_SERVICES:
        assert r.services[svc]["environment"]["REDIS_URL"] == f"redis://:{pw}@redis:6379/0", svc
    for svc, expected in LEGACY.items():
        env = r.services[svc]["environment"]
        for name, value in expected.items():
            assert env[name] == (pw if value is None else value), f"{svc}:{name}"

def test_strict_render_requires_REDIS_PASSWORD(render):
    with pytest.raises(ComposeRenderError, match="REDIS_PASSWORD"):
        render("strict", drop=["REDIS_PASSWORD"])

@pytest.mark.integration
@pytest.mark.e2e
def test_healthcheck_needs_auth_and_noauth_is_rejected(compose_project):
    compose_project.up(["redis"])
    assert "NOAUTH" in compose_project.exec("redis", "redis-cli", "ping")
    assert compose_project.exec("redis", "sh", "-c", 'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli ping').strip() == "PONG"
```
```python
# tests/deploy/test_mvp_040_strict_exposure.py
"""MVP-040: in strict only Caddy publishes host ports; Grafana never boots on admin/admin; dev ports bind loopback."""
import pytest
from deploy.env.compose_render import ComposeRenderError

def test_only_caddy_publishes_host_ports_in_strict(render):
    published = {name for name, svc in render("strict").services.items() if svc.get("ports")}
    assert published == {"caddy"}

def test_grafana_admin_password_is_required_in_strict(render):
    with pytest.raises(ComposeRenderError, match="GRAFANA_ADMIN_PASSWORD"):
        render("strict", drop=["GRAFANA_ADMIN_PASSWORD"])
    r = render("strict")
    assert r.services["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"] == r.env["GRAFANA_ADMIN_PASSWORD"] != "admin"

def test_dev_ports_bind_loopback(render):
    for name, svc in render("dev").services.items():
        for port in svc.get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", (name, port)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_118_redis_auth.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_040_strict_exposure.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_118_redis_auth.py`
Expected: FAIL. The redis command has no requirepass unless `REDIS_PASSWORD` is set and the healthcheck is a bare `redis-cli ping` (docker-compose.yml:41,46-47, MVP-118); `KeyError: 'REDIS_URL'`; `USER_REDIS_PASSWORD` absent; strict publishes postgres, redis, kafka, zookeeper, weaviate, grafana, prometheus, jaeger and every app port because `ports: []` merges instead of resetting (MVP-040); grafana renders `admin`; the strict render succeeds without `REDIS_PASSWORD` and `GRAFANA_ADMIN_PASSWORD`; dev ports bind `0.0.0.0`.
- [ ] **Step 3: Implement** the edits listed under Files. Approach: the Redis password reaches every client from the single `REDIS_PASSWORD` source, so a rotation cannot leave one client behind; `ports: !reset []` (Compose 2.24+ syntax, available in Compose v5) on each internal service. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the three commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.4: Infra health and capacity: Weaviate healthcheck without curl behind an opt-in profile, Kafka heap inside its limit
**Covers:** MVP-002, MVP-045
**Severity:** blocker
**Wave:** 0
**Depends on:** WS-03.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-04 (default assumed: Weaviate writes off, `WEAVIATE_ENABLED=false`, retrieval stays keyword/BM25, so Weaviate is opt-in)
**Real infra:** yes (Weaviate becomes healthy with the new check)
**Cross-workstream prerequisites:** none (memory-service already degrades gracefully without Weaviate, `services/memory_service/src/main.py:137-144`).
**Files:**
- Modify: `docker-compose.yml` (weaviate: `profiles: ["vector"]`, healthcheck `["CMD-SHELL", "wget -q --spider http://127.0.0.1:8080/v1/.well-known/ready || exit 1"]` with `start_period: 60s`, `retries: 10`; memory-service no longer depends on weaviate and gets `WEAVIATE_ENABLED: ${WEAVIATE_ENABLED:-false}` and `MEMORY_EMBEDDINGS_PROVIDER: none`; kafka: `KAFKA_HEAP_OPTS: "-Xms256m -Xmx512m"`, healthcheck `["CMD-SHELL", "KAFKA_HEAP_OPTS='-Xms32m -Xmx96m' kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null"]`, interval 30s, timeout 20s, start_period 60s; zookeeper: `KAFKA_HEAP_OPTS: "-Xms128m -Xmx256m"`; `kafka-ui` pinned to a version tag instead of `latest`)
- Modify: `docker-compose.prod.yml` (kafka memory limit `1g`; zookeeper memory limit `512m`)
- Modify: `deploy/env/compose_render.py` (`profiles` passed as `--profile` flags), `deploy/env/env-contract.yaml`, `.env.example` (RENDER)
- Test: `tests/deploy/test_mvp_002_weaviate_health.py`, `tests/deploy/test_mvp_045_kafka_heap.py`
**Interfaces:**
- Consumes: `render(..., profiles=...)`, `compose_project.up(..., profiles=...)`.
- Produces: compose profile `vector` (Weaviate opt-in); `WEAVIATE_ENABLED` and `MEMORY_EMBEDDINGS_PROVIDER` registry rows (IF-02).

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_002_weaviate_health.py` and `tests/deploy/test_mvp_045_kafka_heap.py`
```python
# tests/deploy/test_mvp_002_weaviate_health.py
"""MVP-002: the Weaviate healthcheck cannot depend on curl (absent from the image), and nothing on the chat path waits for Weaviate."""
import pytest

def test_weaviate_is_opt_in(render):
    assert "weaviate" not in render("dev").services
    assert "weaviate" in render("dev", profiles=["vector"]).services

def test_weaviate_healthcheck_uses_wget(render):
    hc = render("dev", profiles=["vector"]).services["weaviate"]["healthcheck"]
    test = " ".join(hc["test"])
    assert "curl" not in test and "wget" in test and "/v1/.well-known/ready" in test
    assert hc.get("start_period")

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_memory_service_does_not_wait_for_weaviate(render, profile):
    assert "weaviate" not in (render(profile).services["memory-service"].get("depends_on") or {})

@pytest.mark.integration
@pytest.mark.e2e
def test_weaviate_becomes_healthy(compose_project):
    compose_project.up(["weaviate"], profiles=["vector"])  # up --wait fails if the healthcheck never passes
```
```python
# tests/deploy/test_mvp_045_kafka_heap.py
"""MVP-045: the broker heap plus JVM overhead fits its strict memory limit, and the healthcheck JVM is small."""
import re

MIB = 1024 * 1024

def _xmx_bytes(opts: str) -> int:
    value, unit = re.search(r"-Xmx(\d+)([mMgG])", opts).groups()
    return int(value) * (1024 * MIB if unit in "gG" else MIB)

def test_kafka_heap_fits_its_strict_limit(render):
    kafka = render("strict").services["kafka"]
    limit = int(kafka["deploy"]["resources"]["limits"]["memory"])
    assert _xmx_bytes(kafka["environment"]["KAFKA_HEAP_OPTS"]) + 384 * MIB <= limit

def test_kafka_healthcheck_runs_a_small_jvm(render):
    kafka = render("strict").services["kafka"]
    test = " ".join(kafka["healthcheck"]["test"])
    assert _xmx_bytes(test) <= 128 * MIB
    assert kafka["healthcheck"]["interval"] in ("30s", "1m0s")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_002_weaviate_health.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_045_kafka_heap.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_002_weaviate_health.py`
Expected: FAIL. Weaviate is always present and its healthcheck runs `curl` (docker-compose.yml:130-134, MVP-002); memory-service depends on `weaviate: service_healthy` (:363-369); `KeyError: 'KAFKA_HEAP_OPTS'` (the broker falls back to `-Xmx1G` inside a 768M limit, MVP-045); the docker test times out waiting for a healthy Weaviate.
- [ ] **Step 3: Implement** the edits listed under Files. Approach: the healthcheck JVM sets its own `KAFKA_HEAP_OPTS` inline so it never inherits the broker heap. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the three commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.5: Edge: Caddy in both profiles, tolerant /live health checks, env site addresses, blocked internal paths, intact BASIC_AUTH_HASH, CORS and proxy hops
**Covers:** MVP-117, MVP-225, MVP-250
**Severity:** medium
**Wave:** 0
**Depends on:** WS-03.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: the app-level Redis limiter is the control, keyed on the real client IP with `TRUSTED_PROXY_HOPS=1` in strict; stock Caddy, no third-party module)
**Real infra:** yes (`caddy adapt` in the pinned Caddy image)
**Cross-workstream prerequisites:** none for config; WS-05's `docker-compose.test.yml` overrides only `CADDY_API_SITE`, `CADDY_AUTH_SITE` and ports on this base-defined `caddy` service (requested below).
**Files:**
- Modify: `Caddyfile` (site blocks `{$CADDY_API_SITE}` and `{$CADDY_AUTH_SITE}` replace `api.{$DOMAIN}` / `auth.{$DOMAIN}`; a snippet `(never_routed)` responds 404 for `/metrics*`, `/docs*`, `/redoc*`, `/openapi.json`, `/status*`, `/api/v1/internal/*` and is imported first in both sites inside a `route` block so order is literal; `reverse_proxy` to `orchestrator-service:8000` and `user-service:8001` with `health_uri /live`, `health_interval 10s`, `health_timeout 5s`, `health_fails 3`, `health_passes 1`, `lb_try_duration 10s`, `lb_try_interval 250ms`; the `kafka.{$DOMAIN}` vhost for the dev-only kafka-ui is removed; no `log` directive)
- Modify: `docker-compose.yml` (the `caddy` service moves here from the prod overlay: image `caddy:2.8-alpine`, env `DOMAIN: ${DOMAIN:-localhost}`, `CADDY_ADMIN_EMAIL: ${CADDY_ADMIN_EMAIL:-dev-ops@example.invalid}`, `BASIC_AUTH_HASH` with a dev default that is the bcrypt hash of a discarded random password (`$` escaped as `$$`), `CADDY_API_SITE: ${CADDY_API_SITE:-http://api.localhost:8088}`, `CADDY_AUTH_SITE: ${CADDY_AUTH_SITE:-http://auth.localhost:8088}`, ports `127.0.0.1:8088:8088`, volumes Caddyfile/`caddy_data`/`caddy_config`, networks frontend and backend, depends_on orchestrator and user-service started; orchestrator and user-service get `CORS_ALLOWED_ORIGINS: ${CORS_ALLOWED_ORIGINS:-http://localhost:3000}` and `TRUSTED_PROXY_HOPS: "0"`; orchestrator keeps the legacy `ORCHESTRATOR_CORS_ORIGINS` and `ORCHESTRATOR_TRUSTED_PROXY_HOPS` with the same values)
- Modify: `docker-compose.prod.yml` (caddy: `CADDY_API_SITE: api.${DOMAIN:?...}`, `CADDY_AUTH_SITE: auth.${DOMAIN:?...}`, `BASIC_AUTH_HASH: ${BASIC_AUTH_HASH:?...}`, ports `80:80`, `443:443`, `443:443/udp` replacing the dev port via `!reset`; orchestrator and user-service `CORS_ALLOWED_ORIGINS: ${CORS_ALLOWED_ORIGINS:?...}`, `TRUSTED_PROXY_HOPS: "1"`, `ORCHESTRATOR_TRUSTED_PROXY_HOPS: "1"`)
- Modify: `deploy/env/env-contract.yaml` (rows; `BASIC_AUTH_HASH` format `bcrypt`), `.env.example` (RENDER: `BASIC_AUTH_HASH=''` with the single-quote instruction), `docs/STAGING-BRINGUP.md` (the `BASIC_AUTH_HASH` line shows single quotes and why; `CORS_ALLOWED_ORIGINS` added to the required list)
- Modify: `tests/deploy/conftest.py` (add `caddy_adapt(env: dict) -> dict` running `docker run --rm` with the given `-e` values and `-v <repo>/Caddyfile:/etc/caddy/Caddyfile:ro` on `caddy:2.8-alpine caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile`, returning parsed JSON)
- Test: `tests/deploy/test_mvp_250_caddy_edge.py`, `tests/deploy/test_mvp_117_basic_auth_hash.py`, `tests/deploy/test_mvp_225_cors_origins.py`
**Interfaces:**
- Consumes: IF-18 (`/live` never depends on downstream services); `render`, `caddy_adapt`.
- Produces: `caddy` defined in `docker-compose.yml` for both profiles with the dev edge on `127.0.0.1:8088` (New interfaces); `CADDY_API_SITE`, `CADDY_AUTH_SITE`, `CORS_ALLOWED_ORIGINS`, `TRUSTED_PROXY_HOPS` registry rows (IF-02).

- [ ] **Step 1: Write the failing test** in the three files
```python
# tests/deploy/test_mvp_250_caddy_edge.py
"""MVP-250 (+IF-18 edge rules): health checks probe /live with failure tolerance and retry windows;
internal paths are never routed; site addresses come from env; Caddy holds only its own variables."""
import json, re
import pytest
from tests.deploy.conftest import REPO

CADDYFILE = (REPO / "Caddyfile").read_text(encoding="utf-8")
NEVER = ("/metrics", "/docs", "/redoc", "/openapi.json", "/status", "/api/v1/internal/*")

def _proxy_block(upstream: str) -> str:
    m = re.search(rf"reverse_proxy\s+{re.escape(upstream)}\s*\{{(.*?)\n\s*\}}", CADDYFILE, re.S)
    assert m, upstream
    return m.group(1)

@pytest.mark.parametrize("upstream", ["orchestrator-service:8000", "user-service:8001"])
def test_health_checks_probe_live_and_tolerate_one_slow_probe(upstream):
    block = _proxy_block(upstream)
    assert "health_uri /live" in block and "/health" not in block
    assert int(re.search(r"health_fails\s+(\d+)", block).group(1)) >= 3
    assert "lb_try_duration" in block

def test_site_addresses_come_from_env():
    assert "{$CADDY_API_SITE}" in CADDYFILE and "{$CADDY_AUTH_SITE}" in CADDYFILE
    assert "api.{$DOMAIN}" not in CADDYFILE and "kafka.{$DOMAIN}" not in CADDYFILE

def test_internal_paths_are_blocked_and_no_access_log():
    snippet = re.search(r"\(never_routed\)\s*\{(.*?)\n\}", CADDYFILE, re.S).group(1)
    for path in NEVER:
        assert path in snippet, path
    assert not re.search(r"^\s*log\b", CADDYFILE, re.M)

def test_caddy_runs_in_both_profiles_with_minimal_env(render):
    dev, strict = render("dev"), render("strict")
    assert "caddy" in dev.services and all(p["host_ip"] == "127.0.0.1" for p in dev.services["caddy"]["ports"])
    assert set(strict.services["caddy"]["environment"]) <= {"DOMAIN", "CADDY_ADMIN_EMAIL", "BASIC_AUTH_HASH", "CADDY_API_SITE", "CADDY_AUTH_SITE"}
    assert strict.services["caddy"]["environment"]["CADDY_API_SITE"] == f"api.{strict.env['DOMAIN']}"

@pytest.mark.integration
@pytest.mark.realinfra
def test_adapted_config_blocks_metrics_before_proxying(caddy_adapt):
    cfg = caddy_adapt({"CADDY_API_SITE": "http://api.example.test", "CADDY_AUTH_SITE": "http://auth.example.test",
                       "DOMAIN": "example.test", "CADDY_ADMIN_EMAIL": "ops@example.test",
                       "BASIC_AUTH_HASH": "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"})
    text = json.dumps(cfg)
    assert '"uri": "/live"' in text and '"try_duration"' in text
    assert text.index('"/metrics"') < text.index('"orchestrator-service:8000"')
```
```python
# tests/deploy/test_mvp_117_basic_auth_hash.py
"""MVP-117: a bcrypt hash survives compose interpolation intact when single-quoted, and the docs and .env.example say so."""
from tests.deploy.conftest import REPO

HASH = "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"

def test_single_quoted_hash_reaches_caddy_intact(render):
    r = render("strict", overrides={"BASIC_AUTH_HASH": HASH})
    assert r.services["caddy"]["environment"]["BASIC_AUTH_HASH"] == HASH

def test_env_example_single_quotes_the_hash():
    line = next(l for l in (REPO / ".env.example").read_text(encoding="utf-8").splitlines() if l.startswith("BASIC_AUTH_HASH="))
    assert line == "BASIC_AUTH_HASH=''"

def test_runbook_tells_operators_to_single_quote():
    assert "BASIC_AUTH_HASH='" in (REPO / "docs/STAGING-BRINGUP.md").read_text(encoding="utf-8")
```
```python
# tests/deploy/test_mvp_225_cors_origins.py
"""MVP-225: strict deploys must set CORS_ALLOWED_ORIGINS for the two edge services; both trust exactly one proxy hop."""
import pytest
from deploy.env.compose_render import ComposeRenderError

@pytest.mark.parametrize("service", ["user-service", "orchestrator-service"])
def test_cors_origins_reach_edge_services(render, service):
    r = render("strict", overrides={"CORS_ALLOWED_ORIGINS": "https://app.example.test"})
    env = r.services[service]["environment"]
    assert env["CORS_ALLOWED_ORIGINS"] == "https://app.example.test"
    assert env["TRUSTED_PROXY_HOPS"] == "1"

def test_strict_render_requires_cors_origins(render):
    with pytest.raises(ComposeRenderError, match="CORS_ALLOWED_ORIGINS"):
        render("strict", drop=["CORS_ALLOWED_ORIGINS"])
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_250_caddy_edge.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_117_basic_auth_hash.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_225_cors_origins.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_250_caddy_edge.py`
Expected: FAIL. Health checks use `health_uri /health`, no `health_fails` and no `lb_try_duration` (Caddyfile:36-40, 60-63, MVP-250); site addresses are `api.{$DOMAIN}`; no `(never_routed)` snippet; caddy is absent from the dev render; `.env.example` has no `BASIC_AUTH_HASH` line and STAGING-BRINGUP.md:71 shows it unquoted (MVP-117); user-service has no `CORS_ALLOWED_ORIGINS` or `TRUSTED_PROXY_HOPS` (MVP-225).
- [ ] **Step 3: Implement** the Caddyfile, compose, registry and doc edits listed under Files. Approach: `route { import never_routed; reverse_proxy ... }` keeps the 404 ahead of the proxy without relying on directive sorting; the strict sites are plain host names so automatic HTTPS applies, the dev and test sites are explicit `http://` addresses. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the four commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.6: Notification delivery env contract, the crisis acknowledgement route and the W1/W2 env sync
**Covers:** MVP-042
**Severity:** high
**Wave:** 1
**Depends on:** WS-03.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-12 (accepted 2026-10-19: Amazon SES in the pilot Region through its SMTP interface with STARTTLS, `EMAIL_SMTP_TLS_MODE=starttls`; SMS optional for CRITICAL; push out of scope; from W3 the SES SMTP username and password are file secrets from the environment's SOPS file (WS-03.8, D-07), and verifying the SES domain identity is an owner action (WS-03.20, D-25)), D-28 (default assumed: notification-service is the only pager and owns the acknowledgement link)
**Real infra:** yes (`caddy adapt` for the acknowledgement route; BOOTSMOKE)
**Cross-workstream prerequisites:** WS-13 reads `EMAIL_SMTP_TLS_MODE`, `NOTIFICATION_ACK_BASE_URL` and the coalescing/re-page settings in W1 phase B (IF-09); WS-04.7 consumes the Alertmanager rows and the Prometheus mount. This task also lands every env, route and service request that other plans address to WS-03 for wave 1 or wave 2 (WS-03 has no W2 wave).
**Files:**
- Modify: `docker-compose.yml` (notification-service: `EMAIL_ENABLED: ${EMAIL_ENABLED:-true}`, `EMAIL_SMTP_HOST: ${EMAIL_SMTP_HOST:-mailhog}`, `EMAIL_SMTP_PORT: ${EMAIL_SMTP_PORT:-1025}`, `EMAIL_SMTP_USERNAME`, `EMAIL_SMTP_PASSWORD`, `EMAIL_SMTP_TLS_MODE: ${EMAIL_SMTP_TLS_MODE:-none}`, `EMAIL_FROM_EMAIL: ${EMAIL_FROM_EMAIL:-alerts@localhost.invalid}`, `EMAIL_FROM_NAME: ${EMAIL_FROM_NAME:-Solace-AI}`, `NOTIFICATION_FALLBACK_ONCALL_EMAIL: ${NOTIFICATION_FALLBACK_ONCALL_EMAIL:-fallback-oncall@localhost.invalid}`, `NOTIFICATION_ALERT_COALESCE_SECONDS: "900"`, `NOTIFICATION_REPAGE_AFTER_SECONDS: "900"`, `NOTIFICATION_MAX_REPAGES: "3"`, `NOTIFICATION_ACK_BASE_URL: ${NOTIFICATION_ACK_BASE_URL:-http://api.localhost:8088}`, `SMS_ENABLED: ${SMS_ENABLED:-false}` and the `SMS_*` names the code reads, `PUSH_ENABLED: "false"`; the misleading comment about setting `EMAIL_SMTP_HOST` through `.env.prod` is removed; prometheus mounts `./infrastructure/prometheus:/etc/prometheus:ro` instead of the single file)
- Modify: `docker-compose.prod.yml` (notification-service: `EMAIL_SMTP_HOST`, `EMAIL_FROM_EMAIL`, `EMAIL_SMTP_USERNAME`, `EMAIL_SMTP_PASSWORD`, `NOTIFICATION_FALLBACK_ONCALL_EMAIL` are `:?`; `EMAIL_SMTP_PORT: ${EMAIL_SMTP_PORT:-587}`; `EMAIL_SMTP_TLS_MODE: ${EMAIL_SMTP_TLS_MODE:-starttls}`; `NOTIFICATION_ACK_BASE_URL: https://api.${DOMAIN:?...}`)
- Modify: `Caddyfile` (inside the API site's `route`, after `never_routed`: `handle /api/v1/notifications/crisis-alerts/* { reverse_proxy notification-service:8003 { health_uri /live health_interval 10s health_fails 3 lb_try_duration 10s } }` ahead of the orchestrator proxy)
- Modify: `deploy/env/env-contract.yaml` (the notification rows, with the strict `example` for `EMAIL_SMTP_HOST` set to `email-smtp.us-east-1.amazonaws.com`, the SES SMTP endpoint of the pilot Region (D-12); `SMTP_HOST`, `SMTP_PORT`, `EMAIL_FROM`, `TWILIO_*`, `FCM_*` interpolation names dropped; the six `ALERTMANAGER_*` rows with services `[alertmanager]`, `ALERTMANAGER_SMTP_PASSWORD` secret, `ALERTMANAGER_EMAIL_TO` and `ALERTMANAGER_SMTP_SMARTHOST` required in strict, `ALERTMANAGER_SMTP_REQUIRE_TLS` strict default `true`), `.env.example` (RENDER)
- Modify: `tests/deploy/test_env_contract.py` (the strict renders also run with `fragments=["docker-compose.observability.yml"]` when that file exists)
- Test: `tests/deploy/test_mvp_042_notification_env.py`
**Interfaces:**
- Consumes: IF-09 (acknowledgement path, coalescing and re-page settings), IF-02 notification names.
- Produces: the IF-02 notification block and Caddy route; the Alertmanager registry rows consumed by WS-04.7.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_042_notification_env.py`
```python
"""MVP-042: strict notification-service gets a real SMTP relay with TLS and a fallback mailbox, never the dev mailhog;
flags are overridable; .env.example names what the code reads; clinicians' acknowledgement links route to notification."""
import json, re
import pytest
from deploy.env.compose_render import ComposeRenderError
from deploy.env.registry import APP_SERVICES
from tests.deploy.conftest import REPO

def _notif(r):
    return r.services["notification-service"]["environment"]

def test_strict_smtp_comes_from_the_operator(render):
    r = render("strict")
    env = _notif(r)
    assert env["EMAIL_SMTP_HOST"] == r.env["EMAIL_SMTP_HOST"] and env["EMAIL_SMTP_HOST"] != "mailhog"
    assert env["EMAIL_SMTP_TLS_MODE"] in ("starttls", "tls")
    assert env["NOTIFICATION_FALLBACK_ONCALL_EMAIL"] == r.env["NOTIFICATION_FALLBACK_ONCALL_EMAIL"]

@pytest.mark.parametrize("source", ["EMAIL_SMTP_HOST", "EMAIL_FROM_EMAIL", "NOTIFICATION_FALLBACK_ONCALL_EMAIL"])
def test_strict_render_requires_delivery_settings(render, source):
    with pytest.raises(ComposeRenderError, match=source):
        render("strict", drop=[source])

def test_sms_flag_is_overridable(render):
    assert _notif(render("strict", overrides={"SMS_ENABLED": "true"}))["SMS_ENABLED"] == "true"

def test_ack_links_point_at_the_api_site(render):
    r = render("strict")
    assert _notif(r)["NOTIFICATION_ACK_BASE_URL"] == f"https://api.{r.env['DOMAIN']}"

def test_events_enabled_on_every_app_service(render):
    r = render("strict")
    assert all(r.services[s]["environment"]["EVENTS_ENABLED"] == "true" for s in APP_SERVICES)

def test_env_example_names_what_the_code_reads():
    keys = {l.split("=", 1)[0] for l in (REPO / ".env.example").read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#")}
    assert {"EMAIL_SMTP_HOST", "EMAIL_SMTP_TLS_MODE", "EMAIL_FROM_EMAIL", "NOTIFICATION_FALLBACK_ONCALL_EMAIL"} <= keys
    assert not keys & {"SMTP_HOST", "SMTP_PORT", "EMAIL_FROM", "TWILIO_ACCOUNT_SID", "FCM_PROJECT_ID"}

def test_caddyfile_routes_crisis_acknowledgements_to_notification():
    text = (REPO / "Caddyfile").read_text(encoding="utf-8")
    block = re.search(r"handle /api/v1/notifications/crisis-alerts/\*\s*\{(.*?)\n\s*\}\s*\n", text, re.S).group(1)
    assert "reverse_proxy notification-service:8003" in block and "health_uri /live" in block

def test_prometheus_mounts_the_rules_directory(render):
    vols = render("dev").services["prometheus"]["volumes"]
    assert any(v["target"] == "/etc/prometheus" and v["source"].replace("\\", "/").endswith("infrastructure/prometheus") for v in vols)

@pytest.mark.integration
@pytest.mark.realinfra
def test_adapted_ack_route_precedes_the_orchestrator(caddy_adapt):
    text = json.dumps(caddy_adapt({"CADDY_API_SITE": "http://api.example.test", "CADDY_AUTH_SITE": "http://auth.example.test",
                                   "DOMAIN": "example.test", "CADDY_ADMIN_EMAIL": "ops@example.test",
                                   "BASIC_AUTH_HASH": "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"}))
    assert text.index('"/api/v1/notifications/crisis-alerts/*"') < text.index('"orchestrator-service:8000"')
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_042_notification_env.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_042_notification_env.py`
Expected: FAIL. Strict `EMAIL_SMTP_HOST` renders `mailhog` (`${SMTP_HOST:-mailhog}`, docker-compose.yml:285, MVP-042) and `KeyError: 'EMAIL_SMTP_TLS_MODE'`; the strict render succeeds without the delivery settings; `SMS_ENABLED` stays the literal `"false"`; `KeyError: 'NOTIFICATION_ACK_BASE_URL'`; `.env.example` lacks the honoured names; the Caddyfile has no acknowledgement route; prometheus mounts only the single file.
- [ ] **Step 3: Implement** the edits listed under Files. Then apply the requests other plans file for WS-03 for wave 1 or 2: read the "Cross-workstream requests" tables of every `docs/superpowers/plans/2026-10-18-mvp-remediation/WS-*.md` plan, take each row whose owner is WS-03 and whose wave is 1 or 2, and add the registry row, compose line or Caddy route it names (two exceptions: the strict `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"` line of WS-01's R-01.1 is not landed here, because WS-03.18 lands it in W5 once WS-06 creates the registry topics at plane start; and the D-04 LLM-provider rows of WS-09's CR-14 and CR-4 (`LLM_PROVIDER`, `BEDROCK_*` and the dev-only profiles of the `LLM_PORTKEY_*` and `LLM_ANTHROPIC_API_KEY` rows) are landed by WS-03.19, which depends on this task); the registry-driven contract tests cover them without new test code, and each landed row is recorded in the ledger. Mark `retired: true` any legacy name whose readers are gone (`python -m deploy.env.env_readers` lists readers per name). Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.7: File-delivered secrets in both profiles and per-developer generated dev values
**Covers:** MVP-349, MVP-348
**Severity:** high
**Wave:** 3
**Depends on:** WS-03.6, WS-03.19
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (changed 2026-10-19 to SOPS + age: strict secret files are rendered by WS-03.8 from the environment's SOPS file into root-only tmpfs files consumed through NAME_FILE, so the file contract this task builds is unchanged; dev and test values are generated per developer or per run; the committed defaults and `.env.prod` are retired; the keys in the working-tree `.env` are rotated without any value being printed)
**Real infra:** yes (postgres and redis boot from generated files; `docker inspect` shows no secret value)
**Cross-workstream prerequisites:** WS-01's `docker/entrypoint.sh` exports `NAME` from `NAME_FILE` for every app image and `.dockerignore` excludes `.secrets` (requested below); WS-02's `scripts/db/gen_certs.py --out DIR`; WS-05 builds the e2e stack's `SOLACE_SECRETS_DIR` with this generator from W3 (requested below).
**Files:**
- Create: `deploy/env/generate_dev_env.py`
- Create: `deploy/env/scan_env_file.py` (names-only scanner of an env file for non-empty secret values, with in-place blanking; used for the owner's rotation of the working-tree `.env` keys (D-07) and by WS-03.8 for the strict settings file)
- Modify: `deploy/env/registry.py` (`derive` templates; `file_secret_sources(registry) -> list[str]`; `secret_values_for_render`), `deploy/env/env-contract.yaml` (every secret row: `dev_default: null`, delivery `file`; derived rows `REDIS_URL = redis://:{REDIS_PASSWORD}@redis:6379/0`, `DATABASE_URL = postgresql+asyncpg://solace_migrator:{POSTGRES_MIGRATOR_PASSWORD}@postgres:5432/solace`, `SAFETY_REPO_POSTGRES_DSN` while it is still read; `SOLACE_SECRETS_DIR` dev default `./.secrets/dev`; `SOLACE_SECRETS_GID` dev default `1000`)
- Modify: `docker-compose.yml` (top-level `secrets:` with `file: ${SOLACE_SECRETS_DIR:-./.secrets/dev}/<SOURCE>` for every secret source; each service lists only its own secrets and sets `<NAME>_FILE: /run/secrets/<SOURCE>` for each container variable (legacy aliases point at the same file); no `${SECRET:-literal}` remains; postgres uses `POSTGRES_PASSWORD_FILE` and an entrypoint wrapper that, as root, exports `SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD` from their files before exec'ing `docker-entrypoint.sh`; redis builds its config from `/run/secrets/REDIS_PASSWORD`; grafana uses `GF_SECURITY_ADMIN_PASSWORD__FILE`; weaviate (profile `vector`) and caddy export their one secret from the file in a `sh -c` wrapper; every secret-consuming container gets `group_add: ["${SOLACE_SECRETS_GID:-1000}"]`)
- Modify: `docker-compose.prod.yml` (the same per-service secret mounts from `${SOLACE_SECRETS_DIR:?...}`; every `${SECRET:?}` environment line is replaced by its `_FILE` path; each service's `secrets` list is replaced with `!override`, so dev-only sources such as D-04's `LLM_PORTKEY_API_KEY` and `LLM_ANTHROPIC_API_KEY` (WS-03.19) are never mounted in strict)
- Modify: `deploy/env/compose_render.py` (writes placeholder files for every file secret under `<workdir>/secrets` and sets `SOLACE_SECRETS_DIR` to it unless the caller supplies values), `tests/deploy/conftest.py` (`render` can supply real secret values as files; `compose_project` generates its secrets dir with `generate_dev_env`), `.env.example` (RENDER)
- Test: `tests/deploy/test_mvp_349_dev_secrets.py`
**Interfaces:**
- Consumes: IF-22 secret names; WS-01 NAME_FILE loader; WS-02 certificate generator.
- Produces: `deploy.env.generate_dev_env.generate(out_dir: Path, *, registry=None, rotate: bool = False, with_tls: bool = True, mode: int = 0o640) -> list[str]` (returns the names written; raises `FileExistsError` when any target exists and `rotate` is False); CLI `python -m deploy.env.generate_dev_env [--out DIR] [--rotate] [--no-tls] [--mode 0644]` printing names only; per-service secret mounts at `/run/secrets/<SOURCE>` with `<NAME>_FILE` (IF-22); `SOLACE_SECRETS_GID` (New interfaces); `deploy.env.scan_env_file.secret_names_in_env_file(path: Path, registry=None) -> list[str]` and CLI `python -m deploy.env.scan_env_file PATH [--blank]` (names only; exit 1 when a secret value is present, 0 when none or after `--blank` has emptied them in place).

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_349_dev_secrets.py`
```python
"""MVP-349 (+ MVP-348 exposure): no working secret is committed in compose; every container receives only its own
secrets, as files; dev values are generated per developer and never printed."""
import re
import pytest
from deploy.env.registry import load_registry
from deploy.env.generate_dev_env import generate, main
from tests.deploy.conftest import REPO

COMMITTED = ["dev-jwt-secret-key-minimum-32-chars", "dev-encryption-key-32-characters",
             "8kTaL3nRf9xQpZmVwYdE7gHjKbNcSuW2A5iO0tP4rXs=", "3mBvN7xRtYfW0pLqKjH8dS5gC2eA9iZuO6wTrXnMaQs=",
             "solace-dev-key-change-in-production", "solace_app_dev_only", "solace_migrator_dev_only",
             "solace_superuser_dev_only", "solace_redis_dev_only"]

def test_no_secret_has_a_committed_default(registry):
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        text = (REPO / name).read_text(encoding="utf-8")
        for v in registry:
            if v.secret and v.source:
                assert not re.search(rf"\$\{{{v.source}:?-", text), (name, v.source)
        for literal in COMMITTED:
            assert literal not in text, (name, literal)

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_no_secret_value_in_any_container_environment(render, registry, profile):
    r = render(profile)
    secret_values = {r.env[v.source] for v in registry if v.secret and v.source and v.source in r.env}
    for name, svc in r.services.items():
        leaked = [k for k, val in (svc.get("environment") or {}).items() if val in secret_values]
        assert not leaked, (name, leaked)

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_each_container_mounts_exactly_its_secrets(render, registry, profile):
    r = render(profile)
    for v in registry:
        if not (v.secret and v.source) or v.retired or profile not in v.profiles:
            continue
        for svc in v.services:
            if svc in r.services:
                assert r.services[svc]["environment"][f"{v.name}_FILE"] == f"/run/secrets/{v.source}", (svc, v.name)
    expected = {}
    for v in registry:
        if v.secret and v.source and not v.retired and profile in v.profiles:
            for svc in v.services:
                expected.setdefault(svc, set()).add(v.source)
    for svc, svc_cfg in r.services.items():
        assert {s["source"] for s in svc_cfg.get("secrets", [])} - {"postgres-ca.crt", "server.crt", "server.key"} == expected.get(svc, set()), svc

def test_generator_writes_every_secret_with_valid_format(tmp_path):
    names = generate(tmp_path, with_tls=False)
    reg = load_registry()
    assert set(names) >= {v.source for v in reg if v.secret and v.source and not v.retired}
    first = {n: (tmp_path / n).read_text(encoding="utf-8") for n in names}
    assert all(first.values())
    other = tmp_path / "again"
    generate(other, with_tls=False)
    assert all((other / n).read_text(encoding="utf-8") != first[n] for n in names if not n.endswith(".json"))

def test_generator_refuses_to_overwrite(tmp_path):
    generate(tmp_path, with_tls=False)
    with pytest.raises(FileExistsError):
        generate(tmp_path, with_tls=False)
    before = (tmp_path / "AUTH_SECRET_KEY").read_text(encoding="utf-8")
    generate(tmp_path, with_tls=False, rotate=True)
    assert (tmp_path / "AUTH_SECRET_KEY").read_text(encoding="utf-8") != before

def test_cli_prints_names_only(tmp_path, capsys):
    assert main(["--out", str(tmp_path), "--no-tls"]) == 0
    out = capsys.readouterr().out
    for f in tmp_path.iterdir():
        if f.is_file():
            assert f.read_text(encoding="utf-8").strip() not in out

def test_env_file_scan_lists_names_only_and_blanks_in_place(tmp_path, capsys):
    from deploy.env.scan_env_file import main as scan, secret_names_in_env_file
    env = tmp_path / ".env"
    env.write_text("PORT=8000\nOPENAI_API_KEY=synthetic-111\nSECRET_KEY=synthetic-222\nEMPTY_TOKEN=\n", encoding="utf-8")
    assert secret_names_in_env_file(env) == ["OPENAI_API_KEY", "SECRET_KEY"]
    assert scan([str(env)]) == 1
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out and "synthetic-111" not in out and "synthetic-222" not in out
    assert scan([str(env), "--blank"]) == 0
    assert env.read_text(encoding="utf-8") == "PORT=8000\nOPENAI_API_KEY=\nSECRET_KEY=\nEMPTY_TOKEN=\n"
    assert scan([str(env)]) == 0

@pytest.mark.integration
@pytest.mark.e2e
def test_dev_stack_boots_from_generated_files(compose_project):
    compose_project.up(["postgres", "redis"])
    for svc in ("postgres", "redis"):
        env_dump = compose_project.inspect_env(svc)
        assert not any(value in env_dump for value in compose_project.secret_values())
```
(`compose_project.inspect_env(svc)` returns the `docker inspect --format '{{json .Config.Env}}'` text and `secret_values()` the generated values, both added to the fixture in this task.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_349_dev_secrets.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_349_dev_secrets.py`
Expected: FAIL with `ModuleNotFoundError: deploy.env.generate_dev_env`; with the import stubbed, `test_no_secret_has_a_committed_default` fails on `${AUTH_SECRET_KEY:-dev-jwt-secret-key-minimum-32-chars}` and the two literal Fernet keys (MVP-349), and `test_no_secret_value_in_any_container_environment` fails because every secret is an environment value visible to `docker inspect` (MVP-348); `test_env_file_scan_lists_names_only_and_blanks_in_place` fails with `ModuleNotFoundError: deploy.env.scan_env_file` (nothing can find the secret values in a developer's env file without printing them).
- [ ] **Step 3: Implement** `generate(...)`, `main(argv)` and the compose, registry and render-helper edits. Approach: values from `generate_value(format)`, real bcrypt for `BASIC_AUTH_HASH` with the plaintext written to `BASIC_AUTH_PASSWORD` (not mounted anywhere), derived files from `derive` templates, TLS files through `python -m scripts.db.gen_certs --out <dir>` when `with_tls`; files written atomically with `mode` (POSIX `os.chmod`; on Windows the directory ACL is restricted to the current user with `icacls` best effort); secret files are group-readable for `SOLACE_SECRETS_GID` so non-root container users (grafana uid 472, app uid 1000) can read them while root-started entrypoints (postgres, redis, caddy, weaviate) read them before dropping privileges. `scan_env_file` lists, by name only, every variable whose value is non-empty and whose name is a registry secret source, a legacy secret name, `SECRET_KEY` or ends in `_API_KEY`, `_SECRET`, `_SECRET_KEY`, `_TOKEN`, `_PASSWORD` or `_DSN`; it never prints, logs or returns a value, and `--blank` empties exactly those values with an atomic rewrite. Update `docs/runbooks/secrets.md` (dev section: run the generator once; rotate with `--rotate`; the working-tree `.env` procedure below). Owner action (D-07), recorded in the ledger by name and date only: the owner runs `python -m deploy.env.scan_env_file .env` in the repo root, revokes and reissues at each provider every key it lists, and then runs it again with `--blank`; reissued keys a developer still needs go into that developer's generated secrets directory, never back into `.env`. The executor never opens the working-tree `.env`. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE (dev and strict boot from files). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.8: Per-environment SOPS + age secrets rendered into root-only tmpfs files, rotation and re-key, and a token-free deploy
**Covers:** MVP-348, MVP-352
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-03.20, WS-01.14, WS-01.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (changed 2026-10-19: SOPS + age instead of AWS Secrets Manager. Each environment has `deploy/secrets/<env>.sops.yaml`, encrypted to that environment's age recipient(s) and safe to keep in the tree, and its own age key pair whose private key the owner or operator holds off-host (never in the repo, an image or a plan) and supplies at deploy time through `SOPS_AGE_KEY_FILE` at a root-only path. `deploy/render_secrets.py` decrypts with the pinned `sops` (and pinned `age`) into the root-only tmpfs `/run/secrets`, one 0400 file per secret, and fails closed on any decrypt error. The plaintext `.env.prod` and Caddy's env_file stay retired (WS-03.1). The NAME_FILE consumer contract of IF-02/IF-22 is unchanged. No Secrets Manager or Parameter Store call is planned), D-25 (changed 2026-10-19: the AWS staging host is the target; until the owner attaches the deployer policy to `MedCodex-Local` the render is proven only in the local pinned container)
**Real infra:** yes (the pinned sops/age round trip runs in the `solace/sops-tools` container; the unit tests use a fake `sops` and need no binaries)
**Cross-workstream prerequisites:** WS-03.20 (the root-only tmpfs at `/run/secrets` on the host and the `ArtifactBucketName` output); WS-01's `docker/entrypoint.sh` exports `NAME` from `NAME_FILE` (unchanged); WS-01's runtime lock carries hashed pins for the tools image, and its `.dockerignore` keeps `deploy/secrets/` out of every build context (requested below); WS-04.11 enforces NAME_FILE provenance in strict guards using these files.
**Files:**
- Create: `deploy/render_secrets.py`, `docs/runbooks/secrets.md` (strict section; dev section from WS-03.7 is extended)
- Create: `deploy/sops/tools.json` (the pinned `sops` and `age` release versions and the SHA-256 of each linux-amd64 release asset, copied from the projects' published checksum files; never guessed: if the checksum files cannot be read, stop and raise a ledger Ruling), `deploy/sops/install_tools.sh` (POSIX sh: downloads the pinned assets, checks them with `sha256sum -c` against `tools.json`, installs `sops`, `age` and `age-keygen` into /usr/local/bin, exits 1 on any mismatch), `deploy/sops/Dockerfile` (image `solace/sops-tools`: the same digest-pinned Python base as WS-01's app images, the two tools through `install_tools.sh`, and PyYAML and cryptography with their dependencies at the runtime lock's pins with `--require-hashes`; it copies no repo code, no SOPS file and no key; the repo is mounted read-only at run time)
- Modify: `deploy/env/registry.py` (`SECRET_READER_UIDS`: the nine app services and `migrate` 1000, `grafana` 472, `alertmanager` 65534, every other container 0; `ROOT_READERS_WITH_DAC_OVERRIDE`: `postgres`, `redis`, `pg-backup`, `pg-restore`, `backup`, `weaviate`; `class SecretOwnershipError(ValueError)` with `.names`; `secret_file_owner(registry, source) -> int`), `deploy/env/env-contract.yaml` (`EMAIL_SMTP_USERNAME` becomes a file secret beside `EMAIL_SMTP_PASSWORD`: together they are the SES SMTP credentials of D-12), `.env.example` (RENDER)
- Modify: `infrastructure/aws-staging/DEPLOY.md` (sections 5 and 6 rewritten. Code arrives only as a `git archive` tarball of the checkpoint tree id (D-17), uploaded with `aws s3 cp` to the `artifacts/` prefix of the `ArtifactBucketName` bucket and fetched on the host with the instance role, so no clone, token or `.git/config` reaches the host. Non-secret settings live in a root-only settings file under /etc/solace, written from the non-secret rows of the env example and checked with `python -m deploy.env.scan_env_file` (exit 1 if it holds a secret value). The age key: first confirm that Session Manager logs no session data (`aws ssm get-document --name SSM-SessionManagerRunShell`), then write the environment's private key inside the session to a root-only 0400 file under /run/solace-age, never through `ssm send-command` parameters, S3 or a file on disk. Build `solace/sops-tools` and render with `docker run --rm --network none --user 0`, the repo mounted read-only, the host `/run/secrets` mounted, the key file mounted read-only as `SOPS_AGE_KEY_FILE`, running `python -m deploy.render_secrets --env <env> --sops-file deploy/secrets/<env>.sops.yaml --out /run/secrets/<env>`; then `shred -u` the key file. `SOLACE_SECRETS_DIR=/run/secrets/<env>`. After a reboot or an `ec2:recover` the tmpfs is empty, the containers cannot start (fail closed) and the operator repeats the key and render steps. No `.env.prod`)
- Modify: `docs/STAGING-BRINGUP.md` (section 1 replaced: secrets come from the environment's SOPS file through `render_secrets`, never python one-liners into `.env.prod`), `docs/KNOWN-LIMITATIONS.md` (3.2 rewritten: SOPS + age per environment with off-host keys, rotation and re-key, and the residual risks: a host reboot needs the operator to re-supply the key, IMDSv2 hop limit 2 lets every container reach the instance role, and plaintext exists on the operator workstation while `sops edit` runs)
- Modify: `tests/security/test_config_secrets_hardening.py` (`test_p1_8_audit_hmac_key_guarded_in_prod` and `test_p2_16_weaviate_api_key_guarded_in_prod` assert the `_FILE` form and the mounted secret instead of `${...:?`)
- Modify: `tests/deploy/conftest.py` (add the `sops_tools` fixture: builds the tools image from the deploy/sops Dockerfile once per session as `solace/sops-tools:test`, then `sops_tools(script, *, work)` runs `sh -c script` in it with `--rm --network none --user 0`, the repo read-only at `/src` (working directory, `PYTHONPATH=/src`), `work` at `/work` and `--tmpfs /run/secrets:rw,mode=0700`, and returns the CompletedProcess)
- Create: `tests/deploy/test_mvp_348_secrets_store.py` (WS-03.10 extends it in W4), `tests/deploy/test_mvp_352_deploy_runbook.py`
- Test: `tests/deploy/test_mvp_348_secrets_store.py`, `tests/deploy/test_mvp_352_deploy_runbook.py`
**Interfaces:**
- Consumes: the registry (`file_secret_sources`, `derive`, `SECRET_READER_UIDS`); WS-03.20's host tmpfs and `ArtifactBucketName`; WS-01.14's ignore rules and `.dockerignore` exclusion of `deploy/secrets/` and age keys; WS-01.2's hashed runtime-lock pins of PyYAML, cryptography, cffi and pycparser for the tools image; the sops CLI (`sops --version`, `sops --decrypt --output-type json FILE` with `SOPS_AGE_KEY_FILE`); `deploy.env.scan_env_file` (WS-03.7).
- Produces: `deploy.render_secrets`: `class SecretSource(Protocol)` with `get(name: str) -> str | None`; `SopsFileSource(path: Path, *, sops_cmd: Sequence[str] = ("sops",), age_key_file: Path | None = None, tools: Path = TOOLS_PATH)` (`age_key_file` defaults to `SOPS_AGE_KEY_FILE`; checks the pinned sops version, the key file's presence and, on POSIX, that it has no group or world bits; decrypts once; every failure raises `SecretDecryptError`, whose message never carries sops output); `DictSource(mapping)`; `class SecretMissingError(RuntimeError)` with `.names`; `class SecretDecryptError(RuntimeError)`; `class SecretRenderError(RuntimeError)`; `render(environment: str, out_dir: Path, *, source: SecretSource, registry=None, apply_ownership: bool | None = None, require_tmpfs: bool | None = None) -> RenderReport` (both flags default to "running as root on Linux"; refuses non-strict environments and, when required, a target that is not on tmpfs; all-or-nothing, and a failed re-render leaves the previous directory untouched; directory 0700, each file 0400 owned by `secret_file_owner(registry, source)` when ownership applies; never prints or returns values); `@dataclass RenderReport(names: list[str], derived: list[str], out_dir: Path, owners: dict[str, int])`; `check_encrypted(path: Path, registry=None) -> list[str]` (no decryption: reports a missing registry secret source, a value not of the form `ENC[...]` and a file with no `sops.age` recipient, by name only); CLI `python -m deploy.render_secrets --env {staging,production} --sops-file F --out DIR` and `python -m deploy.render_secrets --check --env E --sops-file F` (names and counts only; exit 1 on any problem); `deploy.env.registry.secret_file_owner` and `SecretOwnershipError`; the SOPS layout (New interfaces); image `solace/sops-tools` built from `deploy/sops/Dockerfile`. For WS-01's R-01.18 (WS-01.14's ignore rules and checkpoint-tree secret scan): `deploy/secrets/` holds only `<env>.sops.yaml`, `.sops.yaml`, `escrow-recipients.txt` (public recipients only), an optional `README.md` and `history/<env>/<UTC timestamp>.sops.yaml`; every `<env>.sops.yaml` is fully encrypted (no `unencrypted_suffix`, `unencrypted_regex` or `encrypted_regex`; any value such a rule leaves in plaintext is reported by `check_encrypted` as not of the form `ENC[...]`); `render_secrets` writes only under `--out` on tmpfs, never into the tree; tests keep throwaway age keys and synthetic secrets under `tmp_path`, and no `*.agekey` or `keys.txt` file is planned in the tree. For WS-04's W3 row (WS-04.11): `SECRET_READER_UIDS["alertmanager"] == 65534`, so its 0400 `ALERTMANAGER_SMTP_PASSWORD` file is readable by the image user.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_348_secrets_store.py` and `tests/deploy/test_mvp_352_deploy_runbook.py`
```python
# tests/deploy/test_mvp_348_secrets_store.py
"""MVP-348 (D-07: SOPS + age): strict secrets come from the environment's SOPS file into one root-only 0400 file per
secret on a tmpfs, owned by the uid that reads it; any decrypt problem fails closed; no value is printed or kept."""
import dataclasses, json, os, re, stat, sys, textwrap
from pathlib import Path
import pytest
import yaml
from deploy.env.registry import (SecretOwnershipError, file_secret_sources, generate_value, load_registry,
                                 secret_file_owner)
from deploy.render_secrets import (DictSource, SecretDecryptError, SecretMissingError, SecretRenderError,
                                   SopsFileSource, check_encrypted, main, render)
from tests.deploy.conftest import REPO

TOOLS = json.loads((REPO / "deploy/sops/tools.json").read_text(encoding="utf-8"))
ENC = "ENC[AES256_GCM,data:AAAA,iv:BBBB,tag:CCCC,type:str]"

def _store():
    return {name: f"value-{i:03d}-{'x' * 40}" for i, name in enumerate(file_secret_sources(load_registry()))}

def _render(environment, out, values, **kw):
    return render(environment, out, source=DictSource(values), apply_ownership=False, require_tmpfs=False, **kw)

def _fake_sops(tmp_path, mapping, *, version=None, fail=False):
    """A stand-in for the sops CLI: prints the pinned version, or a fixed JSON mapping, or fails like sops does."""
    (tmp_path / "fake-plain.json").write_text(json.dumps(mapping), encoding="utf-8")
    script = tmp_path / "fake_sops.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        if sys.argv[1:] == ["--version"]:
            print("sops {version or TOOLS['sops']['version']}")
            sys.exit(0)
        if {fail!r}:
            sys.stderr.write("Failed to get the data key required to decrypt the SOPS file.\\n")
            sys.exit(128)
        sys.stdout.write(open({str(tmp_path / 'fake-plain.json')!r}, encoding="utf-8").read())
    """), encoding="utf-8")
    key = tmp_path / "age.key"
    key.write_text("throwaway-test-identity", encoding="utf-8")  # a placeholder, never a real age identity
    if os.name == "posix":
        key.chmod(0o400)
    return [sys.executable, str(script)], key

def test_tools_are_pinned_by_version_and_checksum():
    for tool in ("sops", "age"):
        assert re.fullmatch(r"\d+\.\d+\.\d+", TOOLS[tool]["version"]), tool
        assert re.fullmatch(r"[0-9a-f]{64}", TOOLS[tool]["linux_amd64_sha256"]), tool
    script = (REPO / "deploy/sops/install_tools.sh").read_text(encoding="utf-8")
    assert "sha256sum -c" in script and "tools.json" in script

def test_render_writes_one_0400_file_per_secret(tmp_path):
    values = _store()
    report = _render("production", tmp_path / "secrets", values)
    files = {p.name for p in (tmp_path / "secrets").iterdir()}
    assert set(values) <= files and set(report.derived) <= files
    assert all(v not in repr(report) for v in values.values())
    assert (tmp_path / "secrets" / "AUTH_SECRET_KEY").read_text(encoding="utf-8") == values["AUTH_SECRET_KEY"]
    assert set(report.owners) == set(report.names) | set(report.derived)
    if os.name == "posix":
        assert {stat.S_IMODE((tmp_path / "secrets" / n).stat().st_mode) for n in files} == {0o400}
        assert stat.S_IMODE((tmp_path / "secrets").stat().st_mode) == 0o700

def test_each_file_is_owned_by_the_uid_that_reads_it(registry):
    assert secret_file_owner(registry, "AUTH_SECRET_KEY") == 1000
    assert secret_file_owner(registry, "GRAFANA_ADMIN_PASSWORD") == 472
    assert secret_file_owner(registry, "BASIC_AUTH_HASH") == 0
    assert secret_file_owner(registry, "POSTGRES_SUPERUSER_PASSWORD") == 0
    for source in file_secret_sources(registry):
        secret_file_owner(registry, source)  # every source has readers that can share one 0400 file

def test_a_source_two_uids_cannot_share_is_refused(registry):
    row = next(v for v in registry if v.source == "AUTH_SECRET_KEY")
    with pytest.raises(SecretOwnershipError) as exc:
        secret_file_owner([row, dataclasses.replace(row, services=("grafana",))], "AUTH_SECRET_KEY")
    assert exc.value.names == ["AUTH_SECRET_KEY"]

def test_render_is_all_or_nothing(tmp_path):
    values = _store()
    values.pop("ENCRYPTION_MASTER_KEY")
    with pytest.raises(SecretMissingError) as exc:
        _render("production", tmp_path / "secrets", values)
    assert exc.value.names == ["ENCRYPTION_MASTER_KEY"] and not (tmp_path / "secrets").exists()

def test_a_failed_rerender_keeps_the_previous_files(tmp_path):
    values = _store()
    _render("production", tmp_path / "secrets", values)
    broken = {k: v for k, v in values.items() if k != "AUTH_SECRET_KEY"}
    with pytest.raises(SecretMissingError):
        _render("production", tmp_path / "secrets", broken)
    assert (tmp_path / "secrets" / "AUTH_SECRET_KEY").read_text(encoding="utf-8") == values["AUTH_SECRET_KEY"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["secrets"]

def test_render_refuses_non_strict_environments(tmp_path):
    with pytest.raises(ValueError):
        _render("development", tmp_path / "s", _store())

def test_render_refuses_a_target_that_is_not_tmpfs(tmp_path, monkeypatch):
    monkeypatch.setattr("deploy.render_secrets._filesystem_type", lambda path: "ext4")
    with pytest.raises(SecretRenderError, match="tmpfs"):
        render("production", tmp_path / "s", source=DictSource(_store()), apply_ownership=False, require_tmpfs=True)
    assert not (tmp_path / "s").exists()

def test_sops_source_decrypts_with_the_pinned_binary(tmp_path):
    values = _store()
    cmd, key = _fake_sops(tmp_path, values)
    src = SopsFileSource(tmp_path / "staging.sops.yaml", sops_cmd=cmd, age_key_file=key)
    assert src.get("AUTH_SECRET_KEY") == values["AUTH_SECRET_KEY"] and src.get("NOT_A_SECRET") is None

def test_sops_source_fails_closed_without_echoing_sops(tmp_path):
    cmd, key = _fake_sops(tmp_path, _store(), fail=True)
    with pytest.raises(SecretDecryptError) as exc:
        SopsFileSource(tmp_path / "staging.sops.yaml", sops_cmd=cmd, age_key_file=key).get("AUTH_SECRET_KEY")
    assert "data key" not in str(exc.value)

def test_sops_source_refuses_an_unpinned_sops(tmp_path):
    cmd, key = _fake_sops(tmp_path, _store(), version="0.0.1")
    with pytest.raises(SecretDecryptError, match="version"):
        SopsFileSource(tmp_path / "staging.sops.yaml", sops_cmd=cmd, age_key_file=key).get("AUTH_SECRET_KEY")

def test_sops_source_requires_the_age_key_file(tmp_path, monkeypatch):
    cmd, _ = _fake_sops(tmp_path, _store())
    monkeypatch.delenv("SOPS_AGE_KEY_FILE", raising=False)
    with pytest.raises(SecretDecryptError, match="SOPS_AGE_KEY_FILE"):
        SopsFileSource(tmp_path / "staging.sops.yaml", sops_cmd=cmd).get("AUTH_SECRET_KEY")

@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_sops_source_refuses_a_readable_age_key(tmp_path):
    cmd, key = _fake_sops(tmp_path, _store())
    key.chmod(0o644)
    with pytest.raises(SecretDecryptError, match="permissions"):
        SopsFileSource(tmp_path / "staging.sops.yaml", sops_cmd=cmd, age_key_file=key).get("AUTH_SECRET_KEY")

def test_check_finds_plaintext_and_missing_entries_without_decrypting(tmp_path, registry):
    doc = {s: ENC for s in file_secret_sources(registry)}
    doc["sops"] = {"age": [{"recipient": "age1" + "q" * 58, "enc": "synthetic"}]}
    path = tmp_path / "staging.sops.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert check_encrypted(path, registry) == []
    doc["AUTH_SECRET_KEY"] = "plain-value-oops"
    doc.pop("ENCRYPTION_MASTER_KEY")
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    problems = check_encrypted(path, registry)
    assert any("AUTH_SECRET_KEY" in p and "not encrypted" in p for p in problems)
    assert any("ENCRYPTION_MASTER_KEY" in p and "missing" in p for p in problems)
    assert all("plain-value-oops" not in p for p in problems)

def test_cli_prints_names_only(tmp_path, monkeypatch, capsys):
    values = _store()
    monkeypatch.setattr("deploy.render_secrets._build_source", lambda args: DictSource(values))
    monkeypatch.setattr("deploy.render_secrets._running_as_root_on_linux", lambda: False)
    assert main(["--env", "production", "--sops-file", str(tmp_path / "production.sops.yaml"),
                 "--out", str(tmp_path / "s")]) == 0
    out = capsys.readouterr()
    assert "AUTH_SECRET_KEY" in out.out and all(v not in out.out + out.err for v in values.values())

def test_no_age_private_key_or_secret_store_call_in_the_tree():
    for root in ("deploy", "infrastructure", "docs/runbooks"):
        for p in (REPO / root).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".txt", ""}:
                text = p.read_text(encoding="utf-8", errors="ignore")
                assert "AGE-SECRET-KEY-" not in text, p
                assert "get-secret-value" not in text and "secretsmanager" not in text.lower(), p

@pytest.mark.integration
@pytest.mark.realinfra
def test_real_sops_and_age_round_trip_in_the_pinned_container(sops_tools, tmp_path, registry):
    fmt = {v.source: v.format for v in registry if v.source}
    values = {s: generate_value(fmt[s]) for s in file_secret_sources(registry)}
    (tmp_path / "plain.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")  # synthetic, throwaway
    script = textwrap.dedent(f"""
        set -eu
        trap 'rm -f /work/key.txt /work/other.txt /work/plain.yaml; chmod -R a+rwX /work' EXIT
        sops --version | grep -q "{TOOLS['sops']['version']}"
        age-keygen -o /work/key.txt 2>/dev/null && chmod 0400 /work/key.txt
        sops --encrypt --age "$(age-keygen -y /work/key.txt)" --input-type yaml --output-type yaml \\
            /work/plain.yaml > /work/staging.sops.yaml
        rm /work/plain.yaml
        python -m deploy.render_secrets --check --env staging --sops-file /work/staging.sops.yaml
        SOPS_AGE_KEY_FILE=/work/key.txt python -m deploy.render_secrets --env staging \\
            --sops-file /work/staging.sops.yaml --out /run/secrets/staging
        stat -c '%n %a %u' /run/secrets/staging/* > /work/stat.txt
        stat -c '%a' /run/secrets/staging > /work/dirmode.txt
        age-keygen -o /work/other.txt 2>/dev/null && chmod 0400 /work/other.txt
        if SOPS_AGE_KEY_FILE=/work/other.txt python -m deploy.render_secrets --env staging \\
            --sops-file /work/staging.sops.yaml --out /run/secrets/staging; then exit 9; fi
        stat -c '%n %a %u' /run/secrets/staging/* > /work/stat-after.txt
    """)
    res = sops_tools(script, work=tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert all(v not in res.stdout + res.stderr for v in values.values())
    rows = [line.split() for line in (tmp_path / "stat.txt").read_text(encoding="utf-8").splitlines()]
    assert {Path(n).name for n, _, _ in rows} >= set(values)
    assert {mode for _, mode, _ in rows} == {"400"}
    assert all(int(uid) == secret_file_owner(registry, Path(n).name) for n, _, uid in rows)
    assert (tmp_path / "dirmode.txt").read_text(encoding="utf-8").strip() == "700"
    assert (tmp_path / "stat-after.txt").read_text(encoding="utf-8") == (tmp_path / "stat.txt").read_text(encoding="utf-8")
    encrypted = (tmp_path / "staging.sops.yaml").read_bytes()
    assert b"AUTH_SECRET_KEY" in encrypted and values["AUTH_SECRET_KEY"].encode() not in encrypted
```
```python
# tests/deploy/test_mvp_352_deploy_runbook.py
"""MVP-352 (+ D-07, D-17): the deploy runbook ships a git-archive tarball of the checkpoint tree, never a token in a
clone URL, and supplies secrets only through the SOPS render with an off-host age key that is destroyed after use."""
import re
from tests.deploy.conftest import REPO

DEPLOY = (REPO / "infrastructure/aws-staging/DEPLOY.md").read_text(encoding="utf-8")

def test_no_token_in_any_clone_url():
    assert not re.search(r"https://[^\s/@]+@github\.com", DEPLOY)

def test_code_arrives_as_a_git_archive_tarball():
    assert "git archive" in DEPLOY and "ArtifactBucketName" in DEPLOY
    assert "git clone" not in DEPLOY and "GIT_ASKPASS" not in DEPLOY

def test_secrets_come_only_from_the_sops_render():
    assert "render_secrets" in DEPLOY and "SOPS_AGE_KEY_FILE" in DEPLOY and "shred -u" in DEPLOY
    assert "SSM-SessionManagerRunShell" in DEPLOY and ".env.prod" not in DEPLOY
    assert not re.search(r"ssm (get|put)-parameter|secretsmanager", DEPLOY, re.I)

def test_secrets_runbook_documents_rotation_and_rekey():
    text = (REPO / "docs/runbooks/secrets.md").read_text(encoding="utf-8")
    for needle in ("SOPS_AGE_KEY_FILE", "age-keygen", "--check", "updatekeys", "sops rotate",
                   "deploy/secrets/history/", "0400", "reboot"):
        assert needle in text, needle
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_348_secrets_store.py`; `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_352_deploy_runbook.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_348_secrets_store.py`
Expected: FAIL. `test_mvp_348_secrets_store.py` does not collect: `FileNotFoundError` for `deploy/sops/tools.json`, then `ModuleNotFoundError: deploy.render_secrets` and `ImportError: cannot import name 'secret_file_owner'` (no secrets store integration exists: every secret is hand-written into one plaintext `.env.prod`, MVP-348). In `test_mvp_352_deploy_runbook.py`, `test_no_token_in_any_clone_url` fails on `https://<TOKEN>@github.com` in the section-5 clone step (MVP-352), `test_code_arrives_as_a_git_archive_tarball` on the same `git clone`, `test_secrets_come_only_from_the_sops_render` on the section-6 `.env.prod` steps, and `test_secrets_runbook_documents_rotation_and_rekey` because the runbook has only WS-03.7's dev section. The DOCKER run fails because `deploy/sops/Dockerfile` does not exist, so `sops_tools` cannot build.
- [ ] **Step 3: Implement** `render`, the sources, `check_encrypted`, `main`, `secret_file_owner`, the tools files, the fixture and the doc edits. Approach: `render` validates before it writes anything: a strict environment, every registry source present (`SecretMissingError`), every owner resolvable (`SecretOwnershipError`), and, when required, a target on tmpfs (`_filesystem_type(path)` reads the longest matching mount point in `/proc/self/mountinfo`; anything else, including a platform without that file, counts as not tmpfs). It then writes into a sibling `<out>.tmp-<random>` on the same filesystem (directory 0700; each file opened with `O_CREAT | O_EXCL | O_WRONLY` and mode 0o400, then `os.chown` to `secret_file_owner` when ownership applies), renames an existing `<out>` aside, renames the new directory in and removes the old one; any failure removes the temporary directory and leaves the previous files untouched. `secret_file_owner` collects the reader uid (`SECRET_READER_UIDS`) of every service of every row with that source: one non-root uid, or root only, resolves; two different non-root uids, or a non-root reader beside a root reader outside `ROOT_READERS_WITH_DAC_OVERRIDE`, raise `SecretOwnershipError`. The owner matters because Compose bind-mounts file secrets, so the container sees the host file's owner and mode (Compose ignores `uid`, `gid` and `mode` for file-based secrets). `SopsFileSource` compares `sops --version` with `tools.json`, requires `SOPS_AGE_KEY_FILE` (on POSIX with no group or world bits), runs `sops --decrypt --output-type json <file>` once with an environment of only `PATH` and `SOPS_AGE_KEY_FILE` and a 60 s timeout, accepts only a flat mapping of strings (it drops the `sops` metadata key), and turns every failure into `SecretDecryptError` naming the step, never the output. `check_encrypted` parses the YAML only. `_build_source(args)` returns `SopsFileSource(args.sops_file)`. The strict section of `docs/runbooks/secrets.md` covers: one age key pair per environment, generated offline with `age-keygen` (the public recipient goes into `deploy/secrets/.sops.yaml`, which holds public recipients only, written by the operator); creating and editing the environment file with `sops --config deploy/secrets/.sops.yaml edit` on the operator workstation; `render_secrets --check` before shipping; the host render with its 0400 files and `shred -u` of the key; routine rotation (keep the previous encrypted file under `deploy/secrets/history/<env>/<UTC timestamp>.sops.yaml`, because nothing is committed during the program (D-17), then edit, check, ship, re-render and restart the affected services); re-key after a compromise (a new key pair, the new recipient in `.sops.yaml`, `sops updatekeys` and `sops rotate -i`, then every value rotated, because the old key could read them all, the PHI key through `ENCRYPTION_RETIRED_KEYS_JSON`, and every copy of the old private key destroyed); and the reboot procedure. No runbook command prints a value. The SES SMTP username and password (D-12) are entries of the environment's SOPS file like every other secret.
- [ ] **Step 4: Run it and confirm it passes**
Run: the three commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.9: Backups: continuous WAL archiving, encrypted base backups, restore with point-in-time recovery, and the drill
**Covers:** MVP-037
**Severity:** blocker
**Wave:** 4
**Depends on:** WS-03.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-09 (default assumed: nightly physical base backup plus continuous WAL archiving to an encrypted off-host target, S3 with SSE-KMS, versioning and a 35-day object lock on AWS, or an encrypted external volume on a local host; RPO 15 min, RTO 4 h; Redis, Kafka and Weaviate are not backed up)
**Real infra:** yes (postgres plus the backup sidecar: backup, destroy, restore, compare)
**Cross-workstream prerequisites:** WS-02's pg_hba policy allows `replication` for the image superuser from the compose `data` network (`hostssl` in strict), requested below; WS-04.12 alerts on the textfile metric this task writes; WS-05's W4 drill driver uses this fragment (requested below).
**Files:**
- Create: `docker-compose.backup.yml` (services `pg-backup` (`build: {context: ., dockerfile: deploy/backup/Dockerfile}`, `image: solace/pg-backup:${SOLACE_IMAGE_TAG:-dev}`; volumes `postgres_data:/var/lib/postgresql/data:ro`, `pg_wal_spool:/wal-spool`, `pg_backup_work:/work`, `solace_textfile_metrics:/textfile`, `${BACKUP_TARGET_DIR:?...}:/backup-target`; secrets `POSTGRES_SUPERUSER_PASSWORD`, `BACKUP_ENCRYPTION_KEY`, and in strict `postgres-ca.crt`; env `BACKUP_BASE_INTERVAL_SECONDS=86400`, `BACKUP_WAL_PUSH_SECONDS=30`, `BACKUP_RETENTION_DAYS=35`, optional `BACKUP_S3_URI`, `BACKUP_KMS_KEY_ID`; network `data`; no ports) and `pg-restore` (same image, `profiles: ["restore"]`, volumes `postgres_data` rw, `pg_wal_restore:/wal-restore`, the target read-only, entry `solace-restore`); an override of base `postgres` that only adds the `pg_wal_spool:/wal-spool` volume; top-level volumes; no base service's image or build is redefined)
- Create: `deploy/backup/Dockerfile` (FROM the same `postgres:16-alpine` image as the base service; `apk add --no-cache gnupg aws-cli`; copies the two scripts to `/usr/local/bin`)
- Create: `deploy/backup/solace-backup.sh`, `deploy/backup/solace-restore.sh`, `docs/runbooks/backup-restore.md`
- Modify: `docker-compose.yml` and `docker-compose.prod.yml` (postgres command adds `-c archive_mode=on -c archive_timeout=300 -c "archive_command=if [ -e /wal-spool/.enabled ]; then test ! -f /wal-spool/%f && cp %p /wal-spool/%f.part && mv /wal-spool/%f.part /wal-spool/%f; fi"` and mounts the `pg_wal_spool` volume, so archiving is effective only while the sidecar runs and WAL never accumulates without it)
- Modify: `deploy/env/env-contract.yaml` (`BACKUP_ENCRYPTION_KEY` secret format `urlsafe48`; `BACKUP_TARGET_DIR`, `BACKUP_S3_URI`, `BACKUP_KMS_KEY_ID`, `BACKUP_RETENTION_DAYS`, `BACKUP_BASE_INTERVAL_SECONDS` with services `[pg-backup, pg-restore]`), `deploy/env/generate_dev_env.py` (writes `BACKUP_ENCRYPTION_KEY`), `.env.example` (RENDER)
- Modify: `tests/deploy/conftest.py` (the `compose_project` helpers `with_fragments`, `destroy_volume`, `run(..., profiles=...)` and `textfile` described after the test)
- Test: `tests/deploy/test_mvp_037_backup_restore.py`
**Interfaces:**
- Consumes: the image superuser (D-02); file secrets (WS-03.7); `render(..., fragments=...)` and `compose_project`.
- Produces: `docker-compose.backup.yml` services `pg-backup` and `pg-restore`; `solace-backup base-now` (immediate base backup, prints the backup id); `solace-restore [--backup-id ID] [--target-time ISO8601] [--wipe]` (prints `restore_seconds=<n> backup_id=<id> recovered_to=<timestamp>`; exit 2 if postgres is running, 3 reserved for key-material mismatch in WS-03.10); textfile metric `/textfile/solace_backup.prom` on the named volume `solace_textfile_metrics` with `solace_backup_last_success_timestamp_seconds{store="postgres_base"}` and `{store="postgres_wal"}` (IF-18; consumed by WS-04.12's node-exporter textfile collector and its `BackupStale`/`WalArchiveStale` rules, request from WS-04); target layout `base/<backup_id>/{base.tar.gz.gpg,pg_wal.tar.gz.gpg,manifest.json}` and `wal/<segment>.gpg`.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_037_backup_restore.py`
```python
"""MVP-037: Postgres is backed up continuously and encrypted off the data volume; a destroyed volume is restored
to the last archived transaction within the RTO, and the backup holds no plaintext."""
import time
import pytest
import yaml
from tests.deploy.conftest import REPO

def test_fragment_adds_backup_services_without_redefining_base(render):
    r = render("strict", fragments=["docker-compose.backup.yml"])
    assert "pg-backup" in r.services and not r.services["pg-backup"].get("ports")
    frag = yaml.safe_load((REPO / "docker-compose.backup.yml").read_text(encoding="utf-8"))
    assert "image" not in frag["services"]["postgres"] and "build" not in frag["services"]["postgres"]
    mounted = {s["source"] for s in r.services["pg-backup"]["secrets"]}
    assert {"BACKUP_ENCRYPTION_KEY", "POSTGRES_SUPERUSER_PASSWORD"} <= mounted
    assert "AUTH_SECRET_KEY" not in mounted and "ENCRYPTION_MASTER_KEY" not in mounted

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_postgres_archives_wal_only_through_the_spool(render, profile):
    cmd = " ".join(render(profile).services["postgres"]["command"])
    assert "archive_mode=on" in cmd and "archive_timeout=300" in cmd and "/wal-spool/.enabled" in cmd

@pytest.mark.integration
@pytest.mark.e2e
def test_restore_drill_recovers_every_archived_row(compose_project, tmp_path):
    target = tmp_path / "backup-target"
    target.mkdir()
    p = compose_project.with_fragments(["docker-compose.backup.yml"], env={"BACKUP_TARGET_DIR": str(target)})
    p.up(["postgres", "pg-backup"])
    p.psql("CREATE TABLE backup_drill(id int primary key, body text)")
    p.psql("INSERT INTO backup_drill SELECT g, md5(g::text) FROM generate_series(1, 1000) g")
    assert p.exec("pg-backup", "solace-backup", "base-now").strip()
    p.psql("INSERT INTO backup_drill SELECT g, md5(g::text) FROM generate_series(1001, 1100) g")
    lsn = p.psql("SELECT pg_walfile_name(pg_switch_wal())")[0][0]
    deadline = time.monotonic() + 120
    while not (target / "wal" / f"{lsn}.gpg").exists():
        assert time.monotonic() < deadline, "WAL segment never reached the backup target"
        time.sleep(2)
    checksum = p.psql("SELECT md5(string_agg(id::text || body, ',' ORDER BY id)) FROM backup_drill")
    p.destroy_volume("postgres", "postgres_data")
    started = time.monotonic()
    out = p.run("pg-restore", profiles=["restore"]).stdout
    elapsed = time.monotonic() - started
    assert "restore_seconds=" in out and elapsed < 900
    p.up(["postgres"])
    assert p.psql("SELECT count(*) FROM backup_drill") == [(1100,)]
    assert p.psql("SELECT md5(string_agg(id::text || body, ',' ORDER BY id)) FROM backup_drill") == checksum
    for f in target.rglob("*.gpg"):
        assert b"backup_drill" not in f.read_bytes()
    assert (compose_project.textfile("solace_backup.prom")).count("solace_backup_last_success_timestamp_seconds") == 2
```
(`compose_project.with_fragments`, `destroy_volume(service, volume)` (stops and removes `service` and every other project container that mounts the volume, such as `pg-backup`, whose read-only mount would otherwise keep the volume in use, then runs `docker volume rm <project>_<volume>`; callers `up` the agent again when they need it), `run(..., profiles=...)` and `textfile(name)` (reads the file from the `solace_textfile_metrics` volume with a throwaway busybox container) are added to the fixture in this task.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_restore.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_restore.py`
Expected: FAIL with `FileNotFoundError: docker-compose.backup.yml` and no `archive_mode` in the postgres command (MVP-037: no backup, restore or rollback exists for any PHI store; the only copy of the data is the `postgres_data` volume).
- [ ] **Step 3: Implement** the fragment, image and scripts. Approach for `solace-backup.sh` (POSIX sh, runs as root, then `su-exec postgres` for Postgres client calls): at start `chown 70:70 /wal-spool` (the alpine postgres uid) and create `/wal-spool/.enabled`; every `BACKUP_WAL_PUSH_SECONDS` encrypt each completed spool file with `gpg --batch --pinentry-mode loopback --passphrase-file /run/secrets/BACKUP_ENCRYPTION_KEY --symmetric --cipher-algo AES256`, write it to `wal/<segment>.gpg.part`, fsync, rename, delete the spool file and stamp `postgres_wal`; every `BACKUP_BASE_INTERVAL_SECONDS` (and on `base-now`) run `pg_basebackup -h postgres -U postgres -D /work/<id> -Ft -z -X stream -c fast`, encrypt both tars, write `manifest.json` (`backup_id`, timestamps, server version, `system_identifier`, start LSN and WAL segment), move into `base/<id>/` and stamp `postgres_base`; prune base backups older than `BACKUP_RETENTION_DAYS` and WAL older than the oldest kept base; write the textfile metric atomically (`.prom.tmp` then rename); when `BACKUP_S3_URI` is set, `aws s3 sync /backup-target "$BACKUP_S3_URI" --sse aws:kms --sse-kms-key-id "$BACKUP_KMS_KEY_ID" --only-show-errors` after each base and every minute. Approach for `solace-restore.sh`: refuse (exit 2) if `pg_isready -h postgres` succeeds; refuse a non-empty data directory without `--wipe`; decrypt the chosen base backup into the data directory and every WAL file from its start segment onward into `/wal-restore`; write `recovery.signal` and `restore_command = 'cp /wal-restore/%f %p'`, `recovery_target_timeline = 'latest'`, optional `recovery_target_time`, `recovery_target_action = 'promote'` into `postgresql.auto.conf`; run a private server (`pg_ctl -w -t 14400 -o "-c listen_addresses='' -c archive_mode=off" start`) until `pg_is_in_recovery()` is false, stop it, clear `/wal-restore`, and print the summary line. `docs/runbooks/backup-restore.md` covers schedule, targets, RPO/RTO, the drill command, fresh-host restore and monthly drill cadence. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS (record the measured `restore_seconds` in the ledger)
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE in strict with `-f docker-compose.backup.yml` added by the harness. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.10: Key escrow, key-id manifests and deletion protection for key material
**Covers:** MVP-350
**Severity:** high
**Wave:** 4
**Depends on:** WS-03.9
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (changed 2026-10-19 to SOPS + age: key material lives in the environment's SOPS file, which is also encrypted to an offline escrow age recipient; each previous encrypted version is kept under `deploy/secrets/history/<env>/` (WS-03.8); and an offline escrow bundle is refreshed after every rotation; there is no Secrets Manager secret or secrets KMS key to protect), D-09 (accepted: backups record key ids, never keys; the backup bucket's KMS key is retained by WS-03.11)
**Real infra:** yes (the restore refuses a fresh, mismatched secrets directory)
**Cross-workstream prerequisites:** none.
**Files:**
- Create: `deploy/key_ids.py`, `deploy/key_escrow.py`
- Modify: `deploy/render_secrets.py` and `deploy/env/generate_dev_env.py` (both write `key-ids.json` next to the secret files; under `render_secrets` it is mode 0400 and owned by root, like the other files of its root readers; `check_encrypted` gains `escrow_recipients` and reports a file that is not also encrypted to every escrow recipient, or that is encrypted to escrow recipients only; the `--check` CLI gains `--escrow-recipients FILE`, defaulting to the owner-written escrow list in the secrets directory, and fails closed when the list is missing or empty)
- Modify: `tests/deploy/test_mvp_348_secrets_store.py` (the pinned-container round trip also generates a throwaway escrow identity with `age-keygen`, encrypts to both recipients and passes `--escrow-recipients /work/escrow.txt` to `--check`, so WS-03.8's test stays green)
- Modify: `deploy/backup/solace-backup.sh` (copies `/run/secrets/key-ids.json` into each `manifest.json` under `key_ids`), `deploy/backup/solace-restore.sh` (computes the restoring host's key ids from its mounted secrets and exits 3 with `key material does not match backup manifest: <names>` on any mismatch; `--accept-key-mismatch` exists only for ops-documented forensic restores), `docker-compose.backup.yml` (mounts `key-ids.json` into `pg-backup`; mounts the key-material secrets into `pg-restore` only)
- Modify: `docs/runbooks/backup-restore.md` and `docs/runbooks/secrets.md` (escrow ceremony: two custodians; an offline X25519 identity for the bundle and an offline age identity whose public recipient every environment's SOPS file is also encrypted to, listed one per line in deploy/secrets/escrow-recipients.txt (public keys only, written by the owner); bundle refresh after every rotation; verification step)
- Modify: `tests/deploy/conftest.py` (`compose_project.with_regenerated_secrets(keep=[...])`, described after the test)
- Test: `tests/deploy/test_mvp_350_key_escrow.py`
**Interfaces:**
- Consumes: secret files (WS-03.7, WS-03.8); backup manifest (WS-03.9).
- Produces: `deploy.key_ids`: `KEY_MATERIAL: tuple[str, ...]` (`ENCRYPTION_MASTER_KEY`, `ENCRYPTION_SEARCH_HASH_SALT`, `ENCRYPTION_RETIRED_KEYS_JSON`, `AUDIT_HMAC_KEY`, `AUTH_PRIVATE_KEY_PEM`, `FERNET_TOKEN_KEY`, `FERNET_FIELD_KEY`, `BACKUP_ENCRYPTION_KEY`, and every `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>`), `fingerprint(name: str, value: str) -> str` (first 16 hex of HMAC-SHA256 keyed by the value over `b"solace-key-id-v1:" + name`), `write_key_ids(secrets_dir: Path) -> Path`; `deploy.key_escrow`: `generate_escrow_keypair() -> tuple[bytes, bytes]` (PEM private, PEM public), `export_bundle(source: SecretSource, environment: str, recipient_public_pem: bytes, out: Path) -> list[str]`, `verify_bundle(bundle: Path, identity_private_pem: bytes, key_ids: Mapping[str, str]) -> list[str]`; CLI `python -m deploy.key_escrow export|verify ...` printing names only; `deploy.render_secrets.check_encrypted(path, registry=None, *, escrow_recipients: Sequence[str] = ())`.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_350_key_escrow.py`
```python
"""MVP-350: key material survives the host (escrow), backups record key ids not keys, and a restore onto
mismatched keys is refused instead of producing unreadable PHI."""
import json
import pytest
from deploy.key_ids import KEY_MATERIAL, fingerprint, write_key_ids
from deploy.key_escrow import export_bundle, generate_escrow_keypair, verify_bundle
from deploy.render_secrets import DictSource

def _material():
    return {n: f"material-{n}-{'k' * 32}" for n in KEY_MATERIAL if not n.endswith(".*")}

def test_key_ids_hold_no_key_bytes(tmp_path):
    for n, v in _material().items():
        (tmp_path / n).write_text(v, encoding="utf-8")
    ids_text = write_key_ids(tmp_path).read_text(encoding="utf-8")
    ids = json.loads(ids_text)
    assert set(ids) == set(_material())
    assert all(v not in ids_text for v in _material().values())
    assert ids["AUDIT_HMAC_KEY"] == fingerprint("AUDIT_HMAC_KEY", _material()["AUDIT_HMAC_KEY"])

def test_escrow_round_trip_and_no_plaintext(tmp_path):
    private_pem, public_pem = generate_escrow_keypair()
    material = _material()
    names = export_bundle(DictSource(material), "production", public_pem, tmp_path / "escrow.bundle")
    raw = (tmp_path / "escrow.bundle").read_bytes()
    assert set(names) == set(material) and all(v.encode() not in raw for v in material.values())
    ids = {n: fingerprint(n, v) for n, v in material.items()}
    assert set(verify_bundle(tmp_path / "escrow.bundle", private_pem, ids)) == set(material)

def test_escrow_verify_detects_a_rotated_key(tmp_path):
    private_pem, public_pem = generate_escrow_keypair()
    material = _material()
    export_bundle(DictSource(material), "production", public_pem, tmp_path / "b")
    ids = {n: fingerprint(n, v) for n, v in material.items()}
    ids["ENCRYPTION_MASTER_KEY"] = "0" * 16
    with pytest.raises(ValueError, match="ENCRYPTION_MASTER_KEY"):
        verify_bundle(tmp_path / "b", private_pem, ids)

def test_every_environment_file_is_also_encrypted_to_escrow(tmp_path):
    import yaml
    from deploy.env.registry import file_secret_sources, load_registry
    from deploy.render_secrets import check_encrypted
    escrow, deploy_key = "age1" + "e" * 58, "age1" + "d" * 58  # synthetic recipients, not real keys
    doc = {s: "ENC[AES256_GCM,data:AAAA,iv:BBBB,tag:CCCC,type:str]" for s in file_secret_sources(load_registry())}
    path = tmp_path / "production.sops.yaml"
    def problems(recipients):
        doc["sops"] = {"age": [{"recipient": r, "enc": "synthetic"} for r in recipients]}
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return check_encrypted(path, escrow_recipients=[escrow])
    assert any("escrow" in p for p in problems([deploy_key]))
    assert any("deploy recipient" in p for p in problems([escrow]))
    assert problems([deploy_key, escrow]) == []

@pytest.mark.integration
@pytest.mark.e2e
def test_restore_refuses_mismatched_key_material(compose_project, tmp_path):
    target = tmp_path / "t"
    target.mkdir()
    p = compose_project.with_fragments(["docker-compose.backup.yml"], env={"BACKUP_TARGET_DIR": str(target)})
    p.up(["postgres", "pg-backup"])
    p.exec("pg-backup", "solace-backup", "base-now")
    p.destroy_volume("postgres", "postgres_data")
    fresh = p.with_regenerated_secrets(keep=["BACKUP_ENCRYPTION_KEY", "POSTGRES_SUPERUSER_PASSWORD"])
    res = fresh.run("pg-restore", profiles=["restore"], check=False)
    assert res.returncode == 3 and "ENCRYPTION_MASTER_KEY" in res.stdout + res.stderr
    assert p.run("pg-restore", profiles=["restore"]).returncode == 0
```
(`with_regenerated_secrets(keep=[...])` regenerates every other secret file into a new directory, simulating a fresh host whose operator wrote new keys.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_350_key_escrow.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_350_key_escrow.py`
Expected: FAIL with `ModuleNotFoundError: deploy.key_ids` and `deploy.key_escrow` (MVP-350: the keys exist only on the host volume, nothing escrows them and no backup records which keys it needs); `test_every_environment_file_is_also_encrypted_to_escrow` fails with `TypeError` on `escrow_recipients`, because WS-03.8's check accepts a file encrypted only to the deploy key, so losing that key would lose every secret; the restore drill has no key check, so it would "succeed" on mismatched keys.
- [ ] **Step 3: Implement** the two modules, the escrow-recipient check and the script, fragment and runbook edits. Approach: `check_encrypted` compares the `sops.age[].recipient` set with the escrow list, requires every escrow recipient and at least one other (deploy) recipient, and reports names only; escrow is ECIES with `cryptography` only (ephemeral X25519, HKDF-SHA256 with info `solace-escrow-v1`, AES-256-GCM over the JSON mapping, bundle JSON `{version, environment, created_at, recipient_fingerprint, ephemeral_public, nonce, ciphertext}`); `verify_bundle` decrypts in memory, compares fingerprints and returns names; no value is printed by any CLI.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_restore.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_348_secrets_store.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.11: CloudFormation host: public ingress, stable address and DNS, a separate data volume, the backup bucket and host alarms
**Covers:** MVP-043, MVP-044
**Severity:** high
**Wave:** 4
**Depends on:** WS-03.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-25 (changed 2026-10-19: AWS hosts the release gate and the pilot through the existing IAM user `MedCodex-Local`; this template is validated with cfn-lint and cfn-guard and deployed only after the owner attaches `solace-staging-deployer-policy.json` to that user, because the 2026-10-19 simulation denied every provisioning action; until then all real-infra tests run on local Docker and the W4 restore drill uses the local encrypted-volume target and says so), D-09 (accepted: S3 with SSE-KMS, versioning and a 35-day object lock; RPO 15 min, RTO 4 h)
**Real infra:** no (static template checks, cfn-lint, cfn-guard when installed)
**Cross-workstream prerequisites:** WS-01 runs cfn-lint and cfn-guard with `infrastructure/aws-staging/guard/solace-staging.guard` in `make ci-local` and CI (requested below); WS-04.12 covers the Prometheus side of host alerting.
**Files:**
- Modify: `infrastructure/aws-staging/solace-staging.yaml` (parameters `DomainName`, `HostedZoneId` (both default empty), `OnCallEmail`, `DataVolumeSizeGb` (default 100), `AllowedHttpCidr` (default `0.0.0.0/0`); SecurityGroupIngress 80/tcp, 443/tcp, 443/udp from `AllowedHttpCidr`; `AWS::EC2::EIP` plus `AWS::EC2::EIPAssociation`; condition `HasDns` with `AWS::Route53::RecordSet` A records for the apex, `api.`, `auth.`, `grafana.` to the EIP; `AWS::EC2::Volume` gp3 encrypted, `DeletionPolicy: Snapshot`, `UpdateReplacePolicy: Snapshot`, attached at `/dev/sdf` and mounted at `/var/lib/docker` by UserData before Docker starts (format only when blank, fstab with `nofail`); backup bucket `BackupBucket` with `VersioningConfiguration` Enabled, `ObjectLockEnabled: true` with a 35-day GOVERNANCE default retention, SSE-KMS with the dedicated key `BackupKmsKey` (`EnableKeyRotation: true`, `DeletionPolicy: Retain`, `UpdateReplacePolicy: Retain`, since losing it loses every backup; this replaces the secrets-key retention WS-03.10 no longer needs under D-07), `PublicAccessBlockConfiguration` all true, a TLS-only bucket policy, noncurrent-version expiry 35 days, `DeletionPolicy: Retain`; role permissions `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on that bucket only, `kms:GenerateDataKey` and `kms:Decrypt` on `BackupKmsKey` only with `kms:ViaService = s3.${AWS::Region}.amazonaws.com`, and `CloudWatchAgentServerPolicy`; UserData installs and starts the CloudWatch agent with `mem_used_percent` and `disk_used_percent` for `/` and `/var/lib/docker`; `AWS::SNS::Topic` with an email subscription to `OnCallEmail`; alarms `StatusCheckFailed_System` (recover action `arn:${AWS::Partition}:automate:${AWS::Region}:ec2:recover`), `StatusCheckFailed_Instance` (reboot action), CPU > 90 % for 15 min, memory > 90 %, each disk > 80 %, all notifying the topic; outputs `PublicIp`, `BackupBucketName`, `BackupKmsKeyArn`, `AlarmTopicArn`)
- Create: `infrastructure/aws-staging/guard/solace-staging.guard` (rules: every EBS volume encrypted; no SG ingress beyond 80/443; S3 buckets block public access, enable versioning and SSE-KMS; KMS keys rotate)
- Create: `infrastructure/aws-staging/solace-staging-deployer-host-policy.json` (the W4 additions as a second customer managed policy beside WS-03.20's, so neither exceeds the 6,144-character managed-policy limit: EIP, Route53 record, EC2 volume, S3 backup bucket and object-lock configuration, SNS, CloudWatch alarm and KMS key, alias and rotation actions scoped to `solace-staging-*` names; no Secrets Manager or Parameter Store action; the owner attaches it to `MedCodex-Local` the same way)
- Modify: `infrastructure/aws-staging/DEPLOY.md` (ingress, DNS, data-volume survival on teardown, alarm subscription confirmation, backup bucket wiring via `BACKUP_S3_URI` and `BACKUP_KMS_KEY_ID` from the `BackupBucketName` and `BackupKmsKeyArn` outputs; until the owner has attached the policy and the stack exists, backups and the restore drill use the local encrypted-volume target, and the drill report says so)
- Test: `tests/deploy/test_mvp_043_cfn_template.py`
**Interfaces:**
- Consumes: `cfn_template` fixture (WS-03.20).
- Produces: template outputs `PublicIp`, `BackupBucketName`, `BackupKmsKeyArn`, `AlarmTopicArn` used by DEPLOY.md, `BACKUP_S3_URI` and `BACKUP_KMS_KEY_ID`.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_043_cfn_template.py`
```python
"""MVP-043/MVP-044: users can reach Caddy on 80/443 at a stable address; data outlives the instance; the host pages on-call;
backups are written only through their retained key (D-09) and the deployer policy covers the W4 host (D-25)."""
import json, shutil, subprocess, sys
import pytest
from tests.deploy.conftest import REPO

def _of_type(t, kind):
    return [r for r in t["Resources"].values() if r["Type"] == kind]

def _acts(stmt):
    a = stmt.get("Action", [])
    return [a] if isinstance(a, str) else list(a)

def _role_statements(t):
    res = t["Resources"]
    stmts = [s for p in res["InstanceRole"]["Properties"].get("Policies", []) for s in p["PolicyDocument"]["Statement"]]
    return stmts + [s for r in res.values() if r["Type"] == "AWS::IAM::RolePolicy"
                    for s in r["Properties"]["PolicyDocument"]["Statement"]]

def test_ingress_allows_only_http_and_https(cfn_template):
    sg = cfn_template["Resources"]["SecurityGroup"]["Properties"]
    ports = sorted((r["IpProtocol"], r["FromPort"], r["ToPort"]) for r in sg["SecurityGroupIngress"])
    assert ports == [("tcp", 80, 80), ("tcp", 443, 443), ("udp", 443, 443)]

def test_stable_address_and_optional_dns(cfn_template):
    assert _of_type(cfn_template, "AWS::EC2::EIP")
    records = _of_type(cfn_template, "AWS::Route53::RecordSet")
    assert records and all(r.get("Condition") == "HasDns" for r in records)

def test_data_lives_on_a_separate_snapshotted_volume(cfn_template):
    vol = _of_type(cfn_template, "AWS::EC2::Volume")[0]
    assert vol["Properties"]["Encrypted"] is True and vol["DeletionPolicy"] == "Snapshot"
    assert "/var/lib/docker" in str(cfn_template["Resources"]["Instance"]["Properties"]["UserData"])

def test_backup_bucket_is_locked_versioned_and_private(cfn_template):
    b = cfn_template["Resources"]["BackupBucket"]["Properties"]
    assert b["VersioningConfiguration"]["Status"] == "Enabled" and b["ObjectLockEnabled"] is True
    assert b["ObjectLockConfiguration"]["Rule"]["DefaultRetention"]["Days"] == 35
    assert all(b["PublicAccessBlockConfiguration"].values())
    assert b["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "aws:kms"
    key = cfn_template["Resources"]["BackupKmsKey"]
    assert key["Properties"]["EnableKeyRotation"] is True
    assert key["DeletionPolicy"] == "Retain" and key["UpdateReplacePolicy"] == "Retain"

def test_host_writes_backups_only_through_the_backup_key(cfn_template):
    kms = [s for s in _role_statements(cfn_template) if any(a.startswith("kms:") for a in _acts(s))]
    assert kms and all("BackupKmsKey" in json.dumps(s["Resource"]) and "kms:ViaService" in json.dumps(s.get("Condition", {}))
                       for s in kms)

def test_deployer_policy_covers_the_w4_host():
    names = ("solace-staging-deployer-policy.json", "solace-staging-deployer-host-policy.json")
    policies = [json.loads((REPO / "infrastructure/aws-staging" / n).read_text(encoding="utf-8")) for n in names]
    assert all(len(json.dumps(p, separators=(",", ":"))) <= 6144 for p in policies)  # IAM managed-policy size limit
    acts = {a for p in policies for s in p["Statement"] for a in _acts(s)}
    needed = {"ec2:AllocateAddress", "ec2:AssociateAddress", "ec2:CreateVolume", "ec2:AttachVolume",
              "route53:ChangeResourceRecordSets", "s3:PutBucketObjectLockConfiguration", "kms:CreateKey",
              "kms:EnableKeyRotation", "sns:CreateTopic", "cloudwatch:PutMetricAlarm"}
    assert sorted(needed - acts) == []
    assert not [a for a in acts if a.startswith("secretsmanager:")]

def test_host_alarms_page_on_call_and_auto_recover(cfn_template):
    alarms = {a["Properties"]["MetricName"]: a["Properties"] for a in _of_type(cfn_template, "AWS::CloudWatch::Alarm")}
    assert {"StatusCheckFailed_System", "StatusCheckFailed_Instance", "CPUUtilization", "mem_used_percent", "disk_used_percent"} <= set(alarms)
    assert "ec2:recover" in str(alarms["StatusCheckFailed_System"]["AlarmActions"])
    topic = _of_type(cfn_template, "AWS::SNS::Topic")[0]
    assert topic["Properties"]["Subscription"][0]["Protocol"] == "email"
    role = cfn_template["Resources"]["InstanceRole"]["Properties"]
    assert any("CloudWatchAgentServerPolicy" in str(m) for m in role["ManagedPolicyArns"])

def test_cfn_lint_passes():
    out = subprocess.run([sys.executable, "-m", "cfnlint", str(REPO / "infrastructure/aws-staging/solace-staging.yaml")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr

def test_cfn_guard_passes():
    if shutil.which("cfn-guard") is None:
        pytest.skip("cfn-guard not installed locally; WS-01 runs it in make ci-local")
    out = subprocess.run(["cfn-guard", "validate", "-d", str(REPO / "infrastructure/aws-staging/solace-staging.yaml"),
                          "-r", str(REPO / "infrastructure/aws-staging/guard/solace-staging.guard")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_043_cfn_template.py`
Expected: FAIL. `KeyError: 'SecurityGroupIngress'` (egress-only SG, solace-staging.yaml:82-93, MVP-043); no EIP, Route53, separate volume (root volume `DeleteOnTermination: true`) or backup bucket (`KeyError: 'BackupBucket'`; only WS-03.20's artifact bucket exists), no `BackupKmsKey` and no KMS statement on the role; no `AWS::CloudWatch::Alarm`, SNS topic or `CloudWatchAgentServerPolicy` (MVP-044); `test_deployer_policy_covers_the_w4_host` fails with `FileNotFoundError` for `solace-staging-deployer-host-policy.json` (WS-03.20's policy covers only the W3 host). `test_cfn_lint_passes` passes on the W3 template and must stay green.
- [ ] **Step 3: Implement** the template, guard rules, deployer policy and runbook edits listed under Files. Approach: keep the SSM-only management path (no SSH ingress); the EIP keeps ACME and DNS stable across stop/start; UserData waits for `/dev/nvme1n1` (or `/dev/sdf`) before formatting.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.12: Caddy request-size and timeout limits; no false rate-limit claim
**Covers:** MVP-115
**Severity:** medium
**Wave:** 4
**Depends on:** WS-03.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: Caddy gets request-size and timeout limits but no third-party module; per-user/per-IP throttling stays in the app limiter)
**Real infra:** yes (`caddy adapt`)
**Cross-workstream prerequisites:** WS-07's app-level limiter with `TRUSTED_PROXY_HOPS=1` (IF-04) is the throttle; this task only removes the false Caddy claim and bounds requests.
**Files:**
- Modify: `Caddyfile` (global `servers { timeouts { read_header 10s read_body 30s idle 2m } }`; in both sites `request_body { max_size 1MB }`; each `reverse_proxy` gets `transport http { dial_timeout 5s response_header_timeout 30s }`; the "Per-IP rate limit (Caddy built-in rate_limit...)" comment block is replaced by a one-line pointer to the app limiter)
- Test: `tests/deploy/test_mvp_115_caddy_limits.py`
**Interfaces:**
- Consumes: `caddy_adapt` fixture.
- Produces: the edge limits above (WebSocket upgrades are unaffected because hijacked streams ignore these HTTP timeouts).

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_115_caddy_limits.py`
```python
"""MVP-115: the edge bounds request bodies and slow clients, and no longer promises rate limiting it does not do."""
import json, re
import pytest
from tests.deploy.conftest import REPO

CADDYFILE = (REPO / "Caddyfile").read_text(encoding="utf-8")

def test_no_false_rate_limit_claim():
    assert "rate_limit" not in CADDYFILE and "Per-IP rate limit" not in CADDYFILE

def test_bodies_and_timeouts_are_bounded():
    assert len(re.findall(r"request_body\s*\{\s*max_size\s+1MB\s*\}", CADDYFILE)) >= 2
    assert re.search(r"read_header\s+10s", CADDYFILE) and re.search(r"read_body\s+30s", CADDYFILE)
    assert CADDYFILE.count("response_header_timeout 30s") >= 2

@pytest.mark.integration
@pytest.mark.realinfra
def test_adapted_limits(caddy_adapt):
    text = json.dumps(caddy_adapt({"CADDY_API_SITE": "http://api.example.test", "CADDY_AUTH_SITE": "http://auth.example.test",
                                   "DOMAIN": "example.test", "CADDY_ADMIN_EMAIL": "ops@example.test",
                                   "BASIC_AUTH_HASH": "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"}))
    assert '"max_size": 1000000' in text or '"max_size": 1048576' in text
    assert '"read_header_timeout"' in text and '"response_header_timeout"' in text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_115_caddy_limits.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_115_caddy_limits.py`
Expected: FAIL: the comment at Caddyfile:42-45 still promises a built-in per-IP rate limit and there is no `request_body`, server timeout or transport timeout anywhere (MVP-115).
- [ ] **Step 3: Implement** the Caddyfile edits. Approach: limits are set once per site through a `(limits)` snippet so the auth and API sites cannot drift.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_250_caddy_edge.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_042_notification_env.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.13: Container confinement and autoheal labels
**Covers:** MVP-354
**Severity:** low
**Wave:** 4
**Depends on:** WS-03.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (changed 2026-10-19: strict secret files are 0400 and owned by the uid that reads them (WS-03.8), so every root secret reader listed in `ROOT_READERS_WITH_DAC_OVERRIDE` keeps `DAC_OVERRIDE` under `cap_drop: [ALL]`)
**Real infra:** yes (strict boot smoke under confinement)
**Cross-workstream prerequisites:** WS-04.13 consumes the `solace.autoheal` labels; WS-04 confines its own fragment services.
**Files:**
- Modify: `docker-compose.yml` (anchor `x-hardening` = `cap_drop: [ALL]`, `security_opt: ["no-new-privileges:true"]`, `pids_limit: 256`; the nine app services and `migrate` use it plus `read_only: true` and `tmpfs: ["/tmp"]`; the nine app services get `labels: {solace.autoheal: "true"}`; infra services get the anchor with only the capabilities their entrypoints need: postgres and redis `cap_add: [CHOWN, DAC_OVERRIDE, FOWNER, SETGID, SETUID]`, caddy `cap_add: [NET_BIND_SERVICE]` with `read_only: true` and tmpfs `/tmp`, weaviate (profile `vector`) `cap_add: [DAC_OVERRIDE]` only, because its root wrapper reads `WEAVIATE_API_KEY`, a 0400 file owned by memory-service's uid in strict (WS-03.8, D-07), kafka/zookeeper/grafana/prometheus/jaeger no added capabilities)
- Modify: `docker-compose.prod.yml` (`deploy.resources.limits.cpus` on every service (app services 1.0, kafka 1.0, postgres 2.0, others 0.5) and memory limits on the services that lacked them: notification, analytics, zookeeper, grafana, jaeger, prometheus)
- Modify: `docker-compose.backup.yml` (the same anchor for `pg-backup` and `pg-restore`, with `CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `SETGID`, `SETUID`)
- Test: `tests/deploy/test_mvp_354_confinement.py`
**Interfaces:**
- Consumes: `render`.
- Produces: label `solace.autoheal: "true"` on the nine app services, whose healthchecks already probe `/live` (WS-03.2), consumed by WS-04.13 (request from WS-04, which also lists WS-04.12 for the W4 bundle); the confinement baseline per service.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_354_confinement.py`
```python
"""MVP-354: every strict container drops capabilities, forbids privilege escalation and has pids/CPU limits;
app containers run read-only; Caddy holds no NET_RAW."""
import pytest
from deploy.env.registry import APP_SERVICES

ALLOWED_CAPS = {"postgres": {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"},
                "redis": {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"},
                "pg-backup": {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"},
                "pg-restore": {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"},
                "caddy": {"NET_BIND_SERVICE"}, "weaviate": {"DAC_OVERRIDE"}}

def test_every_strict_container_is_confined(render):
    r = render("strict", fragments=["docker-compose.backup.yml"], profiles=["restore"])
    for name, svc in r.services.items():
        assert svc.get("cap_drop") == ["ALL"], name
        assert "no-new-privileges:true" in svc.get("security_opt", []), name
        assert svc.get("pids_limit"), name
        assert set(svc.get("cap_add", [])) <= ALLOWED_CAPS.get(name, set()), name
        assert svc.get("deploy", {}).get("resources", {}).get("limits", {}).get("cpus"), name

def test_app_containers_are_read_only_and_autohealed(render):
    r = render("strict")
    for name in APP_SERVICES:
        svc = r.services[name]
        assert svc.get("read_only") is True and "/tmp" in " ".join(svc.get("tmpfs", [])), name
        assert svc["labels"]["solace.autoheal"] == "true", name

def test_root_secret_readers_keep_dac_override(render):
    """D-07: strict secret files are 0400 and owned by their non-root reader (WS-03.8), so a root wrapper that also
    reads one must keep DAC_OVERRIDE after cap_drop ALL, or the confined container cannot start."""
    from deploy.env.registry import ROOT_READERS_WITH_DAC_OVERRIDE
    r = render("strict", fragments=["docker-compose.backup.yml"], profiles=["restore", "vector"])
    for name in sorted(ROOT_READERS_WITH_DAC_OVERRIDE & set(r.services)):
        assert "DAC_OVERRIDE" in r.services[name].get("cap_add", []), name
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_354_confinement.py`
Expected: FAIL: no service sets `cap_drop`, `security_opt`, `read_only`, `pids_limit` or `cpus` (docker-compose.prod.yml:59-282, MVP-354), no autoheal label exists, and `test_root_secret_readers_keep_dac_override` fails on `pg-backup` (no `cap_add` anywhere yet).
- [ ] **Step 3: Implement** the compose edits listed under Files. Approach: one anchor merged with `<<: *hardening` per service so a new service cannot be added unconfined without the test noticing; writable paths are only volumes and tmpfs.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE (every service still reaches `/ready` 200 under confinement); `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_restore.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.14: RS256 user-token keys and per-service service-token keys provisioned as files
**Covers:** none (provisioning for IF-04's W4 key model; supports MVP-209, owned by WS-07)
**Severity:** medium
**Wave:** 4
**Depends on:** WS-03.13, WS-07.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: from W4 user tokens are RS256-only and every service signs service tokens with its own RS256 key, kid = service name), D-07 (changed 2026-10-19: in strict the private keys live only in the environment's SOPS file; the operator writes them with `scripts/generate_jwt_keys.py --out-dir` into a temporary owner-only directory, encrypts them into `deploy/secrets/<env>.sops.yaml` and removes the directory, and `render_secrets` writes one 0400 file per source under `/run/secrets`; the NAME_FILE contract is unchanged)
**Real infra:** yes (strict boot smoke with RS256 keys)
**Cross-workstream prerequisites:** WS-07's W4 code reads `AUTH_ALGORITHM`, `AUTH_ACCEPT_LEGACY_HS256`, `AUTH_PUBLIC_KEYS_JSON`, `AUTH_ACTIVE_KID`, `AUTH_PRIVATE_KEY_PEM`, `SERVICE_TOKEN_PRIVATE_KEY_PEM` and `SERVICE_TOKEN_PUBLIC_KEYS_JSON` (IF-02/IF-04). This task also lands every W4 env, route or service request other plans address to WS-03 (same procedure as WS-03.6 Step 3), except those that have their own tasks: WS-01's digest pins (WS-03.17), WS-05's backup CLI (WS-03.16), and WS-04's textfile metric and autoheal labels (WS-03.9, WS-03.13).
**Files:**
- Modify: `deploy/env/env-contract.yaml` (strict `AUTH_ALGORITHM=RS256`, `AUTH_ACCEPT_LEGACY_HS256=false`; `AUTH_PUBLIC_KEYS_JSON` and `SERVICE_TOKEN_PUBLIC_KEYS_JSON` (derived, non-secret, all nine); `AUTH_ACTIVE_KID`; `AUTH_PRIVATE_KEY_PEM` (secret, user-service only); one secret row per service with source `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>` mounted as `/run/secrets/SERVICE_TOKEN_PRIVATE_KEY_PEM` for that service only)
- Modify: `deploy/env/generate_dev_env.py` and `deploy/render_secrets.py` (RSA keypairs through `solace_security.keys.generate_rsa_keypair(kid)`; derived `AUTH_PUBLIC_KEYS_JSON = {kid: public_pem}` and `SERVICE_TOKEN_PUBLIC_KEYS_JSON = {service: public_pem}`; `render_secrets` derives the public JSON from stored private keys)
- Modify: `docker-compose.yml`, `docker-compose.prod.yml` (the mounts and `_FILE` variables above), `.env.example` (RENDER)
- Modify: `docs/runbooks/secrets.md` (signing keys: `python scripts/generate_jwt_keys.py --out-dir <temporary owner-only dir> --services all`, then encrypt every written file into the environment's SOPS file under its source name (`AUTH_PRIVATE_KEY_PEM`, `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>`), remove the directory, re-render; rotation with `--rotate`; no key in `.env.prod` or any plaintext file on the host)
- Test: `tests/deploy/test_mvp_209_key_provisioning.py`
**Interfaces:**
- Consumes: `solace_security.keys.generate_rsa_keypair(kid: str, key_size: int = 2048) -> tuple[str, str, str]`; WS-07.10's `scripts/generate_jwt_keys.py --out-dir DIR --services all|<names> [--rotate]`.
- Produces: secret sources `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service-name>` (New interfaces); derived public keysets. This lands the W4 part of WS-07's row for WS-07.10 (per-service `SERVICE_TOKEN_PRIVATE_KEY_PEM` files in `/run/secrets`, rendered by `deploy/render_secrets.py` from `deploy/secrets/<env>.sops.yaml`); its W1 part (`REDIS_URL`, `TRUSTED_PROXY_HOPS`, `RATE_LIMIT_OVERRIDES_JSON`) is landed by WS-03.6's W1 sync.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_209_key_provisioning.py`
```python
"""Supports MVP-209: each container holds only its own signing key; only user-service holds the user-token issuer key;
strict rejects legacy HS256; the public keysets match the private keys."""
import json
from cryptography.hazmat.primitives import serialization
from deploy.env.generate_dev_env import generate
from deploy.env.registry import APP_SERVICES
from tests.deploy.conftest import REPO

def test_runbook_puts_generated_signing_keys_only_into_the_sops_file():
    text = (REPO / "docs/runbooks/secrets.md").read_text(encoding="utf-8")
    assert "generate_jwt_keys.py --out-dir" in text and "SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>" in text
    assert "sops.yaml" in text and ".env.prod" not in text

def test_each_service_mounts_only_its_own_signing_key(render):
    r = render("strict")
    for name in APP_SERVICES:
        sources = {s["source"] for s in r.services[name]["secrets"]}
        own = f"SERVICE_TOKEN_PRIVATE_KEY_PEM.{name}"
        assert own in sources, name
        assert not {s for s in sources if s.startswith("SERVICE_TOKEN_PRIVATE_KEY_PEM.") and s != own}, name
        assert ("AUTH_PRIVATE_KEY_PEM" in sources) == (name == "user-service"), name
        env = r.services[name]["environment"]
        assert env["AUTH_ACCEPT_LEGACY_HS256"] == "false" and env["AUTH_ALGORITHM"] == "RS256", name

def test_generated_public_keysets_match_private_keys(tmp_path):
    generate(tmp_path, with_tls=False)
    keyset = json.loads((tmp_path / "SERVICE_TOKEN_PUBLIC_KEYS_JSON").read_text(encoding="utf-8"))
    assert set(keyset) == set(APP_SERVICES)
    for name in APP_SERVICES:
        private = serialization.load_pem_private_key((tmp_path / f"SERVICE_TOKEN_PRIVATE_KEY_PEM.{name}").read_bytes(), password=None)
        public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        assert keyset[name].strip() == public_pem.strip(), name
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_209_key_provisioning.py`
Expected: FAIL: no `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>` secret is mounted and `KeyError: 'AUTH_ACCEPT_LEGACY_HS256'` (all ten containers share the HS256 secret today, MVP-209); the generator writes no keyset; the secrets runbook (WS-03.8) has no signing-key section, so the runbook test fails on `generate_jwt_keys.py --out-dir`.
- [ ] **Step 3: Implement** the registry, generator, renderer and compose edits listed under Files, then the W4 request procedure. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.15: Docs that match the system, and retirement of legacy env names
**Covers:** MVP-242, MVP-308
**Severity:** low
**Wave:** 5
**Depends on:** WS-03.14, WS-03.11, WS-03.12, WS-03.16, WS-03.17, WS-03.18, WS-04.13, WS-04.14
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-25 (changed 2026-10-19: the release gate runs on the AWS staging host from `solace-staging.yaml`, reached through SSM, with the strict production profile; provisioning needs the owner to attach `solace-staging-deployer-policy.json` to the existing user `MedCodex-Local`; if the policy is still unattached at W5 entry the gate stops for the owner, with no silent fallback to a local host), D-07 (changed 2026-10-19: the docs describe SOPS + age, never a secrets store we do not use)
**Real infra:** no
**Cross-workstream prerequisites:** WS-12 updates the `/metrics` section of `services/safety_service/README.md` to describe the runtime-served endpoint (requested below); every lane has stopped reading the legacy names it owned (the scanner proves it).
**Files:**
- Modify: `README.md` (replaced: what Solace-AI is today, the ten services and shared packages, dev quickstart (`python -m deploy.env.generate_dev_env`, `docker compose up -d --build`, the dev edge at `http://api.localhost:8088`), test commands, links to the runbooks and the register; the Streamlit/Gemini content removed)
- Modify: `docs/SYSTEM-DESIGN-SUMMARY.md` (status table and correction stamp rewritten to the W5 state: RS256-only user tokens and per-service keys, Prometheus rules plus Alertmanager, CI build and boot gates, Weaviate opt-in; design-only items (Kong, Istio, Kubernetes, mTLS, Vault) labelled design intent)
- Modify: `docs/STAGING-BRINGUP.md` (migrations run by the compose `migrate` job to `SCHEMA_HEAD`; health checked with `docker compose ps` and `docker compose exec <svc> python -c ...` because strict publishes no internal ports; secrets via the SOPS render (`render_secrets` in the `solace/sops-tools` container, WS-03.8); backups via the fragment; the LLM settings of WS-09's CR-4 and CR-14 (D-04): `LLM_PROVIDER=bedrock` (the strict literal), `BEDROCK_REGION`, `BEDROCK_MODEL_ID` chosen with `aws bedrock list-foundation-models` and the inference profiles, never guessed, `LLM_PHI_EGRESS_APPROVED=false` until the owner accepts the AWS BAA in AWS Artifact and it is recorded, credentials only from the instance role; no `PORTKEY_API_KEY` line)
- Modify: `docs/KNOWN-LIMITATIONS.md` (sections 3.2, 3.4, 3.5, 3.7, 4.2 and 8 updated to the delivered controls, with remaining limits stated plainly; a D-29 entry for WS-12's R-12-12: crisis resources are a per-locale registry in `solace_clinical` with only en-US populated, any other or unknown locale is shown the generic fallback (local emergency number plus the Find A Helpline directory) and fires `CrisisResourcesLocaleFallback`, and admitting a non-US user is an operator and cohort decision (D-31); any "US-only" or "non-US users are outside the pilot" wording is replaced)
- Modify: `system-design/README.md` (banner: design intent, not implementation status; link to the register)
- Create: `docs/runbooks/deploy.md` (bring-up, upgrade, rollback with the previous image tag and a pre-change base backup, teardown preserving the data volume; the release gate on the AWS staging host through `aws ssm start-session` with the strict production profile, its prerequisite that the owner has attached `solace-staging-deployer-policy.json` to `MedCodex-Local`, and the stop-for-owner rule when it is unattached at W5 entry (D-25))
- Modify: `deploy/env/env-contract.yaml` (`retired: true` for every legacy name with zero readers), `docker-compose.yml`, `docker-compose.prod.yml` (retired names no longer set), `.env.example` (RENDER), `tests/deploy/conftest.py` (`LEGACY_RENDER_SOURCES` removed)
- Test: `tests/deploy/test_mvp_242_docs_truth.py`
**Interfaces:**
- Consumes: `collect_env_readers`; the final registry; WS-12.2's per-locale registry and fallback names (`crisis_resources`, `FALLBACK_RESOURCES_VERSION`) and WS-04's `CrisisResourcesLocaleFallback` rule (WS-12's R-12-11).
- Produces: no new interface. It lands WS-12's R-12-12 (the D-29 limitations entry) for WS-12.2 and the STAGING-BRINGUP half of WS-09's CR-4 and CR-14 for WS-09.4, WS-09.5 and WS-09.18; those W1 tasks take no dependency on this W5 task.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_242_docs_truth.py`
```python
"""MVP-242/MVP-308: the README, design summary, bring-up runbook and limitations describe the delivered system."""
import re
from pathlib import Path
import pytest
from tests.deploy.conftest import REPO

DOCS = ["README.md", "docs/SYSTEM-DESIGN-SUMMARY.md", "docs/STAGING-BRINGUP.md", "docs/KNOWN-LIMITATIONS.md", "docs/runbooks/deploy.md"]

def _read(p):
    return (REPO / p).read_text(encoding="utf-8")

def test_readme_describes_the_microservices_not_the_legacy_app():
    text = _read("README.md")
    assert "Streamlit" not in text and "launch_api.py" not in text
    assert "deploy.env.generate_dev_env" in text and "docker compose" in text

def test_bringup_uses_the_migrate_job_and_no_host_ports():
    text = _read("docs/STAGING-BRINGUP.md")
    assert not re.search(r"001\s*(\.\.|→|->)\s*006", text)
    assert "migrate" in text and "localhost:5432" not in text
    assert not re.search(r"for p in 8000 8001", text)

def test_design_summary_claims_match_delivery():
    text = _read("docs/SYSTEM-DESIGN-SUMMARY.md")
    assert "No integration tests or Docker build in pipeline" not in text
    assert "alerting rules all coded" not in text
    assert "RS256" in text and "AUTH_ACCEPT_LEGACY_HS256" in text

def test_limitations_no_longer_say_secrets_live_in_env_prod():
    assert "Secrets live in `.env.prod`" not in _read("docs/KNOWN-LIMITATIONS.md")

def test_limitations_record_the_d29_locale_registry_and_fallback():
    text = _read("docs/KNOWN-LIMITATIONS.md")
    for needle in ("D-29", "en-US", "Find A Helpline", "CrisisResourcesLocaleFallback", "D-31"):
        assert needle in text, needle
    assert not re.search(r"US-only|outside the pilot", text, re.I)

def test_bringup_documents_the_bedrock_route_and_no_portkey_key():
    text = _read("docs/STAGING-BRINGUP.md")
    for needle in ("LLM_PROVIDER=bedrock", "BEDROCK_MODEL_ID", "list-foundation-models", "LLM_PHI_EGRESS_APPROVED",
                   "AWS Artifact"):
        assert needle in text, needle
    assert not re.search(r"^\s*PORTKEY_API_KEY\s*=", text, re.M)

def test_deploy_runbook_puts_the_release_gate_on_the_aws_host():
    text = _read("docs/runbooks/deploy.md")
    assert "MedCodex-Local" in text and "solace-staging-deployer-policy.json" in text
    assert "aws ssm start-session" in text and "release gate" in text

@pytest.mark.parametrize("doc", DOCS + ["infrastructure/aws-staging/DEPLOY.md", "docs/runbooks/secrets.md"])
def test_docs_name_the_secrets_mechanism_in_use(doc):
    assert not re.search(r"aws secretsmanager|get-secret-value|ssm get-parameter", _read(doc), re.I), doc
    if doc in ("docs/STAGING-BRINGUP.md", "infrastructure/aws-staging/DEPLOY.md"):
        assert "sops" in _read(doc).lower(), doc

@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_resolve(doc):
    base = (REPO / doc).parent
    for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", _read(doc)):
        assert (base / target).exists(), (doc, target)

def test_no_unretired_legacy_name_is_unread(registry):
    from deploy.env.env_readers import collect_env_readers
    readers = collect_env_readers([REPO / "src", REPO / "services"])
    stale = [v.name for v in registry if v.group == "legacy" and not v.retired and v.name not in readers]
    assert stale == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_242_docs_truth.py`
Expected: FAIL: README.md describes the Streamlit/Gemini monolith (README.md:1-8, MVP-242); STAGING-BRINGUP.md says `001→006` and curls `localhost:8000-8009` (:96-107, MVP-308); SYSTEM-DESIGN-SUMMARY.md still carries "No integration tests or Docker build in pipeline" and "alerting rules all coded" (:39-41); KNOWN-LIMITATIONS.md:83 says secrets live in `.env.prod` and has no D-29 entry (no per-locale registry, fallback or `CrisisResourcesLocaleFallback` line); STAGING-BRINGUP.md names no `LLM_PROVIDER=bedrock` or `BEDROCK_MODEL_ID`; `docs/runbooks/deploy.md` does not exist, so `test_deploy_runbook_puts_the_release_gate_on_the_aws_host` and the `deploy.md` cases fail (the other `test_docs_name_the_secrets_mechanism_in_use` cases already pass after WS-03.8's rewrite and must stay green); legacy rows whose readers are gone are not yet retired.
- [ ] **Step 3: Implement** the doc rewrites and the legacy retirement. Approach: every factual claim in the rewritten docs points at the file or test that proves it; retire only names the scanner reports with zero readers, and remove them from the compose files in the same edit. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.16: Backup CLI for operators and drills: backup-now, restore-latest, status and the age-metric drill hook
**Covers:** none (operator and drill interface over MVP-037's backups from WS-03.9; requested by WS-05 for the MVP-039 restore drill)
**Severity:** high
**Wave:** 4
**Depends on:** WS-03.13
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-09 (accepted: RPO 15 min, RTO 4 h; the freshness thresholds are WS-04.12's `BackupStale` 93600 s and `WalArchiveStale` 1200 s), D-25 (changed 2026-10-19: until the owner attaches the deployer policy the W4 drill runs against the local encrypted-volume target and must say so, so `status` reports the target)
**Real infra:** yes (postgres plus the backup fragment: backup-now, status, destroy, restore-latest, age-metric)
**Cross-workstream prerequisites:** WS-05.17 (`scripts/test/restore_drill.py`) and WS-05.18 (the `BackupStale` fault test) call these commands as `docker compose ... run --rm backup <cmd>` (request 6 of WS-05, landed here); WS-04.12's rules read the metric that `age-metric` ages.
**Files:**
- Create: `deploy/backup/solace-backup-cli.sh` (POSIX sh dispatcher; unknown command or bad argument exits 64 with a usage line)
- Modify: `deploy/backup/Dockerfile` (copies the CLI to `/usr/local/bin/solace-backup-cli`)
- Modify: `deploy/backup/solace-backup.sh` (base backups from the schedule, `base-now` and the CLI serialize on an atomic `mkdir /backup-target/.base.lock`, removed by an EXIT trap; a lock older than 21600 s is logged and broken)
- Modify: `docker-compose.backup.yml` (service `backup`: the `solace/pg-backup:${SOLACE_IMAGE_TAG:-dev}` image without `build`, `profiles: ["ops"]`, `entrypoint: ["solace-backup-cli"]`, no `depends_on` (so `run` never starts postgres) and no ports, network `data`, the union of the `pg-backup` and `pg-restore` volumes (`postgres_data` read-write, `pg_wal_restore`, `pg_backup_work`, `solace_textfile_metrics`, the target) and secrets, the fragment's `x-hardening` anchor with `CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `SETGID`, `SETUID`, and the same CPU and memory limits as `pg-backup`)
- Modify: `docs/runbooks/backup-restore.md` (the four commands, their exit codes, and the drill sequence WS-05 automates)
- Modify: `tests/deploy/conftest.py` (`compose_project.run(service, *args, profiles=(), check=True)` passes `args` to `docker compose run --rm <service>`; new `stop(service)`)
- Test: `tests/deploy/test_mvp_037_backup_cli.py`
**Interfaces:**
- Consumes: `solace-backup base-now` and `solace-restore` (WS-03.9, WS-03.10), the textfile metric `solace_backup_last_success_timestamp_seconds{store}` (WS-03.9).
- Produces: for WS-05.17 and WS-05.18 (New interfaces), `docker compose ... run --rm backup <cmd>` with every timestamp printed as ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`):
  - `backup-now`: waits up to 60 s for `pg_isready`, takes a base backup, prints `backup_id=<id> completed_at=<ts>`; exit 0, or 1 on any failure.
  - `restore-latest`: restores the newest `base/<id>` and every archived WAL after it by calling `solace-restore --backup-id <id>`, prints `restore_seconds=<n> backup_id=<id> recovered_to=<ts>`; exit 0; 1 on failure or when no backup exists; 2 while postgres is running; 3 on a key-material mismatch (WS-03.10).
  - `status`: prints `postgres_base_last_success=<ts|none>`, `postgres_wal_last_success=<ts|none>`, `latest_backup_id=<id|none>` and `backup_target=<local|s3>` (`s3` when `BACKUP_S3_URI` is set, otherwise the local encrypted-volume target; D-25), one per line; exit 0 when the base backup is at most 93600 s old and WAL at most 1200 s old, otherwise 1.
  - `age-metric <seconds>`: the drill hook. It rewrites `/textfile/solace_backup.prom` atomically with each store's timestamp set to `min(current, now - seconds)`, so it can only make a store look older (it can fire `BackupStale`, never hide it), and prints `postgres_base_last_success=<ts> postgres_wal_last_success=<ts>`; exit 0.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_037_backup_cli.py`
```python
"""Backup CLI (WS-05 request 6): backup-now, restore-latest, status and age-metric run as
`docker compose run --rm backup <cmd>`, print ISO-8601 UTC timestamps, use fixed exit codes,
and the drill hook can only age the backup metric, never refresh it."""
import re
import time
import pytest

ISO = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
CAPS = {"CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"}

def _base_ts(p):
    return float(re.search(r'store="postgres_base"\}\s+(\d+(?:\.\d+)?)', p.textfile("solace_backup.prom")).group(1))

def test_backup_cli_is_on_demand_never_starts_postgres_and_is_confined(render):
    svc = render("strict", fragments=["docker-compose.backup.yml"], profiles=["ops"]).services["backup"]
    assert svc["entrypoint"] == ["solace-backup-cli"]
    assert not svc.get("depends_on") and not svc.get("ports")
    assert svc["cap_drop"] == ["ALL"] and set(svc.get("cap_add", [])) <= CAPS
    assert "no-new-privileges:true" in svc["security_opt"] and svc["deploy"]["resources"]["limits"]["cpus"]
    assert "backup" not in render("strict", fragments=["docker-compose.backup.yml"]).services

@pytest.mark.integration
@pytest.mark.e2e
def test_backup_now_status_and_restore_latest(compose_project, tmp_path):
    target = tmp_path / "t"
    target.mkdir()
    p = compose_project.with_fragments(["docker-compose.backup.yml"], env={"BACKUP_TARGET_DIR": str(target)})
    p.up(["postgres", "pg-backup"])
    p.psql("CREATE TABLE cli_drill(id int primary key)")
    p.psql("INSERT INTO cli_drill SELECT generate_series(1, 50)")
    now = p.run("backup", "backup-now")
    assert now.returncode == 0 and re.search(rf"^backup_id=\S+ completed_at={ISO}$", now.stdout, re.M)
    seg = p.psql("SELECT pg_walfile_name(pg_switch_wal())")[0][0]
    deadline = time.monotonic() + 120
    while not (target / "wal" / f"{seg}.gpg").exists():
        assert time.monotonic() < deadline, "WAL segment never reached the backup target"
        time.sleep(2)
    deadline = time.monotonic() + 60  # the postgres_wal stamp follows the push in the same agent cycle
    while (st := p.run("backup", "status", check=False)).returncode != 0:
        assert time.monotonic() < deadline, st.stdout
        time.sleep(3)
    assert re.search(rf"^postgres_base_last_success={ISO}$", st.stdout, re.M)
    assert re.search(rf"^postgres_wal_last_success={ISO}$", st.stdout, re.M)
    assert re.search(r"^backup_target=local$", st.stdout, re.M)  # no BACKUP_S3_URI: the drill must say so (D-25)
    assert p.run("backup", "restore-latest", check=False).returncode == 2  # postgres still running
    p.destroy_volume("postgres", "postgres_data")
    res = p.run("backup", "restore-latest")
    assert res.returncode == 0 and re.search(rf"recovered_to={ISO}", res.stdout)
    p.up(["postgres"])
    assert p.psql("SELECT count(*) FROM cli_drill") == [(50,)]

@pytest.mark.integration
@pytest.mark.e2e
def test_age_metric_only_makes_the_metric_older(compose_project, tmp_path):
    target = tmp_path / "t"
    target.mkdir()
    p = compose_project.with_fragments(["docker-compose.backup.yml"], env={"BACKUP_TARGET_DIR": str(target)})
    p.up(["postgres", "pg-backup"])
    assert p.run("backup", "backup-now").returncode == 0
    p.stop("pg-backup")
    assert p.run("backup", "age-metric", "100000").returncode == 0
    aged = _base_ts(p)
    assert time.time() - aged >= 100000
    assert p.run("backup", "age-metric", "10").returncode == 0
    assert _base_ts(p) == aged
    assert p.run("backup", "status", check=False).returncode == 1
    assert p.run("backup", "age-metric", "soon", check=False).returncode == 64
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_cli.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_cli.py`
Expected: FAIL. `KeyError: 'backup'`: the fragment from WS-03.9 has only `pg-backup` and `pg-restore`, so WS-05's `run --rm backup <cmd>` has no target, no `status` command exists, and nothing can age the metric for the `BackupStale` drill. The docker tests fail at the first `p.run("backup", ...)` with compose's `no such service: backup`.
- [ ] **Step 3: Implement** the CLI, the lock, the service and the runbook section. Approach: `restore-latest` picks the newest `base/*/manifest.json` by `backup_id` (UTC `YYYYMMDDTHHMMSSZ` ids sort by time) and execs `solace-restore` without `--wipe`, so the exit-2 and exit-3 refusals are the restore script's own and a non-empty data directory (for example one a starting postgres has just initialised, before `pg_isready` answers) is refused, never overwritten; `status` and `age-metric` read and write only the textfile metric and list `base/` (no database access), and `status` derives `backup_target` from `BACKUP_S3_URI` alone; `age-metric` writes `.prom.tmp` then renames, like the agent. No command prints a secret, a key id or a row value.
- [ ] **Step 4: Run it and confirm it passes**
Run: the two commands of Step 2. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_037_backup_restore.py`; `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_350_key_escrow.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.17: Third-party images pinned by digest, the postgres image equal to POSTGRES_IMAGE
**Covers:** none (the compose half of MVP-355, which WS-01.10 owns and gates)
**Severity:** medium
**Wave:** 4
**Depends on:** WS-03.14, WS-03.16, WS-02.1
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** yes (every pinned reference is pulled by digest, then BOOTSMOKE)
**Cross-workstream prerequisites:** WS-02.1's `scripts.db.images.POSTGRES_IMAGE` (W0); WS-01.10's `test_mvp_355_compose_images_are_digest_pinned_and_app_images_tagged` reads the two compose files this task pins (request R-01.1, landed here).
**Files:**
- Modify: `docker-compose.yml`, `docker-compose.prod.yml` (every `image:` of a service without `build` becomes `<name>:<tag>@sha256:<digest>`: postgres, redis, zookeeper, kafka, weaviate, caddy, prometheus, grafana, jaeger, mailhog, kafka-ui; postgres uses exactly `scripts.db.images.POSTGRES_IMAGE`)
- Modify: `docker-compose.backup.yml` (`pg-restore` and `backup` reference the built `solace/pg-backup:${SOLACE_IMAGE_TAG:-dev}` image; no other image), `deploy/backup/Dockerfile` (`ARG POSTGRES_IMAGE=<the POSTGRES_IMAGE value>` and `FROM ${POSTGRES_IMAGE}`)
- Test: `tests/deploy/test_mvp_355_image_pins.py`
**Interfaces:**
- Consumes: `scripts.db.images.POSTGRES_IMAGE` (WS-02.1).
- Produces: digest-pinned third-party images in `docker-compose.yml` and `docker-compose.prod.yml`, and built images tagged `solace/<service>:${SOLACE_IMAGE_TAG:-dev}` (from WS-03.2), consumed by WS-01.10's supply-chain gate and image scan.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_355_image_pins.py`
```python
"""MVP-355 (compose half): every third-party image is pinned by digest, built images are tagged
solace/<service>:${SOLACE_IMAGE_TAG}, and postgres everywhere is the WS-02 POSTGRES_IMAGE."""
import re
import pytest
import yaml
from scripts.db.images import POSTGRES_IMAGE
from tests.deploy.conftest import REPO

FILES = ("docker-compose.yml", "docker-compose.prod.yml", "docker-compose.backup.yml")
PINNED = re.compile(r"^[a-z0-9./-]+:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$")

def _services(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8")).get("services", {}) or {}

def test_every_image_is_digest_pinned_or_built_here():
    built = {spec["image"] for f in FILES for spec in _services(f).values() if "build" in spec}
    offenders = []
    for f in FILES:
        for svc, spec in _services(f).items():
            image = spec.get("image")
            if "build" in spec:
                ok = image == f"solace/{svc}:${{SOLACE_IMAGE_TAG:-dev}}"
            elif image and image.startswith("solace/"):
                ok = image in built
            else:
                ok = image is None or bool(PINNED.match(image))
            if not ok:
                offenders.append(f"{f}:{svc}={image}")
    assert offenders == []

def test_postgres_is_the_platform_image():
    assert _services("docker-compose.yml")["postgres"]["image"] == POSTGRES_IMAGE
    assert _services("docker-compose.prod.yml").get("postgres", {}).get("image", POSTGRES_IMAGE) == POSTGRES_IMAGE
    dockerfile = (REPO / "deploy/backup/Dockerfile").read_text(encoding="utf-8")
    assert f"ARG POSTGRES_IMAGE={POSTGRES_IMAGE}" in dockerfile and "FROM ${POSTGRES_IMAGE}" in dockerfile

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_rendered_images_have_no_floating_tag(render, profile):
    r = render(profile, fragments=["docker-compose.backup.yml"], profiles=["vector"])
    assert [s for s, d in r.services.items() if d.get("image", "").endswith(":latest")] == []

@pytest.mark.integration
@pytest.mark.realinfra
def test_every_pinned_reference_pulls(render):
    refs = sorted({d["image"] for prof in ("dev", "strict")
                   for d in render(prof, fragments=["docker-compose.backup.yml"], profiles=["vector"]).services.values()
                   if PINNED.match(d.get("image", ""))})
    assert refs
    for ref in refs:
        out = subprocess.run(["docker", "pull", "-q", ref], capture_output=True, text=True, timeout=900)
        assert out.returncode == 0, (ref, out.stderr[-500:])
```
(`import subprocess` joins the imports.)
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_355_image_pins.py`
Expected: FAIL. `test_every_image_is_digest_pinned_or_built_here` lists every infra service, starting with `docker-compose.yml:postgres=postgres:16-alpine` (docker-compose.yml:18) and `redis=redis:7-alpine` (:39), because tags float (MVP-355); `test_postgres_is_the_platform_image` fails on the same line and on the backup Dockerfile's unpinned `FROM postgres:16-alpine` (WS-03.9).
- [ ] **Step 3: Implement** the pins. Resolve each digest without changing the tag: `docker image inspect --format "{{index .RepoDigests 0}}" <name>:<tag>` when the image is present locally, otherwise `docker buildx imagetools inspect <name>:<tag>` (a read-only registry query; take the index digest so one pin serves the amd64 Linux host and Docker Desktop). Record each `<name>:<tag> -> digest` pair in the ledger. If the controller rules registry queries out, stop and raise a ledger Ruling instead of pinning a guessed digest.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command, then `DOCKER D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_355_image_pins.py`. Expected: PASS, and every pinned reference pulls.
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE; `UNIT D:\Repo\Contextual-Chatbot\tests\build\test_mvp_113_supply_chain.py` (its compose case turns green here). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.18: Strict brokers never auto-create topics
**Covers:** none (the strict-overlay half of MVP-269, which WS-01.12 closes)
**Severity:** low
**Wave:** 5
**Depends on:** WS-03.17
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-24 (default assumed: `solace.safety` and `solace.users` retain 7 days, applied by the topic registry, not by broker defaults)
**Real infra:** yes (strict BOOTSMOKE with auto-create off)
**Cross-workstream prerequisites:** WS-06 applies `solace_events.config.TOPIC_CONFIGS` create-if-missing at event-plane start (request R-01.4(b) from WS-01, repeated below). No WS-06 task does this yet. Entry check: the ledger records a completed WS-06 task for it. Without it, stop and raise a ledger Ruling: with auto-create off and nothing creating topics, strict producers and consumers would fail on unknown topics.
**Files:**
- Modify: `docker-compose.prod.yml` (kafka gains `environment: {KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"}`; the dev base keeps `"true"`, docker-compose.yml:91)
- Test: `tests/deploy/test_mvp_269_strict_topics.py`
**Interfaces:**
- Consumes: WS-06's create-if-missing at plane start (R-01.4(b)).
- Produces: the strict-overlay setting that WS-01.12's `test_mvp_269_strict_overlay_disables_topic_auto_create` reads (request R-01.1, landed here).

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_mvp_269_strict_topics.py`
```python
"""MVP-269 (strict overlay): strict brokers never auto-create topics, so every topic comes from the registry
with its D-24 retention; dev keeps auto-create for convenience."""
import yaml
from tests.deploy.conftest import REPO

def test_strict_overlay_disables_auto_create_as_written():
    env = yaml.safe_load((REPO / "docker-compose.prod.yml").read_text(encoding="utf-8"))["services"]["kafka"]["environment"]
    assert env["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"

def test_rendered_profiles(render):
    assert render("strict").services["kafka"]["environment"]["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"
    assert render("dev").services["kafka"]["environment"]["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "true"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_mvp_269_strict_topics.py`
Expected: FAIL. `KeyError: 'environment'`: the strict overlay's kafka block sets no environment (docker-compose.prod.yml:254), so strict inherits `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"` from docker-compose.yml:91 and the render test reads `'true'` (MVP-269: topics appear with broker defaults instead of the registry's retention).
- [ ] **Step 3: Implement** the overlay line after the entry check passes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE (the strict run shows every service `/ready` with the event plane up, which proves that the plane created its topics); `UNIT D:\Repo\Contextual-Chatbot\tests\build\test_mvp_269_topic_registry.py`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.19: LLM provider env contract: Bedrock only in strict, the fake-LLM endpoint only in dev and test, no static AWS keys
**Covers:** none (the IF-02 half of D-04's Bedrock routing; the provider code and the LLM register entries stay with WS-09)
**Severity:** high
**Wave:** 1
**Depends on:** WS-03.6, WS-09.18
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (changed 2026-10-19: real-user LLM traffic goes to Anthropic models on AWS Bedrock under the AWS BAA. Strict environments require `LLM_PROVIDER=bedrock`; `BEDROCK_MODEL_ID` has no default and is set at deploy after checking availability with `bedrock:ListFoundationModels` and the inference profiles, never guessed in code or tests, and it is required in strict once `LLM_PHI_EGRESS_APPROVED=true` (WS-09.4 makes approved-without-model-id a startup error), so a strict stack before the AWS BAA starts with the LLM off and needs no placeholder; `LLM_PROVIDER=portkey` stays allowed only in development and test with synthetic data; `BEDROCK_REGION` defaults to `us-east-1`, `BEDROCK_TIMEOUT_SECONDS` to `20`; `BEDROCK_ENDPOINT_URL` exists only for test and E2E against the fake LLM and is refused in strict; credentials come from the host's instance role (WS-03.20), never static AWS keys in env or secrets; LLM features stay off for real users until the owner accepts the AWS BAA in AWS Artifact and it is recorded; caching off, no semantic cache, memory's OpenAI embeddings and Weaviate writes off, unchanged)
**Real infra:** no (renders `docker compose config` only)
**Cross-workstream prerequisites:** WS-09.18 (the Bedrock provider in `UnifiedLLMClient`) reads exactly these names, takes credentials from the default boto3 chain and refuses `BEDROCK_ENDPOINT_URL` and `LLM_PROVIDER=portkey` in strict (requested below; this task depends on it); WS-05.1's fake LLM serves `POST /model/{model_id}/converse`, and WS-05.21 sets the e2e values, the endpoint only in the dev-only fragment `infrastructure/test/compose/llm-dev.yml` (WS-05 request 19); WS-01.2 pins `boto3` and `botocore` in the runtime lock (D-04).
**Files:**
- Modify: `deploy/env/registry.py` (`LLM_SERVICES: tuple[str, ...]`, the five services that call the LLM: orchestrator, safety, therapy, diagnosis and personality; `DENIED_NAMES: frozenset[str]`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` and `AWS_PROFILE`, which no registry row may name and no rendered container may receive)
- Modify: `deploy/env/env-contract.yaml` (rows on the five LLM services: `LLM_PROVIDER` (dev default `portkey`; strict the compose literal `bedrock`), `BEDROCK_REGION` (default `us-east-1`), `BEDROCK_MODEL_ID` (`required_in_strict: false` for the render, no dev or strict default, `example` the placeholder `REPLACE_AT_DEPLOY_FROM_LIST_FOUNDATION_MODELS`, which is not a model id; notes: chosen at deploy after checking availability with `bedrock:ListFoundationModels` and the inference profiles; required in strict once `LLM_PHI_EGRESS_APPROVED=true`, enforced at startup by WS-09.4), `BEDROCK_ENDPOINT_URL` (profiles `[dev]`, dev default empty; test and E2E point it at the fake LLM), `BEDROCK_TIMEOUT_SECONDS` (default `20`); `LLM_PORTKEY_API_KEY`, `LLM_PORTKEY_GATEWAY_URL` and `LLM_ANTHROPIC_API_KEY` get profiles `[dev]`, because Portkey and the direct Anthropic API are development and test paths with synthetic data only; `LLM_ENABLE_CACHING=false` and `LLM_PHI_EGRESS_APPROVED=false` unchanged), `.env.example` (RENDER)
- Modify: `docker-compose.yml` (the five LLM services: `LLM_PROVIDER: ${LLM_PROVIDER:-portkey}`, `BEDROCK_REGION: ${BEDROCK_REGION:-us-east-1}`, `BEDROCK_MODEL_ID: ${BEDROCK_MODEL_ID:-}`, `BEDROCK_ENDPOINT_URL: ${BEDROCK_ENDPOINT_URL:-}`, `BEDROCK_TIMEOUT_SECONDS: ${BEDROCK_TIMEOUT_SECONDS:-20}`)
- Modify: `docker-compose.prod.yml` (the five: `LLM_PROVIDER: bedrock` as a literal, so no operator value can select Portkey; `BEDROCK_MODEL_ID: ${BEDROCK_MODEL_ID:-}` (empty keeps the LLM off; with `LLM_PHI_EGRESS_APPROVED=true` an empty value fails startup in WS-09.4); `BEDROCK_ENDPOINT_URL`, `LLM_PORTKEY_API_KEY`, `LLM_PORTKEY_GATEWAY_URL` and `LLM_ANTHROPIC_API_KEY` removed with `!reset null`)
- Modify: `tests/deploy/test_env_contract.py` (`test_registry_entries_are_well_formed` also fails on any row whose `name` or `source` is in `DENIED_NAMES`)
- Test: `tests/deploy/test_d04_llm_provider_env.py`
**Interfaces:**
- Consumes: the `render` and `registry` fixtures and `ComposeRenderError` (WS-03.1).
- Produces: the IF-02 names `LLM_PROVIDER`, `BEDROCK_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_ENDPOINT_URL` and `BEDROCK_TIMEOUT_SECONDS` on the five LLM services, consumed by WS-09.18 and WS-05.21 (New interfaces); `deploy.env.registry.LLM_SERVICES` and `DENIED_NAMES`; the strict rules (Bedrock only, a deploy-time model id, no endpoint override, no Portkey or direct Anthropic key, no static AWS credentials) that WS-03.20's instance role completes. This lands WS-09's CR-14 for WS-09.18 and the LLM-provider half of CR-4 for WS-09.4 and WS-09.5 (WS-03.6's generic W1 sync leaves these rows to this task), and answers WS-05's request 19 for WS-05.21: `BEDROCK_ENDPOINT_URL` is a dev-profile row, so the dev render with WS-05.21's fragment is accepted, and the strict render never carries it, so `test_env_contract.py` has no allow-listed strict difference for it. WS-05.21 depends on this task.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_d04_llm_provider_env.py`
```python
"""D-04: strict LLM traffic goes only to Bedrock with a deploy-time model id; the fake-LLM endpoint and the Portkey and
direct Anthropic keys never reach a strict container; no container is given static AWS credentials."""
import re
import pytest
from deploy.env.registry import DENIED_NAMES, LLM_SERVICES
from tests.deploy.conftest import REPO

STATIC_AWS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"}
DEV_ONLY = ("BEDROCK_ENDPOINT_URL", "LLM_PORTKEY_API_KEY", "LLM_PORTKEY_GATEWAY_URL", "LLM_ANTHROPIC_API_KEY")

def test_llm_services_and_denied_names():
    assert set(LLM_SERVICES) == {"orchestrator-service", "safety-service", "therapy-service",
                                 "diagnosis-service", "personality-service"}
    assert STATIC_AWS <= set(DENIED_NAMES)

@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_routes_every_llm_service_to_bedrock(render, environment):
    r = render("strict", environment=environment,
               overrides={"LLM_PROVIDER": "portkey", "BEDROCK_ENDPOINT_URL": "http://fake-llm:8080",
                          "BEDROCK_MODEL_ID": "synthetic-model-id-not-real"})
    for svc in LLM_SERVICES:
        env = r.services[svc]["environment"]
        assert env["LLM_PROVIDER"] == "bedrock", svc
        assert env["BEDROCK_MODEL_ID"] == "synthetic-model-id-not-real", svc
        assert env["BEDROCK_REGION"] == "us-east-1" and env["BEDROCK_TIMEOUT_SECONDS"] == "20", svc
        assert not [k for n in DEV_ONLY for k in (n, f"{n}_FILE") if env.get(k)], svc

def test_strict_render_without_a_model_id_keeps_the_llm_off(render):
    r = render("strict", drop=["BEDROCK_MODEL_ID"])
    for svc in LLM_SERVICES:
        env = r.services[svc]["environment"]
        assert (env["BEDROCK_MODEL_ID"], env["LLM_PHI_EGRESS_APPROVED"]) == ("", "false"), svc

def test_dev_can_point_bedrock_at_the_fake_llm(render):
    r = render("dev", environment="development",
               overrides={"LLM_PROVIDER": "bedrock", "BEDROCK_ENDPOINT_URL": "http://fake-llm:8080"})
    for svc in LLM_SERVICES:
        env = r.services[svc]["environment"]
        assert (env["LLM_PROVIDER"], env["BEDROCK_ENDPOINT_URL"]) == ("bedrock", "http://fake-llm:8080"), svc

def test_dev_defaults_keep_portkey_and_no_endpoint(render):
    env = render("dev", environment="development",
                 drop=["LLM_PROVIDER", "BEDROCK_ENDPOINT_URL"]).services["therapy-service"]["environment"]
    assert env["LLM_PROVIDER"] == "portkey" and env["BEDROCK_ENDPOINT_URL"] == ""

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_no_container_is_given_static_aws_credentials(render, registry, profile):
    r = render(profile, overrides={n: "synthetic-not-a-key" for n in STATIC_AWS})
    leaked = {s: sorted(STATIC_AWS & set(d.get("environment") or {})) for s, d in r.services.items()}
    assert {s: v for s, v in leaked.items() if v} == {}
    assert not STATIC_AWS & ({v.name for v in registry} | {v.source for v in registry})
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    assert [n for n in STATIC_AWS if re.search(rf"^{n}=", example, re.M)] == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_d04_llm_provider_env.py`
Expected: FAIL. Collection fails with `ImportError: cannot import name 'DENIED_NAMES'` (and `LLM_SERVICES`) from `deploy.env.registry`; with those two constants added, `test_strict_routes_every_llm_service_to_bedrock` fails with `KeyError: 'LLM_PROVIDER'` for `orchestrator-service`, because nothing selects a provider and every LLM service still reaches Portkey through the `LLM_PORTKEY_*` rows in strict too (D-04 changed the strict path to Bedrock), `test_strict_render_without_a_model_id_keeps_the_llm_off` fails with `KeyError: 'BEDROCK_MODEL_ID'`, and the two dev tests fail with `KeyError` for the same names. `test_no_container_is_given_static_aws_credentials` passes today and must stay green.
- [ ] **Step 3: Implement** the registry constants and rows and the compose edits listed under Files. Approach: the strict overlay writes `LLM_PROVIDER: bedrock` as a literal and removes the dev-only names with `!reset null` (Compose 2.24+, the same feature WS-03.3 uses for `ports: !reset []`); if the pinned Compose keeps an environment key under `!reset null`, give those rows a strict row with the empty literal instead, because the test checks the rendered result, not the mechanism. From W3, WS-03.7's `!override` of each service's `secrets` list keeps the dev-only key files out of strict. `test_registry_entries_are_well_formed` is extended to fail on any row named in `DENIED_NAMES`. The `BEDROCK_MODEL_ID` example in `.env.example` is deliberately not a model id, so a strict host that copies it fails at its first Bedrock call instead of reaching an unchosen model; an empty value keeps the LLM off until approval, when WS-09.4 makes it a startup error; no real model id appears anywhere in the tree. Finish with RENDER.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; BOOTSMOKE (strict boots with `LLM_PHI_EGRESS_APPROVED=false`, so no Bedrock call is made). Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-03.20: AWS access for the decided design: the deployer policy for MedCodex-Local, an instance role with scoped Bedrock invoke and no secret store, IMDSv2 for containers, the artifact bucket and the secrets tmpfs
**Covers:** none (host access for D-04, D-07, D-12 and D-25; supports the secrets and token-free deploy entries that WS-03.8 covers)
**Severity:** high
**Wave:** 3
**Depends on:** WS-03.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-25 (changed 2026-10-19: AWS account 253025918446, us-east-1, hosts the release gate and the pilot through the existing IAM user `MedCodex-Local` (profiles default and medcodex-local). The 2026-10-19 read-only IAM simulation allowed only `bedrock:InvokeModel` and `bedrock:ListFoundationModels` and implicitly denied `cloudformation:CreateStack`/`DescribeStacks`, `ec2:RunInstances`/`DescribeInstances`, `ssm:StartSession`/`SendCommand`/`GetParameter`, `iam:CreateRole`/`PassRole`, `s3:CreateBucket`/`PutObject`, `kms:CreateKey` and `ses:SendEmail`/`VerifyDomainIdentity`, so provisioning stays blocked until the owner attaches this policy (option 2 of `AWS-STAGING-IAM-SETUP.pdf`, now the primary path); until then all real-infra tests run on local Docker), D-04 (changed: the instance role may invoke only the chosen Bedrock model or inference-profile ARNs; no static AWS keys), D-07 (changed: no Secrets Manager or Parameter Store for secrets), D-12 (accepted: Amazon SES through SMTP with the SMTP credentials in the SOPS file; identity verification and production access are owner actions)
**Real infra:** no (static template and policy checks plus cfn-lint; nothing is deployed and no AWS API is called)
**Cross-workstream prerequisites:** WS-09's Bedrock provider takes credentials from the default boto3 chain, which in strict resolves to this instance role through IMDSv2 (requested in WS-03.19's row); WS-01 runs cfn-lint in CI (requested for WS-03.11).
**Files:**
- Modify: `infrastructure/aws-staging/solace-staging-deployer-policy.json` (a customer managed policy, not an inline user policy, because inline user policies are capped at 2,048 characters; top-level `Id` `SolaceStagingDeployerForMedCodexLocal`; self-sufficient for the W3 host, so the PDF's managed-policy list is no longer needed: CloudFormation stack and change-set actions on `solace-staging` stacks plus `ValidateTemplate` and `GetTemplateSummary`; EC2 instance, security-group, tag and describe actions conditioned on the `Project=solace-staging` request or resource tag; `ssm:StartSession`, `ssm:SendCommand` and `ssm:GetCommandInvocation` on instances tagged `Project=solace-staging` and on the `SSM-SessionManagerRunShell` and `AWS-RunShellScript` documents, `ssm:DescribeInstanceInformation` (no resource scoping; WS-05.19's preflight), `ssm:TerminateSession` and `ssm:ResumeSession` on the caller's own sessions, `ssm:GetDocument` on `SSM-SessionManagerRunShell` (to confirm session logging is off before the age key is supplied); `ssm:GetParameters` only on the public Amazon Linux AMI parameter path that the template's `LatestAmiId` resolves; the existing IAM role and instance-profile statements, `iam:PassRole` only to `ec2.amazonaws.com`; S3 bucket and object actions on `solace-staging-*` buckets for the artifact bucket; `bedrock:ListFoundationModels`, `bedrock:GetFoundationModel`, `bedrock:ListInferenceProfiles` and `bedrock:GetInferenceProfile` for the deploy-time model check; the SES v2 identity actions `ses:CreateEmailIdentity`, `ses:GetEmailIdentity`, `ses:ListEmailIdentities`, `ses:DeleteEmailIdentity` and `ses:GetAccount` (D-12); the budgets statement kept; no `secretsmanager:*`, no other `ssm:*Parameter*`, no `iam:CreateUser` or `iam:CreateAccessKey`, no `*` action)
- Modify: `infrastructure/aws-staging/solace-staging.yaml` (parameters `EnvironmentName` (`staging|production`) and `BedrockInvokeResourceArns` (`CommaDelimitedList`, default empty: the chosen foundation-model ARN, or the inference-profile ARN plus the foundation-model ARNs of every Region it routes to, read at deploy from `bedrock:ListFoundationModels` and `bedrock:ListInferenceProfiles`, never guessed); condition `HasBedrockModels`; `BedrockInvokePolicy` (`AWS::IAM::RolePolicy` on the instance role, condition `HasBedrockModels`) with exactly `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on `BedrockInvokeResourceArns`; no SES, Secrets Manager or Parameter Store permission on the role, because SES sending uses SMTP credentials from the SOPS file; `ArtifactBucket` (versioned, SSE-KMS under the AWS-managed S3 key, public access fully blocked, a TLS-only bucket policy, 14-day expiry of current and noncurrent versions) with role `s3:GetObject` on its `artifacts/` prefix only and role `s3:PutObject` only on `gate/*/results/*` and `gate/*/ssm-output/*` (the release gate's results and the SSM command output, WS-05.19), never on `artifacts/`; the AWS-managed `aws/s3` key's own key policy authorizes its use through S3 for principals of this account, so neither the role nor the deployer needs a KMS grant for this bucket; Instance `MetadataOptions` with `HttpTokens: required`, `HttpPutResponseHopLimit: 2` (containers reach the role through IMDSv2) and `HttpEndpoint: enabled`; UserData adds an fstab entry for, and mounts, a root-only tmpfs at /run/secrets (`size=16m,mode=0700,uid=0,gid=0,nosuid,nodev,noexec`), and installs the release-gate host tools next to the existing Docker and compose plugin: `python3.12` from the Amazon Linux repository, the AWS CLI that Amazon Linux ships, and uv from parameters `UvVersion` (default `0.8.2`) and `UvSha256` (default: the SHA-256 of the 0.8.2 linux x86_64 release asset copied from the release's published checksum file, never guessed; if it cannot be read, stop with a ledger Ruling as WS-03.8 does for sops/age), checked with `sha256sum -c` before install; the MOTD drops the token clone and `.env.prod` steps and points at DEPLOY.md; outputs `ArtifactBucketName` and the existing `InstanceId`, kept for WS-05.19)
- Modify: `infrastructure/aws-staging/DEPLOY.md` (section 0 rewritten. Primary path: the owner creates a customer managed policy from `solace-staging-deployer-policy.json` and attaches it to `MedCodex-Local` with `aws iam attach-user-policy` (option 2 of the PDF; the PDF's managed-policy list is superseded). It records the 2026-10-19 simulation result, the owner's check with `aws iam simulate-principal-policy` after attaching, and that until then all real-infra tests run on local Docker. The owner actions before real-user traffic: accept the AWS BAA in AWS Artifact; enable Anthropic model access in Bedrock; verify the SES domain identity (DKIM records) and request SES production access; create SES SMTP credentials for a sending-only IAM user whose secret access key is never stored, and put the derived SMTP username and password into the environment's SOPS file. The model check: `aws bedrock list-foundation-models --by-provider anthropic --region us-east-1` and `aws bedrock list-inference-profiles --region us-east-1`, whose result sets `BEDROCK_MODEL_ID` and `BedrockInvokeResourceArns`. The release-gate path (WS-05.19, D-25): the gate's preflight is read-only (`DescribeStacks`, `DescribeInstanceInformation` and the `iam:SimulatePrincipalPolicy` call that `MedCodex-Local`'s existing IAM read-only access already allows); the checkpoint tree's archive goes to `artifacts/<tree>.zip`, the gate runs through `ssm send-command` with the `AWS-RunShellScript` document, and results come back from `gate/<tree>/`; an unattached policy at W5 entry stops the gate for the owner)
- Modify: `tests/deploy/conftest.py` (add the `cfn_template` fixture: YAML `SafeLoader` subclass with a multi-constructor mapping every `!Tag` to `{"Fn::Tag": value}` / `{"Ref": value}`; moved here from WS-03.8)
- Test: `tests/deploy/test_d25_aws_access.py`
**Interfaces:**
- Consumes: the IF-02 Bedrock names from WS-03.19 (`BEDROCK_REGION`, `BEDROCK_MODEL_ID`).
- Produces: the `cfn_template` fixture (used by WS-03.11); template parameters `EnvironmentName`, `BedrockInvokeResourceArns`, `UvVersion` and `UvSha256`; outputs `ArtifactBucketName` (WS-03.8's tarball delivery) and `InstanceId`; the host's root-only tmpfs at `/run/secrets` that WS-03.8 renders into; the deployer policy the owner attaches to `MedCodex-Local`. This lands WS-09's CR-15 for WS-09.18 (the instance role's `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`, scoped to `BedrockInvokeResourceArns`, the inference-profile ARN plus the foundation-model ARNs it routes to, checked with the amazon-bedrock skill) and WS-05's request 17 for WS-05.19 (the `InstanceId` and `ArtifactBucketName` outputs, the `artifacts/<tree>.zip` and `gate/<tree>/` prefixes, the deployer's `DescribeStacks`, `DescribeInstanceInformation`, `SendCommand` on the instance and `AWS-RunShellScript`, `GetCommandInvocation` and S3 object actions, the role's read of `artifacts/*` and writes of `gate/*/results/*` and `gate/*/ssm-output/*`, and the host tools); WS-05.19 depends on this task.

- [ ] **Step 1: Write the failing test** in `tests/deploy/test_d25_aws_access.py`, and add the `cfn_template` fixture to `tests/deploy/conftest.py`
```python
"""D-25/D-04/D-07/D-12: the deployer policy the owner attaches to MedCodex-Local covers the decided design and grants no
secret store; the instance role may invoke only the chosen Bedrock models, has no secret-store or SES permission, and
containers reach it through IMDSv2; secrets land on a root-only tmpfs; the deploy doc records the owner actions."""
import json
from tests.deploy.conftest import REPO

POLICY = json.loads((REPO / "infrastructure/aws-staging/solace-staging-deployer-policy.json").read_text(encoding="utf-8"))
DEPLOY = (REPO / "infrastructure/aws-staging/DEPLOY.md").read_text(encoding="utf-8")

def _acts(stmt):
    a = stmt.get("Action", [])
    return [a] if isinstance(a, str) else list(a)

def _policy_actions():
    return {a: s for s in POLICY["Statement"] for a in _acts(s)}

def _role_statements(t):
    res = t["Resources"]
    stmts = [s for p in res["InstanceRole"]["Properties"].get("Policies", []) for s in p["PolicyDocument"]["Statement"]]
    return stmts + [s for r in res.values() if r["Type"] == "AWS::IAM::RolePolicy"
                    for s in r["Properties"]["PolicyDocument"]["Statement"]]

def test_policy_is_a_managed_policy_for_the_existing_user():
    assert POLICY["Id"] == "SolaceStagingDeployerForMedCodexLocal"
    assert len(json.dumps(POLICY, separators=(",", ":"))) <= 6144  # IAM managed-policy size limit
    assert "MedCodex-Local" in DEPLOY and "attach-user-policy" in DEPLOY
    assert "three managed policies" not in DEPLOY

def test_policy_covers_the_decided_design():
    needed = {"cloudformation:CreateStack", "cloudformation:DescribeStacks", "cloudformation:UpdateStack",
              "cloudformation:DeleteStack", "ec2:RunInstances", "ec2:DescribeInstances", "ssm:StartSession",
              "ssm:SendCommand", "ssm:GetCommandInvocation", "ssm:DescribeInstanceInformation", "ssm:GetDocument",
              "iam:CreateRole", "iam:PassRole", "s3:CreateBucket",
              "s3:PutObject", "bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles",
              "ses:CreateEmailIdentity", "ses:GetEmailIdentity"}
    assert sorted(needed - set(_policy_actions())) == []

def test_policy_grants_no_secret_store_and_no_static_keys():
    acts = _policy_actions()
    assert [a for a in acts if a == "*" or a.endswith(":*") or a.startswith("secretsmanager:")] == []
    assert not {"iam:CreateUser", "iam:CreateAccessKey", "ssm:PutParameter", "ssm:GetParametersByPath"} & set(acts)
    for a in ("ssm:GetParameter", "ssm:GetParameters"):
        if a in acts:
            res = acts[a]["Resource"]
            assert all("parameter/aws/service/ami-amazon-linux-latest/" in r for r in ([res] if isinstance(res, str) else res))
    assert acts["iam:PassRole"]["Condition"]["StringEquals"]["iam:PassedToService"] == "ec2.amazonaws.com"

def test_instance_role_invokes_only_the_chosen_bedrock_models(cfn_template):
    pol = cfn_template["Resources"]["BedrockInvokePolicy"]
    assert pol["Condition"] == "HasBedrockModels"
    (stmt,) = pol["Properties"]["PolicyDocument"]["Statement"]
    assert sorted(_acts(stmt)) == ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    assert stmt["Resource"] == {"Ref": "BedrockInvokeResourceArns"}
    param = cfn_template["Parameters"]["BedrockInvokeResourceArns"]
    assert param["Type"] == "CommaDelimitedList" and param.get("Default", "") == ""
    others = [a for s in _role_statements(cfn_template) if s is not stmt for a in _acts(s) if a.startswith("bedrock:")]
    assert others == []

def test_instance_role_has_no_secret_store_or_ses_access(cfn_template):
    acts = [a for s in _role_statements(cfn_template) for a in _acts(s)]
    assert [a for a in acts if a.split(":")[0] in ("secretsmanager", "ses") or "Parameter" in a] == []
    managed = cfn_template["Resources"]["InstanceRole"]["Properties"]["ManagedPolicyArns"]
    assert [m for m in managed if "SecretsManager" in str(m) or "SES" in str(m)] == []

def test_containers_reach_the_role_through_imdsv2_only(cfn_template):
    md = cfn_template["Resources"]["Instance"]["Properties"]["MetadataOptions"]
    assert md["HttpTokens"] == "required" and md["HttpPutResponseHopLimit"] == 2

def test_host_mounts_a_root_only_secrets_tmpfs_and_no_env_prod(cfn_template):
    user_data = json.dumps(cfn_template["Resources"]["Instance"]["Properties"]["UserData"])
    assert "tmpfs /run/secrets tmpfs" in user_data and "mode=0700" in user_data
    assert ".env.prod" not in user_data and "@github.com" not in user_data

def test_artifact_bucket_is_private_and_the_host_writes_only_gate_results(cfn_template):
    b = cfn_template["Resources"]["ArtifactBucket"]["Properties"]
    assert all(b["PublicAccessBlockConfiguration"].values()) and b["VersioningConfiguration"]["Status"] == "Enabled"
    stmts = _role_statements(cfn_template)
    assert [s for s in stmts if "s3:GetObject" in _acts(s) and "artifacts/*" in json.dumps(s["Resource"])]
    puts = [json.dumps(s["Resource"]) for s in stmts if "s3:PutObject" in _acts(s) and "Artifact" in json.dumps(s["Resource"])]
    assert puts and all("artifacts/" not in r and "gate/*/" in r for r in puts)
    assert any("gate/*/results/*" in r for r in puts) and any("gate/*/ssm-output/*" in r for r in puts)

def test_host_provides_the_release_gate_tools_and_outputs(cfn_template):
    user_data = json.dumps(cfn_template["Resources"]["Instance"]["Properties"]["UserData"])
    assert "python3.12" in user_data and "docker compose version" in user_data
    assert "UvSha256" in user_data and "sha256sum -c" in user_data
    assert cfn_template["Parameters"]["UvVersion"]["Default"] == "0.8.2"
    assert {"InstanceId", "ArtifactBucketName"} <= set(cfn_template["Outputs"])

def test_deploy_doc_records_the_simulation_and_the_owner_actions():
    for text in ("2026-10-19", "simulate-principal-policy", "local Docker", "list-foundation-models",
                 "list-inference-profiles", "AWS Artifact", "SES", "DKIM", "AWS-RunShellScript", "gate/<tree>/"):
        assert text in DEPLOY, text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `UNIT D:\Repo\Contextual-Chatbot\tests\deploy\test_d25_aws_access.py`
Expected: FAIL. `test_policy_is_a_managed_policy_for_the_existing_user` fails with `KeyError: 'Id'`; `test_policy_covers_the_decided_design` lists every CloudFormation, EC2, SSM, S3, Bedrock and SES action, because today's JSON grants only the IAM role and instance-profile actions, `iam:PassRole` and budgets, and the rest came from the PDF's managed policies, which the 2026-10-19 simulation shows are not attached to `MedCodex-Local`; the template tests fail with `KeyError: 'BedrockInvokePolicy'`, `KeyError: 'MetadataOptions'` and `KeyError: 'ArtifactBucket'`, the UserData test finds `.env.prod` and no tmpfs (the MOTD's step 2), and the host-tools test finds no `python3.12` (UserData installs only `python3`) and fails with `KeyError: 'UvVersion'`; the DEPLOY.md test finds no simulation record and no owner actions. `test_policy_grants_no_secret_store_and_no_static_keys` and `test_instance_role_has_no_secret_store_or_ses_access` pass today and must stay green.
- [ ] **Step 3: Implement** the policy, template and runbook edits listed under Files. Approach: least privilege, with resource scoping wherever an action supports it (stack ARNs of `solace-staging` stacks; EC2 through the `aws:RequestTag/Project` and `aws:ResourceTag/Project` conditions, with describes on `*` as EC2 requires; SSM documents by ARN; the existing `solace-staging-*` IAM role and profile ARNs; `solace-staging-*` buckets; SES identities of this account and Region); actions that support no resource scoping (`cloudformation:ValidateTemplate`, `cloudformation:GetTemplateSummary`, the Bedrock list and get calls, `ses:GetAccount`, EC2 describes) take `*` resources, never `*` actions. Keep the SSM-only management path (no SSH, no inbound port in W3). `BedrockInvokeResourceArns` defaults to empty so the stack can exist before the model choice, and `HasBedrockModels` adds the invoke policy only once the operator passes the ARNs checked at deploy (LLM features stay off for real users until the AWS BAA is accepted and recorded). The artifact bucket's SSE-KMS under the AWS-managed S3 key keeps WS-03.11's cfn-guard bucket rules true for it. SES sending uses the SMTP interface with credentials from the SOPS file, so the role gets no SES permission, and creating the SMTP credentials is an owner action outside this policy. Record the simulation in DEPLOY.md section 0 exactly as D-25 states it. Then run `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m cfnlint D:\Repo\Contextual-Chatbot\infrastructure\aws-staging\solace-staging.yaml` until it reports no errors. No AWS API is called: the owner attaches the policy, and nothing is deployed.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** DEPLOYREG; the cfn-lint command above. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-03.20:done"` and record the printed tree id in the ledger. No git commit.

---

## Cross-workstream requests

| Owner | Exact change | Files | Needed by |
|---|---|---|---|
| WS-01 | W0 phase A2 (WS-01.9 for the migrate image, after WS-02.4; WS-01.6 for the entrypoint, phase A1): `docker/migrate.Dockerfile` builds the `migrate` image (compose tags it `solace/migrate:${SOLACE_IMAGE_TAG:-dev}`, the reference WS-01's builder uses) whose working directory lets `python -m scripts.db.migrate upgrade head` run; `docker/entrypoint.sh` exports `NAME` from every `NAME_FILE` (one trailing newline stripped, missing file fatal) and leaves `NAME_FILE` set (WS-04.11 checks provenance) | `docker/migrate.Dockerfile`, `docker/entrypoint.sh` | WS-03.2, WS-03.7 |
| WS-01 | W0 phase A1 (planned in WS-01.5): the image builder owns the reference format and exports it, `image_ref(name, tag)` returning `solace/<name>:<tag>` (used by `build_command`) and `COMPOSE_TAG = "${SOLACE_IMAGE_TAG:-dev}"`, so WS-03.2 writes `image_ref(<service>, COMPOSE_TAG)` as each built compose service's `image:` value and its test imports both names instead of describing the builder's internals. This change used to sit in WS-03.2's Files line. | `scripts/ci/build_images.py` | WS-03.2 |
| WS-01 | W0: `.dockerignore` excludes `.secrets` and `secrets` so generated secret files never enter a build context (WS-01.5); it also excludes `deploy/secrets/` (D-07: the encrypted per-environment SOPS files are mounted at render time and never baked into an image), planned in WS-01.14 (W0, earlier than asked), on which WS-03.8 depends | `.dockerignore` | WS-03.7, WS-03.8 |
| WS-01 | W0 (WS-01.2, because WS-01 has no W3 wave; WS-03.8 depends on WS-01.2): `requirements.lock` carries hashed pins for PyYAML, cryptography, cffi and pycparser, which `deploy/sops/Dockerfile` installs with `--require-hashes`. The earlier request for `boto3` in the dev lock is withdrawn: D-07 replaced AWS Secrets Manager with SOPS + age, so `deploy/render_secrets.py` has no AWS client and no Stubber tests (D-04's `boto3`/`botocore` runtime pins are WS-01's own task) | `requirements.lock` | WS-03.8 |
| WS-01 | W4: `make ci-local` and CI run `cfn-lint` and `cfn-guard validate -r infrastructure/aws-staging/guard/solace-staging.guard` on `infrastructure/aws-staging/solace-staging.yaml`; the CI image matrix excludes config-service (D-19) | `Makefile`, `.github/workflows/ci.yml` | WS-03.11 |
| WS-02 | W0 phase A2: the init-db role script creates `solace_app` and `solace_migrator` with passwords from the postgres-container variables `SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD` (exported by the container's entrypoint wrapper, so no `_FILE` handling is needed in the script); `scripts/db/gen_certs.py --out DIR` writes `postgres-ca.crt`, `server.crt` (SAN `DNS:postgres`) and `server.key`; `scripts.db.migrate` reads `DATABASE_URL`, `POSTGRES_SSL_MODE` and `POSTGRES_SSL_ROOT_CERT` | `infrastructure/init-db/01-create-databases.sql`, `scripts/db/gen_certs.py`, `scripts/db/migrate.py` | WS-03.2 |
| WS-02 | W0 or W2: the pg_hba policy delivered with the TLS work allows `replication` for the image superuser `postgres` from the compose `data` network (`hostssl` in strict, `host` in dev), used only by the backup sidecar | `infrastructure/init-db/01-create-databases.sql` | WS-03.9 |
| WS-04 | W1 (WS-04.7): the observability fragment reads the six `ALERTMANAGER_*` registry variables and relies on the `./infrastructure/prometheus` directory mount; W3 (WS-04.11): its SMTP password moves to `ALERTMANAGER_SMTP_PASSWORD_FILE`; W4 (WS-04.12): node-exporter reads the `solace_textfile_metrics` volume and `BackupStale`/`WalArchiveStale` alert on `store="postgres_base"`/`"postgres_wal"` | `docker-compose.observability.yml`, `infrastructure/prometheus/rules/solace.yml` | WS-03.6, WS-03.9 |
| WS-04 | W3 (WS-04.11): strict guards reject the formerly committed dev values listed in WS-03.7's test, low-entropy secrets and secrets not delivered through `NAME_FILE` | `src/solace_security/production_guards.py` | WS-03.7 |
| WS-05 | W0: `docker-compose.test.yml` overrides only `CADDY_API_SITE`, `CADDY_AUTH_SITE` and ports on the base-defined `caddy`, and builds its per-run env file with `deploy.env.registry.write_synthetic_env(path, profile=..., environment=...)`; `test_env_contract.py` renders strict plus the test overlay and must stay green | `docker-compose.test.yml`, `tests/e2e/conftest.py` | WS-03.5 |
| WS-05 | W3: the e2e stack creates its `SOLACE_SECRETS_DIR` with `python -m deploy.env.generate_dev_env --out <run dir> --mode 0644` (throwaway per-run values) and passes `SOLACE_SECRETS_GID` | `tests/e2e/conftest.py`, `docker-compose.test.yml` | WS-03.7 |
| WS-05 | W4 (WS-05.17, WS-05.18): the restore-drill driver adds `-f docker-compose.backup.yml` and drives `docker compose ... run --rm backup backup-now`, then `restore-latest` on the destroyed volume, then `status` (WS-03.16's exit codes and ISO timestamps), and asserts row checksums, PHI decryption with the original keys and `verify_chain`; the `BackupStale` fault test stops `pg-backup` and runs `backup age-metric <seconds>`; the drill report copies `status`'s `backup_target` line and, when it is `local`, states that the drill used the local encrypted-volume target because the deployer policy was not yet attached (D-25) | `scripts/test/restore_drill.py`, `tests/e2e/test_w4_enterprise.py`, `src/solace_testing/e2e.py` | WS-03.9, WS-03.16 |
| WS-05 | Declined by owner (WS-05, the strict half conflicts with WS-09.18, which refuses a non-empty `BEDROCK_ENDPOINT_URL` in strict, and with the `docker-compose.test.yml` protocol that the overlay never weakens strict; WS-05.21 instead puts the endpoint, the synthetic model id and approval in the dev-only fragment `infrastructure/test/compose/llm-dev.yml` and runs strict stacks with `LLM_PROVIDER=bedrock`, `LLM_PHI_EGRESS_APPROVED=false` and no override, so `test_env_contract.py` needs no allow-listed strict difference; integrator ruling 2026-10-19, WS-05 request 19): W1 (with WS-05.1's Bedrock Converse route, D-04): `docker-compose.test.yml` sets `LLM_PROVIDER=bedrock`, `BEDROCK_ENDPOINT_URL=http://fake-llm:8080` and a fixed fake `BEDROCK_MODEL_ID` that only the fake LLM serves (never a real AWS model id) on the five LLM services, and no `AWS_*` credential variable; because D-04 refuses `BEDROCK_ENDPOINT_URL` in strict, the strict + test render keeps that endpoint as an explicitly allow-listed test difference in `test_env_contract.py` (like IF-19's `EMAIL_SMTP_TLS_MODE=none`), pending the ledger Ruling named under Open risks | `docker-compose.test.yml`, `tests/e2e/conftest.py` | WS-03.19 |
| WS-09 | W1 (the new D-04 Bedrock provider task in `UnifiedLLMClient`): read exactly `LLM_PROVIDER`, `BEDROCK_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_ENDPOINT_URL` and `BEDROCK_TIMEOUT_SECONDS` as WS-03.19 registers them; take credentials only from the default boto3 chain (the instance role through IMDSv2 in strict, `AWS_PROFILE` only for local development outside compose), never from `AWS_*` key variables; refuse `BEDROCK_ENDPOINT_URL` and `LLM_PROVIDER=portkey` in strict | `services/shared/infrastructure/llm_client.py` | WS-03.19, WS-03.20 |
| WS-06 | W1 or W3 phase A (request R-01.4(b) of WS-01, repeated): apply `solace_events.config.TOPIC_CONFIGS` create-if-missing at event-plane start, so strict brokers can run with auto-create off | `src/solace_events/service_plane.py`, `src/solace_events/config.py` | WS-03.18 |
| WS-12 | W5: `services/safety_service/README.md` describes `/metrics` as served by the runtime bootstrap (true from W0) | `services/safety_service/README.md` | WS-03.15 |

## New interfaces

- **Registry schema (extends IF-02).** Each row in `deploy/env/env-contract.yaml` has the IF-02 fields (`name`, `services`, `secret`, `required_in_strict`, `dev_default`, `notes`, `retired`) plus: `source` (env-file or secrets-store key; null means a compose literal), `profiles` (default `[dev, strict]`), `format` (one of `text`, `int`, `bool`, `url`, `email`, `hostport`, `csv`, `json`, `urlsafe24`, `urlsafe48`, `hex64`, `fernet`, `bcrypt`, `pem`), `strict_default` (value when a strict source is unset; null with `required_in_strict` means `:?`), `example` (placeholder for operator-supplied non-secrets), `group` (heading in `.env.example`; `legacy` for aliases awaiting retirement), and from W3 `derive` (a template such as `redis://:{REDIS_PASSWORD}@redis:6379/0`). The same `name` may appear in several rows with different services and sources (for example `POSTGRES_PASSWORD` on apps from source `POSTGRES_PASSWORD`, and on `postgres` from source `POSTGRES_SUPERUSER_PASSWORD`).
- **Deploy tooling API.** `deploy.env.registry` (`APP_SERVICES`, `FORMATS`, `EnvVar`, `load_registry`, `generate_value`, `synthetic_values`, `write_synthetic_env`, `file_secret_sources`, `LLM_SERVICES`, `DENIED_NAMES`, `SECRET_READER_UIDS`, `ROOT_READERS_WITH_DAC_OVERRIDE`, `secret_file_owner`, `SecretOwnershipError`), `deploy.env.compose_render` (`render_compose_config`, `ComposeRenderError`), `deploy.env.env_readers.collect_env_readers`, `deploy.env.render_env_example`, `deploy.env.generate_dev_env`, `deploy.env.scan_env_file`, `deploy.render_secrets` (SOPS only: `SopsFileSource`, `DictSource`, `render`, `check_encrypted`), `deploy.key_ids`, `deploy.key_escrow`, with the signatures given in each task.
- **SOPS layout (D-07, WS-03.8, WS-03.10).** `deploy/secrets/<env>.sops.yaml` per environment (keys in plaintext, every value `ENC[...]`, encrypted to the environment's age recipient(s) and to every escrow recipient); `deploy/secrets/.sops.yaml` creation rules and `deploy/secrets/escrow-recipients.txt`, both public recipients only, written by the operator or owner; previous encrypted versions under `deploy/secrets/history/<env>/<UTC timestamp>.sops.yaml`; age private keys never in the tree, an image or a plan, supplied at deploy through `SOPS_AGE_KEY_FILE`; pinned tools in `deploy/sops/tools.json`, image `solace/sops-tools`. No AWS Secrets Manager or SSM Parameter Store use anywhere.
- **LLM provider names (D-04, WS-03.19).** `LLM_PROVIDER` (`bedrock` | `portkey`; strict literal `bedrock`), `BEDROCK_REGION` (`us-east-1`), `BEDROCK_MODEL_ID` (set at deploy, no default; required in strict once `LLM_PHI_EGRESS_APPROVED=true`, enforced at startup by WS-09.4), `BEDROCK_ENDPOINT_URL` (dev and test only), `BEDROCK_TIMEOUT_SECONDS` (`20`) on the five LLM services; `LLM_PORTKEY_API_KEY`, `LLM_PORTKEY_GATEWAY_URL` and `LLM_ANTHROPIC_API_KEY` are dev and test only, so IF-22's strict secret set no longer includes the two keys; no container ever receives `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` or `AWS_PROFILE`. `EMAIL_SMTP_USERNAME` joins `EMAIL_SMTP_PASSWORD` as a file secret (the SES SMTP credentials, D-12).
- **AWS host (D-25, WS-03.20, WS-03.11).** Parameters `EnvironmentName`, `BedrockInvokeResourceArns`; outputs `ArtifactBucketName`, `BackupBucketName`, `BackupKmsKeyArn`; the two customer managed deployer policies (`solace-staging-deployer-policy.json` for the W3 host, `solace-staging-deployer-host-policy.json` for the W4 resources) that the owner attaches to `MedCodex-Local`.
- **Compose layout.** `caddy` is defined in `docker-compose.yml` for both profiles (dev edge `http://api.localhost:8088` / `http://auth.localhost:8088` on `127.0.0.1:8088`); no service sets `container_name`; profile `vector` holds Weaviate; profile `restore` holds `pg-restore`; service `migrate` gates every app service.
- **Postgres container init variables** `SOLACE_APP_PASSWORD` and `SOLACE_MIGRATOR_PASSWORD` (from sources `POSTGRES_PASSWORD` and `POSTGRES_MIGRATOR_PASSWORD`); the image superuser password comes from source `POSTGRES_SUPERUSER_PASSWORD`.
- **Secret files.** `SOLACE_SECRETS_DIR` (host directory, dev default `./.secrets/dev`, strict `/run/secrets/<env>` on the host's root-only tmpfs at `/run/secrets`) holds one file per secret source plus `key-ids.json`; containers see `/run/secrets/<source>` and `<NAME>_FILE`; in strict every file is mode 0400 and owned by `secret_file_owner` (the uid that reads it, since Compose bind-mounts file secrets and ignores `uid`/`gid`/`mode` for them); in dev `SOLACE_SECRETS_GID` is the group that may read them; per-service service-token keys use source names `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service-name>`.
- **Image names.** Every built service is tagged `solace/<service>:${SOLACE_IMAGE_TAG:-dev}` (`SOLACE_IMAGE_TAG` is compose interpolation only, default `dev`); third-party images are `<name>:<tag>@sha256:<digest>` from W4, postgres is `scripts.db.images.POSTGRES_IMAGE`.
- **Backup CLI (WS-03.16).** Service `backup` (profile `ops`, no `depends_on`) run as `docker compose ... run --rm backup <cmd>`: `backup-now` (exit 0/1, `backup_id=<id> completed_at=<ts>`), `restore-latest` (exit 0/1/2/3, `restore_seconds=<n> backup_id=<id> recovered_to=<ts>`), `status` (exit 0 fresh, 1 stale or missing; `postgres_base_last_success`, `postgres_wal_last_success`, `latest_backup_id` and `backup_target` (`local` | `s3`, D-25) lines), `age-metric <seconds>` (drill hook that only ages the metric); usage errors exit 64; timestamps ISO 8601 UTC.
- **Backups.** Services `pg-backup`/`pg-restore`, commands `solace-backup base-now` and `solace-restore`, volumes `pg_wal_spool`, `pg_backup_work`, `pg_wal_restore`, `solace_textfile_metrics`, variables `BACKUP_TARGET_DIR`, `BACKUP_S3_URI`, `BACKUP_KMS_KEY_ID`, `BACKUP_RETENTION_DAYS`, `BACKUP_BASE_INTERVAL_SECONDS`, `BACKUP_WAL_PUSH_SECONDS`, metric stores `postgres_base` and `postgres_wal`, restore exit codes 2 (postgres running) and 3 (key mismatch).
- **Alertmanager variables** registered for WS-04 (same names as WS-04's New interfaces).
- **Test-only switch** `SOLACE_RUN_DOCKER_TESTS=1` for container-starting tests in `tests/deploy` (shared with WS-04's `tests/bootstrap`).

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-002 | blocker | WS-03.4 |
| MVP-003 | blocker | WS-03.2 |
| MVP-004 | blocker | WS-03.1 |
| MVP-037 | blocker | WS-03.9 (drill also proved end to end by WS-05's W4 suite) |
| MVP-348 | blocker | WS-03.7 (per-service file delivery, no secret in any env), WS-03.8 (SOPS + age render into root-only 0400 tmpfs files, rotation, docs) |
| MVP-040 | high | WS-03.3 |
| MVP-042 | high | WS-03.6 |
| MVP-043 | high | WS-03.11 |
| MVP-044 | high | WS-03.11 (CloudWatch side; Prometheus side in WS-04.12) |
| MVP-045 | high | WS-03.4 |
| MVP-349 | high | WS-03.7 (guard half in WS-04.11) |
| MVP-350 | high | WS-03.10 |
| MVP-115 | medium | WS-03.12 (TRUSTED_PROXY_HOPS in WS-03.5) |
| MVP-116 | medium | WS-03.2 |
| MVP-117 | medium | WS-03.5 |
| MVP-118 | medium | WS-03.3 |
| MVP-225 | medium | WS-03.5 |
| MVP-231 | low | WS-03.1 (Prometheus job removed in WS-04.5; CI matrix by WS-01) |
| MVP-242 | low | WS-03.15 |
| MVP-247 | low | WS-03.1 |
| MVP-250 | low | WS-03.5 |
| MVP-308 | low | WS-03.15 (safety README line by WS-12) |
| MVP-352 | low | WS-03.8 |
| MVP-354 | low | WS-03.13 |

**Interface consistency**
- IF-02 names, values and the "always `${ENVIRONMENT}`" rule match the skeleton; additions are only schema fields and deploy-tool APIs listed above.
- IF-22 secret names are the registry sources; Caddy receives only `DOMAIN`, `CADDY_ADMIN_EMAIL` and `BASIC_AUTH_HASH` (plus its two site addresses, which are not secrets); backups record key ids, never keys. Under D-07 the strict store is the per-environment SOPS file instead of AWS Secrets Manager names `solace/<env>/<NAME>`; the consumer side of IF-22 (`NAME_FILE` under `/run/secrets`) is unchanged, so no consumer plan changes. Under D-04 the Portkey and direct Anthropic keys leave the strict secret set, and `EMAIL_SMTP_USERNAME` joins it (D-12).
- IF-02 gains the five D-04 LLM names (WS-03.19) exactly as the decision record names them.
- IF-18: every app healthcheck, Caddy active check and autoheal label targets `/live`; `/metrics` and the other internal paths are never routed by Caddy.
- The paired WS-04 tasks referenced here (WS-04.7, WS-04.11, WS-04.12, WS-04.13, WS-04.14) exist with matching scope, and WS-04.7/WS-04.11/WS-04.12/WS-04.13 depend on WS-03.6/WS-03.8/WS-03.9/WS-03.13 respectively.
- Requests addressed to WS-03 by other plans are landed as tasks whose Produces line names the requester: R-01.1 of WS-01 in WS-03.2 (W0 compose sync for WS-01.9), WS-03.17 (W4 digest pins for WS-01.10) and WS-03.18 (strict auto-create off for WS-01.12); WS-04's W4 bundle in WS-03.9 (textfile metric for WS-04.12) and WS-03.13 (autoheal labels for WS-04.13); WS-05's request 6 in WS-03.16 (backup CLI for WS-05.17 and WS-05.18).

REVIEW_FOCUS: 4=WS-03.5,WS-03.9; 5=WS-03.1,WS-03.3,WS-03.7,WS-03.10

Review focus mapping (skeleton order): 1 (ingress false negatives), 2 (alert-path concurrency) and 3 (foreign identifiers under RLS) have no code in WS-03. 4 (PHI sinks): WS-03.5 `test_internal_paths_are_blocked_and_no_access_log` pins that Caddy writes no access log (so no WS query token or path PHI lands there) and blocks `/metrics`; WS-03.9 `test_restore_drill_recovers_every_archived_row` pins that no backup artefact holds a plaintext row. 5 (strict boots with partial config and drift): WS-03.1 `test_required_strict_sources_fail_fast` and `test_no_dev_default_survives_in_strict` (a missing secret or renamed variable cannot fall back to a dev default), WS-03.3 `test_every_redis_client_gets_the_password` and `test_strict_render_requires_REDIS_PASSWORD` (the Redis password reaches every client or the render fails), WS-03.7 `test_no_secret_has_a_committed_default` and `test_each_container_mounts_exactly_its_secrets` (secrets only through files, no committed dev value) and WS-03.10 `test_restore_refuses_mismatched_key_material` (a restore onto rotated or fresh keys is refused, not silently unreadable).

**Open risks**
- IF-19 lets the strict test overlay set `EMAIL_SMTP_TLS_MODE=none` for mailpit while IF-02 allows `none` only in development/test; `test_env_contract.py` must allow-list exactly the IF-19 test differences when it renders strict plus the test overlay, or WS-13's strict guard will reject the e2e stack. Raise as a ledger Ruling at W1 entry if WS-05 and WS-13 disagree.
- File permissions for bind-mounted secrets differ between Linux hosts and Docker Desktop; the design (group-readable files plus `group_add`, root-started entrypoints that read before dropping privileges) is proven on this Windows host only through Docker Desktop's permissive mounts, so the Linux strict host must run the WS-05 strict suite once before go-live.
- The backup design depends on WS-02 allowing replication for the image superuser; without it `pg_basebackup` fails and the drill stays red.
- WS-03.17 pins third-party images by digest; images not present locally need a read-only registry query (`docker buildx imagetools inspect`) and a pull, which WS-01.10 already treats as allowed for digest-pinned references. If the controller rules otherwise, WS-03.17 stops with a Ruling instead of guessing a digest.
- WS-03.18 waits on WS-06's create-if-missing of the registry topics (R-01.4(b)), which no WS-06 task plans yet; until it lands, strict brokers keep auto-create on and MVP-269's Kafka part stays open (WS-01.12 records the Ruling at W5 entry).
- User-service's `ServiceConfig.env` accepts only development/staging/production, so an operator who sets `ENVIRONMENT=test` breaks it; the profiles here never do, but WS-14 should widen the literal or reject `test` clearly.
- AWS hosts the release gate and the pilot (D-25), but the 2026-10-19 simulation shows `MedCodex-Local` cannot provision anything yet. Until the owner attaches the two deployer policies, the template is validated only, ACME, DNS, alarms and S3 backups stay unproven on the real host, all real-infra tests run on local Docker and the W4 drill uses the local encrypted-volume target (and says so). If the policy is still unattached at W5 entry, the release gate stops for the owner; there is no silent fallback to a local host.
- D-07's off-host age keys make every host reboot or `ec2:recover` (WS-03.11's alarm action) a fail-closed outage: the tmpfs is empty, no container can start, and the crisis path is down until an operator re-supplies the key and re-renders. This needs the owner's explicit acceptance (or an on-call operator procedure within the D-09 RTO of 4 h); `ContainerUnhealthy` and the host alarms page, but nothing restarts the stack on its own.
- IMDSv2 with hop limit 2 (WS-03.20) lets every container, not only the five LLM services, obtain the instance role's credentials: today that is Bedrock invoke on the chosen models, read of the artifact prefix and, from W4, write of encrypted backups. Restricting the metadata endpoint to the LLM services (for example a host firewall rule in the DOCKER-USER chain for a dedicated network) is not planned; raise it as a ledger Ruling at W3 entry if the owner wants it.
- Resolved (integration 2026-10-19): D-04 refuses `BEDROCK_ENDPOINT_URL` in strict, so strict e2e stacks run with `LLM_PROVIDER=bedrock`, `LLM_PHI_EGRESS_APPROVED=false` and no override (template replies), and only the dev profile reaches the fake LLM through WS-05.21's dev-only fragment. There is no test-only strict allow-list for the endpoint; the WS-05 row that asked for one is declined. Verifying the Bedrock path on the staging host once the LLM is enabled after the AWS BAA is an owner decision.
- The sops and age checksums in `deploy/sops/tools.json` must come from the projects' published checksum files; if the controller rules network reads out, WS-03.8 stops with a Ruling instead of guessing (as WS-03.17 does for digests).

## Review responses

- Checker error: WS-03.6 modifies `tests/deploy/test_env_contract.py`, which no earlier task creates: fixed. WS-03.1 (wave 0, an ancestor of WS-03.6 through WS-03.2 to WS-03.5) always wrote the file, but listed it only on its `Test:` line; it is now on WS-03.1's `Create:` line.
- Checker error: plan a task for WS-01's request R-01.1 (needed by WS-01.9): fixed. W0 part: WS-03.2 now tags every built service `image: solace/<service>:${SOLACE_IMAGE_TAG:-dev}` (the `migrate` one-shot becomes `solace/migrate`, built from `docker/migrate.Dockerfile`), keeps `build: {context: ., ...}`, and pins this with `test_built_services_are_tagged_for_ci`. Its Produces line names WS-01.9; config-service removal (WS-03.1) and `/live` healthchecks (WS-03.2) were already planned. W4 part: new task WS-03.17 pins every third-party image by digest and sets postgres to `scripts.db.images.POSTGRES_IMAGE`; Produces names WS-01.10. Strict auto-create part: new task WS-03.18; Produces names WS-01.12. WS-03.18 sits in W5, not W1, because no WS-06 task creates topics at plane start yet (WS-06.1 to WS-06.13 contain no topic creation; R-01.4(b) is still an open request). Turning auto-create off before that would break strict event delivery, and its only consumer, WS-01.12, runs in W5. WS-03.18 also has an entry check, and the request is repeated to WS-06 under Cross-workstream requests.
- Checker error: plan a task for WS-04's request (needed by WS-04.12): fixed. WS-03.9 already wrote `solace_backup_last_success_timestamp_seconds{store="postgres_base"|"postgres_wal"}` to `/textfile/solace_backup.prom` on the `solace_textfile_metrics` volume, and its Produces line now names WS-04.12. WS-03.13's Produces line now names WS-04.13 for the `solace.autoheal` labels on the `/live`-checked app services.
- Checker error: plan a task for WS-05's request 6 (needed by WS-05.17): fixed. New task WS-03.16 adds the `backup` CLI service (profile `ops`, no `depends_on`), run as `docker compose ... run --rm backup <cmd>`. It provides `backup-now`, `restore-latest` and `status` with fixed exit codes and ISO-8601 UTC timestamps, plus the `age-metric <seconds>` drill hook for WS-05.18's `BackupStale` test. The hook can only make the metric older, so it can fire the alert but never hide it. Produces names WS-05.17 and WS-05.18, and WS-03's own W4 request row to WS-05 now uses these commands.
- Checker warnings: none were reported.
- Also fixed while landing the requests: the backup image is tagged `solace/pg-backup:${SOLACE_IMAGE_TAG:-dev}` (WS-03.9) so WS-03.17's pin test can tell built images from third-party ones. `destroy_volume` now removes every container that mounts the volume, because `pg-backup`'s read-only mount would otherwise make `docker volume rm` fail with "volume is in use". WS-03.9 and WS-03.10 now list the `tests/deploy/conftest.py` helpers they add. WS-03.15 now depends on WS-03.16, WS-03.17 and WS-03.18, so the docs describe the final state.
- Reconciliation 1 (`scripts/ci/build_images.py` is WS-01's, and WS-03.2 named it in W0 next to WS-01.5): fixed. The path is gone from WS-03.2's Files line; the image-reference agreement moved to WS-01.5 (`image_ref`, `COMPOSE_TAG`), requested in the new WS-01 row above. WS-03.2 now depends on WS-01.5, and on WS-01.6 and WS-01.9 as WS-01's R-01.16 asked (the migrate image exists only after WS-02.4, in W0 phase A2, so the old "phase A1" wording in the migrate row is corrected). `test_built_services_are_tagged_for_ci` imports `image_ref`/`COMPOSE_TAG` and checks the built set against `IMAGES`.
- decision 2026-10-19 D-07: AWS Secrets Manager replaced by SOPS + age everywhere in this plan. WS-03.8 is reworked (same id): `deploy/render_secrets.py` decrypts `deploy/secrets/<env>.sops.yaml` with pinned sops/age (`deploy/sops/tools.json`, `install_tools.sh`, the `solace/sops-tools` image) into root-only 0400 files on the tmpfs `/run/secrets`, owned by the reading uid (`secret_file_owner`), with `SOPS_AGE_KEY_FILE` from a root-only path and fail-closed, all-or-nothing rendering; `AwsSecretsManagerSource`, the boto3 Stubber test, the secrets KMS key and the `secretsmanager:*` role grant are gone; tests use a fake sops and synthetic values, and the real sops/age round trip with a throwaway age key is a realinfra test in the pinned container; the runbook covers rotation (encrypted history, since nothing is committed) and re-key. WS-03.7: gate reworded, the strict overlay `!override`s secrets lists, and new `deploy/env/scan_env_file.py` plus an owner action rotate the working-tree `.env` keys without printing a value. WS-03.10: escrow now means an extra offline age escrow recipient on every SOPS file (checked by `check_encrypted`) plus the bundle; the KMS-retention and `DeleteSecret`-deny CFN items and their test are replaced. WS-03.13: weaviate keeps `DAC_OVERRIDE` and a new test pins it for root secret readers. WS-03.15: docs may not name a secrets store we do not use. Architecture, File map, New interfaces, Open risks and the WS-01 rows (`.dockerignore`, hashed lock pins; the `boto3` dev-lock request withdrawn) follow. The NAME_FILE consumer contract (IF-02/IF-22) is unchanged.
- decision 2026-10-19 D-25: the primary path is now attaching the deployer policy to the existing user `MedCodex-Local`. New task WS-03.20 (W3, before WS-03.8) makes `solace-staging-deployer-policy.json` a self-sufficient customer managed policy (managed rather than inline, because inline user policies are capped at 2,048 characters) covering CloudFormation, EC2, SSM sessions, `iam:PassRole` to EC2, the artifact bucket, the Bedrock model check and SES identity setup, with no Secrets Manager, no secret Parameter Store access and no access-key creation. DEPLOY.md records the 2026-10-19 simulation and the owner actions. WS-03.20 also adds the instance-role, IMDSv2, artifact-bucket and tmpfs template changes and takes over the `cfn_template` fixture from WS-03.8. WS-03.11's gate is reworded, and its W4 grants move to a second managed policy, `solace-staging-deployer-host-policy.json`, with S3 + KMS for D-09. WS-03.15's gate and `docs/runbooks/deploy.md` now put the release gate on the AWS staging host through SSM, with the stop-for-owner rule at W5. WS-03.16's `status` reports `backup_target`, so a W4 drill on the local encrypted-volume target says so (WS-05 row added). The Open risks line is rewritten.
- decision 2026-10-19 D-04: new task WS-03.19 (W1) adds `LLM_PROVIDER`, `BEDROCK_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_ENDPOINT_URL` and `BEDROCK_TIMEOUT_SECONDS` to the registry and compose. Strict pins `bedrock` as a literal and requires a deploy-time model id; `BEDROCK_ENDPOINT_URL`, the Portkey key and the direct Anthropic key are dev and test only; `DENIED_NAMES` keeps static AWS credentials out of every container. WS-03.20 scopes the instance role to `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the parameterized `BedrockInvokeResourceArns`, and adds IMDSv2 hop limit 2 so containers use the role. Rows were added for WS-05 (the e2e overlay values and the fake model id) and WS-09 (the exact names and the default credential chain). The strict-e2e endpoint conflict is listed under Open risks. No model id is guessed anywhere. WS-03.7 now depends on WS-03.19, because its strict `!override` keeps WS-03.19's dev-only key files unmounted.
- decision 2026-10-19 D-12: SES stays the accepted default, aligned with D-07 and D-25. WS-03.6's gate names SES SMTP with STARTTLS, and its registry example is the us-east-1 SES SMTP endpoint. WS-03.8 makes `EMAIL_SMTP_USERNAME` a file secret beside `EMAIL_SMTP_PASSWORD`, both living in the SOPS file. WS-03.20 keeps the instance role free of SES permissions, because sending uses SMTP, gives the deployer only the SES identity actions, and records domain-identity verification, DKIM, production access and SMTP-credential creation as owner actions.
- decision 2026-10-19 D-09: the defaults are unchanged (RPO 15 min, RTO 4 h). Only the key that protects backups changed. WS-03.11's `BackupKmsKey` is retained and rotated, and the role uses it only through S3, which replaces the retained secrets key that D-07 removed. The deployer's S3 + KMS grants are in the W4 host policy. WS-03.9 needed no change.
