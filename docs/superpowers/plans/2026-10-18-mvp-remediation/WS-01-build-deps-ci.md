# WS-01: Build, dependency lock, images, import roots and CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every MVP image builds reproducibly from one hash-pinned lock that matches the versions the suites pass on, carries the code layout its CMD imports, and starts; shared packages have one import root; and `make ci-local`, mirrored by `.github/workflows/ci.yml`, runs lint, every suite, the image builds with import probes and the real-infra targets.

**Architecture:** One universal `uv` lock (`requirements.lock`, runtime only) and `requirements-dev.lock` (runtime plus test and lint tools) are compiled against a freeze of the interpreter the 5,377 tests pass on (D-16), so images and tests run the same versions. Nine app Dockerfiles and `docker/migrate.Dockerfile` share one digest-pinned template that syncs the lock with `--require-hashes`, copies exactly the modules their CMD imports, and starts through `docker/entrypoint.sh` (the IF-22 NAME_FILE loader). Stdlib-only scripts under `scripts/ci/` (image registry, suite registry, suite runner, image builder, clean-checkout and closure checks) are the single source for the Makefile and CI, and static guards in `tests/build/` enforce git visibility, the import root and the lint gate, with one test case per owning workstream so each owner turns its own case green.

**Covers:** 22 register entries (blocker 6, high 2, medium 3, low 11).

**Waves:** 0 (phase A1: WS-01.1 to WS-01.8 and WS-01.14; phase A2, after WS-02.4: WS-01.9), 4 (phase A: WS-01.10, then WS-01.13 after WS-03.11), 5 (phase A: WS-01.11, WS-01.12).

**Depends on:** none at workstream level (skeleton). Task level: WS-01.9 consumes WS-02.3 and WS-02.4 (schema_check and the migrate entrypoint the migrate image carries) and, for its `--compose` mode, WS-03's W0 phase-A2 compose sync; WS-01.10 consumes WS-03's W4 compose digest pins; WS-01.13 consumes WS-03.11's guard rules; WS-01.12 consumes WS-06's single topic registry and WS-03's strict overlay. The owner cases in `tests/build/test_import_roots.py` and `tests/build/test_mvp_038_lint_gate.py` are turned green by WS-02, WS-05, WS-06, WS-10, WS-12 to WS-18 in their own tasks (Cross-workstream requests).

**Owned paths:** `requirements.txt`, `requirements-runtime.txt` (not created; superseded by `requirements.lock`), `requirements.in`, `requirements.lock`, `requirements-dev.in`, `requirements-dev.lock`, `pyproject.toml`, `pytest.ini`, `Makefile`, `.dockerignore`, `.gitignore`, `.pre-commit-config.yaml`, `.github`, `docker`, `scripts/ci`, `__init__.py`, `src/__init__.py`, `src/py.typed`, `src/solace_infrastructure/__init__.py`, `src/solace_infrastructure/py.typed`, `services/__init__.py`, `infrastructure/__init__.py`, `infrastructure/api_gateway/{__init__,kong_config,routes,auth_plugin,cors}.py`, `infrastructure/api_gateway/tests`, `tests/build`, `tests/test_syntax_and_imports.py`, and the `Dockerfile` and `requirements.txt` of orchestrator_service, safety_service, therapy_service, diagnosis_service, memory_service, personality_service, user-service, notification-service and analytics-service.

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Conventions used in every task.** Commands run from `D:\Repo\Contextual-Chatbot` (the PowerShell tool's working directory) with Windows PowerShell 5.1. The Python interpreter is always `& "D:\tmp\mvp-exec\venv\Scripts\python.exe"`; `uv` is `& "C:\Users\rayyan.a\AppData\Local\anaconda3\Scripts\uv.exe"` (0.8.2). Real-infra commands add `$env:SOLACE_REQUIRE_REALINFRA="1"` so a missing Docker fails instead of skipping. Generated scratch files go under `D:\tmp\mvp-exec\ws01\`, never inside the repo. Never print `.env` or any generated secret value.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `.gitignore` | Modify | Anchor packaging and virtualenv directory patterns to the root; replace `checkpoint*`; keep every secret pattern (01.1); WS-03's encrypted and public-recipient files under `deploy/secrets/` visible, age keys and decrypted files ignored (01.14) | WS-01.1, WS-01.14 |
| `tests/build/conftest.py` | Create | Fixtures `repo_root` and `require_docker` | WS-01.1 |
| `tests/build/test_mvp_008_git_visibility.py` | Create | Planned paths are visible to git; secrets stay ignored | WS-01.1 |
| `requirements.in` | Create | Direct runtime dependencies (D-16 pins) plus `-r` of the nine service extras files | WS-01.2 |
| `requirements.lock` | Create | Universal, hash-pinned runtime lock for Python 3.12 | WS-01.2 |
| `requirements-dev.in` | Create | Runtime set plus test and lint tools | WS-01.2 |
| `requirements-dev.lock` | Create | Universal, hash-pinned dev lock | WS-01.2 |
| `requirements.txt` | Modify | One-line pointer `-r requirements-dev.lock` | WS-01.2 |
| `pyproject.toml` | Modify | No competing dependency lists (01.2); no pytest table (01.4); coverage gate config (01.11) | WS-01.2, WS-01.4, WS-01.11 |
| `services/<9 MVP services>/requirements.txt` | Modify | Unpinned service extras only, all members of the lock | WS-01.2 |
| `scripts/ci/__init__.py` | Create | Package marker | WS-01.2 |
| `scripts/ci/lock_delta.py` | Create | Lock parser, constraints and pins writers, venv delta, D-20 venv sync list | WS-01.2 |
| `scripts/ci/clean_venv.py` | Create | Lock-only venv `D:\tmp\mvp-exec\venv-clean` for the W0-barrier proof (D-20) | WS-01.2 |
| `tests/build/test_mvp_005_dependency_lock.py` | Create | Lock content, security floor, service extras, import closure, local parity | WS-01.2 |
| `src/solace_infrastructure/__init__.py` | Modify | PEP 562 lazy exports | WS-01.3 |
| `infrastructure/__init__.py` | Modify | Lazy `api_gateway` | WS-01.3 |
| `infrastructure/api_gateway/__init__.py` | Modify | PEP 562 lazy exports | WS-01.3 |
| `tests/build/test_mvp_310_lazy_exports.py` | Create | Import-cost and export-completeness checks | WS-01.3 |
| `scripts/ci/suites.py` | Create | SUITES registry (IF-21 suite list) | WS-01.4 |
| `scripts/ci/run_suites.py` | Create | One pytest process per suite, per tier; coverage mode in 01.11 | WS-01.4, WS-01.11 |
| `scripts/ci/ci_local.py` | Create | `make ci-local` stage runner with a summary table | WS-01.4, WS-01.9, WS-01.10, WS-01.13, WS-01.11 |
| `pytest.ini` | Modify | Register markers realinfra, e2e, clinical, profile (01.4); drop `--cov=src` (01.11) | WS-01.4, WS-01.11 |
| `Makefile` | Modify | IF-21 targets | WS-01.4, WS-01.9, WS-01.10, WS-01.13, WS-01.11 |
| `tests/build/test_mvp_244_suite_runner.py` | Create | Markers, pyproject table removal, Makefile targets, per-suite processes | WS-01.4 |
| `services/<9 MVP services>/Dockerfile` | Modify | Shared template (01.5); entrypoint, uvicorn flags, `/live` healthcheck (01.6) | WS-01.5, WS-01.6 |
| `.dockerignore` | Modify | Keep secrets (incl. generated secret dirs), tests, model caches and docs out of every context; never package data (01.5); `deploy/secrets` and age keys out of every context (01.14) | WS-01.5, WS-01.14 |
| `scripts/ci/images.py` | Create | IMAGES registry and `probe_env()`; migrate entry added in 01.9 | WS-01.5, WS-01.9 |
| `scripts/ci/dockerfiles.py` | Create | Dockerfile parser and runtime-layout replica | WS-01.5 |
| `scripts/ci/build_images.py` | Create | `docker build` (repo-root context), in-image import probes and the image-reference format (`image_ref`, `COMPOSE_TAG`) that WS-03.2's compose `image:` values use | WS-01.5 |
| `tests/build/test_mvp_032_dockerfiles.py` | Create | Static template checks; migrate case added in 01.9 | WS-01.5, WS-01.9 |
| `tests/build/test_mvp_013_image_layout.py` | Create | Replica import of each CMD target and probe module | WS-01.5 |
| `docker/entrypoint.sh` | Create | NAME_FILE loader, then `exec "$@"` | WS-01.6 |
| `tests/build/ws_probe_app.py` | Create | Minimal ASGI app for the log-redaction probe | WS-01.6 |
| `tests/build/test_mvp_211_container_entry.py` | Create | Uvicorn flags never log query tokens; entrypoint semantics in a container | WS-01.6 |
| `src/__init__.py` | Modify | No eager subpackage imports | WS-01.7 |
| `tests/build/test_import_roots.py` | Create | No `src.solace_*` import; one case per owning workstream | WS-01.7 |
| `tests/build/test_mvp_038_lint_gate.py` | Create | ruff E9,F63,F7,F82 clean; one case per owning workstream | WS-01.7 |
| `.github/workflows/ci.yml` | Modify | plan, lint, unit, realinfra, images, e2e jobs (01.8); supply chain (01.10); iac job (01.13); coverage (01.11) | WS-01.8, WS-01.10, WS-01.13, WS-01.11 |
| `tests/build/test_mvp_038_ci_workflow.py` | Create | CI mirrors the runner and registries | WS-01.8 |
| `docker/migrate.Dockerfile` | Create | `solace-migrate` image (reads WS-02's migration files as COPY sources only) | WS-01.9 |
| `scripts/ci/clean_checkout.py` | Create | Build and probe every image from a git-archive export of a tree | WS-01.9 |
| `scripts/ci/check_clean_checkout.ps1` | Create | PowerShell wrapper for clean_checkout.py | WS-01.9 |
| `scripts/ci/check_tracked_closure.py` | Create | Report tracked files that depend on untracked files | WS-01.9 |
| `tests/build/test_mvp_008_clean_checkout.py` | Create | Closure tool on synthetic repos; clean-checkout build of a tree | WS-01.9 |
| `scripts/ci/supply_chain.py` | Create | pip-audit of both locks, image scan, SBOM, gate | WS-01.10 |
| `scripts/ci/accepted_vulns.yaml` | Create | Accepted findings with owner and expiry | WS-01.10 |
| `.github/dependabot.yml` | Create | Update PRs for pip, docker and actions | WS-01.10 |
| `tests/build/test_mvp_113_supply_chain.py` | Create | Gate logic, acceptance expiry, dependabot, compose digests | WS-01.10 |
| `scripts/ci/coverage_gate.py` | Create | Total and safety-critical coverage floors | WS-01.11 |
| `tests/build/test_mvp_243_coverage_gate.py` | Create | Gate logic and configuration | WS-01.11 |
| `tests/build/test_mvp_269_topic_registry.py` | Create | Closure check for the Kafka-topic part of MVP-269 | WS-01.12 |
| `scripts/ci/iac_lint.py` | Create | cfn-lint and cfn-guard over the staging CloudFormation template | WS-01.13 |
| `tests/build/test_mvp_043_iac_lint_stage.py` | Create | IaC lint commands, guard-binary policy, ci-local/Makefile/CI wiring, config-service out of the image matrix | WS-01.13 |
| `scripts/ci/secret_scan.py` | Create | Scan of every checkpoint-visible file for age private keys, stray or decrypted files under `deploy/secrets/` and SOPS files with plaintext values (D-07) | WS-01.14 |
| `tests/build/test_d07_sops_tree_hygiene.py` | Create | SOPS ignore rules, build-context exclusions, scanner rules and a clean working tree | WS-01.14 |

## Tasks

### Task WS-01.1: Make every planned platform path visible to git
**Covers:** MVP-008
**Severity:** blocker
**Wave:** 0
**Phase:** A1 (the first WS-01 task; it must land before any `tests/build` file exists and before WS-03 creates `deploy/env/`)
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-17 (decided 2026-10-19: no commits at all during the program, not even by the owner at barriers. The working tree plus the commit-free checkpoint trees that `checkpoint.ps1` writes through `git add -A` into a temporary index are the only record, barrier notes report the tree id only, and clean-checkout checks build from `git archive <checkpoint tree id>`, so no planned file may be gitignored)
**Real infra:** no
**Files:**
- Modify: `.gitignore` (anchor these directory patterns to the repository root with a leading `/`: `build/`, `develop-eggs/`, `dist/`, `downloads/`, `eggs/`, `lib/`, `lib64/`, `parts/`, `sdist/`, `var/`, `wheels/`, `env/`, `venv/`, `ENV/`, `env.bak/`, `venv.bak/`, `myenv/`, `.myenv/`, `new_venv/`; replace the line `checkpoint*` with `checkpoint-*/` and `*.ckpt`, keeping `checkpoint/`; leave every secret, cache, log and data pattern unchanged)
- Create: `tests/build/conftest.py`
- Test: `tests/build/test_mvp_008_git_visibility.py`
**Interfaces:**
- Consumes: none.
- Produces: fixture `repo_root() -> pathlib.Path` and fixture `require_docker() -> None` in `tests/build/conftest.py` (skips when Docker is unreachable, calls `pytest.fail` instead when `SOLACE_REQUIRE_REALINFRA=1`), used by every later `tests/build` task.

Why: `checkpoint.ps1` stages the working tree with `git add -A` into a temporary index, so ignored files are silently left out of checkpoint trees, review packages, `rollback.ps1` restores and the git-archive builds of checkpoint trees, which are the program's only record (D-17: nothing is committed). Today `.gitignore` line 17 `build/` hides `tests/build/`, line 40 `ENV/` (case-insensitive on Windows; line 38 `env/` on Linux) hides WS-03's `deploy/env/`, and line 182 `checkpoint*` hides WS-10's `services/orchestrator_service/src/langgraph/checkpointer.py`. This is a new defect found while planning; the controller adds it to the register as a new MVP id at the W0 barrier.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_008_git_visibility.py::test_mvp_008_planned_path_is_not_gitignored` and `::test_mvp_008_secret_and_artifact_paths_stay_ignored`
```python
"""MVP-008: every planned platform path must be visible to git (checkpoint trees and their git-archive builds; D-17: no commits)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

MUST_TRACK = [
    "tests/build/test_mvp_008_git_visibility.py",
    "tests/build/conftest.py",
    "deploy/env/env-contract.yaml",
    "deploy/env/render_env_example.py",
    "deploy/env/generate_dev_env.py",
    "services/orchestrator_service/src/langgraph/checkpointer.py",
    "migrations/versions/017_orchestrator_checkpoints.py",
    "docker/entrypoint.sh",
    "docker/migrate.Dockerfile",
    "scripts/ci/run_suites.py",
    "scripts/db/migrate.py",
    "infrastructure/init-db/01-roles.sh",
    "requirements.lock",
    "requirements-dev.lock",
    "src/solace_infrastructure/runtime/bootstrap.py",
    "tests/e2e/expected_failures.yaml",
    "tests/realinfra/data-platform-schema/test_x.py",
]
MUST_IGNORE = [
    ".env", ".env.prod", "deploy/env/.env.dev", "secrets/x.txt", "server.key", "postgres-ca.crt",
    ".venv/x", "venv/x", "env/bin/python", "build/lib/x.py", "__pycache__/x.pyc",
    "tests/logs/pytest.log", "checkpoint/model.bin", "checkpoint-500/weights.bin", "model.ckpt",
]


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", path],
        capture_output=True, text=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", MUST_TRACK)
def test_mvp_008_planned_path_is_not_gitignored(path: str) -> None:
    assert not _ignored(path), f"{path} is gitignored, so checkpoint trees and their git-archive builds would drop it"


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_mvp_008_secret_and_artifact_paths_stay_ignored(path: str) -> None:
    assert _ignored(path), f"{path} must stay ignored"
```
`tests/build/conftest.py` defines `REPO_ROOT = Path(__file__).resolve().parents[2]`, the session fixture `repo_root` returning it, and the session fixture `require_docker` that calls `docker.from_env().ping()` (the `docker` SDK ships with testcontainers) and on any exception calls `pytest.fail(...)` when `os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1"`, else `pytest.skip(...)`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_008_git_visibility.py -o "addopts=" -p no:logging -q`
Expected: FAIL, 6 cases of `test_mvp_008_planned_path_is_not_gitignored`: the two `tests/build/...` paths (`.gitignore:17 build/`), the three `deploy/env/...` paths (`.gitignore:40 ENV/`), and `checkpointer.py` (`.gitignore:182 checkpoint*`). Every `MUST_IGNORE` case passes.
- [ ] **Step 3: Implement** the `.gitignore` edits listed under Files, plus `tests/build/conftest.py`. Anchoring with a leading `/` keeps the root-level virtualenv and packaging directories ignored while freeing same-named directories deeper in the tree; do not touch lines 36, 51 to 66 (secrets) or the log, cache and data sections.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** confirm nothing sensitive became visible. Run: `git -C D:\Repo\Contextual-Chatbot ls-files --others --exclude-standard | Select-String -Pattern '(^|/)\.env|\.key$|\.crt$|\.pem$|/logs/|/models/cache/'` (read-only; prints names only). Expected: no output. Then run the owning suite: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.2: One hash-pinned dependency lock matching the tested versions
**Covers:** MVP-005, MVP-012, MVP-358, MVP-359, MVP-312
**Severity:** blocker
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-16 (default assumed: one universal uv lock for Python 3.12 pinned to the versions the local suites pass on, plus PyJWT >= 2.13, python-jose out of the runtime set, and the missing email-validator, aiokafka, langgraph-checkpoint-postgres with psycopg, pyotp; no torch, transformers, VADER or spaCy in images). D-20 (decided 2026-10-19: a dedicated venv, and anaconda stays untouched. At W0 entry, as a protocol step before this task, the controller creates `D:\tmp\mvp-exec\venv` with `& "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m venv --system-site-packages D:\tmp\mvp-exec\venv`. It installs into that venv only the missing or vulnerable packages at their D-16 pins: aiokafka, langgraph-checkpoint-postgres (with psycopg), PyJWT >= 2.13, and any other lock-pinned test extra the suites need. It records the venv's `pip freeze` before and after, and re-records the per-suite baseline in the venv; a count drop needs a ledger Ruling. This task then re-syncs that venv to the dev lock (Step 3 item 6) and adds the W0-barrier run in a clean, lock-only venv `D:\tmp\mvp-exec\venv-clean` without system site packages (Step 3 item 8), where any failure is a lock defect. Nothing is ever installed into anaconda: never run `pip install` with the anaconda interpreter. Running anaconda's `uv.exe` binary is fine, because it installs nothing into anaconda). D-04 (decided 2026-10-19: real-user LLM traffic goes to Anthropic models on AWS Bedrock, so `boto3` and `botocore` become runtime-lock pins for WS-09's Bedrock provider. They are pinned at the versions installed in the anaconda env, which are boto3 1.40.61 and botocore 1.40.61 as read on 2026-10-19; re-read them at compile time with `pip show` (read-only). The one exception: if an advisory covers that version, pin the lowest fixed version instead and record the bump as a ledger Ruling)
**Real infra:** no
**Files:**
- Create: `requirements.in`, `requirements.lock`, `requirements-dev.in`, `requirements-dev.lock`, `scripts/ci/__init__.py`, `scripts/ci/lock_delta.py`, `scripts/ci/clean_venv.py`
- Modify: `requirements.txt` (becomes the single line `-r requirements-dev.lock` under a header comment)
- Modify: `pyproject.toml` (`[project].dependencies = []` with a comment pointing at `requirements.in`; delete `[project.optional-dependencies]`; nothing else)
- Modify: `services/orchestrator_service/requirements.txt`, `services/safety_service/requirements.txt`, `services/therapy_service/requirements.txt`, `services/diagnosis_service/requirements.txt`, `services/memory_service/requirements.txt`, `services/personality_service/requirements.txt`, `services/user-service/requirements.txt`, `services/notification-service/requirements.txt`, `services/analytics-service/requirements.txt` (unpinned service extras only)
- Test: `tests/build/test_mvp_005_dependency_lock.py`
**Interfaces:**
- Consumes: the W0-entry venv `D:\tmp\mvp-exec\venv` (D-20). Its installed distributions are the D-16 baseline: anaconda's, seen read-only through the system site packages, plus the controller's W0-entry installs.
- Produces (IF-21): `requirements.lock` and `requirements-dev.lock`, each headed by the line `#    uv pip compile <in-file> --universal --generate-hashes --python-version 3.12 -o <lock>`; `scripts/ci/lock_delta.py` with `parse_lock(path: Path) -> dict[str, LockEntry]`, `LockEntry(name: str, version: str, marker: str | None, hashes: tuple[str, ...], via: tuple[str, ...])`, `compute_delta(lock: Path, *, allow_missing: frozenset[str] = frozenset()) -> Delta` with `Delta(missing: dict[str, str], mismatched: dict[str, tuple[str, str]])`, `exclusive_closure(entries, roots: set[str]) -> set[str]`, and CLI flags `--write-constraints OUT --exclude a,b`, `--pins-only LOCK OUT`, `--lock LOCK --write-install OUT [--allow-missing a,b]`. Service extras files hold bare names; every name resolves in the lock.
- Produces for requesting tasks (pins land here in W0 phase A1, and Step 3 item 6 re-syncs the venv to them per D-20): R-02.1 alembic 1.16.2, sqlalchemy[asyncio] 2.0.42, asyncpg 0.30.0, psycopg[binary], cryptography, testcontainers 4.14.2 and docker 7.1.0 for WS-02.1 and WS-02.4. `boto3` and `botocore` (runtime lock, D-04) for WS-09.18, WS-09's Bedrock Converse provider (its CR-16), including its `botocore.stub.Stubber` tests, and for WS-05.1's Converse-route test and WS-05.19's release-gate driver. This also covers WS-03's original dev-lock `boto3` request for WS-03.8. WS-05 request #1 aiokafka, websockets, jsonschema, bcrypt, pyotp, PyYAML, httpx and testcontainers for WS-05.2 and WS-05.4 (without the `redis` extra, see New interfaces); WS-06 #11 aiokafka and prometheus-client for WS-06.4 and WS-06.10; WS-10's langgraph 1.0.x, langgraph-checkpoint-postgres (exact pin, bumped only with a slot-017 follow-up revision), psycopg[binary], psycopg-pool and PyYAML for WS-10.1, WS-10.9 and WS-10.10; WS-04's prometheus-client for WS-04.2; WS-12 R-12-4 PyYAML and hypothesis for WS-12.3; pyotp, PyJWT >= 2.13 and httpx for WS-08.16 (its `WS-01:pyotp-lock` dependency is this task); `cfn-lint` (dev lock) for WS-01.13 and WS-03.11.
- Produces (D-20): `scripts/ci/clean_venv.py` (stdlib only), with these members:
  - `VENV_CLEAN = Path(r"D:\tmp\mvp-exec\venv-clean")`;
  - `venv_python(venv: Path) -> Path`, which is `Scripts\python.exe` on Windows and `bin/python` elsewhere;
  - `commands(*, base_python: str, uv: str, venv: Path, lock: Path) -> list[list[str]]`;
  - `is_isolated(pyvenv_cfg: Path) -> bool`;
  - the CLI `clean_venv.py --build [--venv DIR] [--lock PATH] [--base-python PATH] [--uv PATH]`.

  It also produces the W0-barrier commands in Step 5, which the controller runs at the W0 barrier, so the lock is proven complete without anaconda's packages.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_005_dependency_lock.py`
```python
"""MVP-005/012/312/358/359: one hash-pinned lock that resolves, matches the tested versions and ships no known-bad pins."""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
LOCK, DEV_LOCK = ROOT / "requirements.lock", ROOT / "requirements-dev.lock"
MVP_SERVICES = ["orchestrator_service", "safety_service", "therapy_service", "diagnosis_service",
                "memory_service", "personality_service", "user-service", "notification-service",
                "analytics-service"]
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?==([^\s;\\]+)")
# top-level import name -> candidate distributions (any one in the lock satisfies it)
IMPORT_TO_DIST = {
    "fastapi": {"fastapi"}, "starlette": {"starlette"}, "uvicorn": {"uvicorn"}, "pydantic": {"pydantic"},
    "pydantic_settings": {"pydantic-settings"}, "structlog": {"structlog"}, "httpx": {"httpx"},
    "asyncpg": {"asyncpg"}, "sqlalchemy": {"sqlalchemy"}, "alembic": {"alembic"}, "redis": {"redis"},
    "aiokafka": {"aiokafka"}, "cryptography": {"cryptography"}, "jwt": {"pyjwt"},
    "weaviate": {"weaviate-client"}, "bm25s": {"bm25s"}, "langgraph": {"langgraph"},
    "langchain_core": {"langchain-core"}, "langchain_anthropic": {"langchain-anthropic"},
    "portkey_ai": {"portkey-ai"}, "email_validator": {"email-validator"}, "argon2": {"argon2-cffi"},
    "passlib": {"passlib"}, "jinja2": {"jinja2"}, "aiosmtplib": {"aiosmtplib"}, "flashtext": {"flashtext"},
    "opentelemetry": {"opentelemetry-api"}, "prometheus_fastapi_instrumentator": {"prometheus-fastapi-instrumentator"},
    "prometheus_client": {"prometheus-client"}, "typing_extensions": {"typing-extensions"},
    "psycopg": {"psycopg"}, "psycopg_pool": {"psycopg-pool"}, "pyotp": {"pyotp"}, "yaml": {"pyyaml"},
    "boto3": {"boto3"}, "botocore": {"botocore"},  # D-04 Bedrock Converse provider (WS-09)
}
EXCLUDED = {  # imported somewhere, deliberately not in any image
    "torch": "D-16", "transformers": "D-16", "vaderSentiment": "D-16", "spacy": "D-16",
    "sentence_transformers": "D-16", "clickhouse_connect": "D-19 ClickHouse out of scope",
    "google": "push/FCM out of scope", "jose": "Kong out of scope (D-19); dev lock only",
    "aiohttp": "dead path in solace_infrastructure.kafka.schemas (MVP-269)",
}
RUNTIME_DIRS = ["src", "services", "migrations", "scripts/db", "infrastructure/api_gateway"]
LOCAL_MISSING_OK_ROOTS: set[str] = set()  # D-20: the venv is re-synced to the full dev lock (portkey-ai included)


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PIN.match(line)
        if m:
            pins.setdefault(canonicalize_name(m.group(1)), set()).add(m.group(2))
    forks = {k: v for k, v in pins.items() if len(v) > 1}
    assert not forks, f"forked pins: {forks}"
    return {k: next(iter(v)) for k, v in pins.items()}


def _service_names(svc: str) -> list[str]:
    names = []
    for raw in (ROOT / "services" / svc / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            assert not re.search(r"[=<>~!@]", line), f"{svc}: version specifier in {line!r}"
            names.append(canonicalize_name(re.split(r"[\[\s]", line, 1)[0]))
    return names


def test_mvp_005_lock_files_are_uv_hash_pinned() -> None:
    for path, src in ((LOCK, "requirements.in"), (DEV_LOCK, "requirements-dev.in")):
        text = path.read_text(encoding="utf-8")
        assert f"uv pip compile {src} --universal --generate-hashes --python-version 3.12 -o {path.name}" in text
        blocks = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 1)[1])
        reqs = [b for b in blocks if PIN.match(b)]
        assert reqs and all("--hash=sha256:" in b for b in reqs)


def test_mvp_358_lock_pins_the_tested_baseline() -> None:
    pins = _pins(LOCK)
    assert pins["langgraph"] == "1.0.4"
    assert pins["langgraph-checkpoint"] == "3.0.1"
    assert pins["langchain-core"] == "1.2.28"
    assert pins["fastapi"] == "0.128.0"
    assert pins["pydantic-settings"] == "2.12.0"
    assert pins["asyncpg"] == "0.30.0"
    assert pins["pydantic"].startswith("2.13.")
    assert pins["redis"].startswith("5.3.")
    assert pins["weaviate-client"].startswith("4.19.")
    assert Version(pins["ormsgpack"]) >= Version("1.12.0")
    assert Version(pins["langgraph-checkpoint-postgres"]) >= Version("3.0.0")


def test_mvp_358_checkpoint_postgres_is_exact_pinned_for_slot_017() -> None:
    # WS-10: slot 017 embeds this version's MIGRATIONS; a bump needs a slot-017 follow-up revision
    text = (ROOT / "requirements.in").read_text(encoding="utf-8")
    m = re.search(r"^langgraph-checkpoint-postgres==([0-9.]+)\s+#.*slot-017", text, re.M)
    assert m, "langgraph-checkpoint-postgres must be pinned with == and a slot-017 comment"
    assert _pins(LOCK)["langgraph-checkpoint-postgres"] == m.group(1)


def test_mvp_359_312_security_floor() -> None:
    pins = _pins(LOCK)
    assert Version(pins["pyjwt"]) >= Version("2.13.0")
    for banned in ("python-jose", "ecdsa", "torch", "transformers", "vadersentiment", "spacy", "sentence-transformers"):
        assert banned not in pins, banned
    dev = _pins(DEV_LOCK)
    assert Version(dev.get("python-jose", "3.5.0")) >= Version("3.4.0")


def test_mvp_005_required_runtime_packages_are_locked() -> None:
    pins = _pins(LOCK)
    for name in ("email-validator", "aiokafka", "langgraph-checkpoint-postgres", "psycopg", "psycopg-pool",
                 "pyotp", "prometheus-client", "alembic", "sqlalchemy", "aiosmtplib", "jinja2", "argon2-cffi",
                 "passlib", "bcrypt", "bm25s", "weaviate-client", "portkey-ai", "uvicorn", "structlog",
                 "httpx", "cryptography", "pyyaml", "flashtext", "boto3", "botocore",
                 "cffi", "pycparser"):  # cffi/pycparser: hashed for WS-03.8's sops tools image (D-07)
        assert name in pins, name


def test_d04_bedrock_sdk_is_an_exact_runtime_pin() -> None:
    # D-04: the version is the one installed in the anaconda env (pip show), unless an advisory forces a bump
    text, pins = (ROOT / "requirements.in").read_text(encoding="utf-8"), _pins(LOCK)
    for name in ("boto3", "botocore"):
        m = re.search(rf"^{name}==([0-9.]+)\s+#.*D-04", text, re.M)
        assert m, f"{name} must be pinned with == and a D-04 comment in requirements.in"
        assert pins[name] == m.group(1), name


def test_mvp_005_dev_lock_contains_runtime_lock_at_same_versions() -> None:
    runtime, dev = _pins(LOCK), _pins(DEV_LOCK)
    assert {k: (v, dev.get(k)) for k, v in runtime.items() if dev.get(k) != v} == {}
    for name in ("pytest", "pytest-asyncio", "pytest-timeout", "hypothesis", "testcontainers", "docker",
                 "websockets", "fakeredis", "ruff", "pip-audit", "packaging", "jsonschema", "boto3", "cfn-lint"):
        assert name in dev, name


@pytest.mark.parametrize("svc", MVP_SERVICES)
def test_mvp_005_service_requirements_are_unpinned_lock_members(svc: str) -> None:
    pins = _pins(LOCK)
    for name in _service_names(svc):
        assert name in pins, f"{svc}: {name}"
        assert not name.startswith("pytest"), f"{svc}: test tool {name}"


def test_mvp_012_notification_extras_include_email_validator() -> None:
    assert "email-validator" in _service_names("notification-service")


def test_mvp_005_root_requirements_points_to_dev_lock() -> None:
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert lines == ["-r requirements-dev.lock"]


def test_mvp_359_pyproject_declares_no_competing_pins() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project.get("dependencies", []) == []
    assert "optional-dependencies" not in project


def _is_local(name: str) -> bool:
    candidates = [ROOT / name, ROOT / "src" / name, ROOT / "services" / name]
    candidates += [d / "src" / name for d in (ROOT / "services").iterdir() if d.is_dir()]
    return any(c.exists() or c.with_suffix(".py").exists() for c in candidates) or name.startswith("solace_")


def test_mvp_005_runtime_imports_resolve_from_lock() -> None:
    pins, unknown, missing = _pins(LOCK), {}, {}
    for base in RUNTIME_DIRS:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if ("/tests/" in rel or path.name.startswith("test_")
                    or rel.startswith(("services/config_service/", "src/solace_testing/"))):
                continue  # test code, the frozen config-service and the dev-only solace_testing package
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                for top in {n.split(".")[0] for n in names}:
                    if top in sys.stdlib_module_names or _is_local(top) or top in EXCLUDED:
                        continue
                    dists = IMPORT_TO_DIST.get(top)
                    if dists is None:
                        unknown.setdefault(top, rel)
                    elif not dists & set(pins):
                        missing.setdefault(top, rel)
    assert unknown == {}, f"map these imports in IMPORT_TO_DIST or EXCLUDED: {unknown}"
    assert missing == {}, f"imported but not locked: {missing}"


def test_d16_local_interpreter_matches_dev_lock() -> None:
    from scripts.ci.lock_delta import compute_delta, exclusive_closure, parse_lock

    entries = parse_lock(DEV_LOCK)
    delta = compute_delta(DEV_LOCK, allow_missing=frozenset(exclusive_closure(entries, LOCAL_MISSING_OK_ROOTS)))
    assert delta.mismatched == {}, f"installed version differs from the lock: {delta.mismatched}"
    assert delta.missing == {}, f"locked but not installed (D-20 venv sync list): {delta.missing}"


def test_d20_clean_venv_is_built_only_from_the_dev_lock(tmp_path: Path) -> None:
    from scripts.ci import clean_venv

    venv = tmp_path / "venv-clean"
    cmds = clean_venv.commands(base_python="base-py", uv="uv", venv=venv, lock=DEV_LOCK)
    assert cmds == [["base-py", "-m", "venv", "--clear", str(venv)],
                    ["uv", "pip", "sync", "--python", str(clean_venv.venv_python(venv)), "--require-hashes", str(DEV_LOCK)]]
    assert not any("system-site-packages" in part for cmd in cmds for part in cmd)
    assert clean_venv.VENV_CLEAN == Path(r"D:\tmp\mvp-exec\venv-clean")
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("home = C:\\base\ninclude-system-site-packages = true\n", encoding="utf-8")
    assert clean_venv.is_isolated(cfg) is False
    cfg.write_text("home = C:\\base\ninclude-system-site-packages = false\n", encoding="utf-8")
    assert clean_venv.is_isolated(cfg) is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_005_dependency_lock.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Every test that reads the lock (including all nine `test_mvp_005_service_requirements_are_unpinned_lock_members` cases) raises `FileNotFoundError` for `requirements.lock`, and `test_mvp_358_checkpoint_postgres_is_exact_pinned_for_slot_017` raises it for `requirements.in`; `test_mvp_012_notification_extras_include_email_validator` fails on `version specifier in 'fastapi==0.128.0'` (services/notification-service/requirements.txt pins everything and lacks email-validator); `test_mvp_005_root_requirements_points_to_dev_lock` fails because `requirements.txt` pins `langgraph~=0.2.76` and `PyJWT~=2.9.0`; `test_mvp_359_pyproject_declares_no_competing_pins` fails on `PyJWT>=2.8.0`; `test_d16_local_interpreter_matches_dev_lock` fails with `ModuleNotFoundError: scripts.ci.lock_delta`. Two new tests also fail: `test_d04_bedrock_sdk_is_an_exact_runtime_pin` raises `FileNotFoundError` for `requirements.in`, and `test_d20_clean_venv_is_built_only_from_the_dev_lock` fails with `ModuleNotFoundError` for `scripts.ci.clean_venv`. No lock pins the Bedrock SDK, and nothing proves that the lock alone is a complete environment.
- [ ] **Step 3: Implement** in this order:
  1. `scripts/ci/__init__.py` (docstring only) and `scripts/ci/lock_delta.py` (stdlib plus `packaging`; reads uv's split annotation style `# via` / `#   name`; evaluates markers with `packaging.markers.Marker.evaluate()` for the running interpreter; per distribution name the first copy on `sys.path` wins, as `importlib.metadata.version()` resolves it, so the venv's own copy shadows anaconda's; `--write-constraints` writes `name==version` for every installed distribution from `importlib.metadata`, UTF-8 without BOM, skipping `--exclude`; `--write-install` writes `name==version --hash=...` lines for the missing and mismatched entries that are not allow-listed).
  2. Service extras (bare names, comments allowed): orchestrator `langgraph`, `langgraph-checkpoint`, `langgraph-checkpoint-postgres`, `psycopg[binary]`, `psycopg-pool`, `langchain-core`; safety `flashtext`, `langchain-core`, `langchain-anthropic`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`; memory `weaviate-client`, `bm25s`, `sqlalchemy[asyncio]`; user-service `argon2-cffi`, `passlib[bcrypt]`, `bcrypt`, `email-validator`; notification `aiosmtplib`, `jinja2`, `email-validator`; analytics `prometheus-fastapi-instrumentator`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`; therapy, diagnosis, personality: comment only.
  3. `requirements.in`: header comment (D-16), then `fastapi==0.128.0`, `starlette==0.41.3`, `uvicorn[standard]==0.40.0`, `pydantic==2.13.4`, `pydantic-settings==2.12.0`, `python-multipart==0.0.20`, `structlog==25.5.0`, `httpx==0.28.1`, `asyncpg==0.30.0`, `sqlalchemy[asyncio]==2.0.42`, `alembic==1.16.2`, `psycopg[binary]==3.3.4`, `psycopg-pool==3.3.1`, `redis==5.3.1`, `aiokafka~=0.12.0`, `langgraph==1.0.4`, `langgraph-checkpoint==3.0.1`, `langgraph-checkpoint-postgres==<3.0.x>` (exact: compile once with `~=3.0.0`, then write back the resolved version as `==` with the trailing comment `# bump only with a slot-017 follow-up revision (WS-10)` and recompile), `langchain-core==1.2.28`, `langchain-anthropic==1.3.1`, `portkey-ai~=1.9.0`, `weaviate-client==4.19.2`, `bm25s==0.2.14`, `cryptography==46.0.6`, `PyJWT[crypto]>=2.13,<3`, `passlib[bcrypt]==1.7.4`, `bcrypt==4.3.0`, `email-validator==2.2.0`, `jinja2==3.1.5`, `aiosmtplib==3.0.2`, `pyotp==2.9.0`, `prometheus-client==0.21.1`, `PyYAML==6.0.3`, `flashtext==2.7`, `boto3==<v>  # D-04 Bedrock Converse provider (WS-09)` and `botocore==<v>  # D-04 Bedrock Converse provider (WS-09)`, then `-r services/<svc>/requirements.txt` for the nine services. For `boto3` and `botocore`, `<v>` is the version installed in the anaconda env. Read it with `& "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pip show boto3 botocore` (read-only; 1.40.61 for both on 2026-10-19). If an advisory covers that version, `<v>` is the lowest fixed version and the comment names the advisory. Every other version comes from the local freeze constraint, never typed by hand.
  4. `requirements-dev.in`: `-r requirements.in` plus `pytest==8.3.3`, `pytest-asyncio==1.3.0`, `pytest-timeout==2.4.0`, `pytest-cov==5.0.0`, `pytest-mock==3.15.1`, `hypothesis==6.165.10`, `fakeredis==2.36.1`, `testcontainers[postgres,kafka]==4.14.2` (no `redis` extra: it requires redis>=7, which conflicts with the D-16 redis 5.3.x pin), `docker==7.1.0`, `websockets==15.0.1`, `faker==37.4.2`, `ruff==0.6.9`, `pip-audit==2.9.0`, `packaging==24.2`, `jsonschema==4.26.0` (WS-05 contract framework), `cfn-lint==1.51.5` (WS-01.13, WS-03.11; both are the versions anaconda already holds, which the venv sees through its system site packages, so they add nothing to the venv sync list), and `python-jose[cryptography]==3.5.0` with the comment `# test-only: frozen Kong auth_plugin tests (D-19); never in requirements.lock`.
  5. Baseline constraints and compile:
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\lock_delta.py --write-constraints D:\tmp\mvp-exec\ws01\baseline-constraints.txt --exclude pyjwt,python-jose,ecdsa`
     - `& "C:\Users\rayyan.a\AppData\Local\anaconda3\Scripts\uv.exe" pip compile requirements.in --universal --generate-hashes --python-version 3.12 -c D:\tmp\mvp-exec\ws01\baseline-constraints.txt --custom-compile-command "uv pip compile requirements.in --universal --generate-hashes --python-version 3.12 -o requirements.lock" -o requirements.lock`
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\lock_delta.py --pins-only requirements.lock D:\tmp\mvp-exec\ws01\runtime-pins.txt`
     - `& "C:\Users\rayyan.a\AppData\Local\anaconda3\Scripts\uv.exe" pip compile requirements-dev.in --universal --generate-hashes --python-version 3.12 -c D:\tmp\mvp-exec\ws01\baseline-constraints.txt -c D:\tmp\mvp-exec\ws01\runtime-pins.txt --custom-compile-command "uv pip compile requirements-dev.in --universal --generate-hashes --python-version 3.12 -o requirements-dev.lock" -o requirements-dev.lock`
     - If a compile is unsatisfiable, relax only the conflicting constraint line, keep the pin the code was written against, and record the package in the ledger; a resolved version that differs from the venv enters the venv sync list below.
  6. Re-sync the venv to the dev lock (D-20). This writes into `D:\tmp\mvp-exec\venv` only; anaconda is never written.
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\lock_delta.py --lock requirements-dev.lock --write-install D:\tmp\mvp-exec\ws01\venv-sync.txt`
     - Expected list: what the venv still lacks after the controller's W0-entry install, above all portkey-ai and its exclusive dependencies. It contains nothing the W0-entry install already pinned. A runtime library listed at a version other than the one the venv holds can only come from a constraint relaxed in item 5. Record it in the ledger with that reason before installing.
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 D:\tmp\mvp-exec\ws01\venv-freeze-before-sync.txt`
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pip install --no-deps --require-hashes -r D:\tmp\mvp-exec\ws01\venv-sync.txt`. The venv's own pip writes only into the venv's site-packages, which shadow anaconda's. For a package anaconda also holds, pip reports that it is not uninstalling the copy outside the environment.
     - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pip freeze | Out-File -Encoding utf8 D:\tmp\mvp-exec\ws01\venv-freeze-after-sync.txt`
     - Never run `pip install` with the anaconda interpreter. Running anaconda's `uv.exe` binary (items 5 and 8) is fine, because it installs nothing into anaconda.
  7. `requirements.txt` becomes the pointer, and `pyproject.toml` loses its dependency lists.
  8. `scripts/ci/clean_venv.py` (stdlib only) per Produces:
     - `commands()` returns exactly `[[base_python, "-m", "venv", "--clear", str(venv)], [uv, "pip", "sync", "--python", str(venv_python(venv)), "--require-hashes", str(lock)]]`, and never `--system-site-packages`.
     - `is_isolated()` is true only when `pyvenv.cfg` says `include-system-site-packages = false`.
     - `--build` defaults are `VENV_CLEAN`, `requirements-dev.lock`, `C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe` and anaconda's `uv.exe`. The anaconda interpreter is only the base the new venv is created from; nothing is installed into it.
     - `--build` runs the two commands, then exits 1 unless `is_isolated(<venv>\pyvenv.cfg)`.
     - It prints package names and versions only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** aiokafka and langgraph-checkpoint-postgres first became importable through the controller's W0-entry venv install; the per-suite baseline was re-recorded then. The re-sync in item 6 makes more packages importable for the first time, portkey-ai among them, so run one pytest process per suite: `tests/build`, `tests/solace_events`, `tests/solace_infrastructure`, `tests/solace_security`, `tests/integration`, `services/orchestrator_service/tests`, `services/notification-service/tests`, `services/safety_service/tests`, `services/user-service/tests`. Each command is `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\<suite> -o "addopts=" -p no:logging -q`. Expected: all pass with counts at or above the W0-entry venv baseline. A failure caused only by a newly importable package is not fixed here; an example is a Kafka admin client that now tries `localhost:9092` instead of mock mode. Record it in the ledger with the test id and route it to its owner as a W0 finding: WS-06 for `solace_events`/`solace_infrastructure.kafka`, WS-10 for the checkpointer, WS-09 for `portkey-ai`, `boto3` or `botocore` in `services/shared/infrastructure/llm_client.py`. A count drop needs a ledger Ruling. The controller re-records the per-suite baseline after this task. Then smoke-test the clean venv once:
  - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\clean_venv.py --build`. Expected: exit 0.
  - `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv-clean\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_005_dependency_lock.py -o "addopts=" -p no:logging -q`. Expected: PASS. The parity test now checks that the lock-only venv holds exactly the locked versions.

  W0-barrier check (D-20), run by the controller at the W0 barrier once WS-01.4's runner exists:
  - `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\clean_venv.py --build`
  - `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv-clean\Scripts\python.exe" scripts\ci\run_suites.py --tier unit`
  - the same with `--tier realinfra --require-realinfra`.

  A suite that is green in `D:\tmp\mvp-exec\venv` but red in `D:\tmp\mvp-exec\venv-clean` is a lock defect owned by WS-01. Fix it with an `.in` change and a recompile, never by installing into the clean venv by hand. One exception: a failure that needs a package D-16 keeps out of images (torch, transformers, VADER, spaCy) goes to R-01.15's owner.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.3: Lazy package exports so the rate limiter and the Postgres client import without Kong, jose, weaviate or alembic
**Covers:** MVP-269, MVP-310, MVP-312
**Severity:** low
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.1
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-19 (default assumed: the Kong package stays unimported behind a lazy `__init__`; its modules are not edited)
**Real infra:** no
**Files:**
- Modify: `src/solace_infrastructure/__init__.py` (replace the eager imports at lines 2 to 239 with a `_EXPORTS: dict[str, tuple[str, str]]` map from each public name to `(module, attribute)`, keeping the aliases `KafkaTopicMetrics`, `ObsAlertSeverity`, `LogAggregationLevel`, `LogRetentionPolicy` and `DBBaseModel`; `__all__ = list(_EXPORTS)`; module-level `__getattr__(name)` imports on first access, caches in `globals()`, raises `AttributeError` for unknown names; `__dir__()` returns `sorted(set(globals()) | set(__all__))`)
- Modify: `infrastructure/__init__.py` (drop `from . import api_gateway`; `__getattr__("api_gateway")` imports the subpackage)
- Modify: `infrastructure/api_gateway/__init__.py` (same lazy pattern for its 19 exported names)
- Test: `tests/build/test_mvp_310_lazy_exports.py`
**Interfaces:**
- Consumes: none.
- Produces: `import solace_infrastructure.<submodule>` loads only that submodule's own imports; `from solace_infrastructure import X` and `from infrastructure.api_gateway import X` keep working for every current name (shared_files: lazy exports only, never new eager imports). The `solace_infrastructure.database` package is made lazy by WS-02.7 (supporting part of MVP-269). For requesting tasks: the package `__init__` loads neither FastAPI nor asyncpg, so `import solace_infrastructure.runtime` costs only its own imports (WS-04's request, for WS-04.2); the limiter imports without Kong (WS-07, and WS-11.1, whose `WS-01:lazy-api-gateway` dependency is this task).

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_310_lazy_exports.py`
```python
"""MVP-269/310/312: importing a leaf module must not drag in Kong, jose, weaviate, alembic or the Kafka admin stack."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATEWAY_NAMES = {"KongConfig", "KongSettings", "ServiceConfig", "UpstreamConfig", "RouteConfig", "RouteManager",
                 "RouteDefinition", "ServiceRoutes", "RateLimitPolicy", "RateLimiter", "RateLimitConfig",
                 "RateLimitResult", "JWTAuthPlugin", "JWTConfig", "TokenClaims", "AuthResult", "CORSConfig",
                 "CORSPolicy", "CORSHandler"}
JOSE_BLOCKER = (
    "import sys, importlib.abc\n"
    "class _Block(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'jose' or name.startswith('jose.'):\n"
    "            raise ImportError('blocked: jose')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}", "ENVIRONMENT": "test"}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)


def _loaded(import_line: str, prefixes: tuple[str, ...]) -> list[str]:
    r = _run(f"import json, sys\n{import_line}\n"
             f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({prefixes!r}))))")
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_mvp_310_rate_limiter_import_loads_no_kong_modules() -> None:
    assert _loaded("import infrastructure.api_gateway.rate_limiting",
                   ("jose", "httpx", "infrastructure.api_gateway.kong_config", "infrastructure.api_gateway.routes",
                    "infrastructure.api_gateway.auth_plugin", "infrastructure.api_gateway.cors")) == []


def test_mvp_312_rate_limiter_imports_without_python_jose() -> None:
    r = _run(JOSE_BLOCKER + "import infrastructure.api_gateway.rate_limiting\nprint('OK')")
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]


def test_mvp_269_postgres_import_loads_no_optional_backends() -> None:
    assert _loaded("import solace_infrastructure.postgres",
                   ("weaviate", "alembic", "aiokafka", "solace_infrastructure.kafka",
                    "solace_infrastructure.observability", "solace_infrastructure.weaviate",
                    "solace_infrastructure.redis", "solace_infrastructure.database")) == []


def test_mvp_269_package_import_is_free() -> None:
    assert _loaded("import solace_infrastructure",
                   ("solace_infrastructure.", "weaviate", "alembic", "aiokafka", "sqlalchemy", "asyncpg", "fastapi")) == []


def test_mvp_269_every_declared_export_still_resolves() -> None:
    r = _run("import solace_infrastructure as s\n"
             "missing = [n for n in s.__all__ if getattr(s, n, None) is None]\n"
             "assert {'PostgresClient', 'WeaviateClient', 'KafkaMonitor', 'MigrationRunner', 'DBBaseModel',\n"
             "        'ObsAlertSeverity', 'setup_redis'} <= set(s.__all__)\n"
             "print(missing)")
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]"


def test_mvp_310_gateway_exports_unchanged_and_lazy() -> None:
    r = _run("import infrastructure.api_gateway as g\n"
             f"assert set(g.__all__) == {GATEWAY_NAMES!r}\n"
             "from infrastructure.api_gateway import RateLimiter, KongConfig, JWTAuthPlugin, CORSHandler\n"
             "print('OK')")
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_310_lazy_exports.py -o "addopts=" -p no:logging -q`
Expected: FAIL in 4 tests: the rate-limiter import lists `jose`, `httpx` and the four Kong modules; the jose-blocked import exits with `ImportError: blocked: jose` raised from `infrastructure/api_gateway/__init__.py:18` via `auth_plugin.py:20`; `import solace_infrastructure.postgres` and `import solace_infrastructure` list `weaviate`, `alembic` and `solace_infrastructure.kafka` (the package list also holds `asyncpg`, imported by `postgres.py:14` through the eager `__init__.py:70`). The two export-completeness tests pass today and guard the refactor.
- [ ] **Step 3: Implement** the three lazy `__init__.py` files as described under Files. Ruff 0.6.9 skips F822 for a module that defines `__getattr__`, so the lint gate stays green. Do not edit any Kong module or `rate_limiting.py` (WS-07).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** one process per suite with the standard command for `tests/test_syntax_and_imports.py`, `infrastructure/api_gateway/tests`, `tests/solace_infrastructure` and `services/orchestrator_service/tests`, for example `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\infrastructure\api_gateway\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.4: pytest configuration, markers, the suite runner and the IF-21 Makefile
**Covers:** MVP-244, MVP-245
**Severity:** low
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.2
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `scripts/ci/suites.py` (stdlib only), `scripts/ci/run_suites.py`, `scripts/ci/ci_local.py`
- Modify: `pytest.ini` (append markers `realinfra`, `e2e`, `clinical` and `profile` to the `markers =` list, the last as `profile(name): e2e test for one compose profile (dev or strict); tests/e2e/conftest.py deselects non-matching items`; keep `contract` and every existing marker and option)
- Modify: `pyproject.toml` (delete the `[tool.pytest.ini_options]` table; nothing else)
- Modify: `Makefile` (rewrite: variables `PYTHON ?= python`, `PROFILE ?= dev`, `TAG ?= dev`, `TREE ?=`; targets `help`, `lint`, `test` (alias of `test-unit`), `test-unit`, `test-realinfra`, `test-e2e`, `test-clinical`, `test-contract`, `images`, `clean-checkout`, `migrate`, `ci-local`; every recipe is one `$(PYTHON) ...` command so GNU Make 3.81 on Windows runs it through cmd.exe and Linux runs it through sh)
- Test: `tests/build/test_mvp_244_suite_runner.py`
**Interfaces:**
- Consumes: the global unit-test command (PYTHONPATH, ENVIRONMENT=test, `-o "addopts=" -p no:logging -q`).
- Produces (IF-21): `scripts.ci.suites.Suite(path: str, tier: str = "unit")` and `SUITES: tuple[Suite, ...]` listing `tests/test_syntax_and_imports.py`, `tests/alignment`, `tests/solace_common`, `tests/solace_events`, `tests/solace_security`, `tests/solace_infrastructure`, `tests/solace_testing`, `tests/integration`, `tests/security`, `tests/migrations`, `tests/scripts`, `tests/system`, `tests/config_service`, `tests/bootstrap`, `tests/build`, `tests/deploy`, `tests/contracts`, `tests/clinical`, `infrastructure/api_gateway/tests` and the nine `services/<svc>/tests`; `REALINFRA_PATHS = ("tests/realinfra", "tests/migrations", "tests/build")`; `run_suites.py --tier {unit,realinfra,e2e,clinical,contract} [--suite PATH ...] [--path DIR ...] [--profile dev|strict] [--require-realinfra] [--strict-paths] [--cwd DIR] [--list-json] [--dry-run]` with `build_command(path: str, tier: str) -> list[str]` and `child_env(tier: str, *, profile: str | None, require_realinfra: bool) -> dict[str, str]`; exit code 0 only when every suite passed (pytest exit code 5, "no tests collected", counts as passed; a missing path is reported as `MISSING` and fails only with `--strict-paths`); the summary prints one line per suite, `PASS <path>`, `FAIL <path>` or `MISSING <path>`, with the path exactly as given. `ci_local.py` runs the stages lint, unit, contract, clinical, images, realinfra, e2e dev and e2e strict in order, continues after a failure and exits non-zero with a summary table.
- Produces for requesting tasks: markers `realinfra`, `e2e`, `contract`, `clinical` and `profile` plus the Makefile targets `test-realinfra` and `test-e2e PROFILE=dev|strict` (WS-05 request #1, for WS-05.2 and WS-05.4; the deselection logic of `profile` stays in WS-05's `tests/e2e/conftest.py`); the `clinical` marker for WS-12.3 (R-12-4); the `realinfra` marker for WS-09.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_244_suite_runner.py`
```python
"""MVP-244/245: one pytest process per suite, registered markers, and Makefile targets that work."""
from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "ci" / "run_suites.py"
REQUIRED_SUITES = {"tests/solace_events", "tests/solace_infrastructure", "tests/security", "tests/migrations",
                   "tests/config_service", "tests/solace_testing", "tests/system", "tests/scripts",
                   "tests/integration", "tests/build", "tests/test_syntax_and_imports.py",
                   "infrastructure/api_gateway/tests", "services/user-service/tests",
                   "services/notification-service/tests", "services/analytics-service/tests"}


def test_mvp_245_markers_registered_in_pytest_ini() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "pytest.ini", encoding="utf-8")
    names = {re.split(r"[:(]", line)[0].strip() for line in cfg["pytest"]["markers"].splitlines() if line.strip()}
    assert {"realinfra", "e2e", "contract", "clinical", "profile"} <= names


def test_mvp_245_pyproject_has_no_competing_pytest_table() -> None:
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {})
    assert "pytest" not in tool


def test_mvp_244_makefile_exposes_if21_targets_without_timeout_flag() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("lint", "test-unit", "test-realinfra", "test-e2e", "test-clinical", "test-contract",
                   "images", "clean-checkout", "ci-local", "test"):
        assert re.search(rf"^{re.escape(target)}:", text, re.M), target
    assert "--timeout=60" not in text
    assert "tests/ services/" not in text


def test_mvp_244_runner_lists_every_suite() -> None:
    r = subprocess.run([sys.executable, str(RUNNER), "--list-json"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert REQUIRED_SUITES <= set(json.loads(r.stdout))


def test_mvp_244_unit_command_matches_global_constraint() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.ci.run_suites import build_command, child_env

    assert build_command("tests/solace_events", "unit") == [
        sys.executable, "-m", "pytest", "tests/solace_events", "-o", "addopts=", "-p", "no:logging", "-q",
        "-m", "not realinfra and not e2e"]
    env = child_env("unit", profile=None, require_realinfra=False)
    assert env["ENVIRONMENT"] == "test"
    assert env["PYTHONPATH"] == f"{ROOT}{os.pathsep}{ROOT / 'src'}"


def test_mvp_244_each_suite_runs_in_its_own_process(tmp_path: Path) -> None:
    for name, ok in (("a", True), ("b", False)):
        d = tmp_path / name
        d.mkdir()
        (d / f"test_{name}.py").write_text(
            "import os\n"
            f"def test_{name}():\n"
            f"    open(r'{tmp_path / (name + '.pid')}', 'w').write(str(os.getpid()))\n"
            "    assert os.environ['ENVIRONMENT'] == 'test'\n"
            f"    assert {ok}\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(RUNNER), "--tier", "unit", "--path", str(tmp_path / "a"),
                        "--path", str(tmp_path / "b")], capture_output=True, text=True, timeout=300)
    assert r.returncode == 1
    lines = [l.rstrip() for l in r.stdout.splitlines()]
    assert any(l.startswith("PASS") and l.endswith(str(tmp_path / "a")) for l in lines), r.stdout
    assert any(l.startswith("FAIL") and l.endswith(str(tmp_path / "b")) for l in lines), r.stdout
    assert (tmp_path / "a.pid").read_text() != (tmp_path / "b.pid").read_text()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_244_suite_runner.py -o "addopts=" -p no:logging -q`
Expected: FAIL in all 6 tests: `realinfra`, `e2e`, `clinical` and `profile` are not in pytest.ini's markers (lines 35 to 54 list only `contract` of the five); `[tool.pytest.ini_options]` is present in pyproject.toml (line 143); the Makefile has no `test-unit` target and still contains `--timeout=60` and `tests/ services/` (line 10); `scripts/ci/run_suites.py` does not exist.
- [ ] **Step 3: Implement** `scripts/ci/suites.py`, `scripts/ci/run_suites.py` (one `subprocess.run` per suite with `child_env`; the tier adds the marker filter `-m "not realinfra and not e2e"` for unit, `-m realinfra` over `REALINFRA_PATHS`, `tests/e2e -m e2e` with `SOLACE_E2E_PROFILE=<profile>`, `tests/clinical -m clinical`, `tests/contracts -m contract`; `--require-realinfra` sets `SOLACE_REQUIRE_REALINFRA=1`; `--cwd` runs every child from that directory with absolute suite paths, used to detect suites that silently read the repo's `.env`), `scripts/ci/ci_local.py`, the pytest.ini markers, the pyproject table removal and the Makefile. `make migrate` calls `$(PYTHON) -m scripts.db.migrate upgrade head` (WS-02.4). Keep `make lint` read-only: `$(PYTHON) -m ruff check . --select E9,F63,F7,F82 --no-fix --no-cache`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and a dry run of the full unit tier. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q` then `make -n test-unit PYTHON="D:\tmp\mvp-exec\venv\Scripts\python.exe"` and `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\run_suites.py --tier unit --dry-run`. Expected: all tests pass; the dry run prints one command per suite and `MISSING` only for `tests/bootstrap`, `tests/deploy`, `tests/contracts` and `tests/clinical`, which later workstreams create.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.5: Image layout: repo-root template, lock sync without fallback, the right COPY set, module paths and pinned bases
**Covers:** MVP-032, MVP-013, MVP-006, MVP-248, MVP-355, MVP-112
**Severity:** blocker
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.2, WS-01.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-19 (default assumed: config-service is out of the MVP image set; its Dockerfile is WS-03's frozen path and is not edited)
**Real infra:** no
**Files:**
- Create: `scripts/ci/images.py`, `scripts/ci/dockerfiles.py`, `scripts/ci/build_images.py`
- Modify: `services/orchestrator_service/Dockerfile`, `services/safety_service/Dockerfile`, `services/therapy_service/Dockerfile`, `services/diagnosis_service/Dockerfile`, `services/memory_service/Dockerfile`, `services/personality_service/Dockerfile`, `services/user-service/Dockerfile`, `services/notification-service/Dockerfile`, `services/analytics-service/Dockerfile` (the template below)
- Modify: `.dockerignore` (append the exclusion patterns listed in Step 3; keep every existing line)
- Test: `tests/build/test_mvp_032_dockerfiles.py`, `tests/build/test_mvp_013_image_layout.py`
**Interfaces:**
- Consumes: `requirements.lock` (WS-01.2).
- Produces for requesting tasks: `.dockerignore` keeps the generated secret directories out of every build context (WS-03's request, for WS-03.7); the user-service image also carries `/app/scripts/` (`COPY --chown=appuser:appuser scripts/ ./scripts/`), so `python -m scripts.provision_clinician` runs in the container once WS-14 adds the file in W1 (R-14-1, for WS-14.9); no ignore pattern matches package data under `src/`, so `src/solace_clinical/crisis_lexicon_v1.yaml` ships in every image through `COPY src/` (R-12-4, for WS-12.3); the merged `src.main` layout for notification-service (WS-13.1); the compose image references for WS-03.2 (WS-03's request: `image_ref` and `COMPOSE_TAG` below; WS-03.2 depends on this task and asserts every built compose service's `image:` value with them, so `docker compose build` and `build_images.py` tag the same references).
- Produces (IF-21): `scripts.ci.images.Image(name: str, dockerfile: str, module: str, port: int | None, probes: tuple[str, ...] = (), merged_src: bool = False)`, `IMAGES` (orchestrator-service `services.orchestrator_service.src.main` 8000 probes `infrastructure.api_gateway.rate_limiting`; safety-service 8002; therapy-service 8006; diagnosis-service 8004; memory-service 8005; personality-service 8007; user-service `src.main` 8001 merged, probes `infrastructure.api_gateway.rate_limiting`; notification-service `src.main` 8003 merged, probes `src.api` and `src.domain.channels`; analytics-service `main` 8009), and `probe_env() -> dict[str, str]` (fresh random throwaway values per call for the variables a module import requires; never literals); `scripts.ci.dockerfiles.parse(path) -> Dockerfile` with `.from_images`, `.copies`, `.runtime_copies`, `.env`, `.cmd`, `.entrypoint`, `.healthcheck`, `.user`, `.workdir`, plus `uvicorn_flags(df) -> list[str]`, `replicate_runtime_layout(df, repo_root, app_dir, *, dockerignore=None) -> Path`, `pythonpath_for(df, app_dir) -> str` and `dockerignored(rel_path: str, dockerignore: Path) -> bool` (the matcher `replicate_runtime_layout` uses; Docker semantics: a path is ignored when it or any parent directory matches, the last matching pattern wins, `!` re-includes); `scripts/ci/build_images.py --image NAME | --all --tag TAG [--context DIR] [--probe] [--list-json]` with `build_command(image, tag, context) -> list[str]` returning `["docker", "build", "-f", image.dockerfile, "-t", image_ref(image.name, tag), context]`; `image_ref(name: str, tag: str) -> str` returning `f"solace/{name}:{tag}"` (the only place the reference format lives) and `COMPOSE_TAG = "${SOLACE_IMAGE_TAG:-dev}"`, so `image_ref(<compose service>, COMPOSE_TAG)` is exactly the `image:` string of each built compose service (WS-03.2).

Template for the eight non-analytics images (merged images replace the last `COPY` with `COPY --chown=appuser:appuser services/<svc>/src/ ./src/` and run `src.main:app`; user-service additionally has `COPY --chown=appuser:appuser scripts/ ./scripts/` right after it; analytics copies `services/analytics-service/`, appends `:/app/services/analytics-service/src` to `PYTHONPATH` and runs `main:app`). Resolve the two digests once with `docker buildx imagetools inspect python:3.12-slim` and `docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.8.2` and use the same values in every file.
```dockerfile
# syntax=docker/dockerfile:1.7
# Build context: repository root.  docker build -f services/safety_service/Dockerfile .
FROM python:3.12-slim@sha256:<python digest> AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.2@sha256:<uv digest> /uv /usr/local/bin/uv
WORKDIR /build
COPY requirements.lock ./requirements.lock
RUN uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes requirements.lock

FROM python:3.12-slim@sha256:<python digest> AS runtime
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser \
 && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser services/__init__.py ./services/__init__.py
COPY --chown=appuser:appuser services/shared/ ./services/shared/
COPY --chown=appuser:appuser infrastructure/__init__.py ./infrastructure/__init__.py
COPY --chown=appuser:appuser infrastructure/api_gateway/ ./infrastructure/api_gateway/
COPY --chown=appuser:appuser services/safety_service/ ./services/safety_service/
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app:/app/src \
    SERVICE_NAME=safety-service SERVICE_PORT=8002
USER appuser
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "services.safety_service.src.main:app", "--host", "0.0.0.0", "--port", "8002"]
```
curl stays in the runtime stage only because the compose healthchecks call it until WS-03 moves them to `/live`. WS-01.6 adds the ENTRYPOINT, the uvicorn flags and the image HEALTHCHECK.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_032_dockerfiles.py` and `tests/build/test_mvp_013_image_layout.py`
```python
# tests/build/test_mvp_032_dockerfiles.py
"""MVP-032/013/006/248/355: every MVP Dockerfile follows the IF-21 layout."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES

ROOT = Path(__file__).resolve().parents[2]
PY_BASE = re.compile(r"^python:3\.12-slim@sha256:[0-9a-f]{64}$")
APP_IMAGES = [i for i in IMAGES if i.module]


def _df(image):
    return dockerfiles.parse(ROOT / image.dockerfile)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_355_base_images_pinned_by_digest(image) -> None:
    refs = _df(image).from_images
    assert refs and all(PY_BASE.match(r) for r in refs), refs
    assert len(set(refs)) == 1
    assert re.search(r"COPY --from=ghcr\.io/astral-sh/uv:[0-9.]+@sha256:[0-9a-f]{64} ",
                     (ROOT / image.dockerfile).read_text(encoding="utf-8"))


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_248_lock_synced_with_hashes_and_no_fallback(image) -> None:
    text = (ROOT / image.dockerfile).read_text(encoding="utf-8")
    assert "requirements-runtime.txt" not in text and "service-requirements.txt" not in text
    assert "COPY requirements.lock ./requirements.lock" in text
    assert re.search(r"uv pip sync [^\n]*--require-hashes requirements\.lock", text)
    assert "||" not in text
    assert not re.search(r"\bpip install\b", text)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_032_every_copy_source_exists(image) -> None:
    for instr in _df(image).copies:
        if "from" in instr.flags:
            continue
        for src in instr.sources:
            assert (ROOT / src).exists(), f"{image.dockerfile}: COPY {src}"


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_013_images_carry_the_rate_limiter_package(image) -> None:
    sources = {s.rstrip("/") for c in _df(image).runtime_copies for s in c.sources}
    assert {"infrastructure/__init__.py", "infrastructure/api_gateway"} <= sources


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_006_cmd_runs_the_registered_module(image) -> None:
    df = _df(image)
    assert df.cmd[:3] == ["python", "-m", "uvicorn"] and f"{image.module}:app" in df.cmd
    if image.merged_src:
        svc_src = Path(image.dockerfile).parent.as_posix() + "/src/"
        assert any(c.sources == (svc_src,) and c.dest == "./src/" for c in df.runtime_copies)


@pytest.mark.parametrize("image", IMAGES, ids=lambda i: i.name)
def test_mvp_032_runtime_runs_as_uid_1000_from_app(image) -> None:
    df = _df(image)
    assert df.user == "appuser" and df.workdir == "/app"
    assert df.env["PYTHONPATH"].split(":")[:2] == ["/app", "/app/src"]
    assert "--uid 1000" in (ROOT / image.dockerfile).read_text(encoding="utf-8")


def test_mvp_032_dockerignore_keeps_secrets_tests_and_caches_out() -> None:
    lines = {l.strip() for l in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    assert {".env", ".env.*", "!.env.example", ".git", "**/tests/", "tests/", "src/models/", ".secrets", "secrets"} <= lines


def test_mvp_032_dockerignore_drops_secrets_but_keeps_package_data() -> None:
    ignore = ROOT / ".dockerignore"
    for rel in (".secrets/AUTH_SECRET_KEY", "secrets/postgres-ca.crt", ".env", "services/safety_service/tests/test_x.py"):
        assert dockerfiles.dockerignored(rel, ignore), rel
    for rel in ("src/solace_clinical/crisis_lexicon_v1.yaml", "src/solace_events/schemas.py", ".env.example",
                "scripts/db/migrate.py"):
        assert not dockerfiles.dockerignored(rel, ignore), rel


def test_mvp_032_user_service_image_carries_scripts() -> None:
    df = _df(next(i for i in IMAGES if i.name == "user-service"))
    assert any(c.sources == ("scripts/",) and c.dest == "./scripts/" for c in df.runtime_copies)


def test_mvp_112_build_command_uses_repo_root_context() -> None:
    from scripts.ci.build_images import build_command

    image = next(i for i in IMAGES if i.name == "user-service")
    assert build_command(image, "ci", ".") == [
        "docker", "build", "-f", "services/user-service/Dockerfile", "-t", "solace/user-service:ci", "."]


def test_mvp_112_compose_image_refs_come_from_the_builder() -> None:
    # WS-03.2 writes image_ref(<service>, COMPOSE_TAG) as each built compose service's image: value
    from scripts.ci.build_images import COMPOSE_TAG, build_command, image_ref

    image = next(i for i in IMAGES if i.name == "user-service")
    assert COMPOSE_TAG == "${SOLACE_IMAGE_TAG:-dev}"
    assert image_ref("user-service", COMPOSE_TAG) == "solace/user-service:${SOLACE_IMAGE_TAG:-dev}"
    assert build_command(image, "ci", ".")[5] == image_ref(image.name, "ci")
```
```python
# tests/build/test_mvp_013_image_layout.py
"""MVP-013/006/032: the runtime layout each Dockerfile builds imports its CMD target (replica, no Docker)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES, probe_env

ROOT = Path(__file__).resolve().parents[2]
APP_IMAGES = [i for i in IMAGES if i.module]


def _replica(image, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    app_dir = dockerfiles.replicate_runtime_layout(df, ROOT, tmp_path / "app", dockerignore=ROOT / ".dockerignore")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(probe_env())
    env.update({"PYTHONPATH": dockerfiles.pythonpath_for(df, app_dir), "ENVIRONMENT": "test"})
    return app_dir, env


def _import(module: str, app_dir: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    code = ("import importlib, json; m = importlib.import_module(%r); "
            "print(json.dumps({'file': getattr(m, '__file__', ''), 'pkg': m.__package__, "
            "'app': type(getattr(m, 'app', None)).__name__}))") % module
    return subprocess.run([sys.executable, "-c", code], cwd=app_dir, env=env, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_032_replica_imports_cmd_target_from_image_layout(image, tmp_path: Path) -> None:
    app_dir, env = _replica(image, tmp_path)
    r = _import(image.module, app_dir, env)
    assert r.returncode == 0, r.stderr[-3000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert Path(info["file"]).resolve().is_relative_to(app_dir.resolve())
    assert info["app"] == "FastAPI"
    if image.merged_src:
        assert info["pkg"] == "src"  # MVP-006: the lifespan's package-relative imports resolve


@pytest.mark.parametrize("image", [i for i in APP_IMAGES if i.probes], ids=lambda i: i.name)
def test_mvp_013_replica_imports_probe_modules(image, tmp_path: Path) -> None:
    app_dir, env = _replica(image, tmp_path)
    for module in image.probes:
        r = _import(module, app_dir, env)
        assert r.returncode == 0, f"{image.name}: {module}: {r.stderr[-2000:]}"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_032_dockerfiles.py D:\Repo\Contextual-Chatbot\tests\build\test_mvp_013_image_layout.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Before the registry exists, collection fails with `ModuleNotFoundError: scripts.ci.images`; write `images.py`, `dockerfiles.py` and `build_images.py` first, keep the Dockerfiles unchanged, and re-run. The second RED run shows: `test_mvp_355` fails on `python:3.12-slim` without a digest in all nine; `test_mvp_248` fails on `requirements-runtime.txt` and the `|| pip install` fallback; `test_mvp_032_every_copy_source_exists` fails on `COPY requirements-runtime.txt`; `test_mvp_013_images_carry_the_rate_limiter_package` fails in all nine; `test_mvp_006` fails for notification-service (CMD `main:app`); `test_mvp_032_dockerignore_keeps_secrets_tests_and_caches_out` fails on `**/tests/` and `.secrets`; `test_mvp_032_dockerignore_drops_secrets_but_keeps_package_data` fails on `.secrets/AUTH_SECRET_KEY` (today's `.dockerignore` has no secrets-directory pattern); `test_mvp_032_user_service_image_carries_scripts` fails (no `COPY scripts/` in `services/user-service/Dockerfile`); the replica probe for orchestrator-service fails with `ModuleNotFoundError: No module named 'infrastructure'` and for user-service with `No module named 'infrastructure.api_gateway'` (its own `src/infrastructure` shadows the name); the notification replica fails on `'' == 'src'`.
- [ ] **Step 3: Implement** the nine Dockerfiles from the template and the `.dockerignore` additions: `**/tests/`, `tests/`, `src/models/`, `docs/`, `infrastructure/aws-staging/`, `**/logs/`, `.secrets` and `secrets` (one per line, appended after the existing lines; no pattern may match `*.yaml` or anything else under `src/` except `src/models/`). `replicate_runtime_layout` copies each runtime-stage `COPY` source to `app_dir` joined with its destination resolved against `WORKDIR`, skips `--from` copies, and applies `.dockerignore` patterns (fnmatch semantics with `**`), so the replica matches what the build context ships.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass. A real build of each image belongs to WS-01.9; to spot-check one image now, run `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\build_images.py --image safety-service --tag ws01-5 --probe` and record the result (not gating here).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.6: Container entry contract: NAME_FILE loader, uvicorn flags that never log query tokens, `/live` healthcheck
**Covers:** MVP-211
**Severity:** medium
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-27 (default assumed: `?token=` stays accepted for WebSocket compatibility, so the process must never write query strings to logs). D-07 (decided 2026-10-19: SOPS + age instead of AWS Secrets Manager. WS-03's `deploy/render_secrets.py` decrypts the per-environment `deploy/secrets/<env>.sops.yaml` with pinned `sops` and `age` binaries into root-only tmpfs `/run/secrets`, one file per secret, mode 0400. Strict secrets therefore still arrive as NAME_FILE pointing into `/run/secrets`, and the consumer contract this entrypoint implements is unchanged. The SOPS-aware ignore rules, build-context exclusions and secret scan are WS-01.14)
**Real infra:** yes
**Files:**
- Create: `docker/entrypoint.sh`
- Create: `tests/build/ws_probe_app.py` (an ASGI app with `GET /health`, `POST /auth/verify-email` and a WebSocket route `/ws/{session_id}`; not collected as a test)
- Modify: the nine MVP Dockerfiles (builder stage: `COPY docker/entrypoint.sh /build/entrypoint.sh` then `RUN sed -i 's/\r$//' /build/entrypoint.sh && chmod 0755 /build/entrypoint.sh`; runtime stage: `COPY --from=builder --chown=appuser:appuser /build/entrypoint.sh /app/docker/entrypoint.sh`, `HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS "http://127.0.0.1:${SERVICE_PORT}/live" || exit 1`, `ENTRYPOINT ["/app/docker/entrypoint.sh"]`; CMD gains `"--no-access-log", "--log-level", "warning", "--proxy-headers"` after `--port <port>`; the orchestrator CMD also gains `"--ws-max-size", "16384"`)
- Test: `tests/build/test_mvp_211_container_entry.py`
**Interfaces:**
- Consumes: IF-22 (NAME_FILE semantics), `scripts.ci.dockerfiles.parse` and `uvicorn_flags` (WS-01.5), fixture `require_docker` (WS-01.1).
- Produces (IF-21): `docker/entrypoint.sh`: for every environment variable `NAME_FILE` it exports `NAME` with the file's content minus exactly one trailing newline; a missing or unreadable file exits 1 naming the variable, never a value; `NAME` and `NAME_FILE` both set exits 1; `NAME_FILE` itself stays set in the exec'd environment (WS-04.11 checks provenance); `SSL_CERT_FILE` is left alone; no `set -x`, no echo of values; finally `exec "$@"`. Uvicorn runs with `--no-access-log --log-level warning --proxy-headers` (the `--log-level warning` flag is an addition to IF-21, see New interfaces); the orchestrator adds `--ws-max-size 16384`, so frames far above the 8 KB app limit are refused by uvicorn before they reach the app.
- Produces for requesting tasks: the entrypoint contract for WS-03.2 and WS-03.7 (WS-03's row: one trailing newline stripped, missing file fatal, `NAME_FILE` left set) and for WS-04.11; `--ws-max-size 16384` for WS-11.17 (its `WS-01:ws-max-size` dependency is this task; it lands in W0, earlier than the W3 WS-11 asked for, which is harmless because uvicorn's default is 16 MiB and the app limit stays 8 KB).

The uvicorn WebSocket accept line `'%s - "WebSocket %s" [accepted]'` is written by the `uvicorn.error` logger at INFO (`uvicorn/protocols/websockets/websockets_impl.py:271`), not the access logger, so `--no-access-log` alone still logs `?token=<JWT>`; `--log-level warning` drops it.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_211_container_entry.py`
```python
"""MVP-211: bearer tokens in query strings never reach container logs; the IF-22 entrypoint loads NAME_FILE secrets."""
from __future__ import annotations

import asyncio
import hashlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import websockets

from scripts.ci import dockerfiles
from scripts.ci.images import IMAGES

ROOT = Path(__file__).resolve().parents[2]
APP_IMAGES = [i for i in IMAGES if i.module]
WS_TOKEN, VERIFY_TOKEN = "SYNTHETIC-WS-TOKEN-7f3a", "SYNTHETIC-VERIFY-TOKEN-91c2"


@pytest.mark.parametrize("image", APP_IMAGES, ids=lambda i: i.name)
def test_mvp_211_cmd_flags_and_entry(image) -> None:
    df = dockerfiles.parse(ROOT / image.dockerfile)
    assert "--no-access-log" in df.cmd and "--proxy-headers" in df.cmd
    assert df.cmd[df.cmd.index("--log-level") + 1] == "warning"
    assert df.entrypoint == ["/app/docker/entrypoint.sh"]
    assert df.healthcheck and "/live" in df.healthcheck and "/health" not in df.healthcheck
    if image.name == "orchestrator-service":
        assert df.cmd[df.cmd.index("--ws-max-size") + 1] == "16384"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mvp_211_image_uvicorn_flags_never_log_query_tokens() -> None:
    flags = dockerfiles.uvicorn_flags(dockerfiles.parse(ROOT / "services/orchestrator_service/Dockerfile"))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ws_probe_app:app", "--app-dir", str(Path(__file__).parent),
         "--host", "127.0.0.1", "--port", str(port), *flags],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)

        async def _ws() -> None:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/session-1?token={WS_TOKEN}") as ws:
                await ws.send("ping")
                await ws.recv()

        asyncio.run(_ws())
        httpx.post(f"http://127.0.0.1:{port}/auth/verify-email?token={VERIFY_TOKEN}&user_id=u-1", timeout=5)
    finally:
        proc.terminate()
        out = proc.communicate(timeout=15)[0]
    assert WS_TOKEN not in out and VERIFY_TOKEN not in out


def _run_entrypoint(tmp_path: Path, env: dict[str, str], command: list[str]):
    import docker

    image = dockerfiles.parse(ROOT / "services/orchestrator_service/Dockerfile").from_images[0]
    client = docker.from_env()
    container = client.containers.run(
        image, ["sh", "/entry/entrypoint.sh", *command], detach=True, environment=env,
        volumes={str(ROOT / "docker"): {"bind": "/entry", "mode": "ro"},
                 str(tmp_path): {"bind": "/run/secrets", "mode": "ro"}})
    try:
        status = container.wait(timeout=60)["StatusCode"]
        return status, container.logs().decode("utf-8", "replace")
    finally:
        container.remove(force=True)


PRINT_SHA = ["python", "-c", "import hashlib, os; print(hashlib.sha256(os.environ['AUTH_SECRET_KEY'].encode()).hexdigest()); "
             "print('provenance=' + os.environ.get('AUTH_SECRET_KEY_FILE', 'unset'))"]


@pytest.mark.realinfra
@pytest.mark.parametrize("content,expected", [("v1\n", "v1"), ("v1\n\n", "v1\n"),
                                              ("-----BEGIN K-----\nabc\n-----END K-----\n", "-----BEGIN K-----\nabc\n-----END K-----")])
def test_mvp_211_entrypoint_exports_name_file(require_docker, tmp_path: Path, content: str, expected: str) -> None:
    (tmp_path / "auth_secret").write_bytes(content.encode())
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/run/secrets/auth_secret"}, PRINT_SHA)
    assert status == 0, logs
    assert hashlib.sha256(expected.encode()).hexdigest() in logs
    assert "provenance=/run/secrets/auth_secret" in logs  # NAME_FILE stays set for WS-04.11


@pytest.mark.realinfra
def test_mvp_211_entrypoint_fails_closed_without_echoing(require_docker, tmp_path: Path) -> None:
    (tmp_path / "s").write_text("SYNTHETIC-MARKER-5512\n")
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY_FILE": "/run/secrets/missing"}, ["true"])
    assert status == 1 and "AUTH_SECRET_KEY_FILE" in logs
    status, logs = _run_entrypoint(tmp_path, {"AUTH_SECRET_KEY": "x", "AUTH_SECRET_KEY_FILE": "/run/secrets/s"}, ["true"])
    assert status == 1 and "SYNTHETIC-MARKER-5512" not in logs
    status, logs = _run_entrypoint(tmp_path, {"OTHER_SECRET_FILE": "/run/secrets/s",
                                              "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt"},
                                   ["sh", "-c", "test -z \"${SSL_CERT:-}\" && cat /proc/1/cmdline"])
    assert status == 0 and "SYNTHETIC-MARKER-5512" not in logs
    assert "entrypoint.sh" not in logs  # PID 1 is the exec'd command, not the loader shell
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_211_container_entry.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_211_cmd_flags_and_entry` fails in all nine images (no `--no-access-log`, no ENTRYPOINT, HEALTHCHECK still `/health`; the orchestrator CMD also has no `--ws-max-size`); `test_mvp_211_image_uvicorn_flags_never_log_query_tokens` fails because the captured output contains `"WebSocket /ws/session-1?token=SYNTHETIC-WS-TOKEN-7f3a" [accepted]` and the access line with `SYNTHETIC-VERIFY-TOKEN-91c2`; the entrypoint tests fail with `sh: 0: cannot open /entry/entrypoint.sh` (exit 2).
- [ ] **Step 3: Implement** `docker/entrypoint.sh` (POSIX sh, LF line endings; strip exactly one trailing newline with the `x`-sentinel idiom `value=$(cat "$f"; printf x); value=${value%x}; value=${value%"$nl"}`), `tests/build/ws_probe_app.py` and the Dockerfile edits listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.7: One import root for shared packages, and an ownership map for the import and lint guards
**Covers:** MVP-270
**Severity:** low
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.1
**Implementer model:** opus
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/__init__.py` (docstring only; drop `from . import solace_common` and `from . import solace_events`, which load a second copy of both packages whenever anything imports `src`)
- Create: `tests/build/test_import_roots.py` (the owner-parametrized import guard, including the strict-xfail WS-10 case that allow-lists the orchestrator's events module until WS-10.3 lands in W1, WS-10's request), `tests/build/test_mvp_038_lint_gate.py`
- Test: `tests/build/test_import_roots.py`, `tests/build/test_mvp_038_lint_gate.py`
**Interfaces:**
- Consumes: none.
- Produces: owner-parametrized guards. `test_import_roots.py::test_mvp_270_no_src_root_imports[<WS>]` for WS-01 (`src/__init__.py`), WS-02 (`migrations/env.py`), WS-05 (`tests/integration/test_persistence_roundtrip.py`), WS-06 (the seven `services/*/src/infrastructure/event_bridge.py`), WS-10 (`services/orchestrator_service/src/events.py`), WS-12, WS-13, WS-14, WS-15, WS-16, WS-17 (their `events.py`) and WS-18 (`services/personality_service/src/events.py`, `services/analytics-service/src/main.py`); `::test_mvp_270_no_src_root_imports_outside_registered_files`. `test_mvp_038_lint_gate.py::test_mvp_038_lint_gate_clean[<WS>]` for WS-02 (`src/solace_infrastructure/postgres.py`), WS-13 (`services/notification-service/src/main.py`) and WS-18 (`services/analytics-service/src/main.py`, `services/analytics-service/src/config.py`); `::test_mvp_038_lint_gate_clean_outside_registered_files`. Each owner runs its own case RED then GREEN with `-k "<WS>"`. The WS-10 import case is `xfail(strict=True)` because WS-10 has no W0 wave; WS-10.3 fixes the file in W1 phase B, the case then XPASSes and fails strictly, and the controller deletes the marker by ledger Ruling.
- Produces for requesting tasks: the W0-barrier allow-list of `services/orchestrator_service/src/events.py` as that strict-xfail case (WS-10's request, for WS-10.3); the owner cases run by WS-02.2, WS-02.5, WS-06.2, WS-13.1, WS-14.2, WS-16.1, WS-18.1 and WS-18.2.

- [ ] **Step 1: Write the failing test** in `tests/build/test_import_roots.py` and `tests/build/test_mvp_038_lint_gate.py`
```python
# tests/build/test_import_roots.py
"""MVP-270: shared packages are imported only as top-level solace_* (never src.solace_*)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN = ("src", "services", "migrations", "scripts", "infrastructure", "tests")
SKIP = ("services/config_service/", "tests/build/test_import_roots.py")
BRIDGES = tuple(f"services/{s}/src/infrastructure/event_bridge.py" for s in (
    "orchestrator_service", "safety_service", "therapy_service", "diagnosis_service",
    "personality_service", "user-service", "notification-service"))
OWNERS: dict[str, tuple[str, ...]] = {
    "WS-01": ("src/__init__.py",),
    "WS-02": ("migrations/env.py",),
    "WS-05": ("tests/integration/test_persistence_roundtrip.py",),
    "WS-06": BRIDGES,
    "WS-10": ("services/orchestrator_service/src/events.py",),
    "WS-12": ("services/safety_service/src/events.py",),
    "WS-13": ("services/notification-service/src/events.py",),
    "WS-14": ("services/user-service/src/events.py",),
    "WS-15": ("services/therapy_service/src/events.py",),
    "WS-16": ("services/diagnosis_service/src/events.py",),
    "WS-17": ("services/memory_service/src/events.py",),
    "WS-18": ("services/personality_service/src/events.py", "services/analytics-service/src/main.py"),
}
PATCH_CALLS = {"import_module", "patch", "object"}


def _scan_source(source: str, rel: str = "x.py") -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "src":
                names = [f"src.{a.name}" for a in node.names]
            elif node.level == 0 and node.module:
                names = [node.module]
            elif node.level == 1 and rel == "src/__init__.py":
                names = [f"src.{node.module or a.name}" for a in node.names]
        elif isinstance(node, ast.Call):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func in PATCH_CALLS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                names = [node.args[0].value]
        hits += [f"{getattr(node, 'lineno', 0)}:{n}" for n in names if n.startswith("src.solace_")]
    return hits


def _violations() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for base in SCAN:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(SKIP) or "/__pycache__/" in rel:
                continue
            hits = _scan_source(path.read_text(encoding="utf-8"), rel)
            if hits:
                out[rel] = hits
    return out


def test_mvp_270_scanner_detects_every_form() -> None:
    for src in ("import src.solace_events", "from src.solace_events.config import X",
                "from src import solace_common", "importlib.import_module('src.solace_security.auth')",
                "patch('src.solace_events.publisher.EventPublisher')"):
        assert _scan_source(src), src
    assert _scan_source("from . import solace_common", "src/__init__.py")
    assert not _scan_source("from solace_events.config import X")


@pytest.mark.parametrize("owner", [
    pytest.param(o, marks=pytest.mark.xfail(strict=True, reason="WS-10.3 (W1 phase B) rewrites the import; marker removed by Ruling"))
    if o == "WS-10" else o for o in sorted(OWNERS)])
def test_mvp_270_no_src_root_imports(owner: str) -> None:
    found = {f: h for f, h in _violations().items() if f in OWNERS[owner]}
    assert found == {}


def test_mvp_270_no_src_root_imports_outside_registered_files() -> None:
    known = {f for files in OWNERS.values() for f in files}
    assert {f: h for f, h in _violations().items() if f not in known} == {}
```
```python
# tests/build/test_mvp_038_lint_gate.py
"""MVP-038: the CI lint gate (ruff E9,F63,F7,F82) is clean; each remaining F821 is owned by one workstream."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LINT_OWNERS = {
    "WS-02": ("src/solace_infrastructure/postgres.py",),
    "WS-13": ("services/notification-service/src/main.py",),
    "WS-18": ("services/analytics-service/src/main.py", "services/analytics-service/src/config.py"),
}


def _findings() -> dict[str, list[str]]:
    r = subprocess.run([sys.executable, "-m", "ruff", "check", str(ROOT), "--select", "E9,F63,F7,F82",
                        "--no-fix", "--no-cache", "--output-format", "json"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 1), r.stderr
    out: dict[str, list[str]] = {}
    for f in json.loads(r.stdout or "[]"):
        rel = Path(f["filename"]).resolve().relative_to(ROOT).as_posix()
        out.setdefault(rel, []).append(f"{f['location']['row']}:{f['code']}")
    return out


@pytest.mark.parametrize("owner", sorted(LINT_OWNERS))
def test_mvp_038_lint_gate_clean(owner: str) -> None:
    assert {f: v for f, v in _findings().items() if f in LINT_OWNERS[owner]} == {}


def test_mvp_038_lint_gate_clean_outside_registered_files() -> None:
    known = {f for files in LINT_OWNERS.values() for f in files}
    assert {f: v for f, v in _findings().items() if f not in known} == {}
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py D:\Repo\Contextual-Chatbot\tests\build\test_mvp_038_lint_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL in the 11 owner import cases other than WS-10 (WS-10 reports XFAIL), with `src/__init__.py` showing `4:src.solace_common` and `5:src.solace_events`; the lint cases WS-02 (`94:F821`), WS-13 (3 × F821) and WS-18 (7 × F821) fail. The scanner self-test and both "outside registered files" tests pass.
- [ ] **Step 3: Implement** the `src/__init__.py` change only. Other owners' files are not edited here.
- [ ] **Step 4: Run it and confirm it passes**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest "D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py::test_mvp_270_no_src_root_imports[WS-01]" D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py::test_mvp_270_scanner_detects_every_form D:\Repo\Contextual-Chatbot\tests\build\test_import_roots.py::test_mvp_270_no_src_root_imports_outside_registered_files D:\Repo\Contextual-Chatbot\tests\build\test_mvp_038_lint_gate.py::test_mvp_038_lint_gate_clean_outside_registered_files -o "addopts=" -p no:logging -q`. Expected: PASS. The other owner cases stay red by design until each owner's W0 task (ledger note: expected red at this checkpoint, green at the W0 barrier).
- [ ] **Step 5: Regression:** one process per suite for `tests/test_syntax_and_imports.py`, `tests/solace_common`, `tests/solace_events` and `services/user-service/tests` (the suites that import the `src` package), each with `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\<suite> -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.8: CI that mirrors the runner: lint gate, every suite, repo-root image builds with probes
**Covers:** MVP-038, MVP-246, MVP-112
**Severity:** high
**Wave:** 0
**Phase:** A1
**Depends on:** WS-01.4, WS-01.5, WS-01.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-17 (decided 2026-10-19: no commits at all during the program, so nothing is pushed. The workflow is proven by static checks plus `make ci-local` in this task. Its first real GitHub run can only follow the owner's post-program commit decision)
**Real infra:** yes
**Files:**
- Modify: `.github/workflows/ci.yml` (rewrite)
- Test: `tests/build/test_mvp_038_ci_workflow.py`
**Interfaces:**
- Consumes: `scripts/ci/run_suites.py --list-json` and `--tier ... --suite ...` (WS-01.4); `scripts/ci/build_images.py --list-json` and `--image NAME --tag ci --probe` (WS-01.5); `requirements-dev.lock` (WS-01.2).
- Produces: workflow jobs `plan` (outputs `suites` and `images` as JSON from the two `--list-json` calls), `lint`, `unit` (matrix `suite: ${{ fromJSON(needs.plan.outputs.suites) }}`, `needs: [plan]`), `realinfra` (`--tier realinfra --require-realinfra`), `images` (matrix over `needs.plan.outputs.images`), `e2e` (`if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'`, profiles dev and strict). Every install step is `uv pip sync --system --require-hashes requirements-dev.lock` after installing uv 0.8.2. Test jobs do not depend on `lint`. `permissions: contents: read`. Throwaway test values come from a generation step (`python -c` with `secrets.token_urlsafe`) written to `$GITHUB_ENV`, never literals. The on-demand/nightly `e2e` job is WS-05 request #1's CI part (for WS-05.4).

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_038_ci_workflow.py`
```python
"""MVP-038/112/246: CI runs every suite from the same registry as make ci-local, and builds images from the repo root."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _wf() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _runs(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_mvp_038_jobs_exist_and_tests_do_not_wait_for_lint() -> None:
    jobs = _wf()["jobs"]
    assert {"plan", "lint", "unit", "realinfra", "images", "e2e"} <= set(jobs)
    for name in ("unit", "realinfra", "images"):
        needs = jobs[name].get("needs", [])
        assert "lint" not in (needs if isinstance(needs, list) else [needs]), name


def test_mvp_038_unit_matrix_comes_from_the_suite_registry() -> None:
    jobs = _wf()["jobs"]
    assert "run_suites.py --list-json" in _runs(jobs["plan"])
    assert jobs["unit"]["strategy"]["matrix"]["suite"] == "${{ fromJSON(needs.plan.outputs.suites) }}"
    assert "run_suites.py --tier unit --suite" in _runs(jobs["unit"])
    assert "--tier realinfra --require-realinfra" in _runs(jobs["realinfra"])


def test_mvp_038_installs_only_from_the_hashed_dev_lock() -> None:
    text = CI.read_text(encoding="utf-8")
    assert 'pip install -e' not in text and '.[dev]' not in text
    for name, job in _wf()["jobs"].items():
        if name != "plan":
            assert "uv pip sync --system --require-hashes requirements-dev.lock" in _runs(job), name


def test_mvp_246_no_suite_or_file_is_ignored() -> None:
    assert "--ignore" not in CI.read_text(encoding="utf-8")


def test_mvp_112_images_build_from_repo_root_with_probe() -> None:
    jobs = _wf()["jobs"]
    assert "build_images.py --list-json" in _runs(jobs["plan"])
    assert jobs["images"]["strategy"]["matrix"]["image"] == "${{ fromJSON(needs.plan.outputs.images) }}"
    assert re.search(r"build_images\.py --image \"?\$\{\{ matrix\.image \}\}\"? --tag ci --probe", _runs(jobs["images"]))
    assert "./services/" not in CI.read_text(encoding="utf-8")


def test_mvp_038_lint_gate_and_e2e_trigger() -> None:
    jobs = _wf()["jobs"]
    assert "ruff check . --select E9,F63,F7,F82 --no-fix" in _runs(jobs["lint"])
    assert "workflow_dispatch" in jobs["e2e"]["if"] and "schedule" in jobs["e2e"]["if"]
    triggers = _wf().get("on", _wf().get(True))
    assert {"push", "pull_request", "workflow_dispatch", "schedule"} <= set(triggers)
    assert "ci-test-secret" not in CI.read_text(encoding="utf-8")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_038_ci_workflow.py -o "addopts=" -p no:logging -q`
Expected: FAIL in all 6 tests: there are no `plan`, `unit`, `realinfra`, `images` or `e2e` jobs (today `test`, `service-tests`, `integration-tests`, `docker-build` each `needs: lint`); installs use `pip install -e ".[dev]"`; `--ignore=tests/integration/test_service_integration.py` at line 146; the docker matrix uses `./services/<svc>` contexts (lines 156 to 165); the env block holds literal `ci-test-secret...` values.
- [ ] **Step 3: Implement** the workflow per Produces. PyYAML parses the `on:` key as boolean `True`; the test reads both.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** the shared_files protocol requires `make ci-local` for any workflow change. Run: `make ci-local PYTHON="D:\tmp\mvp-exec\venv\Scripts\python.exe"`. Expected at this point in W0: `lint` red only for the files listed in `LINT_OWNERS`; `unit` green except the owner cases of `tests/build/test_import_roots.py` and `tests/build/test_mvp_038_lint_gate.py` that belong to other workstreams; `images` builds and probes all nine app images; `contract`, `clinical`, `e2e dev` and `e2e strict` report `MISSING` until WS-05 and WS-12 create their suites; `realinfra` green. Record the per-stage table in the ledger. Then run `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\run_suites.py --tier unit --cwd D:\tmp\mvp-exec\ws01\hermetic` once and record, as W0 findings for their owners, every suite that passes from the repo root but fails from a directory without the repo's `.env` (172 BaseSettings classes declare `env_file=".env"`; CI has no `.env`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.9: Migrate image and clean-checkout proof from a git-archive export
**Covers:** MVP-008, MVP-032, MVP-012, MVP-013
**Severity:** blocker
**Wave:** 0
**Phase:** A2 (after WS-02.4)
**Depends on:** WS-01.6, WS-01.8, WS-02.3, WS-02.4
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-17 (decided 2026-10-19: no commits at all, not even at barriers. MVP-008 closes on the clean-checkout build and probe of a checkpoint tree exported with `git archive <checkpoint tree id>`, which is the only proof this program produces; barrier notes report the tree id only. The tracked-closure report never gates anything. It is input for the owner's post-program commit decision, and `make release-gate-closure` is the owner's aid for that decision, not part of any wave gate or the W5 release gate)
**Real infra:** yes
**Files:**
- Create: `docker/migrate.Dockerfile` (layout under Interfaces; it only reads WS-02's files as COPY sources and edits none of them)
- Create: `scripts/ci/clean_checkout.py`, `scripts/ci/check_clean_checkout.ps1`, `scripts/ci/check_tracked_closure.py`
- Modify: `scripts/ci/images.py` (append `Image("migrate", "docker/migrate.Dockerfile", "", None, ("scripts.db.migrate", "solace_infrastructure.database.schema_check"))`)
- Modify: `Makefile` (`clean-checkout` calls `$(PYTHON) scripts/ci/clean_checkout.py --tree $(TREE)`; new target `release-gate-closure` calls `$(PYTHON) scripts/ci/check_tracked_closure.py --strict`)
- Modify: `scripts/ci/ci_local.py` (stage `tracked-closure` running `check_tracked_closure.py --report`, non-gating)
- Test: `tests/build/test_mvp_008_clean_checkout.py`; `tests/build/test_mvp_032_dockerfiles.py` (add `test_mvp_032_migrate_image_carries_schema_tooling`)
**Interfaces:**
- Consumes: `scripts.db.migrate` CLI and `solace_infrastructure.database.schema_check` (WS-02.3, WS-02.4); the image template and entrypoint (WS-01.5, WS-01.6); for `--compose`, WS-03's compose `migrate` service and config-service removal.
- Produces: the migrate image from `docker/migrate.Dockerfile` (`IMAGES` name `migrate`, so `build_images.py` tags it `image_ref("migrate", tag)`, that is `solace/migrate:<tag>`, the reference WS-03.2's compose `migrate` service uses): the app template's builder stage; a runtime stage that copies the Alembic config, the migrations directory, the scripts/db package and the src tree (COPY sources alembic.ini, migrations/, scripts/db/ and src/, owned by other workstreams and only read here) into `/app`, with `PYTHONPATH=/app:/app/src`, ENTRYPOINT `/app/docker/entrypoint.sh`, CMD `["python", "-m", "scripts.db.migrate", "upgrade", "head"]`, user `appuser` and no HEALTHCHECK. This is R-02.1's image part (it packages WS-02.4's CLI, so it follows WS-02.4) and the image WS-03.2's compose `migrate` service and WS-05.4's test stack run.
- Produces: `clean_checkout.py --tree TREE [--out DIR] [--compose] [--tag TAG]` exports `git archive --format=tar TREE` (read-only) into `D:\tmp\mvp-exec\clean-checkout\<tree7>` (or `--out`), builds every `IMAGES` entry from that directory with `build_images.build_command`, and runs each probe inside the image (`docker run --rm --entrypoint python -e ENVIRONMENT=development <probe_env> <image> -c "import importlib; importlib.import_module('<module>')"`); `--compose` also runs `docker compose -f docker-compose.yml build` there; exit 0 only when all succeed. `check_tracked_closure.py [--repo DIR] [--report | --strict]` lists `tracked file -> untracked dependency` pairs for Python imports (resolved against the repo root, `src/` and each `services/<svc>`), Dockerfile `COPY` sources and `-r` includes in requirements files, using only `git ls-files` and `git ls-files --others --exclude-standard`; `--strict` exits 1 when any pair exists. `check_clean_checkout.ps1 -Tree <id>` calls clean_checkout.py with `D:\tmp\mvp-exec\venv\Scripts\python.exe` (D-20).

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_008_clean_checkout.py` and the new migrate case
```python
# tests/build/test_mvp_008_clean_checkout.py
"""MVP-008: a clean checkout of a tree builds and boots; tracked files never depend on untracked ones."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _closure(repo: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts/ci/check_tracked_closure.py"), "--repo", str(repo), *flags],
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "__init__.py").write_text("")
    (repo / "src" / "pkg" / "a.py").write_text("from pkg import b\n")
    (repo / "src" / "pkg" / "b.py").write_text("X = 1\n")
    (repo / "Dockerfile").write_text("FROM scratch\nCOPY extra.txt /x\n")
    (repo / "extra.txt").write_text("x\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/pkg/__init__.py", "src/pkg/a.py", "Dockerfile")  # stage only; no commit
    return repo


def test_mvp_008_closure_reports_tracked_file_importing_untracked_module(synthetic_repo: Path) -> None:
    r = _closure(synthetic_repo, "--strict")
    assert r.returncode == 1
    assert "src/pkg/a.py -> src/pkg/b.py" in r.stdout and "Dockerfile -> extra.txt" in r.stdout


def test_mvp_008_closure_is_clean_once_dependencies_are_tracked(synthetic_repo: Path) -> None:
    _git(synthetic_repo, "add", "src/pkg/b.py", "extra.txt")
    r = _closure(synthetic_repo, "--strict")
    assert r.returncode == 0, r.stdout


@pytest.mark.realinfra
def test_mvp_008_tree_builds_and_probes_from_clean_checkout(require_docker) -> None:
    tree = os.environ.get("SOLACE_CLEAN_CHECKOUT_TREE")
    if not tree:
        pytest.skip("set SOLACE_CLEAN_CHECKOUT_TREE to a checkpoint tree id")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/ci/clean_checkout.py"), "--tree", tree, "--tag", "clean"],
                       capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
```
```python
# appended to tests/build/test_mvp_032_dockerfiles.py
def test_mvp_032_migrate_image_carries_schema_tooling() -> None:
    df = dockerfiles.parse(ROOT / "docker" / "migrate.Dockerfile")
    sources = {s.rstrip("/") for c in df.runtime_copies for s in c.sources}
    assert {"alembic.ini", "migrations", "scripts/db", "src"} <= sources
    assert df.cmd == ["python", "-m", "scripts.db.migrate", "upgrade", "head"]
    assert df.entrypoint == ["/app/docker/entrypoint.sh"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_008_clean_checkout.py D:\Repo\Contextual-Chatbot\tests\build\test_mvp_032_dockerfiles.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the two closure tests fail because `scripts/ci/check_tracked_closure.py` does not exist (exit 2 from the interpreter); `test_mvp_032_migrate_image_carries_schema_tooling` fails with `FileNotFoundError` for `docker/migrate.Dockerfile`.
- [ ] **Step 3: Implement** the files listed. Then show the defect on the old tree as a negative control: `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\clean_checkout.py --tree <W0 baseline tree from D:\tmp\mvp-exec\ledger\progress.md> --tag baseline` must exit non-zero with every image failing at `COPY requirements-runtime.txt` (`not found`). Record the output summary in the report. Run `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\check_tracked_closure.py --report` on the real repo and record its pairs in the ledger, for example `services/memory_service/src/infrastructure/postgres_repo.py -> src/solace_common/request_context.py`. They stay for the whole program, because nothing is committed (D-17). The report is input for the owner's post-program commit decision and gates nothing.
- [ ] **Step 4: Run it and confirm it passes**
Run: take a head tree with `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.9:probe"`, then `$env:SOLACE_CLEAN_CHECKOUT_TREE="<printed tree id>"; $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_008_clean_checkout.py D:\Repo\Contextual-Chatbot\tests\build\test_mvp_032_dockerfiles.py -o "addopts=" -p no:logging -q`. Expected: PASS: all nine app images and the migrate image build from the archive; the notification probe imports `src.api` (MVP-012: email-validator is in the image), and the orchestrator and user-service probes import `infrastructure.api_gateway.rate_limiting` (MVP-013).
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass, except the documented owner cases of `test_import_roots.py` and `test_mvp_038_lint_gate.py` still pending in other lanes. At the W0 barrier the controller runs `clean_checkout.py --tree <barrier tree> --compose` after WS-03's compose sync.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.10: Supply-chain gate: dependency audit, image scan, SBOM and update automation
**Covers:** MVP-113, MVP-355
**Severity:** medium
**Wave:** 4
**Phase:** A
**Depends on:** WS-01.8, WS-01.9
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `scripts/ci/supply_chain.py`, `.github/dependabot.yml`
- Modify: `scripts/ci/accepted_vulns.yaml` (created by WS-01.F1 with entries `{id, target, reason, owner, expires}`; this task gates on it)
- Modify: `.github/workflows/ci.yml` (job `supply-chain`, `needs: [images]`), `Makefile` (target `supply-chain`), `scripts/ci/ci_local.py` (stage `supply-chain`, gating)
- Test: `tests/build/test_mvp_113_supply_chain.py`
**Interfaces:**
- Consumes: `requirements.lock`, `requirements-dev.lock`, `IMAGES` and built `solace/<name>:<tag>` images; WS-03's W4 digest pins in the compose files (request R-01.1).
- Produces: `supply_chain.py audit` (runs `python -m pip_audit -r <lock> --require-hashes --disable-pip --no-deps -f json` for both locks), `scan --tag TAG` (runs Trivy from a digest-pinned `aquasec/trivy` container against each image with `--severity HIGH,CRITICAL -f json`), `sbom --out DIR` (pip-audit `-f cyclonedx-json` per lock and Trivy `-f cyclonedx` per image), `gate` (reads the JSON reports): fails on any pip-audit finding or any HIGH/CRITICAL Trivy finding that has no unexpired entry in `accepted_vulns.yaml`, and on any entry that expires more than 90 days ahead. Pure functions `gate(findings: list[Finding], accepted: list[Acceptance], today: date) -> list[str]` and `load_acceptances(path) -> list[Acceptance]`.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_113_supply_chain.py`
```python
"""MVP-113/355: CI gates on dependency and image vulnerabilities, emits SBOMs and pins every image."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_mvp_113_gate_fails_unaccepted_and_expired_findings() -> None:
    from scripts.ci.supply_chain import Acceptance, Finding, gate

    findings = [Finding("GHSA-aaaa", "pyjwt", "HIGH"), Finding("CVE-2026-1", "solace/user-service", "CRITICAL"),
                Finding("CVE-2026-2", "solace/user-service", "MEDIUM")]
    accepted = [Acceptance("CVE-2026-1", "solace/user-service", "no fix upstream", "ops", date(2026, 11, 1))]
    assert gate(findings, accepted, today=date(2026, 10, 20)) == ["GHSA-aaaa pyjwt HIGH"]
    assert "CVE-2026-1 solace/user-service CRITICAL" in gate(findings, accepted, today=date(2026, 11, 2))


def test_mvp_113_acceptances_expire_within_90_days() -> None:
    from scripts.ci.supply_chain import load_acceptances

    for a in load_acceptances(ROOT / "scripts/ci/accepted_vulns.yaml"):
        assert (a.expires - date.today()).days <= 90 and a.owner and a.reason


def test_mvp_113_dependabot_covers_pip_docker_and_actions() -> None:
    cfg = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    assert {u["package-ecosystem"] for u in cfg["updates"]} >= {"pip", "docker", "github-actions"}


def test_mvp_113_ci_runs_the_supply_chain_gate() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    runs = "\n".join(s.get("run", "") for s in jobs["supply-chain"]["steps"])
    for sub in ("supply_chain.py audit", "supply_chain.py scan", "supply_chain.py sbom", "supply_chain.py gate"):
        assert sub in runs


def test_mvp_355_compose_images_are_digest_pinned_and_app_images_tagged() -> None:
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        services = yaml.safe_load((ROOT / name).read_text(encoding="utf-8")).get("services", {})
        for svc, spec in services.items():
            image = spec.get("image")
            if "build" in spec:
                assert image and re.match(r"^solace/[a-z-]+:\$\{SOLACE_IMAGE_TAG", image), f"{name}:{svc}"
            elif image:
                assert "@sha256:" in image, f"{name}:{svc} {image}"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_113_supply_chain.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `scripts.ci.supply_chain` is missing; `.github/dependabot.yml` is missing; ci.yml has no `supply-chain` job; the compose test fails until WS-03's W4 pins land (for example `postgres:16-alpine` without a digest, and app services with `build:` but no `image:` tag).
- [ ] **Step 3: Implement** the files listed. Resolve the Trivy image digest with `docker buildx imagetools inspect aquasec/trivy:<current release>` and pin it in `supply_chain.py`. Pulling by digest is allowed; an unpinned pull is a stop condition.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command (after WS-03's W4 compose pins). Expected: PASS. Then produce real evidence: `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\build_images.py --all --tag ci`, then `scripts\ci\supply_chain.py audit`, `scan --tag ci`, `sbom --out D:\tmp\mvp-exec\ws01\sbom` and `gate`, each run with the same interpreter. Expected: `gate` exits 0, or every remaining finding has an owner-approved entry recorded as a ledger Ruling.
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.11: Coverage gate over src/ and services/ with a safety-critical floor
**Covers:** MVP-243
**Severity:** low
**Wave:** 5
**Phase:** A
**Depends on:** WS-01.4, WS-01.8
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `scripts/ci/coverage_gate.py`
- Modify: `pytest.ini` (remove `--cov=src`, `--cov-report=term-missing`, `--cov-report=xml:tests/coverage.xml` and `--cov-branch` from `addopts`; coverage runs only through the runner)
- Modify: `pyproject.toml` (`[tool.coverage.report] fail_under = <measured total, rounded down>`; new table `[tool.solace.coverage_floors]` with `"services/safety_service/src" = <measured, rounded down, at least 80>` and `"src/solace_clinical" = <measured, rounded down, at least 80>` once it exists)
- Modify: `scripts/ci/run_suites.py` (`--coverage OUTDIR`: each suite runs with `--cov --cov-branch --cov-report=` and `COVERAGE_FILE=OUTDIR/.coverage.<n>`; afterwards `coverage combine` and `coverage json -o OUTDIR/coverage.json`), `scripts/ci/ci_local.py` (stage `coverage`, gating), `.github/workflows/ci.yml` (job `coverage`), `Makefile` (target `coverage`)
- Test: `tests/build/test_mvp_243_coverage_gate.py`
**Interfaces:**
- Consumes: `[tool.coverage.run] source = ["src", "services"]` (unchanged), the SUITES registry.
- Produces: `coverage_gate.py --report OUTDIR/coverage.json` exits 1 when the total is below `fail_under` or any `coverage_floors` path is below its floor, printing each shortfall; `check(report: dict, fail_under: float, floors: dict[str, float]) -> list[str]`.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_243_coverage_gate.py`
```python
"""MVP-243: services/ is measured and a threshold gates CI, with a floor for safety-critical code."""
from __future__ import annotations

import configparser
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _report(total: float, safety: float) -> dict:
    return {"totals": {"percent_covered": total},
            "files": {"services/safety_service/src/domain/crisis_detector.py": {"summary": {"percent_covered": safety, "num_statements": 100}},
                      "src/solace_common/enums.py": {"summary": {"percent_covered": 100.0, "num_statements": 10}}}}


def test_mvp_243_gate_checks_total_and_safety_floor() -> None:
    from scripts.ci.coverage_gate import check

    floors = {"services/safety_service/src": 80.0}
    assert check(_report(75.0, 90.0), 70.0, floors) == []
    assert check(_report(65.0, 90.0), 70.0, floors) == ["total 65.0 < 70.0"]
    assert check(_report(75.0, 50.0), 70.0, floors) == ["services/safety_service/src 50.0 < 80.0"]


def test_mvp_243_pytest_ini_no_longer_overrides_coverage_source() -> None:
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "pytest.ini", encoding="utf-8")
    assert "--cov" not in cfg["pytest"]["addopts"]


def test_mvp_243_thresholds_configured() -> None:
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
    assert tool["coverage"]["run"]["source"] == ["src", "services"]
    assert tool["coverage"]["report"]["fail_under"] > 0
    assert tool["solace"]["coverage_floors"]["services/safety_service/src"] >= 80
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_243_coverage_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `scripts.ci.coverage_gate` is missing; pytest.ini `addopts` contains `--cov=src` (line 18); pyproject has no `fail_under` and no `[tool.solace]` table.
- [ ] **Step 3: Implement** the gate and runner mode, then measure: `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\run_suites.py --tier unit --coverage D:\tmp\mvp-exec\ws01\coverage`. Set `fail_under` to the measured total rounded down, and the safety floor to `max(80, measured safety_service/src rounded down)`. If safety_service/src measures below 80, set the floor to the measured value, record the shortfall in the ledger as a W5 finding for WS-12, and raise the floor when WS-12 closes it.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command, then `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\coverage_gate.py --report D:\tmp\mvp-exec\ws01\coverage\coverage.json`. Expected: PASS, and the gate exits 0
- [ ] **Step 5: Regression:** Run: `make ci-local PYTHON="D:\tmp\mvp-exec\venv\Scripts\python.exe"`. Expected: all stages green (W5 release-gate condition)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.12: Closure check for the Kafka-topic part of MVP-269
**Covers:** MVP-269
**Severity:** low
**Wave:** 5
**Phase:** A
**Depends on:** WS-01.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-24 (default assumed: Kafka `solace.safety` and `solace.users` retain 7 days)
**Real infra:** no
**Files:**
- Test: `tests/build/test_mvp_269_topic_registry.py` (test-only task: the code changes are WS-06's single topic registry and WS-03's strict overlay, requested below)
**Interfaces:**
- Consumes: `solace_events.config.TOPIC_CONFIGS` and `SolaceTopic` (WS-06), `solace_infrastructure.kafka.topics.TopicManager.register_solace_topics()` and `get_registered_topics()` (WS-06, if it survives), `docker-compose.prod.yml` (WS-03).
- Produces: the MVP-269 acceptance check; the issue closes when this test and WS-01.3/WS-02.7 are green.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_269_topic_registry.py`
```python
"""MVP-269 (Kafka part): one topic registry with the D-24 retention is applied, and strict brokers never auto-create topics."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000


def test_mvp_269_registry_carries_d24_retention() -> None:
    from solace_events.config import TOPIC_CONFIGS, SolaceTopic

    assert TOPIC_CONFIGS[SolaceTopic.SAFETY].retention_ms == SEVEN_DAYS_MS
    assert TOPIC_CONFIGS[SolaceTopic.USERS].retention_ms == SEVEN_DAYS_MS


def test_mvp_269_legacy_registry_agrees_or_is_gone() -> None:
    from solace_events.config import TOPIC_CONFIGS

    try:
        from solace_infrastructure.kafka.topics import TopicManager
    except ImportError:
        return
    manager = TopicManager()
    manager.register_solace_topics()
    legacy = manager.get_registered_topics()
    for cfg in TOPIC_CONFIGS.values():
        if cfg.name in legacy:
            assert (legacy[cfg.name].partitions, legacy[cfg.name].retention_ms) == (cfg.partitions, cfg.retention_ms), cfg.name


def test_mvp_269_strict_overlay_disables_topic_auto_create() -> None:
    prod = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    env = prod["services"]["kafka"]["environment"]
    value = env.get("KAFKA_AUTO_CREATE_TOPICS_ENABLE") if isinstance(env, dict) else next(
        (e.split("=", 1)[1] for e in env if e.startswith("KAFKA_AUTO_CREATE_TOPICS_ENABLE=")), None)
    assert str(value).lower() == "false"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_269_topic_registry.py -o "addopts=" -p no:logging -q`
Expected: FAIL unless WS-06 and WS-03 have already landed their parts: `retention_ms` is `7776000000` (90-day default, `src/solace_events/config.py:45`); the legacy registry gives `solace.safety` `31536000000` (`topics.py:324`); `docker-compose.prod.yml` has no `KAFKA_AUTO_CREATE_TOPICS_ENABLE` override. If a part already passes, record which request it proves.
- [ ] **Step 3: Implement** nothing in WS-01 files. Confirm requests R-01.4 (WS-06) and R-01.1 (WS-03) have completed tasks in the ledger; if either is missing at W5 entry, the controller records a Ruling (schedule the owner's fix or owner-accept MVP-269's Kafka part).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.13: Lint the staging CloudFormation template in make ci-local and CI
**Covers:** MVP-043, MVP-044 (supporting; WS-03 owns both and closes them in WS-03.11)
**Severity:** high (WS-03's entries; this is the CI part WS-03 requested)
**Wave:** 4
**Phase:** A
**Depends on:** WS-01.10, WS-03.11
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-25 (decided 2026-10-19: AWS hosts the release gate and the pilot, using the existing IAM user `MedCodex-Local` in account 253025918446, us-east-1. Provisioning stays blocked until the owner attaches `infrastructure/aws-staging/solace-staging-deployer-policy.json` to that user (option 2 of AWS-STAGING-IAM-SETUP.pdf). This task therefore only lints the template with cfn-lint and cfn-guard and deploys nothing. The W5 release gate runs on the AWS staging host through SSM with the strict production profile (WS-05.19). If the policy is still unattached at W5 entry, that is a stop for the owner, never a silent fallback to local Docker. Until then every real-infra test runs on local Docker). D-19 (default assumed: config-service is not an MVP image)
**Real infra:** no
**Files:**
- Create: `scripts/ci/iac_lint.py` (stdlib only)
- Modify: `scripts/ci/ci_local.py` (stage `iac-lint` right after `lint`, gating), `Makefile` (target `iac-lint`, one recipe `$(PYTHON) scripts/ci/iac_lint.py`), `.github/workflows/ci.yml` (job `iac`: uv 0.8.2, `uv pip sync --system --require-hashes requirements-dev.lock`, cfn-guard installed from a pinned 3.x GitHub release asset, then `python scripts/ci/iac_lint.py --require-guard`; no `needs`)
- Test: `tests/build/test_mvp_043_iac_lint_stage.py`
**Interfaces:**
- Consumes: WS-03.11's staging template and guard rules file (read only, never edited here); `cfn-lint==1.51.5` in the dev lock (WS-01.2); `IMAGES` (WS-01.5); the CI layout from WS-01.8 and WS-01.10 (this task follows WS-01.10 because both edit `ci.yml`, `Makefile` and `ci_local.py` in W4).
- Produces: `scripts.ci.iac_lint` with `TEMPLATE = "infrastructure/aws-staging/solace-staging.yaml"`, `GUARD_RULES = "infrastructure/aws-staging/guard/solace-staging.guard"`, `commands(root: Path, *, python: str) -> list[list[str]]` (cfn-lint as `[python, "-m", "cfnlint", <template>]`, then `["cfn-guard", "validate", "-d", <template>, "-r", <rules>]`), `run(root: Path, *, python: str, require_guard: bool) -> int` and the CLI `iac_lint.py [--require-guard] [--python PATH]`. When `cfn-guard` is not on PATH, `run` prints `SKIP cfn-guard (not installed)` and returns cfn-lint's result, or prints `FAIL cfn-guard not installed` and returns 1 under `--require-guard`. This is WS-03's W4 request for WS-03.11 (whose local `test_cfn_guard_passes` skips without the binary), including the image matrix without config-service.

- [ ] **Step 1: Write the failing test** in `tests/build/test_mvp_043_iac_lint_stage.py`
```python
"""MVP-043/044 (supporting WS-03.11): make ci-local and CI lint the staging CloudFormation template; no config-service image."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from scripts.ci import iac_lint
from scripts.ci.images import IMAGES

ROOT = Path(__file__).resolve().parents[2]


def test_iac_lint_runs_cfn_lint_then_cfn_guard_on_the_staging_template() -> None:
    template = ROOT / "infrastructure" / "aws-staging" / "solace-staging.yaml"
    rules = ROOT / "infrastructure" / "aws-staging" / "guard" / "solace-staging.guard"
    assert iac_lint.commands(ROOT, python="py") == [
        ["py", "-m", "cfnlint", str(template)],
        ["cfn-guard", "validate", "-d", str(template), "-r", str(rules)]]


def test_missing_cfn_guard_fails_only_when_required(monkeypatch, capsys) -> None:
    ran: list[list[str]] = []

    class _Done:
        returncode = 0

    monkeypatch.setattr(iac_lint.shutil, "which", lambda name: None)
    monkeypatch.setattr(iac_lint.subprocess, "run", lambda cmd, **kw: ran.append(cmd) or _Done())
    assert iac_lint.run(ROOT, python=sys.executable, require_guard=False) == 0
    assert "SKIP cfn-guard" in capsys.readouterr().out
    assert iac_lint.run(ROOT, python=sys.executable, require_guard=True) == 1
    assert ran and all(cmd[0] != "cfn-guard" for cmd in ran)


def test_iac_lint_is_wired_into_make_and_ci_local() -> None:
    assert re.search(r"^iac-lint:", (ROOT / "Makefile").read_text(encoding="utf-8"), re.M)
    assert "iac_lint.py" in (ROOT / "scripts" / "ci" / "ci_local.py").read_text(encoding="utf-8")


def test_ci_iac_job_requires_a_pinned_cfn_guard() -> None:
    steps = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]["iac"]["steps"]
    text = "\n".join(str(s.get("run", "")) + str(s.get("with", "")) for s in steps)
    assert "iac_lint.py --require-guard" in text
    assert "uv pip sync --system --require-hashes requirements-dev.lock" in text
    assert re.search(r"cfn-guard[^\n]*\b3\.\d+\.\d+\b", text)


def test_d19_image_matrix_excludes_config_service() -> None:
    assert "config-service" not in {i.name for i in IMAGES}
    assert "config_service" not in (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_043_iac_lint_stage.py -o "addopts=" -p no:logging -q`
Expected: FAIL: collection stops with `ModuleNotFoundError: No module named 'scripts.ci.iac_lint'`. With an empty `iac_lint.py` stub the wiring tests fail: the Makefile has no `iac-lint:` target, `ci_local.py` never mentions `iac_lint.py`, and `jobs["iac"]` raises `KeyError`. `test_d19_image_matrix_excludes_config_service` already passes (WS-01.5, WS-01.8) and guards D-19.
- [ ] **Step 3: Implement** `iac_lint.py` per Produces, the `iac-lint` stage and target, and the `iac` job. Resolve the cfn-guard release once (`https://github.com/aws-cloudformation/cloudformation-guard/releases`, current 3.x) and pin its version and the asset's SHA-256 in the job (`sha256sum -c` before extracting); an unpinned download is a stop condition.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS. Then run `& "D:\tmp\mvp-exec\venv\Scripts\python.exe" scripts\ci\iac_lint.py` and record the result: cfn-lint passes on WS-03.11's template, and cfn-guard prints `SKIP` locally unless the binary is installed (CI runs it with `--require-guard`).
- [ ] **Step 5: Regression:** the shared_files protocol requires `make ci-local` for any workflow change. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`, then `make ci-local PYTHON="D:\tmp\mvp-exec\venv\Scripts\python.exe"`. Expected: all tests pass; the `iac-lint` stage reports PASS in the summary table (record it in the ledger).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.14: SOPS-aware ignore rules, build-context exclusions and a checkpoint-tree secret scan
**Covers:** MVP-348 (supporting; WS-03 owns it and closes it in WS-03.8), MVP-008 (supporting: the encrypted secrets files must reach checkpoint trees)
**Severity:** blocker (WS-03's MVP-348; this is the tree-hygiene part that D-07 needs)
**Wave:** 0
**Phase:** A1 (after WS-01.5; it must land before anyone creates an age key or a file under `deploy/secrets/`)
**Depends on:** WS-01.1, WS-01.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-07 (decided 2026-10-19: SOPS + age, no AWS Secrets Manager. Each environment has a SOPS-encrypted `deploy/secrets/<env>.sops.yaml`, encrypted to that environment's age recipient(s); it is safe in the tree and must reach checkpoint trees. The age private keys are held off-host by the owner or operator. They are never in the repo, the images or the plans, and arrive at deploy time through `SOPS_AGE_KEY_FILE` at a root-only path. Plaintext exists only in tmpfs `/run/secrets`. So age private keys and any decrypted secrets file are ignored, kept out of every build context, and caught by the secret scan). D-17 (no commits: pre-commit hooks never run, so the scan runs as a `tests/build` test in every regression, barrier and `make ci-local`)
**Real infra:** no
**Files:**
- Modify: `.gitignore` (append the SOPS block listed in Step 3 right after the existing secrets-directory line; change no existing line)
- Modify: `.dockerignore` (append the three patterns listed in Step 3; keep every existing line)
- Create: `scripts/ci/secret_scan.py`
- Test: `tests/build/test_d07_sops_tree_hygiene.py`
**Interfaces:**
- Consumes: `scripts.ci.dockerfiles.dockerignored` (WS-01.5); `git ls-files --cached --others --exclude-standard -z`, which lists exactly what `checkpoint.ps1`'s `git add -A` captures.
- Produces:
  - `scripts.ci.secret_scan.Finding(rule: str, path: str)`, a frozen dataclass whose `str()` is `"<rule> <path>"` and never contains file content.
  - `visible_paths(root: Path) -> list[str]`.
  - `scan(root: Path, paths: Iterable[str]) -> list[Finding]`, with these rules:
    - `age-private-key`: the file's bytes contain age's private-key prefix followed by at least 50 bech32 characters.
    - `age-key-file`: a file named `*.agekey` or `keys.txt`.
    - `stray-file-in-deploy-secrets`: a file under `deploy/secrets/` other than the ones WS-03 plans. Those are `<env>.sops.yaml`, the creation-rules file `.sops.yaml`, `escrow-recipients.txt` (public recipients only), `README.md`, and previous encrypted versions at `history/<env>/<UTC timestamp>.sops.yaml`.
    - `sops-metadata-missing`: a `*.sops.yaml` file other than a creation-rules file named exactly `.sops.yaml`, with no top-level `sops` mapping that holds a non-empty `age` list and a `mac`.
    - `sops-plaintext-value`: a scalar leaf outside `sops` that is not a string starting with `ENC[AES256_GCM,`.
  - The CLI `secret_scan.py [--root DIR]`. It prints one finding per line and exits 1 if there is any finding, 0 otherwise.
  - The SOPS conventions WS-03.8 must follow (R-01.18).
  - For WS-03's `.dockerignore` row (needed by WS-03.7 and WS-03.8, D-07): `deploy/secrets`, `**/*.agekey` and `**/keys.txt` are excluded from every build context from W0, before any file exists there; WS-03.8 depends on this task.

  The prefix is never written contiguously in any tracked file: code and tests build it as `"AGE-SECRET-" + "KEY-1"`, so the scanner, its tests and these plans never flag themselves.

- [ ] **Step 1: Write the failing test** in `tests/build/test_d07_sops_tree_hygiene.py`
```python
"""D-07: SOPS-encrypted secrets files reach checkpoint trees; age private keys and decrypted secrets never do."""
from __future__ import annotations

import secrets
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci import dockerfiles, secret_scan

ROOT = Path(__file__).resolve().parents[2]
BECH32 = "QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L"
ENC = "ENC[AES256_GCM,data:c3ludGhldGlj,iv:AAAAAAAAAAAAAAAAAAAAAA==,tag:AAAAAAAAAAAAAAAAAAAAAA==,type:str]"
SOPS_META = ("sops:\n  age:\n    - recipient: age1" + "q" * 58 + "\n      enc: |\n"
             "        -----BEGIN AGE ENCRYPTED FILE-----\n        c3ludGhldGlj\n        -----END AGE ENCRYPTED FILE-----\n"
             "  lastmodified: '2026-10-19T00:00:00Z'\n  mac: " + ENC + "\n  version: 3.9.0\n")


def _synthetic_age_identity() -> str:
    # random characters in the identity format; no valid checksum, never a usable key
    return "AGE-SECRET-" + "KEY-1" + "".join(secrets.choice(BECH32) for _ in range(58))


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", path], capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    return r.returncode == 0


@pytest.mark.parametrize("path", ["deploy/secrets/staging.sops.yaml", "deploy/secrets/production.sops.yaml",
                                  "deploy/secrets/.sops.yaml", "deploy/secrets/README.md",
                                  "deploy/secrets/escrow-recipients.txt",
                                  "deploy/secrets/history/staging/20261019T000000Z.sops.yaml"])
def test_d07_encrypted_secrets_files_are_visible_to_checkpoints(path: str) -> None:
    assert not _ignored(path), f"{path} is gitignored, so checkpoint trees would drop it"


@pytest.mark.parametrize("path", ["deploy/secrets/staging.yaml", "deploy/secrets/staging.dec.yaml",
                                  "deploy/secrets/keys.txt", "deploy/secrets/staging.agekey",
                                  "deploy/secrets/history/staging/plain.yaml", "keys.txt",
                                  "ops/staging.agekey", "secrets/x.txt"])
def test_d07_age_keys_and_decrypted_files_stay_ignored(path: str) -> None:
    assert _ignored(path), f"{path} must stay ignored"


def test_d07_build_contexts_exclude_sops_files_and_age_keys() -> None:
    ignore = ROOT / ".dockerignore"
    for rel in ("deploy/secrets/staging.sops.yaml", "deploy/secrets/staging.yaml", "ops/staging.agekey", "keys.txt"):
        assert dockerfiles.dockerignored(rel, ignore), rel


def test_d07_scan_flags_age_keys_stray_files_and_plaintext(tmp_path: Path) -> None:
    (tmp_path / "deploy" / "secrets").mkdir(parents=True)
    (tmp_path / "notes.txt").write_text("key " + _synthetic_age_identity() + "\n", encoding="utf-8")
    (tmp_path / "deploy/secrets/staging.sops.yaml").write_text("AUTH_SECRET_KEY: SYNTHETIC-PLAINTEXT\n" + SOPS_META, encoding="utf-8")
    (tmp_path / "deploy/secrets/production.sops.yaml").write_text("AUTH_SECRET_KEY: " + ENC + "\n", encoding="utf-8")
    (tmp_path / "deploy/secrets/staging.yaml").write_text("AUTH_SECRET_KEY: SYNTHETIC-PLAINTEXT\n", encoding="utf-8")
    (tmp_path / "ops.agekey").write_text("synthetic\n", encoding="utf-8")
    paths = ["notes.txt", "deploy/secrets/staging.sops.yaml", "deploy/secrets/production.sops.yaml",
             "deploy/secrets/staging.yaml", "ops.agekey"]
    findings = secret_scan.scan(tmp_path, paths)
    assert {(f.rule, f.path) for f in findings} == {
        ("age-private-key", "notes.txt"), ("sops-plaintext-value", "deploy/secrets/staging.sops.yaml"),
        ("sops-metadata-missing", "deploy/secrets/production.sops.yaml"),
        ("stray-file-in-deploy-secrets", "deploy/secrets/staging.yaml"), ("age-key-file", "ops.agekey")}
    assert all("SYNTHETIC-PLAINTEXT" not in str(f) and "KEY-1" not in str(f) for f in findings)  # names only, never content


def test_d07_scan_accepts_the_planned_encrypted_and_public_files(tmp_path: Path) -> None:
    (tmp_path / "deploy" / "secrets" / "history" / "staging").mkdir(parents=True)
    body = "AUTH_SECRET_KEY: " + ENC + "\nnested:\n  X: " + ENC + "\n" + SOPS_META
    (tmp_path / "deploy/secrets/staging.sops.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "deploy/secrets/history/staging/20261019T000000Z.sops.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "deploy/secrets/escrow-recipients.txt").write_text("# public recipients only\nage1" + "q" * 58 + "\n", encoding="utf-8")
    paths = ["deploy/secrets/staging.sops.yaml", "deploy/secrets/history/staging/20261019T000000Z.sops.yaml",
             "deploy/secrets/escrow-recipients.txt"]
    assert secret_scan.scan(tmp_path, paths) == []


def test_d07_checkpoint_visible_tree_has_no_secret_findings() -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ci" / "secret_scan.py"), "--root", str(ROOT)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_d07_sops_tree_hygiene.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Collection first stops with `ModuleNotFoundError: No module named 'scripts.ci.secret_scan'`. Create an empty `secret_scan.py` stub and re-run. The second RED run shows:
- All six `test_d07_encrypted_secrets_files_are_visible_to_checkpoints` cases fail. `.gitignore`'s unanchored `secrets/` line hides all of `deploy/secrets/`, so the encrypted files D-07 keeps in the tree would be missing from every checkpoint tree.
- `test_d07_age_keys_and_decrypted_files_stay_ignored` fails for `keys.txt` and `ops/staging.agekey`, because no pattern covers age keys (`*.key` does not match `.agekey`). The `deploy/secrets/...` cases pass today only because the whole directory is hidden.
- `test_d07_build_contexts_exclude_sops_files_and_age_keys` fails on `deploy/secrets/staging.sops.yaml`, because `.dockerignore`'s `secrets` pattern is anchored at the context root.
- The two scan tests fail with `AttributeError` (no `scan`). The tree test passes against the empty stub (exit 0). From Step 4 on, it is the guard that keeps the real checkpoint-visible tree clean.
- [ ] **Step 3: Implement**
  - `.gitignore`: append these lines right after the existing `secrets/` line, in this order, so the re-inclusion can override it:
    ```
    # D-07 (SOPS + age): only encrypted files and public-recipient lists under deploy/secrets/ are tracked
    !deploy/secrets/
    deploy/secrets/*
    !deploy/secrets/*.sops.yaml
    !deploy/secrets/escrow-recipients.txt
    !deploy/secrets/README.md
    !deploy/secrets/history/
    deploy/secrets/history/**
    !deploy/secrets/history/**/
    !deploy/secrets/history/**/*.sops.yaml
    # age private keys never enter the tree
    *.agekey
    keys.txt
    ```
  - `.dockerignore`: append `deploy/secrets`, `**/*.agekey` and `**/keys.txt`. Images never carry even the encrypted files, because `render_secrets.py` runs on the host.
  - `scripts/ci/secret_scan.py`, with stdlib plus PyYAML (`yaml.safe_load`):
    - `visible_paths()` runs `git -C <root> ls-files --cached --others --exclude-standard -z` (read-only).
    - `scan()` reads each existing path as bytes, so binary files are searched too, and applies the rules under Produces.
    - It compiles the age rule as `re.compile(re.escape(("AGE-SECRET-" + "KEY-1").encode()) + rb"[02-9AC-HJ-NP-Z]{50,}")`.
    - A `*.sops.yaml` that does not parse as a mapping is reported as `sops-metadata-missing`.
    - Output is rule and path only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** confirm nothing sensitive became visible. Run: `git -C D:\Repo\Contextual-Chatbot ls-files --others --exclude-standard | Select-String -Pattern '(^|/)\.env|\.key$|\.crt$|\.pem$|\.agekey$|(^|/)keys\.txt$|^deploy/secrets/(?!.*\.sops\.yaml$)(?!README\.md$)(?!escrow-recipients\.txt$)'` (read-only; prints names only). Expected: no output. Then run `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build -o "addopts=" -p no:logging -q`. Expected: all pass. `test_mvp_008_git_visibility.py` and `test_mvp_032_dockerfiles.py` stay green.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-01.F1: Security bump of the advisory-affected pins in both locks
**Covers:** MVP-359, MVP-005
**Severity:** high
**Wave:** 0
**Depends on:** WS-01.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-16 (the default includes security bumps on top of the tested versions). Added by controller Ruling on 2026-11-02, after WS-01.2's pip-audit run found 70 advisories in 25 runtime packages (`D:\tmp\mvp-exec\ws01\pip-audit-runtime.txt`).
**Real infra:** no
**Files:**
- Modify: `requirements.in`, `requirements-dev.in`, `requirements.lock`, `requirements-dev.lock`
- Create: `scripts/ci/accepted_vulns.yaml` (entries `{id, target, reason, owner, expires}`; `accepted: []` when nothing is accepted). WS-01.10 later modifies it and adds the gate.
- Test: `tests/build/test_mvp_359_lock_has_no_unaccepted_advisories.py`
**Interfaces:**
- Consumes: WS-01.2's locks, `scripts/ci/lock_delta.py` and `scripts/ci/clean_venv.py`; pip-audit 2.9.0 from the dev lock.
- Produces: both locks with every advisory either fixed or listed in `scripts/ci/accepted_vulns.yaml` with a reason, owner `controller` and an expiry of at most 90 days. No lock line names a local path.

- [ ] **Step 1: Write the failing test** `tests/build/test_mvp_359_lock_has_no_unaccepted_advisories.py`:
  - `test_mvp_359_runtime_lock_advisories_are_fixed_or_accepted` runs `python -m pip_audit -r requirements.lock --require-hashes --disable-pip --no-deps -f json --progress-spinner off`. Mark it `realinfra`, because it needs network access to the advisory database. It asserts that every vulnerability id is listed in `accepted_vulns.yaml` for that package with an expiry later than today and no more than 90 days away.
  - `test_mvp_359_dev_lock_advisories_are_fixed_or_accepted` does the same for `requirements-dev.lock`.
  - `test_mvp_005_locks_name_no_local_paths` (offline) asserts that no line of either lock contains `D:/`, `D:\`, `C:/`, `C:\` or `/tmp/`.
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\build\test_mvp_359_lock_has_no_unaccepted_advisories.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The runtime test fails because `accepted_vulns.yaml` does not exist and about 70 ids are unaccepted. The dev test fails the same way. The local-path test fails on the `#   -c D:/tmp/mvp-exec/ws01/...` annotation lines (107 in the runtime lock, 268 in the dev lock).
- [ ] **Step 3: Implement**
  1. For each affected package, raise its pin in `requirements.in` or `requirements-dev.in` to the lowest version that fixes all its advisories, staying within the same major version. If the fix needs a co-dependency bump (for example starlette with fastapi), bump the pair as far as the fix requires.
  2. `langgraph-checkpoint-postgres` may move to the lowest fixed 3.1.x only if it resolves with `langgraph-checkpoint` and `langgraph` as pinned. Update its slot-017 comment to say WS-10's slot 017 must target that version. Otherwise accept it with the reason "needs the slot-017 follow-up (WS-10)".
  3. Recompile both locks with WS-01.2's commands, adding `--no-annotate` so no `-c` path is written. Run `lock_delta.py --lock requirements-dev.lock --write-install D:\tmp\mvp-exec\ws01\venv-sync-f1.txt` and write that list to the report. Install it into `D:\tmp\mvp-exec\venv` only.
  4. Anything that cannot be bumped without breaking a test goes into `accepted_vulns.yaml` with the advisory id, the reason, owner `controller` and an expiry of 2027-01-31 or earlier.
- [ ] **Step 4: Run to pass:** the Step 2 command. Expected: PASS.
- [ ] **Step 5: Regression.** Run every suite with `D:\tmp\mvp-exec\tools\run-suites.ps1 -Out D:\tmp\mvp-exec\baseline\after-WS-01.F1.txt` and rebuild the clean venv with `scripts\ci\clean_venv.py --build`, then run `tests\build` in it. Expected: the same counts as the W0-entry baseline (5,365 passed plus the 12 ruled `kafka/test_topics.py` failures) and `tests\build` green in both venvs. Any other change in a count means the bump that caused it is reverted to an acceptance entry.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.F1:head"`.

### Task WS-01.F2: Patch-level bumps for the checkpoint-path advisories held by exact pins
**Covers:** MVP-358, MVP-359
**Severity:** high
**Wave:** 0
**Depends on:** WS-01.F1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-16 (security bumps). Added by controller Ruling on 2026-11-02 from WS-01.F1's concern 1; the evidence is in `D:\tmp\mvp-exec\reports\WS-01.F1-report.md`.
**Real infra:** no
**Files:**
- Modify: `requirements.in`, `requirements.lock`, `requirements-dev.lock`, `scripts/ci/accepted_vulns.yaml`
- Modify: `tests/build/test_mvp_005_dependency_lock.py` (only the exact pins in `test_mvp_358_*` and the D-16 pin assertions for the bumped packages)
**Interfaces:**
- Consumes: WS-01.F1's locks and acceptances.
- Produces: langgraph, langgraph-checkpoint, langchain-core and pydantic-settings at their lowest fixed version within the same minor line. The fastapi/starlette fix is included only if the full regression stays at baseline. Each fixed id is removed from `accepted_vulns.yaml`.

- [ ] **Step 1: Write the failing test.** In `test_mvp_005_dependency_lock.py`, change the expected pins to the fixed versions (for example langgraph `1.0.10`; take the others from the F1 report's pip-audit fix versions). Add `test_mvp_358_checkpoint_path_has_no_accepted_advisories`, which asserts that `accepted_vulns.yaml` has no entry whose target is langgraph, langgraph-checkpoint or langchain-core.
- [ ] **Step 2: Run it and confirm it fails.** Run tests\build with the venv interpreter. Expected: FAIL on the old pins and on the accepted checkpoint-path entries.
- [ ] **Step 3: Implement.** Raise the pins in `requirements.in` and recompile both locks with the F1 commands (with `--no-annotate`). Write the venv sync list to the report, then install it into `D:\tmp\mvp-exec\venv` only. Remove the fixed ids from `accepted_vulns.yaml`. Then try fastapi at the lowest version that admits a fixed starlette. Keep that change only if Step 5 stays at baseline; otherwise revert it and leave the starlette entries accepted.
- [ ] **Step 4: Run to pass:** tests\build. Expected: PASS.
- [ ] **Step 5: Regression:** `D:\tmp\mvp-exec\tools\run-suites.ps1 -Out D:\tmp\mvp-exec\baseline\after-WS-01.F2.txt`, then `scripts\ci\clean_venv.py --build` and tests\build in the clean venv. Expected: 5,365 passed plus the 12 ruled failures, and both venvs green. Revert any bump that changes a count.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.F2:head"`.

### Task WS-01.F3: The repo's tests package wins over portkey-ai's site-packages "tests"
**Covers:** MVP-005
**Severity:** high
**Wave:** 0
**Depends on:** WS-01.F1
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none. Added by controller Ruling on 2026-11-02 from WS-03.1's concern 1. portkey-ai 1.9.10, which is in the runtime lock, installs a top-level `tests` package into site-packages. The repo's `tests/` has no `__init__.py`, so a regular package beats it. In the venv, `importlib.util.find_spec("tests")` resolves to `site-packages\tests\__init__.py`, and every `from tests.<x> import ...` loads the wrong module.
**Real infra:** no
**Files:**
- Create: `tests/__init__.py` (docstring only), or, if the full regression shows that it breaks pytest collection, the least invasive alternative that makes `tests` resolve to the repo. Choose it and record it in the report.
- Test: `tests/build/test_mvp_005_tests_package_is_the_repo.py`
**Interfaces:**
- Produces: with the repo root on `sys.path` before site-packages, `import tests` resolves to the repo's `tests/`.

- [ ] **Step 1: Write the failing test.** `test_mvp_005_tests_package_resolves_to_the_repo` runs `[sys.executable, "-c", "import importlib.util as u; print(u.find_spec('tests').origin)"]` with `PYTHONPATH=<repo>;<repo>\src` and asserts the printed path is inside the repo.
- [ ] **Step 2: Run it and confirm it fails.** Expected: FAIL; the origin is `...\site-packages\tests\__init__.py`.
- [ ] **Step 3: Implement it in a sandbox first.** Export the current tree with `git -C D:\Repo\Contextual-Chatbot archive --format=zip -o D:\tmp\mvp-exec\f3\tree.zip <checkpoint tree>`, then expand it to `D:\tmp\mvp-exec\f3\tree`. Add the file there and run `D:\tmp\mvp-exec\tools\run-suites.ps1`-style runs against that copy, with `PYTHONPATH` pointing at the copy. Compare every suite's counts with `D:\tmp\mvp-exec\baseline\venv-baseline.txt` (5,365 passed plus 12 ruled failures). Suites that have grown since the baseline are compared to the latest per-task report. Only when the sandbox regression is clean, create the file in the real repo.
- [ ] **Step 4: Run to pass:** the Step 1 test in the real repo. Expected: PASS.
- [ ] **Step 5: Regression:** tests\build, tests\deploy (WS-03.1 aliases `tests.deploy.conftest` in `sys.modules`; it must keep working) and tests\solace_testing in the real repo.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.F3:head"`.

### Task WS-01.F4: Entrypoint hardening from the WS-01.6 review
**Covers:** MVP-211
**Severity:** low
**Wave:** 0
**Depends on:** WS-01.6, WS-01.8
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** none. Added by controller Ruling on 2026-11-02 from the WS-01.6 Opus review. That review approved the task, so these are its minor findings.
**Real infra:** yes
**Files:**
- Modify: the nine service Dockerfiles (`services/*/Dockerfile`): `COPY --from=builder /build/entrypoint.sh /app/docker/entrypoint.sh` without `--chown`, so the loader and its parent directories stay root-owned. This deviation from WS-01.6's verbatim brief is approved by the Ruling.
- Modify: `docker/entrypoint.sh`: fail closed when a `*_FILE` secret contains a NUL byte. Insert `if [ "$(tr -d '\000' < "$_path" | wc -c)" -ne "$(wc -c < "$_path")" ]; then _fail "${_name}_FILE contains a NUL byte"; fi` before the read.
- Modify: `scripts/ci/dockerfiles.py`: `uvicorn_flags` starts scanning after the first non-option argument (the app), so a boolean flag before the app is not mis-parsed.
- Test: `tests/build/test_mvp_211_container_entry.py`, extended with three cases.
**Interfaces:**
- Consumes: WS-01.6's entrypoint contract, which is otherwise unchanged.
- Produces: `/app` and `/app/docker` are root-owned in every image; a NUL byte in a secret file exits 1 with a message that names the variable only.

- [ ] **Step 1: Write the failing tests:**
  - (a) in a built image, `stat -c %U /app/docker/entrypoint.sh /app/docker` prints `root` for both;
  - (b) a secret file containing `a\0b\n` makes the entrypoint exit 1, and the message contains `contains a NUL byte` and not the value;
  - (c) `uvicorn_flags(["uvicorn","--proxy-headers","main:app","--port","1"])` treats `--proxy-headers` as a flag, not as an option that takes `main:app` as its value.
- [ ] **Step 2: Run them and confirm they fail.** Expected: (a) prints `appuser`; (b) exits 0 and exports `ab`; (c) returns `['--proxy-headers','main:app']`.
- [ ] **Step 3: Implement** the three edits above.
- [ ] **Step 4: Run to pass**, with `$env:SOLACE_REQUIRE_REALINFRA="1"`.
- [ ] **Step 5: Regression:** tests\build with real infra. Remove the images you build.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-01.F4:head"`.

## Cross-workstream requests

| Id | Owner WS | Exact change | Files | Needed by |
|---|---|---|---|---|
| R-01.1 | WS-03 | W0 phase A2 env sync: compose app services keep `build: {context: ., dockerfile: <path>}` and gain `image: solace/<service>:${SOLACE_IMAGE_TAG:-dev}`; a `migrate` one-shot built from `docker/migrate.Dockerfile`; config-service removed (D-19); app healthchecks call `/live` (or are dropped so the image HEALTHCHECK applies). W4 phase A: every third-party image in `docker-compose.yml` and `docker-compose.prod.yml` (postgres, redis, zookeeper, kafka, weaviate, caddy, prometheus, grafana, jaeger, mailhog, kafka-ui) pinned by `@sha256` digest; the postgres reference equals `scripts.db.images.POSTGRES_IMAGE`. W1 phase A (after WS-06 creates topics at plane start): `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"` on kafka in the strict overlay. | `docker-compose.yml`, `docker-compose.prod.yml` | WS-01.9 (`--compose`), WS-01.10, WS-01.12 |
| R-01.2 | WS-04 | `configure_service_logging` keeps the `uvicorn.error` logger at WARNING or drops records whose message contains a query string (`"WebSocket %s" [accepted]` and `403` lines), so the IF-01 logging chain cannot undo `--log-level warning`; the conformance test asserts no `?token=` in a captured WS handshake log. | `src/solace_infrastructure/runtime/logging.py` | WS-01.6 |
| R-01.3 | WS-05 | (a) W0: replace the `src.solace_infrastructure` imports in `tests/integration/test_persistence_roundtrip.py` with `solace_infrastructure` and run `tests/build/test_import_roots.py -k WS-05` red then green. (b) W1: the crisis E2E asserts that no container log contains `token=` after a WebSocket session (end-to-end proof of MVP-211). (c) W5 (D-17, decided 2026-10-19: no commits): the release gate builds and probes every image from the `git archive` export of the final checkpoint tree id. On the AWS staging host (D-25) that is `scripts/ci/build_images.py --all --tag <tag> --context <expanded archive> --probe`. The gate does not run `make release-gate-closure`, which stays a post-program aid for the owner. | `tests/integration/test_persistence_roundtrip.py`, `tests/e2e/test_crisis_alert_e2e.py`, `tests/e2e/test_release_gate.py` | WS-01.7, WS-01.6, WS-01.9 |
| R-01.4 | WS-06 | (a) W0 phase A2: the seven `event_bridge.py` files import bare `solace_events` (WS-06.2) and run `tests/build/test_import_roots.py -k WS-06` red then green. (b) W1 or W3 phase A: `solace_events.config.TOPIC_CONFIGS` is the single topic registry (solace.safety and solace.users 7-day retention per D-24, replication that fits the single broker), applied by create-if-missing at event-plane start; `TopicManager.register_solace_topics` is deleted or derived from it. | `services/*/src/infrastructure/event_bridge.py`, `src/solace_events/config.py`, `src/solace_infrastructure/kafka/topics.py` | WS-01.7, WS-01.12 |
| R-01.5 | WS-10 | Rewrite `services/orchestrator_service/src/events.py` to bare `solace_events` in WS-10.3 (W1 phase B). Its case in `tests/build/test_import_roots.py` is strict-xfail until then; when WS-10.3 completes, the controller deletes the marker by Ruling (accepted from WS-10's request). | `services/orchestrator_service/src/events.py` | WS-01.7 |
| R-01.6 | WS-12 | W0 phase B adoption task: `services/safety_service/src/events.py` imports bare `solace_events`; run `tests/build/test_import_roots.py -k WS-12` red then green. | `services/safety_service/src/events.py` | WS-01.7 |
| R-01.7 | WS-14 | W0 phase B adoption task: `services/user-service/src/events.py` imports bare `solace_events`; run `-k WS-14` red then green. | `services/user-service/src/events.py` | WS-01.7 |
| R-01.8 | WS-15 | W0 phase B adoption task: `services/therapy_service/src/events.py` imports bare `solace_events`; run `-k WS-15` red then green. | `services/therapy_service/src/events.py` | WS-01.7 |
| R-01.9 | WS-16 | W0 phase B adoption task: `services/diagnosis_service/src/events.py` imports bare `solace_events`; run `-k WS-16` red then green. | `services/diagnosis_service/src/events.py` | WS-01.7 |
| R-01.10 | WS-17 | W0 phase B adoption task: `services/memory_service/src/events.py` imports bare `solace_events`; run `-k WS-17` red then green. | `services/memory_service/src/events.py` | WS-01.7 |
| R-01.11 | WS-13 | W0 phase B adoption task: `events.py` imports bare `solace_events`; the three F821s in `main.py` are fixed (`tests/build/test_mvp_038_lint_gate.py -k WS-13`); intra-service imports are package-relative so `src.main` (the merged image layout) is the only module path, and the notification tests import through `src.` like user-service's. | `services/notification-service/src/events.py`, `services/notification-service/src/main.py`, `services/notification-service/src/api.py` | WS-01.5 (MVP-006 closure), WS-01.7 |
| R-01.12 | WS-18 | W0 phase B adoption task: bare `solace_events` in `services/personality_service/src/events.py` and `services/analytics-service/src/main.py`; the seven analytics F821s fixed (`main.py` forward references, `config.py:219` `timedelta`). | `services/personality_service/src/events.py`, `services/analytics-service/src/main.py`, `services/analytics-service/src/config.py` | WS-01.7 |
| R-01.13 | WS-02 | `migrations/env.py` imports `solace_infrastructure` (import case WS-02, WS-02.2); `src/solace_infrastructure/postgres.py` imports `ssl` for the annotation at line 94 (lint case WS-02, WS-02.5); `solace_infrastructure.database` exports lazily (WS-02.7, supporting MVP-269); `scripts/db/migrate.py` and `schema_check` exist for the migrate image (WS-02.3, WS-02.4). | `migrations/env.py`, `src/solace_infrastructure/postgres.py`, `src/solace_infrastructure/database/__init__.py`, `scripts/db/migrate.py`, `src/solace_infrastructure/database/schema_check.py` | WS-01.7, WS-01.9, WS-01.12 |
| R-01.14 | WS-09 | Confirm or replace the `portkey-ai~=1.9.0` pin (the series `services/shared/infrastructure/llm_client.py` was written against) before WS-01.2 compiles the lock; a different version is a lock request in phase A. | `services/shared/infrastructure/llm_client.py` | WS-01.2 |
| R-01.15 | WS-12 | D-16 keeps torch, transformers, VADER and spaCy out of images, but the local interpreter has them, so safety tests exercise ML branches (`sentiment_analyzer.py`, `pattern_matcher.py`) that images never run. In the W1 lexicon task, make those branches unreachable unless explicitly enabled (default off) so tests and images take the same path. | `services/safety_service/src/ml/sentiment_analyzer.py`, `services/safety_service/src/ml/pattern_matcher.py` | WS-01.2 |
| R-01.16 | WS-03 | Ordering only: WS-03.2 (compose `migrate` service) adds `WS-01.9` and `WS-01.6` to its Depends on, and WS-03.7 adds `WS-01.5` (secret-directory ignores) and `WS-01.6`. The migrate image exists only after WS-02.4 (W0 phase A2, WS-01.9), not in phase A1 as WS-03's request row says. | none (WS-03's plan) | none in WS-01 (ordering for WS-03.2, WS-03.7) |
| R-01.17 | WS-05 | (a) The dev lock carries `testcontainers[postgres,kafka]==4.14.2` without the `redis` extra (it requires redis>=7, against D-16's redis 5.3.1), so WS-05.2's Redis fixture starts `redis` through the generic `testcontainers.core.container.DockerContainer` and the locked client. (b) `pytest.ini` registers `profile`; WS-05's `tests/e2e/conftest.py` owns the deselection. (c) WS-05.2 and WS-05.4 add `WS-01.2` and `WS-01.4` to their Depends on. | none (WS-05's plan) | none in WS-01 (consistency for WS-05.2, WS-05.4) |
| R-01.18 | WS-03 | D-07 conventions that WS-01.14's ignore rules and secret scan rely on, matching WS-03.8 and WS-03.10 as planned. `deploy/secrets/` holds only these files: `<env>.sops.yaml`; the creation-rules file `.sops.yaml` and `escrow-recipients.txt`, both public age recipients only; an optional `README.md`; and previous encrypted versions at `history/<env>/<UTC timestamp>.sops.yaml`. Any other name there is ignored and flagged; a new kind of file needs a WS-01 request first. Each `<env>.sops.yaml` is fully encrypted: no `unencrypted_suffix`, `unencrypted_regex` or `encrypted_regex` that leaves a value in plaintext. `render_secrets.py` decrypts only into tmpfs `/run/secrets`, never into the tree. Tests generate throwaway age keys and synthetic secrets under pytest's `tmp_path`, never in the repository. No file named `*.agekey` or `keys.txt` is planned anywhere in the tree. `tests/build/test_d07_sops_tree_hygiene.py::test_d07_checkpoint_visible_tree_has_no_secret_findings` fails on any deviation. Planned in WS-03.8 (W3), whose Produces line states these conventions and which depends on WS-01.14; WS-01.14 (W0) takes no dependency on it, and its tree scan checks WS-03.8's and WS-03.10's files at every later regression. | `deploy/secrets/*`, `deploy/render_secrets.py`, `tests/deploy/*` | WS-01.14 (consistency; no reverse dependency) |
| R-01.19 | WS-09 | D-04, consistency with WS-09's CR-16. WS-09.18 (the Bedrock Converse provider in `services/shared/infrastructure/llm_client.py`) keeps `WS-01.2` in its Depends on, for the `boto3`/`botocore` runtime pins planned there. If it needs any other distribution (for example a botocore extra), request it as a lock change before WS-01.2 compiles. Its lazy `boto3`/`botocore` imports resolve through `IMPORT_TO_DIST` in `test_mvp_005_runtime_imports_resolve_from_lock`. | none (WS-09's plan) | none in WS-01 (consistency for WS-09.18) |

Requests received and planned here:
- WS-01.2 (lock pins): R-02.1's pin part (WS-02.1, WS-02.4); `boto3` and `botocore` as runtime pins for WS-09's Bedrock provider (D-04), which also cover WS-03's original dev-lock `boto3` request (WS-03.8); WS-05 #1's pins, `jsonschema` added (WS-05.2, WS-05.4); WS-06 #11 (WS-06.4, WS-06.10); WS-10's langgraph 1.0.x, exact langgraph-checkpoint-postgres, psycopg[binary], psycopg-pool, PyYAML (WS-10.1, WS-10.9, WS-10.10; WS-10.9's `WS-01:IF-21-lock` is WS-01.2); WS-04's prometheus-client (WS-04.2); WS-12 R-12-4's PyYAML and hypothesis (WS-12.3); WS-08's pyotp, PyJWT and httpx (WS-08.16's `WS-01:pyotp-lock` is WS-01.2).
- WS-01.3: WS-04's lazy `solace_infrastructure` (WS-04.2); WS-07's and WS-11's lazy `infrastructure.api_gateway` (WS-11.1's `WS-01:lazy-api-gateway` is WS-01.3).
- WS-01.4: WS-05 #1's markers (with `profile`) and Makefile targets; WS-12 R-12-4's `clinical` marker.
- WS-01.5: WS-03's `.secrets`/`secrets` ignores (WS-03.7); WS-03.2's compose image references (`image_ref`, `COMPOSE_TAG`; the change WS-03.2 used to describe against `build_images.py` now lives here); R-14-1's `scripts/` copy in the user-service image (WS-14.9); R-12-4's lexicon package data (WS-12.3).
- WS-01.6: WS-03's entrypoint contract incl. `NAME_FILE` left set (WS-03.2, WS-03.7, WS-04.11); WS-11's `--ws-max-size 16384` (WS-11.17's `WS-01:ws-max-size` is WS-01.6).
- WS-01.7: WS-10's W0 allow-list for orchestrator `events.py`, as the strict-xfail case (WS-10.3).
- WS-01.8: WS-05 #1's on-demand/nightly e2e job (WS-05.4).
- WS-01.9: R-02.1's migrate image (packaging WS-02.4's CLI) and WS-03's migrate image row (WS-03.2).
- WS-01.13: WS-03's W4 cfn-lint/cfn-guard run and config-service-free image matrix (WS-03.11).
- WS-01.2: WS-03's W3 request for hashed PyYAML, cryptography, cffi and pycparser in `requirements.lock`, used by WS-03.8's `deploy/sops/Dockerfile` (D-07). PyYAML and cryptography are direct pins; cffi and pycparser are their locked dependencies, now asserted in `test_mvp_005_required_runtime_packages_are_locked`. WS-03 withdrew its dev-lock `boto3` request under D-07.
- WS-01.14: WS-03's `.dockerignore` exclusion of `deploy/secrets/` (WS-03.8, D-07). It lands in W0 instead of W3, which is harmless because no image needs the encrypted files.
- Not taken: the `redis` extra of WS-05 #1's `testcontainers[postgres,redis,kafka]` (R-01.17a).

## New interfaces

- `scripts/ci/images.py`: `Image`, `IMAGES`, `probe_env()`; `scripts/ci/suites.py`: `Suite`, `SUITES`, `REALINFRA_PATHS`; `scripts/ci/run_suites.py`: `build_command`, `child_env` and the CLI in WS-01.4 (plus `--coverage` in WS-01.11); `scripts/ci/build_images.py`: `build_command(image, tag, context)`, `image_ref(name, tag)`, `COMPOSE_TAG` and the CLI; `scripts/ci/dockerfiles.py`: `parse`, `uvicorn_flags`, `replicate_runtime_layout`, `pythonpath_for`, `dockerignored`; `scripts/ci/iac_lint.py`: `TEMPLATE`, `GUARD_RULES`, `commands`, `run` and the CLI (WS-01.13); `scripts/ci/lock_delta.py`; `scripts/ci/clean_venv.py` (D-20, WS-01.2); `scripts/ci/clean_checkout.py`, `scripts/ci/check_tracked_closure.py`, `scripts/ci/check_clean_checkout.ps1`; `scripts/ci/supply_chain.py` and `accepted_vulns.yaml`; `scripts/ci/coverage_gate.py`; `scripts/ci/ci_local.py`; `scripts/ci/secret_scan.py` (`Finding`, `visible_paths`, `scan`, the CLI; D-07, WS-01.14). All are stdlib-only except `lock_delta.py` (packaging), `supply_chain.py` and `secret_scan.py` (PyYAML), so the CI `plan` job needs no installs.
- IF-21 additions: uvicorn also runs with `--log-level warning`, because the WebSocket accept line that carries `?token=` is logged by `uvicorn.error` at INFO, which `--no-access-log` does not cover; the orchestrator CMD adds `--ws-max-size 16384` (WS-11's request). The entrypoint refuses `NAME` and `NAME_FILE` set together, leaves `NAME_FILE` set after exporting `NAME` (WS-03/WS-04.11 provenance), and leaves `SSL_CERT_FILE` alone. The user-service image also carries `/app/scripts/` (R-14-1). `pytest.ini` registers a fifth marker, `profile(name)` (WS-05's request).
- Make target beyond IF-21 from W4: `iac-lint` (WS-01.13).
- IF-21 deviation: the dev lock uses `testcontainers[postgres,kafka]==4.14.2` without the `redis` extra, which requires redis>=7 (`Requires-Dist: redis>=7; extra == 'redis'` in the installed testcontainers-4.14.2 METADATA) and conflicts with D-16's redis 5.3.x. WS-05's redis fixture uses the generic container with the locked client (R-01.17).
- Dev-lock additions by request: `jsonschema==4.26.0` and `cfn-lint==1.51.5`. Anaconda already holds both, and the venv sees them, so neither adds to the venv sync list. Runtime-lock addition (D-04): `boto3` and `botocore` at the anaconda-installed versions (1.40.61 on 2026-10-19, re-read with `pip show` at compile time; an advisory may force a bump), for WS-09's Bedrock provider. `langgraph-checkpoint-postgres` is pinned with `==` in `requirements.in` so a bump is deliberate and travels with a slot-017 follow-up revision (WS-10).
- `requirements.txt` is a pointer to `requirements-dev.lock`; `pyproject.toml` declares no dependency lists. `python-jose[cryptography]==3.5.0` lives only in `requirements-dev.lock`, for the frozen Kong tests.
- Make targets beyond IF-21: `test` (alias), `migrate`, `release-gate-closure`, `supply-chain` (W4), `coverage` (W5).
- Test case ids other workstreams run: `tests/build/test_import_roots.py::test_mvp_270_no_src_root_imports[WS-NN]` and `tests/build/test_mvp_038_lint_gate.py::test_mvp_038_lint_gate_clean[WS-NN]`.
- `.gitignore` anchoring (WS-01.1) un-ignores WS-03's `deploy/env/`, WS-10's `checkpointer.py` and this workstream's `tests/build/`.
- D-07 tree rules (WS-01.14):
  - These are visible to checkpoint trees: `deploy/secrets/*.sops.yaml` (including `.sops.yaml`), `deploy/secrets/escrow-recipients.txt`, `deploy/secrets/README.md` and `deploy/secrets/history/**/*.sops.yaml`. Everything else under `deploy/secrets/` is ignored, as are `*.agekey` and `keys.txt`.
  - `.dockerignore` keeps `deploy/secrets`, `**/*.agekey` and `**/keys.txt` out of every build context.
  - `secret_scan.py` checks every checkpoint-visible file.
- D-20 (WS-01.2): two venvs.
  - `D:\tmp\mvp-exec\venv`, created by the controller at W0 entry with system site packages, is re-synced to the dev lock.
  - `D:\tmp\mvp-exec\venv-clean`, lock-only, runs the suites at the W0 barrier.

## Self-review

Coverage (register id -> tasks):

| Id | Sev | Tasks |
|---|---|---|
| MVP-005 | blocker | WS-01.2 |
| MVP-006 | blocker | WS-01.5 (image part; R-01.11 for WS-13's import normalization) |
| MVP-008 | blocker | WS-01.1, WS-01.9 (closes on the clean-checkout build of a checkpoint tree exported with `git archive`; no commit, D-17), WS-01.14 (D-07 paths stay visible) |
| MVP-012 | blocker | WS-01.2 (lock), WS-01.9 (in-image probe of `src.api`) |
| MVP-013 | blocker | WS-01.5 (layout and replica), WS-01.9 (in-image probe) |
| MVP-032 | blocker | WS-01.5, WS-01.9 |
| MVP-038 | high | WS-01.7 (lint gate and owners), WS-01.8 |
| MVP-358 | high | WS-01.2 |
| MVP-112 | medium | WS-01.5 (`build_command`), WS-01.8 |
| MVP-113 | medium | WS-01.10 |
| MVP-211 | medium | WS-01.6 (plus R-01.2, R-01.3b) |
| MVP-243 | low | WS-01.11 |
| MVP-244 | low | WS-01.4 |
| MVP-245 | low | WS-01.4 |
| MVP-246 | low | WS-01.8 |
| MVP-248 | low | WS-01.5 |
| MVP-269 | low | WS-01.3, WS-01.12 (plus WS-02.7 and R-01.4) |
| MVP-270 | low | WS-01.7 (owner cases via R-01.3 to R-01.13) |
| MVP-310 | low | WS-01.3 |
| MVP-312 | low | WS-01.2 (runtime lock), WS-01.3 (import decoupling) |
| MVP-355 | low | WS-01.5 (Dockerfiles), WS-01.10 (compose, R-01.1) |
| MVP-359 | low | WS-01.2 |
| MVP-043, MVP-044 (WS-03's) | high | WS-01.13, supporting (CI run of WS-03.11's checks) |
| MVP-348 (WS-03's) | blocker | WS-01.14, supporting (D-07 tree hygiene: SOPS files visible, age keys and plaintext ignored and scanned; R-01.18) |

Interface consistency: IF-21 is produced as written, apart from the additions and the one deviation listed under New interfaces. IF-22's NAME_FILE rule is implemented by `docker/entrypoint.sh`. IF-19's markers (realinfra, e2e, contract, clinical) are registered in WS-01.4, plus WS-05's `profile`, and `SOLACE_REQUIRE_REALINFRA=1` turns skips into failures in `tests/build/conftest.py`. IF-02 is consumed only by `probe_env()`, which uses throwaway values and never reads `.env`. The skeleton's `python -m scripts.db.migrate upgrade head` is the migrate CMD, matching WS-02.4. WS-02's paired requests are satisfied by WS-01.2 (alembic, sqlalchemy, asyncpg, psycopg, cryptography, testcontainers, docker pins) and WS-01.9 (the migrate image).

Open risks:
- The controller's W0-entry venv install (D-20) makes aiokafka and langgraph-checkpoint-postgres importable for the first time, and WS-01.2's re-sync adds portkey-ai. Code that silently used mock modes may start dialing `localhost:9092` in unit tests. WS-01.2 Step 5 records any such failure for its owner, and the controller re-records the baseline.
- The venv sees anaconda's packages through its system site packages, so a package the lock forgot can still pass locally. The lock-only run in `D:\tmp\mvp-exec\venv-clean` at the W0 barrier catches this, and the images catch it again. Building that venv downloads every locked wheel and builds the `flashtext` sdist, so it needs network access to the package index.
- 172 settings classes read `env_file=".env"` relative to the working directory, so local runs may depend on the developer's real `.env`, which CI does not have. WS-01.8's hermetic run surfaces this, but fixing each suite belongs to its owner.
- `.gitignore` used to hide `tests/build/`, `deploy/env/` and `checkpointer.py`. Any task that ran before WS-01.1 and created files there would be missing from its checkpoint. WS-01.1 is therefore the first task in W0, and the controller verifies the first checkpoint lists `tests/build/*`.
- `flashtext==2.7` ships only an sdist. The builder stage builds it under `--require-hashes` with unhashed build dependencies (setuptools). If uv refuses, the fallback is to add `setuptools` to `requirements.in` and use `--no-build-isolation-package flashtext`, recorded as a Ruling.
- Nothing is committed or pushed during the program (D-17), so the CI workflow is proven only statically and by `make ci-local`. If the owner decides after the program to commit and push, the first real GitHub run may expose runner differences (Docker-in-Docker for testcontainers, pwsh availability).
- MVP-008 closes on the clean-checkout build of a checkpoint tree (`git archive <tree id>`), at the W0 barrier and again in the W5 gate (R-01.3c). Untracked dependencies of HEAD-tracked files remain for the whole program. `check_tracked_closure.py --report` lists them for the owner's post-program commit decision, and nothing gates on it.
- Pre-commit hooks never run, because nobody commits (D-17). The only automatic secret check is `tests/build/test_d07_sops_tree_hygiene.py` (WS-01.14), which runs in every `tests/build` regression, at barriers and in `make ci-local`. A key file created between two of those runs could enter a checkpoint tree object. The ignore rules are the first line of defence; running `secret_scan.py` before each checkpoint is a controller-protocol option (an open question for the controller).
- The user-service image copies all of `scripts/` (R-14-1), so `scripts/ci` and `scripts/db` ship in that image too. They hold no secrets; narrowing the COPY to `scripts/provision_clinician.py` is possible only once WS-14 creates the file in W1 (a later request).

Plan-checker notes: `.dockerignore` patterns (WS-01.5) and the migrate image's COPY sources (WS-01.9) name other workstreams' paths, but WS-01 edits none of them; the Files lines now describe them without naming them as edited paths. `alembic.ini` is edited in W0 only by WS-02.2; WS-01.9 copies it into the image after WS-02.4, which follows WS-02.2.

REVIEW_FOCUS: 4=WS-01.6; 5=WS-01.6

## Review responses

- Error: WS-01.5 edits `infrastructure/aws-staging` (WS-03's): rejected: misparse. The path was a `.dockerignore` exclusion pattern in WS-01.5's Files line; WS-01.5 edits only `.dockerignore` and the nine Dockerfiles. The Files line now points at the pattern list in Step 3.
- Error: WS-01.9 edits `alembic.ini` (WS-02's): rejected: misparse. It was a COPY source of `docker/migrate.Dockerfile`; WS-01.9 never writes it. The Create line no longer names it, and the layout moved to Produces.
- Error: WS-01.9 edits `scripts/db` (WS-02's): rejected: same misparse (COPY source only); reworded the same way.
- Error: plan WS-02's request for WS-02.1 (pins plus the migrate image): fixed. WS-01.2's Produces names R-02.1's pins for WS-02.1 and WS-02.4. WS-01.9's Produces names the migrate image as R-02.1's image part. That image packages WS-02.4's CLI, so it comes after WS-02.4 and is not a WS-02.4 prerequisite.
- Error: plan WS-03's `boto3` request for WS-03.8: fixed. `boto3==1.40.61` is added to `requirements-dev.in` (WS-01.2 Step 3.4), and the dev-lock test asserts it.
- Error: plan WS-05's request (markers, pins, Makefile, CI e2e): fixed. WS-01.4 now also registers `profile` (test updated), and its Produces names WS-05.2 and WS-05.4. WS-01.2 adds `jsonschema==4.26.0`; the other pins were already there. WS-01.8's e2e job is named. The `redis` extra of `testcontainers[postgres,redis,kafka]` is rejected, because testcontainers-4.14.2's METADATA has `Requires-Dist: redis>=7; extra == 'redis'`, which conflicts with D-16's redis 5.3.1. WS-05 is told to use the generic container (R-01.17).
- Error: plan WS-06's request for WS-06.4 (aiokafka, prometheus-client, D-20 install): fixed. It was already in WS-01.2 Steps 3.3 and 3.6; WS-01.2's Produces now names WS-06.4 and WS-06.10.
- Error: plan WS-10's import-root allow-list for WS-10.3: fixed. It already existed as the strict-xfail WS-10 case in `tests/build/test_import_roots.py`; WS-01.7's Produces now names WS-10.3.
- Error: plan WS-10's lock request for WS-10.1: fixed. WS-01.2's Produces names WS-10.1, WS-10.9 and WS-10.10. `langgraph-checkpoint-postgres` is now an exact `==` pin with a slot-017 comment, and the new test `test_mvp_358_checkpoint_postgres_is_exact_pinned_for_slot_017` makes a bump deliberate.
- Warning: WS-01.5 edits `src/models` (unowned): rejected: misparse. It is a `.dockerignore` pattern, not an edited path, and the Files line is reworded.
- Warning: W0 `alembic.ini` edited by WS-01.9 and WS-02.2: rejected. WS-01.9 only copies the file into the image (now worded that way). WS-02.2 is its only W0 editor, and WS-01.9 runs after WS-02.4, which follows WS-02.2 (Self-review plan-checker note).
- Warning: WS-01.9 lists `alembic.ini` under Create: rejected: same misparse. The Create line now names only `docker/migrate.Dockerfile`.
- Reconciliation 1 (`scripts/ci/build_images.py` named by WS-03.2 in W0): fixed. The image-reference agreement WS-03.2 described against the builder now lives in WS-01.5, which creates the file: new `image_ref(name, tag)` (used by `build_command`) and `COMPOSE_TAG = "${SOLACE_IMAGE_TAG:-dev}"`, pinned by `test_mvp_112_compose_image_refs_come_from_the_builder`. WS-01.5's Produces line names WS-03.2. WS-03.2 no longer lists the file, depends on WS-01.5 (and on WS-01.6 and WS-01.9, as R-01.16 asked), and asserts its compose `image:` values with `image_ref`. WS-01.9's Produces line now names the migrate image by its real reference `solace/migrate:<tag>` instead of `solace-migrate`.
- Reconciliation 2 (WS-10's allow-list request for `tests/build/test_import_roots.py`): planned, not declined. WS-01.7 already carried the strict-xfail WS-10 case, but listed the guard files only on its `Test:` line, which the checker does not count as a creation. WS-01.7 now lists both guard files under Create, its Produces line already names WS-10.3, and WS-10.3 now depends on WS-01.7.
- decision 2026-10-19 D-17: "the owner commits at barriers" is gone; nothing is committed, and checks build from `git archive <checkpoint tree id>`. WS-01.1: decision gate, the Why paragraph, the test docstring and the assert message. WS-01.8: decision gate (nothing pushed; the first GitHub run can only follow the owner's post-program decision). WS-01.9: decision gate and Step 3. MVP-008 closes on the clean-checkout build of a checkpoint tree; the tracked-closure report and `make release-gate-closure` gate nothing. Also R-01.3(c) (the W5 gate builds images from the archive instead of running `release-gate-closure`), the MVP-008 coverage row and two open risks.
- decision 2026-10-19 D-20: the anaconda install is gone and replaced by the dedicated venv. In WS-01.2: the decision gate describes the controller's W0-entry venv (system site packages), its installs and the per-suite baseline re-record; the new file `scripts/ci/clean_venv.py` appears in Files, Produces and the File map; Consumes names the W0 venv as the baseline; `LOCAL_MISSING_OK_ROOTS` is now empty, because the venv is synced to the full dev lock (portkey-ai included); the new test `test_d20_clean_venv_is_built_only_from_the_dev_lock` has its Step 2 RED; Step 3 item 1 (first-on-`sys.path` wins) and item 5 are reworded, item 6 is rewritten as the venv re-sync with freezes before and after (no `--allow-missing`, never pip against anaconda; anaconda's `uv.exe` is fine), and item 8 is new; Step 5 adds the clean-venv smoke run and the W0-barrier run of `run_suites.py` in `D:\tmp\mvp-exec\venv-clean`, where failures are lock defects. Also WS-01.9 (`check_clean_checkout.ps1` uses the venv interpreter), New interfaces and the open risks.
- decision 2026-10-19 D-04: `boto3` and `botocore` are runtime-lock pins at the anaconda-installed version (`pip show`; 1.40.61 for both on 2026-10-19) unless an advisory forces a bump. In WS-01.2: the decision gate and Produces name WS-09's Bedrock provider, replacing WS-03.8's dev-lock `boto3` request (its `render_secrets.py` no longer uses Secrets Manager under D-07); `IMPORT_TO_DIST`, the required-runtime list and the new test `test_d04_bedrock_sdk_is_an_exact_runtime_pin`; Step 2; Step 3 item 3 (the requirements.in lines) and item 4 (`boto3` removed from requirements-dev.in). Also New interfaces, requests received, and the new R-01.19 to WS-09.
- decision 2026-10-19 D-07: SOPS + age replaces AWS Secrets Manager. The WS-01.6 decision gate says so (the NAME_FILE contract is unchanged). New task WS-01.14 (W0 phase A1): `.gitignore` makes WS-03's planned `deploy/secrets/` files visible to checkpoint trees, which the unanchored `secrets/` line used to hide. Those files are `<env>.sops.yaml`, `.sops.yaml`, `escrow-recipients.txt`, `README.md` and `history/<env>/*.sops.yaml`. It ignores age keys (`*.agekey`, `keys.txt`) and every other file under `deploy/secrets/`. `.dockerignore` keeps `deploy/secrets` and age keys out of every build context. The new `scripts/ci/secret_scan.py` catches age private keys, stray or decrypted files and SOPS files with plaintext values in every checkpoint-visible file. Also the Waves line, the File map, New interfaces, the MVP-348 supporting coverage row, an open risk, and the new R-01.18 to WS-03 for the file conventions (matching WS-03's planned `escrow-recipients.txt` and `history/<env>/` files). Requests received records WS-03's `deploy/secrets/` `.dockerignore` row and its D-07 lock row; for the latter, `cffi` and `pycparser` are added to WS-01.2's required-runtime assertion.
- decision 2026-10-19 D-25: the WS-01.13 decision gate now says AWS hosts the release gate and the pilot through `MedCodex-Local`. Provisioning is blocked until the owner attaches the deployer policy, so this task only lints. The W5 gate runs on the staging host via SSM with the strict production profile (WS-05.19); an unattached policy at W5 entry is a stop, never a fallback to local; real-infra tests run on local Docker until then.
