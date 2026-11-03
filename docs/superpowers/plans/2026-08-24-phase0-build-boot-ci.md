# Phase 0 — Foundation: Build, Boot & CI Truth — Implementation Plan

**Goal:** Make all 10 service images build from the correct context, every service boot on real infrastructure with migrations auto-applied, and CI actually build→boot→migrate→health-probe the artifact it ships — so green CI proves the deployable stack works.

**Architecture:** Ten Python 3.12 / FastAPI services build from the **repo root** context, layering a per-service `requirements.txt` on a shared root `requirements-runtime.txt` (heavy ML split into `requirements-ml.txt`). Schema/RLS/audit/outbox/DLQ tables are provisioned by a single `alembic upgrade head` run by an init container (compose) and a CI job against a real Postgres service, replacing the current app-startup `ensure_table()` ad-hoc DDL as the source of truth. CI gains a repo-root docker build + `docker run` + `/health` probe gate and a blocking `pip-audit`/Trivy supply-chain gate.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, SQLAlchemy[asyncio] + asyncpg, Alembic, Redis, aiokafka, Weaviate, Docker + Docker Compose, GitHub Actions, uv/pip-tools (lockfile), pip-audit + Trivy (supply chain), pytest / pytest-asyncio.

**Spec:** `docs/ENTERPRISE-READINESS-ASSESSMENT.md` (BLK-01…06, BLK-20, H-01, H-02) and `docs/ENTERPRISE-RELEASE-PLAN.md` (Phase 0 objective + exit criteria).

## Global Constraints

- **TDD, always.** Every task: write the failing test (RED) → run it and see it FAIL → minimal implementation → run it and see it PASS → run the affected suite as the regression gate.
- **Prove on real infrastructure.** Build/boot/migrate items are not "done" until verified against real Postgres/Redis/Kafka via `docker compose` and a real `alembic upgrade head` — not a mock or SQLite. Unit/alignment tests are the fast RED/GREEN gate; a `# REAL-INFRA` step in each infra task states the exact `docker`/`alembic` command that must pass on a Docker host before the item counts as closed.
- **Fail-loud in production.** Any boot-time control (config secrets, migrations, infrastructure imports) must raise, not warn, when `ENVIRONMENT=production` / `*_ENV=production` and the control is absent or default.
- **No silent crisis-path drops.** Nothing in Phase 0 may weaken at-least-once delivery of safety/crisis events; the audit/outbox/DLQ table move to Alembic must preserve the exact DDL those tables already use (`src/solace_events/postgres_stores.py`, `src/solace_security/audit.py`).
- **NO git commits.** This project forbids commits. Each task ends by running the affected pytest suite (the regression gate), NOT `git commit`.
- **Test command (Windows PowerShell; the Bash-tool hook breaks on `;`):**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q
  ```
  (`PYTHONPATH` MUST include `src/`. `-o "addopts="` neutralizes the repo `pytest.ini`/`pyproject` addopts.)

---

## File Structure

| File | Create / Modify | Responsibility |
|---|---|---|
| `requirements-runtime.txt` | Create | Shared runtime dependency manifest COPY'd by all 10 Dockerfiles (BLK-01). |
| `requirements-ml.txt` | Create | Heavy/optional ML deps (torch, transformers) installed only where needed (BLK-01). |
| `requirements-runtime.lock` | Create | Single hash-pinned lockfile compiled from the runtime manifest (H-02). |
| `tests/alignment/test_phase0_build_boot.py` | Create | Alignment tests: manifests exist & layer correctly, Dockerfiles COPY `infrastructure/`, compose env overridable, CI has build-run-health + scan gates. |
| `docker-compose.yml` | Modify (:206) | Make `USER_SERVICE_ENV` overridable; add `db-migrate` init service; gate app services on migrate completion (BLK-02, BLK-05). |
| `docker-compose.prod.yml` | Modify | Same `db-migrate` init service for prod (BLK-05). |
| `services/user-service/Dockerfile` | Modify (after :33) | `COPY infrastructure/` so the rate-limiter import resolves at runtime (BLK-03). |
| `services/orchestrator_service/Dockerfile` | Modify (after :25) | `COPY infrastructure/` (BLK-03). |
| `services/analytics-service/src/main.py` | Modify (:287-288) | Serialize Kafka events with `model_dump(mode="json")` via an extracted, testable helper (BLK-20). |
| `services/config_service/src/settings.py` | Modify (:360-362, :64-87) | Ingest DB/JWT secrets from env so `_validate_production_secrets()` passes in prod (BLK-06). |
| `services/config_service/config/config.production.json` | Create | Non-secret production config baseline loaded by the config manager (BLK-06). |
| `migrations/versions/009_audit_outbox_dlq_tables.py` | Create | Move `audit_events`, `event_outbox`, `event_dead_letter`, `processed_events` DDL into Alembic (BLK-05). |
| `tests/migrations/test_009_audit_outbox_dlq.py` | Create | Prove migration 009 chains from head and creates the four tables on real Postgres. |
| `.github/workflows/ci.yml` | Modify (:148-185, new jobs) | Repo-root docker build + `docker run` + `/health` probe gate; `db-migrate` job; `pip-audit`/Trivy blocking gate (BLK-04, BLK-05, H-01, H-02). |
| `docs/STAGING-BRINGUP.md` | Modify | Note the `USER_SERVICE_ENV=staging` requirement + `db-migrate` init step (BLK-02, BLK-05). |

---

### Task 1: BLK-01 — Root `requirements-runtime.txt` + `requirements-ml.txt`

Every Dockerfile does `COPY requirements-runtime.txt` (e.g. `services/user-service/Dockerfile:16`) but the file exists nowhere → all 10 builds fail. Root `requirements.txt` is the source; the runtime manifest is its non-test, non-heavy-ML subset. Per-service `requirements.txt` files layer on top (verified: `orchestrator_service/requirements.txt` adds langgraph pins; `safety_service`/`personality_service` reqs are comment-only and reference `requirements-ml.txt`).

**Files:**
- Create `requirements-runtime.txt`
- Create `requirements-ml.txt`
- Create `tests/alignment/test_phase0_build_boot.py`

**Interfaces:**
- Consumes: `requirements.txt` (source of truth), each `services/*/requirements.txt` (layered pins), each `services/*/Dockerfile` (`COPY requirements-runtime.txt`).
- Produces: `requirements-runtime.txt` (runtime subset), `requirements-ml.txt` (heavy ML).

**Steps:**

- [ ] Write the failing alignment test. Create `tests/alignment/test_phase0_build_boot.py`:
  ```python
  """Phase 0 alignment tests: build/boot/CI manifests are present and consistent."""
  from pathlib import Path

  ROOT = Path(__file__).resolve().parent.parent.parent

  SERVICE_DIRS = [
      "services/safety_service", "services/diagnosis_service", "services/therapy_service",
      "services/memory_service", "services/personality_service", "services/orchestrator_service",
      "services/notification-service", "services/user-service", "services/analytics-service",
      "services/config_service",
  ]


  class TestRuntimeManifests:
      def test_requirements_runtime_exists(self) -> None:
          f = ROOT / "requirements-runtime.txt"
          assert f.exists(), "BLK-01: requirements-runtime.txt COPY'd by all Dockerfiles must exist"

      def test_requirements_ml_exists(self) -> None:
          assert (ROOT / "requirements-ml.txt").exists(), "BLK-01: requirements-ml.txt must exist"

      def test_runtime_manifest_has_core_runtime_deps(self) -> None:
          text = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").lower()
          for pkg in ("fastapi", "uvicorn", "pydantic", "sqlalchemy", "asyncpg",
                      "redis", "aiokafka", "httpx", "cryptography", "pyjwt",
                      "structlog", "alembic"):
              assert pkg in text, f"BLK-01: runtime manifest missing {pkg}"

      def test_runtime_manifest_excludes_test_and_heavy_ml(self) -> None:
          text = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").lower()
          for pkg in ("pytest", "ruff", "mypy", "torch", "transformers"):
              assert pkg not in text, f"BLK-01: {pkg} must not be in the runtime image manifest"

      def test_ml_manifest_has_heavy_deps(self) -> None:
          text = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8").lower()
          assert "torch" in text and "transformers" in text, "BLK-01: ML manifest must carry torch+transformers"

      def test_every_dockerfile_copies_runtime_manifest(self) -> None:
          missing = []
          for d in SERVICE_DIRS:
              df = (ROOT / d / "Dockerfile").read_text(encoding="utf-8")
              if "COPY requirements-runtime.txt" not in df:
                  missing.append(d)
          assert not missing, f"BLK-01: Dockerfiles not COPYing runtime manifest: {missing}"
  ```

- [ ] Run it — expect FAIL (`requirements-runtime.txt` does not exist):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestRuntimeManifests -o "addopts=" -p no:logging -q
  ```
  Expected: `test_requirements_runtime_exists` and dependents FAIL.

- [ ] Create `requirements-runtime.txt` (the runtime subset of `requirements.txt` — core framework, AI/LLM, DB/ORM, vector, cache/messaging, search, HTTP, security, observability, numpy, utilities; NO testing/code-quality/heavy-ML):
  ```text
  # Solace-AI shared RUNTIME manifest — COPY'd by every service Dockerfile.
  # Source of truth: requirements.txt (this is its non-test, non-heavy-ML subset).
  # Heavy ML (torch/transformers) lives in requirements-ml.txt. Per-service
  # requirements.txt layer additional/pinned deps on top of this.
  # Python 3.12+.

  # ---- Core framework ----
  fastapi~=0.128.0
  uvicorn[standard]~=0.34.0
  pydantic~=2.10.0
  pydantic-settings~=2.7.0
  python-multipart~=0.0.17
  python-dotenv~=1.0.1

  # ---- AI & LLM framework ----
  langgraph~=0.2.76
  langgraph-checkpoint-postgres~=2.0.0
  langchain~=0.3.14
  langchain-core~=0.3.30
  langchain-community~=0.3.14
  langchain-text-splitters~=0.3.5
  langchain-anthropic~=0.3.3
  anthropic~=0.42.0
  langchain-openai~=0.3.0
  openai~=1.58.1
  portkey-ai~=1.9.0

  # ---- Database & ORM ----
  sqlalchemy[asyncio]~=2.0.0
  asyncpg~=0.30.0
  alembic~=1.14.0

  # ---- Vector database ----
  weaviate-client~=4.10.0

  # ---- Cache & messaging ----
  redis~=5.2.0
  aiokafka~=0.12.0

  # ---- Search & retrieval ----
  bm25s~=0.2.0

  # ---- HTTP client ----
  httpx[http2]~=0.28.0
  aiohttp~=3.11.0

  # ---- Authentication & security ----
  PyJWT~=2.9.0
  python-jose[cryptography]~=3.3.0
  passlib[bcrypt]~=1.7.4
  bcrypt~=4.2.0
  cryptography>=42.0.0

  # ---- Observability ----
  structlog~=25.1.0
  orjson~=3.10.0
  prometheus-client~=0.22.0
  prometheus-fastapi-instrumentator~=7.1.0
  opentelemetry-api~=1.29.0
  opentelemetry-sdk~=1.29.0
  opentelemetry-exporter-otlp~=1.29.0
  opentelemetry-instrumentation-fastapi~=0.50b0
  opentelemetry-instrumentation-sqlalchemy~=0.50b0
  opentelemetry-instrumentation-redis~=0.50b0
  opentelemetry-instrumentation-httpx~=0.50b0

  # ---- Numerics (light) ----
  numpy~=2.2.0

  # ---- Utilities ----
  tenacity~=9.0.0
  PyYAML~=6.0.2
  typing-extensions~=4.12.0
  ```

- [ ] Create `requirements-ml.txt` (heavy/optional ML — installed only in images that need the RoBERTa/sentiment heads; imports are guarded per `safety_service`/`personality_service` reqs comments):
  ```text
  # Solace-AI heavy/optional ML manifest (~2GB). Layered ONLY on services whose
  # ML heads are active (safety sentiment, personality RoBERTa). Imports are
  # guarded so absence degrades gracefully. Install: pip install -r requirements-ml.txt
  # (requires requirements-runtime.txt already installed for numpy).
  torch~=2.5.0
  transformers~=4.48.0
  sentence-transformers~=3.3.0
  ```

- [ ] Run the test again — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestRuntimeManifests -o "addopts=" -p no:logging -q
  ```
  Expected: 6 passed.

- [ ] **Regression gate** — run the full alignment suite:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ -o "addopts=" -p no:logging -q
  ```
  Expected: all green (existing `test_config_consistency.py` + new file).

---

### Task 2: H-02 — Single hash-pinned lockfile (`requirements-runtime.lock`)

Three divergent dependency definitions today (pyproject floors, root `~=`, per-service pins). Add ONE lockfile compiled from `requirements-runtime.txt` with hashes, and a test that enforces its presence + hash-pinning. Compilation is a CI/manual step (`uv pip compile --generate-hashes`); the test validates the committed artifact.

**Files:**
- Create `requirements-runtime.lock`
- Modify `tests/alignment/test_phase0_build_boot.py` (add `TestLockfile`)

**Interfaces:**
- Consumes: `requirements-runtime.txt`.
- Produces: `requirements-runtime.lock` (fully pinned, `--hash=sha256:...` per package).

**Steps:**

- [ ] Add the failing test to `tests/alignment/test_phase0_build_boot.py`:
  ```python
  class TestLockfile:
      def test_lockfile_exists(self) -> None:
          assert (ROOT / "requirements-runtime.lock").exists(), "H-02: single hash-pinned lockfile must exist"

      def test_lockfile_is_hash_pinned(self) -> None:
          text = (ROOT / "requirements-runtime.lock").read_text(encoding="utf-8")
          assert "--hash=sha256:" in text, "H-02: lockfile must be generated with --generate-hashes"
          # Every non-comment requirement line must carry an exact '==' pin.
          req_lines = [
              ln.strip() for ln in text.splitlines()
              if ln.strip() and not ln.lstrip().startswith(("#", "--hash"))
              and not ln.strip().endswith("\\") is False and "==" in ln or (
                  ln.strip() and not ln.lstrip().startswith(("#", "--hash")))
          ]
          # Simpler invariant: no fuzzy specifiers survive in the lock.
          for ln in text.splitlines():
              s = ln.strip()
              if not s or s.startswith(("#", "--hash", "--")):
                  continue
              if s.endswith("\\"):
                  s = s[:-1].strip()
              if not s:
                  continue
              assert "~=" not in s and ">=" not in s, f"H-02: fuzzy specifier survived in lock: {s!r}"
  ```

- [ ] Run it — expect FAIL (lockfile absent):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestLockfile -o "addopts=" -p no:logging -q
  ```

- [ ] Generate the lockfile with hashes (run on a machine with network; `uv` is already in the Dockerfiles). This is the impl step — produce `requirements-runtime.lock`:
  ```powershell
  & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pip install uv
  uv pip compile requirements-runtime.txt --generate-hashes -o requirements-runtime.lock
  ```
  (Fallback if `uv` unavailable: `pip install pip-tools; pip-compile --generate-hashes --output-file requirements-runtime.lock requirements-runtime.txt`.) The resulting file has one `pkg==X.Y.Z` per line, each followed by `--hash=sha256:...` continuation lines.

- [ ] Run the test again — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestLockfile -o "addopts=" -p no:logging -q
  ```

- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ -o "addopts=" -p no:logging -q
  ```

---

### Task 3: BLK-02 — `USER_SERVICE_ENV` overridable in compose

`docker-compose.yml:206` sets `USER_SERVICE_ENV: development` as a bare literal, which wins over `env_file` and pins the service to development. Under staging, the REV-32 startup guard (`ProductionGuards.assert_startup`) expects `USER_SERVICE_ENV=staging`; the bare literal makes it impossible to set, and the analytics service already uses the fixed pattern (`ANALYTICS_SERVICE_ENV: ${ANALYTICS_SERVICE_ENV:-development}` at compose :487).

**Files:**
- Modify `docker-compose.yml` (:206)
- Modify `tests/alignment/test_phase0_build_boot.py` (add `TestComposeEnvOverridable`)
- Modify `docs/STAGING-BRINGUP.md` (add note)

**Interfaces:**
- Consumes: `${USER_SERVICE_ENV:-development}` from the deploy environment / `.env.prod`.
- Produces: overridable `USER_SERVICE_ENV` on the `user-service` container.

**Steps:**

- [ ] Add the failing test:
  ```python
  import re

  class TestComposeEnvOverridable:
      def test_user_service_env_is_overridable(self) -> None:
          text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
          # Must NOT contain the bare literal; MUST use the ${VAR:-default} form.
          assert not re.search(r"^\s*USER_SERVICE_ENV:\s*development\s*$", text, re.M), \
              "BLK-02: USER_SERVICE_ENV must not be a non-overridable literal"
          assert "USER_SERVICE_ENV: ${USER_SERVICE_ENV:-development}" in text, \
              "BLK-02: USER_SERVICE_ENV must use the ${USER_SERVICE_ENV:-development} form"
  ```

- [ ] Run it — expect FAIL:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestComposeEnvOverridable -o "addopts=" -p no:logging -q
  ```

- [ ] Edit `docker-compose.yml:206`, replacing the literal:
  ```yaml
      environment:
        # BLK-02: overridable so .env.prod / staging can set the REV-32 startup gate value.
        USER_SERVICE_ENV: ${USER_SERVICE_ENV:-development}
  ```

- [ ] Add the note to `docs/STAGING-BRINGUP.md` (near the env-setup section):
  ```markdown
  - **BLK-02:** set `USER_SERVICE_ENV=staging` in `.env.prod` (compose now reads
    `${USER_SERVICE_ENV:-development}`). The REV-32 `ProductionGuards.assert_startup`
    guard fails-loud unless this is `staging`/`production` AND the DB/JWT/encryption
    secrets are non-default with `POSTGRES_SSL_MODE` != `disable`/`allow`.
  ```

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestComposeEnvOverridable -o "addopts=" -p no:logging -q
  ```

- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ -o "addopts=" -p no:logging -q
  ```

---

### Task 4: BLK-03 — `COPY infrastructure/` into orchestrator + user Dockerfiles

Both services import the repo-root `infrastructure/` package at runtime: `orchestrator_service/src/main.py:334` and `user-service/src/main.py:496` do `from infrastructure.api_gateway.rate_limiting import create_solace_rate_limiter`. Neither Dockerfile COPYs `infrastructure/` (confirmed: `grep "COPY.*infrastructure" services/*/Dockerfile` → none), so rate limiting is silently off and — under literal `production` — the import failure hard-crashes boot. `infrastructure/` has an `__init__.py`, so it imports as a top-level package; `PYTHONPATH=/app` (already set) makes `/app/infrastructure` importable.

**Files:**
- Modify `services/orchestrator_service/Dockerfile` (after :25)
- Modify `services/user-service/Dockerfile` (after :33)
- Modify `tests/alignment/test_phase0_build_boot.py` (add `TestInfraCopied`)

**Interfaces:**
- Consumes: repo-root `infrastructure/` (has `__init__.py`; `api_gateway/rate_limiting.py`).
- Produces: `/app/infrastructure/` present in both runtime images, importable via `PYTHONPATH=/app`.

**Steps:**

- [ ] Add the failing test:
  ```python
  class TestInfraCopied:
      def test_orchestrator_dockerfile_copies_infrastructure(self) -> None:
          df = (ROOT / "services/orchestrator_service/Dockerfile").read_text(encoding="utf-8")
          assert "COPY --chown=appuser:appuser infrastructure/ ./infrastructure/" in df, \
              "BLK-03: orchestrator image must COPY infrastructure/ (rate-limiter import)"

      def test_user_service_dockerfile_copies_infrastructure(self) -> None:
          df = (ROOT / "services/user-service/Dockerfile").read_text(encoding="utf-8")
          assert "COPY --chown=appuser:appuser infrastructure/ ./infrastructure/" in df, \
              "BLK-03: user-service image must COPY infrastructure/ (rate-limiter import)"

      def test_infrastructure_is_importable_package(self) -> None:
          assert (ROOT / "infrastructure" / "__init__.py").exists()
          assert (ROOT / "infrastructure" / "api_gateway" / "rate_limiting.py").exists()
  ```

- [ ] Run it — expect FAIL:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestInfraCopied -o "addopts=" -p no:logging -q
  ```

- [ ] Edit `services/orchestrator_service/Dockerfile` — add after line 25 (`COPY ... services/orchestrator_service/ ...`):
  ```dockerfile
  # BLK-03: rate-limiter lives in the repo-root infrastructure/ package (imported at
  # main.py:334). Copy it so PYTHONPATH=/app resolves `infrastructure.api_gateway...`.
  COPY --chown=appuser:appuser infrastructure/ ./infrastructure/
  ```

- [ ] Edit `services/user-service/Dockerfile` — add after line 33 (`COPY ... services/user-service/src/ ./src/`):
  ```dockerfile
  # BLK-03: rate-limiter lives in the repo-root infrastructure/ package (imported at
  # main.py:496). Copy it so PYTHONPATH=/app resolves `infrastructure.api_gateway...`.
  COPY --chown=appuser:appuser infrastructure/ ./infrastructure/
  ```

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestInfraCopied -o "addopts=" -p no:logging -q
  ```

- [ ] `# REAL-INFRA` — on a Docker host, verify the import resolves inside the built image:
  ```bash
  docker build -f services/orchestrator_service/Dockerfile -t solace-orchestrator:blk03 .
  docker run --rm solace-orchestrator:blk03 python -c "import infrastructure.api_gateway.rate_limiting; print('ok')"
  # expect: ok
  ```

- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ -o "addopts=" -p no:logging -q
  ```

---

### Task 5: BLK-20 — Analytics `model_dump(mode="json")`

`analytics-service/src/main.py:288` calls `await _analytics_consumer.process_event(event.model_dump())`. Default (`mode="python"`) keeps `metadata.timestamp` as a `datetime`; `AnalyticsEvent.from_raw` then calls `datetime.fromisoformat(metadata["timestamp"])` (`consumer.py:86`) → `TypeError` → caught+swallowed at `consumer.py:433-436` → `process_event` returns `False` → 100% of Kafka analytics events dropped. Fix: serialize with `mode="json"` (timestamp becomes an ISO string). Extract the mapping into a module-level, testable helper.

**Files:**
- Modify `services/analytics-service/src/main.py` (:287-290)
- Create `services/analytics-service/tests/test_kafka_event_serialization.py`

**Interfaces:**
- Consumes: a `BaseEvent` subclass (e.g. `SessionStartedEvent`) from the Kafka consumer.
- Produces: `_event_to_analytics_payload(event) -> dict[str, Any]` returning `event.model_dump(mode="json")` (JSON-safe, `metadata.timestamp` is `str`).

**Steps:**

- [ ] Write the failing test. Create `services/analytics-service/tests/test_kafka_event_serialization.py`:
  ```python
  """BLK-20: analytics must serialize Kafka events JSON-safe or every event is dropped."""
  from uuid import uuid4

  import pytest

  from src.solace_events.schemas import SessionStartedEvent


  def _make_event() -> SessionStartedEvent:
      return SessionStartedEvent(user_id=uuid4(), session_id=uuid4(), session_number=1)


  def test_payload_timestamp_is_json_string() -> None:
      from services.analytics_service.src.main import _event_to_analytics_payload  # noqa: E402
      payload = _event_to_analytics_payload(_make_event())
      ts = payload["metadata"]["timestamp"]
      assert isinstance(ts, str), "BLK-20: timestamp must be an ISO string, not datetime"

  @pytest.mark.asyncio
  async def test_process_event_succeeds_on_json_payload() -> None:
      from services.analytics_service.src.consumer import AnalyticsConsumer, ConsumerConfig
      from services.analytics_service.src.main import _event_to_analytics_payload
      consumer = AnalyticsConsumer(ConsumerConfig())
      ok = await consumer.process_event(_event_to_analytics_payload(_make_event()))
      assert ok is True, "BLK-20: a JSON-safe payload must process, not get swallowed"

  @pytest.mark.asyncio
  async def test_python_mode_payload_is_dropped_documents_the_bug() -> None:
      """Guardrail: the OLD behavior (model_dump()) drops the event — proves the fix matters."""
      from services.analytics_service.src.consumer import AnalyticsConsumer, ConsumerConfig
      consumer = AnalyticsConsumer(ConsumerConfig())
      ok = await consumer.process_event(_make_event().model_dump())  # python mode -> datetime
      assert ok is False
  ```
  (Note: the analytics package dir is `services/analytics-service`; its `src` is importable as `services.analytics_service.src` per the existing test suite's import convention — mirror whatever `services/analytics-service/tests/test_consumer.py` uses; if that suite imports `from src...` with a `conftest.py` sys.path insert, use the same form.)

- [ ] Run it — expect FAIL (`_event_to_analytics_payload` does not exist; and the python-mode guardrail proves the drop):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/analytics-service/tests/test_kafka_event_serialization.py -o "addopts=" -p no:logging -q
  ```

- [ ] Edit `services/analytics-service/src/main.py`. Add a module-level helper near the top of the file (after imports) and rewire the closure at :287-290:
  ```python
  def _event_to_analytics_payload(event) -> dict:
      """BLK-20: serialize a Kafka BaseEvent JSON-safe so from_raw's
      datetime.fromisoformat(metadata['timestamp']) gets a str, not a datetime.
      Without mode='json' every event raises TypeError and is silently dropped."""
      return event.model_dump(mode="json")
  ```
  Then replace the inner closure body (:287-288):
  ```python
              # Feed deserialized Kafka events into the analytics processor
              async def _kafka_to_analytics(event):
                  await _analytics_consumer.process_event(_event_to_analytics_payload(event))
  ```

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/analytics-service/tests/test_kafka_event_serialization.py -o "addopts=" -p no:logging -q
  ```

- [ ] **Regression gate** — run the analytics suite:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest services/analytics-service/tests/ -o "addopts=" -p no:logging -q
  ```
  Expected: prior 201 + new tests green.

---

### Task 6: BLK-06 — `config_service` production boot

In production, `ConfigurationManager.load()` calls `_validate_production_secrets()`, which raises unless `database.password` and `security.jwt_secret_key` are non-default. But `_load_from_environment()` (`settings.py:360-362`) returns only `{"environment": ...}` and no config files ship, so both secrets stay at their `SecretStr` defaults (`"changeme"`, `"change-in-production"`) → the service can never boot in prod. Fix: (a) ingest DB/JWT/Redis secrets from env vars in `_load_from_environment`; (b) ship a non-secret `config.production.json` baseline.

**Files:**
- Modify `services/config_service/src/settings.py` (`_load_from_environment`, :360-362; add env fields to `ConfigServiceSettings`, :64-87)
- Create `services/config_service/config/config.production.json`
- Create `tests/config_service/test_production_boot.py`

**Interfaces:**
- Consumes env: `CONFIG_DB_PASSWORD`, `CONFIG_JWT_SECRET_KEY`, `CONFIG_REDIS_PASSWORD` (via the `CONFIG_` prefix already on `ConfigServiceSettings`).
- Produces: `_load_from_environment()` returns a nested dict merging `database`/`security`/`redis` secrets from env so `_validate_production_secrets()` passes.

**Steps:**

- [ ] Write the failing test. Create `tests/config_service/test_production_boot.py`:
  ```python
  """BLK-06: config_service must boot in production when secrets are supplied via env."""
  import pytest

  from services.config_service.src.settings import (
      ConfigEnvironment, ConfigServiceSettings, ConfigurationManager,
  )


  @pytest.mark.asyncio
  async def test_production_boot_raises_without_secrets() -> None:
      settings = ConfigServiceSettings(environment=ConfigEnvironment.PRODUCTION,
                                       hot_reload_enabled=False, config_path="./__nonexistent__")
      mgr = ConfigurationManager(settings)
      with pytest.raises(RuntimeError, match="Insecure default secrets"):
          await mgr.load()

  @pytest.mark.asyncio
  async def test_production_boot_succeeds_with_env_secrets(monkeypatch) -> None:
      monkeypatch.setenv("CONFIG_DB_PASSWORD", "a-real-strong-db-password-123")
      monkeypatch.setenv("CONFIG_JWT_SECRET_KEY", "a-real-strong-jwt-secret-key-456")
      settings = ConfigServiceSettings(environment=ConfigEnvironment.PRODUCTION,
                                       hot_reload_enabled=False, config_path="./__nonexistent__")
      mgr = ConfigurationManager(settings)
      cfg = await mgr.load()  # must NOT raise
      assert cfg.environment == ConfigEnvironment.PRODUCTION
      assert cfg.database.password.get_secret_value() == "a-real-strong-db-password-123"
      assert cfg.security.jwt_secret_key.get_secret_value() == "a-real-strong-jwt-secret-key-456"
  ```

- [ ] Run it — expect FAIL (`test_production_boot_succeeds_with_env_secrets` raises because env is ignored):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/config_service/test_production_boot.py -o "addopts=" -p no:logging -q
  ```

- [ ] Add secret-ingest fields to `ConfigServiceSettings` (in `settings.py`, inside the class near :83, all `CONFIG_`-prefixed, default `None`):
  ```python
      # BLK-06: production secret ingestion (CONFIG_ prefix). When set, these flow
      # into ApplicationConfig via _load_from_environment so _validate_production_secrets passes.
      db_password: SecretStr | None = Field(default=None)
      jwt_secret_key: SecretStr | None = Field(default=None)
      redis_password: SecretStr | None = Field(default=None)
  ```

- [ ] Rewrite `_load_from_environment` (:360-362) to build the nested override dict:
  ```python
      def _load_from_environment(self) -> dict[str, Any]:
          """Load configuration overrides from environment variables.

          BLK-06: ingest DB/JWT/Redis secrets from the CONFIG_-prefixed env so a
          production deployment can supply non-default secrets and boot (otherwise
          _validate_production_secrets raises on the SecretStr defaults).
          """
          override: dict[str, Any] = {"environment": self._settings.environment.value}
          if self._settings.db_password is not None:
              override.setdefault("database", {})["password"] = (
                  self._settings.db_password.get_secret_value())
          if self._settings.jwt_secret_key is not None:
              override.setdefault("security", {})["jwt_secret_key"] = (
                  self._settings.jwt_secret_key.get_secret_value())
          if self._settings.redis_password is not None:
              override.setdefault("redis", {})["password"] = (
                  self._settings.redis_password.get_secret_value())
          return override
  ```

- [ ] Ship the non-secret production baseline. Create `services/config_service/config/config.production.json`:
  ```json
  {
    "environment": "production",
    "database": { "ssl_mode": "require", "pool_size": 20, "pool_max_overflow": 10 },
    "redis": { "ssl": true, "max_connections": 50 },
    "security": {
      "jwt_algorithm": "RS256",
      "phi_encryption_enabled": true,
      "audit_logging_enabled": true
    },
    "observability": { "metrics_enabled": true, "tracing_enabled": true, "tracing_sample_rate": 0.1 }
  }
  ```
  (Secrets are NOT in this file — they arrive via `CONFIG_DB_PASSWORD` / `CONFIG_JWT_SECRET_KEY`. In prod set `CONFIG_CONFIG_PATH=./services/config_service/config` or mount it at the manager's `config_path`.)

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/config_service/test_production_boot.py -o "addopts=" -p no:logging -q
  ```

- [ ] **Regression gate** — run config_service tests:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/config_service/ services/config_service -o "addopts=" -p no:logging -q
  ```

---

### Task 7: BLK-05 — Alembic migration for audit/outbox/DLQ tables + auto-migrate

Migrations are never auto-applied (no alembic step in compose/CI), and `audit_events`, `event_outbox`, `event_dead_letter`, `processed_events` are created ad-hoc by app-startup `ensure_table()` (`src/solace_events/postgres_stores.py:114-118`, `src/solace_security/audit.py:328`) — outside Alembic, so their schema/reversibility is unversioned. Move that DDL into a real migration (`009`, revising head `008_rls_personality`), and add a single `alembic upgrade head` step run by a compose init service and a CI job (Task 8 wires the CI job).

**Files:**
- Create `migrations/versions/009_audit_outbox_dlq_tables.py`
- Create `tests/migrations/test_009_audit_outbox_dlq.py`
- Modify `docker-compose.yml` (add `db-migrate` service; app services `depends_on` it)
- Modify `docker-compose.prod.yml` (add `db-migrate` service)
- Modify `docs/STAGING-BRINGUP.md` (note the migrate step)

**Interfaces:**
- Consumes: `DATABASE_URL` env (honored by `migrations/env.py:21-25`), the DDL constants `OUTBOX_TABLE_DDL`, `DLQ_TABLE_DDL`, `PROCESSED_EVENTS_TABLE_DDL` (`src/solace_events/postgres_stores.py`) and `PostgresAuditStore.CREATE_TABLE_SQL` (`src/solace_security/audit.py`).
- Produces: migration `009_audit_outbox_dlq` at the Alembic head; the four tables provisioned by `alembic upgrade head`.

**Steps:**

- [ ] Write the failing migration test. Create `tests/migrations/test_009_audit_outbox_dlq.py`:
  ```python
  """BLK-05: migration 009 chains from head and provisions audit/outbox/DLQ tables."""
  import importlib.util
  from pathlib import Path

  MIGRATIONS = Path(__file__).resolve().parent.parent.parent / "migrations" / "versions"


  def _load(mod_name: str, filename: str):
      spec = importlib.util.spec_from_file_location(mod_name, MIGRATIONS / filename)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod


  def test_009_chains_from_personality_head() -> None:
      m = _load("m009", "009_audit_outbox_dlq_tables.py")
      assert m.revision == "009_audit_outbox_dlq"
      assert m.down_revision == "008_rls_personality"

  def test_009_upgrade_downgrade_are_defined() -> None:
      m = _load("m009", "009_audit_outbox_dlq_tables.py")
      assert callable(m.upgrade) and callable(m.downgrade)

  def test_009_ddl_covers_all_four_tables() -> None:
      src = (MIGRATIONS / "009_audit_outbox_dlq_tables.py").read_text(encoding="utf-8")
      for table in ("audit_events", "event_outbox", "event_dead_letter", "processed_events"):
          assert table in src, f"BLK-05: migration 009 must create {table}"
  ```
  (Confirm the `008` revision id string in `migrations/versions/008_rls_personality_tables.py` — it is `revision = "008_rls_personality"`; match it exactly.)

- [ ] Run it — expect FAIL (module file missing):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/migrations/test_009_audit_outbox_dlq.py -o "addopts=" -p no:logging -q
  ```

- [ ] Create `migrations/versions/009_audit_outbox_dlq_tables.py`. Import the existing DDL constants so the migration and runtime `ensure_table()` stay byte-identical (no drift), guarded Postgres-only like 006-008:
  ```python
  """Provision the audit/outbox/DLQ tables via Alembic (BLK-05).

  Revision ID: 009_audit_outbox_dlq
  Revises: 008_rls_personality
  Create Date: 2026-08-24

  Historically these four tables were created ad-hoc by app-startup ensure_table()
  calls (src/solace_events/postgres_stores.py, src/solace_security/audit.py), which
  left their schema unversioned and outside migration reversibility. This migration
  runs the SAME DDL constants (imported, not copied, to avoid drift) so an init
  container's `alembic upgrade head` provisions them before any service boots.

  Postgres-only; a no-op on SQLite (tests). Idempotent (all DDL is CREATE TABLE IF
  NOT EXISTS / ADD COLUMN IF NOT EXISTS).
  """
  from __future__ import annotations

  from alembic import context, op

  from src.solace_events.postgres_stores import (
      DLQ_TABLE_DDL,
      OUTBOX_TABLE_DDL,
      PROCESSED_EVENTS_TABLE_DDL,
  )
  from src.solace_security.audit import PostgresAuditStore

  revision = "009_audit_outbox_dlq"
  down_revision = "008_rls_personality"
  branch_labels = None
  depends_on = None


  def _is_postgres() -> bool:
      bind = context.get_bind()
      return bind is not None and bind.dialect.name == "postgresql"


  def upgrade() -> None:
      if not _is_postgres():
          return
      op.execute(PostgresAuditStore.CREATE_TABLE_SQL)
      op.execute(OUTBOX_TABLE_DDL)
      op.execute(DLQ_TABLE_DDL)
      op.execute(PROCESSED_EVENTS_TABLE_DDL)


  def downgrade() -> None:
      if not _is_postgres():
          return
      op.execute("DROP TABLE IF EXISTS processed_events;")
      op.execute("DROP TABLE IF EXISTS event_dead_letter;")
      op.execute("DROP TABLE IF EXISTS event_outbox;")
      op.execute("DROP TABLE IF EXISTS audit_events;")
  ```
  (Verify `PostgresAuditStore.CREATE_TABLE_SQL` is a class attribute — it is, at `audit.py:328`. If the audit DDL references a full statement list that ends mid-column in the file, read `audit.py:328-380` and confirm `CREATE_TABLE_SQL` is a complete, self-contained statement string before relying on it; it is a single triple-quoted `CREATE TABLE IF NOT EXISTS audit_events (...)`.)

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/migrations/test_009_audit_outbox_dlq.py -o "addopts=" -p no:logging -q
  ```

- [ ] Add the `db-migrate` init service to `docker-compose.yml` (in the APPLICATION SERVICES section, before `user-service`). It reuses the user-service image (has alembic + `src/` + `migrations/` — migrations dir must be COPY'd; if the image lacks `migrations/`, run migrate from a lightweight image mounting the repo — see note):
  ```yaml
    db-migrate:
      build:
        context: .
        dockerfile: services/user-service/Dockerfile
      container_name: solace-db-migrate
      command: ["python", "-m", "alembic", "upgrade", "head"]
      working_dir: /app
      environment:
        DATABASE_URL: postgresql+asyncpg://solace:${POSTGRES_PASSWORD:-solace}@postgres:5432/solace
      volumes:
        - ./migrations:/app/migrations:ro
        - ./alembic.ini:/app/alembic.ini:ro
      depends_on:
        postgres:
          condition: service_healthy
      restart: "no"
  ```
  Then add to EACH app service's `depends_on` a gate on migrate completion, e.g. for `user-service`:
  ```yaml
      depends_on:
        postgres:
          condition: service_healthy
        db-migrate:
          condition: service_completed_successfully
  ```

- [ ] Add the same `db-migrate` service to `docker-compose.prod.yml` (same shape; `DATABASE_URL` from the prod Postgres host/secret) and gate prod app services on `service_completed_successfully`.

- [ ] Add a note to `docs/STAGING-BRINGUP.md`:
  ```markdown
  - **BLK-05:** schema is now applied by the `db-migrate` init service
    (`alembic upgrade head`) which runs once after Postgres is healthy and before any
    app service starts (`depends_on: db-migrate: condition: service_completed_successfully`).
    audit/outbox/DLQ tables are versioned in `migrations/versions/009_audit_outbox_dlq_tables.py`.
  ```

- [ ] `# REAL-INFRA` — prove the full chain applies on real Postgres:
  ```bash
  docker compose up -d postgres
  DATABASE_URL=postgresql+asyncpg://solace:solace@localhost:5432/solace alembic upgrade head
  docker compose exec postgres psql -U solace -d solace -c "\dt" | grep -E "audit_events|event_outbox|event_dead_letter|processed_events"
  # expect all four tables listed; `alembic current` shows 009_audit_outbox_dlq
  ```

- [ ] **Regression gate** — run the migrations suite:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/migrations/ tests/alignment/ -o "addopts=" -p no:logging -q
  ```

---

### Task 8: BLK-04 — CI build→boot→migrate→health gate

The CI `docker-build` job (`.github/workflows/ci.yml:148-185`) builds from a **per-service** context (`./services/user-service`) but the Dockerfiles require **repo-root** context (they COPY `requirements-runtime.txt`, `src/`, `infrastructure/`). It also never runs the image, so a crash-on-boot passes green. Rebuild from root, run the container, and probe `/health`. Add a Postgres service + `alembic upgrade head` step so migrations are exercised.

**Files:**
- Modify `.github/workflows/ci.yml` (`docker-build` job :148-185; add `db-migrate` job)
- Modify `tests/alignment/test_phase0_build_boot.py` (add `TestCiGates`)

**Interfaces:**
- Consumes: `services/<x>/Dockerfile`, repo-root context, service `SERVICE_PORT`.
- Produces: a `docker build -f <dockerfile> .` + `docker run` + `curl /health` gate; a `db-migrate` CI job with `services: postgres`.

**Steps:**

- [ ] Add the failing CI-shape test:
  ```python
  class TestCiGates:
      def _ci(self) -> str:
          return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

      def test_docker_build_uses_root_context_with_dockerfile_flag(self) -> None:
          ci = self._ci()
          assert "docker build -f" in ci, "BLK-04: build must use -f <dockerfile> with root context"
          # The old per-service-context form must be gone.
          assert "${{ matrix.service.context }}\n" not in ci or "dockerfile" in ci

      def test_docker_build_runs_and_health_probes(self) -> None:
          ci = self._ci()
          assert "docker run" in ci, "BLK-04: CI must run the image it builds"
          assert "/health" in ci, "BLK-04: CI must probe /health after boot"

      def test_ci_has_db_migrate_job(self) -> None:
          ci = self._ci()
          assert "alembic upgrade head" in ci, "BLK-05: CI must run migrations against real Postgres"
          assert "postgres:" in ci and "services:" in ci, "BLK-05: CI needs a Postgres service"
  ```

- [ ] Run it — expect FAIL:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestCiGates -o "addopts=" -p no:logging -q
  ```

- [ ] Replace the `docker-build` job matrix + steps in `.github/workflows/ci.yml` (:148-185). Change matrix entries to carry the Dockerfile path + a boot port, and rewrite the build/run/probe steps:
  ```yaml
    docker-build:
      name: Docker Build & Boot (${{ matrix.service.name }})
      runs-on: ubuntu-latest
      needs: lint
      strategy:
        fail-fast: false
        matrix:
          service:
            - { dockerfile: "services/user-service/Dockerfile", name: "user-service", port: 8001 }
            - { dockerfile: "services/safety_service/Dockerfile", name: "safety-service", port: 8002 }
            - { dockerfile: "services/notification-service/Dockerfile", name: "notification-service", port: 8007 }
            - { dockerfile: "services/diagnosis_service/Dockerfile", name: "diagnosis-service", port: 8004 }
            - { dockerfile: "services/memory_service/Dockerfile", name: "memory-service", port: 8005 }
            - { dockerfile: "services/therapy_service/Dockerfile", name: "therapy-service", port: 8003 }
            - { dockerfile: "services/personality_service/Dockerfile", name: "personality-service", port: 8006 }
            - { dockerfile: "services/orchestrator_service/Dockerfile", name: "orchestrator-service", port: 8000 }
            - { dockerfile: "services/analytics-service/Dockerfile", name: "analytics-service", port: 8009 }
            - { dockerfile: "services/config_service/Dockerfile", name: "config-service", port: 8008 }
      steps:
        - uses: actions/checkout@v4

        - name: Build image (repo-root context)
          run: |
            docker build -f "${{ matrix.service.dockerfile }}" \
              --tag "solace-ai/${{ matrix.service.name }}:ci" .

        - name: Run container & probe /health
          run: |
            docker run -d --name svc \
              -e ENVIRONMENT=test \
              -e AUTH_SECRET_KEY="${AUTH_SECRET_KEY}" \
              -e USER_SERVICE_JWT_SECRET_KEY="${USER_SERVICE_JWT_SECRET_KEY}" \
              -e ENCRYPTION_MASTER_KEY="${ENCRYPTION_MASTER_KEY}" \
              -e USER_SERVICE_FIELD_ENCRYPTION_KEY="${USER_SERVICE_FIELD_ENCRYPTION_KEY}" \
              -e FERNET_TOKEN_KEY="${FERNET_TOKEN_KEY}" -e FERNET_FIELD_KEY="${FERNET_FIELD_KEY}" \
              -e KAFKA_ENABLED=false \
              -p "${{ matrix.service.port }}:${{ matrix.service.port }}" \
              "solace-ai/${{ matrix.service.name }}:ci"
            for i in $(seq 1 30); do
              if curl -fsS "http://localhost:${{ matrix.service.port }}/health"; then
                echo "  <- ${{ matrix.service.name }} healthy"; ok=1; break
              fi
              sleep 2
            done
            if [ "${ok:-0}" != "1" ]; then
              echo "::error::${{ matrix.service.name }} did not become healthy"; docker logs svc; exit 1
            fi

        - name: Teardown
          if: always()
          run: docker rm -f svc || true
  ```

- [ ] Add a `db-migrate` CI job (after `integration-tests`) that runs `alembic upgrade head` against a real Postgres service:
  ```yaml
    db-migrate:
      name: DB Migrate (real Postgres)
      runs-on: ubuntu-latest
      needs: lint
      services:
        postgres:
          image: postgres:16-alpine
          env:
            POSTGRES_USER: solace
            POSTGRES_PASSWORD: ci_test_password
            POSTGRES_DB: solace
          ports: ["5432:5432"]
          options: >-
            --health-cmd "pg_isready -U solace" --health-interval 5s
            --health-timeout 5s --health-retries 10
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ env.PYTHON_VERSION }}
            cache: pip
        - name: Install runtime deps + alembic
          run: |
            python -m pip install --upgrade pip
            pip install -r requirements-runtime.txt
        - name: alembic upgrade head
          env:
            DATABASE_URL: postgresql+asyncpg://solace:ci_test_password@localhost:5432/solace
          run: alembic upgrade head
        - name: Verify audit/outbox/DLQ tables exist
          env:
            PGPASSWORD: ci_test_password
          run: |
            psql -h localhost -U solace -d solace -c "\dt" | \
              grep -E "audit_events|event_outbox|event_dead_letter|processed_events"
  ```

- [ ] Run the CI-shape test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestCiGates -o "addopts=" -p no:logging -q
  ```

- [ ] `# REAL-INFRA` — the true gate is the CI run itself (needs Docker + Postgres). Locally verify one leg:
  ```bash
  docker build -f services/config_service/Dockerfile -t solace-ai/config-service:ci .
  docker run -d --name svc -e ENVIRONMENT=test -p 8008:8008 solace-ai/config-service:ci
  curl -fsS http://localhost:8008/health && echo OK; docker rm -f svc
  ```

- [ ] **Regression gate:**
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ -o "addopts=" -p no:logging -q
  ```

---

### Task 9: H-01 — Blocking `pip-audit` + Trivy supply-chain CI gate

The only image gate today is broken (Task 8) and there is no dependency-vuln scan or SBOM. Add a `security-scan` CI job that runs `pip-audit` against the lockfile (blocking) and Trivy filesystem/image scan (blocking on HIGH/CRITICAL).

**Files:**
- Modify `.github/workflows/ci.yml` (add `security-scan` job)
- Modify `tests/alignment/test_phase0_build_boot.py` (add `TestSupplyChainGate`)

**Interfaces:**
- Consumes: `requirements-runtime.lock` (Task 2), the built images (Task 8).
- Produces: a `security-scan` job failing the build on a known CVE in a runtime dep.

**Steps:**

- [ ] Add the failing test:
  ```python
  class TestSupplyChainGate:
      def _ci(self) -> str:
          return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

      def test_pip_audit_gate_present(self) -> None:
          ci = self._ci()
          assert "pip-audit" in ci, "H-01: CI must run pip-audit as a blocking gate"

      def test_trivy_gate_present(self) -> None:
          ci = self._ci()
          assert "trivy" in ci.lower(), "H-01: CI must run a Trivy scan"

      def test_scan_is_blocking(self) -> None:
          ci = self._ci()
          # The scan job must not blanket continue-on-error the pip-audit step.
          assert "security-scan" in ci, "H-01: a dedicated security-scan job must exist"
  ```

- [ ] Run it — expect FAIL:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestSupplyChainGate -o "addopts=" -p no:logging -q
  ```

- [ ] Add the `security-scan` job to `.github/workflows/ci.yml`:
  ```yaml
    security-scan:
      name: Supply-chain scan (pip-audit + Trivy)
      runs-on: ubuntu-latest
      needs: lint
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ env.PYTHON_VERSION }}

        - name: pip-audit (blocking) against the lockfile
          run: |
            python -m pip install --upgrade pip pip-audit
            # Fail the build on any known vuln in the pinned runtime deps.
            pip-audit -r requirements-runtime.lock --strict --desc on

        - name: Trivy filesystem scan (blocking on HIGH/CRITICAL)
          uses: aquasecurity/trivy-action@0.28.0
          with:
            scan-type: fs
            scan-ref: .
            severity: HIGH,CRITICAL
            exit-code: "1"
            ignore-unfixed: true
  ```

- [ ] Run the test — expect PASS:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py::TestSupplyChainGate -o "addopts=" -p no:logging -q
  ```

- [ ] `# REAL-INFRA` — confirm pip-audit runs clean against the lock (may flag `python-jose 3.3.0` CVE-2024-33664, which is H-03 in Phase 3; if so, record it and either add a scoped `--ignore-vuln` with a Phase-3 TODO or pull jose out of the runtime lock since PyJWT is primary):
  ```bash
  pip install pip-audit && pip-audit -r requirements-runtime.lock --strict --desc on
  ```

- [ ] **Regression gate** — run the whole Phase 0 alignment file:
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/test_phase0_build_boot.py -o "addopts=" -p no:logging -q
  ```

---

## Phase 0 exit verification (run after all 9 tasks)

- [ ] Full fast-suite regression (unit/alignment/migrations/config/analytics touched by Phase 0):
  ```powershell
  $env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m pytest tests/alignment/ tests/migrations/ tests/config_service/ services/analytics-service/tests/ -o "addopts=" -p no:logging -q
  ```
- [ ] `# REAL-INFRA` (Docker host required) — the true Phase 0 exit per the release plan:
  ```bash
  docker compose build && docker compose up -d
  # db-migrate runs alembic upgrade head; every service becomes healthy; /health returns 200
  docker compose ps   # all services healthy, db-migrate Exited (0)
  ```
- [ ] Adversarial re-verify: confirm no crisis-path DDL drifted (migration 009 imports the runtime DDL constants, not copies) and that the analytics fix does not change any other serialization path.
