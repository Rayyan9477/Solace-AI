# W0 barrier, stage B1: clean-export images, migrations on real Postgres, WS-05.4 live self-test

Branch `w0-barrier`, HEAD `44ea239` (tree `286c934`), run on 2026-11-24 on local Docker 29.7.2 with compose v5.4.0. Every item passed on the first run. Nothing failed, so B1 changed no code, test, compose file or Dockerfile, and there is no review package.

| Item | Command (summary) | Exit | Result |
|---|---|---|---|
| images | `scripts/ci/clean_checkout.py --tree 44ea239 --compose --tests --remove-images --require-registry-match` | 0 | 27/27 steps ok |
| image-probes | `pytest tests/build/test_mvp_008_clean_checkout.py tests/build/test_mvp_211_container_entry.py -m realinfra` | 0 | 17 passed, 62 deselected |
| migrations-real-pg | `pytest tests/migrations/test_chain_real_pg.py` (real infra) | 0 | 13 passed |
| ws-05.4-selftest | `pytest tests/e2e/test_mvp_039_stack_selftest.py` (dev; strict production as extra evidence) | 0 / 0 | 13 passed / 13 passed |

Common environment for every pytest run: `PYTHONPATH=D:/Repo/Contextual-Chatbot;D:/Repo/Contextual-Chatbot/src`, `ENVIRONMENT=test`, `TELEMETRY_ENABLED=false`, interpreter `D:/tmp/mvp-exec/venv/Scripts/python.exe`, flags `-o "addopts=" -p no:logging -q -rA`. Runs were one at a time. The raw logs are in `D:/tmp/mvp-exec/scratch-b1/`.

## 1. Clean-checkout image proof (WS-01.9, W0 exit criterion 2)

I used the repo's own tool, `scripts/ci/clean_checkout.py`. It extracts `git archive --format=tar <tree>` with autocrlf off, then builds and probes every image in the export's own `scripts/ci/images.py`. config-service is not in that registry (D-19).

```
$env:SOLACE_IMAGE_TAG="clean-286c934-b1w0"
D:/tmp/mvp-exec/venv/Scripts/python.exe scripts/ci/clean_checkout.py --tree 44ea239 --out D:/tmp/mvp-exec/export-44ea239 `
  --tag clean-286c934-b1w0 --compose --tests --remove-images --require-registry-match
```
Result: exit 0, `27/27 steps ok`.
- **10 image builds**, all built from the export: orchestrator, safety, therapy, diagnosis, memory, personality, user, notification and analytics, plus migrate.
- **15 import probes**, each a `docker run` against the built image id:
  - every CMD module;
  - the rate-limiting probe for orchestrator and user;
  - `src.api` and `src.domain.channels` for notification;
  - `scripts.db.migrate` and `solace_infrastructure.database.schema_check` for migrate.
- **One `docker compose -f docker-compose.yml build`** in the export. It built the same 10 images; the base file has no config-service.
- **`tests/build` inside the export**, with a private git context and `-m "not realinfra"`. It passed.

`SOLACE_IMAGE_TAG` gave the compose build a private tag, so the shared `solace/*:dev` tags were not overwritten. The tool's `--remove-images` untags only its own per-image builds. The compose build re-pointed those refs, so I removed the ten `solace/*:clean-286c934-b1w0` tags by tag with `docker rmi <ref>` (no `-f`). I deleted the export directory afterwards.

## 2. Image probe tests

```
$env:SOLACE_REQUIRE_REALINFRA="1"; $env:SOLACE_CLEAN_CHECKOUT_TREE="286c93402e2f4009493eb1bd058293f159270509"
python -m pytest tests/build/test_mvp_008_clean_checkout.py tests/build/test_mvp_211_container_entry.py -m realinfra ...
```
Result: exit 0, `17 passed, 62 deselected in 95.01s`. The 17 tests:
- `test_mvp_008_tree_builds_and_probes_from_clean_checkout`: the gate test. It re-runs the tool on the HEAD tree with `--require-registry-match`, asserts `25/25 steps ok` and that the registry has 10 or more images, and checks that the registry drift warning is absent.
- `test_f4_built_image_code_and_loader_are_root_owned_and_unwritable` for each of the 10 images: the code and the loader are root-owned, nothing under `/app` is writable by `appuser`, and the orchestrator and memory modules import as `appuser`.
- The 6 entrypoint tests (MVP-211):
  - the `NAME_FILE` export, in 3 forms;
  - fail-closed without echoing a value;
  - a NUL byte;
  - a procfs-style file.

Afterwards no `solace-f4-*` or `clean-*` image and no container was left. I deleted the gate test's export, `D:/tmp/mvp-exec/clean-checkout/286c934`.

## 3. Migrations on real PostgreSQL 16 (W0 exit criterion 3)

```
$env:SOLACE_REQUIRE_REALINFRA="1"; python -m pytest tests/migrations/test_chain_real_pg.py ...
```
Result: exit 0, `13 passed in 59.21s`, on the digest-pinned `postgres:16-alpine@sha256:16bc17c6...` with `infrastructure/init-db`. Each part of the criterion maps to a test:

| Criterion | Test |
|---|---|
| upgrade head on an empty database; `alembic_version` equals the chain head | `test_mvp_001_upgrade_head_on_empty_database` |
| downgrade base, then upgrade head, as `solace_migrator` through the migrate entrypoint | `test_mvp_001_downgrade_base_then_upgrade_head_round_trips` |
| `solace_app` cannot run DDL | `test_mvp_041_app_role_has_dml_but_no_ddl` |
| every policy uses NULLIF | `test_mvp_058_every_policy_uses_nullif` |
| SCHEMA_HEAD check passes as the app role | `test_mvp_041_schema_head_check_passes_as_app_role` |

Also green: upgrade to 007 needs no runtime outbox; each revision runs in its own transaction; `alembic.ini` holds no credentials; `audit_events` is append-only; a reset GUC returns no rows; the app role is RLS-isolated; user-scoped data is RLS-protected at head; the RLS audit names every cross-user bypass. Afterwards no container was left, and the volume count was unchanged (70 before and after).

## 4. WS-05.4 live self-test (the steps blocked on cloud egress since 2026-11-03)

Dev profile (the brief's Step 4):
```
$env:SOLACE_REQUIRE_REALINFRA="1"; $env:SOLACE_E2E_PROFILE="dev"; $env:SOLACE_E2E_LANE="b1"
python -m pytest tests/e2e/test_mvp_039_stack_selftest.py ...
```
Result: exit 0, `13 passed in 56.29s`. The 13 tests:
- no fixed container names;
- only caddy, mailpit and fake-llm published, and only on 127.0.0.1;
- `sql()` runs as `postgres`;
- `call()` reaches `/live` 200 on all 9 MVP services;
- the edge answers 401 for an unauthenticated orchestrator chat and 404 for `/metrics`;
- the strict overlay changes only `OVERLAY_ALLOWED_KEYS`.

Strict profile, extra evidence: the same command with `SOLACE_E2E_PROFILE=strict` and `SOLACE_E2E_ENVIRONMENT=production` gave exit 0, `13 passed in 54.84s`.

**Observed `service_live` down set: empty in both profiles.** Every service, notification-service included, answers `/live` 200. No `service_live` manifest entry is needed, and none exists (`tests/e2e/expected_failures.yaml` lists only the 8 W1 crisis entries and the `verify_email_via_sql` shortcut). The harness tore down `solace-e2e-b1` after each run: no container, volume, network or run dir was left. I removed the lane's `solace-e2e-b1-fake-llm:latest` image by tag.

Step 5 regression (unit, one process each):
- `tests/scripts`: exit 0, 15 passed and 1 skipped (the POSIX file-mode case, on Windows).
- `tests/solace_testing`: exit 0, 266 passed.

## Rulings (in the ledger)

- The live self-test ran on lane `b1` (project `solace-e2e-b1`), not lane `a`, so it could not collide with a parallel stage. The lane only names the compose project and the run dir. Cost if wrong: none.
- The self-test gate is the dev profile, as in the brief's Step 4. The strict production run is extra evidence. A failure only in strict would have been reported to the boot smoke (WS-05.5), not fixed here. It passed, so the question did not arise.

## Host state left behind

- No container, volume, network, export or run dir from B1.
- The harness's `compose up --build` re-pointed the shared `solace/*:dev` tags to the `solace-e2e-b1` builds. The build labels differ per project, so the ids differ. Lane A's previous ids are now untagged, but they share every layer, and `docker system df` stayed at 36.58 GB for images. I pruned nothing.
- The images B1 used but did not create are kept: `caddy:2.8-alpine`, the pinned mailpit and `postgres:16-alpine`, and `testcontainers/ryuk:0.8.1`.
- Lane A's `solace-e2e-a-fake-llm:latest` is still present; it is not B1's to remove.
- Free disk at the start: C: 8.6 GB, D: 2.7 GB. Docker builder prune was not needed.

## Not covered by B1

- The W0 boot smoke (`tests/e2e/test_w0_boot_smoke.py`, dev plus strict production and staging).
- The crisis E2E dev run.
- The full unit regression.
- The lint gate and `make ci-local`.

These belong to the other barrier stages. The 2 lock-advisory failures lane A recorded (`tests/build/test_mvp_359`, realinfra-marked, advisory-feed drift) were not run here: B1's `tests/build` run in the export excludes realinfra.
