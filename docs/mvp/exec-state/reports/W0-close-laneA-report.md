# W0-close lane A: fix tasks for the 2026-11-07 review findings

Branch `w0-barrier`, workflow base `6ab630b`. Four tasks, one commit each, test-first. Sources: `D:/tmp/mvp-exec/review-2026-11-07/security.md`, `crisis.md` and `skeptics.json` (every finding below survived its 3-skeptic panel with 0 refutes, except SEC-4 with 1 of 3).

| Task | Findings | Commit | Result |
|---|---|---|---|
| WS-03.F1 | SEC-1, SEC-2, owner ruling on `/ready` and `/health` | `0efe9af` | closed |
| WS-04.F2 | SEC-3, SEC-4 (non-audit half) | `e715074` | closed; audit half stays with WS-04.10 |
| WS-05.F1 | CR-4, CR-3 | `a30b6e2` | closed (CR-3 closed, not parked) |
| WS-13.F1 | CR-1 | `ba7debb` | closed |
| CR-2 | `/ready` says the consumer runs while it polls nothing | none | parked to WS-13.8 (W1) |

Test commands: `D:/tmp/mvp-exec/venv/Scripts/python.exe -m pytest <path> -o "addopts=" -p no:logging -q` with `PYTHONPATH=<repo>;<repo>/src`, `ENVIRONMENT=test`, `TELEMETRY_ENABLED=false`, one process at a time. Raw evidence files are in `D:/tmp/mvp-exec/scratch/w0close-laneA-*.txt`.

## WS-03.F1: the edge and the strict overlay (`0efe9af`)

**SEC-1.** Every `never_routed` entry is now a prefix: `/metrics* /docs* /redoc* /openapi.json* /status* /ready* /health* /api/v1/internal*`. Caddy lowercases and cleans the decoded path, and a single trailing `*` is a prefix match. So `/metrics%0A`, `/metrics/`, `/METRICS`, `%0D%0A`, `%09`, `%00`, `%20`, `%2F` and dot or slash variants all get 404 before the proxy. As defence in depth, a path with a decoded control character (`path_regexp [[:cntrl:]]`) gets 400 after the 404 rule. That covers every exact app route, public ones included.

**`/ready` and `/health` (owner-recommended option).** They are never routed on the two proxied sites, `api.` and `auth.`. `/live` stays routed. I checked that nothing needs them through the edge:
- `Stack.call` and `_wait_live` in `src/solace_testing/e2e.py` use `docker compose exec`.
- The container healthchecks and Caddy's active checks use `/live`.
- Prometheus scrapes on the internal network.
- The only other use, `docs/STAGING-BRINGUP.md:116`, uses direct ports.

**SEC-2.** `docker-compose.prod.yml` now refuses any `ENVIRONMENT` other than exactly `staging` or `production`. Caddy, migrate and the nine app services mount the config `strict-environment-${ENVIRONMENT}`, and only `strict-environment-staging` and `strict-environment-production` are defined. Any other value fails both `docker compose config` and `up` before anything starts. `docker-compose.yml` alone still renders with its dev defaults and has no gate. Rendered through the harness:
```
development -> refused: service "safety-service" refers to undefined config strict-environment-development: invalid compose project
test        -> refused: service "therapy-service" refers to undefined config strict-environment-test: invalid compose project
prod        -> refused: service "caddy" refers to undefined config strict-environment-prod: invalid compose project
staging     -> rendered; caddy ports ['80', '443', '443']
production  -> rendered; caddy ports ['80', '443', '443']
```

**RED** (`w0close-laneA-ws03-red-unit.txt`): `16 failed, 21 passed, 9 skipped`. The failures were the 4 Caddyfile cases and 12 overlay cases: 10 non-strict values were not refused, and 2 renders had no `configs` key. The real Caddy run (pinned `caddy:2.8-alpine`, old Caddyfile, raw requests, `w0close-laneA-ws03-red-caddy-probe.txt`) showed:
```
api.localhost '/metrics' 404          api.localhost '/metrics%0A' 502       auth.localhost '/metrics%0A' 503
api.localhost '/METRICS%0A' 503       api.localhost '/ready' 503            api.localhost '/health' 503
```
**GREEN**: the same probe on the new Caddyfile (`w0close-laneA-ws03-green-caddy-probe.txt`):
- Every `/metrics`, `/ready`, `/health`, `/api/v1/internal%0A` and `/openapi.json%0A` variant got 404 on both sites.
- `/live%0A` got 400.
- `/live` and `/api/v1/chat` got 502/503, which means they were proxied.

`caddy validate` with the strict plain-host sites printed `Valid configuration`. The unit tests: `37 passed, 9 skipped`. With `SOLACE_RUN_DOCKER_TESTS=1`, `test_mvp_250_caddy_edge.py` gave `28 passed`. That includes the new real-Caddy tests (all variants × both sites → 404; control characters → 400; `/live` still proxied; `caddy validate` for dev and strict) and the existing adapted-config tests.

**Regression:**
- `tests/deploy`: 125 passed, 19 skipped (baseline 109 passed, 15 skipped).
- `tests/solace_testing`: 259 passed.
- `tests/security/test_config_secrets_hardening.py`: 10 passed.
- Ruff is clean on the changed tests.

**Test edits:** three existing `test_mvp_250` assertions name the old exact tokens. They were rewritten to the prefix tokens (`NEVER`, the adapted-config index, the adapted-path list). The new assertions are stronger, because a prefix covers the exact path.

**Files:** `Caddyfile`, `docker-compose.prod.yml`, `tests/deploy/test_mvp_250_caddy_edge.py`, `tests/deploy/test_mvp_004_environment_derivation.py`. No change to the env registry or `.env.example` was needed.

**Not done:** the app-side control-character refusal in `RequestContextMiddleware` (the review's second fix direction). It is outside lane A's files; see the Ruling below.

## WS-04.F2: strict CORS and staging-strict checks (`e715074`)

**SEC-3.** In strict, `cors_allowed_origins` (`src/solace_infrastructure/runtime/http.py`) refuses an origin when:
- it is not `https`;
- its host is `localhost`, a `*.localhost` name, or ends in a trailing dot after one of those;
- its host is a loopback or unspecified address, including IPv4-mapped IPv6;
- it is malformed or has no host.

The `*` and `null` refusal is unchanged. The message names the entry's position and the rule it breaks, never the origin. Dev and test keep `http://localhost:3000`. A strict edge app with a localhost origin refuses to build.

**SEC-4, non-audit half.** These checks now treat staging as strict, using strict = own environment OR the bare `ENVIRONMENT` (`is_strict_environment`):
- `services/user-service/src/auth.py`: the `SessionConfig` insecure default secret (`USER_SERVICE_ENV`).
- `services/memory_service/src/infrastructure/weaviate_repo.py`: a missing `weaviate-client` package (`MEMORY_ENVIRONMENT`).
- `services/notification-service/src/consumers.py`: the import-time check for a missing `solace_events` package (`NOTIFICATION_SERVICE_ENV`).

**RED** (`w0close-laneA-ws04-red.txt`):
- CORS: 36 failed, 14 passed.
- Session secret: 5 failed, 4 passed.
- Weaviate: 4 failed, 2 passed.
- Notification import check: 4 failed, 2 passed.

The notification check is tested on a fresh copy of the module under a probe name, so the real `src.consumers` is never reloaded.

**GREEN:** 50, 9, 6 and 6 passed.

**Regression** (`w0close-laneA-ws04-regression.txt`):
- `tests/bootstrap`: 310 passed, 1 skipped.
- `tests/deploy`: 125 passed, 19 skipped.
- user-service: 378 passed.
- memory: 492 passed.
- notification: 168 passed, 7 skipped.
- orchestrator: 320 passed, 1 xfailed.

The final re-run of `tests/bootstrap` and `tests/deploy` after all four commits was unchanged. Ruff found no new findings; its remaining findings exist at base.

**Files:** `src/solace_infrastructure/runtime/http.py`, `services/user-service/src/auth.py`, `services/memory_service/src/infrastructure/weaviate_repo.py`, `services/notification-service/src/consumers.py`, `tests/bootstrap/test_ws04_f2_strict_cors_origins.py`, `services/user-service/tests/test_ws04_f2_session_secret_strict.py`, `services/memory_service/tests/test_ws04_f2_weaviate_client_strict.py`, `services/notification-service/tests/test_ws04_f2_strict_package_guard.py`.

## WS-05.F1: CI e2e truth and the manifest cause pin (`a30b6e2`)

**CR-4.** The `workflow_dispatch` e2e job now runs `run_suites.py --tier e2e --profile ${{ matrix.profile }} --require-realinfra`, so a missing Docker daemon fails the job. A new workflow-file test, `test_e2e_job_requires_real_infrastructure`, enforces it.

**CR-3.** A manifest entry may now set an optional `cause`. It is a regular expression searched in `"<ExceptionType>: <message>"` of the error that failed the hop (`HopFailure.__cause__`). At the expected hop, a cause that does not match FAILS with `manifest expects hop '<x>' to fail with a cause matching '<re>', got <Type>`. The verdict names only the exception type; the traceback still carries the text. The schema refuses an invalid or empty pattern. Entries without `cause` behave as before.

The `tests/e2e` regression had Docker available, so the dev crisis suite ran live: the stack came up, 8 tests were XFAIL at `provision_clinician`, and the stack was torn down. With `SOLACE_MANIFEST_IGNORE=1`, the observed cause was:
```
RuntimeError: scripts.provision_clinician exited 1:
/opt/venv/bin/python: No module named scripts.provision_clinician
```
So the 8 crisis entries in `tests/e2e/expected_failures.yaml` now pin `cause: 'No module named scripts\.provision_clinician'`. A live re-run with the pin gave `8 xfailed`.

**RED:**
- The original plugin was put back temporarily, from the HEAD bytes. `tests/build/test_mvp_038_ci_workflow.py` plus the plugin tests gave `3 failed, 32 passed`: the CI flag, `ManifestEntry` got an unexpected keyword `cause`, and the plugin run failed with `unknown key 'cause'`.
- The pin test failed with `cause=None`.
- The schema test passes on both trees, because the old schema also refused the key.

**GREEN:** `35 passed`, then the plugin file at `29 passed`.

**Regression** (`w0close-laneA-ws05-regression.txt`):
- `tests/build`: 289 passed, 1 skipped, 1 xfailed, plus 2 failed. Both failures are in `test_mvp_359`: new advisories GHSA-vvf4-9x63-8xvv (dnspython 2.9.0) and GHSA-mf5v-w3w3-f6cg (opentelemetry-api 1.45.1) in both locks. That is advisory-feed drift; lane A changed no lock or test. It is noted in the ledger for the lock owner, WS-01.
- `tests/solace_testing`: 265 passed.
- `tests/e2e`: 38 passed, 11 deselected, 8 xfailed (live dev).
- `tests/scripts`: 15 passed, 1 skipped.

**Files:** `.github/workflows/ci.yml`, `tests/build/test_mvp_038_ci_workflow.py`, `src/solace_testing/e2e_manifest.py`, `tests/solace_testing/test_mvp_039_manifest_plugin.py`, `tests/e2e/expected_failures.yaml`.

## WS-13.F1: crisis-consumer guard wiring (`ba7debb`)

**CR-1.** The new file `services/notification-service/tests/test_ws13_f1_crisis_guard_wiring.py` drives the real `main.lifespan`. The runtime bracket, service factory, event bridge and consumer start are replaced by recorders. Each strict case runs with the own and bare environments combined four ways (`production/production`, `staging/staging`, `development/production`, `staging/development`):
- Strict without Kafka fails before the consumer starts.
- Strict with a mock consumer fails before the consumer starts.
- Strict with a broker consumer starts it.
- Dev and test without Kafka still boot.
- The guard's allow path is also tested directly.

**RED**, by the review's mutations on `main.py` (`w0close-laneA-ws13-red.txt`), with `main.py` restored byte for byte afterwards (`git diff` empty):
- M1, the guard call commented out: 8 of the new tests failed (`DID NOT RAISE`), while all 16 existing `test_if01_runtime_adoption.py` cases still passed. This is exactly the review's gap.
- M3, the mock branch always raising: the 6 allow-path tests failed.

**GREEN:** 16 passed. **Regression:** notification 168 passed, 7 skipped; conformance `-k notification` 2 passed.

**Files:** `services/notification-service/tests/test_ws13_f1_crisis_guard_wiring.py` (test only; no product code changed).

## CR-2: parked

`/ready` reports `safety_consumer: running` while every poll fails. Per the task, it is not fixed in W0. The owner is WS-13.8 (W1). WS-13.8 deletes `SafetyEventConsumer` and the `safety_consumer` check, and moves notification onto the event plane. Its readiness then comes from WS-06.8's `plane.readiness()`, which currently means "every consumer task is running". WS-13.8 must require that readiness to reflect poll progress, for example a last-successful-poll age. WS-04.7's `CrisisConsumerStalled` stays the alert backstop.

## Rulings (all in the ledger)

- `/ready` and `/health` are never routed on both proxied sites, and `/live` stays routed. This is the owner-recommended option; nothing needs them through the edge.
- "Every site" means `api.` and `auth.`. WS-03.6's notification route sits inside the api site's `route` after `never_routed`, so it inherits the rules. The apex and the basic-auth dashboards are unchanged.
- SEC-1 is fixed with prefix entries plus a control-character 400. The app-side refusal in `RequestContextMiddleware` is not added: it is outside the lane's files, and internal callers are trusted.
- SEC-2 is fixed with a compose-native allow-list (`strict-environment-${ENVIRONMENT}` configs). A runtime `SOLACE_REQUIRE_STRICT` was not used, because it would touch WS-04 files. Case variants such as `Production` are refused too (fail closed).
- SEC-4's audit half (`src/solace_security/audit.py:80-81`) is owned by WS-04.10, which makes `AUDIT_HMAC_KEY` a required strict secret.
- The SEC-3 strict origin rule is as described above. Non-browser schemes (`capacitor://`, `app://`) would need an explicit allowance.
- CR-3 was first parked to WS-05.8, then revised and closed in this lane from the cause observed live. WS-05.8 still moves the entries when WS-14.9 lands the CLI.
- CR-2 is parked to WS-13.8, with WS-06.8's readiness meaning and WS-04.7 as the backstop.

## Housekeeping

- To run the real-Caddy tests I pulled the pinned edge image `caddy:2.8-alpine`, which was not present locally. I kept it on purpose: the dev and strict e2e stacks use it, and the barrier stage needs it.
- The e2e runs built the compose `solace/*:dev` and `solace-e2e-a-fake-llm` images. I left those images in place, as the harness is designed to do.
- No `solace` container, volume or network is left behind (checked with `docker ps -a`, `docker volume ls` and `docker network ls`). The probe containers were removed by name.
- No `.env` file was read. Only synthetic values were used.
- The body of the WS-03.F1 commit message shows `strict-environment-\` where it should say `strict-environment-${ENVIRONMENT}`, because the shell ate the variable. The title and the code are correct.
