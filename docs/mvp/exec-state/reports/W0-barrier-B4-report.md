# W0 barrier B4: full regression and CI parity

Tree: w0-barrier at 7176421 (one regression fix on top of 6ee852f). Interpreter: D:/tmp/mvp-exec/venv. Unit tier run with scripts/ci/run_suites.py --tier unit (one pytest process per suite, sequential).

## Per-suite counts (unit tier)

| Suite | Result | Review 2026-11-07 |
|---|---|---|
| tests/test_syntax_and_imports.py | 55 passed | 55 |
| tests/alignment | 16 passed | 16 |
| tests/solace_common | 170 passed | 170 |
| tests/solace_events | 243 passed | 243 |
| tests/solace_security | 351 passed | 351 |
| tests/solace_infrastructure | 936 passed, 12 failed (baseline red: kafka/test_topics.py) | 936 / 12 |
| tests/solace_testing | 266 passed | 259 |
| tests/integration | 236 passed | 236 |
| tests/security | 10 passed | 10 |
| tests/migrations | 71 passed, 1 skipped, 68 deselected (realinfra) | 72 passed, 4 failed, 64 skipped |
| tests/scripts | 15 passed, 1 skipped (18 passed + 1 skipped after the B4 fix) | 15 |
| tests/system | 6 passed | 6 |
| tests/config_service | 231 passed | 231 |
| tests/bootstrap | 310 passed, 1 deselected | 260 |
| tests/build | 273 passed, 1 xfailed, 19 deselected | 274 + 17 skipped |
| tests/deploy | 125 passed, 19 deselected | 109 |
| tests/contracts, tests/clinical | MISSING (do not exist yet), exit 0 | n/a |
| infrastructure/api_gateway/tests | 64 | 64 |
| services/analytics-service | 198 | 198 |
| services/diagnosis_service | 312 | 312 |
| services/memory_service | 492 | 486 |
| services/notification-service | 168 passed, 7 skipped | n/a |
| services/orchestrator_service | 320 passed, 1 xfailed | 320 |
| services/personality_service | 377 | 377 |
| services/safety_service | 508 passed, 1 xpassed | 508 |
| services/therapy_service | 526 passed, 1 xfailed | 526 |
| services/user-service | 378 | n/a |

No drops. migrations, build and deploy show deselected cases because the realinfra marker moves them to the realinfra tier (below). Only the 12 known kafka test_topics cases are red.

## Step 2

tests/bootstrap/test_service_conformance.py, tests/build/test_import_roots.py, tests/deploy/test_env_contract.py: 49 passed, 1 xfailed. Lint (ruff E9,F63,F7,F82, anaconda ruff 0.6.9): All checks passed.

## Step 3: ci-local stages (run in order as commands)

| Stage | Result |
|---|---|
| lint | PASS |
| unit | PASS except the 12 baseline kafka cases |
| contract / clinical | exit 0 (suites MISSING) |
| images | RED then fixed. Makefile and ci_local called the nonexistent scripts.build.images. RED: tests/scripts/test_w0_local_entrypoints.py 3 failed. GREEN: 3 passed (tests/scripts 18 passed + 1 skipped). Real run: scripts/ci/build_images.py --all --tag ci --probe, exit 0, all 10 images. Commit 7176421 |
| tracked-closure (--report) | exit 0 |
| realinfra (--require-realinfra) | tests/realinfra 37 passed, tests/migrations 68 passed, tests/build 2 failed (test_mvp_359 lock advisories dnspython GHSA-vvf4-9x63-8xvv, opentelemetry-api GHSA-mf5v-w3w3-f6cg), 16 passed, 1 skipped. Owned by WS-01 (D-16), advisory feed drift. |
| e2e dev | 38 passed, 8 xfailed (crisis, hop provision_clinician, W1) |
| e2e strict | 48 passed |

Docker afterwards: no solace-e2e containers; volumes 70 as before; the :ci image tags created here were removed by tag (solace/user-service:ci pre-existed and was kept).

## Verdict

Pass: nothing outside baseline red fails, except the realinfra lock-advisory pair, an owner-ledgered feed drift (Ruling, WS-01). Review package: docs/mvp/exec-state/reviews/W0-barrier-B4-20261124-080918.diff.md
