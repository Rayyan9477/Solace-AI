# W0-close review fix round 1

Branch w0-barrier, base afcfdf3. Five findings survived the skeptic majority. All five are fixed; none is parked.

| Finding | Severity | Result | Commit |
|---|---|---|---|
| LA-SEC-1 | minor (security) | fixed | 9cbec2b |
| BF-ROO-1 | important (security; skeptics: nit) | fixed (D-16 bump) | 2dc0dec |
| BF-ROO-2 | minor | fixed | 05e1e8c |
| BF-ROO-3 | minor | fixed | 05e1e8c |
| DW-DOC-1 | minor (security register entry) | fixed (register wording + test); Covers change handed to the docs stage | e7ab28b |

Interpreter: D:/tmp/mvp-exec/venv. Unit command as in the stage brief (`-o "addopts=" -p no:logging`).

## LA-SEC-1: the strict weaviate-client refusal never reached the boot

Root cause: `MemoryService.initialize` (services/memory_service/src/domain/service.py) caught every `WeaviateRepository.initialize` error, logged `weaviate_init_failed`, set `_weaviate_repo = None` and finished initialized. The strict RuntimeError from WS-04.F2 therefore died there.

Fix (option 1 of the finding, Ruling in the ledger): in strict (own `MEMORY_ENVIRONMENT` OR bare `ENVIRONMENT`, via `is_strict_environment`), `MemoryService.initialize` logs `weaviate_init_failed` at error and re-raises, so `main.lifespan` fails the boot. Dev and test still degrade to keyword/BM25. The repository still logs connection failures itself, so in practice only the missing-package guard reaches the service.

Tests (services/memory_service/tests/test_ws04_f2_weaviate_client_strict.py, through `MemoryService.initialize` with the real `WeaviateRepository` and `sys.modules['weaviate']=None`):
- strict x4 (staging/-, production/-, development/staging, test/production): the boot refuses and `_initialized` stays False;
- strict x4 with a repository that raises: not swallowed;
- dev and test x2 each: the boot completes (degraded).

RED (before the fix): `8 failed, 10 passed`. Each failure was `Failed: DID NOT RAISE <class 'RuntimeError'>`, after the log line `weaviate_init_failed error='weaviate-client package required in staging and production' hint='Vector search unavailable'`.
GREEN: `18 passed`. Regression: services/memory_service/tests `504 passed` (B4: 492; +12 new cases).

## BF-ROO-1: the barrier was called Pass with two security advisories red against an open "before green" hand-off

Fix: closed the gate itself, with a D-16 security bump, not an acceptance. Acceptances need the owner. Both advisories have same-major patch fixes:
- dnspython 2.9.0 -> 2.9.2 (GHSA-vvf4-9x63-8xvv / CVE-2026-128089). A WKS record in zone-file text can exhaust memory. No service imports `dns`, and email-validator runs with `check_deliverability=False`, so it is not reachable today.
- opentelemetry-api 1.45.1 -> 1.45.2 (GHSA-mf5v-w3w3-f6cg / CVE-2026-126360). ReDoS in `TraceContextTextMapPropagator.extract` on a crafted `traceparent`. It is reachable on inbound requests wherever tracing extracts context. sdk and exporter-otlp move to 1.45.2. The proto, otlp-proto-*, semantic-conventions, exporter-http-transport and exporter-otlp-common packages follow (0.66b2 for the betas).

Compiles use the WS-01.F1 and F2 convention: `--no-annotate`, with `--custom-compile-command` keeping the IF-21 header line exact. The dev lock is compiled with `-c` on the runtime pins (`lock_delta.py --pins-only`). Both lock diffs are exactly these 11 packages (66 lines each, versions and hashes only), and no lock names a local path.

The venv sync list (`lock_delta.py --lock requirements-dev.lock --write-install`) was `missing=0 mismatched=11`. It was installed into D:/tmp/mvp-exec/venv only, with `uv pip install --require-hashes --no-deps`. Afterwards: `missing=0 mismatched=0`.

RED (realinfra, `SOLACE_REQUIRE_REALINFRA=1`, before the bump): both test_mvp_359 cases failed. Each reported `dnspython==2.9.0 GHSA-vvf4-9x63-8xvv: neither fixed nor accepted` and `opentelemetry-api==1.45.1 GHSA-mf5v-w3w3-f6cg: neither fixed nor accepted`, in requirements.lock and in requirements-dev.lock.
GREEN: tests/build/test_mvp_359_lock_has_no_unaccepted_advisories.py + test_mvp_005_dependency_lock.py with `SOLACE_REQUIRE_REALINFRA=1`: `26 passed`.

Images: `scripts/ci/build_images.py --all --tag w0fix1 --probe` (the ci-local images stage command with its own tag) exited 0. All 10 images were built from the bumped lock (`uv pip sync --require-hashes`), and every CMD module was imported inside its image. In solace/memory-service:w0fix1: dnspython 2.9.2, opentelemetry-api 1.45.2, sdk 1.45.2, semantic-conventions 0.66b2. The w0fix1 tags were removed by tag. Docker afterwards: 31 containers and 70 volumes, as before.

B4 report: an addendum says the barrier was not green at B4, and that this round closes the pair. The open hand-off in `ledger/handoffs.md` ("WS-01 lock owner (D-16, from lane A) ...") is now met; the docs stage ticks it with this evidence.

## BF-ROO-2: the ci_local target check matched nothing

Fix: tests/scripts/test_w0_local_entrypoints.py imports `scripts.ci.ci_local` and walks the real `STAGES`. Every `*.py` argument must be an existing file, and every `-m scripts.X` must resolve to scripts/X.py. The check must see at least one target per script stage. A self-test feeds it a missing absolute script, a missing relative script and a missing module, and it reports all three. The Makefile check now also verifies every `scripts/...py` path.

Mutation proof: in a scratch copy, ci_local's images stage was pointed at `build_imagez_does_not_exist.py`. The run gave `2 failed, 3 passed`. `test_ci_local_stage_targets_exist` failed with `images: script ...\scripts\ci\build_imagez_does_not_exist.py does not exist`, and `test_ci_local_images_stage_runs_ci_image_probe` failed on the script name. Before this round, all 3 old tests passed on the same mutation (review evidence).

## BF-ROO-3: the ci-local images stage skipped CI's in-image probe

Fix: `scripts/ci/ci_local.py` runs `build_images.py --all --tag ci --probe`. A new test, `test_ci_local_images_stage_runs_ci_image_probe`, reads CI's images line from `.github/workflows/ci.yml` and requires the same `--tag ci` and `--probe` in the ci_local stage. The Makefile images target stays build-only, as the finding allows.
RED: `1 failed, 4 passed`. The failure was `test_ci_local_images_stage_runs_ci_image_probe`, with `'--probe' in [... 'build_images.py', '--all', '--tag', 'ci']` false. GREEN: tests/scripts `22 passed, 1 skipped` (including the DW-DOC-1 test). The stage command was run for real, with its own tag (see BF-ROO-1).

## DW-DOC-1: the register called MVP-364 to MVP-366 fixed while their remaining tasks are todo

Fix: in docs/MVP-ISSUE-REGISTER.md section 4.3, each of the three entries now says "partly fixed in the W0 close lane by WS-xx.F" and names the open remainder:
- MVP-364: the audit half, WS-04.10;
- MVP-365: moving the 8 pinned crisis entries once WS-14.9 lands the CLI, WS-05.8;
- MVP-366: WS-13.8, the entry's other owning task.

MVP-364 also records that the memory refusal now reaches the boot (LA-SEC-1).

Test: tests/scripts/test_register_fix_claims.py fails when a 4.3 sentence claims "fixed in/by" a task that TRACKER does not show `done`. A "partly fixed" sentence is not a claim. The self-test pins the parser on MVP-361.
RED: `register entries said to be fixed by tasks that are not done: {'MVP-364': [('WS-04.10', 'todo')], 'MVP-365': [('WS-05.8', 'todo')], 'MVP-366': [('WS-13.8', 'todo')]}`. GREEN: `2 passed`.

Not done here: the derived count still counts the three entries as fixed, because only the F tasks list them in **Covers**, and plan files belong to the docs stage. This is a hand-off to the docs stage (ledger Ruling). It should add MVP-364 to WS-04.10's Covers, MVP-365 to WS-05.8's and MVP-366 to WS-13.8's, then run `mvp_docs.py refresh`. The fixed count drops by 3.

## Regression

Unit tier: `scripts/ci/run_suites.py --tier unit`, with the venv synced to the bumped dev lock. Exit 1, caused only by the 12 baseline `tests/solace_infrastructure/kafka/test_topics.py` cases. Every other suite is at or above B4:

| Suite | Result | B4 |
|---|---|---|
| tests/test_syntax_and_imports.py | 55 passed | 55 |
| tests/alignment | 16 passed | 16 |
| tests/solace_common | 170 passed | 170 |
| tests/solace_events | 243 passed | 243 |
| tests/solace_security | 351 passed | 351 |
| tests/solace_infrastructure | 936 passed, 12 failed (baseline kafka) | 936 / 12 |
| tests/solace_testing | 266 passed | 266 |
| tests/integration | 236 passed | 236 |
| tests/security | 10 passed | 10 |
| tests/migrations | 71 passed, 1 skipped, 68 deselected | same |
| tests/scripts | 22 passed, 1 skipped | 18 + 1 (4 new cases) |
| tests/system | 6 passed | 6 |
| tests/config_service | 231 passed | 231 |
| tests/bootstrap | 310 passed, 1 deselected | 310 |
| tests/build | 273 passed, 19 deselected, 1 xfailed (incl. test_d16_local_interpreter_matches_dev_lock) | 273 |
| tests/deploy | 125 passed, 19 deselected | 125 |
| tests/contracts, tests/clinical | MISSING (do not exist yet) | same |
| infrastructure/api_gateway/tests | 64 passed | 64 |
| services/analytics-service | 198 passed | 198 |
| services/diagnosis_service | 312 passed | 312 |
| services/memory_service | 504 passed | 492 (12 new cases) |
| services/notification-service | 168 passed, 7 skipped | same |
| services/orchestrator_service | 320 passed, 1 xfailed | same |
| services/personality_service | 377 passed | 377 |
| services/safety_service | 508 passed, 1 xpassed | same |
| services/therapy_service | 526 passed, 1 xfailed | same |
| services/user-service | 378 passed | 378 |

Realinfra tier for tests/build: `run_suites.py --tier realinfra --suite tests/build --require-realinfra`, exit 0, `18 passed, 1 skipped` (B4: 2 failed, 16 passed, 1 skipped). The lock-advisory pair is no longer red, so the "known baseline red" line for the 2 test_mvp_359 cases can be removed. That line is in CLAUDE.md and belongs to the docs or finalize stage.

Lint: ruff `E9,F63,F7,F82` is clean on the touched Python files.

## Rulings (ledger)

- LA-SEC-1, option 1: re-raise in strict; dev and test degrade.
- BF-ROO-1: a bump, not an acceptance; it closes the WS-01 lock hand-off, which the docs stage ticks.
- BF-ROO-2 and BF-ROO-3: the test design; the Makefile stays build-only.
- ci_local's e2e stages still lack `--require-realinfra`, which CI has had since WS-05.F1. That is not a surviving finding, so it is handed to the next review round or WS-05.
- DW-DOC-1: the register is reworded here, and the Covers change goes to the docs stage.

## Files

services/memory_service/src/domain/service.py, services/memory_service/tests/test_ws04_f2_weaviate_client_strict.py, requirements.in, requirements.lock, requirements-dev.lock, docs/mvp/exec-state/reports/W0-barrier-B4-report.md, scripts/ci/ci_local.py, tests/scripts/test_w0_local_entrypoints.py, tests/scripts/test_register_fix_claims.py, docs/MVP-ISSUE-REGISTER.md, this report and the review package. Outside the repo, D:/tmp/mvp-exec/venv was synced to the bumped dev lock (11 packages).

Review package: docs/mvp/exec-state/reviews/W0-close-fix-1-20261124-085404.diff.md (afcfdf3..e7ab28b, restricted to the files above).
