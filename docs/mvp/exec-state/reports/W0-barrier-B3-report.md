# W0 barrier, stage B3: crisis alert E2E in the dev profile (the WS-05.6 live run)

Branch `w0-barrier`, run on 2026-11-24 on local Docker 29.7.2 with compose v5.4.0. The code under test is `fe76512` (tree `7ae8f3f`).

This was WS-05.6's first live run at the barrier. Its Steps 2 and 4 had been blocked on cloud egress since 2026-11-03. Both runs matched the manifest exactly:
- The raw run failed all 8 tests at the recorded hop, `provision_clinician`, with the recorded cause.
- The barrier run at wave 0 exited 0 with 8 XFAIL at that hop.

No hop differed, so B3 changed no code, test, manifest, compose file or Dockerfile. There is no RED to show, because there was no W0 defect to fix.

| Item | Command (summary) | Exit | Result |
|---|---|---|---|
| crisis-e2e-run | `pytest tests/e2e/test_crisis_alert_e2e.py`, dev, `SOLACE_MANIFEST_IGNORE=1` | 1 (expected) | 8 errors, all in module setup at hop `provision_clinician` |
| hops-match | the observed hop and cause, compared with `tests/e2e/expected_failures.yaml` | - | 8 of 8 match (hop and pinned `cause`) |
| barrier-run | same file, dev, manifest applied, `SOLACE_BARRIER_WAVE=0` | 0 | 8 xfailed at `provision_clinician` (target wave 1) |

Common environment:
- Every pytest run used `PYTHONPATH=D:/Repo/Contextual-Chatbot;D:/Repo/Contextual-Chatbot/src`, `ENVIRONMENT=test`, `TELEMETRY_ENABLED=false`, the interpreter `D:/tmp/mvp-exec/venv/Scripts/python.exe` and the flags `-o "addopts=" -p no:logging -q -rA`.
- The E2E settings were `SOLACE_REQUIRE_REALINFRA=1`, `SOLACE_E2E_PROFILE=dev` and `SOLACE_E2E_LANE=b3` (project `solace-e2e-b3`). The lane was not `a`, because a parallel stage may hold lane a (Ruling).
- The runs were one at a time. The raw logs are in `D:/tmp/mvp-exec/scratch-b3/`: `step1-dev-ignore.log` and `step3-dev-barrier.log`.

## 1. Raw run (item crisis-e2e-run, WS-05.6 Step 2)

```
$env:SOLACE_MANIFEST_IGNORE="1"
D:/tmp/mvp-exec/venv/Scripts/python.exe -m pytest tests/e2e/test_crisis_alert_e2e.py -o "addopts=" -p no:logging -q -rA
```

Result: exit 1, `8 errors in 49.72s`. The plugin printed `SOLACE_MANIFEST_IGNORE is set: 1 manifest(s) not applied, raw results shown`.

The stack came up fully. `e2e_stack_up` logged `down=[]` and `environment=development`, so all 9 MVP services answered `/live` 200.

The module's autouse `oncall` fixture failed in setup. So each of the 8 tests is an ERROR at setup, with the same hop and cause (8 of 8 `HopFailure` lines are identical):

```
E   RuntimeError: scripts.provision_clinician exited 1:
E   /opt/venv/bin/python: No module named scripts.provision_clinician
src\solace_testing\e2e.py:924: RuntimeError
...
E   solace_testing.e2e_manifest.HopFailure: hop 'provision_clinician' failed: RuntimeError: scripts.provision_clinician exited 1:
E   /opt/venv/bin/python: No module named scripts.provision_clinician
```

| Test | Observed hop |
|---|---|
| `test_rest_crisis_message_returns_resources_within_3s[abuse]` | `provision_clinician` |
| `test_rest_crisis_message_returns_resources_within_3s[curly_apostrophe]` | `provision_clinician` |
| `test_rest_crisis_message_returns_resources_within_3s[homicidal]` | `provision_clinician` |
| `test_rest_crisis_message_returns_resources_within_3s[indirect]` | `provision_clinician` |
| `test_ws_crisis_message_sends_safety_alert_before_reply[curly_apostrophe]` | `provision_clinician` |
| `test_ws_crisis_message_sends_safety_alert_before_reply[homicidal]` | `provision_clinician` |
| `test_crisis_pages_the_oncall_clinician_once_without_message_text` | `provision_clinician` |
| `test_ack_link_acknowledges_the_episode` | `provision_clinician` |

The 10 warnings are `PytestConfigWarning: Unknown config option: log_*` (caused by `-p no:logging`) and the pytest-asyncio loop-scope deprecation notice. Neither is related to the stack.

## 2. Comparison with the manifest (item hops-match, WS-05.6 Step 3)

All 8 entries in `tests/e2e/expected_failures.yaml` have:
- `expected_failing_hop: provision_clinician`;
- `cause: 'No module named scripts\.provision_clinician'`, pinned by lane A (WS-05.F1, CR-3) from its live run;
- `target_wave: 1` and `profiles: [dev]`.

The observed hop and the observed cause match every entry. No hop differs, so there is nothing to classify as a W0 defect or to move.

Why this hop is a documented W1 gap and not a W0 defect:
- The IF-08 CLI `scripts/provision_clinician.py` does not exist in the repository (`git ls-files scripts` has no match). It is created by WS-14.9, "Clinician provisioning CLI and the one-time account-setup link", which is wave 1, depends on WS-14.6 and WS-14.8, and is `todo` in TRACKER.md.
- The failure is the CLI's absence, not a broken stack. The stack booted fully (`down=[]`), `docker compose exec` reached user-service, and the interpreter ran but could not find the module.
- The WS-05.6 brief (Step 2) names this as the W0 hop: "`scripts.provision_clinician` does not exist, so the module's autouse fixture fails in setup and every test of the module stops at this hop in W0".
- When WS-14.9 lands, WS-05.8 moves the 8 entries to the newly observed hops. That hand-off is already open in `ledger/handoffs.md`.

Consequence, kept visible because the crisis path is safety-critical:
- In W0, none of the downstream crisis hops runs: `rest_crisis_object`, `ws_connect`, `ws_safety_alert`, `clinician_page` and `ack_link`.
- The brief expects them to fail in W1 until their owners land: WS-10/WS-11 for the IF-12 crisis object and the `safety.alert` frame, and WS-13 for the page and the acknowledgement.
- This run shows only that the suite runs live and stops where recorded. It says nothing about those later hops.
- The strict profile is not run, because the module is `profile("dev")`. Widening it to strict is an existing hand-off to WS-05.8 and the W1 barrier.

Ruling ledgered.

## 3. Barrier run (item barrier-run, WS-05.6 Step 4)

```
Remove-Item Env:SOLACE_MANIFEST_IGNORE; $env:SOLACE_BARRIER_WAVE="0"
D:/tmp/mvp-exec/venv/Scripts/python.exe -m pytest tests/e2e/test_crisis_alert_e2e.py -o "addopts=" -p no:logging -q -rA
```

Result: exit 0, `8 xfailed, 10 warnings in 47.43s`. The plugin printed `8 active entries applied (profile dev, barrier wave 0)`. Each test is reported as:

```
XFAIL tests/e2e/test_crisis_alert_e2e.py::<test> - failed at the expected hop 'provision_clinician' (target wave 1): MVP-039 W0: scripts.provision_clinician does not exist, ...
```

The run had:
- no due entry, because every entry targets wave 1, above barrier 0;
- no stale entry;
- no skip, which at a barrier would fail;
- no cause mismatch.

This is the W0 exit criterion: "runs in the dev profile and fails only at the hops recorded".

## 4. Not re-run, and the checkpoint

B3 changed nothing, so the WS-05.6 Step 5 regression (`tests/e2e`, dev) was not repeated. Lane A ran it live and got 38 passed and 8 xfailed. Since lane A's commit `a30b6e2`, the only changes outside `docs/` are `CLAUDE.md` and one notification test file (`git diff --stat a30b6e2 HEAD`), so no code it ran has changed. The full barrier regression is a separate stage.

The Step 6 tree checkpoint is replaced by this report's commit, as in B1. Ruling ledgered.

## 5. Clean-up

- Both runs tore their own stack down. Afterwards there was no `solace-e2e-*` container or network, and `%TEMP%/solace-e2e/solace-e2e-b3` does not exist.
- The volume count is 70, the same as before B3.
- This stage's `solace-e2e-b3-fake-llm:latest` image was removed by tag (no `-f`).
- Lane A's `solace-e2e-a-fake-llm:latest` was left in place. The shared `solace/*:dev` tags were re-pointed by `up --build`, as in B1 and B2. Nothing was pruned.

## 6. Hand-offs (for the docs stage)

- **TRACKER.md:** the WS-05.6 row is still `blocked` ("live crisis E2E waits on egress"). With this report, Steps 2 and 4 are done live, so the row can move to `done`. The manifest is unchanged.
- **No new hand-off.** The open ones still apply:
  - WS-05.8 moves the 8 entries when WS-14.9 lands the CLI;
  - WS-05.8 and the W1 barrier widen the suite to `profile('dev', 'strict')`.
