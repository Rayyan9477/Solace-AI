# W0-close docs stage

Branch `w0-barrier`, base `44ea239`. Docs only: no code, tests, compose files or Caddyfile changed, and no pytest was run. `mvp_docs.py build`, `refresh` and `check` ran clean (334 tasks; no invalid status; none missing from tracker or plans).

Commits:

| Commit | Content |
|---|---|
| `2180734` | Plan sections WS-03.F1, WS-04.F2, WS-05.F1, WS-13.F1; register MVP-361 to MVP-366; owner notes in 17 task plans; TRACKER and README counts |
| `2b3a142` | `handoffs.md` rewritten and reconciled; `OPEN-ITEMS.md` |
| `e647c61` | `DECISIONS.md` D-17 history; `CLAUDE.md` known baseline red |

Review package: `docs/mvp/exec-state/reviews/W0-close-docs-20261124-065500.diff.md`.

## 1. Plan sections and register

The four fix tasks are plan tasks in the shape of WS-04.F1 and set to `done` in TRACKER with their commit ids (0efe9af, e715074, a30b6e2, ba7debb). They cover new register entries, added to sections 3.2, 3.3 and 4.3 of `docs/MVP-ISSUE-REGISTER.md`; the summary and cluster and area counts are bumped (362 open):

| Entry | Severity | Finding | Task |
|---|---|---|---|
| MVP-361 | high | Edge never-routed list matched exact paths; `/ready` and `/health` public (SEC-1) | WS-03.F1 |
| MVP-362 | medium | Strict overlay accepted any `ENVIRONMENT` (SEC-2) | WS-03.F1 |
| MVP-363 | medium | Strict CORS accepted http and localhost origins (SEC-3) | WS-04.F2 |
| MVP-364 | medium | Three strict-only guards ignored staging (SEC-4 non-audit; audit half WS-04.10) | WS-04.F2 |
| MVP-365 | high | CI e2e job did not require real infra; manifest entries had no cause pin (CR-4, CR-3) | WS-05.F1 |
| MVP-366 | medium | Strict crisis-consumer guard not proven through the lifespan (CR-1) | WS-13.F1 |

## 2. Ownerless deferred items (audit section 4)

Each is a "From the W0 review" note in the owner task's plan text (placed before `**Files:**`, so `extract-brief` carries it), a hand-off, and a ledger Ruling.

| Item | Owner | Why |
|---|---|---|
| revoke-token 503 for a garbage bearer | WS-08.4 | owns the 503 mapping in `api_deps.py` |
| `update_preferences` existence check as the caller | WS-14.5 | already edits every except path of the method |
| `KafkaEventBridge` partial-start leak, no-op `stop()`, poller not a runtime task | WS-06.8, with WS-06.9 | the plane is the one composition root |
| `/ready` and `/health` at the edge | done by WS-03.F1 | ticked in hand-offs |
| A-4 config_service echoes validator text | no task (accepted by scope) | config-service is not deployed and is frozen (D-19); risk R-23 |
| CR-2 readiness without poll progress | WS-13.8 | lane A parked it there |
| SEC-4 audit half | WS-04.10 | makes `AUDIT_HMAC_KEY` a required strict secret |
| B-1 not reproducing locally | WS-12.6 | reproduce first, close with evidence if environment-only |
| stale staging docs, `/live` for monitors | WS-03.15 | docs truth task |

No new F task was needed.

## 3. The 12 bug-round findings

A-3, B-2, B-3, B-4, B-5 and C-1 to C-7 are in `handoffs.md` (section "W0 bug-round findings") and as "From the W0 bug round" notes in WS-15.11, WS-18.7, WS-12.8, WS-06.3, WS-13.8, WS-06.4, WS-07.7, WS-12.11, WS-09.11, WS-04.10, WS-03.6, WS-13.7, WS-11.3, WS-18.18, WS-09.13 and WS-06.8. C-1 is ticked (WS-15.F1).

Limit: the hunter and skeptic outputs were never saved. Only the one-line ledger summaries exist. B-2 and B-3 use the detail of the WS-12.1 review hand-off. The others are rebuilt from the ledger line and the owner task's scope, and the note says so ("re-confirm in a RED test"). C-3 to C-7 are the least certain. Ledgered as a Ruling.

## 4. Hand-offs reconciled

The tick rule is now stated at the top of the file: `[x]` only when done or met, with evidence. The routed-only items (A-1, A-2, B-1 were ticked) are open again with their owner. The duplicate A-1 entry is merged. Newly ticked with evidence: the CORS, pull_policy, revoke-token catch and WS-17.2 gate items (checked in the 2026-11-07 audit), and the four lane A fixes. Everything else is open with its owner task id. The lock-advisory failure of `tests/build/test_mvp_359` is a new open item for the WS-01 lock owner.

## 5. Open items, decisions, CLAUDE.md

- `OPEN-ITEMS.md`: X-14 (cloud egress to `pkg-containers.githubusercontent.com`; local runs are not blocked) and X-15 (named clinician to review lexicon v1 and the phrase set, the W1 entry criterion) added. X-04 and X-05 stay open with a refreshed date. R-23 added for A-4.
- D-17 is superseded. `DECISIONS.md` has a history line and the summary row says so. The TRACKER barrier-log header no longer says there are no commits. X-10 is `done`; X-11, X-12, R-08 and R-18 are `n/a`.
- `CLAUDE.md` "Known baseline red": the kafka `test_topics.py` 12 cases; the 4 `tests/migrations` TLS cases (red when Docker is off); the 2 `test_mvp_359` lock-advisory cases (seen by lane A on 2026-11-24). The adopted-service conformance, import-root, lint and `/health` lines were dropped, because the regression and the audit show them green. "State at pause" and "Remaining W0" were not touched.

## Rulings (all in the ledger)

Ten Ruling lines: the F-task and register ids; owners for revoke-token, `update_preferences` and the bridge leak; A-4 accepted by scope; the rebuilt bug-round text; the tick rule; D-17 superseded; X-14 and X-15; the baseline-red list.

## Limits

- The four regression counts in `CLAUDE.md` come from `regression.txt` (2026-11-07) and the lane A report. I did not rerun them. The 4 `tests/migrations` failures are named by the task as TLS cases that need Docker; I did not open the failing test names.
- The statement that X-12 and X-11 are superseded assumes the owner's branch-and-PR workflow described in the task.
