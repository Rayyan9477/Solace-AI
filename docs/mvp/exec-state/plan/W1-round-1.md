# W1 round 1

Computed 2026-11-24 with `pick_ready.py --wave 1 --phase A` (tracker and DAG at the W1-prep commit; no task running). Rules applied: plan section 5.3 (at most 6 implementers, at most 2 real-infra, mutex keys disjoint, longest chain first, then safety-critical, security-critical, standard) and the preflight rulings R-1 to R-13 in `W1-preflight-scan.md`.

## Ready set (phase A, all FREE per pick_ready)

WS-06.3 (h8), WS-09.3 (h4), WS-12.2 (h3), WS-12.4 (h2, RI), WS-07.6 (h2), WS-07.3 (h2), WS-07.2 (h2), WS-05.8 (h2, RI), WS-05.21 (h2), WS-03.6 (h2, RI), WS-14.4 (h1, RI), WS-04.6 (h1, RI). 22 more wait on dependencies.

Removed from the ready set before choosing:
- WS-05.8 and WS-05.21: they depend on W0 tasks WS-05.6 and WS-05.4, still blocked in the tracker (R-5, R-13).
- WS-04.6: undeclared dependency on WS-06.8, WS-07.7 and WS-07.3 (R-3).
- WS-07.2: shares auth.py with WS-07.3 (R-4).
- WS-14.4: shares the migrations key with WS-12.4 (R-6).
- WS-07.6: held back only for the 6-lane cap.

## Round 1 (6 tasks, mutex-disjoint, 2 real-infra)

| # | Task | Model | Review | Chain h | RI | Why now | Brief |
|---|---|---|---|---|---|---|---|
| 1 | WS-06.3 Consumer retry, durable DLQ, idempotency after success | Opus | safety-critical | 8 | no | Longest chain (feeds 06.4 to 06.14); crisis events must not be lost (bug round B-4) | docs/mvp/exec-state/briefs/WS-06.3-brief.md |
| 2 | WS-12.2 solace_clinical core (IF-07: normalize, fuse_risk, resources) | Opus | safety-critical | 3 | no | Producer of IF-07 for WS-05.7, 12.3, 10.2, 13.6 | WS-12.2-brief.md |
| 3 | WS-12.4 Slot 014 escalation decision columns | Opus | safety-critical | 2 | yes (1 of 2) | One auto decision per turn; unblocks WS-13.3 and the slot 015 lane | WS-12.4-brief.md |
| 4 | WS-07.3 Service-token model, obo tokens, outbound headers (IF-04) | Opus | security-critical | 2 | no | Unblocks 07.4 and the WS-10.4 fix (B-6, always-401 safety client), 11.2, 13.5, 14.6, 14.8 | WS-07.3-brief.md |
| 5 | WS-03.6 Notification delivery env contract, ack route, W1/W2 env sync | Opus | security-critical | 2 | yes (2 of 2) | Gates WS-04.7 and 03.19; carries bug round C-5 | WS-03.6-brief.md |
| 6 | WS-09.3 LLMUnavailableError and LLMResult (IF-15) | Sonnet | standard | 4 | no | Gates 09.18 (Bedrock), 09.4, 05.20, 03.19 | WS-09.3-brief.md |

Key check (script over schedule_dag.json mutex lists): no key is shared between any two of the six. WS-09.3's Step 5 regression runs tests/integration while WS-07.3 edits tests/integration/test_service_integration.py: not a mutex key of 09.3, so the cross-talk rule of 5.3 applies (wait for the other lane's checkpoint and re-run before treating a failure as your own) and the named expected failures of R-8.

Common: implementer rules `briefs/phaseB-common.md` are for service tasks only; these are platform tasks, so the brief alone plus the controller notes (checkpoint via `mvp_exec.py checkpoint`, R-12). Each implementer shows a real RED, writes `reports/<task>-report.md`, makes no git writes.

## Next dispatches (as lanes free)

1. WS-07.2 once WS-07.3 is done; then WS-07.4, then WS-07.5. WS-07.6 any time a slot is free, then WS-07.7 in a suite gap.
2. WS-12.3 and WS-05.7 once WS-12.2 is done (12.3 also needs the clinician reviewer named, X-15, only for sign-off, not to build).
3. WS-06.4 then WS-06.5, 06.6, 06.7, 06.8, 06.9, 06.10, 06.14 (the chain; 06.4 and 06.5 share suites so they alternate).
4. WS-13.3 after WS-12.4; WS-14.4 in the next migrations gap.
5. WS-09.18 after 09.3; then 09.4, 09.5, WS-03.19, WS-05.20 (after 07.3, R-7).
6. WS-04.7 after 03.6 (needs a real-infra slot). WS-04.6 only after 06.8, 07.7, 07.3 (R-3).
7. WS-05.21, 05.8, 05.9, 05.22 after the W0 harness tasks close and their producers land (R-5, R-13).

Briefs extracted: all 29 phase A1 tasks plus WS-12.4, 14.4, 04.6, 04.7 and 13.3 (A2 and the first lane) under docs/mvp/exec-state/briefs/.
