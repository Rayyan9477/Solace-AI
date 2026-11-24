# W1-prep report

Stage: W1 entry preparation (docs and ledger only, no code, no pytest). Branch w0-barrier, commit 620c09c on base a25aebd.

## 1. Decisions (ledgered)
All twelve were answered by the owner on 2026-10-19 (DECISIONS.md). Confirmed: D-03, D-11, D-12, D-26, D-27 (default accepted), D-04 and D-29 (changed from default; X-02 BAA done 2026-11-02). Default accepted with sign-off pending, flagged: D-05, D-06, D-22, D-28 (clinician sign-off X-04, X-15 open). D-13 confirmed, names of on-call clinicians (X-05) open until go-live.

## 2. Interfaces
IF-04, 05, 06, 07, 08, 09, 10, 12, 15, 20 are each fully written in 00-interfaces-and-ownership.md with signatures, owner and consumers. Ledgered "frozen". No gap found.

## 3. Phase order
Ledgered as in the plan (A1, A2, B). pick_ready knows only A and B; A1/A2 are advisory (R-1).

## 4. Round 1
docs/mvp/exec-state/plan/W1-round-1.md: WS-06.3, WS-12.2, WS-12.4 (RI), WS-07.3, WS-03.6 (RI), WS-09.3. Mutex check by script: disjoint. Not dispatched until finalize ledgers the W0 exit (R-13).

## 5. Briefs
34 briefs extracted (29 phase A1 tasks and WS-12.4, 14.4, 04.6, 04.7, 13.3). Addenda appended to WS-04.6, 05.8, 05.9, 05.21, 09.3.

## 6. Pre-flight scan (docs/mvp/exec-state/plan/W1-preflight-scan.md)
169 pair rows (shared files or interfaces) and 69 task rows. Findings that change dispatch:
- WS-04.6 has an undeclared dependency on WS-06.8, WS-07.7 and WS-07.3 (held).
- WS-05.8 and WS-05.21 depend on W0 tasks the tracker shows blocked; WS-05.21 also consumes WS-03.19, 09.18, 09.4 and 07.7 outputs it does not list.
- WS-05.9 sets Alertmanager env values (FROM address, no SMTP_REQUIRE_TLS=false) that disagree with WS-04 row 1693 and WS-04.7's e2e; addendum fixes it.
- pick_ready lists 9 phase-B tasks FREE because phase-A deps are omitted from the tracker; B stays closed until A is done.
- WS-13.13 verifies producers (10.3, 12.8) it does not depend on.
- WS-10.4 covers bug round B-6 (owner asked for it first) but the plan carries no note.
- Plans and briefs still cite checkpoint.ps1 and D:\tmp\mvp-exec\tools; mvp_exec.py overrides.
Limits: the per-task automated checks (Modify targets exist, Create targets absent, tests referenced, RED present, imports resolve, header deps against tracker) were run on all 69; hand review of the full task text went deepest on the round-1 tasks and the flagged ones. The other rows say "no finding" on the strength of those checks.

## 7. Rulings
13 rulings (R-1 to R-13) in the scan and the ledger. Docs lane follow-ups (not done here, I may not edit plans): add WS-06.8, 07.7, 07.3 to WS-04.6 Depends on; add WS-03.19, 09.18, 09.4, 07.7 to WS-05.21; add a B-6 note to WS-10.4; align WS-05.9 and WS-04 Alertmanager values.

## 8. Owner actions that block W1
- X-15: name a clinician to review lexicon v1 and the phrase set (W1 entry criterion; builds can start on the default).
- X-04: clinical approver and sign-off for D-05, D-06, D-22, D-28, D-29.
- X-05: two on-call clinicians and the fallback mailbox (before go-live, not before W1 builds).
- Confirm W0 exit: tracker rows WS-05.4, 05.5 and 05.6 still read blocked although B1 ran the live self-test (finalize's job).
- X-14 (cloud egress) only for cloud sessions.
