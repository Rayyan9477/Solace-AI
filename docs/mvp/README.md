# Solace-AI MVP program: tracking hub

This folder is the persistent record of the MVP remediation program. The program makes the Solace-AI backend releasable as the confirmed "Pilot + enterprise basics" MVP: a single-host pilot with real users, every message safety-screened, crisis alerts that reach an on-call clinician, encrypted per-user PHI and working deletion, plus backups with a tested restore, MFA, an audit trail and per-environment secrets.

The **method** is in the [master plan](../superpowers/plans/2026-10-18-mvp-remediation.md). **What is decided, what is done and what is still open** lives here.

## Current state

*Edit this section by hand when the phase changes.*

- **Phase:** **W0 barrier passed (2026-11-24).** W1 has not started. Work is on branch `w0-barrier`, merged through a PR (D-17 is superseded).
- **What passed:**
  - Clean-export build of all 10 images (27/27 steps), real-Postgres migration chain, live harness self-test.
  - Live boot smoke in dev, strict production and strict staging; register and login through the edge.
  - Full unit regression: green except the 12 known kafka `test_topics` cases. Lint, conformance and import-root checks pass.
  - Real-infra tier and e2e tiers green: dev e2e 38 passed + 8 xfailed, strict e2e 48 passed.
  - Barrier tree `fe7022d`, zip `D:\tmp\mvp-exec\barriers\W0-fe7022d.zip` (copy it off-machine, X-12).
  - Fixed on the way: edge `/metrics`, `/ready`, `/health` variants, strict-overlay `ENVIRONMENT`, strict CORS, staging-strict guards, CI e2e real-infra, crisis-consumer lifespan tests, a strict memory boot refusal, a D-16 lock bump, and ci-local images.
- **Open from W0:** the crisis alert E2E stops at `provision_clinician` (XFAIL, WS-14.9 in W1), so the crisis hops from REST to clinician page have not run live. Residual items are routed to W1 tasks in `exec-state/ledger/handoffs.md`.
- **Next:** W1 round 1 per `exec-state/plan/W1-round-1.md`: WS-06.3, WS-12.2, WS-12.4, WS-07.3, WS-03.6, WS-09.3. Briefs are in `exec-state/briefs/`; preflight rulings are in `exec-state/plan/W1-preflight-scan.md`.
- **Owner actions:**
  - X-15: name a clinician for lexicon v1 and the phrase set (W1 entry criterion).
  - X-04: clinical approver and sign-off for D-05, D-06, D-22, D-28, D-29.
  - X-05: at least two on-call clinicians (before go-live).
  - X-14: allow `pkg-containers.githubusercontent.com` for cloud egress (cloud sessions only).
  - X-12: copy the barrier zip off-machine.
  - Review and merge the W0 PR; edge monitors must poll `/live`, since `/ready` and `/health` are 404 at the edge.
  - Full list: [OPEN-ITEMS.md](OPEN-ITEMS.md).
## Progress

<!-- mvp:summary:start -->

- Tasks done: **65 / 334** (active 0, blocked 0, deferred 0).
- Register entries fixed: **60 / 351**.
- Per-wave and per-workstream counts: [TRACKER.md](TRACKER.md).

<!-- mvp:summary:end -->

## Documents

| File | Holds | Updated by | When |
|---|---|---|---|
| [DECISIONS.md](DECISIONS.md) | All 31 owner decisions: the answer, the original recommendation, the tasks gated on it, the plan impact, and sign-off fields | the owner and the controller | when a decision or sign-off changes |
| [TRACKER.md](TRACKER.md) | One row per task with its Status, grouped by wave and phase; the barrier log | the controller (via `tools/mvp_docs.py`) | when a task changes state; at each barrier |
| [OPEN-ITEMS.md](OPEN-ITEMS.md) | Owner actions (X-xx), sign-offs and risks (R-xx) | the owner and the controller | when an item changes |
| [LOG.md](LOG.md) | Dated history: decisions, plan changes, barriers | the controller | at every notable event |
| [../MVP-ISSUE-REGISTER.md](../MVP-ISSUE-REGISTER.md) | The binding issue list (356 entries); its status column is derived from TRACKER.md | the controller (re-rendered) | at each barrier |
| [Master plan](../superpowers/plans/2026-10-18-mvp-remediation.md) | Scope, decisions snapshot, waves, execution protocol, risks | re-rendered from the plan sources | when the plans change |
| [Workstream plans](../superpowers/plans/2026-10-18-mvp-remediation/) | The 18 plans with every task's failing test, files and interfaces | planners | when a task changes |

## How to update

- **A task changes state:** `python docs/mvp/tools/mvp_docs.py set WS-NN.K <status> --note "<tree id or reason>"`. The statuses are `todo`, `in-progress`, `review`, `fixing`, `done`, `blocked` and `deferred`. The command recounts this page and the tracker.
- **A wave barrier passes:**
  1. Fill the row in TRACKER.md's barrier log: tree id, zip, test counts, register delta.
  2. Add a LOG.md entry.
  3. Re-render the register.
  4. Copy the barrier zip off the machine (commits on the branch are also durable).
- **A decision changes:** edit its entry and history in DECISIONS.md, then add a LOG.md line. The controller records a ledger Ruling and updates the affected tasks before they run.
- **An owner action or sign-off completes:** set its Status, date and evidence in OPEN-ITEMS.md (and the sign-off field in DECISIONS.md), then add a LOG.md line.
- **The plans change** (for example, new fix tasks from a bug-finding round): `python docs/mvp/tools/mvp_docs.py build`. This adds the new rows as `todo` and keeps every existing status. `check` reports drift between the plans and the tracker.

The interpreter for these commands is the program venv `D:\tmp\mvp-exec\venv\Scripts\python.exe` once it exists. Until then use `C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe`; the tools use the standard library only.

## Where the working state lives

- **In the repo (persistent):** this folder, the plans and the register.
- **Outside the repo (`D:\tmp`, not persistent):**
  - the execution ledger, briefs, reports and checkpoint log (`D:\tmp\mvp-exec\`);
  - the program venv;
  - the planning generators (`D:\tmp\mvp-plan\`).
- **Backups in the repo:** copies of the execution tools and the planning generators are kept in [tools/exec/](tools/exec/) and [tools/planning/](tools/planning/). If `D:\tmp` is lost, copy them back. The planning scripts expect `D:\tmp\mvp-plan`, and the execution tools expect `D:\tmp\mvp-exec\tools`.
