# Solace-AI MVP program: tracking hub

This folder is the persistent record of the MVP remediation program. The program makes the Solace-AI backend releasable as the confirmed "Pilot + enterprise basics" MVP: a single-host pilot with real users, every message safety-screened, crisis alerts that reach an on-call clinician, encrypted per-user PHI and working deletion, plus backups with a tested restore, MFA, an audit trail and per-environment secrets.

The **method** is in the [master plan](../superpowers/plans/2026-10-18-mvp-remediation.md). **What is decided, what is done and what is still open** lives here.

## Current state

*Edit this section by hand when the phase changes.*

- **Phase:** **in execution since 2026-11-02** (the owner said "go"; the pace is deliberately slow, one task at a time with checks between tasks). W0 entry is in progress.
- **Done at W0 entry:**
  - entry tree `b309d50`, zip `D:\tmp\mvp-exec\barriers\W0-entry-b309d50.zip`;
  - program venv created;
  - aiokafka 0.12.0, langgraph-checkpoint-postgres 3.0.5, PyJWT 2.15.1 and orjson 3.13.0 installed into the venv only.
- **Next:** the venv baseline, then W0: build, migrate and boot.
- **Blocking outside engineering:**
  - X-01: attach the AWS deployer policy to `MedCodex-Local`, the profile in use. Needed by W4 and W5.
  - X-04 and X-06: clinical and legal approvers.
  - X-05: on-call clinicians.
  - The full list is in [OPEN-ITEMS.md](OPEN-ITEMS.md).

## Progress

<!-- mvp:summary:start -->

- Tasks done: **46 / 330** (active 2, blocked 1, deferred 0).
- Register entries fixed: **43 / 345**.
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
  4. Copy the barrier zip off the machine. There are no commits, so the zip is the durable copy.
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
