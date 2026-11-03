## 4. Execution protocol

Execution starts only on the owner's "go". It follows superpowers:subagent-driven-development, adapted to the owner's rules as listed below. Where this section and that skill differ, this section wins: no commits, parallel dependency-ready dispatch, and AGY as the third model.

### 4.0 W0 entry (once, after "go")

1. Checkpoint the untouched tree (`checkpoint.ps1 -Label "W0:entry"`) and export it as the first barrier zip (section 4.1).
2. Create the program venv (D-20): `& "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m venv --system-site-packages D:\tmp\mvp-exec\venv`. The venv reads anaconda's packages and never changes them.
3. Record `D:\tmp\mvp-exec\venv\Scripts\python.exe -m pip freeze` to `D:\tmp\mvp-exec\venv-freeze-before.txt`, install into the venv only the D-16 pins the suites lack (aiokafka, langgraph-checkpoint-postgres with psycopg, PyJWT >= 2.13, and any other lock-pinned test extra WS-01 names), then record `venv-freeze-after.txt`.
4. Run every suite once with the venv interpreter and record the per-suite counts as the W0 baseline in the ledger and in `docs/mvp/TRACKER.md`. A drop against the 5,377 anaconda baseline needs a ledger Ruling before any task starts.
5. Mark the plan "in execution" in `docs/mvp/README.md` and add a `LOG.md` entry.

### 4.1 Workspace, ledger and tools

Everything the controller produces lives outside the repo in `D:\tmp\mvp-exec\`. This keeps it out of the working tree, so `git add -A` snapshots stay clean.

| Path | Purpose |
|---|---|
| `D:\tmp\mvp-exec\ledger\progress.md` | The ledger and recovery map. Its first line is `# SDD ledger — plan: docs/superpowers/plans/2026-10-18-mvp-remediation.md`. |
| `D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label <text>` | Commit-free snapshot of the whole working tree. It stages into a temporary index and writes a git tree object, then prints the tree id. It creates no commit, no ref and no branch, and never touches the real index. `.env` and other ignored files are excluded. Tested on this repo. |
| `D:\tmp\mvp-exec\tools\review-package.ps1 -Base <tree> -Head <tree> -Name <task> -Paths <declared files>` | Review package for one task: changed paths across the whole tree, plus the stat and diff for the task's declared paths. |
| `D:\tmp\mvp-exec\tools\rollback.ps1 -Tree <tree> -Paths <files>` | Restores the declared files to a checkpoint and deletes files the task created. The real index is untouched. |
| `D:\tmp\mvp-exec\tools\extract-brief.ps1 -Task WS-NN.K` | Copies one task's text from its workstream plan to `D:\tmp\mvp-exec\briefs\WS-NN.K-brief.md`. This replaces the skill's bash `task-brief`. |
| `D:\tmp\mvp-exec\reports\<task>-report.md` | The implementer's full report. Fix rounds append to it. |
| `D:\tmp\mvp-exec\venv\Scripts\python.exe` | The program interpreter (D-20), created at W0 entry (section 4.0). Every test and script command uses it. Nothing is installed into anaconda. |
| `D:\tmp\mvp-exec\barriers\W<n>-<tree7>.zip` | `git archive` export of each barrier tree. With no commits (D-17), checkpoint trees are unreferenced git objects that `git gc` may prune after about two weeks, so each barrier zip is also copied to the owner's off-machine location. |
| `docs/mvp/` (in the repo) | The persistent tracking set: `README.md` (hub and status snapshot), `DECISIONS.md`, `TRACKER.md` (per-task status, barrier log), `OPEN-ITEMS.md` (owner actions, sign-offs, risks) and `LOG.md`. Only the controller edits it; it is excluded from scope-breach checks. `docs/mvp/tools/mvp_docs.py` re-renders the tracker tables and keeps the status cells. |

These ledger lines are the only completion record, because there are no commits:

- `Task WS-NN.K: start base=<tree>`
- `Task WS-NN.K: complete (trees <base7>..<head7>, review clean)`
- `Task WS-NN.K: fix round R/5 (...)`
- `Task WS-NN.K: minor (deferred): ...`
- `Ruling: <decision> — <why> — <cost if wrong>`

After any interruption, trust the ledger and the snapshot log (`D:\tmp\mvp-exec\snapshots\snapshots.log`) over memory. Never re-dispatch a task that has a `complete` line.

### 4.2 Per-task loop (TDD plus subagent-driven development)

1. **Brief.** Run `extract-brief.ps1 -Task <id>`, then run `checkpoint.ps1 -Label "<id>:base"` and record `start base=<tree>` in the ledger.
2. **Dispatch the implementer.** Use the task's `Implementer model` (Sonnet 5.5 or Opus 5.5), always set explicitly. Use the skill's `implementer-prompt.md` with these changes:
   - "commit" becomes "run checkpoint.ps1 and report the tree id";
   - the implementer must edit only the task's declared files;
   - it must follow the six TDD steps in order, and the report must show the RED output from Step 2;
   - implementers never spawn subagents.
3. **Check scope.** Run `checkpoint.ps1 -Label "<id>:head"`, then `review-package.ps1 -Paths <declared files>`. Any changed path that no running task declared is a scope breach: roll it back with `rollback.ps1`, or rule on it in the ledger.
4. **Review** according to the task's `Review` class (see 4.4). The reviewer receives the brief, the report and the package. It returns spec compliance and quality verdicts.
5. **Fix loop**, as in the skill: at most 5 rounds. Rounds 1-3 resume the same implementer; rounds 4-5 use a fresh Opus 5.5 implementer. Each round ends with a scoped re-review of the fix diff, using `review-package.ps1` from the previous head tree. At the cap, adjudicate with ledger rulings.
6. **Complete.** Write the completion ledger line and set the task's Status cell in `docs/mvp/TRACKER.md` to `done` (with the head tree in Notes). Register statuses are derived from the tracker: an entry is `fixed (<task ids>)` when every task covering it is `done`, and `in progress (<task ids>)` while only some are. At each wave barrier, re-render `docs/MVP-ISSUE-REGISTER.md` with `compile_register.py`, which reads the tracker.

The RED step is not optional. A defect-fix test that passes before the fix is a finding against the task, and the task is not done until the test proves the defect.

### 4.3 Parallel execution in one working tree

The owner's rule is no commits and no worktrees, so every implementer works in the same tree. Parallelism comes from running only tasks that cannot disturb each other:

- **Ready tasks.** Inside a wave and phase, a task is ready when every task it depends on in that wave and phase is complete. Dependencies in earlier waves, and phase-A dependencies of phase-B tasks, are complete by construction.
- **Mutual exclusion.** Each task has mutex keys: every file it creates or modifies, plus the suite of every file it creates, modifies or tests (`services/<svc>`, `services/shared`, `src/<pkg>`, `tests/<suite>`, `tests/realinfra/<slug>`, `tests/e2e/<x>`, `infrastructure/<x>`, and `migrations` as one key). Two tasks that share a key never run at the same time, because each runs its owning suite as its regression step. `D:\tmp\mvp-plan\schedule_dag.json` lists each task's keys, same-phase dependencies and chain height. `check_plans2.py` regenerates it from the plan files.
- **Dispatch.** The controller keeps up to 6 implementers busy, with at most 2 on real-infra tasks (Docker resources). Among the ready tasks whose keys are free, it picks the longest remaining dependency chain first, then blockers before highs, then safety-critical before security-critical before standard.
- **Phases.** Platform tasks run first in each wave (phase A); service tasks start when phase A is done (phase B). A task is phase A if it touches a platform path as the Global Constraints define it: `src/**`, `services/shared/**`, `migrations/**` (service-owned slot files included), `infrastructure/**`, `deploy/**`, `docker/**`, the seven `event_bridge.py` files, compose, CI, build or requirements files. A task is phase B if it touches only service paths. A task that touches only tests, scripts or docs follows its same-wave dependencies (phase B if any of them is phase B). Every suite imports the shared packages, so a half-edited platform file must never be under a service task's tests.
- **Phase-A cross-talk.** Phase-A tasks can still see each other's half-edited shared packages. If a phase-A implementer's suite fails inside a file that another running task owns, it waits for that task's next checkpoint and re-runs before treating the failure as its own. If this happens twice for the same pair of workstreams, the controller adds a mutex key between them for the rest of the phase.
- **Test scope.** During a task, an implementer runs only its own tests and the owning suite. The full regression across all suites runs only at the wave barrier.
- **Pre-dispatch check.** Before dispatching a task, the controller confirms that its mutex keys are disjoint from every running task's keys. After each task it checks the package's "All changed paths" section for edits outside the task's declared files.

### 4.4 Model and review matrix

| Role | Standard task | Security-critical task | Safety-critical task |
|---|---|---|---|
| Implementer | Sonnet 5.5 (Opus 5.5 if the task says so) | Opus 5.5 | Opus 5.5 |
| Task reviewer (spec + quality) | Sonnet 5.5 | Opus 5.5 | Opus 5.5 |
| Extra independent review | none | AGY adversarial review of the task package | AGY adversarial review of the task package |
| Scoped re-review of fixes | Sonnet 5.5 | Sonnet 5.5 | Opus 5.5 |
| Fix rounds 4-5 | fresh Opus 5.5 | fresh Opus 5.5 | fresh Opus 5.5 |

Other roles:

- **Wave bug-finding round:** Sonnet 5.5 hunters (one per changed area, at most 4) plus one AGY adversarial pass. Findings are verified by Sonnet 5.5 and Opus 5.5 voters, with an Opus tie-break; a majority refute kills a finding.
- **Final whole-branch review:** Opus 5.5 (the skill's `code-reviewer.md`) plus AGY adversarial review.
- **Easy help** (summaries, doc cross-checks, ledger digests): AGY.

AGY runs only through the gemini plugin's MCP tools, with `engine: agy` (the Gemini CLI login is expired; AGY is verified). AGY has no read-only mode, so it never gets the real repo as its workspace. Each AGY review gets an exported snapshot:

```
git -C D:\Repo\Contextual-Chatbot archive --format=zip -o D:\tmp\mvp-exec\agy\<name>.zip <tree>
```

Expand the zip to `D:\tmp\mvp-exec\agy\<name>\`, copy the review package into it, and point AGY at that directory. Then compare `checkpoint.ps1` trees taken before and after the AGY call; the real repo tree must be unchanged.

### 4.5 Wave barrier

A wave is done only when all of these hold, in this order:

1. Every task in the wave has a `complete` ledger line.
2. **Full regression is green.** Every suite runs in its own pytest process (section "Global Constraints" has the command). Test counts are compared with the previous barrier, and any drop in count is investigated.
3. **The wave's real-infra acceptance tests are green** (listed in the wave's exit criteria in section 3).
4. **Wave package.** Run `review-package.ps1 -Base <wave-start tree> -Head <wave-end tree> -Name wave-N`.
5. **Bug-finding rounds** (section 4.6) until a round is dry.
6. The register is re-rendered with the new statuses, and the wave summary is appended to the ledger.
7. **Barrier note.** The barrier tree id, the per-suite test counts and the register delta go into the ledger, the tracker's barrier log, a `docs/mvp/LOG.md` entry and a short note to the owner. Nobody commits (D-17). Export the tree with `git -C D:\Repo\Contextual-Chatbot archive --format=zip -o D:\tmp\mvp-exec\barriers\W<n>-<tree7>.zip <tree>` and ask the owner to copy the zip off the machine. Execution continues without waiting for the copy.

### 4.6 Bug-finding rounds

The input to round 1 is the wave package. The input to each later round is the fix diff from the previous round.

1. **Hunt** in parallel:
   - Sonnet 5.5 hunters, one per changed area (at most 4). They are told what this wave fixed, so they look for regressions, missed paths, and new defects introduced by the fixes.
   - One AGY adversarial review on the exported snapshot.
2. **Verify.** Each finding goes to a Sonnet 5.5 voter and an Opus 5.5 voter. On a split, an Opus 5.5 tie-breaker casts a third vote. A majority refute removes the finding.
3. **Fix.** Each confirmed finding becomes a register entry (the next MVP-xxx id) and a fix task `WS-NN.F<k>` in the owning workstream, dispatched like any other task. It goes through the full per-task loop (TDD, review).
4. **Repeat** until a round confirms nothing new, with a cap of 3 rounds per wave. Findings still open at the cap are ruled on in the ledger and scheduled into the next wave.

### 4.7 Stop conditions

The controller does not pause between tasks or waves. It stops and asks the owner only for:

- an irreversible or destructive operation;
- a security-sensitive action;
- a side effect outside the working tree that norms say to ask about first (installing packages system-wide, pushing, publishing, provisioning cloud resources);
- a plan defect so deep that every path forward is a guess.

The venv bootstrap at W0 entry (section 4.0) is approved by D-20. Installing anything into anaconda or system-wide is a stop. AWS provisioning needs the deployer policy attached to the existing user MedCodex-Local (D-25); if it is still unattached at W5 entry, the release gate is a stop. No AWS call ever reads a secret value (D-07 uses SOPS).

Every other ambiguity becomes a ledger `Ruling:`. At the end, all rulings are reported to the owner in order.

### 4.8 Finish

After the last wave:

1. The final whole-branch review: Opus 5.5 plus AGY on the package from the pre-execution baseline tree to the final tree. Then one fix dispatch, one scoped re-review, and adjudication of the residuals.
2. The release gate in section 3 (the last wave's exit criteria).
3. A report to the owner: the "Rulings I made" list, the register status, the test counts, and the open owner decisions.

There are no commits (D-17). The final report names the final tree id and the barrier zips, and the owner decides what to commit afterwards.
