# Solace-AI: instructions for Claude Code (local and cloud sessions)

Solace-AI is a mental-health AI companion backend. It has 10 FastAPI microservices: orchestrator, user, safety, notification, diagnosis, memory, therapy, personality, config (frozen by D-19) and analytics. The stack is Python 3.12, LangGraph, Kafka with a transactional outbox, PostgreSQL with FORCE row level security, Redis, optional Weaviate and ClickHouse. It uses structlog, Pydantic and async throughout.

Shared packages live in `src/`: `solace_events`, `solace_security`, `solace_infrastructure` (including the IF-01 runtime) and `solace_testing`. The services live in `services/`.

## Standing rules (owner-set, binding)

- **TDD:** write a real failing test (RED) for every change, then GREEN, then the regression. Verify before you claim done, and paste the evidence.
- **PHI and secrets:**
  - Cloud sessions are not HIPAA-eligible. Never put real patient data, production data or production credentials in a cloud session. Synthetic and test data only.
  - Never open, print or commit `.env` files or secret values. Log names only.
  - Use the AWS Secrets Manager pattern (`{{resolve:secretsmanager:...}}` with `asm-exec`); never call `get-secret-value`.
- **AWS:**
  - Prefer IaC (CDK or CloudFormation).
  - No em dashes in AWS resource names or descriptions.
  - AWS work (staging, HA/DR) runs from the owner's machine, not a cloud session, because cloud sessions have no AWS credentials.
- **Security fails closed:** strict environments (staging, production) refuse to start on unsafe settings. Dev and test must boot with the compose defaults.
- **Edits:** keep to a task's declared files, and match the surrounding code's style.
- **Git:**
  - The program's work lives on `main` up to `eb5eb26` (the owner's commits of 2026-11-03), plus this kit on branch `mvp-remediation`. Cloud sessions work from `mvp-remediation`, or the branch the owner names.
  - Commit at every task checkpoint, with messages such as `WS-12.1: fix round 2 (...)`, and push so nothing is lost if the cloud VM is reclaimed.
  - Never force-push, never delete branches, and never push to `main` without the owner.
- **First step in a cloud session:** if `.venv/bin/python` is missing, run `bash scripts/cloud/setup.sh` (about 3-5 minutes). It builds the Python 3.12 venv from the hash-pinned locks.

## How to run tests

Run one pytest process at a time; parallel large runs stall.

```bash
export PYTHONPATH="$PWD:$PWD/src" ENVIRONMENT=test TELEMETRY_ENABLED=false
.venv/bin/python -m pytest <path> -o "addopts=" -p no:logging -q
```

- Real-infrastructure tests (Postgres, Kafka or Redis in Docker) run with `SOLACE_REQUIRE_REALINFRA=1`.
- Remove images by tag (`docker rmi <tag>`, never `-f <id>`), and leave no containers or volumes behind.
- Known baseline red:
  - `tests/solace_infrastructure/kafka/test_topics.py` (12 cases);
  - the per-service conformance cases of services not yet adopted (analytics, notification, personality, user);
  - the import-root and lint owner cases of WS-12/14/16/18;
  - the `/health` cases in `tests/system/test_smoke.py`, until WS-05.5.

## The MVP remediation program (current work)

- **Plan:** `docs/superpowers/plans/2026-10-18-mvp-remediation.md`, with 18 workstream plans in `docs/superpowers/plans/2026-10-18-mvp-remediation/`. There are 330 tasks in waves W0 to W5.
- **Tracking:**
  - `docs/mvp/TRACKER.md` holds task statuses. Update it with `python docs/mvp/tools/mvp_docs.py set WS-NN.K <status> --note "..."`.
  - Also see `docs/mvp/README.md`, `DECISIONS.md`, `OPEN-ITEMS.md` and `LOG.md`.
- **Execution state** (`docs/mvp/exec-state/`):
  - `ledger/progress.md`: every task start and completion, every ruling and every checkpoint.
  - `ledger/handoffs.md`: pending hand-offs; tick each one when you dispatch it.
  - `ledger/agents.md`: historical agent ids.
  - `briefs/`, `reports/` and `reviews/` (`batch*.json` hold review results).
  - `snapshots/snapshots.log`: checkpoint tree ids.
  - `plan/` holds the schedule DAG.
- **State at pause (2026-11-03):** W0 checkpoint 1, tree `edba7d7`. The final docs tree is `b860a79`, and its content equals commit `eb5eb26` on `main`.
  - W0 is 48/61 done, 1 blocked (WS-05.4 part 2) and 12 not started. The program is 48/330 done.
  - Resume only when the owner says so.
- **Remaining W0, in dependency order:**
  - WS-03.5, then WS-05.4 part 2, WS-05.5, WS-05.6;
  - WS-12.17, WS-13.2, WS-14.2, WS-14.3, WS-15.2;
  - WS-17.2, which is a barrier gate because memory still has a private Postgres engine;
  - WS-17.19, WS-18.1, WS-18.2;
  - then the W0 bug-finding round and the barrier: full regression, a strict boot smoke, and a barrier snapshot.
- **Tools** (cross-platform, in `docs/mvp/exec-state/tools/`):
  - `mvp_exec.py` with the subcommands `checkpoint`, `review-package`, `extract-brief` and `ledger`;
  - `pick_ready.py`: the dependency-ready tasks and their mutex conflicts;
  - `review-wf.js` and `review-batch-wf.js`: review workflows.

### Execution protocol (how the controller works)

1. **Picking tasks:** pick dependency-ready tasks with `pick_ready.py`, and run at most 6 implementers at once. Tasks that share mutex keys (files or suites) never run together. A lane in review may overlap only on a suite key, never on files.
2. **Dispatching:**
   - Each implementer is a subagent. It receives the brief (`extract-brief`), plus the common phase-B rules in `docs/mvp/exec-state/briefs/phaseB-common.md` for service tasks.
   - Implementers edit declared files only, make no git writes, spawn no subagents, show a real RED, checkpoint, and write `reports/<task>-report.md`.
   - Models: Opus for safety-critical and security-critical design, Sonnet for standard work, Haiku for docs and easy passes. AGY/Gemini is not available in cloud sessions.
3. **Reviewing:**
   - Every task gets a review: lens-diverse reviewers, and each finding goes to a skeptic panel (3 skeptics for important findings, 1 for minor; a majority refute kills the finding).
   - Use the Workflow tool with `review-wf.js` or `review-batch-wf.js` when it is available. Otherwise run the same panel as Agent subagents.
   - Fix rounds go back to the same implementer. Scoped re-reviews follow.
   - Park minors with a ruling only when they do not touch safety, PHI or security.
4. **Rulings:** record every decision in the ledger as `Ruling: <what> - <why> - <cost if wrong>`.
5. **Lesson:** never defer the only end-to-end run of a tool whose fix touched its main loop.

## Platform facts (binding contracts, decided in W0)

- **Lifespan (IF-01):** `async with service_runtime(app, SPEC) as runtime:` is the only statement of the lifespan. It ends with a bare `yield`. Every stop goes through `runtime.register_shutdown`. `main.py` never calls `structlog.configure`, `ProductionGuards.assert_startup` or `exception_handler`.
- **Postgres:**
  - The default pool connects as `solace_app` (NOBYPASSRLS, DML only). The migrator is `solace_migrator` (BYPASSRLS).
  - `DATABASE_URL` has no query string. TLS comes only from `POSTGRES_SSL_MODE` / `POSTGRES_SSL_ROOT_CERT`, and strict uses verify-full.
  - RLS uses `NULLIF(current_setting('app.current_user_id', true), '')::uuid`.
- **Imports and events:** imports are bare (`solace_events`, never `src.solace_events`). Event bridges pass `source_service`, and the outbox refuses a foreign or blank owner.
- **Redis:** `REDIS_URL=redis://:<urlsafe password>@redis:6379/0`. Strict refuses the dev default and passwords that are not URL-safe.
- **Strictness:** a service is strict if its own environment OR the bare `ENVIRONMENT` is staging or production. A blank service environment under a strict `ENVIRONMENT` is refused.
- **Decisions:** D-04 Bedrock for the LLM; D-07 SOPS + age for secrets; D-15 verify-full; D-16 security bumps; D-29 multi-locale with the US first. See `docs/mvp/DECISIONS.md`.

## Owner preferences

- Brief answers. Do not spend tokens on things that do not need them, and keep a good pace.
- Use they/them for the owner.
- Explain blockers plainly, and ask before anything destructive or outward-facing.
