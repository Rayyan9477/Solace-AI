# W0 phase B: common rules for every service-adoption task (read after your own brief)

Your own brief (`D:\tmp\mvp-exec\briefs\<task>-brief.md`) is the requirement. This file adds what earlier tasks
decided after your brief was written. Where they differ, this file wins.

## 1. The lifespan shape (WS-04.F1, binding)

Use the canonical form (a) exactly as in `D:\tmp\mvp-exec\reports\WS-04.F1-shape.md`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with service_runtime(app, SPEC) as runtime:
        ...  # compose from runtime.pg_pool / runtime.redis; register readiness checks
        ...  # start consumers/tasks; register EVERY service-owned stop with runtime.register_shutdown(name, closer)
        yield
```

- `service_runtime` is exported from `solace_infrastructure.runtime`. The `async with` is the only statement
  of the lifespan body, its target is named `runtime`, and nothing follows the `yield`.
- The `yield` must be a bare `yield` (no value, e.g. not `yield {...}`), it must be the last statement of the
  `async with` body, and there must be only one (WS-04.F1 fix round 1, enforced by `_bracket_form`).
- If your brief shows `runtime = await start_service_runtime(...)` with service stop code after the `yield`,
  translate it to form (a): move every stop into `runtime.register_shutdown(...)`.
- Background tasks: `runtime.register_background_task(name, task)` FIRST, then
  `runtime.register_shutdown(name, graceful_stop)`, so that the graceful stop runs before the cancel.
- `main.py` must not call `structlog.configure(`, `ProductionGuards.assert_startup(` or register an
  `exception_handler`; the runtime does all three.
- WS-04.F1 Step 3b is still landing inside `src/solace_infrastructure/runtime/*`. Do NOT edit those files.
  If you need a runtime change, stop and report it as NEEDS_CONTEXT.

## 2. What must turn green for your service

- `tests/bootstrap/test_service_conformance.py`: every case parametrized with your service
  (`test_service_conforms[<svc>]`, `test_lifespan_is_bracketed_by_the_runtime[<svc>]`, ...).
  Run it with `-k <svc>`. These cases are red today and are your acceptance.
- `tests/integration/test_phi_encryption_activation.py` stays green. A PHI service that drops
  `configure_phi_encryption(...)` must declare `handles_phi=True` as a keyword literal in its `SPEC`.
- `tests/build`: if your brief owns your workstream's case of `test_mvp_270_no_src_root_imports[WS-NN]` or of the
  lint gate, it must turn green. The other workstreams' cases stay red; do not touch them.
- `/live` comes from `create_service_app`. Compose healthchecks probe `/live` (WS-03.2).

## 3. Platform facts from W0 phase A

- Imports are bare: `solace_events`, `solace_infrastructure`, `solace_security`. Never `src.solace_events`.
- Postgres: the default pool connects as `solace_app` (NOBYPASSRLS, DML only). Do not open a private pool.
  `DATABASE_URL` carries no query string; TLS comes only from `POSTGRES_SSL_MODE` and `POSTGRES_SSL_ROOT_CERT`.
- Redis: `REDIS_URL=redis://:<password>@redis:6379/0` is set on every app container (WS-03.3).
- Event bridges already pass `source_service="<service>"` (WS-06.2). `PostgresOutboxStore.save` refuses a foreign
  or blank owner (WS-06.1).
- Strict environments (staging, production) fail closed. Dev and test must still boot with the compose defaults.

## 4. Rules

- Edit only your brief's declared files, plus any file this controller dispatch names. Use the Write/Edit tools
  only: PowerShell Set-Content adds a UTF-8 BOM.
- No git writes. Never open `.env` files or print secret values. Do not spawn subagents. Use the PowerShell tool.
- TDD: a real RED for each new or changed test, then GREEN.
- Run pytest one process at a time:
  `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`
- Real infra only if your brief says so. Remove images by tag (`docker rmi <tag>`, never `-f <id>`) and leave
  no containers or volumes behind.
- Checkpoint with `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "<task>:head"`.
- Write the full report to `D:\tmp\mvp-exec\reports\<task>-report.md`: RED/GREEN evidence, files, deviations and
  concerns. Return only status, tree id, a one-line test summary and concerns.
