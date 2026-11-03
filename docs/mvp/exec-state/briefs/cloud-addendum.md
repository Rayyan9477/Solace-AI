# Cloud-session addendum (read after your brief and phaseB-common.md; this file wins on paths and tools)

The program now runs in a Claude Code cloud session (Ubuntu, bash). Translate every Windows path and tool:

| Old (local Windows) | Cloud |
|---|---|
| `D:\Repo\Contextual-Chatbot` | `/home/user/repo` |
| `D:\tmp\mvp-exec\briefs\...` | `docs/mvp/exec-state/briefs/...` |
| `D:\tmp\mvp-exec\reports\...` (incl. `WS-04.F1-shape.md`) | `docs/mvp/exec-state/reports/...` |
| `D:\tmp\mvp-exec\venv\Scripts\python.exe` | `.venv/bin/python` |
| PowerShell tool | Bash tool |
| `checkpoint.ps1 -Label X` | `python3 docs/mvp/exec-state/tools/mvp_exec.py checkpoint X` (prints the tree id) |

Run tests (one pytest process at a time; other implementers share this VM, so run only the suites your task needs):

```bash
cd /home/user/repo
export PYTHONPATH="$PWD:$PWD/src" ENVIRONMENT=test TELEMETRY_ENABLED=false
.venv/bin/python -m pytest <path> -o "addopts=" -p no:logging -q
```

- Docker is available (daemon started by the controller, pulls go through mirror.gcr.io). Real-infra tests run with
  `SOLACE_REQUIRE_REALINFRA=1`. Use a compose project name unique to your task (e.g. `-p ws035`) so lanes do not
  collide, remove images by tag, and leave no containers, networks or volumes behind.
- Other implementers edit other files in this same working tree at the same time. Touch only your declared files.
  Never run `git checkout`, `git stash`, `git reset`, `git add` or `git commit` (the controller commits).
- Synthetic data only. Never open `.env` files or print secret values.
- Write the report to `docs/mvp/exec-state/reports/<task>-report.md` (fix rounds: `<task>-fixN-report.md`).
- Return only: status (DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT), tree id, a one-line test summary, concerns.
