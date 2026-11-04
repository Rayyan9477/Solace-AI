# Moving the MVP program to Claude Code cloud sessions

Status: **applied on 2026-11-03 after the owner's go-ahead.**

- The owner committed and pushed all program work to `main` (`eb5eb26`).
- This kit was committed on branch `mvp-remediation` and pushed.
- The first cloud session was launched from that branch with `claude --cloud`. It runs `scripts/cloud/setup.sh` itself as its first step, so no environment setup script is required.

## 1. What a cloud session can and cannot see

This summary comes from the Claude Code docs on cloud sessions and cloud environments, checked on 2026-11-03.

| Item | In a cloud session? | What this kit does |
|---|---|---|
| Repo files pushed to GitHub (`Rayyan9477/Solace-AI`) | Yes, as a fresh clone of the pushed branch | The work must be committed and pushed. See section 3. |
| Uncommitted or untracked local files | No (a bundle upload carries tracked edits only) | They are committed to a branch. |
| `D:\tmp\mvp-exec`, `D:\tmp\mvp-plan` (ledger, briefs, reviews, DAG) | No | Copied into `docs/mvp/exec-state/`. |
| User `~/.claude` (CLAUDE.md, skills, agents, plugins, hooks) | No | Rules go in a repo `CLAUDE.md`. Skills are vendored into `.claude/skills/` and agents into `.claude/agents/`. |
| Plugins enabled in settings | No (repo `enabledPlugins` is not installed in the cloud) | Vendored as skills. Skills enabled on claude.ai also load. |
| Auto memory (`~/.claude/projects/.../memory`) | No | Its content is in `CLAUDE.md` and the ledger. |
| MCP: claude.ai connectors | Yes | None needed. |
| MCP: local stdio (aws-mcp via uvx, rea) | Not documented, and there are no AWS credentials there anyway | AWS work stays local. |
| AGY / Gemini CLI | No | Use the Opus, Sonnet and Haiku mix only. |
| Docker / docker compose | Yes (Ubuntu 24.04, 4 vCPU, 16 GB RAM, 30 GB disk) | Real-infra tests can run, but the full 10-service stack may be slow. |
| Workflow tool (multi-agent scripts) | Not documented | Fall back to Agent subagents for review panels. |
| Subagents | Yes | Same protocol. |
| Agent teams | Off by default | Set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in the environment variables if wanted. |
| PHI / HIPAA | A HIPAA configuration blocks cloud sessions | Code and synthetic data only. No real PHI or production credentials, ever. |
| Sessions | Idle sessions pause and can be reclaimed; background tasks do not survive | Checkpoint (commit) after every task. |

## 2. What was backed up (local only, `D:\Backups\solace-cloud-migration-2026-11-03`)

| Folder | Contents |
|---|---|
| `claude-user/` | `~/.claude` CLAUDE.md, KNOWLEDGE_GRAPH_TOOLS.md, settings.json, settings.local.json, skills/, agents/, commands/, hooks/, plugins/ (with caches and marketplaces). Excluded: `.credentials.json`, history, shell and env snapshots, the paste and file-history caches, telemetry and debug. |
| `project-context/` | This project's auto memory and every session transcript (`*.jsonl`, plus subagent and workflow transcripts): the full conversation context. **Local only. Do not commit or upload.** |
| `program-state/` | `D:\tmp\mvp-exec`: ledger, tools, briefs, reports, reviews, snapshots and barrier zips; the venvs, scratch and clean-checkout exports are left out because they can be rebuilt. Also `D:\tmp\mvp-plan`: plan state and scripts. |
| `repo/` | The W0 checkpoint-1 tree zip (`edba7d7`), a git bundle of all history, a patch of uncommitted tracked changes, and lists of untracked files, status, HEAD, remotes and branches. |
| `cloud-kit/` | The files to add to the repo for cloud sessions (this document is one of them). |
| `_tools/` | `secret_scan.py` and its report (file paths and pattern names only). |

## 3. Decisions for the owner (needed before applying)

1. **Commit and push.** Cloud sessions need the work on GitHub. That ends the local "no commits" rule (D-17) for this program.
   - Proposed: a new branch `mvp-remediation` from `main`, with one commit for the checkpoint-1 working tree and one for this kit. After that, one commit per task checkpoint.
   - Your three commits from 2026-11-03 16:02-16:05 are already on `origin/main`.
2. **Where AWS work runs.** Proposed: locally only (W4/W5 staging and HA/DR, X-01). The alternative, AWS keys as cloud environment variables, is readable by anyone with the environment, so it is not recommended.
3. **Network access.** "Trusted" (the default) allows PyPI, apt and Docker registries. Raise it only if a task needs another host.
4. **Credit window.** The included $250 cloud credit expires on 2026-11-05 at 12:59 PM GMT+5, after which normal plan usage applies. At the measured pace, the rest of W0 (3-4 h) fits comfortably; the whole program (about 45-55 h) does not.

## 4. Steps to apply (after the go-ahead)

1. Run a secret scan of the working tree and stop on any hit: `python D:\Backups\...\_tools\secret_scan.py D:\Repo\Contextual-Chatbot`. Confirm that `.env` is ignored.
2. Copy `cloud-kit/` into the repo root: `CLAUDE.md`, `.claude/skills`, `.claude/agents`, `docs/mvp/exec-state`, `docs/mvp/CLOUD-MIGRATION.md` and `scripts/cloud/setup.sh`.
3. Create the branch, commit and push: `git switch -c mvp-remediation`, `git add -A`, `git commit`, `git push -u origin mvp-remediation`.
4. Create the cloud environment on claude.ai/code:
   - repository `Rayyan9477/Solace-AI`, branch `mvp-remediation`;
   - setup script: the contents of `scripts/cloud/setup.sh`;
   - environment variables: `PYTHONPATH=/home/user/Solace-AI:/home/user/Solace-AI/src` (adjust to the clone path that `/context` shows), `TELEMETRY_ENABLED=false`, and optionally `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`;
   - network: Trusted.
5. Smoke-check the first session:
   - `/context` lists CLAUDE.md and the skills;
   - the venv imports;
   - `pytest tests/bootstrap -q` gives the expected 8 pre-adoption failures only;
   - `python docs/mvp/exec-state/tools/pick_ready.py` lists the W0 tasks.
6. Resume prompt: "Resume the MVP program from W0 checkpoint 1 per CLAUDE.md and docs/mvp/exec-state/ledger/progress.md; finish the remaining 12 W0 tasks, the bug-finding round and the W0 barrier."

## 5. Roll back

Nothing local is changed by the cloud move. The local working tree, `D:\tmp` and `~/.claude` stay as they are, and the backup holds copies. To go back, keep working locally. The pushed branch can stay on GitHub or be deleted by the owner (cloud sessions cannot delete branches).
