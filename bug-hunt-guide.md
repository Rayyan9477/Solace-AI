# Multi-Agent Bug Hunt & Fix Guide

A repeatable pipeline that uses a **knowledge-graph backbone**, **parallel agent swarms**, **live documentation**, and **superpowers discipline** to find and fix bugs in an existing codebase.

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────┐
│  GitNexus Knowledge Graph  +  Context7 docs  +  Exa web  │
│  (the brain: code structure, live API docs, CVE feeds)   │
└──────────────────────────────────────────────────────────┘
            ▲                        │
            │ feeds                  │ powers
            │                        ▼
┌──────────────────────┐      ┌─────────────────────────┐
│  Triage Swarm        │─────▶│  Fix Swarm              │
│  (parallel scanners) │      │  (one workspace per bug)│
└──────────────────────┘      └─────────────────────────┘
            │                          │
            ▼                          ▼
     Bug Ledger (ranked)        Verified, tested fixes
```

---

## Toolbelt (what each piece is for)

### Knowledge & context (always-on)
| Tool | Role |
|---|---|
| **GitNexus** (`mcp__gitnexus__*`) | Code graph: impact, call paths, contract drift |
| **Context7** (`mcp__claude_ai_Context7__*`) | Live library docs — verify API usage, catch deprecations |
| **Exa** (`mcp__plugin_exa-mcp-server_exa__*`) | Web search for CVEs, bug reports, advisories |
| **rag-memory-manager** agent | Persists findings across sessions (multi-week hunts) |

### Orchestration
| Tool | Role |
|---|---|
| **ruflo-swarm** (via `/superpowers:dispatching-parallel-agents`) | Parallel agent fan-out |
| **ruflo-autopilot** | Long autonomous sweeps with stop conditions |
| **ruflo-federation** | Cross-repo coordination |
| **`/loop`** | Recurring scans on a cadence |
| **`/schedule`** | One-shot or cron remote agents |

### Specialized scanners (Triage Swarm)
| Agent | Bug class |
|---|---|
| `bug-analyzer-reproducer` | Logic, edge cases, off-by-ones, repros |
| `code-logic-analyzer` | Variable flow, cross-module consistency, races |
| `security-vulnerability-scanner` | OWASP, secrets, injection, authz |
| `error-detective` | Stack traces & log patterns → root cause |
| `dependency-compatibility-manager` | Version conflicts, deprecated APIs |
| `performance-engineer` | Hot paths, N+1, memory, latency |
| `devops-troubleshooter` | Infra, CI, runtime config bugs |
| `test-runner` | Existing test failures |

### Fixers & reviewers
| Agent / Skill | Role |
|---|---|
| `fix-planner-patch` | Generates patch + regression test |
| `debugger` | Deep-dive on stubborn failures |
| `code-refactoring-expert` | Restructure structurally rotten files |
| `architecture-advisor` | Cross-cutting design bugs |
| `code-reviewer` agent | Independent review of fixes |
| `frontend-design` skill | UI bug fixes with quality |
| `Playwright` MCP | UI repro & E2E verification |
| `simplify` skill | Collapse duplication in fix wave |

---

## Phase 0 — Prime the brain

```
/gitnexus-cli   → "index this repo"
/gitnexus-cli   → "generate wiki"   (optional)
```

Unlocks `impact`, `query`, `cypher`, `route_map`, `tool_map`, `shape_check`, `detect_changes`.

Re-index after big merges; schedule weekly auto-reindex.

---

## Phase 1 — Baseline map

```
Agent: codebase-mapper
Input:  graph + README + manifests
Output: docs/_map.md  (entry points, layers, hot paths)
```

Every downstream agent reads this map first — shared briefing prevents drift.

---

## Phase 2 — Triage Swarm (parallel, single message)

Dispatch via `/superpowers:dispatching-parallel-agents`. Each writes to `reports/<agent>.md`.

Brief template for each agent:

> Read `docs/_map.md`. Use `mcp__gitnexus__query` to scope to <area>. For any external library use, verify current API via `mcp__claude_ai_Context7__query-docs`. For suspected CVEs, check `mcp__plugin_exa-mcp-server_exa__web_search_exa`. Report to `reports/<your-name>.md`: file:line, severity (P0/P1/P2), evidence, repro, suggested-fix-class. Under 500 words.

Run alongside (single-pass on diff, not the swarm):
- `/security-review`
- `/simplify`

---

## Phase 3 — Bug Ledger

```
Agent: sprint-task-decomposer
Input:  reports/*.md
Output: bugs/LEDGER.md  (deduped, ranked, acceptance criteria)
```

Each entry:
- ID, severity, title
- file:line, evidence
- **Impact set** (`mcp__gitnexus__impact`) — dependents
- **Library context** (Context7) — if API misuse, link the canonical doc
- **External advisories** (Exa) — if matches a known CVE
- Reproduction steps
- Acceptance criteria (the failing test that must pass)

Sort: P0 (security/crash/data-loss) → P1 (broken feature) → P2 (edge/UX).

Persist the ledger via `rag-memory-manager` so it survives sessions.

---

## Phase 4 — Pre-fix safety check

For each bug:

```
1. mcp__gitnexus__impact <file:symbol>     # blast radius
2. mcp__gitnexus__shape_check              # contract drift
3. /gitnexus-debugging                     # trace fault origin
4. mcp__claude_ai_Context7__query-docs     # confirm library expectations
```

If impact set > 10 callers → route to `code-refactoring-expert`, not a point fix.

---

## Phase 5 — Fix Swarm (parallel workspaces)

For each independent bug:

```
/superpowers:using-git-worktrees      # isolated workspace
/superpowers:systematic-debugging      # hypothesis → repro → minimal fix
/superpowers:test-driven-development   # FAILING test first

Agent: fix-planner-patch               # patch + regression test
   ↳ for UI bugs:  Playwright MCP to repro & verify
   ↳ for perf:     performance-engineer in the loop
   ↳ for stubborn: debugger agent

/superpowers:verification-before-completion
```

Coordinate fan-out via `/superpowers:dispatching-parallel-agents`.

For mechanical sweeps (40+ deprecations, lint waves, dep bumps): **ruflo-autopilot** with explicit stop conditions:

> Fix all P2 bugs in `bugs/LEDGER.md` matching <pattern>. Stop after 10 fixes or any test failure. One workspace per fix.

---

## Phase 6 — Per-fix gate

Every fix must pass:

- [ ] Original repro from ledger no longer fails
- [ ] New regression test passes
- [ ] `mcp__gitnexus__impact` shows no unintended ripples
- [ ] `bug-analyzer-reproducer` rerun on changed files — clean
- [ ] `code-reviewer` agent + `/code-review:code-review` (independent reviewer, not the writer)
- [ ] `/security-review` if it touched auth, IO, or data
- [ ] Playwright run for UI changes
- [ ] Library use confirmed against Context7 docs

Use `/superpowers:requesting-code-review` to require independent review.

---

## Phase 7 — Post-wave hardening

1. **Re-index** the graph.
2. **Hot-file refactor**: files producing ≥3 bugs → `code-refactoring-expert`.
3. `simplify` across changed files — collapse duplicate patterns.
4. `architecture-advisor` if bugs cluster across layers — likely a structural issue.
5. **Recurring triage** via `/schedule`: weekly re-scan, post diff vs last ledger.
6. **rag-memory-manager**: store the wave's lessons (what classes of bugs we found, what fixed them) for next session.

---

## Knowledge-graph queries to keep handy

```cypher
// Untested functions
MATCH (f:Function) WHERE NOT (f)<-[:TESTS]-(:Test) RETURN f

// Public API surface
MATCH (f:Function {exported: true}) RETURN f.name, f.file

// Cross-module hotspots
MATCH (f:Function)<-[:CALLS]-(c)
WITH f, count(DISTINCT c.module) AS m
WHERE m > 5 RETURN f, m ORDER BY m DESC

// Recently changed + untested
MATCH (f:Function {changed_recent: true})
WHERE NOT (f)<-[:TESTS]-(:Test) RETURN f
```

Run via `mcp__gitnexus__cypher` to seed each triage round.

---

## Bug-class → toolchain matrix

| Bug class | Primary agent | Graph tool | Doc/Web tool | Verifier |
|---|---|---|---|---|
| Logic / edge case | bug-analyzer-reproducer | query, route_map | — | new unit test |
| Cross-module consistency | code-logic-analyzer | cypher, impact | — | integration test |
| Concurrency / race | code-logic-analyzer | cypher | Context7 (lib semantics) | stress test |
| Security | security-vulnerability-scanner | route_map | Exa (CVE feeds) | /security-review |
| Performance | performance-engineer | impact | Context7 (perf APIs) | bench harness |
| Dependency / deprecation | dependency-compatibility-manager | manifests | Context7 (current API) | type-check + tests |
| Runtime / infra | devops-troubleshooter | route_map | Exa (advisories) | smoke test |
| UI / UX | frontend-design + Playwright | — | Context7 (framework docs) | Playwright run |

---

## Mode selector

| Situation | Mode |
|---|---|
| < 10 bugs, mixed severities | Manual Fix Swarm |
| 40+ mechanical fixes | ruflo-autopilot + stop conditions |
| Multi-repo bugs | ruflo-federation |
| Needs design decisions | Manual + `/superpowers:brainstorming` |
| Overnight cleanup of known list | Autopilot + `/loop` status pings |
| Long-running multi-week hunt | rag-memory-manager + scheduled re-triage |

---

## Pitfalls

1. Skipping indexing — agents miss cross-file bugs.
2. One mega-agent — specialists outperform generalists.
3. Parallel fixes touching the same file — always isolate workspaces.
4. Trusting the ledger blindly — verify P0 repros yourself.
5. Fix-only mindset on hot files — refactor when ≥3 bugs cluster.
6. No regression test — guarantees the bug returns.
7. Re-running full triage every change — use `detect_changes` for diff-scoped scans.
8. Ignoring Context7 — many "bugs" are outdated API knowledge.
9. Relying on memory across sessions without rag-memory-manager — context evaporates.

---

## Minimum-viable command spine

```
# Phase 0 — Index
/gitnexus-cli  → index this repo

# Phase 1 — Baseline
Agent codebase-mapper → docs/_map.md

# Phase 2 — Triage (parallel, ONE message)
/superpowers:dispatching-parallel-agents
  ├─ bug-analyzer-reproducer
  ├─ code-logic-analyzer
  ├─ security-vulnerability-scanner
  ├─ error-detective
  ├─ dependency-compatibility-manager
  ├─ performance-engineer
  ├─ devops-troubleshooter
  └─ test-runner

# Phase 3 — Ledger
Agent sprint-task-decomposer → bugs/LEDGER.md
rag-memory-manager → persist ledger context

# Phase 4–6 — Per bug
for each bug in LEDGER:
  /gitnexus-impact-analysis
  Context7 doc check (if library involved)
  /superpowers:using-git-worktrees
  /superpowers:systematic-debugging
  /superpowers:test-driven-development
  Agent fix-planner-patch
  Playwright run (if UI)
  /superpowers:verification-before-completion
  /code-review:code-review
  /security-review (if relevant)

# Phase 7 — Harden
/gitnexus-cli  → reindex
Agent code-refactoring-expert (hot files)
simplify
/schedule weekly triage
rag-memory-manager → store wave lessons
```

---

Hand a repo path to start at Phase 0.
