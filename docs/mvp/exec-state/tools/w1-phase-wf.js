export const meta = {
  name: 'mvp-wave-phase',
  description: 'Run one phase of an MVP wave: dispatch ready tasks (max 6 lanes, 2 real-infra, mutex-disjoint), implement, review with skeptic panels, fix, commit, track',
  whenToUse: 'Local Windows controller run of W1 phase A or B. args: { phase: "A"|"B", branch, schedule: [W1-schedule.json rows] }',
  phases: [
    { title: 'Preflight', detail: 'branch, clean tree, Docker, done set from the tracker' },
    { title: 'Dispatch', detail: 'implement, review, skeptics, fix rounds, per task' },
    { title: 'Bookkeeping', detail: 'serialized commits, tracker and ledger, push' },
    { title: 'Phase close', detail: 'phase summary, draft PR' },
  ],
}

const REPO = 'D:/Repo/Contextual-Chatbot'
const PY = 'C:/Users/rayyan.a/AppData/Local/anaconda3/python.exe'
const PHASE = args.phase
const BRANCH = args.branch || 'w1-crisis-path'
const SCHEDULE = args.schedule || []
const MAX_LANES = 6
const MAX_RI = 2
const MAX_FIX_ROUNDS = 3

// ---------- schemas ----------
const STR = { type: 'string' }
const STRS = { type: 'array', items: STR }
const FINDING = {
  type: 'object',
  properties: {
    id: STR, severity: { type: 'string', enum: ['important', 'minor'] }, file: STR, line: { type: 'integer' },
    problem: STR, failure_scenario: STR, fix: STR, evidence: STR,
  },
  required: ['id', 'severity', 'file', 'line', 'problem', 'failure_scenario', 'fix', 'evidence'],
}
const PRE_SCHEMA = {
  type: 'object',
  properties: { ok: { type: 'boolean' }, abort_reason: STR, head: STR, docker_ok: { type: 'boolean' }, done_ids: STRS, notes: STR },
  required: ['ok', 'abort_reason', 'head', 'docker_ok', 'done_ids', 'notes'],
}
const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['DONE', 'DONE_WITH_CONCERNS', 'BLOCKED'] },
    files_changed: STRS, review_package: STR, report_path: STR, red_evidence: STR, green_evidence: STR,
    concerns: STR, rulings: STRS,
  },
  required: ['status', 'files_changed', 'review_package', 'report_path', 'red_evidence', 'green_evidence', 'concerns', 'rulings'],
}
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    spec_verdict: { type: 'string', enum: ['PASS', 'FAIL'] },
    quality_verdict: { type: 'string', enum: ['APPROVED', 'CHANGES_REQUESTED'] },
    findings: { type: 'array', items: FINDING },
  },
  required: ['spec_verdict', 'quality_verdict', 'findings'],
}
const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, severity_ok: { type: 'boolean' }, reason: STR },
  required: ['refuted', 'severity_ok', 'reason'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: { id: STR, result: { type: 'string', enum: ['fixed', 'parked', 'blocked'] }, evidence: STR, ruling: STR },
        required: ['id', 'result', 'evidence', 'ruling'],
      },
    },
    files_changed: STRS, review_package: STR, report_path: STR,
  },
  required: ['items', 'files_changed', 'review_package', 'report_path'],
}
const REREVIEW_SCHEMA = {
  type: 'object',
  properties: {
    prior: {
      type: 'array',
      items: { type: 'object', properties: { id: STR, closed: { type: 'boolean' }, note: STR }, required: ['id', 'closed', 'note'] },
    },
    new_findings: { type: 'array', items: FINDING },
  },
  required: ['prior', 'new_findings'],
}
const BOOK_SCHEMA = {
  type: 'object',
  properties: { sha: STR, pushed: { type: 'boolean' }, leftovers: STRS, notes: STR },
  required: ['sha', 'pushed', 'leftovers', 'notes'],
}
const CLOSE_SCHEMA = {
  type: 'object',
  properties: { pr_url: STR, head: STR, report_path: STR, owner_summary: STR },
  required: ['pr_url', 'head', 'report_path', 'owner_summary'],
}

// ---------- shared context ----------
const COMMON = `You are one agent in an unattended workflow running phase ${PHASE} of Wave 1 (W1, the patient-safety crisis path) of the Solace-AI MVP remediation program. The owner is away; nobody will answer questions. Decide ambiguities yourself and record each one with: ${PY} docs/mvp/exec-state/tools/mvp_exec.py ledger "Ruling: <what> - <why> - <cost if wrong>"

Environment (Windows; repo ${REPO}; branch ${BRANCH}):
- Use the PowerShell tool; the Bash tool's hook breaks on ';' and quotes. Work from ${REPO}.
- Test Python: D:/tmp/mvp-exec/venv/Scripts/python.exe. Repo tools (mvp_exec.py, mvp_docs.py, pick_ready.py) run with ${PY}. Never pip install into anaconda or system-wide.
- Unit command for a path P: $env:PYTHONPATH="${REPO};${REPO}/src"; $env:ENVIRONMENT="test"; $env:TELEMETRY_ENABLED="false"; & "D:/tmp/mvp-exec/venv/Scripts/python.exe" -m pytest P -o "addopts=" -p no:logging -q
- E2E and real-infra commands: "Global Constraints" of docs/superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md (real infra: SOLACE_REQUIRE_REALINFRA=1).
- Where a plan or brief says checkpoint.ps1 or D:/tmp/mvp-exec/tools, use ${PY} docs/mvp/exec-state/tools/mvp_exec.py checkpoint <label> instead (ruling R-12). mvp_exec.py subcommands act when called and have no --help: checkpoint <label>, review-package <base> <head> <name> [paths], extract-brief <task>, ledger "<line>".
- Up to 6 task lanes share this working tree, each on files no other lane touches. If a regression fails in files you did not touch, wait a few minutes and re-run once. If it still fails outside your files, it is cross-talk from another lane (plan section 5.3): say so in your report, and do not edit those files.
- Commands longer than about 9 minutes: run them in the background and wait for completion.

Binding rules (CLAUDE.md):
- TDD: a real failing test first (paste the RED output), then GREEN, then the regression the brief names. Verify before you claim done.
- Never open, print or commit .env files or secret values; log names only. Synthetic data only.
- Security fails closed: strict environments (staging, production) refuse unsafe settings; dev and test boot with compose defaults.
- Never weaken, skip, xfail or delete a test, and never relax a guard, to get green.
- Platform facts in CLAUDE.md (IF-01 lifespan shape, solace_app pool, NULLIF RLS, bare imports, strictness) are binding.
- W1 pre-flight rulings R-1 to R-13 are in docs/mvp/exec-state/plan/W1-preflight-scan.md, part 3. Read the rows that name your task. Also read open entries in docs/mvp/exec-state/ledger/handoffs.md that name your task.
- Only the bookkeeping agent makes git writes. Implementers, fixers and reviewers never run git add, commit, stash, reset, checkout, switch or push.
- Docker: never prune volumes or tagged images, never docker rmi -f <id>, never docker compose down -v on the default dev project. Leave no containers or volumes from your runs.
- Your final message is data for the workflow, not prose for a human. Fill every schema field; use "" or [] when empty.
`

const LENS_TEXT = {
  spec: 'Spec and quality lens: does the diff do exactly what the brief requires (its tests, files, interfaces and exact values), and nothing else? Are the tests meaningful (would they fail if the change were reverted)? Does the style match the surrounding code? Are the rulings well-formed?',
  security: 'Security lens: look for fail-open paths, auth or service-token scope misuse, strict versus dev drift, RLS or acting_as_user gaps, secrets or PHI in logs, events or error bodies, and input variants that bypass a check (encodings, case, blank or unset values). For each test, would it fail if the fix were reverted?',
  crisis: 'Crisis-path lens: could this change let a crisis message go unscreened, return no resources, or let an alert be lost, duplicated, delayed or sent with message text? Check redelivery, restart, partial failure, ordering between the outbox commit and the response, and readiness that lies. Do the tests prove the failure modes, not only the happy path?',
}
const LENSES = {
  standard: [{ key: 'spec', model: 'sonnet' }],
  'security-critical': [{ key: 'spec', model: 'sonnet' }, { key: 'security', model: 'opus' }],
  'safety-critical': [{ key: 'spec', model: 'sonnet' }, { key: 'crisis', model: 'opus' }],
}
const VOTER_LENSES = [
  'Reproduce lens: try to show from the actual code, config or a cheap unit test that the claimed failure scenario cannot happen.',
  'Spec lens: check whether the brief, an interface contract, a ledger Ruling (docs/mvp/exec-state/ledger/progress.md) or a later task in the plan makes the behaviour intended or out of scope for this task.',
  'Severity lens: assume the defect is real and decide whether the stated severity is right; refute if the scenario needs conditions that cannot occur in this program.',
]

const modelOf = t => (t.model === 'opus' ? 'opus' : 'sonnet')
const briefPath = t => 'docs/mvp/exec-state/briefs/' + t.id + '-brief.md'

const IMPL = t => COMMON + `
Task ${t.id}: ${t.title}
Review class: ${t.review}. Real infra: ${t.real_infra ? 'yes' : 'no'}.
Your requirements are in ${briefPath(t)}. Read it first: it holds the exact values to use verbatim. If it is missing, run mvp_exec.py extract-brief ${t.id}.${PHASE === 'B' ? '\nThis is a service task: also follow docs/mvp/exec-state/briefs/phaseB-common.md.' : ''}
Steps:
1. Run mvp_exec.py checkpoint "${t.id}:start" and note the tree id it prints.
2. Follow the brief's steps. Write the failing test, run it and paste the real RED. Implement, then run to GREEN, then the brief's regression.
3. Edit only the files the brief declares. If another file is unavoidable, keep the edit minimal and ledger a Ruling.
4. Run mvp_exec.py checkpoint "${t.id}:head" and note the tree id.
5. Write docs/mvp/exec-state/reports/${t.id}-report.md: what you did, RED and GREEN evidence, regression counts, files, start and head trees, rulings and concerns.
6. Make the review package: mvp_exec.py review-package <start tree> <head tree> ${t.id} <every file you changed>. Return the path it prints.
Do not dispatch subagents. Return BLOCKED (with the reason in concerns) only when you cannot finish; say exactly what is missing.`

const REVIEW = (t, impl, lens) => COMMON + `
You are a task reviewer; read-only. Do not edit files or run git writes; cheap unit tests are fine.
Task ${t.id}: ${t.title} (review class ${t.review}).
Read the brief ${briefPath(t)}, the implementer's report ${impl.report_path} and the review package ${impl.review_package} (commit list, stat and full diff).
${LENS_TEXT[lens.key]}
Report only real defects with evidence, and try to refute each candidate yourself before you report it. Use ids like ${t.id.replace('WS-', 'W').replace('.', '')}-${lens.key.slice(0, 3).toUpperCase()}-1. spec_verdict is PASS when the task's required behaviour is present. quality_verdict is APPROVED when no important finding remains.`

const VOTE = (t, f, i) => COMMON + `
You are a skeptic; read-only. Try to REFUTE this finding from the review of task ${t.id} (brief ${briefPath(t)}). Default to refuted=true if, after checking the code, you are not convinced it is real.
${VOTER_LENSES[i]}
Finding:
${JSON.stringify(f, null, 1)}`

const FIX = (t, impl, open, r) => COMMON + `
Fix round ${r} for task ${t.id}: ${t.title} (review class ${t.review}).
Read the brief ${briefPath(t)}, the report ${impl.report_path} and the review package ${impl.review_package}.
Each finding below survived a skeptic majority. For each one: reproduce it, write the failing test, fix the root cause, and run the brief's regression.
You may park only a minor finding that touches none of safety, PHI or security, with a Ruling (result=parked, ruling filled in). Everything else is fixed or blocked with evidence.
Checkpoint "${t.id}:fix${r}-start" before and "${t.id}:fix${r}-head" after. Append a "Fix round ${r}" section to ${impl.report_path}. Make a review package of the fix diff: review-package <start> <head> ${t.id}-fix${r} <files>.
Findings:
${JSON.stringify(open, null, 1)}`

const REREVIEW = (t, fx, open) => COMMON + `
Scoped re-review for task ${t.id} (review class ${t.review}); read-only, cheap unit tests allowed.
Read the fix package ${fx.review_package} and the report ${fx.report_path}.
1. For each prior finding below, set closed to true or false, with a one-line note naming the test and the code line.
2. Then look only at the fix diff for new defects. Would a test fail if the fix were reverted? Was a guard relaxed or a test weakened? Report only findings with evidence.
Prior findings:
${JSON.stringify(open.map(f => ({ id: f.id, severity: f.severity, file: f.file, problem: f.problem })), null, 1)}
Fix result:
${JSON.stringify(fx.items, null, 1)}`

const BOOK = (t, status, files, note, rulings) => COMMON + `
Stage: bookkeeping for task ${t.id} (${t.title}). Outcome: ${status}. You are the only agent that runs git writes; stay on ${BRANCH}.
1. Commit the task's files: git add -- <files>, then git commit -m "<message>" -- <files>, using only the paths below that exist and have changes. If index.lock is busy, wait and retry.
   Message: "${t.id}: ${status === 'done' ? 'done' : 'blocked'} (${t.title.slice(0, 60)})". Body: ${note}. End with a blank line and "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>".
   Files: ${JSON.stringify(files)}
2. Tracker: ${PY} docs/mvp/tools/mvp_docs.py set ${t.id} ${status} --note "<sha7>; ${note.replace(/"/g, "'").slice(0, 200)}"
3. Ledger: mvp_exec.py ledger "Task ${t.id}: ${status === 'done' ? 'complete' : 'blocked'} (<sha7>)". Then one ledger line per item here: ${JSON.stringify(rulings)}
4. Commit docs/mvp/TRACKER.md, docs/mvp/README.md, docs/mvp/exec-state/ledger/progress.md and docs/mvp/exec-state/snapshots/snapshots.log with pathspecs: "${t.id}: tracker and ledger".
5. git push origin ${BRANCH}. Retry once on failure. Never push to main and never force.
leftovers = the git status --porcelain lines whose paths are not in the file list (do not commit them; other lanes may own them).`

// ---------- preflight ----------
phase('Preflight')
const pre = await agent(COMMON + `
Stage: preflight. Do not edit tracked files.
1. git status --porcelain must be empty; otherwise ok=false and abort_reason="working tree not clean: <paths>".
2. git fetch origin. Switch to ${BRANCH} (git switch ${BRANCH}; if it exists only on origin, git switch -c ${BRANCH} origin/${BRANCH}). If origin/${BRANCH} exists and is ahead, git pull --ff-only. head = git rev-parse HEAD.
3. docker info and docker compose version -> docker_ok. If the daemon is down, start Docker Desktop (Start-Process "C:/Program Files/Docker/Docker/Docker Desktop.exe") and poll for up to 5 minutes. Report free GB on C: and D: and free RAM in notes.
4. done_ids: every task id whose Status cell in docs/mvp/TRACKER.md is done (a short Python read of the table is fine).
5. Ledger: "W1 phase ${PHASE} workflow started on ${BRANCH} at <head 7 chars>". Do not commit it.
ok=true unless step 1 or 2 failed.`, { label: 'preflight', phase: 'Preflight', schema: PRE_SCHEMA, model: 'sonnet' })
if (!pre || !pre.ok) {
  log('Stopped at preflight: ' + (pre ? pre.abort_reason : 'the preflight agent failed'))
  return { stopped: 'preflight', pre }
}
const done = new Set(pre.done_ids)
const tasks = SCHEDULE.filter(t => t.phase === PHASE && !done.has(t.id))
if (PHASE === 'B') {
  const openA = SCHEDULE.filter(t => t.phase === 'A' && !done.has(t.id)).map(t => t.id)
  if (openA.length) { log('Phase B refused: phase A not done (R-2): ' + openA.join(', ')); return { stopped: 'phase A open', openA } }
}
log(`Phase ${PHASE}: ${tasks.length} task(s) to run on ${BRANCH} at ${pre.head.slice(0, 7)}; docker_ok=${pre.docker_ok}`)
if (!pre.docker_ok && tasks.some(t => t.real_infra)) log('Docker is not usable: real-infra tasks will report BLOCKED.')

// ---------- per-task lane ----------
let bookChain = Promise.resolve()
function book(t, status, files, note, rulings) {
  const p = bookChain.then(() => agent(BOOK(t, status, files, note, rulings),
    { label: 'book:' + t.id, phase: 'Bookkeeping', schema: BOOK_SCHEMA, model: 'haiku' }))
  bookChain = p.catch(() => null)
  return p
}

async function reviewTask(t, impl) {
  const lenses = LENSES[t.review] || LENSES.standard
  const reviews = await parallel(lenses.map(l => () =>
    agent(REVIEW(t, impl, l), { label: 'review:' + t.id + ':' + l.key, phase: 'Dispatch', schema: REVIEW_SCHEMA, model: l.model })))
  const findings = reviews.filter(Boolean).flatMap(r => r.findings)
  const judged = await parallel(findings.map(f => async () => {
    const n = f.severity === 'important' ? 3 : 1
    const votes = (await parallel(Array.from({ length: n }, (_, i) => () =>
      agent(VOTE(t, f, i), { label: 'skeptic:' + f.id + ':' + i, phase: 'Dispatch', schema: VERDICT, model: 'sonnet' })))).filter(Boolean)
    const refutes = votes.filter(v => v.refuted).length
    return { ...f, survives: votes.length > 0 && refutes * 2 < votes.length }
  }))
  return {
    verdicts: reviews.filter(Boolean).map(r => r.spec_verdict + '/' + r.quality_verdict),
    open: judged.filter(Boolean).filter(f => f.survives),
    raised: findings.length,
  }
}

async function runLane(t) {
  try {
    const impl = await agent(IMPL(t), { label: 'impl:' + t.id, phase: 'Dispatch', schema: IMPL_SCHEMA, model: modelOf(t) })
    if (!impl || impl.status === 'BLOCKED' || !impl.review_package) {
      const why = impl ? impl.concerns : 'implementer agent failed'
      await book(t, 'blocked', impl ? [...impl.files_changed, impl.report_path] : [], 'implementer blocked: ' + why, impl ? impl.rulings : [])
      return { id: t.id, ok: false, why }
    }
    const rv = await reviewTask(t, impl)
    let open = rv.open
    const parked = []
    let files = [...impl.files_changed, impl.report_path, impl.review_package]
    let rounds = 0
    while (open.length && rounds < MAX_FIX_ROUNDS) {
      rounds++
      const fixModel = rounds >= 3 ? 'opus' : modelOf(t)
      const fx = await agent(FIX(t, impl, open, rounds), { label: 'fix' + rounds + ':' + t.id, phase: 'Dispatch', schema: FIX_SCHEMA, model: fixModel })
      if (!fx) break
      files = files.concat(fx.files_changed, [fx.review_package])
      const parkedIds = new Set(fx.items.filter(i => i.result === 'parked' && i.ruling).map(i => i.id))
      open.filter(f => parkedIds.has(f.id)).forEach(f => parked.push({ id: f.id, ruling: fx.items.find(i => i.id === f.id).ruling }))
      const rr = await agent(REREVIEW(t, fx, open), { label: 're-review' + rounds + ':' + t.id, phase: 'Dispatch', schema: REREVIEW_SCHEMA, model: t.review === 'safety-critical' ? 'opus' : 'sonnet' })
      if (!rr) break
      const still = open.filter(f => !parkedIds.has(f.id) && !(rr.prior.find(p => p.id === f.id) || {}).closed)
      const fresh = rr.new_findings.filter(f => f.severity === 'important' || t.review !== 'standard')
      rr.new_findings.filter(f => f.severity === 'minor' && t.review === 'standard')
        .forEach(f => parked.push({ id: f.id, ruling: 'Ruling: park minor ' + f.id + ' (' + f.problem.slice(0, 120) + ') - standard task, touches no safety, PHI or security - a later task may need a small cleanup' }))
      open = [...still, ...fresh]
    }
    const ok = open.length === 0
    const note = `report ${impl.report_path}; review ${rv.verdicts.join(', ')}; ${rv.raised} raised, ${rv.open.length} confirmed, ${rounds} fix round(s), ${parked.length} parked` + (ok ? '' : `; open: ${open.map(f => f.id).join(', ')}`)
    const rulings = [...impl.rulings, ...parked.map(p => p.ruling)].filter(Boolean)
    if (!ok) rulings.push(`Ruling: ${t.id} blocked after ${rounds} fix round(s) with open findings ${open.map(f => f.id + ' (' + f.problem.slice(0, 80) + ')').join('; ')} - the review protocol does not park safety, PHI or security findings - its dependents wait`)
    const bk = await book(t, ok ? 'done' : 'blocked', [...new Set(files.filter(Boolean))], note, rulings)
    if (bk && bk.leftovers && bk.leftovers.length) log(t.id + ' leftovers after commit: ' + bk.leftovers.join(' | '))
    return { id: t.id, ok, why: ok ? '' : 'open findings: ' + open.map(f => f.id).join(', '), sha: bk ? bk.sha : '' }
  } catch (e) {
    return { id: t.id, ok: false, why: 'lane error: ' + String(e).slice(0, 300) }
  }
}

// ---------- continuous scheduler ----------
phase('Dispatch')
const PRI = { 'safety-critical': 0, 'security-critical': 1, standard: 2 }
const byId = new Map(tasks.map(t => [t.id, t]))
const finished = []
const failed = new Set()
const running = new Map()

function pickNext() {
  const keys = new Set([...running.keys()].flatMap(id => byId.get(id).mutex))
  let ri = [...running.keys()].filter(id => byId.get(id).real_infra).length
  const picks = []
  const ready = tasks
    .filter(t => !running.has(t.id) && !done.has(t.id) && !failed.has(t.id))
    .filter(t => t.deps.every(d => done.has(d) || !byId.has(d) && SCHEDULE.every(s => s.id !== d)))
    .sort((a, b) => b.height - a.height || (PRI[a.review] ?? 3) - (PRI[b.review] ?? 3) || a.id.localeCompare(b.id))
  for (const t of ready) {
    if (running.size + picks.length >= MAX_LANES) break
    if (t.real_infra && ri >= MAX_RI) continue
    if (t.mutex.some(k => keys.has(k))) continue
    picks.push(t)
    t.mutex.forEach(k => keys.add(k))
    if (t.real_infra) ri++
  }
  return picks
}

while (true) {
  for (const t of pickNext()) {
    log('dispatch ' + t.id + ' (' + t.review + ', ' + modelOf(t) + (t.real_infra ? ', real infra' : '') + ')')
    running.set(t.id, runLane(t))
  }
  if (!running.size) break
  const res = await Promise.race([...running.values()])
  running.delete(res.id)
  finished.push(res)
  if (res.ok) done.add(res.id)
  else { failed.add(res.id); log(res.id + ' BLOCKED: ' + res.why) }
  log(`${res.id} ${res.ok ? 'done' : 'blocked'}; ${finished.filter(r => r.ok).length}/${tasks.length} done, ${running.size} running`)
}
await bookChain

const notRun = tasks.filter(t => !done.has(t.id) && !failed.has(t.id)).map(t => ({ id: t.id, waits_on: t.deps.filter(d => !done.has(d)) }))
if (notRun.length) log(notRun.length + ' task(s) never became ready (a dependency is blocked): ' + notRun.map(n => n.id).join(', '))

// ---------- phase close ----------
phase('Phase close')
const facts = { phase: PHASE, branch: BRANCH, start_head: pre.head, finished, not_run: notRun }
const close = await agent(COMMON + `
Stage: phase close for W1 phase ${PHASE}. You may run git writes on ${BRANCH} only (pathspec commits, push; never main, never force).
1. Run ${PY} docs/mvp/tools/mvp_docs.py check and fix drift in TRACKER.md.
2. Write docs/mvp/exec-state/reports/W1-phase-${PHASE}-report.md. Include:
   - per task: outcome, sha, findings confirmed, parked and open;
   - blocked tasks with the reason and what unblocks them;
   - tasks never dispatched and the dependency each waits on;
   - every Ruling this run added to the ledger (grep the ledger from the "W1 phase ${PHASE} workflow started" line);
   - leftovers in git status.
3. Add a LOG.md entry at the top (format in the file), and update the "Current state" section of docs/mvp/README.md and CLAUDE.md's state lines to the real W1 status. Keep the same brief style and keep every rule.
4. Commit those files with pathspecs as "W1 phase ${PHASE}: close", then git push origin ${BRANCH}.
5. If no PR from ${BRANCH} to main is open, open one as a draft: gh pr create --draft --base main --head ${BRANCH} --title "W1: patient-safety crisis path (in progress)". Otherwise add a comment with the phase summary (gh pr comment).
   The body or comment covers: phase ${PHASE} results, blocked items, rulings, owner actions (X-15 clinician for lexicon v1, X-04, X-05) and the next step. End it with a blank line and "🤖 Generated with [Claude Code](https://claude.com/claude-code)". Never merge.
6. owner_summary: 6 to 10 short plain-language lines for the owner, using they/them for the owner.
Facts:
${JSON.stringify(facts, null, 1)}`, { label: 'phase close + PR', phase: 'Phase close', schema: CLOSE_SCHEMA, model: 'sonnet' })

return { phase: PHASE, done: finished.filter(r => r.ok).map(r => r.id), blocked: finished.filter(r => !r.ok), not_run: notRun, close }
