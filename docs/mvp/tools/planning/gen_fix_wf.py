"""Generate wf_fix_pairs.js from wf_detail_pairs.js so CTX, FORMAT and RULES stay verbatim.
Fixers edit plans in place and return a short summary; indexes are re-parsed from the markdown."""
src = open(r"D:\tmp\mvp-plan\wf_detail_pairs.js", encoding="utf-8").read()
ctx = src[src.index("const CTX = `"):src.index("const TASK_INDEX = {")]
fmt_rules = src[src.index("const FORMAT = `"):src.index("const results = await")]
fmt_rules = fmt_rules.replace(
    "or under tests/integration/<your-slug>/.",
    "or under tests/realinfra/<your-slug>/ (real-infra tests, per the Global Constraints) or tests/integration/<your-slug>/.")
assert "tests/realinfra/<your-slug>/" in fmt_rules

head = """export const meta = {
  name: 'mvp-plan-fix-pairs',
  description: 'MVP remediation planning stage 4: Opus fixers revise the workstream plans from automated checks, the AGY grounding review and cross-workstream requests',
  phases: [{ title: 'Fix', detail: 'Opus 5.5 fixer per group of workstream plans' }],
}

const PLAN_DIR = 'docs/superpowers/plans/2026-10-18-mvp-remediation'

"""
summary = """const FIX_SUMMARY = {
  type: 'object',
  properties: {
    plans: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          ws: { type: 'string' },
          fixed: { type: 'integer' },
          rejected: { type: 'integer' },
          new_tasks: { type: 'array', items: { type: 'string' } },
          review_focus: { type: 'string', description: 'the exact REVIEW_FOCUS line written into the plan' },
          open_issues: { type: 'string' },
        },
        required: ['ws', 'fixed', 'rejected', 'new_tasks', 'review_focus', 'open_issues'],
      },
    },
  },
  required: ['plans'],
}

"""
body = r"""
const results = await parallel(args.groups.map(group => async () => {
  const list = group.map((it, i) => `  ${i + 1}. ${it.ws}: plan ${PLAN_DIR}/${it.file} ; findings D:\\tmp\\mvp-plan\\fix\\${it.ws}.md`).join('\n')
  const prompt = `${CTX}

TASK: Revise these workstream plans in place:
${list}
These plan files are the only files you may write. For each plan, resolve every finding in its findings file. The findings come from (1) automated plan checks (Errors must be fixed; Warnings are fixed or justified), (2) an independent AGY (Gemini) grounding review that checked paths, symbols, imports, RED claims, commands and consumed interfaces against the code, and (3) cross-workstream requests addressed to the workstream that its plan does not cover yet. A finding can be wrong (the checker parses markdown heuristically, and AGY can misread code): verify each contested point against the code, then either fix the plan or reject the finding with file:line evidence.
Record every finding under a final section "## Review responses" (create it at the end of the plan), one line each: the finding in short, then "fixed" or "rejected: <reason with evidence>".
WORK ECONOMICALLY: do not re-read whole plan files. Grep for the task headings, sections and lines a finding touches, open only those, and edit them in place. Read code only where a finding needs it.
KEEP STABLE: the existing format and task ids. Never renumber; a new task takes the next free number. Keep the skeleton's interfaces (D:\\tmp\\mvp-plan\\skeleton.json) and the ownership boundaries. All 18 workstream plans now exist in ${PLAN_DIR}/: when a dependency names a task id of another workstream that does not exist, look up that workstream's plan and point the dependency at the task that really produces what you need (or keep it as a cross-workstream request if no task does). A request from another workstream that touches your files becomes a task (or joins an existing task), and that task's Produces line names the requesting task.
REVIEW FOCUS: the skeleton's review_focus lists five failure modes (number them 1-5 in skeleton order). For each one that falls in a revised workstream's code, make sure some task's test pins it. Write one line in the plan's Self-review exactly like "REVIEW_FOCUS: 1=WS-12.3,WS-12.7; 4=WS-12.9" (or "REVIEW_FOCUS: none").
${RULES}
When done, return the short summary for each plan, in the order listed above.`
  return agent(prompt, { label: `fix:${group.map(g => g.ws).join('+')}`, phase: 'Fix', schema: FIX_SUMMARY, model: 'opus' })
}))
const plans = results.filter(Boolean).flatMap(r => r.plans || [])
const want = args.groups.flat().map(g => g.ws)
const got = new Set(plans.map(p => p.ws))
return { plans, missing: want.filter(id => !got.has(id)) }
"""
out = head + ctx + summary + fmt_rules + body
open(r"D:\tmp\mvp-plan\wf_fix_pairs.js", "w", encoding="utf-8").write(out)
print("wrote", len(out), "chars")
