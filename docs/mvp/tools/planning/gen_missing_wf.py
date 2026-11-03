"""Generate wf_missing.js (one Opus planner per still-missing workstream) from wf_detail_pairs.js."""
src = open(r"D:\tmp\mvp-plan\wf_detail_pairs.js", encoding="utf-8").read()
ctx = src[src.index("const CTX = `"):src.index("const TASK_INDEX = {")]
fmt_rules = src[src.index("const FORMAT = `"):src.index("const results = await")]
fmt_rules = fmt_rules.replace(
    "or under tests/integration/<your-slug>/.",
    "or under tests/realinfra/<your-slug>/ (real-infra tests, per the Global Constraints) or tests/integration/<your-slug>/.")
assert "tests/realinfra/<your-slug>/" in fmt_rules

head = """export const meta = {
  name: 'mvp-plan-detail-missing',
  description: 'MVP remediation planning stage 2b: one Opus planner per workstream whose plan was not written before the session limit',
  phases: [{ title: 'Detail', detail: 'Opus 5.5 detail planner per missing workstream' }],
}

const PLAN_DIR = 'docs/superpowers/plans/2026-10-18-mvp-remediation'

"""
summary = """const SUMMARY = {
  type: 'object',
  properties: {
    ws: { type: 'string' },
    file: { type: 'string' },
    task_count: { type: 'integer' },
    uncovered_ids: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string', description: 'one short paragraph: open risks; plus a line "REVIEW_FOCUS: 1=WS-xx.k; ..." or "REVIEW_FOCUS: none"' },
  },
  required: ['ws', 'file', 'task_count', 'uncovered_ids', 'notes'],
}

"""
body = r"""
const results = await parallel(args.items.map(w => async () => {
  const file = `${PLAN_DIR}/${w.id}-${w.slug}.md`
  const prompt = `${CTX}

TASK: Write the detailed TDD implementation plan for workstream ${w.id} "${w.name}" to ${file}. This is the only file you may write.
THE APPROVED SKELETON is D:\\tmp\\mvp-plan\\skeleton.json. Read your workstream entry (${w.id}): waves, owned_paths, issue_ids, depends_on, produces/consumes and notes. Also read the full skeleton's interfaces, shared_files, decisions, global_constraints, review_focus and test_strategy, and the other workstreams' owned_paths so you respect ownership.
YOUR REGISTER ENTRIES: every id in your issue_ids. Read their full details (scenario, evidence, absorbed ids) in D:\\tmp\\mvp-plan\\register_compiled.json, then read the cited code.
PLANS ALREADY WRITTEN: 13 of the 18 workstream plans exist in ${PLAN_DIR}/. Your closest partner plan is ${PLAN_DIR}/${w.partner}: read it first and keep shared interfaces, task ids you depend on and cross-workstream requests consistent with it. Consult any other written plan you interact with.
INCOMING REQUESTS: ${w.requests ? `D:\\tmp\\mvp-plan\\requests\\${w.id}.md lists the cross-workstream requests the written plans address to ${w.id}. Plan a task for each one (or add it to one of your tasks) in a wave no later than the requesting task's wave, and name the requesting task in that task's Produces line. If a request is wrong, say why in your Self-review instead.` : 'none were found in the written plans.'}
REVIEW FOCUS: the skeleton's review_focus lists five failure modes (number them 1-5). For each one that falls in your code, make sure one of your task's tests pins it.
${FORMAT}
${RULES}
When the file is complete, re-read it once against the Self-review checklist and fix what you find. Then return the short summary.`
  return agent(prompt, { label: `detail:${w.id}`, phase: 'Detail', schema: SUMMARY, model: 'opus' })
}))
return { summaries: results.filter(Boolean), failed: args.items.filter((w, i) => !results[i]).map(w => w.id) }
"""
out = head + ctx + summary + fmt_rules + body
open(r"D:\tmp\mvp-plan\wf_missing.js", "w", encoding="utf-8").write(out)
print("wrote", len(out), "chars")
