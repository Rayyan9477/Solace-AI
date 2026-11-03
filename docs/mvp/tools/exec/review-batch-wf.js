export const meta = {
  name: "task-review-batch",
  description: "Lens-diverse reviews of several task diffs at once, each finding verified by a skeptic panel",
  phases: [
    { title: "Review", detail: "one reviewer per (task, lens)" },
    { title: "Verify", detail: "3 skeptics for important findings, 1 for minor; majority refute kills" },
  ],
}

// args: { common: [lines], items: [{ task, prefix, ctx: [lines], lenses: [{key, model, text}] }] }
const COMMON = (args.common || []).join("\n")

const REVIEW_SCHEMA = {
  type: "object",
  properties: {
    spec_verdict: { type: "string", enum: ["PASS", "FAIL"] },
    quality_verdict: { type: "string", enum: ["APPROVED", "CHANGES_REQUESTED"] },
    prior_findings: {
      type: "array",
      items: {
        type: "object",
        properties: { id: { type: "string" }, closed: { type: "boolean" }, note: { type: "string" } },
        required: ["id", "closed", "note"],
      },
    },
    findings: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          severity: { type: "string", enum: ["important", "minor"] },
          file: { type: "string" },
          line: { type: "integer" },
          problem: { type: "string" },
          failure_scenario: { type: "string" },
          fix: { type: "string" },
          evidence: { type: "string" },
        },
        required: ["id", "severity", "file", "line", "problem", "failure_scenario", "fix", "evidence"],
      },
    },
  },
  required: ["spec_verdict", "quality_verdict", "prior_findings", "findings"],
}

const VERDICT_SCHEMA = {
  type: "object",
  properties: { refuted: { type: "boolean" }, severity_ok: { type: "boolean" }, reason: { type: "string" } },
  required: ["refuted", "severity_ok", "reason"],
}

const VOTER_LENSES = [
  "Reproduce lens: try to show from the actual code, config rendering or a cheap unit test that the claimed failure scenario cannot happen.",
  "Spec lens: check whether the brief, the interface contract, a recorded controller ruling (D:\\tmp\\mvp-exec\\ledger\\progress.md, lines with Ruling) or a later task in the plan makes the behaviour intended or out of scope for this task.",
  "Severity lens: assume the defect is real and decide whether the stated severity is right; refute if the scenario needs conditions that cannot occur in this program.",
]

const PAIRS = []
for (const it of args.items) for (const l of it.lenses) PAIRS.push({ it, l })

function ctxOf(it) { return COMMON + "\n" + it.ctx.join("\n") }

const results = await pipeline(
  PAIRS,
  p => agent(ctxOf(p.it) + "\n\n" + p.l.text +
    "\n\nReport only real defects with evidence; try to refute each candidate yourself before you report it. Use ids like " +
    p.it.prefix + "-" + p.l.key.toUpperCase().slice(0, 3) + "-1. spec_verdict is PASS when the task's required behaviour is present; " +
    "quality_verdict is APPROVED when no important finding remains. prior_findings is [] when there was no earlier round.",
    { label: "review:" + p.it.task + ":" + p.l.key, phase: "Review", schema: REVIEW_SCHEMA, model: p.l.model }),
  (review, p) => {
    if (!review) return { task: p.it.task, lens: p.l.key, review: null, verified: [] }
    return parallel(review.findings.map(f => () => {
      const voters = f.severity === "important" ? VOTER_LENSES : VOTER_LENSES.slice(0, 1)
      return parallel(voters.map((lens, i) => () =>
        agent(ctxOf(p.it) + "\n\nYou are a skeptic. Try to REFUTE this finding from a task review. Default to refuted=true if, after checking the code, you are not convinced it is real.\n" +
          lens + "\n\nFinding:\n" + JSON.stringify(f, null, 1),
          { label: "verify:" + f.id + ":" + i, phase: "Verify", schema: VERDICT_SCHEMA, model: "sonnet" })
      )).then(vs => {
        const votes = vs.filter(Boolean)
        const refutes = votes.filter(v => v.refuted).length
        return { ...f, votes, survives: votes.length > 0 && refutes * 2 < votes.length }
      })
    })).then(vf => ({ task: p.it.task, lens: p.l.key, review, verified: vf }))
  }
)
const minors = results.filter(Boolean).flatMap(r => r.verified).filter(f => f.severity === "minor").length
if (minors) log(minors + " minor finding(s) verified by a single skeptic only (important ones got 3).")
return results.filter(Boolean)
