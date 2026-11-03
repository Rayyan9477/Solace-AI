# MVP program log

A dated history of the program: reviews, decisions, plan changes, wave barriers and owner actions. Newest entries go at the top. One entry per event; link the document that holds the detail.

Entry format:

```
## YYYY-MM-DD: <event>
- What happened, in one or two lines.
- Numbers that changed (tasks, register entries, test counts, tree id).
- Links: <files>.
```

## 2026-11-02: "Go": execution started (W0 entry)
- The owner said go and asked for a slow pace. AWS keeps the existing profile `MedCodex-Local`. The owner confirmed the BAA is in place (X-02 done).
- W0 entry: tree `b309d50`, exported to `D:\tmp\mvp-exec\barriers\W0-entry-b309d50.zip`.
- The venv `D:\tmp\mvp-exec\venv` was created with system site packages. Installed into it only: aiokafka 0.12.0, langgraph-checkpoint-postgres 3.0.5, PyJWT 2.15.1 and orjson 3.13.0 (needed by langgraph-checkpoint-postgres). Anaconda is unchanged; its PyJWT is still 2.10.1.
- The freeze before and after is in `D:\tmp\mvp-exec\venv-freeze-*.txt`.

## 2026-10-19: Owner decisions answered; plans updated; tracking docs created
- The owner answered all 31 decisions. Six changed the recommended default:
  - D-04: Bedrock for the LLM.
  - D-07: SOPS + age for secrets.
  - D-17: no commits at all.
  - D-20: a dedicated venv.
  - D-25: AWS on the existing IAM user.
  - D-29: several countries, US first.
- Clinical and legal items are built on their defaults, with sign-off before go-live. See [DECISIONS.md](DECISIONS.md).
- A read-only IAM simulation of the existing user `MedCodex-Local` showed Bedrock allowed, and CloudFormation, EC2, SSM, S3, KMS and SES denied. This is owner action X-01 in [OPEN-ITEMS.md](OPEN-ITEMS.md).
- The workstream plans were updated for the six changes. Each plan's Review responses lists them under "decision 2026-10-19". The master plan was re-rendered with the answers.
- The tracking set was created: README.md, DECISIONS.md, TRACKER.md, OPEN-ITEMS.md, LOG.md and tools/mvp_docs.py. The register now derives its statuses from TRACKER.md.

## 2026-10-19: Remediation plan completed
- 314 tasks in 18 workstream plans, covering 345 of 345 in-scope register entries. The automated plan check ended with 0 errors.
- A reconciliation pass settled 11 cross-plan ownership conflicts. An independent crisis-path review found 13 gaps, all fixed in the plans. Interface amendments followed for IF-05, IF-09 and IF-12.
- Links: [master plan](../superpowers/plans/2026-10-18-mvp-remediation.md), [interfaces and ownership](../superpowers/plans/2026-10-18-mvp-remediation/00-interfaces-and-ownership.md).

## 2026-10-18: Scope confirmed, register compiled
- Scope confirmed by the owner: "Pilot + enterprise basics" (a single-host pilot, plus backups with a tested restore, MFA, an audit trail and per-environment secrets). HA/DR and scaling are out of scope.
- The owner kept the no-commit rule.
- Issue register compiled: 356 entries (42 blocker, 73 high, 126 medium, 104 low, 11 post-MVP). See [MVP-ISSUE-REGISTER.md](../MVP-ISSUE-REGISTER.md).

## 2026-10-17: Earlier review verified
- The earlier 128-finding review was re-verified: 127 of 128 findings are real. See [MVP-BUG-REVIEW.md](../MVP-BUG-REVIEW.md).

## 2026-10-11: Full-picture assessment
- Nothing builds, boots or migrates as deployed, and no crisis alert reaches a human. 5,377 local tests pass, but CI has never passed.
