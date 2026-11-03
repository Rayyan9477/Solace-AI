# MVP open items

Owner actions, sign-offs and risks that sit outside the task plans. Update an item's *Status* and *Updated* cells when it changes, and add a dated line to [LOG.md](LOG.md).

Status values: `open`, `in progress`, `done`, `n/a`.

## Owner actions

| ID | Action | Decision | Needed by | Status | Updated | Notes |
|---|---|---|---|---|---|---|
| X-01 | Attach `infrastructure/aws-staging/solace-staging-deployer-policy.json` to the existing IAM user `MedCodex-Local` (option 2 in AWS-STAGING-IAM-SETUP.pdf). | D-25, D-09, D-12, D-04 | W4 (off-host backup target), W5 (release gate), go-live | `open` | 2026-11-02 | Owner: use the existing profile for now (2026-11-02). IAM simulation on 2026-10-19: only `bedrock:InvokeModel` and `bedrock:ListFoundationModels` allowed; CloudFormation, EC2, SSM, IAM role creation, S3, KMS and SES denied. |
| X-02 | Accept the AWS BAA in AWS Artifact and record the date here. | D-04, D-12 | go-live (any real-user PHI to Bedrock or in SES alerts) | `done` | 2026-11-02 | Owner confirmed on 2026-11-02 that the BAA is in place. |
| X-03 | Set up the SES sending identity: verify the domain with DKIM, request production access (leave the SES sandbox), and create the SES SMTP credentials, which go into the SOPS secrets. | D-12 | W5 staging, go-live | `open` | 2026-10-19 | Needs X-01. |
| X-04 | Name the clinical approver and complete the sign-off of D-05, D-06, D-21, D-22, D-28, D-29 (en-US resources) and D-30. | clinical set | go-live (W1 builds on the defaults) | `open` | 2026-10-19 |  |
| X-05 | Name at least two on-call clinicians (primary and backup, with E.164 phone numbers) and the fallback mailbox. | D-13 | go-live | `open` | 2026-10-19 | The release-gate rehearsal uses synthetic clinicians. |
| X-06 | Legal review of D-23 (consents), D-24 (erasure and retention), D-31 (go-live gate) and the BAA scope for D-04 and D-12. | legal set | go-live | `open` | 2026-10-19 |  |
| X-07 | Generate one age key pair per environment (staging, production). Keep the private keys off the host, decide who holds them and how they are backed up, and give the controller only the public recipients. | D-07 | public recipients by the end of W3; private key at the W5 staging deploy | `open` | 2026-10-19 | Tests use throwaway keys. No private key ever goes into the repo, a prompt or a report. |
| X-08 | Choose the Bedrock model id(s) for `BEDROCK_MODEL_ID` after the controller lists what is available in us-east-1 (foundation models and inference profiles). | D-04 | W5 staging deploy | `open` | 2026-10-19 | Never guessed in code or tests. |
| X-09 | List any further pilot countries and supply clinician-verified crisis resources for each before its users join. | D-29 | before the first non-US user | `open` | 2026-10-19 | W1 ships en-US plus the generic fallback. |
| X-10 | Decide the commit policy after the program. | D-17 | after W5 | `open` | 2026-10-19 | Nothing is committed until then. |
| X-11 | Approve `git config gc.pruneExpire never` in this repo for the program's duration, so checkpoint trees cannot be pruned, or accept the barrier zips as the only protection. | D-17 | W0 entry (recommended) | `open` | 2026-10-19 | Checkpoint trees are unreferenced git objects; by default git gc may prune them after about two weeks. |
| X-12 | Name the off-machine location where barrier zips are copied (another disk or cloud storage). | D-17 | W0 barrier | `open` | 2026-10-19 |  |
| X-13 | Say "go" to start W0. | all | W0 | `done` | 2026-11-02 | Owner said go on 2026-11-02. |

## Sign-offs

| Area | Covers | Approver | Date | Status | Evidence |
|---|---|---|---|---|---|
| Clinical | D-05 lexicon and thresholds, D-06 crisis watch and re-alerting, D-21 clinician access scope, D-22 screening architecture, D-28 single alert path, D-29 crisis resources per locale, D-30 no diagnosis labels | _(name)_ | | `open` | |
| Legal | D-23 consents, D-24 erasure and retention, D-31 go-live gate, BAA scope for D-04 and D-12 | _(name)_ | | `open` | |
| Go-live gate (D-31) | W5 release gate passed twice; X-02 BAA recorded; clinical sign-off; X-05 on-call named; D-09 restore-drill evidence | owner | | `open` | |

## Risks

Risks from the plan (master plan, section 8), with status. Add new risks at the bottom with the next id.

| ID | Risk and mitigation | Status | Updated |
|---|---|---|---|
| R-01 | Clinical sign-off (D-05, D-06, D-22, D-28, D-29) may lag engineering. W1 proceeds on conservative defaults with the owner's acceptance recorded, but W5 and go-live block without a named clinician approver, and a late lexicon change re-runs the phrase suite across five detectors. | `open` | 2026-10-19 |
| R-02 | Without BAAs for Portkey/Anthropic or the SMTP relay (D-04, D-12), LLM layers stay disabled for real users and alerts cannot carry user identity. The product must remain safe and usable with templates and deterministic screening, and the pilot's value may shrink. | `open` | 2026-10-19 |
| R-03 | A recall-first lexicon can flood clinicians with false positives, and clinicians who learn to ignore pages defeat the safety path. Controls: the negative phrase set, ELEVATED-means-resources-only, per-episode coalescing, acknowledgement and monitoring of page volume. | `open` | 2026-10-19 |
| R-04 | The IF-01 bootstrap is the largest design risk: all nine services adopt it in W0, so a defect cascades everywhere. Mitigations: Opus design, AGY adversarial review before phase B, behaviour-preserving adoption, the conformance test and boot smoke in both profiles. | `open` | 2026-10-19 |
| R-05 | Running as solace_app with RLS from W0 will surface failures in flows without request context (registration, consumers, reapers, CLIs). Without acting_as_user discipline and realinfra tests, these fail silently as empty reads. | `open` | 2026-10-19 |
| R-06 | In one shared working tree, a half-finished phase-A platform edit breaks every lane's tests, and a missed handover (the user-service split files, slot files) produces silent edit conflicts. Mitigations: phase gating, the pre-dispatch disjointness check, owning-suite runs per task, full regression only at barriers and rollback.ps1 per declared paths. | `open` | 2026-10-19 |
| R-07 | Docker capacity: two concurrent full stacks (about 15 containers each) can exhaust RAM; the plan relies on at most 2 e2e lanes, Weaviate disabled by default and stack reuse within a lane. | `open` | 2026-10-19 |
| R-08 | The working tree stays uncommitted until the owner acts (D-17). One destructive git command could lose weeks of work; checkpoint trees are the only safety net, and MVP-008 cannot close without the owner's commit. | `open` | 2026-10-19 |
| R-09 | No-commit-on-failure consumer semantics risk poison-message loops and head-of-line blocking of crisis alerts; bounded in-place retries, the malformed-message DLQ path and realinfra poison tests are mandatory. | `open` | 2026-10-19 |
| R-10 | langgraph-checkpoint-postgres is not installed locally and uses psycopg rather than asyncpg; its pinned DDL (including CREATE INDEX CONCURRENTLY) must be applied through slot 017 correctly, or checkpoint durability (MVP-066) slips. D-20 approval is a W0 entry dependency. | `open` | 2026-10-19 |
| R-11 | W2 is the largest wave (about 90 issues across 12 workstreams), and the 6-implementer cap makes it the long pole; low/medium domain items are already deferred to W5, and further slips need explicit owner acceptance. | `open` | 2026-10-19 |
| R-12 | AWS deployer permissions are blocked (D-25), so the release gate may be provable only on the local strict profile, leaving TLS/ACME, DNS, host alarms and S3 backups unproven on the real host. | `open` | 2026-10-19 |
| R-13 | The mobile client currently targets Supabase and mocks (D-11); if it is not re-pointed at IF-12, the hardened backend contract goes unused in the pilot. | `open` | 2026-10-19 |
| R-14 | E2E flakiness under timing (Kafka rebalance, SMTP retries, re-page intervals) could mask real failures or block barriers; deterministic fakes, shortened configurable intervals, bounded generous waits and the twice-green release rule are the controls. | `open` | 2026-10-19 |
| R-15 | Erasure versus duty-of-care retention (D-24) is a legal question: pseudonymizing crisis episodes and escalation records may conflict with counsel's view in either direction. | `open` | 2026-10-19 |
| R-16 | Two refusals still skip the screen (crisis-review residual): WS frames over 8 KB (WS-11.17) and REST bodies over 4000 characters (422, WS-11.4). A long crisis message gets no resources and no page. This is raised for the D-22 sign-off: either accept it, or screen the first 4000 characters before refusing. | `open` | 2026-10-19 |
| R-17 | From W2, the WS-11.7 thread-ownership lookup sits on the chat path, so a Postgres outage refuses the turn before the screen runs. WS-10's crisis path is designed to stay live during a database outage (the in-band resources need no database). Executors should keep the lookup after the screen, or fail open to the local screen, and the W2 Postgres-outage E2E (WS-05.8) must cover it. | `open` | 2026-10-19 |
| R-18 | No commits (D-17) leaves the working tree as the only full copy, and checkpoint trees are unreferenced git objects that git gc can prune after about two weeks. Mitigation: each barrier tree is exported with git archive to a zip outside the repo and copied off the machine by the owner. The owner may also approve git config gc.pruneExpire never for the program's duration (OPEN-ITEMS X-11). Never run git clean, reset or checkout in this tree. | `open` | 2026-10-19 |
| R-19 | AWS provisioning is blocked: the existing user MedCodex-Local can only call Bedrock (IAM simulation of 2026-10-19). If the deployer policy is not attached before W4, the off-host backup target is unproven; if not attached by W5 entry, the release gate stops (D-25). The owner action is X-01. | `open` | 2026-10-19 |
| R-20 | The venv baseline may differ from the 5,377 anaconda baseline (shadowed or upgraded packages, PyJWT >= 2.13). W0 entry records the per-suite venv baseline, and any drop needs a ledger Ruling before tasks start (D-20). | `open` | 2026-10-19 |
| R-21 | Bedrock model availability and quotas for the chosen Anthropic model in us-east-1 are unverified. BEDROCK_MODEL_ID is set at deploy after checking ListFoundationModels and inference profiles, and LLM features stay off for real users until the AWS BAA is recorded (D-04, X-02, X-08). | `open` | 2026-10-19 |
| R-22 | Multi-country (D-29): users from a country without verified resources would get only the generic fallback. Mitigation: pilot admission stays a named cohort (D-31), and each new locale needs clinician-verified resources before its users join (X-09). | `open` | 2026-10-19 |

