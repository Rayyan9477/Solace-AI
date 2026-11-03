# MVP owner decisions

The live record of every owner decision for the MVP remediation program. The master plan's section 3 is a snapshot of these answers as of 2026-10-19; this file wins if they differ.

**How to update.** Change a decision only with the owner's say-so:
1. Edit its entry: the *Decision* line, its *Status*, and a new row in its *History* table.
2. Add a dated line to [LOG.md](LOG.md).
3. Have the controller record a ledger Ruling and update every task named under *Plan impact* before any of them runs.

Sign-off fields are filled in when the named approver confirms, with the date and where the evidence lives.

**Status values:** `Decided (default)` means the plan's recommendation was accepted. `Decided (changed from default)` means the owner chose differently and the plans were updated. `Default accepted, sign-off pending` means engineering builds the default now, and a clinical or legal approver must confirm it before go-live (D-31).

## Summary

| ID | Topic | Owner role | Needed by | Status | Decision (short) |
|---|---|---|---|---|---|
| [D-01](#d-01) | What is the schema of record: the Alembic chain or the column names the repositories were written against? | ops | W0 | Decided (default) | Accepted as the default. |
| [D-02](#d-02) | Which database topology and roles should the pilot use? | security | W0 | Decided (default) | Accepted as the default. |
| [D-03](#d-03) | Which inter-service authentication model should services use? | security | W1 | Decided (default) | Accepted as the default. |
| [D-04](#d-04) | Which LLM and embedding vendors may receive real-user PHI, which BAAs are in place, and are Portkey semantic caching and memory's OpenAI embeddings acceptable? | legal | W1 | Decided (changed from default) | Anthropic models on AWS Bedrock under the AWS BAA are the only route for real-user PHI. Strict environments require LLM_PROVIDER=bedrock; Portkey stays for... |
| [D-05](#d-05) | Clinical sign-off: who approves the crisis lexicon, the CRITICAL threshold, the LLM/fallback fusion rule, negation policy and auto-escalation rules? | clinical | W1 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-06](#d-06) | Crisis-mode semantics: does crisis state persist per thread, how does it clear, and how are repeated alerts deduplicated? | clinical | W1 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-07](#d-07) | Which secrets manager holds per-environment secrets? | security | W3 | Decided (changed from default) | SOPS + age instead of AWS Secrets Manager: per-environment encrypted deploy/secrets/<env>.sops.yaml, one age key pair per environment with the private keys held... |
| [D-08](#d-08) | Which MFA method, and who must use it? | security | W4 | Decided (default) | Accepted as the default. |
| [D-09](#d-09) | Which backup tooling and RPO/RTO targets apply? | ops | W4 | Decided (default) | Accepted as the default. RPO 15 min, RTO 4 h. |
| [D-10](#d-10) | Where is the audit trail stored, and how long is it retained? | security | W3 | Decided (default) | Accepted as the default. |
| [D-11](#d-11) | Client contract: does the code or docs/API-HANDOFF.md define the REST/WS protocol and the OAuth handoff? | product | W1 | Decided (default) | Accepted as the default. |
| [D-12](#d-12) | Which notification providers deliver clinician alerts, and what may an alert contain? | ops | W1 | Decided (default) | Accepted as the default. Amazon SES email. |
| [D-13](#d-13) | How are clinician and admin accounts provisioned and placed on call? | product | W1 | Decided (default) | Accepted as the default. At least two on-call clinicians (primary and backup) plus a fallback mailbox; names are an owner action before go-live. |
| [D-14](#d-14) | Which ENVIRONMENT value does the pilot run with, and is staging strict? | ops | W0 | Decided (default) | Accepted as the default. |
| [D-15](#d-15) | Should the bundled Postgres use TLS on the single host? | security | W0 | Decided (default) | Accepted as the default. TLS with verify-full. |
| [D-16](#d-16) | What is the dependency baseline? | ops | W0 | Decided (default) | Accepted as the default. |
| [D-17](#d-17) | Commit policy: when does the owner commit the interdependent working tree? | ops | W0 | Decided (changed from default) | No commits at all during the program, by agents or the owner. The working tree and commit-free checkpoint trees are the record; clean-checkout checks build from git... |
| [D-18](#d-18) | Is analytics-service part of the pilot? | product | W2 | Decided (default) | Accepted as the default. Analytics stays as a staff-only dashboard. |
| [D-19](#d-19) | What happens to out-of-scope components that are deployed or imported today? | product | W0 | Decided (default) | Accepted as the default. |
| [D-20](#d-20) | May pinned packages be installed into the local Python 3.12 environment? | ops | W0 | Decided (changed from default) | A dedicated venv at D:\tmp\mvp-exec\venv instead of installing into anaconda. Created at W0 entry from the anaconda 3.12.3 interpreter with --system-site-packages... |
| [D-21](#d-21) | How is clinician access to patient data scoped? | clinical | W3 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-22](#d-22) | What is the ingress screening architecture and its fail-safe? | clinical | W1 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-23](#d-23) | Which consents are required, and how is revocation enforced? | legal | W3 | Default accepted, sign-off pending | Build the default; legal reviews before go-live. |
| [D-24](#d-24) | What erasure SLA and retention apply to PHI, audit rows, event payloads, crisis records and backups? | legal | W3 | Default accepted, sign-off pending | Build the default; legal reviews before go-live. |
| [D-25](#d-25) | Where does the release gate run, and which host runs the pilot? | ops | W5 | Decided (changed from default) | AWS (account 253025918446, us-east-1) hosts the release gate and the pilot, using the existing IAM user MedCodex-Local for now. The 2026-10-19 IAM simulation shows... |
| [D-26](#d-26) | Edge rate limiting: a custom Caddy build or app-level limits only, and what happens when the limiter backend fails? | security | W1 | Decided (default) | Accepted as the default. |
| [D-27](#d-27) | How does a WebSocket carry the token, and how does re-authentication work? | security | W1 | Decided (default) | Accepted as the default. |
| [D-28](#d-28) | Should there be one clinician-alert path, and which component is the alert system of record? | clinical | W1 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-29](#d-29) | Which crisis resources are shown, and for which locale? | clinical | W1 | Decided (changed from default) | Several countries, US first: crisis resources become a per-locale versioned registry with only en-US populated (988, 741741, 911); the user's locale reaches the... |
| [D-30](#d-30) | May diagnostic hypotheses or labels appear in user-facing replies during the pilot? | clinical | W2 | Default accepted, sign-off pending | Build on the default now; a named clinician signs off before go-live. |
| [D-31](#d-31) | What is the go-live gate for real users and real PHI? | legal | W5 | Default accepted, sign-off pending | Build the default; legal reviews before go-live. |

Counts: Decided (default): 16; Decided (changed from default): 6; Default accepted, sign-off pending: 9.

## Decisions

### D-01

**Question.** What is the schema of record: the Alembic chain or the column names the repositories were written against?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W0
- **Decision:** Accepted as the default.
- **Default (as built):** Alembic (001 to head plus slots 009-021) is the only schema of record, and repository SQL changes to match it. 001 and 007 are repaired in place in W0 because no database has ever applied them. No service runs DDL at runtime: outbox, DLQ, audit and LangGraph checkpoint DDL move into slots, and the dead orchestrator_checkpoints DDL in state.py is deleted.
- **Why:** Migrations are the only artifact that can be versioned, reviewed and applied by a migrate job, and they already carry the RLS, width and constraint decisions. Repository SQL disagrees with them in five services (MVP-034/060/061/080/096/128/220), so one side must yield, and Alembic is the reviewable side.
- **Affects workstreams:** WS-02, WS-06, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18
- **Tasks gated on it:** [WS-02.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-02.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-06.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-10.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-02

**Question.** Which database topology and roles should the pilot use?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W0
- **Decision:** Accepted as the default.
- **Default (as built):** One PostgreSQL 16 container with one database 'solace' (schema public). solace_migrator owns all tables and is used only by the migrate job; solace_app (NOSUPERUSER NOBYPASSRLS, DML only) is used by every service from Wave 0; the image superuser 'postgres' is ops-only (init, physical backup and restore, test assertions). The nine unused per-service databases in init-db are dropped; per-service roles are post-MVP.
- **Why:** Superuser connections bypass every RLS policy (MVP-041). Moving to the real role in W0 means every later persistence task is proven under RLS instead of being flipped late, and physical backups avoid the pg_dump-versus-FORCE-RLS conflict.
- **Affects workstreams:** WS-02, WS-03, WS-05, WS-06, WS-14, WS-15, WS-16, WS-17, WS-18
- **Tasks gated on it:** [WS-02.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-02.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-02.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-02.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-03.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-07.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-11.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-14.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-03

**Question.** Which inter-service authentication model should services use?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W1
- **Decision:** Accepted as the default.
- **Default (as built):** Short-lived service JWTs from ServiceTokenManager, carrying metadata.obo_sub for user-scoped calls; the shared get_current_user accepts them only from services granted service:act_for_user (orchestrator only); a service handling an obo call forwards the inbound header for same-user downstream calls; service-only endpoints enforce require_service_permission with receiver-side permission resolution; user JWTs are never forwarded or stored in graph state. HS256 with AUTH_SECRET_KEY for service tokens in W1-W3; from W4, per-service RS256 service keys (kid = service name) and RS256-only user tokens.
- **Why:** Today no single credential works across services (MVP-021). Forwarding user JWTs would put credentials into checkpointed LangGraph state and break on 15-minute expiry in long WS sessions. obo tokens leave receivers' ownership checks unchanged, and per-service keys stop a compromised container from minting another service's identity.
- **Affects workstreams:** WS-07, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18, WS-03
- **Tasks gated on it:** [WS-03.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-07.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-07.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-07.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-10.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-15.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-04

**Question.** Which LLM and embedding vendors may receive real-user PHI, which BAAs are in place, and are Portkey semantic caching and memory's OpenAI embeddings acceptable?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** legal; needed by W1
- **Decision:** Anthropic models on AWS Bedrock under the AWS BAA are the only route for real-user PHI. Strict environments require LLM_PROVIDER=bedrock; Portkey stays for development and test with synthetic data only. LLM features stay off for real users until the AWS BAA is accepted in AWS Artifact and recorded. Caching off, no semantic cache, memory's OpenAI embeddings disabled and Weaviate writes off, as in the default. Adds one W1 task in WS-09 (Bedrock provider).
- **Original recommendation (replaced):** Portkey gateway to Anthropic only after BAAs covering both are confirmed (Anthropic via AWS Bedrock under the AWS BAA is the alternative). LLM_ENABLE_CACHING=false for every request and never a semantic cache for PHI prompts. memory-service's direct OpenAI embedding calls disabled and Weaviate writes off (WEAVIATE_ENABLED=false); retrieval stays keyword/BM25. Prompts limited to minimum-necessary content. In strict envs LLM_PHI_EGRESS_APPROVED=true is required to enable a real provider; without BAAs by the release gate the pilot runs template replies only. Tests always use the fake LLM.
- **Why:** PHI currently flows to a cache shared across users (MVP-056) and to OpenAI directly (MVP-145). Both are legal exposures that code alone cannot fix, and the crisis path must already work with the LLM unavailable.
- **Affects workstreams:** WS-09, WS-10, WS-12, WS-15, WS-16, WS-17, WS-18, WS-05
- **Tasks gated on it:** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-05.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-09.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-10.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-15.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-17.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)
- **Plan impact (2026-10-19 change):** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-09.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-10.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-15.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.22](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-17.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-04".
- **Owner actions:** [X-02](OPEN-ITEMS.md#owner-actions), [X-08](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): Anthropic models on AWS Bedrock under the AWS BAA are the only route for real-user PHI. Strict environments require LLM_PROVIDER=bedrock; Portkey stays for development and test with synthetic data onl | owner |

### D-05

**Question.** Clinical sign-off: who approves the crisis lexicon, the CRITICAL threshold, the LLM/fallback fusion rule, negation policy and auto-escalation rules?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W1
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** Ship lexicon v1 in solace_clinical, built from safety's clinical_lexicon.json plus every register gap (suicidal, ending my life, jump off, took all my pills, homicidal, abuse, curly apostrophes), with a positive/negative phrase set of 200 or more. CRITICAL floor 0.85. Max-only fusion: LLM, fallback and stub scores may raise, never lower, a lexicon level and are never averaged in. No HIGH on context-free single words without self-reference. W1 proceeds on this default with the owner's acceptance recorded as a ledger Ruling; a named licensed clinician signs docs/CLINICAL-VALIDATION.md before W5 (required for W5 scoring changes and the release gate).
- **Why:** Detection is the top patient-safety risk (MVP-017, 023-026, 030, 084, 086, 182). Engineering can make detection conservative now, but clinical accountability needs a named approver and one reviewed artifact used by every detector.
- **Affects workstreams:** WS-12, WS-10, WS-15, WS-16, WS-17, WS-05
- **Tasks gated on it:** [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.22](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-10.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-12.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-15.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-06

**Question.** Crisis-mode semantics: does crisis state persist per thread, how does it clear, and how are repeated alerts deduplicated?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W1
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** Safety flags are per turn. After HIGH/CRITICAL the thread enters a 'crisis watch' for the next 3 user turns or 30 minutes: resources appended, the safety-service check mandatory, the ELEVATED threshold lowered; no permanent lock. Pages coalesce per user into one episode for 15 minutes unless the level rises; after acknowledgement, further detections in the window send one update to the acknowledging clinician. ELEVATED shows resources and pages no one; RISK_ALERT monitoring emails are coalesced per user per window.
- **Why:** Today's OR-merge reducer locks a thread into the crisis reply forever after one false positive and re-pages every turn (MVP-019, 181, 299). A bounded watch keeps the protective intent without that failure.
- **Affects workstreams:** WS-10, WS-12, WS-13
- **Tasks gated on it:** [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-10.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-12.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-15.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-07

**Question.** Which secrets manager holds per-environment secrets?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W3
- **Decision:** SOPS + age instead of AWS Secrets Manager: per-environment encrypted deploy/secrets/<env>.sops.yaml, one age key pair per environment with the private keys held off-host, decrypted at deploy by deploy/render_secrets.py into root-only tmpfs /run/secrets (NAME_FILE contract unchanged). The plaintext .env.prod and Caddy env_file are retired and the working-tree .env keys rotated.
- **Original recommendation (replaced):** AWS Secrets Manager, one secret set per environment with separate KMS keys, rendered at deploy time by deploy/render_secrets.py into root-only tmpfs /run/secrets files consumed through NAME_FILE. For a non-AWS host: SOPS-encrypted per-environment files with age keys held off-host. The plaintext .env.prod and Caddy's env_file are retired, and the keys in the working-tree .env are rotated.
- **Why:** Every secret is hand-generated into one plaintext file injected into all 11 containers, Caddy included (MVP-348/349). Per-environment separation, least exposure and rotation need a store.
- **Affects workstreams:** WS-03, WS-04, WS-01, WS-05, WS-07, WS-09
- **Tasks gated on it:** [WS-01.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-09.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md)
- **Plan impact (2026-10-19 change):** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-09.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-07".
- **Owner actions:** [X-07](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): SOPS + age instead of AWS Secrets Manager: per-environment encrypted deploy/secrets/<env>.sops.yaml, one age key pair per environment with the private keys held off-host, decrypted at deploy by deploy | owner |

### D-08

**Question.** Which MFA method, and who must use it?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W4
- **Decision:** Accepted as the default.
- **Default (as built):** TOTP (RFC 6238, 30 s, 6 digits, +-1 step) with 10 one-time recovery codes; mandatory for CLINICIAN and ADMIN (enrollment forced at first login; no PHI, admin or acknowledgement API without mfa=true); optional for end users; TOTP secrets encrypted with the PHI codec. The one-time acknowledgement link in alert emails stays usable without login. WebAuthn and SMS OTP are post-MVP.
- **Why:** Clinicians and admins reach PHI across users and are password-only today (MVP-344). TOTP needs no new vendor or BAA, and pyotp is already available; the amr/mfa claims are designed in W2 so W4 is additive.
- **Affects workstreams:** WS-08, WS-07, WS-13, WS-14, WS-05
- **Tasks gated on it:** [WS-05.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-13.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-09

**Question.** Which backup tooling and RPO/RTO targets apply?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W4
- **Decision:** Accepted as the default. RPO 15 min, RTO 4 h.
- **Default (as built):** Postgres: wal-g (or pgBackRest) physical base backup nightly plus continuous WAL archiving to an encrypted off-host target (S3 with SSE-KMS, versioning and a 35-day object lock on AWS; an encrypted external volume on a local host); RPO 15 min, RTO 4 h. Redis and Kafka are not backed up (ephemeral by design; the Postgres outbox is the durable event source). Weaviate is not backed up while writes are off (D-04). Keys are escrowed in the secrets store. An automated restore drill runs in W4, before release and then monthly. Fallback if wal-g is too heavy: 6-hourly physical snapshots (RPO 6 h).
- **Why:** No backup exists for any PHI store and all data sits on one deletable volume (MVP-037/043/350). A backup counts only once a restore has been tested, and physical backups avoid the FORCE RLS restriction on pg_dump.
- **Affects workstreams:** WS-03, WS-04, WS-05
- **Tasks gated on it:** [WS-03.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md)
- **Plan impact (2026-10-19 change):** [WS-03.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-09".

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. RPO 15 min, RTO 4 h. | owner |

### D-10

**Question.** Where is the audit trail stored, and how long is it retained?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W3
- **Decision:** Accepted as the default.
- **Default (as built):** Postgres audit_events in the same database (slot 011); solace_app has INSERT and SELECT only with append-only rules; one global HMAC chain serialized by advisory lock; AUDIT_HMAC_KEY in the secrets store; rows hold ids and action codes only, so they survive erasure; retained 6 years and included in backups; a nightly verify_chain job alerts on mismatch.
- **Why:** The audit logger is never called (MVP-091), the chain cannot detect tampering (MVP-345) and per-process tips fork it (MVP-346). One serialized, backed-up store that recomputes HMACs fixes all three.
- **Affects workstreams:** WS-09, WS-04, WS-08, WS-13, WS-14
- **Tasks gated on it:** [WS-04.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-04.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-08.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-09.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-09.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-13.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-11

**Question.** Client contract: does the code or docs/API-HANDOFF.md define the REST/WS protocol and the OAuth handoff?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** product; needed by W1
- **Decision:** Accepted as the default.
- **Default (as built):** The code becomes the contract and API-HANDOFF is rewritten from it (IF-12). WS frames use the documented names (chat.message/content in; safety.alert, chat.response, pong, error out) and accept the legacy {type:'message', message} as an alias; unknown frames get an error frame instead of a silent drop. Errors on the edge services (orchestrator, user-service) use {error:{code,message,request_id}} from W2; internal services keep FastAPI's {detail} shape. OAuth returns by 302 with a one-time code exchanged by POST. The pilot client targets this API, not Supabase.
- **Why:** Clients built from the doc today have their frames dropped silently (MVP-072) and OAuth cannot hand tokens to a client (MVP-104). No client exists yet, so adopting the documented names costs nothing and the crisis E2E in W1 needs the final frames.
- **Affects workstreams:** WS-11, WS-08, WS-03, WS-05
- **Tasks gated on it:** [WS-04.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-04.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-08.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-11.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-15.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-12

**Question.** Which notification providers deliver clinician alerts, and what may an alert contain?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W1
- **Decision:** Accepted as the default. Amazon SES email.
- **Default (as built):** Email through an SMTP relay covered by a BAA (e.g. Amazon SES in the pilot region) with STARTTLS (EMAIL_SMTP_TLS_MODE=starttls); SMS through Twilio with a BAA as an optional second channel for CRITICAL; push out of scope. Alerts carry level, categories, detection sources, time, alert id and the acknowledgement link, plus the user's display name and contact only when the user gave crisis-contact consent (D-23); never message text or diagnosis. The fallback mailbox is a monitored shared clinical inbox on a different provider.
- **Why:** Alerts cannot reach anyone today (MVP-011/064), an alert without the user's identity is not actionable, and the provider and content rules decide what PHI leaves the system.
- **Affects workstreams:** WS-13, WS-03, WS-12, WS-14
- **Tasks gated on it:** [WS-03.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-13.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md)
- **Plan impact (2026-10-19 change):** [WS-03.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-12".
- **Owner actions:** [X-02](OPEN-ITEMS.md#owner-actions), [X-03](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. Amazon SES email. | owner |

### D-13

**Question.** How are clinician and admin accounts provisioned and placed on call?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** product; needed by W1
- **Decision:** Accepted as the default. At least two on-call clinicians (primary and backup) plus a fallback mailbox; names are an owner action before go-live.
- **Default (as built):** Only an operator creates them, with scripts/provision_clinician.py run in the user-service container: a verified ACTIVE account with an E.164 phone, on-call/backup flags and a one-time set-password link. ADMIN toggles on-call status and assignments through the admin endpoints (IF-08). No self-registration for privileged roles. Before go-live at least two on-call clinicians and the fallback mailbox are configured. MFA becomes mandatory once W4 lands.
- **Why:** No supported path creates a clinician or stores a phone number, so the on-call roster is empty (MVP-105).
- **Affects workstreams:** WS-14, WS-13, WS-05
- **Tasks gated on it:** [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-13.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md)
- **Owner actions:** [X-05](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. At least two on-call clinicians (primary and backup) plus a fallback mailbox; names are an owner action before go-live. | owner |

### D-14

**Question.** Which ENVIRONMENT value does the pilot run with, and is staging strict?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W0
- **Decision:** Accepted as the default.
- **Default (as built):** The pilot runs ENVIRONMENT=production; a staging rehearsal uses ENVIRONMENT=staging with identical strictness; every guard keyed on 'production' only switches to is_strict_environment().
- **Why:** The documented deploy runs staging, which silently skips the production-only guards: in-memory outbox, DLQ, escalations and the verification gate (MVP-028/052/088/213).
- **Affects workstreams:** WS-04, WS-03, WS-06, WS-08, WS-12, WS-13, WS-15
- **Tasks gated on it:** [WS-02.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-02.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-03.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-04.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-06.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-07.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-11.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-18.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-15

**Question.** Should the bundled Postgres use TLS on the single host?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W0
- **Decision:** Accepted as the default. TLS with verify-full.
- **Default (as built):** Yes: TLS on, with a CA and server certificate generated at deploy time (scripts/db/gen_certs.py) and stored in the secrets store; clients use POSTGRES_SSL_MODE=verify-full against host 'postgres' with POSTGRES_SSL_ROOT_CERT; the dev profile may use 'prefer'.
- **Why:** Strict envs already force sslmode=require, which the TLS-less bundled Postgres rejects (MVP-010), and 'require' skips certificate verification (MVP-271). verify-full satisfies the guards and protects PHI in transit.
- **Affects workstreams:** WS-02, WS-03, WS-05
- **Tasks gated on it:** [WS-02.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-03.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-12.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. TLS with verify-full. | owner |

### D-16

**Question.** What is the dependency baseline?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W0
- **Decision:** Accepted as the default.
- **Default (as built):** One uv lock for Python 3.12 pinned to the versions the 5,377 local tests pass on (langgraph 1.0.4, langgraph-checkpoint 3.0.1, langchain-core 1.2.28, pydantic 2.13.x, pydantic-settings 2.12.0, fastapi 0.128.0, asyncpg 0.30.0, redis 5.3.x, weaviate-client 4.19.x), plus security bumps (PyJWT >= 2.13, python-jose removed) and the missing packages (email-validator, aiokafka, langgraph-checkpoint-postgres with psycopg, pyotp). No torch, transformers or VADER in images (lexicon-only safety). Base images pinned by digest.
- **Why:** Images would ship langgraph 0.2.x with a checkpoint CVE while tests run on 1.x (MVP-358); PyJWT pins sit in 2026 advisory ranges (MVP-359); the current pins cannot resolve (MVP-005).
- **Affects workstreams:** WS-01, WS-10, WS-11, WS-07
- **Tasks gated on it:** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-06.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-12.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |
| 2026-11-02 | As built (security bumps within D-16; controller Rulings WS-01.F1 and WS-01.F2): pip-audit found 92 advisories in the tested pins. F1 fixed 60 with same-major bumps and accepted 32 with reasons, each expiring by 2027-01-31 (`scripts/ci/accepted_vulns.yaml`). | controller |
| 2026-11-03 | As built (WS-01.F2), fixing the checkpoint-path advisories, including deserialization/RCE. The full regression stayed at baseline. New pins:<br>- langgraph 1.0.10<br>- langgraph-checkpoint 4.1.1 (major)<br>- langgraph-checkpoint-postgres 3.1.1<br>- langchain-core 1.3.3<br>- pydantic-settings 2.14.2<br>- fastapi 0.133.0<br>- starlette 1.3.1 (major)<br>- prometheus-fastapi-instrumentator 8.0.0<br><br>9 runtime advisories remain accepted, with reasons:<br>- aiosmtplib (WS-13.7)<br>- cryptography (WS-01.10)<br>- langchain-anthropic<br>- the checkpoint-postgres PostgresStore issue (unused)<br>- langgraph-sdk | controller |

### D-17

**Question.** Commit policy: when does the owner commit the interdependent working tree?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W0
- **Decision:** No commits at all during the program, by agents or the owner. The working tree and commit-free checkpoint trees are the record; clean-checkout checks build from git archive of a checkpoint tree; each barrier tree is also exported as a zip outside the repo. The owner decides about committing after the program.
- **Original recommendation (replaced):** The owner commits the whole working tree, untracked files included, as one unit at the W0 barrier and at each later barrier. Until then, clean-checkout consistency is proven by building images from a git archive of the checkpoint tree.
- **Why:** HEAD does not work without the uncommitted tree (MVP-008), and checkpoint trees are the only safety net against an accidental clean.
- **Affects workstreams:** WS-01, WS-05
- **Tasks gated on it:** [WS-01.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md)
- **Plan impact (2026-10-19 change):** [WS-01.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-17".
- **Owner actions:** [X-10](OPEN-ITEMS.md#owner-actions), [X-11](OPEN-ITEMS.md#owner-actions), [X-12](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): No commits at all during the program, by agents or the owner. The working tree and commit-free checkpoint trees are the record; clean-checkout checks build from git archive of a checkpoint tree; each  | owner |

### D-18

**Question.** Is analytics-service part of the pilot?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** product; needed by W2
- **Decision:** Accepted as the default. Analytics stays as a staff-only dashboard.
- **Default (as built):** Keep it as a non-critical, staff-only operator dashboard: fix ingestion, PHI logging and role checks; no clinical workflow depends on it; ClickHouse stays out of scope and counters hold no PHI.
- **Why:** Its issues are in scope and its outputs help monitor the pilot, but it must not hold up the critical path.
- **Affects workstreams:** WS-18, WS-04
- **Tasks gated on it:** [WS-04.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-18.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. Analytics stays as a staff-only dashboard. | owner |

### D-19

**Question.** What happens to out-of-scope components that are deployed or imported today?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** product; needed by W0
- **Decision:** Accepted as the default.
- **Default (as built):** config-service is removed from the MVP compose, prod overlay and CI image matrix and its code frozen; feature flags stay in solace_infrastructure.feature_flags. The Kong package stays unimported behind a lazy __init__. ClickHouse, the memory RAG pipeline and the safety contraindication engine stay unwired, and code or docs that claim they run are corrected.
- **Why:** config-service has zero consumers and is permanently unhealthy (MVP-231); eager imports of dead packages break boot (MVP-269, 310); dead engines mislead reviewers (MVP-174).
- **Affects workstreams:** WS-03, WS-01, WS-12, WS-17, WS-18
- **Tasks gated on it:** [WS-01.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-02.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-02-data-platform-schema.md), [WS-03.1](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-12.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-20

**Question.** May pinned packages be installed into the local Python 3.12 environment?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W0
- **Decision:** A dedicated venv at D:\tmp\mvp-exec\venv instead of installing into anaconda. Created at W0 entry from the anaconda 3.12.3 interpreter with --system-site-packages (anaconda is read, never modified); the missing pins (aiokafka, langgraph-checkpoint-postgres with psycopg, PyJWT >= 2.13) go into the venv only, with pip freeze before and after; per-suite baseline re-recorded. After WS-01's lock lands the venv is re-synced to it, and the W0 barrier also runs the suites in a clean venv built only from the lock.
- **Original recommendation (replaced):** Yes, once at W0 entry: install exactly the lock versions of the packages the suites need but the anaconda env lacks or has at vulnerable versions (aiokafka, langgraph-checkpoint-postgres, PyJWT upgraded to >= 2.13, plus any other lock-pinned test extra), recording 'pip freeze' before and after. Later additions need owner approval. Fallback if refused: Kafka-client and checkpointer tests run inside a test-runner container built from requirements-dev.lock.
- **Why:** aiokafka and langgraph-checkpoint-postgres are not installed, and real-infra event-plane and checkpointer tests need them; installing packages is a stop condition in the execution protocol.
- **Affects workstreams:** WS-01, WS-05, WS-06, WS-10
- **Tasks gated on it:** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-05.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-06.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md)
- **Plan impact (2026-10-19 change):** [WS-01.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-01.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-05.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-06.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-20".

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): A dedicated venv at D:\tmp\mvp-exec\venv instead of installing into anaconda. Created at W0 entry from the anaconda 3.12.3 interpreter with --system-site-packages (anaconda is read, never modified); t | owner |

### D-21

**Question.** How is clinician access to patient data scoped?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W3
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** A clinician reads or writes a user's PHI only when assigned to that user (clinician_patient_assignments) or when acknowledging an open crisis episode for that user within 72 h. Admins have no PHI read endpoints. Cross-user DB work sets the RLS user to the data owner only after can_access_subject passes. No BYPASSRLS role for services.
- **Why:** Any clinician can reach every user's data today (MVP-120), and the RLS context uses the caller instead of the data owner (MVP-141).
- **Affects workstreams:** WS-07, WS-15, WS-16, WS-17, WS-14
- **Tasks gated on it:** [WS-07.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-14.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-16.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-17.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-22

**Question.** What is the ingress screening architecture and its fail-safe?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W1
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** The local shared-lexicon screen runs first on every message; the safety-service /check runs on every message with a 2 s timeout and can only raise the level (fuse_risk); if /check fails or times out the local result stands and the turn is marked degraded; the LLM intent classifier always runs for non-crisis messages when available and an LLM crisis_disclosure routes to the crisis path; in-band resources never depend on the LLM, downstream services, Kafka or the database.
- **Why:** The safety-service precheck is never enabled (MVP-017), LLM classification is skipped or never routed (MVP-023/024), and a graph exception drops crisis resources (MVP-079).
- **Affects workstreams:** WS-10, WS-12, WS-11
- **Tasks gated on it:** [WS-05.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-10.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-23

**Question.** Which consents are required, and how is revocation enforced?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** legal; needed by W3
- **Decision:** Build the default; legal reviews before go-live.
- **Default (as built):** Terms of service, privacy notice, and an AI-companion plus crisis-handling disclosure (versioned) must be accepted before the first chat; crisis-contact consent (a clinician may contact the user in a crisis, and the alert may carry their name and contact) is requested at signup and recorded separately. All are stored in consent_records with the document version. Access tokens carry the 'cns' claim and the orchestrator returns 403 CONSENT_REQUIRED without it; revoking a required consent ends sessions and blocks chat without deleting the account. Research and marketing consents are optional.
- **Why:** Consent is recorded but never enforced (MVP-100), and contacting a user in crisis needs a documented basis.
- **Affects workstreams:** WS-14, WS-08, WS-11, WS-13
- **Tasks gated on it:** [WS-05.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-08.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-11.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-13.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md)
- **Owner actions:** [X-06](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build the default; legal reviews before go-live. | owner |

### D-24

**Question.** What erasure SLA and retention apply to PHI, audit rows, event payloads, crisis records and backups?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** legal; needed by W3
- **Decision:** Build the default; legal reviews before go-live.
- **Default (as built):** Deletion is immediate and irreversible after re-authentication and confirmation. PHI is hard-deleted everywhere within 15 minutes (target) and at most 24 h: rows, checkpoints, Redis, Weaviate, caches, and outbox/DLQ rows for the user. Crisis episodes and escalation records are pseudonymized (user reference replaced by a salted hash, contact data removed) rather than deleted, for duty-of-care documentation. Audit rows (no PHI) and consent records are kept for the legal retention period. Outbox and DLQ payloads are purged 7 and 30 days after publish or resolution; Kafka solace.safety and solace.users retain 7 days; backups age out within 35 days, documented as the erasure lag.
- **Why:** Erasure never fans out (MVP-029), event tables keep plaintext PHI forever (MVP-054), and crisis-response records may need to outlive the account; retention needs a legal basis.
- **Affects workstreams:** WS-14, WS-06, WS-03, WS-09, WS-12, WS-13
- **Tasks gated on it:** [WS-01.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-05.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-06.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-06.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-06.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-06-event-plane.md), [WS-08.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-09.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-09-phi-codec-audit-llm.md), [WS-10.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.18](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-14.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.14](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-17.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-17-memory-service.md), [WS-18.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md), [WS-18.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)
- **Owner actions:** [X-06](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build the default; legal reviews before go-live. | owner |

### D-25

**Question.** Where does the release gate run, and which host runs the pilot?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** ops; needed by W5
- **Decision:** AWS (account 253025918446, us-east-1) hosts the release gate and the pilot, using the existing IAM user MedCodex-Local for now. The 2026-10-19 IAM simulation shows only Bedrock allowed; CloudFormation, EC2, SSM, S3, KMS and SES are denied, so the deployer policy must be attached to MedCodex-Local before W4's off-host backups, W5's release gate and go-live. Until then real-infra tests run on local Docker; an unattached policy at W5 entry is a stop.
- **Original recommendation (replaced):** The release gate runs on a local Docker host with the strict production profile. AWS staging is provisioned only after the owner grants the deployer IAM policy. Go-live needs the AWS host (or an owner-approved alternative) with D-09 off-host backups.
- **Why:** AWS provisioning is blocked on IAM permissions and the CFN template has never been deployed (MVP-043).
- **Affects workstreams:** WS-03, WS-05
- **Tasks gated on it:** [WS-01.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-05.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md)
- **Plan impact (2026-10-19 change):** [WS-01.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-03.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-05.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-25".
- **Owner actions:** [X-01](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): AWS (account 253025918446, us-east-1) hosts the release gate and the pilot, using the existing IAM user MedCodex-Local for now. The 2026-10-19 IAM simulation shows only Bedrock allowed; CloudFormation | owner |

### D-26

**Question.** Edge rate limiting: a custom Caddy build or app-level limits only, and what happens when the limiter backend fails?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W1
- **Decision:** Accepted as the default.
- **Default (as built):** The app-level Redis limiter (IF-04), keyed on user id or the real client IP with TRUSTED_PROXY_HOPS=1, is the control. On Redis failure it fails closed (503) for auth routes and open for chat, with a metric and alert. Caddy gets request-size and timeout limits but no third-party module, and the misleading rate-limit comment is removed.
- **Why:** Stock Caddy has no rate_limit module (MVP-115), proxy-IP keying lets about 21 bad logins a minute lock out everyone (MVP-101), and a Redis outage currently returns 500 on every login (MVP-185).
- **Affects workstreams:** WS-07, WS-03, WS-11, WS-14
- **Tasks gated on it:** [WS-03.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-03.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-04.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-04-service-runtime-observability.md), [WS-05.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-07.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-11.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-14.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-27

**Question.** How does a WebSocket carry the token, and how does re-authentication work?

- **Status:** Decided (default) (answered 2026-10-19 by the owner)
- **Owner role:** security; needed by W1
- **Decision:** Accepted as the default.
- **Default (as built):** Prefer the Sec-WebSocket-Protocol pair ('solace.v1', 'bearer.<jwt>'); keep '?token=' for compatibility with uvicorn and Caddy access logs disabled; check exp and revocation on every frame; accept {type:'auth', token} re-auth frames; close with 4401 on expiry, revocation or account deletion.
- **Why:** The WS validates the token only at the handshake (MVP-163) and query tokens land in logs (MVP-211).
- **Affects workstreams:** WS-11, WS-01, WS-03
- **Tasks gated on it:** [WS-01.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-01-build-deps-ci.md), [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-11.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (default): Accepted as the default. | owner |

### D-28

**Question.** Should there be one clinician-alert path, and which component is the alert system of record?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W1
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** notification-service is the only component that pages clinicians and the system of record for crisis episodes (coalescing, acknowledgement, re-paging). safety-service records its escalation decisions durably and publishes them only through its transactional outbox; it stops calling /notifications/send inline. The orchestrator, safety, diagnosis and therapy publish CrisisDetected with the turn correlation id, and notification coalesces them into one episode.
- **Why:** Inline escalation can never deliver, blocks /check for up to 300 s (MVP-085/087) and triples pages (MVP-181/299); acknowledgement belongs where pages are sent, and keeping it there avoids a race between the first page and the escalation record (MVP-082).
- **Affects workstreams:** WS-12, WS-13, WS-10, WS-15, WS-16
- **Tasks gated on it:** [WS-03.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-03-deploy-secrets-backups.md), [WS-05.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-10.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-12.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-13.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-13.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-13-notification-alert-delivery.md), [WS-15.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-29

**Question.** Which crisis resources are shown, and for which locale?

- **Status:** Decided (changed from default) (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W1
- **Decision:** Several countries, US first: crisis resources become a per-locale versioned registry with only en-US populated (988, 741741, 911); the user's locale reaches the screen without a DB or network call; an unsupported locale gets a safe generic fallback. Each further country is added with clinician-verified resources before its users join. Resources still need the clinician's sign-off.
- **Original recommendation (replaced):** US pilot (en-US): 988 Suicide and Crisis Lifeline (call or text 988), Crisis Text Line (text HOME to 741741), 911 for immediate danger. Versioned in solace_clinical and returned identically by the orchestrator and safety-service. Non-US users are outside the pilot.
- **Why:** Resources are hard-coded in several places today; a single versioned source is needed for clinical sign-off.
- **Affects workstreams:** WS-12, WS-10
- **Tasks gated on it:** [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-07.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-07-service-auth-authz.md), [WS-08.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-10.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-14.22](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.23](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md)
- **Plan impact (2026-10-19 change):** [WS-05.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-08.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.13](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.16](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.17](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-08.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-08-identity-auth-mfa.md), [WS-10.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-10.8](../superpowers/plans/2026-10-18-mvp-remediation/WS-10-orchestrator-graph-crisis.md), [WS-11.4](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.5](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.20](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.21](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-12.2](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.3](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-12.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-12-safety-detection-lexicon.md), [WS-14.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-14.22](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md), [WS-15.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.7](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-15.23](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md). Each plan's *Review responses* lists the exact change under "decision 2026-10-19 D-29".
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions), [X-09](OPEN-ITEMS.md#owner-actions)

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Decided (changed from default): Several countries, US first: crisis resources become a per-locale versioned registry with only en-US populated (988, 741741, 911); the user's locale reaches the screen without a DB or network call; an | owner |

### D-30

**Question.** May diagnostic hypotheses or labels appear in user-facing replies during the pilot?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** clinical; needed by W2
- **Decision:** Build on the default now; a named clinician signs off before go-live.
- **Default (as built):** No. Diagnosis contributes symptom and risk signals to the orchestrator only; therapy never asserts a diagnosis the user did not state (no default 'Depression' plan); personality style adaptation is allowed.
- **Why:** The product is screening-grade, not a diagnostic device, and diagnosis and therapy outputs are clinically unreliable today (MVP-048, 125, 202).
- **Affects workstreams:** WS-16, WS-15, WS-11
- **Tasks gated on it:** [WS-05.11](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-11.9](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-11.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-11-orchestrator-edge-contracts.md), [WS-15.12](../superpowers/plans/2026-10-18-mvp-remediation/WS-15-therapy-service.md), [WS-16.6](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-16.10](../superpowers/plans/2026-10-18-mvp-remediation/WS-16-diagnosis-service.md), [WS-18.15](../superpowers/plans/2026-10-18-mvp-remediation/WS-18-personality-analytics.md)
- **Owner actions:** [X-04](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build on the default now; a named clinician signs off before go-live. | owner |

### D-31

**Question.** What is the go-live gate for real users and real PHI?

- **Status:** Default accepted, sign-off pending (answered 2026-10-19 by the owner)
- **Owner role:** legal; needed by W5
- **Decision:** Build the default; legal reviews before go-live.
- **Default (as built):** Only synthetic data until the W5 release gate passes and, in addition, the D-04 BAAs, the D-05 clinical sign-off, D-13 on-call coverage and D-09 restore-drill evidence are recorded. The first real users are a named pilot cohort.
- **Why:** Every PHI and crisis control was nominal until this program; real users must not arrive before those controls are proven.
- **Affects workstreams:** WS-05, WS-12, WS-03, WS-04
- **Tasks gated on it:** [WS-05.19](../superpowers/plans/2026-10-18-mvp-remediation/WS-05-realinfra-harness-e2e.md), [WS-14.22](../superpowers/plans/2026-10-18-mvp-remediation/WS-14-accounts-roster-consent-erasure.md)
- **Owner actions:** [X-06](OPEN-ITEMS.md#owner-actions)
- **Sign-off:** approver: _(name)_; date: _(date)_; evidence: _(link)_

| Date | Change | By |
|---|---|---|
| 2026-10-18 | Recommended default written in the plan | planning |
| 2026-10-19 | Default accepted, sign-off pending: Build the default; legal reviews before go-live. | owner |

