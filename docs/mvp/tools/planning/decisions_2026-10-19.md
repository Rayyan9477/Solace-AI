# Owner decisions, 2026-10-19: what changed and the binding contracts

Repo: D:\Repo\Contextual-Chatbot. The plans are in docs/superpowers/plans/2026-10-18-mvp-remediation/WS-*.md. The decision texts (question, original default, rationale) are in D:\tmp\mvp-plan\skeleton.json under "decisions".

On 2026-10-19 the owner answered every decision D-01 to D-31.

- **Accepted as the plan's default:** D-01, D-02, D-03, D-08, D-09, D-10, D-11, D-12, D-13, D-14, D-15, D-16, D-18, D-19, D-26, D-27.
  - D-09 keeps RPO 15 min / RTO 4 h.
  - D-12 is Amazon SES email.
  - D-13 is two or more on-call clinicians plus a fallback mailbox.
  - D-18 keeps analytics as a staff-only service.
- **Default accepted, sign-off pending:**
  - Clinical (D-05, D-06, D-21, D-22, D-28, D-30): build on the defaults now; a named clinician signs off before go-live.
  - Legal (D-23, D-24, D-31): build the defaults; legal reviews before go-live.
- **Changed from the default:** D-04, D-07, D-17, D-20, D-25, D-29, below.
  - Every task text that assumes the old default must be updated.
  - New tasks get the next free number in their workstream.
  - Existing task ids stay stable.

## D-17 (changed): no commits at all

Nobody commits during the program, not even the owner at barriers.

- The working tree plus the commit-free checkpoint tree snapshots (D:\tmp\mvp-exec\tools\checkpoint.ps1) are the only record.
- Clean-checkout checks build from `git archive <checkpoint tree id>`.
- Committing is decided by the owner after the program.
- Remove every "the owner commits at the barrier" step or assumption.
- Barrier notes report the tree id only.

## D-20 (changed): a dedicated venv; anaconda stays untouched

- **Interpreter for every test and script command:** `D:\tmp\mvp-exec\venv\Scripts\python.exe`. Every plan command has already been switched to this path mechanically.
- **Created once at W0 entry,** by the controller (a protocol step, not a task):
  - Command: `& "C:\Users\rayyan.a\AppData\Local\anaconda3\python.exe" -m venv --system-site-packages D:\tmp\mvp-exec\venv`.
  - With system site packages, the venv sees anaconda's packages read-only, so the 5,377-test baseline reproduces unchanged.
  - Into the venv only, install exactly the missing or vulnerable packages at their D-16 pins: aiokafka, langgraph-checkpoint-postgres (with psycopg) and PyJWT >= 2.13, plus any other lock-pinned test extra the suites need.
  - Record `pip freeze` of the venv before and after. Re-record the baseline per suite in the venv; a drop needs a ledger Ruling.
  - Nothing is ever installed into anaconda.
- **After WS-01's lock exists,** the venv is re-synced to the lock's dev set, still into the venv.
- **The lock is proven complete in clean images.** W0's barrier also runs the suites once in a second, clean venv without system site packages (`D:\tmp\mvp-exec\venv-clean`), built only from the lock. Failures there are lock defects.
- Running anaconda's `uv.exe` binary is fine because it installs nothing into anaconda. Never run `pip install` against anaconda.

## D-04 (changed): real-user LLM traffic goes to Anthropic models on AWS Bedrock under the AWS BAA

- **Routing:**
  - Strict environments (staging, production) require `LLM_PROVIDER=bedrock`.
  - `LLM_PROVIDER=portkey` stays allowed only in development and test, with synthetic data.
  - LLM features stay off for real users until the AWS BAA is accepted (the owner accepts it in AWS Artifact) and recorded.
  - The rest of D-04 is unchanged: caching off, no semantic cache, memory's OpenAI embeddings disabled, Weaviate writes off, minimum-necessary prompts.
- **Settings (IF-02 names; WS-03 owns the env registry):**
  - `LLM_PROVIDER` in {bedrock, portkey}.
  - `BEDROCK_REGION`, default `us-east-1`.
  - `BEDROCK_MODEL_ID`: no default in strict environments. It is set at deploy after checking availability with `bedrock:ListFoundationModels` and inference profiles; never guess a model id in code or tests.
  - `BEDROCK_ENDPOINT_URL`: test and E2E only, pointing at the fake LLM; refused in strict environments.
  - `BEDROCK_TIMEOUT_SECONDS`, default `20`.
  - Credentials come from the host's instance role. No static AWS keys in env or secrets; local development may use `AWS_PROFILE`.
- **Code (WS-09):** a new task adds a Bedrock provider to `services/shared/infrastructure/llm_client.py` (the UnifiedLLMClient).
  - It calls the bedrock-runtime Converse API through boto3, wrapped with `asyncio.to_thread`, with an explicit botocore Config (connect/read timeouts, bounded retries).
  - It honors the settings above, never logs or caches prompt text, and maps errors to the client's existing error types.
  - Unit tests use `botocore.stub.Stubber` or a fake client; they never call AWS.
  - The executor loads the amazon-bedrock skill before writing it.
- **Dependency (WS-01):** `boto3` and `botocore` are pinned in the runtime lock, at the version installed in the anaconda env (check with `pip show boto3`), unless an advisory forces a bump.
- **Fake LLM (WS-05.1):** adds a Bedrock Converse route `POST /model/{model_id}/converse` that returns a Converse-shaped body with the same deterministic behavior as its existing route. It ignores SigV4. E2E stacks set `LLM_PROVIDER=bedrock` and `BEDROCK_ENDPOINT_URL=http://fake-llm:8080`.
- **IAM (WS-03):** the pilot host's instance role in infrastructure/aws-staging gets `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`, scoped to the chosen model or inference-profile ARNs (parameterized), and nothing broader.

## D-07 (changed): SOPS + age instead of AWS Secrets Manager

- **The consumer contract is unchanged:** every service still reads secrets as `NAME_FILE` files under root-only tmpfs `/run/secrets` (IF-02). Tasks that only consume `NAME_FILE` need no change beyond wording.
- **Storage:**
  - Per environment, a SOPS-encrypted file `deploy/secrets/<env>.sops.yaml`, encrypted to that environment's age recipient(s). The encrypted file is safe to keep in the tree.
  - Each environment has its own age key pair. The private keys are held off-host by the owner or operator, are never in the repo, the images or the plans, and are supplied at deploy time through `SOPS_AGE_KEY_FILE` to a root-only path.
- **Rendering:** `deploy/render_secrets.py` decrypts with a pinned `sops` binary (with a pinned `age`) into tmpfs `/run/secrets`, one file per secret with mode 0400, and fails closed on any decrypt error.
  - The plaintext `.env.prod` and Caddy's env_file are retired.
  - The keys in the working-tree `.env` are rotated, without the value ever being printed.
- **Rotation:** a documented re-encrypt-and-redeploy procedure. A key compromise means re-keying with a new age recipient.
- **Tests:** generate a throwaway age key in a temp dir and use synthetic secrets only. Never print a secret value. If the sops/age binaries are not available locally, tests that need them carry the realinfra marker and run in a container with pinned binaries.
- **Rules:**
  - Do not plan any AWS Secrets Manager calls.
  - The global rule stays: any AWS secrets work would load the aws-secrets-manager skill, and no agent ever calls `get-secret-value`.
  - No secret value enters a prompt, report or log.

## D-25 (changed): AWS hosts the release gate and the pilot, using the existing IAM user for now

- **Account:** 253025918446, us-east-1, existing IAM user `MedCodex-Local` (profiles default and medcodex-local).
- **Permission check (2026-10-19, read-only IAM simulation):**
  - Allowed: `bedrock:InvokeModel`, `bedrock:ListFoundationModels`.
  - Denied (implicit): cloudformation:CreateStack/DescribeStacks, ec2:RunInstances/DescribeInstances, ssm:StartSession/SendCommand/GetParameter, iam:CreateRole/PassRole, s3:CreateBucket/PutObject, kms:CreateKey, ses:SendEmail/VerifyDomainIdentity.
- So provisioning stays blocked until the owner attaches `infrastructure/aws-staging/solace-staging-deployer-policy.json` to `MedCodex-Local` (option 2 of AWS-STAGING-IAM-SETUP.pdf). This is an external owner action, needed before:
  - W4's off-host S3 backup target (until then the restore drill may use the local encrypted-volume target and say so);
  - W5's release gate;
  - go-live.
- Until then all real-infra tests run on local Docker.
- **Release gate:** it runs on the AWS staging host (the EC2 host from solace-staging.yaml, reached via SSM) with the strict production profile. If the policy is still unattached at W5 entry, that is a stop for the owner (no silent fallback to local).
- **Deploy docs:** `infrastructure/aws-staging/DEPLOY.md` and the deployer policy say "attach to the existing user MedCodex-Local" as the primary path.
- **The policy must also cover** what the decided design needs: S3+KMS for backups (D-09), SES sending identity setup (D-12), and Bedrock invocation through the instance role (D-04).
- **No SSM Parameter Store or Secrets Manager for secrets** (D-07 is SOPS).

## D-29 (changed): several countries, US first

- **Resource registry:** crisis resources become a per-locale, versioned registry in `solace_clinical`.
  - Only `en-US` is populated now (988 Suicide and Crisis Lifeline, call or text 988; Crisis Text Line, text HOME to 741741; 911). Each further country is added with clinician-verified resources before its users join.
  - `crisis_resources("en-US")` must keep returning exactly today's en-US set, so existing en-US assertions stay valid.
  - The version is tracked per locale.
  - An unsupported or unknown locale gets a safe generic fallback: contact your local emergency number, plus an international helpline directory. It is never empty and never an error. It is logged without PHI, with a metric.
- **Crisis invariant (unchanged):** in-band resources never depend on the LLM, a downstream service, Kafka or the database. The user's locale must therefore be available at screen time without a DB or network call.
  - Preferred: a locale claim in the access token (for example `loc`), issued at login from the user's profile locale, default `en-US`.
  - A request-supplied locale may refine it only within the supported set.
  - Choose and document one contract; IF-06 and IF-12 consumers must agree.
- **Who admits users:** pilot admission of a non-US user is an operator and cohort decision (D-31). Engineering only guarantees the fallback, and the alerts for an unsupported locale.
