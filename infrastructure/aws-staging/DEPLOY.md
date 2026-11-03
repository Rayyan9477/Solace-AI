# Solace-AI AWS Staging Deploy Runbook

The AWS-specific wrapper around `docs/STAGING-BRINGUP.md`. It creates the staging host
with CloudFormation, then runs the documented bring-up on it over SSM. Ephemeral and
cost-capped; **synthetic data only** (no live PHI until the HIPAA BAA is signed).

Artifacts in this folder:
- `solace-staging.yaml` -- the CloudFormation template (cfn-lint clean).
- `solace-staging-deployer-policy.json` -- the least-privilege deployer IAM policy.
- `AWS-STAGING-IAM-SETUP.pdf` -- how to grant the deployer permissions (two options).

Account `253025918446`, region `us-east-1`. All resources are named/tagged `solace-staging`.

---

## 0. Prerequisite: the deployer grant

Complete `AWS-STAGING-IAM-SETUP.pdf` first (attach the three managed policies + the inline
policy to a `solace-staging-deployer` user or to `MedCodex-Local`). Then, using that profile
(`solace-staging` for the dedicated user, or `default` / `medcodex-local` for the existing one):

```bash
PROFILE=solace-staging          # or default / medcodex-local
aws sts get-caller-identity --profile "$PROFILE"
```

## 1. Pick a public subnet in the default VPC

```bash
VPC=vpc-07664ce966d106387
aws ec2 describe-subnets --profile "$PROFILE" --region us-east-1 \
  --filters "Name=vpc-id,Values=$VPC" "Name=map-public-ip-on-launch,Values=true" \
  --query "Subnets[].{Subnet:SubnetId,AZ:AvailabilityZone}" --output table
```
Pick any `SubnetId` from the list.

## 2. Create the stack

```bash
aws cloudformation create-stack \
  --stack-name solace-staging \
  --template-body file://solace-staging.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
      ParameterKey=SubnetId,ParameterValue=<SUBNET_ID> \
      ParameterKey=InstanceType,ParameterValue=m5.xlarge \
  --tags Key=Project,Value=solace-staging \
  --profile "$PROFILE" --region us-east-1

# Wait until it is up, then read the outputs.
aws cloudformation wait stack-create-complete \
  --stack-name solace-staging --profile "$PROFILE" --region us-east-1
aws cloudformation describe-stacks --stack-name solace-staging \
  --profile "$PROFILE" --region us-east-1 \
  --query "Stacks[0].Outputs" --output table
```
`CAPABILITY_NAMED_IAM` is required because the template creates the named
`solace-staging-role` / `solace-staging-instance-profile`.

## 3. (Optional) cost-guardrail budget

```bash
cat >/tmp/solace-staging-budget.json <<'JSON'
{ "BudgetName": "solace-staging-monthly", "BudgetLimit": {"Amount":"50","Unit":"USD"},
  "TimeUnit": "MONTHLY", "BudgetType": "COST" }
JSON
aws budgets create-budget --account-id 253025918446 \
  --budget file:///tmp/solace-staging-budget.json --profile "$PROFILE"
```

## 4. Open a shell on the host (SSM, no SSH)

```bash
INSTANCE=$(aws cloudformation describe-stacks --stack-name solace-staging \
  --profile "$PROFILE" --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
aws ssm start-session --target "$INSTANCE" --profile "$PROFILE" --region us-east-1
```
Wait for `/opt/solace-staging-ready` to exist (UserData finished installing Docker):
`sudo cloud-init status --wait` then `test -f /opt/solace-staging-ready && echo ready`.

## 5. Get the repo onto the host

The repo is private, so the host cannot auto-clone. On the host (inside the SSM session):

```bash
sudo dnf install -y git   # already installed by UserData; safe to repeat
git clone https://<TOKEN>@github.com/<org>/Contextual-Chatbot.git solace   # or use S3 / scp
cd solace
```
Prefer a short-lived, read-only deploy token. Alternatively push a tarball to S3 and
`aws s3 cp` it down using the instance role (grant S3 read if you take this path).

## 6. Bring up the stack

Follow `docs/STAGING-BRINGUP.md` sections 1-5 on the host:
1. Create `.env.prod` with `ENVIRONMENT=staging` and the required secrets (the startup guards
   now fail loud without `POSTGRES_PASSWORD` / `JWT_SECRET_KEY` / encryption keys and with
   `POSTGRES_SSL_MODE` below `require`).
2. `docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml up -d --build`
3. `alembic upgrade head`
4. Health-check ports 8000-8009.

## 7. Hand back -- what Claude then drives (Stage 3)

Tell Claude the stack is green. Claude then verifies the infra-gated bucket: all 10 image
builds, crisis E2E on real Kafka, RLS + GDPR erasure on real Postgres (incl. P1-4 cross-user),
the LangGraph `AsyncPostgresSaver.setup()`, outbox two-replica double-publish, Postgres
`verify-full` + Redis TLS floors, distributed rate-limit + perf smoke, and the CI docker gates.
`scripts/seed_demo_data.py` provides the synthetic users/sessions for the E2E and RLS proofs.

## 8. Teardown (when verification is done)

```bash
aws cloudformation delete-stack --stack-name solace-staging --profile "$PROFILE" --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name solace-staging --profile "$PROFILE" --region us-east-1
# optional:
aws budgets delete-budget --account-id 253025918446 --budget-name solace-staging-monthly --profile "$PROFILE"
```
Deleting the stack terminates the instance and its EBS volume (DeleteOnTermination=true) and
removes the `solace-staging-role` / instance profile / security group. To also revoke the
deployer grant, follow section 7 of `AWS-STAGING-IAM-SETUP.pdf`.
