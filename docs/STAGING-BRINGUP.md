# Staging Bring-Up Runbook

Purpose: stand up the full Solace-AI stack on a staging host so the infra-gated work
(Stage 3 in `docs/REMAINING-WORK-PLAN.md`) can be verified: live image builds, crisis E2E,
RLS + GDPR erasure on real Postgres, the LangGraph checkpointer fix, OAuth/WS, perf smoke.

Grounded in the B.1/B.2 work: repo-root Docker build contexts, `docker-compose.yml` +
`docker-compose.prod.yml`, the required-secret guards, and `scripts/generate_jwt_keys.py`.

---

## 0. Prerequisites (on the staging host)
- Docker Engine + Docker Compose v2 running (`docker info` must succeed — the daemon, not just the CLI).
- The repo checked out. ~8 GB RAM free (personality/diagnosis/orchestrator are the heavy ones).
- Outbound network for base images + Python wheels on first build.

## 1. Create the staging secrets file (`.env.prod` at repo root)
These are the values the prod override REQUIRES (it fails fast if any are missing). Generate
strong random values — do NOT reuse the dev defaults.

```bash
# 32-byte secrets
python -c "import secrets; print('JWT_SECRET_KEY='+secrets.token_urlsafe(48))"
python -c "import secrets; print('FIELD_ENCRYPTION_KEY='+secrets.token_urlsafe(32))"
python -c "import secrets,base64; print('ENCRYPTION_MASTER_KEY='+secrets.token_bytes(32).hex())"
python -c "import secrets; print('ENCRYPTION_SEARCH_HASH_SALT='+secrets.token_urlsafe(32))"
python -c "import secrets; print('POSTGRES_PASSWORD='+secrets.token_urlsafe(24))"
python -c "import secrets; print('AUDIT_HMAC_KEY='+secrets.token_urlsafe(48))"
python -c "import secrets; print('WEAVIATE_API_KEY='+secrets.token_urlsafe(32))"
# Fernet keys (urlsafe base64, 32 bytes)
python -c "from cryptography.fernet import Fernet; print('FERNET_TOKEN_KEY='+Fernet.generate_key().decode())"
python -c "from cryptography.fernet import Fernet; print('FERNET_FIELD_KEY='+Fernet.generate_key().decode())"
```

`.env.prod` must contain (append the outputs above):
```
ENVIRONMENT=staging          # REQUIRED — global gate for SSL/audit-HMAC/encryption. The prod
                             # override guards it with :? (fails fast if unset) AND derives the
                             # memory/notification/analytics per-service gates from it (P1-8/P1-9),
                             # so you do NOT set MEMORY_ENVIRONMENT / NOTIFICATION_SERVICE_ENV /
                             # ANALYTICS_SERVICE_ENV by hand. Use `production` on a real prod deploy.
AUDIT_HMAC_KEY=...           # REQUIRED (P1-8) — without it audit hashes fall back to non-HMAC.
# REQUIRED (Phase C gate P0-2/REV-32): the base compose defaults every service's
# environment to `development`; the services below read their OWN `<SVC>_ENVIRONMENT`
# (which the prod override does NOT auto-derive), so you MUST set them or the production
# fail-loud guards (SSL, audit-HMAC, encryption, crisis event-delivery) silently no-op.
# Use `production` on a real prod deploy.
SAFETY_ENVIRONMENT=staging
DIAGNOSIS_ENVIRONMENT=staging
THERAPY_ENVIRONMENT=staging
PERSONALITY_ENVIRONMENT=staging
ORCHESTRATOR_ENVIRONMENT=staging
CONFIG_ENVIRONMENT=staging
# P1-9 FIXED: memory/notification/analytics no longer need a hand-set gate var — the base
# compose now binds the correct names (MEMORY_ENVIRONMENT / NOTIFICATION_SERVICE_ENV /
# ANALYTICS_SERVICE_ENV, all overridable) and the prod override pins them to ${ENVIRONMENT}.
JWT_SECRET_KEY=...
FIELD_ENCRYPTION_KEY=...
FERNET_TOKEN_KEY=...
FERNET_FIELD_KEY=...
ENCRYPTION_MASTER_KEY=...
ENCRYPTION_SEARCH_HASH_SALT=...
POSTGRES_PASSWORD=...
WEAVIATE_API_KEY=<random>    # REQUIRED (P2-16) — prod override guards it with :?; a missing key
                             # would otherwise boot Weaviate on the public dev default key.
ORCHESTRATOR_TRUSTED_PROXY_HOPS=1   # P2-12 — Caddy appends one XFF hop; the prod override
                             # already defaults this to 1, set here only to override.
PORTKEY_API_KEY=<from portkey.ai, for LLM calls>   # or leave LLM disabled for infra smoke
DOMAIN=solace-staging.local
CADDY_ADMIN_EMAIL=ops@example.com
BASIC_AUTH_HASH=<caddy hash-password output, for Grafana/Jaeger>
# GOOGLE_OAUTH_CLIENT_ID / _SECRET — only needed once OAuth endpoints land (REV-22)
```
Keep `.env.prod` OUT of git (already gitignored; never commit it).

## 2. (Optional now / required for prod) RS256 JWT keys
Default is HS256 (works out of the box). To exercise the C.3 asymmetric path on staging:
```bash
python scripts/generate_jwt_keys.py --kid staging-2026-07   # prints AUTH_* vars
# add its output to .env.prod, and set AUTH_ALGORITHM=RS256 (AUTH_PUBLIC_KEYS_JSON on all
# services; AUTH_PRIVATE_KEY_PEM + AUTH_ACTIVE_KID only on user-service).
```

## 3. Build + start the stack
```bash
# Build all 10 service images from the repo root (validates B.1/REV-10):
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml build

# Bring up infra + services (prod override: no mailhog/kafka-ui, secrets required, Caddy TLS):
docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml up -d

# For a plain internal staging WITHOUT Caddy/host-port-closing, the dev compose also works:
#   docker compose up -d --build           # (dev defaults; add --profile dev for mailhog/kafka-ui)
```

## 4. Run migrations (creates schema + RLS policies 001→006)
```bash
# Point Alembic at the staging DB (migrations/env.py honors DATABASE_URL):
export DATABASE_URL="postgresql+asyncpg://solace:${POSTGRES_PASSWORD}@localhost:5432/solace"
alembic upgrade head     # applies 001..006, incl. RLS on the 9 user-token tables
```

## 5. Health checks (each service exposes /health)
```bash
for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009; do
  printf "port %s: " "$p"; curl -fsS "http://localhost:$p/health" || echo "DOWN"; echo
done
# Infra: postgres (pg_isready), redis (redis-cli ping), weaviate (:8080/v1/.well-known/ready),
#        kafka (broker-api-versions). JWKS: curl :8001/.well-known/jwks.json
```

## 6. Hand back to me — what I then drive (Stage 3)
Once the stack is green, tell me and I will:
1. Verify all 10 images built (REV-10/11) + fix any build gaps.
2. Run the crisis E2E on real PG/Redis/Kafka/Weaviate.
3. **Verify RLS** (a query as user A cannot see user B's rows once the GUC is set) and **GDPR erasure** across PG/Redis/Weaviate.
4. Fix + verify the **orchestrator LangGraph checkpointer** (`AsyncPostgresSaver.setup()` needs live PG).
5. OAuth live endpoints + WS reconnect replay; distributed rate-limit + G6 perf smoke.
6. Add the real CI docker-build + integration gates.

## Known gotchas
- **Daemon must be up** — earlier work hit `npipe ... dockerDesktopLinuxEngine` when Docker Desktop wasn't running.
- **`ENVIRONMENT` must be set** in the container env or SSL/audit-HMAC/encryption fail-loud gates behave inconsistently (REV-32). The prod override sets it via `.env.prod`.
- First build is slow (Python wheels incl. langgraph/weaviate); subsequent builds cache.
- `require` SSL currently uses `CERT_NONE` (encryption, no cert verify) — fine for staging; prod needs `verify-full` + pinned cert (C.4).
