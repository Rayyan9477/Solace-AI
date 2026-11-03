# Interfaces and file ownership

> Appendix to the [MVP remediation plan](../2026-10-18-mvp-remediation.md). Implementers use these names, signatures, env vars, topics and headers exactly. A change to a file another workstream owns goes through that owner (the plan's cross-workstream requests).

## Interfaces

### IF-01: Service runtime bootstrap

**Owner:** WS-04. **Consumers:** WS-03, WS-05, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

Package solace_infrastructure.runtime (src/solace_infrastructure/runtime/: spec.py, bootstrap.py, logging.py, http.py, health.py, metrics.py, datastores.py). from solace_infrastructure.runtime import ServiceRuntimeSpec, ServiceRuntime, create_service_app, start_service_runtime, stop_service_runtime, is_strict_environment, StartupDependencyError.

def is_strict_environment(env: str | None = None) -> bool: True iff env (or os.environ['ENVIRONMENT'], default 'development', when None) lower-cased is 'staging' or 'production'. Every guard uses it; none compares to 'production' alone.

@dataclass(frozen=True) class ServiceRuntimeSpec:
- service_name: str (ServiceIdentity value, e.g. 'safety-service'); environment: str (the service's resolved setting); version: str = '1.0.0'; log_level: str = 'INFO'; debug: bool = False
- uses_postgres: bool = False; uses_redis: bool = False
- event_plane: Literal['none','publish','publish_consume'] = 'none'
- handles_phi: bool = False; audited: bool = False; rate_limited: bool = False
- edge: bool = False (orchestrator and user-service: CORS from CORS_ALLOWED_ORIGINS, TRUSTED_PROXY_HOPS)
- docs_enabled: bool | None = None (None means /docs and /openapi.json only when not strict)

def create_service_app(spec, *, lifespan, routers: Sequence[tuple[APIRouter, str]], title: str) -> FastAPI installs, in order:
- request-context middleware: mints or echoes X-Request-ID, X-Correlation-ID and X-Turn-ID, binds them into structlog contextvars, writes one access line per request with route template, method, status and duration, never the query string;
- CORS when spec.edge;
- exception handlers: RequestValidationError -> 422 {error:{code:'VALIDATION_ERROR',message:'Request validation failed',details:[{field,message,type}],request_id}}; ValueError -> 400 {error:{code:'BAD_REQUEST',message:'Invalid request',request_id}}; unhandled Exception -> 500 {error:{code:'INTERNAL_ERROR',message:'An unexpected error occurred',request_id}}; never str(exc); HTTPException: edge services (spec.edge) render {error:{code,message,request_id}} from W2 per D-11, internal services keep FastAPI's {detail} body;
- GET /live -> 200 {status:'alive'}, or 503 {status:'dead',task} when a registered background task has exited;
- GET /ready and GET /health -> 200 {status:'ready',service,version,checks:{<name>:{ok,critical,detail}}}, or 503 with status 'not_ready' when any critical check fails;
- GET /metrics (prometheus_client text);
- from W1, solace_security.rate_limit.install_rate_limiting(app, spec) when spec.rate_limited.

async def start_service_runtime(app, spec) -> ServiceRuntime is the first statement of every lifespan. Order: (1) ProductionGuards.assert_startup(spec.environment, service_name=spec.service_name); (2) configure_service_logging(spec); (3) Postgres pool via ConnectionPoolManager.get_pool('default') plus assert_schema_head in strict when uses_postgres, Redis from REDIS_URL when uses_redis; (4) [W1] ServiceTokenManager and start_service_event_plane when event_plane != 'none'; (5) [W2] get_phi_codec() when handles_phi; (6) [W3] configure_service_audit when audited; (7) readiness checks for every started component. Sets app.state.runtime. Strict: any failure raises StartupDependencyError and the process exits non-zero. development/test: WARNING log and the component's readiness check reports ok=False (/ready 503).
async def stop_service_runtime(runtime) -> None is the last statement of every lifespan: registered closers in reverse order, 5 s timeout each.

class ServiceRuntime: spec; pg_pool: asyncpg.Pool | None; redis: redis.asyncio.Redis | None; event_plane: ServiceEventPlane | None; service_tokens: ServiceTokenManager | None; phi_codec: PHICodec | None; register_readiness_check(name: str, check: Callable[[], Awaitable[tuple[bool, str]]], *, critical: bool = True) -> None; register_background_task(name: str, task: asyncio.Task) -> None; register_shutdown(name: str, closer: Callable[[], Awaitable[None]]) -> None; outbound_headers() -> dict[str, str]. Critical checks cover only the service's own datastores, broker and event plane; peers are non-critical.

configure_service_logging(spec): merge_contextvars, add_log_level, TimeStamper(fmt='iso'), StackInfoRenderer, format_exc_info, phi_sanitizer_processor (mandatory in strict; an import failure is fatal), JSONRenderer (ConsoleRenderer only when debug and not strict); uvicorn loggers routed through it.

Adoption rule (W0): main.py builds SPEC at import with every flag the service needs through W5; the lifespan's first statement is 'runtime = await start_service_runtime(app, SPEC)' and its last is 'await stop_service_runtime(runtime)'; composition uses runtime.pg_pool, runtime.redis and runtime.event_plane; no try/except around platform components; 'app = create_service_app(SPEC, lifespan=lifespan, routers=[...], title=...)'. tests/bootstrap/test_service_conformance.py imports the nine apps and asserts SPEC flags, the /live /ready /health /metrics routes, the logging chain and the absence of local exception handlers.

### IF-02: Deployment environment contract

**Owner:** WS-03. **Consumers:** WS-01, WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

Registry deploy/env/env-contract.yaml: one entry per variable {name, services[], secret: bool, required_in_strict: bool, dev_default, notes, retired: bool}. .env.example is rendered from it by deploy/env/render_env_example.py. tests/deploy/test_env_contract.py renders 'docker compose -f docker-compose.yml config' (dev) and '-f docker-compose.yml -f docker-compose.prod.yml config' (strict) and asserts: every required_in_strict variable is set for its services; no dev default survives in strict; no internal service publishes a host port in strict; no code under src/ or services/ reads a retired name.

Environment: ENVIRONMENT in {development, test, staging, production}. ORCHESTRATOR_ENVIRONMENT, SAFETY_ENVIRONMENT, THERAPY_ENVIRONMENT, DIAGNOSIS_ENVIRONMENT, MEMORY_ENVIRONMENT, PERSONALITY_ENVIRONMENT, USER_SERVICE_ENV, NOTIFICATION_SERVICE_ENV and ANALYTICS_SERVICE_ENV are always ${ENVIRONMENT}, never literals.
Postgres (DB-using services): POSTGRES_HOST=postgres, POSTGRES_PORT=5432, POSTGRES_DATABASE=solace, POSTGRES_USER=solace_app, POSTGRES_PASSWORD, POSTGRES_SSL_MODE (verify-full in strict, prefer in dev), POSTGRES_SSL_ROOT_CERT=/run/secrets/postgres-ca.crt, POSTGRES_MIN_POOL_SIZE=1, POSTGRES_MAX_POOL_SIZE=8. Migrate job only: DATABASE_URL for solace_migrator. POSTGRES_SUPERUSER_PASSWORD and POSTGRES_MIGRATOR_PASSWORD are compose-interpolation/ops only.
Redis: REDIS_URL=redis://:<password>@redis:6379/0 (secret) on every Redis-using service; REDIS_PASSWORD on the redis container.
Kafka: KAFKA_BOOTSTRAP_SERVERS=kafka:9092 on every app container; EVENTS_ENABLED=true (false allowed only in development/test) replaces SAFETY_KAFKA_ENABLED, THERAPY_KAFKA_ENABLED, notification KAFKA_ENABLED and the other per-service Kafka enable flags in W1 (they are set true until then). Every app service, the orchestrator included, joins the data network.
Peers (every app container): SERVICE_SAFETY_SERVICE_URL=http://safety-service:8002, SERVICE_NOTIFICATION_SERVICE_URL=http://notification-service:8003, SERVICE_DIAGNOSIS_SERVICE_URL=http://diagnosis-service:8004, SERVICE_MEMORY_SERVICE_URL=http://memory-service:8005, SERVICE_THERAPY_SERVICE_URL=http://therapy-service:8006, SERVICE_PERSONALITY_SERVICE_URL=http://personality-service:8007, SERVICE_USER_SERVICE_URL=http://user-service:8001.
Auth: AUTH_SECRET_KEY (service tokens; user tokens until W4), AUTH_ALGORITHM, AUTH_PUBLIC_KEYS_JSON, AUTH_ACTIVE_KID, AUTH_ACCEPT_LEGACY_HS256 (false from W4), AUTH_PRIVATE_KEY_PEM (user-service only), USER_SERVICE_JWT_SECRET_KEY (= AUTH_SECRET_KEY until W4); from W4 SERVICE_TOKEN_PRIVATE_KEY_PEM (each service its own) and SERVICE_TOKEN_PUBLIC_KEYS_JSON (all services).
PHI and audit: ENCRYPTION_MASTER_KEY, ENCRYPTION_SEARCH_HASH_SALT, ENCRYPTION_KEY_ID, ENCRYPTION_RETIRED_KEYS_JSON, AUDIT_HMAC_KEY; FERNET_TOKEN_KEY and FERNET_FIELD_KEY (user-service until codec adoption).
Edge: CORS_ALLOWED_ORIGINS and TRUSTED_PROXY_HOPS (1 in strict) on orchestrator and user-service; DOMAIN, CADDY_ADMIN_EMAIL, BASIC_AUTH_HASH (quoted), CADDY_API_SITE, CADDY_AUTH_SITE.
LLM (orchestrator, safety, therapy, diagnosis, personality; existing LLM_ prefix): LLM_PORTKEY_API_KEY, LLM_PORTKEY_GATEWAY_URL (the harness points it at fake-llm), LLM_ANTHROPIC_API_KEY, LLM_ENABLE_CACHING=false, LLM_TIMEOUT_SECONDS, LLM_PHI_EGRESS_APPROVED=false.
Notification: EMAIL_ENABLED, EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, EMAIL_SMTP_USERNAME, EMAIL_SMTP_PASSWORD, EMAIL_SMTP_TLS_MODE (starttls|tls|none; none only in development/test), EMAIL_FROM_EMAIL, EMAIL_FROM_NAME, NOTIFICATION_FALLBACK_ONCALL_EMAIL, NOTIFICATION_ALERT_COALESCE_SECONDS=900, NOTIFICATION_REPAGE_AFTER_SECONDS=900, NOTIFICATION_MAX_REPAGES=3, NOTIFICATION_ACK_BASE_URL, SMS_ENABLED and SMS_* (optional), PUSH_ENABLED=false.
OAuth (user-service): GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REDIRECT_URI, GOOGLE_OAUTH_ISSUER, GOOGLE_OAUTH_AUTHORIZE_URL, GOOGLE_OAUTH_TOKEN_URL, GOOGLE_OAUTH_JWKS_URL, OAUTH_ALLOWED_REDIRECT_ORIGINS.
Orchestrator: ORCHESTRATOR_TURN_TIMEOUT_SECONDS=20, ORCHESTRATOR_AGENT_TIMEOUT_SECONDS=8, ORCHESTRATOR_SAFETY_CHECK_TIMEOUT_SECONDS=2, ORCHESTRATOR_WS_MAX_PER_USER=3.
Memory: WEAVIATE_ENABLED=false (D-04), WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT, WEAVIATE_API_KEY, MEMORY_EMBEDDINGS_PROVIDER=none.
Any secret NAME may be supplied as NAME_FILE (IF-22).
Legacy names (USER_DB_*, DIAGNOSIS_DB_*, THERAPY_DB_*, MEMORY_DB_*, PERSONALITY_DB_*, *_REDIS_HOST/PORT/PASSWORD, THERAPY_SESSION_REDIS_*, DIAGNOSIS_SESSION_REDIS_*, NOTIFICATION_USER_SERVICE_URL, ESCALATION_NOTIFICATION_SERVICE_URL, ORCHESTRATOR_*_AGENT_SERVICE_URL, ORCHESTRATOR_CORS_ORIGINS, SECRET_KEY, ENCRYPTION_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY) are set to identical values until the owning lane stops reading them, then marked retired.

### IF-03: Database roles, RLS context and migration slots

**Owner:** WS-02. **Consumers:** WS-03, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

Roles (infrastructure/init-db creates them; 009_platform_roles_rls grants):
- solace_migrator: LOGIN NOSUPERUSER, owns every table, runs migrations.
- solace_app: LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE; USAGE on schema public; SELECT/INSERT/UPDATE/DELETE on tables and USAGE/SELECT on sequences via ALTER DEFAULT PRIVILEGES FOR ROLE solace_migrator; INSERT/SELECT only on audit_events; no DDL. Services connect only as solace_app.
- The image superuser 'postgres' is used only by init-db, ops (physical backup and restore) and test assertions.

RLS: every policy is 'user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid' for USING and WITH CHECK. ConnectionPoolManager.acquire() and .transaction() set the GUC from solace_common.request_context.get_current_user_id() and reset it to '' on release; RLS tables are accessed only through them, never through a raw pool.acquire(). Work without a request (consumers, registration, reapers, CLIs) runs inside 'with acting_as_user(user_id):' or touches only non-RLS tables (users, escalations, event_outbox, processed_events, event_dead_letter, notifications, delivery_attempts, crisis_alerts, audit_events).

Pools: 'await ConnectionPoolManager.get_pool('default')' is the one pool per service, built from POSTGRES_*; legacy names (user_db, diagnosis_db, therapy_db, personality_db) alias 'default'; strict envs raise DatabaseConfigError when POSTGRES_HOST is unset or localhost or the password is missing; DSN passwords are percent-encoded; JSONB parameters are Python objects (the pool codec serializes; never pass json.dumps output).

Migrations: env.py uses transaction_per_migration=True; no service runs DDL at runtime (ensure_table() is verify-only); compose service 'migrate' (image solace-migrate) runs 'python -m scripts.db.migrate upgrade head' as solace_migrator under pg_advisory_lock(hashtext('solace_migrate')); app services depend_on migrate with condition service_completed_successfully. solace_infrastructure.database.schema_check.assert_schema_head(pool) raises SchemaVersionMismatch in strict envs unless alembic_version equals SCHEMA_HEAD = '021_personality_analytics'.

Slots: one linear chain after '008_rls_personality'; revision id = slot name; file migrations/versions/<slot>.py:
- 009_platform_roles_rls: WS-02
- 010_event_plane: WS-06
- 011_audit_trail: WS-09
- 012_user_accounts: WS-14
- 013_identity_mfa: WS-08
- 014_safety: WS-12
- 015_notification: WS-13
- 016_orchestrator_threads: WS-11
- 017_orchestrator_checkpoints: WS-10
- 018_memory: WS-17
- 019_therapy: WS-15
- 020_diagnosis: WS-16
- 021_personality_analytics: WS-18
Owners edit only upgrade() and downgrade(); every slot downgrades cleanly. Slot 017 applies the checkpoint DDL of the pinned langgraph-checkpoint-postgres version (statements with CONCURRENTLY inside op.get_context().autocommit_block()) and records it in checkpoint_migrations, so AsyncPostgresSaver.setup() is never needed at runtime. tests/migrations/test_chain_real_pg.py on postgres:16: upgrade head, downgrade base, upgrade head; solace_app cannot CREATE TABLE; every policy uses NULLIF.

### IF-04: Inter-service authentication, authorization and rate limiting

**Owner:** WS-07. **Consumers:** WS-04, WS-08, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

solace_security.service_auth:
- ServiceTokenManager.create_service_token(service_name: str, target_service: str | None = None, additional_permissions: list[str] | None = None, expire_minutes: int | None = None, *, on_behalf_of: str | None = None) -> ServiceCredentials. Claims: sub='service:<name>', type='service', permissions, metadata.service_name, iss='solace-ai', aud='solace-ai-api'. obo tokens add metadata.obo_sub=<user uuid>, live at most 300 s and are never cached across users.
- async def service_auth_headers(target_service: str, *, on_behalf_of: str | None = None) -> dict[str, str]: 'Authorization: Bearer <service token>' plus X-Request-ID, X-Correlation-ID and X-Turn-ID from context.
- def forwarded_auth_headers(request: Request) -> dict[str, str]: the inbound Authorization header unchanged plus the correlation headers. Used when a service handling an obo request calls another service for the same user (therapy to memory); services never mint their own obo tokens.
- New ServicePermission values: ACT_FOR_USER='service:act_for_user' (orchestrator-service only) and READ_ONCALL_ROSTER='service:read:oncall' (notification-service only). Existing WRITE_NOTIFICATION='service:write:notification' is granted to user-service (safety's grant is removed with D-28); orchestrator keeps READ_SAFETY for /check. Receivers resolve permissions from SERVICE_PERMISSIONS[ServiceIdentity(verified service name)], never from token claims.

solace_security.middleware (names stable):
- get_current_user accepts ACCESS tokens and SERVICE tokens whose service holds ACT_FOR_USER and whose metadata.obo_sub is a UUID. The latter yields AuthenticatedUser(user_id=obo_sub, token_type=SERVICE, roles=['user'], metadata={'via_service': name}) and sets the RLS user. Anything else is 401.
- AuthenticatedUser.via_service: str | None. get_current_service: SERVICE tokens only, revocation-aware. require_service_permission(*perms) on every service-only route.
- W3: can_access_subject(principal, subject_user_id) -> bool and set_subject_context(subject_user_id), which sets the RLS user to the data owner only after authorization (D-21). W4: require_mfa_for_roles(*roles).

solace_common.request_context.acting_as_user(user_id): context manager that sets and then resets the contextvar (W0).

Blacklist: RedisTokenBlacklist from REDIS_URL in every service; in-memory only in development/test; the URL is never logged.

Rate limiting: solace_security.rate_limit.install_rate_limiting(app, spec). Key user:<sub> with a valid bearer, else ip:<client> from client_ip(request, trusted_proxy_hops=TRUSTED_PROXY_HOPS). Redis backend; on Redis failure auth routes return 503 and other routes pass, counting solace_rate_limit_backend_errors_total (D-26). Bounded in-memory store in development/test.

User JWTs are never forwarded downstream and never enter LangGraph state or checkpoints. From W4: user tokens are RS256-only (AUTH_ACCEPT_LEGACY_HS256=false); service tokens are RS256, signed with the caller's own SERVICE_TOKEN_PRIVATE_KEY_PEM (kid = service name) and verified against SERVICE_TOKEN_PUBLIC_KEYS_JSON; a token whose kid does not match its sub is rejected.

### IF-05: Event plane API and delivery semantics

**Owner:** WS-06. **Consumers:** WS-04, WS-05, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

solace_events.service_plane (new):
- async def start_service_event_plane(service_name: str, *, pool: asyncpg.Pool | None, kafka_settings: KafkaSettings | None = None, strict: bool) -> ServiceEventPlane (KafkaSettings reads KAFKA_BOOTSTRAP_SERVERS); def get_service_event_plane() -> ServiceEventPlane returns the process singleton that the bridges use.
- ServiceEventPlane.publish(event: BaseEvent, *, conn: asyncpg.Connection | None = None) -> UUID: returns after the outbox row is committed, or inside the caller's transaction when conn is given; raises EventPublishError; never an in-memory outbox when strict.
- .subscribe(topic: str, event_types: Iterable[str], handler: Callable[[BaseEvent], Awaitable[None]], *, group_id: str, max_attempts: int = 5) -> None, called before start().
- .start(): the producer connects with exponential backoff up to 60 s and keeps retrying in the background; readiness is false until connected. .stop(): finishes the in-flight flush and releases claims. .readiness() -> tuple[bool, str].
- Errors (solace_events.errors): RetryableEventError, NonRetryableEventError, EventPublishError.

Bridges: services/*/src/infrastructure/event_bridge.py keep their initialize_* names and keyword arguments, delegate to the plane and always start an OutboxPoller.

Semantics:
- event_outbox.source_service VARCHAR(64) NOT NULL; each poller claims only its own rows; claims are released on cancel; state transitions are claim-guarded.
- A handler exception (or RetryableEventError) means no offset commit and in-place retry with exponential backoff starting at 1 s, doubling, capped at 16 s, until max_attempts (default 5) attempts have failed, then a durable event_dead_letter row and commit. Each attempt may run up to handler_timeout_seconds (30 s), so one message can hold its partition for about 165 s; partitions are processed concurrently (one worker per partition, order kept per partition, max_inflight_partitions 16), so a failing message never blocks another partition (crisis-review F1). NonRetryableEventError or a schema-invalid or malformed message goes to the DLQ immediately with its raw payload. Each message is processed and committed independently.
- processed_events(event_id, consumer_group) is written only after handler success; handlers make external side effects idempotent on business keys.
- A DLQ reaper per group re-drives retriable records every 60 s, at most 10 times each, without head-of-line blocking.
- Message key str(user_id); topics from solace_events.schemas; no Kafka *.dlq topics.
- W3: no free text in payloads; published outbox rows purged after 7 days and resolved DLQ rows after 30 days; async def erase_user_events(user_id: UUID) -> int for erasure.

MVP topics and events: solace.safety: safety.crisis.detected, safety.escalation.triggered, safety.assessment.completed; solace.users: user.deleted (UserDeletedKafkaEvent), user.erasure.completed (new UserErasureCompletedKafkaEvent: user_id, service_name: str, erasure_request_id: UUID, stores: dict[str, int], completed_at), user.consent_changed; analytics feeds solace.assessments, solace.therapy, solace.personality. Consumer groups: solace-group-notifications, solace-group-analytics, solace-group-erasure-<service>.

Metrics: solace_outbox_pending{service}, solace_outbox_oldest_pending_seconds{service}, solace_consumer_lag{group,topic} (summed over all partitions from broker state; the crisis group is sampled by every plane, crisis-review F13), solace_dlq_unresolved{group}, solace_event_publish_failures_total{service} (incremented by record_publish_failure(service_name), WS-06.10), solace_consumer_handler_failures_total{group,event_type}.

### IF-06: Crisis signal contract

**Owner:** WS-12. **Consumers:** WS-05, WS-10, WS-11, WS-13, WS-15, WS-16.

Levels: solace_common.enums.CrisisLevel {NONE, LOW, ELEVATED, HIGH, CRITICAL}. ELEVATED and above means in-band resources. HIGH and CRITICAL also mean a CrisisDetectedEvent, which pages (IF-09).

turn_id: a UUID the orchestrator mints per user message and sends downstream as header X-Turn-ID; every detector uses it as metadata.correlation_id; a direct caller without X-Turn-ID gets a fresh UUID.

CrisisDetectedEvent (solace.safety, 'safety.crisis.detected'): user_id UUID (never fabricated); session_id UUID (a non-UUID client id is mapped with uuid5(NAMESPACE_URL, 'solace:session:' + id)); crisis_level; trigger_indicators: category codes from solace_clinical.CATEGORIES only, never raw text; detection_layer (orchestrator 1 = local screen, 3 = LLM intent; safety 1-4; therapy 1; diagnosis 1 or 3); confidence; escalation_action='page_on_call'; requires_human_review=True; metadata.correlation_id = turn_id; metadata.source_service = producer. SafetyAssessmentEvent.risk_factors[].evidence and EscalationTriggeredEvent.escalation_reason carry category codes, never text.

Producers: orchestrator-service (ingress), safety-service (/check), therapy-service, diagnosis-service; each publishes at most one CrisisDetectedEvent per turn.

Orchestrator REST and WS responses carry crisis: {level, categories, resources: [{name, contact, kind: 'phone'|'text'|'url'}], resources_version} whenever the fused level is ELEVATED or higher.

safety-service POST /api/v1/safety/check (service token with service:read:safety): request {user_id, session_id, content, turn_id, context?}; response {level, risk_score, categories, requires_escalation, resources, detection_layer} within 2 s; never blocks on delivery (D-28). The orchestrator fuses the result with fuse_risk; a timeout keeps the local result and adds degraded 'safety_service_timeout'.

### IF-07: Shared crisis lexicon, fusion and resources

**Owner:** WS-12. **Consumers:** WS-05, WS-10, WS-15, WS-16, WS-17.

Package solace_clinical (src/solace_clinical; pure, no I/O, total over every str input):
- def normalize_text(text: str) -> str: NFKC; casefold; U+2018, U+2019, U+02BC and U+FF07 become an apostrophe; U+201C and U+201D become a double quote; zero-width characters removed; whitespace collapsed.
- def screen_text(text: str) -> ScreenResult. @dataclass(frozen=True) ScreenResult: level: CrisisLevel; categories: frozenset[str]; rule_ids: tuple[str, ...] (never matched text); negated_rule_ids: tuple[str, ...]; lexicon_version: str. Under 5 ms for 10,000 characters.
- CATEGORIES: frozenset {suicidal_ideation, suicide_plan, means_access, self_harm, overdose, homicidal_ideation, abuse_disclosure, imminent_danger, hopelessness (ELEVATED at most)}; LEXICON_VERSION: str.
- @dataclass(frozen=True) RiskSignal: source: str; level: CrisisLevel; confidence: float; reliable: bool. def fuse_risk(signals: Sequence[RiskSignal]) -> CrisisLevel takes the maximum; unreliable signals (stub, fallback, timeout, low-confidence LLM) may raise the level but never lower it.
- def crisis_resources(locale: str = 'en-US') -> list[dict] with {name, contact, kind}; RESOURCES_VERSION: str (D-29).
- Data: src/solace_clinical/crisis_lexicon_v1.yaml with metadata {version, approved_by, approved_at, jurisdiction} and rules {id, category, level, patterns, requires_first_person, negation_policy}. Negation applies only within the same clause and never to first-person intent, plan or means rules or to CRITICAL categories; context-free single words (cutting, pills, rope, bridge) need self-reference before HIGH.

Regression set tests/clinical/crisis_phrases_v1.yaml {positive: [{text, min_level, categories}], negative: [{text, max_level}]}, at least 200 phrases incl. curly apostrophes, indirect phrasing, homicidal and abuse disclosures, and benign 'cutting vegetables' / 'I could kill for a coffee'; marker clinical. Every detector must pass it: orchestrator precheck (WS-10), safety layer 1 (WS-12), therapy _check_safety (WS-15), diagnosis risk detection (WS-16), memory crisis retention (WS-17). hypothesis property tests: normalize_text and screen_text never raise and normalize_text is idempotent.

### IF-08: On-call roster, care team and clinician provisioning

**Owner:** WS-14. **Consumers:** WS-05, WS-08, WS-13.

New router services/user-service/src/internal_api.py mounted at /api/v1/internal; Caddy never routes it.
- GET /api/v1/internal/clinicians/on-call: SERVICE token with service:read:oncall (user tokens get 403) -> {clinicians: [{user_id, display_name, email, phone_e164, role, is_backup}]} for role CLINICIAN, status ACTIVE, on call.
- GET /api/v1/internal/users/{user_id}/care-team: same auth -> {user_id, display_name, contact_email, contact_phone_e164, crisis_contact_consent: bool, clinicians: [same shape]} from clinician_patient_assignments. Contact fields are null unless crisis_contact_consent is true (D-23).
- Public GET /api/v1/users/on-call-clinicians and PUT /api/v1/users/{user_id}/on-call become ADMIN-only (MFA-verified from W4).
- PUT /api/v1/admin/clinicians/{user_id}/on-call {on_call: bool, backup: bool} and POST /api/v1/admin/assignments {clinician_id, user_id}: ADMIN.
- Provisioning CLI, run inside the user-service container: 'python -m scripts.provision_clinician --email E --display-name N --phone +1XXXXXXXXXX [--on-call] [--backup] [--role CLINICIAN|ADMIN] [--assign USER_ID]'. Creates a verified ACTIVE account and prints only the new user id and a one-time set-password link; writes an audit event from W3; forces MFA enrollment from W4. No public route creates privileged roles.
- Access tokens carry 'cns': {ok: bool, version: str} for the required consents (D-23); revoking a required consent revokes sessions (W3).

### IF-09: Clinician alert delivery, crisis episodes and acknowledgement

**Owner:** WS-13. **Consumers:** WS-03, WS-04, WS-05, WS-08, WS-14.

Consumer group solace-group-notifications on solace.safety. safety.crisis.detected (HIGH/CRITICAL) and safety.escalation.triggered (HIGH/CRITICAL priority) run on a priority path that never waits behind safety.assessment.completed monitoring (RISK_ALERT), which is coalesced per user per window.

Episodes (crisis_alerts, slot 015): alert_id UUID PK, user_id, opened_at, max_level, status open|acknowledged|resolved, correlation_ids UUID[], sources TEXT[], page_count, last_paged_at, ack_token_hash, acknowledged_by, acknowledged_at, resolved_at. An event with an already-seen correlation_id at the same or a lower level, or for the same user within NOTIFICATION_ALERT_COALESCE_SECONDS (900) at the same or a lower level, folds into the open episode without paging. A strictly higher level always escalates: max_level is raised and the episode pages again, including for a seen correlation_id (a later, higher layer for the same turn, crisis-review F8) and for an acknowledged episode (crisis-review F9: the primary tier, the acknowledging clinician and, for CRITICAL, the non-backup on-call clinicians). Once an episode is acknowledged, further detections at the same or a lower level within the window send one update email to the acknowledging clinician instead of a new page cascade (D-06).

Recipients: the care team (IF-08); if none, the on-call roster; if none, NOTIFICATION_FALLBACK_ONCALL_EMAIL. Delivery is per recipient. A new page is delivered once and the handler returns; duplicates never deliver inline. If no clinician delivery succeeds, the fallback mailbox gets the page and the page stays pending: the pager-owned sweep retry_pending_pages() (task crisis_page_retry, every 15 s) retries it until a clinician gets it or the episode is resolved, skipping recipients already delivered for the episode, with delivery_attempts as the per-recipient idempotency record. A pass with no clinician recipient counts as outcome failed, so CrisisAlertDeliveryFailed fires while the page is pending. Only unexpected exceptions reach the event plane's retry (crisis-review F1). Every attempt is persisted to notifications and delivery_attempts (channel, template id, recipient hash, outcome; never the body).

Acknowledgement: a one-time link {NOTIFICATION_ACK_BASE_URL}/api/v1/notifications/crisis-alerts/ack/{token} (24 h, only the hash stored; GET shows a confirm form, POST acknowledges), routed by Caddy; POST /api/v1/notifications/crisis-alerts/{alert_id}/acknowledge and /resolve for CLINICIAN or ADMIN (MFA from W4). Re-page: a background task (registered with the runtime) re-pages the backup on-call and the fallback mailbox for open episodes unacknowledged after NOTIFICATION_REPAGE_AFTER_SECONDS (900), at most NOTIFICATION_MAX_REPAGES (3) times.

Email: subject '[Solace CRISIS <LEVEL>] action required - ref <first 8 chars of alert_id>'; plain-text body with level, categories, detection sources, detected_at UTC, the user's display name and contact only when crisis_contact_consent is true (D-12), alert id and the acknowledgement link; Date and Message-ID headers; text not HTML-escaped; never message text or diagnosis. CRITICAL also sends a compact SMS when SMS_ENABLED.

REST: POST /api/v1/notifications/send and /email require SERVICE tokens with service:write:notification and return 202 only when delivery is accepted, else 502 {error:{code:'DELIVERY_FAILED'}}.

Metrics: solace_crisis_alert_delivery_total{outcome,channel}, solace_crisis_alert_latency_seconds (detection to first successful delivery), solace_crisis_alerts_unacknowledged.

### IF-10: Orchestrator turn runner

**Owner:** WS-10. **Consumers:** WS-11.

In services/orchestrator_service/src/langgraph/graph_builder.py:
- classmethod OrchestratorGraphBuilder.from_runtime(runtime: ServiceRuntime, settings: GraphBuilderSettings | None = None) -> OrchestratorGraphBuilder: main.py's only construction call from W1.
- async def run_turn(self, *, user_id: str, thread_id: str, session_id: str, message: str, turn_id: UUID, channel: Literal['rest','ws']) -> TurnResult. Holds a per-thread asyncio lock; enforces ORCHESTRATOR_TURN_TIMEOUT_SECONDS (20); runs the IF-07 screen and the safety-service /check (2 s, fused with fuse_risk) before the graph; never raises for graph, agent or LLM failure. On failure it returns resources if the fused level is ELEVATED or higher (and publishes the crisis event for HIGH/CRITICAL), otherwise a safe reply with degraded populated. It publishes crisis events itself, at most once per turn, before returning.
- @dataclass TurnResult: response_text: str; crisis: CrisisInfo | None ({level, categories, resources, resources_version}); degraded: list[str] (e.g. 'safety_service_timeout', 'llm_unavailable', 'therapy_unavailable'); agents: list[str]; turn_id: UUID; safety_flags: dict (this turn only); crisis_watch_until: datetime | None (D-06).
- async def delete_threads(self, thread_ids: Sequence[str]) -> int: deletes checkpoints for erasure (W2).
- Checkpointer (W2): AsyncPostgresSaver on a psycopg AsyncConnectionPool for solace_app over the slot-017 tables, with a serializer whose payloads are encrypted by the IF-13 codec, plus TTL cleanup; InMemorySaver only outside strict envs; a failure in strict envs is fatal at boot.
- No user credential is ever stored in OrchestratorState.

### IF-11: Orchestrator-to-service REST contracts

**Owner:** WS-11. **Consumers:** WS-05, WS-12, WS-15, WS-16, WS-17, WS-18.

Provider schemas are canonical; the orchestrator adapts. Every call carries an IF-04 obo token (safety and personality also accept it as a service token) plus X-Request-ID, X-Correlation-ID and X-Turn-ID, and has a per-call timeout of ORCHESTRATOR_AGENT_TIMEOUT_SECONDS (8) inside the 20 s turn deadline. A POST is retried only when the provider honours an Idempotency-Key header, and never on 4xx.

Calls:
- safety: POST /api/v1/safety/check (IF-06).
- therapy: POST /api/v1/therapy/sessions/start once per thread, then POST /api/v1/therapy/sessions/{session_id}/message.
- diagnosis: POST /api/v1/diagnosis/session/start, POST /api/v1/diagnosis/assess, POST /api/v1/diagnosis/session/end.
- personality: POST /api/v1/personality/detect with sources including 'ensemble' when the LLM is wanted; style via the existing style route.
- memory: POST /api/v1/memory/context to read; POST /api/v1/memory/session/message to write each user and assistant turn.
- therapy to memory uses forwarded_auth_headers (IF-04).

Enums are exchanged as the provider's string values. A non-success domain result is non-2xx: therapy SESSION_NOT_FOUND 404, USER_ID_MISMATCH 403, TREATMENT_PLAN_NOT_FOUND 409. Diagnosis hypotheses are never shown to users (D-30).

Contracts are JSON files under tests/contracts/orchestrator__<provider>/ (and therapy__memory/), verified by providers via IF-20.

### IF-12: Client chat API and WebSocket protocol

**Owner:** WS-11. **Consumers:** WS-03, WS-05.

REST POST /api/v1/orchestrator/chat with a user ACCESS token:
- Request {message: str 1..4000, thread_id?: str, session_id?: str}; any body user_id is ignored (identity comes from the token).
- thread_id must belong to the caller (orchestrator_threads, slot 016, from W2), else 404 THREAD_NOT_FOUND; new threads are bound on first use.
- Response {response, thread_id, session_id, turn_id, crisis: IF-06 object | null, degraded: [str], agents: [str], request_id}.
- Errors {error:{code, message, request_id}}; 403 {error:{code:'CONSENT_REQUIRED', message, request_id}, crisis: IF-06 object | null} when the 'cns' claim is not ok (W3). Every refusal of a received message (consent, expired token, rate limit) first runs the orchestrator's crisis_for_refused_message (local screen only): the crisis object is returned or sent, and HIGH/CRITICAL pages (crisis-review F3).

WebSocket /api/v1/orchestrator/ws/{session_id}:
- Token via Sec-WebSocket-Protocol (client offers 'solace.v1' and 'bearer.<jwt>'; server selects 'solace.v1') or the deprecated ?token= (access logs off), per D-27.
- Client frames: {type:'chat.message', content, client_message_id?, thread_id?} | {type:'ping'} | {type:'auth', token} (re-auth before expiry). The legacy {type:'message', message} is accepted as an alias.
- Server frames: {type:'connected', session_id, thread_id, connection_id}, always the first server frame after accept ("safety.alert first" means first after the client's message, crisis-review F5); {type:'safety.alert', level, categories, resources, resources_version} sent before the reply when crisis is set; {type:'chat.response', content, thread_id, turn_id, client_message_id, degraded, agents}; {type:'pong'}; on a refused frame (AUTH_EXPIRED, RATE_LIMITED) a crisis message still gets safety.alert before the error or close; {type:'error', code: 'UNSUPPORTED_FRAME'|'INVALID_FRAME'|'RATE_LIMITED'|'AUTH_EXPIRED'|'INTERNAL', message}.
- Close codes: 4401 expired, revoked or deleted user; 4403 foreign thread; 4429 rate limited. Token exp and revocation are checked on every frame (W3); at most ORCHESTRATOR_WS_MAX_PER_USER (3) sockets per user; 8 KB per frame; 30 messages per minute per user.

/batch: at most 10 items, concurrency 2, no exception text. GET /api/v1/orchestrator/sessions/{session_id}/history?thread_id= returns only the caller's thread.

docs/API-HANDOFF.md documents exactly this; contract files live in tests/contracts/client__orchestrator/.

### IF-13: PHI field codec

**Owner:** WS-09. **Consumers:** WS-04, WS-08, WS-10, WS-11, WS-12, WS-14, WS-15, WS-16, WS-17, WS-18.

solace_security.phi_codec (new):
- def get_phi_codec() -> PHICodec: process singleton from EncryptionSettings; in strict envs raises PHIConfigError on first call when keys are missing or invalid (the bootstrap calls it at startup).
- PHICodec.encrypt(value: str | None, *, field: str, owner_id: str | UUID) -> str | None returns 'enc:v1:<key_id>:<base64url(nonce||ciphertext||tag)>'; AES-256-GCM with AAD = f'{field}:{owner_id}', so ciphertext cannot be moved between rows or users.
- PHICodec.decrypt(value: str | None, *, field: str, owner_id: str | UUID) -> str | None; legacy plaintext accepted only in development/test.
- PHICodec.search_hash(value: str, *, purpose: str) -> str: HMAC-SHA256 with ENCRYPTION_SEARCH_HASH_SALT, hex, for equality lookups (e.g. users.email_hash).
- encrypt_fields(row: Mapping[str, Any], fields: Iterable[str], *, owner_id) -> dict and decrypt_fields(row, fields, *, owner_id) -> dict. Each service declares its field set in its repository module; there is no central registry.
- active_key_id -> str, written to encryption_key_id columns.

Keys: data key = HKDF-SHA256(master, info=b'solace-phi:' + key_id), derived once per key id (no per-call PBKDF2). ENCRYPTION_MASTER_KEY accepts 64 hex characters (the documented recipe), 'base64:<32 bytes>', or a legacy exactly-32-character value (development/test only). ENCRYPTION_KEY_ID names the active key; ENCRYPTION_RETIRED_KEYS_JSON {key_id: key} is decrypt-only. The ORM-only configure_phi_encryption listeners are retired.

Columns, adopted in the owning lanes in W2: users email/display_name/phone/bio/tokens (WS-14, blind index for email); escalation reason/context and safety_assessments content (WS-12); diagnosis records and Redis blobs (WS-16); therapy transcripts, notes and plan text (WS-15); memory_records.content, session_summaries, user_facts (WS-17); personality evidence (WS-18); orchestrator checkpoints (WS-10); TOTP secrets (WS-08, W4).

### IF-14: Audit trail API

**Owner:** WS-09. **Consumers:** WS-04, WS-08, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

solace_security.audit_runtime (new):
- async def configure_service_audit(service_name: str, pool: asyncpg.Pool) -> None (called by the bootstrap when spec.audited).
- async def audit(event_type: AuditEventType, *, actor_id: str, actor_type: Literal['user','service','system'], action: str, outcome: AuditOutcome, resource_type: str, resource_id: str | None = None, subject_user_id: str | None = None, details: Mapping[str, str | int | bool] | None = None) -> None. details keys are checked against an allow-list and never hold PHI. Failures never raise to callers; in strict envs a failure is retried 3 times, then logged CRITICAL and counted in solace_audit_write_failures_total.
- def audit_phi_access(resource_type: str, *, subject: Callable[[Request], str | None] | None = None) -> Callable: FastAPI dependency that records actor, subject user id, route template, method and status after the response, never bodies.

Store: audit_events (slot 011); solace_app has INSERT and SELECT only, with UPDATE/DELETE revoked and blocked by rules. Chain: one global chain serialized by pg_advisory_xact_lock(hashtext('solace_audit_chain')) inside the insert transaction, with chain_seq BIGSERIAL; event_hash = HMAC-SHA256(AUDIT_HMAC_KEY, canonical_json(row fields incl. writer_service and a sha256 of details) || previous_hash). verify_chain(start_seq: int | None = None, end_seq: int | None = None) -> ChainVerification recomputes every HMAC and link; scripts run nightly and set solace_audit_chain_verify_ok.

New AuditEventType values: AUTH_LOGIN_SUCCEEDED, AUTH_LOGIN_FAILED, AUTH_LOCKOUT, AUTH_LOGOUT, AUTH_TOKEN_REFRESHED, AUTH_TOKEN_REVOKED, AUTH_PASSWORD_CHANGED, AUTH_PASSWORD_RESET, AUTH_MFA_ENROLLED, AUTH_MFA_FAILED, AUTH_OAUTH_LOGIN, ACCOUNT_DELETED, ACCOUNT_ERASED_IN_SERVICE, PHI_READ, PHI_WRITE, PHI_DELETE, CRISIS_ALERT_SENT, CRISIS_ALERT_FAILED, CRISIS_ALERT_ACKNOWLEDGED, ADMIN_CLINICIAN_PROVISIONED, ADMIN_ONCALL_CHANGED.

### IF-15: LLM gateway client

**Owner:** WS-09. **Consumers:** WS-05, WS-10, WS-12, WS-15, WS-16, WS-18.

services/shared/infrastructure/llm_client.py (existing LLM_ prefix):
- UnifiedLLMClient.generate(...) raises LLMUnavailableError (new, same module) on provider, auth or timeout failure and never returns '' or '(LLM unavailable)'. New keyword pinned_context: list[dict] | None, never trimmed; history is trimmed oldest-first and every leading system message is kept; roles are restricted to system, user and assistant.
- generate_with_fallback(...) -> LLMResult(text: str, degraded: bool, provider: str).
- initialize() never raises for missing credentials outside strict envs (is_available is False instead). In strict envs: LLM_PHI_EGRESS_APPROVED=false makes is_available False and /ready reports a non-critical 'llm: disabled'; approved with missing credentials is a startup error.
- Settings: LLM_PORTKEY_API_KEY, LLM_PORTKEY_GATEWAY_URL (the harness's fake LLM), LLM_ANTHROPIC_API_KEY, LLM_TIMEOUT_SECONDS, LLM_ENABLE_CACHING (default false; if ever enabled the cache namespace includes user_id and semantic cache is never used for PHI-bearing prompts, D-04), LLM_PHI_EGRESS_APPROVED.
- The fake LLM (infrastructure/test/fake-llm) is OpenAI/Portkey-compatible POST /v1/chat/completions with deterministic replies, POST /__control/scenario {mode: 'ok'|'error'|'timeout'|'crisis_intent'|'echo', match?} and GET /__control/requests (recorded prompts for cross-user assertions).

### IF-16: User erasure fan-out

**Owner:** WS-14. **Consumers:** WS-05, WS-06, WS-08, WS-10, WS-11, WS-12, WS-13, WS-15, WS-16, WS-17, WS-18.

Flow:
1. DELETE /api/v1/users/me (api_auth.py, WS-08) requires header X-Reauth-Token (IF-17) and calls UserService.delete_user(user_id, reason) -> UUID erasure_request_id.
2. Local erasure runs through UserDataErasure (idempotent, retry-safe); the users row is soft-deleted and anonymized last.
3. In the same transaction as the erasure_requests row (slot 012), UserDeletedKafkaEvent(user_id, reason, metadata.correlation_id = erasure_request_id) is published on solace.users through publish(conn=).
4. Each PHI service's group solace-group-erasure-<service> runs its UserDataErasure under acting_as_user(user_id), purges in-process caches, writes ACCOUNT_ERASED_IN_SERVICE (IF-14) and publishes user.erasure.completed; ErasureIncomplete becomes RetryableEventError.
5. user-service records completions per service; erasure is complete when every expected service has reported; GET /api/v1/admin/erasure-requests/{request_id} (ADMIN) shows per-service status.

Helper src/solace_infrastructure/gdpr/consumer.py: def register_user_erasure_consumer(plane: ServiceEventPlane, *, service_name: str, erasure: UserDataErasure, purge_caches: Callable[[UUID], Awaitable[None]] | None = None) -> None.

Expected services and stores: orchestrator (threads, checkpoints via delete_threads), safety (assessments, history; escalations pseudonymized), therapy, diagnosis, memory (Postgres, Redis, Weaviate when enabled, BM25), personality (with tombstone), notification (delivery records erased, crisis episodes pseudonymized), event tables via erase_user_events (WS-06). Audit rows hold no PHI and are retained (D-24).

### IF-17: Identity, session and MFA API

**Owner:** WS-08. **Consumers:** WS-03, WS-05, WS-07, WS-11, WS-13, WS-14.

Access-token claims: sub, type='access', roles, sid (session id), amr (['pwd'] | ['pwd','otp'] | ['google'] | ['google','otp']), mfa: bool, cns (IF-08), iss 'solace-ai', aud 'solace-ai-api', iat, exp (15 min). Refresh tokens carry sid and jti (30 days). Refresh requires the sid to be active; rotation is atomic (Redis Lua swap) with a 30 s grace for an exact replay of the same jti; reuse after the grace revokes the session; token type is checked before the blacklist; status and role are re-read; session activity is extended. Logout or password change revokes the sid and its refresh tokens in the shared blacklist.

Endpoints (prefix /api/v1/auth, api_auth.py):
- POST /register: always 202-shaped, no email-exists oracle (an existing owner gets an email).
- POST /login: tokens, or {mfa_required: true, mfa_token} (5 min) when MFA is enrolled, or {mfa_enrollment_required: true, enrollment_token} for a CLINICIAN/ADMIN without a factor.
- POST /mfa/verify {mfa_token, code | recovery_code}; POST /mfa/enroll/start -> {otpauth_uri, secret} shown once; POST /mfa/enroll/confirm {code} -> {recovery_codes: [10]} shown once. TOTP is RFC 6238 (30 s, 6 digits, +-1 step) and rejects replay of the last used step.
- POST /password/forgot {email}: always 202. POST /password/reset {token, new_password}: 30 min, single use, hashed at rest; revokes all sessions.
- POST /reauth {password | code} -> {reauth_token} (5 min, type='reauth').
- POST /verify-email {token}: expiring, hashed token; strict envs refuse login for unverified accounts.
- GET /oauth/google/start?redirect_uri=<allow-listed in OAUTH_ALLOWED_REDIRECT_ORIGINS> sets an HttpOnly SameSite=Lax state cookie; GET /oauth/google/callback validates state against the cookie, links only by the verified Google sub in oauth_accounts (an existing password account with the same email links only after proving its password) and 302-redirects with a one-time 60 s code; POST /oauth/google/exchange {code} -> token pair. Google endpoints come from GOOGLE_OAUTH_* so the stub IdP can stand in.

Account routes in api_auth.py: POST /api/v1/users/me/password (current password verified before hashing, throttled); DELETE /api/v1/users/me (requires X-Reauth-Token).

CLINICIAN and ADMIN tokens without mfa=true get 403 on PHI, admin and acknowledgement endpoints via require_mfa_for_roles (IF-04, W4). JWKS at GET /.well-known/jwks.json once RS256 is on.

### IF-18: Observability: health, metrics and alerts

**Owner:** WS-04. **Consumers:** WS-03, WS-05, WS-06, WS-11, WS-12, WS-13.

Every service exposes /live, /ready, /health (= /ready) and /metrics on its own port (IF-01). Docker HEALTHCHECK, compose depends_on health conditions, autoheal and Caddy active health checks use /live, never /ready, so a dependency outage never cuts users off from in-band crisis resources; /ready drives readiness tests and alerts.

Standard metrics (prometheus_client; labels never hold user ids or text): solace_http_requests_total{service,route,method,status}, solace_http_request_duration_seconds{service,route}, solace_readiness_check_ok{service,check}, solace_background_task_up{service,task}, solace_crisis_detected_total{source,level}, solace_llm_fallback_total{service,reason}, solace_audit_write_failures_total{service}, solace_audit_chain_verify_ok, solace_backup_last_success_timestamp_seconds{store}, plus the IF-05 and IF-09 metrics.

Prometheus scrapes <compose-service>:<port>/metrics for the 9 MVP services. Rules in infrastructure/prometheus/rules/solace.yml, promtool-tested:
- W1: CrisisAlertDeliveryFailed (any failed outcome in 5 m), CrisisOutboxBacklog (oldest pending > 120 s), NotificationConsumerLag (> 10 for 5 m), DLQUnresolved (> 0), CrisisAlertsUnacknowledged (> 0 for 20 m), CrisisEventPublishFailures (> 0), ServiceNotReady (2 m).
- W4: ContainerUnhealthy, HostDiskUsage (> 80%), BackupStale (> 26 h), AuditWriteFailures (> 0), AuditChainVerifyFailed.
Alertmanager (docker-compose.observability.yml) routes to ALERTMANAGER_EMAIL_TO through the SMTP relay (mailpit in test). Autoheal restarts unhealthy containers (W4).

### IF-19: Real-infra and E2E harness

**Owner:** WS-05. **Consumers:** WS-01, WS-02, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18.

Markers (registered by WS-01): realinfra, e2e, contract, clinical. SOLACE_REQUIRE_REALINFRA=1 turns skips into failures at barriers.

src/solace_testing/realinfra.py session fixtures (testcontainers):
- pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool): postgres:16 pinned, migrated to head by scripts.db.migrate as solace_migrator, accessed as solace_app with RLS;
- redis_url: redis:7 with a password;
- kafka_bootstrap: cp-kafka 7.5 with aiokafka helpers produce(topic, event) and consume_until(topic, predicate, timeout);
- smtp_sink -> SmtpSink(api_url).wait_for(to, subject_contains, timeout=60) on mailpit;
- fake_llm -> base_url.
Component tests live in tests/realinfra/<ws-slug>/.

tests/e2e/conftest.py:
- e2e_stack (session) -> Stack(urls: dict[service, base_url], edge_url, project, mailpit_api, fake_llm_control, pg_superuser_dsn), built with 'docker compose -p solace-e2e-<lane> -f docker-compose.yml [-f docker-compose.prod.yml] -f docker-compose.test.yml' on Docker-assigned 127.0.0.1 ports; client traffic goes through the edge Caddy so TRUSTED_PROXY_HOPS=1 is exercised.
- Helpers: make_user(stack, role='user', verified=True) -> TestUser{id, email, password, access_token} (register, then the mailpit verification link); provision_clinician(stack, email, phone, on_call=True, backup=False) via scripts.provision_clinician; fault.stop/start/restart/pause(service); read_logs(service); kafka_lag(group) via the broker CLI in the kafka container; llm.scenario(mode, match=None) and llm.requests(); oidc stub (W3).
- Env: SOLACE_E2E_PROFILE in {dev, strict}; SOLACE_E2E_ENVIRONMENT in {staging, production}; SOLACE_E2E_KEEP=1 keeps the stack.

docker-compose.test.yml adds fake-llm, mailpit (axllent/mailpit pinned by digest) and a stub OIDC provider; generates per-run test secrets; sets EMAIL_SMTP_HOST=mailpit, EMAIL_SMTP_TLS_MODE=none, LLM_PORTKEY_GATEWAY_URL=http://fake-llm:8080/v1, LLM_PHI_EGRESS_APPROVED=true (stub only), NOTIFICATION_FALLBACK_ONCALL_EMAIL=fallback-oncall@test.invalid, http-only CADDY_*_SITE values and ephemeral loopback ports. It never weakens strict settings otherwise.

Manifest tests/e2e/expected_failures.yaml lists {test_id, expected_failing_hop, target_wave}; the plugin applies strict xfail and fails the barrier when a listed test passes early or fails at a different hop.

### IF-20: Contract-test framework

**Owner:** WS-05. **Consumers:** WS-06, WS-11, WS-12, WS-13, WS-15, WS-16, WS-17, WS-18.

solace_testing.contracts:
- Contract files are JSON under tests/contracts/<consumer>__<provider>/<interaction>.json: {consumer, provider, interaction, request: {method, path, headers_required: [...], body_example}, response: {status, body_schema (JSON Schema)}}.
- verify_provider(app: FastAPI, contracts_dir: Path, *, auth_headers: Callable[[str], dict]) runs in the provider's suite against the real app through httpx ASGITransport.
- assert_request_matches(contract, request) runs in consumer suites.
- Event contracts {event_type, example} are validated against solace_events.schemas.EVENT_REGISTRY for every producer/consumer pair (CrisisDetectedEvent, EscalationTriggeredEvent, SafetyAssessmentEvent, UserDeletedKafkaEvent, UserErasureCompletedKafkaEvent).
- The consumer workstream writes the contract file; the provider workstream makes verify_provider pass. Marker contract.

### IF-21: Build, image and test tooling

**Owner:** WS-01. **Consumers:** WS-02, WS-03, WS-04, WS-05, WS-06, WS-07, WS-09.

Dependencies: requirements.in (direct runtime deps) compiled by 'uv pip compile --generate-hashes --python-version 3.12' into requirements.lock; requirements-dev.in into requirements-dev.lock (pytest, pytest-asyncio, pytest-timeout, hypothesis, testcontainers[postgres,redis,kafka], aiokafka, websockets); services/<svc>/requirements.txt hold service extras only.

Images: build context is the repo root; FROM python:3.12-slim@sha256:<pinned>; 'uv pip sync requirements.lock' into /opt/venv with no fallback; non-root uid 1000; WORKDIR /app; PYTHONPATH=/app:/app/src.
- Each image's CMD imports the same module path its test suite imports. Underscore services run 'python -m uvicorn services.<pkg>.src.main:app'; user-service and notification-service merge their src into /app/src and run 'src.main:app' (notification's package-relative imports require it); analytics-service keeps its current top-level layout unless its owner moves it.
- ENTRYPOINT ['/app/docker/entrypoint.sh']: for every NAME_FILE variable it exports NAME from the file (one trailing newline stripped; a missing file is fatal), then exec "$@".
- uvicorn runs with --no-access-log --proxy-headers; HEALTHCHECK probes /live.
- docker/migrate.Dockerfile builds solace-migrate carrying migrations/, alembic.ini and scripts/db.

Import roots: shared packages are imported only as top-level solace_* (never src.solace_*), enforced by tests/build/test_import_roots.py; src/solace_infrastructure/__init__.py and infrastructure/api_gateway/__init__.py export lazily.

Makefile targets: lint (ruff E9,F63,F7,F82), test-unit (one pytest process per suite), test-realinfra, test-e2e PROFILE=dev|strict, test-clinical, test-contract, images, clean-checkout TREE=<tree id> (build from a git archive), ci-local (all of the above). .github/workflows/ci.yml mirrors them.

### IF-22: Secrets and key-material contract

**Owner:** WS-03. **Consumers:** WS-01, WS-02, WS-04, WS-05, WS-07, WS-08, WS-09.

Every strict-environment secret comes from the per-environment secrets store (D-07): AWS Secrets Manager names solace/<env>/<NAME>, fetched at deploy time by deploy/render_secrets.py (host instance role) into a root-owned tmpfs at /run/secrets/<name> and referenced by NAME_FILE, which docker/entrypoint.sh loads.

Secret names: AUTH_SECRET_KEY, AUTH_PRIVATE_KEY_PEM, SERVICE_TOKEN_PRIVATE_KEY_PEM (one per service, W4), ENCRYPTION_MASTER_KEY, ENCRYPTION_SEARCH_HASH_SALT, ENCRYPTION_RETIRED_KEYS_JSON, AUDIT_HMAC_KEY, FERNET_TOKEN_KEY, FERNET_FIELD_KEY, POSTGRES_PASSWORD (solace_app), POSTGRES_MIGRATOR_PASSWORD, POSTGRES_SUPERUSER_PASSWORD, REDIS_PASSWORD, REDIS_URL, WEAVIATE_API_KEY, LLM_PORTKEY_API_KEY, LLM_ANTHROPIC_API_KEY, EMAIL_SMTP_PASSWORD, SMS_AUTH_TOKEN, GOOGLE_OAUTH_CLIENT_SECRET, GRAFANA_ADMIN_PASSWORD, BASIC_AUTH_HASH, BACKUP_ENCRYPTION_KEY, and the Postgres TLS files postgres-ca.crt, server.crt, server.key.

Rules:
- Each container receives only the secrets its service reads; Caddy receives only DOMAIN, CADDY_ADMIN_EMAIL and BASIC_AUTH_HASH.
- Dev and test values are generated per developer or per run by deploy/env/generate_dev_env.py (writes files with restrictive ACLs, prints names only, refuses to overwrite without --rotate); ProductionGuards rejects known dev values and low-entropy values in strict envs.
- Key material (ENCRYPTION_*, AUDIT_HMAC_KEY, AUTH_PRIVATE_KEY_PEM, SERVICE_TOKEN_PRIVATE_KEY_PEM, BACKUP_ENCRYPTION_KEY) is versioned in the store with deletion protection and escrow; backups record key ids, never keys.
- No secret is ever in the repo, an image layer, a compose literal, a log line, a test report or tool output.

## Shared files

| File | Owner | Needed by | Protocol |
|---|---|---|---|
| `docker-compose.yml` | WS-03 | WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | WS-03 is the single writer. Other workstreams file env-registry or compose requests (requester, wave, variable or service block, value per profile) in their plans; WS-03 lands each wave's requests in a phase-A env-sync task. Requests raised after a wave starts go to the next phase A or to a ledger Ruling. New containers go in fragment files (docker-compose.observability.yml WS-04, docker-compose.backup.yml WS-03, docker-compose.test.yml WS-05); a fragment may add services and override environment, volumes and ports but never redefines a base service's build or image. tests/deploy/test_env_contract.py must stay green. |
| `docker-compose.prod.yml` | WS-03 | WS-04, WS-05, WS-07, WS-09, WS-11, WS-12, WS-13, WS-14 | Same request protocol as docker-compose.yml. The strict overlay uses 'ports: !reset []' for internal services, derives every *_ENVIRONMENT from ENVIRONMENT, holds no dev defaults, from W3 injects secrets only per IF-22 (NAME_FILE pointing at /run/secrets) and never gives Caddy an env_file holding app secrets. The test overlay never weakens it. |
| `.env.example` | WS-03 | WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Rendered from deploy/env/env-contract.yaml by deploy/env/render_env_example.py; hand edits are forbidden. It holds names and safe placeholders only, never a working secret. New variables are requested as registry entries. |
| `Caddyfile` | WS-03 | WS-04, WS-05, WS-08, WS-11, WS-13 | Route, health and header changes are requested via IF-12 (orchestrator REST and WS upgrade), IF-17 (auth vhost, OAuth callback), IF-09 (notification acknowledgement path) or IF-18 (health on /live). WS-03 lands them in phase A. /metrics, /docs, /redoc, /openapi.json, /status and /api/v1/internal/* are never routed; no access log includes query strings; site addresses come from {$CADDY_API_SITE} and {$CADDY_AUTH_SITE} so the test overlay reuses this file. |
| `docker-compose.test.yml` | WS-05 | WS-03, WS-06, WS-08, WS-12, WS-13, WS-14 | Only WS-05 edits it. It overlays base (dev) or base + prod (strict) with fake-llm, mailpit, the stub OIDC provider, generated per-run secrets, http-only Caddy site addresses and ephemeral loopback ports. Other workstreams request test-only services or env through WS-05 tasks; it never weakens strict settings, which test_env_contract.py checks on the rendered strict + test config. |
| `docker-compose.observability.yml` | WS-04 | WS-03, WS-05, WS-06, WS-13 | Only WS-04 edits it (Alertmanager from W1, autoheal and log shipping from W4). It adds services and attaches to existing networks but never redefines a base service; metric producers request scrape jobs and alert rules from WS-04. |
| `docker-compose.backup.yml` | WS-03 | WS-02, WS-04, WS-05 | Only WS-03 edits it (W4): backup and WAL-archiving jobs, restore tooling and the drill target. WS-02 supplies the role and archive_command requirements, WS-04 the backup metric, WS-05 the drill driver. |
| `deploy/env/env-contract.yaml` | WS-03 | WS-01, WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | The machine-readable IF-02 registry. A workstream that adds or renames a setting requests an entry (name, services, secret, required_in_strict, dev_default) in the same wave; WS-03 adds it in phase A and test_env_contract.py enforces it. Retired names are marked retired and must not be read by any code. |
| `services/orchestrator_service/src/main.py` | WS-11 | WS-04, WS-06, WS-07, WS-09, WS-10 | Only the owner edits it. The W0 phase-B adoption task replaces the logging, guards, PHI-encryption, pool, limiter and bridge boilerplate with IF-01 and declares the full ServiceRuntimeSpec up front (event_plane='publish_consume', uses_postgres, uses_redis, handles_phi, audited, rate_limited, edge). After W0, cross-cutting behaviour is added only in bootstrap.py; WS-10 needs (from_runtime construction in W1) are one-line call-site changes made in the WS-11 lane. |
| `services/safety_service/src/main.py` | WS-12 | WS-04, WS-06, WS-07, WS-09 | Same main.py protocol: one W0 phase-B IF-01 adoption task by the owner declaring the full SPEC; afterwards cross-cutting changes go only through bootstrap.py. |
| `services/notification-service/src/main.py` | WS-13 | WS-01, WS-04, WS-06, WS-07 | Same main.py protocol. The W0 adoption task also removes the relative-import and F821 defects, after WS-01's merged image layout lands. |
| `services/user-service/src/main.py` | WS-14 | WS-04, WS-06, WS-07, WS-08, WS-09 | Same main.py protocol. The W0 adoption task moves identity composition verbatim into identity_wiring.py (start_identity/stop_identity, owned by WS-08 afterwards) and includes the api_auth, api and internal_api routers once. The local rate limiter is removed in W1 once WS-04 wires install_rate_limiting. WS-08 never edits main.py. |
| `services/therapy_service/src/main.py` | WS-15 | WS-04, WS-06, WS-07, WS-09 | Same main.py protocol: single W0 phase-B adoption by the owner; afterwards cross-cutting changes go only through bootstrap.py. |
| `services/diagnosis_service/src/main.py` | WS-16 | WS-04, WS-06, WS-07, WS-09 | Same main.py protocol: single W0 phase-B adoption by the owner; afterwards cross-cutting changes go only through bootstrap.py. |
| `services/memory_service/src/main.py` | WS-17 | WS-04, WS-07, WS-09 | Same main.py protocol: single W0 phase-B adoption by the owner; afterwards cross-cutting changes go only through bootstrap.py. |
| `services/personality_service/src/main.py` | WS-18 | WS-04, WS-06, WS-07, WS-09 | Same main.py protocol: single W0 phase-B adoption by the owner; afterwards cross-cutting changes go only through bootstrap.py. |
| `services/analytics-service/src/main.py` | WS-18 | WS-01, WS-04, WS-06, WS-07, WS-09 | Same main.py protocol. The W0 adoption task also fixes the F821 forward references that block the CI lint gate. |
| `services/config_service/src/main.py` | WS-03 |  | Frozen. config-service is removed from the MVP compose and the CI image matrix (D-19); no workstream edits its code unless the owner reverses D-19. |
| `src/solace_infrastructure/runtime/bootstrap.py` | WS-04 | WS-02, WS-06, WS-07, WS-09, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Component wiring lives only here. The component owner delivers its library function per its IF in phase A (WS-06 start_service_event_plane, WS-07 ServiceTokenManager and install_rate_limiting, WS-09 get_phi_codec and configure_service_audit, WS-02 assert_schema_head); WS-04 then wires it in a later phase-A task of the same wave, gated by a ServiceRuntimeSpec flag that services already declared in W0, so no main.py changes. Each wiring task runs the boot smoke in dev and strict. |
| `src/solace_security/middleware.py` | WS-07 | WS-08, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Public dependency names and signatures are stable: get_current_user, get_current_user_optional, get_current_service, require_roles, require_permissions, require_service_permission, AuthenticatedUser, AuthenticatedService. WS-07 makes only additive changes (W1 obo acceptance and via_service, W3 subject scoping, W4 require_mfa_for_roles). No other workstream edits it; needs are requested as IF-04 revisions recorded as ledger Rulings. |
| `migrations/versions/*` | WS-02 | WS-06, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | In W0 phase A WS-02 repairs 001/007/env.py, writes 009 and creates no-op slot files 010-021 with fixed revision ids on one linear chain (IF-03). Each slot file is then owned by its workstream, which edits only upgrade() and downgrade(); revision and down_revision never change. Slot tasks are phase-A tasks and run one at a time in a per-wave schema lane in slot order, each keeping tests/migrations/test_chain_real_pg.py green; WS-02 reviews each slot diff. Slot bodies may be amended in place until the first real deployment; after it, new revisions are requested from WS-02 and appended at head. |
| `requirements.txt` | WS-01 | WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Same single-writer protocol as requirements.lock: requests (package, exact version, reason, images, wave) land in phase A of the requesting wave; WS-01 regenerates the lock with uv for Python 3.12 and rebuilds images. |
| `requirements.lock` | WS-01 | WS-02, WS-04, WS-05, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | WS-01 is the only writer of requirements.in, requirements.lock, requirements-dev.*, requirements.txt, requirements-runtime.txt and pyproject.toml. A request names the package, exact version, reason, affected images and wave; WS-01 lands it in phase A, regenerates the lock with uv for Python 3.12 and rebuilds images. Local installs beyond the D-20 set need owner approval. |
| `services/*/requirements.txt` | WS-01 | WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Each holds only service-specific extras that resolve with requirements.lock (no conflicting exact pins). Edited only by WS-01, on request. |
| `services/*/Dockerfile` | WS-01 | WS-02, WS-03, WS-05, WS-13 | Only WS-01 edits Dockerfiles, using the IF-21 layout; docker/migrate.Dockerfile and docker/entrypoint.sh are WS-01's too. Other workstreams request changes. |
| `services/user-service/src/api_auth.py` | WS-08 | WS-14 | Created by WS-14 in W0 phase B as a verbatim move of /auth/*, POST /users/me/password, DELETE /users/me and the JWKS route, exposing one auth_router that later aggregates the oauth and MFA routers, so main.py never changes for identity. Ownership passes to WS-08 at the W0 barrier; WS-14 never edits it afterwards. |
| `services/user-service/src/api_deps.py` | WS-08 | WS-14 | Created by WS-14 in W0 as a verbatim move of the dependency providers, get_current_user and the shared response models (UserResponse, user_to_response). Owned by WS-08 after the W0 barrier; additive only; WS-14 requests new dependencies (e.g. the service-token dependency for internal routes) through IF-08/IF-04 and WS-08 lands them before the dependent WS-14 task. |
| `services/user-service/src/identity_wiring.py` | WS-08 | WS-14 | Created by WS-14 in W0 from main.py's JWT, token, session, OAuth-state and key-manager composition; main.py calls start_identity(runtime, app) and stop_identity() once. Owned by WS-08 after the barrier. |
| `services/user-service/src/api.py` | WS-14 | WS-08 | After the W0 split it holds only profile, preferences, consent, user lookup and admin routes. WS-08 never edits it. |
| `services/user-service/src/domain/entities.py` | WS-08 | WS-14 | User entity auth state (lockout, verification, can_login) is WS-08's. WS-14 requests additive profile fields (e.g. is_backup; phone_number and is_on_call already exist) via IF-08; WS-08 lands them before the dependent WS-14 task, recorded in the ledger. |
| `services/user-service/src/domain/service.py` | WS-14 | WS-08 | UserService is WS-14's. WS-08 calls change_password, verify_email and delete_user and requests changes, which WS-14 lands before the dependent WS-08 task. New identity logic goes in WS-08-owned modules (mfa.py, password_reset.py). |
| `services/*/src/infrastructure/event_bridge.py` | WS-06 | WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-18 | All seven bridge files belong to WS-06 and are changed only in phase A. initialize_event_bridge / initialize_notification_bridge keep their names and keyword arguments so main.py call sites never change. Service workstreams change only their own events.py (domain events and to_kafka_event mapping). |
| `src/solace_events/schemas.py` | WS-06 | WS-10, WS-12, WS-13, WS-14, WS-16, WS-18 | Schema additions are made only by WS-06 in phase A. Existing fields are never removed or renamed during the MVP. Producers never put raw user text in any field (IF-06). |
| `services/shared/infrastructure/llm_client.py` | WS-09 | WS-10, WS-11, WS-12, WS-15, WS-16, WS-18 | Behaviour changes (W0 init, W1 LLMUnavailableError/LLMResult, W2 pinned context) land in phase A; consumers adapt call sites in their own files in phase B of the same or next wave. Public method names stay stable. |
| `src/solace_security/__init__.py` | WS-07 | WS-04, WS-09 | No new eager re-exports; new modules (phi_codec, audit_runtime) are imported by full path. |
| `src/solace_infrastructure/__init__.py` | WS-01 | WS-02, WS-04, WS-06, WS-14 | Lazy (PEP 562) exports only, never new eager imports. Consumers import submodules by full path (e.g. solace_infrastructure.runtime). |
| `src/solace_security/production_guards.py` | WS-04 | WS-03, WS-07, WS-09 | WS-04 fixes the guard names and staging strictness in W0. In W3 WS-03 requests secret-source checks and WS-09 requests AUDIT_HMAC_KEY; WS-04 adds them in phase A. |
| `src/solace_common/request_context.py` | WS-07 | WS-02, WS-06, WS-14, WS-15, WS-16, WS-17, WS-18 | get_current_user_id/set_current_user_id semantics are unchanged; acting_as_user(user_id) is added in W0 phase A1. connection_manager (WS-02) keeps reading the same contextvar. |
| `src/solace_common/enums.py` | WS-07 | WS-10, WS-12, WS-15, WS-16 | CrisisLevel and SeverityLevel values are frozen: no renames or removals. Additions only on request, in phase A. |
| `pytest.ini` | WS-01 | WS-02, WS-05, WS-12 | Markers realinfra, e2e, contract and clinical are registered in W0; the conflicting pyproject pytest table is removed. Other changes by request only. |
| `.github/workflows/ci.yml` | WS-01 | WS-02, WS-05 | CI mirrors the Makefile targets (IF-21). New jobs are added by WS-01 on request; because nothing is pushed, every workflow change is proven with 'make ci-local' in the same task. The e2e job runs on demand or nightly. |
| `infrastructure/prometheus/prometheus.yml` | WS-04 | WS-06, WS-13, WS-03 | Only WS-04 edits scrape config and infrastructure/prometheus/rules/*.yml; metric producers request jobs and rules, which ship with promtool tests. |
| `services/orchestrator_service/src/langgraph/graph_builder.py` | WS-10 | WS-11 | Graph topology, run_turn, from_runtime, delete_threads and the checkpointer are WS-10's. WS-11 needs (memory node registration, agent node signatures) become WS-10 tasks requested through IF-10. |
| `services/orchestrator_service/src/langgraph/state_schema.py` | WS-10 | WS-11 | WS-10 owns state fields and reducers, which stay turn-scoped; WS-11 requests new fields as WS-10 tasks. No credential or token is ever a state field. |
| `docs/API-HANDOFF.md` | WS-11 | WS-08, WS-03 | Index plus the orchestrator contract (IF-12), rewritten at the W2 barrier. The auth contract lives in docs/api/auth.md (WS-08) and is linked, not copied. |
| `docs/CLINICAL-VALIDATION.md` | WS-12 | WS-10, WS-13, WS-15, WS-16, WS-17 | WS-12 records the clinical sign-off for the lexicon version, thresholds, fusion rule, crisis watch and episode semantics with the phrase-set version (D-05, D-06). Other lanes' vignette sign-offs are added as supporting tasks in the WS-12 lane. |
| `services/orchestrator_service/tests` | WS-11 | WS-10 | Shared test dir. WS-11 stewards conftest.py. An existing test file is edited only by the workstream that owns the code under test; new files are named test_mvp_<id>_<slug>.py. |
| `services/user-service/tests` | WS-14 | WS-08 | Shared test dir. WS-14 stewards conftest.py. An existing test file is edited only by the owner of the code under test; new files are named test_mvp_<id>_<slug>.py. |
| `infrastructure/api_gateway/tests` | WS-01 | WS-07 | WS-01 stewards it; WS-07 adds new files test_mvp_<id>_<slug>.py for rate_limiting.py and edits existing rate-limiter tests only in its own tasks. |
| `tests/integration` | WS-05 | WS-04, WS-06, WS-07, WS-09, WS-10, WS-11, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | Shared test dir. New files are named test_mvp_<id>_<slug>.py; an existing file is edited only by the owner of the code it tests (e.g. test_diagnosis_vignette.py by WS-16). tests/solace_security, tests/solace_infrastructure, tests/solace_common and tests/security follow the same rule. |
| `tests/e2e` | WS-05 | WS-08, WS-10, WS-11, WS-12, WS-13, WS-14 | WS-05 owns conftest.py, the fakes, the wave acceptance suites and tests/e2e/expected_failures.yaml. Other workstreams may add files named test_mvp_<id>_<slug>.py that use only IF-19 fixtures. A task that makes a listed journey pass reports it, and WS-05 removes the manifest entry; a listed test that passes early or fails at a different hop blocks the barrier until explained. |
| `tests/realinfra` | WS-05 | WS-02, WS-06, WS-07, WS-08, WS-09, WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18 | WS-05 owns tests/realinfra/conftest.py. Each workstream writes component tests in tests/realinfra/<ws-slug>/ named test_mvp_<id>_<slug>.py, using only IF-19 fixtures. |

## Owned paths

### WS-01: Build, dependency lock, images, import roots and CI

`requirements.txt`, `requirements-runtime.txt`, `requirements.in`, `requirements.lock`, `requirements-dev.in`, `requirements-dev.lock`, `pyproject.toml`, `pytest.ini`, `Makefile`, `.dockerignore`, `.gitignore`, `.pre-commit-config.yaml`, `.github`, `docker`, `scripts/ci`, `__init__.py`, `src/__init__.py`, `src/py.typed`, `src/solace_infrastructure/__init__.py`, `src/solace_infrastructure/py.typed`, `services/__init__.py`, `infrastructure/__init__.py`, `infrastructure/api_gateway/__init__.py`, `infrastructure/api_gateway/kong_config.py`, `infrastructure/api_gateway/routes.py`, `infrastructure/api_gateway/auth_plugin.py`, `infrastructure/api_gateway/cors.py`, `infrastructure/api_gateway/tests`, `tests/build`, `tests/test_syntax_and_imports.py`, `services/orchestrator_service/Dockerfile`, `services/orchestrator_service/requirements.txt`, `services/safety_service/Dockerfile`, `services/safety_service/requirements.txt`, `services/therapy_service/Dockerfile`, `services/therapy_service/requirements.txt`, `services/diagnosis_service/Dockerfile`, `services/diagnosis_service/requirements.txt`, `services/memory_service/Dockerfile`, `services/memory_service/requirements.txt`, `services/personality_service/Dockerfile`, `services/personality_service/requirements.txt`, `services/user-service/Dockerfile`, `services/user-service/requirements.txt`, `services/notification-service/Dockerfile`, `services/notification-service/requirements.txt`, `services/analytics-service/Dockerfile`, `services/analytics-service/requirements.txt`

Produces: IF-21. Consumes: IF-02, IF-19, IF-22.

### WS-02: Schema of record, database roles, TLS and connection layer

`migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/001_initial_schema.py`, `migrations/versions/002_enable_rls_clinical_tables.py`, `migrations/versions/003_safety_escalations_table.py`, `migrations/versions/004_oauth_accounts.py`, `migrations/versions/005_widen_encrypted_phi_columns.py`, `migrations/versions/006_extend_rls_phi_tables.py`, `migrations/versions/007_outbox_claimed_at.py`, `migrations/versions/008_rls_personality_tables.py`, `migrations/versions/009_platform_roles_rls.py`, `alembic.ini`, `infrastructure/init-db`, `src/solace_infrastructure/database`, `src/solace_infrastructure/postgres.py`, `src/solace_infrastructure/redis.py`, `src/solace_infrastructure/weaviate.py`, `src/solace_infrastructure/feature_flags.py`, `scripts/db`, `tests/migrations`

Produces: IF-03. Consumes: IF-02, IF-19, IF-21, IF-22.

### WS-03: Deploy topology, env contract, edge, secrets, backups and host

`docker-compose.yml`, `docker-compose.prod.yml`, `docker-compose.backup.yml`, `.env.example`, `Caddyfile`, `deploy`, `infrastructure/aws-staging`, `services/config_service`, `tests/deploy`, `tests/config_service`, `README.md`, `system-design`, `docs/STAGING-BRINGUP.md`, `docs/SYSTEM-DESIGN-SUMMARY.md`, `docs/KNOWN-LIMITATIONS.md`, `docs/runbooks/deploy.md`, `docs/runbooks/secrets.md`, `docs/runbooks/backup-restore.md`

Produces: IF-02, IF-22. Consumes: IF-01, IF-03, IF-09, IF-12, IF-17, IF-18, IF-21.

### WS-04: Service runtime bootstrap, health, metrics and alerting

`src/solace_infrastructure/runtime`, `src/solace_infrastructure/health.py`, `src/solace_infrastructure/observability_core.py`, `src/solace_infrastructure/observability`, `src/solace_security/production_guards.py`, `infrastructure/prometheus`, `infrastructure/grafana`, `infrastructure/alertmanager`, `docker-compose.observability.yml`, `tests/bootstrap`, `docs/runbooks/alerting.md`

Produces: IF-01, IF-18. Consumes: IF-02, IF-03, IF-04, IF-05, IF-09, IF-13, IF-14, IF-21, IF-22.

### WS-05: Real-infra harness, contract framework and wave acceptance suites

`src/solace_testing`, `tests/e2e`, `tests/contracts`, `tests/realinfra/conftest.py`, `tests/conftest.py`, `tests/alignment`, `tests/system`, `tests/solace_testing`, `tests/scripts`, `docker-compose.test.yml`, `scripts/seed_demo_data.py`, `scripts/test`, `infrastructure/test`, `docs/release`

Produces: IF-19, IF-20. Consumes: IF-01, IF-02, IF-03, IF-05, IF-06, IF-07, IF-08, IF-09, IF-11, IF-12, IF-15, IF-16, IF-17, IF-18, IF-21, IF-22.

### WS-06: Durable event plane: outbox, pollers, consumer retry, DLQ and lag signals

`src/solace_events`, `src/solace_infrastructure/kafka`, `services/orchestrator_service/src/infrastructure/event_bridge.py`, `services/safety_service/src/infrastructure/event_bridge.py`, `services/therapy_service/src/infrastructure/event_bridge.py`, `services/diagnosis_service/src/infrastructure/event_bridge.py`, `services/personality_service/src/infrastructure/event_bridge.py`, `services/user-service/src/infrastructure/event_bridge.py`, `services/notification-service/src/infrastructure/event_bridge.py`, `migrations/versions/010_event_plane.py`, `tests/solace_events`

Produces: IF-05. Consumes: IF-02, IF-03, IF-16, IF-18, IF-19, IF-20, IF-21.

### WS-07: Inter-service credentials, authorization, revocation and rate limiting

`src/solace_security/auth.py`, `src/solace_security/middleware.py`, `src/solace_security/service_auth.py`, `src/solace_security/keys.py`, `src/solace_security/rate_limit.py`, `src/solace_security/authorization.py`, `src/solace_security/__init__.py`, `src/solace_security/py.typed`, `src/solace_common`, `infrastructure/api_gateway/rate_limiting.py`, `scripts/generate_jwt_keys.py`

Produces: IF-04. Consumes: IF-02, IF-03, IF-17, IF-19, IF-21, IF-22.

### WS-08: Identity: login, sessions, OAuth, re-authentication, MFA and password reset

`services/user-service/src/auth.py`, `services/user-service/src/oauth.py`, `services/user-service/src/api_auth.py`, `services/user-service/src/api_deps.py`, `services/user-service/src/identity_wiring.py`, `services/user-service/src/mfa.py`, `services/user-service/src/password_reset.py`, `services/user-service/src/domain/entities.py`, `services/user-service/src/infrastructure/jwt_service.py`, `services/user-service/src/infrastructure/password_service.py`, `services/user-service/src/infrastructure/token_service.py`, `services/user-service/src/infrastructure/oauth_google.py`, `services/user-service/src/infrastructure/oauth_account_repository.py`, `services/user-service/src/infrastructure/mfa_repository.py`, `migrations/versions/013_identity_mfa.py`, `docs/api/auth.md`

Produces: IF-17. Consumes: IF-02, IF-03, IF-04, IF-08, IF-09, IF-13, IF-14, IF-16, IF-19, IF-22.

### WS-09: PHI protection: field codec, audit trail, log sanitizer and LLM gateway client

`src/solace_security/encryption.py`, `src/solace_security/phi_protection.py`, `src/solace_security/audit.py`, `src/solace_security/phi_codec.py`, `src/solace_security/audit_runtime.py`, `services/shared`, `migrations/versions/011_audit_trail.py`

Produces: IF-13, IF-14, IF-15. Consumes: IF-02, IF-03, IF-19, IF-21, IF-22.

### WS-10: Orchestrator graph, crisis ingress and turn runner

`services/orchestrator_service/src/langgraph/graph_builder.py`, `services/orchestrator_service/src/langgraph/supervisor.py`, `services/orchestrator_service/src/langgraph/state_schema.py`, `services/orchestrator_service/src/langgraph/aggregator.py`, `services/orchestrator_service/src/langgraph/__init__.py`, `services/orchestrator_service/src/langgraph/crisis_screen.py`, `services/orchestrator_service/src/langgraph/checkpointer.py`, `services/orchestrator_service/src/agents/safety_agent.py`, `services/orchestrator_service/src/agents/chat_agent.py`, `services/orchestrator_service/src/agents/emotion_agent.py`, `services/orchestrator_service/src/events.py`, `services/orchestrator_service/src/response`, `migrations/versions/017_orchestrator_checkpoints.py`

Produces: IF-10. Consumes: IF-02, IF-03, IF-04, IF-05, IF-06, IF-07, IF-13, IF-15, IF-16, IF-19.

### WS-11: Orchestrator edge: REST/WS API, agent clients and provider contracts

`services/orchestrator_service/src/api.py`, `services/orchestrator_service/src/main.py`, `services/orchestrator_service/src/config.py`, `services/orchestrator_service/src/websocket.py`, `services/orchestrator_service/src/replay_buffer.py`, `services/orchestrator_service/src/__init__.py`, `services/orchestrator_service/src/infrastructure/clients.py`, `services/orchestrator_service/src/infrastructure/state.py`, `services/orchestrator_service/src/infrastructure/__init__.py`, `services/orchestrator_service/src/infrastructure/thread_registry.py`, `services/orchestrator_service/src/agents/therapy_agent.py`, `services/orchestrator_service/src/agents/diagnosis_agent.py`, `services/orchestrator_service/src/agents/personality_agent.py`, `services/orchestrator_service/src/agents/assessment_agent.py`, `services/orchestrator_service/src/agents/__init__.py`, `services/orchestrator_service/src/langgraph/memory_node.py`, `services/orchestrator_service/tests`, `migrations/versions/016_orchestrator_threads.py`, `docs/API-HANDOFF.md`, `docs/api/orchestrator.md`

Produces: IF-11, IF-12. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-06, IF-10, IF-13, IF-14, IF-16, IF-17, IF-18, IF-19, IF-20.

### WS-12: Safety service detection, escalation records and the shared crisis lexicon

`services/safety_service/src/domain`, `services/safety_service/src/ml`, `services/safety_service/src/db`, `services/safety_service/src/observability`, `services/safety_service/src/api.py`, `services/safety_service/src/config.py`, `services/safety_service/src/events.py`, `services/safety_service/src/main.py`, `services/safety_service/src/__init__.py`, `services/safety_service/src/infrastructure/clinician_registry.py`, `services/safety_service/src/infrastructure/database.py`, `services/safety_service/src/infrastructure/gdpr.py`, `services/safety_service/src/infrastructure/repository.py`, `services/safety_service/src/infrastructure/telemetry.py`, `services/safety_service/src/infrastructure/__init__.py`, `services/safety_service/config`, `services/safety_service/db`, `services/safety_service/tests`, `services/safety_service/README.md`, `services/safety_service/__init__.py`, `src/solace_clinical`, `tests/clinical`, `migrations/versions/014_safety.py`, `docs/CLINICAL-VALIDATION.md`

Produces: IF-06, IF-07. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-11, IF-13, IF-14, IF-15, IF-16, IF-18, IF-19, IF-20.

### WS-13: Notification: clinician alert delivery, crisis episodes and acknowledgement

`services/notification-service/src/domain`, `services/notification-service/src/api.py`, `services/notification-service/src/consumers.py`, `services/notification-service/src/events.py`, `services/notification-service/src/main.py`, `services/notification-service/src/__init__.py`, `services/notification-service/src/infrastructure/__init__.py`, `services/notification-service/src/infrastructure/delivery_repository.py`, `services/notification-service/src/infrastructure/crisis_alert_repository.py`, `services/notification-service/tests`, `migrations/versions/015_notification.py`, `docs/runbooks/crisis-escalation.md`

Produces: IF-09. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-06, IF-08, IF-14, IF-16, IF-17, IF-18, IF-19, IF-20.

### WS-14: User accounts, clinician roster, consent and erasure fan-out

`services/user-service/src/main.py`, `services/user-service/src/config.py`, `services/user-service/src/api.py`, `services/user-service/src/internal_api.py`, `services/user-service/src/events.py`, `services/user-service/src/__init__.py`, `services/user-service/src/domain/service.py`, `services/user-service/src/domain/consent.py`, `services/user-service/src/domain/erasure.py`, `services/user-service/src/domain/value_objects.py`, `services/user-service/src/domain/__init__.py`, `services/user-service/src/infrastructure/postgres_repository.py`, `services/user-service/src/infrastructure/repository.py`, `services/user-service/src/infrastructure/encryption_service.py`, `services/user-service/src/infrastructure/__init__.py`, `services/user-service/README.md`, `services/user-service/tests`, `src/solace_infrastructure/gdpr`, `scripts/provision_clinician.py`, `migrations/versions/012_user_accounts.py`, `docs/runbooks/clinician-onboarding.md`

Produces: IF-08, IF-16. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-09, IF-13, IF-14, IF-17, IF-19.

### WS-15: Therapy service: crisis signals, persistence, contracts and erasure

`services/therapy_service/src/domain`, `services/therapy_service/src/api.py`, `services/therapy_service/src/config.py`, `services/therapy_service/src/events.py`, `services/therapy_service/src/main.py`, `services/therapy_service/src/schemas.py`, `services/therapy_service/src/__init__.py`, `services/therapy_service/src/infrastructure/context_assembler.py`, `services/therapy_service/src/infrastructure/gdpr.py`, `services/therapy_service/src/infrastructure/postgres_repository.py`, `services/therapy_service/src/infrastructure/redis_session_repository.py`, `services/therapy_service/src/infrastructure/repository.py`, `services/therapy_service/src/infrastructure/__init__.py`, `services/therapy_service/tests`, `services/therapy_service/__init__.py`, `migrations/versions/019_therapy.py`

Produces: none. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-06, IF-07, IF-11, IF-13, IF-14, IF-15, IF-16, IF-19, IF-20.

### WS-16: Diagnosis service: risk signals, phase safety, persistence and clinical logic

`services/diagnosis_service/src/domain`, `services/diagnosis_service/src/api.py`, `services/diagnosis_service/src/config.py`, `services/diagnosis_service/src/events.py`, `services/diagnosis_service/src/main.py`, `services/diagnosis_service/src/schemas.py`, `services/diagnosis_service/src/__init__.py`, `services/diagnosis_service/src/infrastructure/postgres_repository.py`, `services/diagnosis_service/src/infrastructure/redis_session_repository.py`, `services/diagnosis_service/src/infrastructure/repository.py`, `services/diagnosis_service/src/infrastructure/__init__.py`, `services/diagnosis_service/tests`, `services/diagnosis_service/__init__.py`, `migrations/versions/020_diagnosis.py`

Produces: none. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-06, IF-07, IF-11, IF-13, IF-14, IF-15, IF-16, IF-19, IF-20.

### WS-17: Memory service: isolation, durable storage, context assembly and erasure

`services/memory_service/src`, `services/memory_service/tests`, `services/memory_service/__init__.py`, `migrations/versions/018_memory.py`

Produces: none. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-07, IF-11, IF-13, IF-14, IF-16, IF-19, IF-20.

### WS-18: Personality and analytics services

`services/personality_service/src/domain`, `services/personality_service/src/ml`, `services/personality_service/src/api.py`, `services/personality_service/src/config.py`, `services/personality_service/src/events.py`, `services/personality_service/src/main.py`, `services/personality_service/src/schemas.py`, `services/personality_service/src/__init__.py`, `services/personality_service/src/infrastructure/postgres_repository.py`, `services/personality_service/src/infrastructure/repository.py`, `services/personality_service/src/infrastructure/__init__.py`, `services/personality_service/tests`, `services/analytics-service/src`, `services/analytics-service/tests`, `migrations/versions/021_personality_analytics.py`

Produces: none. Consumes: IF-01, IF-02, IF-03, IF-04, IF-05, IF-11, IF-13, IF-14, IF-15, IF-16, IF-19, IF-20.
