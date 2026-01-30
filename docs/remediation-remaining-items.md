# Solace-AI Remediation - Remaining Items & Recommendations

**Date:** January 2026

---

## Outstanding Manual Tasks

### 1. Delete Archive Directory (High Priority)

The `archive/` directory contains ~118 legacy files (~51,500 LOC) from the old monolith. No active code imports from it.

```bash
git rm -r archive/
git commit -m "Remove legacy archive directory (~51,500 LOC unused monolith code)"
```

### 2. Clean Up Dependencies (Medium Priority)

Remove unused packages from `requirements.txt`:
- `chromadb` — Not used (Weaviate is the vector database)
- `qdrant-client` — Not used
- `faiss-cpu` — Not used

Fix Python version inconsistency:
- `requirements.txt` says Python 3.12+
- `pyproject.toml` says Python >=3.11
- **Recommendation:** Standardize on 3.12+

### 3. S608 Lint Suppression (Low Priority)

8 `S608` (Possible SQL injection) warnings in `src/solace_infrastructure/postgres.py` are false positives. The table names are class-level constants validated in `__init__()`, not user input.

Add inline suppression:
```python
query = f"SELECT * FROM {self.qualified_table} WHERE id = $1"  # noqa: S608
```

Or add to `pyproject.toml`:
```toml
[tool.ruff.lint.per-file-ignores]
"src/solace_infrastructure/postgres.py" = ["S608"]
```

---

## Recommended Follow-Up Work

### Security Enhancements

1. **Rate Limiting:** Add rate limiting middleware to all public endpoints (currently only configured, not enforced)
2. **CORS Configuration:** Review and restrict CORS origins in all service `api.py` files
3. **Content Security Policy:** Add CSP headers for any web-facing endpoints
4. **Secret Rotation Automation:** Implement automated key rotation schedule for encryption keys
5. **Penetration Testing:** Run a formal penetration test against the deployed services
6. **HIPAA Compliance Audit:** Engage a compliance specialist to verify the audit store, encryption, and access control meet HIPAA requirements

### Infrastructure Improvements

1. **Database Migrations:** Run `alembic upgrade head` against a test database to verify the initial schema migration works
2. **Connection Pool Tuning:** Benchmark PostgreSQL connection pool sizes under load (currently uses asyncpg defaults)
3. **Redis Sentinel/Cluster:** Consider Redis Sentinel or Cluster for high availability in production
4. **Weaviate Backup:** Configure automated vector database backups
5. **Service Mesh:** Consider Istio or Linkerd for mTLS between services
6. **Centralized Logging:** Set up ELK/Grafana Loki for aggregated structured logging

### Testing Gaps

1. **Integration Tests:** The security test suite tests individual components — add end-to-end integration tests that verify auth flow across service boundaries
2. **Load Testing:** Add k6 or Locust load tests for critical paths (session creation, message sending, crisis detection)
3. **Contract Tests:** Wire up the fixed `ProviderContractTest` to run against each service's actual API
4. **Chaos Testing:** Test circuit breaker behavior when downstream services are unavailable
5. **PHI Leak Tests:** Add tests that verify no PHI appears in logs, error messages, or API responses

### Code Quality

1. **mypy Strict Mode:** Run `mypy --strict` across the full codebase and fix type errors
2. **Test Coverage:** Target >80% coverage (currently unknown — run `pytest --cov` to baseline)
3. **API Documentation:** Add OpenAPI schema validation tests to ensure API docs match implementation
4. **Dependency Scanning:** Add `pip-audit` or `safety` to CI pipeline for vulnerability scanning
5. **Pre-commit Hooks:** Add `.pre-commit-config.yaml` with ruff, mypy, and bandit

---

## Architecture Observations

### Strengths
- Clean hexagonal architecture (domain/infrastructure separation) in all services
- Comprehensive domain exception hierarchy with structured error codes
- Factory pattern for backend-swappable persistence
- JSONB storage for complex nested entities (good balance of relational + document patterns)
- Structured logging via `structlog` throughout

### Areas for Improvement
- **Event Sourcing:** The therapy service's treatment plan versioning would benefit from event sourcing rather than UPSERT-with-version-counter
- **CQRS:** Read-heavy endpoints (session history, outcome trends) could use materialized views or read replicas
- **API Gateway:** Services currently handle auth individually — an API gateway (Kong, Ambassador) would centralize auth, rate limiting, and routing
- **Message Queue Reliability:** Kafka consumers need dead-letter queue handling and idempotent processing
- **Configuration Management:** Environment variables scattered across services — consider a centralized config service (Consul, etcd) or sealed secrets

---

## Critical Files Reference

| File | Phases Modified | Security Impact |
|------|----------------|-----------------|
| `src/solace_infrastructure/postgres.py` | 1, 6, 10 | SQL injection fix, retry logic, context manager |
| `src/solace_security/auth.py` | 2, 3 | Token revocation, Redis sessions, lockout, rotation |
| `src/solace_security/encryption.py` | 3 | Silent failures, salt fix, key rotation |
| `src/solace_security/service_auth.py` | 2, 10 | Privilege escalation fix, thread safety |
| `src/solace_security/middleware.py` | 2, 4 | RBAC resolution, auth enforcement |
| `src/solace_security/authorization.py` | 10 | DELETE permission fix |
| `src/solace_security/audit.py` | 3 | PostgreSQL audit store (HIPAA) |
| `services/shared/infrastructure/llm_client.py` | 5 | Portkey gateway, PHI safety |
| `src/solace_infrastructure/observability_core.py` | 6 | Thread safety, memory leak |
| `src/solace_infrastructure/database/base_models.py` | 6 | FK constraints, audit table |
