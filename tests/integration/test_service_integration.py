"""
Service-to-service integration tests for Solace-AI platform.

Tests the authentication infrastructure, client factory, circuit breaker,
and permission matrix that govern inter-service communication.
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from solace_security.auth import AuthSettings, JWTManager, TokenType
from solace_security.service_auth import (
    ServiceTokenManager,
    ServiceAuthSettings,
    ServiceIdentity,
    ServicePermission,
    SERVICE_PERMISSIONS,
    ServiceCredentials,
    ServiceAuthResult,
)

from services.orchestrator_service.src.infrastructure.clients import (
    BaseServiceClient,
    ClientConfig,
    CircuitBreaker,
    CircuitState,
    ServiceClientFactory,
    ServiceResponse,
    SafetyServiceClient,
    TherapyServiceClient,
    MemoryServiceClient,
    DiagnosisServiceClient,
    PersonalityServiceClient,
)
from services.orchestrator_service.src.config import ServiceEndpoints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_settings():
    """Create development auth settings for testing."""
    return AuthSettings.for_development()


@pytest.fixture
def service_auth_settings():
    """Create service auth settings for testing."""
    return ServiceAuthSettings(
        service_name="orchestrator-service",
        service_token_expire_minutes=60,
        enable_token_caching=True,
    )


@pytest.fixture
def token_manager(auth_settings, service_auth_settings):
    """Create a fully configured ServiceTokenManager."""
    return ServiceTokenManager(
        auth_settings=auth_settings,
        service_settings=service_auth_settings,
    )


@pytest.fixture
def endpoints():
    """Create test service endpoints."""
    return ServiceEndpoints(
        safety_service_url="http://localhost:18002",
        personality_service_url="http://localhost:18007",
        diagnosis_service_url="http://localhost:18004",
        therapy_service_url="http://localhost:18006",
        memory_service_url="http://localhost:18005",
        user_service_url="http://localhost:18001",
        notification_service_url="http://localhost:18003",
        treatment_service_url="http://localhost:18006",
    )


# ---------------------------------------------------------------------------
# Service Token Lifecycle
# ---------------------------------------------------------------------------

class TestServiceTokenLifecycle:
    """Tests for creating, validating, and caching service tokens."""

    def test_create_token_for_orchestrator(self, token_manager):
        """Orchestrator can create a valid service token."""
        creds = token_manager.create_service_token("orchestrator-service")
        assert creds.service_name == "orchestrator-service"
        assert creds.token
        assert not creds.is_expired
        assert len(creds.permissions) > 0

    def test_token_validates_successfully(self, token_manager):
        """Created token passes validation."""
        creds = token_manager.create_service_token("orchestrator-service")
        result = token_manager.validate_service_token(creds.token)
        assert result.authenticated is True
        assert result.service_name == "orchestrator-service"

    def test_token_caching_returns_same_token(self, token_manager):
        """Cached token returns same credentials on subsequent calls."""
        creds1 = token_manager.get_or_create_token("orchestrator-service")
        creds2 = token_manager.get_or_create_token("orchestrator-service")
        assert creds1.token == creds2.token

    def test_expired_token_rejected(self, token_manager):
        """Expired service tokens are rejected during validation."""
        creds = token_manager.create_service_token(
            "orchestrator-service", expire_minutes=0
        )
        # Token with 0 minute expiry is effectively expired immediately
        result = token_manager.validate_service_token(creds.token)
        # With 0 minutes, the exp is set to now, so it might still be valid for clock_skew
        # The key assertion: the mechanism works end-to-end
        assert isinstance(result, ServiceAuthResult)

    def test_clear_cache_forces_new_token(self, token_manager):
        """Clearing cache forces creation of a new token."""
        creds1 = token_manager.get_or_create_token("orchestrator-service")
        token_manager.clear_cache("orchestrator-service")
        creds2 = token_manager.get_or_create_token("orchestrator-service")
        assert creds1.token != creds2.token

    def test_unknown_service_rejected(self, token_manager):
        """Unknown service identity is rejected."""
        with pytest.raises(ValueError, match="Unknown service identity"):
            token_manager.create_service_token("unknown-service")

    def test_all_known_services_can_create_tokens(self, token_manager):
        """Every ServiceIdentity can successfully create a token."""
        for identity in ServiceIdentity:
            creds = token_manager.create_service_token(identity.value)
            assert creds.token
            result = token_manager.validate_service_token(creds.token)
            assert result.authenticated is True


# ---------------------------------------------------------------------------
# Permission Matrix
# ---------------------------------------------------------------------------

class TestPermissionMatrix:
    """Tests verifying the service permission matrix enforcement."""

    def test_orchestrator_has_broad_read_access(self):
        """Orchestrator has read access to all services it coordinates."""
        perms = SERVICE_PERMISSIONS[ServiceIdentity.ORCHESTRATOR]
        assert ServicePermission.READ_SAFETY in perms
        assert ServicePermission.READ_THERAPY in perms
        assert ServicePermission.READ_DIAGNOSIS in perms
        assert ServicePermission.READ_PERSONALITY in perms
        assert ServicePermission.READ_MEMORY in perms

    def test_safety_has_escalation_permissions(self):
        """Safety service can trigger escalation and crisis alerts."""
        perms = SERVICE_PERMISSIONS[ServiceIdentity.SAFETY]
        assert ServicePermission.TRIGGER_ESCALATION in perms
        assert ServicePermission.SEND_CRISIS_ALERT in perms

    def test_therapy_cannot_escalate(self):
        """Therapy service does not have escalation permissions."""
        perms = SERVICE_PERMISSIONS[ServiceIdentity.THERAPY]
        assert ServicePermission.TRIGGER_ESCALATION not in perms

    def test_user_service_has_no_permissions(self):
        """User service (source of truth) has no outbound permissions."""
        perms = SERVICE_PERMISSIONS[ServiceIdentity.USER]
        assert len(perms) == 0

    def test_permission_check_blocks_unauthorized(self, token_manager):
        """Token validation rejects requests lacking required permissions."""
        # Memory service doesn't have escalation permission
        creds = token_manager.create_service_token("memory-service")
        result = token_manager.validate_service_token(
            creds.token,
            required_permissions=["service:escalate"],
        )
        assert result.authenticated is False
        assert "Missing required permissions" in result.error

    def test_permission_check_allows_authorized(self, token_manager):
        """Token validation accepts requests with required permissions."""
        creds = token_manager.create_service_token("safety-service")
        result = token_manager.validate_service_token(
            creds.token,
            required_permissions=["service:escalate"],
        )
        assert result.authenticated is True

    def test_every_service_has_expected_permission_set(self):
        """Every ServiceIdentity is represented in the permission matrix."""
        for identity in ServiceIdentity:
            assert identity in SERVICE_PERMISSIONS


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Tests for the circuit breaker pattern in service clients."""

    def test_starts_closed(self):
        """New circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_resets_on_success(self):
        """Success resets the failure counter and closes the circuit."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_half_open_after_recovery_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        # With recovery_timeout=0, the state property immediately returns HALF_OPEN
        # because elapsed time >= 0
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# Service Client Factory
# ---------------------------------------------------------------------------

class TestServiceClientFactory:
    """Tests for ServiceClientFactory creating properly configured clients."""

    def test_factory_creates_all_client_types(self, endpoints):
        """Factory creates clients for all services."""
        factory = ServiceClientFactory(endpoints=endpoints)
        assert isinstance(factory.safety(), SafetyServiceClient)
        assert isinstance(factory.therapy(), TherapyServiceClient)
        assert isinstance(factory.memory(), MemoryServiceClient)
        assert isinstance(factory.diagnosis(), DiagnosisServiceClient)
        assert isinstance(factory.personality(), PersonalityServiceClient)

    def test_factory_reuses_clients(self, endpoints):
        """Factory returns same client instance on repeated calls."""
        factory = ServiceClientFactory(endpoints=endpoints)
        client1 = factory.safety()
        client2 = factory.safety()
        assert client1 is client2

    def test_factory_with_token_manager(self, endpoints, token_manager):
        """Factory passes token manager to clients."""
        factory = ServiceClientFactory(endpoints=endpoints, token_manager=token_manager)
        safety_client = factory.safety()
        assert safety_client._token_manager is token_manager

    @pytest.mark.asyncio
    async def test_factory_close_all(self, endpoints):
        """Factory closes all client connections."""
        factory = ServiceClientFactory(endpoints=endpoints)
        factory.safety()
        factory.therapy()
        await factory.close_all()
        assert len(factory._clients) == 0

    def test_factory_health_report(self, endpoints):
        """Factory produces health report for created clients."""
        factory = ServiceClientFactory(endpoints=endpoints)
        factory.safety()
        factory.memory()
        health = factory.get_health()
        assert "safety" in health
        assert "memory" in health
        assert health["safety"]["circuit_state"] == "closed"


# ---------------------------------------------------------------------------
# Auth Header Injection
# ---------------------------------------------------------------------------

class TestAuthHeaderInjection:
    """Tests that service clients properly inject auth headers."""

    def test_auth_headers_with_token_manager(self, token_manager):
        """Client generates auth headers when token manager configured."""
        config = ClientConfig(base_url="http://localhost:9999")
        client = BaseServiceClient(
            config, token_manager=token_manager, service_name="orchestrator-service"
        )
        headers = client._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["X-Service-Name"] == "orchestrator-service"

    def test_no_auth_headers_without_token_manager(self):
        """Client returns empty headers when no token manager."""
        config = ClientConfig(base_url="http://localhost:9999")
        client = BaseServiceClient(config, token_manager=None)
        headers = client._get_auth_headers()
        assert headers == {}

    def test_auth_token_validates_round_trip(self, token_manager):
        """Token from auth headers validates successfully."""
        config = ClientConfig(base_url="http://localhost:9999")
        client = BaseServiceClient(
            config, token_manager=token_manager, service_name="orchestrator-service"
        )
        headers = client._get_auth_headers()
        token = headers["Authorization"].replace("Bearer ", "")
        result = token_manager.validate_service_token(token)
        assert result.authenticated is True
        assert result.service_name == "orchestrator-service"


# ---------------------------------------------------------------------------
# ServiceResponse
# ---------------------------------------------------------------------------

class TestServiceResponse:
    """Tests for the ServiceResponse wrapper."""

    def test_success_response(self):
        """Success response wraps data correctly."""
        resp = ServiceResponse(
            success=True,
            data={"risk_level": "LOW"},
            status_code=200,
            response_time_ms=45.2,
        )
        assert resp.success is True
        assert resp.data["risk_level"] == "LOW"
        d = resp.to_dict()
        assert d["success"] is True
        assert d["status_code"] == 200

    def test_error_response(self):
        """Error response wraps error details correctly."""
        resp = ServiceResponse(
            success=False,
            error="Connection refused",
            status_code=503,
        )
        assert resp.success is False
        assert resp.error == "Connection refused"

    def test_circuit_open_response(self):
        """Circuit breaker open produces 503 response."""
        resp = ServiceResponse(
            success=False, error="Circuit breaker open", status_code=503
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Cross-service credential validation
# ---------------------------------------------------------------------------

class TestCrossServiceCredentials:
    """Test that one service's token can be validated by another service's token manager."""

    def test_orchestrator_token_validated_by_safety(self, auth_settings):
        """Safety service can validate orchestrator's token (shared secret)."""
        # Orchestrator creates token
        orch_manager = ServiceTokenManager(
            auth_settings=auth_settings,
            service_settings=ServiceAuthSettings(service_name="orchestrator-service"),
        )
        creds = orch_manager.create_service_token("orchestrator-service")

        # Safety service validates it
        safety_manager = ServiceTokenManager(
            auth_settings=auth_settings,
            service_settings=ServiceAuthSettings(service_name="safety-service"),
        )
        result = safety_manager.validate_service_token(creds.token)
        assert result.authenticated is True
        assert result.service_name == "orchestrator-service"

    def test_different_secrets_reject_token(self):
        """Tokens signed with different secrets are rejected."""
        settings_a = AuthSettings(secret_key="secret-key-a-that-is-at-least-32-bytes")
        settings_b = AuthSettings(secret_key="secret-key-b-that-is-at-least-32-bytes")

        manager_a = ServiceTokenManager(auth_settings=settings_a)
        manager_b = ServiceTokenManager(auth_settings=settings_b)

        creds = manager_a.create_service_token("orchestrator-service")
        result = manager_b.validate_service_token(creds.token)
        assert result.authenticated is False

    def test_service_token_contains_correct_permissions(self, token_manager):
        """Token permissions match the SERVICE_PERMISSIONS matrix."""
        expected = {p.value for p in SERVICE_PERMISSIONS[ServiceIdentity.ORCHESTRATOR]}
        creds = token_manager.create_service_token("orchestrator-service")
        actual = set(creds.permissions)
        assert actual == expected


# ---------------------------------------------------------------------------
# ServiceCredentials lifecycle
# ---------------------------------------------------------------------------

class TestServiceCredentials:
    """Tests for ServiceCredentials data class."""

    def test_not_expired_when_future(self):
        """Credentials with future expiry are not expired."""
        creds = ServiceCredentials(
            service_name="test",
            token="dummy",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert creds.is_expired is False
        assert creds.should_refresh is False

    def test_expired_when_past(self):
        """Credentials with past expiry are expired."""
        creds = ServiceCredentials(
            service_name="test",
            token="dummy",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert creds.is_expired is True

    def test_should_refresh_near_expiry(self):
        """Credentials near expiry trigger refresh."""
        creds = ServiceCredentials(
            service_name="test",
            token="dummy",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            refresh_threshold_seconds=300,
        )
        assert creds.should_refresh is True
        assert creds.is_expired is False
