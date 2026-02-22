"""
Unit tests for Solace-AI Exception Hierarchy.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError

from solace_common.exceptions import (
    ApplicationError,
    AuthenticationError,
    AuthorizationError,
    BusinessRuleViolationError,
    CacheError,
    ConcurrencyError,
    ConfigurationError,
    DatabaseError,
    DomainError,
    EntityConflictError,
    EntityNotFoundError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    ExternalServiceError,
    InfrastructureError,
    InvariantViolationError,
    LLMServiceError,
    RateLimitExceededError,
    SafetyError,
    SolaceError,
    ValidationError,
)


class TestErrorContext:
    """Tests for ErrorContext."""

    def test_default_values(self) -> None:
        """Test default context values are set."""
        ctx = ErrorContext()

        assert ctx.correlation_id is not None
        assert len(ctx.correlation_id) == 36  # UUID length
        assert ctx.timestamp is not None
        assert ctx.service_name == "solace-ai"
        assert ctx.operation is None
        assert ctx.user_id is None

    def test_with_operation(self) -> None:
        """Test creating context with operation."""
        ctx = ErrorContext()
        new_ctx = ctx.with_operation("create_user")

        assert new_ctx.operation == "create_user"
        assert new_ctx.correlation_id == ctx.correlation_id

    def test_with_user(self) -> None:
        """Test creating context with user info."""
        ctx = ErrorContext()
        new_ctx = ctx.with_user("user-123", "session-456")

        assert new_ctx.user_id == "user-123"
        assert new_ctx.session_id == "session-456"
        assert new_ctx.correlation_id == ctx.correlation_id

    def test_immutability(self) -> None:
        """Test context is immutable."""
        ctx = ErrorContext()
        with pytest.raises((PydanticValidationError, TypeError, AttributeError)):
            ctx.operation = "test"  # type: ignore[misc]


class TestSolaceError:
    """Tests for base SolaceError."""

    def test_basic_creation(self) -> None:
        """Test basic error creation."""
        error = SolaceError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.user_message == "An error occurred. Please try again."
        assert error.context is not None
        assert error.error_code == "SOLACE_ERROR"

    def test_with_custom_user_message(self) -> None:
        """Test error with custom user message."""
        error = SolaceError("Internal message", user_message="User-friendly message")

        assert error.message == "Internal message"
        assert error.user_message == "User-friendly message"

    def test_with_context(self) -> None:
        """Test error with custom context."""
        ctx = ErrorContext(operation="test_op", user_id="user-123")
        error = SolaceError("Test error", context=ctx)

        assert error.context.operation == "test_op"
        assert error.context.user_id == "user-123"

    def test_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = SolaceError("Wrapped error", cause=cause)

        assert error.cause is cause

    def test_to_dict_safe_output(self) -> None:
        """Test to_dict returns safe external format."""
        error = SolaceError("Internal details", user_message="Safe message")
        result = error.to_dict()

        assert result["error"]["message"] == "Safe message"
        assert "Internal details" not in str(result)
        assert "correlation_id" in result["error"]

    def test_to_internal_dict(self) -> None:
        """Test to_internal_dict includes details."""
        error = SolaceError("Internal message", details={"key": "value"})
        result = error.to_internal_dict()

        assert result["internal"]["message"] == "Internal message"
        assert result["internal"]["details"]["key"] == "value"


class TestDomainErrors:
    """Tests for domain layer exceptions."""

    def test_validation_error(self) -> None:
        """Test ValidationError with field info."""
        error = ValidationError(
            "Email format invalid",
            field="email",
            value="not-an-email",
            constraint="email_format",
        )

        assert error.error_code == "VALIDATION_ERROR"
        assert error.http_status == 400
        assert error.field == "email"
        assert error.constraint == "email_format"
        assert "email" in error.user_message.lower()

    def test_entity_not_found_error(self) -> None:
        """Test EntityNotFoundError."""
        error = EntityNotFoundError("User", "user-123")

        assert error.error_code == "ENTITY_NOT_FOUND"
        assert error.http_status == 404
        assert error.entity_type == "User"
        assert error.entity_id == "user-123"
        assert "user" in error.user_message.lower()

    def test_concurrency_error(self) -> None:
        """Test ConcurrencyError."""
        error = ConcurrencyError(
            entity_type="Session",
            entity_id="session-123",
            expected_version=5,
            actual_version=7,
        )

        assert error.error_code == "CONCURRENCY_ERROR"
        assert error.http_status == 409
        assert error.expected_version == 5
        assert error.actual_version == 7

    def test_business_rule_violation(self) -> None:
        """Test BusinessRuleViolationError."""
        error = BusinessRuleViolationError(
            rule="max_sessions_per_user",
            message="User cannot have more than 5 active sessions",
        )

        assert error.error_code == "BUSINESS_RULE_VIOLATION"
        assert error.rule == "max_sessions_per_user"

    def test_invariant_violation(self) -> None:
        """Test InvariantViolationError severity."""
        error = InvariantViolationError("Aggregate invariant violated")

        assert error.severity == ErrorSeverity.HIGH


class TestApplicationErrors:
    """Tests for application layer exceptions."""

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        error = AuthenticationError()

        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.http_status == 401
        assert error.category == ErrorCategory.AUTHENTICATION

    def test_authorization_error(self) -> None:
        """Test AuthorizationError with permission."""
        error = AuthorizationError(
            "Access denied to resource",
            required_permission="admin:read",
        )

        assert error.error_code == "AUTHORIZATION_ERROR"
        assert error.http_status == 403
        assert "admin:read" in str(error.details)

    def test_rate_limit_error(self) -> None:
        """Test RateLimitExceededError."""
        error = RateLimitExceededError(
            limit=100,
            window_seconds=60,
            retry_after=30,
        )

        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.http_status == 429
        assert error.retry_after == 30

    def test_safety_error(self) -> None:
        """Test SafetyError is critical."""
        error = SafetyError("Crisis detected", safety_level=3)

        assert error.error_code == "SAFETY_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.SAFETY


class TestInfrastructureErrors:
    """Tests for infrastructure layer exceptions."""

    def test_database_error(self) -> None:
        """Test DatabaseError."""
        error = DatabaseError("Connection failed", operation="INSERT")

        assert error.error_code == "DATABASE_ERROR"
        assert error.http_status == 503
        assert error.category == ErrorCategory.DATABASE

    def test_cache_error(self) -> None:
        """Test CacheError severity."""
        error = CacheError("Redis connection lost")

        assert error.error_code == "CACHE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_external_service_error(self) -> None:
        """Test ExternalServiceError."""
        error = ExternalServiceError(
            service_name="payment-gateway",
            message="Service timeout",
            status_code=504,
        )

        assert error.service_name == "payment-gateway"
        assert error.details["upstream_status"] == 504

    def test_llm_service_error(self) -> None:
        """Test LLMServiceError."""
        error = LLMServiceError(
            provider="anthropic",
            message="Rate limit exceeded",
        )

        assert error.error_code == "LLM_SERVICE_ERROR"
        assert "anthropic" in error.details["provider"]

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Missing required config",
            config_key="DATABASE_URL",
        )

        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
