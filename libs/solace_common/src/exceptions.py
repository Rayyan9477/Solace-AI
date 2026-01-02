"""
Solace-AI Exception Hierarchy.
Enterprise-grade structured exception handling with correlation tracking.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_RULE = "business_rule"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    CACHE = "cache"
    CONFIGURATION = "configuration"
    RATE_LIMIT = "rate_limit"
    SAFETY = "safety"
    INTERNAL = "internal"


class ErrorContext(BaseModel):
    """Structured context for error tracking."""
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    service_name: str = Field(default="solace-ai")
    operation: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    additional_data: dict[str, Any] = Field(default_factory=dict)
    model_config = {"frozen": True}

    def with_operation(self, operation: str) -> ErrorContext:
        return ErrorContext(
            correlation_id=self.correlation_id, timestamp=self.timestamp,
            service_name=self.service_name, operation=operation,
            user_id=self.user_id, session_id=self.session_id,
            request_id=self.request_id, additional_data=self.additional_data,
        )

    def with_user(self, user_id: str, session_id: str | None = None) -> ErrorContext:
        return ErrorContext(
            correlation_id=self.correlation_id, timestamp=self.timestamp,
            service_name=self.service_name, operation=self.operation,
            user_id=user_id, session_id=session_id or self.session_id,
            request_id=self.request_id, additional_data=self.additional_data,
        )


class SolaceError(Exception):
    """Base exception for all Solace-AI errors with structured tracking."""
    error_code: str = "SOLACE_ERROR"
    category: ErrorCategory = ErrorCategory.INTERNAL
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    http_status: int = 500

    def __init__(self, message: str, *, user_message: str | None = None,
                 context: ErrorContext | None = None, cause: Exception | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "An error occurred. Please try again."
        self.context = context or ErrorContext()
        self.cause = cause
        self.details = details or {}
        self._log_error()

    def _log_error(self) -> None:
        log_data = {
            "error_code": self.error_code, "category": self.category.value,
            "severity": self.severity.value, "correlation_id": self.context.correlation_id,
            "operation": self.context.operation, "details": self.details,
        }
        if self.context.user_id:
            log_data["user_id"] = self.context.user_id
        if self.cause:
            log_data["cause_type"] = type(self.cause).__name__
            log_data["cause_message"] = str(self.cause)
        if self.severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL):
            logger.error(self.message, **log_data)
        else:
            logger.warning(self.message, **log_data)

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.error_code, "message": self.user_message,
                         "correlation_id": self.context.correlation_id,
                         "timestamp": self.context.timestamp.isoformat()}}

    def to_internal_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["internal"] = {"message": self.message, "category": self.category.value,
                              "severity": self.severity.value, "details": self.details,
                              "operation": self.context.operation}
        if self.cause:
            result["internal"]["cause"] = {"type": type(self.cause).__name__,
                                           "message": str(self.cause)}
        return result


# Domain Layer Exceptions
class DomainError(SolaceError):
    error_code = "DOMAIN_ERROR"
    category = ErrorCategory.BUSINESS_RULE
    severity = ErrorSeverity.MEDIUM
    http_status = 422


class ValidationError(DomainError):
    error_code = "VALIDATION_ERROR"
    category = ErrorCategory.VALIDATION
    severity = ErrorSeverity.LOW
    http_status = 400

    def __init__(self, message: str, *, field: str | None = None, value: Any = None,
                 constraint: str | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if constraint:
            details["constraint"] = constraint
        user_message = f"Invalid value for {field}" if field else "Validation failed"
        super().__init__(message, user_message=user_message, details=details, **kwargs)
        self.field, self.value, self.constraint = field, value, constraint


class EntityNotFoundError(DomainError):
    error_code = "ENTITY_NOT_FOUND"
    category = ErrorCategory.NOT_FOUND
    severity = ErrorSeverity.LOW
    http_status = 404

    def __init__(self, entity_type: str, entity_id: str, **kwargs: Any) -> None:
        message = f"{entity_type} with ID '{entity_id}' not found"
        user_message = f"The requested {entity_type.lower()} was not found"
        details = kwargs.pop("details", {})
        details.update({"entity_type": entity_type, "entity_id": entity_id})
        super().__init__(message, user_message=user_message, details=details, **kwargs)
        self.entity_type, self.entity_id = entity_type, entity_id


class EntityConflictError(DomainError):
    error_code = "ENTITY_CONFLICT"
    category = ErrorCategory.CONFLICT
    severity = ErrorSeverity.MEDIUM
    http_status = 409

    def __init__(self, message: str, *, entity_type: str | None = None,
                 entity_id: str | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if entity_type:
            details["entity_type"] = entity_type
        if entity_id:
            details["entity_id"] = entity_id
        super().__init__(message, user_message="A conflict occurred", details=details, **kwargs)


class ConcurrencyError(EntityConflictError):
    error_code = "CONCURRENCY_ERROR"

    def __init__(self, entity_type: str, entity_id: str, expected_version: int,
                 actual_version: int, **kwargs: Any) -> None:
        message = (f"Concurrency conflict for {entity_type} '{entity_id}': "
                   f"expected version {expected_version}, found {actual_version}")
        details = kwargs.pop("details", {})
        details.update({"expected_version": expected_version, "actual_version": actual_version})
        super().__init__(message, entity_type=entity_type, entity_id=entity_id,
                         details=details, **kwargs)
        self.expected_version, self.actual_version = expected_version, actual_version


class BusinessRuleViolationError(DomainError):
    error_code = "BUSINESS_RULE_VIOLATION"

    def __init__(self, rule: str, message: str, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details["rule"] = rule
        super().__init__(message, user_message="Operation not allowed", details=details, **kwargs)
        self.rule = rule


class InvariantViolationError(DomainError):
    error_code = "INVARIANT_VIOLATION"
    severity = ErrorSeverity.HIGH


# Application Layer Exceptions
class ApplicationError(SolaceError):
    error_code = "APPLICATION_ERROR"
    severity = ErrorSeverity.MEDIUM
    http_status = 500


class AuthenticationError(ApplicationError):
    error_code = "AUTHENTICATION_ERROR"
    category = ErrorCategory.AUTHENTICATION
    http_status = 401

    def __init__(self, message: str = "Authentication failed", **kwargs: Any) -> None:
        super().__init__(message, user_message="Invalid credentials", **kwargs)


class AuthorizationError(ApplicationError):
    error_code = "AUTHORIZATION_ERROR"
    category = ErrorCategory.AUTHORIZATION
    http_status = 403

    def __init__(self, message: str = "Access denied", *,
                 required_permission: str | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if required_permission:
            details["required_permission"] = required_permission
        super().__init__(message, user_message="You don't have permission for this action",
                         details=details, **kwargs)


class RateLimitExceededError(ApplicationError):
    error_code = "RATE_LIMIT_EXCEEDED"
    category = ErrorCategory.RATE_LIMIT
    severity = ErrorSeverity.LOW
    http_status = 429

    def __init__(self, limit: int, window_seconds: int,
                 retry_after: int | None = None, **kwargs: Any) -> None:
        message = f"Rate limit of {limit} requests per {window_seconds}s exceeded"
        details = kwargs.pop("details", {})
        details.update({"limit": limit, "window_seconds": window_seconds})
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, user_message="Too many requests. Please try again later.",
                         details=details, **kwargs)
        self.retry_after = retry_after


class SafetyError(ApplicationError):
    error_code = "SAFETY_ERROR"
    category = ErrorCategory.SAFETY
    severity = ErrorSeverity.CRITICAL
    http_status = 422

    def __init__(self, message: str, *, safety_level: int | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if safety_level is not None:
            details["safety_level"] = safety_level
        super().__init__(message, details=details, **kwargs)


# Infrastructure Layer Exceptions
class InfrastructureError(SolaceError):
    error_code = "INFRASTRUCTURE_ERROR"
    category = ErrorCategory.EXTERNAL_SERVICE
    severity = ErrorSeverity.HIGH
    http_status = 503


class DatabaseError(InfrastructureError):
    error_code = "DATABASE_ERROR"
    category = ErrorCategory.DATABASE

    def __init__(self, message: str, *, operation: str | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if operation:
            details["db_operation"] = operation
        super().__init__(message, user_message="A database error occurred",
                         details=details, **kwargs)


class CacheError(InfrastructureError):
    error_code = "CACHE_ERROR"
    category = ErrorCategory.CACHE
    severity = ErrorSeverity.MEDIUM


class ExternalServiceError(InfrastructureError):
    error_code = "EXTERNAL_SERVICE_ERROR"

    def __init__(self, service_name: str, message: str, *,
                 status_code: int | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details["service_name"] = service_name
        if status_code:
            details["upstream_status"] = status_code
        super().__init__(message, user_message="An external service is temporarily unavailable",
                         details=details, **kwargs)
        self.service_name = service_name


class LLMServiceError(ExternalServiceError):
    error_code = "LLM_SERVICE_ERROR"

    def __init__(self, provider: str, message: str, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        details["provider"] = provider
        super().__init__(service_name=f"LLM:{provider}", message=message,
                         details=details, **kwargs)


class ConfigurationError(InfrastructureError):
    error_code = "CONFIGURATION_ERROR"
    category = ErrorCategory.CONFIGURATION
    severity = ErrorSeverity.CRITICAL
    http_status = 500

    def __init__(self, message: str, *, config_key: str | None = None, **kwargs: Any) -> None:
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, user_message="Service configuration error",
                         details=details, **kwargs)
