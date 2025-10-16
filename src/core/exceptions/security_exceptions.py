"""
Security-related exception classes for the Contextual-Chatbot application.

This module defines exceptions for security vulnerabilities, authentication failures,
and authorization issues identified in the security audit.
"""

from typing import Optional, Dict, Any
from .base_exceptions import ChatbotBaseException


class SecurityException(ChatbotBaseException):
    """
    Base exception for all security-related errors.

    This includes authentication, authorization, input validation failures,
    and detected security threats.
    """

    def __init__(
        self,
        message: str,
        security_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if security_context:
            context.update(security_context)

        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class AuthenticationError(SecurityException):
    """
    Exception raised for authentication failures.

    This includes invalid credentials, expired tokens, and missing authentication.
    """

    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if auth_method:
            context['auth_method'] = auth_method
        if user_id:
            context['user_id'] = user_id

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="AUTH_ERROR",
            **kwargs
        )


class AuthorizationError(SecurityException):
    """
    Exception raised for authorization failures.

    This includes insufficient permissions and access denied errors.
    """

    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if required_permission:
            context['required_permission'] = required_permission
        if user_id:
            context['user_id'] = user_id
        if resource:
            context['resource'] = resource

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="AUTHZ_ERROR",
            **kwargs
        )


class InputValidationError(SecurityException):
    """
    Exception raised for input validation security failures.

    This includes detected injection attacks, XSS attempts, and malformed input.
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        attack_type: Optional[str] = None,
        sanitized_value: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field_name:
            context['field_name'] = field_name
        if attack_type:
            context['attack_type'] = attack_type
        if sanitized_value:
            context['sanitized_value'] = sanitized_value

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="INPUT_VALIDATION_ERROR",
            **kwargs
        )


class InjectionAttackDetected(InputValidationError):
    """
    Exception raised when an injection attack is detected.

    This includes SQL injection, command injection, and code injection attempts.
    """

    def __init__(
        self,
        message: str,
        injection_type: str,
        detected_pattern: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['injection_type'] = injection_type
        if detected_pattern:
            context['detected_pattern'] = detected_pattern

        kwargs['context'] = context
        kwargs['attack_type'] = injection_type
        super().__init__(
            message,
            error_code="INJECTION_DETECTED",
            **kwargs
        )


class XSSAttackDetected(InputValidationError):
    """
    Exception raised when a Cross-Site Scripting (XSS) attack is detected.
    """

    def __init__(
        self,
        message: str,
        detected_pattern: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if detected_pattern:
            context['detected_pattern'] = detected_pattern

        kwargs['context'] = context
        kwargs['attack_type'] = 'xss'
        super().__init__(
            message,
            error_code="XSS_DETECTED",
            **kwargs
        )


class SecretValidationError(SecurityException):
    """
    Exception raised for secret and API key validation failures.

    This includes invalid API keys, malformed secrets, and placeholder values.
    """

    def __init__(
        self,
        message: str,
        secret_name: Optional[str] = None,
        validation_failure: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if secret_name:
            context['secret_name'] = secret_name
        if validation_failure:
            context['validation_failure'] = validation_failure

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="SECRET_VALIDATION_ERROR",
            **kwargs
        )


class SecretRotationRequired(SecurityException):
    """
    Exception raised when a secret requires rotation.

    This is used for warning/alerting purposes when secrets are past their rotation date.
    """

    def __init__(
        self,
        message: str,
        secret_name: str,
        days_since_rotation: int,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['secret_name'] = secret_name
        context['days_since_rotation'] = days_since_rotation

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="SECRET_ROTATION_REQUIRED",
            **kwargs
        )


class RateLimitExceeded(SecurityException):
    """
    Exception raised when rate limits are exceeded.

    This helps prevent abuse and DOS attacks.
    """

    def __init__(
        self,
        message: str,
        limit: int,
        current_count: int,
        window_seconds: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['limit'] = limit
        context['current_count'] = current_count
        if window_seconds:
            context['window_seconds'] = window_seconds

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="RATE_LIMIT_EXCEEDED",
            **kwargs
        )


class EncryptionError(SecurityException):
    """
    Exception raised for encryption/decryption failures.

    This includes key management errors and cryptographic operation failures.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="ENCRYPTION_ERROR",
            **kwargs
        )


class DataExposureRisk(SecurityException):
    """
    Exception raised when there's a risk of data exposure.

    This includes logging sensitive data, insecure transmission, etc.
    """

    def __init__(
        self,
        message: str,
        risk_type: str,
        affected_data: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['risk_type'] = risk_type
        if affected_data:
            context['affected_data'] = affected_data

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="DATA_EXPOSURE_RISK",
            **kwargs
        )


class CircuitBreakerOpen(SecurityException):
    """
    Exception raised when a circuit breaker is open.

    This prevents cascading failures and helps maintain system stability.
    """

    def __init__(
        self,
        message: str,
        service_name: str,
        failure_count: int,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['service_name'] = service_name
        context['failure_count'] = failure_count

        kwargs['context'] = context
        super().__init__(
            message,
            error_code="CIRCUIT_BREAKER_OPEN",
            **kwargs
        )
