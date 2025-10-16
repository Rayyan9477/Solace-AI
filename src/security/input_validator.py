"""
Comprehensive input validation and sanitization for Contextual-Chatbot.

This module provides robust input validation to prevent injection attacks,
XSS, and other security vulnerabilities identified in the code review.
"""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import bleach

from ..core.exceptions import (
    InputValidationError,
    InjectionAttackDetected,
    XSSAttackDetected
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation failures"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_value: Any
    errors: List[str]
    warnings: List[str]
    severity: ValidationSeverity

    def has_errors(self) -> bool:
        """Check if validation has errors"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if validation has warnings"""
        return len(self.warnings) > 0


class InputValidator:
    """
    Comprehensive input validator with sanitization.

    Addresses security vulnerability: "Input Validation - Insufficient input
    validation across various modules, potentially leading to injection attacks"
    """

    # Dangerous patterns that should be blocked
    SQL_INJECTION_PATTERNS = [
        r"(\bOR\b|\bAND\b).*?=.*?",
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r";\s*UPDATE\s+.*?SET",
        r";\s*INSERT\s+INTO",
        r"UNION\s+SELECT",
        r"'.*?--",
        r"'.*?#",
        r"'.*?/\*",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",  # Shell metacharacters
        r"\$\{.*?\}",  # Variable expansion
        r"\$\(.*?\)",  # Command substitution
        r"`.*?`",  # Backtick command execution
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick=
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e/",
        r"%2e%2e\\",
    ]

    # Allowed HTML tags for sanitization (very restrictive)
    ALLOWED_HTML_TAGS = ['p', 'br', 'strong', 'em', 'u']
    ALLOWED_HTML_ATTRIBUTES = {}

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the input validator.

        Args:
            strict_mode: If True, applies more restrictive validation rules
        """
        self.strict_mode = strict_mode
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.sql_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
        self.cmd_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.COMMAND_INJECTION_PATTERNS
        ]
        self.xss_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.XSS_PATTERNS
        ]
        self.path_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PATH_TRAVERSAL_PATTERNS
        ]

    def validate_string(
        self,
        value: str,
        field_name: str,
        min_length: int = 0,
        max_length: int = 10000,
        allow_special_chars: bool = True,
        patterns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate and sanitize a string input.

        Args:
            value: Input string to validate
            field_name: Name of the field being validated
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_special_chars: Whether to allow special characters
            patterns: List of allowed regex patterns

        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW

        # Type check
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                sanitized_value="",
                errors=[f"{field_name} must be a string"],
                warnings=[],
                severity=ValidationSeverity.HIGH
            )

        original_value = value

        # Length validation
        if len(value) < min_length:
            errors.append(f"{field_name} must be at least {min_length} characters")
            severity = ValidationSeverity.MEDIUM

        if len(value) > max_length:
            errors.append(f"{field_name} must not exceed {max_length} characters")
            value = value[:max_length]
            warnings.append(f"{field_name} was truncated to {max_length} characters")
            severity = ValidationSeverity.MEDIUM

        # Check for SQL injection
        sql_detected = self._check_patterns(value, self.sql_patterns)
        if sql_detected:
            errors.append(f"{field_name} contains potential SQL injection patterns")
            severity = ValidationSeverity.CRITICAL

        # Check for command injection
        cmd_detected = self._check_patterns(value, self.cmd_patterns)
        if cmd_detected:
            errors.append(f"{field_name} contains potential command injection patterns")
            severity = ValidationSeverity.CRITICAL

        # Check for XSS
        xss_detected = self._check_patterns(value, self.xss_patterns)
        if xss_detected:
            errors.append(f"{field_name} contains potential XSS patterns")
            severity = ValidationSeverity.CRITICAL

        # Check for path traversal
        path_detected = self._check_patterns(value, self.path_patterns)
        if path_detected:
            errors.append(f"{field_name} contains path traversal patterns")
            severity = ValidationSeverity.HIGH

        # Sanitize the value
        sanitized = self._sanitize_string(value)

        # Check if sanitization changed the value significantly
        if sanitized != original_value and not errors:
            warnings.append(f"{field_name} was sanitized")

        # Pattern validation if provided
        if patterns:
            pattern_matched = any(re.match(pattern, sanitized) for pattern in patterns)
            if not pattern_matched:
                errors.append(f"{field_name} does not match required pattern")
                severity = ValidationSeverity.MEDIUM

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            severity=severity
        )

    def validate_user_input(self, user_input: str, max_length: int = 5000) -> ValidationResult:
        """
        Validate user chat input with appropriate security checks.

        Args:
            user_input: User's chat message
            max_length: Maximum allowed message length

        Returns:
            ValidationResult
        """
        return self.validate_string(
            value=user_input,
            field_name="user_input",
            min_length=1,
            max_length=max_length,
            allow_special_chars=True
        )

    def validate_json(
        self,
        json_str: str,
        field_name: str,
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate JSON input.

        Args:
            json_str: JSON string to validate
            field_name: Name of the field
            required_fields: List of required field names in the JSON

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                errors=[f"{field_name} contains invalid JSON: {str(e)}"],
                warnings=[],
                severity=ValidationSeverity.HIGH
            )

        # Check required fields
        if required_fields and isinstance(parsed, dict):
            missing_fields = [
                field for field in required_fields
                if field not in parsed
            ]
            if missing_fields:
                errors.append(
                    f"{field_name} missing required fields: {', '.join(missing_fields)}"
                )

        # Recursively sanitize nested values
        sanitized = self._sanitize_nested(parsed)

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.MEDIUM if errors else ValidationSeverity.LOW
        )

    def validate_dict(
        self,
        data: Dict[str, Any],
        field_name: str,
        allowed_keys: Optional[List[str]] = None,
        required_keys: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate dictionary input.

        Args:
            data: Dictionary to validate
            field_name: Name of the field
            allowed_keys: List of allowed keys (None = allow all)
            required_keys: List of required keys

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                sanitized_value={},
                errors=[f"{field_name} must be a dictionary"],
                warnings=[],
                severity=ValidationSeverity.HIGH
            )

        # Check required keys
        if required_keys:
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                errors.append(
                    f"{field_name} missing required keys: {', '.join(missing_keys)}"
                )

        # Check allowed keys
        if allowed_keys:
            invalid_keys = [key for key in data.keys() if key not in allowed_keys]
            if invalid_keys:
                warnings.append(
                    f"{field_name} contains unexpected keys: {', '.join(invalid_keys)}"
                )
                # Remove invalid keys
                data = {k: v for k, v in data.items() if k in allowed_keys}

        # Sanitize nested values
        sanitized = self._sanitize_nested(data)

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.MEDIUM if errors else ValidationSeverity.LOW
        )

    def validate_email(self, email: str, field_name: str = "email") -> ValidationResult:
        """Validate email address format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        if not re.match(email_pattern, email):
            return ValidationResult(
                is_valid=False,
                sanitized_value="",
                errors=[f"{field_name} is not a valid email address"],
                warnings=[],
                severity=ValidationSeverity.MEDIUM
            )

        return ValidationResult(
            is_valid=True,
            sanitized_value=email.lower().strip(),
            errors=[],
            warnings=[],
            severity=ValidationSeverity.LOW
        )

    def validate_url(self, url: str, field_name: str = "url") -> ValidationResult:
        """Validate URL format and protocol"""
        errors = []

        # Check protocol
        allowed_protocols = ['http://', 'https://']
        if not any(url.startswith(protocol) for protocol in allowed_protocols):
            errors.append(f"{field_name} must start with http:// or https://")

        # Check for dangerous patterns
        if 'javascript:' in url.lower() or 'data:' in url.lower():
            errors.append(f"{field_name} contains dangerous protocol")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=url,
            errors=errors,
            warnings=[],
            severity=ValidationSeverity.HIGH if errors else ValidationSeverity.LOW
        )

    def validate_integer(
        self,
        value: Any,
        field_name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """Validate integer input with range checks"""
        errors = []

        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                sanitized_value=0,
                errors=[f"{field_name} must be an integer"],
                warnings=[],
                severity=ValidationSeverity.MEDIUM
            )

        if min_value is not None and int_value < min_value:
            errors.append(f"{field_name} must be at least {min_value}")

        if max_value is not None and int_value > max_value:
            errors.append(f"{field_name} must not exceed {max_value}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=int_value,
            errors=errors,
            warnings=[],
            severity=ValidationSeverity.MEDIUM if errors else ValidationSeverity.LOW
        )

    def validate_float(
        self,
        value: Any,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> ValidationResult:
        """Validate float input with range checks"""
        errors = []

        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                sanitized_value=0.0,
                errors=[f"{field_name} must be a number"],
                warnings=[],
                severity=ValidationSeverity.MEDIUM
            )

        if min_value is not None and float_value < min_value:
            errors.append(f"{field_name} must be at least {min_value}")

        if max_value is not None and float_value > max_value:
            errors.append(f"{field_name} must not exceed {max_value}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=float_value,
            errors=errors,
            warnings=[],
            severity=ValidationSeverity.MEDIUM if errors else ValidationSeverity.LOW
        )

    def _check_patterns(self, value: str, patterns: List[re.Pattern]) -> bool:
        """Check if value matches any of the dangerous patterns"""
        return any(pattern.search(value) for pattern in patterns)

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string by removing/escaping dangerous content"""
        # HTML escape
        sanitized = html.escape(value)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()

        return sanitized

    def _sanitize_nested(self, obj: Any) -> Any:
        """Recursively sanitize nested structures"""
        if isinstance(obj, dict):
            return {
                key: self._sanitize_nested(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._sanitize_nested(item) for item in obj]
        elif isinstance(obj, str):
            return self._sanitize_string(obj)
        else:
            return obj

    def sanitize_html(self, html_content: str) -> str:
        """
        Sanitize HTML content using bleach library.

        Args:
            html_content: HTML string to sanitize

        Returns:
            Sanitized HTML with only allowed tags and attributes
        """
        try:
            return bleach.clean(
                html_content,
                tags=self.ALLOWED_HTML_TAGS,
                attributes=self.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
        except Exception as e:
            logger.error(f"Error sanitizing HTML: {e}")
            return html.escape(html_content)


# Singleton instance
_input_validator: Optional[InputValidator] = None


def get_input_validator(strict_mode: bool = True) -> InputValidator:
    """Get the global input validator instance"""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator(strict_mode=strict_mode)
    return _input_validator


def validate_user_message(message: str) -> ValidationResult:
    """
    Convenience function to validate user chat messages.

    Args:
        message: User's message

    Returns:
        ValidationResult
    """
    validator = get_input_validator()
    return validator.validate_user_input(message)
