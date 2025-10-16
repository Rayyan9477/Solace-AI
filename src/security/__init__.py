"""
Security module for Contextual-Chatbot.

This module provides comprehensive security features including:
- Secrets management and API key validation
- Input validation and sanitization
- Environment security checks
"""

from .secrets_manager import (
    SecretsManager,
    EnvironmentValidator,
    SecretType,
    SecretMetadata,
    get_secrets_manager,
    validate_environment_security
)

from .input_validator import (
    InputValidator,
    ValidationResult,
    ValidationSeverity,
    get_input_validator,
    validate_user_message
)

__all__ = [
    # Secrets management
    'SecretsManager',
    'EnvironmentValidator',
    'SecretType',
    'SecretMetadata',
    'get_secrets_manager',
    'validate_environment_security',

    # Input validation
    'InputValidator',
    'ValidationResult',
    'ValidationSeverity',
    'get_input_validator',
    'validate_user_message',
]
