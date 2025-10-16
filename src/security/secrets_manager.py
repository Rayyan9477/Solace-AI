"""
Secure secrets and API key management for Contextual-Chatbot.

This module provides secure handling of API keys, secrets rotation,
and validation to address security vulnerabilities identified in the code review.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import hashlib

from ..core.exceptions import (
    SecretValidationError,
    SecretRotationRequired,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets managed by the system"""
    API_KEY = "api_key"
    DATABASE_URL = "database_url"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_TOKEN = "oauth_token"
    WEBHOOK_SECRET = "webhook_secret"


@dataclass
class SecretMetadata:
    """Metadata about a secret"""
    name: str
    secret_type: SecretType
    created_at: datetime
    last_rotated: datetime
    rotation_days: int
    masked_value: str
    is_valid: bool = True

    def needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        if self.rotation_days <= 0:
            return False
        days_since_rotation = (datetime.now() - self.last_rotated).days
        return days_since_rotation >= self.rotation_days


class SecretsManager:
    """
    Secure secrets manager with validation, rotation tracking, and audit logging.

    This class addresses the security vulnerabilities identified in the code review:
    - Validates API keys and secrets
    - Tracks secret rotation
    - Provides secure access patterns
    - Masks sensitive data in logs
    - Validates environment variable configuration
    """

    # API key validation patterns
    API_KEY_PATTERNS = {
        'gemini': re.compile(r'^[A-Za-z0-9_-]{39}$'),  # Google API keys are typically 39 chars
        'openai': re.compile(r'^sk-[A-Za-z0-9]{48}$'),  # OpenAI keys start with sk-
        'generic': re.compile(r'^[A-Za-z0-9_-]{20,}$')  # Generic minimum length
    }

    def __init__(self, rotation_days: int = 90):
        """
        Initialize the secrets manager.

        Args:
            rotation_days: Default number of days before secrets should be rotated
        """
        self.rotation_days = rotation_days
        self._secrets_metadata: Dict[str, SecretMetadata] = {}
        self._validation_errors: List[str] = []

    def register_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        rotation_days: Optional[int] = None
    ) -> bool:
        """
        Register a secret with validation and metadata tracking.

        Args:
            name: Secret name (e.g., 'GEMINI_API_KEY')
            value: Secret value
            secret_type: Type of secret
            rotation_days: Days before rotation needed (None = use default)

        Returns:
            True if secret is valid and registered, False otherwise
        """
        # Validate the secret
        is_valid, error_msg = self._validate_secret(name, value, secret_type)

        if not is_valid:
            self._validation_errors.append(f"{name}: {error_msg}")
            logger.warning(f"Invalid secret {name}: {error_msg}")
            return False

        # Create metadata
        metadata = SecretMetadata(
            name=name,
            secret_type=secret_type,
            created_at=datetime.now(),
            last_rotated=datetime.now(),
            rotation_days=rotation_days or self.rotation_days,
            masked_value=self._mask_secret(value),
            is_valid=True
        )

        self._secrets_metadata[name] = metadata
        logger.info(f"Registered secret: {name} (type: {secret_type.value})")

        return True

    def _validate_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType
    ) -> tuple[bool, str]:
        """
        Validate a secret value.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if value is empty
        if not value or not value.strip():
            return False, "Secret value is empty"

        # Check for placeholder values
        placeholder_patterns = [
            'your_api_key_here',
            'replace_me',
            'changeme',
            'example',
            'test',
            'dummy',
            '12345'
        ]
        if any(placeholder in value.lower() for placeholder in placeholder_patterns):
            return False, "Secret appears to be a placeholder value"

        # Check minimum length
        if len(value) < 20:
            return False, f"Secret too short (minimum 20 characters, got {len(value)})"

        # Type-specific validation
        if secret_type == SecretType.API_KEY:
            return self._validate_api_key(name, value)
        elif secret_type == SecretType.DATABASE_URL:
            return self._validate_database_url(value)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return self._validate_encryption_key(value)

        # Default validation for other types
        return True, ""

    def _validate_api_key(self, name: str, value: str) -> tuple[bool, str]:
        """Validate API key format"""
        # Determine provider from name
        provider = None
        name_lower = name.lower()

        if 'gemini' in name_lower or 'google' in name_lower:
            provider = 'gemini'
        elif 'openai' in name_lower or 'gpt' in name_lower:
            provider = 'openai'

        # Check format
        if provider and provider in self.API_KEY_PATTERNS:
            pattern = self.API_KEY_PATTERNS[provider]
            if not pattern.match(value):
                return False, f"Invalid {provider} API key format"
        else:
            # Generic validation
            pattern = self.API_KEY_PATTERNS['generic']
            if not pattern.match(value):
                return False, "Invalid API key format"

        # Check for common invalid patterns
        if value.count('0') > len(value) * 0.5:
            return False, "API key contains too many zeros (possible dummy value)"

        return True, ""

    def _validate_database_url(self, value: str) -> tuple[bool, str]:
        """Validate database URL format"""
        # Basic URL validation
        if not value.startswith(('postgresql://', 'mysql://', 'sqlite://', 'mongodb://')):
            return False, "Invalid database URL protocol"

        # Check for localhost or test patterns in production
        if any(pattern in value for pattern in ['localhost', '127.0.0.1', 'test', 'example']):
            logger.warning("Database URL contains localhost or test patterns")

        return True, ""

    def _validate_encryption_key(self, value: str) -> tuple[bool, str]:
        """Validate encryption key"""
        # Must be at least 32 characters for AES-256
        if len(value) < 32:
            return False, "Encryption key must be at least 32 characters for AES-256"

        # Check entropy (should have good mix of characters)
        unique_chars = len(set(value))
        if unique_chars < 10:
            return False, "Encryption key has low entropy"

        return True, ""

    def _mask_secret(self, value: str) -> str:
        """Mask a secret value for logging"""
        if len(value) <= 8:
            return "***"

        # Show first 4 and last 4 characters
        return f"{value[:4]}...{value[-4:]}"

    def get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret"""
        return self._secrets_metadata.get(name)

    def check_rotation_needed(self) -> List[SecretMetadata]:
        """Get list of secrets that need rotation"""
        return [
            metadata for metadata in self._secrets_metadata.values()
            if metadata.needs_rotation()
        ]

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self._validation_errors.copy()

    def mark_rotated(self, name: str) -> bool:
        """Mark a secret as rotated"""
        if name in self._secrets_metadata:
            self._secrets_metadata[name].last_rotated = datetime.now()
            logger.info(f"Secret {name} marked as rotated")
            return True
        return False

    @staticmethod
    def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary by masking sensitive fields for safe logging.

        Args:
            data: Dictionary that may contain sensitive data

        Returns:
            Sanitized dictionary with masked sensitive values
        """
        sensitive_keys = {
            'api_key', 'apikey', 'api-key',
            'password', 'passwd', 'pwd',
            'secret', 'token', 'auth',
            'private_key', 'private-key',
            'access_token', 'refresh_token',
            'client_secret', 'webhook_secret',
            'encryption_key', 'salt'
        }

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower().replace('_', '').replace('-', '')

            # Check if key contains sensitive terms
            is_sensitive = any(
                sensitive_term in key_lower
                for sensitive_term in sensitive_keys
            )

            if is_sensitive and isinstance(value, str):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = SecretsManager.sanitize_for_logging(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    SecretsManager.sanitize_for_logging(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized


class EnvironmentValidator:
    """
    Validates environment configuration and ensures proper security settings.

    Addresses the security vulnerability: "Environment Variable Handling -
    Potential for sensitive data exposure through insecure environment variable handling"
    """

    REQUIRED_VARS = [
        'LLM_PROVIDER',
        'MODEL_NAME',
    ]

    SENSITIVE_VARS = [
        'GEMINI_API_KEY',
        'OPENAI_API_KEY',
        'DATABASE_URL',
        'SENTRY_DSN',
    ]

    def __init__(self, secrets_manager: SecretsManager):
        """Initialize with a secrets manager"""
        self.secrets_manager = secrets_manager
        self.validation_results: Dict[str, Any] = {}

    def validate_environment(self) -> bool:
        """
        Validate the complete environment configuration.

        Returns:
            True if environment is valid, False otherwise
        """
        all_valid = True

        # Check required variables
        for var_name in self.REQUIRED_VARS:
            value = os.getenv(var_name)
            if not value:
                logger.error(f"Required environment variable {var_name} is not set")
                self.validation_results[var_name] = {"status": "missing", "required": True}
                all_valid = False
            else:
                self.validation_results[var_name] = {"status": "present", "required": True}

        # Validate sensitive variables
        provider = os.getenv('LLM_PROVIDER', '').lower()

        # Check provider-specific API keys
        if provider == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                logger.error("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'")
                all_valid = False
            else:
                is_valid = self.secrets_manager.register_secret(
                    'GEMINI_API_KEY',
                    api_key,
                    SecretType.API_KEY
                )
                if not is_valid:
                    all_valid = False

        elif provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
                all_valid = False
            else:
                is_valid = self.secrets_manager.register_secret(
                    'OPENAI_API_KEY',
                    api_key,
                    SecretType.API_KEY
                )
                if not is_valid:
                    all_valid = False

        # Check for .env file exposure risk
        self._check_env_file_security()

        # Validate debug mode is not enabled in production
        self._check_debug_mode()

        return all_valid

    def _check_env_file_security(self):
        """Check if .env file has appropriate permissions"""
        env_file = Path('.env')

        if env_file.exists():
            # Check if .env is in .gitignore
            gitignore = Path('.gitignore')
            if gitignore.exists():
                with open(gitignore, 'r') as f:
                    if '.env' not in f.read():
                        logger.warning(".env file is not in .gitignore - risk of secret exposure!")
                        self.validation_results['.gitignore'] = {
                            "status": "warning",
                            "message": ".env not in .gitignore"
                        }
            else:
                logger.warning(".gitignore file not found - create one and add .env to it")
        else:
            logger.warning(".env file not found - using system environment variables")

    def _check_debug_mode(self):
        """Check if debug mode is enabled"""
        debug = os.getenv('DEBUG', 'False').lower() == 'true'

        if debug:
            logger.warning("DEBUG mode is enabled - disable in production!")
            self.validation_results['DEBUG'] = {
                "status": "warning",
                "message": "Debug mode enabled - security risk in production"
            }

    def get_validation_report(self) -> Dict[str, Any]:
        """Get a full validation report"""
        errors = self.secrets_manager.get_validation_errors()
        rotation_needed = self.secrets_manager.check_rotation_needed()

        return {
            "environment_variables": self.validation_results,
            "validation_errors": errors,
            "secrets_needing_rotation": [
                {
                    "name": secret.name,
                    "days_since_rotation": (datetime.now() - secret.last_rotated).days,
                    "rotation_threshold": secret.rotation_days
                }
                for secret in rotation_needed
            ],
            "overall_status": "valid" if not errors else "invalid"
        }


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def validate_environment_security() -> bool:
    """
    Validate environment security configuration.

    This function should be called during application startup to ensure
    all security requirements are met.

    Returns:
        True if environment is secure, False otherwise
    """
    secrets_manager = get_secrets_manager()
    validator = EnvironmentValidator(secrets_manager)

    is_valid = validator.validate_environment()

    if not is_valid:
        logger.error("Environment validation failed!")
        report = validator.get_validation_report()
        logger.error(f"Validation report: {report}")

    return is_valid
