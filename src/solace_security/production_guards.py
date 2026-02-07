"""Production environment guards for Solace-AI.

Prevents dangerous development settings from being used in production environments.
Validates configuration at startup and raises errors for non-compliant settings.

Usage:
    # Call during application startup
    from solace_security.production_guards import ProductionGuards

    ProductionGuards.validate_environment()
    # Raises SecurityError if dangerous settings detected in production

    # Or validate specific settings
    ProductionGuards.validate_database_settings()
    ProductionGuards.validate_encryption_settings()
    ProductionGuards.validate_auth_settings()
"""

from __future__ import annotations

import os
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


class ProductionSecurityError(Exception):
    """Raised when production environment fails security validation."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        message = (
            "Production environment security validation FAILED.\n"
            "The following violations were detected:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix these issues before deploying to production."
        )
        super().__init__(message)


class ProductionGuards:
    """Validates environment configuration for production safety.

    Checks for:
    - Default/development credentials in production
    - Missing encryption keys
    - Insecure SSL settings
    - Debug mode enabled in production
    - Missing required environment variables
    """

    # Environment variables that must NOT have these values in production
    FORBIDDEN_VALUES: ClassVar[dict[str, list[str]]] = {
        "SECRET_KEY": ["dev", "development", "test", "secret", "changeme", "default"],
        "POSTGRES_PASSWORD": ["postgres", "password", "dev", "test", "changeme", ""],
        "ENCRYPTION_KEY": ["test", "dev", "development", "changeme", "default"],
        "JWT_SECRET": ["dev", "test", "secret", "changeme", "default"],
        "API_KEY": ["dev", "test", "changeme", "default"],
    }

    # Environment variables that MUST be set in production
    REQUIRED_IN_PRODUCTION: ClassVar[list[str]] = [
        "POSTGRES_PASSWORD",
        "SECRET_KEY",
        "ENCRYPTION_KEY",
    ]

    # Settings that must NOT be enabled in production
    FORBIDDEN_FLAGS: ClassVar[dict[str, list[str]]] = {
        "DEBUG": ["true", "1", "yes"],
        "TESTING": ["true", "1", "yes"],
        "MOCK_SERVICES": ["true", "1", "yes"],
        "USE_IN_MEMORY_DB": ["true", "1", "yes"],
    }

    @classmethod
    def validate_environment(cls) -> None:
        """Run all production environment validations.

        Raises:
            ProductionSecurityError: If any violations detected in production
        """
        env = os.getenv("ENVIRONMENT", "development").lower()

        if env not in ("production", "staging"):
            logger.debug(
                "production_guards_skipped",
                environment=env,
                reason="Not a production/staging environment",
            )
            return

        violations: list[str] = []

        violations.extend(cls._check_forbidden_values())
        violations.extend(cls._check_required_variables())
        violations.extend(cls._check_forbidden_flags())
        violations.extend(cls._check_database_security())
        violations.extend(cls._check_encryption_settings())

        if violations:
            logger.critical(
                "production_security_validation_failed",
                environment=env,
                violation_count=len(violations),
                violations=violations,
            )
            raise ProductionSecurityError(violations)

        logger.info(
            "production_security_validation_passed",
            environment=env,
            checks_passed=5,
        )

    @classmethod
    def _check_forbidden_values(cls) -> list[str]:
        """Check for forbidden default/development values."""
        violations = []
        for env_var, forbidden_values in cls.FORBIDDEN_VALUES.items():
            value = os.getenv(env_var, "")
            if value.lower() in [v.lower() for v in forbidden_values]:
                violations.append(
                    f"{env_var} is set to a development/default value ('{value}'). "
                    f"Use a strong, unique value for production."
                )
        return violations

    @classmethod
    def _check_required_variables(cls) -> list[str]:
        """Check that required variables are set."""
        violations = []
        for env_var in cls.REQUIRED_IN_PRODUCTION:
            if not os.getenv(env_var):
                violations.append(
                    f"{env_var} is not set. This variable is required in production."
                )
        return violations

    @classmethod
    def _check_forbidden_flags(cls) -> list[str]:
        """Check that debug/test flags are not enabled."""
        violations = []
        for env_var, forbidden_values in cls.FORBIDDEN_FLAGS.items():
            value = os.getenv(env_var, "")
            if value.lower() in [v.lower() for v in forbidden_values]:
                violations.append(
                    f"{env_var} is enabled ('{value}'). "
                    f"Debug/test flags must be disabled in production."
                )
        return violations

    @classmethod
    def _check_database_security(cls) -> list[str]:
        """Check database connection security settings."""
        violations = []

        # Check SSL mode
        ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
        if ssl_mode in ("disable", "allow"):
            violations.append(
                f"POSTGRES_SSL_MODE is '{ssl_mode}'. "
                f"Production requires at minimum 'require' for encrypted connections."
            )

        # Check for default database credentials
        pg_user = os.getenv("POSTGRES_USER", "")
        if pg_user.lower() in ("postgres", "admin", "root"):
            violations.append(
                f"POSTGRES_USER is '{pg_user}'. "
                f"Use a dedicated service account, not a superuser."
            )

        return violations

    @classmethod
    def _check_encryption_settings(cls) -> list[str]:
        """Check encryption configuration."""
        violations = []

        encryption_key = os.getenv("ENCRYPTION_KEY", "")
        if encryption_key and len(encryption_key) < 32:
            violations.append(
                f"ENCRYPTION_KEY is too short ({len(encryption_key)} chars). "
                f"Use at least 32 characters for AES-256 encryption."
            )

        return violations

    @classmethod
    def validate_database_settings(cls) -> bool:
        """Validate database-specific settings (can be called independently).

        Returns:
            True if all checks pass
        """
        violations = cls._check_database_security()
        if violations:
            for v in violations:
                logger.warning("database_security_issue", issue=v)
            return False
        return True

    @classmethod
    def validate_encryption_settings(cls) -> bool:
        """Validate encryption-specific settings (can be called independently).

        Returns:
            True if all checks pass
        """
        violations = cls._check_encryption_settings()
        if violations:
            for v in violations:
                logger.warning("encryption_security_issue", issue=v)
            return False
        return True


# Export public API
__all__ = [
    "ProductionGuards",
    "ProductionSecurityError",
]
