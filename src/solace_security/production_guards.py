"""Production environment guards for Solace-AI.

Strict environments (``staging`` and ``production``; see ``is_strict_environment``) must
provide the secrets the services actually read, must not run with development values or
debug flags, and must talk to Postgres over ``verify-full`` TLS as a non-superuser.
Validation runs at startup and raises ``ProductionSecurityError`` on any violation.

Every violation string starts with the variable name followed by a space and never
contains the variable's value, so neither the exception nor the log line leaks a secret.

Usage:
    # Call once during service startup with the service's own resolved environment
    from solace_security.production_guards import ProductionGuards

    ProductionGuards.assert_startup(settings.environment, service_name="safety-service")

    # A service that reads fewer secrets names exactly what it reads
    ProductionGuards.assert_startup(
        settings.environment,
        service_name="analytics-service",
        required_vars=("AUTH_SECRET_KEY", "POSTGRES_PASSWORD"),
    )

    # Or validate specific settings independently
    ProductionGuards.validate_database_settings()
    ProductionGuards.validate_encryption_settings()
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection
from typing import ClassVar
from urllib.parse import unquote, urlsplit

import structlog

logger = structlog.get_logger(__name__)

# Environment names that are strict: staging is exactly as strict as production (D-14).
STRICT_ENVIRONMENT_NAMES: frozenset[str] = frozenset({"production", "staging"})

# The secrets the services read in a strict environment, used when a caller does not
# name its own set (MVP-031: the guard used to require SECRET_KEY/ENCRYPTION_KEY, which
# nothing reads or provides).
DEFAULT_REQUIRED_IN_STRICT: tuple[str, ...] = (
    "AUTH_SECRET_KEY",
    "ENCRYPTION_MASTER_KEY",
    "ENCRYPTION_SEARCH_HASH_SALT",
    "POSTGRES_PASSWORD",
)


def is_strict_environment(env: str | None = None) -> bool:
    """Return True iff ``env`` names a strict environment (staging or production).

    When ``env`` is None the bare ``ENVIRONMENT`` variable is read, defaulting to
    ``development``. Surrounding whitespace and case are ignored.
    """
    if env is None:
        env = os.environ.get("ENVIRONMENT", "development")
    return env.strip().lower() in STRICT_ENVIRONMENT_NAMES


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
    """Validates environment configuration for strict (staging/production) safety.

    Checks for:
    - Missing required secrets (the caller's set, else ``DEFAULT_REQUIRED_IN_STRICT``)
    - Known development/default values for those secrets
    - Debug/test flags enabled
    - Postgres not on ``verify-full`` TLS, or connecting as a superuser
    - An encryption master key that does not parse as exactly 32 bytes (``parse_master_key(strict=True)``)
    """

    # Values a required secret must NOT have in a strict environment (compared after
    # stripping whitespace, case-insensitively). Only names in the required set are checked.
    # Includes the repo's own dev defaults (docker-compose.yml, encryption.py) and the
    # .env.example placeholders, so a copied example or dev default never passes as strict.
    FORBIDDEN_VALUES: ClassVar[dict[str, list[str]]] = {
        "AUTH_SECRET_KEY": [
            "dev", "development", "test", "secret", "changeme", "default",
            "dev-jwt-secret-key-minimum-32-chars",
            "your-secure-auth-secret-key-32-chars",
            "dev-only-insecure-key-32-bytes!!",
        ],
        "ENCRYPTION_MASTER_KEY": [
            "dev", "development", "test", "secret", "changeme", "default",
            "dev-only-insecure-key-32-bytes!!",
            "your-secure-encryption-key-32-chars",
        ],
        "ENCRYPTION_SEARCH_HASH_SALT": ["dev", "development", "test", "salt", "changeme", "default"],
        "POSTGRES_PASSWORD": [
            "postgres", "password", "dev", "test", "changeme",
            "solace_dev_password",
            "your-secure-database-password",
            # docker-compose.yml dev defaults of the three platform roles (WS-03.2)
            "solace_app_dev_only", "solace_migrator_dev_only", "solace_superuser_dev_only",
        ],
    }

    # Variables that MUST be set in a strict environment when the caller names none.
    REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = DEFAULT_REQUIRED_IN_STRICT

    # Settings that must NOT be enabled in a strict environment
    FORBIDDEN_FLAGS: ClassVar[dict[str, list[str]]] = {
        "DEBUG": ["true", "1", "yes"],
        "TESTING": ["true", "1", "yes"],
        "MOCK_SERVICES": ["true", "1", "yes"],
        "USE_IN_MEMORY_DB": ["true", "1", "yes"],
    }

    # Strict Postgres clients verify the server certificate and host name (D-15).
    REQUIRED_POSTGRES_SSL_MODE: ClassVar[str] = "verify-full"

    # Role names that are superusers or default administrative accounts.
    SUPERUSER_NAMES: ClassVar[frozenset[str]] = frozenset({"postgres", "admin", "root"})


    @classmethod
    def assert_environment_consistency(
        cls, service_environment: str | None, *, service_name: str | None = None
    ) -> None:
        """Fail loud when a service's own resolved environment disagrees with bare ENVIRONMENT.

        REV-32 (code half): each service resolves ``settings.environment`` from its OWN
        prefixed var (e.g. ``SAFETY_ENVIRONMENT`` / ``MEMORY_ENVIRONMENT``). If the prefixed
        var is unset/misnamed the field falls back to its ``development`` default while the
        deployment's bare ``ENVIRONMENT`` says ``production`` — the production fail-loud guards
        (PHI encryption, log sanitizer, SSL floor) then silently no-op. This reconciles the two.

        Behavior:
        - Either side unset => no-op (dev convenience; nothing to reconcile).
        - Case-insensitive match => no-op.
        - Mismatch where EITHER side is strict (``is_strict_environment``) => raise
          ``ProductionSecurityError``.
        - Any other mismatch (e.g. test vs development) => warn only, never crash startup.

        Args:
            service_environment: The environment the service resolved (``settings.environment``).
            service_name: Optional service label for clearer diagnostics.

        Raises:
            ProductionSecurityError: On a production/staging-relevant mismatch.
        """
        bare = os.getenv("ENVIRONMENT")
        if not bare or not service_environment:
            return

        bare_norm = bare.strip().lower()
        svc_norm = service_environment.strip().lower()
        if bare_norm == svc_norm:
            return

        message = (
            f"Environment mismatch: service resolved environment '{service_environment}' "
            f"but the deployment's bare ENVIRONMENT is '{bare}'"
            + (f" (service: {service_name})" if service_name else "")
            + ". The service reads its own prefixed *_ENVIRONMENT var; a misnamed/unset value "
            "silently pins it to the development default, disabling the production guards. "
            "Set the service's prefixed ENVIRONMENT var to match the bare ENVIRONMENT."
        )

        if is_strict_environment(bare) or is_strict_environment(service_environment):
            logger.critical(
                "environment_consistency_violation",
                bare_environment=bare,
                service_environment=service_environment,
                service_name=service_name,
            )
            raise ProductionSecurityError([message])

        logger.warning(
            "environment_consistency_mismatch",
            bare_environment=bare,
            service_environment=service_environment,
            service_name=service_name,
        )

    @classmethod
    def assert_startup(
        cls,
        service_environment: str | None,
        *,
        service_name: str | None = None,
        required_vars: Collection[str] | None = None,
    ) -> None:
        """One-call startup guard for a FastAPI service (REV-32 wiring).

        Call this ONCE during each service's startup (FastAPI lifespan) with the service's
        own resolved environment (``settings.environment`` / ``settings.service.env``). It
        runs, in order:

        1. ``assert_environment_consistency`` — fail loud when the service's resolved
           environment disagrees with the deployment's bare ``ENVIRONMENT`` and either side
           is strict. This is the REV-32 gap: a misnamed/unset prefixed ``*_ENVIRONMENT``
           var silently pins the service to its ``development`` default, so the OTHER
           fail-loud guards (PHI encryption, log sanitizer, SSL floor) never engage.
        2. ``validate_environment(required_vars=...)`` — the strict-secret guard. It
           self-skips outside staging/production, so it is inert in dev/test.

        Before both, a service that resolved a strict environment must also see a non-blank
        bare ``ENVIRONMENT``: the consistency check is a no-op when the bare side is unset and
        ``validate_environment`` keys on the bare side, so without this a strict service with
        no bare ``ENVIRONMENT`` would skip every check.

        All steps no-op / self-skip in dev/test, so this call is safe to run unconditionally
        at every service startup.

        Args:
            service_environment: The environment the service resolved for itself.
            service_name: Optional service label for clearer diagnostics.
            required_vars: The secret names this service reads. None means
                ``DEFAULT_REQUIRED_IN_STRICT``.

        Raises:
            ProductionSecurityError: When a strict service has no bare ``ENVIRONMENT``, on a
                strict-relevant env mismatch, or when the strict-secret validation finds
                violations.
        """
        bare = os.getenv("ENVIRONMENT", "")
        if is_strict_environment(bare) and not (service_environment or "").strip():
            # The mirror case: the consistency check returns early on a blank service side, so a
            # blank prefixed *_ENVIRONMENT under a strict deployment would turn off every
            # service-owned strict-only guard while the runtime itself runs strict (WS121F1-CLO-1).
            logger.critical(
                "environment_consistency_violation",
                bare_environment=bare,
                service_environment=service_environment,
                service_name=service_name,
            )
            raise ProductionSecurityError(
                [
                    "The service's own environment is blank while the bare ENVIRONMENT is strict. "
                    "Set the service's prefixed ENVIRONMENT var to match the bare ENVIRONMENT."
                ]
            )
        if is_strict_environment(service_environment or "") and not bare.strip():
            logger.critical(
                "environment_consistency_violation",
                bare_environment=bare,
                service_environment=service_environment,
                service_name=service_name,
            )
            raise ProductionSecurityError(
                [
                    "ENVIRONMENT is not set. A service that resolves a strict environment "
                    "requires the bare ENVIRONMENT to match it."
                ]
            )
        cls.assert_environment_consistency(service_environment, service_name=service_name)
        cls.validate_environment(required_vars=required_vars)

    @classmethod
    def validate_environment(cls, *, required_vars: Collection[str] | None = None) -> None:
        """Run all strict-environment validations; a no-op outside staging/production.

        Args:
            required_vars: The secret names that must be set. None means
                ``DEFAULT_REQUIRED_IN_STRICT``. Forbidden-value checks run only for these
                names, and the Postgres checks run only when ``POSTGRES_PASSWORD`` is one
                of them.

        Raises:
            ProductionSecurityError: If any violations are detected in a strict environment.
        """
        environment = os.getenv("ENVIRONMENT", "development")
        if not is_strict_environment(environment):
            logger.debug(
                "production_guards_skipped",
                environment=environment,
                reason="Not a production/staging environment",
            )
            return

        required = cls._resolve_required(required_vars)
        checks: list[str] = ["required_variables", "forbidden_values", "forbidden_flags"]
        violations: list[str] = []

        violations.extend(cls._check_required_variables(required))
        violations.extend(cls._check_forbidden_values(required))
        violations.extend(cls._check_redis_url(required))
        violations.extend(cls._check_forbidden_flags())
        if "POSTGRES_PASSWORD" in required:
            checks.append("database_security")
            violations.extend(cls._check_database_security())
        if "ENCRYPTION_MASTER_KEY" in required:
            checks.append("encryption_settings")
            violations.extend(cls._check_encryption_settings())

        if violations:
            logger.critical(
                "production_security_validation_failed",
                environment=environment,
                violation_count=len(violations),
                violations=violations,
            )
            raise ProductionSecurityError(violations)

        logger.info(
            "production_security_validation_passed",
            environment=environment,
            checks_passed=checks,
            required_vars=list(required),
        )

    @staticmethod
    def _resolve_required(required_vars: Collection[str] | None) -> tuple[str, ...]:
        """Return the required names: the caller's (de-duplicated, in order) or the default."""
        if required_vars is None:
            return DEFAULT_REQUIRED_IN_STRICT
        if isinstance(required_vars, str):
            raise TypeError(
                "required_vars must be a collection of variable names, not a single string"
            )
        return tuple(dict.fromkeys(required_vars))

    @staticmethod
    def _is_set(name: str) -> bool:
        return bool(os.getenv(name, "").strip())

    @classmethod
    def _check_required_variables(cls, required: Collection[str]) -> list[str]:
        """Check that every required variable is set to a non-blank value."""
        return [
            f"{name} is not set. Strict environments (staging, production) require it."
            for name in required
            if not cls._is_set(name)
        ]

    @classmethod
    def _check_forbidden_values(cls, required: Collection[str]) -> list[str]:
        """Check required variables for known development/default values (never echoed)."""
        violations = []
        for name in required:
            forbidden = cls.FORBIDDEN_VALUES.get(name)
            if not forbidden or not cls._is_set(name):
                continue
            value = os.getenv(name, "").strip().lower()
            if value in {v.lower() for v in forbidden}:
                violations.append(
                    f"{name} is set to a known development/default value. "
                    "Use a strong, unique value in strict environments."
                )
        return violations

    # Redis passwords a strict REDIS_URL must not carry: the compose dev default (WS-03.3) and
    # common placeholders. Compared after unquoting and stripping, case-insensitively.
    FORBIDDEN_REDIS_PASSWORDS: ClassVar[frozenset[str]] = frozenset({
        "solace_redis_dev_only", "redis", "password", "changeme", "dev", "test", "default",
    })
    MIN_REDIS_PASSWORD_LENGTH: ClassVar[int] = 16
    _REDIS_PASSWORD_FORMAT: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]{16,}")

    @classmethod
    def _check_redis_url(cls, required: Collection[str]) -> list[str]:
        """A required REDIS_URL must carry a real password in strict (value never echoed)."""
        if "REDIS_URL" not in required or not cls._is_set("REDIS_URL"):
            return []
        invalid = ["REDIS_URL is not a valid URL. Use redis://:<password>@<host>:<port>/<db>."]
        try:
            parts = urlsplit(os.getenv("REDIS_URL", "").strip())
            raw, host = parts.password, parts.hostname
        except ValueError:
            return invalid
        if not host:
            return invalid
        password = unquote(raw or "").strip()
        if not password:
            return ["REDIS_URL has no password. Strict environments require Redis authentication."]
        if password.lower() in cls.FORBIDDEN_REDIS_PASSWORDS:
            return [
                "REDIS_URL uses a known development/default password. "
                "Use a strong, unique value in strict environments."
            ]
        # The registry format is urlsafe24; redis.conf and every client URL take the value raw.
        if raw != password or not cls._REDIS_PASSWORD_FORMAT.fullmatch(password):
            return [
                "REDIS_URL password must be URL-safe ([A-Za-z0-9_-]) and at least "
                f"{cls.MIN_REDIS_PASSWORD_LENGTH} characters."
            ]
        return []

    @classmethod
    def _check_forbidden_flags(cls) -> list[str]:
        """Check that debug/test flags are not enabled (values never echoed)."""
        violations = []
        for name, enabled_values in cls.FORBIDDEN_FLAGS.items():
            value = os.getenv(name, "").strip().lower()
            if value in {v.lower() for v in enabled_values}:
                violations.append(
                    f"{name} is enabled. Debug/test flags must be disabled in strict environments."
                )
        return violations

    @classmethod
    def _check_database_security(cls) -> list[str]:
        """Check Postgres TLS mode and role (values never echoed)."""
        violations = []

        if os.getenv("POSTGRES_SSL_MODE", "prefer") != cls.REQUIRED_POSTGRES_SSL_MODE:
            violations.append(
                f"POSTGRES_SSL_MODE is not {cls.REQUIRED_POSTGRES_SSL_MODE}. Strict environments "
                "require TLS with server certificate and host-name verification."
            )

        if os.getenv("POSTGRES_USER", "").strip().lower() in cls.SUPERUSER_NAMES:
            violations.append(
                "POSTGRES_USER is a superuser or default administrative account. "
                "Use a dedicated non-superuser service account."
            )

        return violations

    @classmethod
    def _check_encryption_settings(cls) -> list[str]:
        """Check the master key parses exactly as EncryptionSettings will (CR-2; value never echoed)."""
        raw = os.getenv("ENCRYPTION_MASTER_KEY", "")
        if not raw.strip():
            return []  # absence is reported by the required-variables check
        from solace_security.encryption import parse_master_key

        try:
            # The raw value, unstripped: EncryptionSettings parses it unstripped too, so a key
            # with stray whitespace fails here at startup instead of later at PHI setup.
            parse_master_key(raw, strict=True)
        except ValueError:
            return [
                "ENCRYPTION_MASTER_KEY is not a valid 32-byte key. Use 64 hex characters "
                "(secrets.token_hex(32)) or 'base64:' followed by 32 base64-encoded bytes."
            ]
        return []

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
    "DEFAULT_REQUIRED_IN_STRICT",
    "STRICT_ENVIRONMENT_NAMES",
    "ProductionGuards",
    "ProductionSecurityError",
    "is_strict_environment",
]
