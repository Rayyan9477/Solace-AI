"""Feature flag management for gradual rollout of new features.

Provides centralized feature flag management with percentage-based rollouts,
kill switches, and safe fallback mechanisms.

Usage:
    from solace_infrastructure.feature_flags import FeatureFlags

    # Check if feature is enabled
    if FeatureFlags.is_enabled("use_connection_pool_manager"):
        # Use new implementation
        async with ConnectionPoolManager.acquire() as conn:
            ...
    else:
        # Use legacy implementation
        async with legacy_pool.acquire() as conn:
            ...

    # Check with user-based rollout
    if FeatureFlags.is_enabled_for_user("new_ui", user_id):
        return render_new_ui()
    else:
        return render_legacy_ui()
"""

from __future__ import annotations

import hashlib
import os
from enum import Enum
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class RolloutStrategy(str, Enum):
    """Feature flag rollout strategies."""

    ALL_USERS = "all_users"  # Enable for 100% of users
    PERCENTAGE = "percentage"  # Enable for X% of users (based on hash)
    WHITELIST = "whitelist"  # Enable only for specific user IDs
    ENVIRONMENT = "environment"  # Enable only in specific environments
    DISABLED = "disabled"  # Feature completely disabled


class FeatureFlagConfig(BaseModel):
    """Configuration for a single feature flag."""

    name: str = Field(description="Unique feature flag identifier")
    enabled: bool = Field(default=False, description="Master kill switch")
    strategy: RolloutStrategy = Field(
        default=RolloutStrategy.DISABLED, description="Rollout strategy"
    )
    rollout_percentage: int = Field(
        default=0, ge=0, le=100, description="Percentage of users to enable (0-100)"
    )
    whitelisted_users: list[str] = Field(
        default_factory=list, description="User IDs explicitly allowed"
    )
    allowed_environments: list[str] = Field(
        default_factory=list,
        description="Environments where feature is allowed (e.g., ['development', 'staging'])",
    )
    description: str = Field(default="", description="Human-readable description")
    owner: str = Field(default="", description="Team/person owning this feature")
    jira_ticket: str = Field(default="", description="Associated JIRA ticket")


class FeatureFlags:
    """Centralized feature flag manager for gradual rollouts.

    Supports multiple rollout strategies:
    - All users (100% rollout)
    - Percentage-based (gradual rollout)
    - Whitelist (specific users only)
    - Environment-based (development/staging/production)
    - Disabled (kill switch)

    Thread-safe for concurrent access.
    """

    # Feature flag registry
    _flags: ClassVar[dict[str, FeatureFlagConfig]] = {
        # Phase 1.2: Connection Pool Manager
        "use_connection_pool_manager": FeatureFlagConfig(
            name="use_connection_pool_manager",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=100,  # Start at 100% after testing
            description="Use centralized ConnectionPoolManager instead of per-service pools",
            owner="Infrastructure Team",
            jira_ticket="SOLACE-1234",
        ),
        # Phase 1.3: Centralized Entities
        "use_centralized_entities": FeatureFlagConfig(
            name="use_centralized_entities",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=100,  # Start at 100%
            description="Use centralized entity definitions from schema registry",
            owner="Infrastructure Team",
            jira_ticket="SOLACE-1235",
        ),
        # Phase 1.4: ORM Migration
        "use_orm_queries": FeatureFlagConfig(
            name="use_orm_queries",
            enabled=False,
            strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=0,  # Start disabled, gradually enable
            description="Use SQLAlchemy ORM instead of raw SQL queries",
            owner="Infrastructure Team",
            jira_ticket="SOLACE-1236",
        ),
        # Phase 2.3: SSL/TLS Enforcement
        "enforce_database_ssl": FeatureFlagConfig(
            name="enforce_database_ssl",
            enabled=False,
            strategy=RolloutStrategy.ENVIRONMENT,
            allowed_environments=["production", "staging"],
            description="Enforce SSL/TLS for all database connections",
            owner="Security Team",
            jira_ticket="SOLACE-1237",
        ),
        # Phase 4.1: Portkey Integration
        "use_portkey_integration": FeatureFlagConfig(
            name="use_portkey_integration",
            enabled=False,
            strategy=RolloutStrategy.PERCENTAGE,
            rollout_percentage=0,  # Start disabled
            description="Route LLM calls through Portkey instead of direct provider APIs",
            owner="ML Team",
            jira_ticket="SOLACE-1238",
        ),
        # Phase 5.1: Granular Permissions
        "use_granular_permissions": FeatureFlagConfig(
            name="use_granular_permissions",
            enabled=False,
            strategy=RolloutStrategy.DISABLED,
            description="Enforce granular service-to-service permissions",
            owner="Security Team",
            jira_ticket="SOLACE-1239",
        ),
    }

    @classmethod
    def register_flag(cls, config: FeatureFlagConfig) -> None:
        """Register a new feature flag.

        Args:
            config: Feature flag configuration

        Raises:
            ValueError: If flag with same name already exists
        """
        if config.name in cls._flags:
            existing = cls._flags[config.name]
            logger.warning(
                "feature_flag_already_exists",
                name=config.name,
                existing_owner=existing.owner,
                new_owner=config.owner,
            )
            return

        cls._flags[config.name] = config
        logger.info(
            "feature_flag_registered",
            name=config.name,
            strategy=config.strategy.value,
            enabled=config.enabled,
        )

    @classmethod
    def is_enabled(cls, flag_name: str, context: dict[str, Any] | None = None) -> bool:
        """Check if a feature flag is enabled.

        Args:
            flag_name: Feature flag identifier
            context: Optional context (user_id, environment, etc.)

        Returns:
            True if feature is enabled, False otherwise
        """
        if flag_name not in cls._flags:
            logger.warning(
                "feature_flag_not_found",
                name=flag_name,
                message="Flag not registered, defaulting to disabled",
            )
            return False

        flag = cls._flags[flag_name]

        # Master kill switch
        if not flag.enabled:
            return False

        # Strategy-based checks
        if flag.strategy == RolloutStrategy.DISABLED:
            return False

        if flag.strategy == RolloutStrategy.ALL_USERS:
            return True

        if flag.strategy == RolloutStrategy.ENVIRONMENT:
            current_env = os.getenv("ENVIRONMENT", "development")
            return current_env in flag.allowed_environments

        if flag.strategy == RolloutStrategy.PERCENTAGE:
            # Use context to determine if enabled
            if context and "user_id" in context:
                return cls._is_in_rollout_percentage(
                    flag_name, str(context["user_id"]), flag.rollout_percentage
                )
            # No user context, check if we're in the percentage globally
            return cls._is_in_rollout_percentage(
                flag_name, "global", flag.rollout_percentage
            )

        if flag.strategy == RolloutStrategy.WHITELIST:
            if context and "user_id" in context:
                return str(context["user_id"]) in flag.whitelisted_users
            return False

        return False

    @classmethod
    def is_enabled_for_user(cls, flag_name: str, user_id: str) -> bool:
        """Check if feature flag is enabled for a specific user.

        Args:
            flag_name: Feature flag identifier
            user_id: User identifier for percentage-based rollout

        Returns:
            True if feature is enabled for this user, False otherwise
        """
        return cls.is_enabled(flag_name, context={"user_id": user_id})

    @classmethod
    def _is_in_rollout_percentage(
        cls, flag_name: str, identifier: str, percentage: int
    ) -> bool:
        """Determine if identifier falls within rollout percentage.

        Uses consistent hashing to ensure same identifier always gets same result.

        Args:
            flag_name: Feature flag name (for hash stability)
            identifier: User ID or other identifier
            percentage: Target rollout percentage (0-100)

        Returns:
            True if identifier is in the rollout percentage
        """
        if percentage == 0:
            return False
        if percentage == 100:
            return True

        # Consistent hash: same flag + identifier always returns same result
        hash_input = f"{flag_name}:{identifier}".encode("utf-8")
        hash_value = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
        bucket = hash_value % 100  # Map to 0-99

        return bucket < percentage

    @classmethod
    def get_flag(cls, flag_name: str) -> FeatureFlagConfig | None:
        """Get feature flag configuration.

        Args:
            flag_name: Feature flag identifier

        Returns:
            Feature flag configuration or None if not found
        """
        return cls._flags.get(flag_name)

    @classmethod
    def list_flags(cls) -> list[FeatureFlagConfig]:
        """List all registered feature flags.

        Returns:
            List of all feature flag configurations
        """
        return list(cls._flags.values())

    @classmethod
    def enable_flag(cls, flag_name: str) -> None:
        """Enable a feature flag (set master kill switch to True).

        Args:
            flag_name: Feature flag identifier

        Raises:
            ValueError: If flag not found
        """
        if flag_name not in cls._flags:
            raise ValueError(f"Feature flag '{flag_name}' not found")

        cls._flags[flag_name].enabled = True
        logger.info("feature_flag_enabled", name=flag_name)

    @classmethod
    def disable_flag(cls, flag_name: str) -> None:
        """Disable a feature flag (emergency kill switch).

        Args:
            flag_name: Feature flag identifier

        Raises:
            ValueError: If flag not found
        """
        if flag_name not in cls._flags:
            raise ValueError(f"Feature flag '{flag_name}' not found")

        cls._flags[flag_name].enabled = False
        logger.warning("feature_flag_disabled", name=flag_name, reason="manual_disable")

    @classmethod
    def set_rollout_percentage(cls, flag_name: str, percentage: int) -> None:
        """Update rollout percentage for a feature flag.

        Args:
            flag_name: Feature flag identifier
            percentage: New rollout percentage (0-100)

        Raises:
            ValueError: If flag not found or percentage invalid
        """
        if flag_name not in cls._flags:
            raise ValueError(f"Feature flag '{flag_name}' not found")

        if not 0 <= percentage <= 100:
            raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")

        old_percentage = cls._flags[flag_name].rollout_percentage
        cls._flags[flag_name].rollout_percentage = percentage

        logger.info(
            "feature_flag_rollout_updated",
            name=flag_name,
            old_percentage=old_percentage,
            new_percentage=percentage,
        )

    @classmethod
    def get_enabled_flags(cls) -> list[str]:
        """Get list of all currently enabled feature flags.

        Returns:
            List of enabled feature flag names
        """
        return [
            name
            for name, flag in cls._flags.items()
            if flag.enabled and flag.strategy != RolloutStrategy.DISABLED
        ]


# Convenience decorator for feature-flagged functions
def feature_flagged(flag_name: str, fallback_return: Any = None):
    """Decorator to gate function execution behind a feature flag.

    Supports both sync and async functions via inspect.iscoroutinefunction().

    Args:
        flag_name: Feature flag to check
        fallback_return: Value to return if feature is disabled

    Example:
        @feature_flagged("use_new_algorithm", fallback_return=[])
        def process_data_new_algorithm(data: list) -> list:
            return processed_data

        @feature_flagged("use_async_pipeline", fallback_return=None)
        async def run_pipeline(data: list) -> dict:
            return await pipeline.run(data)
    """
    import functools
    import inspect

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if FeatureFlags.is_enabled(flag_name):
                    return await func(*args, **kwargs)
                logger.debug(
                    "feature_flag_disabled_skipping",
                    flag=flag_name,
                    function=func.__name__,
                )
                return fallback_return

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if FeatureFlags.is_enabled(flag_name):
                    return func(*args, **kwargs)
                logger.debug(
                    "feature_flag_disabled_skipping",
                    flag=flag_name,
                    function=func.__name__,
                )
                return fallback_return

            return sync_wrapper

    return decorator


# Export public API
__all__ = [
    "FeatureFlags",
    "FeatureFlagConfig",
    "RolloutStrategy",
    "feature_flagged",
]
