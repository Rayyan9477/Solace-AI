"""
Solace-AI Feature Flag Management.
Enterprise-grade feature toggles with targeting, rollouts, and A/B testing support.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
from collections.abc import Callable, Awaitable
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class FlagStatus(str, Enum):
    """Feature flag status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    CONDITIONAL = "conditional"


class RolloutStrategy(str, Enum):
    """Rollout strategy types."""
    ALL = "all"
    NONE = "none"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    USER_ATTRIBUTE = "user_attribute"
    GRADUAL = "gradual"


class TargetingOperator(str, Enum):
    """Targeting rule operators."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    MATCHES_REGEX = "matches_regex"


class TargetingRule(BaseModel):
    """Individual targeting rule for feature flags."""
    attribute: str = Field(..., description="User attribute to evaluate")
    operator: TargetingOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    negate: bool = Field(default=False, description="Negate the rule result")

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate rule against user context."""
        actual = context.get(self.attribute)
        if actual is None:
            return self.negate
        result = self._compare(actual)
        return not result if self.negate else result

    def _compare(self, actual: Any) -> bool:
        """Perform comparison based on operator."""
        if self.operator == TargetingOperator.EQUALS:
            return actual == self.value
        if self.operator == TargetingOperator.NOT_EQUALS:
            return actual != self.value
        if self.operator == TargetingOperator.CONTAINS:
            return self.value in str(actual)
        if self.operator == TargetingOperator.NOT_CONTAINS:
            return self.value not in str(actual)
        if self.operator == TargetingOperator.IN_LIST:
            return actual in (self.value if isinstance(self.value, list) else [self.value])
        if self.operator == TargetingOperator.NOT_IN_LIST:
            return actual not in (self.value if isinstance(self.value, list) else [self.value])
        if self.operator == TargetingOperator.GREATER_THAN:
            return float(actual) > float(self.value)
        if self.operator == TargetingOperator.LESS_THAN:
            return float(actual) < float(self.value)
        if self.operator == TargetingOperator.MATCHES_REGEX:
            import re
            return bool(re.match(str(self.value), str(actual)))
        return False


class TargetingGroup(BaseModel):
    """Group of targeting rules with AND/OR logic."""
    rules: list[TargetingRule] = Field(default_factory=list)
    match_all: bool = Field(default=True, description="True for AND, False for OR")
    rollout_percentage: int = Field(default=100, ge=0, le=100)

    def evaluate(self, context: dict[str, Any], user_id: str | None = None) -> bool:
        """Evaluate all rules in the group."""
        if self.rules:
            results = [rule.evaluate(context) for rule in self.rules]
            rules_match = all(results) if self.match_all else any(results)
            if not rules_match:
                return False
        # Check rollout percentage (applies even with no rules)
        if self.rollout_percentage < 100 and user_id:
            return self._in_rollout(user_id)
        return self.rollout_percentage == 100

    def _in_rollout(self, user_id: str) -> bool:
        """Determine if user falls within rollout percentage."""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        return bucket < self.rollout_percentage


class FeatureFlag(BaseModel):
    """Complete feature flag definition."""
    key: str = Field(..., description="Unique flag identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Flag description")
    status: FlagStatus = Field(default=FlagStatus.DISABLED)
    strategy: RolloutStrategy = Field(default=RolloutStrategy.NONE)
    default_value: bool = Field(default=False)
    targeting_groups: list[TargetingGroup] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    blocked_users: list[str] = Field(default_factory=list)
    percentage: int = Field(default=0, ge=0, le=100)
    variants: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(default=None)
    owner: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Key must be alphanumeric with hyphens/underscores")
        return v.lower()

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def evaluate(self, user_id: str | None = None, context: dict[str, Any] | None = None) -> bool:
        """Evaluate flag for given user and context."""
        if self.is_expired:
            return self.default_value
        if self.status == FlagStatus.DISABLED:
            return False
        if self.status == FlagStatus.ENABLED:
            return self._evaluate_strategy(user_id, context or {})
        return self._evaluate_conditional(user_id, context or {})

    def _evaluate_strategy(self, user_id: str | None, context: dict[str, Any]) -> bool:
        """Evaluate based on rollout strategy."""
        if user_id and user_id in self.blocked_users:
            return False
        if user_id and user_id in self.allowed_users:
            return True
        if self.strategy == RolloutStrategy.ALL:
            return True
        if self.strategy == RolloutStrategy.NONE:
            return False
        if self.strategy == RolloutStrategy.PERCENTAGE and user_id:
            return self._in_percentage_rollout(user_id)
        if self.strategy == RolloutStrategy.USER_LIST:
            return user_id in self.allowed_users if user_id else False
        if self.strategy == RolloutStrategy.USER_ATTRIBUTE:
            return self._evaluate_targeting(user_id, context)
        if self.strategy == RolloutStrategy.GRADUAL and user_id:
            return self._in_percentage_rollout(user_id)
        return self.default_value

    def _evaluate_conditional(self, user_id: str | None, context: dict[str, Any]) -> bool:
        """Evaluate conditional targeting rules."""
        if user_id and user_id in self.blocked_users:
            return False
        if user_id and user_id in self.allowed_users:
            return True
        return self._evaluate_targeting(user_id, context)

    def _evaluate_targeting(self, user_id: str | None, context: dict[str, Any]) -> bool:
        """Evaluate targeting groups."""
        if not self.targeting_groups:
            return self.default_value
        for group in self.targeting_groups:
            if group.evaluate(context, user_id):
                return True
        return False

    def _in_percentage_rollout(self, user_id: str) -> bool:
        """Check if user is in percentage rollout."""
        combined = f"{self.key}:{user_id}"
        hash_value = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        bucket = hash_value % 100
        return bucket < self.percentage


class FeatureFlagSettings(BaseSettings):
    """Feature flag service settings."""
    storage_backend: str = Field(default="memory")
    redis_url: str = Field(default="redis://localhost:6379/1")
    cache_ttl_seconds: int = Field(default=60, ge=0)
    sync_interval_seconds: int = Field(default=30, ge=5)
    evaluation_logging: bool = Field(default=True)
    default_enabled: bool = Field(default=False)
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_FLAGS_", env_file=".env", extra="ignore"
    )


class EvaluationResult(BaseModel):
    """Result of flag evaluation with metadata."""
    flag_key: str
    enabled: bool
    variant: str | None = None
    reason: str
    user_id: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureFlagManager:
    """Feature flag management with caching and real-time sync."""

    def __init__(self, settings: FeatureFlagSettings | None = None) -> None:
        self._settings = settings or FeatureFlagSettings()
        self._flags: dict[str, FeatureFlag] = {}
        self._cache: dict[str, tuple[bool, datetime]] = {}
        self._listeners: list[Callable[[str, bool], Awaitable[None]]] = []
        self._sync_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def flags(self) -> dict[str, FeatureFlag]:
        return self._flags.copy()

    async def initialize(self) -> None:
        """Initialize feature flag manager."""
        logger.info("feature_flag_manager_initialized", storage=self._settings.storage_backend)

    async def shutdown(self) -> None:
        """Shutdown feature flag manager."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        self._cache.clear()
        logger.info("feature_flag_manager_shutdown")

    async def register_flag(self, flag: FeatureFlag) -> None:
        """Register a new feature flag."""
        async with self._lock:
            self._flags[flag.key] = flag
            self._invalidate_cache(flag.key)
            logger.info("feature_flag_registered", key=flag.key, status=flag.status.value)

    async def update_flag(self, key: str, updates: dict[str, Any]) -> FeatureFlag | None:
        """Update an existing feature flag."""
        async with self._lock:
            if key not in self._flags:
                return None
            current = self._flags[key]
            updated_data = current.model_dump()
            updated_data.update(updates)
            updated_data["updated_at"] = datetime.now(timezone.utc)
            self._flags[key] = FeatureFlag(**updated_data)
            self._invalidate_cache(key)
            logger.info("feature_flag_updated", key=key)
            return self._flags[key]

    async def delete_flag(self, key: str) -> bool:
        """Delete a feature flag."""
        async with self._lock:
            if key not in self._flags:
                return False
            del self._flags[key]
            self._invalidate_cache(key)
            logger.info("feature_flag_deleted", key=key)
            return True

    async def get_flag(self, key: str) -> FeatureFlag | None:
        """Get feature flag by key."""
        return self._flags.get(key)

    async def list_flags(self, tags: list[str] | None = None,
                         status: FlagStatus | None = None) -> list[FeatureFlag]:
        """List feature flags with optional filtering."""
        result: list[FeatureFlag] = []
        for flag in self._flags.values():
            if status and flag.status != status:
                continue
            if tags and not any(t in flag.tags for t in tags):
                continue
            result.append(flag)
        return result

    def is_enabled(self, key: str, user_id: str | None = None,
                   context: dict[str, Any] | None = None, default: bool | None = None) -> bool:
        """Check if feature flag is enabled for user."""
        flag = self._flags.get(key)
        if flag is None:
            # Don't cache when flag doesn't exist - return default directly
            result = default if default is not None else self._settings.default_enabled
            if self._settings.evaluation_logging:
                logger.debug("feature_flag_not_found", key=key, user_id=user_id, default=result)
            return result
        # Check cache for existing flags
        cache_key = f"{key}:{user_id or 'anonymous'}"
        if cache_key in self._cache:
            cached_value, cached_at = self._cache[cache_key]
            elapsed = (datetime.now(timezone.utc) - cached_at).total_seconds()
            if elapsed < self._settings.cache_ttl_seconds:
                return cached_value
        result = flag.evaluate(user_id, context)
        self._cache[cache_key] = (result, datetime.now(timezone.utc))
        if self._settings.evaluation_logging:
            logger.debug("feature_flag_evaluated", key=key, user_id=user_id, enabled=result)
        return result

    async def evaluate(self, key: str, user_id: str | None = None,
                       context: dict[str, Any] | None = None) -> EvaluationResult:
        """Evaluate flag with detailed result."""
        flag = self._flags.get(key)
        if flag is None:
            return EvaluationResult(
                flag_key=key, enabled=self._settings.default_enabled,
                reason="flag_not_found", user_id=user_id
            )
        if flag.is_expired:
            return EvaluationResult(
                flag_key=key, enabled=flag.default_value,
                reason="flag_expired", user_id=user_id
            )
        enabled = flag.evaluate(user_id, context)
        reason = self._determine_reason(flag, user_id, context, enabled)
        return EvaluationResult(
            flag_key=key, enabled=enabled, reason=reason, user_id=user_id
        )

    def _determine_reason(self, flag: FeatureFlag, user_id: str | None,
                          context: dict[str, Any] | None, enabled: bool) -> str:
        """Determine the reason for evaluation result."""
        if flag.status == FlagStatus.DISABLED:
            return "flag_disabled"
        if user_id and user_id in flag.blocked_users:
            return "user_blocked"
        if user_id and user_id in flag.allowed_users:
            return "user_allowed"
        if flag.strategy == RolloutStrategy.PERCENTAGE:
            return "percentage_rollout"
        if flag.strategy == RolloutStrategy.USER_ATTRIBUTE:
            return "targeting_rule_match" if enabled else "targeting_rule_no_match"
        return "default_strategy"

    def _invalidate_cache(self, key: str) -> None:
        """Invalidate cache entries for a flag."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{key}:")]
        for cache_key in keys_to_remove:
            del self._cache[cache_key]

    def register_listener(self, callback: Callable[[str, bool], Awaitable[None]]) -> None:
        """Register callback for flag changes."""
        self._listeners.append(callback)

    async def bulk_evaluate(self, keys: list[str], user_id: str | None = None,
                            context: dict[str, Any] | None = None) -> dict[str, bool]:
        """Evaluate multiple flags at once."""
        return {key: self.is_enabled(key, user_id, context) for key in keys}


_manager: FeatureFlagManager | None = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get singleton feature flag manager."""
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager()
    return _manager


async def initialize_feature_flags(settings: FeatureFlagSettings | None = None) -> FeatureFlagManager:
    """Initialize feature flag manager."""
    global _manager
    _manager = FeatureFlagManager(settings)
    await _manager.initialize()
    return _manager
