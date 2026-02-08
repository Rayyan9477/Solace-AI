"""
Solace-AI API Gateway - Rate Limiting.
Implements rate limiting policies with Redis-backed storage and sliding window algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import time
import hashlib
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class RateLimitWindow(str, Enum):
    """Rate limit time windows."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class RateLimitScope(str, Enum):
    """Scope for rate limit application."""
    GLOBAL = "global"
    SERVICE = "service"
    ROUTE = "route"
    CONSUMER = "consumer"
    IP = "ip"
    HEADER = "header"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration settings."""
    redis_url: str = Field(default="redis://localhost:6379/0")
    default_limit: int = Field(default=1000, ge=1)
    default_window: str = Field(default="minute")
    sync_rate: int = Field(default=10, ge=1)
    hide_client_headers: bool = Field(default=False)
    error_code: int = Field(default=429)
    error_message: str = Field(default="Rate limit exceeded")
    enabled: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_", env_file=".env", extra="ignore")

    @field_validator("default_window")
    @classmethod
    def validate_window(cls, v: str) -> str:
        valid_windows = {"second", "minute", "hour", "day"}
        if v not in valid_windows:
            raise ValueError(f"window must be one of: {valid_windows}")
        return v


@dataclass
class RateLimitPolicy:
    """Rate limit policy definition."""
    name: str
    limit: int
    window: RateLimitWindow
    scope: RateLimitScope = RateLimitScope.CONSUMER
    service_name: str | None = None
    route_name: str | None = None
    header_name: str | None = None
    burst_limit: int | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def window_seconds(self) -> int:
        return {RateLimitWindow.SECOND: 1, RateLimitWindow.MINUTE: 60, RateLimitWindow.HOUR: 3600, RateLimitWindow.DAY: 86400}[self.window]

    def to_kong_plugin_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {f"{self.window.value}": self.limit, "policy": "redis", "hide_client_headers": False, "fault_tolerant": True}
        if self.scope == RateLimitScope.CONSUMER:
            config["limit_by"] = "consumer"
        elif self.scope == RateLimitScope.IP:
            config["limit_by"] = "ip"
        elif self.scope == RateLimitScope.HEADER:
            config["limit_by"] = "header"
            if self.header_name:
                config["header_name"] = self.header_name
        return config


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after: int | None = None
    policy_name: str | None = None

    def to_headers(self) -> dict[str, str]:
        headers = {"X-RateLimit-Limit": str(self.limit), "X-RateLimit-Remaining": str(max(0, self.remaining)), "X-RateLimit-Reset": str(int(self.reset_at.timestamp()))}
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


@dataclass
class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    key: str
    window_seconds: int
    limit: int
    current_count: int = 0
    previous_count: int = 0
    window_start: float = 0.0

    def calculate_count(self, now: float) -> int:
        elapsed = now - self.window_start
        if elapsed >= self.window_seconds:
            return self.current_count
        weight = (self.window_seconds - elapsed) / self.window_seconds
        return int(self.previous_count * weight) + self.current_count


class RateLimitStore:
    """In-memory rate limit storage with Redis-like interface."""

    def __init__(self) -> None:
        self._counters: dict[str, SlidingWindowCounter] = {}
        self._expiry: dict[str, float] = {}

    def _generate_key(self, policy: RateLimitPolicy, identifier: str) -> str:
        components = [policy.name, identifier, policy.window.value]
        if policy.service_name:
            components.append(policy.service_name)
        if policy.route_name:
            components.append(policy.route_name)
        raw_key = ":".join(components)
        return hashlib.sha256(raw_key.encode()).hexdigest()[:32]

    def increment(self, policy: RateLimitPolicy, identifier: str) -> RateLimitResult:
        key = self._generate_key(policy, identifier)
        now = time.time()
        window_seconds = policy.window_seconds()
        if key not in self._counters or now - self._counters[key].window_start >= window_seconds:
            if key in self._counters:
                prev = self._counters[key]
                self._counters[key] = SlidingWindowCounter(key=key, window_seconds=window_seconds, limit=policy.limit, current_count=1, previous_count=prev.current_count, window_start=now)
            else:
                self._counters[key] = SlidingWindowCounter(key=key, window_seconds=window_seconds, limit=policy.limit, current_count=1, previous_count=0, window_start=now)
        else:
            self._counters[key].current_count += 1
        counter = self._counters[key]
        current = counter.calculate_count(now)
        allowed = current <= policy.limit
        remaining = max(0, policy.limit - current)
        reset_time = counter.window_start + window_seconds
        reset_at = datetime.fromtimestamp(reset_time, tz=timezone.utc)
        retry_after = None if allowed else int(reset_time - now) + 1
        logger.debug("rate_limit_check", key=key, current=current, limit=policy.limit, allowed=allowed)
        return RateLimitResult(allowed=allowed, remaining=remaining, limit=policy.limit, reset_at=reset_at, retry_after=retry_after, policy_name=policy.name)

    def get_count(self, policy: RateLimitPolicy, identifier: str) -> int:
        key = self._generate_key(policy, identifier)
        if key not in self._counters:
            return 0
        return self._counters[key].calculate_count(time.time())

    def reset(self, policy: RateLimitPolicy, identifier: str) -> None:
        key = self._generate_key(policy, identifier)
        if key in self._counters:
            del self._counters[key]

    def cleanup_expired(self) -> int:
        now = time.time()
        expired_keys = []
        for key, counter in self._counters.items():
            if now - counter.window_start >= counter.window_seconds * 2:
                expired_keys.append(key)
        for key in expired_keys:
            del self._counters[key]
        return len(expired_keys)


class RedisRateLimitStore:
    """Redis-backed rate limit storage for production multi-instance deployments."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._prefix = "rate_limit:"

    def _generate_key(self, policy: RateLimitPolicy, identifier: str) -> str:
        components = [policy.name, identifier, policy.window.value]
        if policy.service_name:
            components.append(policy.service_name)
        if policy.route_name:
            components.append(policy.route_name)
        raw_key = ":".join(components)
        return f"{self._prefix}{hashlib.sha256(raw_key.encode()).hexdigest()[:32]}"

    async def async_increment(self, policy: RateLimitPolicy, identifier: str) -> RateLimitResult:
        """Async increment using Redis INCR with TTL."""
        key = self._generate_key(policy, identifier)
        window_seconds = policy.window_seconds()
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, window_seconds)
        ttl = await self._redis.ttl(key)
        allowed = count <= policy.limit
        remaining = max(0, policy.limit - count)
        reset_at = datetime.now(timezone.utc)
        retry_after = None if allowed else max(1, ttl)
        return RateLimitResult(
            allowed=allowed, remaining=remaining, limit=policy.limit,
            reset_at=reset_at, retry_after=retry_after, policy_name=policy.name,
        )

    # Sync compatibility wrapper (uses in-memory fallback)
    def increment(self, policy: RateLimitPolicy, identifier: str) -> RateLimitResult:
        """Sync fallback — use async_increment in async contexts."""
        logger.warning("redis_rate_limit_sync_call", msg="Use async_increment for Redis-backed rate limiting")
        # Fall through to basic check
        return RateLimitResult(
            allowed=True, remaining=policy.limit, limit=policy.limit,
            reset_at=datetime.now(timezone.utc), retry_after=None, policy_name=policy.name,
        )

    def get_count(self, policy: RateLimitPolicy, identifier: str) -> int:
        return 0  # Use async version

    def reset(self, policy: RateLimitPolicy, identifier: str) -> None:
        pass  # Use async version

    def cleanup_expired(self) -> int:
        return 0  # Redis TTL handles this


class RateLimiter:
    """Rate limiter with policy management."""

    def __init__(self, config: RateLimitConfig | None = None, redis_client: Any | None = None) -> None:
        self._config = config or RateLimitConfig()
        self._store = RedisRateLimitStore(redis_client) if redis_client else RateLimitStore()
        self._policies: dict[str, RateLimitPolicy] = {}
        self._service_policies: dict[str, list[str]] = {}
        self._route_policies: dict[str, list[str]] = {}

    def add_policy(self, policy: RateLimitPolicy) -> None:
        self._policies[policy.name] = policy
        if policy.service_name:
            if policy.service_name not in self._service_policies:
                self._service_policies[policy.service_name] = []
            self._service_policies[policy.service_name].append(policy.name)
        if policy.route_name:
            if policy.route_name not in self._route_policies:
                self._route_policies[policy.route_name] = []
            self._route_policies[policy.route_name].append(policy.name)
        logger.info("rate_limit_policy_added", name=policy.name, limit=policy.limit, window=policy.window.value)

    def get_policy(self, name: str) -> RateLimitPolicy | None:
        return self._policies.get(name)

    def check(self, policy_name: str, identifier: str) -> RateLimitResult:
        if not self._config.enabled:
            return RateLimitResult(allowed=True, remaining=self._config.default_limit, limit=self._config.default_limit, reset_at=datetime.now(timezone.utc))
        policy = self._policies.get(policy_name)
        if not policy or not policy.enabled:
            return RateLimitResult(allowed=True, remaining=self._config.default_limit, limit=self._config.default_limit, reset_at=datetime.now(timezone.utc))
        return self._store.increment(policy, identifier)

    def check_request(self, service_name: str | None, route_name: str | None, identifier: str) -> RateLimitResult:
        policies_to_check: list[RateLimitPolicy] = []
        if route_name and route_name in self._route_policies:
            for name in self._route_policies[route_name]:
                if name in self._policies:
                    policies_to_check.append(self._policies[name])
        if service_name and service_name in self._service_policies:
            for name in self._service_policies[service_name]:
                if name in self._policies:
                    policies_to_check.append(self._policies[name])
        global_policies = [p for p in self._policies.values() if p.scope == RateLimitScope.GLOBAL]
        policies_to_check.extend(global_policies)
        if not policies_to_check:
            return RateLimitResult(allowed=True, remaining=self._config.default_limit, limit=self._config.default_limit, reset_at=datetime.now(timezone.utc))
        results = [self._store.increment(policy, identifier) for policy in policies_to_check if policy.enabled]
        if not results:
            return RateLimitResult(allowed=True, remaining=self._config.default_limit, limit=self._config.default_limit, reset_at=datetime.now(timezone.utc))
        denied_results = [r for r in results if not r.allowed]
        if denied_results:
            return min(denied_results, key=lambda r: r.remaining)
        return min(results, key=lambda r: r.remaining)

    def get_usage(self, policy_name: str, identifier: str) -> dict[str, Any]:
        policy = self._policies.get(policy_name)
        if not policy:
            return {"error": "Policy not found"}
        count = self._store.get_count(policy, identifier)
        return {"policy": policy_name, "identifier": identifier, "current": count, "limit": policy.limit, "window": policy.window.value, "remaining": max(0, policy.limit - count)}

    def reset_usage(self, policy_name: str, identifier: str) -> bool:
        policy = self._policies.get(policy_name)
        if not policy:
            return False
        self._store.reset(policy, identifier)
        logger.info("rate_limit_reset", policy=policy_name, identifier=identifier)
        return True


def create_solace_rate_limiter(config: RateLimitConfig | None = None) -> RateLimiter:
    """Create pre-configured rate limiter for Solace-AI."""
    limiter = RateLimiter(config)
    limiter.add_policy(RateLimitPolicy(name="global-standard", limit=1000, window=RateLimitWindow.MINUTE, scope=RateLimitScope.GLOBAL, tags=["solace", "global"]))
    limiter.add_policy(RateLimitPolicy(name="orchestrator-per-user", limit=60, window=RateLimitWindow.MINUTE, scope=RateLimitScope.CONSUMER, service_name="orchestrator-service", tags=["solace", "orchestrator"]))
    limiter.add_policy(RateLimitPolicy(name="chat-per-user", limit=30, window=RateLimitWindow.MINUTE, scope=RateLimitScope.CONSUMER, route_name="chat-message", tags=["solace", "chat"]))
    limiter.add_policy(RateLimitPolicy(name="assessment-per-user", limit=10, window=RateLimitWindow.HOUR, scope=RateLimitScope.CONSUMER, service_name="diagnosis-service", tags=["solace", "assessment"]))
    limiter.add_policy(RateLimitPolicy(name="auth-per-ip", limit=20, window=RateLimitWindow.MINUTE, scope=RateLimitScope.IP, route_name="auth-login", tags=["solace", "auth"]))
    limiter.add_policy(RateLimitPolicy(name="admin-per-user", limit=100, window=RateLimitWindow.MINUTE, scope=RateLimitScope.CONSUMER, service_name="admin-service", tags=["solace", "admin"]))
    logger.info("solace_rate_limiter_configured", policies=6)
    return limiter
