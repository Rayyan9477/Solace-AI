"""Solace-AI Health Check Utilities - Service health monitoring and readiness probes."""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, Protocol
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of infrastructure components."""
    DATABASE = "database"
    CACHE = "cache"
    VECTOR_STORE = "vector_store"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    FILE_STORAGE = "file_storage"
    CUSTOM = "custom"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    component_type: ComponentType
    latency_ms: float = 0.0
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status.value, "type": self.component_type.value,
            "latency_ms": round(self.latency_ms, 2), "message": self.message,
            "details": self.details, "checked_at": self.checked_at.isoformat(),
        }


class HealthCheckResult(BaseModel):
    """Aggregated health check result for the entire service."""
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    service_name: str = Field(default="solace-ai")
    version: str = Field(default="1.0.0")
    uptime_seconds: float = Field(default=0.0)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthCheckable(Protocol):
    """Protocol for components that support health checks."""
    async def check_health(self) -> dict[str, Any]: ...


class HealthChecker(ABC):
    """Abstract base class for health check implementations."""
    def __init__(self, name: str, component_type: ComponentType, timeout_seconds: float = 5.0,
                 critical: bool = True) -> None:
        self.name = name
        self.component_type = component_type
        self.timeout_seconds = timeout_seconds
        self.critical = critical

    @abstractmethod
    async def check(self) -> ComponentHealth:
        """Perform health check and return status."""

    async def check_with_timeout(self) -> ComponentHealth:
        """Perform health check with timeout."""
        start = asyncio.get_running_loop().time()
        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout_seconds)
            result.latency_ms = (asyncio.get_running_loop().time() - start) * 1000
            return result
        except asyncio.TimeoutError:
            elapsed = (asyncio.get_running_loop().time() - start) * 1000
            return ComponentHealth(
                name=self.name, status=HealthStatus.UNHEALTHY, component_type=self.component_type,
                latency_ms=elapsed, message=f"Health check timed out after {self.timeout_seconds}s",
            )
        except Exception as e:
            elapsed = (asyncio.get_running_loop().time() - start) * 1000
            return ComponentHealth(
                name=self.name, status=HealthStatus.UNHEALTHY, component_type=self.component_type,
                latency_ms=elapsed, message=f"Health check failed: {e}",
            )


class ClientHealthChecker(HealthChecker):
    """Health checker for clients implementing HealthCheckable protocol."""
    def __init__(self, name: str, component_type: ComponentType, client: HealthCheckable,
                 timeout_seconds: float = 5.0, critical: bool = True) -> None:
        super().__init__(name, component_type, timeout_seconds, critical)
        self._client = client

    async def check(self) -> ComponentHealth:
        result = await self._client.check_health()
        status_str = result.get("status", "unknown")
        status = HealthStatus(status_str) if status_str in HealthStatus._value2member_map_ else HealthStatus.UNKNOWN
        return ComponentHealth(
            name=self.name, status=status, component_type=self.component_type,
            details={k: v for k, v in result.items() if k != "status"},
        )


class CallableHealthChecker(HealthChecker):
    """Health checker using a callable function."""
    def __init__(self, name: str, component_type: ComponentType,
                 check_fn: Callable[[], Awaitable[dict[str, Any]]],
                 timeout_seconds: float = 5.0, critical: bool = True) -> None:
        super().__init__(name, component_type, timeout_seconds, critical)
        self._check_fn = check_fn

    async def check(self) -> ComponentHealth:
        result = await self._check_fn()
        status_str = result.get("status", "unknown")
        status = HealthStatus(status_str) if status_str in HealthStatus._value2member_map_ else HealthStatus.UNKNOWN
        return ComponentHealth(
            name=self.name, status=status, component_type=self.component_type,
            details={k: v for k, v in result.items() if k != "status"},
        )


class HealthMonitor:
    """Central health monitoring service for all infrastructure components."""

    def __init__(self, service_name: str = "solace-ai", version: str = "1.0.0") -> None:
        self._service_name = service_name
        self._version = version
        self._checkers: list[HealthChecker] = []
        self._start_time = datetime.now(timezone.utc)
        self._last_check: HealthCheckResult | None = None

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    def register(self, checker: HealthChecker) -> None:
        """Register a health checker."""
        self._checkers.append(checker)
        logger.debug("health_checker_registered", name=checker.name, type=checker.component_type.value)

    def register_client(self, name: str, component_type: ComponentType, client: HealthCheckable,
                        timeout_seconds: float = 5.0, critical: bool = True) -> None:
        """Register a client for health checking."""
        checker = ClientHealthChecker(name, component_type, client, timeout_seconds, critical)
        self.register(checker)

    def register_callable(self, name: str, component_type: ComponentType,
                          check_fn: Callable[[], Awaitable[dict[str, Any]]],
                          timeout_seconds: float = 5.0, critical: bool = True) -> None:
        """Register a callable for health checking."""
        checker = CallableHealthChecker(name, component_type, check_fn, timeout_seconds, critical)
        self.register(checker)

    def unregister(self, name: str) -> bool:
        """Unregister a health checker by name."""
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                del self._checkers[i]
                return True
        return False

    async def check_all(self, parallel: bool = True) -> HealthCheckResult:
        """Run all health checks and aggregate results."""
        if parallel:
            results = await asyncio.gather(*[c.check_with_timeout() for c in self._checkers])
        else:
            results = [await c.check_with_timeout() for c in self._checkers]
        overall_status = self._calculate_overall_status(results)
        result = HealthCheckResult(
            status=overall_status, service_name=self._service_name, version=self._version,
            uptime_seconds=self.uptime_seconds, components=[r.to_dict() for r in results],
        )
        self._last_check = result
        self._log_health_status(result, results)
        return result

    def _calculate_overall_status(self, results: list[ComponentHealth]) -> HealthStatus:
        """Calculate overall status from component results."""
        if not results:
            return HealthStatus.UNKNOWN
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r, c in zip(results, self._checkers) if c.critical
        )
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        any_degraded = any(r.status == HealthStatus.DEGRADED for r in results)
        any_unhealthy_noncritical = any(
            r.status == HealthStatus.UNHEALTHY
            for r, c in zip(results, self._checkers) if not c.critical
        )
        if any_degraded or any_unhealthy_noncritical:
            return HealthStatus.DEGRADED
        all_healthy = all(r.status == HealthStatus.HEALTHY for r in results)
        return HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED

    def _log_health_status(self, result: HealthCheckResult, components: list[ComponentHealth]) -> None:
        """Log health check results."""
        unhealthy = [c.name for c in components if c.status == HealthStatus.UNHEALTHY]
        degraded = [c.name for c in components if c.status == HealthStatus.DEGRADED]
        if result.status == HealthStatus.UNHEALTHY:
            logger.error("health_check_unhealthy", status=result.status.value, unhealthy=unhealthy)
        elif result.status == HealthStatus.DEGRADED:
            logger.warning("health_check_degraded", status=result.status.value, degraded=degraded)
        else:
            logger.debug("health_check_healthy", status=result.status.value)

    async def liveness_probe(self) -> dict[str, Any]:
        """Kubernetes liveness probe - basic process health."""
        return {"status": "alive", "uptime_seconds": self.uptime_seconds}

    async def readiness_probe(self) -> dict[str, Any]:
        """Kubernetes readiness probe - full dependency check."""
        result = await self.check_all()
        return {"ready": result.is_ready, "status": result.status.value,
                "components": len(result.components)}

    async def startup_probe(self, required_components: list[str] | None = None) -> dict[str, Any]:
        """Kubernetes startup probe - required components ready."""
        result = await self.check_all()
        if required_components:
            required_healthy = all(
                any(c.get("name") == name and c.get("status") == "healthy" for c in result.components)
                for name in required_components
            )
            return {"started": required_healthy, "status": result.status.value}
        return {"started": result.is_ready, "status": result.status.value}

    def get_last_check(self) -> HealthCheckResult | None:
        """Get cached last health check result."""
        return self._last_check


def create_health_monitor(service_name: str = "solace-ai", version: str = "1.0.0") -> HealthMonitor:
    """Factory function to create a health monitor."""
    return HealthMonitor(service_name=service_name, version=version)
