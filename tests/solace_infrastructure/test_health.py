"""Unit tests for health check module."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import pytest
from solace_infrastructure.health import (
    HealthMonitor,
    HealthChecker,
    ClientHealthChecker,
    CallableHealthChecker,
    HealthCheckResult,
    ComponentHealth,
    HealthStatus,
    ComponentType,
    create_health_monitor,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_component_types(self):
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.CACHE.value == "cache"
        assert ComponentType.VECTOR_STORE.value == "vector_store"
        assert ComponentType.MESSAGE_QUEUE.value == "message_queue"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_healthy_component(self):
        health = ComponentHealth(
            name="postgres",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.DATABASE,
            latency_ms=5.5
        )
        assert health.is_healthy
        assert health.latency_ms == 5.5

    def test_unhealthy_component(self):
        health = ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            component_type=ComponentType.CACHE,
            message="Connection refused"
        )
        assert not health.is_healthy
        assert health.message == "Connection refused"

    def test_to_dict(self):
        health = ComponentHealth(
            name="weaviate",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.VECTOR_STORE,
            details={"version": "1.25.0"}
        )
        result = health.to_dict()
        assert result["name"] == "weaviate"
        assert result["status"] == "healthy"
        assert result["type"] == "vector_store"
        assert result["details"]["version"] == "1.25.0"


class TestHealthCheckResult:
    """Tests for HealthCheckResult model."""

    def test_default_result(self):
        result = HealthCheckResult()
        assert result.status == HealthStatus.UNKNOWN
        assert result.service_name == "solace-ai"
        assert result.components == []

    def test_healthy_result(self):
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            service_name="test-service",
            uptime_seconds=3600.0
        )
        assert result.is_healthy
        assert result.is_ready
        assert result.uptime_seconds == 3600.0

    def test_degraded_result(self):
        result = HealthCheckResult(status=HealthStatus.DEGRADED)
        assert not result.is_healthy
        assert result.is_ready

    def test_unhealthy_result(self):
        result = HealthCheckResult(status=HealthStatus.UNHEALTHY)
        assert not result.is_healthy
        assert not result.is_ready


class MockHealthCheckable:
    """Mock client implementing HealthCheckable protocol."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy

    async def check_health(self) -> dict[str, any]:
        if self.healthy:
            return {"status": "healthy", "extra": "info"}
        return {"status": "unhealthy", "error": "Connection failed"}


class TestClientHealthChecker:
    """Tests for ClientHealthChecker."""

    @pytest.mark.asyncio
    async def test_healthy_client(self):
        client = MockHealthCheckable(healthy=True)
        checker = ClientHealthChecker(
            name="test-db", component_type=ComponentType.DATABASE,
            client=client, timeout_seconds=5.0
        )
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("extra") == "info"

    @pytest.mark.asyncio
    async def test_unhealthy_client(self):
        client = MockHealthCheckable(healthy=False)
        checker = ClientHealthChecker(
            name="test-cache", component_type=ComponentType.CACHE,
            client=client
        )
        result = await checker.check()
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_with_timeout(self):
        client = MockHealthCheckable(healthy=True)
        checker = ClientHealthChecker(
            name="test", component_type=ComponentType.DATABASE,
            client=client, timeout_seconds=1.0
        )
        result = await checker.check_with_timeout()
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0


class TestCallableHealthChecker:
    """Tests for CallableHealthChecker."""

    @pytest.mark.asyncio
    async def test_healthy_callable(self):
        async def check_fn():
            return {"status": "healthy", "version": "1.0.0"}
        checker = CallableHealthChecker(
            name="custom", component_type=ComponentType.CUSTOM,
            check_fn=check_fn
        )
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_callable(self):
        async def check_fn():
            return {"status": "unhealthy", "error": "Service down"}
        checker = CallableHealthChecker(
            name="custom", component_type=ComponentType.EXTERNAL_API,
            check_fn=check_fn
        )
        result = await checker.check()
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        checker = CallableHealthChecker(
            name="slow", component_type=ComponentType.EXTERNAL_API,
            check_fn=slow_check, timeout_seconds=0.1
        )
        result = await checker.check_with_timeout()
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        async def failing_check():
            raise RuntimeError("Check failed")
        checker = CallableHealthChecker(
            name="failing", component_type=ComponentType.DATABASE,
            check_fn=failing_check, timeout_seconds=1.0
        )
        result = await checker.check_with_timeout()
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        return HealthMonitor(service_name="test-service", version="1.0.0")

    def test_creation(self, monitor):
        assert monitor._service_name == "test-service"
        assert monitor._version == "1.0.0"
        assert monitor.uptime_seconds >= 0

    def test_register_checker(self, monitor):
        client = MockHealthCheckable()
        monitor.register_client("db", ComponentType.DATABASE, client)
        assert len(monitor._checkers) == 1

    def test_register_callable(self, monitor):
        async def check():
            return {"status": "healthy"}
        monitor.register_callable("custom", ComponentType.CUSTOM, check)
        assert len(monitor._checkers) == 1

    def test_unregister(self, monitor):
        client = MockHealthCheckable()
        monitor.register_client("db", ComponentType.DATABASE, client)
        assert monitor.unregister("db")
        assert len(monitor._checkers) == 0

    def test_unregister_not_found(self, monitor):
        assert not monitor.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, monitor):
        client1 = MockHealthCheckable(healthy=True)
        client2 = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client1)
        monitor.register_client("cache", ComponentType.CACHE, client2)
        result = await monitor.check_all()
        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    @pytest.mark.asyncio
    async def test_check_all_with_critical_unhealthy(self, monitor):
        healthy_client = MockHealthCheckable(healthy=True)
        unhealthy_client = MockHealthCheckable(healthy=False)
        monitor.register_client("db", ComponentType.DATABASE, unhealthy_client, critical=True)
        monitor.register_client("cache", ComponentType.CACHE, healthy_client, critical=False)
        result = await monitor.check_all()
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_all_with_noncritical_unhealthy(self, monitor):
        healthy_client = MockHealthCheckable(healthy=True)
        unhealthy_client = MockHealthCheckable(healthy=False)
        monitor.register_client("db", ComponentType.DATABASE, healthy_client, critical=True)
        monitor.register_client("metrics", ComponentType.EXTERNAL_API, unhealthy_client, critical=False)
        result = await monitor.check_all()
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_sequential(self, monitor):
        client1 = MockHealthCheckable(healthy=True)
        client2 = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client1)
        monitor.register_client("cache", ComponentType.CACHE, client2)
        result = await monitor.check_all(parallel=False)
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all_empty(self, monitor):
        result = await monitor.check_all()
        assert result.status == HealthStatus.UNKNOWN
        assert len(result.components) == 0

    @pytest.mark.asyncio
    async def test_liveness_probe(self, monitor):
        result = await monitor.liveness_probe()
        assert result["status"] == "alive"
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_readiness_probe_ready(self, monitor):
        client = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client)
        result = await monitor.readiness_probe()
        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_readiness_probe_not_ready(self, monitor):
        client = MockHealthCheckable(healthy=False)
        monitor.register_client("db", ComponentType.DATABASE, client, critical=True)
        result = await monitor.readiness_probe()
        assert result["ready"] is False

    @pytest.mark.asyncio
    async def test_startup_probe_started(self, monitor):
        client = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client)
        result = await monitor.startup_probe()
        assert result["started"] is True

    @pytest.mark.asyncio
    async def test_startup_probe_with_required_components(self, monitor):
        client = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client)
        result = await monitor.startup_probe(required_components=["db"])
        assert result["started"] is True

    @pytest.mark.asyncio
    async def test_startup_probe_missing_required(self, monitor):
        client = MockHealthCheckable(healthy=False)
        monitor.register_client("db", ComponentType.DATABASE, client)
        result = await monitor.startup_probe(required_components=["db"])
        assert result["started"] is False

    def test_get_last_check_none(self, monitor):
        assert monitor.get_last_check() is None

    @pytest.mark.asyncio
    async def test_get_last_check_cached(self, monitor):
        client = MockHealthCheckable(healthy=True)
        monitor.register_client("db", ComponentType.DATABASE, client)
        await monitor.check_all()
        last = monitor.get_last_check()
        assert last is not None
        assert last.status == HealthStatus.HEALTHY


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_health_monitor(self):
        monitor = create_health_monitor("my-service", "2.0.0")
        assert monitor._service_name == "my-service"
        assert monitor._version == "2.0.0"

    def test_create_health_monitor_defaults(self):
        monitor = create_health_monitor()
        assert monitor._service_name == "solace-ai"
        assert monitor._version == "1.0.0"
