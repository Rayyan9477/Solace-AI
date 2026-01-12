"""Comprehensive Infrastructure Tests - Batch 2 Extended Coverage.

This module provides exhaustive coverage for:
- Database connections and pooling
- Redis cache operations
- Weaviate vector operations
- Health checks and monitoring
- Error handling and resilience
"""
from __future__ import annotations
import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from solace_infrastructure.postgres import (
    PostgresSettings, IsolationLevel, QueryResult,
    PostgresClient, PostgresRepository, create_postgres_client,
)
from solace_infrastructure.redis import (
    RedisMode,
    RedisSettings, RedisClient, create_redis_client,
)
from solace_infrastructure.weaviate import (
    WeaviateSettings, PropertyDataType, VectorDistanceMetric,
    PropertyConfig, CollectionConfig, SearchResult,
    WeaviateClient, create_weaviate_client,
)
from solace_infrastructure.health import (
    HealthStatus, ComponentType, ComponentHealth, HealthCheckResult,
    HealthChecker, ClientHealthChecker, CallableHealthChecker,
    HealthMonitor, create_health_monitor,
)


# =============================================================================
# PostgreSQL Infrastructure Tests
# =============================================================================

class TestPostgresSettings:
    """Tests for PostgresSettings."""

    def test_default_settings(self) -> None:
        """Test default Postgres settings."""
        settings = PostgresSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "solace"

    def test_custom_settings(self) -> None:
        """Test custom Postgres settings."""
        settings = PostgresSettings(
            host="db.example.com",
            port=5433,
            database="custom_db",
        )
        assert settings.host == "db.example.com"
        assert settings.port == 5433
        assert settings.database == "custom_db"

    def test_dsn_format(self) -> None:
        """Test DSN format."""
        settings = PostgresSettings(
            host="db.example.com",
            port=5433,
            database="test_db",
            user="test_user",
        )
        dsn = settings.get_dsn()
        assert "postgresql" in dsn
        assert "db.example.com" in dsn
        assert "5433" in dsn
        assert "test_db" in dsn
        assert "test_user" in dsn

    def test_pool_size_boundaries(self) -> None:
        """Test pool size boundaries."""
        settings_min = PostgresSettings(min_pool_size=1, max_pool_size=1)
        assert settings_min.min_pool_size == 1

        settings_max = PostgresSettings(min_pool_size=50, max_pool_size=100)
        assert settings_max.max_pool_size == 100

    def test_ssl_mode_options(self) -> None:
        """Test SSL mode options."""
        for mode in ["disable", "prefer", "require", "verify-ca", "verify-full"]:
            settings = PostgresSettings(ssl_mode=mode)
            assert settings.ssl_mode == mode


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_all_isolation_levels(self) -> None:
        """Test all isolation levels exist."""
        expected = {"read_committed", "repeatable_read", "serializable"}
        actual = {level.value for level in IsolationLevel}
        assert actual == expected


class TestQueryResult:
    """Tests for QueryResult."""

    def test_empty_result(self) -> None:
        """Test empty query result."""
        result = QueryResult()
        assert result.rows == []
        assert result.row_count == 0
        assert result.first is None
        assert result.scalar is None

    def test_result_with_rows(self) -> None:
        """Test query result with rows."""
        result = QueryResult(
            rows=[{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}],
            row_count=2,
            execution_time_ms=5.5,
        )
        assert result.row_count == 2
        assert result.first == {"id": 1, "name": "test"}

    def test_scalar_property(self) -> None:
        """Test scalar property."""
        result = QueryResult(rows=[{"count": 42}], row_count=1)
        assert result.scalar == 42


class TestPostgresClient:
    """Tests for PostgresClient."""

    @pytest.fixture
    def client(self) -> PostgresClient:
        return PostgresClient(PostgresSettings())

    def test_initial_state(self, client) -> None:
        """Test client initial state."""
        assert client._pool is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_not_connected_error(self, client) -> None:
        """Test error when not connected."""
        from solace_common.exceptions import DatabaseError
        with pytest.raises(DatabaseError, match="not connected"):
            client._ensure_connected()


class TestPostgresRepository:
    """Tests for PostgresRepository."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        return MagicMock(spec=PostgresClient)

    def test_qualified_table_name(self, mock_client) -> None:
        """Test qualified table name."""
        repo = PostgresRepository(mock_client, "users", "public")
        assert repo.qualified_table == "public.users"

    def test_custom_schema(self, mock_client) -> None:
        """Test custom schema."""
        repo = PostgresRepository(mock_client, "sessions", "memory")
        assert repo.qualified_table == "memory.sessions"


# =============================================================================
# Redis Infrastructure Tests
# =============================================================================

class TestRedisSettings:
    """Tests for RedisSettings."""

    def test_default_settings(self) -> None:
        """Test default Redis settings."""
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.db == 0

    def test_custom_settings(self) -> None:
        """Test custom Redis settings."""
        settings = RedisSettings(host="redis.example.com", port=6380, db=5)
        assert settings.host == "redis.example.com"
        assert settings.port == 6380
        assert settings.db == 5

    def test_cluster_mode(self) -> None:
        """Test cluster mode settings."""
        settings = RedisSettings(mode=RedisMode.CLUSTER, cluster_nodes="node1:6379,node2:6379")
        assert settings.mode == RedisMode.CLUSTER
        assert "node1" in settings.cluster_nodes

    def test_ssl_settings(self) -> None:
        """Test SSL settings."""
        settings = RedisSettings(ssl=True)
        assert settings.ssl is True


class TestRedisClient:
    """Tests for RedisClient."""

    @pytest.fixture
    def client(self) -> RedisClient:
        return RedisClient(RedisSettings())

    def test_initial_state(self, client) -> None:
        """Test client initial state."""
        assert client.is_connected is False


# =============================================================================
# Weaviate Infrastructure Tests
# =============================================================================

class TestWeaviateSettings:
    """Tests for WeaviateSettings."""

    def test_default_settings(self) -> None:
        """Test default Weaviate settings."""
        settings = WeaviateSettings()
        assert settings.host == "localhost"
        assert settings.port == 8080
        assert settings.grpc_port == 50051

    def test_get_url_local(self) -> None:
        """Test URL for local instance."""
        settings = WeaviateSettings(host="localhost", port=8080)
        url = settings.get_url()
        assert "localhost" in url
        assert "8080" in url

    def test_get_url_cloud(self) -> None:
        """Test URL for cloud instance."""
        settings = WeaviateSettings(cluster_url="https://my-cluster.weaviate.network")
        url = settings.get_url()
        assert "my-cluster.weaviate.network" in url

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = WeaviateSettings(
            host="weaviate.example.com",
            port=8081,
            grpc_port=50052,
            
            
        )
        assert settings.host == "weaviate.example.com"
        assert settings.port == 8081
        


class TestPropertyDataType:
    """Tests for PropertyDataType enum."""

    def test_data_types(self) -> None:
        """Test all data types exist."""
        expected = {"text", "text[]", "int", "int[]", "number", "number[]",
                    "boolean", "boolean[]", "date", "date[]", "uuid", "uuid[]",
                    "blob", "object", "object[]"}
        actual = {t.value for t in PropertyDataType}
        assert expected.issubset(actual)


class TestVectorDistanceMetric:
    """Tests for VectorDistanceMetric enum."""

    def test_distance_metrics(self) -> None:
        """Test all distance metrics exist."""
        expected = {"cosine", "dot", "l2-squared", "hamming", "manhattan"}
        actual = {m.value for m in VectorDistanceMetric}
        assert expected == actual


class TestPropertyConfig:
    """Tests for PropertyConfig dataclass."""

    def test_basic_property(self) -> None:
        """Test basic property config."""
        prop = PropertyConfig(
            name="title",
            data_type=PropertyDataType.TEXT,
        )
        assert prop.name == "title"
        assert prop.data_type == PropertyDataType.TEXT
        assert prop.skip_vectorization is False

    def test_property_with_options(self) -> None:
        """Test property with all options."""
        prop = PropertyConfig(
            name="embedding_text",
            data_type=PropertyDataType.TEXT,
            skip_vectorization=True,
            tokenization="word",
            index_filterable=True,
            index_searchable=True,
        )
        assert prop.skip_vectorization is True
        assert prop.tokenization == "word"


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_basic_collection(self) -> None:
        """Test basic collection config."""
        config = CollectionConfig(
            name="TestCollection",
            properties=[PropertyConfig(name="content", data_type=PropertyDataType.TEXT)],
        )
        assert config.name == "TestCollection"
        assert len(config.properties) == 1

    def test_collection_with_properties(self) -> None:
        """Test collection with multiple properties."""
        config = CollectionConfig(
            name="Documents",
            properties=[
                PropertyConfig(name="title", data_type=PropertyDataType.TEXT),
                PropertyConfig(name="content", data_type=PropertyDataType.TEXT),
                PropertyConfig(name="created_at", data_type=PropertyDataType.DATE),
            ],
            vectorizer="text2vec-openai",
            
        )
        assert len(config.properties) == 3
        assert config.vectorizer == "text2vec-openai"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic search result."""
        result = SearchResult(
            uuid=str(uuid4()),
            score=0.85,
            properties={"content": "test content"},
        )
        assert result.score == 0.85
        assert result.properties["content"] == "test content"

    def test_result_with_metadata(self) -> None:
        """Test search result with metadata."""
        result = SearchResult(
            uuid=str(uuid4()),
            score=0.9,
            properties={"title": "Test"},
            vector=[0.1, 0.2, 0.3],
            distance=0.1,
        )
        assert result.vector is not None
        assert len(result.vector) == 3


class TestWeaviateClient:
    """Tests for WeaviateClient operations."""

    @pytest.fixture
    def client(self) -> WeaviateClient:
        return WeaviateClient(WeaviateSettings())

    def test_initial_state(self, client) -> None:
        """Test initial client state."""
        assert client._client is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, client) -> None:
        """Test disconnect operation."""
        # client is_connected is a property, test via mock
        client._client = MagicMock()
        await client.disconnect()
        assert client.is_connected is False


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_weaviate_client(self) -> None:
        """Test creating Weaviate client."""
        settings = WeaviateSettings()
        client = create_weaviate_client(settings)
        assert client is not None


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_all_statuses(self) -> None:
        """Test all health statuses exist."""
        expected = {"healthy", "degraded", "unhealthy", "unknown"}
        actual = {s.value for s in HealthStatus}
        assert actual == expected


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_all_component_types(self) -> None:
        """Test all component types exist."""
        expected = {"database", "cache", "vector_store", "message_queue",
                    "external_api", "file_storage", "custom"}
        actual = {t.value for t in ComponentType}
        assert actual == expected


class TestComponentHealth:
    """Tests for ComponentHealth."""

    def test_healthy_component(self) -> None:
        """Test healthy component."""
        health = ComponentHealth(
            name="postgres",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.DATABASE,
            latency_ms=5.0,
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.is_healthy is True
        assert health.message is None

    def test_unhealthy_component(self) -> None:
        """Test unhealthy component with error."""
        health = ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            component_type=ComponentType.CACHE,
            latency_ms=0.0,
            message="Connection refused",
        )
        assert health.status == HealthStatus.UNHEALTHY
        assert health.is_healthy is False
        assert health.message == "Connection refused"

    def test_degraded_component(self) -> None:
        """Test degraded component."""
        health = ComponentHealth(
            name="weaviate",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.VECTOR_STORE,
            latency_ms=500.0,
            details={"slow_queries": 10},
        )
        assert health.status == HealthStatus.DEGRADED
        assert health.details["slow_queries"] == 10

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        health = ComponentHealth(
            name="postgres",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.DATABASE,
            latency_ms=5.5,
        )
        data = health.to_dict()
        assert data["name"] == "postgres"
        assert data["status"] == "healthy"
        assert data["type"] == "database"
        assert data["latency_ms"] == 5.5


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_default_result(self) -> None:
        """Test default health check result."""
        result = HealthCheckResult()
        assert result.status == HealthStatus.UNKNOWN
        assert result.service_name == "solace-ai"
        assert result.is_healthy is False
        assert result.is_ready is False

    def test_healthy_result(self) -> None:
        """Test healthy result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            service_name="memory-service",
            version="1.0.0",
        )
        assert result.is_healthy is True
        assert result.is_ready is True

    def test_degraded_result(self) -> None:
        """Test degraded result is ready."""
        result = HealthCheckResult(status=HealthStatus.DEGRADED)
        assert result.is_healthy is False
        assert result.is_ready is True


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self) -> HealthMonitor:
        return HealthMonitor(service_name="test-service", version="1.0.0")

    def test_uptime(self, monitor) -> None:
        """Test uptime calculation."""
        uptime = monitor.uptime_seconds
        assert uptime >= 0

    def test_register_unregister(self, monitor) -> None:
        """Test registering and unregistering checker."""
        class MockChecker(HealthChecker):
            async def check(self) -> ComponentHealth:
                return ComponentHealth(
                    name="mock",
                    status=HealthStatus.HEALTHY,
                    component_type=ComponentType.CUSTOM,
                )

        checker = MockChecker("mock", ComponentType.CUSTOM)
        monitor.register(checker)
        assert len(monitor._checkers) == 1

        result = monitor.unregister("mock")
        assert result is True
        assert len(monitor._checkers) == 0

    def test_unregister_nonexistent(self, monitor) -> None:
        """Test unregistering nonexistent checker."""
        result = monitor.unregister("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_all_empty(self, monitor) -> None:
        """Test check_all with no checkers."""
        result = await monitor.check_all()
        assert result.status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, monitor) -> None:
        """Test check_all with healthy checkers."""
        async def healthy_check() -> dict:
            return {"status": "healthy"}

        monitor.register_callable("test1", ComponentType.DATABASE, healthy_check)
        monitor.register_callable("test2", ComponentType.CACHE, healthy_check)

        result = await monitor.check_all()
        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    @pytest.mark.asyncio
    async def test_check_all_one_unhealthy(self, monitor) -> None:
        """Test check_all with one unhealthy critical component."""
        async def healthy_check() -> dict:
            return {"status": "healthy"}

        async def unhealthy_check() -> dict:
            return {"status": "unhealthy"}

        monitor.register_callable("healthy", ComponentType.DATABASE, healthy_check)
        monitor.register_callable("unhealthy", ComponentType.CACHE, unhealthy_check, critical=True)

        result = await monitor.check_all()
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_all_noncritical_unhealthy(self, monitor) -> None:
        """Test check_all with non-critical unhealthy component."""
        async def healthy_check() -> dict:
            return {"status": "healthy"}

        async def unhealthy_check() -> dict:
            return {"status": "unhealthy"}

        monitor.register_callable("healthy", ComponentType.DATABASE, healthy_check)
        monitor.register_callable("unhealthy", ComponentType.CACHE, unhealthy_check, critical=False)

        result = await monitor.check_all()
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_liveness_probe(self, monitor) -> None:
        """Test liveness probe."""
        result = await monitor.liveness_probe()
        assert result["status"] == "alive"
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_readiness_probe(self, monitor) -> None:
        """Test readiness probe."""
        result = await monitor.readiness_probe()
        assert "ready" in result
        assert "status" in result

    @pytest.mark.asyncio
    async def test_startup_probe(self, monitor) -> None:
        """Test startup probe."""
        result = await monitor.startup_probe()
        assert "started" in result
        assert "status" in result

    def test_get_last_check_initially_none(self, monitor) -> None:
        """Test get_last_check initially returns None."""
        assert monitor.get_last_check() is None

    @pytest.mark.asyncio
    async def test_get_last_check_after_check(self, monitor) -> None:
        """Test get_last_check after running check."""
        await monitor.check_all()
        assert monitor.get_last_check() is not None


class TestCallableHealthChecker:
    """Tests for CallableHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy(self) -> None:
        """Test callable checker with healthy response."""
        async def check_fn() -> dict:
            return {"status": "healthy", "connections": 10}

        checker = CallableHealthChecker(
            name="test",
            component_type=ComponentType.DATABASE,
            check_fn=check_fn,
        )
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["connections"] == 10

    @pytest.mark.asyncio
    async def test_check_with_timeout(self) -> None:
        """Test callable checker with timeout."""
        async def slow_check() -> dict:
            await asyncio.sleep(10)
            return {"status": "healthy"}

        checker = CallableHealthChecker(
            name="slow",
            component_type=ComponentType.DATABASE,
            check_fn=slow_check,
            timeout_seconds=0.1,
        )
        result = await checker.check_with_timeout()
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message


class TestClientHealthChecker:
    """Tests for ClientHealthChecker."""

    @pytest.mark.asyncio
    async def test_check_healthy_client(self) -> None:
        """Test checking healthy client."""
        mock_client = MagicMock()
        mock_client.check_health = AsyncMock(return_value={"status": "healthy", "pool_size": 10})

        checker = ClientHealthChecker(
            name="postgres",
            component_type=ComponentType.DATABASE,
            client=mock_client,
        )
        result = await checker.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["pool_size"] == 10


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateHealthMonitor:
    """Tests for create_health_monitor factory."""

    def test_create_with_defaults(self) -> None:
        """Test creating with defaults."""
        monitor = create_health_monitor()
        assert monitor._service_name == "solace-ai"
        assert monitor._version == "1.0.0"

    def test_create_with_custom_values(self) -> None:
        """Test creating with custom values."""
        monitor = create_health_monitor(
            service_name="custom-service",
            version="2.0.0",
        )
        assert monitor._service_name == "custom-service"
        assert monitor._version == "2.0.0"


# =============================================================================
# Concurrent Access Tests
# =============================================================================

class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self) -> None:
        """Test concurrent health checks."""
        monitor = HealthMonitor(service_name="test", version="1.0.0")

        async def mock_check() -> dict:
            await asyncio.sleep(0.01)
            return {"status": "healthy"}

        monitor.register_callable("test", ComponentType.DATABASE, mock_check)

        results = await asyncio.gather(*[monitor.check_all() for _ in range(10)])
        assert len(results) == 10
        assert all(r.status == HealthStatus.HEALTHY for r in results)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_component_health_defaults(self) -> None:
        """Test ComponentHealth defaults."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.CUSTOM,
        )
        assert health.latency_ms == 0.0
        assert health.message is None
        assert health.details == {}

    def test_health_check_result_timestamp(self) -> None:
        """Test health check result has timestamp."""
        result = HealthCheckResult()
        assert result.checked_at is not None
        assert result.checked_at.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_check_all_parallel_vs_sequential(self) -> None:
        """Test parallel vs sequential execution."""
        monitor = HealthMonitor(service_name="test", version="1.0.0")

        call_order = []

        async def check1() -> dict:
            call_order.append("check1_start")
            await asyncio.sleep(0.01)
            call_order.append("check1_end")
            return {"status": "healthy"}

        async def check2() -> dict:
            call_order.append("check2_start")
            await asyncio.sleep(0.01)
            call_order.append("check2_end")
            return {"status": "healthy"}

        monitor.register_callable("c1", ComponentType.DATABASE, check1)
        monitor.register_callable("c2", ComponentType.CACHE, check2)

        # Parallel execution
        call_order.clear()
        await monitor.check_all(parallel=True)
        # Both should start before either ends
        assert call_order[0:2] == ["check1_start", "check2_start"] or \
               call_order[0:2] == ["check2_start", "check1_start"]

        # Sequential execution
        call_order.clear()
        await monitor.check_all(parallel=False)
        # One should complete before the other starts
        assert call_order == ["check1_start", "check1_end", "check2_start", "check2_end"]
