"""Unit tests for Solace-AI Testing Library - Integration module."""

from __future__ import annotations

import os
import pytest

from solace_testing.integration import (
    APITestClient,
    DataSeeder,
    HealthWaiter,
    IntegrationTestRunner,
    ServiceConfig,
    ServiceContainer,
    ServiceState,
    ServiceStatus,
    ServiceType,
    IntegrationEnvironment,
    create_test_environment,
)


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_default_values(self) -> None:
        config = ServiceConfig(
            name="test-service",
            service_type=ServiceType.API,
        )
        assert config.health_endpoint == "/health"
        assert config.startup_timeout == 60.0
        assert len(config.depends_on) == 0

    def test_custom_values(self) -> None:
        config = ServiceConfig(
            name="postgres",
            service_type=ServiceType.POSTGRES,
            port=5432,
            depends_on=["redis"],
        )
        assert config.port == 5432
        assert "redis" in config.depends_on


class TestServiceState:
    """Tests for ServiceState."""

    def test_default_state(self) -> None:
        state = ServiceState(name="test")
        assert state.status == ServiceStatus.UNKNOWN
        assert state.host == "localhost"

    def test_connection_string(self) -> None:
        state = ServiceState(name="test", host="localhost", port=5432)
        conn = state.connection_string("postgresql")
        assert conn == "postgresql://localhost:5432"


class TestServiceContainer:
    """Tests for ServiceContainer."""

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        config = ServiceConfig(name="test", service_type=ServiceType.API, port=8000)
        container = ServiceContainer(config)
        await container.start()
        assert container.state.status == ServiceStatus.HEALTHY
        await container.stop()
        assert container.state.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        config = ServiceConfig(name="test", service_type=ServiceType.API)
        container = ServiceContainer(config)
        await container.start()
        is_healthy = await container.health_check()
        assert is_healthy is True
        await container.stop()

    def test_get_connection_info(self) -> None:
        config = ServiceConfig(name="test", service_type=ServiceType.REDIS, port=6379)
        container = ServiceContainer(config)
        info = container.get_connection_info()
        assert info["port"] == 6379
        assert "localhost" in info["connection_string"]


class TestHealthWaiter:
    """Tests for HealthWaiter."""

    @pytest.mark.asyncio
    async def test_wait_for_healthy_service(self) -> None:
        waiter = HealthWaiter(timeout=5.0, check_interval=0.1)
        call_count = 0

        async def health_check() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        result = await waiter.wait_for_service(health_check, "test-service")
        assert result is True
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_wait_timeout(self) -> None:
        waiter = HealthWaiter(timeout=0.5, check_interval=0.1)

        async def always_unhealthy() -> bool:
            return False

        result = await waiter.wait_for_service(always_unhealthy, "failing-service")
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_all(self) -> None:
        waiter = HealthWaiter(timeout=2.0, check_interval=0.1)

        async def healthy() -> bool:
            return True

        results = await waiter.wait_for_all({
            "service1": healthy,
            "service2": healthy,
        })
        assert results["service1"] is True
        assert results["service2"] is True


class TestIntegrationEnvironment:
    """Tests for IntegrationEnvironment."""

    def test_add_service(self) -> None:
        env = IntegrationEnvironment()
        config = ServiceConfig(name="redis", service_type=ServiceType.REDIS, port=6379)
        container = env.add_service(config)
        assert container is not None
        assert env.get_service("redis") is container

    def test_get_nonexistent_service(self) -> None:
        env = IntegrationEnvironment()
        with pytest.raises(KeyError, match="not found"):
            env.get_service("nonexistent")

    @pytest.mark.asyncio
    async def test_start_stop_all(self) -> None:
        env = IntegrationEnvironment()
        env.add_service(ServiceConfig(name="s1", service_type=ServiceType.API))
        env.add_service(ServiceConfig(name="s2", service_type=ServiceType.API, depends_on=["s1"]))
        results = await env.start_all()
        assert results["s1"] is True
        assert results["s2"] is True
        await env.stop_all()

    def test_env_var_management(self) -> None:
        env = IntegrationEnvironment()
        original = os.environ.get("TEST_VAR_12345")
        env.set_env_var("TEST_VAR_12345", "test_value")
        assert os.environ.get("TEST_VAR_12345") == "test_value"
        env.restore_env_vars()
        assert os.environ.get("TEST_VAR_12345") == original


class TestDataSeeder:
    """Tests for DataSeeder."""

    def test_add_seed_data(self) -> None:
        seeder = DataSeeder()
        seeder.add_seed_data("users", [{"id": "1", "name": "Test"}])
        assert "users" in seeder._seed_data
        assert len(seeder._seed_data["users"]) == 1

    @pytest.mark.asyncio
    async def test_seed_execution(self) -> None:
        seeder = DataSeeder()
        seeder.add_seed_data("users", [{"id": "1"}, {"id": "2"}])
        inserted = []

        def executor(table: str, records: list) -> None:
            inserted.extend(records)

        count = await seeder.seed(executor)
        assert count == 2
        assert len(inserted) == 2

    @pytest.mark.asyncio
    async def test_cleanup_handlers(self) -> None:
        seeder = DataSeeder()
        cleaned = []

        def cleanup() -> None:
            cleaned.append("done")

        seeder.add_cleanup_handler(cleanup)
        await seeder.cleanup()
        assert "done" in cleaned


class TestIntegrationTestRunner:
    """Tests for IntegrationTestRunner."""

    @pytest.mark.asyncio
    async def test_setup_teardown(self) -> None:
        env = IntegrationEnvironment()
        env.add_service(ServiceConfig(name="test", service_type=ServiceType.API))
        runner = IntegrationTestRunner(env)
        results = await runner.setup()
        assert runner.is_ready() is True
        await runner.teardown()
        assert runner.is_ready() is False

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        env = IntegrationEnvironment()
        env.add_service(ServiceConfig(name="test", service_type=ServiceType.API))
        runner = IntegrationTestRunner(env)
        async with runner.test_context() as ctx:
            assert ctx.is_ready() is True
        assert runner.is_ready() is False


class TestAPITestClient:
    """Tests for APITestClient."""

    def test_build_url(self) -> None:
        client = APITestClient(base_url="http://localhost:8000")
        url = client._build_url("/api/users")
        assert url == "http://localhost:8000/api/users"

    def test_auth_token(self) -> None:
        client = APITestClient()
        client.set_auth_token("test-token")
        assert client._headers["Authorization"] == "Bearer test-token"
        client.clear_auth()
        assert "Authorization" not in client._headers

    @pytest.mark.asyncio
    async def test_get_request(self) -> None:
        client = APITestClient()
        response = await client.get("/api/test", params={"q": "search"})
        assert response["method"] == "GET"
        assert response["params"]["q"] == "search"

    @pytest.mark.asyncio
    async def test_post_request(self) -> None:
        client = APITestClient()
        response = await client.post("/api/users", json_data={"name": "Test"})
        assert response["method"] == "POST"
        assert response["json"]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_put_request(self) -> None:
        client = APITestClient()
        response = await client.put("/api/users/1", json_data={"name": "Updated"})
        assert response["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_delete_request(self) -> None:
        client = APITestClient()
        response = await client.delete("/api/users/1")
        assert response["method"] == "DELETE"
        assert response["status"] == 204

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        client = APITestClient()
        is_healthy = await client.health_check()
        assert isinstance(is_healthy, bool)


class TestCreateTestEnvironment:
    """Tests for create_test_environment function."""

    def test_creates_default_services(self) -> None:
        env = create_test_environment()
        assert env.get_service("postgres") is not None
        assert env.get_service("redis") is not None
        assert env.get_service("weaviate") is not None


class TestServiceTypes:
    """Tests for ServiceType enum."""

    def test_all_types_exist(self) -> None:
        assert ServiceType.POSTGRES == "postgres"
        assert ServiceType.REDIS == "redis"
        assert ServiceType.WEAVIATE == "weaviate"
        assert ServiceType.KAFKA == "kafka"
        assert ServiceType.API == "api"
        assert ServiceType.WORKER == "worker"


class TestServiceStatus:
    """Tests for ServiceStatus enum."""

    def test_all_statuses_exist(self) -> None:
        assert ServiceStatus.UNKNOWN == "unknown"
        assert ServiceStatus.STARTING == "starting"
        assert ServiceStatus.HEALTHY == "healthy"
        assert ServiceStatus.UNHEALTHY == "unhealthy"
        assert ServiceStatus.STOPPED == "stopped"
