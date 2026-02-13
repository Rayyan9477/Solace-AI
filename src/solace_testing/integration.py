"""Solace-AI Testing Library - Integration test utilities."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class ServiceType(str, Enum):
    """Types of services in the test environment."""
    POSTGRES = "postgres"
    REDIS = "redis"
    WEAVIATE = "weaviate"
    KAFKA = "kafka"
    API = "api"
    WORKER = "worker"


class ServiceStatus(str, Enum):
    """Service health status."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class ServiceConfig(BaseModel):
    """Configuration for a test service."""
    name: str
    service_type: ServiceType
    image: str = ""
    port: int = 0
    health_endpoint: str = "/health"
    environment: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    startup_timeout: float = 60.0
    health_check_interval: float = 1.0


class ServiceState(BaseModel):
    """Current state of a service."""
    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    host: str = "localhost"
    port: int = 0
    started_at: datetime | None = None
    healthy_at: datetime | None = None
    error: str | None = None

    def connection_string(self, protocol: str = "tcp") -> str:
        return f"{protocol}://{self.host}:{self.port}"


@dataclass
class ServiceContainer:
    """Container for managing test services."""
    config: ServiceConfig
    state: ServiceState = field(default_factory=lambda: ServiceState(name=""))

    def __post_init__(self) -> None:
        self.state = ServiceState(name=self.config.name, port=self.config.port)

    async def start(self) -> None:
        logger.info("Starting service", name=self.config.name, type=self.config.service_type)
        self.state.status = ServiceStatus.STARTING
        self.state.started_at = datetime.now(timezone.utc)
        await asyncio.sleep(0.1)
        self.state.status = ServiceStatus.HEALTHY
        self.state.healthy_at = datetime.now(timezone.utc)
        logger.info("Service started", name=self.config.name)

    async def stop(self) -> None:
        logger.info("Stopping service", name=self.config.name)
        self.state.status = ServiceStatus.STOPPED
        logger.info("Service stopped", name=self.config.name)

    async def health_check(self) -> bool:
        return self.state.status == ServiceStatus.HEALTHY

    def get_connection_info(self) -> dict[str, Any]:
        return {"host": self.state.host, "port": self.state.port, "connection_string": self.state.connection_string()}


class HealthWaiter:
    """Utility for waiting on service health."""

    def __init__(self, timeout: float = 60.0, check_interval: float = 1.0, backoff_multiplier: float = 1.5) -> None:
        self.timeout = timeout
        self.check_interval = check_interval
        self.backoff_multiplier = backoff_multiplier

    async def wait_for_service(self, health_check: Callable[[], Any], service_name: str = "service") -> bool:
        start_time = time.monotonic()
        current_interval = self.check_interval
        while time.monotonic() - start_time < self.timeout:
            try:
                result = health_check()
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    logger.info("Service healthy", service=service_name, elapsed=time.monotonic() - start_time)
                    return True
            except Exception as e:
                logger.debug("Health check failed", service=service_name, error=str(e))
            await asyncio.sleep(current_interval)
            current_interval = min(current_interval * self.backoff_multiplier, 10.0)
        logger.error("Service health timeout", service=service_name, timeout=self.timeout)
        return False

    async def wait_for_all(self, health_checks: dict[str, Callable[[], Any]]) -> dict[str, bool]:
        tasks = [self._check_with_name(name, check) for name, check in health_checks.items()]
        completed = await asyncio.gather(*tasks)
        return {name: healthy for name, healthy in completed}

    async def _check_with_name(self, name: str, check: Callable[[], Any]) -> tuple[str, bool]:
        healthy = await self.wait_for_service(check, name)
        return name, healthy


class IntegrationEnvironment:
    """Test environment configuration and management."""

    def __init__(self) -> None:
        self._services: dict[str, ServiceContainer] = {}
        self._health_waiter = HealthWaiter()
        self._env_vars: dict[str, str] = {}
        self._original_env: dict[str, str | None] = {}

    def add_service(self, config: ServiceConfig) -> ServiceContainer:
        container = ServiceContainer(config)
        self._services[config.name] = container
        return container

    def set_env_var(self, name: str, value: str) -> None:
        if name not in self._original_env:
            self._original_env[name] = os.environ.get(name)
        os.environ[name] = value
        self._env_vars[name] = value

    def restore_env_vars(self) -> None:
        for name, original in self._original_env.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
        self._original_env.clear()
        self._env_vars.clear()

    async def start_all(self) -> dict[str, bool]:
        started: set[str] = set()
        results: dict[str, bool] = {}
        while len(started) < len(self._services):
            ready = [n for n, s in self._services.items() if n not in started and all(d in started for d in s.config.depends_on)]
            if not ready:
                raise RuntimeError(f"Circular dependency detected: {set(self._services.keys()) - started}")
            service_results = await asyncio.gather(*[self._start_service(name) for name in ready])
            for name, success in zip(ready, service_results):
                results[name] = success
                started.add(name)
        return results

    async def _start_service(self, name: str) -> bool:
        container = self._services[name]
        try:
            await container.start()
            return await self._health_waiter.wait_for_service(container.health_check, name)
        except Exception as e:
            logger.error("Service start failed", service=name, error=str(e))
            container.state.status = ServiceStatus.UNHEALTHY
            container.state.error = str(e)
            return False

    async def stop_all(self) -> None:
        for name in reversed(list(self._services.keys())):
            try:
                await self._services[name].stop()
            except Exception as e:
                logger.error("Service stop failed", service=name, error=str(e))

    def get_service(self, name: str) -> ServiceContainer:
        if name not in self._services:
            raise KeyError(f"Service '{name}' not found")
        return self._services[name]

    def get_connection_info(self, service_name: str) -> dict[str, Any]:
        return self.get_service(service_name).get_connection_info()


class DataSeeder:
    """Database seeding utility for integration tests."""

    def __init__(self) -> None:
        self._seed_data: dict[str, list[dict[str, Any]]] = {}
        self._cleanup_handlers: list[Callable[[], Any]] = []

    def add_seed_data(self, table: str, records: list[dict[str, Any]]) -> None:
        self._seed_data.setdefault(table, []).extend(records)

    def add_cleanup_handler(self, handler: Callable[[], Any]) -> None:
        self._cleanup_handlers.append(handler)

    async def seed(self, executor: Callable[[str, list[dict[str, Any]]], Any]) -> int:
        total = 0
        for table, records in self._seed_data.items():
            result = executor(table, records)
            if asyncio.iscoroutine(result):
                await result
            total += len(records)
            logger.info("Seeded data", table=table, count=len(records))
        return total

    async def cleanup(self) -> None:
        for handler in reversed(self._cleanup_handlers):
            try:
                result = handler()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Cleanup failed", error=str(e))
        self._cleanup_handlers.clear()

    def clear_seed_data(self) -> None:
        self._seed_data.clear()


class IntegrationTestRunner:
    """Coordinated integration test execution."""

    def __init__(self, environment: IntegrationEnvironment | None = None) -> None:
        self.environment = environment or IntegrationEnvironment()
        self.seeder = DataSeeder()
        self._setup_complete = False

    async def setup(self) -> dict[str, bool]:
        logger.info("Setting up integration test environment")
        results = await self.environment.start_all()
        self._setup_complete = all(results.values())
        if not self._setup_complete:
            logger.error("Environment setup failed", failed_services=[n for n, ok in results.items() if not ok])
        return results

    async def teardown(self) -> None:
        logger.info("Tearing down integration test environment")
        await self.seeder.cleanup()
        await self.environment.stop_all()
        self.environment.restore_env_vars()
        self._setup_complete = False

    @asynccontextmanager
    async def test_context(self) -> AsyncIterator[IntegrationTestRunner]:
        try:
            await self.setup()
            yield self
        finally:
            await self.teardown()

    def is_ready(self) -> bool:
        return self._setup_complete


class APITestClient:
    """HTTP client for API integration testing.

    Uses ``httpx.AsyncClient`` to make real HTTP requests against a running
    service, making it suitable for integration and contract tests.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0) -> None:
        import httpx
        self.base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        self._auth_token: str | None = None
        self._timeout = timeout
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def set_auth_token(self, token: str) -> None:
        self._auth_token = token
        self._headers["Authorization"] = f"Bearer {token}"

    def clear_auth(self) -> None:
        self._auth_token = None
        self._headers.pop("Authorization", None)

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    async def request(self, method: str, url: str, headers: dict[str, str] | None = None,
                      params: dict[str, Any] | None = None,
                      json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an HTTP request and return a normalised response dict."""
        merged_headers = {**self._headers, **(headers or {})}
        response = await self._client.request(
            method=method,
            url=self._build_url(url),
            headers=merged_headers,
            params=params,
            json=json_data,
        )
        body: Any = None
        try:
            body = response.json()
        except Exception:
            body = response.text
        return {
            "status": response.status_code,
            "headers": dict(response.headers),
            "json": body if isinstance(body, dict) else {},
            "body": body,
            "text": response.text,
        }

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.request("POST", path, json_data=json_data)

    async def put(self, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.request("PUT", path, json_data=json_data)

    async def delete(self, path: str) -> dict[str, Any]:
        return await self.request("DELETE", path)

    async def health_check(self) -> bool:
        try:
            result = await self.get("/health")
            return result.get("status", 500) < 400
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def create_test_environment() -> IntegrationEnvironment:
    """Create a default test environment with common services."""
    env = IntegrationEnvironment()
    env.add_service(ServiceConfig(name="postgres", service_type=ServiceType.POSTGRES, port=5432, health_endpoint="/"))
    env.add_service(ServiceConfig(name="redis", service_type=ServiceType.REDIS, port=6379, health_endpoint="/"))
    env.add_service(ServiceConfig(name="weaviate", service_type=ServiceType.WEAVIATE, port=8080, health_endpoint="/v1/.well-known/ready"))
    return env
