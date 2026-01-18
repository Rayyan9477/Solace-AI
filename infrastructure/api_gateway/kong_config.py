"""
Solace-AI API Gateway - Kong Configuration.
Provides Kong gateway configuration for service routing, load balancing, and health checks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import json
import httpx
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithms supported by Kong."""
    ROUND_ROBIN = "round-robin"
    CONSISTENT_HASHING = "consistent-hashing"
    LEAST_CONNECTIONS = "least-connections"
    LATENCY = "latency"


class HealthCheckType(str, Enum):
    """Health check types for upstreams."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"


class KongSettings(BaseSettings):
    """Kong Admin API configuration settings."""
    admin_url: str = Field(default="http://localhost:8001")
    admin_token: str = Field(default="")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    enable_ssl_verify: bool = Field(default=True)
    workspace: str = Field(default="default")
    default_protocol: str = Field(default="http")
    default_connect_timeout: int = Field(default=60000, ge=1000)
    default_write_timeout: int = Field(default=60000, ge=1000)
    default_read_timeout: int = Field(default=60000, ge=1000)
    model_config = SettingsConfigDict(env_prefix="KONG_", env_file=".env", extra="ignore")

    @field_validator("admin_url")
    @classmethod
    def validate_admin_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("admin_url must start with http:// or https://")
        return v.rstrip("/")


@dataclass
class HealthCheckConfig:
    """Health check configuration for upstream targets."""
    active_enabled: bool = True
    active_type: HealthCheckType = HealthCheckType.HTTP
    active_http_path: str = "/health"
    active_healthy_interval: int = 10
    active_healthy_successes: int = 2
    active_unhealthy_interval: int = 5
    active_unhealthy_http_failures: int = 3
    active_unhealthy_tcp_failures: int = 3
    active_unhealthy_timeouts: int = 3
    passive_enabled: bool = True
    passive_healthy_successes: int = 2
    passive_unhealthy_http_failures: int = 5
    passive_unhealthy_tcp_failures: int = 5
    passive_unhealthy_timeouts: int = 5

    def to_kong_format(self) -> dict[str, Any]:
        return {
            "active": {
                "type": self.active_type.value,
                "http_path": self.active_http_path,
                "healthy": {"interval": self.active_healthy_interval, "successes": self.active_healthy_successes},
                "unhealthy": {"interval": self.active_unhealthy_interval, "http_failures": self.active_unhealthy_http_failures, "tcp_failures": self.active_unhealthy_tcp_failures, "timeouts": self.active_unhealthy_timeouts},
            },
            "passive": {
                "healthy": {"successes": self.passive_healthy_successes},
                "unhealthy": {"http_failures": self.passive_unhealthy_http_failures, "tcp_failures": self.passive_unhealthy_tcp_failures, "timeouts": self.passive_unhealthy_timeouts},
            },
        }


@dataclass
class UpstreamTarget:
    """Individual upstream target (backend server)."""
    target: str
    weight: int = 100
    tags: list[str] = field(default_factory=list)

    def to_kong_format(self) -> dict[str, Any]:
        return {"target": self.target, "weight": self.weight, "tags": self.tags}


@dataclass
class UpstreamConfig:
    """Kong upstream configuration for load balancing."""
    name: str
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    slots: int = 10000
    hash_on: str = "none"
    hash_fallback: str = "none"
    hash_on_header: str | None = None
    targets: list[UpstreamTarget] = field(default_factory=list)
    healthchecks: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    tags: list[str] = field(default_factory=list)

    def to_kong_format(self) -> dict[str, Any]:
        config: dict[str, Any] = {"name": self.name, "algorithm": self.algorithm.value, "slots": self.slots, "hash_on": self.hash_on, "hash_fallback": self.hash_fallback, "tags": self.tags, "healthchecks": self.healthchecks.to_kong_format()}
        if self.hash_on_header:
            config["hash_on_header"] = self.hash_on_header
        return config


@dataclass
class ServiceConfig:
    """Kong service configuration."""
    name: str
    host: str
    port: int = 80
    protocol: str = "http"
    path: str = ""
    retries: int = 5
    connect_timeout: int = 60000
    write_timeout: int = 60000
    read_timeout: int = 60000
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_kong_format(self) -> dict[str, Any]:
        return {"name": self.name, "host": self.host, "port": self.port, "protocol": self.protocol, "path": self.path or None, "retries": self.retries, "connect_timeout": self.connect_timeout, "write_timeout": self.write_timeout, "read_timeout": self.read_timeout, "enabled": self.enabled, "tags": self.tags}


@dataclass
class PluginConfig:
    """Kong plugin configuration."""
    name: str
    service_id: str | None = None
    route_id: str | None = None
    consumer_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_kong_format(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "config": self.config, "enabled": self.enabled, "tags": self.tags}
        if self.service_id:
            data["service"] = {"id": self.service_id}
        if self.route_id:
            data["route"] = {"id": self.route_id}
        if self.consumer_id:
            data["consumer"] = {"id": self.consumer_id}
        return data


@dataclass
class ConsumerConfig:
    """Kong consumer configuration."""
    username: str
    custom_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_kong_format(self) -> dict[str, Any]:
        data: dict[str, Any] = {"username": self.username, "tags": self.tags}
        if self.custom_id:
            data["custom_id"] = self.custom_id
        return data


class KongAdminClient:
    """HTTP client for Kong Admin API operations."""

    def __init__(self, settings: KongSettings) -> None:
        self._settings = settings
        self._headers = {"Content-Type": "application/json"}
        if settings.admin_token:
            self._headers["Kong-Admin-Token"] = settings.admin_token

    async def _request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._settings.admin_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds, verify=self._settings.enable_ssl_verify) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    if method == "GET":
                        response = await client.get(url, headers=self._headers)
                    elif method == "POST":
                        response = await client.post(url, headers=self._headers, json=data)
                    elif method == "PUT":
                        response = await client.put(url, headers=self._headers, json=data)
                    elif method == "PATCH":
                        response = await client.patch(url, headers=self._headers, json=data)
                    elif method == "DELETE":
                        response = await client.delete(url, headers=self._headers)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    response.raise_for_status()
                    return response.json() if response.content else {}
                except httpx.HTTPStatusError as e:
                    logger.warning("kong_admin_http_error", status_code=e.response.status_code, endpoint=endpoint, attempt=attempt + 1)
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning("kong_admin_request_error", error=str(e), endpoint=endpoint, attempt=attempt + 1)
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError(f"Kong Admin API request failed: {method} {endpoint}")

    async def create_service(self, config: ServiceConfig) -> dict[str, Any]:
        return await self._request("POST", "/services", config.to_kong_format())

    async def get_service(self, name_or_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/services/{name_or_id}")

    async def update_service(self, name_or_id: str, config: ServiceConfig) -> dict[str, Any]:
        return await self._request("PATCH", f"/services/{name_or_id}", config.to_kong_format())

    async def delete_service(self, name_or_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/services/{name_or_id}")

    async def create_upstream(self, config: UpstreamConfig) -> dict[str, Any]:
        return await self._request("POST", "/upstreams", config.to_kong_format())

    async def add_upstream_target(self, upstream_name: str, target: UpstreamTarget) -> dict[str, Any]:
        return await self._request("POST", f"/upstreams/{upstream_name}/targets", target.to_kong_format())

    async def create_plugin(self, config: PluginConfig) -> dict[str, Any]:
        return await self._request("POST", "/plugins", config.to_kong_format())

    async def create_consumer(self, config: ConsumerConfig) -> dict[str, Any]:
        return await self._request("POST", "/consumers", config.to_kong_format())

    async def get_status(self) -> dict[str, Any]:
        return await self._request("GET", "/status")


class KongConfig:
    """Kong gateway configuration manager."""

    def __init__(self, settings: KongSettings | None = None) -> None:
        self._settings = settings or KongSettings()
        self._client = KongAdminClient(self._settings)
        self._services: dict[str, ServiceConfig] = {}
        self._upstreams: dict[str, UpstreamConfig] = {}

    def define_service(self, name: str, host: str, port: int = 80, protocol: str = "http", path: str = "", **kwargs: Any) -> ServiceConfig:
        config = ServiceConfig(name=name, host=host, port=port, protocol=protocol, path=path, **kwargs)
        self._services[name] = config
        logger.info("service_defined", name=name, host=host, port=port)
        return config

    def define_upstream(self, name: str, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN, **kwargs: Any) -> UpstreamConfig:
        config = UpstreamConfig(name=name, algorithm=algorithm, **kwargs)
        self._upstreams[name] = config
        logger.info("upstream_defined", name=name, algorithm=algorithm.value)
        return config

    def add_target_to_upstream(self, upstream_name: str, target: str, weight: int = 100) -> None:
        if upstream_name not in self._upstreams:
            raise ValueError(f"Upstream '{upstream_name}' not defined")
        self._upstreams[upstream_name].targets.append(UpstreamTarget(target=target, weight=weight))
        logger.info("target_added", upstream=upstream_name, target=target, weight=weight)

    async def apply_configuration(self) -> dict[str, Any]:
        results: dict[str, Any] = {"services": [], "upstreams": []}
        for upstream in self._upstreams.values():
            try:
                result = await self._client.create_upstream(upstream)
                results["upstreams"].append({"name": upstream.name, "status": "created", "id": result.get("id")})
                for target in upstream.targets:
                    await self._client.add_upstream_target(upstream.name, target)
            except Exception as e:
                results["upstreams"].append({"name": upstream.name, "status": "failed", "error": str(e)})
                logger.error("upstream_creation_failed", name=upstream.name, error=str(e))
        for service in self._services.values():
            try:
                result = await self._client.create_service(service)
                results["services"].append({"name": service.name, "status": "created", "id": result.get("id")})
            except Exception as e:
                results["services"].append({"name": service.name, "status": "failed", "error": str(e)})
                logger.error("service_creation_failed", name=service.name, error=str(e))
        logger.info("configuration_applied", services=len(self._services), upstreams=len(self._upstreams))
        return results

    def export_declarative_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {"_format_version": "3.0", "services": [], "upstreams": []}
        for service in self._services.values():
            config["services"].append(service.to_kong_format())
        for upstream in self._upstreams.values():
            upstream_config = upstream.to_kong_format()
            upstream_config["targets"] = [t.to_kong_format() for t in upstream.targets]
            config["upstreams"].append(upstream_config)
        return config

    def export_to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.export_declarative_config(), default_flow_style=False, sort_keys=False)

    def get_service(self, name: str) -> ServiceConfig | None:
        return self._services.get(name)

    def get_upstream(self, name: str) -> UpstreamConfig | None:
        return self._upstreams.get(name)

    async def health_check(self) -> dict[str, Any]:
        try:
            status = await self._client.get_status()
            return {"healthy": True, "status": status}
        except Exception as e:
            logger.error("kong_health_check_failed", error=str(e))
            return {"healthy": False, "error": str(e)}


def create_solace_gateway_config(settings: KongSettings | None = None) -> KongConfig:
    """Create pre-configured Kong gateway for Solace-AI services."""
    config = KongConfig(settings)
    config.define_upstream("orchestrator-upstream", algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
    config.add_target_to_upstream("orchestrator-upstream", "orchestrator-service:8001", weight=100)
    config.define_upstream("user-upstream", algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
    config.add_target_to_upstream("user-upstream", "user-service:8007", weight=100)
    config.define_upstream("safety-upstream", algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
    config.add_target_to_upstream("safety-upstream", "safety-service:8002", weight=100)
    config.define_service("orchestrator-service", host="orchestrator-upstream", port=8001, path="/api/v1/orchestrator", tags=["solace", "orchestrator"])
    config.define_service("user-service", host="user-upstream", port=8007, path="/api/v1/users", tags=["solace", "user"])
    config.define_service("safety-service", host="safety-upstream", port=8002, path="/api/v1/safety", tags=["solace", "safety"])
    logger.info("solace_gateway_config_created", services=3, upstreams=3)
    return config
