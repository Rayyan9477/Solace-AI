"""
Solace-AI API Gateway - Route Definitions.
Manages Kong route configuration for service routing and path matching.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re
import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class HttpMethod(str, Enum):
    """Supported HTTP methods for routes."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RouteProtocol(str, Enum):
    """Supported protocols for routes."""
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    GRPCS = "grpcs"


class RouteSettings(BaseSettings):
    """Route configuration settings."""
    admin_url: str = Field(default="http://localhost:8001")
    admin_token: str = Field(default="")
    timeout_seconds: float = Field(default=10.0)
    default_strip_path: bool = Field(default=True)
    default_preserve_host: bool = Field(default=False)
    regex_priority: int = Field(default=0)
    model_config = SettingsConfigDict(env_prefix="KONG_ROUTE_", env_file=".env", extra="ignore")


@dataclass
class RouteDefinition:
    """Individual route definition."""
    name: str
    paths: list[str]
    service_name: str
    methods: list[HttpMethod] = field(default_factory=lambda: [HttpMethod.GET, HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE])
    protocols: list[RouteProtocol] = field(default_factory=lambda: [RouteProtocol.HTTP, RouteProtocol.HTTPS])
    hosts: list[str] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)
    strip_path: bool = True
    preserve_host: bool = False
    regex_priority: int = 0
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_kong_format(self, service_id: str) -> dict[str, Any]:
        return {"name": self.name, "paths": self.paths, "methods": [m.value for m in self.methods], "protocols": [p.value for p in self.protocols], "hosts": self.hosts if self.hosts else None, "headers": self.headers if self.headers else None, "strip_path": self.strip_path, "preserve_host": self.preserve_host, "regex_priority": self.regex_priority, "tags": self.tags, "service": {"id": service_id}}

    def matches_path(self, path: str) -> bool:
        for route_path in self.paths:
            if route_path.startswith("~"):
                pattern = route_path[1:]
                if re.match(pattern, path):
                    return True
            elif path.startswith(route_path):
                return True
        return False

    def matches_method(self, method: str) -> bool:
        return HttpMethod(method.upper()) in self.methods


@dataclass
class RouteGroup:
    """Group of related routes for a service."""
    name: str
    base_path: str
    service_name: str
    routes: list[RouteDefinition] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def add_route(self, name: str, sub_path: str, methods: list[HttpMethod] | None = None, **kwargs: Any) -> RouteDefinition:
        full_path = f"{self.base_path.rstrip('/')}/{sub_path.lstrip('/')}" if sub_path else self.base_path
        route = RouteDefinition(name=f"{self.name}-{name}", paths=[full_path], service_name=self.service_name, methods=methods or [HttpMethod.GET, HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE], tags=self.tags + kwargs.pop("tags", []), **kwargs)
        self.routes.append(route)
        return route


class RouteConfig:
    """Route configuration builder."""

    def __init__(self, settings: RouteSettings | None = None) -> None:
        self._settings = settings or RouteSettings()
        self._routes: dict[str, RouteDefinition] = {}
        self._groups: dict[str, RouteGroup] = {}
        self._service_routes: dict[str, list[str]] = {}

    def define_route(self, name: str, paths: list[str], service_name: str, methods: list[HttpMethod] | None = None, **kwargs: Any) -> RouteDefinition:
        route = RouteDefinition(name=name, paths=paths, service_name=service_name, methods=methods or [HttpMethod.GET, HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE], strip_path=kwargs.pop("strip_path", self._settings.default_strip_path), preserve_host=kwargs.pop("preserve_host", self._settings.default_preserve_host), **kwargs)
        self._routes[name] = route
        if service_name not in self._service_routes:
            self._service_routes[service_name] = []
        self._service_routes[service_name].append(name)
        logger.info("route_defined", name=name, paths=paths, service=service_name)
        return route

    def define_group(self, name: str, base_path: str, service_name: str, tags: list[str] | None = None) -> RouteGroup:
        group = RouteGroup(name=name, base_path=base_path, service_name=service_name, tags=tags or [])
        self._groups[name] = group
        logger.info("route_group_defined", name=name, base_path=base_path, service=service_name)
        return group

    def get_route(self, name: str) -> RouteDefinition | None:
        return self._routes.get(name)

    def get_routes_for_service(self, service_name: str) -> list[RouteDefinition]:
        route_names = self._service_routes.get(service_name, [])
        return [self._routes[name] for name in route_names if name in self._routes]

    def find_matching_route(self, path: str, method: str) -> RouteDefinition | None:
        for route in self._routes.values():
            if route.enabled and route.matches_path(path) and route.matches_method(method):
                return route
        for group in self._groups.values():
            for route in group.routes:
                if route.enabled and route.matches_path(path) and route.matches_method(method):
                    return route
        return None

    def export_routes(self) -> list[dict[str, Any]]:
        routes = [route.to_kong_format("") for route in self._routes.values()]
        for group in self._groups.values():
            routes.extend([route.to_kong_format("") for route in group.routes])
        return routes


class RouteManager:
    """Manages Kong route lifecycle operations."""

    def __init__(self, settings: RouteSettings | None = None) -> None:
        self._settings = settings or RouteSettings()
        self._headers = {"Content-Type": "application/json"}
        if self._settings.admin_token:
            self._headers["Kong-Admin-Token"] = self._settings.admin_token

    async def _request(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._settings.admin_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
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

    async def create_route(self, route: RouteDefinition, service_id: str) -> dict[str, Any]:
        data = route.to_kong_format(service_id)
        result = await self._request("POST", "/routes", data)
        logger.info("route_created", name=route.name, id=result.get("id"))
        return result

    async def get_route(self, name_or_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/routes/{name_or_id}")

    async def update_route(self, name_or_id: str, route: RouteDefinition, service_id: str) -> dict[str, Any]:
        data = route.to_kong_format(service_id)
        result = await self._request("PATCH", f"/routes/{name_or_id}", data)
        logger.info("route_updated", name=route.name)
        return result

    async def delete_route(self, name_or_id: str) -> dict[str, Any]:
        result = await self._request("DELETE", f"/routes/{name_or_id}")
        logger.info("route_deleted", name_or_id=name_or_id)
        return result

    async def list_routes(self, service_name: str | None = None) -> list[dict[str, Any]]:
        endpoint = f"/services/{service_name}/routes" if service_name else "/routes"
        result = await self._request("GET", endpoint)
        return result.get("data", [])


class ServiceRoutes:
    """Pre-defined route configurations for Solace-AI services."""

    @staticmethod
    def orchestrator_routes() -> RouteGroup:
        group = RouteGroup(name="orchestrator", base_path="/api/v1/orchestrator", service_name="orchestrator-service", tags=["solace", "orchestrator"])
        group.add_route("process", "/process", methods=[HttpMethod.POST])
        group.add_route("stream", "/stream", methods=[HttpMethod.POST])
        group.add_route("websocket", "/ws", methods=[HttpMethod.GET])
        group.add_route("sessions", "/sessions", methods=[HttpMethod.GET, HttpMethod.POST])
        group.add_route("session-detail", "/sessions/{id}", methods=[HttpMethod.GET, HttpMethod.PUT, HttpMethod.DELETE])
        group.add_route("health", "/health", methods=[HttpMethod.GET])
        return group

    @staticmethod
    def user_routes() -> RouteGroup:
        group = RouteGroup(name="users", base_path="/api/v1/users", service_name="user-service", tags=["solace", "user"])
        group.add_route("profile", "/profile", methods=[HttpMethod.GET, HttpMethod.PUT])
        group.add_route("preferences", "/preferences", methods=[HttpMethod.GET, HttpMethod.PUT])
        group.add_route("progress", "/progress", methods=[HttpMethod.GET])
        group.add_route("consent", "/consent", methods=[HttpMethod.GET, HttpMethod.POST])
        group.add_route("sessions", "/sessions", methods=[HttpMethod.GET])
        return group

    @staticmethod
    def session_routes() -> RouteGroup:
        group = RouteGroup(name="sessions", base_path="/api/v1/sessions", service_name="session-service", tags=["solace", "session"])
        group.add_route("create", "", methods=[HttpMethod.POST])
        group.add_route("detail", "/{id}", methods=[HttpMethod.GET])
        group.add_route("end", "/{id}/end", methods=[HttpMethod.PUT])
        group.add_route("history", "/{id}/history", methods=[HttpMethod.GET])
        return group

    @staticmethod
    def assessment_routes() -> RouteGroup:
        group = RouteGroup(name="assessments", base_path="/api/v1/assessments", service_name="diagnosis-service", tags=["solace", "assessment"])
        group.add_route("phq9", "/phq9", methods=[HttpMethod.POST, HttpMethod.GET])
        group.add_route("gad7", "/gad7", methods=[HttpMethod.POST, HttpMethod.GET])
        group.add_route("history", "/history", methods=[HttpMethod.GET])
        group.add_route("result", "/{id}", methods=[HttpMethod.GET])
        return group

    @staticmethod
    def admin_routes() -> RouteGroup:
        group = RouteGroup(name="admin", base_path="/api/v1/admin", service_name="admin-service", tags=["solace", "admin"])
        group.add_route("users", "/users", methods=[HttpMethod.GET])
        group.add_route("analytics", "/analytics", methods=[HttpMethod.GET])
        group.add_route("alerts", "/alerts", methods=[HttpMethod.GET, HttpMethod.POST])
        group.add_route("config", "/config", methods=[HttpMethod.GET, HttpMethod.PUT])
        return group

    @staticmethod
    def chat_routes() -> RouteGroup:
        group = RouteGroup(name="chat", base_path="/api/v1/chat", service_name="orchestrator-service", tags=["solace", "chat"])
        group.add_route("message", "/message", methods=[HttpMethod.POST])
        group.add_route("history", "/history", methods=[HttpMethod.GET])
        group.add_route("voice", "/voice", methods=[HttpMethod.POST])
        return group


def create_solace_route_config(settings: RouteSettings | None = None) -> RouteConfig:
    """Create pre-configured routes for Solace-AI platform."""
    config = RouteConfig(settings)
    config._groups["orchestrator"] = ServiceRoutes.orchestrator_routes()
    config._groups["users"] = ServiceRoutes.user_routes()
    config._groups["sessions"] = ServiceRoutes.session_routes()
    config._groups["assessments"] = ServiceRoutes.assessment_routes()
    config._groups["admin"] = ServiceRoutes.admin_routes()
    config._groups["chat"] = ServiceRoutes.chat_routes()
    logger.info("solace_routes_configured", groups=6)
    return config
