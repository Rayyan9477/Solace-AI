"""
Solace-AI API Gateway - CORS Configuration.
Implements Cross-Origin Resource Sharing (CORS) policy management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import re
import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class CORSPreset(str, Enum):
    """Pre-defined CORS configuration presets."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    DEVELOPMENT = "development"


class CORSConfig(BaseSettings):
    """CORS configuration settings."""
    origins: str = Field(default="", description="Comma-separated allowed origins. Must be set explicitly - wildcard not allowed with credentials.")
    methods: str = Field(default="GET,POST,PUT,PATCH,DELETE,OPTIONS")
    headers: str = Field(default="Content-Type,Authorization,X-Request-ID,X-Correlation-ID")
    expose_headers: str = Field(default="X-Request-ID,X-Correlation-ID,X-RateLimit-Limit,X-RateLimit-Remaining")
    max_age: int = Field(default=86400, ge=0)
    credentials: bool = Field(default=True)
    preflight_continue: bool = Field(default=False)
    private_network: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="CORS_", env_file=".env", extra="ignore")

    @property
    def origins_list(self) -> list[str]:
        if self.origins == "*":
            return ["*"]
        return [o.strip() for o in self.origins.split(",") if o.strip()]

    @property
    def methods_list(self) -> list[str]:
        return [m.strip().upper() for m in self.methods.split(",") if m.strip()]

    @property
    def headers_list(self) -> list[str]:
        return [h.strip() for h in self.headers.split(",") if h.strip()]

    @property
    def expose_headers_list(self) -> list[str]:
        return [h.strip() for h in self.expose_headers.split(",") if h.strip()]


@dataclass
class CORSPolicy:
    """CORS policy definition for a route or service."""
    name: str
    origins: list[str] = field(default_factory=lambda: ["*"])
    methods: list[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    headers: list[str] = field(default_factory=lambda: ["Content-Type", "Authorization"])
    expose_headers: list[str] = field(default_factory=list)
    max_age: int = 86400
    credentials: bool = True
    preflight_continue: bool = False
    private_network: bool = False
    origin_patterns: list[str] = field(default_factory=list)
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def allows_origin(self, origin: str) -> bool:
        if "*" in self.origins:
            return True
        if origin in self.origins:
            return True
        for pattern in self.origin_patterns:
            if re.match(pattern, origin):
                return True
        return False

    def allows_method(self, method: str) -> bool:
        return method.upper() in [m.upper() for m in self.methods]

    def allows_header(self, header: str) -> bool:
        if "*" in self.headers:
            return True
        return header.lower() in [h.lower() for h in self.headers]

    def to_kong_plugin_config(self) -> dict[str, Any]:
        return {"name": "cors", "config": {"origins": self.origins, "methods": self.methods, "headers": self.headers, "exposed_headers": self.expose_headers, "max_age": self.max_age, "credentials": self.credentials, "preflight_continue": self.preflight_continue, "private_network": self.private_network}}

    @classmethod
    def from_preset(cls, preset: CORSPreset, name: str = "default") -> CORSPolicy:
        presets = {
            CORSPreset.STRICT: cls(name=name, origins=[], methods=["GET", "POST"], headers=["Content-Type", "Authorization"], max_age=3600, credentials=False),
            CORSPreset.STANDARD: cls(name=name, origins=["https://app.solace-ai.com", "https://www.solace-ai.com"], methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], headers=["Content-Type", "Authorization", "X-Request-ID"], expose_headers=["X-Request-ID", "X-RateLimit-Remaining"], max_age=86400, credentials=True),
            CORSPreset.PERMISSIVE: cls(name=name, origins=["*"], methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"], headers=["*"], expose_headers=["*"], max_age=86400, credentials=False),
            CORSPreset.DEVELOPMENT: cls(name=name, origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"], methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], headers=["*"], expose_headers=["*"], max_age=86400, credentials=True),
        }
        return presets[preset]


@dataclass
class CORSRequest:
    """Parsed CORS request information."""
    origin: str | None
    method: str
    request_method: str | None = None
    request_headers: list[str] = field(default_factory=list)
    is_preflight: bool = False

    @classmethod
    def from_headers(cls, headers: dict[str, str], method: str) -> CORSRequest:
        origin = headers.get("Origin") or headers.get("origin")
        is_preflight = method.upper() == "OPTIONS" and "Access-Control-Request-Method" in headers
        request_method = headers.get("Access-Control-Request-Method")
        request_headers_str = headers.get("Access-Control-Request-Headers", "")
        request_headers = [h.strip() for h in request_headers_str.split(",") if h.strip()]
        return cls(origin=origin, method=method, request_method=request_method, request_headers=request_headers, is_preflight=is_preflight)


@dataclass
class CORSResponse:
    """CORS response headers."""
    allow_origin: str | None = None
    allow_methods: list[str] | None = None
    allow_headers: list[str] | None = None
    expose_headers: list[str] | None = None
    max_age: int | None = None
    allow_credentials: bool = False
    allow_private_network: bool = False

    def to_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.allow_origin:
            headers["Access-Control-Allow-Origin"] = self.allow_origin
        if self.allow_methods:
            headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        if self.allow_headers:
            headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        if self.max_age is not None:
            headers["Access-Control-Max-Age"] = str(self.max_age)
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        if self.allow_private_network:
            headers["Access-Control-Allow-Private-Network"] = "true"
        return headers


class CORSHandler:
    """CORS request handler."""

    def __init__(self, config: CORSConfig | None = None) -> None:
        self._config = config or CORSConfig()
        self._policies: dict[str, CORSPolicy] = {}
        self._route_policies: dict[str, str] = {}
        self._service_policies: dict[str, str] = {}
        self._default_policy = CORSPolicy(name="default", origins=self._config.origins_list, methods=self._config.methods_list, headers=self._config.headers_list, expose_headers=self._config.expose_headers_list, max_age=self._config.max_age, credentials=self._config.credentials, preflight_continue=self._config.preflight_continue, private_network=self._config.private_network)

    def add_policy(self, policy: CORSPolicy) -> None:
        self._policies[policy.name] = policy
        logger.info("cors_policy_added", name=policy.name, origins=policy.origins)

    def set_route_policy(self, route_name: str, policy_name: str) -> None:
        if policy_name not in self._policies:
            raise ValueError(f"Policy '{policy_name}' not found")
        self._route_policies[route_name] = policy_name

    def set_service_policy(self, service_name: str, policy_name: str) -> None:
        if policy_name not in self._policies:
            raise ValueError(f"Policy '{policy_name}' not found")
        self._service_policies[service_name] = policy_name

    def get_policy(self, route_name: str | None = None, service_name: str | None = None) -> CORSPolicy:
        if route_name and route_name in self._route_policies:
            return self._policies[self._route_policies[route_name]]
        if service_name and service_name in self._service_policies:
            return self._policies[self._service_policies[service_name]]
        return self._default_policy

    def handle_request(self, request: CORSRequest, route_name: str | None = None, service_name: str | None = None) -> CORSResponse:
        if not request.origin:
            return CORSResponse()
        policy = self.get_policy(route_name, service_name)
        if not policy.enabled:
            return CORSResponse()
        if not policy.allows_origin(request.origin):
            logger.warning("cors_origin_rejected", origin=request.origin, policy=policy.name)
            return CORSResponse()
        allow_origin = request.origin if policy.credentials else ("*" if "*" in policy.origins else request.origin)
        if request.is_preflight:
            method_to_check = request.request_method or request.method
            if not policy.allows_method(method_to_check):
                logger.warning("cors_method_rejected", method=method_to_check, policy=policy.name)
                return CORSResponse()
            for header in request.request_headers:
                if not policy.allows_header(header):
                    logger.warning("cors_header_rejected", header=header, policy=policy.name)
                    return CORSResponse()
            return CORSResponse(allow_origin=allow_origin, allow_methods=policy.methods, allow_headers=policy.headers, max_age=policy.max_age, allow_credentials=policy.credentials, allow_private_network=policy.private_network)
        return CORSResponse(allow_origin=allow_origin, expose_headers=policy.expose_headers if policy.expose_headers else None, allow_credentials=policy.credentials)

    def handle_headers(self, headers: dict[str, str], method: str, route_name: str | None = None, service_name: str | None = None) -> dict[str, str]:
        request = CORSRequest.from_headers(headers, method)
        response = self.handle_request(request, route_name, service_name)
        return response.to_headers()


def create_solace_cors_handler(config: CORSConfig | None = None) -> CORSHandler:
    """Create pre-configured CORS handler for Solace-AI."""
    handler = CORSHandler(config)
    handler.add_policy(CORSPolicy.from_preset(CORSPreset.STANDARD, "production"))
    handler.add_policy(CORSPolicy.from_preset(CORSPreset.DEVELOPMENT, "development"))
    handler.add_policy(CORSPolicy(name="api-clients", origins=["*"], methods=["GET", "POST", "PUT", "PATCH", "DELETE"], headers=["Content-Type", "Authorization", "X-API-Key"], credentials=False, tags=["api"]))
    handler.add_policy(CORSPolicy(name="admin", origins=["https://admin.solace-ai.com"], methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], headers=["Content-Type", "Authorization", "X-Request-ID"], credentials=True, tags=["admin"]))
    handler.set_service_policy("orchestrator-service", "production")
    handler.set_service_policy("user-service", "production")
    handler.set_service_policy("admin-service", "admin")
    logger.info("solace_cors_handler_created", policies=4)
    return handler
