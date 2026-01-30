"""
Solace-AI Service-to-Service Authentication.
Provides service identity verification and token management for inter-service communication.
"""
from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .auth import AuthSettings, JWTManager, TokenType

logger = structlog.get_logger(__name__)


class ServiceIdentity(str, Enum):
    """Known service identities in the Solace-AI platform."""
    ORCHESTRATOR = "orchestrator-service"
    SAFETY = "safety-service"
    THERAPY = "therapy-service"
    DIAGNOSIS = "diagnosis-service"
    PERSONALITY = "personality-service"
    MEMORY = "memory-service"
    USER = "user-service"
    NOTIFICATION = "notification-service"


class ServicePermission(str, Enum):
    """Permissions for inter-service operations."""
    # Read operations
    READ_USER = "service:read:user"
    READ_SAFETY = "service:read:safety"
    READ_THERAPY = "service:read:therapy"
    READ_DIAGNOSIS = "service:read:diagnosis"
    READ_PERSONALITY = "service:read:personality"
    READ_MEMORY = "service:read:memory"
    READ_NOTIFICATION = "service:read:notification"
    # Write operations
    WRITE_USER = "service:write:user"
    WRITE_SAFETY = "service:write:safety"
    WRITE_THERAPY = "service:write:therapy"
    WRITE_DIAGNOSIS = "service:write:diagnosis"
    WRITE_PERSONALITY = "service:write:personality"
    WRITE_MEMORY = "service:write:memory"
    WRITE_NOTIFICATION = "service:write:notification"
    # Critical operations
    TRIGGER_ESCALATION = "service:escalate"
    SEND_CRISIS_ALERT = "service:crisis_alert"
    DELETE_USER_DATA = "service:delete_user_data"


# Service permission matrix - defines what each service can access
SERVICE_PERMISSIONS: dict[ServiceIdentity, list[ServicePermission]] = {
    ServiceIdentity.ORCHESTRATOR: [
        # Orchestrator needs access to all services
        ServicePermission.READ_USER,
        ServicePermission.READ_SAFETY,
        ServicePermission.READ_THERAPY,
        ServicePermission.READ_DIAGNOSIS,
        ServicePermission.READ_PERSONALITY,
        ServicePermission.READ_MEMORY,
        ServicePermission.WRITE_THERAPY,
        ServicePermission.WRITE_MEMORY,
    ],
    ServiceIdentity.SAFETY: [
        ServicePermission.READ_USER,
        ServicePermission.WRITE_NOTIFICATION,
        ServicePermission.TRIGGER_ESCALATION,
        ServicePermission.SEND_CRISIS_ALERT,
    ],
    ServiceIdentity.THERAPY: [
        ServicePermission.READ_USER,
        ServicePermission.READ_MEMORY,
        ServicePermission.READ_PERSONALITY,
        ServicePermission.READ_DIAGNOSIS,
        ServicePermission.WRITE_MEMORY,
    ],
    ServiceIdentity.DIAGNOSIS: [
        ServicePermission.READ_USER,
        ServicePermission.READ_MEMORY,
        ServicePermission.WRITE_MEMORY,
    ],
    ServiceIdentity.PERSONALITY: [
        ServicePermission.READ_USER,
        ServicePermission.READ_MEMORY,
    ],
    ServiceIdentity.MEMORY: [
        ServicePermission.READ_USER,
    ],
    ServiceIdentity.USER: [],  # User service is the source of truth for users
    ServiceIdentity.NOTIFICATION: [
        ServicePermission.READ_USER,
    ],
}


class ServiceAuthSettings(BaseSettings):
    """Service authentication configuration."""
    service_name: str = Field(..., description="Name of this service")
    service_token_expire_minutes: int = Field(default=60, ge=5, le=1440)
    token_refresh_threshold_seconds: int = Field(default=300)
    enable_token_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300)
    require_service_auth: bool = Field(default=True)
    allowed_services: list[str] = Field(
        default_factory=lambda: [s.value for s in ServiceIdentity],
        description="List of allowed service names that can call this service",
    )
    model_config = SettingsConfigDict(
        env_prefix="SERVICE_AUTH_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class ServiceCredentials:
    """Service credentials for authentication."""
    service_name: str
    token: str
    expires_at: datetime
    permissions: list[str] = field(default_factory=list)
    refresh_threshold_seconds: int = 300

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at

    @property
    def should_refresh(self) -> bool:
        """Check if token should be refreshed based on threshold."""
        remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        return remaining < self.refresh_threshold_seconds


@dataclass
class ServiceAuthResult:
    """Result of service authentication."""
    authenticated: bool
    service_name: str | None = None
    permissions: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def success(cls, service_name: str, permissions: list[str]) -> ServiceAuthResult:
        return cls(authenticated=True, service_name=service_name, permissions=permissions)

    @classmethod
    def failure(cls, error: str) -> ServiceAuthResult:
        return cls(authenticated=False, error=error)


class ServiceTokenManager:
    """
    Manages service tokens for inter-service authentication.

    Handles token creation, validation, caching, and automatic refresh.
    """

    def __init__(
        self,
        auth_settings: AuthSettings | None = None,
        service_settings: ServiceAuthSettings | None = None,
    ) -> None:
        self._auth_settings = auth_settings
        self._service_settings = service_settings
        self._jwt_manager = JWTManager(auth_settings) if auth_settings else None
        self._token_cache: dict[str, ServiceCredentials] = {}

    def _get_jwt_manager(self) -> JWTManager:
        """Get or create JWT manager."""
        if self._jwt_manager is None:
            try:
                self._jwt_manager = JWTManager(self._auth_settings)
            except Exception as e:
                logger.error("jwt_manager_creation_failed", error=str(e))
                raise ValueError(
                    "JWT manager not configured. Set AUTH_SECRET_KEY environment variable."
                ) from e
        return self._jwt_manager

    def create_service_token(
        self,
        service_name: str,
        target_service: str | None = None,
        additional_permissions: list[str] | None = None,
        expire_minutes: int | None = None,
    ) -> ServiceCredentials:
        """
        Create a service token for inter-service communication.

        Args:
            service_name: Name of the calling service
            target_service: Optional target service for scoped permissions
            additional_permissions: Additional permissions to include
            expire_minutes: Custom expiration time

        Returns:
            ServiceCredentials with the token and metadata
        """
        jwt_manager = self._get_jwt_manager()

        # Get base permissions for the service
        service_identity = self._get_service_identity(service_name)
        base_permissions = [
            p.value for p in SERVICE_PERMISSIONS.get(service_identity, [])
        ]

        # Add additional permissions if provided
        all_permissions = list(set(base_permissions + (additional_permissions or [])))

        # Create expiration
        expire_mins = expire_minutes or (
            self._service_settings.service_token_expire_minutes
            if self._service_settings
            else 60
        )
        expires_at = datetime.now(UTC) + __import__("datetime").timedelta(
            minutes=expire_mins
        )

        # Create the token
        token = jwt_manager.create_service_token(
            service_name=service_name,
            permissions=all_permissions,
            expire_minutes=expire_mins,
        )

        refresh_threshold = (
            self._service_settings.token_refresh_threshold_seconds
            if self._service_settings
            else 300
        )
        credentials = ServiceCredentials(
            service_name=service_name,
            token=token,
            expires_at=expires_at,
            permissions=all_permissions,
            refresh_threshold_seconds=refresh_threshold,
        )

        # Cache the token if enabled
        if self._service_settings and self._service_settings.enable_token_caching:
            cache_key = f"{service_name}:{target_service or 'all'}"
            self._token_cache[cache_key] = credentials

        logger.debug(
            "service_token_created",
            service_name=service_name,
            target_service=target_service,
            expires_at=expires_at.isoformat(),
        )

        return credentials

    def get_cached_token(
        self,
        service_name: str,
        target_service: str | None = None,
    ) -> ServiceCredentials | None:
        """Get cached token if valid, or None if expired/missing."""
        cache_key = f"{service_name}:{target_service or 'all'}"
        credentials = self._token_cache.get(cache_key)

        if credentials is None:
            return None

        # Check if expired or should refresh
        if credentials.is_expired or credentials.should_refresh:
            del self._token_cache[cache_key]
            return None

        return credentials

    def get_or_create_token(
        self,
        service_name: str,
        target_service: str | None = None,
    ) -> ServiceCredentials:
        """Get cached token or create a new one."""
        cached = self.get_cached_token(service_name, target_service)
        if cached:
            return cached
        return self.create_service_token(service_name, target_service)

    def validate_service_token(
        self,
        token: str,
        required_permissions: list[str] | None = None,
    ) -> ServiceAuthResult:
        """
        Validate a service token.

        Args:
            token: The JWT token to validate
            required_permissions: Permissions required for the operation

        Returns:
            ServiceAuthResult indicating success or failure
        """
        jwt_manager = self._get_jwt_manager()

        # Decode and validate the token
        result = jwt_manager.decode_token(token, TokenType.SERVICE)

        if not result.success or not result.payload:
            return ServiceAuthResult.failure(
                result.error_message or "Invalid service token"
            )

        payload = result.payload

        # Verify service identity
        if not payload.sub.startswith("service:"):
            return ServiceAuthResult.failure("Not a service token")

        service_name = payload.sub.replace("service:", "")

        # Check if service is allowed
        if self._service_settings:
            if service_name not in self._service_settings.allowed_services:
                return ServiceAuthResult.failure(
                    f"Service {service_name} not allowed"
                )

        # Check required permissions
        if required_permissions:
            missing = set(required_permissions) - set(payload.permissions)
            if missing:
                return ServiceAuthResult.failure(
                    f"Missing required permissions: {missing}"
                )

        return ServiceAuthResult.success(service_name, payload.permissions)

    def clear_cache(self, service_name: str | None = None) -> None:
        """Clear token cache."""
        if service_name:
            keys_to_remove = [
                k for k in self._token_cache if k.startswith(f"{service_name}:")
            ]
            for key in keys_to_remove:
                del self._token_cache[key]
        else:
            self._token_cache.clear()

    def _get_service_identity(self, service_name: str) -> ServiceIdentity:
        """Get ServiceIdentity enum from service name.

        Raises:
            ValueError: If service_name doesn't match any known ServiceIdentity.
                Unknown services must NOT be granted default permissions.
        """
        try:
            return ServiceIdentity(service_name)
        except ValueError:
            logger.error(
                "unknown_service_identity_rejected",
                service_name=service_name,
                known_services=[s.value for s in ServiceIdentity],
            )
            raise ValueError(
                f"Unknown service identity: '{service_name}'. "
                f"Known services: {[s.value for s in ServiceIdentity]}"
            ) from None


class ServiceAuthenticatedClient:
    """
    HTTP client wrapper that automatically adds service authentication.

    Wraps httpx.AsyncClient with automatic service token injection.
    """

    def __init__(
        self,
        service_name: str,
        token_manager: ServiceTokenManager,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._service_name = service_name
        self._token_manager = token_manager
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_auth_headers(self, target_service: str | None = None) -> dict[str, str]:
        """Get authentication headers with service token."""
        credentials = self._token_manager.get_or_create_token(
            self._service_name, target_service
        )
        return {
            "Authorization": f"Bearer {credentials.token}",
            "X-Service-Name": self._service_name,
        }

    async def request(
        self,
        method: str,
        url: str,
        target_service: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated HTTP request."""
        client = await self._get_client()
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers(target_service))
        return await client.request(method, url, headers=headers, **kwargs)

    async def get(
        self,
        url: str,
        target_service: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated GET request."""
        return await self.request("GET", url, target_service, **kwargs)

    async def post(
        self,
        url: str,
        target_service: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated POST request."""
        return await self.request("POST", url, target_service, **kwargs)

    async def put(
        self,
        url: str,
        target_service: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated PUT request."""
        return await self.request("PUT", url, target_service, **kwargs)

    async def delete(
        self,
        url: str,
        target_service: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make authenticated DELETE request."""
        return await self.request("DELETE", url, target_service, **kwargs)


# FastAPI dependency for service authentication
async def get_service_auth_dependency(
    token_manager: ServiceTokenManager,
    required_permissions: list[str] | None = None,
) -> Callable[[str], Awaitable[ServiceAuthResult]]:
    """
    Create a FastAPI dependency for service authentication.

    Usage:
        from fastapi import Depends, Header

        @app.get("/internal/endpoint")
        async def internal_endpoint(
            service_auth: ServiceAuthResult = Depends(
                get_service_auth_dependency(token_manager, ["service:read:memory"])
            ),
        ):
            if not service_auth.authenticated:
                raise HTTPException(status_code=401, detail=service_auth.error)
            ...
    """
    async def _verify_service(
        authorization: str | None = None,
    ) -> ServiceAuthResult:
        if not authorization:
            return ServiceAuthResult.failure("Missing Authorization header")

        if not authorization.startswith("Bearer "):
            return ServiceAuthResult.failure("Invalid Authorization format")

        token = authorization[7:]  # Remove "Bearer " prefix
        return token_manager.validate_service_token(token, required_permissions)

    return _verify_service


# Module-level singleton
_service_token_manager: ServiceTokenManager | None = None
_service_token_manager_lock = threading.Lock()


def get_service_token_manager() -> ServiceTokenManager:
    """Get or create the global service token manager (thread-safe)."""
    global _service_token_manager
    if _service_token_manager is None:
        with _service_token_manager_lock:
            if _service_token_manager is None:
                _service_token_manager = ServiceTokenManager()
    return _service_token_manager


def initialize_service_auth(
    auth_settings: AuthSettings | None = None,
    service_settings: ServiceAuthSettings | None = None,
) -> ServiceTokenManager:
    """Initialize the service authentication system."""
    global _service_token_manager
    _service_token_manager = ServiceTokenManager(auth_settings, service_settings)
    return _service_token_manager
