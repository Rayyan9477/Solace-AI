"""
Solace-AI Orchestrator Service - Service Clients.
HTTP clients for service-to-service communication with retry and circuit breaker.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID
import asyncio
import random
import structlog
import httpx

from ..config import ServiceEndpoints, get_config

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ClientConfig:
    """HTTP client configuration."""

    base_url: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 30
    verify_ssl: bool = True
    ssl_cert_path: str | None = None


@dataclass
class ServiceResponse(Generic[T]):
    """Generic service response wrapper."""

    success: bool
    data: T | None = None
    error: str | None = None
    status_code: int = 0
    response_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        """Record successful call."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning("circuit_breaker_opened", failures=self._failure_count)

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True
        return False


class BaseServiceClient:
    """Base HTTP client with retry, circuit breaker, and optional service auth."""

    def __init__(
        self,
        config: ClientConfig,
        token_manager: Any = None,
        service_name: str = "orchestrator-service",
    ) -> None:
        self._config = config
        self._circuit = CircuitBreaker(
            config.circuit_failure_threshold, config.circuit_recovery_timeout
        )
        self._client: httpx.AsyncClient | None = None
        self._token_manager = token_manager
        self._service_name = service_name

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with SSL verification."""
        if self._client is None:
            verify: bool | str = self._config.ssl_cert_path or self._config.verify_ssl
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds),
                headers={"Content-Type": "application/json"},
                verify=verify,
            )
        return self._client

    def _get_auth_headers(self) -> dict[str, str]:
        """Get service authentication headers if token manager is configured."""
        if self._token_manager is None:
            return {}
        try:
            credentials = self._token_manager.get_or_create_token(self._service_name)
            return {
                "Authorization": f"Bearer {credentials.token}",
                "X-Service-Name": self._service_name,
            }
        except Exception as e:
            logger.warning("service_auth_header_failed", error=str(e))
            return {}

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> ServiceResponse[dict[str, Any]]:
        """Execute HTTP request with retry logic."""
        if not self._circuit.allow_request():
            return ServiceResponse(success=False, error="Circuit breaker open", status_code=503)
        start_time = datetime.now(timezone.utc)
        last_error: str | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                client = await self._get_client()
                request_headers = {**self._get_auth_headers(), **(headers or {})}
                response = await client.request(
                    method, path, json=data, params=params, headers=request_headers
                )
                elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                if response.status_code >= 200 and response.status_code < 300:
                    self._circuit.record_success()
                    try:
                        response_data = response.json() if response.content else None
                    except (ValueError, UnicodeDecodeError):
                        response_data = {"raw": response.text}
                    return ServiceResponse(
                        success=True,
                        data=response_data,
                        status_code=response.status_code,
                        response_time_ms=elapsed_ms,
                    )
                if response.status_code >= 500:
                    last_error = f"Server error: {response.status_code}"
                    self._circuit.record_failure()
                else:
                    return ServiceResponse(
                        success=False,
                        error=response.text,
                        status_code=response.status_code,
                        response_time_ms=elapsed_ms,
                    )
            except httpx.TimeoutException:
                last_error = "Request timeout"
                self._circuit.record_failure()
            except httpx.RequestError as e:
                last_error = str(e)
                self._circuit.record_failure()
            if attempt < self._config.max_retries:
                base_delay = self._config.retry_delay_seconds * (2 ** attempt)
                jitter = random.uniform(0, base_delay * 0.1)
                delay = min(base_delay + jitter, 30.0)
                await asyncio.sleep(delay)
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ServiceResponse(
            success=False, error=last_error, status_code=503, response_time_ms=elapsed_ms
        )

    async def get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> ServiceResponse[dict[str, Any]]:
        """Execute GET request."""
        return await self._request("GET", path, params=params)

    async def post(
        self, path: str, data: dict[str, Any] | None = None
    ) -> ServiceResponse[dict[str, Any]]:
        """Execute POST request."""
        return await self._request("POST", path, data=data)

    async def put(
        self, path: str, data: dict[str, Any] | None = None
    ) -> ServiceResponse[dict[str, Any]]:
        """Execute PUT request."""
        return await self._request("PUT", path, data=data)

    def get_health(self) -> dict[str, Any]:
        """Get client health status."""
        return {"circuit_state": self._circuit.state.value, "base_url": self._config.base_url}


class PersonalityServiceClient(BaseServiceClient):
    """Client for Personality Service."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        super().__init__(ClientConfig(base_url=eps.personality_service_url), token_manager=token_manager)

    async def get_style(self, user_id: UUID) -> ServiceResponse[dict[str, Any]]:
        """Get personality style for user."""
        return await self.get(f"/api/v1/personality/{user_id}/style")

    async def analyze_text(self, user_id: UUID, text: str) -> ServiceResponse[dict[str, Any]]:
        """Analyze text for personality indicators."""
        return await self.post(f"/api/v1/personality/{user_id}/analyze", {"text": text})


class DiagnosisServiceClient(BaseServiceClient):
    """Client for Diagnosis Service."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        super().__init__(ClientConfig(base_url=eps.diagnosis_service_url), token_manager=token_manager)

    async def get_assessment(
        self, user_id: UUID, session_id: UUID
    ) -> ServiceResponse[dict[str, Any]]:
        """Get current assessment for user."""
        return await self.get(
            f"/api/v1/diagnosis/{user_id}/assessment", {"session_id": str(session_id)}
        )

    async def analyze_symptoms(
        self, user_id: UUID, symptoms: list[str]
    ) -> ServiceResponse[dict[str, Any]]:
        """Analyze symptoms for assessment."""
        return await self.post(f"/api/v1/diagnosis/{user_id}/symptoms", {"symptoms": symptoms})


class TherapyServiceClient(BaseServiceClient):
    """Client for Therapy Service."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        super().__init__(ClientConfig(base_url=eps.therapy_service_url), token_manager=token_manager)

    async def process_message(
        self,
        session_id: UUID,
        user_id: UUID,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> ServiceResponse[dict[str, Any]]:
        """Process a message in the therapy session."""
        return await self.post(
            f"/api/v1/therapy/sessions/{session_id}/message",
            {
                "session_id": str(session_id),
                "user_id": str(user_id),
                "message": message,
                "conversation_history": conversation_history or [],
            },
        )

    async def get_techniques(self, modality: str | None = None) -> ServiceResponse[dict[str, Any]]:
        """Get available therapeutic techniques."""
        params = {"modality": modality} if modality else None
        return await self.get("/api/v1/therapy/techniques", params)

    async def start_session(
        self, user_id: UUID, treatment_plan_id: UUID
    ) -> ServiceResponse[dict[str, Any]]:
        """Start a new therapy session."""
        return await self.post(
            "/api/v1/therapy/sessions/start",
            {"user_id": str(user_id), "treatment_plan_id": str(treatment_plan_id)},
        )

    async def end_session(
        self, session_id: UUID, user_id: UUID, generate_summary: bool = True
    ) -> ServiceResponse[dict[str, Any]]:
        """End a therapy session."""
        return await self.post(
            f"/api/v1/therapy/sessions/{session_id}/end",
            {
                "session_id": str(session_id),
                "user_id": str(user_id),
                "generate_summary": generate_summary,
            },
        )


# Alias for backwards compatibility
class TreatmentServiceClient(TherapyServiceClient):
    """Client for Treatment Service (alias for TherapyServiceClient)."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        # Use treatment_service_url for backwards compatibility
        BaseServiceClient.__init__(self, ClientConfig(base_url=eps.treatment_service_url), token_manager=token_manager)

    async def get_plan(self, user_id: UUID) -> ServiceResponse[dict[str, Any]]:
        """Get treatment plan for user."""
        return await self.get(f"/api/v1/therapy/{user_id}/plan")

    async def get_interventions(
        self, user_id: UUID, context: str
    ) -> ServiceResponse[dict[str, Any]]:
        """Get recommended interventions."""
        return await self.post(f"/api/v1/therapy/{user_id}/interventions", {"context": context})


class MemoryServiceClient(BaseServiceClient):
    """Client for Memory Service."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        super().__init__(ClientConfig(base_url=eps.memory_service_url), token_manager=token_manager)

    async def get_context(self, user_id: UUID, session_id: UUID) -> ServiceResponse[dict[str, Any]]:
        """Get conversation context."""
        return await self.post(
            "/api/v1/memory/context", {"user_id": str(user_id), "session_id": str(session_id)}
        )

    async def store_memory(
        self, user_id: UUID, content: str, memory_type: str
    ) -> ServiceResponse[dict[str, Any]]:
        """Store memory entry."""
        return await self.post(
            "/api/v1/memory/store",
            {"user_id": str(user_id), "content": content, "content_type": memory_type},
        )

    async def assemble_context(
        self,
        user_id: UUID,
        session_id: UUID,
        current_message: str,
        token_budget: int = 4000,
    ) -> ServiceResponse[dict[str, Any]]:
        """Assemble context for LLM within token budget."""
        return await self.post(
            "/api/v1/memory/context",
            {
                "user_id": str(user_id),
                "session_id": str(session_id),
                "current_message": current_message,
                "token_budget": token_budget,
            },
        )


class SafetyServiceClient(BaseServiceClient):
    """Client for Safety Service."""

    def __init__(self, endpoints: ServiceEndpoints | None = None, token_manager: Any = None) -> None:
        eps = endpoints or get_config().endpoints()
        super().__init__(ClientConfig(base_url=eps.safety_service_url), token_manager=token_manager)

    async def check_safety(
        self,
        user_id: UUID,
        session_id: UUID,
        message: str,
        check_type: str = "FULL_ASSESSMENT",
    ) -> ServiceResponse[dict[str, Any]]:
        """Perform safety check on message content."""
        return await self.post(
            "/api/v1/safety/check",
            {
                "user_id": str(user_id),
                "session_id": str(session_id),
                "content": message,
                "check_type": check_type,
            },
        )

    async def get_crisis_resources(self, severity: str) -> ServiceResponse[dict[str, Any]]:
        """Get crisis resources for given severity level."""
        return await self.get("/api/v1/safety/resources", {"severity": severity})

    async def trigger_escalation(
        self,
        user_id: UUID,
        session_id: UUID,
        crisis_level: str,
        reason: str,
    ) -> ServiceResponse[dict[str, Any]]:
        """Trigger escalation for crisis situation."""
        return await self.post(
            "/api/v1/safety/escalate",
            {
                "user_id": str(user_id),
                "session_id": str(session_id),
                "crisis_level": crisis_level,
                "reason": reason,
            },
        )


class ServiceClientFactory:
    """Factory for creating service clients with optional service auth."""

    def __init__(
        self,
        endpoints: ServiceEndpoints | None = None,
        token_manager: Any = None,
    ) -> None:
        self._endpoints = endpoints or get_config().endpoints()
        self._token_manager = token_manager
        self._clients: dict[str, BaseServiceClient] = {}

    def personality(self) -> PersonalityServiceClient:
        """Get personality service client."""
        if "personality" not in self._clients:
            self._clients["personality"] = PersonalityServiceClient(self._endpoints, self._token_manager)
        return self._clients["personality"]  # type: ignore

    def diagnosis(self) -> DiagnosisServiceClient:
        """Get diagnosis service client."""
        if "diagnosis" not in self._clients:
            self._clients["diagnosis"] = DiagnosisServiceClient(self._endpoints, self._token_manager)
        return self._clients["diagnosis"]  # type: ignore

    def treatment(self) -> TreatmentServiceClient:
        """Get treatment service client (alias for therapy)."""
        if "treatment" not in self._clients:
            self._clients["treatment"] = TreatmentServiceClient(self._endpoints, self._token_manager)
        return self._clients["treatment"]  # type: ignore

    def therapy(self) -> TherapyServiceClient:
        """Get therapy service client."""
        if "therapy" not in self._clients:
            self._clients["therapy"] = TherapyServiceClient(self._endpoints, self._token_manager)
        return self._clients["therapy"]  # type: ignore

    def memory(self) -> MemoryServiceClient:
        """Get memory service client."""
        if "memory" not in self._clients:
            self._clients["memory"] = MemoryServiceClient(self._endpoints, self._token_manager)
        return self._clients["memory"]  # type: ignore

    def safety(self) -> SafetyServiceClient:
        """Get safety service client."""
        if "safety" not in self._clients:
            self._clients["safety"] = SafetyServiceClient(self._endpoints, self._token_manager)
        return self._clients["safety"]  # type: ignore

    async def close_all(self) -> None:
        """Close all clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    def get_health(self) -> dict[str, Any]:
        """Get health status of all clients."""
        return {name: client.get_health() for name, client in self._clients.items()}
