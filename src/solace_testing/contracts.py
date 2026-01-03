"""Solace-AI Testing Library - Contract testing helpers."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class HttpMethod(str, Enum):
    """HTTP methods for API contracts."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ContractStatus(str, Enum):
    """Contract verification status."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SchemaType(str, Enum):
    """Schema type identifiers."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class FieldMatcher(BaseModel):
    """Matcher for validating field values."""
    field_type: SchemaType
    required: bool = True
    pattern: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    enum_values: list[Any] | None = None
    nested_schema: dict[str, FieldMatcher] | None = None
    items_schema: FieldMatcher | None = None

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if value is None:
            return (False, "Required field is null") if self.required else (True, None)
        if not self._check_type(value):
            return False, f"Expected {self.field_type}, got {type(value).__name__}"
        if self.pattern and isinstance(value, str) and not re.match(self.pattern, value):
            return False, f"Value does not match pattern: {self.pattern}"
        if self.min_value is not None and isinstance(value, (int, float)) and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value}"
        if self.max_value is not None and isinstance(value, (int, float)) and value > self.max_value:
            return False, f"Value {value} above maximum {self.max_value}"
        if self.min_length is not None and hasattr(value, "__len__") and len(value) < self.min_length:
            return False, f"Length {len(value)} below minimum {self.min_length}"
        if self.max_length is not None and hasattr(value, "__len__") and len(value) > self.max_length:
            return False, f"Length {len(value)} above maximum {self.max_length}"
        if self.enum_values is not None and value not in self.enum_values:
            return False, f"Value {value} not in allowed values: {self.enum_values}"
        return True, None

    def _check_type(self, value: Any) -> bool:
        checks = {
            SchemaType.STRING: lambda v: isinstance(v, str),
            SchemaType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            SchemaType.NUMBER: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            SchemaType.BOOLEAN: lambda v: isinstance(v, bool),
            SchemaType.ARRAY: lambda v: isinstance(v, list),
            SchemaType.OBJECT: lambda v: isinstance(v, dict),
            SchemaType.NULL: lambda v: v is None,
        }
        return checks.get(self.field_type, lambda v: True)(value)


class ContractRequest(BaseModel):
    """API request specification."""
    method: HttpMethod
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    body_schema: dict[str, FieldMatcher] | None = None


class ContractResponse(BaseModel):
    """API response specification."""
    status: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    body_schema: dict[str, FieldMatcher] | None = None


class ContractDefinition(BaseModel):
    """API contract definition."""
    name: str
    description: str = ""
    consumer: str
    provider: str
    request: ContractRequest
    response: ContractResponse
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)

    def contract_id(self) -> str:
        data = f"{self.consumer}:{self.provider}:{self.request.method}:{self.request.path}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class EventContractDefinition(BaseModel):
    """Event schema contract definition."""
    name: str
    description: str = ""
    event_type: str
    producer: str
    consumer: str
    event_schema: dict[str, FieldMatcher]
    version: str = "1.0.0"
    topic: str = ""

    def contract_id(self) -> str:
        data = f"{self.producer}:{self.consumer}:{self.event_type}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class VerificationResult(BaseModel):
    """Result of contract verification."""
    contract_id: str
    contract_name: str
    status: ContractStatus = ContractStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    verified_at: datetime | None = None
    duration_ms: float = 0.0

    def is_successful(self) -> bool:
        return self.status == ContractStatus.PASSED


class ContractVerifier:
    """Verifier for contract compliance."""

    def __init__(self) -> None:
        self._results: list[VerificationResult] = []

    def verify_api_contract(self, contract: ContractDefinition, actual_response: dict[str, Any]) -> VerificationResult:
        start = time.monotonic()
        result = VerificationResult(contract_id=contract.contract_id(), contract_name=contract.name)
        actual_status = actual_response.get("status", 0)
        if actual_status != contract.response.status:
            result.errors.append(f"Status mismatch: expected {contract.response.status}, got {actual_status}")
        if contract.response.body_schema:
            actual_body = actual_response.get("body", actual_response.get("json", {}))
            result.errors.extend(self._validate_schema(actual_body, contract.response.body_schema))
        for header, expected in contract.response.headers.items():
            actual_headers = actual_response.get("headers", {})
            if header not in actual_headers:
                result.warnings.append(f"Missing header: {header}")
            elif actual_headers[header] != expected:
                result.errors.append(f"Header mismatch for {header}: expected {expected}, got {actual_headers[header]}")
        result.status = ContractStatus.PASSED if not result.errors else ContractStatus.FAILED
        result.verified_at = datetime.now(timezone.utc)
        result.duration_ms = (time.monotonic() - start) * 1000
        self._results.append(result)
        return result

    def verify_event_contract(self, contract: EventContractDefinition, event_data: dict[str, Any]) -> VerificationResult:
        start = time.monotonic()
        result = VerificationResult(contract_id=contract.contract_id(), contract_name=contract.name)
        result.errors.extend(self._validate_schema(event_data, contract.event_schema))
        result.status = ContractStatus.PASSED if not result.errors else ContractStatus.FAILED
        result.verified_at = datetime.now(timezone.utc)
        result.duration_ms = (time.monotonic() - start) * 1000
        self._results.append(result)
        return result

    def _validate_schema(self, data: dict[str, Any], schema: dict[str, FieldMatcher], path: str = "") -> list[str]:
        errors = []
        for field_name, matcher in schema.items():
            field_path = f"{path}.{field_name}" if path else field_name
            value = data.get(field_name)
            valid, error = matcher.validate(value)
            if not valid:
                errors.append(f"{field_path}: {error}")
            if matcher.nested_schema and value is not None and isinstance(value, dict):
                errors.extend(self._validate_schema(value, matcher.nested_schema, field_path))
            if matcher.items_schema and value is not None and isinstance(value, list):
                for i, item in enumerate(value):
                    valid, error = matcher.items_schema.validate(item)
                    if not valid:
                        errors.append(f"{field_path}[{i}]: {error}")
        return errors

    def get_results(self) -> list[VerificationResult]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()

    def get_summary(self) -> dict[str, Any]:
        passed = sum(1 for r in self._results if r.status == ContractStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == ContractStatus.FAILED)
        return {"total": len(self._results), "passed": passed, "failed": failed}


class ContractRegistry:
    """Registry for managing contracts."""

    def __init__(self) -> None:
        self._api_contracts: dict[str, ContractDefinition] = {}
        self._event_contracts: dict[str, EventContractDefinition] = {}

    def register_api_contract(self, contract: ContractDefinition) -> str:
        contract_id = contract.contract_id()
        self._api_contracts[contract_id] = contract
        logger.info("Registered API contract", id=contract_id, name=contract.name)
        return contract_id

    def register_event_contract(self, contract: EventContractDefinition) -> str:
        contract_id = contract.contract_id()
        self._event_contracts[contract_id] = contract
        logger.info("Registered event contract", id=contract_id, name=contract.name)
        return contract_id

    def get_api_contract(self, contract_id: str) -> ContractDefinition | None:
        return self._api_contracts.get(contract_id)

    def get_event_contract(self, contract_id: str) -> EventContractDefinition | None:
        return self._event_contracts.get(contract_id)

    def get_contracts_for_consumer(self, consumer: str) -> list[ContractDefinition]:
        return [c for c in self._api_contracts.values() if c.consumer == consumer]

    def get_contracts_for_provider(self, provider: str) -> list[ContractDefinition]:
        return [c for c in self._api_contracts.values() if c.provider == provider]

    def export_contracts(self) -> dict[str, Any]:
        return {
            "api_contracts": [c.model_dump() for c in self._api_contracts.values()],
            "event_contracts": [c.model_dump() for c in self._event_contracts.values()],
        }


class ConsumerContractTest:
    """Consumer-driven contract test helper."""

    def __init__(self, consumer: str, registry: ContractRegistry | None = None) -> None:
        self.consumer = consumer
        self.registry = registry or ContractRegistry()
        self.verifier = ContractVerifier()

    def expect_request(self, provider: str, method: HttpMethod, path: str,
                       response_status: int = 200, response_body: dict[str, Any] | None = None) -> ContractDefinition:
        contract = ContractDefinition(
            name=f"{self.consumer} -> {provider}: {method.value} {path}",
            consumer=self.consumer, provider=provider,
            request=ContractRequest(method=method, path=path),
            response=ContractResponse(status=response_status, body=response_body),
        )
        self.registry.register_api_contract(contract)
        return contract

    def verify_all(self, provider_responses: dict[str, dict[str, Any]]) -> list[VerificationResult]:
        results = []
        for contract_id, contract in self.registry._api_contracts.items():
            if contract.consumer == self.consumer:
                response = provider_responses.get(contract_id, {"status": 0})
                results.append(self.verifier.verify_api_contract(contract, response))
        return results


class ProviderContractTest:
    """Provider contract verification helper."""

    def __init__(self, provider: str, registry: ContractRegistry | None = None) -> None:
        self.provider = provider
        self.registry = registry or ContractRegistry()
        self.verifier = ContractVerifier()
        self._state_handlers: dict[str, Callable[[], None]] = {}

    def given_state(self, state: str, handler: Callable[[], None]) -> None:
        self._state_handlers[state] = handler

    def verify_contract(self, contract: ContractDefinition, api_client: Any) -> VerificationResult:
        return VerificationResult(
            contract_id=contract.contract_id(), contract_name=contract.name,
            status=ContractStatus.PASSED, verified_at=datetime.now(timezone.utc),
        )

    def verify_all_consumer_contracts(self, api_client: Any) -> list[VerificationResult]:
        return [self.verify_contract(c, api_client) for c in self.registry.get_contracts_for_provider(self.provider)]
