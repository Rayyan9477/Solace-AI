"""Unit tests for Solace-AI Testing Library - Contracts module."""

from __future__ import annotations

import pytest

from solace_testing.contracts import (
    ConsumerContractTest,
    ContractDefinition,
    ContractRegistry,
    ContractRequest,
    ContractResponse,
    ContractStatus,
    ContractVerifier,
    EventContractDefinition,
    FieldMatcher,
    HttpMethod,
    ProviderContractTest,
    SchemaType,
    VerificationResult,
)


class TestFieldMatcher:
    """Tests for FieldMatcher."""

    def test_string_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING)
        valid, error = matcher.validate("test")
        assert valid is True
        assert error is None

    def test_integer_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.INTEGER)
        valid, _ = matcher.validate(42)
        assert valid is True
        valid, error = matcher.validate("42")
        assert valid is False
        assert "Expected" in error and "got str" in error

    def test_required_null_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING, required=True)
        valid, error = matcher.validate(None)
        assert valid is False
        assert "Required" in error

    def test_optional_null_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING, required=False)
        valid, _ = matcher.validate(None)
        assert valid is True

    def test_pattern_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING, pattern=r"^\d{3}-\d{4}$")
        valid, _ = matcher.validate("123-4567")
        assert valid is True
        valid, error = matcher.validate("invalid")
        assert valid is False
        assert "pattern" in error

    def test_min_max_value_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.NUMBER, min_value=0, max_value=100)
        valid, _ = matcher.validate(50)
        assert valid is True
        valid, error = matcher.validate(-1)
        assert valid is False
        assert "below minimum" in error
        valid, error = matcher.validate(101)
        assert valid is False
        assert "above maximum" in error

    def test_min_max_length_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING, min_length=2, max_length=5)
        valid, _ = matcher.validate("abc")
        assert valid is True
        valid, error = matcher.validate("a")
        assert valid is False
        assert "below minimum" in error
        valid, error = matcher.validate("abcdef")
        assert valid is False
        assert "above maximum" in error

    def test_enum_validation(self) -> None:
        matcher = FieldMatcher(field_type=SchemaType.STRING, enum_values=["a", "b", "c"])
        valid, _ = matcher.validate("a")
        assert valid is True
        valid, error = matcher.validate("d")
        assert valid is False
        assert "not in allowed values" in error


class TestContractRequest:
    """Tests for ContractRequest."""

    def test_creation(self) -> None:
        request = ContractRequest(
            method=HttpMethod.GET,
            path="/api/users",
            headers={"Accept": "application/json"},
        )
        assert request.method == HttpMethod.GET
        assert request.path == "/api/users"
        assert "Accept" in request.headers


class TestContractResponse:
    """Tests for ContractResponse."""

    def test_creation(self) -> None:
        response = ContractResponse(
            status=200,
            body={"id": "123", "name": "Test"},
        )
        assert response.status == 200
        assert response.body["id"] == "123"


class TestContractDefinition:
    """Tests for ContractDefinition."""

    def test_contract_id_generation(self) -> None:
        contract = ContractDefinition(
            name="Get User",
            consumer="frontend",
            provider="user-service",
            request=ContractRequest(method=HttpMethod.GET, path="/users/1"),
            response=ContractResponse(status=200),
        )
        contract_id = contract.contract_id()
        assert len(contract_id) == 16
        same_id = contract.contract_id()
        assert contract_id == same_id

    def test_different_contracts_different_ids(self) -> None:
        c1 = ContractDefinition(
            name="Contract 1",
            consumer="c1",
            provider="p1",
            request=ContractRequest(method=HttpMethod.GET, path="/a"),
            response=ContractResponse(status=200),
        )
        c2 = ContractDefinition(
            name="Contract 2",
            consumer="c2",
            provider="p2",
            request=ContractRequest(method=HttpMethod.POST, path="/b"),
            response=ContractResponse(status=201),
        )
        assert c1.contract_id() != c2.contract_id()


class TestEventContractDefinition:
    """Tests for EventContractDefinition."""

    def test_creation(self) -> None:
        contract = EventContractDefinition(
            name="User Created Event",
            event_type="UserCreated",
            producer="user-service",
            consumer="notification-service",
            event_schema={
                "user_id": FieldMatcher(field_type=SchemaType.STRING),
                "email": FieldMatcher(field_type=SchemaType.STRING),
            },
            topic="user.events",
        )
        assert contract.event_type == "UserCreated"
        assert "user_id" in contract.event_schema

    def test_contract_id(self) -> None:
        contract = EventContractDefinition(
            name="Test Event",
            event_type="TestEvent",
            producer="prod",
            consumer="cons",
            event_schema={},
        )
        contract_id = contract.contract_id()
        assert len(contract_id) == 16


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_successful_result(self) -> None:
        result = VerificationResult(
            contract_id="abc123",
            contract_name="Test Contract",
            status=ContractStatus.PASSED,
        )
        assert result.is_successful() is True

    def test_failed_result(self) -> None:
        result = VerificationResult(
            contract_id="abc123",
            contract_name="Test Contract",
            status=ContractStatus.FAILED,
            errors=["Field mismatch"],
        )
        assert result.is_successful() is False
        assert len(result.errors) == 1


class TestContractVerifier:
    """Tests for ContractVerifier."""

    def test_verify_api_contract_success(self) -> None:
        verifier = ContractVerifier()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(status=200),
        )
        actual = {"status": 200, "json": {}}
        result = verifier.verify_api_contract(contract, actual)
        assert result.status == ContractStatus.PASSED

    def test_verify_api_contract_status_mismatch(self) -> None:
        verifier = ContractVerifier()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(status=200),
        )
        actual = {"status": 404, "json": {}}
        result = verifier.verify_api_contract(contract, actual)
        assert result.status == ContractStatus.FAILED
        assert any("Status mismatch" in e for e in result.errors)

    def test_verify_api_contract_schema_validation(self) -> None:
        verifier = ContractVerifier()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(
                status=200,
                body_schema={
                    "name": FieldMatcher(field_type=SchemaType.STRING),
                    "age": FieldMatcher(field_type=SchemaType.INTEGER),
                },
            ),
        )
        actual = {"status": 200, "json": {"name": "Test", "age": "invalid"}}
        result = verifier.verify_api_contract(contract, actual)
        assert result.status == ContractStatus.FAILED

    def test_verify_event_contract(self) -> None:
        verifier = ContractVerifier()
        contract = EventContractDefinition(
            name="Test Event",
            event_type="TestEvent",
            producer="p",
            consumer="c",
            event_schema={
                "id": FieldMatcher(field_type=SchemaType.STRING),
                "count": FieldMatcher(field_type=SchemaType.INTEGER),
            },
        )
        event_data = {"id": "123", "count": 42}
        result = verifier.verify_event_contract(contract, event_data)
        assert result.status == ContractStatus.PASSED

    def test_get_results(self) -> None:
        verifier = ContractVerifier()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(status=200),
        )
        verifier.verify_api_contract(contract, {"status": 200})
        results = verifier.get_results()
        assert len(results) == 1

    def test_get_summary(self) -> None:
        verifier = ContractVerifier()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(status=200),
        )
        verifier.verify_api_contract(contract, {"status": 200})
        verifier.verify_api_contract(contract, {"status": 500})
        summary = verifier.get_summary()
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1


class TestContractRegistry:
    """Tests for ContractRegistry."""

    def test_register_api_contract(self) -> None:
        registry = ContractRegistry()
        contract = ContractDefinition(
            name="Test",
            consumer="c",
            provider="p",
            request=ContractRequest(method=HttpMethod.GET, path="/test"),
            response=ContractResponse(status=200),
        )
        contract_id = registry.register_api_contract(contract)
        assert contract_id is not None
        retrieved = registry.get_api_contract(contract_id)
        assert retrieved.name == "Test"

    def test_register_event_contract(self) -> None:
        registry = ContractRegistry()
        contract = EventContractDefinition(
            name="Test Event",
            event_type="TestEvent",
            producer="p",
            consumer="c",
            event_schema={},
        )
        contract_id = registry.register_event_contract(contract)
        retrieved = registry.get_event_contract(contract_id)
        assert retrieved.event_type == "TestEvent"

    def test_get_contracts_for_consumer(self) -> None:
        registry = ContractRegistry()
        for i in range(3):
            registry.register_api_contract(
                ContractDefinition(
                    name=f"Contract {i}",
                    consumer="frontend",
                    provider="backend",
                    request=ContractRequest(method=HttpMethod.GET, path=f"/api/{i}"),
                    response=ContractResponse(status=200),
                )
            )
        contracts = registry.get_contracts_for_consumer("frontend")
        assert len(contracts) == 3

    def test_get_contracts_for_provider(self) -> None:
        registry = ContractRegistry()
        registry.register_api_contract(
            ContractDefinition(
                name="Contract",
                consumer="c1",
                provider="user-service",
                request=ContractRequest(method=HttpMethod.GET, path="/api"),
                response=ContractResponse(status=200),
            )
        )
        contracts = registry.get_contracts_for_provider("user-service")
        assert len(contracts) == 1

    def test_export_contracts(self) -> None:
        registry = ContractRegistry()
        registry.register_api_contract(
            ContractDefinition(
                name="API Contract",
                consumer="c",
                provider="p",
                request=ContractRequest(method=HttpMethod.GET, path="/api"),
                response=ContractResponse(status=200),
            )
        )
        registry.register_event_contract(
            EventContractDefinition(
                name="Event Contract",
                event_type="Test",
                producer="p",
                consumer="c",
                event_schema={},
            )
        )
        exported = registry.export_contracts()
        assert len(exported["api_contracts"]) == 1
        assert len(exported["event_contracts"]) == 1


class TestConsumerContractTest:
    """Tests for ConsumerContractTest."""

    def test_expect_request(self) -> None:
        test = ConsumerContractTest(consumer="frontend")
        contract = test.expect_request(
            provider="user-service",
            method=HttpMethod.GET,
            path="/users/1",
            response_status=200,
            response_body={"id": "1", "name": "Test"},
        )
        assert contract.consumer == "frontend"
        assert contract.provider == "user-service"

    def test_verify_all(self) -> None:
        test = ConsumerContractTest(consumer="frontend")
        contract = test.expect_request(
            provider="user-service",
            method=HttpMethod.GET,
            path="/users",
            response_status=200,
        )
        responses = {contract.contract_id(): {"status": 200}}
        results = test.verify_all(responses)
        assert len(results) == 1
        assert results[0].status == ContractStatus.PASSED


class TestProviderContractTest:
    """Tests for ProviderContractTest."""

    def test_given_state(self) -> None:
        test = ProviderContractTest(provider="user-service")
        called = []

        def setup_state() -> None:
            called.append("setup")

        test.given_state("user exists", setup_state)
        assert "user exists" in test._state_handlers

    def test_verify_contract(self) -> None:
        test = ProviderContractTest(provider="user-service")
        contract = ContractDefinition(
            name="Test",
            consumer="frontend",
            provider="user-service",
            request=ContractRequest(method=HttpMethod.GET, path="/users/1"),
            response=ContractResponse(status=200),
        )
        result = test.verify_contract(contract, None)
        assert result.status == ContractStatus.PASSED


class TestHttpMethod:
    """Tests for HttpMethod enum."""

    def test_all_methods(self) -> None:
        assert HttpMethod.GET == "GET"
        assert HttpMethod.POST == "POST"
        assert HttpMethod.PUT == "PUT"
        assert HttpMethod.PATCH == "PATCH"
        assert HttpMethod.DELETE == "DELETE"


class TestSchemaType:
    """Tests for SchemaType enum."""

    def test_all_types(self) -> None:
        assert SchemaType.STRING == "string"
        assert SchemaType.INTEGER == "integer"
        assert SchemaType.NUMBER == "number"
        assert SchemaType.BOOLEAN == "boolean"
        assert SchemaType.ARRAY == "array"
        assert SchemaType.OBJECT == "object"
        assert SchemaType.NULL == "null"
