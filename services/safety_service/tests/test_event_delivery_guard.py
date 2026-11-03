"""M1 gate P0: safety service must fail loud if the crisis event bridge is
disabled in a strict environment (staging or production).

The Kafka bridge (and its outbox poller) is gated behind ``kafka_enabled``, which
defaults False and is unset in the reference docker-compose. Without this guard a
strict deploy would silently accept crisis events into the durable outbox that
never reach the notification service — crises detected, never delivered.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.safety_service.src.main import _require_event_delivery_in_strict


def _settings(environment: str, kafka_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(environment=environment, kafka_enabled=kafka_enabled)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_without_kafka_fails_loud(env: str) -> None:
    with pytest.raises(RuntimeError, match="required in staging and production"):
        _require_event_delivery_in_strict(_settings(env, kafka_enabled=False))


def test_production_with_kafka_is_allowed() -> None:
    _require_event_delivery_in_strict(_settings("production", kafka_enabled=True))


@pytest.mark.parametrize("env", ["development", "test"])
def test_non_production_without_kafka_is_allowed(env: str) -> None:
    # dev/test run against local non-Kafka stacks — must not fail loud.
    _require_event_delivery_in_strict(_settings(env, kafka_enabled=False))


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_with_mock_producer_fails_loud(env: str) -> None:
    # WS121-CRI-1: a mock producer marks crisis rows PUBLISHED without sending them anywhere.
    settings = SimpleNamespace(environment=env, kafka_enabled=True, kafka_use_mock=True)
    with pytest.raises(RuntimeError, match="mock Kafka producer is not allowed in staging and production"):
        _require_event_delivery_in_strict(settings)


@pytest.mark.parametrize("env", ["development", "test"])
def test_non_strict_mock_producer_is_allowed(env: str) -> None:
    _require_event_delivery_in_strict(
        SimpleNamespace(environment=env, kafka_enabled=True, kafka_use_mock=True))


# WS121F1-CLO-1: the runtime is strict when the service environment OR the bare ENVIRONMENT is
# strict. A blank (or non-strict) SAFETY_ENVIRONMENT under a strict bare ENVIRONMENT must not
# switch the guard off.
_BARE_STRICT = pytest.mark.parametrize("bare_env", ["staging", "production"])
_SERVICE_NOT_STRICT = pytest.mark.parametrize("service_env", ["", "development"])


@_BARE_STRICT
@_SERVICE_NOT_STRICT
def test_strict_bare_environment_refuses_mock_producer(bare_env, service_env, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", bare_env)
    settings = SimpleNamespace(environment=service_env, kafka_enabled=True, kafka_use_mock=True)
    with pytest.raises(RuntimeError, match="mock Kafka producer is not allowed in staging and production"):
        _require_event_delivery_in_strict(settings)


@_BARE_STRICT
@_SERVICE_NOT_STRICT
def test_strict_bare_environment_requires_kafka(bare_env, service_env, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", bare_env)
    with pytest.raises(RuntimeError, match="required in staging and production"):
        _require_event_delivery_in_strict(_settings(service_env, kafka_enabled=False))


@pytest.mark.parametrize("bare_env", ["development", "test"])
@_SERVICE_NOT_STRICT
def test_non_strict_bare_environment_allows_mock(bare_env, service_env, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", bare_env)
    _require_event_delivery_in_strict(
        SimpleNamespace(environment=service_env, kafka_enabled=True, kafka_use_mock=True))
