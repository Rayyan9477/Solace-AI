"""M1 gate round-2 (P0): the notification safety-event consumer must run in production.

The crisis consumer start is gated behind ``settings.kafka.enabled`` (default False)
and its start failure was swallowed (logged, then continued). So a production deploy
that forgets to enable Kafka boots "healthy" and consumes ZERO crisis events — the
consume side of the crisis path silently dead. This is the consumer-side sibling of
REV-37 (safety publish side).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.main import _require_crisis_consumer_in_production


def _settings(env: str, kafka_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        service=SimpleNamespace(env=env),
        kafka=SimpleNamespace(enabled=kafka_enabled),
    )


def test_production_without_kafka_fails_loud() -> None:
    with pytest.raises(RuntimeError, match="required in production"):
        _require_crisis_consumer_in_production(_settings("production", kafka_enabled=False))


def test_production_with_kafka_is_allowed() -> None:
    _require_crisis_consumer_in_production(_settings("production", kafka_enabled=True))


@pytest.mark.parametrize("env", ["development", "staging"])
def test_non_production_without_kafka_is_allowed(env: str) -> None:
    _require_crisis_consumer_in_production(_settings(env, kafka_enabled=False))
