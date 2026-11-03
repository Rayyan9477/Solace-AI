"""MVP-199: UnifiedLLMClient.initialize() never raises for missing credentials."""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from services.shared.infrastructure.llm_client import LLMClientSettings, UnifiedLLMClient


class _StrictSdk:
    """Mimics portkey-ai 1.9.x: AsyncPortkey(api_key='') with no PORTKEY_API_KEY raises ValueError."""

    def __init__(self, *, api_key, base_url=None, config=None, **_):
        if not api_key:
            raise ValueError("Portkey API Key Not Found")
        self.api_key = api_key
        self.config = config


@pytest.fixture
def fake_portkey(monkeypatch):
    module = types.ModuleType("portkey_ai")
    module.AsyncPortkey = _StrictSdk
    monkeypatch.setitem(sys.modules, "portkey_ai", module)
    for name in ("PORTKEY_API_KEY", "LLM_PROVIDER", "BEDROCK_MODEL_ID", "BEDROCK_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    return module


def _settings(**kwargs):
    return LLMClientSettings(_env_file=None, **kwargs)


@pytest.mark.parametrize("env", ["development", "test"])
async def test_mvp_199_missing_key_does_not_raise_outside_strict(fake_portkey, monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("")))
    await client.initialize()
    assert client.is_initialized is True
    assert client.is_available is False


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_199_missing_key_in_strict_disables_without_raising(fake_portkey, monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")  # D-04: the only strict route (WS-09.18); no setting reads it before WS-09.18
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("")))
    await client.initialize()
    assert client.is_available is False


async def test_mvp_199_configured_key_builds_client_via_factory(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return object()

    client = UnifiedLLMClient(
        _settings(portkey_api_key=SecretStr("pk-test"), anthropic_api_key=SecretStr("sk-test")),
        client_factory=factory,
    )
    await client.initialize()
    assert client.is_available is True
    assert calls[0]["api_key"] == "pk-test"


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_199_missing_sdk_is_a_startup_error_in_strict(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setitem(sys.modules, "portkey_ai", None)  # makes the lazy import raise ImportError
    from services.shared.infrastructure.llm_client import LLMConfigurationError

    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("pk-test")))
    with pytest.raises(LLMConfigurationError):
        await client.initialize()


class _ClosableClient:
    def __init__(self, fail):
        self.closed = False
        self.fail = fail

    async def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("connection pool already closed")


@pytest.mark.parametrize("fail", [False, True])
async def test_mvp_332_shutdown_closes_the_underlying_client(monkeypatch, fail):
    monkeypatch.setenv("ENVIRONMENT", "test")
    inner = _ClosableClient(fail)
    client = UnifiedLLMClient(_settings(portkey_api_key=SecretStr("pk-test")), client_factory=lambda **_: inner)
    await client.initialize()
    await client.shutdown()  # never raises, even when close() does
    assert inner.closed is True
    assert client.is_available is False
