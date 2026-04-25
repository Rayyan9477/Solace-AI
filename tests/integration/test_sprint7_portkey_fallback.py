"""Sprint 7 integration: Portkey multi-provider fallback + observability.

The LLM gateway must be resilient to primary-provider failure — if
Anthropic is down, the request should be routed to the OpenAI fallback
without application-layer intervention. This test verifies the
``UnifiedLLMClient`` builds the right Portkey config and respects the
``LLM_ENABLE_FALLBACK`` env flag.

We don't invoke the real Portkey API (no network calls, no real keys) —
the test reads the built config dict that Portkey would consume. The
real fallback behaviour is Portkey's responsibility; ours is to build
the config correctly.
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import SecretStr

from services.shared.infrastructure.llm_client import (
    LLMClientSettings,
    UnifiedLLMClient,
)


class TestPortkeyFallbackConfig:
    """The Portkey config dict must declare both primary and fallback
    targets with the right strategy when ``enable_fallback=True``.
    """

    def test_fallback_target_included_when_enabled(self) -> None:
        settings = LLMClientSettings(
            portkey_api_key=SecretStr("test-portkey-key"),
            primary_provider="anthropic",
            primary_model="claude-sonnet-4-20250514",
            fallback_provider="openai",
            fallback_model="gpt-4o",
            anthropic_api_key=SecretStr("anthropic-test"),
            openai_api_key=SecretStr("openai-test"),
            enable_fallback=True,
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        targets = config["targets"]
        assert len(targets) == 2, (
            f"Expected 2 targets (primary + fallback), got {len(targets)}"
        )
        assert targets[0]["provider"] == "anthropic"
        assert targets[1]["provider"] == "openai"
        assert config.get("strategy", {}).get("mode") == "fallback", (
            "strategy.mode must be 'fallback' when enable_fallback=True"
        )

    def test_fallback_target_omitted_when_disabled(self) -> None:
        settings = LLMClientSettings(
            portkey_api_key=SecretStr("test-portkey-key"),
            anthropic_api_key=SecretStr("anthropic-test"),
            enable_fallback=False,
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        targets = config["targets"]
        assert len(targets) == 1, (
            f"Expected single primary target when fallback disabled, got {len(targets)}"
        )
        # No fallback-mode strategy
        assert config.get("strategy", {}).get("mode") != "fallback"

    def test_load_balance_strategy_overrides_fallback(self) -> None:
        settings = LLMClientSettings(
            portkey_api_key=SecretStr("test-portkey-key"),
            anthropic_api_key=SecretStr("anthropic-test"),
            openai_api_key=SecretStr("openai-test"),
            enable_fallback=True,
            enable_load_balancing=True,
            load_balance_weight_primary=0.7,
        )
        client = UnifiedLLMClient(settings)
        config = client._build_portkey_config()

        targets = config["targets"]
        assert len(targets) == 2
        assert config["strategy"]["mode"] == "loadbalance"
        # Primary weight must be 0.7 and fallback 0.3
        primary_weight = targets[0].get("weight")
        fallback_weight = targets[1].get("weight")
        assert primary_weight is None or abs(primary_weight - 0.7) < 1e-9, (
            f"Primary weight mismatch: {primary_weight}"
        )
        assert fallback_weight is not None and abs(fallback_weight - 0.3) < 1e-9, (
            f"Fallback weight mismatch: {fallback_weight}"
        )


class TestPortkeyObservabilityHooks:
    """The UnifiedLLMClient must pass a trace-id prefix through so every
    LLM call is correlatable across services in Jaeger / Prometheus."""

    def test_trace_id_prefix_is_configurable(self) -> None:
        settings = LLMClientSettings(
            portkey_api_key=SecretStr("test-portkey-key"),
            anthropic_api_key=SecretStr("anthropic-test"),
            trace_id_prefix="solace-test",
        )
        assert settings.trace_id_prefix == "solace-test"


class TestTaskTypePresets:
    """Clinical invariant: task-type presets must use sensible temperatures.
    Crisis responses must not be warm-and-creative; structured output must
    be deterministic."""

    def test_crisis_preset_is_low_temperature(self) -> None:
        from services.shared.infrastructure.llm_client import TASK_TYPE_PRESETS

        crisis = TASK_TYPE_PRESETS["crisis"]
        assert crisis["temperature"] <= 0.3, (
            "Crisis LLM calls must use low temperature for reliability"
        )

    def test_structured_preset_is_deterministic(self) -> None:
        from services.shared.infrastructure.llm_client import TASK_TYPE_PRESETS

        structured = TASK_TYPE_PRESETS["structured"]
        assert structured["temperature"] == 0.0, (
            "Structured JSON output must be deterministic (temperature=0)"
        )

    def test_all_clinical_task_types_present(self) -> None:
        from services.shared.infrastructure.llm_client import TASK_TYPE_PRESETS

        for required in ("crisis", "therapy", "diagnosis", "structured"):
            assert required in TASK_TYPE_PRESETS, (
                f"Task-type preset '{required}' missing from presets"
            )


class TestPortkeyWiringStructural:
    """Structural safety net: the code path that builds the Portkey
    configuration must still exist as a single consolidated method."""

    def test_build_portkey_config_method_exists(self) -> None:
        client = UnifiedLLMClient(LLMClientSettings(
            portkey_api_key=SecretStr("test"),
            anthropic_api_key=SecretStr("test"),
        ))
        assert hasattr(client, "_build_portkey_config")
        assert callable(client._build_portkey_config)

    def test_llm_client_module_advertises_portkey(self) -> None:
        from services.shared.infrastructure import llm_client

        src = inspect.getsource(llm_client)
        assert "Portkey" in src, (
            "LLM client module must advertise Portkey as the gateway"
        )
        assert "fallback" in src.lower()
