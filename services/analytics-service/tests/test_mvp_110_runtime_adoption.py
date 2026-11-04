"""MVP-110 + IF-01 adoption for analytics-service (WS-18.2)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import main as analytics_main  # noqa: E402
from aggregations import AnalyticsAggregator  # noqa: E402
from consumer import AnalyticsConsumer  # noqa: E402


def _failing_consumer(**kwargs):
    raise ConnectionError("broker unreachable")


def _kafka_settings(env: str | None = None):
    settings = analytics_main.SETTINGS.model_copy(deep=True)
    settings.consumer.kafka_enabled = True
    if env:
        settings.service.env = env
    return settings


def test_if01_analytics_spec_flags() -> None:
    spec = analytics_main.SPEC
    assert spec.service_name == "analytics-service"
    assert (spec.uses_postgres, spec.uses_redis, spec.event_plane) == (True, True, "publish_consume")
    assert (spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (False, False, False, False)


def test_if01_analytics_has_no_local_probes_or_instrumentator() -> None:
    for name in ("create_app", "_setup_prometheus", "_setup_opentelemetry"):
        assert not hasattr(analytics_main, name), name
    assert "/api/v1/analytics/health" not in {route.path for route in analytics_main.app.routes}


def test_if01_service_env_accepts_test_and_follows_environment(monkeypatch) -> None:
    monkeypatch.delenv("ANALYTICS_SERVICE_ENV", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert analytics_main.ServiceConfig().env == "test"


def test_mvp_110_consumer_failure_is_visible_in_ready(monkeypatch) -> None:
    monkeypatch.setattr(analytics_main, "SETTINGS", _kafka_settings())
    monkeypatch.setattr(analytics_main, "create_kafka_consumer", _failing_consumer)
    with TestClient(analytics_main.app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["kafka_consumer"]["ok"] is False


async def test_mvp_110_consumer_failure_is_fatal_in_strict(monkeypatch) -> None:
    from solace_infrastructure.runtime import StartupDependencyError

    monkeypatch.setattr(analytics_main, "create_kafka_consumer", _failing_consumer)
    runtime = SimpleNamespace(register_readiness_check=lambda *a, **k: None,
                              register_background_task=lambda *a, **k: None,
                              register_shutdown=lambda *a, **k: None)
    with pytest.raises(StartupDependencyError):
        await analytics_main.start_kafka_ingestion(
            runtime, AnalyticsConsumer(aggregator=AnalyticsAggregator()), _kafka_settings("staging"))


def test_fix1_shutdown_runs_clean_and_clears_both_global_sets(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import api as analytics_api
    from solace_infrastructure.runtime import bootstrap

    log = MagicMock()
    monkeypatch.setattr(bootstrap, "_log", log)
    with TestClient(analytics_main.app):
        assert analytics_main._analytics_aggregator is not None
        assert analytics_api._analytics_aggregator is not None
    failed = [c for c in log.warning.call_args_list if c.args and c.args[0] == "runtime_shutdown_step_failed"]
    assert failed == []
    for module in (analytics_main, analytics_api):
        assert module._analytics_aggregator is None
        assert module._report_service is None
        assert module._analytics_consumer is None
