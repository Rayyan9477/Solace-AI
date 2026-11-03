"""P2-12: behind Caddy (which appends exactly one X-Forwarded-For hop) a
``trusted_proxy_hops`` of 0 collapses every real client into one rate-limit bucket,
so a single abusive IP can exhaust the shared limit for everyone. In production the
effective default must be 1 (Caddy's single hop) while dev/test stay at 0 (no proxy,
XFF is spoofable and must be ignored). An explicit ORCHESTRATOR_TRUSTED_PROXY_HOPS
always wins so non-Caddy deployments can override.
"""
from __future__ import annotations

import pytest


def _hops(monkeypatch, **env):
    for k in ("ORCHESTRATOR_TRUSTED_PROXY_HOPS", "ORCHESTRATOR_ENVIRONMENT", "ENVIRONMENT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from services.orchestrator_service.src.main import _trusted_proxy_hops

    return _trusted_proxy_hops()


def test_production_defaults_to_one_hop(monkeypatch):
    assert _hops(monkeypatch, ORCHESTRATOR_ENVIRONMENT="production") == 1


def test_bare_production_env_defaults_to_one_hop(monkeypatch):
    assert _hops(monkeypatch, ENVIRONMENT="production") == 1


def test_staging_defaults_to_one_hop(monkeypatch):
    assert _hops(monkeypatch, ORCHESTRATOR_ENVIRONMENT="staging") == 1


def test_development_defaults_to_zero(monkeypatch):
    assert _hops(monkeypatch, ORCHESTRATOR_ENVIRONMENT="development") == 0


def test_test_env_defaults_to_zero(monkeypatch):
    assert _hops(monkeypatch, ENVIRONMENT="test") == 0


def test_explicit_value_overrides_production_default(monkeypatch):
    # A deployment not behind Caddy can pin 0 even in production.
    assert _hops(monkeypatch, ORCHESTRATOR_ENVIRONMENT="production",
                 ORCHESTRATOR_TRUSTED_PROXY_HOPS="0") == 0


def test_explicit_multi_hop_is_honored(monkeypatch):
    assert _hops(monkeypatch, ORCHESTRATOR_ENVIRONMENT="production",
                 ORCHESTRATOR_TRUSTED_PROXY_HOPS="2") == 2
