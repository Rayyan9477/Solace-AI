"""
Phase C P1-9 regression for the analytics service.

The runtime gate is ``ServiceConfig`` (src/main.py) with ``env_prefix="ANALYTICS_SERVICE_"``
and field ``env`` -> binds ``ANALYTICS_SERVICE_ENV``. The production behaviour (docs/redoc
disabled, CORS locked, PHI log sanitizer required at main.py:216) reads
``settings.service.env``. The base compose set ``ANALYTICS_SERVICE_ENV`` as a
non-overridable literal ``development``, which (because ``environment:`` wins over
``env_file:``) pinned the service to ``development`` even under ``.env.prod``.
"""
from __future__ import annotations

import pytest

# analytics tests add <service>/src to sys.path via conftest.py.
from main import ServiceConfig


def test_env_binds_from_analytics_service_env(monkeypatch):
    """The var the compose file must set (ANALYTICS_SERVICE_ENV) actually binds."""
    monkeypatch.setenv("ANALYTICS_SERVICE_ENV", "production")
    cfg = ServiceConfig()
    assert cfg.env == "production"


def test_default_is_development_when_unset(monkeypatch):
    monkeypatch.delenv("ANALYTICS_SERVICE_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    cfg = ServiceConfig()
    assert cfg.env == "development"
