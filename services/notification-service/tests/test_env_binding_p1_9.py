"""
Phase C P1-9 regression for the notification service.

The runtime gate is ``ServiceConfig`` (src/main.py) with ``env_prefix="NOTIFICATION_SERVICE_"``
and field ``env`` -> binds ``NOTIFICATION_SERVICE_ENV``. The production fail-loud guards
(crisis-consumer-required at main.py:164, PHI log sanitizer at main.py:183) read
``settings.service.env``. The base compose set ``NOTIFICATION_SERVICE_ENV`` as a
non-overridable literal ``development``, which (because ``environment:`` wins over
``env_file:``) pinned the service to ``development`` even under ``.env.prod``.
"""
from __future__ import annotations

import pytest

# notification tests add the service root to sys.path via conftest.py (package: src).
from src.main import ServiceConfig


def test_env_binds_from_notification_service_env(monkeypatch):
    """The var the compose file must set (NOTIFICATION_SERVICE_ENV) actually binds."""
    monkeypatch.setenv("NOTIFICATION_SERVICE_ENV", "production")
    cfg = ServiceConfig()
    assert cfg.env == "production"


def test_default_is_development_when_unset(monkeypatch):
    monkeypatch.delenv("NOTIFICATION_SERVICE_ENV", raising=False)
    cfg = ServiceConfig()
    assert cfg.env == "development"
