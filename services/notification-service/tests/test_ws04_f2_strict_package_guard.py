"""WS-04.F2 (W0-close review SEC-4): consumers.py refuses to import without solace_events in staging as well as
production. Strict = the service's own NOTIFICATION_SERVICE_ENV OR the bare ENVIRONMENT (D-14); the import-time
check used to compare the bare ENVIRONMENT with 'production' alone.

The module is executed as a fresh copy under a probe name, so the real ``src.consumers`` is never reloaded.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from src import consumers


def _load_without_solace_events(monkeypatch, *, bare: str, own: str | None):
    monkeypatch.setitem(sys.modules, "solace_events.schemas", None)   # the import raises ImportError
    monkeypatch.setenv("ENVIRONMENT", bare)
    if own is None:
        monkeypatch.delenv("NOTIFICATION_SERVICE_ENV", raising=False)
    else:
        monkeypatch.setenv("NOTIFICATION_SERVICE_ENV", own)
    name = f"{consumers.__package__}._ws04_f2_consumers_probe"
    spec = importlib.util.spec_from_file_location(name, consumers.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("bare", "own"), [
    ("staging", None), ("production", None), ("development", "staging"), ("test", "production"),
])
def test_strict_refuses_to_import_without_solace_events(monkeypatch, bare, own):
    with pytest.raises(RuntimeError, match="solace_events package required in staging and production"):
        _load_without_solace_events(monkeypatch, bare=bare, own=own)


@pytest.mark.parametrize("bare", ["development", "test"])
def test_dev_and_test_import_without_solace_events(monkeypatch, bare):
    module = _load_without_solace_events(monkeypatch, bare=bare, own=bare)
    assert module._KAFKA_AVAILABLE is False
