"""WS-04.F2 (W0-close review SEC-4): the insecure SessionConfig default secret is refused in staging as well as
production. Strict = the service's own USER_SERVICE_ENV OR the bare ENVIRONMENT (D-14); the check used to compare
the bare ENVIRONMENT with 'production' alone, so staging accepted the default as its signing secret."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.auth import SessionConfig

INSECURE_DEFAULT = "your-secret-key-change-in-production-32-chars"


def _env(monkeypatch, *, bare: str, own: str | None) -> None:
    monkeypatch.setenv("ENVIRONMENT", bare)
    if own is None:
        monkeypatch.delenv("USER_SERVICE_ENV", raising=False)
    else:
        monkeypatch.setenv("USER_SERVICE_ENV", own)
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)


@pytest.mark.parametrize(("bare", "own"), [
    ("staging", None), ("production", None), ("Staging ", None),
    ("development", "staging"), ("test", "production"),
])
def test_strict_refuses_the_insecure_default_secret(monkeypatch, bare, own):
    _env(monkeypatch, bare=bare, own=own)
    with pytest.raises(ValidationError, match="staging and production"):
        SessionConfig(_env_file=None)


@pytest.mark.parametrize("bare", ["development", "test"])
def test_dev_and_test_keep_the_default(monkeypatch, bare):
    _env(monkeypatch, bare=bare, own=bare)
    assert SessionConfig(_env_file=None).secret_key == INSECURE_DEFAULT


@pytest.mark.parametrize("bare", ["staging", "production"])
def test_strict_accepts_a_real_secret(monkeypatch, bare):
    _env(monkeypatch, bare=bare, own=bare)
    assert SessionConfig(_env_file=None, secret_key="s" * 48).secret_key == "s" * 48
