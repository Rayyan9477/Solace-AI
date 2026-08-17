"""H-39: PostgreSQL SSL/TLS enforcement in production/staging.

PHI transiting a service→DB connection unencrypted is a HIPAA violation. The
enforcement lives in ``PostgresSettings.get_effective_ssl_mode`` /
``get_ssl_context`` (a permissive mode is upgraded to "require" in prod/staging,
producing a real SSLContext) and is wired into the asyncpg pool in
``ConnectionPoolManager.get_pool`` (``ssl=ssl_context``).

These tests validate the security property end to end:
  1. prod/staging upgrade permissive modes and produce a real SSLContext;
  2. development stays permissive so local non-SSL Postgres keeps working
     (this dev/prod contrast is what gives the tests teeth — if enforcement were
     removed, the prod assertions would collapse to the dev behavior and fail);
  3. the SSLContext is actually passed to ``asyncpg.create_pool``.
"""
from __future__ import annotations

import ssl

import pytest

from solace_infrastructure.postgres import PostgresSettings


def _settings(ssl_mode: str = "prefer") -> PostgresSettings:
    return PostgresSettings(ssl_mode=ssl_mode, password="test_password")


class TestSslModeEnforcement:
    @pytest.mark.parametrize("env", ["production", "staging"])
    @pytest.mark.parametrize("permissive", ["disable", "allow", "prefer"])
    def test_permissive_mode_upgraded_to_require(
        self, monkeypatch: pytest.MonkeyPatch, env: str, permissive: str
    ) -> None:
        """In prod/staging, any permissive ssl_mode is forced up to 'require'."""
        monkeypatch.setenv("ENVIRONMENT", env)
        assert _settings(permissive).get_effective_ssl_mode() == "require"

    @pytest.mark.parametrize("permissive", ["disable", "allow", "prefer"])
    def test_development_keeps_permissive_mode(
        self, monkeypatch: pytest.MonkeyPatch, permissive: str
    ) -> None:
        """Development must NOT be upgraded — local non-SSL Postgres keeps working."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert _settings(permissive).get_effective_ssl_mode() == permissive

    @pytest.mark.parametrize("strict", ["require", "verify-ca", "verify-full"])
    def test_strict_mode_preserved(
        self, monkeypatch: pytest.MonkeyPatch, strict: str
    ) -> None:
        """An explicitly strict mode is preserved (not downgraded) everywhere."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        assert _settings(strict).get_effective_ssl_mode() == strict


class TestSslContextEnforcement:
    def test_production_yields_real_ssl_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prod + permissive mode → a real SSLContext (encryption enforced)."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        ctx = _settings("prefer").get_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)

    def test_development_permissive_yields_no_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dev + permissive mode → None (asyncpg negotiates, local dev works)."""
        monkeypatch.setenv("ENVIRONMENT", "development")
        assert _settings("prefer").get_ssl_context() is None

    def test_production_cannot_disable_ssl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even an explicit 'disable' cannot turn off encryption in production."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        assert isinstance(_settings("disable").get_ssl_context(), ssl.SSLContext)


class TestSslContextReachesPool:
    """The exact SSLContext must be handed to asyncpg.create_pool (H-39 deliverable)."""

    def test_ssl_context_passed_to_create_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import asyncpg

        from solace_infrastructure.database.connection_manager import (
            ConnectionPoolManager,
        )

        monkeypatch.setenv("ENVIRONMENT", "production")
        captured: dict = {}

        async def _fake_create_pool(*args, **kwargs):
            captured.update(kwargs)
            return object()  # stand-in pool

        monkeypatch.setattr(asyncpg, "create_pool", _fake_create_pool)

        name = "h39_ssl_test"

        async def _run() -> None:
            try:
                await ConnectionPoolManager.register_pool(name, _settings("prefer"))
                await ConnectionPoolManager.get_pool(name)
            finally:
                # Clean up class-level state so we don't leak into other tests.
                ConnectionPoolManager._pools.pop(name, None)
                ConnectionPoolManager._pool_configs.pop(name, None)

        asyncio.run(_run())

        assert "ssl" in captured
        assert isinstance(captured["ssl"], ssl.SSLContext), (
            "H-39: a real SSLContext must be passed to asyncpg.create_pool in production"
        )
