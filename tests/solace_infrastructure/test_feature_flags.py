"""Unit tests for solace_infrastructure.feature_flags.

Includes H-39 guard: ``enforce_database_ssl`` must be ON by default for the
``production`` and ``staging`` environments. PHI transiting unencrypted from
a FastAPI service to Postgres on an internal network is a HIPAA violation,
so this flag is a required default, not an opt-in.
"""
from __future__ import annotations

import os

from solace_infrastructure.feature_flags import (
    FeatureFlags,
    RolloutStrategy,
)


class TestEnforceDatabaseSslFlag:
    """H-39: SSL enforcement must be default-enabled for production + staging."""

    def setup_method(self) -> None:
        """Preserve ENVIRONMENT so tests don't leak into each other."""
        self._prev_env = os.environ.get("ENVIRONMENT")

    def teardown_method(self) -> None:
        if self._prev_env is None:
            os.environ.pop("ENVIRONMENT", None)
        else:
            os.environ["ENVIRONMENT"] = self._prev_env

    def test_flag_registered(self) -> None:
        flag = FeatureFlags.get_flag("enforce_database_ssl")
        assert flag is not None

    def test_enabled_true_by_default(self) -> None:
        """The master kill-switch must be ON so the env-based strategy evaluates."""
        flag = FeatureFlags.get_flag("enforce_database_ssl")
        assert flag is not None
        assert flag.enabled is True, (
            "H-39: enforce_database_ssl.enabled must default to True. "
            "With enabled=False the environment strategy is short-circuited "
            "and SSL enforcement is silently skipped in prod."
        )

    def test_strategy_is_environment_based(self) -> None:
        flag = FeatureFlags.get_flag("enforce_database_ssl")
        assert flag is not None
        assert flag.strategy == RolloutStrategy.ENVIRONMENT
        assert "production" in flag.allowed_environments
        assert "staging" in flag.allowed_environments

    def test_enabled_in_production(self) -> None:
        os.environ["ENVIRONMENT"] = "production"
        assert FeatureFlags.is_enabled("enforce_database_ssl") is True

    def test_enabled_in_staging(self) -> None:
        os.environ["ENVIRONMENT"] = "staging"
        assert FeatureFlags.is_enabled("enforce_database_ssl") is True

    def test_disabled_in_development(self) -> None:
        """Dev environments can still connect without SSL for local Postgres."""
        os.environ["ENVIRONMENT"] = "development"
        assert FeatureFlags.is_enabled("enforce_database_ssl") is False

    def test_disabled_in_testing(self) -> None:
        """Test runs use in-memory / non-SSL postgres; must not be enforced."""
        os.environ["ENVIRONMENT"] = "testing"
        assert FeatureFlags.is_enabled("enforce_database_ssl") is False
