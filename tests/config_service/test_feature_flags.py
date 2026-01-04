"""Unit tests for Configuration Service - Feature Flags Module."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))

from config_service.src.feature_flags import (
    FlagStatus,
    RolloutStrategy,
    TargetingOperator,
    TargetingRule,
    TargetingGroup,
    FeatureFlag,
    FeatureFlagSettings,
    EvaluationResult,
    FeatureFlagManager,
    get_feature_flag_manager,
    initialize_feature_flags,
)


class TestFlagStatus:
    """Tests for FlagStatus enum."""

    def test_status_values(self) -> None:
        assert FlagStatus.ENABLED.value == "enabled"
        assert FlagStatus.DISABLED.value == "disabled"
        assert FlagStatus.CONDITIONAL.value == "conditional"


class TestRolloutStrategy:
    """Tests for RolloutStrategy enum."""

    def test_strategy_values(self) -> None:
        assert RolloutStrategy.ALL.value == "all"
        assert RolloutStrategy.NONE.value == "none"
        assert RolloutStrategy.PERCENTAGE.value == "percentage"
        assert RolloutStrategy.USER_LIST.value == "user_list"
        assert RolloutStrategy.GRADUAL.value == "gradual"


class TestTargetingOperator:
    """Tests for TargetingOperator enum."""

    def test_operator_values(self) -> None:
        assert TargetingOperator.EQUALS.value == "equals"
        assert TargetingOperator.NOT_EQUALS.value == "not_equals"
        assert TargetingOperator.IN_LIST.value == "in_list"
        assert TargetingOperator.GREATER_THAN.value == "greater_than"


class TestTargetingRule:
    """Tests for TargetingRule model."""

    def test_equals_rule_match(self) -> None:
        rule = TargetingRule(attribute="country", operator=TargetingOperator.EQUALS, value="US")
        assert rule.evaluate({"country": "US"}) is True
        assert rule.evaluate({"country": "UK"}) is False

    def test_not_equals_rule(self) -> None:
        rule = TargetingRule(attribute="tier", operator=TargetingOperator.NOT_EQUALS, value="free")
        assert rule.evaluate({"tier": "premium"}) is True
        assert rule.evaluate({"tier": "free"}) is False

    def test_contains_rule(self) -> None:
        rule = TargetingRule(attribute="email", operator=TargetingOperator.CONTAINS, value="@company.com")
        assert rule.evaluate({"email": "user@company.com"}) is True
        assert rule.evaluate({"email": "user@other.com"}) is False

    def test_in_list_rule(self) -> None:
        rule = TargetingRule(attribute="role", operator=TargetingOperator.IN_LIST, value=["admin", "moderator"])
        assert rule.evaluate({"role": "admin"}) is True
        assert rule.evaluate({"role": "user"}) is False

    def test_greater_than_rule(self) -> None:
        rule = TargetingRule(attribute="age", operator=TargetingOperator.GREATER_THAN, value=18)
        assert rule.evaluate({"age": 21}) is True
        assert rule.evaluate({"age": 16}) is False

    def test_negated_rule(self) -> None:
        rule = TargetingRule(attribute="country", operator=TargetingOperator.EQUALS, value="US", negate=True)
        assert rule.evaluate({"country": "US"}) is False
        assert rule.evaluate({"country": "UK"}) is True

    def test_missing_attribute(self) -> None:
        rule = TargetingRule(attribute="missing", operator=TargetingOperator.EQUALS, value="test")
        assert rule.evaluate({}) is False


class TestTargetingGroup:
    """Tests for TargetingGroup model."""

    def test_empty_rules_match(self) -> None:
        group = TargetingGroup()
        assert group.evaluate({}) is True

    def test_all_rules_must_match(self) -> None:
        group = TargetingGroup(
            rules=[
                TargetingRule(attribute="tier", operator=TargetingOperator.EQUALS, value="premium"),
                TargetingRule(attribute="country", operator=TargetingOperator.EQUALS, value="US"),
            ],
            match_all=True,
        )
        assert group.evaluate({"tier": "premium", "country": "US"}) is True
        assert group.evaluate({"tier": "premium", "country": "UK"}) is False

    def test_any_rule_match(self) -> None:
        group = TargetingGroup(
            rules=[
                TargetingRule(attribute="tier", operator=TargetingOperator.EQUALS, value="premium"),
                TargetingRule(attribute="beta_tester", operator=TargetingOperator.EQUALS, value=True),
            ],
            match_all=False,
        )
        assert group.evaluate({"tier": "premium", "beta_tester": False}) is True
        assert group.evaluate({"tier": "free", "beta_tester": True}) is True
        assert group.evaluate({"tier": "free", "beta_tester": False}) is False

    def test_rollout_percentage(self) -> None:
        group = TargetingGroup(rollout_percentage=50)
        results = [group.evaluate({}, f"user-{i}") for i in range(100)]
        enabled_count = sum(results)
        assert 30 <= enabled_count <= 70


class TestFeatureFlag:
    """Tests for FeatureFlag model."""

    def test_default_flag(self) -> None:
        flag = FeatureFlag(key="test-flag", name="Test Flag")
        assert flag.key == "test-flag"
        assert flag.name == "Test Flag"
        assert flag.status == FlagStatus.DISABLED
        assert flag.strategy == RolloutStrategy.NONE
        assert flag.default_value is False

    def test_key_validation(self) -> None:
        flag = FeatureFlag(key="My-Feature_Flag", name="Test")
        assert flag.key == "my-feature_flag"

    def test_invalid_key(self) -> None:
        with pytest.raises(ValueError):
            FeatureFlag(key="invalid@key!", name="Test")

    def test_disabled_flag_evaluation(self) -> None:
        flag = FeatureFlag(key="disabled-flag", name="Disabled", status=FlagStatus.DISABLED)
        assert flag.evaluate("user-1") is False

    def test_enabled_all_strategy(self) -> None:
        flag = FeatureFlag(
            key="all-enabled", name="All Enabled",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL
        )
        assert flag.evaluate("any-user") is True

    def test_enabled_none_strategy(self) -> None:
        flag = FeatureFlag(
            key="none-enabled", name="None Enabled",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.NONE
        )
        assert flag.evaluate("any-user") is False

    def test_percentage_rollout(self) -> None:
        flag = FeatureFlag(
            key="percentage-flag", name="Percentage",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.PERCENTAGE,
            percentage=50
        )
        results = [flag.evaluate(f"user-{i}") for i in range(100)]
        enabled_count = sum(results)
        assert 30 <= enabled_count <= 70

    def test_allowed_users(self) -> None:
        flag = FeatureFlag(
            key="allowlist-flag", name="Allowlist",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.USER_LIST,
            allowed_users=["vip-user-1", "vip-user-2"]
        )
        assert flag.evaluate("vip-user-1") is True
        assert flag.evaluate("regular-user") is False

    def test_blocked_users(self) -> None:
        flag = FeatureFlag(
            key="blocklist-flag", name="Blocklist",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL,
            blocked_users=["banned-user"]
        )
        assert flag.evaluate("banned-user") is False
        assert flag.evaluate("normal-user") is True

    def test_expired_flag(self) -> None:
        flag = FeatureFlag(
            key="expired-flag", name="Expired",
            status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL,
            default_value=False,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)
        )
        assert flag.is_expired is True
        assert flag.evaluate("user") is False

    def test_conditional_with_targeting(self) -> None:
        flag = FeatureFlag(
            key="conditional-flag", name="Conditional",
            status=FlagStatus.CONDITIONAL,
            targeting_groups=[
                TargetingGroup(rules=[
                    TargetingRule(attribute="tier", operator=TargetingOperator.EQUALS, value="premium")
                ])
            ]
        )
        assert flag.evaluate("user-1", {"tier": "premium"}) is True
        assert flag.evaluate("user-2", {"tier": "free"}) is False


class TestFeatureFlagSettings:
    """Tests for FeatureFlagSettings model."""

    def test_default_settings(self) -> None:
        settings = FeatureFlagSettings()
        assert settings.storage_backend == "memory"
        assert settings.cache_ttl_seconds == 60
        assert settings.sync_interval_seconds == 30
        assert settings.evaluation_logging is True
        assert settings.default_enabled is False


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_evaluation_result(self) -> None:
        result = EvaluationResult(
            flag_key="test-flag",
            enabled=True,
            reason="user_allowed",
            user_id="user-123"
        )
        assert result.flag_key == "test-flag"
        assert result.enabled is True
        assert result.reason == "user_allowed"
        assert isinstance(result.evaluated_at, datetime)


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager."""

    @pytest.fixture
    def manager(self) -> FeatureFlagManager:
        return FeatureFlagManager(FeatureFlagSettings(evaluation_logging=False))

    @pytest.mark.asyncio
    async def test_initialize(self, manager: FeatureFlagManager) -> None:
        await manager.initialize()

    @pytest.mark.asyncio
    async def test_register_and_get_flag(self, manager: FeatureFlagManager) -> None:
        flag = FeatureFlag(key="new-feature", name="New Feature", status=FlagStatus.ENABLED)
        await manager.register_flag(flag)
        retrieved = await manager.get_flag("new-feature")
        assert retrieved is not None
        assert retrieved.key == "new-feature"

    @pytest.mark.asyncio
    async def test_update_flag(self, manager: FeatureFlagManager) -> None:
        flag = FeatureFlag(key="updatable", name="Original", status=FlagStatus.DISABLED)
        await manager.register_flag(flag)
        updated = await manager.update_flag("updatable", {"status": FlagStatus.ENABLED, "name": "Updated"})
        assert updated is not None
        assert updated.status == FlagStatus.ENABLED
        assert updated.name == "Updated"

    @pytest.mark.asyncio
    async def test_update_nonexistent_flag(self, manager: FeatureFlagManager) -> None:
        result = await manager.update_flag("nonexistent", {"status": FlagStatus.ENABLED})
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_flag(self, manager: FeatureFlagManager) -> None:
        flag = FeatureFlag(key="to-delete", name="Delete Me")
        await manager.register_flag(flag)
        deleted = await manager.delete_flag("to-delete")
        assert deleted is True
        retrieved = await manager.get_flag("to-delete")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_flag(self, manager: FeatureFlagManager) -> None:
        deleted = await manager.delete_flag("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_flags(self, manager: FeatureFlagManager) -> None:
        await manager.register_flag(FeatureFlag(key="flag-1", name="Flag 1", tags=["beta"]))
        await manager.register_flag(FeatureFlag(key="flag-2", name="Flag 2", tags=["stable"]))
        all_flags = await manager.list_flags()
        assert len(all_flags) == 2

    @pytest.mark.asyncio
    async def test_list_flags_by_status(self, manager: FeatureFlagManager) -> None:
        await manager.register_flag(FeatureFlag(key="enabled-1", name="E1", status=FlagStatus.ENABLED))
        await manager.register_flag(FeatureFlag(key="disabled-1", name="D1", status=FlagStatus.DISABLED))
        enabled = await manager.list_flags(status=FlagStatus.ENABLED)
        assert len(enabled) == 1
        assert enabled[0].key == "enabled-1"

    @pytest.mark.asyncio
    async def test_list_flags_by_tags(self, manager: FeatureFlagManager) -> None:
        await manager.register_flag(FeatureFlag(key="beta-flag", name="Beta", tags=["beta", "experimental"]))
        await manager.register_flag(FeatureFlag(key="stable-flag", name="Stable", tags=["stable"]))
        beta_flags = await manager.list_flags(tags=["beta"])
        assert len(beta_flags) == 1
        assert beta_flags[0].key == "beta-flag"

    def test_is_enabled(self, manager: FeatureFlagManager) -> None:
        manager._flags["sync-flag"] = FeatureFlag(
            key="sync-flag", name="Sync", status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL
        )
        assert manager.is_enabled("sync-flag") is True
        assert manager.is_enabled("nonexistent") is False

    def test_is_enabled_with_default(self, manager: FeatureFlagManager) -> None:
        assert manager.is_enabled("missing", default=True) is True
        assert manager.is_enabled("missing", default=False) is False

    @pytest.mark.asyncio
    async def test_evaluate(self, manager: FeatureFlagManager) -> None:
        await manager.register_flag(FeatureFlag(
            key="eval-flag", name="Eval", status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL
        ))
        result = await manager.evaluate("eval-flag", "user-1")
        assert result.enabled is True
        assert result.flag_key == "eval-flag"

    @pytest.mark.asyncio
    async def test_evaluate_nonexistent(self, manager: FeatureFlagManager) -> None:
        result = await manager.evaluate("nonexistent", "user-1")
        assert result.enabled is False
        assert result.reason == "flag_not_found"

    @pytest.mark.asyncio
    async def test_bulk_evaluate(self, manager: FeatureFlagManager) -> None:
        await manager.register_flag(FeatureFlag(
            key="bulk-1", name="B1", status=FlagStatus.ENABLED, strategy=RolloutStrategy.ALL
        ))
        await manager.register_flag(FeatureFlag(
            key="bulk-2", name="B2", status=FlagStatus.DISABLED
        ))
        results = await manager.bulk_evaluate(["bulk-1", "bulk-2", "missing"], "user-1")
        assert results["bulk-1"] is True
        assert results["bulk-2"] is False
        assert results["missing"] is False

    @pytest.mark.asyncio
    async def test_shutdown(self, manager: FeatureFlagManager) -> None:
        await manager.initialize()
        await manager.shutdown()
        assert len(manager._cache) == 0

    def test_register_listener(self, manager: FeatureFlagManager) -> None:
        async def callback(key: str, enabled: bool) -> None:
            pass
        manager.register_listener(callback)
        assert len(manager._listeners) == 1


class TestGetFeatureFlagManager:
    """Tests for get_feature_flag_manager singleton."""

    def test_get_singleton(self) -> None:
        import config_service.src.feature_flags as ff_module
        ff_module._manager = None
        mgr1 = get_feature_flag_manager()
        mgr2 = get_feature_flag_manager()
        assert mgr1 is mgr2


class TestInitializeFeatureFlags:
    """Tests for initialize_feature_flags function."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        import config_service.src.feature_flags as ff_module
        ff_module._manager = None
        mgr = await initialize_feature_flags(FeatureFlagSettings())
        assert isinstance(mgr, FeatureFlagManager)
        await mgr.shutdown()
