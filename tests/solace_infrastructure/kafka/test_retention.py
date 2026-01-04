"""Unit tests for Kafka Retention Policy Management module."""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from solace_infrastructure.kafka.retention import (
    RetentionType,
    ComplianceCategory,
    RetentionPriority,
    RetentionMetrics,
    RetentionPolicy,
    TopicRetentionAssignment,
    RetentionManager,
    PRESET_POLICIES,
    create_retention_manager,
    get_hipaa_policy,
)


class TestRetentionEnums:
    """Tests for retention-related enums."""

    def test_retention_type_values(self) -> None:
        assert RetentionType.TIME_BASED.value == "time_based"
        assert RetentionType.SIZE_BASED.value == "size_based"
        assert RetentionType.COMPACTION.value == "compaction"
        assert RetentionType.HYBRID.value == "hybrid"

    def test_compliance_category_values(self) -> None:
        assert ComplianceCategory.HIPAA_PHI.value == "hipaa_phi"
        assert ComplianceCategory.HIPAA_AUDIT.value == "hipaa_audit"
        assert ComplianceCategory.PII.value == "pii"
        assert ComplianceCategory.OPERATIONAL.value == "operational"

    def test_retention_priority_values(self) -> None:
        assert RetentionPriority.CRITICAL.value == "critical"
        assert RetentionPriority.HIGH.value == "high"
        assert RetentionPriority.NORMAL.value == "normal"
        assert RetentionPriority.LOW.value == "low"


class TestRetentionPolicy:
    """Tests for RetentionPolicy model."""

    def test_default_policy(self) -> None:
        policy = RetentionPolicy(name="test-policy")
        assert policy.retention_type == RetentionType.TIME_BASED
        assert policy.retention_ms == 604800000  # 7 days
        assert policy.compliance_category == ComplianceCategory.OPERATIONAL

    def test_custom_policy(self) -> None:
        policy = RetentionPolicy(
            name="custom-policy",
            retention_type=RetentionType.COMPACTION,
            retention_ms=86400000,
            min_cleanable_ratio=0.3,
        )
        assert policy.retention_type == RetentionType.COMPACTION
        assert policy.retention_ms == 86400000
        assert policy.min_cleanable_ratio == 0.3

    def test_retention_days_calculation(self) -> None:
        policy = RetentionPolicy(name="test", retention_ms=86400000)  # 1 day
        assert policy.retention_days == 1.0

    def test_retention_days_infinite(self) -> None:
        policy = RetentionPolicy(name="test", retention_ms=-1)
        assert policy.retention_days == float("inf")

    def test_to_kafka_config_delete(self) -> None:
        policy = RetentionPolicy(
            name="delete-policy",
            retention_type=RetentionType.TIME_BASED,
            retention_ms=604800000,
        )
        config = policy.to_kafka_config()
        assert config["cleanup.policy"] == "delete"
        assert config["retention.ms"] == "604800000"

    def test_to_kafka_config_compact(self) -> None:
        policy = RetentionPolicy(
            name="compact-policy",
            retention_type=RetentionType.COMPACTION,
        )
        config = policy.to_kafka_config()
        assert config["cleanup.policy"] == "compact"

    def test_to_kafka_config_hybrid(self) -> None:
        policy = RetentionPolicy(
            name="hybrid-policy",
            retention_type=RetentionType.HYBRID,
        )
        config = policy.to_kafka_config()
        assert config["cleanup.policy"] == "compact,delete"


class TestPresetPolicies:
    """Tests for preset retention policies."""

    def test_hipaa_audit_6yr_exists(self) -> None:
        policy = PRESET_POLICIES.get("hipaa_audit_6yr")
        assert policy is not None
        assert policy.compliance_category == ComplianceCategory.HIPAA_AUDIT
        assert policy.retention_ms == 189216000000  # 6 years

    def test_hipaa_phi_7yr_exists(self) -> None:
        policy = PRESET_POLICIES.get("hipaa_phi_7yr")
        assert policy is not None
        assert policy.compliance_category == ComplianceCategory.HIPAA_PHI
        assert policy.retention_ms == 220752000000  # 7 years

    def test_safety_1yr_exists(self) -> None:
        policy = PRESET_POLICIES.get("safety_1yr")
        assert policy is not None
        assert policy.retention_ms == 31536000000  # 1 year

    def test_session_90d_exists(self) -> None:
        policy = PRESET_POLICIES.get("session_90d")
        assert policy is not None
        assert policy.retention_ms == 7776000000  # 90 days

    def test_analytics_30d_exists(self) -> None:
        policy = PRESET_POLICIES.get("analytics_30d")
        assert policy is not None
        assert policy.retention_ms == 2592000000  # 30 days

    def test_operational_7d_exists(self) -> None:
        policy = PRESET_POLICIES.get("operational_7d")
        assert policy is not None
        assert policy.retention_ms == 604800000  # 7 days

    def test_compacted_profile_exists(self) -> None:
        policy = PRESET_POLICIES.get("compacted_profile")
        assert policy is not None
        assert policy.retention_type == RetentionType.COMPACTION


class TestRetentionManager:
    """Tests for RetentionManager."""

    @pytest.fixture
    def manager(self) -> RetentionManager:
        return RetentionManager()

    def test_default_assignments(self, manager: RetentionManager) -> None:
        assignments = manager.list_assignments()
        topic_names = [a.topic_name for a in assignments]
        assert "solace.sessions" in topic_names
        assert "solace.safety" in topic_names
        assert "solace.assessments" in topic_names

    def test_solace_sessions_assignment(self, manager: RetentionManager) -> None:
        assignment = manager.get_topic_retention("solace.sessions")
        assert assignment is not None
        assert assignment.policy.name == "session_90d"

    def test_solace_safety_assignment(self, manager: RetentionManager) -> None:
        assignment = manager.get_topic_retention("solace.safety")
        assert assignment is not None
        assert assignment.policy.name == "safety_1yr"

    def test_dlq_topics_assigned(self, manager: RetentionManager) -> None:
        assignment = manager.get_topic_retention("solace.sessions.dlq")
        assert assignment is not None
        assert assignment.policy.name == "operational_7d"

    def test_register_custom_policy(self, manager: RetentionManager) -> None:
        custom = RetentionPolicy(
            name="custom-180d",
            retention_ms=15552000000,  # 180 days
            compliance_category=ComplianceCategory.PII,
        )
        manager.register_policy(custom)
        retrieved = manager.get_policy("custom-180d")
        assert retrieved is not None
        assert retrieved.retention_ms == 15552000000

    def test_assign_policy(self, manager: RetentionManager) -> None:
        success = manager.assign_policy("custom.topic", "session_90d")
        assert success is True
        assignment = manager.get_topic_retention("custom.topic")
        assert assignment is not None
        assert assignment.policy.name == "session_90d"

    def test_assign_unknown_policy(self, manager: RetentionManager) -> None:
        success = manager.assign_policy("custom.topic", "nonexistent-policy")
        assert success is False

    def test_assign_with_override(self, manager: RetentionManager) -> None:
        override = {"retention.ms": "1000000"}
        manager.assign_policy("custom.topic", "session_90d", override)
        config = manager.get_topic_config("custom.topic")
        assert config["retention.ms"] == "1000000"

    def test_get_topic_config_unknown(self, manager: RetentionManager) -> None:
        config = manager.get_topic_config("unknown.topic")
        # Should return operational_7d as default
        assert config["retention.ms"] == "604800000"

    def test_list_policies(self, manager: RetentionManager) -> None:
        policies = manager.list_policies()
        assert len(policies) >= len(PRESET_POLICIES)

    def test_get_compliance_topics(self, manager: RetentionManager) -> None:
        hipaa_phi_topics = manager.get_compliance_topics(ComplianceCategory.HIPAA_PHI)
        assert "solace.assessments" in hipaa_phi_topics
        assert "solace.therapy" in hipaa_phi_topics

    def test_validate_compliance_success(self, manager: RetentionManager) -> None:
        valid, issues = manager.validate_compliance("solace.safety")
        assert valid is True
        assert len(issues) == 0

    def test_validate_compliance_no_policy(self, manager: RetentionManager) -> None:
        valid, issues = manager.validate_compliance("unknown.topic")
        assert valid is False
        assert any("No retention policy" in issue for issue in issues)


class TestTopicRetentionAssignment:
    """Tests for TopicRetentionAssignment dataclass."""

    def test_assignment_creation(self) -> None:
        policy = RetentionPolicy(name="test-policy")
        assignment = TopicRetentionAssignment(
            topic_name="test.topic",
            policy=policy,
        )
        assert assignment.topic_name == "test.topic"
        assert assignment.policy.name == "test-policy"

    def test_assignment_with_override(self) -> None:
        policy = RetentionPolicy(name="test-policy")
        override = {"retention.ms": "1000"}
        assignment = TopicRetentionAssignment(
            topic_name="test.topic",
            policy=policy,
            override_config=override,
        )
        assert assignment.override_config["retention.ms"] == "1000"


class TestRetentionMetrics:
    """Tests for RetentionMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        metrics = RetentionMetrics(
            topic_name="test.topic",
            current_size_bytes=1024000,
            segments_count=10,
        )
        assert metrics.topic_name == "test.topic"
        assert metrics.current_size_bytes == 1024000
        assert metrics.segments_count == 10


class TestHipaaPolicy:
    """Tests for get_hipaa_policy function."""

    def test_get_hipaa_audit_policy(self) -> None:
        policy = get_hipaa_policy(ComplianceCategory.HIPAA_AUDIT)
        assert policy is not None
        assert policy.name == "hipaa_audit_6yr"

    def test_get_hipaa_phi_policy(self) -> None:
        policy = get_hipaa_policy(ComplianceCategory.HIPAA_PHI)
        assert policy is not None
        assert policy.name == "hipaa_phi_7yr"

    def test_get_pii_policy(self) -> None:
        policy = get_hipaa_policy(ComplianceCategory.PII)
        assert policy is not None
        assert policy.name == "session_90d"

    def test_get_operational_policy(self) -> None:
        policy = get_hipaa_policy(ComplianceCategory.OPERATIONAL)
        assert policy is not None
        assert policy.name == "operational_7d"


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_retention_manager(self) -> None:
        manager = create_retention_manager()
        assert isinstance(manager, RetentionManager)
        # Should have default assignments
        assignments = manager.list_assignments()
        assert len(assignments) > 0
