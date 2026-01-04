"""
Solace-AI Kafka Retention Policies - Data lifecycle management.
HIPAA-compliant retention with configurable policies per topic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class RetentionType(str, Enum):
    """Types of retention policies."""
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    COMPACTION = "compaction"
    HYBRID = "hybrid"


class ComplianceCategory(str, Enum):
    """Data compliance categories affecting retention."""
    HIPAA_PHI = "hipaa_phi"
    HIPAA_AUDIT = "hipaa_audit"
    PII = "pii"
    OPERATIONAL = "operational"
    ANALYTICS = "analytics"


class RetentionPriority(str, Enum):
    """Priority for retention enforcement."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class RetentionMetrics:
    """Metrics for retention policy enforcement."""
    topic_name: str
    current_size_bytes: int = 0
    current_age_ms: int = 0
    segments_count: int = 0
    segments_deleted: int = 0
    compaction_ratio: float = 0.0
    last_cleanup_time: int | None = None


class RetentionPolicy(BaseModel):
    """Complete retention policy definition."""

    name: str = Field(..., min_length=1)
    retention_type: RetentionType = Field(default=RetentionType.TIME_BASED)
    retention_ms: int = Field(default=604800000, ge=-1)  # 7 days
    retention_bytes: int = Field(default=-1, ge=-1)
    segment_ms: int = Field(default=86400000, ge=3600000)  # Min 1 hour
    segment_bytes: int = Field(default=1073741824, ge=1048576)  # Min 1MB
    min_cleanable_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    delete_retention_ms: int = Field(default=86400000, ge=0)  # 1 day
    min_compaction_lag_ms: int = Field(default=0, ge=0)
    max_compaction_lag_ms: int = Field(default=9223372036854775807, ge=0)
    compliance_category: ComplianceCategory = Field(default=ComplianceCategory.OPERATIONAL)
    priority: RetentionPriority = Field(default=RetentionPriority.NORMAL)
    description: str = Field(default="")

    @property
    def retention_days(self) -> float:
        """Get retention period in days."""
        if self.retention_ms <= 0:
            return float("inf")
        return self.retention_ms / (24 * 60 * 60 * 1000)

    def to_kafka_config(self) -> dict[str, str]:
        """Convert to Kafka topic config format."""
        config: dict[str, str] = {
            "retention.ms": str(self.retention_ms),
            "retention.bytes": str(self.retention_bytes),
            "segment.ms": str(self.segment_ms),
            "segment.bytes": str(self.segment_bytes),
            "min.cleanable.dirty.ratio": str(self.min_cleanable_ratio),
            "delete.retention.ms": str(self.delete_retention_ms),
            "min.compaction.lag.ms": str(self.min_compaction_lag_ms),
            "max.compaction.lag.ms": str(self.max_compaction_lag_ms),
        }
        if self.retention_type == RetentionType.COMPACTION:
            config["cleanup.policy"] = "compact"
        elif self.retention_type == RetentionType.HYBRID:
            config["cleanup.policy"] = "compact,delete"
        else:
            config["cleanup.policy"] = "delete"
        return config


# HIPAA-compliant preset policies
PRESET_POLICIES: dict[str, RetentionPolicy] = {
    "hipaa_audit_6yr": RetentionPolicy(
        name="hipaa_audit_6yr",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=189216000000,  # 6 years (HIPAA requirement)
        segment_ms=604800000,  # 7 days
        compliance_category=ComplianceCategory.HIPAA_AUDIT,
        priority=RetentionPriority.CRITICAL,
        description="HIPAA audit log retention - 6 years minimum",
    ),
    "hipaa_phi_7yr": RetentionPolicy(
        name="hipaa_phi_7yr",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=220752000000,  # 7 years
        segment_ms=604800000,
        compliance_category=ComplianceCategory.HIPAA_PHI,
        priority=RetentionPriority.CRITICAL,
        description="HIPAA PHI retention - 7 years minimum",
    ),
    "safety_1yr": RetentionPolicy(
        name="safety_1yr",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=31536000000,  # 1 year
        segment_ms=86400000,
        compliance_category=ComplianceCategory.HIPAA_PHI,
        priority=RetentionPriority.CRITICAL,
        description="Safety events - 1 year retention",
    ),
    "session_90d": RetentionPolicy(
        name="session_90d",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=7776000000,  # 90 days
        segment_ms=86400000,
        compliance_category=ComplianceCategory.PII,
        priority=RetentionPriority.HIGH,
        description="Session data - 90 days retention",
    ),
    "analytics_30d": RetentionPolicy(
        name="analytics_30d",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=2592000000,  # 30 days
        segment_ms=86400000,
        compliance_category=ComplianceCategory.ANALYTICS,
        priority=RetentionPriority.NORMAL,
        description="Analytics events - 30 days retention",
    ),
    "operational_7d": RetentionPolicy(
        name="operational_7d",
        retention_type=RetentionType.TIME_BASED,
        retention_ms=604800000,  # 7 days
        segment_ms=86400000,
        compliance_category=ComplianceCategory.OPERATIONAL,
        priority=RetentionPriority.LOW,
        description="Operational data - 7 days retention",
    ),
    "compacted_profile": RetentionPolicy(
        name="compacted_profile",
        retention_type=RetentionType.COMPACTION,
        retention_ms=-1,
        min_cleanable_ratio=0.3,
        min_compaction_lag_ms=3600000,  # 1 hour
        compliance_category=ComplianceCategory.PII,
        priority=RetentionPriority.HIGH,
        description="User profiles - compacted, latest state only",
    ),
}


@dataclass
class TopicRetentionAssignment:
    """Assignment of retention policy to topic."""
    topic_name: str
    policy: RetentionPolicy
    override_config: dict[str, str] = field(default_factory=dict)


class RetentionManager:
    """Manages retention policies across topics."""

    def __init__(self) -> None:
        self._policies: dict[str, RetentionPolicy] = dict(PRESET_POLICIES)
        self._assignments: dict[str, TopicRetentionAssignment] = {}
        self._setup_default_assignments()

    def _setup_default_assignments(self) -> None:
        """Configure default retention for Solace topics."""
        defaults: dict[str, str] = {
            "solace.sessions": "session_90d",
            "solace.assessments": "hipaa_phi_7yr",
            "solace.therapy": "hipaa_phi_7yr",
            "solace.safety": "safety_1yr",
            "solace.memory": "session_90d",
            "solace.analytics": "analytics_30d",
            "solace.personality": "session_90d",
        }
        for topic, policy_name in defaults.items():
            policy = self._policies.get(policy_name)
            if policy:
                self._assignments[topic] = TopicRetentionAssignment(topic, policy)
            self._assignments[f"{topic}.dlq"] = TopicRetentionAssignment(
                f"{topic}.dlq", self._policies["operational_7d"]
            )

    def register_policy(self, policy: RetentionPolicy) -> None:
        """Register a custom retention policy."""
        self._policies[policy.name] = policy
        logger.info("retention_policy_registered", name=policy.name, type=policy.retention_type.value)

    def get_policy(self, name: str) -> RetentionPolicy | None:
        """Get policy by name."""
        return self._policies.get(name)

    def assign_policy(self, topic: str, policy_name: str,
                     override_config: dict[str, str] | None = None) -> bool:
        """Assign retention policy to topic."""
        policy = self._policies.get(policy_name)
        if not policy:
            logger.error("unknown_policy", policy=policy_name)
            return False
        self._assignments[topic] = TopicRetentionAssignment(topic, policy, override_config or {})
        logger.info("retention_assigned", topic=topic, policy=policy_name)
        return True

    def get_topic_retention(self, topic: str) -> TopicRetentionAssignment | None:
        """Get retention assignment for topic."""
        return self._assignments.get(topic)

    def get_topic_config(self, topic: str) -> dict[str, str]:
        """Get complete Kafka config for topic retention."""
        assignment = self._assignments.get(topic)
        if not assignment:
            return self._policies["operational_7d"].to_kafka_config()
        config = assignment.policy.to_kafka_config()
        config.update(assignment.override_config)
        return config

    def list_policies(self) -> list[RetentionPolicy]:
        """List all registered policies."""
        return list(self._policies.values())

    def list_assignments(self) -> list[TopicRetentionAssignment]:
        """List all topic assignments."""
        return list(self._assignments.values())

    def get_compliance_topics(self, category: ComplianceCategory) -> list[str]:
        """Get topics matching compliance category."""
        return [
            assignment.topic_name
            for assignment in self._assignments.values()
            if assignment.policy.compliance_category == category
        ]

    def validate_compliance(self, topic: str) -> tuple[bool, list[str]]:
        """Validate topic meets compliance requirements."""
        assignment = self._assignments.get(topic)
        if not assignment:
            return False, ["No retention policy assigned"]
        issues: list[str] = []
        policy = assignment.policy
        if policy.compliance_category == ComplianceCategory.HIPAA_AUDIT:
            min_retention = 189216000000  # 6 years
            if 0 < policy.retention_ms < min_retention:
                issues.append(f"HIPAA audit requires 6 year retention")
        if policy.compliance_category == ComplianceCategory.HIPAA_PHI:
            if policy.retention_type == RetentionType.SIZE_BASED:
                issues.append("PHI should not use size-based retention alone")
        return len(issues) == 0, issues


def create_retention_manager() -> RetentionManager:
    """Factory function to create retention manager."""
    return RetentionManager()


def get_hipaa_policy(category: ComplianceCategory) -> RetentionPolicy | None:
    """Get appropriate HIPAA-compliant policy for category."""
    mapping: dict[ComplianceCategory, str] = {
        ComplianceCategory.HIPAA_AUDIT: "hipaa_audit_6yr",
        ComplianceCategory.HIPAA_PHI: "hipaa_phi_7yr",
        ComplianceCategory.PII: "session_90d",
        ComplianceCategory.OPERATIONAL: "operational_7d",
        ComplianceCategory.ANALYTICS: "analytics_30d",
    }
    policy_name = mapping.get(category)
    return PRESET_POLICIES.get(policy_name) if policy_name else None
