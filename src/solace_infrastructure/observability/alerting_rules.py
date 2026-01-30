"""Solace-AI AlertManager Rules - Alerting configuration, routing, and receivers."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ReceiverType(str, Enum):
    """AlertManager receiver types."""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    OPSGENIE = "opsgenie"


class AlertManagerSettings(BaseSettings):
    """AlertManager configuration from environment."""
    resolve_timeout: str = Field(default="5m")
    smtp_smarthost: str = Field(default="smtp.example.com:587")
    smtp_from: str = Field(default="alertmanager@solace-ai.com")
    smtp_require_tls: bool = Field(default=True)
    slack_api_url: SecretStr | None = Field(default=None)
    pagerduty_service_key: SecretStr | None = Field(default=None)
    opsgenie_api_key: SecretStr | None = Field(default=None)
    webhook_url: str | None = Field(default=None)
    group_wait: str = Field(default="30s")
    group_interval: str = Field(default="5m")
    repeat_interval: str = Field(default="4h")
    model_config = SettingsConfigDict(env_prefix="ALERTMANAGER_", env_file=".env", extra="ignore")


@dataclass
class AlertLabel:
    """Label matcher for alert routing."""
    name: str
    value: str
    regex: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"match_re" if self.regex else "match": {self.name: self.value}}


@dataclass
class AlertRule:
    """Prometheus alerting rule definition."""
    alert_name: str
    expr: str
    duration: str
    severity: AlertSeverity
    summary: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    runbook_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        labels = {"severity": self.severity.value, **self.labels}
        annotations = {"summary": self.summary, "description": self.description}
        if self.runbook_url:
            annotations["runbook_url"] = self.runbook_url
        return {"alert": self.alert_name, "expr": self.expr, "for": self.duration,
                "labels": labels, "annotations": annotations}


@dataclass
class AlertRuleGroup:
    """Group of related alert rules."""
    name: str
    rules: list[AlertRule]
    interval: str = "30s"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "interval": self.interval, "rules": [r.to_dict() for r in self.rules]}


class SolaceAlertRules:
    """Factory for Solace-AI alert rule definitions."""

    @staticmethod
    def safety_critical_rules() -> AlertRuleGroup:
        """CRITICAL safety service alerts - highest priority."""
        return AlertRuleGroup("solace-safety-critical", [
            AlertRule("SafetyServiceDown", "up{job='solace-safety'} == 0", "30s", AlertSeverity.CRITICAL,
                      "Safety service is DOWN", "Safety service {{$labels.instance}} is not responding",
                      {"service": "safety", "team": "critical"}),
            AlertRule("CrisisDetectionLatencyHigh", "histogram_quantile(0.99, sum(rate(solace_crisis_detection_duration_bucket[5m])) by (le)) > 0.050", "1m",
                      AlertSeverity.CRITICAL, "Crisis detection latency exceeds 50ms",
                      "P99 crisis detection latency is {{$value}}s, exceeding 50ms SLO",
                      {"service": "safety"}),
            AlertRule("EscalationFailures", "rate(solace_escalation_failures_total[5m]) > 0", "1m",
                      AlertSeverity.CRITICAL, "Escalation failures detected",
                      "{{$value}} escalation failures in the last 5 minutes",
                      {"service": "safety"}),
        ])

    @staticmethod
    def system_health_rules() -> AlertRuleGroup:
        """System-wide health alerts."""
        return AlertRuleGroup("solace-system-health", [
            AlertRule("HighErrorRate", "sum(rate(solace_errors_total[5m]))/sum(rate(solace_requests_total[5m])) > 0.05", "5m",
                      AlertSeverity.WARNING, "Error rate exceeds 5%",
                      "Error rate is {{$value | humanizePercentage}}",
                      {"component": "system"}),
            AlertRule("HighLatency", "histogram_quantile(0.99, sum(rate(solace_request_duration_bucket[5m])) by (le)) > 2", "5m",
                      AlertSeverity.WARNING, "P99 latency exceeds 2s",
                      "P99 latency is {{$value}}s",
                      {"component": "system"}),
            AlertRule("ServiceDown", "up == 0", "1m", AlertSeverity.CRITICAL, "Service is DOWN",
                      "{{$labels.job}} service on {{$labels.instance}} is not responding",
                      {"component": "infrastructure"}),
        ])

    @staticmethod
    def infrastructure_rules() -> AlertRuleGroup:
        """Infrastructure component alerts."""
        return AlertRuleGroup("solace-infrastructure", [
            AlertRule("PostgresConnectionsHigh", "pg_stat_activity_count > 100", "5m", AlertSeverity.WARNING,
                      "High PostgreSQL connections", "{{$value}} active PostgreSQL connections"),
            AlertRule("RedisMemoryHigh", "redis_memory_used_bytes / redis_memory_max_bytes > 0.85", "5m",
                      AlertSeverity.WARNING, "Redis memory usage above 85%",
                      "Redis memory at {{$value | humanizePercentage}}"),
            AlertRule("KafkaConsumerLag", "kafka_consumer_group_lag > 10000", "5m", AlertSeverity.WARNING,
                      "High Kafka consumer lag", "Consumer group {{$labels.group}} lag is {{$value}}"),
            AlertRule("WeaviateQueryLatency", "histogram_quantile(0.95, rate(weaviate_query_duration_seconds_bucket[5m])) > 0.5", "5m",
                      AlertSeverity.WARNING, "Weaviate query latency high", "P95 query latency is {{$value}}s"),
        ])

    @staticmethod
    def llm_rules() -> AlertRuleGroup:
        """LLM provider alerts."""
        return AlertRuleGroup("solace-llm", [
            AlertRule("LLMProviderErrors", "rate(solace_llm_errors_total[5m]) > 0.1", "5m", AlertSeverity.WARNING,
                      "LLM provider errors", "{{$labels.provider}} error rate: {{$value}}/s"),
            AlertRule("LLMLatencyHigh", "histogram_quantile(0.95, sum(rate(solace_llm_inference_duration_bucket[5m])) by (le)) > 10", "5m",
                      AlertSeverity.WARNING, "LLM inference latency high", "P95 inference latency is {{$value}}s"),
        ])

    @classmethod
    def all_rule_groups(cls) -> list[AlertRuleGroup]:
        """All Solace-AI alert rule groups."""
        return [cls.safety_critical_rules(), cls.system_health_rules(), cls.infrastructure_rules(), cls.llm_rules()]


@dataclass
class Receiver:
    """AlertManager receiver configuration."""
    name: str
    receiver_type: ReceiverType
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        receiver: dict[str, Any] = {"name": self.name}
        if self.receiver_type == ReceiverType.SLACK:
            receiver["slack_configs"] = [self.config]
        elif self.receiver_type == ReceiverType.PAGERDUTY:
            receiver["pagerduty_configs"] = [self.config]
        elif self.receiver_type == ReceiverType.EMAIL:
            receiver["email_configs"] = [self.config]
        elif self.receiver_type == ReceiverType.WEBHOOK:
            receiver["webhook_configs"] = [self.config]
        return receiver


@dataclass
class Route:
    """AlertManager routing rule."""
    receiver: str
    matchers: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=lambda: ["alertname"])
    group_wait: str = "30s"
    group_interval: str = "5m"
    repeat_interval: str = "4h"
    continue_matching: bool = False
    child_routes: list["Route"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {"receiver": self.receiver, "group_by": self.group_by,
                                   "group_wait": self.group_wait, "group_interval": self.group_interval,
                                   "repeat_interval": self.repeat_interval}
        if self.matchers:
            config["matchers"] = self.matchers
        if self.continue_matching:
            config["continue"] = True
        if self.child_routes:
            config["routes"] = [r.to_dict() for r in self.child_routes]
        return config


@dataclass
class InhibitRule:
    """Alert inhibition rule."""
    source_matchers: list[str]
    target_matchers: list[str]
    equal: list[str] = field(default_factory=lambda: ["alertname"])

    def to_dict(self) -> dict[str, Any]:
        return {"source_matchers": self.source_matchers, "target_matchers": self.target_matchers, "equal": self.equal}


class AlertManagerConfigGenerator:
    """Generates complete AlertManager configuration."""

    def __init__(self, settings: AlertManagerSettings | None = None) -> None:
        self._settings = settings or AlertManagerSettings()

    def generate_global_config(self) -> dict[str, Any]:
        """Generate global configuration."""
        return {"resolve_timeout": self._settings.resolve_timeout,
                "smtp_smarthost": self._settings.smtp_smarthost,
                "smtp_from": self._settings.smtp_from,
                "smtp_require_tls": self._settings.smtp_require_tls}

    def generate_default_receivers(self) -> list[Receiver]:
        """Generate default receiver configurations."""
        receivers = [Receiver("default", ReceiverType.WEBHOOK, {"url": self._settings.webhook_url or "http://localhost:9999"})]
        if self._settings.slack_api_url:
            slack_url = self._settings.slack_api_url.get_secret_value()
            receivers.append(Receiver("slack-critical", ReceiverType.SLACK,
                                       {"api_url": slack_url, "channel": "#solace-alerts-critical"}))
            receivers.append(Receiver("slack-warning", ReceiverType.SLACK,
                                       {"api_url": slack_url, "channel": "#solace-alerts"}))
        if self._settings.pagerduty_service_key:
            receivers.append(Receiver("pagerduty-critical", ReceiverType.PAGERDUTY,
                                       {"service_key": self._settings.pagerduty_service_key.get_secret_value()}))
        return receivers

    def generate_routing(self, default_receiver: str = "default") -> Route:
        """Generate alert routing configuration."""
        return Route(
            receiver=default_receiver, group_by=["alertname", "service"],
            child_routes=[
                Route("pagerduty-critical" if self._settings.pagerduty_service_key else "slack-critical",
                      matchers=['severity="critical"'], group_wait="10s", repeat_interval="1h"),
                Route("slack-warning" if self._settings.slack_api_url else "default",
                      matchers=['severity="warning"'], repeat_interval="4h"),
            ],
        )

    def generate_inhibit_rules(self) -> list[InhibitRule]:
        """Generate inhibition rules."""
        return [InhibitRule(['severity="critical"'], ['severity="warning"'], ["alertname", "service"])]

    def generate_full_config(self) -> dict[str, Any]:
        """Generate complete AlertManager configuration."""
        receivers = self.generate_default_receivers()
        config = {
            "global": self.generate_global_config(),
            "receivers": [r.to_dict() for r in receivers],
            "route": self.generate_routing().to_dict(),
            "inhibit_rules": [r.to_dict() for r in self.generate_inhibit_rules()],
        }
        logger.info("alertmanager_config_generated", receivers=len(receivers))
        return config

    def generate_prometheus_rules(self, groups: list[AlertRuleGroup] | None = None) -> dict[str, Any]:
        """Generate Prometheus alerting rules file."""
        groups = groups or SolaceAlertRules.all_rule_groups()
        return {"groups": [g.to_dict() for g in groups]}


def create_alertmanager_config(settings: AlertManagerSettings | None = None) -> dict[str, Any]:
    """Create default AlertManager configuration for Solace-AI."""
    generator = AlertManagerConfigGenerator(settings)
    return generator.generate_full_config()


def create_prometheus_alerting_rules() -> dict[str, Any]:
    """Create Prometheus alerting rules for Solace-AI."""
    generator = AlertManagerConfigGenerator()
    return generator.generate_prometheus_rules()
