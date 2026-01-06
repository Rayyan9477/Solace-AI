"""Unit tests for AlertManager rules module."""
from __future__ import annotations

import pytest

from solace_infrastructure.observability.alerting_rules import (
    AlertSeverity,
    ReceiverType,
    AlertManagerSettings,
    AlertLabel,
    AlertRule,
    AlertRuleGroup,
    SolaceAlertRules,
    Receiver,
    Route,
    InhibitRule,
    AlertManagerConfigGenerator,
    create_alertmanager_config,
    create_prometheus_alerting_rules,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_critical_value(self) -> None:
        """Test critical severity."""
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_warning_value(self) -> None:
        """Test warning severity."""
        assert AlertSeverity.WARNING.value == "warning"

    def test_info_value(self) -> None:
        """Test info severity."""
        assert AlertSeverity.INFO.value == "info"


class TestReceiverType:
    """Tests for ReceiverType enum."""

    def test_slack_value(self) -> None:
        """Test slack receiver."""
        assert ReceiverType.SLACK.value == "slack"

    def test_pagerduty_value(self) -> None:
        """Test pagerduty receiver."""
        assert ReceiverType.PAGERDUTY.value == "pagerduty"

    def test_email_value(self) -> None:
        """Test email receiver."""
        assert ReceiverType.EMAIL.value == "email"


class TestAlertManagerSettings:
    """Tests for AlertManagerSettings."""

    def test_default_resolve_timeout(self) -> None:
        """Test default resolve timeout."""
        settings = AlertManagerSettings()
        assert settings.resolve_timeout == "5m"

    def test_default_group_wait(self) -> None:
        """Test default group wait."""
        settings = AlertManagerSettings()
        assert settings.group_wait == "30s"

    def test_default_repeat_interval(self) -> None:
        """Test default repeat interval."""
        settings = AlertManagerSettings()
        assert settings.repeat_interval == "4h"


class TestAlertLabel:
    """Tests for AlertLabel dataclass."""

    def test_label_creation(self) -> None:
        """Test creating an alert label."""
        label = AlertLabel("severity", "critical")
        assert label.name == "severity"
        assert label.value == "critical"

    def test_to_dict_exact_match(self) -> None:
        """Test to_dict for exact match."""
        label = AlertLabel("severity", "critical")
        result = label.to_dict()
        assert "match" in result
        assert result["match"]["severity"] == "critical"

    def test_to_dict_regex_match(self) -> None:
        """Test to_dict for regex match."""
        label = AlertLabel("job", "solace-.*", regex=True)
        result = label.to_dict()
        assert "match_re" in result


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_rule_creation(self) -> None:
        """Test creating an alert rule."""
        rule = AlertRule("TestAlert", "up == 0", "1m", AlertSeverity.CRITICAL,
                        "Test is down", "Service {{$labels.instance}} is not responding")
        assert rule.alert_name == "TestAlert"
        assert rule.severity == AlertSeverity.CRITICAL

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        rule = AlertRule("TestAlert", "up == 0", "1m", AlertSeverity.CRITICAL,
                        "Test is down", "Description")
        result = rule.to_dict()
        assert result["alert"] == "TestAlert"
        assert result["expr"] == "up == 0"
        assert result["for"] == "1m"
        assert result["labels"]["severity"] == "critical"
        assert "summary" in result["annotations"]


class TestAlertRuleGroup:
    """Tests for AlertRuleGroup dataclass."""

    def test_group_creation(self) -> None:
        """Test creating a rule group."""
        rule = AlertRule("Test", "up==0", "1m", AlertSeverity.WARNING, "Test", "Desc")
        group = AlertRuleGroup("test-group", [rule])
        assert group.name == "test-group"
        assert len(group.rules) == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        rule = AlertRule("Test", "up==0", "1m", AlertSeverity.WARNING, "Test", "Desc")
        group = AlertRuleGroup("test-group", [rule])
        result = group.to_dict()
        assert result["name"] == "test-group"
        assert len(result["rules"]) == 1


class TestSolaceAlertRules:
    """Tests for SolaceAlertRules factory."""

    def test_safety_critical_rules(self) -> None:
        """Test safety critical rules group."""
        group = SolaceAlertRules.safety_critical_rules()
        assert group.name == "solace-safety-critical"
        assert len(group.rules) > 0

    def test_safety_rules_are_critical(self) -> None:
        """Test all safety rules are critical severity."""
        group = SolaceAlertRules.safety_critical_rules()
        for rule in group.rules:
            assert rule.severity == AlertSeverity.CRITICAL

    def test_system_health_rules(self) -> None:
        """Test system health rules group."""
        group = SolaceAlertRules.system_health_rules()
        assert group.name == "solace-system-health"

    def test_infrastructure_rules(self) -> None:
        """Test infrastructure rules group."""
        group = SolaceAlertRules.infrastructure_rules()
        assert group.name == "solace-infrastructure"

    def test_all_rule_groups(self) -> None:
        """Test all_rule_groups returns complete list."""
        groups = SolaceAlertRules.all_rule_groups()
        assert len(groups) == 4
        names = [g.name for g in groups]
        assert "solace-safety-critical" in names


class TestReceiver:
    """Tests for Receiver dataclass."""

    def test_slack_receiver(self) -> None:
        """Test slack receiver creation."""
        receiver = Receiver("slack-test", ReceiverType.SLACK, {"channel": "#alerts"})
        assert receiver.name == "slack-test"
        assert receiver.receiver_type == ReceiverType.SLACK

    def test_to_dict_slack(self) -> None:
        """Test to_dict for slack receiver."""
        receiver = Receiver("slack-test", ReceiverType.SLACK, {"channel": "#alerts"})
        result = receiver.to_dict()
        assert result["name"] == "slack-test"
        assert "slack_configs" in result


class TestRoute:
    """Tests for Route dataclass."""

    def test_route_creation(self) -> None:
        """Test creating a route."""
        route = Route("default")
        assert route.receiver == "default"
        assert route.group_wait == "30s"

    def test_route_with_matchers(self) -> None:
        """Test route with matchers."""
        route = Route("critical", matchers=['severity="critical"'])
        assert len(route.matchers) == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        route = Route("default", group_by=["alertname", "service"])
        result = route.to_dict()
        assert result["receiver"] == "default"
        assert result["group_by"] == ["alertname", "service"]


class TestInhibitRule:
    """Tests for InhibitRule dataclass."""

    def test_inhibit_rule_creation(self) -> None:
        """Test creating an inhibit rule."""
        rule = InhibitRule(['severity="critical"'], ['severity="warning"'])
        assert len(rule.source_matchers) == 1
        assert len(rule.target_matchers) == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        rule = InhibitRule(['severity="critical"'], ['severity="warning"'], ["alertname"])
        result = rule.to_dict()
        assert result["source_matchers"] == ['severity="critical"']
        assert result["equal"] == ["alertname"]


class TestAlertManagerConfigGenerator:
    """Tests for AlertManagerConfigGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be initialized."""
        generator = AlertManagerConfigGenerator()
        assert generator is not None

    def test_generate_global_config(self) -> None:
        """Test generate global config."""
        generator = AlertManagerConfigGenerator()
        config = generator.generate_global_config()
        assert "resolve_timeout" in config

    def test_generate_default_receivers(self) -> None:
        """Test generate default receivers."""
        generator = AlertManagerConfigGenerator()
        receivers = generator.generate_default_receivers()
        assert len(receivers) >= 1

    def test_generate_routing(self) -> None:
        """Test generate routing."""
        generator = AlertManagerConfigGenerator()
        route = generator.generate_routing()
        assert route.receiver == "default"

    def test_generate_full_config(self) -> None:
        """Test generate full config."""
        generator = AlertManagerConfigGenerator()
        config = generator.generate_full_config()
        assert "global" in config
        assert "receivers" in config
        assert "route" in config

    def test_generate_prometheus_rules(self) -> None:
        """Test generate prometheus rules."""
        generator = AlertManagerConfigGenerator()
        config = generator.generate_prometheus_rules()
        assert "groups" in config


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_alertmanager_config(self) -> None:
        """Test create_alertmanager_config factory."""
        config = create_alertmanager_config()
        assert isinstance(config, dict)
        assert "global" in config

    def test_create_prometheus_alerting_rules(self) -> None:
        """Test create_prometheus_alerting_rules factory."""
        config = create_prometheus_alerting_rules()
        assert isinstance(config, dict)
        assert "groups" in config
