"""Unit tests for Grafana dashboards module."""
from __future__ import annotations

import pytest

from solace_infrastructure.observability.grafana_dashboards import (
    PanelType,
    DatasourceType,
    RefreshInterval,
    GrafanaSettings,
    PanelThreshold,
    PanelTarget,
    Panel,
    DashboardRow,
    Dashboard,
    SolaceDashboards,
    GrafanaDashboardGenerator,
    create_grafana_dashboards,
)


class TestPanelType:
    """Tests for PanelType enum."""

    def test_graph_value(self) -> None:
        """Test graph panel type."""
        assert PanelType.GRAPH.value == "graph"

    def test_timeseries_value(self) -> None:
        """Test timeseries panel type."""
        assert PanelType.TIMESERIES.value == "timeseries"

    def test_stat_value(self) -> None:
        """Test stat panel type."""
        assert PanelType.STAT.value == "stat"

    def test_gauge_value(self) -> None:
        """Test gauge panel type."""
        assert PanelType.GAUGE.value == "gauge"


class TestDatasourceType:
    """Tests for DatasourceType enum."""

    def test_prometheus_value(self) -> None:
        """Test prometheus datasource."""
        assert DatasourceType.PROMETHEUS.value == "prometheus"

    def test_loki_value(self) -> None:
        """Test loki datasource."""
        assert DatasourceType.LOKI.value == "loki"


class TestRefreshInterval:
    """Tests for RefreshInterval enum."""

    def test_five_sec_value(self) -> None:
        """Test 5 second refresh."""
        assert RefreshInterval.FIVE_SEC.value == "5s"

    def test_one_min_value(self) -> None:
        """Test 1 minute refresh."""
        assert RefreshInterval.ONE_MIN.value == "1m"


class TestGrafanaSettings:
    """Tests for GrafanaSettings."""

    def test_default_org_id(self) -> None:
        """Test default org ID."""
        settings = GrafanaSettings()
        assert settings.org_id == 1

    def test_default_folder_name(self) -> None:
        """Test default folder name."""
        settings = GrafanaSettings()
        assert settings.folder_name == "Solace-AI"

    def test_default_refresh(self) -> None:
        """Test default refresh interval."""
        settings = GrafanaSettings()
        assert settings.default_refresh == RefreshInterval.TEN_SEC


class TestPanelThreshold:
    """Tests for PanelThreshold dataclass."""

    def test_threshold_creation(self) -> None:
        """Test creating a threshold."""
        threshold = PanelThreshold(value=80.0, color="yellow")
        assert threshold.value == 80.0
        assert threshold.color == "yellow"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        threshold = PanelThreshold(value=80.0, color="yellow", op="lt")
        result = threshold.to_dict()
        assert result["value"] == 80.0
        assert result["color"] == "yellow"
        assert result["op"] == "lt"


class TestPanelTarget:
    """Tests for PanelTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test creating a target."""
        target = PanelTarget(expr="up{job='test'}", legend_format="{{instance}}")
        assert target.expr == "up{job='test'}"
        assert target.legend_format == "{{instance}}"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        target = PanelTarget(expr="up{job='test'}")
        result = target.to_dict()
        assert result["expr"] == "up{job='test'}"
        assert result["refId"] == "A"


class TestPanel:
    """Tests for Panel dataclass."""

    def test_panel_creation(self) -> None:
        """Test creating a panel."""
        target = PanelTarget(expr="up")
        panel = Panel("Test Panel", PanelType.STAT, [target], {"x": 0, "y": 0, "w": 6, "h": 4})
        assert panel.title == "Test Panel"
        assert panel.panel_type == PanelType.STAT

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        target = PanelTarget(expr="up")
        panel = Panel("Test Panel", PanelType.STAT, [target], {"x": 0, "y": 0, "w": 6, "h": 4})
        result = panel.to_dict(1)
        assert result["id"] == 1
        assert result["title"] == "Test Panel"
        assert result["type"] == "stat"


class TestDashboard:
    """Tests for Dashboard dataclass."""

    def test_dashboard_creation(self) -> None:
        """Test creating a dashboard."""
        dashboard = Dashboard(uid="test", title="Test Dashboard", description="Test")
        assert dashboard.uid == "test"
        assert dashboard.title == "Test Dashboard"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        dashboard = Dashboard(uid="test", title="Test Dashboard", description="Test",
                             tags=["test"], refresh=RefreshInterval.FIVE_SEC)
        result = dashboard.to_dict()
        assert result["uid"] == "test"
        assert result["title"] == "Test Dashboard"
        assert result["refresh"] == "5s"


class TestSolaceDashboards:
    """Tests for SolaceDashboards factory."""

    def test_system_overview(self) -> None:
        """Test system overview dashboard."""
        dashboard = SolaceDashboards.system_overview()
        assert dashboard.uid == "solace-overview"
        assert "overview" in dashboard.tags

    def test_safety_service(self) -> None:
        """Test safety service dashboard."""
        dashboard = SolaceDashboards.safety_service()
        assert dashboard.uid == "solace-safety"
        assert "critical" in dashboard.tags

    def test_memory_service(self) -> None:
        """Test memory service dashboard."""
        dashboard = SolaceDashboards.memory_service()
        assert dashboard.uid == "solace-memory"

    def test_llm_inference(self) -> None:
        """Test LLM inference dashboard."""
        dashboard = SolaceDashboards.llm_inference()
        assert dashboard.uid == "solace-llm"

    def test_all_dashboards(self) -> None:
        """Test all_dashboards returns complete list."""
        dashboards = SolaceDashboards.all_dashboards()
        assert len(dashboards) == 4
        uids = [d.uid for d in dashboards]
        assert "solace-overview" in uids
        assert "solace-safety" in uids


class TestGrafanaDashboardGenerator:
    """Tests for GrafanaDashboardGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be initialized."""
        generator = GrafanaDashboardGenerator()
        assert generator is not None

    def test_export_dashboard(self) -> None:
        """Test export single dashboard."""
        generator = GrafanaDashboardGenerator()
        dashboard = SolaceDashboards.system_overview()
        result = generator.export_dashboard(dashboard)
        assert "dashboard" in result
        assert result["overwrite"] is True

    def test_export_all(self) -> None:
        """Test export all dashboards."""
        generator = GrafanaDashboardGenerator()
        result = generator.export_all()
        assert len(result) == 4


class TestCreateGrafanaDashboards:
    """Tests for create_grafana_dashboards factory."""

    def test_create_returns_list(self) -> None:
        """Test factory returns list."""
        dashboards = create_grafana_dashboards()
        assert isinstance(dashboards, list)

    def test_create_returns_valid_dashboards(self) -> None:
        """Test factory returns valid dashboard configs."""
        dashboards = create_grafana_dashboards()
        assert len(dashboards) > 0
        for d in dashboards:
            assert "dashboard" in d
