"""Solace-AI Grafana Dashboards - Dashboard panels, templates, and visualizations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class PanelType(str, Enum):
    """Grafana panel visualization types."""
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    LOGS = "logs"
    ALERTLIST = "alertlist"
    TEXT = "text"


class DatasourceType(str, Enum):
    """Supported Grafana datasources."""
    PROMETHEUS = "prometheus"
    LOKI = "loki"
    JAEGER = "jaeger"
    ELASTICSEARCH = "elasticsearch"


class RefreshInterval(str, Enum):
    """Dashboard refresh intervals."""
    OFF = ""
    FIVE_SEC = "5s"
    TEN_SEC = "10s"
    THIRTY_SEC = "30s"
    ONE_MIN = "1m"
    FIVE_MIN = "5m"


class GrafanaSettings(BaseSettings):
    """Grafana configuration from environment."""
    org_id: int = Field(default=1)
    folder_name: str = Field(default="Solace-AI")
    datasource_prometheus: str = Field(default="Prometheus")
    datasource_loki: str = Field(default="Loki")
    datasource_jaeger: str = Field(default="Jaeger")
    default_refresh: RefreshInterval = Field(default=RefreshInterval.TEN_SEC)
    time_from: str = Field(default="now-1h")
    time_to: str = Field(default="now")
    model_config = SettingsConfigDict(env_prefix="GRAFANA_", env_file=".env", extra="ignore")


@dataclass
class PanelThreshold:
    """Threshold configuration for panels."""
    value: float
    color: str
    op: str = "gt"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "color": self.color, "op": self.op}


@dataclass
class PanelTarget:
    """Prometheus query target for a panel."""
    expr: str
    legend_format: str = ""
    ref_id: str = "A"
    interval: str = ""
    instant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"expr": self.expr, "legendFormat": self.legend_format, "refId": self.ref_id,
                "interval": self.interval, "instant": self.instant}


@dataclass
class Panel:
    """Grafana panel configuration."""
    title: str
    panel_type: PanelType
    targets: list[PanelTarget]
    grid_pos: dict[str, int]
    thresholds: list[PanelThreshold] = field(default_factory=list)
    unit: str = ""
    description: str = ""
    datasource: str = "Prometheus"

    def to_dict(self, panel_id: int) -> dict[str, Any]:
        config: dict[str, Any] = {
            "id": panel_id, "title": self.title, "type": self.panel_type.value,
            "gridPos": self.grid_pos, "datasource": self.datasource,
            "targets": [t.to_dict() for t in self.targets],
        }
        if self.description:
            config["description"] = self.description
        if self.thresholds:
            config["fieldConfig"] = {"defaults": {"thresholds": {"steps": [t.to_dict() for t in self.thresholds]}}}
        if self.unit:
            config.setdefault("fieldConfig", {}).setdefault("defaults", {})["unit"] = self.unit
        return config


@dataclass
class DashboardRow:
    """Dashboard row for organizing panels."""
    title: str
    collapsed: bool = False
    panels: list[Panel] = field(default_factory=list)


@dataclass
class Dashboard:
    """Complete Grafana dashboard configuration."""
    uid: str
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=list)
    rows: list[DashboardRow] = field(default_factory=list)
    refresh: RefreshInterval = RefreshInterval.TEN_SEC
    time_from: str = "now-1h"
    time_to: str = "now"

    def to_dict(self) -> dict[str, Any]:
        panel_list = []
        panel_id = 1
        for panel in self.panels:
            panel_list.append(panel.to_dict(panel_id))
            panel_id += 1
        for row in self.rows:
            for panel in row.panels:
                panel_list.append(panel.to_dict(panel_id))
                panel_id += 1
        return {
            "uid": self.uid, "title": self.title, "description": self.description,
            "tags": self.tags, "panels": panel_list, "refresh": self.refresh.value,
            "time": {"from": self.time_from, "to": self.time_to}, "schemaVersion": 39,
        }


class SolaceDashboards:
    """Factory for Solace-AI Grafana dashboard definitions."""

    @staticmethod
    def _gp(x: int, y: int, w: int = 12, h: int = 8) -> dict[str, int]:
        return {"x": x, "y": y, "w": w, "h": h}

    @staticmethod
    def system_overview() -> Dashboard:
        """System-wide health and performance dashboard."""
        return Dashboard(
            uid="solace-overview", title="Solace-AI System Overview",
            description="Overall system health, latency, and throughput metrics",
            tags=["solace", "overview", "system"],
            panels=[
                Panel("Active Sessions", PanelType.STAT, [PanelTarget("sum(solace_active_sessions)", "Sessions")],
                      SolaceDashboards._gp(0, 0, 6, 4)),
                Panel("Request Rate", PanelType.STAT,
                      [PanelTarget("sum(rate(solace_requests_total[5m]))", "req/s")],
                      SolaceDashboards._gp(6, 0, 6, 4), unit="reqps"),
                Panel("Error Rate", PanelType.STAT,
                      [PanelTarget("sum(rate(solace_errors_total[5m]))/sum(rate(solace_requests_total[5m]))*100", "%")],
                      SolaceDashboards._gp(12, 0, 6, 4), unit="percent",
                      thresholds=[PanelThreshold(0, "green"), PanelThreshold(1, "yellow"), PanelThreshold(5, "red")]),
                Panel("P99 Latency", PanelType.STAT,
                      [PanelTarget("histogram_quantile(0.99, sum(rate(solace_request_duration_bucket[5m])) by (le))", "p99")],
                      SolaceDashboards._gp(18, 0, 6, 4), unit="ms"),
                Panel("Request Latency Distribution", PanelType.TIMESERIES,
                      [PanelTarget("histogram_quantile(0.50, sum(rate(solace_request_duration_bucket[5m])) by (le))", "p50"),
                       PanelTarget("histogram_quantile(0.95, sum(rate(solace_request_duration_bucket[5m])) by (le))", "p95"),
                       PanelTarget("histogram_quantile(0.99, sum(rate(solace_request_duration_bucket[5m])) by (le))", "p99")],
                      SolaceDashboards._gp(0, 4, 24, 8), unit="ms"),
            ],
        )

    @staticmethod
    def safety_service() -> Dashboard:
        """Safety service monitoring dashboard - CRITICAL."""
        return Dashboard(
            uid="solace-safety", title="Solace-AI Safety Service",
            description="Crisis detection, escalation, and safety monitoring - CRITICAL SERVICE",
            tags=["solace", "safety", "critical"],
            panels=[
                Panel("Crisis Detections (24h)", PanelType.STAT,
                      [PanelTarget("sum(increase(solace_crisis_detections_total[24h]))", "Detections")],
                      SolaceDashboards._gp(0, 0, 6, 4)),
                Panel("Detection Latency (p99)", PanelType.STAT,
                      [PanelTarget("histogram_quantile(0.99, sum(rate(solace_crisis_detection_duration_bucket[5m])) by (le))", "p99")],
                      SolaceDashboards._gp(6, 0, 6, 4), unit="ms",
                      thresholds=[PanelThreshold(0, "green"), PanelThreshold(10, "yellow"), PanelThreshold(50, "red")]),
                Panel("Escalation Rate", PanelType.STAT,
                      [PanelTarget("sum(rate(solace_escalations_total[1h]))*3600", "per hour")],
                      SolaceDashboards._gp(12, 0, 6, 4)),
                Panel("Safety Check Success Rate", PanelType.GAUGE,
                      [PanelTarget("sum(rate(solace_safety_checks_success[5m]))/sum(rate(solace_safety_checks_total[5m]))*100", "%")],
                      SolaceDashboards._gp(18, 0, 6, 4), unit="percent",
                      thresholds=[PanelThreshold(0, "red"), PanelThreshold(95, "yellow"), PanelThreshold(99, "green")]),
                Panel("Crisis Detection Over Time", PanelType.TIMESERIES,
                      [PanelTarget("sum(rate(solace_crisis_detections_total[5m])) by (severity)", "{{severity}}")],
                      SolaceDashboards._gp(0, 4, 24, 8)),
            ],
        )

    @staticmethod
    def memory_service() -> Dashboard:
        """Memory service monitoring dashboard."""
        return Dashboard(
            uid="solace-memory", title="Solace-AI Memory Service",
            description="Context assembly, memory tiers, and retrieval performance",
            tags=["solace", "memory", "context"],
            panels=[
                Panel("Context Assembly Time (p99)", PanelType.STAT,
                      [PanelTarget("histogram_quantile(0.99, sum(rate(solace_context_assembly_duration_bucket[5m])) by (le))", "p99")],
                      SolaceDashboards._gp(0, 0, 6, 4), unit="ms"),
                Panel("Cache Hit Rate", PanelType.GAUGE,
                      [PanelTarget("sum(rate(solace_cache_hits[5m]))/(sum(rate(solace_cache_hits[5m]))+sum(rate(solace_cache_misses[5m])))*100", "%")],
                      SolaceDashboards._gp(6, 0, 6, 4), unit="percent",
                      thresholds=[PanelThreshold(0, "red"), PanelThreshold(80, "yellow"), PanelThreshold(95, "green")]),
                Panel("Memory Tier Distribution", PanelType.TIMESERIES,
                      [PanelTarget("sum(solace_memory_entries) by (tier)", "{{tier}}")],
                      SolaceDashboards._gp(0, 4, 12, 8)),
                Panel("Retrieval Latency by Tier", PanelType.TIMESERIES,
                      [PanelTarget("histogram_quantile(0.95, sum(rate(solace_memory_retrieval_duration_bucket[5m])) by (tier, le))", "{{tier}}")],
                      SolaceDashboards._gp(12, 4, 12, 8), unit="ms"),
            ],
        )

    @staticmethod
    def llm_inference() -> Dashboard:
        """LLM inference monitoring dashboard."""
        return Dashboard(
            uid="solace-llm", title="Solace-AI LLM Inference",
            description="LLM provider latency, token usage, and costs",
            tags=["solace", "llm", "inference"],
            panels=[
                Panel("Inference Latency (p99)", PanelType.STAT,
                      [PanelTarget("histogram_quantile(0.99, sum(rate(solace_llm_inference_duration_bucket[5m])) by (le))", "p99")],
                      SolaceDashboards._gp(0, 0, 6, 4), unit="s"),
                Panel("Tokens/Second", PanelType.STAT,
                      [PanelTarget("sum(rate(solace_llm_tokens_total[5m]))", "tokens/s")],
                      SolaceDashboards._gp(6, 0, 6, 4)),
                Panel("Provider Error Rate", PanelType.TIMESERIES,
                      [PanelTarget("sum(rate(solace_llm_errors_total[5m])) by (provider)", "{{provider}}")],
                      SolaceDashboards._gp(0, 4, 12, 8)),
                Panel("Token Usage by Model", PanelType.TIMESERIES,
                      [PanelTarget("sum(rate(solace_llm_tokens_total[5m])) by (model)", "{{model}}")],
                      SolaceDashboards._gp(12, 4, 12, 8)),
            ],
        )

    @classmethod
    def all_dashboards(cls) -> list[Dashboard]:
        """All Solace-AI dashboards."""
        return [cls.system_overview(), cls.safety_service(), cls.memory_service(), cls.llm_inference()]


class GrafanaDashboardGenerator:
    """Generates Grafana dashboard configurations."""

    def __init__(self, settings: GrafanaSettings | None = None) -> None:
        self._settings = settings or GrafanaSettings()

    def export_dashboard(self, dashboard: Dashboard) -> dict[str, Any]:
        """Export dashboard to Grafana API format."""
        config = dashboard.to_dict()
        return {"dashboard": config, "folderId": 0, "folderUid": "", "message": "Auto-generated by Solace-AI",
                "overwrite": True}

    def export_all(self, dashboards: list[Dashboard] | None = None) -> list[dict[str, Any]]:
        """Export all dashboards."""
        dashboards = dashboards or SolaceDashboards.all_dashboards()
        result = [self.export_dashboard(d) for d in dashboards]
        logger.info("grafana_dashboards_generated", count=len(result))
        return result


def create_grafana_dashboards(settings: GrafanaSettings | None = None) -> list[dict[str, Any]]:
    """Create all Solace-AI Grafana dashboards."""
    generator = GrafanaDashboardGenerator(settings)
    return generator.export_all()
