# tests/bootstrap/test_mvp_036_prometheus_scrape.py
"""MVP-036: Prometheus scrapes the nine MVP services on their real ports and nothing that has no exporter."""
import subprocess
from pathlib import Path
import pytest, yaml

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {"orchestrator-service:8000": "orchestrator-service", "user-service:8001": "user-service",
            "safety-service:8002": "safety-service", "notification-service:8003": "notification-service",
            "diagnosis-service:8004": "diagnosis-service", "memory-service:8005": "memory-service",
            "therapy-service:8006": "therapy-service", "personality-service:8007": "personality-service",
            "analytics-service:8009": "analytics-service"}

def _cfg():
    return yaml.safe_load((REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8"))

def test_one_job_scrapes_the_nine_services():
    jobs = {j["job_name"]: j for j in _cfg()["scrape_configs"]}
    job = jobs["solace-services"]
    targets = {t: sc["labels"]["service"] for sc in job["static_configs"] for t in sc["targets"]}
    assert targets == EXPECTED
    assert job.get("metrics_path", "/metrics") == "/metrics"

def test_no_dead_jobs():
    names = {j["job_name"] for j in _cfg()["scrape_configs"]}
    assert names.isdisjoint({"config-service", "postgres", "redis", "kafka"})
    assert "8010" not in (REPO / "infrastructure/prometheus/prometheus.yml").read_text(encoding="utf-8")

@pytest.mark.integration
@pytest.mark.realinfra
def test_promtool_accepts_the_config():
    out = subprocess.run(["docker", "run", "--rm", "-v", f"{REPO / 'infrastructure' / 'prometheus'}:/etc/prometheus:ro",
                          "--entrypoint", "promtool", "prom/prometheus:v2.47.0", "check", "config", "/etc/prometheus/prometheus.yml"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
