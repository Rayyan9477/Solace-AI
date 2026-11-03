"""
Phase C config-secrets cluster - structural contract tests.

These lock in the compose / prod-override / .dockerignore / .env hardening for findings
P1-8, P1-9, P2-12, P2-15, P2-16. They are pure file-content assertions (no Docker needed);
real behaviour (fail-fast on a missing secret, per-service ``settings.environment`` binding
under real containers) is Stage-3 / staging verified. The exact bind of each service's
runtime settings class is proven by the per-service tests under each service's tests/ dir.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "docker-compose.yml").exists():
            return parent
    raise AssertionError("docker-compose.yml not found walking up from test file")


REPO = _repo_root()
BASE = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
PROD = (REPO / "docker-compose.prod.yml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P1-9 - base compose env-var names must MATCH each service's runtime settings
#        env_prefix, and must be overridable (${VAR:-development}).
# ---------------------------------------------------------------------------

def test_p1_9_memory_environment_var_is_correct_and_overridable():
    # runtime gate: MemoryServiceAppSettings(env_prefix="MEMORY_").environment
    assert "MEMORY_ENVIRONMENT: ${ENVIRONMENT:-development}" in BASE
    # the old name bound nothing and must be gone
    assert "MEMORY_SERVICE_ENV:" not in BASE


def test_p1_9_notification_env_var_is_correct_and_overridable():
    # runtime gate: ServiceConfig(env_prefix="NOTIFICATION_SERVICE_").env
    assert "NOTIFICATION_SERVICE_ENV: ${ENVIRONMENT:-development}" in BASE
    # must no longer be a bare, non-overridable literal
    assert "NOTIFICATION_SERVICE_ENV: development" not in BASE


def test_p1_9_analytics_env_var_is_correct_and_overridable():
    # runtime gate: ServiceConfig(env_prefix="ANALYTICS_SERVICE_").env
    assert "ANALYTICS_SERVICE_ENV: ${ENVIRONMENT:-development}" in BASE
    assert "ANALYTICS_SERVICE_ENV: development" not in BASE


def test_p1_9_prod_pins_the_three_gates_to_global_environment():
    # prod override wires each gate to the global ENVIRONMENT so a single
    # ENVIRONMENT=production in .env.prod drives all three (defence in depth vs a
    # forgotten per-service var), with a fail-fast :? guard.
    assert "MEMORY_ENVIRONMENT: ${ENVIRONMENT:?" in PROD
    assert "NOTIFICATION_SERVICE_ENV: ${ENVIRONMENT:?" in PROD
    assert "ANALYTICS_SERVICE_ENV: ${ENVIRONMENT:?" in PROD


# ---------------------------------------------------------------------------
# P1-8 - prod compose must set/enforce the global ENVIRONMENT and AUDIT_HMAC_KEY
#        with fail-fast ${VAR:?} guards.
# ---------------------------------------------------------------------------

def test_p1_8_global_environment_guarded_in_prod():
    assert "ENVIRONMENT: ${ENVIRONMENT:?" in PROD


def test_p1_8_audit_hmac_key_guarded_in_prod():
    assert "AUDIT_HMAC_KEY: ${AUDIT_HMAC_KEY:?" in PROD


# ---------------------------------------------------------------------------
# P2-12 - orchestrator behind Caddy needs ORCHESTRATOR_TRUSTED_PROXY_HOPS=1.
# ---------------------------------------------------------------------------

def test_p2_12_trusted_proxy_hops_set_in_prod():
    assert "ORCHESTRATOR_TRUSTED_PROXY_HOPS:" in PROD
    assert '"1"' in PROD or "ORCHESTRATOR_TRUSTED_PROXY_HOPS: 1" in PROD


# ---------------------------------------------------------------------------
# P2-16 - Weaviate must fail the prod deploy on a missing/default API key.
# ---------------------------------------------------------------------------

def test_p2_16_weaviate_api_key_guarded_in_prod():
    assert "WEAVIATE_API_KEY: ${WEAVIATE_API_KEY:?" in PROD


# ---------------------------------------------------------------------------
# P2-15 - .dockerignore excludes .env from all build contexts; .env has DEBUG=false.
# ---------------------------------------------------------------------------

def test_p2_15_dockerignore_excludes_env():
    dockerignore = REPO / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore must exist at repo root (single shared build context)"
    lines = {ln.strip() for ln in dockerignore.read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines


def test_p2_15_env_debug_disabled():
    env_text = (REPO / ".env").read_text(encoding="utf-8")
    # never assert on env_text directly: a failure would print the .env file
    ok = "DEBUG=false" in env_text and "DEBUG=true" not in env_text
    assert ok, "DEBUG must be false in .env"
