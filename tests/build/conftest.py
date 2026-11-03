"""Shared fixtures for the tests/build suite (build, image and clean-checkout checks)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def require_docker() -> None:
    """Skip the test when Docker is unreachable; fail instead when SOLACE_REQUIRE_REALINFRA=1."""
    try:
        import docker  # the docker SDK ships with testcontainers

        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:  # any failure (SDK missing, daemon down, bad DOCKER_HOST) means unreachable
        reason = f"Docker is unreachable: {type(exc).__name__}: {exc}"
        if os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1":
            pytest.fail(f"SOLACE_REQUIRE_REALINFRA=1 but {reason}", pytrace=False)
        pytest.skip(reason)
