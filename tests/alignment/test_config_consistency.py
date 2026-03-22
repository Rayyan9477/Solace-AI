"""Alignment tests for configuration consistency across the project.

Verifies that required infrastructure files, Dockerfiles, CI workflows,
and environment configuration are present and properly structured.
"""

from pathlib import Path

import pytest

# Project root is two levels up from this test file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All ten services in the Solace-AI platform
SERVICE_DIRS = [
    "services/safety_service",
    "services/diagnosis_service",
    "services/therapy_service",
    "services/memory_service",
    "services/personality_service",
    "services/orchestrator_service",
    "services/notification-service",
    "services/user-service",
    "services/analytics-service",
    "services/config_service",
]


class TestConfigConsistency:
    """Tests that project configuration files are present and consistent."""

    def test_all_dockerfiles_exist(self) -> None:
        """Verify every service directory contains a Dockerfile.

        Each of the 10 Solace-AI services must have its own Dockerfile
        for containerized deployment.
        """
        missing = []
        for service_dir in SERVICE_DIRS:
            dockerfile = PROJECT_ROOT / service_dir / "Dockerfile"
            if not dockerfile.exists():
                missing.append(service_dir)

        assert not missing, (
            f"Missing Dockerfile in service(s): {missing}"
        )

    def test_infrastructure_dirs_exist(self) -> None:
        """Verify required infrastructure directories are present.

        The deployment stack requires init-db scripts, Prometheus config,
        and Grafana dashboards.
        """
        required_dirs = [
            "infrastructure/init-db",
            "infrastructure/prometheus",
            "infrastructure/grafana",
        ]

        missing = []
        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            if not full_path.is_dir():
                missing.append(dir_path)

        assert not missing, (
            f"Missing infrastructure directories: {missing}"
        )

    def test_alembic_ini_exists(self) -> None:
        """Verify alembic.ini exists at the project root.

        Alembic is used for database migration management across all
        services sharing a PostgreSQL backend.
        """
        alembic_ini = PROJECT_ROOT / "alembic.ini"
        assert alembic_ini.exists(), (
            f"alembic.ini not found at {alembic_ini}"
        )

    def test_ci_workflow_exists(self) -> None:
        """Verify .github/workflows/ci.yml exists.

        The CI pipeline must be defined for automated testing and quality
        checks on pull requests.
        """
        ci_workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_workflow.exists(), (
            f"CI workflow not found at {ci_workflow}"
        )

    def test_env_example_has_fernet_keys(self) -> None:
        """Verify .env.example contains FERNET_TOKEN_KEY and FERNET_FIELD_KEY.

        These encryption keys are required for PHI field-level encryption
        and token encryption. Their presence in .env.example ensures
        developers know to configure them.
        """
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), (
            f".env.example not found at {env_example}"
        )

        content = env_example.read_text(encoding="utf-8")

        assert "FERNET_TOKEN_KEY" in content, (
            "FERNET_TOKEN_KEY not found in .env.example"
        )
        assert "FERNET_FIELD_KEY" in content, (
            "FERNET_FIELD_KEY not found in .env.example"
        )
