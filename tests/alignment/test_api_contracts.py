"""Alignment tests for API contracts between callers and services.

Verifies that HTTP client defaults, URL patterns, and port numbers are
consistent between the orchestrator's service clients and the actual
service implementations they target.
"""

import importlib
import inspect
import sys
from pathlib import Path

import pytest


class TestApiContractAlignment:
    """Tests that API contracts match between callers and services."""

    def test_safety_check_type_lowercase(self) -> None:
        """Verify the orchestrator SafetyServiceClient's default check_type
        is the lowercase string 'full_assessment'.

        The safety service expects lowercase check type identifiers; using
        an uppercase value would cause a validation mismatch.
        """
        from services.orchestrator_service.src.infrastructure.clients import (
            SafetyServiceClient,
        )

        sig = inspect.signature(SafetyServiceClient.check_safety)
        check_type_param = sig.parameters.get("check_type")

        assert check_type_param is not None, (
            "SafetyServiceClient.check_safety is missing 'check_type' parameter"
        )
        assert check_type_param.default == "full_assessment", (
            f"Expected default check_type='full_assessment', "
            f"got '{check_type_param.default}'"
        )
        assert check_type_param.default == check_type_param.default.lower(), (
            f"check_type default must be lowercase, got '{check_type_param.default}'"
        )

    def test_personality_agent_urls_include_prefix(self) -> None:
        """Verify personality agent URL paths contain '/api/v1/personality/'.

        This ensures the orchestrator's personality service client targets
        the correct versioned API prefix.
        """
        from services.orchestrator_service.src.agents.personality_agent import (
            PersonalityServiceClient,
            PersonalityAgentSettings,
        )

        # Inspect the source code of the client methods that build URLs
        detect_source = inspect.getsource(PersonalityServiceClient.detect_personality)
        get_style_source = inspect.getsource(PersonalityServiceClient.get_style)

        assert "/api/v1/personality/" in detect_source, (
            "PersonalityServiceClient.detect_personality URL does not contain "
            "'/api/v1/personality/' prefix"
        )
        assert "/api/v1/personality/" in get_style_source, (
            "PersonalityServiceClient.get_style URL does not contain "
            "'/api/v1/personality/' prefix"
        )

    def test_notification_service_user_url_port(self) -> None:
        """Verify the notification consumer's user_service_url default
        points to port 8001.

        The user service runs on port 8001; a misconfigured default would
        cause inter-service communication failures.

        Uses AST parsing to avoid importing the full notification-service
        module tree (which has optional dependencies like aiosmtplib).
        """
        import ast

        project_root = Path(__file__).resolve().parent.parent.parent
        consumers_path = (
            project_root / "services" / "notification-service" / "src" / "consumers.py"
        )

        assert consumers_path.exists(), (
            f"notification-service consumers.py not found at {consumers_path}"
        )

        source = consumers_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the UserServiceSettings class and extract the
        # user_service_url default value.
        default_url = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "UserServiceSettings":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        if item.target.id == "user_service_url" and item.value is not None:
                            # The default is inside Field(default=...)
                            if isinstance(item.value, ast.Call):
                                for kw in item.value.keywords:
                                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                                        default_url = kw.value.value

        assert default_url is not None, (
            "Could not find user_service_url default in UserServiceSettings"
        )
        assert ":8001" in default_url, (
            f"Expected user_service_url to include port 8001, "
            f"got '{default_url}'"
        )
        assert default_url == "http://localhost:8001", (
            f"Expected default user_service_url='http://localhost:8001', "
            f"got '{default_url}'"
        )
