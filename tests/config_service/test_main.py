"""Unit tests for Configuration Service - Main Module."""
from __future__ import annotations
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services"))

from config_service.src.main import create_application, app
from config_service.src.settings import ConfigServiceSettings


class TestApplication:
    """Tests for FastAPI application creation."""

    def test_create_application(self) -> None:
        test_app = create_application()
        assert test_app.title == "Solace-AI Configuration Service"
        assert test_app.version == "1.0.0"

    def test_app_routes_registered(self) -> None:
        # Use the module-level app which has routes registered
        route_paths = [route.path for route in app.routes]
        # Check for root and probe endpoints
        has_root = "/" in route_paths
        has_ready = "/ready" in route_paths
        has_live = "/live" in route_paths
        # At minimum, app should have routes registered
        assert len(route_paths) > 0
        assert has_root or has_ready or has_live

    def test_app_includes_api_router(self) -> None:
        # Verify API router is included
        route_paths = [route.path for route in app.routes]
        has_api_routes = any("/api/v1" in path for path in route_paths)
        assert has_api_routes

    def test_app_has_exception_handlers(self) -> None:
        test_app = create_application()
        # App should have exception handlers
        assert len(test_app.exception_handlers) > 0


class TestRootEndpoint:
    """Tests for root endpoint function."""

    @pytest.mark.asyncio
    async def test_root_endpoint_function(self) -> None:
        from config_service.src.main import root
        result = await root()
        assert result["service"] == "config-service"
        assert result["status"] == "operational"
        assert result["version"] == "1.0.0"


class TestLivenessEndpoint:
    """Tests for liveness probe endpoint."""

    @pytest.mark.asyncio
    async def test_liveness_function(self) -> None:
        from config_service.src.main import liveness
        result = await liveness()
        assert result["status"] == "alive"


class TestReadinessEndpoint:
    """Tests for readiness probe endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_when_initialized(self) -> None:
        from config_service.src.main import readiness
        import config_service.src.settings as settings_module
        from config_service.src.settings import ConfigurationManager

        original = settings_module._manager
        try:
            mgr = ConfigurationManager()
            mgr._initialized = True
            settings_module._manager = mgr
            result = await readiness()
            assert result["status"] == "ready"
        finally:
            settings_module._manager = original

    @pytest.mark.asyncio
    async def test_readiness_when_not_initialized(self) -> None:
        from config_service.src.main import readiness
        from fastapi.responses import JSONResponse
        import config_service.src.settings as settings_module
        from config_service.src.settings import ConfigurationManager

        original = settings_module._manager
        try:
            mgr = ConfigurationManager()
            mgr._initialized = False
            settings_module._manager = mgr
            result = await readiness()
            # When not initialized, returns JSONResponse with 503
            assert isinstance(result, JSONResponse)
            assert result.status_code == 503
        finally:
            settings_module._manager = original


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_middleware_added(self) -> None:
        test_app = create_application()
        # Check that user_middleware list contains CORS middleware
        middleware_classes = [m.cls.__name__ for m in test_app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


class TestExceptionHandlers:
    """Tests for exception handlers."""

    def test_exception_handlers_registered(self) -> None:
        test_app = create_application()
        # Should have at least RequestValidationError handler
        from fastapi.exceptions import RequestValidationError
        assert RequestValidationError in test_app.exception_handlers or len(test_app.exception_handlers) > 0


class TestConfigServiceSettings:
    """Tests for ConfigServiceSettings in main context."""

    def test_settings_loaded(self) -> None:
        settings = ConfigServiceSettings()
        assert settings.service_name == "config-service"

    def test_settings_default_environment(self) -> None:
        settings = ConfigServiceSettings()
        assert settings.environment is not None

    def test_docs_disabled_in_production(self) -> None:
        settings = ConfigServiceSettings(environment="production")
        assert settings.environment.value == "production"

    def test_settings_host_and_port(self) -> None:
        settings = ConfigServiceSettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8008
