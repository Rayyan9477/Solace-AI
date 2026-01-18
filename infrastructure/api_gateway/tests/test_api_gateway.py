"""
Unit tests for Solace-AI API Gateway components.
Tests Kong configuration, routes, rate limiting, JWT auth, and CORS.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
from datetime import datetime, timezone, timedelta
import time

from infrastructure.api_gateway.kong_config import (
    KongSettings, KongConfig, ServiceConfig, UpstreamConfig, UpstreamTarget,
    HealthCheckConfig, PluginConfig, LoadBalancingAlgorithm, HealthCheckType,
    create_solace_gateway_config,
)
from infrastructure.api_gateway.routes import (
    RouteSettings, RouteConfig, RouteDefinition, RouteGroup, RouteManager,
    HttpMethod, RouteProtocol, ServiceRoutes, create_solace_route_config,
)
from infrastructure.api_gateway.rate_limiting import (
    RateLimitConfig, RateLimitPolicy, RateLimiter, RateLimitResult,
    RateLimitWindow, RateLimitScope, RateLimitStore,
    create_solace_rate_limiter,
)
from infrastructure.api_gateway.auth_plugin import (
    JWTConfig, JWTAuthPlugin, TokenClaims, AuthResult, TokenType, UserRole,
    JWTCodec, create_solace_auth_plugin,
)
from infrastructure.api_gateway.cors import (
    CORSConfig, CORSPolicy, CORSHandler, CORSRequest, CORSResponse,
    CORSPreset, create_solace_cors_handler,
)


class TestKongConfig:
    def test_kong_settings_defaults(self):
        settings = KongSettings()
        assert settings.admin_url == "http://localhost:8001"
        assert settings.timeout_seconds == 10.0
        assert settings.max_retries == 3

    def test_kong_settings_url_validation(self):
        with pytest.raises(ValueError, match="must start with http"):
            KongSettings(admin_url="invalid-url")

    def test_service_config_to_kong_format(self):
        config = ServiceConfig(name="test-service", host="test-host", port=8080, protocol="http", path="/api")
        result = config.to_kong_format()
        assert result["name"] == "test-service"
        assert result["host"] == "test-host"
        assert result["port"] == 8080
        assert result["protocol"] == "http"
        assert result["path"] == "/api"

    def test_upstream_config_with_targets(self):
        config = UpstreamConfig(name="test-upstream", algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
        config.targets.append(UpstreamTarget(target="service1:8001", weight=100))
        config.targets.append(UpstreamTarget(target="service2:8001", weight=50))
        result = config.to_kong_format()
        assert result["name"] == "test-upstream"
        assert result["algorithm"] == "round-robin"
        assert len(config.targets) == 2

    def test_health_check_config(self):
        config = HealthCheckConfig(active_enabled=True, active_http_path="/health", active_healthy_interval=10)
        result = config.to_kong_format()
        assert result["active"]["http_path"] == "/health"
        assert result["active"]["healthy"]["interval"] == 10

    def test_kong_config_define_service(self):
        config = KongConfig()
        service = config.define_service("test", "test-host", port=8080)
        assert service.name == "test"
        assert config.get_service("test") is not None

    def test_kong_config_define_upstream(self):
        config = KongConfig()
        upstream = config.define_upstream("test-upstream", LoadBalancingAlgorithm.LEAST_CONNECTIONS)
        assert upstream.name == "test-upstream"
        assert upstream.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS

    def test_kong_config_add_target(self):
        config = KongConfig()
        config.define_upstream("test-upstream")
        config.add_target_to_upstream("test-upstream", "target:8001", weight=80)
        upstream = config.get_upstream("test-upstream")
        assert len(upstream.targets) == 1
        assert upstream.targets[0].weight == 80

    def test_kong_config_export_declarative(self):
        config = KongConfig()
        config.define_service("svc1", "host1")
        config.define_upstream("up1")
        result = config.export_declarative_config()
        assert result["_format_version"] == "3.0"
        assert len(result["services"]) == 1
        assert len(result["upstreams"]) == 1

    def test_create_solace_gateway_config(self):
        config = create_solace_gateway_config()
        assert config.get_service("orchestrator-service") is not None
        assert config.get_service("user-service") is not None
        assert config.get_upstream("orchestrator-upstream") is not None


class TestRoutes:
    def test_route_settings_defaults(self):
        settings = RouteSettings()
        assert settings.admin_url == "http://localhost:8001"
        assert settings.default_strip_path is True

    def test_route_definition_to_kong_format(self):
        route = RouteDefinition(name="test-route", paths=["/api/test"], service_name="test-service", methods=[HttpMethod.GET, HttpMethod.POST])
        result = route.to_kong_format("service-123")
        assert result["name"] == "test-route"
        assert result["paths"] == ["/api/test"]
        assert "GET" in result["methods"]
        assert result["service"]["id"] == "service-123"

    def test_route_definition_matches_path(self):
        route = RouteDefinition(name="test", paths=["/api/v1/users"], service_name="test")
        assert route.matches_path("/api/v1/users") is True
        assert route.matches_path("/api/v1/users/123") is True
        assert route.matches_path("/api/v2/users") is False

    def test_route_definition_matches_method(self):
        route = RouteDefinition(name="test", paths=["/"], service_name="test", methods=[HttpMethod.GET, HttpMethod.POST])
        assert route.matches_method("GET") is True
        assert route.matches_method("POST") is True
        assert route.matches_method("DELETE") is False

    def test_route_group_add_route(self):
        group = RouteGroup(name="users", base_path="/api/v1/users", service_name="user-service")
        group.add_route("list", "", methods=[HttpMethod.GET])
        group.add_route("create", "", methods=[HttpMethod.POST])
        group.add_route("detail", "/{id}", methods=[HttpMethod.GET])
        assert len(group.routes) == 3

    def test_route_config_define_route(self):
        config = RouteConfig()
        route = config.define_route("test-route", ["/api/test"], "test-service")
        assert route.name == "test-route"
        assert config.get_route("test-route") is not None

    def test_route_config_find_matching_route(self):
        config = RouteConfig()
        config.define_route("users", ["/api/users"], "user-service", methods=[HttpMethod.GET, HttpMethod.POST])
        route = config.find_matching_route("/api/users/123", "GET")
        assert route is not None
        assert route.name == "users"

    def test_service_routes_orchestrator(self):
        group = ServiceRoutes.orchestrator_routes()
        assert group.name == "orchestrator"
        assert len(group.routes) >= 4

    def test_service_routes_user(self):
        group = ServiceRoutes.user_routes()
        assert group.name == "users"
        assert len(group.routes) >= 4

    def test_create_solace_route_config(self):
        config = create_solace_route_config()
        assert "orchestrator" in config._groups
        assert "users" in config._groups
        assert "sessions" in config._groups


class TestRateLimiting:
    def test_rate_limit_config_defaults(self):
        config = RateLimitConfig()
        assert config.default_limit == 1000
        assert config.default_window == "minute"

    def test_rate_limit_policy_window_seconds(self):
        policy = RateLimitPolicy(name="test", limit=100, window=RateLimitWindow.MINUTE)
        assert policy.window_seconds() == 60
        policy2 = RateLimitPolicy(name="test2", limit=100, window=RateLimitWindow.HOUR)
        assert policy2.window_seconds() == 3600

    def test_rate_limit_policy_to_kong_format(self):
        policy = RateLimitPolicy(name="test", limit=100, window=RateLimitWindow.MINUTE, scope=RateLimitScope.CONSUMER)
        result = policy.to_kong_plugin_config()
        assert result["minute"] == 100
        assert result["limit_by"] == "consumer"

    def test_rate_limit_result_headers(self):
        result = RateLimitResult(allowed=True, remaining=50, limit=100, reset_at=datetime.now(timezone.utc))
        headers = result.to_headers()
        assert "X-RateLimit-Limit" in headers
        assert headers["X-RateLimit-Limit"] == "100"
        assert headers["X-RateLimit-Remaining"] == "50"

    def test_rate_limit_store_increment(self):
        store = RateLimitStore()
        policy = RateLimitPolicy(name="test", limit=5, window=RateLimitWindow.MINUTE)
        for i in range(5):
            result = store.increment(policy, "user-123")
            assert result.allowed is True
        result = store.increment(policy, "user-123")
        assert result.allowed is False

    def test_rate_limiter_add_policy(self):
        limiter = RateLimiter()
        policy = RateLimitPolicy(name="test", limit=100, window=RateLimitWindow.MINUTE)
        limiter.add_policy(policy)
        assert limiter.get_policy("test") is not None

    def test_rate_limiter_check(self):
        limiter = RateLimiter()
        policy = RateLimitPolicy(name="test", limit=3, window=RateLimitWindow.MINUTE)
        limiter.add_policy(policy)
        for _ in range(3):
            result = limiter.check("test", "user-1")
            assert result.allowed is True
        result = limiter.check("test", "user-1")
        assert result.allowed is False

    def test_rate_limiter_disabled(self):
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        result = limiter.check("nonexistent", "user-1")
        assert result.allowed is True

    def test_create_solace_rate_limiter(self):
        limiter = create_solace_rate_limiter()
        assert limiter.get_policy("global-standard") is not None
        assert limiter.get_policy("orchestrator-per-user") is not None


class TestJWTAuth:
    def test_jwt_config_defaults(self):
        config = JWTConfig()
        assert config.algorithm == "HS256"
        assert config.issuer == "solace-ai"
        assert config.access_token_expire_minutes == 30

    def test_jwt_codec_base64url(self):
        data = b"test data"
        encoded = JWTCodec.base64url_encode(data)
        decoded = JWTCodec.base64url_decode(encoded)
        assert decoded == data

    def test_token_claims_to_dict(self):
        now = datetime.now(timezone.utc)
        claims = TokenClaims(sub="user-123", exp=now + timedelta(hours=1), iat=now, iss="solace-ai", aud="solace-ai-api", jti="token-123", roles=[UserRole.USER, UserRole.PREMIUM])
        result = claims.to_dict()
        assert result["sub"] == "user-123"
        assert result["iss"] == "solace-ai"
        assert "user" in result["roles"]
        assert "premium" in result["roles"]

    def test_token_claims_from_dict(self):
        now = int(datetime.now(timezone.utc).timestamp())
        data = {"sub": "user-123", "exp": now + 3600, "iat": now, "iss": "solace-ai", "aud": "solace-ai-api", "jti": "token-123", "roles": ["user"]}
        claims = TokenClaims.from_dict(data)
        assert claims.sub == "user-123"
        assert claims.has_role(UserRole.USER) is True

    def test_token_claims_is_expired(self):
        now = datetime.now(timezone.utc)
        expired_claims = TokenClaims(sub="user", exp=now - timedelta(hours=1), iat=now - timedelta(hours=2), iss="test", aud="test", jti="123")
        assert expired_claims.is_expired() is True
        valid_claims = TokenClaims(sub="user", exp=now + timedelta(hours=1), iat=now, iss="test", aud="test", jti="456")
        assert valid_claims.is_expired() is False

    def test_jwt_auth_plugin_create_token(self):
        plugin = JWTAuthPlugin()
        token = plugin.create_token(subject="user-123", roles=[UserRole.USER])
        assert token is not None
        parts = token.split(".")
        assert len(parts) == 3

    def test_jwt_auth_plugin_verify_token(self):
        plugin = JWTAuthPlugin()
        token = plugin.create_token(subject="user-123", roles=[UserRole.USER], email="test@example.com")
        result = plugin.verify_token(token)
        assert result.authenticated is True
        assert result.claims.sub == "user-123"
        assert result.claims.email == "test@example.com"

    def test_jwt_auth_plugin_invalid_token(self):
        plugin = JWTAuthPlugin()
        result = plugin.verify_token("invalid.token.here")
        assert result.authenticated is False
        assert result.error_code in ("INVALID_JSON", "VERIFICATION_FAILED")

    def test_jwt_auth_plugin_malformed_token(self):
        plugin = JWTAuthPlugin()
        result = plugin.verify_token("only.two")
        assert result.authenticated is False
        assert result.error_code == "INVALID_TOKEN"

    def test_jwt_auth_plugin_revoke_token(self):
        plugin = JWTAuthPlugin()
        token = plugin.create_token(subject="user-123")
        result = plugin.verify_token(token)
        assert result.authenticated is True
        plugin.revoke_token(result.claims.jti)
        result2 = plugin.verify_token(token)
        assert result2.authenticated is False
        assert result2.error_code == "TOKEN_REVOKED"

    def test_jwt_auth_plugin_extract_token(self):
        plugin = JWTAuthPlugin()
        headers = {"Authorization": "Bearer test-token-123"}
        token = plugin.extract_token(headers)
        assert token == "test-token-123"

    def test_jwt_auth_plugin_authorize(self):
        plugin = JWTAuthPlugin()
        claims = TokenClaims(sub="user", exp=datetime.now(timezone.utc) + timedelta(hours=1), iat=datetime.now(timezone.utc), iss="test", aud="test", jti="123", roles=[UserRole.USER])
        assert plugin.authorize(claims, [UserRole.USER]) is True
        assert plugin.authorize(claims, [UserRole.ADMIN]) is False
        admin_claims = TokenClaims(sub="admin", exp=datetime.now(timezone.utc) + timedelta(hours=1), iat=datetime.now(timezone.utc), iss="test", aud="test", jti="456", roles=[UserRole.ADMIN])
        assert plugin.authorize(admin_claims, [UserRole.USER]) is True

    def test_create_solace_auth_plugin(self):
        plugin = create_solace_auth_plugin()
        assert plugin._config.issuer == "solace-ai"


class TestCORS:
    def test_cors_config_explicit(self):
        config = CORSConfig(origins="*", credentials=True, max_age=86400)
        assert config.origins == "*"
        assert config.credentials is True
        assert config.max_age == 86400

    def test_cors_config_origins_list(self):
        config = CORSConfig(origins="https://app.example.com,https://www.example.com")
        assert len(config.origins_list) == 2
        assert "https://app.example.com" in config.origins_list

    def test_cors_policy_allows_origin(self):
        policy = CORSPolicy(name="test", origins=["https://example.com", "https://app.example.com"])
        assert policy.allows_origin("https://example.com") is True
        assert policy.allows_origin("https://evil.com") is False

    def test_cors_policy_allows_origin_wildcard(self):
        policy = CORSPolicy(name="test", origins=["*"])
        assert policy.allows_origin("https://any-domain.com") is True

    def test_cors_policy_allows_method(self):
        policy = CORSPolicy(name="test", methods=["GET", "POST"])
        assert policy.allows_method("GET") is True
        assert policy.allows_method("DELETE") is False

    def test_cors_policy_from_preset(self):
        strict = CORSPolicy.from_preset(CORSPreset.STRICT)
        assert strict.credentials is False
        assert "DELETE" not in strict.methods
        dev = CORSPolicy.from_preset(CORSPreset.DEVELOPMENT)
        assert "http://localhost:3000" in dev.origins

    def test_cors_request_from_headers(self):
        headers = {"Origin": "https://example.com", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Content-Type, Authorization"}
        request = CORSRequest.from_headers(headers, "OPTIONS")
        assert request.origin == "https://example.com"
        assert request.is_preflight is True
        assert request.request_method == "POST"
        assert "Content-Type" in request.request_headers

    def test_cors_response_to_headers(self):
        response = CORSResponse(allow_origin="https://example.com", allow_methods=["GET", "POST"], allow_credentials=True, max_age=3600)
        headers = response.to_headers()
        assert headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert "GET" in headers["Access-Control-Allow-Methods"]
        assert headers["Access-Control-Allow-Credentials"] == "true"

    def test_cors_handler_add_policy(self):
        handler = CORSHandler()
        policy = CORSPolicy(name="custom", origins=["https://custom.com"])
        handler.add_policy(policy)
        handler.set_service_policy("test-service", "custom")
        retrieved = handler.get_policy(service_name="test-service")
        assert retrieved.name == "custom"

    def test_cors_handler_handle_request(self):
        config = CORSConfig(origins="*")
        handler = CORSHandler(config)
        request = CORSRequest(origin="https://example.com", method="GET")
        response = handler.handle_request(request)
        assert response.allow_origin is not None

    def test_cors_handler_preflight_request(self):
        config = CORSConfig(origins="https://example.com", methods="GET,POST")
        handler = CORSHandler(config)
        request = CORSRequest(origin="https://example.com", method="OPTIONS", request_method="POST", request_headers=["Content-Type"], is_preflight=True)
        response = handler.handle_request(request)
        assert response.allow_origin == "https://example.com"
        assert response.allow_methods is not None

    def test_cors_handler_reject_invalid_origin(self):
        config = CORSConfig(origins="https://allowed.com")
        handler = CORSHandler(config)
        request = CORSRequest(origin="https://evil.com", method="GET")
        response = handler.handle_request(request)
        assert response.allow_origin is None

    def test_create_solace_cors_handler(self):
        handler = create_solace_cors_handler()
        assert "production" in handler._policies
        assert "development" in handler._policies


class TestIntegration:
    def test_full_auth_flow(self):
        auth = JWTAuthPlugin()
        token = auth.create_token(subject="user-123", roles=[UserRole.USER, UserRole.PREMIUM], email="user@example.com", session_id="session-456")
        result = auth.verify_token(token)
        assert result.authenticated is True
        assert result.claims.sub == "user-123"
        assert result.claims.has_role(UserRole.PREMIUM) is True
        assert auth.authorize(result.claims, [UserRole.USER]) is True
        assert auth.authorize(result.claims, [UserRole.ADMIN]) is False

    def test_rate_limit_with_multiple_policies(self):
        limiter = RateLimiter()
        limiter.add_policy(RateLimitPolicy(name="global", limit=100, window=RateLimitWindow.MINUTE, scope=RateLimitScope.GLOBAL))
        limiter.add_policy(RateLimitPolicy(name="service", limit=10, window=RateLimitWindow.MINUTE, scope=RateLimitScope.SERVICE, service_name="test-service"))
        for _ in range(10):
            result = limiter.check_request("test-service", None, "user-1")
            assert result.allowed is True
        result = limiter.check_request("test-service", None, "user-1")
        assert result.allowed is False

    def test_cors_with_auth_headers(self):
        config = CORSConfig(origins="https://app.solace-ai.com")
        handler = CORSHandler(config)
        headers = {"Origin": "https://app.solace-ai.com", "Authorization": "Bearer token-123"}
        cors_headers = handler.handle_headers(headers, "GET")
        assert "Access-Control-Allow-Origin" in cors_headers
