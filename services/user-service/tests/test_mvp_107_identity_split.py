"""W0 split (supports MVP-107): identity routes, dependencies and composition move verbatim; the route table is unchanged."""
from __future__ import annotations

import inspect
import pathlib

from fastapi.routing import APIRoute

AUTH = {("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/login"), ("POST", "/api/v1/auth/refresh"),
        ("POST", "/api/v1/auth/logout"), ("POST", "/api/v1/auth/revoke-token"),
        ("POST", "/api/v1/auth/verify-email"), ("POST", "/api/v1/auth/resend-verification"),
        ("POST", "/api/v1/users/me/password"), ("DELETE", "/api/v1/users/me"), ("GET", "/.well-known/jwks.json"),
        ("GET", "/api/v1/auth/oauth/google/start"), ("GET", "/api/v1/auth/oauth/google/callback")}
PROFILE = {("GET", "/users/me"), ("PUT", "/users/me"), ("GET", "/users/me/preferences"),
           ("PUT", "/users/me/preferences"), ("GET", "/users/me/consent"), ("POST", "/users/me/consent"),
           ("GET", "/users/{user_id:uuid}"), ("GET", "/users/on-call-clinicians"),
           ("PUT", "/users/{user_id}/on-call")}


def _routes(routes) -> list[tuple[str, str]]:
    return [(m, r.path) for r in routes if isinstance(r, APIRoute) for m in r.methods]


def test_split_routers_own_their_routes() -> None:
    # Subset checks, so later additive tasks (WS-14.8/14.9 admin routes, WS-08 MFA routes) keep this green.
    from fastapi import APIRouter
    from src.api import router as api_router
    from src.api_auth import auth_router
    from src.internal_api import internal_router

    assert AUTH <= set(_routes(auth_router.routes))
    profile = set(_routes(api_router.routes))
    assert PROFILE <= profile
    assert not [r for r in profile if r[1].startswith("/auth/")]
    assert ("DELETE", "/users/me") not in profile and ("POST", "/users/me/password") not in profile
    assert isinstance(internal_router, APIRouter)


def test_split_app_serves_each_route_once() -> None:
    import src.main as main

    served = _routes(main.app.routes)
    expected = AUTH | {(m, "/api/v1" + p) for m, p in PROFILE}
    assert expected <= set(served) and len(served) == len(set(served))


def test_split_dependencies_and_identity_wiring_exist() -> None:
    import src.api as api
    import src.api_deps as deps
    import src.identity_wiring as wiring
    import src.main as main

    for name in ("get_jwt_service", "get_password_service", "get_user_service", "get_user_repository",
                 "get_session_manager", "get_current_user", "UserResponse", "user_to_response", "TokenResponse"):
        assert hasattr(deps, name), name
    assert api.get_current_user is deps.get_current_user
    assert api.TokenResponse is deps.TokenResponse  # src/oauth.py still imports it from .api
    assert inspect.iscoroutinefunction(wiring.start_identity) and inspect.iscoroutinefunction(wiring.stop_identity)
    text = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert "JWTService(" not in text and "RedisSessionManager(" not in text


def test_split_keeps_handler_names() -> None:
    from src.api_auth import auth_router

    names = {r.endpoint.__name__ for r in auth_router.routes if isinstance(r, APIRoute)}
    assert {"register", "login", "refresh_token", "logout", "revoke_token", "verify_email",
            "resend_verification", "change_password", "delete_current_user", "jwks"} <= names


async def test_split_lifespan_registers_the_identity_stop(monkeypatch) -> None:
    # Controller check (WS-14.3): the lifespan registers stop_identity as "identity" via the runtime.
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    import src.identity_wiring as wiring
    import src.infrastructure.event_bridge as event_bridge
    import src.main as main
    from runtime_fakes import fake_runtime

    runtime = fake_runtime(redis=None, environment="development")

    @asynccontextmanager
    async def _fake_service_runtime(app, spec):
        yield runtime

    async def _failing_bridge(**_kwargs):
        raise ConnectionError("broker down")

    monkeypatch.setattr(main, "service_runtime", _fake_service_runtime)
    monkeypatch.setattr(event_bridge, "initialize_event_bridge", _failing_bridge)
    async with main.lifespan(FastAPI()):
        assert runtime.shutdowns.get("identity") is wiring.stop_identity
