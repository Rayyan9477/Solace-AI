"""Safety Service source package."""
from __future__ import annotations

__all__ = ["app", "create_application"]


def __getattr__(name: str):
    """Lazy imports to avoid triggering full app initialization during test collection."""
    if name in ("app", "create_application"):
        from .main import app, create_application
        return {"app": app, "create_application": create_application}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
