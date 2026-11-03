"""Shared service runtime bootstrap (IF-01).

    from solace_infrastructure.runtime import (
        ServiceRuntimeSpec, ServiceRuntime, create_service_app, service_runtime,
        start_service_runtime, stop_service_runtime, is_strict_environment,
        StartupDependencyError,
    )

Every lifespan is ``async with service_runtime(app, SPEC) as runtime:`` (see the bootstrap
module).

Every export resolves lazily (PEP 562), so ``import solace_infrastructure.runtime`` imports
neither FastAPI nor asyncpg. Names outside ``__all__`` raise ``AttributeError``, which keeps
submodule imports such as ``from solace_infrastructure.runtime import bootstrap`` working.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "ServiceRuntime",
    "ServiceRuntimeSpec",
    "StartupDependencyError",
    "create_service_app",
    "is_strict_environment",
    "service_runtime",
    "start_service_runtime",
    "stop_service_runtime",
]

_SPEC = "solace_infrastructure.runtime.spec"
_BOOTSTRAP = "solace_infrastructure.runtime.bootstrap"

_EXPORTS: dict[str, str] = {
    "ServiceRuntimeSpec": _SPEC,
    "StartupDependencyError": _SPEC,
    "ServiceRuntime": _BOOTSTRAP,
    "create_service_app": _BOOTSTRAP,
    "service_runtime": _BOOTSTRAP,
    "start_service_runtime": _BOOTSTRAP,
    "stop_service_runtime": _BOOTSTRAP,
    "is_strict_environment": "solace_security.production_guards",
}

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from solace_infrastructure.runtime.spec import ServiceRuntimeSpec, StartupDependencyError
    from solace_security.production_guards import is_strict_environment


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # The module that defines the name does not exist yet (bootstrap.py lands in a later
        # task). Report a missing attribute so "from ... import name" fails as an ImportError
        # naming the attribute. A missing dependency inside that module propagates unchanged.
        if exc.name != module_name:
            raise
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} ({module_name} is not available)"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
