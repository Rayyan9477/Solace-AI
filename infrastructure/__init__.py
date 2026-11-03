"""
Solace-AI Infrastructure Components.

Subpackages are imported lazily so that importing ``infrastructure`` does not pull in the
Kong gateway package (and its python-jose dependency).
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = ["api_gateway"]


def __getattr__(name: str) -> Any:
    if name == "api_gateway":
        module = importlib.import_module(f"{__name__}.api_gateway")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
