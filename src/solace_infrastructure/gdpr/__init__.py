"""GDPR compliance utilities (C.2 / REV-17).

Currently exposes the shared right-to-erasure orchestrator; per-service store
deleters wire their ``delete_user_data`` into ``UserDataErasure``.
"""
from __future__ import annotations

from solace_infrastructure.gdpr.erasure import (
    ErasureIncomplete,
    ErasureReport,
    StoreDeleter,
    UserDataErasure,
)

__all__ = [
    "ErasureIncomplete",
    "ErasureReport",
    "StoreDeleter",
    "UserDataErasure",
]
