"""
Solace-AI User Service - internal (service-to-service) REST endpoints.

``internal_router`` is mounted at ``/api/v1/internal`` by ``main.ROUTERS``. It is empty until
WS-14.8 adds the internal routes (WS-14.3).
"""
from __future__ import annotations

from fastapi import APIRouter

internal_router = APIRouter()
