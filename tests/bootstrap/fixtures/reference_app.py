"""A minimal conforming service module: the conformance probe's positive control (WS-04.5).

It has the shape every MVP ``main.py`` takes in its W0 phase-B adoption: ``SPEC`` declared at
import, the app built by ``create_service_app``, and a lifespan whose only statement is
``async with service_runtime(app, SPEC) as runtime:`` (WS-04.F1, form (a)). All composition and
the ``yield`` go inside it, and every service-owned stop is registered with
``runtime.register_shutdown``, so the runtime is stopped on every exit.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from solace_infrastructure.runtime import ServiceRuntimeSpec, create_service_app, service_runtime

SPEC = ServiceRuntimeSpec(
    service_name="reference-service",
    environment=os.environ.get("ENVIRONMENT", "development"),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The canonical shape names the runtime; a real service composes from it here.
    async with service_runtime(app, SPEC) as runtime:  # noqa: F841
        yield


app = create_service_app(SPEC, lifespan=lifespan, routers=[], title="ref")
