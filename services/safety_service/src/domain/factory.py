"""Safety service composition: the escalation store and the crisis detector (WS-12.1, MVP-088).

``build_escalation_repository`` decides where crisis escalation records live. Whenever the
runtime has a Postgres pool they go to ``PostgresEscalationRepository``. Without a pool, a strict
environment (staging or production) refuses to start: an escalation held in process memory is
lost on a restart, and an escalated crisis must stay visible to a human. Development and test
fall back to ``InMemoryEscalationRepository`` with one WARNING.

``build_crisis_detector`` holds today's detector composition unchanged (it still reads
``config.crisis_detection``); WS-12.6 corrects the threshold mapping test-first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from solace_infrastructure.runtime import StartupDependencyError, is_strict_environment

from ..config import SafetyConfig, get_safety_config
from .crisis_detector import CrisisDetector, CrisisDetectorSettings
from .escalation import (
    EscalationRepository,
    InMemoryEscalationRepository,
    PostgresEscalationRepository,
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    import asyncpg

logger = structlog.get_logger(__name__)

# The dependency a strict start-up is missing when there is no pool (StartupDependencyError.component).
ESCALATION_STORE_DEPENDENCY = "postgres"


def build_escalation_repository(
    pg_pool: asyncpg.Pool | None, *, environment: str
) -> EscalationRepository:
    """The escalation store for this process.

    Args:
        pg_pool: The runtime's Postgres pool (``ServiceRuntime.pg_pool``), or None when the
            runtime started without one (only possible outside strict environments).
        environment: The service's resolved environment.

    Returns:
        ``PostgresEscalationRepository(pg_pool)`` whenever a pool exists, otherwise
        ``InMemoryEscalationRepository()`` (development and test only).

    Raises:
        StartupDependencyError: no pool in a strict environment, that is when ``environment``
            or the bare ``ENVIRONMENT`` is staging or production.
    """
    if pg_pool is not None:
        logger.info("escalation_repository_postgres")
        return PostgresEscalationRepository(pg_pool)
    # The runtime's strictness rule: the service's environment OR the bare ENVIRONMENT
    # (WS121F1-CLO-1), so a blank service environment under a strict deploy stays strict.
    if is_strict_environment(environment or "") or is_strict_environment():
        raise StartupDependencyError(
            ESCALATION_STORE_DEPENDENCY,
            "escalation records require Postgres in strict environments",
        )
    logger.warning("escalation_repository_in_memory", environment=environment)
    return InMemoryEscalationRepository()


def build_crisis_detector(
    config: SafetyConfig | None = None, *, llm_assessor: Any | None = None
) -> CrisisDetector:
    """The crisis detector, configured from the centralized ``SafetyConfig`` (M-06).

    Args:
        config: The safety configuration; ``get_safety_config()`` when None.
        llm_assessor: Passed to ``CrisisDetector``; None keeps its own default assessor.
    """
    safety_config = config if config is not None else get_safety_config()
    return CrisisDetector(
        CrisisDetectorSettings(
            **{k: v for k, v in safety_config.crisis_detection.model_dump().items()
               if k in CrisisDetectorSettings.model_fields},
        ),
        llm_assessor=llm_assessor,
    )


__all__ = [
    "ESCALATION_STORE_DEPENDENCY",
    "build_crisis_detector",
    "build_escalation_repository",
]
