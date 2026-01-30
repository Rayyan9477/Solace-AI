"""
Solace-AI Service Base Module.
Abstract base class defining common interface for all service orchestrators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ServiceBase(ABC):
    """
    Abstract base class for all service orchestrators.

    Defines common interface that all services must implement for:
    - Service lifecycle management (initialize, shutdown)
    - Status and health reporting
    - Statistics tracking

    All service orchestrators should inherit from this class to ensure
    consistent behavior across the platform.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the service.

        This method should be called before the service starts handling
        requests. Implementations should set up connections, load
        configurations, and prepare any required resources.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the service gracefully.

        This method should be called when the service is stopping.
        Implementations should close connections, flush buffers,
        and release resources.
        """
        ...

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """
        Get current service status and statistics.

        Returns
        -------
        dict[str, Any]
            Status dictionary containing at minimum:
            - status: str - Service status ("operational", "initializing", "degraded", "error")
            - initialized: bool - Whether service has been initialized
            - statistics: dict[str, int] - Service-specific statistics
        """
        ...

    @property
    @abstractmethod
    def stats(self) -> dict[str, int]:
        """
        Get service statistics.

        Returns
        -------
        dict[str, int]
            Dictionary of service-specific statistics counters.
        """
        ...

    @property
    def is_initialized(self) -> bool:
        """
        Check if service is initialized.

        Returns
        -------
        bool
            True if service has been initialized.

        Note
        ----
        Subclasses should override this if they use a different
        attribute name for tracking initialization state.
        """
        return getattr(self, "_initialized", False)
