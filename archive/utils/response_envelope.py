"""
Response Envelope - Standard response format for all agents.

This module provides a standardized response envelope structure to ensure
consistency across all agent responses in the system.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResponseStatus(Enum):
    """Standard response status codes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    BLOCKED = "blocked"
    PENDING = "pending"


class ResponseEnvelope:
    """
    Standard response envelope for agent responses.

    Provides a consistent structure for all agent responses including
    status, data, metadata, errors, and warnings.
    """

    @staticmethod
    def success(
        data: Any,
        agent_name: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a success response envelope.

        Args:
            data: The main response data (can be any type)
            agent_name: Name of the agent generating the response
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata about the response
            warnings: List of non-critical warnings

        Returns:
            Standardized response dictionary

        Example:
            >>> response = ResponseEnvelope.success(
            ...     data={"message": "Analysis complete"},
            ...     agent_name="emotion_agent",
            ...     confidence=0.85
            ... )
        """
        return {
            "status": ResponseStatus.SUCCESS.value,
            "data": data,
            "metadata": {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence if confidence is not None else 1.0,
                "processing_time": None,  # To be filled by caller if available
                **(metadata or {})
            },
            "warnings": warnings or [],
            "error": None
        }

    @staticmethod
    def error(
        error_message: str,
        agent_name: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        partial_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an error response envelope.

        Args:
            error_message: Human-readable error message
            agent_name: Name of the agent generating the response
            error_code: Machine-readable error code
            error_details: Additional error details
            partial_data: Any partial data that was computed before error
            metadata: Additional metadata

        Returns:
            Standardized error response dictionary

        Example:
            >>> response = ResponseEnvelope.error(
            ...     error_message="Failed to analyze sentiment",
            ...     agent_name="emotion_agent",
            ...     error_code="SENTIMENT_ANALYSIS_FAILED"
            ... )
        """
        return {
            "status": ResponseStatus.ERROR.value,
            "data": partial_data,
            "metadata": {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.0,
                **(metadata or {})
            },
            "warnings": [],
            "error": {
                "message": error_message,
                "code": error_code or "UNKNOWN_ERROR",
                "details": error_details or {}
            }
        }

    @staticmethod
    def partial_success(
        data: Any,
        agent_name: str,
        warnings: List[str],
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a partial success response envelope.

        Used when operation completed but with issues or limitations.

        Args:
            data: The partial response data
            agent_name: Name of the agent
            warnings: List of warnings explaining limitations
            confidence: Confidence score (typically lower for partial success)
            metadata: Additional metadata

        Returns:
            Standardized partial success response dictionary

        Example:
            >>> response = ResponseEnvelope.partial_success(
            ...     data={"emotion": "happy"},
            ...     agent_name="emotion_agent",
            ...     warnings=["Fallback method used", "Limited accuracy"],
            ...     confidence=0.6
            ... )
        """
        return {
            "status": ResponseStatus.PARTIAL_SUCCESS.value,
            "data": data,
            "metadata": {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence if confidence is not None else 0.7,
                **(metadata or {})
            },
            "warnings": warnings,
            "error": None
        }

    @staticmethod
    def blocked(
        reason: str,
        agent_name: str,
        block_type: str = "safety",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a blocked response envelope.

        Used when operation was blocked for safety or policy reasons.

        Args:
            reason: Reason for blocking
            agent_name: Name of the agent
            block_type: Type of block (safety, policy, validation)
            metadata: Additional metadata

        Returns:
            Standardized blocked response dictionary

        Example:
            >>> response = ResponseEnvelope.blocked(
            ...     reason="High-risk content detected",
            ...     agent_name="safety_agent",
            ...     block_type="safety"
            ... )
        """
        return {
            "status": ResponseStatus.BLOCKED.value,
            "data": None,
            "metadata": {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": 1.0,
                "block_type": block_type,
                **(metadata or {})
            },
            "warnings": [],
            "error": {
                "message": f"Operation blocked: {reason}",
                "code": f"BLOCKED_{block_type.upper()}",
                "details": {"block_type": block_type, "reason": reason}
            }
        }

    @staticmethod
    def wrap_legacy_response(
        legacy_response: Any,
        agent_name: str,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wrap a legacy response format into the standard envelope.

        Useful for gradual migration of existing agents to standard format.

        Args:
            legacy_response: The existing response in any format
            agent_name: Name of the agent
            confidence: Optional confidence score

        Returns:
            Standardized response dictionary

        Example:
            >>> legacy = {"result": "happy", "score": 0.8}
            >>> wrapped = ResponseEnvelope.wrap_legacy_response(legacy, "emotion_agent")
        """
        # Handle dict responses
        if isinstance(legacy_response, dict):
            # Check if already has error
            if "error" in legacy_response and legacy_response["error"]:
                return ResponseEnvelope.error(
                    error_message=str(legacy_response["error"]),
                    agent_name=agent_name,
                    partial_data=legacy_response.get("response") or legacy_response.get("data")
                )

            # Extract confidence if present
            response_confidence = confidence or legacy_response.get("confidence", 0.8)

            # Check for warnings
            warnings = legacy_response.get("warnings", [])

            return ResponseEnvelope.success(
                data=legacy_response,
                agent_name=agent_name,
                confidence=response_confidence,
                warnings=warnings if warnings else None
            )

        # Handle non-dict responses
        return ResponseEnvelope.success(
            data=legacy_response,
            agent_name=agent_name,
            confidence=confidence or 0.8
        )

    @staticmethod
    def add_processing_time(
        response: Dict[str, Any],
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Add processing time to an existing response envelope.

        Args:
            response: Existing response envelope
            processing_time: Processing time in seconds

        Returns:
            Updated response with processing time
        """
        if "metadata" in response:
            response["metadata"]["processing_time"] = processing_time
        return response

    @staticmethod
    def add_context_updates(
        response: Dict[str, Any],
        context_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add context updates to an existing response envelope.

        Context updates are used by the orchestrator to maintain state
        across agent interactions.

        Args:
            response: Existing response envelope
            context_updates: Dictionary of context updates

        Returns:
            Updated response with context updates
        """
        if "metadata" in response:
            response["metadata"]["context_updates"] = context_updates
        return response

    @staticmethod
    def is_success(response: Dict[str, Any]) -> bool:
        """Check if a response indicates success."""
        return response.get("status") in [
            ResponseStatus.SUCCESS.value,
            ResponseStatus.PARTIAL_SUCCESS.value
        ]

    @staticmethod
    def is_error(response: Dict[str, Any]) -> bool:
        """Check if a response indicates an error."""
        return response.get("status") == ResponseStatus.ERROR.value

    @staticmethod
    def is_blocked(response: Dict[str, Any]) -> bool:
        """Check if a response indicates blocking."""
        return response.get("status") == ResponseStatus.BLOCKED.value

    @staticmethod
    def get_data(response: Dict[str, Any], default: Any = None) -> Any:
        """Safely extract data from a response envelope."""
        return response.get("data", default)

    @staticmethod
    def get_error(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract error information from a response envelope."""
        return response.get("error")

    @staticmethod
    def get_confidence(response: Dict[str, Any]) -> float:
        """Extract confidence score from a response envelope."""
        return response.get("metadata", {}).get("confidence", 0.0)
