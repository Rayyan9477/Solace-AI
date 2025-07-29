"""
Agent-related exception classes.
"""

from typing import Any, Optional, Dict, List
from .base_exceptions import ChatbotBaseException


class AgentError(ChatbotBaseException):
    """
    Base exception for agent-related errors.
    """
    
    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if agent_id:
            context['agent_id'] = agent_id
        if agent_type:
            context['agent_type'] = agent_type
        
        super().__init__(
            message,
            error_code="AGENT_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class AgentInitializationError(AgentError):
    """
    Exception raised when agent initialization fails.
    """
    
    def __init__(
        self,
        message: str,
        initialization_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if initialization_step:
            context['initialization_step'] = initialization_step
        
        super().__init__(
            message,
            error_code="AGENT_INIT_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class AgentProcessingError(AgentError):
    """
    Exception raised when agent fails to process a request.
    """
    
    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        processing_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if request_id:
            context['request_id'] = request_id
        if processing_step:
            context['processing_step'] = processing_step
        
        super().__init__(
            message,
            error_code="AGENT_PROCESSING_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class AgentDependencyError(AgentError):
    """
    Exception raised when agent dependencies are missing or invalid.
    """
    
    def __init__(
        self,
        message: str,
        missing_dependencies: Optional[List[str]] = None,
        invalid_dependencies: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if missing_dependencies:
            context['missing_dependencies'] = missing_dependencies
        if invalid_dependencies:
            context['invalid_dependencies'] = invalid_dependencies
        
        super().__init__(
            message,
            error_code="AGENT_DEPENDENCY_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )