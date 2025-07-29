"""
Storage-related exception classes.
"""

from typing import Any, Optional, Dict
from .base_exceptions import ChatbotBaseException


class StorageError(ChatbotBaseException):
    """
    Base exception for storage-related errors.
    """
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if storage_type:
            context['storage_type'] = storage_type
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message,
            error_code="STORAGE_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class StorageConnectionError(StorageError):
    """
    Exception raised when connection to storage system fails.
    """
    
    def __init__(
        self,
        message: str,
        connection_string: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if connection_string:
            # Don't include full connection string for security
            context['connection_type'] = connection_string.split('://')[0] if '://' in connection_string else 'unknown'
        
        super().__init__(
            message,
            error_code="STORAGE_CONNECTION_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class DocumentNotFoundError(StorageError):
    """
    Exception raised when a requested document is not found.
    """
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if document_id:
            context['document_id'] = document_id
        
        super().__init__(
            message,
            error_code="DOCUMENT_NOT_FOUND",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )


class StoragePermissionError(StorageError):
    """
    Exception raised for storage permission/access errors.
    """
    
    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        required_permission: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if resource:
            context['resource'] = resource
        if required_permission:
            context['required_permission'] = required_permission
        
        super().__init__(
            message,
            error_code="STORAGE_PERMISSION_ERROR",
            context=context,
            **{k: v for k, v in kwargs.items() if k != 'context'}
        )