"""
Dependency Injection infrastructure.

This module provides a dependency injection container that enables
loose coupling between components and makes testing easier.
"""

from .container import DIContainer, Injectable, Singleton
from .decorators import inject, provides

__all__ = [
    'DIContainer',
    'Injectable', 
    'Singleton',
    'inject',
    'provides'
]