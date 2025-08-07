"""
Solace-AI Integration Module

This module provides event-driven integration capabilities for the Solace-AI platform,
including event bus messaging, distributed supervision, and therapeutic friction coordination.

Components:
- EventBus: Asynchronous pub/sub messaging system with event sourcing
- SupervisionMesh: Distributed validation and quality assurance
- FrictionEngine: Cross-agent therapeutic friction coordination

The integration module enables seamless communication and coordination between
all agents in the Solace-AI ecosystem while maintaining therapeutic boundaries
and clinical oversight.
"""

from .event_bus import EventBus, Event, EventHandler, EventSubscription
from .supervision_mesh import SupervisionMesh, ValidationGate, ConsensusValidator
from .friction_engine import FrictionEngine, FrictionStrategy, CrossAgentFriction

__all__ = [
    'EventBus',
    'Event', 
    'EventHandler',
    'EventSubscription',
    'SupervisionMesh',
    'ValidationGate',
    'ConsensusValidator',
    'FrictionEngine',
    'FrictionStrategy',
    'CrossAgentFriction'
]

__version__ = "1.0.0"