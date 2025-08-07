"""
Cross-Agent Therapeutic Friction Engine for Solace-AI

This module provides comprehensive therapeutic friction coordination across all agents,
integrating with the existing therapeutic friction agent to provide:
- Cross-agent friction coordination
- User readiness tracking and assessment
- Contextual friction strategies
- Breakthrough detection and response
- Adaptive friction intensity management
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import math

from src.agents.therapeutic_friction_agent import (
    TherapeuticFrictionAgent,
    ChallengeLevel,
    UserReadinessIndicator,
    InterventionType
)
from src.integration.event_bus import EventBus, Event, EventType, get_event_bus
from src.integration.supervision_mesh import SupervisionMesh
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrictionScope(Enum):
    """Scope of therapeutic friction application."""
    INDIVIDUAL_AGENT = "individual_agent"      # Single agent interaction
    CROSS_AGENT = "cross_agent"               # Multiple agent coordination
    SESSION_WIDE = "session_wide"             # Entire therapy session
    LONGITUDINAL = "longitudinal"             # Across multiple sessions


class FrictionTiming(Enum):
    """Timing strategies for friction application."""
    IMMEDIATE = "immediate"                   # Apply immediately
    DELAYED = "delayed"                       # Apply after delay
    CUMULATIVE = "cumulative"                 # Build up over time
    BREAKTHROUGH_MOMENT = "breakthrough"      # Apply at breakthrough opportunities


class BreakthroughIndicator(Enum):
    """Indicators of potential breakthrough moments."""
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    PATTERN_RECOGNITION = "pattern_recognition"
    DEFENSIVE_LOWERING = "defensive_lowering"
    INSIGHT_EMERGENCE = "insight_emergence"
    BEHAVIORAL_READINESS = "behavioral_readiness"
    VALUE_CONFLICT = "value_conflict"


@dataclass
class UserReadinessProfile:
    """Comprehensive user readiness assessment."""
    
    user_id: str
    session_id: str
    overall_readiness: UserReadinessIndicator = UserReadinessIndicator.AMBIVALENT
    readiness_confidence: float = 0.5
    
    # Domain-specific readiness
    emotional_readiness: float = 0.5         # 0-1 scale
    cognitive_readiness: float = 0.5
    behavioral_readiness: float = 0.5
    relational_readiness: float = 0.5
    
    # Contextual factors
    stress_level: float = 0.5               # 0-1, higher is more stressed
    defensive_patterns: List[str] = field(default_factory=list)
    growth_indicators: List[str] = field(default_factory=list)
    recent_breakthroughs: List[datetime] = field(default_factory=list)
    
    # History tracking
    friction_tolerance_history: deque = field(default_factory=lambda: deque(maxlen=50))
    breakthrough_history: List[Dict[str, Any]] = field(default_factory=list)
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_readiness(
        self,
        new_indicator: UserReadinessIndicator,
        confidence: float,
        evidence: Dict[str, Any] = None
    ) -> None:
        """Update readiness assessment with new evidence."""
        self.overall_readiness = new_indicator
        self.readiness_confidence = confidence
        self.last_updated = datetime.now()
        
        if evidence:
            # Update domain-specific readiness
            for domain in ['emotional', 'cognitive', 'behavioral', 'relational']:
                if f"{domain}_readiness" in evidence:
                    setattr(self, f"{domain}_readiness", evidence[f"{domain}_readiness"])
    
    def get_friction_tolerance(self) -> float:
        """Calculate current friction tolerance (0-1)."""
        # Base tolerance on overall readiness
        base_tolerance = {
            UserReadinessIndicator.RESISTANT: 0.1,
            UserReadinessIndicator.DEFENSIVE: 0.2,
            UserReadinessIndicator.AMBIVALENT: 0.4,
            UserReadinessIndicator.OPEN: 0.7,
            UserReadinessIndicator.MOTIVATED: 0.8,
            UserReadinessIndicator.BREAKTHROUGH_READY: 0.9
        }.get(self.overall_readiness, 0.4)
        
        # Adjust based on stress level (higher stress = lower tolerance)
        stress_adjustment = 1.0 - (self.stress_level * 0.3)
        
        # Adjust based on recent friction history
        recent_history = list(self.friction_tolerance_history)[-10:]  # Last 10 interactions
        if recent_history:
            history_avg = sum(recent_history) / len(recent_history)
            history_adjustment = 0.7 + (history_avg * 0.3)  # Blend with history
        else:
            history_adjustment = 1.0
        
        tolerance = base_tolerance * stress_adjustment * history_adjustment
        return max(0.0, min(1.0, tolerance))  # Clamp to 0-1
    
    def is_breakthrough_ready(self) -> bool:
        """Check if user is ready for breakthrough-level interventions."""
        return (
            self.overall_readiness in [UserReadinessIndicator.MOTIVATED, UserReadinessIndicator.BREAKTHROUGH_READY] and
            self.readiness_confidence > 0.7 and
            self.get_friction_tolerance() > 0.6
        )


@dataclass
class FrictionStrategy:
    """Strategy for applying therapeutic friction."""
    
    strategy_id: str
    name: str
    intervention_type: InterventionType
    challenge_level: ChallengeLevel
    scope: FrictionScope
    timing: FrictionTiming
    
    # Target criteria
    min_readiness: UserReadinessIndicator = UserReadinessIndicator.AMBIVALENT
    min_tolerance: float = 0.3
    required_indicators: Set[BreakthroughIndicator] = field(default_factory=set)
    
    # Strategy configuration
    intensity: float = 0.5                  # 0-1 intensity scale
    duration_minutes: Optional[int] = None  # How long to sustain friction
    agents_involved: Set[str] = field(default_factory=set)
    
    # Adaptive parameters
    escalation_threshold: float = 0.8       # When to escalate intensity
    de_escalation_threshold: float = 0.2    # When to reduce intensity
    max_intensity: float = 1.0              # Maximum intensity allowed
    
    # Success criteria
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_applicable(self, readiness_profile: UserReadinessProfile, context: Dict[str, Any]) -> bool:
        """Check if this strategy is applicable to the current context."""
        # Check readiness level
        readiness_levels = {
            UserReadinessIndicator.RESISTANT: 1,
            UserReadinessIndicator.DEFENSIVE: 2,
            UserReadinessIndicator.AMBIVALENT: 3,
            UserReadinessIndicator.OPEN: 4,
            UserReadinessIndicator.MOTIVATED: 5,
            UserReadinessIndicator.BREAKTHROUGH_READY: 6
        }
        
        user_level = readiness_levels.get(readiness_profile.overall_readiness, 3)
        min_level = readiness_levels.get(self.min_readiness, 3)
        
        if user_level < min_level:
            return False
        
        # Check tolerance level
        if readiness_profile.get_friction_tolerance() < self.min_tolerance:
            return False
        
        # Check required indicators
        if self.required_indicators:
            available_indicators = set(context.get('breakthrough_indicators', []))
            if not self.required_indicators.issubset(available_indicators):
                return False
        
        return True
    
    def calculate_effectiveness_score(
        self,
        readiness_profile: UserReadinessProfile,
        context: Dict[str, Any]
    ) -> float:
        """Calculate expected effectiveness score for this strategy."""
        if not self.is_applicable(readiness_profile, context):
            return 0.0
        
        # Base effectiveness on readiness alignment
        readiness_alignment = min(1.0, readiness_profile.get_friction_tolerance() / self.min_tolerance)
        
        # Factor in historical effectiveness
        user_history = context.get('strategy_history', {})
        historical_effectiveness = user_history.get(self.strategy_id, 0.5)
        
        # Factor in context relevance
        context_relevance = len(self.required_indicators.intersection(
            set(context.get('breakthrough_indicators', []))
        )) / max(1, len(self.required_indicators))
        
        # Calculate composite score
        effectiveness = (
            readiness_alignment * 0.4 +
            historical_effectiveness * 0.3 +
            context_relevance * 0.2 +
            self.intensity * 0.1
        )
        
        return min(1.0, effectiveness)


@dataclass
class CrossAgentFriction:
    """Coordinates friction across multiple agents."""
    
    coordination_id: str
    user_id: str
    session_id: str
    participating_agents: Set[str]
    primary_strategy: FrictionStrategy
    
    # Coordination state
    is_active: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_intensity: float = 0.0
    
    # Agent-specific configurations
    agent_roles: Dict[str, str] = field(default_factory=dict)  # agent_id -> role
    agent_intensities: Dict[str, float] = field(default_factory=dict)  # agent_id -> intensity
    
    # Synchronization
    sync_events: List[Dict[str, Any]] = field(default_factory=list)
    coordination_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Outcomes tracking
    user_responses: List[Dict[str, Any]] = field(default_factory=list)
    breakthrough_moments: List[Dict[str, Any]] = field(default_factory=list)
    effectiveness_metrics: Dict[str, float] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def start_coordination(self) -> None:
        """Start the cross-agent friction coordination."""
        self.is_active = True
        self.start_time = datetime.now()
        self.current_intensity = self.primary_strategy.intensity
        
        # Set initial agent intensities
        for agent_id in self.participating_agents:
            self.agent_intensities[agent_id] = self.current_intensity
    
    def update_intensity(self, new_intensity: float, agent_id: Optional[str] = None) -> None:
        """Update friction intensity globally or for specific agent."""
        new_intensity = max(0.0, min(1.0, new_intensity))
        
        if agent_id and agent_id in self.participating_agents:
            self.agent_intensities[agent_id] = new_intensity
        else:
            # Update all agents
            for aid in self.participating_agents:
                self.agent_intensities[aid] = new_intensity
            self.current_intensity = new_intensity
    
    def add_user_response(self, response: Dict[str, Any]) -> None:
        """Record user response to friction."""
        response['timestamp'] = datetime.now().isoformat()
        self.user_responses.append(response)
    
    def detect_breakthrough(self, indicators: List[BreakthroughIndicator], context: Dict[str, Any]) -> bool:
        """Detect if a breakthrough moment has occurred."""
        breakthrough_detected = False
        
        # Look for breakthrough patterns
        if BreakthroughIndicator.INSIGHT_EMERGENCE in indicators:
            breakthrough_detected = True
        elif (BreakthroughIndicator.EMOTIONAL_VULNERABILITY in indicators and 
              BreakthroughIndicator.DEFENSIVE_LOWERING in indicators):
            breakthrough_detected = True
        
        if breakthrough_detected:
            self.breakthrough_moments.append({
                'timestamp': datetime.now().isoformat(),
                'indicators': [i.value for i in indicators],
                'context': context,
                'current_intensity': self.current_intensity
            })
        
        return breakthrough_detected
    
    def end_coordination(self, reason: str = "completed") -> None:
        """End the cross-agent friction coordination."""
        self.is_active = False
        self.end_time = datetime.now()
        self.current_intensity = 0.0
        
        # Calculate final effectiveness metrics
        duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        self.effectiveness_metrics = {
            'duration_minutes': duration_minutes,
            'breakthrough_count': len(self.breakthrough_moments),
            'user_engagement_score': self._calculate_engagement_score(),
            'completion_reason': reason
        }
    
    def _calculate_engagement_score(self) -> float:
        """Calculate user engagement score based on responses."""
        if not self.user_responses:
            return 0.0
        
        engagement_indicators = 0
        total_responses = len(self.user_responses)
        
        for response in self.user_responses:
            # Look for engagement indicators
            if response.get('emotional_intensity', 0) > 0.6:
                engagement_indicators += 1
            if response.get('cognitive_engagement', 0) > 0.7:
                engagement_indicators += 1
            if 'insight' in response.get('content', '').lower():
                engagement_indicators += 1
        
        return engagement_indicators / (total_responses * 3)  # Normalize to 0-1


class FrictionEngine:
    """
    Comprehensive therapeutic friction engine that coordinates friction
    across all agents in the Solace-AI ecosystem.
    """
    
    def __init__(
        self,
        friction_agent: TherapeuticFrictionAgent,
        supervision_mesh: SupervisionMesh,
        event_bus: Optional[EventBus] = None
    ):
        self.friction_agent = friction_agent
        self.supervision_mesh = supervision_mesh
        self.event_bus = event_bus or get_event_bus()
        
        # User readiness tracking
        self.user_profiles: Dict[str, UserReadinessProfile] = {}
        
        # Strategy management
        self.strategies: Dict[str, FrictionStrategy] = {}
        self.active_coordinations: Dict[str, CrossAgentFriction] = {}
        
        # Analytics and metrics
        self.session_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.effectiveness_history: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.config = {
            'max_concurrent_coordinations': 3,
            'default_friction_timeout': 1800,  # 30 minutes
            'breakthrough_detection_threshold': 0.7,
            'adaptive_intensity_enabled': True,
            'cross_agent_sync_interval': 30  # seconds
        }
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Set up event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("FrictionEngine initialized")
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default friction strategies."""
        strategies = [
            FrictionStrategy(
                strategy_id="gentle_socratic",
                name="Gentle Socratic Questioning",
                intervention_type=InterventionType.SOCRATIC_QUESTIONING,
                challenge_level=ChallengeLevel.GENTLE_INQUIRY,
                scope=FrictionScope.INDIVIDUAL_AGENT,
                timing=FrictionTiming.IMMEDIATE,
                min_readiness=UserReadinessIndicator.AMBIVALENT,
                min_tolerance=0.3,
                intensity=0.4
            ),
            FrictionStrategy(
                strategy_id="cognitive_reframe_push",
                name="Cognitive Reframing Challenge",
                intervention_type=InterventionType.COGNITIVE_REFRAMING,
                challenge_level=ChallengeLevel.MODERATE_CHALLENGE,
                scope=FrictionScope.CROSS_AGENT,
                timing=FrictionTiming.DELAYED,
                min_readiness=UserReadinessIndicator.OPEN,
                min_tolerance=0.5,
                intensity=0.6,
                required_indicators={BreakthroughIndicator.COGNITIVE_DISSONANCE}
            ),
            FrictionStrategy(
                strategy_id="breakthrough_intensive",
                name="Breakthrough Intensive Intervention",
                intervention_type=InterventionType.EXPOSURE_CHALLENGE,
                challenge_level=ChallengeLevel.BREAKTHROUGH_PUSH,
                scope=FrictionScope.SESSION_WIDE,
                timing=FrictionTiming.BREAKTHROUGH_MOMENT,
                min_readiness=UserReadinessIndicator.BREAKTHROUGH_READY,
                min_tolerance=0.7,
                intensity=0.8,
                required_indicators={
                    BreakthroughIndicator.EMOTIONAL_VULNERABILITY,
                    BreakthroughIndicator.DEFENSIVE_LOWERING
                }
            )
        ]
        
        for strategy in strategies:
            self.strategies[strategy.strategy_id] = strategy
    
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for friction coordination."""
        # Subscribe to friction-related events
        self.event_bus.subscribe(
            [EventType.FRICTION_ASSESSMENT, EventType.FRICTION_APPLICATION, EventType.BREAKTHROUGH_DETECTED],
            self._handle_friction_event,
            agent_id="friction_engine"
        )
        
        # Subscribe to readiness updates
        self.event_bus.subscribe(
            EventType.READINESS_UPDATE,
            self._handle_readiness_update,
            agent_id="friction_engine"
        )
    
    async def assess_user_readiness(
        self,
        user_id: str,
        session_id: str,
        interaction_data: Dict[str, Any]
    ) -> UserReadinessProfile:
        """
        Assess user readiness for therapeutic friction.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            interaction_data: Recent interaction data
            
        Returns:
            Updated user readiness profile
        """
        # Get or create user profile
        profile_key = f"{user_id}_{session_id}"
        if profile_key not in self.user_profiles:
            self.user_profiles[profile_key] = UserReadinessProfile(
                user_id=user_id,
                session_id=session_id
            )
        
        profile = self.user_profiles[profile_key]
        
        # Use friction agent for detailed assessment
        readiness_result = await self.friction_agent.assess_user_readiness(
            interaction_data.get('message', ''),
            interaction_data.get('user_profile', {}),
            interaction_data.get('session_context', {})
        )
        
        # Update profile with new assessment
        profile.update_readiness(
            readiness_result.readiness_level,
            readiness_result.confidence,
            {
                'emotional_readiness': readiness_result.emotional_state.intensity,
                'cognitive_readiness': readiness_result.cognitive_engagement,
                'behavioral_readiness': readiness_result.behavioral_indicators.get('openness', 0.5),
                'stress_level': readiness_result.emotional_state.stress_level
            }
        )
        
        # Update friction tolerance history
        tolerance = profile.get_friction_tolerance()
        profile.friction_tolerance_history.append(tolerance)
        
        # Publish readiness update event
        await self.event_bus.publish(Event(
            event_type=EventType.READINESS_UPDATE,
            source_agent="friction_engine",
            user_id=user_id,
            session_id=session_id,
            data={
                'user_id': user_id,
                'session_id': session_id,
                'readiness': profile.overall_readiness.value,
                'confidence': profile.readiness_confidence,
                'tolerance': tolerance
            }
        ))
        
        return profile
    
    async def select_friction_strategy(
        self,
        user_id: str,
        session_id: str,
        context: Dict[str, Any],
        available_agents: Set[str] = None
    ) -> Optional[FrictionStrategy]:
        """
        Select the most appropriate friction strategy for the current context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Current interaction context
            available_agents: Set of agents available for coordination
            
        Returns:
            Selected friction strategy or None
        """
        profile_key = f"{user_id}_{session_id}"
        profile = self.user_profiles.get(profile_key)
        
        if not profile:
            logger.warning(f"No readiness profile found for {user_id}")
            return None
        
        # Evaluate all strategies
        strategy_scores = {}
        for strategy_id, strategy in self.strategies.items():
            if available_agents and strategy.scope == FrictionScope.CROSS_AGENT:
                # Check if enough agents are available
                if len(available_agents) < 2:
                    continue
            
            effectiveness_score = strategy.calculate_effectiveness_score(profile, context)
            if effectiveness_score > 0:
                strategy_scores[strategy_id] = effectiveness_score
        
        if not strategy_scores:
            logger.info(f"No applicable strategies found for {user_id}")
            return None
        
        # Select strategy with highest effectiveness score
        best_strategy_id = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        best_strategy = self.strategies[best_strategy_id]
        
        logger.info(f"Selected strategy {best_strategy_id} with score {strategy_scores[best_strategy_id]:.2f}")
        return best_strategy
    
    async def coordinate_cross_agent_friction(
        self,
        user_id: str,
        session_id: str,
        strategy: FrictionStrategy,
        participating_agents: Set[str],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Coordinate therapeutic friction across multiple agents.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            strategy: Friction strategy to implement
            participating_agents: Agents to coordinate
            context: Additional context
            
        Returns:
            Coordination ID
        """
        # Check if we're at max capacity
        if len(self.active_coordinations) >= self.config['max_concurrent_coordinations']:
            logger.warning("Maximum concurrent coordinations reached")
            return None
        
        # Create coordination
        coordination = CrossAgentFriction(
            coordination_id=f"coord_{user_id}_{int(time.time())}",
            user_id=user_id,
            session_id=session_id,
            participating_agents=participating_agents,
            primary_strategy=strategy
        )
        
        # Assign agent roles based on strategy
        agent_roles = self._assign_agent_roles(strategy, participating_agents, context or {})
        coordination.agent_roles = agent_roles
        
        # Validate coordination through supervision mesh
        validation_result = await self.supervision_mesh.validate(
            content={'coordination_plan': asdict(coordination)},
            context={'friction_strategy': asdict(strategy)},
            requires_consensus=True
        )
        
        if validation_result.final_result not in ['pass', 'warning']:
            logger.warning(f"Coordination validation failed: {validation_result.message}")
            return None
        
        # Start coordination
        coordination.start_coordination()
        self.active_coordinations[coordination.coordination_id] = coordination
        
        # Send coordination messages to participating agents
        for agent_id in participating_agents:
            await self.event_bus.publish(Event(
                event_type=EventType.FRICTION_APPLICATION,
                source_agent="friction_engine",
                target_agent=agent_id,
                user_id=user_id,
                session_id=session_id,
                data={
                    'coordination_id': coordination.coordination_id,
                    'strategy': asdict(strategy),
                    'role': agent_roles.get(agent_id, 'supporter'),
                    'intensity': coordination.agent_intensities.get(agent_id, strategy.intensity),
                    'context': context or {}
                }
            ))
        
        logger.info(f"Started cross-agent friction coordination {coordination.coordination_id}")
        return coordination.coordination_id
    
    def _assign_agent_roles(
        self,
        strategy: FrictionStrategy,
        agents: Set[str],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Assign roles to agents in the coordination."""
        roles = {}
        agent_list = list(agents)
        
        if strategy.intervention_type == InterventionType.SOCRATIC_QUESTIONING:
            # Primary questioner and supporters
            roles[agent_list[0]] = "primary_questioner"
            for agent in agent_list[1:]:
                roles[agent] = "supporting_questioner"
        
        elif strategy.intervention_type == InterventionType.COGNITIVE_REFRAMING:
            # Challenger and validator
            roles[agent_list[0]] = "primary_challenger"
            if len(agent_list) > 1:
                roles[agent_list[1]] = "reframe_validator"
            for agent in agent_list[2:]:
                roles[agent] = "supporter"
        
        elif strategy.intervention_type == InterventionType.EXPOSURE_CHALLENGE:
            # Exposure guide and safety monitor
            roles[agent_list[0]] = "exposure_guide"
            if len(agent_list) > 1:
                roles[agent_list[1]] = "safety_monitor"
            for agent in agent_list[2:]:
                roles[agent] = "supporter"
        
        else:
            # Default roles
            roles[agent_list[0]] = "primary"
            for agent in agent_list[1:]:
                roles[agent] = "supporter"
        
        return roles
    
    async def detect_breakthrough_opportunity(
        self,
        user_id: str,
        session_id: str,
        interaction_data: Dict[str, Any]
    ) -> Tuple[bool, List[BreakthroughIndicator]]:
        """
        Detect breakthrough opportunities in user interactions.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            interaction_data: Current interaction data
            
        Returns:
            Tuple of (breakthrough_detected, indicators)
        """
        indicators = []
        
        # Use friction agent's breakthrough detection
        breakthrough_assessment = await self.friction_agent.detect_breakthrough_opportunity(
            interaction_data.get('message', ''),
            interaction_data.get('user_profile', {}),
            interaction_data.get('session_context', {})
        )
        
        # Map friction agent results to our indicators
        if breakthrough_assessment.emotional_vulnerability > 0.7:
            indicators.append(BreakthroughIndicator.EMOTIONAL_VULNERABILITY)
        
        if breakthrough_assessment.cognitive_dissonance > 0.6:
            indicators.append(BreakthroughIndicator.COGNITIVE_DISSONANCE)
        
        if breakthrough_assessment.defense_lowering > 0.6:
            indicators.append(BreakthroughIndicator.DEFENSIVE_LOWERING)
        
        # Additional analysis for our specific indicators
        content = interaction_data.get('message', '').lower()
        
        if any(phrase in content for phrase in ['i realize', 'i understand now', 'that makes sense']):
            indicators.append(BreakthroughIndicator.INSIGHT_EMERGENCE)
        
        if any(phrase in content for phrase in ['i want to change', 'ready to try', 'willing to do']):
            indicators.append(BreakthroughIndicator.BEHAVIORAL_READINESS)
        
        if any(phrase in content for phrase in ['pattern', 'always do this', 'same thing']):
            indicators.append(BreakthroughIndicator.PATTERN_RECOGNITION)
        
        breakthrough_detected = len(indicators) >= 2 or breakthrough_assessment.readiness_for_challenge > 0.8
        
        if breakthrough_detected:
            await self.event_bus.publish(Event(
                event_type=EventType.BREAKTHROUGH_DETECTED,
                source_agent="friction_engine",
                user_id=user_id,
                session_id=session_id,
                data={
                    'indicators': [i.value for i in indicators],
                    'confidence': breakthrough_assessment.readiness_for_challenge,
                    'context': interaction_data
                }
            ))
        
        return breakthrough_detected, indicators
    
    async def adapt_friction_intensity(
        self,
        coordination_id: str,
        user_response: Dict[str, Any]
    ) -> bool:
        """
        Adapt friction intensity based on user response.
        
        Args:
            coordination_id: Active coordination ID
            user_response: User's response to current friction
            
        Returns:
            True if intensity was adapted
        """
        if coordination_id not in self.active_coordinations:
            return False
        
        coordination = self.active_coordinations[coordination_id]
        if not coordination.is_active:
            return False
        
        # Record user response
        coordination.add_user_response(user_response)
        
        # Analyze response for intensity adjustment
        current_intensity = coordination.current_intensity
        
        # Indicators for escalation
        escalation_indicators = [
            user_response.get('engagement_level', 0) > 0.7,
            user_response.get('emotional_intensity', 0) > 0.6,
            user_response.get('defensive_level', 1) < 0.4,
            'insight' in user_response.get('content', '').lower()
        ]
        
        # Indicators for de-escalation
        deescalation_indicators = [
            user_response.get('stress_level', 0) > 0.8,
            user_response.get('defensive_level', 0) > 0.8,
            user_response.get('withdrawal_signs', 0) > 0.6,
            'stop' in user_response.get('content', '').lower()
        ]
        
        escalation_score = sum(escalation_indicators) / len(escalation_indicators)
        deescalation_score = sum(deescalation_indicators) / len(deescalation_indicators)
        
        new_intensity = current_intensity
        
        if escalation_score > coordination.primary_strategy.escalation_threshold:
            # Increase intensity
            new_intensity = min(
                coordination.primary_strategy.max_intensity,
                current_intensity + 0.1
            )
        elif deescalation_score > coordination.primary_strategy.de_escalation_threshold:
            # Decrease intensity
            new_intensity = max(0.1, current_intensity - 0.2)
        
        if abs(new_intensity - current_intensity) > 0.05:  # Significant change
            coordination.update_intensity(new_intensity)
            
            # Notify participating agents of intensity change
            for agent_id in coordination.participating_agents:
                await self.event_bus.publish(Event(
                    event_type=EventType.FRICTION_APPLICATION,
                    source_agent="friction_engine",
                    target_agent=agent_id,
                    data={
                        'coordination_id': coordination_id,
                        'intensity_update': new_intensity,
                        'reason': 'adaptive_adjustment',
                        'user_response_analysis': {
                            'escalation_score': escalation_score,
                            'deescalation_score': deescalation_score
                        }
                    }
                ))
            
            logger.info(f"Adapted friction intensity from {current_intensity:.2f} to {new_intensity:.2f}")
            return True
        
        return False
    
    async def end_friction_coordination(
        self,
        coordination_id: str,
        reason: str = "completed"
    ) -> bool:
        """
        End an active friction coordination.
        
        Args:
            coordination_id: Coordination to end
            reason: Reason for ending
            
        Returns:
            True if successfully ended
        """
        if coordination_id not in self.active_coordinations:
            return False
        
        coordination = self.active_coordinations[coordination_id]
        coordination.end_coordination(reason)
        
        # Notify participating agents
        for agent_id in coordination.participating_agents:
            await self.event_bus.publish(Event(
                event_type=EventType.FRICTION_APPLICATION,
                source_agent="friction_engine",
                target_agent=agent_id,
                data={
                    'coordination_id': coordination_id,
                    'action': 'end_coordination',
                    'reason': reason,
                    'final_metrics': coordination.effectiveness_metrics
                }
            ))
        
        # Store analytics
        self.session_analytics[coordination.session_id][coordination_id] = {
            'strategy': coordination.primary_strategy.strategy_id,
            'metrics': coordination.effectiveness_metrics,
            'breakthrough_count': len(coordination.breakthrough_moments),
            'duration': coordination.effectiveness_metrics.get('duration_minutes', 0)
        }
        
        # Update strategy effectiveness history
        effectiveness = coordination.effectiveness_metrics.get('user_engagement_score', 0.5)
        self.effectiveness_history[coordination.primary_strategy.strategy_id].append(effectiveness)
        
        # Remove from active coordinations
        del self.active_coordinations[coordination_id]
        
        logger.info(f"Ended friction coordination {coordination_id}: {reason}")
        return True
    
    async def _handle_friction_event(self, event: Event) -> None:
        """Handle incoming friction-related events."""
        try:
            if event.event_type == EventType.FRICTION_ASSESSMENT:
                # Handle friction assessment request
                user_id = event.user_id
                session_id = event.session_id
                interaction_data = event.data
                
                profile = await self.assess_user_readiness(user_id, session_id, interaction_data)
                
                # Send assessment result
                await self.event_bus.publish(Event(
                    event_type=EventType.FRICTION_ASSESSMENT,
                    source_agent="friction_engine",
                    target_agent=event.source_agent,
                    correlation_id=event.correlation_id,
                    user_id=user_id,
                    session_id=session_id,
                    data={
                        'readiness_profile': asdict(profile),
                        'assessment_timestamp': datetime.now().isoformat()
                    }
                ))
            
            elif event.event_type == EventType.BREAKTHROUGH_DETECTED:
                # Handle breakthrough detection
                coordination_ids = [
                    coord_id for coord_id, coord in self.active_coordinations.items()
                    if coord.user_id == event.user_id and coord.session_id == event.session_id
                ]
                
                for coord_id in coordination_ids:
                    coordination = self.active_coordinations[coord_id]
                    indicators = [BreakthroughIndicator(i) for i in event.data.get('indicators', [])]
                    
                    if coordination.detect_breakthrough(indicators, event.data.get('context', {})):
                        # Opportunity for breakthrough-level intervention
                        if coordination.primary_strategy.timing == FrictionTiming.BREAKTHROUGH_MOMENT:
                            # Escalate to breakthrough intensity
                            coordination.update_intensity(0.8)
                            
                            # Notify agents
                            for agent_id in coordination.participating_agents:
                                await self.event_bus.publish(Event(
                                    event_type=EventType.FRICTION_APPLICATION,
                                    source_agent="friction_engine",
                                    target_agent=agent_id,
                                    data={
                                        'coordination_id': coord_id,
                                        'breakthrough_opportunity': True,
                                        'indicators': event.data.get('indicators', []),
                                        'suggested_intensity': 0.8
                                    }
                                ))
        
        except Exception as e:
            logger.error(f"Error handling friction event: {e}")
    
    async def _handle_readiness_update(self, event: Event) -> None:
        """Handle readiness update events from other agents."""
        try:
            user_id = event.data.get('user_id')
            session_id = event.data.get('session_id')
            
            if user_id and session_id:
                profile_key = f"{user_id}_{session_id}"
                if profile_key in self.user_profiles:
                    profile = self.user_profiles[profile_key]
                    
                    # Update profile with external assessment
                    new_readiness = UserReadinessIndicator(event.data.get('readiness'))
                    confidence = event.data.get('confidence', 0.5)
                    
                    profile.update_readiness(new_readiness, confidence)
        
        except Exception as e:
            logger.error(f"Error handling readiness update: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive friction engine metrics."""
        active_count = len(self.active_coordinations)
        total_users = len(self.user_profiles)
        
        # Calculate strategy effectiveness averages
        strategy_effectiveness = {}
        for strategy_id, effectiveness_list in self.effectiveness_history.items():
            if effectiveness_list:
                strategy_effectiveness[strategy_id] = {
                    'average': sum(effectiveness_list) / len(effectiveness_list),
                    'count': len(effectiveness_list),
                    'recent_trend': effectiveness_list[-5:] if len(effectiveness_list) >= 5 else effectiveness_list
                }
        
        # Calculate breakthrough rates
        total_breakthroughs = sum(
            len(coord.breakthrough_moments)
            for coord in self.active_coordinations.values()
        )
        
        return {
            'active_coordinations': active_count,
            'total_user_profiles': total_users,
            'strategy_effectiveness': strategy_effectiveness,
            'total_breakthroughs': total_breakthroughs,
            'session_analytics_count': len(self.session_analytics),
            'average_coordination_duration': self._calculate_avg_duration(),
            'config': self.config
        }
    
    def _calculate_avg_duration(self) -> float:
        """Calculate average coordination duration."""
        durations = []
        
        for session_data in self.session_analytics.values():
            for coord_data in session_data.values():
                if 'duration' in coord_data:
                    durations.append(coord_data['duration'])
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def get_user_friction_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get friction history for a specific user and session."""
        profile_key = f"{user_id}_{session_id}"
        profile = self.user_profiles.get(profile_key)
        
        if not profile:
            return {}
        
        session_data = self.session_analytics.get(session_id, {})
        
        return {
            'readiness_profile': asdict(profile),
            'friction_tolerance_history': list(profile.friction_tolerance_history),
            'breakthrough_history': profile.breakthrough_history,
            'session_coordinations': session_data
        }


# Factory function for creating a production-ready friction engine

def create_friction_engine(
    friction_agent: TherapeuticFrictionAgent,
    supervision_mesh: SupervisionMesh,
    event_bus: Optional[EventBus] = None
) -> FrictionEngine:
    """Create a production-ready friction engine with default configuration."""
    engine = FrictionEngine(friction_agent, supervision_mesh, event_bus)
    
    # Add additional production strategies
    production_strategies = [
        FrictionStrategy(
            strategy_id="values_exploration",
            name="Values Exploration Challenge",
            intervention_type=InterventionType.VALUES_CLARIFICATION,
            challenge_level=ChallengeLevel.MODERATE_CHALLENGE,
            scope=FrictionScope.CROSS_AGENT,
            timing=FrictionTiming.CUMULATIVE,
            min_readiness=UserReadinessIndicator.OPEN,
            min_tolerance=0.5,
            intensity=0.6,
            required_indicators={BreakthroughIndicator.VALUE_CONFLICT}
        ),
        FrictionStrategy(
            strategy_id="behavioral_experiment",
            name="Guided Behavioral Experiment",
            intervention_type=InterventionType.BEHAVIORAL_EXPERIMENT,
            challenge_level=ChallengeLevel.STRONG_CHALLENGE,
            scope=FrictionScope.SESSION_WIDE,
            timing=FrictionTiming.DELAYED,
            min_readiness=UserReadinessIndicator.MOTIVATED,
            min_tolerance=0.6,
            intensity=0.7,
            required_indicators={BreakthroughIndicator.BEHAVIORAL_READINESS}
        )
    ]
    
    for strategy in production_strategies:
        engine.strategies[strategy.strategy_id] = strategy
    
    return engine