"""
Adaptive Learning System Integration

This module provides integration between the AdaptiveLearningAgent and the existing
Solace-AI system architecture. It handles the coordination between all adaptive
learning components and provides a unified interface for system-wide learning
and optimization.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

# Import existing system components
from .base_agent import BaseAgent
from .agent_orchestrator import AgentOrchestrator
from .supervisor_agent import SupervisorAgent
from ..diagnosis.adaptive_learning import AdaptiveLearningEngine, InterventionOutcome, UserProfile

# Import our new components
from .adaptive_learning_agent import AdaptiveLearningAgent, AdvancedRLEngine, SystemState, RewardSignal
from .pattern_recognition_engine import PatternRecognitionEngine, TherapeuticPattern, UserBehaviorPattern
from .personalization_engine import PersonalizationEngine, PersonalizationRecommendation, PersonalizationProfile
from .feedback_integration_system import FeedbackProcessor, FeedbackEntry, FeedbackSummary
from .outcome_tracker import OutcomeTracker, OutcomeMeasurement, UserOutcomeProfile, OutcomeAlert
from .privacy_protection_system import PrivacyProtectionSystem, DataSensitivity, PrivacyLevel
from .insights_generation_system import InsightGenerationSystem, SystemInsight, InsightType, InsightPriority

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SystemLearningUpdate:
    """Comprehensive system learning update"""
    update_id: str
    timestamp: datetime
    
    # Sources of learning
    feedback_insights: Optional[Dict[str, Any]] = None
    pattern_discoveries: Optional[Dict[str, Any]] = None
    outcome_changes: Optional[Dict[str, Any]] = None
    personalization_updates: Optional[Dict[str, Any]] = None
    rl_optimizations: Optional[Dict[str, Any]] = None
    
    # Targets for updates
    agent_updates: Dict[str, Any] = None
    workflow_optimizations: Dict[str, Any] = None
    intervention_adjustments: Dict[str, Any] = None
    
    # Implementation details
    priority: str = "medium"  # low, medium, high, critical
    confidence: float = 0.0
    expected_impact: float = 0.0
    
    # Results tracking
    applied: bool = False
    application_timestamp: Optional[datetime] = None
    measured_impact: Optional[float] = None

class AdaptiveLearningSystemIntegrator:
    """
    Main integrator for the adaptive learning system.
    
    This class coordinates all adaptive learning components and provides
    a unified interface for system-wide learning and optimization.
    """
    
    def __init__(self, 
                 orchestrator: AgentOrchestrator,
                 vector_db: Optional[CentralVectorDB] = None,
                 config: Dict[str, Any] = None):
        
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Core system components
        self.orchestrator = orchestrator
        self.vector_db = vector_db
        
        # Initialize adaptive learning components
        self.adaptive_learning_agent = AdaptiveLearningAgent(
            model=config.get('model_provider') if config else None,
            config=config.get('adaptive_agent_config', {})
        )
        
        self.pattern_engine = PatternRecognitionEngine(
            config=config.get('pattern_recognition_config', {})
        )
        
        self.personalization_engine = PersonalizationEngine(
            config=config.get('personalization_config', {})
        )
        
        self.feedback_processor = FeedbackProcessor(
            config=config.get('feedback_processing_config', {})
        )
        
        self.outcome_tracker = OutcomeTracker(
            config=config.get('outcome_tracking_config', {})
        )
        
        self.privacy_protection = PrivacyProtectionSystem(
            config=config.get('privacy_protection_config', {})
        )
        
        self.insights_generator = InsightGenerationSystem(
            outcome_tracker=self.outcome_tracker,
            feedback_processor=self.feedback_processor,
            personalization_engine=self.personalization_engine,
            pattern_engine=self.pattern_engine,
            config=config.get('insights_generation_config', {})
        )
        
        # Integration state
        self.learning_updates = []
        self.system_performance_history = []
        self.integration_metrics = defaultdict(list)
        self.generated_insights = []
        
        # Configuration
        self.learning_cycle_interval = config.get('learning_cycle_interval', 3600)  # 1 hour
        self.enable_real_time_learning = config.get('enable_real_time_learning', True)
        self.enable_background_optimization = config.get('enable_background_optimization', True)
        self.enable_privacy_protection = config.get('enable_privacy_protection', True)
        self.enable_insights_generation = config.get('enable_insights_generation', True)
        self.insights_generation_interval = config.get('insights_generation_interval', 86400)  # Daily
        
        # Performance tracking
        self.last_optimization_cycle = None
        self.optimization_cycle_count = 0
        
        self.logger.info("Adaptive Learning System Integrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all adaptive learning components"""
        
        try:
            self.logger.info("Initializing Adaptive Learning System")
            
            # Register the adaptive learning agent with the orchestrator
            await self.orchestrator.register_agent(
                "adaptive_learning_agent", 
                self.adaptive_learning_agent
            )
            
            # Initialize all components
            initialization_tasks = [
                self._initialize_pattern_engine(),
                self._initialize_personalization_engine(),
                self._initialize_feedback_processor(),
                self._initialize_outcome_tracker()
            ]
            
            await asyncio.gather(*initialization_tasks)
            
            # Start background learning cycle if enabled
            if self.enable_background_optimization:
                asyncio.create_task(self._background_learning_cycle())
            
            self.logger.info("Adaptive Learning System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Adaptive Learning System: {str(e)}")
            return False
    
    async def _initialize_pattern_engine(self):
        """Initialize pattern recognition engine"""
        # Pattern engine initialization is handled in constructor
        self.logger.info("Pattern Recognition Engine initialized")
    
    async def _initialize_personalization_engine(self):
        """Initialize personalization engine"""
        # Personalization engine initialization is handled in constructor
        self.logger.info("Personalization Engine initialized")
    
    async def _initialize_feedback_processor(self):
        """Initialize feedback processing system"""
        # Feedback processor initialization is handled in constructor
        self.logger.info("Feedback Processing System initialized")
    
    async def _initialize_outcome_tracker(self):
        """Initialize outcome tracking system"""
        # Outcome tracker initialization is handled in constructor
        self.logger.info("Outcome Tracking System initialized")
    
    async def process_user_interaction(self,
                                     user_id: str,
                                     session_id: str,
                                     interaction_data: Dict[str, Any],
                                     agent_response: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a complete user interaction through the adaptive learning system.
        
        This is the main entry point for learning from user interactions.
        """
        
        try:
            self.logger.info(f"Processing adaptive learning for interaction: {session_id}")
            
            learning_results = {
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'components_processed': [],
                'insights_generated': [],
                'recommendations': [],
                'alerts': [],
                'learning_updates_applied': [],
                'privacy_protected': False
            }
            
            # Step 0: Apply privacy protection if enabled
            protected_interaction_data = interaction_data
            if self.enable_privacy_protection:
                # Determine data sensitivity
                sensitivity = DataSensitivity.CONFIDENTIAL
                if any(key in interaction_data for key in ['diagnosis', 'medical_history', 'personal_info']):
                    sensitivity = DataSensitivity.RESTRICTED
                elif any(key in interaction_data for key in ['feedback', 'rating', 'session_data']):
                    sensitivity = DataSensitivity.INTERNAL
                
                # Process data through privacy protection
                protected_data = self.privacy_protection.process_learning_data(
                    user_id=user_id,
                    data=interaction_data,
                    sensitivity=sensitivity
                )
                
                # Use anonymized data for further processing
                protected_interaction_data = protected_data.features
                learning_results['privacy_protected'] = True
                learning_results['components_processed'].append('privacy_protection')
                
                self.logger.debug(f"Applied privacy protection with {sensitivity.value} sensitivity level")
            
            # 1. Process feedback if available
            if 'feedback' in protected_interaction_data or 'user_rating' in protected_interaction_data:
                feedback_result = await self._process_interaction_feedback(
                    user_id, session_id, protected_interaction_data, context
                )
                learning_results['components_processed'].append('feedback_processing')
                learning_results['insights_generated'].extend(feedback_result.get('insights', []))
            
            # 2. Record outcome measurements if available
            if 'outcome_measurements' in protected_interaction_data:
                outcome_result = await self._process_outcome_measurements(
                    user_id, protected_interaction_data['outcome_measurements'], context
                )
                learning_results['components_processed'].append('outcome_tracking')
                learning_results['alerts'].extend(outcome_result.get('alerts', []))
            
            # 3. Update personalization based on interaction
            personalization_result = await self._update_personalization_from_interaction(
                user_id, session_id, protected_interaction_data, agent_response, context
            )
            learning_results['components_processed'].append('personalization')
            learning_results['recommendations'].extend(personalization_result.get('recommendations', []))
            
            # 4. Real-time pattern analysis if enabled
            if self.enable_real_time_learning and len(protected_interaction_data) > 0:
                pattern_result = await self._analyze_interaction_patterns(
                    user_id, protected_interaction_data, agent_response, context
                )
                learning_results['components_processed'].append('pattern_recognition')
                learning_results['insights_generated'].extend(pattern_result.get('patterns', []))
            
            # 5. System-wide optimization trigger
            optimization_result = await self._trigger_system_optimization(
                user_id, protected_interaction_data, agent_response
            )
            
            if optimization_result.get('optimization_triggered'):
                learning_results['components_processed'].append('system_optimization')
                learning_results['learning_updates_applied'].extend(
                    optimization_result.get('updates_applied', [])
                )
            
            # 6. Generate insights if enabled and sufficient data
            if self.enable_insights_generation:
                await self._trigger_insights_generation(learning_results)
            
            # 7. Update integration metrics
            await self._update_integration_metrics(learning_results)
            
            # 8. Validate privacy compliance
            if self.enable_privacy_protection:
                privacy_status = self.privacy_protection.validate_privacy_compliance()
                learning_results['privacy_compliance'] = privacy_status
                
                if not privacy_status.get('overall_compliant', True):
                    self.logger.warning("Privacy compliance issue detected")
                    learning_results['alerts'].append({
                        'type': 'privacy_compliance',
                        'severity': 'high',
                        'message': 'Privacy compliance validation failed'
                    })
            
            return learning_results
            
        except Exception as e:
            self.logger.error(f"Error in adaptive learning processing: {str(e)}")
            return {
                'error': str(e),
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _process_interaction_feedback(self,
                                          user_id: str,
                                          session_id: str,
                                          interaction_data: Dict[str, Any],
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process feedback from user interaction"""
        
        # Extract feedback data
        feedback_data = {}
        
        if 'user_rating' in interaction_data:
            feedback_data['rating'] = interaction_data['user_rating']
        
        if 'feedback' in interaction_data:
            feedback_data['text_feedback'] = interaction_data['feedback']
        
        if 'response_time' in interaction_data:
            feedback_data['response_time'] = interaction_data['response_time']
        
        if 'interaction_duration' in interaction_data:
            feedback_data['interaction_duration'] = interaction_data['interaction_duration']
        
        # Process feedback through feedback processor
        feedback_entry = await self.feedback_processor.process_feedback(
            user_id=user_id,
            session_id=session_id,
            feedback_data=feedback_data,
            intervention_context=context
        )
        
        # Generate insights
        insights = []
        if feedback_entry.sentiment:
            insights.append({
                'type': 'sentiment_analysis',
                'sentiment': feedback_entry.sentiment.value,
                'confidence': feedback_entry.confidence,
                'topics': feedback_entry.topics
            })
        
        return {
            'feedback_entry_id': feedback_entry.feedback_id,
            'insights': insights,
            'urgent_indicators': feedback_entry.urgency_level == 'critical'
        }
    
    async def _process_outcome_measurements(self,
                                          user_id: str,
                                          measurements: Dict[str, Any],
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process outcome measurements from user interaction"""
        
        from .outcome_tracker import OutcomeType
        
        alerts = []
        
        for outcome_name, score in measurements.items():
            try:
                # Convert string to OutcomeType enum
                outcome_type = OutcomeType(outcome_name.lower())
                
                # Record measurement
                measurement = await self.outcome_tracker.record_outcome_measurement(
                    user_id=user_id,
                    outcome_type=outcome_type,
                    primary_score=float(score),
                    context=context,
                    data_source="user_interaction"
                )
                
                # Check if alerts were generated
                if measurement.alert_generated:
                    user_alerts = self.outcome_tracker.get_active_alerts(user_id=user_id)
                    alerts.extend(user_alerts)
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not process outcome measurement {outcome_name}: {str(e)}")
        
        return {
            'measurements_processed': len(measurements),
            'alerts': alerts
        }
    
    async def _update_personalization_from_interaction(self,
                                                     user_id: str,
                                                     session_id: str,
                                                     interaction_data: Dict[str, Any],
                                                     agent_response: Dict[str, Any],
                                                     context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Update personalization based on interaction outcome"""
        
        # Create a simulated intervention outcome for learning
        outcome = InterventionOutcome(
            intervention_id=f"interaction_{session_id}",
            user_id=user_id,
            intervention_type=agent_response.get('agent_name', 'general'),
            intervention_content=str(agent_response.get('response', '')),
            context=context or {},
            timestamp=datetime.now(),
            user_response=interaction_data.get('feedback', ''),
            engagement_score=self._calculate_engagement_score(interaction_data),
            effectiveness_score=self._calculate_effectiveness_score(interaction_data),
            breakthrough_indicator=interaction_data.get('breakthrough', False)
        )
        
        # Learn from the outcome
        await self.personalization_engine.learn_from_outcome(
            user_id=user_id,
            intervention_id=outcome.intervention_id,
            outcome=outcome
        )
        
        # Generate new recommendations based on updated learning
        recommendations = []
        if context and 'available_interventions' in context:
            personalization_rec = await self.personalization_engine.personalize_response(
                user_id=user_id,
                current_context=context,
                available_interventions=context['available_interventions']
            )
            
            recommendations.append({
                'type': 'intervention_personalization',
                'recommended_interventions': personalization_rec.recommended_interventions,
                'confidence': personalization_rec.confidence,
                'reasoning': personalization_rec.reasoning
            })
        
        return {
            'personalization_updated': True,
            'recommendations': recommendations
        }
    
    def _calculate_engagement_score(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate engagement score from interaction data"""
        
        score = 0.5  # Base score
        
        # Response time (faster = more engaged)
        if 'response_time' in interaction_data:
            response_time = interaction_data['response_time']
            if response_time < 30:  # Less than 30 seconds
                score += 0.2
            elif response_time > 300:  # More than 5 minutes
                score -= 0.2
        
        # Interaction duration (longer = more engaged, up to a point)
        if 'interaction_duration' in interaction_data:
            duration = interaction_data['interaction_duration']
            if 60 <= duration <= 1800:  # 1-30 minutes is good engagement
                score += 0.2
            elif duration < 30:  # Too short
                score -= 0.2
        
        # Feedback provision (providing feedback = more engaged)
        if 'feedback' in interaction_data and interaction_data['feedback']:
            score += 0.2
        
        # Rating provision
        if 'user_rating' in interaction_data:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_effectiveness_score(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate effectiveness score from interaction data"""
        
        # Use rating if available
        if 'user_rating' in interaction_data:
            rating = interaction_data['user_rating']
            return max(0.0, min(1.0, rating / 10.0))  # Convert to 0-1 scale
        
        # Use feedback sentiment analysis
        if 'feedback' in interaction_data:
            feedback = interaction_data['feedback'].lower()
            
            positive_indicators = ['helpful', 'good', 'better', 'useful', 'thank']
            negative_indicators = ['unhelpful', 'bad', 'worse', 'useless', 'frustrated']
            
            positive_count = sum(1 for word in positive_indicators if word in feedback)
            negative_count = sum(1 for word in negative_indicators if word in feedback)
            
            if positive_count > negative_count:
                return 0.7
            elif negative_count > positive_count:
                return 0.3
            else:
                return 0.5
        
        # Default moderate effectiveness if no clear indicators
        return 0.5
    
    async def _analyze_interaction_patterns(self,
                                          user_id: str,
                                          interaction_data: Dict[str, Any],
                                          agent_response: Dict[str, Any],
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in the interaction for learning"""
        
        # This would involve more complex pattern analysis
        # For now, return a simplified pattern detection
        
        patterns = []
        
        # Detect high engagement pattern
        if self._calculate_engagement_score(interaction_data) > 0.8:
            patterns.append({
                'type': 'high_engagement',
                'description': 'User showing high engagement with current approach',
                'confidence': 0.8,
                'recommendation': 'Continue current interaction style'
            })
        
        # Detect effectiveness pattern
        effectiveness = self._calculate_effectiveness_score(interaction_data)
        if effectiveness > 0.8:
            patterns.append({
                'type': 'high_effectiveness',
                'description': 'Intervention approach proving highly effective',
                'confidence': 0.8,
                'recommendation': 'Reinforce successful intervention strategies'
            })
        elif effectiveness < 0.3:
            patterns.append({
                'type': 'low_effectiveness',
                'description': 'Current approach showing limited effectiveness',
                'confidence': 0.7,
                'recommendation': 'Consider alternative intervention approaches'
            })
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns)
        }
    
    async def _trigger_system_optimization(self,
                                         user_id: str,
                                         interaction_data: Dict[str, Any],
                                         agent_response: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger system-wide optimization based on interaction"""
        
        # Check if optimization cycle is due
        now = datetime.now()
        
        if (self.last_optimization_cycle is None or 
            (now - self.last_optimization_cycle).seconds >= self.learning_cycle_interval):
            
            # Trigger optimization cycle
            optimization_result = await self._run_optimization_cycle()
            self.last_optimization_cycle = now
            self.optimization_cycle_count += 1
            
            return {
                'optimization_triggered': True,
                'optimization_cycle': self.optimization_cycle_count,
                'updates_applied': optimization_result.get('updates_applied', [])
            }
        
        return {
            'optimization_triggered': False,
            'next_optimization_due': (
                self.last_optimization_cycle + 
                timedelta(seconds=self.learning_cycle_interval)
            ).isoformat() if self.last_optimization_cycle else 'now'
        }
    
    async def _run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete system optimization cycle"""
        
        try:
            self.logger.info(f"Running optimization cycle #{self.optimization_cycle_count + 1}")
            
            # Collect system state information
            system_state = await self._collect_system_state()
            
            # Get recent outcomes for RL optimization
            recent_outcomes = await self._get_recent_outcomes()
            
            # Run RL-based system optimization
            rl_optimization = await self.adaptive_learning_agent.rl_engine.optimize_system_performance(
                system_state, recent_outcomes
            )
            
            # Process optimization results
            updates_applied = []
            
            for action in rl_optimization.get('optimization_actions', []):
                # Apply optimization action (simplified)
                update_result = await self._apply_optimization_action(action)
                if update_result.get('success'):
                    updates_applied.append(action['action_type'])
            
            self.logger.info(f"Optimization cycle completed: {len(updates_applied)} updates applied")
            
            return {
                'cycle_number': self.optimization_cycle_count + 1,
                'updates_applied': updates_applied,
                'rl_optimization_result': rl_optimization,
                'system_state_analyzed': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {str(e)}")
            return {'error': str(e)}
    
    async def _trigger_insights_generation(self, learning_results: Dict[str, Any]):
        """Trigger insights generation based on interaction results"""
        
        try:
            # Check if we have enough new data for insights generation
            interaction_count = self.integration_metrics.get('total_interactions', 0)
            last_insights_generation = getattr(self, 'last_insights_generation', None)
            
            should_generate_insights = (
                # Generate insights on first run
                last_insights_generation is None or
                # Generate insights on schedule
                (datetime.now() - last_insights_generation).seconds >= self.insights_generation_interval or
                # Generate insights if critical alerts detected
                any(alert.get('severity') == 'critical' for alert in learning_results.get('alerts', []))
            )
            
            if should_generate_insights:
                # Generate comprehensive insights
                new_insights = await self.insights_generator.generate_comprehensive_insights(
                    time_period_days=7  # Weekly insights
                )
                
                # Store insights
                self.generated_insights.extend(new_insights)
                
                # Add high-priority insights to learning results
                high_priority_insights = [
                    insight for insight in new_insights
                    if insight.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]
                ]
                
                learning_results['insights_generated'].extend([
                    {
                        'insight_id': insight.insight_id,
                        'title': insight.title,
                        'priority': insight.priority.value,
                        'recommendations': insight.recommendations[:3]  # Top 3 recommendations
                    }
                    for insight in high_priority_insights
                ])
                
                learning_results['components_processed'].append('insights_generation')
                self.last_insights_generation = datetime.now()
                
                self.logger.info(f"Generated {len(new_insights)} insights, {len(high_priority_insights)} high-priority")
        
        except Exception as e:
            self.logger.error(f"Error in insights generation: {str(e)}")
    
    async def _collect_system_state(self) -> SystemState:
        """Collect current system state for optimization"""
        
        # Simulate system state collection
        # In real implementation, this would interface with actual system components
        
        return SystemState(
            timestamp=datetime.now(),
            active_agents={
                'diagnosis_agent': {
                    'response_time': 1.2,
                    'accuracy': 0.85,
                    'user_satisfaction': 0.78,
                    'resource_usage': 0.45
                },
                'therapy_agent': {
                    'response_time': 0.9,
                    'accuracy': 0.88,
                    'user_satisfaction': 0.82,
                    'resource_usage': 0.38
                }
            },
            user_interactions={
                'total_sessions_today': 150,
                'average_session_duration': 18.5,
                'user_satisfaction_avg': 7.2
            },
            resource_utilization={
                'cpu': 0.65,
                'memory': 0.58,
                'gpu': 0.42,
                'network': 0.23
            },
            performance_metrics={
                'avg_response_time': 1.05,
                'throughput': 125.5,
                'error_rate': 0.02,
                'availability': 0.998
            },
            workflow_efficiency={
                'intake': 0.92,
                'assessment': 0.87,
                'intervention': 0.89,
                'follow_up': 0.84
            },
            user_satisfaction_scores={f'user_{i}': float(7 + i % 4) for i in range(1, 21)},
            intervention_success_rates={
                'CBT': 0.82,
                'DBT': 0.78,
                'supportive': 0.75,
                'mindfulness': 0.85
            }
        )
    
    async def _get_recent_outcomes(self) -> List[InterventionOutcome]:
        """Get recent intervention outcomes for analysis"""
        
        # Get recent outcomes from the system
        # This is a simplified implementation
        
        outcomes = []
        
        # Collect from feedback processor
        recent_feedback = list(self.feedback_processor.feedback_entries.values())[-20:]  # Last 20
        
        for feedback_entry in recent_feedback:
            outcome = InterventionOutcome(
                intervention_id=feedback_entry.feedback_id,
                user_id=feedback_entry.user_id,
                intervention_type=feedback_entry.intervention_type or 'general',
                intervention_content='System intervention',
                context=feedback_entry.context_at_feedback or {},
                timestamp=feedback_entry.timestamp,
                user_response=feedback_entry.text_content or '',
                engagement_score=feedback_entry.rating_value / 10.0 if feedback_entry.rating_value else 0.5,
                effectiveness_score=feedback_entry.rating_value / 10.0 if feedback_entry.rating_value else 0.5
            )
            outcomes.append(outcome)
        
        return outcomes
    
    async def _apply_optimization_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an optimization action to the system"""
        
        try:
            action_type = action.get('action_type', '')
            target_agent = action.get('target_agent', '')
            parameters = action.get('parameters', {})
            
            self.logger.info(f"Applying optimization action: {action_type} to {target_agent}")
            
            # Simulate action application
            # In real implementation, this would modify actual system parameters
            
            if action_type == 'adjust_learning_rate':
                # Simulate learning rate adjustment
                new_lr = parameters.get('new_learning_rate', 0.001)
                self.logger.debug(f"Adjusted learning rate to {new_lr}")
                
            elif action_type == 'optimize_agent_allocation':
                # Simulate agent workload reallocation
                target_load = parameters.get('target_load', 0.5)
                self.logger.debug(f"Adjusted {target_agent} load to {target_load}")
                
            elif action_type == 'tune_intervention_parameters':
                # Simulate intervention parameter tuning
                weight_adjustments = parameters.get('weight_adjustments', {})
                self.logger.debug(f"Adjusted intervention weights: {weight_adjustments}")
            
            return {
                'success': True,
                'action_type': action_type,
                'target_agent': target_agent,
                'applied_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error applying optimization action: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'action_type': action.get('action_type', 'unknown')
            }
    
    async def _background_learning_cycle(self):
        """Background task for continuous learning and optimization"""
        
        cycle_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                await asyncio.sleep(self.learning_cycle_interval)
                cycle_count += 1
                
                self.logger.info(f"Starting background learning cycle #{cycle_count}")
                
                # Safety check: Validate system health before proceeding
                if not await self._validate_system_health():
                    self.logger.warning("System health check failed, skipping this cycle")
                    continue
                
                # Run periodic optimization with error handling
                try:
                    await self._run_optimization_cycle()
                except Exception as opt_error:
                    self.logger.error(f"Error in optimization cycle: {str(opt_error)}")
                    consecutive_errors += 1
                
                # Generate periodic reports with error handling
                try:
                    await self._generate_periodic_reports()
                except Exception as report_error:
                    self.logger.error(f"Error in report generation: {str(report_error)}")
                
                # Generate insights periodically
                if self.enable_insights_generation and cycle_count % 24 == 0:  # Every 24 cycles (daily if hourly)
                    try:
                        await self.insights_generator.generate_comprehensive_insights()
                        self.logger.info("Generated scheduled insights")
                    except Exception as insight_error:
                        self.logger.error(f"Error in insights generation: {str(insight_error)}")
                
                # Privacy compliance check
                if self.enable_privacy_protection:
                    try:
                        privacy_status = self.privacy_protection.validate_privacy_compliance()
                        if not privacy_status.get('overall_compliant', True):
                            self.logger.critical("Privacy compliance failure detected!")
                            await self._handle_privacy_compliance_failure(privacy_status)
                    except Exception as privacy_error:
                        self.logger.error(f"Error in privacy compliance check: {str(privacy_error)}")
                
                # Cleanup old data with error handling
                try:
                    await self._cleanup_old_data()
                except Exception as cleanup_error:
                    self.logger.error(f"Error in data cleanup: {str(cleanup_error)}")
                
                # Reset consecutive error count on successful cycle
                consecutive_errors = 0
                
                # System performance monitoring
                await self._monitor_system_performance()
                
                self.logger.info(f"Background learning cycle #{cycle_count} completed successfully")
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in background learning cycle #{cycle_count}: {str(e)}")
                
                # If too many consecutive errors, increase sleep time and alert
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical(f"Too many consecutive errors ({consecutive_errors}), reducing cycle frequency")
                    await asyncio.sleep(self.learning_cycle_interval * 2)  # Double the sleep time
                    consecutive_errors = 0  # Reset to try again
    
    async def _update_integration_metrics(self, learning_results: Dict[str, Any]):
        """Update integration performance metrics"""
        
        timestamp = datetime.now()
        
        # Track component processing
        components_processed = learning_results.get('components_processed', [])
        for component in components_processed:
            self.integration_metrics[f'{component}_processed'].append(timestamp)
        
        # Track insights generated
        insights_count = len(learning_results.get('insights_generated', []))
        self.integration_metrics['insights_generated'].append((timestamp, insights_count))
        
        # Track recommendations made
        recommendations_count = len(learning_results.get('recommendations', []))
        self.integration_metrics['recommendations_generated'].append((timestamp, recommendations_count))
        
        # Track alerts raised
        alerts_count = len(learning_results.get('alerts', []))
        if alerts_count > 0:
            self.integration_metrics['alerts_raised'].append((timestamp, alerts_count))
    
    async def _generate_periodic_reports(self):
        """Generate periodic reports on system learning performance"""
        
        try:
            # Generate feedback summary
            feedback_summary = await self.feedback_processor.generate_feedback_summary(
                time_period_hours=24  # Daily summary
            )
            
            # Generate system outcome report
            outcome_report = await self.outcome_tracker.generate_outcome_report(
                time_period_days=7  # Weekly report
            )
            
            # Log key metrics
            self.logger.info(f"Daily feedback processed: {feedback_summary.total_feedback_count}")
            self.logger.info(f"Weekly outcome measurements: {outcome_report['summary_statistics']}")
            
            # Store reports for analysis
            report_timestamp = datetime.now()
            self.system_performance_history.append({
                'timestamp': report_timestamp,
                'feedback_summary': asdict(feedback_summary),
                'outcome_report': outcome_report,
                'integration_metrics': self._get_integration_metrics_summary()
            })
            
            # Keep only last 30 days of reports
            cutoff_date = report_timestamp - timedelta(days=30)
            self.system_performance_history = [
                report for report in self.system_performance_history
                if report['timestamp'] >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error generating periodic reports: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=90)  # Keep 90 days
            
            # Cleanup feedback processor data
            old_feedback_ids = [
                fid for fid, entry in self.feedback_processor.feedback_entries.items()
                if entry.timestamp < cutoff_date
            ]
            
            for fid in old_feedback_ids:
                del self.feedback_processor.feedback_entries[fid]
            
            # Cleanup integration metrics
            for metric_name, metric_data in self.integration_metrics.items():
                if isinstance(metric_data, list) and len(metric_data) > 1000:
                    # Keep only recent data
                    self.integration_metrics[metric_name] = metric_data[-1000:]
            
            if old_feedback_ids:
                self.logger.info(f"Cleaned up {len(old_feedback_ids)} old feedback entries")
            
        except Exception as e:
            self.logger.error(f"Error in data cleanup: {str(e)}")
    
    def _get_integration_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of integration metrics"""
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        summary = {}
        
        for metric_name, metric_data in self.integration_metrics.items():
            if not metric_data:
                continue
            
            # Count recent entries
            if isinstance(metric_data[0], datetime):
                recent_count = len([entry for entry in metric_data if entry >= last_24h])
                summary[f'{metric_name}_last_24h'] = recent_count
            elif isinstance(metric_data[0], tuple):
                recent_entries = [entry for entry in metric_data if entry[0] >= last_24h]
                summary[f'{metric_name}_last_24h'] = len(recent_entries)
                if recent_entries:
                    summary[f'{metric_name}_avg_last_24h'] = sum(entry[1] for entry in recent_entries) / len(recent_entries)
        
        return summary
    
    async def _validate_system_health(self) -> bool:
        """Validate system health before proceeding with learning cycles"""
        
        try:
            # Check component availability
            if not self.adaptive_learning_agent:
                self.logger.error("Adaptive learning agent not available")
                return False
            
            # Check privacy protection if enabled
            if self.enable_privacy_protection:
                privacy_status = self.privacy_protection.validate_privacy_compliance()
                if not privacy_status.get('overall_compliant', True):
                    self.logger.error("Privacy compliance failed health check")
                    return False
            
            # Check data availability
            if len(self.outcome_tracker.user_profiles) == 0:
                self.logger.warning("No user profiles available for learning")
                # This is a warning, not a failure
            
            # Check memory usage (simplified)
            if len(self.generated_insights) > 10000:
                self.logger.warning("Large number of insights in memory, cleanup recommended")
                await self.insights_generator.cleanup_old_insights()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in system health validation: {str(e)}")
            return False
    
    async def _handle_privacy_compliance_failure(self, privacy_status: Dict[str, Any]):
        """Handle privacy compliance failures"""
        
        try:
            self.logger.critical("Handling privacy compliance failure")
            
            # Stop real-time learning if privacy is compromised
            if not privacy_status.get('data_retention_compliant', True):
                self.logger.critical("Data retention compliance failed - cleaning up old data immediately")
                await self._cleanup_old_data()
                
            if not privacy_status.get('privacy_budget_compliant', True):
                self.logger.critical("Privacy budget exhausted - resetting budget")
                self.privacy_protection.reset_privacy_budget()
                
            if not privacy_status.get('encryption_compliant', True):
                self.logger.critical("Encryption compliance failed - system needs immediate attention")
                # In production, this might trigger system shutdown or alerts
                
            # Generate alert
            alert = {
                'type': 'privacy_compliance_failure',
                'severity': 'critical',
                'timestamp': datetime.now().isoformat(),
                'details': privacy_status,
                'actions_taken': ['data_cleanup', 'budget_reset']
            }
            
            # Store alert for reporting
            if not hasattr(self, 'system_alerts'):
                self.system_alerts = []
            self.system_alerts.append(alert)
            
        except Exception as e:
            self.logger.error(f"Error handling privacy compliance failure: {str(e)}")
    
    async def _monitor_system_performance(self):
        """Monitor overall system performance"""
        
        try:
            current_time = datetime.now()
            
            # Collect performance metrics
            performance_data = {
                'timestamp': current_time,
                'components_active': {
                    'adaptive_learning_agent': bool(self.adaptive_learning_agent),
                    'pattern_engine': bool(self.pattern_engine),
                    'personalization_engine': bool(self.personalization_engine),
                    'feedback_processor': bool(self.feedback_processor),
                    'outcome_tracker': bool(self.outcome_tracker),
                    'privacy_protection': bool(self.privacy_protection),
                    'insights_generator': bool(self.insights_generator)
                },
                'data_statistics': {
                    'user_profiles_count': len(self.outcome_tracker.user_profiles),
                    'feedback_entries_count': len(getattr(self.feedback_processor, 'feedback_entries', {})),
                    'generated_insights_count': len(self.generated_insights),
                    'learning_updates_count': len(self.learning_updates)
                },
                'system_metrics': {
                    'optimization_cycles_completed': self.optimization_cycle_count,
                    'integration_metrics_count': len(self.integration_metrics),
                    'performance_history_count': len(self.system_performance_history)
                }
            }
            
            # Check for performance degradation
            alerts = []
            
            if performance_data['data_statistics']['user_profiles_count'] == 0:
                alerts.append("No active user profiles detected")
            
            if performance_data['data_statistics']['feedback_entries_count'] == 0:
                alerts.append("No feedback entries detected")
            
            if not all(performance_data['components_active'].values()):
                inactive_components = [
                    comp for comp, active in performance_data['components_active'].items()
                    if not active
                ]
                alerts.append(f"Inactive components detected: {', '.join(inactive_components)}")
            
            # Log performance status
            if alerts:
                self.logger.warning(f"Performance monitoring alerts: {'; '.join(alerts)}")
            else:
                self.logger.debug("System performance monitoring: All systems operational")
            
            # Store performance data
            performance_data['alerts'] = alerts
            if not hasattr(self, 'performance_monitoring_history'):
                self.performance_monitoring_history = []
            
            self.performance_monitoring_history.append(performance_data)
            
            # Keep only recent performance data
            cutoff_time = current_time - timedelta(days=7)
            self.performance_monitoring_history = [
                perf for perf in self.performance_monitoring_history
                if perf['timestamp'] >= cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error in system performance monitoring: {str(e)}")
    
    async def get_system_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the adaptive learning system"""
        
        # Get current system health status
        system_health_valid = await self._validate_system_health()
        
        return {
            'system_info': {
                'integration_active': True,
                'system_health': 'healthy' if system_health_valid else 'degraded',
                'components_initialized': [
                    'adaptive_learning_agent',
                    'pattern_recognition_engine',
                    'personalization_engine',
                    'feedback_processor',
                    'outcome_tracker',
                    'privacy_protection_system',
                    'insights_generation_system'
                ],
                'optimization_cycles_completed': self.optimization_cycle_count,
                'last_optimization_cycle': self.last_optimization_cycle.isoformat() if self.last_optimization_cycle else None,
                'background_learning_enabled': self.enable_background_optimization,
                'real_time_learning_enabled': self.enable_real_time_learning,
                'privacy_protection_enabled': self.enable_privacy_protection,
                'insights_generation_enabled': self.enable_insights_generation,
                'last_insights_generation': getattr(self, 'last_insights_generation', None).isoformat() if hasattr(self, 'last_insights_generation') and self.last_insights_generation else None
            },
            'component_status': {
                'adaptive_learning_agent': self.adaptive_learning_agent.get_status(),
                'personalization_engine': self.personalization_engine.get_personalization_summary(),
                'feedback_processor': self.feedback_processor.get_feedback_statistics(),
                'outcome_tracker': self.outcome_tracker.get_system_statistics(),
                'pattern_engine': {
                    'patterns_discovered': len(getattr(self.pattern_engine, 'discovered_patterns', [])),
                    'behavior_patterns': len(getattr(self.pattern_engine, 'user_behavior_patterns', [])),
                    'intervention_sequences': len(getattr(self.pattern_engine, 'intervention_sequences', []))
                },
                'privacy_protection': self.privacy_protection.get_privacy_report() if self.enable_privacy_protection else {'status': 'disabled'},
                'insights_generator': self.insights_generator.get_insights_dashboard() if self.enable_insights_generation else {'status': 'disabled'}
            },
            'performance_metrics': self._get_integration_metrics_summary(),
            'active_alerts': self.outcome_tracker.get_active_alerts(),
            'recent_learning_updates': len(self.learning_updates),
            'system_performance_reports': len(self.system_performance_history),
            'generated_insights': len(self.generated_insights),
            'system_alerts': len(getattr(self, 'system_alerts', [])),
            'performance_monitoring': {
                'monitoring_active': hasattr(self, 'performance_monitoring_history'),
                'recent_performance_data': len(getattr(self, 'performance_monitoring_history', []))
            }
        }
    
    async def shutdown(self):
        """Shutdown the adaptive learning system gracefully"""
        
        try:
            self.logger.info("Shutting down Adaptive Learning System")
            
            # Generate final insights report before shutdown
            if self.enable_insights_generation:
                try:
                    final_insights = await self.insights_generator.generate_comprehensive_insights()
                    if final_insights:
                        self.logger.info(f"Generated {len(final_insights)} final insights before shutdown")
                except Exception as e:
                    self.logger.warning(f"Could not generate final insights: {str(e)}")
            
            # Save all component data with error handling
            save_tasks = [
                self._save_component_data('personalization_engine'),
                self._save_component_data('feedback_processor'),
                self._save_component_data('outcome_tracker'),
                self._save_component_data('pattern_engine'),
                self._save_component_data('insights_generator'),
                self._save_component_data('privacy_protection')
            ]
            
            # Execute save tasks with error handling
            results = await asyncio.gather(*save_tasks, return_exceptions=True)
            
            # Log any save errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_name = ['personalization_engine', 'feedback_processor', 'outcome_tracker', 
                                    'pattern_engine', 'insights_generator', 'privacy_protection'][i]
                    self.logger.warning(f"Error saving {component_name}: {str(result)}")
            
            # Save adaptive learning agent models with error handling
            try:
                model_path = self.config.get('model_path', 'src/data/adaptive_learning/rl_models.pt')
                self.adaptive_learning_agent.rl_engine.save_model(model_path)
                self.logger.info("Saved adaptive learning models")
            except Exception as e:
                self.logger.warning(f"Could not save RL models: {str(e)}")
            
            # Save system state and alerts
            try:
                await self._save_system_state()
                self.logger.info("Saved system state")
            except Exception as e:
                self.logger.warning(f"Could not save system state: {str(e)}")
            
            # Final privacy compliance check
            if self.enable_privacy_protection:
                try:
                    final_privacy_status = self.privacy_protection.validate_privacy_compliance()
                    if final_privacy_status.get('overall_compliant', True):
                        self.logger.info("Privacy compliance validated at shutdown")
                    else:
                        self.logger.warning("Privacy compliance issues detected at shutdown")
                except Exception as e:
                    self.logger.warning(f"Could not validate privacy compliance at shutdown: {str(e)}")
            
            self.logger.info("Adaptive Learning System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def _save_component_data(self, component_name: str):
        """Save individual component data"""
        
        try:
            if component_name == 'personalization_engine':
                if hasattr(self.personalization_engine, 'save_models'):
                    self.personalization_engine.save_models()
                
            elif component_name == 'feedback_processor':
                # Save feedback processor state if method exists
                if hasattr(self.feedback_processor, 'save_data'):
                    self.feedback_processor.save_data()
                
            elif component_name == 'outcome_tracker':
                if hasattr(self.outcome_tracker, 'save_data'):
                    self.outcome_tracker.save_data()
                    
            elif component_name == 'pattern_engine':
                if hasattr(self.pattern_engine, 'save_models'):
                    self.pattern_engine.save_models()
                    
            elif component_name == 'insights_generator':
                # Save insights to file
                if self.generated_insights:
                    insights_data = {
                        'insights': [
                            {
                                'insight_id': insight.insight_id,
                                'title': insight.title,
                                'description': insight.description,
                                'priority': insight.priority.value,
                                'type': insight.insight_type.value,
                                'generated_at': insight.generated_at.isoformat(),
                                'implemented': insight.implemented,
                                'recommendations': insight.recommendations
                            }
                            for insight in self.generated_insights
                        ],
                        'saved_at': datetime.now().isoformat()
                    }
                    
                    # Save to data directory
                    import os
                    import json
                    data_dir = 'src/data/adaptive_learning'
                    os.makedirs(data_dir, exist_ok=True)
                    
                    with open(f'{data_dir}/insights_data.json', 'w') as f:
                        json.dump(insights_data, f, indent=2, default=str)
                
            elif component_name == 'privacy_protection':
                # Save privacy metrics and report
                if self.enable_privacy_protection:
                    privacy_report = self.privacy_protection.get_privacy_report()
                    
                    import os
                    import json
                    data_dir = 'src/data/adaptive_learning'
                    os.makedirs(data_dir, exist_ok=True)
                    
                    with open(f'{data_dir}/privacy_report.json', 'w') as f:
                        json.dump(privacy_report, f, indent=2, default=str)
            
            self.logger.debug(f"Saved {component_name} data")
            
        except Exception as e:
            self.logger.warning(f"Could not save {component_name} data: {str(e)}")
    
    async def _save_system_state(self):
        """Save overall system state"""
        
        try:
            system_state = {
                'shutdown_timestamp': datetime.now().isoformat(),
                'system_info': {
                    'optimization_cycles_completed': self.optimization_cycle_count,
                    'learning_updates_count': len(self.learning_updates),
                    'generated_insights_count': len(self.generated_insights),
                    'performance_reports_count': len(self.system_performance_history)
                },
                'system_alerts': getattr(self, 'system_alerts', []),
                'performance_monitoring_history': getattr(self, 'performance_monitoring_history', [])[-10:],  # Last 10 entries
                'integration_metrics_summary': self._get_integration_metrics_summary(),
                'configuration': {
                    'learning_cycle_interval': self.learning_cycle_interval,
                    'enable_real_time_learning': self.enable_real_time_learning,
                    'enable_background_optimization': self.enable_background_optimization,
                    'enable_privacy_protection': self.enable_privacy_protection,
                    'enable_insights_generation': self.enable_insights_generation
                }
            }
            
            # Save system state
            import os
            import json
            data_dir = 'src/data/adaptive_learning'
            os.makedirs(data_dir, exist_ok=True)
            
            with open(f'{data_dir}/system_state.json', 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {str(e)}")


# Convenience function for easy integration
async def create_adaptive_learning_system(orchestrator: AgentOrchestrator,
                                        vector_db: Optional[CentralVectorDB] = None,
                                        config: Optional[Dict[str, Any]] = None) -> AdaptiveLearningSystemIntegrator:
    """
    Create and initialize the complete adaptive learning system.
    
    Args:
        orchestrator: The main agent orchestrator
        vector_db: Optional central vector database
        config: Configuration dictionary
    
    Returns:
        Initialized AdaptiveLearningSystemIntegrator
    """
    
    integrator = AdaptiveLearningSystemIntegrator(
        orchestrator=orchestrator,
        vector_db=vector_db,
        config=config
    )
    
    success = await integrator.initialize()
    
    if not success:
        raise RuntimeError("Failed to initialize Adaptive Learning System")
    
    return integrator