"""
Friction Coordinator for Therapeutic Sub-Agents.

Orchestrates multiple therapeutic friction sub-agents, manages consensus building,
handles conflict resolution, and maintains the external interface compatibility
with the original TherapeuticFrictionAgent.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import statistics

from .base_friction_agent import BaseFrictionAgent, FrictionAgentType, friction_agent_registry
from .readiness_assessment_agent import ReadinessAssessmentAgent
from .breakthrough_detection_agent import BreakthroughDetectionAgent
from src.agents.base_agent import BaseAgent
from src.agents.therapeutic_friction_agent import ChallengeLevel, InterventionType, UserReadinessIndicator
from src.utils.logger import get_logger


class CoordinationStrategy(Enum):
    """Strategies for coordinating sub-agent assessments."""
    WEIGHTED_CONSENSUS = "weighted_consensus"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    DOMAIN_EXPERTISE = "domain_expertise"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts between sub-agents."""
    CONSERVATIVE_FALLBACK = "conservative_fallback"
    EXPERT_OVERRIDE = "expert_override"
    WEIGHTED_AVERAGE = "weighted_average"
    TEMPORAL_PRIORITY = "temporal_priority"


class FrictionCoordinator(BaseAgent):
    """
    Coordinates therapeutic friction sub-agents while maintaining
    backward compatibility with the original TherapeuticFrictionAgent interface.
    """
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the friction coordinator."""
        super().__init__(
            model=model_provider,
            name="friction_coordinator",
            role="Therapeutic Friction Coordinator",
            description="Orchestrates therapeutic friction sub-agents for comprehensive assessment"
        )
        
        self.config = config or {}
        self.logger = get_logger(self.name)
        
        # Coordination configuration
        self.coordination_strategy = CoordinationStrategy(
            self.config.get("coordination_strategy", CoordinationStrategy.WEIGHTED_CONSENSUS.value)
        )
        self.conflict_resolution = ConflictResolutionStrategy(
            self.config.get("conflict_resolution", ConflictResolutionStrategy.CONSERVATIVE_FALLBACK.value)
        )
        
        # Sub-agent management
        self.sub_agents: Dict[FrictionAgentType, BaseFrictionAgent] = {}
        self.agent_weights: Dict[FrictionAgentType, float] = {
            FrictionAgentType.READINESS_ASSESSMENT: 0.3,
            FrictionAgentType.BREAKTHROUGH_DETECTION: 0.25,
            FrictionAgentType.RELATIONSHIP_MONITORING: 0.2,
            FrictionAgentType.INTERVENTION_STRATEGY: 0.15,
            FrictionAgentType.PROGRESS_TRACKING: 0.1
        }
        
        # Coordination metrics
        self.coordination_history = []
        self.conflict_resolution_history = []
        self.performance_metrics = {}
        
        # Compatibility layer for original interface
        self.user_progress = None
        self.therapeutic_relationship = None
        self.intervention_history = []
        self.outcome_metrics = {}
        
        # Initialize sub-agents
        self._initialize_sub_agents()
    
    def _initialize_sub_agents(self):
        """Initialize and register therapeutic friction sub-agents."""
        try:
            # Create sub-agents
            self.sub_agents[FrictionAgentType.READINESS_ASSESSMENT] = ReadinessAssessmentAgent(
                model_provider=self.llm,
                config=self.config.get("readiness_assessment", {})
            )
            
            self.sub_agents[FrictionAgentType.BREAKTHROUGH_DETECTION] = BreakthroughDetectionAgent(
                model_provider=self.llm,
                config=self.config.get("breakthrough_detection", {})
            )
            
            # Register sub-agents in global registry
            for agent in self.sub_agents.values():
                friction_agent_registry.register_agent(agent)
                
                # Set up integration callbacks
                agent.register_integration_callback(
                    "coordinator_update",
                    self._handle_sub_agent_update
                )
            
            self.logger.info(f"Initialized {len(self.sub_agents)} therapeutic friction sub-agents")
            
        except Exception as e:
            self.logger.error(f"Error initializing sub-agents: {str(e)}")
            raise
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input through coordinated sub-agent analysis.
        
        Maintains compatibility with original TherapeuticFrictionAgent interface.
        """
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Phase 1: Parallel sub-agent execution
            sub_agent_results = await self._execute_sub_agents_parallel(user_input, context)
            
            # Phase 2: Coordination and consensus building
            coordinated_assessment = await self._coordinate_assessments(
                sub_agent_results, user_input, context
            )
            
            # Phase 3: Conflict resolution if needed
            if coordinated_assessment.get("conflicts_detected"):
                coordinated_assessment = await self._resolve_conflicts(
                    coordinated_assessment, sub_agent_results, context
                )
            
            # Phase 4: Generate unified response
            unified_response = await self._generate_unified_response(
                coordinated_assessment, sub_agent_results, user_input, context
            )
            
            # Phase 5: Update compatibility layer
            self._update_compatibility_layer(unified_response, sub_agent_results)
            
            # Record coordination metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._record_coordination_metrics(
                sub_agent_results, coordinated_assessment, processing_time
            )
            
            # Return result in original format for compatibility
            return self._format_compatible_response(unified_response, processing_time)
            
        except Exception as e:
            self.logger.error(f"Error in friction coordination: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_sub_agents_parallel(self, user_input: str, 
                                         context: Dict[str, Any]) -> Dict[FrictionAgentType, Dict[str, Any]]:
        """Execute sub-agents in parallel for efficiency."""
        tasks = {}
        
        for agent_type, agent in self.sub_agents.items():
            task = asyncio.create_task(
                agent.process(user_input, context),
                name=f"{agent_type.value}_task"
            )
            tasks[agent_type] = task
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=self.config.get("sub_agent_timeout", 30.0)
            )
            
            # Map results back to agent types
            sub_agent_results = {}
            for i, (agent_type, task) in enumerate(tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Sub-agent {agent_type.value} failed: {str(result)}")
                    sub_agent_results[agent_type] = {
                        'error': str(result),
                        'agent_type': agent_type.value,
                        'confidence': 0.0
                    }
                else:
                    sub_agent_results[agent_type] = result
            
            return sub_agent_results
            
        except asyncio.TimeoutError:
            self.logger.error("Sub-agent execution timeout")
            # Cancel remaining tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            
            # Return partial results
            sub_agent_results = {}
            for agent_type, task in tasks.items():
                if task.done() and not task.exception():
                    sub_agent_results[agent_type] = task.result()
                else:
                    sub_agent_results[agent_type] = {
                        'error': 'timeout',
                        'agent_type': agent_type.value,
                        'confidence': 0.0
                    }
            
            return sub_agent_results
    
    async def _coordinate_assessments(self, sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                    user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate sub-agent assessments using configured strategy."""
        coordination_data = {
            "strategy": self.coordination_strategy.value,
            "timestamp": datetime.now().isoformat(),
            "participating_agents": list(sub_agent_results.keys()),
            "conflicts_detected": False,
            "consensus_strength": 0.0
        }
        
        # Filter valid results
        valid_results = {
            agent_type: result for agent_type, result in sub_agent_results.items()
            if not result.get('error') and result.get('is_valid', True)
        }
        
        if not valid_results:
            coordination_data["error"] = "No valid sub-agent results available"
            return coordination_data
        
        # Apply coordination strategy
        if self.coordination_strategy == CoordinationStrategy.WEIGHTED_CONSENSUS:
            coordinated_result = await self._weighted_consensus_coordination(valid_results)
        elif self.coordination_strategy == CoordinationStrategy.HIGHEST_CONFIDENCE:
            coordinated_result = await self._highest_confidence_coordination(valid_results)
        elif self.coordination_strategy == CoordinationStrategy.MAJORITY_VOTE:
            coordinated_result = await self._majority_vote_coordination(valid_results)
        elif self.coordination_strategy == CoordinationStrategy.DOMAIN_EXPERTISE:
            coordinated_result = await self._domain_expertise_coordination(valid_results, user_input, context)
        else:
            # Default to weighted consensus
            coordinated_result = await self._weighted_consensus_coordination(valid_results)
        
        coordination_data.update(coordinated_result)
        return coordination_data
    
    async def _weighted_consensus_coordination(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate using weighted consensus of sub-agent assessments."""
        weighted_scores = {}
        confidence_scores = []
        
        # Extract key assessments from each agent
        for agent_type, result in valid_results.items():
            weight = self.agent_weights.get(agent_type, 0.1)
            confidence = result.get('confidence', 0.0)
            confidence_scores.append(confidence * weight)
            
            # Extract agent-specific key metrics
            if agent_type == FrictionAgentType.READINESS_ASSESSMENT:
                assessment = result.get('assessment', {})
                primary_readiness = assessment.get('primary_readiness', 'ambivalent')
                readiness_score = assessment.get('readiness_score', 0.0)
                
                weighted_scores['readiness_assessment'] = {
                    'primary_readiness': primary_readiness,
                    'score': readiness_score * weight,
                    'weight': weight,
                    'confidence': confidence
                }
            
            elif agent_type == FrictionAgentType.BREAKTHROUGH_DETECTION:
                assessment = result.get('assessment', {})
                breakthrough_detected = assessment.get('breakthrough_detected', False)
                breakthrough_potential = assessment.get('breakthrough_potential', {}).get('overall_potential', 0.0)
                
                weighted_scores['breakthrough_detection'] = {
                    'breakthrough_detected': breakthrough_detected,
                    'potential': breakthrough_potential * weight,
                    'weight': weight,
                    'confidence': confidence
                }
        
        # Calculate consensus metrics
        overall_confidence = sum(confidence_scores) if confidence_scores else 0.0
        consensus_strength = self._calculate_consensus_strength(valid_results)
        
        # Detect conflicts
        conflicts = self._detect_assessment_conflicts(valid_results)
        
        return {
            "weighted_scores": weighted_scores,
            "overall_confidence": overall_confidence,
            "consensus_strength": consensus_strength,
            "conflicts_detected": len(conflicts) > 0,
            "conflicts": conflicts,
            "participating_agents": len(valid_results)
        }
    
    async def _highest_confidence_coordination(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Use assessment from agent with highest confidence."""
        if not valid_results:
            return {"error": "No valid results for highest confidence coordination"}
        
        # Find agent with highest confidence
        highest_confidence_agent = max(
            valid_results.items(),
            key=lambda x: x[1].get('confidence', 0.0)
        )
        
        agent_type, result = highest_confidence_agent
        
        return {
            "selected_agent": agent_type.value,
            "selected_confidence": result.get('confidence', 0.0),
            "selected_assessment": result.get('assessment', {}),
            "consensus_strength": 1.0,  # Single agent decision
            "conflicts_detected": False
        }
    
    async def _majority_vote_coordination(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Use majority vote coordination for discrete decisions."""
        # Extract key binary decisions
        readiness_votes = {}
        breakthrough_votes = {"detected": 0, "not_detected": 0}
        
        for agent_type, result in valid_results.items():
            assessment = result.get('assessment', {})
            
            if agent_type == FrictionAgentType.READINESS_ASSESSMENT:
                readiness = assessment.get('primary_readiness', 'ambivalent')
                readiness_votes[readiness] = readiness_votes.get(readiness, 0) + 1
            
            elif agent_type == FrictionAgentType.BREAKTHROUGH_DETECTION:
                detected = assessment.get('breakthrough_detected', False)
                if detected:
                    breakthrough_votes["detected"] += 1
                else:
                    breakthrough_votes["not_detected"] += 1
        
        # Determine majority decisions
        majority_readiness = max(readiness_votes.items(), key=lambda x: x[1])[0] if readiness_votes else 'ambivalent'
        majority_breakthrough = breakthrough_votes["detected"] > breakthrough_votes["not_detected"]
        
        return {
            "majority_readiness": majority_readiness,
            "majority_breakthrough": majority_breakthrough,
            "readiness_votes": readiness_votes,
            "breakthrough_votes": breakthrough_votes,
            "consensus_strength": self._calculate_majority_strength(readiness_votes, breakthrough_votes),
            "conflicts_detected": self._detect_majority_conflicts(readiness_votes, breakthrough_votes)
        }
    
    async def _domain_expertise_coordination(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]],
                                           user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use domain expertise priority for coordination."""
        # Determine primary domain based on content analysis
        primary_domain = self._determine_primary_domain(user_input, context)
        
        # Get assessment from domain expert
        if primary_domain in valid_results:
            primary_result = valid_results[primary_domain]
            primary_assessment = primary_result.get('assessment', {})
        else:
            # Fall back to highest confidence if domain expert not available
            return await self._highest_confidence_coordination(valid_results)
        
        # Get supporting assessments from other agents
        supporting_assessments = {
            agent_type: result.get('assessment', {})
            for agent_type, result in valid_results.items()
            if agent_type != primary_domain
        }
        
        return {
            "primary_domain": primary_domain.value,
            "primary_assessment": primary_assessment,
            "supporting_assessments": supporting_assessments,
            "domain_confidence": primary_result.get('confidence', 0.0),
            "consensus_strength": self._calculate_domain_consensus_strength(primary_result, supporting_assessments),
            "conflicts_detected": self._detect_domain_conflicts(primary_result, supporting_assessments)
        }
    
    async def _resolve_conflicts(self, coordinated_assessment: Dict[str, Any],
                               sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between sub-agent assessments."""
        conflicts = coordinated_assessment.get("conflicts", [])
        
        if not conflicts:
            return coordinated_assessment
        
        resolution_data = {
            "resolution_strategy": self.conflict_resolution.value,
            "conflicts_resolved": [],
            "resolution_timestamp": datetime.now().isoformat()
        }
        
        # Apply conflict resolution strategy
        if self.conflict_resolution == ConflictResolutionStrategy.CONSERVATIVE_FALLBACK:
            resolved_assessment = await self._conservative_fallback_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
        elif self.conflict_resolution == ConflictResolutionStrategy.EXPERT_OVERRIDE:
            resolved_assessment = await self._expert_override_resolution(
                coordinated_assessment, sub_agent_results, conflicts, context
            )
        elif self.conflict_resolution == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            resolved_assessment = await self._weighted_average_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
        elif self.conflict_resolution == ConflictResolutionStrategy.TEMPORAL_PRIORITY:
            resolved_assessment = await self._temporal_priority_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
        else:
            # Default to conservative fallback
            resolved_assessment = await self._conservative_fallback_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
        
        # Record conflict resolution
        self.conflict_resolution_history.append({
            "timestamp": datetime.now().isoformat(),
            "conflicts": conflicts,
            "resolution_strategy": self.conflict_resolution.value,
            "resolution_data": resolution_data
        })
        
        coordinated_assessment.update(resolved_assessment)
        coordinated_assessment["conflict_resolution"] = resolution_data
        coordinated_assessment["conflicts_detected"] = False  # Mark as resolved
        
        return coordinated_assessment
    
    async def _conservative_fallback_resolution(self, coordinated_assessment: Dict[str, Any],
                                              sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                              conflicts: List[str]) -> Dict[str, Any]:
        """Use conservative fallback for conflict resolution."""
        # Default to most conservative assessments
        conservative_resolution = {
            "readiness_level": UserReadinessIndicator.AMBIVALENT.value,
            "challenge_level": ChallengeLevel.GENTLE_INQUIRY.value,
            "breakthrough_detected": False,
            "intervention_type": InterventionType.SOCRATIC_QUESTIONING.value,
            "confidence_adjustment": -0.2  # Reduce confidence due to conflicts
        }
        
        return {
            "resolved_assessment": conservative_resolution,
            "resolution_reason": "Conservative fallback due to conflicts"
        }
    
    async def _expert_override_resolution(self, coordinated_assessment: Dict[str, Any],
                                        sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                        conflicts: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use expert agent override for conflict resolution."""
        # Determine which agent is the expert for the specific conflict
        primary_domain = self._determine_primary_domain_for_conflicts(conflicts, context)
        
        if primary_domain and primary_domain in sub_agent_results:
            expert_assessment = sub_agent_results[primary_domain].get('assessment', {})
            return {
                "resolved_assessment": expert_assessment,
                "expert_agent": primary_domain.value,
                "resolution_reason": f"Expert override by {primary_domain.value}"
            }
        else:
            # Fall back to conservative resolution
            return await self._conservative_fallback_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
    
    async def _weighted_average_resolution(self, coordinated_assessment: Dict[str, Any],
                                         sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                         conflicts: List[str]) -> Dict[str, Any]:
        """Use weighted average for conflict resolution."""
        # Extract numeric values and compute weighted averages
        numeric_assessments = {}
        
        for agent_type, result in sub_agent_results.items():
            if result.get('error'):
                continue
                
            weight = self.agent_weights.get(agent_type, 0.1)
            confidence = result.get('confidence', 0.0)
            
            # Extract numeric values from assessments
            assessment = result.get('assessment', {})
            
            # Convert readiness to numeric
            if 'primary_readiness' in assessment:
                readiness_numeric = self._readiness_to_numeric(assessment['primary_readiness'])
                if 'readiness' not in numeric_assessments:
                    numeric_assessments['readiness'] = []
                numeric_assessments['readiness'].append((readiness_numeric, weight * confidence))
        
        # Calculate weighted averages
        averaged_values = {}
        for metric, values in numeric_assessments.items():
            if values:
                weighted_sum = sum(value * weight for value, weight in values)
                total_weight = sum(weight for _, weight in values)
                if total_weight > 0:
                    averaged_values[metric] = weighted_sum / total_weight
        
        # Convert back to categorical values
        resolved_assessment = {}
        if 'readiness' in averaged_values:
            resolved_assessment['readiness_level'] = self._numeric_to_readiness(averaged_values['readiness'])
        
        return {
            "resolved_assessment": resolved_assessment,
            "averaged_values": averaged_values,
            "resolution_reason": "Weighted average of conflicting assessments"
        }
    
    async def _temporal_priority_resolution(self, coordinated_assessment: Dict[str, Any],
                                          sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                          conflicts: List[str]) -> Dict[str, Any]:
        """Use temporal priority for conflict resolution (most recent assessment wins)."""
        # Find most recent assessment
        most_recent_agent = None
        most_recent_time = None
        
        for agent_type, result in sub_agent_results.items():
            if result.get('error'):
                continue
                
            timestamp_str = result.get('metadata', {}).get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if most_recent_time is None or timestamp > most_recent_time:
                        most_recent_time = timestamp
                        most_recent_agent = agent_type
                except:
                    continue
        
        if most_recent_agent and most_recent_agent in sub_agent_results:
            recent_assessment = sub_agent_results[most_recent_agent].get('assessment', {})
            return {
                "resolved_assessment": recent_assessment,
                "temporal_priority_agent": most_recent_agent.value,
                "resolution_reason": f"Temporal priority to {most_recent_agent.value}"
            }
        else:
            # Fall back to conservative resolution
            return await self._conservative_fallback_resolution(
                coordinated_assessment, sub_agent_results, conflicts
            )
    
    async def _generate_unified_response(self, coordinated_assessment: Dict[str, Any],
                                       sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                       user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unified response from coordinated assessments."""
        # Extract key decisions from coordination
        unified_decisions = self._extract_unified_decisions(coordinated_assessment, sub_agent_results)
        
        # Generate response strategy
        response_strategy = await self._generate_response_strategy(
            unified_decisions, user_input, context
        )
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            coordinated_assessment, sub_agent_results
        )
        
        # Generate recommendations
        recommendations = self._generate_coordinated_recommendations(
            unified_decisions, coordinated_assessment, sub_agent_results
        )
        
        unified_response = {
            "unified_decisions": unified_decisions,
            "response_strategy": response_strategy,
            "overall_metrics": overall_metrics,
            "recommendations": recommendations,
            "coordination_summary": {
                "strategy_used": coordinated_assessment.get("strategy"),
                "consensus_strength": coordinated_assessment.get("consensus_strength", 0.0),
                "conflicts_resolved": coordinated_assessment.get("conflicts_detected", False),
                "participating_agents": coordinated_assessment.get("participating_agents", [])
            },
            "sub_agent_contributions": self._summarize_sub_agent_contributions(sub_agent_results)
        }
        
        return unified_response
    
    def _extract_unified_decisions(self, coordinated_assessment: Dict[str, Any],
                                 sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract unified decisions from coordinated assessment."""
        decisions = {}
        
        # Extract based on coordination strategy
        if "weighted_scores" in coordinated_assessment:
            # Weighted consensus results
            readiness_data = coordinated_assessment["weighted_scores"].get("readiness_assessment", {})
            breakthrough_data = coordinated_assessment["weighted_scores"].get("breakthrough_detection", {})
            
            decisions["user_readiness"] = readiness_data.get("primary_readiness", "ambivalent")
            decisions["breakthrough_detected"] = breakthrough_data.get("breakthrough_detected", False)
            
        elif "majority_readiness" in coordinated_assessment:
            # Majority vote results
            decisions["user_readiness"] = coordinated_assessment["majority_readiness"]
            decisions["breakthrough_detected"] = coordinated_assessment["majority_breakthrough"]
            
        elif "primary_assessment" in coordinated_assessment:
            # Domain expertise results
            primary = coordinated_assessment["primary_assessment"]
            decisions["user_readiness"] = primary.get("primary_readiness", "ambivalent")
            decisions["breakthrough_detected"] = primary.get("breakthrough_detected", False)
            
        elif "resolved_assessment" in coordinated_assessment:
            # Conflict resolution results
            resolved = coordinated_assessment["resolved_assessment"]
            decisions.update(resolved)
        
        # Determine challenge level and intervention type based on readiness
        readiness = decisions.get("user_readiness", "ambivalent")
        decisions["challenge_level"] = self._determine_challenge_level_from_readiness(readiness)
        decisions["intervention_type"] = self._determine_intervention_type_from_readiness(readiness)
        
        return decisions
    
    async def _generate_response_strategy(self, unified_decisions: Dict[str, Any],
                                        user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response strategy based on unified decisions."""
        strategy = {
            "approach": "coordinated_friction",
            "primary_focus": unified_decisions.get("user_readiness", "ambivalent"),
            "challenge_level": unified_decisions.get("challenge_level", "gentle_inquiry"),
            "intervention_type": unified_decisions.get("intervention_type", "socratic_questioning"),
            "breakthrough_focus": unified_decisions.get("breakthrough_detected", False)
        }
        
        # Add strategy components based on decisions
        if unified_decisions.get("breakthrough_detected"):
            strategy["components"] = [
                "breakthrough_consolidation",
                "insight_expansion",
                "integration_support"
            ]
        else:
            readiness = unified_decisions.get("user_readiness", "ambivalent")
            if readiness in ["resistant", "defensive"]:
                strategy["components"] = [
                    "validation",
                    "trust_building",
                    "gentle_exploration"
                ]
            elif readiness in ["open", "motivated"]:
                strategy["components"] = [
                    "skill_building",
                    "behavioral_experiments",
                    "pattern_exploration"
                ]
            else:  # ambivalent
                strategy["components"] = [
                    "ambivalence_exploration",
                    "motivational_enhancement",
                    "gentle_challenges"
                ]
        
        return strategy
    
    def _calculate_overall_metrics(self, coordinated_assessment: Dict[str, Any],
                                 sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall coordination metrics."""
        # Confidence metrics
        confidences = [result.get('confidence', 0.0) for result in sub_agent_results.values() 
                      if not result.get('error')]
        
        overall_confidence = statistics.mean(confidences) if confidences else 0.0
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        
        # Consensus metrics
        consensus_strength = coordinated_assessment.get("consensus_strength", 0.0)
        
        # Performance metrics
        successful_agents = len([r for r in sub_agent_results.values() if not r.get('error')])
        total_agents = len(sub_agent_results)
        success_rate = successful_agents / total_agents if total_agents > 0 else 0.0
        
        return {
            "overall_confidence": overall_confidence,
            "confidence_variance": confidence_variance,
            "consensus_strength": consensus_strength,
            "agent_success_rate": success_rate,
            "successful_agents": successful_agents,
            "total_agents": total_agents,
            "coordination_quality": min(1.0, overall_confidence * consensus_strength * success_rate)
        }
    
    def _generate_coordinated_recommendations(self, unified_decisions: Dict[str, Any],
                                            coordinated_assessment: Dict[str, Any],
                                            sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on coordinated assessment."""
        recommendations = []
        
        # Add decision-based recommendations
        readiness = unified_decisions.get("user_readiness", "ambivalent")
        challenge_level = unified_decisions.get("challenge_level", "gentle_inquiry")
        
        if readiness == "resistant":
            recommendations.extend([
                "Focus on validation and building therapeutic alliance",
                "Avoid challenging interventions until trust is established",
                "Use motivational interviewing techniques"
            ])
        elif readiness == "breakthrough_ready":
            recommendations.extend([
                "Leverage this moment for maximum therapeutic impact",
                "Use intensive interventions to consolidate insights",
                "Support integration of breakthrough realizations"
            ])
        
        # Add coordination-specific recommendations
        consensus_strength = coordinated_assessment.get("consensus_strength", 0.0)
        if consensus_strength < 0.5:
            recommendations.append("Consider additional assessment due to low consensus among evaluations")
        
        conflicts_detected = coordinated_assessment.get("conflicts_detected", False)
        if conflicts_detected:
            recommendations.append("Monitor for conflicting therapeutic indicators and adjust approach as needed")
        
        # Add sub-agent specific recommendations
        for agent_type, result in sub_agent_results.items():
            agent_recommendations = result.get('assessment', {}).get('recommendations', [])
            if agent_recommendations:
                recommendations.extend(agent_recommendations[:2])  # Limit to prevent overwhelming
        
        return list(set(recommendations))  # Remove duplicates
    
    def _format_compatible_response(self, unified_response: Dict[str, Any], 
                                  processing_time: float) -> Dict[str, Any]:
        """Format response for compatibility with original TherapeuticFrictionAgent interface."""
        unified_decisions = unified_response.get("unified_decisions", {})
        response_strategy = unified_response.get("response_strategy", {})
        overall_metrics = unified_response.get("overall_metrics", {})
        
        # Create compatible response format
        compatible_response = {
            # Original interface fields
            "response_strategy": response_strategy,
            "user_readiness": unified_decisions.get("user_readiness", "ambivalent"),
            "challenge_level": unified_decisions.get("challenge_level", "gentle_inquiry"),
            "intervention_type": unified_decisions.get("intervention_type", "socratic_questioning"),
            "breakthrough_detected": unified_decisions.get("breakthrough_detected", False),
            "therapeutic_relationship": self._generate_compatible_relationship_data(overall_metrics),
            "progress_metrics": self._generate_compatible_progress_metrics(overall_metrics),
            "friction_recommendation": self._generate_friction_recommendation(
                unified_decisions.get("challenge_level", "gentle_inquiry")
            ),
            
            # Enhanced coordination data
            "coordination_data": {
                "consensus_strength": overall_metrics.get("consensus_strength", 0.0),
                "agent_success_rate": overall_metrics.get("agent_success_rate", 0.0),
                "coordination_quality": overall_metrics.get("coordination_quality", 0.0),
                "sub_agent_contributions": unified_response.get("sub_agent_contributions", {})
            },
            
            # Context updates for downstream agents
            "context_updates": {
                "therapeutic_friction": {
                    "coordination_approach": "sub_agent_orchestration",
                    "readiness": unified_decisions.get("user_readiness", "ambivalent"),
                    "challenge_level": unified_decisions.get("challenge_level", "gentle_inquiry"),
                    "relationship_quality": overall_metrics.get("overall_confidence", 0.5),
                    "breakthrough_potential": 1.0 if unified_decisions.get("breakthrough_detected") else 0.3,
                    "coordination_quality": overall_metrics.get("coordination_quality", 0.0)
                }
            },
            
            # Processing metadata
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name,
                "processing_time": processing_time,
                "coordination_strategy": self.coordination_strategy.value,
                "sub_agents_used": len(self.sub_agents)
            }
        }
        
        return compatible_response
    
    # Utility methods for coordination
    
    def _calculate_consensus_strength(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]]) -> float:
        """Calculate strength of consensus among sub-agents."""
        if len(valid_results) < 2:
            return 1.0  # Perfect consensus with single agent
        
        confidences = [result.get('confidence', 0.0) for result in valid_results.values()]
        
        # High consensus when confidences are both high and similar
        mean_confidence = statistics.mean(confidences)
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        
        consensus_strength = mean_confidence * (1.0 - min(1.0, confidence_variance))
        return max(0.0, min(1.0, consensus_strength))
    
    def _detect_assessment_conflicts(self, valid_results: Dict[FrictionAgentType, Dict[str, Any]]) -> List[str]:
        """Detect conflicts between sub-agent assessments."""
        conflicts = []
        
        # Extract assessments for comparison
        readiness_assessments = []
        breakthrough_assessments = []
        
        for agent_type, result in valid_results.items():
            assessment = result.get('assessment', {})
            
            if agent_type == FrictionAgentType.READINESS_ASSESSMENT:
                readiness = assessment.get('primary_readiness')
                if readiness:
                    readiness_assessments.append((agent_type, readiness))
            
            elif agent_type == FrictionAgentType.BREAKTHROUGH_DETECTION:
                breakthrough = assessment.get('breakthrough_detected', False)
                breakthrough_assessments.append((agent_type, breakthrough))
        
        # Check for readiness conflicts (simplified)
        if len(readiness_assessments) > 1:
            readiness_values = [r[1] for r in readiness_assessments]
            if len(set(readiness_values)) > 1:
                conflicts.append(f"Readiness assessment conflict: {readiness_values}")
        
        # Check for breakthrough conflicts
        if len(breakthrough_assessments) > 1:
            breakthrough_values = [b[1] for b in breakthrough_assessments]
            if len(set(breakthrough_values)) > 1:
                conflicts.append(f"Breakthrough detection conflict: {breakthrough_values}")
        
        return conflicts
    
    def _determine_primary_domain(self, user_input: str, context: Dict[str, Any]) -> Optional[FrictionAgentType]:
        """Determine primary domain based on content analysis."""
        user_input_lower = user_input.lower()
        
        # Simple heuristics for domain detection
        if any(word in user_input_lower for word in ["ready", "willing", "can't", "won't", "resistance"]):
            return FrictionAgentType.READINESS_ASSESSMENT
        
        if any(word in user_input_lower for word in ["realize", "understand", "see", "breakthrough", "insight"]):
            return FrictionAgentType.BREAKTHROUGH_DETECTION
        
        # Default to readiness assessment as primary
        return FrictionAgentType.READINESS_ASSESSMENT
    
    def _readiness_to_numeric(self, readiness: str) -> float:
        """Convert readiness indicator to numeric value."""
        mapping = {
            "resistant": 0.0,
            "defensive": 1.0,
            "ambivalent": 2.0,
            "open": 3.0,
            "motivated": 4.0,
            "breakthrough_ready": 5.0
        }
        return mapping.get(readiness, 2.0)
    
    def _numeric_to_readiness(self, numeric: float) -> str:
        """Convert numeric value back to readiness indicator."""
        if numeric <= 0.5:
            return "resistant"
        elif numeric <= 1.5:
            return "defensive"
        elif numeric <= 2.5:
            return "ambivalent"
        elif numeric <= 3.5:
            return "open"
        elif numeric <= 4.5:
            return "motivated"
        else:
            return "breakthrough_ready"
    
    def _determine_challenge_level_from_readiness(self, readiness: str) -> str:
        """Determine challenge level based on readiness."""
        mapping = {
            "resistant": "validation_only",
            "defensive": "gentle_inquiry",
            "ambivalent": "gentle_inquiry",
            "open": "moderate_challenge",
            "motivated": "strong_challenge",
            "breakthrough_ready": "breakthrough_push"
        }
        return mapping.get(readiness, "gentle_inquiry")
    
    def _determine_intervention_type_from_readiness(self, readiness: str) -> str:
        """Determine intervention type based on readiness."""
        mapping = {
            "resistant": "strategic_resistance",
            "defensive": "socratic_questioning",
            "ambivalent": "values_clarification",
            "open": "cognitive_reframing",
            "motivated": "behavioral_experiment",
            "breakthrough_ready": "exposure_challenge"
        }
        return mapping.get(readiness, "socratic_questioning")
    
    def _generate_compatible_relationship_data(self, overall_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate therapeutic relationship data compatible with original interface."""
        return {
            "therapeutic_bond_strength": overall_metrics.get("overall_confidence", 0.5),
            "trust_level": overall_metrics.get("consensus_strength", 0.5),
            "engagement_score": overall_metrics.get("agent_success_rate", 0.5),
            "receptivity_to_challenge": overall_metrics.get("coordination_quality", 0.5)
        }
    
    def _generate_compatible_progress_metrics(self, overall_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate progress metrics compatible with original interface."""
        return {
            "session_count": len(self.coordination_history),
            "coordination_quality": overall_metrics.get("coordination_quality", 0.0),
            "consensus_strength": overall_metrics.get("consensus_strength", 0.0),
            "agent_success_rate": overall_metrics.get("agent_success_rate", 0.0)
        }
    
    def _generate_friction_recommendation(self, challenge_level: str) -> str:
        """Generate friction recommendation based on challenge level."""
        recommendations = {
            "validation_only": "Focus on safety and trust-building. No challenges recommended.",
            "gentle_inquiry": "Use curious questions and gentle exploration. Avoid direct challenges.",
            "moderate_challenge": "Balanced approach with supportive challenges. Monitor response closely.",
            "strong_challenge": "Direct challenges are appropriate. Push for growth and insight.",
            "breakthrough_push": "Maximum therapeutic leverage. Use powerful interventions for breakthrough."
        }
        return recommendations.get(challenge_level, "Assess individual needs and adjust accordingly.")
    
    def _summarize_sub_agent_contributions(self, sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize contributions from each sub-agent."""
        contributions = {}
        
        for agent_type, result in sub_agent_results.items():
            if not result.get('error'):
                contributions[agent_type.value] = {
                    "confidence": result.get('confidence', 0.0),
                    "key_findings": result.get('assessment', {}).get('key_findings', {}),
                    "processing_time": result.get('metadata', {}).get('processing_time', 0.0)
                }
            else:
                contributions[agent_type.value] = {
                    "error": result.get('error'),
                    "confidence": 0.0
                }
        
        return contributions
    
    def _record_coordination_metrics(self, sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]],
                                   coordinated_assessment: Dict[str, Any], processing_time: float):
        """Record coordination metrics for monitoring and improvement."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time,
            "sub_agent_count": len(sub_agent_results),
            "successful_agents": len([r for r in sub_agent_results.values() if not r.get('error')]),
            "consensus_strength": coordinated_assessment.get("consensus_strength", 0.0),
            "conflicts_detected": coordinated_assessment.get("conflicts_detected", False),
            "coordination_strategy": self.coordination_strategy.value,
            "conflict_resolution_used": coordinated_assessment.get("conflict_resolution") is not None
        }
        
        self.coordination_history.append(metrics)
        
        # Keep only last 100 coordination events
        if len(self.coordination_history) > 100:
            self.coordination_history = self.coordination_history[-100:]
    
    async def _handle_sub_agent_update(self, result: Dict[str, Any], context: Dict[str, Any]):
        """Handle updates from sub-agents."""
        # This method is called when sub-agents complete their processing
        # Can be used for real-time coordination or monitoring
        agent_type = result.get('agent_type')
        confidence = result.get('confidence', 0.0)
        
        self.logger.debug(f"Received update from {agent_type} with confidence {confidence}")
    
    # Compatibility methods for original interface
    
    def enhance_response(self, response: str, friction_result: Dict[str, Any]) -> str:
        """Enhance response with coordinated therapeutic friction."""
        if not friction_result:
            return response
        
        strategy = friction_result.get("response_strategy", {})
        challenge_level = friction_result.get("challenge_level", "gentle_inquiry")
        
        enhanced_parts = [response]
        
        # Add coordinated friction components
        coordination_data = friction_result.get("coordination_data", {})
        if coordination_data.get("consensus_strength", 0) > 0.7:
            enhanced_parts.append("\n**Coordinated Assessment:** Multiple therapeutic perspectives align on this approach.")
        
        # Add sub-agent contributions
        sub_agent_contributions = coordination_data.get("sub_agent_contributions", {})
        if sub_agent_contributions:
            for agent_type, contribution in sub_agent_contributions.items():
                if contribution.get("confidence", 0) > 0.8:
                    enhanced_parts.append(f"\n**{agent_type.replace('_', ' ').title()} Insight:** High confidence assessment supports this direction.")
        
        return "\n".join(enhanced_parts)
    
    def get_comprehensive_assessment(self) -> Dict[str, Any]:
        """Get comprehensive assessment including coordination metrics."""
        return {
            "coordination_approach": "sub_agent_orchestration",
            "sub_agents_active": len(self.sub_agents),
            "coordination_history": len(self.coordination_history),
            "conflict_resolution_history": len(self.conflict_resolution_history),
            "coordination_strategy": self.coordination_strategy.value,
            "conflict_resolution_strategy": self.conflict_resolution.value,
            "average_consensus_strength": statistics.mean([
                h.get("consensus_strength", 0) for h in self.coordination_history
            ]) if self.coordination_history else 0.0,
            "sub_agent_health": asyncio.create_task(friction_agent_registry.get_system_health())
        }
    
    def _update_compatibility_layer(self, unified_response: Dict[str, Any], 
                                  sub_agent_results: Dict[FrictionAgentType, Dict[str, Any]]):
        """Update compatibility layer for legacy interface support."""
        # This method updates instance variables that the original interface expects
        # Simplified implementation for now
        pass
    
    # Additional utility methods for conflict resolution and consensus building
    
    def _calculate_majority_strength(self, readiness_votes: Dict[str, int], 
                                   breakthrough_votes: Dict[str, int]) -> float:
        """Calculate strength of majority consensus."""
        total_readiness = sum(readiness_votes.values())
        total_breakthrough = sum(breakthrough_votes.values())
        
        if total_readiness == 0 and total_breakthrough == 0:
            return 0.0
        
        readiness_strength = max(readiness_votes.values()) / total_readiness if total_readiness > 0 else 0
        breakthrough_strength = max(breakthrough_votes.values()) / total_breakthrough if total_breakthrough > 0 else 0
        
        return (readiness_strength + breakthrough_strength) / 2.0
    
    def _detect_majority_conflicts(self, readiness_votes: Dict[str, int], 
                                 breakthrough_votes: Dict[str, int]) -> bool:
        """Detect conflicts in majority voting."""
        # Check if there are ties or very close votes
        if readiness_votes:
            max_readiness_votes = max(readiness_votes.values())
            tied_readiness = sum(1 for v in readiness_votes.values() if v == max_readiness_votes)
            if tied_readiness > 1:
                return True
        
        if breakthrough_votes:
            if abs(breakthrough_votes.get("detected", 0) - breakthrough_votes.get("not_detected", 0)) <= 1:
                return True
        
        return False
    
    def _calculate_domain_consensus_strength(self, primary_result: Dict[str, Any], 
                                           supporting_assessments: Dict[FrictionAgentType, Dict[str, Any]]) -> float:
        """Calculate consensus strength for domain expertise coordination."""
        primary_confidence = primary_result.get('confidence', 0.0)
        
        if not supporting_assessments:
            return primary_confidence
        
        supporting_confidences = [
            result.get('confidence', 0.0) 
            for result in supporting_assessments.values()
        ]
        
        avg_supporting_confidence = statistics.mean(supporting_confidences)
        
        # Consensus is strong when both primary and supporting agents are confident
        return (primary_confidence * 0.7 + avg_supporting_confidence * 0.3)
    
    def _detect_domain_conflicts(self, primary_result: Dict[str, Any], 
                               supporting_assessments: Dict[FrictionAgentType, Dict[str, Any]]) -> bool:
        """Detect conflicts in domain expertise coordination."""
        primary_confidence = primary_result.get('confidence', 0.0)
        
        # Conflict if primary expert has low confidence but supporting agents have high confidence
        supporting_confidences = [
            result.get('confidence', 0.0) 
            for result in supporting_assessments.values()
        ]
        
        if supporting_confidences:
            avg_supporting_confidence = statistics.mean(supporting_confidences)
            
            # Conflict detected if there's a significant confidence mismatch
            if primary_confidence < 0.5 and avg_supporting_confidence > 0.7:
                return True
            elif primary_confidence > 0.7 and avg_supporting_confidence < 0.3:
                return True
        
        return False
    
    def _determine_primary_domain_for_conflicts(self, conflicts: List[str], context: Dict[str, Any]) -> Optional[FrictionAgentType]:
        """Determine which agent should be primary for resolving specific conflicts."""
        # Analyze conflict types and determine appropriate expert
        for conflict in conflicts:
            if "readiness" in conflict.lower():
                return FrictionAgentType.READINESS_ASSESSMENT
            elif "breakthrough" in conflict.lower():
                return FrictionAgentType.BREAKTHROUGH_DETECTION
        
        # Default to readiness assessment as it's foundational
        return FrictionAgentType.READINESS_ASSESSMENT