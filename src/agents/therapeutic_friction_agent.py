"""
Therapeutic Friction Agent for Growth-Oriented Responses and Challenges.

This agent implements sophisticated therapeutic friction techniques to promote user growth
through appropriate challenges, Socratic questioning, and behavioral experiments while
maintaining therapeutic boundaries and tracking user readiness for intervention.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import random
from datetime import datetime, timedelta
import json
from enum import Enum
import math
from dataclasses import dataclass, asdict

from src.agents.base_agent import BaseAgent
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService


class ChallengeLevel(Enum):
    """Challenge intensity levels based on user readiness."""
    VALIDATION_ONLY = "validation_only"
    GENTLE_INQUIRY = "gentle_inquiry"
    MODERATE_CHALLENGE = "moderate_challenge"
    STRONG_CHALLENGE = "strong_challenge"
    BREAKTHROUGH_PUSH = "breakthrough_push"


class UserReadinessIndicator(Enum):
    """Indicators of user readiness for therapeutic challenges."""
    RESISTANT = "resistant"
    DEFENSIVE = "defensive"
    AMBIVALENT = "ambivalent"
    OPEN = "open"
    MOTIVATED = "motivated"
    BREAKTHROUGH_READY = "breakthrough_ready"


class InterventionType(Enum):
    """Types of growth-oriented interventions."""
    SOCRATIC_QUESTIONING = "socratic_questioning"
    COGNITIVE_REFRAMING = "cognitive_reframing"
    BEHAVIORAL_EXPERIMENT = "behavioral_experiment"
    EXPOSURE_CHALLENGE = "exposure_challenge"
    VALUES_CLARIFICATION = "values_clarification"
    PARADOXICAL_INTERVENTION = "paradoxical_intervention"
    STRATEGIC_RESISTANCE = "strategic_resistance"


@dataclass
class UserProgress:
    """Tracks user progress and development indicators."""
    session_count: int = 0
    challenge_acceptance_rate: float = 0.0
    insight_frequency: float = 0.0
    behavioral_change_indicators: List[str] = None
    resistance_patterns: List[str] = None
    breakthrough_moments: List[Dict[str, Any]] = None
    growth_trajectory: str = "stable"  # stable, declining, improving, breakthrough
    readiness_history: List[UserReadinessIndicator] = None
    
    def __post_init__(self):
        if self.behavioral_change_indicators is None:
            self.behavioral_change_indicators = []
        if self.resistance_patterns is None:
            self.resistance_patterns = []
        if self.breakthrough_moments is None:
            self.breakthrough_moments = []
        if self.readiness_history is None:
            self.readiness_history = []


@dataclass
class TherapeuticRelationship:
    """Monitors therapeutic alliance and relationship quality."""
    trust_level: float = 0.5
    engagement_score: float = 0.5
    receptivity_to_challenge: float = 0.3
    collaborative_spirit: float = 0.5
    emotional_safety: float = 0.7
    therapeutic_bond_strength: float = 0.5
    repair_needed: bool = False
    rupture_risk: float = 0.0


class TherapeuticFrictionAgent(BaseAgent):
    """
    Advanced Therapeutic Friction Agent for Growth-Oriented Interventions.
    
    This agent specializes in providing appropriately challenging therapeutic responses
    that promote growth through strategic friction while maintaining therapeutic alliance.
    """
    
    def __init__(self, model_provider=None):
        """Initialize the therapeutic friction agent."""
        super().__init__(
            model=model_provider,
            name="therapeutic_friction_agent",
            role="Growth-Oriented Therapeutic Challenger",
            description="An advanced therapeutic agent specializing in growth-promoting challenges and strategic friction"
        )
        
        self.technique_service = TherapeuticTechniqueService(model_provider)
        self.technique_service.initialize_vector_store()
        
        # Core tracking systems
        self.user_progress = UserProgress()
        self.therapeutic_relationship = TherapeuticRelationship()
        self.intervention_history = []
        self.outcome_metrics = {}
        
        # Readiness assessment patterns
        self.readiness_indicators = {
            UserReadinessIndicator.RESISTANT: {
                "verbal": ["won't work", "tried everything", "pointless", "don't want to", "can't change"],
                "behavioral": ["short responses", "topic avoidance", "dismissiveness"],
                "emotional": ["anger", "frustration", "hopelessness", "cynicism"]
            },
            UserReadinessIndicator.DEFENSIVE: {
                "verbal": ["but", "yes but", "I know", "already tried", "it's not my fault"],
                "behavioral": ["justification", "blame externalization", "intellectualization"],
                "emotional": ["anxiety", "shame", "vulnerability"]
            },
            UserReadinessIndicator.AMBIVALENT: {
                "verbal": ["maybe", "sometimes", "I guess", "kind of", "not sure"],
                "behavioral": ["inconsistent engagement", "mixed signals"],
                "emotional": ["confusion", "uncertainty", "conflicted"]
            },
            UserReadinessIndicator.OPEN: {
                "verbal": ["what do you think", "how can I", "tell me more", "I'm willing"],
                "behavioral": ["active listening", "question asking", "exploration"],
                "emotional": ["curiosity", "hope", "motivation"]
            },
            UserReadinessIndicator.MOTIVATED: {
                "verbal": ["I want to change", "ready to try", "what's next", "committed"],
                "behavioral": ["homework completion", "initiative taking", "self-reflection"],
                "emotional": ["determination", "optimism", "energy"]
            },
            UserReadinessIndicator.BREAKTHROUGH_READY: {
                "verbal": ["I see it now", "everything makes sense", "I understand", "ready for more"],
                "behavioral": ["pattern recognition", "insight expression", "change readiness"],
                "emotional": ["clarity", "empowerment", "transformation"]
            }
        }
        
        # Challenge strategies by readiness level
        self.challenge_strategies = {
            UserReadinessIndicator.RESISTANT: ChallengeLevel.VALIDATION_ONLY,
            UserReadinessIndicator.DEFENSIVE: ChallengeLevel.GENTLE_INQUIRY,
            UserReadinessIndicator.AMBIVALENT: ChallengeLevel.GENTLE_INQUIRY,
            UserReadinessIndicator.OPEN: ChallengeLevel.MODERATE_CHALLENGE,
            UserReadinessIndicator.MOTIVATED: ChallengeLevel.STRONG_CHALLENGE,
            UserReadinessIndicator.BREAKTHROUGH_READY: ChallengeLevel.BREAKTHROUGH_PUSH
        }
        
        # Breakthrough detection patterns
        self.breakthrough_indicators = [
            "sudden insight", "aha moment", "I never realized", "it all makes sense",
            "I see the pattern", "everything is connected", "I understand now",
            "this changes everything", "I can do this", "I feel different"
        ]
        
        # Strategic intervention templates
        self.intervention_templates = self._initialize_intervention_templates()
        
    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input with sophisticated readiness assessment and strategic friction."""
        self.user_progress.session_count += 1
        
        # Assess user readiness for challenge
        readiness = self._assess_user_readiness(user_input, context)
        self.user_progress.readiness_history.append(readiness)
        
        # Detect breakthrough moments
        breakthrough_detected = self._detect_breakthrough_moment(user_input, context)
        
        # Update therapeutic relationship
        self._update_therapeutic_relationship(user_input, readiness, context)
        
        # Determine appropriate challenge level
        challenge_level = self._determine_challenge_level(readiness, breakthrough_detected)
        
        # Select intervention type
        intervention_type = self._select_intervention_type(user_input, readiness, context)
        
        # Generate growth-oriented response strategy
        response_strategy = self._generate_growth_response_strategy(
            user_input, readiness, challenge_level, intervention_type, context
        )
        
        # Track progress indicators
        self._track_progress_indicators(user_input, response_strategy)
        
        # Update long-term outcomes
        self._update_outcome_metrics()
        
        return {
            "response_strategy": response_strategy,
            "user_readiness": readiness.value,
            "challenge_level": challenge_level.value,
            "intervention_type": intervention_type.value,
            "breakthrough_detected": breakthrough_detected,
            "therapeutic_relationship": asdict(self.therapeutic_relationship),
            "progress_metrics": self._get_progress_metrics(),
            "friction_recommendation": self._get_friction_recommendation(challenge_level),
            "context_updates": {
                "therapeutic_friction": {
                    "readiness": readiness.value,
                    "challenge_level": challenge_level.value,
                    "relationship_quality": self.therapeutic_relationship.therapeutic_bond_strength,
                    "breakthrough_potential": self._calculate_breakthrough_potential()
                }
            }
        }
    
    def enhance_response(self, response: str, friction_result: Dict[str, Any]) -> str:
        """Enhance response with strategic therapeutic friction and growth challenges."""
        if not friction_result:
            return response
        
        # Get strategy components
        strategy = friction_result.get("response_strategy", {})
        challenge_level = ChallengeLevel(friction_result.get("challenge_level", "gentle_inquiry"))
        intervention_type = InterventionType(friction_result.get("intervention_type", "socratic_questioning"))
        
        # Build enhanced response
        enhanced_parts = [response]
        
        # Add strategic friction based on challenge level
        friction_component = self._generate_friction_component(strategy, challenge_level, intervention_type)
        if friction_component:
            enhanced_parts.append(friction_component)
        
        # Add growth-oriented questions
        growth_questions = strategy.get("growth_questions", [])
        if growth_questions:
            enhanced_parts.append("\n**Growth Questions:**")
            for question in growth_questions[:2]:
                enhanced_parts.append(f"• {question}")
        
        # Add behavioral experiments
        experiments = strategy.get("behavioral_experiments", [])
        if experiments:
            enhanced_parts.append(f"\n**Growth Challenge: {experiments[0]['title']}**")
            enhanced_parts.append(experiments[0]['description'])
            enhanced_parts.append("**Try this:** " + experiments[0]['action'])
        
        # Add progress acknowledgment
        progress_note = self._generate_progress_acknowledgment(friction_result)
        if progress_note:
            enhanced_parts.append(progress_note)
        
        return "\n\n".join(enhanced_parts)
    
    def _assess_user_readiness(self, user_input: str, context: Dict[str, Any]) -> UserReadinessIndicator:
        """Assess user's readiness for therapeutic challenges using NLP analysis."""
        user_input_lower = user_input.lower()
        readiness_scores = {}
        
        # Analyze verbal indicators
        for readiness_level, indicators in self.readiness_indicators.items():
            score = 0
            
            # Check verbal patterns
            verbal_matches = sum(1 for pattern in indicators["verbal"] 
                               if pattern in user_input_lower)
            score += verbal_matches * 2
            
            # Check emotional context if available
            if context.get("emotion_analysis"):
                emotion = context["emotion_analysis"].get("primary_emotion", "").lower()
                if emotion in [e.lower() for e in indicators["emotional"]]:
                    score += 3
            
            # Length and engagement indicators
            if len(user_input) > 100:  # Detailed responses suggest openness
                if readiness_level in [UserReadinessIndicator.OPEN, UserReadinessIndicator.MOTIVATED]:
                    score += 1
            
            readiness_scores[readiness_level] = score
        
        # Determine highest scoring readiness level
        if not any(readiness_scores.values()):
            return UserReadinessIndicator.AMBIVALENT  # Default
        
        return max(readiness_scores.items(), key=lambda x: x[1])[0]
    
    def _detect_breakthrough_moment(self, user_input: str, context: Dict[str, Any]) -> bool:
        """Detect if user is experiencing a breakthrough moment."""
        user_input_lower = user_input.lower()
        
        # Check for breakthrough language patterns
        breakthrough_detected = any(indicator in user_input_lower 
                                  for indicator in self.breakthrough_indicators)
        
        # Check for insight indicators in emotional context
        if context.get("emotion_analysis"):
            insight_emotions = ["clarity", "understanding", "empowerment", "relief"]
            emotion = context["emotion_analysis"].get("primary_emotion", "").lower()
            if emotion in insight_emotions:
                breakthrough_detected = True
        
        # Store breakthrough moments
        if breakthrough_detected:
            breakthrough_data = {
                "timestamp": datetime.now().isoformat(),
                "trigger_input": user_input,
                "context": context.get("emotion_analysis", {}),
                "session_count": self.user_progress.session_count
            }
            self.user_progress.breakthrough_moments.append(breakthrough_data)
        
        return breakthrough_detected
    
    def _update_therapeutic_relationship(self, user_input: str, readiness: UserReadinessIndicator, 
                                       context: Dict[str, Any]) -> None:
        """Update therapeutic relationship metrics based on interaction."""
        # Update trust based on openness
        if readiness in [UserReadinessIndicator.OPEN, UserReadinessIndicator.MOTIVATED]:
            self.therapeutic_relationship.trust_level = min(1.0, 
                self.therapeutic_relationship.trust_level + 0.02)
        elif readiness == UserReadinessIndicator.RESISTANT:
            self.therapeutic_relationship.trust_level = max(0.1, 
                self.therapeutic_relationship.trust_level - 0.01)
        
        # Update engagement based on response length and depth
        engagement_boost = min(0.03, len(user_input) / 2000)
        self.therapeutic_relationship.engagement_score = min(1.0,
            self.therapeutic_relationship.engagement_score + engagement_boost)
        
        # Update receptivity to challenge
        if readiness in [UserReadinessIndicator.MOTIVATED, UserReadinessIndicator.BREAKTHROUGH_READY]:
            self.therapeutic_relationship.receptivity_to_challenge = min(1.0,
                self.therapeutic_relationship.receptivity_to_challenge + 0.05)
        elif readiness in [UserReadinessIndicator.RESISTANT, UserReadinessIndicator.DEFENSIVE]:
            self.therapeutic_relationship.receptivity_to_challenge = max(0.0,
                self.therapeutic_relationship.receptivity_to_challenge - 0.02)
        
        # Calculate overall therapeutic bond
        self.therapeutic_relationship.therapeutic_bond_strength = (
            self.therapeutic_relationship.trust_level * 0.3 +
            self.therapeutic_relationship.engagement_score * 0.2 +
            self.therapeutic_relationship.receptivity_to_challenge * 0.25 +
            self.therapeutic_relationship.collaborative_spirit * 0.25
        )
        
        # Assess rupture risk
        if (self.therapeutic_relationship.trust_level < 0.3 or 
            self.therapeutic_relationship.receptivity_to_challenge < 0.1):
            self.therapeutic_relationship.rupture_risk = 0.7
            self.therapeutic_relationship.repair_needed = True
        else:
            self.therapeutic_relationship.rupture_risk = max(0.0, 
                self.therapeutic_relationship.rupture_risk - 0.1)
    
    def _determine_challenge_level(self, readiness: UserReadinessIndicator, 
                                 breakthrough_detected: bool) -> ChallengeLevel:
        """Determine appropriate challenge level based on readiness and context."""
        base_level = self.challenge_strategies.get(readiness, ChallengeLevel.GENTLE_INQUIRY)
        
        # Adjust for breakthrough moments
        if breakthrough_detected:
            return ChallengeLevel.BREAKTHROUGH_PUSH
        
        # Adjust for therapeutic relationship quality
        if self.therapeutic_relationship.rupture_risk > 0.5:
            return ChallengeLevel.VALIDATION_ONLY
        
        # Adjust based on challenge acceptance history
        if hasattr(self, 'challenge_acceptance_rate'):
            if self.user_progress.challenge_acceptance_rate > 0.8:
                # User accepts challenges well, can push harder
                level_progression = {
                    ChallengeLevel.VALIDATION_ONLY: ChallengeLevel.GENTLE_INQUIRY,
                    ChallengeLevel.GENTLE_INQUIRY: ChallengeLevel.MODERATE_CHALLENGE,
                    ChallengeLevel.MODERATE_CHALLENGE: ChallengeLevel.STRONG_CHALLENGE,
                    ChallengeLevel.STRONG_CHALLENGE: ChallengeLevel.BREAKTHROUGH_PUSH
                }
                return level_progression.get(base_level, base_level)
        
        return base_level
    
    def _select_intervention_type(self, user_input: str, readiness: UserReadinessIndicator, 
                                context: Dict[str, Any]) -> InterventionType:
        """Select most appropriate intervention type based on context and readiness."""
        # Default interventions by readiness level
        readiness_interventions = {
            UserReadinessIndicator.RESISTANT: InterventionType.STRATEGIC_RESISTANCE,
            UserReadinessIndicator.DEFENSIVE: InterventionType.SOCRATIC_QUESTIONING,
            UserReadinessIndicator.AMBIVALENT: InterventionType.VALUES_CLARIFICATION,
            UserReadinessIndicator.OPEN: InterventionType.COGNITIVE_REFRAMING,
            UserReadinessIndicator.MOTIVATED: InterventionType.BEHAVIORAL_EXPERIMENT,
            UserReadinessIndicator.BREAKTHROUGH_READY: InterventionType.EXPOSURE_CHALLENGE
        }
        
        base_intervention = readiness_interventions.get(readiness, InterventionType.SOCRATIC_QUESTIONING)
        
        # Adjust based on content analysis
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["avoid", "scared", "anxious", "fear"]):
            return InterventionType.EXPOSURE_CHALLENGE
        
        if any(word in user_input_lower for word in ["think", "believe", "should", "must"]):
            return InterventionType.COGNITIVE_REFRAMING
        
        if any(word in user_input_lower for word in ["value", "important", "matter", "meaning"]):
            return InterventionType.VALUES_CLARIFICATION
        
        if "paradox" in user_input_lower or self.therapeutic_relationship.rupture_risk > 0.4:
            return InterventionType.PARADOXICAL_INTERVENTION
        
        return base_intervention
    
    def _generate_growth_response_strategy(self, user_input: str, readiness: UserReadinessIndicator,
                                         challenge_level: ChallengeLevel, intervention_type: InterventionType,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive growth-oriented response strategy."""
        strategy = {
            "approach": "growth_oriented",
            "friction_level": challenge_level.value,
            "intervention_focus": intervention_type.value,
            "growth_questions": [],
            "behavioral_experiments": [],
            "strategic_challenges": [],
            "validation_components": [],
            "insight_prompts": []
        }
        
        # Generate intervention-specific components
        if intervention_type == InterventionType.SOCRATIC_QUESTIONING:
            strategy["growth_questions"] = self._generate_socratic_questions(user_input, challenge_level)
        
        elif intervention_type == InterventionType.BEHAVIORAL_EXPERIMENT:
            strategy["behavioral_experiments"] = self._generate_behavioral_experiments(user_input, context)
        
        elif intervention_type == InterventionType.COGNITIVE_REFRAMING:
            strategy["strategic_challenges"] = self._generate_cognitive_challenges(user_input, challenge_level)
        
        elif intervention_type == InterventionType.VALUES_CLARIFICATION:
            strategy["insight_prompts"] = self._generate_values_exploration(user_input)
        
        elif intervention_type == InterventionType.STRATEGIC_RESISTANCE:
            strategy["strategic_challenges"] = self._generate_strategic_resistance(user_input)
        
        elif intervention_type == InterventionType.PARADOXICAL_INTERVENTION:
            strategy["strategic_challenges"] = self._generate_paradoxical_interventions(user_input)
        
        # Add validation when needed
        if challenge_level in [ChallengeLevel.VALIDATION_ONLY, ChallengeLevel.GENTLE_INQUIRY]:
            strategy["validation_components"] = self._generate_validation_components(user_input, context)
        
        return strategy
    
    def _generate_socratic_questions(self, user_input: str, challenge_level: ChallengeLevel) -> List[str]:
        """Generate Socratic questions tailored to challenge level."""
        questions = []
        user_input_lower = user_input.lower()
        
        # Base question templates by challenge level
        if challenge_level == ChallengeLevel.GENTLE_INQUIRY:
            templates = [
                "I'm curious about your perspective on this - what stands out to you?",
                "What do you think might be contributing to this situation?",
                "How do you typically handle situations like this?"
            ]
        elif challenge_level == ChallengeLevel.MODERATE_CHALLENGE:
            templates = [
                "What assumptions might you be making here that we could examine together?",
                "If your best friend were in this exact situation, what would you tell them?",
                "What would need to change for you to feel differently about this?"
            ]
        elif challenge_level == ChallengeLevel.STRONG_CHALLENGE:
            templates = [
                "I notice a pattern here - what do you make of that?",
                "What are you gaining by maintaining this perspective, even if it's painful?",
                "What would you have to give up to see this differently?"
            ]
        elif challenge_level == ChallengeLevel.BREAKTHROUGH_PUSH:
            templates = [
                "You seem ready for a deeper truth - what is it you already know but haven't said?",
                "What would happen if you stopped protecting yourself from this realization?",
                "What's the cost of staying where you are versus the risk of growing?"
            ]
        else:  # VALIDATION_ONLY
            templates = [
                "Tell me more about what this experience is like for you.",
                "What's most important for me to understand about your situation?",
                "How are you taking care of yourself through this?"
            ]
        
        # Select and customize questions
        selected_questions = random.sample(templates, min(3, len(templates)))
        
        # Add context-specific questions
        if "relationship" in user_input_lower:
            questions.append("What patterns do you notice in your relationships?")
        if "work" in user_input_lower or "job" in user_input_lower:
            questions.append("How does this situation align with your professional values?")
        if "family" in user_input_lower:
            questions.append("What family patterns might be influencing this?")
        
        questions.extend(selected_questions)
        return questions[:3]  # Limit to avoid overwhelming
    
    def _generate_behavioral_experiments(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavioral experiments that challenge user's current patterns."""
        experiments = []
        user_input_lower = user_input.lower()
        
        # Avoidance-based experiments
        if any(word in user_input_lower for word in ["avoid", "scared", "can't", "won't"]):
            experiments.append({
                "title": "Micro-Exposure Challenge",
                "description": "Take one small step toward what you're avoiding",
                "action": "Identify the smallest possible step you could take today toward what you're avoiding, then do it",
                "rationale": "Small actions build confidence and challenge avoidance patterns",
                "timeframe": "24 hours"
            })
        
        # Social experiments
        if any(word in user_input_lower for word in ["people", "social", "others", "relationship"]):
            experiments.append({
                "title": "Connection Experiment",
                "description": "Practice vulnerable connection with someone safe",
                "action": "Share one genuine feeling or concern with someone you trust",
                "rationale": "Authentic connection builds resilience and challenges isolation",
                "timeframe": "This week"
            })
        
        # Cognitive experiments
        if any(word in user_input_lower for word in ["think", "thoughts", "worry", "overthink"]):
            experiments.append({
                "title": "Thought Defusion Practice",
                "description": "Create distance from your thoughts",
                "action": "When you notice a troubling thought, say 'I'm having the thought that...' before it",
                "rationale": "This creates psychological distance from difficult thoughts",
                "timeframe": "Daily for one week"
            })
        
        return experiments[:2]  # Limit to avoid overwhelming
    
    def _generate_cognitive_challenges(self, user_input: str, challenge_level: ChallengeLevel) -> List[str]:
        """Generate cognitive challenges appropriate to the challenge level."""
        challenges = []
        user_input_lower = user_input.lower()
        
        # Identify cognitive patterns
        if any(word in user_input_lower for word in ["always", "never", "everyone", "nobody"]):
            if challenge_level == ChallengeLevel.GENTLE_INQUIRY:
                challenges.append("I notice you're using some absolute terms. I wonder if reality might be more nuanced?")
            elif challenge_level >= ChallengeLevel.MODERATE_CHALLENGE:
                challenges.append("Those are pretty absolute statements. What evidence contradicts this all-or-nothing view?")
        
        if any(word in user_input_lower for word in ["should", "must", "have to", "supposed to"]):
            if challenge_level == ChallengeLevel.GENTLE_INQUIRY:
                challenges.append("You seem to have some strong expectations. Where do these 'shoulds' come from?")
            elif challenge_level >= ChallengeLevel.MODERATE_CHALLENGE:
                challenges.append("Who decided these rules? What happens if you challenge these 'shoulds'?")
        
        if any(word in user_input_lower for word in ["fault", "blame", "responsible"]):
            if challenge_level >= ChallengeLevel.MODERATE_CHALLENGE:
                challenges.append("I hear you taking responsibility. How much of this is actually within your control?")
        
        return challenges
    
    def _generate_values_exploration(self, user_input: str) -> List[str]:
        """Generate prompts for values clarification and exploration."""
        prompts = [
            "What does this situation reveal about what matters most to you?",
            "If you were living fully according to your values, how would you handle this?",
            "What kind of person do you want to be in this situation?",
            "What would you want to be remembered for in how you handle challenges like this?",
            "What values are in conflict here, and which one deserves priority?"
        ]
        return random.sample(prompts, min(3, len(prompts)))
    
    def _generate_strategic_resistance(self, user_input: str) -> List[str]:
        """Generate strategic resistance for defensive clients."""
        return [
            "Maybe you're right that change is too difficult. Sometimes staying stuck is safer.",
            "I wonder if we're pushing too hard. Perhaps this isn't the right time for you.",
            "You know yourself best. If you say it won't work, you're probably right."
        ]
    
    def _generate_paradoxical_interventions(self, user_input: str) -> List[str]:
        """Generate paradoxical interventions for complex resistance."""
        return [
            "I'm going to suggest something unusual: try to feel worse about this situation.",
            "Maybe the problem isn't that you need to change, but that you're changing too fast.",
            "What if the solution is to fully embrace this problem rather than fight it?"
        ]
    
    def _generate_validation_components(self, user_input: str, context: Dict[str, Any]) -> List[str]:
        """Generate validation components for when challenge is not appropriate."""
        validations = [
            "It makes complete sense that you would feel this way given what you've experienced.",
            "You're showing incredible strength just by sharing this with me.",
            "Your feelings are completely valid and understandable.",
            "Anyone in your situation would struggle with this."
        ]
        return random.sample(validations, 2)
    
    def _generate_friction_component(self, strategy: Dict[str, Any], challenge_level: ChallengeLevel,
                                   intervention_type: InterventionType) -> str:
        """Generate the friction component of the response."""
        friction_parts = []
        
        # Add strategic challenges
        challenges = strategy.get("strategic_challenges", [])
        if challenges:
            friction_parts.append(f"\n**A gentle challenge:** {challenges[0]}")
        
        # Add insight prompts
        insights = strategy.get("insight_prompts", [])
        if insights:
            friction_parts.append(f"\n**Something to consider:** {insights[0]}")
        
        # Add validation if needed
        validations = strategy.get("validation_components", [])
        if validations and challenge_level == ChallengeLevel.VALIDATION_ONLY:
            friction_parts.append(f"\n**Validation:** {validations[0]}")
        
        return "".join(friction_parts)
    
    def _track_progress_indicators(self, user_input: str, strategy: Dict[str, Any]) -> None:
        """Track various progress indicators from user interaction."""
        user_input_lower = user_input.lower()
        
        # Track behavioral change indicators
        change_indicators = [
            "tried something new", "did something different", "took action",
            "made a change", "stepped out of comfort zone", "faced my fear"
        ]
        
        for indicator in change_indicators:
            if indicator in user_input_lower:
                self.user_progress.behavioral_change_indicators.append({
                    "indicator": indicator,
                    "session": self.user_progress.session_count,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Track resistance patterns
        resistance_indicators = [
            "can't", "won't work", "tried everything", "pointless",
            "too hard", "impossible", "don't want to"
        ]
        
        for indicator in resistance_indicators:
            if indicator in user_input_lower:
                self.user_progress.resistance_patterns.append({
                    "pattern": indicator,
                    "session": self.user_progress.session_count,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Update growth trajectory
        recent_readiness = self.user_progress.readiness_history[-5:] if len(self.user_progress.readiness_history) >= 5 else self.user_progress.readiness_history
        
        if recent_readiness:
            readiness_values = [self._readiness_to_score(r) for r in recent_readiness]
            if len(readiness_values) > 1:
                trend = sum(readiness_values[-3:]) - sum(readiness_values[:3]) if len(readiness_values) >= 3 else readiness_values[-1] - readiness_values[0]
                
                if trend > 1:
                    self.user_progress.growth_trajectory = "improving"
                elif trend < -1:
                    self.user_progress.growth_trajectory = "declining" 
                else:
                    self.user_progress.growth_trajectory = "stable"
                    
                # Check for breakthrough trajectory
                if any(r == UserReadinessIndicator.BREAKTHROUGH_READY for r in recent_readiness):
                    self.user_progress.growth_trajectory = "breakthrough"
    
    def _readiness_to_score(self, readiness: UserReadinessIndicator) -> int:
        """Convert readiness indicator to numeric score for trend analysis."""
        scores = {
            UserReadinessIndicator.RESISTANT: 0,
            UserReadinessIndicator.DEFENSIVE: 1,
            UserReadinessIndicator.AMBIVALENT: 2,
            UserReadinessIndicator.OPEN: 3,
            UserReadinessIndicator.MOTIVATED: 4,
            UserReadinessIndicator.BREAKTHROUGH_READY: 5
        }
        return scores.get(readiness, 2)
    
    def _update_outcome_metrics(self) -> None:
        """Update long-term outcome metrics."""
        # Calculate challenge acceptance rate
        total_challenges = len(self.intervention_history)
        if total_challenges > 0:
            accepted_challenges = sum(1 for intervention in self.intervention_history 
                                    if intervention.get("accepted", False))
            self.user_progress.challenge_acceptance_rate = accepted_challenges / total_challenges
        
        # Calculate insight frequency
        insight_sessions = sum(1 for moment in self.user_progress.breakthrough_moments)
        if self.user_progress.session_count > 0:
            self.user_progress.insight_frequency = insight_sessions / self.user_progress.session_count
        
        # Update outcome metrics
        self.outcome_metrics.update({
            "total_sessions": self.user_progress.session_count,
            "breakthrough_count": len(self.user_progress.breakthrough_moments),
            "challenge_acceptance_rate": self.user_progress.challenge_acceptance_rate,
            "therapeutic_bond_strength": self.therapeutic_relationship.therapeutic_bond_strength,
            "growth_trajectory": self.user_progress.growth_trajectory,
            "last_updated": datetime.now().isoformat()
        })
    
    def _get_progress_metrics(self) -> Dict[str, Any]:
        """Get current progress metrics."""
        return {
            "session_count": self.user_progress.session_count,
            "growth_trajectory": self.user_progress.growth_trajectory,
            "challenge_acceptance_rate": round(self.user_progress.challenge_acceptance_rate, 2),
            "insight_frequency": round(self.user_progress.insight_frequency, 2),
            "breakthrough_moments": len(self.user_progress.breakthrough_moments),
            "therapeutic_bond_strength": round(self.therapeutic_relationship.therapeutic_bond_strength, 2),
            "current_readiness": self.user_progress.readiness_history[-1].value if self.user_progress.readiness_history else "unknown"
        }
    
    def _get_friction_recommendation(self, challenge_level: ChallengeLevel) -> str:
        """Get recommendation for applying therapeutic friction."""
        recommendations = {
            ChallengeLevel.VALIDATION_ONLY: "Focus on safety and trust-building. No challenges recommended.",
            ChallengeLevel.GENTLE_INQUIRY: "Use curious questions and gentle exploration. Avoid direct challenges.",
            ChallengeLevel.MODERATE_CHALLENGE: "Balanced approach with supportive challenges. Monitor response closely.",
            ChallengeLevel.STRONG_CHALLENGE: "Direct challenges are appropriate. Push for growth and insight.",
            ChallengeLevel.BREAKTHROUGH_PUSH: "Maximum therapeutic leverage. Use powerful interventions for breakthrough."
        }
        return recommendations.get(challenge_level, "Assess individual needs and adjust accordingly.")
    
    def _calculate_breakthrough_potential(self) -> float:
        """Calculate the potential for breakthrough based on current indicators."""
        factors = [
            self.therapeutic_relationship.therapeutic_bond_strength * 0.3,
            self.therapeutic_relationship.receptivity_to_challenge * 0.25,
            self.user_progress.challenge_acceptance_rate * 0.2,
            (1 - self.therapeutic_relationship.rupture_risk) * 0.15,
            min(self.user_progress.insight_frequency * 2, 1.0) * 0.1
        ]
        return sum(factors)
    
    def _generate_progress_acknowledgment(self, friction_result: Dict[str, Any]) -> Optional[str]:
        """Generate progress acknowledgment based on therapeutic relationship and growth."""
        if self.user_progress.session_count < 2:
            return None
        
        bond_strength = self.therapeutic_relationship.therapeutic_bond_strength
        growth_trajectory = self.user_progress.growth_trajectory
        
        if growth_trajectory == "breakthrough":
            return "\n*I'm noticing significant growth in your insights and readiness for change. This is powerful work.*"
        elif growth_trajectory == "improving":
            return f"\n*Your openness to exploration has grown over our {self.user_progress.session_count} sessions. That takes courage.*"
        elif bond_strength > 0.8:
            return "\n*The trust we've built allows us to explore these challenging areas together safely.*"
        elif self.therapeutic_relationship.repair_needed:
            return "\n*I want to make sure we're moving at a pace that feels right for you. How is this sitting with you?*"
        
        return None
    
    def _initialize_intervention_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for different intervention types."""
        return {
            InterventionType.SOCRATIC_QUESTIONING.value: [
                "What do you make of this pattern?",
                "How does this serve you?",
                "What would change if you believed differently?",
                "What are you not telling yourself about this?"
            ],
            InterventionType.COGNITIVE_REFRAMING.value: [
                "Let's examine this belief together.",
                "What evidence challenges this thought?",
                "How would you view this if it happened to someone else?",
                "What's a more balanced way to see this?"
            ],
            InterventionType.BEHAVIORAL_EXPERIMENT.value: [
                "What small step could you take to test this?",
                "How could you experiment with a different approach?",
                "What would happen if you tried the opposite?",
                "What's one behavior you could change this week?"
            ],
            InterventionType.VALUES_CLARIFICATION.value: [
                "What does this reveal about your values?",
                "How would your ideal self handle this?",
                "What matters most to you in this situation?",
                "What kind of person do you want to be here?"
            ]
        }
    
    def get_comprehensive_assessment(self) -> Dict[str, Any]:
        """Get comprehensive assessment of user progress and therapeutic relationship."""
        return {
            "user_progress": asdict(self.user_progress),
            "therapeutic_relationship": asdict(self.therapeutic_relationship),
            "outcome_metrics": self.outcome_metrics,
            "breakthrough_potential": self._calculate_breakthrough_potential(),
            "recommendations": self._generate_therapeutic_recommendations(),
            "session_summary": {
                "total_sessions": self.user_progress.session_count,
                "current_phase": self._determine_therapy_phase(),
                "next_steps": self._recommend_next_steps()
            }
        }
    
    def _generate_therapeutic_recommendations(self) -> List[str]:
        """Generate recommendations for therapeutic approach."""
        recommendations = []
        
        if self.therapeutic_relationship.rupture_risk > 0.5:
            recommendations.append("Focus on relationship repair before introducing challenges")
        
        if self.user_progress.challenge_acceptance_rate < 0.3:
            recommendations.append("Reduce challenge intensity and focus on building alliance")
        
        if self.user_progress.insight_frequency > 0.3:
            recommendations.append("User is showing good insight - can handle stronger interventions")
        
        if len(self.user_progress.breakthrough_moments) > 2:
            recommendations.append("Consider transitioning to maintenance and integration phase")
        
        return recommendations
    
    def _determine_therapy_phase(self) -> str:
        """Determine current phase of therapy."""
        if self.user_progress.session_count < 3:
            return "engagement_building"
        elif self.therapeutic_relationship.therapeutic_bond_strength < 0.6:
            return "alliance_building"
        elif self.user_progress.growth_trajectory in ["improving", "breakthrough"]:
            return "active_change"
        elif len(self.user_progress.breakthrough_moments) > 2:
            return "integration"
        else:
            return "exploration"
    
    def _recommend_next_steps(self) -> List[str]:
        """Recommend next therapeutic steps."""
        phase = self._determine_therapy_phase()
        
        phase_recommendations = {
            "engagement_building": [
                "Continue building safety and trust",
                "Focus on understanding and validation",
                "Assess readiness for therapeutic work"
            ],
            "alliance_building": [
                "Address any relationship concerns",
                "Use gentle challenges to test alliance",
                "Increase collaborative planning"
            ],
            "active_change": [
                "Implement behavioral experiments",
                "Use moderate to strong challenges",
                "Focus on skill development and practice"
            ],
            "integration": [
                "Help consolidate insights and changes",
                "Plan for relapse prevention",
                "Prepare for therapy completion"
            ],
            "exploration": [
                "Continue exploring patterns and themes",
                "Increase self-awareness through questioning",
                "Prepare for more active interventions"
            ]
        }
        
        return phase_recommendations.get(phase, ["Continue current therapeutic approach"])