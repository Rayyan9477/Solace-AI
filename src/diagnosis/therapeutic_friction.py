"""
Therapeutic Friction and Growth-Oriented Response Engine

This module implements the therapeutic friction approach that challenges users 
to grow rather than just validating their feelings, based on the Solace AI philosophy.
It creates responses that promote growth and self-reflection while maintaining support.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np
from collections import defaultdict
import re

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger
from ..models.llm import get_llm

logger = get_logger(__name__)

@dataclass
class GrowthMoment:
    """Represents a moment of potential growth or breakthrough"""
    moment_id: str
    user_id: str
    timestamp: datetime
    trigger_message: str
    growth_opportunity: str
    challenge_level: float  # 0.0 (gentle) to 1.0 (direct)
    user_readiness: float   # 0.0 (not ready) to 1.0 (very ready)
    intervention_type: str  # question, reframe, challenge, insight
    expected_outcome: str
    follow_up_required: bool

@dataclass
class TherapeuticResponse:
    """Structured therapeutic response with friction elements"""
    response_text: str
    friction_level: float  # Amount of challenge vs validation
    growth_target: str     # What growth this aims to promote
    validation_elements: List[str]  # Parts that validate feelings
    challenge_elements: List[str]   # Parts that challenge thinking
    follow_up_questions: List[str]  # Questions to promote reflection
    therapeutic_technique: str      # CBT, DBT, Socratic questioning, etc.
    expected_user_reaction: str
    success_indicators: List[str]

@dataclass
class UserReadinessProfile:
    """Profile of user's readiness for therapeutic challenges"""
    user_id: str
    current_emotional_state: str
    stress_level: float
    previous_response_to_challenges: List[str]
    breakthrough_history: List[str]
    defense_mechanisms: List[str]
    optimal_challenge_level: float
    preferred_approach: str
    last_updated: datetime

class TherapeuticFrictionEngine:
    """
    Engine that applies therapeutic friction by balancing validation with
    growth-oriented challenges based on user readiness and therapeutic goals.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the therapeutic friction engine"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # User readiness profiles
        self.user_profiles = {}
        
        # Therapeutic techniques database
        self.therapeutic_techniques = self._load_therapeutic_techniques()
        
        # Growth tracking
        self.growth_moments = defaultdict(list)
        self.breakthrough_tracker = defaultdict(list)
        
        # Configuration
        self.min_friction_level = 0.1
        self.max_friction_level = 0.9
        self.breakthrough_threshold = 0.7
        self.readiness_adaptation_rate = 0.1
        
    async def generate_therapeutic_response(self,
                                          user_id: str,
                                          user_message: str,
                                          emotional_context: Dict[str, Any],
                                          conversation_history: List[Dict[str, Any]],
                                          session_data: Dict[str, Any] = None) -> TherapeuticResponse:
        """
        Generate a therapeutic response with appropriate friction level
        
        Args:
            user_id: User identifier
            user_message: Current user message
            emotional_context: Emotional state and voice analysis
            conversation_history: Recent conversation context
            session_data: Additional session information
            
        Returns:
            Structured therapeutic response with friction elements
        """
        try:
            self.logger.info(f"Generating therapeutic response for user {user_id}")
            
            # Step 1: Assess user readiness for challenge
            readiness_profile = await self._assess_user_readiness(
                user_id, user_message, emotional_context, conversation_history
            )
            
            # Step 2: Identify growth opportunities
            growth_opportunities = await self._identify_growth_opportunities(
                user_message, emotional_context, conversation_history
            )
            
            # Step 3: Determine optimal friction level
            optimal_friction = self._calculate_optimal_friction(
                readiness_profile, growth_opportunities, emotional_context
            )
            
            # Step 4: Select therapeutic technique
            technique = await self._select_therapeutic_technique(
                user_message, emotional_context, readiness_profile
            )
            
            # Step 5: Generate response components
            validation_elements = await self._generate_validation_elements(
                user_message, emotional_context
            )
            
            challenge_elements = await self._generate_challenge_elements(
                user_message, growth_opportunities, optimal_friction, technique
            )
            
            # Step 6: Construct balanced response
            response_text = await self._construct_balanced_response(
                validation_elements, challenge_elements, optimal_friction, technique
            )
            
            # Step 7: Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                growth_opportunities, technique, optimal_friction
            )
            
            # Step 8: Create therapeutic response
            therapeutic_response = TherapeuticResponse(
                response_text=response_text,
                friction_level=optimal_friction,
                growth_target=growth_opportunities[0] if growth_opportunities else "emotional awareness",
                validation_elements=validation_elements,
                challenge_elements=challenge_elements,
                follow_up_questions=follow_up_questions,
                therapeutic_technique=technique,
                expected_user_reaction=self._predict_user_reaction(readiness_profile, optimal_friction),
                success_indicators=self._define_success_indicators(growth_opportunities)
            )
            
            # Step 9: Record growth moment
            await self._record_growth_moment(
                user_id, user_message, growth_opportunities, optimal_friction
            )
            
            # Step 10: Update user profile
            await self._update_user_readiness_profile(user_id, readiness_profile, therapeutic_response)
            
            return therapeutic_response
            
        except Exception as e:
            self.logger.error(f"Error generating therapeutic response: {str(e)}")
            # Fallback to supportive response
            return TherapeuticResponse(
                response_text="I hear you and want to understand better. Can you tell me more about what you're experiencing?",
                friction_level=0.1,
                growth_target="emotional expression",
                validation_elements=["I hear you"],
                challenge_elements=[],
                follow_up_questions=["Can you tell me more about what you're experiencing?"],
                therapeutic_technique="supportive listening",
                expected_user_reaction="continued sharing",
                success_indicators=["user continues conversation"]
            )
    
    async def _assess_user_readiness(self,
                                   user_id: str,
                                   user_message: str,
                                   emotional_context: Dict[str, Any],
                                   conversation_history: List[Dict[str, Any]]) -> UserReadinessProfile:
        """Assess user's readiness for therapeutic challenges"""
        
        # Get existing profile or create new one
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
        else:
            profile = UserReadinessProfile(
                user_id=user_id,
                current_emotional_state="unknown",
                stress_level=0.5,
                previous_response_to_challenges=[],
                breakthrough_history=[],
                defense_mechanisms=[],
                optimal_challenge_level=0.3,
                preferred_approach="supportive",
                last_updated=datetime.now()
            )
        
        # Update current emotional state
        profile.current_emotional_state = self._assess_emotional_state(
            user_message, emotional_context
        )
        
        # Update stress level
        profile.stress_level = self._calculate_stress_level(
            user_message, emotional_context, conversation_history
        )
        
        # Analyze defensive patterns
        profile.defense_mechanisms = self._identify_defense_mechanisms(
            user_message, conversation_history
        )
        
        # Calculate optimal challenge level
        profile.optimal_challenge_level = self._calculate_optimal_challenge_level(
            profile.stress_level, profile.current_emotional_state, 
            profile.previous_response_to_challenges
        )
        
        # Update preferred approach
        profile.preferred_approach = self._determine_preferred_approach(
            profile.previous_response_to_challenges, profile.breakthrough_history
        )
        
        profile.last_updated = datetime.now()
        self.user_profiles[user_id] = profile
        
        return profile
    
    async def _identify_growth_opportunities(self,
                                           user_message: str,
                                           emotional_context: Dict[str, Any],
                                           conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Identify potential growth opportunities in the user's sharing"""
        
        opportunities = []
        
        # Cognitive patterns
        if self._detect_cognitive_distortions(user_message):
            opportunities.append("cognitive_reframing")
        
        # Emotional awareness
        if self._detect_emotion_avoidance(user_message):
            opportunities.append("emotional_awareness")
        
        # Behavioral patterns
        if self._detect_avoidance_patterns(user_message, conversation_history):
            opportunities.append("behavioral_activation")
        
        # Self-efficacy
        if self._detect_learned_helplessness(user_message):
            opportunities.append("self_efficacy_building")
        
        # Relationship patterns
        if self._detect_relationship_issues(user_message):
            opportunities.append("interpersonal_skills")
        
        # Values clarification
        if self._detect_values_conflicts(user_message):
            opportunities.append("values_clarification")
        
        # Mindfulness opportunities
        if self._detect_rumination_patterns(user_message):
            opportunities.append("mindfulness_practice")
        
        return opportunities[:3]  # Focus on top 3 opportunities
    
    def _calculate_optimal_friction(self,
                                  readiness_profile: UserReadinessProfile,
                                  growth_opportunities: List[str],
                                  emotional_context: Dict[str, Any]) -> float:
        """Calculate the optimal level of therapeutic friction"""
        
        base_friction = readiness_profile.optimal_challenge_level
        
        # Adjust based on current emotional state
        if readiness_profile.current_emotional_state in ["crisis", "severe_distress"]:
            base_friction *= 0.3  # Much gentler approach
        elif readiness_profile.current_emotional_state in ["moderate_distress", "anxious"]:
            base_friction *= 0.7  # Somewhat gentler
        elif readiness_profile.current_emotional_state in ["stable", "motivated"]:
            base_friction *= 1.2  # Can handle more challenge
        
        # Adjust based on stress level
        stress_adjustment = 1.0 - (readiness_profile.stress_level * 0.5)
        base_friction *= stress_adjustment
        
        # Adjust based on growth opportunities
        if "cognitive_reframing" in growth_opportunities:
            base_friction += 0.1  # Cognitive work can handle more friction
        if "emotional_awareness" in growth_opportunities and readiness_profile.stress_level > 0.7:
            base_friction -= 0.1  # Be gentler with emotional exploration under stress
        
        # Adjust based on defense mechanisms
        if "intellectualization" in readiness_profile.defense_mechanisms:
            base_friction += 0.1  # Can push through intellectual defenses
        if "emotional_shutdown" in readiness_profile.defense_mechanisms:
            base_friction -= 0.2  # Need to be much gentler
        
        # Clamp to valid range
        return max(self.min_friction_level, min(self.max_friction_level, base_friction))
    
    async def _select_therapeutic_technique(self,
                                          user_message: str,
                                          emotional_context: Dict[str, Any],
                                          readiness_profile: UserReadinessProfile) -> str:
        """Select appropriate therapeutic technique"""
        
        # Map conditions to techniques
        if "cognitive_reframing" in user_message or self._detect_cognitive_distortions(user_message):
            return "cognitive_behavioral_therapy"
        
        if readiness_profile.current_emotional_state in ["anxious", "overwhelmed"]:
            return "dialectical_behavior_therapy"
        
        if self._detect_values_conflicts(user_message):
            return "acceptance_commitment_therapy"
        
        if self._detect_relationship_issues(user_message):
            return "interpersonal_therapy"
        
        if readiness_profile.optimal_challenge_level > 0.6:
            return "socratic_questioning"
        
        return "person_centered_therapy"  # Default supportive approach
    
    async def _generate_validation_elements(self,
                                          user_message: str,
                                          emotional_context: Dict[str, Any]) -> List[str]:
        """Generate validation elements for the response"""
        
        validation_elements = []
        
        # Emotional validation
        detected_emotions = self._extract_emotions_from_message(user_message)
        for emotion in detected_emotions:
            validation_elements.append(f"It makes sense that you're feeling {emotion}")
        
        # Experience validation
        if self._detect_struggle_language(user_message):
            validation_elements.append("What you're going through sounds really difficult")
        
        # Effort validation
        if self._detect_effort_language(user_message):
            validation_elements.append("I can see you're really trying to work through this")
        
        # Strength validation
        strengths = self._identify_user_strengths(user_message)
        for strength in strengths[:2]:  # Limit to 2 strengths
            validation_elements.append(f"It shows {strength} that you're addressing this")
        
        return validation_elements
    
    async def _generate_challenge_elements(self,
                                         user_message: str,
                                         growth_opportunities: List[str],
                                         friction_level: float,
                                         technique: str) -> List[str]:
        """Generate challenge elements based on growth opportunities"""
        
        challenge_elements = []
        
        for opportunity in growth_opportunities:
            if opportunity == "cognitive_reframing":
                if friction_level > 0.5:
                    challenge_elements.append("I'm wondering if there might be another way to look at this situation")
                else:
                    challenge_elements.append("What do you think someone you trust might say about this?")
            
            elif opportunity == "emotional_awareness":
                if friction_level > 0.6:
                    challenge_elements.append("What emotions are you trying to avoid feeling right now?")
                else:
                    challenge_elements.append("I'm curious about what you're feeling beneath the surface")
            
            elif opportunity == "behavioral_activation":
                if friction_level > 0.5:
                    challenge_elements.append("What would it look like to take one small step forward?")
                else:
                    challenge_elements.append("I wonder what might happen if you tried something different")
            
            elif opportunity == "self_efficacy_building":
                if friction_level > 0.4:
                    challenge_elements.append("What evidence do you have that you can't handle this?")
                else:
                    challenge_elements.append("Can you think of a time when you overcame something difficult?")
            
            elif opportunity == "values_clarification":
                challenge_elements.append("What matters most to you in this situation?")
            
            elif opportunity == "mindfulness_practice":
                challenge_elements.append("What would it be like to sit with these feelings instead of fighting them?")
        
        # Apply technique-specific challenges
        if technique == "socratic_questioning" and friction_level > 0.6:
            challenge_elements.append("How do you know that's true?")
            challenge_elements.append("What assumptions are you making?")
        
        return challenge_elements[:2]  # Limit to 2 main challenges per response
    
    async def _construct_balanced_response(self,
                                         validation_elements: List[str],
                                         challenge_elements: List[str],
                                         friction_level: float,
                                         technique: str) -> str:
        """Construct a balanced response with validation and challenge"""
        
        response_parts = []
        
        # Start with validation (always lead with understanding)
        if validation_elements:
            response_parts.append(validation_elements[0])
        
        # Add transitional language based on friction level
        if friction_level > 0.6:
            transition = "And I'm wondering..."
        elif friction_level > 0.4:
            transition = "At the same time, I'm curious..."
        else:
            transition = "I'm gently wondering..."
        
        # Add challenge element
        if challenge_elements:
            response_parts.append(f"{transition} {challenge_elements[0]}")
        
        # Add additional validation if friction is high
        if friction_level > 0.6 and len(validation_elements) > 1:
            response_parts.append(f"I want to be clear that {validation_elements[1].lower()}")
        
        # Add technique-specific elements
        if technique == "dialectical_behavior_therapy":
            response_parts.append("Can we hold both of these truths together?")
        elif technique == "acceptance_commitment_therapy":
            response_parts.append("What would acting in line with your values look like here?")
        
        return " ".join(response_parts)
    
    async def _generate_follow_up_questions(self,
                                          growth_opportunities: List[str],
                                          technique: str,
                                          friction_level: float) -> List[str]:
        """Generate follow-up questions to promote reflection"""
        
        questions = []
        
        # Growth-opportunity specific questions
        for opportunity in growth_opportunities:
            if opportunity == "cognitive_reframing":
                questions.append("What evidence supports or challenges this thought?")
            elif opportunity == "emotional_awareness":
                questions.append("What is this emotion trying to tell you?")
            elif opportunity == "behavioral_activation":
                questions.append("What's one small action you could take today?")
            elif opportunity == "self_efficacy_building":
                questions.append("What strengths have helped you before?")
            elif opportunity == "values_clarification":
                questions.append("How does this align with what's important to you?")
        
        # Technique-specific questions
        if technique == "socratic_questioning":
            questions.extend([
                "What would you tell a friend in this situation?",
                "How might this look different in a week/month/year?"
            ])
        
        # Adjust question intensity based on friction level
        if friction_level < 0.3:
            # Make questions gentler
            questions = [q.replace("What", "I wonder what").replace("How", "I wonder how") for q in questions]
        
        return questions[:3]  # Limit to 3 follow-up questions
    
    # Helper methods for pattern detection
    
    def _assess_emotional_state(self, user_message: str, emotional_context: Dict[str, Any]) -> str:
        """Assess current emotional state from message and context"""
        
        # Crisis indicators
        crisis_words = ["suicide", "kill myself", "end it all", "can't go on", "hopeless"]
        if any(word in user_message.lower() for word in crisis_words):
            return "crisis"
        
        # Severe distress indicators
        severe_distress = ["unbearable", "can't take it", "breaking down", "falling apart"]
        if any(phrase in user_message.lower() for phrase in severe_distress):
            return "severe_distress"
        
        # Voice emotion context
        if emotional_context:
            emotions = emotional_context.get("emotions", {})
            if emotions.get("sadness", 0) > 0.8 or emotions.get("anxiety", 0) > 0.8:
                return "moderate_distress"
            elif emotions.get("anger", 0) > 0.7:
                return "angry"
            elif emotions.get("joy", 0) > 0.6:
                return "stable"
        
        # Text-based assessment
        if any(word in user_message.lower() for word in ["anxious", "worried", "scared"]):
            return "anxious"
        elif any(word in user_message.lower() for word in ["sad", "depressed", "down"]):
            return "depressed"
        elif any(word in user_message.lower() for word in ["frustrated", "angry", "mad"]):
            return "frustrated"
        elif any(word in user_message.lower() for word in ["motivated", "ready", "determined"]):
            return "motivated"
        
        return "stable"
    
    def _calculate_stress_level(self,
                              user_message: str,
                              emotional_context: Dict[str, Any],
                              conversation_history: List[Dict[str, Any]]) -> float:
        """Calculate current stress level (0.0 to 1.0)"""
        
        stress_indicators = {
            "overwhelmed": 0.8,
            "can't cope": 0.9,
            "stressed": 0.6,
            "pressure": 0.5,
            "tired": 0.4,
            "exhausted": 0.7,
            "breaking point": 0.9
        }
        
        stress_score = 0.0
        for indicator, score in stress_indicators.items():
            if indicator in user_message.lower():
                stress_score = max(stress_score, score)
        
        # Factor in voice emotion data
        if emotional_context:
            emotions = emotional_context.get("emotions", {})
            stress_emotions = emotions.get("anxiety", 0) + emotions.get("fear", 0)
            stress_score = max(stress_score, stress_emotions)
        
        return min(1.0, stress_score)
    
    def _identify_defense_mechanisms(self,
                                   user_message: str,
                                   conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Identify active defense mechanisms"""
        
        mechanisms = []
        
        # Intellectualization
        if len(re.findall(r'\b(analyze|understand|figure out|makes sense)\b', user_message.lower())) > 2:
            mechanisms.append("intellectualization")
        
        # Minimization
        if any(phrase in user_message.lower() for phrase in ["it's not that bad", "others have it worse", "i'm fine"]):
            mechanisms.append("minimization")
        
        # Avoidance
        if any(phrase in user_message.lower() for phrase in ["don't want to talk about", "change the subject", "anyway"]):
            mechanisms.append("avoidance")
        
        # Rationalization
        if user_message.lower().count("because") > 2:
            mechanisms.append("rationalization")
        
        # Emotional shutdown
        if any(phrase in user_message.lower() for phrase in ["don't feel anything", "numb", "shut down"]):
            mechanisms.append("emotional_shutdown")
        
        return mechanisms
    
    def _calculate_optimal_challenge_level(self,
                                         stress_level: float,
                                         emotional_state: str,
                                         previous_responses: List[str]) -> float:
        """Calculate optimal challenge level for this user"""
        
        base_level = 0.5  # Start with moderate challenge
        
        # Adjust for stress
        base_level -= stress_level * 0.3
        
        # Adjust for emotional state
        if emotional_state in ["crisis", "severe_distress"]:
            base_level = 0.1
        elif emotional_state in ["moderate_distress", "anxious"]:
            base_level *= 0.6
        elif emotional_state in ["motivated", "stable"]:
            base_level *= 1.3
        
        # Learn from previous responses
        if "defensive" in previous_responses:
            base_level *= 0.8
        if "breakthrough" in previous_responses:
            base_level *= 1.2
        if "engaged" in previous_responses:
            base_level *= 1.1
        
        return max(0.1, min(0.9, base_level))
    
    def _determine_preferred_approach(self,
                                    previous_responses: List[str],
                                    breakthrough_history: List[str]) -> str:
        """Determine user's preferred therapeutic approach"""
        
        if not previous_responses:
            return "supportive"
        
        if "cognitive_work" in previous_responses and "breakthrough" in breakthrough_history:
            return "cognitive_behavioral"
        elif "emotional_exploration" in previous_responses:
            return "emotion_focused"
        elif "values_work" in previous_responses:
            return "acceptance_based"
        elif "questioning" in previous_responses:
            return "socratic"
        
        return "supportive"
    
    # Pattern detection methods
    
    def _detect_cognitive_distortions(self, message: str) -> bool:
        """Detect cognitive distortions in user message"""
        distortions = [
            "always", "never", "everyone", "no one", "everything", "nothing",
            "should", "must", "have to", "can't", "impossible", "terrible",
            "awful", "disaster", "ruined", "worthless"
        ]
        return sum(1 for word in distortions if word in message.lower()) >= 2
    
    def _detect_emotion_avoidance(self, message: str) -> bool:
        """Detect emotional avoidance patterns"""
        avoidance_phrases = [
            "don't want to feel", "trying not to think", "push it down",
            "ignore it", "don't deal with", "avoid feeling"
        ]
        return any(phrase in message.lower() for phrase in avoidance_phrases)
    
    def _detect_avoidance_patterns(self, message: str, history: List[Dict[str, Any]]) -> bool:
        """Detect behavioral avoidance patterns"""
        avoidance_words = ["avoiding", "putting off", "procrastinating", "can't bring myself"]
        return any(word in message.lower() for word in avoidance_words)
    
    def _detect_learned_helplessness(self, message: str) -> bool:
        """Detect learned helplessness patterns"""
        helplessness_phrases = [
            "can't do anything", "no point", "what's the use", "nothing works",
            "can't change", "stuck", "hopeless", "powerless"
        ]
        return any(phrase in message.lower() for phrase in helplessness_phrases)
    
    def _detect_relationship_issues(self, message: str) -> bool:
        """Detect relationship-related concerns"""
        relationship_words = [
            "relationship", "partner", "friends", "family", "conflict",
            "argument", "lonely", "isolated", "misunderstood"
        ]
        return any(word in message.lower() for word in relationship_words)
    
    def _detect_values_conflicts(self, message: str) -> bool:
        """Detect values conflicts"""
        values_phrases = [
            "don't know what's right", "conflicted", "torn between",
            "what should i do", "moral dilemma", "against my values"
        ]
        return any(phrase in message.lower() for phrase in values_phrases)
    
    def _detect_rumination_patterns(self, message: str) -> bool:
        """Detect rumination patterns"""
        rumination_phrases = [
            "keep thinking about", "can't stop thinking", "over and over",
            "replaying", "stuck on", "obsessing"
        ]
        return any(phrase in message.lower() for phrase in rumination_phrases)
    
    def _extract_emotions_from_message(self, message: str) -> List[str]:
        """Extract emotions mentioned in the message"""
        emotion_words = [
            "sad", "angry", "anxious", "worried", "scared", "frustrated",
            "overwhelmed", "hopeless", "guilty", "ashamed", "lonely",
            "hurt", "disappointed", "confused", "stressed"
        ]
        return [emotion for emotion in emotion_words if emotion in message.lower()]
    
    def _detect_struggle_language(self, message: str) -> bool:
        """Detect language indicating struggle"""
        struggle_phrases = [
            "struggling with", "having a hard time", "difficult", "challenging",
            "tough", "hard", "overwhelming", "exhausting"
        ]
        return any(phrase in message.lower() for phrase in struggle_phrases)
    
    def _detect_effort_language(self, message: str) -> bool:
        """Detect language indicating effort"""
        effort_phrases = [
            "trying to", "working on", "attempting", "doing my best",
            "making an effort", "pushing myself"
        ]
        return any(phrase in message.lower() for phrase in effort_phrases)
    
    def _identify_user_strengths(self, message: str) -> List[str]:
        """Identify strengths demonstrated in the message"""
        strengths = []
        
        if any(word in message.lower() for word in ["trying", "working", "effort"]):
            strengths.append("persistence")
        
        if any(word in message.lower() for word in ["help", "support", "therapy"]):
            strengths.append("help-seeking")
        
        if "awareness" in message.lower() or "realize" in message.lower():
            strengths.append("self-awareness")
        
        if any(word in message.lower() for word in ["care", "love", "important"]):
            strengths.append("caring")
        
        return strengths
    
    def _predict_user_reaction(self, profile: UserReadinessProfile, friction_level: float) -> str:
        """Predict likely user reaction to therapeutic friction"""
        
        if friction_level < 0.3:
            return "likely to feel heard and supported"
        elif friction_level < 0.6:
            if profile.optimal_challenge_level > 0.5:
                return "likely to engage with gentle challenge"
            else:
                return "may feel slightly pushed but will likely engage"
        else:
            if "intellectualization" in profile.defense_mechanisms:
                return "may intellectualize but likely to engage"
            elif "emotional_shutdown" in profile.defense_mechanisms:
                return "risk of defensive reaction or withdrawal"
            else:
                return "likely to be challenged but motivated to reflect"
    
    def _define_success_indicators(self, growth_opportunities: List[str]) -> List[str]:
        """Define indicators of successful therapeutic friction"""
        
        indicators = []
        
        for opportunity in growth_opportunities:
            if opportunity == "cognitive_reframing":
                indicators.append("user questions their initial thought")
            elif opportunity == "emotional_awareness":
                indicators.append("user identifies underlying emotions")
            elif opportunity == "behavioral_activation":
                indicators.append("user considers taking action")
            elif opportunity == "self_efficacy_building":
                indicators.append("user recalls past successes")
        
        # General indicators
        indicators.extend([
            "user asks follow-up questions",
            "user shares more vulnerable information",
            "user expresses new insights",
            "user shows curiosity about themselves"
        ])
        
        return indicators
    
    async def _record_growth_moment(self,
                                  user_id: str,
                                  trigger_message: str,
                                  growth_opportunities: List[str],
                                  friction_level: float):
        """Record a potential growth moment"""
        
        moment = GrowthMoment(
            moment_id=f"{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            timestamp=datetime.now(),
            trigger_message=trigger_message,
            growth_opportunity=growth_opportunities[0] if growth_opportunities else "general_awareness",
            challenge_level=friction_level,
            user_readiness=self.user_profiles[user_id].optimal_challenge_level if user_id in self.user_profiles else 0.5,
            intervention_type="therapeutic_friction",
            expected_outcome="increased_self_awareness",
            follow_up_required=friction_level > 0.6
        )
        
        self.growth_moments[user_id].append(moment)
        
        # Store in vector database if available
        if self.vector_db:
            await self._store_growth_moment(moment)
    
    async def _update_user_readiness_profile(self,
                                           user_id: str,
                                           profile: UserReadinessProfile,
                                           response: TherapeuticResponse):
        """Update user readiness profile based on interaction"""
        
        # This would be updated based on actual user response
        # For now, just update the timestamp
        profile.last_updated = datetime.now()
        self.user_profiles[user_id] = profile
    
    async def _store_growth_moment(self, moment: GrowthMoment):
        """Store growth moment in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "growth_moment",
                "user_id": moment.user_id,
                "growth_opportunity": moment.growth_opportunity,
                "challenge_level": moment.challenge_level,
                "timestamp": moment.timestamp.isoformat(),
                "trigger_message": moment.trigger_message,
                "intervention_type": moment.intervention_type
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=moment.moment_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing growth moment: {str(e)}")
    
    def _load_therapeutic_techniques(self) -> Dict[str, Any]:
        """Load therapeutic techniques database"""
        return {
            "cognitive_behavioral_therapy": {
                "description": "Focuses on identifying and changing negative thought patterns",
                "suitable_for": ["cognitive_distortions", "anxiety", "depression"],
                "friction_tolerance": 0.7
            },
            "dialectical_behavior_therapy": {
                "description": "Emphasizes acceptance and change, mindfulness",
                "suitable_for": ["emotional_dysregulation", "crisis", "overwhelm"],
                "friction_tolerance": 0.4
            },
            "acceptance_commitment_therapy": {
                "description": "Focuses on values-based action and psychological flexibility",
                "suitable_for": ["values_conflicts", "avoidance", "meaning"],
                "friction_tolerance": 0.6
            },
            "socratic_questioning": {
                "description": "Uses questions to promote self-discovery",
                "suitable_for": ["intellectualization", "insight", "self_awareness"],
                "friction_tolerance": 0.8
            },
            "person_centered_therapy": {
                "description": "Emphasizes unconditional positive regard and empathy",
                "suitable_for": ["low_readiness", "crisis", "building_rapport"],
                "friction_tolerance": 0.2
            }
        }