"""
Sophisticated AI Psychologist Agent with Evidence-Based Therapeutic Interventions.

This agent embodies a professional psychologist persona capable of providing
evidence-based therapeutic interventions including CBT, DBT, ACT, and other modalities.
It provides challenging responses when necessary while maintaining therapeutic boundaries.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import random
from datetime import datetime, timedelta
import json
from enum import Enum

from src.agents.base.base_agent import BaseAgent
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService

class TherapeuticModality(Enum):
    """Enumeration of therapeutic modalities."""
    CBT = "Cognitive Behavioral Therapy"
    DBT = "Dialectical Behavior Therapy"
    ACT = "Acceptance and Commitment Therapy"
    EXPOSURE = "Exposure Therapy"
    MINDFULNESS = "Mindfulness-Based Intervention"
    BEHAVIORAL_ACTIVATION = "Behavioral Activation"
    PSYCHOEDUCATION = "Psychoeducation"

class CrisisLevel(Enum):
    """Crisis intervention levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EMERGENCY = "emergency"

class TherapyAgent(BaseAgent):
    """Sophisticated AI Psychologist with Evidence-Based Interventions."""
    
    def __init__(self, model_provider=None):
        """Initialize the sophisticated therapy agent.
        
        Args:
            model_provider: LLM provider for generating embeddings and content
        """
        super().__init__(
            model=model_provider,
            name="therapy_agent",
            role="Professional AI Psychologist",
            description="A sophisticated AI psychologist providing evidence-based therapeutic interventions"
        )
        self.technique_service = TherapeuticTechniqueService(model_provider)
        
        # Initialize the vector store with therapeutic techniques
        self.technique_service.initialize_vector_store()
        
        # Professional persona attributes
        self.therapeutic_alliance_score = 0.7
        self.session_count = 0
        self.client_progress_history = []
        self.homework_assignments = []
        self.active_interventions = []
        
        # Crisis intervention keywords
        self.crisis_keywords = {
            CrisisLevel.EMERGENCY: [
                "suicide", "kill myself", "end it all", "not worth living",
                "hurt myself", "self-harm", "overdose", "jump off"
            ],
            CrisisLevel.HIGH: [
                "hopeless", "can't go on", "everything is pointless",
                "no way out", "better off dead", "harm"
            ],
            CrisisLevel.MODERATE: [
                "overwhelmed", "can't cope", "falling apart",
                "losing control", "breaking down"
            ]
        }
        
        # Cognitive distortions patterns
        self.cognitive_distortions = {
            "catastrophizing": ["worst case", "disaster", "terrible", "awful", "horrible"],
            "all_or_nothing": ["always", "never", "completely", "totally", "every time"],
            "mind_reading": ["they think", "everyone knows", "people see me as"],
            "fortune_telling": ["will fail", "going to happen", "bound to", "definitely will"],
            "emotional_reasoning": ["feel like", "seems like", "must be true because"]
        }
        
    def process(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input with sophisticated therapeutic analysis.
        
        Args:
            user_input: The user's message
            context: Contextual information including conversation history,
                     emotion analysis, etc.
        
        Returns:
            Dictionary containing comprehensive therapeutic response
        """
        self.session_count += 1
        
        # Crisis assessment
        crisis_level = self._assess_crisis_level(user_input)
        
        # Cognitive distortion detection
        distortions = self._detect_cognitive_distortions(user_input)
        
        # Extract emotion if available in context
        emotion = None
        if context.get("emotion_analysis") and context["emotion_analysis"].get("primary_emotion"):
            emotion = context["emotion_analysis"]["primary_emotion"]
        
        # Determine appropriate therapeutic modality
        modality = self._select_therapeutic_modality(user_input, emotion, distortions)
        
        # Get relevant therapeutic techniques
        techniques = self.technique_service.get_relevant_techniques(
            query=user_input,
            emotion=emotion,
            top_k=2
        )
        
        # Generate therapeutic response strategy
        response_strategy = self._generate_response_strategy(
            user_input, emotion, distortions, crisis_level, modality
        )
        
        # Create homework assignment if appropriate
        homework = self._generate_homework_assignment(modality, emotion, user_input)
        
        # Update therapeutic alliance based on interaction
        self._update_therapeutic_alliance(user_input, emotion)
        
        return {
            "therapeutic_techniques": techniques,
            "response_strategy": response_strategy,
            "crisis_level": crisis_level.value if crisis_level else None,
            "cognitive_distortions": distortions,
            "therapeutic_modality": modality.value if modality else None,
            "homework_assignment": homework,
            "therapeutic_alliance_score": self.therapeutic_alliance_score,
            "session_count": self.session_count,
            "challenging_response_needed": self._should_challenge(user_input, emotion)
        }
    
    def enhance_response(self, response: str, therapeutic_result: Dict[str, Any]) -> str:
        """Enhance response with sophisticated therapeutic interventions.
        
        Args:
            response: Original response from the chatbot
            therapeutic_result: Result from the process method
            
        Returns:
            Enhanced response with professional therapeutic interventions
        """
        if not therapeutic_result:
            return response
        
        # Handle crisis situations first
        if therapeutic_result.get("crisis_level") in ["high", "emergency"]:
            return self._generate_crisis_response(therapeutic_result)
        
        # Generate professional therapeutic response
        enhanced_response = self._generate_therapeutic_response(
            response, therapeutic_result
        )
        
        return enhanced_response
    
    def integrate_friction_insights(self, friction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate insights from TherapeuticFrictionAgent to enhance therapeutic approach.
        
        Args:
            friction_result: Results from TherapeuticFrictionAgent processing
            
        Returns:
            Enhanced therapeutic insights combining both approaches
        """
        if not friction_result:
            return {}
        
        # Update therapeutic alliance based on friction agent insights
        friction_relationship = friction_result.get("therapeutic_relationship", {})
        if friction_relationship:
            self.therapeutic_alliance_score = max(
                self.therapeutic_alliance_score,
                friction_relationship.get("therapeutic_bond_strength", self.therapeutic_alliance_score)
            )
        
        # Incorporate readiness assessment for modulating interventions
        user_readiness = friction_result.get("user_readiness", "ambivalent")
        challenge_level = friction_result.get("challenge_level", "gentle_inquiry")
        
        # Adjust challenging response threshold based on friction agent's assessment
        if challenge_level in ["validation_only", "gentle_inquiry"]:
            # Reduce challenging for vulnerable users
            self.therapeutic_alliance_score = max(0.5, self.therapeutic_alliance_score - 0.1)
        elif challenge_level in ["strong_challenge", "breakthrough_push"]:
            # User can handle more challenge
            self.therapeutic_alliance_score = min(1.0, self.therapeutic_alliance_score + 0.05)
        
        # Combine intervention recommendations
        friction_strategy = friction_result.get("response_strategy", {})
        therapeutic_integration = {
            "combined_approach": True,
            "friction_readiness": user_readiness,
            "friction_challenge_level": challenge_level,
            "friction_questions": friction_strategy.get("growth_questions", []),
            "friction_experiments": friction_strategy.get("behavioral_experiments", []),
            "breakthrough_potential": friction_result.get("context_updates", {}).get("therapeutic_friction", {}).get("breakthrough_potential", 0),
            "integrated_recommendations": self._generate_integrated_recommendations(friction_result)
        }
        
        return therapeutic_integration
    
    def _get_agent_specific_context(self, query: str, response: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate therapy-specific context updates for supervisor tracking"""
        agent_context = {}
        
        # Add therapeutic context
        agent_context["therapeutic_context"] = {
            "alliance_score": self.therapeutic_alliance_score,
            "session_count": self.session_count,
            "active_homework_count": len(self.homework_assignments),
            "challenging_response_indicated": self._should_challenge(str(query), None)
        }
        
        # Add therapeutic assessment data if available
        if hasattr(self, '_last_assessment_result'):
            agent_context["last_therapeutic_assessment"] = self._last_assessment_result
        
        # Add crisis indicators if detected
        crisis_level = self._assess_crisis_level(str(query))
        if crisis_level:
            agent_context["crisis_detection"] = {
                "level": crisis_level.value,
                "timestamp": datetime.now().isoformat(),
                "requires_immediate_attention": crisis_level in [CrisisLevel.HIGH, CrisisLevel.EMERGENCY]
            }
        
        # Add cognitive distortions detected
        distortions = self._detect_cognitive_distortions(str(query))
        if distortions:
            agent_context["cognitive_distortions"] = {
                "detected": distortions,
                "count": len(distortions)
            }
        
        return agent_context
    
    def _generate_integrated_recommendations(self, friction_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations that integrate both therapeutic approaches."""
        recommendations = []
        
        challenge_level = friction_result.get("challenge_level", "gentle_inquiry")
        user_readiness = friction_result.get("user_readiness", "ambivalent")
        progress_metrics = friction_result.get("progress_metrics", {})
        
        # Evidence-based technique recommendations
        if self.therapeutic_alliance_score > 0.7:
            recommendations.append("Strong therapeutic alliance allows for CBT interventions")
        
        # Friction-based recommendations
        if challenge_level == "breakthrough_push":
            recommendations.append("User ready for intensive therapeutic work - combine exposure with cognitive restructuring")
        elif challenge_level == "validation_only":
            recommendations.append("Focus on psychoeducation and supportive interventions")
        
        # Progress-based recommendations
        growth_trajectory = progress_metrics.get("growth_trajectory", "stable")
        if growth_trajectory == "breakthrough":
            recommendations.append("Consolidate insights with behavioral activation assignments")
        elif growth_trajectory == "declining":
            recommendations.append("Return to alliance building and reduce intervention intensity")
        
        return recommendations
    
    def _assess_crisis_level(self, user_input: str) -> Optional[CrisisLevel]:
        """Assess crisis level based on user input."""
        user_input_lower = user_input.lower()
        
        for level, keywords in self.crisis_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return level
        
        return None
    
    def _detect_cognitive_distortions(self, user_input: str) -> List[str]:
        """Detect cognitive distortions in user input."""
        user_input_lower = user_input.lower()
        detected_distortions = []
        
        for distortion, patterns in self.cognitive_distortions.items():
            if any(pattern in user_input_lower for pattern in patterns):
                detected_distortions.append(distortion)
        
        return detected_distortions
    
    def _select_therapeutic_modality(self, user_input: str, emotion: str, 
                                   distortions: List[str]) -> Optional[TherapeuticModality]:
        """Select appropriate therapeutic modality based on analysis."""
        if distortions:
            return TherapeuticModality.CBT
        
        if emotion in ["anxiety", "fear", "panic"]:
            return TherapeuticModality.EXPOSURE
        
        if emotion in ["anger", "rage", "irritation"]:
            return TherapeuticModality.DBT
        
        if emotion in ["sadness", "depression", "hopelessness"]:
            return TherapeuticModality.BEHAVIORAL_ACTIVATION
        
        if "stress" in user_input.lower() or "overwhelm" in user_input.lower():
            return TherapeuticModality.MINDFULNESS
        
        return TherapeuticModality.CBT  # Default
    
    def _generate_response_strategy(self, user_input: str, emotion: str, 
                                  distortions: List[str], crisis_level: Optional[CrisisLevel],
                                  modality: Optional[TherapeuticModality]) -> Dict[str, Any]:
        """Generate comprehensive response strategy."""
        strategy = {
            "approach": "supportive" if crisis_level else "challenging",
            "focus_areas": [],
            "intervention_type": modality.value if modality else "general_support",
            "socratic_questions": [],
            "behavioral_experiments": [],
            "psychoeducation_topics": []
        }
        
        if distortions:
            strategy["focus_areas"].extend(["cognitive_restructuring", "thought_challenging"])
            strategy["socratic_questions"] = self._generate_socratic_questions(distortions)
        
        if emotion in ["depression", "sadness"]:
            strategy["focus_areas"].append("behavioral_activation")
            strategy["behavioral_experiments"] = self._suggest_behavioral_experiments(emotion)
        
        if emotion in ["anxiety", "fear"]:
            strategy["focus_areas"].append("exposure_therapy")
            strategy["psychoeducation_topics"].append("anxiety_physiology")
        
        return strategy
    
    def _generate_socratic_questions(self, distortions: List[str]) -> List[str]:
        """Generate Socratic questions to challenge cognitive distortions."""
        questions = []
        
        question_bank = {
            "catastrophizing": [
                "What evidence do you have that this worst-case scenario will definitely happen?",
                "What would you tell a friend who was thinking this way?",
                "What's the most likely outcome, realistically?"
            ],
            "all_or_nothing": [
                "Are there any shades of gray in this situation?",
                "Can you think of times when this wasn't always or never true?",
                "What would 70% success look like instead of perfect or failure?"
            ],
            "mind_reading": [
                "What actual evidence do you have that people think this about you?",
                "Is it possible they're thinking something completely different?",
                "How could you test whether this assumption is accurate?"
            ],
            "fortune_telling": [
                "What are some alternative outcomes that could happen?",
                "If you were wrong about predicting the future before, how do you know you're right now?",
                "What would change if you approached this without assuming the outcome?"
            ]
        }
        
        for distortion in distortions:
            if distortion in question_bank:
                questions.extend(random.sample(question_bank[distortion], 2))
        
        return questions[:3]  # Limit to 3 questions to avoid overwhelming
    
    def _suggest_behavioral_experiments(self, emotion: str) -> List[Dict[str, str]]:
        """Suggest behavioral experiments based on emotion."""
        experiments = {
            "depression": [
                {"activity": "Daily walk", "duration": "15 minutes", "goal": "Increase physical activity"},
                {"activity": "Social contact", "duration": "10 minutes", "goal": "Combat isolation"},
                {"activity": "Pleasant activity", "duration": "20 minutes", "goal": "Increase positive emotions"}
            ],
            "anxiety": [
                {"activity": "Gradual exposure", "duration": "5-10 minutes", "goal": "Reduce avoidance"},
                {"activity": "Breathing exercise", "duration": "5 minutes", "goal": "Manage physical symptoms"},
                {"activity": "Worry time", "duration": "15 minutes", "goal": "Contain anxious thoughts"}
            ]
        }
        
        return experiments.get(emotion, [])
    
    def _generate_homework_assignment(self, modality: Optional[TherapeuticModality], 
                                    emotion: str, user_input: str) -> Optional[Dict[str, Any]]:
        """Generate appropriate homework assignment."""
        if not modality:
            return None
        
        assignments = {
            TherapeuticModality.CBT: {
                "title": "Thought Record",
                "description": "Track negative thoughts and identify more balanced alternatives",
                "instructions": [
                    "When you notice a strong negative emotion, write down the thought",
                    "Rate how much you believe the thought (0-100%)",
                    "Look for evidence for and against the thought",
                    "Write a more balanced thought",
                    "Rate your belief in the balanced thought"
                ],
                "duration": "daily for 1 week"
            },
            TherapeuticModality.DBT: {
                "title": "Distress Tolerance Skills Practice",
                "description": "Practice TIPP skills when experiencing intense emotions",
                "instructions": [
                    "Temperature: Use cold water on face/hands when distressed",
                    "Intense exercise: 10 minutes of vigorous activity",
                    "Paced breathing: 4 seconds in, 6 seconds out",
                    "Paired muscle relaxation: Tense and release muscle groups"
                ],
                "duration": "practice when needed"
            },
            TherapeuticModality.BEHAVIORAL_ACTIVATION: {
                "title": "Activity Scheduling",
                "description": "Schedule and engage in meaningful activities",
                "instructions": [
                    "List 5 activities that used to bring you joy",
                    "Schedule one activity each day",
                    "Rate your mood before and after (1-10)",
                    "Note any obstacles and how you overcame them"
                ],
                "duration": "daily for 1 week"
            }
        }
        
        assignment = assignments.get(modality)
        if assignment:
            assignment["assigned_date"] = datetime.now().isoformat()
            assignment["due_date"] = (datetime.now() + timedelta(days=7)).isoformat()
            self.homework_assignments.append(assignment)
        
        return assignment
    
    def _update_therapeutic_alliance(self, user_input: str, emotion: str) -> None:
        """Update therapeutic alliance score based on interaction."""
        # Increase alliance for engagement
        if len(user_input) > 50:  # Detailed responses
            self.therapeutic_alliance_score = min(1.0, self.therapeutic_alliance_score + 0.02)
        
        # Decrease for resistance indicators
        resistance_indicators = ["don't want to", "won't work", "tried everything", "pointless"]
        if any(indicator in user_input.lower() for indicator in resistance_indicators):
            self.therapeutic_alliance_score = max(0.3, self.therapeutic_alliance_score - 0.05)
    
    def _should_challenge(self, user_input: str, emotion: str) -> bool:
        """Determine if challenging response is needed."""
        # Challenge when therapeutic alliance is strong enough
        if self.therapeutic_alliance_score < 0.6:
            return False
        
        # Challenge cognitive distortions
        if self._detect_cognitive_distortions(user_input):
            return True
        
        # Challenge avoidance behaviors
        avoidance_patterns = ["can't", "impossible", "too hard", "will never"]
        if any(pattern in user_input.lower() for pattern in avoidance_patterns):
            return True
        
        return False
    
    def _generate_crisis_response(self, therapeutic_result: Dict[str, Any]) -> str:
        """Generate appropriate crisis intervention response."""
        crisis_level = therapeutic_result.get("crisis_level")
        
        if crisis_level == "emergency":
            return (
                "I'm very concerned about what you've shared. Your safety is the most important thing right now. "
                "Please reach out to a crisis helpline immediately:\n\n"
                "• National Suicide Prevention Lifeline: 988\n"
                "• Crisis Text Line: Text HOME to 741741\n"
                "• Or go to your nearest emergency room\n\n"
                "You don't have to go through this alone. Professional help is available 24/7."
            )
        
        elif crisis_level == "high":
            return (
                "I can hear how much pain you're in right now. These feelings are temporary, even though they feel overwhelming. "
                "Let's focus on keeping you safe and getting through this moment.\n\n"
                "Some immediate coping strategies:\n"
                "• Reach out to someone you trust\n"
                "• Use grounding techniques (5-4-3-2-1: 5 things you see, 4 you touch, etc.)\n"
                "• Remove any means of self-harm from your immediate environment\n\n"
                "Consider calling a crisis line if these feelings persist: 988"
            )
        
        return "I'm here to support you through this difficult time."
    
    def _generate_therapeutic_response(self, original_response: str, 
                                     therapeutic_result: Dict[str, Any]) -> str:
        """Generate sophisticated therapeutic response."""
        response_parts = [original_response]
        
        # Add challenging element if needed
        if therapeutic_result.get("challenging_response_needed"):
            response_parts.append(self._generate_challenging_response(therapeutic_result))
        
        # Add Socratic questions
        socratic_questions = therapeutic_result.get("response_strategy", {}).get("socratic_questions", [])
        if socratic_questions:
            response_parts.append("\n**Let's explore this together:**")
            for question in socratic_questions[:2]:  # Limit to 2 questions
                response_parts.append(f"• {question}")
        
        # Add homework assignment
        homework = therapeutic_result.get("homework_assignment")
        if homework:
            response_parts.append(f"\n**Therapeutic Assignment: {homework['title']}**")
            response_parts.append(homework['description'])
            response_parts.append("Instructions:")
            for instruction in homework['instructions'][:3]:  # Limit instructions
                response_parts.append(f"• {instruction}")
        
        # Add progress acknowledgment
        if self.session_count > 1:
            response_parts.append(f"\n*We've been working together for {self.session_count} sessions. "
                                f"Your engagement in this process shows strength and commitment to growth.*")
        
        return "\n\n".join(response_parts)
    
    def _generate_challenging_response(self, therapeutic_result: Dict[str, Any]) -> str:
        """Generate appropriately challenging therapeutic response."""
        distortions = therapeutic_result.get("cognitive_distortions", [])
        
        challenging_responses = {
            "catastrophizing": "I notice you're focusing on the worst possible outcome. While it's natural to be concerned, let's examine whether this catastrophic thinking is helping or hindering you.",
            "all_or_nothing": "You're using very absolute terms - 'always' and 'never.' Life rarely operates in such extremes. What would a more nuanced view look like?",
            "mind_reading": "You seem certain about what others are thinking. How can we be sure about thoughts we can't actually access?",
            "fortune_telling": "You're predicting a negative future as if it's certain. What if we approached this with curiosity about multiple possible outcomes instead?"
        }
        
        if distortions:
            default_response = "Let's examine this thought pattern together."
            return f"\n**A gentle challenge:** {challenging_responses.get(distortions[0], default_response)}"
        
        return "\n**Reflection:** I wonder if there might be another way to look at this situation."
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Generate progress summary for the client."""
        return {
            "session_count": self.session_count,
            "therapeutic_alliance_score": round(self.therapeutic_alliance_score, 2),
            "active_homework_assignments": len(self.homework_assignments),
            "progress_indicators": {
                "engagement_level": "high" if self.therapeutic_alliance_score > 0.8 else "moderate" if self.therapeutic_alliance_score > 0.6 else "building",
                "readiness_for_challenge": self.therapeutic_alliance_score > 0.7
            }
        }
    
    def complete_homework_assignment(self, assignment_id: int, completion_notes: str) -> Dict[str, Any]:
        """Mark homework assignment as completed."""
        if assignment_id < len(self.homework_assignments):
            assignment = self.homework_assignments[assignment_id]
            assignment["completed"] = True
            assignment["completion_date"] = datetime.now().isoformat()
            assignment["notes"] = completion_notes
            
            # Increase therapeutic alliance for homework completion
            self.therapeutic_alliance_score = min(1.0, self.therapeutic_alliance_score + 0.05)
            
            return {"status": "completed", "assignment": assignment}
        
        return {"status": "error", "message": "Assignment not found"}