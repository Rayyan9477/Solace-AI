"""
Breakthrough Detection Agent for Therapeutic Friction.

Specializes in identifying breakthrough moments, insight patterns, and
transformational opportunities in therapeutic conversations.
"""

from typing import Dict, List, Optional, Any, Tuple
import re
from datetime import datetime, timedelta
import math

from .base_friction_agent import BaseFrictionAgent, FrictionAgentType


class BreakthroughType:
    """Types of therapeutic breakthroughs."""
    COGNITIVE_INSIGHT = "cognitive_insight"
    EMOTIONAL_BREAKTHROUGH = "emotional_breakthrough"
    BEHAVIORAL_REALIZATION = "behavioral_realization"
    PATTERN_RECOGNITION = "pattern_recognition"
    VALUES_CLARIFICATION = "values_clarification"
    TRAUMA_INTEGRATION = "trauma_integration"
    RELATIONSHIP_INSIGHT = "relationship_insight"


class BreakthroughDetectionAgent(BaseFrictionAgent):
    """
    Specialized agent for detecting breakthrough moments and therapeutic insights.
    
    Uses advanced pattern recognition to identify moments of significant
    therapeutic progress, insight, and transformation.
    """
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the breakthrough detection agent."""
        super().__init__(
            model_provider=model_provider,
            agent_type=FrictionAgentType.BREAKTHROUGH_DETECTION,
            config=config
        )
        
        # Breakthrough detection patterns with weights and contexts
        self.breakthrough_patterns = {
            BreakthroughType.COGNITIVE_INSIGHT: {
                "markers": [
                    ("I never realized", 4.0), ("it just clicked", 3.5), ("now I understand", 3.0),
                    ("I see it now", 3.5), ("makes so much sense", 3.0), ("the light bulb went off", 4.0),
                    ("I get it", 2.5), ("suddenly clear", 3.0), ("pieces fit together", 3.5)
                ],
                "contextual_cues": ["understanding", "clarity", "realization", "awareness"],
                "emotional_indicators": ["relief", "surprise", "excitement", "clarity"],
                "linguistic_features": ["present tense realizations", "metaphorical language", "connecting language"]
            },
            BreakthroughType.EMOTIONAL_BREAKTHROUGH: {
                "markers": [
                    ("I can feel", 3.0), ("emotionally free", 4.0), ("weight lifted", 3.5),
                    ("finally grieving", 3.5), ("letting go", 3.0), ("feeling again", 3.0),
                    ("tears of relief", 3.5), ("heart opening", 3.0), ("safe to feel", 3.5)
                ],
                "contextual_cues": ["emotion", "feeling", "heart", "soul", "tears"],
                "emotional_indicators": ["catharsis", "relief", "joy", "sadness", "liberation"],
                "linguistic_features": ["somatic language", "emotional metaphors", "vulnerability expressions"]
            },
            BreakthroughType.BEHAVIORAL_REALIZATION: {
                "markers": [
                    ("I keep doing", 2.5), ("same pattern", 3.0), ("I always", 2.0),
                    ("my behavior", 2.5), ("I react by", 2.0), ("automatic response", 3.0),
                    ("I can choose", 3.5), ("different way", 2.5), ("break the cycle", 3.5)
                ],
                "contextual_cues": ["behavior", "action", "response", "choice", "pattern"],
                "emotional_indicators": ["determination", "empowerment", "hope", "motivation"],
                "linguistic_features": ["behavioral language", "choice language", "pattern recognition"]
            },
            BreakthroughType.PATTERN_RECOGNITION: {
                "markers": [
                    ("same thing happens", 3.0), ("every time", 2.5), ("pattern is", 3.5),
                    ("I notice", 2.0), ("happens when", 2.5), ("trigger is", 3.0),
                    ("connects to", 2.5), ("all related", 3.0), ("theme throughout", 3.5)
                ],
                "contextual_cues": ["pattern", "theme", "connection", "relationship", "link"],
                "emotional_indicators": ["insight", "understanding", "clarity", "surprise"],
                "linguistic_features": ["connecting language", "temporal markers", "causal language"]
            },
            BreakthroughType.VALUES_CLARIFICATION: {
                "markers": [
                    ("what matters", 3.0), ("really important", 2.5), ("my values", 3.5),
                    ("stands for", 2.5), ("believe in", 2.0), ("core of who", 3.5),
                    ("authentic self", 3.5), ("true to", 3.0), ("what I want", 2.5)
                ],
                "contextual_cues": ["values", "beliefs", "authentic", "true", "core", "important"],
                "emotional_indicators": ["clarity", "determination", "peace", "alignment"],
                "linguistic_features": ["identity language", "value statements", "priority language"]
            },
            BreakthroughType.RELATIONSHIP_INSIGHT: {
                "markers": [
                    ("in relationships", 2.5), ("with people", 2.0), ("family pattern", 3.5),
                    ("attachment style", 3.5), ("boundary issues", 3.0), ("intimacy fear", 3.0),
                    ("learned from", 2.5), ("childhood", 3.0), ("trust issues", 2.5)
                ],
                "contextual_cues": ["relationship", "family", "attachment", "connection", "intimacy"],
                "emotional_indicators": ["vulnerability", "sadness", "understanding", "compassion"],
                "linguistic_features": ["relational language", "historical references", "interpersonal terms"]
            }
        }
        
        # Breakthrough intensity indicators
        self.intensity_markers = {
            "high": [
                "life-changing", "transformative", "profound", "shocking", "mind-blowing",
                "overwhelming", "incredible", "amazing", "powerful", "deep"
            ],
            "medium": [
                "significant", "important", "meaningful", "clear", "helpful",
                "useful", "valuable", "interesting", "insightful"
            ],
            "low": [
                "somewhat", "a little", "minor", "small", "slight", "bit"
            ]
        }
        
        # Temporal breakthrough indicators
        self.temporal_patterns = {
            "immediate": ["just", "right now", "suddenly", "instant", "immediate"],
            "recent": ["today", "yesterday", "this week", "recently", "lately"],
            "ongoing": ["been", "always", "usually", "tend to", "keep"]
        }
        
        # Historical breakthrough tracking
        self.breakthrough_history = []
        self.insight_accumulation = []
        self.pattern_recognition_events = []
        
        # Breakthrough validation thresholds
        self.validation_thresholds = {
            "minimum_score": 2.5,
            "emotional_alignment": 0.6,
            "contextual_support": 0.7,
            "temporal_consistency": 0.5
        }
    
    async def assess(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect breakthrough moments and assess their significance."""
        try:
            # Primary detection: Pattern matching
            breakthrough_scores = self._analyze_breakthrough_patterns(user_input)
            
            # Secondary analysis: Emotional congruence
            emotional_analysis = self._analyze_emotional_congruence(user_input, context)
            
            # Tertiary analysis: Contextual support
            contextual_analysis = self._analyze_contextual_support(user_input, context)
            
            # Quaternary analysis: Temporal factors
            temporal_analysis = self._analyze_temporal_factors(user_input, context)
            
            # Intensity assessment
            intensity_analysis = self._assess_breakthrough_intensity(user_input)
            
            # Combine all analyses
            combined_assessment = self._combine_breakthrough_analyses(
                breakthrough_scores, emotional_analysis, contextual_analysis, 
                temporal_analysis, intensity_analysis
            )
            
            # Determine if breakthrough occurred
            breakthrough_detected, breakthrough_details = self._determine_breakthrough_status(
                combined_assessment, user_input, context
            )
            
            # Calculate breakthrough potential for future sessions
            breakthrough_potential = self._calculate_breakthrough_potential(context)
            
            # Generate therapeutic recommendations
            recommendations = self._generate_breakthrough_recommendations(
                breakthrough_detected, breakthrough_details, combined_assessment
            )
            
            # Update breakthrough history
            if breakthrough_detected:
                self._record_breakthrough_event(breakthrough_details, context)
            
            assessment_result = {
                "breakthrough_detected": breakthrough_detected,
                "breakthrough_details": breakthrough_details,
                "breakthrough_scores": breakthrough_scores,
                "emotional_analysis": emotional_analysis,
                "contextual_analysis": contextual_analysis,
                "temporal_analysis": temporal_analysis,
                "intensity_analysis": intensity_analysis,
                "breakthrough_potential": breakthrough_potential,
                "recommendations": recommendations,
                "historical_context": self._get_breakthrough_historical_context(),
                "validation_metrics": self._calculate_validation_metrics(combined_assessment)
            }
            
            return assessment_result
            
        except Exception as e:
            self.logger.error(f"Error in breakthrough detection: {str(e)}")
            return {
                "breakthrough_detected": False,
                "error": str(e)
            }
    
    def _analyze_breakthrough_patterns(self, user_input: str) -> Dict[str, float]:
        """Analyze text for breakthrough pattern markers."""
        user_input_lower = user_input.lower()
        breakthrough_scores = {}
        
        for breakthrough_type, patterns in self.breakthrough_patterns.items():
            score = 0.0
            matched_markers = []
            
            # Check breakthrough markers
            for marker, weight in patterns["markers"]:
                if marker in user_input_lower:
                    score += weight
                    matched_markers.append(marker)
                    self.logger.debug(f"Found breakthrough marker '{marker}' for {breakthrough_type}")
            
            # Check contextual cues
            contextual_boost = 0.0
            for cue in patterns["contextual_cues"]:
                if cue in user_input_lower:
                    contextual_boost += 0.5
            
            # Check linguistic features
            linguistic_boost = 0.0
            for feature in patterns["linguistic_features"]:
                if self._check_linguistic_feature(user_input_lower, feature):
                    linguistic_boost += 0.5
            
            final_score = score + (contextual_boost * 0.3) + (linguistic_boost * 0.2)
            
            breakthrough_scores[breakthrough_type] = {
                "raw_score": score,
                "contextual_boost": contextual_boost,
                "linguistic_boost": linguistic_boost,
                "final_score": final_score,
                "matched_markers": matched_markers
            }
        
        return breakthrough_scores
    
    def _analyze_emotional_congruence(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional congruence with breakthrough patterns."""
        emotion_analysis = context.get("emotion_analysis", {})
        primary_emotion = emotion_analysis.get("primary_emotion", "").lower()
        emotion_confidence = emotion_analysis.get("confidence", 0.0)
        
        congruence_scores = {}
        
        for breakthrough_type, patterns in self.breakthrough_patterns.items():
            congruence_score = 0.0
            
            # Check if emotion matches expected breakthrough emotions
            expected_emotions = [e.lower() for e in patterns["emotional_indicators"]]
            
            if primary_emotion in expected_emotions:
                congruence_score = emotion_confidence * 1.5  # Strong positive match
            elif primary_emotion and primary_emotion not in expected_emotions:
                # Check for related emotions (simplified approach)
                related_emotions = self._get_related_emotions(expected_emotions)
                if primary_emotion in related_emotions:
                    congruence_score = emotion_confidence * 0.8  # Moderate match
                else:
                    congruence_score = -0.3  # Mismatch penalty
            
            congruence_scores[breakthrough_type] = congruence_score
        
        return {
            "primary_emotion": primary_emotion,
            "emotion_confidence": emotion_confidence,
            "congruence_scores": congruence_scores,
            "overall_congruence": max(congruence_scores.values()) if congruence_scores else 0.0
        }
    
    def _analyze_contextual_support(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual factors that support breakthrough interpretation."""
        # Session context
        session_count = context.get("session_count", 0)
        recent_stress = context.get("recent_stress_indicators", [])
        therapeutic_alliance = context.get("therapeutic_alliance_score", 0.5)
        
        # Historical context
        previous_insights = len(self.breakthrough_history)
        recent_breakthroughs = len([b for b in self.breakthrough_history 
                                  if self._is_recent_breakthrough(b)])
        
        # Input characteristics
        input_length = len(user_input)
        exclamation_count = user_input.count('!')
        question_count = user_input.count('?')
        
        # Calculate contextual support factors
        session_readiness = min(1.0, session_count / 10.0)  # More likely in later sessions
        alliance_support = therapeutic_alliance
        length_support = min(1.0, input_length / 200.0)  # Longer inputs often indicate insight
        emotional_markers = (exclamation_count + question_count) / 10.0
        
        overall_support = (
            session_readiness * 0.3 +
            alliance_support * 0.3 +
            length_support * 0.2 +
            min(1.0, emotional_markers) * 0.2
        )
        
        return {
            "session_readiness": session_readiness,
            "alliance_support": alliance_support,
            "length_support": length_support,
            "emotional_markers": emotional_markers,
            "overall_support": overall_support,
            "previous_insights": previous_insights,
            "recent_breakthroughs": recent_breakthroughs
        }
    
    def _analyze_temporal_factors(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal aspects of breakthrough language."""
        user_input_lower = user_input.lower()
        temporal_scores = {}
        
        for temporal_type, markers in self.temporal_patterns.items():
            score = sum(2.0 for marker in markers if marker in user_input_lower)
            temporal_scores[temporal_type] = score
        
        # Determine dominant temporal frame
        if temporal_scores:
            dominant_temporal = max(temporal_scores.items(), key=lambda x: x[1])
        else:
            dominant_temporal = ("ongoing", 0.0)
        
        # Temporal consistency with breakthrough types
        temporal_breakthrough_alignment = self._assess_temporal_breakthrough_alignment(
            dominant_temporal[0], user_input_lower
        )
        
        return {
            "temporal_scores": temporal_scores,
            "dominant_temporal": dominant_temporal[0],
            "temporal_strength": dominant_temporal[1],
            "temporal_breakthrough_alignment": temporal_breakthrough_alignment
        }
    
    def _assess_breakthrough_intensity(self, user_input: str) -> Dict[str, Any]:
        """Assess the intensity level of potential breakthrough."""
        user_input_lower = user_input.lower()
        intensity_scores = {"high": 0, "medium": 0, "low": 0}
        
        for intensity_level, markers in self.intensity_markers.items():
            for marker in markers:
                if marker in user_input_lower:
                    intensity_scores[intensity_level] += 1
        
        # Determine overall intensity
        total_markers = sum(intensity_scores.values())
        if total_markers == 0:
            overall_intensity = "medium"  # Default
            intensity_confidence = 0.3
        else:
            max_intensity = max(intensity_scores.items(), key=lambda x: x[1])
            overall_intensity = max_intensity[0]
            intensity_confidence = max_intensity[1] / total_markers
        
        # Additional intensity indicators
        caps_ratio = sum(1 for c in user_input if c.isupper()) / len(user_input) if user_input else 0
        exclamation_density = user_input.count('!') / len(user_input.split()) if user_input else 0
        
        # Adjust intensity based on additional factors
        if caps_ratio > 0.1 or exclamation_density > 0.1:
            if overall_intensity == "low":
                overall_intensity = "medium"
            elif overall_intensity == "medium" and (caps_ratio > 0.2 or exclamation_density > 0.2):
                overall_intensity = "high"
        
        return {
            "intensity_scores": intensity_scores,
            "overall_intensity": overall_intensity,
            "intensity_confidence": intensity_confidence,
            "caps_ratio": caps_ratio,
            "exclamation_density": exclamation_density
        }
    
    def _combine_breakthrough_analyses(self, breakthrough_scores: Dict, emotional_analysis: Dict,
                                     contextual_analysis: Dict, temporal_analysis: Dict,
                                     intensity_analysis: Dict) -> Dict[str, Any]:
        """Combine all breakthrough analyses into unified assessment."""
        combined_scores = {}
        
        for breakthrough_type in breakthrough_scores.keys():
            # Base score from pattern matching
            base_score = breakthrough_scores[breakthrough_type]["final_score"]
            
            # Emotional congruence modifier
            emotion_modifier = emotional_analysis["congruence_scores"].get(breakthrough_type, 0.0)
            
            # Contextual support modifier
            context_modifier = contextual_analysis["overall_support"] * 2.0
            
            # Temporal alignment modifier
            temporal_modifier = temporal_analysis["temporal_breakthrough_alignment"]
            
            # Intensity modifier
            intensity_modifier = self._get_intensity_modifier(intensity_analysis["overall_intensity"])
            
            # Calculate weighted combined score
            combined_score = (
                base_score * 0.4 +
                emotion_modifier * 0.25 +
                context_modifier * 0.15 +
                temporal_modifier * 0.1 +
                intensity_modifier * 0.1
            )
            
            combined_scores[breakthrough_type] = {
                "combined_score": combined_score,
                "base_score": base_score,
                "emotion_modifier": emotion_modifier,
                "context_modifier": context_modifier,
                "temporal_modifier": temporal_modifier,
                "intensity_modifier": intensity_modifier
            }
        
        return combined_scores
    
    def _determine_breakthrough_status(self, combined_assessment: Dict, 
                                     user_input: str, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Determine if a breakthrough has occurred and its characteristics."""
        # Find highest scoring breakthrough type
        if not combined_assessment:
            return False, {}
        
        max_breakthrough = max(combined_assessment.items(), 
                             key=lambda x: x[1]["combined_score"])
        
        breakthrough_type = max_breakthrough[0]
        breakthrough_data = max_breakthrough[1]
        max_score = breakthrough_data["combined_score"]
        
        # Check against validation thresholds
        breakthrough_detected = (
            max_score >= self.validation_thresholds["minimum_score"] and
            breakthrough_data["emotion_modifier"] >= -0.2 and  # Not strong emotional mismatch
            breakthrough_data["context_modifier"] >= self.validation_thresholds["contextual_support"]
        )
        
        if breakthrough_detected:
            breakthrough_details = {
                "type": breakthrough_type,
                "score": max_score,
                "confidence": min(1.0, max_score / 5.0),
                "intensity": self._categorize_breakthrough_intensity(max_score),
                "timestamp": datetime.now().isoformat(),
                "trigger_text": user_input,
                "components": breakthrough_data,
                "validation_passed": True
            }
        else:
            breakthrough_details = {
                "type": None,
                "score": max_score,
                "confidence": 0.0,
                "validation_passed": False,
                "reason": "Below threshold or validation criteria not met"
            }
        
        return breakthrough_detected, breakthrough_details
    
    def _calculate_breakthrough_potential(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate potential for breakthrough in upcoming sessions."""
        # Factors that increase breakthrough potential
        session_count = context.get("session_count", 0)
        therapeutic_alliance = context.get("therapeutic_alliance_score", 0.5)
        recent_insights = len([b for b in self.breakthrough_history 
                             if self._is_recent_breakthrough(b)])
        
        # Base breakthrough potential calculation
        session_factor = min(1.0, (session_count - 3) / 10.0)  # Increases after session 3
        alliance_factor = therapeutic_alliance
        insight_momentum = min(1.0, recent_insights / 3.0)
        
        # Historical breakthrough pattern
        if len(self.breakthrough_history) >= 2:
            # Calculate average time between breakthroughs
            breakthrough_intervals = []
            for i in range(1, len(self.breakthrough_history)):
                prev_time = datetime.fromisoformat(self.breakthrough_history[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(self.breakthrough_history[i]["timestamp"])
                interval = (curr_time - prev_time).days
                breakthrough_intervals.append(interval)
            
            avg_interval = sum(breakthrough_intervals) / len(breakthrough_intervals)
            last_breakthrough = datetime.fromisoformat(self.breakthrough_history[-1]["timestamp"])
            days_since_last = (datetime.now() - last_breakthrough).days
            
            timing_factor = min(1.0, days_since_last / avg_interval)
        else:
            timing_factor = 0.5
        
        overall_potential = (
            session_factor * 0.3 +
            alliance_factor * 0.3 +
            insight_momentum * 0.2 +
            timing_factor * 0.2
        )
        
        return {
            "overall_potential": overall_potential,
            "session_factor": session_factor,
            "alliance_factor": alliance_factor,
            "insight_momentum": insight_momentum,
            "timing_factor": timing_factor,
            "readiness_level": self._categorize_breakthrough_readiness(overall_potential)
        }
    
    def _generate_breakthrough_recommendations(self, breakthrough_detected: bool,
                                            breakthrough_details: Dict[str, Any],
                                            combined_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on breakthrough analysis."""
        recommendations = []
        
        if breakthrough_detected:
            breakthrough_type = breakthrough_details["type"]
            intensity = breakthrough_details["intensity"]
            
            # Type-specific recommendations
            if breakthrough_type == BreakthroughType.COGNITIVE_INSIGHT:
                recommendations.extend([
                    "Consolidate and expand on this new understanding",
                    "Explore implications of this insight for daily life",
                    "Connect this realization to past experiences",
                    "Consider how this changes your perspective"
                ])
            elif breakthrough_type == BreakthroughType.EMOTIONAL_BREAKTHROUGH:
                recommendations.extend([
                    "Allow space for emotional processing",
                    "Explore the emotions that emerged",
                    "Consider the meaning of this emotional shift",
                    "Practice self-compassion during integration"
                ])
            elif breakthrough_type == BreakthroughType.BEHAVIORAL_REALIZATION:
                recommendations.extend([
                    "Develop concrete strategies for behavior change",
                    "Identify triggers and alternative responses",
                    "Practice new behavioral patterns gradually",
                    "Monitor progress and adjust approaches"
                ])
            elif breakthrough_type == BreakthroughType.PATTERN_RECOGNITION:
                recommendations.extend([
                    "Explore other areas where this pattern might exist",
                    "Investigate the origins of this pattern",
                    "Develop strategies to interrupt the pattern",
                    "Build awareness of pattern triggers"
                ])
            
            # Intensity-specific recommendations
            if intensity == "high":
                recommendations.append("Take time to integrate this profound realization")
            elif intensity == "low":
                recommendations.append("Build on this initial awareness with deeper exploration")
                
        else:
            # No breakthrough detected - recommendations for building toward breakthrough
            top_type = max(combined_assessment.items(), 
                          key=lambda x: x[1]["combined_score"])[0]
            
            recommendations.extend([
                f"Continue exploring themes related to {top_type.replace('_', ' ')}",
                "Build therapeutic alliance for deeper work",
                "Increase self-reflection and awareness practices",
                "Consider what might be ready to emerge"
            ])
        
        return recommendations
    
    def _record_breakthrough_event(self, breakthrough_details: Dict[str, Any], context: Dict[str, Any]):
        """Record breakthrough event in history."""
        breakthrough_record = {
            **breakthrough_details,
            "session_context": {
                "session_count": context.get("session_count", 0),
                "therapeutic_alliance": context.get("therapeutic_alliance_score", 0.5),
                "user_readiness": context.get("user_readiness", "unknown")
            }
        }
        
        self.breakthrough_history.append(breakthrough_record)
        
        # Keep only last 50 breakthrough events
        if len(self.breakthrough_history) > 50:
            self.breakthrough_history = self.breakthrough_history[-50:]
    
    def _get_breakthrough_historical_context(self) -> Dict[str, Any]:
        """Get historical context of breakthrough events."""
        if not self.breakthrough_history:
            return {
                "total_breakthroughs": 0,
                "breakthrough_types": {},
                "average_intensity": "unknown",
                "last_breakthrough": None
            }
        
        # Analyze breakthrough types
        type_counts = {}
        intensities = []
        
        for breakthrough in self.breakthrough_history:
            breakthrough_type = breakthrough["type"]
            type_counts[breakthrough_type] = type_counts.get(breakthrough_type, 0) + 1
            intensities.append(breakthrough["intensity"])
        
        # Calculate average intensity
        intensity_values = {"low": 1, "medium": 2, "high": 3}
        avg_intensity_value = sum(intensity_values.get(i, 2) for i in intensities) / len(intensities)
        
        if avg_intensity_value >= 2.5:
            avg_intensity = "high"
        elif avg_intensity_value >= 1.5:
            avg_intensity = "medium"
        else:
            avg_intensity = "low"
        
        return {
            "total_breakthroughs": len(self.breakthrough_history),
            "breakthrough_types": type_counts,
            "average_intensity": avg_intensity,
            "last_breakthrough": self.breakthrough_history[-1]["timestamp"] if self.breakthrough_history else None,
            "recent_breakthroughs": len([b for b in self.breakthrough_history 
                                       if self._is_recent_breakthrough(b)])
        }
    
    def _check_linguistic_feature(self, text: str, feature: str) -> bool:
        """Check for specific linguistic features in breakthrough context."""
        feature_patterns = {
            "present tense realizations": r'\b(realize|see|understand|know)\b',
            "metaphorical language": r'\b(like|as if|feels like|seems like)\b',
            "connecting language": r'\b(connects|links|relates|ties together)\b',
            "somatic language": r'\b(feel|body|heart|gut|chest|throat)\b',
            "emotional metaphors": r'\b(weight|burden|light|heavy|free|trapped)\b',
            "vulnerability expressions": r'\b(scared|vulnerable|afraid|open|raw)\b',
            "behavioral language": r'\b(do|act|behave|respond|react)\b',
            "choice language": r'\b(choose|decide|option|alternative|different)\b',
            "pattern recognition": r'\b(pattern|cycle|repeat|always|tendency)\b',
            "identity language": r'\b(I am|who I am|myself|identity|self)\b',
            "value statements": r'\b(important|matter|care about|value|believe)\b',
            "priority language": r'\b(priority|focus|goal|want|need)\b'
        }
        
        pattern = feature_patterns.get(feature, "")
        if pattern:
            return bool(re.search(pattern, text, re.IGNORECASE))
        return False
    
    def _get_related_emotions(self, emotions: List[str]) -> List[str]:
        """Get emotions related to the given emotion list."""
        emotion_clusters = {
            "joy": ["happiness", "excitement", "elation", "pleasure"],
            "sadness": ["grief", "sorrow", "melancholy", "despair"],
            "anger": ["frustration", "irritation", "rage", "annoyance"],
            "fear": ["anxiety", "worry", "panic", "nervousness"],
            "surprise": ["shock", "amazement", "wonder", "astonishment"],
            "relief": ["comfort", "ease", "peace", "calm"],
            "clarity": ["understanding", "insight", "awareness", "realization"]
        }
        
        related = []
        for emotion in emotions:
            for cluster_emotion, related_emotions in emotion_clusters.items():
                if emotion in related_emotions or cluster_emotion == emotion:
                    related.extend(related_emotions)
        
        return list(set(related))
    
    def _is_recent_breakthrough(self, breakthrough: Dict[str, Any], days: int = 7) -> bool:
        """Check if breakthrough occurred within recent timeframe."""
        try:
            breakthrough_time = datetime.fromisoformat(breakthrough["timestamp"])
            return (datetime.now() - breakthrough_time).days <= days
        except (ValueError, TypeError, KeyError, AttributeError):
            # Invalid or missing timestamp - treat as not recent
            return False
    
    def _assess_temporal_breakthrough_alignment(self, temporal_type: str, text: str) -> float:
        """Assess how well temporal markers align with breakthrough indicators."""
        alignment_scores = {
            "immediate": 1.5,  # Strong alignment with breakthrough
            "recent": 1.0,     # Good alignment
            "ongoing": 0.5     # Moderate alignment (pattern recognition)
        }
        return alignment_scores.get(temporal_type, 0.5)
    
    def _get_intensity_modifier(self, intensity: str) -> float:
        """Get intensity modifier for breakthrough scoring."""
        modifiers = {
            "high": 1.5,
            "medium": 1.0,
            "low": 0.7
        }
        return modifiers.get(intensity, 1.0)
    
    def _categorize_breakthrough_intensity(self, score: float) -> str:
        """Categorize breakthrough intensity based on score."""
        if score >= 4.5:
            return "high"
        elif score >= 3.0:
            return "medium"
        else:
            return "low"
    
    def _categorize_breakthrough_readiness(self, potential: float) -> str:
        """Categorize breakthrough readiness level."""
        if potential >= 0.8:
            return "high"
        elif potential >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_validation_metrics(self, combined_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate validation metrics for breakthrough assessment."""
        if not combined_assessment:
            return {"overall_validity": 0.0, "consistency": 0.0}
        
        scores = [data["combined_score"] for data in combined_assessment.values()]
        
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        score_variance = sum((x - mean_score)**2 for x in scores) / len(scores)
        
        # Higher validity when there's a clear winner with low variance
        score_clarity = max_score - mean_score
        overall_validity = min(1.0, (max_score / 5.0) * (1.0 - score_variance/10.0))
        consistency = max(0.0, 1.0 - score_variance)
        
        return {
            "overall_validity": overall_validity,
            "consistency": consistency,
            "score_clarity": score_clarity,
            "max_score": max_score,
            "score_variance": score_variance
        }
    
    def get_specialized_knowledge_query(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate specialized query for breakthrough detection knowledge."""
        query_components = [
            "therapeutic breakthrough",
            "insight moments",
            "transformational change"
        ]
        
        # Add context from detected patterns
        for breakthrough_type, patterns in self.breakthrough_patterns.items():
            for marker, _ in patterns["markers"]:
                if marker in user_input.lower():
                    query_components.append(f"{breakthrough_type.replace('_', ' ')} therapy")
                    break
        
        emotion = context.get("emotion_analysis", {}).get("primary_emotion", "")
        if emotion:
            query_components.append(f"breakthrough with {emotion}")
        
        return " ".join(query_components)
    
    def validate_assessment(self, assessment: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate breakthrough detection assessment."""
        issues = []
        
        # Check required fields
        required_fields = ["breakthrough_detected", "breakthrough_details", "breakthrough_potential"]
        for field in required_fields:
            if field not in assessment:
                issues.append(f"Missing required field: {field}")
        
        # Validate breakthrough detection logic
        if "breakthrough_detected" in assessment and "breakthrough_details" in assessment:
            detected = assessment["breakthrough_detected"]
            details = assessment["breakthrough_details"]
            
            if detected and not details.get("type"):
                issues.append("Breakthrough detected but no type specified")
            
            if detected and not details.get("score", 0) >= self.validation_thresholds["minimum_score"]:
                issues.append("Breakthrough detected but score below threshold")
        
        # Validate breakthrough potential
        if "breakthrough_potential" in assessment:
            potential = assessment["breakthrough_potential"].get("overall_potential", 0)
            if not isinstance(potential, (int, float)) or not (0 <= potential <= 1):
                issues.append("Invalid breakthrough potential score")
        
        return len(issues) == 0, issues
    
    def _get_agent_specific_confidence(self, assessment: Dict[str, Any]) -> float:
        """Calculate agent-specific confidence adjustments."""
        base_confidence = 1.0
        
        # Adjust based on validation metrics
        validation_metrics = assessment.get("validation_metrics", {})
        validity = validation_metrics.get("overall_validity", 0.5)
        consistency = validation_metrics.get("consistency", 0.5)
        
        base_confidence *= (0.5 + 0.5 * validity * consistency)
        
        # Adjust based on historical accuracy
        if len(self.breakthrough_history) > 3:
            # Simple confidence adjustment based on breakthrough frequency
            # (More sophisticated validation would track prediction accuracy)
            recent_frequency = len([b for b in self.breakthrough_history[-10:] 
                                  if self._is_recent_breakthrough(b, days=30)])
            if recent_frequency > 5:
                base_confidence *= 0.9  # Slight reduction if detecting too frequently
        
        return min(1.0, base_confidence)
    
    def _extract_key_findings(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from breakthrough assessment."""
        return {
            "breakthrough_detected": assessment.get("breakthrough_detected"),
            "breakthrough_type": assessment.get("breakthrough_details", {}).get("type"),
            "breakthrough_intensity": assessment.get("breakthrough_details", {}).get("intensity"),
            "breakthrough_potential": assessment.get("breakthrough_potential", {}).get("overall_potential"),
            "validation_passed": assessment.get("breakthrough_details", {}).get("validation_passed")
        }