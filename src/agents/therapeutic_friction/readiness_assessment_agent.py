"""
Readiness Assessment Agent for Therapeutic Friction.

Specializes in assessing user readiness for therapeutic challenges through
sophisticated NLP analysis, emotional pattern recognition, and behavioral indicators.
"""

from typing import Dict, List, Optional, Any
import re
from datetime import datetime

from .base_friction_agent import BaseFrictionAgent, FrictionAgentType, UserReadinessIndicator


class ReadinessAssessmentAgent(BaseFrictionAgent):
    """
    Specialized agent for assessing user readiness for therapeutic challenges.
    
    Uses advanced pattern recognition to evaluate user openness, resistance,
    and capacity for therapeutic work.
    """
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the readiness assessment agent."""
        super().__init__(
            model_provider=model_provider,
            agent_type=FrictionAgentType.READINESS_ASSESSMENT,
            config=config
        )
        
        # Enhanced readiness indicators with weights
        self.readiness_patterns = {
            UserReadinessIndicator.RESISTANT: {
                "verbal_patterns": [
                    ("won't work", 3.0), ("tried everything", 2.5), ("pointless", 2.0),
                    ("don't want to", 2.0), ("can't change", 2.5), ("waste of time", 2.0),
                    ("already know", 1.5), ("doesn't help", 1.5)
                ],
                "emotional_markers": ["anger", "frustration", "hopelessness", "cynicism"],
                "linguistic_features": ["absolute negatives", "closed questions", "dismissive language"],
                "threshold": 3.0
            },
            UserReadinessIndicator.DEFENSIVE: {
                "verbal_patterns": [
                    ("yes but", 2.5), ("it's not my fault", 2.0), ("I know", 1.5),
                    ("already tried", 1.5), ("everyone else", 1.0), ("they made me", 2.0)
                ],
                "emotional_markers": ["anxiety", "shame", "vulnerability", "irritation"],
                "linguistic_features": ["justification", "blame externalization", "intellectualization"],
                "threshold": 2.5
            },
            UserReadinessIndicator.AMBIVALENT: {
                "verbal_patterns": [
                    ("maybe", 1.0), ("sometimes", 1.0), ("I guess", 1.5),
                    ("kind of", 1.0), ("not sure", 1.5), ("might", 1.0)
                ],
                "emotional_markers": ["confusion", "uncertainty", "conflicted", "neutral"],
                "linguistic_features": ["hedging language", "uncertainty markers", "mixed signals"],
                "threshold": 2.0
            },
            UserReadinessIndicator.OPEN: {
                "verbal_patterns": [
                    ("what do you think", 2.0), ("how can I", 2.5), ("tell me more", 2.0),
                    ("I'm willing", 3.0), ("want to understand", 2.5), ("help me", 2.0)
                ],
                "emotional_markers": ["curiosity", "hope", "motivation", "interest"],
                "linguistic_features": ["open questions", "exploration language", "receptive tone"],
                "threshold": 3.0
            },
            UserReadinessIndicator.MOTIVATED: {
                "verbal_patterns": [
                    ("want to change", 3.5), ("ready to try", 3.0), ("what's next", 2.5),
                    ("committed", 3.0), ("determined", 2.5), ("need to do", 2.0)
                ],
                "emotional_markers": ["determination", "optimism", "energy", "resolve"],
                "linguistic_features": ["action language", "commitment statements", "future focus"],
                "threshold": 4.0
            },
            UserReadinessIndicator.BREAKTHROUGH_READY: {
                "verbal_patterns": [
                    ("I see it now", 4.0), ("makes sense", 3.0), ("understand", 2.5),
                    ("ready for more", 3.5), ("want to go deeper", 3.0), ("push me", 2.5)
                ],
                "emotional_markers": ["clarity", "empowerment", "transformation", "insight"],
                "linguistic_features": ["insight language", "pattern recognition", "meta-awareness"],
                "threshold": 4.5
            }
        }
        
        # Contextual modifiers
        self.contextual_factors = {
            "session_history": 0.2,  # Weight for historical readiness
            "emotional_congruence": 0.3,  # Weight for emotion-language alignment
            "engagement_level": 0.2,  # Weight for response depth/length
            "temporal_consistency": 0.3  # Weight for consistency over time
        }
        
        # Historical readiness tracking
        self.readiness_history = []
        self.engagement_metrics = []
    
    async def assess(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess user readiness for therapeutic challenges."""
        try:
            # Primary assessment: Pattern matching
            pattern_scores = self._analyze_verbal_patterns(user_input)
            
            # Secondary assessment: Emotional alignment
            emotion_alignment = self._assess_emotional_alignment(user_input, context)
            
            # Tertiary assessment: Engagement analysis
            engagement_analysis = self._analyze_engagement_level(user_input, context)
            
            # Quaternary assessment: Historical context
            historical_context = self._analyze_historical_readiness(context)
            
            # Combine assessments with weighted scoring
            combined_assessment = self._combine_assessments(
                pattern_scores, emotion_alignment, engagement_analysis, historical_context
            )
            
            # Determine primary readiness indicator
            primary_readiness = self._determine_primary_readiness(combined_assessment)
            
            # Calculate confidence and stability metrics
            confidence_metrics = self._calculate_confidence_metrics(combined_assessment, context)
            
            # Store in history for future analysis
            self._update_readiness_history(primary_readiness, confidence_metrics)
            
            assessment_result = {
                "primary_readiness": primary_readiness.value,
                "readiness_score": combined_assessment.get(primary_readiness, 0.0),
                "pattern_analysis": pattern_scores,
                "emotion_alignment": emotion_alignment,
                "engagement_analysis": engagement_analysis,
                "historical_context": historical_context,
                "confidence_metrics": confidence_metrics,
                "readiness_trajectory": self._calculate_readiness_trajectory(),
                "recommendations": self._generate_readiness_recommendations(primary_readiness, confidence_metrics)
            }
            
            return assessment_result
            
        except Exception as e:
            self.logger.error(f"Error in readiness assessment: {str(e)}")
            return {
                "primary_readiness": UserReadinessIndicator.AMBIVALENT.value,
                "readiness_score": 0.0,
                "error": str(e)
            }
    
    def _analyze_verbal_patterns(self, user_input: str) -> Dict[UserReadinessIndicator, float]:
        """Analyze verbal patterns to assess readiness indicators."""
        user_input_lower = user_input.lower()
        pattern_scores = {indicator: 0.0 for indicator in UserReadinessIndicator}
        
        for indicator, patterns in self.readiness_patterns.items():
            score = 0.0
            
            # Check verbal patterns with weights
            for pattern, weight in patterns["verbal_patterns"]:
                if pattern in user_input_lower:
                    score += weight
                    self.logger.debug(f"Found pattern '{pattern}' for {indicator.value}, weight: {weight}")
            
            # Check linguistic features (simplified for now)
            for feature in patterns["linguistic_features"]:
                if self._check_linguistic_feature(user_input_lower, feature):
                    score += 1.0
            
            pattern_scores[indicator] = score
        
        return pattern_scores
    
    def _assess_emotional_alignment(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess alignment between emotional context and readiness patterns."""
        emotion_analysis = context.get("emotion_analysis", {})
        primary_emotion = emotion_analysis.get("primary_emotion", "").lower()
        emotion_confidence = emotion_analysis.get("confidence", 0.0)
        
        alignment_scores = {}
        
        for indicator, patterns in self.readiness_patterns.items():
            alignment_score = 0.0
            
            if primary_emotion in [e.lower() for e in patterns["emotional_markers"]]:
                alignment_score = emotion_confidence * 2.0  # Strong alignment
            elif primary_emotion and primary_emotion not in [e.lower() for e in patterns["emotional_markers"]]:
                alignment_score = -0.5  # Misalignment penalty
            
            alignment_scores[indicator.value] = alignment_score
        
        return {
            "primary_emotion": primary_emotion,
            "emotion_confidence": emotion_confidence,
            "alignment_scores": alignment_scores,
            "overall_alignment": max(alignment_scores.values()) if alignment_scores else 0.0
        }
    
    def _analyze_engagement_level(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user engagement level as indicator of readiness."""
        input_length = len(user_input)
        word_count = len(user_input.split())
        sentence_count = len([s for s in user_input.split('.') if s.strip()])
        
        # Calculate engagement metrics
        engagement_score = min(1.0, (input_length / 200.0))  # Normalize to 0-1
        detail_score = min(1.0, (word_count / 50.0))  # Normalize to 0-1
        complexity_score = min(1.0, (sentence_count / 5.0))  # Normalize to 0-1
        
        # Check for engagement indicators
        engagement_indicators = [
            "question", "example", "detail", "specific", "experience", "feeling"
        ]
        
        indicator_count = sum(1 for indicator in engagement_indicators 
                            if indicator in user_input.lower())
        indicator_score = min(1.0, indicator_count / 3.0)
        
        overall_engagement = (engagement_score + detail_score + complexity_score + indicator_score) / 4.0
        
        # Store for historical tracking
        self.engagement_metrics.append({
            "timestamp": datetime.now().isoformat(),
            "engagement_score": overall_engagement,
            "input_length": input_length,
            "word_count": word_count
        })
        
        return {
            "engagement_score": engagement_score,
            "detail_score": detail_score,
            "complexity_score": complexity_score,
            "indicator_score": indicator_score,
            "overall_engagement": overall_engagement,
            "engagement_level": self._categorize_engagement_level(overall_engagement)
        }
    
    def _analyze_historical_readiness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical readiness patterns for context."""
        if len(self.readiness_history) < 2:
            return {
                "historical_trend": "insufficient_data",
                "stability_score": 0.5,
                "trajectory": "unknown"
            }
        
        recent_readiness = self.readiness_history[-5:]  # Last 5 assessments
        
        # Calculate trend
        readiness_values = [self._readiness_to_numeric(r["readiness"]) for r in recent_readiness]
        
        if len(readiness_values) >= 3:
            early_avg = sum(readiness_values[:len(readiness_values)//2]) / (len(readiness_values)//2)
            late_avg = sum(readiness_values[len(readiness_values)//2:]) / (len(readiness_values) - len(readiness_values)//2)
            trend_direction = late_avg - early_avg
        else:
            trend_direction = 0
        
        # Calculate stability
        if len(readiness_values) > 1:
            variance = sum((x - sum(readiness_values)/len(readiness_values))**2 for x in readiness_values) / len(readiness_values)
            stability_score = max(0.0, 1.0 - (variance / 2.0))
        else:
            stability_score = 0.5
        
        trajectory = "improving" if trend_direction > 0.5 else "declining" if trend_direction < -0.5 else "stable"
        
        return {
            "historical_trend": trend_direction,
            "stability_score": stability_score,
            "trajectory": trajectory,
            "recent_assessments": len(recent_readiness),
            "average_readiness": sum(readiness_values) / len(readiness_values)
        }
    
    def _check_linguistic_feature(self, text: str, feature: str) -> bool:
        """Check for specific linguistic features in text."""
        feature_patterns = {
            "absolute negatives": r'\b(never|nothing|nobody|nowhere|no one|none)\b',
            "closed questions": r'\b(is|are|do|does|did|will|would|could|should)\b.*\?',
            "dismissive language": r'\b(whatever|fine|okay|sure)\b',
            "justification": r'\bbecause|since|due to\b',
            "blame externalization": r'\bthey|them|he|she|it\b.*\b(made|forced|caused)\b',
            "intellectualization": r'\b(analyze|understand|think|believe|concept)\b',
            "hedging language": r'\b(maybe|perhaps|possibly|might|could)\b',
            "uncertainty markers": r'\b(not sure|don\'t know|unclear|confused)\b',
            "mixed signals": r'\bbut|however|although|though\b',
            "open questions": r'\b(what|how|why|when|where)\b.*\?',
            "exploration language": r'\b(explore|discover|learn|understand)\b',
            "receptive tone": r'\b(willing|open|ready|interested)\b',
            "action language": r'\b(do|act|try|attempt|start|begin)\b',
            "commitment statements": r'\b(will|going to|committed|dedicated)\b',
            "future focus": r'\b(will|going to|plan|intend|hope)\b',
            "insight language": r'\b(realize|understand|see|recognize|notice)\b',
            "pattern recognition": r'\b(pattern|always|usually|tend to)\b',
            "meta-awareness": r'\b(aware|conscious|mindful|notice)\b'
        }
        
        pattern = feature_patterns.get(feature, "")
        if pattern:
            return bool(re.search(pattern, text, re.IGNORECASE))
        return False
    
    def _combine_assessments(self, pattern_scores: Dict, emotion_alignment: Dict,
                           engagement_analysis: Dict, historical_context: Dict) -> Dict[UserReadinessIndicator, float]:
        """Combine all assessment components with weighted scoring."""
        combined_scores = {}
        
        for indicator in UserReadinessIndicator:
            # Base score from pattern analysis
            pattern_score = pattern_scores.get(indicator, 0.0)
            
            # Emotional alignment modifier
            emotion_modifier = emotion_alignment["alignment_scores"].get(indicator.value, 0.0)
            
            # Engagement modifier
            engagement_modifier = engagement_analysis["overall_engagement"] * 2.0  # Boost for high engagement
            
            # Historical modifier
            if historical_context["trajectory"] == "improving":
                historical_modifier = 0.5
            elif historical_context["trajectory"] == "declining":
                historical_modifier = -0.5
            else:
                historical_modifier = 0.0
            
            # Combined weighted score
            combined_score = (
                pattern_score * 0.4 +
                emotion_modifier * 0.3 +
                engagement_modifier * 0.2 +
                historical_modifier * 0.1
            )
            
            combined_scores[indicator] = max(0.0, combined_score)
        
        return combined_scores
    
    def _determine_primary_readiness(self, combined_scores: Dict[UserReadinessIndicator, float]) -> UserReadinessIndicator:
        """Determine primary readiness indicator from combined scores."""
        if not combined_scores:
            return UserReadinessIndicator.AMBIVALENT
        
        # Find highest scoring indicator
        max_indicator = max(combined_scores.items(), key=lambda x: x[1])
        
        # Check if score meets threshold
        threshold = self.readiness_patterns[max_indicator[0]]["threshold"]
        
        if max_indicator[1] >= threshold:
            return max_indicator[0]
        else:
            # Default to ambivalent if no clear indicator
            return UserReadinessIndicator.AMBIVALENT
    
    def _calculate_confidence_metrics(self, combined_scores: Dict, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for the assessment."""
        scores_list = list(combined_scores.values())
        
        if not scores_list:
            return {"overall_confidence": 0.0, "score_variance": 0.0, "clarity": "low"}
        
        max_score = max(scores_list)
        mean_score = sum(scores_list) / len(scores_list)
        variance = sum((x - mean_score)**2 for x in scores_list) / len(scores_list)
        
        # Higher confidence when there's a clear winner with low variance
        score_clarity = max_score - mean_score
        overall_confidence = min(1.0, (max_score / 5.0) * (1.0 - variance/10.0) * (score_clarity + 1.0))
        
        clarity_level = "high" if score_clarity > 2.0 else "medium" if score_clarity > 1.0 else "low"
        
        return {
            "overall_confidence": overall_confidence,
            "max_score": max_score,
            "mean_score": mean_score,
            "score_variance": variance,
            "score_clarity": score_clarity,
            "clarity": clarity_level
        }
    
    def _update_readiness_history(self, readiness: UserReadinessIndicator, confidence_metrics: Dict[str, Any]):
        """Update historical readiness tracking."""
        self.readiness_history.append({
            "timestamp": datetime.now().isoformat(),
            "readiness": readiness,
            "confidence": confidence_metrics["overall_confidence"],
            "clarity": confidence_metrics["clarity"]
        })
        
        # Keep only last 20 assessments
        if len(self.readiness_history) > 20:
            self.readiness_history = self.readiness_history[-20:]
    
    def _calculate_readiness_trajectory(self) -> Dict[str, Any]:
        """Calculate readiness trajectory over time."""
        if len(self.readiness_history) < 3:
            return {"trajectory": "insufficient_data", "trend_strength": 0.0}
        
        recent_values = [self._readiness_to_numeric(r["readiness"]) for r in self.readiness_history[-5:]]
        
        # Simple linear trend calculation
        n = len(recent_values)
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(recent_values) / n
        
        numerator = sum((x_vals[i] - x_mean) * (recent_values[i] - y_mean) for i in range(n))
        denominator = sum((x - x_mean)**2 for x in x_vals)
        
        if denominator != 0:
            slope = numerator / denominator
            trend_strength = abs(slope)
            
            if slope > 0.3:
                trajectory = "improving"
            elif slope < -0.3:
                trajectory = "declining"
            else:
                trajectory = "stable"
        else:
            trajectory = "stable"
            trend_strength = 0.0
        
        return {
            "trajectory": trajectory,
            "trend_strength": trend_strength,
            "recent_assessments": n
        }
    
    def _generate_readiness_recommendations(self, readiness: UserReadinessIndicator, 
                                         confidence_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on readiness assessment."""
        recommendations = []
        
        if readiness == UserReadinessIndicator.RESISTANT:
            recommendations.extend([
                "Focus on validation and trust-building",
                "Avoid direct challenges or confrontation",
                "Use motivational interviewing techniques",
                "Explore ambivalence without pushing"
            ])
        elif readiness == UserReadinessIndicator.DEFENSIVE:
            recommendations.extend([
                "Use gentle inquiry and curiosity",
                "Avoid interpretation or direct feedback",
                "Reflect and normalize experiences",
                "Build therapeutic alliance gradually"
            ])
        elif readiness == UserReadinessIndicator.AMBIVALENT:
            recommendations.extend([
                "Explore both sides of ambivalence",
                "Use gentle challenges with support",
                "Increase awareness of patterns",
                "Build motivation for change"
            ])
        elif readiness == UserReadinessIndicator.OPEN:
            recommendations.extend([
                "Use moderate therapeutic challenges",
                "Introduce skill-building exercises",
                "Explore deeper patterns and themes",
                "Provide psychoeducation"
            ])
        elif readiness == UserReadinessIndicator.MOTIVATED:
            recommendations.extend([
                "Use strong therapeutic challenges",
                "Implement behavioral experiments",
                "Focus on skill practice and application",
                "Set specific therapeutic goals"
            ])
        elif readiness == UserReadinessIndicator.BREAKTHROUGH_READY:
            recommendations.extend([
                "Use maximum therapeutic leverage",
                "Facilitate deep insight work",
                "Challenge core beliefs and patterns",
                "Support integration of insights"
            ])
        
        # Add confidence-based recommendations
        if confidence_metrics["clarity"] == "low":
            recommendations.append("Consider additional assessment before intervention")
        
        return recommendations
    
    def _readiness_to_numeric(self, readiness: UserReadinessIndicator) -> float:
        """Convert readiness indicator to numeric value for trend analysis."""
        mapping = {
            UserReadinessIndicator.RESISTANT: 0.0,
            UserReadinessIndicator.DEFENSIVE: 1.0,
            UserReadinessIndicator.AMBIVALENT: 2.0,
            UserReadinessIndicator.OPEN: 3.0,
            UserReadinessIndicator.MOTIVATED: 4.0,
            UserReadinessIndicator.BREAKTHROUGH_READY: 5.0
        }
        return mapping.get(readiness, 2.0)
    
    def _categorize_engagement_level(self, score: float) -> str:
        """Categorize engagement level based on score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_specialized_knowledge_query(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate specialized query for readiness assessment knowledge."""
        emotion = context.get("emotion_analysis", {}).get("primary_emotion", "")
        
        query_components = [
            "user readiness assessment",
            "therapeutic alliance",
            "motivation for change"
        ]
        
        if emotion:
            query_components.append(f"readiness with {emotion}")
        
        # Add specific patterns found
        for indicator, patterns in self.readiness_patterns.items():
            for pattern, _ in patterns["verbal_patterns"]:
                if pattern in user_input.lower():
                    query_components.append(f"{indicator.value} therapeutic response")
                    break
        
        return " ".join(query_components)
    
    def validate_assessment(self, assessment: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the quality and consistency of readiness assessment."""
        issues = []
        
        # Check required fields
        required_fields = ["primary_readiness", "readiness_score", "confidence_metrics"]
        for field in required_fields:
            if field not in assessment:
                issues.append(f"Missing required field: {field}")
        
        # Check score validity
        if "readiness_score" in assessment:
            score = assessment["readiness_score"]
            if not isinstance(score, (int, float)) or score < 0:
                issues.append("Invalid readiness score")
        
        # Check confidence metrics
        if "confidence_metrics" in assessment:
            confidence = assessment["confidence_metrics"].get("overall_confidence", 0)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                issues.append("Invalid confidence score")
        
        # Check readiness indicator validity
        if "primary_readiness" in assessment:
            try:
                UserReadinessIndicator(assessment["primary_readiness"])
            except ValueError:
                issues.append("Invalid readiness indicator")
        
        return len(issues) == 0, issues
    
    def _get_agent_specific_confidence(self, assessment: Dict[str, Any]) -> float:
        """Calculate agent-specific confidence adjustments."""
        base_confidence = 1.0
        
        # Adjust based on assessment clarity
        confidence_metrics = assessment.get("confidence_metrics", {})
        clarity = confidence_metrics.get("clarity", "medium")
        
        if clarity == "high":
            base_confidence *= 1.2
        elif clarity == "low":
            base_confidence *= 0.8
        
        # Adjust based on historical consistency
        if len(self.readiness_history) > 3:
            recent_confidences = [r["confidence"] for r in self.readiness_history[-3:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            base_confidence *= (0.7 + 0.3 * avg_confidence)
        
        return min(1.0, base_confidence)
    
    def _extract_key_findings(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from readiness assessment."""
        return {
            "primary_readiness": assessment.get("primary_readiness"),
            "readiness_score": assessment.get("readiness_score"),
            "confidence_level": assessment.get("confidence_metrics", {}).get("clarity"),
            "trajectory": assessment.get("readiness_trajectory", {}).get("trajectory"),
            "engagement_level": assessment.get("engagement_analysis", {}).get("engagement_level")
        }