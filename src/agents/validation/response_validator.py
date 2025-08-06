"""
Advanced Response Validation System for Mental Health AI.

This module provides comprehensive validation of AI agent responses using
multiple validation techniques including semantic analysis, clinical compliance,
and therapeutic appropriateness assessment.
"""

import re
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime

from src.utils.logger import get_logger
from src.knowledge.clinical.clinical_guidelines_db import ClinicalGuidelinesDB, ViolationSeverity
from src.utils.vector_db_integration import search_relevant_data

logger = get_logger(__name__)

class ValidationDimension(Enum):
    """Dimensions of response validation."""
    CLINICAL_ACCURACY = "clinical_accuracy"
    THERAPEUTIC_APPROPRIATENESS = "therapeutic_appropriateness"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    SAFETY_ASSESSMENT = "safety_assessment"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    PROFESSIONAL_BOUNDARIES = "professional_boundaries"
    EVIDENCE_BASED = "evidence_based"
    HARM_POTENTIAL = "harm_potential"

class RiskLevel(Enum):
    """Risk levels for validation results."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationScore:
    """Individual validation dimension score."""
    dimension: ValidationDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    issues_found: List[str]
    recommendations: List[str]
    supporting_evidence: List[str]

@dataclass
class ComprehensiveValidationResult:
    """Comprehensive validation result across all dimensions."""
    overall_score: float
    overall_risk: RiskLevel
    dimension_scores: Dict[ValidationDimension, ValidationScore]
    critical_issues: List[str]
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    alternative_response_needed: bool
    confidence_score: float
    validation_timestamp: str
    processing_time: float

class SemanticAnalyzer:
    """Semantic analysis for response validation."""
    
    def __init__(self, model_provider=None):
        self.model_provider = model_provider
        
        # Therapeutic concepts and their indicators
        self.therapeutic_concepts = {
            "empathy": [
                "understand how you feel", "that must be difficult", "I can see why",
                "it sounds like", "I hear that you", "that makes sense"
            ],
            "validation": [
                "your feelings are valid", "that's understandable", "many people feel",
                "it's normal to", "you're not alone", "that's a common experience"
            ],
            "hope": [
                "things can get better", "there is hope", "you have strengths",
                "positive changes are possible", "you've overcome before", "recovery is possible"
            ],
            "professional_support": [
                "professional help", "qualified therapist", "mental health professional",
                "clinical assessment", "professional guidance", "medical consultation"
            ]
        }
        
        # Harmful concepts to detect
        self.harmful_concepts = {
            "dismissive": [
                "just get over it", "it's not that bad", "everyone goes through this",
                "stop being dramatic", "you're overreacting", "just think positive"
            ],
            "minimizing": [
                "it could be worse", "at least you have", "others have it harder",
                "that's not really a problem", "you're lucky that", "be grateful for"
            ],
            "inappropriate_advice": [
                "you should leave them", "just quit your job", "cut off contact",
                "that person is toxic", "you need to", "you have to"
            ],
            "false_promises": [
                "everything will be fine", "this will definitely work", "you'll feel better soon",
                "guaranteed to help", "never worry again", "completely cured"
            ]
        }
    
    def analyze_therapeutic_quality(self, response_text: str, user_input: str = "") -> Dict[str, Any]:
        """Analyze therapeutic quality of response."""
        response_lower = response_text.lower()
        user_lower = user_input.lower()
        
        # Check for positive therapeutic concepts
        positive_score = 0
        positive_concepts_found = []
        
        for concept, indicators in self.therapeutic_concepts.items():
            concept_found = False
            for indicator in indicators:
                if indicator in response_lower:
                    positive_score += 1
                    concept_found = True
                    break
            if concept_found:
                positive_concepts_found.append(concept)
        
        # Check for harmful concepts
        negative_score = 0
        harmful_concepts_found = []
        
        for concept, indicators in self.harmful_concepts.items():
            for indicator in indicators:
                if indicator in response_lower:
                    negative_score += 1
                    harmful_concepts_found.append(concept)
                    break
        
        # Calculate overall therapeutic quality score
        max_positive = len(self.therapeutic_concepts)
        max_negative = len(self.harmful_concepts)
        
        positive_ratio = positive_score / max_positive if max_positive > 0 else 0
        negative_ratio = negative_score / max_negative if max_negative > 0 else 0
        
        # Therapeutic quality score (0-1, higher is better)
        therapeutic_score = max(0, positive_ratio - (negative_ratio * 1.5))
        
        return {
            "therapeutic_score": therapeutic_score,
            "positive_concepts": positive_concepts_found,
            "harmful_concepts": harmful_concepts_found,
            "empathy_indicators": sum(1 for indicator in self.therapeutic_concepts["empathy"] 
                                    if indicator in response_lower),
            "validation_indicators": sum(1 for indicator in self.therapeutic_concepts["validation"] 
                                       if indicator in response_lower),
            "hope_indicators": sum(1 for indicator in self.therapeutic_concepts["hope"] 
                                 if indicator in response_lower)
        }
    
    def analyze_response_appropriateness(self, response_text: str, user_input: str, 
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze appropriateness of response to user input."""
        # Extract emotional tone from user input
        user_emotion = self._detect_emotional_tone(user_input)
        response_emotion = self._detect_emotional_tone(response_text)
        
        # Check length appropriateness
        response_length = len(response_text.split())
        user_length = len(user_input.split())
        
        length_appropriate = True
        length_issues = []
        
        if response_length < 10:
            length_appropriate = False
            length_issues.append("Response too brief for meaningful therapeutic engagement")
        elif response_length > 300:
            length_appropriate = False
            length_issues.append("Response too lengthy, may overwhelm user")
        
        # Check emotional appropriateness
        emotion_appropriate = True
        emotion_issues = []
        
        if user_emotion == "distressed" and response_emotion == "cheerful":
            emotion_appropriate = False
            emotion_issues.append("Response tone mismatched to user's distressed state")
        elif user_emotion == "angry" and response_emotion == "dismissive":
            emotion_appropriate = False
            emotion_issues.append("Response may escalate user's anger")
        
        # Calculate appropriateness score
        appropriateness_score = 1.0
        if not length_appropriate:
            appropriateness_score -= 0.3
        if not emotion_appropriate:
            appropriateness_score -= 0.4
        
        return {
            "appropriateness_score": max(0, appropriateness_score),
            "length_appropriate": length_appropriate,
            "emotion_appropriate": emotion_appropriate,
            "user_emotion": user_emotion,
            "response_emotion": response_emotion,
            "issues": length_issues + emotion_issues
        }
    
    def _detect_emotional_tone(self, text: str) -> str:
        """Detect emotional tone of text."""
        text_lower = text.lower()
        
        # Emotional indicators
        distressed_words = ["sad", "depressed", "anxious", "worried", "scared", "hopeless", "overwhelmed"]
        angry_words = ["angry", "furious", "frustrated", "mad", "irritated", "annoyed"]
        positive_words = ["happy", "good", "great", "wonderful", "excited", "grateful"]
        calm_words = ["calm", "peaceful", "relaxed", "okay", "fine", "stable"]
        cheerful_words = ["cheerful", "upbeat", "optimistic", "bright", "joyful"]
        dismissive_words = ["whatever", "doesn't matter", "who cares", "so what"]
        
        # Count indicators
        distressed_count = sum(1 for word in distressed_words if word in text_lower)
        angry_count = sum(1 for word in angry_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        calm_count = sum(1 for word in calm_words if word in text_lower)
        cheerful_count = sum(1 for word in cheerful_words if word in text_lower)
        dismissive_count = sum(1 for word in dismissive_words if word in text_lower)
        
        # Determine dominant emotion
        emotion_scores = {
            "distressed": distressed_count,
            "angry": angry_count,
            "positive": positive_count,
            "calm": calm_count,
            "cheerful": cheerful_count,
            "dismissive": dismissive_count
        }
        
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        return max_emotion if emotion_scores[max_emotion] > 0 else "neutral"

class ClinicalComplianceValidator:
    """Validator for clinical compliance and professional standards."""
    
    def __init__(self):
        self.clinical_guidelines_db = ClinicalGuidelinesDB()
        
        # Professional language patterns
        self.professional_patterns = {
            "appropriate": [
                r"I understand", r"Let's explore", r"How does that feel",
                r"What would be helpful", r"I'm here to support",
                r"professional help", r"qualified therapist"
            ],
            "inappropriate": [
                r"you should definitely", r"you must", r"you have to",
                r"the answer is", r"just do this", r"trust me",
                r"I know what's best", r"listen to me"
            ]
        }
    
    def validate_clinical_compliance(self, response_text: str, user_input: str = "") -> Dict[str, Any]:
        """Validate response against clinical guidelines."""
        # Use clinical guidelines database for validation
        validation_result = self.clinical_guidelines_db.validate_response(response_text, user_input)
        
        # Check professional language patterns
        professional_score = self._assess_professional_language(response_text)
        
        # Check for scope of practice violations
        scope_violations = self._check_scope_violations(response_text)
        
        # Assess overall clinical compliance
        compliance_score = 1.0
        
        # Deduct points for violations
        if validation_result["violations"]:
            severe_violations = [v for v in validation_result["violations"] 
                               if v["severity"] in ["severe", "critical"]]
            compliance_score -= len(severe_violations) * 0.3
            compliance_score -= (len(validation_result["violations"]) - len(severe_violations)) * 0.1
        
        # Adjust for professional language
        compliance_score = (compliance_score + professional_score) / 2
        
        # Adjust for scope violations
        compliance_score -= len(scope_violations) * 0.2
        
        compliance_score = max(0, compliance_score)
        
        return {
            "compliance_score": compliance_score,
            "guideline_violations": validation_result["violations"],
            "professional_language_score": professional_score,
            "scope_violations": scope_violations,
            "risk_level": validation_result["risk_level"],
            "recommendations": validation_result["recommendations"]
        }
    
    def _assess_professional_language(self, response_text: str) -> float:
        """Assess professional language usage."""
        response_lower = response_text.lower()
        
        appropriate_count = 0
        inappropriate_count = 0
        
        for pattern in self.professional_patterns["appropriate"]:
            if re.search(pattern, response_lower):
                appropriate_count += 1
        
        for pattern in self.professional_patterns["inappropriate"]:
            if re.search(pattern, response_lower):
                inappropriate_count += 1
        
        # Calculate professional language score
        total_patterns = len(self.professional_patterns["appropriate"]) + len(self.professional_patterns["inappropriate"])
        if total_patterns == 0:
            return 0.8  # Default moderate score
        
        professional_score = (appropriate_count - inappropriate_count * 2) / total_patterns
        return max(0, min(1, professional_score + 0.8))  # Bias toward positive
    
    def _check_scope_violations(self, response_text: str) -> List[str]:
        """Check for scope of practice violations."""
        violations = []
        response_lower = response_text.lower()
        
        # Medical diagnosis violations
        diagnostic_patterns = [
            r"you have \w+ disorder", r"diagnosed with", r"you suffer from",
            r"this is definitely", r"you clearly have"
        ]
        
        for pattern in diagnostic_patterns:
            if re.search(pattern, response_lower):
                violations.append("Inappropriate diagnostic statement")
                break
        
        # Medication advice violations
        medication_patterns = [
            r"take \w+ medication", r"stop taking", r"increase dosage",
            r"try this drug", r"don't take your"
        ]
        
        for pattern in medication_patterns:
            if re.search(pattern, response_lower):
                violations.append("Inappropriate medication advice")
                break
        
        # Legal advice violations
        legal_patterns = [
            r"you should sue", r"file a complaint", r"take legal action",
            r"get a lawyer", r"press charges"
        ]
        
        for pattern in legal_patterns:
            if re.search(pattern, response_lower):
                violations.append("Inappropriate legal advice")
                break
        
        return violations

class SafetyAssessmentValidator:
    """Validator for safety assessment and crisis detection."""
    
    def __init__(self):
        self.crisis_indicators = {
            "suicide": [
                "kill myself", "end my life", "suicide", "not worth living",
                "better off dead", "want to die", "end it all"
            ],
            "self_harm": [
                "hurt myself", "cut myself", "harm myself", "punish myself",
                "self-injury", "self-harm", "pain myself"
            ],
            "violence": [
                "hurt someone", "kill them", "violence", "attack",
                "harm others", "dangerous thoughts", "violent urges"
            ],
            "substance_abuse": [
                "overdose", "too many pills", "drink heavily", "use drugs",
                "substance abuse", "addiction", "getting high"
            ],
            "psychosis": [
                "hearing voices", "seeing things", "not real", "conspiracy",
                "paranoid", "delusional", "losing touch"
            ]
        }
        
        self.protective_factors = [
            "support system", "family", "friends", "therapist", "doctor",
            "reasons to live", "hope", "future plans", "goals", "pets",
            "responsibilities", "children", "job", "school"
        ]
    
    def assess_safety_risk(self, response_text: str, user_input: str = "") -> Dict[str, Any]:
        """Assess safety risk in user input and response appropriateness."""
        combined_text = f"{user_input} {response_text}".lower()
        
        # Detect crisis indicators
        crisis_detected = {}
        total_crisis_score = 0
        
        for crisis_type, indicators in self.crisis_indicators.items():
            crisis_score = 0
            detected_indicators = []
            
            for indicator in indicators:
                if indicator in combined_text:
                    crisis_score += 1
                    detected_indicators.append(indicator)
            
            if crisis_score > 0:
                crisis_detected[crisis_type] = {
                    "score": crisis_score,
                    "indicators": detected_indicators
                }
                total_crisis_score += crisis_score
        
        # Check for protective factors
        protective_score = 0
        protective_found = []
        
        for factor in self.protective_factors:
            if factor in combined_text:
                protective_score += 1
                protective_found.append(factor)
        
        # Assess response appropriateness to crisis
        response_appropriate = True
        response_issues = []
        
        if crisis_detected and not self._contains_safety_response(response_text):
            response_appropriate = False
            response_issues.append("Response lacks appropriate safety intervention for detected crisis")
        
        # Calculate overall safety score
        safety_score = 1.0
        
        if total_crisis_score > 0:
            crisis_weight = min(total_crisis_score / 10, 1.0)  # Max weight of 1.0
            protective_weight = min(protective_score / 5, 0.5)  # Max weight of 0.5
            safety_score = max(0, 1.0 - crisis_weight + protective_weight)
        
        if not response_appropriate:
            safety_score *= 0.5  # Significantly reduce score for inappropriate response
        
        # Determine risk level
        if total_crisis_score >= 3 or "suicide" in crisis_detected:
            risk_level = RiskLevel.CRITICAL
        elif total_crisis_score >= 2:
            risk_level = RiskLevel.HIGH
        elif total_crisis_score >= 1:
            risk_level = RiskLevel.MODERATE
        elif protective_score > 0:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
        
        return {
            "safety_score": safety_score,
            "risk_level": risk_level,
            "crisis_indicators": crisis_detected,
            "protective_factors": protective_found,
            "response_appropriate": response_appropriate,
            "response_issues": response_issues,
            "requires_immediate_intervention": risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        }
    
    def _contains_safety_response(self, response_text: str) -> bool:
        """Check if response contains appropriate safety intervention."""
        response_lower = response_text.lower()
        
        safety_responses = [
            "crisis", "safety", "professional help", "emergency",
            "hotline", "911", "988", "immediate support",
            "mental health professional", "doctor", "therapist"
        ]
        
        return any(safety_term in response_lower for safety_term in safety_responses)

class ComprehensiveResponseValidator:
    """Main comprehensive response validator."""
    
    def __init__(self, model_provider=None):
        self.semantic_analyzer = SemanticAnalyzer(model_provider)
        self.clinical_validator = ClinicalComplianceValidator()
        self.safety_validator = SafetyAssessmentValidator()
        self.model_provider = model_provider
    
    async def validate_response(self, agent_name: str, response_text: str, 
                              user_input: str, context: Dict[str, Any] = None) -> ComprehensiveValidationResult:
        """Perform comprehensive validation of agent response."""
        start_time = datetime.now()
        
        # Initialize validation scores
        dimension_scores = {}
        critical_issues = []
        blocking_issues = []
        warnings = []
        recommendations = []
        
        try:
            # Clinical accuracy validation
            clinical_result = self.clinical_validator.validate_clinical_compliance(response_text, user_input)
            clinical_score = ValidationScore(
                dimension=ValidationDimension.CLINICAL_ACCURACY,
                score=clinical_result["compliance_score"],
                confidence=0.8,
                risk_level=self._convert_to_risk_level(clinical_result["compliance_score"]),
                issues_found=[v["message"] for v in clinical_result["guideline_violations"]],
                recommendations=clinical_result["recommendations"],
                supporting_evidence=[]
            )
            dimension_scores[ValidationDimension.CLINICAL_ACCURACY] = clinical_score
            
            # Therapeutic appropriateness validation
            therapeutic_result = self.semantic_analyzer.analyze_therapeutic_quality(response_text, user_input)
            appropriateness_result = self.semantic_analyzer.analyze_response_appropriateness(
                response_text, user_input, context
            )
            
            therapeutic_score = ValidationScore(
                dimension=ValidationDimension.THERAPEUTIC_APPROPRIATENESS,
                score=(therapeutic_result["therapeutic_score"] + appropriateness_result["appropriateness_score"]) / 2,
                confidence=0.7,
                risk_level=self._convert_to_risk_level(therapeutic_result["therapeutic_score"]),
                issues_found=therapeutic_result["harmful_concepts"] + appropriateness_result["issues"],
                recommendations=self._generate_therapeutic_recommendations(therapeutic_result),
                supporting_evidence=therapeutic_result["positive_concepts"]
            )
            dimension_scores[ValidationDimension.THERAPEUTIC_APPROPRIATENESS] = therapeutic_score
            
            # Safety assessment validation
            safety_result = self.safety_validator.assess_safety_risk(response_text, user_input)
            safety_score = ValidationScore(
                dimension=ValidationDimension.SAFETY_ASSESSMENT,
                score=safety_result["safety_score"],
                confidence=0.9,
                risk_level=safety_result["risk_level"],
                issues_found=safety_result["response_issues"],
                recommendations=self._generate_safety_recommendations(safety_result),
                supporting_evidence=safety_result["protective_factors"]
            )
            dimension_scores[ValidationDimension.SAFETY_ASSESSMENT] = safety_score
            
            # Collect critical issues and blocking issues
            for dimension, score in dimension_scores.items():
                if score.risk_level == RiskLevel.CRITICAL:
                    critical_issues.extend(score.issues_found)
                    if dimension == ValidationDimension.SAFETY_ASSESSMENT:
                        blocking_issues.extend(score.issues_found)
                elif score.risk_level == RiskLevel.HIGH:
                    if dimension in [ValidationDimension.SAFETY_ASSESSMENT, ValidationDimension.CLINICAL_ACCURACY]:
                        blocking_issues.extend(score.issues_found)
                    else:
                        warnings.extend(score.issues_found)
                elif score.risk_level == RiskLevel.MODERATE:
                    warnings.extend(score.issues_found)
                
                recommendations.extend(score.recommendations)
            
            # Calculate overall score and risk
            overall_score = np.mean([score.score for score in dimension_scores.values()])
            overall_risk = max([score.risk_level for score in dimension_scores.values()], 
                             key=lambda x: self._risk_level_to_numeric(x))
            
            # Determine if alternative response is needed
            alternative_needed = (
                overall_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH] or
                len(blocking_issues) > 0 or
                overall_score < 0.4
            )
            
            # Calculate confidence score
            confidence_score = np.mean([score.confidence for score in dimension_scores.values()])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ComprehensiveValidationResult(
                overall_score=overall_score,
                overall_risk=overall_risk,
                dimension_scores=dimension_scores,
                critical_issues=list(set(critical_issues)),
                blocking_issues=list(set(blocking_issues)),
                warnings=list(set(warnings)),
                recommendations=list(set(recommendations)),
                alternative_response_needed=alternative_needed,
                confidence_score=confidence_score,
                validation_timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {str(e)}")
            
            # Return default validation result in case of error
            return ComprehensiveValidationResult(
                overall_score=0.5,
                overall_risk=RiskLevel.MODERATE,
                dimension_scores={},
                critical_issues=[f"Validation error: {str(e)}"],
                blocking_issues=[],
                warnings=["Validation system encountered an error"],
                recommendations=["Manual review recommended due to validation error"],
                alternative_response_needed=True,
                confidence_score=0.3,
                validation_timestamp=datetime.now().isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _convert_to_risk_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 0.8:
            return RiskLevel.MINIMAL
        elif score >= 0.6:
            return RiskLevel.LOW
        elif score >= 0.4:
            return RiskLevel.MODERATE
        elif score >= 0.2:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _risk_level_to_numeric(self, risk_level: RiskLevel) -> int:
        """Convert risk level to numeric value for comparison."""
        risk_values = {
            RiskLevel.MINIMAL: 1,
            RiskLevel.LOW: 2,
            RiskLevel.MODERATE: 3,
            RiskLevel.HIGH: 4,
            RiskLevel.CRITICAL: 5
        }
        return risk_values.get(risk_level, 3)
    
    def _generate_therapeutic_recommendations(self, therapeutic_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for therapeutic improvement."""
        recommendations = []
        
        if therapeutic_result["therapeutic_score"] < 0.6:
            recommendations.append("Increase empathetic language and validation")
        
        if therapeutic_result["empathy_indicators"] == 0:
            recommendations.append("Include more empathetic responses")
        
        if therapeutic_result["validation_indicators"] == 0:
            recommendations.append("Add validation of user's experiences")
        
        if therapeutic_result["harmful_concepts"]:
            recommendations.append("Remove dismissive or minimizing language")
        
        return recommendations
    
    def _generate_safety_recommendations(self, safety_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for safety improvement."""
        recommendations = []
        
        if safety_result["requires_immediate_intervention"]:
            recommendations.append("Immediate crisis intervention required")
            recommendations.append("Provide crisis resources and professional referral")
        
        if not safety_result["response_appropriate"]:
            recommendations.append("Include appropriate safety responses")
        
        if safety_result["crisis_indicators"]:
            recommendations.append("Address safety concerns directly")
        
        return recommendations