"""
Comprehensive SupervisorAgent for Quality Assurance and Oversight.

This agent provides quality assurance and oversight for all other agents in the mental health AI system,
ensuring clinical accuracy, ethical compliance, and therapeutic boundaries.
"""

import asyncio
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np

from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.vector_db_integration import get_central_vector_db, add_user_data, search_relevant_data
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService

logger = get_logger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    PASS = "pass"
    WARNING = "warning" 
    CRITICAL = "critical"
    BLOCKED = "blocked"

class ClinicalRiskLevel(Enum):
    """Clinical risk assessment levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

class EthicalConcern(Enum):
    """Types of ethical concerns."""
    BOUNDARY_VIOLATION = "boundary_violation"
    DUAL_RELATIONSHIP = "dual_relationship"
    CONFIDENTIALITY_BREACH = "confidentiality_breach" 
    COMPETENCY_EXCEEDED = "competency_exceeded"
    HARM_POTENTIAL = "harm_potential"
    BIAS_DETECTED = "bias_detected"

@dataclass
class ValidationResult:
    """Result of agent response validation."""
    validation_level: ValidationLevel
    clinical_risk: ClinicalRiskLevel
    ethical_concerns: List[EthicalConcern]
    accuracy_score: float
    consistency_score: float
    appropriateness_score: float
    recommendations: List[str]
    blocked_content: Optional[str] = None
    alternative_response: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AgentInteraction:
    """Tracked interaction between agents."""
    session_id: str
    agent_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    validation_result: ValidationResult
    processing_time: float
    timestamp: str
    user_feedback: Optional[Dict[str, Any]] = None

class SupervisorAgent(BaseAgent):
    """
    Comprehensive SupervisorAgent providing quality assurance and oversight.
    
    This agent validates all other agent responses for clinical accuracy,
    ethical compliance, consistency, and therapeutic appropriateness.
    """
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the supervisor agent.
        
        Args:
            model_provider: LLM provider for embeddings and analysis
            config: Configuration dictionary
        """
        super().__init__(
            model=model_provider,
            name="supervisor_agent",
            role="Clinical Quality Assurance Supervisor",
            description="Comprehensive quality assurance and oversight for mental health AI agents"
        )
        
        self.config = config or {}
        self.technique_service = TherapeuticTechniqueService(model_provider)
        self.model_provider = model_provider
        
        # Initialize clinical guidelines knowledge base
        self._initialize_clinical_guidelines()
        
        # Performance metrics tracking
        self.metrics = {
            "total_validations": 0,
            "blocked_responses": 0,
            "warning_responses": 0,
            "average_accuracy_score": 0.0,
            "average_processing_time": 0.0,
            "ethical_violations_detected": 0
        }
        
        # Interaction audit trail
        self.interaction_history: List[AgentInteraction] = []
        self.max_history_size = self.config.get("max_history_size", 10000)
        
        # Clinical validation patterns
        self.clinical_patterns = self._load_clinical_patterns()
        self.ethical_guidelines = self._load_ethical_guidelines()
        self.risk_indicators = self._load_risk_indicators()
        
        # Therapeutic boundaries
        self.therapeutic_boundaries = {
            "avoid_diagnosis_terms": [
                "you have", "you suffer from", "you are diagnosed with",
                "this confirms", "this proves", "definitely indicates"
            ],
            "crisis_escalation_required": [
                "suicide", "kill myself", "end my life", "hurt myself",
                "overdose", "jump off", "not worth living"
            ],
            "inappropriate_advice": [
                "stop taking medication", "don't see a doctor", "ignore your therapist",
                "medication is bad", "therapy doesn't work"
            ]
        }
        
        # NLP analysis tools
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        self.consistency_checker = ConsistencyChecker()
        
        logger.info("SupervisorAgent initialized with comprehensive validation capabilities")
    
    def _initialize_clinical_guidelines(self):
        """Initialize clinical guidelines vector database."""
        try:
            # Create clinical guidelines if they don't exist
            guidelines = self._get_default_clinical_guidelines()
            
            for guideline in guidelines:
                add_user_data("clinical_guidelines", guideline, guideline.get("id"))
                
            logger.info(f"Initialized {len(guidelines)} clinical guidelines")
            
        except Exception as e:
            logger.error(f"Error initializing clinical guidelines: {str(e)}")
    
    def _get_default_clinical_guidelines(self) -> List[Dict[str, Any]]:
        """Get default clinical guidelines."""
        return [
            {
                "id": "therapeutic_boundaries",
                "title": "Therapeutic Boundaries",
                "content": "AI agents must maintain clear therapeutic boundaries and avoid dual relationships",
                "category": "boundaries",
                "risk_level": "high"
            },
            {
                "id": "crisis_intervention",
                "title": "Crisis Intervention Protocol", 
                "content": "Immediate escalation required for suicide ideation or self-harm expressions",
                "category": "crisis",
                "risk_level": "severe"
            },
            {
                "id": "diagnostic_limitations",
                "title": "Diagnostic Limitations",
                "content": "AI agents cannot provide formal diagnoses and must refer to qualified professionals",
                "category": "limitations",
                "risk_level": "moderate"
            },
            {
                "id": "medication_advice",
                "title": "Medication Guidance",
                "content": "No medication recommendations or changes without qualified professional oversight",
                "category": "medication",
                "risk_level": "high"
            },
            {
                "id": "confidentiality",
                "title": "Confidentiality Standards",
                "content": "Protect user privacy and confidential information at all times",
                "category": "privacy",
                "risk_level": "high"
            }
        ]
    
    def _load_clinical_patterns(self) -> Dict[str, List[str]]:
        """Load clinical validation patterns."""
        return {
            "medication_advice": [
                r"take \w+ medication", r"stop taking", r"increase dosage",
                r"reduce medication", r"switch to", r"try this drug"
            ],
            "diagnostic_statements": [
                r"you have \w+ disorder", r"you are \w+", r"this indicates \w+ condition",
                r"you suffer from", r"diagnosis of", r"symptoms suggest"
            ],
            "crisis_language": [
                r"suicide|kill.*self", r"end.*life", r"hurt.*self",
                r"overdose", r"jump.*off", r"not worth living"
            ],
            "inappropriate_advice": [
                r"don't see.*doctor", r"ignore.*therapist", r"therapy.*useless",
                r"medication.*harmful", r"professionals.*wrong"
            ]
        }
    
    def _load_ethical_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Load ethical guidelines for validation."""
        return {
            "autonomy": {
                "description": "Respect client autonomy and decision-making capacity",
                "violations": ["forcing decisions", "manipulative language", "paternalistic advice"]
            },
            "beneficence": {
                "description": "Act in the client's best interest",
                "violations": ["harmful advice", "enabling destructive behavior", "inappropriate reassurance"]
            },
            "non_maleficence": {
                "description": "Do no harm to the client",
                "violations": ["triggering content", "retraumatization", "dangerous suggestions"]
            },
            "justice": {
                "description": "Provide fair and equitable treatment",
                "violations": ["discriminatory language", "bias against groups", "unequal treatment"]
            },
            "fidelity": {
                "description": "Maintain trustworthiness and reliability",
                "violations": ["false promises", "inconsistent advice", "unreliable information"]
            }
        }
    
    def _load_risk_indicators(self) -> Dict[str, List[str]]:
        """Load risk indicators for assessment."""
        return {
            "suicide_risk": [
                "suicide", "kill myself", "end it all", "not worth living",
                "better off dead", "tired of living", "hopeless"
            ],
            "self_harm": [
                "hurt myself", "cut myself", "harm myself", "punish myself",
                "deserve pain", "self-injury", "self-mutilation"
            ],
            "substance_abuse": [
                "need a drink", "getting high", "using drugs", "overdose",
                "can't stop drinking", "addiction", "substance abuse"
            ],
            "psychosis_indicators": [
                "hearing voices", "seeing things", "not real", "conspiracy",
                "they're watching", "paranoid", "delusional"
            ],
            "dissociation": [
                "not real", "detached", "floating", "watching myself",
                "dreamlike", "disconnected", "out of body"
            ]
        }
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis capability."""
        # Simple sentiment analyzer implementation
        # In production, you might use more sophisticated tools
        return SimpleSentimentAnalyzer()
    
    async def validate_agent_response(self, agent_name: str, input_data: Dict[str, Any],
                                    output_data: Dict[str, Any], session_id: str = None) -> ValidationResult:
        """
        Comprehensive validation of agent response.
        
        Args:
            agent_name: Name of the agent being validated
            input_data: Input that was provided to the agent
            output_data: Output generated by the agent
            session_id: Session identifier for tracking
            
        Returns:
            ValidationResult with comprehensive assessment
        """
        start_time = time.time()
        
        try:
            # Extract response text for analysis
            response_text = self._extract_response_text(output_data)
            user_input = self._extract_user_input(input_data)
            
            # Perform validation checks
            clinical_risk = await self._assess_clinical_risk(response_text, user_input)
            ethical_concerns = await self._check_ethical_compliance(response_text, user_input)
            accuracy_score = await self._calculate_accuracy_score(response_text, user_input, agent_name)
            consistency_score = await self._check_consistency(response_text, session_id, agent_name)
            appropriateness_score = await self._assess_appropriateness(response_text, user_input)
            
            # Determine overall validation level
            validation_level = self._determine_validation_level(
                clinical_risk, ethical_concerns, accuracy_score, 
                consistency_score, appropriateness_score
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                validation_level, clinical_risk, ethical_concerns,
                accuracy_score, consistency_score, appropriateness_score
            )
            
            # Check if response should be blocked
            blocked_content = None
            alternative_response = None
            
            if validation_level == ValidationLevel.BLOCKED:
                blocked_content = response_text
                alternative_response = await self._generate_alternative_response(
                    user_input, clinical_risk, ethical_concerns
                )
            
            # Create validation result
            result = ValidationResult(
                validation_level=validation_level,
                clinical_risk=clinical_risk,
                ethical_concerns=ethical_concerns,
                accuracy_score=accuracy_score,
                consistency_score=consistency_score,
                appropriateness_score=appropriateness_score,
                recommendations=recommendations,
                blocked_content=blocked_content,
                alternative_response=alternative_response
            )
            
            # Record interaction
            processing_time = time.time() - start_time
            await self._record_interaction(
                session_id or f"session_{int(time.time())}",
                agent_name, input_data, output_data, result, processing_time
            )
            
            # Update metrics
            self._update_metrics(result, processing_time)
            
            logger.info(
                f"Validated {agent_name} response: {validation_level.value}",
                {
                    "session_id": session_id,
                    "agent": agent_name,
                    "validation_level": validation_level.value,
                    "clinical_risk": clinical_risk.value,
                    "accuracy_score": accuracy_score,
                    "processing_time": processing_time
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating agent response: {str(e)}")
            return ValidationResult(
                validation_level=ValidationLevel.WARNING,
                clinical_risk=ClinicalRiskLevel.MODERATE,
                ethical_concerns=[],
                accuracy_score=0.5,
                consistency_score=0.5,
                appropriateness_score=0.5,
                recommendations=[f"Validation error: {str(e)}"]
            )
    
    async def _assess_clinical_risk(self, response_text: str, user_input: str) -> ClinicalRiskLevel:
        """Assess clinical risk level of the response."""
        risk_score = 0
        
        # Check for crisis indicators
        for risk_type, indicators in self.risk_indicators.items():
            for indicator in indicators:
                if re.search(indicator, response_text.lower()) or re.search(indicator, user_input.lower()):
                    if risk_type == "suicide_risk":
                        return ClinicalRiskLevel.SEVERE
                    elif risk_type == "self_harm":
                        risk_score += 3
                    elif risk_type in ["substance_abuse", "psychosis_indicators"]:
                        risk_score += 2
                    else:
                        risk_score += 1
        
        # Check clinical patterns
        for pattern_type, patterns in self.clinical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_text.lower()):
                    if pattern_type == "medication_advice":
                        risk_score += 3
                    elif pattern_type == "diagnostic_statements":
                        risk_score += 2
                    else:
                        risk_score += 1
        
        # Map score to risk level
        if risk_score >= 5:
            return ClinicalRiskLevel.SEVERE
        elif risk_score >= 3:
            return ClinicalRiskLevel.HIGH
        elif risk_score >= 2:
            return ClinicalRiskLevel.MODERATE
        elif risk_score >= 1:
            return ClinicalRiskLevel.LOW
        else:
            return ClinicalRiskLevel.MINIMAL
    
    async def _check_ethical_compliance(self, response_text: str, user_input: str) -> List[EthicalConcern]:
        """Check for ethical compliance issues."""
        concerns = []
        
        # Check therapeutic boundaries
        response_lower = response_text.lower()
        for boundary_type, indicators in self.therapeutic_boundaries.items():
            for indicator in indicators:
                if indicator.lower() in response_lower:
                    if boundary_type == "avoid_diagnosis_terms":
                        concerns.append(EthicalConcern.COMPETENCY_EXCEEDED)
                    elif boundary_type == "crisis_escalation_required":
                        concerns.append(EthicalConcern.HARM_POTENTIAL)
                    elif boundary_type == "inappropriate_advice":
                        concerns.append(EthicalConcern.HARM_POTENTIAL)
        
        # Additional regex-based catch for common boundary phrases not covered by simple substrings
        boundary_regexes = [
            r"\bbe\s+friends\b",
            r"\bfriends?\s+after\s+therapy\b",
            r"\bmeet\s+outside\b",
        ]
        for pattern in boundary_regexes:
            if re.search(pattern, response_lower):
                concerns.append(EthicalConcern.BOUNDARY_VIOLATION)
                break
        
        # Additional regex-based catch for medication stop/change advice (e.g., "stop taking your medication")
        med_advice_regexes = [
            r"\bstop\s+taking\b.*\bmedication\b",
            r"\bdon't\s+see\b.*\bdoctor\b",
        ]
        for pattern in med_advice_regexes:
            if re.search(pattern, response_lower):
                concerns.append(EthicalConcern.HARM_POTENTIAL)
                break
        
        # Check for bias indicators
        bias_patterns = [
            r"people like you", r"your type", r"typical for \w+",
            r"all \w+ are", r"women/men always", r"because you're"
        ]
        
        for pattern in bias_patterns:
            if re.search(pattern, response_text.lower()):
                concerns.append(EthicalConcern.BIAS_DETECTED)
                break
        
        # Check for boundary violations
        boundary_patterns = [
            r"i love you", r"we should meet", r"personal relationship",
            r"beyond our session", r"outside therapy", r"be friends", r"friends after therapy"
        ]
        
        for pattern in boundary_patterns:
            if re.search(pattern, response_text.lower()):
                concerns.append(EthicalConcern.BOUNDARY_VIOLATION)
                break
        
        return list(set(concerns))  # Remove duplicates
    
    async def _calculate_accuracy_score(self, response_text: str, user_input: str, agent_name: str) -> float:
        """Calculate accuracy score based on clinical knowledge."""
        try:
            # Search for relevant clinical guidelines
            relevant_guidelines = search_relevant_data(
                query=user_input,
                data_types=["clinical_guidelines"],
                limit=3
            )
            
            if not relevant_guidelines:
                return 0.7  # Default moderate accuracy
            
            # Simple accuracy calculation based on guideline alignment
            # In production, you might use more sophisticated NLP analysis
            accuracy_score = 0.8
            
            # Check if response violates any retrieved guidelines
            for guideline in relevant_guidelines:
                guideline_content = guideline.get("content", "").lower()
                if any(word in response_text.lower() for word in guideline_content.split()):
                    # Positive alignment with guidelines
                    accuracy_score += 0.1
                else:
                    # Potential misalignment
                    accuracy_score -= 0.05
            
            return max(0.0, min(1.0, accuracy_score))
            
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {str(e)}")
            return 0.5
    
    async def _check_consistency(self, response_text: str, session_id: str, agent_name: str) -> float:
        """Check consistency with previous responses in the session."""
        if not session_id:
            return 0.8  # Default consistency score
        
        try:
            # Get previous interactions for this session and agent
            previous_interactions = [
                interaction for interaction in self.interaction_history
                if interaction.session_id == session_id and interaction.agent_name == agent_name
            ]
            
            if not previous_interactions:
                return 0.8  # No previous interactions to compare
            
            # Use consistency checker
            return self.consistency_checker.check_consistency(
                response_text, 
                [interaction.output_data for interaction in previous_interactions[-5:]]  # Last 5 interactions
            )
            
        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
            return 0.5
    
    async def _assess_appropriateness(self, response_text: str, user_input: str) -> float:
        """Assess appropriateness of the response."""
        try:
            # Sentiment analysis
            user_sentiment = self.sentiment_analyzer.analyze(user_input)
            response_sentiment = self.sentiment_analyzer.analyze(response_text)
            
            # Check for appropriate emotional tone matching
            appropriateness_score = 0.8
            
            # Adjust based on sentiment appropriateness
            if user_sentiment["emotion"] == "sad" and response_sentiment["emotion"] == "happy":
                appropriateness_score -= 0.3
            elif user_sentiment["emotion"] == "angry" and response_sentiment["emotion"] == "dismissive":
                appropriateness_score -= 0.4
            elif user_sentiment["emotion"] == "anxious" and response_sentiment["emotion"] == "calm":
                appropriateness_score += 0.1
            
            # Check response length appropriateness
            if len(response_text) < 50:
                appropriateness_score -= 0.1  # Too brief
            elif len(response_text) > 1000:
                appropriateness_score -= 0.2  # Too verbose
            
            return max(0.0, min(1.0, appropriateness_score))
            
        except Exception as e:
            logger.error(f"Error assessing appropriateness: {str(e)}")
            return 0.5
    
    def _determine_validation_level(self, clinical_risk: ClinicalRiskLevel,
                                  ethical_concerns: List[EthicalConcern],
                                  accuracy_score: float, consistency_score: float,
                                  appropriateness_score: float) -> ValidationLevel:
        """Determine overall validation level."""
        # Block for severe clinical risk
        if clinical_risk == ClinicalRiskLevel.SEVERE:
            return ValidationLevel.BLOCKED
        
        # Block for serious ethical concerns
        serious_concerns = [
            EthicalConcern.HARM_POTENTIAL,
            EthicalConcern.BOUNDARY_VIOLATION,
            EthicalConcern.COMPETENCY_EXCEEDED
        ]
        
        if any(concern in ethical_concerns for concern in serious_concerns):
            return ValidationLevel.BLOCKED
        
        # Critical for high risk or low scores
        if clinical_risk == ClinicalRiskLevel.HIGH or accuracy_score < 0.4:
            return ValidationLevel.CRITICAL
        
        # Warning for moderate issues
        if (clinical_risk == ClinicalRiskLevel.MODERATE or 
            len(ethical_concerns) > 0 or 
            accuracy_score < 0.6 or 
            consistency_score < 0.6 or 
            appropriateness_score < 0.6):
            return ValidationLevel.WARNING
        
        return ValidationLevel.PASS
    
    async def _generate_recommendations(self, validation_level: ValidationLevel,
                                      clinical_risk: ClinicalRiskLevel,
                                      ethical_concerns: List[EthicalConcern],
                                      accuracy_score: float, consistency_score: float,
                                      appropriateness_score: float) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if clinical_risk in [ClinicalRiskLevel.HIGH, ClinicalRiskLevel.SEVERE]:
            recommendations.append("Immediate clinical supervision required")
            recommendations.append("Escalate to mental health professional")
        
        if EthicalConcern.HARM_POTENTIAL in ethical_concerns:
            recommendations.append("Review response for potential harm")
            recommendations.append("Consider alternative therapeutic approach")
        
        if EthicalConcern.BOUNDARY_VIOLATION in ethical_concerns:
            recommendations.append("Maintain therapeutic boundaries")
            recommendations.append("Avoid personal relationship suggestions")
        
        if accuracy_score < 0.6:
            recommendations.append("Verify clinical accuracy against guidelines")
            recommendations.append("Consult additional therapeutic resources")
        
        if consistency_score < 0.6:
            recommendations.append("Ensure consistency with previous responses")
            recommendations.append("Review session context for continuity")
        
        if appropriateness_score < 0.6:
            recommendations.append("Adjust response tone and length")
            recommendations.append("Better match emotional context")
        
        return recommendations
    
    async def _generate_alternative_response(self, user_input: str, clinical_risk: ClinicalRiskLevel,
                                           ethical_concerns: List[EthicalConcern]) -> str:
        """Generate alternative response for blocked content."""
        if clinical_risk == ClinicalRiskLevel.SEVERE:
            return (
                "I'm very concerned about what you've shared. Your safety is my priority. "
                "Please contact a mental health professional immediately:\n"
                "• National Suicide Prevention Lifeline: 988\n"
                "• Crisis Text Line: Text HOME to 741741\n"
                "• Or visit your nearest emergency room"
            )
        
        if EthicalConcern.HARM_POTENTIAL in ethical_concerns:
            return (
                "I understand you're going through a difficult time. "
                "I'd like to help you connect with professional support that can provide "
                "the specialized care you need. Would you like information about mental health resources?"
            )
        
        if EthicalConcern.COMPETENCY_EXCEEDED in ethical_concerns:
            return (
                "Thank you for sharing that with me. What you're describing involves clinical "
                "considerations that require professional evaluation. I'd recommend speaking "
                "with a qualified mental health professional who can provide proper assessment and guidance."
            )
        
        return (
            "I want to make sure I provide you with the most helpful response. "
            "Let me rephrase that in a way that better supports your well-being."
        )
    
    async def _record_interaction(self, session_id: str, agent_name: str, 
                                input_data: Dict[str, Any], output_data: Dict[str, Any],
                                validation_result: ValidationResult, processing_time: float):
        """Record interaction in audit trail."""
        interaction = AgentInteraction(
            session_id=session_id,
            agent_name=agent_name,
            input_data=input_data,
            output_data=output_data,
            validation_result=validation_result,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.interaction_history.append(interaction)
        
        # Trim history if it gets too large
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[-int(self.max_history_size * 0.8):]
        
        # Store critical interactions in vector database
        if validation_result.validation_level in [ValidationLevel.CRITICAL, ValidationLevel.BLOCKED]:
            try:
                add_user_data("supervisor_audit", asdict(interaction), f"audit_{int(time.time())}")
            except Exception as e:
                logger.error(f"Error storing audit data: {str(e)}")
    
    def _update_metrics(self, validation_result: ValidationResult, processing_time: float):
        """Update performance metrics."""
        self.metrics["total_validations"] += 1
        
        if validation_result.validation_level == ValidationLevel.BLOCKED:
            self.metrics["blocked_responses"] += 1
        elif validation_result.validation_level == ValidationLevel.WARNING:
            self.metrics["warning_responses"] += 1
        
        if validation_result.ethical_concerns:
            self.metrics["ethical_violations_detected"] += len(validation_result.ethical_concerns)
        
        # Update averages
        total = self.metrics["total_validations"]
        self.metrics["average_accuracy_score"] = (
            (self.metrics["average_accuracy_score"] * (total - 1) + validation_result.accuracy_score) / total
        )
        self.metrics["average_processing_time"] = (
            (self.metrics["average_processing_time"] * (total - 1) + processing_time) / total
        )
    
    def _extract_response_text(self, output_data: Dict[str, Any]) -> str:
        """Extract response text from output data."""
        if isinstance(output_data, str):
            return output_data
        elif isinstance(output_data, dict):
            # Try common response keys
            for key in ["response", "output", "text", "message", "content"]:
                if key in output_data:
                    return str(output_data[key])
            # If no specific key, convert entire dict to string
            return json.dumps(output_data, indent=2)
        else:
            return str(output_data)
    
    def _extract_user_input(self, input_data: Dict[str, Any]) -> str:
        """Extract user input from input data."""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            # Try common input keys
            for key in ["message", "input", "query", "text", "user_input"]:
                if key in input_data:
                    return str(input_data[key])
            # If no specific key, convert entire dict to string
            return json.dumps(input_data, indent=2)
        else:
            return str(input_data)
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary with quality metrics."""
        session_interactions = [
            interaction for interaction in self.interaction_history
            if interaction.session_id == session_id
        ]
        
        if not session_interactions:
            return {"error": "No interactions found for session"}
        
        # Calculate session metrics
        total_interactions = len(session_interactions)
        blocked_count = sum(1 for i in session_interactions 
                          if i.validation_result.validation_level == ValidationLevel.BLOCKED)
        warning_count = sum(1 for i in session_interactions 
                          if i.validation_result.validation_level == ValidationLevel.WARNING)
        
        avg_accuracy = np.mean([i.validation_result.accuracy_score for i in session_interactions])
        avg_consistency = np.mean([i.validation_result.consistency_score for i in session_interactions])
        avg_appropriateness = np.mean([i.validation_result.appropriateness_score for i in session_interactions])
        
        # Identify patterns
        most_common_risk = max(
            [i.validation_result.clinical_risk for i in session_interactions],
            key=lambda x: [i.validation_result.clinical_risk for i in session_interactions].count(x)
        ).value
        
        ethical_concerns = []
        for interaction in session_interactions:
            ethical_concerns.extend(interaction.validation_result.ethical_concerns)
        
        return {
            "session_id": session_id,
            "total_interactions": total_interactions,
            "quality_metrics": {
                "blocked_responses": blocked_count,
                "warning_responses": warning_count,
                "pass_rate": (total_interactions - blocked_count - warning_count) / total_interactions,
                "average_accuracy": round(avg_accuracy, 3),
                "average_consistency": round(avg_consistency, 3),
                "average_appropriateness": round(avg_appropriateness, 3)
            },
            "risk_assessment": {
                "most_common_risk_level": most_common_risk,
                "total_ethical_concerns": len(ethical_concerns),
                "unique_ethical_concerns": len(set(ethical_concerns))
            },
            "recommendations": self._generate_session_recommendations(session_interactions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_session_recommendations(self, interactions: List[AgentInteraction]) -> List[str]:
        """Generate recommendations for the entire session."""
        recommendations = []
        
        # Check for patterns
        blocked_count = sum(1 for i in interactions 
                          if i.validation_result.validation_level == ValidationLevel.BLOCKED)
        
        if blocked_count > len(interactions) * 0.2:  # More than 20% blocked
            recommendations.append("High rate of blocked responses indicates need for agent retraining")
        
        avg_accuracy = np.mean([i.validation_result.accuracy_score for i in interactions])
        if avg_accuracy < 0.7:
            recommendations.append("Below-average accuracy scores suggest need for knowledge base updates")
        
        # Check for recurring ethical concerns
        all_concerns = []
        for interaction in interactions:
            all_concerns.extend(interaction.validation_result.ethical_concerns)
        
        if len(all_concerns) > len(interactions) * 0.3:  # More than 30% have ethical concerns
            recommendations.append("Frequent ethical concerns require immediate attention and training")
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "validation_metrics": self.metrics.copy(),
            "interaction_statistics": {
                "total_interactions_recorded": len(self.interaction_history),
                "sessions_tracked": len(set(i.session_id for i in self.interaction_history)),
                "agents_monitored": len(set(i.agent_name for i in self.interaction_history))
            },
            "quality_indicators": {
                "block_rate": self.metrics["blocked_responses"] / max(1, self.metrics["total_validations"]),
                "warning_rate": self.metrics["warning_responses"] / max(1, self.metrics["total_validations"]),
                "ethical_violation_rate": self.metrics["ethical_violations_detected"] / max(1, self.metrics["total_validations"])
            },
            "timestamp": datetime.now().isoformat()
        }


class ConsistencyChecker:
    """Helper class for checking response consistency."""
    
    def check_consistency(self, current_response: str, previous_responses: List[Dict[str, Any]]) -> float:
        """Check consistency between current and previous responses."""
        if not previous_responses:
            return 0.8  # Default score for no history
        
        # Simple consistency check based on key terms and sentiment
        # In production, you might use more sophisticated semantic similarity
        
        current_terms = set(current_response.lower().split())
        consistency_scores = []
        
        for prev_response in previous_responses[-3:]:  # Check last 3 responses
            prev_text = self._extract_text(prev_response)
            prev_terms = set(prev_text.lower().split())
            
            # Calculate term overlap
            overlap = len(current_terms.intersection(prev_terms))
            total_terms = len(current_terms.union(prev_terms))
            
            if total_terms > 0:
                consistency_scores.append(overlap / total_terms)
        
        return np.mean(consistency_scores) if consistency_scores else 0.8
    
    def _extract_text(self, response_data: Dict[str, Any]) -> str:
        """Extract text from response data."""
        if isinstance(response_data, str):
            return response_data
        elif isinstance(response_data, dict):
            for key in ["response", "output", "text", "message"]:
                if key in response_data:
                    return str(response_data[key])
        return str(response_data)


class SimpleSentimentAnalyzer:
    """Simple sentiment analyzer for response appropriateness."""
    
    def __init__(self):
        self.positive_words = [
            "happy", "joy", "good", "great", "wonderful", "excellent",
            "positive", "optimistic", "hopeful", "grateful", "content"
        ]
        self.negative_words = [
            "sad", "depressed", "angry", "frustrated", "anxious", "worried",
            "upset", "disappointed", "hopeless", "terrible", "awful"
        ]
        self.calm_words = [
            "calm", "peaceful", "relaxed", "serene", "tranquil", "composed",
            "balanced", "centered", "stable", "grounded"
        ]
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        calm_count = sum(1 for word in self.calm_words if word in text_lower)
        
        total_words = len(text.split())
        
        if negative_count > positive_count and negative_count > calm_count:
            emotion = "negative"
            intensity = negative_count / max(1, total_words)
        elif positive_count > negative_count and positive_count > calm_count:
            emotion = "positive"
            intensity = positive_count / max(1, total_words)
        elif calm_count > 0:
            emotion = "calm"
            intensity = calm_count / max(1, total_words)
        else:
            emotion = "neutral"
            intensity = 0.0
        
        return {
            "emotion": emotion,
            "intensity": min(1.0, intensity * 10),  # Scale up intensity
            "confidence": 0.7  # Simple analyzer has moderate confidence
        }