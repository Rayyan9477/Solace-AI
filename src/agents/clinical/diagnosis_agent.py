"""
Enhanced DiagnosisAgent that uses DSPy-powered Agentic RAG for more accurate mental health assessment.
"""

from typing import Dict, Any, Optional, List
from ..base.base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import spacy
from transformers import pipeline
from datetime import datetime
import logging
import os
import sys
from pathlib import Path
import asyncio
import time
import uuid

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the AgenticRAG system
from src.utils.agentic_rag import AgenticRAG

# Import integration components
from src.integration.event_bus import EventBus, Event, EventType, EventPriority, get_event_bus
from src.integration.supervision_mesh import SupervisionMesh, QualityGateType
from src.integration.friction_engine import FrictionEngine

logger = logging.getLogger(__name__)

# Load models
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Failed to load spaCy model: {str(e)}. Using fallback methods.")
    nlp = None

try:
    symptom_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # Use CPU
    )
    diagnostic_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
except Exception as e:
    logger.warning(f"Failed to load transformers pipeline: {str(e)}. Using fallback methods.")
    symptom_classifier = None
    diagnostic_classifier = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool("symptom_extraction")
async def extract_symptoms(text: str) -> Dict[str, Any]:
    """
    Extracts mental health symptoms from text using NLP
    
    Args:
        text: The text to analyze for symptoms
        
    Returns:
        Dictionary containing extracted symptoms and categories
    """
    # Extract entities using spaCy
    doc = nlp(text)
    entities = [
        ent.text for ent in doc.ents 
        if ent.label_ in ["SYMPTOM", "CONDITION", "BEHAVIOR"]
    ]
    
    # Define symptom categories
    symptom_categories = [
        "mood symptoms",
        "anxiety symptoms",
        "cognitive symptoms",
        "behavioral symptoms",
        "physical symptoms",
        "social symptoms"
    ]
    
    # Classify symptoms into categories
    if text:
        classifications = symptom_classifier(
            text,
            symptom_categories,
            multi_label=True
        )
        
        detected_categories = [
            cat for cat, score in zip(classifications['labels'], classifications['scores'])
            if score > 0.5
        ]
    else:
        detected_categories = []
        
    return {
        'extracted_symptoms': entities,
        'symptom_categories': detected_categories,
        'raw_text': text
    }

@tool("diagnostic_criteria")
async def analyze_diagnostic_criteria(symptoms: List[str]) -> Dict[str, Any]:
    """
    Matches symptoms to diagnostic criteria
    
    Args:
        symptoms: List of extracted symptoms to analyze
        
    Returns:
        Dictionary containing potential diagnoses and confidence scores
    """
    symptom_text = " ".join(symptoms)
    
    # Define diagnostic categories
    diagnostic_categories = [
        "Major Depressive Disorder",
        "Generalized Anxiety Disorder",
        "Bipolar Disorder",
        "Post-Traumatic Stress Disorder",
        "Social Anxiety Disorder",
        "Panic Disorder"
    ]
    
    if symptom_text:
        # Classify symptoms against diagnostic criteria
        classifications = diagnostic_classifier(
            symptom_text,
            diagnostic_categories,
            multi_label=True
        )
        
        potential_diagnoses = [
            {
                'condition': label,
                'confidence': score,
                'severity': _estimate_severity(score)
            }
            for label, score in zip(classifications['labels'], classifications['scores'])
            if score > 0.3  # Lower threshold for potential matches
        ]
    else:
        potential_diagnoses = []
        
    return {
        'potential_diagnoses': potential_diagnoses,
        'symptom_count': len(symptoms)
    }

@tool("enhanced_diagnosis")
async def enhanced_diagnosis(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform an enhanced diagnosis using Agentic RAG with structured reasoning
    
    Args:
        text: The text containing symptoms or mental health descriptions
        context: Optional additional context to consider
        
    Returns:
        Comprehensive diagnostic assessment with reasoning steps
    """
    if not hasattr(enhanced_diagnosis, "rag_system"):
        # Initialize the RAG system if not already available
        try:
            from src.models.llm import get_llm
            llm = get_llm()
            
            # Set up the knowledge base directory
            project_root = Path(__file__).parent.parent.parent
            kb_dir = project_root / "src" / "data" / "personality"
            
            # Initialize the Agentic RAG system
            enhanced_diagnosis.rag_system = AgenticRAG(
                llm=llm,
                knowledge_base_dir=str(kb_dir)
            )
            logger.info("Initialized Agentic RAG system for enhanced diagnosis")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return {
                "error": f"Failed to initialize diagnostic system: {str(e)}",
                "fallback": "Using standard assessment"
            }
    
    try:
        # Use the Agentic RAG system for diagnosis
        result = await enhanced_diagnosis.rag_system.enhance_diagnosis(text, context)
        return result
    except Exception as e:
        logger.error(f"Error in enhanced diagnosis: {str(e)}")
        return {
            "error": str(e),
            "fallback": "Using standard assessment"
        }

@tool("phq9_assessment")
async def phq9_assessment(responses: List[int]) -> Dict[str, Any]:
    """
    Administers PHQ-9 questionnaire and computes depression severity.
    Args:
        responses: List of 9 integer responses (0-3) for PHQ-9 items
    Returns:
        Dictionary with total score, severity category, and interpretation
    """
    score = sum(responses)
    if score >= 20:
        severity = "severe"
    elif score >= 15:
        severity = "moderately severe"
    elif score >= 10:
        severity = "moderate"
    elif score >= 5:
        severity = "mild"
    else:
        severity = "minimal"
    interpretation = (
        f"PHQ-9 total score {score} indicates {severity} depressive symptoms."
        " Recommend referral to mental health professional for comprehensive evaluation."
    )
    return {"score": score, "severity": severity, "interpretation": interpretation}

@tool("gad7_assessment")
async def gad7_assessment(responses: List[int]) -> Dict[str, Any]:
    """
    Administers GAD-7 questionnaire and computes anxiety severity.
    Args:
        responses: List of 7 integer responses (0-3) for GAD-7 items
    Returns:
        Dictionary with total score, severity category, and interpretation
    """
    score = sum(responses)
    if score >= 15:
        severity = "severe"
    elif score >= 10:
        severity = "moderate"
    elif score >= 5:
        severity = "mild"
    else:
        severity = "minimal"
    interpretation = (
        f"GAD-7 total score {score} indicates {severity} anxiety symptoms."
        " Consider stress management strategies and professional consultation."
    )
    return {"score": score, "severity": severity, "interpretation": interpretation}

def _estimate_severity(confidence_score: float) -> str:
    """Estimate severity level based on confidence score"""
    if confidence_score > 0.8:
        return "severe"
    elif confidence_score > 0.6:
        return "moderate"
    else:
        return "mild"

class EnhancedDiagnosisAgent(BaseAgent):
    def __init__(
        self, 
        model: BaseLanguageModel,
        event_bus: Optional[EventBus] = None,
        supervision_mesh: Optional[SupervisionMesh] = None,
        friction_engine: Optional[FrictionEngine] = None
    ):
        super().__init__(
            model=model,
            name="enhanced_mental_health_diagnostician",
            role="Advanced system for mental health symptom analysis and diagnosis with AI reasoning",
            description="""An enhanced AI agent specialized in analyzing mental health symptoms and providing diagnostic insights.
            Uses evidence-based criteria, structured reasoning, and retrieval-augmented generation to maintain clinical accuracy 
            while emphasizing the importance of professional evaluation. Integrated with event-driven communication, 
            supervision validation, and therapeutic friction coordination.""",
            tools=[extract_symptoms, analyze_diagnostic_criteria, enhanced_diagnosis, phq9_assessment, gad7_assessment],
            memory=Memory(memory="diagnosis_memory", storage="local_storage"),
            knowledge=AgentKnowledge()
        )
        
        # Integration components
        self.event_bus = event_bus or get_event_bus()
        self.supervision_mesh = supervision_mesh
        self.friction_engine = friction_engine
        
        # Agent state
        self.agent_id = f"enhanced_diagnosis_{uuid.uuid4().hex[:8]}"
        self.is_active = False
        self.current_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Metrics and monitoring
        self.metrics = {
            'diagnoses_performed': 0,
            'supervision_validations': 0,
            'friction_coordinations': 0,
            'average_processing_time': 0.0,
            'processing_times': [],
            'validation_failures': 0,
            'integration_errors': 0
        }
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()
        
        # Initialize the Agentic RAG system
        try:
            project_root = Path(__file__).parent.parent.parent
            kb_dir = project_root / "src" / "data" / "personality"
            
            self.rag_system = AgenticRAG(
                llm=model,
                knowledge_base_dir=str(kb_dir)
            )
            logger.info("Initialized Agentic RAG system for EnhancedDiagnosisAgent")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG: {str(e)}")
            self.rag_system = None
    
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for integration."""
        try:
            # Subscribe to diagnosis requests
            self.event_bus.subscribe(
                EventType.AGENT_REQUEST,
                self._handle_diagnosis_request,
                agent_id=self.agent_id,
                filters={'target_agent': 'enhanced_diagnosis', 'request_type': 'diagnosis'}
            )
            
            # Subscribe to validation results
            self.event_bus.subscribe(
                EventType.VALIDATION_RESULT,
                self._handle_validation_result,
                agent_id=self.agent_id
            )
            
            # Subscribe to friction updates
            self.event_bus.subscribe(
                EventType.FRICTION_APPLICATION,
                self._handle_friction_update,
                agent_id=self.agent_id
            )
            
            logger.info(f"Enhanced diagnosis agent {self.agent_id} subscribed to events")
        except Exception as e:
            logger.error(f"Failed to set up event subscriptions: {e}")
            self.metrics['integration_errors'] += 1
    
    async def start(self) -> None:
        """Start the enhanced diagnosis agent."""
        if self.is_active:
            return
        
        self.is_active = True
        
        # Publish startup event
        await self.event_bus.publish(Event(
            event_type=EventType.AGENT_STATUS,
            source_agent=self.agent_id,
            data={'status': 'active', 'agent_type': 'enhanced_diagnosis'}
        ))
        
        logger.info(f"Enhanced diagnosis agent {self.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the enhanced diagnosis agent."""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # Publish shutdown event
        await self.event_bus.publish(Event(
            event_type=EventType.AGENT_STATUS,
            source_agent=self.agent_id,
            data={'status': 'inactive', 'shutdown_reason': 'requested'}
        ))
        
        logger.info(f"Enhanced diagnosis agent {self.agent_id} stopped")
        
        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health diagnosis with advanced AI reasoning capabilities.
Your role is to analyze symptoms and provide professional insights while:
1. Using structured reasoning to analyze symptom patterns
2. Maintaining clinical accuracy with evidence-based criteria
3. Considering differential diagnoses and their likelihood
4. Evaluating symptom severity and impact
5. Providing appropriate recommendations based on clinical guidelines

Guidelines:
- Focus on observable symptoms and their patterns
- Consider multiple diagnostic possibilities
- Evaluate contextual factors that may influence symptoms
- Maintain professional boundaries
- Emphasize the importance of professional evaluation
- Provide evidence-based recommendations
- Clearly explain your reasoning process"""),
            HumanMessage(content="""Extracted Symptoms: {symptoms}
Symptom Categories: {symptom_categories}
Enhanced Analysis: {enhanced_analysis}
Diagnostic Matches: {diagnostic_matches}
Previous History: {history}

Provide a comprehensive diagnostic assessment:
Primary Concerns: [list main symptoms]
Reasoning Process: [explain your step-by-step diagnostic reasoning]
Potential Conditions: [possible diagnoses with confidence levels]
Severity Level: [mild/moderate/severe]
Recommendations: [professional and self-help suggestions]
Additional Considerations: [important factors to consider]""")
        ])

    async def diagnose_with_integration(
        self,
        symptoms: List[str],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        require_supervision: bool = True,
        enable_friction: bool = True
    ) -> Dict[str, Any]:
        """Generate an enhanced diagnostic assessment with full integration support."""
        start_time = time.time()
        diagnosis_id = f"diag_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Starting integrated diagnosis {diagnosis_id} for symptoms: {symptoms}")
            
            # Track session
            if session_id:
                self.current_sessions[session_id] = {
                    'diagnosis_id': diagnosis_id,
                    'user_id': user_id,
                    'start_time': datetime.now(),
                    'symptoms': symptoms,
                    'context': context or {}
                }
            
            # Perform core diagnosis
            diagnosis_result = await self.diagnose(symptoms, context)
            
            if not diagnosis_result.get('success', False):
                return diagnosis_result
            
            # Enhance with integration features
            if require_supervision and self.supervision_mesh:
                diagnosis_result = await self._validate_with_supervision(diagnosis_result, context or {})
            
            if enable_friction and self.friction_engine and user_id and session_id:
                diagnosis_result = await self._coordinate_therapeutic_friction(
                    diagnosis_result, user_id, session_id, context or {}
                )
            
            # Publish diagnosis event
            await self.event_bus.publish(Event(
                event_type=EventType.CLINICAL_ASSESSMENT,
                source_agent=self.agent_id,
                user_id=user_id,
                session_id=session_id,
                priority=EventPriority.HIGH if diagnosis_result.get('severity') == 'severe' else EventPriority.NORMAL,
                data={
                    'diagnosis_id': diagnosis_id,
                    'symptoms': symptoms,
                    'diagnosis_result': diagnosis_result,
                    'processing_time': time.time() - start_time
                }
            ))
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['diagnoses_performed'] += 1
            self.metrics['processing_times'].append(processing_time)
            self.metrics['average_processing_time'] = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            
            diagnosis_result.update({
                'diagnosis_id': diagnosis_id,
                'integration_features': {
                    'supervision_validated': require_supervision and self.supervision_mesh is not None,
                    'friction_coordinated': enable_friction and self.friction_engine is not None,
                    'event_published': True
                },
                'processing_time': processing_time
            })
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"Error in integrated diagnosis {diagnosis_id}: {str(e)}")
            self.metrics['integration_errors'] += 1
            
            # Publish error event
            await self.event_bus.publish(Event(
                event_type=EventType.AGENT_ERROR,
                source_agent=self.agent_id,
                user_id=user_id,
                session_id=session_id,
                data={
                    'diagnosis_id': diagnosis_id,
                    'error': str(e),
                    'symptoms': symptoms
                }
            ))
            
            return {
                'success': False,
                'error': f'Integrated diagnosis failed: {str(e)}',
                'diagnosis_id': diagnosis_id,
                'fallback_available': True
            }
    
    async def diagnose(self, symptoms: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an enhanced diagnostic assessment for the given symptoms"""
        try:
            logger.info(f"Diagnosing symptoms: {symptoms}")
            
            if not symptoms:
                logger.warning("Empty symptoms list provided")
                return {
                    "error": "Insufficient information for diagnosis",
                    "success": False
                }
            
            # Convert symptoms to string for processing
            symptoms_text = " ".join(symptoms)
            
            # Use the Agentic RAG system if available
            if self.rag_system:
                enhanced_result = await self.rag_system.enhance_diagnosis(symptoms_text, context)
                
                if enhanced_result.get("success", False):
                    logger.info("Successfully generated enhanced diagnosis")
                    return {
                        "success": True,
                        "symptoms": enhanced_result.get("symptoms", []),
                        "reasoning": enhanced_result.get("reasoning", ""),
                        "potential_diagnoses": enhanced_result.get("potential_diagnoses", []),
                        "severity": enhanced_result.get("severity", "mild"),
                        "recommendations": enhanced_result.get("recommendations", [])
                    }
            
            # If RAG fails or isn't available, use the legacy approach
            logger.info("Using legacy diagnosis approach")
            legacy_result = await self._legacy_diagnose(symptoms)
            
            return {
                "success": True,
                "method": "legacy",
                **legacy_result
            }
            
        except Exception as e:
            logger.error(f"Error in diagnosis process: {str(e)}")
            return {
                "success": False,
                "error": f"Unable to complete diagnostic assessment: {str(e)}"
            }

    async def _legacy_diagnose(self, symptoms: List[str]) -> Dict[str, Any]:
        """Legacy diagnosis method as a fallback"""
        try:
            # Check if required NLP models are available
            if nlp is None or diagnostic_classifier is None:
                logger.error("Required NLP models not available for diagnosis")
                return self._fallback_diagnosis(" ".join(symptoms))
            
            # Extract symptoms using spaCy
            doc = nlp(" ".join(symptoms))
            entities = [
                ent.text for ent in doc.ents 
                if ent.label_ in ["SYMPTOM", "CONDITION", "BEHAVIOR"]
            ]
            
            # If no entities were found, use the original symptoms
            if not entities:
                logger.info("No entities extracted, using original symptoms")
                entities = symptoms
            
            # Classify symptoms
            symptom_text = " ".join(entities)
            diagnostic_categories = [
                "Major Depressive Disorder",
                "Generalized Anxiety Disorder",
                "Bipolar Disorder",
                "Post-Traumatic Stress Disorder",
                "Social Anxiety Disorder",
                "Panic Disorder"
            ]
            
            # Get diagnostic classifications
            classifications = diagnostic_classifier(
                symptom_text,
                diagnostic_categories,
                multi_label=True
            )
            
            # Create potential diagnoses
            potential_diagnoses = [
                {
                    'condition': label,
                    'confidence': score,
                    'severity': _estimate_severity(score)
                }
                for label, score in zip(classifications['labels'], classifications['scores'])
                if score > 0.3
            ]
            
            # Sort by confidence
            potential_diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Determine overall severity
            if potential_diagnoses:
                severity = max(potential_diagnoses, key=lambda x: x['confidence'])['severity']
            else:
                severity = "mild"
            
            # Generate recommendations
            if severity == "severe":
                recommendations = ["Consult with a mental health professional promptly", 
                                  "Consider professional evaluation for treatment options",
                                  "Monitor symptoms closely"]
            elif severity == "moderate":
                recommendations = ["Consider consulting with a mental health professional", 
                                  "Implement self-care strategies",
                                  "Track symptoms to monitor progress"]
            else:
                recommendations = ["Practice self-care and stress management", 
                                  "Maintain supportive relationships",
                                  "Consider professional consultation if symptoms worsen"]
            
            return {
                "symptoms": entities,
                "potential_diagnoses": potential_diagnoses,
                "severity": severity,
                "recommendations": recommendations
            }
        
        except Exception as e:
            logger.error(f"Error in legacy diagnosis: {str(e)}")
            return self._fallback_diagnosis(" ".join(symptoms))

    def _fallback_diagnosis(self, symptom_text: str) -> Dict[str, Any]:
        """Simple fallback for diagnosis when all other methods fail"""
        keywords = {
            "Major Depressive Disorder": ["sad", "depressed", "hopeless", "tired", "sleep", "interest"],
            "Generalized Anxiety Disorder": ["worry", "anxious", "nervous", "stress", "fear"],
            "Bipolar Disorder": ["mood", "energy", "high", "low", "irritable"],
            "Post-Traumatic Stress Disorder": ["trauma", "flashback", "nightmare", "avoid"],
            "Social Anxiety Disorder": ["social", "embarrass", "fear", "shy", "awkward"],
            "Panic Disorder": ["panic", "attack", "heart", "breath", "dizzy"]
        }
        
        symptom_words = set(symptom_text.lower().split())
        potential_diagnoses = []
        
        for condition, keywords_list in keywords.items():
            count = sum(1 for keyword in keywords_list if keyword in symptom_words)
            if count > 0:
                confidence = min(0.3 + (count * 0.1), 0.8)
                potential_diagnoses.append({
                    'condition': condition,
                    'confidence': confidence,
                    'severity': _estimate_severity(confidence)
                })
        
        # Sort by confidence
        potential_diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Determine overall severity
        if potential_diagnoses:
            severity = max(potential_diagnoses, key=lambda x: x['confidence'])['severity']
        else:
            severity = "mild"
            potential_diagnoses = [{
                'condition': 'Unspecified condition',
                'confidence': 0.3,
                'severity': 'mild'
            }]
        
        # Generate basic recommendations
        recommendations = [
            "Consider consulting with a mental health professional",
            "Track your symptoms and their frequency",
            "Practice self-care strategies for overall well-being"
        ]
        
        return {
            "symptoms": symptom_text.split(),
            "potential_diagnoses": potential_diagnoses,
            "severity": severity,
            "recommendations": recommendations
        }
    
    async def _validate_with_supervision(
        self,
        diagnosis_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate diagnosis through supervision mesh."""
        try:
            validation_result = await self.supervision_mesh.validate(
                content={
                    'diagnosis': diagnosis_result,
                    'agent_type': 'enhanced_diagnosis'
                },
                context={
                    'user_profile': context.get('user_profile', {}),
                    'session_context': context.get('session_context', {}),
                    'agent_response': diagnosis_result
                },
                required_gates={QualityGateType.CLINICAL_SAFETY, QualityGateType.RISK_ASSESSMENT},
                requires_consensus=diagnosis_result.get('severity') == 'severe',
                requesting_agent=self.agent_id
            )
            
            self.metrics['supervision_validations'] += 1
            
            # Process validation result
            if hasattr(validation_result, 'final_result'):
                # Consensus result
                if validation_result.final_result.value in ['CRITICAL', 'BLOCKED']:
                    diagnosis_result['supervision_blocked'] = True
                    diagnosis_result['supervision_message'] = "Diagnosis blocked by supervision validation"
                    diagnosis_result['success'] = False
                else:
                    diagnosis_result['supervision_validated'] = True
                    diagnosis_result['supervision_confidence'] = validation_result.confidence
            else:
                # Single validation result
                if validation_result.result.value in ['CRITICAL', 'BLOCKED']:
                    diagnosis_result['supervision_blocked'] = True
                    diagnosis_result['supervision_message'] = validation_result.message
                    diagnosis_result['success'] = False
                else:
                    diagnosis_result['supervision_validated'] = True
                    diagnosis_result['supervision_confidence'] = validation_result.confidence
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"Supervision validation error: {e}")
            self.metrics['validation_failures'] += 1
            diagnosis_result['supervision_error'] = str(e)
            return diagnosis_result
    
    async def _coordinate_therapeutic_friction(
        self,
        diagnosis_result: Dict[str, Any],
        user_id: str,
        session_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate therapeutic friction based on diagnosis."""
        try:
            # Assess if friction is appropriate
            severity = diagnosis_result.get('severity', 'mild')
            potential_diagnoses = diagnosis_result.get('potential_diagnoses', [])
            
            # Don't apply friction for severe conditions or crisis situations
            if severity == 'severe' or any('crisis' in str(diag).lower() for diag in potential_diagnoses):
                return diagnosis_result
            
            # Assess user readiness for friction
            readiness_profile = await self.friction_engine.assess_user_readiness(
                user_id,
                session_id,
                {
                    'message': ' '.join(diagnosis_result.get('symptoms', [])),
                    'user_profile': context.get('user_profile', {}),
                    'session_context': context.get('session_context', {})
                }
            )
            
            # Select appropriate friction strategy
            friction_context = {
                'diagnosis_result': diagnosis_result,
                'severity': severity,
                'breakthrough_indicators': self._identify_breakthrough_indicators(diagnosis_result)
            }
            
            strategy = await self.friction_engine.select_friction_strategy(
                user_id,
                session_id,
                friction_context,
                available_agents={self.agent_id}
            )
            
            if strategy:
                # Coordinate friction application
                coordination_id = await self.friction_engine.coordinate_cross_agent_friction(
                    user_id,
                    session_id,
                    strategy,
                    {self.agent_id},
                    friction_context
                )
                
                if coordination_id:
                    diagnosis_result['friction_coordination'] = {
                        'coordination_id': coordination_id,
                        'strategy_id': strategy.strategy_id,
                        'intensity': strategy.intensity,
                        'readiness_level': readiness_profile.overall_readiness.value
                    }
                    
                    self.metrics['friction_coordinations'] += 1
            
            return diagnosis_result
            
        except Exception as e:
            logger.error(f"Friction coordination error: {e}")
            diagnosis_result['friction_error'] = str(e)
            return diagnosis_result
    
    def _identify_breakthrough_indicators(self, diagnosis_result: Dict[str, Any]) -> List[str]:
        """Identify potential breakthrough indicators from diagnosis."""
        indicators = []
        
        symptoms = diagnosis_result.get('symptoms', [])
        severity = diagnosis_result.get('severity', 'mild')
        
        # Look for patterns that suggest breakthrough opportunities
        if severity in ['moderate', 'severe']:
            indicators.append('emotional_vulnerability')
        
        symptom_text = ' '.join(str(s) for s in symptoms).lower()
        
        if any(phrase in symptom_text for phrase in ['pattern', 'always', 'keep doing']):
            indicators.append('pattern_recognition')
        
        if any(phrase in symptom_text for phrase in ['confused', 'conflicted', 'torn']):
            indicators.append('cognitive_dissonance')
        
        if any(phrase in symptom_text for phrase in ['ready', 'want to change', 'tired of']):
            indicators.append('behavioral_readiness')
        
        return indicators
    
    async def _handle_diagnosis_request(self, event: Event) -> None:
        """Handle incoming diagnosis requests."""
        try:
            request_data = event.data
            symptoms = request_data.get('symptoms', [])
            context = request_data.get('context', {})
            
            # Process diagnosis with full integration
            result = await self.diagnose_with_integration(
                symptoms=symptoms,
                context=context,
                user_id=event.user_id,
                session_id=event.session_id,
                require_supervision=request_data.get('require_supervision', True),
                enable_friction=request_data.get('enable_friction', True)
            )
            
            # Send response
            if event.reply_to:
                await self.event_bus.publish(Event(
                    event_type=EventType.AGENT_RESPONSE,
                    source_agent=self.agent_id,
                    target_agent=event.reply_to,
                    correlation_id=event.correlation_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    data={
                        'request_id': event.event_id,
                        'diagnosis_result': result
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error handling diagnosis request: {e}")
            self.metrics['integration_errors'] += 1
    
    async def _handle_validation_result(self, event: Event) -> None:
        """Handle validation results from supervision mesh."""
        try:
            result_data = event.data
            request_id = result_data.get('request_id')
            
            # Update session with validation result
            for session_id, session_data in self.current_sessions.items():
                if session_data.get('diagnosis_id') == request_id:
                    session_data['validation_result'] = result_data
                    break
        
        except Exception as e:
            logger.error(f"Error handling validation result: {e}")
    
    async def _handle_friction_update(self, event: Event) -> None:
        """Handle friction coordination updates."""
        try:
            update_data = event.data
            coordination_id = update_data.get('coordination_id')
            
            if 'intensity_update' in update_data:
                # Adjust diagnostic approach based on friction intensity
                new_intensity = update_data['intensity_update']
                logger.info(f"Adjusted friction intensity to {new_intensity}")
            
            if update_data.get('action') == 'end_coordination':
                # Clean up coordination resources
                logger.info(f"Friction coordination {coordination_id} ended")
        
        except Exception as e:
            logger.error(f"Error handling friction update: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics and status."""
        return {
            'agent_id': self.agent_id,
            'is_active': self.is_active,
            'active_sessions': len(self.current_sessions),
            'performance_metrics': self.metrics.copy(),
            'integration_status': {
                'event_bus_connected': self.event_bus is not None,
                'supervision_mesh_available': self.supervision_mesh is not None,
                'friction_engine_available': self.friction_engine is not None
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        error_rate = self.metrics['integration_errors'] / max(1, self.metrics['diagnoses_performed'])
        avg_processing_time = self.metrics['average_processing_time']
        
        status = 'healthy'
        if error_rate > 0.1:  # More than 10% error rate
            status = 'degraded'
        if error_rate > 0.25 or avg_processing_time > 30:  # More than 25% error rate or >30s processing
            status = 'unhealthy'
        
        return {
            'status': status,
            'error_rate': error_rate,
            'average_processing_time': avg_processing_time,
            'active_sessions': len(self.current_sessions),
            'last_activity': max((
                session_data['start_time'] for session_data in self.current_sessions.values()
            ), default=None)
        }

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Get conversation text
            message = input_data.get('text', '')
            
            # 1. Extract symptoms
            symptom_result = await extract_symptoms(message)
            extracted_symptoms = symptom_result.get('extracted_symptoms', [])
            symptom_categories = symptom_result.get('symptom_categories', [])
            
            # 2. Get enhanced analysis if available
            enhanced_result = await enhanced_diagnosis(message)
            
            # 3. Analyze diagnostic criteria
            diagnostic_result = await analyze_diagnostic_criteria(extracted_symptoms)
            
            # 4. Get memory and history
            history = context.get('memory', {}).get('last_diagnosis', {})
            
            # 5. Generate the diagnostic assessment
            llm_response = await self.model.agenerate_messages([
                self.diagnosis_prompt.format_messages(
                    symptoms=extracted_symptoms,
                    symptom_categories=symptom_categories,
                    enhanced_analysis=enhanced_result,
                    diagnostic_matches=diagnostic_result,
                    history=self._format_history(history)
                )[0]
            ])
            
            # Parse the response
            response_text = llm_response.generations[0][0].text
            parsed_response = self._parse_assessment(response_text)
            
            # Add metadata
            parsed_response['timestamp'] = datetime.now().isoformat()
            
            # Update memory
            try:
                await self.memory.add("last_diagnosis", parsed_response)
            except Exception as memory_error:
                logger.warning(f"Failed to update memory: {str(memory_error)}")
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "error": str(e),
                "primary_concerns": ["Error generating diagnostic assessment"],
                "recommendations": [
                    "Please try again with more specific information",
                    "Consider consulting a mental health professional"
                ]
            }

    def _parse_assessment(self, text: str) -> Dict[str, Any]:
        """Parse the diagnostic assessment into structured format"""
        result = {
            "primary_concerns": [],
            "reasoning_process": "",
            "potential_conditions": [],
            "severity_level": "mild",
            "recommendations": [],
            "additional_considerations": []
        }
        
        try:
            current_section = None
            
            for line in text.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith("Primary Concerns:"):
                    current_section = "primary_concerns"
                    line = line[len("Primary Concerns:"):].strip()
                elif line.startswith("Reasoning Process:"):
                    current_section = "reasoning_process"
                    line = line[len("Reasoning Process:"):].strip()
                elif line.startswith("Potential Conditions:"):
                    current_section = "potential_conditions"
                    line = line[len("Potential Conditions:"):].strip()
                elif line.startswith("Severity Level:"):
                    current_section = "severity_level"
                    line = line[len("Severity Level:"):].strip()
                    result["severity_level"] = line.lower()
                    continue
                elif line.startswith("Recommendations:"):
                    current_section = "recommendations"
                    line = line[len("Recommendations:"):].strip()
                elif line.startswith("Additional Considerations:"):
                    current_section = "additional_considerations"
                    line = line[len("Additional Considerations:"):].strip()
                
                # Add content to the current section
                if current_section == "primary_concerns":
                    if line:
                        if line.startswith("- "):
                            result["primary_concerns"].append(line[2:])
                        elif "," in line:
                            result["primary_concerns"].extend([item.strip() for item in line.split(",")])
                        else:
                            result["primary_concerns"].append(line)
                elif current_section == "reasoning_process":
                    result["reasoning_process"] += line + " "
                elif current_section == "potential_conditions":
                    if line:
                        if line.startswith("- "):
                            result["potential_conditions"].append(line[2:])
                        elif "," in line:
                            result["potential_conditions"].extend([item.strip() for item in line.split(",")])
                        else:
                            result["potential_conditions"].append(line)
                elif current_section == "recommendations":
                    if line:
                        if line.startswith("- "):
                            result["recommendations"].append(line[2:])
                        elif "," in line:
                            result["recommendations"].extend([item.strip() for item in line.split(",")])
                        else:
                            result["recommendations"].append(line)
                elif current_section == "additional_considerations":
                    if line:
                        if line.startswith("- "):
                            result["additional_considerations"].append(line[2:])
                        elif "," in line:
                            result["additional_considerations"].extend([item.strip() for item in line.split(",")])
                        else:
                            result["additional_considerations"].append(line)
            
            return result
            
        except Exception as parse_error:
            logger.error(f"Error parsing assessment: {str(parse_error)}")
            return {
                "primary_concerns": ["Unable to parse assessment"],
                "reasoning_process": "Assessment parsing error",
                "potential_conditions": [],
                "severity_level": "unknown",
                "recommendations": ["Consult with a mental health professional"],
                "additional_considerations": ["Assessment processing error"],
                "raw_text": text
            }

    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format diagnostic history"""
        if not history:
            return "No previous diagnostic history available"
            
        return f"""Previous Assessment:
- Primary Concerns: {', '.join(history.get('primary_concerns', []))}
- Potential Conditions: {', '.join(history.get('potential_conditions', []))}
- Severity: {history.get('severity_level', 'unknown')}"""