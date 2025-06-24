from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
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

# Add vector database integration
from src.utils.vector_db_integration import add_user_data, get_user_data

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

def _estimate_severity(confidence_score: float) -> str:
    """Estimate severity level based on confidence score"""
    if confidence_score > 0.8:
        return "severe"
    elif confidence_score > 0.6:
        return "moderate"
    else:
        return "mild"

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

class DiagnosisAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="mental_health_diagnostician",
            role="Expert system for mental health symptom analysis and diagnosis",
            description="""An AI agent specialized in analyzing mental health symptoms and providing diagnostic insights.
            Uses evidence-based criteria and maintains clinical accuracy while emphasizing the importance of professional evaluation.""",
            tools=[extract_symptoms, analyze_diagnostic_criteria, phq9_assessment, gad7_assessment],
            memory=Memory(memory="diagnosis_memory", storage="local_storage"),  # Initialize with string values
            knowledge=AgentKnowledge()
        )
        
        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health diagnosis.
Your role is to analyze symptoms and provide professional insights while:
1. Maintaining clinical accuracy
2. Using evidence-based diagnostic criteria
3. Considering symptom patterns and severity
4. Accounting for differential diagnoses
5. Providing appropriate recommendations

Guidelines:
- Focus on observable symptoms
- Consider multiple diagnostic possibilities
- Maintain professional boundaries
- Emphasize the importance of professional evaluation
- Provide evidence-based recommendations"""),
            HumanMessage(content="""Extracted Symptoms: {symptoms}
Symptom Categories: {symptom_categories}
Diagnostic Matches: {diagnostic_matches}
Previous History: {history}

Provide a structured diagnostic assessment:
Primary Concerns: [list main symptoms]
Potential Conditions: [possible diagnoses with confidence levels]
Severity Level: [mild/moderate/severe]
Recommendations: [professional and self-help suggestions]
Additional Considerations: [important factors to consider]""")
        ])

    def diagnose(self, symptoms: List[str]) -> str:
        """Generate a diagnostic assessment for the given symptoms"""
        try:
            logger.info(f"Diagnosing symptoms: {symptoms}")
            
            if not symptoms:
                logger.warning("Empty symptoms list provided")
                return "Insufficient information for diagnosis"
            
            # Check if required NLP models are available
            if nlp is None or diagnostic_classifier is None:
                logger.error("Required NLP models not available for diagnosis")
                return "Diagnostic service temporarily unavailable"
            
            # Extract symptoms using spaCy
            try:
                doc = nlp(" ".join(symptoms))
                entities = [
                    ent.text for ent in doc.ents 
                    if ent.label_ in ["SYMPTOM", "CONDITION", "BEHAVIOR"]
                ]
                
                # If no entities were found, use the original symptoms
                if not entities:
                    logger.info("No entities extracted, using original symptoms")
                    entities = symptoms
            except Exception as spacy_error:
                logger.error(f"Error in spaCy processing: {str(spacy_error)}")
                entities = symptoms  # Fallback to using raw symptoms
            
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
            
            try:
                if symptom_text:
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
                        if score > 0.3
                    ]
                else:
                    logger.warning("Empty symptom text after processing")
                    potential_diagnoses = []
            except Exception as classifier_error:
                logger.error(f"Error in diagnostic classification: {str(classifier_error)}")
                # Fallback - simple keyword matching
                potential_diagnoses = self._fallback_diagnosis(symptom_text, diagnostic_categories)
            
            # Format the response
            if potential_diagnoses:
                # Sort by confidence and get the highest confidence diagnosis
                best_diagnosis = max(potential_diagnoses, key=lambda x: x['confidence'])
                logger.info(f"Diagnosis result: {best_diagnosis['condition']} ({best_diagnosis['severity']} severity)")
                return f"{best_diagnosis['condition']} ({best_diagnosis['severity']} severity)"
            else:
                logger.info("No specific diagnosis available from symptoms")
                return "No specific diagnosis available"
                
        except Exception as e:
            logger.error(f"Error in diagnosis process: {str(e)}")
            return "Unable to complete diagnostic assessment"

    def _fallback_diagnosis(self, symptom_text: str, categories: List[str]) -> List[Dict[str, Any]]:
        """Simple fallback for diagnosis when the classifier fails"""
        results = []
        symptom_words = set(symptom_text.lower().split())
        
        # Simple keyword matching
        keywords = {
            "Major Depressive Disorder": ["sad", "depressed", "hopeless", "tired", "sleep", "interest"],
            "Generalized Anxiety Disorder": ["worry", "anxious", "nervous", "stress", "fear"],
            "Bipolar Disorder": ["mood", "energy", "high", "low", "irritable"],
            "Post-Traumatic Stress Disorder": ["trauma", "flashback", "nightmare", "avoid"],
            "Social Anxiety Disorder": ["social", "embarrass", "fear", "shy", "awkward"],
            "Panic Disorder": ["panic", "attack", "heart", "breath", "dizzy"]
        }
        
        for category in categories:
            count = 0
            relevant_keywords = keywords.get(category, [])
            for keyword in relevant_keywords:
                if keyword in symptom_words:
                    count += 1
                    
            if count > 0:
                confidence = min(0.3 + (count * 0.1), 0.8)  # Limit to 0.8 max
                results.append({
                    'condition': category,
                    'confidence': confidence,
                    'severity': _estimate_severity(confidence)
                })
                
        return results

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Extract symptoms
            symptoms_result = await extract_symptoms(input_data.get('text', ''))
            
            # Get message and history
            message = input_data.get('text', '')
            history = context.get('memory', {}).get('last_diagnosis', {})
            
            # Generate LLM analysis
            try:
                # Try to use agenerate_messages if available
                llm_response = await self.model.agenerate_messages([
                    self.diagnosis_prompt.format_messages(
                        message=message,
                        history=self._format_history(history),
                        symptoms=symptoms_result
                    )[0]
                ])
                
                # Parse response
                parsed = self._parse_result(llm_response.generations[0][0].text)
            except (AttributeError, TypeError):
                # Fallback for LLMs that don't support agenerate_messages
                logger.warning("LLM does not support agenerate_messages, using fallback method")
                messages = self.diagnosis_prompt.format_messages(
                    message=message,
                    history=self._format_history(history),
                    symptoms=symptoms_result
                )
                
                # Use a synchronous approach as fallback
                response = self.model.generate([messages[0]])
                parsed = self._parse_result(response.generations[0][0].text)
            
            # Add metadata
            parsed['confidence'] = self._calculate_confidence(parsed, symptoms_result)
            parsed['timestamp'] = input_data.get('timestamp')
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error generating diagnosis response: {str(e)}")
            return self._fallback_analysis(input_data.get('text', ''))

    def _parse_result(self, text: str) -> Dict[str, Any]:
        """Parse the structured diagnostic assessment"""
        result = {
            'primary_concerns': [],
            'potential_conditions': [],
            'severity_level': 'mild',
            'recommendations': [],
            'additional_considerations': []
        }
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [x.strip() for x in line.split(':', 1)]
                
                if 'Primary Concerns' in key:
                    result['primary_concerns'] = [c.strip() for c in value.split(',')]
                elif 'Potential Conditions' in key:
                    result['potential_conditions'] = [c.strip() for c in value.split(',')]
                elif 'Severity Level' in key:
                    result['severity_level'] = value.lower()
                elif 'Recommendations' in key:
                    result['recommendations'] = [r.strip() for r in value.split(',')]
                elif 'Additional Considerations' in key:
                    result['additional_considerations'] = [a.strip() for a in value.split(',')]
                    
        except Exception:
            pass
            
        return result

    def _calculate_confidence(
        self,
        analysis: Dict[str, Any],
        diagnostic_matches: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in diagnostic assessment"""
        confidence = 1.0
        
        # Lower confidence if analysis is incomplete
        if not analysis['primary_concerns'] or not analysis['potential_conditions']:
            confidence *= 0.7
            
        # Lower confidence if no strong diagnostic matches
        if not any(d.get('confidence', 0) > 0.7 for d in diagnostic_matches):
            confidence *= 0.8
            
        # Lower confidence if multiple high-confidence diagnoses
        high_confidence_matches = sum(1 for d in diagnostic_matches if d.get('confidence', 0) > 0.7)
        if high_confidence_matches > 1:
            confidence *= 0.9
            
        return confidence

    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format diagnostic history"""
        if not history:
            return "No previous diagnostic history available"
            
        return f"""Previous Assessment:
- Primary Concerns: {', '.join(history.get('primary_concerns', []))}
- Conditions: {', '.join(history.get('potential_conditions', []))}
- Severity: {history.get('severity_level', 'unknown')}"""

    async def store_to_vector_db(self, query: str, response: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Store diagnostic results in the central vector database
        
        Args:
            query: User's query
            response: Agent's response
            context: Processing context
        """
        try:
            # Check if this contains diagnostic data
            if isinstance(response, dict) and 'response' in response and isinstance(response['response'], dict):
                diagnosis_data = response['response']
                
                # Add metadata
                diagnosis_data["timestamp"] = datetime.now().isoformat()
                diagnosis_data["query"] = query
                
                # Add user ID if available in context
                if context and "user_id" in context:
                    diagnosis_data["user_id"] = context["user_id"]
                
                # Store in vector database
                doc_id = add_user_data("diagnosis", diagnosis_data)
                
                if doc_id:
                    logger.info(f"Stored diagnostic assessment in vector DB: {doc_id}")
                else:
                    logger.warning("Failed to store diagnostic assessment in vector DB")
            
        except Exception as e:
            logger.error(f"Error storing diagnostic data in vector DB: {str(e)}")
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Conservative fallback analysis"""
        return {
            'primary_concerns': ['Unable to complete full diagnostic assessment'],
            'potential_conditions': [],
            'severity_level': 'unknown',
            'recommendations': [
                'Consult with a mental health professional',
                'Keep track of symptoms and their frequency',
                'Consider seeking a comprehensive evaluation'
            ],
            'additional_considerations': [
                'System limitations in diagnostic assessment',
                'Need for professional evaluation'
            ],
            'confidence': 0.3,  # Very low confidence for fallback
            'timestamp': datetime.now().isoformat()
        }