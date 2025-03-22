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

# Load models
nlp = spacy.load("en_core_web_sm")
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

class DiagnosisAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="mental_health_diagnostician",
            role="Expert system for mental health symptom analysis and diagnosis",
            description="""An AI agent specialized in analyzing mental health symptoms and providing diagnostic insights.
            Uses evidence-based criteria and maintains clinical accuracy while emphasizing the importance of professional evaluation.""",
            tools=[extract_symptoms, analyze_diagnostic_criteria],
            memory=Memory(storage={}),  # Initialize with storage parameter
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

    async def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate diagnostic assessment"""
        try:
            # Extract symptoms
            symptom_data = await extract_symptoms(query)
            
            # Analyze diagnostic criteria
            diagnostic_data = await analyze_diagnostic_criteria(
                symptom_data['extracted_symptoms']
            )
            
            # Get history
            history = context.get('memory', {}).get('last_diagnosis', {}) if context else {}
            
            # Generate diagnostic assessment
            llm_response = await self.model.agenerate_messages([
                self.diagnosis_prompt.format_messages(
                    symptoms=symptom_data['extracted_symptoms'],
                    symptom_categories=symptom_data['symptom_categories'],
                    diagnostic_matches=diagnostic_data['potential_diagnoses'],
                    history=self._format_history(history)
                )[0]
            ])
            
            # Parse response
            analysis = self._parse_result(llm_response.generations[0][0].text)
            
            # Add metadata
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['confidence'] = self._calculate_confidence(
                analysis,
                diagnostic_data['potential_diagnoses']
            )
            
            return analysis
            
        except Exception as e:
            return self._fallback_analysis()

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

    def _fallback_analysis(self) -> Dict[str, Any]:
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