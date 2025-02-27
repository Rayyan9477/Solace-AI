from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from agno.tools import Tool
from agno.memory import ConversationMemory
from agno.knowledge import VectorKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import spacy
from transformers import pipeline
from datetime import datetime

class SymptomExtractionTool(Tool):
    def __init__(self):
        super().__init__(
            name="symptom_extraction",
            description="Extracts mental health symptoms from text using NLP"
        )
        # Load spaCy model for medical entity recognition
        self.nlp = spacy.load("en_core_web_sm")
        # Load zero-shot classifier for symptom classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU
        )
        
        self.symptom_categories = [
            "mood symptoms",
            "anxiety symptoms",
            "cognitive symptoms",
            "behavioral symptoms",
            "physical symptoms",
            "social symptoms"
        ]
        
    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get('text', '')
        
        # Extract entities using spaCy
        doc = self.nlp(text)
        entities = [
            ent.text for ent in doc.ents 
            if ent.label_ in ["SYMPTOM", "CONDITION", "BEHAVIOR"]
        ]
        
        # Classify symptoms into categories
        if text:
            classifications = self.classifier(
                text,
                self.symptom_categories,
                multi_label=True
            )
            
            symptom_categories = [
                cat for cat, score in zip(classifications['labels'], classifications['scores'])
                if score > 0.5
            ]
        else:
            symptom_categories = []
            
        return {
            'extracted_symptoms': entities,
            'symptom_categories': symptom_categories,
            'raw_text': text
        }

class DiagnosticCriteriaTool(Tool):
    def __init__(self):
        super().__init__(
            name="diagnostic_criteria",
            description="Matches symptoms to diagnostic criteria"
        )
        # Load diagnostic criteria classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        
        self.diagnostic_categories = [
            "Major Depressive Disorder",
            "Generalized Anxiety Disorder",
            "Bipolar Disorder",
            "Post-Traumatic Stress Disorder",
            "Social Anxiety Disorder",
            "Panic Disorder"
        ]
        
    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = input_data.get('symptoms', [])
        symptom_text = " ".join(symptoms)
        
        if symptom_text:
            # Classify symptoms against diagnostic criteria
            classifications = self.classifier(
                symptom_text,
                self.diagnostic_categories,
                multi_label=True
            )
            
            potential_diagnoses = [
                {
                    'condition': label,
                    'confidence': score,
                    'severity': self._estimate_severity(score)
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
        
    def _estimate_severity(self, confidence_score: float) -> str:
        if confidence_score > 0.8:
            return "severe"
        elif confidence_score > 0.6:
            return "moderate"
        else:
            return "mild"

class DiagnosisAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="mental_health_diagnostician",
            description="Expert system for mental health symptom analysis and diagnosis",
            tools=[
                SymptomExtractionTool(),
                DiagnosticCriteriaTool()
            ],
            memory=ConversationMemory(),
            knowledge=VectorKnowledge()
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

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Get symptom analysis
            symptom_data = tool_results.get('symptom_extraction', {})
            diagnostic_data = tool_results.get('diagnostic_criteria', {})
            
            # Get history
            history = context.get('memory', {}).get('last_diagnosis', {})
            
            # Generate diagnostic assessment
            llm_response = await self.llm.agenerate_messages([
                self.diagnosis_prompt.format_messages(
                    symptoms=symptom_data.get('extracted_symptoms', []),
                    symptom_categories=symptom_data.get('symptom_categories', []),
                    diagnostic_matches=diagnostic_data.get('potential_diagnoses', []),
                    history=self._format_history(history)
                )[0]
            ])
            
            # Parse response
            analysis = self._parse_result(llm_response.generations[0][0].text)
            
            # Add metadata
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['confidence'] = self._calculate_confidence(
                analysis,
                diagnostic_data.get('potential_diagnoses', [])
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