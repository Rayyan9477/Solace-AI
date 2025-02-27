from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from agno.tools import tool as Tool
from agno.memory import ConversationMemory
from agno.knowledge import VectorKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime

class RiskAssessmentTool(Tool):
    def __init__(self):
        super().__init__(
            name="risk_assessment",
            description="Assesses risk levels in mental health contexts"
        )
        
        self.risk_indicators = {
            'severe': [
                'suicide', 'kill', 'die', 'end it all', 'no reason to live',
                'hurt others', 'violence', 'weapon'
            ],
            'high': [
                'hopeless', 'worthless', 'trapped', 'burden', 'pain',
                'can\'t go on', 'give up'
            ],
            'moderate': [
                'depressed', 'anxious', 'stressed', 'overwhelmed',
                'scared', 'alone'
            ]
        }
        
    async def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get('text', '').lower()
        
        # Check for risk indicators
        risk_level = 'SAFE'
        found_indicators = []
        
        for level, indicators in self.risk_indicators.items():
            if any(indicator in text for indicator in indicators):
                risk_level = level.upper()
                found_indicators.extend([i for i in indicators if i in text])
                break
                
        return {
            'initial_risk_level': risk_level,
            'found_indicators': found_indicators,
            'requires_immediate_action': risk_level == 'SEVERE'
        }

class SafetyAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="safety_monitor",
            description="Expert system for mental health crisis assessment and intervention",
            tools=[RiskAssessmentTool()],
            memory=ConversationMemory(),
            knowledge=VectorKnowledge()
        )
        
        self.safety_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health crisis assessment and intervention.
Your role is to analyze messages for safety concerns and risk factors, considering:
1. Suicidal ideation or self-harm indicators
2. Risk of harm to others
3. Acute psychological distress
4. Signs of crisis or emergency
5. Need for immediate intervention

Provide a thorough safety assessment while maintaining:
- High sensitivity to risk indicators
- Clear actionable recommendations
- Appropriate urgency levels
- Professional clinical judgment

Risk Levels:
- SEVERE: Immediate danger, emergency services needed
- HIGH: Urgent intervention required
- MODERATE: Close monitoring needed
- LOW: General support appropriate
- SAFE: No immediate concerns"""),
            HumanMessage(content="""Message: {message}
Emotional Context: {emotion_data}
Previous Safety State: {history}
Initial Risk Assessment: {risk_assessment}

Provide a structured safety assessment in the following format:
Risk Level: [level]
Primary Concerns: [comma-separated list]
Warning Signs: [comma-separated list]
Recommended Actions: [comma-separated list]
Emergency Protocol: [Yes/No]
Resources: [relevant crisis resources]""")
        ])

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Get risk assessment
            risk_result = tool_results.get('risk_assessment', {})
            
            # Get message and contexts
            message = input_data.get('text', '')
            emotion_data = input_data.get('emotion_data', {})
            history = context.get('memory', {}).get('last_assessment', {})
            
            # Generate LLM analysis
            llm_response = await self.llm.agenerate_messages([
                self.safety_prompt.format_messages(
                    message=message,
                    emotion_data=self._format_emotion_data(emotion_data),
                    history=self._format_history(history),
                    risk_assessment=risk_result
                )[0]
            ])
            
            # Parse response
            analysis = self._parse_result(llm_response.generations[0][0].text)
            
            # Add metadata
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['confidence'] = self._calculate_confidence(analysis, risk_result)
            
            return analysis
            
        except Exception as e:
            return self._fallback_analysis()

    def _parse_result(self, text: str) -> Dict[str, Any]:
        """Parse the structured output from Claude"""
        result = {
            'safe': True,
            'risk_level': 'SAFE',
            'concerns': [],
            'warning_signs': [],
            'recommendations': [],
            'emergency_protocol': False,
            'resources': []
        }
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [x.strip() for x in line.split(':', 1)]
                
                if 'Risk Level' in key:
                    risk_level = value.upper()
                    result['risk_level'] = risk_level
                    result['safe'] = risk_level in ['SAFE', 'LOW']
                elif 'Primary Concerns' in key:
                    result['concerns'] = [c.strip() for c in value.split(',')]
                elif 'Warning Signs' in key:
                    result['warning_signs'] = [w.strip() for w in value.split(',')]
                elif 'Recommended Actions' in key:
                    result['recommendations'] = [r.strip() for r in value.split(',')]
                elif 'Emergency Protocol' in key:
                    result['emergency_protocol'] = value.lower() == 'yes'
                elif 'Resources' in key:
                    result['resources'] = [r.strip() for r in value.split(',')]
                    
        except Exception:
            pass
            
        return result

    def _format_emotion_data(self, emotion_data: Dict[str, Any]) -> str:
        """Format emotional context for safety analysis"""
        if not emotion_data:
            return "No emotional context available"
            
        return f"""Emotional State:
- Primary: {emotion_data.get('primary_emotion', 'unknown')}
- Secondary: {', '.join(emotion_data.get('secondary_emotions', []))}
- Intensity: {emotion_data.get('intensity', 'unknown')}
- Clinical Indicators: {', '.join(emotion_data.get('clinical_indicators', []))}"""

    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format safety history"""
        if not history:
            return "No previous safety context available"
            
        return f"""Previous Safety State:
- Risk Level: {history.get('risk_level', 'unknown')}
- Concerns: {', '.join(history.get('concerns', []))}
- Protocol: {'Emergency protocol was activated' if history.get('emergency_protocol') else 'No emergency protocol'}"""

    def _calculate_confidence(self, analysis: Dict[str, Any], risk_result: Dict[str, Any]) -> float:
        """Calculate confidence in safety assessment"""
        confidence = 1.0
        
        # Lower confidence if analysis is incomplete
        if not analysis['concerns'] or not analysis['warning_signs']:
            confidence *= 0.8
            
        # Lower confidence for severe cases without clear indicators
        if analysis['risk_level'] in ['SEVERE', 'HIGH'] and not risk_result.get('found_indicators'):
            confidence *= 0.6
            
        # Lower confidence if risk levels don't match
        if analysis['risk_level'] != risk_result.get('initial_risk_level', 'SAFE'):
            confidence *= 0.7
            
        return confidence

    def _fallback_analysis(self) -> Dict[str, Any]:
        """Conservative fallback analysis"""
        return {
            'safe': False,
            'risk_level': 'MODERATE',
            'concerns': ['Unable to complete full safety analysis'],
            'warning_signs': ['System limitation in risk assessment'],
            'recommendations': [
                'Conduct manual safety assessment',
                'Consult with mental health professional',
                'Monitor situation closely'
            ],
            'emergency_protocol': False,
            'resources': [
                'National Crisis Hotline: 988',
                'Emergency Services: 911'
            ],
            'confidence': 0.4,  # Low confidence for fallback
            'timestamp': datetime.now().isoformat()
        }