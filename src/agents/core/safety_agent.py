from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

# Define risk indicators
RISK_INDICATORS = {
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

@tool(name="risk_assessment", description="Assesses risk levels in mental health contexts")
async def assess_risk(text: str) -> Dict[str, Any]:
    """
    Assesses risk levels in mental health contexts
    
    Args:
        text: The text to analyze for risk indicators
        
    Returns:
        Dictionary containing risk assessment results
    """
    try:
        text = text.lower()
        
        # Check for risk indicators
        risk_level = 'SAFE'
        found_indicators = []
        
        for level, indicators in RISK_INDICATORS.items():
            if any(indicator in text for indicator in indicators):
                risk_level = level.upper()
                found_indicators.extend([i for i in indicators if i in text])
                break
                
        return {
            'initial_risk_level': risk_level,
            'found_indicators': found_indicators,
            'requires_immediate_action': risk_level == 'SEVERE'
        }
    except Exception as e:
        return {
            'initial_risk_level': 'MODERATE',  # Conservative fallback
            'found_indicators': [],
            'requires_immediate_action': False,
            'error': str(e)
        }

class SafetyAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        # Create a langchain memory instance
        langchain_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Create memory dict for agno Memory
        memory_dict = {
            "memory": "chat_memory",  # Memory parameter should be a string
            "storage": "local_storage",  # Storage parameter should be a string
            "memory_key": "chat_history",
            "chat_memory": langchain_memory,
            "input_key": "input",
            "output_key": "output",
            "return_messages": True
        }
        
        # Initialize Memory with the dictionary
        memory = Memory(**memory_dict)
        
        super().__init__(
            model=model,
            name="safety_monitor",
            description="Expert system for mental health crisis assessment and intervention",
            tools=[assess_risk],
            memory=memory,
            knowledge=AgentKnowledge()
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

    async def check_message(self, message: str) -> Dict[str, Any]:
        """
        Check a message for safety concerns
        
        Args:
            message: The message to check
            
        Returns:
            Dictionary containing safety assessment
        """
        try:
            # Create a basic safety assessment
            assessment = {
                'risk_level': 'low',
                'risk_factors': [],
                'safety_concerns': [],
                'recommendations': [],
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for high-risk keywords
            message_lower = message.lower()
            
            # Check for immediate risk indicators
            for indicator in RISK_INDICATORS['severe']:
                if indicator in message_lower:
                    assessment['risk_level'] = 'high'
                    assessment['risk_factors'].append(f'immediate_risk_{indicator}')
                    assessment['safety_concerns'].append(f'Immediate risk detected: {indicator}')
                    assessment['recommendations'].append('Immediate professional help recommended')
            
            # Check for moderate risk indicators
            for indicator in RISK_INDICATORS['high']:
                if indicator in message_lower:
                    if assessment['risk_level'] != 'high':
                        assessment['risk_level'] = 'moderate'
                    assessment['risk_factors'].append(f'moderate_risk_{indicator}')
                    assessment['safety_concerns'].append(f'Moderate risk detected: {indicator}')
                    assessment['recommendations'].append('Professional support recommended')
            
            # Check for low risk indicators
            for indicator in RISK_INDICATORS['moderate']:
                if indicator in message_lower:
                    if assessment['risk_level'] not in ['high', 'moderate']:
                        assessment['risk_level'] = 'low'
                    assessment['risk_factors'].append(f'low_risk_{indicator}')
                    assessment['safety_concerns'].append(f'Low risk detected: {indicator}')
                    assessment['recommendations'].append('Monitor and provide support')
            
            # Add general recommendations based on risk level
            if assessment['risk_level'] == 'high':
                assessment['recommendations'].extend([
                    'Contact emergency services if immediate danger',
                    'Do not leave the person alone',
                    'Remove any potential means of self-harm'
                ])
            elif assessment['risk_level'] == 'moderate':
                assessment['recommendations'].extend([
                    'Schedule a mental health professional appointment',
                    'Provide crisis hotline numbers',
                    'Check in regularly'
                ])
            else:
                assessment['recommendations'].extend([
                    'Continue monitoring',
                    'Provide emotional support',
                    'Encourage healthy coping mechanisms'
                ])
            
            # Try to update memory, but don't fail if it doesn't work
            try:
                await self.memory.add("last_safety_check", assessment)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error checking message safety: {str(e)}")
            return {
                'risk_level': 'unknown',
                'risk_factors': [],
                'safety_concerns': ['Error analyzing safety'],
                'recommendations': ['Seek professional help if concerned'],
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

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