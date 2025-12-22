from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from datetime import datetime
import logging

# Import memory factory for centralized memory management
from src.utils.memory_factory import create_agent_memory

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
        # Create memory using centralized factory
        memory = create_agent_memory()

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
        """Parse the structured output from Claude.

        SECURITY: Defaults to UNSAFE (safe=False) to ensure errors fail safely.
        Only marks as safe if explicit 'SAFE' or 'LOW' risk level is parsed.
        """
        result = {
            'safe': False,  # SECURITY: Default to unsafe - must be explicitly set to safe
            'risk_level': 'UNKNOWN',
            'concerns': [],
            'warning_signs': [],
            'recommendations': ['Seek professional evaluation due to parsing error'],
            'emergency_protocol': False,
            'resources': ['National Crisis Hotline: 988 (24/7)']
        }

        try:
            lines = text.strip().split('\n')
            risk_level_found = False

            for line in lines:
                if ':' not in line:
                    continue

                key, value = [x.strip() for x in line.split(':', 1)]

                if 'Risk Level' in key:
                    risk_level = value.upper()
                    result['risk_level'] = risk_level
                    result['safe'] = risk_level in ['SAFE', 'LOW']
                    risk_level_found = True
                elif 'Primary Concerns' in key:
                    result['concerns'] = [c.strip() for c in value.split(',') if c.strip()]
                elif 'Warning Signs' in key:
                    result['warning_signs'] = [w.strip() for w in value.split(',') if w.strip()]
                elif 'Recommended Actions' in key:
                    result['recommendations'] = [r.strip() for r in value.split(',') if r.strip()]
                elif 'Emergency Protocol' in key:
                    result['emergency_protocol'] = value.lower() == 'yes'
                elif 'Resources' in key:
                    result['resources'] = [r.strip() for r in value.split(',') if r.strip()]

            # If no risk level was found in parsing, ensure conservative defaults
            if not risk_level_found:
                logger.warning("No risk level found in safety assessment - defaulting to UNKNOWN/unsafe")
                result['concerns'].append('Risk level could not be determined')

        except Exception as e:
            logger.error(f"Error parsing safety assessment result: {str(e)}")
            # SECURITY: On parse error, return conservative unsafe result
            result['safe'] = False
            result['risk_level'] = 'UNKNOWN'
            result['concerns'] = ['Error parsing safety assessment', str(e)]
            result['recommendations'] = [
                'Seek professional evaluation',
                'Contact crisis services if concerned'
            ]

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

    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """
        Enhanced fallback analysis with keyword-based risk detection.

        This fallback performs actual risk assessment using predefined
        risk indicators when the primary LLM-based analysis fails.

        Args:
            message: User's message text

        Returns:
            Dict with safety assessment including risk level, concerns,
            warning signs, recommendations, and emergency protocol status
        """
        try:
            message_lower = message.lower()

            # Detect risk indicators using existing RISK_INDICATORS
            risk_level = 'SAFE'
            found_indicators = []
            concerns = []
            warning_signs = []

            # Check for severe risk indicators
            for indicator in RISK_INDICATORS['severe']:
                if indicator in message_lower:
                    risk_level = 'SEVERE'
                    found_indicators.append(indicator)
                    concerns.append(f"Severe risk indicator detected: {indicator}")
                    warning_signs.append(f"Language suggesting {indicator}")

            # Check for high risk indicators if not already severe
            if risk_level != 'SEVERE':
                for indicator in RISK_INDICATORS['high']:
                    if indicator in message_lower:
                        risk_level = 'HIGH'
                        found_indicators.append(indicator)
                        concerns.append(f"High risk indicator: {indicator}")
                        warning_signs.append(f"Expression of {indicator}")

            # Check for moderate risk indicators if not already elevated
            if risk_level not in ['SEVERE', 'HIGH']:
                for indicator in RISK_INDICATORS['moderate']:
                    if indicator in message_lower:
                        risk_level = 'MODERATE'
                        found_indicators.append(indicator)
                        concerns.append(f"Moderate concern: {indicator}")
                        warning_signs.append(f"Indication of {indicator}")

            # Determine if emergency protocol is needed
            emergency_protocol = risk_level == 'SEVERE'

            # Generate appropriate recommendations
            recommendations = self._generate_fallback_recommendations(risk_level)

            # Determine safety status
            is_safe = risk_level == 'SAFE'

            # Calculate confidence based on findings
            confidence = 0.7 if found_indicators else 0.5
            if risk_level == 'SEVERE':
                confidence = 0.8  # Higher confidence in severe cases

            return {
                'safe': is_safe,
                'risk_level': risk_level,
                'concerns': concerns if concerns else ['No immediate concerns detected'],
                'warning_signs': warning_signs if warning_signs else ['No warning signs detected'],
                'recommendations': recommendations,
                'emergency_protocol': emergency_protocol,
                'resources': self._get_crisis_resources(risk_level),
                'confidence': confidence,
                'found_indicators': found_indicators,
                'analysis_method': 'fallback_keyword_based',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in safety fallback analysis: {str(e)}", exc_info=True)
            # Last resort ultra-conservative fallback
            return {
                'safe': False,
                'risk_level': 'MODERATE',
                'concerns': ['Unable to complete safety analysis', str(e)],
                'warning_signs': ['System error in risk assessment'],
                'recommendations': [
                    'Seek immediate professional evaluation',
                    'Contact mental health crisis services',
                    'Do not proceed without professional assessment'
                ],
                'emergency_protocol': False,
                'resources': [
                    'National Crisis Hotline: 988',
                    'Emergency Services: 911',
                    'Crisis Text Line: Text HOME to 741741'
                ],
                'confidence': 0.2,
                'error': str(e),
                'analysis_method': 'minimal_fallback',
                'timestamp': datetime.now().isoformat()
            }

    def _generate_fallback_recommendations(self, risk_level: str) -> list:
        """
        Generate risk-appropriate safety recommendations and intervention strategies.

        Provides tiered recommendations based on assessed risk level, ranging from
        emergency intervention protocols to preventive self-care strategies. All
        recommendations are evidence-based and follow mental health crisis response
        best practices.

        Args:
            risk_level (str): Assessed risk level - one of:
                - 'SEVERE': Immediate danger, requires emergency intervention
                - 'HIGH': Significant risk, urgent professional help needed
                - 'MODERATE': Some concern, monitoring and professional consultation advised
                - 'SAFE': No immediate risk, general wellness strategies appropriate

        Returns:
            list[str]: Ordered list of actionable recommendations, with most urgent/critical
            actions first. List length varies by risk level (5 items for SEVERE/HIGH, 4-5
            items for MODERATE/SAFE).

        Example:
            >>> recommendations = self._generate_fallback_recommendations('HIGH')
            >>> print(recommendations[0])
            'Contact mental health crisis services immediately'
            >>> recommendations = self._generate_fallback_recommendations('SAFE')
            >>> print(recommendations[0])
            'Continue current coping strategies'

        Note:
            SEVERE and HIGH risk recommendations always include 988 Crisis Hotline and
            emphasize immediate action. Recommendations are intentionally directive and
            clear to facilitate rapid response during crisis situations.
        """
        if risk_level == 'SEVERE':
            return [
                'IMMEDIATE ACTION REQUIRED: Contact emergency services (911) if in immediate danger',
                'Call National Crisis Hotline: 988',
                'Do not leave the person alone',
                'Remove access to means of self-harm',
                'Seek emergency psychiatric evaluation'
            ]
        elif risk_level == 'HIGH':
            return [
                'Contact mental health crisis services immediately',
                'Call National Crisis Hotline: 988',
                'Schedule urgent appointment with mental health professional',
                'Ensure someone is available for support',
                'Create a safety plan with professional guidance'
            ]
        elif risk_level == 'MODERATE':
            return [
                'Consider scheduling appointment with mental health professional',
                'Reach out to trusted support network',
                'Practice self-care and coping strategies',
                'Monitor symptoms and seek help if they worsen',
                'National Crisis Hotline available 24/7: 988'
            ]
        else:  # SAFE
            return [
                'Continue current coping strategies',
                'Maintain connection with support network',
                'Practice regular self-care',
                'Seek professional support if needed in future'
            ]

    def _get_crisis_resources(self, risk_level: str) -> list:
        """
        Provide crisis intervention resources tailored to assessed risk severity.

        Returns appropriate mental health crisis resources, with resource selection
        based on risk level. Higher risk scenarios receive comprehensive resource
        lists including specialized crisis services.

        Args:
            risk_level (str): Assessed risk level ('SAFE', 'MODERATE', 'HIGH', 'SEVERE')

        Returns:
            list[str]: Crisis resources with contact information, ordered by relevance:
                - For HIGH/SEVERE: 5 resources including emergency services and specialized crisis lines
                - For MODERATE/SAFE: 3 basic resources (988, Crisis Text Line, 911)

        Example:
            >>> resources = self._get_crisis_resources('SEVERE')
            >>> print(len(resources))
            5
            >>> print(resources[0])
            'National Crisis Hotline: 988 (24/7)'
            >>> resources = self._get_crisis_resources('MODERATE')
            >>> print(len(resources))
            3

        Note:
            - 988 is the new 3-digit crisis number (launched July 2022, replacing 1-800-273-8255)
            - All resources listed are free, confidential, and available 24/7
            - Resources are U.S.-specific; international users should be directed to local services
        """
        base_resources = [
            'National Crisis Hotline: 988 (24/7)',
            'Crisis Text Line: Text HOME to 741741',
            'Emergency Services: 911'
        ]

        if risk_level in ['SEVERE', 'HIGH']:
            return base_resources + [
                'National Suicide Prevention Lifeline: 1-800-273-8255',
                'SAMHSA National Helpline: 1-800-662-4357'
            ]
        else:
            return base_resources