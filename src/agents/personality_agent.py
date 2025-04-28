from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import logging
from datetime import datetime
import json
import os
import sys

# Add the project root to the path to import the personality modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from personality.big_five import BigFiveAssessment
from personality.mbti import MBTIAssessment

# Initialize logger
logger = logging.getLogger(__name__)

@tool(name="personality_assessment", description="Conducts personality assessments using Big Five or MBTI models")
async def assess_personality(assessment_type: str, responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conducts a personality assessment based on user responses
    
    Args:
        assessment_type: The type of assessment to conduct ('big_five' or 'mbti')
        responses: Dictionary containing user responses to assessment questions
        
    Returns:
        Dictionary containing assessment results and interpretation
    """
    try:
        if assessment_type.lower() == 'big_five':
            assessment = BigFiveAssessment()
            return assessment.compute_results(responses)
        elif assessment_type.lower() == 'mbti':
            assessment = MBTIAssessment()
            return assessment.compute_results(responses)
        else:
            return {
                "error": f"Unknown assessment type: {assessment_type}",
                "valid_types": ["big_five", "mbti"]
            }
    except Exception as e:
        logger.error(f"Error in personality assessment: {str(e)}")
        return {
            "error": str(e),
            "assessment_type": assessment_type
        }

class PersonalityAgent(BaseAgent):
    def __init__(self, model: BaseLanguageModel):
        # Create a memory instance
        memory = Memory(
            memory="personality_memory",
            storage="local_storage"
        )
        
        super().__init__(
            model=model,
            name="personality_analyzer",
            description="Expert system for personality assessment and interpretation",
            tools=[assess_personality],
            memory=memory,
            knowledge=AgentKnowledge()
        )
        
        self.interpretation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in personality psychology and assessment.
Your role is to interpret personality assessment results and provide insights on:
1. Key personality traits and their implications
2. Potential strengths and growth areas
3. Communication preferences and learning styles
4. Stress responses and coping mechanisms
5. Relationship dynamics and teamwork preferences

Provide a balanced, nuanced interpretation that avoids stereotyping or overgeneralizing.
Focus on how understanding personality can help with personal growth and self-awareness."""),
            HumanMessage(content="""Assessment Type: {assessment_type}
Assessment Results: {assessment_results}
User Context: {user_context}

Provide an interpretation of these results, focusing on:
- Key insights about the personality profile
- Potential strengths and growth areas
- Communication and learning preferences
- Stress responses and coping strategies
- How this information might help with the user's mental health journey""")
        ])

    async def conduct_assessment(self, assessment_type: str, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct a personality assessment and provide interpretation
        
        Args:
            assessment_type: The type of assessment ('big_five' or 'mbti')
            responses: User responses to assessment questions
            
        Returns:
            Dictionary containing assessment results and interpretation
        """
        try:
            # Get assessment results
            results = await assess_personality(assessment_type, responses)
            
            # Store results in memory
            try:
                await self.memory.add("assessment_results", {
                    "type": assessment_type,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            # Generate interpretation
            interpretation = await self._generate_interpretation(assessment_type, results)
            
            # Combine results and interpretation
            return {
                "assessment_type": assessment_type,
                "results": results,
                "interpretation": interpretation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error conducting personality assessment: {str(e)}")
            return {
                "error": str(e),
                "assessment_type": assessment_type
            }
    
    async def _generate_interpretation(self, assessment_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an interpretation of assessment results"""
        try:
            # Get user context from memory
            user_context = await self.memory.get("user_context", {})
            
            # Generate interpretation using LLM
            llm_response = await self.llm.agenerate_messages([
                self.interpretation_prompt.format_messages(
                    assessment_type=assessment_type,
                    assessment_results=json.dumps(results, indent=2),
                    user_context=json.dumps(user_context, indent=2)
                )[0]
            ])
            
            # Parse response
            interpretation = self._parse_interpretation(llm_response.generations[0][0].text)
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error generating interpretation: {str(e)}")
            return {
                "error": str(e),
                "fallback_interpretation": "Unable to generate detailed interpretation at this time."
            }
    
    def _parse_interpretation(self, text: str) -> Dict[str, Any]:
        """Parse the structured interpretation from LLM response"""
        sections = {
            "key_insights": [],
            "strengths": [],
            "growth_areas": [],
            "communication_preferences": [],
            "stress_responses": [],
            "mental_health_implications": []
        }
        
        current_section = None
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                lower_line = line.lower()
                if "key insight" in lower_line or "overview" in lower_line:
                    current_section = "key_insights"
                elif "strength" in lower_line:
                    current_section = "strengths"
                elif "growth" in lower_line or "challenge" in lower_line or "development" in lower_line:
                    current_section = "growth_areas"
                elif "communication" in lower_line or "learning" in lower_line:
                    current_section = "communication_preferences"
                elif "stress" in lower_line or "coping" in lower_line:
                    current_section = "stress_responses"
                elif "mental health" in lower_line or "wellbeing" in lower_line or "well-being" in lower_line:
                    current_section = "mental_health_implications"
                elif current_section and line.startswith('-'):
                    # Add bullet points to current section
                    sections[current_section].append(line[1:].strip())
                elif current_section and not any(header in lower_line for header in ["key", "strength", "growth", "communication", "stress", "mental"]):
                    # Add non-header text to current section
                    sections[current_section].append(line)
        except Exception:
            # If parsing fails, return the raw text
            return {
                "raw_interpretation": text,
                "parsing_error": True
            }
            
        return sections
    
    async def get_previous_assessment(self) -> Dict[str, Any]:
        """Retrieve the most recent assessment results from memory"""
        try:
            return await self.memory.get("assessment_results", {})
        except Exception as e:
            logger.warning(f"Failed to retrieve assessment from memory: {str(e)}")
            return {}
    
    async def update_user_context(self, context: Dict[str, Any]) -> None:
        """Update the user context stored in memory"""
        try:
            await self.memory.add("user_context", context)
        except Exception as e:
            logger.warning(f"Failed to update user context: {str(e)}")
