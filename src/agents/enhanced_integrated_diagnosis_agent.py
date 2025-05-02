"""
Enhanced integrated diagnosis agent that combines personality assessment and mental health diagnosis
using DSPy-powered Agentic RAG for more comprehensive and personalized results.
"""

from typing import Dict, Any, Optional, List, Union
from .base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import AgenticRAG system and other components
from src.utils.agentic_rag import AgenticRAG
from src.components.integrated_assessment import IntegratedAssessment
from src.components.diagnosis_results import DiagnosisResults

logger = logging.getLogger(__name__)

@tool("integrated_assessment")
async def perform_integrated_assessment(
    personality_data: Dict[str, Any],
    diagnosis_data: Dict[str, Any],
    emotion_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs an integrated assessment combining personality traits, diagnostic data, and emotion analysis
    
    Args:
        personality_data: Results from personality assessment
        diagnosis_data: Results from diagnostic assessment
        emotion_data: Optional emotion analysis data
        
    Returns:
        Integrated insights combining all assessment data
    """
    assessment = IntegratedAssessment()
    return await assessment.generate_integrated_insights(
        personality_data=personality_data,
        diagnosis_data=diagnosis_data,
        emotion_data=emotion_data
    )

@tool("enhanced_integrated_assessment")
async def enhanced_integrated_assessment(
    personality_data: Dict[str, Any],
    diagnosis_data: Dict[str, Any],
    emotion_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform an enhanced integrated assessment using Agentic RAG with structured reasoning
    
    Args:
        personality_data: Results from personality assessment
        diagnosis_data: Results from diagnostic assessment
        emotion_data: Optional emotion analysis data
        
    Returns:
        Enhanced integrated insights with more sophisticated reasoning
    """
    if not hasattr(enhanced_integrated_assessment, "rag_system"):
        # Initialize the RAG system if not already available
        try:
            from src.models.llm import get_llm
            llm = get_llm()
            
            # Set up the knowledge base directory
            project_root = Path(__file__).parent.parent.parent
            kb_dir = project_root / "src" / "data" / "personality"
            
            # Initialize the Agentic RAG system
            enhanced_integrated_assessment.rag_system = AgenticRAG(
                llm=llm,
                knowledge_base_dir=str(kb_dir)
            )
            logger.info("Initialized Agentic RAG system for enhanced integrated assessment")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return {
                "error": f"Failed to initialize assessment system: {str(e)}",
                "fallback": "Using standard assessment"
            }
    
    try:
        # Create a combined dataset for assessment
        combined_data = {
            "personality": personality_data,
            "diagnosis": diagnosis_data
        }
        
        if emotion_data:
            combined_data["emotion"] = emotion_data
        
        # Use the Agentic RAG system to enhance the personality assessment first
        personality_insights = await enhanced_integrated_assessment.rag_system.enhance_personality_assessment(
            assessment_data=personality_data,
            emotional_context=emotion_data
        )
        
        # Combine all results into an integrated assessment
        result = {
            "success": True,
            "integrated_insights": {
                "personality_analysis": personality_insights.get("analysis", ""),
                "personality_strengths": personality_insights.get("strengths", ""),
                "growth_areas": personality_insights.get("growth_areas", ""),
                "mental_health_implications": {
                    "symptoms": diagnosis_data.get("symptoms", []),
                    "potential_diagnoses": diagnosis_data.get("potential_diagnoses", []),
                    "severity": diagnosis_data.get("severity", "mild")
                },
                "communication_style": personality_insights.get("communication_style", ""),
                "learning_style": personality_insights.get("learning_style", ""),
                "stress_response": personality_insights.get("stress_response", ""),
                "recommendations": [
                    *personality_insights.get("emotion_recommendations", []),
                    *diagnosis_data.get("recommendations", [])
                ],
                "emotional_patterns": personality_insights.get("emotion_correlations", "")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in enhanced integrated assessment: {str(e)}")
        return {
            "error": str(e),
            "fallback": "Using standard assessment"
        }

class EnhancedIntegratedDiagnosisAgent(BaseAgent):
    """
    Enhanced Agent that integrates personality assessment, emotion analysis, and mental health diagnosis
    using DSPy-powered Agentic RAG for more comprehensive assessments.
    """
    
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="enhanced_integrated_diagnosis_agent",
            role="Advanced integrated mental health analysis system",
            description="""An enhanced AI agent that integrates personality assessment, emotion analysis, and mental health diagnosis.
            Uses DSPy-powered Agentic RAG for more accurate assessments, structured reasoning, and personalized recommendations.
            Combines clinical psychology with computational methods to provide a holistic view of mental well-being.""",
            tools=[perform_integrated_assessment, enhanced_integrated_assessment],
            memory=Memory(memory="integrated_diagnosis_memory", storage="local_storage"),
            knowledge=AgentKnowledge()
        )
        
        # Initialize the Agentic RAG system
        try:
            project_root = Path(__file__).parent.parent.parent
            kb_dir = project_root / "src" / "data" / "personality"
            
            self.rag_system = AgenticRAG(
                llm=model,
                knowledge_base_dir=str(kb_dir)
            )
            logger.info("Initialized Agentic RAG system for EnhancedIntegratedDiagnosisAgent")
        except Exception as e:
            logger.error(f"Failed to initialize AgenticRAG: {str(e)}")
            self.rag_system = None
        
        # Initialize IntegratedAssessment component for fallback
        self.integrated_assessment = IntegratedAssessment()
        
        # Set up prompts
        self.integration_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert system that integrates personality assessment, emotion analysis, and mental health diagnosis.
Your role is to provide a comprehensive understanding of an individual's psychological profile by:
1. Connecting personality traits with mental health symptoms
2. Identifying how emotions influence both personality and mental health
3. Recognizing patterns that span across different assessment domains
4. Providing personalized recommendations based on the integrated profile
5. Maintaining a strengths-based approach while acknowledging areas of concern

Guidelines:
- Emphasize interconnections between personality, emotions, and mental health
- Provide a holistic interpretation rather than isolated assessments
- Maintain a balance between recognizing strengths and areas of concern
- Prioritize recommendations that align with the individual's personality traits
- Consider how personality traits may influence both symptoms and treatment approaches
- Highlight how the integrated understanding provides more value than separate assessments"""),
            HumanMessage(content="""Personality Assessment: {personality_data}
Diagnostic Assessment: {diagnosis_data}
Emotion Analysis: {emotion_data}
Previous Assessments: {previous_assessments}

Provide a comprehensive integrated assessment:
Personality Profile: [summary of key personality traits and patterns]
Mental Health Overview: [summary of mental health symptoms and concerns]
Emotional Patterns: [summary of emotional tendencies and influences]
Integrated Insights: [how personality, emotions, and mental health interconnect]
Key Strengths: [strengths that can support mental well-being]
Growth Opportunities: [areas where growth could benefit mental health]
Personalized Recommendations: [tailored suggestions based on the integrated profile]
Next Steps: [concrete actions based on the integrated assessment]""")
        ])
    
    async def get_integrated_assessment(
        self,
        personality_data: Dict[str, Any],
        diagnosis_data: Dict[str, Any],
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an integrated assessment combining personality, diagnosis, and emotion data
        
        Args:
            personality_data: Results from personality assessment
            diagnosis_data: Results from diagnostic assessment
            emotion_data: Optional emotion analysis data
            
        Returns:
            Integrated assessment with insights across domains
        """
        try:
            # Use the Agentic RAG system if available
            if self.rag_system:
                enhanced_result = await enhanced_integrated_assessment(
                    personality_data=personality_data,
                    diagnosis_data=diagnosis_data,
                    emotion_data=emotion_data
                )
                
                if enhanced_result.get("success", False) and "error" not in enhanced_result:
                    return enhanced_result
            
            # Fallback to traditional integration if RAG fails
            return await self.integrated_assessment.generate_integrated_insights(
                personality_data=personality_data,
                diagnosis_data=diagnosis_data,
                emotion_data=emotion_data
            )
            
        except Exception as e:
            logger.error(f"Error in integrated assessment: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def generate_integrated_report(
        self,
        personality_data: Dict[str, Any],
        diagnosis_data: Dict[str, Any],
        emotion_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report integrating all assessment data
        
        Args:
            personality_data: Results from personality assessment
            diagnosis_data: Results from diagnostic assessment
            emotion_data: Optional emotion analysis data
            
        Returns:
            Comprehensive integrated report with insights and recommendations
        """
        try:
            logger.info("Generating integrated report")
            
            # Get previous assessments from memory
            previous_assessments = await self._get_previous_assessments()
            
            # Convert data to string format for the prompt
            personality_str = json.dumps(personality_data, indent=2)
            diagnosis_str = json.dumps(diagnosis_data, indent=2)
            emotion_str = json.dumps(emotion_data, indent=2) if emotion_data else "{}"
            previous_str = json.dumps(previous_assessments, indent=2)
            
            # Generate the integrated assessment using LLM
            llm_response = await self.model.agenerate_messages([
                self.integration_prompt.format_messages(
                    personality_data=personality_str,
                    diagnosis_data=diagnosis_str,
                    emotion_data=emotion_str,
                    previous_assessments=previous_str
                )[0]
            ])
            
            # Parse the response
            response_text = llm_response.generations[0][0].text
            parsed_report = self._parse_integrated_report(response_text)
            
            # Get the enhanced insights if RAG system is available
            if self.rag_system:
                enhanced_insights = await self.rag_system.enhance_personality_assessment(
                    assessment_data=personality_data,
                    emotional_context=emotion_data
                )
                
                # Add enhanced insights to the report
                if enhanced_insights.get("success", False):
                    parsed_report["enhanced_insights"] = {
                        "personality_analysis": enhanced_insights.get("analysis", ""),
                        "strengths": enhanced_insights.get("strengths", ""),
                        "growth_areas": enhanced_insights.get("growth_areas", ""),
                        "communication_style": enhanced_insights.get("communication_style", ""),
                        "learning_style": enhanced_insights.get("learning_style", ""),
                        "stress_response": enhanced_insights.get("stress_response", "")
                    }
            
            # Add metadata
            parsed_report["timestamp"] = datetime.now().isoformat()
            parsed_report["success"] = True
            
            # Store in memory
            try:
                await self.memory.add("last_integrated_assessment", parsed_report)
            except Exception as memory_error:
                logger.warning(f"Failed to update memory: {str(memory_error)}")
            
            return parsed_report
            
        except Exception as e:
            logger.error(f"Error generating integrated report: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _parse_integrated_report(self, text: str) -> Dict[str, Any]:
        """Parse the integrated report into structured format"""
        result = {
            "personality_profile": "",
            "mental_health_overview": "",
            "emotional_patterns": "",
            "integrated_insights": "",
            "key_strengths": [],
            "growth_opportunities": [],
            "personalized_recommendations": [],
            "next_steps": []
        }
        
        try:
            current_section = None
            
            for line in text.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith("Personality Profile:"):
                    current_section = "personality_profile"
                    line = line[len("Personality Profile:"):].strip()
                elif line.startswith("Mental Health Overview:"):
                    current_section = "mental_health_overview"
                    line = line[len("Mental Health Overview:"):].strip()
                elif line.startswith("Emotional Patterns:"):
                    current_section = "emotional_patterns"
                    line = line[len("Emotional Patterns:"):].strip()
                elif line.startswith("Integrated Insights:"):
                    current_section = "integrated_insights"
                    line = line[len("Integrated Insights:"):].strip()
                elif line.startswith("Key Strengths:"):
                    current_section = "key_strengths"
                    line = line[len("Key Strengths:"):].strip()
                elif line.startswith("Growth Opportunities:"):
                    current_section = "growth_opportunities"
                    line = line[len("Growth Opportunities:"):].strip()
                elif line.startswith("Personalized Recommendations:"):
                    current_section = "personalized_recommendations"
                    line = line[len("Personalized Recommendations:"):].strip()
                elif line.startswith("Next Steps:"):
                    current_section = "next_steps"
                    line = line[len("Next Steps:"):].strip()
                
                # Add content to the current section
                if current_section in ["personality_profile", "mental_health_overview", "emotional_patterns", "integrated_insights"]:
                    if line:
                        result[current_section] += line + " "
                elif current_section in ["key_strengths", "growth_opportunities", "personalized_recommendations", "next_steps"]:
                    if line:
                        if line.startswith("- "):
                            result[current_section].append(line[2:])
                        elif "," in line:
                            result[current_section].extend([item.strip() for item in line.split(",")])
                        else:
                            result[current_section].append(line)
            
            return result
            
        except Exception as parse_error:
            logger.error(f"Error parsing integrated report: {str(parse_error)}")
            return {
                "error": str(parse_error),
                "raw_text": text
            }
    
    async def _get_previous_assessments(self) -> Dict[str, Any]:
        """Retrieve previous assessment data from memory"""
        try:
            return await self.memory.get("last_integrated_assessment", {})
        except Exception as e:
            logger.warning(f"Failed to retrieve previous assessments: {str(e)}")
            return {}
    
    async def save_results(self, results: Dict[str, Any]) -> bool:
        """Save assessment results to database or file system"""
        try:
            # Create a results manager
            results_manager = DiagnosisResults()
            
            # Save the results
            saved = await results_manager.save_results(results)
            
            # Also store in memory
            await self.memory.add("last_integrated_assessment", results)
            
            return saved
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
    
    async def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.rag_system:
            return await self.rag_system.query_knowledge_base(query, top_k)
        return []