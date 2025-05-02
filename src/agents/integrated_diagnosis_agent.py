"""
Integrated Diagnosis Agent that combines mental health assessment with personality testing.
"""

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from .diagnosis_agent import DiagnosisAgent
from .personality_agent import PersonalityAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import json
import os
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

@tool(name="integrated_assessment")
async def integrated_assessment(mental_health_responses: Dict[str, int], personality_responses: Dict[str, int]) -> Dict[str, Any]:
    """
    Conducts an integrated assessment combining mental health and personality data
    
    Args:
        mental_health_responses: Dictionary containing responses to mental health questions
        personality_responses: Dictionary containing responses to personality questions
        
    Returns:
        Dictionary containing integrated assessment results
    """
    try:
        # Load assessment questions
        questions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'data', 'personality', 'diagnosis_questions.json')
        
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)
            
        # Process mental health responses
        mental_health_scores = {
            "depression": 0,
            "anxiety": 0,
            "stress": 0,
            "sleep": 0,
            "cognitive": 0,
            "social": 0,
            "physical": 0,
            "suicidal": 0
        }
        
        total_score = 0
        question_count = 0
        
        for question in questions_data["mental_health"]:
            q_id = str(question["id"])
            if q_id in mental_health_responses:
                score = mental_health_responses[q_id]
                category = question["category"]
                if category in mental_health_scores:
                    mental_health_scores[category] += score
                total_score += score
                question_count += 1
        
        # Calculate average scores
        avg_score = total_score / max(question_count, 1)
        
        # Determine severity levels
        severity_levels = {}
        for category, score in mental_health_scores.items():
            # Normalize by number of questions in that category
            category_questions = [q for q in questions_data["mental_health"] if q["category"] == category]
            max_possible = len(category_questions) * 3  # Maximum score per question is 3
            
            if max_possible > 0:
                normalized_score = score / max_possible
                
                if normalized_score >= 0.67:
                    severity_levels[category] = "severe"
                elif normalized_score >= 0.33:
                    severity_levels[category] = "moderate"
                else:
                    severity_levels[category] = "mild"
        
        # Process personality responses
        personality_scores = {
            "extraversion": [],
            "agreeableness": [],
            "conscientiousness": [],
            "neuroticism": [],
            "openness": []
        }
        
        for question in questions_data["personality"]:
            q_id = str(question["id"])
            if q_id in personality_responses:
                score = personality_responses[q_id]
                trait = question["trait"]
                if trait in personality_scores:
                    personality_scores[trait].append(score)
        
        # Calculate average trait scores
        personality_traits = {}
        for trait, scores in personality_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                # Convert to percentile (1-5 scale to 0-100%)
                percentile = (avg - 1) / 4 * 100
                
                # Determine category
                if percentile < 30:
                    category = "low"
                elif percentile > 70:
                    category = "high"
                else:
                    category = "average"
                    
                personality_traits[trait] = {
                    "score": percentile,
                    "category": category
                }
        
        # Determine overall mental health status
        if avg_score >= 2:
            overall_status = "severe"
        elif avg_score >= 1:
            overall_status = "moderate"
        else:
            overall_status = "mild"
            
        # Check for crisis indicators
        crisis_indicators = []
        if mental_health_scores.get("suicidal", 0) > 0:
            crisis_indicators.append("suicidal ideation")
        
        # Combine results
        return {
            "mental_health": {
                "overall_status": overall_status,
                "scores": mental_health_scores,
                "severity_levels": severity_levels,
                "crisis_indicators": crisis_indicators
            },
            "personality": {
                "traits": personality_traits
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in integrated assessment: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

class IntegratedDiagnosisAgent(BaseAgent):
    """Agent that combines mental health diagnosis with personality assessment"""
    
    def __init__(self, model: BaseLanguageModel):
        super().__init__(
            model=model,
            name="integrated_diagnostician",
            role="Expert system for comprehensive mental health and personality assessment",
            description="""An AI agent specialized in analyzing mental health symptoms and personality traits to provide a comprehensive assessment.
            Uses evidence-based criteria and maintains clinical accuracy while emphasizing the importance of professional evaluation.""",
            tools=[integrated_assessment],
            memory=Memory(memory="integrated_diagnosis_memory", storage="local_storage"),
            knowledge=AgentKnowledge()
        )
        
        self.interpretation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in mental health and personality assessment.
Your role is to interpret assessment results and provide insights on:
1. Key mental health concerns and their severity
2. Personality traits and their implications for mental health
3. Personalized coping strategies based on the combined profile
4. Recommendations for support and treatment
5. Compassionate framing that emphasizes hope and recovery

Guidelines:
- Be warm and empathetic
- Validate feelings without reinforcing negative patterns
- Suggest practical coping strategies tailored to personality traits
- Encourage professional help when appropriate
- Maintain a supportive, non-judgmental tone
- Emphasize strengths and resilience factors"""),
            HumanMessage(content="""Assessment Results: {assessment_results}

Provide a compassionate interpretation of these results, focusing on:
- Key insights about the mental health and personality profile
- How personality traits may influence mental health challenges
- Personalized coping strategies based on the combined profile
- Supportive and hopeful framing that acknowledges challenges while emphasizing potential for improvement
- Clear next steps and recommendations""")
        ])
    
    async def conduct_assessment(self, mental_health_responses: Dict[str, int], personality_responses: Dict[str, int]) -> Dict[str, Any]:
        """
        Conduct an integrated assessment and provide interpretation
        
        Args:
            mental_health_responses: Dictionary containing responses to mental health questions
            personality_responses: Dictionary containing responses to personality questions
            
        Returns:
            Dictionary containing assessment results and interpretation
        """
        try:
            # Get assessment results
            results = await integrated_assessment(mental_health_responses, personality_responses)
            
            # Store results in memory
            try:
                await self.memory.add("assessment_results", results)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            # Generate interpretation
            interpretation = await self._generate_interpretation(results)
            
            # Combine results and interpretation
            return {
                "assessment_results": results,
                "interpretation": interpretation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in integrated assessment: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def _generate_interpretation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an interpretation of assessment results"""
        try:
            # Generate interpretation using LLM
            llm_response = await self.model.agenerate_messages([
                self.interpretation_prompt.format_messages(
                    assessment_results=json.dumps(results, indent=2)
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
            "coping_strategies": [],
            "recommendations": [],
            "supportive_framing": [],
            "next_steps": []
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
                elif "cop" in lower_line or "strateg" in lower_line:
                    current_section = "coping_strategies"
                elif "recommend" in lower_line:
                    current_section = "recommendations"
                elif "support" in lower_line or "hope" in lower_line or "frame" in lower_line:
                    current_section = "supportive_framing"
                elif "next" in lower_line or "step" in lower_line:
                    current_section = "next_steps"
                elif current_section and line.startswith('-'):
                    # Add bullet points to current section
                    sections[current_section].append(line[1:].strip())
                elif current_section and not any(header in lower_line for header in ["key", "cop", "recommend", "support", "next"]):
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
    
    def generate_empathy_response(self, assessment_results: Dict[str, Any]) -> str:
        """
        Generate an empathetic response based on assessment results
        
        Args:
            assessment_results: Dictionary containing assessment results
            
        Returns:
            String containing empathetic response
        """
        try:
            mental_health = assessment_results.get("mental_health", {})
            personality = assessment_results.get("personality", {})
            
            overall_status = mental_health.get("overall_status", "mild")
            severity_levels = mental_health.get("severity_levels", {})
            personality_traits = personality.get("traits", {})
            
            # Base empathy statements by severity
            empathy_statements = {
                "severe": [
                    "I can see you're going through a really difficult time right now. That takes incredible courage to share.",
                    "What you're experiencing sounds truly challenging. Please know that you're not alone in this struggle.",
                    "I'm truly sorry to hear how much you've been suffering. Your resilience in reaching out is remarkable."
                ],
                "moderate": [
                    "It sounds like you've been dealing with some significant challenges lately. Thank you for sharing that with me.",
                    "I can hear that things have been quite difficult for you. Your awareness of these feelings is an important step.",
                    "What you're going through matters, and it takes strength to acknowledge these feelings."
                ],
                "mild": [
                    "I appreciate you sharing how you've been feeling. Even mild symptoms deserve attention and care.",
                    "Thank you for opening up about your experiences. Being proactive about your mental health is really commendable.",
                    "It's important to address these feelings, even when they might seem manageable. I'm glad you're reaching out."
                ]
            }
            
            # Select base empathy statement
            import random
            base_statement = random.choice(empathy_statements.get(overall_status, empathy_statements["mild"]))
            
            # Add personality-specific insights
            personality_insights = ""
            if "extraversion" in personality_traits:
                if personality_traits["extraversion"]["category"] == "high":
                    personality_insights += " Your outgoing nature can be a strength in seeking support from others."
                else:
                    personality_insights += " Your more reflective nature can help you process your experiences deeply."
                    
            if "neuroticism" in personality_traits:
                if personality_traits["neuroticism"]["category"] == "high":
                    personality_insights += " It's understandable that you might feel emotions intensely, which can be challenging but also gives you depth of understanding."
                    
            if "conscientiousness" in personality_traits:
                if personality_traits["conscientiousness"]["category"] == "high":
                    personality_insights += " Your organized approach to life can be helpful in developing consistent self-care routines."
            
            # Add hope statement
            hope_statement = "While I understand these challenges feel significant right now, with the right support and strategies, things can improve. You've already taken an important step by seeking help."
            
            # Combine all elements
            full_response = f"{base_statement}{personality_insights}\n\n{hope_statement}"
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating empathy response: {str(e)}")
            return "I can see you're going through a difficult time. Thank you for sharing your experiences with me. With support and the right strategies, things can improve, and you've already taken an important step by reaching out."
    
    def generate_immediate_actions(self, assessment_results: Dict[str, Any]) -> List[str]:
        """
        Generate immediate actions based on assessment results
        
        Args:
            assessment_results: Dictionary containing assessment results
            
        Returns:
            List of immediate actions
        """
        try:
            mental_health = assessment_results.get("mental_health", {})
            personality = assessment_results.get("personality", {})
            
            overall_status = mental_health.get("overall_status", "mild")
            severity_levels = mental_health.get("severity_levels", {})
            crisis_indicators = mental_health.get("crisis_indicators", [])
            personality_traits = personality.get("traits", {})
            
            actions = []
            
            # Crisis actions take precedence
            if crisis_indicators:
                actions.append("Consider contacting a crisis helpline immediately (988 in the US)")
                actions.append("Reach out to a trusted person who can provide immediate support")
                actions.append("Schedule an urgent appointment with a mental health professional")
                return actions
            
            # General actions based on severity
            if overall_status == "severe":
                actions.append("Schedule an appointment with a mental health professional as soon as possible")
                actions.append("Consider talking to your primary care doctor about your symptoms")
                actions.append("Establish a daily check-in with a trusted friend or family member")
            elif overall_status == "moderate":
                actions.append("Consider scheduling an appointment with a mental health professional")
                actions.append("Establish a consistent self-care routine that includes rest, nutrition, and movement")
                actions.append("Identify one supportive person you can talk to about how you're feeling")
            else:
                actions.append("Consider adding regular self-care practices to your routine")
                actions.append("Monitor your symptoms and reach out for professional help if they worsen")
                actions.append("Explore resources like books, apps, or online communities related to mental wellness")
            
            # Personality-specific actions
            if "extraversion" in personality_traits:
                if personality_traits["extraversion"]["category"] == "high":
                    actions.append("Schedule social activities that energize you while being mindful of not overextending yourself")
                else:
                    actions.append("Balance social interaction with adequate alone time for recharging")
            
            if "openness" in personality_traits:
                if personality_traits["openness"]["category"] == "high":
                    actions.append("Explore creative outlets like journaling, art, or music as emotional expression")
                else:
                    actions.append("Establish comfortable routines that provide stability while gradually introducing new coping strategies")
            
            if "conscientiousness" in personality_traits:
                if personality_traits["conscientiousness"]["category"] == "high":
                    actions.append("Create a structured self-care plan with specific goals and checkpoints")
                else:
                    actions.append("Start with small, manageable self-care steps that don't feel overwhelming")
            
            # Add breathing exercise for everyone
            actions.append("Practice a simple breathing exercise: breathe in for 4 counts, hold for 2, exhale for 6 counts")
            
            return actions[:5]  # Limit to top 5 most relevant actions
            
        except Exception as e:
            logger.error(f"Error generating immediate actions: {str(e)}")
            return [
                "Consider reaching out to a mental health professional",
                "Practice basic self-care: adequate sleep, nutrition, and movement",
                "Connect with supportive people in your life",
                "Try a simple breathing exercise when feeling overwhelmed",
                "Be compassionate with yourself during this challenging time"
            ]
