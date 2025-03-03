from typing import Optional, Dict
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import AppConfig
import anthropic
import httpx

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        # Remove proxies argument if present
        kwargs.pop("proxies", None)
        super().__init__(*args, **kwargs)

class ChatAgent:
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", use_cpu: bool = False):
        # Create a custom HTTP client for Anthropic
        http_client = CustomHTTPClient()
        
        # Initialize the ChatAnthropic model
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=AppConfig.ANTHROPIC_API_KEY,
            max_tokens=AppConfig.MAX_RESPONSE_TOKENS,
            temperature=0.7,
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a compassionate mental health counselor working as part of an AI agent team. 
Your role is to provide empathetic, evidence-based support while considering:
1. The user's emotional state and intensity
2. Any safety concerns
3. Relevant context from other agents
4. Treatment history and preferences

Guidelines:
- Respond with genuine empathy and validation
- Provide practical, evidence-based suggestions
- Maintain a conversational, non-clinical tone
- Prioritize user safety above all else
- Collaborate with other agents' insights
- Be transparent about AI limitations"""),
            HumanMessage(content="""Context: {context}
Emotional State: {emotion}
Safety Assessment: {safety}
Search Results: {search_results}
Diagnosis Info: {diagnosis}

User Query: {question}""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_response(self, 
                         context: str, 
                         question: str, 
                         emotion: Dict, 
                         safety: Dict,
                         search_results: str = "",
                         diagnosis: str = "") -> str:
        try:
            # Format emotional state
            emotion_str = f"{emotion.get('primary_emotion', 'neutral')} (intensity: {emotion.get('intensity', 5)})"
            if emotion.get('triggers'):
                emotion_str += f"\nTriggers: {', '.join(emotion.get('triggers'))}"
            
            # Format safety concerns
            safety_str = "Safety Concerns:\n"
            if not safety.get('safe', True):
                safety_str += "- " + safety.get('concerns', ['Potential risk detected'])[0]
                if safety.get('recommendations'):
                    safety_str += f"\nRecommendations: {safety.get('recommendations')}"
            else:
                safety_str += "No immediate safety concerns identified"

            response = self.chain.run({
                "context": context,
                "question": question,
                "emotion": emotion_str,
                "safety": safety_str,
                "search_results": search_results,
                "diagnosis": diagnosis
            })
            
            return response.strip()
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"