from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

@tool(name="emotion_analysis", description="Analyzes emotional content in text using VADER sentiment analysis")
async def analyze_emotion(text: str) -> Dict[str, Any]:
    """
    Analyzes emotional content in text using VADER sentiment analysis
    
    Args:
        text: The text to analyze for emotional content
        
    Returns:
        Dictionary containing sentiment scores and analysis
    """
    try:
        sentiment = sentiment_analyzer.polarity_scores(text)
        return {
            'sentiment_scores': sentiment,
            'compound_score': sentiment['compound'],
            'normalized_intensity': abs(sentiment['compound']) * 10
        }
    except Exception as e:
        return {
            'sentiment_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
            'compound_score': 0.0,
            'normalized_intensity': 0.0,
            'error': str(e)
        }

class EmotionAgent(BaseAgent):
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
            name="emotion_analyzer",
            description="Expert system for emotional analysis and mental health assessment",
            tools=[analyze_emotion],
            memory=memory,
            knowledge=AgentKnowledge()
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert in emotional analysis and mental health assessment.
Your role is to analyze messages for emotional content, considering:
1. Primary and secondary emotions
2. Emotional intensity and volatility
3. Underlying psychological triggers
4. Potential mental health indicators
5. Changes in emotional patterns over time

Use the following emotion categories:
- Basic: sad, anxious, angry, happy, neutral
- Complex: overwhelmed, hopeless, frustrated, grateful, confused
- Clinical: depressed, manic, dissociative, paranoid

Provide detailed analysis while maintaining clinical accuracy."""),
            HumanMessage(content="""Message: {message}
Previous Emotion State: {history}
Sentiment Analysis: {sentiment}

Analyze the emotional content and provide structured output in the following format:
Primary Emotion: [emotion]
Secondary Emotions: [comma-separated list]
Intensity (1-10): [number]
Triggers: [comma-separated list]
Clinical Indicators: [relevant observations]
Pattern Changes: [changes from previous state]""")
        ])

    async def _generate_response(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            # Get sentiment analysis
            sentiment_result = await analyze_emotion(input_data.get('text', ''))
            
            # Get message and history
            message = input_data.get('text', '')
            history = context.get('memory', {}).get('last_analysis', {})
            
            # Generate LLM analysis
            llm_response = await self.llm.agenerate_messages([
                self.prompt_template.format_messages(
                    message=message,
                    history=self._format_history(history),
                    sentiment=sentiment_result
                )[0]
            ])
            
            # Parse response
            parsed = self._parse_result(llm_response.generations[0][0].text)
            
            # Add metadata
            parsed['confidence'] = self._calculate_confidence(parsed, sentiment_result)
            parsed['timestamp'] = input_data.get('timestamp')
            
            return parsed
            
        except Exception as e:
            return self._fallback_analysis(input_data.get('text', ''))

    def _parse_result(self, text: str) -> Dict[str, Any]:
        """Parse the structured output from Claude"""
        result = {
            'primary_emotion': 'neutral',
            'secondary_emotions': [],
            'intensity': 5,
            'triggers': [],
            'clinical_indicators': [],
            'pattern_changes': []
        }
        
        try:
            lines = text.strip().split('\n')
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [x.strip() for x in line.split(':', 1)]
                
                if 'Primary Emotion' in key:
                    result['primary_emotion'] = value.lower()
                elif 'Secondary Emotions' in key:
                    result['secondary_emotions'] = [e.strip().lower() for e in value.split(',')]
                elif 'Intensity' in key:
                    result['intensity'] = int(value.split()[0])
                elif 'Triggers' in key:
                    result['triggers'] = [t.strip() for t in value.split(',')]
                elif 'Clinical Indicators' in key:
                    result['clinical_indicators'] = [i.strip() for i in value.split(',')]
                elif 'Pattern Changes' in key:
                    result['pattern_changes'] = [p.strip() for p in value.split(',')]
                    
        except Exception:
            pass
            
        return result

    def _calculate_confidence(self, analysis: Dict[str, Any], sentiment: Dict[str, Any]) -> float:
        """Calculate confidence score in the analysis"""
        confidence = 1.0
        
        # Lower confidence if analysis is incomplete
        if not analysis['triggers'] or not analysis['secondary_emotions']:
            confidence *= 0.8
            
        # Check sentiment alignment
        compound_score = sentiment.get('compound_score', 0)
        if (compound_score > 0.5 and analysis['primary_emotion'] in ['sad', 'angry', 'anxious']) or \
           (compound_score < -0.5 and analysis['primary_emotion'] in ['happy', 'grateful']):
            confidence *= 0.6
            
        return confidence

    def _format_history(self, history: Dict[str, Any]) -> str:
        """Format historical emotional context"""
        if not history:
            return "No previous emotional context available"
            
        return f"""Previous State:
- Emotion: {history.get('primary_emotion', 'unknown')}
- Intensity: {history.get('intensity', 'unknown')}
- Notable Patterns: {', '.join(history.get('pattern_changes', []))}"""

    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """Enhanced fallback analysis using VADER"""
        sentiment = sentiment_analyzer.polarity_scores(message)
        
        # Map compound score to emotion
        if sentiment['compound'] >= 0.5:
            emotion = 'happy'
        elif sentiment['compound'] <= -0.5:
            emotion = 'sad'
        elif sentiment['neu'] >= 0.8:
            emotion = 'neutral'
        else:
            emotion = 'mixed'
            
        return {
            'primary_emotion': emotion,
            'secondary_emotions': [],
            'intensity': int(abs(sentiment['compound']) * 10),
            'triggers': [],
            'clinical_indicators': [],
            'pattern_changes': [],
            'confidence': 0.4  # Low confidence for fallback
        }