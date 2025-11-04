from typing import Dict, Any, Optional
from ..base.base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.memory import ConversationBufferMemory
import logging
from datetime import datetime

# Import vector database integration
from src.utils.vector_db_integration import add_user_data

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize logger
logger = logging.getLogger(__name__)

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

@tool(name="voice_emotion_analysis", description="Analyzes emotional content from voice data")
async def analyze_voice_emotion(emotion_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes voice emotion analysis data
    
    Args:
        emotion_data: Dictionary containing voice emotion analysis results
        
    Returns:
        Dictionary with processed emotion data
    """
    try:
        if not emotion_data or not emotion_data.get("success", False):
            return {
                "success": False,
                "primary_emotion": "unknown",
                "confidence": 0.0,
                "emotions": {}
            }
        
        return {
            "success": True,
            "primary_emotion": emotion_data.get("primary_emotion", "neutral"),
            "confidence": emotion_data.get("confidence", 0.0),
            "emotions": emotion_data.get("emotions", {})
        }
    except Exception as e:
        logger.error(f"Error processing voice emotion data: {str(e)}")
        return {
            "success": False,
            "primary_emotion": "unknown",
            "confidence": 0.0,
            "emotions": {},
            "error": str(e)
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
            tools=[analyze_emotion, analyze_voice_emotion],
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
6. Voice emotion data when available

Use the following emotion categories:
- Basic: sad, anxious, angry, happy, neutral
- Complex: overwhelmed, hopeless, frustrated, grateful, confused
- Clinical: depressed, manic, dissociative, paranoid

Provide detailed analysis while maintaining clinical accuracy."""),
            HumanMessage(content="""Message: {message}
Previous Emotion State: {history}
Text Sentiment Analysis: {sentiment}
Voice Emotion Analysis: {voice_emotion}

Analyze the emotional content and provide structured output in the following format:
Primary Emotion: [emotion]
Secondary Emotions: [comma-separated list]
Intensity (1-10): [number]
Triggers: [comma-separated list]
Clinical Indicators: [relevant observations]
Pattern Changes: [changes from previous state]
Congruence: [match between voice tone and text content]""")
        ])

    async def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotional content of a message
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing emotional analysis
        """
        try:
            # Instead of using the analyze_emotion function directly, implement a simpler approach
            # to avoid the 'Function' object is not callable error
            sentiment_result = {
                'sentiment_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
                'compound_score': 0.0,
                'normalized_intensity': 0.0
            }
            
            # Use VADER sentiment analyzer directly
            try:
                sentiment = sentiment_analyzer.polarity_scores(text)
                sentiment_result = {
                    'sentiment_scores': sentiment,
                    'compound_score': sentiment['compound'],
                    'normalized_intensity': abs(sentiment['compound']) * 10
                }
            except Exception as e:
                logger.warning(f"VADER sentiment analysis failed: {str(e)}")
            
            # Get history - handle potential memory errors
            history = {}
            try:
                history = await self.memory.get("last_analysis", {})
            except Exception as e:
                logger.warning(f"Failed to get memory: {str(e)}")
            
            # Create a simplified analysis based on sentiment
            analysis = {
                'primary_emotion': 'neutral',
                'secondary_emotions': [],
                'intensity': 5,
                'triggers': [],
                'clinical_indicators': [],
                'pattern_changes': [],
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine primary emotion based on sentiment
            compound = sentiment_result.get('compound_score', 0)
            if compound > 0.05:
                analysis['primary_emotion'] = 'happy'
                analysis['secondary_emotions'] = ['content', 'satisfied']
            elif compound < -0.05:
                analysis['primary_emotion'] = 'sad'
                analysis['secondary_emotions'] = ['disappointed', 'frustrated']
            else:
                analysis['primary_emotion'] = 'neutral'
                analysis['secondary_emotions'] = ['calm', 'balanced']
            
            # Set intensity based on sentiment
            analysis['intensity'] = min(10, max(1, int(abs(compound) * 10)))
            
            # Add some basic triggers based on common words
            text_lower = text.lower()
            if 'work' in text_lower or 'job' in text_lower:
                analysis['triggers'].append('work-related stress')
            if 'family' in text_lower or 'parent' in text_lower:
                analysis['triggers'].append('family dynamics')
            if 'health' in text_lower or 'sick' in text_lower:
                analysis['triggers'].append('health concerns')
            
            # Add some basic clinical indicators
            if 'depressed' in text_lower or 'hopeless' in text_lower:
                analysis['clinical_indicators'].append('depression symptoms')
            if 'anxious' in text_lower or 'worried' in text_lower:
                analysis['clinical_indicators'].append('anxiety symptoms')
            if 'angry' in text_lower or 'frustrated' in text_lower:
                analysis['clinical_indicators'].append('emotional dysregulation')
            
            # Try to update memory, but don't fail if it doesn't work
            try:
                await self.memory.add("last_analysis", analysis)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {str(e)}")
            return self._fallback_analysis(text)

    async def analyze_with_voice_and_text(self, text: str, voice_emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze both text content and voice emotion data for comprehensive emotion analysis
        
        Args:
            text: The text to analyze
            voice_emotion_data: Optional voice emotion analysis results
            
        Returns:
            Dictionary containing combined emotional analysis
        """
        try:
            # Get text sentiment analysis
            sentiment_result = {
                'sentiment_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
                'compound_score': 0.0,
                'normalized_intensity': 0.0
            }
            
            # Use VADER sentiment analyzer for text
            try:
                sentiment = sentiment_analyzer.polarity_scores(text)
                sentiment_result = {
                    'sentiment_scores': sentiment,
                    'compound_score': sentiment['compound'],
                    'normalized_intensity': abs(sentiment['compound']) * 10
                }
            except Exception as e:
                logger.warning(f"VADER sentiment analysis failed: {str(e)}")
            
            # Get history
            history = {}
            try:
                history = await self.memory.get("last_analysis", {})
            except Exception as e:
                logger.warning(f"Failed to get memory: {str(e)}")
            
            # Create base analysis from text
            text_analysis = self._create_text_analysis(text, sentiment_result)
            
            # If voice emotion data is available, integrate it
            if (voice_emotion_data and voice_emotion_data.get("success", False)):
                combined_analysis = self._integrate_voice_and_text(text_analysis, voice_emotion_data)
            else:
                combined_analysis = text_analysis
            
            # Add voice emotion data to the analysis for reference
            if voice_emotion_data:
                combined_analysis['voice_emotion'] = voice_emotion_data
            
            # Add timestamp
            combined_analysis['timestamp'] = datetime.now().isoformat()
            
            # Check for emotion congruence between voice and text
            combined_analysis['congruence'] = self._check_emotion_congruence(
                text_emotion=text_analysis.get('primary_emotion', 'neutral'),
                voice_emotion=voice_emotion_data.get('primary_emotion', 'neutral') if voice_emotion_data else 'unknown'
            )
            
            # Try to update memory with the new analysis
            try:
                await self.memory.add("last_analysis", combined_analysis)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error in combined emotion analysis: {str(e)}")
            return self._fallback_analysis(text)
    
    def _create_text_analysis(self, text: str, sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic emotion analysis from text content"""
        # Base analysis structure
        analysis = {
            'primary_emotion': 'neutral',
            'secondary_emotions': [],
            'intensity': 5,
            'triggers': [],
            'clinical_indicators': [],
            'pattern_changes': [],
            'confidence': 0.7
        }
        
        # Determine primary emotion based on sentiment
        compound = sentiment_result.get('compound_score', 0)
        if compound > 0.05:
            analysis['primary_emotion'] = 'happy'
            analysis['secondary_emotions'] = ['content', 'satisfied']
        elif compound < -0.05:
            analysis['primary_emotion'] = 'sad'
            analysis['secondary_emotions'] = ['disappointed', 'frustrated']
        else:
            analysis['primary_emotion'] = 'neutral'
            analysis['secondary_emotions'] = ['calm', 'balanced']
        
        # Set intensity based on sentiment
        analysis['intensity'] = min(10, max(1, int(abs(compound) * 10)))
        
        # Add basic triggers based on common words
        text_lower = text.lower()
        if 'work' in text_lower or 'job' in text_lower:
            analysis['triggers'].append('work-related stress')
        if 'family' in text_lower or 'parent' in text_lower:
            analysis['triggers'].append('family dynamics')
        if 'health' in text_lower or 'sick' in text_lower:
            analysis['triggers'].append('health concerns')
        
        # Add basic clinical indicators
        if 'depressed' in text_lower or 'hopeless' in text_lower:
            analysis['clinical_indicators'].append('depression symptoms')
        if 'anxious' in text_lower or 'worried' in text_lower:
            analysis['clinical_indicators'].append('anxiety symptoms')
        if 'angry' in text_lower or 'frustrated' in text_lower:
            analysis['clinical_indicators'].append('emotional dysregulation')
        
        return analysis
    
    def _integrate_voice_and_text(self, text_analysis: Dict[str, Any], voice_emotion: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate voice emotion data with text analysis"""
        # Start with the text analysis
        combined = text_analysis.copy()
        
        # Get voice emotion primary emotion and confidence
        voice_primary = voice_emotion.get('primary_emotion', 'neutral')
        voice_confidence = voice_emotion.get('confidence', 0.0)
        voice_source = voice_emotion.get('source', 'unknown')
        
        # Keep track of the voice_source for reference
        combined['voice_source'] = voice_source
        
        # If voice emotion has high confidence, adjust the primary emotion
        if voice_confidence > 0.6:
            # Keep track of the text-based emotion
            combined['text_primary_emotion'] = combined['primary_emotion']
            
            # Weight voice emotion higher for emotional expression
            combined['primary_emotion'] = voice_primary
            
            # Add voice emotion to secondary emotions if not already there
            if voice_primary not in combined['secondary_emotions']:
                combined['secondary_emotions'].insert(0, voice_primary)
        else:
            # Lower confidence in voice emotion, add as secondary if strong enough
            if voice_confidence > 0.4 and voice_primary not in combined['secondary_emotions']:
                combined['secondary_emotions'].append(voice_primary)
        
        # Adjust intensity based on voice emotion intensity
        voice_emotions = voice_emotion.get('emotions', {})
        if voice_emotions:
            # Get the highest emotion score as a proxy for intensity
            max_score = max(voice_emotions.values()) if voice_emotions else 0
            
            # Blend text and voice intensity, weight voice emotion more heavily (60/40 split)
            voice_intensity = int(max_score * 10)
            combined['intensity'] = int((combined['intensity'] * 0.4) + (voice_intensity * 0.6))
        
        # Add voice emotion confidence to the analysis
        combined['voice_confidence'] = voice_confidence
        
        # Increase overall confidence when voice and text align
        if voice_primary == combined.get('text_primary_emotion', combined['primary_emotion']):
            combined['confidence'] = min(1.0, combined['confidence'] + 0.2)
        
        # Add detailed voice emotion analysis
        combined['voice_emotions_detailed'] = voice_emotions
        
        # Extract top 3 voice emotions for quick reference
        if voice_emotions:
            top_voice_emotions = sorted(voice_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            combined['top_voice_emotions'] = {emotion: score for emotion, score in top_voice_emotions}
        
        # Add special note if voice emotion is significantly different from text emotion
        if voice_primary != combined.get('text_primary_emotion', combined['primary_emotion']):
            if 'clinical_indicators' not in combined:
                combined['clinical_indicators'] = []
            combined['clinical_indicators'].append(f"voice-text emotion mismatch ({voice_primary} vs {combined.get('text_primary_emotion', combined['primary_emotion'])})")
        
        return combined
    
    def _check_emotion_congruence(self, text_emotion: str, voice_emotion: str) -> str:
        """
        Check if emotions expressed in text and voice are congruent
        
        Returns:
            String describing congruence: "high", "medium", "low", or "unknown"
        """
        if voice_emotion == "unknown":
            return "unknown"
            
        # Map emotions to higher-level categories
        positive_emotions = ["happy", "excitement", "joy", "content", "satisfaction"]
        negative_emotions = ["sad", "anger", "fear", "disgust", "anxiety", "frustration"]
        neutral_emotions = ["neutral", "calm", "surprise"]
        
        # Determine emotional valence categories
        text_valence = "positive" if text_emotion in positive_emotions else "negative" if text_emotion in negative_emotions else "neutral"
        voice_valence = "positive" if voice_emotion in positive_emotions else "negative" if voice_emotion in negative_emotions else "neutral"
        
        # Check for exact match
        if text_emotion == voice_emotion:
            return "high"
            
        # Check for same valence category
        elif text_valence == voice_valence:
            return "medium"
            
        # Different valence categories
        else:
            return "low"

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
    
    async def store_to_vector_db(self, query: str, response: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Store emotion analysis data in the central vector database
        
        Args:
            query: User's query
            response: Agent's response
            context: Processing context
        """
        try:
            # Check if this contains emotion analysis data
            if isinstance(response, dict) and 'emotion_analysis' in response:
                emotion_data = response['emotion_analysis']
                
                # Add metadata
                emotion_data["timestamp"] = datetime.now().isoformat()
                emotion_data["user_message"] = query
                
                # Add user ID if available in context
                if context and "user_id" in context:
                    emotion_data["user_id"] = context["user_id"]
                
                # Store in vector database
                doc_id = add_user_data("emotion", emotion_data)
                
                if doc_id:
                    logger.info(f"Stored emotion data in vector DB: {doc_id}")
                else:
                    logger.warning("Failed to store emotion data in vector DB")
            
        except Exception as e:
            logger.error(f"Error storing emotion data in vector DB: {str(e)}")
            
    def generate_response_sync(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response synchronously
        
        Args:
            query: User's query
            context: Optional processing context
            
        Returns:
            Dictionary containing the response
        """
        # Placeholder for synchronous response generation logic
        return {}