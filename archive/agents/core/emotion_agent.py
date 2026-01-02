from typing import Dict, Any, Optional, List
from ..base.base_agent import BaseAgent
from agno.tools import tool
from agno.memory import Memory
from agno.knowledge import AgentKnowledge
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.language_model import BaseLanguageModel
import logging
from datetime import datetime

# Import vector database integration
from src.utils.vector_db_integration import add_user_data
# Import memory factory for centralized memory management
from src.utils.memory_factory import create_agent_memory
# Import sentiment analysis utilities
from src.utils.sentiment_utils import (
    analyze_text_sentiment,
    get_emotion_from_sentiment,
    detect_emotional_triggers
)

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
        # Create memory using centralized factory
        memory = create_agent_memory()

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
            # Use centralized sentiment analysis utility
            sentiment_result = analyze_text_sentiment(text)

            # Get history - handle potential memory errors
            history = {}
            try:
                history = await self.memory.get("last_analysis", {})
            except Exception as e:
                logger.warning(f"Failed to get memory: {str(e)}")

            # Create base analysis structure
            analysis = {
                'triggers': [],
                'clinical_indicators': [],
                'pattern_changes': [],
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }

            # Get emotion data from sentiment using utility
            compound = sentiment_result.get('compound_score', 0)
            emotion_data = get_emotion_from_sentiment(compound)
            analysis.update(emotion_data)

            # Detect emotional triggers using utility
            analysis['triggers'] = detect_emotional_triggers(text)

            # Add some basic clinical indicators
            text_lower = text.lower()
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
            # Use centralized sentiment analysis utility
            sentiment_result = analyze_text_sentiment(text)
            
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

        except Exception as e:
            logger.warning(f"Error parsing emotion analysis result: {str(e)}")
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
        """
        Enhanced fallback analysis with comprehensive sentiment and trigger detection.

        This fallback provides robust emotion analysis when the primary LLM-based
        analysis fails, using VADER sentiment analysis, trigger detection, and
        clinical indicator identification.

        Args:
            message: User's message text

        Returns:
            Dict with emotion analysis including primary/secondary emotions,
            intensity, triggers, clinical indicators, and confidence score
        """
        try:
            # Use centralized sentiment analysis utilities
            sentiment_result = analyze_text_sentiment(message)

            # Get emotion mapping from sentiment score
            emotion_data = get_emotion_from_sentiment(sentiment_result['compound_score'])

            # Detect emotional triggers
            triggers = detect_emotional_triggers(message)

            # Detect clinical indicators based on message content
            clinical_indicators = self._detect_clinical_indicators_fallback(message)

            # Calculate more sophisticated intensity
            intensity = emotion_data['intensity']

            # Add context-based secondary emotions
            secondary_emotions = emotion_data.get('secondary_emotions', [])

            # Enhance secondary emotions based on message analysis
            message_lower = message.lower()
            if 'anxious' in message_lower or 'worry' in message_lower or 'nervous' in message_lower:
                if 'anxious' not in secondary_emotions:
                    secondary_emotions.append('anxious')
            if 'stress' in message_lower or 'overwhelm' in message_lower:
                if 'stressed' not in secondary_emotions:
                    secondary_emotions.append('stressed')
            if 'angry' in message_lower or 'frustrated' in message_lower or 'mad' in message_lower:
                if 'angry' not in secondary_emotions:
                    secondary_emotions.append('angry')

            return {
                'primary_emotion': emotion_data['primary_emotion'],
                'secondary_emotions': secondary_emotions,
                'intensity': intensity,
                'triggers': triggers,
                'clinical_indicators': clinical_indicators,
                'pattern_changes': [],  # Cannot detect without history in fallback
                'confidence': 0.6,  # Moderate confidence for enhanced fallback
                'sentiment_scores': sentiment_result['sentiment_scores'],
                'analysis_method': 'fallback_vader_enhanced'
            }

        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}", exc_info=True)
            # Last resort minimal fallback
            return {
                'primary_emotion': 'neutral',
                'secondary_emotions': [],
                'intensity': 5,
                'triggers': [],
                'clinical_indicators': [],
                'pattern_changes': [],
                'confidence': 0.2,
                'error': str(e),
                'analysis_method': 'minimal_fallback'
            }

    def _detect_clinical_indicators_fallback(self, message: str) -> List[str]:
        """
        Detect clinical mental health indicators using keyword-based pattern matching.

        Performs rule-based detection of seven clinical indicator categories commonly
        associated with mental health conditions. This fallback method provides baseline
        screening when advanced NLP models are unavailable.

        Detection categories:
            1. depression_symptoms: Feelings of sadness, hopelessness, emptiness
            2. anxiety_symptoms: Worry, fear, panic, nervousness
            3. sleep_disturbance: Insomnia, fatigue, exhaustion
            4. mood_changes: Irritability, mood swings, anger
            5. social_withdrawal: Isolation, loneliness, avoidance
            6. concentration_issues: Focus problems, memory issues, distraction
            7. self_harm_ideation: Thoughts of self-harm or suicide (CRITICAL)

        Args:
            message (str): User's message text to analyze for clinical indicators

        Returns:
            List[str]: List of detected indicator categories (empty if none detected).
            Multiple indicators can be present in a single message.

        Example:
            >>> message = "I've been feeling hopeless and can't sleep at night"
            >>> indicators = self._detect_clinical_indicators_fallback(message)
            >>> print(indicators)
            ['depression_symptoms', 'sleep_disturbance']
            >>>
            >>> crisis_msg = "I feel worthless and want to hurt myself"
            >>> indicators = self._detect_clinical_indicators_fallback(crisis_msg)
            >>> print(indicators)
            ['depression_symptoms', 'self_harm_ideation']

        Note:
            - Detection is case-insensitive and uses keyword matching
            - self_harm_ideation triggers immediate escalation to SafetyAgent
            - This is a screening tool, not a diagnostic instrument
            - False positives/negatives are possible with keyword-based detection
        """
        indicators = []
        message_lower = message.lower()

        # Depression indicators
        if any(word in message_lower for word in ['depressed', 'hopeless', 'worthless', 'empty', 'numb']):
            indicators.append('depression_symptoms')

        # Anxiety indicators
        if any(word in message_lower for word in ['panic', 'anxious', 'worry', 'fear', 'nervous', 'scared']):
            indicators.append('anxiety_symptoms')

        # Sleep disturbances
        if any(word in message_lower for word in ['insomnia', 'sleep', 'tired', 'exhausted', 'fatigue']):
            indicators.append('sleep_disturbance')

        # Mood changes
        if any(word in message_lower for word in ['mood swing', 'irritable', 'angry', 'rage', 'upset']):
            indicators.append('mood_changes')

        # Social withdrawal
        if any(word in message_lower for word in ['alone', 'isolated', 'withdraw', 'avoid people', 'lonely']):
            indicators.append('social_withdrawal')

        # Concentration issues
        if any(word in message_lower for word in ['focus', 'concentrate', 'attention', 'distracted', 'memory']):
            indicators.append('concentration_issues')

        # Self-harm ideation (requires immediate attention)
        if any(word in message_lower for word in ['hurt myself', 'self-harm', 'cut', 'harm', 'suicide']):
            indicators.append('self_harm_ideation')

        return indicators
    
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