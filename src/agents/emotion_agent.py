from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class EmotionAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.emotion_prompt = PromptTemplate(
            input_variables=["message"],
            template="""Analyze emotional content:
            Message: {message}
            
            Identify:
            1. Primary emotion (choose from: sad, anxious, angry, happy, neutral)
            2. Intensity (1-10)
            3. Key emotional triggers
            
            Format response as:
            Emotion: <emotion>
            Intensity: <number>
            Triggers: <comma-separated list>"""
        )
        self.emotion_chain = LLMChain(llm=self.llm, prompt=self.emotion_prompt)

    def analyze(self, message: str) -> Dict:
        """Hybrid analysis with fallback to sentiment analysis"""
        try:
            llm_result = self.emotion_chain.run(message=message)
            parsed = self._parse_result(llm_result)
            
            # Validate with sentiment analysis
            sentiment = self.sentiment_analyzer.polarity_scores(message)
            return self._combine_results(parsed, sentiment)
        except Exception:
            return self._fallback_analysis(message)

    def _parse_result(self, text: str) -> Dict:
        emotion = 'neutral'
        intensity = 5
        triggers = []
        
        try:
            emotion_line = next(line for line in text.split('\n') if 'Emotion:' in line)
            emotion = re.search(r'Emotion:\s*(.+)', emotion_line).group(1).lower()
            
            intensity_line = next(line for line in text.split('\n') if 'Intensity:' in line)
            intensity = int(re.search(r'\d+', intensity_line).group())
            
            triggers_line = next(line for line in text.split('\n') if 'Triggers:' in line)
            triggers = [t.strip() for t in triggers_line.split(':')[1].split(',')]
        except Exception:
            pass
            
        return {
            'primary_emotion': emotion,
            'intensity': max(1, min(10, intensity)),
            'triggers': triggers[:3]
        }

    def _combine_results(self, parsed: Dict, sentiment: Dict) -> Dict:
        """Combine LLM analysis with sentiment scores"""
        sentiment_map = {
            'pos': 'happy',
            'neg': 'sad',
            'neu': 'neutral'
        }
        dominant_sentiment = max(sentiment, key=lambda k: sentiment[k] if k != 'compound' else 0)
        
        if parsed['intensity'] < 3:  # Low confidence in LLM result
            parsed['primary_emotion'] = sentiment_map.get(dominant_sentiment, 'neutral')
            parsed['intensity'] = int(abs(sentiment['compound']) * 10)
            
        return parsed

    def _fallback_analysis(self, message: str) -> Dict:
        """Fallback to sentiment analysis"""
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        return {
            'primary_emotion': 'positive' if sentiment['compound'] >= 0 else 'negative',
            'intensity': int(abs(sentiment['compound']) * 10),
            'triggers': []
        }