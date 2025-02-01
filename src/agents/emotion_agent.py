from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class EmotionAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.emotion_prompt = PromptTemplate(
            input_variables=["message"],
            template="""Analyze the emotional content of this message:
            Message: {message}
            
            Identify:
            1. Primary emotion
            2. Emotion intensity (1-10)
            3. Key emotional triggers or concerns
            
            Provide a structured analysis."""
        )
        self.emotion_chain = LLMChain(llm=self.llm, prompt=self.emotion_prompt)

    def analyze(self, message: str) -> dict:
        """
        Analyzes emotional content of message and returns structured analysis
        """
        result = self.emotion_chain.run(message=message)
        
        # Parse the result
        emotion_data = self._parse_emotion_data(result)
        
        return {
            'primary_emotion': emotion_data['primary_emotion'],
            'intensity': emotion_data['intensity'],
            'triggers': emotion_data['triggers'],
            'raw_analysis': result
        }

    def _parse_emotion_data(self, analysis: str) -> dict:
        try:
            lines = analysis.split('\n')
            
            # Extract primary emotion
            primary_emotion = next(
                (line.split(':')[1].strip() 
                 for line in lines 
                 if 'primary emotion' in line.lower()),
                'neutral'
            )
            
            # Extract intensity
            intensity_str = next(
                (line for line in lines if 'intensity' in line.lower()),
                '5'
            )
            intensity = int(''.join(filter(str.isdigit, intensity_str)) or 5)
            
            # Extract triggers
            triggers = [
                line.strip() 
                for line in lines 
                if 'trigger' in line.lower() or 'concern' in line.lower()
            ]
            
            return {
                'primary_emotion': primary_emotion,
                'intensity': intensity,
                'triggers': triggers
            }
        except Exception as e:
            return {
                'primary_emotion': 'neutral',
                'intensity': 5,
                'triggers': []
            }