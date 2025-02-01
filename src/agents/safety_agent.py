# safety_agent.py
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class SafetyAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.safety_prompt = PromptTemplate(
            input_variables=["message"],
            template="""Analyze the following message for signs of crisis or immediate danger:
            Message: {message}
            
            Determine:
            1. Is there any indication of immediate self-harm or suicide risk?
            2. Is there any indication of harm to others?
            3. What is the severity level (1-10)?
            
            Provide a structured assessment."""
        )
        self.safety_chain = LLMChain(llm=self.llm, prompt=self.safety_prompt)

    def check_message(self, message: str) -> dict:
        """
        Analyzes message for safety concerns and returns structured assessment
        """
        result = self.safety_chain.run(message=message)
        
        # Parse the result and extract key information
        severity = self._extract_severity(result)
        concerns = self._identify_concerns(result)
        
        return {
            'safe': severity < 7,
            'severity': severity,
            'concerns': concerns,
            'raw_assessment': result
        }

    def _extract_severity(self, assessment: str) -> int:
        try:
            # Extract severity number from the assessment
            severity_line = [line for line in assessment.split('\n') 
                           if 'severity' in line.lower()]
            if severity_line:
                return int([num for num in severity_line[0] if num.isdigit()][0])
            return 0
        except:
            return 0

    def _identify_concerns(self, assessment: str) -> list:
        concerns = []
        if 'self-harm' in assessment.lower() or 'suicide' in assessment.lower():
            concerns.append('self_harm_risk')
        if 'harm to others' in assessment.lower():
            concerns.append('harm_to_others')
        return concerns
