from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict

class SafetyAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["message"],
            template="""Analyze this message for safety concerns:
            Message: {message}
            
            Evaluate:
            1. Immediate self-harm/suicide risk (1-10)
            2. Harm to others risk (1-10)
            3. Key risk factors
            
            Format response as:
            Risk Level: <self_harm_risk>/<harm_to_others_risk>
            Factors: <comma_separated_factors>"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def check_message(self, message: str) -> Dict:
        try:
            response = self.chain.run(message=message)
            return self._parse_response(response)
        except Exception:
            return {"safe": False, "severity": 10, "factors": ["system_error"]}

    def _parse_response(self, response: str) -> Dict:
        try:
            lines = response.split("\n")
            risk_line = [l for l in lines if "Risk Level:" in l][0]
            factors_line = [l for l in lines if "Factors:" in l][0]
            
            risks = risk_line.split(":")[1].strip().split("/")
            factors = factors_line.split(":")[1].strip().split(",")
            
            return {
                "safe": int(risks[0]) < 7 and int(risks[1]) < 7,
                "self_harm_risk": int(risks[0]),
                "harm_to_others_risk": int(risks[1]),
                "factors": [f.strip() for f in factors],
                "raw": response
            }
        except Exception:
            return {"safe": False, "severity": 8, "factors": ["parse_error"]}