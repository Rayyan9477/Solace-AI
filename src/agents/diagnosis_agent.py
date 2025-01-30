from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DiagnosisAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.diagnosis_prompt = PromptTemplate(
            input_variables=["symptoms"],
            template=(
                "You are a compassionate mental health bot. Based on the following "
                "symptoms: {symptoms}, what mental health condition might the person be experiencing? "
                "Provide potential next steps and mention resources that might help. "
                "Be sure to respond with empathy and reassurance."
            )
        )
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=self.diagnosis_prompt)

    def diagnose(self, symptoms: list) -> str:
        symptoms_str = ", ".join(symptoms) if symptoms else "no clear symptoms"
        diagnosis = self.diagnosis_chain.run(symptoms=symptoms_str)
        return diagnosis.strip()