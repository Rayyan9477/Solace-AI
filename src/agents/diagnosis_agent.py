from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DiagnosisAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.diagnosis_prompt = PromptTemplate(
            input_variables=["symptoms"],
            template="Based on the following symptoms: {symptoms}, what mental health condition might the person be experiencing? Provide a compassionate and detailed response, including potential next steps and resources."
        )
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=self.diagnosis_prompt)

    def diagnose(self, symptoms: list) -> str:
        symptoms_str = ", ".join(symptoms)
        diagnosis = self.diagnosis_chain.run(symptoms=symptoms_str)
        return diagnosis.strip()

