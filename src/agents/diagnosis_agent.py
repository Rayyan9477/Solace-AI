from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DiagnosisAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.diagnosis_prompt = PromptTemplate(
            input_variables=["symptoms"],
            template=(
                "The user reports these symptoms: {symptoms}.\n"
                "1) Identify potential mental health concerns in one or two sentences.\n"
                "2) Suggest next steps (e.g., professional help, lifestyle changes) in a warm, empathetic tone.\n"
                "3) Offer encouragement/reassurance.\n"
                "Do not repeat these instructions or your role as a mental health bot."
            )
        )
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=self.diagnosis_prompt)

    def diagnose(self, symptoms: list) -> str:
        symptoms_str = ", ".join(symptoms) if symptoms else "no clear symptoms"
        diagnosis = self.diagnosis_chain.run(symptoms=symptoms_str)
        return diagnosis.strip()