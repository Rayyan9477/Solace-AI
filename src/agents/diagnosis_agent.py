from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DiagnosisAgent:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # Updated prompt to remove extraneous details and ensure we don't mention them in output
        self.diagnosis_prompt = PromptTemplate(
            input_variables=["symptoms"],
            template=(
                "Below are reported symptoms: {symptoms}.\n"
                "You are analyzing mental health symptoms.\n"
                "Briefly list possible concerns.\n"
                "Suggest next steps (e.g., professional help, lifestyle changes) in a warm, empathetic tone.\n"
                "Offer brief encouragement.\n"
                "1. Provide a short, medically plausible mental health diagnosis (e.g., 'Major Depression, moderate severity').\n"
                "2. Do not reference user age, gender, or any details not in the symptom list.\n"
                "3. Avoid repeating these instructions or your role.\n"
                "Do not include these instructions in your output."
            )
        )
        self.diagnosis_chain = LLMChain(llm=self.llm, prompt=self.diagnosis_prompt)

    def diagnose(self, symptoms: list) -> str:
        """Run the provided symptoms through the LLM chain, returning a short diagnosis."""
        symptoms_str = ", ".join(symptoms) if symptoms else "no clear symptoms"
        result = self.diagnosis_chain.run(symptoms=symptoms_str).strip()

        result = result.split('\n')[0]  # Take only the first line if multiple lines are returned
        result = result.replace('\n', ' ').strip()  # Replace any remaining newlines with spaces and trim whitespace
        
        return result