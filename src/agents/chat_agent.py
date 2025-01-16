import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Optional
from langchain.llms import HuggingFacePipeline

class ChatAgent:
    """
    ChatAgent manages interactions with the language model for a mental health chatbot.
    Uses a local LLM (meta-llama/Llama-3.2-3B-Instruct) running on CPU.
    Incorporates RAG-based retrieval if needed (via a search or vector store).
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_cpu: bool = True
    ):
        self.model_name = model_name
        self.use_cpu = use_cpu
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.use_cpu:
            self.model = self.model.cpu()
        
        # Initialize HuggingFace pipeline for LangChain
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if self.use_cpu else 0
        )
        self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generates a response from the LLM based on prompt + optional retrieval context.
        Example usage:
            response = chat_agent.generate_response("I feel anxious all the time.")
        """
        if context:
            combined_input = f"Context:\n{context}\nUser Query:\n{prompt}\nResponse:"
        else:
            combined_input = f"User Query:\n{prompt}\nResponse:"

        inputs = self.tokenizer.encode(combined_input, return_tensors="pt")
        if self.use_cpu:
            inputs = inputs.cpu()

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_length=256,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
            )
        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Response:" in result:
            result = result.split("Response:")[-1].strip()
        return result

    def collect_symptoms(self, existing_symptoms: List[str]) -> str:
        """
        Collects user mental health symptoms in a structured manner.
        """
        return f"Collected symptoms: {', '.join(existing_symptoms)}"

    def diagnose_condition(self, symptoms: List[str]) -> str:
        """
        Simple placeholder for diagnosing mental health conditions based on user symptoms.
        """
        # Example logic: match certain keywords for a basic classification
        if any(keyword in symptoms for keyword in ["sadness", "loss of interest", "hopeless"]):
            return "User might be showing signs of depression."
        elif any(keyword in symptoms for keyword in ["worry", "panic", "fear"]):
            return "User might be showing signs of anxiety."
        return "No clear diagnosis based on the provided symptoms."