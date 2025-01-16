import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from typing import Optional
from transformers import pipeline

class LocalLLM:
    """
    Wrapper for a local Causal Language Model that runs on CPU.
    Manages loading, tokenization, and text generation.
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

    def generate_text(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        num_beams: int = 1,
        context: Optional[str] = None
    ) -> str:
        """
        Generates text from the local LLM.
        Optionally includes context for RAG or additional prompt data.
        """
        if context:
            combined_prompt = f"Context:\n{context}\nUser Query:\n{prompt}\nResponse:"
        else:
            combined_prompt = f"User Query:\n{prompt}\nResponse:"

        inputs = self.tokenizer.encode(combined_prompt, return_tensors="pt")
        if self.use_cpu:
            inputs = inputs.cpu()

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_length=max_length + inputs['input_ids'].shape[1],
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract assistant's response
        response_parts = response.split("Response:")
        if len(response_parts) > 1:
            response = response_parts[1].strip()
        else:
            response = response_parts[0].replace(prompt, "").strip()

        return response