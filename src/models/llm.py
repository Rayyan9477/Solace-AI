import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

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
                max_length=max_length,
                num_beams=num_beams,
                do_sample=True,
                temperature=temperature,
            )
        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Trim prompt from final output if "Response:" is present
        if "Response:" in result:
            result = result.split("Response:")[-1].strip()
        return result