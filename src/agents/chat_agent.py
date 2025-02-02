from typing import Optional
from langchain.llms import HuggingFacePipeline, BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
from config.settings import AppConfig

class ChatAgent:
    def __init__(self, model_name: str, use_cpu: bool = True, llm: Optional[BaseLLM] = None):
        self.model_name = model_name
        if llm:
            self.llm = llm
        else:
            device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                max_new_tokens=AppConfig.MAX_RESPONSE_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "emotion", "safety"],
            template="""[INST] <<SYS>>
You are a compassionate mental health counselor. Consider:
- User's emotional state: {emotion}
- Safety concerns: {safety}
- Context: {context}

Guidelines:
1. Respond with genuine empathy and validation
2. Provide practical, evidence-based suggestions
3. Maintain conversational, non-clinical tone
4. Prioritize user safety in all responses
<</SYS>>

User: {question} [/INST]"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_response(self, context: str, question: str, emotion: dict, safety: dict) -> str:
        try:
            response = self.chain.run({
                "context": context,
                "question": question,
                "emotion": f"{emotion.get('primary_emotion', 'neutral')} (intensity {emotion.get('intensity', 5)})",
                "safety": "Safety concern detected" if not safety.get('safe', True) else "No immediate safety concerns"
            })
            return self._postprocess_response(response)
        except Exception as e:
            return "I'm having trouble generating a response right now. Please try again."
    
    def _postprocess_response(self, response: str) -> str:
        for token in ["<s>", "</s>", "[INST]", "[/INST]"]:
            response = response.replace(token, "")
        return response.strip()