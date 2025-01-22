from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ChatAgent:
    def __init__(self, model_name: str, use_cpu: bool = True):
        self.model_name = model_name
        self.use_cpu = use_cpu
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.use_cpu:
            self.model = self.model.to("cpu")
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if self.use_cpu else 0
        )
        self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)

        self.chat_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Context: {context}\nHuman: {question}\nAI:"
        )
        self.chat_chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)

    def generate_response(self, question: str, context: str = "") -> str:
        response = self.chat_chain.run(context=context, question=question)
        return response.strip()

