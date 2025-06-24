"""
Agentic RAG module for enhancing the diagnosis and personality assessment processes.
Uses DSPy for structured reasoning, LlamaIndex for knowledge retrieval, and LangChain for agent integration.
"""

import os
import json
import logging
import numpy as np
import dspy
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion import IngestionPipeline
# Fix for vector_stores.chroma import
try:
    # Try the new package we just installed
    from llama_index_vector_stores_chroma import ChromaVectorStore
except ImportError:
    try:
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError:
        from llama_index.core.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

# LangChain imports
from langchain.schema.language_model import BaseLanguageModel
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.schema import Document as LCDocument

# Project imports - vector store for caching
from src.database.vector_store import VectorStore as ProjectVectorStore

# DSPy modules
from dspy.retrieve import Retrieve
import dspy
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
# Fix for dspy.pipeline import
try:
    from dspy.pipeline import DSPYPipeline
except ImportError:
    # Alternative implementation if the module can't be found
    class DSPYPipeline:
        """Fallback implementation for DSPYPipeline"""
        def __init__(self, modules=None):
            self.modules = modules or []

# Configure logging
logger = logging.getLogger(__name__)

# Define DSPy modules for mental health diagnosis
class SymptomExtractor(dspy.Module):
    """Extract mental health symptoms from user responses"""
    def __init__(self):
        super().__init__()
        self.extract_gen = dspy.Predict(
            dspy.Signature(
                {
                    "conversation": "The user's message or conversation history",
                    "symptoms": "A list of identified mental health symptoms and their severity (mild, moderate, severe)"
                }
            )
        )
    
    def forward(self, conversation: str) -> Dict[str, Any]:
        """Extract symptoms from user input"""
        result = self.extract_gen(conversation=conversation)
        try:
            # Try to convert string representation to actual list if needed
            symptoms = result.symptoms
            if isinstance(symptoms, str):
                # Parse the string into a structured format
                parsed_symptoms = []
                for line in symptoms.split('\n'):
                    if ':' in line:
                        symptom, severity = line.split(':', 1)
                        parsed_symptoms.append({
                            "symptom": symptom.strip(),
                            "severity": severity.strip()
                        })
                return {"symptoms": parsed_symptoms}
            return {"symptoms": symptoms}
        except Exception as e:
            logger.error(f"Error parsing symptoms: {str(e)}")
            return {"symptoms": []}

class DiagnosticReasoner(dspy.Module):
    """Perform diagnostic reasoning based on extracted symptoms"""
    def __init__(self, retriever: Optional[dspy.Retrieve] = None):
        super().__init__()
        self.retriever = retriever or Retrieve()
        
        self.diagnose = dspy.Predict(
            dspy.Signature(
                {
                    "symptoms": "A list of mental health symptoms and their severity",
                    "retrieved_context": "Retrieved diagnostic criteria and information",
                    "reasoning": "Step-by-step reasoning process for diagnostic assessment",
                    "potential_diagnoses": "A list of potential mental health conditions with confidence levels",
                    "severity": "Overall assessment severity (mild, moderate, severe)",
                    "recommendations": "Clinical recommendations based on the assessment"
                }
            )
        )
    
    def forward(self, symptoms: List[Dict[str, str]]) -> Dict[str, Any]:
        """Perform diagnostic reasoning"""
        # Convert symptoms to string format for retrieval
        symptoms_text = ", ".join([f"{s['symptom']} ({s['severity']})" for s in symptoms])
        
        # Retrieve relevant diagnostic information
        if hasattr(self.retriever, 'retrieve'):
            retrieved_docs = self.retriever.retrieve(symptoms_text, k=3)
            retrieved_context = "\n".join([doc.text for doc in retrieved_docs])
        else:
            retrieved_context = "No specific diagnostic criteria retrieved."
        
        # Generate diagnostic reasoning
        result = self.diagnose(
            symptoms=symptoms_text,
            retrieved_context=retrieved_context
        )
        
        return {
            "reasoning": result.reasoning,
            "potential_diagnoses": result.potential_diagnoses,
            "severity": result.severity,
            "recommendations": result.recommendations
        }

class PersonalityProfiler(dspy.Module):
    """Generate personality insights using retrieval-augmented generation"""
    def __init__(self, retriever: Optional[dspy.Retrieve] = None):
        super().__init__()
        self.retriever = retriever or Retrieve()
        
        self.profile = dspy.Predict(
            dspy.Signature(
                {
                    "assessment_data": "Results from a personality assessment",
                    "retrieved_context": "Retrieved information about personality traits",
                    "analysis": "In-depth analysis of personality traits",
                    "strengths": "Key strengths based on the personality profile",
                    "growth_areas": "Potential areas for personal growth",
                    "communication_style": "Communication preferences and patterns",
                    "learning_style": "Preferred learning approaches",
                    "stress_response": "Typical responses to stress based on personality"
                }
            )
        )
    
    def forward(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personality insights"""
        # Convert assessment data to string for retrieval
        assessment_text = json.dumps(assessment_data)
        
        # Retrieve relevant personality information
        if hasattr(self.retriever, 'retrieve'):
            retrieved_docs = self.retriever.retrieve(assessment_text, k=3)
            retrieved_context = "\n".join([doc.text for doc in retrieved_docs])
        else:
            retrieved_context = "No specific personality information retrieved."
        
        # Generate personality profile
        result = self.profile(
            assessment_data=assessment_text,
            retrieved_context=retrieved_context
        )
        
        return {
            "analysis": result.analysis,
            "strengths": result.strengths,
            "growth_areas": result.growth_areas,
            "communication_style": result.communication_style,
            "learning_style": result.learning_style,
            "stress_response": result.stress_response
        }

class EmotionPersonalityIntegrator(dspy.Module):
    """Integrate emotion data with personality insights"""
    
    def __init__(self):
        super().__init__()
        self.integrate = dspy.Predict(
            dspy.Signature(
                {
                    "personality_data": "Personality assessment results",
                    "emotion_data": "Emotion analysis data",
                    "integrated_insights": "Insights that connect personality traits with emotional patterns",
                    "correlations": "Specific correlations between personality traits and emotions",
                    "recommendations": "Personalized recommendations based on personality-emotion patterns"
                }
            )
        )
    
    def forward(self, personality_data: Dict[str, Any], emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate personality and emotion data"""
        result = self.integrate(
            personality_data=json.dumps(personality_data),
            emotion_data=json.dumps(emotion_data)
        )
        
        return {
            "integrated_insights": result.integrated_insights,
            "correlations": result.correlations,
            "recommendations": result.recommendations
        }

class AgenticRAG:
    """
    Agentic RAG implementation for mental health diagnosis and personality assessment.
    Combines DSPy for structured reasoning, LlamaIndex for retrieval, and LangChain for integration.
    """
    
    def __init__(
        self, 
        llm: BaseLanguageModel,
        embedding_model: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        knowledge_base_dir: Optional[str] = None,
        dspy_model_name: Optional[str] = None,
        use_result_caching: bool = True
    ):
        """
        Initialize the Agentic RAG system
        
        Args:
            llm: LangChain language model for generation
            embedding_model: Optional embedding model for vector search
            vector_store: Optional vector store for document retrieval
            knowledge_base_dir: Directory containing knowledge base documents
            dspy_model_name: Name of the DSPy-compatible model to use
            use_result_caching: Whether to use result caching with FAISS
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.use_result_caching = use_result_caching
        
        # Initialize FAISS vector store for result caching
        if self.use_result_caching:
            try:
                self.result_cache = ProjectVectorStore.create("faiss")
                self.result_cache.connect()
                logger.info("Successfully initialized FAISS result cache")
            except Exception as e:
                logger.warning(f"Could not initialize result cache: {str(e)}")
                self.result_cache = None
                self.use_result_caching = False
        
        # Set up DSPy
        if dspy_model_name:
            self.dspy_llm = dspy.OpenAI(model=dspy_model_name)
        else:
            # Use a wrapper for the LangChain LLM
            self.dspy_llm = self._create_dspy_llm_from_langchain(llm)
        dspy.settings.configure(lm=self.dspy_llm)
        
        # Initialize DSPy modules
        self.symptom_extractor = SymptomExtractor()
        self.retriever = self._initialize_retriever(knowledge_base_dir)
        self.diagnostic_reasoner = DiagnosticReasoner(retriever=self.retriever)
        self.personality_profiler = PersonalityProfiler(retriever=self.retriever)
        self.emotion_integrator = EmotionPersonalityIntegrator()
        
        # Initialize vectorstore and knowledge base
        self._initialize_knowledge_base(knowledge_base_dir)
    
    def _create_dspy_llm_from_langchain(self, llm: BaseLanguageModel) -> dspy.LM:
        """Create a DSPy-compatible LM from a LangChain LLM"""
        # This is a simple wrapper to adapt LangChain LLMs to DSPy interface
        class LangChainLMAdapter(dspy.LM):
            def __init__(self, langchain_llm):
                self.llm = langchain_llm
              def basic_request(self, prompt, **kwargs):
                try:
                    response = self.llm.invoke(prompt)
                    return response
                except Exception as e:
                    logger.error(f"Error in LLM request: {str(e)}")
                    return "Error generating response"
        
        return LangChainLMAdapter(llm)
    
    def _initialize_retriever(self, knowledge_base_dir: Optional[str]) -> dspy.Retrieve:
        """Initialize the DSPy retriever"""
        # With updated dspy, we'll use the Retrieve class instead
        # If we have a knowledge base directory, use it with Retrieve
        # Otherwise, fallback to Retrieve with default settings
        if knowledge_base_dir and os.path.exists(knowledge_base_dir):
            try:
                return Retrieve()  # Simplified for compatibility
            except Exception as e:
                logger.warning(f"Could not initialize retriever: {str(e)}")
                return Retrieve()
        return Retrieve()
    
    def _initialize_knowledge_base(self, knowledge_base_dir: Optional[str]) -> None:
        """Initialize the knowledge base for retrieval"""
        if not knowledge_base_dir or not os.path.exists(knowledge_base_dir):
            logger.warning(f"Knowledge base directory not found: {knowledge_base_dir}")
            return
        
        try:
            # Load diagnostic criteria data
            project_root = Path(__file__).parent.parent.parent
            diagnostic_data_path = project_root / "src" / "data" / "personality"
            
            # Create documents from the diagnostic criteria
            documents = []
            
            # Add diagnosis questions
            diagnosis_path = diagnostic_data_path / "diagnosis_questions.json"
            if diagnosis_path.exists():
                with open(diagnosis_path, 'r') as f:
                    diagnosis_data = json.load(f)
                    for category, questions in diagnosis_data.items():
                        for question in questions:
                            doc_text = f"{category} - {question.get('text', '')}"
                            if 'category' in question:
                                doc_text += f" (Category: {question['category']})"
                            documents.append(Document(text=doc_text, metadata={"source": "diagnosis_questions", "category": question.get('category', 'unknown')}))
            
            # Add Big Five questions
            big_five_path = diagnostic_data_path / "big_five_questions.json"
            if big_five_path.exists():
                with open(big_five_path, 'r') as f:
                    big_five_data = json.load(f)
                    for trait, questions in big_five_data.items():
                        for question in questions:
                            doc_text = f"{trait} - {question.get('text', '')}"
                            documents.append(Document(text=doc_text, metadata={"source": "big_five_questions", "trait": trait}))
            
            # Add any additional documents from the knowledge base directory
            if os.path.exists(knowledge_base_dir):
                kb_docs = SimpleDirectoryReader(knowledge_base_dir).load_data()
                documents.extend(kb_docs)
            
            # Create a vector store and index
            if not documents:
                logger.warning("No documents found for knowledge base")
                return
            
            # Create embedding model and vector store
            embed_model = OpenAIEmbedding()
            
            # Create a Chroma vector store
            chroma_dir = os.path.join(project_root, "src", "data", "vector_store", "chroma_db")
            os.makedirs(chroma_dir, exist_ok=True)
            
            chroma_store = ChromaVectorStore(collection_name="mental_health_kb", persist_dir=chroma_dir)
            storage_context = StorageContext.from_defaults(vector_store=chroma_store)
            
            # Parse nodes and index documents
            parser = SimpleNodeParser.from_defaults()
            pipeline = IngestionPipeline(
                transformations=[parser],
                vector_store=chroma_store
            )
            pipeline.run(documents=documents)
            
            # Create index from the ingested documents
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=chroma_store,
                embed_model=embed_model
            )
            
            logger.info(f"Successfully initialized knowledge base with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
    
    async def enhance_diagnosis(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance diagnosis using Agentic RAG with result caching
        
        Args:
            text: User text to analyze
            context: Optional additional context
            
        Returns:
            Enhanced diagnostic assessment
        """
        try:
            # Check cache first if enabled
            if self.use_result_caching and self.result_cache:
                cache_key = f"diagnosis_{text[:50]}"
                similar_results = self.result_cache.find_similar_results(cache_key, threshold=0.85, k=1)
                
                if similar_results:
                    logger.info(f"Using cached diagnosis result for similar query")
                    cached_result = similar_results[0].get('parsed_content')
                    if cached_result and isinstance(cached_result, dict) and cached_result.get('success'):
                        return cached_result
            
            # If no cache hit, proceed with normal processing
            # 1. Extract symptoms
            extraction_result = self.symptom_extractor(conversation=text)
            symptoms = extraction_result.get("symptoms", [])
            
            # 2. Perform diagnostic reasoning
            if symptoms:
                diagnosis_result = self.diagnostic_reasoner(symptoms=symptoms)
                
                # 3. Create the enhanced diagnosis result
                result = {
                    "success": True,
                    "symptoms": symptoms,
                    "reasoning": diagnosis_result.get("reasoning", ""),
                    "potential_diagnoses": diagnosis_result.get("potential_diagnoses", []),
                    "severity": diagnosis_result.get("severity", "mild"),
                    "recommendations": diagnosis_result.get("recommendations", []),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {
                    "success": False,
                    "error": "No symptoms extracted from the text",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Store result in cache if enabled
            if self.use_result_caching and self.result_cache and result.get('success'):
                cache_key = f"diagnosis_{text[:50]}"
                self.result_cache.add_processed_result(cache_key, result)
                logger.info(f"Cached diagnosis result for future use")
            
            return result
        except Exception as e:
            logger.error(f"Error in enhanced diagnosis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def enhance_personality_assessment(self, assessment_data: Dict[str, Any], emotional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance personality assessment using Agentic RAG with result caching
        
        Args:
            assessment_data: Results from personality assessment
            emotional_context: Optional emotional context data
            
        Returns:
            Enhanced personality insights
        """
        try:
            # Check cache first if enabled
            if self.use_result_caching and self.result_cache:
                # Create a deterministic cache key from the assessment data
                if isinstance(assessment_data, dict):
                    # Sort keys to ensure deterministic serialization
                    assessment_str = json.dumps(assessment_data, sort_keys=True)[:100]
                    cache_key = f"personality_{assessment_str}"
                    
                    similar_results = self.result_cache.find_similar_results(cache_key, threshold=0.9, k=1)
                    
                    if similar_results:
                        logger.info(f"Using cached personality assessment result")
                        cached_result = similar_results[0].get('parsed_content')
                        if cached_result and isinstance(cached_result, dict) and cached_result.get('success'):
                            return cached_result
            
            # If no cache hit, proceed with normal processing
            # 1. Generate personality insights
            profile_result = self.personality_profiler(assessment_data=assessment_data)
            
            # 2. Integrate emotion data if available
            if emotional_context:
                integration_result = self.emotion_integrator(
                    personality_data=assessment_data,
                    emotion_data=emotional_context
                )
                
                integrated_insights = integration_result.get("integrated_insights", "")
                correlations = integration_result.get("correlations", "")
                emotion_recommendations = integration_result.get("recommendations", "")
            else:
                integrated_insights = ""
                correlations = ""
                emotion_recommendations = ""
            
            # 3. Create the enhanced result
            result = {
                "success": True,
                "analysis": profile_result.get("analysis", ""),
                "strengths": profile_result.get("strengths", ""),
                "growth_areas": profile_result.get("growth_areas", ""),
                "communication_style": profile_result.get("communication_style", ""),
                "learning_style": profile_result.get("learning_style", ""),
                "stress_response": profile_result.get("stress_response", ""),
                "integrated_insights": integrated_insights,
                "emotion_correlations": correlations,
                "emotion_recommendations": emotion_recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store result in cache if enabled
            if self.use_result_caching and self.result_cache and result.get('success'):
                if isinstance(assessment_data, dict):
                    assessment_str = json.dumps(assessment_data, sort_keys=True)[:100]
                    cache_key = f"personality_{assessment_str}"
                    self.result_cache.add_processed_result(cache_key, result)
                    logger.info(f"Cached personality assessment result for future use")
            
            return result
        except Exception as e:
            logger.error(f"Error in enhanced personality assessment: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information with caching
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            # Check cache first if enabled
            if self.use_result_caching and self.result_cache:
                cache_key = f"query_{query[:50]}"
                similar_results = self.result_cache.find_similar_results(cache_key, threshold=0.9, k=1)
                
                if similar_results:
                    logger.info(f"Using cached query result for similar query")
                    cached_result = similar_results[0].get('parsed_content')
                    if cached_result and isinstance(cached_result, list):
                        return cached_result
            
            # If no cache hit, proceed with normal processing
            if not hasattr(self, 'index'):
                return []
            
            # Create a query engine
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            
            # Execute the query
            response = query_engine.query(query)
            
            # Extract relevant nodes/documents
            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    results.append({
                        "text": node.node.text,
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "metadata": node.node.metadata
                    })
            
            # Store result in cache if enabled
            if self.use_result_caching and self.result_cache and results:
                cache_key = f"query_{query[:50]}"
                self.result_cache.add_processed_result(cache_key, results)
                logger.info(f"Cached query result for future use")
            
            return results
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return []
    
    def clear_result_cache(self) -> bool:
        """Clear the FAISS result cache"""
        if not self.use_result_caching or not self.result_cache:
            logger.warning("Result caching is not enabled")
            return False
            
        try:
            self.result_cache.clear()
            logger.info("Successfully cleared result cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing result cache: {str(e)}")
            return False
    
    def train_with_examples(self, example_data: List[Dict[str, Any]], module_name: str) -> None:
        """
        Train a DSPy module with examples for better performance
        
        Args:
            example_data: List of example data for training
            module_name: Name of the module to train ('symptom_extractor', 'diagnostic_reasoner', 'personality_profiler')
        """
        try:
            if not example_data:
                logger.warning("No example data provided for training")
                return
            
            # Select the module to train
            if module_name == 'symptom_extractor':
                module = self.symptom_extractor
            elif module_name == 'diagnostic_reasoner':
                module = self.diagnostic_reasoner
            elif module_name == 'personality_profiler':
                module = self.personality_profiler
            else:
                logger.warning(f"Unknown module name: {module_name}")
                return
            
            # Create a few-shot trainer
            trainer = BootstrapFewShot(metric="exact_match")
            
            # Train the module
            compiled_module = trainer.compile(
                module=module,
                trainset=example_data,
                valset=example_data[:min(len(example_data), 2)]  # Use a small validation set
            )
            
            # Update the module
            if module_name == 'symptom_extractor':
                self.symptom_extractor = compiled_module
            elif module_name == 'diagnostic_reasoner':
                self.diagnostic_reasoner = compiled_module
            elif module_name == 'personality_profiler':
                self.personality_profiler = compiled_module
            
            logger.info(f"Successfully trained {module_name} with {len(example_data)} examples")
        except Exception as e:
            logger.error(f"Error training DSPy module: {str(e)}")