"""
Comprehensive Diagnosis Module

This module provides an advanced system for mental health assessment by integrating:
1. Voice emotion analysis - Detecting emotional states from voice patterns
2. Conversational AI - Extracting insights from chat interactions
3. Personality assessments - Using structured personality test results
4. Vector similarity - Using embeddings for enhanced pattern recognition
5. Agentic RAG - Leveraging retrieval-augmented generation for improved reasoning

The system provides severity assessments, confidence scores, and personalized recommendations.
"""

import os
import json
import numpy as np
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import faiss
from collections import defaultdict

# Project imports
from src.database.vector_store import VectorStore
from src.utils.agentic_rag import AgenticRAG
from src.config.settings import AppConfig
from src.models.llm import get_llm

# Configure logging
logger = logging.getLogger(__name__)

# Mental health condition definitions with associated symptoms and patterns
CONDITION_DEFINITIONS = {
    "depression": {
        "name": "Depression",
        "symptoms": [
            "persistent sadness", "loss of interest", "fatigue", "sleep problems",
            "appetite changes", "feelings of worthlessness", "difficulty concentrating", 
            "negative thoughts", "low energy", "social withdrawal", "low self-esteem",
            "suicidal thoughts", "feeling empty", "guilt", "self-blame", "helplessness",
            "hopelessness", "psychomotor retardation", "crying spells"
        ],
        "voice_indicators": [
            "monotone voice", "slower speech", "reduced pitch variation", "quieter volume",
            "flat affect", "audible sighs", "long pauses", "reduced speech rate"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "low",
                "openness": "variable",
                "agreeableness": "variable",
                "conscientiousness": "low"
            }
        },
        "severity_thresholds": {
            "mild": 3,      # 3+ symptoms present
            "moderate": 5,  # 5+ symptoms present
            "severe": 7     # 7+ symptoms present
        }
    },
    "anxiety": {
        "name": "Anxiety",
        "symptoms": [
            "excessive worry", "restlessness", "fatigue", "difficulty concentrating", 
            "irritability", "muscle tension", "sleep problems", "racing thoughts",
            "feeling on edge", "anticipating worst outcomes", "avoiding situations",
            "panic attacks", "sweating", "trembling", "heart palpitations",
            "shortness of breath", "fear of losing control", "difficulty making decisions"
        ],
        "voice_indicators": [
            "faster speech", "higher pitch", "trembling voice", "rapid breathing",
            "voice cracks", "stuttering", "frequent clearing of throat", "halting speech"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high", 
                "extraversion": "variable",
                "openness": "variable",
                "agreeableness": "variable", 
                "conscientiousness": "high"
            }
        },
        "severity_thresholds": {
            "mild": 3,      # 3+ symptoms present
            "moderate": 5,  # 5+ symptoms present
            "severe": 7     # 7+ symptoms present
        }
    },
    "stress": {
        "name": "Stress",
        "symptoms": [
            "feeling overwhelmed", "racing thoughts", "difficulty relaxing", 
            "irritability", "muscle tension", "headaches", "fatigue",
            "sleep problems", "difficulty concentrating", "mood changes",
            "increased heart rate", "digestive issues", "feeling pressured",
            "inability to switch off", "worrying about the future"
        ],
        "voice_indicators": [
            "faster speech", "tense tone", "higher pitch", "louder volume",
            "rapid breathing", "voice strain", "clipped sentences"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "variable",
                "openness": "variable",
                "agreeableness": "low when stressed",
                "conscientiousness": "variable"
            }
        },
        "severity_thresholds": {
            "mild": 3,      # 3+ symptoms present
            "moderate": 5,  # 5+ symptoms present
            "severe": 7     # 7+ symptoms present
        }
    },
    "ptsd": {
        "name": "PTSD",
        "symptoms": [
            "flashbacks", "nightmares", "intrusive memories", "distress at reminders",
            "avoiding trauma reminders", "negative mood", "feeling detached",
            "hypervigilance", "exaggerated startle response", "difficulty sleeping",
            "irritability", "concentration problems", "memory gaps", "self-destructive behavior"
        ],
        "voice_indicators": [
            "emotional numbing in voice", "sudden vocal shifts", "voice trembling",
            "hesitation when discussing triggers", "heightened vocal tension"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "low",
                "openness": "variable",
                "agreeableness": "low",
                "conscientiousness": "variable"
            }
        },
        "severity_thresholds": {
            "mild": 3,     
            "moderate": 6,  
            "severe": 9     
        }
    },
    "bipolar": {
        "name": "Bipolar Disorder",
        "symptoms": [
            "mood swings", "periods of depression", "periods of elation", 
            "increased energy", "decreased need for sleep", "grandiose ideas",
            "racing thoughts", "rapid speech", "impulsive behavior", "irritability",
            "risky behavior", "poor judgment", "inflated self-esteem"
        ],
        "voice_indicators": [
            "rapid speech during mania", "pressured speech", "flight of ideas in speech",
            "loud volume during mania", "monotone during depression", "dramatic tone shifts"
        ],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "variable",
                "extraversion": "variable/cyclical",
                "openness": "high",
                "agreeableness": "variable",
                "conscientiousness": "variable/cyclical"
            }
        },
        "severity_thresholds": {
            "mild": 3,     
            "moderate": 6, 
            "severe": 9    
        }
    }
}

# Response templates for different diagnosis severities
RESPONSE_TEMPLATES = {
    "severe": {
        "depression": (
            "I'm noticing several indicators in our conversation that align with symptoms of depression "
            "at a significant level. These include {symptoms}. Based on these patterns, it may be beneficial "
            "to speak with a mental health professional soon. They can provide proper evaluation and support."
        ),
        "anxiety": (
            "Our conversation suggests you may be experiencing several symptoms associated with anxiety "
            "at a concerning level, such as {symptoms}. Speaking with a mental health professional "
            "could provide you with effective strategies and support for managing these feelings."
        ),
        "stress": (
            "I'm detecting multiple signs of significant stress in our conversation, including {symptoms}. "
            "This level of stress can impact your wellbeing if sustained. Connecting with a healthcare "
            "provider could help you develop effective stress management techniques."
        ),
        "ptsd": (
            "I've noticed several patterns in our conversation that align with post-traumatic stress, "
            "including {symptoms}. These symptoms appear to be at a significant level. Speaking with a "
            "trauma-informed mental health professional would be beneficial for proper assessment and care."
        ),
        "bipolar": (
            "Our conversation contains several indicators that align with bipolar patterns, including {symptoms}. "
            "The nature of these symptoms suggests it would be beneficial to consult with a psychiatrist "
            "who can provide proper evaluation and discuss management strategies."
        )
    },
    "moderate": {
        "depression": (
            "I'm noticing some patterns in our conversation that have similarities to depression symptoms, "
            "like {symptoms}. These indicators are at a moderate level. If these feelings are affecting "
            "your daily life, consider speaking with a healthcare provider."
        ),
        "anxiety": (
            "Our conversation suggests some moderate anxiety-related patterns, including {symptoms}. "
            "If you find these feelings are interfering with your daily activities, speaking with a "
            "mental health professional could be helpful."
        ),
        "stress": (
            "I'm recognizing moderate stress indicators in our conversation, such as {symptoms}. "
            "Finding effective ways to manage stress can prevent it from becoming more severe. "
            "Consider activities like mindfulness, exercise, or speaking with a professional."
        ),
        "ptsd": (
            "Some patterns in our conversation align with moderate post-traumatic stress responses, "
            "including {symptoms}. Speaking with a mental health professional who specializes in "
            "trauma could provide helpful insights and support."
        ),
        "bipolar": (
            "I'm noticing some patterns in our interaction that have similarities to mood cycling, "
            "including {symptoms}. These patterns are at a moderate level. A mental health professional "
            "could help determine if these experiences warrant further attention."
        )
    },
    "mild": {
        "depression": (
            "I'm noticing some mild indicators in our conversation that sometimes accompany low mood, "
            "including {symptoms}. While these are mild, monitoring how they affect you over time "
            "is important."
        ),
        "anxiety": (
            "There are some mild patterns in our conversation that can sometimes be associated with "
            "anxiety, such as {symptoms}. These appear to be at a mild level, but it's good to be "
            "aware of them."
        ),
        "stress": (
            "I'm detecting some mild stress indicators in our conversation, like {symptoms}. "
            "While mild stress is a normal part of life, having good coping strategies is important."
        ),
        "ptsd": (
            "I've noticed some mild stress response patterns in our conversation, including {symptoms}. "
            "While these are mild, paying attention to how they affect you is important, especially "
            "if they're connected to difficult experiences."
        ),
        "bipolar": (
            "There are some mild mood variation patterns in our conversation, including {symptoms}. "
            "These appear to be at a mild level. Monitoring how your mood patterns affect your life "
            "can be helpful."
        )
    }
}

class ComprehensiveDiagnosisModule:
    """
    Integrates voice emotion analysis, conversational AI output, and personality test results
    to provide comprehensive mental health insights using advanced techniques including
    vector similarity search and agentic RAG.
    """
    
    def __init__(self, 
                agentic_rag: Optional[AgenticRAG] = None, 
                use_vector_cache: bool = True,
                model_context_window: int = 4096,
                confidence_threshold: float = 0.3):
        """
        Initialize the comprehensive diagnosis module
        
        Args:
            agentic_rag: Optional AgenticRAG instance for enhanced reasoning
            use_vector_cache: Whether to use vector caching for previous diagnoses
            model_context_window: Maximum context window size for the LLM
            confidence_threshold: Minimum confidence threshold for condition reporting
        """
        self.agentic_rag = agentic_rag
        self.use_vector_cache = use_vector_cache
        self.model_context_window = model_context_window
        self.confidence_threshold = confidence_threshold
        
        # Initialize LLM
        try:
            self.llm = get_llm()
            logger.info("Successfully initialized LLM for diagnosis")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {str(e)}. Some features will be limited.")
            self.llm = None
        
        # Initialize vector store for result caching if enabled
        if self.use_vector_cache:
            try:
                self.vector_store = VectorStore.create("faiss")
                self.vector_store.connect()
                logger.info("Successfully initialized vector store for diagnosis caching")
            except Exception as e:
                logger.warning(f"Could not initialize vector store: {str(e)}")
                self.vector_store = None
                self.use_vector_cache = False
        else:
            self.vector_store = None
            
        # Load symptom embeddings for improved matching
        self.symptom_embeddings = {}
        self.symptom_index = None
        self._initialize_symptom_embeddings()
    
    def _initialize_symptom_embeddings(self):
        """Initialize embeddings for all symptoms for vector matching"""
        try:
            # Extract all symptoms from condition definitions
            all_symptoms = []
            for condition in CONDITION_DEFINITIONS.values():
                all_symptoms.extend(condition["symptoms"])
            
            # Remove duplicates
            all_symptoms = list(set(all_symptoms))
            
            # If we have an LLM with embedding capability, create embeddings
            if self.llm and hasattr(self.llm, "get_embedding"):
                # Create embeddings for each symptom
                for symptom in all_symptoms:
                    embedding = self.llm.get_embedding(symptom)
                    if embedding is not None:
                        self.symptom_embeddings[symptom] = embedding
                
                # Create a FAISS index for fast similarity search
                if self.symptom_embeddings:
                    embedding_dim = len(next(iter(self.symptom_embeddings.values())))
                    self.symptom_index = faiss.IndexFlatL2(embedding_dim)
                    
                    # Add embeddings to the index
                    embeddings_array = np.array(list(self.symptom_embeddings.values())).astype('float32')
                    self.symptom_index.add(embeddings_array)
                    
                    logger.info(f"Successfully initialized symptom embeddings for {len(self.symptom_embeddings)} symptoms")
            else:
                logger.warning("LLM does not support embeddings, using keyword matching only")
        except Exception as e:
            logger.error(f"Error initializing symptom embeddings: {str(e)}")
    
    async def generate_diagnosis(
        self,
        conversation_data: Optional[Dict[str, Any]] = None,
        voice_emotion_data: Optional[Dict[str, Any]] = None,
        personality_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        external_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnosis by integrating multiple data sources
        
        Args:
            conversation_data: Data extracted from chatbot conversations
            voice_emotion_data: Emotional patterns detected from voice analysis
            personality_data: Results from personality assessments
            user_id: Optional user identifier for caching
            session_id: Optional session identifier for contextual awareness
            external_context: Any additional context that might be relevant
            
        Returns:
            Comprehensive diagnostic assessment
        """
        start_time = datetime.now()
        
        # Check cache for similar diagnosis if enabled
        if self.use_vector_cache and self.vector_store and user_id:
            cache_key = f"diagnosis_{user_id}_{session_id}_{start_time.strftime('%Y%m%d')}"
            cached_result = await self._check_diagnosis_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached diagnosis for user {user_id}")
                return cached_result
        
        # Initialize result structure
        result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "conditions": [],
            "severity": "none",
            "confidence": 0.0,
            "recommendations": [],
            "insights": {},
            "data_sources_used": []
        }
        
        # Track which data sources we're using
        if conversation_data:
            result["data_sources_used"].append("conversation")
        if voice_emotion_data:
            result["data_sources_used"].append("voice_emotion")
        if personality_data:
            result["data_sources_used"].append("personality")
            
        if not result["data_sources_used"]:
            result["error"] = "No data sources provided for diagnosis"
            return result
        
        # Extract symptoms and indicators from conversation data
        conversation_symptoms = []
        if conversation_data:
            conversation_symptoms = await self._extract_symptoms_from_conversation(conversation_data)
            
        # Extract emotional indicators from voice data
        voice_indicators = []
        if voice_emotion_data:
            voice_indicators = self._extract_indicators_from_voice(voice_emotion_data)
            
        # Extract personality traits from personality assessment
        personality_traits = {}
        if personality_data:
            personality_traits = self._extract_traits_from_personality(personality_data)
        
        # Combine all indicators into a unified list
        all_indicators = conversation_symptoms + voice_indicators
        
        # Enhanced symptom detection with vector similarity
        if self.symptom_index is not None and hasattr(self.llm, "get_embedding"):
            vector_symptoms = await self._detect_symptoms_with_vectors(conversation_data)
            all_indicators.extend(vector_symptoms)
        
        # Use AgenticRAG if available for enhanced analysis
        agentic_insights = {}
        if self.agentic_rag and conversation_data and "text" in conversation_data:
            try:
                conversation_text = conversation_data["text"]
                
                # Include history if available for better context
                if "history" in conversation_data and isinstance(conversation_data["history"], list):
                    history_texts = [msg.get("text", "") for msg in conversation_data["history"] 
                                   if isinstance(msg, dict) and "text" in msg]
                    if history_texts:
                        # Limit history to fit in context window
                        combined_history = " ".join(history_texts[-5:])  # Use last 5 messages
                        if len(combined_history) + len(conversation_text) < self.model_context_window:
                            conversation_text = f"{combined_history}\n\nCurrent message: {conversation_text}"
                
                enhanced_results = await self.agentic_rag.enhance_diagnosis(conversation_text)
                if enhanced_results and enhanced_results.get("success"):
                    agentic_insights = enhanced_results
                    
                    # Add symptoms from enhanced analysis if available
                    if "symptoms" in enhanced_results and isinstance(enhanced_results["symptoms"], list):
                        for symptom in enhanced_results["symptoms"]:
                            if isinstance(symptom, dict) and "symptom" in symptom:
                                symptom_text = symptom["symptom"]
                                source = "agentic_rag"
                                confidence = symptom.get("confidence", 0.7)  # Default confidence
                                
                                # Check if this symptom is already in the list
                                if not any(s["text"].lower() == symptom_text.lower() for s in all_indicators):
                                    all_indicators.append({
                                        "text": symptom_text, 
                                        "source": source,
                                        "confidence": confidence
                                    })
            except Exception as e:
                logger.error(f"Error in AgenticRAG processing: {str(e)}")
        
        # Add LLM-based symptom extraction if available
        if self.llm and conversation_data and "text" in conversation_data:
            llm_symptoms = await self._extract_symptoms_with_llm(conversation_data["text"])
            
            # Add LLM-detected symptoms to the indicators
            for symptom in llm_symptoms:
                # Check if this symptom is already in the list
                if not any(s["text"].lower() == symptom["text"].lower() for s in all_indicators):
                    all_indicators.append(symptom)
        
        # Match all indicators against condition definitions using rule-based approach
        conditions_matched = await self._match_conditions(all_indicators, personality_traits)
        
        # If we have matches, build the diagnosis result
        if conditions_matched:
            result["success"] = True
            result["conditions"] = []
            
            # Add condition information
            for cond in conditions_matched:
                result["conditions"].append({
                    "name": cond["name"],
                    "confidence": cond["confidence"], 
                    "severity": cond["severity"],
                    "symptom_count": len(cond["matching_symptoms"])
                })
            
            # Use the highest confidence condition as the primary one
            primary_condition = max(conditions_matched, key=lambda x: x["confidence"])
            result["severity"] = primary_condition["severity"]
            result["confidence"] = primary_condition["confidence"]
            result["primary_condition"] = primary_condition["id"]
            
            # Generate recommendations based on conditions
            result["recommendations"] = await self._generate_recommendations(conditions_matched, personality_traits)
            
            # Add insights
            result["insights"] = {
                "symptoms_identified": [s["text"] for s in conversation_symptoms],
                "emotional_indicators": [v["text"] for v in voice_indicators],
                "personality_factors": personality_traits,
                "evidence": self._generate_evidence_summary(conditions_matched, all_indicators),
                "data_quality": self._assess_data_quality(conversation_data, voice_emotion_data, personality_data)
            }
            
            # Add agentic insights if available
            if agentic_insights:
                result["agentic_insights"] = {
                    "reasoning": agentic_insights.get("reasoning", ""),
                    "potential_diagnoses": agentic_insights.get("potential_diagnoses", []),
                    "recommendations": agentic_insights.get("recommendations", [])
                }
                
                # Add AgenticRAG recommendations if available
                if "recommendations" in agentic_insights and isinstance(agentic_insights["recommendations"], list):
                    for rec in agentic_insights["recommendations"]:
                        if rec not in result["recommendations"]:
                            result["recommendations"].append(rec)
                    
                    # Limit to a reasonable number
                    result["recommendations"] = result["recommendations"][:10]
        else:
            result["message"] = "No significant conditions detected based on available data"
            result["recommendations"] = [
                "Continue monitoring your mental wellbeing",
                "Practice regular self-care activities",
                "Maintain social connections and support systems"
            ]
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        result["processing_time_seconds"] = processing_time
        
        # Cache the result if enabled
        if self.use_vector_cache and self.vector_store and user_id and result["success"]:
            cache_key = f"diagnosis_{user_id}_{session_id}_{start_time.strftime('%Y%m%d')}"
            await self._cache_diagnosis(cache_key, result)
        
        return result
    
    async def _extract_symptoms_from_conversation(self, conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract symptoms from conversation data"""
        symptoms = []
        
        if "text" in conversation_data:
            text = conversation_data["text"].lower()
            
            # Include conversation history if available
            if "history" in conversation_data and isinstance(conversation_data["history"], list):
                for msg in conversation_data["history"]:
                    if isinstance(msg, dict) and "text" in msg:
                        text += " " + msg["text"].lower()
            
            # Flatten all symptoms from all conditions into a single set for matching
            all_symptoms = set()
            for condition in CONDITION_DEFINITIONS.values():
                all_symptoms.update(condition["symptoms"])
            
            # Simple keyword matching for symptoms in text
            for symptom in all_symptoms:
                symptom_lower = symptom.lower()
                if symptom_lower in text:
                    # Calculate a simple confidence score based on exact match (1.0) or partial match
                    confidence = 1.0 if f" {symptom_lower} " in f" {text} " else 0.7
                    symptoms.append({
                        "text": symptom, 
                        "source": "conversation_keyword",
                        "confidence": confidence
                    })
        
        if "extracted_symptoms" in conversation_data:
            # If the conversation analyzer already extracted symptoms
            for symptom in conversation_data["extracted_symptoms"]:
                if isinstance(symptom, str):
                    symptoms.append({
                        "text": symptom, 
                        "source": "conversation_analyzer",
                        "confidence": 0.8
                    })
                elif isinstance(symptom, dict) and "text" in symptom:
                    # If confidence is already provided, use it
                    confidence = symptom.get("confidence", 0.8)
                    source = symptom.get("source", "conversation_analyzer")
                    symptoms.append({
                        "text": symptom["text"],
                        "source": source,
                        "confidence": confidence
                    })
        
        return symptoms
    
    async def _detect_symptoms_with_vectors(self, conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use vector similarity to detect symptoms in conversation"""
        if not self.symptom_index or not conversation_data or "text" not in conversation_data:
            return []
        
        try:
            # Get embedding for the conversation text
            text = conversation_data["text"]
            text_embedding = self.llm.get_embedding(text)
            
            if text_embedding is None:
                return []
            
            # Convert to numpy array
            query_vector = np.array([text_embedding]).astype('float32')
            
            # Search for similar symptoms
            k = min(10, self.symptom_index.ntotal)  # Search for top k matches
            if k == 0:
                return []
                
            distances, indices = self.symptom_index.search(query_vector, k)
            
            # Convert index results back to symptoms
            symptoms = []
            symptom_list = list(self.symptom_embeddings.keys())
            
            # Calculate a confidence score from distance
            # Lower distance = higher confidence
            max_meaningful_distance = 1.5  # Threshold for meaningful similarity
            
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(symptom_list):
                    symptom = symptom_list[idx]
                    
                    # Convert distance to confidence score (inverse relationship)
                    # Using a sigmoid-like function to map distance to [0,1]
                    if distance > max_meaningful_distance:
                        confidence = 0.0
                    else:
                        confidence = 1.0 - (distance / max_meaningful_distance)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    
                    # Only include if confidence is above threshold
                    if confidence >= 0.6:
                        symptoms.append({
                            "text": symptom,
                            "source": "vector_similarity",
                            "confidence": confidence,
                            "distance": float(distance)
                        })
            
            return symptoms
        except Exception as e:
            logger.error(f"Error in vector symptom detection: {str(e)}")
            return []
    
    async def _extract_symptoms_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Use the LLM to identify potential symptoms in text"""
        if not self.llm:
            return []
        
        try:
            # Create a prompt for symptom extraction
            prompt = f"""
            Analyze the following text and identify any potential mental health symptoms.
            Focus on symptoms related to depression, anxiety, stress, PTSD, or bipolar disorder.
            
            Text: "{text}"
            
            For each symptom you identify, rate your confidence from 0.0 to 1.0.
            Format your response as a JSON list of objects with "symptom" and "confidence" fields:
            [
                {{"symptom": "example symptom", "confidence": 0.8}},
                ...
            ]
            
            Only include symptoms if your confidence is at least 0.6.
            """
            
            # Get response from LLM
            response = await self.llm.agenerate(prompt)
            
            # Parse the response to extract symptoms
            symptoms = []
            try:
                # Try to extract JSON from the response
                response_text = response.get("choices", [{}])[0].get("text", "")
                
                # Find JSON content (may be embedded in explanatory text)
                import re
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_symptoms = json.loads(json_str)
                    
                    # Process extracted symptoms
                    for item in extracted_symptoms:
                        if isinstance(item, dict) and "symptom" in item and "confidence" in item:
                            symptom = item["symptom"]
                            confidence = float(item["confidence"])
                            
                            if confidence >= 0.6:
                                symptoms.append({
                                    "text": symptom,
                                    "source": "llm_extraction",
                                    "confidence": confidence
                                })
            except Exception as e:
                logger.warning(f"Error parsing LLM symptom response: {str(e)}")
            
            return symptoms
        except Exception as e:
            logger.error(f"Error in LLM symptom extraction: {str(e)}")
            return []
    
    def _extract_indicators_from_voice(self, voice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract emotional indicators from voice analysis data"""
        indicators = []
        
        # Extract emotion labels
        if "emotions" in voice_data:
            emotions = voice_data["emotions"]
            if isinstance(emotions, dict):
                # Format: {"sad": 0.8, "angry": 0.2, ...}
                for emotion, score in emotions.items():
                    if score >= 0.5:  # Only consider significant emotions
                        indicators.append({
                            "text": f"high {emotion} tone", 
                            "source": "voice_emotion", 
                            "confidence": score
                        })
        
        # Extract voice characteristics
        if "characteristics" in voice_data:
            chars = voice_data["characteristics"]
            if isinstance(chars, dict):
                # Get all voice indicators from conditions
                all_voice_indicators = set()
                for condition in CONDITION_DEFINITIONS.values():
                    all_voice_indicators.update(condition["voice_indicators"])
                
                # Match characteristics to known indicators
                for indicator in all_voice_indicators:
                    for char, value in chars.items():
                        if char in indicator and value >= 0.5:
                            indicators.append({
                                "text": indicator,
                                "source": "voice_characteristics",
                                "confidence": value
                            })
        
        # Process acoustic features if available
        if "acoustic_features" in voice_data:
            features = voice_data["acoustic_features"]
            if isinstance(features, dict):
                # Map acoustic features to potential indicators
                feature_mappings = {
                    "speech_rate": {
                        "low": {"text": "slower speech", "threshold": 0.3},
                        "high": {"text": "faster speech", "threshold": 0.7}
                    },
                    "pitch_variation": {
                        "low": {"text": "reduced pitch variation", "threshold": 0.3},
                        "high": {"text": "high pitch variation", "threshold": 0.7}
                    },
                    "volume": {
                        "low": {"text": "quieter volume", "threshold": 0.3},
                        "high": {"text": "louder volume", "threshold": 0.7}
                    },
                    "pause_frequency": {
                        "high": {"text": "frequent pauses in speech", "threshold": 0.7}
                    },
                    "voice_tremor": {
                        "high": {"text": "trembling voice", "threshold": 0.6}
                    }
                }
                
                # Process each feature
                for feature, value in features.items():
                    if feature in feature_mappings:
                        mapping = feature_mappings[feature]
                        
                        # Check for low values
                        if "low" in mapping and value <= mapping["low"]["threshold"]:
                            indicators.append({
                                "text": mapping["low"]["text"],
                                "source": "acoustic_analysis",
                                "confidence": 1.0 - (value / mapping["low"]["threshold"])
                            })
                        
                        # Check for high values
                        if "high" in mapping and value >= mapping["high"]["threshold"]:
                            indicators.append({
                                "text": mapping["high"]["text"],
                                "source": "acoustic_analysis",
                                "confidence": (value - mapping["high"]["threshold"]) / (1.0 - mapping["high"]["threshold"])
                            })
        
        return indicators
    
    def _extract_traits_from_personality(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traits from personality assessment"""
        traits = {}
        
        # Extract Big Five traits if available
        if "big_five" in personality_data:
            big_five = personality_data["big_five"]
            if isinstance(big_five, dict):
                traits["big_five"] = {}
                for trait, score in big_five.items():
                    if isinstance(score, (int, float)):
                        # Convert numerical scores to qualitative level (low, medium, high)
                        level = "medium"
                        if score >= 0.7:
                            level = "high"
                        elif score <= 0.3:
                            level = "low"
                        
                        traits["big_five"][trait.lower()] = {
                            "score": score,
                            "level": level
                        }
        
        # Extract MBTI if available
        if "mbti" in personality_data:
            traits["mbti"] = personality_data["mbti"]
            
            # Map MBTI dimensions to potential indicators
            if isinstance(traits["mbti"], str) and len(traits["mbti"]) == 4:
                # Extract the dimensions
                mbti_type = traits["mbti"]
                traits["mbti_dimensions"] = {
                    "introversion_extraversion": "I" if "I" in mbti_type[0] else "E",
                    "sensing_intuition": "S" if "S" in mbti_type[1] else "N",
                    "thinking_feeling": "T" if "T" in mbti_type[2] else "F",
                    "judging_perceiving": "J" if "J" in mbti_type[3] else "P"
                }
        
        # Add personality insights if we have Big Five data
        if "big_five" in traits:
            traits["insights"] = self._generate_personality_insights(traits["big_five"])
        
        return traits
    
    def _generate_personality_insights(self, big_five: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights based on Big Five personality traits"""
        insights = {
            "cognitive_style": "",
            "stress_response": "",
            "interpersonal_approach": ""
        }
        
        # Determine cognitive style based on openness and conscientiousness
        if "openness" in big_five and "conscientiousness" in big_five:
            openness_level = big_five["openness"]["level"]
            conscient_level = big_five["conscientiousness"]["level"]
            
            if openness_level == "high" and conscient_level == "high":
                insights["cognitive_style"] = "Structured creativity - combines imagination with organization"
            elif openness_level == "high" and conscient_level == "low":
                insights["cognitive_style"] = "Free-flowing creativity - generates ideas but may struggle with follow-through"
            elif openness_level == "low" and conscient_level == "high":
                insights["cognitive_style"] = "Methodical practicality - prefers established procedures and concrete thinking"
            elif openness_level == "low" and conscient_level == "low":
                insights["cognitive_style"] = "Relaxed conventionality - comfortable with familiar approaches without strict organization"
        
        # Determine stress response based on neuroticism
        if "neuroticism" in big_five:
            neuroticism_level = big_five["neuroticism"]["level"]
            
            if neuroticism_level == "high":
                insights["stress_response"] = "Reactive to stressors - experiences emotions intensely and may need more recovery time"
            elif neuroticism_level == "medium":
                insights["stress_response"] = "Moderately resilient - balances emotional awareness with reasonable recovery"
            else:
                insights["stress_response"] = "Emotionally stable - maintains calm under pressure and recovers quickly"
        
        # Determine interpersonal approach based on extraversion and agreeableness
        if "extraversion" in big_five and "agreeableness" in big_five:
            extraversion_level = big_five["extraversion"]["level"]
            agreeableness_level = big_five["agreeableness"]["level"]
            
            if extraversion_level == "high" and agreeableness_level == "high":
                insights["interpersonal_approach"] = "Engaging and harmonious - builds connections easily and values group cohesion"
            elif extraversion_level == "high" and agreeableness_level == "low":
                insights["interpersonal_approach"] = "Assertive and direct - socially confident but may prioritize goals over harmony"
            elif extraversion_level == "low" and agreeableness_level == "high":
                insights["interpersonal_approach"] = "Reserved and accommodating - values harmony but may need space for reflection"
            elif extraversion_level == "low" and agreeableness_level == "low":
                insights["interpersonal_approach"] = "Independent and analytical - values autonomy and may prefer logical over emotional considerations"
        
        return insights
    
    async def _match_conditions(self, indicators: List[Dict[str, Any]], personality_traits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match indicators against condition definitions using rule-based approach"""
        matched_conditions = []
        
        # Create symptom confidence mapping for weighted calculations
        symptom_confidence = {}
        for indicator in indicators:
            indicator_text = indicator["text"].lower()
            confidence = indicator.get("confidence", 0.7)  # Default confidence if not specified
            
            # If we already have this symptom with higher confidence, skip
            if indicator_text in symptom_confidence and symptom_confidence[indicator_text] > confidence:
                continue
                
            symptom_confidence[indicator_text] = confidence
        
        for condition_id, condition in CONDITION_DEFINITIONS.items():
            # Count matching symptoms with their confidence levels
            matching_symptoms = []
            
            for symptom in condition["symptoms"]:
                best_match = None
                best_confidence = 0.0
                
                symptom_lower = symptom.lower()
                
                # Find the best matching indicator for this symptom
                for indicator in indicators:
                    indicator_text = indicator["text"].lower()
                    
                    # Check for direct or partial matches
                    is_match = False
                    match_quality = 0.0
                    
                    if symptom_lower == indicator_text:
                        # Exact match
                        is_match = True
                        match_quality = 1.0
                    elif symptom_lower in indicator_text or indicator_text in symptom_lower:
                        # Partial match
                        is_match = True
                        # Calculate match quality based on overlap
                        overlap = len(set(symptom_lower.split()) & set(indicator_text.split()))
                        total_words = max(len(symptom_lower.split()), len(indicator_text.split()))
                        match_quality = overlap / total_words if total_words > 0 else 0.5
                    
                    if is_match:
                        # Calculate effective confidence (indicator confidence * match quality)
                        effective_confidence = indicator.get("confidence", 0.7) * match_quality
                        
                        if effective_confidence > best_confidence:
                            best_match = indicator
                            best_confidence = effective_confidence
                
                # If we found a match for this symptom, add it
                if best_match:
                    matching_symptoms.append({
                        "symptom": symptom,
                        "indicator": best_match,
                        "confidence": best_confidence
                    })
            
            # Count matching voice indicators
            matching_voice = []
            for indicator in indicators:
                if indicator["source"] in ["voice_emotion", "voice_characteristics", "acoustic_analysis"]:
                    indicator_text = indicator["text"].lower()
                    for voice_ind in condition["voice_indicators"]:
                        voice_ind_lower = voice_ind.lower()
                        if voice_ind_lower == indicator_text or voice_ind_lower in indicator_text or indicator_text in voice_ind_lower:
                            matching_voice.append({
                                "voice_indicator": voice_ind,
                                "indicator": indicator,
                                "confidence": indicator.get("confidence", 0.7)
                            })
                            break
            
            # Count matching personality traits
            matching_traits = []
            if "big_five" in personality_traits and "big_five" in condition["personality_correlations"]:
                for trait, expected_level in condition["personality_correlations"]["big_five"].items():
                    if (trait in personality_traits["big_five"] and 
                        personality_traits["big_five"][trait]["level"] == expected_level):
                        
                        # Calculate confidence based on how extreme the trait value is
                        trait_score = personality_traits["big_five"][trait]["score"]
                        trait_confidence = 0.7  # Default medium confidence
                        
                        # Adjust confidence based on extremity of the trait value
                        if expected_level == "high" and trait_score >= 0.8:
                            trait_confidence = 0.9  # Very high trait score increases confidence
                        elif expected_level == "low" and trait_score <= 0.2:
                            trait_confidence = 0.9  # Very low trait score increases confidence
                        
                        matching_traits.append({
                            "trait": trait, 
                            "expected": expected_level,
                            "actual": personality_traits["big_five"][trait]["level"],
                            "confidence": trait_confidence
                        })
            
            # Calculate confidence based on matches
            # Symptoms are weighted most heavily, followed by voice indicators and personality traits
            symptom_weight = 0.65
            voice_weight = 0.20
            personality_weight = 0.15
            
            # Calculate weighted average of confidence scores
            total_symptom_confidence = sum(match["confidence"] for match in matching_symptoms)
            avg_symptom_confidence = total_symptom_confidence / len(matching_symptoms) if matching_symptoms else 0
            
            total_voice_confidence = sum(match["confidence"] for match in matching_voice)
            avg_voice_confidence = total_voice_confidence / len(matching_voice) if matching_voice else 0
            
            total_trait_confidence = sum(match["confidence"] for match in matching_traits)
            avg_trait_confidence = total_trait_confidence / len(matching_traits) if matching_traits else 0
            
            # Calculate proportional matches (with confidence weighting)
            symptom_match = len(matching_symptoms) / len(condition["symptoms"]) if condition["symptoms"] else 0
            symptom_match_weighted = symptom_match * avg_symptom_confidence
            
            voice_match = len(matching_voice) / len(condition["voice_indicators"]) if condition["voice_indicators"] else 0
            voice_match_weighted = voice_match * avg_voice_confidence
            
            trait_count = len(condition["personality_correlations"]["big_five"]) if "big_five" in condition["personality_correlations"] else 0
            personality_match = len(matching_traits) / trait_count if trait_count > 0 else 0
            personality_match_weighted = personality_match * avg_trait_confidence
            
            # Weighted confidence
            confidence = (symptom_match_weighted * symptom_weight) + (voice_match_weighted * voice_weight) + (personality_match_weighted * personality_weight)
            
            # Boost confidence if multiple data sources confirm the same condition
            data_source_bonus = 0.0
            has_symptom_evidence = symptom_match > 0
            has_voice_evidence = voice_match > 0
            has_personality_evidence = personality_match > 0
            
            # Count how many sources provide evidence
            evidence_sources = sum([has_symptom_evidence, has_voice_evidence, has_personality_evidence])
            
            # Bonus for having multiple sources of evidence
            if evidence_sources >= 2:
                data_source_bonus = 0.05 * (evidence_sources - 1)
            
            # Apply the bonus
            confidence = min(1.0, confidence + data_source_bonus)
            
            # Determine severity based on number of symptoms
            severity = "none"
            symptom_count = len(matching_symptoms)
            if symptom_count >= condition["severity_thresholds"]["severe"]:
                severity = "severe"
            elif symptom_count >= condition["severity_thresholds"]["moderate"]:
                severity = "moderate"
            elif symptom_count >= condition["severity_thresholds"]["mild"]:
                severity = "mild"
            
            # Only include if confidence is above threshold and at least mild severity
            if confidence >= self.confidence_threshold and severity != "none":
                matched_conditions.append({
                    "id": condition_id,
                    "name": condition["name"],
                    "confidence": confidence,
                    "severity": severity,
                    "matching_symptoms": matching_symptoms,
                    "matching_voice": matching_voice,
                    "matching_traits": matching_traits,
                    "symptom_match": symptom_match,
                    "voice_match": voice_match,
                    "personality_match": personality_match
                })
        
        # Sort by confidence
        matched_conditions.sort(key=lambda x: x["confidence"], reverse=True)
        return matched_conditions
    
    async def _generate_recommendations(self, conditions: List[Dict[str, Any]], personality_traits: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on conditions and personality"""
        recommendations = []
        
        # General recommendations based on condition
        for condition in conditions:
            if condition["id"] == "depression":
                recommendations.extend([
                    "Consider speaking with a mental health professional about depressive symptoms",
                    "Establish a regular physical exercise routine",
                    "Maintain social connections even when you don't feel like it",
                    "Create a structured daily routine",
                    "Practice mindfulness meditation to stay present",
                    "Keep a gratitude journal noting positive experiences",
                    "Set small, achievable goals to build momentum"
                ])
            elif condition["id"] == "anxiety":
                recommendations.extend([
                    "Practice deep breathing exercises when feeling anxious",
                    "Consider speaking with a mental health professional about anxiety management",
                    "Reduce caffeine and alcohol consumption",
                    "Establish a regular sleep schedule",
                    "Try progressive muscle relaxation for physical tension",
                    "Challenge catastrophic thinking with evidence-based alternatives",
                    "Schedule specific 'worry time' rather than worrying throughout the day"
                ])
            elif condition["id"] == "stress":
                recommendations.extend([
                    "Identify and reduce sources of stress when possible",
                    "Practice time management techniques",
                    "Set boundaries in work and personal life",
                    "Take regular breaks during work",
                    "Engage in enjoyable activities daily",
                    "Use relaxation techniques like progressive muscle relaxation",
                    "Connect with supportive people"
                ])
            elif condition["id"] == "ptsd":
                recommendations.extend([
                    "Consider trauma-focused therapy with a mental health professional",
                    "Practice grounding techniques when feeling triggered",
                    "Establish predictable routines to create a sense of safety",
                    "Learn to recognize trauma triggers and develop coping plans",
                    "Join a support group for people with similar experiences",
                    "Practice self-compassion and patience in your healing journey"
                ])
            elif condition["id"] == "bipolar":
                recommendations.extend([
                    "Consult with a psychiatrist for comprehensive evaluation and treatment",
                    "Maintain a consistent sleep schedule",
                    "Track mood patterns to identify triggers and early warning signs",
                    "Create a crisis plan for episodes of mania or depression",
                    "Build a support network aware of your condition",
                    "Avoid alcohol and recreational drugs",
                    "Establish regular daily routines"
                ])
        
        # Personality-specific recommendations
        if "big_five" in personality_traits:
            big_five = personality_traits["big_five"]
            
            # High neuroticism recommendations
            if "neuroticism" in big_five and big_five["neuroticism"]["level"] == "high":
                recommendations.extend([
                    "Practice cognitive restructuring to challenge negative thought patterns",
                    "Keep a thought journal to identify cognitive distortions",
                    "Learn to recognize and name emotions as they arise",
                    "Build resilience through gradual exposure to mild stressors",
                    "Develop a personalized emotional regulation toolkit"
                ])
            
            # Low extraversion recommendations
            if "extraversion" in big_five and big_five["extraversion"]["level"] == "low":
                recommendations.extend([
                    "Balance social energy with needed alone time for recovery",
                    "Set small, achievable goals for social interaction",
                    "Find social activities aligned with your interests",
                    "Practice social skills in low-pressure environments",
                    "Honor your need for solitude while maintaining key relationships"
                ])
            
            # High conscientiousness recommendations
            if "conscientiousness" in big_five and big_five["conscientiousness"]["level"] == "high":
                recommendations.extend([
                    "Practice self-compassion when you don't meet your own high standards",
                    "Schedule regular downtime to prevent burnout",
                    "Delegate tasks when appropriate",
                    "Balance achievement with enjoyment and presence",
                    "Recognize when perfectionism becomes counterproductive"
                ])
            
            # Low openness recommendations
            if "openness" in big_five and big_five["openness"]["level"] == "low":
                recommendations.extend([
                    "Gradually introduce small, manageable changes to your routine",
                    "Explore new experiences in familiar contexts",
                    "Build on existing strengths when facing challenges",
                    "Recognize the value of stability and consistency"
                ])
            
            # Low agreeableness recommendations
            if "agreeableness" in big_five and big_five["agreeableness"]["level"] == "low":
                recommendations.extend([
                    "Practice active listening in conversations",
                    "Look for common ground in disagreements",
                    "Consider others' perspectives explicitly",
                    "Balance assertiveness with cooperation"
                ])
        
        # Custom LLM-generated recommendations if available
        if self.llm and conditions and personality_traits:
            try:
                # Get top condition
                primary_condition = conditions[0]
                condition_name = primary_condition["name"]
                severity = primary_condition["severity"]
                
                # Extract personality information
                personality_info = ""
                if "big_five" in personality_traits:
                    big_five = personality_traits["big_five"]
                    personality_info = "Personality traits: " + ", ".join([
                        f"{trait}: {data['level']}" for trait, data in big_five.items()
                    ])
                
                # Create prompt for personalized recommendations
                prompt = f"""
                Generate 3 personalized mental health recommendations for someone showing {severity} indicators of {condition_name}.
                
                {personality_info}
                
                Make recommendations specific, actionable, and evidence-based. Format as a simple bullet list.
                Each recommendation should be one sentence, starting with an action verb.
                """
                
                # Get response from LLM
                response = await self.llm.agenerate(prompt)
                response_text = response.get("choices", [{}])[0].get("text", "")
                
                # Extract recommendations from the response
                import re
                llm_recommendations = []
                
                # Look for bullet points, numbered lists, or line-by-line recommendations
                lines = response_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Remove bullets, numbers, or other list markers
                    cleaned_line = re.sub(r'^[\s•\-\*\d]+\.?\s*', '', line)
                    
                    if cleaned_line and len(cleaned_line) > 10:  # Ensure it's not just a heading
                        llm_recommendations.append(cleaned_line)
                
                # Add LLM-generated recommendations
                recommendations.extend(llm_recommendations)
                
            except Exception as e:
                logger.error(f"Error generating LLM recommendations: {str(e)}")
        
        # Remove duplicates and limit number of recommendations
        unique_recommendations = []
        for rec in recommendations:
            # Normalize recommendation for comparison
            rec_lower = rec.lower()
            # Check if similar recommendation already exists
            if not any(self._recommendation_similarity(rec_lower, existing.lower()) > 0.7 for existing in unique_recommendations):
                unique_recommendations.append(rec)
        
        # Limit to a reasonable number
        return unique_recommendations[:10]
    
    def _recommendation_similarity(self, rec1: str, rec2: str) -> float:
        """Calculate similarity between two recommendations to avoid duplicates"""
        # Simple word overlap metric
        words1 = set(rec1.split())
        words2 = set(rec2.split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        total = max(len(words1), len(words2))
        
        return overlap / total
    
    def _generate_evidence_summary(self, conditions: List[Dict[str, Any]], indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the evidence used for diagnosis"""
        evidence = {
            "total_indicators": len(indicators),
            "indicators_by_source": {},
            "key_symptoms": []
        }
        
        # Count indicators by source
        source_counts = defaultdict(int)
        for indicator in indicators:
            source = indicator.get("source", "unknown")
            source_counts[source] += 1
        
        evidence["indicators_by_source"] = dict(source_counts)
        
        # Add key symptoms per condition
        for condition in conditions:
            condition_symptoms = []
            for match in condition["matching_symptoms"]:
                symptom = match["symptom"]
                confidence = match.get("confidence", 0.0)
                if symptom not in [s["symptom"] for s in condition_symptoms]:
                    condition_symptoms.append({
                        "symptom": symptom,
                        "confidence": confidence
                    })
            
            # Sort by confidence
            condition_symptoms.sort(key=lambda x: x["confidence"], reverse=True)
            
            evidence["key_symptoms"].append({
                "condition": condition["name"],
                "symptoms": [s["symptom"] for s in condition_symptoms[:5]]  # Top 5 symptoms
            })
        
        return evidence
    
    def _assess_data_quality(self, 
                            conversation_data: Optional[Dict[str, Any]],
                            voice_emotion_data: Optional[Dict[str, Any]],
                            personality_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality and completeness of the input data"""
        quality = {
            "overall_quality": "low",
            "conversation_data": "missing",
            "voice_emotion_data": "missing",
            "personality_data": "missing",
            "improvement_suggestions": []
        }
        
        # Assess conversation data
        if conversation_data:
            if "text" in conversation_data and len(conversation_data["text"]) > 100:
                quality["conversation_data"] = "good"
            elif "text" in conversation_data and len(conversation_data["text"]) > 20:
                quality["conversation_data"] = "limited"
                quality["improvement_suggestions"].append("Longer conversation would improve assessment accuracy")
            else:
                quality["conversation_data"] = "poor"
                quality["improvement_suggestions"].append("Very limited conversation data available")
        else:
            quality["improvement_suggestions"].append("No conversation data provided")
        
        # Assess voice emotion data
        if voice_emotion_data:
            has_emotions = "emotions" in voice_emotion_data and voice_emotion_data["emotions"]
            has_characteristics = "characteristics" in voice_emotion_data and voice_emotion_data["characteristics"]
            
            if has_emotions and has_characteristics:
                quality["voice_emotion_data"] = "good"
            elif has_emotions or has_characteristics:
                quality["voice_emotion_data"] = "limited"
                quality["improvement_suggestions"].append("Voice emotion data is incomplete")
            else:
                quality["voice_emotion_data"] = "poor"
                quality["improvement_suggestions"].append("Voice emotion data has limited useful information")
        else:
            quality["improvement_suggestions"].append("No voice emotion data provided")
        
        # Assess personality data
        if personality_data:
            has_big_five = "big_five" in personality_data and personality_data["big_five"]
            has_mbti = "mbti" in personality_data and personality_data["mbti"]
            
            if has_big_five and has_mbti:
                quality["personality_data"] = "good"
            elif has_big_five or has_mbti:
                quality["personality_data"] = "limited"
                quality["improvement_suggestions"].append("Personality data is incomplete")
            else:
                quality["personality_data"] = "poor"
                quality["improvement_suggestions"].append("Personality data has limited useful information")
        else:
            quality["improvement_suggestions"].append("No personality data provided")
        
        # Determine overall quality
        quality_scores = {
            "missing": 0,
            "poor": 1,
            "limited": 2,
            "good": 3
        }
        
        total_score = (
            quality_scores[quality["conversation_data"]] +
            quality_scores[quality["voice_emotion_data"]] +
            quality_scores[quality["personality_data"]]
        )
        
        if total_score >= 7:
            quality["overall_quality"] = "good"
        elif total_score >= 4:
            quality["overall_quality"] = "moderate"
        else:
            quality["overall_quality"] = "poor"
            
        return quality
    
    async def _check_diagnosis_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check for a cached diagnosis"""
        if not self.use_vector_cache or not self.vector_store:
            return None
        
        try:
            similar_results = await self.vector_store.find_similar_results(cache_key, threshold=0.9, k=1)
            if similar_results:
                cached_result = similar_results[0].get('parsed_content')
                if cached_result and isinstance(cached_result, dict) and cached_result.get('success'):
                    # Add cache information
                    cached_result["from_cache"] = True
                    cached_result["cache_key"] = cache_key
                    return cached_result
        except Exception as e:
            logger.error(f"Error checking diagnosis cache: {str(e)}")
        
        return None
    
    async def _cache_diagnosis(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """Cache a diagnosis result"""
        if not self.use_vector_cache or not self.vector_store:
            return False
        
        try:
            await self.vector_store.add_processed_result(cache_key, result)
            logger.info(f"Cached diagnosis with key: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Error caching diagnosis: {str(e)}")
            return False

    def get_formatter(self, condition_id: str, severity: str) -> callable:
        """Get response formatter for specific condition and severity"""
        if condition_id in RESPONSE_TEMPLATES and severity in RESPONSE_TEMPLATES[condition_id]:
            template = RESPONSE_TEMPLATES[condition_id][severity]
            
            def formatter(symptoms: List[str]) -> str:
                # Format the symptoms as a readable list
                if not symptoms:
                    symptom_text = "various indicators"
                elif len(symptoms) == 1:
                    symptom_text = symptoms[0]
                elif len(symptoms) == 2:
                    symptom_text = f"{symptoms[0]} and {symptoms[1]}"
                else:
                    symptom_text = ", ".join(symptoms[:-1]) + f", and {symptoms[-1]}"
                
                return template.format(symptoms=symptom_text)
                
            return formatter
        
        # Fallback generic formatter
        def generic_formatter(symptoms: List[str]) -> str:
            return f"Based on our conversation, I've noticed patterns that may relate to {severity} {condition_id}."
            
        return generic_formatter

def create_diagnosis_module(use_agentic_rag: bool = True, use_cache: bool = True) -> ComprehensiveDiagnosisModule:
    """
    Factory function to create a properly configured diagnosis module
    
    Args:
        use_agentic_rag: Whether to use AgenticRAG for enhanced diagnosis
        use_cache: Whether to use vector caching for diagnosis results
        
    Returns:
        Configured ComprehensiveDiagnosisModule instance
    """
    # Initialize AgenticRAG if requested
    agentic_rag = None
    if use_agentic_rag:
        try:
            from utils.agentic_rag import AgenticRAG
            agentic_rag = AgenticRAG.create_for_mental_health()
            logger.info("Successfully initialized AgenticRAG for diagnosis")
        except Exception as e:
            logger.warning(f"Could not initialize AgenticRAG: {str(e)}")
    
    # Create and return the module
    return ComprehensiveDiagnosisModule(
        agentic_rag=agentic_rag,
        use_vector_cache=use_cache
    )