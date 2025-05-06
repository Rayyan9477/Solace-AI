"""
Enhanced Diagnosis Module

This module extends the integrated diagnosis approach with advanced techniques:
1. Vector similarity for more accurate pattern matching
2. Multi-modal integration with weighted confidence scores
3. Real-time research integration for up-to-date clinical insights
4. Temporal analysis of symptom progression
5. Personalized recommendation engine

The goal is to provide a more nuanced and accurate assessment while maintaining
the focus on supporting but not replacing professional clinical evaluation.
"""

import os
import json
import numpy as np
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from pathlib import Path
import faiss

# Project imports
from database.vector_store import VectorStore
from utils.agentic_rag import AgenticRAG
from config.settings import AppConfig
from .integrated_diagnosis import DiagnosisModule, CONDITION_DEFINITIONS

# Configure logging
logger = logging.getLogger(__name__)

# Extended condition definitions with more detailed symptom patterns
EXTENDED_CONDITIONS = {
    "depression": {
        "subtypes": {
            "major_depressive": {
                "name": "Major Depressive Disorder",
                "key_symptoms": ["persistent sadness", "loss of interest", "feelings of worthlessness"],
                "duration_requirement": 14,  # days
                "min_symptom_count": 5,
                "exclusion_criteria": ["manic episodes", "substance-induced"]
            },
            "persistent_depressive": {
                "name": "Persistent Depressive Disorder",
                "key_symptoms": ["persistent low mood", "fatigue", "low self-esteem"],
                "duration_requirement": 730,  # 2 years in days
                "min_symptom_count": 2,
                "exclusion_criteria": ["major depressive episodes within first 2 years"]
            },
            "seasonal_affective": {
                "name": "Seasonal Affective Disorder",
                "key_symptoms": ["winter depression", "increased sleep", "weight gain", "social withdrawal"],
                "seasonal_pattern": True,
                "min_symptom_count": 3,
                "exclusion_criteria": []
            }
        },
        "comorbidities": ["anxiety", "stress", "insomnia"],
        "risk_factors": ["family history", "trauma", "chronic illness", "substance abuse"]
    },
    "anxiety": {
        "subtypes": {
            "generalized_anxiety": {
                "name": "Generalized Anxiety Disorder",
                "key_symptoms": ["excessive worry", "difficulty controlling worry", "restlessness"],
                "duration_requirement": 180,  # 6 months in days
                "min_symptom_count": 3,
                "exclusion_criteria": ["substance-induced", "medical condition"]
            },
            "social_anxiety": {
                "name": "Social Anxiety Disorder",
                "key_symptoms": ["fear of social situations", "fear of judgment", "avoidance of social situations"],
                "duration_requirement": 180,  # 6 months in days
                "min_symptom_count": 2,
                "exclusion_criteria": []
            },
            "panic": {
                "name": "Panic Disorder",
                "key_symptoms": ["recurring panic attacks", "fear of panic attacks", "physical symptoms during panic"],
                "duration_requirement": 30,  # 1 month in days
                "min_symptom_count": 2,
                "exclusion_criteria": []
            }
        },
        "comorbidities": ["depression", "insomnia", "substance abuse"],
        "risk_factors": ["family history", "trauma", "personality traits", "medical conditions"]
    },
    "stress": {
        "subtypes": {
            "acute_stress": {
                "name": "Acute Stress Disorder",
                "key_symptoms": ["exposure to trauma", "intrusive thoughts", "negative mood", "arousal symptoms"],
                "duration_requirement": [3, 30],  # between 3 days and 1 month
                "min_symptom_count": 9,
                "exclusion_criteria": ["substance-induced"]
            },
            "post_traumatic_stress": {
                "name": "Post-Traumatic Stress Disorder",
                "key_symptoms": ["exposure to trauma", "intrusive memories", "avoidance", "hyperarousal"],
                "duration_requirement": 30,  # more than 1 month
                "min_symptom_count": 6,
                "exclusion_criteria": []
            },
            "adjustment": {
                "name": "Adjustment Disorder",
                "key_symptoms": ["emotional distress", "functional impairment", "identifiable stressor"],
                "duration_requirement": [0, 90],  # up to 3 months after stressor
                "min_symptom_count": 2,
                "exclusion_criteria": ["meets criteria for another disorder"]
            }
        },
        "comorbidities": ["depression", "anxiety", "substance abuse"],
        "risk_factors": ["major life changes", "ongoing stressors", "trauma history", "limited support"]
    },
    "bipolar": {
        "name": "Bipolar Disorder",
        "symptoms": [
            "mood swings", "periods of depression", "periods of elevated mood", "increased energy", 
            "decreased need for sleep", "racing thoughts", "impulsive behavior", "grandiosity",
            "irritability", "distractibility", "flight of ideas", "risky behavior"
        ],
        "voice_indicators": ["rapid speech", "loud volume", "pressured speech", "flight of ideas"],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high", 
                "extraversion": "variable (high during mania, low during depression)",
                "openness": "high",
                "agreeableness": "variable", 
                "conscientiousness": "variable (low during mania)"
            }
        },
        "severity_thresholds": {
            "mild": 3,
            "moderate": 5,
            "severe": 7
        },
        "subtypes": {
            "bipolar_1": {
                "name": "Bipolar I Disorder",
                "key_symptoms": ["manic episodes", "may have depressive episodes"],
                "duration_requirement": 7,  # at least 7 days of mania
                "min_symptom_count": 3,
                "exclusion_criteria": ["substance-induced", "medical condition"]
            },
            "bipolar_2": {
                "name": "Bipolar II Disorder",
                "key_symptoms": ["hypomanic episodes", "depressive episodes"],
                "duration_requirement": 4,  # at least 4 days of hypomania
                "min_symptom_count": 3,
                "exclusion_criteria": ["no full manic episodes"]
            },
            "cyclothymic": {
                "name": "Cyclothymic Disorder",
                "key_symptoms": ["numerous periods of hypomanic and depressive symptoms"],
                "duration_requirement": 730,  # 2 years
                "min_symptom_count": 2,
                "exclusion_criteria": ["no major depressive, manic, or hypomanic episodes in first 2 years"]
            }
        },
        "comorbidities": ["anxiety disorders", "substance use disorders", "ADHD"],
        "risk_factors": ["family history", "high stress", "trauma", "sleep disruptions"]
    },
    "insomnia": {
        "name": "Insomnia Disorder",
        "symptoms": [
            "difficulty falling asleep", "difficulty staying asleep", "early morning awakening", 
            "non-restorative sleep", "daytime fatigue", "mood disturbances", "difficulty concentrating",
            "worry about sleep", "impaired social functioning", "irritability"
        ],
        "voice_indicators": ["tired tone", "slower speech", "lower energy", "irritable tone"],
        "personality_correlations": {
            "big_five": {
                "neuroticism": "high",
                "extraversion": "variable",
                "openness": "variable",
                "agreeableness": "low when sleep-deprived",
                "conscientiousness": "variable"
            }
        },
        "severity_thresholds": {
            "mild": 3,
            "moderate": 5,
            "severe": 7
        },
        "subtypes": {
            "acute_insomnia": {
                "name": "Acute Insomnia",
                "key_symptoms": ["short-term sleep difficulties", "identifiable stressor"],
                "duration_requirement": [0, 90],  # less than 3 months
                "min_symptom_count": 2,
                "exclusion_criteria": []
            },
            "chronic_insomnia": {
                "name": "Chronic Insomnia",
                "key_symptoms": ["long-term difficulty sleeping", "occurs at least 3 nights per week"],
                "duration_requirement": 90,  # 3 months or more
                "min_symptom_count": 3,
                "exclusion_criteria": []
            }
        },
        "comorbidities": ["depression", "anxiety", "substance use disorders"],
        "risk_factors": ["stress", "irregular sleep schedule", "poor sleep habits", "medical conditions"]
    }
}

# Symptom severity mapping based on linguistic markers
SEVERITY_MARKERS = {
    "mild": ["a bit", "slightly", "somewhat", "occasionally", "some", "a little"],
    "moderate": ["quite", "rather", "moderately", "frequently", "often", "regularly"],
    "severe": ["extremely", "very", "severely", "constantly", "always", "overwhelming", "unbearable"]
}

# Time-related phrases for duration assessment
TIME_MARKERS = {
    "recent": ["recently", "lately", "past few days", "this week", "today", "since yesterday"],
    "short_term": ["past few weeks", "this month", "for weeks", "several weeks", "a month"],
    "medium_term": ["for months", "several months", "half a year", "past few months"],
    "long_term": ["for years", "always been", "my whole life", "as long as I can remember", "chronic"]
}

class EnhancedDiagnosisModule(DiagnosisModule):
    """
    Enhanced module that builds on the integrated diagnosis approach with
    advanced techniques for more nuanced mental health assessment.
    """
    
    def __init__(
        self, 
        agentic_rag: Optional[AgenticRAG] = None, 
        use_vector_cache: bool = True,
        enable_temporal_analysis: bool = True,
        enable_research_integration: bool = False,
        confidence_threshold: float = 0.35,
        symptom_tracking_days: int = 30
    ):
        """
        Initialize the enhanced diagnosis module
        
        Args:
            agentic_rag: Optional AgenticRAG instance for enhanced reasoning
            use_vector_cache: Whether to use vector caching for previous diagnoses
            enable_temporal_analysis: Whether to analyze symptom changes over time
            enable_research_integration: Whether to integrate recent research findings
            confidence_threshold: Minimum confidence threshold for condition matching
            symptom_tracking_days: Number of days to track symptom patterns
        """
        # Initialize the base module
        super().__init__(agentic_rag, use_vector_cache)
        
        # Enhanced module settings
        self.enable_temporal_analysis = enable_temporal_analysis
        self.enable_research_integration = enable_research_integration
        self.confidence_threshold = confidence_threshold
        self.symptom_tracking_days = symptom_tracking_days
        
        # Initialize the embedded vector model for symptom similarity
        self.symptom_vectors = None
        self.symptoms_index = None
        self._initialize_symptom_vectors()
        
        # Initialize the temporal tracking data structure
        self.temporal_data = {}
        
        # Extended condition definitions with more clinical detail
        self.extended_conditions = EXTENDED_CONDITIONS
        
        # Enhanced recommendation engine
        self.recommendation_engine = RecommendationEngine()
        
        # Create a research integrator if enabled
        self.research_integrator = None
        if self.enable_research_integration:
            self.research_integrator = ResearchIntegrator()
            
        # Set up the vector embedder for symptoms
        self._setup_vector_embedder()
    
    def _setup_vector_embedder(self):
        """Set up the vector embedder for symptom similarity"""
        try:
            import sentence_transformers
            self.embedder = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully initialized sentence transformer for symptom similarity")
        except ImportError:
            logger.warning("Sentence Transformers not available. Using fallback similarity method.")
            self.embedder = None
    
    def _initialize_symptom_vectors(self):
        """Initialize the symptom vectors for similarity matching"""
        try:
            # Collect all symptoms from condition definitions
            all_symptoms = set()
            for condition in CONDITION_DEFINITIONS.values():
                all_symptoms.update(condition["symptoms"])
            
            # Also add symptoms from extended conditions
            for condition in EXTENDED_CONDITIONS.values():
                if "symptoms" in condition:
                    all_symptoms.update(condition["symptoms"])
                    
                # Add symptoms from subtypes if available
                if "subtypes" in condition:
                    for subtype in condition["subtypes"].values():
                        if "key_symptoms" in subtype:
                            all_symptoms.update(subtype["key_symptoms"])
            
            self.all_symptoms = list(all_symptoms)
            
            if self.embedder:
                # Create embeddings for all symptoms
                self.symptom_vectors = self.embedder.encode(self.all_symptoms)
                
                # Create a FAISS index for fast similarity search
                d = self.symptom_vectors.shape[1]  # Embedding dimension
                self.symptoms_index = faiss.IndexFlatL2(d)
                self.symptoms_index.add(np.array(self.symptom_vectors).astype('float32'))
                
                logger.info(f"Initialized symptom vectors with {len(self.all_symptoms)} symptoms")
            else:
                logger.info("Using fallback symptom similarity without embeddings")
                
        except Exception as e:
            logger.error(f"Error initializing symptom vectors: {str(e)}")
            self.symptom_vectors = None
            self.symptoms_index = None
    
    async def generate_enhanced_diagnosis(
        self,
        conversation_data: Optional[Dict[str, Any]] = None,
        voice_emotion_data: Optional[Dict[str, Any]] = None,
        personality_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        previous_diagnoses: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate an enhanced diagnostic assessment with more detailed analysis
        
        Args:
            conversation_data: Data extracted from chatbot conversations
            voice_emotion_data: Emotional patterns detected from voice analysis
            personality_data: Results from personality assessments
            user_id: Optional user identifier for tracking and caching
            session_id: Optional session identifier for temporal analysis
            previous_diagnoses: List of previous diagnostic assessments
            
        Returns:
            Comprehensive enhanced diagnostic assessment
        """
        start_time = datetime.now()
        
        # Start with the base diagnosis
        base_diagnosis = await super().generate_diagnosis(
            conversation_data=conversation_data,
            voice_emotion_data=voice_emotion_data,
            personality_data=personality_data,
            user_id=user_id
        )
        
        # Create enhanced result structure building on the base
        enhanced_result = {
            **base_diagnosis,
            "enhanced": True,
            "clinical_details": {},
            "confidence_breakdown": {},
            "temporal_analysis": {},
            "differential_diagnosis": [],
            "enhanced_recommendations": [],
            "explanation": "",
            "research_insights": []
        }
        
        # Skip enhancement if base diagnosis failed
        if not base_diagnosis.get("success", False):
            enhanced_result["enhancement_error"] = "Base diagnosis failed, cannot enhance"
            return enhanced_result
        
        try:
            # 1. Add detailed clinical assessment
            if base_diagnosis.get("conditions", []):
                enhanced_result["clinical_details"] = await self._generate_clinical_details(
                    base_diagnosis["conditions"],
                    conversation_data
                )
            
            # 2. Calculate more detailed confidence scores with breakdown
            enhanced_result["confidence_breakdown"] = self._calculate_confidence_breakdown(
                base_diagnosis["conditions"],
                conversation_data,
                voice_emotion_data,
                personality_data
            )
            
            # 3. Generate differential diagnoses (alternative explanations)
            enhanced_result["differential_diagnosis"] = self._generate_differential_diagnosis(
                base_diagnosis["conditions"],
                conversation_data
            )
            
            # 4. Add temporal analysis if enabled and previous data available
            if self.enable_temporal_analysis and previous_diagnoses:
                enhanced_result["temporal_analysis"] = self._analyze_temporal_patterns(
                    base_diagnosis,
                    previous_diagnoses,
                    user_id
                )
            
            # 5. Generate enhanced personalized recommendations
            enhanced_result["enhanced_recommendations"] = await self.recommendation_engine.generate_recommendations(
                conditions=base_diagnosis.get("conditions", []),
                personality_traits=personality_data,
                previous_diagnoses=previous_diagnoses,
                user_id=user_id
            )
            
            # 6. Add research insights if enabled
            if self.enable_research_integration and self.research_integrator:
                condition_names = [c["name"] for c in base_diagnosis.get("conditions", [])]
                if condition_names:
                    enhanced_result["research_insights"] = await self.research_integrator.get_insights(condition_names)
            
            # 7. Use AgenticRAG for enhanced explanation if available
            if self.agentic_rag and conversation_data and "text" in conversation_data:
                enhanced_explanation = await self._generate_enhanced_explanation(
                    base_diagnosis,
                    conversation_data,
                    enhanced_result
                )
                enhanced_result["explanation"] = enhanced_explanation
            
            # 8. Store temporal data for future analysis if user_id provided
            if user_id and self.enable_temporal_analysis:
                self._store_temporal_data(user_id, enhanced_result)
            
            # Mark success and calculate processing time
            enhanced_result["success"] = True
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            enhanced_result["processing_time_seconds"] = processing_time
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced diagnosis generation: {str(e)}")
            # Return the base diagnosis with error information
            enhanced_result["enhancement_error"] = str(e)
            enhanced_result["fallback_to_base"] = True
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            enhanced_result["processing_time_seconds"] = processing_time
            
            return enhanced_result
    
    async def _generate_clinical_details(
        self, 
        conditions: List[Dict[str, Any]],
        conversation_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate detailed clinical assessment for matched conditions"""
        clinical_details = {}
        
        for condition in conditions:
            condition_id = next((cid for cid, cdef in CONDITION_DEFINITIONS.items() 
                                if cdef["name"] == condition["name"]), None)
            
            if not condition_id:
                continue
                
            # Get basic details from condition definition
            base_details = CONDITION_DEFINITIONS.get(condition_id, {})
            
            # Check if we have extended details for this condition
            extended_details = self.extended_conditions.get(condition_id, {})
            
            # Extract text for duration analysis if conversation data available
            duration_estimate = "unknown"
            if conversation_data and "text" in conversation_data:
                duration_estimate = self._estimate_duration(conversation_data["text"])
            
            # Check for subtypes
            matched_subtype = None
            subtype_confidence = 0.0
            
            if "subtypes" in extended_details and conversation_data:
                subtype_match = self._match_condition_subtype(
                    condition_id,
                    extended_details["subtypes"],
                    conversation_data,
                    duration_estimate
                )
                
                if subtype_match:
                    matched_subtype = subtype_match["subtype"]
                    subtype_confidence = subtype_match["confidence"]
            
            # Check for comorbidities
            comorbidities = []
            if "comorbidities" in extended_details:
                # Check if other detected conditions match known comorbidities
                for other_condition in conditions:
                    if other_condition["name"] != condition["name"]:
                        other_id = next((cid for cid, cdef in CONDITION_DEFINITIONS.items() 
                                        if cdef["name"] == other_condition["name"]), None)
                        if other_id and other_id in extended_details["comorbidities"]:
                            comorbidities.append({
                                "condition": other_condition["name"],
                                "confidence": other_condition["confidence"]
                            })
            
            # Compile the clinical details
            clinical_details[condition["name"]] = {
                "severity": condition["severity"],
                "confidence": condition["confidence"],
                "duration_estimate": duration_estimate,
                "matched_subtype": matched_subtype,
                "subtype_confidence": subtype_confidence,
                "comorbidities": comorbidities,
                "risk_factors": extended_details.get("risk_factors", [])
            }
        
        return clinical_details
    
    def _match_condition_subtype(
        self,
        condition_id: str,
        subtypes: Dict[str, Any],
        conversation_data: Dict[str, Any],
        duration_estimate: str
    ) -> Optional[Dict[str, Any]]:
        """Match conversation data to specific subtypes of a condition"""
        if not conversation_data or "text" not in conversation_data:
            return None
            
        text = conversation_data["text"].lower()
        best_match = None
        best_confidence = 0.0
        
        for subtype_id, subtype in subtypes.items():
            # Count matching key symptoms
            symptom_matches = 0
            total_symptoms = len(subtype["key_symptoms"])
            
            for symptom in subtype["key_symptoms"]:
                if symptom.lower() in text:
                    symptom_matches += 1
            
            # Calculate basic confidence based on symptom matches
            if total_symptoms > 0:
                confidence = symptom_matches / total_symptoms
                
                # Check if minimum symptom count is met
                if symptom_matches < subtype.get("min_symptom_count", 1):
                    confidence *= 0.5  # Penalize if minimum symptoms not met
                
                # Check duration requirements if specified
                if "duration_requirement" in subtype:
                    duration_req = subtype["duration_requirement"]
                    if isinstance(duration_req, int):
                        # Single duration requirement (minimum days)
                        if duration_estimate == "recent" and duration_req > 14:
                            confidence *= 0.6
                        elif duration_estimate == "short_term" and duration_req > 60:
                            confidence *= 0.7
                        elif duration_estimate == "medium_term" and duration_req > 180:
                            confidence *= 0.8
                        elif duration_estimate == "long_term":
                            confidence *= 1.1  # Bonus for chronic conditions
                    elif isinstance(duration_req, list) and len(duration_req) == 2:
                        # Range of days [min, max]
                        min_days, max_days = duration_req
                        if duration_estimate == "recent" and min_days > 14:
                            confidence *= 0.6
                        elif duration_estimate == "short_term" and (min_days > 60 or max_days < 7):
                            confidence *= 0.7
                        elif duration_estimate == "medium_term" and (min_days > 180 or max_days < 30):
                            confidence *= 0.8
                        elif duration_estimate == "long_term" and max_days < 180:
                            confidence *= 0.7
                
                # Check exclusion criteria
                for criterion in subtype.get("exclusion_criteria", []):
                    if criterion.lower() in text:
                        confidence *= 0.3  # Heavily penalize if exclusion criteria present
                
                # Update best match if this is better
                if confidence > best_confidence and confidence > 0.3:
                    best_confidence = confidence
                    best_match = {
                        "subtype": subtype["name"],
                        "confidence": confidence,
                        "matched_symptoms": symptom_matches,
                        "required_symptoms": subtype.get("min_symptom_count", 1)
                    }
        
        return best_match
    
    def _estimate_duration(self, text: str) -> str:
        """Estimate the duration of symptoms from conversation text"""
        text = text.lower()
        
        # Check for time-related phrases in each category
        for category, phrases in TIME_MARKERS.items():
            for phrase in phrases:
                if phrase in text:
                    return category
        
        # Default to recent if no time markers found
        return "recent"
    
    def _calculate_confidence_breakdown(
        self,
        conditions: List[Dict[str, Any]],
        conversation_data: Optional[Dict[str, Any]],
        voice_emotion_data: Optional[Dict[str, Any]],
        personality_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate detailed confidence breakdown for matched conditions"""
        confidence_breakdown = {}
        
        for condition in conditions:
            condition_name = condition["name"]
            confidence_breakdown[condition_name] = {
                "overall": condition["confidence"],
                "factors": {
                    "symptom_match": 0.0,
                    "voice_emotion": 0.0,
                    "personality_correlation": 0.0,
                    "symptom_severity": 0.0,
                    "symptom_duration": 0.0,
                    "symptom_consistency": 0.0
                },
                "weights": {
                    "symptom_match": 0.6,
                    "voice_emotion": 0.15,
                    "personality_correlation": 0.1,
                    "symptom_severity": 0.05,
                    "symptom_duration": 0.05,
                    "symptom_consistency": 0.05
                }
            }
            
            # Get the condition ID
            condition_id = next((cid for cid, cdef in CONDITION_DEFINITIONS.items() 
                                if cdef["name"] == condition_name), None)
            
            if not condition_id:
                continue
            
            # Calculate symptom match confidence
            if conversation_data:
                symptoms = CONDITION_DEFINITIONS[condition_id]["symptoms"]
                text = conversation_data.get("text", "").lower()
                
                matched_symptoms = sum(1 for symptom in symptoms if symptom.lower() in text)
                if symptoms:
                    confidence_breakdown[condition_name]["factors"]["symptom_match"] = matched_symptoms / len(symptoms)
                
                # Calculate symptom severity
                severity_score = 0.0
                severity_count = 0
                for marker_type, markers in SEVERITY_MARKERS.items():
                    marker_weight = 0.3 if marker_type == "mild" else 0.6 if marker_type == "moderate" else 0.9
                    for marker in markers:
                        if marker in text:
                            severity_score += marker_weight
                            severity_count += 1
                
                if severity_count > 0:
                    confidence_breakdown[condition_name]["factors"]["symptom_severity"] = severity_score / severity_count
                
                # Calculate symptom duration confidence
                duration_estimate = self._estimate_duration(text)
                duration_scores = {
                    "recent": 0.3,
                    "short_term": 0.6,
                    "medium_term": 0.8,
                    "long_term": 1.0
                }
                confidence_breakdown[condition_name]["factors"]["symptom_duration"] = duration_scores.get(duration_estimate, 0.5)
            
            # Calculate voice emotion confidence
            if voice_emotion_data and "emotions" in voice_emotion_data:
                emotions = voice_emotion_data["emotions"]
                
                # Different conditions correlate with different emotions
                emotion_correlations = {
                    "depression": ["sad", "flat", "tired"],
                    "anxiety": ["anxious", "tense", "nervous", "afraid"],
                    "stress": ["tense", "overwhelmed", "irritable"],
                    "bipolar": ["variable", "irritable", "elevated", "energetic"],
                    "insomnia": ["tired", "irritable", "flat"]
                }
                
                if condition_id in emotion_correlations:
                    relevant_emotions = emotion_correlations[condition_id]
                    emotion_score = sum(emotions.get(emotion, 0) for emotion in relevant_emotions if emotion in emotions)
                    max_possible = len(relevant_emotions)
                    
                    if max_possible > 0:
                        confidence_breakdown[condition_name]["factors"]["voice_emotion"] = emotion_score / max_possible
            
            # Calculate personality correlation confidence
            if personality_data and "big_five" in personality_data:
                big_five = personality_data["big_five"]
                
                # Get expected personality traits for the condition
                condition_def = CONDITION_DEFINITIONS.get(condition_id, {})
                if "personality_correlations" in condition_def and "big_five" in condition_def["personality_correlations"]:
                    expected_traits = condition_def["personality_correlations"]["big_five"]
                    
                    # Count matching traits
                    matches = 0
                    total_traits = 0
                    
                    for trait, expected_level in expected_traits.items():
                        if trait in big_five:
                            total_traits += 1
                            
                            # Skip traits with "variable" expectation
                            if "variable" in expected_level:
                                continue
                                
                            actual_level = None
                            trait_score = big_five[trait]
                            
                            if isinstance(trait_score, dict) and "level" in trait_score:
                                actual_level = trait_score["level"]
                            elif isinstance(trait_score, (int, float)):
                                if trait_score >= 0.7:
                                    actual_level = "high"
                                elif trait_score <= 0.3:
                                    actual_level = "low"
                                else:
                                    actual_level = "medium"
                            
                            if actual_level and expected_level == actual_level:
                                matches += 1
                    
                    if total_traits > 0:
                        confidence_breakdown[condition_name]["factors"]["personality_correlation"] = matches / total_traits
            
            # Calculate consistency with previous diagnoses if temporal data available
            if self.enable_temporal_analysis and condition_name in self.temporal_data:
                confidence_breakdown[condition_name]["factors"]["symptom_consistency"] = 0.8  # Placeholder for consistent reporting
        
        return confidence_breakdown
    
    def _generate_differential_diagnosis(
        self,
        conditions: List[Dict[str, Any]],
        conversation_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate alternative diagnostic explanations (differential diagnosis)"""
        differential_diagnoses = []
        
        if not conversation_data or not conditions:
            return differential_diagnoses
        
        # Get the text from conversation data
        text = conversation_data.get("text", "").lower()
        if not text:
            return differential_diagnoses
        
        # Get all conditions that weren't in the primary diagnosis
        primary_condition_names = [c["name"] for c in conditions]
        
        # Check all conditions for potential matches
        for condition_id, condition in CONDITION_DEFINITIONS.items():
            condition_name = condition["name"]
            
            # Skip conditions already in the primary diagnosis
            if condition_name in primary_condition_names:
                continue
            
            # Count matching symptoms
            symptoms = condition["symptoms"]
            matched_symptoms = sum(1 for symptom in symptoms if symptom.lower() in text)
            
            # Only include if there's at least some symptom match
            if matched_symptoms > 0 and len(symptoms) > 0:
                match_ratio = matched_symptoms / len(symptoms)
                
                # Only include if the match ratio is significant but not enough for primary diagnosis
                if 0.2 <= match_ratio < 0.5:
                    differential_diagnoses.append({
                        "name": condition_name,
                        "confidence": match_ratio,
                        "matched_symptoms": matched_symptoms,
                        "total_symptoms": len(symptoms),
                        "explanation": self._generate_differential_explanation(condition_name, matched_symptoms, symptoms)
                    })
        
        # Sort by confidence and limit to top 3
        differential_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
        return differential_diagnoses[:3]
    
    def _generate_differential_explanation(
        self,
        condition_name: str,
        matched_symptoms: int,
        symptoms: List[str]
    ) -> str:
        """Generate an explanation for why a condition is in the differential diagnosis"""
        
        if matched_symptoms / len(symptoms) < 0.3:
            return f"{condition_name} shows some symptom overlap but insufficient evidence for diagnosis."
        elif matched_symptoms / len(symptoms) < 0.4:
            return f"{condition_name} has moderate symptom match but doesn't meet full diagnostic criteria."
        else:
            return f"{condition_name} is a possible alternative explanation that should be considered."
    
    def _analyze_temporal_patterns(
        self,
        current_diagnosis: Dict[str, Any],
        previous_diagnoses: List[Dict[str, Any]],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze changes in symptoms and diagnoses over time"""
        if not previous_diagnoses:
            return {"status": "insufficient_data"}
        
        temporal_analysis = {
            "condition_trends": {},
            "symptom_trends": {},
            "overall_trajectory": "stable",
            "significant_changes": []
        }
        
        # Sort diagnoses by timestamp if available
        sorted_diagnoses = sorted(
            [current_diagnosis] + previous_diagnoses,
            key=lambda d: d.get("timestamp", ""),
            reverse=True
        )
        
        if len(sorted_diagnoses) < 2:
            return {"status": "insufficient_data"}
        
        # Analyze condition trends
        all_conditions = set()
        for diagnosis in sorted_diagnoses:
            all_conditions.update(c["name"] for c in diagnosis.get("conditions", []))
        
        for condition in all_conditions:
            condition_data = {
                "timestamps": [],
                "severities": [],
                "confidences": []
            }
            
            for diagnosis in sorted_diagnoses:
                if "timestamp" not in diagnosis:
                    continue
                    
                matched_condition = next((c for c in diagnosis.get("conditions", []) 
                                        if c["name"] == condition), None)
                
                if matched_condition:
                    condition_data["timestamps"].append(diagnosis["timestamp"])
                    condition_data["severities"].append(matched_condition.get("severity", "unknown"))
                    condition_data["confidences"].append(matched_condition.get("confidence", 0.0))
            
            # Only include conditions with at least 2 data points
            if len(condition_data["timestamps"]) >= 2:
                # Calculate trend
                trend = "stable"
                if len(condition_data["confidences"]) >= 2:
                    first_confidence = condition_data["confidences"][-1]
                    last_confidence = condition_data["confidences"][0]
                    confidence_change = last_confidence - first_confidence
                    
                    if confidence_change > 0.2:
                        trend = "worsening"
                    elif confidence_change < -0.2:
                        trend = "improving"
                
                condition_data["trend"] = trend
                temporal_analysis["condition_trends"][condition] = condition_data
        
        # Determine overall trajectory
        improving_count = sum(1 for c in temporal_analysis["condition_trends"].values() 
                             if c.get("trend") == "improving")
        worsening_count = sum(1 for c in temporal_analysis["condition_trends"].values() 
                             if c.get("trend") == "worsening")
        
        if improving_count > worsening_count:
            temporal_analysis["overall_trajectory"] = "improving"
        elif worsening_count > improving_count:
            temporal_analysis["overall_trajectory"] = "worsening"
        
        # Identify significant changes
        for condition, data in temporal_analysis["condition_trends"].items():
            if len(data["confidences"]) >= 2:
                first_confidence = data["confidences"][-1]
                last_confidence = data["confidences"][0]
                
                if abs(last_confidence - first_confidence) > 0.3:
                    change_type = "increase" if last_confidence > first_confidence else "decrease"
                    temporal_analysis["significant_changes"].append({
                        "condition": condition,
                        "change_type": change_type,
                        "magnitude": abs(last_confidence - first_confidence),
                        "from_confidence": first_confidence,
                        "to_confidence": last_confidence
                    })
        
        return temporal_analysis
    
    def _store_temporal_data(self, user_id: str, diagnosis: Dict[str, Any]):
        """Store diagnostic data for temporal analysis"""
        if user_id not in self.temporal_data:
            self.temporal_data[user_id] = []
        
        # Limit storage to prevent memory issues
        if len(self.temporal_data[user_id]) >= self.symptom_tracking_days:
            self.temporal_data[user_id].pop(0)  # Remove oldest entry
        
        # Store a simplified version with just the essentials
        simplified = {
            "timestamp": diagnosis.get("timestamp", datetime.now().isoformat()),
            "conditions": diagnosis.get("conditions", []),
            "severity": diagnosis.get("severity", "unknown"),
            "confidence": diagnosis.get("confidence", 0.0)
        }
        
        self.temporal_data[user_id].append(simplified)
    
    async def _generate_enhanced_explanation(
        self,
        base_diagnosis: Dict[str, Any],
        conversation_data: Dict[str, Any],
        enhanced_result: Dict[str, Any]
    ) -> str:
        """Generate an enhanced explanation using AgenticRAG"""
        try:
            if not self.agentic_rag:
                return "Enhanced explanation not available without AgenticRAG."
            
            text = conversation_data.get("text", "")
            if not text:
                return "Enhanced explanation not available without conversation data."
            
            enhanced_results = await self.agentic_rag.enhance_diagnosis(text)
            
            if enhanced_results and enhanced_results.get("success"):
                return enhanced_results.get("reasoning", "No reasoning provided.")
            
            return "Unable to generate enhanced explanation."
            
        except Exception as e:
            logger.error(f"Error generating enhanced explanation: {str(e)}")
            return "Error generating enhanced explanation."
    
    async def get_similar_symptoms(self, symptom_text: str, threshold: float = 0.7, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar symptoms based on semantic similarity"""
        if not self.embedder or not self.symptoms_index:
            # Fallback to string matching
            return self._fallback_symptom_similarity(symptom_text, threshold, k)
        
        try:
            # Encode the query symptom
            query_vector = self.embedder.encode([symptom_text])
            
            # Search for similar symptoms
            D, I = self.symptoms_index.search(np.array(query_vector).astype('float32'), k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.all_symptoms):
                    similarity = 1.0 - (dist / 2.0)  # Convert L2 distance to similarity score
                    
                    if similarity >= threshold:
                        results.append({
                            "symptom": self.all_symptoms[idx],
                            "similarity": similarity
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in symptom similarity search: {str(e)}")
            return self._fallback_symptom_similarity(symptom_text, threshold, k)
    
    def _fallback_symptom_similarity(self, symptom_text: str, threshold: float = 0.7, k: int = 5) -> List[Dict[str, Any]]:
        """Fallback method for finding similar symptoms using string matching"""
        results = []
        symptom_text = symptom_text.lower()
        
        for symptom in self.all_symptoms:
            # Calculate simple Jaccard similarity
            words1 = set(symptom_text.split())
            words2 = set(symptom.lower().split())
            
            if not words1 or not words2:
                continue
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append({
                    "symptom": symptom,
                    "similarity": similarity
                })
        
        # Sort by similarity and limit to k results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
    
    def get_symptom_associations(self, symptom: str) -> Dict[str, Any]:
        """Get condition associations for a specific symptom"""
        associations = {
            "conditions": [],
            "co_occurring_symptoms": []
        }
        
        # Check each condition for the symptom
        for condition_id, condition in CONDITION_DEFINITIONS.items():
            if symptom in condition["symptoms"]:
                associations["conditions"].append({
                    "name": condition["name"],
                    "id": condition_id
                })
                
                # Add other symptoms from this condition as co-occurring
                for other_symptom in condition["symptoms"]:
                    if other_symptom != symptom and other_symptom not in associations["co_occurring_symptoms"]:
                        associations["co_occurring_symptoms"].append(other_symptom)
        
        return associations


class RecommendationEngine:
    """Personalized recommendation engine for mental health support"""
    
    def __init__(self):
        """Initialize the recommendation engine"""
        self.recommendation_templates = self._load_recommendation_templates()
        
    def _load_recommendation_templates(self) -> Dict[str, List[str]]:
        """Load recommendation templates from file or use defaults"""
        try:
            # Try to load from data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            recommendation_path = os.path.join(data_dir, 'recommendations.json')
            
            if os.path.exists(recommendation_path):
                with open(recommendation_path, 'r') as f:
                    return json.load(f)
            
            # If file doesn't exist, use defaults
            return self._get_default_recommendations()
            
        except Exception as e:
            logger.error(f"Error loading recommendation templates: {str(e)}")
            return self._get_default_recommendations()
    
    def _get_default_recommendations(self) -> Dict[str, List[str]]:
        """Get default recommendation templates"""
        return {
            "depression": [
                "Consider talking to a mental health professional about your depressive symptoms",
                "Try to maintain a regular daily routine, including consistent sleep and wake times",
                "Engage in physical activity, even brief walks can help improve mood",
                "Practice mindfulness meditation to help manage negative thoughts",
                "Connect with supportive friends or family members, even when you don't feel like it",
                "Set small, achievable goals each day to build a sense of accomplishment",
                "Limit alcohol and avoid recreational drugs, which can worsen depression",
                "Challenge negative thoughts by asking yourself if they're based on facts or emotions"
            ],
            "anxiety": [
                "Practice deep breathing exercises when feeling anxious (4-7-8 technique: inhale for 4, hold for 7, exhale for 8)",
                "Consider consulting with a mental health professional about anxiety management strategies",
                "Reduce caffeine intake, which can exacerbate anxiety symptoms",
                "Establish a regular sleep schedule to improve resilience to stress",
                "Try progressive muscle relaxation to release physical tension",
                "Challenge anxious thoughts by examining evidence for and against them",
                "Create a worry schedule: set aside specific time to address worries, then let them go",
                "Practice grounding techniques (name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste)"
            ],
            "stress": [
                "Identify and reduce sources of stress where possible",
                "Practice time management techniques like the Pomodoro method",
                "Set boundaries in work and personal life",
                "Take regular breaks during work or stressful activities",
                "Engage in enjoyable activities daily, even if just for a few minutes",
                "Practice mindfulness or meditation to center yourself",
                "Use deep breathing exercises when feeling overwhelmed",
                "Prioritize adequate sleep and nutrition during stressful periods"
            ],
            "insomnia": [
                "Maintain a consistent sleep schedule, even on weekends",
                "Create a relaxing bedtime routine to signal your body it's time to sleep",
                "Limit screen time 1-2 hours before bed due to blue light exposure",
                "Create a comfortable sleep environment (cool, dark, quiet)",
                "Avoid caffeine after noon and limit alcohol, which disrupts sleep quality",
                "Consider speaking with a healthcare provider about sleep therapy options",
                "Try relaxation techniques like progressive muscle relaxation before bed",
                "If you can't sleep after 20 minutes, get up and do something relaxing until you feel sleepy"
            ],
            "bipolar": [
                "Work with a psychiatrist to find the right medication regimen",
                "Maintain a consistent sleep schedule to help stabilize mood",
                "Track your moods, sleep, and activities to identify patterns and triggers",
                "Create a crisis plan for managing manic or depressive episodes",
                "Establish a routine that includes regular exercise and healthy eating",
                "Learn to recognize early warning signs of mood episodes",
                "Consider therapy approaches like cognitive behavioral therapy or interpersonal therapy",
                "Engage trusted friends or family members in your treatment plan"
            ],
            "personality_traits": {
                "high_neuroticism": [
                    "Practice cognitive restructuring to challenge negative thought patterns",
                    "Keep a thought journal to identify and address cognitive distortions",
                    "Learn and practice emotional regulation techniques",
                    "Establish a consistent self-care routine"
                ],
                "low_extraversion": [
                    "Balance social energy with needed alone time for recovery",
                    "Set small, achievable goals for social interaction",
                    "Find social activities aligned with your personal interests",
                    "Practice self-compassion about your social needs"
                ],
                "high_conscientiousness": [
                    "Practice self-compassion when you don't meet your own high standards",
                    "Schedule regular downtime to prevent burnout",
                    "Delegate tasks when appropriate",
                    "Set realistic expectations and boundaries"
                ],
                "low_conscientiousness": [
                    "Break large tasks into smaller, manageable steps",
                    "Use external organization tools like reminders and planners",
                    "Create routines for regular tasks",
                    "Set specific goals with deadlines and accountability"
                ],
                "high_openness": [
                    "Channel creativity into structured projects or activities",
                    "Explore mindfulness to help focus your creative energy",
                    "Balance exploration with consistency in daily routines",
                    "Connect with others who share your interests"
                ],
                "low_openness": [
                    "Gradually introduce small changes to your routine",
                    "Approach new ideas and experiences with curiosity",
                    "Practice cognitive flexibility through puzzles or new skills",
                    "Consider the practical benefits of trying new approaches"
                ],
                "high_agreeableness": [
                    "Practice setting and maintaining personal boundaries",
                    "Learn assertiveness techniques for expressing your needs",
                    "Balance caring for others with self-care",
                    "Recognize when helping others might be detrimental to yourself"
                ],
                "low_agreeableness": [
                    "Practice active listening and empathy in conversations",
                    "Consider multiple perspectives before responding",
                    "Use 'I' statements when expressing concerns",
                    "Find constructive ways to express disagreement"
                ]
            }
        }
    
    async def generate_recommendations(
        self,
        conditions: List[Dict[str, Any]],
        personality_traits: Optional[Dict[str, Any]] = None,
        previous_diagnoses: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on conditions and personality"""
        if not conditions:
            return []
            
        recommendations = []
        
        # Get condition-specific recommendations
        for condition in conditions:
            condition_name = condition["name"].lower()
            condition_id = None
            
            # Find the condition ID
            for cid, cond in CONDITION_DEFINITIONS.items():
                if cond["name"].lower() == condition_name:
                    condition_id = cid
                    break
            
            if not condition_id:
                continue
                
            # Get recommendations for this condition
            condition_recs = self._get_condition_recommendations(condition_id, condition["severity"])
            
            for rec in condition_recs:
                if rec not in [r["text"] for r in recommendations]:
                    recommendations.append({
                        "text": rec,
                        "source": condition["name"],
                        "type": "condition_specific",
                        "severity": condition["severity"]
                    })
        
        # Add personality-specific recommendations if available
        if personality_traits and "big_five" in personality_traits:
            big_five = personality_traits["big_five"]
            personality_recs = self._get_personality_recommendations(big_five)
            
            for rec in personality_recs:
                if rec["text"] not in [r["text"] for r in recommendations]:
                    recommendations.append(rec)
        
        # Add consistency-based recommendations if we have previous diagnoses
        if previous_diagnoses and len(previous_diagnoses) >= 2:
            consistency_recs = self._get_consistency_recommendations(conditions, previous_diagnoses)
            
            for rec in consistency_recs:
                if rec["text"] not in [r["text"] for r in recommendations]:
                    recommendations.append(rec)
        
        # Add general mental health recommendations
        general_recs = [
            "Consider speaking with a mental health professional for a comprehensive evaluation",
            "Practice self-compassion during difficult times",
            "Maintain social connections with supportive people",
            "Prioritize adequate sleep, nutrition, and physical activity",
            "Develop a regular mindfulness or meditation practice"
        ]
        
        for rec in general_recs:
            if rec not in [r["text"] for r in recommendations]:
                recommendations.append({
                    "text": rec,
                    "source": "general",
                    "type": "general_wellbeing",
                    "severity": "all"
                })
        
        # Limit to 10 recommendations, prioritizing by condition severity
        recommendations.sort(key=lambda x: self._get_recommendation_priority(x))
        return recommendations[:10]
    
    def _get_condition_recommendations(self, condition_id: str, severity: str) -> List[str]:
        """Get recommendations for a specific condition and severity"""
        recommendations = []
        
        # Get general recommendations for the condition
        if condition_id in self.recommendation_templates:
            recommendations.extend(self.recommendation_templates[condition_id])
        
        # Add severity-specific recommendations
        if severity == "severe":
            if condition_id == "depression":
                recommendations.append("Consider speaking with a healthcare provider about treatment options, including medication")
                recommendations.append("Establish a daily check-in with a trusted person for support")
            elif condition_id == "anxiety":
                recommendations.append("Consider working with a mental health professional on structured anxiety management")
                recommendations.append("Learn to recognize and manage panic symptoms if they occur")
            elif condition_id == "stress":
                recommendations.append("Evaluate your commitments and consider temporarily reducing responsibilities")
                recommendations.append("Create a structured self-care plan with daily stress reduction activities")
            elif condition_id == "bipolar":
                recommendations.append("Work closely with your healthcare provider on medication management")
                recommendations.append("Establish a crisis plan with your support network")
            elif condition_id == "insomnia":
                recommendations.append("Discuss with a healthcare provider about short-term solutions for severe sleep difficulties")
                recommendations.append("Consider cognitive behavioral therapy for insomnia (CBT-I)")
        
        return recommendations
    
    def _get_personality_recommendations(self, big_five: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on personality traits"""
        recommendations = []
        
        for trait, data in big_five.items():
            trait_level = None
            
            if isinstance(data, dict) and "level" in data:
                trait_level = data["level"]
            elif isinstance(data, (int, float)):
                if data >= 0.7:
                    trait_level = "high"
                elif data <= 0.3:
                    trait_level = "low"
                else:
                    trait_level = "medium"
            
            if trait_level in ["high", "low"]:
                trait_key = f"{trait_level}_{trait.lower()}"
                
                if "personality_traits" in self.recommendation_templates and trait_key in self.recommendation_templates["personality_traits"]:
                    for rec in self.recommendation_templates["personality_traits"][trait_key]:
                        recommendations.append({
                            "text": rec,
                            "source": f"{trait} ({trait_level})",
                            "type": "personality_specific",
                            "severity": "all"
                        })
        
        return recommendations
    
    def _get_consistency_recommendations(
        self,
        current_conditions: List[Dict[str, Any]],
        previous_diagnoses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on condition consistency over time"""
        recommendations = []
        
        # Identify persistent conditions (present in multiple assessments)
        current_condition_names = [c["name"] for c in current_conditions]
        
        persistent_conditions = {}
        for diagnosis in previous_diagnoses:
            for condition in diagnosis.get("conditions", []):
                name = condition["name"]
                if name not in persistent_conditions:
                    persistent_conditions[name] = 0
                persistent_conditions[name] += 1
        
        # Add recommendations for persistent conditions
        for condition, count in persistent_conditions.items():
            if condition in current_condition_names and count >= 2:
                # This is a persistent condition
                if "depression" in condition.lower():
                    recommendations.append({
                        "text": "Since your depressive symptoms have been persistent, consider a structured approach like cognitive behavioral therapy",
                        "source": "pattern analysis",
                        "type": "consistency_based",
                        "severity": "persistent"
                    })
                elif "anxiety" in condition.lower():
                    recommendations.append({
                        "text": "For your persistent anxiety, consider learning long-term management techniques like systematic desensitization",
                        "source": "pattern analysis",
                        "type": "consistency_based",
                        "severity": "persistent"
                    })
                elif "stress" in condition.lower():
                    recommendations.append({
                        "text": "Your ongoing stress suggests a need for sustainable lifestyle changes rather than short-term fixes",
                        "source": "pattern analysis",
                        "type": "consistency_based",
                        "severity": "persistent"
                    })
        
        return recommendations
    
    def _get_recommendation_priority(self, recommendation: Dict[str, Any]) -> int:
        """Get a priority score for ordering recommendations (lower is higher priority)"""
        # Start with base priority
        priority = 100
        
        # Adjust based on type
        if recommendation["type"] == "condition_specific":
            priority -= 50
        elif recommendation["type"] == "consistency_based":
            priority -= 40
        elif recommendation["type"] == "personality_specific":
            priority -= 30
        
        # Adjust based on severity
        if recommendation["severity"] == "severe":
            priority -= 20
        elif recommendation["severity"] == "moderate":
            priority -= 10
        elif recommendation["severity"] == "persistent":
            priority -= 15
        
        return priority


class ResearchIntegrator:
    """Integrates recent research findings into diagnosis"""
    
    def __init__(self, update_frequency_days: int = 30):
        """Initialize the research integrator"""
        self.update_frequency_days = update_frequency_days
        self.research_cache = {}
        self.last_update = {}
    
    async def get_insights(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Get research insights for specific conditions"""
        insights = []
        
        for condition in conditions:
            # Check if we need to update the cache
            if self._should_update_cache(condition):
                await self._update_research_cache(condition)
            
            # Get insights from cache
            if condition in self.research_cache:
                for insight in self.research_cache[condition]:
                    if insight not in insights:
                        insights.append(insight)
        
        return insights
    
    def _should_update_cache(self, condition: str) -> bool:
        """Check if we should update the research cache for a condition"""
        if condition not in self.last_update:
            return True
            
        last_update_time = self.last_update[condition]
        current_time = datetime.now()
        
        # Update if it's been more than the update frequency
        return (current_time - last_update_time).days >= self.update_frequency_days
    
    async def _update_research_cache(self, condition: str):
        """Update the research cache for a condition"""
        try:
            # This would be where you'd implement a real research retrieval system
            # For now, we'll use placeholder data
            
            self.research_cache[condition] = [
                {
                    "title": f"Recent findings on {condition} treatment approaches",
                    "summary": f"Research suggests that combined therapy and lifestyle changes may be more effective for {condition} than either approach alone.",
                    "relevance": "treatment",
                    "year": 2025,
                    "source": "Journal of Mental Health Research"
                }
            ]
            
            self.last_update[condition] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating research cache for {condition}: {str(e)}")
            # Ensure we have at least an empty list
            if condition not in self.research_cache:
                self.research_cache[condition] = []