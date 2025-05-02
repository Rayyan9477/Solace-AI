"""
Integrated Diagnosis Module

This module combines data from multiple sources to provide a comprehensive mental health assessment:
1. Voice emotion analysis - Detecting emotional states from voice patterns
2. Conversational AI - Extracting insights from chat interactions
3. Personality assessments - Using structured personality test results

The integration uses a rule-based approach combined with vector similarity to generate insights.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import faiss

# Project imports
from database.vector_store import VectorStore
from utils.agentic_rag import AgenticRAG
from config.settings import AppConfig

# Configure logging
logger = logging.getLogger(__name__)

# Mental health condition definitions with associated symptoms and patterns
CONDITION_DEFINITIONS = {
    "depression": {
        "name": "Depression",
        "symptoms": [
            "persistent sadness", "loss of interest", "fatigue", "sleep problems",
            "appetite changes", "feelings of worthlessness", "difficulty concentrating", 
            "negative thoughts", "low energy", "social withdrawal"
        ],
        "voice_indicators": ["monotone voice", "slower speech", "reduced pitch variation", "quieter volume"],
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
            "feeling on edge", "anticipating worst outcomes"
        ],
        "voice_indicators": ["faster speech", "higher pitch", "trembling voice", "rapid breathing"],
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
            "sleep problems", "difficulty concentrating", "mood changes"
        ],
        "voice_indicators": ["faster speech", "tense tone", "higher pitch", "louder volume"],
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
    }
}

class DiagnosisModule:
    """
    Integrates voice emotion analysis, conversational AI output, and personality test results
    to provide comprehensive mental health insights using a rule-based system.
    """
    
    def __init__(self, agentic_rag: Optional[AgenticRAG] = None, use_vector_cache: bool = True):
        """
        Initialize the diagnosis module
        
        Args:
            agentic_rag: Optional AgenticRAG instance for enhanced reasoning
            use_vector_cache: Whether to use vector caching for previous diagnoses
        """
        self.agentic_rag = agentic_rag
        self.use_vector_cache = use_vector_cache
        
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
    
    async def generate_diagnosis(
        self,
        conversation_data: Optional[Dict[str, Any]] = None,
        voice_emotion_data: Optional[Dict[str, Any]] = None,
        personality_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnosis by integrating multiple data sources
        
        Args:
            conversation_data: Data extracted from chatbot conversations
            voice_emotion_data: Emotional patterns detected from voice analysis
            personality_data: Results from personality assessments
            user_id: Optional user identifier for caching
            
        Returns:
            Comprehensive diagnostic assessment
        """
        start_time = datetime.now()
        
        # Check cache for similar diagnosis if enabled
        if self.use_vector_cache and self.vector_store and user_id:
            cache_key = f"diagnosis_{user_id}_{start_time.strftime('%Y%m%d')}"
            cached_result = self._check_diagnosis_cache(cache_key)
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
            conversation_symptoms = self._extract_symptoms_from_conversation(conversation_data)
            
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
        
        # Use AgenticRAG if available for enhanced analysis
        agentic_insights = {}
        if self.agentic_rag and conversation_data and "text" in conversation_data:
            try:
                enhanced_results = await self.agentic_rag.enhance_diagnosis(conversation_data["text"])
                if enhanced_results and enhanced_results.get("success"):
                    agentic_insights = enhanced_results
                    
                    # Add symptoms from enhanced analysis if available
                    if "symptoms" in enhanced_results:
                        for symptom in enhanced_results["symptoms"]:
                            if isinstance(symptom, dict) and "symptom" in symptom:
                                if symptom["symptom"] not in [s["text"] for s in conversation_symptoms]:
                                    conversation_symptoms.append({"text": symptom["symptom"], "source": "agentic_rag"})
                                    all_indicators.append({"text": symptom["symptom"], "source": "agentic_rag"})
            except Exception as e:
                logger.error(f"Error in AgenticRAG processing: {str(e)}")
        
        # Match all indicators against condition definitions using rule-based approach
        conditions_matched = self._match_conditions(all_indicators, personality_traits)
        
        # If we have matches, build the diagnosis result
        if conditions_matched:
            result["success"] = True
            result["conditions"] = [{"name": cond["name"], "confidence": cond["confidence"], "severity": cond["severity"]} 
                                   for cond in conditions_matched]
            
            # Use the highest confidence condition as the primary one
            primary_condition = max(conditions_matched, key=lambda x: x["confidence"])
            result["severity"] = primary_condition["severity"]
            result["confidence"] = primary_condition["confidence"]
            
            # Generate recommendations based on conditions
            result["recommendations"] = self._generate_recommendations(conditions_matched, personality_traits)
            
            # Add insights
            result["insights"] = {
                "symptoms_identified": [s["text"] for s in conversation_symptoms],
                "emotional_indicators": [v["text"] for v in voice_indicators],
                "personality_factors": personality_traits,
                "evidence": self._generate_evidence_summary(conditions_matched, all_indicators)
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
                    result["recommendations"].extend(agentic_insights["recommendations"])
                    
                # Remove duplicate recommendations
                result["recommendations"] = list(set(result["recommendations"]))
        else:
            result["message"] = "No significant conditions detected based on available data"
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        result["processing_time_seconds"] = processing_time
        
        # Cache the result if enabled
        if self.use_vector_cache and self.vector_store and user_id and result["success"]:
            cache_key = f"diagnosis_{user_id}_{start_time.strftime('%Y%m%d')}"
            self._cache_diagnosis(cache_key, result)
        
        return result
    
    def _extract_symptoms_from_conversation(self, conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract symptoms from conversation data"""
        symptoms = []
        
        if "text" in conversation_data:
            text = conversation_data["text"]
            # Flatten all symptoms from all conditions into a single set for matching
            all_symptoms = set()
            for condition in CONDITION_DEFINITIONS.values():
                all_symptoms.update(condition["symptoms"])
            
            # Simple keyword matching for symptoms in text
            for symptom in all_symptoms:
                if symptom.lower() in text.lower():
                    symptoms.append({"text": symptom, "source": "conversation"})
        
        if "extracted_symptoms" in conversation_data:
            # If the conversation analyzer already extracted symptoms
            for symptom in conversation_data["extracted_symptoms"]:
                if isinstance(symptom, str):
                    symptoms.append({"text": symptom, "source": "conversation_analyzer"})
                elif isinstance(symptom, dict) and "text" in symptom:
                    symptoms.append(symptom)
        
        return symptoms
    
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
                            "source": "voice", 
                            "score": score
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
                                "score": value
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
        
        return traits
    
    def _match_conditions(self, indicators: List[Dict[str, Any]], personality_traits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match indicators against condition definitions using rule-based approach"""
        matched_conditions = []
        
        for condition_id, condition in CONDITION_DEFINITIONS.items():
            # Count matching symptoms
            matching_symptoms = []
            for indicator in indicators:
                indicator_text = indicator["text"].lower()
                for symptom in condition["symptoms"]:
                    if symptom.lower() in indicator_text or indicator_text in symptom.lower():
                        matching_symptoms.append({
                            "symptom": symptom,
                            "indicator": indicator
                        })
                        break
            
            # Count matching voice indicators
            matching_voice = []
            for indicator in indicators:
                if indicator["source"] in ["voice", "voice_characteristics"]:
                    indicator_text = indicator["text"].lower()
                    for voice_ind in condition["voice_indicators"]:
                        if voice_ind.lower() in indicator_text or indicator_text in voice_ind.lower():
                            matching_voice.append({
                                "voice_indicator": voice_ind,
                                "indicator": indicator
                            })
                            break
            
            # Count matching personality traits
            matching_traits = []
            if "big_five" in personality_traits and "big_five" in condition["personality_correlations"]:
                for trait, expected_level in condition["personality_correlations"]["big_five"].items():
                    if (trait in personality_traits["big_five"] and 
                        personality_traits["big_five"][trait]["level"] == expected_level):
                        matching_traits.append({
                            "trait": trait, 
                            "expected": expected_level,
                            "actual": personality_traits["big_five"][trait]["level"]
                        })
            
            # Calculate confidence based on matches
            # Symptoms are weighted most heavily, followed by voice indicators and personality traits
            symptom_weight = 0.6
            voice_weight = 0.25
            personality_weight = 0.15
            
            # Calculate proportional matches
            symptom_match = len(matching_symptoms) / len(condition["symptoms"]) if condition["symptoms"] else 0
            voice_match = len(matching_voice) / len(condition["voice_indicators"]) if condition["voice_indicators"] else 0
            personality_match = len(matching_traits) / len(condition["personality_correlations"]["big_five"]) if "big_five" in condition["personality_correlations"] else 0
            
            # Weighted confidence
            confidence = (symptom_match * symptom_weight) + (voice_match * voice_weight) + (personality_match * personality_weight)
            
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
            if confidence >= 0.3 and severity != "none":
                matched_conditions.append({
                    "id": condition_id,
                    "name": condition["name"],
                    "confidence": confidence,
                    "severity": severity,
                    "matching_symptoms": matching_symptoms,
                    "matching_voice": matching_voice,
                    "matching_traits": matching_traits
                })
        
        # Sort by confidence
        matched_conditions.sort(key=lambda x: x["confidence"], reverse=True)
        return matched_conditions
    
    def _generate_recommendations(self, conditions: List[Dict[str, Any]], personality_traits: Dict[str, Any]) -> List[str]:
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
                    "Practice mindfulness meditation to stay present"
                ])
            elif condition["id"] == "anxiety":
                recommendations.extend([
                    "Practice deep breathing exercises when feeling anxious",
                    "Consider speaking with a mental health professional about anxiety management",
                    "Reduce caffeine and alcohol consumption",
                    "Establish a regular sleep schedule",
                    "Try progressive muscle relaxation for physical tension"
                ])
            elif condition["id"] == "stress":
                recommendations.extend([
                    "Identify and reduce sources of stress when possible",
                    "Practice time management techniques",
                    "Set boundaries in work and personal life",
                    "Take regular breaks during work",
                    "Engage in enjoyable activities daily"
                ])
        
        # Personality-specific recommendations
        if "big_five" in personality_traits:
            big_five = personality_traits["big_five"]
            
            # High neuroticism recommendations
            if "neuroticism" in big_five and big_five["neuroticism"]["level"] == "high":
                recommendations.extend([
                    "Practice cognitive restructuring to challenge negative thought patterns",
                    "Keep a thought journal to identify cognitive distortions",
                    "Learn to recognize and name emotions as they arise"
                ])
            
            # Low extraversion recommendations
            if "extraversion" in big_five and big_five["extraversion"]["level"] == "low":
                recommendations.extend([
                    "Balance social energy with needed alone time for recovery",
                    "Set small, achievable goals for social interaction",
                    "Find social activities aligned with your interests"
                ])
            
            # High conscientiousness recommendations
            if "conscientiousness" in big_five and big_five["conscientiousness"]["level"] == "high":
                recommendations.extend([
                    "Practice self-compassion when you don't meet your own high standards",
                    "Schedule regular downtime to prevent burnout",
                    "Delegate tasks when appropriate"
                ])
        
        # Remove duplicates and limit number of recommendations
        recommendations = list(set(recommendations))
        return recommendations[:7]  # Limit to 7 recommendations
    
    def _generate_evidence_summary(self, conditions: List[Dict[str, Any]], indicators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the evidence used for diagnosis"""
        evidence = {
            "total_indicators": len(indicators),
            "indicators_by_source": {},
            "key_symptoms": []
        }
        
        # Count indicators by source
        for indicator in indicators:
            source = indicator["source"]
            if source not in evidence["indicators_by_source"]:
                evidence["indicators_by_source"][source] = 0
            evidence["indicators_by_source"][source] += 1
        
        # Add key symptoms per condition
        for condition in conditions:
            condition_symptoms = []
            for match in condition["matching_symptoms"]:
                symptom = match["symptom"]
                if symptom not in condition_symptoms:
                    condition_symptoms.append(symptom)
            
            evidence["key_symptoms"].append({
                "condition": condition["name"],
                "symptoms": condition_symptoms
            })
        
        return evidence
    
    def _check_diagnosis_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check for a cached diagnosis"""
        if not self.use_vector_cache or not self.vector_store:
            return None
        
        try:
            similar_results = self.vector_store.find_similar_results(cache_key, threshold=0.9, k=1)
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
    
    def _cache_diagnosis(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """Cache a diagnosis result"""
        if not self.use_vector_cache or not self.vector_store:
            return False
        
        try:
            self.vector_store.add_processed_result(cache_key, result)
            logger.info(f"Cached diagnosis with key: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Error caching diagnosis: {str(e)}")
            return False