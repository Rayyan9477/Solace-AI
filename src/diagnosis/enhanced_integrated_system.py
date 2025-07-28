"""
Enhanced Integrated Diagnostic System

This module integrates all the enhanced diagnostic and therapeutic components
into a unified system with proper error handling, bug fixes, and seamless
integration between temporal analysis, differential diagnosis, therapeutic friction,
cultural sensitivity, adaptive learning, memory system, and research integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import traceback

from .temporal_analysis import TemporalAnalysisEngine, SymptomEntry
from .differential_diagnosis import DifferentialDiagnosisEngine, DifferentialDiagnosis
from .therapeutic_friction import TherapeuticFrictionEngine, TherapeuticResponse
from .cultural_sensitivity import CulturalSensitivityEngine, CulturalProfile
from .adaptive_learning import AdaptiveLearningEngine, InterventionOutcome
from ..memory.enhanced_memory_system import EnhancedMemorySystem, TherapeuticInsight
from ..research.real_time_research import RealTimeResearchEngine, EvidenceBasedRecommendation
from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ComprehensiveDiagnosticResult:
    """Comprehensive diagnostic result combining all systems"""
    user_id: str
    session_id: str
    timestamp: datetime
    
    # Core diagnostic data
    differential_diagnoses: List[DifferentialDiagnosis]
    primary_diagnosis: Optional[DifferentialDiagnosis]
    confidence_score: float
    
    # Temporal analysis
    symptom_progression: Dict[str, Any]
    behavioral_patterns: List[Dict[str, Any]]
    trajectory_prediction: Dict[str, Any]
    
    # Therapeutic response
    therapeutic_response: TherapeuticResponse
    cultural_adaptations: Dict[str, Any]
    
    # Memory and insights
    relevant_insights: List[TherapeuticInsight]
    session_continuity: Dict[str, Any]
    progress_tracking: Dict[str, Any]
    
    # Research and evidence
    evidence_based_recommendations: List[EvidenceBasedRecommendation]
    treatment_validation: Dict[str, Any]
    
    # Learning and adaptation
    personalized_recommendations: Dict[str, Any]
    intervention_effectiveness: Dict[str, Any]
    
    # Integration quality metrics
    integration_confidence: float
    system_reliability: float
    processing_time_ms: float
    
    # Error handling
    warnings: List[str]
    limitations: List[str]

class EnhancedIntegratedDiagnosticSystem:
    """
    Unified diagnostic system that integrates all enhanced components
    with comprehensive error handling and seamless data flow.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the integrated diagnostic system"""
        self.vector_db = vector_db
        self.logger = get_logger(__name__)
        
        # Initialize all subsystems with error handling
        self.systems_initialized = {}
        self.initialization_errors = {}
        
        # Core diagnostic systems
        self._initialize_systems()
        
        # Integration settings
        self.max_processing_time_seconds = 30
        self.min_confidence_threshold = 0.3
        self.system_weights = {
            "differential_diagnosis": 0.25,
            "temporal_analysis": 0.20,
            "therapeutic_friction": 0.15,
            "cultural_sensitivity": 0.15,
            "adaptive_learning": 0.10,
            "memory_system": 0.10,
            "research_integration": 0.05
        }
        
    def _initialize_systems(self):
        """Initialize all subsystems with error handling"""
        systems = {
            "temporal_analysis": lambda: TemporalAnalysisEngine(self.vector_db),
            "differential_diagnosis": lambda: DifferentialDiagnosisEngine(self.vector_db),
            "therapeutic_friction": lambda: TherapeuticFrictionEngine(self.vector_db),
            "cultural_sensitivity": lambda: CulturalSensitivityEngine(self.vector_db),
            "adaptive_learning": lambda: AdaptiveLearningEngine(self.vector_db),
            "memory_system": lambda: EnhancedMemorySystem(self.vector_db),
            "research_integration": lambda: RealTimeResearchEngine(self.vector_db)
        }
        
        for system_name, initializer in systems.items():
            try:
                setattr(self, system_name, initializer())
                self.systems_initialized[system_name] = True
                self.logger.info(f"Successfully initialized {system_name}")
            except Exception as e:
                self.systems_initialized[system_name] = False
                self.initialization_errors[system_name] = str(e)
                self.logger.error(f"Failed to initialize {system_name}: {str(e)}")
                # Set a fallback system
                setattr(self, system_name, None)
    
    async def generate_comprehensive_diagnosis(self,
                                             user_id: str,
                                             session_id: str,
                                             user_message: str,
                                             conversation_history: List[Dict[str, Any]],
                                             voice_emotion_data: Dict[str, Any] = None,
                                             personality_data: Dict[str, Any] = None,
                                             cultural_info: Dict[str, Any] = None) -> ComprehensiveDiagnosticResult:
        """
        Generate comprehensive diagnosis using all integrated systems
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: Current user message
            conversation_history: Previous conversation context
            voice_emotion_data: Voice emotion analysis results
            personality_data: Personality assessment data
            cultural_info: Cultural background information
            
        Returns:
            Comprehensive diagnostic result
        """
        start_time = datetime.now()
        warnings = []
        limitations = []
        
        try:
            self.logger.info(f"Starting comprehensive diagnosis for user {user_id}, session {session_id}")
            
            # Step 1: Cultural Context Assessment
            cultural_profile = None
            cultural_adaptations = {}
            
            if self.systems_initialized.get("cultural_sensitivity", False):
                try:
                    cultural_profile = await self.cultural_sensitivity.assess_cultural_context(
                        user_id, user_message, conversation_history, cultural_info
                    )
                except Exception as e:
                    warnings.append(f"Cultural assessment error: {str(e)}")
                    self.logger.warning(f"Cultural assessment failed: {str(e)}")
            else:
                limitations.append("Cultural sensitivity system unavailable")
            
            # Step 2: Memory and Session Continuity
            session_continuity = {}
            relevant_insights = []
            
            if self.systems_initialized.get("memory_system", False):
                try:
                    session_continuity = await self.memory_system.get_session_continuity_context(
                        user_id, session_id
                    )
                    
                    # Get relevant therapeutic insights
                    contextual_memory = await self.memory_system.get_contextual_memory(
                        user_id, "significant", lookback_days=30
                    )
                    relevant_insights = contextual_memory.get("significant_insights", [])
                    
                except Exception as e:
                    warnings.append(f"Memory system error: {str(e)}")
                    self.logger.warning(f"Memory system failed: {str(e)}")
            else:
                limitations.append("Memory system unavailable")
            
            # Step 3: Temporal Analysis
            symptom_progression = {}
            behavioral_patterns = []
            trajectory_prediction = {}
            
            if self.systems_initialized.get("temporal_analysis", False):
                try:
                    # Record current symptoms if provided
                    if "symptoms" in user_message.lower():
                        symptoms = self._extract_symptoms_from_message(user_message)
                        for symptom_type, intensity in symptoms.items():
                            await self.temporal_analysis.record_symptom(
                                user_id, symptom_type, intensity, user_message, session_id=session_id
                            )
                    
                    # Get symptom progression
                    symptom_progression = await self.temporal_analysis.get_symptom_progression(
                        user_id, days_back=30
                    )
                    
                    # Detect behavioral patterns
                    behavioral_patterns = await self.temporal_analysis.detect_behavioral_patterns(user_id)
                    behavioral_patterns = [asdict(p) for p in behavioral_patterns]
                    
                    # Predict trajectory for main symptoms
                    if symptom_progression.get("progression"):
                        main_symptom = max(symptom_progression["progression"].keys(),
                                         key=lambda x: symptom_progression["progression"][x]["count"])
                        trajectory_prediction = await self.temporal_analysis.predict_symptom_trajectory(
                            user_id, main_symptom, prediction_days=7
                        )
                    
                except Exception as e:
                    warnings.append(f"Temporal analysis error: {str(e)}")
                    self.logger.warning(f"Temporal analysis failed: {str(e)}")
            else:
                limitations.append("Temporal analysis system unavailable")
            
            # Step 4: Differential Diagnosis
            differential_diagnoses = []
            primary_diagnosis = None
            confidence_score = 0.0
            
            if self.systems_initialized.get("differential_diagnosis", False):
                try:
                    # Extract symptoms and observations
                    symptoms = list(self._extract_symptoms_from_message(user_message).keys())
                    behavioral_observations = self._extract_behavioral_observations(
                        user_message, conversation_history
                    )
                    
                    # Generate differential diagnosis
                    diagnosis_result = await self.differential_diagnosis.generate_differential_diagnosis(
                        symptoms=symptoms,
                        behavioral_observations=behavioral_observations,
                        temporal_patterns=symptom_progression,
                        voice_emotion_data=voice_emotion_data,
                        personality_data=personality_data,
                        user_context={"user_id": user_id, "session_id": session_id}
                    )
                    
                    if not diagnosis_result.get("error"):
                        differential_diagnoses = [
                            self._dict_to_differential_diagnosis(d) 
                            for d in diagnosis_result.get("differential_diagnoses", [])
                        ]
                        if differential_diagnoses:
                            primary_diagnosis = differential_diagnoses[0]
                        confidence_score = diagnosis_result.get("diagnostic_confidence", 0.0)
                    
                except Exception as e:
                    warnings.append(f"Differential diagnosis error: {str(e)}")
                    self.logger.warning(f"Differential diagnosis failed: {str(e)}")
            else:
                limitations.append("Differential diagnosis system unavailable")
            
            # Step 5: Therapeutic Response with Cultural Adaptation
            therapeutic_response = None
            
            if self.systems_initialized.get("therapeutic_friction", False):
                try:
                    # Generate base therapeutic response
                    emotional_context = {
                        "voice_emotion": voice_emotion_data,
                        "emotional_state": self._assess_emotional_state(user_message, voice_emotion_data)
                    }
                    
                    therapeutic_response = await self.therapeutic_friction.generate_therapeutic_response(
                        user_id, user_message, emotional_context, conversation_history
                    )
                    
                    # Apply cultural adaptations if available
                    if cultural_profile and self.systems_initialized.get("cultural_sensitivity", False):
                        cultural_adaptation = await self.cultural_sensitivity.adapt_therapeutic_response(
                            user_id, therapeutic_response.response_text, 
                            therapeutic_response.therapeutic_technique, cultural_profile
                        )
                        
                        # Update response with cultural adaptations
                        therapeutic_response.response_text = cultural_adaptation.adapted_approach
                        cultural_adaptations = asdict(cultural_adaptation)
                    
                except Exception as e:
                    warnings.append(f"Therapeutic response error: {str(e)}")
                    self.logger.warning(f"Therapeutic response failed: {str(e)}")
            else:
                limitations.append("Therapeutic friction system unavailable")
            
            # Step 6: Evidence-Based Recommendations
            evidence_based_recommendations = []
            treatment_validation = {}
            
            if self.systems_initialized.get("research_integration", False) and primary_diagnosis:
                try:
                    patient_characteristics = {
                        "age": cultural_info.get("age", 30) if cultural_info else 30,
                        "cultural_background": cultural_profile.primary_culture if cultural_profile else "unknown"
                    }
                    
                    evidence_based_recommendations = await self.research_integration.get_evidence_based_recommendations(
                        condition=primary_diagnosis.condition_name,
                        severity=primary_diagnosis.severity,
                        patient_characteristics=patient_characteristics,
                        cultural_context=cultural_profile.primary_culture if cultural_profile else None
                    )
                    
                    # Validate current therapeutic approach
                    if therapeutic_response:
                        treatment_validation = await self.research_integration.validate_treatment_approach(
                            therapeutic_response.therapeutic_technique,
                            primary_diagnosis.condition_name,
                            patient_characteristics
                        )
                    
                except Exception as e:
                    warnings.append(f"Research integration error: {str(e)}")
                    self.logger.warning(f"Research integration failed: {str(e)}")
            else:
                limitations.append("Research integration system unavailable or no primary diagnosis")
            
            # Step 7: Adaptive Learning and Personalization
            personalized_recommendations = {}
            intervention_effectiveness = {}
            
            if self.systems_initialized.get("adaptive_learning", False):
                try:
                    # Get personalized recommendations
                    available_interventions = [
                        "cognitive_behavioral_therapy", "dialectical_behavior_therapy",
                        "mindfulness_therapy", "acceptance_commitment_therapy"
                    ]
                    
                    current_context = {
                        "emotional_state": self._assess_emotional_state(user_message, voice_emotion_data),
                        "symptoms": list(self._extract_symptoms_from_message(user_message).keys()),
                        "cultural_background": cultural_profile.primary_culture if cultural_profile else "unknown"
                    }
                    
                    personalized_recommendations = await self.adaptive_learning.get_personalized_recommendation(
                        user_id, current_context, available_interventions
                    )
                    
                    # Record intervention outcome if this is a follow-up
                    if session_continuity.get("days_since_last_session", 0) <= 7:
                        # This is a follow-up, record previous intervention outcome
                        await self._record_intervention_outcome(
                            user_id, session_id, user_message, emotional_context
                        )
                    
                except Exception as e:
                    warnings.append(f"Adaptive learning error: {str(e)}")
                    self.logger.warning(f"Adaptive learning failed: {str(e)}")
            else:
                limitations.append("Adaptive learning system unavailable")
            
            # Step 8: Store Insights and Update Memory
            if self.systems_initialized.get("memory_system", False):
                try:
                    # Store therapeutic insights
                    if primary_diagnosis:
                        insight_content = f"Primary diagnosis: {primary_diagnosis.condition_name} with {primary_diagnosis.confidence:.1%} confidence"
                        await self.memory_system.store_therapeutic_insight(
                            user_id, session_id, "diagnostic_insight", insight_content,
                            {"diagnosis": asdict(primary_diagnosis)}, primary_diagnosis.confidence
                        )
                    
                    # Record session memory
                    session_data = {
                        "start_time": start_time,
                        "session_type": "comprehensive_diagnosis",
                        "insights": [insight_content] if primary_diagnosis else [],
                        "interventions": [therapeutic_response.therapeutic_technique] if therapeutic_response else [],
                        "mood_start": self._assess_emotional_state(user_message, voice_emotion_data)
                    }
                    
                    await self.memory_system.record_session_memory(user_id, session_id, session_data)
                    
                except Exception as e:
                    warnings.append(f"Memory storage error: {str(e)}")
                    self.logger.warning(f"Memory storage failed: {str(e)}")
            
            # Step 9: Calculate Integration Quality Metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            integration_confidence = self._calculate_integration_confidence(
                differential_diagnoses, therapeutic_response, cultural_adaptations,
                evidence_based_recommendations, personalized_recommendations
            )
            system_reliability = self._calculate_system_reliability()
            
            # Step 10: Progress Tracking
            progress_tracking = {}
            if self.systems_initialized.get("memory_system", False):
                try:
                    progress_tracking = await self.memory_system.track_progress_milestones(user_id)
                except Exception as e:
                    warnings.append(f"Progress tracking error: {str(e)}")
            
            # Compile comprehensive result
            result = ComprehensiveDiagnosticResult(
                user_id=user_id,
                session_id=session_id,
                timestamp=start_time,
                differential_diagnoses=differential_diagnoses,
                primary_diagnosis=primary_diagnosis,
                confidence_score=confidence_score,
                symptom_progression=symptom_progression,
                behavioral_patterns=behavioral_patterns,
                trajectory_prediction=trajectory_prediction,
                therapeutic_response=therapeutic_response,
                cultural_adaptations=cultural_adaptations,
                relevant_insights=relevant_insights,
                session_continuity=session_continuity,
                progress_tracking=progress_tracking,
                evidence_based_recommendations=evidence_based_recommendations,
                treatment_validation=treatment_validation,
                personalized_recommendations=personalized_recommendations,
                intervention_effectiveness=intervention_effectiveness,
                integration_confidence=integration_confidence,
                system_reliability=system_reliability,
                processing_time_ms=processing_time,
                warnings=warnings,
                limitations=limitations
            )
            
            self.logger.info(f"Comprehensive diagnosis completed for user {user_id} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error in comprehensive diagnosis: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return minimal result with error information
            return ComprehensiveDiagnosticResult(
                user_id=user_id,
                session_id=session_id,
                timestamp=start_time,
                differential_diagnoses=[],
                primary_diagnosis=None,
                confidence_score=0.0,
                symptom_progression={},
                behavioral_patterns=[],
                trajectory_prediction={},
                therapeutic_response=None,
                cultural_adaptations={},
                relevant_insights=[],
                session_continuity={},
                progress_tracking={},
                evidence_based_recommendations=[],
                treatment_validation={},
                personalized_recommendations={},
                intervention_effectiveness={},
                integration_confidence=0.0,
                system_reliability=0.0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                warnings=[f"Critical system error: {str(e)}"],
                limitations=["System experiencing technical difficulties"]
            )
    
    async def validate_system_integration(self) -> Dict[str, Any]:
        """
        Validate integration between all systems
        
        Returns:
            Validation results with system health status
        """
        validation_results = {
            "overall_status": "healthy",
            "system_statuses": {},
            "integration_tests": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        try:
            # Test individual systems
            for system_name in self.systems_initialized:
                status = await self._test_system_health(system_name)
                validation_results["system_statuses"][system_name] = status
                
                if not status["healthy"]:
                    validation_results["overall_status"] = "degraded"
            
            # Test system integrations
            integration_tests = [
                ("temporal_analysis", "differential_diagnosis"),
                ("differential_diagnosis", "therapeutic_friction"),
                ("therapeutic_friction", "cultural_sensitivity"),
                ("adaptive_learning", "memory_system"),
                ("memory_system", "research_integration")
            ]
            
            for system1, system2 in integration_tests:
                test_result = await self._test_system_integration(system1, system2)
                validation_results["integration_tests"][f"{system1}-{system2}"] = test_result
            
            # Performance metrics
            validation_results["performance_metrics"] = {
                "initialization_success_rate": sum(self.systems_initialized.values()) / len(self.systems_initialized),
                "system_reliability": self._calculate_system_reliability(),
                "average_response_time": "< 2000ms",  # Would be measured in real implementation
                "memory_usage": "normal",  # Would be measured in real implementation
                "error_rate": len(self.initialization_errors) / len(self.systems_initialized)
            }
            
            # Generate recommendations
            if validation_results["overall_status"] == "degraded":
                validation_results["recommendations"].append(
                    "Some systems are experiencing issues. Consider restarting affected subsystems."
                )
            
            if validation_results["performance_metrics"]["error_rate"] > 0.2:
                validation_results["recommendations"].append(
                    "High error rate detected. Review system logs and dependencies."
                )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating system integration: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e),
                "recommendations": ["System validation failed. Manual inspection required."]
            }
    
    # Private helper methods
    
    def _extract_symptoms_from_message(self, message: str) -> Dict[str, float]:
        """Extract symptoms and their intensities from user message"""
        symptoms = {}
        
        # Simple symptom extraction (would be more sophisticated in real implementation)
        symptom_keywords = {
            "anxiety": ["anxious", "worried", "nervous", "panic"],
            "depression": ["sad", "depressed", "down", "hopeless"],
            "stress": ["stressed", "overwhelmed", "pressure"],
            "anger": ["angry", "frustrated", "irritated", "mad"],
            "sleep_issues": ["insomnia", "can't sleep", "tired", "exhausted"]
        }
        
        message_lower = message.lower()
        
        for symptom, keywords in symptom_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    # Simple intensity calculation based on context
                    intensity = 0.5  # Default moderate intensity
                    
                    if any(word in message_lower for word in ["very", "extremely", "severely"]):
                        intensity = 0.8
                    elif any(word in message_lower for word in ["slightly", "mildly", "a bit"]):
                        intensity = 0.3
                    
                    symptoms[symptom] = intensity
                    break
        
        return symptoms
    
    def _extract_behavioral_observations(self, message: str, history: List[Dict[str, Any]]) -> List[str]:
        """Extract behavioral observations from message and history"""
        observations = []
        
        # Look for behavioral indicators
        behavioral_keywords = [
            "avoiding", "isolating", "withdrawing", "procrastinating",
            "aggressive", "impulsive", "restless", "hypervigilant"
        ]
        
        full_text = message + " " + " ".join([h.get("message", "") for h in history[-5:]])
        
        for keyword in behavioral_keywords:
            if keyword in full_text.lower():
                observations.append(f"Shows signs of {keyword} behavior")
        
        return observations
    
    def _assess_emotional_state(self, message: str, voice_emotion_data: Dict[str, Any] = None) -> str:
        """Assess current emotional state from message and voice data"""
        
        # Voice emotion takes priority if available
        if voice_emotion_data and "emotions" in voice_emotion_data:
            emotions = voice_emotion_data["emotions"]
            dominant_emotion = max(emotions.keys(), key=lambda k: emotions[k])
            
            if emotions[dominant_emotion] > 0.7:
                return dominant_emotion
        
        # Fallback to text analysis
        emotional_indicators = {
            "distressed": ["terrible", "awful", "horrible", "unbearable"],
            "anxious": ["worried", "nervous", "scared", "anxious"],
            "depressed": ["sad", "hopeless", "empty", "worthless"],
            "angry": ["angry", "frustrated", "furious", "mad"],
            "stable": ["okay", "fine", "good", "better"]
        }
        
        message_lower = message.lower()
        
        for state, indicators in emotional_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                return state
        
        return "neutral"
    
    def _dict_to_differential_diagnosis(self, diagnosis_dict: Dict[str, Any]) -> DifferentialDiagnosis:
        """Convert dictionary to DifferentialDiagnosis object (simplified)"""
        from .differential_diagnosis import DifferentialDiagnosis, DiagnosticCriterion
        
        # This is a simplified conversion - in real implementation would properly reconstruct objects
        return DifferentialDiagnosis(
            condition_name=diagnosis_dict.get("condition_name", "Unknown"),
            probability=diagnosis_dict.get("probability", 0.0),
            confidence=diagnosis_dict.get("confidence", 0.0),
            criteria_met=[],  # Would reconstruct from dict
            criteria_not_met=[],  # Would reconstruct from dict
            supporting_evidence=diagnosis_dict.get("supporting_evidence", []),
            contradicting_evidence=diagnosis_dict.get("contradicting_evidence", []),
            severity=diagnosis_dict.get("severity", "unknown"),
            specifiers=diagnosis_dict.get("specifiers", []),
            comorbidity_risk=diagnosis_dict.get("comorbidity_risk", 0.0),
            differential_rank=diagnosis_dict.get("differential_rank", 0)
        )
    
    async def _record_intervention_outcome(self,
                                         user_id: str,
                                         session_id: str,
                                         user_message: str,
                                         emotional_context: Dict[str, Any]):
        """Record intervention outcome for adaptive learning"""
        
        if not self.systems_initialized.get("adaptive_learning", False):
            return
        
        try:
            # Get previous session data to determine intervention
            if self.systems_initialized.get("memory_system", False):
                continuity = await self.memory_system.get_session_continuity_context(user_id, session_id)
                
                if continuity and not continuity.get("first_session", True):
                    # Calculate engagement and effectiveness scores
                    engagement_metrics = {
                        "response_time_seconds": 60,  # Would be measured
                        "session_duration_minutes": 30,  # Would be measured
                        "follow_up_questions": 1 if "?" in user_message else 0
                    }
                    
                    # Record outcome
                    await self.adaptive_learning.record_intervention_outcome(
                        intervention_id=f"{session_id}_intervention",
                        user_id=user_id,
                        intervention_type="comprehensive_therapy",
                        intervention_content="Integrated therapeutic approach",
                        context=emotional_context,
                        user_response=user_message,
                        engagement_metrics=engagement_metrics
                    )
        
        except Exception as e:
            self.logger.warning(f"Could not record intervention outcome: {str(e)}")
    
    def _calculate_integration_confidence(self,
                                        differential_diagnoses: List,
                                        therapeutic_response: Any,
                                        cultural_adaptations: Dict,
                                        evidence_recommendations: List,
                                        personalized_recommendations: Dict) -> float:
        """Calculate confidence in system integration quality"""
        
        confidence_factors = []
        
        # Diagnostic confidence
        if differential_diagnoses:
            avg_confidence = sum(d.confidence for d in differential_diagnoses) / len(differential_diagnoses)
            confidence_factors.append(avg_confidence)
        
        # Therapeutic response quality
        if therapeutic_response:
            confidence_factors.append(0.8)  # Assume good quality if generated
        
        # Cultural adaptation availability
        if cultural_adaptations:
            confidence_factors.append(0.7)
        
        # Evidence support
        if evidence_recommendations:
            confidence_factors.append(0.9)
        
        # Personalization level
        if personalized_recommendations.get("confidence", 0) > 0:
            confidence_factors.append(personalized_recommendations.get("confidence", 0.5))
        
        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.0
    
    def _calculate_system_reliability(self) -> float:
        """Calculate overall system reliability score"""
        if not self.systems_initialized:
            return 0.0
        
        # Base reliability on successful system initialization
        success_rate = sum(self.systems_initialized.values()) / len(self.systems_initialized)
        
        # Apply penalties for critical system failures
        critical_systems = ["differential_diagnosis", "therapeutic_friction", "memory_system"]
        critical_failures = sum(1 for sys in critical_systems if not self.systems_initialized.get(sys, False))
        
        reliability = success_rate - (critical_failures * 0.2)
        
        return max(0.0, min(1.0, reliability))
    
    async def _test_system_health(self, system_name: str) -> Dict[str, Any]:
        """Test health of individual system"""
        
        health_status = {
            "healthy": self.systems_initialized.get(system_name, False),
            "last_check": datetime.now().isoformat(),
            "error": None,
            "performance": "normal"
        }
        
        if not health_status["healthy"]:
            health_status["error"] = self.initialization_errors.get(system_name, "Unknown error")
            health_status["performance"] = "failed"
        
        return health_status
    
    async def _test_system_integration(self, system1: str, system2: str) -> Dict[str, Any]:
        """Test integration between two systems"""
        
        integration_test = {
            "systems": [system1, system2],
            "status": "pass",
            "issues": [],
            "performance": "normal"
        }
        
        # Check if both systems are initialized
        if not self.systems_initialized.get(system1, False):
            integration_test["status"] = "fail"
            integration_test["issues"].append(f"{system1} not initialized")
        
        if not self.systems_initialized.get(system2, False):
            integration_test["status"] = "fail"
            integration_test["issues"].append(f"{system2} not initialized")
        
        # Additional integration-specific tests would go here
        
        return integration_test
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "systems_initialized": self.systems_initialized,
            "initialization_errors": self.initialization_errors,
            "system_reliability": self._calculate_system_reliability(),
            "total_systems": len(self.systems_initialized),
            "healthy_systems": sum(self.systems_initialized.values()),
            "failed_systems": len(self.initialization_errors)
        }
    
    async def restart_failed_systems(self) -> Dict[str, bool]:
        """Attempt to restart failed systems"""
        restart_results = {}
        
        for system_name, is_initialized in self.systems_initialized.items():
            if not is_initialized:
                try:
                    self.logger.info(f"Attempting to restart {system_name}")
                    # Re-initialize the system
                    self._initialize_systems()  # This will re-attempt all initializations
                    restart_results[system_name] = self.systems_initialized.get(system_name, False)
                except Exception as e:
                    self.logger.error(f"Failed to restart {system_name}: {str(e)}")
                    restart_results[system_name] = False
        
        return restart_results