"""
Differential Diagnosis Engine

This module implements advanced differential diagnosis capabilities that consider
multiple conditions, assign probability scores, detect comorbidities, and provide
evidence-based reasoning for diagnostic conclusions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np
from collections import defaultdict
import re

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger
from ..models.llm import get_llm

logger = get_logger(__name__)

@dataclass
class DiagnosticCriterion:
    """Single diagnostic criterion"""
    criterion_id: str
    description: str
    weight: float  # Importance weight (0.0 to 1.0)
    met: bool
    confidence: float  # Confidence that criterion is met (0.0 to 1.0)
    evidence: List[str]
    source: str  # DSM-5, ICD-11, clinical observation, etc.

@dataclass
class DifferentialDiagnosis:
    """Individual diagnosis in differential"""
    condition_name: str
    probability: float  # Overall probability (0.0 to 1.0)
    confidence: float  # Confidence in this assessment
    criteria_met: List[DiagnosticCriterion]
    criteria_not_met: List[DiagnosticCriterion]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    severity: str  # mild, moderate, severe
    specifiers: List[str]  # Additional specifiers
    comorbidity_risk: float  # Risk of comorbid conditions
    differential_rank: int  # Ranking in differential

@dataclass
class ComorbidityAssessment:
    """Assessment of potential comorbid conditions"""
    primary_condition: str
    comorbid_condition: str
    likelihood: float
    shared_symptoms: List[str]
    unique_symptoms: List[str]
    interaction_effects: List[str]
    treatment_implications: List[str]

class DifferentialDiagnosisEngine:
    """
    Advanced differential diagnosis engine that evaluates multiple potential
    conditions and provides evidence-based diagnostic reasoning.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the differential diagnosis engine"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # Load diagnostic criteria databases
        self.dsm5_criteria = self._load_dsm5_criteria()
        self.icd11_criteria = self._load_icd11_criteria()
        self.comorbidity_patterns = self._load_comorbidity_patterns()
        
        # Diagnostic thresholds
        self.min_probability_threshold = 0.1  # Minimum to include in differential
        self.high_confidence_threshold = 0.8  # High confidence diagnosis
        self.comorbidity_threshold = 0.6  # Threshold for suggesting comorbidity
        
    async def generate_differential_diagnosis(self,
                                            symptoms: List[str],
                                            behavioral_observations: List[str],
                                            temporal_patterns: Dict[str, Any],
                                            voice_emotion_data: Dict[str, Any] = None,
                                            personality_data: Dict[str, Any] = None,
                                            user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive differential diagnosis
        
        Args:
            symptoms: List of reported symptoms
            behavioral_observations: Observed behavioral patterns
            temporal_patterns: Temporal analysis data
            voice_emotion_data: Voice emotion analysis results
            personality_data: Personality assessment data
            user_context: Additional user context
            
        Returns:
            Comprehensive differential diagnosis with ranked conditions
        """
        try:
            self.logger.info("Starting differential diagnosis generation")
            
            # Step 1: Identify potential conditions based on symptoms
            candidate_conditions = await self._identify_candidate_conditions(
                symptoms, behavioral_observations
            )
            
            # Step 2: Evaluate each condition against diagnostic criteria
            differential_diagnoses = []
            for condition in candidate_conditions:
                diagnosis = await self._evaluate_condition(
                    condition, symptoms, behavioral_observations, 
                    temporal_patterns, voice_emotion_data, personality_data
                )
                if diagnosis.probability >= self.min_probability_threshold:
                    differential_diagnoses.append(diagnosis)
            
            # Step 3: Rank diagnoses by probability and confidence
            differential_diagnoses.sort(
                key=lambda x: (x.probability * x.confidence), reverse=True
            )
            
            # Assign rankings
            for i, diagnosis in enumerate(differential_diagnoses):
                diagnosis.differential_rank = i + 1
            
            # Step 4: Assess comorbidities
            comorbidity_assessments = await self._assess_comorbidities(
                differential_diagnoses, symptoms
            )
            
            # Step 5: Generate clinical reasoning
            clinical_reasoning = await self._generate_clinical_reasoning(
                differential_diagnoses, symptoms, temporal_patterns
            )
            
            # Step 6: Provide recommendations
            recommendations = await self._generate_diagnostic_recommendations(
                differential_diagnoses, comorbidity_assessments
            )
            
            return {
                "differential_diagnoses": [asdict(d) for d in differential_diagnoses],
                "primary_diagnosis": asdict(differential_diagnoses[0]) if differential_diagnoses else None,
                "comorbidity_assessments": [asdict(c) for c in comorbidity_assessments],
                "clinical_reasoning": clinical_reasoning,
                "recommendations": recommendations,
                "diagnostic_confidence": self._calculate_overall_confidence(differential_diagnoses),
                "timestamp": datetime.now().isoformat(),
                "evidence_summary": self._summarize_evidence(differential_diagnoses)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating differential diagnosis: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _identify_candidate_conditions(self,
                                           symptoms: List[str],
                                           behavioral_observations: List[str]) -> List[str]:
        """Identify candidate conditions based on symptoms and observations"""
        candidate_conditions = set()
        
        # Map symptoms to potential conditions using DSM-5 criteria
        for symptom in symptoms:
            normalized_symptom = symptom.lower().strip()
            for condition, criteria in self.dsm5_criteria.items():
                if self._symptom_matches_criteria(normalized_symptom, criteria):
                    candidate_conditions.add(condition)
        
        # Map behavioral observations
        for observation in behavioral_observations:
            normalized_obs = observation.lower().strip()
            for condition, criteria in self.dsm5_criteria.items():
                if self._observation_matches_criteria(normalized_obs, criteria):
                    candidate_conditions.add(condition)
        
        # Use LLM for additional candidate identification
        llm_candidates = await self._llm_identify_candidates(symptoms, behavioral_observations)
        candidate_conditions.update(llm_candidates)
        
        return list(candidate_conditions)
    
    async def _evaluate_condition(self,
                                condition: str,
                                symptoms: List[str],
                                behavioral_observations: List[str],
                                temporal_patterns: Dict[str, Any],
                                voice_emotion_data: Dict[str, Any] = None,
                                personality_data: Dict[str, Any] = None) -> DifferentialDiagnosis:
        """Evaluate a specific condition against all available evidence"""
        
        # Get diagnostic criteria for the condition
        criteria = self.dsm5_criteria.get(condition, {})
        if not criteria:
            criteria = self.icd11_criteria.get(condition, {})
        
        if not criteria:
            # Fallback: generate criteria using LLM
            criteria = await self._generate_criteria_with_llm(condition)
        
        # Evaluate each criterion
        criteria_met = []
        criteria_not_met = []
        supporting_evidence = []
        contradicting_evidence = []
        
        for criterion_data in criteria.get("criteria", []):
            criterion = await self._evaluate_criterion(
                criterion_data, symptoms, behavioral_observations,
                temporal_patterns, voice_emotion_data, personality_data
            )
            
            if criterion.met:
                criteria_met.append(criterion)
                supporting_evidence.extend(criterion.evidence)
            else:
                criteria_not_met.append(criterion)
                if criterion.evidence:  # Evidence against the criterion
                    contradicting_evidence.extend(criterion.evidence)
        
        # Calculate probability based on criteria met
        probability = self._calculate_condition_probability(
            criteria_met, criteria_not_met, criteria.get("required_criteria", 0)
        )
        
        # Calculate confidence
        confidence = self._calculate_condition_confidence(
            criteria_met, criteria_not_met, len(symptoms)
        )
        
        # Determine severity
        severity = self._determine_severity(
            criteria_met, temporal_patterns, voice_emotion_data
        )
        
        # Extract specifiers
        specifiers = self._extract_specifiers(
            condition, criteria_met, temporal_patterns
        )
        
        # Calculate comorbidity risk
        comorbidity_risk = self._calculate_comorbidity_risk(
            condition, symptoms, behavioral_observations
        )
        
        return DifferentialDiagnosis(
            condition_name=condition,
            probability=probability,
            confidence=confidence,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            severity=severity,
            specifiers=specifiers,
            comorbidity_risk=comorbidity_risk,
            differential_rank=0  # Will be set later
        )
    
    async def _evaluate_criterion(self,
                                criterion_data: Dict[str, Any],
                                symptoms: List[str],
                                behavioral_observations: List[str],
                                temporal_patterns: Dict[str, Any],
                                voice_emotion_data: Dict[str, Any] = None,
                                personality_data: Dict[str, Any] = None) -> DiagnosticCriterion:
        """Evaluate a single diagnostic criterion"""
        
        criterion_id = criterion_data.get("id", "unknown")
        description = criterion_data.get("description", "")
        weight = criterion_data.get("weight", 0.5)
        
        # Check against symptoms
        symptom_matches = []
        for symptom in symptoms:
            if self._text_matches_criterion(symptom, criterion_data):
                symptom_matches.append(f"Symptom: {symptom}")
        
        # Check against behavioral observations
        behavior_matches = []
        for observation in behavioral_observations:
            if self._text_matches_criterion(observation, criterion_data):
                behavior_matches.append(f"Behavior: {observation}")
        
        # Check temporal patterns
        temporal_matches = []
        if temporal_patterns and self._temporal_matches_criterion(temporal_patterns, criterion_data):
            temporal_matches.append("Temporal pattern supports criterion")
        
        # Check voice emotion data
        emotion_matches = []
        if voice_emotion_data and self._emotion_matches_criterion(voice_emotion_data, criterion_data):
            emotion_matches.append("Voice emotion analysis supports criterion")
        
        # Check personality data
        personality_matches = []
        if personality_data and self._personality_matches_criterion(personality_data, criterion_data):
            personality_matches.append("Personality assessment supports criterion")
        
        # Combine all evidence
        all_evidence = symptom_matches + behavior_matches + temporal_matches + emotion_matches + personality_matches
        
        # Determine if criterion is met
        met = len(all_evidence) > 0
        
        # Calculate confidence based on strength and quantity of evidence
        confidence = min(1.0, len(all_evidence) * 0.3) if met else 0.0
        
        return DiagnosticCriterion(
            criterion_id=criterion_id,
            description=description,
            weight=weight,
            met=met,
            confidence=confidence,
            evidence=all_evidence,
            source=criterion_data.get("source", "DSM-5")
        )
    
    def _calculate_condition_probability(self,
                                       criteria_met: List[DiagnosticCriterion],
                                       criteria_not_met: List[DiagnosticCriterion],
                                       required_criteria: int) -> float:
        """Calculate probability that condition is present"""
        
        if not criteria_met:
            return 0.0
        
        # Weighted score of met criteria
        met_score = sum(c.weight * c.confidence for c in criteria_met)
        total_possible = sum(c.weight for c in criteria_met + criteria_not_met)
        
        if total_possible == 0:
            return 0.0
        
        # Base probability from criteria
        base_probability = met_score / total_possible
        
        # Bonus for meeting required criteria threshold
        if len(criteria_met) >= required_criteria:
            base_probability *= 1.2  # 20% bonus
        
        # Penalty for strong contradicting evidence
        strong_contradictions = [c for c in criteria_not_met if c.weight > 0.8]
        if strong_contradictions:
            base_probability *= 0.8  # 20% penalty
        
        return min(1.0, base_probability)
    
    def _calculate_condition_confidence(self,
                                      criteria_met: List[DiagnosticCriterion],
                                      criteria_not_met: List[DiagnosticCriterion],
                                      total_symptoms: int) -> float:
        """Calculate confidence in the condition assessment"""
        
        if not criteria_met:
            return 0.0
        
        # Average confidence of met criteria
        avg_confidence = np.mean([c.confidence for c in criteria_met])
        
        # Factor in number of criteria evaluated
        total_criteria = len(criteria_met) + len(criteria_not_met)
        coverage_factor = min(1.0, total_criteria / 5)  # Ideal: 5+ criteria
        
        # Factor in symptom coverage
        symptom_factor = min(1.0, total_symptoms / 3)  # Ideal: 3+ symptoms
        
        return avg_confidence * coverage_factor * symptom_factor
    
    def _determine_severity(self,
                          criteria_met: List[DiagnosticCriterion],
                          temporal_patterns: Dict[str, Any],
                          voice_emotion_data: Dict[str, Any] = None) -> str:
        """Determine severity level based on criteria and patterns"""
        
        if not criteria_met:
            return "none"
        
        # Count high-weight criteria (severe symptoms)
        severe_criteria = [c for c in criteria_met if c.weight > 0.8 and c.confidence > 0.7]
        moderate_criteria = [c for c in criteria_met if 0.5 < c.weight <= 0.8 and c.confidence > 0.6]
        
        # Factor in temporal patterns
        severity_score = 0
        if severe_criteria:
            severity_score += len(severe_criteria) * 3
        if moderate_criteria:
            severity_score += len(moderate_criteria) * 2
        
        # Temporal severity factors
        if temporal_patterns:
            trends = temporal_patterns.get("trends", {})
            if trends.get("direction") == "worsening":
                severity_score += 2
            if trends.get("volatility", 0) > 0.5:
                severity_score += 1
        
        # Voice emotion severity factors
        if voice_emotion_data:
            emotions = voice_emotion_data.get("emotions", {})
            distress_emotions = ["anxiety", "sadness", "anger", "fear"]
            high_distress = sum(emotions.get(e, 0) for e in distress_emotions if emotions.get(e, 0) > 0.7)
            if high_distress > 2:
                severity_score += 2
            elif high_distress > 0:
                severity_score += 1
        
        # Classify severity
        if severity_score >= 8:
            return "severe"
        elif severity_score >= 4:
            return "moderate"
        elif severity_score >= 1:
            return "mild"
        else:
            return "subclinical"
    
    async def _assess_comorbidities(self,
                                  differential_diagnoses: List[DifferentialDiagnosis],
                                  symptoms: List[str]) -> List[ComorbidityAssessment]:
        """Assess potential comorbid conditions"""
        
        if len(differential_diagnoses) < 2:
            return []
        
        comorbidity_assessments = []
        
        # Check each pair of conditions
        for i, primary in enumerate(differential_diagnoses):
            if primary.probability < 0.3:  # Only consider likely primary diagnoses
                continue
                
            for secondary in differential_diagnoses[i+1:]:
                if secondary.probability < 0.2:  # Lower threshold for secondary
                    continue
                
                # Check if these conditions commonly co-occur
                comorbidity_data = self.comorbidity_patterns.get(
                    f"{primary.condition_name}+{secondary.condition_name}"
                ) or self.comorbidity_patterns.get(
                    f"{secondary.condition_name}+{primary.condition_name}"
                )
                
                if comorbidity_data or self._conditions_share_symptoms(primary, secondary):
                    assessment = await self._evaluate_comorbidity(
                        primary, secondary, symptoms, comorbidity_data
                    )
                    if assessment.likelihood >= self.comorbidity_threshold:
                        comorbidity_assessments.append(assessment)
        
        # Sort by likelihood
        comorbidity_assessments.sort(key=lambda x: x.likelihood, reverse=True)
        
        return comorbidity_assessments
    
    async def _evaluate_comorbidity(self,
                                  primary: DifferentialDiagnosis,
                                  secondary: DifferentialDiagnosis,
                                  symptoms: List[str],
                                  comorbidity_data: Dict[str, Any] = None) -> ComorbidityAssessment:
        """Evaluate likelihood of comorbidity between two conditions"""
        
        # Find shared and unique symptoms
        primary_symptoms = set(e.split(": ")[1] for e in primary.supporting_evidence if ": " in e)
        secondary_symptoms = set(e.split(": ")[1] for e in secondary.supporting_evidence if ": " in e)
        
        shared_symptoms = list(primary_symptoms.intersection(secondary_symptoms))
        primary_unique = list(primary_symptoms - secondary_symptoms)
        secondary_unique = list(secondary_symptoms - primary_symptoms)
        
        # Calculate base likelihood
        base_likelihood = min(primary.probability, secondary.probability)
        
        # Adjust based on known comorbidity patterns
        if comorbidity_data:
            pattern_likelihood = comorbidity_data.get("likelihood", 0.5)
            base_likelihood = (base_likelihood + pattern_likelihood) / 2
        
        # Adjust based on symptom overlap
        if shared_symptoms:
            overlap_factor = len(shared_symptoms) / max(len(symptoms), 1)
            base_likelihood += overlap_factor * 0.2
        
        # Generate interaction effects and treatment implications
        interaction_effects = await self._generate_interaction_effects(
            primary.condition_name, secondary.condition_name
        )
        
        treatment_implications = await self._generate_treatment_implications(
            primary.condition_name, secondary.condition_name
        )
        
        return ComorbidityAssessment(
            primary_condition=primary.condition_name,
            comorbid_condition=secondary.condition_name,
            likelihood=min(1.0, base_likelihood),
            shared_symptoms=shared_symptoms,
            unique_symptoms=primary_unique + secondary_unique,
            interaction_effects=interaction_effects,
            treatment_implications=treatment_implications
        )
    
    # Helper methods for loading and matching criteria
    
    def _load_dsm5_criteria(self) -> Dict[str, Any]:
        """Load DSM-5 diagnostic criteria"""
        # This would typically load from a comprehensive database
        # For now, return a sample structure
        return {
            "Major Depressive Disorder": {
                "criteria": [
                    {
                        "id": "A1",
                        "description": "Depressed mood most of the day",
                        "weight": 0.9,
                        "keywords": ["depressed", "sad", "hopeless", "empty"],
                        "source": "DSM-5"
                    },
                    {
                        "id": "A2", 
                        "description": "Diminished interest or pleasure",
                        "weight": 0.9,
                        "keywords": ["anhedonia", "no interest", "no pleasure", "apathy"],
                        "source": "DSM-5"
                    },
                    {
                        "id": "A3",
                        "description": "Significant weight loss or gain",
                        "weight": 0.6,
                        "keywords": ["weight loss", "weight gain", "appetite"],
                        "source": "DSM-5"
                    }
                ],
                "required_criteria": 5,
                "duration": "2 weeks"
            },
            "Generalized Anxiety Disorder": {
                "criteria": [
                    {
                        "id": "A",
                        "description": "Excessive anxiety and worry",
                        "weight": 0.9,
                        "keywords": ["anxiety", "worry", "anxious", "nervous"],
                        "source": "DSM-5"
                    },
                    {
                        "id": "B",
                        "description": "Difficulty controlling worry",
                        "weight": 0.8,
                        "keywords": ["can't control", "persistent worry", "uncontrollable"],
                        "source": "DSM-5"
                    }
                ],
                "required_criteria": 3,
                "duration": "6 months"
            }
        }
    
    def _load_icd11_criteria(self) -> Dict[str, Any]:
        """Load ICD-11 diagnostic criteria"""
        # Similar structure to DSM-5 but with ICD-11 criteria
        return {}
    
    def _load_comorbidity_patterns(self) -> Dict[str, Any]:
        """Load known comorbidity patterns"""
        return {
            "Major Depressive Disorder+Generalized Anxiety Disorder": {
                "likelihood": 0.8,
                "prevalence": "Very common",
                "interaction_effects": ["Increased severity", "Treatment resistance"],
                "treatment_implications": ["Consider combined therapy", "Monitor for suicidal ideation"]
            }
        }
    
    def _symptom_matches_criteria(self, symptom: str, criteria: Dict[str, Any]) -> bool:
        """Check if symptom matches any criteria"""
        for criterion in criteria.get("criteria", []):
            if self._text_matches_criterion(symptom, criterion):
                return True
        return False
    
    def _observation_matches_criteria(self, observation: str, criteria: Dict[str, Any]) -> bool:
        """Check if behavioral observation matches criteria"""
        return self._symptom_matches_criteria(observation, criteria)
    
    def _text_matches_criterion(self, text: str, criterion: Dict[str, Any]) -> bool:
        """Check if text matches a specific criterion"""
        keywords = criterion.get("keywords", [])
        text_lower = text.lower()
        
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def _temporal_matches_criterion(self, temporal_patterns: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
        """Check if temporal patterns support a criterion"""
        # Implementation would check temporal requirements
        return False
    
    def _emotion_matches_criterion(self, voice_emotion_data: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
        """Check if voice emotion data supports a criterion"""
        # Implementation would analyze emotion patterns
        return False
    
    def _personality_matches_criterion(self, personality_data: Dict[str, Any], criterion: Dict[str, Any]) -> bool:
        """Check if personality data supports a criterion"""
        # Implementation would analyze personality factors
        return False
    
    async def _llm_identify_candidates(self, symptoms: List[str], behavioral_observations: List[str]) -> List[str]:
        """Use LLM to identify additional candidate conditions"""
        try:
            prompt = f"""
            Based on the following symptoms and behavioral observations, identify potential mental health conditions that should be considered in a differential diagnosis:
            
            Symptoms: {', '.join(symptoms)}
            Behavioral Observations: {', '.join(behavioral_observations)}
            
            Return only the condition names, one per line.
            """
            
            response = await self.llm.generate_response(prompt)
            conditions = [line.strip() for line in response.split('\n') if line.strip()]
            return conditions[:10]  # Limit to top 10
            
        except Exception as e:
            self.logger.error(f"Error using LLM for candidate identification: {str(e)}")
            return []
    
    async def _generate_criteria_with_llm(self, condition: str) -> Dict[str, Any]:
        """Generate diagnostic criteria using LLM for unknown conditions"""
        try:
            prompt = f"""
            Provide the key diagnostic criteria for {condition} in a structured format.
            Include the main symptoms, behavioral indicators, and any duration requirements.
            """
            
            response = await self.llm.generate_response(prompt)
            # Parse response into criteria structure
            # This is a simplified implementation
            return {
                "criteria": [
                    {
                        "id": "LLM1",
                        "description": response[:100],
                        "weight": 0.5,
                        "keywords": [],
                        "source": "LLM-Generated"
                    }
                ],
                "required_criteria": 1
            }
            
        except Exception as e:
            self.logger.error(f"Error generating criteria with LLM: {str(e)}")
            return {"criteria": [], "required_criteria": 0}
    
    def _conditions_share_symptoms(self, condition1: DifferentialDiagnosis, condition2: DifferentialDiagnosis) -> bool:
        """Check if two conditions share significant symptoms"""
        symptoms1 = set(e for e in condition1.supporting_evidence)
        symptoms2 = set(e for e in condition2.supporting_evidence)
        
        overlap = len(symptoms1.intersection(symptoms2))
        total = len(symptoms1.union(symptoms2))
        
        return overlap / max(total, 1) > 0.3  # 30% overlap threshold
    
    async def _generate_interaction_effects(self, condition1: str, condition2: str) -> List[str]:
        """Generate interaction effects between conditions"""
        # This would use clinical knowledge or LLM
        return ["Symptom exacerbation", "Treatment complexity"]
    
    async def _generate_treatment_implications(self, condition1: str, condition2: str) -> List[str]:
        """Generate treatment implications for comorbid conditions"""
        return ["Integrated treatment approach recommended", "Monitor for drug interactions"]
    
    async def _generate_clinical_reasoning(self,
                                         differential_diagnoses: List[DifferentialDiagnosis],
                                         symptoms: List[str],
                                         temporal_patterns: Dict[str, Any]) -> str:
        """Generate clinical reasoning for the differential diagnosis"""
        
        if not differential_diagnoses:
            return "Insufficient evidence for diagnostic hypothesis generation."
        
        primary = differential_diagnoses[0]
        
        reasoning = f"Clinical reasoning for differential diagnosis:\n\n"
        reasoning += f"Primary consideration: {primary.condition_name} (probability: {primary.probability:.2f}, confidence: {primary.confidence:.2f})\n"
        
        reasoning += f"Supporting evidence:\n"
        for evidence in primary.supporting_evidence[:3]:  # Top 3 pieces of evidence
            reasoning += f"- {evidence}\n"
        
        if len(differential_diagnoses) > 1:
            reasoning += f"\nAlternative considerations:\n"
            for alt in differential_diagnoses[1:3]:  # Next 2 alternatives
                reasoning += f"- {alt.condition_name} (probability: {alt.probability:.2f})\n"
        
        if primary.contradicting_evidence:
            reasoning += f"\nFactors against primary diagnosis:\n"
            for evidence in primary.contradicting_evidence[:2]:
                reasoning += f"- {evidence}\n"
        
        return reasoning
    
    async def _generate_diagnostic_recommendations(self,
                                                 differential_diagnoses: List[DifferentialDiagnosis],
                                                 comorbidity_assessments: List[ComorbidityAssessment]) -> List[str]:
        """Generate diagnostic recommendations"""
        recommendations = []
        
        if not differential_diagnoses:
            return ["Continue symptom monitoring and reassessment"]
        
        primary = differential_diagnoses[0]
        
        if primary.confidence < 0.6:
            recommendations.append("Additional clinical assessment recommended due to diagnostic uncertainty")
        
        if primary.severity == "severe":
            recommendations.append("Urgent clinical evaluation recommended due to symptom severity")
        elif primary.severity == "moderate":
            recommendations.append("Clinical evaluation recommended within 1-2 weeks")
        
        if comorbidity_assessments:
            recommendations.append("Comprehensive assessment for potential comorbid conditions recommended")
        
        if primary.probability < 0.7:
            recommendations.append("Consider structured diagnostic interview for confirmation")
        
        return recommendations
    
    def _calculate_overall_confidence(self, differential_diagnoses: List[DifferentialDiagnosis]) -> float:
        """Calculate overall diagnostic confidence"""
        if not differential_diagnoses:
            return 0.0
        
        primary = differential_diagnoses[0]
        return primary.probability * primary.confidence
    
    def _summarize_evidence(self, differential_diagnoses: List[DifferentialDiagnosis]) -> Dict[str, Any]:
        """Summarize evidence across all diagnoses"""
        if not differential_diagnoses:
            return {}
        
        all_evidence = []
        for diagnosis in differential_diagnoses:
            all_evidence.extend(diagnosis.supporting_evidence)
        
        return {
            "total_evidence_points": len(all_evidence),
            "conditions_considered": len(differential_diagnoses),
            "high_confidence_diagnoses": len([d for d in differential_diagnoses if d.confidence > 0.8]),
            "evidence_strength": "strong" if len(all_evidence) > 10 else "moderate" if len(all_evidence) > 5 else "limited"
        }
    
    def _extract_specifiers(self,
                          condition: str,
                          criteria_met: List[DiagnosticCriterion],
                          temporal_patterns: Dict[str, Any]) -> List[str]:
        """Extract diagnostic specifiers"""
        specifiers = []
        
        # Severity specifier already handled
        
        # Episode specifiers for mood disorders
        if "depression" in condition.lower() or "bipolar" in condition.lower():
            if temporal_patterns.get("trends", {}).get("direction") == "improving":
                specifiers.append("in partial remission")
            elif temporal_patterns.get("trends", {}).get("direction") == "worsening":
                specifiers.append("with worsening course")
        
        # Anxiety specifiers
        if "anxiety" in condition.lower():
            high_intensity_criteria = [c for c in criteria_met if c.confidence > 0.8]
            if len(high_intensity_criteria) > 3:
                specifiers.append("with prominent anxiety")
        
        return specifiers
    
    def _calculate_comorbidity_risk(self,
                                  condition: str,
                                  symptoms: List[str],
                                  behavioral_observations: List[str]) -> float:
        """Calculate risk of comorbid conditions"""
        # Base risk on condition type
        high_comorbidity_conditions = [
            "Major Depressive Disorder",
            "Generalized Anxiety Disorder",
            "Borderline Personality Disorder"
        ]
        
        if condition in high_comorbidity_conditions:
            base_risk = 0.7
        else:
            base_risk = 0.3
        
        # Adjust based on symptom diversity
        symptom_diversity = len(set(symptoms)) / max(len(symptoms), 1)
        risk_adjustment = symptom_diversity * 0.3
        
        return min(1.0, base_risk + risk_adjustment)