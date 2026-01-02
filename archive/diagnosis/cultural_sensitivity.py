"""
Cultural Context Integration and Adaptation System

This module implements deep cultural sensitivity for mental health diagnosis and treatment,
adapting responses based on cultural perspectives, communication styles, and healing practices.
It addresses cultural stigma and provides culturally-relevant interventions.
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
class CulturalProfile:
    """User's cultural background and preferences"""
    user_id: str
    primary_culture: str
    cultural_subgroups: List[str]
    language_preferences: List[str]
    religious_background: Optional[str]
    traditional_healing_practices: List[str]
    mental_health_stigma_level: float  # 0.0 (no stigma) to 1.0 (high stigma)
    collectivist_vs_individualist: float  # 0.0 (collectivist) to 1.0 (individualist)
    communication_style: str  # direct, indirect, high_context, low_context
    family_involvement_preference: str  # high, medium, low
    authority_relationship: str  # hierarchical, egalitarian
    created_at: datetime
    last_updated: datetime

@dataclass
class CulturalAdaptation:
    """Adaptation strategy for cultural sensitivity"""
    adaptation_id: str
    culture_context: str
    original_approach: str
    adapted_approach: str
    reasoning: str
    cultural_considerations: List[str]
    language_adaptations: List[str]
    concept_translations: Dict[str, str]
    stigma_mitigation: List[str]
    traditional_integration: List[str]

@dataclass
class CulturalIntervention:
    """Culturally-adapted therapeutic intervention"""
    intervention_id: str
    base_intervention: str
    cultural_context: str
    adapted_content: str
    cultural_metaphors: List[str]
    traditional_practices_integrated: List[str]
    communication_style_adapted: str
    family_involvement_level: str
    expected_cultural_response: str

class CulturalSensitivityEngine:
    """
    Engine for integrating cultural context into mental health diagnosis and treatment,
    providing culturally-sensitive and adapted therapeutic responses.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the cultural sensitivity engine"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # Cultural profiles database
        self.user_profiles = {}
        
        # Cultural knowledge database
        self.cultural_knowledge = self._load_cultural_knowledge()
        self.stigma_patterns = self._load_stigma_patterns()
        self.communication_styles = self._load_communication_styles()
        self.traditional_practices = self._load_traditional_practices()
        
        # Adaptation cache
        self.adaptation_cache = {}
        
    async def assess_cultural_context(self,
                                    user_id: str,
                                    user_message: str,
                                    conversation_history: List[Dict[str, Any]],
                                    explicit_cultural_info: Dict[str, Any] = None) -> CulturalProfile:
        """
        Assess user's cultural context from conversation and explicit information
        
        Args:
            user_id: User identifier
            user_message: Current user message
            conversation_history: Previous conversation context
            explicit_cultural_info: Explicitly provided cultural information
            
        Returns:
            Cultural profile for the user
        """
        try:
            self.logger.info(f"Assessing cultural context for user {user_id}")
            
            # Get existing profile or create new one
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
            else:
                profile = CulturalProfile(
                    user_id=user_id,
                    primary_culture="unknown",
                    cultural_subgroups=[],
                    language_preferences=["english"],
                    religious_background=None,
                    traditional_healing_practices=[],
                    mental_health_stigma_level=0.5,
                    collectivist_vs_individualist=0.5,
                    communication_style="unknown",
                    family_involvement_preference="medium",
                    authority_relationship="unknown",
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
            
            # Update profile with explicit information
            if explicit_cultural_info:
                profile = self._update_profile_with_explicit_info(profile, explicit_cultural_info)
            
            # Infer cultural context from conversation
            cultural_indicators = await self._extract_cultural_indicators(
                user_message, conversation_history
            )
            
            profile = await self._update_profile_with_indicators(profile, cultural_indicators)
            
            # Assess communication style
            profile.communication_style = self._assess_communication_style(
                user_message, conversation_history
            )
            
            # Assess stigma level
            profile.mental_health_stigma_level = self._assess_mental_health_stigma(
                user_message, conversation_history, profile.primary_culture
            )
            
            # Assess cultural orientation
            profile.collectivist_vs_individualist = self._assess_cultural_orientation(
                user_message, conversation_history
            )
            
            profile.last_updated = datetime.now()
            self.user_profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error assessing cultural context: {str(e)}")
            return self._create_default_profile(user_id)
    
    async def adapt_therapeutic_response(self,
                                       user_id: str,
                                       base_response: str,
                                       therapeutic_approach: str,
                                       cultural_profile: CulturalProfile) -> CulturalAdaptation:
        """
        Adapt therapeutic response for cultural sensitivity
        
        Args:
            user_id: User identifier
            base_response: Original therapeutic response
            therapeutic_approach: Type of therapeutic approach
            cultural_profile: User's cultural profile
            
        Returns:
            Culturally adapted response
        """
        try:
            self.logger.info(f"Adapting therapeutic response for cultural context: {cultural_profile.primary_culture}")
            
            # Check cache for similar adaptations
            cache_key = f"{cultural_profile.primary_culture}_{therapeutic_approach}_{hash(base_response[:100])}"
            if cache_key in self.adaptation_cache:
                return self.adaptation_cache[cache_key]
            
            # Identify cultural considerations
            cultural_considerations = self._identify_cultural_considerations(
                base_response, cultural_profile
            )
            
            # Adapt language and concepts
            language_adaptations = await self._adapt_language_for_culture(
                base_response, cultural_profile
            )
            
            # Translate psychological concepts
            concept_translations = self._translate_psychological_concepts(
                base_response, cultural_profile
            )
            
            # Mitigate stigma
            stigma_mitigation = self._apply_stigma_mitigation(
                base_response, cultural_profile
            )
            
            # Integrate traditional practices
            traditional_integration = await self._integrate_traditional_practices(
                base_response, therapeutic_approach, cultural_profile
            )
            
            # Generate adapted response
            adapted_response = await self._generate_adapted_response(
                base_response, cultural_considerations, language_adaptations,
                concept_translations, stigma_mitigation, traditional_integration,
                cultural_profile
            )
            
            # Create adaptation object
            adaptation = CulturalAdaptation(
                adaptation_id=f"{user_id}_{int(datetime.now().timestamp())}",
                culture_context=cultural_profile.primary_culture,
                original_approach=base_response,
                adapted_approach=adapted_response,
                reasoning=self._generate_adaptation_reasoning(cultural_considerations),
                cultural_considerations=cultural_considerations,
                language_adaptations=language_adaptations,
                concept_translations=concept_translations,
                stigma_mitigation=stigma_mitigation,
                traditional_integration=traditional_integration
            )
            
            # Cache the adaptation
            self.adaptation_cache[cache_key] = adaptation
            
            # Store in vector database
            if self.vector_db:
                await self._store_cultural_adaptation(adaptation)
            
            return adaptation
            
        except Exception as e:
            self.logger.error(f"Error adapting therapeutic response: {str(e)}")
            return CulturalAdaptation(
                adaptation_id="error",
                culture_context=cultural_profile.primary_culture,
                original_approach=base_response,
                adapted_approach=base_response,  # Fallback to original
                reasoning="Error in cultural adaptation",
                cultural_considerations=[],
                language_adaptations=[],
                concept_translations={},
                stigma_mitigation=[],
                traditional_integration=[]
            )
    
    async def create_cultural_intervention(self,
                                         base_intervention: str,
                                         cultural_profile: CulturalProfile,
                                         specific_issue: str = None) -> CulturalIntervention:
        """
        Create culturally-adapted therapeutic intervention
        
        Args:
            base_intervention: Base therapeutic intervention
            cultural_profile: User's cultural profile
            specific_issue: Specific cultural issue to address
            
        Returns:
            Culturally-adapted intervention
        """
        try:
            # Generate cultural metaphors
            cultural_metaphors = await self._generate_cultural_metaphors(
                base_intervention, cultural_profile, specific_issue
            )
            
            # Integrate traditional practices
            traditional_practices = self._select_traditional_practices(
                base_intervention, cultural_profile
            )
            
            # Adapt communication style
            communication_adapted = self._adapt_communication_style(
                base_intervention, cultural_profile.communication_style
            )
            
            # Determine family involvement
            family_involvement = self._determine_family_involvement(
                base_intervention, cultural_profile
            )
            
            # Generate adapted content
            adapted_content = await self._generate_culturally_adapted_content(
                base_intervention, cultural_metaphors, traditional_practices,
                communication_adapted, family_involvement, cultural_profile
            )
            
            # Predict cultural response
            expected_response = self._predict_cultural_response(
                adapted_content, cultural_profile
            )
            
            return CulturalIntervention(
                intervention_id=f"cultural_{int(datetime.now().timestamp())}",
                base_intervention=base_intervention,
                cultural_context=cultural_profile.primary_culture,
                adapted_content=adapted_content,
                cultural_metaphors=cultural_metaphors,
                traditional_practices_integrated=traditional_practices,
                communication_style_adapted=communication_adapted,
                family_involvement_level=family_involvement,
                expected_cultural_response=expected_response
            )
            
        except Exception as e:
            self.logger.error(f"Error creating cultural intervention: {str(e)}")
            return CulturalIntervention(
                intervention_id="error",
                base_intervention=base_intervention,
                cultural_context=cultural_profile.primary_culture,
                adapted_content=base_intervention,
                cultural_metaphors=[],
                traditional_practices_integrated=[],
                communication_style_adapted="standard",
                family_involvement_level="medium",
                expected_cultural_response="unknown"
            )
    
    # Private helper methods
    
    def _update_profile_with_explicit_info(self,
                                         profile: CulturalProfile,
                                         explicit_info: Dict[str, Any]) -> CulturalProfile:
        """Update profile with explicitly provided cultural information"""
        
        if "culture" in explicit_info:
            profile.primary_culture = explicit_info["culture"]
        
        if "language" in explicit_info:
            profile.language_preferences = [explicit_info["language"]]
        
        if "religion" in explicit_info:
            profile.religious_background = explicit_info["religion"]
        
        if "family_structure" in explicit_info:
            if explicit_info["family_structure"] in ["collectivist", "extended_family"]:
                profile.collectivist_vs_individualist = 0.2
                profile.family_involvement_preference = "high"
            elif explicit_info["family_structure"] in ["individualist", "nuclear_family"]:
                profile.collectivist_vs_individualist = 0.8
                profile.family_involvement_preference = "low"
        
        return profile
    
    async def _extract_cultural_indicators(self,
                                         user_message: str,
                                         conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract cultural indicators from conversation"""
        
        indicators = {
            "language_patterns": [],
            "cultural_references": [],
            "value_indicators": [],
            "family_references": [],
            "religious_references": [],
            "traditional_practice_mentions": []
        }
        
        # Language patterns
        indicators["language_patterns"] = self._detect_language_patterns(user_message)
        
        # Cultural references
        cultural_keywords = {
            "asian": ["family honor", "respect elders", "face", "harmony", "community"],
            "hispanic": ["familia", "respeto", "personalismo", "machismo", "marianismo"],
            "african": ["ubuntu", "community", "extended family", "spirituality", "ancestors"],
            "middle_eastern": ["honor", "family", "traditional", "religious", "community"],
            "western": ["individual", "self", "independence", "personal choice", "therapy"]
        }
        
        full_text = user_message + " " + " ".join([h.get("message", "") for h in conversation_history])
        
        for culture, keywords in cultural_keywords.items():
            matches = [kw for kw in keywords if kw.lower() in full_text.lower()]
            if matches:
                indicators["cultural_references"].append({"culture": culture, "matches": matches})
        
        # Family references
        family_words = ["family", "parents", "siblings", "relatives", "clan", "tribe", "ancestors"]
        indicators["family_references"] = [word for word in family_words if word in full_text.lower()]
        
        # Religious references
        religious_words = ["god", "prayer", "faith", "church", "mosque", "temple", "spiritual", "divine"]
        indicators["religious_references"] = [word for word in religious_words if word in full_text.lower()]
        
        return indicators
    
    async def _update_profile_with_indicators(self,
                                            profile: CulturalProfile,
                                            indicators: Dict[str, Any]) -> CulturalProfile:
        """Update cultural profile based on conversation indicators"""
        
        # Update primary culture based on strongest indicators
        cultural_scores = {}
        for ref in indicators["cultural_references"]:
            culture = ref["culture"]
            score = len(ref["matches"])
            cultural_scores[culture] = cultural_scores.get(culture, 0) + score
        
        if cultural_scores and profile.primary_culture == "unknown":
            profile.primary_culture = max(cultural_scores, key=cultural_scores.get)
        
        # Update family involvement based on family references
        if len(indicators["family_references"]) > 3:
            profile.family_involvement_preference = "high"
            profile.collectivist_vs_individualist = max(0.0, profile.collectivist_vs_individualist - 0.2)
        
        # Update religious background
        if indicators["religious_references"] and not profile.religious_background:
            profile.religious_background = "present"
        
        return profile
    
    def _assess_communication_style(self,
                                  user_message: str,
                                  conversation_history: List[Dict[str, Any]]) -> str:
        """Assess user's communication style"""
        
        full_text = user_message + " " + " ".join([h.get("message", "") for h in conversation_history[-5:]])
        
        # Direct vs Indirect indicators
        direct_indicators = ["I think", "I believe", "I want", "I need", "directly", "clearly"]
        indirect_indicators = ["maybe", "perhaps", "might", "could be", "sort of", "kind of"]
        
        direct_count = sum(1 for indicator in direct_indicators if indicator.lower() in full_text.lower())
        indirect_count = sum(1 for indicator in indirect_indicators if indicator.lower() in full_text.lower())
        
        if direct_count > indirect_count * 1.5:
            return "direct"
        elif indirect_count > direct_count * 1.5:
            return "indirect"
        
        # High context vs Low context
        context_indicators = ["you know", "as you can imagine", "obviously", "of course"]
        if any(indicator in full_text.lower() for indicator in context_indicators):
            return "high_context"
        
        return "balanced"
    
    def _assess_mental_health_stigma(self,
                                   user_message: str,
                                   conversation_history: List[Dict[str, Any]],
                                   culture: str) -> float:
        """Assess level of mental health stigma"""
        
        # Base stigma levels by culture (generalized, should be refined)
        base_stigma = {
            "asian": 0.7,
            "hispanic": 0.6,
            "african": 0.6,
            "middle_eastern": 0.7,
            "western": 0.3,
            "unknown": 0.5
        }
        
        stigma_level = base_stigma.get(culture, 0.5)
        
        # Adjust based on conversation content
        full_text = user_message + " " + " ".join([h.get("message", "") for h in conversation_history])
        
        # Stigma-reducing indicators
        positive_indicators = ["therapy", "counseling", "mental health", "seeking help", "professional help"]
        positive_count = sum(1 for indicator in positive_indicators if indicator.lower() in full_text.lower())
        
        # Stigma-increasing indicators
        negative_indicators = ["crazy", "weak", "shameful", "embarrassing", "family shame", "what will people think"]
        negative_count = sum(1 for indicator in negative_indicators if indicator.lower() in full_text.lower())
        
        # Adjust stigma level
        stigma_level -= positive_count * 0.1
        stigma_level += negative_count * 0.1
        
        return max(0.0, min(1.0, stigma_level))
    
    def _assess_cultural_orientation(self,
                                   user_message: str,
                                   conversation_history: List[Dict[str, Any]]) -> float:
        """Assess collectivist vs individualist orientation (0.0 = collectivist, 1.0 = individualist)"""
        
        full_text = user_message + " " + " ".join([h.get("message", "") for h in conversation_history])
        
        collectivist_indicators = [
            "family", "community", "group", "we", "us", "together", "harmony",
            "obligation", "duty", "respect", "honor", "collective"
        ]
        
        individualist_indicators = [
            "I", "me", "myself", "personal", "individual", "independent", "self",
            "choice", "freedom", "personal goals", "self-expression"
        ]
        
        collectivist_count = sum(1 for indicator in collectivist_indicators if indicator.lower() in full_text.lower())
        individualist_count = sum(1 for indicator in individualist_indicators if indicator.lower() in full_text.lower())
        
        total_count = collectivist_count + individualist_count
        if total_count == 0:
            return 0.5  # Default balanced
        
        return individualist_count / total_count
    
    def _identify_cultural_considerations(self,
                                        base_response: str,
                                        cultural_profile: CulturalProfile) -> List[str]:
        """Identify cultural considerations for response adaptation"""
        
        considerations = []
        
        # Stigma considerations
        if cultural_profile.mental_health_stigma_level > 0.6:
            considerations.append("high_mental_health_stigma")
            considerations.append("avoid_pathological_language")
            considerations.append("normalize_seeking_help")
        
        # Communication style considerations
        if cultural_profile.communication_style == "indirect":
            considerations.append("use_indirect_communication")
            considerations.append("avoid_confrontational_language")
        elif cultural_profile.communication_style == "high_context":
            considerations.append("provide_contextual_understanding")
            considerations.append("acknowledge_implicit_meanings")
        
        # Family involvement considerations
        if cultural_profile.collectivist_vs_individualist < 0.4:
            considerations.append("consider_family_impact")
            considerations.append("acknowledge_collective_responsibility")
        
        # Religious considerations
        if cultural_profile.religious_background:
            considerations.append("respect_religious_beliefs")
            considerations.append("integrate_spiritual_perspectives")
        
        # Authority relationship considerations
        if cultural_profile.authority_relationship == "hierarchical":
            considerations.append("maintain_respectful_authority")
            considerations.append("avoid_challenging_directly")
        
        return considerations
    
    async def _adapt_language_for_culture(self,
                                        base_response: str,
                                        cultural_profile: CulturalProfile) -> List[str]:
        """Adapt language for cultural sensitivity"""
        
        adaptations = []
        
        # Stigma-sensitive language adaptations
        if cultural_profile.mental_health_stigma_level > 0.6:
            adaptations.extend([
                "Replace 'mental illness' with 'emotional challenges'",
                "Replace 'disorder' with 'experience'",
                "Replace 'symptoms' with 'feelings' or 'experiences'",
                "Emphasize strength and resilience"
            ])
        
        # Communication style adaptations
        if cultural_profile.communication_style == "indirect":
            adaptations.extend([
                "Use softer language ('might', 'perhaps', 'could')",
                "Avoid direct challenges or confrontations",
                "Frame suggestions as options rather than recommendations"
            ])
        
        # Collectivist adaptations
        if cultural_profile.collectivist_vs_individualist < 0.4:
            adaptations.extend([
                "Acknowledge family and community context",
                "Use 'we' language when appropriate",
                "Consider collective well-being in suggestions"
            ])
        
        return adaptations
    
    def _translate_psychological_concepts(self,
                                        base_response: str,
                                        cultural_profile: CulturalProfile) -> Dict[str, str]:
        """Translate psychological concepts for cultural understanding"""
        
        translations = {}
        culture = cultural_profile.primary_culture
        
        # Culture-specific concept translations
        if culture == "asian":
            translations.update({
                "depression": "feeling heavy-hearted or experiencing life imbalance",
                "anxiety": "worried thoughts or nervous energy",
                "therapy": "talking with someone who can help bring harmony",
                "boundaries": "maintaining respectful distance while honoring relationships"
            })
        elif culture == "hispanic":
            translations.update({
                "depression": "feeling deeply sad or having a heavy spirit",
                "anxiety": "nervios or worried feelings",
                "therapy": "platica therapeutic or healing conversation",
                "self-care": "caring for yourself to better serve others"
            })
        elif culture == "african":
            translations.update({
                "depression": "feeling disconnected from community and spirit",
                "anxiety": "worry that disrupts inner peace",
                "therapy": "healing through conversation and community support",
                "mental health": "emotional and spiritual well-being"
            })
        
        return translations
    
    def _apply_stigma_mitigation(self,
                               base_response: str,
                               cultural_profile: CulturalProfile) -> List[str]:
        """Apply strategies to mitigate mental health stigma"""
        
        mitigation_strategies = []
        
        if cultural_profile.mental_health_stigma_level > 0.6:
            mitigation_strategies.extend([
                "Normalize emotional experiences as part of human condition",
                "Emphasize strength in seeking help",
                "Reference respected cultural figures who sought support",
                "Frame help-seeking as responsibility to family/community",
                "Use culturally-accepted metaphors for emotional healing"
            ])
        
        if cultural_profile.mental_health_stigma_level > 0.8:
            mitigation_strategies.extend([
                "Avoid clinical/medical terminology",
                "Focus on practical coping rather than pathology",
                "Emphasize confidentiality and privacy",
                "Suggest gradual steps rather than formal therapy"
            ])
        
        return mitigation_strategies
    
    async def _integrate_traditional_practices(self,
                                             base_response: str,
                                             therapeutic_approach: str,
                                             cultural_profile: CulturalProfile) -> List[str]:
        """Integrate traditional healing practices"""
        
        integrations = []
        culture = cultural_profile.primary_culture
        
        traditional_practices = self.traditional_practices.get(culture, {})
        
        for practice_name, practice_info in traditional_practices.items():
            if practice_info.get("compatible_with", {}).get(therapeutic_approach, False):
                integrations.append(f"Consider incorporating {practice_name}: {practice_info['description']}")
        
        # Add general cultural healing approaches
        if culture == "asian":
            integrations.extend([
                "Consider mindfulness practices from Buddhist or Taoist traditions",
                "Explore balance concepts from Traditional Chinese Medicine",
                "Honor family wisdom and ancestral guidance"
            ])
        elif culture == "hispanic":
            integrations.extend([
                "Consider familia support and consejos (advice) from elders",
                "Explore spiritual practices like sobadoras or curanderismo",
                "Honor personalismo in therapeutic relationship"
            ])
        elif culture == "african":
            integrations.extend([
                "Consider Ubuntu philosophy and community healing",
                "Explore spiritual practices and ancestor connection",
                "Honor oral tradition and storytelling for healing"
            ])
        
        return integrations
    
    async def _generate_adapted_response(self,
                                       base_response: str,
                                       cultural_considerations: List[str],
                                       language_adaptations: List[str],
                                       concept_translations: Dict[str, str],
                                       stigma_mitigation: List[str],
                                       traditional_integration: List[str],
                                       cultural_profile: CulturalProfile) -> str:
        """Generate culturally adapted response"""
        
        try:
            adaptation_prompt = f"""
            Adapt the following therapeutic response for cultural sensitivity:
            
            Original response: {base_response}
            
            Cultural context: {cultural_profile.primary_culture}
            Communication style: {cultural_profile.communication_style}
            Stigma level: {cultural_profile.mental_health_stigma_level}
            Cultural orientation: {'Collectivist' if cultural_profile.collectivist_vs_individualist < 0.5 else 'Individualist'}
            
            Cultural considerations: {', '.join(cultural_considerations)}
            Language adaptations needed: {', '.join(language_adaptations)}
            Concept translations: {json.dumps(concept_translations)}
            Stigma mitigation strategies: {', '.join(stigma_mitigation)}
            Traditional practice integration: {', '.join(traditional_integration)}
            
            Please provide a culturally adapted version that:
            1. Respects the cultural background and values
            2. Uses appropriate communication style
            3. Minimizes stigma while maintaining therapeutic value
            4. Integrates relevant traditional concepts where appropriate
            5. Maintains the core therapeutic message
            
            Adapted response:
            """
            
            adapted_response = await self.llm.generate_response(adaptation_prompt)
            return adapted_response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating adapted response: {str(e)}")
            return base_response  # Fallback to original
    
    def _generate_adaptation_reasoning(self, cultural_considerations: List[str]) -> str:
        """Generate reasoning for cultural adaptation"""
        
        if not cultural_considerations:
            return "No specific cultural adaptations needed"
        
        reasoning_parts = []
        
        if "high_mental_health_stigma" in cultural_considerations:
            reasoning_parts.append("Adapted language to reduce mental health stigma")
        
        if "use_indirect_communication" in cultural_considerations:
            reasoning_parts.append("Modified communication style to be more indirect and respectful")
        
        if "consider_family_impact" in cultural_considerations:
            reasoning_parts.append("Incorporated family and community context")
        
        if "respect_religious_beliefs" in cultural_considerations:
            reasoning_parts.append("Integrated spiritual and religious perspectives")
        
        return "; ".join(reasoning_parts)
    
    async def _generate_cultural_metaphors(self,
                                         base_intervention: str,
                                         cultural_profile: CulturalProfile,
                                         specific_issue: str = None) -> List[str]:
        """Generate culturally relevant metaphors"""
        
        culture = cultural_profile.primary_culture
        metaphors = []
        
        # Culture-specific metaphors
        if culture == "asian":
            metaphors.extend([
                "Like a river finding its way around obstacles",
                "Balancing yin and yang energies within yourself",
                "Tending to your inner garden with patience"
            ])
        elif culture == "hispanic":
            metaphors.extend([
                "Building bridges between your heart and mind",
                "Nurturing your spirit like a growing plant",
                "Finding your camino (path) through life's challenges"
            ])
        elif culture == "african":
            metaphors.extend([
                "Drawing strength from your roots while reaching toward the sky",
                "Weaving your story into the larger tapestry of community",
                "Finding the rhythm that brings harmony to your life"
            ])
        
        return metaphors[:3]  # Limit to top 3
    
    def _select_traditional_practices(self,
                                    base_intervention: str,
                                    cultural_profile: CulturalProfile) -> List[str]:
        """Select relevant traditional healing practices"""
        
        culture = cultural_profile.primary_culture
        practices = []
        
        # Select practices based on intervention type and culture
        if "mindfulness" in base_intervention.lower():
            if culture == "asian":
                practices.append("Buddhist mindfulness meditation")
            elif culture == "hispanic":
                practices.append("Meditación or contemplative prayer")
        
        if "community" in base_intervention.lower() or "support" in base_intervention.lower():
            if culture == "african":
                practices.append("Ubuntu community circles")
            elif culture == "hispanic":
                practices.append("Familia gatherings and consejos")
        
        return practices
    
    def _adapt_communication_style(self,
                                 base_intervention: str,
                                 communication_style: str) -> str:
        """Adapt intervention for communication style"""
        
        if communication_style == "indirect":
            return "Use gentle suggestions and questions rather than direct instructions"
        elif communication_style == "high_context":
            return "Provide rich context and acknowledge unspoken understanding"
        elif communication_style == "direct":
            return "Use clear, straightforward communication"
        
        return "Use balanced communication approach"
    
    def _determine_family_involvement(self,
                                    base_intervention: str,
                                    cultural_profile: CulturalProfile) -> str:
        """Determine appropriate level of family involvement"""
        
        if cultural_profile.collectivist_vs_individualist < 0.3:
            return "high"  # Strong collectivist orientation
        elif cultural_profile.collectivist_vs_individualist > 0.7:
            return "low"   # Strong individualist orientation
        else:
            return "medium"  # Balanced approach
    
    async def _generate_culturally_adapted_content(self,
                                                 base_intervention: str,
                                                 cultural_metaphors: List[str],
                                                 traditional_practices: List[str],
                                                 communication_adapted: str,
                                                 family_involvement: str,
                                                 cultural_profile: CulturalProfile) -> str:
        """Generate culturally adapted intervention content"""
        
        try:
            prompt = f"""
            Create a culturally adapted therapeutic intervention based on:
            
            Base intervention: {base_intervention}
            Cultural context: {cultural_profile.primary_culture}
            Cultural metaphors to use: {', '.join(cultural_metaphors)}
            Traditional practices to integrate: {', '.join(traditional_practices)}
            Communication style: {communication_adapted}
            Family involvement level: {family_involvement}
            
            Create an intervention that respectfully integrates these cultural elements
            while maintaining therapeutic effectiveness.
            """
            
            adapted_content = await self.llm.generate_response(prompt)
            return adapted_content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating adapted content: {str(e)}")
            return base_intervention
    
    def _predict_cultural_response(self,
                                 adapted_content: str,
                                 cultural_profile: CulturalProfile) -> str:
        """Predict likely cultural response to adapted intervention"""
        
        if cultural_profile.mental_health_stigma_level > 0.7:
            return "May initially be hesitant due to stigma, but culturally sensitive approach should increase acceptance"
        elif cultural_profile.collectivist_vs_individualist < 0.3:
            return "Likely to appreciate family/community context and collective approach"
        elif cultural_profile.communication_style == "indirect":
            return "Should feel comfortable with gentle, non-confrontational approach"
        else:
            return "Expected to be receptive to culturally adapted intervention"
    
    def _create_default_profile(self, user_id: str) -> CulturalProfile:
        """Create default cultural profile"""
        return CulturalProfile(
            user_id=user_id,
            primary_culture="unknown",
            cultural_subgroups=[],
            language_preferences=["english"],
            religious_background=None,
            traditional_healing_practices=[],
            mental_health_stigma_level=0.5,
            collectivist_vs_individualist=0.5,
            communication_style="balanced",
            family_involvement_preference="medium",
            authority_relationship="egalitarian",
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    def _detect_language_patterns(self, text: str) -> List[str]:
        """Detect language patterns that indicate cultural background"""
        patterns = []
        
        # Add patterns for different languages/dialects
        # This is a simplified implementation
        if re.search(r'[你我他她它们]', text):
            patterns.append("chinese_characters")
        
        # Add more sophisticated language detection
        
        return patterns
    
    async def _store_cultural_adaptation(self, adaptation: CulturalAdaptation):
        """Store cultural adaptation in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "cultural_adaptation",
                "culture_context": adaptation.culture_context,
                "cultural_considerations": adaptation.cultural_considerations,
                "adaptation_reasoning": adaptation.reasoning,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=adaptation.adaptation_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing cultural adaptation: {str(e)}")
    
    def _load_cultural_knowledge(self) -> Dict[str, Any]:
        """Load cultural knowledge database"""
        return {
            "asian": {
                "values": ["harmony", "respect", "family honor", "education", "collective good"],
                "communication": "indirect, high-context, respectful",
                "mental_health_views": "often stigmatized, seen as weakness or shame",
                "traditional_healing": ["acupuncture", "herbal medicine", "qi gong", "meditation"]
            },
            "hispanic": {
                "values": ["familia", "respeto", "personalismo", "spirituality", "community"],
                "communication": "warm, personal, relationship-focused",
                "mental_health_views": "may be stigmatized, spiritual causes considered",
                "traditional_healing": ["curanderismo", "sobadoras", "religious practices", "herbal remedies"]
            },
            "african": {
                "values": ["ubuntu", "community", "spirituality", "oral tradition", "extended family"],
                "communication": "storytelling, community-oriented, respectful of elders",
                "mental_health_views": "often spiritual/community issue rather than individual",
                "traditional_healing": ["spiritual practices", "community rituals", "herbal medicine", "storytelling"]
            }
        }
    
    def _load_stigma_patterns(self) -> Dict[str, Any]:
        """Load mental health stigma patterns by culture"""
        return {
            "high_stigma_cultures": ["asian", "middle_eastern", "some_african"],
            "medium_stigma_cultures": ["hispanic", "eastern_european"],
            "lower_stigma_cultures": ["western", "scandinavian"],
            "mitigation_strategies": {
                "normalize": "Frame as normal human experience",
                "strength": "Emphasize strength in seeking help",
                "privacy": "Assure confidentiality and privacy",
                "gradual": "Suggest gradual approach to formal help"
            }
        }
    
    def _load_communication_styles(self) -> Dict[str, Any]:
        """Load communication style patterns"""
        return {
            "direct": {
                "cultures": ["german", "dutch", "scandinavian"],
                "characteristics": ["straightforward", "explicit", "task-focused"]
            },
            "indirect": {
                "cultures": ["asian", "middle_eastern", "some_african"],
                "characteristics": ["implicit", "context-dependent", "relationship-preserving"]
            },
            "high_context": {
                "cultures": ["japanese", "arab", "latin_american"],
                "characteristics": ["implicit meaning", "relationship-focused", "contextual understanding"]
            }
        }
    
    def _load_traditional_practices(self) -> Dict[str, Any]:
        """Load traditional healing practices database"""
        return {
            "asian": {
                "meditation": {
                    "description": "Mindfulness and breathing practices",
                    "compatible_with": {"CBT": True, "mindfulness_therapy": True}
                },
                "acupuncture": {
                    "description": "Energy balancing through needle therapy",
                    "compatible_with": {"holistic_therapy": True}
                }
            },
            "hispanic": {
                "curanderismo": {
                    "description": "Traditional healing combining spiritual and herbal practices",
                    "compatible_with": {"spiritual_therapy": True, "holistic_therapy": True}
                },
                "consejos": {
                    "description": "Elder wisdom and family advice",
                    "compatible_with": {"family_therapy": True, "narrative_therapy": True}
                }
            },
            "african": {
                "ubuntu_circles": {
                    "description": "Community-based healing and support",
                    "compatible_with": {"group_therapy": True, "community_therapy": True}
                },
                "storytelling": {
                    "description": "Healing through narrative and oral tradition",
                    "compatible_with": {"narrative_therapy": True, "expressive_therapy": True}
                }
            }
        }