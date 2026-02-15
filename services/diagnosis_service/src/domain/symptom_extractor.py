"""
Solace-AI Diagnosis Service - Symptom Extraction from Conversation.
Extracts symptoms, temporal information, and contextual factors from user messages.
"""
from __future__ import annotations
import json
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import SymptomDTO, SymptomType, SeverityLevel

if TYPE_CHECKING:
    from services.shared.infrastructure.llm_client import UnifiedLLMClient

logger = structlog.get_logger(__name__)

SYMPTOM_EXTRACTION_PROMPT = (
    "You are a clinical symptom extractor. Analyze the user's message and extract "
    "mental health symptoms. For each symptom found, provide:\n"
    "- name: a snake_case identifier (e.g. depressed_mood, anxiety, sleep_disturbance)\n"
    "- type: one of EMOTIONAL, COGNITIVE, SOMATIC, BEHAVIORAL\n"
    "- severity: one of MINIMAL, MILD, MODERATE, MODERATELY_SEVERE, SEVERE\n"
    "- confidence: 0.0-1.0\n"
    "- evidence: the phrase from the text that indicates this symptom\n\n"
    "Respond with ONLY valid JSON: {\"symptoms\": [{\"name\": \"...\", \"type\": \"...\", "
    "\"severity\": \"...\", \"confidence\": 0.0, \"evidence\": \"...\"}]}"
)

_SYMPTOM_TYPE_MAP = {v.value: v for v in SymptomType}
_SEVERITY_MAP = {v.value: v for v in SeverityLevel}


class SymptomExtractorSettings(BaseSettings):
    """Symptom extractor configuration."""
    min_symptom_confidence: float = Field(default=0.5)
    max_symptoms_per_message: int = Field(default=10)
    enable_temporal_extraction: bool = Field(default=True)
    enable_severity_detection: bool = Field(default=True)
    therapeutic_keywords_boost: float = Field(default=0.15)
    model_config = SettingsConfigDict(env_prefix="SYMPTOM_EXTRACTOR_", env_file=".env", extra="ignore")


@dataclass
class ExtractionResult:
    """Result from symptom extraction."""
    symptoms: list[SymptomDTO] = field(default_factory=list)
    temporal_info: dict[str, str] = field(default_factory=dict)
    contextual_factors: list[str] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)


class SymptomExtractor:
    """Extracts symptoms from conversation using pattern matching and NLP."""

    def __init__(
        self,
        settings: SymptomExtractorSettings | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self._settings = settings or SymptomExtractorSettings()
        self._llm_client = llm_client
        self._symptom_patterns = self._build_symptom_patterns()
        self._temporal_patterns = self._build_temporal_patterns()
        self._severity_indicators = self._build_severity_indicators()
        self._risk_patterns = self._build_risk_patterns()
        self._stats = {"extractions": 0, "symptoms_found": 0, "risks_detected": 0}

    def _build_symptom_patterns(self) -> dict[str, dict[str, Any]]:
        """Build symptom detection patterns."""
        return {
            "depressed_mood": {
                "patterns": [r"\b(sad|depressed|down|hopeless|empty|worthless)\b",
                            r"\b(feeling\s+low|feeling\s+blue|can't\s+feel)\b"],
                "type": SymptomType.EMOTIONAL,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "mood_disorders",
            },
            "anhedonia": {
                "patterns": [r"\b(no\s+interest|lost\s+interest|don't\s+enjoy|nothing\s+fun)\b",
                            r"\b(can't\s+enjoy|pleasure|motivation)\b"],
                "type": SymptomType.EMOTIONAL,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "mood_disorders",
            },
            "sleep_disturbance": {
                "patterns": [r"\b(can't\s+sleep|insomnia|sleeping\s+too\s+much|oversleeping)\b",
                            r"\b(wake\s+up|trouble\s+sleeping|nightmares)\b"],
                "type": SymptomType.SOMATIC,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "sleep_disorders",
            },
            "fatigue": {
                "patterns": [r"\b(tired|exhausted|no\s+energy|fatigue|drained)\b",
                            r"\b(can't\s+get\s+up|low\s+energy)\b"],
                "type": SymptomType.SOMATIC,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "general_symptoms",
            },
            "appetite_change": {
                "patterns": [r"\b(not\s+eating|eating\s+too\s+much|no\s+appetite|overeating)\b",
                            r"\b(weight\s+gain|weight\s+loss|food)\b"],
                "type": SymptomType.SOMATIC,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "eating_disorders",
            },
            "concentration_difficulty": {
                "patterns": [r"\b(can't\s+concentrate|focus|distracted|forgetful)\b",
                            r"\b(brain\s+fog|hard\s+to\s+think|memory)\b"],
                "type": SymptomType.COGNITIVE,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "cognitive_symptoms",
            },
            "anxiety": {
                "patterns": [r"\b(anxious|worried|nervous|panic|fear|scared)\b",
                            r"\b(on\s+edge|restless|tense)\b"],
                "type": SymptomType.EMOTIONAL,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "anxiety_disorders",
            },
            "social_withdrawal": {
                "patterns": [r"\b(isolat|alone|avoid\s+people|don't\s+want\s+to\s+see)\b",
                            r"\b(staying\s+home|hiding|withdrawn)\b"],
                "type": SymptomType.BEHAVIORAL,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "social_symptoms",
            },
            "irritability": {
                "patterns": [r"\b(irritable|angry|frustrated|annoyed|snapping)\b",
                            r"\b(short\s+temper|rage|outbursts)\b"],
                "type": SymptomType.EMOTIONAL,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "mood_disorders",
            },
            "guilt": {
                "patterns": [r"\b(guilt|blame\s+myself|my\s+fault|shame)\b",
                            r"\b(feel\s+bad|regret)\b"],
                "type": SymptomType.COGNITIVE,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "mood_disorders",
            },
            "physical_tension": {
                "patterns": [r"\b(headache|muscle\s+tension|chest\s+tight|heart\s+racing)\b",
                            r"\b(sweating|shaking|trembling)\b"],
                "type": SymptomType.SOMATIC,
                "base_severity": SeverityLevel.MILD,
                "dsm_category": "anxiety_disorders",
            },
            "intrusive_thoughts": {
                "patterns": [r"\b(can't\s+stop\s+thinking|thoughts\s+won't\s+stop|obsess)\b",
                            r"\b(ruminating|overthinking|intrusive)\b"],
                "type": SymptomType.COGNITIVE,
                "base_severity": SeverityLevel.MODERATE,
                "dsm_category": "anxiety_disorders",
            },
        }

    def _build_temporal_patterns(self) -> list[tuple[str, str]]:
        """Build temporal extraction patterns."""
        return [
            (r"for\s+(\d+)\s+(day|week|month|year)s?", "duration"),
            (r"since\s+(\w+)", "onset"),
            (r"started\s+(\w+\s+\w+|\w+)", "onset"),
            (r"(always|sometimes|often|rarely|never)", "frequency"),
            (r"(morning|evening|night|afternoon)", "time_of_day"),
            (r"(every\s+day|daily|weekly|constantly)", "frequency"),
            (r"(getting\s+worse|getting\s+better|same)", "progression"),
            (r"(suddenly|gradually|slowly)", "onset_type"),
        ]

    def _build_severity_indicators(self) -> dict[str, SeverityLevel]:
        """Build severity indicator mapping."""
        return {
            "extremely": SeverityLevel.SEVERE,
            "severely": SeverityLevel.SEVERE,
            "very": SeverityLevel.MODERATELY_SEVERE,
            "really": SeverityLevel.MODERATELY_SEVERE,
            "quite": SeverityLevel.MODERATE,
            "somewhat": SeverityLevel.MILD,
            "a bit": SeverityLevel.MILD,
            "slightly": SeverityLevel.MINIMAL,
            "a little": SeverityLevel.MINIMAL,
            "constant": SeverityLevel.SEVERE,
            "unbearable": SeverityLevel.SEVERE,
            "overwhelming": SeverityLevel.SEVERE,
            "can't function": SeverityLevel.SEVERE,
            "all the time": SeverityLevel.MODERATELY_SEVERE,
        }

    def _build_risk_patterns(self) -> list[tuple[str, str]]:
        """Build risk indicator patterns."""
        return [
            (r"\b(suicid|kill\s+myself|end\s+it|don't\s+want\s+to\s+live)\b", "suicidal_ideation"),
            (r"\b(hurt\w*\s+myself|self.?harm|cutting|burning)\b", "self_harm"),
            (r"\b(hurt\s+someone|harm\s+others|violent)\b", "harm_to_others"),
            (r"\b(can't\s+go\s+on|no\s+point|give\s+up)\b", "hopelessness"),
            (r"\b(psychotic|hallucin|voices|seeing\s+things)\b", "psychotic_features"),
            (r"\b(abuse\w*|trauma\w*|assault\w*)\b", "trauma_disclosure"),
        ]

    async def extract(self, message: str, conversation_history: list[dict[str, str]],
                      existing_symptoms: list[SymptomDTO]) -> ExtractionResult:
        """Extract symptoms from message and conversation history."""
        start_time = time.perf_counter()
        self._stats["extractions"] += 1
        result = ExtractionResult()
        combined_text = self._build_context(message, conversation_history)
        result.symptoms = self._extract_symptoms(message, existing_symptoms)

        # Enhance with LLM when regex finds few symptoms
        if (
            len(result.symptoms) < 2
            and self._llm_client is not None
            and self._llm_client.is_available
        ):
            llm_symptoms = await self._llm_extract_symptoms(message, existing_symptoms)
            if llm_symptoms:
                result.symptoms = self.merge_symptoms(result.symptoms, llm_symptoms)

        if self._settings.enable_temporal_extraction:
            result.temporal_info = self._extract_temporal_info(combined_text)
        result.contextual_factors = self._extract_contextual_factors(combined_text)
        result.risk_indicators = self._detect_risk_indicators(message)
        self._stats["symptoms_found"] += len(result.symptoms)
        self._stats["risks_detected"] += len(result.risk_indicators)
        logger.debug("extraction_completed", symptoms=len(result.symptoms),
                    risks=len(result.risk_indicators),
                    time_ms=int((time.perf_counter() - start_time) * 1000))
        return result

    async def _llm_extract_symptoms(
        self, message: str, existing_symptoms: list[SymptomDTO],
    ) -> list[SymptomDTO]:
        """Use LLM to extract symptoms when regex finds few."""
        try:
            response = await self._llm_client.generate(
                system_prompt=SYMPTOM_EXTRACTION_PROMPT,
                user_message=message,
                service_name="diagnosis_symptom_extractor",
                task_type="diagnosis",
                max_tokens=500,
            )
            if not response:
                return []
            parsed = json.loads(response.strip())
            existing_names = {s.name for s in existing_symptoms}
            symptoms: list[SymptomDTO] = []
            for item in parsed.get("symptoms", []):
                name = item.get("name", "")
                if not name or name in existing_names:
                    continue
                sym_type = _SYMPTOM_TYPE_MAP.get(item.get("type", ""), SymptomType.EMOTIONAL)
                severity = _SEVERITY_MAP.get(item.get("severity", ""), SeverityLevel.MODERATE)
                confidence = float(item.get("confidence", 0.6))
                if confidence < self._settings.min_symptom_confidence:
                    continue
                symptoms.append(SymptomDTO(
                    symptom_id=uuid4(),
                    name=name,
                    description=item.get("evidence", f"LLM-extracted: {name}"),
                    symptom_type=sym_type,
                    severity=severity,
                    extracted_from=message[:200],
                    confidence=Decimal(str(round(confidence, 2))),
                ))
            logger.info("llm_symptom_extraction", count=len(symptoms))
            return symptoms
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("llm_symptom_parse_failed", error=str(e))
            return []
        except Exception as e:
            logger.warning("llm_symptom_extraction_failed", error=str(e))
            return []

    def _build_context(self, message: str, history: list[dict[str, str]]) -> str:
        """Build context from message and history."""
        parts = [message]
        for entry in history[-5:]:
            if entry.get("role") == "user":
                parts.append(entry.get("content", ""))
        return " ".join(parts)

    def _extract_symptoms(self, text: str, existing: list[SymptomDTO]) -> list[SymptomDTO]:
        """Extract symptoms from text."""
        text_lower = text.lower()
        symptoms: list[SymptomDTO] = []
        existing_names = {s.name for s in existing}
        for symptom_name, config in self._symptom_patterns.items():
            if symptom_name in existing_names:
                continue
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    severity = self._detect_severity(text_lower, config["base_severity"])
                    confidence = self._calculate_confidence(text_lower, pattern)
                    if confidence >= self._settings.min_symptom_confidence:
                        symptoms.append(SymptomDTO(
                            symptom_id=uuid4(),
                            name=symptom_name,
                            description=f"Detected from: {text[:100]}...",
                            symptom_type=config["type"],
                            severity=severity,
                            extracted_from=text[:200],
                            confidence=Decimal(str(round(confidence, 2))),
                        ))
                    break
            if len(symptoms) >= self._settings.max_symptoms_per_message:
                break
        return symptoms

    def _detect_severity(self, text: str, base_severity: SeverityLevel) -> SeverityLevel:
        """Detect severity from text modifiers."""
        if not self._settings.enable_severity_detection:
            return base_severity
        for indicator, severity in self._severity_indicators.items():
            if indicator in text:
                return severity
        return base_severity

    def _calculate_confidence(self, text: str, pattern: str) -> float:
        """Calculate confidence score for extraction."""
        base_confidence = 0.6
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        confidence = base_confidence + (0.1 * min(matches, 3))
        therapeutic_keywords = {"therapy", "treatment", "help", "support", "cope", "feeling"}
        text_words = set(text.split())
        if text_words & therapeutic_keywords:
            confidence += self._settings.therapeutic_keywords_boost
        return min(confidence, 1.0)

    def _extract_temporal_info(self, text: str) -> dict[str, str]:
        """Extract temporal information from text."""
        temporal: dict[str, str] = {}
        for pattern, info_type in self._temporal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                temporal[info_type] = match.group(0)
        return temporal

    def _extract_contextual_factors(self, text: str) -> list[str]:
        """Extract contextual factors from text."""
        factors: list[str] = []
        contextual_patterns = [
            (r"\b(work|job|career|boss)\b", "work_related"),
            (r"\b(relationship|partner|spouse|marriage|divorce)\b", "relationship"),
            (r"\b(family|parent|child|sibling)\b", "family_related"),
            (r"\b(school|college|university|exam)\b", "academic"),
            (r"\b(money|financial|debt|bills)\b", "financial"),
            (r"\b(health|illness|pain|chronic)\b", "health_related"),
            (r"\b(loss|lost|grief|death|died)\b", "loss_grief"),
            (r"\b(stress|pressure|overwhelm)\b", "stress"),
        ]
        for pattern, factor in contextual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                factors.append(factor)
        return factors

    def _detect_risk_indicators(self, text: str) -> list[str]:
        """Detect risk indicators in text."""
        risks: list[str] = []
        for pattern, risk_type in self._risk_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                risks.append(risk_type)
                logger.warning("risk_indicator_detected", risk_type=risk_type)
        return risks

    def merge_symptoms(self, existing: list[SymptomDTO], new_symptoms: list[SymptomDTO]) -> list[SymptomDTO]:
        """Merge new symptoms with existing, updating severity if needed."""
        merged: dict[str, SymptomDTO] = {s.name: s for s in existing}
        severity_order = [SeverityLevel.MINIMAL, SeverityLevel.MILD, SeverityLevel.MODERATE,
                         SeverityLevel.MODERATELY_SEVERE, SeverityLevel.SEVERE]
        for symptom in new_symptoms:
            if symptom.name in merged:
                existing_idx = severity_order.index(merged[symptom.name].severity)
                new_idx = severity_order.index(symptom.severity)
                if new_idx > existing_idx:
                    merged[symptom.name] = symptom
            else:
                merged[symptom.name] = symptom
        return list(merged.values())

    def get_symptom_categories(self, symptoms: list[SymptomDTO]) -> dict[str, list[SymptomDTO]]:
        """Categorize symptoms by type."""
        categories: dict[str, list[SymptomDTO]] = {}
        for symptom in symptoms:
            category = symptom.symptom_type.value
            if category not in categories:
                categories[category] = []
            categories[category].append(symptom)
        return categories

    def calculate_symptom_burden(self, symptoms: list[SymptomDTO]) -> dict[str, Any]:
        """Calculate overall symptom burden score."""
        severity_weights = {
            SeverityLevel.MINIMAL: 0.2,
            SeverityLevel.MILD: 0.4,
            SeverityLevel.MODERATE: 0.6,
            SeverityLevel.MODERATELY_SEVERE: 0.8,
            SeverityLevel.SEVERE: 1.0,
        }
        total_weight = sum(severity_weights[s.severity] for s in symptoms)
        avg_severity = total_weight / len(symptoms) if symptoms else 0
        return {
            "symptom_count": len(symptoms),
            "total_burden": round(total_weight, 2),
            "average_severity": round(avg_severity, 2),
            "categories": len(self.get_symptom_categories(symptoms)),
        }

    def get_statistics(self) -> dict[str, int]:
        """Get extraction statistics."""
        return self._stats.copy()
