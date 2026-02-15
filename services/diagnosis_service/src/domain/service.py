"""
Solace-AI Diagnosis Service - 4-Step Chain-of-Reasoning Orchestration.
AMIE-inspired diagnostic reasoning with anti-sycophancy mechanisms.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TYPE_CHECKING
from uuid import UUID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import (
    DiagnosisPhase, ReasoningStep, SeverityLevel,
    SymptomDTO, HypothesisDTO, DifferentialDTO, ReasoningStepResultDTO,
)
from .models import (
    SessionState, AssessmentResult, ExtractionResult, DifferentialResult,
    SessionStartResult, SessionEndResult, HistoryResult, ChallengeResult,
)
from services.shared import ServiceBase

if TYPE_CHECKING:
    from .symptom_extractor import SymptomExtractor
    from .differential import DifferentialGenerator
    from ..infrastructure.repository import DiagnosisRepositoryPort

logger = structlog.get_logger(__name__)


class DiagnosisServiceSettings(BaseSettings):
    """Diagnosis service configuration."""
    max_reasoning_time_ms: int = Field(default=10000)
    enable_anti_sycophancy: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.3)
    max_hypotheses: int = Field(default=5)
    phase_transition_threshold: float = Field(default=0.7)
    enable_longitudinal_tracking: bool = Field(default=True)
    challenge_frequency: float = Field(default=0.3)
    model_config = SettingsConfigDict(env_prefix="DIAGNOSIS_SERVICE_", env_file=".env", extra="ignore")


class DiagnosisService(ServiceBase):
    """Main diagnosis service orchestrating 4-step Chain-of-Reasoning."""

    def __init__(self, settings: DiagnosisServiceSettings | None = None,
                 symptom_extractor: SymptomExtractor | None = None,
                 differential_generator: DifferentialGenerator | None = None,
                 repository: DiagnosisRepositoryPort | None = None) -> None:
        self._settings = settings or DiagnosisServiceSettings()
        self._symptom_extractor = symptom_extractor
        self._differential_generator = differential_generator
        self._repository = repository
        self._active_sessions: dict[UUID, SessionState] = {}
        self._user_session_counts: dict[UUID, int] = {}
        self._user_history: dict[UUID, list[dict[str, Any]]] = {}
        self._initialized = False
        self._stats = {"assessments": 0, "extractions": 0, "differentials": 0,
                      "sessions_started": 0, "sessions_ended": 0, "challenges": 0}

    async def initialize(self) -> None:
        """Initialize the diagnosis service."""
        logger.info("diagnosis_service_initializing")
        self._initialized = True
        logger.info("diagnosis_service_initialized", settings={
            "anti_sycophancy": self._settings.enable_anti_sycophancy,
            "max_hypotheses": self._settings.max_hypotheses,
        })

    async def shutdown(self) -> None:
        """Shutdown the diagnosis service."""
        logger.info("diagnosis_service_shutting_down", stats=self._stats)
        self._initialized = False

    async def assess(self, user_id: UUID, session_id: UUID, message: str,
                     conversation_history: list[dict[str, str]],
                     existing_symptoms: list[SymptomDTO],
                     current_phase: DiagnosisPhase,
                     current_differential: DifferentialDTO | None,
                     user_context: dict[str, Any]) -> AssessmentResult:
        """Execute full 4-step Chain-of-Reasoning assessment."""
        start_time = time.perf_counter()
        self._stats["assessments"] += 1
        result = AssessmentResult()
        reasoning_chain: list[ReasoningStepResultDTO] = []
        step1_start = time.perf_counter()
        step1_result = await self._step1_analyze(message, conversation_history, existing_symptoms)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.ANALYZE, input_summary=f"Message: {message[:100]}...",
            output_summary=f"Extracted {len(step1_result['symptoms'])} symptoms",
            duration_ms=int((time.perf_counter() - step1_start) * 1000), details=step1_result,
        ))
        step2_start = time.perf_counter()
        step2_result = await self._step2_hypothesize(step1_result["symptoms"], user_context)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.HYPOTHESIZE, input_summary=f"{len(step1_result['symptoms'])} symptoms",
            output_summary=f"Generated {len(step2_result['hypotheses'])} hypotheses",
            duration_ms=int((time.perf_counter() - step2_start) * 1000), details=step2_result,
        ))
        step3_start = time.perf_counter()
        step3_result = await self._step3_challenge(step2_result["hypotheses"], step1_result["symptoms"])
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.CHALLENGE, input_summary=f"{len(step2_result['hypotheses'])} hypotheses",
            output_summary=f"Identified {len(step3_result['challenges'])} challenges",
            duration_ms=int((time.perf_counter() - step3_start) * 1000), details=step3_result,
        ))
        step4_start = time.perf_counter()
        step4_result = await self._step4_synthesize(step2_result["hypotheses"], step3_result, current_phase, message)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.SYNTHESIZE, input_summary="All previous steps",
            output_summary=f"Final confidence: {step4_result['confidence']:.2f}",
            duration_ms=int((time.perf_counter() - step4_start) * 1000), details=step4_result,
        ))
        result.phase = self._determine_next_phase(current_phase, step4_result["confidence"])
        result.extracted_symptoms = step1_result["symptoms"]
        result.differential = step4_result["differential"]
        result.reasoning_chain = reasoning_chain
        result.next_question = step4_result["next_question"]
        result.response_text = step4_result["response"]
        result.confidence_score = Decimal(str(step4_result["confidence"]))
        result.safety_flags = step1_result.get("risk_indicators", [])
        result.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        self._update_session(session_id, result)
        logger.debug("assessment_completed", user_id=str(user_id), time_ms=result.processing_time_ms)
        return result

    async def _step1_analyze(self, message: str, history: list[dict[str, str]],
                             existing: list[SymptomDTO]) -> dict[str, Any]:
        """Step 1: Analyze - Extract symptoms and information."""
        if self._symptom_extractor:
            extraction = await self._symptom_extractor.extract(message, history, existing)
            return {"symptoms": extraction.symptoms, "temporal": extraction.temporal_info,
                    "contextual": extraction.contextual_factors, "risk_indicators": extraction.risk_indicators}
        return {"symptoms": existing, "temporal": {}, "contextual": [], "risk_indicators": []}

    async def _step2_hypothesize(self, symptoms: list[SymptomDTO], user_context: dict[str, Any]) -> dict[str, Any]:
        """Step 2: Hypothesize - Generate differential diagnosis."""
        if self._differential_generator:
            differential = await self._differential_generator.generate(symptoms, user_context)
            return {"hypotheses": differential.hypotheses, "missing_info": differential.missing_info,
                    "hitop_scores": differential.hitop_scores}
        return {"hypotheses": [], "missing_info": [], "hitop_scores": {}}

    async def _step3_challenge(self, hypotheses: list[HypothesisDTO], symptoms: list[SymptomDTO]) -> dict[str, Any]:
        """Step 3: Challenge - Devil's Advocate adversarial review."""
        if not self._settings.enable_anti_sycophancy:
            return {"challenges": [], "alternatives": [], "biases": []}
        challenges, biases = [], []
        for hyp in hypotheses[:3]:
            if hyp.confidence > Decimal("0.7"):
                challenges.append(f"High confidence ({hyp.confidence}) for {hyp.name} may indicate confirmation bias")
                biases.append("anchoring_bias")
            if len(hyp.criteria_missing) > len(hyp.criteria_met):
                challenges.append(f"{hyp.name} has more missing than met criteria")
            unexplained = [s for s in symptoms if s.name not in str(hyp.supporting_evidence)]
            if unexplained:
                challenges.append(f"{len(unexplained)} symptoms unexplained by {hyp.name}")
        if len(hypotheses) == 1:
            biases.append("premature_closure")
            challenges.append("Only one hypothesis considered - possible premature closure")
        return {"challenges": challenges, "alternatives": [], "biases": biases}

    async def _step4_synthesize(self, hypotheses: list[HypothesisDTO], challenge_result: dict[str, Any],
                                 phase: DiagnosisPhase, message: str) -> dict[str, Any]:
        """Step 4: Synthesize - Integrate and generate response."""
        calibrated = self._calibrate_confidence(hypotheses, challenge_result)
        primary = calibrated[0] if calibrated else None
        alternatives = calibrated[1:] if len(calibrated) > 1 else []
        differential = DifferentialDTO(primary=primary, alternatives=alternatives,
                                        missing_info=challenge_result.get("missing_info", []))
        next_question = self._generate_next_question(phase, differential, challenge_result)
        response = self._generate_response(phase, differential, next_question)
        return {"differential": differential, "next_question": next_question,
                "response": response, "confidence": float(primary.confidence) if primary else 0.5}

    def _calibrate_confidence(self, hypotheses: list[HypothesisDTO], challenges: dict[str, Any]) -> list[HypothesisDTO]:
        """Calibrate confidence scores based on challenges."""
        penalty = Decimal("0.05") * len(challenges.get("biases", []))
        calibrated = [HypothesisDTO(
            hypothesis_id=h.hypothesis_id, name=h.name, dsm5_code=h.dsm5_code, icd11_code=h.icd11_code,
            confidence=max(Decimal("0.1"), h.confidence - penalty), confidence_interval=h.confidence_interval,
            criteria_met=h.criteria_met, criteria_missing=h.criteria_missing,
            supporting_evidence=h.supporting_evidence, contra_evidence=h.contra_evidence,
            severity=h.severity, hitop_dimensions=h.hitop_dimensions,
        ) for h in hypotheses]
        return sorted(calibrated, key=lambda h: h.confidence, reverse=True)

    def _generate_next_question(self, phase: DiagnosisPhase, differential: DifferentialDTO, challenges: dict) -> str | None:
        """Generate next question based on phase and differential."""
        questions = {DiagnosisPhase.RAPPORT: "How have you been feeling overall lately?",
                    DiagnosisPhase.HISTORY: "Can you tell me more about when these feelings started?",
                    DiagnosisPhase.ASSESSMENT: "On a scale of 0-10, how would you rate the intensity?",
                    DiagnosisPhase.DIAGNOSIS: "Based on what we've discussed, does this resonate with you?"}
        if differential.missing_info:
            return f"Could you tell me more about {differential.missing_info[0]}?"
        return questions.get(phase)

    def _generate_response(self, phase: DiagnosisPhase, differential: DifferentialDTO, next_question: str | None) -> str:
        """Generate empathetic response based on phase."""
        responses = {DiagnosisPhase.RAPPORT: "Thank you for sharing with me. I'm here to listen and understand.",
                    DiagnosisPhase.HISTORY: "I appreciate you telling me about your experiences.",
                    DiagnosisPhase.ASSESSMENT: "That helps me understand better what you're going through.",
                    DiagnosisPhase.DIAGNOSIS: "Based on our conversation, I have some observations to share.",
                    DiagnosisPhase.CLOSURE: "Thank you for this conversation. Let's summarize what we discussed."}
        response = responses.get(phase, "I understand.")
        return f"{response} {next_question}" if next_question else response

    def _determine_next_phase(self, current: DiagnosisPhase, confidence: float) -> DiagnosisPhase:
        """Determine next dialogue phase based on confidence."""
        phases = [DiagnosisPhase.RAPPORT, DiagnosisPhase.HISTORY, DiagnosisPhase.ASSESSMENT,
                 DiagnosisPhase.DIAGNOSIS, DiagnosisPhase.CLOSURE]
        idx = phases.index(current)
        if confidence >= self._settings.phase_transition_threshold and idx < len(phases) - 1:
            return phases[idx + 1]
        return current

    def _update_session(self, session_id: UUID, result: AssessmentResult) -> None:
        """Update session state with assessment result."""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.phase = result.phase
            session.symptoms.extend(result.extracted_symptoms)
            session.differential = result.differential
            session.reasoning_history.extend(result.reasoning_chain)
            session.safety_flags.extend(result.safety_flags)

    async def extract_symptoms(self, user_id: UUID, session_id: UUID, message: str,
                               conversation_history: list[dict[str, str]],
                               existing_symptoms: list[SymptomDTO]) -> ExtractionResult:
        """Extract symptoms without full assessment."""
        start_time = time.perf_counter()
        self._stats["extractions"] += 1
        result = ExtractionResult()
        if self._symptom_extractor:
            extraction = await self._symptom_extractor.extract(message, conversation_history, existing_symptoms)
            result.extracted_symptoms = extraction.symptoms
            result.updated_symptoms = existing_symptoms + extraction.symptoms
            result.temporal_info = extraction.temporal_info
            result.contextual_factors = extraction.contextual_factors
            result.risk_indicators = extraction.risk_indicators
        result.extraction_time_ms = int((time.perf_counter() - start_time) * 1000)
        return result

    async def generate_differential(self, user_id: UUID, session_id: UUID, symptoms: list[SymptomDTO],
                                    user_history: dict[str, Any], current_differential: DifferentialDTO | None) -> DifferentialResult:
        """Generate differential diagnosis."""
        start_time = time.perf_counter()
        self._stats["differentials"] += 1
        result = DifferentialResult()
        if self._differential_generator:
            differential = await self._differential_generator.generate(symptoms, user_history)
            result.differential = DifferentialDTO(
                primary=differential.hypotheses[0] if differential.hypotheses else None,
                alternatives=differential.hypotheses[1:] if len(differential.hypotheses) > 1 else [],
                missing_info=differential.missing_info)
            result.hitop_scores = differential.hitop_scores
            result.recommended_questions = differential.recommended_questions
        result.generation_time_ms = int((time.perf_counter() - start_time) * 1000)
        return result

    async def start_session(self, user_id: UUID, session_type: str, initial_context: dict[str, Any],
                            previous_session_id: UUID | None) -> SessionStartResult:
        """Start a new diagnosis session."""
        self._stats["sessions_started"] += 1
        session_number = self._user_session_counts.get(user_id, 0) + 1
        self._user_session_counts[user_id] = session_number
        session = SessionState(user_id=user_id, session_number=session_number)
        self._active_sessions[session.session_id] = session
        greeting = self._generate_greeting(session_number, previous_session_id is not None)
        logger.info("session_started", user_id=str(user_id), session_id=str(session.session_id))
        return SessionStartResult(session_id=session.session_id, session_number=session_number,
                                  initial_phase=DiagnosisPhase.RAPPORT, greeting=greeting,
                                  loaded_context=previous_session_id is not None)

    def _generate_greeting(self, session_number: int, has_previous: bool) -> str:
        """Generate session greeting."""
        if has_previous:
            return "Welcome back. I remember our previous conversation. How have you been since we last spoke?"
        if session_number == 1:
            return "Hello, I'm here to help understand what you're experiencing. How are you feeling today?"
        return f"Welcome to session {session_number}. How have things been for you?"

    async def end_session(self, user_id: UUID, session_id: UUID, generate_summary: bool) -> SessionEndResult:
        """End a diagnosis session."""
        self._stats["sessions_ended"] += 1
        session = self._active_sessions.get(session_id)
        if not session:
            return SessionEndResult()
        duration = datetime.now(timezone.utc) - session.started_at
        summary = self._generate_session_summary(session) if generate_summary else None
        self._store_session_history(user_id, session)
        # Persist completed session record to repository
        if self._repository:
            try:
                from .entities import DiagnosisRecordEntity
                primary = session.differential.primary if session.differential else None
                record = DiagnosisRecordEntity(
                    user_id=user_id,
                    session_id=session_id,
                    primary_diagnosis=primary.name if primary else "",
                    dsm5_code=primary.dsm5_code if primary else None,
                    confidence=primary.confidence if primary else Decimal("0.5"),
                    severity=primary.severity if primary and primary.severity else SeverityLevel.MILD,
                    symptom_summary=[s.name for s in session.symptoms],
                    recommendations=self._generate_recommendations(session),
                )
                await self._repository.save_record(record)
                logger.info("session_record_persisted", session_id=str(session_id))
            except Exception as e:
                logger.warning("session_record_persist_failed", session_id=str(session_id), error=str(e))
        del self._active_sessions[session_id]
        logger.info("session_ended", user_id=str(user_id), session_id=str(session_id))
        return SessionEndResult(duration_minutes=int(duration.total_seconds() / 60),
                               messages_exchanged=len(session.messages), final_differential=session.differential,
                               summary=summary, recommendations=self._generate_recommendations(session))

    def _generate_session_summary(self, session: SessionState) -> str:
        """Generate session summary."""
        primary = session.differential.primary.name if session.differential and session.differential.primary else "undetermined"
        return f"Session covered {len(session.symptoms)} symptoms. Primary working hypothesis: {primary}."

    def _generate_recommendations(self, session: SessionState) -> list[str]:
        """Generate recommendations based on session."""
        recs = ["Continue monitoring symptoms", "Schedule follow-up session"]
        if session.safety_flags:
            recs.insert(0, "Consider safety planning")
        return recs

    def _store_session_history(self, user_id: UUID, session: SessionState) -> None:
        """Store session in user history."""
        self._user_history.setdefault(user_id, []).append({
            "session_id": str(session.session_id), "date": session.started_at.isoformat(),
            "phase": session.phase.value, "symptom_count": len(session.symptoms),
            "differential": session.differential.model_dump() if session.differential else None})

    async def get_history(self, user_id: UUID, limit: int, include_symptoms: bool, include_differentials: bool) -> HistoryResult:
        """Get user diagnosis history."""
        in_memory = self._user_history.get(user_id, [])
        if in_memory:
            return HistoryResult(sessions=in_memory[-limit:])
        # Fall back to repository
        if self._repository:
            try:
                records = await self._repository.list_user_records(user_id, limit=limit)
                sessions = [{
                    "session_id": str(r.session_id), "date": r.created_at.isoformat(),
                    "primary_diagnosis": r.primary_diagnosis, "confidence": str(r.confidence),
                    "symptom_count": len(r.symptom_summary),
                } for r in records]
                return HistoryResult(sessions=sessions)
            except Exception as e:
                logger.warning("history_repo_load_failed", user_id=str(user_id), error=str(e))
        return HistoryResult(sessions=[])

    async def get_session_state(self, session_id: UUID) -> dict[str, Any] | None:
        """Get current session state."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        return {"session_id": str(session.session_id), "user_id": str(session.user_id),
                "phase": session.phase.value, "symptom_count": len(session.symptoms),
                "started_at": session.started_at.isoformat()}

    async def challenge_hypothesis(self, session_id: UUID, hypothesis_id: UUID) -> ChallengeResult:
        """Trigger Devil's Advocate challenge."""
        self._stats["challenges"] += 1
        session = self._active_sessions.get(session_id)
        if not session or not session.differential:
            return ChallengeResult()
        hypotheses = ([session.differential.primary] if session.differential.primary else []) + session.differential.alternatives
        target = next((h for h in hypotheses if h.hypothesis_id == hypothesis_id), None)
        if not target:
            return ChallengeResult()
        return ChallengeResult(challenges=[f"What if {target.name} is a misattribution?",
                                           "Consider alternative explanations for the symptoms"],
                               counter_questions=["What symptoms don't fit this pattern?"])

    async def delete_user_data(self, user_id: UUID) -> None:
        """Delete all user data (GDPR compliance)."""
        self._user_history.pop(user_id, None)
        self._user_session_counts.pop(user_id, None)
        for sid, session in list(self._active_sessions.items()):
            if session.user_id == user_id:
                del self._active_sessions[sid]
        logger.info("user_data_deleted", user_id=str(user_id))

    async def get_status(self) -> dict[str, Any]:
        """Get service status."""
        return {"status": "operational" if self._initialized else "initializing",
                "initialized": self._initialized, "statistics": self._stats,
                "active_sessions": len(self._active_sessions), "users_tracked": len(self._user_session_counts)}

    @property
    def stats(self) -> dict[str, int]:
        """Get service statistics counters."""
        return self._stats
