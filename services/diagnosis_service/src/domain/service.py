"""
Solace-AI Diagnosis Service - 4-Step Chain-of-Reasoning Orchestration.
AMIE-inspired diagnostic reasoning with anti-sycophancy mechanisms.
"""
from __future__ import annotations
import asyncio
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
from .advocate import DevilsAdvocate, AdvocateSettings
from .confidence import ConfidenceCalibrator, ConfidenceSettings
from .erasure import build_user_data_erasure
from ..events import (
    EventDispatcher, SymptomExtractedEvent, HypothesisGeneratedEvent,
    SafetyFlagRaisedEvent, DiagnosisRecordedEvent,
)
from services.shared import ServiceBase

if TYPE_CHECKING:
    from .symptom_extractor import SymptomExtractor
    from .differential import DifferentialGenerator
    from ..infrastructure.repository import DiagnosisRepositoryPort
    from solace_infrastructure.gdpr import ErasureReport

logger = structlog.get_logger(__name__)

# Safety flags that escalate a turn to CRISIS (mirrors _step1_analyze /
# _determine_next_phase). A turn carrying any of these MUST be durably persisted:
# see assess() / _persist_session fail-loud path (P2-9).
_CRISIS_SAFETY_FLAGS = ("suicidal_ideation", "self_harm", "harm_to_others")


class DiagnosisServiceSettings(BaseSettings):
    """Diagnosis service configuration."""
    max_reasoning_time_ms: int = Field(default=10000)
    enable_anti_sycophancy: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.3)
    max_hypotheses: int = Field(default=5)
    phase_transition_threshold: float = Field(default=0.7)
    enable_longitudinal_tracking: bool = Field(default=True)
    challenge_frequency: float = Field(default=0.3)
    model_config = SettingsConfigDict(
        env_prefix="DIAGNOSIS_SERVICE_", env_file=".env", extra="ignore", populate_by_name=True
    )


class DiagnosisService(ServiceBase):
    """Main diagnosis service orchestrating 4-step Chain-of-Reasoning."""

    def __init__(self, settings: DiagnosisServiceSettings | None = None,
                 symptom_extractor: SymptomExtractor | None = None,
                 differential_generator: DifferentialGenerator | None = None,
                 repository: DiagnosisRepositoryPort | None = None,
                 advocate: DevilsAdvocate | None = None,
                 calibrator: ConfidenceCalibrator | None = None,
                 event_dispatcher: EventDispatcher | None = None,
                 session_store: Any = None) -> None:
        self._settings = settings or DiagnosisServiceSettings()
        self._symptom_extractor = symptom_extractor
        self._differential_generator = differential_generator
        self._repository = repository
        self._advocate = advocate or DevilsAdvocate()
        self._calibrator = calibrator or ConfidenceCalibrator()
        self._event_dispatcher = event_dispatcher
        self._session_store = session_store  # REV-16 durable Redis live-session store (optional)
        self._active_sessions: dict[UUID, SessionState] = {}
        self._user_session_counts: dict[UUID, int] = {}
        self._user_history: dict[UUID, list[dict[str, Any]]] = {}
        self._session_lock = asyncio.Lock()
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

    async def _load_session(self, session_id: UUID):
        """Recover a live session from the durable store on a cache miss (REV-16
        read-through), hydrating the in-process cache. Returns None if unknown.
        When no store is configured this is just the in-memory lookup.

        P1-5: on a cache HIT with a durable store configured, reconcile before use.
        A miss-only read-through trusts a possibly-stale cache: on a non-sticky
        multi-worker LB another worker can have advanced the durable session (e.g.
        escalated current risk + safety_flags) after this worker cached it. Re-read
        the durable copy and adopt it when it is newer (``durable.version >
        cached.version``), otherwise keep the cache — so this turn never operates on,
        and never re-persists over, downgraded clinical state."""
        async with self._session_lock:
            session = self._active_sessions.get(session_id)
            if self._session_store is not None:
                durable = await self._session_store.get_session(session_id)
                if session is None:
                    # Miss: hydrate from the durable store (unchanged behavior).
                    if durable is not None:
                        self._active_sessions[session_id] = durable
                        session = durable
                elif durable is not None and durable.version > session.version:
                    # Hit but stale: adopt the newer durable copy.
                    self._active_sessions[session_id] = durable
                    session = durable
            return session

    async def _persist_session(self, session, *, fail_loud: bool = False) -> None:
        """Durable write-through with CAS-merge on ``safety_flags`` (REV-16). No-op
        when no store is configured.

        A persistent CAS conflict is logged loud. It is normally best-effort — the
        in-memory working copy is authoritative for the current turn and the next
        turn re-persists — so it does NOT fail the turn. BUT when ``fail_loud`` is
        set (P2-9), the turn carries a crisis safety escalation that has no
        guaranteed "next turn": swallowing CAS exhaustion could leave the durable
        session reflecting the PRE-crisis state if this worker then crashes or the
        next turn lands elsewhere. In that case re-raise so the API surfaces a 500
        and the escalation is retried rather than silently un-persisted."""
        if self._session_store is None or session is None:
            return
        from ..infrastructure.redis_session_repository import SessionPersistConflict
        try:
            await self._session_store.save_session_cas_merge(session)
        except SessionPersistConflict:
            logger.error(
                "diagnosis_session_persist_cas_exhausted",
                session_id=str(session.session_id),
                crisis=fail_loud,
            )
            if fail_loud:
                raise

    async def assess(self, user_id: UUID, session_id: UUID, message: str,
                     conversation_history: list[dict[str, str]],
                     existing_symptoms: list[SymptomDTO],
                     current_phase: DiagnosisPhase,
                     current_differential: DifferentialDTO | None,
                     user_context: dict[str, Any]) -> AssessmentResult:
        """Execute full 4-step Chain-of-Reasoning assessment.

        The entire reasoning chain is guarded by a timeout derived from
        ``max_reasoning_time_ms``.  On timeout a partial result is returned
        with whatever steps completed before the deadline (L-06).
        """
        start_time = time.perf_counter()
        self._stats["assessments"] += 1
        # REV-16 read-through: recover the session from the durable store if it was
        # created on another worker (or before a restart), so the IDOR check and
        # _update_session operate on the real session instead of dropping the turn.
        await self._load_session(session_id)
        # IDOR guard (REV-07): assess mutates the session's clinical state, so an
        # already-existing session must belong to the caller.
        existing = self._active_sessions.get(session_id)
        if existing is not None and existing.user_id != user_id:
            logger.warning("assess_owner_mismatch", session_id=str(session_id), user_id=str(user_id))
            return AssessmentResult(
                phase=current_phase,
                extracted_symptoms=existing_symptoms,
                response_text="",
                confidence_score=Decimal("0"),
                processing_time_ms=0,
                safety_flags=["unauthorized_session"],
            )
        try:
            result = await asyncio.wait_for(
                self._run_assessment_chain(
                    user_id, session_id, message, conversation_history,
                    existing_symptoms, current_phase, current_differential,
                    user_context, start_time,
                ),
                timeout=self._settings.max_reasoning_time_ms / 1000,
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(
                "assessment_timeout",
                user_id=str(user_id),
                session_id=str(session_id),
                elapsed_ms=elapsed_ms,
                timeout_ms=self._settings.max_reasoning_time_ms,
            )
            result = AssessmentResult(
                phase=current_phase,
                extracted_symptoms=existing_symptoms,
                response_text="I need a moment to think through this more carefully. Could you tell me a bit more?",
                confidence_score=Decimal("0.3"),
                processing_time_ms=elapsed_ms,
                safety_flags=["reasoning_timeout"],
            )
        await self._update_session(session_id, result, message=message)
        # REV-16: durably write-through the updated session (symptoms, differential,
        # phase, safety_flags) with CAS-merge so a concurrent write can't clobber it.
        # P2-9: a turn carrying a crisis safety escalation must be durably recorded —
        # if CAS contention exhausts the retry budget, fail loud (raise) so the
        # escalation is retried, never silently dropped. Non-crisis turns stay
        # best-effort (the next turn re-persists).
        crisis_turn = any(f in _CRISIS_SAFETY_FLAGS for f in result.safety_flags)
        await self._persist_session(
            self._active_sessions.get(session_id), fail_loud=crisis_turn
        )
        logger.debug("assessment_completed", user_id=str(user_id), time_ms=result.processing_time_ms)
        return result

    async def _run_assessment_chain(
        self, user_id: UUID, session_id: UUID, message: str,
        conversation_history: list[dict[str, str]],
        existing_symptoms: list[SymptomDTO],
        current_phase: DiagnosisPhase,
        current_differential: DifferentialDTO | None,
        user_context: dict[str, Any],
        start_time: float,
    ) -> AssessmentResult:
        """Inner coroutine executing the 4-step reasoning chain."""
        result = AssessmentResult()
        reasoning_chain: list[ReasoningStepResultDTO] = []
        step1_start = time.perf_counter()
        step1_result = await self._step1_analyze(message, conversation_history, existing_symptoms,
                                                  user_id=user_id, session_id=session_id)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.ANALYZE, input_summary=f"Message: {message[:100]}...",
            output_summary=f"Extracted {len(step1_result['symptoms'])} symptoms",
            duration_ms=int((time.perf_counter() - step1_start) * 1000), details=step1_result,
        ))
        step2_start = time.perf_counter()
        step2_result = await self._step2_hypothesize(step1_result["symptoms"], user_context,
                                                      user_id=user_id, session_id=session_id)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.HYPOTHESIZE, input_summary=f"{len(step1_result['symptoms'])} symptoms",
            output_summary=f"Generated {len(step2_result['hypotheses'])} hypotheses",
            duration_ms=int((time.perf_counter() - step2_start) * 1000), details=step2_result,
        ))
        step3_start = time.perf_counter()
        step3_result = await self._step3_challenge(step2_result["hypotheses"], step1_result["symptoms"])
        # Forward missing_info from step 2 so step 4 can use it
        step3_result["missing_info"] = step2_result.get("missing_info", [])
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.CHALLENGE, input_summary=f"{len(step2_result['hypotheses'])} hypotheses",
            output_summary=f"Identified {len(step3_result['challenges'])} challenges",
            duration_ms=int((time.perf_counter() - step3_start) * 1000), details=step3_result,
        ))
        step4_start = time.perf_counter()
        # Retrieve session for calibration context (H-08)
        session = self._active_sessions.get(session_id)
        # REV-30: the current message's freshly-extracted symptoms (step 1) are not
        # merged into session.symptoms until _update_session runs AFTER this chain
        # completes. Without merging them here, step-4 Bayesian calibration would see
        # only PRIOR-turn symptoms and ignore THIS utterance's symptoms. Build a
        # current-turn-inclusive working set, deduplicated by name (case-insensitive)
        # against the persisted session symptoms — mirroring the dedup _update_session
        # later applies — so calibration counts this turn's symptoms without the
        # subsequent persistence double-counting them. This is a local view only;
        # session.symptoms is left untouched here.
        prior_symptoms = list(session.symptoms) if session else []
        prior_names = {s.name.lower() for s in prior_symptoms}
        current_new = [s for s in step1_result["symptoms"] if s.name.lower() not in prior_names]
        working_symptoms = prior_symptoms + current_new
        step4_result = await self._step4_synthesize(step2_result["hypotheses"], step3_result,
                                                     current_phase, message, session=session,
                                                     working_symptoms=working_symptoms)
        reasoning_chain.append(ReasoningStepResultDTO(
            step=ReasoningStep.SYNTHESIZE, input_summary="All previous steps",
            output_summary=f"Final confidence: {step4_result['confidence']:.2f}",
            duration_ms=int((time.perf_counter() - step4_start) * 1000), details=step4_result,
        ))
        result.safety_flags = step1_result.get("risk_indicators", [])
        result.phase = self._determine_next_phase(current_phase, step4_result["confidence"],
                                                   safety_flags=result.safety_flags)
        result.extracted_symptoms = step1_result["symptoms"]
        result.differential = step4_result["differential"]
        result.reasoning_chain = reasoning_chain
        result.next_question = step4_result["next_question"]
        result.response_text = step4_result["response"]
        result.confidence_score = Decimal(str(step4_result["confidence"]))
        result.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        return result

    async def _step1_analyze(self, message: str, history: list[dict[str, str]],
                             existing: list[SymptomDTO],
                             user_id: UUID | None = None,
                             session_id: UUID | None = None) -> dict[str, Any]:
        """Step 1: Analyze - Extract symptoms and information."""
        if self._symptom_extractor:
            extraction = await self._symptom_extractor.extract(message, history, existing)
            result = {"symptoms": extraction.symptoms, "temporal": extraction.temporal_info,
                      "contextual": extraction.contextual_factors, "risk_indicators": extraction.risk_indicators}
        else:
            result = {"symptoms": existing, "temporal": {}, "contextual": [], "risk_indicators": []}
        # Dispatch symptom extraction events
        if self._event_dispatcher and result["symptoms"]:
            for s in result["symptoms"]:
                await self._event_dispatcher.dispatch(SymptomExtractedEvent(
                    user_id=user_id, session_id=session_id,
                    symptom_id=s.symptom_id, symptom_name=s.name,
                    severity=s.severity, confidence=s.confidence,
                    extracted_from=message[:200],
                ))
        # Dispatch safety flag events for risk indicators
        if self._event_dispatcher and result["risk_indicators"]:
            safety_flags = result["risk_indicators"]
            for flag in safety_flags:
                if flag in ("suicidal_ideation", "self_harm", "harm_to_others"):
                    await self._event_dispatcher.dispatch(SafetyFlagRaisedEvent(
                        user_id=user_id, session_id=session_id,
                        flag_type=flag, severity="high",
                        trigger_text=message[:200],
                        recommended_action="Immediate clinical review",
                    ))
        return result

    async def _step2_hypothesize(self, symptoms: list[SymptomDTO], user_context: dict[str, Any],
                                user_id: UUID | None = None,
                                session_id: UUID | None = None) -> dict[str, Any]:
        """Step 2: Hypothesize - Generate differential diagnosis."""
        if self._differential_generator:
            differential = await self._differential_generator.generate(symptoms, user_context)
            result = {"hypotheses": differential.hypotheses, "missing_info": differential.missing_info,
                      "hitop_scores": differential.hitop_scores}
        else:
            result = {"hypotheses": [], "missing_info": [], "hitop_scores": {}}
        # Dispatch hypothesis generated events
        if self._event_dispatcher and result["hypotheses"]:
            for h in result["hypotheses"]:
                await self._event_dispatcher.dispatch(HypothesisGeneratedEvent(
                    user_id=user_id, session_id=session_id,
                    hypothesis_id=h.hypothesis_id, hypothesis_name=h.name,
                    confidence=h.confidence, dsm5_code=h.dsm5_code,
                    criteria_met_count=len(h.criteria_met),
                ))
        return result

    async def _step3_challenge(self, hypotheses: list[HypothesisDTO], symptoms: list[SymptomDTO]) -> dict[str, Any]:
        """Step 3: Challenge - Devil's Advocate adversarial review."""
        if not self._settings.enable_anti_sycophancy:
            return {"challenges": [], "alternatives": [], "biases": [],
                    "per_hypothesis_adjustments": {}}
        all_challenges: list[str] = []
        all_alternatives: list[str] = []
        all_biases: list[str] = []
        per_hypothesis_adjustments: dict[UUID, Decimal] = {}
        for hyp in hypotheses[:self._settings.max_hypotheses]:
            challenge_result = await self._advocate.challenge_hypothesis(hyp, symptoms, {})
            all_challenges.extend(challenge_result.challenges)
            all_alternatives.extend(challenge_result.alternative_explanations)
            all_biases.extend(challenge_result.bias_flags)
            per_hypothesis_adjustments[hyp.hypothesis_id] = challenge_result.confidence_adjustment
        bias_result = await self._advocate.analyze_bias(hypotheses, symptoms, [])
        all_biases.extend(bias_result.detected_biases)
        return {
            "challenges": all_challenges,
            "alternatives": all_alternatives,
            "biases": list(set(all_biases)),
            "per_hypothesis_adjustments": per_hypothesis_adjustments,
            "bias_analysis": bias_result,
        }

    async def _step4_synthesize(self, hypotheses: list[HypothesisDTO], challenge_result: dict[str, Any],
                                 phase: DiagnosisPhase, message: str,
                                 session: SessionState | None = None,
                                 working_symptoms: list[SymptomDTO] | None = None) -> dict[str, Any]:
        """Step 4: Synthesize - Integrate and generate response."""
        calibrated = await self._calibrate_confidence(hypotheses, challenge_result, session=session,
                                                       working_symptoms=working_symptoms)
        primary = calibrated[0] if calibrated else None
        alternatives = calibrated[1:] if len(calibrated) > 1 else []
        differential = DifferentialDTO(primary=primary, alternatives=alternatives,
                                        missing_info=challenge_result.get("missing_info", []))
        next_question = self._generate_next_question(phase, differential, challenge_result)
        response = self._generate_response(phase, differential, next_question)
        return {"differential": differential, "next_question": next_question,
                "response": response, "confidence": float(primary.confidence) if primary else 0.5}

    async def _calibrate_confidence(self, hypotheses: list[HypothesisDTO], challenges: dict[str, Any],
                                     session: SessionState | None = None,
                                     working_symptoms: list[SymptomDTO] | None = None) -> list[HypothesisDTO]:
        """Calibrate confidence scores using Bayesian calibrator and challenge results."""
        calibrated_hypotheses: list[HypothesisDTO] = []
        per_hypothesis_adjustments: dict[UUID, Decimal] = challenges.get("per_hypothesis_adjustments", {})
        # REV-30: prefer the current-turn-inclusive working symptom set so THIS
        # message's symptoms inform its own calibration. Fall back to the persisted
        # session symptoms (prior turns only, H-08) when no working set is supplied.
        if working_symptoms is not None:
            session_symptoms = working_symptoms
        else:
            session_symptoms = session.symptoms if session else []
        session_phase = session.phase.value if session else "unknown"
        for h in hypotheses:
            cal_result = await self._calibrator.calibrate(
                h, session_symptoms, {"phase": session_phase}
            )
            # Apply per-hypothesis adjustment instead of total (H-07)
            hyp_adjustment = per_hypothesis_adjustments.get(h.hypothesis_id, Decimal("0.0"))
            adjusted = max(Decimal("0.1"), cal_result.calibrated_confidence + hyp_adjustment)
            adjusted = min(Decimal("0.95"), adjusted)
            calibrated_hypotheses.append(HypothesisDTO(
                hypothesis_id=h.hypothesis_id, name=h.name, dsm5_code=h.dsm5_code, icd11_code=h.icd11_code,
                confidence=adjusted, confidence_interval=cal_result.confidence_interval,
                criteria_met=h.criteria_met, criteria_missing=h.criteria_missing,
                supporting_evidence=h.supporting_evidence, contra_evidence=h.contra_evidence,
                severity=h.severity, hitop_dimensions=h.hitop_dimensions,
            ))
        return sorted(calibrated_hypotheses, key=lambda h: h.confidence, reverse=True)

    def _generate_next_question(self, phase: DiagnosisPhase, differential: DifferentialDTO, challenges: dict) -> str | None:
        """Generate next question based on phase and differential."""
        questions = {DiagnosisPhase.RAPPORT: "How have you been feeling overall lately?",
                    DiagnosisPhase.HISTORY: "Can you tell me more about when these feelings started?",
                    DiagnosisPhase.ASSESSMENT: "On a scale of 0-10, how would you rate the intensity?",
                    DiagnosisPhase.DIAGNOSIS: "Based on what we've discussed, does this resonate with you?",
                    DiagnosisPhase.CRISIS: "I want to make sure you're safe right now. Can you tell me more about how you're feeling?"}
        if differential.missing_info:
            return f"Could you tell me more about {differential.missing_info[0]}?"
        return questions.get(phase)

    def _generate_response(self, phase: DiagnosisPhase, differential: DifferentialDTO, next_question: str | None) -> str:
        """Generate empathetic response based on phase."""
        responses = {DiagnosisPhase.RAPPORT: "Thank you for sharing with me. I'm here to listen and understand.",
                    DiagnosisPhase.HISTORY: "I appreciate you telling me about your experiences.",
                    DiagnosisPhase.ASSESSMENT: "That helps me understand better what you're going through.",
                    DiagnosisPhase.DIAGNOSIS: "Based on our conversation, I have some observations to share.",
                    DiagnosisPhase.CLOSURE: "Thank you for this conversation. Let's summarize what we discussed.",
                    DiagnosisPhase.CRISIS: "I hear you, and your safety is my top priority right now."}
        response = responses.get(phase, "I understand.")
        return f"{response} {next_question}" if next_question else response

    def _determine_next_phase(self, current: DiagnosisPhase, confidence: float,
                              safety_flags: list[str] | None = None) -> DiagnosisPhase:
        """Determine next dialogue phase based on confidence and safety flags."""
        # Safety flags override: escalate to CRISIS immediately
        if any(f in (safety_flags or []) for f in ("suicidal_ideation", "self_harm", "harm_to_others")):
            return DiagnosisPhase.CRISIS
        phases = [DiagnosisPhase.RAPPORT, DiagnosisPhase.HISTORY, DiagnosisPhase.ASSESSMENT,
                 DiagnosisPhase.DIAGNOSIS, DiagnosisPhase.CLOSURE]
        idx = phases.index(current) if current in phases else 0
        if current == DiagnosisPhase.CRISIS:
            return DiagnosisPhase.CRISIS
        if confidence >= self._settings.phase_transition_threshold and idx < len(phases) - 1:
            return phases[idx + 1]
        return current

    async def _update_session(self, session_id: UUID, result: AssessmentResult,
                              message: str | None = None) -> None:
        """Update session state with assessment result."""
        async with self._session_lock:
            session = self._active_sessions.get(session_id)
            if session is None:
                # REV-16: never SILENTLY drop clinical state. A missing session
                # here means assess() ran against a session that was never started
                # or was lost (e.g. a process restart before session durability
                # lands). Record it loud with exactly what is being dropped so it
                # is observable; escalate to ERROR when crisis safety_flags —
                # which must never vanish without a trace — would be lost.
                self._stats["unknown_session_updates"] = (
                    self._stats.get("unknown_session_updates", 0) + 1
                )
                log = logger.error if result.safety_flags else logger.warning
                log(
                    "diagnosis_update_unknown_session",
                    session_id=str(session_id),
                    dropped_symptoms=len(result.extracted_symptoms),
                    dropped_safety_flags=len(result.safety_flags),
                    safety_flags=list(result.safety_flags),
                )
                return
            session.phase = result.phase
            # Append user message and generated response to session messages (M-07)
            if message is not None:
                session.messages.append({"role": "user", "content": message})
            if result.response_text:
                session.messages.append({"role": "assistant", "content": result.response_text})
            # Deduplicate symptoms before extending
            existing_names = {s.name.lower() for s in session.symptoms}
            new_symptoms = [s for s in result.extracted_symptoms if s.name.lower() not in existing_names]
            session.symptoms.extend(new_symptoms)
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
        async with self._session_lock:
            session_number = self._user_session_counts.get(user_id, 0) + 1
            self._user_session_counts[user_id] = session_number
            session = SessionState(user_id=user_id, session_number=session_number)
            self._active_sessions[session.session_id] = session
        # REV-16: persist the new session so a later assess on any worker (or after
        # a restart) recovers it via read-through instead of dropping the turn.
        await self._persist_session(session)
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
        session = await self._load_session(session_id)
        if not session:
            return SessionEndResult()
        # IDOR guard (REV-07): only the session's owner may end it. Sessions are
        # loaded by id, so without this a caller could end another user's active
        # clinical session.
        if session.user_id != user_id:
            logger.warning("end_session_owner_mismatch", session_id=str(session_id))
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
                # M1-gate P0: do NOT swallow a clinical-record persistence failure.
                # Returning a successful SessionEndResult while the diagnosis record
                # never landed silently loses clinical data. Fail loud so the caller
                # /API surfaces it (500) and can retry.
                logger.error("session_record_persist_failed", session_id=str(session_id), error=str(e))
                raise
        # Dispatch diagnosis recorded event (C-14)
        if self._event_dispatcher:
            primary = session.differential.primary if session.differential else None
            await self._event_dispatcher.dispatch(DiagnosisRecordedEvent(
                user_id=user_id, session_id=session_id,
                primary_diagnosis=primary.name if primary else "",
                dsm5_code=primary.dsm5_code if primary else None,
                severity=primary.severity if primary and primary.severity else SeverityLevel.MILD,
                confidence=primary.confidence if primary else Decimal("0.5"),
            ))
        async with self._session_lock:
            self._active_sessions.pop(session_id, None)
        # REV-16: evict the live session from the durable store so a stale blob is
        # not later recovered by read-through after the session has ended.
        if self._session_store is not None:
            await self._session_store.delete_session(session_id)
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
        async with self._session_lock:
            session = self._active_sessions.get(session_id)
            if not session:
                return None
            return {"session_id": str(session.session_id), "user_id": str(session.user_id),
                    "phase": session.phase.value, "symptom_count": len(session.symptoms),
                    "started_at": session.started_at.isoformat()}

    async def challenge_hypothesis(
        self, session_id: UUID, hypothesis_id: UUID, user_id: UUID | None = None
    ) -> ChallengeResult:
        """Trigger Devil's Advocate challenge.

        Delegates to the DevilsAdvocate domain component for structured
        adversarial review instead of returning hardcoded strings (M-11).

        ``user_id`` is the authenticated caller; when provided it is checked
        against the session owner (REV-07 IDOR guard).
        """
        self._stats["challenges"] += 1
        async with self._session_lock:
            session = self._active_sessions.get(session_id)
        if not session or not session.differential:
            return ChallengeResult()
        # IDOR guard (REV-07): only the session owner may challenge its hypotheses.
        if user_id is not None and str(session.user_id) != str(user_id):
            logger.warning("challenge_owner_mismatch", session_id=str(session_id))
            return ChallengeResult()
        hypotheses = ([session.differential.primary] if session.differential.primary else []) + session.differential.alternatives
        target = next((h for h in hypotheses if h.hypothesis_id == hypothesis_id), None)
        if not target:
            return ChallengeResult()
        advocate_result = await self._advocate.challenge_hypothesis(target, session.symptoms, {})
        return ChallengeResult(
            challenges=advocate_result.challenges,
            counter_questions=advocate_result.counter_questions,
            bias_flags=advocate_result.bias_flags,
        )

    def _emit_erasure_audit(self, report: "ErasureReport") -> None:
        """Audit sink for a completed GDPR erasure.

        Invoked by the orchestrator ONLY on a fully successful erasure, giving a
        durable, structured audit trail of the right-to-erasure fulfilment.
        """
        logger.info("gdpr_erasure_completed", **report.to_dict())

    async def delete_user_data(self, user_id: UUID) -> None:
        """Delete all user data (GDPR compliance).

        Ephemeral, process-local caches are cleared directly; the durable
        Postgres store is erased through the shared fail-loud orchestrator
        (REV-17) so a failed delete raises ``ErasureIncomplete`` and is NEVER
        reported as a completed erasure. Preserves the existing ``None`` return
        (the endpoint maps success to 204).
        """
        # Process-local caches (not a durable GDPR store) — clear directly.
        self._user_history.pop(user_id, None)
        self._user_session_counts.pop(user_id, None)
        async with self._session_lock:
            for sid, session in list(self._active_sessions.items()):
                if session.user_id == user_id:
                    self._active_sessions.pop(sid, None)
        # Delete persisted data from Postgres through the auditable, fail-loud
        # right-to-erasure orchestrator (M-08, REV-17).
        if self._repository:
            # P0 (Phase C gate): also erase the durable live-session Redis store,
            # which holds special-category PHI (symptoms/differential/safety_flags)
            # that would otherwise escape erasure while it is attested complete.
            erasure = build_user_data_erasure(self._repository, session_store=self._session_store)
            # P1 (Phase C gate) / C.2 RLS: scope the per-user GUC to the TARGET user
            # for the erase — the diagnosis Postgres tables are RLS-protected, so an
            # admin-triggered erasure (request GUC = admin) would be RLS-filtered to
            # the admin's own rows and silently delete nothing. Restore afterwards.
            from solace_common.request_context import (
                reset_current_user_id,
                set_current_user_id,
            )
            token = set_current_user_id(str(user_id))
            try:
                report = await erasure.erase(user_id, audit_emit=self._emit_erasure_audit)
            finally:
                reset_current_user_id(token)
            logger.info(
                "user_data_deleted",
                user_id=str(user_id),
                total_deleted=report.total_deleted,
                stores=list(report.deleted),
            )
        else:
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
