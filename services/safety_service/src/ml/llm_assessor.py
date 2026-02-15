"""
Solace-AI LLM Assessor - Deep risk assessment using LLM with structured output.
Uses Claude API (via LangChain) for nuanced clinical assessment of crisis risk.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import structlog

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from safety_service.src.infrastructure.telemetry import traced, get_telemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

    def traced(*args, **kwargs):
        """No-op decorator when telemetry unavailable."""
        def decorator(func):
            return func
        return decorator

logger = structlog.get_logger(__name__)


from solace_common.enums import CrisisLevel as RiskLevel  # noqa: E402


class RiskDimension(str, Enum):
    """Dimensions of risk assessment."""
    SUICIDAL_IDEATION = "SUICIDAL_IDEATION"
    SELF_HARM = "SELF_HARM"
    HOPELESSNESS = "HOPELESSNESS"
    EMOTIONAL_DISTRESS = "EMOTIONAL_DISTRESS"
    SOCIAL_ISOLATION = "SOCIAL_ISOLATION"
    SUBSTANCE_USE = "SUBSTANCE_USE"
    TRAUMA_SYMPTOMS = "TRAUMA_SYMPTOMS"
    IMPULSE_CONTROL = "IMPULSE_CONTROL"


class RiskFactor(BaseModel):
    """Identified risk factor from LLM analysis."""
    dimension: RiskDimension = Field(..., description="Risk dimension")
    severity: Decimal = Field(..., ge=0, le=1, description="Severity score")
    evidence: str = Field(..., description="Textual evidence")
    rationale: str = Field(..., description="Clinical reasoning")


class ProtectiveFactor(BaseModel):
    """Identified protective factor from LLM analysis."""
    factor: str = Field(..., description="Protective factor description")
    strength: Decimal = Field(..., ge=0, le=1, description="Strength of protective factor")
    evidence: str = Field(..., description="Supporting evidence")


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment from LLM."""
    assessment_id: UUID = Field(default_factory=uuid4, description="Unique assessment ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: Decimal = Field(..., ge=0, le=1, description="Quantitative risk score")
    confidence: Decimal = Field(..., ge=0, le=1, description="Assessment confidence")
    risk_factors: list[RiskFactor] = Field(default_factory=list, description="Identified risk factors")
    protective_factors: list[ProtectiveFactor] = Field(default_factory=list, description="Protective factors")
    clinical_summary: str = Field(..., description="Clinical interpretation")
    recommended_actions: list[str] = Field(default_factory=list, description="Recommended interventions")
    immediate_risk: bool = Field(default=False, description="Requires immediate intervention")
    contextual_notes: str = Field(default="", description="Additional context")


class LLMAssessorConfig(BaseSettings):
    """Configuration for LLM assessor."""
    model_name: str = Field(default="claude-sonnet-4-5-20250929", description="Claude model to use")
    max_tokens: int = Field(default=2000, ge=100, le=4096, description="Maximum response tokens")
    temperature: Decimal = Field(default=Decimal("0.3"), ge=0, le=1, description="Sampling temperature")
    enable_structured_output: bool = Field(default=True, description="Use structured JSON output")
    api_timeout_seconds: int = Field(default=30, ge=5, le=120, description="API timeout")
    enable_caching: bool = Field(default=True, description="Enable prompt caching")
    cache_ttl_minutes: int = Field(default=60, ge=1, le=1440, description="Cache TTL")
    max_input_chars: int = Field(default=12000, ge=1000, le=100000, description="Max input characters before truncation")
    min_tokens_floor: int = Field(default=500, ge=100, le=2000, description="Minimum response tokens guaranteed")

    model_config = SettingsConfigDict(env_prefix="LLM_ASSESSOR_", env_file=".env", extra="ignore")


class LLMAssessor:
    """
    LLM-based deep risk assessment using Claude API.
    Provides nuanced clinical assessment with structured outputs.
    """

    def __init__(
        self,
        config: LLMAssessorConfig | None = None,
        anthropic_client: Any | None = None,
        unified_llm_client: Any | None = None,
    ) -> None:
        """
        Initialize LLM assessor.

        Args:
            config: Configuration settings
            anthropic_client: Optional Anthropic client instance (deprecated, use LangChain instead)
            unified_llm_client: Optional UnifiedLLMClient instance (Portkey gateway)
        """
        self._config = config or LLMAssessorConfig()
        self._unified_llm_client = unified_llm_client
        self._assessment_cache: dict[str, RiskAssessment] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Initialize LangChain components
        if LANGCHAIN_AVAILABLE:
            try:
                # Ensure max_tokens is at least min_tokens_floor
                effective_max_tokens = max(
                    self._config.max_tokens,
                    self._config.min_tokens_floor,
                )
                self._llm = ChatAnthropic(
                    model=self._config.model_name,
                    max_tokens=effective_max_tokens,
                    temperature=float(self._config.temperature),
                    timeout=self._config.api_timeout_seconds
                )
                self._output_parser = JsonOutputParser()
                self._use_langchain = True
                logger.info("llm_assessor_initialized", mode="langchain", model=self._config.model_name)
            except Exception as e:
                logger.warning("langchain_init_failed", error=str(e), fallback="mock")
                self._llm = None
                self._output_parser = None
                self._use_langchain = False
        else:
            logger.warning("langchain_unavailable", fallback="mock")
            self._llm = None
            self._output_parser = None
            self._use_langchain = False

        # Fallback to legacy client if provided
        if not self._use_langchain and anthropic_client:
            self._client = anthropic_client
        else:
            self._client = None

    def _build_system_prompt(self) -> str:
        """Build system prompt for risk assessment."""
        return """You are an expert clinical psychologist specializing in crisis risk assessment.
Your role is to analyze text for signs of suicidal ideation, self-harm, and mental health crisis.

Assess the following dimensions:
- Suicidal ideation (thoughts, plans, intent)
- Self-harm behaviors or urges
- Hopelessness and worthlessness
- Emotional distress level
- Social isolation and withdrawal
- Substance use concerns
- Trauma symptoms (flashbacks, dissociation)
- Impulse control

For each dimension present, provide:
1. Severity score (0.0-1.0)
2. Specific evidence from the text
3. Clinical rationale

Also identify protective factors (hope, social support, coping skills, etc.).

Provide an overall risk level (CRITICAL, HIGH, ELEVATED, LOW, NONE) and quantitative risk score.

If CRITICAL or HIGH risk, mark as requiring immediate intervention.

Output must be valid JSON matching this schema:
{{
  "risk_level": "CRITICAL|HIGH|ELEVATED|LOW|NONE",
  "risk_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "risk_factors": [
    {{
      "dimension": "SUICIDAL_IDEATION|SELF_HARM|...",
      "severity": 0.0-1.0,
      "evidence": "quoted text from input",
      "rationale": "clinical reasoning"
    }}
  ],
  "protective_factors": [
    {{
      "factor": "description",
      "strength": 0.0-1.0,
      "evidence": "quoted text"
    }}
  ],
  "clinical_summary": "brief clinical interpretation",
  "recommended_actions": ["action1", "action2"],
  "immediate_risk": true|false,
  "contextual_notes": "additional context"
}}

Be conservative: when uncertain, err on the side of caution and assess higher risk."""

    def _build_user_prompt(self, text: str, context: dict[str, Any] | None = None) -> str:
        """Build user prompt with text to assess.

        Truncates input if it exceeds max_input_chars to stay within token budget
        while ensuring min_tokens_floor is available for the response.
        """
        max_chars = self._config.max_input_chars
        if len(text) > max_chars:
            logger.warning(
                "llm_assessor_input_truncated",
                original_length=len(text),
                truncated_to=max_chars,
            )
            text = text[:max_chars] + "\n\n[... content truncated for safety assessment ...]"

        prompt = f"Analyze the following text for mental health crisis risk:\n\n{text}"

        if context:
            context_str = json.dumps(context, indent=2)
            remaining = max(0, max_chars - len(prompt))
            if len(context_str) > remaining:
                context_str = context_str[:remaining] + "..."
            prompt += f"\n\nAdditional context:\n{context_str}"

        prompt += "\n\nProvide your risk assessment as JSON:"
        return prompt

    @traced(name="llm_assessor.assess", attributes={"component": "llm_assessor"})
    async def assess(self, text: str, user_id: UUID | None = None,
                    context: dict[str, Any] | None = None) -> RiskAssessment:
        """
        Perform deep risk assessment using LLM.

        Args:
            text: Text to assess
            user_id: Optional user ID for logging
            context: Optional additional context (prior assessments, demographics, etc.)

        Returns:
            Comprehensive risk assessment
        """
        if not text:
            return self._create_minimal_assessment("Empty input text")

        # Check cache if enabled
        if self._config.enable_caching:
            cache_key = self._generate_cache_key(text, context)
            if cached := self._get_cached_assessment(cache_key):
                logger.info("llm_assessment_cached", user_id=str(user_id) if user_id else None)
                return cached

        # Prepare prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, context)

        try:
            # Call LLM (mock implementation for now - replace with actual Anthropic API call)
            assessment_json = await self._call_llm(system_prompt, user_prompt)

            # Parse response
            assessment = self._parse_llm_response(assessment_json)

            # Cache result
            if self._config.enable_caching:
                self._cache_assessment(cache_key, assessment)

            if user_id:
                logger.info("llm_assessment_completed", user_id=str(user_id),
                           risk_level=assessment.risk_level.value,
                           immediate_risk=assessment.immediate_risk)

            return assessment

        except Exception as e:
            logger.error("llm_assessment_failed", error=str(e), user_id=str(user_id) if user_id else None)
            # Fallback to rule-based assessment
            return self._create_fallback_assessment(text, str(e))

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Call LLM API for assessment using LangChain.

        Uses LangChain's ChatAnthropic with structured output parsing.
        Falls back to mock response if LangChain is unavailable.
        """
        if self._use_langchain and self._llm:
            try:
                # Create prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", user_prompt)
                ])

                # Create chain: prompt | llm | output_parser
                chain = prompt | self._llm | self._output_parser

                # Invoke chain (LangChain handles retries and error handling)
                response = await chain.ainvoke({})

                return response

            except Exception as e:
                logger.error("langchain_call_failed", error=str(e), fallback="mock")
                # Fall through to mock response

        # Legacy client fallback (if provided)
        if self._client:
            try:
                response = await self._client.messages.create(
                    model=self._config.model_name,
                    max_tokens=self._config.max_tokens,
                    temperature=float(self._config.temperature),
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return json.loads(response.content[0].text)
            except Exception as e:
                logger.error("anthropic_call_failed", error=str(e), fallback="unified_llm")

        # UnifiedLLMClient fallback (Portkey gateway)
        if self._unified_llm_client and getattr(self._unified_llm_client, "is_available", False):
            try:
                response_text = await self._unified_llm_client.generate(
                    system_prompt=system_prompt,
                    user_message=user_prompt,
                    service_name="safety_llm_assessor",
                    task_type="crisis",
                    max_tokens=self._config.max_tokens,
                )
                if response_text:
                    parsed = json.loads(response_text)
                    logger.info("unified_llm_assessment_completed")
                    return parsed
            except json.JSONDecodeError as e:
                logger.error("unified_llm_json_parse_failed", error=str(e), fallback="rules")
            except Exception as e:
                logger.error("unified_llm_call_failed", error=str(e), fallback="rules")

        # Rule-based fallback when LLM is unavailable
        logger.warning("llm_unavailable_using_rule_based_fallback")

        # Analyze user_prompt text for crisis keywords
        text_lower = user_prompt.lower()
        critical_keywords = ["suicide", "kill myself", "end my life", "want to die", "take my life"]
        high_keywords = ["self-harm", "hurt myself", "cutting", "overdose", "no reason to live", "hopeless"]
        elevated_keywords = ["depressed", "anxious", "overwhelmed", "can't cope", "breaking down"]

        has_critical = any(kw in text_lower for kw in critical_keywords)
        has_high = any(kw in text_lower for kw in high_keywords)
        has_elevated = any(kw in text_lower for kw in elevated_keywords)

        if has_critical:
            risk_level, risk_score, immediate = "CRITICAL", 0.9, True
        elif has_high:
            risk_level, risk_score, immediate = "HIGH", 0.75, False
        elif has_elevated:
            risk_level, risk_score, immediate = "ELEVATED", 0.5, False
        else:
            risk_level, risk_score, immediate = "LOW", 0.2, False

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": 0.6,
            "risk_factors": [],
            "protective_factors": [],
            "clinical_summary": f"Rule-based assessment (LLM unavailable): {risk_level}",
            "recommended_actions": (
                ["Escalate to human clinician", "Provide crisis resources"]
                if immediate
                else ["Continue monitoring", "Provide support resources"]
            ),
            "immediate_risk": immediate,
            "contextual_notes": "LLM assessment not configured — rule-based keyword fallback used",
        }

    def _parse_llm_response(self, response: dict[str, Any]) -> RiskAssessment:
        """Parse LLM JSON response into RiskAssessment object."""
        # Parse risk factors
        risk_factors = [
            RiskFactor(
                dimension=RiskDimension(rf["dimension"]),
                severity=Decimal(str(rf["severity"])),
                evidence=rf["evidence"],
                rationale=rf["rationale"]
            )
            for rf in response.get("risk_factors", [])
        ]

        # Parse protective factors
        protective_factors = [
            ProtectiveFactor(
                factor=pf["factor"],
                strength=Decimal(str(pf["strength"])),
                evidence=pf["evidence"]
            )
            for pf in response.get("protective_factors", [])
        ]

        return RiskAssessment(
            risk_level=RiskLevel.from_string(response["risk_level"]),
            risk_score=Decimal(str(response["risk_score"])),
            confidence=Decimal(str(response["confidence"])),
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            clinical_summary=response["clinical_summary"],
            recommended_actions=response.get("recommended_actions", []),
            immediate_risk=response.get("immediate_risk", False),
            contextual_notes=response.get("contextual_notes", "")
        )

    def _create_minimal_assessment(self, reason: str) -> RiskAssessment:
        """Create minimal risk assessment."""
        return RiskAssessment(
            risk_level=RiskLevel.NONE,
            risk_score=Decimal("0.0"),
            confidence=Decimal("1.0"),
            clinical_summary=f"Minimal assessment: {reason}",
            recommended_actions=[]
        )

    def _create_fallback_assessment(self, text: str, error: str) -> RiskAssessment:
        """Create fallback assessment when LLM fails."""
        # Simple keyword-based fallback
        text_lower = text.lower()
        critical_keywords = ["suicide", "kill myself", "end my life", "want to die"]
        has_critical = any(kw in text_lower for kw in critical_keywords)

        risk_level = RiskLevel.HIGH if has_critical else RiskLevel.ELEVATED
        risk_score = Decimal("0.8") if has_critical else Decimal("0.5")

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=Decimal("0.5"),
            clinical_summary=f"Fallback assessment due to LLM error: {error}",
            recommended_actions=["Manual review required", "Escalate to human clinician"],
            immediate_risk=has_critical,
            contextual_notes="Assessment performed using rule-based fallback"
        )

    def _generate_cache_key(self, text: str, context: dict[str, Any] | None) -> str:
        """Generate cache key from text and context."""
        import hashlib
        content = text + json.dumps(context or {}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_assessment(self, cache_key: str) -> RiskAssessment | None:
        """Retrieve cached assessment if valid."""
        if cache_key not in self._assessment_cache:
            return None

        # Check if cache expired
        cache_time = self._cache_timestamps.get(cache_key)
        if not cache_time:
            return None

        ttl_delta = datetime.now(timezone.utc) - cache_time
        if ttl_delta.total_seconds() > self._config.cache_ttl_minutes * 60:
            # Expired
            del self._assessment_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._assessment_cache[cache_key]

    def _cache_assessment(self, cache_key: str, assessment: RiskAssessment) -> None:
        """Cache assessment result."""
        self._assessment_cache[cache_key] = assessment
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        # Limit cache size
        if len(self._assessment_cache) > 1000:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._assessment_cache[oldest_key]
            del self._cache_timestamps[oldest_key]

    def get_highest_risk_dimension(self, assessment: RiskAssessment) -> RiskDimension | None:
        """Get the dimension with highest risk severity."""
        if not assessment.risk_factors:
            return None
        return max(assessment.risk_factors, key=lambda rf: rf.severity).dimension
