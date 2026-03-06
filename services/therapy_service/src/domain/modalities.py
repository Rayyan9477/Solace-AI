"""
Solace-AI Therapy Service - Evidence-Based Therapy Modality Implementations.
CBT, DBT, ACT, MI, and Mindfulness modality frameworks with technique protocols.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4
import structlog

from ..schemas import TherapyModality, SessionPhase, SeverityLevel

logger = structlog.get_logger(__name__)


class ModalityPhase(str, Enum):
    """Modality-specific treatment phases."""
    PSYCHOEDUCATION = "psychoeducation"
    SKILL_INTRODUCTION = "skill_introduction"
    SKILL_PRACTICE = "skill_practice"
    SKILL_GENERALIZATION = "skill_generalization"
    MAINTENANCE = "maintenance"


@dataclass
class ModalityProtocol:
    """Protocol definition for a modality."""
    modality: TherapyModality
    name: str
    description: str
    core_principles: list[str] = field(default_factory=list)
    key_techniques: list[str] = field(default_factory=list)
    session_structure: dict[str, list[str]] = field(default_factory=dict)
    contraindications: list[str] = field(default_factory=list)
    efficacy_evidence: str = ""


@dataclass
class TechniqueStep:
    """A step in a technique protocol."""
    step_number: int
    instruction: str
    therapist_action: str
    expected_response: str = ""
    tips: list[str] = field(default_factory=list)


@dataclass
class TechniqueProtocol:
    """Detailed protocol for a therapeutic technique."""
    technique_id: UUID = field(default_factory=uuid4)
    name: str = ""
    modality: TherapyModality = TherapyModality.CBT
    category: str = ""
    description: str = ""
    rationale: str = ""
    steps: list[TechniqueStep] = field(default_factory=list)
    duration_minutes: int = 15
    materials_needed: list[str] = field(default_factory=list)
    adaptations: dict[str, str] = field(default_factory=dict)
    common_challenges: list[str] = field(default_factory=list)


@dataclass
class InterventionContext:
    """Context for intervention delivery."""
    user_id: UUID
    session_phase: SessionPhase
    severity: SeverityLevel
    current_concern: str
    previous_techniques: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    contraindications: list[str] = field(default_factory=list)


@dataclass
class InterventionResult:
    """Result from intervention delivery."""
    success: bool
    technique_name: str
    response_generated: str = ""
    follow_up_prompts: list[str] = field(default_factory=list)
    homework_suggestion: str = ""
    notes: list[str] = field(default_factory=list)


class ModalityProvider(ABC):
    """Abstract base for modality implementations."""

    @property
    @abstractmethod
    def modality(self) -> TherapyModality:
        """Get the modality type."""
        pass

    @property
    @abstractmethod
    def protocol(self) -> ModalityProtocol:
        """Get the modality protocol."""
        pass

    @abstractmethod
    def get_techniques(self) -> list[TechniqueProtocol]:
        """Get available techniques for this modality."""
        pass

    @abstractmethod
    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        """Select appropriate intervention for context."""
        pass

    @abstractmethod
    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        """Generate therapeutic response using technique."""
        pass


class CBTProvider(ModalityProvider):
    """Cognitive Behavioral Therapy implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.CBT, name="Cognitive Behavioral Therapy",
            description="Evidence-based therapy focusing on the relationship between thoughts, feelings, and behaviors",
            core_principles=["Thoughts influence emotions and behaviors", "Cognitive distortions maintain distress",
                            "Behavioral experiments test beliefs", "Skills generalize to daily life"],
            key_techniques=["Thought Record", "Behavioral Activation", "Cognitive Restructuring",
                           "Exposure Therapy", "Problem Solving", "Socratic Questioning"],
            session_structure={"opening": ["Mood check", "Agenda setting", "Homework review"],
                              "working": ["Technique application", "Skill practice", "Discussion"],
                              "closing": ["Summary", "Homework assignment", "Feedback"]},
            contraindications=["Active psychosis", "Severe cognitive impairment", "Acute crisis without stabilization"],
            efficacy_evidence="Meta-analyses show large effect sizes (d=0.8-1.0) for depression and anxiety",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.CBT

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        """Initialize CBT techniques."""
        return [
            TechniqueProtocol(
                name="Thought Record", modality=TherapyModality.CBT, category="cognitive_restructuring",
                description="Systematic examination and challenging of automatic negative thoughts",
                rationale="Helps identify cognitive distortions and develop more balanced thinking",
                steps=[
                    TechniqueStep(1, "Identify the situation that triggered distress", "Ask about recent difficult moment", "Client describes specific situation"),
                    TechniqueStep(2, "Identify the automatic thought", "What went through your mind?", "Client states the thought"),
                    TechniqueStep(3, "Rate emotion intensity 0-100", "How strong was the feeling?", "Numerical rating"),
                    TechniqueStep(4, "Examine evidence for the thought", "What supports this thought?", "Client lists evidence"),
                    TechniqueStep(5, "Examine evidence against", "What contradicts this thought?", "Alternative evidence"),
                    TechniqueStep(6, "Generate balanced alternative", "What's a more balanced view?", "New perspective"),
                ],
                duration_minutes=20, materials_needed=["Thought record worksheet"],
            ),
            TechniqueProtocol(
                name="Behavioral Activation", modality=TherapyModality.CBT, category="behavioral",
                description="Scheduling and engaging in mood-boosting activities",
                rationale="Activity and mood are connected; increasing pleasant activities improves mood",
                steps=[
                    TechniqueStep(1, "Review activity-mood connection", "Explain behavioral activation rationale", "Client understands concept"),
                    TechniqueStep(2, "Identify values-based activities", "What activities align with what matters to you?", "Activity list"),
                    TechniqueStep(3, "Schedule specific activities", "When exactly will you do this?", "Scheduled activities"),
                    TechniqueStep(4, "Identify barriers", "What might get in the way?", "Potential obstacles"),
                    TechniqueStep(5, "Problem-solve barriers", "How can we address these?", "Solutions identified"),
                ],
                duration_minutes=15, materials_needed=["Activity schedule"],
            ),
            TechniqueProtocol(
                name="Socratic Questioning", modality=TherapyModality.CBT, category="cognitive_restructuring",
                description="Guided questioning to help examine and challenge thoughts",
                rationale="Questions help clients discover alternative perspectives themselves",
                steps=[
                    TechniqueStep(1, "Identify the belief to examine", "What thought would you like to look at?", "Target belief stated"),
                    TechniqueStep(2, "Ask about evidence", "What evidence supports this?", "Evidence reviewed"),
                    TechniqueStep(3, "Explore alternatives", "Is there another way to see this?", "Alternatives considered"),
                    TechniqueStep(4, "Consider implications", "If this were true, what would it mean?", "Implications explored"),
                    TechniqueStep(5, "Test the thought", "How could we test this belief?", "Experiment designed"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        """Select appropriate CBT technique."""
        concern_lower = context.current_concern.lower()
        if "thought" in concern_lower or "think" in concern_lower or "believe" in concern_lower:
            return next((t for t in self._techniques if t.name == "Thought Record"), None)
        if "do" in concern_lower or "activity" in concern_lower or "motivation" in concern_lower:
            return next((t for t in self._techniques if t.name == "Behavioral Activation"), None)
        if context.session_phase == SessionPhase.WORKING:
            return next((t for t in self._techniques if t.name == "Socratic Questioning"), None)
        return self._techniques[0] if self._techniques else None

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        """Generate CBT response."""
        if technique.name == "Thought Record":
            response = "Let's examine that thought more closely. First, can you tell me about the specific situation that triggered this thought? What was happening right before you noticed this feeling?"
            follow_ups = ["What went through your mind in that moment?", "How intense was the emotion on a scale of 0-100?"]
        elif technique.name == "Behavioral Activation":
            response = "It sounds like connecting with activities that matter to you could be helpful. What are some activities that used to bring you a sense of accomplishment or pleasure, even small ones?"
            follow_ups = ["When could you schedule this activity this week?", "What might get in the way?"]
        else:
            response = "I'm curious about that thought. What evidence do you have that supports it? And is there any evidence that might point in a different direction?"
            follow_ups = ["Is there another way to look at this situation?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion=f"Practice {technique.name.lower()} technique daily", notes=["Used CBT framework"],
        )


class DBTProvider(ModalityProvider):
    """Dialectical Behavior Therapy implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.DBT, name="Dialectical Behavior Therapy",
            description="Skills-based therapy balancing acceptance and change for emotion regulation",
            core_principles=["Dialectical thinking - both/and", "Radical acceptance", "Wise mind integration",
                            "Behavior is functional", "Validation and change strategies"],
            key_techniques=["STOP Skill", "DEAR MAN", "Radical Acceptance", "Opposite Action",
                           "TIPP", "Mindfulness of Current Emotion"],
            session_structure={"opening": ["Diary card review", "Mindfulness exercise", "Agenda"],
                              "working": ["Skills teaching", "Behavioral analysis", "Practice"],
                              "closing": ["Summary", "Commitment", "Diary card prep"]},
            contraindications=["Inability to commit to treatment", "Active substance intoxication"],
            efficacy_evidence="RCTs show efficacy for BPD, suicidal behavior, and emotion dysregulation",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.DBT

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        """Initialize DBT techniques."""
        return [
            TechniqueProtocol(
                name="STOP Skill", modality=TherapyModality.DBT, category="distress_tolerance",
                description="Interrupt impulsive reactions in moments of distress",
                rationale="Creates space between stimulus and response to allow wise action",
                steps=[
                    TechniqueStep(1, "Stop - freeze and don't react", "When you notice distress, pause completely", "Client pauses"),
                    TechniqueStep(2, "Take a step back", "Take a breath. Don't act immediately", "Breathing observed"),
                    TechniqueStep(3, "Observe", "Notice what's happening inside and around you", "Observations shared"),
                    TechniqueStep(4, "Proceed mindfully", "Act with awareness based on your values", "Thoughtful action"),
                ],
                duration_minutes=10, materials_needed=[],
            ),
            TechniqueProtocol(
                name="DEAR MAN", modality=TherapyModality.DBT, category="interpersonal_effectiveness",
                description="Assertive communication framework for getting needs met",
                rationale="Structured approach to express needs while maintaining relationships",
                steps=[
                    TechniqueStep(1, "Describe the situation objectively", "Stick to facts without judgment", "Situation described"),
                    TechniqueStep(2, "Express feelings using I-statements", "Share how you feel about it", "Feelings expressed"),
                    TechniqueStep(3, "Assert your needs clearly", "Ask for what you want specifically", "Request stated"),
                    TechniqueStep(4, "Reinforce the benefit", "Explain positive outcomes of compliance", "Benefits explained"),
                    TechniqueStep(5, "Mindful - stay focused", "Keep to the point, ignore attacks", "Focus maintained"),
                    TechniqueStep(6, "Appear confident", "Use confident tone and posture", "Confidence shown"),
                    TechniqueStep(7, "Negotiate", "Be willing to give to get", "Compromise explored"),
                ],
                duration_minutes=20, materials_needed=["DEAR MAN worksheet"],
            ),
            TechniqueProtocol(
                name="Radical Acceptance", modality=TherapyModality.DBT, category="distress_tolerance",
                description="Complete acceptance of reality without judgment",
                rationale="Suffering = Pain + Non-acceptance; acceptance reduces suffering",
                steps=[
                    TechniqueStep(1, "Identify what needs accepting", "What reality are you fighting?", "Reality identified"),
                    TechniqueStep(2, "Acknowledge the pain", "It makes sense this is hard", "Pain validated"),
                    TechniqueStep(3, "Practice acceptance statements", "This is how it is right now", "Acceptance practiced"),
                    TechniqueStep(4, "Note resistance", "Notice urges to fight reality", "Resistance observed"),
                    TechniqueStep(5, "Return to acceptance", "Gently come back to accepting", "Recentered"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        """Select appropriate DBT technique."""
        if context.severity in [SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE]:
            return next((t for t in self._techniques if t.name == "STOP Skill"), None)
        concern_lower = context.current_concern.lower()
        if any(word in concern_lower for word in ["relationship", "communicate", "boundary", "assert"]):
            return next((t for t in self._techniques if t.name == "DEAR MAN"), None)
        if any(word in concern_lower for word in ["accept", "can't change", "stuck", "hopeless"]):
            return next((t for t in self._techniques if t.name == "Radical Acceptance"), None)
        return self._techniques[0] if self._techniques else None

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        """Generate DBT response."""
        if technique.name == "STOP Skill":
            response = "When you notice that intensity rising, let's practice STOP. First, can you freeze right where you are? Take a breath. Now observe - what sensations do you notice in your body? What thoughts are present?"
            follow_ups = ["What do you notice when you pause?", "How might you proceed mindfully?"]
        elif technique.name == "DEAR MAN":
            response = "It sounds like there's something important you need to communicate. Let's use DEAR MAN. First, describe the situation in just facts - what happened without any interpretation?"
            follow_ups = ["How did that make you feel?", "What specifically would you like to ask for?"]
        else:
            response = "This sounds like a place where radical acceptance might help. What is the reality that's hard to accept right now? Remember, acceptance doesn't mean approval - it means acknowledging what is."
            follow_ups = ["What would it feel like to stop fighting this reality?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion=f"Practice {technique.name} when you notice distress", notes=["Used DBT framework"],
        )


class ACTProvider(ModalityProvider):
    """Acceptance and Commitment Therapy implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.ACT, name="Acceptance and Commitment Therapy",
            description="Values-based therapy promoting psychological flexibility",
            core_principles=["Psychological flexibility", "Acceptance over avoidance", "Cognitive defusion",
                            "Present moment awareness", "Values-guided action", "Self-as-context"],
            key_techniques=["Values Clarification", "Cognitive Defusion", "Committed Action",
                           "The Observer Self", "Willingness", "Creative Hopelessness"],
            session_structure={"opening": ["Present moment check-in", "Values connection", "Agenda"],
                              "working": ["Experiential exercises", "Metaphor exploration", "Values work"],
                              "closing": ["Committed action planning", "Summary"]},
            contraindications=["Active psychosis", "Severe dissociation during exercises"],
            efficacy_evidence="Meta-analyses show moderate to large effects for anxiety, depression, chronic pain",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.ACT

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        """Initialize ACT techniques."""
        return [
            TechniqueProtocol(
                name="Values Clarification", modality=TherapyModality.ACT, category="values",
                description="Explore and clarify core personal values",
                rationale="Values provide direction for meaningful action",
                steps=[
                    TechniqueStep(1, "Explore life domains", "Consider relationships, work, health, growth", "Domains identified"),
                    TechniqueStep(2, "Identify what matters", "What's most important in each area?", "Values named"),
                    TechniqueStep(3, "Distinguish values from goals", "Values are directions, not destinations", "Understanding clarified"),
                    TechniqueStep(4, "Rate importance and alignment", "How important? How aligned is current life?", "Gaps identified"),
                    TechniqueStep(5, "Choose a value to focus on", "Which value calls to you most?", "Focus selected"),
                ],
                duration_minutes=25, materials_needed=["Values card sort", "Values worksheet"],
            ),
            TechniqueProtocol(
                name="Cognitive Defusion", modality=TherapyModality.ACT, category="cognitive",
                description="Create distance from unhelpful thoughts",
                rationale="We can have thoughts without being controlled by them",
                steps=[
                    TechniqueStep(1, "Notice the thought", "What thought is showing up?", "Thought identified"),
                    TechniqueStep(2, "Name the thought", "I notice I'm having the thought that...", "Distance created"),
                    TechniqueStep(3, "Thank your mind", "Thanks mind for that thought", "Relationship shifted"),
                    TechniqueStep(4, "Notice the thought is just words", "Repeat word until it loses meaning", "Defusion experienced"),
                    TechniqueStep(5, "Return to present", "What's here right now?", "Presence restored"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Committed Action", modality=TherapyModality.ACT, category="behavioral",
                description="Take values-aligned action despite discomfort",
                rationale="Living according to values is more fulfilling than comfort-seeking",
                steps=[
                    TechniqueStep(1, "Connect to the value", "What value is calling you to act?", "Value identified"),
                    TechniqueStep(2, "Identify a small step", "What's one small action aligned with this?", "Action identified"),
                    TechniqueStep(3, "Notice barriers", "What shows up that makes this hard?", "Barriers noticed"),
                    TechniqueStep(4, "Make room for discomfort", "Can you take the action AND have the discomfort?", "Willingness developed"),
                    TechniqueStep(5, "Commit to the action", "Are you willing to commit to this step?", "Commitment made"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        """Select appropriate ACT technique."""
        concern_lower = context.current_concern.lower()
        if any(word in concern_lower for word in ["meaning", "purpose", "direction", "value"]):
            return next((t for t in self._techniques if t.name == "Values Clarification"), None)
        if any(word in concern_lower for word in ["thought", "worry", "stuck", "can't stop"]):
            return next((t for t in self._techniques if t.name == "Cognitive Defusion"), None)
        if any(word in concern_lower for word in ["do", "action", "change", "want to"]):
            return next((t for t in self._techniques if t.name == "Committed Action"), None)
        return self._techniques[0] if self._techniques else None

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        """Generate ACT response."""
        if technique.name == "Values Clarification":
            response = "Let's explore what really matters to you. If you imagine your life as a journey, what direction do you want to be heading? What kind of person do you want to be in your relationships? Your work? Your personal growth?"
            follow_ups = ["Which of these areas feels most important right now?", "What would living this value look like in small, daily actions?"]
        elif technique.name == "Cognitive Defusion":
            response = "I notice that thought seems to have a strong grip. What if we tried something? Can you say 'I notice I'm having the thought that...' before stating it? Notice how that creates a bit of space between you and the thought."
            follow_ups = ["What happens when you see it as just a thought rather than truth?"]
        else:
            response = "So there's something you value, and something getting in the way. What if you could take a step toward what matters AND bring the discomfort along with you? What small step could you take this week that honors your values?"
            follow_ups = ["What shows up when you imagine taking that step?", "Are you willing to have that discomfort in service of what matters?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion=f"Practice noticing when {technique.name.lower()} applies in daily life",
            notes=["Used ACT framework"],
        )


class MIProvider(ModalityProvider):
    """Motivational Interviewing implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.MI, name="Motivational Interviewing",
            description="Collaborative conversation to strengthen motivation for change",
            core_principles=["Express empathy", "Develop discrepancy", "Roll with resistance",
                            "Support self-efficacy", "Evoke change talk"],
            key_techniques=["Reflective Listening", "Open Questions", "Affirming",
                           "Summarizing", "Change Talk Elicitation", "Decisional Balance"],
            session_structure={"opening": ["Build rapport", "Ask permission", "Set agenda collaboratively"],
                              "working": ["Explore ambivalence", "Elicit change talk", "Build motivation"],
                              "closing": ["Summarize", "Strengthen commitment", "Plan next steps"]},
            contraindications=["Client not ambivalent about change", "External mandate without any ambivalence"],
            efficacy_evidence="Strong evidence for substance use, health behaviors; moderate for mental health",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.MI

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        """Initialize MI techniques."""
        return [
            TechniqueProtocol(
                name="Reflective Listening", modality=TherapyModality.MI, category="engagement",
                description="Deep listening and reflection to demonstrate understanding",
                rationale="Reflection builds connection and helps clients hear themselves",
                steps=[
                    TechniqueStep(1, "Listen without interrupting", "Give full attention", "Client speaks freely"),
                    TechniqueStep(2, "Reflect content", "Repeat back what you heard", "Understanding confirmed"),
                    TechniqueStep(3, "Reflect feeling", "Acknowledge the emotion beneath", "Emotion validated"),
                    TechniqueStep(4, "Reflect meaning", "Connect to deeper values/needs", "Deeper understanding"),
                    TechniqueStep(5, "Double-sided reflection", "Acknowledge both sides of ambivalence", "Ambivalence explored"),
                ],
                duration_minutes=10, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Change Talk Elicitation", modality=TherapyModality.MI, category="engagement",
                description="Draw out client's own arguments for change",
                rationale="People are more motivated by their own reasons than external advice",
                steps=[
                    TechniqueStep(1, "Ask evocative questions", "Why might you want to make this change?", "Change talk emerges"),
                    TechniqueStep(2, "Explore importance", "How important is this to you? Why not lower?", "Importance clarified"),
                    TechniqueStep(3, "Explore confidence", "How confident are you? What would increase it?", "Confidence explored"),
                    TechniqueStep(4, "Query extremes", "What concerns you most about not changing?", "Concerns voiced"),
                    TechniqueStep(5, "Look back/forward", "How was life before? How might it be after?", "Vision developed"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        """Select appropriate MI technique."""
        concern_lower = context.current_concern.lower()
        if any(word in concern_lower for word in ["change", "want", "should", "but", "ambivalent"]):
            return next((t for t in self._techniques if t.name == "Change Talk Elicitation"), None)
        return next((t for t in self._techniques if t.name == "Reflective Listening"), None)

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        """Generate MI response."""
        if technique.name == "Change Talk Elicitation":
            response = "It sounds like part of you wants things to be different. I'm curious - if you did make this change, what would be different for you? What matters most about that possibility?"
            follow_ups = ["What gives you hope that change is possible?", "On a scale of 0-10, how important is this change to you?"]
        else:
            response = "So if I'm hearing you right, there's a part of you that wants change, and another part that finds it difficult. Both of those make sense. What would help you move forward despite the difficulty?"
            follow_ups = ["What's at stake if things stay the same?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion="Notice moments this week when you feel motivated toward change",
            notes=["Used MI framework"],
        )


class SFBTProvider(ModalityProvider):
    """Solution-Focused Brief Therapy implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.SFBT, name="Solution-Focused Brief Therapy",
            description="Future-oriented therapy focusing on solutions rather than problems",
            core_principles=["Focus on solutions not problems", "If it works, do more of it",
                            "Small changes lead to big changes", "Clients are the experts on their lives",
                            "The future is created and negotiable"],
            key_techniques=["Miracle Question", "Scaling Questions", "Exception Finding",
                           "Coping Questions", "Compliments"],
            session_structure={"opening": ["Best hopes", "Problem-free talk", "Goal setting"],
                              "working": ["Miracle question", "Exception finding", "Scaling"],
                              "closing": ["Compliments", "Task suggestion", "Next steps"]},
            contraindications=["Active psychosis", "Severe cognitive impairment",
                              "Client requires extensive trauma processing"],
            efficacy_evidence="Moderate evidence for depression, anxiety; strong for brief interventions",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.SFBT

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        return [
            TechniqueProtocol(
                name="Miracle Question", modality=TherapyModality.SFBT, category="future_orientation",
                description="Imagine a future where the problem is solved",
                rationale="Shifts focus from problems to solutions and desired outcomes",
                steps=[
                    TechniqueStep(1, "Set up the miracle question", "I'd like to ask you an unusual question...", "Client is curious"),
                    TechniqueStep(2, "Ask the miracle question", "Suppose tonight while you sleep, a miracle happens and the problem is solved. What would be different tomorrow?", "Client describes desired future"),
                    TechniqueStep(3, "Elaborate the vision", "What else would be different?", "Detailed vision emerges"),
                    TechniqueStep(4, "Identify small signs", "What small thing would you notice first?", "Observable changes identified"),
                    TechniqueStep(5, "Connect to current resources", "What parts of this are already happening, even a little?", "Exceptions found"),
                ],
                duration_minutes=20, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Scaling Questions", modality=TherapyModality.SFBT, category="assessment",
                description="Use scales to measure progress and set goals",
                rationale="Provides concrete measurement and highlights progress",
                steps=[
                    TechniqueStep(1, "Introduce the scale", "On a scale of 0-10, where 10 is your best hopes...", "Scale understood"),
                    TechniqueStep(2, "Rate current position", "Where would you say you are right now?", "Number given"),
                    TechniqueStep(3, "Explore current position", "What puts you at that number and not lower?", "Strengths identified"),
                    TechniqueStep(4, "Identify next step", "What would one point higher look like?", "Concrete goal set"),
                    TechniqueStep(5, "Build confidence", "What would help you move up one point?", "Action plan emerging"),
                ],
                duration_minutes=10, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Exception Finding", modality=TherapyModality.SFBT, category="resource_activation",
                description="Identify times when the problem is absent or less severe",
                rationale="Exceptions contain solutions - what works can be amplified",
                steps=[
                    TechniqueStep(1, "Ask about exceptions", "Tell me about a time recently when this problem wasn't happening", "Exception identified"),
                    TechniqueStep(2, "Detail the exception", "What was different about that time?", "Differences noted"),
                    TechniqueStep(3, "Identify contributing factors", "What did you do differently?", "Client actions identified"),
                    TechniqueStep(4, "Amplify the exception", "How could you do more of what worked?", "Plan developing"),
                ],
                duration_minutes=12, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        concern_lower = context.current_concern.lower()
        if any(word in concern_lower for word in ["future", "hope", "wish", "want", "dream"]):
            return next((t for t in self._techniques if t.name == "Miracle Question"), None)
        if any(word in concern_lower for word in ["progress", "better", "scale", "rate"]):
            return next((t for t in self._techniques if t.name == "Scaling Questions"), None)
        if any(word in concern_lower for word in ["sometimes", "except", "used to", "before"]):
            return next((t for t in self._techniques if t.name == "Exception Finding"), None)
        return self._techniques[0] if self._techniques else None

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        if technique.name == "Miracle Question":
            response = "I'd like to ask you something a bit unusual. Suppose tonight, while you're sleeping, a miracle happens and the problem that brought you here is completely solved. When you wake up tomorrow, what's the first small thing you'd notice that tells you something is different?"
            follow_ups = ["What else would be different?", "Who would be the first person to notice?"]
        elif technique.name == "Scaling Questions":
            response = "On a scale of 0 to 10, where 10 means things are exactly how you'd like them to be and 0 is the worst it's been, where would you say you are right now?"
            follow_ups = ["What puts you at that number and not lower?", "What would one point higher look like?"]
        else:
            response = "It sounds like there are times when this problem isn't as strong. Can you tell me about a recent time when things were even a little bit better? What was different about that time?"
            follow_ups = ["What were you doing differently?", "How could you do more of that?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion=f"Notice moments when {technique.name.lower()} applies in your daily life",
            notes=["Used SFBT framework"],
        )


class MindfulnessProvider(ModalityProvider):
    """Mindfulness-Based Therapy implementation."""

    def __init__(self) -> None:
        self._protocol = ModalityProtocol(
            modality=TherapyModality.MINDFULNESS, name="Mindfulness-Based Therapy",
            description="Present-moment awareness practices for emotional regulation and wellbeing",
            core_principles=["Non-judgmental awareness", "Present moment focus", "Acceptance of experience",
                            "Beginner's mind", "Letting go of attachment"],
            key_techniques=["Body Scan", "Breath Awareness", "Loving-Kindness",
                           "Progressive Relaxation", "Mindful Movement"],
            session_structure={"opening": ["Brief centering", "Check-in", "Intention setting"],
                              "working": ["Guided practice", "Inquiry", "Discussion"],
                              "closing": ["Brief practice", "Home practice suggestion", "Reflection"]},
            contraindications=["Acute trauma flashbacks", "Severe dissociation",
                              "Active psychosis with poor reality testing"],
            efficacy_evidence="Strong evidence for depression relapse prevention, anxiety, stress, chronic pain",
        )
        self._techniques = self._initialize_techniques()

    @property
    def modality(self) -> TherapyModality:
        return TherapyModality.MINDFULNESS

    @property
    def protocol(self) -> ModalityProtocol:
        return self._protocol

    def _initialize_techniques(self) -> list[TechniqueProtocol]:
        return [
            TechniqueProtocol(
                name="Body Scan", modality=TherapyModality.MINDFULNESS, category="awareness",
                description="Systematic attention to body sensations from head to toe",
                rationale="Develops interoceptive awareness and mind-body connection",
                steps=[
                    TechniqueStep(1, "Find a comfortable position", "Settle in, close eyes if comfortable", "Client settled"),
                    TechniqueStep(2, "Bring attention to feet", "Notice any sensations in your feet", "Awareness begins"),
                    TechniqueStep(3, "Slowly move upward", "Move attention through legs, torso, arms", "Body scanned"),
                    TechniqueStep(4, "Notice without judgment", "Whatever you find is okay", "Acceptance practiced"),
                    TechniqueStep(5, "Return to whole body", "Hold awareness of your entire body", "Integration complete"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Breath Awareness", modality=TherapyModality.MINDFULNESS, category="grounding",
                description="Focus attention on the breath as an anchor to the present",
                rationale="Breath is always available as an anchor to the present moment",
                steps=[
                    TechniqueStep(1, "Find the breath", "Notice where you feel the breath most clearly", "Anchor found"),
                    TechniqueStep(2, "Follow the breath", "Follow the natural rhythm of breathing", "Attention focused"),
                    TechniqueStep(3, "Notice wandering", "When mind wanders, gently note it", "Meta-awareness develops"),
                    TechniqueStep(4, "Return to breath", "Kindly bring attention back each time", "Practice sustained"),
                ],
                duration_minutes=8, materials_needed=[],
            ),
            TechniqueProtocol(
                name="Loving-Kindness", modality=TherapyModality.MINDFULNESS, category="compassion",
                description="Generate feelings of warmth and goodwill toward self and others",
                rationale="Cultivates self-compassion and reduces self-criticism",
                steps=[
                    TechniqueStep(1, "Begin with self", "May I be happy, may I be healthy, may I be safe", "Self-compassion practiced"),
                    TechniqueStep(2, "Extend to loved one", "Bring to mind someone you care about", "Warmth generated"),
                    TechniqueStep(3, "Extend to neutral person", "Someone you neither like nor dislike", "Circle expanding"),
                    TechniqueStep(4, "Extend to difficult person", "If comfortable, include someone challenging", "Compassion stretches"),
                    TechniqueStep(5, "Extend to all beings", "May all beings be happy and free from suffering", "Universal compassion"),
                ],
                duration_minutes=15, materials_needed=[],
            ),
        ]

    def get_techniques(self) -> list[TechniqueProtocol]:
        return self._techniques

    def select_intervention(self, context: InterventionContext) -> TechniqueProtocol | None:
        concern_lower = context.current_concern.lower()
        if any(word in concern_lower for word in ["body", "tension", "physical", "pain"]):
            return next((t for t in self._techniques if t.name == "Body Scan"), None)
        if any(word in concern_lower for word in ["self-critical", "harsh", "compassion", "kind"]):
            return next((t for t in self._techniques if t.name == "Loving-Kindness"), None)
        if any(word in concern_lower for word in ["anxious", "racing", "calm", "breathe"]):
            return next((t for t in self._techniques if t.name == "Breath Awareness"), None)
        return self._techniques[0] if self._techniques else None

    def generate_response(self, technique: TechniqueProtocol, user_input: str, context: InterventionContext) -> InterventionResult:
        if technique.name == "Body Scan":
            response = "Let's try a brief body scan. Find a comfortable position and, if you'd like, close your eyes. Starting at the top of your head, gently bring your attention downward, noticing any sensations you encounter - warmth, tension, tingling, or nothing at all. There's no right or wrong thing to feel."
            follow_ups = ["What did you notice?", "Where did you feel the most tension?"]
        elif technique.name == "Loving-Kindness":
            response = "Let's practice some self-compassion. Place a hand on your heart if that feels comfortable. Silently repeat to yourself: 'May I be happy. May I be healthy. May I be safe. May I live with ease.' Notice what comes up without judging it."
            follow_ups = ["How did that feel?", "What was it like to direct kindness toward yourself?"]
        else:
            response = "Let's take a moment together. Notice the natural rhythm of your breathing - the rise and fall. You don't need to change it. When your mind wanders, and it will, just gently bring attention back to the breath. Let's practice for a few moments."
            follow_ups = ["What did you notice during the practice?", "How do you feel now compared to before?"]
        return InterventionResult(
            success=True, technique_name=technique.name, response_generated=response, follow_up_prompts=follow_ups,
            homework_suggestion=f"Practice {technique.name.lower()} for 5-10 minutes daily",
            notes=["Used Mindfulness framework"],
        )


class ModalityRegistry:
    """Registry of available therapy modalities."""

    def __init__(self) -> None:
        self._providers: dict[TherapyModality, ModalityProvider] = {
            TherapyModality.CBT: CBTProvider(),
            TherapyModality.DBT: DBTProvider(),
            TherapyModality.ACT: ACTProvider(),
            TherapyModality.MI: MIProvider(),
            TherapyModality.SFBT: SFBTProvider(),
            TherapyModality.MINDFULNESS: MindfulnessProvider(),
        }
        logger.info("modality_registry_initialized", modalities=list(self._providers.keys()))

    def get_provider(self, modality: TherapyModality) -> ModalityProvider | None:
        """Get provider for a modality."""
        return self._providers.get(modality)

    def list_modalities(self) -> list[TherapyModality]:
        """List available modalities."""
        return list(self._providers.keys())

    def get_all_techniques(self) -> list[TechniqueProtocol]:
        """Get all techniques from all modalities."""
        techniques = []
        for provider in self._providers.values():
            techniques.extend(provider.get_techniques())
        return techniques

    def select_intervention_for_context(self, context: InterventionContext, preferred_modality: TherapyModality | None = None) -> tuple[ModalityProvider, TechniqueProtocol] | None:
        """Select best intervention across modalities."""
        if preferred_modality and preferred_modality in self._providers:
            provider = self._providers[preferred_modality]
            technique = provider.select_intervention(context)
            if technique:
                return provider, technique
        for modality, provider in self._providers.items():
            technique = provider.select_intervention(context)
            if technique:
                return provider, technique
        return None
