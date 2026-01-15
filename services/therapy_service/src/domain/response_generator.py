"""
Solace-AI Therapy Service - Response Generation Helpers.
Helper functions for generating therapeutic responses, summaries, and recommendations.
"""
from __future__ import annotations
from typing import Any

from ..schemas import SessionPhase, RiskLevel, TechniqueDTO, TreatmentPlanDTO, SessionSummaryDTO


class ResponseGenerator:
    """Helper class for generating therapeutic responses and summaries."""

    @staticmethod
    def generate_therapeutic_response(
        session: Any,
        message: str,
        technique: TechniqueDTO | None,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Generate therapeutic response with empathy and technique integration."""
        phase_map = {
            SessionPhase.PRE_SESSION: "Let's prepare to begin our session.",
            SessionPhase.OPENING: "Thank you for sharing that with me. How are you feeling today?",
            SessionPhase.WORKING: "I appreciate you opening up about this. Let's explore that together.",
            SessionPhase.CLOSING: "That's a valuable insight. Let's think about how you can use this going forward.",
            SessionPhase.POST_SESSION: "Thank you for our session today. Take care until we meet again.",
        }
        response = phase_map.get(session.current_phase, "I'm here to support you.")
        if technique:
            response += f"\n\nI'd like to try a technique called '{technique.name}' with you. {technique.description}"
        if session.current_phase == SessionPhase.CLOSING and session.insights_gained:
            response += f"\n\nToday we covered: {', '.join(session.insights_gained[:2])}."
        return response

    @staticmethod
    def generate_initial_message(session_number: int, treatment_plan: TreatmentPlanDTO) -> str:
        """Generate initial session greeting."""
        if session_number == 1:
            return f"Welcome to your first therapy session. I'm here to support you with {treatment_plan.primary_diagnosis}. This is a safe space where we can work together on your well-being. How are you feeling today?"
        return f"Welcome back to session {session_number}. I'm glad you're here. How have things been since we last spoke?"

    @staticmethod
    def generate_suggested_agenda(treatment_plan: TreatmentPlanDTO, session_number: int) -> list[str]:
        """Generate suggested session agenda."""
        agenda = ["Check in on current mood and wellbeing"]
        if session_number > 1:
            agenda.append("Review homework from last session")
        phase_items = {1: "Build understanding of your experience", 2: "Practice core therapeutic skills"}
        agenda.append(phase_items.get(treatment_plan.current_phase, "Consolidate progress and skills"))
        agenda.append("Identify today's focus area")
        return agenda

    @staticmethod
    def generate_session_summary(session: Any, duration_minutes: int) -> SessionSummaryDTO:
        """Generate comprehensive session summary."""
        return SessionSummaryDTO(
            session_id=session.session_id,
            user_id=session.user_id,
            session_number=session.session_number,
            duration_minutes=duration_minutes,
            techniques_used=session.techniques_used,
            skills_practiced=session.skills_practiced,
            insights_gained=session.insights_gained,
            homework_assigned=session.homework_assigned,
            session_rating=session.session_rating,
            summary_text=ResponseGenerator.generate_summary_text(session),
            next_session_focus=ResponseGenerator.generate_next_focus(session),
        )

    @staticmethod
    def generate_summary_text(session: Any) -> str:
        """Generate natural language summary."""
        parts = [f"Session {session.session_number} covered {len(session.topics_covered)} key areas."]
        if session.techniques_used:
            parts.append(f"We worked with {', '.join([t.name for t in session.techniques_used[:2]])}.")
        if session.skills_practiced:
            parts.append(f"You practiced {len(session.skills_practiced)} skills.")
        if session.insights_gained:
            parts.append("You gained important insights about your experience.")
        return " ".join(parts)

    @staticmethod
    def generate_next_focus(session: Any) -> str:
        """Generate next session focus suggestion."""
        if len(session.skills_practiced) > 0:
            return f"Continue practicing {session.skills_practiced[0]} and build on today's progress."
        return "Continue building on the skills and insights from today's session."

    @staticmethod
    def generate_recommendations(session: Any) -> list[str]:
        """Generate post-session recommendations."""
        recs = []
        if session.current_risk != RiskLevel.NONE:
            recs.append("Maintain awareness of safety resources")
        if session.homework_assigned:
            recs.append("Complete assigned homework before next session")
        if session.skills_practiced:
            recs.append(f"Practice {session.skills_practiced[0]} in daily life")
        recs.extend(["Monitor mood and note any significant changes", "Schedule next session within one week"])
        return recs

    @staticmethod
    def generate_next_steps(session: Any) -> list[str]:
        """Generate immediate next steps for user."""
        phase_steps = {
            SessionPhase.OPENING: ["Share more about what brings you here today"],
            SessionPhase.WORKING: ["Continue exploring your thoughts and feelings"],
            SessionPhase.CLOSING: ["Reflect on today's insights"],
            SessionPhase.POST_SESSION: ["Practice techniques between sessions"],
        }
        return phase_steps.get(session.current_phase, ["Continue our work together"])

    @staticmethod
    def generate_crisis_response(alerts: list[str]) -> str:
        """Generate crisis intervention response."""
        return (
            "I'm hearing that you're going through something very serious right now. Your safety is the top priority. "
            "I want to make sure you have immediate support.\n\n"
            "Please reach out to a crisis counselor:\n• National Suicide Prevention Lifeline: 988\n"
            "• Crisis Text Line: Text HOME to 741741\n• Emergency Services: 911\n\n"
            "These services are available 24/7 and trained professionals can help you right now. "
            "Would you like to talk about what's happening?"
        )
