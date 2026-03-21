"""
Solace-AI Therapy Service - False-Positive Crisis Detection Unit Tests.
Verify that the crisis detection regex patterns do not trigger on benign
clinical language while still catching genuine harm/danger phrases.
"""
from __future__ import annotations

import re
import pytest


class TestFalsePositiveCrisis:
    """Test that crisis regex patterns avoid false positives on therapy-related language."""

    # ---- Harm pattern tests ----

    def test_harm_my_progress_not_crisis(self) -> None:
        """'harm my progress' should NOT trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert not re.search(pattern, "I don't want to harm my progress")

    def test_harm_myself_is_crisis(self) -> None:
        """'harm myself' should trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert re.search(pattern, "I want to harm myself")

    def test_harm_herself_is_crisis(self) -> None:
        """'harm herself' should trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert re.search(pattern, "She wants to harm herself")

    def test_harm_himself_is_crisis(self) -> None:
        """'harm himself' should trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert re.search(pattern, "He wants to harm himself")

    def test_harm_themselves_is_crisis(self) -> None:
        """'harm themselves' should trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert re.search(pattern, "They want to harm themselves")

    def test_harm_me_is_crisis(self) -> None:
        """'harm me' should trigger the harm-to-self pattern."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert re.search(pattern, "Someone is trying to harm me")

    def test_harmful_thoughts_not_crisis(self) -> None:
        """'harmful thoughts' should NOT trigger (no pronoun after 'harm')."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert not re.search(pattern, "I have harmful thoughts sometimes")

    def test_harm_reduction_not_crisis(self) -> None:
        """'harm reduction' should NOT trigger."""
        pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        assert not re.search(pattern, "We discussed harm reduction strategies")

    # ---- Danger pattern tests ----

    def test_dangerous_thoughts_in_cbt_context_not_crisis(self) -> None:
        """'dangerous thoughts' in a CBT context should NOT trigger the danger pattern."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert not re.search(pattern, "CBT helps manage dangerous thoughts")

    def test_in_danger_is_crisis(self) -> None:
        """'in danger' should trigger the danger pattern."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert re.search(pattern, "I feel like I'm in danger")

    def test_dangerous_to_self_is_crisis(self) -> None:
        """'dangerous to' should trigger the danger pattern."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert re.search(pattern, "I feel dangerous to myself")

    def test_dangerous_to_others_is_crisis(self) -> None:
        """'dangerous to others' should trigger the danger pattern."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert re.search(pattern, "I feel dangerous to others")

    def test_dangerous_situation_not_crisis(self) -> None:
        """'dangerous situation' without 'to' should NOT trigger."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert not re.search(pattern, "I was in a dangerous situation last year")

    def test_danger_zone_not_crisis(self) -> None:
        """'danger zone' should NOT trigger (not 'in danger')."""
        pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        assert not re.search(pattern, "I feel like I'm entering the danger zone of burnout")

    # ---- Combined patterns matching service._check_safety ----

    def test_combined_pattern_harm_myself(self) -> None:
        """Combined regex from _check_safety should match 'harm myself'."""
        message = "I want to harm myself"
        message_lower = message.lower()
        harm_match = re.search(r'\bharm\s+(myself|herself|himself|themselves|me)\b', message_lower)
        danger_match = re.search(r'\b(in\s+danger|dangerous\s+to)\b', message_lower)
        assert harm_match or danger_match

    def test_combined_pattern_therapy_language_no_trigger(self) -> None:
        """Clinical/therapy language should NOT trigger either harm or danger pattern."""
        benign_messages = [
            "I don't want to harm my progress in therapy",
            "CBT helps manage dangerous thoughts",
            "We practiced harm reduction techniques",
            "The therapist said my harmful beliefs can be challenged",
            "I learned to recognize cognitive distortions that seem dangerous",
            "Exposure therapy felt dangerous but was helpful",
        ]
        harm_pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        danger_pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        for msg in benign_messages:
            msg_lower = msg.lower()
            harm_match = re.search(harm_pattern, msg_lower)
            danger_match = re.search(danger_pattern, msg_lower)
            assert not (harm_match or danger_match), (
                f"False positive crisis trigger on benign message: '{msg}'"
            )

    def test_combined_pattern_genuine_crisis_triggers(self) -> None:
        """Genuine crisis language should trigger at least one pattern."""
        crisis_messages = [
            "I want to harm myself tonight",
            "She is trying to harm herself",
            "I feel like I'm in danger",
            "I feel dangerous to myself and others",
            "Someone is trying to harm me",
        ]
        harm_pattern = r'\bharm\s+(myself|herself|himself|themselves|me)\b'
        danger_pattern = r'\b(in\s+danger|dangerous\s+to)\b'
        for msg in crisis_messages:
            msg_lower = msg.lower()
            harm_match = re.search(harm_pattern, msg_lower)
            danger_match = re.search(danger_pattern, msg_lower)
            assert harm_match or danger_match, (
                f"Genuine crisis message not detected: '{msg}'"
            )
