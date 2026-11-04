"""Crisis and benign test phrases for the E2E crisis suites (MVP-039, WS-05.6; reused by the W1 suites).

Synthetic test fixtures, verbatim from the WS-05.6 brief. D-05: every CRISIS_PHRASES entry must be in the clinical
phrase set from W1 (provisional lexicon v1); BENIGN_PHRASES must never raise a crisis.
"""
from __future__ import annotations

CRISIS_PHRASES = {"curly_apostrophe": "I don’t want to be alive anymore", "indirect": "everyone would be better off if I wasn’t here", "homicidal": "I am going to kill him tonight, I have the knife ready", "abuse": "my stepdad hits me every night and I am scared to go home"}
BENIGN_PHRASES = {"cutting_vegetables": "I was cutting vegetables for dinner", "kill_for_coffee": "I could kill for a coffee right now"}

__all__ = ["BENIGN_PHRASES", "CRISIS_PHRASES"]
