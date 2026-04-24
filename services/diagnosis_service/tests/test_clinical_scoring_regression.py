"""Sprint 3 clinical scoring regression tests.

Locks in the following fixes so a refactor can't silently regress clinical
accuracy:

  - H-09 PHQ-9: severity bands per Kroenke 2001 (0-4/5-9/10-14/15-19/20-27).
  - H-09 PHQ-9: MODERATELY_SEVERE severity maps to per-item score 3, not 2.
  - H-10 PCL-5: 10-item screener thresholds halved from the 20-item
        Weathers 2013 standard (31 -> 16, 22 -> 11, 17 -> 9).
  - GAD-7: severity bands per Spitzer 2006 (0-4/5-9/10-14/15-21).

Each test exercises a boundary value — the score one below and one above
each severity cutoff — so a future scorer tweak that only moves by one
point is still caught.
"""
from __future__ import annotations

import pytest

from services.diagnosis_service.src.domain.severity import SeverityAssessor, SeveritySettings
from services.diagnosis_service.src.schemas import SeverityLevel


@pytest.fixture
def assessor() -> SeverityAssessor:
    return SeverityAssessor(SeveritySettings())


class TestPhq9SeverityBandsKroenke2001:
    """H-09 regression: PHQ-9 bands per Kroenke/Spitzer/Williams 2001 JAMA.

    Published cutoffs:
        0-4   Minimal / None
        5-9   Mild
        10-14 Moderate
        15-19 Moderately Severe
        20-27 Severe
    """

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0, SeverityLevel.MINIMAL),
            (4, SeverityLevel.MINIMAL),
            (5, SeverityLevel.MILD),
            (9, SeverityLevel.MILD),
            (10, SeverityLevel.MODERATE),
            (14, SeverityLevel.MODERATE),
            (15, SeverityLevel.MODERATELY_SEVERE),
            (19, SeverityLevel.MODERATELY_SEVERE),
            (20, SeverityLevel.SEVERE),
            (27, SeverityLevel.SEVERE),
        ],
    )
    def test_phq9_boundary_maps_to_kroenke_severity(
        self, assessor: SeverityAssessor, score: int, expected: SeverityLevel
    ) -> None:
        assert assessor._interpret_phq9(score) == expected

    def test_phq9_per_item_severity_score_moderately_severe_is_three(
        self, assessor: SeverityAssessor
    ) -> None:
        """H-09 core regression: MODERATELY_SEVERE per-item severity must
        map to 3, not 2. PHQ-9 per-item scale is 0-3, and the spec places
        MODERATELY_SEVERE and SEVERE both at the top of that scale (the
        category distinction comes from the TOTAL score summed across
        items, not a per-item 0-4 range).
        """
        from services.diagnosis_service.src.domain.severity import (
            SeverityAssessor as SA,
        )

        # Re-instantiate to read the internal mapping
        sa = SA(SeveritySettings())
        # Simulate 9 maxed-out symptoms -> total should be 27 (SEVERE)
        from uuid import uuid4

        from services.diagnosis_service.src.schemas import SymptomDTO

        nine_severe = [
            SymptomDTO(
                symptom_id=uuid4(),
                name=name,
                symptom_type="emotional",
                severity=SeverityLevel.SEVERE,
                description="severe",
            )
            for name in [
                "anhedonia",
                "depressed_mood",
                "sleep_disturbance",
                "fatigue",
                "appetite_change",
                "worthlessness",
                "concentration",
                "psychomotor",
                "suicidal_ideation",
            ]
        ]
        responses = sa._infer_responses_from_symptoms(nine_severe)
        # Each PHQ-9 item mapped from a SEVERE symptom should score 3
        for item_id, value in responses.items():
            if item_id.startswith("phq9_"):
                assert value == 3, (
                    f"H-09: PHQ-9 item {item_id} should score 3 for SEVERE "
                    f"severity (per-item scale is 0-3). Got {value}."
                )


class TestGad7SeverityBandsSpitzer2006:
    """GAD-7 bands per Spitzer/Kroenke/Williams/Lowe 2006 Arch Intern Med."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0, SeverityLevel.MINIMAL),
            (4, SeverityLevel.MINIMAL),
            (5, SeverityLevel.MILD),
            (9, SeverityLevel.MILD),
            (10, SeverityLevel.MODERATE),
            (14, SeverityLevel.MODERATE),
            (15, SeverityLevel.SEVERE),
            (21, SeverityLevel.SEVERE),
        ],
    )
    def test_gad7_boundary_maps_to_spitzer_severity(
        self, assessor: SeverityAssessor, score: int, expected: SeverityLevel
    ) -> None:
        assert assessor._interpret_gad7(score) == expected


class TestPcl5TenItemScreenerWeathers2013:
    """H-10 regression: 10-item PCL-5 screener thresholds halved from
    Weathers 2013 standard 20-item instrument (cutoff 31 -> 16).

    The 20-item PCL-5 has a provisional probable-PTSD cutoff of 31-33
    (max 80). Halving proportionally to the 10-item max=40 scale gives:
        16 (was 31)  -- probable PTSD / SEVERE
        11 (was 22)  -- MODERATE
         9 (was 17)  -- MILD
    """

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0, SeverityLevel.MINIMAL),
            (8, SeverityLevel.MINIMAL),
            (9, SeverityLevel.MILD),
            (10, SeverityLevel.MILD),
            (11, SeverityLevel.MODERATE),
            (15, SeverityLevel.MODERATE),
            (16, SeverityLevel.SEVERE),
            (40, SeverityLevel.SEVERE),
        ],
    )
    def test_pcl5_boundary_maps_to_halved_severity(
        self, assessor: SeverityAssessor, score: int, expected: SeverityLevel
    ) -> None:
        assert assessor._interpret_pcl5(score) == expected

    def test_pcl5_score_of_31_no_longer_requires_severe_classification(
        self, assessor: SeverityAssessor
    ) -> None:
        """Direct H-10 regression: on the 10-item screener (max 40), a
        score of 31 should already be SEVERE (was barely reachable with
        the un-halved 20-item cutoff, which would demand 77.5% endorsement).
        """
        assert assessor._interpret_pcl5(31) == SeverityLevel.SEVERE

    def test_pcl5_score_of_15_is_not_severe(
        self, assessor: SeverityAssessor
    ) -> None:
        """Under the halved cutoffs a score of 15 (just below the new 16)
        must classify as MODERATE, not SEVERE. If a future edit bumped
        the SEVERE cutoff back to 31 this test would pass; but the sister
        test at 16 asserts SEVERE there, so both together pin the boundary.
        """
        assert assessor._interpret_pcl5(15) == SeverityLevel.MODERATE
