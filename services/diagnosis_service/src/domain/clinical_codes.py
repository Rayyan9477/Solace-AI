"""
Solace-AI Diagnosis Service - DSM-5-TR/ICD-11 Clinical Code Mapping.
Provides comprehensive clinical code lookup, validation, and crosswalk.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class ClinicalCodesSettings(BaseSettings):
    """Clinical codes configuration."""
    enable_icd11_mapping: bool = Field(default=True)
    enable_dsm5_mapping: bool = Field(default=True)
    include_deprecated_codes: bool = Field(default=False)
    strict_validation: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="CLINICAL_CODES_", env_file=".env", extra="ignore")


@dataclass
class ClinicalCode:
    """Represents a clinical diagnostic code."""
    code: str
    system: str
    name: str
    category: str
    description: str
    severity_specifiers: list[str] = field(default_factory=list)
    course_specifiers: list[str] = field(default_factory=list)
    related_codes: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    parent_code: str | None = None


@dataclass
class CodeLookupResult:
    """Result from code lookup."""
    lookup_id: UUID = field(default_factory=uuid4)
    found: bool = False
    code: ClinicalCode | None = None
    crosswalk: dict[str, str] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from code validation."""
    validation_id: UUID = field(default_factory=uuid4)
    valid: bool = False
    code: str = ""
    system: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ClinicalCodeMapper:
    """Maps and validates DSM-5-TR and ICD-11 clinical codes."""

    def __init__(self, settings: ClinicalCodesSettings | None = None) -> None:
        self._settings = settings or ClinicalCodesSettings()
        self._dsm5_codes = self._build_dsm5_codes()
        self._icd11_codes = self._build_icd11_codes()
        self._crosswalk = self._build_crosswalk()
        self._stats = {"lookups": 0, "validations": 0, "crosswalks": 0}

    def _build_dsm5_codes(self) -> dict[str, ClinicalCode]:
        """Build DSM-5-TR code database."""
        return {
            "F32": ClinicalCode(
                code="F32", system="DSM-5-TR", name="Major Depressive Disorder",
                category="Depressive Disorders",
                description="Single episode of major depressive disorder",
                severity_specifiers=["mild", "moderate", "severe", "with_psychotic_features"],
                course_specifiers=["in_partial_remission", "in_full_remission", "unspecified"],
                related_codes=["F32.0", "F32.1", "F32.2", "F32.3", "F32.4", "F32.5", "F33"],
                exclusions=["bipolar_disorder", "schizoaffective_disorder"],
            ),
            "F32.0": ClinicalCode(
                code="F32.0", system="DSM-5-TR", name="Major Depressive Disorder, Single Episode, Mild",
                category="Depressive Disorders",
                description="Single episode of MDD with mild severity",
                severity_specifiers=[], course_specifiers=[], parent_code="F32",
            ),
            "F32.1": ClinicalCode(
                code="F32.1", system="DSM-5-TR", name="Major Depressive Disorder, Single Episode, Moderate",
                category="Depressive Disorders",
                description="Single episode of MDD with moderate severity",
                severity_specifiers=[], course_specifiers=[], parent_code="F32",
            ),
            "F32.2": ClinicalCode(
                code="F32.2", system="DSM-5-TR", name="Major Depressive Disorder, Single Episode, Severe",
                category="Depressive Disorders",
                description="Single episode of MDD with severe severity without psychotic features",
                severity_specifiers=[], course_specifiers=[], parent_code="F32",
            ),
            "F33": ClinicalCode(
                code="F33", system="DSM-5-TR", name="Major Depressive Disorder, Recurrent",
                category="Depressive Disorders",
                description="Recurrent episodes of major depressive disorder",
                severity_specifiers=["mild", "moderate", "severe", "with_psychotic_features"],
                course_specifiers=["in_partial_remission", "in_full_remission"],
                related_codes=["F33.0", "F33.1", "F33.2", "F33.3", "F32"],
            ),
            "F34.1": ClinicalCode(
                code="F34.1", system="DSM-5-TR", name="Persistent Depressive Disorder",
                category="Depressive Disorders",
                description="Chronic depressive symptoms lasting at least 2 years",
                severity_specifiers=["mild", "moderate", "severe"],
                course_specifiers=["early_onset", "late_onset", "with_intermittent_major_depressive_episodes"],
            ),
            "F41.1": ClinicalCode(
                code="F41.1", system="DSM-5-TR", name="Generalized Anxiety Disorder",
                category="Anxiety Disorders",
                description="Excessive anxiety and worry occurring more days than not for at least 6 months",
                severity_specifiers=["mild", "moderate", "severe"],
                exclusions=["panic_disorder", "social_anxiety_disorder", "ocd"],
            ),
            "F41.0": ClinicalCode(
                code="F41.0", system="DSM-5-TR", name="Panic Disorder",
                category="Anxiety Disorders",
                description="Recurrent unexpected panic attacks with persistent concern",
                severity_specifiers=["mild", "moderate", "severe"],
                related_codes=["F40.00"],
            ),
            "F40.10": ClinicalCode(
                code="F40.10", system="DSM-5-TR", name="Social Anxiety Disorder",
                category="Anxiety Disorders",
                description="Marked fear or anxiety about social situations",
                severity_specifiers=["performance_only", "generalized"],
            ),
            "F43.10": ClinicalCode(
                code="F43.10", system="DSM-5-TR", name="Post-Traumatic Stress Disorder",
                category="Trauma and Stressor-Related Disorders",
                description="Development of symptoms following exposure to traumatic event",
                severity_specifiers=["with_dissociative_symptoms", "with_delayed_expression"],
                course_specifiers=["acute", "chronic"],
            ),
            "F43.0": ClinicalCode(
                code="F43.0", system="DSM-5-TR", name="Acute Stress Disorder",
                category="Trauma and Stressor-Related Disorders",
                description="Development of symptoms 3 days to 1 month after trauma",
            ),
            "F43.2": ClinicalCode(
                code="F43.2", system="DSM-5-TR", name="Adjustment Disorder",
                category="Trauma and Stressor-Related Disorders",
                description="Emotional or behavioral symptoms in response to stressor",
                severity_specifiers=["with_depressed_mood", "with_anxiety", "with_mixed_anxiety_and_depressed_mood"],
            ),
            "F42": ClinicalCode(
                code="F42", system="DSM-5-TR", name="Obsessive-Compulsive Disorder",
                category="Obsessive-Compulsive and Related Disorders",
                description="Presence of obsessions, compulsions, or both",
                severity_specifiers=["good_insight", "poor_insight", "absent_insight"],
            ),
        }

    def _build_icd11_codes(self) -> dict[str, ClinicalCode]:
        """Build ICD-11 code database."""
        return {
            "6A70": ClinicalCode(
                code="6A70", system="ICD-11", name="Single Episode Depressive Disorder",
                category="Mood Disorders",
                description="Single depressive episode meeting diagnostic criteria",
                severity_specifiers=["mild", "moderate", "severe"],
                related_codes=["6A71", "6A72"],
            ),
            "6A71": ClinicalCode(
                code="6A71", system="ICD-11", name="Recurrent Depressive Disorder",
                category="Mood Disorders",
                description="Two or more depressive episodes",
                severity_specifiers=["mild", "moderate", "severe", "with_psychotic_symptoms"],
            ),
            "6A72": ClinicalCode(
                code="6A72", system="ICD-11", name="Dysthymic Disorder",
                category="Mood Disorders",
                description="Persistent depressive symptoms for at least 2 years",
            ),
            "6B00": ClinicalCode(
                code="6B00", system="ICD-11", name="Generalized Anxiety Disorder",
                category="Anxiety or Fear-Related Disorders",
                description="Excessive apprehension and worry about everyday events",
            ),
            "6B01": ClinicalCode(
                code="6B01", system="ICD-11", name="Panic Disorder",
                category="Anxiety or Fear-Related Disorders",
                description="Recurrent unexpected panic attacks",
            ),
            "6B04": ClinicalCode(
                code="6B04", system="ICD-11", name="Social Anxiety Disorder",
                category="Anxiety or Fear-Related Disorders",
                description="Marked fear or anxiety about social situations",
            ),
            "6B40": ClinicalCode(
                code="6B40", system="ICD-11", name="Post-Traumatic Stress Disorder",
                category="Disorders Specifically Associated with Stress",
                description="Development of characteristic symptoms following traumatic event",
            ),
            "6B41": ClinicalCode(
                code="6B41", system="ICD-11", name="Complex Post-Traumatic Stress Disorder",
                category="Disorders Specifically Associated with Stress",
                description="PTSD with additional disturbances in self-organization",
            ),
            "6B43": ClinicalCode(
                code="6B43", system="ICD-11", name="Adjustment Disorder",
                category="Disorders Specifically Associated with Stress",
                description="Maladaptive reaction to identifiable psychosocial stressor",
            ),
            "6B20": ClinicalCode(
                code="6B20", system="ICD-11", name="Obsessive-Compulsive Disorder",
                category="Obsessive-Compulsive or Related Disorders",
                description="Persistent obsessions or compulsions or both",
            ),
        }

    def _build_crosswalk(self) -> dict[str, dict[str, str]]:
        """Build DSM-5-TR to ICD-11 crosswalk."""
        return {
            "F32": {"icd11": "6A70", "notes": "Single episode mapping"},
            "F32.0": {"icd11": "6A70", "notes": "Mild single episode"},
            "F32.1": {"icd11": "6A70", "notes": "Moderate single episode"},
            "F32.2": {"icd11": "6A70", "notes": "Severe single episode"},
            "F33": {"icd11": "6A71", "notes": "Recurrent episode mapping"},
            "F34.1": {"icd11": "6A72", "notes": "Persistent/Dysthymic mapping"},
            "F41.1": {"icd11": "6B00", "notes": "GAD direct mapping"},
            "F41.0": {"icd11": "6B01", "notes": "Panic disorder direct mapping"},
            "F40.10": {"icd11": "6B04", "notes": "Social anxiety direct mapping"},
            "F43.10": {"icd11": "6B40", "notes": "PTSD direct mapping"},
            "F43.0": {"icd11": "6B40", "notes": "Acute stress maps to PTSD category"},
            "F43.2": {"icd11": "6B43", "notes": "Adjustment disorder direct mapping"},
            "F42": {"icd11": "6B20", "notes": "OCD direct mapping"},
        }

    def lookup(self, code: str) -> CodeLookupResult:
        """Look up a clinical code."""
        self._stats["lookups"] += 1
        result = CodeLookupResult()
        code_upper = code.upper()
        if code_upper in self._dsm5_codes:
            result.found = True
            result.code = self._dsm5_codes[code_upper]
            crosswalk = self._crosswalk.get(code_upper)
            if crosswalk:
                result.crosswalk = {"icd11": crosswalk["icd11"]}
        elif code_upper in self._icd11_codes:
            result.found = True
            result.code = self._icd11_codes[code_upper]
            for dsm_code, mapping in self._crosswalk.items():
                if mapping["icd11"] == code_upper:
                    result.crosswalk = {"dsm5": dsm_code}
                    break
        else:
            result.suggestions = self._find_similar_codes(code_upper)
        logger.debug("code_lookup", code=code, found=result.found)
        return result

    def _find_similar_codes(self, code: str) -> list[str]:
        """Find similar codes for suggestions."""
        suggestions: list[str] = []
        prefix = code[:2] if len(code) >= 2 else code
        for dsm_code in self._dsm5_codes:
            if dsm_code.startswith(prefix):
                suggestions.append(dsm_code)
        for icd_code in self._icd11_codes:
            if icd_code.startswith(prefix[0]) or code[0].isdigit() and icd_code[0].isdigit():
                suggestions.append(icd_code)
        return suggestions[:5]

    def validate(self, code: str, system: str | None = None) -> ValidationResult:
        """Validate a clinical code."""
        self._stats["validations"] += 1
        result = ValidationResult(code=code)
        if not code:
            result.errors.append("Code cannot be empty")
            return result
        code_upper = code.upper()
        if system:
            result.system = system
            if system.upper() in ["DSM-5", "DSM5", "DSM-5-TR", "DSM5TR"]:
                if code_upper in self._dsm5_codes:
                    result.valid = True
                else:
                    result.errors.append(f"Code {code} not found in DSM-5-TR")
            elif system.upper() in ["ICD-11", "ICD11"]:
                if code_upper in self._icd11_codes:
                    result.valid = True
                else:
                    result.errors.append(f"Code {code} not found in ICD-11")
            else:
                result.warnings.append(f"Unknown system: {system}")
        else:
            if code_upper in self._dsm5_codes:
                result.valid = True
                result.system = "DSM-5-TR"
            elif code_upper in self._icd11_codes:
                result.valid = True
                result.system = "ICD-11"
            else:
                result.errors.append(f"Code {code} not found in any supported system")
        if result.valid and self._settings.strict_validation:
            self._apply_strict_validation(code_upper, result)
        return result

    def _apply_strict_validation(self, code: str, result: ValidationResult) -> None:
        """Apply strict validation rules."""
        clinical_code = self._dsm5_codes.get(code) or self._icd11_codes.get(code)
        if clinical_code and clinical_code.parent_code:
            result.warnings.append(f"Consider using parent code {clinical_code.parent_code} for billing")

    def crosswalk_code(self, code: str, target_system: str) -> CodeLookupResult:
        """Convert code between DSM-5-TR and ICD-11."""
        self._stats["crosswalks"] += 1
        result = CodeLookupResult()
        code_upper = code.upper()
        target_upper = target_system.upper()
        if target_upper in ["ICD-11", "ICD11"]:
            if code_upper in self._crosswalk:
                icd_code = self._crosswalk[code_upper]["icd11"]
                if icd_code in self._icd11_codes:
                    result.found = True
                    result.code = self._icd11_codes[icd_code]
                    result.crosswalk = {"source": code_upper, "target": icd_code}
        elif target_upper in ["DSM-5", "DSM5", "DSM-5-TR", "DSM5TR"]:
            for dsm_code, mapping in self._crosswalk.items():
                if mapping["icd11"] == code_upper:
                    if dsm_code in self._dsm5_codes:
                        result.found = True
                        result.code = self._dsm5_codes[dsm_code]
                        result.crosswalk = {"source": code_upper, "target": dsm_code}
                    break
        if not result.found:
            result.suggestions = self._find_similar_codes(code_upper)
        logger.debug("code_crosswalk", source=code, target_system=target_system, found=result.found)
        return result

    def get_codes_by_category(self, category: str, system: str | None = None) -> list[ClinicalCode]:
        """Get all codes in a category."""
        codes: list[ClinicalCode] = []
        category_lower = category.lower()
        if system is None or system.upper() in ["DSM-5", "DSM5", "DSM-5-TR"]:
            for code in self._dsm5_codes.values():
                if category_lower in code.category.lower():
                    codes.append(code)
        if system is None or system.upper() in ["ICD-11", "ICD11"]:
            for code in self._icd11_codes.values():
                if category_lower in code.category.lower():
                    codes.append(code)
        return codes

    def get_severity_specifiers(self, code: str) -> list[str]:
        """Get valid severity specifiers for a code."""
        code_upper = code.upper()
        clinical_code = self._dsm5_codes.get(code_upper) or self._icd11_codes.get(code_upper)
        if clinical_code:
            return clinical_code.severity_specifiers
        return []

    def get_related_codes(self, code: str) -> list[str]:
        """Get related codes for a given code."""
        code_upper = code.upper()
        clinical_code = self._dsm5_codes.get(code_upper) or self._icd11_codes.get(code_upper)
        if clinical_code:
            return clinical_code.related_codes
        return []

    def get_statistics(self) -> dict[str, int]:
        """Get code mapper statistics."""
        return self._stats.copy()
