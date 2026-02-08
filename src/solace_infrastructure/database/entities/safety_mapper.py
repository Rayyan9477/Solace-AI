"""
Safety entity mapper - converts between domain entities and SQLAlchemy entities.

Handles field name mismatches (e.g., assessment_id ↔ id, plan_id ↔ id)
between the safety service domain layer and the centralized database schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .safety_entities import (
    SafetyAssessment as SafetyAssessmentDB,
    SafetyPlan as SafetyPlanDB,
)


def domain_assessment_to_db(domain: dict[str, Any]) -> dict[str, Any]:
    """Convert a domain SafetyAssessment dict to DB column values.

    Maps domain field names to SQLAlchemy column names.
    """
    db_values: dict[str, Any] = {}

    # Map assessment_id → id
    if "assessment_id" in domain:
        db_values["id"] = domain["assessment_id"]

    # Direct mappings (same field names)
    direct_fields = [
        "user_id", "session_id", "assessment_type", "content_assessed",
        "risk_score", "crisis_level", "is_safe", "risk_factors",
        "protective_factors", "trigger_indicators", "detection_layers_triggered",
        "recommended_action", "requires_escalation", "requires_human_review",
        "detection_time_ms", "context", "assessment_notes", "reviewed_at",
        "reviewed_by", "review_notes", "assessment_method",
    ]
    for field in direct_fields:
        if field in domain:
            db_values[field] = domain[field]

    # Map risk_level from crisis_level or explicit
    if "risk_level" in domain:
        db_values["risk_level"] = domain["risk_level"]
    elif "crisis_level" in domain and "risk_level" not in db_values:
        db_values["risk_level"] = domain["crisis_level"]

    # Map created_at → assessed_at
    if "created_at" in domain and "assessed_at" not in db_values:
        db_values["assessed_at"] = domain["created_at"]

    return db_values


def db_assessment_to_domain(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict to domain SafetyAssessment field names.

    Maps SQLAlchemy column names back to domain field names.
    """
    domain: dict[str, Any] = {}

    # Map id → assessment_id
    if "id" in row:
        domain["assessment_id"] = row["id"]

    # Direct mappings
    direct_fields = [
        "user_id", "session_id", "assessment_type", "content_assessed",
        "risk_score", "crisis_level", "is_safe", "risk_factors",
        "protective_factors", "trigger_indicators", "detection_layers_triggered",
        "recommended_action", "requires_escalation", "requires_human_review",
        "detection_time_ms", "context", "review_notes", "reviewed_at",
        "reviewed_by",
    ]
    for field in direct_fields:
        if field in row:
            domain[field] = row[field]

    # Map assessed_at → created_at
    if "assessed_at" in row:
        domain["created_at"] = row["assessed_at"]

    return domain


def domain_plan_to_db(domain: dict[str, Any]) -> dict[str, Any]:
    """Convert a domain SafetyPlan dict to DB column values."""
    db_values: dict[str, Any] = {}

    # Map plan_id → id
    if "plan_id" in domain:
        db_values["id"] = domain["plan_id"]

    direct_fields = [
        "user_id", "status", "assessment_id", "warning_signs",
        "coping_strategies", "emergency_contacts", "safe_environment_actions",
        "reasons_to_live", "professional_resources", "expires_at",
        "last_reviewed_at", "next_review_due",
    ]
    for field in direct_fields:
        if field in domain:
            db_values[field] = domain[field]

    # Map clinician_notes ↔ clinician_notes (was plan_notes in old schema)
    if "clinician_notes" in domain:
        db_values["clinician_notes"] = domain["clinician_notes"]

    # Map last_reviewed_by
    if "last_reviewed_by" in domain:
        db_values["last_reviewed_by"] = domain["last_reviewed_by"]

    # Map metadata
    if "metadata" in domain:
        db_values["plan_metadata"] = domain["metadata"]

    return db_values


def db_plan_to_domain(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict to domain SafetyPlan field names."""
    domain: dict[str, Any] = {}

    # Map id → plan_id
    if "id" in row:
        domain["plan_id"] = row["id"]

    direct_fields = [
        "user_id", "status", "version", "warning_signs",
        "coping_strategies", "emergency_contacts", "safe_environment_actions",
        "reasons_to_live", "professional_resources", "clinician_notes",
        "last_reviewed_at", "expires_at", "next_review_due",
    ]
    for field in direct_fields:
        if field in row:
            domain[field] = row[field]

    # Map last_reviewed_by
    if "last_reviewed_by" in row:
        domain["last_reviewed_by"] = row["last_reviewed_by"]

    # Map plan_metadata → metadata
    if "plan_metadata" in row:
        domain["metadata"] = row["plan_metadata"]

    # Map timestamps
    if "created_at" in row:
        domain["created_at"] = row["created_at"]
    if "updated_at" in row:
        domain["updated_at"] = row["updated_at"]

    return domain
