"""Initial schema - all tables for Solace-AI platform.

Creates every table required by the centralized ORM entities plus reference
tables and cross-service support tables.

Tables created (in dependency order):
  1. users                          (anchor table)
  2. user_preferences               (FK -> users)
  3. consent_records                 (FK -> users)
  4. clinician_patient_assignments   (FK -> users x2)
  5. audit_logs                      (standalone)
  6. system_configurations           (standalone)
  7. safety_resources                (standalone reference)
  8. clinical_references             (standalone reference)
  9. therapy_techniques              (standalone reference)
 10. safety_events                   (FK -> users)
 11. safety_assessments              (FK -> users)
 12. safety_plans                    (FK -> users, safety_assessments)
 13. risk_factors                    (FK -> users, safety_assessments)
 14. contraindication_checks         (FK -> users)
 15. contraindication_rules          (standalone)
 16. rule_alternatives               (FK -> contraindication_rules)
 17. rule_prerequisites              (FK -> contraindication_rules)
 18. treatment_plans                 (FK -> users)
 19. therapy_sessions                (FK -> users, treatment_plans)
 20. therapy_interventions           (FK -> users, therapy_sessions)
 21. homework_assignments            (FK -> users, therapy_sessions)
 22. diagnosis_sessions              (FK -> users)
 23. diagnosis_symptoms              (FK -> users, diagnosis_sessions)
 24. diagnosis_hypotheses            (FK -> users, diagnosis_sessions)
 25. diagnosis_records               (FK -> users, diagnosis_sessions)
 26. memory_records                  (FK -> users)
 27. memory_user_profiles            (FK -> users)
 28. session_summaries               (FK -> users)
 29. therapeutic_events              (FK -> users)
 30. personality_profiles            (FK -> users)
 31. trait_assessments               (FK -> users, personality_profiles)
 32. profile_snapshots               (FK -> users, personality_profiles)
 33. notifications                   (FK -> users)
 34. delivery_attempts               (FK -> notifications)
 35. user_notification_preferences   (FK -> users)
 36. notification_batches            (standalone)

Revision ID: 001_initial
Revises:
Create Date: 2026-01-29
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Helper: columns inherited from base model classes
# ---------------------------------------------------------------------------

def _timestamp_columns() -> list[sa.Column]:
    """TimestampMixin: created_at, updated_at."""
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def _version_column() -> sa.Column:
    """VersionMixin: optimistic locking."""
    return sa.Column("version", sa.Integer, nullable=False, server_default="1")


def _soft_delete_columns() -> list[sa.Column]:
    """SoftDeleteMixin: deleted_at, state."""
    return [
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("state", sa.String(20), nullable=False, server_default="active", index=True),
    ]


def _audit_columns() -> list[sa.Column]:
    """AuditMixin: created_by, updated_by."""
    return [
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column("updated_by", sa.String(64), nullable=True),
    ]


def _encrypted_field_columns() -> list[sa.Column]:
    """EncryptedFieldMixin: encryption metadata."""
    return [
        sa.Column("encryption_key_id", sa.String(64), nullable=False, index=True),
        sa.Column("encryption_algorithm", sa.String(50), nullable=False, server_default="AES-256-GCM"),
        sa.Column("encryption_version", sa.String(20), nullable=False, server_default="v1"),
    ]


def _audit_trail_columns() -> list[sa.Column]:
    """AuditTrailMixin: comprehensive access tracking."""
    return [
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_accessed_by", sa.String(64), nullable=True),
        sa.Column("change_history", JSONB, nullable=False, server_default="{}"),
    ]


def _base_model_pk() -> sa.Column:
    """UUID primary key used by BaseModel subclasses."""
    return sa.Column("id", UUID(as_uuid=True), primary_key=True)


def _auditable_model_columns() -> list[sa.Column]:
    """All columns from AuditableModel (BaseModel + AuditMixin + SoftDeleteMixin)."""
    return (
        _timestamp_columns()
        + [_version_column()]
        + _audit_columns()
        + _soft_delete_columns()
    )


def _clinical_base_columns(user_fk_ondelete: str = "CASCADE") -> list[sa.Column]:
    """All columns from ClinicalBase (AuditableModel + EncryptedFieldMixin + AuditTrailMixin + user_id + is_phi)."""
    return (
        _auditable_model_columns()
        + [
            sa.Column(
                "user_id", UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete=user_fk_ondelete),
                nullable=False, index=True,
            ),
            sa.Column("is_phi", sa.Boolean, nullable=False, server_default="true"),
        ]
        + _encrypted_field_columns()
        + _audit_trail_columns()
    )


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # -----------------------------------------------------------------------
    # 1. users (anchor table)
    # -----------------------------------------------------------------------
    op.create_table(
        "users",
        _base_model_pk(),
        *_timestamp_columns(),
        _version_column(),
        *_audit_columns(),
        *_soft_delete_columns(),
        # EncryptedFieldMixin columns
        *_encrypted_field_columns(),
        # Core identity
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="USER", index=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="PENDING_VERIFICATION", index=True),
        # Contact
        sa.Column("phone_number", sa.String(20), nullable=True),
        sa.Column("is_on_call", sa.Boolean, nullable=False, server_default="false", index=True),
        # Locale
        sa.Column("timezone", sa.String(50), nullable=False, server_default="UTC"),
        sa.Column("locale", sa.String(10), nullable=False, server_default="en"),
        # Profile
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("bio", sa.Text, nullable=True),
        # Verification
        sa.Column("email_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("email_verification_token", sa.String(256), nullable=True),
        # Security
        sa.Column("login_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 2. user_preferences
    # -----------------------------------------------------------------------
    op.create_table(
        "user_preferences",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, unique=True, index=True,
        ),
        # Notification preferences
        sa.Column("notification_email", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("notification_sms", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("notification_push", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("notification_channels", JSONB, nullable=False, server_default="[]"),
        sa.Column("session_reminders", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("progress_updates", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("marketing_emails", sa.Boolean, nullable=False, server_default="false"),
        # Data sharing
        sa.Column("data_sharing_research", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("data_sharing_improvement", sa.Boolean, nullable=False, server_default="false"),
        # UI
        sa.Column("theme", sa.String(10), nullable=False, server_default="system"),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        # Accessibility
        sa.Column("accessibility_high_contrast", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("accessibility_large_text", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("accessibility_screen_reader", sa.Boolean, nullable=False, server_default="false"),
    )

    # -----------------------------------------------------------------------
    # 3. consent_records
    # -----------------------------------------------------------------------
    op.create_table(
        "consent_records",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("consent_type", sa.String(50), nullable=False, index=True),
        sa.Column("granted", sa.Boolean, nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 4. clinician_patient_assignments (cross-service support)
    # -----------------------------------------------------------------------
    op.create_table(
        "clinician_patient_assignments",
        _base_model_pk(),
        sa.Column(
            "clinician_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column(
            "patient_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("clinician_id", "patient_id", name="uq_clinician_patient"),
    )

    # -----------------------------------------------------------------------
    # 5. audit_logs (immutable HIPAA compliance)
    # -----------------------------------------------------------------------
    op.create_table(
        "audit_logs",
        _base_model_pk(),
        *_timestamp_columns(),
        sa.Column("event_type", sa.String(64), nullable=False, index=True),
        sa.Column("actor_id", sa.String(128), nullable=True, index=True),
        sa.Column("actor_type", sa.String(32), nullable=False, server_default="user"),
        sa.Column("resource_type", sa.String(64), nullable=True, index=True),
        sa.Column("resource_id", sa.String(128), nullable=True),
        sa.Column("action", sa.String(64), nullable=False, index=True),
        sa.Column("outcome", sa.String(32), nullable=False, server_default="success"),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 6. system_configurations
    # -----------------------------------------------------------------------
    op.create_table(
        "system_configurations",
        _base_model_pk(),
        *_timestamp_columns(),
        _version_column(),
        sa.Column("key", sa.String(128), nullable=False, unique=True, index=True),
        sa.Column("value", JSONB, nullable=False, server_default="{}"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
    )

    # -----------------------------------------------------------------------
    # 7. safety_resources (reference)
    # -----------------------------------------------------------------------
    op.create_table(
        "safety_resources",
        _base_model_pk(),
        sa.Column("resource_code", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("contact", sa.String(255), nullable=False),
        sa.Column("url", sa.String(512), nullable=True),
        sa.Column("available_24_7", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
    )

    # -----------------------------------------------------------------------
    # 8. clinical_references (DSM-5-TR codes)
    # -----------------------------------------------------------------------
    op.create_table(
        "clinical_references",
        _base_model_pk(),
        sa.Column("code", sa.String(32), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(64), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )

    # -----------------------------------------------------------------------
    # 9. therapy_techniques (reference)
    # -----------------------------------------------------------------------
    op.create_table(
        "therapy_techniques",
        _base_model_pk(),
        sa.Column("technique_code", sa.String(128), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("modality", sa.String(64), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("contraindications", JSONB, nullable=False, server_default="[]"),
        sa.Column("evidence_level", sa.String(32), nullable=False),
    )

    # -----------------------------------------------------------------------
    # 10. safety_events (immutable audit trail, FK -> users RESTRICT)
    # -----------------------------------------------------------------------
    op.create_table(
        "safety_events",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False, index=True,
        ),
        sa.Column("severity_level", sa.Integer, nullable=False, index=True),
        sa.Column("event_type", sa.String(64), nullable=False, index=True),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
    )

    # -----------------------------------------------------------------------
    # 11. safety_assessments (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "safety_assessments",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("assessment_type", sa.String(50), nullable=False, index=True),
        sa.Column("content_assessed", sa.Text, nullable=True),
        sa.Column("risk_level", sa.String(20), nullable=False, index=True),
        sa.Column("risk_score", sa.Float, nullable=True),
        sa.Column("crisis_level", sa.String(20), nullable=True, index=True),
        sa.Column("is_safe", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("risk_factors", JSONB, nullable=False, server_default="{}"),
        sa.Column("protective_factors", JSONB, nullable=False, server_default="{}"),
        sa.Column("trigger_indicators", JSONB, nullable=False, server_default="[]"),
        sa.Column("detection_layers_triggered", JSONB, nullable=False, server_default="[]"),
        sa.Column("recommended_action", sa.String(100), nullable=True),
        sa.Column("recommended_interventions", JSONB, nullable=False, server_default="{}"),
        sa.Column("requires_escalation", sa.Boolean, nullable=False, server_default="false", index=True),
        sa.Column("requires_human_review", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("detection_time_ms", sa.Integer, nullable=True),
        sa.Column("context", JSONB, nullable=False, server_default="{}"),
        sa.Column("assessment_notes", sa.Text, nullable=True),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("next_assessment_due", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_by", sa.String(64), nullable=True),
        sa.Column("review_notes", sa.Text, nullable=True),
        sa.Column(
            "assessor_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("assessment_method", sa.String(100), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 12. safety_plans (ClinicalBase, FK -> safety_assessments)
    # -----------------------------------------------------------------------
    op.create_table(
        "safety_plans",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("status", sa.String(50), nullable=False, server_default="DRAFT", index=True),
        sa.Column(
            "assessment_id", UUID(as_uuid=True),
            sa.ForeignKey("safety_assessments.id", ondelete="SET NULL"),
            nullable=True, index=True,
        ),
        sa.Column("warning_signs", JSONB, nullable=False, server_default="[]"),
        sa.Column("coping_strategies", JSONB, nullable=False, server_default="[]"),
        sa.Column("emergency_contacts", JSONB, nullable=False, server_default="[]"),
        sa.Column("safe_environment_actions", JSONB, nullable=False, server_default="[]"),
        sa.Column("reasons_to_live", JSONB, nullable=False, server_default="[]"),
        sa.Column("professional_resources", JSONB, nullable=False, server_default="[]"),
        sa.Column("clinician_notes", sa.Text, nullable=True),
        sa.Column("plan_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_reviewed_by", sa.String(64), nullable=True),
        sa.Column("next_review_due", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 13. risk_factors (ClinicalBase, FK -> safety_assessments)
    # -----------------------------------------------------------------------
    op.create_table(
        "risk_factors",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "assessment_id", UUID(as_uuid=True),
            sa.ForeignKey("safety_assessments.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("factor_type", sa.String(100), nullable=False, index=True),
        sa.Column("factor_description", sa.Text, nullable=False),
        sa.Column("severity_level", sa.Integer, nullable=False),
        sa.Column("identified_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true", index=True),
        sa.Column("context_data", JSONB, nullable=False, server_default="{}"),
    )

    # -----------------------------------------------------------------------
    # 14. contraindication_checks (SafetyEventBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "contraindication_checks",
        _base_model_pk(),
        *_auditable_model_columns(),
        # SafetyEventBase columns
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="RESTRICT"),
            nullable=False, index=True,
        ),
        sa.Column("severity_level", sa.Integer, nullable=False, index=True),
        sa.Column("event_type", sa.String(64), nullable=False, index=True),
        # ContraindicationCheck-specific columns
        sa.Column("check_type", sa.String(100), nullable=False, index=True),
        sa.Column("subject_identifier", sa.String(200), nullable=False),
        sa.Column("subject_details", JSONB, nullable=False, server_default="{}"),
        sa.Column("contraindications_found", sa.Boolean, nullable=False, index=True),
        sa.Column("contraindication_details", JSONB, nullable=False, server_default="{}"),
        sa.Column("risk_assessment", sa.String(20), nullable=False),
        sa.Column("recommended_action", sa.String(50), nullable=False),
        sa.Column("recommendation_rationale", sa.Text, nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("checked_by_system", sa.String(100), nullable=False),
        sa.Column("check_version", sa.String(50), nullable=False),
        sa.Column("data_sources_consulted", JSONB, nullable=False, server_default="{}"),
    )

    # -----------------------------------------------------------------------
    # 15. contraindication_rules (standalone, from safety service SQL schema)
    # -----------------------------------------------------------------------
    op.create_table(
        "contraindication_rules",
        _base_model_pk(),
        *_timestamp_columns(),
        _version_column(),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("updated_by", sa.String(255), nullable=True),
        sa.Column("technique", sa.String(64), nullable=False, index=True),
        sa.Column("condition", sa.String(64), nullable=False, index=True),
        sa.Column("contraindication_type", sa.String(32), nullable=False, index=True),
        sa.Column("severity", sa.Numeric(3, 2), nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true", index=True),
        sa.UniqueConstraint("technique", "condition", name="uq_technique_condition"),
    )

    # -----------------------------------------------------------------------
    # 16. rule_alternatives (FK -> contraindication_rules)
    # -----------------------------------------------------------------------
    op.create_table(
        "rule_alternatives",
        _base_model_pk(),
        sa.Column(
            "rule_id", UUID(as_uuid=True),
            sa.ForeignKey("contraindication_rules.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("alternative_technique", sa.String(64), nullable=False),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("rule_id", "alternative_technique", name="uq_rule_alternative"),
    )

    # -----------------------------------------------------------------------
    # 17. rule_prerequisites (FK -> contraindication_rules)
    # -----------------------------------------------------------------------
    op.create_table(
        "rule_prerequisites",
        _base_model_pk(),
        sa.Column(
            "rule_id", UUID(as_uuid=True),
            sa.ForeignKey("contraindication_rules.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("prerequisite", sa.Text, nullable=False),
        sa.Column("display_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_required", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # -----------------------------------------------------------------------
    # 18. treatment_plans (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "treatment_plans",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("primary_diagnosis", sa.String(200), nullable=False),
        sa.Column("secondary_diagnoses", JSONB, nullable=False, server_default="[]"),
        sa.Column("severity", sa.String(30), nullable=False, index=True),
        sa.Column("stepped_care_level", sa.String(30), nullable=False),
        sa.Column("primary_modality", sa.String(30), nullable=False, index=True),
        sa.Column("adjunct_modalities", JSONB, nullable=False, server_default="[]"),
        sa.Column("current_phase", sa.String(30), nullable=False, server_default="FOUNDATION"),
        sa.Column("phase_sessions_completed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_sessions_completed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("session_frequency_per_week", sa.Integer, nullable=False, server_default="1"),
        sa.Column("response_status", sa.String(30), nullable=False, server_default="NOT_STARTED"),
        sa.Column("skills_acquired", JSONB, nullable=False, server_default="[]"),
        sa.Column("skills_in_progress", JSONB, nullable=False, server_default="[]"),
        sa.Column("contraindications", JSONB, nullable=False, server_default="[]"),
        sa.Column("baseline_phq9", sa.Integer, nullable=True),
        sa.Column("latest_phq9", sa.Integer, nullable=True),
        sa.Column("baseline_gad7", sa.Integer, nullable=True),
        sa.Column("latest_gad7", sa.Integer, nullable=True),
        sa.Column("target_completion", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true", index=True),
        sa.Column("termination_reason", sa.String(500), nullable=True),
        sa.Column("treatment_goals", JSONB, nullable=False, server_default="[]"),
    )

    # -----------------------------------------------------------------------
    # 19. therapy_sessions (ClinicalBase, FK -> treatment_plans)
    # -----------------------------------------------------------------------
    op.create_table(
        "therapy_sessions",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "treatment_plan_id", UUID(as_uuid=True),
            sa.ForeignKey("treatment_plans.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("session_number", sa.Integer, nullable=False),
        sa.Column("current_phase", sa.String(20), nullable=False, server_default="PRE_SESSION", index=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mood_rating_start", sa.Integer, nullable=True),
        sa.Column("mood_rating_end", sa.Integer, nullable=True),
        sa.Column("agenda_items", JSONB, nullable=False, server_default="[]"),
        sa.Column("topics_covered", JSONB, nullable=False, server_default="[]"),
        sa.Column("skills_practiced", JSONB, nullable=False, server_default="[]"),
        sa.Column("insights_gained", JSONB, nullable=False, server_default="[]"),
        sa.Column("current_risk", sa.String(20), nullable=False, server_default="NONE", index=True),
        sa.Column("safety_flags", JSONB, nullable=False, server_default="[]"),
        sa.Column("session_rating", sa.Float, nullable=True),
        sa.Column("alliance_score", sa.Float, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("next_session_focus", sa.Text, nullable=True),
    )

    # -----------------------------------------------------------------------
    # 20. therapy_interventions (ClinicalBase, FK -> therapy_sessions)
    # -----------------------------------------------------------------------
    op.create_table(
        "therapy_interventions",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("therapy_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("technique_id", UUID(as_uuid=True), nullable=True),
        sa.Column("technique_name", sa.String(200), nullable=False),
        sa.Column("modality", sa.String(30), nullable=False),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("messages_exchanged", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("engagement_score", sa.Float, nullable=True),
        sa.Column("skills_practiced", JSONB, nullable=False, server_default="[]"),
        sa.Column("insights_gained", JSONB, nullable=False, server_default="[]"),
    )

    # -----------------------------------------------------------------------
    # 21. homework_assignments (ClinicalBase, FK -> therapy_sessions)
    # -----------------------------------------------------------------------
    op.create_table(
        "homework_assignments",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("therapy_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("technique_id", UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="ASSIGNED", index=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completion_notes", sa.Text, nullable=True),
        sa.Column("rating", sa.Integer, nullable=True),
    )

    # -----------------------------------------------------------------------
    # 22. diagnosis_sessions (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "diagnosis_sessions",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("session_number", sa.Integer, nullable=False),
        sa.Column("phase", sa.String(20), nullable=False, server_default="RAPPORT", index=True),
        sa.Column("primary_hypothesis_id", UUID(as_uuid=True), nullable=True),
        sa.Column("messages", JSONB, nullable=False, server_default="[]"),
        sa.Column("safety_flags", JSONB, nullable=False, server_default="[]"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true", index=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("recommendations", JSONB, nullable=False, server_default="[]"),
        sa.Column("session_metadata", JSONB, nullable=False, server_default="{}"),
    )

    # -----------------------------------------------------------------------
    # 23. diagnosis_symptoms (ClinicalBase, FK -> diagnosis_sessions)
    # -----------------------------------------------------------------------
    op.create_table(
        "diagnosis_symptoms",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("name", sa.String(200), nullable=False, index=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("symptom_type", sa.String(20), nullable=False, index=True),
        sa.Column("severity", sa.String(30), nullable=False),
        sa.Column("onset", sa.String(200), nullable=True),
        sa.Column("duration", sa.String(200), nullable=True),
        sa.Column("frequency", sa.String(200), nullable=True),
        sa.Column("triggers", JSONB, nullable=False, server_default="[]"),
        sa.Column("extracted_from", sa.Text, nullable=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("validated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("validation_source", sa.String(100), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 24. diagnosis_hypotheses (ClinicalBase, FK -> diagnosis_sessions)
    # -----------------------------------------------------------------------
    op.create_table(
        "diagnosis_hypotheses",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("dsm5_code", sa.String(20), nullable=True, index=True),
        sa.Column("icd11_code", sa.String(20), nullable=True, index=True),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("confidence_level", sa.String(20), nullable=False, server_default="MEDIUM"),
        sa.Column("criteria_met", JSONB, nullable=False, server_default="[]"),
        sa.Column("criteria_missing", JSONB, nullable=False, server_default="[]"),
        sa.Column("supporting_evidence", JSONB, nullable=False, server_default="[]"),
        sa.Column("contra_evidence", JSONB, nullable=False, server_default="[]"),
        sa.Column("severity", sa.String(30), nullable=False, server_default="MODERATE"),
        sa.Column("hitop_dimensions", JSONB, nullable=False, server_default="{}"),
        sa.Column("challenged", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("challenge_results", JSONB, nullable=False, server_default="[]"),
        sa.Column("calibrated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("original_confidence", sa.Float, nullable=True),
    )

    # -----------------------------------------------------------------------
    # 25. diagnosis_records (ClinicalBase, FK -> diagnosis_sessions)
    # -----------------------------------------------------------------------
    op.create_table(
        "diagnosis_records",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "session_id", UUID(as_uuid=True),
            sa.ForeignKey("diagnosis_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("primary_diagnosis", sa.String(300), nullable=False),
        sa.Column("dsm5_code", sa.String(20), nullable=True, index=True),
        sa.Column("icd11_code", sa.String(20), nullable=True, index=True),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("severity", sa.String(30), nullable=False),
        sa.Column("symptom_summary", JSONB, nullable=False, server_default="[]"),
        sa.Column("supporting_evidence", JSONB, nullable=False, server_default="[]"),
        sa.Column("differential_diagnoses", JSONB, nullable=False, server_default="[]"),
        sa.Column("recommendations", JSONB, nullable=False, server_default="[]"),
        sa.Column("assessment_scores", JSONB, nullable=False, server_default="{}"),
        sa.Column("clinician_notes", sa.Text, nullable=True),
        sa.Column("reviewed", sa.Boolean, nullable=False, server_default="false", index=True),
        sa.Column("reviewed_by", sa.String(64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 26. memory_records (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "memory_records",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("tier", sa.String(30), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_type", sa.String(30), nullable=False, index=True),
        sa.Column("retention_category", sa.String(20), nullable=False, index=True),
        sa.Column("importance_score", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("emotional_valence", sa.Float, nullable=True),
        sa.Column("record_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("tags", JSONB, nullable=False, server_default="[]"),
        sa.Column("related_records", JSONB, nullable=False, server_default="[]"),
        sa.Column("retention_strength", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("access_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("is_archived", sa.Boolean, nullable=False, server_default="false", index=True),
        sa.Column("is_safety_critical", sa.Boolean, nullable=False, server_default="false", index=True),
    )

    # -----------------------------------------------------------------------
    # 27. memory_user_profiles (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "memory_user_profiles",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("profile_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("personal_facts", JSONB, nullable=False, server_default="{}"),
        sa.Column("therapeutic_context", JSONB, nullable=False, server_default="{}"),
        sa.Column("communication_preferences", JSONB, nullable=False, server_default="{}"),
        sa.Column("safety_information", JSONB, nullable=False, server_default="{}"),
        sa.Column("diagnosed_conditions", JSONB, nullable=False, server_default="[]"),
        sa.Column("current_treatments", JSONB, nullable=False, server_default="[]"),
        sa.Column("support_network", JSONB, nullable=False, server_default="[]"),
        sa.Column("triggers", JSONB, nullable=False, server_default="[]"),
        sa.Column("coping_strategies", JSONB, nullable=False, server_default="[]"),
        sa.Column("personality_traits", JSONB, nullable=False, server_default="{}"),
        sa.Column("total_sessions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("first_session_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_session_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_crisis_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("crisis_count", sa.Integer, nullable=False, server_default="0"),
    )

    # -----------------------------------------------------------------------
    # 28. session_summaries (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "session_summaries",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False, unique=True, index=True),
        sa.Column("session_number", sa.Integer, nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("key_topics", JSONB, nullable=False, server_default="[]"),
        sa.Column("emotional_arc", JSONB, nullable=False, server_default="[]"),
        sa.Column("techniques_used", JSONB, nullable=False, server_default="[]"),
        sa.Column("key_insights", JSONB, nullable=False, server_default="[]"),
        sa.Column("homework_assigned", JSONB, nullable=False, server_default="[]"),
        sa.Column("homework_reviewed", JSONB, nullable=False, server_default="[]"),
        sa.Column("progress_notes", sa.Text, nullable=True),
        sa.Column("risk_level_start", sa.String(20), nullable=True),
        sa.Column("risk_level_end", sa.String(20), nullable=True),
        sa.Column("message_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("duration_minutes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("session_date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("retention_strength", sa.Float, nullable=False, server_default="1.0"),
    )

    # -----------------------------------------------------------------------
    # 29. therapeutic_events (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "therapeutic_events",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("event_type", sa.String(50), nullable=False, index=True),
        sa.Column("severity", sa.String(20), nullable=False, server_default="low"),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("related_events", JSONB, nullable=False, server_default="[]"),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("retention_strength", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_safety_critical", sa.Boolean, nullable=False, server_default="false", index=True),
    )

    # -----------------------------------------------------------------------
    # 30. personality_profiles (ClinicalBase)
    # -----------------------------------------------------------------------
    op.create_table(
        "personality_profiles",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column("ocean_scores", JSONB, nullable=False, server_default="{}"),
        sa.Column("communication_style", JSONB, nullable=False, server_default="{}"),
        sa.Column("assessment_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stability_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("assessment_history", JSONB, nullable=False, server_default="[]"),
    )

    # -----------------------------------------------------------------------
    # 31. trait_assessments (ClinicalBase, FK -> personality_profiles)
    # -----------------------------------------------------------------------
    op.create_table(
        "trait_assessments",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "profile_id", UUID(as_uuid=True),
            sa.ForeignKey("personality_profiles.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("ocean_scores", JSONB, nullable=False, server_default="{}"),
        sa.Column("source", sa.String(30), nullable=False, index=True),
        sa.Column("assessment_metadata", JSONB, nullable=True, server_default="{}"),
        sa.Column("evidence", JSONB, nullable=False, server_default="[]"),
    )

    # -----------------------------------------------------------------------
    # 32. profile_snapshots (ClinicalBase, FK -> personality_profiles)
    # -----------------------------------------------------------------------
    op.create_table(
        "profile_snapshots",
        _base_model_pk(),
        *_clinical_base_columns(),
        sa.Column(
            "profile_id", UUID(as_uuid=True),
            sa.ForeignKey("personality_profiles.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("ocean_scores", JSONB, nullable=False, server_default="{}"),
        sa.Column("communication_style", JSONB, nullable=True, server_default="{}"),
        sa.Column("assessment_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stability_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("reason", sa.String(200), nullable=False),
    )

    # -----------------------------------------------------------------------
    # 33. notifications (AuditableModel, FK -> users)
    # -----------------------------------------------------------------------
    op.create_table(
        "notifications",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("category", sa.String(20), nullable=False, index=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("html_body", sa.Text, nullable=True),
        sa.Column("channel", sa.String(10), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING", index=True),
        sa.Column("priority", sa.Integer, nullable=False, server_default="3"),
        sa.Column("correlation_id", UUID(as_uuid=True), nullable=True, index=True),
        sa.Column("reference_id", sa.String(128), nullable=True),
        sa.Column("reference_type", sa.String(64), nullable=True),
        sa.Column("template_id", sa.String(64), nullable=True, index=True),
        sa.Column("variables", JSONB, nullable=False, server_default="{}"),
        sa.Column("notification_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer, nullable=False, server_default="3"),
        sa.Column("external_message_id", sa.String(256), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("clicked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 34. delivery_attempts (AuditableModel, FK -> notifications)
    # -----------------------------------------------------------------------
    op.create_table(
        "delivery_attempts",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "notification_id", UUID(as_uuid=True),
            sa.ForeignKey("notifications.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("channel", sa.String(10), nullable=False),
        sa.Column("attempt_number", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("external_message_id", sa.String(256), nullable=True),
        sa.Column("provider_response", JSONB, nullable=False, server_default="{}"),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )

    # -----------------------------------------------------------------------
    # 35. user_notification_preferences (AuditableModel, FK -> users)
    # -----------------------------------------------------------------------
    op.create_table(
        "user_notification_preferences",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column(
            "user_id", UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False, unique=True, index=True,
        ),
        sa.Column("global_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("channels", JSONB, nullable=False, server_default="{}"),
        sa.Column("category_overrides", JSONB, nullable=False, server_default="{}"),
        sa.Column("unsubscribed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("unsubscribe_reason", sa.String(500), nullable=True),
        sa.Column("user_timezone", sa.String(50), nullable=False, server_default="UTC"),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
    )

    # -----------------------------------------------------------------------
    # 36. notification_batches (AuditableModel, standalone)
    # -----------------------------------------------------------------------
    op.create_table(
        "notification_batches",
        _base_model_pk(),
        *_auditable_model_columns(),
        sa.Column("notification_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("total_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pending_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("delivered_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


# ---------------------------------------------------------------------------
# TODO: Post-MVP — Enable Row-Level Security for PHI tables
# ---------------------------------------------------------------------------
# ALTER TABLE users ENABLE ROW LEVEL SECURITY;
# CREATE POLICY user_isolation ON users USING (id = current_setting('app.current_user_id')::uuid);
# Tables requiring RLS: users, diagnosis_sessions, diagnosis_symptoms,
# diagnosis_hypotheses, diagnosis_records, therapy_sessions, therapy_interventions,
# homework_assignments, treatment_plans, memory_records, session_summaries,
# user_facts, personality_profiles, trait_assessments, safety_assessments,
# safety_plans, consent_records, notifications
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Downgrade — drop all tables in reverse dependency order
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("notification_batches")
    op.drop_table("user_notification_preferences")
    op.drop_table("delivery_attempts")
    op.drop_table("notifications")
    op.drop_table("profile_snapshots")
    op.drop_table("trait_assessments")
    op.drop_table("personality_profiles")
    op.drop_table("therapeutic_events")
    op.drop_table("session_summaries")
    op.drop_table("memory_user_profiles")
    op.drop_table("memory_records")
    op.drop_table("diagnosis_records")
    op.drop_table("diagnosis_hypotheses")
    op.drop_table("diagnosis_symptoms")
    op.drop_table("diagnosis_sessions")
    op.drop_table("homework_assignments")
    op.drop_table("therapy_interventions")
    op.drop_table("therapy_sessions")
    op.drop_table("treatment_plans")
    op.drop_table("rule_prerequisites")
    op.drop_table("rule_alternatives")
    op.drop_table("contraindication_rules")
    op.drop_table("contraindication_checks")
    op.drop_table("risk_factors")
    op.drop_table("safety_plans")
    op.drop_table("safety_assessments")
    op.drop_table("safety_events")
    op.drop_table("therapy_techniques")
    op.drop_table("clinical_references")
    op.drop_table("safety_resources")
    op.drop_table("system_configurations")
    op.drop_table("audit_logs")
    op.drop_table("clinician_patient_assignments")
    op.drop_table("consent_records")
    op.drop_table("user_preferences")
    op.drop_table("users")
