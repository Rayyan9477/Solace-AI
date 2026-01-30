"""Initial schema - core tables for Solace-AI platform.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-29
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users table (referenced by FK from all user-related tables)
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(128), nullable=True, unique=True),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("is_verified", sa.Boolean, nullable=False, default=False),
        sa.Column("roles", JSONB, nullable=False, server_default="[]"),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state", sa.String(20), nullable=False, default="active"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_created_at", "users", ["created_at"])
    op.create_index("ix_users_state", "users", ["state"])

    # Audit logs table (HIPAA compliance - immutable)
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=True),
        sa.Column("actor_type", sa.String(32), nullable=False, default="user"),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", sa.String(128), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False, default="success"),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(512), nullable=True),
    )
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"])
    op.create_index("ix_audit_logs_actor_id", "audit_logs", ["actor_id"])
    op.create_index("ix_audit_logs_resource_type", "audit_logs", ["resource_type"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    # System configurations table
    op.create_table(
        "system_configurations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("key", sa.String(128), nullable=False, unique=True),
        sa.Column("value", JSONB, nullable=False, server_default="{}"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
    )
    op.create_index("ix_system_configurations_key", "system_configurations", ["key"])

    # Safety resources reference table
    op.create_table(
        "safety_resources",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("resource_code", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("contact", sa.String(255), nullable=False),
        sa.Column("url", sa.String(512), nullable=True),
        sa.Column("available_24_7", sa.Boolean, nullable=False, default=True),
        sa.Column("priority", sa.Integer, nullable=False, default=0),
    )
    op.create_index("ix_safety_resources_code", "safety_resources", ["resource_code"])

    # Clinical references table (DSM-5-TR codes)
    op.create_table(
        "clinical_references",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("code", sa.String(32), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_clinical_references_code", "clinical_references", ["code"])
    op.create_index("ix_clinical_references_category", "clinical_references", ["category"])

    # Therapy techniques table
    op.create_table(
        "therapy_techniques",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("technique_code", sa.String(128), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("modality", sa.String(64), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("contraindications", JSONB, nullable=False, server_default="[]"),
        sa.Column("evidence_level", sa.String(32), nullable=False),
    )
    op.create_index("ix_therapy_techniques_code", "therapy_techniques", ["technique_code"])
    op.create_index("ix_therapy_techniques_modality", "therapy_techniques", ["modality"])

    # Safety events table (immutable audit trail)
    op.create_table(
        "safety_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("severity_level", sa.Integer, nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("state", sa.String(20), nullable=False, default="active"),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column("updated_by", sa.String(64), nullable=True),
    )
    op.create_index("ix_safety_events_user_id", "safety_events", ["user_id"])
    op.create_index("ix_safety_events_severity", "safety_events", ["severity_level"])
    op.create_index("ix_safety_events_type", "safety_events", ["event_type"])
    op.create_index("ix_safety_events_created_at", "safety_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("safety_events")
    op.drop_table("therapy_techniques")
    op.drop_table("clinical_references")
    op.drop_table("safety_resources")
    op.drop_table("system_configurations")
    op.drop_table("audit_logs")
    op.drop_table("users")
