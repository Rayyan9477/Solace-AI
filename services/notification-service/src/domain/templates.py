"""
Solace-AI Notification Service - Template Management.

Notification template definitions and rendering engine using Jinja2.
Supports multiple template types for email, SMS, and push notifications.

Architecture Layer: Domain
Principles: Template Method Pattern, Factory Pattern, Immutability
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from jinja2 import Environment, BaseLoader, TemplateError, StrictUndefined
from pydantic import BaseModel, Field, field_validator
import structlog

logger = structlog.get_logger(__name__)


class TemplateType(str, Enum):
    """Notification template types."""
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    WELCOME = "welcome"
    ACCOUNT_LOCKED = "account_locked"
    CLINICIAN_ALERT = "clinician_alert"
    CRISIS_ESCALATION = "crisis_escalation"
    APPOINTMENT_REMINDER = "appointment_reminder"
    SESSION_SUMMARY = "session_summary"
    CONSENT_REQUEST = "consent_request"
    SYSTEM_ALERT = "system_alert"
    # Safety event-driven templates
    CRISIS_ALERT = "crisis_alert"
    ESCALATION_ALERT = "escalation_alert"
    RISK_ALERT = "risk_alert"


class TemplateNotFoundError(Exception):
    """Raised when a template is not found."""
    def __init__(self, template_type: TemplateType) -> None:
        self.template_type = template_type
        super().__init__(f"Template not found: {template_type.value}")


class TemplateRenderError(Exception):
    """Raised when template rendering fails."""
    def __init__(self, template_type: TemplateType, reason: str) -> None:
        self.template_type = template_type
        self.reason = reason
        super().__init__(f"Failed to render template {template_type.value}: {reason}")


class NotificationTemplate(BaseModel):
    """
    Notification template definition.

    Templates use Jinja2 syntax for variable interpolation.
    Each template has subject, body, and optional HTML body.
    """
    template_id: UUID = Field(default_factory=uuid4, description="Unique template ID")
    template_type: TemplateType = Field(..., description="Template type")
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    description: str = Field(default="", max_length=500, description="Template description")

    subject_template: str = Field(..., min_length=1, description="Subject line template (Jinja2)")
    body_template: str = Field(..., min_length=1, description="Plain text body template (Jinja2)")
    html_template: str | None = Field(default=None, description="HTML body template (Jinja2)")

    required_variables: list[str] = Field(default_factory=list, description="Required template variables")
    default_values: dict[str, Any] = Field(default_factory=dict, description="Default variable values")

    version: str = Field(default="1.0", description="Template version")
    is_active: bool = Field(default=True, description="Whether template is active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"frozen": False}

    @field_validator("required_variables", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        return list(v) if isinstance(v, (list, tuple, set)) else [v]


class RenderedTemplate(BaseModel):
    """Result of template rendering."""
    template_type: TemplateType
    subject: str
    body: str
    html_body: str | None = None
    rendered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TemplateEngine:
    """
    Jinja2-based template rendering engine.

    Thread-safe rendering with strict undefined variable handling.
    """
    def __init__(self) -> None:
        self._env = Environment(
            loader=BaseLoader(),
            autoescape=True,
            undefined=StrictUndefined,  # Strict mode - fail on undefined vars
        )
        logger.info("template_engine_initialized")

    def render(
        self,
        template: NotificationTemplate,
        variables: dict[str, Any],
    ) -> RenderedTemplate:
        """
        Render a template with provided variables.

        Args:
            template: The notification template to render
            variables: Variables to substitute in template

        Returns:
            RenderedTemplate with subject and body rendered

        Raises:
            TemplateRenderError: If rendering fails or required variables missing
        """
        merged_vars = {**template.default_values, **variables}
        missing = [v for v in template.required_variables if v not in merged_vars]
        if missing:
            logger.warning("template_missing_variables",
                         template_type=template.template_type.value, missing=missing)
            raise TemplateRenderError(template.template_type, f"Missing required variables: {missing}")

        try:
            subject = self._render_string(template.subject_template, merged_vars)
            body = self._render_string(template.body_template, merged_vars)
            html_body = None
            if template.html_template:
                html_body = self._render_string(template.html_template, merged_vars)

            logger.debug("template_rendered", template_type=template.template_type.value)
            return RenderedTemplate(
                template_type=template.template_type,
                subject=subject,
                body=body,
                html_body=html_body,
            )
        except TemplateError as e:
            logger.error("template_render_failed",
                        template_type=template.template_type.value, error=str(e))
            raise TemplateRenderError(template.template_type, str(e)) from e

    def _render_string(self, template_str: str, variables: dict[str, Any]) -> str:
        """Render a single template string."""
        jinja_template = self._env.from_string(template_str)
        return jinja_template.render(**variables)


class TemplateRegistry:
    """
    Registry of notification templates.

    Provides default templates and allows custom template registration.
    Thread-safe template storage and retrieval.
    """
    def __init__(self) -> None:
        self._templates: dict[TemplateType, NotificationTemplate] = {}
        self._engine = TemplateEngine()
        self._register_default_templates()
        logger.info("template_registry_initialized", template_count=len(self._templates))

    def _register_default_templates(self) -> None:
        """Register built-in default templates."""
        defaults = [
            NotificationTemplate(
                template_type=TemplateType.EMAIL_VERIFICATION,
                name="Email Verification",
                description="Email verification link notification",
                subject_template="Verify your Solace-AI account",
                body_template="Hi {{ display_name }},\n\nPlease verify your email by clicking: {{ verification_link }}\n\nThis link expires in {{ expiry_hours }} hours.\n\nBest,\nSolace-AI Team",
                html_template="<p>Hi {{ display_name }},</p><p>Please <a href=\"{{ verification_link }}\">verify your email</a>.</p><p>This link expires in {{ expiry_hours }} hours.</p>",
                required_variables=["display_name", "verification_link", "expiry_hours"],
            ),
            NotificationTemplate(
                template_type=TemplateType.PASSWORD_RESET,
                name="Password Reset",
                description="Password reset link notification",
                subject_template="Reset your Solace-AI password",
                body_template="Hi {{ display_name }},\n\nReset your password: {{ reset_link }}\n\nThis link expires in {{ expiry_minutes }} minutes.\n\nIf you didn't request this, ignore this email.\n\nBest,\nSolace-AI Team",
                html_template="<p>Hi {{ display_name }},</p><p><a href=\"{{ reset_link }}\">Reset your password</a>.</p><p>Expires in {{ expiry_minutes }} minutes.</p>",
                required_variables=["display_name", "reset_link", "expiry_minutes"],
            ),
            NotificationTemplate(
                template_type=TemplateType.WELCOME,
                name="Welcome",
                description="Welcome notification for new users",
                subject_template="Welcome to Solace-AI, {{ display_name }}!",
                body_template="Hi {{ display_name }},\n\nWelcome to Solace-AI! We're here to support your mental health journey.\n\nGet started: {{ getting_started_link }}\n\nBest,\nSolace-AI Team",
                html_template="<p>Hi {{ display_name }},</p><p>Welcome to Solace-AI!</p><p><a href=\"{{ getting_started_link }}\">Get started</a></p>",
                required_variables=["display_name", "getting_started_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.ACCOUNT_LOCKED,
                name="Account Locked",
                description="Account locked due to failed login attempts",
                subject_template="Your Solace-AI account has been locked",
                body_template="Hi {{ display_name }},\n\nYour account was locked after {{ attempt_count }} failed login attempts.\n\nUnlock: {{ unlock_link }}\n\nIf this wasn't you, contact support.\n\nBest,\nSolace-AI Team",
                required_variables=["display_name", "attempt_count", "unlock_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.CLINICIAN_ALERT,
                name="Clinician Alert",
                description="Alert notification for clinicians",
                subject_template="[{{ severity }}] Patient Alert: {{ patient_name }}",
                body_template="Alert for patient {{ patient_name }} (ID: {{ patient_id }})\n\nSeverity: {{ severity }}\nType: {{ alert_type }}\nDetails: {{ alert_details }}\n\nReview: {{ dashboard_link }}",
                required_variables=["severity", "patient_name", "patient_id", "alert_type", "alert_details", "dashboard_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.CRISIS_ESCALATION,
                name="Crisis Escalation",
                description="Crisis escalation notification",
                subject_template="URGENT: Crisis Escalation - {{ patient_name }}",
                body_template="URGENT CRISIS ESCALATION\n\nPatient: {{ patient_name }} (ID: {{ patient_id }})\nRisk Level: {{ risk_level }}\nAssessment: {{ assessment_summary }}\n\nImmediate action required.\n\nDashboard: {{ dashboard_link }}",
                required_variables=["patient_name", "patient_id", "risk_level", "assessment_summary", "dashboard_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.APPOINTMENT_REMINDER,
                name="Appointment Reminder",
                description="Appointment reminder notification",
                subject_template="Reminder: Your Solace-AI session on {{ appointment_date }}",
                body_template="Hi {{ display_name }},\n\nReminder: You have a session scheduled for {{ appointment_date }} at {{ appointment_time }}.\n\nJoin: {{ session_link }}\n\nBest,\nSolace-AI Team",
                required_variables=["display_name", "appointment_date", "appointment_time", "session_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.SESSION_SUMMARY,
                name="Session Summary",
                description="Post-session summary notification",
                subject_template="Your Solace-AI session summary - {{ session_date }}",
                body_template="Hi {{ display_name }},\n\nHere's your session summary from {{ session_date }}:\n\n{{ summary_content }}\n\nView full report: {{ report_link }}\n\nBest,\nSolace-AI Team",
                required_variables=["display_name", "session_date", "summary_content", "report_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.CONSENT_REQUEST,
                name="Consent Request",
                description="Request for consent notification",
                subject_template="Consent required: {{ consent_type }}",
                body_template="Hi {{ display_name }},\n\nWe need your consent for: {{ consent_type }}\n\nDetails: {{ consent_description }}\n\nReview and respond: {{ consent_link }}\n\nBest,\nSolace-AI Team",
                required_variables=["display_name", "consent_type", "consent_description", "consent_link"],
            ),
            NotificationTemplate(
                template_type=TemplateType.SYSTEM_ALERT,
                name="System Alert",
                description="System alert notification",
                subject_template="[{{ severity }}] System Alert: {{ alert_title }}",
                body_template="System Alert\n\nSeverity: {{ severity }}\nTitle: {{ alert_title }}\nDetails: {{ alert_message }}\nTime: {{ timestamp }}\n\nAction: {{ action_required }}",
                required_variables=["severity", "alert_title", "alert_message", "timestamp", "action_required"],
            ),
            # Safety event-driven templates
            NotificationTemplate(
                template_type=TemplateType.CRISIS_ALERT,
                name="Crisis Alert",
                description="Automated crisis detection alert from safety service",
                subject_template="🚨 CRISIS DETECTED [{{ crisis_level }}] - User {{ user_id }}",
                body_template="CRISIS ALERT\n\n⚠️ Crisis Level: {{ crisis_level }}\n📊 Confidence: {{ confidence }}\n🔍 Detection Layer: {{ detection_layer }}\n\nUser ID: {{ user_id }}\nSession ID: {{ session_id }}\n\nTrigger Indicators:\n{{ trigger_indicators }}\n\nRecommended Action: {{ escalation_action }}\nRequires Human Review: {{ requires_human_review }}\n\nTime: {{ timestamp }}\n\n---\nThis is an automated alert from Solace-AI Safety Service.",
                html_template="<div style='font-family:sans-serif;max-width:600px;'><h2 style='color:#d32f2f;'>🚨 CRISIS ALERT</h2><table style='width:100%;border-collapse:collapse;'><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Crisis Level</strong></td><td style='padding:8px;border:1px solid #ddd;color:#d32f2f;font-weight:bold;'>{{ crisis_level }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Confidence</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ confidence }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>User ID</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ user_id }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Session ID</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ session_id }}</td></tr></table><h3>Trigger Indicators</h3><p>{{ trigger_indicators }}</p><p><strong>Recommended Action:</strong> {{ escalation_action }}</p><p><strong>Human Review Required:</strong> {{ requires_human_review }}</p><p style='color:#666;font-size:12px;margin-top:20px;'>Timestamp: {{ timestamp }}</p></div>",
                required_variables=["crisis_level", "user_id", "session_id", "trigger_indicators", "confidence", "escalation_action", "requires_human_review", "timestamp"],
                default_values={"detection_layer": "1"},
            ),
            NotificationTemplate(
                template_type=TemplateType.ESCALATION_ALERT,
                name="Escalation Alert",
                description="Case escalation notification to clinicians",
                subject_template="⚡ ESCALATION [{{ priority }}] - Case Requires Attention",
                body_template="ESCALATION ALERT\n\n🔴 Priority: {{ priority }}\n📝 Reason: {{ escalation_reason }}\n\nUser ID: {{ user_id }}\nSession ID: {{ session_id }}\nAssigned To: {{ assigned_clinician_id }}\n\nTime: {{ timestamp }}\n\n---\nPlease respond to this escalation immediately.\nThis is an automated alert from Solace-AI Safety Service.",
                html_template="<div style='font-family:sans-serif;max-width:600px;'><h2 style='color:#ff5722;'>⚡ ESCALATION ALERT</h2><table style='width:100%;border-collapse:collapse;'><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Priority</strong></td><td style='padding:8px;border:1px solid #ddd;color:#ff5722;font-weight:bold;'>{{ priority }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Reason</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ escalation_reason }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>User ID</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ user_id }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Assigned To</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ assigned_clinician_id }}</td></tr></table><p style='color:#666;font-size:12px;margin-top:20px;'>Timestamp: {{ timestamp }}</p><p style='background:#fff3e0;padding:10px;border-radius:4px;'><strong>Please respond to this escalation immediately.</strong></p></div>",
                required_variables=["priority", "escalation_reason", "user_id", "session_id", "assigned_clinician_id", "timestamp"],
            ),
            NotificationTemplate(
                template_type=TemplateType.RISK_ALERT,
                name="Risk Assessment Alert",
                description="Elevated risk assessment monitoring notification",
                subject_template="📈 Risk Alert [{{ risk_level }}] - User {{ user_id }}",
                body_template="RISK MONITORING ALERT\n\n📊 Risk Level: {{ risk_level }}\n📈 Risk Score: {{ risk_score }}\n🔍 Detection Layer: {{ detection_layer }}\n\nUser ID: {{ user_id }}\nSession ID: {{ session_id }}\n\nRecommended Action: {{ recommended_action }}\n\nTime: {{ timestamp }}\n\n---\nThis is an automated monitoring alert from Solace-AI Safety Service.",
                html_template="<div style='font-family:sans-serif;max-width:600px;'><h2 style='color:#ff9800;'>📈 RISK MONITORING ALERT</h2><table style='width:100%;border-collapse:collapse;'><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Risk Level</strong></td><td style='padding:8px;border:1px solid #ddd;color:#ff9800;font-weight:bold;'>{{ risk_level }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Risk Score</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ risk_score }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>User ID</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ user_id }}</td></tr><tr><td style='padding:8px;border:1px solid #ddd;'><strong>Session ID</strong></td><td style='padding:8px;border:1px solid #ddd;'>{{ session_id }}</td></tr></table><p><strong>Recommended Action:</strong> {{ recommended_action }}</p><p style='color:#666;font-size:12px;margin-top:20px;'>Timestamp: {{ timestamp }}</p></div>",
                required_variables=["risk_level", "risk_score", "user_id", "session_id", "recommended_action", "timestamp"],
                default_values={"detection_layer": "1"},
            ),
        ]
        for template in defaults:
            self._templates[template.template_type] = template

    def get_template(self, template_type: TemplateType) -> NotificationTemplate:
        """Get a template by type."""
        template = self._templates.get(template_type)
        if not template:
            logger.warning("template_not_found", template_type=template_type.value)
            raise TemplateNotFoundError(template_type)
        return template

    def register_template(self, template: NotificationTemplate) -> None:
        """Register or update a template."""
        self._templates[template.template_type] = template
        logger.info("template_registered", template_type=template.template_type.value,
                   name=template.name)

    def render_template(
        self,
        template_type: TemplateType,
        variables: dict[str, Any],
    ) -> RenderedTemplate:
        """Render a template by type with variables."""
        template = self.get_template(template_type)
        return self._engine.render(template, variables)

    def list_templates(self, active_only: bool = True) -> list[NotificationTemplate]:
        """List all registered templates."""
        templates = list(self._templates.values())
        if active_only:
            templates = [t for t in templates if t.is_active]
        return templates
