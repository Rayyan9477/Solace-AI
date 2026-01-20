"""
Unit tests for notification templates module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from uuid import uuid4

from domain.templates import (
    TemplateType,
    NotificationTemplate,
    TemplateRegistry,
    TemplateEngine,
    RenderedTemplate,
    TemplateNotFoundError,
    TemplateRenderError,
)


class TestNotificationTemplate:
    """Tests for NotificationTemplate model."""

    def test_create_template_with_required_fields(self):
        """Test creating a template with required fields."""
        template = NotificationTemplate(
            template_type=TemplateType.WELCOME,
            name="Test Welcome",
            subject_template="Welcome {{ name }}",
            body_template="Hello {{ name }}, welcome!",
        )
        assert template.template_type == TemplateType.WELCOME
        assert template.name == "Test Welcome"
        assert template.is_active is True

    def test_create_template_with_all_fields(self):
        """Test creating a template with all fields."""
        template = NotificationTemplate(
            template_type=TemplateType.EMAIL_VERIFICATION,
            name="Email Verification",
            description="Verify user email",
            subject_template="Verify your email",
            body_template="Click {{ link }}",
            html_template="<a href='{{ link }}'>Verify</a>",
            required_variables=["link"],
            default_values={"expiry": "24 hours"},
            version="2.0",
            is_active=True,
        )
        assert template.required_variables == ["link"]
        assert template.default_values == {"expiry": "24 hours"}
        assert template.version == "2.0"

    def test_required_variables_default_empty(self):
        """Test that required_variables defaults to empty list."""
        template = NotificationTemplate(
            template_type=TemplateType.WELCOME,
            name="Test",
            subject_template="Subject",
            body_template="Body",
        )
        assert template.required_variables == []


class TestTemplateEngine:
    """Tests for TemplateEngine."""

    @pytest.fixture
    def engine(self):
        return TemplateEngine()

    @pytest.fixture
    def simple_template(self):
        return NotificationTemplate(
            template_type=TemplateType.WELCOME,
            name="Simple Template",
            subject_template="Hello {{ name }}",
            body_template="Welcome to {{ service }}, {{ name }}!",
            required_variables=["name", "service"],
        )

    def test_render_template_success(self, engine, simple_template):
        """Test successful template rendering."""
        result = engine.render(simple_template, {"name": "John", "service": "Solace-AI"})
        assert isinstance(result, RenderedTemplate)
        assert result.subject == "Hello John"
        assert result.body == "Welcome to Solace-AI, John!"
        assert result.html_body is None

    def test_render_template_with_html(self, engine):
        """Test rendering template with HTML body."""
        template = NotificationTemplate(
            template_type=TemplateType.EMAIL_VERIFICATION,
            name="HTML Template",
            subject_template="Verify Email",
            body_template="Click: {{ link }}",
            html_template="<a href='{{ link }}'>Click here</a>",
            required_variables=["link"],
        )
        result = engine.render(template, {"link": "https://example.com"})
        assert result.html_body == "<a href='https://example.com'>Click here</a>"

    def test_render_template_with_defaults(self, engine):
        """Test rendering with default values."""
        template = NotificationTemplate(
            template_type=TemplateType.WELCOME,
            name="Defaults Template",
            subject_template="Hello {{ name }}",
            body_template="Expiry: {{ expiry }}",
            required_variables=["name", "expiry"],
            default_values={"expiry": "24 hours"},
        )
        result = engine.render(template, {"name": "John"})
        assert result.body == "Expiry: 24 hours"

    def test_render_template_missing_required_variables(self, engine, simple_template):
        """Test rendering fails with missing required variables."""
        with pytest.raises(TemplateRenderError) as exc_info:
            engine.render(simple_template, {"name": "John"})
        assert "Missing required variables" in str(exc_info.value)
        assert "service" in str(exc_info.value)

    def test_render_template_override_defaults(self, engine):
        """Test provided values override defaults."""
        template = NotificationTemplate(
            template_type=TemplateType.WELCOME,
            name="Override Template",
            subject_template="Hello",
            body_template="Expiry: {{ expiry }}",
            required_variables=["expiry"],
            default_values={"expiry": "24 hours"},
        )
        result = engine.render(template, {"expiry": "1 hour"})
        assert result.body == "Expiry: 1 hour"


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    @pytest.fixture
    def registry(self):
        return TemplateRegistry()

    def test_registry_has_default_templates(self, registry):
        """Test that registry initializes with default templates."""
        templates = registry.list_templates()
        assert len(templates) > 0
        template_types = [t.template_type for t in templates]
        assert TemplateType.EMAIL_VERIFICATION in template_types
        assert TemplateType.PASSWORD_RESET in template_types
        assert TemplateType.WELCOME in template_types

    def test_get_template_success(self, registry):
        """Test getting an existing template."""
        template = registry.get_template(TemplateType.EMAIL_VERIFICATION)
        assert template.template_type == TemplateType.EMAIL_VERIFICATION
        assert "display_name" in template.required_variables

    def test_get_template_not_found(self, registry):
        """Test getting a non-existent template raises error."""
        registry._templates.clear()
        with pytest.raises(TemplateNotFoundError):
            registry.get_template(TemplateType.EMAIL_VERIFICATION)

    def test_register_custom_template(self, registry):
        """Test registering a custom template."""
        custom = NotificationTemplate(
            template_type=TemplateType.SYSTEM_ALERT,
            name="Custom Alert",
            subject_template="Alert: {{ title }}",
            body_template="{{ message }}",
            required_variables=["title", "message"],
        )
        registry.register_template(custom)
        retrieved = registry.get_template(TemplateType.SYSTEM_ALERT)
        assert retrieved.name == "Custom Alert"

    def test_render_template_via_registry(self, registry):
        """Test rendering through registry."""
        result = registry.render_template(
            TemplateType.WELCOME,
            {
                "display_name": "John",
                "getting_started_link": "https://app.solace-ai.com",
            },
        )
        assert "John" in result.subject
        assert "https://app.solace-ai.com" in result.body

    def test_list_active_templates_only(self, registry):
        """Test listing only active templates."""
        inactive = NotificationTemplate(
            template_type=TemplateType.SYSTEM_ALERT,
            name="Inactive",
            subject_template="Test",
            body_template="Test",
            is_active=False,
        )
        registry.register_template(inactive)

        active_templates = registry.list_templates(active_only=True)
        all_templates = registry.list_templates(active_only=False)

        assert len(all_templates) > len(active_templates)

    def test_all_default_templates_have_required_vars(self, registry):
        """Test all default templates have documented required variables."""
        for template in registry.list_templates():
            assert isinstance(template.required_variables, list)
            for var in template.required_variables:
                assert isinstance(var, str)
                assert len(var) > 0


class TestTemplateTypes:
    """Tests for TemplateType enum."""

    def test_all_template_types_are_strings(self):
        """Test all template types have string values."""
        for template_type in TemplateType:
            assert isinstance(template_type.value, str)

    def test_template_types_are_unique(self):
        """Test all template type values are unique."""
        values = [t.value for t in TemplateType]
        assert len(values) == len(set(values))

    def test_expected_template_types_exist(self):
        """Test expected template types are defined."""
        expected = [
            "email_verification",
            "password_reset",
            "welcome",
            "clinician_alert",
            "crisis_escalation",
        ]
        actual = [t.value for t in TemplateType]
        for expected_type in expected:
            assert expected_type in actual
