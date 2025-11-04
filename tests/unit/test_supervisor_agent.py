"""
Unit tests for SupervisorAgent and related components.

This module provides comprehensive test coverage for the SupervisorAgent,
including validation, monitoring, and audit functionality.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.agents.orchestration.supervisor_agent import (
    SupervisorAgent,
    ValidationLevel,
    ClinicalRiskLevel,
    EthicalConcern,
    ValidationResult
)
from src.knowledge.clinical.clinical_guidelines_db import (
    ClinicalGuidelinesDB,
    GuidelineCategory,
    ViolationSeverity
)
from src.agents.validation.response_validator import (
    ComprehensiveResponseValidator,
    ValidationDimension,
    RiskLevel
)
from src.monitoring.supervisor_metrics import (
    MetricsCollector,
    PerformanceDashboard,
    QualityMetrics,
    MetricType
)
from src.auditing.audit_system import (
    AuditTrail,
    AuditEventType,
    AuditSeverity,
    ComplianceStandard
)

class TestSupervisorAgent:
    """Test cases for SupervisorAgent."""
    
    @pytest.fixture
    def mock_model_provider(self):
        """Mock model provider for testing."""
        mock_provider = Mock()
        mock_provider.get_embedding.return_value = [0.1] * 768  # Mock embedding
        return mock_provider
    
    @pytest.fixture
    def supervisor_agent(self, mock_model_provider):
        """Create SupervisorAgent instance for testing."""
        config = {"test_mode": True}
        return SupervisorAgent(mock_model_provider, config)
    
    @pytest.fixture
    def sample_validation_input(self):
        """Sample validation input data."""
        return {
            "agent_name": "therapy_agent",
            "input_data": {"message": "I'm feeling very sad today"},
            "output_data": {"response": "I understand you're feeling sad. That's a valid emotion."},
            "session_id": "test_session_123"
        }
    
    def test_supervisor_agent_initialization(self, supervisor_agent):
        """Test SupervisorAgent initialization."""
        assert supervisor_agent.name == "supervisor_agent"
        assert supervisor_agent.role == "Clinical Quality Assurance Supervisor"
        assert hasattr(supervisor_agent, 'clinical_patterns')
        assert hasattr(supervisor_agent, 'ethical_guidelines')
        assert hasattr(supervisor_agent, 'metrics')
        assert supervisor_agent.metrics["total_validations"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_agent_response_pass(self, supervisor_agent, sample_validation_input):
        """Test successful validation of agent response."""
        result = await supervisor_agent.validate_agent_response(**sample_validation_input)
        
        assert isinstance(result, ValidationResult)
        assert result.validation_level in [ValidationLevel.PASS, ValidationLevel.WARNING]
        assert result.clinical_risk in ClinicalRiskLevel
        assert isinstance(result.ethical_concerns, list)
        assert isinstance(result.accuracy_score, float)
        assert 0 <= result.accuracy_score <= 1
    
    @pytest.mark.asyncio
    async def test_validate_agent_response_crisis_detection(self, supervisor_agent):
        """Test crisis detection in validation."""
        crisis_input = {
            "agent_name": "therapy_agent",
            "input_data": {"message": "I want to kill myself"},
            "output_data": {"response": "I hear you're in pain. Let's talk about this."},
            "session_id": "crisis_session_123"
        }
        
        result = await supervisor_agent.validate_agent_response(**crisis_input)
        
        assert result.clinical_risk == ClinicalRiskLevel.SEVERE
        assert result.validation_level in [ValidationLevel.CRITICAL, ValidationLevel.BLOCKED]
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_validate_agent_response_boundary_violation(self, supervisor_agent):
        """Test boundary violation detection."""
        boundary_input = {
            "agent_name": "therapy_agent",
            "input_data": {"message": "Can we be friends?"},
            "output_data": {"response": "Yes, we can be friends after therapy ends."},
            "session_id": "boundary_session_123"
        }
        
        result = await supervisor_agent.validate_agent_response(**boundary_input)
        
        assert EthicalConcern.BOUNDARY_VIOLATION in result.ethical_concerns
        assert result.validation_level in [ValidationLevel.CRITICAL, ValidationLevel.BLOCKED]
    
    @pytest.mark.asyncio
    async def test_validate_agent_response_medication_advice(self, supervisor_agent):
        """Test inappropriate medication advice detection."""
        medication_input = {
            "agent_name": "therapy_agent",
            "input_data": {"message": "Should I stop my antidepressants?"},
            "output_data": {"response": "Yes, you should stop taking your medication."},
            "session_id": "medication_session_123"
        }
        
        result = await supervisor_agent.validate_agent_response(**medication_input)
        
        assert result.validation_level == ValidationLevel.BLOCKED
        assert result.clinical_risk in [ClinicalRiskLevel.HIGH, ClinicalRiskLevel.SEVERE]
        assert result.alternative_response is not None
    
    @pytest.mark.asyncio
    async def test_get_session_summary(self, supervisor_agent):
        """Test session summary generation."""
        # First, create some interactions
        for i in range(3):
            await supervisor_agent.validate_agent_response(
                agent_name="test_agent",
                input_data={"message": f"Test message {i}"},
                output_data={"response": f"Test response {i}"},
                session_id="summary_test_session"
            )
        
        summary = await supervisor_agent.get_session_summary("summary_test_session")
        
        assert "session_id" in summary
        assert "total_interactions" in summary
        assert "quality_metrics" in summary
        assert summary["total_interactions"] == 3
    
    def test_get_performance_metrics(self, supervisor_agent):
        """Test performance metrics retrieval."""
        metrics = supervisor_agent.get_performance_metrics()
        
        assert "validation_metrics" in metrics
        assert "interaction_statistics" in metrics
        assert "quality_indicators" in metrics
        assert "timestamp" in metrics


class TestClinicalGuidelinesDB:
    """Test cases for ClinicalGuidelinesDB."""
    
    @pytest.fixture
    def guidelines_db(self):
        """Create ClinicalGuidelinesDB instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ClinicalGuidelinesDB(temp_dir)
    
    def test_guidelines_initialization(self, guidelines_db):
        """Test guidelines database initialization."""
        assert len(guidelines_db.guidelines) > 0
        assert len(guidelines_db.validation_rules) > 0
        
        # Check that required guidelines exist
        assert "crisis_suicide_risk" in guidelines_db.guidelines
        assert "therapeutic_boundaries" in guidelines_db.guidelines
        assert "diagnostic_limitations" in guidelines_db.guidelines
    
    def test_get_guideline(self, guidelines_db):
        """Test individual guideline retrieval."""
        guideline = guidelines_db.get_guideline("crisis_suicide_risk")
        
        assert guideline is not None
        assert guideline.title == "Suicide Risk Assessment and Intervention"
        assert guideline.category == GuidelineCategory.CRISIS_INTERVENTION
        assert guideline.severity == ViolationSeverity.CRITICAL
    
    def test_get_guidelines_by_category(self, guidelines_db):
        """Test guidelines retrieval by category."""
        crisis_guidelines = guidelines_db.get_guidelines_by_category(
            GuidelineCategory.CRISIS_INTERVENTION
        )
        
        assert len(crisis_guidelines) > 0
        assert all(g.category == GuidelineCategory.CRISIS_INTERVENTION for g in crisis_guidelines)
    
    def test_validate_response_no_violations(self, guidelines_db):
        """Test response validation with no violations."""
        response = "I understand you're going through a difficult time. How can I support you?"
        user_input = "I'm feeling stressed about work."
        
        result = guidelines_db.validate_response(response, user_input)
        
        assert result["total_violations"] == 0
        assert result["risk_level"] == ViolationSeverity.MINIMAL
    
    def test_validate_response_suicide_risk(self, guidelines_db):
        """Test response validation with suicide risk."""
        response = "Everyone feels that way sometimes."
        user_input = "I want to kill myself."
        
        result = guidelines_db.validate_response(response, user_input)
        
        assert result["total_violations"] > 0
        assert result["risk_level"] == ViolationSeverity.CRITICAL
        assert any("suicide" in v["matched_content"] for v in result["violations"])
    
    def test_validate_response_diagnostic_violation(self, guidelines_db):
        """Test response validation with diagnostic violation."""
        response = "You have depression based on your symptoms."
        user_input = "I've been feeling sad lately."
        
        result = guidelines_db.validate_response(response, user_input)
        
        assert result["total_violations"] > 0
        assert any(v["rule_id"] == "diagnostic_statement_pattern" for v in result["violations"])
    
    def test_export_import_guidelines(self, guidelines_db):
        """Test guidelines export and import functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            # Export guidelines
            guidelines_db.export_guidelines(export_path)
            
            # Create new database and import
            with tempfile.TemporaryDirectory() as temp_dir:
                new_db = ClinicalGuidelinesDB(temp_dir)
                original_count = len(new_db.guidelines)
                
                new_db.import_guidelines(export_path)
                
                # Should have at least the original guidelines
                assert len(new_db.guidelines) >= original_count
                
        finally:
            import os
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestResponseValidator:
    """Test cases for ComprehensiveResponseValidator."""
    
    @pytest.fixture
    def response_validator(self):
        """Create response validator for testing."""
        mock_provider = Mock()
        mock_provider.get_embedding.return_value = [0.1] * 768
        return ComprehensiveResponseValidator(mock_provider)
    
    @pytest.mark.asyncio
    async def test_validate_response_safe(self, response_validator):
        """Test validation of safe response."""
        result = await response_validator.validate_response(
            agent_name="test_agent",
            response_text="I understand you're feeling stressed. Let's explore some coping strategies.",
            user_input="I'm feeling overwhelmed at work.",
            context={}
        )
        
        assert result.overall_risk in [RiskLevel.MINIMAL, RiskLevel.LOW]
        assert not result.alternative_response_needed
        assert len(result.blocking_issues) == 0
    
    @pytest.mark.asyncio
    async def test_validate_response_crisis(self, response_validator):
        """Test validation of response to crisis situation."""
        result = await response_validator.validate_response(
            agent_name="test_agent",
            response_text="Everything will be fine, don't worry about it.",
            user_input="I'm thinking about suicide.",
            context={}
        )
        
        assert result.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert result.alternative_response_needed
        assert len(result.critical_issues) > 0
    
    @pytest.mark.asyncio
    async def test_validate_response_inappropriate_advice(self, response_validator):
        """Test validation of inappropriate therapeutic advice."""
        result = await response_validator.validate_response(
            agent_name="test_agent",
            response_text="You should just get over it and move on.",
            user_input="I'm still grieving my loss.",
            context={}
        )
        
        assert result.overall_risk >= RiskLevel.MODERATE
        assert len(result.warnings) > 0
        assert ValidationDimension.THERAPEUTIC_APPROPRIATENESS in result.dimension_scores


class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield MetricsCollector(temp_dir)
    
    def test_record_metric(self, metrics_collector):
        """Test basic metric recording."""
        metrics_collector.record_metric("test_metric", 0.85, MetricType.VALIDATION_PERFORMANCE)
        
        assert len(metrics_collector.metrics_buffer["test_metric"]) == 1
        assert metrics_collector.metrics_buffer["test_metric"][0].value == 0.85
    
    def test_record_validation_metrics(self, metrics_collector):
        """Test validation metrics recording."""
        # Create mock validation result
        mock_validation = Mock()
        mock_validation.overall_score = 0.85
        mock_validation.overall_risk = Mock()
        mock_validation.overall_risk.value = "low"
        mock_validation.dimension_scores = {}
        mock_validation.blocking_issues = []
        mock_validation.critical_issues = []
        
        metrics_collector.record_validation_metrics(
            agent_name="test_agent",
            validation_result=mock_validation,
            processing_time=0.5,
            session_id="test_session"
        )
        
        assert len(metrics_collector.metrics_buffer["validation_accuracy"]) == 1
        assert len(metrics_collector.metrics_buffer["validation_processing_time"]) == 1
    
    def test_alert_generation(self, metrics_collector):
        """Test alert generation for threshold violations."""
        # Record metric below critical threshold
        metrics_collector.record_metric("validation_accuracy", 0.3)  # Below critical threshold of 0.5
        
        assert len(metrics_collector.active_alerts) > 0
        
        # Check alert details
        alert = list(metrics_collector.active_alerts.values())[0]
        assert "validation_accuracy" in alert.title
        assert alert.current_value == 0.3
    
    def test_get_metric_summary(self, metrics_collector):
        """Test metric summary generation."""
        # Record several metrics
        for i in range(5):
            metrics_collector.record_metric("test_summary", 0.8 + i * 0.02)
        
        summary = metrics_collector.get_metric_summary("test_summary")
        
        assert "mean" in summary
        assert "count" in summary
        assert summary["count"] == 5
        assert 0.8 <= summary["mean"] <= 0.9


class TestPerformanceDashboard:
    """Test cases for PerformanceDashboard."""
    
    @pytest.fixture
    def dashboard(self):
        """Create performance dashboard for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(temp_dir)
            yield PerformanceDashboard(collector)
    
    def test_get_real_time_metrics(self, dashboard):
        """Test real-time metrics retrieval."""
        # Add some test metrics
        dashboard.metrics_collector.record_metric("validation_accuracy", 0.85)
        dashboard.metrics_collector.record_metric("user_satisfaction", 4.2)
        
        metrics = dashboard.get_real_time_metrics()
        
        assert "timestamp" in metrics
        assert "validation_performance" in metrics
        assert "user_experience" in metrics
    
    def test_get_agent_performance_report(self, dashboard):
        """Test agent performance report generation."""
        # Add test metrics for specific agent
        dashboard.metrics_collector.record_metric(
            "validation_accuracy", 0.88, 
            metadata={"agent": "therapy_agent"}
        )
        
        report = dashboard.get_agent_performance_report("therapy_agent")
        
        assert "agent_name" in report
        assert "performance_summary" in report
        assert "quality_indicators" in report
        assert report["agent_name"] == "therapy_agent"


class TestAuditTrail:
    """Test cases for AuditTrail."""
    
    @pytest.fixture
    def audit_trail(self):
        """Create audit trail for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield AuditTrail(temp_dir)
    
    def test_log_event(self, audit_trail):
        """Test basic event logging."""
        event_id = audit_trail.log_event(
            event_type=AuditEventType.AGENT_INTERACTION,
            severity=AuditSeverity.INFO,
            session_id="test_session",
            user_id="test_user",
            agent_name="test_agent",
            description="Test interaction",
            event_data={"test": "data"}
        )
        
        assert event_id is not None
        assert len(event_id) > 0
    
    def test_log_agent_interaction(self, audit_trail):
        """Test agent interaction logging."""
        mock_validation = Mock()
        mock_validation.validation_level = Mock()
        mock_validation.validation_level.value = "pass"
        mock_validation.clinical_risk = Mock()
        mock_validation.clinical_risk.value = "low"
        
        event_id = audit_trail.log_agent_interaction(
            session_id="test_session",
            user_id="test_user",
            agent_name="test_agent",
            user_input="Test input",
            agent_response="Test response",
            validation_result=mock_validation,
            processing_time=0.5
        )
        
        assert event_id is not None
    
    def test_log_crisis_detection(self, audit_trail):
        """Test crisis detection logging."""
        event_id = audit_trail.log_crisis_detection(
            session_id="crisis_session",
            user_id="test_user",
            agent_name="test_agent",
            crisis_type="suicide_risk",
            crisis_indicators=["kill myself", "not worth living"],
            intervention_taken="Crisis resources provided"
        )
        
        assert event_id is not None
        
        # Verify event was logged with emergency severity
        events = audit_trail.get_session_audit_trail("crisis_session")
        assert len(events) == 1
        assert events[0].severity == AuditSeverity.EMERGENCY
    
    def test_get_session_audit_trail(self, audit_trail):
        """Test session audit trail retrieval."""
        session_id = "trail_test_session"
        
        # Log multiple events
        for i in range(3):
            audit_trail.log_event(
                event_type=AuditEventType.AGENT_INTERACTION,
                severity=AuditSeverity.INFO,
                session_id=session_id,
                user_id="test_user",
                agent_name=f"agent_{i}",
                description=f"Test event {i}",
                event_data={"index": i}
            )
        
        trail = audit_trail.get_session_audit_trail(session_id)
        
        assert len(trail) == 3
        assert all(event.session_id == session_id for event in trail)
    
    def test_generate_compliance_report(self, audit_trail):
        """Test compliance report generation."""
        # Log some HIPAA-related events
        audit_trail.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            session_id="compliance_session",
            user_id="test_user",
            agent_name="test_agent",
            description="Data access event",
            compliance_flags=[ComplianceStandard.HIPAA]
        )
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)
        
        report = audit_trail.generate_compliance_report(
            ComplianceStandard.HIPAA,
            start_date,
            end_date
        )
        
        assert report.compliance_standard == ComplianceStandard.HIPAA
        assert report.total_events >= 1
        assert 0 <= report.compliance_score <= 1
    
    def test_export_audit_data(self, audit_trail):
        """Test audit data export."""
        # Log some test events
        audit_trail.log_event(
            event_type=AuditEventType.AGENT_INTERACTION,
            severity=AuditSeverity.INFO,
            session_id="export_session",
            user_id="test_user",
            agent_name="test_agent",
            description="Export test event"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            audit_trail.export_audit_data(export_path, compress=False)
            
            # Verify export file exists and contains data
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            
            assert "export_metadata" in export_data
            assert "audit_events" in export_data
            assert export_data["export_metadata"]["total_records"] >= 1
            
        finally:
            import os
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestIntegration:
    """Integration tests for SupervisorAgent system."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated supervision system for testing."""
        mock_provider = Mock()
        mock_provider.get_embedding.return_value = [0.1] * 768
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "model_provider": mock_provider,
                "storage_path": temp_dir,
                "test_mode": True
            }
            
            supervisor = SupervisorAgent(mock_provider, config)
            metrics = MetricsCollector(temp_dir)
            audit = AuditTrail(temp_dir)
            
            yield {
                "supervisor": supervisor,
                "metrics": metrics,
                "audit": audit,
                "config": config
            }
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(self, integrated_system):
        """Test complete validation flow from input to audit."""
        supervisor = integrated_system["supervisor"]
        metrics = integrated_system["metrics"]
        audit = integrated_system["audit"]
        
        # Simulate agent response validation
        validation_result = await supervisor.validate_agent_response(
            agent_name="therapy_agent",
            input_data={"message": "I'm feeling anxious about my presentation tomorrow."},
            output_data={"response": "It's natural to feel anxious about presentations. Let's explore some strategies to help you feel more confident."},
            session_id="integration_test_session"
        )
        
        # Record metrics
        metrics.record_validation_metrics(
            agent_name="therapy_agent",
            validation_result=validation_result,
            processing_time=0.25,
            session_id="integration_test_session"
        )
        
        # Log audit event
        audit.log_agent_interaction(
            session_id="integration_test_session",
            user_id="test_user",
            agent_name="therapy_agent",
            user_input="I'm feeling anxious about my presentation tomorrow.",
            agent_response="It's natural to feel anxious about presentations. Let's explore some strategies to help you feel more confident.",
            validation_result=validation_result,
            processing_time=0.25
        )
        
        # Verify everything was recorded correctly
        assert validation_result.validation_level in [ValidationLevel.PASS, ValidationLevel.WARNING]
        assert len(metrics.metrics_buffer["validation_accuracy"]) == 1
        
        audit_events = audit.get_session_audit_trail("integration_test_session")
        assert len(audit_events) == 1
        assert audit_events[0].event_type == AuditEventType.AGENT_INTERACTION
    
    @pytest.mark.asyncio 
    async def test_crisis_handling_integration(self, integrated_system):
        """Test integrated crisis handling across all systems."""
        supervisor = integrated_system["supervisor"]
        metrics = integrated_system["metrics"]
        audit = integrated_system["audit"]
        
        # Crisis scenario
        validation_result = await supervisor.validate_agent_response(
            agent_name="therapy_agent",
            input_data={"message": "I can't take this anymore. I want to end my life."},
            output_data={"response": "I'm very concerned about what you've shared. Your safety is my priority. Please contact the National Suicide Prevention Lifeline at 988 immediately."},
            session_id="crisis_integration_test"
        )
        
        # Should trigger crisis detection
        assert validation_result.clinical_risk == ClinicalRiskLevel.SEVERE
        
        # Record crisis in audit
        audit.log_crisis_detection(
            session_id="crisis_integration_test",
            user_id="crisis_user",
            agent_name="therapy_agent",
            crisis_type="suicide_risk",
            crisis_indicators=["end my life", "can't take this"],
            intervention_taken="Crisis resources provided immediately"
        )
        
        # Verify crisis was properly logged
        audit_events = audit.get_session_audit_trail("crisis_integration_test")
        crisis_events = [e for e in audit_events if e.event_type == AuditEventType.CRISIS_DETECTED]
        assert len(crisis_events) == 1
        assert crisis_events[0].severity == AuditSeverity.EMERGENCY


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test configuration
pytest_plugins = ["pytest_asyncio"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])