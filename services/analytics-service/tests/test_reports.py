"""
Unit tests for analytics reports module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from reports import (
    ReportType,
    ReportFormat,
    ReportPeriod,
    ReportStatus,
    ReportTimeRange,
    ReportSection,
    Report,
    ReportService,
    SessionSummaryReportGenerator,
    SafetyOverviewReportGenerator,
    ClinicalOutcomesReportGenerator,
    OperationalHealthReportGenerator,
)


class TestReportType:
    """Tests for ReportType enum."""

    def test_report_type_values(self):
        """Test report type enum values."""
        assert ReportType.SESSION_SUMMARY.value == "session_summary"
        assert ReportType.SAFETY_OVERVIEW.value == "safety_overview"
        assert ReportType.CLINICAL_OUTCOMES.value == "clinical_outcomes"
        assert ReportType.OPERATIONAL_HEALTH.value == "operational_health"


class TestReportTimeRange:
    """Tests for ReportTimeRange dataclass."""

    def test_for_last_hour(self):
        """Test creating last hour time range."""
        time_range = ReportTimeRange.for_last_hour()

        assert time_range.period == ReportPeriod.HOURLY
        assert (time_range.end - time_range.start).total_seconds() == 3600

    def test_for_last_day(self):
        """Test creating last day time range."""
        time_range = ReportTimeRange.for_last_day()

        assert time_range.period == ReportPeriod.DAILY
        assert abs((time_range.end - time_range.start).total_seconds() - 86400) < 1

    def test_for_last_week(self):
        """Test creating last week time range."""
        time_range = ReportTimeRange.for_last_week()

        assert time_range.period == ReportPeriod.WEEKLY
        assert abs((time_range.end - time_range.start).days - 7) < 1

    def test_for_last_month(self):
        """Test creating last month time range."""
        time_range = ReportTimeRange.for_last_month()

        assert time_range.period == ReportPeriod.MONTHLY
        assert abs((time_range.end - time_range.start).days - 30) < 1

    def test_custom_range(self):
        """Test creating custom time range."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 15, tzinfo=timezone.utc)

        time_range = ReportTimeRange.custom(start, end)

        assert time_range.start == start
        assert time_range.end == end
        assert time_range.period == ReportPeriod.CUSTOM

    def test_duration_hours(self):
        """Test duration calculation."""
        time_range = ReportTimeRange.for_last_day()

        assert abs(time_range.duration_hours - 24) < 0.01


class TestReportSection:
    """Tests for ReportSection model."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Test Section",
            description="A test section",
            data={"key": "value"},
        )

        assert section.title == "Test Section"
        assert section.data == {"key": "value"}


class TestReport:
    """Tests for Report model."""

    def test_report_creation(self):
        """Test creating a report."""
        now = datetime.now(timezone.utc)
        report = Report(
            report_type=ReportType.SESSION_SUMMARY,
            title="Test Report",
            description="A test report",
            time_range_start=now - timedelta(hours=1),
            time_range_end=now,
            period=ReportPeriod.HOURLY,
        )

        assert report.report_type == ReportType.SESSION_SUMMARY
        assert report.title == "Test Report"
        assert report.status == ReportStatus.COMPLETED

    def test_report_with_sections(self):
        """Test report with sections."""
        now = datetime.now(timezone.utc)
        section = ReportSection(title="Section 1", description="First section")

        report = Report(
            report_type=ReportType.SAFETY_OVERVIEW,
            title="Safety Report",
            time_range_start=now - timedelta(hours=1),
            time_range_end=now,
            period=ReportPeriod.HOURLY,
            sections=[section],
        )

        assert len(report.sections) == 1
        assert report.sections[0].title == "Section 1"


class TestSessionSummaryReportGenerator:
    """Tests for SessionSummaryReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create session summary generator."""
        return SessionSummaryReportGenerator()

    def test_report_type(self, generator):
        """Test generator report type."""
        assert generator.report_type == ReportType.SESSION_SUMMARY

    @pytest.mark.asyncio
    async def test_generate_report(self, generator, analytics_aggregator, time_range_last_hour, sample_user_id):
        """Test generating session summary report."""
        await analytics_aggregator.track_session_event(
            event_type="session.started",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={"duration_seconds": 300},
        )

        report = await generator.generate(analytics_aggregator, time_range_last_hour)

        assert report.report_type == ReportType.SESSION_SUMMARY
        assert len(report.sections) >= 1
        assert "total_sessions" in report.summary


class TestSafetyOverviewReportGenerator:
    """Tests for SafetyOverviewReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create safety overview generator."""
        return SafetyOverviewReportGenerator()

    def test_report_type(self, generator):
        """Test generator report type."""
        assert generator.report_type == ReportType.SAFETY_OVERVIEW

    @pytest.mark.asyncio
    async def test_generate_report(self, generator, analytics_aggregator, time_range_last_hour):
        """Test generating safety overview report."""
        await analytics_aggregator.track_safety_event(
            risk_level="LOW",
            detection_layer=1,
            metadata={},
        )

        report = await generator.generate(analytics_aggregator, time_range_last_hour)

        assert report.report_type == ReportType.SAFETY_OVERVIEW
        assert "total_checks" in report.summary


class TestClinicalOutcomesReportGenerator:
    """Tests for ClinicalOutcomesReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create clinical outcomes generator."""
        return ClinicalOutcomesReportGenerator()

    def test_report_type(self, generator):
        """Test generator report type."""
        assert generator.report_type == ReportType.CLINICAL_OUTCOMES

    @pytest.mark.asyncio
    async def test_generate_report(self, generator, analytics_aggregator, time_range_last_day):
        """Test generating clinical outcomes report."""
        await analytics_aggregator.track_diagnosis_event(
            assessment_type="diagnosis.completed",
            severity="MILD",
            stepped_care_level=2,
        )
        await analytics_aggregator.track_therapy_event(
            modality="CBT",
            technique="cognitive_restructuring",
            engagement_score=Decimal("0.8"),
        )

        report = await generator.generate(analytics_aggregator, time_range_last_day)

        assert report.report_type == ReportType.CLINICAL_OUTCOMES
        assert "total_diagnoses" in report.summary
        assert "total_interventions" in report.summary


class TestOperationalHealthReportGenerator:
    """Tests for OperationalHealthReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create operational health generator."""
        return OperationalHealthReportGenerator()

    def test_report_type(self, generator):
        """Test generator report type."""
        assert generator.report_type == ReportType.OPERATIONAL_HEALTH

    @pytest.mark.asyncio
    async def test_generate_report(self, generator, analytics_aggregator, time_range_last_hour, sample_user_id):
        """Test generating operational health report."""
        await analytics_aggregator.track_session_event(
            event_type="session.response.generated",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={"generation_time_ms": 150, "model_used": "gpt-4"},
        )

        report = await generator.generate(analytics_aggregator, time_range_last_hour)

        assert report.report_type == ReportType.OPERATIONAL_HEALTH
        assert "total_responses" in report.summary


class TestReportService:
    """Tests for ReportService."""

    @pytest.mark.asyncio
    async def test_generate_session_report(self, report_service, sample_user_id, analytics_aggregator):
        """Test generating a session summary report."""
        await analytics_aggregator.track_session_event(
            event_type="session.started",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={},
        )

        report = await report_service.generate_report(ReportType.SESSION_SUMMARY)

        assert report.report_type == ReportType.SESSION_SUMMARY

    @pytest.mark.asyncio
    async def test_generate_safety_report(self, report_service, analytics_aggregator):
        """Test generating a safety overview report."""
        await analytics_aggregator.track_safety_event(
            risk_level="LOW",
            detection_layer=1,
            metadata={},
        )

        report = await report_service.generate_report(ReportType.SAFETY_OVERVIEW)

        assert report.report_type == ReportType.SAFETY_OVERVIEW

    @pytest.mark.asyncio
    async def test_report_caching(self, report_service, analytics_aggregator, sample_user_id):
        """Test that reports are cached."""
        await analytics_aggregator.track_session_event(
            event_type="session.started",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={},
        )

        time_range = ReportTimeRange.for_last_hour()
        report1 = await report_service.generate_report(
            ReportType.SESSION_SUMMARY, time_range, use_cache=True
        )
        report2 = await report_service.generate_report(
            ReportType.SESSION_SUMMARY, time_range, use_cache=True
        )

        assert report1.report_id == report2.report_id

        stats = await report_service.get_statistics()
        assert stats["cache_hits"] >= 1

    @pytest.mark.asyncio
    async def test_get_available_report_types(self, report_service):
        """Test getting available report types."""
        types = await report_service.get_available_report_types()

        assert ReportType.SESSION_SUMMARY in types
        assert ReportType.SAFETY_OVERVIEW in types
        assert ReportType.CLINICAL_OUTCOMES in types
        assert ReportType.OPERATIONAL_HEALTH in types

    @pytest.mark.asyncio
    async def test_clear_cache(self, report_service):
        """Test clearing the report cache."""
        report_service.clear_cache()

        stats = await report_service.get_statistics()
        assert stats["cached_reports"] == 0

    @pytest.mark.asyncio
    async def test_invalid_report_type(self, report_service):
        """Test that invalid report type raises error."""
        with pytest.raises(ValueError):
            await report_service.generate_report(ReportType.COMPLIANCE_AUDIT)
