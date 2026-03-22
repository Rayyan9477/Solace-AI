"""
Comprehensive syntax and import verification for Solace-AI.

Validates that every Python source file parses without syntax errors and that
all core library modules resolve their imports cleanly.  This acts as a fast
smoke-test gate -- if any file has a typo, circular import, or missing
dependency the relevant test will fail with a clear diagnostic message.

Test classes
------------
TestSyntaxParsing
    Uses ``ast.parse()`` to verify every ``.py`` file under ``src/`` and
    ``services/`` compiles without ``SyntaxError``.

TestImportResolution
    Uses ``importlib.import_module()`` to verify the five core shared
    libraries (common, events, infrastructure, security, testing) and their
    sub-modules can be imported.

TestEventExports
    Confirms that all 22+ Phase 0B event classes are exported from
    ``solace_events`` and are importable by name.

TestTopicConfiguration
    Validates key enum members (``SolaceTopic``, ``TherapyModality``) and
    the internal ``_TOPIC_MAP`` routing table.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SERVICES_DIR = PROJECT_ROOT / "services"


# =========================================================================
# 1. Syntax Parsing
# =========================================================================


class TestSyntaxParsing:
    """Verify that every .py file in src/ and services/ parses without SyntaxError."""

    @staticmethod
    def _collect_python_files(*directories: Path) -> list[Path]:
        """Recursively collect all .py files from the given directories."""
        files: list[Path] = []
        for directory in directories:
            if directory.is_dir():
                files.extend(sorted(directory.rglob("*.py")))
        return files

    # -- src/ ---------------------------------------------------------------

    def test_all_src_files_parse(self) -> None:
        """Every Python file under src/ must be valid syntax."""
        py_files = self._collect_python_files(SRC_DIR)
        assert py_files, f"No .py files found under {SRC_DIR}"

        errors: list[str] = []
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                errors.append(
                    f"{py_file.relative_to(PROJECT_ROOT)}: "
                    f"line {exc.lineno} -- {exc.msg}"
                )

        assert not errors, (
            f"Syntax errors found in {len(errors)} file(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # -- services/ ----------------------------------------------------------

    def test_all_services_files_parse(self) -> None:
        """Every Python file under services/ must be valid syntax."""
        py_files = self._collect_python_files(SERVICES_DIR)
        assert py_files, f"No .py files found under {SERVICES_DIR}"

        errors: list[str] = []
        for py_file in py_files:
            try:
                source = py_file.read_text(encoding="utf-8")
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                errors.append(
                    f"{py_file.relative_to(PROJECT_ROOT)}: "
                    f"line {exc.lineno} -- {exc.msg}"
                )

        assert not errors, (
            f"Syntax errors found in {len(errors)} file(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_src_directory_exists(self) -> None:
        """The src/ directory must exist and contain Python files."""
        assert SRC_DIR.is_dir(), f"src/ directory not found at {SRC_DIR}"
        assert list(SRC_DIR.rglob("*.py")), "src/ contains no .py files"

    def test_services_directory_exists(self) -> None:
        """The services/ directory must exist and contain Python files."""
        assert SERVICES_DIR.is_dir(), f"services/ directory not found at {SERVICES_DIR}"
        assert list(SERVICES_DIR.rglob("*.py")), "services/ contains no .py files"


# =========================================================================
# 2. Import Resolution
# =========================================================================


class TestImportResolution:
    """Verify core library modules and their sub-modules import without error."""

    # -- solace_common -------------------------------------------------------

    @pytest.mark.parametrize(
        "module_name",
        [
            "solace_common",
            "solace_common.utils",
            "solace_common.enums",
            "solace_common.exceptions",
        ],
        ids=lambda m: m,
    )
    def test_solace_common_imports(self, module_name: str) -> None:
        """solace_common and its sub-modules must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"

    # -- solace_events -------------------------------------------------------

    @pytest.mark.parametrize(
        "module_name",
        [
            "solace_events",
            "solace_events.schemas",
            "solace_events.config",
        ],
        ids=lambda m: m,
    )
    def test_solace_events_imports(self, module_name: str) -> None:
        """solace_events and its sub-modules must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"

    # -- solace_infrastructure -----------------------------------------------

    @pytest.mark.parametrize(
        "module_name",
        [
            "solace_infrastructure",
            "solace_infrastructure.postgres",
            "solace_infrastructure.health",
        ],
        ids=lambda m: m,
    )
    def test_solace_infrastructure_imports(self, module_name: str) -> None:
        """solace_infrastructure and its sub-modules must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"

    # -- solace_security -----------------------------------------------------

    @pytest.mark.parametrize(
        "module_name",
        [
            "solace_security",
            "solace_security.middleware",
            "solace_security.auth",
            "solace_security.audit",
            "solace_security.encryption",
        ],
        ids=lambda m: m,
    )
    def test_solace_security_imports(self, module_name: str) -> None:
        """solace_security and its sub-modules must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"

    # -- solace_testing ------------------------------------------------------

    @pytest.mark.parametrize(
        "module_name",
        [
            "solace_testing",
            "solace_testing.fixtures",
            "solace_testing.mocks",
            "solace_testing.factories",
        ],
        ids=lambda m: m,
    )
    def test_solace_testing_imports(self, module_name: str) -> None:
        """solace_testing and its sub-modules must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


# =========================================================================
# 3. Event Exports
# =========================================================================


class TestEventExports:
    """Verify all Phase 0B event classes are exported from solace_events."""

    PHASE_0B_EVENT_CLASSES: list[str] = [
        # Safety domain events
        "CrisisResolvedEvent",
        "EscalationAcknowledgedEvent",
        "EscalationResolvedEvent",
        "SafetyPlanCreatedEvent",
        "SafetyPlanActivatedEvent",
        "SafetyPlanUpdatedEvent",
        "OutputFilteredEvent",
        "TrajectoryAlertEvent",
        # Memory domain events
        "MemoryDecayedEvent",
        "ContextAssembledEvent",
        # Personality domain events
        "PersonalityProfileUpdatedEvent",
        "PersonalityTraitChangedEvent",
        # Notification domain events
        "NotificationSentKafkaEvent",
        "NotificationDeliveredKafkaEvent",
        "NotificationFailedKafkaEvent",
        # User domain events
        "UserCreatedKafkaEvent",
        "UserDeletedKafkaEvent",
        "ConsentChangedKafkaEvent",
        # Homework & treatment events
        "HomeworkAssignedEvent",
        "TreatmentResponseEvent",
    ]

    def test_minimum_event_count(self) -> None:
        """solace_events must export at least 22 Phase 0B event classes."""
        import solace_events

        exported = [
            name
            for name in dir(solace_events)
            if name.endswith("Event") and not name.startswith("_")
        ]
        assert len(exported) >= 22, (
            f"Expected at least 22 event classes, found {len(exported)}: "
            f"{sorted(exported)}"
        )

    @pytest.mark.parametrize(
        "event_class_name",
        PHASE_0B_EVENT_CLASSES,
        ids=lambda name: name,
    )
    def test_event_class_exported(self, event_class_name: str) -> None:
        """Each Phase 0B event class must be importable from solace_events."""
        import solace_events

        assert hasattr(solace_events, event_class_name), (
            f"{event_class_name} is not exported from solace_events. "
            f"Check __init__.py imports and __all__."
        )
        cls = getattr(solace_events, event_class_name)
        assert isinstance(cls, type), (
            f"{event_class_name} is exported but is not a class "
            f"(got {type(cls).__name__})"
        )

    def test_event_classes_in_all(self) -> None:
        """All Phase 0B event classes must appear in solace_events.__all__."""
        import solace_events

        all_exports = set(solace_events.__all__)
        missing = [
            name
            for name in self.PHASE_0B_EVENT_CLASSES
            if name not in all_exports
        ]
        assert not missing, (
            f"Event classes missing from solace_events.__all__: {missing}"
        )

    def test_event_classes_are_base_event_subclasses(self) -> None:
        """All Phase 0B event classes must be subclasses of BaseEvent."""
        from solace_events import BaseEvent
        import solace_events

        non_subclasses: list[str] = []
        for name in self.PHASE_0B_EVENT_CLASSES:
            cls = getattr(solace_events, name)
            if not issubclass(cls, BaseEvent):
                non_subclasses.append(name)

        assert not non_subclasses, (
            f"Event classes not inheriting from BaseEvent: {non_subclasses}"
        )


# =========================================================================
# 4. Topic Configuration
# =========================================================================


class TestTopicConfiguration:
    """Verify topic enums, therapy modalities, and the internal topic map."""

    # -- SolaceTopic ---------------------------------------------------------

    def test_solace_topic_has_notifications(self) -> None:
        """SolaceTopic enum must have a NOTIFICATIONS member."""
        from solace_events.config import SolaceTopic

        assert hasattr(SolaceTopic, "NOTIFICATIONS"), (
            "SolaceTopic is missing NOTIFICATIONS member"
        )
        assert SolaceTopic.NOTIFICATIONS.value == "solace.notifications"

    def test_solace_topic_has_audit(self) -> None:
        """SolaceTopic enum must have an AUDIT member."""
        from solace_events.config import SolaceTopic

        assert hasattr(SolaceTopic, "AUDIT"), (
            "SolaceTopic is missing AUDIT member"
        )
        assert SolaceTopic.AUDIT.value == "solace.audit"

    def test_solace_topic_has_users(self) -> None:
        """SolaceTopic enum must have a USERS member."""
        from solace_events.config import SolaceTopic

        assert hasattr(SolaceTopic, "USERS"), (
            "SolaceTopic is missing USERS member"
        )
        assert SolaceTopic.USERS.value == "solace.users"

    def test_solace_topic_all_members_have_configs(self) -> None:
        """Every SolaceTopic member must have a corresponding TOPIC_CONFIGS entry."""
        from solace_events.config import SolaceTopic, TOPIC_CONFIGS

        missing = [
            topic.name
            for topic in SolaceTopic
            if topic not in TOPIC_CONFIGS
        ]
        assert not missing, (
            f"SolaceTopic members missing from TOPIC_CONFIGS: {missing}"
        )

    # -- TherapyModality ----------------------------------------------------

    def test_therapy_modality_has_sfbt(self) -> None:
        """TherapyModality enum must include SFBT."""
        from solace_events.schemas import TherapyModality

        assert hasattr(TherapyModality, "SFBT"), (
            "TherapyModality is missing SFBT member"
        )
        assert TherapyModality.SFBT.value == "SFBT"

    def test_therapy_modality_core_members(self) -> None:
        """TherapyModality must contain the foundational therapy approaches."""
        from solace_events.schemas import TherapyModality

        expected = {"CBT", "DBT", "ACT", "MI", "MINDFULNESS", "PSYCHOEDUCATION", "SFBT"}
        actual = {member.name for member in TherapyModality}
        missing = expected - actual
        assert not missing, (
            f"TherapyModality is missing expected members: {missing}"
        )

    # -- _TOPIC_MAP ----------------------------------------------------------

    def test_topic_map_has_user_prefix(self) -> None:
        """The internal _TOPIC_MAP must route 'user.' events to solace.users."""
        from solace_events.schemas import _TOPIC_MAP

        assert "user." in _TOPIC_MAP, (
            "_TOPIC_MAP is missing 'user.' prefix entry. "
            f"Current keys: {list(_TOPIC_MAP.keys())}"
        )
        assert _TOPIC_MAP["user."] == "solace.users"

    def test_topic_map_has_required_prefixes(self) -> None:
        """_TOPIC_MAP must contain routing entries for all major event domains."""
        from solace_events.schemas import _TOPIC_MAP

        required_prefixes = [
            "session.",
            "safety.",
            "diagnosis.",
            "therapy.",
            "memory.",
            "personality.",
            "notification.",
            "system.",
            "user.",
        ]
        missing = [p for p in required_prefixes if p not in _TOPIC_MAP]
        assert not missing, (
            f"_TOPIC_MAP missing required prefix entries: {missing}"
        )

    def test_topic_map_values_are_valid_topics(self) -> None:
        """Every _TOPIC_MAP value must correspond to a valid SolaceTopic value."""
        from solace_events.config import SolaceTopic
        from solace_events.schemas import _TOPIC_MAP

        valid_values = {topic.value for topic in SolaceTopic}
        invalid = {
            prefix: topic
            for prefix, topic in _TOPIC_MAP.items()
            if topic not in valid_values
        }
        assert not invalid, (
            f"_TOPIC_MAP contains invalid topic values: {invalid}. "
            f"Valid topics: {sorted(valid_values)}"
        )
