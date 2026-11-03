"""REV-12: lock PHI-encryption activation across services (comment-agnostic).

The 2026-07-21 review flagged that ``configure_phi_encryption()`` is called in
only 6 of 10 services and implied the other 4 leak PHI at rest. Investigation
showed that framing was a file-count, not a data-flow: ``configure_phi_encryption``
installs a *process-global* encryptor that the SQLAlchemy ``ClinicalBase`` event
listeners consult, so it only matters in a service process that persists
ClinicalBase clinical rows.

The six clinical services do call it; the four flagged services persist **no**
ClinicalBase PHI and are exempt with a specific, documented reason:
  * user-service   — own non-ClinicalBase models + dedicated Fernet EncryptionService.
  * notification   — Postgres used only for the operational event outbox.
  * analytics      — events live in ClickHouse, not SQLAlchemy.
  * config         — system configuration / feature flags / secrets, no PHI.

Adding the call to those four would be a no-op that also forces the PHI master key
into services that never touch PHI (a least-privilege regression). So instead of a
cosmetic call, this test locks the real state:

  1. every service is *explicitly classified* as PHI-encrypting (legacy), PHI via
     the runtime, or exempt — a new service forces a conscious decision here;
  2. two activation routes exist: the legacy route (``_PHI_ENCRYPTING``: the service
     *calls* configure_phi_encryption) and the IF-01 route (``_PHI_VIA_RUNTIME``: the
     service declares ``ServiceRuntimeSpec(..., handles_phi=True)``, or still has the
     legacy call until its adoption lands). Both are detected via AST, so comments
     and non-literal values do not count;
  3. the detectors have teeth (proven against synthetic positive/negative cases).
"""
from __future__ import annotations

import ast
import pathlib

# Repo root = two levels up from tests/integration/<file>.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SERVICES_DIR = _REPO_ROOT / "services"

# Services that persist ClinicalBase PHI and MUST activate encryption at startup.
_PHI_ENCRYPTING = {"safety_service"}

# Services that activate PHI handling through the IF-01 runtime SPEC
# (``handles_phi=True``) or, until their adoption lands, the legacy call.
_PHI_VIA_RUNTIME = {
    "diagnosis_service",
    "memory_service",
    "orchestrator_service",
    "personality_service",
    "therapy_service",
}

# Services deliberately exempt because they persist no ClinicalBase PHI. Explicit
# so the exemption stays a conscious, reviewed decision (see module docstring).
_EXEMPT = {
    "user-service",
    "notification-service",
    "analytics-service",
    "config_service",
}

_ACTIVATION_FUNC = "configure_phi_encryption"


def _imports_clinical_base(src: pathlib.Path) -> bool:
    """True iff some .py under ``src`` imports ClinicalBase or a clinical entity module.

    AST-based (ignores comments/docstrings), so the exemption claim is checked
    against real imports, not a stray mention.
    """
    for py in src.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "solace_infrastructure.database.entities" in mod:
                    return True
                if any(alias.name == "ClinicalBase" for alias in node.names):
                    return True
            elif isinstance(node, ast.Import):
                if any("database.entities" in alias.name for alias in node.names):
                    return True
    return False


def _calls_activation(src: pathlib.Path) -> bool:
    """True iff some .py under ``src`` contains a real call to configure_phi_encryption.

    AST-based, so comments and docstrings that merely mention the name do not count.
    """
    for py in src.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name == _ACTIVATION_FUNC:
                return True
    return False


def _declares_phi_spec(src: pathlib.Path) -> bool:
    """True iff some .py under ``src`` calls ServiceRuntimeSpec(..., handles_phi=True).

    AST-based; only the literal ``True`` keyword form counts (a non-literal does not).
    """
    for py in src.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name != "ServiceRuntimeSpec":
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "handles_phi"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    return True
    return False


def _service_names() -> list[str]:
    return sorted(
        d.name
        for d in _SERVICES_DIR.iterdir()
        if d.is_dir() and (d / "src").is_dir()
    )


class TestPhiEncryptionActivation:
    def test_every_service_is_explicitly_classified(self) -> None:
        """A new service must be consciously placed in exactly one bucket."""
        buckets = (_PHI_ENCRYPTING, _PHI_VIA_RUNTIME, _EXEMPT)
        assert all(
            a.isdisjoint(b) for i, a in enumerate(buckets) for b in buckets[i + 1:]
        ), "a service is in more than one bucket"
        unclassified = [s for s in _service_names() if s not in set().union(*buckets)]
        assert not unclassified, (
            f"Unclassified services {unclassified}: add each to _PHI_ENCRYPTING "
            f"(legacy configure_phi_encryption call), _PHI_VIA_RUNTIME (handles_phi=True "
            f"SPEC) or to _EXEMPT with a documented reason."
        )

    def test_runtime_phi_services_activate_through_spec_or_legacy_call(self) -> None:
        """Each runtime-route PHI service has a handles_phi=True SPEC or the legacy call."""
        missing = [
            n for n in sorted(_PHI_VIA_RUNTIME)
            if not (
                _declares_phi_spec(_SERVICES_DIR / n / "src")
                or _calls_activation(_SERVICES_DIR / n / "src")
            )
        ]
        assert not missing, (
            f"PHI services with neither a handles_phi=True SPEC nor "
            f"configure_phi_encryption(): {missing}"
        )

    def test_phi_services_activate_encryption(self) -> None:
        """Each clinical service actually calls configure_phi_encryption()."""
        missing = [
            name
            for name in _PHI_ENCRYPTING
            if (_SERVICES_DIR / name / "src").is_dir()
            and not _calls_activation(_SERVICES_DIR / name / "src")
        ]
        assert not missing, (
            f"PHI-encrypting services missing a configure_phi_encryption() call: {missing}"
        )

    def test_classification_is_non_vacuous(self) -> None:
        """Guard against the scanner matching nothing (e.g. renamed service dirs)."""
        present = set(_service_names())
        expected = _PHI_ENCRYPTING | _PHI_VIA_RUNTIME
        assert expected <= present, (
            f"expected all PHI services present; missing {expected - present}"
        )

    def test_exempt_services_do_not_import_clinical_base(self) -> None:
        """The exemption is only valid while a service imports no ClinicalBase entity.

        Backstops the runtime NOT NULL encryption_key_id guard with a *test-time*
        catch: if an exempt service starts importing a clinical entity, it must be
        reclassified into _PHI_ENCRYPTING (and wire configure_phi_encryption).
        """
        offenders = [
            name
            for name in _EXEMPT
            if (_SERVICES_DIR / name / "src").is_dir()
            and _imports_clinical_base(_SERVICES_DIR / name / "src")
        ]
        assert not offenders, (
            f"'Exempt' services import ClinicalBase and must be reclassified as "
            f"PHI-encrypting: {offenders}"
        )


def test_spec_detector_has_teeth(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    main = src / "main.py"
    main.write_text(
        "from solace_infrastructure.runtime import ServiceRuntimeSpec\n"
        "# SPEC = ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=True)\n"
        "SPEC = ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=False)\n",
        encoding="utf-8",
    )
    assert _declares_phi_spec(src) is False
    main.write_text(
        "from solace_infrastructure import runtime\n"
        "SPEC = runtime.ServiceRuntimeSpec(service_name='x', environment='test', handles_phi=True)\n",
        encoding="utf-8",
    )
    assert _declares_phi_spec(src) is True


def test_adoption_that_drops_phi_handling_is_caught(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "from solace_infrastructure.runtime import ServiceRuntimeSpec\n"
        "SPEC = ServiceRuntimeSpec(service_name='x', environment='test')\n",
        encoding="utf-8",
    )
    assert not (_declares_phi_spec(src) or _calls_activation(src))


def test_detector_has_teeth(tmp_path: pathlib.Path) -> None:
    """Positive/negative proof the AST detector distinguishes a call from a comment."""
    src = tmp_path / "src"
    src.mkdir()

    # Negative: only a comment mentioning the function — must NOT count as activation.
    (src / "commented.py").write_text(
        "# we should call configure_phi_encryption() here someday\n"
        "x = 1\n",
        encoding="utf-8",
    )
    assert _calls_activation(src) is False

    # Positive: a real call — must count.
    (src / "wired.py").write_text(
        "from solace_infrastructure.database.base_models import configure_phi_encryption\n"
        "configure_phi_encryption(object())\n",
        encoding="utf-8",
    )
    assert _calls_activation(src) is True
