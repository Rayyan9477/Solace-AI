"""
E2E test for PHI at-rest encryption via the SQLAlchemy event listeners.

Sprint 1 goal: prove that the ClinicalBase lifecycle hooks
(``_encrypt_phi_before_insert``, ``_encrypt_phi_before_update``,
``_decrypt_phi_after_load``) actually encrypt and decrypt PHI fields —
including the JSONB list/dict payloads (NEW-03 / NEW-04) — when the global
FieldEncryptor is configured via ``configure_phi_encryption``.

We exercise the listener functions directly against a fake entity rather
than setting up a full database. The SQLAlchemy event dispatch itself is
library-tested; what matters here is that *our* listener body calls the
encryptor with the right inputs and leaves the instance in the expected
state. The listeners only touch ``__phi_fields__`` attributes and
``encryption_key_id``, so a plain Python object with those slots is a
faithful stand-in for a mapped entity.
"""
from __future__ import annotations

import uuid
from typing import Any, ClassVar

import pytest
from pydantic import SecretStr

from solace_infrastructure.database.base_models import (
    ClinicalBase,
    _decrypt_phi_after_load,
    _encrypt_phi_before_insert,
    _encrypt_phi_before_update,
    configure_phi_encryption,
    get_phi_encryptor,
)


class _FakeClinical:
    """Stand-in for a ClinicalBase-mapped entity.

    Holds only the attributes the event listeners inspect, plus the
    ``encrypt_phi_fields``/``decrypt_phi_fields`` methods rebound from
    ``ClinicalBase`` so the listener body runs the real encryption logic.
    """

    __phi_fields__: ClassVar[list[str]]

    # Bind the real methods so the listener's ``target.encrypt_phi_fields(...)``
    # call resolves to the production code under test.
    encrypt_phi_fields = ClinicalBase.encrypt_phi_fields
    decrypt_phi_fields = ClinicalBase.decrypt_phi_fields
    _PHI_JSON_LIST_MARKER = ClinicalBase._PHI_JSON_LIST_MARKER
    _PHI_JSON_DICT_MARKER = ClinicalBase._PHI_JSON_DICT_MARKER

    def __init__(
        self,
        phi_fields: list[str],
        is_phi: bool = True,
        encryption_key_id: str = "",
        **attrs: Any,
    ) -> None:
        # Per-instance declaration so each test is isolated
        self.__phi_fields__ = phi_fields
        self.is_phi = is_phi
        self.encryption_key_id = encryption_key_id
        self.id = uuid.uuid4()
        self.user_id = uuid.uuid4()
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.fixture
def field_encryptor() -> Any:
    """Build and install a dev-only FieldEncryptor globally for the test.

    The listeners are module-level, so they read the global encryptor. Make
    sure to tear it back down or subsequent tests will leak this state.
    """
    from solace_security.encryption import (
        EncryptionSettings,
        Encryptor,
        FieldEncryptor,
    )

    settings = EncryptionSettings.for_development()
    settings = settings.model_copy(
        update={"search_hash_salt": SecretStr("phi-at-rest-e2e-salt-32-bytes!!")}
    )
    fe = FieldEncryptor(Encryptor(settings), settings=settings)
    configure_phi_encryption(fe)
    yield fe
    configure_phi_encryption(None)  # type: ignore[arg-type]


class TestPhiAtRestListeners:
    """End-to-end: the three ClinicalBase event listeners must transform PHI."""

    def test_global_encryptor_configured(self, field_encryptor: Any) -> None:
        """Sanity check that the fixture actually installs a global encryptor."""
        assert get_phi_encryptor() is field_encryptor

    def test_before_insert_encrypts_str(self, field_encryptor: Any) -> None:
        """str PHI value becomes ciphertext after the insert listener fires."""
        entity = _FakeClinical(
            phi_fields=["summary"], summary="Patient reports severe depression."
        )
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert isinstance(entity.summary, str)
        assert entity.summary.startswith("v1$")
        assert "depression" not in entity.summary

    def test_before_insert_encrypts_list(self, field_encryptor: Any) -> None:
        """NEW-03: list PHI value (JSONB conversation) becomes ciphertext."""
        conversation = [
            {"role": "user", "content": "I can't sleep, chest tight for 3 weeks"},
            {"role": "assistant", "content": "Tell me more about the tightness"},
        ]
        entity = _FakeClinical(phi_fields=["messages"], messages=conversation)
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        # After encryption the list slot holds a ciphertext string
        assert isinstance(entity.messages, str)
        assert entity.messages.startswith("v1$")
        assert "can't sleep" not in entity.messages
        assert "chest tight" not in entity.messages

    def test_before_insert_encrypts_dict(self, field_encryptor: Any) -> None:
        """NEW-04: dict PHI value (JSONB evidence) becomes ciphertext."""
        evidence = {
            "dsm5_code": "F33.1",
            "criteria_met": ["anhedonia", "fatigue", "insomnia"],
            "notes": "Probable MDD, moderate severity",
        }
        entity = _FakeClinical(phi_fields=["evidence"], evidence=evidence)
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert isinstance(entity.evidence, str)
        assert entity.evidence.startswith("v1$")
        assert "anhedonia" not in entity.evidence
        assert "Probable MDD" not in entity.evidence

    def test_before_insert_populates_empty_encryption_key_id(
        self, field_encryptor: Any
    ) -> None:
        """H-57 wiring: listener fills the required key id if unset."""
        entity = _FakeClinical(phi_fields=["summary"], summary="plaintext")
        assert entity.encryption_key_id == ""
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert entity.encryption_key_id  # non-empty
        # Dev config's current_key_id is "primary"
        assert entity.encryption_key_id == "primary"

    def test_before_insert_skips_when_is_phi_false(
        self, field_encryptor: Any
    ) -> None:
        """Non-PHI records are not touched even if __phi_fields__ is declared."""
        entity = _FakeClinical(
            phi_fields=["summary"], is_phi=False, summary="not actually PHI"
        )
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert entity.summary == "not actually PHI"

    def test_before_update_re_encrypts(self, field_encryptor: Any) -> None:
        """An updated plaintext value is re-encrypted on the update hook."""
        entity = _FakeClinical(
            phi_fields=["summary"],
            encryption_key_id="primary",
            summary="original",
        )
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        ciphertext_v1 = entity.summary
        assert ciphertext_v1.startswith("v1$")

        # Simulate the app mutating the plaintext mid-session
        _decrypt_phi_after_load(target=entity, context=None)
        entity.summary = "updated"
        _encrypt_phi_before_update(mapper=None, connection=None, target=entity)
        assert entity.summary.startswith("v1$")
        assert entity.summary != ciphertext_v1  # different ciphertext

    def test_load_listener_decrypts_list_back_to_native_type(
        self, field_encryptor: Any
    ) -> None:
        """NEW-03 round-trip: stored list is restored as a list on load."""
        conversation = [{"role": "user", "content": "hello"}]
        entity = _FakeClinical(phi_fields=["messages"], messages=conversation)
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert isinstance(entity.messages, str)  # encrypted

        _decrypt_phi_after_load(target=entity, context=None)
        assert entity.messages == conversation  # back to original list

    def test_load_listener_decrypts_dict_back_to_native_type(
        self, field_encryptor: Any
    ) -> None:
        """NEW-04 round-trip: stored dict is restored as a dict on load."""
        evidence = {"dsm5_code": "F41.1", "criteria_met": 5}
        entity = _FakeClinical(phi_fields=["evidence"], evidence=evidence)
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert isinstance(entity.evidence, str)

        _decrypt_phi_after_load(target=entity, context=None)
        assert entity.evidence == evidence


class TestPhiAtRestWithoutGlobalEncryptor:
    """If no encryptor is configured, listeners must no-op (don't crash)."""

    def test_before_insert_noop_without_encryptor(self) -> None:
        """Without configure_phi_encryption, the listener leaves the entity alone."""
        configure_phi_encryption(None)  # type: ignore[arg-type]
        entity = _FakeClinical(phi_fields=["summary"], summary="plaintext")
        # Should not raise, should not encrypt
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert entity.summary == "plaintext"

    def test_diagnosis_session_phi_fields_wired_correctly(self) -> None:
        """Regression: verify the diagnosis entities declare the right PHI fields."""
        from solace_infrastructure.database.entities.diagnosis_entities import (
            DiagnosisSession,
            Hypothesis,
        )

        assert "summary" in DiagnosisSession.__phi_fields__
        assert "messages" in DiagnosisSession.__phi_fields__  # NEW-03

        # NEW-04
        for f in (
            "supporting_evidence",
            "contra_evidence",
            "challenge_results",
            "criteria_met",
            "criteria_missing",
        ):
            assert f in Hypothesis.__phi_fields__, f"missing PHI field: {f}"


class TestSafetyAndMemoryPhiFieldsWired:
    """REV-14: safety + memory clinical entities must declare all PHI fields.

    Before this fix, SafetyPlan only encrypted ``clinician_notes`` (leaving the
    user's warning signs, coping strategies and emergency contacts in plaintext),
    RiskFactor declared no PHI fields at all (``factor_description`` stored raw),
    and MemoryUserProfile — an entire clinical profile of diagnoses, treatments,
    triggers and safety info — had no ``__phi_fields__``, so every column was
    written to the database in the clear.

    The columns stay JSONB: the encrypt path serializes list/dict PHI to a
    ``v1$`` ciphertext string that lives inside the JSONB envelope (the same
    proven pattern as ``DiagnosisSession.messages``), so no migration is needed.
    """

    def test_safety_plan_encrypts_all_personal_content(self) -> None:
        """Every user-authored safety-plan field is declared PHI, not just notes."""
        from solace_infrastructure.database.entities.safety_entities import SafetyPlan

        for f in (
            "clinician_notes",
            "warning_signs",
            "coping_strategies",
            "emergency_contacts",
            "safe_environment_actions",
            "reasons_to_live",
            "professional_resources",
        ):
            assert f in SafetyPlan.__phi_fields__, f"SafetyPlan PHI field missing: {f}"

    def test_risk_factor_encrypts_description(self) -> None:
        """RiskFactor.factor_description holds clinical risk detail — must be PHI."""
        from solace_infrastructure.database.entities.safety_entities import RiskFactor

        assert "factor_description" in RiskFactor.__phi_fields__

    def test_memory_user_profile_encrypts_clinical_profile(self) -> None:
        """The whole longitudinal clinical profile must be encrypted at rest."""
        from solace_infrastructure.database.entities.memory_entities import (
            MemoryUserProfile,
        )

        for f in (
            "personal_facts",
            "therapeutic_context",
            "communication_preferences",
            "safety_information",
            "diagnosed_conditions",
            "current_treatments",
            "support_network",
            "triggers",
            "coping_strategies",
            "personality_traits",
        ):
            assert f in MemoryUserProfile.__phi_fields__, (
                f"MemoryUserProfile PHI field missing: {f}"
            )

    def test_encrypted_string_columns_are_text_not_bounded(self) -> None:
        """REV-13: encrypted PHI must live on TEXT columns, not String(N).

        A ``v1$`` ciphertext envelope is far longer than its plaintext (nonce +
        tag + base64), so a String(200/300/500) column silently truncates or
        rejects the encrypted value at write time. Every encrypted PHI field
        must be length-unbounded.
        """
        from sqlalchemy import Text

        from solace_infrastructure.database.entities.diagnosis_entities import (
            DiagnosisRecord,
        )
        from solace_infrastructure.database.entities.memory_entities import (
            TherapeuticEvent,
        )
        from solace_infrastructure.database.entities.therapy_entities import (
            TreatmentPlan,
        )

        bounded = [
            (DiagnosisRecord, "primary_diagnosis"),
            (TherapeuticEvent, "title"),
            (TreatmentPlan, "primary_diagnosis"),
            (TreatmentPlan, "termination_reason"),
        ]
        for entity, field in bounded:
            assert field in entity.__phi_fields__, f"{entity.__name__}.{field} not PHI?"
            col = entity.__table__.c[field]
            assert isinstance(col.type, Text), (
                f"{entity.__name__}.{field} is {col.type!r}; encrypted PHI needs Text "
                f"or the ciphertext overflows the column"
            )

    def test_safety_plan_list_fields_encrypt_to_ciphertext(
        self, field_encryptor: Any
    ) -> None:
        """A safety plan's warning signs must be ciphertext after the insert hook."""
        from solace_infrastructure.database.entities.safety_entities import SafetyPlan

        entity = _FakeClinical(
            phi_fields=list(SafetyPlan.__phi_fields__),
            warning_signs=["I stop answering my phone", "I skip meals"],
            coping_strategies=["call my sister", "box breathing"],
            emergency_contacts=[{"name": "Jane Doe", "phone": "555-0100"}],
        )
        _encrypt_phi_before_insert(mapper=None, connection=None, target=entity)
        assert entity.warning_signs.startswith("v1$")
        assert "skip meals" not in entity.warning_signs
        assert entity.emergency_contacts.startswith("v1$")
        assert "555-0100" not in entity.emergency_contacts

        _decrypt_phi_after_load(target=entity, context=None)
        assert entity.warning_signs == ["I stop answering my phone", "I skip meals"]
        assert entity.emergency_contacts == [{"name": "Jane Doe", "phone": "555-0100"}]
