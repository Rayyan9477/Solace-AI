"""MVP-092: ENCRYPTION_MASTER_KEY accepts the documented 64-hex recipe and 'base64:' keys."""
from __future__ import annotations

import base64
import secrets

import pytest
from pydantic import SecretStr

from solace_security.encryption import EncryptionSettings, Encryptor


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")


def test_mvp_092_documented_hex_recipe_is_accepted():
    key = secrets.token_bytes(32).hex()  # docs/STAGING-BRINGUP.md recipe: 64 hex characters
    settings = EncryptionSettings(master_key=key, _env_file=None)
    assert settings.master_key_bytes == bytes.fromhex(key)
    assert len(settings.master_key_bytes) == 32


def test_mvp_092_base64_prefixed_key_is_accepted():
    raw = bytes(range(32))
    settings = EncryptionSettings(master_key="base64:" + base64.b64encode(raw).decode(), _env_file=None)
    assert settings.master_key_bytes == raw


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_hex_key_accepted_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    assert EncryptionSettings(master_key="ab" * 32, _env_file=None).master_key_bytes == bytes.fromhex("ab" * 32)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_legacy_32_char_key_rejected_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(ValueError, match="64 hex characters"):
        EncryptionSettings(master_key="a" * 32, _env_file=None)


def test_mvp_092_legacy_32_char_key_still_accepted_in_test():
    assert EncryptionSettings(master_key="a" * 32, _env_file=None).master_key_bytes == b"a" * 32


@pytest.mark.parametrize("bad", ["short-key", "zz" * 32, "base64:" + base64.b64encode(b"x" * 31).decode()])
def test_mvp_092_error_message_names_working_formats(bad):
    with pytest.raises(ValueError) as exc:
        EncryptionSettings(master_key=bad, _env_file=None)
    message = str(exc.value)
    assert "exactly 32 bytes" in message
    assert "64 hex characters" in message
    assert "secrets.token_hex(32)" in message
    assert bad not in message  # never echo key material


def test_mvp_092_encryptor_round_trips_with_hex_key():
    encryptor = Encryptor(EncryptionSettings(master_key="cd" * 32, _env_file=None))
    assert encryptor.decrypt_to_string(encryptor.encrypt("phi-value", "field"), "field") == "phi-value"


def test_mvp_092_parse_master_key_strict_flag():
    from solace_security.encryption import parse_master_key

    assert parse_master_key("ef" * 32, strict=True) == bytes.fromhex("ef" * 32)
    with pytest.raises(ValueError):
        parse_master_key("a" * 32, strict=True)


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_092_for_development_refused_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(RuntimeError):
        EncryptionSettings.for_development()


_CANARY_SHORT = "leak-canary-short"  # 17 bytes: invalid everywhere
_CANARY_RAW32 = "leak-canary-strict-32-chars-0000"  # 32 bytes: invalid only in strict envs


@pytest.mark.parametrize(
    ("env", "bad", "source"),
    [
        ("test", _CANARY_SHORT, "kwarg"),
        ("test", _CANARY_SHORT, "env"),
        ("staging", _CANARY_RAW32, "kwarg"),
        ("staging", _CANARY_RAW32, "env"),
    ],
)
def test_mvp_092_bad_key_never_reaches_any_error_surface(monkeypatch, env, bad, source):
    monkeypatch.setenv("ENVIRONMENT", env)
    if source == "env":
        monkeypatch.setenv("ENCRYPTION_MASTER_KEY", bad)
        kwargs = {}
    else:
        kwargs = {"master_key": bad}
    with pytest.raises(ValueError) as exc:
        EncryptionSettings(_env_file=None, **kwargs)
    errors = getattr(exc.value, "errors", lambda **k: [])()
    for surface in (repr(errors), str(exc.value), repr(exc.value)):
        assert bad not in surface
    assert exc.value.__context__ is None
    assert "64 hex characters" in str(exc.value)


def test_mvp_092_master_key_bytes_follow_model_copy():
    settings = EncryptionSettings(master_key="ab" * 32, _env_file=None)
    copied = settings.model_copy(update={"master_key": SecretStr("cd" * 32)})
    assert copied.master_key_bytes == bytes.fromhex("cd" * 32)
    assert settings.master_key_bytes == bytes.fromhex("ab" * 32)
