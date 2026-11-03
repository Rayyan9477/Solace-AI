"""WS-05.4 hand-off (controller ruling): the test-only USER_EMAIL_ALLOW_SPECIAL_USE_TLDS setting.

The e2e harness registers users at ``@test.invalid``. email_validator 2.2 rejects every special-use domain name
(and so does pydantic ``EmailStr``, which calls it). When the setting is non-empty, user-service startup removes
exactly the listed TLDs from ``email_validator.SPECIAL_USE_DOMAIN_NAMES``; a name that is not on that list fails
startup with an error naming the setting.
"""
from __future__ import annotations

import email_validator
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

import src.main as main
from src import config as user_config
from src.api import RegisterRequest

SETTING = "USER_EMAIL_ALLOW_SPECIAL_USE_TLDS"
_PASSWORD = "Harness-Passw0rd"
_OTHER_SPECIAL_USE = ("x@host.localhost", "x@host.test", "x@host.local", "x@host.onion", "x@host.arpa")


def _register(email: str) -> RegisterRequest:
    return RegisterRequest(email=email, password=_PASSWORD, display_name="Harness")


@pytest.fixture(autouse=True)
def _library_list(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each test edits a copy of the library's list, so the process-wide list is restored afterwards."""
    names = list(email_validator.SPECIAL_USE_DOMAIN_NAMES)
    assert "invalid" in names and "localhost" in names
    monkeypatch.setattr(email_validator, "SPECIAL_USE_DOMAIN_NAMES", names)
    monkeypatch.delenv(SETTING, raising=False)
    return names


def test_default_setting_is_empty_and_rejects_test_invalid(_library_list: list[str]) -> None:
    assert user_config.EmailValidationConfig(_env_file=None).allow_special_use_tlds == ""
    assert user_config.allow_special_use_tlds("") == ()
    assert email_validator.SPECIAL_USE_DOMAIN_NAMES == _library_list
    with pytest.raises(ValidationError, match="special-use"):
        _register("user-a@test.invalid")


def test_setting_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETTING, "invalid")
    assert user_config.EmailValidationConfig(_env_file=None).allow_special_use_tlds == "invalid"


def test_invalid_allowance_accepts_test_invalid(_library_list: list[str]) -> None:
    before = list(_library_list)
    assert user_config.allow_special_use_tlds("invalid") == ("invalid",)
    assert email_validator.SPECIAL_USE_DOMAIN_NAMES == [n for n in before if n != "invalid"]
    assert _register("User-A@Test.Invalid").email == "User-A@test.invalid"


def test_allowance_is_idempotent(_library_list: list[str]) -> None:
    assert user_config.allow_special_use_tlds(" invalid , invalid ") == ("invalid",)
    assert user_config.allow_special_use_tlds("invalid") == ("invalid",)
    assert "invalid" not in email_validator.SPECIAL_USE_DOMAIN_NAMES
    assert _register("user-b@test.invalid").email == "user-b@test.invalid"


@pytest.mark.parametrize("address", _OTHER_SPECIAL_USE)
def test_localhost_and_other_special_use_names_stay_rejected(address: str) -> None:
    user_config.allow_special_use_tlds("invalid")
    with pytest.raises(ValidationError, match="special-use"):
        _register(address)


@pytest.mark.parametrize("value", ["example", "com", "invalid,example", ".invalid", "test.invalid"])
def test_unknown_value_is_refused_naming_the_setting(_library_list: list[str], value: str) -> None:
    before = list(_library_list)
    with pytest.raises(ValueError, match=SETTING):
        user_config.allow_special_use_tlds(value)
    assert email_validator.SPECIAL_USE_DOMAIN_NAMES == before  # nothing is removed when any name is refused


async def test_unknown_value_fails_startup(monkeypatch: pytest.MonkeyPatch, _library_list: list[str]) -> None:
    before = list(_library_list)
    monkeypatch.setenv(SETTING, "invalid,example")
    with pytest.raises(ValueError, match=SETTING):
        async with main.lifespan(FastAPI()):
            pytest.fail("the lifespan must not start with an unknown special-use TLD")
    assert email_validator.SPECIAL_USE_DOMAIN_NAMES == before
