"""MVP-180: logger.exception() must emit the traceback, and the traceback text must pass the PHI sanitizer."""
import dataclasses, json, logging, subprocess, sys
import pytest, structlog
from solace_infrastructure.runtime import ServiceRuntimeSpec, StartupDependencyError
from solace_infrastructure.runtime.logging import configure_service_logging, logging_chain_names

def _spec(env="development", **kw):
    return ServiceRuntimeSpec(service_name="safety-service", environment=env, **kw)

def _last_json(out: str) -> dict:
    return json.loads(out.strip().splitlines()[-1])

def test_exception_log_carries_the_traceback(capsys):
    configure_service_logging(_spec())
    try:
        raise RuntimeError("boom-xyz")
    except RuntimeError:
        structlog.get_logger("t").exception("crisis_handler_failed")
    line = _last_json(capsys.readouterr().out)
    assert line["event"] == "crisis_handler_failed"
    assert "Traceback (most recent call last)" in line["exception"]
    assert "RuntimeError" in line["exception"]
    assert "exc_info" not in line

def test_traceback_text_is_phi_sanitized(capsys):
    configure_service_logging(_spec("production"))
    try:
        raise ValueError("lookup failed for jane.doe@example.com 555-867-5309")
    except ValueError:
        structlog.get_logger("t").exception("lookup_failed")
    out = capsys.readouterr().out
    assert "jane.doe@example.com" not in out
    assert "555-867-5309" not in out

def test_chain_order_puts_exc_info_before_sanitizer_before_renderer():
    configure_service_logging(_spec("production"))
    names = logging_chain_names()
    assert (
        names.index("format_exc_info")
        < names.index("redact_credentials")
        < names.index("phi_sanitizer_processor")
        < names.index("JSONRenderer")
    )
    assert names[names.index("phi_sanitizer_processor") - 1] == "redact_credentials"
    assert names[:4] == ["merge_contextvars", "add_log_level", "TimeStamper", "StackInfoRenderer"]

def test_strict_requires_the_phi_sanitizer(monkeypatch):
    monkeypatch.setitem(sys.modules, "solace_security.phi_protection", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("staging"))
    assert exc.value.component == "logging"

def test_dev_without_sanitizer_still_configures(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "solace_security.phi_protection", None)
    configure_service_logging(_spec("development"))
    structlog.get_logger("t").info("hello")
    assert _last_json(capsys.readouterr().out)["event"] == "hello"
    assert "redact_credentials" in logging_chain_names()

def test_console_renderer_never_in_strict(capsys):
    configure_service_logging(_spec("production", debug=True))
    structlog.get_logger("t").info("hello")
    assert _last_json(capsys.readouterr().out)["event"] == "hello"

def test_stdlib_and_uvicorn_records_are_json(capsys):
    configure_service_logging(_spec())
    logging.getLogger("uvicorn.error").error("bind failed")
    assert _last_json(capsys.readouterr().out)["event"] == "bind failed"
    assert logging.getLogger("uvicorn.access").disabled is True

def test_spec_is_frozen_and_validated():
    spec = _spec()
    assert (spec.event_plane, spec.docs_enabled, spec.uses_postgres) == ("none", None, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.debug = True
    with pytest.raises(ValueError):
        ServiceRuntimeSpec(service_name="x", environment="development", event_plane="consume")

def test_runtime_package_import_is_light():
    code = "import sys, solace_infrastructure.runtime as r; print('fastapi' in sys.modules, 'asyncpg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False False"

# --- WS-04.2 review fix round 1 ---

def test_a_record_that_fails_to_format_never_echoes_its_args(capsys):
    configure_service_logging(_spec("production"))
    logging.getLogger("svc").error("patient %s has %s", "jane.doe@example.com")
    captured = capsys.readouterr()
    assert "jane.doe@example.com" not in captured.out
    assert "jane.doe@example.com" not in captured.err
    dropped = _last_json(captured.out)
    assert (dropped["event"], dropped["logger"], dropped["level"], dropped["error_type"]) == (
        "log_record_dropped", "svc", "error", "TypeError")

@pytest.mark.parametrize("switch", ["PHI_ENABLE_MASKING", "PHI_ENABLE_DETECTION"])
def test_strict_rejects_an_inactive_sanitizer(monkeypatch, switch):
    from solace_security import phi_protection
    monkeypatch.setenv(switch, "false")
    monkeypatch.setattr(phi_protection, "_structlog_sanitizer", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("staging"))
    assert exc.value.component == "logging"

def test_dsn_password_is_redacted_from_exception_text(capsys):
    configure_service_logging(_spec())
    try:
        raise ConnectionError("connect failed postgresql://solace:S3cretPw9@db:5432/solace")
    except ConnectionError:
        structlog.get_logger("t").exception("pool_failed")
    out = capsys.readouterr().out
    assert "S3cretPw9" not in out
    assert "postgresql://[REDACTED]@db" in _last_json(out)["exception"]

def test_bearer_token_in_a_stdlib_argument_is_redacted(capsys):
    configure_service_logging(_spec())
    logging.getLogger("svc.http").warning("upstream call sent %s", "Authorization: Bearer abcDEF123456789xyz")
    out = capsys.readouterr().out
    assert "abcDEF123456789xyz" not in out
    assert "Bearer [REDACTED]" in _last_json(out)["event"]

def test_query_token_in_a_message_is_redacted(capsys):
    configure_service_logging(_spec())
    structlog.get_logger("t").info("handshake GET /ws?token=tok9abc8def&room=1",
                                   links=[{"callback": "/cb?code=c0de5ecret"}])
    out = capsys.readouterr().out
    assert "tok9abc8def" not in out and "c0de5ecret" not in out
    line = _last_json(out)
    assert line["event"] == "handshake GET /ws?token=[REDACTED]&room=1"
    assert line["links"] == [{"callback": "/cb?code=[REDACTED]"}]

def test_uvicorn_handshake_lines_never_log_the_token(capsys):
    configure_service_logging(_spec(log_level="DEBUG"))
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.info('%s - "WebSocket %s" [accepted]', "1.2.3.4:5", "/ws?token=abc")
    captured = capsys.readouterr()
    assert "token=abc" not in captured.out + captured.err
    assert uvicorn_error.getEffectiveLevel() >= logging.WARNING

def test_bare_strict_environment_requires_the_sanitizer(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setitem(sys.modules, "solace_security.phi_protection", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("development"))
    assert exc.value.component == "logging"

def test_httpx_request_lines_are_held_at_warning(capsys):
    configure_service_logging(_spec(log_level="DEBUG"))
    capsys.readouterr()
    logging.getLogger("httpx").info("HTTP Request: GET https://x/y?token=abc")
    logging.getLogger("httpcore").info("send_request_headers.started request=<Request [b'GET']>")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert all(logging.getLogger(n).getEffectiveLevel() >= logging.WARNING for n in ("httpx", "httpcore"))

# --- WS-04.2 review fix round 2 ---

@pytest.mark.parametrize("visible", ["-2", "5"])
def test_strict_rejects_a_sanitizer_that_shows_the_phone_digits(monkeypatch, visible):
    from solace_security import phi_protection
    monkeypatch.setenv("PHI_PARTIAL_MASK_VISIBLE", visible)
    monkeypatch.setattr(phi_protection, "_structlog_sanitizer", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("staging"))
    assert exc.value.component == "logging"

def test_strict_starts_with_zero_visible_digits_and_leaks_none(monkeypatch, capsys):
    import re
    from solace_security import phi_protection
    monkeypatch.setenv("PHI_PARTIAL_MASK_VISIBLE", "0")
    monkeypatch.setattr(phi_protection, "_structlog_sanitizer", None)
    configure_service_logging(_spec("staging"))
    structlog.get_logger("t").info("callback phone 555-867-5309")
    event = _last_json(capsys.readouterr().out)["event"]
    assert event.startswith("callback phone ")
    assert re.search(r"\d", event) is None

def test_strict_digit_window_check_catches_a_partial_phone_mask(monkeypatch):
    # Defence in depth: settings that skip validation and show six digits ("****675309")
    # carry neither the dashed literal nor the e-mail, so only the digit-window check sees it.
    from solace_security import phi_protection as pp
    settings = pp.PHISettings.model_construct(**{**pp.PHISettings().model_dump(), "partial_mask_visible": 6})
    monkeypatch.setattr(pp, "_structlog_sanitizer", pp.PHISanitizer(pp.PHIDetector(settings)))
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("staging"))
    assert (exc.value.component, exc.value.reason) == ("logging", "PHI log sanitizer inactive")

def test_credential_redaction_is_fast_on_a_long_string():
    import time
    from solace_infrastructure.runtime.logging import redact_credentials
    payload = "ab." * 21846  # 64 KB with no URL in it
    started = time.perf_counter()
    redact_credentials(None, "info", {"event": payload})
    assert time.perf_counter() - started < 0.5

# --- WS-04.2 review fix round 3 ---

def test_uvicorn_unhandled_asgi_exception_logs_only_the_error_type(capsys):
    from pydantic import BaseModel, ValidationError

    class Note(BaseModel):
        age: int

    try:
        Note(age="Margaret Hollis reports chest pain")
    except ValidationError as exc:
        error = exc
    configure_service_logging(_spec("production"))
    configure_service_logging(_spec("production"))  # the rewrite survives a repeated call
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.error("Exception in ASGI application\n", exc_info=error)
    captured = capsys.readouterr()
    assert "Margaret Hollis" not in captured.out + captured.err
    lines = [json.loads(l) for l in captured.out.splitlines() if l.startswith("{")]
    asgi = [l for l in lines if l.get("event") == "asgi_unhandled_exception"]
    assert len(asgi) == 1
    assert (asgi[0]["error_type"], asgi[0]["level"]) == ("ValidationError", "error")
    assert "exception" not in asgi[0]
    assert len(uvicorn_error.filters) == len({type(f) for f in uvicorn_error.filters})

def test_uvicorn_lifespan_settings_error_logs_fields_not_values(monkeypatch, capsys):
    from typing import Annotated
    from pydantic import Field, ValidationError
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class KeySettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="WS042_")
        encryption_master_key: Annotated[str, Field(min_length=32)]

    monkeypatch.setenv("WS042_ENCRYPTION_MASTER_KEY", "S3cretValue123")
    try:
        KeySettings()
    except ValidationError as exc:
        error = exc
    configure_service_logging(_spec("production"))
    configure_service_logging(_spec("production"))
    logging.getLogger("uvicorn.error").error("Exception in 'lifespan' protocol\n", exc_info=error)
    captured = capsys.readouterr()
    assert "S3cretValue123" not in captured.out + captured.err
    lines = [json.loads(l) for l in captured.out.splitlines() if l.startswith("{")]
    failed = [l for l in lines if l.get("event") == "lifespan_startup_failed"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "ValidationError" and failed[0]["level"] == "error"
    assert failed[0]["validation_errors"] == [{"type": "string_too_short", "field": "encryption_master_key"}]
    assert "exception" not in failed[0]

def test_uvicorn_lifespan_other_error_keeps_its_traceback(capsys):
    try:
        raise RuntimeError("startup step failed")
    except RuntimeError as exc:
        error = exc
    configure_service_logging(_spec("production"))
    logging.getLogger("uvicorn.error").error("Exception in 'lifespan' protocol\n", exc_info=error)
    line = _last_json(capsys.readouterr().out)
    assert line["event"].startswith("Exception in 'lifespan' protocol")
    assert "Traceback (most recent call last)" in line["exception"]
    assert "RuntimeError: startup step failed" in line["exception"]

def _key_settings_class(monkeypatch):
    from typing import Annotated
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class KeySettings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="WS042_")
        encryption_master_key: Annotated[str, Field(min_length=32)]

    monkeypatch.setenv("WS042_ENCRYPTION_MASTER_KEY", "S3cretValue123")
    return KeySettings

def _lifespan_lines(captured):
    assert "S3cretValue123" not in captured.out + captured.err
    lines = [json.loads(l) for l in captured.out.splitlines() if l.startswith("{")]
    return [l for l in lines if l.get("event") == "lifespan_startup_failed"]

def test_starlette_lifespan_failure_text_logs_fields_not_values(monkeypatch, capsys):
    # Starlette sends traceback.format_exc() as lifespan.startup.failed, and uvicorn logs that
    # text on uvicorn.error from inside Starlette's except block (no exc_info on the record).
    import traceback
    from pydantic import ValidationError
    key_settings = _key_settings_class(monkeypatch)
    configure_service_logging(_spec("production"))
    try:
        key_settings()
    except ValidationError:
        logging.getLogger("uvicorn.error").error(traceback.format_exc())
    failed = _lifespan_lines(capsys.readouterr())
    assert len(failed) == 1
    assert failed[0]["error_type"] == "ValidationError"
    assert failed[0]["validation_errors"] == [{"type": "string_too_short", "field": "encryption_master_key"}]

def test_lifespan_failure_text_outside_the_except_block_still_drops_the_input(monkeypatch, capsys):
    import traceback
    from pydantic import ValidationError
    key_settings = _key_settings_class(monkeypatch)
    try:
        key_settings()
    except ValidationError:
        text = traceback.format_exc()
    configure_service_logging(_spec("production"))
    logging.getLogger("uvicorn.error").error(text)
    failed = _lifespan_lines(capsys.readouterr())
    assert len(failed) == 1 and failed[0]["error_type"] == "ValidationError"
    assert failed[0]["validation_errors"] == []

def test_lifespan_chained_settings_error_logs_fields_not_values(monkeypatch, capsys):
    from pydantic import ValidationError
    key_settings = _key_settings_class(monkeypatch)
    try:
        try:
            key_settings()
        except ValidationError as inner:
            raise RuntimeError("settings could not be loaded") from inner
    except RuntimeError as exc:
        error = exc
    configure_service_logging(_spec("production"))
    logging.getLogger("uvicorn.error").error("Exception in 'lifespan' protocol\n", exc_info=error)
    failed = _lifespan_lines(capsys.readouterr())
    assert len(failed) == 1 and failed[0]["error_type"] == "RuntimeError"
    assert failed[0]["validation_errors"] == [{"type": "string_too_short", "field": "encryption_master_key"}]

def test_lifespan_failure_text_without_pydantic_input_is_kept(capsys):
    import traceback
    configure_service_logging(_spec("production"))
    try:
        raise RuntimeError("startup step failed")
    except RuntimeError:
        logging.getLogger("uvicorn.error").error(traceback.format_exc())
    line = _last_json(capsys.readouterr().out)
    assert line["event"].startswith("Traceback (most recent call last)")
    assert "RuntimeError: startup step failed" in line["event"]

def test_dev_rejects_invalid_phi_settings_at_configure_time(monkeypatch):
    from solace_security import phi_protection
    monkeypatch.setenv("PHI_PARTIAL_MASK_VISIBLE", "5")
    monkeypatch.setattr(phi_protection, "_structlog_sanitizer", None)
    with pytest.raises(StartupDependencyError) as exc:
        configure_service_logging(_spec("development"))
    assert exc.value.component == "logging"

def test_startup_error_never_carries_a_dsn_password():
    assert "S3cret" not in str(StartupDependencyError("postgres", "failed postgresql://u:S3cret@h/db"))
