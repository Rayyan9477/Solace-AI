"""MVP-031: guards must require AUTH_SECRET_KEY, ENCRYPTION_MASTER_KEY, ENCRYPTION_SEARCH_HASH_SALT
and POSTGRES_PASSWORD (what the services read), not SECRET_KEY/ENCRYPTION_KEY; staging is strict;
violations never contain a value."""
import secrets
import pytest
from solace_security.production_guards import ProductionGuards, ProductionSecurityError

READ_NAMES = ("AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT", "POSTGRES_PASSWORD")

@pytest.fixture
def strict_ok(monkeypatch):
    for gone in ("SECRET_KEY", "ENCRYPTION_KEY", "JWT_SECRET", "API_KEY", "DEBUG", "TESTING", "MOCK_SERVICES", "USE_IN_MEMORY_DB"):
        monkeypatch.delenv(gone, raising=False)
    values = {"AUTH_SECRET_KEY": secrets.token_urlsafe(48), "ENCRYPTION_MASTER_KEY": secrets.token_hex(32),
              "ENCRYPTION_SEARCH_HASH_SALT": secrets.token_urlsafe(32), "POSTGRES_PASSWORD": secrets.token_urlsafe(24),
              "POSTGRES_USER": "solace_app", "POSTGRES_SSL_MODE": "verify-full"}
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    return values

@pytest.mark.parametrize("env", ["staging", "production", "Staging", " PRODUCTION "])
def test_is_strict_environment_true(env):
    from solace_security.production_guards import is_strict_environment
    assert is_strict_environment(env) is True

@pytest.mark.parametrize("env", ["development", "test", ""])
def test_is_strict_environment_false(env):
    from solace_security.production_guards import is_strict_environment
    assert is_strict_environment(env) is False

def test_is_strict_environment_reads_ENVIRONMENT_when_none(monkeypatch):
    from solace_security.production_guards import is_strict_environment
    monkeypatch.setenv("ENVIRONMENT", "staging")
    assert is_strict_environment() is True
    monkeypatch.delenv("ENVIRONMENT")
    assert is_strict_environment() is False

@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_passes_with_the_variables_services_read(monkeypatch, strict_ok, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)
    ProductionGuards.validate_environment()

@pytest.mark.parametrize("missing", READ_NAMES)
def test_each_read_variable_is_required_in_strict(monkeypatch, strict_ok, missing):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.delenv(missing)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith(f"{missing} ") for v in exc.value.violations)

def test_required_vars_follow_the_caller(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ENCRYPTION_MASTER_KEY")
    monkeypatch.delenv("ENCRYPTION_SEARCH_HASH_SALT")
    ProductionGuards.assert_startup("production", service_name="analytics-service",
                                    required_vars=("AUTH_SECRET_KEY", "POSTGRES_PASSWORD"))

# WS-12.1 re-review WS121F1-CLO-1: a blank service environment under a strict bare ENVIRONMENT used to
# skip the consistency check, so the service's own strict-only guards (e.g. crisis delivery) never ran.
@pytest.mark.parametrize("blank", ["", "   ", None])
@pytest.mark.parametrize("bare", ["production", "staging", " Production "])
def test_blank_service_environment_under_strict_bare_environment_is_refused(monkeypatch, strict_ok, bare, blank):
    monkeypatch.setenv("ENVIRONMENT", bare)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.assert_startup(blank, service_name="safety-service")
    assert any("environment is blank" in v for v in exc.value.violations)

@pytest.mark.parametrize("bare", ["development", "test", ""])
def test_blank_service_environment_outside_strict_is_allowed(monkeypatch, bare):
    monkeypatch.setenv("ENVIRONMENT", bare)
    ProductionGuards.assert_startup("", service_name="safety-service")

def test_violation_never_contains_the_value(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "changeme")
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert "changeme" not in str(exc.value)
    assert all("changeme" not in v for v in exc.value.violations)

def test_strict_requires_verify_full(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("POSTGRES_SSL_MODE", "require")
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith("POSTGRES_SSL_MODE ") for v in exc.value.violations)

def test_dev_is_never_blocked(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    for name in READ_NAMES:
        monkeypatch.delenv(name, raising=False)
    ProductionGuards.validate_environment()

# The repo's own dev defaults (docker-compose.yml, encryption.py) and .env.example placeholders.
REPO_DEV_AND_PLACEHOLDER_SECRETS = [
    ("AUTH_SECRET_KEY", "dev-jwt-secret-key-minimum-32-chars"),
    ("AUTH_SECRET_KEY", "your-secure-auth-secret-key-32-chars"),
    ("AUTH_SECRET_KEY", "dev-only-insecure-key-32-bytes!!"),
    ("ENCRYPTION_MASTER_KEY", "dev-only-insecure-key-32-bytes!!"),
    ("ENCRYPTION_MASTER_KEY", "your-secure-encryption-key-32-chars"),
    ("POSTGRES_PASSWORD", "solace_dev_password"),
    ("POSTGRES_PASSWORD", "your-secure-database-password"),
    # WS-03.2 compose dev defaults for the three platform roles
    ("POSTGRES_PASSWORD", "solace_app_dev_only"),
    ("POSTGRES_PASSWORD", "solace_migrator_dev_only"),
    ("POSTGRES_PASSWORD", "solace_superuser_dev_only"),
]

@pytest.mark.parametrize(("name", "value"), REPO_DEV_AND_PLACEHOLDER_SECRETS)
def test_repo_dev_and_placeholder_secrets_are_rejected(monkeypatch, strict_ok, name, value):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv(name, value)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith(f"{name} ") for v in exc.value.violations)
    assert value not in str(exc.value)

# WS-03.3 hand-off: REDIS_URL is required for every Redis-using service; in strict it must carry a
# password that is not the compose dev default (solace_redis_dev_only) or another known weak value.
REDIS_REQUIRED = ("AUTH_SECRET_KEY", "ENCRYPTION_MASTER_KEY", "ENCRYPTION_SEARCH_HASH_SALT",
                  "POSTGRES_PASSWORD", "REDIS_URL")

@pytest.mark.parametrize("url", [
    "redis://:solace_redis_dev_only@redis:6379/0",
    "redis://:SOLACE_REDIS_DEV_ONLY@cache.internal:6380/2",
    "rediss://default:solace_redis_dev_only@redis:6379/0",
    "redis://redis:6379/0",
    "redis://:@redis:6379/0",
    "redis://:%20@redis:6379/0",
    "redis://:changeme@redis:6379/0",
    # WS-03.3 review SEC-1: the password must be URL-safe (registry format urlsafe24) and long enough
    "redis://:AbCdEf/ghI+jk=@redis:6379/0",
    "redis://:" + "a b c d e f g h i j k l" + "@redis:6379/0",
    "redis://:%2Fabcdefghijklmnopq@redis:6379/0",
    "redis://:shortpw123@redis:6379/0",
    "redis://:abcdefghijklmnopqrstuvwx@:6379/0",
])
def test_strict_redis_url_needs_a_real_password(monkeypatch, strict_ok, url):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("REDIS_URL", url)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment(required_vars=REDIS_REQUIRED)
    assert any(v.startswith("REDIS_URL ") for v in exc.value.violations)
    assert "solace_redis_dev_only" not in str(exc.value).lower()
    assert "changeme" not in str(exc.value)

def test_strict_redis_url_with_a_strong_password_passes(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", f"redis://:{secrets.token_urlsafe(24)}@redis:6379/0")
    ProductionGuards.validate_environment(required_vars=REDIS_REQUIRED)

def test_redis_url_is_not_checked_when_not_required(monkeypatch, strict_ok):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    ProductionGuards.validate_environment()

@pytest.mark.parametrize("bad_key", ["k" * 32, "k" * 40, secrets.token_urlsafe(32),
                                     secrets.token_hex(32) + "\n", " " + secrets.token_hex(32)])
def test_strict_master_key_must_parse_as_32_bytes(monkeypatch, strict_ok, bad_key):
    """CR-2 (WS-09.1): the guard uses parse_master_key(strict=True), not a length floor."""
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", bad_key)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.validate_environment()
    assert any(v.startswith("ENCRYPTION_MASTER_KEY ") for v in exc.value.violations)
    assert bad_key not in str(exc.value)

def test_strict_master_key_accepts_hex_and_base64_prefix(monkeypatch, strict_ok):
    import base64
    monkeypatch.setenv("ENVIRONMENT", "staging")
    for good in (secrets.token_hex(32), "base64:" + base64.b64encode(secrets.token_bytes(32)).decode()):
        monkeypatch.setenv("ENCRYPTION_MASTER_KEY", good)
        ProductionGuards.validate_environment()

@pytest.mark.parametrize("bare", [None, "", "   "])
def test_strict_service_requires_the_bare_environment(monkeypatch, strict_ok, bare):
    if bare is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", bare)
    with pytest.raises(ProductionSecurityError) as exc:
        ProductionGuards.assert_startup("production", service_name="memory-service")
    assert any(v.startswith("ENVIRONMENT ") for v in exc.value.violations)

def test_dev_service_without_bare_environment_is_not_blocked(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    ProductionGuards.assert_startup("development", service_name="memory-service")
