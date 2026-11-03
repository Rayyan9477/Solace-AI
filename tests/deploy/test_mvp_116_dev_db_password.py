# tests/deploy/test_mvp_116_dev_db_password.py
"""MVP-116: the dev database password passes user-service's unsafe-secret check, so user-service boots in dev."""
import importlib.util
from tests.deploy.conftest import REPO

def _user_service_unsafe_check():
    spec = importlib.util.spec_from_file_location("user_service_config_probe", REPO / "services/user-service/src/config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._is_unsafe_secret

def test_dev_db_password_passes_user_service_validation(render):
    env = render("dev", environment="development").services["user-service"]["environment"]
    pw = env["USER_DB_PASSWORD"]
    assert pw == env["POSTGRES_PASSWORD"]
    assert "password" not in pw.lower()
    assert _user_service_unsafe_check()(pw) is False
