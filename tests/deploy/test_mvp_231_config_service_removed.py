"""MVP-231: config-service (zero consumers, permanently unhealthy) is not deployed in the MVP profiles."""
import pytest

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_config_service_not_deployed(render, profile):
    r = render(profile)
    names = set(r.services)  # names only: an assert on the rendered dict would print generated values
    assert "config-service" not in names
    assert all("config-service" not in (svc.get("depends_on") or {}) for svc in r.services.values())
