# tests/deploy/test_mvp_002_weaviate_health.py
"""MVP-002: the Weaviate healthcheck cannot depend on curl (absent from the image), and nothing on the chat path waits for Weaviate."""
import pytest

def test_weaviate_is_opt_in(render):
    assert "weaviate" not in render("dev").services
    assert "weaviate" in render("dev", profiles=["vector"]).services

def test_weaviate_healthcheck_uses_wget(render):
    hc = render("dev", profiles=["vector"]).services["weaviate"]["healthcheck"]
    test = " ".join(hc["test"])
    assert "curl" not in test and "wget" in test and "/v1/.well-known/ready" in test
    assert hc.get("start_period")

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_memory_service_does_not_wait_for_weaviate(render, profile):
    assert "weaviate" not in (render(profile).services["memory-service"].get("depends_on") or {})

@pytest.mark.integration
@pytest.mark.e2e
def test_weaviate_becomes_healthy(compose_project):
    compose_project.up(["weaviate"], profiles=["vector"])  # up --wait fails if the healthcheck never passes

# --- WS-03.4 additions ------------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_memory_service_keeps_weaviate_off_by_default(render, profile):
    # D-04: Weaviate writes off, retrieval stays keyword/BM25
    env = render(profile).services["memory-service"]["environment"]
    assert (env["WEAVIATE_ENABLED"], env["MEMORY_EMBEDDINGS_PROVIDER"]) == ("false", "none")

def test_weaviate_stays_opt_in_in_strict(render):
    # names only: a strict render holds generated values, so a failure must never print the services
    assert "weaviate" not in set(render("strict").services)
    weaviate = render("strict", profiles=["vector"]).services["weaviate"]
    healthcheck, published = " ".join(weaviate["healthcheck"]["test"]), bool(weaviate.get("ports"))
    assert "wget" in healthcheck and not published

def test_weaviate_healthcheck_waits_for_a_slow_start(render):
    hc = render("dev", profiles=["vector"]).services["weaviate"]["healthcheck"]
    assert (hc["start_period"], hc["retries"]) == ("1m0s", 10)
