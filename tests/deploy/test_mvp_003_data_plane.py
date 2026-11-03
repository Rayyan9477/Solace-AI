# tests/deploy/test_mvp_003_data_plane.py
"""MVP-003: the orchestrator (and every app service) reaches Postgres, Kafka and its peers; app services start only after migrations."""
import pytest
from deploy.env.registry import APP_SERVICES

PEERS = {"SERVICE_SAFETY_SERVICE_URL": "http://safety-service:8002", "SERVICE_NOTIFICATION_SERVICE_URL": "http://notification-service:8003",
         "SERVICE_DIAGNOSIS_SERVICE_URL": "http://diagnosis-service:8004", "SERVICE_MEMORY_SERVICE_URL": "http://memory-service:8005",
         "SERVICE_THERAPY_SERVICE_URL": "http://therapy-service:8006", "SERVICE_PERSONALITY_SERVICE_URL": "http://personality-service:8007",
         "SERVICE_USER_SERVICE_URL": "http://user-service:8001"}

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_orchestrator_joins_the_data_network(render, profile):
    assert "data" in render(profile).services["orchestrator-service"]["networks"]

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_every_app_service_gets_the_data_plane(render, profile):
    r = render(profile)
    for svc in APP_SERVICES:
        env = r.services[svc]["environment"]
        assert (env["POSTGRES_HOST"], env["POSTGRES_USER"], env["POSTGRES_DATABASE"]) == ("postgres", "solace_app", "solace"), svc
        assert env["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:9092" and env["EVENTS_ENABLED"] == "true", svc
        assert {k: env.get(k) for k in PEERS} == PEERS, svc
        assert r.services[svc]["depends_on"]["migrate"]["condition"] == "service_completed_successfully", svc
        assert "/live" in " ".join(r.services[svc]["healthcheck"]["test"]), svc

def test_strict_db_clients_verify_tls(render):
    r = render("strict")
    for svc in (*APP_SERVICES, "migrate"):
        env = r.services[svc]["environment"]
        assert env["POSTGRES_SSL_MODE"] == "verify-full", svc
        assert env["POSTGRES_SSL_ROOT_CERT"] == "/run/secrets/postgres-ca.crt", svc
        assert "postgres-ca.crt" in {s["source"] for s in r.services[svc]["secrets"]}, svc
    assert "ssl=on" in " ".join(r.services["postgres"]["command"])

def test_migrate_runs_as_solace_migrator(render):
    m = render("dev").services["migrate"]
    assert m["environment"]["DATABASE_URL"].startswith("postgresql+asyncpg://solace_migrator:")
    assert m["command"] == ["python", "-m", "scripts.db.migrate", "upgrade", "head"] and m["restart"] == "no"

def test_built_services_are_tagged_for_ci():
    # raw YAML, not the render: WS-01's image gate (WS-01.9 --compose, WS-01.10) reads the files as written
    import yaml
    from scripts.ci.build_images import COMPOSE_TAG, image_ref   # WS-01.5: the reference format is WS-01's
    from scripts.ci.images import IMAGES
    from tests.deploy.conftest import REPO
    services = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    built = {name for name, spec in services.items() if "build" in spec}
    assert built == {*APP_SERVICES, "migrate"} == {i.name for i in IMAGES}
    for name in built:
        assert services[name]["build"]["context"] == ".", name
        assert services[name]["image"] == image_ref(name, COMPOSE_TAG), name
    assert services["migrate"]["build"]["dockerfile"] == "docker/migrate.Dockerfile"

def test_app_password_is_not_the_superuser_password(render):
    r = render("strict")
    su = r.services["postgres"]["environment"]["POSTGRES_PASSWORD"]
    assert su == r.env["POSTGRES_SUPERUSER_PASSWORD"]
    assert all(r.services[s]["environment"]["POSTGRES_PASSWORD"] != su for s in APP_SERVICES)

@pytest.mark.integration
@pytest.mark.e2e
def test_migrate_creates_head_as_solace_migrator(compose_project):
    compose_project.up(["postgres"])
    assert compose_project.run("migrate").returncode == 0
    roles = compose_project.psql("SELECT rolname, rolsuper, rolbypassrls FROM pg_roles "
                                 "WHERE rolname IN ('solace_app','solace_migrator') ORDER BY 1")
    # WS-03.2 deviation from the brief's (solace_migrator, False, False): init-db creates solace_migrator
    # BYPASSRLS (WS-02.1) and the migrate preflight refuses a migrator without it (WS-02.4), so data
    # migrations under FORCE RLS see every row. Neither role is a superuser; solace_app never bypasses RLS.
    assert roles == [("solace_app", False, False), ("solace_migrator", False, True)]
    assert compose_project.psql("SELECT count(*) FROM alembic_version") == [(1,)]

# --- WS-03.2 additions (hand-off from WS-02.1) ------------------------------------------------------

LEGACY_DB_PASSWORDS = {"user-service": "USER_DB_PASSWORD", "diagnosis-service": "DIAGNOSIS_DB_PASSWORD",
                       "therapy-service": "THERAPY_DB_PASSWORD", "memory-service": "MEMORY_DB_PASSWORD",
                       "personality-service": "PERSONALITY_DB_PASSWORD"}

def _password_mismatches(r):
    """Names (never values) of every role password that differs from the one init-db creates the role with."""
    from urllib.parse import unquote, urlsplit
    pg = r.services["postgres"]["environment"]
    app_pw, migrator_pw = pg["SOLACE_APP_PASSWORD"], pg["SOLACE_MIGRATOR_PASSWORD"]
    bad = []
    for svc in APP_SERVICES:
        env = r.services[svc]["environment"]
        if env.get("POSTGRES_PASSWORD") != app_pw:
            bad.append(f"{svc}:POSTGRES_PASSWORD")
        alias = LEGACY_DB_PASSWORDS.get(svc)
        if alias and env.get(alias) != app_pw:
            bad.append(f"{svc}:{alias}")
    for svc, name, user, expected in (("safety-service", "SAFETY_REPO_POSTGRES_DSN", "solace_app", app_pw),
                                      ("migrate", "DATABASE_URL", "solace_migrator", migrator_pw)):
        url = urlsplit(r.services[svc]["environment"][name])
        if url.username != user or url.password is None or unquote(url.password) != expected:
            bad.append(f"{svc}:{name}")
    return bad

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_role_passwords_are_the_ones_init_db_creates(render, profile):
    # hand-off 8: one POSTGRES_PASSWORD source for solace_app everywhere, POSTGRES_MIGRATOR_PASSWORD for the job
    assert _password_mismatches(render(profile)) == []

@pytest.mark.parametrize("profile", ["dev", "strict"])
def test_postgres_healthcheck_proves_tcp_before_the_role_check(render, profile):
    # the entrypoint's temporary init server listens on the unix socket only; a socket-only check can pass
    # before the real server accepts TCP. pg_isready needs no password, so it works under hostssl-only hba.
    test = render(profile).services["postgres"]["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    tcp, _, roles = test[1].partition(" && ")
    assert tcp == "pg_isready -h 127.0.0.1 -U postgres -d solace"
    assert roles.startswith("psql ") and "rolname in ('solace_app','solace_migrator')" in roles

@pytest.mark.integration
@pytest.mark.e2e
def test_postgres_healthcheck_requires_both_platform_roles(compose_project):
    # A first init that fails part-way leaves a data directory, so init-db never runs again; after the
    # restart pg_isready alone reports the role-less server healthy and the stack starts on it.
    import yaml
    from tests.deploy.conftest import REPO
    check = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["postgres"]["healthcheck"]["test"]
    assert check[0] == "CMD-SHELL"
    probe = f"if {{ {check[1]}; }} >/dev/null 2>&1; then echo HEALTHY; else echo UNHEALTHY; fi"
    compose_project.up(["postgres"])
    assert compose_project.exec("postgres", "sh", "-c", probe).split()[-1] == "HEALTHY"
    compose_project.exec("postgres", "psql", "-X", "-q", "-U", "postgres", "-d", "solace", "-c",
                         "ALTER ROLE solace_app RENAME TO solace_app_gone")
    assert compose_project.exec("postgres", "sh", "-c", probe).split()[-1] == "UNHEALTHY"
