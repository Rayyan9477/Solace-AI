# tests/deploy/test_mvp_250_caddy_edge.py
"""MVP-250 (+IF-18 edge rules): health checks probe /live with failure tolerance and retry windows;
internal paths are never routed; site addresses come from env; Caddy holds only its own variables."""
import json, re
import pytest
from tests.deploy.conftest import REPO

CADDYFILE = (REPO / "Caddyfile").read_text(encoding="utf-8")
# WS-03.F1: every never-routed entry is a prefix (SEC-1), and /ready and /health are never routed (owner ruling)
NEVER = ("/metrics*", "/docs*", "/redoc*", "/openapi.json*", "/status*", "/ready*", "/health*", "/api/v1/internal*")

def _proxy_block(upstream: str) -> str:
    m = re.search(rf"reverse_proxy\s+{re.escape(upstream)}\s*\{{(.*?)\n\s*\}}", CADDYFILE, re.S)
    assert m, upstream
    return m.group(1)

@pytest.mark.parametrize("upstream", ["orchestrator-service:8000", "user-service:8001"])
def test_health_checks_probe_live_and_tolerate_one_slow_probe(upstream):
    block = _proxy_block(upstream)
    assert "health_uri /live" in block and "/health" not in block
    assert int(re.search(r"health_fails\s+(\d+)", block).group(1)) >= 3
    assert "lb_try_duration" in block

def test_site_addresses_come_from_env():
    assert "{$CADDY_API_SITE}" in CADDYFILE and "{$CADDY_AUTH_SITE}" in CADDYFILE
    assert "api.{$DOMAIN}" not in CADDYFILE and "kafka.{$DOMAIN}" not in CADDYFILE

def test_internal_paths_are_blocked_and_no_access_log():
    snippet = re.search(r"\(never_routed\)\s*\{(.*?)\n\}", CADDYFILE, re.S).group(1)
    for path in NEVER:
        assert path in snippet, path
    assert not re.search(r"^\s*log\b", CADDYFILE, re.M)

def test_caddy_runs_in_both_profiles_with_minimal_env(render):
    dev, strict = render("dev"), render("strict")
    assert "caddy" in dev.services and all(p["host_ip"] == "127.0.0.1" for p in dev.services["caddy"]["ports"])
    assert set(strict.services["caddy"]["environment"]) <= {"DOMAIN", "CADDY_ADMIN_EMAIL", "BASIC_AUTH_HASH", "CADDY_API_SITE", "CADDY_AUTH_SITE"}
    assert strict.services["caddy"]["environment"]["CADDY_API_SITE"] == f"api.{strict.env['DOMAIN']}"

@pytest.mark.integration
@pytest.mark.realinfra
def test_adapted_config_blocks_metrics_before_proxying(caddy_adapt):
    cfg = caddy_adapt({"CADDY_API_SITE": "http://api.example.test", "CADDY_AUTH_SITE": "http://auth.example.test",
                       "DOMAIN": "example.test", "CADDY_ADMIN_EMAIL": "ops@example.test",
                       "BASIC_AUTH_HASH": "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"})
    text = json.dumps(cfg)
    assert '"uri": "/live"' in text and '"try_duration"' in text
    assert text.index('"/metrics*"') < text.index('"orchestrator-service:8000"')

# --- WS-03.5 additions ------------------------------------------------------------------------------
import secrets  # noqa: E402,I001  (below the brief's verbatim block)
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from tests.deploy.conftest import CADDY_IMAGE, _free_port  # noqa: E402

API_SITE_ENV = {"CADDY_API_SITE": "http://api.example.test", "CADDY_AUTH_SITE": "http://auth.example.test",
                "DOMAIN": "example.test", "CADDY_ADMIN_EMAIL": "ops@example.test",
                "BASIC_AUTH_HASH": "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"}

def _site_routes(cfg: dict, host: str) -> list:
    """The handler list of the one server route whose host matcher names ``host``."""
    for server in cfg["apps"]["http"]["servers"].values():
        for route in server.get("routes", []):
            if any(host in m.get("host", []) for m in route.get("match", [])):
                return route["handle"]
    raise AssertionError(f"no site for {host}")

def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)

def _first_handler_order(handlers) -> list:
    """Handler names in document order (static_response, reverse_proxy, ...)."""
    return [n["handler"] for n in _walk(handlers) if "handler" in n]

@pytest.mark.parametrize("upstream", ["orchestrator-service:8000", "user-service:8001"])
def test_health_check_and_retry_windows_match_the_brief(upstream):
    block = _proxy_block(upstream)
    expected = {"health_interval": "10s", "health_timeout": "5s", "health_fails": "3", "health_passes": "1",
                "lb_try_duration": "10s", "lb_try_interval": "250ms"}
    found = {k: v for k, v in re.findall(r"^\s*(\w+)\s+(\S+)\s*$", block, re.M) if k in expected}
    assert found == expected

def test_both_sites_import_never_routed_first_inside_a_route():
    # route{} keeps literal order: the 404 must come before the proxy, in both sites
    for upstream in ("orchestrator-service:8000", "user-service:8001"):
        m = re.search(rf"route\s*\{{\s*import never_routed\s+reverse_proxy\s+{re.escape(upstream)}", CADDYFILE)
        assert m, upstream

def test_no_dev_only_vhost_and_no_wildcard_trust():
    code = "\n".join(line.split("#", 1)[0] for line in CADDYFILE.splitlines())
    assert "kafka-ui" not in code and "kafka." not in code
    # Caddy must not trust a client's X-Forwarded-For: the app trusts exactly one hop (Caddy's)
    assert "trusted_proxies" not in code

def test_dev_edge_listens_on_loopback_8088_with_http_sites(render):
    caddy = render("dev").services["caddy"]
    assert [(p["host_ip"], str(p["published"]), p["target"]) for p in caddy["ports"]] == [("127.0.0.1", "8088", 8088)]
    env = caddy["environment"]
    assert (env["CADDY_API_SITE"], env["CADDY_AUTH_SITE"]) == ("http://api.localhost:8088", "http://auth.localhost:8088")
    assert caddy["image"] == CADDY_IMAGE and set(caddy["networks"]) == {"frontend", "backend"}
    assert {s: d["condition"] for s, d in caddy["depends_on"].items()} == {
        "orchestrator-service": "service_started", "user-service": "service_started"}

def test_dev_basic_auth_hash_is_a_real_bcrypt_hash_intact_through_compose(render, registry):
    import bcrypt
    value = render("dev").services["caddy"]["environment"]["BASIC_AUTH_HASH"]
    row = next(v for v in registry if v.name == "BASIC_AUTH_HASH")
    assert value == row.dev_default and re.fullmatch(r"\$2[aby]\$\d\d\$[./A-Za-z0-9]{53}", value)
    # the hash of a discarded random password: no guessable dev password opens the dashboards
    for guess in ("admin", "password", "solace", "changeme", "dev", ""):
        assert not bcrypt.checkpw(guess.encode(), value.encode()), guess

def test_strict_edge_publishes_80_443_only_and_uses_plain_hosts(render):
    caddy = render("strict").services["caddy"]
    ports = sorted((str(p["published"]), p["target"], p.get("protocol", "tcp")) for p in caddy["ports"])
    assert ports == [("443", 443, "tcp"), ("443", 443, "udp"), ("80", 80, "tcp")]
    env, domain = caddy["environment"], render("strict").env["DOMAIN"]
    assert env["CADDY_AUTH_SITE"] == f"auth.{domain}"
    assert not env["CADDY_API_SITE"].startswith("http")   # automatic HTTPS applies only to a plain host

def test_strict_site_addresses_ignore_an_env_override(render):
    # strict never takes CADDY_API_SITE/CADDY_AUTH_SITE from the env file (no http:// edge in production)
    r = render("strict", overrides={"CADDY_API_SITE": "http://evil.example.test", "CADDY_AUTH_SITE": "http://evil.example.test"})
    env = r.services["caddy"]["environment"]
    assert (env["CADDY_API_SITE"], env["CADDY_AUTH_SITE"]) == (f"api.{r.env['DOMAIN']}", f"auth.{r.env['DOMAIN']}")

@pytest.mark.parametrize("source", ["DOMAIN", "BASIC_AUTH_HASH", "CADDY_ADMIN_EMAIL"])
def test_strict_render_requires_edge_sources(render, source):
    from deploy.env.compose_render import ComposeRenderError
    with pytest.raises(ComposeRenderError, match=source):
        render("strict", drop=[source])

@pytest.mark.integration
@pytest.mark.realinfra
@pytest.mark.parametrize("host,upstream", [("api.example.test", "orchestrator-service:8000"),
                                           ("auth.example.test", "user-service:8001")])
def test_adapted_never_routed_precedes_the_proxy_in_both_sites(caddy_adapt, host, upstream):
    handlers = _site_routes(caddy_adapt(API_SITE_ENV), host)
    paths = [p for n in _walk(handlers) for m in n.get("match", []) if isinstance(m, dict) for p in m.get("path", [])]
    for path in NEVER:
        assert path in paths, path
    order = _first_handler_order(handlers)
    assert order.index("static_response") < order.index("reverse_proxy")
    proxy = next(n for n in _walk(handlers) if n.get("handler") == "reverse_proxy")
    assert [u["dial"] for u in proxy["upstreams"]] == [upstream]
    active = proxy["health_checks"]["active"]
    assert (active["uri"], active["passes"], active["fails"]) == ("/live", 1, 3)

@pytest.mark.integration
@pytest.mark.realinfra
def test_adapted_strict_sites_are_plain_hosts(caddy_adapt):
    cfg = caddy_adapt({**API_SITE_ENV, "CADDY_API_SITE": "api.example.test", "CADDY_AUTH_SITE": "auth.example.test"})
    # plain hosts: Caddy serves them on :443 with automatic HTTPS (and redirects :80)
    listens = {a for s in cfg["apps"]["http"]["servers"].values() for a in s["listen"]}
    assert ":443" in listens
    assert _site_routes(cfg, "api.example.test") and _site_routes(cfg, "auth.example.test")

@pytest.mark.integration
@pytest.mark.realinfra
def test_running_dev_edge_answers_404_for_internal_paths():
    """The dev Caddyfile boots in the pinned image; internal paths 404 at the edge, the rest reach the proxy."""
    docker = shutil.which("docker") or "docker"
    port, name = _free_port(), f"ws035-caddy-{secrets.token_hex(4)}"
    env = {"CADDY_API_SITE": "http://api.localhost:8088", "CADDY_AUTH_SITE": "http://auth.localhost:8088",
           "DOMAIN": "localhost", "CADDY_ADMIN_EMAIL": "dev-ops@example.invalid",
           "BASIC_AUTH_HASH": API_SITE_ENV["BASIC_AUTH_HASH"]}
    cmd = [docker, "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8088",
           "-v", f"{(REPO / 'Caddyfile').as_posix()}:/etc/caddy/Caddyfile:ro"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    # S603: a docker argv list (no shell) built from the test's own literals
    subprocess.run([*cmd, CADDY_IMAGE], check=True, capture_output=True, timeout=120)  # noqa: S603

    def status(host: str, path: str) -> int:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Host": f"{host}:8088"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (a loopback http URL)
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                status("api.localhost", "/metrics")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        for host in ("api.localhost", "auth.localhost"):
            for path in ("/metrics", "/metrics/x", "/docs", "/redoc", "/openapi.json", "/status",
                         "/api/v1/internal/x", "/METRICS", "/%6Detrics"):
                assert status(host, path) == 404, (host, path)
        # a public path is proxied: no upstream on this network, so the proxy answers 502/503
        assert status("api.localhost", "/api/v1/chat") in (502, 503)
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, check=False)  # noqa: S603

# --- WS-03.F1 (W0-close review SEC-1; owner ruling: /ready and /health never routed) -------------------
# Starlette compiles an exact route such as /metrics to ^/metrics$ and matches it with re.match, and Python's $
# also matches just before a trailing newline, so GET /metrics%0A (decoded /metrics\n) reached the app past an
# exact never-routed entry. Caddy's path matcher lowercases and cleans the decoded path and treats a single
# trailing * as a prefix match, so a prefix entry covers every suffix and case variant.
import http.client  # noqa: E402
import urllib.parse  # noqa: E402

NEVER_BASES = tuple(p[:-1] for p in NEVER)
# suffixes (decoded control characters, slash, dot, space, path parameter) and case or encoding of the base
_VARIANT_SUFFIXES = ("", "/", "/x", "%0A", "%0a", "%0D%0A", "%0D", "%09", "%0B", "%0C", "%1F", "%7F", "%20",
                     "%2F", "%2e", ".", ";x", "%3B", "%3Bx")

def _never_routed_snippet() -> str:
    return re.search(r"\(never_routed\)\s*\{(.*?)\n\}", CADDYFILE, re.S).group(1)

def _never_routed_paths() -> list:
    return re.search(r"@never_routed\s+path\s+([^\n]+)", _never_routed_snippet()).group(1).split()

def _variants(base: str) -> list:
    out = [base + suffix for suffix in _VARIANT_SUFFIXES]
    out += [base.upper(), base.title(), "/" + base, "/./" + base[1:], "/x/.." + base, base + "/../" + base.rsplit("/", 1)[1],
            "/%" + format(ord(base[1]), "02X") + base[2:]]   # e.g. /%6Detrics
    return out

def test_every_never_routed_entry_is_a_prefix():
    # exactly the bases, each as a prefix: no exact entry is left for a %0A (or other suffix) to slip past
    assert sorted(_never_routed_paths()) == sorted(NEVER)

def test_live_stays_routed_and_is_the_only_health_route():
    # /live stays public (an uptime monitor and Caddy's own active checks); /ready and /health do not
    assert not any("/live".startswith(p.rstrip("*")) for p in _never_routed_paths())
    for base in ("/ready", "/health"):
        assert base + "*" in _never_routed_paths(), base

def test_control_characters_in_a_path_are_refused_before_the_proxy():
    # defence in depth for every exact route (public ones too): a decoded control character never reaches an app
    snippet = _never_routed_snippet()
    assert re.search(r"@control_characters\s+path_regexp\s+\[\[:cntrl:\]\]\s*$", snippet, re.M)
    assert re.search(r"^\s*respond\s+@control_characters\s+400\s*$", snippet, re.M)
    # the never-routed 404 comes first, so a never-routed path answers 404 whatever its suffix
    assert snippet.index("respond @never_routed 404") < snippet.index("respond @control_characters 400")

def _raw_status(port: int, host: str, path: str) -> int:
    """One GET with ``path`` sent as is (no client-side normalisation or re-encoding)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", f"{host}:8088")
        conn.endheaders()
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()

def _start_dev_edge() -> tuple:
    """The repo Caddyfile in the pinned image on a loopback port, dev (plain-HTTP) sites; (docker, name, port)."""
    docker = shutil.which("docker") or "docker"
    port, name = _free_port(), f"ws03f1-caddy-{secrets.token_hex(4)}"
    env = {"CADDY_API_SITE": "http://api.localhost:8088", "CADDY_AUTH_SITE": "http://auth.localhost:8088",
           "DOMAIN": "localhost", "CADDY_ADMIN_EMAIL": "dev-ops@example.invalid",
           "BASIC_AUTH_HASH": API_SITE_ENV["BASIC_AUTH_HASH"]}
    cmd = [docker, "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:8088",
           "-v", f"{(REPO / 'Caddyfile').as_posix()}:/etc/caddy/Caddyfile:ro"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    subprocess.run([*cmd, CADDY_IMAGE], check=True, capture_output=True, timeout=120)  # noqa: S603
    deadline = time.monotonic() + 60
    while True:
        try:
            _raw_status(port, "api.localhost", "/metrics")
            return docker, name, port
        except OSError:
            if time.monotonic() >= deadline:
                subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, check=False)  # noqa: S603
                raise
            time.sleep(1)

@pytest.mark.integration
@pytest.mark.realinfra
def test_running_edge_answers_404_for_every_never_routed_variant():
    docker, name, port = _start_dev_edge()
    try:
        leaked = []
        for host in ("api.localhost", "auth.localhost"):
            for base in NEVER_BASES:
                for path in _variants(base):
                    status = _raw_status(port, host, path)
                    if status != 404:
                        leaked.append((host, path, status))
        assert leaked == []
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, check=False)  # noqa: S603

@pytest.mark.integration
@pytest.mark.realinfra
def test_running_edge_refuses_control_characters_and_still_proxies_live():
    docker, name, port = _start_dev_edge()
    try:
        for host in ("api.localhost", "auth.localhost"):
            for path in ("/live%0A", "/api/v1/chat%0A", "/api/v1/auth/login%0D%0A", "/x%00", "/x%7F"):
                assert _raw_status(port, host, path) == 400, (host, path)
            # /live is still proxied: no upstream on this network, so the proxy answers 502/503 (never 404 or 400)
            assert _raw_status(port, host, "/live") in (502, 503), host
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, check=False)  # noqa: S603

@pytest.mark.integration
@pytest.mark.realinfra
@pytest.mark.parametrize("sites", [("http://api.example.test", "http://auth.example.test"),
                                   ("api.example.test", "auth.example.test")], ids=["dev", "strict"])
def test_caddy_validate_accepts_the_caddyfile(sites):
    docker = shutil.which("docker") or "docker"
    env = {**API_SITE_ENV, "CADDY_API_SITE": sites[0], "CADDY_AUTH_SITE": sites[1]}
    cmd = [docker, "run", "--rm", "--network", "none", "-v", f"{(REPO / 'Caddyfile').as_posix()}:/etc/caddy/Caddyfile:ro"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += [CADDY_IMAGE, "caddy", "validate", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",  # noqa: S603
                          timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Valid configuration" in proc.stdout + proc.stderr