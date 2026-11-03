# tests/deploy/test_mvp_117_basic_auth_hash.py
"""MVP-117: a bcrypt hash survives compose interpolation intact when single-quoted, and the docs and .env.example say so."""
from tests.deploy.conftest import REPO

HASH = "$2a$14$Zkx19XLiW6VYouLHR5NmfOFU0z2GTNmpkT/5qqR7hx4IjWJPDhjvG"

def test_single_quoted_hash_reaches_caddy_intact(render):
    r = render("strict", overrides={"BASIC_AUTH_HASH": HASH})
    assert r.services["caddy"]["environment"]["BASIC_AUTH_HASH"] == HASH

def test_env_example_single_quotes_the_hash():
    line = next(l for l in (REPO / ".env.example").read_text(encoding="utf-8").splitlines() if l.startswith("BASIC_AUTH_HASH="))
    assert line == "BASIC_AUTH_HASH=''"

def test_runbook_tells_operators_to_single_quote():
    assert "BASIC_AUTH_HASH='" in (REPO / "docs/STAGING-BRINGUP.md").read_text(encoding="utf-8")

# --- WS-03.5 additions ------------------------------------------------------------------------------
import re  # noqa: E402  (below the brief's verbatim block)

def test_registry_marks_the_hash_bcrypt_secret_in_both_profiles(registry):
    row = next(v for v in registry if v.name == "BASIC_AUTH_HASH")
    assert (row.format, row.secret, row.required_in_strict, set(row.profiles)) == ("bcrypt", True, True, {"dev", "strict"})

def test_dev_file_escapes_every_dollar_of_the_default_hash():
    # an unescaped $2a in the base file would be interpolated away (MVP-117)
    line = next((l for l in (REPO / "docker-compose.yml").read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("BASIC_AUTH_HASH:")), "")
    assert line, "docker-compose.yml sets no BASIC_AUTH_HASH"
    default = re.search(r"\$\{BASIC_AUTH_HASH:-(.*)\}", line).group(1)
    assert default.startswith("$$2") and default.count("$$") == 3 and "$" not in default.replace("$$", "")

def test_runbook_explains_why_the_hash_is_single_quoted():
    text = (REPO / "docs/STAGING-BRINGUP.md").read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if l.startswith("BASIC_AUTH_HASH="))
    assert line.startswith("BASIC_AUTH_HASH='") and line.rstrip().endswith("'")
    assert "single quote" in text.lower() or "single-quote" in text.lower()
