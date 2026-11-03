# tests/migrations/test_mvp_010_gen_certs.py
"""MVP-010: deploy-time CA and server certificate for verify-full against host 'postgres'."""
from __future__ import annotations

from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID


def test_mvp_010_gen_certs_writes_ca_and_server_pair(tmp_path: Path, capsys) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path), "--san", "postgres", "--san", "localhost", "--ip", "127.0.0.1"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {"postgres-ca.crt", "server.crt", "server.key"}
    ca = x509.load_pem_x509_certificate((tmp_path / "postgres-ca.crt").read_bytes())
    srv = x509.load_pem_x509_certificate((tmp_path / "server.crt").read_bytes())
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(san.get_values_for_type(x509.DNSName)) == {"postgres", "localhost"}
    assert {str(i) for i in san.get_values_for_type(x509.IPAddress)} == {"127.0.0.1"}
    ca.public_key().verify(srv.signature, srv.tbs_certificate_bytes, ec.ECDSA(srv.signature_hash_algorithm))
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is True
    assert ExtendedKeyUsageOID.SERVER_AUTH in srv.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    out = capsys.readouterr().out
    assert "BEGIN" not in out and "PRIVATE" not in out


def test_mvp_010_gen_certs_refuses_to_overwrite(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    assert gen_certs.main(["--out", str(tmp_path)]) == 1
    assert gen_certs.main(["--out", str(tmp_path), "--force"]) == 0


# --- WS-02.5 additions -------------------------------------------------------------------------------------------
# Defaults, the key pair, the CA's shape, validity, file modes, argument validation, refusal hygiene and the
# ``python -m`` entrypoint. Nothing here prints a certificate or a key.

import datetime as dt  # noqa: E402
import os  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("postgres-ca.crt", "server.crt", "server.key")


def _load(out: Path) -> tuple[x509.Certificate, x509.Certificate, ec.EllipticCurvePrivateKey]:
    ca = x509.load_pem_x509_certificate((out / "postgres-ca.crt").read_bytes())
    srv = x509.load_pem_x509_certificate((out / "server.crt").read_bytes())
    key = serialization.load_pem_private_key((out / "server.key").read_bytes(), password=None)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    return ca, srv, key


def test_gen_certs_defaults_are_postgres_localhost_and_loopback(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    _, srv, _ = _load(tmp_path)
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert set(san.get_values_for_type(x509.DNSName)) == {"postgres", "localhost"}
    assert {str(i) for i in san.get_values_for_type(x509.IPAddress)} == {"127.0.0.1"}


def test_gen_certs_explicit_names_replace_the_defaults(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path), "--san", "db.internal", "--ip", "::1"]) == 0
    _, srv, _ = _load(tmp_path)
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["db.internal"]
    assert [str(i) for i in san.get_values_for_type(x509.IPAddress)] == ["::1"]


def test_gen_certs_key_pair_and_ca_shape(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    ca, srv, key = _load(tmp_path)
    # ECDSA P-256 for both keys; the server key is the one in server.crt.
    assert isinstance(key.curve, ec.SECP256R1)
    assert isinstance(ca.public_key(), ec.EllipticCurvePublicKey) and isinstance(ca.public_key().curve, ec.SECP256R1)
    assert key.public_key().public_numbers() == srv.public_key().public_numbers()
    # A self-signed root that may sign only end-entity certificates; the server certificate is not a CA.
    assert ca.issuer == ca.subject and srv.issuer == ca.subject
    ca.public_key().verify(ca.signature, ca.tbs_certificate_bytes, ec.ECDSA(ca.signature_hash_algorithm))
    ca_bc = ca.extensions.get_extension_for_class(x509.BasicConstraints)
    assert ca_bc.critical and ca_bc.value.ca is True and ca_bc.value.path_length == 0
    ca_ku = ca.extensions.get_extension_for_class(x509.KeyUsage)
    assert ca_ku.critical and ca_ku.value.key_cert_sign and ca_ku.value.crl_sign
    srv_bc = srv.extensions.get_extension_for_class(x509.BasicConstraints)
    assert srv_bc.critical and srv_bc.value.ca is False
    srv_ku = srv.extensions.get_extension_for_class(x509.KeyUsage)
    assert srv_ku.critical and srv_ku.value.digital_signature and not srv_ku.value.key_cert_sign
    assert list(srv.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [ExtendedKeyUsageOID.SERVER_AUTH]
    # Chain-building links (OpenSSL's strict mode wants both).
    aki = srv.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value.key_identifier
    assert aki == ca.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value.digest
    assert srv.signature_hash_algorithm.name == "sha256"


def test_gen_certs_days_sets_the_validity(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    before = dt.datetime.now(dt.UTC)
    assert gen_certs.main(["--out", str(tmp_path), "--days", "30"]) == 0
    ca, srv, _ = _load(tmp_path)
    assert srv.not_valid_before_utc <= before  # back-dated a little for clock skew
    assert abs(srv.not_valid_after_utc - (before + dt.timedelta(days=30))) < dt.timedelta(minutes=5)
    assert ca.not_valid_before_utc <= srv.not_valid_before_utc and ca.not_valid_after_utc >= srv.not_valid_after_utc


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_gen_certs_key_is_mode_0600(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    assert stat.S_IMODE((tmp_path / "server.key").stat().st_mode) == 0o600


@pytest.mark.parametrize("argv", [["--san", "not a name"], ["--san", ""], ["--san", "-bad.example"],
                                  ["--ip", "999.1.1.1"], ["--ip", "postgres"], ["--days", "0"], ["--days", "x"],
                                  [],
                                  # Fix round 1 (R2): an IP literal as a DNS SAN never matches an IP host.
                                  ["--san", "127.0.0.1"], ["--san", "10.0.0.5"]])
def test_gen_certs_rejects_bad_arguments_without_writing(tmp_path: Path, argv: list[str]) -> None:
    from scripts.db import gen_certs

    out = tmp_path / "out"
    full = ["--out", str(out), *argv] if argv else []  # [] -> --out missing
    assert gen_certs.main(full) == 2
    assert not out.exists()


def test_gen_certs_ip_as_san_points_at_ip(tmp_path: Path, capsys) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path / "out"), "--san", "127.0.0.1"]) == 2
    assert "pass it with --ip" in capsys.readouterr().err


def test_gen_certs_refusal_names_files_and_leaves_them_untouched(tmp_path: Path, capsys) -> None:
    from scripts.db import gen_certs

    (tmp_path / "server.key").write_bytes(b"pre-existing")
    assert gen_certs.main(["--out", str(tmp_path)]) == 1
    assert (tmp_path / "server.key").read_bytes() == b"pre-existing"
    assert {p.name for p in tmp_path.iterdir()} == {"server.key"}
    captured = capsys.readouterr()
    assert "server.key" in captured.err and "--force" in captured.err
    assert "pre-existing" not in captured.out + captured.err


def test_gen_certs_force_makes_a_new_ca_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    assert gen_certs.main(["--out", str(tmp_path)]) == 0
    first = {n: (tmp_path / n).read_bytes() for n in NAMES}
    assert gen_certs.main(["--out", str(tmp_path), "--force"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == set(NAMES)
    assert all((tmp_path / n).read_bytes() != first[n] for n in NAMES)  # rotation regenerates all three


def test_gen_certs_out_must_be_a_directory(tmp_path: Path) -> None:
    from scripts.db import gen_certs

    target = tmp_path / "a-file"
    target.write_bytes(b"")
    assert gen_certs.main(["--out", str(target)]) == 1
    assert target.read_bytes() == b""


def test_gen_certs_runs_as_a_module(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "certs"
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"}
    r = subprocess.run([sys.executable, "-m", "scripts.db.gen_certs", "--out", str(out), "--san", "postgres"],  # noqa: S603
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    assert {p.name for p in out.iterdir()} == set(NAMES)
    for name in NAMES:
        assert name in r.stdout
    assert "BEGIN" not in r.stdout + r.stderr and "PRIVATE" not in r.stdout + r.stderr
