"""Deploy-time TLS files for the bundled Postgres (D-15; MVP-010, MVP-271).

    python -m scripts.db.gen_certs --out DIR [--san NAME ...] [--ip ADDR ...] [--days N] [--force]

Writes three files into DIR (created when missing), under the IF-22 names:

* ``postgres-ca.crt``: a new self-signed CA (ECDSA P-256) that may sign end-entity certificates only. Clients
  trust it through ``POSTGRES_SSL_ROOT_CERT`` and verify the server with ``verify-full``.
* ``server.crt``: the server certificate (ECDSA P-256, SHA-256, extended key usage serverAuth) signed by that CA,
  for the DNS names given with ``--san`` and the addresses given with ``--ip``. With neither option the names are
  ``postgres`` and ``localhost`` and the address is ``127.0.0.1``; with either, exactly the names given.
* ``server.key``: the server's key (PKCS#8 PEM, unencrypted, mode 0600 where the OS supports file modes), which
  scripts/db/pg_tls_entrypoint.sh hands to the server.

The CA's own key only ever exists in memory and is discarded, so nothing else can be signed by this CA. Rotation
regenerates all three files (``--force``) and restarts the server and its clients. Every file is written to a
temporary name and then renamed into place.

Existing files are never replaced without ``--force`` (exit 1). Only file names are printed, never certificate or
key material. Exit codes: 0 written; 1 refused or failed; 2 bad arguments. ``main(argv)`` returns the code.
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import re
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CA_FILE: Final = "postgres-ca.crt"
CERT_FILE: Final = "server.crt"
KEY_FILE: Final = "server.key"
FILES: Final = (CA_FILE, CERT_FILE, KEY_FILE)

DEFAULT_DNS_NAMES: Final = ("postgres", "localhost")
DEFAULT_IP_ADDRESSES: Final = ("127.0.0.1",)
DEFAULT_DAYS: Final = 365
MAX_DAYS: Final = 3650
ORGANIZATION: Final = "Solace-AI"

_CLOCK_SKEW: Final = dt.timedelta(minutes=5)  # not_valid_before is back-dated by this much
_DNS_LABEL: Final = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)")
_MAX_COMMON_NAME: Final = 64
_FILE_MODES: Final = {CA_FILE: 0o644, CERT_FILE: 0o644, KEY_FILE: 0o600}

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _note(message: str) -> None:
    print(f"gen_certs: {message}", file=sys.stderr, flush=True)


def _dns_name(text: str) -> str:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        pass
    else:  # a dNSName SAN never matches an IP host: TLS clients compare IP hosts with iPAddress SANs only
        raise argparse.ArgumentTypeError(f"{text!r} is an IP address; pass it with --ip")
    if not text or len(text) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in text.split(".")):
        raise argparse.ArgumentTypeError(f"not a DNS name: {text!r}")
    return text


def _ip_address(text: str) -> IpAddress:
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an IP address: {text!r}") from None


def _days(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number of days: {text!r}") from None
    if not 1 <= value <= MAX_DAYS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_DAYS} days: {text!r}")
    return value


def _out_dir(text: str) -> Path:
    if not text.strip():
        raise argparse.ArgumentTypeError("--out must name a directory")
    return Path(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.db.gen_certs",
        description="Write a new CA (postgres-ca.crt) and a server certificate and key (server.crt, server.key) "
                    "for the bundled Postgres. The CA's key is not kept.")
    parser.add_argument("--out", required=True, type=_out_dir, metavar="DIR",
                        help="directory for the three files (created when missing)")
    parser.add_argument("--san", action="append", type=_dns_name, dest="dns_names", metavar="NAME",
                        help="a DNS name the server certificate is valid for (repeatable)")
    parser.add_argument("--ip", action="append", type=_ip_address, dest="ip_addresses", metavar="ADDR",
                        help="an IP address the server certificate is valid for (repeatable)")
    parser.add_argument("--days", type=_days, default=DEFAULT_DAYS, metavar="N",
                        help=f"validity in days (default %(default)s, at most {MAX_DAYS})")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    return parser


def _unique(values: Sequence[object]) -> list:
    return list(dict.fromkeys(values))


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION),
                      x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _key_usage(*, signing_certificates: bool) -> x509.KeyUsage:
    return x509.KeyUsage(digital_signature=not signing_certificates, content_commitment=False,
                         key_encipherment=False, data_encipherment=False, key_agreement=False,
                         key_cert_sign=signing_certificates, crl_sign=signing_certificates,
                         encipher_only=False, decipher_only=False)


def generate(dns_names: Sequence[str], ip_addresses: Sequence[IpAddress], days: int) -> dict[str, bytes]:
    """The PEM bytes of the three files, keyed by file name. The CA key is a local and is dropped on return."""
    if not dns_names and not ip_addresses:
        raise ValueError("the server certificate needs at least one DNS name or IP address")
    now = dt.datetime.now(dt.UTC)
    not_before, not_after = now - _CLOCK_SKEW, now + dt.timedelta(days=days)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = _name(f"{ORGANIZATION} Postgres CA {secrets.token_hex(4)}")
    ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(signing_certificates=True), critical=True)
        .add_extension(ca_ski, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    server_key = ec.generate_private_key(ec.SECP256R1())
    first = dns_names[0] if dns_names else str(ip_addresses[0])
    common_name = first if len(first) <= _MAX_COMMON_NAME else f"{ORGANIZATION} Postgres server"
    alt_names: list[x509.GeneralName] = [x509.DNSName(n) for n in dns_names]
    alt_names += [x509.IPAddress(a) for a in ip_addresses]
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(_name(common_name))
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(signing_certificates=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(server_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ca_ski), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    del ca_key
    return {
        CA_FILE: ca_cert.public_bytes(serialization.Encoding.PEM),
        CERT_FILE: server_cert.public_bytes(serialization.Encoding.PEM),
        KEY_FILE: server_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()),
    }


def _write_all(out: Path, material: dict[str, bytes]) -> None:
    """Write every file to a temporary name in ``out``, then rename each into place; no temporary file survives."""
    out.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, data in material.items():
            temporary = out / f".{name}.{secrets.token_hex(6)}.tmp"
            mode = _FILE_MODES[name]
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), mode)
            staged.append((temporary, out / name))
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.chmod(temporary, mode)  # exactly this mode, whatever the umask (Windows: the read-only bit only)
        for temporary, final in staged:
            os.replace(temporary, final)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) or a usage error (2), already reported on stderr
        return exc.code if isinstance(exc.code, int) else 2
    dns_names: list[str] = _unique(args.dns_names or [])
    ip_addresses: list[IpAddress] = _unique(args.ip_addresses or [])
    if not dns_names and not ip_addresses:
        dns_names = list(DEFAULT_DNS_NAMES)
        ip_addresses = [ipaddress.ip_address(a) for a in DEFAULT_IP_ADDRESSES]
    out: Path = args.out
    if out.exists() and not out.is_dir():
        _note(f"error: --out {out} exists and is not a directory")
        return 1
    existing = [name for name in FILES if os.path.lexists(out / name)]
    if existing and not args.force:
        _note(f"refusing to overwrite {', '.join(existing)} in {out}; pass --force to replace all three files")
        return 1
    try:
        _write_all(out, generate(dns_names, ip_addresses, args.days))
    except OSError as exc:
        _note(f"error: could not write the files to {out}: {type(exc).__name__}: {exc.strerror or 'failed'}")
        return 1
    print(f"gen_certs: wrote {', '.join(FILES)} (the CA's key is not kept)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
