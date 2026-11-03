"""D-07: SOPS-encrypted secrets files reach checkpoint trees; age private keys and decrypted secrets never do."""
from __future__ import annotations

import secrets
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci import dockerfiles, secret_scan

ROOT = Path(__file__).resolve().parents[2]
BECH32 = "QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L"
ENC = "ENC[AES256_GCM,data:c3ludGhldGlj,iv:AAAAAAAAAAAAAAAAAAAAAA==,tag:AAAAAAAAAAAAAAAAAAAAAA==,type:str]"
SOPS_META = ("sops:\n  age:\n    - recipient: age1" + "q" * 58 + "\n      enc: |\n"
             "        -----BEGIN AGE ENCRYPTED FILE-----\n        c3ludGhldGlj\n        -----END AGE ENCRYPTED FILE-----\n"
             "  lastmodified: '2026-10-19T00:00:00Z'\n  mac: " + ENC + "\n  version: 3.9.0\n")


def _synthetic_age_identity() -> str:
    # random characters in the identity format; no valid checksum, never a usable key
    return "AGE-SECRET-" + "KEY-1" + "".join(secrets.choice(BECH32) for _ in range(58))


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "-q", path], capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    return r.returncode == 0


@pytest.mark.parametrize("path", ["deploy/secrets/staging.sops.yaml", "deploy/secrets/production.sops.yaml",
                                  "deploy/secrets/.sops.yaml", "deploy/secrets/README.md",
                                  "deploy/secrets/escrow-recipients.txt",
                                  "deploy/secrets/history/staging/20261019T000000Z.sops.yaml"])
def test_d07_encrypted_secrets_files_are_visible_to_checkpoints(path: str) -> None:
    assert not _ignored(path), f"{path} is gitignored, so checkpoint trees would drop it"


@pytest.mark.parametrize("path", ["deploy/secrets/staging.yaml", "deploy/secrets/staging.dec.yaml",
                                  "deploy/secrets/keys.txt", "deploy/secrets/staging.agekey",
                                  "deploy/secrets/history/staging/plain.yaml", "keys.txt",
                                  "ops/staging.agekey", "secrets/x.txt",
                                  "key.txt", "ops/key.txt"])  # fix1 R2: age-keygen's documented default name
def test_d07_age_keys_and_decrypted_files_stay_ignored(path: str) -> None:
    assert _ignored(path), f"{path} must stay ignored"


def test_d07_build_contexts_exclude_sops_files_and_age_keys() -> None:
    ignore = ROOT / ".dockerignore"
    for rel in ("deploy/secrets/staging.sops.yaml", "deploy/secrets/staging.yaml", "ops/staging.agekey", "keys.txt"):
        assert dockerfiles.dockerignored(rel, ignore), rel


# fix1 R2/R3: key.txt and *.key too, and every letter case. Docker matches case-sensitively, while git on the
# Windows host (core.ignorecase=true) hides case variants from the scan, so only .dockerignore can stop them.
@pytest.mark.parametrize("rel", ["key.txt", "services/x/key.txt", "services/x/age.key", "services/x/KEYS.TXT",
                                 "ops/staging.AgeKey", "services/x/Key.Txt", "ops/AGE.KEY",
                                 "Deploy/Secrets/staging.yaml"])
def test_d07_build_contexts_exclude_key_files_in_any_case(rel: str) -> None:
    assert dockerfiles.dockerignored(rel, ROOT / ".dockerignore"), rel


def test_d07_dockerignore_replica_honours_bracket_classes(tmp_path: Path) -> None:
    # fix1 R3: Docker copies a [...] class into its regex; the replica must agree with it
    ignore = tmp_path / ".dockerignore"
    ignore.write_text("**/*.[aA][gG][eE]\nlog[0-9].txt\n", encoding="utf-8")
    assert dockerfiles.dockerignored("a/b/x.AgE", ignore)
    assert dockerfiles.dockerignored("log7.txt", ignore)
    assert not dockerfiles.dockerignored("logx.txt", ignore)
    assert not dockerfiles.dockerignored("log[0-9].txt", ignore)
    assert not dockerfiles.dockerignored("a/b/x.agf", ignore)


def test_d07_scan_flags_age_keys_stray_files_and_plaintext(tmp_path: Path) -> None:
    (tmp_path / "deploy" / "secrets").mkdir(parents=True)
    (tmp_path / "notes.txt").write_text("key " + _synthetic_age_identity() + "\n", encoding="utf-8")
    (tmp_path / "deploy/secrets/staging.sops.yaml").write_text("AUTH_SECRET_KEY: SYNTHETIC-PLAINTEXT\n" + SOPS_META, encoding="utf-8")
    (tmp_path / "deploy/secrets/production.sops.yaml").write_text("AUTH_SECRET_KEY: " + ENC + "\n", encoding="utf-8")
    (tmp_path / "deploy/secrets/staging.yaml").write_text("AUTH_SECRET_KEY: SYNTHETIC-PLAINTEXT\n", encoding="utf-8")
    (tmp_path / "ops.agekey").write_text("synthetic\n", encoding="utf-8")
    (tmp_path / "key.txt").write_text("synthetic\n", encoding="utf-8")
    paths = ["notes.txt", "deploy/secrets/staging.sops.yaml", "deploy/secrets/production.sops.yaml",
             "deploy/secrets/staging.yaml", "ops.agekey", "key.txt"]
    findings = secret_scan.scan(tmp_path, paths)
    assert {(f.rule, f.path) for f in findings} == {
        ("age-private-key", "notes.txt"), ("sops-plaintext-value", "deploy/secrets/staging.sops.yaml"),
        ("sops-metadata-missing", "deploy/secrets/production.sops.yaml"),
        ("stray-file-in-deploy-secrets", "deploy/secrets/staging.yaml"), ("age-key-file", "ops.agekey"),
        ("age-key-file", "key.txt")}
    assert all("SYNTHETIC-PLAINTEXT" not in str(f) and "KEY-1" not in str(f) for f in findings)  # names only, never content


def _key_file_text(identity: str, newline: str = "\n") -> str:
    # laid out the way age-keygen writes a key file; synthetic characters only
    return newline.join(["# created: 2026-10-19T00:00:00Z", "# public key: age1" + "q" * 58, identity, ""])


_PQ = "AGE-SECRET-" + "KEY-PQ-1"


# fix1 R1: post-quantum identities (age >= 1.3), UTF-16 files (Windows PowerShell 5.1's `>` default) and lower case
_KEY_FILE_FORMS = {  # built inside the test, so no key-shaped string ever lands in a test id
    "pq.txt": lambda: _key_file_text(_PQ + "".join(secrets.choice(BECH32) for _ in range(58))).encode("utf-8"),
    "utf16le-bom.txt": lambda: b"\xff\xfe" + _key_file_text(_synthetic_age_identity(), "\r\n").encode("utf-16-le"),
    "utf16be.txt": lambda: _key_file_text(_synthetic_age_identity()).encode("utf-16-be"),
    "pq-utf16le.txt": lambda: b"\xff\xfe" + _key_file_text(_PQ + "Q" * 58, "\r\n").encode("utf-16-le"),
    "lower.txt": lambda: _key_file_text(_synthetic_age_identity().lower()).encode("utf-8"),
    "utf8-bom-crlf.txt": lambda: b"\xef\xbb\xbf" + _key_file_text(_synthetic_age_identity(), "\r\n").encode("utf-8"),
}


@pytest.mark.parametrize("name", sorted(_KEY_FILE_FORMS))
def test_d07_scan_flags_age_identities_in_any_form(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(_KEY_FILE_FORMS[name]())
    findings = secret_scan.scan(tmp_path, [name])
    assert [(f.rule, f.path) for f in findings] == [("age-private-key", name)]
    assert all("KEY-" not in str(f) for f in findings)


_REAL_SOPS_META = ("sops:\n    kms: []\n    gcp_kms: []\n    azure_kv: []\n    hc_vault: []\n    age:\n"
                   "        - recipient: age1" + "q" * 58 + "\n          enc: |\n"
                   "            -----BEGIN AGE ENCRYPTED FILE-----\n            c3ludGhldGlj\n"
                   "            -----END AGE ENCRYPTED FILE-----\n"
                   "    lastmodified: '2026-10-19T00:00:00Z'\n    mac: " + ENC + "\n    pgp: []\n"
                   "    unencrypted_suffix: _unencrypted\n    version: 3.9.0\n")


# fix1 R4: only the one top-level `sops` mapping is exempt, and only its known metadata children
_HIDDEN_PLAINTEXT = {
    "dupmeta": "sops:\n  DB_PASSWORD: SYNTHETIC-PLAINTEXT\nAUTH_SECRET_KEY: " + ENC + "\n" + SOPS_META,
    "childplain": "AUTH_SECRET_KEY: " + ENC + "\n" + SOPS_META + "  DB_PASSWORD: SYNTHETIC-PLAINTEXT\n",
    "unencregex": "AUTH_SECRET_KEY: " + ENC + "\n" + SOPS_META + "  unencrypted_regex: ^public_\n",
    "encregex": "AUTH_SECRET_KEY: " + ENC + "\n" + SOPS_META + "  encrypted_regex: ^secret_\n",
    "customsuffix": "AUTH_SECRET_KEY: " + ENC + "\n" + SOPS_META + "  unencrypted_suffix: _public\n",
}


@pytest.mark.parametrize("name", sorted(_HIDDEN_PLAINTEXT))
def test_d07_scan_flags_plaintext_hidden_in_or_beside_the_sops_block(tmp_path: Path, name: str) -> None:
    rel = f"deploy/secrets/{name}.sops.yaml"
    (tmp_path / "deploy" / "secrets").mkdir(parents=True)
    (tmp_path / rel).write_text(_HIDDEN_PLAINTEXT[name], encoding="utf-8")
    findings = secret_scan.scan(tmp_path, [rel])
    assert [(f.rule, f.path) for f in findings] == [("sops-plaintext-value", rel)]
    assert all("SYNTHETIC-PLAINTEXT" not in str(f) for f in findings)


def test_d07_scan_accepts_the_metadata_sops_actually_writes(tmp_path: Path) -> None:
    # sops 3.9 records empty key-type lists and its default `unencrypted_suffix: _unencrypted` in every file
    (tmp_path / "deploy" / "secrets").mkdir(parents=True)
    rel = "deploy/secrets/staging.sops.yaml"
    (tmp_path / rel).write_text("AUTH_SECRET_KEY: " + ENC + "\nN:\n    - " + ENC + "\n" + _REAL_SOPS_META, encoding="utf-8")
    assert secret_scan.scan(tmp_path, [rel]) == []


def test_d07_scan_accepts_the_planned_encrypted_and_public_files(tmp_path: Path) -> None:
    (tmp_path / "deploy" / "secrets" / "history" / "staging").mkdir(parents=True)
    body = "AUTH_SECRET_KEY: " + ENC + "\nnested:\n  X: " + ENC + "\n" + SOPS_META
    (tmp_path / "deploy/secrets/staging.sops.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "deploy/secrets/history/staging/20261019T000000Z.sops.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "deploy/secrets/escrow-recipients.txt").write_text("# public recipients only\nage1" + "q" * 58 + "\n", encoding="utf-8")
    paths = ["deploy/secrets/staging.sops.yaml", "deploy/secrets/history/staging/20261019T000000Z.sops.yaml",
             "deploy/secrets/escrow-recipients.txt"]
    assert secret_scan.scan(tmp_path, paths) == []


def test_d07_checkpoint_visible_tree_has_no_secret_findings() -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ci" / "secret_scan.py"), "--root", str(ROOT)],
                       capture_output=True, text=True, timeout=600)
    # fix1 R5: exit 2 (git unavailable, unreadable file) reports on stderr; names, counts and errors only
    assert r.returncode == 0, (r.returncode, r.stdout[-4000:], r.stderr[-2000:])
