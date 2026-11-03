"""D-07 (SOPS + age): secret scan of every checkpoint-visible file. Stdlib plus PyYAML.

``checkpoint.ps1`` snapshots the working tree with ``git add -A``; ``visible_paths`` lists exactly that set
(``git ls-files --cached --others --exclude-standard``). Pre-commit hooks never run (D-17: no commits), so this
scan runs as a ``tests/build`` test in every regression, barrier and ``make ci-local``.

Rules. A finding names the rule and the path, never any file content:
    age-private-key               the file holds age's private-key prefix, X25519 or post-quantum (``PQ-``) form, in
                                  any letter case, followed by 50+ bech32 characters; searched in the raw bytes and,
                                  when the file has NUL bytes (UTF-16 or UTF-32, as Windows PowerShell 5.1's ``>``
                                  writes), again with the NULs dropped
    age-key-file                  a file named ``*.agekey``, ``keys.txt`` or ``key.txt`` (age-keygen's documented
                                  default), in any letter case
    stray-file-in-deploy-secrets  a file under ``deploy/secrets/`` that is not one of the planned files below
    sops-metadata-missing         a ``*.sops.yaml`` other than the creation-rules file named exactly ``.sops.yaml``
                                  that is not a YAML mapping, or has no top-level ``sops`` mapping holding a
                                  non-empty ``age`` list and a ``mac``
    sops-plaintext-value          in such a file: a scalar leaf outside the top-level ``sops`` mapping that is not a
                                  string starting with ``ENC[AES256_GCM,``; a second top-level ``sops`` key; or a
                                  child of the ``sops`` mapping that is not SOPS metadata (``_SOPS_METADATA_KEYS``,
                                  plus ``unencrypted_suffix`` only at SOPS's default ``_unencrypted``, which SOPS
                                  records in every file and whose keys this leaf rule already covers) or that
                                  repeats. The composed YAML node graph is walked, so a duplicate key cannot hide a
                                  plaintext value, and an ``unencrypted_regex``, ``encrypted_regex`` or custom
                                  suffix rule is reported. The metadata list is fail-closed: a key a newer pinned
                                  SOPS adds must be added here.

SOPS conventions (R-01.18; WS-03.8 follows them):
    deploy/secrets/ holds only
        <env>.sops.yaml                                  one per environment, encrypted to its age recipient(s)
        .sops.yaml                                       SOPS creation rules, public recipients only
        escrow-recipients.txt                            public recipients only
        README.md                                        optional
        history/<env>/<YYYYMMDDTHHMMSSZ>.sops.yaml       previous encrypted versions, named by UTC timestamp
    <env> is lower-case: a letter, then letters, digits, ``-`` or ``_``.
    Every <env>.sops.yaml is fully encrypted: no configured ``unencrypted_suffix``, ``unencrypted_regex``,
    ``encrypted_regex`` or ``encrypted_suffix`` rule (SOPS still records its default
    ``unencrypted_suffix: _unencrypted``), and no key ending in that default ``_unencrypted`` suffix.
    No ``*.agekey``, ``keys.txt`` or ``key.txt`` file is planned anywhere in the tree. Age private keys stay
    off-host and arrive at deploy time through ``SOPS_AGE_KEY_FILE`` at a root-only path; decrypted values exist
    only on tmpfs /run/secrets.

CLI:
    secret_scan.py [--root DIR]   one finding per line on stdout; exit 1 on any finding, 0 when clean,
                                  2 when the scan cannot run (git failure or an unreadable file)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[2]

# The prefix is never written contiguously, so this file, its tests and the plans never flag themselves.
_AGE_PRIVATE_KEY = re.compile(
    re.escape(("AGE-SECRET-" + "KEY-").encode()) + rb"(?:PQ-)?1[02-9AC-HJ-NP-Z]{50,}", re.IGNORECASE
)
_AGE_KEY_FILE_NAMES = frozenset({"keys.txt", "key.txt"})
_ENC_PREFIX = "ENC[AES256_GCM,"
_STR_TAG = "tag:yaml.org,2002:str"
_SOPS_METADATA_KEYS = frozenset({
    "age", "kms", "gcp_kms", "azure_kv", "hc_vault", "pgp", "key_groups", "shamir_threshold",
    "lastmodified", "mac", "version",
})
_SOPS_DEFAULT_UNENCRYPTED_SUFFIX = "_unencrypted"
_SECRETS_DIR = "deploy/secrets/"
_CREATION_RULES = ".sops.yaml"
_ENV = r"[a-z][a-z0-9_-]*"
_PLANNED_SECRETS_FILE = re.compile(
    rf"{_ENV}\.sops\.yaml|\.sops\.yaml|escrow-recipients\.txt|README\.md"
    rf"|history/{_ENV}/[0-9]{{8}}T[0-9]{{6}}Z\.sops\.yaml"
)


@dataclass(frozen=True)
class Finding:
    """One rule hit on one checkpoint-visible file; carries names only, never content."""

    rule: str
    path: str

    def __str__(self) -> str:
        return f"{self.rule} {self.path}"


def visible_paths(root: Path) -> list[str]:
    """Every path ``git add -A`` would capture: tracked plus untracked, minus ignored (read-only git)."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True,
    ).stdout
    return sorted({p for p in out.decode("utf-8", "surrogateescape").split("\0") if p})


def _has_sops_metadata(doc: dict) -> bool:
    meta = doc.get("sops")
    if not isinstance(meta, dict):
        return False
    age, mac = meta.get("age"), meta.get("mac")
    return isinstance(age, list) and bool(age) and isinstance(mac, str) and bool(mac)


def _str_key(node: yaml.Node) -> str | None:
    return node.value if isinstance(node, yaml.ScalarNode) and node.tag == _STR_TAG else None


def _metadata_leaks(meta: yaml.MappingNode) -> bool:
    """A child of the ``sops`` mapping that is not SOPS metadata, or a repeated one, may carry plaintext."""
    names = [_str_key(key) for key, _ in meta.value]
    if len(names) != len(set(names)):
        return True
    for (_, value), name in zip(meta.value, names, strict=True):
        if name in _SOPS_METADATA_KEYS:
            continue
        if (name == "unencrypted_suffix" and isinstance(value, yaml.ScalarNode)
                and value.value == _SOPS_DEFAULT_UNENCRYPTED_SUFFIX):
            continue
        return True
    return False


def _has_plaintext_leaf(root: yaml.MappingNode) -> bool:
    metadata = [value for key, value in root.value if _str_key(key) == "sops"]
    if len(metadata) > 1:  # the constructed document keeps only the last block, so an earlier one could hide plaintext
        return True
    stack: list[yaml.Node] = [value for key, value in root.value if _str_key(key) != "sops"]
    if metadata and isinstance(metadata[0], yaml.MappingNode):
        if _metadata_leaks(metadata[0]):
            return True
    else:
        stack.extend(metadata)  # a non-mapping `sops` value is checked like any other value
    seen: set[int] = set()  # an alias points at an already-visited node; also stops recursive anchors
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            stack.extend(value for _, value in node.value)
        elif isinstance(node, yaml.SequenceNode):
            stack.extend(node.value)
        elif not (node.tag == _STR_TAG and node.value.startswith(_ENC_PREFIX)):
            return True
    return False


def _sops_rules(data: bytes) -> Iterator[str]:
    try:
        loader = yaml.SafeLoader(data)  # the reader decodes the first bytes here, so this can raise too
        try:
            node = loader.get_single_node()
            doc = loader.construct_document(node) if node is not None else None
        finally:
            loader.dispose()
    except (yaml.YAMLError, ValueError, RecursionError):
        node = doc = None
    if not isinstance(node, yaml.MappingNode) or not isinstance(doc, dict):
        yield "sops-metadata-missing"
        return
    if not _has_sops_metadata(doc):
        yield "sops-metadata-missing"
    if _has_plaintext_leaf(node):
        yield "sops-plaintext-value"


def _rules(rel: str, data: bytes) -> Iterator[str]:
    name = PurePosixPath(rel).name
    lowered = name.lower()
    if _AGE_PRIVATE_KEY.search(data) or (b"\x00" in data and _AGE_PRIVATE_KEY.search(data.replace(b"\x00", b""))):
        yield "age-private-key"
    if lowered.endswith(".agekey") or lowered in _AGE_KEY_FILE_NAMES:
        yield "age-key-file"
    if rel.lower().startswith(_SECRETS_DIR) and not _PLANNED_SECRETS_FILE.fullmatch(rel[len(_SECRETS_DIR):]):
        yield "stray-file-in-deploy-secrets"
    if lowered.endswith(".sops.yaml") and name != _CREATION_RULES:
        yield from _sops_rules(data)


def scan(root: Path, paths: Iterable[str]) -> list[Finding]:
    """Apply every rule to each existing file (read as bytes, so binaries are searched too).

    Paths are repository-relative with ``/`` separators, as git prints them; a listed path that no longer exists
    (deleted in the working tree) or is not a regular file is skipped. An unreadable file raises ``OSError``.
    """
    root = Path(root)
    findings: list[Finding] = []
    for raw in paths:
        rel = raw.replace("\\", "/")
        file = root / rel
        if not file.is_file():
            continue
        findings.extend(Finding(rule, rel) for rule in _rules(rel, file.read_bytes()))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="repository to scan (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        paths = visible_paths(args.root)
        findings = scan(args.root, paths)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"secret_scan: cannot scan {args.root}: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    print(f"secret_scan: {len(paths)} checkpoint-visible paths, {len(findings)} findings", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
