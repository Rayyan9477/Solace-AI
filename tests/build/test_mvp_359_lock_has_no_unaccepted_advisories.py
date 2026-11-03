"""MVP-359/005: every advisory in either lock is fixed or accepted with an expiry, and no lock names a local path."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pytest
import yaml
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
LOCK, DEV_LOCK = ROOT / "requirements.lock", ROOT / "requirements-dev.lock"
ACCEPTED = ROOT / "scripts" / "ci" / "accepted_vulns.yaml"
MAX_ACCEPT_DAYS = 90
LOCAL_PATHS = ("D:/", "D:\\", "C:/", "C:\\", "/tmp/")
ADVISORY_DB = "https://pypi.org/pypi/pip/json"  # pip-audit's default service is the PyPI JSON API


def _require_advisory_db() -> None:
    """Skip when the advisory database is unreachable; fail instead when SOLACE_REQUIRE_REALINFRA=1."""
    try:
        with urllib.request.urlopen(ADVISORY_DB, timeout=15):
            return
    except OSError as exc:
        reason = f"advisory database unreachable: {type(exc).__name__}: {exc}"
        if os.environ.get("SOLACE_REQUIRE_REALINFRA") == "1":
            pytest.fail(f"SOLACE_REQUIRE_REALINFRA=1 but {reason}", pytrace=False)
        pytest.skip(reason)


def _audit(lock: Path) -> list[tuple[str, str, str, frozenset[str]]]:
    """(package, version, advisory id, aliases) for every finding pip-audit reports against the lock."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "-r", str(lock), "--require-hashes", "--disable-pip", "--no-deps",
         "-f", "json", "--progress-spinner", "off"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    # pip-audit exits 1 when it reports vulnerabilities; any other non-zero code is a tool failure
    assert proc.returncode in (0, 1), f"pip-audit exited {proc.returncode}: {proc.stderr[-3000:]}"
    deps = json.loads(proc.stdout)["dependencies"]
    skipped = {d["name"]: d["skip_reason"] for d in deps if "skip_reason" in d}
    assert skipped == {}, f"pip-audit could not audit these pins: {skipped}"
    return [(canonicalize_name(d["name"]), d["version"], v["id"], frozenset(v.get("aliases", [])))
            for d in deps for v in d.get("vulns", [])]


def _acceptances() -> dict[tuple[str, str], dict]:
    """(target, advisory id) -> entry; empty when the file is missing (the caller reports that separately)."""
    if not ACCEPTED.is_file():
        return {}
    entries = (yaml.safe_load(ACCEPTED.read_text(encoding="utf-8")) or {}).get("accepted") or []
    return {(canonicalize_name(str(e["target"])), str(e["id"])): e for e in entries}


def _expiry(entry: dict) -> date:
    value = entry["expires"]
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _unaccepted(lock: Path) -> list[str]:
    findings, accepted, today = _audit(lock), _acceptances(), date.today()
    problems = [] if ACCEPTED.is_file() else [f"{ACCEPTED.relative_to(ROOT).as_posix()} is missing"]
    for pkg, version, vid, aliases in sorted(findings):
        entry = next((accepted[(pkg, i)] for i in (vid, *sorted(aliases)) if (pkg, i) in accepted), None)
        if entry is None:
            problems.append(f"{pkg}=={version} {vid}: neither fixed nor accepted")
            continue
        days = (_expiry(entry) - today).days
        if not 0 < days <= MAX_ACCEPT_DAYS:
            problems.append(f"{pkg}=={version} {vid}: acceptance expires {entry['expires']} "
                            f"({days} days away; must be 1 to {MAX_ACCEPT_DAYS})")
        if not str(entry.get("reason") or "").strip() or not str(entry.get("owner") or "").strip():
            problems.append(f"{pkg}=={version} {vid}: acceptance needs a reason and an owner")
    return problems


@pytest.mark.realinfra
def test_mvp_359_runtime_lock_advisories_are_fixed_or_accepted() -> None:
    _require_advisory_db()
    problems = _unaccepted(LOCK)
    assert problems == [], f"{len(problems)} problem(s) in requirements.lock:\n" + "\n".join(problems)


@pytest.mark.realinfra
def test_mvp_359_dev_lock_advisories_are_fixed_or_accepted() -> None:
    _require_advisory_db()
    problems = _unaccepted(DEV_LOCK)
    assert problems == [], f"{len(problems)} problem(s) in requirements-dev.lock:\n" + "\n".join(problems)


def test_mvp_005_locks_name_no_local_paths() -> None:
    hits = {}
    for lock in (LOCK, DEV_LOCK):
        lines = [n for n, line in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1)
                 if any(p in line for p in LOCAL_PATHS)]
        if lines:
            hits[lock.name] = (len(lines), lines[:5])
    assert hits == {}, f"lock lines naming a local path (count, first line numbers): {hits}"
