"""Run every test suite in its own pytest process and print one PASS/FAIL/MISSING line per suite."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.suites import REALINFRA_PATHS, SUITES  # noqa: E402

TIERS = ("unit", "realinfra", "e2e", "clinical", "contract")
_BASE = ["-o", "addopts=", "-p", "no:logging", "-q"]


def build_command(path: str, tier: str) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", path, *_BASE]
    if tier == "unit":
        cmd += ["-m", "not realinfra and not e2e"]
    elif tier == "realinfra":
        cmd += ["-m", "realinfra"]
    elif tier == "e2e":
        cmd += ["-m", "e2e"]
    elif tier == "clinical":
        cmd += ["-m", "clinical"]
    elif tier == "contract":
        cmd += ["-m", "contract"]
    else:
        raise ValueError(f"unknown tier: {tier}")
    return cmd


def child_env(tier: str, *, profile: str | None, require_realinfra: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["ENVIRONMENT"] = "test"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{ROOT / 'src'}"
    if tier == "e2e" and profile:
        env["SOLACE_E2E_PROFILE"] = profile
    if require_realinfra:
        env["SOLACE_REQUIRE_REALINFRA"] = "1"
    return env


def tier_paths(tier: str) -> list[str]:
    if tier == "unit":
        return [s.path for s in SUITES]
    if tier == "realinfra":
        return list(REALINFRA_PATHS)
    if tier == "e2e":
        return ["tests/e2e"]
    if tier == "clinical":
        return ["tests/clinical"]
    return ["tests/contracts"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=TIERS, default="unit")
    ap.add_argument("--suite", action="append", default=[], help="select a listed suite by path")
    ap.add_argument("--path", action="append", default=[], help="run an arbitrary directory as a suite")
    ap.add_argument("--profile", choices=("dev", "strict"), default=None)
    ap.add_argument("--require-realinfra", action="store_true")
    ap.add_argument("--strict-paths", action="store_true")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--list-json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    paths = tier_paths(args.tier)
    if args.suite:
        paths = list(args.suite)
    if args.path:
        paths = (paths if args.suite else []) + list(args.path)

    if args.list_json:
        print(json.dumps(paths))
        return 0

    env = child_env(args.tier, profile=args.profile, require_realinfra=args.require_realinfra)
    cwd = str(Path(args.cwd).resolve()) if args.cwd else str(ROOT)
    results: list[tuple[str, str]] = []
    for p in paths:
        target = Path(p) if Path(p).is_absolute() else ROOT / p
        if not target.exists():
            results.append(("MISSING", p))
            continue
        run_path = str(target) if args.cwd else p
        cmd = build_command(run_path, args.tier)
        if not target.resolve().is_relative_to(ROOT):
            # a suite outside the repo must not let pytest derive its rootdir from the drive root
            cmd += ["--rootdir", str(target.resolve())]
        if args.dry_run:
            print(" ".join(cmd))
            results.append(("DRY", p))
            continue
        rc = subprocess.run(cmd, env=env, cwd=cwd).returncode
        results.append(("PASS" if rc in (0, 5) else "FAIL", p))

    print()
    for status, p in results:
        print(f"{status} {p}")
    failed = any(s == "FAIL" for s, _ in results)
    missing = any(s == "MISSING" for s, _ in results)
    return 1 if failed or (missing and args.strict_paths) else 0


if __name__ == "__main__":
    sys.exit(main())
