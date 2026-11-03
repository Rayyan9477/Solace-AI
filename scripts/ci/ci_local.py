"""Run every CI stage locally in order, continue after failures, print a summary table."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RUN = [PY, str(ROOT / "scripts" / "ci" / "run_suites.py")]

STAGES: tuple[tuple[str, list[str]], ...] = (
    ("lint", [PY, "-m", "ruff", "check", ".", "--select", "E9,F63,F7,F82", "--no-fix", "--no-cache"]),
    ("unit", [*RUN, "--tier", "unit"]),
    ("contract", [*RUN, "--tier", "contract"]),
    ("clinical", [*RUN, "--tier", "clinical"]),
    ("images", [PY, "-m", "scripts.build.images"]),
    # non-gating (D-17): --report always exits 0; the pairs are input for the owner's post-program commit decision
    ("tracked-closure", [PY, str(ROOT / "scripts" / "ci" / "check_tracked_closure.py"), "--report"]),
    ("realinfra", [*RUN, "--tier", "realinfra", "--require-realinfra"]),
    ("e2e dev", [*RUN, "--tier", "e2e", "--profile", "dev"]),
    ("e2e strict", [*RUN, "--tier", "e2e", "--profile", "strict"]),
)


def main() -> int:
    rows: list[tuple[str, str]] = []
    for name, cmd in STAGES:
        print(f"=== {name} ===", flush=True)
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        except OSError as exc:
            print(f"cannot start stage {name}: {exc}")
            rc = 127
        rows.append((name, "PASS" if rc == 0 else f"FAIL ({rc})"))
    print("\nStage        Result")
    for name, res in rows:
        print(f"{name:<12} {res}")
    return 0 if all(r == "PASS" for _, r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
