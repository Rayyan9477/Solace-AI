"""Render ``.env.example`` from the IF-02 registry (``deploy/env/env-contract.yaml``).

    python -m deploy.env.render_env_example --write   # rewrite .env.example
    python -m deploy.env.render_env_example --check   # exit 1 when .env.example is stale

One entry per source (env-file key), grouped under ``# --- <group> ---`` headings in registry
order, each preceded by one comment line with the secret flag, the required-in-strict flag,
the format and the notes. Secret sources are written blank, except a ``bcrypt`` secret, written
as ``''`` so the operator keeps the single quotes its $ signs need (MVP-117). Non-secret sources show
their dev default (else their example). A value containing ``$`` or whitespace is single-quoted.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from deploy.env.registry import PROFILES, REGISTRY_PATH, EnvVar, load_registry, sources

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"

HEADER = (
    "# Solace-AI deployment environment (IF-02).",
    "# GENERATED from deploy/env/env-contract.yaml by: python -m deploy.env.render_env_example --write",
    "# Do not edit by hand. Change the registry and re-render; tests/deploy/test_env_contract.py checks it.",
    "#",
    "# Secrets are blank here (a bcrypt hash shows '' to keep its quotes); never commit a real value. Single-quote any value that contains",
    "# a $ (a bcrypt hash, for example) so compose does not interpolate it. Unset values fall back to",
    "# the dev defaults shown; the strict profile (docker-compose.prod.yml) refuses to render while a",
    "# source marked 'required in strict' is unset.",
)

_NEEDS_QUOTES = set("$ \t#\"\\`")


def _primary(source: str, rows: list[EnvVar]) -> EnvVar:
    """The row that names the source's heading and notes."""
    for row in rows:
        if row.name == source:
            return row
    for row in rows:
        if row.group != "legacy":
            return row
    return rows[0]


def _value(primary: EnvVar, rows: list[EnvVar]) -> str:
    if primary.secret:
        # a bcrypt hash contains $: keep the quotes the operator must use (MVP-117)
        return "''" if primary.format == "bcrypt" else ""
    candidates = [primary, *rows]
    value = next((r.dev_default for r in candidates if r.dev_default is not None), None)
    if value is None:
        value = next((r.example for r in candidates if r.example is not None), "")
    if "'" in value or "\n" in value:
        raise ValueError(f"{primary.source}: a value with a quote or line break cannot be rendered")
    return f"'{value}'" if any(ch in _NEEDS_QUOTES for ch in value) else value


def _comment(primary: EnvVar, rows: list[EnvVar]) -> str:
    required = any(r.required_in_strict for r in rows)
    profiles = [p for p in PROFILES if any(p in r.profiles for r in rows)]
    parts = [
        f"secret: {'yes' if primary.secret else 'no'}",
        f"required in strict: {'yes' if required else 'no'}",
        f"format: {primary.format}",
    ]
    if profiles != list(PROFILES):
        parts.append(f"profiles: {', '.join(profiles)}")
    notes = primary.notes or next((r.notes for r in rows if r.notes), "")
    if notes:
        parts.append(" ".join(notes.split()))
    return "# " + " | ".join(parts)


def render_env_example(registry: Iterable[EnvVar]) -> str:
    """The full ``.env.example`` text for ``registry``."""
    groups: dict[str, list[tuple[str, EnvVar, list[EnvVar]]]] = {}
    for source, rows in sources(list(registry)).items():
        primary = _primary(source, rows)
        groups.setdefault(primary.group, []).append((source, primary, rows))
    lines = list(HEADER)
    for group, entries in groups.items():
        lines += ["", f"# --- {group} ---"]
        for source, primary, rows in entries:
            lines.append(_comment(primary, rows))
            lines.append(f"{source}={_value(primary, rows)}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deploy.env.render_env_example", description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the output file")
    mode.add_argument("--check", action="store_true", help="exit 1 if the output file is stale")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=ENV_EXAMPLE_PATH)
    args = parser.parse_args(argv)

    text = render_env_example(load_registry(args.registry))
    if args.write:
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
        return 0
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if current != text:
            print(f"{args.output} is stale; run: python -m deploy.env.render_env_example --write", file=sys.stderr)
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
