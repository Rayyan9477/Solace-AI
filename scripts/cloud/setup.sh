#!/usr/bin/env bash
# Cloud environment setup script for Solace-AI (paste into the environment's "Setup script" field,
# or run from a SessionStart hook when CLAUDE_CODE_REMOTE=true). Runs as root on Ubuntu 24.04.
# Builds the program venv from the hash-pinned locks (the same set as the local venv-clean), so
# test runs match local results. Must exit 0 and should finish in under ~5 minutes to be cached.
set -euo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$REPO"

# Python 3.12 is the project's version; uv is preinstalled in cloud sessions. If uv cannot download
# 3.12 (network policy), fall back to a system python3 that is 3.12 or newer.
if [ ! -x .venv/bin/python ]; then
  if uv python install 3.12; then
    uv venv --python 3.12 .venv
  else
    echo "uv could not install Python 3.12; using the system python3" >&2
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
      || { echo "system python3 is older than 3.12" >&2; exit 1; }
    uv venv --python "$(command -v python3)" .venv
  fi
fi
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python -r requirements-dev.lock

# PostgreSQL 16 and Redis 7 are preinstalled but not running; the real-infra tests start their own
# containers through Docker, so nothing is started here.
.venv/bin/python -c "import fastapi, pydantic, structlog, asyncpg, aiokafka; print('venv ok')"
