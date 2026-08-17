# syntax=docker/dockerfile:1
# Solace-AI User Service - Production Dockerfile
# IMPORTANT: build context is the REPOSITORY ROOT:
#   docker build -f services/user-service/Dockerfile -t solace-user-service .
#
# user-service uses RELATIVE imports in main.py (from .config, ...) AND references both
# its OWN modules as `src.*` (from src.domain) and the SHARED packages as `src.*`
# (from src.solace_events) — a dual import-root (REV-31). Until those roots are
# normalized (D.1), the correct container layout is to physically MERGE the shared
# repo-root src/ and this service's src/ into a single /app/src and run as `src.main:app`.
# Shared packages are copied first; the service's own modules are layered on top.
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY requirements-runtime.txt ./requirements-runtime.txt
COPY services/user-service/requirements.txt ./service-requirements.txt
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -r requirements-runtime.txt -r service-requirements.txt \
    || pip install --no-cache-dir -r requirements-runtime.txt -r service-requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# 1) shared packages (solace_*) under /app/src ; 2) this service's src merged on top.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser services/__init__.py ./services/__init__.py
COPY --chown=appuser:appuser services/shared/ ./services/shared/
COPY --chown=appuser:appuser services/user-service/src/ ./src/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app:/app/src \
    SERVICE_NAME=user-service SERVICE_PORT=8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
USER appuser
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
