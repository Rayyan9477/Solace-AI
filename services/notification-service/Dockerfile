# syntax=docker/dockerfile:1
# Solace-AI Notification Service - Production Dockerfile
# IMPORTANT: build context is the REPOSITORY ROOT:
#   docker build -f services/notification-service/Dockerfile -t solace-notification-service .
# Hyphenated dir -> runs as top-level `main:app` with its src on PYTHONPATH (see REV-31).
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY requirements-runtime.txt ./requirements-runtime.txt
COPY services/notification-service/requirements.txt ./service-requirements.txt
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
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser services/__init__.py ./services/__init__.py
COPY --chown=appuser:appuser services/shared/ ./services/shared/
COPY --chown=appuser:appuser services/notification-service/ ./services/notification-service/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src:/app/services/notification-service/src \
    SERVICE_NAME=notification-service SERVICE_PORT=8003
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
USER appuser
EXPOSE 8003
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
