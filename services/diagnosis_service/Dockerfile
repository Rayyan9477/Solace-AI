# Solace-AI Diagnosis Service - Production Dockerfile
FROM python:3.12-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY requirements*.txt ./
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN if [ -f requirements.txt ]; then uv pip install --no-cache -r requirements.txt 2>/dev/null || pip install --no-cache-dir -r requirements.txt; fi

FROM python:3.12-slim as runtime
WORKDIR /app
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --chown=appuser:appuser src/ ./src/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app SERVICE_NAME=diagnosis-service SERVICE_PORT=8004
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
USER appuser
EXPOSE ${SERVICE_PORT}
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8004"]
