# syntax=docker/dockerfile:1.7
# Build context: repository root.  docker build -f services/analytics-service/Dockerfile .
# The hyphenated service dir is not an importable package, so it runs as top-level main:app (REV-31).
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8.2@sha256:a7999d42cba0e5af47ef3c06ac310229c7f29c5314e35902f8353e8e170eeed1 /uv /usr/local/bin/uv
WORKDIR /build
COPY requirements.lock ./requirements.lock
RUN uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes requirements.lock
COPY docker/entrypoint.sh /build/entrypoint.sh
RUN sed -i 's/\r$//' /build/entrypoint.sh && chmod 0755 /build/entrypoint.sh

FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30 AS runtime
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser \
 && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/entrypoint.sh /app/docker/entrypoint.sh
WORKDIR /app
COPY src/ ./src/
COPY services/__init__.py ./services/__init__.py
COPY services/shared/ ./services/shared/
COPY infrastructure/__init__.py ./infrastructure/__init__.py
COPY infrastructure/api_gateway/ ./infrastructure/api_gateway/
COPY services/analytics-service/ ./services/analytics-service/
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src:/app/services/analytics-service/src \
    SERVICE_NAME=analytics-service SERVICE_PORT=8009
USER appuser
EXPOSE 8009
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS "http://127.0.0.1:${SERVICE_PORT}/live" || exit 1
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009", "--no-access-log", "--log-level", "warning", "--proxy-headers"]
