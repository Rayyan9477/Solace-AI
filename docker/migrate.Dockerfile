# syntax=docker/dockerfile:1.7
# Build context: repository root.  docker build -f docker/migrate.Dockerfile -t solace/migrate:<tag> .
# One-shot schema job: `python -m scripts.db.migrate upgrade head` as the migrator role (IF-03). It only packages
# alembic.ini, migrations/, scripts/db/ and src/; it carries no service code and runs no HTTP server.
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
 && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/entrypoint.sh /app/docker/entrypoint.sh
WORKDIR /app
COPY alembic.ini ./alembic.ini
COPY migrations/ ./migrations/
COPY scripts/db/ ./scripts/db/
COPY src/ ./src/
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app:/app/src
USER appuser
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "-m", "scripts.db.migrate", "upgrade", "head"]
