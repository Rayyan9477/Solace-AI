# Solace-AI Safety Service - Production Dockerfile
# Multi-stage build for optimized container size

# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./
COPY requirements*.txt ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -r requirements.txt 2>/dev/null || pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.12-slim as runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 safety && \
    useradd --uid 1000 --gid safety --shell /bin/bash --create-home safety

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=safety:safety src/ ./src/
COPY --chown=safety:safety config/ ./config/
COPY --chown=safety:safety pyproject.toml ./

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SERVICE_NAME=safety-service \
    SERVICE_PORT=8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Switch to non-root user
USER safety

# Expose service port
EXPOSE ${SERVICE_PORT}

# Run the service
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
