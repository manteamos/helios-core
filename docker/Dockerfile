# syntax=docker/dockerfile:1
# ============================================================
# Helios Core — multi-stage, non-root, no GUI libs
#
# To pin base image for production (no floating tags):
#   docker pull python:3.11-slim
#   docker inspect python:3.11-slim --format '{{index .RepoDigests 0}}'
# Then replace the FROM lines below with:
#   FROM python:3.11-slim@sha256:<digest> AS builder
# ============================================================

# ----------------------------------------------------------
# Stage 1: builder — compile wheels for all dependencies
# ----------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /build

# Build essentials only — not copied into the runtime image
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (context = repo root, set in docker-compose.yml)
COPY pyproject.toml ./
COPY core/      core/
COPY registry/  registry/
COPY api/        api/

# Build helios-core + download all transitive dependency wheels.
# --wheel-dir collects everything; no network required in the runtime stage.
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# ----------------------------------------------------------
# Stage 2: runtime — lean, non-root, no build tools
# ----------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Non-root user — no shell, no home directory with write permissions
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install the application and all dependencies from the pre-built wheels.
# No network access required: --no-index --find-links /wheels.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels helios-core && \
    rm -rf /wheels

# Copy source tree so Python finds the packages at /app/core, /app/api, etc.
# (The installed wheel already provides them in site-packages, but having the
# tree here avoids PYTHONPATH juggling in dev and matches the editable install.)
COPY --from=builder /build/core/      core/
COPY --from=builder /build/registry/  registry/
COPY --from=builder /build/api/       api/

RUN chown -R appuser:appuser /app

USER appuser

# Liveness probe — uses only the stdlib (no curl/wget required in slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=4); \
         sys.exit(0 if r.status == 200 else 1)"

EXPOSE 8000

# Single uvicorn worker — horizontal scaling via replica count, not --workers.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
