# ──────────────────────────────────────────────────────────────────────
# Multi-stage production image for the AI Meeting Intelligence Platform.
# Stage 1: build wheels / install deps into a venv.
# Stage 2: slim runtime with ffmpeg + non-root user.
# ──────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app

# Create a virtualenv to copy into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `transcription` extra pulls torch/pyannote — install per-image via ARG.
ARG INSTALL_EXTRAS=ai
RUN pip install --upgrade pip setuptools wheel \
    && if [ "$INSTALL_EXTRAS" = "full" ]; then \
         pip install ".[full]"; \
       else \
         pip install ".[${INSTALL_EXTRAS}]"; \
       fi

# ── Stage 2: runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# ffmpeg is required for audio decode/validation; libpq5 for asyncpg.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libpq5 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home app

WORKDIR /opt/app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app ./app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic

# Store artifacts (audio, chunk cache) outside the code tree.
RUN mkdir -p /opt/app/artifacts /opt/app/audio_cache \
    && chown -R app:app /opt/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# tini reaps zombies from ffmpeg/subprocess calls.
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
