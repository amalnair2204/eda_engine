# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# EDA Engine — production image (FastAPI + uvicorn)
#
# Secrets (GROQ_API_KEY) are NEVER baked in — they are read from the runtime
# environment.  Build layers are ordered so dependency installs cache across
# code changes: requirements.txt is copied and installed before the app source.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
# - PYTHONUNBUFFERED: stream logs straight to stdout (PaaS-friendly)
# - PIP_NO_CACHE_DIR: smaller image
# - HF_HOME: cache the sentence-transformers embedding model under /app so a
#   writable, predictable location is used at runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System libs some wheels expect at runtime (libgomp for torch/sklearn).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first — cached unless requirements.txt changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source (everything not excluded by .dockerignore).
COPY . .

# Documents the listen port; the actual value is read from $PORT at runtime.
EXPOSE 8000

# Shell form so ${HOST}/${PORT} env vars expand at container start. Mirrors the
# app.py __main__ entry-point; either works.
CMD ["sh", "-c", "uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
