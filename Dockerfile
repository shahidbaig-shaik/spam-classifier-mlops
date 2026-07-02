# Multi-stage build: builder installs deps; runtime image omits build toolchain.

# ── Stage 1: builder ────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements before source so Docker can cache the pip layer across code-only rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Run as non-root to limit blast radius if the app is compromised
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ app/
COPY model/ model/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 0.0.0.0 required to accept connections from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
