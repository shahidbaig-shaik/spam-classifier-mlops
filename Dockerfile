# =============================================================
#  DOCKERFILE — Packages the Spam Classifier API into a Container
# =============================================================
#
# WHAT IS A DOCKERFILE?
# ---------------------
# A Dockerfile is a recipe for building a Docker "image".
# Think of it like a cooking recipe:
#   - Start with a base ingredient (Python)
#   - Add your dependencies (pip install)
#   - Add your code (COPY)
#   - Tell it how to run (CMD)
#
# Once built, the image can run ANYWHERE Docker is installed —
# your Mac, a Linux server, AWS, Render, etc. Same behavior guaranteed.
#
# WHY MULTI-STAGE BUILD?
# ----------------------
# We use TWO stages:
#   Stage 1 (builder): Install all dependencies (this stage has pip, gcc, etc.)
#   Stage 2 (runtime): Copy ONLY the installed packages + our code
#
# Result: The final image is much smaller because it doesn't include
# build tools we don't need at runtime.
#
# ANALOGY: You need a kitchen to cook a meal, but you only bring
# the finished dish to the dining table — not the entire kitchen.


# ============================================================
#  STAGE 1: BUILDER — Install dependencies
# ============================================================
FROM python:3.12-slim AS builder

# These environment variables prevent Python from creating .pyc files
# and ensure output isn't buffered (so you see logs in real-time)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements FIRST (before the rest of the code)
# WHY? Docker caches each step as a "layer". If requirements.txt
# hasn't changed, Docker reuses the cached layer and skips pip install.
# This makes rebuilds MUCH faster when you only change code.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
#  STAGE 2: RUNTIME — Lean final image
# ============================================================
FROM python:3.12-slim AS runtime

# Create a non-root user for security
# WHY? If an attacker exploits a vulnerability, they only get
# limited permissions — not root access to the container.
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy installed Python packages from the builder stage
# This is the key part of multi-stage: we only take what we need
COPY --from=builder /install /usr/local

# Copy our application code and model
COPY app/ app/
COPY model/ model/

# Set ownership to our non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Tell Docker this container listens on port 8000
# (This is documentation — it doesn't actually open the port)
EXPOSE 8000

# The command to run when the container starts
# --host 0.0.0.0: Listen on all interfaces (required in Docker)
# --port 8000: Match the EXPOSE above
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
