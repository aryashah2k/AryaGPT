# ── Stage 1: builder — install all Python deps into a venv ──────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to build wheels (chromadb, bcrypt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime — lean final image ─────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Health-check so orchestrators know the app is live
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
