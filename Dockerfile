# ==============================================================================
# SubNeutralize v0.2.0: Official Production GPU Inference Gateway Dockerfile
# Author: Arnab Dutta | Apache-2.0 License
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration and source
COPY pyproject.toml README.md LICENSE ./
COPY subneutralize/ subneutralize/

# Build standalone wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build \
    && python -m build --wheel --outdir /app/dist

# ==============================================================================
# Production Runtime Stage
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SUBNEUTRALIZE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    PORT=8000 \
    HOST=0.0.0.0

# Install runtime dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install built wheel with serve dependencies
COPY --from=builder /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl[serve] \
    && rm -f /app/*.whl

# Expose OpenAI-compatible API port
EXPOSE 8000

# Container healthcheck against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch SubNeutralize gateway
ENTRYPOINT ["subneutralize", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
