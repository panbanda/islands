# Islands - Codebase Indexing and Inquiry System
# Multi-stage build for optimized container size

FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache-dir .

# Production image
FROM python:3.12-slim

LABEL maintainer="Islands Contributors"
LABEL description="Codebase Indexing and Inquiry System using LEANN"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 islands

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create data directories
RUN mkdir -p /data/islands/{repos,indexes,cache} \
    && chown -R islands:islands /data/islands

# Copy source code
COPY --chown=islands:islands src/ ./src/

# Set default environment variables
ENV ISLANDS_STORAGE__BASE_PATH=/data/islands
ENV ISLANDS_STORAGE__REPOS_PATH=/data/islands/repos
ENV ISLANDS_STORAGE__INDEXES_PATH=/data/islands/indexes
ENV ISLANDS_STORAGE__CACHE_PATH=/data/islands/cache
ENV ISLANDS_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import islands; print('healthy')" || exit 1

USER islands

# Default command runs the MCP server
CMD ["islands-mcp"]

# Expose MCP and webhook ports
EXPOSE 8080 9000
