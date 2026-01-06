# Islands - Codebase Indexing and Inquiry System
# Multi-stage build for optimized Rust binary container

# Build stage
FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Install build dependencies for git operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libgit2-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source files to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release 2>/dev/null || true

# Copy actual source code
COPY src/ src/
COPY benches/ benches/

# Touch source files to invalidate cache and rebuild with real code
RUN touch src/main.rs src/lib.rs

# Build the release binary
RUN cargo build --release --bin islands

# Runtime stage - minimal image
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="Islands"
LABEL org.opencontainers.image.description="Codebase Indexing and Inquiry System using LEANN"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/panbanda/islands"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    openssh-client \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 islands

# Copy binary from builder
COPY --from=builder /build/target/release/islands /usr/local/bin/islands

# Create data directories
RUN mkdir -p /data/islands/{repos,indexes,cache} \
    && chown -R islands:islands /data/islands

# Set default environment variables
ENV ISLANDS_STORAGE__BASE_PATH=/data/islands
ENV ISLANDS_STORAGE__REPOS_PATH=/data/islands/repos
ENV ISLANDS_STORAGE__INDEXES_PATH=/data/islands/indexes
ENV ISLANDS_STORAGE__CACHE_PATH=/data/islands/cache
ENV ISLANDS_LOG_LEVEL=INFO
ENV RUST_LOG=info

# Health check using the CLI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD islands --version || exit 1

USER islands

# Default command runs the MCP server
ENTRYPOINT ["islands"]
CMD ["mcp"]

# Expose MCP and webhook ports
EXPOSE 8080 9000
