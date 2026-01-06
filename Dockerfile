# Islands - Codebase Indexing and Inquiry System
# Multi-stage build for optimized Rust binary container

# Build stage
FROM rust:1.92-bookworm AS builder

WORKDIR /build

# Install build dependencies for git operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libgit2-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/islands-core/Cargo.toml crates/islands-core/
COPY crates/islands-providers/Cargo.toml crates/islands-providers/
COPY crates/islands-indexer/Cargo.toml crates/islands-indexer/
COPY crates/islands-mcp/Cargo.toml crates/islands-mcp/
COPY crates/islands-cli/Cargo.toml crates/islands-cli/
COPY crates/islands-agent/Cargo.toml crates/islands-agent/

# Create dummy source files to build dependencies
RUN mkdir -p crates/islands-core/src \
    && echo "pub fn lib() {}" > crates/islands-core/src/lib.rs \
    && mkdir -p crates/islands-providers/src \
    && echo "pub fn lib() {}" > crates/islands-providers/src/lib.rs \
    && mkdir -p crates/islands-indexer/src \
    && echo "pub fn lib() {}" > crates/islands-indexer/src/lib.rs \
    && mkdir -p crates/islands-mcp/src \
    && echo "pub fn lib() {}" > crates/islands-mcp/src/lib.rs \
    && mkdir -p crates/islands-cli/src \
    && echo "fn main() {}" > crates/islands-cli/src/main.rs \
    && mkdir -p crates/islands-agent/src \
    && echo "pub fn lib() {}" > crates/islands-agent/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --workspace 2>/dev/null || true

# Copy actual source code
COPY crates/ crates/

# Touch source files to invalidate cache and rebuild with real code
RUN find crates -name "*.rs" -exec touch {} \;

# Build the release binary
RUN cargo build --release --bin islands

# Runtime stage - minimal image
FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="Islands"
LABEL org.opencontainers.image.description="Codebase Indexing and Inquiry System using LEANN"
LABEL org.opencontainers.image.version="0.3.0"
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
CMD ["mcp", "serve"]

# Expose MCP and webhook ports
EXPOSE 8080 9000
