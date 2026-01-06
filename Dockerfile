# Islands - Codebase Indexing and Inquiry System
# Multi-stage build with cargo-chef for optimal caching

# Chef stage - prepare recipe
FROM rust:1.92-bookworm AS chef
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo install cargo-chef
WORKDIR /build

# Planner stage - analyze dependencies
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
RUN cargo chef prepare --recipe-path recipe.json

# Builder stage - build dependencies then app
FROM chef AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libgit2-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies (cached unless Cargo.toml/lock changes)
COPY --from=planner /build/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo chef cook --release --recipe-path recipe.json

# Build application
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY benches/ benches/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin islands && \
    cp target/release/islands /usr/local/bin/islands

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
COPY --from=builder /usr/local/bin/islands /usr/local/bin/islands

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD islands --version || exit 1

USER islands

ENTRYPOINT ["islands"]
CMD ["mcp"]

EXPOSE 8080 9000
