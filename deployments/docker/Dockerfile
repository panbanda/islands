# syntax=docker/dockerfile:1

# Build stage
FROM golang:1.23-alpine AS builder

# Build arguments
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /build

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w \
        -X github.com/jon/pythia/cmd/pythia.Version=${VERSION} \
        -X github.com/jon/pythia/cmd/pythia.Commit=${COMMIT} \
        -X github.com/jon/pythia/cmd/pythia.BuildDate=${BUILD_DATE}" \
    -o /pythia .

# Security scan the binary
FROM aquasec/trivy:latest AS scanner
COPY --from=builder /pythia /pythia
RUN trivy rootfs --no-progress --severity HIGH,CRITICAL /pythia

# Runtime stage - using distroless for minimal attack surface
FROM gcr.io/distroless/static-debian12:nonroot

# Labels for OCI compliance
LABEL org.opencontainers.image.title="Pythia"
LABEL org.opencontainers.image.description="AI-powered codebase analysis tool with MCP integration"
LABEL org.opencontainers.image.vendor="Jon"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/jon/pythia"

# Copy binary from builder
COPY --from=builder /pythia /pythia

# Copy timezone data and CA certificates
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Create data directory (owned by nonroot user 65532)
WORKDIR /data

# Use non-root user (provided by distroless:nonroot)
USER nonroot:nonroot

# Expose MCP HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/pythia", "version"]

# Default command
ENTRYPOINT ["/pythia"]
CMD ["serve"]
