# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Build
cargo build --release

# Run all tests
cargo test

# Run a single test
cargo test test_name

# Run tests with logging
RUST_LOG=debug cargo test

# Lint
cargo clippy --all-targets -- -D warnings

# Format
cargo fmt

# Build documentation
cargo doc --no-deps

# Run benchmarks
cargo bench
```

## Architecture

Islands is a single Rust crate implementing LEANN (arXiv:2506.08276) for low-storage vector search. The key insight is storing only the proximity graph structure (CSR format) and recomputing embeddings on-demand during search, achieving ~95% storage reduction.

### Module Structure

```
islands (binary + library)
├── mcp (MCP server)
├── agent (OpenAI integration)
├── indexer
│   └── providers (GitHub/GitLab/Bitbucket/Gitea)
└── core (LEANN/HNSW/PQ)
```

### Core Algorithms (core module)

- **leann.rs**: `LeannIndex` with `CsrGraph` for graph-only storage, `EmbeddingProvider` trait for on-demand recomputation
- **hnsw.rs**: Traditional HNSW implementation for comparison
- **pq.rs**: Product Quantization for vector compression
- **distance.rs**: Cosine, Euclidean, dot product metrics
- **search.rs**: Two-level search with approximate/exact distance queues

### LEANN Parameters (from paper Section 5)

- $M = 30$ connections per node
- $\text{efConstruction} = 128$
- High-degree preserving pruning: hub nodes (top 2%) retain more connections

### Provider Pattern (providers module)

All git providers implement the same async trait pattern in `base.rs`. Add new providers by implementing the trait and registering in `factory.rs`.

## Docker and Kubernetes Deployment

### Building Docker Image

```bash
# Build locally
docker build -t islands:latest .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t islands:latest .
```

### Helm Chart Deployment

The Helm chart is located in `charts/islands/` and published to the GitHub Container Registry.

```bash
# Add the OCI registry (GitHub Container Registry)
helm registry login ghcr.io -u <github-username>

# Install from OCI registry
helm install islands oci://ghcr.io/panbanda/charts/islands \
  --namespace islands \
  --create-namespace \
  --set secrets.openaiApiKey="your-key" \
  --set secrets.github.token="your-token"

# Install from local chart
helm install islands ./charts/islands \
  --namespace islands \
  --create-namespace \
  -f values-override.yaml

# Upgrade existing release
helm upgrade islands oci://ghcr.io/panbanda/charts/islands \
  --namespace islands

# Uninstall
helm uninstall islands --namespace islands
```

### Key Helm Values

```yaml
# Image configuration
image:
  repository: ghcr.io/panbanda/islands
  tag: "0.3.0"

# Persistence (EFS for AWS)
persistence:
  enabled: true
  storageClass: "efs-sc"
  size: 100Gi

# Secrets
secrets:
  openaiApiKey: ""
  github:
    token: ""
    webhookSecret: ""

# Ingress
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: islands.example.com
      paths:
        - path: /
          pathType: Prefix
          port: mcp
```

### CI/CD Workflows

- **Docker**: `.github/workflows/docker.yaml` - Builds and pushes images on main/tags
- **Helm**: `.github/workflows/helm.yaml` - Lints, tests, and publishes chart to OCI registry

## Conventional Commits

This project uses release-please. Use conventional commit prefixes:
- `feat:` - new feature (minor version bump)
- `fix:` - bug fix (patch version bump)
- `docs:` - documentation only
- `chore:` - maintenance tasks
