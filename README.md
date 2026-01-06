# Islands

[![CI](https://github.com/panbanda/islands/actions/workflows/rust.yaml/badge.svg)](https://github.com/panbanda/islands/actions/workflows/rust.yaml)
[![Crates.io](https://img.shields.io/crates/v/islands-cli.svg)](https://crates.io/crates/islands-cli)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Islands** is a high-performance codebase indexing and semantic search system written in Rust, built on the LEANN algorithm. It provides an MCP (Model Context Protocol) interface for AI assistants and supports multi-provider git integration.

## LEANN: Low-Storage Vector Search

Islands implements the [LEANN algorithm](https://arxiv.org/abs/2506.08276) (arXiv:2506.08276) for efficient vector similarity search:

> "LEANN reduces storage requirements to approximately 5% of original data size" by storing only the graph structure and recomputing embeddings on-the-fly during search.

**Key features from the paper:**

- **Graph-only storage**: Stores proximity graph in CSR format, not embeddings. Storage scales with edge count O(n*M), not embedding dimension O(n*d).
- **Selective recomputation**: Embeddings computed on-demand only for nodes in the search path.
- **High-degree preserving pruning**: Hub nodes (top 2% by degree) retain more connections to maintain graph navigability.
- **Parameters**: M=30 connections per node, efConstruction=128 (from Section 5 of paper).

For typical embedding dimensions (d=768-4096) with M=30, this yields ~25x storage reduction.

## Features

- **LEANN Core** (Rust): High-performance vector indexing with ~95% storage savings
- **Multi-Provider Support**: GitHub, GitLab, Bitbucket, and Gitea integration
- **MCP Interface**: Integrates with Claude Code and other MCP-compatible AI assistants
- **OpenAI Agent**: Interactive Q&A using OpenAI's SDK
- **Kubernetes Native**: Designed for EFS storage with automatic sync and webhooks

## Quick Start

### Installation

```bash
# Install via Homebrew (macOS/Linux)
brew tap panbanda/islands
brew install islands

# Or install via Cargo
cargo install islands-cli

# Or build from source
git clone https://github.com/panbanda/islands.git
cd islands
cargo build --release
```

### Basic Usage

```bash
# Add a repository by URL
islands add https://github.com/owner/repo --token $GITHUB_TOKEN

# Supports various URL formats
islands add https://gitlab.com/owner/repo
islands add git@github.com:owner/repo.git

# Search across indexed codebases
islands search "authentication middleware"

# List all indexes
islands list

# Start the MCP server
islands serve

# Interactive Q&A session
islands ask
```

### Docker

```bash
# Build the image
docker build -t islands -f docker/Dockerfile .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -k k8s/

# Or with kustomize
kustomize build k8s/ | kubectl apply -f -
```

## Configuration

Islands is configured through environment variables or a config file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ISLANDS_DEBUG` | Enable debug mode | `false` |
| `ISLANDS_LOG_LEVEL` | Log level | `INFO` |
| `ISLANDS_OPENAI_API_KEY` | OpenAI API key for agent | - |
| `ISLANDS_SYNC_INTERVAL` | Sync interval in seconds | `300` |
| `ISLANDS_MAX_CONCURRENT_SYNCS` | Max concurrent syncs | `4` |
| `ISLANDS_STORAGE__BASE_PATH` | Base storage path | `/data/islands` |
| `ISLANDS_PROVIDERS__0__TYPE` | First provider type | - |
| `ISLANDS_PROVIDERS__0__TOKEN` | First provider token | - |

### Config File

```yaml
debug: false
log_level: INFO
sync_interval: 300

storage:
  base_path: /data/islands
  repos_path: /data/islands/repos
  indexes_path: /data/islands/indexes

leann:
  backend: hnsw
  no_recompute: true

providers:
  - type: github
    token: ${GITHUB_TOKEN}
  - type: gitlab
    token: ${GITLAB_TOKEN}

agent:
  model: gpt-4o
  temperature: 0.1
```

## MCP Integration

Islands provides an MCP server for integration with Claude Code and other AI assistants.

### Available Tools

| Tool | Description |
|------|-------------|
| `islands_list` | List all indexed codebases |
| `islands_search` | Semantic search across codebases |
| `islands_add_repo` | Add and index a repository by URL |
| `islands_sync` | Sync and re-index a repository |
| `islands_status` | Get index status |

### Claude Code Integration

```bash
# Add Islands as an MCP server
claude mcp add islands-server -- islands-mcp
```

## Architecture

Islands is built entirely in Rust as a workspace of specialized crates:

```
islands/
├── crates/
│   ├── islands-core/          # LEANN implementation (arXiv:2506.08276)
│   │   ├── leann.rs           # LeannIndex, CsrGraph, EmbeddingProvider
│   │   ├── hnsw.rs            # Traditional HNSW (for comparison)
│   │   ├── pq.rs              # Product Quantization
│   │   ├── distance.rs        # Distance metrics (cosine, euclidean, dot)
│   │   └── search.rs          # Search algorithms
│   │
│   ├── islands-providers/     # Git provider implementations
│   │   ├── github.rs          # GitHub API
│   │   ├── gitlab.rs          # GitLab API
│   │   ├── bitbucket.rs       # Bitbucket API
│   │   └── gitea.rs           # Gitea API
│   │
│   ├── islands-indexer/       # Repository indexing service
│   ├── islands-mcp/           # MCP server for AI assistants
│   ├── islands-agent/         # OpenAI agent integration
│   └── islands-cli/           # Command-line interface
│
├── k8s/                       # Kubernetes manifests
├── docker/                    # Docker configuration
└── examples/                  # Usage examples
```

### Crates

| Crate | Description |
|-------|-------------|
| `islands-core` | LEANN vector indexing, HNSW, PQ, distance metrics |
| `islands-providers` | Git provider implementations (GitHub, GitLab, Bitbucket, Gitea) |
| `islands-indexer` | Code chunking, embedding generation, file watching |
| `islands-mcp` | MCP server for AI assistant integration |
| `islands-agent` | OpenAI agent for interactive Q&A |
| `islands-cli` | Command-line interface |

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/panbanda/islands.git
cd islands

# Install Rust toolchain
rustup update stable
```

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p islands-core

# Run with logging
RUST_LOG=debug cargo test

# Run doc tests
cargo test --doc
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets -- -D warnings

# Build documentation
cargo doc --no-deps
```

### Benchmarks

```bash
# Run benchmarks
cargo bench -p islands-core
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with EFS CSI driver
- EFS filesystem mounted as a StorageClass

### Deployment Steps

1. **Create namespace and secrets**:
   ```bash
   kubectl create namespace islands
   kubectl create secret generic islands-secrets \
     --from-literal=ISLANDS_OPENAI_API_KEY=$OPENAI_API_KEY \
     --from-literal=ISLANDS_PROVIDERS__0__TOKEN=$GITHUB_TOKEN \
     -n islands
   ```

2. **Configure EFS StorageClass**:
   ```yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: efs-sc
   provisioner: efs.csi.aws.com
   parameters:
     provisioningMode: efs-ap
     fileSystemId: fs-xxxxxxxxx
   ```

3. **Deploy Islands**:
   ```bash
   kubectl apply -k k8s/
   ```

4. **Configure webhooks** (optional):
   Set up webhooks in your git providers pointing to the Ingress endpoint.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- **LEANN Paper**: Wang et al., "LEANN: A Low-Storage Embedding-based Retrieval System for Large Scale Generative AI Applications" ([arXiv:2506.08276](https://arxiv.org/abs/2506.08276))
- **LEANN Implementation**: [github.com/yichuan-w/LEANN](https://github.com/yichuan-w/LEANN)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)
