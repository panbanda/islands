# Pythia

[![CI](https://github.com/panbanda/pythia/actions/workflows/ci.yaml/badge.svg)](https://github.com/panbanda/pythia/actions/workflows/ci.yaml)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pythia** is a codebase indexing and semantic search system built on the LEANN algorithm. It provides an MCP (Model Context Protocol) interface for AI assistants and supports multi-provider git integration.

## LEANN: Low-Storage Vector Search

Pythia implements the [LEANN algorithm](https://arxiv.org/abs/2506.08276) (arXiv:2506.08276) for efficient vector similarity search:

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
# Install with pip
pip install pythia

# Or install from source
git clone https://github.com/panbanda/pythia.git
cd pythia
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Add a repository by URL
pythia add https://github.com/owner/repo --token $GITHUB_TOKEN

# Supports various URL formats
pythia add https://gitlab.com/owner/repo
pythia add git@github.com:owner/repo.git

# Search across indexed codebases
pythia search "authentication middleware"

# List all indexes
pythia list

# Start the MCP server
pythia serve

# Interactive Q&A session
pythia ask
```

### Docker

```bash
# Build the image
docker build -t pythia -f docker/Dockerfile .

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

Pythia is configured through environment variables or a config file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHIA_DEBUG` | Enable debug mode | `false` |
| `PYTHIA_LOG_LEVEL` | Log level | `INFO` |
| `PYTHIA_OPENAI_API_KEY` | OpenAI API key for agent | - |
| `PYTHIA_SYNC_INTERVAL` | Sync interval in seconds | `300` |
| `PYTHIA_MAX_CONCURRENT_SYNCS` | Max concurrent syncs | `4` |
| `PYTHIA_STORAGE__BASE_PATH` | Base storage path | `/data/pythia` |
| `PYTHIA_PROVIDERS__0__TYPE` | First provider type | - |
| `PYTHIA_PROVIDERS__0__TOKEN` | First provider token | - |

### Config File

```yaml
debug: false
log_level: INFO
sync_interval: 300

storage:
  base_path: /data/pythia
  repos_path: /data/pythia/repos
  indexes_path: /data/pythia/indexes

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

Pythia provides an MCP server for integration with Claude Code and other AI assistants.

### Available Tools

| Tool | Description |
|------|-------------|
| `pythia_list` | List all indexed codebases |
| `pythia_search` | Semantic search across codebases |
| `pythia_add_repo` | Add and index a repository by URL |
| `pythia_sync` | Sync and re-index a repository |
| `pythia_status` | Get index status |

### Claude Code Integration

```bash
# Add Pythia as an MCP server
claude mcp add pythia-server -- pythia-mcp
```

## Architecture

Pythia uses a hybrid Rust/Python architecture:

```
pythia/
├── crates/                    # Rust core libraries
│   ├── pythia-core/           # LEANN implementation (arXiv:2506.08276)
│   │   ├── src/
│   │   │   ├── leann.rs       # LeannIndex, CsrGraph, EmbeddingProvider
│   │   │   ├── hnsw.rs        # Traditional HNSW (for comparison)
│   │   │   ├── pq.rs          # Product Quantization
│   │   │   ├── distance.rs    # Distance metrics (cosine, euclidean, dot)
│   │   │   └── search.rs      # Search algorithms
│   ├── pythia-indexer/        # Repository indexing
│   ├── pythia-providers/      # Git provider implementations
│   ├── pythia-mcp/            # MCP server
│   ├── pythia-agent/          # AI agent integration
│   └── pythia-cli/            # Command-line interface
│
├── src/pythia/                # Python package (uses Rust via PyO3)
│   ├── providers/             # Git provider implementations
│   │   ├── github.py          # GitHub provider
│   │   ├── gitlab.py          # GitLab provider
│   │   ├── bitbucket.py       # Bitbucket provider
│   │   └── gitea.py           # Gitea provider
│   ├── indexer/               # Repository indexing
│   ├── mcp/                   # MCP server
│   ├── agent/                 # OpenAI agent
│   └── storage/               # EFS storage
│
├── k8s/                       # Kubernetes manifests
└── docker/                    # Docker configuration
```

### Core Components

| Component | Language | Description |
|-----------|----------|-------------|
| `pythia-core` | Rust | LEANN vector indexing, HNSW, PQ, distance metrics |
| `pythia-indexer` | Rust | Code chunking, embedding generation |
| `pythia-mcp` | Rust | MCP server for AI assistant integration |
| `pythia-cli` | Rust | Command-line interface |
| `src/pythia` | Python | High-level API, git providers, agent |

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/panbanda/pythia.git
cd pythia

# Rust setup (required for pythia-core)
rustup update stable

# Python setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Rust tests (pythia-core)
cargo test -p pythia-core

# Python tests
pytest

# With coverage
pytest --cov=pythia --cov-report=html
cargo llvm-cov --html
```

### Code Quality

```bash
# Rust
cargo fmt --check
cargo clippy

# Python
ruff check src tests
ruff format src tests
mypy src
```

### Benchmarks

```bash
# Run Rust benchmarks
cargo bench -p pythia-core
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with EFS CSI driver
- EFS filesystem mounted as a StorageClass

### Deployment Steps

1. **Create namespace and secrets**:
   ```bash
   kubectl create namespace pythia
   kubectl create secret generic pythia-secrets \
     --from-literal=PYTHIA_OPENAI_API_KEY=$OPENAI_API_KEY \
     --from-literal=PYTHIA_PROVIDERS__0__TOKEN=$GITHUB_TOKEN \
     -n pythia
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

3. **Deploy Pythia**:
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
