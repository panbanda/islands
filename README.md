# Pythia

[![CI](https://github.com/panbanda/pythia/actions/workflows/ci.yaml/badge.svg)](https://github.com/panbanda/pythia/actions/workflows/ci.yaml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Pythia** is a codebase indexing and inquiry system that uses [LEANN](https://github.com/yichuan-w/LEANN) for semantic search with an MCP (Model Context Protocol) interface. Designed for Kubernetes environments with EFS storage, it enables AI agents to understand and answer questions about your codebases.

## Features

- **Semantic Code Search**: Uses LEANN's graph-based vector indexing for 97% storage savings compared to traditional vector databases
- **Multi-Provider Support**: Works with GitHub, GitLab, Bitbucket, and Gitea
- **MCP Interface**: Integrates with Claude Code and other MCP-compatible AI assistants
- **OpenAI Agent**: Interactive Q&A agent using OpenAI's SDK
- **Kubernetes Native**: Designed for EFS storage with automatic sync and webhook support
- **Extensible Architecture**: Easy to add new git providers and indexing backends

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

```
pythia/
├── providers/       # Git provider implementations
│   ├── base.py      # Abstract base class
│   ├── github.py    # GitHub provider
│   ├── gitlab.py    # GitLab provider
│   ├── bitbucket.py # Bitbucket provider
│   └── gitea.py     # Gitea provider
├── indexer/         # Repository indexing
│   ├── repository.py # Repository management
│   └── service.py    # Indexer service
├── mcp/             # MCP server
│   └── server.py    # MCP tool handlers
├── agent/           # OpenAI agent
│   └── assistant.py # Q&A assistant
├── storage/         # EFS storage
│   ├── efs.py       # Metadata management
│   └── watcher.py   # File change detection
└── config/          # Configuration
    └── settings.py  # Pydantic settings
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/panbanda/pythia.git
cd pythia

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pythia --cov-report=html

# Run specific tests
pytest tests/test_providers.py -v
```

### Code Quality

```bash
# Run linter
ruff check src tests

# Run formatter
ruff format src tests

# Run type checker
mypy src
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

## Acknowledgments

- [LEANN](https://github.com/yichuan-w/LEANN) - The lightweight vector database powering semantic search
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol specification
