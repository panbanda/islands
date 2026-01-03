# Pythia

[![CI](https://github.com/jon/pythia/actions/workflows/ci.yml/badge.svg)](https://github.com/jon/pythia/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/jon/pythia)](https://goreportcard.com/report/github.com/jon/pythia)
[![codecov](https://codecov.io/gh/jon/pythia/graph/badge.svg)](https://codecov.io/gh/jon/pythia)
[![Go Reference](https://pkg.go.dev/badge/github.com/jon/pythia.svg)](https://pkg.go.dev/github.com/jon/pythia)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/jon/pythia)](https://github.com/jon/pythia/releases)

AI-powered codebase analysis tool with semantic search and MCP integration for LLMs.

## Features

- **Semantic Code Search**: Natural language queries powered by [LEANN](https://github.com/yichuan-w/LEANN) vector database
- **MCP Integration**: Model Context Protocol server for Claude Code and other LLMs
- **Multi-Provider Git**: Support for GitHub, GitLab, Bitbucket, and custom providers
- **Local & Remote**: Index local directories or clone repositories
- **Container Ready**: Docker image and Helm chart for Kubernetes deployment
- **Extensible**: Plugin architecture with configurable providers
- **AI-Optimized**: Claude Code skills for enhanced LLM interaction

## Quick Start

### Installation

```bash
# Using Go
go install github.com/jon/pythia@latest

# Using Homebrew (macOS/Linux)
brew install jon/tap/pythia

# Using Docker
docker pull ghcr.io/jon/pythia:latest
```

### Basic Usage

```bash
# Index a codebase
pythia index ./my-project

# Search with natural language
pythia search "authentication middleware"

# Start MCP server for Claude Code
pythia serve
```

### Claude Code Integration

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "pythia": {
      "command": "pythia",
      "args": ["serve"]
    }
  }
}
```

## Commands

| Command | Description |
|---------|-------------|
| `pythia index <path>` | Index a codebase for semantic search |
| `pythia search <query>` | Search indexed codebases |
| `pythia serve` | Start MCP server |
| `pythia list` | List indexed codebases |
| `pythia version` | Show version information |

## Configuration

Create `~/.pythia.toml` or `./pythia.toml`:

```toml
log_level = "info"
data_dir = "~/.pythia"

[index]
max_file_size = 1048576
chunk_size = 512
concurrent_files = 4

[mcp]
transport = "stdio"
port = 8080

[[git.providers]]
name = "github"
type = "github"
token_env = "GITHUB_TOKEN"
```

See [pythia.example.toml](pythia.example.toml) for full configuration options.

## Architecture

```
pythia/
├── cmd/pythia/          # CLI commands
├── internal/
│   ├── analyzer/        # Semantic search and analysis
│   ├── config/          # Configuration management
│   ├── git/             # Multi-provider Git support
│   ├── indexer/         # Codebase indexing
│   ├── mcp/             # MCP server implementation
│   └── storage/         # Persistent storage
├── pkg/
│   ├── leann/           # LEANN client wrapper
│   └── types/           # Shared types
└── deployments/
    ├── docker/          # Docker configuration
    └── helm/            # Kubernetes Helm chart
```

## CI/CD & Quality

This project enforces strict code quality standards:

- **Cyclomatic Complexity**: Max 12 per function
- **Cognitive Complexity**: Max 15 per function
- **Test Coverage**: 90%+ required
- **Security Scanning**: gosec, Trivy, govulncheck
- **Linting**: golangci-lint with 50+ linters

### Quality Gates

| Check | Threshold |
|-------|-----------|
| Cyclomatic Complexity | <= 12 |
| Cognitive Complexity | <= 15 |
| Function Length | <= 80 lines |
| Test Coverage | >= 90% |

## Deployment

### Docker

```bash
docker run -v ~/.pythia:/data ghcr.io/jon/pythia:latest serve
```

### Kubernetes

```bash
helm repo add pythia https://jon.github.io/pythia
helm install pythia pythia/pythia
```

### Helm Values

```yaml
persistence:
  enabled: true
  size: 10Gi

config:
  mcp:
    transport: http
    port: 8080
```

## Development

### Prerequisites

- Go 1.23+
- golangci-lint
- Docker (for container builds)

### Building

```bash
# Build binary
go build -o pythia .

# Run tests
go test -race -cover ./...

# Run linter
golangci-lint run

# Build Docker image
docker build -f deployments/docker/Dockerfile -t pythia .
```

### Testing

```bash
# Unit tests with coverage
go test -v -race -coverprofile=coverage.out ./...

# View coverage report
go tool cover -html=coverage.out

# Integration tests
go test -v -tags=integration ./tests/integration/...
```

## MCP Tools

The MCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `search` | Semantic search across indexed codebases |
| `ask` | Question-answering about code |
| `list_indexes` | List all indexed codebases |
| `get_index` | Get index metadata |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI passes (lint, test, coverage)
5. Submit a pull request

## License

Apache 2.0 - see [LICENSE](LICENSE)

## Acknowledgments

- [LEANN](https://github.com/yichuan-w/LEANN) - Low-storage vector database
- [MCP Go SDK](https://github.com/mark3labs/mcp-go) - Model Context Protocol implementation
- [Cobra](https://github.com/spf13/cobra) - CLI framework
- [Viper](https://github.com/spf13/viper) - Configuration management

## Related Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Code](https://claude.ai/claude-code)
- [LEANN Paper](https://arxiv.org/abs/2412.xxxxx)
