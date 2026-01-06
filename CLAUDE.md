# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Build
cargo build --release

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p islands-core

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
cargo bench -p islands-core
```

## Architecture

Islands is a Rust workspace implementing LEANN (arXiv:2506.08276) for low-storage vector search. The key insight is storing only the proximity graph structure (CSR format) and recomputing embeddings on-demand during search, achieving ~95% storage reduction.

### Crate Dependencies

```
islands-cli (binary)
    ├── islands-mcp (MCP server)
    │   ├── islands-indexer
    │   │   ├── islands-core (LEANN/HNSW/PQ)
    │   │   └── islands-providers (GitHub/GitLab/Bitbucket/Gitea)
    │   └── islands-core
    ├── islands-agent (OpenAI integration)
    │   └── islands-core
    └── islands-indexer
```

### Core Algorithms (islands-core)

- **leann.rs**: `LeannIndex` with `CsrGraph` for graph-only storage, `EmbeddingProvider` trait for on-demand recomputation
- **hnsw.rs**: Traditional HNSW implementation for comparison
- **pq.rs**: Product Quantization for vector compression
- **distance.rs**: Cosine, Euclidean, dot product metrics
- **search.rs**: Two-level search with approximate/exact distance queues

### LEANN Parameters (from paper Section 5)

- $M = 30$ connections per node
- $\text{efConstruction} = 128$
- High-degree preserving pruning: hub nodes (top 2%) retain more connections

### Provider Pattern (islands-providers)

All git providers implement the same async trait pattern in `base.rs`. Add new providers by implementing the trait and registering in `factory.rs`.

## Conventional Commits

This project uses release-please. Use conventional commit prefixes:
- `feat:` - new feature (minor version bump)
- `fix:` - bug fix (patch version bump)
- `docs:` - documentation only
- `chore:` - maintenance tasks
