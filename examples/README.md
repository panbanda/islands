# Pythia Examples

This directory contains examples demonstrating different ways to use Pythia for semantic code search.

## Examples

| Example | Description |
|---------|-------------|
| [multi-repo-search](./multi-repo-search/) | Index multiple GitHub repositories and search across them |
| [local-indexing](./local-indexing/) | Index local directories and projects |
| [embedding-providers](./embedding-providers/) | Configure different embedding backends (local, OpenAI, Cohere) |
| [mcp-integration](./mcp-integration/) | Use Pythia as an MCP server for AI assistants |

## Prerequisites

1. **Rust toolchain** (1.75 or later)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Build Pythia with embeddings**
   ```bash
   cargo build --release --features embeddings
   ```

3. **Optional: GPU acceleration**
   ```bash
   # For NVIDIA GPUs
   cargo build --release --features "embeddings cuda"

   # For Apple Silicon
   cargo build --release --features "embeddings metal"
   ```

## Running Examples

```bash
# Build all examples
cargo build --examples --features embeddings

# Run multi-repo search
cargo run --example multi-repo-search --features embeddings

# Run local indexing (pass a path)
cargo run --example local-indexing --features embeddings -- /path/to/project

# Test different embedding providers
cargo run --example embedding-providers --features embeddings -- local
cargo run --example embedding-providers --features embeddings -- openai
cargo run --example embedding-providers --features embeddings -- cohere

# Start MCP server
cargo run --example mcp-integration --features embeddings
```

## Common Configuration

All examples use similar configuration. Create a `config.toml`:

```toml
[indexer]
repos_path = "./data/repos"
indexes_path = "./data/indexes"
max_concurrent_syncs = 4
sync_interval_secs = 3600

# File extensions to index
index_extensions = [
    "rs", "py", "js", "ts", "jsx", "tsx",
    "go", "java", "c", "cpp", "h", "hpp",
    "md", "txt", "yaml", "toml", "json"
]

# Embedding configuration
[indexer.embedding]
provider = "local"
model = "bge-small"
batch_size = 32
```

## Environment Variables

```bash
# For OpenAI embeddings
export OPENAI_API_KEY="sk-..."

# For Cohere embeddings
export COHERE_API_KEY="..."

# For GitHub private repositories
export GITHUB_TOKEN="ghp_..."
```

## Quick Start

1. **Index a repository**
   ```rust
   let mut service = IndexerService::new(config);
   service.init_embedder().await?;
   service.add_repository(repo).await?;
   ```

2. **Search semantically**
   ```rust
   let results = service.search("how does error handling work", 10, None).await?;
   ```

3. **Get file contents**
   ```rust
   let content = service.get_file("owner/repo", "src/main.rs").await?;
   ```
