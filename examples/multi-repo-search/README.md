# Multi-Repository Codebase Search

Index multiple GitHub repositories and ask semantic questions across the entire codebase.

## Overview

This example demonstrates how to:
1. Configure multiple GitHub repositories for indexing
2. Generate embeddings for all code files
3. Perform semantic search across repositories
4. Use the agent to answer questions about the codebase

## Prerequisites

- Rust toolchain (1.75+)
- GitHub personal access token (for private repos)
- OpenAI API key (optional, for cloud embeddings)

## Configuration

Create a `config.toml`:

```toml
[indexer]
repos_path = "./data/repos"
indexes_path = "./data/indexes"
max_concurrent_syncs = 4
sync_interval_secs = 3600

# Use local embeddings (no API key needed)
[indexer.embedding]
provider = "local"
model = "bge-small"
batch_size = 32

# Or use OpenAI embeddings
# [indexer.embedding]
# provider = "openai"
# model = "text-embedding-3-small"

[repositories]
# Add repositories to index
repos = [
    { provider = "github", owner = "tokio-rs", name = "tokio" },
    { provider = "github", owner = "tokio-rs", name = "axum" },
    { provider = "github", owner = "tokio-rs", name = "tracing" },
]
```

## Usage

### 1. Build Pythia

```bash
cargo build --release --features embeddings
```

### 2. Index Repositories

```rust
use pythia_indexer::{IndexerConfig, IndexerService, EmbeddingConfig};
use pythia_providers::{GitHubProvider, Repository};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create indexer with local embeddings
    let config = IndexerConfig {
        repos_path: "./data/repos".into(),
        indexes_path: "./data/indexes".into(),
        embedding: EmbeddingConfig::Local {
            model: "bge-small".to_string(),
            batch_size: 32,
        },
        ..Default::default()
    };

    let mut service = IndexerService::new(config);

    // Initialize the embedding model
    service.init_embedder().await?;

    // Add repositories
    let repos = vec![
        Repository::new("github", "tokio-rs", "tokio"),
        Repository::new("github", "tokio-rs", "axum"),
        Repository::new("github", "tokio-rs", "tracing"),
    ];

    for repo in repos {
        println!("Indexing {}/{}...", repo.owner, repo.name);
        service.add_repository(repo).await?;
    }

    println!("Indexing complete!");
    Ok(())
}
```

### 3. Search Across Repositories

```rust
use pythia_indexer::IndexerService;

async fn search_codebase(
    service: &IndexerService,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Search across all indexed repositories
    let results = service.search(query, 10, None).await?;

    println!("Found {} results for: {}", results.len(), query);
    for result in results {
        println!("\n--- {}", result.file_path);
        println!("Score: {:.4}", result.score);
        println!("{}", result.content);
    }

    Ok(())
}

// Example queries
search_codebase(&service, "how does tokio handle async task spawning").await?;
search_codebase(&service, "axum router middleware chain").await?;
search_codebase(&service, "tracing span attributes and fields").await?;
```

### 4. Ask Questions with the Agent

```rust
use pythia_agent::{Agent, AgentConfig};

async fn ask_question(
    agent: &Agent,
    question: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = agent.ask(question).await?;
    println!("Q: {}\n", question);
    println!("A: {}\n", response);
    Ok(())
}

// Create agent with search capability
let agent = Agent::new(AgentConfig {
    indexer: service,
    llm_api_key: std::env::var("OPENAI_API_KEY")?,
    ..Default::default()
});

// Ask questions about the multi-repo codebase
ask_question(&agent, "How do I create an Axum router with middleware?").await?;
ask_question(&agent, "What's the relationship between tokio::spawn and tracing spans?").await?;
ask_question(&agent, "How does Axum integrate with the Tower ecosystem?").await?;
```

## Example Questions

Once indexed, you can ask questions like:

- "How do I set up connection pooling in Axum?"
- "What tracing macros are available for logging?"
- "How does tokio's task scheduler work?"
- "Show me examples of Axum extractors"
- "How do I propagate tracing context across async boundaries?"

## Performance Tips

1. **Use local embeddings** for development (no API costs)
2. **Increase batch_size** for faster indexing on GPU
3. **Filter file extensions** to index only relevant code
4. **Use incremental sync** to update only changed files

## File Structure

```
data/
  repos/
    github/
      tokio-rs/
        tokio/
        axum/
        tracing/
  indexes/
    github/
      tokio-rs/
        tokio.leann
        axum.leann
        tracing.leann
```
