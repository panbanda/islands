# Local Repository Indexing

Index local directories and search code without cloning from remote.

## Overview

This example shows how to:
1. Index a local directory or git repository
2. Watch for file changes and update the index
3. Search through local code semantically

## Usage

### Basic Local Indexing

```rust
use std::path::PathBuf;
use islands_indexer::{IndexerConfig, IndexerService};
use islands_providers::Repository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = IndexerConfig {
        repos_path: PathBuf::from("./indexed"),
        indexes_path: PathBuf::from("./indexes"),
        ..Default::default()
    };

    let mut service = IndexerService::new(config);

    #[cfg(feature = "embeddings")]
    service.init_embedder().await?;

    // Index a local project
    let local_repo = Repository {
        provider: "local".to_string(),
        owner: "my-projects".to_string(),
        name: "my-app".to_string(),
        clone_url: Some("/path/to/my-app".to_string()),
        default_branch: Some("main".to_string()),
        ..Default::default()
    };

    service.add_repository(local_repo).await?;

    // Search the local codebase
    let results = service.search("database connection pooling", 5, None).await?;

    for result in results {
        println!("{}: {:.4}", result.file_path, result.score);
    }

    Ok(())
}
```

### With File Watching

Enable the `watch` feature to automatically re-index on file changes:

```rust
use islands_indexer::watcher::FileWatcher;

// Create a watcher for the indexed directory
let watcher = FileWatcher::new(&service)?;
watcher.watch("/path/to/my-app")?;

// The index will update automatically when files change
```

## Configuration Options

```toml
[indexer]
repos_path = "./indexed"
indexes_path = "./indexes"

# Files to index
index_extensions = [
    "rs", "py", "js", "ts", "jsx", "tsx",
    "go", "java", "c", "cpp", "h",
    "md", "txt", "yaml", "toml", "json"
]

# Embedding settings
[indexer.embedding]
provider = "local"
model = "bge-small"
batch_size = 64
```

## Ignoring Files

Islands respects `.gitignore` patterns. You can also create a `.islandsignore` file:

```
# Ignore build artifacts
target/
dist/
node_modules/

# Ignore generated files
*.generated.rs
*.min.js

# Ignore large data files
*.csv
*.parquet
```

## CLI Usage

```bash
# Index a local directory
islands index /path/to/project

# Search with natural language
islands search "how does authentication work"

# Watch for changes
islands watch /path/to/project
```
