//! Multi-repository codebase search example
//!
//! This example demonstrates indexing multiple GitHub repositories
//! and performing semantic search across all of them.

use std::path::PathBuf;

use islands_indexer::{IndexerConfig, IndexerService};
use islands_providers::Repository;

#[cfg(feature = "embeddings")]
use islands_indexer::EmbeddingConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logs
    tracing_subscriber::fmt::init();

    // Configure the indexer
    let config = IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes"),
        max_concurrent_syncs: 4,
        sync_interval_secs: 3600,
        index_extensions: vec![
            "rs", "py", "js", "ts", "go", "java", "md",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        #[cfg(feature = "embeddings")]
        embedding: EmbeddingConfig::Local {
            model: "bge-small".to_string(),
            batch_size: 32,
        },
    };

    println!("Creating indexer service...");
    let mut service = IndexerService::new(config);

    // Initialize embeddings (downloads model on first run)
    #[cfg(feature = "embeddings")]
    {
        println!("Initializing embedding model (this may download the model)...");
        service.init_embedder().await?;
    }

    // Define repositories to index (using URL shorthand)
    let repos = vec![
        Repository::from_url("tokio-rs/tokio").expect("valid URL"),
        Repository::from_url("tokio-rs/axum").expect("valid URL"),
        Repository::from_url("tokio-rs/tracing").expect("valid URL"),
    ];

    // Index each repository
    for repo in &repos {
        println!("\nIndexing {}/{}...", repo.owner, repo.name);
        match service.add_repository(repo.clone()).await {
            Ok(info) => {
                println!(
                    "  Indexed {} files ({} bytes)",
                    info.file_count, info.size_bytes
                );
            }
            Err(e) => {
                eprintln!("  Failed to index: {}", e);
            }
        }
    }

    // Perform semantic searches
    let queries = [
        "async task spawning and scheduling",
        "HTTP router with middleware",
        "structured logging with spans",
        "error handling patterns",
    ];

    println!("\n--- Semantic Search Results ---\n");

    for query in queries {
        println!("Query: {}", query);
        println!("{}", "-".repeat(50));

        let results = service.search(query, 3, None).await?;

        if results.is_empty() {
            println!("  No results found\n");
            continue;
        }

        for (i, result) in results.iter().enumerate() {
            println!(
                "  {}. {} (score: {:.4})",
                i + 1,
                result.file_path,
                result.score
            );
            // Show first 100 chars of content
            let preview: String = result.content.chars().take(100).collect();
            println!("     {}", preview.replace('\n', " "));
        }
        println!();
    }

    Ok(())
}
