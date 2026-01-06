//! Embedding provider configuration example
//!
//! Demonstrates how to configure different embedding providers.

use std::path::PathBuf;

use islands_indexer::{IndexerConfig, IndexerService};
use islands_providers::Repository;

#[cfg(feature = "embeddings")]
use islands_indexer::EmbeddingConfig;

/// Create an indexer with local embeddings
#[cfg(feature = "embeddings")]
fn create_local_indexer() -> IndexerConfig {
    IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes/local"),
        embedding: EmbeddingConfig::Local {
            model: "bge-small".to_string(),
            batch_size: 32,
        },
        ..Default::default()
    }
}

/// Create an indexer with OpenAI embeddings
#[cfg(feature = "embeddings")]
fn create_openai_indexer() -> IndexerConfig {
    IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes/openai"),
        embedding: EmbeddingConfig::OpenAI {
            model: "text-embedding-3-small".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            batch_size: 100,
        },
        ..Default::default()
    }
}

/// Create an indexer with Cohere embeddings
#[cfg(feature = "embeddings")]
fn create_cohere_indexer() -> IndexerConfig {
    IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes/cohere"),
        embedding: EmbeddingConfig::Cohere {
            model: "embed-english-v3.0".to_string(),
            api_key: std::env::var("COHERE_API_KEY").ok(),
            batch_size: 96,
        },
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Parse provider from command line
    let provider = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "local".to_string());

    #[cfg(feature = "embeddings")]
    let config = match provider.as_str() {
        "local" => {
            println!("Using LOCAL embeddings (bge-small)");
            create_local_indexer()
        }
        "openai" => {
            println!("Using OPENAI embeddings (text-embedding-3-small)");
            if std::env::var("OPENAI_API_KEY").is_err() {
                eprintln!("Warning: OPENAI_API_KEY not set");
            }
            create_openai_indexer()
        }
        "cohere" => {
            println!("Using COHERE embeddings (embed-english-v3.0)");
            if std::env::var("COHERE_API_KEY").is_err() {
                eprintln!("Warning: COHERE_API_KEY not set");
            }
            create_cohere_indexer()
        }
        _ => {
            eprintln!("Unknown provider: {}. Use: local, openai, or cohere", provider);
            std::process::exit(1);
        }
    };

    #[cfg(not(feature = "embeddings"))]
    let config = {
        eprintln!("Embeddings feature not enabled. Build with: --features embeddings");
        std::process::exit(1);
    };

    let mut service = IndexerService::new(config);

    #[cfg(feature = "embeddings")]
    {
        println!("Initializing embedding model...");
        service.init_embedder().await?;

        if let Some(embedder) = service.embedder() {
            println!("Embedding dimension: {}", embedder.dimension());
        }
    }

    // Index a sample repository
    let repo = Repository::from_url("rust-lang/rust-by-example").expect("valid URL");

    println!("\nIndexing {}...", repo.name);
    let info = service.add_repository(repo).await?;

    println!(
        "Indexed {} files ({:.2} KB)",
        info.file_count,
        info.size_bytes as f64 / 1024.0
    );

    // Test search
    let query = "how to create a struct with methods";
    println!("\nSearching: \"{}\"", query);

    let results = service.search(query, 3, None).await?;

    for (i, result) in results.iter().enumerate() {
        println!(
            "\n{}. {} (score: {:.4})",
            i + 1,
            result.file_path,
            result.score
        );
    }

    Ok(())
}
