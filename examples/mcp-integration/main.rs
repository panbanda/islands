//! MCP Server integration example
//!
//! Run Pythia as an MCP server for AI assistant integration.

use std::path::PathBuf;

use pythia_indexer::{IndexerConfig, IndexerService};
use pythia_mcp::{McpConfig, McpServer};

#[cfg(feature = "embeddings")]
use pythia_indexer::EmbeddingConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging (to stderr to not interfere with MCP protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    // Create indexer configuration
    let indexer_config = IndexerConfig {
        repos_path: PathBuf::from("./data/repos"),
        indexes_path: PathBuf::from("./data/indexes"),
        #[cfg(feature = "embeddings")]
        embedding: EmbeddingConfig::Local {
            model: "bge-small".to_string(),
            batch_size: 32,
        },
        ..Default::default()
    };

    // Initialize the indexer
    let mut indexer = IndexerService::new(indexer_config);

    #[cfg(feature = "embeddings")]
    {
        tracing::info!("Initializing embedding model...");
        indexer.init_embedder().await?;
    }

    // Create MCP server configuration
    let mcp_config = McpConfig {
        name: "pythia".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    // Create and run the MCP server
    let server = McpServer::new(mcp_config, indexer);

    tracing::info!("Starting MCP server on stdio...");
    server.run_stdio().await?;

    Ok(())
}
