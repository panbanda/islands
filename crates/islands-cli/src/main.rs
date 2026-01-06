//! Islands CLI - Main entry point

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use islands_cli::{Config, commands};

#[derive(Parser)]
#[command(name = "islands")]
#[command(version, about = "Islands - Codebase Indexing and Inquiry System")]
struct Cli {
    /// Enable debug logging
    #[arg(long, global = true)]
    debug: bool,

    /// Path to configuration file
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add and index a repository
    Add {
        /// Repository URL (e.g. github.com/owner/repo)
        url: String,

        /// Git provider token
        #[arg(long, env = "ISLANDS_GIT_TOKEN")]
        token: Option<String>,
    },

    /// Search across indexed codebases
    Search {
        /// Search query
        query: String,

        /// Specific indexes to search
        #[arg(short, long)]
        index: Vec<String>,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
    },

    /// List all indexed repositories
    List,

    /// Sync and re-index a repository
    Sync {
        /// Index name (provider/owner/repo)
        index_name: String,
    },

    /// Start the MCP server
    #[cfg(feature = "mcp")]
    Serve,

    /// Start interactive Q&A session
    #[cfg(feature = "agent")]
    Ask,

    /// Show system status
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // Setup logging
    let filter = if cli.debug {
        EnvFilter::from_default_env().add_directive("islands=debug".parse()?)
    } else {
        EnvFilter::from_default_env().add_directive("islands=info".parse()?)
    };

    fmt().with_env_filter(filter).with_target(false).init();

    // Load configuration
    let mut config = if let Some(path) = cli.config {
        Config::from_file(&path)?
    } else {
        Config::from_env()
    };

    config.debug = cli.debug;

    // Execute command
    match cli.command {
        Commands::Add { url, token } => {
            commands::add_repository(&config, &url, token.as_deref()).await?;
        }
        Commands::Search {
            query,
            index,
            top_k,
        } => {
            let indexes = if index.is_empty() { None } else { Some(index) };
            commands::search(&config, &query, indexes, top_k).await?;
        }
        Commands::List => {
            commands::list_indexes(&config).await?;
        }
        Commands::Sync { index_name } => {
            commands::sync_repository(&config, &index_name).await?;
        }
        #[cfg(feature = "mcp")]
        Commands::Serve => {
            commands::serve_mcp(&config).await?;
        }
        #[cfg(feature = "agent")]
        Commands::Ask => {
            commands::interactive_ask(&config).await?;
        }
        Commands::Status => {
            commands::show_status(&config).await?;
        }
    }

    Ok(())
}
