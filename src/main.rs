//! Islands CLI - Main entry point

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::{EnvFilter, fmt};

use islands::{Config, commands};

/// Output format for commands
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// Machine-readable JSON output
    Json,
}

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

    /// Output format (text or json)
    #[arg(long, global = true, default_value = "text")]
    format: OutputFormat,

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

    /// Remove an indexed repository
    Remove {
        /// Index name (provider/owner/repo)
        index_name: String,

        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
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

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Start the MCP server (stdio transport)
    #[cfg(feature = "mcp")]
    Mcp,

    /// Start interactive Q&A session
    #[cfg(all(feature = "agent", feature = "openai"))]
    Ask,

    /// Show system status
    Status,
}

/// Configuration subcommands
#[derive(Subcommand)]
enum ConfigAction {
    /// Show current effective configuration
    Show,

    /// Initialize a default configuration file
    Init {
        /// Output path for the configuration file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
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
        Commands::Remove { index_name, force } => {
            commands::remove_index(&config, &index_name, force).await?;
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
        Commands::Config { action } => match action {
            ConfigAction::Show => {
                commands::config_show(&config).await?;
            }
            ConfigAction::Init { output } => {
                commands::config_init(output).await?;
            }
        },
        #[cfg(feature = "mcp")]
        Commands::Mcp => {
            commands::serve_mcp(&config).await?;
        }
        #[cfg(all(feature = "agent", feature = "openai"))]
        Commands::Ask => {
            commands::interactive_ask(&config).await?;
        }
        Commands::Status => {
            commands::show_status(&config).await?;
        }
    }

    Ok(())
}
