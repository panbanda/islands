//! CLI command implementations

use std::collections::HashMap;
use std::sync::Arc;

use crate::indexer::service::IndexerService;
use crate::providers::factory::{create_provider, parse_repo_url};

use crate::Result;
use crate::config::Config;
use crate::output;

/// Add a repository to the index
pub async fn add_repository(config: &Config, url: &str, token: Option<&str>) -> Result<()> {
    let spinner = output::spinner(&format!("Adding repository: {}", url));

    // Parse URL
    let (provider_type, owner, name, base_url) = parse_repo_url(url)?;

    // Create provider
    let provider_type_str = format!("{:?}", provider_type).to_lowercase();
    let provider = create_provider(&provider_type_str, base_url.as_deref(), token, None)?;

    // Create indexer
    let mut providers = HashMap::new();
    providers.insert(provider.provider_name().to_string(), provider.clone());

    let indexer = IndexerService::new(config.indexer.clone(), providers);

    // Get repository info
    let repo = provider.get_repository(&owner, &name).await?;

    spinner.set_message(format!("Cloning {}...", repo.full_name));

    // Add and index
    let state = indexer.add_repository(&repo).await?;

    spinner.finish_and_clear();

    output::success(&format!("Successfully indexed {}", repo.full_name));
    println!("  Commit: {}", state.last_commit.unwrap_or_default());
    println!("  Path: {:?}", state.local_path);

    Ok(())
}

/// Search across indexed repositories
pub async fn search(
    config: &Config,
    query: &str,
    indexes: Option<Vec<String>>,
    top_k: usize,
) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let results = indexer.search(query, indexes.as_deref(), top_k).await?;

    if results.is_empty() {
        output::warning(&format!("No results found for: {}", query));
        return Ok(());
    }

    println!("\nSearch Results: {}\n", query);

    for (i, result) in results.iter().enumerate() {
        println!(
            "{} {} - {}",
            console::style(format!("{}.", i + 1)).cyan(),
            result
                .get("repository")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
            result
                .get("file")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
        );

        if let Some(score) = result.get("score").and_then(|v| v.as_f64()) {
            println!("   Score: {:.4}", score);
        }

        if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
            let preview: String = content.chars().take(100).collect();
            println!("   {}", console::style(preview).dim());
        }

        println!();
    }

    Ok(())
}

/// List all indexed repositories
pub async fn list_indexes(config: &Config) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let indexes = indexer.list_indexes().await;

    if indexes.is_empty() {
        output::warning("No indexes found. Use 'islands add' to add repositories.");
        return Ok(());
    }

    println!("\nIslands Indexes\n");

    for info in indexes {
        println!(
            "{} ({})",
            console::style(&info.name).cyan().bold(),
            info.repository.provider
        );
        println!(
            "  Files: {}, Size: {:.2} MB",
            info.file_count,
            info.size_bytes as f64 / (1024.0 * 1024.0)
        );
        println!("  Updated: {}", info.updated_at);
        println!();
    }

    Ok(())
}

/// Sync a repository
pub async fn sync_repository(config: &Config, index_name: &str) -> Result<()> {
    let spinner = output::spinner(&format!("Syncing {}", index_name));

    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    let info = indexer
        .get_index(index_name)
        .await
        .ok_or_else(|| crate::Error::InvalidArgument(format!("Index not found: {}", index_name)))?;

    let state = indexer.sync_repository(&info.repository).await?;

    spinner.finish_and_clear();

    output::success(&format!("Synced {}", index_name));
    println!("  Commit: {}", state.last_commit.unwrap_or_default());
    println!("  Indexed: {}", state.indexed);

    Ok(())
}

/// Start the MCP server
#[cfg(feature = "mcp")]
pub async fn serve_mcp(config: &Config) -> Result<()> {
    use crate::mcp::server::run_server;

    output::info("Starting Islands MCP server...");

    let indexer = Arc::new(IndexerService::new(config.indexer.clone(), HashMap::new()));

    run_server(indexer)
        .await
        .map_err(|e| crate::Error::Config(e.to_string()))?;

    Ok(())
}

/// Start interactive Q&A session
#[cfg(all(feature = "agent", feature = "openai"))]
pub async fn interactive_ask(config: &Config) -> Result<()> {
    use crate::agent::{IslandsAgent, llm::openai::OpenAiProvider};

    let api_key = config
        .openai_api_key
        .as_ref()
        .ok_or_else(|| crate::Error::Config("OPENAI_API_KEY not set".into()))?;

    let indexer = Arc::new(IndexerService::new(config.indexer.clone(), HashMap::new()));
    let llm = Arc::new(OpenAiProvider::new(api_key));

    let mut agent = IslandsAgent::new(indexer, llm, None);

    output::info("Islands Codebase Assistant");
    println!("Ask questions about your indexed codebases. Type 'quit' to exit.\n");

    let indexes = agent.list_indexes().await;
    if indexes.is_empty() {
        output::warning("No indexes available. Add repositories first.");
    } else {
        println!("Available indexes: {}\n", indexes.join(", "));
    }

    loop {
        print!("\n{} ", console::style("You:").cyan().bold());
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" || input == "q" {
            break;
        }

        if input == "clear" {
            agent.clear_conversation();
            output::info("Conversation cleared.");
            continue;
        }

        println!("\n{}", console::style("Islands:").magenta().bold());

        match agent.ask(input, true).await {
            Ok(response) => {
                println!("{}", response);
            }
            Err(e) => {
                output::error(&format!("Error: {}", e));
            }
        }
    }

    println!("\nGoodbye!");
    Ok(())
}

/// Show system status
pub async fn show_status(config: &Config) -> Result<()> {
    let indexer = IndexerService::new(config.indexer.clone(), HashMap::new());

    println!("\n{}\n", console::style("Islands Status").bold());

    let indexes = indexer.list_indexes().await;
    let total_files: usize = indexes.iter().map(|i| i.file_count).sum();
    let total_size: u64 = indexes.iter().map(|i| i.size_bytes).sum();

    println!("Total Indexes: {}", indexes.len());
    println!("Total Files: {}", total_files);
    println!(
        "Total Size: {:.2} MB",
        total_size as f64 / (1024.0 * 1024.0)
    );
    println!("Repos Path: {:?}", config.indexer.repos_path);
    println!("Indexes Path: {:?}", config.indexer.indexes_path);

    Ok(())
}
